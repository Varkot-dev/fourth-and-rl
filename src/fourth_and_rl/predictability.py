"""M10: how much does being predictable cost an offense?

The pieces used by `scripts/m10_predictability.py`:

- pre-snap features of every called run and pass: the situation, the offense's own
  tendencies from EARLIER games only (`team_tendencies`), and the calls and results
  earlier in the same game and drive (`add_history`);
- call models that give P(pass) before the snap: gradient boosting on the situation, on
  the situation plus hand-made history features, and a small GRU over the game's play
  sequence (`SeqModel`, PyTorch, imported only when used);
- scoring helpers: log loss, Brier score, calibration, paired differences with a
  game-level bootstrap;
- the cost of predictability: EPA as a function of P(the call actually made) with the
  situation, the offense-season and the defense-season absorbed as fixed effects
  (`fe_fit`, weighted alternating projections, rerun on every bootstrap resample);
- team estimates with empirical-Bayes shrinkage (`shrink`).

EPA is nflfastR's model, used as given and quoted as theirs. Play classification, the
neutral filter and the situation cells are question 2's (`run_pass`), unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from fourth_and_rl import run_pass as rp
from fourth_and_rl.paths import CONFIGS

EPS = 1e-6


def load_config(path: Path | None = None) -> dict:
    """Read configs/predictability.yaml."""
    with open(path or CONFIGS / "predictability.yaml") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------


def called_plays(plays: pl.DataFrame, extra: pl.DataFrame | None, rp_cfg: dict) -> pl.DataFrame:
    """Called runs and passes (question 2's classification) with the base columns used
    here, sorted by game, offense and snap order. `row` is the position in that order."""
    df = rp.prepare(rp.with_extra(plays, extra) if extra is not None else plays, rp_cfg)
    for name in ("roof", "season_type", "is_home", "wind", "temp", "total_line",
                 "posteam_spread", "goal_to_go", "week"):  # fmt: skip
        if name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    df = df.with_columns(
        pl.col("roof").cast(pl.String).is_in(["dome", "closed"]).cast(pl.Float64).alias("dome"),
        (pl.col("season_type").cast(pl.String) == "POST").cast(pl.Float64).alias("postseason"),
        pl.col("is_home").cast(pl.Float64),
    )
    order = [c for c in ("game_id", "posteam", "order_sequence", "play_id") if c in df.columns]
    return df.sort(order).with_row_index("row")


def add_history(df: pl.DataFrame) -> pl.DataFrame:
    """Hand-made features of the calls and results earlier in the same game and drive.

    Everything is shifted by one play inside (game, offense) or (game, offense, drive),
    so a play's own call and result never enter its features. `df` must be sorted by
    game, offense and snap order (as `called_plays` returns it).
    """
    g = ["game_id", "posteam"]
    d = [*g, "fixed_drive"]
    p = pl.col("is_pass").cast(pl.Float64)
    succ = pl.col("success").cast(pl.Float64)
    is_run = 1.0 - p
    n_prev = pl.int_range(pl.len()).cast(pl.Float64)

    def share(num: pl.Expr, den: pl.Expr) -> pl.Expr:
        return pl.when(den > 0).then(num / den).otherwise(None)

    pass_epa = (pl.col("epa") * p).cum_sum().shift(1).over(g)
    run_epa = (pl.col("epa") * is_run).cum_sum().shift(1).over(g)
    n_pass = p.cum_sum().shift(1).over(g)
    n_run = is_run.cum_sum().shift(1).over(g)
    out = df.with_columns(
        n_prev.over(g).alias("g_n_prev"),
        share(n_pass, n_prev.over(g)).alias("g_pass_share_prev"),
        p.shift(1).over(g).alias("g_prev_pass"),
        p.shift(2).over(g).alias("g_prev2_pass"),
        pl.col("yards").shift(1).over(g).alias("g_prev_yards"),
        succ.shift(1).over(g).alias("g_prev_success"),
        share(pass_epa, n_pass).alias("g_pass_epa_prev"),
        share(run_epa, n_run).alias("g_run_epa_prev"),
        pl.col("fixed_drive").rank("dense").over(g).cast(pl.Float64).alias("g_drive_index"),
        n_prev.over(d).alias("d_n_prev"),
        p.shift(1).over(d).alias("d_prev_pass"),
        pl.col("yards").shift(1).over(d).alias("d_prev_yards"),
        pl.col("incomplete_pass").cast(pl.Float64).fill_null(0.0).shift(1).over(d)
        .alias("d_prev_incomplete"),
        pl.col("first_down").cast(pl.Float64).shift(1).over(d).alias("d_prev_first_down"),
        share(p.cum_sum().shift(1).over(d), n_prev.over(d)).alias("d_pass_share_prev"),
    )  # fmt: skip
    return out


def team_game_aggregates(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """One row per offense and game: the counts and EPA sums that tendencies are built
    from, and the date order. Also the same game seen from the defense."""
    del cfg  # same signature as the other builders; nothing configurable here
    ne = pl.col("neutral") & pl.col("early")
    p = pl.col("is_pass") == 1
    return df.group_by("season", "week", "game_id", "posteam", "defteam").agg(
        pl.len().cast(pl.Float64).alias("n"),
        p.sum().cast(pl.Float64).alias("n_pass"),
        ne.sum().cast(pl.Float64).alias("n_ne"),
        (ne & p).sum().cast(pl.Float64).alias("n_pass_ne"),
        pl.col("epa").filter(p).sum().alias("s_pass_epa"),
        pl.col("epa").filter(~p).sum().alias("s_run_epa"),
        pl.col("epa").sum().alias("s_epa"),
    ).sort("season", "week", "game_id")


def decayed_prior_sums(key: np.ndarray, season: np.ndarray, x: np.ndarray, decay: float,
                       carry: float) -> np.ndarray:  # fmt: skip
    """For rows in date order, the decayed sum of `x` over EARLIER rows with the same key.

    out[i] = sum over earlier rows j of the same key of x[j] * decay^(games between)
    * carry^(season boundaries crossed). Row i's own x never enters out[i].
    """
    out = np.zeros_like(x, dtype=float)
    state: dict[Any, tuple[np.ndarray, int]] = {}
    for i in range(len(key)):
        k = key[i]
        if k in state:
            s, last_season = state[k]
            if season[i] != last_season:
                s = s * carry ** (int(season[i]) - int(last_season))
        else:
            s = np.zeros(x.shape[1])
        out[i] = s
        state[k] = (decay * (s + x[i]), int(season[i]))
    return out


def team_tendencies(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Per (game_id, posteam): the offense's tendencies and efficiency from its earlier
    games only (decayed, shrunk toward fixed priors), and the opponent defense's EPA
    allowed in ITS earlier games. Nothing from the game itself or any later game."""
    t = cfg["tendency"]
    agg = team_game_aggregates(df, cfg)
    off_cols = ["n", "n_pass", "n_ne", "n_pass_ne", "s_pass_epa", "s_run_epa"]
    x = agg.select(off_cols).to_numpy().astype(float)
    season = agg["season"].to_numpy()
    s = decayed_prior_sums(agg["posteam"].to_numpy(), season, x, t["game_decay"],
                           t["season_carry"])  # fmt: skip
    k = float(t["prior_plays"])
    n, n_pass, n_ne, n_pass_ne, s_pe, s_re = (s[:, i] for i in range(6))
    n_run = n - n_pass
    dx = agg.select("n", "s_epa").to_numpy().astype(float)
    sd = decayed_prior_sums(agg["defteam"].to_numpy(), season, dx, t["game_decay"],
                            t["season_carry"])  # fmt: skip
    return agg.select("game_id", "posteam").with_columns(
        pl.Series("tend_pass_rate", (n_pass + k * t["prior_pass_rate"]) / (n + k)),
        pl.Series("tend_pass_rate_ne",
                  (n_pass_ne + k * t["prior_pass_rate_neutral_early"]) / (n_ne + k)),
        pl.Series("tend_pass_epa", (s_pe + k * t["prior_pass_epa"]) / (n_pass + k)),
        pl.Series("tend_run_epa", (s_re + k * t["prior_run_epa"]) / (n_run + k)),
        pl.Series("tend_n_eff", n),
        pl.Series("opp_def_epa_allowed",
                  (sd[:, 1] + k * t["prior_allowed_epa"]) / (sd[:, 0] + k)),
    )  # fmt: skip


def build_features(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """`called_plays` output plus history and prior-games tendency features."""
    out = add_history(df).join(team_tendencies(df, cfg), on=["game_id", "posteam"],
                               how="left")  # fmt: skip
    return out.sort("row")


def feature_names(cfg: dict, groups: list[str]) -> list[str]:
    return [f for g in groups for f in cfg[f"{g}_features"]]


def matrix(df: pl.DataFrame, names: list[str]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float64) for c in names]).to_numpy()


# --------------------------------------------------------------------------------------
# Call models
# --------------------------------------------------------------------------------------


class GBMModel:
    """LightGBM classifier for P(pass), early-stopped on a held-out season."""

    def __init__(self, names: list[str], params: dict, seed: int, threads: int) -> None:
        self.names = names
        self.params = params
        self.seed = seed
        self.threads = threads

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame) -> GBMModel:
        import lightgbm as lgb

        p = dict(self.params)
        stop = p.pop("early_stopping_rounds")
        rounds = p.pop("n_estimators")
        p.update(objective="binary", seed=self.seed, num_threads=self.threads, verbose=-1,
                 deterministic=True, force_col_wise=True)  # fmt: skip
        dtrain = lgb.Dataset(matrix(train, self.names), train["is_pass"].to_numpy(),
                             feature_name=self.names, free_raw_data=True)  # fmt: skip
        dvalid = lgb.Dataset(matrix(valid, self.names), valid["is_pass"].to_numpy(),
                             reference=dtrain)  # fmt: skip
        self.model_ = lgb.train(p, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
                                callbacks=[lgb.early_stopping(stop, verbose=False)])  # fmt: skip
        self.best_iteration_ = int(self.model_.best_iteration or rounds)
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        prob = self.model_.predict(matrix(df, self.names), num_iteration=self.best_iteration_)
        return np.clip(prob, EPS, 1 - EPS)

    def importance(self) -> dict[str, float]:
        """Share of total gain by feature."""
        g = self.model_.feature_importance("gain", iteration=self.best_iteration_)
        return {n: float(v / g.sum()) for n, v in zip(self.names, g, strict=True)}


# Token columns of the sequence model: one vector per earlier called play.
TOKEN_COLUMNS = ["is_pass", "yards", "success", "first_down", "incomplete_pass", "sack",
                 "turnover", "down", "ydstogo", "yardline_100", "epa"]  # fmt: skip


def tokens(df: pl.DataFrame) -> np.ndarray:
    """Scaled per-play vectors that the GRU reads for later plays (never for the play
    itself: `history_batch` takes only earlier positions)."""
    t = matrix(df, TOKEN_COLUMNS)
    t = np.nan_to_num(t)
    t[:, 1] = np.clip(t[:, 1], -20, 50) / 10.0
    t[:, 7] = t[:, 7] / 4.0
    t[:, 8] = np.clip(t[:, 8], 0, 30) / 10.0
    t[:, 9] = t[:, 9] / 100.0
    t[:, 10] = np.clip(t[:, 10], -6, 6) / 3.0
    return t.astype(np.float32)


def group_starts(df: pl.DataFrame) -> np.ndarray:
    """For each row, the row index where its (game, offense) sequence starts. `df` must
    be in `row` order (sorted by game, offense, snap)."""
    key = (df["game_id"] + "|" + df["posteam"]).to_numpy()
    start = np.zeros(len(key), dtype=np.int64)
    for i in range(1, len(key)):
        start[i] = start[i - 1] if key[i] == key[i - 1] else i
    return start


def history_batch(rows: np.ndarray, start: np.ndarray, drive: np.ndarray, tok: np.ndarray,
                  k: int) -> np.ndarray:  # fmt: skip
    """The last `k` earlier plays of each row's sequence, oldest first, left-padded with
    zeros. Each step adds three columns: valid, same drive, plays ago / k. Only indices
    strictly below the row itself are used."""
    idx = rows[:, None] - np.arange(k, 0, -1)[None, :]
    valid = idx >= start[rows][:, None]
    safe = np.where(valid, idx, 0)
    h = tok[safe] * valid[..., None]
    same = (drive[safe] == drive[rows][:, None]) & valid
    ago = (rows[:, None] - idx) / k * valid
    extra = np.stack([valid, same, ago], axis=-1).astype(np.float32)
    return np.concatenate([h, extra], axis=-1)


class SeqModel:
    """P(pass) from the situation and (optionally) a GRU over the offense's earlier calls
    and results in the same game. With `use_history=False` it is the same network
    without the GRU, so the comparison isolates what the history adds."""

    def __init__(self, names: list[str], params: dict, seed: int, threads: int,
                 use_history: bool) -> None:  # fmt: skip
        self.names = names
        self.params = params
        self.seed = seed
        self.threads = threads
        self.use_history = use_history

    def _situation(self, df: pl.DataFrame) -> np.ndarray:
        x = matrix(df, self.names)
        miss = np.isnan(x[:, self.miss_cols_]).astype(np.float32)
        x = np.where(np.isnan(x), self.mean_, x)
        return np.concatenate([(x - self.mean_) / self.sd_, miss], axis=1).astype(np.float32)

    def _net(self, n_sit: int, n_tok: int) -> Any:
        import torch
        from torch import nn

        hidden, mlp, use_h = self.params["hidden"], self.params["mlp"], self.use_history

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(n_tok, hidden, batch_first=True) if use_h else None
                width = n_sit + (hidden if use_h else 0)
                self.head = nn.Sequential(nn.Linear(width, mlp), nn.ReLU(),
                                          nn.Linear(mlp, mlp // 2), nn.ReLU(),
                                          nn.Linear(mlp // 2, 1))  # fmt: skip

            def forward(self, sit: torch.Tensor, hist: torch.Tensor | None) -> torch.Tensor:
                if self.gru is not None and hist is not None:
                    out, _ = self.gru(hist)
                    sit = torch.cat([sit, out[:, -1]], dim=1)
                return self.head(sit).squeeze(1)

        return Net()

    def _batches(self, df: pl.DataFrame, rows: np.ndarray, shuffle: bool,
                 rng: np.random.Generator) -> Any:  # fmt: skip
        import torch

        sit = torch.from_numpy(self._situation(df))
        y = torch.from_numpy(df["is_pass"].to_numpy().astype(np.float32))
        order = rng.permutation(len(rows)) if shuffle else np.arange(len(rows))
        bs = self.params["batch_size"]
        for b in range(0, len(order), bs):
            local = order[b : b + bs]
            hist = None
            if self.use_history:
                hist = torch.from_numpy(history_batch(rows[local], self.start_, self.drive_,
                                                      self.tok_, self.params["history_len"]))
            yield sit[local], hist, y[local]

    def _bind(self, df: pl.DataFrame) -> None:
        """Sequence arrays for `df` (which must be the whole table in `row` order)."""
        self.start_ = group_starts(df)
        self.drive_ = df["fixed_drive"].fill_null(-1).to_numpy()
        self.tok_ = tokens(df)

    def _logits(self, table: pl.DataFrame, rows: np.ndarray) -> np.ndarray:
        import torch

        self.net_.eval()
        out = []
        with torch.no_grad():
            for sit, hist, _ in self._batches(table[rows], rows, False,
                                              np.random.default_rng(0)):  # fmt: skip
                out.append(self.net_(sit, hist).numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def fit(self, table: pl.DataFrame, train_rows: np.ndarray, valid_rows: np.ndarray
            ) -> SeqModel:  # fmt: skip
        """Fit on `table[train_rows]`, keeping the epoch with the best log loss on
        `table[valid_rows]`. `table` is the full feature table in `row` order (the GRU
        reads earlier rows of the same game, which are always in it)."""
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        rng = np.random.default_rng(self.seed)
        tr = table[train_rows]
        x = matrix(tr, self.names)
        self.mean_ = np.nanmean(x, axis=0)
        self.mean_ = np.where(np.isnan(self.mean_), 0.0, self.mean_)
        sd = np.nanstd(x, axis=0)
        self.sd_ = np.where((sd > 0) & np.isfinite(sd), sd, 1.0)
        self.miss_cols_ = np.where(np.isnan(x).any(axis=0))[0]
        self._bind(table)
        n_sit = len(self.names) + len(self.miss_cols_)
        self.net_ = self._net(n_sit, len(TOKEN_COLUMNS) + 3)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.params["lr"],
                               weight_decay=self.params["weight_decay"])  # fmt: skip
        loss_fn = torch.nn.BCEWithLogitsLoss()
        y_val = table[valid_rows]["is_pass"].to_numpy()
        best, best_state, self.curve_ = np.inf, None, []
        for _ in range(self.params["epochs"]):
            self.net_.train()
            for sit, hist, y in self._batches(tr, train_rows, True, rng):
                opt.zero_grad()
                loss = loss_fn(self.net_(sit, hist), y)
                loss.backward()
                opt.step()
            p = self._prob(table, valid_rows)
            ll = float(log_loss_rows(y_val, p).mean())
            self.curve_.append(ll)
            if ll < best:
                best = ll
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
        self.net_.load_state_dict(best_state)
        self.best_epoch_ = int(np.argmin(self.curve_)) + 1
        return self

    def _prob(self, table: pl.DataFrame, rows: np.ndarray) -> np.ndarray:
        z = self._logits(table, rows)
        return np.clip(1.0 / (1.0 + np.exp(-z)), EPS, 1 - EPS)

    def predict(self, table: pl.DataFrame, rows: np.ndarray) -> np.ndarray:
        """P(pass) for `table[rows]`; `table` is a full feature table in `row` order."""
        self._bind(table)
        return self._prob(table, rows)


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def log_loss_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Log loss of each call in nats: -log P(the call actually made)."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def p_call(y: np.ndarray, p_pass: np.ndarray) -> np.ndarray:
    """P(the call actually made): p for a pass, 1 - p for a run."""
    return np.where(y == 1, p_pass, 1.0 - p_pass)


def calibration(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Reliability table in equal-width bins of predicted P(pass), and the expected
    calibration error (play-weighted mean |observed - predicted|)."""
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows, ece = [], 0.0
    for k in range(n_bins):
        m = b == k
        if not m.any():
            continue
        obs, pred = float(y[m].mean()), float(p[m].mean())
        ece += m.mean() * abs(obs - pred)
        rows.append({"bin": k, "n": int(m.sum()), "predicted": pred, "observed": obs})
    return {"bins": rows, "ece": float(ece)}


def auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def score(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    ll = log_loss_rows(y, p)
    base = float(y.mean())
    return {
        "n": int(len(y)),
        "log_loss": float(ll.mean()),
        "log_loss_base_rate": float(log_loss_rows(y, np.full_like(p, base)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "auc": auc(y, p),
        "calibration": calibration(y, p),
        "mean_p_call": float(p_call(y, p).mean()),
    }


def paired_diff(a: np.ndarray, b: np.ndarray, game: np.ndarray, weights: np.ndarray,
                level: float = 0.95) -> dict[str, float | None]:  # fmt: skip
    """Mean of (a - b) per row with a game-level bootstrap interval. `game` holds codes
    0..n_games-1 that index the columns of `weights` (from `run_pass.boot_weights`)."""
    n_games = weights.shape[1]
    s = np.bincount(game, weights=a - b, minlength=n_games)
    c = np.bincount(game, minlength=n_games).astype(float)
    boot = (weights @ s) / (weights @ c)
    return rp.summary(float(s.sum() / c.sum()), boot, level)


def game_codes(game_id: np.ndarray) -> np.ndarray:
    _, codes = np.unique(game_id, return_inverse=True)
    return codes.astype(np.int64)


# --------------------------------------------------------------------------------------
# The cost of predictability: fixed effects by weighted alternating projections
# --------------------------------------------------------------------------------------


@dataclass
class FEFit:
    """y = sum of group effects (+ X @ beta) + error, by weighted least squares.

    `effects[k]` has one value per level of group k. The first group is the one whose
    effects are reported (e.g. bins of P(call)); its effects are normalized to a
    weighted mean of zero over the rows, and the other groups absorb the level.
    """

    effects: list[np.ndarray]
    beta: np.ndarray
    iterations: int


def fe_fit(y: np.ndarray, groups: list[np.ndarray], w: np.ndarray,
           x: np.ndarray | None = None, n_iter: int = 200, tol: float = 1e-9,
           start: FEFit | None = None) -> FEFit:  # fmt: skip
    """Weighted least squares with several sets of fixed effects (and optional continuous
    regressors `x`), solved by alternating projections: each set of effects is the
    weighted mean of what the others leave unexplained, repeated until nothing moves.
    `start` warm-starts the iteration (used for bootstrap refits)."""
    sizes = [int(g.max()) + 1 for g in groups]
    if start is not None:
        eff = [e.copy() for e in start.effects]
        beta = start.beta.copy()
    else:
        eff = [np.zeros(s) for s in sizes]
        beta = np.zeros(0 if x is None else x.shape[1])
    den = [np.bincount(g, weights=w, minlength=s) for g, s in zip(groups, sizes, strict=True)]
    xtwx = None if x is None else (x * w[:, None]).T @ x
    fitted = sum(e[g] for e, g in zip(eff, groups, strict=True)) + (0 if x is None else x @ beta)
    it = 0
    for _ in range(n_iter):
        it += 1
        biggest = 0.0
        for k, g in enumerate(groups):
            r = y - fitted + eff[k][g]
            new = np.divide(np.bincount(g, weights=w * r, minlength=sizes[k]), den[k],
                            out=np.zeros(sizes[k]), where=den[k] > 0)  # fmt: skip
            fitted += (new - eff[k])[g]
            biggest = max(biggest, float(np.abs(new - eff[k]).max()))
            eff[k] = new
        if x is not None:
            r = y - fitted + x @ beta
            new_b = np.linalg.solve(xtwx + 1e-12 * np.eye(len(beta)), (x * w[:, None]).T @ r)
            fitted += x @ (new_b - beta)
            biggest = max(biggest, float(np.abs(new_b - beta).max()))
            beta = new_b
        if biggest < tol:
            break
    shift = float(np.sum(w * eff[0][groups[0]]) / np.sum(w))
    eff[0] = eff[0] - shift
    if len(eff) > 1:
        eff[1] = eff[1] + shift
    return FEFit(eff, beta, it)


def bin_curve(y: np.ndarray, pbin: np.ndarray, n_bins: int, controls: list[np.ndarray],
              game: np.ndarray, weights: np.ndarray, cfg: dict) -> dict[str, Any]:  # fmt: skip
    """Mean of `y` by bin of P(call), with the control fixed effects absorbed. Values are
    the bin effects plus the overall mean of `y`, so they read as "EPA per play in this
    bin at the average situation, team and opponent". Bootstrap: every game resample
    refits the whole model (warm-started)."""
    c = cfg["cost"]
    groups = [pbin, *controls]
    w0 = np.ones_like(y)
    point = fe_fit(y, groups, w0, n_iter=c["fe_iterations"], tol=c["fe_tolerance"])
    base = float(y.mean())
    n = np.bincount(pbin, minlength=n_bins)
    eff0 = np.full(n_bins, np.nan)
    eff0[: len(point.effects[0])] = point.effects[0]
    boot = np.full((weights.shape[0], n_bins), np.nan)
    for b in range(weights.shape[0]):
        w = weights[b][game].astype(float)
        fit = fe_fit(y, groups, w, n_iter=c["boot_fe_iterations"],
                     tol=c["boot_fe_tolerance"], start=point)  # fmt: skip
        mean_b = float(np.sum(w * y) / np.sum(w))
        boot[b, : len(fit.effects[0])] = fit.effects[0] + mean_b
    lo, hi = rp._interval(boot, cfg["level"])
    vals = eff0 + base
    return {
        "n": n.tolist(),
        "value": [rp._f(v) for v in vals],
        "low": [rp._f(v) for v in lo],
        "high": [rp._f(v) for v in hi],
        "boot": boot,
    }


def _resid(v: np.ndarray, fit: FEFit, groups: list[np.ndarray],
           w: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """`v` minus its fixed-effects fit, centered to a (weighted) mean of zero. Centering
    matters: `fe_fit` hands the overall level to the second group, so with a single
    group the raw residuals keep the mean of `v`, and a slope through the origin would
    then pick up mean(y) / mean(x)."""
    r = v - sum(e[g] for e, g in zip(fit.effects, groups, strict=True))
    w = np.ones_like(r) if w is None else w
    return r - np.sum(w * r) / np.sum(w)


def slope_fit(y: np.ndarray, x: np.ndarray, controls: list[np.ndarray], game: np.ndarray,
              weights: np.ndarray, cfg: dict) -> dict[str, Any]:  # fmt: skip
    """The slope of `y` on the continuous `x` (P(call)) with the control fixed effects
    absorbed, by Frisch-Waugh-Lovell: residualize x and y on the fixed effects, then
    regress residual on residual. The game-level bootstrap redoes both residualizations
    with the resample's weights (warm-started from the full-sample fits)."""
    c = cfg["cost"]
    x = x.astype(float)
    w0 = np.ones_like(y)
    fy = fe_fit(y, controls, w0, n_iter=c["fe_iterations"], tol=c["fe_tolerance"])
    fx = fe_fit(x, controls, w0, n_iter=c["fe_iterations"], tol=c["fe_tolerance"])
    yr, xr = _resid(y, fy, controls), _resid(x, fx, controls)
    point = float(np.sum(xr * yr) / np.sum(xr * xr))
    boot = np.empty(weights.shape[0])
    for b in range(weights.shape[0]):
        w = weights[b][game].astype(float)
        kw = {"n_iter": c["boot_fe_iterations"], "tol": c["boot_fe_tolerance"]}
        yb = _resid(y, fe_fit(y, controls, w, start=fy, **kw), controls, w)
        xb = _resid(x, fe_fit(x, controls, w, start=fx, **kw), controls, w)
        boot[b] = np.sum(w * xb * yb) / np.sum(w * xb * xb)
    out = rp.summary(point, boot, cfg["level"])
    out["boot"] = boot
    out["iterations"] = max(fy.iterations, fx.iterations)
    return out


def residualize(v: np.ndarray, controls: list[np.ndarray], cfg: dict) -> np.ndarray:
    """`v` minus its fixed-effects fit (unweighted)."""
    c = cfg["cost"]
    fit = fe_fit(v, controls, np.ones_like(v), n_iter=c["fe_iterations"],
                 tol=c["fe_tolerance"])  # fmt: skip
    return _resid(v, fit, controls)


# --------------------------------------------------------------------------------------
# Team estimates
# --------------------------------------------------------------------------------------


def shrink(est: np.ndarray, se: np.ndarray) -> dict[str, Any]:
    """Normal-normal empirical Bayes: true values ~ N(mu, tau^2), estimates ~ N(true, se^2).

    tau^2 by the method of moments (variance of the estimates minus their mean squared
    standard error, floored at zero), mu the precision-weighted mean. Returns the shrunk
    values, their posterior SDs, tau, and `noise_share`, the share of the observed
    variance across teams that sampling noise accounts for.
    """
    est, se = np.asarray(est, float), np.asarray(se, float)
    var_obs = float(np.var(est, ddof=1))
    noise = float(np.mean(se**2))
    tau2 = max(var_obs - noise, 0.0)
    wts = 1.0 / (tau2 + se**2)
    mu = float(np.sum(wts * est) / np.sum(wts))
    factor = tau2 / (tau2 + se**2)
    post = mu + factor * (est - mu)
    post_sd = np.sqrt(factor * se**2) if tau2 > 0 else np.zeros_like(se)
    return {
        "mu": mu,
        "tau": float(np.sqrt(tau2)),
        "var_observed": var_obs,
        "mean_se2": noise,
        "noise_share": float(min(noise / var_obs, 1.0)) if var_obs > 0 else 1.0,
        "reliability": factor,
        "shrunk": post,
        "post_sd": post_sd,
    }


def cluster_slope(xr: np.ndarray, yr: np.ndarray, game: np.ndarray) -> tuple[float, float]:
    """Slope of residual y on residual x through the origin, with a game-clustered
    (sandwich) standard error."""
    sxx = float(np.sum(xr * xr))
    if sxx <= 0:
        return float("nan"), float("nan")
    beta = float(np.sum(xr * yr) / sxx)
    e = yr - beta * xr
    _, codes = np.unique(game, return_inverse=True)
    score_g = np.bincount(codes, weights=xr * e)
    return beta, float(np.sqrt(np.sum(score_g**2)) / sxx)


def persistence(table: pl.DataFrame, value: str) -> dict[str, Any]:
    """Correlation of a team-season value with the same team's value the next season."""
    nxt = table.select("posteam", (pl.col("season") - 1).alias("season"),
                       pl.col(value).alias("next"))  # fmt: skip
    j = table.join(nxt, on=["posteam", "season"], how="inner")
    if j.height < 3:
        return {"n_pairs": j.height, "corr": None}
    a, b = j[value].to_numpy(), j["next"].to_numpy()
    return {"n_pairs": j.height, "corr": float(np.corrcoef(a, b)[0, 1])}
