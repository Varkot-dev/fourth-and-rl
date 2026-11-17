"""M1 win-probability model: P(the team with the ball wins the game | pre-play state).

A LightGBM gradient-boosted model with monotone constraints, followed by a
two-parameter calibration map (Platt scaling on the raw score) fitted on seasons
the trees never saw. Every feature is computed from a bare game state, so later
milestones can score hypothetical states built with `state_frame`.

Public API used by other milestones: FEATURES, make_features, WPModel, state_frame.
The teaching notes for this module are in docs/notes/m1_wp.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from fourth_and_rl.paths import CONFIGS

LABEL = "posteam_won"
DEFAULT_SEED = 20261117

# Columns a row must carry before features can be built. `down` may be null.
STATE_COLUMNS: list[str] = [
    "down", "ydstogo", "yardline_100", "game_seconds_remaining", "half_seconds_remaining",
    "qtr", "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "posteam_spread", "receive_2h_ko", "is_home",
]  # fmt: skip

# A row with a null in any of these cannot be scored (predict returns NaN for it).
REQUIRED_COLUMNS: list[str] = [
    "ydstogo", "yardline_100", "game_seconds_remaining", "half_seconds_remaining", "qtr",
    "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "posteam_spread", "receive_2h_ko", "is_home",
]  # fmt: skip

# Model inputs, in order. The first block is the raw state, the second is engineered.
FEATURES: list[str] = [
    "down",                        # 1 to 4, or 0 for plays with no down (kickoff, PAT)
    "ydstogo",
    "yardline_100",
    "goal_to_go",                  # ydstogo >= yardline_100 on a scrimmage down
    "game_seconds_remaining",
    "half_seconds_remaining",
    "qtr",                         # 1 to 4, 5 for any overtime period
    "is_overtime",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "posteam_spread",
    "receive_2h_ko",
    "is_home",
    "spread_time",                 # spread decayed as the game is played
    "expected_margin",             # lead now + spread still to be "earned"
    "margin_per_sqrt_time",        # expected margin in units of remaining randomness
]  # fmt: skip

# +1: the prediction may never fall when the feature rises, all else equal. -1: never rise.
# The three engineered features all rise with score differential and with the spread, so
# constraining them keeps the whole model monotone in those two raw inputs.
MONOTONE: dict[str, int] = {
    "score_differential": 1,
    "posteam_spread": 1,
    "spread_time": 1,
    "expected_margin": 1,
    "margin_per_sqrt_time": 1,
    "posteam_timeouts_remaining": 1,
    "defteam_timeouts_remaining": -1,
}

REGULATION_SECONDS = 3600.0
SPREAD_DECAY = 4.0  # spread_time = spread * exp(-4 * share of regulation played)
TIME_FLOOR_SECONDS = 30.0  # keeps margin_per_sqrt_time finite at 0 seconds

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.06,
    "num_leaves": 31,
    "min_child_samples": 400,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 10.0,
    "max_bin": 127,
    "monotone_constraints_method": "intermediate",
}

EPS = 1e-6  # predictions are clipped to [EPS, 1 - EPS] before any log is taken


def load_config(path: Path | None = None) -> dict:
    """Read configs/wp.yaml (split, seed, hyperparameters)."""
    with open(path or CONFIGS / "wp.yaml") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- features


def _feature_exprs() -> list[pl.Expr]:
    gsr = pl.col("game_seconds_remaining").cast(pl.Float64)
    diff = pl.col("score_differential").cast(pl.Float64)
    spread = pl.col("posteam_spread").cast(pl.Float64)
    down = pl.col("down").cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
    overtime = pl.col("qtr") >= 5
    # Share of regulation still to play. In overtime the clock column restarts, so the
    # share is set to zero there: the pregame spread has had a full game to show itself.
    remaining = pl.when(overtime).then(0.0).otherwise(gsr / REGULATION_SECONDS)
    expected_margin = diff + spread * remaining
    return [
        down.alias("down"),
        pl.col("ydstogo").cast(pl.Float64),
        pl.col("yardline_100").cast(pl.Float64),
        ((down > 0) & (pl.col("ydstogo") >= pl.col("yardline_100")))
        .cast(pl.Float64)
        .alias("goal_to_go"),
        gsr.alias("game_seconds_remaining"),
        pl.col("half_seconds_remaining").cast(pl.Float64),
        pl.col("qtr").cast(pl.Float64).clip(1.0, 5.0).alias("qtr"),
        overtime.cast(pl.Float64).alias("is_overtime"),
        diff.alias("score_differential"),
        pl.col("posteam_timeouts_remaining").cast(pl.Float64),
        pl.col("defteam_timeouts_remaining").cast(pl.Float64),
        spread.alias("posteam_spread"),
        pl.col("receive_2h_ko").cast(pl.Float64),
        pl.col("is_home").cast(pl.Float64),
        (spread * (-SPREAD_DECAY * (1.0 - remaining)).exp()).alias("spread_time"),
        expected_margin.alias("expected_margin"),
        (expected_margin / ((gsr + TIME_FLOOR_SECONDS) / REGULATION_SECONDS).sqrt()).alias(
            "margin_per_sqrt_time"
        ),
    ]


def make_features(df: pl.DataFrame) -> pl.DataFrame:
    """Build the model inputs from a frame holding STATE_COLUMNS.

    Returns a Float64 frame with exactly the columns in FEATURES, one row per input
    row, in the same order. Nothing outside the pre-play state is read, so this works
    on real plays and on hypothetical states from `state_frame` alike.
    """
    missing = [c for c in STATE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"make_features needs columns {missing}")
    return df.select(_feature_exprs()).select(FEATURES)


def scoreable(df: pl.DataFrame) -> np.ndarray:
    """Boolean mask of rows whose state is complete enough to score (`down` may be null)."""
    ok = pl.all_horizontal(
        [
            pl.col(c).is_not_null() & pl.col(c).cast(pl.Float64).is_not_nan()
            for c in REQUIRED_COLUMNS
        ]
    )
    return df.select(ok.alias("ok"))["ok"].to_numpy()


def _derive_clock(gsr: np.ndarray, overtime: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quarter and seconds left in the half from seconds left in the game.

    Boundaries follow the start of the next period: exactly 1800 seconds left is the
    second-half kickoff (quarter 3, 1800 seconds in the half), not the end of the
    second quarter.
    """
    qtr = np.select([gsr > 2700, gsr > 1800, gsr > 900], [1.0, 2.0, 3.0], default=4.0)
    half = np.where(gsr > 1800, gsr - 1800, gsr)
    qtr = np.where(overtime, 5.0, qtr)
    half = np.where(overtime, gsr, half)
    return qtr, half


def state_frame(
    down: Any = 1,
    ydstogo: Any = 10,
    yardline_100: Any = 75,
    game_seconds_remaining: Any = 1800,
    score_differential: Any = 0,
    posteam_timeouts_remaining: Any = 3,
    defteam_timeouts_remaining: Any = 3,
    posteam_spread: Any = 0.0,
    receive_2h_ko: Any = 0,
    is_home: Any = 0,
    overtime: Any = False,
) -> pl.DataFrame:
    """Build hypothetical game states that `WPModel.predict` can score.

    Every argument is a scalar or an array; they are broadcast against each other.
    Everything is from the point of view of the team with the ball. `down=0` (or
    None) means a play with no down, such as a kickoff, where `yardline_100` is the
    kicking spot as in the play-by-play data. With `overtime=True`,
    `game_seconds_remaining` is the time left in the overtime period.

    Derived columns: `qtr` and `half_seconds_remaining` come from the game clock, and
    `receive_2h_ko` is forced to 0 outside the first half, as in the real data.
    """
    down = 0 if down is None else down
    arrays = np.broadcast_arrays(
        *[
            np.atleast_1d(np.asarray(a, dtype=float))
            for a in (
                down, ydstogo, yardline_100, game_seconds_remaining, score_differential,
                posteam_timeouts_remaining, defteam_timeouts_remaining, posteam_spread,
                receive_2h_ko, is_home, overtime,
            )
        ]
    )  # fmt: skip
    (dn, togo, yl, gsr, diff, pto, dto, spread, r2h, home, ot) = (a.copy() for a in arrays)
    ot = ot > 0
    if np.any((gsr < 0) | (gsr > REGULATION_SECONDS)):
        raise ValueError("game_seconds_remaining must be between 0 and 3600")
    if np.any((yl < 1) | (yl > 99)):
        raise ValueError("yardline_100 must be between 1 and 99")
    dn = np.where(np.isnan(dn), 0.0, dn)
    if np.any(~np.isin(dn, [0, 1, 2, 3, 4])):
        raise ValueError("down must be 1 to 4, or 0/None for a play with no down")
    qtr, half = _derive_clock(gsr, ot)
    r2h = np.where((qtr <= 2) & ~ot, r2h, 0.0)
    return pl.DataFrame(
        {
            "down": pl.Series(dn).replace(0.0, None),
            "ydstogo": togo,
            "yardline_100": yl,
            "game_seconds_remaining": gsr,
            "half_seconds_remaining": half,
            "qtr": qtr,
            "score_differential": diff,
            "posteam_timeouts_remaining": pto,
            "defteam_timeouts_remaining": dto,
            "posteam_spread": spread,
            "receive_2h_ko": r2h.astype(np.int8),
            "is_home": home.astype(np.int8),
        }
    )


# ------------------------------------------------------------------------ calibration


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """Maps the booster's raw score (log-odds) to a calibrated probability.

    method "platt":    p = sigmoid(a * score + b), two numbers fitted by logistic
                       regression. Smooth and strictly increasing.
    method "isotonic": a non-decreasing step-like function fitted by isotonic
                       regression. More flexible, but flat in places.
    method "none":     p = sigmoid(score), the booster's own probability.
    """

    def __init__(self, method: str = "platt") -> None:
        if method not in ("platt", "isotonic", "none"):
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self.a = 1.0
        self.b = 0.0
        self.iso: IsotonicRegression | None = None

    def fit(self, score: np.ndarray, y: np.ndarray) -> Calibrator:
        if self.method == "platt":
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            lr.fit(score.reshape(-1, 1), y.astype(int))
            self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
            if self.a <= 0:
                raise RuntimeError("Platt slope is not positive; the model is not informative")
        elif self.method == "isotonic":
            self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.iso.fit(_sigmoid(score), y)
        return self

    def transform(self, score: np.ndarray) -> np.ndarray:
        if self.method == "platt":
            p = _sigmoid(self.a * score + self.b)
        elif self.method == "isotonic":
            assert self.iso is not None
            p = self.iso.predict(_sigmoid(score))
        else:
            p = _sigmoid(score)
        return np.clip(p, EPS, 1 - EPS)

    def describe(self) -> dict:
        out: dict[str, Any] = {"method": self.method}
        if self.method == "platt":
            out.update(slope=self.a, intercept=self.b)
        return out


# ------------------------------------------------------------------------------ model


class WPModel:
    """Win probability for the team with the ball.

    fit(train, calib) grows the trees on `train` and fits the calibration map on
    `calib`. The two frames must come from different games (in this project,
    different seasons). With calib=None the booster's own probabilities are used.
    """

    def __init__(
        self,
        params: dict | None = None,
        calibration: str = "platt",
        n_threads: int = 2,
    ) -> None:
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
        self.calibration = calibration
        self.n_threads = n_threads
        self.booster: lgb.Booster | None = None
        self.calibrator = Calibrator("none")
        self.seed: int | None = None
        self.n_train_rows = 0
        self.n_calib_rows = 0

    # -- fitting

    @staticmethod
    def _xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Features and labels for the rows that can be used for fitting."""
        if LABEL not in df.columns:
            raise KeyError(f"fit needs the label column {LABEL!r}")
        keep = scoreable(df) & df[LABEL].is_not_null().to_numpy()
        sub = df.filter(pl.Series(keep))
        return make_features(sub).to_numpy(), sub[LABEL].to_numpy().astype(np.float64)

    def fit(
        self,
        train: pl.DataFrame,
        calib: pl.DataFrame | None = None,
        seed: int = DEFAULT_SEED,
        n_estimators: int | None = None,
    ) -> WPModel:
        """Fit the trees on `train` and the calibration map on `calib`.

        Rows with no label (tied games) or an incomplete state are dropped. Pass
        `n_estimators` to override the number of trees (fewer trees for fast
        bootstrap refits).
        """
        p = dict(self.params)
        rounds = int(n_estimators or p.pop("n_estimators"))
        p.pop("n_estimators", None)
        p.update(
            objective="binary",
            seed=int(seed),
            num_threads=self.n_threads,
            verbose=-1,
            deterministic=True,
            force_col_wise=True,
            monotone_constraints=[MONOTONE.get(f, 0) for f in FEATURES],
        )
        x, y = self._xy(train)
        if len(np.unique(y)) < 2:
            raise ValueError("training data must contain both wins and losses")
        dataset = lgb.Dataset(x, label=y, feature_name=FEATURES, free_raw_data=True)
        self.booster = lgb.train(p, dataset, num_boost_round=rounds)
        self.seed, self.n_train_rows = int(seed), len(y)
        if calib is not None and calib.height > 0:
            xc, yc = self._xy(calib)
            score = self.booster.predict(xc, raw_score=True, num_threads=self.n_threads)
            self.calibrator = Calibrator(self.calibration).fit(score, yc)
            self.n_calib_rows = len(yc)
        else:
            self.calibrator = Calibrator("none")
            self.n_calib_rows = 0
        return self

    # -- prediction

    def raw_score(self, df: pl.DataFrame) -> np.ndarray:
        """The booster's log-odds, NaN for rows that cannot be scored."""
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        ok = scoreable(df)
        out = np.full(df.height, np.nan)
        if ok.any():
            x = make_features(df.filter(pl.Series(ok))).to_numpy()
            out[ok] = self.booster.predict(x, raw_score=True, num_threads=self.n_threads)
        return out

    def predict(self, df: pl.DataFrame, calibrated: bool = True) -> np.ndarray:
        """P(posteam wins), one value per row. NaN where the state is incomplete."""
        score = self.raw_score(df)
        ok = ~np.isnan(score)
        out = np.full(df.height, np.nan)
        if calibrated:
            out[ok] = self.calibrator.transform(score[ok])
        else:
            out[ok] = np.clip(_sigmoid(score[ok]), EPS, 1 - EPS)
        return out

    def feature_importance(self) -> dict[str, float]:
        """Share of the total loss reduction (gain) credited to each feature."""
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        gain = self.booster.feature_importance(importance_type="gain").astype(float)
        share = gain / gain.sum()
        order = np.argsort(-share)
        return {FEATURES[i]: float(share[i]) for i in order}

    # -- persistence

    def save(self, path: str | Path) -> Path:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "booster": self.booster.model_to_string(),
                "features": FEATURES,
                "params": self.params,
                "calibration": self.calibration,
                "calibrator": self.calibrator,
                "n_threads": self.n_threads,
                "seed": self.seed,
                "n_train_rows": self.n_train_rows,
                "n_calib_rows": self.n_calib_rows,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> WPModel:
        blob = joblib.load(path)
        if blob["features"] != FEATURES:
            raise ValueError("saved model was fitted with a different feature list; refit it")
        model = cls(blob["params"], blob["calibration"], blob["n_threads"])
        model.booster = lgb.Booster(model_str=blob["booster"])
        model.calibrator = blob["calibrator"]
        model.seed = blob["seed"]
        model.n_train_rows = blob["n_train_rows"]
        model.n_calib_rows = blob["n_calib_rows"]
        return model


# ---------------------------------------------------------------------------- metrics


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean of -[y log p + (1 - y) log(1 - p)]. Lower is better; 0.6931 is a coin flip."""
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probability. Lower is better; 0.25 is a coin flip."""
    return float(np.mean((p - y) ** 2))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, list]:
    """Equal-width bins of the prediction: count, mean prediction, observed win rate."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_pred = np.bincount(idx, weights=p, minlength=n_bins) / count
        observed = np.bincount(idx, weights=y, minlength=n_bins) / count
    return {
        "bin_lower": edges[:-1].tolist(),
        "bin_upper": edges[1:].tolist(),
        "count": count.astype(int).tolist(),
        "mean_pred": [None if np.isnan(v) else float(v) for v in mean_pred],
        "observed": [None if np.isnan(v) else float(v) for v in observed],
    }


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: the bin-size-weighted mean of |observed - predicted|."""
    t = reliability_table(y, p, n_bins)
    n = float(sum(t["count"]))
    total = 0.0
    for c, mp, ob in zip(t["count"], t["mean_pred"], t["observed"], strict=True):
        if c > 0:
            total += c / n * abs(ob - mp)
    return float(total)


def all_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    return {
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p, n_bins),
        "mean_pred": float(np.mean(p)),
        "win_rate": float(np.mean(y)),
    }


def game_bootstrap_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    game_ids: Sequence[Any],
    n_boot: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, float]]:
    """Game-level bootstrap of (metric of model a) minus (metric of model b).

    Whole games are resampled with replacement, because every play of a game shares
    one win/loss label. Returns the point estimate and a 95% interval for the
    difference in log loss and in Brier score. Negative means model a is better.
    """
    _, inv = np.unique(np.asarray(game_ids), return_inverse=True)
    n_games = int(inv.max()) + 1
    pa, pb = np.clip(p_a, EPS, 1 - EPS), np.clip(p_b, EPS, 1 - EPS)
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))  # noqa: E731
    per_play = {"log_loss": ll(pa) - ll(pb), "brier": (p_a - y) ** 2 - (p_b - y) ** 2}
    n_per_game = np.bincount(inv, minlength=n_games).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_games, size=(n_boot, n_games))
    denom = n_per_game[draws].sum(axis=1)
    out = {}
    for name, d in per_play.items():
        per_game = np.bincount(inv, weights=d, minlength=n_games)
        boot = per_game[draws].sum(axis=1) / denom
        lo, hi = np.quantile(boot, [0.025, 0.975])
        out[name] = {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi)}
    return out


# ----------------------------------------------------------------------- cross-fitting


def season_filter(df: pl.DataFrame, seasons: Sequence[int]) -> pl.DataFrame:
    return df.filter(pl.col("season").is_in(list(seasons)))


def crossfit_plan(
    seasons: Sequence[int], fold_size: int = 4, n_calib_seasons: int = 2
) -> list[dict[str, list[int]]]:
    """Leave-seasons-out folds of consecutive seasons.

    For each fold, the held-out seasons are predicted by a model whose trees are
    grown on the other seasons except the most recent `n_calib_seasons` of them,
    which are used for the calibration map.
    """
    seasons = sorted(seasons)
    plan = []
    for i in range(0, len(seasons), fold_size):
        held = seasons[i : i + fold_size]
        rest = [s for s in seasons if s not in held]
        if len(rest) <= n_calib_seasons:
            raise ValueError("not enough seasons outside the fold to train and calibrate")
        plan.append(
            {
                "predict": held,
                "train": rest[:-n_calib_seasons],
                "calib": rest[-n_calib_seasons:],
            }
        )
    return plan


def crossfit_predict(
    plays: pl.DataFrame,
    params: dict | None = None,
    calibration: str = "platt",
    fold_size: int = 4,
    n_calib_seasons: int = 2,
    seed: int = DEFAULT_SEED,
    n_threads: int = 2,
) -> tuple[pl.DataFrame, list[dict[str, list[int]]]]:
    """Out-of-fold win probability for every play.

    Returns (frame with game_id, play_id, wp_ours; plan). `wp_ours` is null for rows
    whose state is incomplete. No game is ever predicted by a model that saw it.
    """
    plan = crossfit_plan(plays["season"].unique().to_list(), fold_size, n_calib_seasons)
    parts = []
    for fold in plan:
        model = WPModel(params, calibration, n_threads).fit(
            season_filter(plays, fold["train"]), season_filter(plays, fold["calib"]), seed=seed
        )
        held = season_filter(plays, fold["predict"])
        pred = model.predict(held)
        parts.append(
            held.select("game_id", "play_id").with_columns(
                pl.Series("wp_ours", pred).fill_nan(None)
            )
        )
    return pl.concat(parts), plan


# ----------------------------------------------------------------------- sanity checks


def sanity_checks(model: WPModel) -> dict[str, Any]:
    """Behaviour any sensible win-probability model must show, on hypothetical states.

    Returns the measured quantities and a `passed` flag for each check. Used by
    scripts/fit_wp.py (which fails loudly if a check fails) and by the tests.
    """
    out: dict[str, Any] = {}

    # 1 and 2. Never worse with more points or a better spread, everything else fixed.
    # Guaranteed by the monotone constraints, checked here end to end.
    secs = np.array([3300.0, 2000.0, 1700.0, 900.0, 300.0, 60.0])
    yards = np.array([90.0, 60.0, 25.0, 5.0])
    diffs = np.arange(-30.0, 31.0)
    spreads = np.arange(-14.0, 14.5, 0.5)
    s, yl, d = np.meshgrid(secs, yards, diffs, indexing="ij")
    p = model.predict(
        state_frame(yardline_100=yl.ravel(), game_seconds_remaining=s.ravel(),
                    score_differential=d.ravel(), ydstogo=np.minimum(10, yl.ravel()))
    ).reshape(s.shape)  # fmt: skip
    worst = float(np.min(np.diff(p, axis=2)))
    out["score_differential"] = {"min_step": worst, "passed": worst >= -1e-9}
    s, yl, sp = np.meshgrid(secs, yards, spreads, indexing="ij")
    p = model.predict(
        state_frame(yardline_100=yl.ravel(), game_seconds_remaining=s.ravel(),
                    posteam_spread=sp.ravel(), ydstogo=np.minimum(10, yl.ravel()))
    ).reshape(s.shape)  # fmt: skip
    worst = float(np.min(np.diff(p, axis=2)))
    out["posteam_spread"] = {"min_step": worst, "passed": worst >= -1e-9}

    # 3. Field position. Not constrained, so it is checked on averages: first and 10,
    # tied, averaged over the clock, WP at the opponent 20 must beat WP at the own 20,
    # and the curve along the field should have no large reversals.
    field = np.arange(10.0, 91.0)
    s, yl = np.meshgrid(secs, field, indexing="ij")
    p = model.predict(
        state_frame(yardline_100=yl.ravel(), game_seconds_remaining=s.ravel())
    ).reshape(s.shape).mean(axis=0)  # fmt: skip
    gain = float(p[field == 20][0] - p[field == 80][0])
    reversal = float(np.max(np.diff(p)))  # WP should fall as yardline_100 grows
    out["field_position"] = {
        "wp_opp_20_minus_own_20": gain,
        "largest_reversal": reversal,
        "passed": gain > 0.03 and reversal < 0.01,
    }

    # 4. A 30-point lead with two minutes left is all but decided.
    up = float(model.predict(state_frame(score_differential=30, game_seconds_remaining=120))[0])
    down = float(model.predict(state_frame(score_differential=-30, game_seconds_remaining=120))[0])
    out["blowout"] = {"up_30": up, "down_30": down, "passed": up > 0.99 and down < 0.01}

    # 5. Mirror at the opening kickoff. The model only speaks for the team with the
    # ball, so a state cannot be flipped directly. But take one matchup in which the
    # home side is favored by s. If the home side receives, its WP is the first
    # prediction below; if the away side receives, its WP is the second. Those are
    # the two opponents of the same matchup, differing only in who receives, so the
    # two numbers must sum to 1 plus twice the (small) value of receiving first.
    # The tolerance is loose on purpose: it catches a model that treats the two
    # sides of a game inconsistently, not one that is off by a point or two.
    sp = np.arange(-10.0, 10.5, 0.5)
    kick = dict(down=0, ydstogo=0, yardline_100=35, game_seconds_remaining=3600)
    home = model.predict(state_frame(posteam_spread=sp, is_home=1, **kick))
    away = model.predict(state_frame(posteam_spread=-sp, is_home=0, **kick))
    dev = float(np.max(np.abs(home + away - 1.0)))
    out["kickoff_mirror"] = {
        "max_abs_deviation_from_1": dev,
        "mean_sum": float(np.mean(home + away)),
        "even_game_home_receiver": float(home[sp == 0][0]),
        "even_game_away_receiver": float(away[sp == 0][0]),
        "passed": dev < 0.08,
    }
    out["all_passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out
