"""Burning question 2: why do offenses run when passing gains more?

The pieces used by `scripts/questions/q2_run_vs_pass.py`:

- classifying each play as a called run or a called pass (scrambles are called passes),
- the neutral game-script filter and the situation cells,
- run-versus-pass comparisons with a game-level bootstrap (whole games are resampled,
  because plays of one game are not independent),
- the cell-by-cell comparison that separates "passes are better on average" from
  "a pass is better than a run in the same situation",
- leave-one-game-out team tendencies, sequential dependence and box counts, for the
  question of whether defenses adjust,
- `RunPassOutcomes`: outcome distributions for a run and for a pass in the array format
  of `ComponentModels.play_outcomes`, so the dynamic program can add a run-or-pass action.

`epa`, `wp`, `wpa` and `xpass` are nflfastR's models, not ours. They are used as given
and quoted as theirs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from fourth_and_rl import components as cm
from fourth_and_rl.paths import CONFIGS, ROOT

CALLS = ("run", "pass")

# Columns read from the raw nflverse play-by-play that are not in our state table.
EXTRA_PBP_COLUMNS = ["game_id", "play_id", "qb_scramble", "qb_dropback", "shotgun", "no_huddle",
                     "xpass", "pass_oe", "wpa", "complete_pass", "incomplete_pass",
                     "out_of_bounds", "pass", "rush"]  # fmt: skip
EXTRA_COLUMNS = [c for c in EXTRA_PBP_COLUMNS if c not in ("game_id", "play_id")] + [
    "defenders_in_box"
]

# Columns `prepare` adds; the outcome measures compared between runs and passes.
METRICS = ["epa", "success", "yards", "first_down", "turnover", "touchdown"]


def load_config(path: Path | None = None) -> dict:
    """Read configs/run_pass.yaml."""
    with open(path or CONFIGS / "run_pass.yaml") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------------------
# Extra nflverse columns
# --------------------------------------------------------------------------------------


def download_extra(seasons: list[int], cfg: dict, force: bool = False) -> list[Path]:
    """Cache the extra play-by-play columns, and `defenders_in_box` from the nflverse
    participation files where they exist, one Parquet file per season."""
    import nflreadpy as nfl

    out_dir = ROOT / cfg["extra_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for season in seasons:
        path = out_dir / f"extra_{season}.parquet"
        if force or not path.exists():
            pbp = nfl.load_pbp(season).select(EXTRA_PBP_COLUMNS)
            box = None
            if season >= cfg["participation_first_season"]:
                box = nfl.load_participation(season).select(
                    pl.col("nflverse_game_id").alias("game_id"),
                    pl.col("play_id").cast(pl.Float64),
                    # 0 is how the newer files write "not charted".
                    pl.when(pl.col("defenders_in_box") > 0)
                    .then(pl.col("defenders_in_box"))
                    .otherwise(None)
                    .cast(pl.Float64)
                    .alias("defenders_in_box"),
                ).unique(["game_id", "play_id"])
            pbp = pbp.with_columns(pl.col("play_id").cast(pl.Float64))
            if box is None:
                pbp = pbp.with_columns(pl.lit(None, dtype=pl.Float64).alias("defenders_in_box"))
            else:
                pbp = pbp.join(box, on=["game_id", "play_id"], how="left")
            pbp.write_parquet(path)
        paths.append(path)
    return paths


def load_extra(seasons: list[int], cfg: dict) -> pl.DataFrame:
    """The cached extra columns for `seasons` (downloads what is missing)."""
    frames = [pl.read_parquet(p) for p in download_extra(seasons, cfg)]
    return pl.concat(frames, how="vertical_relaxed")


def with_extra(plays: pl.DataFrame, extra: pl.DataFrame) -> pl.DataFrame:
    """Join the extra columns onto the state table by game and play id."""
    out = plays.join(extra, on=["game_id", "play_id"], how="left")
    if out.height != plays.height:
        raise ValueError("extra columns are not unique per (game_id, play_id)")
    return out


# --------------------------------------------------------------------------------------
# Classification, neutral filter, cells
# --------------------------------------------------------------------------------------


def _col(df: pl.DataFrame, name: str) -> pl.Expr:
    """The column if the frame has it, else a null literal (so toy frames stay small)."""
    return pl.col(name) if name in df.columns else pl.lit(None, dtype=pl.Float64)


def classify_calls(df: pl.DataFrame, scrambles_as: str = "pass") -> pl.DataFrame:
    """Add `call`: "pass", "run" or null.

    A play counts when it is a run or a pass on downs 1 to 4 that was actually run:
    kneels, spikes, two-point tries and penalty no-plays are null. Sacks are passes
    (nflverse already records them as `play_type` "pass"). A quarterback scramble is
    recorded by nflverse as a run; `scrambles_as` says how to count it:
      "pass"  a called pass that the quarterback ran with (the default),
      "run"   as recorded,
      "drop"  left out.
    Without a `qb_scramble` column every recorded run stays a run.
    """
    if scrambles_as not in ("pass", "run", "drop"):
        raise ValueError("scrambles_as must be 'pass', 'run' or 'drop'")
    live = (
        pl.col("play_type").is_in(["run", "pass"])
        & pl.col("down").is_between(1, 4)
        & (_col(df, "qb_kneel").fill_null(0) != 1)
        & (_col(df, "qb_spike").fill_null(0) != 1)
        & (_col(df, "two_point_attempt").fill_null(0) != 1)
        & pl.col("yardline_100").is_between(1, 99)
        & (pl.col("ydstogo") >= 1)
    ).fill_null(False)
    scramble = (pl.col("play_type") == "run") & (_col(df, "qb_scramble").fill_null(0) == 1)
    as_run = pl.col("play_type") == "run"
    if scrambles_as == "pass":
        as_run = as_run & ~scramble
    call = pl.when(as_run).then(pl.lit("run")).otherwise(pl.lit("pass"))
    if scrambles_as == "drop":
        live = live & ~scramble
    return df.with_columns(pl.when(live).then(call).otherwise(None).alias("call"))


def neutral_expr(cfg: dict) -> pl.Expr:
    """True where the game script does not push the play call: close score, win
    probability (nflfastR's) in the middle, not the end of a half, no overtime."""
    n = cfg["neutral"]
    return (
        (pl.col("score_differential").abs() <= n["max_abs_score_diff"])
        & pl.col("wp").is_between(n["wp_low"], n["wp_high"])
        & (pl.col("half_seconds_remaining") > n["min_half_seconds"])
        & (pl.col("qtr") <= n["max_qtr"])
    ).fill_null(False)


def bin_index(x: np.ndarray, edges: list[int] | list[float] | np.ndarray) -> np.ndarray:
    """Index of the bin each value falls in; `edges` are lower edges."""
    e = np.asarray(edges)
    return np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 1)


def n_cells(cfg: dict) -> int:
    return 4 * len(cfg["dist_edges"]) * len(cfg["field_edges"])


def cell_labels(cfg: dict) -> list[dict[str, Any]]:
    """Down, distance label and field label of every cell, in cell-index order."""
    return [
        {"down": d, "distance": dl, "yardline_100": fl}
        for d in (1, 2, 3, 4)
        for dl in cfg["dist_labels"]
        for fl in cfg["field_labels"]
    ]


def prepare(plays: pl.DataFrame, cfg: dict, scrambles_as: str | None = None) -> pl.DataFrame:
    """Called runs and passes with the columns every analysis here uses.

    Adds `call`, `is_pass`, the outcome measures (`success` is EPA > 0, `turnover` is an
    interception or a lost fumble, `touchdown` an offensive touchdown), `neutral`,
    `early`, `dist_bin`, `field_bin`, `cell`, `game_code` (0..number of games - 1, for
    the bootstrap), `seconds_elapsed` (game clock from this snap to the next row of the
    same half), `defense_timeout_after` (the defense has one timeout fewer at the
    offense's next snap) and the previous call by the same offense. Plays without an EPA
    value are dropped.
    """
    df = classify_calls(plays, scrambles_as or cfg["scrambles_as"])
    order = [c for c in ("game_id", "order_sequence", "play_id") if c in df.columns]
    df = df.sort(order)
    if "game_seconds_remaining" in df.columns and "game_half" in df.columns:
        gap = pl.col("game_seconds_remaining") - pl.col("game_seconds_remaining").shift(-1)
        same_half = pl.col("game_half") == pl.col("game_half").shift(-1)
        same_offense = same_half & (pl.col("posteam") == pl.col("posteam").shift(-1))
        spent = _col(df, "defteam_timeouts_remaining").shift(-1) < _col(
            df, "defteam_timeouts_remaining"
        )
        df = df.with_columns(
            pl.when(same_half).then(gap).otherwise(None).over("game_id").alias("seconds_elapsed"),
            (same_offense & spent).fill_null(False).cast(pl.Float64).over("game_id")
            .alias("defense_timeout_after"),
        )  # fmt: skip
    else:
        df = df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("seconds_elapsed"),
            pl.lit(0.0).alias("defense_timeout_after"),
        )
    df = df.filter(pl.col("call").is_not_null() & pl.col("epa").is_not_null())
    turnover = (_col(df, "interception").fill_null(0) == 1) | (
        _col(df, "fumble_lost").fill_null(0) == 1
    )
    off_td = (_col(df, "touchdown").fill_null(0) == 1) & (
        (pl.col("td_team") == pl.col("posteam")) if "td_team" in df.columns else pl.lit(True)
    ).fill_null(False)
    for name in EXTRA_COLUMNS + ["sack", "fixed_drive", "season"]:
        if name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    df = df.with_columns(
        (pl.col("call") == "pass").cast(pl.Int8).alias("is_pass"),
        (pl.col("epa") > 0).cast(pl.Float64).alias("success"),
        pl.col("yards_gained").cast(pl.Float64).fill_null(0.0).alias("yards"),
        turnover.cast(pl.Float64).alias("turnover"),
        (off_td & ~turnover).cast(pl.Float64).alias("touchdown"),
        pl.col("first_down").cast(pl.Float64).fill_null(0.0),
        neutral_expr(cfg).alias("neutral"),
        pl.col("down").cast(pl.Int64).is_in(cfg["early_downs"]).alias("early"),
        pl.col("game_id").rank("dense").cast(pl.Int64).sub(1).alias("game_code"),
    )
    dist = bin_index(df["ydstogo"].to_numpy(), cfg["dist_edges"])
    field = bin_index(df["yardline_100"].to_numpy(), cfg["field_edges"])
    down = df["down"].to_numpy().astype(int)
    nd, nf = len(cfg["dist_edges"]), len(cfg["field_edges"])
    df = df.with_columns(
        pl.Series("dist_bin", dist.astype(np.int64)),
        pl.Series("field_bin", field.astype(np.int64)),
        pl.Series("cell", ((down - 1) * nd + dist) * nf + field),
    )
    by_offense = ["game_id", "posteam"]
    k = cfg["defense"]["recent_window"]
    is_pass = pl.col("is_pass").cast(pl.Float64)
    return df.with_columns(
        is_pass.shift(1).rolling_mean(k).over(by_offense).alias("recent_pass_share"),
        is_pass.shift(1).over([*by_offense, "fixed_drive"]).alias("prev_is_pass"),
        pl.col("down").shift(1).over([*by_offense, "fixed_drive"]).alias("prev_down"),
        pl.col("ydstogo").shift(1).over([*by_offense, "fixed_drive"]).alias("prev_ydstogo"),
        pl.col("yards").shift(1).over([*by_offense, "fixed_drive"]).alias("prev_yards"),
        pl.col("incomplete_pass").shift(1).over([*by_offense, "fixed_drive"])
        .alias("prev_incomplete"),
    )  # fmt: skip


# --------------------------------------------------------------------------------------
# Game-level bootstrap
# --------------------------------------------------------------------------------------


def boot_weights(n_games: int, n_boot: int, seed: int) -> np.ndarray:
    """How many times each game is drawn in each resample: shape (n_boot, n_games).

    Drawing n_games games with replacement is the same as giving every game a count
    from a multinomial distribution. One matrix is used for every statistic in the
    analysis, so all intervals come from the same resamples.
    """
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_games, np.full(n_games, 1.0 / n_games), size=n_boot).astype(
        np.float32
    )


def game_sums(game: np.ndarray, n_games: int, col: np.ndarray, n_cols: int,
              x: np.ndarray) -> np.ndarray:  # fmt: skip
    """Per-game sums: out[g, c, j] = sum of x[i, j] over rows with game g and column c."""
    flat = game.astype(np.int64) * n_cols + col
    out = np.empty((n_games, n_cols, x.shape[1]))
    for j in range(x.shape[1]):
        out[:, :, j] = np.bincount(flat, weights=x[:, j], minlength=n_games * n_cols).reshape(
            n_games, n_cols
        )
    return out


def _interval(boot: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    a = (1.0 - level) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanquantile(boot, a, axis=0), np.nanquantile(boot, 1.0 - a, axis=0)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full(np.broadcast(num, den).shape, np.nan), where=den > 0)


@dataclass
class Comparison:
    """Run-versus-pass means by group, with bootstrap replicates.

    `est`, `lo`, `hi` have shape (groups, 3, metrics); the middle axis is run, pass,
    pass minus run. `boot` has shape (n_boot, groups, 3, metrics). `n` and `boot_n` are
    play counts with a last axis of (run, pass).
    """

    names: list[str]
    n: np.ndarray
    est: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    boot: np.ndarray
    boot_n: np.ndarray

    def rows(self, labels: list[Any], key: str = "group") -> list[dict[str, Any]]:
        out = []
        for g, label in enumerate(labels):
            row: dict[str, Any] = dict(label) if isinstance(label, dict) else {key: label}
            total = int(self.n[g].sum())
            row.update(n_run=int(self.n[g, 0]), n_pass=int(self.n[g, 1]),
                       pass_rate=float(self.n[g, 1] / total) if total else None)  # fmt: skip
            for m, name in enumerate(self.names):
                row[name] = {
                    side: {"value": _f(self.est[g, s, m]), "low": _f(self.lo[g, s, m]),
                           "high": _f(self.hi[g, s, m])}
                    for s, side in enumerate(("run", "pass", "pass_minus_run"))
                }  # fmt: skip
            out.append(row)
        return out


def _f(x: float) -> float | None:
    return None if not np.isfinite(x) else float(x)


def compare(
    x: dict[str, np.ndarray],
    is_pass: np.ndarray,
    group: np.ndarray,
    n_groups: int,
    game: np.ndarray,
    weights: np.ndarray,
    level: float = 0.95,
    sd_of: tuple[str, ...] = (),
) -> Comparison:
    """Mean of each measure in `x` for runs and for passes within each group, and the
    pass-minus-run difference, with game-level bootstrap intervals.

    `group` holds a group index per play; rows with a negative index are left out.
    For each name in `sd_of` the standard deviation of that measure is added as
    "<name>_sd" (how spread out single-play outcomes are).
    """
    keep = group >= 0
    names = list(x)
    cols = [np.ones(int(keep.sum()))] + [np.asarray(x[k], dtype=float)[keep] for k in names]
    cols += [np.asarray(x[k], dtype=float)[keep] ** 2 for k in sd_of]
    n_games = weights.shape[1]
    s = game_sums(game[keep], n_games, group[keep] * 2 + is_pass[keep], n_groups * 2,
                  np.column_stack(cols))  # fmt: skip
    flat = s.reshape(n_games, -1)
    point = flat.sum(axis=0).reshape(n_groups, 2, -1)
    boot = (weights @ flat.astype(np.float32)).astype(float).reshape(-1, n_groups, 2, len(cols))

    def means(t: np.ndarray) -> np.ndarray:
        m = _ratio(t[..., 1 : 1 + len(names)], t[..., :1])
        if sd_of:
            idx = [names.index(k) for k in sd_of]
            second = _ratio(t[..., 1 + len(names) :], t[..., :1])
            sd = np.sqrt(np.maximum(second - m[..., idx] ** 2, 0.0))
            m = np.concatenate([m, sd], axis=-1)
        return np.concatenate([m, (m[..., 1:2, :] - m[..., 0:1, :])], axis=-2)

    est, b = means(point), means(boot)
    lo, hi = _interval(b, level)
    return Comparison([*names, *[f"{k}_sd" for k in sd_of]], point[..., 0].astype(int), est, lo,
                      hi, b, boot[..., 0])  # fmt: skip


def frame_compare(
    df: pl.DataFrame,
    group: np.ndarray,
    n_groups: int,
    weights: np.ndarray,
    level: float = 0.95,
    metrics: list[str] | None = None,
    sd_of: tuple[str, ...] = (),
) -> Comparison:
    """`compare` on a frame made by `prepare`."""
    x = {m: df[m].to_numpy().astype(float) for m in (metrics or METRICS)}
    return compare(x, df["is_pass"].to_numpy().astype(int), np.asarray(group, dtype=np.int64),
                   n_groups, df["game_code"].to_numpy(), weights, level, sd_of)  # fmt: skip


def group_means(
    x: np.ndarray,
    group: np.ndarray,
    n_groups: int,
    game: np.ndarray,
    weights: np.ndarray,
    level: float = 0.95,
) -> list[dict[str, Any]]:
    """Mean of one measure by group with game-level bootstrap intervals."""
    keep = (group >= 0) & np.isfinite(x)
    n_games = weights.shape[1]
    s = game_sums(game[keep], n_games, group[keep], n_groups,
                  np.column_stack([np.ones(int(keep.sum())), x[keep]]))  # fmt: skip
    flat = s.reshape(n_games, -1)
    point = flat.sum(axis=0).reshape(n_groups, 2)
    boot = (weights @ flat.astype(np.float32)).astype(float).reshape(-1, n_groups, 2)
    est, b = _ratio(point[:, 1], point[:, 0]), _ratio(boot[..., 1], boot[..., 0])
    lo, hi = _interval(b, level)
    return [
        {"n": int(point[g, 0]), "value": _f(est[g]), "low": _f(lo[g]), "high": _f(hi[g])}
        for g in range(n_groups)
    ]


def summary(point: float, boot: np.ndarray, level: float = 0.95) -> dict[str, float | None]:
    """A point estimate with its percentile interval."""
    lo, hi = _interval(np.asarray(boot, dtype=float), level)
    return {"value": _f(point), "low": _f(float(lo)), "high": _f(float(hi))}


# --------------------------------------------------------------------------------------
# Average versus marginal: the cell-by-cell comparison
# --------------------------------------------------------------------------------------


def classify_cells(comp: Comparison, metric: str, min_n: int) -> np.ndarray:
    """For each cell: 1 pass better (interval above zero), -1 run better (interval below
    zero), 0 a tie (interval covers zero), -9 too few plays to compare."""
    m = comp.names.index(metric)
    enough = (comp.n[:, 0] >= min_n) & (comp.n[:, 1] >= min_n)
    verdict = np.where(comp.lo[:, 2, m] > 0, 1, np.where(comp.hi[:, 2, m] < 0, -1, 0))
    return np.where(enough, verdict, -9)


def standardized_gap(
    comp: Comparison,
    metric: str,
    min_n: int,
    weight: str = "all",
    level: float = 0.95,
    cells: np.ndarray | None = None,
) -> dict[str, Any]:
    """The pass-minus-run gap with the situation held fixed.

    Within each cell take (pass mean - run mean), then average the cells with weights
    equal to the number of plays in them ("all") or the number of runs ("run", which
    answers: what would the runs have gained as passes in the same situations?).
    Compare with `raw`, the plain difference of means over the same plays, in which runs
    and passes come from different mixes of situations. Cells with fewer than `min_n`
    runs or passes are left out; `coverage` is the share of plays kept.
    """
    m = comp.names.index(metric)
    enough = (comp.n[:, 0] >= min_n) & (comp.n[:, 1] >= min_n)
    if cells is not None:
        enough &= cells
    if not enough.any():
        return {"standardized": None, "raw": None, "coverage": 0.0, "n_cells": 0}

    def gap(n: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, mean = n[..., enough, :], mean[..., m][..., enough, :]
        w = n.sum(axis=-1) if weight == "all" else n[..., 0]
        diff = np.nan_to_num(mean[..., 2])
        std = (w * diff).sum(axis=-1) / w.sum(axis=-1)
        tot = np.nan_to_num(mean[..., :2]) * n
        raw = tot[..., 1].sum(axis=-1) / n[..., 1].sum(axis=-1) - tot[..., 0].sum(
            axis=-1
        ) / n[..., 0].sum(axis=-1)
        return std, raw

    std, raw = gap(comp.n.astype(float), comp.est)
    std_b, raw_b = gap(comp.boot_n, comp.boot)
    return {
        "standardized": summary(float(std), std_b, level),
        "raw": summary(float(raw), raw_b, level),
        "raw_minus_standardized": summary(float(raw - std), raw_b - std_b, level),
        "coverage": float(comp.n[enough].sum() / max(comp.n.sum(), 1)),
        "n_cells": int(enough.sum()),
        "n_plays": int(comp.n[enough].sum()),
        "weight": weight,
        "boot": std_b,
    }


def cell_shares(comp: Comparison, verdict: np.ndarray) -> dict[str, float]:
    """Share of plays in cells where passing is better, tied, running is better, and in
    cells too small to compare."""
    total = max(int(comp.n.sum()), 1)
    names = {1: "pass_better", 0: "tie", -1: "run_better", -9: "too_few_plays"}
    return {v: float(comp.n[verdict == k].sum() / total) for k, v in names.items()}


# --------------------------------------------------------------------------------------
# Regression with game-clustered bootstrap
# --------------------------------------------------------------------------------------


def cluster_ols(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    game: np.ndarray,
    weights: np.ndarray,
    level: float = 0.95,
) -> dict[str, np.ndarray]:
    """Weighted least squares with game-level bootstrap intervals.

    Returns the coefficients and their percentile intervals. The cross-products are
    summed per game once, so a resample is a weighted sum of them (no refit loop over
    rows).
    """
    n_games, p = weights.shape[1], x.shape[1]
    xtx = np.zeros((n_games, p, p))
    xty = np.zeros((n_games, p))
    for a in range(p):
        xty[:, a] = np.bincount(game, weights=w * x[:, a] * y, minlength=n_games)
        for b in range(a, p):
            xtx[:, a, b] = np.bincount(game, weights=w * x[:, a] * x[:, b], minlength=n_games)
            xtx[:, b, a] = xtx[:, a, b]
    ridge = 1e-9 * np.eye(p)
    coef = np.linalg.solve(xtx.sum(axis=0) + ridge, xty.sum(axis=0))
    a_b = (weights.astype(float) @ xtx.reshape(n_games, -1)).reshape(-1, p, p) + ridge
    b_b = weights.astype(float) @ xty
    boot = np.linalg.solve(a_b, b_b[..., None])[..., 0]
    lo, hi = _interval(boot, level)
    return {"coef": coef, "low": lo, "high": hi, "boot": boot}


def dummies(codes: np.ndarray) -> np.ndarray:
    """Indicator columns for every level of `codes` except the first."""
    levels = np.unique(codes)
    return (codes[:, None] == levels[None, 1:]).astype(float)


# --------------------------------------------------------------------------------------
# Team tendencies from other games
# --------------------------------------------------------------------------------------


def team_game_table(df: pl.DataFrame) -> pl.DataFrame:
    """One row per offense and game over the plays in `df`, with the offense's tendency
    and efficiency measured in its OTHER games of the same season (leave one game out).

    Columns: n_run, n_pass, run_epa, pass_epa, box (means within the game) and
    loo_pass_rate, loo_run_epa, loo_pass_epa (from the other games; null if the team has
    no other game). A game's own plays never enter its own predictors.
    """
    run, pas = pl.col("is_pass") == 0, pl.col("is_pass") == 1
    g = df.group_by("season", "posteam", "game_id", "game_code").agg(
        run.sum().alias("n_run"),
        pas.sum().alias("n_pass"),
        pl.col("epa").filter(run).sum().alias("s_run"),
        pl.col("epa").filter(pas).sum().alias("s_pass"),
        pl.col("defenders_in_box").mean().alias("box"),
        pl.col("defenders_in_box").count().alias("n_box"),
    )
    ts = ["season", "posteam"]
    other_run = pl.col("n_run").sum().over(ts) - pl.col("n_run")
    other_pass = pl.col("n_pass").sum().over(ts) - pl.col("n_pass")
    other = other_run + other_pass

    def safe(num: pl.Expr, den: pl.Expr) -> pl.Expr:
        return pl.when(den > 0).then(num / den).otherwise(None)

    return g.with_columns(
        safe(pl.col("s_run"), pl.col("n_run")).alias("run_epa"),
        safe(pl.col("s_pass"), pl.col("n_pass")).alias("pass_epa"),
        safe(other_pass, other).alias("loo_pass_rate"),
        safe(pl.col("s_run").sum().over(ts) - pl.col("s_run"), other_run).alias("loo_run_epa"),
        safe(pl.col("s_pass").sum().over(ts) - pl.col("s_pass"), other_pass).alias("loo_pass_epa"),
    ).sort("season", "posteam", "game_id")


# --------------------------------------------------------------------------------------
# The hook for the decision model
# --------------------------------------------------------------------------------------


class RunPassOutcomes:
    """Empirical outcome distributions for a called run and a called pass.

    The same count-based tables as `components.PlayModel` (Kaplan-Meier yards gained by
    down, distance bin and field bin, with small cells borrowing from wider groups, and
    a separate turnover share), fitted once on called runs and once on called passes.
    `outcomes` returns the same keys and array shapes as `ComponentModels.play_outcomes`.

    These are what happened when teams chose each play type against the defenses they
    faced. They are not what would happen if a team changed how often it passes.
    """

    def __init__(
        self,
        min_cell_n: int = 300,
        min_turnover_n: int = 50,
        first_season: int | None = None,
        scrambles_as: str = "pass",
    ) -> None:
        self.min_cell_n = min_cell_n
        self.min_turnover_n = min_turnover_n
        self.first_season = first_season
        self.scrambles_as = scrambles_as

    def fit(self, plays: pl.DataFrame) -> RunPassOutcomes:
        """Fit on a play table. A `qb_scramble` column (see `with_extra`) is needed to
        count scrambles as passes; without it they stay runs."""
        df = cm.add_next_state(plays)  # before any filtering: it looks at the next row
        df = classify_calls(cm._season_filter(df, self.first_season), self.scrambles_as)
        self.tables_: dict[str, cm._ContextTable] = {}
        self.n_: dict[str, int] = {}
        counts = {}
        for call in CALLS:
            s = cm._scrimmage(df.filter(pl.col("call") == call), 1, 3)
            self.n_[call] = int(s["down"].size)
            self.tables_[call] = cm._ContextTable(3, self.min_cell_n, self.min_turnover_n).fit(
                s, s["down"] - 1
            )
            counts[call] = self.tables_[call].n
        total = counts["run"] + counts["pass"]
        self.pass_share_ = np.divide(counts["pass"], total, out=np.full(total.shape, np.nan),
                                     where=total > 0)  # fmt: skip
        return self

    def outcomes(self, call: str, down: int, ydstogo: int, yardline_100: int) -> dict[str, Any]:
        """Outcome distribution of one called run or pass on down 1, 2 or 3.

        Keys as in `ComponentModels.play_outcomes`: p_touchdown, p_safety, keep (length
        100), turnover (length 100, the other team's yardline_100, index 0 = returned
        for a touchdown), p_first_down, p_turnover.
        p_touchdown + p_safety + keep.sum() + turnover.sum() = 1.
        """
        if call not in CALLS:
            raise ValueError("call must be 'run' or 'pass'")
        if down not in (1, 2, 3):
            raise ValueError("outcomes covers downs 1 to 3")
        table = self.tables_[call]
        yl, d = int(yardline_100), int(min(max(ydstogo, 1), yardline_100))
        pmf, p_to = table.cell(down - 1, d, yl)
        p_td, p_safety, keep = cm._spread_gains((1.0 - p_to) * pmf, yl)
        return {
            "p_touchdown": p_td,
            "p_safety": p_safety,
            "keep": keep,
            "turnover": p_to * table.turnover_dist(yl),
            "p_first_down": float(keep[: max(yl - d, 0) + 1].sum()) + p_td,
            "p_turnover": p_to,
        }

    def pass_share(self, down: int, ydstogo: int, yardline_100: int) -> float:
        """Share of called plays that were passes in this situation's cell (nan if the
        cell has no plays). Mixing the two distributions with this share gives back the
        "typical play" of `ComponentModels.play_outcomes`, up to cell pooling."""
        j = int(cm._bin(min(max(ydstogo, 1), yardline_100), cm.DIST_EDGES))
        f = int(cm._bin(yardline_100, cm.FIELD_EDGES))
        return float(self.pass_share_[down - 1, j, f])

    def expected_yards(self, call: str, down: int, ydstogo: int, yardline_100: int) -> float:
        """Mean yards gained on plays without a turnover (touchdowns count as the yards
        to the goal line, safeties as the yards back to it)."""
        o = self.outcomes(call, down, ydstogo, yardline_100)
        yl = int(yardline_100)
        kept = float((o["keep"] * (yl - np.arange(100))).sum())
        total = o["p_touchdown"] * yl + o["p_safety"] * (yl - 100) + kept
        return total / max(1.0 - o["p_turnover"], 1e-12)
