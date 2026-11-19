"""M7 coach evaluation and the coach cost ledger.

Four pieces, each usable on toy data (the tests run without the play-by-play):

1. **Honest grading of real fourth downs.** Every real fourth down from 2011 to 2025 is
   valued by the one-step engine (go, field goal and punt win probability) with models
   that never saw its season: development seasons by leave-seasons-out fold models
   (`fold_plan`, `ensure_fold_models`), test seasons by the headline development models.
   The conversion chance uses the play's own pregame spread (`spread_adjusted_go`). A
   decision's cost is the win probability it gave up against the engine's best action
   (`grade`); only decisions whose best action the bootstrap calls confident count toward
   the headline cost, and the total is reported next to it.
2. **Tables with intervals and shrinkage.** Sums per coach, team or season with a
   bootstrap over games (`group_bootstrap`), and an empirical-Bayes shrinkage of noisy
   rates toward the league (`dersimonian_laird`).
3. **Procedure penalties priced in win probability.** False starts and delays of game by
   the offense (`coach_fouls`), excluding delays taken on purpose before a punt or field
   goal, each priced as the win-probability change between the state with the foul and
   the same state without its yardage (`penalty_states`, `price_states`).
4. **The test of the owner's assumption** that those penalties are on the coach:
   year-to-year stability, split halves, coaches who change teams, and a Poisson model
   with crossed random effects for coach, franchise and team-season fitted by the Laplace
   approximation (`fit_poisson_crossed`).

Public API: load_config, Fold, fold_plan, check_fold_plan, ensure_fold_models,
spread_adjusted_go, grade, fold_bootstrap_plan, fold_replicate_values, coach_lookup,
attach_coaches, group_bootstrap, dersimonian_laird, download_penalties,
download_schedules, coach_fouls, penalty_states, price_states, snap_rows,
expected_counts, rate_correlations, split_half, fit_poisson_crossed,
profile_variance. Teaching notes: docs/notes/m7_coaches.md.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from scipy import linalg, optimize, sparse
from scipy.special import gammaln

from fourth_and_rl import splits
from fourth_and_rl.paths import CONFIGS

ACTIONS: tuple[str, str, str] = ("go", "field_goal", "punt")
SNAP_TYPES: tuple[str, ...] = ("run", "pass", "no_play", "qb_kneel", "qb_spike", "punt",
                               "field_goal")  # fmt: skip

PENALTY_COLUMNS: tuple[str, ...] = (
    "game_id", "play_id", "penalty", "penalty_type", "penalty_team", "penalty_yards",
    "penalty_player_name", "play_type_nfl",
)  # fmt: skip
SCHEDULE_COLUMNS: tuple[str, ...] = (
    "game_id", "season", "game_type", "week", "gameday", "home_team", "away_team",
    "home_coach", "away_coach", "roof",
)  # fmt: skip


def load_config(path: Path | None = None) -> dict:
    """Read configs/coaches.yaml."""
    with open(path or CONFIGS / "coaches.yaml") as f:
        return yaml.safe_load(f)


# ======================================================================================
# 1. Which model grades which season
# ======================================================================================


@dataclass(frozen=True)
class Fold:
    """One grading fold: the seasons it grades and the seasons its models are fitted on."""

    name: str
    grade: tuple[int, ...]
    fit: tuple[int, ...]
    headline: bool = False  # True for the test-season fold (graded by the headline models)


def fold_plan(
    dev_seasons: Sequence[int],
    test_seasons: Sequence[int],
    grade_seasons: Sequence[int],
    fold_size: int,
) -> list[Fold]:
    """Leave-seasons-out folds of `fold_size` consecutive development seasons.

    Each development fold grades its own seasons (those in `grade_seasons`) with models
    fitted on every OTHER development season. A last fold grades the test seasons with
    models fitted on all development seasons (the headline models). No fold is fitted on
    a test season, and no season is graded by a model that saw it (`check_fold_plan`).
    """
    if fold_size < 1:
        raise ValueError("fold_size must be at least 1")
    dev = sorted(int(s) for s in dev_seasons)
    test = sorted(int(s) for s in test_seasons)
    grade = {int(s) for s in grade_seasons}
    folds = []
    for i in range(0, len(dev), fold_size):
        block = dev[i : i + fold_size]
        graded = tuple(s for s in block if s in grade)
        if not graded:
            continue
        fit = tuple(s for s in dev if s not in block)
        folds.append(Fold(f"dev_{block[0]}_{block[-1]}", graded, fit))
    graded_test = tuple(s for s in test if s in grade)
    if graded_test:
        folds.append(Fold(f"test_{graded_test[0]}_{graded_test[-1]}", graded_test,
                          tuple(dev), headline=True))  # fmt: skip
    return folds


def check_fold_plan(
    plan: Sequence[Fold], test_seasons: Sequence[int], holdout_seasons: Sequence[int]
) -> None:
    """Raise `splits.LeakageError` if any fold grades a season its models saw, fits on a
    test or holdout season, grades a holdout season, or if a season is graded twice."""
    test, held = set(test_seasons), set(holdout_seasons)
    seen: set[int] = set()
    for f in plan:
        g, fit = set(f.grade), set(f.fit)
        if g & fit:
            raise splits.LeakageError(
                f"{f.name}: grades seasons it was fitted on {sorted(g & fit)}")
        if fit & test:
            raise splits.LeakageError(f"{f.name}: fitted on test seasons {sorted(fit & test)}")
        if (fit | g) & held:
            raise splits.LeakageError(f"{f.name}: touches the final holdout {sorted(held)}")
        if g & seen:
            raise splits.LeakageError(f"{f.name}: seasons graded twice {sorted(g & seen)}")
        seen |= g


def fold_for_season(plan: Sequence[Fold], season: int) -> Fold:
    for f in plan:
        if int(season) in f.grade:
            return f
    raise KeyError(f"no fold grades season {season}")


def _hash(obj: Any) -> str:
    import hashlib

    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode(),
                        usedforsecurity=False).hexdigest()[:16]  # fmt: skip


def fold_key(fold: Fold, fit_rows: pl.DataFrame, wp_config: dict, components_config: dict
             ) -> dict[str, Any]:  # fmt: skip
    """Everything a cached fold model depends on (compared before reuse)."""
    from fourth_and_rl import fingerprint

    return {
        "fold": fold.name,
        "fit_seasons": list(fold.fit),
        "grade_seasons": list(fold.grade),
        "wp_lgbm_hash": _hash(wp_config["lgbm"]),
        "wp_seed": int(wp_config["seed"]),
        "wp_spec": "v2",
        "calibration": "none",
        "components_config_hash": _hash(components_config),
        "data_fingerprint": fingerprint.data_fingerprint(fit_rows),
    }


def ensure_fold_models(
    fold: Fold,
    dev_plays: pl.DataFrame,
    directory: Path,
    wp_config: dict,
    components_config: dict | None = None,
    n_threads: int = 3,
) -> tuple[Any, Any]:
    """The win-probability model (version 2 spec, no calibration) and the play-outcome
    bundle of one development fold, fitted on `fold.fit` only and cached in `directory`.

    The cache is reused only when its key file matches `fold_key` (seasons, configs and a
    fingerprint of the fitted rows). Both fits pass `splits.assert_no_test_seasons`.
    """
    from fourth_and_rl import components as cm
    from fourth_and_rl import wp as wp_mod

    if fold.headline:
        raise ValueError("the test-season fold uses the headline models (decision.load_engine)")
    ccfg = components_config if components_config is not None else cm.load_config()
    fit_rows = dev_plays.filter(pl.col("season").is_in(list(fold.fit)))
    splits.assert_no_test_seasons(fit_rows, f"M7 fold {fold.name}")
    key = fold_key(fold, fit_rows, wp_config, ccfg)
    directory.mkdir(parents=True, exist_ok=True)
    p_wp = directory / f"{fold.name}_wp.joblib"
    p_cm = directory / f"{fold.name}_components.joblib"
    p_key = directory / f"{fold.name}_key.json"
    if p_wp.exists() and p_cm.exists() and p_key.exists():
        try:
            if json.loads(p_key.read_text()) == json.loads(json.dumps(key)):
                return wp_mod.WPModel.load(p_wp), cm.ComponentModels.load(p_cm)
        except (OSError, ValueError):
            pass
    model = wp_mod.WPModel(wp_config["lgbm"], "none", n_threads, wp_mod.V2).fit(
        fit_rows, None, seed=int(wp_config["seed"]),
        what=f"M7 fold model grading {fold.grade[0]} to {fold.grade[-1]}",
    )  # fmt: skip
    comps = cm.ComponentModels(ccfg).fit(fit_rows)
    model.save(p_wp)
    comps.save(p_cm)
    p_key.write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
    return model, comps


# ======================================================================================
# 2. Grading a decision
# ======================================================================================


def engine_ydstogo(ydstogo: np.ndarray, yardline_100: np.ndarray) -> np.ndarray:
    """Distance as the engine uses it: capped at the goal line and at 40 yards."""
    return np.minimum(np.minimum(ydstogo, yardline_100), 40)


def spread_adjusted_go(
    p_convert_spread: np.ndarray, wp_if_convert: np.ndarray, wp_if_fail: np.ndarray,
    go_wp: np.ndarray,
) -> np.ndarray:  # fmt: skip
    """Value of going with the conversion chance at the play's own spread.

    The engine values going as p * WP(if converted) + (1 - p) * WP(if not) with p at an
    even spread. Re-weighting its two conditional values by the spread-adjusted p changes
    only the chance of converting, not what a conversion or a failure is worth. Where a
    conditional value is undefined (p is 0 or 1) the engine's own value is kept.
    """
    p = np.asarray(p_convert_spread, dtype=float)
    a, b = np.asarray(wp_if_convert, dtype=float), np.asarray(wp_if_fail, dtype=float)
    adj = p * a + (1.0 - p) * b
    return np.where(np.isfinite(adj), adj, np.asarray(go_wp, dtype=float))


def grade(table: np.ndarray, actual: Sequence[str | None]) -> dict[str, np.ndarray]:
    """Grade decisions from an (n, 3) array of go, field goal and punt win probability.

    Returns best (index, -1 if no go value), best_wp, margin over the second best,
    actual (index, -1 if no decision), actual_wp (NaN where the chosen action has no
    value, for example a field goal beyond the model's range) and wp_lost = best_wp -
    actual_wp (NaN where either is undefined; never negative).
    """
    from fourth_and_rl.decision import summarize_actions

    t = np.asarray(table, dtype=float)
    best, best_wp, margin, _ = summarize_actions(t)
    code = {a: i for i, a in enumerate(ACTIONS)}
    act = np.array([code.get(a, -1) if a is not None else -1 for a in actual], dtype=np.int64)
    actual_wp = np.full(t.shape[0], np.nan)
    has = act >= 0
    actual_wp[has] = t[np.nonzero(has)[0], act[has]]
    lost = best_wp - actual_wp
    with np.errstate(invalid="ignore"):
        lost = np.where(np.isfinite(lost), np.maximum(lost, 0.0), np.nan)
    return {"best": best, "best_wp": best_wp, "margin": margin, "actual": act,
            "actual_wp": actual_wp, "wp_lost": lost}  # fmt: skip


def value_table(values: dict[str, np.ndarray], p_spread: np.ndarray) -> np.ndarray:
    """(n, 3) go (spread-adjusted), field goal and punt values from engine output."""
    go = spread_adjusted_go(p_spread, values["wp_if_convert"], values["wp_if_fail"],
                            values["go_wp"])  # fmt: skip
    return np.column_stack([go, values["fg_wp"], values["punt_wp"]])


def p_convert_at_spread(components: Any, states: pl.DataFrame) -> np.ndarray:
    """Conversion chance of each fourth down at its own pregame spread (engine capping)."""
    f = lambda c: states[c].cast(pl.Float64).fill_null(np.nan).to_numpy()  # noqa: E731
    yl, togo, spread = f("yardline_100"), f("ydstogo"), f("posteam_spread")
    out = np.full(states.height, np.nan)
    ok = np.isfinite(yl) & np.isfinite(togo) & np.isfinite(spread) & (yl >= 1) & (togo >= 1)
    if ok.any():
        yl_i = np.clip(yl[ok], 1, 99).astype(int)
        d = engine_ydstogo(togo[ok], yl[ok]).astype(int)
        out[ok] = np.asarray(components.p_convert(yl_i, d, spread=spread[ok]), dtype=float)
    return out


# ======================================================================================
# 3. The cheaper bootstrap for development folds
# ======================================================================================


def fold_bootstrap_plan(prepared_fit: pl.DataFrame, seed: int) -> Any:
    """A `bootstrap.BootstrapPlan` that draws whole games from a fold's fit seasons only.

    `prepared_fit` is `bootstrap.prepare_plays` of the fold's fit rows, so the graded
    seasons are not in the table at all. The games are drawn in two groups (the last two
    fit seasons, and the rest), as the M3 bootstrap draws training and calibration games;
    with no calibration map the trees grow on both groups together.
    """
    from fourth_and_rl.bootstrap import BootstrapPlan

    seasons = sorted(splits.seasons_in(prepared_fit))
    if len(seasons) < 3:
        raise ValueError("a fold bootstrap needs at least three fit seasons")
    return BootstrapPlan(prepared_fit, (seasons[0], seasons[-3]), (seasons[-2], seasons[-1]),
                         seed, "none", 1.0)  # fmt: skip


def fold_replicate_values(
    states: pl.DataFrame, directory: Path, replicate: int, engine_config: dict | None,
    cache_tag: str,
) -> np.ndarray:  # fmt: skip
    """(n, 3) spread-adjusted values of `states` under one saved replicate, cached on disk
    next to the replicate under a name tied to the replicate's key file and the states."""
    from fourth_and_rl import bootstrap

    tag = bootstrap.replicate_tag(directory, replicate)
    path = directory / "m7_values" / f"{cache_tag}_rep_{replicate:03d}_{tag}.npy"
    if path.exists():
        return np.load(path)
    engine = bootstrap.load_replicate_engine(directory, replicate, engine_config)
    v = engine.action_values(states)
    out = value_table(v, p_convert_at_spread(engine.components, states)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


# ======================================================================================
# 4. Coaches
# ======================================================================================


def download_schedules(seasons: list[int], path: Path, force: bool = False) -> Path:
    """Cache the nflverse schedule columns (head coaches per game) for `seasons`."""
    import nflreadpy as nfl

    banned = set(seasons) & set(splits.holdout_seasons())
    if banned:
        raise splits.LeakageError(f"schedule download: {sorted(banned)} is the final holdout")
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        nfl.load_schedules(seasons).select(list(SCHEDULE_COLUMNS)).write_parquet(path)
    return path


def coach_lookup(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per (game, team) with that team's head coach for that game.

    Coaches come from each game's own schedule row, so a mid-season firing or an interim
    coach is handled game by game.
    """
    home = schedules.select("game_id", pl.col("home_team").alias("team"),
                            pl.col("home_coach").alias("coach"), pl.lit(1).alias("is_home_s"))
    away = schedules.select("game_id", pl.col("away_team").alias("team"),
                            pl.col("away_coach").alias("coach"), pl.lit(0).alias("is_home_s"))
    return pl.concat([home, away]).unique(["game_id", "team"])  # fmt: skip


def attach_coaches(df: pl.DataFrame, lookup: pl.DataFrame, team_col: str = "posteam"
                   ) -> pl.DataFrame:  # fmt: skip
    """Add `coach`, the head coach of the team with the ball in that game.

    The join is on the game and on which side of it the team is (`is_home`), not on the
    team code: the schedule keeps historical codes (SD, STL, OAK) while the play-by-play
    uses today's (LAC, LA, LV). Raises if any row with a team gets no coach, so a silent
    mismatch cannot drop rows."""
    side = lookup.select("game_id", pl.col("is_home_s").cast(pl.Int8).alias("_home"), "coach")
    out = df.with_columns(pl.col("is_home").cast(pl.Int8).alias("_home")).join(
        side, on=["game_id", "_home"], how="left").drop("_home")  # fmt: skip
    missing = out.filter(pl.col(team_col).is_not_null() & pl.col("coach").is_null())
    if missing.height:
        raise ValueError(f"{missing.height} rows have no coach, e.g. "
                         f"{missing.select('game_id', team_col).head(3).rows()}")  # fmt: skip
    return out


# ======================================================================================
# 5. Intervals and shrinkage
# ======================================================================================


def group_bootstrap(
    group: np.ndarray,
    game: np.ndarray,
    values: dict[str, np.ndarray],
    n_boot: int,
    seed: int,
) -> dict[Any, dict[str, Any]]:
    """Sums of `values` per group, with a bootstrap over each group's games.

    Rows are first summed per (group, game). For each group, its games are drawn with
    replacement `n_boot` times and the per-game sums added up, so a game drawn twice counts
    twice. Returns {group: {"n_games": g, "point": {name: sum}, "boot": {name: (n_boot,)
    array of sums}}}. Ratios (cost per game, per opportunity) are formed from these sums by
    the caller, replicate by replicate.
    """
    rng = np.random.default_rng(seed)
    frame = pl.DataFrame({"g": group, "game": game, **{k: np.asarray(v, dtype=float)
                                                       for k, v in values.items()}})
    per = frame.group_by("g", "game", maintain_order=True).agg(
        [pl.col(k).sum() for k in values]
    ).sort("g", "game")  # fmt: skip
    out: dict[Any, dict[str, Any]] = {}
    for key, sub in per.group_by("g", maintain_order=True):
        k = key[0]
        n = sub.height
        idx = rng.integers(0, n, size=(n_boot, n))
        point, boot = {}, {}
        for name in values:
            arr = sub[name].to_numpy()
            point[name] = float(arr.sum())
            boot[name] = arr[idx].sum(axis=1)
        out[k] = {"n_games": n, "point": point, "boot": boot}
    return out


def ratio_summary(point_num: float, point_den: float, boot_num: np.ndarray,
                  boot_den: np.ndarray, interval: Sequence[float]) -> dict[str, float]:
    """Point ratio and bootstrap interval of num / den (NaN where den is 0)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(boot_den > 0, boot_num / boot_den, np.nan)
    r = r[np.isfinite(r)]
    est = point_num / point_den if point_den > 0 else float("nan")
    if r.size == 0:
        return {"est": est, "lo": float("nan"), "hi": float("nan"), "se": float("nan")}
    lo, hi = np.quantile(r, [interval[0], interval[1]])
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "se": float(r.std(ddof=1))}


def dersimonian_laird(x: np.ndarray, se2: np.ndarray) -> dict[str, Any]:
    """Empirical-Bayes shrinkage of noisy estimates `x` with sampling variances `se2`.

    The spread of the TRUE values, tau^2, is the method-of-moments estimate of
    DerSimonian and Laird: the weighted spread of the estimates beyond what their own
    sampling variances explain. Each estimate is then pulled toward the precision-weighted
    mean by B = tau^2 / (tau^2 + se2):

        shrunk = mean + B * (x - mean),   posterior variance about B * se2

    `reliability` = tau^2 / (tau^2 + mean se2) is the share of the observed spread that is
    signal; 0 means the ranking is all noise.
    """
    x, se2 = np.asarray(x, dtype=float), np.asarray(se2, dtype=float)
    ok = np.isfinite(x) & np.isfinite(se2) & (se2 > 0)
    k = int(ok.sum())
    if k < 2:
        raise ValueError("need at least two estimates with positive variance")
    w = 1.0 / se2[ok]
    mean_w = float(np.sum(w * x[ok]) / w.sum())
    q = float(np.sum(w * (x[ok] - mean_w) ** 2))
    denom = float(w.sum() - np.sum(w**2) / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / denom) if denom > 0 else 0.0
    w_star = 1.0 / (se2[ok] + tau2)
    mean = float(np.sum(w_star * x[ok]) / w_star.sum())
    b = np.full(x.shape, np.nan)
    b[ok] = tau2 / (tau2 + se2[ok])
    shrunk = mean + b * (x - mean)
    return {
        "tau2": tau2, "mean": mean, "q": q, "k": k, "B": b, "shrunk": shrunk,
        "shrunk_sd": np.sqrt(b * se2),
        "reliability": float(tau2 / (tau2 + np.mean(se2[ok]))) if tau2 > 0 else 0.0,
    }  # fmt: skip


# ======================================================================================
# 6. Procedure penalties
# ======================================================================================


def download_penalties(seasons: list[int], out_dir: Path, force: bool = False) -> list[Path]:
    """Cache the nflverse penalty columns, one Parquet file per season. The state table
    does not carry the penalty type, so it is read from the raw files and joined back by
    game and play id. The final holdout seasons are refused."""
    import nflreadpy as nfl

    banned = set(seasons) & set(splits.holdout_seasons())
    if banned:
        raise splits.LeakageError(f"penalty download: {sorted(banned)} is the final holdout")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for season in seasons:
        path = out_dir / f"penalties_{season}.parquet"
        if force or not path.exists():
            raw = nfl.load_pbp(season).select(list(PENALTY_COLUMNS))
            raw.with_columns(pl.col("play_id").cast(pl.Float64)).write_parquet(path)
        paths.append(path)
    return paths


def next_snap(plays: pl.DataFrame) -> pl.DataFrame:
    """Add `next_snap_type` and `next_snap_posteam`: the play type and team of the next
    snap in the same game (kickoffs, timeouts and extra points are not snaps)."""
    df = plays.with_row_index("_i").sort("game_id", "order_sequence", "play_id")
    snap = df["play_type"].is_in(list(SNAP_TYPES)) & df["down"].is_not_null()
    s = df.filter(snap).select(
        "_i", "game_id",
        pl.col("play_type").shift(-1).over("game_id").alias("next_snap_type"),
        pl.col("posteam").shift(-1).over("game_id").alias("next_snap_posteam"),
    )  # fmt: skip
    out = df.join(s.drop("game_id"), on="_i", how="left").sort("_i").drop("_i")
    return out


def coach_fouls(plays: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Label the procedure fouls of the owner's rule on rows that carry penalty columns.

    Adds:
      foul_type      "false_start", "delay_of_game" or null (offense fouls only, enforced
                     before the snap: play type "no_play", yards > 0, a down)
      deliberate     True for a delay of game the offense took on purpose: lined up in
                     punt or field-goal formation (`cfg["deliberate_formations"]` in the
                     description), or on fourth down with the offense's next snap a punt
                     or field-goal attempt
      coach_foul     foul_type is set and the foul is not deliberate
    Encroachment, defensive offside and neutral-zone infractions are defensive fouls and
    are never labelled (`cfg["not_coach_types"]`, recorded for the record).
    """
    need = {"penalty_type", "penalty_team", "penalty_yards", "posteam", "play_type", "down",
            "desc"}  # fmt: skip
    missing = need - set(plays.columns)
    if missing:
        raise ValueError(f"coach_fouls needs columns {sorted(missing)}")
    df = plays if "next_snap_type" in plays.columns else next_snap(plays)
    names = {"False Start": "false_start", "Delay of Game": "delay_of_game"}
    kinds = [t for t in cfg["coach_types"] if t in names]
    offense = (
        (pl.col("penalty_team") == pl.col("posteam"))
        & pl.col("penalty_type").is_in(kinds)
        & (pl.col("play_type") == "no_play")
        & (pl.col("penalty_yards").fill_null(0) > 0)
        & pl.col("down").is_not_null()
    ).fill_null(False)
    foul = pl.when(offense).then(pl.col("penalty_type").replace_strict(names, default=None))
    formation = pl.lit(False)
    for text in cfg["deliberate_formations"]:
        formation = formation | pl.col("desc").fill_null("").str.contains(text, literal=True)
    before_kick = (
        (pl.col("down") == 4)
        & pl.col("next_snap_type").is_in(["punt", "field_goal"])
        & (pl.col("next_snap_posteam") == pl.col("posteam"))
    ).fill_null(False)
    df = df.with_columns(foul.otherwise(None).alias("foul_type"))
    df = df.with_columns(
        ((pl.col("foul_type") == "delay_of_game") & (formation | before_kick))
        .fill_null(False).alias("deliberate"),
    )  # fmt: skip
    return df.with_columns(
        (pl.col("foul_type").is_not_null() & ~pl.col("deliberate")).alias("coach_foul")
    )


def penalty_states(rows: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The state without the foul and the state with it, for pre-snap offense fouls.

    Without: the row's own state (the snap that would have happened). With: the same down,
    clock, score and timeouts, the ball `penalty_yards` further from the end zone and the
    distance longer by the same yards (in goal-to-go, the distance is the yard line).
    The nflverse yardage already applies the half-the-distance rule. Yard lines are capped
    at 99 (a foul at the 1 cannot put the ball behind the goal line).
    """
    from fourth_and_rl import wp as wp_mod

    yds = rows["penalty_yards"].cast(pl.Float64).fill_null(0.0)
    yl = rows["yardline_100"].cast(pl.Float64)
    togo = rows["ydstogo"].cast(pl.Float64)
    new_yl = (yl + yds).clip(upper_bound=99.0)
    moved = new_yl - yl
    goal = togo >= yl
    new_togo = pl.Series(np.where(goal.to_numpy(), new_yl.to_numpy(),
                                  (togo + moved).to_numpy()))  # fmt: skip
    without = rows.select(wp_mod.STATE_COLUMNS).cast(pl.Float64)
    with_ = without.with_columns(new_yl.alias("yardline_100"), new_togo.alias("ydstogo"))
    return with_, without


def price_states(model: Any, with_: pl.DataFrame, without: pl.DataFrame) -> np.ndarray:
    """Win probability lost by the offense: WP(without the foul) - WP(with it)."""
    return np.asarray(model.predict(without), dtype=float) - np.asarray(model.predict(with_),
                                                                        dtype=float)


# ======================================================================================
# 7. The assumption test
# ======================================================================================


def snap_rows(plays: pl.DataFrame, long_distance: int) -> pl.DataFrame:
    """Offensive snap opportunities (scrimmage plays and pre-snap no-plays with a down) with
    the situation cells the assumption test controls for."""
    dome = pl.col("roof").is_in(["dome", "closed"]).fill_null(False)
    return plays.filter(
        pl.col("down").is_not_null() & pl.col("play_type").is_in(list(SNAP_TYPES))
        & (pl.col("qtr") <= 4)
    ).with_columns(
        ((pl.col("down") >= 3) & (pl.col("ydstogo") >= long_distance))
        .fill_null(False).cast(pl.Int8).alias("long"),
        dome.cast(pl.Int8).alias("dome"),
    )  # fmt: skip


def expected_counts(snaps: pl.DataFrame, flag: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """League rate of `flag` per snap in each cell (season, home or road, dome, long) and
    each snap's expected count. Returns (snaps with `expected`, the cell table)."""
    cells = ["season", "is_home", "dome", "long"]
    table = snaps.group_by(cells).agg(
        pl.col(flag).cast(pl.Float64).mean().alias("rate"), pl.len().alias("snaps"),
        pl.col(flag).cast(pl.Float64).sum().alias("fouls"),
    ).sort(cells)  # fmt: skip
    return snaps.join(table.select(*cells, "rate"), on=cells, how="left").rename(
        {"rate": "expected"}
    ), table


def rate_ratios(table: pl.DataFrame) -> dict[str, float]:
    """Pooled rate ratios of the cell table (all seasons): road over home, in domes and
    outdoors, and third or fourth and long over other snaps."""

    def rate(mask: pl.Expr) -> float:
        sub = table.filter(mask)
        return float(sub["fouls"].sum() / max(sub["snaps"].sum(), 1))

    road, home = pl.col("is_home") == 0, pl.col("is_home") == 1
    dome, out = pl.col("dome") == 1, pl.col("dome") == 0
    return {
        "road_over_home": rate(road) / rate(home),
        "road_over_home_in_domes": rate(road & dome) / rate(home & dome),
        "road_over_home_outdoors": rate(road & out) / rate(home & out),
        "long_over_other": rate(pl.col("long") == 1) / rate(pl.col("long") == 0),
        "rate_per_100_snaps": 100 * rate(pl.lit(True)),
    }


def correlation(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0
                ) -> dict[str, float]:  # fmt: skip
    """Pearson correlation with a bootstrap interval over the pairs (90%)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 4:
        return {"r": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": n}
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    xs, ys = x[idx], y[idx]
    xs = xs - xs.mean(axis=1, keepdims=True)
    ys = ys - ys.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = (xs * ys).sum(1) / np.sqrt((xs**2).sum(1) * (ys**2).sum(1))
    lo, hi = np.nanquantile(rb, [0.05, 0.95])
    return {"r": r, "lo": float(lo), "hi": float(hi), "n": int(n)}


def rate_correlations(units: pl.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    """Year-to-year correlation of a season-adjusted rate, split by what stayed the same.

    `units` has one row per team-season with columns team, season, coach (the coach of
    most of its games), fouls and expected. The rate is log((fouls + 0.5) / (expected +
    0.5)), so the league level of each season is removed. Pairs of consecutive seasons of
    the same team are split three ways:
      same_coach      the same coach both seasons
      new_coach       the team changed coach (does the rate stay with the team?)
    and coach moves (the same coach, consecutive seasons, a different team):
      coach_moved     does the rate follow the coach?
    """
    u = units.with_columns(
        ((pl.col("fouls") + 0.5) / (pl.col("expected") + 0.5)).log().alias("lr")
    )
    nxt = u.select(pl.col("team"), (pl.col("season") - 1).alias("season"),
                   pl.col("coach").alias("coach_next"), pl.col("lr").alias("lr_next"))
    team_pairs = u.join(nxt, on=["team", "season"], how="inner")
    same = team_pairs.filter(pl.col("coach") == pl.col("coach_next"))
    new = team_pairs.filter(pl.col("coach") != pl.col("coach_next"))
    c_next = u.select(pl.col("coach"), (pl.col("season") - 1).alias("season"),
                      pl.col("team").alias("team_next"), pl.col("lr").alias("lr_next"))
    moved = u.join(c_next, on=["coach", "season"], how="inner").filter(
        pl.col("team") != pl.col("team_next"))  # fmt: skip
    out = {}
    for name, d in (("same_coach_same_team", same), ("same_team_new_coach", new),
                    ("coach_moved_to_new_team", moved)):  # fmt: skip
        out[name] = correlation(d["lr"].to_numpy(), d["lr_next"].to_numpy(), n_boot, seed)
    return out


def stints(units: pl.DataFrame) -> pl.DataFrame:
    """Coach-team stints: runs of consecutive team-seasons with the same coach, pooled.

    `units` is one row per team-season (team, season, coach, fouls, expected). Returns one
    row per stint with first and last season, fouls, expected and the season-adjusted log
    rate `lr`."""
    u = units.sort("team", "season").with_columns(
        ((pl.col("coach") != pl.col("coach").shift(1).over("team"))
         | ((pl.col("season") - pl.col("season").shift(1).over("team")) != 1))
        .fill_null(True).cum_sum().over("team").alias("_stint"),
    )  # fmt: skip
    s = u.group_by("team", "_stint").agg(
        pl.col("coach").first(), pl.col("season").min().alias("first"),
        pl.col("season").max().alias("last"), pl.col("fouls").sum(), pl.col("expected").sum(),
    ).drop("_stint")  # fmt: skip
    return s.with_columns(
        ((pl.col("fouls") + 0.5) / (pl.col("expected") + 0.5)).log().alias("lr")
    ).sort("coach", "first")


def mover_test(units: pl.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    """For every coach who moves from one team to another (consecutive stints of the same
    coach at different teams, any gap allowed), compare two correlations over the same
    moves:

      follows_coach    his rate at the old team against his rate at the new team
      stays_with_team  the new team's rate in the stint just before he arrived (another
                       coach) against his rate there

    If the rate belongs to the coach, the first should be clearly larger."""
    s = stints(units)
    moves = s.with_columns(
        pl.col("team").shift(1).over("coach").alias("old_team"),
        pl.col("lr").shift(1).over("coach").alias("lr_old_team"),
    ).filter(pl.col("old_team").is_not_null() & (pl.col("old_team") != pl.col("team")))
    before = s.select(pl.col("team"), pl.col("last").alias("prev_last"),
                      pl.col("coach").alias("prev_coach"), pl.col("lr").alias("lr_team_before"))
    moves = moves.sort("first").join_asof(
        before.sort("prev_last"), left_on="first", right_on="prev_last", by="team",
        strategy="backward", allow_exact_matches=False, check_sortedness=False,
    ).filter(pl.col("prev_coach").is_not_null() & (pl.col("prev_coach") != pl.col("coach")))
    both = moves.drop_nulls(["lr_old_team", "lr_team_before"])
    return {
        "moves": both.height,
        "coaches": sorted(both["coach"].unique().to_list()),
        "follows_coach": correlation(both["lr_old_team"].to_numpy(), both["lr"].to_numpy(),
                                     n_boot, seed),
        "stays_with_team": correlation(both["lr_team_before"].to_numpy(), both["lr"].to_numpy(),
                                       n_boot, seed),
    }  # fmt: skip


def split_half(group: np.ndarray, order: np.ndarray, fouls: np.ndarray, expected: np.ndarray,
               n_boot: int, seed: int) -> dict[str, float]:  # fmt: skip
    """Split-half reliability: each group's games alternately into two halves (by `order`),
    the season-adjusted rate of each half, their correlation across groups, and the
    Spearman-Brown reliability of the full-length rate, 2r / (1 + r)."""
    df = pl.DataFrame({"g": group, "o": order, "f": fouls, "e": expected}).sort("g", "o")
    df = df.with_columns((pl.col("o").rank("ordinal").over("g") % 2).alias("half"))
    h = df.group_by("g", "half").agg(pl.col("f").sum(), pl.col("e").sum())
    h = h.with_columns(((pl.col("f") + 0.5) / (pl.col("e") + 0.5)).log().alias("lr"))
    w = h.pivot(on="half", index="g", values="lr").drop_nulls()
    c = correlation(w["0"].to_numpy(), w["1"].to_numpy(), n_boot, seed)
    r = c["r"]
    c["spearman_brown"] = float(2 * r / (1 + r)) if np.isfinite(r) and r > -1 else float("nan")
    return c


def _design(groups: dict[str, np.ndarray]) -> tuple[sparse.csr_matrix, np.ndarray, list[str]]:
    """Sparse indicator matrix of the random effects, the group of each column, and names."""
    blocks, owner, names = [], [], []
    for k, (name, codes) in enumerate(groups.items()):
        codes = np.asarray(codes, dtype=np.int64)
        n_lev = int(codes.max()) + 1
        rows = np.arange(codes.size)
        blocks.append(sparse.csr_matrix((np.ones(codes.size), (rows, codes)),
                                        shape=(codes.size, n_lev)))  # fmt: skip
        owner.append(np.full(n_lev, k))
        names.append(name)
    return sparse.hstack(blocks, format="csr"), np.concatenate(owner), names


class _Laplace:
    """Mode and Laplace log marginal likelihood of a Poisson model with Gaussian random
    effects, for given log variances. Kept between calls so the mode is warm-started."""

    def __init__(self, y: np.ndarray, log_offset: np.ndarray, z: sparse.csr_matrix,
                 owner: np.ndarray) -> None:  # fmt: skip
        self.y, self.off, self.z, self.owner = y, log_offset, z, owner
        self.q = z.shape[1]
        self.u = np.zeros(self.q)
        self.b0 = float(np.log(max(y.sum(), 1e-9) / np.exp(log_offset).sum()))
        self.const = float(gammaln(y + 1).sum())

    def mode(self, log_var: np.ndarray, tol: float = 1e-8, max_iter: int = 100
             ) -> tuple[float, np.ndarray]:  # fmt: skip
        prec = np.exp(-np.asarray(log_var))[self.owner]
        u, b0 = self.u.copy(), self.b0
        zt = self.z.T.tocsr()
        for _ in range(max_iter):
            eta = self.off + b0 + self.z @ u
            mu = np.exp(eta)
            r = self.y - mu
            g = np.concatenate([[r.sum()], zt @ r - prec * u])
            h_uu = (zt @ sparse.diags(mu) @ self.z).toarray() + np.diag(prec)
            h_bu = zt @ mu
            h = np.empty((self.q + 1, self.q + 1))
            h[0, 0], h[0, 1:], h[1:, 0], h[1:, 1:] = mu.sum(), h_bu, h_bu, h_uu
            step = linalg.solve(h, g, assume_a="pos")
            b0 += step[0]
            u += step[1:]
            if np.max(np.abs(step)) < tol:
                break
        self.u, self.b0 = u, b0
        eta = self.off + b0 + self.z @ u
        mu = np.exp(eta)
        h_uu = (zt @ sparse.diags(mu) @ self.z).toarray() + np.diag(prec)
        logdet = 2.0 * np.sum(np.log(np.diag(linalg.cholesky(h_uu, lower=True))))
        loglik = float(np.sum(self.y * eta - mu) - self.const)
        lv = np.asarray(log_var)[self.owner]
        logml = loglik - 0.5 * np.sum(prec * u**2) - 0.5 * np.sum(lv) - 0.5 * logdet
        return float(logml), u.copy()


def fit_poisson_crossed(
    y: np.ndarray,
    expected: np.ndarray,
    groups: dict[str, np.ndarray],
    start_log_var: float = -3.0,
    fixed: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Poisson model with crossed random effects, fitted by the Laplace approximation.

        y_i ~ Poisson(expected_i * exp(b0 + sum over groups k of u_k[group_k(i)]))
        u_k[j] ~ Normal(0, var_k), independent

    `expected` carries the fixed part (league rate by season and situation times snaps).
    The variances are chosen by maximizing the Laplace approximation of the marginal
    likelihood (the method of lme4's glmer with its default settings), with the random
    effects and b0 at their joint mode for each set of variances. `fixed` holds log
    variances that are not estimated (for profiling). Returns the variances, the
    effects of each group (conditional modes, i.e. shrunk estimates) and the log
    marginal likelihood.
    """
    y = np.asarray(y, dtype=float)
    off = np.log(np.asarray(expected, dtype=float))
    z, owner, names = _design(groups)
    lap = _Laplace(y, off, z, owner)
    fixed = dict(fixed or {})
    free = [n for n in names if n not in fixed]

    def full(theta: np.ndarray) -> np.ndarray:
        it = iter(theta)
        return np.array([fixed[n] if n in fixed else next(it) for n in names])

    def neg(theta: np.ndarray) -> float:
        return -lap.mode(full(np.clip(theta, -14.0, 3.0)))[0]

    if free:
        res = optimize.minimize(neg, np.full(len(free), start_log_var), method="Nelder-Mead",
                                options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 600})
        theta = np.clip(res.x, -14.0, 3.0)
    else:
        theta = np.zeros(0)
    log_var = full(theta)
    logml, u = lap.mode(log_var)
    effects, start = {}, 0
    for k, n in enumerate(names):
        size = int(np.sum(owner == k))
        effects[n] = u[start : start + size]
        start += size
    return {"log_var": dict(zip(names, log_var.tolist(), strict=True)),
            "var": dict(zip(names, np.exp(log_var).tolist(), strict=True)),
            "sd": dict(zip(names, np.exp(log_var / 2).tolist(), strict=True)),
            "intercept": lap.b0, "logml": logml, "effects": effects}  # fmt: skip


def profile_variance(
    y: np.ndarray, expected: np.ndarray, groups: dict[str, np.ndarray], name: str,
    grid: Sequence[float], best_logml: float,
) -> dict[str, Any]:  # fmt: skip
    """Profile likelihood of one group's standard deviation: refit with that log variance
    fixed at each grid value and the others free. The 95% interval is the set of values
    whose log likelihood is within 1.92 of the best (chi-square with one degree of
    freedom); a lower end at the grid's bottom means the data cannot rule out zero."""
    prof = []
    for lv in grid:
        fit = fit_poisson_crossed(y, expected, groups, fixed={name: float(lv)})
        prof.append(fit["logml"])
    prof_arr = np.array(prof)
    ok = prof_arr >= best_logml - 1.92
    sds = np.exp(np.asarray(grid, dtype=float) / 2)
    return {
        "grid_sd": sds.tolist(), "logml": prof_arr.tolist(),
        "sd_lo": float(sds[ok].min()) if ok.any() else float("nan"),
        "sd_hi": float(sds[ok].max()) if ok.any() else float("nan"),
        "lower_end_at_grid_bottom": bool(ok[0]),
    }  # fmt: skip
