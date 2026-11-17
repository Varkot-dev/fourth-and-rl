"""M8: team strength beyond the point spread ("does the model know our defense is terrible?").

Three parts, each usable on its own.

1. Ratings without look-ahead. For every team, before every game, an offense rating and a
   defense rating on two things the fourth-down decision depends on:

   * short-yardage conversion: third and fourth downs with 1 to 3 to go. A play's residual
     is converted (1 or 0) minus the league conversion curve for its down, distance and
     field position (the M1 conversion model at an even spread). Units: probability.
   * points per drive: offense points on the drive minus the league average for a drive
     starting at that yard line. Units: points.

   The defense rating is what the defense ALLOWS above average, so a bad defense has a
   positive rating. Ratings come from a Kalman filter (`run_filter`): a team's rating is a
   hidden quantity, each game gives one noisy look at "offense rating + opposing defense
   rating", and the filter weighs each look by how noisy it is. Most of the variation in a
   few games is luck, so the filter shrinks hard; its constants (true-effect spread tau,
   season carry-over rho, weekly drift) are tuned on development seasons with
   season-blocked validation. A rating for week w uses only games of earlier weeks, and the
   league curves the residuals are measured against are fitted on earlier seasons only.

2. Effect models. Logistic models of conversion and of drive outcome (touchdown, field
   goal, nothing) with the pregame spread, and the same models with the ratings added. The
   question "does the team rating add anything beyond the spread?" is the held-out log-loss
   difference between the two.

3. The team-aware decision engine (`TeamAwareEngine`, an extension of
   `decision.DecisionEngine` that leaves the original untouched). The conversion
   probability moves by the team model's change in log-odds, and every next state in which
   a team starts with the ball gets the change in that team's next-drive scoring chances,
   valued through the win-probability model. With all ratings at zero it reproduces the
   spread-only engine exactly.

Teaching notes: docs/notes/m8_team_strength.md.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

from fourth_and_rl import components as cm
from fourth_and_rl import decision, splits
from fourth_and_rl import wp as wp_mod
from fourth_and_rl.paths import CONFIGS

# The four rating features of an offense facing a defense, in model units: conversion
# ratings in percentage points of conversion probability, drive ratings in points.
RATING_FEATURES: tuple[str, ...] = ("off_conv", "def_conv", "off_drive", "def_drive")
METRICS: tuple[str, ...] = ("conv", "drive")
DRIVE_CLASSES: tuple[str, ...] = ("none", "fg", "td")  # class index 0, 1, 2
DRIVE_CLASS_POINTS = {"none": 0, "fg": 3, "td": 7}

# These mirror decision.py's private constants (the tests check they agree).
OWNER_US, OWNER_OPP = 0, 1
N_SLOTS = 5  # go converts, go fails, field goal made, field goal missed, punt
SLOT_CONVERT, SLOT_FAIL, SLOT_FG_MAKE, SLOT_FG_MISS, SLOT_PUNT = range(N_SLOTS)


def load_config(path: Path | None = None) -> dict:
    """Read configs/team_strength.yaml."""
    with open(path or CONFIGS / "team_strength.yaml") as f:
        return yaml.safe_load(f)


def _guard(df: pl.DataFrame, what: str) -> None:
    if "season" in df.columns and df.height:
        splits.assert_no_test_seasons(df, what)


def _f(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].cast(pl.Float64).fill_null(np.nan).to_numpy()


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ======================================================================================
# 1. Samples
# ======================================================================================


def schedule(plays: pl.DataFrame) -> pl.DataFrame:
    """One row per game: game_id, season, week, home_team, away_team."""
    return (
        plays.group_by("game_id")
        .agg(
            pl.col("season").first(),
            pl.col("week").first(),
            pl.col("home_team").first(),
            pl.col("away_team").first(),
        )  # fmt: skip
        .sort("season", "week", "game_id")
    )


def conversion_sample(plays: pl.DataFrame, config: dict | None = None) -> pl.DataFrame:
    """Short-yardage runs and passes, with `converted` (first down or touchdown, keeping
    the ball). Two-point tries are excluded."""
    s = (config or load_config())["samples"]
    off_td = (pl.col("touchdown") == 1) & (pl.col("td_team") == pl.col("posteam"))
    turnover = (pl.col("interception") == 1) | (pl.col("fumble_lost") == 1)
    df = plays.filter(
        pl.col("play_type").is_in(["run", "pass"])
        & pl.col("down").cast(pl.Int64).is_in([int(x) for x in s["conversion_downs"]])
        & pl.col("ydstogo").is_between(s["conversion_min_ydstogo"], s["conversion_max_ydstogo"])
        & pl.col("yardline_100").is_between(1, 99)
        & pl.col("yards_gained").is_not_null()
        & (pl.col("two_point_attempt").fill_null(0) != 1)
        & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
    )
    converted = (((pl.col("first_down") == 1) | off_td) & ~turnover).fill_null(False)
    return df.select(
        "game_id", "season", "week", "posteam", "defteam",
        pl.col("down").cast(pl.Int64), pl.col("ydstogo").cast(pl.Int64),
        pl.col("yardline_100").cast(pl.Int64),
        pl.col("posteam_spread").cast(pl.Float64).fill_null(0.0),
        converted.cast(pl.Int64).alias("converted"),
    )  # fmt: skip


def drive_sample(plays: pl.DataFrame, config: dict | None = None) -> pl.DataFrame:
    """One row per drive that starts in regulation with more than
    `samples.drive_min_half_seconds` left in the half.

    Columns: game_id, season, week, posteam, defteam, start (yardline_100 of the first
    snap), posteam_spread, result, outcome (none / fg / td), points (offense points by the
    result: touchdown 7, field goal 3, returned for a touchdown -7, safety -2, else 0),
    seconds (game clock used), next_start (the other team's yardline_100 at its next drive's
    first snap, null if the next drive is not theirs).
    """
    s = (config or load_config())["samples"]
    snap = (
        pl.col("down").is_not_null()
        & pl.col("yardline_100").is_not_null()
        & pl.col("posteam").is_not_null()
    )
    d = (
        plays.filter(pl.col("fixed_drive").is_not_null())
        .sort("game_id", "order_sequence", "play_id")
        .group_by("game_id", "fixed_drive", maintain_order=True)
        .agg(
            pl.col("season").first(), pl.col("week").first(),
            pl.col("posteam").filter(snap).first(), pl.col("defteam").filter(snap).first(),
            pl.col("yardline_100").filter(snap).first().alias("start"),
            pl.col("half_seconds_remaining").filter(snap).first().alias("start_half_seconds"),
            pl.col("game_seconds_remaining").filter(snap).first().alias("start_seconds"),
            pl.col("game_seconds_remaining").last().alias("end_seconds"),
            pl.col("qtr").filter(snap).first().alias("qtr"),
            pl.col("game_half").filter(snap).first().alias("half"),
            pl.col("posteam_spread").filter(snap).first().alias("posteam_spread"),
            pl.col("fixed_drive_result").drop_nulls().first().alias("result"),
        )
        .sort("game_id", "fixed_drive")
        .with_columns(
            pl.col("start").shift(-1).over("game_id").alias("_next_start"),
            pl.col("posteam").shift(-1).over("game_id").alias("_next_team"),
            pl.col("half").shift(-1).over("game_id").alias("_next_half"),
        )
    )  # fmt: skip
    points = {k: int(v) for k, v in s["drive_points"].items()}
    d = d.filter(
        pl.col("start").is_not_null() & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null() & pl.col("result").is_not_null()
        & (pl.col("qtr") <= 4) & (pl.col("start_half_seconds") > s["drive_min_half_seconds"])
    ).with_columns(
        pl.col("start").cast(pl.Int64),
        pl.col("posteam_spread").cast(pl.Float64).fill_null(0.0),
        pl.col("result").replace_strict(points, default=0, return_dtype=pl.Int64).alias("points"),
        pl.when(pl.col("result") == "Touchdown").then(pl.lit("td"))
        .when(pl.col("result") == "Field goal").then(pl.lit("fg"))
        .otherwise(pl.lit("none")).alias("outcome"),
        (pl.col("start_seconds") - pl.col("end_seconds")).clip(0, None).alias("seconds"),
        pl.when((pl.col("_next_team") == pl.col("defteam"))
                & (pl.col("_next_half") == pl.col("half")))
        .then(pl.col("_next_start")).otherwise(None).alias("next_start"),
    )  # fmt: skip
    return d.select(
        "game_id", "season", "week", "posteam", "defteam", "start", "posteam_spread",
        "result", "outcome", "points", "seconds", "next_start",
    )  # fmt: skip


# ======================================================================================
# 2. League curves the residuals are measured against (fitted on earlier seasons only)
# ======================================================================================


def base_fit_seasons(season: int, first_season: int, last_dev_season: int) -> list[int]:
    """Seasons the base curve used for `season` is fitted on: every season from the first
    up to `season - 1`, but never past the last development season (a later season uses
    the base fitted on all development seasons)."""
    hi = min(int(season) - 1, int(last_dev_season))
    return list(range(int(first_season), hi + 1))


@dataclass
class DrivePointsBase:
    """Expected offense points of a drive by start yard line: a polynomial in y / 100."""

    degree: int = 3
    coef_: np.ndarray | None = None
    resid_var_: float = float("nan")
    seasons_: list[int] = field(default_factory=list)

    def fit(self, drives: pl.DataFrame) -> DrivePointsBase:
        _guard(drives, "drive points base")
        y = _f(drives, "start") / 100.0
        pts = _f(drives, "points")
        x = np.vander(y, self.degree + 1)
        self.coef_ = np.linalg.lstsq(x, pts, rcond=None)[0]
        self.resid_var_ = float(np.var(pts - x @ self.coef_))
        self.seasons_ = sorted({int(s) for s in drives["season"].unique().to_list()})
        return self

    def expected(self, start: np.ndarray) -> np.ndarray:
        return np.vander(np.asarray(start, dtype=float) / 100.0, self.degree + 1) @ self.coef_


@dataclass
class Bases:
    """Per-season league curves: a conversion model (`components.GoModel`) and a drive
    points curve for each season, each fitted on earlier seasons only."""

    conversion: dict[int, cm.GoModel]
    drive: dict[int, DrivePointsBase]
    fitted_on: dict[int, list[int]]

    def conversion_logit(self, sample: pl.DataFrame) -> np.ndarray:
        """League log-odds of converting for each play of a conversion sample, at an even
        spread, with the conversion model's season trend moved to the play's season."""
        out = np.full(sample.height, np.nan)
        seasons = sample["season"].to_numpy()
        for s in np.unique(seasons):
            m = seasons == s
            go = self.conversion[int(s)]
            sub = sample.filter(pl.Series(m))
            p = np.zeros(sub.height)
            downs = sub["down"].to_numpy()
            for down in np.unique(downs):
                k = downs == down
                p[k] = go.p_convert(sub["yardline_100"].to_numpy()[k],
                                    sub["ydstogo"].to_numpy()[k], down=int(down))  # fmt: skip
            shift = go.coef_.get("season_per_decade", 0.0) * (
                cm._season_t(float(s)) - cm._season_t(go.reference_season_)
            )
            out[m] = _logit(p) + shift
        return out

    def drive_expected(self, drives: pl.DataFrame) -> np.ndarray:
        out = np.full(drives.height, np.nan)
        seasons = drives["season"].to_numpy()
        for s in np.unique(seasons):
            m = seasons == s
            out[m] = self.drive[int(s)].expected(drives["start"].to_numpy()[m])
        return out


def fit_bases(
    dev_plays: pl.DataFrame,
    seasons: Sequence[int],
    config: dict | None = None,
    components_config: dict | None = None,
) -> Bases:
    """Fit the base curves for each of `seasons` from `dev_plays` (development seasons
    only; the guard refuses anything else). Seasons are assigned by `base_fit_seasons`."""
    cfg = config or load_config()
    ccfg = components_config or cm.load_config()
    _guard(dev_plays, "M8 base curves")
    dev_seasons = sorted({int(s) for s in dev_plays["season"].unique().to_list()})
    first, last = dev_seasons[0], dev_seasons[-1]
    plays = cm.add_next_state(dev_plays)
    drives = drive_sample(dev_plays, cfg)
    conv: dict[int, cm.GoModel] = {}
    drv: dict[int, DrivePointsBase] = {}
    fitted: dict[int, list[int]] = {}
    cache: dict[tuple[int, ...], tuple[cm.GoModel, DrivePointsBase]] = {}
    for s in sorted({int(x) for x in seasons}):
        use = base_fit_seasons(s, first, last)
        if not use:
            raise ValueError(f"season {s} has no earlier season to fit its base on")
        key = tuple(use)
        if key not in cache:
            go = cm.GoModel(era_first_season=None, **ccfg.get("go", {})).fit(
                plays.filter(pl.col("season").is_in(use))
            )
            dp = DrivePointsBase(cfg["bases"]["drive_points_degree"]).fit(
                drives.filter(pl.col("season").is_in(use))
            )
            cache[key] = (go, dp)
        conv[s], drv[s] = cache[key]
        fitted[s] = use
    return Bases(conv, drv, fitted)


def add_residuals(
    conv: pl.DataFrame, drives: pl.DataFrame, bases: Bases
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add base predictions, residuals and per-row noise variances.

    Conversion: base_logit, base_p, resid = converted - base_p, noise = base_p (1 - base_p).
    Drives: base_points, resid = points - base_points, noise = the base's residual variance.
    """
    bl = bases.conversion_logit(conv)
    bp = _sigmoid(bl)
    conv = conv.with_columns(
        pl.Series("base_logit", bl), pl.Series("base_p", bp),
        pl.Series("resid", conv["converted"].to_numpy() - bp), pl.Series("noise", bp * (1 - bp)),
    )  # fmt: skip
    be = bases.drive_expected(drives)
    var = np.array([bases.drive[int(s)].resid_var_ for s in drives["season"].to_numpy()])
    drives = drives.with_columns(
        pl.Series("base_points", be), pl.Series("resid", drives["points"].to_numpy() - be),
        pl.Series("noise", var),
    )  # fmt: skip
    return conv, drives


# ======================================================================================
# 3. The rating filter
# ======================================================================================


@dataclass(frozen=True)
class FilterParams:
    """Constants of the rating filter (see configs/team_strength.yaml, `filter`).

    tau_off, tau_def, rho and q_frac are tuned. league_sd is fixed: the prior spread of a
    league-wide level (scoring drifts from season to season, so a residual measured against
    last season's curve has a small common offset that must not be credited to teams)."""

    tau_off: float
    tau_def: float
    rho: float
    q_frac: float
    league_sd: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"tau_off": self.tau_off, "tau_def": self.tau_def, "rho": self.rho,
                "q_frac": self.q_frac, "league_sd": self.league_sd}  # fmt: skip


@dataclass
class Observations:
    """One row per (game, offense): mean residual y, its noise variance r, play count n."""

    season: np.ndarray
    week: np.ndarray
    game: np.ndarray  # index into the schedule
    off: np.ndarray  # team index of the offense
    deff: np.ndarray  # team index of the defense
    y: np.ndarray
    r: np.ndarray
    n: np.ndarray
    row_obs: np.ndarray  # for each row of the sample, the index of its observation


@dataclass
class Schedule:
    teams: list[str]
    game_ids: list[str]
    season: np.ndarray
    week: np.ndarray
    home: np.ndarray
    away: np.ndarray

    @classmethod
    def from_frame(cls, sched: pl.DataFrame, teams: Sequence[str] | None = None) -> Schedule:
        sched = sched.sort("season", "week", "game_id")
        names = sorted(teams) if teams is not None else sorted(
            set(sched["home_team"].to_list()) | set(sched["away_team"].to_list()))  # fmt: skip
        idx = {t: i for i, t in enumerate(names)}
        return cls(
            teams=names, game_ids=sched["game_id"].to_list(),
            season=sched["season"].to_numpy().astype(int),
            week=sched["week"].to_numpy().astype(int),
            home=np.array([idx[t] for t in sched["home_team"].to_list()]),
            away=np.array([idx[t] for t in sched["away_team"].to_list()]),
        )  # fmt: skip

    def index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}


def observations(sample: pl.DataFrame, sched: Schedule) -> Observations:
    """Group a residual sample (conversion plays or drives) into one observation per
    (game, offense): y = mean residual, r = sum of row noise / n^2."""
    tidx = sched.index()
    gidx = {g: i for i, g in enumerate(sched.game_ids)}
    df = sample.with_row_index("_row")
    grouped = (
        df.group_by("game_id", "posteam", "defteam", maintain_order=True).agg(
            pl.col("season").first(),
            pl.col("week").first(),
            pl.col("resid").sum(),
            pl.col("noise").sum(),
            pl.len().alias("n"),
            pl.col("_row"),
        )  # fmt: skip
    )
    row_obs = np.empty(df.height, dtype=np.int64)
    for i, rows in enumerate(grouped["_row"].to_list()):
        row_obs[np.asarray(rows)] = i
    n = grouped["n"].to_numpy().astype(float)
    return Observations(
        season=grouped["season"].to_numpy().astype(int),
        week=grouped["week"].to_numpy().astype(int),
        game=np.array([gidx[g] for g in grouped["game_id"].to_list()]),
        off=np.array([tidx[t] for t in grouped["posteam"].to_list()]),
        deff=np.array([tidx[t] for t in grouped["defteam"].to_list()]),
        y=grouped["resid"].to_numpy() / n, r=grouped["noise"].to_numpy() / n**2, n=n,
        row_obs=row_obs,
    )  # fmt: skip


@dataclass
class FilterResult:
    """What the filter knew BEFORE each game, and at each season's end.

    pre[g, side] = (offense mean, offense sd, defense mean, defense sd) of the home
    (side 0) and away (side 1) team before game g. obs_mean and obs_var are the predicted
    mean and variance of each observation before its game. season_end[s] = (mean vector of
    length 2 * teams, sd vector), after the last game of season s (before the season break).
    """

    teams: list[str]
    game_ids: list[str]
    pre: np.ndarray
    obs_mean: np.ndarray
    obs_var: np.ndarray
    season_end: dict[int, tuple[np.ndarray, np.ndarray]]


def run_filter(
    obs: Observations, sched: Schedule, params: FilterParams, start_season: int | None = None
) -> FilterResult:
    """Kalman filter of offense and defense ratings, one week at a time.

    State x: 32 offense ratings, 32 defense ratings and one league level L, all starting at
    0 (variances tau_off^2, tau_def^2, league_sd^2). For a week: (1) every game of the week
    reads the state as it is, so a rating never contains its own game or a later one;
    (2) each observation y = x[off] + x[32 + def] + L + noise (variance r) updates the state
    by the usual Kalman step, with h the sum of the three columns of P involved:

        S = h[o] + h[d] + h[L] + r              (predicted variance of y)
        K = h / S                               (how much each rating moves per unit surprise)
        x <- x + K (y - x[o] - x[d] - L),   P <- P - K h'

    Between weeks the team variances grow by q_frac * tau^2 (ratings may drift). At a new
    season the team part shrinks, x <- rho x and P <- rho^2 P + (1 - rho^2) tau^2, and the
    league level keeps its value but gains variance league_sd^2. Seasons before
    `start_season` are skipped.
    """
    nt = len(sched.teams)
    lg = 2 * nt
    tau2 = np.concatenate([np.full(nt, params.tau_off**2), np.full(nt, params.tau_def**2)])
    x = np.zeros(2 * nt + 1)
    p = np.diag(np.concatenate([tau2, [params.league_sd**2]]))
    pre = np.zeros((len(sched.game_ids), 2, 4))
    obs_mean = np.full(obs.y.size, np.nan)
    obs_var = np.full(obs.y.size, np.nan)
    season_end: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    keys_s = np.concatenate([sched.season, obs.season])
    keys_w = np.concatenate([sched.week, obs.week])
    seasons = sorted({int(s) for s in keys_s if start_season is None or s >= start_season})
    g_order: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(zip(sched.season.tolist(), sched.week.tolist(), strict=True)):
        g_order.setdefault(key, []).append(i)
    o_order: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(zip(obs.season.tolist(), obs.week.tolist(), strict=True)):
        o_order.setdefault(key, []).append(i)
    team = np.arange(2 * nt)
    shrink = np.concatenate([np.full(2 * nt, params.rho), [1.0]])
    first = True
    for s in seasons:
        if not first:
            x = shrink * x
            p = p * np.outer(shrink, shrink)
            p[team, team] += (1 - params.rho**2) * tau2
            p[lg, lg] += params.league_sd**2
        first = False
        weeks = sorted({int(w) for ss, w in zip(keys_s, keys_w, strict=True) if ss == s})
        for j, w in enumerate(weeks):
            if j > 0 and params.q_frac > 0:
                p[team, team] += params.q_frac * tau2
            sd = np.sqrt(np.maximum(np.diag(p), 0.0))
            for g in g_order.get((s, w), []):
                for side, t in enumerate((sched.home[g], sched.away[g])):
                    pre[g, side] = (x[t], sd[t], x[nt + t], sd[nt + t])
            batch = o_order.get((s, w), [])
            for i in batch:  # predictions use the state before any game of the week
                o, d = obs.off[i], nt + obs.deff[i]
                h = p[:, o] + p[:, d] + p[:, lg]
                obs_mean[i] = x[o] + x[d] + x[lg]
                obs_var[i] = h[o] + h[d] + h[lg] + obs.r[i]
            for i in batch:
                o, d = obs.off[i], nt + obs.deff[i]
                h = p[:, o] + p[:, d] + p[:, lg]
                big_s = h[o] + h[d] + h[lg] + obs.r[i]
                if big_s <= 0:
                    continue
                k = h / big_s
                x = x + k * (obs.y[i] - x[o] - x[d] - x[lg])
                p = p - np.outer(k, h)
        season_end[s] = (x[: 2 * nt].copy(), np.sqrt(np.maximum(np.diag(p)[: 2 * nt], 0.0)))
    return FilterResult(sched.teams, sched.game_ids, pre, obs_mean, obs_var, season_end)


def ratings_frame(
    results: dict[str, FilterResult], sched: Schedule, start_season: int | None = None
) -> pl.DataFrame:
    """Pre-game ratings in long form: one row per (game, team) with, for each metric m,
    m_off, m_off_sd, m_def, m_def_sd. Defense ratings are what the defense allows above
    average (positive = bad defense)."""
    cols: dict[str, Any] = {
        "game_id": sched.game_ids * 2,
        "season": np.concatenate([sched.season, sched.season]),
        "week": np.concatenate([sched.week, sched.week]),
        "team": [sched.teams[t] for t in sched.home] + [sched.teams[t] for t in sched.away],
    }
    for m, res in results.items():
        for k, name in enumerate(("off", "off_sd", "def", "def_sd")):
            cols[f"{m}_{name}"] = np.concatenate([res.pre[:, 0, k], res.pre[:, 1, k]])
    out = pl.DataFrame(cols)
    if start_season is not None:
        out = out.filter(pl.col("season") >= start_season)
    return out


def season_end_frame(results: dict[str, FilterResult]) -> pl.DataFrame:
    """End-of-season ratings: one row per (season, team)."""
    rows = []
    any_res = next(iter(results.values()))
    nt = len(any_res.teams)
    for s in sorted(any_res.season_end):
        for t, team in enumerate(any_res.teams):
            row: dict[str, Any] = {"season": s, "team": team}
            for m, res in results.items():
                x, sd = res.season_end[s]
                row.update({f"{m}_off": x[t], f"{m}_off_sd": sd[t], f"{m}_def": x[nt + t],
                            f"{m}_def_sd": sd[nt + t]})  # fmt: skip
            rows.append(row)
    return pl.DataFrame(rows)


def attach_ratings(df: pl.DataFrame, ratings: pl.DataFrame) -> pl.DataFrame:
    """Join pre-game ratings onto rows with posteam and defteam.

    Adds us_* (the posteam's ratings) and them_* (the defteam's) for every rating column:
    us_conv_off is our offense's conversion rating, them_conv_def the conversion rating of
    the defense we face, and so on."""
    value_cols = [c for c in ratings.columns if c not in ("game_id", "season", "week", "team")]
    us = ratings.select("game_id", pl.col("team").alias("posteam"),
                        *[pl.col(c).alias(f"us_{c}") for c in value_cols])  # fmt: skip
    them = ratings.select("game_id", pl.col("team").alias("defteam"),
                          *[pl.col(c).alias(f"them_{c}") for c in value_cols])  # fmt: skip
    return df.join(us, on=["game_id", "posteam"], how="left").join(
        them, on=["game_id", "defteam"], how="left"
    )


def rating_features(df: pl.DataFrame, side: str = "us") -> dict[str, np.ndarray]:
    """Model features of the offense on `side` ("us": posteam offense against the
    defteam defense; "them": the defteam offense against our defense)."""
    a, b = ("us", "them") if side == "us" else ("them", "us")
    return {
        "off_conv": 100.0 * np.nan_to_num(_f(df, f"{a}_conv_off")),
        "def_conv": 100.0 * np.nan_to_num(_f(df, f"{b}_conv_def")),
        "off_drive": np.nan_to_num(_f(df, f"{a}_drive_off")),
        "def_drive": np.nan_to_num(_f(df, f"{b}_drive_def")),
    }


# ---- tuning ----


def filter_scores(
    metric: str, sample: pl.DataFrame, obs: Observations, res: FilterResult
) -> np.ndarray:
    """Per-row out-of-sample score of the filter's pre-game prediction (lower is better):
    conversion rows: log loss of base_p + rating sum; drive rows: squared error."""
    pred = res.obs_mean[obs.row_obs]
    if metric == "conv":
        p = np.clip(_f(sample, "base_p") + pred, 0.01, 0.99)
        y = _f(sample, "converted")
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return (_f(sample, "resid") - pred) ** 2


def grid_params(grid: dict, league_sd: float = 0.0) -> list[FilterParams]:
    return [FilterParams(float(a), float(b), float(r), float(q), float(league_sd))
            for a in grid["tau_off"] for b in grid["tau_def"]
            for r in grid["rho"] for q in grid["q_frac"]]  # fmt: skip


def tune_filter(
    metric: str,
    sample: pl.DataFrame,
    sched: Schedule,
    grid: dict,
    blocks: Sequence[Sequence[int]],
    start_season: int,
    league_sd: float = 0.0,
) -> dict[str, Any]:
    """Score every grid point season by season (development seasons only).

    Returns {"params": [...], "season_loss": array (grid points x seasons, summed),
    "seasons": [...], "n_rows": per-season row counts}."""
    _guard(sample, f"M8 filter tuning ({metric})")
    obs = observations(sample, sched)
    seasons = sorted({int(s) for blk in blocks for s in range(blk[0], blk[1] + 1)})
    row_season = sample["season"].to_numpy()
    params = grid_params(grid, league_sd)
    loss = np.zeros((len(params), len(seasons)))
    for i, prm in enumerate(params):
        sc = filter_scores(metric, sample, obs, run_filter(obs, sched, prm, start_season))
        for j, s in enumerate(seasons):
            loss[i, j] = sc[row_season == s].sum()
    counts = np.array([(row_season == s).sum() for s in seasons])
    return {"params": params, "season_loss": loss, "seasons": seasons, "n_rows": counts}


def choose_params(tuned: dict[str, Any], seasons: Sequence[int]) -> tuple[FilterParams, int]:
    """The grid point with the lowest total loss over `seasons`."""
    cols = [tuned["seasons"].index(int(s)) for s in seasons]
    total = tuned["season_loss"][:, cols].sum(axis=1)
    i = int(np.argmin(total))
    return tuned["params"][i], i


def no_effect_index(tuned: dict[str, Any]) -> int:
    """Index of a grid point with tau_off = tau_def = 0 (no team effect at all)."""
    for i, prm in enumerate(tuned["params"]):
        if prm.tau_off == 0 and prm.tau_def == 0:
            return i
    raise ValueError("the grid has no point with tau_off = tau_def = 0")


# ======================================================================================
# 4. Effect models: spread only, and spread plus ratings
# ======================================================================================


def _fit_offset_logit(
    x: np.ndarray, y: np.ndarray, offset: np.ndarray, w: np.ndarray, l2: float
) -> np.ndarray:
    """Weighted logistic regression with a fixed offset: log-odds = offset + x @ beta.
    A tiny ridge penalty `l2` on every coefficient but the first keeps the solver stable."""

    def loss(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = offset + x @ beta
        p = _sigmoid(z)
        ll = np.sum(w * (np.logaddexp(0.0, z) - y * z)) + 0.5 * l2 * np.sum(beta[1:] ** 2)
        grad = x.T @ (w * (p - y))
        grad[1:] += l2 * beta[1:]
        return float(ll), grad

    res = minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    return res.x


@dataclass
class ConversionModel:
    """log-odds(convert) = league base + a + b * spread / 10 + sum_k c_k * rating_k.

    The league base (down, distance, field position, season) is the M1 conversion curve at
    an even spread, used as a fixed offset. With `ratings = []` this is the spread-only
    model; `use_spread = False` drops the spread term (a diagnostic only).
    """

    ratings: list[str]
    use_spread: bool = True
    l2: float = 1e-4
    coef_: dict[str, float] = field(default_factory=dict)

    def _x(self, spread: np.ndarray, feats: dict[str, np.ndarray]) -> np.ndarray:
        cols = [np.ones_like(spread)]
        if self.use_spread:
            cols.append(spread / 10.0)
        cols += [feats[k] for k in self.ratings]
        return np.column_stack(cols)

    def names(self) -> list[str]:
        return ["intercept"] + (["spread_per_10"] if self.use_spread else []) + list(self.ratings)

    def fit(self, sample: pl.DataFrame, w: np.ndarray | None = None) -> ConversionModel:
        _guard(sample, "M8 conversion effect model")
        spread = _f(sample, "posteam_spread")
        x = self._x(spread, rating_features(sample))
        w = np.ones(sample.height) if w is None else w
        beta = _fit_offset_logit(x, _f(sample, "converted"), _f(sample, "base_logit"), w, self.l2)
        self.coef_ = dict(zip(self.names(), beta.tolist(), strict=True))
        return self

    def linear_predictor(self, spread: np.ndarray, feats: dict[str, np.ndarray]) -> np.ndarray:
        """Everything but the league base."""
        beta = np.array([self.coef_[k] for k in self.names()])
        return self._x(np.asarray(spread, dtype=float), feats) @ beta

    def predict(self, sample: pl.DataFrame) -> np.ndarray:
        lp = self.linear_predictor(_f(sample, "posteam_spread"), rating_features(sample))
        return _sigmoid(_f(sample, "base_logit") + lp)


@dataclass
class DriveModel:
    """P(drive ends in a touchdown, a field goal, or neither) by start yard line (a spline),
    the offense's pregame spread and, optionally, ratings: multinomial logistic regression.
    `use_spread = False` drops the spread (a diagnostic only)."""

    ratings: list[str]
    knots: tuple[int, ...] = (1, 10, 20, 35, 50, 65, 80, 99)
    l2: float = 1e-4
    use_spread: bool = True
    seasons_: list[int] = field(default_factory=list)

    def _spline(self) -> SplineTransformer:
        if not hasattr(self, "_spl"):
            self._spl = SplineTransformer(
                degree=3, knots=np.array(self.knots, dtype=float).reshape(-1, 1),
                include_bias=False,
            ).fit(np.array([[1.0], [99.0]]))  # fmt: skip
        return self._spl

    def _x(self, start: np.ndarray, spread: np.ndarray, feats: dict[str, np.ndarray]
           ) -> np.ndarray:  # fmt: skip
        basis = self._spline().transform(np.asarray(start, dtype=float).reshape(-1, 1))
        cols = [basis]
        if self.use_spread:
            cols.append((np.asarray(spread, dtype=float) / 10.0).reshape(-1, 1))
        cols += [np.asarray(feats[k], dtype=float).reshape(-1, 1) for k in self.ratings]
        x = np.hstack(cols)
        if not hasattr(self, "_center"):  # fit sets the scaling from its rows; then reused
            self._center, self._scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-9)
        return (x - self._center) / self._scale  # standardized: the solver converges fast

    def fit(
        self, drives: pl.DataFrame, w: np.ndarray | None = None, init: DriveModel | None = None
    ) -> DriveModel:
        """Fit by weighted maximum likelihood. `init`, a fitted model of the same form,
        starts the solver at its coefficients (used for bootstrap refits)."""
        _guard(drives, "M8 drive outcome model")
        y = drives["outcome"].replace_strict({c: i for i, c in enumerate(DRIVE_CLASSES)},
                                             return_dtype=pl.Int64).to_numpy()  # fmt: skip
        if init is not None:
            self._center, self._scale = init._center, init._scale
        else:
            self.__dict__.pop("_center", None)  # a refit recomputes the scaling
        x = self._x(_f(drives, "start"), _f(drives, "posteam_spread"), rating_features(drives))
        if init is not None:
            self._lr = copy.deepcopy(init._lr)
            self._lr.set_params(warm_start=True, tol=1e-6)
        else:
            self._lr = LogisticRegression(C=1.0 / self.l2, max_iter=3000, tol=1e-8)
        self._lr.fit(x, y, sample_weight=w)
        if list(self._lr.classes_) != [0, 1, 2]:
            raise ValueError("drive model needs all three outcome classes in the data")
        self.seasons_ = sorted({int(s) for s in drives["season"].unique().to_list()})
        return self

    def rating_coef(self) -> dict[str, dict[str, float]]:
        """Log-odds coefficient of each rating per unit (conversion ratings per percentage
        point, drive ratings per point), by outcome class. Only differences between classes
        matter in a multinomial model; "none" is the natural reference."""
        k = len(self.ratings)
        if k == 0:
            return {}
        per_unit = self._lr.coef_[:, -k:] / self._scale[-k:]
        return {c: dict(zip(self.ratings, (per_unit[i] - per_unit[0]).tolist(), strict=True))
                for i, c in enumerate(DRIVE_CLASSES)}  # fmt: skip

    def proba(self, start: np.ndarray, spread: np.ndarray, feats: dict[str, np.ndarray]
              ) -> np.ndarray:  # fmt: skip
        """(n, 3) probabilities of none, field goal, touchdown."""
        return self._lr.predict_proba(self._x(start, spread, feats))

    def predict(self, drives: pl.DataFrame) -> np.ndarray:
        return self.proba(_f(drives, "start"), _f(drives, "posteam_spread"),
                          rating_features(drives))  # fmt: skip


def conversion_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def drive_log_loss(outcome: Sequence[str], proba: np.ndarray) -> np.ndarray:
    idx = np.array([DRIVE_CLASSES.index(o) for o in outcome])
    return -np.log(np.clip(proba[np.arange(idx.size), idx], 1e-12, 1.0))


def game_bootstrap_mean_diff(
    games: Sequence[str], diff: np.ndarray, n_boot: int, seed: int, level: float = 0.95
) -> dict[str, float]:
    """Mean of a per-row difference with a bootstrap interval that resamples whole games."""
    codes, inv = np.unique(np.asarray(games), return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=codes.size)
    counts = np.bincount(inv, minlength=codes.size).astype(float)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.bincount(rng.integers(0, codes.size, codes.size), minlength=codes.size)
        reps[b] = (pick @ sums) / (pick @ counts)
    lo, hi = np.quantile(reps, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi),
            "n_rows": int(diff.size), "n_games": int(codes.size)}  # fmt: skip


def game_weights(games: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    """Row weights of one game-level bootstrap draw (how often each row's game is drawn)."""
    codes, inv = np.unique(np.asarray(games), return_inverse=True)
    counts = np.bincount(rng.integers(0, codes.size, codes.size), minlength=codes.size)
    return counts[inv].astype(float)


def choose_variant(losses: dict[str, float], order: Sequence[str], min_improvement: float
                   ) -> str:  # fmt: skip
    """The rating variant to carry to the test seasons: the lowest out-of-fold loss among
    `order` (simplest first), except that a simpler variant within `min_improvement` of the
    best is preferred."""
    best = min(order, key=lambda k: losses[k])
    for k in order:
        if losses[k] <= losses[best] + min_improvement:
            return k
    return best


def prepare_samples(
    plays: pl.DataFrame, bases: Bases, config: dict, first_season: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Conversion and drive samples from `first_season` on, with residuals against the
    per-season bases (which must cover every season present)."""
    rows = plays.filter(pl.col("season") >= first_season)
    return add_residuals(conversion_sample(rows, config), drive_sample(rows, config), bases)


def compute_ratings(
    conv: pl.DataFrame,
    drives: pl.DataFrame,
    sched: Schedule,
    params: dict[str, FilterParams],
    start_season: int,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, FilterResult]]:
    """Run both filters. Returns (pre-game ratings per game and team, end-of-season
    ratings per season and team, the raw filter results)."""
    res = {
        "conv": run_filter(observations(conv, sched), sched, params["conv"], start_season),
        "drive": run_filter(observations(drives, sched), sched, params["drive"], start_season),
    }
    return ratings_frame(res, sched, start_season), season_end_frame(res), res


def group_ratings(season_end: pl.DataFrame, seasons: Sequence[int], k: int = 5
                  ) -> dict[str, dict[str, float]]:  # fmt: skip
    """Average end-of-season ratings of the top and bottom `k` units of each season, ranked
    by drive rating. Returns offense groups {top, bottom, average} with conv_off and
    drive_off, and defense groups {best, worst, average} with conv_def and drive_def."""
    df = season_end.filter(pl.col("season").is_in([int(s) for s in seasons]))
    out: dict[str, dict[str, float]] = {
        "average": {"conv_off": 0.0, "drive_off": 0.0, "conv_def": 0.0, "drive_def": 0.0}
    }
    for name, col, desc, keep in (
        ("top", "drive_off", True, ("conv_off", "drive_off")),
        ("bottom", "drive_off", False, ("conv_off", "drive_off")),
        ("worst", "drive_def", True, ("conv_def", "drive_def")),
        ("best", "drive_def", False, ("conv_def", "drive_def")),
    ):
        sel = df.sort(col, descending=desc).group_by("season", maintain_order=True).head(k)
        out[name] = {c: float(sel[c].mean()) for c in keep}
    return out  # fmt: skip


def unit_features(offense: dict[str, float], defense: dict[str, float]) -> dict[str, float]:
    """Model features (see `rating_features`) of an offense unit facing a defense unit."""
    return {"off_conv": 100.0 * offense.get("conv_off", 0.0),
            "off_drive": offense.get("drive_off", 0.0),
            "def_conv": 100.0 * defense.get("conv_def", 0.0),
            "def_drive": defense.get("drive_def", 0.0)}  # fmt: skip


SPREAD_REGRESSORS: tuple[str, ...] = (
    "us_conv_off", "us_conv_def", "us_drive_off", "us_drive_def",
    "them_conv_off", "them_conv_def", "them_drive_off", "them_drive_def",
)  # fmt: skip


def fit_spread_on_ratings(games: pl.DataFrame) -> dict[str, Any]:
    """How much of the pregame spread the pre-game ratings explain: least squares of the
    posteam's spread on both teams' ratings (conversion ratings in points), one row per
    team and game, no intercept (each game appears from both sides)."""
    _guard(games, "M8 spread-on-ratings regression")
    x = np.column_stack([_f(games, c) * (100.0 if "_conv_" in c else 1.0)
                         for c in SPREAD_REGRESSORS])  # fmt: skip
    y = _f(games, "posteam_spread")
    ok = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
    beta = np.linalg.lstsq(x[ok], y[ok], rcond=None)[0]
    resid = y[ok] - x[ok] @ beta
    r2 = 1.0 - float(resid @ resid) / float(y[ok] @ y[ok])
    return {"coef": dict(zip(SPREAD_REGRESSORS, beta.tolist(), strict=True)),
            "r2_uncentered": r2, "n": int(ok.sum())}  # fmt: skip


def implied_spread(
    fit: dict[str, Any],
    us_off: dict[str, float],
    us_def: dict[str, float],
    them_off: dict[str, float],
    them_def: dict[str, float],
) -> float:
    """Spread the regression implies for a pairing of units (ratings in raw units)."""
    vals = {
        "us_conv_off": us_off.get("conv_off", 0.0), "us_drive_off": us_off.get("drive_off", 0.0),
        "us_conv_def": us_def.get("conv_def", 0.0), "us_drive_def": us_def.get("drive_def", 0.0),
        "them_conv_off": them_off.get("conv_off", 0.0),
        "them_drive_off": them_off.get("drive_off", 0.0),
        "them_conv_def": them_def.get("conv_def", 0.0),
        "them_drive_def": them_def.get("drive_def", 0.0),
    }  # fmt: skip
    return float(sum(fit["coef"][k] * vals[k] * (100.0 if "_conv_" in k else 1.0)
                     for k in SPREAD_REGRESSORS))  # fmt: skip


# ======================================================================================
# 5. Drive constants used by the engine extension
# ======================================================================================


def drive_constants(drives: pl.DataFrame, fallback: dict[str, float]) -> dict[str, Any]:
    """Mean seconds a drive takes by outcome, and a straight-line map from a drive's start
    to where the other team starts after a drive that scored nothing (both from
    development drives)."""
    _guard(drives, "M8 drive constants")
    secs = {}
    for c in DRIVE_CLASSES:
        sub = drives.filter(pl.col("outcome") == c)
        secs[c] = float(sub["seconds"].mean()) if sub.height else float(fallback[c])
    none = drives.filter((pl.col("outcome") == "none") & pl.col("next_start").is_not_null())
    if none.height > 10:
        b, a = np.polyfit(_f(none, "start"), _f(none, "next_start"), 1)
    else:
        a, b = 70.0, 0.0
    return {"seconds": secs, "next_start_intercept": float(a), "next_start_slope": float(b),
            "n_none_with_next": int(none.height)}  # fmt: skip


# ======================================================================================
# 6. The team-aware engine
# ======================================================================================


class TeamAwareEngine(decision.DecisionEngine):
    """The M2 engine plus a record of how each action's value responds to team strength.

    `sensitivities(states)` returns the engine's usual `action_values` and an array
    S of shape (n, 5 slots, 2 owners, 100 yard lines, 2) where

        S[i, slot, owner, y, k] = sum over next states of action-slot `slot` of state i in
            which `owner` (0 = us, 1 = them) starts with the ball at yard line y, of
            (probability of that next state) * (our win probability if that team's drive
            scores k (0 = touchdown, 1 = field goal) minus if it scores nothing).

    For any change dP[owner, y, k] in a drive's touchdown and field-goal chances, the
    change in the slot's value is sum over (owner, y, k) of S * dP: the next drive's
    outcome is the one thing the team adjustment changes, and value is linear in its
    probabilities. Next states with `min_half_seconds` or less left in the half get no
    adjustment. Only `_chunk_rows`, `_values_chunk` and `_value_leaves` of the parent are
    extended; every value the parent computes is unchanged.
    """

    def __init__(
        self,
        wp_model: Any,
        components: Any,
        drive_consts: dict[str, Any],
        config: dict | None = None,
        min_half_seconds: int = 360,
        state_chunk_rows: int = 3000,
    ) -> None:
        cfg = copy.deepcopy(config or {})
        cfg["state_chunk_rows"] = int(state_chunk_rows)
        super().__init__(wp_model, components, cfg)
        self.drive_consts = drive_consts
        self.min_half_seconds = int(min_half_seconds)
        self._sens_parts: list[np.ndarray] = []
        self._sens_chunk = np.zeros((0, N_SLOTS, 2, 100, 2))
        self._sens_rows = np.zeros(0, dtype=np.int64)

    # ---- hooks into the parent's chunk loop
    def _chunk_rows(self, states: pl.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        rows, st = super()._chunk_rows(states)
        self._sens_rows = rows
        self._sens_chunk = np.zeros((rows.size, N_SLOTS, 2, 100, 2))
        return rows, st

    def _values_chunk(self, states: pl.DataFrame) -> dict[str, np.ndarray]:
        out = super()._values_chunk(states)
        sens = np.zeros((states.height, N_SLOTS, 2, 100, 2))
        if self._sens_rows.size:
            sens[self._sens_rows] = self._sens_chunk
        self._sens_parts.append(sens)
        return out

    def _value_leaves(
        self, acc: np.ndarray, st: dict[str, np.ndarray], leaves: list[tuple[np.ndarray, ...]]
    ) -> None:
        super()._value_leaves(acc, st, leaves)
        l_row, l_slot, l_w, l_owner, l_y, l_time, _l_diff, l_reset = (
            np.concatenate(x) for x in zip(*leaves, strict=True)
        )
        l_diff = _l_diff
        half_left = np.where(l_time > decision.HALFTIME_SECONDS,
                             l_time - decision.HALFTIME_SECONDS, l_time)  # fmt: skip
        ok = half_left > self.min_half_seconds
        if not ok.any():
            return
        sel = np.nonzero(ok)[0]
        a_td, a_fg = self._drive_value_gaps(
            l_row[sel], l_owner[sel], l_y[sel], l_time[sel], l_diff[sel], l_reset[sel], st
        )
        sign = np.where(l_owner[sel] == OWNER_US, 1.0, -1.0)
        idx = (l_row[sel], l_slot[sel], l_owner[sel], l_y[sel])
        np.add.at(self._sens_chunk, (*idx, np.zeros(sel.size, dtype=np.int64)),
                  l_w[sel] * sign * a_td)  # fmt: skip
        np.add.at(self._sens_chunk, (*idx, np.ones(sel.size, dtype=np.int64)),
                  l_w[sel] * sign * a_fg)  # fmt: skip

    def _drive_value_gaps(
        self, row: np.ndarray, owner: np.ndarray, y: np.ndarray, time: np.ndarray,
        diff: np.ndarray, reset: np.ndarray, st: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
        """For each next state, the owner's win probability after its drive scores a
        touchdown (and after a field goal) minus after it scores nothing.

        After the drive the other team has the ball: at the mean kickoff start after a
        score, or at the fitted "next start" spot after a drive that scored nothing, with
        the clock run down by the mean drive time of that outcome and timeouts unchanged.
        """
        theirs = owner == OWNER_OPP
        o_diff = np.where(theirs, -diff, diff)
        o_pos = np.where(reset == 1, 3, np.where(theirs, st["def_to"][row], st["pos_to"][row]))
        o_def = np.where(reset == 1, 3, np.where(theirs, st["pos_to"][row], st["def_to"][row]))
        o_home = np.where(theirs, 1 - st["home"][row], st["home"][row])
        o_r2h = np.where(time > decision.HALFTIME_SECONDS,
                         np.where(theirs, 1 - st["r2h"][row], st["r2h"][row]), 0)  # fmt: skip
        o_spread = np.where(theirs, -st["spread"][row], st["spread"][row])
        dc = self.drive_consts
        none_y = np.clip(np.rint(dc["next_start_intercept"] + dc["next_start_slope"] * y),
                         1, 99).astype(np.int64)  # fmt: skip
        ko_y = np.full(y.size, int(self._ko_mean_start))
        n = y.size
        nxt_y = np.concatenate([ko_y, ko_y, none_y])
        pts = np.concatenate(
            [
                np.full(n, DRIVE_CLASS_POINTS["td"]),
                np.full(n, DRIVE_CLASS_POINTS["fg"]),
                np.zeros(n, dtype=int),
            ]
        )
        secs = np.concatenate([np.full(n, dc["seconds"]["td"]), np.full(n, dc["seconds"]["fg"]),
                               np.full(n, dc["seconds"]["none"])])  # fmt: skip
        t3 = np.tile(time, 3)
        n_time = np.maximum(np.rint(t3 - secs), 1).astype(np.int64)
        n_diff = np.clip(-(np.tile(o_diff, 3) + pts), -127, 127)
        n_r2h = np.where(n_time > decision.HALFTIME_SECONDS, 1 - np.tile(o_r2h, 3), 0)
        frame_args = {
            "yardline_100": nxt_y, "game_seconds_remaining": n_time, "score_differential": n_diff,
            "posteam_timeouts_remaining": np.tile(o_def, 3),
            "defteam_timeouts_remaining": np.tile(o_pos, 3),
            "posteam_spread": -np.tile(o_spread, 3), "receive_2h_ko": n_r2h,
            "is_home": 1 - np.tile(o_home, 3),
        }  # fmt: skip
        wp_next = self._predict_unique(frame_args)
        w_owner = 1.0 - wp_next
        w_td, w_fg, w_none = w_owner[:n], w_owner[n : 2 * n], w_owner[2 * n :]
        return w_td - w_none, w_fg - w_none

    def _predict_unique(self, args: dict[str, np.ndarray]) -> np.ndarray:
        """Win probability of first-and-10 states, each distinct state scored once."""
        spread_vals, spread_inv = np.unique(args["posteam_spread"], return_inverse=True)
        key = np.asarray(args["yardline_100"], dtype=np.int64)
        for part, bits in ((args["game_seconds_remaining"], 12),
                           (np.asarray(args["score_differential"]) + 128, 8),
                           (args["posteam_timeouts_remaining"], 2),
                           (args["defteam_timeouts_remaining"], 2), (args["is_home"], 1),
                           (args["receive_2h_ko"], 1), (spread_inv, 22)):  # fmt: skip
            key = (key << bits) | np.asarray(part, dtype=np.int64)
        uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
        frame = wp_mod.state_frame(
            down=1, ydstogo=np.minimum(10, args["yardline_100"][first]),
            yardline_100=args["yardline_100"][first],
            game_seconds_remaining=args["game_seconds_remaining"][first],
            score_differential=np.asarray(args["score_differential"])[first],
            posteam_timeouts_remaining=args["posteam_timeouts_remaining"][first],
            defteam_timeouts_remaining=args["defteam_timeouts_remaining"][first],
            posteam_spread=spread_vals[spread_inv[first]],
            receive_2h_ko=args["receive_2h_ko"][first], is_home=args["is_home"][first],
        )  # fmt: skip
        del uniq
        return np.asarray(self.wp_model.predict(frame), dtype=float)[inv]

    # ---- public
    def sensitivities(self, states: pl.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """`action_values(states)` and the sensitivity array S (see the class docstring).

        Rows are grouped by kickoff rule era first, so each call of the parent's
        `action_values` sees one era and processes its chunks in order."""
        n = states.height
        eras = self._kickoff_eras(states)
        groups = ([np.arange(n)] if eras is None
                  else [np.nonzero(eras == e)[0] for e in sorted(set(eras.tolist()))])  # fmt: skip
        values: dict[str, np.ndarray] = {}
        sens = np.zeros((n, N_SLOTS, 2, 100, 2))
        for idx in groups:
            self._sens_parts = []
            v = self.action_values(states[idx])
            sens[idx] = np.concatenate(self._sens_parts)[: idx.size]
            for k, arr in v.items():
                values.setdefault(k, np.full(n, np.nan))[idx] = arr
        self._sens_parts = []
        return values, sens


@dataclass
class TeamContext:
    """Everything team-specific about a set of fourth-down states.

    spread: the offense's pregame spread. us / them: rating features (see
    `rating_features`) of our offense against their defense, and of their offense against
    our defense."""

    spread: np.ndarray
    us: dict[str, np.ndarray]
    them: dict[str, np.ndarray]

    @classmethod
    def from_frame(cls, df: pl.DataFrame) -> TeamContext:
        return cls(_f(df, "posteam_spread"), rating_features(df, "us"),
                   rating_features(df, "them"))  # fmt: skip

    @classmethod
    def constant(cls, n: int, spread: float, us: dict[str, float], them: dict[str, float]
                 ) -> TeamContext:  # fmt: skip
        return cls(np.full(n, float(spread)),
                   {k: np.full(n, float(us.get(k, 0.0))) for k in RATING_FEATURES},
                   {k: np.full(n, float(them.get(k, 0.0))) for k in RATING_FEATURES})  # fmt: skip


def drive_shift(team: DriveModel, base: DriveModel, ctx: TeamContext) -> np.ndarray:
    """dP[i, owner, y, k]: team model minus spread-only model touchdown (k = 0) and field
    goal (k = 1) chances of a drive starting at yard line y (1 to 99; index 0 unused), for
    our drives (owner 0, our spread) and theirs (owner 1, minus our spread)."""
    n = ctx.spread.size
    out = np.zeros((n, 2, 100, 2))
    ys = np.arange(1, 100)
    for owner, (spread, feats) in enumerate(((ctx.spread, ctx.us), (-ctx.spread, ctx.them))):
        start = np.tile(ys, n)
        sp = np.repeat(spread, ys.size)
        f = {k: np.repeat(v, ys.size) for k, v in feats.items()}
        d = team.proba(start, sp, f) - base.proba(start, sp, f)
        out[:, owner, 1:, 0] = d[:, 2].reshape(n, ys.size)
        out[:, owner, 1:, 1] = d[:, 1].reshape(n, ys.size)
    return out


def team_aware_table(
    values: dict[str, np.ndarray],
    sens: np.ndarray,
    p_spread: np.ndarray,
    dlogit: np.ndarray,
    dp: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Spread-only and team-aware action values from one engine pass.

    values: the engine's `action_values` (its conversion probability p_engine is at an
        even spread). p_spread: P(convert) at the offense's spread (the spread-only
        engine). dlogit: team-model change in conversion log-odds. dp: drive shift from
        `drive_shift` (None = no drive adjustment).

    Returns spread_table and team_table (n x 3 arrays of go, field goal, punt) and the
    conversion probabilities. With dlogit = 0 and dp = 0 the two tables are identical.
    """
    p_eng = values["p_convert"]
    wpc, wpf = values["wp_if_convert"], values["wp_if_fail"]
    fg, punt = values["fg_wp"], values["punt_wp"]
    d = np.zeros((p_eng.size, N_SLOTS)) if dp is None else np.einsum("nsoyk,noyk->ns", sens, dp)
    with np.errstate(invalid="ignore", divide="ignore"):
        wpc_t = np.where(p_eng > 0, wpc + d[:, SLOT_CONVERT] / p_eng, wpc)
        wpf_t = np.where(p_eng < 1, wpf + d[:, SLOT_FAIL] / (1 - p_eng), wpf)
    p_team = _sigmoid(_logit(p_spread) + dlogit)
    go_s = p_spread * wpc + (1 - p_spread) * wpf
    go_t = p_team * wpc_t + (1 - p_team) * wpf_t
    fg_t = fg + d[:, SLOT_FG_MAKE] + d[:, SLOT_FG_MISS]
    punt_t = punt + d[:, SLOT_PUNT]
    return {
        "spread_table": np.column_stack([go_s, fg, punt]),
        "team_table": np.column_stack([go_t, fg_t, punt_t]),
        "p_spread": p_spread,
        "p_team": p_team,
    }


@dataclass
class TeamEffects:
    """The fitted pieces the decision step needs (saved by scripts/m8_team_strength.py)."""

    conv_spread: ConversionModel
    conv_team: ConversionModel
    drive_spread: DriveModel
    drive_team: DriveModel
    filter_params: dict[str, FilterParams]
    drive_consts: dict[str, Any]
    stamp: dict[str, Any] = field(default_factory=dict)

    def dlogit(self, ctx: TeamContext) -> np.ndarray:
        return (self.conv_team.linear_predictor(ctx.spread, ctx.us)
                - self.conv_spread.linear_predictor(ctx.spread, ctx.us))  # fmt: skip

    def drive_shift(self, ctx: TeamContext) -> np.ndarray:
        return drive_shift(self.drive_team, self.drive_spread, ctx)


def flips(spread_table: np.ndarray, team_table: np.ndarray) -> dict[str, np.ndarray]:
    """Best action of each table, whether it changed, and the change in the go gain."""
    b_s, _, m_s, g_s = decision.summarize_actions(spread_table)
    b_t, _, m_t, g_t = decision.summarize_actions(team_table)
    return {"best_spread": b_s, "best_team": b_t, "changed": (b_s != b_t) & (b_s >= 0),
            "margin_spread": m_s, "margin_team": m_t, "go_gain_spread": g_s,
            "go_gain_team": g_t}  # fmt: skip
