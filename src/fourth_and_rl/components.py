"""M1 play-outcome ("component") models: what happens after each choice.

Everything here answers one question: given the situation and an action, what is the
probability of each next situation? The one-step decision engine (M2) and the Markov
decision process (M4, M5) both read these models, so every model returns a full
discrete probability distribution as a plain numpy array, never just an average.

Conventions used throughout
---------------------------
* `yardline_100` is the distance in yards from the ball to the end zone the team with
  the ball is attacking (1 to 99). After a change of possession the other team's
  yardline_100 is 100 minus the old one.
* An array of length 100 indexed by yardline_100 uses indices 1 to 99 for field
  positions. Index 0 means "reached the end zone", so it holds a touchdown: in
  `turnover` and punt arrays it is a return touchdown by the team that took the ball.
* Kick distance of a field goal is yardline_100 + 18 (snap 8 yards back, goal posts
  10 yards deep).

The models are simple on purpose. Each `fit` takes a few seconds so that a later
milestone can refit all of them on bootstrap samples of games.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

from fourth_and_rl.paths import CONFIGS, DATA

MODEL_PATH = DATA / "models" / "components.joblib"

# Yards gained on one play are stored on a fixed grid from G_MIN to G_MAX.
# G_MAX = 100 means "longer than any gain we could observe": a touchdown from anywhere.
G_MIN, G_MAX = -30, 100
GAINS = np.arange(G_MIN, G_MAX + 1)
N_G = GAINS.size

# Context bins of the count-based models (lower edges).
DIST_EDGES = np.array([1, 2, 3, 4, 6, 8, 10, 11, 16])  # 1, 2, 3, 4-5, 6-7, 8-9, 10, 11-15, 16+
FIELD_EDGES = np.array([1, 5, 11, 21, 41, 76, 96])  # 1-4, 5-10, 11-20, 21-40, 41-75, 76-95, 96-99
TURNOVER_DIST_EDGES = np.array([1, 3, 7])  # 1-2, 3-6, 7+ (go model, share of failures)

_T = np.arange(-99, 100)  # yards downfield of the line of scrimmage where a turnover ends
_SEASON_ZERO, _SEASON_SCALE = 2018.0, 10.0
_NEXT_COLS = {"next_posteam", "next_yardline_100", "next_half", "next_row_seconds",
              "next_row_half"}  # fmt: skip


def load_config(path: Path | None = None) -> dict:
    """Read configs/components.yaml."""
    with open(path or CONFIGS / "components.yaml") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def add_next_state(plays: pl.DataFrame) -> pl.DataFrame:
    """Add, for every row, who has the ball and where at the next snap of the same game.

    Columns added: next_posteam, next_yardline_100, next_half (from the next row that
    has a down), and next_row_seconds, next_row_half (from the very next row, used for
    the clock). Rows come back sorted by game and play order.

    The lookup is done within `game_id`, so call this once on the full table BEFORE
    resampling games for a bootstrap (a resample repeats game ids). Frames that already
    have the columns are returned unchanged.
    """
    if _NEXT_COLS <= set(plays.columns):
        return plays
    live = (
        pl.col("down").is_not_null()
        & pl.col("yardline_100").is_not_null()
        & pl.col("posteam").is_not_null()
    )

    def nxt(col: str) -> pl.Expr:
        return pl.when(live).then(pl.col(col)).otherwise(None).shift(-1).backward_fill()

    return plays.sort("game_id", "order_sequence", "play_id").with_columns(
        nxt("posteam").over("game_id").alias("next_posteam"),
        nxt("yardline_100").over("game_id").alias("next_yardline_100"),
        nxt("game_half").over("game_id").alias("next_half"),
        pl.col("game_seconds_remaining").shift(-1).over("game_id").alias("next_row_seconds"),
        pl.col("game_half").shift(-1).over("game_id").alias("next_row_half"),
    )


def _bin(x: np.ndarray | float, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 1)


def _num(df: pl.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in df.columns:
        return np.full(df.height, default, dtype=float)
    return df[name].cast(pl.Float64).fill_null(default).to_numpy()


def _flag(df: pl.DataFrame, expr: pl.Expr) -> np.ndarray:
    return df.select(expr.fill_null(False).alias("x"))["x"].to_numpy()


def _season_filter(plays: pl.DataFrame, first_season: int | None) -> pl.DataFrame:
    if first_season is None or "season" not in plays.columns:
        return plays
    return plays.filter(pl.col("season") >= first_season)


def _season_t(season: np.ndarray | float) -> np.ndarray | float:
    return (season - _SEASON_ZERO) / _SEASON_SCALE


def _opponent_next_yardline(df: pl.DataFrame) -> np.ndarray:
    """The other team's yardline_100 at the next snap, or nan if they do not have the
    ball then or the half ended first."""
    known = (
        (pl.col("next_posteam") != pl.col("posteam"))
        & (pl.col("next_half") == pl.col("game_half"))
        & pl.col("next_yardline_100").is_not_null()
    )
    return df.select(
        pl.when(known.fill_null(False)).then(pl.col("next_yardline_100")).otherwise(None).alias("x")
    )["x"].cast(pl.Float64).fill_null(np.nan).to_numpy()


_OFF_TD = (pl.col("touchdown") == 1) & (pl.col("td_team") == pl.col("posteam"))
_DEF_TD = (pl.col("touchdown") == 1) & (pl.col("td_team") == pl.col("defteam"))
_TURNOVER = (pl.col("interception") == 1) | (pl.col("fumble_lost") == 1)


def _scrimmage(plays: pl.DataFrame, min_down: int, max_down: int) -> dict[str, np.ndarray]:
    """Runs and passes on the given downs, as numpy columns."""
    df = add_next_state(plays).filter(
        pl.col("play_type").is_in(["run", "pass"])
        & pl.col("down").is_between(min_down, max_down)
        & pl.col("yardline_100").is_between(1, 99)
        & (pl.col("ydstogo") >= 1)
        & pl.col("yards_gained").is_not_null()
    )
    turnover = _flag(df, _TURNOVER)
    off_td = _flag(df, _OFF_TD) & ~turnover
    return {
        "down": _num(df, "down").astype(int),
        "ydstogo": _num(df, "ydstogo").astype(int),
        "yardline": _num(df, "yardline_100").astype(int),
        "gain": _num(df, "yards_gained").astype(int),
        "season": _num(df, "season", _SEASON_ZERO),
        "spread": _num(df, "posteam_spread", 0.0),
        "first_down": _num(df, "first_down") == 1,
        "off_td": off_td,
        "def_td": _flag(df, _DEF_TD),
        "turnover": turnover,
        "opp_next": _opponent_next_yardline(df),
    }


def km_gain_pmf(gain: np.ndarray, yardline: np.ndarray) -> np.ndarray:
    """Distribution of the yards a play would gain on an endless field.

    A play that starts k yards from the end zone cannot show a gain above k: every
    longer gain is recorded as a touchdown. Those plays are "censored" at k. The
    Kaplan-Meier estimator handles this: for each gain g it estimates
        h(g) = P(gain = g | gain >= g)
    using only plays that could have shown g (those that started more than g yards out
    and gained at least g), then multiplies the survival chances back together.

    Returns probabilities over `GAINS`. Whatever probability is left beyond the longest
    gain that could be observed goes to the last entry (G_MAX), which is a touchdown
    from any field position.
    """
    g = np.clip(np.asarray(gain, dtype=int), G_MIN, 99)
    yl = np.asarray(yardline, dtype=int)
    censored = g >= yl
    obs = np.minimum(g, yl) - G_MIN
    seen = np.bincount(obs, minlength=N_G).astype(float)
    events = np.bincount(obs[~censored], minlength=N_G).astype(float)
    # At risk of showing a gain of exactly g: gained at least g AND started more than g
    # yards out (a play censored at g could only show "g or more", never "exactly g").
    at_risk = seen[::-1].cumsum()[::-1] - (seen - events)
    hazard = np.divide(events, at_risk, out=np.zeros(N_G), where=at_risk > 0)
    survive_before = np.concatenate([[1.0], np.cumprod(1.0 - hazard)[:-1]])
    pmf = survive_before * hazard
    pmf[-1] += max(0.0, 1.0 - pmf.sum())
    return pmf


def _as_result(values: np.ndarray, *inputs: Any) -> float | np.ndarray:
    return float(values) if all(np.ndim(i) == 0 for i in inputs) else values


class _ContextTable:
    """Yards-gained and turnover distributions by (down key, distance bin, field bin).

    For each cell: the share of plays that end in a turnover, and the Kaplan-Meier
    distribution of yards gained on the other plays. Cells with fewer than `min_cell_n`
    plays borrow the distribution of a wider group: first the same down key and distance
    over all field positions, then the distance bin over all down keys, then everything.

    Touchdown shares are then calibrated by exact yard line (see `fit`). Turnovers are
    placed with one distribution per field bin of "yards downfield of the
    line of scrimmage where the other team takes over".
    """

    def __init__(
        self, n_keys: int, min_cell_n: int = 300, min_turnover_n: int = 50, td_prior: float = 30.0
    ) -> None:
        self.n_keys = n_keys
        self.min_cell_n = min_cell_n
        self.min_turnover_n = min_turnover_n
        self.td_prior = td_prior

    def fit(self, s: dict[str, np.ndarray], key: np.ndarray) -> _ContextTable:
        nd, nf = len(DIST_EDGES), len(FIELD_EDGES)
        di, fi = _bin(s["ydstogo"], DIST_EDGES), _bin(s["yardline"], FIELD_EDGES)
        gain, yl, turnover = s["gain"], s["yardline"], s["turnover"]
        self.n = np.zeros((self.n_keys, nd, nf), dtype=int)
        self.level = np.zeros((self.n_keys, nd, nf), dtype=int)
        self.pmf = np.zeros((self.n_keys, nd, nf, N_G))
        self.p_turnover = np.zeros((self.n_keys, nd, nf))

        def stats(mask: np.ndarray) -> tuple[np.ndarray, float]:
            keep = mask & ~turnover
            return km_gain_pmf(gain[keep], yl[keep]), float(turnover[mask].mean())

        everything = np.ones(gain.size, dtype=bool)
        if gain.size == 0:
            raise ValueError("no scrimmage plays to fit on")
        all_stats = stats(everything)
        for j in range(nd):
            mj = di == j
            dist_stats = stats(mj) if mj.sum() >= self.min_cell_n else None
            for k in range(self.n_keys):
                mkj = mj & (key == k)
                key_stats = stats(mkj) if mkj.sum() >= self.min_cell_n else None
                for f in range(nf):
                    m = mkj & (fi == f)
                    self.n[k, j, f] = m.sum()
                    if m.sum() >= self.min_cell_n:
                        chosen, level = stats(m), 0
                    elif key_stats is not None:
                        chosen, level = key_stats, 1
                    elif dist_stats is not None:
                        chosen, level = dist_stats, 2
                    else:
                        chosen, level = all_stats, 3
                    self.pmf[k, j, f], self.p_turnover[k, j, f] = chosen
                    self.level[k, j, f] = level

        # Touchdown calibration. Kaplan-Meier assumes the yards a play "would" gain do
        # not depend on where in the bin it starts. Near the goal line that is not
        # quite true (the defense has less field to cover), and the cells come out with
        # a few too many touchdowns. So for each exact yard line, compare the touchdowns
        # the cells predict for the fitted plays with the touchdowns that happened, and
        # keep the ratio. `td_prior` pseudo-touchdowns on both sides pull the ratio
        # toward 1 where touchdowns are few, so it does not chase noise.
        tail = self.pmf[..., ::-1].cumsum(axis=-1)[..., ::-1]  # P(gain >= g)
        clean = ~turnover
        predicted = tail[key[clean], di[clean], fi[clean], yl[clean] - G_MIN]
        pred_sum = np.bincount(yl[clean], weights=predicted, minlength=100)
        actual_sum = np.bincount(yl[clean], weights=s["off_td"][clean].astype(float), minlength=100)
        self.td_ratio = (actual_sum + self.td_prior) / (pred_sum + self.td_prior)
        self.td_predicted, self.td_actual = pred_sum, actual_sum
        self.plays_by_yardline = np.bincount(yl[clean], minlength=100)

        # Where the other team takes over after a turnover, by field bin.
        located = turnover & ~s["def_td"] & ~np.isnan(s["opp_next"])
        t = np.where(located, yl - (100 - np.nan_to_num(s["opp_next"])), 0).astype(int)
        self.turnover_hist = np.zeros((nf, _T.size))
        self.turnover_def_td = np.zeros(nf)
        self.turnover_n = np.zeros(nf, dtype=int)
        pooled_hist = np.bincount(t[located] + 99, minlength=_T.size).astype(float)
        pooled_td = float(s["def_td"][turnover].mean()) if turnover.any() else 0.0
        for f in range(nf):
            m = fi == f
            self.turnover_n[f] = int((m & turnover).sum())
            if (m & located).sum() >= self.min_turnover_n:
                hist = np.bincount(t[m & located] + 99, minlength=_T.size).astype(float)
                p_td = float(s["def_td"][m & turnover].mean())
            else:
                hist, p_td = pooled_hist, pooled_td
            self.turnover_hist[f] = hist / hist.sum() if hist.sum() > 0 else hist
            self.turnover_def_td[f] = p_td
        return self

    def cell(self, key: int, ydstogo: int, yardline_100: int) -> tuple[np.ndarray, float]:
        """(gain distribution over GAINS for plays without a turnover, turnover share).

        The share of the distribution that is a touchdown from this yard line is
        multiplied by the calibration ratio and the rest is rescaled to keep the sum 1.
        """
        j, f = int(_bin(ydstogo, DIST_EDGES)), int(_bin(yardline_100, FIELD_EDGES))
        pmf = self.pmf[key, j, f]
        scores = GAINS >= yardline_100
        p_td = float(pmf[scores].sum())
        if 0.0 < p_td < 1.0:
            new_td = min(p_td * float(self.td_ratio[int(yardline_100)]), 0.999)
            pmf = np.where(scores, pmf * new_td / p_td, pmf * (1.0 - new_td) / (1.0 - p_td))
        return pmf, float(self.p_turnover[key, j, f])

    def turnover_dist(self, yardline_100: int) -> np.ndarray:
        """Given a turnover: distribution of the other team's yardline_100 (length 100,
        index 0 = returned for a touchdown). Sums to 1."""
        f = int(_bin(yardline_100, FIELD_EDGES))
        hist, p_td = self.turnover_hist[f], self.turnover_def_td[f]
        out = np.zeros(100)
        if hist.sum() == 0:
            out[100 - yardline_100] = 1.0
            return out
        opp = 100 - yardline_100 + _T
        opp = np.where(opp >= 100, 80, opp)  # taken over in the end zone: touchback
        opp = np.maximum(opp, 0)  # returned past the goal line: touchdown
        np.add.at(out, opp, (1.0 - p_td) * hist)
        out[0] += p_td
        return out


def _spread_gains(pmf: np.ndarray, yardline_100: int) -> tuple[float, float, np.ndarray]:
    """Apply a gain distribution at one field position.

    Returns (touchdown probability, safety probability, array of length 100 with the
    probability of the offense holding the ball at each new yardline_100).
    """
    new = yardline_100 - GAINS
    keep = np.zeros(100)
    inside = (new >= 1) & (new <= 99)
    keep[new[inside]] = pmf[inside]
    return float(pmf[new <= 0].sum()), float(pmf[new >= 100].sum()), keep


def _fit_logit(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    # C is large, so this is close to plain maximum likelihood; the tiny penalty only
    # keeps the solver stable when a feature is nearly constant (small synthetic data).
    return LogisticRegression(C=1000.0, max_iter=5000).fit(x, y)


# --------------------------------------------------------------------------------------
# 1. Fourth-down go model
# --------------------------------------------------------------------------------------


class GoModel:
    """Conversion probability and outcome distribution when going for it.

    P(convert) is a logistic regression fitted on third AND fourth downs together:
        log-odds = a + spline(d) + spline(yardline_100) + b1*fourth + b2*season
                   + b3*spread          [+ b4*fourth*log(d), off by default]
    where d is yards to go and a spline is a smooth curve built from a few bumps.
    and evaluated with fourth = 1, at the last season fitted, for an evenly matched
    game (spread 0). The distance spline has knots at 1, 2, 3 and 4 yards so it can
    follow the sharp drop from one yard to go (the quarterback sneak) to two. Third
    downs give the shape in distance and field position (ten
    times more plays, and every team has to run them); the fourth-down term shifts it.
    The pregame spread stands in for team strength, so that the fourth-down term is not
    just picking up which teams choose to go. "Convert" means a first down or touchdown
    without losing the ball. The fitted table is forced to be non-increasing in distance.

    Where the ball ends up comes from the pooled third-and-fourth-down gain
    distribution for the cell, split at the line to gain and rescaled so the two sides
    carry exactly P(convert) and 1 - P(convert).
    """

    def __init__(
        self,
        max_ydstogo: int = 30,
        season_trend: bool = True,
        fourth_by_distance: bool = False,
        use_spread: bool = True,
        distance_knots: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 10, 13, 17, 22, 30),
        fit_downs: tuple[int, ...] = (3, 4),
        yardline_knots: tuple[int, ...] = (1, 5, 10, 20, 40, 60, 80, 99),
        min_cell_n: int = 300,
        min_turnover_n: int = 100,
        era_first_season: int | None = None,
    ) -> None:
        self.max_ydstogo = max_ydstogo
        self.season_trend = season_trend
        self.fourth_by_distance = fourth_by_distance
        self.use_spread = use_spread
        self.distance_knots = tuple(distance_knots)
        self.fit_downs = tuple(fit_downs)
        self.yardline_knots = tuple(yardline_knots)
        self.min_cell_n = min_cell_n
        self.min_turnover_n = min_turnover_n
        self.era_first_season = era_first_season

    def _features(
        self,
        yl: np.ndarray,
        d: np.ndarray,
        fourth: np.ndarray,
        t: np.ndarray,
        spread: np.ndarray,
    ) -> np.ndarray:
        d = np.clip(d, 1, self.max_ydstogo).astype(float)
        d_basis = self._d_spline.transform(d.reshape(-1, 1))
        cols = [d_basis]
        self.feature_names_ = [f"distance_spline_{i}" for i in range(d_basis.shape[1])]
        if len(self.fit_downs) > 1:
            cols.append(fourth)
            self.feature_names_.append("fourth_down")
        if self.use_spread:
            cols.append(spread / 10.0)
            self.feature_names_.append("spread_per_10")
        if self.fourth_by_distance and len(self.fit_downs) > 1:
            cols.append(fourth * np.log(d))
            self.feature_names_.append("fourth_down_x_log_distance")
        if self.season_trend:
            cols.append(t)
            self.feature_names_.append("season_per_decade")
        basis = self._spline.transform(np.asarray(yl, dtype=float).reshape(-1, 1))
        self.feature_names_ += [f"yardline_spline_{i}" for i in range(basis.shape[1])]
        return np.column_stack([*cols, basis])

    def fit(self, plays: pl.DataFrame) -> GoModel:
        s = _scrimmage(plays, 3, 4)
        fourth = (s["down"] == 4).astype(float)
        converted = (s["first_down"] | s["off_td"]) & ~s["turnover"]
        if "season" not in plays.columns:
            self.season_trend = False
        if "posteam_spread" not in plays.columns:
            self.use_spread = False
        in_fit = np.isin(s["down"], self.fit_downs)
        self.reference_season_ = float(s["season"].max())
        self._spline = SplineTransformer(
            degree=3,
            knots=np.array(self.yardline_knots, dtype=float).reshape(-1, 1),
            include_bias=False,
        ).fit(np.array([[1.0], [99.0]]))
        self._d_spline = SplineTransformer(
            degree=3,
            knots=np.array(self.distance_knots, dtype=float).reshape(-1, 1),
            include_bias=False,
        ).fit(np.array([[1.0], [float(self.max_ydstogo)]]))
        x = self._features(
            s["yardline"], s["ydstogo"], fourth, _season_t(s["season"]), s["spread"]
        )
        model = _fit_logit(x[in_fit], converted[in_fit])
        self.intercept_ = float(model.intercept_[0])
        self.coef_ = dict(zip(self.feature_names_, model.coef_[0].tolist(), strict=True))
        self.n_third_ = int((fourth == 0).sum())
        self.n_fourth_ = int((fourth == 1).sum())

        # Lookup tables [down 3 or 4][yardline_100 0..99][ydstogo 0..max], at the
        # reference season. Row and column 0 are copies of 1 so any index is safe.
        yl_grid, d_grid = np.meshgrid(
            np.clip(np.arange(100), 1, 99),
            np.clip(np.arange(self.max_ydstogo + 1), 1, None),
            indexing="ij",
        )
        t_ref = np.full(yl_grid.size, _season_t(self.reference_season_))
        even = np.zeros(yl_grid.size)
        self.table_ = {}
        self.monotone_fix_ = {}
        for down in (3, 4):
            is4 = np.full(yl_grid.size, float(down == 4))
            x_grid = self._features(yl_grid.ravel(), d_grid.ravel(), is4, t_ref, even)
            raw = model.predict_proba(x_grid)[:, 1].reshape(yl_grid.shape)
            fixed = np.minimum.accumulate(raw, axis=1)
            self.table_[down] = fixed
            self.monotone_fix_[down] = float(np.abs(fixed - raw).max())

        # Gain and turnover distributions, third and fourth downs pooled (recent era).
        era = s["season"] >= (self.era_first_season or -np.inf)
        s_era = {k: v[era] for k, v in s.items()}
        self.context_ = _ContextTable(1, self.min_cell_n).fit(
            s_era, np.zeros(era.sum(), dtype=int)
        )

        # Share of failed fourth downs that are an interception or lost fumble.
        failed4 = (s_era["down"] == 4) & ~converted[era]
        group = _bin(s_era["ydstogo"], TURNOVER_DIST_EDGES)
        pooled = float(s_era["turnover"][failed4].mean()) if failed4.any() else 0.0
        self.turnover_share_ = np.full(len(TURNOVER_DIST_EDGES), pooled)
        self.turnover_share_n_ = np.zeros(len(TURNOVER_DIST_EDGES), dtype=int)
        for g in range(len(TURNOVER_DIST_EDGES)):
            m = failed4 & (group == g)
            self.turnover_share_n_[g] = int(m.sum())
            if m.sum() >= self.min_turnover_n:
                self.turnover_share_[g] = float(s_era["turnover"][m].mean())
        return self

    def p_convert(
        self,
        yardline_100: int | np.ndarray,
        ydstogo: int | np.ndarray,
        down: int = 4,
        spread: float | np.ndarray = 0.0,
    ) -> float | np.ndarray:
        """P(first down or touchdown, keeping the ball). `down=3` gives the third-down
        curve of the same model, for comparison. `spread` is the points the offense was
        favored by before the game (0 = evenly matched, the default everywhere else)."""
        yl = np.clip(np.asarray(yardline_100, dtype=int), 1, 99)
        d = np.clip(np.asarray(ydstogo, dtype=int), 1, self.max_ydstogo)
        p = self.table_[down][yl, d]
        shift = self.coef_.get("spread_per_10", 0.0) * np.asarray(spread, dtype=float) / 10.0
        if np.any(shift != 0):
            p = 1.0 / (1.0 + np.exp(-(np.log(p / (1.0 - p)) + shift)))
        return _as_result(p, yardline_100, ydstogo, spread)

    def outcomes(self, yardline_100: int, ydstogo: int) -> dict[str, Any]:
        """Full outcome distribution of a fourth-down attempt. See
        `ComponentModels.go_outcomes`."""
        yl, d = int(yardline_100), int(min(max(ydstogo, 1), yardline_100))
        p = float(self.p_convert(yl, d))
        pmf, _ = self.context_.cell(0, d, yl)
        made = np.where(GAINS >= d, pmf, 0.0)
        short = np.where(GAINS < d, pmf, 0.0)
        if made.sum() <= 0:  # no observed conversion in the cell: gain exactly the distance
            made = (GAINS == d).astype(float)
        if short.sum() <= 0:  # no observed failure in the cell: stopped for no gain
            short = (GAINS == min(0, d - 1)).astype(float)
        share_to = float(self.turnover_share_[int(_bin(d, TURNOVER_DIST_EDGES))])

        p_td, _, first_down = _spread_gains(p * made / made.sum(), yl)
        fail_no_to = (1.0 - p) * (1.0 - share_to)
        _, p_safety, stopped_at = _spread_gains(fail_no_to * short / short.sum(), yl)
        turnover = np.zeros(100)
        turnover[1:] = stopped_at[:0:-1]  # offense at y  ->  opponent at 100 - y
        turnover += (1.0 - p) * share_to * self.context_.turnover_dist(yl)
        return {
            "p_convert": p,
            "p_touchdown": p_td,
            "first_down": first_down,
            "turnover": turnover,
            "p_safety": p_safety,
        }


# --------------------------------------------------------------------------------------
# 2. Field goal model
# --------------------------------------------------------------------------------------


class FieldGoalModel:
    """P(make) by kick distance, with a season trend.

    With x = (kick distance - 40) / 10 and t = (season - 2018) / 10:
        log-odds = a + b1*x + b2*x^2 + b3*x^3 + b4*t
    evaluated at the last season fitted. The cubic lets the curve fall off a cliff past
    55 yards, as the data does. Blocked kicks count as misses. P(make) is 0 beyond
    `max_kick_distance`.
    """

    def __init__(
        self,
        snap_to_kick_yards: int = 18,
        miss_spot_yards: int = 8,
        miss_min_own_yardline: int = 20,
        season_trend: bool = True,
        max_kick_distance: int = 68,
        first_season: int | None = None,
    ) -> None:
        self.snap_to_kick_yards = snap_to_kick_yards
        self.miss_spot_yards = miss_spot_yards
        self.miss_min_own_yardline = miss_min_own_yardline
        self.season_trend = season_trend
        self.max_kick_distance = max_kick_distance
        self.first_season = first_season

    def _features(self, distance: np.ndarray, t: np.ndarray) -> np.ndarray:
        x = (np.asarray(distance, dtype=float) - 40.0) / 10.0
        cols = [x, x**2, x**3]
        self.feature_names_ = ["distance_per_10", "distance_per_10_squared",
                               "distance_per_10_cubed"]  # fmt: skip
        if self.season_trend:
            cols.append(t)
            self.feature_names_.append("season_per_decade")
        return np.column_stack(cols)

    def fit(self, plays: pl.DataFrame) -> FieldGoalModel:
        df = _season_filter(plays, self.first_season).filter(
            (pl.col("play_type") == "field_goal")
            & pl.col("yardline_100").is_between(1, 99)
            & pl.col("field_goal_result").is_not_null()
        )
        if "season" not in df.columns:
            self.season_trend = False
        season = _num(df, "season", _SEASON_ZERO)
        distance = _num(df, "yardline_100") + self.snap_to_kick_yards
        made = (df["field_goal_result"] == "made").to_numpy()
        self.n_ = int(made.size)
        self.longest_made_ = int(distance[made].max()) if made.any() else 0
        self.reference_season_ = float(season.max())
        model = _fit_logit(self._features(distance, _season_t(season)), made)
        self.intercept_ = float(model.intercept_[0])
        self.coef_ = dict(zip(self.feature_names_, model.coef_[0].tolist(), strict=True))

        grid = np.arange(100) + self.snap_to_kick_yards  # index = yardline_100
        t_ref = np.full(grid.size, _season_t(self.reference_season_))
        raw = model.predict_proba(self._features(grid, t_ref))[:, 1]
        raw[grid > self.max_kick_distance] = 0.0
        fixed = np.minimum.accumulate(raw)  # non-increasing in distance
        self.monotone_fix_ = float(np.abs(fixed - raw).max())
        self.table_ = fixed
        return self

    def p_make(
        self, yardline_100: int | np.ndarray, season: int | None = None
    ) -> float | np.ndarray:
        """P(make) from this line of scrimmage. By default at the last season fitted;
        pass `season` to read the fitted trend at another season."""
        yl = np.clip(np.asarray(yardline_100, dtype=int), 1, 99)
        if season is None:
            return _as_result(self.table_[yl], yardline_100)
        distance = yl + self.snap_to_kick_yards
        x = self._features(distance, np.full(np.shape(distance), _season_t(float(season))))
        logit = self.intercept_ + x @ np.array([self.coef_[k] for k in self.feature_names_])
        p = np.where(distance > self.max_kick_distance, 0.0, 1.0 / (1.0 + np.exp(-logit)))
        return _as_result(p.reshape(np.shape(yl)), yardline_100)

    def miss_yardline(self, yardline_100: int) -> int:
        """Opponent's yardline_100 after a miss: they take over at the spot of the kick
        (8 yards behind the line of scrimmage) or their own 20, whichever is farther
        from their goal line."""
        spot = 100 - (int(yardline_100) + self.miss_spot_yards)
        return int(np.clip(spot, 1, 100 - self.miss_min_own_yardline))


# --------------------------------------------------------------------------------------
# 3. Punt model
# --------------------------------------------------------------------------------------


class PuntModel:
    """Distribution of where the other team starts after a punt, for each yardline_100.

    The target is the receiving team's field position at its next snap, so returns,
    touchbacks, blocks, shanks and penalties on the play are all included. Punts from
    nearby field positions are pooled with a Gaussian kernel. From deep (yardline_100 at
    or above `landing_below`) the pooled quantity is the NET yards, because a punter
    with room kicks about the same distance from anywhere (touchbacks are pooled as
    their own outcome and always land on the 20). Closer in, the pooled quantity is the
    LANDING SPOT, because the punter aims at the same patch near the goal line from
    anywhere.
    """

    def __init__(
        self,
        bandwidth: float = 2.0,
        min_effective_n: int = 400,
        landing_below: int = 50,
        first_season: int | None = None,
    ) -> None:
        self.bandwidth = bandwidth
        self.min_effective_n = min_effective_n
        self.landing_below = landing_below
        self.first_season = first_season

    def fit(self, plays: pl.DataFrame) -> PuntModel:
        df = add_next_state(_season_filter(plays, self.first_season)).filter(
            (pl.col("play_type") == "punt") & pl.col("yardline_100").is_between(1, 99)
        )
        yl = _num(df, "yardline_100").astype(int)
        opp_next = _opponent_next_yardline(df)
        scored = _num(df, "touchdown") == 1
        return_td = _flag(df, _DEF_TD)
        same_half = _flag(df, pl.col("next_half") == pl.col("game_half"))
        kept = (
            _flag(df, pl.col("next_posteam") == pl.col("posteam"))
            & same_half & ~scored & (_num(df, "safety") != 1)
        )  # fmt: skip
        to_opp = ~np.isnan(opp_next) & ~scored & (_num(df, "safety") != 1)
        land = np.where(to_opp, np.nan_to_num(opp_next), 0).astype(int)
        net = land - (100 - yl)

        self.n_ = int(yl.size)
        self.n_used_ = int((to_opp | return_td).sum())
        self.rates_ = {
            "return_touchdown": float(return_td.mean()),
            "kicking_team_keeps_ball": float(kept.mean()),
            "blocked": float((_num(df, "punt_blocked") == 1).mean()),
            "touchback": float((_num(df, "touchback") == 1).mean()),
            "dropped_end_of_half_safety_or_kicking_team_score": float(
                (~(to_opp | return_td | kept)).mean()
            ),
        }

        # Counts by source field position (rows 0..99).
        touchback = to_opp & (_num(df, "touchback") == 1)
        returned = to_opp & ~touchback
        net_idx = np.clip(net, -100, 99) + 100
        c_net = np.zeros((100, 200))
        np.add.at(c_net, (yl[returned], net_idx[returned]), 1.0)
        c_tb = np.bincount(yl[touchback], minlength=100).astype(float)
        c_land = np.zeros((100, 100))
        np.add.at(c_land, (yl[to_opp], land[to_opp]), 1.0)
        c_td = np.bincount(yl[return_td], minlength=100).astype(float)
        c_all = c_land.sum(axis=1) + c_td
        if c_all.sum() == 0:
            raise ValueError("no punts to fit on")
        self.raw_n_ = c_all.astype(int)
        raw_sum = c_land @ np.arange(100)
        self.raw_expected_ = np.divide(raw_sum, c_all, out=np.full(100, np.nan), where=c_all > 0)

        source = np.arange(100)
        self.table_ = np.zeros((100, 100))
        self.bandwidth_used_ = np.zeros(100)
        for y in range(1, 100):
            h = self.bandwidth
            while True:
                w = np.exp(-0.5 * ((source - y) / h) ** 2)
                if w @ c_all >= min(self.min_effective_n, 0.9 * c_all.sum()) or h > 200:
                    break
                h *= 1.5
            self.bandwidth_used_[y] = h
            out = np.zeros(100)
            out[0] = w @ c_td
            if y >= self.landing_below:
                pooled_net = w @ c_net
                spot = np.clip(100 - y + (np.arange(200) - 100), 1, 99)
                np.add.at(out, spot, pooled_net)
                out[80] += w @ c_tb
            else:
                out += w @ c_land
            self.table_[y] = out / out.sum()
        self.table_[0] = self.table_[1]

        # The kicking team keeps the ball (muffed or fumbled return): rare, one pooled
        # rate and one pooled distribution of how far downfield they recover it.
        self.p_kept_ = float(kept.sum() / max(1, (kept | to_opp | return_td).sum()))
        gained = yl[kept] - _num(df, "next_yardline_100")[kept].astype(int)
        self.kept_gain_hist_ = np.bincount(np.clip(gained, 0, 99), minlength=100).astype(float)
        if self.kept_gain_hist_.sum() > 0:
            self.kept_gain_hist_ /= self.kept_gain_hist_.sum()
        return self

    def outcomes(self, yardline_100: int) -> np.ndarray:
        """See `ComponentModels.punt_outcomes`."""
        return self.table_[int(np.clip(yardline_100, 1, 99))].copy()

    def outcomes_full(self, yardline_100: int) -> dict[str, np.ndarray]:
        """See `ComponentModels.punt_outcomes_full`."""
        y = int(np.clip(yardline_100, 1, 99))
        kicking = np.zeros(100)
        if self.kept_gain_hist_.sum() > 0:
            np.add.at(kicking, np.clip(y - np.arange(100), 1, 99), self.kept_gain_hist_)
        p_kept = self.p_kept_ if kicking.sum() > 0 else 0.0
        return {"opponent": (1.0 - p_kept) * self.table_[y], "kicking_team": p_kept * kicking}

    def expected_opponent_yardline(self) -> np.ndarray:
        """Mean of the receiving team's yardline_100 for each punting yardline_100
        (index 0 unused). A return touchdown counts as 0."""
        return self.table_ @ np.arange(100)


# --------------------------------------------------------------------------------------
# 4. Generic play model for downs 1 to 3
# --------------------------------------------------------------------------------------


class PlayModel:
    """One "typical" offensive play on downs 1 to 3, runs and passes mixed in the
    proportion teams actually call them in that context."""

    def __init__(
        self,
        min_cell_n: int = 300,
        min_turnover_n: int = 50,
        first_season: int | None = None,
    ) -> None:
        self.min_cell_n = min_cell_n
        self.min_turnover_n = min_turnover_n
        self.first_season = first_season

    def fit(self, plays: pl.DataFrame) -> PlayModel:
        s = _scrimmage(_season_filter(plays, self.first_season), 1, 3)
        self.n_ = int(s["down"].size)
        self.context_ = _ContextTable(3, self.min_cell_n, self.min_turnover_n).fit(
            s, s["down"] - 1
        )
        return self

    def outcomes(self, down: int, ydstogo: int, yardline_100: int) -> dict[str, Any]:
        """See `ComponentModels.play_outcomes`."""
        if down not in (1, 2, 3):
            raise ValueError("play_outcomes covers downs 1 to 3; use go_outcomes on fourth down")
        yl, d = int(yardline_100), int(min(max(ydstogo, 1), yardline_100))
        pmf, p_to = self.context_.cell(down - 1, d, yl)
        p_td, p_safety, keep = _spread_gains((1.0 - p_to) * pmf, yl)
        first_down = float(keep[: max(yl - d, 0) + 1].sum()) + p_td
        return {
            "p_touchdown": p_td,
            "p_safety": p_safety,
            "keep": keep,
            "turnover": p_to * self.context_.turnover_dist(yl),
            "p_first_down": first_down,
            "p_turnover": p_to,
        }


# --------------------------------------------------------------------------------------
# Small count-based pieces: kickoff, clock, conversions after a touchdown
# --------------------------------------------------------------------------------------


def fit_kickoff(plays: pl.DataFrame, seasons: list[int] | None) -> tuple[np.ndarray, dict]:
    """Distribution of the receiving team's yardline_100 at its first snap after a
    kickoff (index 0 = kick returned for a touchdown), given that the receiving team
    ends up with the ball. If none of `seasons` is in the data, the last season is used.
    """
    df = add_next_state(plays).filter(pl.col("play_type") == "kickoff")
    if "season" in df.columns and df.height:
        available = set(df["season"].unique().to_list())
        use = [s for s in (seasons or []) if s in available] or [max(available)]
        df = df.filter(pl.col("season").is_in(use))
    else:
        use = []
    # On a kickoff row the posteam is the receiving team.
    return_td = _flag(df, _OFF_TD)
    scored = _num(df, "touchdown") == 1
    same_half = _flag(df, pl.col("next_half") == pl.col("game_half"))
    received = _flag(df, pl.col("next_posteam") == pl.col("posteam")) & same_half & ~scored
    kicking_team = (
        _flag(df, pl.col("next_posteam") != pl.col("posteam")) & same_half & ~scored
    ) | (scored & ~return_td)
    start = _num(df, "next_yardline_100")[received].astype(int)
    hist = np.bincount(start, minlength=100).astype(float)[:100]
    hist[0] = return_td.sum()
    info = {
        "seasons": [int(s) for s in use],
        "n": int(df.height),
        "n_used": int(hist.sum()),
        "kicking_team_recovers_or_scores": float(kicking_team.mean()) if df.height else 0.0,
        "return_touchdown": float(return_td.mean()) if df.height else 0.0,
    }
    if hist.sum() == 0:
        hist[75] = 1.0
    return hist / hist.sum(), info


def fit_seconds(plays: pl.DataFrame, first_season: int | None) -> dict[str, Any]:
    """Mean game-clock seconds from one snap to the next, by play type and by whether
    the clock is likely to keep running after the play.

    "Clock stops" means the play was an incomplete pass, went out of bounds, scored or
    turned the ball over. "Two-minute" means the snap came with 120 seconds or less left
    in the half. Timeouts are not removed, so "clock runs" means are a little low.
    """
    df = add_next_state(_season_filter(plays, first_season)).filter(
        pl.col("play_type").is_in(["run", "pass", "punt", "field_goal", "kickoff"])
        & pl.col("game_seconds_remaining").is_not_null()
    )
    if df.height == 0:
        return {"mean_scrimmage_play": float("nan"), "by_type": [], "kicks": {}}
    desc = pl.col("desc").fill_null("") if "desc" in df.columns else pl.lit("")
    same_half = (pl.col("next_row_half") == pl.col("game_half")).fill_null(False)
    elapsed = (
        pl.when(same_half & pl.col("next_row_seconds").is_not_null())
        .then(pl.col("game_seconds_remaining") - pl.col("next_row_seconds"))
        .otherwise(pl.col("half_seconds_remaining"))
        .clip(0, 60)
    )
    sack = pl.col("sack").fill_null(0) == 1 if "sack" in df.columns else pl.lit(False)
    incomplete = desc.str.contains("incomplete")
    stops = (
        incomplete
        | desc.str.contains(r"\bob\b|out of bounds")
        | (pl.col("touchdown").fill_null(0) == 1)
        | _TURNOVER.fill_null(False)
    )
    df = df.with_columns(
        elapsed.alias("_sec"),
        pl.when(pl.col("play_type") == "run").then(pl.lit("run"))
        .when(pl.col("play_type") != "pass").then(pl.col("play_type"))
        .when(sack).then(pl.lit("sack"))
        .when(incomplete).then(pl.lit("pass_incomplete"))
        .otherwise(pl.lit("pass_complete"))
        .alias("_kind"),
        pl.when(stops).then(pl.lit("stops")).otherwise(pl.lit("runs")).alias("_clock"),
        pl.when(pl.col("half_seconds_remaining") <= 120)
        .then(pl.lit("two_minute")).otherwise(pl.lit("normal")).alias("_phase"),
    ).filter(pl.col("_sec").is_not_null())  # fmt: skip
    scrim = df.filter(pl.col("play_type").is_in(["run", "pass"]))
    by_type = (
        scrim.group_by("_kind", "_clock", "_phase")
        .agg(pl.len().alias("n"), pl.col("_sec").mean().alias("mean_seconds"))
        .sort("_kind", "_clock", "_phase")
        .rename({"_kind": "play", "_clock": "clock", "_phase": "phase"})
    )
    kicks = df.filter(~pl.col("play_type").is_in(["run", "pass"])).group_by("play_type").agg(
        pl.len().alias("n"), pl.col("_sec").mean().alias("mean_seconds")
    )
    by_clock = scrim.group_by("_clock").agg(pl.col("_sec").mean().alias("m"))
    return {
        "mean_scrimmage_play": float(scrim["_sec"].mean()) if scrim.height else float("nan"),
        "mean_by_clock": {r["_clock"]: float(r["m"]) for r in by_clock.to_dicts()},
        "by_type": by_type.to_dicts(),
        "kicks": {
            r["play_type"]: {"n": r["n"], "mean_seconds": r["mean_seconds"]}
            for r in kicks.sort("play_type").to_dicts()
        },
    }


def fit_conversions(plays: pl.DataFrame, first_season: int | None) -> dict[str, Any]:
    """Extra-point and two-point success rates."""
    df = _season_filter(plays, first_season)
    xp = df.filter(
        (pl.col("play_type") == "extra_point") & pl.col("extra_point_result").is_not_null()
    )
    two = df.filter(
        (pl.col("two_point_attempt") == 1) & pl.col("play_type").is_in(["run", "pass"])
    )
    two_ok = two["desc"].fill_null("").str.contains("ATTEMPT SUCCEEDS") if two.height else None
    return {
        "p_extra_point": float((xp["extra_point_result"] == "good").mean()) if xp.height else 0.94,
        "n_extra_point": xp.height,
        "p_two_point": float(two_ok.mean()) if two.height else 0.48,
        "n_two_point": two.height,
    }


# --------------------------------------------------------------------------------------
# 5. Drive-level validation targets (real data, for the simulator check in M4)
# --------------------------------------------------------------------------------------

DRIVE_OUTCOMES = {
    "Touchdown": "touchdown",
    "Field goal": "field_goal",
    "Missed field goal": "missed_field_goal",
    "Punt": "punt",
    "Turnover": "turnover",
    "Opp touchdown": "opp_touchdown",
    "Turnover on downs": "downs",
    "Safety": "safety",
    "End of half": "end_of_half",
}
# Start-of-drive bins by yardline_100 (lower edge inclusive, upper exclusive).
DRIVE_START_BINS = {
    "own_1_to_20": (80, 100),
    "own_21_to_40": (60, 80),
    "own_41_to_opp_41": (41, 60),
    "opp_40_to_21": (21, 41),
    "opp_20_to_goal": (1, 21),
}


def drive_targets(plays: pl.DataFrame, first_season: int | None = None) -> dict[str, Any]:
    """Real drive statistics the simulator must reproduce.

    A drive is one `fixed_drive` of one game. Its start is the first snap with a down.
    `plays` counts runs and passes (including kneels and spikes), not the kick that ends
    the drive. `points` are the offense's points on the drive (6 per touchdown plus the
    actual extra point or two-point result, 3 per field goal). `opp_touchdown` (a
    turnover returned for a score) and `safety` are kept as their own outcomes.
    """
    df = _season_filter(plays, first_season).filter(pl.col("fixed_drive").is_not_null())
    off_td = _OFF_TD.fill_null(False)
    two_ok = (
        (pl.col("two_point_attempt") == 1)
        & pl.col("play_type").is_in(["run", "pass"])
        & pl.col("desc").fill_null("").str.contains("ATTEMPT SUCCEEDS")
    ).fill_null(False)
    points = (
        6 * off_td.cast(pl.Int32)
        + (pl.col("extra_point_result") == "good").fill_null(False).cast(pl.Int32)
        + 2 * two_ok.cast(pl.Int32)
        + 3 * (pl.col("field_goal_result") == "made").fill_null(False).cast(pl.Int32)
    )
    snap = pl.col("down").is_not_null() & pl.col("yardline_100").is_not_null()
    scrimmage = pl.col("play_type").is_in(["run", "pass", "qb_kneel", "qb_spike"]) & snap
    drives = (
        df.sort("game_id", "order_sequence", "play_id")
        .group_by("game_id", "fixed_drive", maintain_order=True)
        .agg(
            pl.col("fixed_drive_result").drop_nulls().first().alias("result"),
            pl.col("yardline_100").filter(snap).first().alias("start"),
            scrimmage.sum().alias("plays"),
            points.sum().alias("points"),
        )
        .filter(pl.col("start").is_not_null() & pl.col("result").is_not_null())
        .with_columns(pl.col("result").replace_strict(DRIVE_OUTCOMES, default="other"))
    )

    def summary(d: pl.DataFrame) -> dict[str, Any]:
        share = {k: 0.0 for k in DRIVE_OUTCOMES.values()}
        if d.height:
            for row in d.group_by("result").len().to_dicts():
                share[row["result"]] = row["len"] / d.height
        counts = np.bincount(d["plays"].to_numpy().astype(int), minlength=21)
        plays_dist = np.concatenate([counts[:20], [counts[20:].sum()]]) / max(1, d.height)
        return {
            "n_drives": d.height,
            "outcome_share": share,
            "points_per_drive": float(d["points"].mean()) if d.height else float("nan"),
            "plays_per_drive": float(d["plays"].mean()) if d.height else float("nan"),
            "mean_start_yardline_100": float(d["start"].mean()) if d.height else float("nan"),
            "plays_per_drive_distribution_0_to_20_plus": plays_dist.tolist(),
        }

    out = summary(drives)
    out["by_start"] = {
        name: summary(drives.filter(pl.col("start").is_between(lo, hi, closed="left")))
        for name, (lo, hi) in DRIVE_START_BINS.items()
    }
    games = df["game_id"].n_unique()
    out["n_games"] = games
    out["drives_per_game"] = drives.height / max(1, games)
    out["scrimmage_plays_per_game"] = float(drives["plays"].sum()) / max(1, games)
    if "season" in df.columns and df.height:
        out["seasons"] = [int(df["season"].min()), int(df["season"].max())]
    return out


# --------------------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------------------


class ComponentModels:
    """All play-outcome models behind one interface.

    Usage:
        models = ComponentModels().fit(plays, seed=20261117)
        models.save()
        models = ComponentModels.load()
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = copy.deepcopy(config) if config is not None else load_config()

    def fit(self, plays: pl.DataFrame, seed: int | None = None) -> ComponentModels:
        """Fit every component on `plays` (the state table, or any subset of games).

        `seed` is recorded on the bundle. Nothing in the fit is random, so two fits on
        the same plays give identical models whatever the seed.
        """
        cfg = self.config
        era = cfg.get("era_first_season")
        plays = add_next_state(plays)
        self.seed = cfg.get("seed") if seed is None else seed
        self.go = GoModel(era_first_season=era, **cfg.get("go", {})).fit(plays)
        self.field_goal = FieldGoalModel(**cfg.get("field_goal", {})).fit(plays)
        self.punt = PuntModel(first_season=era, **cfg.get("punt", {})).fit(plays)
        self.play = PlayModel(first_season=era, **cfg.get("play", {})).fit(plays)
        self.kickoff, self.kickoff_info = fit_kickoff(plays, cfg.get("kickoff", {}).get("seasons"))
        self.seconds = fit_seconds(plays, era)
        self.conversions = fit_conversions(plays, era)
        if "season" in plays.columns:
            self.seasons = [int(plays["season"].min()), int(plays["season"].max())]
        else:
            self.seasons = []
        self.n_plays = plays.height
        return self

    # ---- fourth down: go ----
    def p_convert(
        self, yardline_100: int | np.ndarray, ydstogo: int | np.ndarray
    ) -> float | np.ndarray:
        """P(first down or touchdown) when going for it on fourth down. Non-increasing
        in `ydstogo`. Scalars give a float, arrays give an array."""
        return self.go.p_convert(yardline_100, ydstogo)

    def go_outcomes(self, yardline_100: int, ydstogo: int) -> dict[str, Any]:
        """Outcome distribution of going for it on fourth down.

        Keys:
          p_convert    P(first down or touchdown) = p_touchdown + first_down.sum()
          p_touchdown  touchdown on the play
          first_down   length 100; first_down[y] = P(first down with the ball at the
                       offense's new yardline_100 = y)
          turnover     length 100; turnover[y] = P(the other team takes over at THEIR
                       yardline_100 = y), on downs, by interception or by fumble.
                       turnover[0] = P(they return it for a touchdown)
          p_safety     tackled in the offense's own end zone
        p_touchdown + first_down.sum() + turnover.sum() + p_safety = 1.
        """
        return self.go.outcomes(yardline_100, ydstogo)

    # ---- fourth down: field goal ----
    def p_field_goal(self, yardline_100: int | np.ndarray) -> float | np.ndarray:
        """P(make) for a kick from this line of scrimmage (kick distance is
        yardline_100 + 18). Non-increasing in yardline_100."""
        return self.field_goal.p_make(yardline_100)

    def fg_miss_yardline(self, yardline_100: int) -> int:
        """The other team's yardline_100 after a missed field goal."""
        return self.field_goal.miss_yardline(yardline_100)

    # ---- fourth down: punt ----
    def punt_outcomes(self, yardline_100: int) -> np.ndarray:
        """Length-100 distribution after a punt from `yardline_100`.

        Index y in 1..99 = P(the receiving team's next snap is at THEIR yardline_100 = y).
        Index 0 = P(punt returned for a touchdown). Sums to 1. It is conditional on the
        receiving team ending up with the ball; the roughly 1% of punts the kicking team
        recovers are left out here and included in `punt_outcomes_full`.
        """
        return self.punt.outcomes(yardline_100)

    def punt_outcomes_full(self, yardline_100: int) -> dict[str, np.ndarray]:
        """Like `punt_outcomes` but with the kicking-team recoveries.

        `opponent` is `punt_outcomes` scaled by 1 - P(kicking team keeps the ball).
        `kicking_team[y]` = P(the punting team has the ball again at its own
        yardline_100 = y). The two arrays together sum to 1.
        """
        return self.punt.outcomes_full(yardline_100)

    # ---- downs 1 to 3 ----
    def play_outcomes(self, down: int, ydstogo: int, yardline_100: int) -> dict[str, Any]:
        """Outcome distribution of one typical play on down 1, 2 or 3.

        Keys:
          p_touchdown  offensive touchdown
          p_safety     tackled in the offense's own end zone
          keep         length 100; keep[y] = P(offense still has the ball at its new
                       yardline_100 = y). It is a first down if yardline_100 - y >= ydstogo.
          turnover     length 100; turnover[y] = P(interception or lost fumble with the
                       other team taking over at THEIR yardline_100 = y), turnover[0] =
                       returned for a touchdown
          p_first_down convenience: P(first down or touchdown)
          p_turnover   convenience: turnover.sum()
        p_touchdown + p_safety + keep.sum() + turnover.sum() = 1.
        """
        return self.play.outcomes(down, ydstogo, yardline_100)

    # ---- everything else ----
    def kickoff_start(self) -> np.ndarray:
        """Length-100 distribution of the receiving team's yardline_100 at its first
        snap after a kickoff. Index 0 = kickoff returned for a touchdown. Sums to 1."""
        return self.kickoff.copy()

    def seconds_per_play(self) -> dict[str, Any]:
        """Mean seconds between snaps (see `fit_seconds`)."""
        return copy.deepcopy(self.seconds)

    def p_extra_point(self) -> float:
        return float(self.conversions["p_extra_point"])

    def p_two_point(self) -> float:
        return float(self.conversions["p_two_point"])

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path is not None else MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> ComponentModels:
        obj = joblib.load(Path(path) if path is not None else MODEL_PATH)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not hold a ComponentModels bundle")
        return obj
