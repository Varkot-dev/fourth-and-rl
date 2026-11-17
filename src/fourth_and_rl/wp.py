"""M1 win-probability model: P(the team with the ball wins the game | pre-play state).

Version 2 (the default) has two parts that add up on the log-odds scale:

1. A structural baseline, the classic normal approximation
       WP = Phi(expected margin / (sigma * sqrt(share of the game left + tau))),
   with two numbers (sigma and the clock offset tau) fitted by maximum likelihood on
   the training rows, and one rule of the game: where the team with the ball leads and
   can kneel out the clock, the baseline says the game is won. It is smooth in the
   score, the spread and the clock.
2. LightGBM trees with monotone constraints that start from the baseline
   (`init_score`) and learn corrections to it: field position, down and distance,
   timeouts, and the end-game features (`kneel_slack`, `possessions`,
   `possessions_per_minute`).

An optional calibration map on seasons the trees never saw can follow; version 2 uses
none (see configs/wp.yaml, decisions, for the development evidence). Version 1
(trees only, 10,000-play leaves, two-season Platt map) is kept as `V1` so that it can
be refitted and compared. Every feature is computed from a bare game state, so later
milestones can score hypothetical states built with `state_frame`.

Leakage guards: `fit` refuses frames holding test seasons (`splits.assert_no_test_seasons`)
and every fitted model carries a stamp of the seasons it saw (`model.stamp`).

Public API used by other milestones: FEATURES, make_features, WPModel, state_frame.
The teaching notes for this module are in docs/notes/m1_wp.md.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import yaml
from scipy.optimize import minimize
from scipy.special import log_ndtr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from fourth_and_rl import splits
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

# Version 1 model inputs, in order. Kept so version 1 can be refitted for comparison.
FEATURES_V1: list[str] = [
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

# Version 2 model inputs: version 1 without `is_home` (home advantage is already in the
# pregame spread), plus three end-game features built from the rules of the clock. The
# evidence for each change is in results/m1_wp_v2_dev_study.json (development seasons).
FEATURES: list[str] = [f for f in FEATURES_V1 if f != "is_home"] + [
    "kneel_slack",                 # seconds to spare for kneeling out the game (see below)
    "possessions",                 # scores of up to 8 points separating the teams, signed
    "possessions_per_minute",      # the same per minute left: how hurried the trailer is
]  # fmt: skip

# +1: the prediction may never fall when the feature rises, all else equal. -1: never rise.
# Every engineered feature rises with score differential and with the spread, so
# constraining them keeps the whole model monotone in those two raw inputs.
MONOTONE: dict[str, int] = {
    "score_differential": 1,
    "posteam_spread": 1,
    "spread_time": 1,
    "expected_margin": 1,
    "margin_per_sqrt_time": 1,
    "posteam_timeouts_remaining": 1,
    "defteam_timeouts_remaining": -1,
    "kneel_slack": 1,
    "possessions": 1,
    "possessions_per_minute": 1,
    "posteam_timeout_share": 1,
    "defteam_timeout_share": -1,
}

REGULATION_SECONDS = 3600.0
SPREAD_DECAY = 4.0  # spread_time = spread * exp(-4 * share of regulation played)
TIME_FLOOR_SECONDS_V1 = 30.0  # version 1: keeps margin_per_sqrt_time finite at 0 seconds
TIME_FLOOR_SECONDS = 30.0  # version 2 (see the development study for the comparison)

# Kneel-down arithmetic (docs/research/endgame_clock.md). A kneel takes about 2 seconds
# and leaves the clock running, so the next snap can wait 40 more unless the defense
# spends a timeout. With m kneels available and the defense holding n timeouts the
# leading team can burn T = 40 * max(m - n, 0) + 2 * m seconds, at most 120 (the
# two-minute warning acts as a free timeout).
KNEEL_SECONDS = 2.0
PLAY_CLOCK_SECONDS = 40.0
KNEEL_CAP_SECONDS = 120.0
KNEEL_SLACK_FLOOR = -600.0  # "more than ten minutes from kneeling out" is all one value
BASELINE_LOGIT_CAP = 9.0  # the baseline never claims more than about 1 in 8,000

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


@dataclass(frozen=True)
class ModelSpec:
    """What a model is built from: its inputs and whether trees start from the baseline."""

    name: str
    features: tuple[str, ...]
    baseline: bool  # trees start from the normal-approximation baseline (init_score)
    time_floor: float = TIME_FLOOR_SECONDS  # seconds added to the clock in margin_per_sqrt_time
    baseline_tau: float | None = None  # seconds added to the baseline's clock; None: fitted
    baseline_possession: bool = False  # baseline fits a value, in points, for having the ball
    baseline_kneel: bool = False  # baseline is certain where the leader can kneel out

    def __post_init__(self) -> None:
        if self.baseline_kneel and "kneel_slack" not in self.features:
            raise ValueError("baseline_kneel needs the kneel_slack feature")

    def monotone(self) -> list[int]:
        return [MONOTONE.get(f, 0) for f in self.features]


V1 = ModelSpec("v1", tuple(FEATURES_V1), baseline=False, time_floor=TIME_FLOOR_SECONDS_V1)
V2 = ModelSpec("v2", tuple(FEATURES), baseline=True, time_floor=TIME_FLOOR_SECONDS,
               baseline_kneel=True)  # fmt: skip

EPS = 1e-6  # predictions are clipped to [EPS, 1 - EPS] before any log is taken


def load_config(path: Path | None = None) -> dict:
    """Read configs/wp.yaml (split, seed, hyperparameters)."""
    with open(path or CONFIGS / "wp.yaml") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- features


def _feature_exprs(time_floor: float = TIME_FLOOR_SECONDS) -> dict[str, pl.Expr]:
    """Every feature any model version uses, by name."""
    gsr = pl.col("game_seconds_remaining").cast(pl.Float64)
    half = pl.col("half_seconds_remaining").cast(pl.Float64)
    diff = pl.col("score_differential").cast(pl.Float64)
    spread = pl.col("posteam_spread").cast(pl.Float64)
    down = pl.col("down").cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
    pos_to = pl.col("posteam_timeouts_remaining").cast(pl.Float64)
    def_to = pl.col("defteam_timeouts_remaining").cast(pl.Float64)
    overtime = pl.col("qtr") >= 5
    # Share of regulation still to play. In overtime the clock column restarts, so the
    # share is set to zero there: the pregame spread has had a full game to show itself.
    remaining = pl.when(overtime).then(0.0).otherwise(gsr / REGULATION_SECONDS)
    expected_margin = diff + spread * remaining
    # Kneel-down rule: seconds to spare. Zero or more means the team with the ball and
    # the lead can end the game by kneeling. Negative means that many seconds must still
    # be played for real. Defined on downs 1 to 4 of regulation for a team in the lead;
    # everywhere else it is the floor value ("nowhere near").
    kneels = (4.0 - down).clip(0.0, 3.0)
    burn = (
        PLAY_CLOCK_SECONDS * (kneels - def_to).clip(lower_bound=0.0) + KNEEL_SECONDS * kneels
    ).clip(upper_bound=KNEEL_CAP_SECONDS)
    kneel_slack = (
        pl.when((diff >= 1) & (down >= 1) & ~overtime)
        .then((burn - gsr).clip(lower_bound=KNEEL_SLACK_FLOOR))
        .otherwise(KNEEL_SLACK_FLOOR)
    )
    # Scores of up to 8 points that separate the teams, signed like the lead.
    possessions = diff.sign() * (diff.abs() / 8.0).ceil()
    exprs = [
        down.alias("down"),
        pl.col("ydstogo").cast(pl.Float64),
        pl.col("yardline_100").cast(pl.Float64),
        ((down > 0) & (pl.col("ydstogo") >= pl.col("yardline_100")))
        .cast(pl.Float64)
        .alias("goal_to_go"),
        gsr.alias("game_seconds_remaining"),
        half.alias("half_seconds_remaining"),
        pl.col("qtr").cast(pl.Float64).clip(1.0, 5.0).alias("qtr"),
        overtime.cast(pl.Float64).alias("is_overtime"),
        diff.alias("score_differential"),
        pos_to.alias("posteam_timeouts_remaining"),
        def_to.alias("defteam_timeouts_remaining"),
        spread.alias("posteam_spread"),
        pl.col("receive_2h_ko").cast(pl.Float64),
        pl.col("is_home").cast(pl.Float64),
        (spread * (-SPREAD_DECAY * (1.0 - remaining)).exp()).alias("spread_time"),
        expected_margin.alias("expected_margin"),
        (expected_margin / ((gsr + time_floor) / REGULATION_SECONDS).sqrt()).alias(
            "margin_per_sqrt_time"
        ),
        kneel_slack.alias("kneel_slack"),
        possessions.alias("possessions"),
        (possessions / ((gsr + 30.0) / 60.0)).alias("possessions_per_minute"),
        # Share of the half's remaining clock that a team's timeouts could stop.
        (PLAY_CLOCK_SECONDS * pos_to / (half + PLAY_CLOCK_SECONDS)).alias(
            "posteam_timeout_share"
        ),
        (PLAY_CLOCK_SECONDS * def_to / (half + PLAY_CLOCK_SECONDS)).alias(
            "defteam_timeout_share"
        ),
    ]
    return {e.meta.output_name(): e for e in exprs}


def make_features(df: pl.DataFrame, spec: ModelSpec | None = None) -> pl.DataFrame:
    """Build the model inputs from a frame holding STATE_COLUMNS.

    Returns a Float64 frame with exactly the columns in FEATURES (or in `spec.features`
    when a spec is given), one row per input row, in the same order. Nothing outside
    the pre-play state is read, so this works on real plays and on hypothetical states
    from `state_frame` alike.
    """
    spec = spec or V2
    missing = [c for c in STATE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"make_features needs columns {missing}")
    exprs = _feature_exprs(spec.time_floor)
    return df.select([exprs[f] for f in spec.features])


# --------------------------------------------------------------------------- baseline


def baseline_logit(
    expected_margin: np.ndarray,
    seconds: np.ndarray,
    sigma: float,
    tau: float,
    possession_value: float = 0.0,
    has_ball: np.ndarray | None = None,
    can_kneel: np.ndarray | None = None,
) -> np.ndarray:
    """Log-odds of the normal approximation, with two optional rules of the game.

    z = (expected margin + possession_value * has_ball) / (sigma * sqrt((seconds + tau) / 3600))
    and WP = Phi(z). sigma is the standard deviation, in points, of the change in the
    margin over a full game; tau is seconds added to the clock, which keeps z finite at
    0:00 and stands for the scoring still possible on the last snaps. possession_value is
    what having the ball on a scrimmage down is worth, in points. Where `can_kneel` is
    true the team with the ball can end the game by kneeling, and the baseline says so
    (the largest allowed log-odds). The result is clipped to +/- BASELINE_LOGIT_CAP.
    """
    margin = expected_margin + (possession_value * has_ball if has_ball is not None else 0.0)
    z = margin / (sigma * np.sqrt((seconds + tau) / REGULATION_SECONDS))
    out = np.clip(log_ndtr(z) - log_ndtr(-z), -BASELINE_LOGIT_CAP, BASELINE_LOGIT_CAP)
    if can_kneel is not None:
        out = np.where(can_kneel, BASELINE_LOGIT_CAP, out)
    return out


def fit_baseline(
    expected_margin: np.ndarray,
    seconds: np.ndarray,
    y: np.ndarray,
    tau: float | None = None,
    has_ball: np.ndarray | None = None,
    can_kneel: np.ndarray | None = None,
) -> dict:
    """Maximum-likelihood parameters of `baseline_logit` (Nelder-Mead).

    sigma and tau are always fitted (tau only when not given); the possession value is
    fitted when `has_ball` is given. `can_kneel` is a fixed rule, not a parameter.
    """
    fit_tau, fit_poss = tau is None, has_ball is not None

    def unpack(theta: np.ndarray) -> tuple[float, float, float]:
        sigma = float(np.exp(theta[0]))
        tau_hat = float(np.exp(theta[1])) if fit_tau else float(tau)
        poss = float(theta[-1]) if fit_poss else 0.0
        return sigma, tau_hat, poss

    def loss(theta: np.ndarray) -> float:
        score = baseline_logit(expected_margin, seconds, *unpack(theta), has_ball, can_kneel)
        return float(np.mean(np.logaddexp(0.0, score) - y * score))

    start = [np.log(13.5)] + ([np.log(30.0)] if fit_tau else []) + ([0.0] if fit_poss else [])
    best = minimize(loss, x0=np.array(start), method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 800})  # fmt: skip
    sigma, tau_hat, poss = unpack(best.x)
    return {"sigma": sigma, "tau": tau_hat, "possession_value": poss,
            "log_loss": float(best.fun)}  # fmt: skip


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

    `spec` selects the model version (default: version 2, trees on top of the
    normal-approximation baseline). After a fit, `stamp` records the seasons and games
    the model saw; `splits.assert_can_score(model.stamp, rows, what)` uses it.
    """

    def __init__(
        self,
        params: dict | None = None,
        calibration: str = "platt",
        n_threads: int = 2,
        spec: ModelSpec | None = None,
    ) -> None:
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
        self.calibration = calibration
        self.n_threads = n_threads
        self.spec = spec or V2
        self.booster: lgb.Booster | None = None
        self.late_booster: lgb.Booster | None = None  # end-game trees, when configured
        self.baseline: dict[str, float] | None = None
        self.calibrator = Calibrator("none")
        self.seed: int | None = None
        self.n_train_rows = 0
        self.n_calib_rows = 0
        self.stamp: dict | None = None

    # -- fitting

    def _usable(self, df: pl.DataFrame) -> pl.DataFrame:
        """The rows that can be used for fitting: complete state and a label."""
        if LABEL not in df.columns:
            raise KeyError(f"fit needs the label column {LABEL!r}")
        keep = scoreable(df) & df[LABEL].is_not_null().to_numpy()
        return df.filter(pl.Series(keep))

    def _xy(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Features and labels for the rows that can be used for fitting."""
        sub = self._usable(df)
        return (
            make_features(sub, self.spec).to_numpy(),
            sub[LABEL].to_numpy().astype(np.float64),
        )

    def _base(self, x: np.ndarray) -> np.ndarray:
        """The baseline's log-odds for a feature matrix (zeros when the spec has none)."""
        if not self.spec.baseline:
            return np.zeros(x.shape[0])
        assert self.baseline is not None
        return baseline_logit(
            *self._baseline_inputs(x)[:2],
            self.baseline["sigma"],
            self.baseline["tau"],
            self.baseline.get("possession_value", 0.0),
            *self._baseline_inputs(x)[2:],
        )

    def _baseline_inputs(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """expected margin, seconds left, has_ball and can_kneel (None when not used)."""
        names = self.spec.features
        has_ball = (x[:, names.index("down")] > 0).astype(float)
        can_kneel = x[:, names.index("kneel_slack")] >= 0 if self.spec.baseline_kneel else None
        return (
            x[:, names.index("expected_margin")],
            x[:, names.index("game_seconds_remaining")],
            has_ball if self.spec.baseline_possession else None,
            can_kneel,
        )

    def _late(self, x: np.ndarray) -> np.ndarray:
        """Rows the end-game trees apply to: fourth quarter, inside the configured window."""
        names = self.spec.features
        window = float(self.params["endgame"]["seconds"])
        return (x[:, names.index("qtr")] == 4) & (
            x[:, names.index("game_seconds_remaining")] <= window
        )

    def _score(self, x: np.ndarray) -> np.ndarray:
        """Log-odds before calibration: baseline + trees (+ end-game trees late on)."""
        assert self.booster is not None
        score = self._base(x) + self.booster.predict(
            x, raw_score=True, num_threads=self.n_threads
        )
        if self.late_booster is not None:
            late = self._late(x)
            if late.any():
                score[late] += self.late_booster.predict(
                    x[late], raw_score=True, num_threads=self.n_threads
                )
        return score

    def fit(
        self,
        train: pl.DataFrame,
        calib: pl.DataFrame | None = None,
        seed: int = DEFAULT_SEED,
        n_estimators: int | None = None,
        allow_test_seasons: bool = False,
        reason: str | None = None,
        what: str = "WPModel.fit",
    ) -> WPModel:
        """Fit the baseline and the trees on `train` and the calibration map on `calib`.

        Rows with no label (tied games) or an incomplete state are dropped. Pass
        `n_estimators` to override the number of trees (fewer trees for fast
        bootstrap refits).

        Both frames need `season` and `game_id` columns, because the fit refuses rows
        from the test seasons (`splits.LeakageError`). The only way past that is
        `allow_test_seasons=True` with a written `reason`, which is logged; it is
        meant for fits whose predictions are never reported on the test seasons.
        """
        frames = {"training rows": train}
        if calib is not None and calib.height > 0:
            frames["calibration rows"] = calib
        for label, frame in frames.items():
            for col in ("season", "game_id"):
                if col not in frame.columns:
                    raise KeyError(f"{what}: the {label} need a {col!r} column (leakage guard)")
            splits.assert_no_test_seasons(frame, f"{what} ({label})", allow_test_seasons, reason)
        if len(frames) == 2:
            splits.assert_disjoint_games(train, calib, f"{what} (training versus calibration)")

        p = dict(self.params)
        endgame = p.pop("endgame", None)
        rounds = int(n_estimators or p.pop("n_estimators"))
        p.pop("n_estimators", None)
        p.update(
            objective="binary",
            seed=int(seed),
            num_threads=self.n_threads,
            verbose=-1,
            deterministic=True,
            force_col_wise=True,
            monotone_constraints=self.spec.monotone(),
        )
        x, y = self._xy(train)
        if len(np.unique(y)) < 2:
            raise ValueError("training data must contain both wins and losses")
        names = list(self.spec.features)
        if self.spec.baseline:
            margin, seconds, has_ball, can_kneel = self._baseline_inputs(x)
            fitted = fit_baseline(margin, seconds, y, self.spec.baseline_tau, has_ball, can_kneel)
            self.baseline = {k: fitted[k] for k in ("sigma", "tau", "possession_value")}
            init = self._base(x)
            p["boost_from_average"] = False  # the baseline is the starting point
        else:
            self.baseline, init = None, None
        dataset = lgb.Dataset(x, label=y, init_score=init, feature_name=names, free_raw_data=True)
        self.booster = lgb.train(p, dataset, num_boost_round=rounds)
        self.late_booster = None
        if endgame:
            # Second stage: trees that see only the end of regulation and start from
            # the first stage's score, so small leaves are allowed only where the
            # state nearly decides the game.
            late = self._late(x)
            start = self._score(x[late])
            late_params = {**p, **{k: v for k, v in endgame.items()
                                   if k not in ("seconds", "n_estimators")}}  # fmt: skip
            late_params["boost_from_average"] = False
            late_set = lgb.Dataset(x[late], label=y[late], init_score=start,
                                   feature_name=names, free_raw_data=True)  # fmt: skip
            self.late_booster = lgb.train(
                late_params, late_set, num_boost_round=int(endgame["n_estimators"])
            )
        self.seed, self.n_train_rows = int(seed), len(y)
        if "calibration rows" in frames:
            xc, yc = self._xy(calib)
            score = self._score(xc)
            self.calibrator = Calibrator(self.calibration).fit(score, yc)
            self.n_calib_rows = len(yc)
        else:
            self.calibrator = Calibrator("none")
            self.n_calib_rows = 0
        self.stamp = splits.fit_stamp(
            pl.concat([f.select("season", "game_id") for f in frames.values()]), what
        )
        self.stamp.update(
            spec=self.spec.name,
            train_seasons=sorted(splits.seasons_in(train)),
            calib_seasons=sorted(splits.seasons_in(calib)) if "calibration rows" in frames else [],
            allow_test_seasons=bool(allow_test_seasons),
            reason=reason,
        )
        return self

    # -- prediction

    def raw_score(self, df: pl.DataFrame) -> np.ndarray:
        """Log-odds before calibration (baseline plus trees), NaN for unscoreable rows."""
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        ok = scoreable(df)
        out = np.full(df.height, np.nan)
        if ok.any():
            x = make_features(df.filter(pl.Series(ok)), self.spec).to_numpy()
            out[ok] = self._score(x)
        return out

    def baseline_predict(self, df: pl.DataFrame) -> np.ndarray:
        """P(posteam wins) from the structural baseline alone (NaN for unscoreable rows)."""
        if self.booster is None or not self.spec.baseline:
            raise RuntimeError("model has no fitted baseline")
        ok = scoreable(df)
        out = np.full(df.height, np.nan)
        if ok.any():
            x = make_features(df.filter(pl.Series(ok)), self.spec).to_numpy()
            out[ok] = _sigmoid(self._base(x))
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
        """Share of the trees' total loss reduction (gain) credited to each feature.

        In version 2 this is the gain on top of the baseline, so the score and the
        clock, which the baseline already explains, get less credit than in version 1.
        """
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        gain = self.booster.feature_importance(importance_type="gain").astype(float)
        share = gain / gain.sum()
        order = np.argsort(-share)
        return {self.spec.features[i]: float(share[i]) for i in order}

    # -- persistence

    def save(self, path: str | Path) -> Path:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "booster": self.booster.model_to_string(),
                "late_booster": self.late_booster.model_to_string() if self.late_booster else None,
                "features": list(self.spec.features),
                "spec": self.spec,
                "baseline": self.baseline,
                "params": self.params,
                "calibration": self.calibration,
                "calibrator": self.calibrator,
                "n_threads": self.n_threads,
                "seed": self.seed,
                "n_train_rows": self.n_train_rows,
                "n_calib_rows": self.n_calib_rows,
                "stamp": self.stamp,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> WPModel:
        """Load a saved model. Files saved before version 2 load as version 1, unstamped."""
        blob = joblib.load(path)
        spec = blob.get("spec")
        if spec is None:
            if blob["features"] != FEATURES_V1:
                raise ValueError("saved model was fitted with a different feature list; refit it")
            spec = V1
        elif spec.name == V2.name and spec != V2:
            raise ValueError("saved model was fitted with a different feature list; refit it")
        model = cls(blob["params"], blob["calibration"], blob["n_threads"], spec)
        model.booster = lgb.Booster(model_str=blob["booster"])
        if blob.get("late_booster"):
            model.late_booster = lgb.Booster(model_str=blob["late_booster"])
        model.baseline = blob.get("baseline")
        model.calibrator = blob["calibrator"]
        model.seed = blob["seed"]
        model.n_train_rows = blob["n_train_rows"]
        model.n_calib_rows = blob["n_calib_rows"]
        model.stamp = blob.get("stamp")
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


# ------------------------------------------------------------------- end-game slices

# Last five minutes of regulation, scrimmage downs. The margin is from the point of
# view of the team with the ball, so "+1 to +3" is a small lead with possession and
# "-3 to -1" is a small deficit with possession.
ENDGAME_TIME_BINS: list[tuple[str, float, float]] = [
    ("0:00-0:40", 0.0, 40.0),
    ("0:41-2:00", 41.0, 120.0),
    ("2:01-5:00", 121.0, 300.0),
]
ENDGAME_MARGIN_BINS: list[tuple[str, float, float]] = [
    ("trail 9+", -99.0, -9.0),
    ("trail 4-8", -8.0, -4.0),
    ("trail 1-3", -3.0, -1.0),
    ("tied", 0.0, 0.0),
    ("lead 1-3", 1.0, 3.0),
    ("lead 4-8", 4.0, 8.0),
    ("lead 9+", 9.0, 99.0),
]


def endgame_rows(df: pl.DataFrame) -> pl.Series:
    """Mask of fourth-quarter scrimmage downs with 5:00 or less left."""
    return df.select(
        ((pl.col("qtr") == 4) & pl.col("down").is_not_null()
         & (pl.col("game_seconds_remaining") <= 300)).alias("m")
    )["m"]  # fmt: skip


def endgame_slice_table(
    df: pl.DataFrame,
    p: np.ndarray,
    n_boot: int = 1000,
    seed: int = DEFAULT_SEED,
    first_down_only: bool = False,
) -> list[dict[str, Any]]:
    """Predicted versus observed win rate in time-by-margin cells of the end game.

    `df` needs qtr, down, game_seconds_remaining, score_differential, game_id and the
    label; `p` is one prediction per row. Rows without a label or prediction are
    dropped. `gap` is predicted minus observed, with a 95% interval from resampling
    whole games (a game contributes many plays to a cell, all with the same result).
    """
    frame = df.select("game_id", "qtr", "down", "game_seconds_remaining", "score_differential",
                      LABEL).with_columns(pl.Series("_p", p).fill_nan(None))  # fmt: skip
    frame = frame.filter(endgame_rows(frame) & pl.col(LABEL).is_not_null()
                         & pl.col("_p").is_not_null())  # fmt: skip
    if first_down_only:
        frame = frame.filter(pl.col("down") == 1)
    rng = np.random.default_rng(seed)
    out = []
    for t_name, t_lo, t_hi in ENDGAME_TIME_BINS:
        for m_name, m_lo, m_hi in ENDGAME_MARGIN_BINS:
            cell = frame.filter(
                pl.col("game_seconds_remaining").is_between(t_lo, t_hi)
                & pl.col("score_differential").is_between(m_lo, m_hi)
            )
            row: dict[str, Any] = {"time": t_name, "margin": m_name, "rows": cell.height,
                                   "games": cell["game_id"].n_unique()}  # fmt: skip
            if cell.height:
                y, pred = cell[LABEL].to_numpy(), cell["_p"].to_numpy()
                _, g = np.unique(cell["game_id"].to_numpy(), return_inverse=True)
                n_games = int(g.max()) + 1
                cnt = np.bincount(g, minlength=n_games).astype(float)
                sum_d = np.bincount(g, weights=pred - y, minlength=n_games)
                draws = rng.integers(0, n_games, size=(n_boot, n_games))
                boot = sum_d[draws].sum(axis=1) / cnt[draws].sum(axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                row.update(observed=float(y.mean()), predicted=float(pred.mean()),
                           gap=float(pred.mean() - y.mean()), gap_ci_low=float(lo),
                           gap_ci_high=float(hi), log_loss=log_loss(y, pred))  # fmt: skip
            out.append(row)
    return out


def endgame_summary(table: list[dict[str, Any]]) -> dict[str, float]:
    """One number per table: the row-weighted mean absolute gap across the cells."""
    cells = [c for c in table if c["rows"] > 0]
    n = float(sum(c["rows"] for c in cells))
    return {
        "rows": int(n),
        "mean_abs_gap": float(sum(c["rows"] * abs(c["gap"]) for c in cells) / n),
        "max_abs_gap": float(max(abs(c["gap"]) for c in cells)),
        "cells_whose_interval_excludes_zero": int(
            sum(1 for c in cells if c["gap_ci_low"] > 0 or c["gap_ci_high"] < 0)
        ),
        "cells": len(cells),
    }


# ----------------------------------------------------------------------- cross-fitting


def season_filter(df: pl.DataFrame, seasons: Sequence[int]) -> pl.DataFrame:
    return df.filter(pl.col("season").is_in(list(seasons)))


def crossfit_plan(
    seasons: Sequence[int],
    fold_size: int = 4,
    n_calib_seasons: int = 2,
    never_fit: Sequence[int] = (),
) -> list[dict[str, list[int]]]:
    """Leave-seasons-out folds of consecutive seasons.

    For each fold, the held-out seasons are predicted by a model whose trees are
    grown on the other seasons except the most recent `n_calib_seasons` of them,
    which are used for the calibration map (none when `n_calib_seasons` is 0).

    Seasons in `never_fit` (the test seasons) are in no fold's training or calibration
    set. They form one last fold, predicted by a model fitted on all the other seasons.
    """
    never = sorted(set(never_fit) & set(seasons))
    usable = sorted(set(seasons) - set(never))
    blocks = [usable[i : i + fold_size] for i in range(0, len(usable), fold_size)]
    if never:
        blocks.append(never)
    plan = []
    for held in blocks:
        rest = [s for s in usable if s not in held]
        if len(rest) <= n_calib_seasons:
            raise ValueError("not enough seasons outside the fold to train and calibrate")
        cut = len(rest) - n_calib_seasons
        plan.append({"predict": held, "train": rest[:cut], "calib": rest[cut:]})
    return plan


def crossfit_predict(
    plays: pl.DataFrame,
    params: dict | None = None,
    calibration: str = "platt",
    fold_size: int = 4,
    n_calib_seasons: int = 2,
    seed: int = DEFAULT_SEED,
    n_threads: int = 2,
    spec: ModelSpec | None = None,
    never_fit: Sequence[int] | None = None,
    allow_test_seasons: bool = False,
    reason: str | None = None,
) -> tuple[pl.DataFrame, list[dict[str, list[int]]]]:
    """Out-of-fold win probability for every play.

    Returns (frame with game_id, play_id, wp_ours; plan). `wp_ours` is null for rows
    whose state is incomplete. No game is ever predicted by a model that saw it, and
    each fold's model is checked against the rows it scores (`assert_can_score`).

    By default (`never_fit=None`) the test seasons are in no fold's training or
    calibration set: they are predicted by a model fitted on every development season.
    Passing `never_fit=()` lets other folds' models train on test seasons, which then
    requires `allow_test_seasons=True` and a written `reason` that is logged.
    """
    seasons = plays["season"].unique().to_list()
    if never_fit is None:
        never_fit = splits.test_seasons()
    plan = crossfit_plan(seasons, fold_size, n_calib_seasons, never_fit)
    parts = []
    for fold in plan:
        calib = season_filter(plays, fold["calib"]) if fold["calib"] else None
        model = WPModel(params, calibration, n_threads, spec).fit(
            season_filter(plays, fold["train"]), calib, seed=seed,
            allow_test_seasons=allow_test_seasons, reason=reason,
            what=f"cross-fit model predicting {fold['predict'][0]} to {fold['predict'][-1]}",
        )  # fmt: skip
        held = season_filter(plays, fold["predict"])
        splits.assert_can_score(model.stamp, held, "cross-fitted predictions")
        pred = model.predict(held)
        parts.append(
            held.select("game_id", "play_id").with_columns(
                pl.Series("wp_ours", pred).fill_nan(None)
            )
        )
    return pl.concat(parts), plan


def season_blocked_cv(
    plays: pl.DataFrame,
    params: dict | None = None,
    calibration: str = "none",
    spec: ModelSpec | None = None,
    fold_size: int = 2,
    n_calib_seasons: int = 0,
    seed: int = DEFAULT_SEED,
    n_threads: int = 2,
    in_sample_rows: int = 150_000,
) -> dict[str, Any]:
    """Season-blocked cross-validation of one model design.

    Whole seasons are held out, `fold_size` consecutive seasons at a time, and each
    is predicted by a model fitted on all the other seasons in `plays`. Pass
    development plays only: the fit guard refuses test seasons.

    Returns the usable rows (`frame`, with a label and a complete state), their
    held-out predictions (`pred`) and uncalibrated log-odds (`raw`), and per fold the
    held-out log loss, the log loss on a sample of the rows the model was fitted on
    (their difference is the memorization gap), the fit time and the fitted baseline.
    """
    frame = plays.filter(pl.Series(scoreable(plays)) & pl.col(LABEL).is_not_null())
    season, y = frame["season"].to_numpy(), frame[LABEL].to_numpy()
    pred, raw = np.full(frame.height, np.nan), np.full(frame.height, np.nan)
    plan = crossfit_plan(sorted(set(season.tolist())), fold_size, n_calib_seasons)
    folds = []
    for fold in plan:
        calib = season_filter(frame, fold["calib"]) if fold["calib"] else None
        train = season_filter(frame, fold["train"])
        start = time.perf_counter()
        model = WPModel(params, calibration, n_threads, spec).fit(
            train, calib, seed=seed,
            what=f"season-blocked CV, held out {fold['predict'][0]} to {fold['predict'][-1]}",
        )  # fmt: skip
        seconds = time.perf_counter() - start
        mask = np.isin(season, fold["predict"])
        held = frame.filter(pl.Series(mask))
        splits.assert_can_score(model.stamp, held, "season-blocked CV, held-out fold")
        raw[mask], pred[mask] = model.raw_score(held), model.predict(held)
        sample = train.sample(min(train.height, in_sample_rows), seed=seed)
        folds.append(
            {
                **fold,
                "held_out_log_loss": log_loss(y[mask], pred[mask]),
                "in_sample_log_loss": log_loss(sample[LABEL].to_numpy(), model.predict(sample)),
                "fit_seconds": seconds,
                "baseline": model.baseline,
                "calibrator": model.calibrator.describe(),
            }
        )
    return {"frame": frame, "pred": pred, "raw": raw, "folds": folds}


# ----------------------------------------------------------------------- sanity checks


def sanity_checks(model: WPModel) -> dict[str, Any]:
    """Behaviour any sensible win-probability model must show, on hypothetical states.

    Returns the measured quantities and a `passed` flag for each check, plus
    `core_passed` (checks 1 to 5, which scripts/fit_wp.py requires), `endgame_passed`
    (checks 6 to 9, see `endgame_sanity_checks`, recorded and reported) and
    `all_passed` (both).
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
    out["core_passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    end = endgame_sanity_checks(model)
    out.update(end)
    out["endgame_passed"] = all(v["passed"] for v in end.values())
    out["all_passed"] = out["core_passed"] and out["endgame_passed"]
    return out


def endgame_sanity_checks(model: Any) -> dict[str, Any]:
    """End-of-game behaviour that follows from the clock rules, on hypothetical states.

    Works for anything with a `predict(frame of STATE_COLUMNS)` method. The thresholds
    are deliberately loose compared with the truth (the facts are in
    docs/research/endgame_clock.md): they catch a model that is blind to the end game.
    """
    out: dict[str, Any] = {}
    yards = np.array([90.0, 75.0, 50.0, 20.0])

    # 6. Kneel-out. A lead of 1 to 3, first down, under a minute left and the defense
    # out of timeouts: three kneels end the game (real teams won 99.9% of these).
    secs = np.arange(1.0, 60.0)
    d, s, yl = np.meshgrid(np.array([1.0, 2.0, 3.0]), secs, yards, indexing="ij")
    p = model.predict(
        state_frame(score_differential=d.ravel(), game_seconds_remaining=s.ravel(),
                    yardline_100=yl.ravel(), defteam_timeouts_remaining=0)
    )  # fmt: skip
    out["kneel_out"] = {"min_wp": float(p.min()), "mean_wp": float(p.mean()),
                        "passed": float(p.min()) > 0.95}  # fmt: skip

    # 7. No time for two scores. Down 9 or more with under 30 seconds left.
    secs = np.arange(0.0, 30.0)
    d, s, yl = np.meshgrid(np.arange(-24.0, -8.0), secs, yards, indexing="ij")
    p = model.predict(
        state_frame(score_differential=d.ravel(), game_seconds_remaining=s.ravel(),
                    yardline_100=yl.ravel(), ydstogo=np.minimum(10.0, yl.ravel()))
    )  # fmt: skip
    out["two_scores_no_time"] = {"max_wp": float(p.max()), "mean_wp": float(p.mean()),
                                 "passed": float(p.max()) < 0.02}  # fmt: skip

    # 8. The defense's timeouts matter. Up 3, first down at the own 40, 80 seconds
    # left: with no timeouts the defense is beaten (80 <= 120); with three it gets
    # the ball back. The model must see a difference.
    p = model.predict(
        state_frame(score_differential=3, game_seconds_remaining=80, yardline_100=60,
                    defteam_timeouts_remaining=[0, 1, 2, 3])
    )  # fmt: skip
    out["defense_timeouts"] = {
        "wp_by_defense_timeouts": [float(v) for v in p],
        "zero_minus_three": float(p[0] - p[3]),
        "passed": float(p[0] - p[3]) > 0.01 and bool(np.all(np.diff(p) <= 1e-9)),
    }

    # 9. The state that exposed version 1: up 1, first down at the own 25, 13 seconds
    # left, both sides with all their timeouts. Version 1 said 0.72.
    p = model.predict(
        state_frame(score_differential=1, game_seconds_remaining=13, yardline_100=75)
    )
    out["one_point_lead_13_seconds"] = {"wp": float(p[0]), "passed": float(p[0]) > 0.90}
    return out
