"""Field-goal make probability that separates the kick from the kicker who was sent out.

Why this exists. The raw make rate at long range is not the make rate of an average
kicker. Coaches send strong-legged kickers out for long tries, mostly indoors or in calm
weather, which pushes the raw rate up; end-of-half heaves, kicked because time ran out,
push it down. `components.FieldGoalModel` (M1) fits distance and a season trend only, so
it inherits whatever mix of kickers and conditions produced the long attempts. This
module adds the missing terms and reads the curve for a stated kicker and stated
conditions.

The model (a hierarchical, or mixed-effects, logistic regression). For attempt i by
kicker k, with z = (distance - 45) / 10:

    log-odds(make) = s(distance)                            smooth distance curve
                     + roof and weather terms (outdoor games)
                     + season trend and season by z
                     + heave flag
                     + gamma * (usage_k - usage_league) / 10  kicker usage (Mundlak term)
                     + u_k + v_k * z                         kicker intercept and slope
    (u_k, v_k) ~ Normal(0, Sigma), Sigma a 2 by 2 covariance.

usage_k is the mean distance of kicker k's training attempts. Random effects assume a
kicker's strength is unrelated to the distances he attempts, which is exactly what
selection breaks; the usage term absorbs that relation so the distance curve is learned
from comparisons within kickers.
s(distance) is a natural cubic spline (linear in log-odds beyond the outer knots).
The fixed effects and kicker effects are found together by penalized likelihood (the
kicker effects are shrunk toward 0 by Sigma), and Sigma by maximizing the Laplace
approximation of the marginal likelihood: empirical Bayes. A kicker with few attempts is
pulled toward the league average; a kicker with hundreds keeps most of his own effect.

Fitting is refused on any season outside the training seasons of `configs/splits.yaml`
(2010 to 2021): calibration seasons are for checks, test seasons for the one
preregistered check, and the 2026 holdout is never read.

Planner interface: `FGMakeModel.p_make(distance, strength=..., roof=..., temp=...,
wind=..., season=..., heave=...)`, documented in `docs/notes/fg_selection_bias.md`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import yaml
from scipy import optimize, stats

from fourth_and_rl import splits
from fourth_and_rl.paths import CONFIGS, DATA

KICKER_DIR = DATA / "kicker"
MODEL_PATH = DATA / "models" / "kicker_fg.joblib"
KICKER_COLUMNS: tuple[str, ...] = ("game_id", "play_id", "kicker_player_id",
                                   "kicker_player_name")  # fmt: skip
# The `outdoor` column codes the roof: 1 outdoors, 0.5 a retractable roof left open (its
# own effect; nflverse records no weather under it), 0 indoors (dome or closed roof).
ROOF_CODE = {"outdoors": 1.0, "open": 0.5, "dome": 0.0, "closed": 0.0}


def roof_code(roof: str | None) -> float:
    """The `outdoor` code of a roof value (unknown or missing counts as outdoors)."""
    return ROOF_CODE.get(roof or "outdoors", 1.0)


def _roof_expr(col: str = "roof") -> pl.Expr:
    return (pl.col(col).fill_null("outdoors").replace_strict(ROOF_CODE, default=1.0)
            .cast(pl.Float64))  # fmt: skip
# Plays that count as "something happened after the kick" when deciding whether a kick
# was the last play of its half.
_LATER_PLAY_TYPES = ("run", "pass", "field_goal", "punt", "kickoff", "qb_kneel", "qb_spike",
                     "no_play", "extra_point")  # fmt: skip


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIGS / "kicker.yaml") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    text = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Data: kicker ids (not in the state table) and the attempt table
# --------------------------------------------------------------------------------------


def download_kicker_ids(
    seasons: Sequence[int], out_dir: Path | None = None, force: bool = False
) -> list[Path]:
    """Cache `kicker_player_id` and `kicker_player_name` per season from nflreadpy.

    The state table does not carry the kicker, so the ids are read from the nflverse
    play-by-play and joined back by game and play id (the same pattern as
    `coaches.download_penalties`). The final holdout seasons are refused.
    """
    import nflreadpy as nfl

    banned = set(seasons) & set(splits.holdout_seasons())
    if banned:
        raise splits.LeakageError(f"kicker id download: {sorted(banned)} is the final holdout")
    out_dir = out_dir or KICKER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for season in seasons:
        path = out_dir / f"kickers_{season}.parquet"
        if force or not path.exists():
            raw = nfl.load_pbp(int(season)).select(list(KICKER_COLUMNS))
            raw = raw.filter(pl.col("kicker_player_id").is_not_null())
            raw.with_columns(pl.col("play_id").cast(pl.Float64)).write_parquet(path)
        paths.append(path)
    return paths


def load_kicker_ids(seasons: Sequence[int], out_dir: Path | None = None) -> pl.DataFrame:
    """Kicker ids for `seasons`, downloading any season not yet cached."""
    paths = download_kicker_ids(seasons, out_dir)
    return pl.concat([pl.read_parquet(p) for p in paths], how="vertical_relaxed")


def heave_flags(plays: pl.DataFrame, max_half_seconds: float) -> pl.DataFrame:
    """Add `heave`: a field-goal attempt with at most `max_half_seconds` left in the first
    half or regulation and no later snap or kickoff in the same half.

    Needs every play of each game (not only the field goals), sorted by game and
    `order_sequence`, to see what came after the kick.
    """
    order = [c for c in ("order_sequence", "play_id") if c in plays.columns]
    df = plays.with_row_index("_i").sort(["game_id", *order])
    later = pl.col("play_type").is_in(list(_LATER_PLAY_TYPES)).fill_null(False)
    # Index of each row within its (game, half), and of the last "later" play in that half.
    df = df.with_columns(pl.int_range(pl.len()).over("game_id", "game_half").alias("_pos"))
    last = (
        df.filter(later)
        .group_by("game_id", "game_half")
        .agg(pl.col("_pos").max().alias("_last_pos"))
    )
    df = df.join(last, on=["game_id", "game_half"], how="left")
    is_fg = (pl.col("play_type") == "field_goal").fill_null(False)
    heave = (
        is_fg
        & pl.col("game_half").is_in(["Half1", "Half2"]).fill_null(False)
        & (pl.col("half_seconds_remaining").fill_null(1e9) <= max_half_seconds)
        & (pl.col("_pos") >= pl.col("_last_pos"))
    )
    out = df.with_columns(heave.fill_null(False).alias("heave"))
    return out.sort("_i").drop("_i", "_pos", "_last_pos")


def field_goal_attempts(
    plays: pl.DataFrame, kicker_ids: pl.DataFrame | None, cfg: dict | None = None
) -> pl.DataFrame:
    """One row per field-goal attempt with the columns the model reads.

    `plays` must be the full play table for the games involved (the heave flag looks at
    the plays after each kick). Distance is the recorded `kick_distance`, or
    yardline_100 + 18 when it is missing; blocked kicks count as misses. Attempts with no
    kicker id are dropped (their count is kept in the attribute-free return via the
    `kicker_id` null filter; callers report it).
    """
    cfg = cfg or load_config()
    df = heave_flags(plays, cfg["heave"]["max_half_seconds"]).filter(
        (pl.col("play_type") == "field_goal") & pl.col("field_goal_result").is_not_null()
    )
    if kicker_ids is not None:
        k = kicker_ids.select(
            "game_id", pl.col("play_id").cast(pl.Float64),
            pl.col("kicker_player_id").alias("kicker_id"),
            pl.col("kicker_player_name").alias("kicker_name"),
        ).unique(subset=["game_id", "play_id"])  # fmt: skip
        df = df.with_columns(pl.col("play_id").cast(pl.Float64)).join(
            k, on=["game_id", "play_id"], how="left"
        )
    distance = pl.coalesce(pl.col("kick_distance"), pl.col("yardline_100") + 18)
    roof = pl.col("roof").fill_null("outdoors")
    return df.select(
        "game_id", "play_id", "season", "kicker_id", "kicker_name",
        distance.cast(pl.Float64).alias("distance"),
        (pl.col("yardline_100") + 18).cast(pl.Float64).alias("distance_yl18"),
        (pl.col("field_goal_result") == "made").cast(pl.Int8).alias("made"),
        roof.alias("roof"),
        _roof_expr().alias("outdoor"),
        pl.col("temp").cast(pl.Float64), pl.col("wind").cast(pl.Float64),
        pl.col("heave").cast(pl.Int8),
        "half_seconds_remaining", "game_half",
    )  # fmt: skip


# --------------------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------------------


def natural_spline_basis(x: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    """Restricted (natural) cubic spline basis: x itself plus len(knots) - 2 columns.

    Cubic between knots, linear beyond the outer knots (Harrell's parameterization,
    each nonlinear column scaled by (last knot - first knot)^2).
    """
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    kk, k1, scale = k[-1], k[-2], (k[-1] - k[0]) ** 2

    def p3(v: np.ndarray) -> np.ndarray:
        return np.clip(v, 0.0, None) ** 3

    cols = [x]
    for j in range(len(k) - 2):
        c = (p3(x - k[j]) - p3(x - k1) * (kk - k[j]) / (kk - k1)
             + p3(x - kk) * (k1 - k[j]) / (kk - k1))  # fmt: skip
        cols.append(c / scale)
    return np.column_stack(cols)


@dataclass
class Conditions:
    """Game conditions for a prediction. `outdoor` is the roof code (1 outdoors, 0.5 open
    retractable roof, 0 indoors); temp in F, wind in mph (used outdoors only; None means
    "not recorded" and is filled with the training median)."""

    outdoor: float = 1.0
    temp: float | None = None
    wind: float | None = None


@dataclass
class Design:
    """Builds the fixed-effect matrix from attempt columns, with fitted constants."""

    cfg: dict
    usage_league: float = 38.0
    temp_fill: float = 60.0
    wind_fill: float = 8.0
    names: list[str] = field(default_factory=list)

    def z(self, distance: np.ndarray) -> np.ndarray:
        return (np.asarray(distance, dtype=float) - self.cfg["distance"]["center"]) / 10.0

    def matrix(
        self,
        distance: np.ndarray,
        outdoor: np.ndarray,
        temp: np.ndarray,
        wind: np.ndarray,
        season: np.ndarray,
        heave: np.ndarray,
        usage: np.ndarray | None = None,
    ) -> np.ndarray:
        t = self.cfg["terms"]
        w = self.cfg["weather"]
        n = np.asarray(distance).shape[0]
        distance = np.asarray(distance, dtype=float)
        z = self.z(distance)
        spline = natural_spline_basis(distance, self.cfg["distance"]["knots"])
        spline[:, 0] = z  # the linear column on the z scale, for a readable coefficient
        cols: list[np.ndarray] = [np.ones(n), *spline.T]
        names = ["intercept", "z"] + [f"spline_{j}" for j in range(1, spline.shape[1])]
        if t.get("roof", True):
            # outdoor: 1 outdoors, 0.5 a retractable roof left open, 0 indoors. The open
            # roof gets its own flag (outdoor - its share is not identified separately
            # from weather, because nflverse records no weather under an open roof).
            out = np.asarray(outdoor, dtype=float)
            fully = (out > 0.75).astype(float)
            open_roof = ((out > 0.25) & (out <= 0.75)).astype(float)
            cols += [fully, open_roof]
            names += ["outdoor", "open_roof"]
            if t.get("weather", True):
                temp = np.asarray(temp, dtype=float)
                wind = np.asarray(wind, dtype=float)
                tc = (np.where(np.isnan(temp), self.temp_fill, temp) - w["temp_center"]) / w[
                    "temp_scale"
                ]
                wc = np.where(np.isnan(wind), self.wind_fill, wind) / w["wind_scale"]
                cols += [fully * tc, fully * wc, fully * wc * z, fully * tc * z]
                names += ["outdoor_temp", "outdoor_wind", "outdoor_wind_x_z",
                          "outdoor_temp_x_z"]  # fmt: skip
        if t.get("season", True):
            s = (np.asarray(season, dtype=float) - self.cfg["season"]["center"]) / self.cfg[
                "season"
            ]["scale"]
            cols += [s, s * z]
            names += ["season", "season_x_z"]
        if t.get("heave", True):
            cols.append(np.asarray(heave, dtype=float))
            names.append("heave")
        if t.get("kicker_usage", True):
            use = np.full(n, self.usage_league) if usage is None else np.asarray(usage, float)
            use = np.where(np.isnan(use), self.usage_league, use)
            cols.append((use - self.usage_league) / 10.0)
            names.append("kicker_usage")
        self.names = names
        return np.column_stack(cols)


# --------------------------------------------------------------------------------------
# Penalized likelihood with kicker effects, and the Laplace marginal likelihood
# --------------------------------------------------------------------------------------


def _sigma_from_theta(theta: np.ndarray, slope: bool) -> np.ndarray:
    """theta = (log sd_u, log sd_v, atanh rho) -> 2 by 2 covariance (or 1 by 1)."""
    lo, hi = np.log(1e-3), np.log(5.0)  # standard deviations kept in [0.001, 5]
    if not slope:
        return np.array([[np.exp(2 * np.clip(theta[0], lo, hi))]])
    su, sv = np.exp(np.clip(theta[0], lo, hi)), np.exp(np.clip(theta[1], lo, hi))
    rho = np.tanh(np.clip(theta[2], -3.0, 3.0))
    return np.array([[su * su, rho * su * sv], [rho * su * sv, sv * sv]])


@dataclass
class _Mode:
    beta: np.ndarray
    b: np.ndarray  # (K, q)
    loglik: float
    penalized: float
    hbb: np.ndarray  # (K, q, q)
    hbx: np.ndarray  # (K, q, p)
    hxx: np.ndarray  # (p, p)


def _fit_mode(
    x: np.ndarray,
    zr: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    n_groups: int,
    sigma: np.ndarray,
    ridge: np.ndarray,
    beta0: np.ndarray,
    b0: np.ndarray,
    tol: float,
    max_iter: int,
) -> _Mode:
    """Newton's method for the joint mode of fixed effects and kicker effects.

    Maximizes  sum_i loglik_i - 0.5 * sum_k b_k' Sigma^-1 b_k - 0.5 * beta' R beta,
    using the block structure: each attempt has one kicker, so the kicker block of the
    Hessian is block diagonal (q by q per kicker) and the fixed effects are solved with a
    Schur complement.
    """
    p, q = x.shape[1], zr.shape[1]
    sinv = np.linalg.inv(sigma)
    beta, b = beta0.copy(), b0.copy()

    def objective(beta: np.ndarray, b: np.ndarray) -> tuple[float, float, np.ndarray]:
        eta = x @ beta + np.einsum("iq,iq->i", zr, b[group])
        ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
        pen = ll - 0.5 * float(np.einsum("kq,qr,kr->", b, sinv, b)) - 0.5 * float(
            beta @ (ridge * beta)
        )
        return ll, pen, eta

    ll, pen, eta = objective(beta, b)
    for _ in range(max_iter):
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1.0 - mu)
        r = y - mu
        gx = x.T @ r - ridge * beta
        gb = np.zeros((n_groups, q))
        for j in range(q):
            gb[:, j] = np.bincount(group, weights=r * zr[:, j], minlength=n_groups)
        gb -= b @ sinv
        hxx = (x * w[:, None]).T @ x + np.diag(ridge)
        hbb = np.zeros((n_groups, q, q))
        hbx = np.zeros((n_groups, q, p))
        for j in range(q):
            wz = w * zr[:, j]
            for m in range(q):
                hbb[:, j, m] = np.bincount(group, weights=wz * zr[:, m], minlength=n_groups)
            for m in range(p):
                hbx[:, j, m] = np.bincount(group, weights=wz * x[:, m], minlength=n_groups)
        hbb += sinv[None, :, :]
        hbb_inv = np.linalg.inv(hbb)
        a = np.einsum("kqr,krp->kqp", hbb_inv, hbx)  # Hbb^-1 Hbx
        schur = hxx - np.einsum("kqp,kqm->pm", hbx, a)
        rhs = gx - np.einsum("kqp,kq->p", hbx, np.einsum("kqr,kr->kq", hbb_inv, gb))
        dbeta = np.linalg.solve(schur, rhs)
        db = np.einsum("kqr,kr->kq", hbb_inv, gb - np.einsum("kqp,p->kq", hbx, dbeta))
        step = 1.0
        while True:
            nb, nbb = beta + step * dbeta, b + step * db
            nll, npen, neta = objective(nb, nbb)
            if npen >= pen - 1e-10 or step < 1e-6:
                break
            step *= 0.5
        gain = npen - pen
        beta, b, ll, pen, eta = nb, nbb, nll, npen, neta
        if abs(gain) < tol:
            break
    # Hessian at the final point (needed for the Laplace term and the covariances).
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = mu * (1.0 - mu)
    hxx = (x * w[:, None]).T @ x + np.diag(ridge)
    hbb = np.zeros((n_groups, q, q))
    hbx = np.zeros((n_groups, q, p))
    for j in range(q):
        wz = w * zr[:, j]
        for m in range(q):
            hbb[:, j, m] = np.bincount(group, weights=wz * zr[:, m], minlength=n_groups)
        for m in range(p):
            hbx[:, j, m] = np.bincount(group, weights=wz * x[:, m], minlength=n_groups)
    hbb += sinv[None, :, :]
    return _Mode(beta, b, ll, pen, hbb, hbx, hxx)


def _laplace(mode: _Mode, sigma: np.ndarray) -> float:
    """Laplace approximation of the log marginal likelihood (kicker effects integrated
    out at the joint mode)."""
    k = mode.b.shape[0]
    _, logdet_sigma = np.linalg.slogdet(sigma)
    _, logdet_h = np.linalg.slogdet(mode.hbb)
    quad = float(np.einsum("kq,qr,kr->", mode.b, np.linalg.inv(sigma), mode.b))
    return mode.loglik - 0.5 * quad - 0.5 * k * logdet_sigma - 0.5 * float(logdet_h.sum())


# --------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------


class FGMakeModel:
    """Hierarchical logistic field-goal make model (see the module docstring).

    A kicker enters in two parts. His random effects (u_k, v_k) are what the data say
    about him beyond the league curve. His usage, the mean distance of his training
    attempts, is a fixed covariate (the Mundlak, or correlated-random-effects, term): it
    soaks up the part of kicker strength that is correlated with how far coaches send
    him out, so that the distance curve is learned from comparisons within kickers and
    is not tilted up by strong kickers taking the long kicks. Without it the random
    effects are assumed independent of distance, which is exactly the assumption that
    selection breaks. A kicker's total effect is gamma * (usage_k - usage_league) / 10
    + u_k + v_k * z; the league-average kicker has usage_league and zero effects.
    """

    stamp: dict | None = None

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = copy_cfg(cfg or load_config())

    # ---------------------------------------------------------------- fitting

    def _usage_of(self, att: pl.DataFrame) -> np.ndarray:
        """Each row's kicker usage (training mean attempt distance); league usage for a
        kicker not seen in training."""
        league = getattr(self, "usage_league", np.nan)
        if "kicker_id" not in att.columns or not hasattr(self, "usage_by_kicker"):
            return np.full(att.height, league)
        u = self.usage_by_kicker
        return np.array([u.get(k, league) for k in att["kicker_id"].to_list()], dtype=float)

    def _arrays(self, att: pl.DataFrame) -> dict[str, np.ndarray]:
        def col(name: str, default: float) -> np.ndarray:
            if name not in att.columns:
                return np.full(att.height, default)
            return att[name].cast(pl.Float64).fill_null(np.nan).to_numpy()

        return {
            "distance": col("distance", np.nan),
            "outdoor": np.nan_to_num(col("outdoor", 1.0), nan=1.0),
            "temp": col("temp", np.nan),
            "wind": col("wind", np.nan),
            "season": col("season", self.cfg["season"]["center"]),
            "heave": np.nan_to_num(col("heave", 0.0), nan=0.0),
            "usage": self._usage_of(att),
        }

    def fit(self, att: pl.DataFrame) -> FGMakeModel:
        """Fit on training-season attempts only (`field_goal_attempts` output).

        Raises `splits.LeakageError` for any row outside the training seasons: the
        calibration seasons are for checks, the test seasons for the one preregistered
        check, the 2026 holdout for nothing.
        """
        if "season" not in att.columns:
            raise splits.LeakageError("kicker model: attempts need a season column")
        splits.assert_no_test_seasons(att, "kicker field-goal model")
        outside = splits.seasons_in(att) - set(splits.train_seasons())
        if outside:
            raise splits.LeakageError(
                f"kicker field-goal model: seasons {sorted(outside)} are not training "
                f"seasons {list(splits.train_seasons())}; calibration seasons are for checks only"
            )
        att = att.filter(pl.col("kicker_id").is_not_null() & pl.col("distance").is_not_null())
        if att.height == 0:
            raise ValueError("kicker model: no attempts to fit")
        cfg = self.cfg
        usage = att.group_by("kicker_id").agg(pl.col("distance").mean())
        self.usage_by_kicker = dict(zip(usage["kicker_id"].to_list(),
                                        usage["distance"].to_list(), strict=True))  # fmt: skip
        # League usage: the mean over attempts (a kicker used like the league as a whole).
        self.usage_league = float(att["distance"].mean())
        a = self._arrays(att)
        self.design = Design(cfg, usage_league=self.usage_league)
        out = a["outdoor"] > 0.75
        temp_ok = out & ~np.isnan(a["temp"])
        wind_ok = out & ~np.isnan(a["wind"])
        if temp_ok.any():
            self.design.temp_fill = float(np.median(a["temp"][temp_ok]))
        if wind_ok.any():
            self.design.wind_fill = float(np.median(a["wind"][wind_ok]))
        x = self.design.matrix(**a)
        self.feature_names = list(self.design.names)
        y = att["made"].cast(pl.Float64).to_numpy()
        ids = att["kicker_id"].to_numpy()
        self.kicker_ids, group = np.unique(ids, return_inverse=True)
        names = att.group_by("kicker_id").agg(pl.col("kicker_name").drop_nulls().mode().first())
        name_map = dict(zip(names["kicker_id"].to_list(), names["kicker_name"].to_list(),
                            strict=True))  # fmt: skip
        self.kicker_names = [name_map.get(k) or str(k) for k in self.kicker_ids]
        self.kicker_attempts = np.bincount(group, minlength=len(self.kicker_ids))
        z = self.design.z(a["distance"])
        slope = bool(cfg["terms"].get("kicker_slope", True))
        zr = np.column_stack([np.ones_like(z), z]) if slope else np.ones((z.size, 1))
        self.slope = slope
        p = x.shape[1]
        ridge = np.full(p, cfg["fit"]["ridge"])
        ridge[0] = 0.0
        f = cfg["fit"]
        k = len(self.kicker_ids)
        # Start from the plain logistic fit (no kicker effects).
        start_beta = optimize.minimize(
            lambda bt: -float(np.sum(y * (x @ bt) - np.logaddexp(0.0, x @ bt)))
            + 0.5 * float(bt @ (ridge * bt)),
            np.zeros(p),
            jac=lambda bt: -(x.T @ (y - 1.0 / (1.0 + np.exp(-(x @ bt))))) + ridge * bt,
            method="L-BFGS-B",
        ).x
        state = {"beta": start_beta, "b": np.zeros((k, zr.shape[1]))}

        def neg_marginal(theta: np.ndarray) -> float:
            sigma = _sigma_from_theta(theta, slope)
            mode = _fit_mode(x, zr, y, group, k, sigma, ridge, state["beta"], state["b"],
                             f["newton_tol"], f["newton_max_iter"])  # fmt: skip
            state["beta"], state["b"] = mode.beta, mode.b
            return -_laplace(mode, sigma)

        theta0 = (np.array([np.log(f["start_sd_intercept"]), np.log(f["start_sd_slope"]), 0.0])
                  if slope else np.array([np.log(f["start_sd_intercept"])]))  # fmt: skip
        res = optimize.minimize(neg_marginal, theta0, method="Nelder-Mead",
                                options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 600})  # fmt: skip
        self.theta = res.x
        self.sigma = _sigma_from_theta(res.x, slope)
        mode = _fit_mode(x, zr, y, group, k, self.sigma, ridge, state["beta"], state["b"],
                         f["newton_tol"], f["newton_max_iter"])  # fmt: skip
        self.beta, self.b = mode.beta, mode.b
        self.loglik, self.log_marginal = mode.loglik, _laplace(mode, self.sigma)
        self.converged = bool(res.success)
        self._covariances(mode)
        self.n = int(y.size)
        self.reference_season = int(np.max(a["season"]))
        self.mean_season = float(np.mean(a["season"]))
        self._kicker_index = {kid: i for i, kid in enumerate(self.kicker_ids)}
        # Spread of usage across established kickers (for strength quantiles).
        est = self.kicker_attempts >= cfg["kickers"]["min_attempts_ranked"]
        if est.sum() < 2:
            est = self.kicker_attempts > 0
        us = np.array([self.usage_by_kicker[kid] for kid in self.kicker_ids])
        wts = self.kicker_attempts[est] / self.kicker_attempts[est].sum()
        dev10 = (us[est] - self.usage_league) / 10.0
        self.usage_var10 = float(np.sum(wts * dev10**2))
        # Until `set_condition_mix` is called, "average weather" means neutral conditions.
        self.condition_mix = [(self.neutral_conditions(), 1.0)]
        self.support_distance = float("inf")
        self.stamp = {
            "what": "kicker field-goal model",
            "seasons": sorted(splits.seasons_in(att)),
            "n_games": int(att["game_id"].n_unique()) if "game_id" in att.columns else None,
            "n_rows": int(att.height),
            "config_sha256_16": config_hash(cfg),
        }
        return self

    def _covariances(self, mode: _Mode) -> None:
        """Approximate posterior covariance of (beta, b) at the mode, given Sigma: the
        inverse of the joint negative Hessian, by blocks (the empirical Bayes
        "conditional variance"; it ignores the uncertainty in Sigma itself)."""
        hbb_inv = np.linalg.inv(mode.hbb)
        a = np.einsum("kqr,krp->kqp", hbb_inv, mode.hbx)
        schur = mode.hxx - np.einsum("kqp,kqm->pm", mode.hbx, a)
        self.cov_beta = np.linalg.inv(schur)
        # Cov(beta, b_k) = -S^-1 Hxb_k Hbb_k^-1 ; Cov(b_k) = Hbb_k^-1 + A_k S^-1 A_k'
        self.cov_beta_b = -np.einsum("pm,kqm->kpq", self.cov_beta, a)
        self.cov_b = hbb_inv + np.einsum("kqp,pm,krm->kqr", a, self.cov_beta, a)

    # ---------------------------------------------------------------- kicker strength

    @property
    def usage_coef(self) -> float:
        """gamma: log-odds per 10 yards of usage (0 when the usage term is off)."""
        if "kicker_usage" not in self.feature_names:
            return 0.0
        return float(self.beta[self.feature_names.index("kicker_usage")])

    def strength_vector(self) -> np.ndarray:
        """w such that w' b is a kicker's random-effect shift at the strength distance."""
        zs = (self.cfg["kickers"]["strength_distance"] - self.cfg["distance"]["center"]) / 10
        return np.array([1.0, zs]) if self.slope else np.array([1.0])

    def kicker_shift(self, kicker_id: str) -> float:
        """A kicker's estimated total log-odds shift at the strength distance (50 yards):
        usage part plus random-effect part. 0 for a kicker not seen in training."""
        j = self._kicker_index.get(kicker_id)
        if j is None:
            return 0.0
        use = (self.usage_by_kicker[kicker_id] - self.usage_league) / 10.0
        return float(self.usage_coef * use + self.b[j] @ self.strength_vector())

    def profile_for_quantile(self, quantile: float) -> tuple[np.ndarray, float]:
        """(effects, usage) of a kicker at population quantile `quantile` of strength.

        Strength is the total log-odds shift at 50 yards, t = gamma * d + w'b, with
        d = (usage - league usage) / 10. Across kickers d and b are independent (the
        usage term makes them so), so Var(t) = gamma^2 Var(d) + w' Sigma w, and the
        expected (d, b) given t are gamma Var(d) t / Var(t) and Sigma w t / Var(t). So a
        top-decile kicker also gets the usage and slope the population pairs with that
        strength. `quantile` 0.5 is the league-average kicker.
        """
        w = self.strength_vector()
        g = self.usage_coef
        var_b = float(w @ self.sigma @ w)
        var_t = g * g * self.usage_var10 + var_b
        t = stats.norm.ppf(float(quantile)) * np.sqrt(var_t)
        eff = self.sigma @ w * t / var_t
        usage = self.usage_league + 10.0 * g * self.usage_var10 * t / var_t
        return eff, float(usage)

    def strength_sd(self) -> float:
        """Standard deviation across kickers of the total 50-yard log-odds shift."""
        w = self.strength_vector()
        return float(np.sqrt(self.usage_coef**2 * self.usage_var10 + w @ self.sigma @ w))

    def ranked_kickers(self, min_attempts: int | None = None) -> pl.DataFrame:
        """Every kicker with his effects, usage, 50-yard shift and attempts."""
        m = self.cfg["kickers"]["min_attempts_ranked"] if min_attempts is None else min_attempts
        df = pl.DataFrame({
            "kicker_id": list(self.kicker_ids), "kicker_name": self.kicker_names,
            "attempts": self.kicker_attempts.astype(int),
            "usage_mean_distance": [self.usage_by_kicker[k] for k in self.kicker_ids],
            "shift_50": [self.kicker_shift(k) for k in self.kicker_ids],
            "u": self.b[:, 0], "v": self.b[:, 1] if self.slope else np.zeros(len(self.b)),
        })  # fmt: skip
        return df.filter(pl.col("attempts") >= m).sort("shift_50", descending=True)

    def top_decile_kickers(self) -> list[str]:
        r = self.ranked_kickers()
        cut = r["shift_50"].quantile(self.cfg["kickers"]["top_quantile"], "nearest")
        return r.filter(pl.col("shift_50") >= cut)["kicker_id"].to_list()

    # ---------------------------------------------------------------- prediction

    def _x(
        self,
        distance: np.ndarray,
        cond: Conditions | None,
        season: float | np.ndarray | None,
        heave: float | np.ndarray,
        usage: float | None = None,
    ) -> np.ndarray:
        d = np.atleast_1d(np.asarray(distance, dtype=float))
        c = cond or self.neutral_conditions()
        full = np.full_like(d, np.nan)
        s = self.reference_season if season is None else season
        return self.design.matrix(
            distance=d,
            outdoor=np.full_like(d, c.outdoor),
            temp=full if c.temp is None else np.full_like(d, c.temp),
            wind=full if c.wind is None else np.full_like(d, c.wind),
            season=np.broadcast_to(np.asarray(s, dtype=float), d.shape),
            heave=np.broadcast_to(np.asarray(heave, dtype=float), d.shape),
            usage=np.full_like(d, self.usage_league if usage is None else usage),
        )

    def neutral_conditions(self) -> Conditions:
        """Outdoors at the training median outdoor temperature and wind."""
        return Conditions(outdoor=1.0, temp=self.design.temp_fill, wind=self.design.wind_fill)

    def _zr(self, distance: np.ndarray) -> np.ndarray:
        z = self.design.z(np.atleast_1d(distance))
        return np.column_stack([np.ones_like(z), z]) if self.slope else np.ones((z.size, 1))

    def logit(
        self,
        distance: float | np.ndarray,
        effects: np.ndarray | None = None,
        cond: Conditions | None = None,
        season: float | np.ndarray | None = None,
        heave: float | np.ndarray = 0.0,
        usage: float | None = None,
    ) -> np.ndarray:
        """Log-odds along `distance` for a kicker with random `effects` and `usage`
        (None: league-average kicker)."""
        x = self._x(distance, cond, season, heave, usage)
        eta = x @ self.beta
        if effects is not None:
            eta = eta + self._zr(distance) @ np.asarray(effects, dtype=float)
        return eta

    def predict_attempts(self, att: pl.DataFrame, kicker: str = "own",
                         override: dict | None = None) -> np.ndarray:  # fmt: skip
        """P(make) for each attempt row.

        `kicker`: "own" uses each kicker's usage and estimated effects (league usage and
        zero effects for a kicker not seen in training); "average" uses the league-average
        kicker. `override` replaces columns before prediction, for example
        {"outdoor": 1, "temp": 60, "wind": 8, "heave": 0}.
        """
        a = self._arrays(att)
        if kicker == "average":
            a["usage"] = np.full(att.height, self.usage_league)
        for k, v in (override or {}).items():
            a[k] = np.broadcast_to(np.asarray(v, dtype=float), a["distance"].shape).copy()
        eta = self.design.matrix(**a) @ self.beta
        if kicker == "own":
            eta = eta + np.einsum("iq,iq->i", self._zr(a["distance"]), self._effects_of(att))
        return 1.0 / (1.0 + np.exp(-eta))

    def _effects_of(self, att: pl.DataFrame) -> np.ndarray:
        q = self.b.shape[1]
        eff = np.zeros((att.height, q))
        if "kicker_id" not in att.columns:
            return eff
        for i, kid in enumerate(att["kicker_id"].to_list()):
            j = self._kicker_index.get(kid)
            if j is not None:
                eff[i] = self.b[j]
        return eff

    def known(self, att: pl.DataFrame) -> np.ndarray:
        """True for attempts by a kicker seen in training."""
        return np.array([k in self._kicker_index for k in att["kicker_id"].to_list()])

    def curve_interval(
        self,
        distance: np.ndarray,
        kicker_id: str | None = None,
        strength: float | None = None,
        cond: Conditions | None = None,
        season: float | None = None,
        level: float = 0.9,
    ) -> dict[str, np.ndarray]:
        """Point estimate and `level` interval of P(make) along `distance`.

        For a named kicker the variance includes his own effect's uncertainty and its
        covariance with the fixed effects. For a kicker given by a strength quantile only
        the fixed effects' uncertainty is counted (his effects are a stated profile).
        Default: the league-average kicker.
        """
        d = np.atleast_1d(np.asarray(distance, dtype=float))
        effects, usage = None, None
        if kicker_id is not None:
            usage = self.usage_by_kicker[kicker_id]
        elif strength is not None:
            effects, usage = self.profile_for_quantile(strength)
        x = self._x(d, cond, season, 0.0, usage)
        zr = self._zr(d)
        var = np.einsum("ip,pm,im->i", x, self.cov_beta, x)
        if kicker_id is not None:
            j = self._kicker_index[kicker_id]
            effects = self.b[j]
            var = var + 2 * np.einsum("ip,pq,iq->i", x, self.cov_beta_b[j], zr)
            var = var + np.einsum("iq,qr,ir->i", zr, self.cov_b[j], zr)
        eta = x @ self.beta + (zr @ effects if effects is not None else 0.0)
        se = np.sqrt(np.maximum(var, 0.0))
        half = stats.norm.ppf(0.5 + level / 2) * se

        def sig(v: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-v))

        return {"p": sig(eta), "lo": sig(eta - half), "hi": sig(eta + half), "se_logit": se}

    def draws(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Joint draws of (beta, b) from the Gaussian approximation at the mode:
        beta (n, p) and b (n, K, q). Sigma is held at its estimate."""
        rng = np.random.default_rng(seed)
        p = self.beta.size
        k, q = self.b.shape
        dim = p + k * q
        cov = np.zeros((dim, dim))
        cov[:p, :p] = self.cov_beta
        # Cov(b_j, b_m) = A_j S^-1 A_m' + [j == m] Hbb_j^-1, with A_j' = -S Cov(beta, b_j).
        a_t = -np.einsum("pm,kpq->kmq", np.linalg.inv(self.cov_beta), self.cov_beta_b)
        for j in range(k):
            sj = slice(p + j * q, p + (j + 1) * q)
            cov[:p, sj] = self.cov_beta_b[j]
            cov[sj, :p] = self.cov_beta_b[j].T
            cov[sj, sj] = self.cov_b[j]
            for m in range(j + 1, k):
                sm = slice(p + m * q, p + (m + 1) * q)
                c = a_t[j].T @ self.cov_beta @ a_t[m]
                cov[sj, sm] = c
                cov[sm, sj] = c.T
        mean = np.concatenate([self.beta, self.b.ravel()])
        z = rng.multivariate_normal(mean, 0.5 * (cov + cov.T), size=n, method="eigh")
        return z[:, :p], z[:, p:].reshape(n, k, q)

    # ---------------------------------------------------------------- planner interface

    def _profile(self, strength: float | str | None) -> tuple[np.ndarray | None, float | None]:
        if strength is None:
            return None, None
        if isinstance(strength, str):
            return self.b[self._kicker_index[strength]], self.usage_by_kicker[strength]
        return self.profile_for_quantile(float(strength))

    def _monotone(self, distance: np.ndarray, fn: Any) -> np.ndarray:
        """fn(distance) made non-increasing in distance: each value is capped by the
        smallest value at any whole-yard distance from 15 up to it. On the training fit
        this changes nothing (see the results file); it guards the planner against a
        spline wiggle at short range in other fits."""
        d = np.atleast_1d(np.asarray(distance, dtype=float))
        grid = np.arange(15.0, max(16.0, np.ceil(d.max())) + 1)
        run = np.minimum.accumulate(fn(grid))
        idx = np.clip(np.floor(d).astype(int) - 15, 0, grid.size - 1)
        return np.minimum(fn(d), np.where(d >= 15, run[idx], np.inf))

    def p_make(
        self,
        distance: float | np.ndarray,
        strength: float | str | None = None,
        roof: str | None = None,
        temp: float | None = None,
        wind: float | None = None,
        season: int | None = None,
        heave: bool = False,
    ) -> float | np.ndarray:
        """P(field goal good) for the world-model planner.

        distance  kick distance in yards (the repo convention: yardline_100 + 18).
        strength  None or 0.5: league-average kicker. A float in (0, 1): a kicker at that
                  population quantile of 50-yard strength (0.9 = top decile). A string:
                  a kicker id seen in training (his usage and shrunken effects).
        roof      None: average over the weather of every training game (a planner that
                  does not know the weather should average over it, not plug in a median).
                  "dome"/"closed": indoors. "open": retractable roof left open.
                  "outdoors": outdoors at `temp` and `wind` (training medians when
                  not given).
        season    default: the last training season (2021). Later seasons extrapolate
                  the linear trend; see `in_support` and the notes before relying on it.
        heave     True for an end-of-half last-play attempt.
        Returns a float for scalar input, an array otherwise. Non-increasing in distance.
        """
        d = np.asarray(distance, dtype=float)
        eff, usage = self._profile(strength)
        if roof is None:
            mix = self.condition_mix
        else:
            cond = Conditions(roof_code(roof), self.design.temp_fill if temp is None else temp,
                              self.design.wind_fill if wind is None else wind)  # fmt: skip
            mix = [(cond, 1.0)]

        def fn(dd: np.ndarray) -> np.ndarray:
            ps = np.zeros(np.atleast_1d(dd).shape)
            for c, wgt in mix:
                eta = self.logit(dd, eff, c, season, float(heave), usage)
                ps = ps + wgt / (1.0 + np.exp(-eta))
            return ps

        p = self._monotone(d, fn)
        return float(p[0]) if d.ndim == 0 else p.reshape(d.shape)

    def in_support(self, distance: float | np.ndarray, season: int | None = None
                   ) -> bool | np.ndarray:  # fmt: skip
        """True where the prediction is backed by data (see `support` in the config):
        distance at most the support distance and season within the training seasons."""
        d = np.asarray(distance, dtype=float)
        ok = d <= self.support_distance
        if season is not None and int(season) > self.reference_season:
            ok = np.zeros_like(ok, dtype=bool)
        return bool(ok) if d.ndim == 0 else ok

    def capped_rate(self, simulated: float | np.ndarray, distance: float | np.ndarray,
                    strength: float | None = 0.9, level: float = 0.9
                    ) -> float | np.ndarray:  # fmt: skip
        """POST HOC (designed after seeing the world-model run): cap a simulator's make
        rate at the upper end of this model's `level` interval for a kicker of the given
        strength (default top decile), averaged over training-game weather. Below the cap
        the simulator is left alone. The planner may instead override its rate with
        `p_make` directly; either is a post hoc change."""
        d = np.atleast_1d(np.asarray(distance, dtype=float))
        hi = np.zeros(d.shape)
        for c, wgt in self.condition_mix:
            hi = hi + wgt * self.curve_interval(d, strength=strength, cond=c, level=level)["hi"]
        out = np.minimum(np.asarray(simulated, dtype=float), hi)
        return float(out[0]) if np.ndim(distance) == 0 else out

    # ---------------------------------------------------------------- persistence

    def set_condition_mix(self, games: pl.DataFrame, max_points: int = 200) -> None:
        """The weather of every training game, compressed to at most `max_points`
        weighted conditions (roof, temp to 10 F, wind to 5 mph), for `p_make(roof=None)`."""
        g = games.select(
            _roof_expr().alias("outdoor"),
            (pl.col("temp").cast(pl.Float64) / 10).round(0).mul(10).alias("temp"),
            (pl.col("wind").cast(pl.Float64) / 5).round(0).mul(5).alias("wind"),
        )
        g = g.with_columns(
            pl.when(pl.col("outdoor") < 1).then(None).otherwise(pl.col("temp")).alias("temp"),
            pl.when(pl.col("outdoor") < 1).then(None).otherwise(pl.col("wind")).alias("wind"),
        )
        c = g.group_by("outdoor", "temp", "wind").len().sort("len", descending=True)
        c = c.head(max_points)
        tot = c["len"].sum()
        self.condition_mix = [
            (Conditions(r["outdoor"], r["temp"], r["wind"]), r["len"] / tot)
            for r in c.iter_rows(named=True)
        ]

    def save(self, path: Path = MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path = MODEL_PATH) -> FGMakeModel:
        return joblib.load(path)


def copy_cfg(cfg: dict) -> dict:
    return json.loads(json.dumps(cfg))


def support_distance(
    model: FGMakeModel, att: pl.DataFrame, cfg: dict | None = None
) -> dict[str, Any]:
    """The support distance by the rule in `configs/kicker.yaml` (`support`), with the
    per-distance evidence. `att` are the training attempts."""
    cfg = cfg or model.cfg
    s = cfg["support"]
    top = model.top_decile_kickers()
    top_att = att.filter(pl.col("kicker_id").is_in(top))
    dist = top_att["distance"].to_numpy()
    rows = []
    support = None
    for d in range(int(s["start_distance"]), 76):
        n = int(np.sum(np.abs(dist - d) <= s["half_window"]))
        widths = []
        for kid in top:
            ci = model.curve_interval(np.array([d]), kicker_id=kid, level=s["interval_level"])
            widths.append(float(ci["hi"][0] - ci["lo"][0]))
        med = float(np.median(widths))
        ok = n >= s["min_attempts"] and med <= s["max_interval_width"]
        rows.append({"distance": d, "top_decile_attempts_in_window": n,
                     "median_interval_width": med, "passes": ok})  # fmt: skip
        if ok and (support is None or support == d - 1):
            support = d
    if support is None:
        support = int(s["start_distance"]) - 1
    first_fail = next((r for r in rows if r["distance"] == support + 1), None)
    reason = None
    if first_fail is not None:
        bits = []
        if first_fail["top_decile_attempts_in_window"] < s["min_attempts"]:
            bits.append("too few top-decile attempts")
        if first_fail["median_interval_width"] > s["max_interval_width"]:
            bits.append("interval too wide")
        reason = " and ".join(bits)
    widths = [r["median_interval_width"] for r in rows]
    counts_fail = [r["distance"] for r in rows
                   if r["top_decile_attempts_in_window"] < s["min_attempts"]]  # fmt: skip
    width_fail = [r["distance"] for r in rows if r["median_interval_width"]
                  > s["max_interval_width"]]  # fmt: skip
    return {"support_distance": int(support), "first_failure_reason": reason,
            "top_decile_kickers": len(top),
            "top_decile_kicker_names": [model.kicker_names[model._kicker_index[k]]
                                        for k in top],
            "first_distance_failing_attempt_count": counts_fail[0] if counts_fail else None,
            "first_distance_failing_interval_width": width_fail[0] if width_fail else None,
            "max_median_interval_width": float(max(widths)),
            "by_distance": rows}  # fmt: skip


def shapley(values: dict[frozenset, float], factors: Sequence[str]) -> dict[str, float]:
    """Shapley shares of v(all) - v(none) for a value function over factor subsets."""
    from itertools import combinations
    from math import factorial

    n = len(factors)
    out = {}
    for f in factors:
        rest = [g for g in factors if g != f]
        tot = 0.0
        for r in range(n):
            for sub in combinations(rest, r):
                s = frozenset(sub)
                wgt = factorial(r) * factorial(n - r - 1) / factorial(n)
                tot += wgt * (values[s | {f}] - values[s])
        out[f] = tot
    return out


def select_knots(att: pl.DataFrame, cfg: dict) -> tuple[list[float], list[dict[str, Any]]]:
    """Fit each knot candidate of `cfg["distance"]["knot_candidates"]` on training
    attempts and pick the lowest AIC (-2 x Laplace log marginal + 2 x fixed effects).
    Returns the chosen knots and one row per candidate (with the in-sample check on
    60+ yard attempts that prompted the candidate list)."""
    long = att.filter(pl.col("distance") >= 60)
    rows = []
    for knots in cfg["distance"]["knot_candidates"]:
        c = copy_cfg(cfg)
        c["distance"]["knots"] = list(knots)
        m = FGMakeModel(c).fit(att)
        rows.append({
            "knots": list(knots), "fixed_effects": int(m.beta.size),
            "log_marginal": float(m.log_marginal),
            "aic": float(-2 * m.log_marginal + 2 * m.beta.size),
            "in_sample_60_plus": {"attempts": int(long.height),
                                  "raw": float(long["made"].mean()) if long.height else None,
                                  "predicted": float(m.predict_attempts(long).mean())
                                  if long.height else None},
        })  # fmt: skip
    best = min(rows, key=lambda r: r["aic"])
    for r in rows:
        r["chosen"] = r is best
    return list(best["knots"]), rows


FACTORS = ("kicker", "conditions", "heave", "era")


def selection_decomposition(
    model: FGMakeModel,
    att: pl.DataFrame,
    beta_draws: np.ndarray | None = None,
    b_draws: np.ndarray | None = None,
) -> dict[str, Any]:
    """Split the raw make rate of `att` into what the model attributes to each source.

    v(S) is the mean predicted make rate of the attempts when the factors in S keep their
    observed values and the others are set to neutral:
      kicker      each kicker's usage and own effects, or the league-average kicker
      conditions  observed roof and weather, or outdoors at the training median weather
      heave       observed heave flag, or 0
      era         each attempt's season, or the mean training season
    v(all) is the model's fit to the attempts and v(none) the "selection-free" rate: a
    league-average kicker in neutral conditions, not a heave, at the training-average
    season, kicking from exactly the same distances. The gap v(all) - v(none) is split
    by Shapley values (the average of each factor's marginal effect over every order of
    switching the factors on), so the shares add up exactly. raw - v(all) is fit error.
    With draws of (beta, b), each quantity also gets a 90% interval.
    """
    from itertools import combinations

    a = model._arrays(att)
    eff = model._effects_of(att)
    zr = model._zr(a["distance"])
    nc = model.neutral_conditions()
    subsets = [frozenset(c) for r in range(len(FACTORS) + 1)
               for c in combinations(FACTORS, r)]  # fmt: skip
    xs: dict[frozenset, np.ndarray] = {}
    for s in subsets:
        arr = dict(a)
        if "kicker" not in s:
            arr["usage"] = np.full_like(a["distance"], model.usage_league)
        if "conditions" not in s:
            arr["outdoor"] = np.ones_like(a["distance"])
            arr["temp"] = np.full_like(a["distance"], nc.temp)
            arr["wind"] = np.full_like(a["distance"], nc.wind)
        if "heave" not in s:
            arr["heave"] = np.zeros_like(a["distance"])
        if "era" not in s:
            arr["season"] = np.full_like(a["distance"], model.mean_season)
        xs[s] = model.design.matrix(**arr)

    def values(beta: np.ndarray, b_eff: np.ndarray) -> dict[frozenset, float]:
        out = {}
        for s in subsets:
            eta = xs[s] @ beta
            if "kicker" in s:
                eta = eta + np.einsum("iq,iq->i", zr, b_eff)
            out[s] = float(np.mean(1.0 / (1.0 + np.exp(-eta))))
        return out

    v = values(model.beta, eff)
    raw = float(att["made"].mean())
    share = shapley(v, FACTORS)
    res: dict[str, Any] = {
        "attempts": int(att.height),
        "raw": raw,
        "model_fit": v[frozenset(FACTORS)],
        "selection_free": v[frozenset()],
        "fit_error": raw - v[frozenset(FACTORS)],
        "contributions": share,
        "total_selection": v[frozenset(FACTORS)] - v[frozenset()],
    }
    if beta_draws is not None and b_draws is not None:
        idx = np.array([model._kicker_index.get(k, -1) for k in att["kicker_id"].to_list()])
        sims = []
        for bt, bb in zip(beta_draws, b_draws, strict=True):
            be = np.where(idx[:, None] >= 0, bb[np.maximum(idx, 0)], 0.0)
            vv = values(bt, be)
            sh = shapley(vv, FACTORS)
            sims.append([vv[frozenset(FACTORS)], vv[frozenset()],
                         vv[frozenset(FACTORS)] - vv[frozenset()]]
                        + [sh[f] for f in FACTORS])  # fmt: skip
        sims_a = np.array(sims)
        lo, hi = np.quantile(sims_a, [0.05, 0.95], axis=0)
        keys = ["model_fit", "selection_free", "total_selection"] + [
            f"contribution_{f}" for f in FACTORS]
        res["interval_90"] = {k: [float(lo[i]), float(hi[i])] for i, k in enumerate(keys)}
    return res


# --------------------------------------------------------------------------------------
# A synthetic world with selection built in (tests and the results file)
# --------------------------------------------------------------------------------------


def synthetic_true_logit(distance: np.ndarray) -> np.ndarray:
    """The league-average kicker's log-odds in the synthetic world."""
    z = (np.asarray(distance, dtype=float) - 45.0) / 10.0
    return 2.2 - 1.3 * z - 0.25 * z * z


def simulate_selected_attempts(
    seed: int = 0, n_kickers: int = 80, per_kicker: int = 150, sd_strength: float = 0.5
) -> tuple[pl.DataFrame, np.ndarray]:
    """Attempts where coaches send strong kickers out for longer kicks.

    Kicker k has strength u_k ~ Normal(0, sd_strength) in log-odds. His attempt distances
    are Normal(38 + 12 u_k, 9), rounded and clipped to 18 to 66 yards, so a kicker one
    standard deviation above average attempts kicks about 6 yards longer. P(make) is
    logistic(synthetic_true_logit(distance) + u_k). Season 2015 throughout (a training
    season); no roof, weather or heave columns. Returns the attempts and the true u.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, sd_strength, n_kickers)
    frames = []
    for k in range(n_kickers):
        d = np.clip(np.rint(rng.normal(38.0 + 12.0 * u[k], 9.0, per_kicker)), 18, 66)
        p = 1.0 / (1.0 + np.exp(-(synthetic_true_logit(d) + u[k])))
        made = (rng.random(per_kicker) < p).astype(np.int8)
        frames.append(pl.DataFrame({
            "game_id": [f"g{k}_{j}" for j in range(per_kicker)],
            "season": np.full(per_kicker, 2015, dtype=np.int32),
            "kicker_id": [f"K{k:03d}"] * per_kicker,
            "kicker_name": [f"Kicker {k}"] * per_kicker,
            "distance": d.astype(float), "made": made,
        }))  # fmt: skip
    return pl.concat(frames), u


def synthetic_cfg(cfg: dict | None = None, usage: bool = True) -> dict:
    """The model config for the synthetic world: distance and kicker terms only."""
    c = copy_cfg(cfg or load_config())
    c["terms"] = {"roof": False, "weather": False, "season": False, "heave": False,
                  "kicker_slope": True, "kicker_usage": usage}  # fmt: skip
    return c


def synthetic_recovery(seed: int, cfg: dict | None = None, min_distance: int = 55
                       ) -> dict[str, float]:  # fmt: skip
    """Raw rate, true average-kicker rate and the two models' average-kicker rates on the
    synthetic attempts of `min_distance` yards or more."""
    att, _ = simulate_selected_attempts(seed)
    long = att.filter(pl.col("distance") >= min_distance)
    truth = float(np.mean(1.0 / (1.0 + np.exp(-synthetic_true_logit(long["distance"])))))
    with_usage = FGMakeModel(synthetic_cfg(cfg, usage=True)).fit(att)
    plain = FGMakeModel(synthetic_cfg(cfg, usage=False)).fit(att)
    return {
        "seed": seed, "attempts": int(long.height), "raw": float(long["made"].mean()),
        "truth_average_kicker": truth,
        "model_with_usage_term": float(with_usage.predict_attempts(long, "average").mean()),
        "plain_random_effects": float(plain.predict_attempts(long, "average").mean()),
        "usage_coefficient": with_usage.usage_coef,
    }  # fmt: skip
