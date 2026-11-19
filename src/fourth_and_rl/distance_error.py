"""M9 "Was it really fourth and 1?": the hidden fraction of a yard in recorded distance.

Play-by-play records yards to go as a whole number. The true distance D from the ball to
the line to gain is continuous. This module builds four things:

1. A measurement-error model of D given the recorded distance d.
   Every spot on the field is recorded as the whole yard (the "cell") it lies in, and the
   record is kept consistent: a ball short of the line is always recorded at least one
   yard short. Writing the line to gain as L = c_L + v, with c_L its cell and v in [0, 1)
   its position inside that cell, the recorded distance is

       d = 1                     if 0 < D <= 1 + v
       d = k  (k >= 2)           if k - 1 + v < D <= k + v

   i.e. D rounded up after subtracting the hidden offset v. The offset is not recorded,
   but it leaves a footprint: a play that ends past the line but inside the line's own
   cell is recorded as gaining exactly d, and that cell is only (1 - v) yards wide, so
   plays gaining exactly d are rarer than their neighbours (`fit_line_offset`). An
   `ErrorModel` is a distribution for v (uniform, a beta with a given mean, or a point),
   an optional Gaussian spot error, and the offset used in goal-to-go, where the line to
   gain is the goal line. `distance_kernel` turns a model into the distribution of D
   given d, assuming the true distance has a flat density inside one recorded yard.

2. A continuous conversion curve (`fit_latent_curve`). Conversion probability is
   logit q(D) = offset + h(D), with h non-increasing and piecewise linear in D, fitted by
   maximum likelihood of the observed (recorded) outcomes, each averaged over the
   distance kernel: P(convert | d) = sum_j w_j(d) * q(D_j). That is a deconvolution: it
   finds the curve in true distance whose average over the hidden fraction reproduces
   the whole-yard rates. The offset carries field position and down from the existing
   whole-yard go model, so only the distance shape is new.

3. Propagation into a decision (`propagate`). The one-step go value is linear in the
   conversion probability: go(p) = p * wp_if_convert + (1 - p) * wp_if_fail. Each play's
   curve is anchored so that its kernel average equals the engine's whole-yard p, then
   evaluated across the plausible range of D. Kicks do not depend on the distance.

4. `ConversionOverride`, a components wrapper that substitutes a conversion probability
   into the real decision engine, used to check the linear shortcut on real states.

Public API: load_config, ErrorModel, error_models, offset_grid, distance_kernel,
KernelTable, kernel_table, kernel_quantiles, line_offset_counts, fit_line_offset,
pooled_line_offset, simulate_recorded, LatentCurve, CellCounts, aggregate_cells,
fit_latent_curve, implied_whole_yard, anchor_offsets, propagate, summarize_flips,
ConversionOverride, go_value, breakeven_probability.
Teaching notes: docs/notes/m9_distance_error.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from fourth_and_rl import splits
from fourth_and_rl.paths import CONFIGS


def load_config(path: Path | None = None) -> dict:
    """Read configs/distance_error.yaml."""
    with open(path or CONFIGS / "distance_error.yaml") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------------------
# 1. Measurement-error models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorModel:
    """How the true distance relates to the recorded one.

    offset_kind: "uniform" (v ~ Uniform(0, 1)), "beta" (mean `offset_mean`, concentration
    `offset_concentration`; 2 with mean 0.5 is the uniform) or "point" (v = offset_mean).
    spot_sd: standard deviation in yards of an extra Gaussian spot error added to D.
    goal_to_go_offset: v when the line to gain is the goal line (a point, not a draw).
    """

    name: str
    offset_kind: str = "uniform"
    offset_mean: float = 0.5
    offset_concentration: float = 2.0
    spot_sd: float = 0.0
    goal_to_go_offset: float = 0.5
    label: str = ""

    def __post_init__(self) -> None:
        if self.offset_kind not in ("uniform", "beta", "point"):
            raise ValueError(f"unknown offset_kind {self.offset_kind!r}")
        if not 0.0 <= self.offset_mean <= 1.0 or not 0.0 <= self.goal_to_go_offset <= 1.0:
            raise ValueError("offsets are fractions of a yard in [0, 1]")
        if self.spot_sd < 0:
            raise ValueError("spot_sd must be >= 0")


def error_models(config: dict, fitted_mean: float | None = None) -> list[ErrorModel]:
    """The error models listed in the config. A mean written as "fitted" is replaced by
    `fitted_mean` (the estimate from `fit_line_offset`)."""
    out = []
    for spec in config["error_models"]:
        mean = spec.get("offset_mean", 0.5)
        if mean == "fitted":
            if fitted_mean is None:
                raise ValueError(f"model {spec['name']} needs the fitted offset mean")
            mean = float(np.clip(fitted_mean, 0.01, 0.99))
        out.append(ErrorModel(
            name=spec["name"],
            offset_kind=spec.get("offset_kind", "uniform"),
            offset_mean=float(mean),
            offset_concentration=float(spec.get("offset_concentration", 2.0)),
            spot_sd=float(spec.get("spot_sd", 0.0)),
            goal_to_go_offset=float(spec.get("goal_to_go_offset", 0.5)),
            label=spec.get("label", spec["name"]),
        ))  # fmt: skip
    return out


def offset_grid(model: ErrorModel, goal_to_go: bool = False, n: int = 400
                ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Quadrature points and weights for the hidden offset v of the line to gain."""
    if goal_to_go:
        return np.array([model.goal_to_go_offset]), np.array([1.0])
    if model.offset_kind == "point":
        return np.array([model.offset_mean]), np.array([1.0])
    v = (np.arange(n) + 0.5) / n
    if model.offset_kind == "uniform":
        return v, np.full(n, 1.0 / n)
    a = model.offset_mean * model.offset_concentration
    b = (1.0 - model.offset_mean) * model.offset_concentration
    w = beta_dist.pdf(v, a, b)
    return v, w / w.sum()


def distance_kernel(
    d: int, goal_to_go: bool, model: ErrorModel, step: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Distribution of the true distance D given recorded distance `d`.

    Returns (nodes, weights) on a grid of width `step`; weights sum to 1. With a flat
    density of D inside each recorded yard, the posterior is proportional to the chance
    that the offset puts D in the recorded bin:
        w(D) proportional to  sum over v of P(v) * 1{ D in bin(d, v) }
    where with a spot error the bin is the one the noisy, seen distance falls in, and D is
    truncated at D > 0 (a ball past the line would have been a first down).
    """
    d = int(d)
    if d < 1:
        raise ValueError("recorded distance must be at least 1")
    pad = 4.0 * model.spot_sd + step
    lo = max(0.0, d - 1.0 - pad)
    hi = d + 1.0 + pad
    nodes = np.arange(lo + step / 2, hi, step)
    v, pv = offset_grid(model, goal_to_go)
    # A recorded 1 holds everything up to 1 + v; with a spot error the seen distance can
    # even fall below zero and is still recorded as 1, so its bin has no lower end.
    lower = np.full(v.size, -np.inf) if d == 1 else d - 1.0 + v
    upper = d + v
    if model.spot_sd > 0:
        s = model.spot_sd
        inside = norm.cdf((upper[None, :] - nodes[:, None]) / s) - norm.cdf(
            (lower[None, :] - nodes[:, None]) / s)
    else:
        inside = ((nodes[:, None] > lower[None, :]) & (nodes[:, None] <= upper[None, :])
                  ).astype(float)  # fmt: skip
    w = inside @ pv
    w = np.where(nodes > 0, w, 0.0)
    return nodes, w / w.sum()


@dataclass
class KernelTable:
    """Distance kernels for every recorded distance 1..max_d and goal-to-go flag,
    padded to one width. Cell index g = (d - 1) * 2 + goal_to_go."""

    model: ErrorModel
    max_d: int
    nodes: np.ndarray  # (n_cells, J)
    weights: np.ndarray  # (n_cells, J), zero padding

    def cell(self, d: np.ndarray | int, goal_to_go: np.ndarray | bool | int) -> np.ndarray:
        d = np.clip(np.asarray(d, dtype=int), 1, self.max_d)
        return (d - 1) * 2 + np.asarray(goal_to_go, dtype=int)

    def mean(self) -> np.ndarray:
        return (self.nodes * self.weights).sum(axis=1)


def kernel_table(model: ErrorModel, max_d: int = 30, step: float = 0.02) -> KernelTable:
    """All kernels of one error model (see `distance_kernel`)."""
    ks = [distance_kernel(d, bool(g), model, step) for d in range(1, max_d + 1) for g in (0, 1)]
    width = max(k[0].size for k in ks)
    nodes = np.zeros((len(ks), width))
    weights = np.zeros((len(ks), width))
    for i, (x, w) in enumerate(ks):
        nodes[i, : x.size] = x
        nodes[i, x.size :] = x[-1]
        weights[i, : x.size] = w
    return KernelTable(model, max_d, nodes, weights)


def kernel_quantiles(table: KernelTable, cells: np.ndarray, q: float) -> np.ndarray:
    """The q-quantile of D for each cell index."""
    cum = np.cumsum(table.weights[cells], axis=1)
    j = np.argmax(cum >= q - 1e-12, axis=1)
    return table.nodes[cells, j]


# --------------------------------------------------------------------------------------
# 2. Evidence for the offset: plays that gain exactly the recorded distance
# --------------------------------------------------------------------------------------


def line_offset_plays(plays: pl.DataFrame, downs: tuple[int, int] = (2, 4)) -> pl.DataFrame:
    """Runs and completed passes with a clean spot: no penalty, turnover or incompletion
    (an incomplete pass is recorded as no gain, a spike in the gain histogram that has
    nothing to do with spots), not goal-to-go (the line is the goal line there). Adds
    k = yards_gained - ydstogo, the recorded end spot relative to the line to gain."""
    return plays.filter(
        pl.col("play_type").is_in(["run", "pass"])
        & pl.col("down").is_between(*downs)
        & (pl.col("penalty").fill_null(0) == 0)
        & (pl.col("interception").fill_null(0) == 0)
        & (pl.col("fumble_lost").fill_null(0) == 0)
        & (pl.col("goal_to_go").fill_null(0) == 0)
        & pl.col("yards_gained").is_not_null()
        & ~pl.col("desc").fill_null("").str.contains("(?i)incomplete|spiked")
    ).with_columns((pl.col("yards_gained") - pl.col("ydstogo")).cast(pl.Int64).alias("k"))


def line_offset_counts(
    k: np.ndarray, window: int = 4, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Counts of plays ending k = -window..window yards from the recorded line (each play
    counted `weights` times, for bootstrap multiplicities)."""
    ks = np.arange(-window, window + 1)
    k = np.asarray(k)
    inside = np.abs(k) <= window
    w = np.ones(k.size) if weights is None else np.asarray(weights, dtype=float)
    counts = np.bincount(k[inside] + window, weights=w[inside], minlength=ks.size)
    return ks, counts.astype(float)


def fit_line_offset(ks: np.ndarray, counts: np.ndarray) -> dict[str, float]:
    """Estimate the mean offset m = E[v] from the dip at k = 0.

    Model (Poisson): E[count_k] = exp(a + b*k + c*k^2) * width_k, with width 1 for every
    recorded yard except the two next to the line: the last yard short of it is
    (1 + v) yards wide (it also holds balls inside the line's own cell but short of the
    line) and the yard holding the line, past it, is (1 - v) wide. With a smooth density
    of where plays end, the deficit at k = 0 and the excess at k = -1 identify m.
    Returns the estimate, a standard error from the curvature of the likelihood, and n.
    """
    ks = np.asarray(ks, dtype=float)
    counts = np.asarray(counts, dtype=float)

    def nll(p: np.ndarray) -> float:
        a, b, c, m = p
        w = np.ones_like(ks)
        w[ks == -1] = 1 + m
        w[ks == 0] = 1 - m
        mu = np.exp(a + b * ks + c * ks**2) * w
        return float((mu - counts * np.log(mu)).sum())

    start = np.array([np.log(max(counts.mean(), 1.0)), 0.0, 0.0, 0.5])
    res = minimize(nll, start, method="L-BFGS-B",
                   bounds=[(None, None), (None, None), (None, None), (-0.99, 0.99)])  # fmt: skip
    m = float(res.x[3])
    h = 1e-4
    f0, fp, fm = nll(res.x), nll(res.x + [0, 0, 0, h]), nll(res.x - [0, 0, 0, h])
    # Profile-free curvature in m alone understates the error slightly; the bootstrap in
    # the script gives the reported interval.
    curv = (fp - 2 * f0 + fm) / h**2
    se = float(1.0 / np.sqrt(curv)) if curv > 0 else float("nan")
    return {"mean_offset": m, "se": se, "n": int(counts.sum())}


def pooled_line_offset(
    plays: pl.DataFrame, distances: tuple[int, int], downs: tuple[int, int], window: int = 4
) -> dict[str, Any]:
    """`fit_line_offset` on the counts pooled over the recorded distances in `distances`
    (inclusive), plus the fit for each distance on its own."""
    sel = line_offset_plays(plays, downs).filter(pl.col("ydstogo").is_between(*distances))
    per = {}
    for d in range(distances[0], distances[1] + 1):
        k = sel.filter(pl.col("ydstogo") == d)["k"].to_numpy()
        per[str(d)] = fit_line_offset(*line_offset_counts(k, window))
    pooled = fit_line_offset(*line_offset_counts(sel["k"].to_numpy(), window))
    return {**pooled, "by_distance": per}


# --------------------------------------------------------------------------------------
# Simulation with a known truth
# --------------------------------------------------------------------------------------


def sample_offsets(model: ErrorModel, n: int, rng: np.random.Generator) -> np.ndarray:
    if model.offset_kind == "uniform":
        return rng.uniform(0.0, 1.0, n)
    if model.offset_kind == "point":
        return np.full(n, model.offset_mean)
    a = model.offset_mean * model.offset_concentration
    b = (1.0 - model.offset_mean) * model.offset_concentration
    return rng.beta(a, b, n)


def record_distance(true_distance: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """The recording rule: D rounded up after subtracting the offset, at least 1."""
    return np.maximum(1, np.ceil(np.asarray(true_distance) - np.asarray(offset))).astype(int)


def record_end_spot(end: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Recorded end spot relative to the line's cell (k), for a play ending `end` yards
    past the line to gain (negative = short). The line sits at `offset` inside cell 0.
    A ball short of the line is recorded at least one yard short (k <= -1)."""
    pos = np.asarray(offset) + np.asarray(end)  # position relative to the cell's start
    k = np.floor(pos).astype(int)
    return np.where(end >= 0, k, np.minimum(k, -1))


def simulate_recorded(
    n: int,
    model: ErrorModel,
    rng: np.random.Generator,
    curve: Any | None = None,
    true_distance_scale: float = 3.0,
    max_distance: float = 15.0,
    gain_mean: float = 4.0,
    gain_sd: float = 6.0,
) -> dict[str, np.ndarray]:
    """Plays with a known true distance, recorded by the rule of this module.

    True distance: exponential with mean `true_distance_scale`, truncated to
    (0, max_distance]. Offset: drawn from `model`, plus the spot error on D. Conversion:
    Bernoulli(curve(D)) if a curve (callable on D) is given. Yards gained: a Gaussian
    true gain, recorded with `record_end_spot`, so the k histogram carries the offset.
    """
    D = rng.exponential(true_distance_scale, 4 * n)
    D = D[(D > 0) & (D <= max_distance)][:n]
    v = sample_offsets(model, D.size, rng)
    if model.spot_sd > 0:
        # The spot error is on what was recorded, not on the truth: the recorded bin is
        # decided by a noisy version of D.
        seen = np.maximum(D + rng.normal(0.0, model.spot_sd, D.size), 1e-6)
    else:
        seen = D
    d = record_distance(seen, v)
    gain = rng.normal(gain_mean, gain_sd, D.size)
    k = record_end_spot(gain - D, v)
    out = {"true_distance": D, "offset": v, "recorded": d, "k": k}
    if curve is not None:
        out["converted"] = rng.uniform(size=D.size) < curve(D)
    return out


# --------------------------------------------------------------------------------------
# 3. The continuous conversion curve
# --------------------------------------------------------------------------------------


@dataclass
class LatentCurve:
    """logit q(D) = offset + h(D), h piecewise linear between `knots`, non-increasing,
    constant beyond the last knot."""

    knots: np.ndarray
    values: np.ndarray
    model_name: str = ""
    penalty: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    def h(self, distance: np.ndarray | float) -> np.ndarray:
        return np.interp(np.asarray(distance, dtype=float), self.knots, self.values)

    def p(self, distance: np.ndarray | float, offset: np.ndarray | float = 0.0) -> np.ndarray:
        return expit(np.asarray(offset, dtype=float) + self.h(distance))

    def to_dict(self) -> dict[str, Any]:
        return {"knots": self.knots.tolist(), "values": self.values.tolist(),
                "model": self.model_name, "penalty": self.penalty, **self.info}  # fmt: skip


@dataclass
class CellCounts:
    """Aggregated attempts: kernel cell index, offset (logit scale), attempts, successes."""

    cell: np.ndarray
    offset: np.ndarray
    n: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        keep = self.n > 0
        self.cell = np.asarray(self.cell)[keep].astype(int)
        self.offset = np.asarray(self.offset, dtype=float)[keep]
        self.n = np.asarray(self.n, dtype=float)[keep]
        self.y = np.asarray(self.y, dtype=float)[keep]


def conversion_attempts(plays: pl.DataFrame, downs: tuple[int, ...] = (3, 4)) -> pl.DataFrame:
    """Third- and fourth-down runs and passes, as the whole-yard go model selects them,
    with `converted` (first down or offensive touchdown, ball kept) and `goal_to_go`."""
    df = plays.filter(
        pl.col("play_type").is_in(["run", "pass"])
        & pl.col("down").cast(pl.Int64).is_in([int(x) for x in downs])
        & pl.col("yardline_100").is_between(1, 99)
        & (pl.col("ydstogo") >= 1)
        & pl.col("yards_gained").is_not_null()
    )
    turnover = (pl.col("interception").fill_null(0) == 1) | (
        pl.col("fumble_lost").fill_null(0) == 1
    )
    off_td = (pl.col("touchdown").fill_null(0) == 1) & (pl.col("td_team") == pl.col("posteam"))
    return df.with_columns(
        (((pl.col("first_down").fill_null(0) == 1) | off_td.fill_null(False)) & ~turnover)
        .alias("converted"),
        (pl.col("goal_to_go").fill_null(0) == 1).alias("gtg"),
    )


def aggregate_cells(
    attempts: pl.DataFrame,
    offset_fn: Any,
    table: KernelTable,
    weight: np.ndarray | None = None,
) -> CellCounts:
    """Group attempts by (recorded distance, goal-to-go, yard line, down). `offset_fn(yl,
    down)` gives the logit offset of each group (field position and down from the
    whole-yard model). `weight` multiplies each attempt (bootstrap multiplicities)."""
    d = np.clip(attempts["ydstogo"].to_numpy().astype(int), 1, table.max_d)
    g = attempts["gtg"].to_numpy().astype(int)
    yl = attempts["yardline_100"].to_numpy().astype(int)
    down = attempts["down"].to_numpy().astype(int)
    y = attempts["converted"].to_numpy().astype(float)
    w = np.ones(d.size) if weight is None else np.asarray(weight, dtype=float)
    key = ((d * 2 + g) * 100 + yl) * 5 + down
    uniq, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv, weights=w, minlength=uniq.size)
    s = np.bincount(inv, weights=w * y, minlength=uniq.size)
    u_down = uniq % 5
    u_yl = (uniq // 5) % 100
    u_dg = uniq // 500
    u_d, u_g = u_dg // 2, u_dg % 2
    cell = table.cell(u_d, u_g)
    return CellCounts(cell, offset_fn(u_yl, u_down), n, s)


def _penalty_matrix(knots: np.ndarray) -> np.ndarray:
    """Rows give the change in slope at each interior knot (a second difference that
    respects uneven spacing), scaled by the mean spacing of the two sides."""
    k = knots.size
    rows = []
    for i in range(1, k - 1):
        left, right = knots[i] - knots[i - 1], knots[i + 1] - knots[i]
        r = np.zeros(k)
        r[i - 1], r[i], r[i + 1] = 1 / left, -1 / left - 1 / right, 1 / right
        rows.append(r * 0.5 * (left + right))
    return np.array(rows)


def _interp_matrix(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Matrix B with B @ values = np.interp(x, knots, values)."""
    x = np.clip(np.asarray(x, dtype=float).ravel(), knots[0], knots[-1])
    j = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, knots.size - 2)
    t = (x - knots[j]) / (knots[j + 1] - knots[j])
    B = np.zeros((x.size, knots.size))
    B[np.arange(x.size), j] = 1 - t
    B[np.arange(x.size), j + 1] = t
    return B


def fit_latent_curve(
    cells: CellCounts,
    table: KernelTable,
    knots: np.ndarray | list[float],
    penalty: float,
    max_iter: int = 2000,
) -> LatentCurve:
    """Deconvolve: the non-increasing h(D) maximizing

        sum over cells of  y log P + (n - y) log(1 - P)  -  penalty * |S h|^2
        P(cell) = sum_j w_j(cell) * expit(offset(cell) + h(D_j))

    where S takes second differences at the knots. The parameters are h at the first knot
    and the non-negative drops between knots, so monotonicity is a simple bound.
    """
    knots = np.asarray(knots, dtype=float)
    K = knots.size
    used = np.unique(cells.cell)
    remap = np.full(table.nodes.shape[0], -1)
    remap[used] = np.arange(used.size)
    nodes, W = table.nodes[used], table.weights[used]
    B = _interp_matrix(nodes, knots)  # (used*J, K)
    S = _penalty_matrix(knots)
    C = np.tril(np.ones((K, K)))[:, 1:]  # h = a - C @ drops
    ci = remap[cells.cell]
    n, y, off = cells.n, cells.y, cells.offset
    total = n.sum()

    def unpack(x: np.ndarray) -> np.ndarray:
        return x[0] - C @ x[1:]

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        h = unpack(x)
        H = (B @ h).reshape(W.shape)
        lin = off[:, None] + H[ci]
        q = expit(lin)
        P = np.clip((W[ci] * q).sum(axis=1), 1e-12, 1 - 1e-12)
        ll = (y * np.log(P) + (n - y) * np.log1p(-P)).sum()
        r = -(y / P - (n - y) / (1 - P))  # d(-ll)/dP
        gH_cells = r[:, None] * W[ci] * q * (1 - q)
        gH = np.zeros_like(W)
        np.add.at(gH, ci, gH_cells)
        gh = B.T @ gH.ravel()
        Sh = S @ h
        f = -ll / total + penalty * float(Sh @ Sh) / total
        gh = gh / total + 2 * penalty * (S.T @ Sh) / total
        grad = np.concatenate([[gh.sum()], -(C.T @ gh)])
        return f, grad

    p0 = np.clip(y.sum() / total, 0.05, 0.95)
    x0 = np.concatenate([[logit(p0) - np.mean(off) + 1.0], np.full(K - 1, 2.0 / K)])
    res = minimize(objective, x0, jac=True, method="L-BFGS-B",
                   bounds=[(None, None)] + [(0.0, None)] * (K - 1),
                   options={"maxiter": max_iter})  # fmt: skip
    h = unpack(res.x)
    info = {"converged": bool(res.success), "objective": float(res.fun),
            "n_attempts": float(total), "iterations": int(res.nit)}  # fmt: skip
    return LatentCurve(knots, h, table.model.name, float(penalty), info)


def implied_whole_yard(curve: LatentCurve, table: KernelTable, cells: np.ndarray,
                       offset: np.ndarray) -> np.ndarray:  # fmt: skip
    """P(convert | recorded cell) implied by the curve: the kernel average of q(D)."""
    q = curve.p(table.nodes[cells], np.asarray(offset, dtype=float)[:, None])
    return (table.weights[cells] * q).sum(axis=1)


def anchor_offsets(
    curve: LatentCurve, table: KernelTable, cells: np.ndarray, p_target: np.ndarray,
    n_iter: int = 30,
) -> np.ndarray:  # fmt: skip
    """For each row, the logit offset c with kernel average of expit(c + h(D)) equal to
    `p_target` (the engine's whole-yard conversion probability). Newton's method."""
    p_target = np.clip(np.asarray(p_target, dtype=float), 1e-6, 1 - 1e-6)
    H = curve.h(table.nodes[cells])
    W = table.weights[cells]
    c = logit(p_target) - (W * H).sum(axis=1)
    for _ in range(n_iter):
        q = expit(c[:, None] + H)
        f = (W * q).sum(axis=1) - p_target
        fp = (W * q * (1 - q)).sum(axis=1)
        step = f / np.maximum(fp, 1e-12)
        c = c - step
        if np.max(np.abs(step)) < 1e-10:
            break
    return c


# --------------------------------------------------------------------------------------
# 4. Propagation into the decision
# --------------------------------------------------------------------------------------


def go_value(p: np.ndarray, wp_if_convert: np.ndarray, wp_if_fail: np.ndarray) -> np.ndarray:
    """One-step value of going: p * wp_if_convert + (1 - p) * wp_if_fail."""
    p = np.asarray(p, dtype=float)
    return p * wp_if_convert + (1.0 - p) * wp_if_fail


def breakeven_probability(wp_if_convert: np.ndarray, wp_if_fail: np.ndarray,
                          best_kick: np.ndarray) -> np.ndarray:  # fmt: skip
    """The conversion probability at which going equals the best kick."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.asarray(best_kick) - wp_if_fail) / (wp_if_convert - wp_if_fail)


def propagate(
    curve: LatentCurve,
    table: KernelTable,
    d: np.ndarray,
    goal_to_go: np.ndarray,
    p_whole: np.ndarray,
    wp_if_convert: np.ndarray,
    wp_if_fail: np.ndarray,
    best_kick: np.ndarray,
    coverage: float = 0.9,
) -> dict[str, np.ndarray]:
    """Go value across the plausible true distances of each fourth down.

    Rows need a whole-yard conversion probability and both conditional values, and a
    kick to compare with (best_kick = max(fg_wp, punt_wp), NaN if none). Returns per row:
    the plausible range of D (central `coverage` interval of the kernel), the go-minus-
    kick gain at the long end (gain_lo) and short end (gain_hi) and integrated over the
    kernel (gain_mean, equal to the whole-yard gain by construction), `flips` (the sign
    of the gain differs between the two ends), `p_other_call` (probability under the
    kernel that the call at the true distance differs from the whole-yard call) and
    `evpi` (expected win probability gained by knowing D before choosing).
    """
    cells = table.cell(d, goal_to_go)
    c = anchor_offsets(curve, table, cells, p_whole)
    lo_q, hi_q = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    d_lo, d_hi = kernel_quantiles(table, cells, lo_q), kernel_quantiles(table, cells, hi_q)
    a, b, k = (np.asarray(x, dtype=float) for x in (wp_if_convert, wp_if_fail, best_kick))
    p_short = curve.p(d_lo, c)  # short end of the range: more likely to convert
    p_long = curve.p(d_hi, c)
    gain_hi = go_value(p_short, a, b) - k
    gain_lo = go_value(p_long, a, b) - k
    W = table.weights[cells]
    q = curve.p(table.nodes[cells], c[:, None])
    go_nodes = q * a[:, None] + (1 - q) * b[:, None]
    gain_nodes = go_nodes - k[:, None]
    gain_mean = (W * gain_nodes).sum(axis=1)
    whole_go = np.sign(go_value(p_whole, a, b) - k)
    other = np.sign(gain_nodes) != whole_go[:, None]
    p_other = (W * other).sum(axis=1)
    evpi = (W * np.maximum(gain_nodes, 0)).sum(axis=1) - np.maximum(gain_mean, 0)
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(k) & np.isfinite(p_whole)
    nan = np.where(valid, 1.0, np.nan)
    return {
        "d_lo": d_lo, "d_hi": d_hi, "p_short": p_short * nan, "p_long": p_long * nan,
        "gain_lo": gain_lo * nan, "gain_hi": gain_hi * nan, "gain_mean": gain_mean * nan,
        "flips": valid & (np.sign(gain_lo) != np.sign(gain_hi)),
        "p_other_call": p_other * nan, "evpi": evpi * nan, "valid": valid,
    }  # fmt: skip


def summarize_flips(
    prop: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Shares and win probability at stake over the rows in `mask` (valid rows only).

    share_flip_in_range: the call differs between the two ends of the plausible range.
    share_call_differs_at_true_distance: expected share whose call at the true distance
    differs from the whole-yard call. swing: go-minus-kick at the short end minus at the
    long end. evpi: expected value of knowing the true distance. All in win-probability
    points (x100). `weights` counts each row that many times (bootstrap)."""
    ok = prop["valid"] if mask is None else prop["valid"] & mask
    w = np.ones(ok.size) if weights is None else np.asarray(weights, dtype=float)
    w = np.where(ok, w, 0.0)
    total = w.sum()
    if total <= 0:
        return {"n": 0}

    def mean(x: np.ndarray) -> float:
        return float((w * np.nan_to_num(x)).sum() / total)

    swing = prop["gain_hi"] - prop["gain_lo"]
    flip_w = w * prop["flips"]
    return {
        "n": int(ok.sum()) if weights is None else float(total),
        "share_flip_in_range": mean(prop["flips"].astype(float)),
        "share_call_differs_at_true_distance": mean(prop["p_other_call"]),
        "mean_swing_wp_points": 100 * mean(swing),
        "median_swing_wp_points": float(100 * np.median(swing[ok])),
        "mean_evpi_wp_points": 100 * mean(prop["evpi"]),
        "mean_evpi_wp_points_flip_rows": float(
            100 * (flip_w * np.nan_to_num(prop["evpi"])).sum() / flip_w.sum())
        if flip_w.sum() > 0 else 0.0,
    }  # fmt: skip


# --------------------------------------------------------------------------------------
# Substituting a conversion probability into the real engine
# --------------------------------------------------------------------------------------


class ConversionOverride:
    """Components wrapper for `decision.DecisionEngine` that replaces P(convert).

    `p_for(yardline_100, ydstogo) -> float` gives the new probability. The outcome
    arrays are rescaled: first downs and touchdowns by p_new / p_old, turnovers and
    safeties by (1 - p_new) / (1 - p_old), so where the ball ends up given success or
    failure is unchanged. Every other attribute is the wrapped components'.
    """

    def __init__(self, base: Any, p_for: Any) -> None:
        self._base = base
        self._p_for = p_for

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def go_outcomes(self, yardline_100: int, ydstogo: int) -> dict[str, Any]:
        out = dict(self._base.go_outcomes(yardline_100, ydstogo))
        p_old = float(out["p_convert"])
        p_new = float(self._p_for(yardline_100, ydstogo))
        up = p_new / p_old if p_old > 0 else 0.0
        down = (1 - p_new) / (1 - p_old) if p_old < 1 else 0.0
        out["first_down"] = np.asarray(out["first_down"], dtype=float) * up
        out["p_touchdown"] = float(out["p_touchdown"]) * up
        out["turnover"] = np.asarray(out["turnover"], dtype=float) * down
        out["p_safety"] = float(out.get("p_safety", 0.0)) * down
        out["p_convert"] = p_new
        return out


def assert_development_only(plays: pl.DataFrame, what: str) -> None:
    """Every fit in this module is on development seasons only."""
    splits.assert_no_test_seasons(plays, what)
