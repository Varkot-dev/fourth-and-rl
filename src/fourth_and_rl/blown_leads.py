"""M3b: blown leads as a calibration test for win-probability models.

A calibrated win probability is a martingale that ends at 0 or 1. For such a path
with continuous movement, the largest value reached by the team that goes on to
lose has a known distribution given the starting value p0:

    P(loser's peak >= x | lost) = p0 * (1 - x) / (x * (1 - p0))    for p0 <= x < 1

This is Theorem 2 of Pipping-Gamon and Wyner, "The Blown Lead Paradox"
(arXiv 2601.18774). This module holds:

  1. the reference distribution (survival, CDF, quantile, PIT) and the two-team
     version used in the paper,
  2. extraction of each game's losing-team path from play-by-play,
  3. the calibration tests (PIT uniformity, extreme-peak counts, first crossings),
  4. a discrete-path simulator that measures how far real, jumpy paths sit from the
     continuous formula, and a simulated reference that replaces it,
  5. the two collapse rankings.

Everything here is pure computation. Scripts under scripts/ do the reading, writing
and plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats
from scipy.special import ndtr, ndtri

ArrayLike = np.ndarray | float

# Columns the path extraction needs besides the win-probability column itself.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id", "play_id", "order_sequence", "season", "week", "home_team", "away_team",
    "posteam", "play_type", "result", "home_score", "away_score", "qtr",
    "game_seconds_remaining", "posteam_score", "defteam_score", "desc",
)  # fmt: skip

# ---------------------------------------------------------------------------
# 1. Theory: the continuous-path reference
# ---------------------------------------------------------------------------


def reach_prob(x: ArrayLike, p0: ArrayLike) -> np.ndarray:
    """P(a continuous martingale started at p0 ever reaches x) = p0 / x for x >= p0."""
    x, p0 = np.broadcast_arrays(np.asarray(x, float), np.asarray(p0, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(x <= p0, 1.0, p0 / x)
    return np.clip(out, 0.0, 1.0)


def peak_sf(x: ArrayLike, p0: ArrayLike) -> np.ndarray:
    """P(peak >= x | the team lost), for a team that started at p0.

    Equals 1 for x <= p0 (the peak is at least the starting value), the formula in
    the module docstring for p0 <= x < 1, and 0 at x >= 1 (a loser never reaches 1).
    """
    x, p0 = np.broadcast_arrays(np.asarray(x, float), np.asarray(p0, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        tail = p0 * (1.0 - x) / (x * (1.0 - p0))
    out = np.where(x <= p0, 1.0, np.where(x >= 1.0, 0.0, tail))
    return np.clip(out, 0.0, 1.0)


def peak_cdf(x: ArrayLike, p0: ArrayLike) -> np.ndarray:
    """P(peak <= x | the team lost). The distribution has no atoms below 1."""
    return 1.0 - peak_sf(x, p0)


def peak_quantile(u: ArrayLike, p0: ArrayLike) -> np.ndarray:
    """Inverse of `peak_cdf` in x: the peak that a fraction u of losers stay below.

    Solving 1 - p0 (1 - x) / (x (1 - p0)) = u gives x = p0 / (p0 + (1 - u)(1 - p0)).
    """
    u, p0 = np.broadcast_arrays(np.asarray(u, float), np.asarray(p0, float))
    return p0 / (p0 + (1.0 - u) * (1.0 - p0))


def peak_pit(peak: ArrayLike, p0: ArrayLike) -> np.ndarray:
    """Probability integral transform u = F(peak | p0).

    For a calibrated model with continuous paths, u is Uniform(0, 1) whatever p0 is,
    which is what lets games with different starting values be pooled.
    """
    return peak_cdf(peak, p0)


def game_peak_cdf(x: ArrayLike, p0: ArrayLike) -> np.ndarray:
    """The paper's two-team reference: CDF of the eventual loser's peak for a game.

    Here the loser is not conditioned on. p0 may be either team's starting value;
    only the favorite's value f = max(p0, 1 - p0) matters:
        0              for x < 1 - f
        1 - (1 - f)/x  for 1 - f <= x < f
        2 - 1/x        for f <= x < 1
        1              at x = 1
    """
    x, p0 = np.broadcast_arrays(np.asarray(x, float), np.asarray(p0, float))
    fav = np.maximum(p0, 1.0 - p0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mid = 1.0 - (1.0 - fav) / x
        top = 2.0 - 1.0 / x
    out = np.where(x < 1.0 - fav, 0.0, np.where(x < fav, mid, np.where(x < 1.0, top, 1.0)))
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 2. Path extraction from play-by-play
# ---------------------------------------------------------------------------


def home_paths(plays: pl.DataFrame, wp_col: str) -> pl.DataFrame:
    """One row per usable play with the win probability from the home team's side.

    `wp_col` holds the probability for the team with the ball (the posteam) before
    the play. The home team's value is that number when the home team has the ball
    and 1 minus it otherwise.

    Rows are dropped when:
      - the game ended tied (no loser),
      - `play_type` is null: these are timeouts, quarter ends and similar rows, and
        their posteam is not reliable (it sometimes names the wrong team),
      - the win probability is null.
    Rows are ordered by `order_sequence` within a game. `step` numbers them from 0.
    """
    missing = [c for c in (*REQUIRED_COLUMNS, wp_col) if c not in plays.columns]
    if missing:
        raise KeyError(f"plays is missing columns: {missing}")
    df = (
        plays.filter(
            (pl.col("result") != 0)
            & pl.col("play_type").is_not_null()
            & pl.col(wp_col).is_not_null()
            & pl.col(wp_col).is_not_nan()
        )
        .sort("game_id", "order_sequence", "play_id")
        .with_columns(
            (pl.col("posteam") == pl.col("home_team")).alias("_home_ball"),
            pl.col(wp_col).cast(pl.Float64).clip(0.0, 1.0).alias("_wp"),
        )
        .with_columns(
            pl.when("_home_ball").then(pl.col("_wp")).otherwise(1.0 - pl.col("_wp")).alias(
                "home_wp_path"
            ),
            pl.when("_home_ball")
            .then(pl.col("posteam_score"))
            .otherwise(pl.col("defteam_score"))
            .alias("home_pts"),
            pl.when("_home_ball")
            .then(pl.col("defteam_score"))
            .otherwise(pl.col("posteam_score"))
            .alias("away_pts"),
            (pl.col("result") > 0).alias("home_won"),
        )
        .with_columns(pl.int_range(pl.len()).over("game_id").alias("step"))
    )
    return df.select(
        "game_id", "season", "week", "home_team", "away_team", "home_score", "away_score",
        "home_won", "step", "play_id", "qtr", "game_seconds_remaining", "home_pts",
        "away_pts", "desc", "home_wp_path",
    )  # fmt: skip


def extract_loser_paths(
    plays: pl.DataFrame, wp_col: str, min_plays: int = 50
) -> pl.DataFrame:
    """One row per non-tied game describing the losing team's win-probability path.

    Columns: game_id, season, week, home_team, away_team, winner, loser, loser_score,
    winner_score, p0 (the loser's first value), peak (the loser's maximum), the play at
    which the peak was first reached (peak_step, peak_qtr, peak_seconds_remaining,
    peak_loser_pts, peak_winner_pts, peak_desc), n_plays, max_jump (largest absolute
    change between consecutive plays), last_wp (the loser's value before the final play).

    The observed values are all before a play, so none of them is the final 0 or 1.
    The path is treated as ending at 0 for the loser after the last play; that final
    value can never be the maximum, so it does not enter `peak`, and the final drop is
    not counted in `max_jump`. Games with fewer than `min_plays` usable values are
    dropped (the path is too incomplete to trust its maximum).
    """
    hp = home_paths(plays, wp_col).with_columns(
        pl.when("home_won")
        .then(1.0 - pl.col("home_wp_path"))
        .otherwise(pl.col("home_wp_path"))
        .alias("loser_wp"),
        pl.when("home_won").then(pl.col("away_pts")).otherwise(pl.col("home_pts")).alias(
            "loser_pts"
        ),
        pl.when("home_won").then(pl.col("home_pts")).otherwise(pl.col("away_pts")).alias(
            "winner_pts"
        ),
    )
    at_peak = pl.col("loser_wp").arg_max()
    out = (
        hp.group_by("game_id", maintain_order=True)
        .agg(
            pl.col("season").first(),
            pl.col("week").first(),
            pl.col("home_team").first(),
            pl.col("away_team").first(),
            pl.col("home_won").first(),
            pl.col("home_score").first(),
            pl.col("away_score").first(),
            pl.col("loser_wp").first().alias("p0"),
            pl.col("loser_wp").max().alias("peak"),
            at_peak.alias("peak_step"),
            pl.col("qtr").get(at_peak).alias("peak_qtr"),
            pl.col("game_seconds_remaining").get(at_peak).alias("peak_seconds_remaining"),
            pl.col("loser_pts").get(at_peak).alias("peak_loser_pts"),
            pl.col("winner_pts").get(at_peak).alias("peak_winner_pts"),
            pl.col("desc").get(at_peak).alias("peak_desc"),
            pl.len().alias("n_plays"),
            pl.col("loser_wp").diff().abs().max().fill_null(0.0).alias("max_jump"),
            pl.col("loser_wp").last().alias("last_wp"),
        )
        .filter(pl.col("n_plays") >= min_plays)
        .with_columns(
            pl.when("home_won").then("home_team").otherwise("away_team").alias("winner"),
            pl.when("home_won").then("away_team").otherwise("home_team").alias("loser"),
            pl.when("home_won").then("away_score").otherwise("home_score").alias("loser_score"),
            pl.when("home_won").then("home_score").otherwise("away_score").alias(
                "winner_score"
            ),
        )
    )
    return out.select(
        "game_id", "season", "week", "home_team", "away_team", "winner", "loser",
        "loser_score", "winner_score", "p0", "peak", "peak_step", "peak_qtr",
        "peak_seconds_remaining", "peak_loser_pts", "peak_winner_pts", "peak_desc",
        "n_plays", "max_jump", "last_wp",
    )  # fmt: skip


def jump_sizes(hp: pl.DataFrame) -> np.ndarray:
    """Absolute play-to-play changes in win probability, pooled over games.

    `hp` is the output of `home_paths`. The final drop to 0 or 1 is not included.
    """
    d = hp.select(pl.col("home_wp_path").diff().over("game_id").abs().alias("d")).drop_nulls()
    return d["d"].to_numpy()


def jump_summary(jumps: np.ndarray) -> dict:
    """Quantiles and tail shares of absolute jump sizes."""
    q = np.quantile(jumps, [0.5, 0.9, 0.99, 0.999])
    return {
        "n_jumps": int(jumps.size),
        "mean": float(jumps.mean()),
        "median": float(q[0]),
        "q90": float(q[1]),
        "q99": float(q[2]),
        "q999": float(q[3]),
        "share_above_0.05": float((jumps > 0.05).mean()),
        "share_above_0.10": float((jumps > 0.10).mean()),
        "share_above_0.20": float((jumps > 0.20).mean()),
        "max": float(jumps.max()),
    }


def first_crossings(hp: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """First time each team's path is at or above `threshold`, one row per team side.

    Returns game_id, season, home_team, away_team, side ("home" or "away"), p_cross
    (the value at the first crossing, at or above the threshold) and won (whether
    that team won). Only crossings before the outcome is known are possible here,
    because every observed value is from before a play.
    """
    sides = []
    for side in ("home", "away"):
        value = pl.col("home_wp_path") if side == "home" else 1.0 - pl.col("home_wp_path")
        won = pl.col("home_won") if side == "home" else ~pl.col("home_won")
        sides.append(
            hp.with_columns(value.alias("p_side"), won.alias("won"))
            .filter(pl.col("p_side") >= threshold)
            .group_by("game_id", maintain_order=True)
            .agg(
                pl.col("season").first(),
                pl.col("home_team").first(),
                pl.col("away_team").first(),
                pl.col("p_side").first().alias("p_cross"),
                pl.col("won").first(),
            )
            .with_columns(pl.lit(side).alias("side"))
        )
    return pl.concat(sides)


# ---------------------------------------------------------------------------
# 3. Calibration tests
# ---------------------------------------------------------------------------


def ks_uniform(u: np.ndarray) -> dict:
    """Kolmogorov-Smirnov tests of PIT values against Uniform(0, 1).

    Reports the two-sided statistic and the one-sided statistic that looks for too
    much mass near 1 (too many surprising peaks), d_upper = sup over t of
    (t - empirical CDF just below t). The p-values assume independent games.
    """
    u = np.sort(np.asarray(u, float))
    n = u.size
    two = stats.kstest(u, "uniform")
    d_upper = float(np.max(u - np.arange(n) / n))  # t - F(t-) at the observed points
    d_lower = float(np.max(np.arange(1, n + 1) / n - u))
    return {
        "n": int(n),
        "ks_two_sided": float(two.statistic),
        "ks_two_sided_p": float(two.pvalue),
        "d_upper": d_upper,
        "d_upper_p_iid": float(stats.ksone.sf(d_upper, n)),
        "d_lower": d_lower,
        "d_lower_p_iid": float(stats.ksone.sf(d_lower, n)),
    }


def pigeonhole_weights(
    season: np.ndarray, home: np.ndarray, away: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Game weights for a season-stratified team (pigeonhole) bootstrap.

    Within a season, teams are resampled with replacement; a game's weight is the
    product of its two teams' counts, rescaled so each season keeps its total weight.
    Games that share a team therefore move together, which allows for dependence
    between a team's games. This is the resampling scheme used in the paper
    (Owen 2007). Returns an array of shape (n_boot, n_games).
    """
    n = season.size
    w = np.zeros((n_boot, n), dtype=np.float64)
    for s in np.unique(season):
        idx = np.flatnonzero(season == s)
        teams, inv = np.unique(np.concatenate([home[idx], away[idx]]), return_inverse=True)
        h, a = inv[: idx.size], inv[idx.size :]
        counts = rng.multinomial(teams.size, np.full(teams.size, 1.0 / teams.size), size=n_boot)
        ws = (counts[:, h] * counts[:, a]).astype(np.float64)
        tot = ws.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        w[:, idx] = ws * idx.size / tot
    return w


def pit_bootstrap_test(
    u: np.ndarray, weights: np.ndarray, tail_levels: tuple[float, ...] = (0.90, 0.95, 0.99)
) -> dict:
    """The paper's league-level test, with bootstrap game weights from `pigeonhole_weights`.

    Statistic: d_upper = sup(t - F(t-)) of the PIT values. Null distribution: the
    recentered bootstrap statistic sup(F - F*), where F* is the reweighted empirical
    CDF. Also returns, for each tail level q, the share of games with u >= q, its
    excess over the nominal 1 - q, and a 95% percentile interval for that excess.
    """
    u = np.asarray(u, float)
    n = u.size
    order = np.argsort(u, kind="stable")
    us = u[order]
    d_upper = float(np.max(us - np.arange(n) / n))
    f_obs = np.arange(1, n + 1) / n
    w = weights[:, order]
    f_star = np.cumsum(w, axis=1) / w.sum(axis=1, keepdims=True)
    d_star = np.max(f_obs[None, :] - f_star, axis=1)
    out = {
        "d_upper": d_upper,
        "d_upper_p_bootstrap": float((1 + np.sum(d_star >= d_upper)) / (weights.shape[0] + 1)),
        "bootstrap_critical_value_95": float(np.quantile(d_star, 0.95)),
        "n_boot": int(weights.shape[0]),
        "tails": {},
    }
    wsum = weights.sum(axis=1)
    for q in tail_levels:
        ind = (u >= q).astype(float)
        rate = float(ind.mean())
        boot = (weights @ ind) / wsum
        lo, hi = np.quantile(boot, [0.025, 0.975])
        out["tails"][f"{q:.2f}"] = {
            "observed_share": rate,
            "nominal_share": float(1 - q),
            "excess": rate - (1 - q),
            "excess_ci95": [float(lo - (1 - q)), float(hi - (1 - q))],
        }
    return out


def poisson_binomial_interval(
    probs: np.ndarray, rng: np.random.Generator, n_sim: int = 20000, level: float = 0.95
) -> tuple[float, float]:
    """Central interval for a sum of independent Bernoulli(probs), by simulation."""
    probs = np.asarray(probs, float)
    sims = np.zeros(n_sim, dtype=np.int64)
    chunk = max(1, int(2e7 // max(probs.size, 1)))
    for start in range(0, n_sim, chunk):
        m = min(chunk, n_sim - start)
        sims[start : start + m] = (rng.random((m, probs.size)) < probs).sum(axis=1)
    lo, hi = np.quantile(sims, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def extreme_peak_counts(
    hit: np.ndarray,
    tail_prob: np.ndarray,
    rng: np.random.Generator,
    n_sim: int = 20000,
) -> dict:
    """Observed versus expected number of games whose loser peaked at or above a level.

    `hit[i]` says whether game i's loser did; `tail_prob[i]` is the reference
    probability that game i's loser peaks at or above the level. Expected = their sum;
    the interval is the central 95% range of the
    count if games were independent draws from the reference; `p_upper` is the chance
    of a count at least as large as the observed one (normal approximation with the
    exact mean and variance).
    """
    tail_prob = np.clip(np.asarray(tail_prob, float), 0.0, 1.0)
    observed = int(np.sum(hit))
    expected = float(tail_prob.sum())
    sd = float(np.sqrt(np.sum(tail_prob * (1 - tail_prob))))
    lo, hi = poisson_binomial_interval(tail_prob, rng, n_sim)
    z = (observed - expected) / sd if sd > 0 else float("nan")
    return {
        "observed": observed,
        "expected": expected,
        "expected_interval95": [lo, hi],
        "ratio_observed_to_expected": observed / expected if expected > 0 else float("nan"),
        "z": float(z),
        "p_upper": float(stats.norm.sf((observed - 0.5 - expected) / sd)) if sd > 0 else None,
    }


def crossing_test(
    crossings: pl.DataFrame, rng: np.random.Generator, n_sim: int = 20000
) -> dict:
    """Blown leads after a first crossing: observed losses versus implied losses.

    At the first moment a team's win probability is at or above a threshold, a
    calibrated model says it loses with probability 1 - p_cross. Summing that over
    crossings gives the expected number of losers whose peak reached the threshold.
    This holds exactly for paths that jump (optional stopping needs no continuity),
    so it is the discreteness-proof version of the peak count.
    """
    p = crossings["p_cross"].to_numpy()
    lost = ~crossings["won"].to_numpy()
    implied = 1.0 - p
    out = extreme_peak_counts(lost, implied, rng, n_sim)
    out.update(
        {
            "episodes": int(p.size),
            "games": int(crossings["game_id"].n_unique()),
            "observed_loss_rate": float(lost.mean()) if p.size else float("nan"),
            "implied_loss_rate": float(implied.mean()) if p.size else float("nan"),
            "mean_p_at_crossing": float(p.mean()) if p.size else float("nan"),
        }
    )
    return out


# ---------------------------------------------------------------------------
# 4. Discrete paths: simulator and simulated reference
# ---------------------------------------------------------------------------
#
# The simulated game. The home lead is a Brownian motion W on game time [0, 1] with
# unit variance, plus a starting handicap x0 = Phi^-1(p0). The team wins if
# x0 + W(1) > 0. Its exact win probability at time t is
#
#     p(t) = Phi( (x0 + W(t)) / sqrt(1 - t) )
#
# (the remaining movement is Normal(0, 1 - t)). This is a continuous martingale, so
# the closed-form reference holds for it exactly. A "feed" reports it only at
# n_steps play times 0 = t_0 < t_1 < ... < t_{n-1} < 1. Reported values are still an
# exact martingale (a martingale looked at on a fixed grid stays one), but the
# reported maximum can miss a peak between plays. Play times are either equally
# spaced, or drawn with Gamma(gap_shape) gaps so that a few plays carry much more
# game time than the rest, which produces the occasional large jump seen in real
# paths. Because Phi is increasing, the peak of p equals Phi of the peak of
# z = (x0 + W) / sqrt(1 - t), so only z is tracked.


def _walk(
    n_steps: int, n_paths: int, rng: np.random.Generator, gap_shape: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the Brownian lead at the reported play times.

    There are n_steps reported values per game: the starting value at time 0 and one
    before each later play, at times t_1 < ... < t_{n_steps - 1} < 1. Returns
      w      W(t_1..t_{n_steps - 1}), shape (n_paths, n_steps - 1)
      scale  1 / sqrt(1 - t_k) for the same times (one row if times are shared)
      w_end  W(1), shape (n_paths,)
    """
    if gap_shape is None:
        gaps = np.full((1, n_steps), 1.0 / n_steps)
    else:
        gaps = rng.standard_gamma(gap_shape, size=(n_paths, n_steps))
        gaps /= gaps.sum(axis=1, keepdims=True)
    # time left after each play, summed from the end so it is never negative
    left = np.cumsum(gaps[:, ::-1], axis=1)[:, ::-1][:, 1:]
    scale = 1.0 / np.sqrt(np.maximum(left, 1e-12))
    steps = rng.standard_normal((n_paths, n_steps)) * np.sqrt(gaps)
    w = np.cumsum(steps, axis=1)
    return w[:, :-1], scale, w[:, -1].copy()


def simulate_paths(
    p0: ArrayLike,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    gap_shape: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate reported win-probability paths for a team that starts at p0.

    `p0` is one number or an array of length n_paths. Returns (paths, won): `paths`
    has shape (n_paths, n_steps), the values reported before each play (the first is
    p0); `won` says whether the team won. Meant for small runs (tests, figures,
    jump sizes); `simulate_loser_peaks` and `build_discrete_reference` are the fast
    path for peaks.
    """
    x0 = np.broadcast_to(ndtri(np.asarray(p0, float)), (n_paths,))[:, None]
    w, scale, w_end = _walk(n_steps, n_paths, rng, gap_shape)
    paths = np.concatenate([ndtr(x0), ndtr((x0 + w) * scale)], axis=1)
    return paths, (x0[:, 0] + w_end) > 0


def simulate_loser_peaks(
    p0: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    gap_shape: float | None = None,
    batch: int | None = None,
) -> np.ndarray:
    """Reported peaks of a team that starts at p0, for the simulated games it loses."""
    x0 = float(ndtri(p0))
    batch = batch or int(min(20000, max(100, 4_000_000 // n_steps)))
    peaks = []
    for start in range(0, n_paths, batch):
        m = min(batch, n_paths - start)
        w, scale, w_end = _walk(n_steps, m, rng, gap_shape)
        zmax = np.maximum(x0, np.max((x0 + w) * scale, axis=1))
        peaks.append(ndtr(zmax[x0 + w_end <= 0]))
    return np.concatenate(peaks)


@dataclass
class DiscreteReference:
    """Simulated distribution of the loser's reported peak on a grid of starting values.

    For each starting value the simulated peaks are stored on the scale of the
    continuous PIT, u = peak_cdf(peak, p0). On that scale every starting value shares
    the range [0, 1] and the distribution changes slowly with p0, so interpolating
    between grid values is safe. `scores[j]` is the sorted array for `p0_grid[j]`.
    """

    p0_grid: np.ndarray
    scores: list[np.ndarray]
    n_steps: int
    gap_shape: float | None

    def _interp(self, x: ArrayLike, p0: ArrayLike, side: str) -> np.ndarray:
        """P(score < u) for side="left" or P(score <= u) for side="right"."""
        x, p0 = np.broadcast_arrays(np.asarray(x, float), np.asarray(p0, float))
        x, p0 = x.ravel(), p0.ravel()
        u = peak_cdf(x, p0)
        pc = np.clip(p0, self.p0_grid[0], self.p0_grid[-1])
        hi = np.clip(np.searchsorted(self.p0_grid, pc), 1, self.p0_grid.size - 1)
        lo = hi - 1
        wt = (pc - self.p0_grid[lo]) / (self.p0_grid[hi] - self.p0_grid[lo])
        out = np.zeros(x.size)
        for nb, weight in ((lo, 1.0 - wt), (hi, wt)):
            for j in np.unique(nb):
                m = nb == j
                frac = np.searchsorted(self.scores[j], u[m], side=side) / self.scores[j].size
                out[m] += weight[m] * frac
        return out

    def cdf(self, x: ArrayLike, p0: ArrayLike) -> np.ndarray:
        """P(reported peak <= x | lost), interpolated linearly between grid values of p0."""
        return self._interp(x, p0, "right")

    def sf(self, x: ArrayLike, p0: ArrayLike) -> np.ndarray:
        """P(reported peak >= x | lost)."""
        return 1.0 - self._interp(x, p0, "left")

    def pit(self, peak: ArrayLike, p0: ArrayLike, rng: np.random.Generator) -> np.ndarray:
        """Randomized PIT: uniform between F(x-) and F(x).

        The simulated distribution has an atom at the starting value (some losers
        never rise above where they began), so the plain PIT is not uniform there.
        Spreading the atom uniformly restores exact uniformity under the reference.
        """
        lo = self._interp(peak, p0, "left")
        hi = self._interp(peak, p0, "right")
        return lo + rng.random(lo.size) * (hi - lo)


def build_discrete_reference(
    p0_grid: np.ndarray,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    gap_shape: float | None = None,
    batch: int = 20000,
) -> DiscreteReference:
    """Simulate the loser's reported peak for every starting value in `p0_grid`.

    The same simulated walks are reused for every starting value (only the handicap
    x0 changes), which is cheap and makes the reference vary smoothly in p0.
    """
    p0_grid = np.asarray(p0_grid, float)
    x0s = ndtri(p0_grid)
    acc: list[list[np.ndarray]] = [[] for _ in x0s]
    for start in range(0, n_paths, batch):
        m = min(batch, n_paths - start)
        w, scale, w_end = _walk(n_steps, m, rng, gap_shape)
        ws = w * scale
        for j, x0 in enumerate(x0s):
            lost = x0 + w_end <= 0
            sc = scale if scale.shape[0] == 1 else scale[lost]
            zmax = np.maximum(x0, np.max(x0 * sc + ws[lost], axis=1))
            acc[j].append(peak_cdf(ndtr(zmax), p0_grid[j]))
    scores = [np.sort(np.concatenate(a)) for a in acc]
    return DiscreteReference(p0_grid, scores, n_steps, gap_shape)


def simulated_jump_sizes(
    p0: ArrayLike,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    gap_shape: float | None = None,
) -> np.ndarray:
    """Absolute play-to-play changes of simulated reported paths (final drop excluded)."""
    paths, _ = simulate_paths(p0, n_steps, n_paths, rng, gap_shape)
    return np.abs(np.diff(paths, axis=1)).ravel()


# ---------------------------------------------------------------------------
# 5. Collapse rankings
# ---------------------------------------------------------------------------


def add_reference_columns(games: pl.DataFrame) -> pl.DataFrame:
    """Add the continuous-reference columns to the output of `extract_loser_paths`.

    pit            F(peak | p0), conditioning on which team lost
    tail_prob      P(a loser starting at p0 peaks at least this high) = 1 - pit
    pit_game       the paper's two-team score, not conditioning on who lost
    surprise_bits  -log2(tail_prob), a readable scale for how unusual the peak is
    """
    p0 = games["p0"].to_numpy()
    peak = games["peak"].to_numpy()
    tail = peak_sf(peak, p0)
    with np.errstate(divide="ignore"):
        bits = -np.log2(tail)
    return games.with_columns(
        pl.Series("pit", 1.0 - tail),
        pl.Series("tail_prob", tail),
        pl.Series("pit_game", game_peak_cdf(peak, p0)),
        pl.Series("surprise_bits", bits),
    )


def rank_collapses(games: pl.DataFrame, top: int = 25, peak_decimals: int = 3) -> dict:
    """Top collapses by raw peak and by surprise.

    Raw-peak ranking: many games sit above 0.99 and differ only in digits no model
    can support, so peaks are compared after rounding to `peak_decimals`; ties go to
    the peak that came later in the game (less time left), then to game_id.
    Surprise ranking: smallest reference tail probability first, same tie-breaks.
    `games` must already have the columns from `add_reference_columns`.
    """
    g = games.with_columns(pl.col("peak").round(peak_decimals).alias("peak_rounded"))
    by_peak = g.sort(
        ["peak_rounded", "peak_seconds_remaining", "game_id"],
        descending=[True, False, False],
        nulls_last=True,
    ).with_row_index("rank_by_peak", offset=1)
    by_surprise = g.sort(
        ["tail_prob", "peak_seconds_remaining", "game_id"], nulls_last=True
    ).with_row_index("rank_by_surprise", offset=1)
    both = by_peak.join(
        by_surprise.select("game_id", "rank_by_surprise"), on="game_id", how="left"
    )
    top_peak = both.sort("rank_by_peak").head(top)
    top_surprise = both.sort("rank_by_surprise").head(top)
    a, b = set(top_peak["game_id"]), set(top_surprise["game_id"])
    rho = stats.spearmanr(both["rank_by_peak"].to_numpy(), both["rank_by_surprise"].to_numpy())
    return {
        "by_peak": top_peak,
        "by_surprise": top_surprise,
        "overlap": len(a & b),
        "only_in_peak_list": sorted(a - b),
        "only_in_surprise_list": sorted(b - a),
        "spearman_all_games": float(rho.statistic),
        "ties_at_top_rounded_peak": int(
            (g["peak_rounded"] == g["peak_rounded"].max()).sum()
        ),
        "games_with_peak_at_least_0.99": int((g["peak"] >= 0.99).sum()),
    }
