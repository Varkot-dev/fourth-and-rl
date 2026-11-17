"""Power of the blown-lead calibration tests against known overconfidence (track A1).

M3b found that real win-probability feeds pass, or nearly pass, the blown-lead tests.
That only means something if the tests could have failed. This module measures how
often each test catches a feed whose miscalibration is known.

Simulated league. Each game's true home win probability is the M3b simulator path
(a Brownian lead reported at n_steps play times with Gamma gaps), which is an exact
martingale. A miscalibrated feed reports

    q = expit(k * logit(p))

where p is the true value. k = 1 is calibrated, k > 1 is overconfident (pushed toward
0 and 1), k < 1 is underconfident. Both teams' feeds are distorted the same way, so
the away feed is 1 - q.

Two tests, both at the 5% level:

  1. Continuous-formula KS: the one-sided statistic d_upper on the loser's peak PIT
     under the continuous reference (too many extreme peaks means overconfidence).
     p-value from the iid KS distribution, as in M3b.
  2. First-crossing test: at the first value at or above each threshold c, a
     calibrated feed implies a loss probability 1 - q. For each c,
     z_c = (observed losses - implied losses) / sd. The statistic is max over c of
     |z_c|, and its 5% critical value is simulated under k = 1, so the test has the
     right size under the simulator whatever the dependence between thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import expit, logit, ndtr

from fourth_and_rl.blown_leads import _walk, peak_pit


@dataclass
class League:
    """True home win-probability paths for one simulated league."""

    p: np.ndarray  # shape (n_games, n_steps), the true value before each play
    home_won: np.ndarray  # shape (n_games,)


def simulate_league(
    n_games: int,
    n_steps: int,
    rng: np.random.Generator,
    gap_shape: float | None,
    p0_sd: float,
) -> League:
    """Simulate a league of calibrated paths with pregame value Phi(Z), Z ~ N(0, p0_sd)."""
    x0 = rng.normal(0.0, p0_sd, size=(n_games, 1))
    w, scale, w_end = _walk(n_steps, n_games, rng, gap_shape)
    p = np.concatenate([ndtr(x0), ndtr((x0 + w) * scale)], axis=1)
    return League(p=p, home_won=(x0[:, 0] + w_end) > 0)


def distort(p: np.ndarray, k: float) -> np.ndarray:
    """Temperature-scaled feed: expit(k * logit(p)), kept strictly inside (0, 1)."""
    eps = 1e-12
    return expit(k * logit(np.clip(p, eps, 1.0 - eps)))


def crossing_z(q_home: np.ndarray, home_won: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """z statistic of the first-crossing test at each threshold, both sides pooled."""
    out = np.empty(len(thresholds))
    sides = ((q_home, home_won), (1.0 - q_home, ~home_won))
    for i, c in enumerate(thresholds):
        lost_all, implied_all = [], []
        for path, won in sides:
            mask = path >= c
            hit = mask.any(axis=1)
            first = np.argmax(mask, axis=1)
            p_cross = path[np.arange(path.shape[0]), first][hit]
            lost_all.append(~won[hit])
            implied_all.append(1.0 - p_cross)
        lost = np.concatenate(lost_all)
        implied = np.concatenate(implied_all)
        sd = np.sqrt(np.sum(implied * (1.0 - implied)))
        out[i] = (lost.sum() - implied.sum()) / sd if sd > 0 else 0.0
    return out


def continuous_ks_p(q_home: np.ndarray, home_won: np.ndarray) -> float:
    """p-value of the one-sided KS test (too many extreme loser peaks), continuous reference."""
    loser = np.where(home_won[:, None], 1.0 - q_home, q_home)
    peak = loser.max(axis=1)
    u = np.sort(peak_pit(peak, loser[:, 0]))
    n = u.size
    d_upper = float(np.max(u - np.arange(n) / n))
    return float(stats.ksone.sf(d_upper, n))


def run_power(
    n_games: int,
    ks: list[float],
    thresholds: list[float],
    n_null: int,
    n_rep: int,
    n_steps: int,
    gap_shape: float | None,
    p0_sd: float,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict:
    """Rejection rates of both tests at each k for one league size.

    The critical value of the first-crossing statistic comes from n_null leagues at
    k = 1. Power is then measured on n_rep fresh leagues, with every k applied to the
    same leagues (common random numbers, so differences between k are not noise).
    """
    null_stats = np.empty(n_null)
    for r in range(n_null):
        lg = simulate_league(n_games, n_steps, rng, gap_shape, p0_sd)
        null_stats[r] = np.max(np.abs(crossing_z(lg.p, lg.home_won, thresholds)))
    crit = float(np.quantile(null_stats, 1.0 - alpha))

    rej_cross = np.zeros(len(ks))
    rej_ks = np.zeros(len(ks))
    mean_z = np.zeros((len(ks), len(thresholds)))
    for _ in range(n_rep):
        lg = simulate_league(n_games, n_steps, rng, gap_shape, p0_sd)
        for j, k in enumerate(ks):
            q = distort(lg.p, k)
            z = crossing_z(q, lg.home_won, thresholds)
            mean_z[j] += z / n_rep
            rej_cross[j] += np.max(np.abs(z)) > crit
            rej_ks[j] += continuous_ks_p(q, lg.home_won) < alpha
    rej_cross /= n_rep
    rej_ks /= n_rep
    mc_se = np.sqrt(rej_cross * (1 - rej_cross) / n_rep)
    return {
        "n_games": n_games,
        "critical_value": crit,
        "ks": list(ks),
        "power_first_crossing": rej_cross.tolist(),
        "power_first_crossing_mc_se": mc_se.tolist(),
        "power_continuous_ks": rej_ks.tolist(),
        "mean_z_by_threshold": {
            str(c): mean_z[:, i].tolist() for i, c in enumerate(thresholds)
        },
    }
