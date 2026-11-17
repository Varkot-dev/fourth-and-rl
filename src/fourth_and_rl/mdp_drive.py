"""M4 stage 1: the single-drive problem, valued in expected points.

State: a field state (down, distance bin, yard line). No clock and no score, so the
problem never ends: possessions follow one another forever. The value V(s) is the
expected net points (ours minus theirs) from here on, for two identical teams.

Bellman equation, written for the team with the ball:

    V(s) = max over actions a of  sum over outcomes o of  P(o | s, a) * W(o)

    W(keep the ball in state s')            = V(s')
    W(other team takes over at their y)     = -V(first and 10 at y)
    W(touchdown)                            = td_value - K
    W(field goal made)                      = 3 - K
    W(safety against us)                    = -2 - F
    W(they score a touchdown on the play)   = -td_value + K

K is the value, to the receiving team, of receiving a kickoff (the average of V over the
kickoff distribution), F the same for the free kick after a safety. With
`after_score="absorb"` K and F are 0: the value is then "expected points of the next
score", which is what nflfastR's `ep` column estimates, and is used only to compare.

There is no terminal state to start a backward pass from, so the equation is solved by
value iteration: start from V = 0 and apply the right-hand side until V stops changing.

One detail. A change of possession flips the sign of the value, so the update can
overshoot and oscillate (in the one-state game of the unit test it jumps between 0 and
3.5 forever). The fix is damping: move only part of the way, V <- (1 - w) V + w T(V).
The fixed point is the same, and the oscillating component now dies out.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fourth_and_rl.mdp import FieldKernels

DRIVE_ACTIONS = ("go", "field_goal", "punt")


def solve_drive(
    kern: FieldKernels,
    td_value: float,
    fg_points: float = 3.0,
    safety_points: float = 2.0,
    after_score: str = "kickoff",
    fourth_policy: np.ndarray | None = None,
    mode: str = "neutral",
    tolerance: float = 1e-10,
    max_iterations: int = 5000,
    damping: float = 0.5,
) -> dict[str, Any]:
    """Value iteration for the single-drive expected-points problem.

    `fourth_policy` (nf, 3), if given, fixes fourth-down behavior to those probabilities of
    (go, field goal, punt): the result is then policy evaluation, not optimization. Actions
    that are not allowed in a state get probability 0 and the rest are renormalized.

    Returns V (nf,), q (nf, 3) action values on every row (only fourth-down rows are
    decisions), policy (nf,) index into DRIVE_ACTIONS, iterations, residual (the largest
    change one undamped update would make), converged.
    """
    if after_score not in ("kickoff", "absorb"):
        raise ValueError("after_score must be 'kickoff' or 'absorb'")
    g = kern.grid
    nf = g.nf
    f10 = g.first10()[1:]
    downs, _, _ = g.unpack(np.arange(nf))
    fourth = downs == g.n_down
    k_all, k_punt = kern.k_all[mode], kern.k_punt
    fg_ok = kern.fg_ok & fourth
    allowed = np.column_stack([np.ones(nf, dtype=bool), fg_ok, kern.punt_ok])
    if fourth_policy is not None:
        mix = np.where(allowed, fourth_policy, 0.0)
        mix = mix / mix.sum(axis=1, keepdims=True)
    use_kick = 1.0 if after_score == "kickoff" else 0.0

    v = np.zeros(nf)
    q = np.zeros((nf, 3))
    residual, it = np.inf, 0
    for it in range(1, max_iterations + 1):  # noqa: B007
        k = use_kick * float(kern.kickoff[1:] @ v[f10])
        fk = use_kick * float(kern.free_kick[1:] @ v[f10])
        aug = np.concatenate([v, -v[f10], [td_value - k, -safety_points - fk, -td_value + k]])
        q[:, 0] = k_all @ aug
        q[:, 1] = kern.fg_make * (fg_points - k) + (1.0 - kern.fg_make) * aug[kern.fg_miss_col]
        q[:, 2] = k_punt @ aug
        if fourth_policy is None:
            best = np.where(allowed, q, -np.inf).max(axis=1)
        else:
            best = (mix * q).sum(axis=1)
        new = np.where(fourth, best, q[:, 0])
        residual = float(np.abs(new - v).max())
        v = (1.0 - damping) * v + damping * new
        if residual < tolerance:
            break
    policy = np.where(allowed, q, -np.inf).argmax(axis=1)
    return {
        "v": v,
        "q": q,
        "policy": policy,
        "allowed": allowed,
        "iterations": it,
        "residual": residual,
        "converged": bool(residual < tolerance),
        "kickoff_value": float(kern.kickoff[1:] @ v[f10]),
    }


def first_and_ten_curve(kern: FieldKernels, v: np.ndarray) -> np.ndarray:
    """Value of a first and 10 (or goal) at each yard line 1..n_yl."""
    return v[kern.grid.first10()[1:]]


def fourth_down_chart(kern: FieldKernels, q: np.ndarray, allowed: np.ndarray) -> dict[str, Any]:
    """Best fourth-down action and its margin over the runner-up, as (n_dist, n_yl) arrays."""
    g = kern.grid
    rows = g.fidx(g.n_down, np.arange(g.n_dist)[:, None], np.arange(1, g.n_yl + 1)[None, :])
    qq = np.where(allowed[rows], q[rows], -np.inf)
    order = np.sort(qq, axis=-1)
    return {"best": qq.argmax(axis=-1), "margin": order[..., -1] - order[..., -2], "q": qq}
