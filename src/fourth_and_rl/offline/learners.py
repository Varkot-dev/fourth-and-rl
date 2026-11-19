"""Offline learners, all written by hand and all fed the same logged data.

Every learner sees only the count tensor C[a, s, c] of `datasets.counts` (and, for the
plug-in's pooling, the layout of the field), plus two things that define the problem
rather than the data: which actions are allowed where, and the terminal reward of each
way a drive can end (`term_reward`, the same numbers M5's DriveEnv pays). So every
learner targets the same optimum, the stage 1 optimum of the solved model.

Notation: n(s, a) = sum over c of C[a, s, c]; P_hat(c | s, a) = C[a, s, c] / n(s, a);
y(c) = the value of column c given state values v (v of the next fourth-down state, or
the terminal reward); T(s, a) = sum over c of P_hat(c | s, a) y(c), the average of the
regression targets r + max over a' of Q(s', a') of the transitions logged in the cell.

Squared-error regression on individual transitions equals count-weighted regression on
cells: sum over transitions i of (q(s_i, a_i) - y_i)^2 = sum over cells of
n(s, a) (q(s, a) - T(s, a))^2 + a term that does not depend on q. So fitting on the
counts is exact, not a shortcut, and it makes every learner's cost independent of the
dataset size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from fourth_and_rl.rl.envs import DriveModel
from fourth_and_rl.rl.evaluate import greedy

NEG = -np.inf


@dataclass
class Fit:
    """What a learner returns. `q` (n, 3) is the learner's claimed action values (-inf
    where an action is not allowed, nan where the learner has no value at all);
    `policy` is a deterministic action per state or (n, 3) probabilities."""

    name: str
    policy: np.ndarray
    q: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------------------


def cell_counts(c: np.ndarray) -> np.ndarray:
    """n(s, a) as an (n, 3) array."""
    return c.sum(axis=2).T


def empirical(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(P_hat (3, n, n_cols) with zero rows for unseen cells, n(s, a) (n, 3))."""
    n_sa = cell_counts(c)
    p = np.divide(c, n_sa.T[:, :, None], out=np.zeros_like(c), where=n_sa.T[:, :, None] > 0)
    return p, n_sa


def state_values(q: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """max over allowed actions of q (nan treated as not available)."""
    qq = np.where(allowed & np.isfinite(q), q, NEG)
    v = qq.max(axis=1)
    return np.where(np.isfinite(v), v, 0.0)


def backup(model: DriveModel, p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """T(s, a) = sum over c of p[a, s, c] * y(c), with y the outcome values given v."""
    return np.einsum("asc,c->sa", p, model.outcome_values(v))


def greedy_policy(q: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    return greedy(np.where(np.isfinite(q), q, NEG), allowed)


# --------------------------------------------------------------------------------------
# Behavior cloning
# --------------------------------------------------------------------------------------


def behavior_cloning(model: DriveModel, c: np.ndarray, smoothing: float = 1.0) -> Fit:
    """A classifier of the logged action: in each state, the logged action shares with
    `smoothing` pseudo-counts spread uniformly over the allowed actions. Unseen states
    get the uniform policy. `info["mode"]` is the most common logged action (the
    deterministic "do what coaches usually do" rule)."""
    n_sa = cell_counts(c)
    prior = model.allowed / model.allowed.sum(axis=1, keepdims=True)
    probs = (n_sa + smoothing * prior) / (n_sa.sum(axis=1, keepdims=True) + smoothing)
    probs = np.where(model.allowed, probs, 0.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return Fit("bc", probs, None, {"mode": greedy(probs, model.allowed)})


# --------------------------------------------------------------------------------------
# Fitted Q iteration, tabular (and with count-based pessimism)
# --------------------------------------------------------------------------------------


def fqi_tabular(
    model: DriveModel,
    c: np.ndarray,
    q_init: float = 0.0,
    max_iter: int = 500,
    tol: float = 1e-9,
    lcb_beta: float = 0.0,
    unseen_value: float | None = None,
) -> Fit:
    """Fitted Q iteration with a table as the function class.

    Q_0 = q_init everywhere. Then repeat: v_k(s) = max over allowed a of Q_k(s, a);
    Q_{k+1}(s, a) = T(s, a) for every logged cell; cells never logged keep their value
    (a table cannot generalize, so an unlogged action is whatever the initialization
    says). This is exactly value iteration in the empirical MDP P_hat, so its limit is
    the "certainty-equivalent" model-based answer with unlogged cells frozen at q_init.

    With `lcb_beta` > 0 it becomes pessimistic value iteration: each logged cell's target
    is lowered by lcb_beta * sigma / sqrt(n(s, a)), sigma being the pooled within-cell
    standard deviation of the regression targets (recomputed each iteration), and
    unlogged cells are set to `unseen_value` (by default the worst terminal reward, the
    lowest value anything can have here).
    """
    p, n_sa = empirical(c)
    seen = (n_sa > 0) & model.allowed
    pessimistic = lcb_beta > 0
    fill = q_init
    if pessimistic:
        fill = float(model.term_reward.min()) if unseen_value is None else unseen_value
    q = np.where(model.allowed, fill, NEG)
    total = max(n_sa.sum(), 1.0)
    sigma = 0.0
    it, delta = 0, np.inf
    for it in range(1, max_iter + 1):  # noqa: B007
        v = state_values(q, model.allowed)
        t = backup(model, p, v)
        if pessimistic:
            y = model.outcome_values(v)
            second = np.einsum("asc,c->sa", p, y * y)
            within = np.maximum(second - t * t, 0.0)
            sigma = float(np.sqrt((n_sa * within).sum() / total))
            t = t - lcb_beta * sigma / np.sqrt(np.maximum(n_sa, 1.0))
        new = np.where(seen, t, np.where(model.allowed, fill, NEG))
        delta = float(np.abs(new[model.allowed] - q[model.allowed]).max())
        q = new
        if delta < tol:
            break
    name = "fqi_lcb" if pessimistic else "fqi_tabular"
    info = {"iterations": it, "last_change": delta, "q_init": fill, "lcb_beta": lcb_beta,
            "sigma": sigma}
    return Fit(name, greedy_policy(q, model.allowed), q, info)


def fqe_tabular(
    model: DriveModel,
    c: np.ndarray,
    policy: np.ndarray,
    q_init: float = 0.0,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> np.ndarray:
    """Fitted Q evaluation of a deterministic policy from the logged data (tabular):
    like `fqi_tabular` with max over a' replaced by the policy's action. Returns Q_hat^pi
    (n, 3). Uses only the data, so it is a legitimate way to choose a hyperparameter."""
    p, n_sa = empirical(c)
    seen = (n_sa > 0) & model.allowed
    q = np.where(model.allowed, q_init, NEG)
    rows = np.arange(model.n)
    for _ in range(max_iter):
        v = q[rows, policy]
        v = np.where(np.isfinite(v), v, 0.0)
        new = np.where(seen, backup(model, p, v), np.where(model.allowed, q_init, NEG))
        delta = float(np.abs(new[model.allowed] - q[model.allowed]).max())
        q = new
        if delta < tol:
            break
    return q


# --------------------------------------------------------------------------------------
# Conservative Q-learning, tabular
# --------------------------------------------------------------------------------------


def _softmax(q: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    z = np.where(allowed, q, NEG)
    z = z - z.max(axis=1, keepdims=True)
    e = np.where(allowed, np.exp(z), 0.0)
    return e / e.sum(axis=1, keepdims=True)


def cql_tabular(
    model: DriveModel,
    c: np.ndarray,
    alpha: float,
    q_init: float = 0.0,
    outer_iter: int = 100,
    inner_steps: int = 25,
) -> Fit:
    """Conservative Q-learning (the CQL(H) objective of Kumar et al. 2020) with a table.

    Each outer iteration freezes the targets T(s, a) computed from the current Q (as FQI
    does) and then minimizes, over Q, the per-sample average of

        1/2 (Q(s, a) - target)^2  +  alpha * ( log sum over a' of exp Q(s, a')  -  Q(s, a) )

    over the logged transitions (the same loss d3rlpy's DiscreteCQL uses, with a squared
    error in place of its Huber error). Divided by the visits n(s) of each state, the
    part that belongs to state s is

        f_s(q) = sum over a of mu(a|s) 1/2 (q_a - T_a)^2
                 + alpha (lse(q) - sum over a of mu(a|s) q_a)

    with mu(a|s) = n(s, a) / n(s), the logged action shares. Its gradient is
    mu(a|s) (q_a - T_a) + alpha (pi_q(a|s) - mu(a|s)) with pi_q = softmax(q): the penalty
    pushes Q down in proportion to the softmax policy and up in proportion to the data.
    For a logged action the stationary point is q_a = T_a - alpha (pi_q(a|s) / mu(a|s) - 1);
    an action never logged (mu = 0) is only ever pushed down. The inner minimization is
    `inner_steps` steps of gradient descent with step 1 / (1 + alpha) (the gradient is
    (1 + alpha)-Lipschitz), started from the current Q, so an unlogged action keeps
    sinking by about alpha * pi_q(a|s) per step: its final value is finite and depends on
    the step budget, as it does for a network trained by gradient descent. States never
    logged have no loss at all and keep q_init.
    """
    p, n_sa = empirical(c)
    n_s = n_sa.sum(axis=1, keepdims=True)
    mu = np.divide(n_sa, n_s, out=np.zeros_like(n_sa), where=n_s > 0)
    seen = (n_sa > 0) & model.allowed
    live = (n_s[:, 0] > 0)[:, None] & model.allowed
    q = np.where(model.allowed, q_init, NEG)
    eta = 1.0 / (1.0 + alpha)
    for _ in range(outer_iter):
        v = state_values(q, model.allowed)
        t = np.where(seen, backup(model, p, v), 0.0)
        for _ in range(inner_steps):
            pi = _softmax(q, model.allowed)
            grad = mu * (np.where(seen, q, 0.0) - t) + alpha * (pi - mu)
            q = np.where(live, q - eta * grad, q)
    pi = _softmax(q, model.allowed)
    with np.errstate(invalid="ignore"):
        resid = (mu * (q - t) + alpha * (pi - mu))[seen]
    info = {"alpha": alpha, "outer_iter": outer_iter, "inner_steps": inner_steps,
            "stationarity_residual_logged": float(np.abs(resid).max())}
    return Fit("cql_tabular", greedy_policy(q, model.allowed), q, info)


# --------------------------------------------------------------------------------------
# Model-based plug-in: pooled counts, solved exactly
# --------------------------------------------------------------------------------------


def shift_columns(model: DriveModel, delta: int) -> np.ndarray:
    """Where each outcome column moves if the whole play happens `delta` yards further
    from the opponent's end zone (yard line + delta). Used to borrow outcomes from a
    neighboring yard line. Next fourth-down states move by delta (past the goal line is a
    touchdown, behind our own goal line a safety); "they take over at their y" moves to
    y - delta, clipped to 1..99; scores keep their column."""
    g = model.grid
    n, n_yl = model.n, g.n_yl
    out = np.arange(n + model.n_term)
    _, dbin, yl = g.unpack(model.states)
    pos = np.full(g.nf, -1)
    pos[model.states] = np.arange(n)
    new_yl = yl + delta
    inside = (new_yl >= 1) & (new_yl <= n_yl)
    target = np.where(inside, pos[g.fidx(g.n_down, dbin, np.clip(new_yl, 1, n_yl))], -1)
    out[:n] = np.where(inside, target, np.where(new_yl < 1, n + n_yl, n + n_yl + 1))
    y = np.arange(1, n_yl + 1)
    out[n : n + n_yl] = n + np.clip(y - delta, 1, n_yl) - 1
    return out


def pooled_prior(
    model: DriveModel,
    c: np.ndarray,
    bandwidth: float,
    max_shift: int,
    translate: tuple[bool, bool, bool] = (True, True, False),
) -> tuple[np.ndarray, np.ndarray]:
    """For every (action, state): a prior outcome distribution borrowed from the same
    action at nearby yard lines with the same distance bin, each neighbor's outcomes
    translated by the yard difference (`shift_columns`) for the actions flagged in
    `translate` (go and field goal: a play from two yards further back ends two yards
    further back) and taken as they are for the others (punt: touchbacks and fair catches
    land at fixed spots, so a punt from the 52 lands where a punt from the 50 does; this was
    checked against the true model, where translating punts was the worse
    approximation), and weighted by its count times
    exp(-delta^2 / (2 bandwidth^2)), for 1 <= |delta| <= max_shift. A cell whose
    neighborhood holds no data borrows from every yard line of its distance bin with the
    same weights (no cut-off). Returns (prior (3, n, n_cols), total neighbor weight (n, 3))."""
    g = model.grid
    n, n_cols = model.n, c.shape[2]
    _, dbin, yl = g.unpack(model.states)
    pos = np.full(g.nf, -1)
    pos[model.states] = np.arange(n)
    shift = tuple(bool(x) for x in translate)

    def accumulate(deltas: range, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prior = np.zeros_like(c)
        weight = np.zeros((n, 3))
        for d in deltas:
            if d == 0:
                continue
            src_yl = yl - d
            ok = rows & (src_yl >= 1) & (src_yl <= g.n_yl)
            src = np.where(ok, pos[g.fidx(g.n_down, dbin, np.clip(src_yl, 1, g.n_yl))], -1)
            ok &= src >= 0
            dst = np.nonzero(ok)[0]
            if not len(dst):
                continue
            cols = shift_columns(model, d)
            move = sp.csr_matrix(
                (np.ones(n_cols), (np.arange(n_cols), cols)), shape=(n_cols, n_cols)
            )
            w = np.exp(-(d * d) / (2.0 * bandwidth**2))
            for a in range(3):
                block = c[a, src[dst]]  # raw counts: weight = count * kernel
                moved = np.asarray((move.T @ block.T).T) if shift[a] else block
                prior[a, dst] += w * moved
                weight[dst, a] += w * block.sum(axis=1)
        return prior, weight

    prior, weight = accumulate(range(-max_shift, max_shift + 1), np.ones(n, dtype=bool))
    empty = (weight <= 0) & model.allowed
    if empty.any():
        wide, wide_w = accumulate(range(-(g.n_yl - 1), g.n_yl), empty.any(axis=1))
        for a in range(3):
            e = empty[:, a]
            prior[a, e] = wide[a, e]
            weight[e, a] = wide_w[e, a]
    tot = prior.sum(axis=2, keepdims=True)
    prior = np.divide(prior, tot, out=np.zeros_like(prior), where=tot > 0)
    return prior, weight


def solve_tabular_mdp(
    model: DriveModel, p: np.ndarray, available: np.ndarray, max_iter: int = 5000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, int]:
    """Value iteration on a drive model with transition probabilities p (3, n, n_cols)
    and the true terminal rewards. Unavailable cells hold -inf. Returns (Q, iterations)."""
    v = np.zeros(model.n)
    it = 0
    q = np.full((model.n, 3), NEG)
    for it in range(1, max_iter + 1):  # noqa: B007
        q = np.where(available, backup(model, p, v), NEG)
        new = state_values(q, available)
        delta = float(np.abs(new - v).max())
        v = new
        if delta < tol:
            break
    return q, it


def plugin(
    model: DriveModel,
    c: np.ndarray,
    prior_strength: float = 5.0,
    bandwidth: float = 3.0,
    max_shift: int = 10,
    translate: tuple[bool, bool, bool] = (True, True, False),
) -> Fit:
    """The analytics-style approach: estimate the transition model from the logged data,
    solve it exactly, act greedily on the solution.

    P_tilde(. | s, a) = (C[a, s, .] + k * prior(. | s, a)) / (n(s, a) + k), with k =
    `prior_strength` pseudo-counts and the prior from `pooled_prior` (the same action at
    nearby yard lines). This pooling is the one modelling assumption the plug-in adds:
    a play from the 30 behaves like the same play from the 32, shifted by two yards. An
    action with no data in its whole distance bin is treated as unavailable.
    """
    prior, weight = pooled_prior(model, c, bandwidth, max_shift, translate)
    n_sa = cell_counts(c)
    has_prior = weight > 0
    k = np.where(has_prior, prior_strength, 0.0)
    num = c + k.T[:, :, None] * prior
    den = n_sa + k
    p = np.divide(num, den.T[:, :, None], out=np.zeros_like(c), where=den.T[:, :, None] > 0)
    available = model.allowed & (den > 0)
    q, it = solve_tabular_mdp(model, p, available)
    q = np.where(model.allowed & ~available, np.nan, q)
    info = {"prior_strength": prior_strength, "bandwidth": bandwidth, "max_shift": max_shift,
            "iterations": it, "cells_without_any_data": int((model.allowed & ~available).sum())}
    return Fit("plugin", greedy_policy(q, model.allowed), q, info)


# --------------------------------------------------------------------------------------
# Fitted Q iteration with gradient-boosted trees
# --------------------------------------------------------------------------------------


def fqi_gbt(
    model: DriveModel,
    c: np.ndarray,
    n_iter: int = 20,
    seed: int = 0,
    params: dict[str, Any] | None = None,
) -> Fit:
    """Fitted Q iteration with a function approximator: one gradient-boosted regression
    tree model per action (scikit-learn HistGradientBoostingRegressor) on the M5 features
    (yard line, distance, distance-bin one-hot).

    Each iteration fits Q_{k+1}(., a) to the targets r + max over allowed a' of Q_k(s', a')
    of the logged transitions with action a, as count-weighted cell means (exact for the
    squared loss, see the module docstring; the one difference from fitting raw rows is
    that the leaf-size limit counts cells, not transitions). Q of an action in a state where
    it was never logged is the trees' extrapolation from where it was.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    params = {"max_iter": 50, "learning_rate": 0.1, "max_leaf_nodes": 15,
              "min_samples_leaf": 5, **(params or {})}
    p, n_sa = empirical(c)
    x = model.features().astype(np.float64)
    q = np.where(model.allowed, 0.0, NEG)
    for _ in range(n_iter):
        v = state_values(q, model.allowed)
        t = backup(model, p, v)
        new = np.full((model.n, 3), NEG)
        for a in range(3):
            rows = np.nonzero((n_sa[:, a] > 0) & model.allowed[:, a])[0]
            if len(rows) == 0:
                new[:, a] = np.where(model.allowed[:, a], np.nan, NEG)
                continue
            if len(rows) < 2 * params["min_samples_leaf"]:
                pred = np.full(model.n, np.average(t[rows, a], weights=n_sa[rows, a]))
            else:
                reg = HistGradientBoostingRegressor(early_stopping=False, random_state=seed,
                                                    **params)
                reg.fit(x[rows], t[rows, a], sample_weight=n_sa[rows, a])
                pred = reg.predict(x)
            new[:, a] = np.where(model.allowed[:, a], pred, NEG)
        q = new
    return Fit("fqi_gbt", greedy_policy(q, model.allowed), q, {"n_iter": n_iter, **params})


# --------------------------------------------------------------------------------------
# Abstain: the plug-in when its bootstrap agrees, otherwise the coaches' usual action
# --------------------------------------------------------------------------------------


def bootstrap_agreement(policies: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Share of bootstrap policies (B, n) that pick the reference action in each state."""
    return (policies == reference[None, :]).mean(axis=0)


def abstain(reference: np.ndarray, agreement: np.ndarray, default: np.ndarray,
            threshold: float) -> np.ndarray:
    """The reference action where agreement >= threshold, else the default action."""
    return np.where(agreement >= threshold, reference, default)
