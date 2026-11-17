"""Exact measurement of offline learners in the known-truth drive model.

Everything here is computed, not sampled: the true value of a learned policy comes from
exact policy evaluation (`rl.evaluate.policy_value`, a linear solve), the true action
values from `model.q_star` (the optimum) or `rl.evaluate.policy_q` (a given policy).

Three families of numbers:

1. Regret: start-weighted V* minus V_pi, in expected points per drive.
2. Distribution shift: the learned Q minus the true Q*, by how often the (state, action)
   cell was logged. Fitted Q iteration targets Q*, so Q* is its truth; CQL's values are
   compared with the same Q* so the two can be put side by side.
3. The optimizer's curse (research note, experiment E1): in each logged state, the value
   a method claims for the action it picks, Q_hat(s, pi_hat(s)), against what that action
   is truly worth when the method's own policy is followed afterwards,
   Q^pi_hat(s, pi_hat(s)) = V^pi_hat(s). Split by whether the chosen action is the one the
   behavior policy usually takes in that state ("usual") or not ("rare").
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fourth_and_rl.offline.learners import Fit, cell_counts
from fourth_and_rl.rl import evaluate as ev
from fourth_and_rl.rl.envs import GO, DriveModel

COUNT_BUCKETS = ((0, 0), (1, 9), (10, 99), (100, 999), (1000, None))


def bucket_label(lo: int, hi: int | None) -> str:
    if hi is None:
        return f"{lo}+"
    return str(lo) if lo == hi else f"{lo}-{hi}"


def reachable(model: DriveModel) -> np.ndarray:
    """States some policy can reach (visited by the uniform policy)."""
    uniform = model.allowed / model.allowed.sum(axis=1, keepdims=True)
    return ev.occupancy(model, uniform) > 1e-12


_OCC_OPT: dict[int, np.ndarray] = {}


def optimal_occupancy(model: DriveModel) -> np.ndarray:
    """Visits per drive under the optimal policy (cached per model object)."""
    key = id(model)
    if key not in _OCC_OPT:
        _OCC_OPT[key] = ev.occupancy(model, model.policy_star)
    return _OCC_OPT[key]


def exact_q(model: DriveModel, v: np.ndarray) -> np.ndarray:
    """Q_pi (n, 3) from the exact V_pi: take the action once, then follow pi."""
    q = np.einsum("asc,c->sa", model.dist, model.outcome_values(v))
    return np.where(model.allowed, q, -np.inf)


def policy_stats(model: DriveModel, policy: np.ndarray, behavior: np.ndarray,
                 data_weight: np.ndarray, v: np.ndarray | None = None) -> dict[str, Any]:
    """Regret, go rate and where a deterministic or stochastic policy differs from the
    optimum and from the behavior policy's most common action. `v` is the policy's exact
    value if already computed."""
    probs = ev.as_probs(policy, model.allowed)
    occ = ev.occupancy(model, probs)
    v = ev.policy_value(model, probs) if v is None else v
    out: dict[str, Any] = {
        "regret": float(model.start @ (model.v_star - v)),
        "value": float(model.start @ v),
        "go_rate_own_visits": float((occ * probs[:, GO]).sum() / occ.sum()),
    }
    occ_opt = optimal_occupancy(model)
    q = np.where(model.allowed, model.q_star, 0.0)
    loss = model.v_star - (probs * q).sum(axis=1)
    non_opt = loss > 1e-9
    mode = ev.greedy(behavior, model.allowed)
    differs_mode = 1.0 - probs[np.arange(model.n), mode]
    w = data_weight / max(data_weight.sum(), 1e-12)
    out.update(
        share_suboptimal_optimal_visits=float((occ_opt * non_opt).sum() / occ_opt.sum()),
        share_suboptimal_data_states=float((w * non_opt).sum()),
        share_differs_from_behavior_mode_data_states=float((w * differs_mode).sum()),
        share_go_where_optimum_kicks_data_states=float(
            (w * probs[:, GO] * (model.policy_star != GO)).sum()
        ),
        share_kick_where_optimum_goes_data_states=float(
            (w * (1 - probs[:, GO]) * (model.policy_star == GO)).sum()
        ),
    )
    return out


def q_error_by_count(model: DriveModel, q: np.ndarray, c: np.ndarray,
                     reach: np.ndarray) -> dict[str, Any]:
    """Mean and spread of Q_hat - Q* per action and count bucket, over reachable states
    (each state counts once). Cells where the learner has no value (nan) are counted
    separately."""
    n_sa = cell_counts(c)
    err = q - np.where(model.allowed, model.q_star, np.nan)
    out: dict[str, Any] = {}
    for a, name in enumerate(("go", "field_goal", "punt")):
        rows = {}
        for lo, hi in COUNT_BUCKETS:
            sel = reach & model.allowed[:, a] & (n_sa[:, a] >= lo)
            if hi is not None:
                sel &= n_sa[:, a] <= hi
            e = err[sel, a]
            fin = e[np.isfinite(e)]
            rows[bucket_label(lo, hi)] = {
                "n_states": int(sel.sum()),
                "n_without_value": int((~np.isfinite(e)).sum()),
                "mean": float(fin.mean()) if len(fin) else None,
                "median": float(np.median(fin)) if len(fin) else None,
                "p90": float(np.quantile(fin, 0.9)) if len(fin) else None,
                "share_over_1_point": float((fin > 1.0).mean()) if len(fin) else None,
            }
        out[name] = rows
    return out


def optimizers_curse(model: DriveModel, fit: Fit, c: np.ndarray, behavior: np.ndarray,
                     true_q: np.ndarray | None = None) -> dict[str, Any]:
    """Claimed versus true value of each method's chosen action in the logged states,
    weighted by how often each state was logged. `true_q` defaults to the exact
    Q^pi_hat of the method's own greedy policy."""
    if fit.q is None or fit.policy.ndim != 1:
        return {}
    pol = fit.policy
    n_s = cell_counts(c).sum(axis=1)
    rows = np.arange(model.n)
    tq = ev.policy_q(model, pol) if true_q is None else true_q
    claim = fit.q[rows, pol]
    true = tq[rows, pol]
    mode = ev.greedy(behavior, model.allowed)
    usual = pol == mode
    ok = (n_s > 0) & np.isfinite(claim) & np.isfinite(true)
    gap = claim - true
    out: dict[str, Any] = {}
    for label, sel in (("all", ok), ("usual", ok & usual), ("rare", ok & ~usual)):
        w = n_s[sel]
        tot = w.sum()
        out[label] = {
            "share_of_logged_decisions": float(tot / max(n_s[ok].sum(), 1)),
            "n_states": int(sel.sum()),
            "mean_claimed_minus_true": float((w * gap[sel]).sum() / tot) if tot else None,
            "mean_claimed": float((w * claim[sel]).sum() / tot) if tot else None,
            "mean_true": float((w * true[sel]).sum() / tot) if tot else None,
        }
    # Claimed gain over the usual action versus its true gain, where the method departs.
    sel = ok & ~usual & np.isfinite(fit.q[rows, mode])
    if sel.sum() >= 3:
        g_claim = claim[sel] - fit.q[rows, mode][sel]
        g_true = true[sel] - tq[rows, mode][sel]
        w = n_s[sel]
        xm, ym = np.average(g_claim, weights=w), np.average(g_true, weights=w)
        var = np.average((g_claim - xm) ** 2, weights=w)
        slope = float(np.average((g_claim - xm) * (g_true - ym), weights=w) / var) if var > 0 \
            else None
        out["gain_over_usual_action"] = {
            "n_states": int(sel.sum()),
            "mean_claimed_gain": float(xm),
            "mean_true_gain": float(ym),
            "share_truly_worse": float(np.average(g_true < 0, weights=w)),
            "slope_true_on_claimed": slope,
        }
    return out


def heatmap_rows(model: DriveModel, fit: Fit, c: np.ndarray, action: int = GO) -> dict:
    """Per-state arrays for the distribution-shift figure, laid out as charts."""
    n_sa = cell_counts(c)
    q = fit.q[:, action] if fit.q is not None else np.full(model.n, np.nan)
    err = q - model.q_star[:, action]
    err = np.where(model.allowed[:, action], err, np.nan)
    to_list = lambda x: np.where(np.isfinite(x), x, None).tolist()  # noqa: E731
    return {
        "count": ev.chart(model, n_sa[:, action]).tolist(),
        "error": to_list(ev.chart(model, err)),
        "q": to_list(ev.chart(model, np.where(np.isfinite(q), q, np.nan))),
        "policy": ev.chart(model, fit.policy if fit.policy.ndim == 1
                           else fit.policy.argmax(axis=1)).tolist(),
    }


def error_vs_count_points(model: DriveModel, fit: Fit, c: np.ndarray, reach: np.ndarray,
                          action: int = GO) -> dict[str, list]:
    """(count, error) of every reachable state for one action, for scatter plots."""
    n_sa = cell_counts(c)[:, action]
    err = (fit.q[:, action] if fit.q is not None else np.nan) - model.q_star[:, action]
    sel = reach & model.allowed[:, action] & np.isfinite(err)
    return {"count": n_sa[sel].astype(int).tolist(), "error": err[sel].round(4).tolist()}


def evaluate_fit(model: DriveModel, fit: Fit, c: np.ndarray, behavior: np.ndarray,
                 reach: np.ndarray, curse: bool = True) -> dict[str, Any]:
    """Every summary number of one learned policy on one dataset."""
    n_s = cell_counts(c).sum(axis=1)
    v = ev.policy_value(model, ev.as_probs(fit.policy, model.allowed))
    out = policy_stats(model, fit.policy, behavior, n_s, v)
    if fit.q is not None:
        out["q_error_by_count"] = q_error_by_count(model, fit.q, c, reach)
        if curse:
            out["optimizers_curse"] = optimizers_curse(model, fit, c, behavior, exact_q(model, v))
    out["info"] = {k: v for k, v in fit.info.items() if np.isscalar(v) or v is None}
    return out
