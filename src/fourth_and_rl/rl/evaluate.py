"""M5 measurement: how far is a learned policy from the exact optimum?

Drive (exact). With the decision-level model of `envs.DriveModel` a policy pi turns the
problem into a Markov chain with transition matrix P_pi and expected one-step reward
r_pi, and its value solves the linear system

    V_pi = r_pi + P_pi V_pi          =>          V_pi = (I - P_pi)^-1 r_pi

No sampling is involved, so "regret" below is a true number, not an estimate:

    regret(pi) = sum over start states s of start(s) * (V*(s) - V_pi(s))      [points]

End game (nearly exact). The state space is too large to tabulate a policy network, so a
policy is run on a fixed set of start states. Instead of counting wins (noisy: a win is a
coin flip) we add up, along each simulated game, the exact loss of every action the agent
took, read from the solved model:

    regret(pi) = E_pi [ sum over the agent's decisions of  V*(s) - Q*(s, a) ]

This is the performance-difference identity: the optimal value drops by exactly
V*(s) - Q*(s, a) in expectation each time a worse action is played, and never otherwise.
The estimate is unbiased and has a far smaller variance than the win rate, because a game
in which the agent played optimally contributes exactly zero.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from fourth_and_rl import mdp_game as mg
from fourth_and_rl.rl.envs import FIELD_GOAL, GO, PUNT, DriveModel, EndgameBatch, StartSpec

# --------------------------------------------------------------------------------------
# Drive: exact
# --------------------------------------------------------------------------------------


def as_probs(
    policy: np.ndarray, allowed: np.ndarray, return_fallbacks: bool = False
) -> np.ndarray | tuple[np.ndarray, int]:
    """A policy as probabilities (n, 3). Accepts action indices (n,) or probabilities;
    probability on actions that are not allowed is removed and the rest renormalized.

    A state where the policy puts no mass on any allowed action falls back to uniform over
    the allowed actions. With `return_fallbacks=True` the result is (probabilities, number
    of states that fell back), so a caller can see that it happened."""
    policy = np.asarray(policy)
    if policy.ndim == 1:
        probs = np.zeros(allowed.shape)
        probs[np.arange(len(policy)), policy] = 1.0
    else:
        probs = policy.astype(float).copy()
    probs = np.where(allowed, probs, 0.0)
    total = probs.sum(axis=1, keepdims=True)
    fallback = allowed / allowed.sum(axis=1, keepdims=True)
    out = np.where(total > 0, probs / np.where(total > 0, total, 1.0), fallback)
    if return_fallbacks:
        return out, int(np.sum(total[:, 0] <= 0))
    return out


def greedy(q: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Best allowed action of each state under action values q (n, 3)."""
    return np.where(allowed, q, -np.inf).argmax(axis=1)


def policy_chain(model: DriveModel, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(P_pi (n, n), r_pi (n,)) of a policy on the decision-level drive model."""
    probs = as_probs(policy, model.allowed)
    mixed = np.einsum("sa,asc->sc", probs, model.dist)
    return mixed[:, : model.n], mixed[:, model.n :] @ model.term_reward


def policy_value(model: DriveModel, policy: np.ndarray) -> np.ndarray:
    """Exact value V_pi (n,) of a policy: solve (I - P_pi) V = r_pi."""
    p, r = policy_chain(model, policy)
    return np.linalg.solve(np.eye(model.n) - p, r)


def policy_q(model: DriveModel, policy: np.ndarray) -> np.ndarray:
    """Exact action values Q_pi (n, 3): take the action once, then follow the policy."""
    v = policy_value(model, policy)
    q = np.einsum("asc,c->sa", model.dist, model.outcome_values(v))
    return np.where(model.allowed, q, -np.inf)


def occupancy(model: DriveModel, policy: np.ndarray) -> np.ndarray:
    """Expected number of visits to each fourth-down state in one episode under a policy:
    d = start (I - P_pi)^-1."""
    p, _ = policy_chain(model, policy)
    return np.linalg.solve((np.eye(model.n) - p).T, model.start)


def drive_regret(model: DriveModel, policy: np.ndarray) -> float:
    """Exact regret in points: start-weighted V* minus V_pi."""
    return float(model.start @ (model.v_star - policy_value(model, policy)))


def regret_by_state(model: DriveModel, policy: np.ndarray) -> np.ndarray:
    """Where the regret comes from: visits under the policy times the loss of its action
    in each state, occupancy_pi(s) * (V*(s) - Q*(s, pi(s))). Sums to `drive_regret`."""
    probs = as_probs(policy, model.allowed)
    q = np.where(model.allowed, model.q_star, 0.0)
    loss = model.v_star - (probs * q).sum(axis=1)
    return occupancy(model, policy) * loss


def drive_agreement(model: DriveModel, policy: np.ndarray, tie: float = 1e-9) -> dict[str, float]:
    """Share of fourth-down decisions on which a deterministic policy picks an optimal
    action: weighted by how often the optimal policy visits each state, and unweighted
    over the states that can be reached at all."""
    policy = np.asarray(policy)
    q = model.q_star[np.arange(model.n), policy]
    same = q >= model.v_star - tie
    visits = occupancy(model, model.policy_star)
    reach = occupancy(model, model.allowed / model.allowed.sum(axis=1, keepdims=True)) > 1e-12
    return {
        "weighted_by_optimal_visits": float((visits * same).sum() / visits.sum()),
        "unweighted_reachable_states": float(same[reach].mean()),
        "n_reachable_states": int(reach.sum()),
        "n_states_differing": int((~same & reach).sum()),
    }


def best_restricted_policy(model: DriveModel, actions: tuple[int, ...]) -> np.ndarray:
    """The best policy that only ever uses `actions` (policy iteration on the exact model).
    States where none of them is allowed fall back to every allowed action."""
    allow = np.zeros_like(model.allowed)
    allow[:, list(actions)] = model.allowed[:, list(actions)]
    allow = np.where(allow.any(axis=1, keepdims=True), allow, model.allowed)
    policy = allow.argmax(axis=1)
    for _ in range(100):
        v = policy_value(model, policy)
        q = np.einsum("asc,c->sa", model.dist, model.outcome_values(v))
        new = np.where(allow, q, -np.inf).argmax(axis=1)
        keep = q[np.arange(model.n), policy] >= q[np.arange(model.n), new] - 1e-12
        new = np.where(keep, policy, new)
        if (new == policy).all():
            break
        policy = new
    return policy


def coach_drive_policy(model: DriveModel, coach_table: np.ndarray) -> np.ndarray:
    """Observed coach frequencies (neutral situation) as a policy on the drive model."""
    _, dbin, yl = model.grid.unpack(model.states)
    return as_probs(coach_table[0, dbin, yl - 1], model.allowed)


def drive_baselines(model: DriveModel, coach_table: np.ndarray | None = None) -> dict[str, Any]:
    """Exact regret of the reference policies."""
    never = best_restricted_policy(model, (FIELD_GOAL, PUNT))
    always = np.full(model.n, GO)
    uniform = model.allowed / model.allowed.sum(axis=1, keepdims=True)
    out = {
        "never_go": drive_regret(model, never),
        "always_go": drive_regret(model, always),
        "random": drive_regret(model, uniform),
    }
    if coach_table is not None:
        out["coach"] = drive_regret(model, coach_drive_policy(model, coach_table))
    return out


def chart(model: DriveModel, per_state: np.ndarray) -> np.ndarray:
    """A per-state array laid out as a fourth-down chart (n_dist, n_yl)."""
    g = model.grid
    _, dbin, yl = g.unpack(model.states)
    out = np.full((g.n_dist, g.n_yl), np.nan)
    out[dbin, yl - 1] = per_state
    return out


# --------------------------------------------------------------------------------------
# Learning curves and sample efficiency
# --------------------------------------------------------------------------------------


def log_checkpoints(n_steps: int, n_points: int, first: int = 1000) -> list[int]:
    """About `n_points` step counts spaced evenly on a log axis from `first` to `n_steps`."""
    pts = np.unique(np.round(np.geomspace(min(first, n_steps), n_steps, n_points)).astype(int))
    return [int(p) for p in pts]


def steps_to_threshold(
    steps: np.ndarray, regret: np.ndarray, threshold: float
) -> float | None:
    """First recorded step count from which the regret stays at or below `threshold` for
    the rest of the run (None if that never happens). "Stays" matters: a noisy curve can
    dip under a line once and come back."""
    below = np.asarray(regret) <= threshold
    if not below[-1]:
        return None
    last_above = np.nonzero(~below)[0]
    first = 0 if len(last_above) == 0 else int(last_above[-1]) + 1
    return float(np.asarray(steps)[first])


def summarize_curves(
    steps: list[int], curves: np.ndarray, thresholds: list[float]
) -> dict[str, Any]:
    """Mean, spread and sample efficiency of regret curves (n_seeds, n_checkpoints)."""
    curves = np.asarray(curves, dtype=float)
    n = curves.shape[0]
    final = curves[:, -1]
    sd = float(final.std(ddof=1)) if n > 1 else 0.0
    out: dict[str, Any] = {
        "steps": [int(s) for s in steps],
        "mean": curves.mean(axis=0).tolist(),
        "min": curves.min(axis=0).tolist(),
        "max": curves.max(axis=0).tolist(),
        "per_seed": curves.tolist(),
        "final_regret": {
            "mean": float(final.mean()),
            "sd_over_seeds": sd,
            "se_over_seeds": sd / np.sqrt(n),
            "min": float(final.min()),
            "max": float(final.max()),
            "n_seeds": int(n),
        },
        "steps_to_threshold": {},
    }
    for thr in thresholds:
        per_seed = [steps_to_threshold(np.asarray(steps), c, thr) for c in curves]
        reached = [s for s in per_seed if s is not None]
        out["steps_to_threshold"][str(thr)] = {
            "mean_curve": steps_to_threshold(np.asarray(steps), curves.mean(axis=0), thr),
            "per_seed": per_seed,
            "seeds_reached": len(reached),
            "median_of_seeds_reached": float(np.median(reached)) if reached else None,
        }
    return out


# --------------------------------------------------------------------------------------
# End game
# --------------------------------------------------------------------------------------

Policy = Callable[[EndgameBatch, np.ndarray], np.ndarray]


def build_loss_table(sol: mg.Solution, path: Path) -> np.ndarray:
    """loss[t - 1, f, s, a, b, action] = V*(state) - Q*(state, action) for every state of
    the end phase, computed with the solver's own backup, saved as float16 (a loss is a
    small number, so half precision keeps three significant digits) and memory-mapped.
    Actions that are not allowed hold +inf."""
    m = sol.model
    te = m.clock.endgame_ticks
    shape = (te, sol.grid.nf, m.n_score, *m.extra_shape("end"), 6)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    for t in range(1, te + 1):
        q, _ = sol.backup(t)
        best = np.maximum.reduce(list(q.values()))
        for code in range(6):
            if code in q:
                with np.errstate(invalid="ignore"):
                    out[t - 1, ..., code] = np.maximum(best - q[code], 0.0)
            else:
                out[t - 1, ..., code] = np.inf
    out.flush()
    return np.load(path, mmap_mode="r")


def load_loss_table(sol: mg.Solution, path: Path, rebuild: bool = False) -> np.ndarray:
    """The loss table of a solution, built on first use. A small text file next to it
    records which solution it belongs to, so a new solve rebuilds it."""
    path = Path(path)
    tag = path.with_suffix(".tag")
    stamp = repr(sorted((k, str(v)) for k, v in sol.meta.items()))
    if path.exists() and tag.exists() and tag.read_text() == stamp and not rebuild:
        return np.load(path, mmap_mode="r")
    table = build_loss_table(sol, path)
    tag.write_text(stamp)
    return table


def optimal_policy(batch: EndgameBatch, idx: np.ndarray) -> np.ndarray:
    """The solved policy."""
    return batch.sim.optimal_actions(batch.st, idx)


def coach_policy(batch: EndgameBatch, idx: np.ndarray) -> np.ndarray:
    """The simulator's coach-like policy (needs a coach table)."""
    return batch.sim.coach_actions(batch.st, idx, batch.rng)


def random_policy(batch: EndgameBatch, idx: np.ndarray) -> np.ndarray:
    """A uniformly random allowed action."""
    mask = batch.mask(idx)
    u = batch.rng.random(mask.shape)
    return np.where(mask, u, -1.0).argmax(axis=1)


def make_never_go(loss: np.ndarray) -> Policy:
    """Optimal on downs 1 to 3; on fourth down the best action that is not a play (field
    goal, punt or kneel), read from the loss table."""

    def policy(batch: EndgameBatch, idx: np.ndarray) -> np.ndarray:
        st, g, m = batch.st, batch.sim.grid, batch.sim.model
        act = batch.sim.optimal_actions(st, idx)
        down = g.unpack(st["f"][idx])[0]
        j = idx[down == g.n_down]
        if len(j):
            rows = np.asarray(
                loss[st["t"][j] - 1, st["f"][j], st["s"][j] + m.score_max, st["a"][j], st["b"][j]]
            ).astype(np.float32)
            act[down == g.n_down] = mg.A_FG + rows[:, mg.A_FG :].argmin(axis=1)
        return act

    return policy


def evaluate_endgame(
    sim: mg.Simulator,
    policy: Policy,
    starts: dict[str, np.ndarray],
    seed: int,
    loss: np.ndarray | None = None,
    opponent: str = "optimal",
    max_steps: int = 2000,
) -> dict[str, Any]:
    """Play one game from every start state with `policy` calling the agent's plays.

    Returns the win rate (ties count one half) with its standard error, the mean solved
    value of the start states, and, if `loss` is given (the table of `load_loss_table`,
    valid only against the optimal opponent), the regret estimated by summing exact
    per-decision losses, its standard error, and the share of decisions with zero loss.
    """
    n = len(starts["t"])
    batch = EndgameBatch(sim, n, seed=seed, opponent=opponent)
    batch.reset(states=starts)
    v_start = sim.values(batch.st).astype(float)
    m = sim.model
    game_loss = np.zeros(n)
    n_dec, n_opt, n_same = 0, 0, 0
    fourth_dec, fourth_opt = 0, 0
    for _ in range(max_steps):
        idx = np.nonzero(~batch.finished)[0]
        if len(idx) == 0:
            break
        best = sim.optimal_actions(batch.st, idx)
        actions = np.zeros(n, dtype=np.int64)
        actions[idx] = policy(batch, idx)
        batch.step(actions)
        last = batch.last
        n_dec += len(idx)
        n_same += int((last["action"] == best).sum())
        if loss is not None:
            step_loss = np.asarray(
                loss[
                    last["t"] - 1,
                    last["f"],
                    last["s"] + m.score_max,
                    last["a"],
                    last["b"],
                    last["action"],
                ]
            ).astype(float)
            game_loss[idx] += step_loss
            zero = step_loss <= 1e-6
            n_opt += int(zero.sum())
            fourth = sim.grid.unpack(last["f"])[0] == sim.grid.n_down
            fourth_dec += int(fourth.sum())
            fourth_opt += int((zero & fourth).sum())
    if not batch.finished.all():
        raise RuntimeError("evaluation games did not finish; raise max_steps")
    res = batch.result
    out: dict[str, Any] = {
        "n_games": int(n),
        "win_rate": float(res.mean()),
        "win_rate_se": float(res.std(ddof=1) / np.sqrt(n)),
        "optimal_win_rate_of_starts": float(v_start.mean()),
        "decisions_per_game": n_dec / n,
        "same_action_as_optimal": n_same / max(n_dec, 1),
    }
    if loss is not None:
        out.update(
            regret=float(game_loss.mean()),
            regret_se=float(game_loss.std(ddof=1) / np.sqrt(n)),
            zero_loss_decisions=n_opt / max(n_dec, 1),
            zero_loss_fourth_downs=fourth_opt / max(fourth_dec, 1),
            fourth_down_decisions_per_game=fourth_dec / n,
        )
    return out


def sample_starts(sim: mg.Simulator, n: int, seed: int, spec: StartSpec | None = None) -> dict:
    """A fixed set of n start states (the evaluation set)."""
    return EndgameBatch(sim, 1, seed=seed, start=spec).sample_states(n)
