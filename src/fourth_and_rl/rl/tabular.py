"""Hand-written tabular agents on the drive problem: Q-learning, SARSA, Expected SARSA.

Plain Python on purpose: every line of the update is visible, and with lists instead of
NumPy arrays one step costs a few microseconds. The environment is `envs.CompiledDrive`,
which samples from the decision-level drive model; a "step" is one fourth-down decision.

The three updates differ only in the target (r is the reward, s2 the next state):

    Q-learning       target = r + max over a2 of Q(s2, a2)              (off-policy)
    SARSA            target = r + Q(s2, a2), a2 the action actually taken next (on-policy)
    Expected SARSA   target = r + sum over a2 of pi(a2 | s2) Q(s2, a2)  (on-policy, less noise)

    Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))

and at the end of an episode target = r. No discount: every episode ends with probability
one, so the undiscounted return is finite and is exactly the quantity the DP optimized.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from fourth_and_rl.rl.envs import CompiledDrive


@dataclass
class Schedule:
    """Exploration and step-size schedules.

    epsilon: "linear" decays from eps_start to eps_end over the first `eps_decay_frac` of
    the run, then stays; "constant" is eps_start throughout.
    step size: "count" uses alpha = max(alpha0 / n(s, a) ** omega, alpha_min), where
    n(s, a) is the number of updates of that pair so far (omega = 1 is a plain running
    average); "constant" uses alpha0.
    """

    eps_kind: str = "linear"
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.5
    alpha_kind: str = "count"
    alpha0: float = 1.0
    omega: float = 0.7
    alpha_min: float = 0.0

    def eps(self, t: int, n_steps: int) -> float:
        if self.eps_kind == "constant":
            return self.eps_start
        frac = min(t / max(self.eps_decay_frac * n_steps, 1.0), 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * frac


def _uniforms(rng: np.random.Generator, size: int = 1 << 20):
    """An endless stream of uniform numbers, drawn in chunks (fast and reproducible)."""
    while True:
        yield from rng.random(size).tolist()


def td_control(
    env: CompiledDrive,
    n_steps: int,
    seed: int,
    algo: str = "q_learning",
    schedule: Schedule | None = None,
    q_init: float = 0.0,
    checkpoints: list[int] | None = None,
    on_checkpoint: Callable[[np.ndarray, int, int], Any] | None = None,
) -> dict[str, Any]:
    """Run one learner for `n_steps` decisions.

    algo: "q_learning", "sarsa" or "expected_sarsa". `q_init` sets every allowed Q(s, a)
    at the start (0, or a large value for optimistic initialization). At each step count
    in `checkpoints`, `on_checkpoint(Q, steps, episodes)` is called with a copy of Q as an
    (n, 3) array (-inf where not allowed) and its result recorded.
    Returns Q, visit counts, the checkpoint records and the number of episodes.
    """
    if algo not in ("q_learning", "sarsa", "expected_sarsa"):
        raise ValueError(f"unknown algorithm {algo}")
    sch = schedule or Schedule()
    rng = np.random.default_rng(seed)
    u = _uniforms(rng)
    nxt = u.__next__
    n_s = env.n_states
    valid = env.valid
    q = [[q_init if a in valid[s] else float("-inf") for a in range(3)] for s in range(n_s)]
    cnt = [[0, 0, 0] for _ in range(n_s)]
    marks = sorted(set(checkpoints or []))
    records: list[Any] = []
    next_mark = marks.pop(0) if marks else None
    count_alpha = sch.alpha_kind == "count"
    a0, om, amin = sch.alpha0, sch.omega, sch.alpha_min

    def choose(s: int, eps: float) -> int:
        vs = valid[s]
        if nxt() < eps:
            return vs[int(nxt() * len(vs))]
        row = q[s]
        best = vs[0]
        for a in vs[1:]:
            if row[a] > row[best]:
                best = a
        return best

    def as_array() -> np.ndarray:
        return np.array(q, dtype=float)

    episodes = 0
    s = env.reset(nxt())
    eps = sch.eps(0, n_steps)
    a = choose(s, eps)
    for t in range(1, n_steps + 1):
        if t & 1023 == 0:
            eps = sch.eps(t, n_steps)
        s2, r, done = env.step(s, a, nxt())
        c = cnt[s][a] + 1
        cnt[s][a] = c
        alpha = max(a0 / c**om, amin) if count_alpha else a0
        if done:
            target = r
            episodes += 1
            s2 = env.reset(nxt())
            a2 = choose(s2, eps)
        else:
            row = q[s2]
            vs = valid[s2]
            if algo == "q_learning":
                best = row[vs[0]]
                for b in vs[1:]:
                    if row[b] > best:
                        best = row[b]
                target = r + best
                a2 = choose(s2, eps)
            elif algo == "sarsa":
                a2 = choose(s2, eps)
                target = r + row[a2]
            else:
                g = vs[0]
                for b in vs[1:]:
                    if row[b] > row[g]:
                        g = b
                share = eps / len(vs)
                target = r + (1.0 - eps) * row[g] + share * sum(row[b] for b in vs)
                a2 = choose(s2, eps)
        q[s][a] += alpha * (target - q[s][a])
        s, a = s2, a2
        if next_mark is not None and t >= next_mark:
            if on_checkpoint is not None:
                records.append(on_checkpoint(as_array(), t, episodes))
            next_mark = marks.pop(0) if marks else None
    return {"q": as_array(), "counts": np.array(cnt), "records": records, "episodes": episodes}
