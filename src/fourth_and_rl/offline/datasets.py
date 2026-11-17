"""Behavior policies and logged datasets in the known-truth drive model.

A logged dataset is a list of fourth-down decisions, each one a transition

    (s, a, c)    state index, action (0 go, 1 field goal, 2 punt), outcome column

in the representation of `rl.envs.DriveModel`: column c < n is the next fourth-down state
of the same drive (reward 0), column c >= n is one of the ways a drive ends, with reward
`model.term_reward[c - n]` (the points of the play minus the solved value of what the
other team then holds, exactly as in M5's DriveEnv). Rewards and "done" flags are
functions of c, so they are not stored.

Every tabular learner only needs the count tensor C[a, s, c] (how often action a in state
s led to column c). For squared-error regression that is not an approximation: the sum of
squared errors over individual transitions equals the count-weighted sum over cells plus
a constant, so fitting on C gives exactly the same minimizer (see `learners`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fourth_and_rl.rl.envs import GO, DriveModel, RowSampler
from fourth_and_rl.rl.evaluate import as_probs, coach_drive_policy

# --------------------------------------------------------------------------------------
# Behavior policies
# --------------------------------------------------------------------------------------


def uniform_policy(model: DriveModel) -> np.ndarray:
    """Uniform over the allowed actions of each state, (n, 3)."""
    return model.allowed / model.allowed.sum(axis=1, keepdims=True)


def coach_policy(model: DriveModel, coach_table: np.ndarray) -> np.ndarray:
    """The coach-like policy on the drive's fourth-down states, (n, 3).

    Mapping: the solved model's coach table holds observed P(go, field goal, punt) per
    (situation, distance bin, yard line), fitted on 2018 to 2023 in cells of (distance
    group 1, 2, 3, 4-5, 6-9, 10+) by 5-yard band with hierarchical smoothing, already
    expanded to the grid. The drive model has no clock or score, so its states take the
    "neutral" situation (index 0). Both long-distance bins (11-15 and 16+) map to the
    "10+" group. Probability on a field goal where the model does not allow one is
    removed and the rest renormalized (`rl.evaluate.coach_drive_policy`).
    """
    return coach_drive_policy(model, coach_table)


def behavior_policy(
    model: DriveModel, coach_table: np.ndarray, epsilon: float = 0.0, go_logit_shift: float = 0.0
) -> np.ndarray:
    """A member of the behavior family: the coach-like policy with `go_logit_shift` added
    to the log-odds of going for it (negative means more conservative than coaches), then
    mixed with uniform random actions at rate `epsilon`."""
    mu = coach_policy(model, coach_table)
    if go_logit_shift:
        w = mu.copy()
        w[:, GO] *= np.exp(go_logit_shift)
        mu = as_probs(w, model.allowed)
    return (1.0 - epsilon) * mu + epsilon * uniform_policy(model)


# --------------------------------------------------------------------------------------
# Logged data
# --------------------------------------------------------------------------------------


@dataclass
class Logged:
    """Transitions in episode order. `first[i]` marks the first decision of a drive.
    The last episode may be cut short by the size limit (`done` is then False on its
    last transition, a timeout). Generated data infer episode starts from `done`; real
    data, where a drive can be cut without ending (end of half, or its next decision
    falls outside the sample), pass them explicitly in `starts`."""

    s: np.ndarray  # (N,) int32
    a: np.ndarray  # (N,) int8
    c: np.ndarray  # (N,) int32
    n_states: int
    starts: np.ndarray | None = None  # (N,) bool, explicit episode starts

    @property
    def size(self) -> int:
        return len(self.s)

    @property
    def done(self) -> np.ndarray:
        return self.c >= self.n_states

    @property
    def first(self) -> np.ndarray:
        if self.starts is not None:
            return self.starts
        out = np.ones(self.size, dtype=bool)
        out[1:] = self.done[:-1]
        return out

    @property
    def episode(self) -> np.ndarray:
        """Episode number of every transition (0, 1, ...)."""
        return np.cumsum(self.first) - 1

    def head(self, n: int) -> Logged:
        """The first n transitions (a dataset of size n)."""
        st = None if self.starts is None else self.starts[:n]
        return Logged(self.s[:n], self.a[:n], self.c[:n], self.n_states, st)

    def take_episodes(self, ids: np.ndarray) -> Logged:
        """The episodes `ids` (with repeats, in that order): a bootstrap resample."""
        ep = self.episode
        starts = np.searchsorted(ep, np.arange(ep[-1] + 2))
        idx = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in ids])
        st = None if self.starts is None else self.starts[idx]
        return Logged(self.s[idx], self.a[idx], self.c[idx], self.n_states, st)


def generate(
    model: DriveModel, behavior: np.ndarray, n_decisions: int, seed: int, n_parallel: int = 4096
) -> Logged:
    """Log `n_decisions` fourth-down decisions of drives played by `behavior` (n, 3).

    Drives start at the model's first-fourth-down distribution (`model.start`) and are
    sampled exactly from `model.dist`, as in M5. Many drives run in parallel; each gets an
    episode number when it starts. Once enough decisions are logged no new drive starts,
    the running ones are finished, and the transitions are sorted by episode, so that the
    first n transitions are complete drives (an i.i.d. sample of drives) plus at most one
    cut short. The result is a deterministic function of `seed`.
    """
    rng = np.random.default_rng(seed)
    behavior = as_probs(behavior, model.allowed)
    act = RowSampler(behavior)
    step = [RowSampler(model.dist[a]) for a in range(3)]
    start = RowSampler(model.start[None, :])
    n = model.n

    def draw_start(k: int) -> np.ndarray:
        return start.sample(np.zeros(k, dtype=np.int64), rng.random(k))

    k = int(min(n_parallel, max(n_decisions, 1)))
    s = draw_start(k)
    ep = np.arange(k, dtype=np.int64)
    t = np.zeros(k, dtype=np.int64)
    next_ep = k
    active = np.ones(k, dtype=bool)
    logged = 0
    rec_s, rec_a, rec_c, rec_e, rec_t = [], [], [], [], []
    while active.any():
        idx = np.nonzero(active)[0]
        a = act.sample(s[idx], rng.random(len(idx)))
        u = rng.random(len(idx))
        c = np.zeros(len(idx), dtype=np.int64)
        for code in range(3):
            sel = np.nonzero(a == code)[0]
            if len(sel):
                c[sel] = step[code].sample(s[idx[sel]], u[sel])
        rec_s.append(s[idx].copy())
        rec_a.append(a)
        rec_c.append(c)
        rec_e.append(ep[idx].copy())
        rec_t.append(t[idx].copy())
        logged += len(idx)
        done = c >= n
        cont = idx[~done]
        s[cont] = c[~done]
        t[cont] += 1
        fin = idx[done]
        if logged < n_decisions:
            m = len(fin)
            s[fin] = draw_start(m)
            ep[fin] = np.arange(next_ep, next_ep + m)
            t[fin] = 0
            next_ep += m
        else:
            active[fin] = False
    e, tt = np.concatenate(rec_e), np.concatenate(rec_t)
    order = np.lexsort((tt, e))[:n_decisions]
    return Logged(
        np.concatenate(rec_s)[order].astype(np.int32),
        np.concatenate(rec_a)[order].astype(np.int8),
        np.concatenate(rec_c)[order].astype(np.int32),
        n,
    )


def counts(data: Logged, n_cols: int) -> np.ndarray:
    """C[a, s, c]: how often action a in state s led to outcome column c (float64)."""
    out = np.zeros((3, data.n_states, n_cols))
    np.add.at(out, (data.a, data.s, data.c), 1.0)
    return out


def counts_by_size(data: Logged, n_cols: int, sizes: list[int]) -> dict[int, np.ndarray]:
    """Count tensors of the first `size` transitions, for each size (nested datasets)."""
    out: dict[int, np.ndarray] = {}
    acc = np.zeros((3, data.n_states, n_cols))
    done = 0
    for size in sorted(sizes):
        sl = slice(done, size)
        np.add.at(acc, (data.a[sl], data.s[sl], data.c[sl]), 1.0)
        done = size
        out[size] = acc.copy()
    return out


def start_states(data: Logged) -> np.ndarray:
    """Empirical distribution of the first decision of a drive, (n,)."""
    out = np.bincount(data.s[data.first], minlength=data.n_states).astype(float)
    return out / max(out.sum(), 1.0)
