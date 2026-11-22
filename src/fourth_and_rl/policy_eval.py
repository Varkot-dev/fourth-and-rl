"""Exact evaluation of fixed stochastic policies in the game MDP.

The solver (`mdp_game.Solution`) computes the best value, with a max over actions in
every state. Replacing that max by an average under a fixed policy gives the exact value
of the policy instead:

    V_pi(t, f, s, e) = sum over a of pi(a | t, f, s, e) * Q_pi(t, f, s, e, a)

where Q_pi is the same one-step backup the solver uses, read from V_pi at earlier ticks.
Every action uses at least one tick, so one sweep t = 1, 2, ..., n_ticks evaluates the
policy exactly, like the solver.

Two teams
---------
A `PolicyPair` evaluates team A playing one policy against team B playing another. Each
team is a `RingSolution` holding its own value when it has the ball, and the two are
linked the way `mdp_team.link` links two teams (`Solution.other`), so every change of
possession reads the other team's values. With one policy for both teams the game is
symmetric and one self-linked solution holds both sides, as in the solver.

Ring buffer
-----------
A backup at tick t reads values at most `RingSolution.lag` ticks back (the longest play,
the clock-burning runoff, the punt), plus the halftime slice. `RingSolution` keeps only
those slices, so evaluating many policies in one pass needs a few hundred megabytes per
team instead of the 1.6 GB of a full value array.

Post-decisions
--------------
Timeouts and the try after a touchdown follow `post`: "model" (the default, the solver's
own choices made with each team's own values) or "coach" (the simulator's coach rules,
which do not depend on values; see `mdp_game.Solution`).

Policies
--------
A policy is a callable `policy(t, q) -> {action code: probability array}` for the slice at
tick t, where q is the team's own action values at t (`Solution.backup`). Probability
arrays broadcast to the slice shape (nf, n_score, E1, E2). `CoachPolicy` is the simulator's
coach-like policy (`Simulator.coach_actions`) written as probabilities; its `with_fourth`
method replaces the fourth-down choice by any other distribution.
"""

from __future__ import annotations

import resource
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from fourth_and_rl import mdp_game as mg
from fourth_and_rl.mdp import coach_situation

Probs = dict[int, np.ndarray]
Policy = Callable[[int, dict[int, np.ndarray]], Probs]
FOURTH_CHOICES = ("go", "field_goal", "punt")


# --------------------------------------------------------------------------------------
# Ring-buffer solution
# --------------------------------------------------------------------------------------


def clock_lag(clock: mg.ClockSpec) -> int:
    """How many ticks back a backup can read: the longest play-length histogram, the
    clock-burning runoff and the punt, and the tick before (timeouts)."""
    hists = [clock.neutral_hist, clock.hurry_hist]
    if clock.hurry_hist_late is not None:
        hists.append(clock.hurry_hist_late)
    return max(max(len(h) for h in hists), clock.burn_ticks, clock.punt_ticks, 1)


class RingSolution(mg.Solution):
    """A `Solution` whose values live in a dict {tick: slice}, keeping only the slices a
    later backup can still read (the last `lag` ticks, halftime and the newest tick).

    `put(t, v)` stores the slice at t; the tick-0 slice (the final score) is there from
    the start. Everything else (`backup`, `post`, `_aug_at`, `other`) is the solver's."""

    def __init__(
        self, model: mg.GameModel, post: str = "model", coach: dict[str, Any] | None = None
    ) -> None:
        super().__init__(model, v={}, pol={}, post=post, coach=coach)
        self.lag = clock_lag(model.clock)
        self.buf: dict[int, np.ndarray] = {}
        e = model.extra_shape(model.phase(0))
        self.buf[0] = np.ascontiguousarray(
            np.broadcast_to(self.terminal[None, :, None, None], (self.grid.nf, model.n_score, *e))
        ).astype(model.dtype)
        self.newest = 0

    def native(self, t: int) -> np.ndarray:
        try:
            return self.buf[t]
        except KeyError:
            raise KeyError(f"tick {t} is no longer (or not yet) in the ring buffer") from None

    def put(self, t: int, v: np.ndarray) -> None:
        self.buf[t] = np.ascontiguousarray(v, dtype=self.model.dtype)
        self.newest = max(self.newest, t)
        old = t - self.lag - 1
        if old >= 0 and old != self.model.clock.th and old in self.buf:
            del self.buf[old]
            for cache in (self._posts, self._bases):
                for key in [k for k in cache if k[0] == old]:
                    del cache[key]
        self._q_cache = None

    def kickoff_value(self, t: int | None = None) -> np.ndarray:
        """Value of receiving a kickoff at tick t (default: the opening kickoff), shape
        (n_score, E1, E2)."""
        t = self.model.clock.n_ticks if t is None else t
        return self._kicks(self.native(t))[0]


# --------------------------------------------------------------------------------------
# Mixing action values by a policy
# --------------------------------------------------------------------------------------


def mix(
    q: dict[int, np.ndarray], probs: Probs, shape: tuple[int, ...], over: np.ndarray | None,
    dtype: Any, check: bool = True,
) -> np.ndarray:  # fmt: skip
    """sum over a of probs[a] * q[a], in the model's dtype, adding terms in action-code
    order, and 1 where the game is over (the leader kneels it out). A zero probability
    never touches a disallowed (-inf) action. For a policy that puts all its mass on one
    action this is exactly that action's values, so a deterministic policy reproduces the
    solver bit for bit. Raises if the policy puts mass on an action that is not allowed,
    or (with `check`) if its probabilities do not sum to 1."""
    v = None
    total = np.zeros(shape, dtype=np.float64) if check else None
    for a in sorted(probs):
        p = np.asarray(probs[a])
        if not p.any():
            continue
        if a not in q:
            if over is None or ((np.broadcast_to(p, shape) > 0) & ~over).any():
                raise ValueError(f"policy puts mass on {mg.ACTION_NAMES[a]}, which is not "
                                 "available in this phase")  # fmt: skip
            continue
        pb = np.broadcast_to(p, shape).astype(dtype)
        live = pb > 0
        if over is not None:
            live = live & ~over
        if (live & ~np.isfinite(q[a])).any():
            raise ValueError(f"policy puts mass on {mg.ACTION_NAMES[a]} where it is not allowed")
        term = np.where(live, q[a], 0.0).astype(dtype) * pb
        v = term if v is None else v + term
        if total is not None:
            total += np.where(live, pb, 0.0)
    if total is not None:
        need = np.ones(shape, dtype=bool) if over is None else ~over
        if np.abs(total[need] - 1.0).max(initial=0.0) > 1e-5:
            raise ValueError("policy probabilities do not sum to 1")
    if v is None:
        v = np.zeros(shape, dtype=dtype)
    if over is not None:
        v = np.where(over, np.asarray(1.0, dtype=dtype), v)
    return np.asarray(v, dtype=dtype)


# --------------------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------------------


@dataclass
class CoachSlice:
    """The coach policy at one tick, over (field state, score); all arrays (nf, n_score).

    base     mode code of a play (neutral, protect or chase), used on downs 1 to 3 and when
             going for it on fourth down
    p        (nf, n_score, 3) fourth-down probabilities of (go, field goal, punt), after a
             disallowed kick's share is moved to going for it, as the simulator does
    forced   action code forced by the end-of-half rules, or -1
    fourth   fourth down
    """

    base: np.ndarray
    p: np.ndarray
    forced: np.ndarray
    fourth: np.ndarray

    @property
    def free_fourth(self) -> np.ndarray:
        """Fourth downs whose choice is not forced: where another policy may differ."""
        return self.fourth & (self.forced < 0)


class CoachPolicy:
    """`Simulator.coach_actions` as exact probabilities, one slice at a time.

    Fourth downs: (go, field goal, punt) from the coach table by situation, a disallowed
    kick's share moved to going for it. Other downs: a play in the coach's mode (protect a
    lead and chase a deficit in the last 5:00). Then the end-of-half overrides, in the
    simulator's order: kick when a field goal ties or wins as time expires (or at the end
    of the first half), go for it when trailing by more than a field goal late, and kick
    or go in the last minute when a field goal is enough."""

    def __init__(self, model: mg.GameModel, coach: dict[str, Any]) -> None:
        self.model = model
        self.coach = coach
        g = model.kern.grid
        self.f4 = slice((g.n_down - 1) * g.n_dist * g.n_yl, g.nf)
        self._cache: tuple[int, CoachSlice] | None = None

    def setting(self, name: str) -> Any:
        return self.coach[name] if name in self.coach else mg.COACH_DEFAULTS[name]

    def slice(self, t: int) -> CoachSlice:
        if self._cache is not None and self._cache[0] == t:
            return self._cache[1]
        m, c = self.model, self.model.clock
        g, k = m.kern.grid, m.kern
        f = np.arange(g.nf)[:, None]
        s = np.arange(-m.score_max, m.score_max + 1)[None, :]
        f, s = np.broadcast_arrays(f, s)
        down, dbin, yl = g.unpack(f)
        sec = np.full(f.shape, t * c.tick_seconds)
        half_sec = np.full(f.shape, (t - c.th) * c.tick_seconds if t > c.th else t * c.tick_seconds)
        end = (t <= c.endgame_ticks) and c.endgame_ticks > 0
        warn = c.warning_tick if c.warning_tick is not None else 0
        chase = (s < 0) | ((s == 0) & (t <= warn))
        lead_mode = np.where(end & (s > 0), mg.A_PROTECT, mg.A_NEUTRAL)
        base = np.where(end & chase, mg.A_CHASE, lead_mode)
        fourth = down == g.n_down
        sit = coach_situation(sec, half_sec, s)
        p = np.asarray(self.coach["table"], dtype=np.float64)[sit, dbin, yl - 1]
        pgo, pfg = p[..., 0], p[..., 1]
        ppu = 1.0 - pgo - pfg  # the simulator's choice index 2 or more is a punt
        okf, okp = k.fg_ok[f], k.punt_ok[f]
        pgo = pgo + np.where(okf, 0.0, pfg) + np.where(okp, 0.0, ppu)
        pfg = np.where(okf, pfg, 0.0)
        ppu = np.where(okp, ppu, 0.0)
        p3 = np.stack([pgo, pfg, ppu], axis=-1)
        p3 = np.where(fourth[..., None], p3, 0.0)
        forced = np.full(f.shape, -1, dtype=np.int64)
        in_range = (yl <= self.setting("end_fg_max_yardline")) & okf
        ticks = self.setting("end_fg_ticks")
        last_game = (t <= ticks) & (s <= 0) & (s >= -m.fg_points)
        last_half = (t > c.th) and (t - c.th <= ticks)
        forced = np.where((last_game | last_half) & in_range, mg.A_FG, forced)
        late4 = fourth & end & (t <= warn)
        forced = np.where(late4 & (s < -m.fg_points), mg.A_CHASE, forced)
        lm = late4 & (t <= warn // 2) & (s < 0) & (s >= -m.fg_points)
        forced = np.where(lm, np.where(in_range, mg.A_FG, mg.A_CHASE), forced)
        out = CoachSlice(base=base, p=p3, forced=forced, fourth=fourth)
        self._cache = (t, out)
        return out

    def probs(self, t: int, fourth: np.ndarray | None = None) -> Probs:
        """Action probabilities at tick t. `fourth`, if given, replaces the coach's
        (go, field goal, punt) distribution at unforced fourth downs: an array
        (n_fourth, n_score, E1, E2, 3) over the fourth-down rows `self.f4` (or anything
        that broadcasts to it). Arrays are (nf, n_score, 1, 1), or the full slice shape
        when `fourth` has extra-state axes."""
        cs = self.slice(t)
        ex = (slice(None), slice(None), None, None)
        p3 = cs.p[ex]  # (nf, S, 1, 1, 3)
        if fourth is not None:
            fourth = np.asarray(fourth, dtype=np.float64)
            e = fourth.shape[2:4] if fourth.ndim == 5 else (1, 1)
            full = np.broadcast_to(p3, (*p3.shape[:2], *e, 3)).copy()
            free = cs.free_fourth[self.f4][ex]
            full[self.f4] = np.where(free[..., None], fourth, full[self.f4])
            p3 = full
        base = cs.base[ex]
        out: Probs = {}
        for a in (mg.A_NEUTRAL, mg.A_PROTECT, mg.A_CHASE):
            sel = base == a
            if sel.any():
                go = np.where(cs.fourth[ex], p3[..., 0], 1.0)
                out[a] = np.where(sel, go, 0.0)
        out[mg.A_FG] = p3[..., 1].copy()
        out[mg.A_PUNT] = p3[..., 2].copy()
        forced = cs.forced[ex]
        if (forced >= 0).any():
            shape = np.broadcast_shapes(*(x.shape for x in out.values()))
            for a in list(out):
                out[a] = np.where(forced >= 0, 0.0, np.broadcast_to(out[a], shape))
            for a in (mg.A_FG, mg.A_CHASE):
                hit = forced == a
                if hit.any():
                    cur = out.get(a, np.zeros(shape))
                    out[a] = np.where(hit, 1.0, cur)
        return out

    def __call__(self, t: int, q: dict[int, np.ndarray] | None = None) -> Probs:
        return self.probs(t)

    def with_fourth(self, fn: Callable[[int, dict[int, np.ndarray]], np.ndarray]) -> Policy:
        """The coach everywhere except at unforced fourth downs, where `fn(t, q)` gives the
        (go, field goal, punt) distribution (see `probs`)."""

        def policy(t: int, q: dict[int, np.ndarray]) -> Probs:
            return self.probs(t, fn(t, q))

        return policy


def fourth_q(
    q: dict[int, np.ndarray], cs: CoachSlice, f4: slice, shape: tuple[int, ...]
) -> np.ndarray:
    """Action values of (go in the coach's mode, field goal, punt) at the fourth-down rows,
    shape (n_fourth, n_score, E1, E2, 3); -inf where an action is not allowed."""

    def qa(a: int) -> np.ndarray:
        return q[a][f4] if a in q else np.full((f4.stop - f4.start, *shape[1:]), -np.inf)

    base = cs.base[f4][:, :, None, None]
    go = np.where(
        base == mg.A_NEUTRAL,
        qa(mg.A_NEUTRAL),
        np.where(base == mg.A_PROTECT, qa(mg.A_PROTECT), qa(mg.A_CHASE)),
    )
    return np.stack([go, qa(mg.A_FG), qa(mg.A_PUNT)], axis=-1)


def array_policy(pol: dict[str, np.ndarray], model: mg.GameModel) -> Policy:
    """A deterministic policy from solver-style action-code arrays (e.g. `Solution.pol`).
    Codes A_OVER are left to the kneel-out mask."""

    def policy(t: int, q: dict[int, np.ndarray]) -> Probs:
        ph = model.phase(t)
        codes = np.asarray(pol[ph][t - model.phase_range(ph)[0]])
        return {int(a): (codes == a) for a in np.unique(codes)}

    return policy


# --------------------------------------------------------------------------------------
# Evaluating a pair of policies
# --------------------------------------------------------------------------------------


class PolicyPair:
    """Team A playing `policy_a` against team B playing `policy_b`, evaluated exactly.

    `policy_b=None` means both teams play `policy_a`: one self-linked solution then holds
    both sides. `post` and `coach` set the post-decisions (see `mdp_game.Solution`).
    `record` lists (team, GameState) pairs whose values are kept as the pass goes; the
    state is described from the side of the team with the ball, and team is "A" or "B".
    `keep_slices` keeps every slice of team A (for tests on small games).
    """

    def __init__(
        self,
        model: mg.GameModel,
        policy_a: Policy,
        policy_b: Policy | None = None,
        record: list[tuple[str, mg.GameState]] | None = None,
        keep_slices: bool = False,
        check: bool = True,
        post: str = "model",
        coach: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.symmetric = policy_b is None
        a = RingSolution(model, post=post, coach=coach)
        if self.symmetric:
            self.sols = {"A": a}
            self.policies = {"A": policy_a}
        else:
            b = RingSolution(model, post=post, coach=coach)
            a._other, b._other = b, a
            self.sols = {"A": a, "B": b}
            self.policies = {"A": policy_a, "B": policy_b}
        self.check = check
        self.records: dict[tuple[str, int], list[tuple[int, tuple[int, ...]]]] = {}
        self.recorded: list[float] = [float("nan")] * len(record or [])
        for i, (team, state) in enumerate(record or []):
            team = "A" if self.symmetric else team
            ph, ix = a.index(state)
            t = ix[0] + model.phase_range(ph)[0]
            self.records.setdefault((team, t), []).append((i, ix[1:]))
        self.slices: dict[int, np.ndarray] | None = {} if keep_slices else None
        if self.slices is not None:
            self.slices[0] = a.native(0).copy()
        self._q: dict[str, dict[int, np.ndarray]] = {}

    @property
    def a(self) -> RingSolution:
        return self.sols["A"]

    @property
    def b(self) -> RingSolution:
        return self.sols["B"] if not self.symmetric else self.sols["A"]

    def step(self, t: int) -> None:
        """Evaluate tick t for both teams (every earlier tick must be done)."""
        m = self.model
        shape = (m.kern.grid.nf, m.n_score, *m.extra_shape(m.phase(t)))
        new = {}
        for team, sol in self.sols.items():
            q, over = sol.backup(t)
            self._q[team] = q
            probs = self.policies[team](t, q)
            new[team] = mix(q, probs, shape, over, np.dtype(m.dtype), self.check)
        for team, sol in self.sols.items():
            sol.put(t, new[team])
            for i, ix in self.records.get((team, t), []):
                self.recorded[i] = float(new[team][ix])
        if self.slices is not None:
            self.slices[t] = new["A"].copy()

    def last_q(self, team: str = "A") -> dict[int, np.ndarray]:
        """Team's own action values at the tick just evaluated."""
        return self._q[team]

    def start_value(self) -> float:
        """Win probability of team A at the opening kickoff, tied, with a coin flip for
        which team receives it (the receiver kicks off the second half)."""
        s0 = self.model.score_max
        ka = float(self.a.kickoff_value()[s0, 0, 0])
        kb = float(self.b.kickoff_value()[s0, 0, 0])
        return 0.5 * ka + 0.5 * (1.0 - kb)

    def value(self, state: mg.GameState) -> float:
        """Value of a state of the kept slices (needs keep_slices=True)."""
        if self.slices is None:
            raise ValueError("slices were not kept; use record= or keep_slices=True")
        ph, ix = self.a.index(state)
        return float(self.slices[ix[0] + self.model.phase_range(ph)[0]][ix[1:]])


def run(
    pairs: list[PolicyPair],
    before: Callable[[int], None] | None = None,
    after: Callable[[int], None] | None = None,
    progress: bool = False,
) -> dict[str, float]:
    """One backward sweep over every tick for all pairs (they must share one model).
    `before(t)` and `after(t)` run around each tick. Returns seconds and peak memory."""
    n = pairs[0].model.clock.n_ticks
    t0 = time.time()
    for t in range(1, n + 1):
        if before is not None:
            before(t)
        for p in pairs:
            p.step(t)
        if after is not None:
            after(t)
        if progress and (t % 60 == 0 or t == n):
            print(f"  t = {t:4d} / {n}   {time.time() - t0:7.1f} s", flush=True)
    return {"seconds": time.time() - t0, "peak_rss_gb": peak_rss_gb()}


def peak_rss_gb() -> float:
    """Peak resident memory of this process so far, in GB (Linux reports kilobytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def evaluate(
    model: mg.GameModel,
    policy_a: Policy,
    policy_b: Policy | None = None,
    record: list[tuple[str, mg.GameState]] | None = None,
    keep_slices: bool = False,
    progress: bool = False,
    post: str = "model",
    coach: dict[str, Any] | None = None,
) -> tuple[PolicyPair, dict[str, float]]:
    """Evaluate one pair of policies over the whole game. Returns the pair (start value,
    recorded values, kept slices) and the time and memory of the pass."""
    pair = PolicyPair(model, policy_a, policy_b, record, keep_slices, post=post, coach=coach)
    stats = run([pair], progress=progress)
    return pair, stats
