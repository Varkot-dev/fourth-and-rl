"""M4 stage 2: the whole game as a finite zero-sum game, solved exactly.

Value
-----
V(t, f, s, e) is the probability that the team with the ball wins, where t is the time
left in ticks, f the field state (down, distance, yard line), s the score differential
from that team's side (clipped), and e the extra state of the phase:

  phase "end"    t <= endgame_ticks: e = (offense timeouts, defense timeouts)
  phase "mid"    rest of the second half: no extra state
  phase "first"  first half: e = (does the team with the ball receive the second-half
                 kickoff?)

The teams are identical, so the other team's value in any state is 1 minus ours after
swapping roles: `opp(X)[s, e] = 1 - X[-s, swapped e]`. That one array holds both teams.
(`mdp_team.py` solves two different teams: each `Solution` then reads the other team's
values from a linked `Solution`, `Solution.other`; by default a solution is its own other.)

Bellman equation (one tick = `tick_seconds`):

    V(t, f, s, e) = max over actions a of
        sum over outcomes o of P(o | f, a) * value_of(o, at time t - duration(o))

Every action uses at least one tick, so V at time t only needs V at earlier-solved times
(smaller t). One sweep t = 1, 2, ..., n_ticks solves the game exactly: backward
induction, no iteration to convergence. At t = 0 the value is 1, 0.5 or 0 by the score.

Clock (docs/research/endgame_clock.md)
--------------------------------------
* A play after which the clock is stopped (incomplete, out of bounds, score, change of
  possession) costs 1 tick; a punt `punt_ticks`.
* End phase. The offense picks a mode with the play. "protect" (allowed unless trailing):
  play as leading teams do, and after an in-bounds play let the clock run, 1 + `burn`
  ticks. "chase" (allowed unless leading): play as trailing teams do and hurry, the
  empirical histogram of ticks, faster inside the two-minute warning. "neutral": the
  generic play at normal tempo. A neutral or chasing offense may spend its own timeout
  to make an in-bounds play cost 1 tick. After any in-bounds play the defense may spend
  a timeout to the same effect; it does so when that is worse for the offense (a min),
  or with empirical rates if `defense_timeout == "empirical"`. A runoff that would cross
  the two-minute warning stops at it.
* If the team in front has a down to spare and at most T(defense timeouts, down)
  seconds left, it kneels the game out: the state is terminal with value 1.
* Before the end phase there are no timeouts or modes: an in-bounds play lasts a number
  of ticks drawn from the empirical histogram.

The module also holds the simulator (`simulate`, `step`), which samples from exactly the
same transition model, and the loader API (`load_solution`, `value`, `policy`).
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import scipy.sparse as sp

from fourth_and_rl.mdp import KERNEL_KEYS, FieldKernels, coach_situation
from fourth_and_rl.paths import DATA

A_NEUTRAL, A_PROTECT, A_CHASE, A_FG, A_PUNT, A_KNEEL, A_OVER = range(7)
ACTION_NAMES = ("play", "play_protect", "play_chase", "field_goal", "punt", "kneel", "game_over")
_MODE_CODE = {"neutral": A_NEUTRAL, "protect": A_PROTECT, "chase": A_CHASE}
K_KEEP, K_TURN, K_TD, K_SAFETY, K_OPPTD, K_FG = range(6)
DRIVE_RESULTS = (
    "touchdown",
    "field_goal",
    "missed_field_goal",
    "punt",
    "turnover",
    "opp_touchdown",
    "downs",
    "safety",
    "end_of_half",
)
SOLUTION_DIR = DATA / "mdp"

# Coach end-of-half settings for coach tables saved before these keys were written
# (scripts/solve_mdp.py now copies all four from configs/mdp.yaml, section `coach`, and a
# missing key there raises). This is the one place the old defaults live; they equal the
# config values the keys were introduced with.
COACH_DEFAULTS: dict[str, Any] = {
    "end_fg_max_yardline": 42,
    "end_fg_ticks": 2,
    "two_point_seconds": 900,
    "two_point_leads": (-5, -2, 1, 5),
}


# --------------------------------------------------------------------------------------
# Model description
# --------------------------------------------------------------------------------------


@dataclass
class ClockSpec:
    """Clock rules, all in ticks. See the module docstring."""

    n_ticks: int
    tick_seconds: int = 5
    endgame_ticks: int = 0
    half_tick: int | None = None  # ticks left at halftime; None = no first half
    warning_tick: int | None = None
    burn_ticks: int = 9
    punt_ticks: int = 2
    n_timeouts: int = 0
    neutral_hist: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    hurry_hist: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    hurry_hist_late: np.ndarray | None = None  # inside the warning; None = same as hurry_hist
    kneel_ticks: np.ndarray | None = None  # [defense timeouts, down - 1] -> ticks, 0 = never
    shares_leading: np.ndarray | None = None
    shares_trailing: np.ndarray | None = None
    defense_timeout: str = "optimal"
    def_rate_before: np.ndarray | None = None
    def_rate_after: np.ndarray | None = None

    @property
    def th(self) -> int:
        return self.n_ticks if self.half_tick is None else self.half_tick

    def late(self, t: int) -> bool:
        """Inside the two-minute warning?"""
        return self.warning_tick is not None and t <= self.warning_tick

    def kernel_key(self, mode: str, t: int) -> str:
        """Which kernel table a play in `mode` at time t uses."""
        if t > self.endgame_ticks or self.endgame_ticks == 0:
            return "neutral"
        if mode == "neutral":
            return "neutral_end"
        return "chase_late" if (mode == "chase" and self.late(t)) else mode

    def tempo_hist(self, mode: str, t: int) -> np.ndarray:
        """Ticks used by an in-bounds play when the offense does not burn the clock."""
        if mode == "chase":
            late = self.hurry_hist_late is not None and self.late(t)
            return self.hurry_hist_late if late else self.hurry_hist
        return self.neutral_hist


def kneel_threshold_ticks(
    n_timeouts: int, n_down: int, tick: int, kneel_seconds: float, play_clock: float, cap: float
) -> np.ndarray:
    """Closed-form kneel rule: T(n, m) = play_clock * max(m - n, 0) + kneel_seconds * m,
    capped at the two-minute warning, in whole ticks. m = kneels available on that down."""
    out = np.zeros((n_timeouts + 1, n_down), dtype=np.int64)
    for n in range(n_timeouts + 1):
        for down in range(1, n_down):
            m = n_down - down
            t = min(play_clock * max(m - n, 0) + kneel_seconds * m, cap)
            out[n, down - 1] = int(t // tick)
    return out


def _normalized(shares: list[float]) -> np.ndarray:
    """Published shares are rounded and may sum to 1.001; probabilities must sum to 1."""
    x = np.asarray(shares, dtype=float)
    return x / x.sum()


def clock_from_config(cfg: dict, tables: dict[str, Any], horizon: str = "game") -> ClockSpec:
    """Build the clock. `horizon` is "game" (all 60 minutes) or "endgame" (last 5:00 only)."""
    c = cfg["clock"]
    tick = c["tick_seconds"]
    te = c["endgame_seconds"] // tick
    whole = horizon == "game"
    return ClockSpec(
        n_ticks=c["game_seconds"] // tick if whole else te,
        tick_seconds=tick,
        endgame_ticks=te,
        half_tick=c["half_seconds"] // tick if whole else None,
        warning_tick=c["warning_seconds"] // tick,
        burn_ticks=1 + c["burn_runoff_ticks"],
        punt_ticks=c["punt_ticks"],
        n_timeouts=c["n_timeouts"],
        neutral_hist=np.asarray(tables["neutral_hist"], dtype=float),
        hurry_hist=np.asarray(tables["hurry_hist"], dtype=float),
        hurry_hist_late=np.asarray(tables["hurry_hist_late"], dtype=float),
        kneel_ticks=kneel_threshold_ticks(
            c["n_timeouts"],
            4,
            tick,
            c["kneel_seconds"],
            c["play_clock_seconds"],
            c["warning_seconds"],
        ),
        shares_leading=_normalized(c["timeout_shares_leading"]),
        shares_trailing=_normalized(c["timeout_shares_trailing"]),
        defense_timeout=c["defense_timeout"],
        def_rate_before=np.asarray(c["defense_timeout_rate_before_warning"], dtype=float),
        def_rate_after=np.asarray(c["defense_timeout_rate_after_warning"], dtype=float),
    )


@dataclass
class GameModel:
    """Everything the solver and the simulator need."""

    kern: FieldKernels
    clock: ClockSpec
    score_max: int = 21
    td_points: int = 6
    fg_points: int = 3
    safety_points: int = 2
    p_xp: float = 0.945
    p_two: float = 0.49
    two_point_choice: bool = True
    dtype: str = "float32"

    @property
    def n_score(self) -> int:
        return 2 * self.score_max + 1

    def extra_shape(self, phase: str) -> tuple[int, int]:
        if phase == "end":
            n = self.clock.n_timeouts + 1 if self.clock.endgame_ticks > 0 else 1
            return (n, n)
        return (2, 1) if phase == "first" else (1, 1)

    def phase(self, t: int) -> str:
        if t <= self.clock.endgame_ticks:
            return "end"
        return "mid" if t <= self.clock.th else "first"

    def phase_range(self, phase: str) -> tuple[int, int]:
        """(first tick, last tick) of a phase; empty if first > last."""
        c = self.clock
        if phase == "end":
            return 0, c.endgame_ticks
        if phase == "mid":
            return c.endgame_ticks + 1, c.th
        return c.th + 1, c.n_ticks

    def n_states(self) -> dict[str, int]:
        out = {}
        for ph in ("end", "mid", "first"):
            lo, hi = self.phase_range(ph)
            e = self.extra_shape(ph)
            out[ph] = max(hi - lo + 1, 0) * self.kern.grid.nf * self.n_score * e[0] * e[1]
        out["total"] = sum(out.values())
        return out


def model_from_config(
    cfg: dict, kern: FieldKernels, tables: dict[str, Any], models: Any, horizon: str = "game"
) -> GameModel:
    s = cfg["scoring"]
    return GameModel(
        kern=kern,
        clock=clock_from_config(cfg, tables, horizon),
        score_max=cfg["score_max"],
        td_points=s["touchdown"],
        fg_points=s["field_goal"],
        safety_points=s["safety"],
        p_xp=models.p_extra_point(),
        p_two=models.p_two_point(),
        two_point_choice=s["two_point_choice"],
        dtype=cfg["solve"]["dtype"],
    )


class _Post(NamedTuple):
    """Values at one moment, in one phase's shape, used to value the outcomes of a play."""

    v: np.ndarray  # (nf, S, E1, E2) value of having the ball in each field state
    ko: np.ndarray  # (S, E1, E2) value to a team about to receive a kickoff
    td: np.ndarray  # value to a team that has just scored a touchdown (score before the try)
    two: np.ndarray  # bool: the two-point try is the better choice
    fg: np.ndarray  # value to a team that has just made a field goal (score before it)
    safety: np.ndarray  # value to a team that has just conceded a safety


# --------------------------------------------------------------------------------------
# Solution: arrays plus the backup operator
# --------------------------------------------------------------------------------------


class Solution:
    """Value and policy arrays of a solved (or partly solved) game, and the Bellman backup.

    v[phase] has shape (ticks in phase, nf, n_score, E1, E2); pol[phase] the same, uint8
    codes into ACTION_NAMES. On fourth down "play", "play_protect" and "play_chase" mean
    going for it.

    `post` sets who makes the decisions that follow a play, as `Simulator`'s argument of
    the same name does. "model" (the default): the offense takes its own timeout when that
    is worth more to it, the defense takes one when that is worth less to the offense (or
    at the measured rates if `clock.defense_timeout` is "empirical"), and the scoring team
    takes the better try, all judged by this solution's own values. "coach": the
    simulator's coach rules, which do not depend on any values: a chasing offense spends
    its own timeout after an in-bounds play inside the two-minute warning, the defense
    spends one at the measured rates when the offense leads, and the scorer goes for two
    only in the last `two_point_seconds` at the leads in `two_point_leads` (read from
    `coach`, the coach table, or `COACH_DEFAULTS`). With "coach" the backup is linear in
    the values, which is what fixed-policy evaluation and the forward occupancy pass of
    `policy_eval` need.
    """

    def __init__(
        self,
        model: GameModel,
        v: dict[str, np.ndarray] | None = None,
        pol: dict[str, np.ndarray] | None = None,
        meta: dict | None = None,
        post: str = "model",
        coach: dict[str, Any] | None = None,
    ) -> None:
        if post not in ("model", "coach"):
            raise ValueError(f"post must be 'model' or 'coach', not {post!r}")
        self.model = model
        self.meta = meta or {}
        self.post_mode = post
        self.coach_post: dict[str, Any] | None = None
        if post == "coach":
            src = coach or {}
            self.coach_post = {
                k: src[k] if k in src else COACH_DEFAULTS[k]
                for k in ("two_point_seconds", "two_point_leads")
            }
        self.grid = model.kern.grid
        self.f10 = self.grid.first10()[1:]
        m, g = model, self.grid
        if v is None:
            v, pol = {}, {}
            for ph in ("end", "mid", "first"):
                lo, hi = m.phase_range(ph)
                if hi >= lo:
                    shape = (hi - lo + 1, g.nf, m.n_score, *m.extra_shape(ph))
                    v[ph] = np.zeros(shape, dtype=m.dtype)
                    pol[ph] = np.zeros(shape, dtype=np.uint8)
        self.v, self.pol = v, pol
        self._posts: dict[tuple[int, str], _Post] = {}
        self._bases: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._other: Solution | None = None
        dt = np.dtype(m.dtype)
        self._k_all = {k: x.astype(dt) for k, x in m.kern.k_all.items()}
        self._k_in = {k: x.astype(dt) for k, x in m.kern.k_in.items()}
        self._k_punt = m.kern.k_punt.astype(dt)
        s = np.arange(-m.score_max, m.score_max + 1)
        self.scores = s
        self.terminal = np.where(s > 0, 1.0, np.where(s < 0, 0.0, 0.5)).astype(dt)
        self._weights = self._boundary_weights()
        self._q_cache: tuple[int, dict[int, np.ndarray]] | None = None

    # ---- the other team ----
    @property
    def other(self) -> Solution:
        """The solution that holds the other team's values. Identical teams: `self`, and
        `opp` of our own arrays is the other team's value. `mdp_team.link` points two
        solutions of different teams at each other."""
        return self if self._other is None else self._other

    # ---- small helpers ----
    def _boundary_weights(self) -> np.ndarray:
        """w[s, a, b] = P(offense holds a timeouts, defense b) on entering the end phase."""
        m, c = self.model, self.model.clock
        n = m.extra_shape("end")[0]
        if n == 1 or c.shares_leading is None:
            return np.full((m.n_score, n, n), 1.0 / (n * n))
        lead, trail = c.shares_leading, c.shares_trailing
        mean = 0.5 * (lead + trail)
        w = np.empty((m.n_score, n, n))
        w[self.scores > 0] = np.outer(lead, trail)
        w[self.scores < 0] = np.outer(trail, lead)
        w[self.scores == 0] = np.outer(mean, mean)
        return w

    def native(self, t: int) -> np.ndarray:
        ph = self.model.phase(t)
        return self.v[ph][t - self.model.phase_range(ph)[0]]

    def _shift(self, pts: int) -> np.ndarray:
        return np.clip(np.arange(self.model.n_score) + pts, 0, self.model.n_score - 1)

    @staticmethod
    def opp(x: np.ndarray, phase: str) -> np.ndarray:
        """Turn values held from the other team's side into ours (x[..., S, E1, E2])."""
        y = 1.0 - x[..., ::-1, :, :]
        if phase == "end":
            return y.swapaxes(-1, -2)
        return y[..., ::-1, :] if phase == "first" else y

    def next_tick(self, t: int, k: int) -> int:
        """Time after k ticks, stopping at the two-minute warning if the play crosses it."""
        w = self.model.clock.warning_tick
        t2 = max(t - k, 0)
        if w is not None and t <= self.model.clock.endgame_ticks and t > w > t2:
            return w
        return t2

    def _kicks(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Value of receiving a kickoff and a free kick, from values of holding the ball."""
        k = self.model.kern
        ko = np.tensordot(k.kickoff[1:].astype(v.dtype), v[self.f10], axes=(0, 0))
        fk = np.tensordot(k.free_kick[1:].astype(v.dtype), v[self.f10], axes=(0, 0))
        return ko, fk

    def _derive(
        self,
        v: np.ndarray,
        phase: str,
        ko: np.ndarray | None = None,
        opp_kicks: tuple[np.ndarray, np.ndarray] | None = None,
        fk: np.ndarray | None = None,
        t: int | None = None,
    ) -> _Post:
        """Values of the outcomes that end a possession, at one moment. `ko` and `fk` are
        our values of receiving a kickoff and a free kick (computed from `v` if `ko` is
        None; `fk` defaults to `ko`). `opp_kicks` is the other team's (kickoff, free kick)
        values at that moment, held from its side; None means the teams are identical and
        our own are used. `t` is the moment (ticks left); the coach's try rule needs it."""
        m = self.model
        if ko is None:
            ko, fk = self._kicks(v)
        elif fk is None:
            fk = ko
        oko, ofk = (ko, fk) if opp_kicks is None else opp_kicks
        opp_ko = self.opp(oko, phase)
        td0 = opp_ko[self._shift(m.td_points)]
        xp = m.p_xp * opp_ko[self._shift(m.td_points + 1)] + (1.0 - m.p_xp) * td0
        two = m.p_two * opp_ko[self._shift(m.td_points + 2)] + (1.0 - m.p_two) * td0
        if self.post_mode == "coach":
            if t is None:
                raise ValueError("the coach's try rule needs the time of the touchdown")
            cp = self.coach_post
            late = t * m.clock.tick_seconds <= cp["two_point_seconds"]
            lead = np.isin(self.scores + m.td_points, list(cp["two_point_leads"]))
            better = np.broadcast_to(late & lead[:, None, None], xp.shape).copy()
        else:
            better = (two > xp) if m.two_point_choice else np.zeros(xp.shape, dtype=bool)
        return _Post(
            v=v,
            ko=ko,
            td=np.where(better, two, xp).astype(v.dtype),
            two=better,
            fg=opp_ko[self._shift(m.fg_points)],
            safety=self.opp(ofk, phase)[self._shift(-m.safety_points)],
        )

    def _base(self, t: int, phase: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(value of holding the ball, of receiving a kickoff, of receiving a free kick)
        at time t, seen from a play that started in `phase`, for this team only."""
        key = (t, phase)
        if key in self._bases:
            return self._bases[key]
        m = self.model
        own = m.phase(t)
        if own == phase:
            v = self.native(t)
            ko, fk = self._kicks(v)
        elif phase == "mid" and own == "end":
            # Entering the last 5:00: timeouts are drawn from the measured shares.
            avg = np.einsum("fsab,sab->fs", self.native(t), self._weights.astype(m.dtype))
            v = avg[:, :, None, None]
            ko, fk = self._kicks(v)
        elif phase == "first":
            # Halftime: the half ends whatever the field state; the second half starts
            # with a kickoff to the team whose flag says it receives (flag 0: the other
            # team receives, and we hold 1 minus its value).
            ko2 = self._base(m.clock.th, "mid")[1][:, 0, 0]
            oko2 = self.other._base(m.clock.th, "mid")[1][:, 0, 0]
            ht = np.stack([1.0 - oko2[::-1], ko2], axis=1)[:, :, None].astype(m.dtype)
            v = np.broadcast_to(ht, (self.grid.nf, *ht.shape))
            ko = fk = ht
        else:
            raise ValueError(f"time {t} is not reachable from phase {phase}")
        self._bases[key] = (v, ko, fk)
        return v, ko, fk

    def post(self, t: int, phase: str) -> _Post:
        """Values at time t as seen from a play that started in `phase`."""
        key = (t, phase)
        if key in self._posts:
            return self._posts[key]
        v, ko, fk = self._base(t, phase)
        opp_kicks = None if self.other is self else self.other._base(t, phase)[1:]
        p = self._derive(v, phase, ko=ko, opp_kicks=opp_kicks, fk=fk, t=t)
        self._posts[key] = p
        return p

    def _aug(self, p: _Post, phase: str, po: _Post | None = None) -> np.ndarray:
        """Values of every outcome column (see the `mdp` module docstring). `po` is the
        other team's `_Post` at the same moment (None: identical teams, use `p`)."""
        g = self.grid
        po = p if po is None else po
        aug = np.empty((g.na, *p.v.shape[1:]), dtype=p.v.dtype)
        aug[: g.nf] = p.v
        aug[g.nf : g.nf + g.n_yl] = self.opp(po.v[self.f10], phase)
        aug[g.col_td] = p.td
        aug[g.col_safety] = p.safety
        aug[g.col_opp_td] = self.opp(po.td, phase)
        return aug

    def _aug_at(self, t: int, phase: str) -> np.ndarray:
        """`_aug` at time t, with the other team's values from `other`."""
        return self._aug(self.post(t, phase), phase, self.other.post(t, phase))

    def _defense_rate(self, t: int) -> np.ndarray:
        """q[s, b]: probability the defense spends a timeout (empirical mode)."""
        c = self.model.clock
        rate = (
            c.def_rate_after
            if (c.warning_tick is not None and t <= c.warning_tick)
            else (c.def_rate_before)
        )
        q = np.zeros((self.model.n_score, len(rate)))
        q[self.scores > 0] = rate
        return q

    def _running_values(self, t: int, phase: str) -> dict[str, np.ndarray]:
        """U[mode][f, s, e]: value of holding field state f at time t right after a play
        that ended in bounds (clock running), before anyone decides about timeouts."""
        c = self.model.clock
        if phase != "end":
            u = None
            for h, p in enumerate(c.neutral_hist, start=1):
                if p > 0:
                    term = p * self.post(max(t - h, 0), phase).v
                    u = term if u is None else u + term
            return {"neutral": np.asarray(u, dtype=self.model.dtype)}
        v1 = self.native(t - 1)
        coach = self.post_mode == "coach"
        out = {"protect": self.native(self.next_tick(t, c.burn_ticks)).copy()}
        for mode in ("neutral", "chase"):
            u = None
            for h, p in enumerate(c.tempo_hist(mode, t), start=1):
                if p > 0:
                    term = np.dtype(self.model.dtype).type(p) * self.native(self.next_tick(t, h))
                    u = term if u is None else u + term
            u = u.astype(self.model.dtype)
            if not coach:
                # An offense that is not burning clock may spend its own timeout instead.
                u[:, :, 1:, :] = np.maximum(u[:, :, 1:, :], v1[:, :, :-1, :])
            elif mode == "chase" and c.late(t):
                # The coach rule: a chasing offense always stops the clock inside the warning.
                u[:, :, 1:, :] = v1[:, :, :-1, :]
            out[mode] = u
        for u in out.values():
            if c.defense_timeout == "optimal" and not coach:
                u[..., 1:] = np.minimum(u[..., 1:], v1[..., :-1])
            else:
                q = self._defense_rate(t)[:, None, 1:].astype(self.model.dtype)
                u[..., 1:] = q * v1[..., :-1] + (1.0 - q) * u[..., 1:]
        return out

    def backup(self, t: int) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
        """One Bellman backup for time slice t (t >= 1), from already-solved slices.

        Returns ({action code: Q array (nf, S, E1, E2)}, game_over mask or None). Entries
        of actions that are not allowed are -inf.
        """
        m, g, c = self.model, self.grid, self.model.clock
        k = m.kern
        phase = m.phase(t)
        shape = (g.nf, m.n_score, *m.extra_shape(phase))
        p1 = self.post(max(t - 1, 0), phase)
        aug1 = self._aug_at(max(t - 1, 0), phase).reshape(g.na, -1)
        u = self._running_values(t, phase)
        v1 = aug1[: g.nf]
        q: dict[int, np.ndarray] = {}
        lead = np.broadcast_to((self.scores > 0)[:, None, None], shape[1:]).reshape(-1)
        trail = np.broadcast_to((self.scores < 0)[:, None, None], shape[1:]).reshape(-1)
        for mode, um in u.items():
            key = c.kernel_key(mode, t)
            qm = self._k_all[key] @ aug1 + self._k_in[key] @ (um.reshape(g.nf, -1) - v1)
            if phase == "end" and mode == "protect":
                qm[:, trail] = -np.inf
            if phase == "end" and mode == "chase":
                qm[:, lead] = -np.inf
            q[_MODE_CODE[mode]] = qm
        if k.fg_ok.any():
            make = k.fg_make.astype(m.dtype)[:, None]
            qf = make * p1.fg.reshape(1, -1) + (1.0 - make) * aug1[k.fg_miss_col]
            qf[~k.fg_ok] = -np.inf
            q[A_FG] = qf
        if k.punt_ok.any():
            qp = self._k_punt @ self._aug_at(max(t - c.punt_ticks, 0), phase).reshape(g.na, -1)
            qp[~k.punt_ok] = -np.inf
            q[A_PUNT] = qp
        if k.kneel_ok.any():
            um = u["protect" if phase == "end" else "neutral"].reshape(g.nf, -1)
            qk = np.empty_like(v1)
            keep = k.kneel_keep
            qk[keep] = um[k.kneel_col[keep]]
            qk[~keep] = aug1[k.kneel_col[~keep]]
            qk[~k.kneel_ok] = -np.inf
            q[A_KNEEL] = qk
        over = None
        if phase == "end" and c.kneel_ticks is not None and c.endgame_ticks > 0:
            downs, _, _ = g.unpack(np.arange(g.nf))
            thr = c.kneel_ticks[:, downs - 1].T  # (nf, B)
            over = (
                (t <= thr)[:, None, None, :]
                & (self.scores > 0)[None, :, None, None]
                & np.ones(shape, dtype=bool)
            )
        return {a: x.reshape(shape) for a, x in q.items()}, over

    def solve_slice(self, t: int) -> None:
        q, over = self.backup(t)
        codes = list(q)
        v = q[codes[0]].copy()
        pol = np.full(v.shape, codes[0], dtype=np.uint8)
        for a in codes[1:]:
            better = q[a] > v
            v = np.where(better, q[a], v)
            pol[better] = a
        if over is not None:
            v[over] = 1.0
            pol[over] = A_OVER
        ph = self.model.phase(t)
        lo = self.model.phase_range(ph)[0]
        self.v[ph][t - lo] = v
        self.pol[ph][t - lo] = pol

    def q_slice(self, t: int) -> dict[int, np.ndarray]:
        """Action values of every state at time t (cached for the last t asked)."""
        if self._q_cache is None or self._q_cache[0] != t:
            self._q_cache = (t, self.backup(t)[0])
        return self._q_cache[1]

    # ---- public lookups ----
    def index(self, state: GameState) -> tuple[str, tuple[int, ...]]:
        """(phase, index into v[phase]) of a state."""
        m, g = self.model, self.grid
        t = int(round(state.seconds / m.clock.tick_seconds))
        ph = m.phase(t)
        d = min(state.ydstogo, state.yardline_100)
        f = int(g.fidx(state.down, g.dist_bin(d), state.yardline_100))
        s = int(np.clip(state.score, -m.score_max, m.score_max)) + m.score_max
        e = m.extra_shape(ph)
        if ph == "end":
            ex = (min(state.off_timeouts, e[0] - 1), min(state.def_timeouts, e[1] - 1))
        else:
            ex = (int(state.receives_second_half) if ph == "first" else 0, 0)
        return ph, (t - m.phase_range(ph)[0], f, s, *ex)

    def value(self, state: GameState) -> float:
        """Win probability of the team with the ball."""
        ph, ix = self.index(state)
        return float(self.v[ph][ix])

    def policy(self, state: GameState) -> str:
        """Name of the optimal action (see ACTION_NAMES)."""
        ph, ix = self.index(state)
        return ACTION_NAMES[int(self.pol[ph][ix])]

    def q_values(self, state: GameState) -> dict[str, float]:
        """Win probability of each allowed action in a state (recomputes one time slice)."""
        ph, ix = self.index(state)
        t = ix[0] + self.model.phase_range(ph)[0]
        q = self.q_slice(t)
        return {ACTION_NAMES[a]: float(x[ix[1:]]) for a, x in q.items() if np.isfinite(x[ix[1:]])}

    def play_split(self, state: GameState, mode: str | None = None) -> dict[str, Any]:
        """Take apart the value of running a play (on fourth down: going for it).

        Returns the mode used (the best one if `mode` is None), the probability that the
        offense keeps the ball or scores, and the win probability given that it did and
        given that it did not. p * value_if_kept + (1 - p) * value_if_lost = value.
        """
        g, c = self.grid, self.model.clock
        ph, ix = self.index(state)
        t = ix[0] + self.model.phase_range(ph)[0]
        f, col = ix[1], ix[2:]
        if mode is None:
            q = self.q_slice(t)
            codes = [a for a in (A_NEUTRAL, A_PROTECT, A_CHASE) if a in q]
            best = max(codes, key=lambda a: float(q[a][ix[1:]]))
            mode = {v: k for k, v in _MODE_CODE.items()}[best]
        key = c.kernel_key(mode, t)
        aug = self._aug_at(t - 1, ph)[(slice(None), *col)].astype(float)
        u = self._running_values(t, ph)[mode][(slice(None), *col)].astype(float)
        row = self.model.kern.k_all[key][f].toarray().ravel()
        row_in = self.model.kern.k_in[key][f].toarray().ravel()
        val = row * aug
        val[: g.nf] += row_in * (u - aug[: g.nf])
        kept = np.zeros(g.na, dtype=bool)
        kept[: g.nf] = True
        kept[g.col_td] = True
        p = float(row[kept].sum())
        kept_mass, lost_mass = float(val[kept].sum()), float(val[~kept].sum())
        return {
            "mode": mode,
            "p_keep_or_score": p,
            "p_touchdown_on_the_play": float(row[g.col_td]),
            "p_clock_running_given_kept": float(row_in.sum() / max(row[: g.nf].sum(), 1e-12)),
            # Conditional values are undefined (NaN) when their branch cannot happen.
            "value_if_kept": kept_mass / p if p > 0 else float("nan"),
            "value_if_lost": lost_mass / (1.0 - p) if p < 1 else float("nan"),
            "value": float(val.sum()),
        }


class GameState(NamedTuple):
    """A state from the side of the team with the ball. Timeouts matter only in the last
    5:00; `receives_second_half` only in the first half."""

    seconds: int
    score: int
    down: int
    ydstogo: int
    yardline_100: int
    off_timeouts: int = 3
    def_timeouts: int = 3
    receives_second_half: int = 0


def backward_induction(sols: list[Solution], progress: bool = False) -> float:
    """Fill every solution's arrays, one time slice at a time, oldest slice first. All
    solutions share one clock; at each t every solution's slice is solved before any
    t + 1, because a team's backup at t reads the other team's values at earlier times
    (`Solution.other`). Returns the seconds taken."""
    n_ticks = sols[0].model.clock.n_ticks
    t0 = time.time()
    for sol in sols:
        for ph in sol.v:
            if sol.model.phase_range(ph)[0] == 0:
                sol.v[ph][0] = sol.terminal[None, :, None, None]
                sol.pol[ph][0] = A_OVER
    for t in range(1, n_ticks + 1):
        for sol in sols:
            sol.solve_slice(t)
        if progress and (t % 20 == 0 or t == n_ticks):
            print(f"  t = {t:4d} / {n_ticks}   {time.time() - t0:7.1f} s", flush=True)
    return time.time() - t0


def solve(model: GameModel, progress: bool = False) -> Solution:
    """Backward induction over every time slice. Returns the solved game."""
    sol = Solution(model)
    seconds = backward_induction([sol], progress)
    sol.meta.update(
        solve_seconds=seconds,
        peak_memory_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        n_states=model.n_states(),
    )
    return sol


# --------------------------------------------------------------------------------------
# Saving and loading
# --------------------------------------------------------------------------------------


_SOLUTION_FILES = ("model.joblib", "meta.json")


def _is_solution_file(name: str) -> bool:
    """Files `save_solution` owns: anything else in the folder (the cached tables) is kept."""
    return name in _SOLUTION_FILES or (
        name.endswith(".npy") and name.startswith(("value_", "policy_"))
    )


def save_solution(sol: Solution, path: Path | str | None = None) -> Path:
    """Write the solved game to `path` (default `SOLUTION_DIR`), atomically.

    Everything is written to a temporary folder next to the target first; only when every
    array, the model and meta.json are on disk is the old folder swapped out by two
    renames. An interrupted save therefore leaves the previous solution intact (and a
    stray temporary folder), never a mix of old and new arrays. Files in the target that
    are not part of a solution (for example tables.joblib, written by scripts/solve_mdp.py)
    are carried over into the new folder; old value and policy arrays are not, so a phase
    the new model lacks cannot survive from an earlier solve.
    """
    path = Path(path) if path is not None else SOLUTION_DIR
    path.parent.mkdir(parents=True, exist_ok=True)
    tag = f"{os.getpid()}-{time.time_ns()}"
    tmp = path.parent / f".{path.name}.tmp-{tag}"
    old = path.parent / f".{path.name}.old-{tag}"
    tmp.mkdir()
    try:
        for ph in sol.v:
            np.save(tmp / f"value_{ph}.npy", sol.v[ph])
            np.save(tmp / f"policy_{ph}.npy", sol.pol[ph])
        joblib.dump(sol.model, tmp / "model.joblib")
        with open(tmp / "meta.json", "w") as f:
            json.dump(sol.meta, f, indent=2, default=float)
        if path.exists():
            for entry in path.iterdir():
                if not _is_solution_file(entry.name) and not (tmp / entry.name).exists():
                    if entry.is_dir():
                        shutil.copytree(entry, tmp / entry.name)
                    else:
                        try:
                            os.link(entry, tmp / entry.name)
                        except OSError:
                            shutil.copy2(entry, tmp / entry.name)
            path.rename(old)
        tmp.rename(path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        if old.exists() and not path.exists():
            old.rename(path)
        raise
    shutil.rmtree(old, ignore_errors=True)
    return path


def solution_exists(path: Path | str | None = None) -> bool:
    path = Path(path) if path is not None else SOLUTION_DIR
    return (path / "model.joblib").exists() and (path / "value_end.npy").exists()


def expected_shapes(model: GameModel) -> dict[str, tuple[int, ...]]:
    """Shape of v[phase] and pol[phase] for every phase the model has."""
    g = model.kern.grid
    out = {}
    for ph in ("end", "mid", "first"):
        lo, hi = model.phase_range(ph)
        if hi >= lo:
            out[ph] = (hi - lo + 1, g.nf, model.n_score, *model.extra_shape(ph))
    return out


def load_solution(path: Path | str | None = None, mmap: bool = True) -> Solution:
    """Load the solved game written by scripts/solve_mdp.py. Arrays are memory-mapped by
    default, so loading is instant and only the slices you touch are read.

    Every phase the saved model has must be present, with value and policy arrays of the
    shape and dtype the model implies, and no array may exist for a phase the model lacks;
    otherwise ValueError (a partial or mixed folder is never loaded)."""
    path = Path(path) if path is not None else SOLUTION_DIR
    if not solution_exists(path):
        raise FileNotFoundError(f"no solved game under {path}. Run: python scripts/solve_mdp.py")
    model = joblib.load(path / "model.joblib")
    shapes = expected_shapes(model)
    problems = []
    for ph in ("end", "mid", "first"):
        if ph not in shapes:
            stray = [n for n in (f"value_{ph}.npy", f"policy_{ph}.npy") if (path / n).exists()]
            if stray:
                problems.append(f"{stray} exist but the model has no {ph!r} phase")
    v, pol = {}, {}
    mode = "r" if mmap else None
    for ph, shape in shapes.items():
        for name, store, dtype in (("value", v, np.dtype(model.dtype)), ("policy", pol, np.uint8)):
            file = path / f"{name}_{ph}.npy"
            if not file.exists():
                problems.append(f"{file.name} is missing")
                continue
            arr = np.load(file, mmap_mode=mode)
            if arr.shape != shape or arr.dtype != dtype:
                problems.append(
                    f"{file.name} has shape {arr.shape} and dtype {arr.dtype}; the model "
                    f"implies {shape} and {dtype}"
                )
            store[ph] = arr
    if not (path / "meta.json").exists():
        problems.append("meta.json is missing")
    if problems:
        raise ValueError(f"the solved game under {path} is incomplete or inconsistent: "
                         + "; ".join(problems))  # fmt: skip
    with open(path / "meta.json") as f:
        meta = json.load(f)
    return Solution(model, v, pol, meta)


def comparable(x: Any) -> Any:
    """`x` as it reads back from JSON, without the "fitted_at" timestamps of fit stamps,
    so a stamp or key held in memory can be compared with one saved in meta.json."""

    def strip(y: Any) -> Any:
        if isinstance(y, dict):
            return {k: strip(w) for k, w in y.items() if k != "fitted_at"}
        if isinstance(y, list):
            return [strip(w) for w in y]
        return y

    return strip(json.loads(json.dumps(x, default=float)))


def reuse_mismatches(meta: dict, stamp: dict, tables_key: dict) -> list[str]:
    """Why a saved solution (its meta.json) was not built from these inputs; empty if it
    was. Compares the solution's stamp and tables key with the current ones (fit times
    aside), so `solve_mdp.py --reuse` cannot pair a saved game with different tables."""
    out = []
    for name, now in (("stamp", stamp), ("tables_key", tables_key)):
        saved = meta.get(name)
        if saved is None:
            out.append(f"the saved solution has no {name}")
            continue
        a, b = comparable(saved), comparable(now)
        if a != b:
            keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            out.append(f"{name} differs in {keys}")
    return out


# Solutions and simulators loaded by the module-level helpers, keyed by the folder and
# the time its meta.json was written, so a re-solve (or another folder) is never served
# from a stale cache.
_DEFAULT: dict[tuple[str, int], Solution] = {}
_DEFAULT_SIM: dict[tuple[str, int], Simulator] = {}


def _default_key(path: Path | str | None = None) -> tuple[str, int]:
    path = Path(path) if path is not None else SOLUTION_DIR
    meta = path / "meta.json"
    return str(path.resolve()), meta.stat().st_mtime_ns if meta.exists() else -1


def _default(path: Path | str | None = None) -> Solution:
    key = _default_key(path)
    if key not in _DEFAULT:
        _DEFAULT.clear()
        _DEFAULT[key] = load_solution(path)
    return _DEFAULT[key]


def _default_sim(path: Path | str | None = None) -> Simulator:
    key = _default_key(path)
    if key not in _DEFAULT_SIM:
        _DEFAULT_SIM.clear()
        _DEFAULT_SIM[key] = Simulator(_default(path))
    return _DEFAULT_SIM[key]


def value(state: GameState) -> float:
    """Win probability of the team with the ball, from the saved solution."""
    return _default().value(state)


def policy(state: GameState) -> str:
    """Optimal action in a state, from the saved solution."""
    return _default().policy(state)


# --------------------------------------------------------------------------------------
# Simulator: samples from the same transition model the solver used
# --------------------------------------------------------------------------------------


class _Sampler:
    """Draws one column per requested row of a sparse probability matrix."""

    def __init__(self, mat: sp.spmatrix) -> None:
        m = sp.csr_matrix(mat, dtype=float)
        m.sort_indices()
        self.indptr, self.indices = m.indptr, m.indices
        counts = np.diff(m.indptr)
        row = np.repeat(np.arange(m.shape[0]), counts)
        data = np.maximum(m.data, 0.0)
        csum = np.cumsum(data)
        start = np.concatenate([[0.0], csum])[m.indptr[:-1]]
        total = np.concatenate([[0.0], csum])[m.indptr[1:]] - start
        within = (csum - start[row]) / np.where(total[row] > 0, total[row], 1.0)
        self.cum = row + np.minimum(within, 1.0)
        last = m.indptr[1:] - 1
        self.cum[last[counts > 0]] = np.arange(m.shape[0])[counts > 0] + 1.0

    def sample(self, rows: np.ndarray, u: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.cum, rows + u, side="right")
        pos = np.clip(pos, self.indptr[rows], self.indptr[rows + 1] - 1)
        return self.indices[pos]


def _sample_dist(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    cum = np.cumsum(p)
    cum[-1] = 1.0
    return np.searchsorted(cum, u, side="right")


class Simulator:
    """Batch simulator of a solved game.

    The state of a batch is a dict of equal-length arrays: t (ticks), s (clipped score of
    the team with the ball), f (field state), a, b (timeouts of offense and defense), r
    (first-half flag), side (0 if the team with the ball is the one that had it at the
    start), pts (n, 2) points scored by side 0 and side 1 since the start.

    `post` chooses who makes the decisions that follow a play (defensive and offensive
    timeouts, the try after a touchdown): "model" replays exactly what the solver
    assumed, so simulated win frequencies converge to the solved values; "coach" uses the
    measured timeout rates and simple rules instead.
    """

    def __init__(self, sol: Solution, coach: dict[str, Any] | None = None) -> None:
        self.sol, self.model, self.grid = sol, sol.model, sol.grid
        self.coach = coach
        k, g = self.model.kern, self.grid
        self.play = {}
        for mode in KERNEL_KEYS:
            pad = sp.csr_matrix((g.nf, g.na - g.nf))
            stopped = k.k_all[mode] - sp.hstack([k.k_in[mode], pad]).tocsr()
            self.play[mode] = _Sampler(sp.hstack([stopped, k.k_in[mode]]))
        self.punt = _Sampler(k.k_punt)
        self.f10 = g.first10()
        self._two: dict[str, np.ndarray] = {}
        self._build_two_tables()
        # The simulator of the other team (see `Solution.other`); itself for identical teams.
        self.other: Simulator = self

    def _build_two_tables(self) -> None:
        """two[phase][t, s, e1, e2]: is the two-point try better after a touchdown at t?"""
        m, c, sol = self.model, self.model.clock, self.sol
        reach = max(len(c.neutral_hist), len(c.hurry_hist), c.punt_ticks, c.burn_ticks, 10)
        for ph in sol.v:
            lo, hi = m.phase_range(ph)
            tab = np.zeros((c.n_ticks + 1, m.n_score, *m.extra_shape(ph)), dtype=bool)
            for t in range(max(lo - reach, 0), hi + 1):
                tab[t] = sol.post(t, ph).two
            self._two[ph] = tab

    def start(self, n: int, **kw: Any) -> dict[str, np.ndarray]:
        """A batch of n copies of one state (keywords t, s, f, a, b, r; scalars or arrays).
        `s` is the true score differential; it is clipped for the model. Timeouts `a` and
        `b` default to a full set (`clock.n_timeouts`, as `GameState` defaults to 3 for the
        real model) and must lie in 0..n_timeouts; the others default to 0."""
        full = self.model.clock.n_timeouts
        unknown = set(kw) - {"t", "s", "f", "a", "b", "r"}
        if unknown:
            raise TypeError(f"Simulator.start got unknown state keys {sorted(unknown)}")
        st = {
            k: np.broadcast_to(
                np.asarray(kw.get(k, full if k in ("a", "b") else 0), dtype=np.int64), (n,)
            ).copy()
            for k in ("t", "s", "f", "a", "b", "r")
        }
        for k in ("a", "b"):
            if st[k].size and (st[k].min() < 0 or st[k].max() > full):
                raise ValueError(f"timeouts {k!r} must be in 0..{full}")
        st["s"] = np.clip(st["s"], -self.model.score_max, self.model.score_max)
        st["side"] = np.zeros(n, dtype=np.int64)
        st["pts"] = np.zeros((n, 2), dtype=np.int64)
        return st

    # ---- lookups ----
    def _phase_code(self, t: np.ndarray) -> np.ndarray:
        c = self.model.clock
        return np.where(t <= c.endgame_ticks, 0, np.where(t <= c.th, 1, 2))

    def _lookup(
        self,
        arrs: dict[str, np.ndarray],
        t: np.ndarray,
        f: np.ndarray,
        s: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        r: np.ndarray,
    ) -> np.ndarray:
        m = self.model
        si = s + m.score_max
        ph = self._phase_code(t)
        out = np.zeros(len(t), dtype=next(iter(arrs.values())).dtype)
        for code, name in enumerate(("end", "mid", "first")):
            sel = ph == code
            if not sel.any():
                continue
            lo = m.phase_range(name)[0]
            if name == "end":
                e1 = np.minimum(a[sel], m.extra_shape("end")[0] - 1)
                e2 = np.minimum(b[sel], m.extra_shape("end")[1] - 1)
            else:
                e1 = r[sel] if name == "first" else np.zeros(int(sel.sum()), dtype=np.int64)
                e2 = np.zeros(int(sel.sum()), dtype=np.int64)
            out[sel] = arrs[name][t[sel] - lo, f[sel], si[sel], e1, e2]
        return out

    def values(self, st: dict[str, np.ndarray], idx: np.ndarray | None = None) -> np.ndarray:
        """Solved win probability of the team with the ball, for each batch member."""
        idx = np.arange(len(st["t"])) if idx is None else idx
        return self._lookup(self.sol.v, *(st[k][idx] for k in ("t", "f", "s", "a", "b", "r")))

    def optimal_actions(self, st: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
        return self._lookup(
            self.sol.pol, *(st[k][idx] for k in ("t", "f", "s", "a", "b", "r"))
        ).astype(np.int64)

    def game_over(self, st: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
        """True where the team in front can kneel the game out (closed-form rule)."""
        c = self.model.clock
        if c.kneel_ticks is None or c.endgame_ticks == 0:
            return np.zeros(len(idx), dtype=bool)
        t, s, b = st["t"][idx], st["s"][idx], st["b"][idx]
        down = self.grid.unpack(st["f"][idx])[0]
        return (t <= c.endgame_ticks) & (s > 0) & (t <= c.kneel_ticks[b, down - 1])

    def coach_actions(
        self, st: dict[str, np.ndarray], idx: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Coach-like policy: fourth downs sampled from observed behavior by situation;
        protect a lead and chase a deficit in the last 5:00; kick as time expires."""
        m, c, g = self.model, self.model.clock, self.grid
        if self.coach is None:
            raise ValueError("Simulator needs a coach table for the coach policy")
        t, s, f = st["t"][idx], st["s"][idx], st["f"][idx]
        down, dbin, yl = g.unpack(f)
        sec = t * c.tick_seconds
        half_sec = np.where(t > c.th, (t - c.th) * c.tick_seconds, sec)
        end = (t <= c.endgame_ticks) & (c.endgame_ticks > 0)
        warn = c.warning_tick if c.warning_tick is not None else 0
        chase = (s < 0) | ((s == 0) & (t <= warn))
        act = np.where(end & chase, A_CHASE, np.where(end & (s > 0), A_PROTECT, A_NEUTRAL))
        fourth = down == g.n_down
        if fourth.any():
            sit = coach_situation(sec[fourth], half_sec[fourth], s[fourth])
            p = self.coach["table"][sit, dbin[fourth], yl[fourth] - 1]
            choice = (rng.random(int(fourth.sum()))[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
            choice = np.minimum(choice, 2)
            ok_fg, ok_punt = m.kern.fg_ok[f[fourth]], m.kern.punt_ok[f[fourth]]
            choice = np.where((choice == 1) & ~ok_fg, 0, choice)
            choice = np.where((choice == 2) & ~ok_punt, 0, choice)
            act[fourth] = np.where(choice == 1, A_FG, np.where(choice == 2, A_PUNT, act[fourth]))
        # As time expires: kick if a field goal ties or wins (or it is the end of the first
        # half), otherwise a trailing team must go for it.
        cc = (self._coach_setting("end_fg_max_yardline"), self._coach_setting("end_fg_ticks"))
        in_range = (yl <= cc[0]) & m.kern.fg_ok[f]
        last_game = (t <= cc[1]) & (s <= 0) & (s >= -m.fg_points)
        last_half = (t > c.th) & (t - c.th <= cc[1])
        act = np.where((last_game | last_half) & in_range, A_FG, act)
        late4 = fourth & end & (t <= warn)
        act = np.where(late4 & (s < -m.fg_points), A_CHASE, act)
        last_minute = late4 & (t <= warn // 2) & (s < 0) & (s >= -m.fg_points)
        act = np.where(last_minute, np.where(in_range, A_FG, A_CHASE), act)
        return act

    def _coach_setting(self, name: str) -> Any:
        """A coach end-of-half setting from the coach table, or its documented default
        (`COACH_DEFAULTS`) for tables saved before the key existed."""
        if self.coach is None:
            raise ValueError("Simulator needs a coach table for the coach policy")
        return self.coach[name] if name in self.coach else COACH_DEFAULTS[name]

    # ---- one play ----
    #
    # `advance` plays one action for a batch. It is split into the same pieces as the
    # solver's Bellman backup, in the order the solver values a play, so each helper can be
    # read next to the solver function it mirrors:
    #
    #   _sample_outcomes   the action terms of `Solution.backup` (play kernels, field goal,
    #                      punt, kneel): where the ball ends up and whether the clock runs
    #   _run_clock         `Solution._running_values` and `next_tick`: ticks used, and the
    #                      offense's and defense's timeout decisions after an in-bounds play
    #   _score             `Solution._derive`: points on the play and the try after a
    #                      touchdown
    #   _change_possession `Solution._aug` and `opp`: kickoffs, free kicks, turnovers and
    #                      the switch to the other team's side, with halftime
    #                      (`Solution.post` for the first half)
    #   _enter_end_phase   `Solution._boundary_weights`: timeouts drawn on entering the
    #                      last 5:00
    #
    # The random numbers are drawn in a fixed order (outcomes, clock, try, kickoffs,
    # halftime kickoff, end-phase timeouts), so a fixed seed gives the same games.

    def _sample_outcomes(
        self, f: np.ndarray, act: np.ndarray, t: np.ndarray, ph: np.ndarray,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:  # fmt: skip
        """Outcome of each action (the action terms of `Solution.backup`).

        Returns kind (K_* code), newf (field state if the offense keeps the ball), opp_yl
        (the other team's yard line after a turnover), inb (the play ended in bounds with
        the clock running) and dt (ticks used when the clock is stopped)."""
        c, g, k = self.model.clock, self.grid, self.model.kern
        n = len(f)
        out = {
            "kind": np.zeros(n, dtype=np.int64),
            "newf": f.copy(),
            "opp_yl": np.ones(n, dtype=np.int64),
            "inb": np.zeros(n, dtype=bool),
            "dt": np.ones(n, dtype=np.int64),
        }

        def decode(sel: np.ndarray, col: np.ndarray) -> None:
            """Turn kernel columns (see the `mdp` module docstring) into outcomes."""
            kd = np.full(len(col), K_KEEP)
            kd = np.where(col >= g.nf, K_TURN, kd)
            kd = np.where(col == g.col_td, K_TD, kd)
            kd = np.where(col == g.col_safety, K_SAFETY, kd)
            kd = np.where(col == g.col_opp_td, K_OPPTD, kd)
            out["kind"][sel] = kd
            out["newf"][sel] = np.where(kd == K_KEEP, col, f[sel])
            out["opp_yl"][sel] = np.where(kd == K_TURN, col - g.nf + 1, 1)

        # Plays: the kernel table depends on the mode and the time (`ClockSpec.kernel_key`).
        endp = (ph == 0) & (c.endgame_ticks > 0)
        inside = np.zeros(n, dtype=bool) if c.warning_tick is None else t <= c.warning_tick
        table = {
            "neutral": (act == A_NEUTRAL) & ~endp,
            "neutral_end": (act == A_NEUTRAL) & endp,
            "protect": act == A_PROTECT,
            "chase": (act == A_CHASE) & ~inside,
            "chase_late": (act == A_CHASE) & inside,
        }
        for key, mask in table.items():
            sel = np.nonzero(mask)[0]
            if len(sel):
                col = self.play[key].sample(f[sel], rng.random(len(sel)))
                running = col >= g.na  # the sampler marks clock-running outcomes
                decode(sel, np.where(running, col - g.na, col))
                out["inb"][sel] = running
        # Field goal: made, or the other team takes over at the spot of the miss.
        sel = np.nonzero(act == A_FG)[0]
        if len(sel):
            made = rng.random(len(sel)) < k.fg_make[f[sel]]
            out["kind"][sel] = np.where(made, K_FG, K_TURN)
            out["opp_yl"][sel] = k.fg_miss_col[f[sel]] - g.nf + 1
        # Punt: uses `punt_ticks` of clock.
        sel = np.nonzero(act == A_PUNT)[0]
        if len(sel):
            decode(sel, self.punt.sample(f[sel], rng.random(len(sel))))
            out["dt"][sel] = c.punt_ticks
        # Kneel: deterministic; the clock runs unless it was fourth down (a turnover).
        sel = np.nonzero(act == A_KNEEL)[0]
        if len(sel):
            keep = k.kneel_keep[f[sel]]
            decode(sel, k.kneel_col[f[sel]])
            out["inb"][sel] = keep
        return out

    def _next_ticks(self, tt: np.ndarray, kk: np.ndarray | int) -> np.ndarray:
        """Vector form of `Solution.next_tick`: time after kk ticks, stopping at the
        two-minute warning when an end-phase play crosses it."""
        c = self.model.clock
        t2 = np.maximum(tt - kk, 0)
        if c.warning_tick is None:
            return t2
        w = c.warning_tick
        return np.where((tt <= c.endgame_ticks) & (tt > w) & (t2 < w), w, t2)

    def _run_clock(
        self, st_now: dict[str, np.ndarray], act: np.ndarray, ph: np.ndarray, oc: dict,
        rng: np.random.Generator, post: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
        """Time after the play and both teams' timeouts (`Solution._running_values`).

        A stopped clock costs `dt` ticks. Before the end phase an in-bounds play takes a
        number of ticks from the neutral histogram. In the end phase the offense burns
        clock (protect, kneel) or plays at its tempo (neutral, chase); then, as the solver
        assumes with post="model", a neutral or chasing offense spends its own timeout when
        that is worth more, and the defense spends one when that lowers the offense's value
        (or, with post="coach" or an empirical defense, at the measured rates). Returns
        (new time, offense timeouts, defense timeouts)."""
        c = self.model.clock
        t, s, a, b, r = (st_now[x] for x in ("t", "s", "a", "b", "r"))
        a, b = a.copy(), b.copy()
        inb, newf = oc["inb"], oc["newf"]
        w = c.warning_tick
        inside = np.zeros(len(t), dtype=bool) if w is None else t <= w
        t1 = np.maximum(t - oc["dt"], 0)

        early = np.nonzero(inb & (ph != 0))[0]
        if len(early):
            h = _sample_dist(c.neutral_hist, rng.random(len(early))) + 1
            t1[early] = np.maximum(t[early] - h, 0)
        late = np.nonzero(inb & (ph == 0))[0]
        if not len(late):
            return t1, a, b
        tl, fl, sl, al, bl, rl = t[late], newf[late], s[late], a[late], b[late], r[late]
        al_act = act[late]
        model_post = post != "coach"

        def look(q: np.ndarray, tt: np.ndarray, aa: np.ndarray, bb: np.ndarray) -> np.ndarray:
            return self._lookup(self.sol.v, tt, fl[q], sl[q], aa, bb, rl[q]).astype(float)

        # Without any timeout: protect (and kneel) burn the clock, the others take a
        # number of ticks from their tempo histogram (`ClockSpec.tempo_hist`).
        t_run = self._next_ticks(tl, c.burn_ticks)
        run_val = np.zeros(len(late))
        free = (al_act == A_NEUTRAL) | (al_act == A_CHASE)
        hurry_late = c.hurry_hist if c.hurry_hist_late is None else c.hurry_hist_late
        hists = (
            (al_act == A_NEUTRAL, c.neutral_hist),
            ((al_act == A_CHASE) & ~inside[late], c.hurry_hist),
            ((al_act == A_CHASE) & inside[late], hurry_late),
        )
        for mask, hist in hists:
            q = np.nonzero(mask)[0]
            if len(q) == 0:
                continue
            h = _sample_dist(hist, rng.random(len(q))) + 1
            t_run[q] = self._next_ticks(tl[q], h)
            if model_post:
                for hh, p in enumerate(hist, start=1):
                    if p > 0:
                        run_val[q] += p * look(q, self._next_ticks(tl[q], hh), al[q], bl[q])
        q = np.nonzero(~free)[0]
        if model_post and len(q):
            run_val[q] = look(q, t_run[q], al[q], bl[q])
        # The offense's own timeout (neutral or chase only).
        every = np.arange(len(late))
        if model_post:
            own = np.where(al > 0, look(every, tl - 1, np.maximum(al - 1, 0), bl), -np.inf)
            off_to = free & (own > run_val)
        else:
            off_to = (al_act == A_CHASE) & (al > 0) & (tl <= (w or 0))
        # The defense's timeout.
        if post == "coach" or c.defense_timeout != "optimal":
            rate = np.where(tl <= (w or 0), c.def_rate_after[bl], c.def_rate_before[bl])
            def_to = (bl > 0) & (sl > 0) & (rng.random(len(late)) < rate)
        else:
            base = np.where(off_to, own, run_val)
            stop = np.where(bl > 0, look(every, tl - 1, al, np.maximum(bl - 1, 0)), np.inf)
            def_to = stop < base
        off_to = off_to & ~def_to
        t1[late] = np.where(def_to | off_to, tl - 1, t_run)
        a[late] = al - off_to
        b[late] = bl - def_to
        return t1, a, b

    def _score(
        self, st_now: dict[str, np.ndarray], kind: np.ndarray, t1: np.ndarray, ph: np.ndarray,
        rng: np.random.Generator, post: str,
    ) -> np.ndarray:  # fmt: skip
        """Points on the play, to the scoring side (`Solution._derive`). After a
        touchdown the scoring team takes the try the solver chose (post="model": the
        better of kick and two-point try at that state) or the coach rule (post="coach")."""
        m, c = self.model, self.model.clock
        s, a, b, r = (st_now[x] for x in ("s", "a", "b", "r"))
        pts = np.zeros(len(kind), dtype=np.int64)
        scored = (kind == K_TD) | (kind == K_OPPTD)
        if scored.any():
            j = np.nonzero(scored)[0]
            mine = kind[j] == K_TD
            # The scoring team's own view of the state at the time of the try.
            ss = np.where(mine, s[j], -s[j])
            aa, bb = np.where(mine, a[j], b[j]), np.where(mine, b[j], a[j])
            rr = np.where(mine, r[j], 1 - r[j])
            if post == "coach":
                lead = ss + m.td_points
                # In seconds of game time left, so the rule means the same thing whether
                # the horizon is the whole game or only the end game (tables saved before
                # these keys existed get COACH_DEFAULTS: the fourth quarter).
                two_sec = self._coach_setting("two_point_seconds")
                leads = tuple(self._coach_setting("two_point_leads"))
                two = (t1[j] * c.tick_seconds <= two_sec) & np.isin(lead, leads)
            else:
                two = np.zeros(len(j), dtype=bool)
                for code, name in enumerate(("end", "mid", "first")):
                    q = ph[j] == code
                    if q.any():
                        e = m.extra_shape(name)
                        e1 = np.minimum(aa[q], e[0] - 1) if name == "end" else rr[q] * (e[0] - 1)
                        e2 = np.minimum(bb[q], e[1] - 1) if name == "end" else 0
                        ix = (t1[j][q], ss[q] + m.score_max, e1, e2)
                        # A return touchdown is scored by the other team: its own choice.
                        two[q] = np.where(mine[q], self._two[name][ix], self.other._two[name][ix])
            u = rng.random(len(j))
            pts[j] = m.td_points + np.where(two, 2 * (u < m.p_two), 1 * (u < m.p_xp))
        pts = np.where(kind == K_FG, m.fg_points, pts)
        return np.where(kind == K_SAFETY, m.safety_points, pts)

    def _change_possession(
        self, kind: np.ndarray, newf: np.ndarray, opp_yl: np.ndarray, r: np.ndarray,
        cross_half: np.ndarray, rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
        """Next field state and whether the other team now has the ball (`Solution._aug`
        and `opp`). A score is followed by a kickoff (a safety by a free kick), a turnover
        gives the other team the ball where it was lost, and at halftime the team whose
        flag says so receives the second-half kickoff (`Solution.post` for the first
        half). Returns (field state, flip)."""
        k = self.model.kern
        yl_new = np.where(kind == K_TURN, opp_yl, 1)
        kick = (kind == K_TD) | (kind == K_FG) | (kind == K_OPPTD)
        nk = int(kick.sum())
        if nk:
            yl_new[kick] = _sample_dist(k.kickoff[1:], rng.random(nk)) + 1
        fk = kind == K_SAFETY
        if fk.any():
            yl_new[fk] = _sample_dist(k.free_kick[1:], rng.random(int(fk.sum()))) + 1
        flip = (kind == K_TURN) | (kind == K_TD) | (kind == K_FG) | (kind == K_SAFETY)
        f = np.where(kind == K_KEEP, newf, self.f10[yl_new])
        if cross_half.any():
            r_now = np.where(flip, 1 - r, r)
            flip = np.where(cross_half, flip ^ (r_now == 0), flip)
            nh = int(cross_half.sum())
            f[cross_half] = self.f10[_sample_dist(k.kickoff[1:], rng.random(nh)) + 1]
        return f, flip

    def _enter_end_phase(
        self, s: np.ndarray, a: np.ndarray, b: np.ndarray, cross_end: np.ndarray,
        rng: np.random.Generator,
    ) -> None:  # fmt: skip
        """Draw both teams' timeouts on entering the last 5:00, from the measured shares
        for leading, trailing and tied teams (`Solution._boundary_weights`). In place."""
        if not cross_end.any():
            return
        c = self.model.clock
        j = np.nonzero(cross_end)[0]
        lead, trail = c.shares_leading, c.shares_trailing
        mean = 0.5 * (lead + trail)
        u1, u2 = rng.random(len(j)), rng.random(len(j))
        for sign, pa, pb in ((1, lead, trail), (-1, trail, lead), (0, mean, mean)):
            q = np.sign(s[j]) == sign
            a[j[q]] = _sample_dist(pa, u1[q])
            b[j[q]] = _sample_dist(pb, u2[q])

    def advance(
        self,
        st: dict[str, np.ndarray],
        idx: np.ndarray,
        act: np.ndarray,
        rng: np.random.Generator,
        post: str = "model",
    ) -> dict[str, np.ndarray]:
        """Play one action for the batch members `idx` (in place). Returns what happened:
        kind (K_* code), points (to the scoring side), scrimmage (bool), half_end (bool),
        flip (the other team has the ball now). See the comment above for the pieces."""
        m, c = self.model, self.model.clock
        now = {key: st[key][idx] for key in ("t", "s", "f", "a", "b", "r")}
        t, s, r = now["t"], now["s"], now["r"]
        ph = self._phase_code(t)

        oc = self._sample_outcomes(now["f"], act, t, ph, rng)
        kind = oc["kind"]
        scrimmage = (act != A_FG) & (act != A_PUNT)

        t1, a, b = self._run_clock(now, act, ph, oc, rng, post)
        # Halftime and the start of the end phase are boundaries the clock stops at.
        cross_half = (t > c.th) & (t1 <= c.th)
        t1 = np.where(cross_half, c.th, t1)
        cross_end = (t > c.endgame_ticks) & (t1 <= c.endgame_ticks) & (c.endgame_ticks > 0)

        # The try after a touchdown sees the timeouts as they are after the play.
        pts = self._score({**now, "a": a, "b": b}, kind, t1, ph, rng, post)
        to_me = (kind == K_TD) | (kind == K_FG)
        to_them = (kind == K_OPPTD) | (kind == K_SAFETY)
        s = s + np.where(to_me, pts, 0) - np.where(to_them, pts, 0)
        side = st["side"][idx]
        np.add.at(st["pts"], (idx[to_me], side[to_me]), pts[to_me])
        np.add.at(st["pts"], (idx[to_them], 1 - side[to_them]), pts[to_them])

        f, flip = self._change_possession(kind, oc["newf"], oc["opp_yl"], r, cross_half, rng)
        s = np.where(flip, -s, s)
        a, b = np.where(flip, b, a), np.where(flip, a, b)
        r = np.where(flip, 1 - r, r)
        r = np.where(cross_half, 0, r)
        s = np.clip(s, -m.score_max, m.score_max)
        self._enter_end_phase(s, a, b, cross_end, rng)

        st["t"][idx], st["s"][idx], st["f"][idx] = t1, s, f
        st["a"][idx], st["b"][idx], st["r"][idx] = a, b, r
        st["side"][idx] = np.where(flip, 1 - side, side)
        return {
            "kind": kind,
            "points": pts,
            "scrimmage": scrimmage,
            "half_end": cross_half,
            "flip": flip,
        }


def simulate(
    sim: Simulator,
    st: dict[str, np.ndarray],
    rng: np.random.Generator,
    policy: str = "optimal",
    post: str | None = None,
    first_action: np.ndarray | None = None,
    log_drives: bool = False,
    checkpoints: list[int] | None = None,
    max_steps: int = 5000,
) -> dict[str, Any]:
    """Play every game of the batch to the end.

    policy: "optimal" (the solved policy) or "coach" (see `Simulator.coach_actions`).
    post: who decides timeouts and tries after a play; defaults to "model" for the optimal
    policy and "coach" for the coach policy.
    first_action: optional action codes forced on the first play.

    Returns result (1, 0.5 or 0 for the team that had the ball at the start, by the
    model's clipped score), margin (true points margin for that team, counting the
    starting score), drives (a dict of arrays, one row per drive, if `log_drives`), and
    checkpoint_margin {tick: margin of side 0 at the first snap at or inside that time}.
    """
    post = post or ("coach" if policy == "coach" else "model")
    n = len(st["t"])
    s0 = st["s"].copy()
    done = np.zeros(n, dtype=bool)
    result = np.full(n, np.nan)
    d_plays = np.zeros(n, dtype=np.int64)
    d_new = np.ones(n, dtype=bool)
    d_start = {k: np.zeros(n, dtype=np.int64) for k in ("t", "s", "yl", "a", "b", "side")}
    logs: dict[str, list] = {
        k: []
        for k in ("game", "side", "t", "s", "yl", "a", "b", "plays", "result", "points", "end_t")
    }
    cps = {c: np.full(n, np.nan) for c in (checkpoints or [])}
    grid, terminal = sim.grid, sim.sol.terminal

    def margin(idx: np.ndarray) -> np.ndarray:
        return s0[idx] + st["pts"][idx, 0] - st["pts"][idx, 1]

    def close(idx: np.ndarray, res: np.ndarray, pts: np.ndarray) -> None:
        if not log_drives or len(idx) == 0:
            return
        logs["game"].append(idx)
        for k in ("side", "t", "s", "yl", "a", "b"):
            logs[k].append(d_start[k][idx].copy())
        logs["plays"].append(d_plays[idx].copy())
        logs["result"].append(res)
        logs["points"].append(pts)
        logs["end_t"].append(st["t"][idx].copy())
        d_plays[idx] = 0
        d_new[idx] = True

    for step_no in range(max_steps):
        idx = np.nonzero(~done)[0]
        if len(idx) == 0:
            break
        for c, arr in cps.items():
            hit = idx[(st["t"][idx] <= c) & np.isnan(arr[idx])]
            arr[hit] = margin(hit)
        t = st["t"][idx]
        ended = t == 0
        over = sim.game_over(st, idx) & ~ended
        fin = idx[ended | over]
        if len(fin):
            val = np.where(over[ended | over], 1.0, terminal[st["s"][fin] + sim.model.score_max])
            result[fin] = np.where(st["side"][fin] == 0, val, 1.0 - val)
            done[fin] = True
            open_drive = fin[~d_new[fin]]
            close(
                open_drive,
                np.full(len(open_drive), DRIVE_RESULTS.index("end_of_half")),
                np.zeros(len(open_drive), dtype=np.int64),
            )
            idx = idx[~(ended | over)]
            if len(idx) == 0:
                continue
        if step_no == 0 and first_action is not None:
            act = np.asarray(first_action)[idx]
        elif policy == "coach":
            act = sim.coach_actions(st, idx, rng)
        else:
            act = sim.optimal_actions(st, idx)
        if log_drives:
            new = idx[d_new[idx]]
            d_start["t"][new], d_start["s"][new] = st["t"][new], st["s"][new]
            d_start["yl"][new] = grid.unpack(st["f"][new])[2]
            d_start["a"][new], d_start["b"][new] = st["a"][new], st["b"][new]
            d_start["side"][new] = st["side"][new]
            d_new[new] = False
        down = grid.unpack(st["f"][idx])[0]
        info = sim.advance(st, idx, act, rng, post)
        if log_drives:
            d_plays[idx] += info["scrimmage"]
            kind = info["kind"]
            res = np.full(len(idx), -1)
            name = DRIVE_RESULTS.index
            res = np.where(info["half_end"] | (st["t"][idx] == 0), name("end_of_half"), res)
            res = np.where(
                kind == K_TURN, np.where(down == grid.n_down, name("downs"), name("turnover")), res
            )
            res = np.where(kind == K_OPPTD, name("opp_touchdown"), res)
            res = np.where(kind == K_SAFETY, name("safety"), res)
            res = np.where(kind == K_TD, name("touchdown"), res)
            res = np.where(
                act == A_FG,
                np.where(kind == K_FG, name("field_goal"), name("missed_field_goal")),
                res,
            )
            res = np.where(act == A_PUNT, name("punt"), res)
            ends = res >= 0
            own_pts = np.where((kind == K_TD) | (kind == K_FG), info["points"], 0)
            close(idx[ends], res[ends], own_pts[ends])
    if not done.all():
        raise RuntimeError("simulation did not finish; raise max_steps")
    out: dict[str, Any] = {
        "result": result,
        "margin": s0 + st["pts"][:, 0] - st["pts"][:, 1],
        "points": st["pts"].copy(),
        "checkpoint_margin": cps,
    }
    if log_drives:
        out["drives"] = {
            k: np.concatenate(v) if v else np.zeros(0, dtype=np.int64) for k, v in logs.items()
        }
    return out


def step(
    state: GameState,
    action: str,
    rng: np.random.Generator,
    sim: Simulator | None = None,
) -> tuple[GameState, dict[str, Any]]:
    """One play of the solved game's transition model, for one state.

    `action` is a name from ACTION_NAMES. Decisions that are not the offense's play call
    (the defense's timeout, the try after a touchdown, the hurrying offense's own
    timeout) are made as the solver assumed. Returns the next state, from the side of
    the team that then has the ball, and info: possession_changed, points (scored on the
    play by `scorer`: "offense" or "defense"), done, result (for the team that had the
    ball before the play: 1, 0.5 or 0, once done).

    "game_over" is the kneel-out: it is only legal where the team in front can kneel the
    game out (`Simulator.game_over`); anywhere else it raises ValueError. Timeouts above
    the model's maximum are clipped, as `Solution.index` does.
    """
    sim = _default_sim() if sim is None else sim
    sol, m = sim.sol, sim.model
    ph, ix = sol.index(state)
    t = ix[0] + m.phase_range(ph)[0]
    full = m.clock.n_timeouts
    st = sim.start(
        1,
        t=t,
        s=state.score,
        f=ix[1],
        a=min(state.off_timeouts, full) if ph == "end" else 0,
        b=min(state.def_timeouts, full) if ph == "end" else 0,
        r=state.receives_second_half,
    )
    idx = np.array([0])
    code = ACTION_NAMES.index(action)
    can_kneel_out = bool(sim.game_over(st, idx)[0])
    if code == A_OVER and t > 0 and not can_kneel_out:
        raise ValueError(
            f"'game_over' (kneel out) is not available in {state}: the team with the ball "
            "cannot kneel the game out from here"
        )
    if t == 0 or can_kneel_out or code == A_OVER:
        res = 1.0 if t > 0 else float(sol.terminal[st["s"][0] + m.score_max])
        return state, {
            "possession_changed": False,
            "points": 0,
            "scorer": None,
            "done": True,
            "result": res,
        }
    info = sim.advance(st, idx, np.array([code]), rng, "model")
    down, dbin, yl = sim.grid.unpack(st["f"])
    nxt = GameState(
        seconds=int(st["t"][0]) * m.clock.tick_seconds,
        score=int(st["s"][0]),
        down=int(down[0]),
        ydstogo=int(min(sim.grid.dist_rep(dbin)[0], yl[0])),
        yardline_100=int(yl[0]),
        off_timeouts=int(st["a"][0]),
        def_timeouts=int(st["b"][0]),
        receives_second_half=int(st["r"][0]),
    )
    flipped = bool(st["side"][0] == 1)
    done = bool(st["t"][0] == 0 or sim.game_over(st, idx)[0])
    result = None
    if done:
        val = 1.0 if st["t"][0] > 0 else float(sol.terminal[st["s"][0] + m.score_max])
        result = 1.0 - val if flipped else val
    kind = int(info["kind"][0])
    scorer = {K_TD: "offense", K_FG: "offense", K_OPPTD: "defense", K_SAFETY: "defense"}.get(kind)
    return nxt, {
        "possession_changed": flipped,
        "points": int(info["points"][0]),
        "scorer": scorer,
        "done": done,
        "result": result,
    }


def build_model(
    cfg: dict,
    models: Any,
    tables: dict[str, Any],
    horizon: str = "game",
    go_convert_shift: float | None = None,
    chase_gain_tilt: float | None = None,
    defense_timeout: str | None = None,
) -> GameModel:
    """Kernels plus clock from the config, the M1 bundle and the fitted count tables.
    The keyword overrides are the sensitivity knobs; None means the config value."""
    from fourth_and_rl.mdp import (
        build_field_kernels,
        grid_from_config,
        mode_adjustments,
        stop_prob_by_gain,
    )

    sens, acts = cfg["sensitivity"], cfg["actions"]
    tilt = sens["chase_gain_tilt"] if chase_gain_tilt is None else chase_gain_tilt
    kern = build_field_kernels(
        models,
        grid_from_config(cfg),
        stop_prob_by_gain(tables),
        free_kick=None if tables.get("free_kick") is None else np.asarray(tables["free_kick"]),
        go_convert_shift=sens["go_convert_shift"] if go_convert_shift is None else go_convert_shift,
        adjust=mode_adjustments(tables, cfg, chase_extra_tilt=tilt),
        field_goal_any_down=acts["field_goal_any_down"],
        kneel=acts["kneel"],
    )
    model = model_from_config(cfg, kern, tables, models, horizon)
    if defense_timeout is not None:
        model.clock.defense_timeout = defense_timeout
    return model


def late_trailing_drive_stats(
    drives: dict[str, np.ndarray], result: np.ndarray, tick_seconds: int = 5
) -> dict[str, Any]:
    """The checks of docs/research/endgame_clock.md section 4, computed on simulated drives.

    Drives by a team down 4 to 8 that start with 1:00 to 5:00 left: touchdown rate (overall,
    by time left, by field position), how long touchdown drives take, whether the team that
    led gets the ball back, and who wins. `drives` and `result` come from `simulate`.
    """
    sec = drives["t"] * tick_seconds
    won = np.where(drives["side"] == 0, result[drives["game"]], 1.0 - result[drives["game"]])
    td = drives["result"] == DRIVE_RESULTS.index("touchdown")
    sel = (sec >= 60) & (sec <= 300) & (drives["s"] <= -4) & (drives["s"] >= -8)

    def rate(mask: np.ndarray, x: np.ndarray) -> dict[str, Any]:
        return {"n": int(mask.sum()), "rate": float(x[mask].mean()) if mask.any() else None}

    out: dict[str, Any] = {
        "n_drives": int(sel.sum()),
        "td_rate": rate(sel, td),
        "trailing_team_win_rate": float(won[sel].mean()) if sel.any() else None,
        "by_time": {
            name: rate(sel & (sec > lo) & (sec <= hi), td)
            for name, lo, hi in (
                ("1_to_2_min", 59, 120),
                ("2_to_3_min", 120, 180),
                ("3_to_4_min", 180, 240),
                ("4_to_5_min", 240, 300),
            )
        },
        "by_field_position": {
            name: rate(sel & (drives["yl"] >= lo) & (drives["yl"] <= hi), td)
            for name, lo, hi in (
                ("own_1_to_20", 80, 99),
                ("own_21_to_30", 70, 79),
                ("own_31_to_50", 50, 69),
                ("opp_territory", 1, 49),
            )
        },
        "own_1_to_20_with_1_to_2_min": rate(sel & (drives["yl"] >= 80) & (sec <= 120), td),
        "own_21_to_30_with_3_to_4_min": rate(
            sel & (drives["yl"] >= 70) & (drives["yl"] <= 79) & (sec > 180) & (sec <= 240), td
        ),
    }
    used = (drives["t"] - drives["end_t"]) * tick_seconds
    m = sel & td
    if m.any():
        out["td_drive_seconds_used_median"] = float(np.median(used[m]))
        out["td_drive_plays_median"] = float(np.median(drives["plays"][m]))
    # The next drive of the same game (rows of one game are in time order).
    n = len(sec)
    order = np.lexsort((np.arange(n), drives["game"]))
    nxt = np.full(n, -1)
    same = drives["game"][order][1:] == drives["game"][order][:-1]
    nxt[order[:-1][same]] = order[1:][same]
    has = nxt >= 0
    answer = has & (drives["side"][nxt] != drives["side"]) & (drives["t"][nxt] > 0)
    ans_sec = np.where(answer, drives["t"][nxt] * tick_seconds, 0)
    out["after_trailing_touchdown"] = {
        "n": int(m.sum()),
        "leader_gets_ball_in_regulation": float(answer[m].mean()) if m.any() else None,
        "seconds_left_at_answer_drive_median": (
            float(np.median(ans_sec[m & answer])) if (m & answer).any() else None
        ),
        "share_with_30s_or_more": float((ans_sec[m] >= 30).mean()) if m.any() else None,
        "share_with_60s_or_more": float((ans_sec[m] >= 60).mean()) if m.any() else None,
        "original_leader_win_rate": float(1.0 - won[m].mean()) if m.any() else None,
    }
    need = m & answer & (ans_sec >= 30) & (drives["s"][nxt] <= 0)
    scored = np.isin(
        drives["result"][nxt], (DRIVE_RESULTS.index("touchdown"), DRIVE_RESULTS.index("field_goal"))
    )
    out["answer_drive_needing_points"] = {
        "n": int(need.sum()),
        "scores_td_or_fg": float(scored[need].mean()) if need.any() else None,
        "original_leader_win_rate": float(1.0 - won[need].mean()) if need.any() else None,
    }
    no_td = sel & ~td
    out["after_no_touchdown"] = {
        "n": int(no_td.sum()),
        "original_leader_win_rate": float(1.0 - won[no_td].mean()) if no_td.any() else None,
    }
    return out
