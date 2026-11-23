"""Exact state occupancy of two fixed policies: the forward pass of the game MDP.

`policy_eval` runs the game backward in time (values). This module runs it forward:
probability mass starts at the opening kickoff and flows down the clock through exactly
the transitions the backup uses, so the mass that arrives at a pre-snap state is the
probability that the game passes through it. That gives, inside the exact computation,
any expectation of a sum over the states a game visits, for example the sequence KL of a
policy from the coach,

    E[ sum over A's fourth downs of KL(pi(. | s) || coach(. | s)) ] = sum over s of d_A(s) KL(s)

where d_A(s) is the occupancy of state s with team A on offense. With the same mass, the
terminal states give the win probability, which must equal the backward start value.

How the forward pass mirrors the backup
---------------------------------------
The backup at tick t is linear in values at earlier ticks once post-decisions are fixed
(post="coach": timeouts and tries follow rules that do not depend on values). Every term
"value at time t' times a weight" of the backup becomes "mass at time t' plus mass times
the same weight" here, term by term:

  play kernels     K_all^T and K_in^T push the mass of a play to outcome columns at t-1,
                   or (clock running) through the timeout rules and the tempo histogram
  outcome columns  own field states, the other team's field state (`opp` is a
                   permutation: score reversed, timeouts swapped, flag flipped), the try
                   after a touchdown (coach rule), field goals and safeties, which hand
                   the other team a kickoff or free kick
  kickoffs         the receiving team's field position from the kickoff table
  boundaries       entering the last 5:00 draws timeouts from the measured shares;
                   halftime gives the second-half kickoff to the team whose flag says so
  kneel-out        mass in a state the leader can kneel out is a win for the leader

Policies are `policy_eval.Policy` callables; they are called with an empty dict for q, so
they must not depend on the team's own action values (a tilt on a stored proxy Q is fine,
a best response is not). Only post="coach" is supported: the model's own post-decisions
are maxima over values, which a forward pass cannot replay without the values.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from fourth_and_rl import mdp_game as mg
from fourth_and_rl.policy_eval import Policy, peak_rss_gb

TEAMS = ("A", "B")
StateHook = Callable[[str, int, np.ndarray], None]


def flip(x: np.ndarray, phase: str) -> np.ndarray:
    """`Solution.opp` without the "1 -": re-describe arrays (..., S, E1, E2) from the other
    team's side (score reversed, timeouts swapped in the end phase, flag flipped in the
    first half). It is a permutation and its own inverse, so it moves mass as well."""
    y = x[..., ::-1, :, :]
    if phase == "end":
        return y.swapaxes(-1, -2)
    return y[..., ::-1, :] if phase == "first" else y


class ForwardPass:
    """Occupancy of team A playing `policy_a` against team B playing `policy_b` (the same
    policy if None), from the opening kickoff with a coin flip for the receiver.

    `hooks` are called as hook(team, t, mass) with the pre-snap mass of each team at each
    tick (shape (nf, S, E1, E2), kneel-out states already removed), before it moves on.
    After `run()`: `result_a` (A's win probability, ties counted half), `terminal` (mass
    by team and outcome) and `seconds`."""

    def __init__(
        self,
        model: mg.GameModel,
        policy_a: Policy,
        policy_b: Policy | None = None,
        coach: dict[str, Any] | None = None,
        hooks: list[StateHook] | None = None,
        dtype: Any = np.float64,
    ) -> None:
        self.m, self.c = model, model.clock
        self.k, self.g = model.kern, model.kern.grid
        self.dtype = np.dtype(dtype)
        self.policies = {"A": policy_a, "B": policy_b or policy_a}
        self.hooks = hooks or []
        src = coach or {}
        self.two_sec = src.get("two_point_seconds", mg.COACH_DEFAULTS["two_point_seconds"])
        self.two_leads = list(src.get("two_point_leads", mg.COACH_DEFAULTS["two_point_leads"]))
        g, k = self.g, self.k
        self.f10 = g.first10()[1:]
        if len(np.unique(self.f10)) != len(self.f10):
            raise ValueError("first-and-10 states are not distinct")
        self.kall_t = {key: k.k_all[key].T.tocsr().astype(self.dtype) for key in k.k_all}
        self.kin_t = {key: k.k_in[key].T.tocsr().astype(self.dtype) for key in k.k_in}
        self.kpunt_t = k.k_punt.T.tocsr().astype(self.dtype)
        self.scores = np.arange(-model.score_max, model.score_max + 1)
        self.weights = mg.Solution(model)._weights.astype(self.dtype)  # (S, A, B)
        self.downs = g.unpack(np.arange(g.nf))[0]
        self.D: dict[str, dict[int, np.ndarray]] = {x: {} for x in TEAMS}
        self.K: dict[str, dict[tuple[int, str], dict[str, np.ndarray]]] = {x: {} for x in TEAMS}
        self.H = {x: np.zeros((model.n_score, 2), dtype=self.dtype) for x in TEAMS}
        self.terminal = {x: {"win": 0.0, "tie": 0.0, "loss": 0.0, "kneel_out": 0.0}
                         for x in TEAMS}  # fmt: skip
        self.seconds = 0.0
        self.peak_rss_gb = 0.0

    # ---- shapes and buffers ----
    def shape(self, phase: str) -> tuple[int, ...]:
        return (self.g.nf, self.m.n_score, *self.m.extra_shape(phase))

    def _d(self, team: str, t: int) -> np.ndarray:
        if t not in self.D[team]:
            self.D[team][t] = np.zeros(self.shape(self.m.phase(t)), dtype=self.dtype)
        return self.D[team][t]

    def _k(self, team: str, t: int, phase: str, kind: str) -> np.ndarray:
        slot = self.K[team].setdefault((t, phase), {})
        if kind not in slot:
            shape = self.shape(phase) if kind == "vmid" else self.shape(phase)[1:]
            slot[kind] = np.zeros(shape, dtype=self.dtype)
        return slot[kind]

    @staticmethod
    def other(team: str) -> str:
        return "B" if team == "A" else "A"

    # ---- mass arriving at a post (the outcome values at one moment, `Solution.post`) ----
    def add_post(self, team: str, t: int, phase: str, kind: str, arr: np.ndarray,
                 rows: np.ndarray | None = None) -> None:  # fmt: skip
        """Mass on team's post value `kind` (v, ko, fk, td, fg, saf) at time t seen from a
        play that started in `phase`. For kind v, `rows` are the field states of arr's
        rows (all of them if None)."""
        th = self.c.th
        if phase == "first" and t <= th:
            # Halftime: the half is over whatever the field state (`Solution._base`).
            if kind == "v":
                self.H[team] += arr.sum(0)[:, :, 0]
            elif kind in ("ko", "fk"):
                self.H[team] += arr[:, :, 0]
            else:
                self._derive_t(team, t, phase, {kind: arr})
            return
        if kind == "v":
            if self.m.phase(t) == phase:
                dst = self._d(team, t)
            else:  # a play of the "mid" phase ends inside the last 5:00
                dst = self._k(team, t, phase, "vmid")
            if rows is None:
                dst += arr
            else:
                np.add.at(dst, rows, arr)
            return
        self._k(team, t, phase, kind)[...] += arr

    def _derive_t(self, team: str, t: int, phase: str, slot: dict[str, np.ndarray]) -> None:
        """Transpose of `Solution._derive`: the try, field goals and safeties hand the
        other team a kickoff (or a free kick after a safety)."""
        m = self.m
        n = m.n_score

        def shift_add(dst: np.ndarray, src: np.ndarray, pts: int, w: Any = 1.0) -> None:
            idx = np.clip(np.arange(n) + pts, 0, n - 1)
            np.add.at(dst, idx, src * w)

        ko = np.zeros(self.shape(phase)[1:], dtype=self.dtype)
        fk = np.zeros_like(ko)
        if "td" in slot:
            x = slot["td"]
            late = t * self.c.tick_seconds <= self.two_sec
            two = (late & np.isin(self.scores + m.td_points, self.two_leads))[:, None, None]
            xt, xx = np.where(two, x, 0.0), np.where(two, 0.0, x)
            shift_add(ko, xt, m.td_points + 2, m.p_two)
            shift_add(ko, xt, m.td_points, 1.0 - m.p_two)
            shift_add(ko, xx, m.td_points + 1, m.p_xp)
            shift_add(ko, xx, m.td_points, 1.0 - m.p_xp)
        if "fg" in slot:
            shift_add(ko, slot["fg"], m.fg_points)
        if "saf" in slot:
            shift_add(fk, slot["saf"], -m.safety_points)
        o = self.other(team)
        if ko.any():
            self.add_post(o, t, phase, "ko", np.ascontiguousarray(flip(ko, phase)))
        if fk.any():
            self.add_post(o, t, phase, "fk", np.ascontiguousarray(flip(fk, phase)))

    def _resolve(self, t: int) -> None:
        """Turn every post mass at time t into pre-snap mass at t (`Solution._base`)."""
        k, c = self.k, self.c
        if t == c.th:
            for team in TEAMS:
                h = self.H[team]
                self._k(team, t, "mid", "ko")[:, 0, 0] += h[:, 1]
                self._k(self.other(team), t, "mid", "ko")[::-1, 0, 0] += h[:, 0]
                h[...] = 0.0
        phases = sorted({ph for team in TEAMS for (tt, ph) in self.K[team] if tt == t})
        for ph in phases:
            for team in TEAMS:
                slot = self.K[team].get((t, ph), {})
                tries = {kd: slot.pop(kd) for kd in ("td", "fg", "saf") if kd in slot}
                if tries:
                    self._derive_t(team, t, ph, tries)
            for team in TEAMS:
                slot = self.K[team].pop((t, ph), {})
                v = slot.get("vmid")
                kicks = {"ko": k.kickoff[1:], "fk": k.free_kick[1:]}
                for kind, dist in kicks.items():
                    if kind in slot:
                        if v is None:
                            v = np.zeros(self.shape(ph), dtype=self.dtype)
                        w = dist.astype(self.dtype)[:, None, None, None]
                        v[self.f10] += w * slot[kind][None]
                if v is None:
                    continue
                if self.m.phase(t) == ph:
                    self._d(team, t)[...] += v
                else:  # "mid" into the last 5:00: timeouts from the measured shares
                    self._d(team, t)[...] += v[:, :, 0, 0][:, :, None, None] * self.weights[None]

    # ---- mass leaving a pre-snap state ----
    def _push_aug(self, team: str, t: int, phase: str, G: np.ndarray) -> None:
        """Transpose of `Solution._aug`: mass on outcome columns (na, S, E1, E2)."""
        g = self.g
        self.add_post(team, t, phase, "v", G[: g.nf])
        opp = G[g.nf : g.nf + g.n_yl]
        if opp.any():
            self.add_post(self.other(team), t, phase, "v", np.ascontiguousarray(flip(opp, phase)),
                          rows=self.f10)  # fmt: skip
        if G[g.col_td].any():
            self.add_post(team, t, phase, "td", G[g.col_td])
        if G[g.col_safety].any():
            self.add_post(team, t, phase, "saf", G[g.col_safety])
        if G[g.col_opp_td].any():
            self.add_post(self.other(team), t, phase, "td",
                          np.ascontiguousarray(flip(G[g.col_opp_td], phase)))  # fmt: skip

    def _push_running(self, team: str, t: int, phase: str, mode: str, R: np.ndarray) -> None:
        """Transpose of `Solution._running_values` with post="coach": mass of plays that
        ended in bounds with the clock running, shape (nf, S, E1, E2)."""
        c = self.c
        if phase != "end":
            for h, p in enumerate(c.neutral_hist, start=1):
                if p > 0:
                    self.add_post(team, max(t - h, 0), phase, "v", p * R)
            return
        rate = c.def_rate_after if c.late(t) else c.def_rate_before
        q = np.zeros((self.m.n_score, len(rate)), dtype=self.dtype)
        q[self.scores > 0] = rate
        R = R.copy()
        v1 = self._d(team, t - 1)
        stop = R[..., 1:] * q[None, :, None, 1:]
        v1[..., :-1] += stop
        R[..., 1:] -= stop
        if mode == "chase" and c.late(t):
            v1[:, :, :-1, :] += R[:, :, 1:, :]
            R[:, :, 1:, :] = 0.0
        sol = mg.Solution.next_tick
        if mode == "protect":
            self._d(team, sol(self, t, c.burn_ticks))[...] += R
            return
        for h, p in enumerate(c.tempo_hist(mode, t), start=1):
            if p > 0:
                self._d(team, sol(self, t, h))[...] += p * R

    # `Solution.next_tick` reads only `model`, so the forward pass can borrow it.
    @property
    def model(self) -> mg.GameModel:
        return self.m

    def _over(self, t: int) -> np.ndarray | None:
        c = self.c
        if self.m.phase(t) != "end" or c.kneel_ticks is None or c.endgame_ticks == 0:
            return None
        thr = c.kneel_ticks[:, self.downs - 1].T  # (nf, B)
        return ((t <= thr)[:, None, None, :] & (self.scores > 0)[None, :, None, None]
                & np.ones(self.shape("end"), dtype=bool))  # fmt: skip

    def _allowed(self, a: int, t: int, phase: str) -> np.ndarray | bool:
        k = self.k
        sc = self.scores[None, :, None, None]
        if a == mg.A_NEUTRAL:
            return True
        if a in (mg.A_PROTECT, mg.A_CHASE):
            if phase != "end":
                return False
            return sc >= 0 if a == mg.A_PROTECT else sc <= 0
        ok = {mg.A_FG: k.fg_ok, mg.A_PUNT: k.punt_ok, mg.A_KNEEL: k.kneel_ok}.get(a)
        return False if ok is None else ok[:, None, None, None]

    def _step(self, team: str, t: int) -> None:
        mass = self.D[team].pop(t, None)
        if mass is None:
            return
        term = self.terminal[team]
        if t == 0:
            by = mass.sum(axis=(0, 2, 3))
            term["win"] += float(by[self.scores > 0].sum())
            term["tie"] += float(by[self.scores == 0].sum())
            term["loss"] += float(by[self.scores < 0].sum())
            return
        m, g, k, c = self.m, self.g, self.k, self.c
        ph = m.phase(t)
        over = self._over(t)
        if over is not None:
            term["kneel_out"] += float(mass[over].sum())
            mass = np.where(over, 0.0, mass)
        for hook in self.hooks:
            hook(team, t, mass)
        probs = self.policies[team](t, {})
        shape = mass.shape
        flat = shape[1] * shape[2] * shape[3]
        total = None
        for a, p in probs.items():
            p = np.asarray(p, dtype=self.dtype)
            if not p.any():
                continue
            M = np.broadcast_to(p, shape) * mass
            if not M.any():
                continue
            total = M.sum() if total is None else total + M.sum()
            ok = self._allowed(a, t, ph)
            if ok is not True and (np.where(ok, 0.0, M) > 0).any():
                name = mg.ACTION_NAMES[a]
                raise ValueError(f"policy puts mass on {name} where it is not allowed")
            Mf = M.reshape(g.nf, flat)
            if a in (mg.A_NEUTRAL, mg.A_PROTECT, mg.A_CHASE):
                mode = {mg.A_NEUTRAL: "neutral", mg.A_PROTECT: "protect", mg.A_CHASE: "chase"}[a]
                key = c.kernel_key(mode, t)
                G = self.kall_t[key] @ Mf
                R = self.kin_t[key] @ Mf
                G[: g.nf] -= R
                self._push_aug(team, t - 1, ph, G.reshape(g.na, *shape[1:]))
                self._push_running(team, t, ph, mode, R.reshape(shape))
            elif a == mg.A_FG:
                make = k.fg_make.astype(self.dtype)[:, None]
                self.add_post(team, t - 1, ph, "fg", (make * Mf).sum(0).reshape(shape[1:]))
                G = np.zeros((g.na, flat), dtype=self.dtype)
                np.add.at(G, k.fg_miss_col, (1.0 - make) * Mf)
                self._push_aug(team, t - 1, ph, G.reshape(g.na, *shape[1:]))
            elif a == mg.A_PUNT:
                G = self.kpunt_t @ Mf
                self._push_aug(team, max(t - c.punt_ticks, 0), ph, G.reshape(g.na, *shape[1:]))
            elif a == mg.A_KNEEL:
                keep = k.kneel_keep
                R = np.zeros((g.nf, flat), dtype=self.dtype)
                np.add.at(R, k.kneel_col[keep], Mf[keep])
                self._push_running(team, t, ph, "protect" if ph == "end" else "neutral",
                                   R.reshape(shape))  # fmt: skip
                G = np.zeros((g.na, flat), dtype=self.dtype)
                np.add.at(G, k.kneel_col[~keep], Mf[~keep])
                self._push_aug(team, t - 1, ph, G.reshape(g.na, *shape[1:]))
            else:
                raise ValueError(f"policy puts mass on {mg.ACTION_NAMES[a]}")
        live = float(mass.sum())
        if live > 0 and abs(float(total or 0.0) - live) > 1e-6 * max(live, 1.0):
            raise ValueError(f"policy probabilities do not sum to 1 at tick {t}")

    def start_coin_flip(self) -> None:
        """Opening kickoff, tied, receiver chosen by a coin flip (flag 0: it kicks off the
        second half)."""
        T = self.c.n_ticks
        s0 = self.m.score_max
        for team in TEAMS:
            d = self._d(team, T)
            d[self.f10, s0, 0, 0] += 0.5 * self.k.kickoff[1:].astype(self.dtype)

    def run(self, progress: bool = False) -> ForwardPass:
        if not any(self.D[x] for x in TEAMS):
            self.start_coin_flip()
        t0 = time.time()
        T = self.c.n_ticks
        for t in range(T, -1, -1):
            self._resolve(t)
            for team in TEAMS:
                self._step(team, t)
            if progress and (t % 60 == 0):
                print(f"  forward t = {t:4d}   {time.time() - t0:7.1f} s", flush=True)
        left = sum(float(x.sum()) for team in TEAMS for x in self.D[team].values())
        if left > 1e-9 or any(self.K[x] for x in TEAMS):
            raise RuntimeError("mass left over after the forward pass")
        self.seconds = time.time() - t0
        self.peak_rss_gb = peak_rss_gb()
        return self

    @property
    def total_mass(self) -> float:
        return sum(v for team in TEAMS for v in self.terminal[team].values())

    @property
    def result_a(self) -> float:
        """Win probability of team A (a tie counts half; a kneel-out is a win for the
        team in front)."""
        a, b = self.terminal["A"], self.terminal["B"]
        return (a["win"] + a["kneel_out"] + 0.5 * a["tie"] + b["loss"] + 0.5 * b["tie"])
