"""Tests for the M4 decision processes.

Everything above the `data` section builds tiny games by hand and runs without the
play-by-play cache or the solved arrays. The reference solver `Brute` is a separate,
slow, recursive implementation of the rules written straight from the model
description; the fast solver must agree with it state by state.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
import scipy.sparse as sp

from fourth_and_rl import mdp_game as mg
from fourth_and_rl.mdp import KERNEL_KEYS, FieldKernels, Grid, coach_situation
from fourth_and_rl.mdp_drive import solve_drive

# --------------------------------------------------------------------------------------
# Builders for tiny games
# --------------------------------------------------------------------------------------


def _csr(entries: dict[tuple[int, int], float], shape: tuple[int, int]) -> sp.csr_matrix:
    r = [k[0] for k in entries]
    c = [k[1] for k in entries]
    return sp.coo_matrix((list(entries.values()), (r, c)), shape=shape).tocsr()


def kernels_from_rules(grid: Grid, rules: dict, stop: dict[str, float]) -> FieldKernels:
    """Kernels for a toy game. rules[f] is a list of (probability, outcome) for running a
    play in field state f, with outcome ("keep", f2) | ("opp", yl) | ("td",) | ("safety",)
    | ("opp_td",). `stop[mode]` is the chance a kept-ball play leaves the clock stopped."""
    nf, na = grid.nf, grid.na
    k_all, k_in = {}, {}
    for mode in KERNEL_KEYS:
        ea: dict[tuple[int, int], float] = {}
        ei: dict[tuple[int, int], float] = {}
        for f, outs in rules.items():
            for p, o in outs:
                col = {
                    "keep": lambda o: o[1],
                    "opp": lambda o: grid.col_opp(o[1]),
                    "td": lambda o: grid.col_td,
                    "safety": lambda o: grid.col_safety,
                    "opp_td": lambda o: grid.col_opp_td,
                }[o[0]](o)
                ea[(f, int(col))] = ea.get((f, int(col)), 0.0) + p
                if o[0] == "keep" and stop[mode] < 1.0:
                    ei[(f, int(col))] = ei.get((f, int(col)), 0.0) + p * (1.0 - stop[mode])
        k_all[mode] = _csr(ea, (nf, na))
        k_in[mode] = _csr(ei, (nf, nf)) if ei else sp.csr_matrix((nf, nf))
    kick = np.zeros(grid.n_yl + 1)
    kick[(grid.n_yl + 1) // 2] = 1.0
    none = np.zeros(nf, dtype=bool)
    return FieldKernels(
        grid=grid,
        k_all=k_all,
        k_in=k_in,
        fg_make=np.zeros(nf),
        fg_miss_col=np.full(nf, grid.col_opp(1)),
        fg_ok=none.copy(),
        k_punt=sp.csr_matrix((nf, na)),
        punt_ok=none.copy(),
        kneel_col=np.zeros(nf, dtype=np.int64),
        kneel_keep=none.copy(),
        kneel_ok=none.copy(),
        kickoff=kick,
        free_kick=kick.copy(),
    )


def toy_football(seed: int = 0, n_timeouts: int = 2, defense: str = "optimal") -> mg.GameModel:
    """A small random football-like game that exercises every rule of the solver: three
    downs, a short field, gains and losses, turnovers, safeties, field goals, punts,
    kneels, timeouts, the warning, the 5:00 boundary and halftime."""
    rng = np.random.default_rng(seed)
    grid = Grid(n_down=3, n_yl=5, dist_exact=2, dist_bins=((3, 9, 3),))
    f10 = grid.first10()
    rules: dict[int, list] = {}
    for f in range(grid.nf):
        down, dbin, yl = (int(x) for x in grid.unpack(f))
        d = min(int(grid.dist_rep(dbin)), yl)
        p = rng.dirichlet(np.ones(6))
        outs = [(p[4] * 0.6, ("opp", grid.n_yl + 1 - yl)), (p[4] * 0.4, ("opp_td",))]
        for prob, gain in zip(p[:4], (-1, 0, 1, 2), strict=True):
            new = yl - gain
            if new <= 0:
                outs.append((prob, ("td",)))
            elif new > grid.n_yl:
                outs.append((prob, ("safety",)))
            elif gain >= d:
                outs.append((prob, ("keep", int(f10[new]))))
            elif down == grid.n_down:
                outs.append((prob, ("opp", grid.n_yl + 1 - new)))
            else:
                outs.append(
                    (prob, ("keep", int(grid.fidx(down + 1, grid.dist_bin(d - gain), new))))
                )
        outs.append((p[5], ("td",) if yl <= 2 else ("keep", int(f10[max(yl - 3, 1)]))))
        rules[f] = outs
    kern = kernels_from_rules(
        grid,
        rules,
        {"neutral": 0.4, "neutral_end": 0.5, "protect": 0.2, "chase": 0.7, "chase_late": 0.85},
    )
    # Make the protect table a different game from the others, as in the real model.
    shift = sp.diags(rng.uniform(0.6, 1.0, grid.nf))
    kern.k_in["protect"] = (shift @ kern.k_in["protect"]).tocsr()
    downs, dbins, yls = grid.unpack(np.arange(grid.nf))
    kern.fg_make = np.clip(0.95 - 0.2 * (yls - 1), 0.0, 1.0)
    kern.fg_miss_col = np.asarray(grid.col_opp(grid.n_yl + 1 - yls))
    kern.fg_ok = np.ones(grid.nf, dtype=bool)
    pe: dict[tuple[int, int], float] = {}
    for f in np.nonzero(downs == grid.n_down)[0]:
        yl = int(yls[f])
        land = min(grid.n_yl, grid.n_yl + 1 - yl + 3)
        pe[(int(f), int(grid.col_opp(land)))] = 0.7
        pe[(int(f), int(grid.col_opp(max(land - 2, 1))))] = 0.25
        pe[(int(f), grid.col_opp_td)] = 0.03
        pe[(int(f), int(f10[yl]))] = 0.02
    kern.k_punt = _csr(pe, (grid.nf, grid.na))
    kern.punt_ok = downs == grid.n_down
    back = np.minimum(yls + 1, grid.n_yl)
    kern.kneel_keep = downs < grid.n_down
    d_rep = np.minimum(grid.dist_rep(dbins), yls)
    kern.kneel_col = np.where(
        kern.kneel_keep,
        grid.fidx(np.minimum(downs + 1, grid.n_down), grid.dist_bin(d_rep + 1), back),
        grid.col_opp(grid.n_yl + 1 - back),
    )
    kern.kneel_ok = np.ones(grid.nf, dtype=bool)
    kern.kickoff = np.array([0.0, 0.0, 0.1, 0.3, 0.4, 0.2])
    kern.free_kick = np.array([0.0, 0.1, 0.2, 0.4, 0.3, 0.0])
    n = n_timeouts + 1
    shares = rng.dirichlet(np.ones(n), size=2)
    clock = mg.ClockSpec(
        n_ticks=20,
        tick_seconds=5,
        endgame_ticks=9,
        half_tick=14,
        warning_tick=4,
        burn_ticks=3,
        punt_ticks=2,
        n_timeouts=n_timeouts,
        neutral_hist=np.array([0.2, 0.5, 0.3]),
        hurry_hist=np.array([0.5, 0.5]),
        hurry_hist_late=np.array([0.8, 0.2]),
        kneel_ticks=mg.kneel_threshold_ticks(n_timeouts, 3, 5, 2.0, 10.0, 20.0),
        shares_leading=shares[0],
        shares_trailing=shares[1],
        defense_timeout=defense,
        def_rate_before=np.linspace(0.0, 0.6, n),
        def_rate_after=np.linspace(0.0, 0.9, n),
    )
    return mg.GameModel(
        kern=kern,
        clock=clock,
        score_max=4,
        td_points=2,
        fg_points=1,
        safety_points=1,
        p_xp=0.8,
        p_two=0.45,
        two_point_choice=True,
        dtype="float64",
    )


class Brute:
    """Slow recursive reference solver, written from the rules and sharing no code with
    `Solution.backup`."""

    def __init__(self, model: mg.GameModel) -> None:
        self.m, self.c, self.k, self.g = model, model.clock, model.kern, model.kern.grid
        self.f10 = self.g.first10()
        self.all = {m: model.kern.k_all[m].tocsr() for m in KERNEL_KEYS}
        self.inb = {m: model.kern.k_in[m].tocsr() for m in KERNEL_KEYS}
        self.punt = model.kern.k_punt.tocsr()
        self.v = lru_cache(maxsize=None)(self._v)
        self.halftime = lru_cache(maxsize=None)(self._halftime)

    def clip(self, s: int) -> int:
        return max(-self.m.score_max, min(self.m.score_max, s))

    @staticmethod
    def row(mat: sp.csr_matrix, f: int) -> list[tuple[int, float]]:
        return list(
            zip(
                mat.indices[mat.indptr[f] : mat.indptr[f + 1]].tolist(),
                mat.data[mat.indptr[f] : mat.indptr[f + 1]].tolist(),
                strict=True,
            )
        )

    def swap(self, e: tuple[int, int], phase: str) -> tuple[int, int]:
        if phase == "end":
            return (e[1], e[0])
        return (1 - e[0], 0) if phase == "first" else e

    def nxt(self, t: int, k: int) -> int:
        t2 = max(t - k, 0)
        w = self.c.warning_tick
        if w is not None and t <= self.c.endgame_ticks and t > w and t2 < w:
            return w
        return t2

    def weights(self, s: int) -> np.ndarray:
        lead, trail = self.c.shares_leading, self.c.shares_trailing
        if lead is None:
            return np.ones((1, 1))
        mean = 0.5 * (lead + trail)
        if s > 0:
            return np.outer(lead, trail)
        return np.outer(trail, lead) if s < 0 else np.outer(mean, mean)

    def hold(self, t: int, f: int, s: int, e: tuple[int, int], phase: str) -> float:
        """Value of having the ball in field state f at time t, seen from a play that
        started in `phase` with extra state e."""
        own = self.m.phase(t)
        if own == phase:
            return self.v(t, f, s, e[0], e[1])
        if phase == "mid":
            w = self.weights(s)
            return sum(
                w[a, b] * self.v(t, f, s, a, b)
                for a in range(w.shape[0])
                for b in range(w.shape[1])
            )
        return self.halftime(s, e[0])

    def _halftime(self, s: int, r: int) -> float:
        th = self.c.th

        def receive(score: int) -> float:
            return sum(
                p * self.hold(th, int(self.f10[y]), score, (0, 0), "mid")
                for y, p in enumerate(self.k.kickoff)
                if p > 0
            )

        return receive(s) if r == 1 else 1.0 - receive(-s)

    def receive(self, t: int, s: int, e: tuple[int, int], phase: str, dist: np.ndarray) -> float:
        if phase == "first" and t <= self.c.th:
            return self.halftime(s, e[0])
        return sum(
            p * self.hold(t, int(self.f10[y]), s, e, phase) for y, p in enumerate(dist) if p > 0
        )

    def after_td(self, t: int, s: int, e: tuple[int, int], phase: str) -> float:
        m = self.m

        def scored(points: int) -> float:
            return 1.0 - self.receive(
                t, -self.clip(s + points), self.swap(e, phase), phase, self.k.kickoff
            )

        xp = m.p_xp * scored(m.td_points + 1) + (1 - m.p_xp) * scored(m.td_points)
        two = m.p_two * scored(m.td_points + 2) + (1 - m.p_two) * scored(m.td_points)
        return max(xp, two) if m.two_point_choice else xp

    def outcome(self, col: int, t: int, s: int, e: tuple[int, int], phase: str) -> float:
        g, m = self.g, self.m
        if col < g.nf:
            return self.hold(t, col, s, e, phase)
        if col < g.nf + g.n_yl:
            return 1.0 - self.hold(t, int(self.f10[col - g.nf + 1]), -s, self.swap(e, phase), phase)
        if col == g.col_td:
            return self.after_td(t, s, e, phase)
        if col == g.col_safety:
            return 1.0 - self.receive(
                t, -self.clip(s - m.safety_points), self.swap(e, phase), phase, self.k.free_kick
            )
        return 1.0 - self.after_td(t, -s, self.swap(e, phase), phase)

    def running(self, t: int, f: int, s: int, e: tuple[int, int], phase: str, mode: str) -> float:
        c = self.c
        if phase != "end":
            return sum(
                p * self.hold(max(t - h, 0), f, s, e, phase)
                for h, p in enumerate(c.neutral_hist, start=1)
            )
        a, b = e
        if mode == "protect":
            base = self.v(self.nxt(t, c.burn_ticks), f, s, a, b)
        else:
            hist = c.neutral_hist
            if mode == "chase":
                hist = c.hurry_hist_late if t <= c.warning_tick else c.hurry_hist
            base = sum(p * self.v(self.nxt(t, h), f, s, a, b) for h, p in enumerate(hist, start=1))
            if a > 0:
                base = max(base, self.v(t - 1, f, s, a - 1, b))
        if b > 0:
            stop = self.v(t - 1, f, s, a, b - 1)
            if c.defense_timeout == "optimal":
                base = min(base, stop)
            else:
                rate = c.def_rate_after if t <= c.warning_tick else c.def_rate_before
                q = rate[b] if s > 0 else 0.0
                base = q * stop + (1 - q) * base
        return base

    def _v(self, t: int, f: int, s: int, e1: int, e2: int) -> float:
        if t == 0:
            return 1.0 if s > 0 else (0.0 if s < 0 else 0.5)
        m, c, k, g = self.m, self.c, self.k, self.g
        phase, e = m.phase(t), (e1, e2)
        down = int(g.unpack(f)[0])
        if phase == "end" and s > 0 and t <= c.kneel_ticks[e2, down - 1]:
            return 1.0
        best = -1.0
        if phase != "end":
            options = [("neutral", "neutral")]
        else:
            options = [("neutral", "neutral_end")]
            if s >= 0:
                options.append(("protect", "protect"))
            if s <= 0:
                options.append(("chase", "chase_late" if t <= c.warning_tick else "chase"))
        for mode, key in options:
            q = sum(
                p * self.outcome(col, t - 1, s, e, phase) for col, p in self.row(self.all[key], f)
            )
            q += sum(
                p * (self.running(t, col, s, e, phase, mode) - self.hold(t - 1, col, s, e, phase))
                for col, p in self.row(self.inb[key], f)
            )
            best = max(best, q)
        if k.fg_ok[f]:
            made = 1.0 - self.receive(
                t - 1, -self.clip(s + m.fg_points), self.swap(e, phase), phase, k.kickoff
            )
            miss = self.outcome(int(k.fg_miss_col[f]), t - 1, s, e, phase)
            best = max(best, k.fg_make[f] * made + (1 - k.fg_make[f]) * miss)
        if k.punt_ok[f]:
            t2 = max(t - c.punt_ticks, 0)
            best = max(
                best,
                sum(p * self.outcome(col, t2, s, e, phase) for col, p in self.row(self.punt, f)),
            )
        if k.kneel_ok[f]:
            col = int(k.kneel_col[f])
            if k.kneel_keep[f]:
                mode = "protect" if phase == "end" else "neutral"
                best = max(best, self.running(t, col, s, e, phase, mode))
            else:
                best = max(best, self.outcome(col, t - 1, s, e, phase))
        return best


def all_values(sol: mg.Solution) -> list[tuple[tuple, float]]:
    m = sol.model
    out = []
    for ph, arr in sol.v.items():
        lo = m.phase_range(ph)[0]
        for ix in np.ndindex(arr.shape):
            out.append(((ix[0] + lo, ix[1], ix[2] - m.score_max, ix[3], ix[4]), float(arr[ix])))
    return out


# --------------------------------------------------------------------------------------
# 1. A game small enough to solve by hand
# --------------------------------------------------------------------------------------


def hand_game(n_ticks: int = 2) -> mg.GameModel:
    """Two yard lines, one down. From the 1: touchdown (1 point) half the time, otherwise
    the other team takes over at its 2. From the 2: gain a yard half the time, otherwise
    the other team takes over at its 1. Kickoffs go to the 2. Every play takes one tick."""
    grid = Grid(n_down=1, n_yl=2, dist_exact=1, dist_bins=())
    rules = {0: [(0.5, ("td",)), (0.5, ("opp", 2))], 1: [(0.5, ("keep", 0)), (0.5, ("opp", 1))]}
    kern = kernels_from_rules(grid, rules, {m: 1.0 for m in KERNEL_KEYS})
    kern.kickoff = np.array([0.0, 0.0, 1.0])
    kern.free_kick = kern.kickoff.copy()
    return mg.GameModel(
        kern=kern,
        clock=mg.ClockSpec(n_ticks=n_ticks),
        score_max=3,
        td_points=1,
        p_xp=0.0,
        two_point_choice=False,
        dtype="float64",
    )


def test_hand_solved_game() -> None:
    sol = mg.solve(hand_game())
    v = sol.v["mid"]  # index: [t - 1, yard line - 1, score + 3, 0, 0]
    z = 3
    # One play left, tied. From the 1: score and win (1/2), or stay tied (1/2 * 1/2).
    assert v[0, 0, z, 0, 0] == pytest.approx(0.75)
    assert v[0, 1, z, 0, 0] == pytest.approx(0.5)  # cannot score from the 2
    assert v[0, 0, z - 1, 0, 0] == pytest.approx(0.25)  # down 1: a score only ties
    assert v[0, 1, z - 1, 0, 0] == pytest.approx(0.0)
    assert v[0, 0, z + 1, 0, 0] == pytest.approx(1.0)
    # Two plays left, tied, at the 1. Score (1/2): they get the ball at their 2 down one
    # with one play, value 0 for them, so 1 for us. Fail (1/2): they have the 2, tied, one
    # play: 0.5 for them, 0.5 for us. Total 0.75.
    assert v[1, 0, z, 0, 0] == pytest.approx(0.75)
    # At the 2: gain (1/2) -> the 1, tied, one play: 0.75. Lose it (1/2) -> they are at
    # their 1, tied, one play: 0.75 for them, 0.25 for us. Total 0.5.
    assert v[1, 1, z, 0, 0] == pytest.approx(0.5)
    # Down 1 at the 1: score (1/2) -> tied, they have the 2 with one play: 0.5 each.
    # Fail (1/2) -> they lead with the ball and one play: 0 for us. Total 0.25.
    assert v[1, 0, z - 1, 0, 0] == pytest.approx(0.25)


def test_hand_game_matches_brute_force() -> None:
    sol = mg.solve(hand_game(6))
    brute = Brute(sol.model)
    for (t, f, s, e1, e2), val in all_values(sol):
        assert val == pytest.approx(brute.v(t, f, s, e1, e2), abs=1e-12)


# --------------------------------------------------------------------------------------
# 2. Random walk football (arXiv 2406.16171, section 2.1)
# --------------------------------------------------------------------------------------


def random_walk_wp(length: int, n_plays: int) -> dict[tuple[int, int, int], float]:
    """The paper's recursion, written directly. WP(n, x, s): n plays left, ball at x in
    1..length-1, score differential s. The ball moves one yard either way with
    probability 1/2. Reaching 0 scores +1, reaching `length` scores -1, and the ball
    goes back to length/2. A tie at the end is a coin flip."""
    wp: dict[tuple[int, int, int], float] = {}
    for s in range(-n_plays, n_plays + 1):
        for x in range(1, length):
            wp[(0, x, s)] = 1.0 if s > 0 else (0.0 if s < 0 else 0.5)
    for n in range(1, n_plays + 1):
        for s in range(-(n_plays - n), n_plays - n + 1):
            for x in range(1, length):
                left = (n - 1, length // 2, s + 1) if x - 1 == 0 else (n - 1, x - 1, s)
                right = (n - 1, length // 2, s - 1) if x + 1 == length else (n - 1, x + 1, s)
                wp[(n, x, s)] = 0.5 * wp[left] + 0.5 * wp[right]
    return wp


def random_walk_model(length: int, n_plays: int) -> mg.GameModel:
    """The same game in the solver's language: one down, the 'offense' is the team
    attacking yard line 0. A score hands the ball to the other team at midfield, which by
    the symmetry of the walk is the same state seen from the other side."""
    grid = Grid(n_down=1, n_yl=length - 1, dist_exact=1, dist_bins=())
    rules = {}
    for x in range(1, length):
        left = ("td",) if x - 1 == 0 else ("keep", x - 2)
        right = ("safety",) if x + 1 == length else ("keep", x)
        rules[x - 1] = [(0.5, left), (0.5, right)]
    kern = kernels_from_rules(grid, rules, {m: 1.0 for m in KERNEL_KEYS})
    return mg.GameModel(
        kern=kern,
        clock=mg.ClockSpec(n_ticks=n_plays),
        score_max=n_plays,
        td_points=1,
        safety_points=1,
        p_xp=0.0,
        two_point_choice=False,
        dtype="float64",
    )


@pytest.mark.parametrize(("length", "n_plays"), [(4, 56), (6, 30)])
def test_random_walk_football(length: int, n_plays: int) -> None:
    truth = random_walk_wp(length, n_plays)
    sol = mg.solve(random_walk_model(length, n_plays))
    v = sol.v["mid"]
    assert np.allclose(sol.v["end"][0, 0, :, 0, 0], sol.terminal)
    worst = 0.0
    for (n, x, s), w in truth.items():
        if n >= 1:
            worst = max(worst, abs(float(v[n - 1, x - 1, s + n_plays, 0, 0]) - w))
    assert worst < 1e-12
    # The paper's setting: midfield, tied, 56 plays left is an even game.
    if length == 4:
        assert v[55, 1, n_plays, 0, 0] == pytest.approx(0.5)
        assert truth[(20, 1, 0)] > 0.5  # closer to scoring is better


# --------------------------------------------------------------------------------------
# 3. Every rule at once, against the reference solver; invariants
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy() -> mg.Solution:
    return mg.solve(toy_football(seed=1))


@pytest.mark.parametrize(("seed", "defense"), [(1, "optimal"), (2, "empirical")])
def test_toy_football_matches_brute_force(seed: int, defense: str) -> None:
    sol = mg.solve(toy_football(seed=seed, defense=defense))
    brute = Brute(sol.model)
    rng = np.random.default_rng(seed)
    states = all_values(sol)
    picks = rng.choice(len(states), size=1500, replace=False)
    worst = max(abs(states[i][1] - brute.v(*states[i][0])) for i in picks)
    assert worst < 1e-10


def test_invariants(toy: mg.Solution) -> None:
    m = toy.model
    for ph, v in toy.v.items():
        assert v.min() >= -1e-12 and v.max() <= 1 + 1e-12, ph
        # More points never hurt.
        assert (np.diff(v, axis=2) >= -1e-9).all(), ph
    end = toy.v["end"]
    assert np.allclose(end[0], toy.terminal[None, :, None, None])
    # Kneel rule: ahead, a down to spare, inside the threshold -> the game is won.
    g = toy.grid
    f = int(g.fidx(1, 0, 3))
    thr = m.clock.kneel_ticks
    for b in range(m.clock.n_timeouts + 1):
        t = int(thr[b, 0])
        if t >= 1:
            assert np.allclose(end[t, f, m.score_max + 1 :, :, b], 1.0)
            assert toy.pol["end"][t, f, m.score_max + 1, 0, b] == mg.A_OVER
    # A defensive timeout never helps the offense that is burning clock, and an extra
    # timeout never hurts the team that holds it.
    assert (np.diff(end, axis=3) >= -1e-9).all()
    assert (np.diff(end, axis=4) <= 1e-9).all()


def test_symmetry_of_the_two_sides() -> None:
    # In random walk football the two teams are mirror images: V(n, x, s) = 1 - V(n, L-x, -s).
    sol = mg.solve(random_walk_model(6, 20))
    v = sol.v["mid"][:, :, :, 0, 0]
    assert np.allclose(v, 1.0 - v[:, ::-1, ::-1])


def test_kneel_threshold_arithmetic() -> None:
    # T(n, m) = 40 * max(m - n, 0) + 2 * m capped at 120: first down 120, 86, 46, 6 seconds.
    thr = mg.kneel_threshold_ticks(3, 4, 5, 2.0, 40.0, 120.0)
    assert (thr[:, 0] * 5).tolist() == [120, 85, 45, 5]
    assert (thr[:, 1] * 5).tolist() == [80, 40, 0, 0]
    assert (thr[:, 2] * 5).tolist() == [40, 0, 0, 0]
    assert (thr[:, 3] == 0).all()


# --------------------------------------------------------------------------------------
# 4. Simulator: same transition model as the solver
# --------------------------------------------------------------------------------------


def test_monte_carlo_matches_values_on_toy(toy: mg.Solution) -> None:
    sim = mg.Simulator(toy)
    rng = np.random.default_rng(7)
    g, m = toy.grid, toy.model
    n = 40000
    cases = [
        dict(t=20, s=0, f=int(g.fidx(1, 1, 4)), r=1),
        dict(t=20, s=-1, f=int(g.fidx(2, 0, 2)), r=0),
        dict(t=13, s=1, f=int(g.fidx(1, 1, 3))),
        dict(t=9, s=-2, f=int(g.fidx(1, 1, 5)), a=2, b=1),
        dict(t=7, s=1, f=int(g.fidx(2, 2, 4)), a=0, b=2),
        dict(t=5, s=0, f=int(g.fidx(3, 0, 3)), a=1, b=1),
    ]
    for case in cases:
        st = sim.start(n, **case)
        expect = float(sim.values(st)[0])
        out = mg.simulate(sim, st, rng)
        se = max(float(out["result"].std()) / np.sqrt(n), 1e-4)
        assert abs(out["result"].mean() - expect) < 4.5 * se, (case, expect, out["result"].mean())
    assert m.n_states()["total"] == sum(v.size for v in toy.v.values())


def test_step_and_lookups(toy: mg.Solution) -> None:
    sim = mg.Simulator(toy)
    rng = np.random.default_rng(3)
    state = mg.GameState(
        seconds=40, score=-1, down=1, ydstogo=2, yardline_100=4, off_timeouts=2, def_timeouts=1
    )
    assert 0.0 <= toy.value(state) <= 1.0
    q = toy.q_values(state)
    assert toy.policy(state) in q
    assert max(q.values()) == pytest.approx(toy.value(state))
    seen_done = False
    for _ in range(200):
        s, total = state, 0
        for _ in range(100):
            nxt, info = mg.step(s, toy.policy(s), rng, sim)
            total += 1
            if info["done"]:
                assert info["result"] in (0.0, 0.5, 1.0)
                seen_done = True
                break
            assert nxt.seconds < s.seconds
            s = nxt
        assert total < 100
    assert seen_done


def test_drive_log_accounts_for_every_point(toy: mg.Solution) -> None:
    sim = mg.Simulator(toy)
    st = sim.start(3000, t=20, s=0, f=int(toy.grid.fidx(1, 1, 4)), r=1)
    out = mg.simulate(sim, st, np.random.default_rng(5), log_drives=True, checkpoints=[14, 9])
    d = out["drives"]
    own = np.zeros((3000, 2))
    np.add.at(own, (d["game"], d["side"]), d["points"])
    # Points on offense never exceed a team's total (the rest came from returns and safeties).
    assert (own <= out["points"]).all()
    assert set(np.unique(d["result"])) <= set(range(len(mg.DRIVE_RESULTS)))
    assert not np.isnan(out["checkpoint_margin"][9]).any()


# --------------------------------------------------------------------------------------
# 5. Stage 1: expected points over endless possessions
# --------------------------------------------------------------------------------------


def test_drive_value_by_hand() -> None:
    # One yard line, one down: touchdown worth 7 with probability p, otherwise the other
    # team gets the same chance. "Next score" value: V = 7p - (1 - p) V  =>  V = 7p / (2 - p).
    grid = Grid(n_down=1, n_yl=1, dist_exact=1, dist_bins=())
    kern = kernels_from_rules(
        grid, {0: [(0.5, ("td",)), (0.5, ("opp", 1))]}, {m: 1.0 for m in KERNEL_KEYS}
    )
    kern.kickoff = np.array([0.0, 1.0])
    kern.free_kick = kern.kickoff.copy()
    out = solve_drive(kern, td_value=7.0, after_score="absorb")
    assert out["converged"]
    assert out["v"][0] == pytest.approx(7 * 0.5 / 1.5)
    # With the kickoff counted the scorer hands over the same state: V = p(7 - V) - (1-p)V.
    out = solve_drive(kern, td_value=7.0, after_score="kickoff")
    assert out["v"][0] == pytest.approx(3.5 / 2.0)


def test_drive_policy_evaluation_solves_the_linear_system() -> None:
    kern = toy_football(seed=3).kern
    g = kern.grid
    nf = g.nf
    rng = np.random.default_rng(0)
    pol = rng.dirichlet(np.ones(3), size=nf)
    out = solve_drive(
        kern, td_value=2.8, fg_points=1.0, safety_points=1.0, fourth_policy=pol, tolerance=1e-13
    )
    assert out["converged"]
    # Independent check: V = r + M V with the fixed policy, solved directly.
    f10 = g.first10()[1:]
    lift = np.zeros((g.na, nf))  # outcome column -> coefficients on V
    lift[np.arange(nf), np.arange(nf)] = 1.0
    lift[g.nf + np.arange(g.n_yl), f10] -= 1.0
    kv = np.zeros(nf)
    np.add.at(kv, f10, kern.kickoff[1:])
    fv = np.zeros(nf)
    np.add.at(fv, f10, kern.free_kick[1:])
    const = np.zeros(g.na)
    lift[g.col_td], const[g.col_td] = -kv, 2.8
    lift[g.col_safety], const[g.col_safety] = -fv, -1.0
    lift[g.col_opp_td], const[g.col_opp_td] = kv, -2.8
    play = kern.k_all["neutral"].toarray()
    punt = kern.k_punt.toarray()
    fg_m = (
        kern.fg_make[:, None] * (-kv)[None, :]
        + (1 - kern.fg_make)[:, None] * lift[kern.fg_miss_col]
    )
    fg_c = kern.fg_make * 1.0 + (1 - kern.fg_make) * const[kern.fg_miss_col]
    fourth = g.unpack(np.arange(nf))[0] == g.n_down
    w = np.where(fourth[:, None], pol, np.array([1.0, 0.0, 0.0]))
    mat = w[:, [0]] * (play @ lift) + w[:, [1]] * fg_m + w[:, [2]] * (punt @ lift)
    rhs = w[:, 0] * (play @ const) + w[:, 1] * fg_c + w[:, 2] * (punt @ const)
    direct = np.linalg.solve(np.eye(nf) - mat, rhs)
    assert np.allclose(out["v"], direct, atol=1e-9)


def test_coach_situations_cover_the_clock() -> None:
    sec = np.array([3000, 1900, 1000, 800, 800, 250, 250, 100, 100, 200, 200])
    half = np.array([1200, 100, 1000, 800, 800, 250, 250, 100, 100, 200, 200])
    score = np.array([0, 0, -20, -5, -12, -2, -6, -3, -7, 4, 0])
    assert coach_situation(sec, half, score).tolist() == [0, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8]


# --------------------------------------------------------------------------------------
# 6. The real solved game (needs data/mdp, written by scripts/solve_mdp.py)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def solved() -> mg.Solution:
    if not mg.solution_exists():
        pytest.skip("solved game not built (run scripts/solve_mdp.py)")
    return mg.load_solution()


@pytest.mark.data
def test_real_solution_invariants(solved: mg.Solution) -> None:
    m, g = solved.model, solved.grid
    end = solved.v["end"]
    assert np.allclose(end[0], solved.terminal[None, :, None, None])
    for t in (1, 12, 24, 36, 48, 60):
        v = np.asarray(end[t])
        assert v.min() >= -1e-5 and v.max() <= 1 + 1e-5
        # More points never hurt, with one known exception of the model: a tied team may use
        # the chase tables and a team one point ahead may not, so a few states are worth a
        # hair more when tied (largest gap seen: 0.0008).
        d = np.diff(v, axis=1)
        assert d.min() >= -5e-3 and (d < -1e-4).mean() < 1e-4
    for name in ("mid", "first"):
        v = np.asarray(solved.v[name][::40])
        assert v.min() >= -1e-5 and v.max() <= 1 + 1e-5
        d = np.diff(v, axis=2)
        assert d.min() >= -5e-3 and (d < -1e-4).mean() < 1e-4
    # First down inside the kneel threshold with a lead: the game is over (data: 99.92%).
    for b, seconds in enumerate((120, 85, 45, 5)):
        state = mg.GameState(seconds, 4, 1, 10, 60, off_timeouts=1, def_timeouts=b)
        assert solved.value(state) == pytest.approx(1.0)
        assert solved.policy(state) == "game_over"
    # Just outside it the game is not over.
    assert solved.value(mg.GameState(125, 4, 1, 10, 60, def_timeouts=3)) < 1.0
    # Receiving the opening kickoff (the other team gets the second-half one) is close to even.
    opening = float(solved.post(m.clock.n_ticks, "first").ko[m.score_max, 0, 0])
    assert opening == pytest.approx(0.5, abs=0.03)
    assert g.nf == 4752


@pytest.mark.data
def test_real_solution_monte_carlo(solved: mg.Solution) -> None:
    sim = mg.Simulator(solved)
    rng = np.random.default_rng(11)
    g = solved.grid
    n = 6000
    cases = [
        dict(t=48, s=4, f=int(g.fidx(4, 1, 60)), a=3, b=3),
        dict(t=30, s=-4, f=int(g.fidx(1, 9, 75)), a=2, b=3),
        dict(t=60, s=0, f=int(g.fidx(2, 6, 45)), a=3, b=2),
        dict(t=200, s=-3, f=int(g.fidx(1, 9, 70))),
    ]
    for case in cases:
        st = sim.start(n, **case)
        expect = float(sim.values(st)[0])
        out = mg.simulate(sim, st, rng)
        se = max(float(out["result"].std()) / np.sqrt(n), 1e-3)
        assert abs(out["result"].mean() - expect) < 4.5 * se, (case, expect, out["result"].mean())
