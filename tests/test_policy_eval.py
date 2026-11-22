"""Tests for the exact fixed-policy evaluator (`fourth_and_rl.policy_eval`).

The toy game of `test_mdp.py` exercises every rule of the solver. `TwoTeamBrute` below is
a separate, slow, recursive evaluator of two fixed policies written from the rules, which
shares no code with `Solution.backup`; the fast evaluator must agree with it state by
state. Tests marked `data` use the real solved game in data/mdp.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from test_mdp import all_values, toy_football

from fourth_and_rl import mdp
from fourth_and_rl import mdp_game as mg
from fourth_and_rl import policy_eval as pe
from fourth_and_rl.mdp import KERNEL_KEYS

MODES = {"neutral": mg.A_NEUTRAL, "protect": mg.A_PROTECT, "chase": mg.A_CHASE}


def toy_coach(grid: mdp.Grid, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    table = rng.dirichlet(np.ones(3), size=(len(mdp.SITUATIONS), grid.n_dist, grid.n_yl))
    return {"table": table, "end_fg_max_yardline": 3, "end_fg_ticks": 2,
            "two_point_seconds": 60, "two_point_leads": [-1, 0, 1]}  # fmt: skip


def random_fourth(model: mg.GameModel, seed: int) -> dict[int, np.ndarray]:
    """A random (go, field goal, punt) distribution at every fourth down of every tick."""
    rng = np.random.default_rng(seed)
    g = model.kern.grid
    n4 = g.n_dist * g.n_yl
    out = {}
    for t in range(1, model.clock.n_ticks + 1):
        e = model.extra_shape(model.phase(t))
        out[t] = rng.dirichlet(np.ones(3) * 0.7, size=(n4, model.n_score, *e))
    return out


class TwoTeamBrute:
    """Slow recursive value of two fixed policies, written from the rules. team 0 is A,
    team 1 is B; v(team, t, f, s, e1, e2) is the win probability of `team`, which has the
    ball. probs[team](t) -> {code: array broadcastable to the slice}. Post-decisions: the
    team with the ball takes its own timeout when that is worth more, the defense takes
    one when that is worth less to the offense (or at the empirical rates), and the scorer
    takes the better try. With post="coach": the simulator's coach rules instead."""

    def __init__(self, model: mg.GameModel, probs: list, post: str = "model",
                 coach: dict | None = None) -> None:  # fmt: skip
        self.post, self.coach = post, coach or {}
        self.m, self.c, self.k, self.g = model, model.clock, model.kern, model.kern.grid
        self.f10 = self.g.first10()
        self.all = {m: model.kern.k_all[m].tocsr() for m in KERNEL_KEYS}
        self.inb = {m: model.kern.k_in[m].tocsr() for m in KERNEL_KEYS}
        self.punt = model.kern.k_punt.tocsr()
        self.probs = probs
        self.v = lru_cache(maxsize=None)(self._v)
        self._p = lru_cache(maxsize=None)(self._probs_at)

    def _probs_at(self, team: int, t: int) -> dict:
        return {a: np.asarray(p, dtype=float) for a, p in self.probs[team](t).items()}

    def p(self, team: int, t: int, a: int, f: int, s: int, e: tuple[int, int]) -> float:
        arrs = self._p(team, t)
        if a not in arrs:
            return 0.0
        x = arrs[a]
        ix = tuple(min(i, n - 1) for i, n in zip((f, s + self.m.score_max, *e), x.shape,
                                                 strict=True))  # fmt: skip
        return float(x[ix])

    def clip(self, s: int) -> int:
        return max(-self.m.score_max, min(self.m.score_max, s))

    @staticmethod
    def row(mat, f: int) -> list[tuple[int, float]]:
        lo, hi = mat.indptr[f], mat.indptr[f + 1]
        return list(zip(mat.indices[lo:hi].tolist(), mat.data[lo:hi].tolist(), strict=True))

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
        mean = 0.5 * (lead + trail)
        if s > 0:
            return np.outer(lead, trail)
        return np.outer(trail, lead) if s < 0 else np.outer(mean, mean)

    def hold(self, team: int, t: int, f: int, s: int, e: tuple[int, int], phase: str) -> float:
        own = self.m.phase(t)
        if own == phase:
            return self.v(team, t, f, s, e[0], e[1])
        if phase == "mid":
            w = self.weights(s)
            return sum(w[a, b] * self.v(team, t, f, s, a, b)
                       for a in range(w.shape[0]) for b in range(w.shape[1]))  # fmt: skip
        return self.halftime(team, s, e[0])

    def halftime(self, team: int, s: int, r: int) -> float:
        def receive(tm: int, score: int) -> float:
            return sum(p * self.hold(tm, self.c.th, int(self.f10[y]), score, (0, 0), "mid")
                       for y, p in enumerate(self.k.kickoff) if p > 0)  # fmt: skip

        return receive(team, s) if r == 1 else 1.0 - receive(1 - team, -s)

    def receive(self, team: int, t: int, s: int, e: tuple[int, int], phase: str,
                dist: np.ndarray) -> float:  # fmt: skip
        if phase == "first" and t <= self.c.th:
            return self.halftime(team, s, e[0])
        return sum(p * self.hold(team, t, int(self.f10[y]), s, e, phase)
                   for y, p in enumerate(dist) if p > 0)  # fmt: skip

    def after_td(self, team: int, t: int, s: int, e: tuple[int, int], phase: str) -> float:
        m = self.m

        def scored(points: int) -> float:
            return 1.0 - self.receive(1 - team, t, -self.clip(s + points), self.swap(e, phase),
                                      phase, self.k.kickoff)  # fmt: skip

        xp = m.p_xp * scored(m.td_points + 1) + (1 - m.p_xp) * scored(m.td_points)
        two = m.p_two * scored(m.td_points + 2) + (1 - m.p_two) * scored(m.td_points)
        if self.post == "coach":
            go = (t * self.c.tick_seconds <= self.coach["two_point_seconds"]
                  and s + m.td_points in self.coach["two_point_leads"])  # fmt: skip
            return two if go else xp
        return max(xp, two) if m.two_point_choice else xp

    def outcome(self, team: int, col: int, t: int, s: int, e: tuple[int, int],
                phase: str) -> float:  # fmt: skip
        g, m = self.g, self.m
        if col < g.nf:
            return self.hold(team, t, col, s, e, phase)
        if col < g.nf + g.n_yl:
            return 1.0 - self.hold(1 - team, t, int(self.f10[col - g.nf + 1]), -s,
                                   self.swap(e, phase), phase)  # fmt: skip
        if col == g.col_td:
            return self.after_td(team, t, s, e, phase)
        if col == g.col_safety:
            return 1.0 - self.receive(1 - team, t, -self.clip(s - m.safety_points),
                                      self.swap(e, phase), phase, self.k.free_kick)  # fmt: skip
        return 1.0 - self.after_td(1 - team, t, -s, self.swap(e, phase), phase)

    def running(self, team: int, t: int, f: int, s: int, e: tuple[int, int], phase: str,
                mode: str) -> float:  # fmt: skip
        c = self.c
        if phase != "end":
            return sum(p * self.hold(team, max(t - h, 0), f, s, e, phase)
                       for h, p in enumerate(c.neutral_hist, start=1))  # fmt: skip
        a, b = e
        if mode == "protect":
            base = self.v(team, self.nxt(t, c.burn_ticks), f, s, a, b)
        else:
            hist = c.neutral_hist
            if mode == "chase":
                hist = c.hurry_hist_late if t <= c.warning_tick else c.hurry_hist
            base = sum(p * self.v(team, self.nxt(t, h), f, s, a, b)
                       for h, p in enumerate(hist, start=1))  # fmt: skip
            if a > 0 and self.post == "model":
                base = max(base, self.v(team, t - 1, f, s, a - 1, b))
            if a > 0 and self.post == "coach" and mode == "chase" and t <= c.warning_tick:
                base = self.v(team, t - 1, f, s, a - 1, b)
        if b > 0:
            stop = self.v(team, t - 1, f, s, a, b - 1)
            if c.defense_timeout == "optimal" and self.post == "model":
                base = min(base, stop)
            else:
                rate = c.def_rate_after if t <= c.warning_tick else c.def_rate_before
                q = rate[b] if s > 0 else 0.0
                base = q * stop + (1 - q) * base
        return base

    def q(self, team: int, a: int, t: int, f: int, s: int, e: tuple[int, int]) -> float:
        m, c, k = self.m, self.c, self.k
        phase = m.phase(t)
        if a in MODES.values():
            mode = {v: n for n, v in MODES.items()}[a]
            key = c.kernel_key(mode, t)
            out = sum(p * self.outcome(team, col, t - 1, s, e, phase)
                      for col, p in self.row(self.all[key], f))  # fmt: skip
            out += sum(p * (self.running(team, t, col, s, e, phase, mode)
                            - self.hold(team, t - 1, col, s, e, phase))
                       for col, p in self.row(self.inb[key], f))  # fmt: skip
            return out
        if a == mg.A_FG:
            made = 1.0 - self.receive(1 - team, t - 1, -self.clip(s + m.fg_points),
                                      self.swap(e, phase), phase, k.kickoff)  # fmt: skip
            miss = self.outcome(team, int(k.fg_miss_col[f]), t - 1, s, e, phase)
            return k.fg_make[f] * made + (1 - k.fg_make[f]) * miss
        if a == mg.A_PUNT:
            t2 = max(t - c.punt_ticks, 0)
            return sum(p * self.outcome(team, col, t2, s, e, phase)
                       for col, p in self.row(self.punt, f))  # fmt: skip
        col = int(k.kneel_col[f])
        if k.kneel_keep[f]:
            return self.running(team, t, col, s, e, phase, "protect" if phase == "end" else
                                "neutral")  # fmt: skip
        return self.outcome(team, col, t - 1, s, e, phase)

    def _v(self, team: int, t: int, f: int, s: int, e1: int, e2: int) -> float:
        if t == 0:
            return 1.0 if s > 0 else (0.0 if s < 0 else 0.5)
        m, c, g = self.m, self.c, self.g
        phase, e = m.phase(t), (e1, e2)
        down = int(g.unpack(f)[0])
        if phase == "end" and s > 0 and t <= c.kneel_ticks[e2, down - 1]:
            return 1.0
        total = 0.0
        for a in range(6):
            p = self.p(team, t, a, f, s, e)
            if p > 0:
                total += p * self.q(team, a, t, f, s, e)
        return total


@pytest.fixture(scope="module")
def toy() -> mg.Solution:
    return mg.solve(toy_football(seed=1))


def test_solver_policy_reproduces_the_solver_bit_for_bit(toy: mg.Solution) -> None:
    for linked in (False, True):
        pol = pe.array_policy(toy.pol, toy.model)
        pair, _ = pe.evaluate(toy.model, pol, pol if linked else None, keep_slices=True)
        for ph, arr in toy.v.items():
            lo = toy.model.phase_range(ph)[0]
            for i in range(arr.shape[0]):
                assert np.array_equal(pair.slices[lo + i], arr[i]), (linked, ph, i)


def test_coach_against_coach_is_even_at_kickoff(toy: mg.Solution) -> None:
    m = toy.model
    coach = pe.CoachPolicy(m, toy_coach(toy.grid))
    sym, _ = pe.evaluate(m, coach, keep_slices=True)
    two, _ = pe.evaluate(m, coach, pe.CoachPolicy(m, toy_coach(toy.grid)), keep_slices=True)
    assert sym.start_value() == pytest.approx(0.5, abs=1e-12)
    assert two.start_value() == pytest.approx(0.5, abs=1e-12)
    for t in sym.slices:
        assert np.allclose(sym.slices[t], two.slices[t], atol=1e-13, rtol=0)
    # A best response to the coach (A maximizes its own action values, B plays the coach)
    # is worth at least as much as the coach in every state.
    def best(t: int, q: dict) -> dict:
        top = np.max(np.stack(list(q.values())), axis=0)
        hit = {a: x == top for a, x in q.items()}
        n = sum(h.astype(float) for h in hit.values())
        return {a: h / n for a, h in hit.items()}

    br, _ = pe.evaluate(m, best, coach, keep_slices=True)
    for t in sym.slices:
        assert (br.slices[t] >= sym.slices[t] - 1e-12).all()
    assert br.start_value() > 0.5


@pytest.mark.parametrize(("seed", "defense"), [(1, "optimal"), (2, "empirical")])
def test_matches_the_reference_evaluator_on_the_toy_game(seed: int, defense: str) -> None:
    m = toy_football(seed=seed, defense=defense)
    g = m.kern.grid
    coach = pe.CoachPolicy(m, toy_coach(g, seed))
    rand = random_fourth(m, seed)
    pol_a = coach.with_fourth(lambda t, q: rand[t])
    pair, _ = pe.evaluate(m, pol_a, coach, keep_slices=True)
    brute = TwoTeamBrute(m, [lambda t: pol_a(t, {}), lambda t: coach(t, {})])
    shell = mg.Solution(m)
    states = all_values(shell)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for i in rng.choice(len(states), size=1200, replace=False):
        (t, f, s, e1, e2), _ = states[i]
        exact = float(pair.slices[t][f, s + m.score_max, e1, e2])
        worst = max(worst, abs(exact - brute.v(0, t, f, s, e1, e2)))
    assert worst < 1e-10
    # Team B's side too, through the start value: coin flip over who receives.
    ka = sum(p * brute.v(0, m.clock.n_ticks, int(g.first10()[y]), 0, 0, 0)
             for y, p in enumerate(m.kern.kickoff) if p > 0)  # fmt: skip
    kb = sum(p * brute.v(1, m.clock.n_ticks, int(g.first10()[y]), 0, 0, 0)
             for y, p in enumerate(m.kern.kickoff) if p > 0)  # fmt: skip
    assert pair.start_value() == pytest.approx(0.5 * ka + 0.5 * (1 - kb), abs=1e-10)


class _FixedUniforms:
    """Stands in for a Generator: random(n) returns evenly spaced uniforms."""

    def __init__(self, k: int) -> None:
        self.k = k

    def random(self, n: int) -> np.ndarray:
        return np.full(n, (self.k + 0.5) / 1000.0)


def test_coach_probabilities_match_the_simulator(toy: mg.Solution) -> None:
    m, g = toy.model, toy.grid
    table = toy_coach(g, 3)
    coach = pe.CoachPolicy(m, table)
    sim = mg.Simulator(toy, table)
    for t in (20, 15, 14, 12, 9, 6, 4, 2, 1):
        f, s = np.meshgrid(np.arange(g.nf), np.arange(-m.score_max, m.score_max + 1),
                           indexing="ij")  # fmt: skip
        n = f.size
        st = sim.start(n, t=t, s=s.ravel(), f=f.ravel())
        idx = np.arange(n)
        counts = np.zeros((6, n))
        for k in range(1000):
            act = sim.coach_actions(st, idx, _FixedUniforms(k))
            counts[act, idx] += 1
        probs = coach.probs(t)
        for a in range(6):
            want = np.broadcast_to(probs.get(a, np.zeros((1, 1, 1, 1))), (g.nf, m.n_score, 1, 1))
            assert np.abs(counts[a] / 1000 - want[..., 0, 0].ravel()).max() <= 1.5e-3, (t, a)


def test_policy_mass_on_a_disallowed_action_raises(toy: mg.Solution) -> None:
    m = toy.model

    def bad(t: int, q: dict) -> dict:
        return {mg.A_PUNT: np.ones((1, 1, 1, 1))}  # punts are allowed on the last down only

    with pytest.raises(ValueError, match="not allowed"):
        pe.evaluate(m, bad)


def test_ring_buffer_keeps_only_recent_slices(toy: mg.Solution) -> None:
    m = toy.model
    pol = pe.array_policy(toy.pol, m)
    pair = pe.PolicyPair(m, pol)
    pe.run([pair])
    kept = set(pair.a.buf)
    assert kept <= set(range(m.clock.n_ticks - pair.a.lag - 1, m.clock.n_ticks + 1)) | {
        0, m.clock.th}  # fmt: skip
    with pytest.raises(KeyError):
        pair.a.native(1)


# --------------------------------------------------------------------------------------
# Post-decisions fixed to the coach's rules (Solution(post="coach"))
# --------------------------------------------------------------------------------------


def test_default_post_decisions_are_unchanged(toy: mg.Solution) -> None:
    explicit = mg.Solution(toy.model, post="model")
    mg.backward_induction([explicit])
    for ph in toy.v:
        assert np.array_equal(explicit.v[ph], toy.v[ph])
        assert np.array_equal(explicit.pol[ph], toy.pol[ph])
    with pytest.raises(ValueError, match="post must be"):
        mg.Solution(toy.model, post="other")


@pytest.mark.parametrize(("seed", "defense"), [(1, "optimal"), (2, "empirical")])
def test_coach_post_decisions_match_the_reference(seed: int, defense: str) -> None:
    m = toy_football(seed=seed, defense=defense)
    g = m.kern.grid
    table = toy_coach(g, seed)
    coach = pe.CoachPolicy(m, table)
    rand = random_fourth(m, seed + 10)
    pol_a = coach.with_fourth(lambda t, q: rand[t])
    pair, _ = pe.evaluate(m, pol_a, coach, keep_slices=True, post="coach", coach=table)
    model_pair, _ = pe.evaluate(m, pol_a, coach, keep_slices=True)
    brute = TwoTeamBrute(m, [lambda t: pol_a(t, {}), lambda t: coach(t, {})], post="coach",
                         coach=table)  # fmt: skip
    states = all_values(mg.Solution(m))
    rng = np.random.default_rng(seed)
    worst, moved = 0.0, 0.0
    for i in rng.choice(len(states), size=1200, replace=False):
        (t, f, s, e1, e2), _ = states[i]
        exact = float(pair.slices[t][f, s + m.score_max, e1, e2])
        worst = max(worst, abs(exact - brute.v(0, t, f, s, e1, e2)))
        moved = max(moved, abs(exact - float(model_pair.slices[t][f, s + m.score_max, e1, e2])))
    assert worst < 1e-10
    assert moved > 1e-3  # the flag really changes the game


def test_coach_post_decisions_match_the_simulator_on_the_toy_game(toy: mg.Solution) -> None:
    m, g = toy.model, toy.grid
    table = toy_coach(g, 5)
    coach = pe.CoachPolicy(m, table)
    cases = [
        dict(t=20, s=0, f=int(g.fidx(1, 1, 4)), r=1),
        dict(t=13, s=1, f=int(g.fidx(1, 1, 3))),
        dict(t=9, s=-2, f=int(g.fidx(1, 1, 5)), a=2, b=1),
        dict(t=7, s=1, f=int(g.fidx(2, 2, 4)), a=0, b=2),
        dict(t=5, s=0, f=int(g.fidx(3, 0, 3)), a=1, b=1),
        dict(t=4, s=-1, f=int(g.fidx(1, 1, 4)), a=2, b=2),
    ]
    pair, _ = pe.evaluate(m, coach, keep_slices=True, post="coach", coach=table)
    sim = mg.Simulator(toy, table)
    rng = np.random.default_rng(17)
    n = 40000
    for case in cases:
        st = sim.start(n, **case)
        ph = m.phase(case["t"])
        e = (case.get("a", 2), case.get("b", 2)) if ph == "end" else (case.get("r", 0), 0)
        exact = float(pair.slices[case["t"]][case["f"], case["s"] + m.score_max, *e])
        out = mg.simulate(sim, st, rng, policy="coach", post="coach")["result"]
        se = max(float(out.std()) / np.sqrt(n), 1e-4)
        assert abs(out.mean() - exact) < 4.5 * se, (case, exact, out.mean())


# --------------------------------------------------------------------------------------
# The real solved game (needs data/mdp)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def solved() -> mg.Solution:
    if not mg.solution_exists():
        pytest.skip("solved game not built (run scripts/solve_mdp.py)")
    return mg.load_solution()


@pytest.mark.data
def test_real_game_solver_policy_and_coach(solved: mg.Solution) -> None:
    import joblib

    m = solved.model
    pol = pe.array_policy(solved.pol, m)
    pair = pe.PolicyPair(m, pol)
    worst = [0.0]

    def check(t: int) -> None:
        worst[0] = max(worst[0], float(np.abs(pair.a.native(t) - solved.native(t)).max()))

    pe.run([pair], after=check)
    assert worst[0] == 0.0
    coach = pe.CoachPolicy(m, joblib.load(mg.SOLUTION_DIR / "tables.joblib")["coach"])
    two = pe.PolicyPair(m, coach, coach)
    pe.run([two])
    assert two.start_value() == pytest.approx(0.5, abs=1e-6)
