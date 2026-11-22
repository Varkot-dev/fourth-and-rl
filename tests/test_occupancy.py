"""Tests for the exact forward occupancy pass (`fourth_and_rl.occupancy`) and the tilted
fourth-down policies of R1 (`fourth_and_rl.overopt`)."""

from __future__ import annotations

import numpy as np
import pytest
from test_mdp import toy_football
from test_policy_eval import random_fourth, toy_coach

from fourth_and_rl import mdp_game as mg
from fourth_and_rl import occupancy as oc
from fourth_and_rl import overopt as oo
from fourth_and_rl import policy_eval as pe


def _setup(seed: int, defense: str) -> tuple:
    m = toy_football(seed=seed, defense=defense)
    table = toy_coach(m.kern.grid, seed)
    return m, table, pe.CoachPolicy(m, table)


@pytest.mark.parametrize(("seed", "defense"), [(1, "optimal"), (2, "empirical")])
def test_forward_win_probability_equals_the_backward_value(seed: int, defense: str) -> None:
    m, table, coach = _setup(seed, defense)
    rand = random_fourth(m, seed)
    pol_a = coach.with_fourth(lambda t, q: rand[t])
    # The solver's own policy for team B exercises every action, the kneel included.
    pol_b = pe.array_policy(mg.solve(m).pol, m)
    for a, b in ((pol_a, coach), (coach, pol_b), (pol_b, pol_a), (coach, None)):
        pair, _ = pe.evaluate(m, a, b, post="coach", coach=table)
        fw = oc.ForwardPass(m, a, b, coach=table).run()
        assert fw.total_mass == pytest.approx(1.0, abs=1e-12)
        assert fw.result_a == pytest.approx(pair.start_value(), abs=1e-12)


def test_flip_is_its_own_inverse() -> None:
    rng = np.random.default_rng(0)
    for ph, e in (("end", (3, 3)), ("mid", (1, 1)), ("first", (2, 1))):
        x = rng.random((5, 9, *e))
        assert np.array_equal(oc.flip(oc.flip(x, ph), ph), x)


def _oracle_q(m: mg.GameModel, table: dict, coach: pe.CoachPolicy) -> oo.FourthQ:
    q4 = oo.FourthQ(m)
    pair = pe.PolicyPair(m, coach, post="coach", coach=table)
    pe.run([pair], after=oo.record_fourth_q(pair, coach, q4))
    return q4


def _random_q(m: mg.GameModel, seed: int) -> oo.FourthQ:
    rng = np.random.default_rng(seed)
    q4 = oo.FourthQ(m)
    for arr in q4.q.values():
        arr[...] = rng.random(arr.shape).astype(np.float32)
    return q4


@pytest.mark.parametrize("beta", [3.0, 30.0])
def test_performance_difference_identity_is_exact(beta: float) -> None:
    m, table, coach = _setup(3, "empirical")
    oracle = _oracle_q(m, table, coach)
    proxy = _random_q(m, 5)
    pol = oo.tilt_policy(coach, proxy, beta)
    pair, _ = pe.evaluate(m, pol, coach, post="coach", coach=table)
    stats = oo.TiltStats(coach, proxy, beta, {"oracle": oracle, "proxy": proxy})
    fw = oc.ForwardPass(m, pol, coach, coach=table, hooks=[stats]).run()
    gain = pair.start_value() - 0.5
    assert abs(gain) > 1e-5
    # Exact up to the float32 storage of the action values (about 1e-8 relative).
    assert stats.claim["oracle"] == pytest.approx(gain, abs=1e-9)
    assert fw.result_a == pytest.approx(pair.start_value(), abs=1e-12)
    assert stats.kl > 0 and stats.decisions > 0


def test_exact_kl_matches_monte_carlo_on_the_toy_game() -> None:
    m, table, coach = _setup(4, "optimal")
    sol = mg.solve(m)
    oracle = _oracle_q(m, table, coach)
    proxy = _random_q(m, 6)
    beta = 8.0
    pol = oo.tilt_policy(coach, proxy, beta)
    pair, _ = pe.evaluate(m, pol, coach, post="coach", coach=table)
    stats = oo.TiltStats(coach, proxy, beta, {"oracle": oracle})
    oc.ForwardPass(m, pol, coach, coach=table, hooks=[stats]).run()
    sim = oo.TiltSimulator(sol, table, proxy, {"oracle": oracle})
    mc = oo.tilt_monte_carlo(sim, beta, 60000, seed=8)
    for exact, key in ((stats.kl, "kl"), (stats.decisions, "decisions"),
                       (pair.start_value(), "win_a"), (stats.claim["oracle"], "claim_oracle")):
        assert abs(mc[key] - exact) < 4.0 * mc[f"{key}_se"], (key, exact, mc[key])  # fmt: skip


def test_fourth_q_lookup_matches_the_arrays() -> None:
    m, _, _ = _setup(1, "optimal")
    q4 = _random_q(m, 1)
    g = m.kern.grid
    rng = np.random.default_rng(2)
    n = 500
    t = rng.integers(1, m.clock.n_ticks + 1, n)
    f = q4.f0 + rng.integers(0, q4.n4, n)
    s = rng.integers(-m.score_max, m.score_max + 1, n)
    a, b, r = rng.integers(0, 3, n), rng.integers(0, 3, n), rng.integers(0, 2, n)
    got = q4.lookup(t, f, s, a, b, r)
    for i in range(n):
        ph = m.phase(int(t[i]))
        e = (a[i], b[i]) if ph == "end" else ((r[i], 0) if ph == "first" else (0, 0))
        want = q4.get(int(t[i]))[f[i] - q4.f0, s[i] + m.score_max, e[0], e[1]]
        assert np.array_equal(got[i], want)
    assert g.unpack(q4.f0)[0] == g.n_down
