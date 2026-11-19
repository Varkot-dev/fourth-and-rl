"""Tests for the JAX end-phase backend (`mdp_team_jax.py`), on CPU JAX.

Skipped when JAX is not installed (it is an optional extra, not a core dependency). The
reference is always the tested CPU solver, `mdp_team.solve_matchup`, on the same
end-phase models. JAX computes in float32, so values are compared to 1e-5; policies are
compared allowing exact or near ties: where the two solvers pick different actions, the
CPU solver's own action values for the two picks must be within 1e-5.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("jax")

from test_mdp import toy_football  # noqa: E402

from fourth_and_rl import mdp_game as mg  # noqa: E402
from fourth_and_rl import mdp_team as mt  # noqa: E402
from fourth_and_rl import mdp_team_jax as mj  # noqa: E402
from fourth_and_rl.paths import DATA  # noqa: E402

TOL = 1e-5
BUNDLE = DATA / "models" / "team_strength.joblib"


def f32(model: mg.GameModel) -> mg.GameModel:
    return dataclasses.replace(model, dtype="float32")


def toy_teams(defense: str = "optimal") -> tuple[mg.GameModel, mg.GameModel]:
    a = f32(toy_football(seed=1, defense=defense))
    return a, dataclasses.replace(a, kern=toy_football(seed=5).kern)


def policy_gap(sol: mg.Solution, pol: np.ndarray, ticks: range) -> float:
    """Largest loss, under `sol`'s own action values, of playing `pol` instead of `sol`'s
    policy (0 where they agree or the game is over)."""
    worst = 0.0
    for t in ticks:
        q = sol.q_slice(t)
        mine = sol.pol["end"][t]
        alt = pol[t]
        diff = (mine != alt) & (alt != mg.A_OVER) & (mine != mg.A_OVER)
        if not diff.any():
            continue
        codes = sorted(q)
        pos = np.zeros(len(mg.ACTION_NAMES), dtype=np.int64)
        pos[codes] = np.arange(len(codes))
        vals = np.stack([q[c][diff] for c in codes])  # (actions, states)
        cols = np.arange(int(diff.sum()))
        a, b = vals[pos[mine[diff]], cols], vals[pos[alt[diff]], cols]
        worst = max(worst, float(np.abs(a - b).max()))
    return worst


def assert_matches_cpu(res: mj.EndPhaseResult, cpu: mt.Matchup) -> None:
    n = res.v.shape[1]
    for k in range(2):
        ref = cpu.sols[k]
        assert np.abs(res.v[k] - np.asarray(ref.v["end"])).max() <= TOL, k
        assert policy_gap(ref, res.pol[k], range(1, n)) <= TOL, k


# --------------------------------------------------------------------------------------
# Toy game (seconds)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("defense", ["optimal", "empirical"])
def test_toy_matches_the_cpu_solver(defense: str) -> None:
    a, b = toy_teams(defense)
    for pair in ((a, a), (a, b)):  # identical and asymmetric teams
        cpu = mt.solve_matchup(*(mt.endgame_model(m) for m in pair))
        res = mj.solve_end_phases([pair])[0]
        assert_matches_cpu(res, cpu)
    # The asymmetric pair really is asymmetric.
    assert np.abs(res.v[0] - res.v[1]).max() > 0.01


def test_toy_end_phase_equals_the_whole_game_end_phase() -> None:
    a, b = toy_teams()
    whole = mt.solve_matchup(a, b)
    res = mj.solve_end_phases([(a, b)])[0]
    for k in range(2):
        assert np.abs(res.v[k] - whole.sols[k].v["end"]).max() <= TOL


def test_swapping_the_teams_swaps_the_arrays() -> None:
    a, b = toy_teams()
    ab = mj.solve_end_phases([(a, b)])[0]
    ba = mj.solve_end_phases([(b, a)])[0]
    assert np.array_equal(ab.v[0], ba.v[1]) and np.array_equal(ab.v[1], ba.v[0])
    assert np.array_equal(ab.pol[0], ba.pol[1]) and np.array_equal(ab.pol[1], ba.pol[0])


def test_batched_matchups_equal_one_at_a_time() -> None:
    a, b = toy_teams()
    c = dataclasses.replace(a, kern=toy_football(seed=9).kern)
    pairs = [(a, b), (b, c), (c, a)]
    together = mj.solve_end_phases(pairs, batch_size=3)
    alone = [mj.solve_end_phases([p])[0] for p in pairs]
    for x, y in zip(together, alone, strict=True):
        assert np.abs(x.v - y.v).max() <= 1e-6
        assert x.meta["batch_size"] == 3


def test_action_values_and_matchup_queries_match_the_cpu() -> None:
    a, b = toy_teams()
    cpu = mt.solve_matchup(mt.endgame_model(a), mt.endgame_model(b))
    res = mj.solve_end_phases([(a, b)], q_ticks=(5, 9))[0]
    for t in (5, 9):
        for k in range(2):
            q = cpu.sols[k].q_slice(t)
            assert set(q) == set(res.q[t])
            for code, x in q.items():
                y = res.q[t][code][k]
                assert np.array_equal(np.isfinite(x), np.isfinite(y))
                fin = np.isfinite(x)
                assert np.abs(x[fin] - y[fin]).max() <= TOL
    # Wrapped as a Matchup, the existing query code works on the JAX arrays.
    pair = res.matchup()
    g = a.kern.grid
    state = mg.GameState(seconds=35, score=1, down=3, ydstogo=2, yardline_100=3,
                         off_timeouts=2, def_timeouts=1)  # fmt: skip
    for team in ("A", "B"):
        qa, qb = pair.q_values(state, team), cpu.q_values(state, team)
        assert qa.keys() == qb.keys()
        assert all(abs(qa[k] - qb[k]) <= TOL for k in qa)
        assert pair.value(state, team) == pytest.approx(cpu.value(state, team), abs=TOL)
    assert g.nf == pair.sols[0].v["end"].shape[1]


def test_sparse_backends_equal_dense() -> None:
    a, b = toy_teams("empirical")
    dense = mj.solve_end_phases([(a, b)], backend="dense")[0]
    for name in ("bcoo", "bcoo_cusparse"):
        sparse = mj.solve_end_phases([(a, b)], backend=name)[0]
        assert np.abs(sparse.v - dense.v).max() <= 1e-6
    bench = mj.benchmark_backends((a, b), ticks=2)
    assert bench["choice"] in mj.BACKENDS


def test_models_must_play_the_same_game() -> None:
    a, b = toy_teams()
    other = f32(toy_football(seed=2, n_timeouts=1))
    with pytest.raises(ValueError, match="differ"):
        mj.solve_end_phases([(a, b), (other, other)])
    with pytest.raises(ValueError, match="backend"):
        mj.solve_end_phases([(a, b)], backend="nope")


# --------------------------------------------------------------------------------------
# The real end phase (needs data/mdp and the team-strength bundle; a few minutes)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real() -> mg.Solution:
    if not mg.solution_exists() or not BUNDLE.exists():
        pytest.skip("solved game or team-strength bundle not built")
    return mg.load_solution()


@pytest.mark.data
def test_real_identical_teams_equal_the_saved_solution(real: mg.Solution) -> None:
    base = mt.endgame_model(real.model)
    res = mj.solve_end_phases([(base, base)])[0]
    saved = real.v["end"]
    for k in range(2):
        for t in range(saved.shape[0]):
            assert np.abs(res.v[k][t] - np.asarray(saved[t])).max() <= TOL, (k, t)
    assert np.array_equal(res.v[0], res.v[1])


@pytest.mark.data
def test_real_asymmetric_matchup_matches_the_cpu_solver(real: mg.Solution) -> None:
    import joblib

    bundle = joblib.load(BUNDLE)
    eff = mt.RatingEffects.from_bundle(bundle)
    groups = bundle["groups"]
    base = mt.endgame_model(real.model)
    strong, _ = mt.matchup_model(base, groups["top"], groups["average"], eff)
    res = mj.solve_end_phases([(strong, base)], q_ticks=(48,))[0]
    cpu = mt.solve_matchup(strong, base)
    for k in range(2):
        assert np.abs(res.v[k] - cpu.sols[k].v["end"]).max() <= TOL, k
        # Policies: ties only, checked at a late, a middle and the headline tick.
        assert policy_gap(cpu.sols[k], res.pol[k], range(12, 61, 18)) <= TOL, k
    q = cpu.sols[0].q_slice(48)
    for code, x in q.items():
        fin = np.isfinite(x)
        assert np.array_equal(fin, np.isfinite(res.q[48][code][0]))
        assert np.abs(x[fin] - res.q[48][code][0][fin]).max() <= TOL
