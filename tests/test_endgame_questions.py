"""Tests for M13 (goal-line and two-point decisions from the solved game).

Everything above the `data` section builds toy games small enough to solve by hand and
runs in CI without the play-by-play cache or the solved arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from fourth_and_rl import endgame_questions as eq
from fourth_and_rl import mdp_game as mg
from fourth_and_rl.mdp import KERNEL_KEYS, FieldKernels, Grid

GRID = Grid(n_down=1, n_yl=3, dist_exact=1, dist_bins=())  # one down: every down is "fourth"
NONE = (0, 0, 0)


def kernels(rules: dict[int, list], stop: float = 1.0, fg_make: tuple = (0.9, 0.8, 0.7)):
    """Kernels for a game on GRID. rules[yard line] = [(p, outcome)], outcome "td",
    ("opp", their yard line) or ("keep", yard line). `stop`: chance a kept-ball play
    leaves the clock stopped. Field goals from every yard line, a miss gives the other
    team the ball at its 3; kickoffs go to the receiver's 3."""
    g = GRID
    nf, na = g.nf, g.na
    ea: dict[tuple[int, int], float] = {}
    ei: dict[tuple[int, int], float] = {}
    for yl, outs in rules.items():
        f = int(g.fidx(1, 0, yl))
        for p, o in outs:
            if o == "td":
                col = g.col_td
            elif o[0] == "opp":
                col = int(g.col_opp(o[1]))
            else:
                col = int(g.fidx(1, 0, o[1]))
                if stop < 1.0:
                    ei[(f, col)] = ei.get((f, col), 0.0) + p * (1.0 - stop)
            ea[(f, col)] = ea.get((f, col), 0.0) + p

    def csr(e: dict, shape: tuple[int, int]) -> sp.csr_matrix:
        if not e:
            return sp.csr_matrix(shape)
        r, c = zip(*e, strict=True)
        return sp.coo_matrix((list(e.values()), (r, c)), shape=shape).tocsr()

    kick = np.zeros(g.n_yl + 1)
    kick[3] = 1.0
    none = np.zeros(nf, dtype=bool)
    return FieldKernels(
        grid=g,
        k_all={m: csr(ea, (nf, na)) for m in KERNEL_KEYS},
        k_in={m: csr(ei, (nf, nf)) for m in KERNEL_KEYS},
        fg_make=np.asarray(fg_make, dtype=float),
        fg_miss_col=np.full(nf, int(g.col_opp(3))),
        fg_ok=np.asarray(fg_make) > 0,
        k_punt=sp.csr_matrix((nf, na)),
        punt_ok=none.copy(),
        kneel_col=np.zeros(nf, dtype=np.int64),
        kneel_keep=none.copy(),
        kneel_ok=none.copy(),
        kickoff=kick,
        free_kick=kick.copy(),
    )


RULES = {
    1: [(0.6, "td"), (0.4, ("opp", 3))],
    2: [(0.4, "td"), (0.6, ("opp", 3))],
    3: [(0.5, ("keep", 2)), (0.5, ("opp", 1))],
}


def game(clock: mg.ClockSpec, kern: FieldKernels | None = None) -> mg.GameModel:
    return mg.GameModel(
        kern=kern or kernels(RULES),
        clock=clock,
        score_max=21,
        td_points=6,
        fg_points=3,
        safety_points=2,
        p_xp=0.9,
        p_two=0.4,
        two_point_choice=True,
        dtype="float64",
    )


def timed_clock(n_ticks: int = 14) -> mg.ClockSpec:
    """A clock with a first half, a middle phase and a last phase with one timeout each."""
    return mg.ClockSpec(
        n_ticks=n_ticks,
        tick_seconds=5,
        endgame_ticks=4,
        half_tick=9,
        warning_tick=2,
        burn_ticks=3,
        n_timeouts=1,
        neutral_hist=np.array([0.5, 0.5]),
        hurry_hist=np.array([1.0]),
    )


@pytest.fixture(scope="module")
def one_play() -> mg.Solution:
    """One tick left: every play ends the game."""
    return mg.solve(game(mg.ClockSpec(n_ticks=1)))


@pytest.fixture(scope="module")
def timed() -> mg.Solution:
    return mg.solve(game(timed_clock(), kernels(RULES, stop=0.5)))


# --------------------------------------------------------------------------------------
# Goal line, by hand
# --------------------------------------------------------------------------------------


def test_goal_line_values_by_hand(one_play: mg.Solution) -> None:
    # Last play, fourth and goal. From the 1 a touchdown comes 60% of the time, from the 2
    # 40%; a field goal is good 90% from the 1. After a touchdown the try (kick 90%, two
    # 40%) decides the final score; a failed play ends the game at the current score.
    v = eq.goal_line_values(one_play, [5], [0, -3, -7, -8], [1, 2], [NONE])
    go, fg = v["go"][0, :, 0], v["field_goal"][0, :, 0]
    # Tied at the 1: go 0.6 * 1 + 0.4 * 0.5 = 0.8; kick 0.9 * 1 + 0.1 * 0.5 = 0.95.
    assert go[0, 0] == pytest.approx(0.8)
    assert fg[0, 0] == pytest.approx(0.95)
    # Down 3: a touchdown wins (0.6), a good kick only ties (0.9 * 0.5 = 0.45).
    assert go[1, 0] == pytest.approx(0.6)
    assert fg[1, 0] == pytest.approx(0.45)
    # Down 7: the touchdown leaves it down 1 before the try; kick to tie 0.9 * 0.5 = 0.45
    # beats two to win 0.4, so go = 0.6 * 0.45 = 0.27; a field goal still loses.
    assert go[2, 0] == pytest.approx(0.27)
    assert fg[2, 0] == pytest.approx(0.0)
    # Down 8 at the 2: touchdown (0.4), then two to tie (0.4 * 0.5): 0.08.
    assert go[3, 1] == pytest.approx(0.08)


def test_goal_line_case_decomposes(one_play: mg.Solution) -> None:
    c = eq.goal_line_case(one_play, mg.GameState(5, -3, 1, 1, 1, 0, 0))
    assert c["p_touchdown"] == pytest.approx(0.6)
    assert c["win_probability_if_touchdown"] == pytest.approx(1.0)
    assert c["win_probability_if_stopped"] == pytest.approx(0.0)
    assert c["p_field_goal_good"] == pytest.approx(0.9)
    assert c["win_probability_if_field_goal_good"] == pytest.approx(0.5)
    assert c["win_probability_if_field_goal_missed"] == pytest.approx(0.0)
    assert c["break_even_p_touchdown"] == pytest.approx(0.45)
    assert c["best"] == "go"
    assert c["go_minus_field_goal"] == pytest.approx(0.15)


def test_goal_line_case_identities_and_missed_kick(timed: mg.Solution) -> None:
    """go = p * if_td + (1 - p) * if_stopped; the value after a missed field goal equals
    1 minus the other team's value at its 3 one tick later (computed independently)."""
    for sec, lead, yl in ((35, 0, 1), (40, -3, 2), (60, 4, 1), (15, 7, 2), (65, -1, 3)):
        state = mg.GameState(sec, lead, 1, yl, yl, 1, 1)
        c = eq.goal_line_case(timed, state)
        p = c["p_keep_or_score"]
        go = p * c["win_probability_if_touchdown"] + (1 - p) * c["win_probability_if_stopped"]
        assert go == pytest.approx(c["go"], abs=1e-9)
        mk = c["p_field_goal_good"]
        fg = mk * c["win_probability_if_field_goal_good"]
        fg += (1 - mk) * c["win_probability_if_field_goal_missed"]
        assert fg == pytest.approx(c["field_goal"], abs=1e-9)
        if sec > 25:  # outside the last phase, so no timeouts to swap
            # In the first half the other team receives the second-half kickoff if we do
            # not (our flag is 0 here).
            them = timed.value(mg.GameState(sec - 5, -lead, 1, 1, 3, 1, 1, 1))
            assert c["win_probability_if_field_goal_missed"] == pytest.approx(1 - them, abs=1e-9)


# --------------------------------------------------------------------------------------
# The try after a touchdown
# --------------------------------------------------------------------------------------


def test_try_values_by_hand(one_play: mg.Solution) -> None:
    tv = eq.try_values(one_play, [0], [-2, -1, 0], [NONE])
    kick, two = tv["kick"][0, :, 0], tv["two"][0, :, 0]
    # Down 2 before the try, game over after it: kicking loses, two ties 40% of the time.
    assert kick[0] == pytest.approx(0.0)
    assert two[0] == pytest.approx(0.2)
    # Down 1: kick to tie 0.9 * 0.5 = 0.45 beats two to win 0.4.
    assert kick[1] == pytest.approx(0.45)
    assert two[1] == pytest.approx(0.4)
    # Tied: kick 0.9 + 0.1 * 0.5 = 0.95, two 0.4 + 0.6 * 0.5 = 0.7.
    assert kick[2] == pytest.approx(0.95)
    assert two[2] == pytest.approx(0.7)
    assert list(tv["two_better"][0, :, 0]) == [True, False, False]
    case = eq.try_case(one_play, 0, -2, NONE)
    assert case["best"] == "two"
    assert case["break_even_p_two_point"] == pytest.approx(0.0)


def test_try_values_match_the_solver_everywhere(timed: mg.Solution) -> None:
    """The best of kick and two equals the solver's own value after a touchdown, and the
    two-point flag agrees, at every time, margin and timeout state (all three phases)."""
    m = timed.model
    margins = list(range(-14, 15))
    extras = [(a, b, r) for a in (0, 1) for b in (0, 1) for r in (0, 1)]
    secs = [5 * t for t in range(m.clock.n_ticks)]
    tv = eq.try_values(timed, secs, margins, extras)
    for i, sec in enumerate(secs):
        t = eq.tick_of(timed, sec)
        p = timed.post(t, m.phase(t))
        for k, ex in enumerate(extras):
            e1, e2 = eq.extra_index(timed, t, ex)
            idx = eq.score_index(timed, np.asarray(margins) - m.td_points)
            np.testing.assert_allclose(
                np.maximum(tv["kick"][i, :, k], tv["two"][i, :, k]), p.td[idx, e1, e2], atol=1e-12
            )
            close = np.isclose(tv["kick"][i, :, k], tv["two"][i, :, k])
            better = tv["two"][i, :, k] > tv["kick"][i, :, k]
            assert np.array_equal(better[~close], tv["two_better"][i, :, k][~close])


def forced_two_touchdowns() -> mg.GameModel:
    """A game whose only path is: we score, they turn it over at once, we score again.
    From the 1 a touchdown is certain; the receiver of a kickoff (at its 3) always loses
    the ball, giving us the 1; no field goals. Every play takes one tick."""
    rules = {1: [(1.0, "td")], 2: [(1.0, "td")], 3: [(1.0, ("opp", 1))]}
    return game(mg.ClockSpec(n_ticks=2), kernels(rules, fg_make=(0.0, 0.0, 0.0)))


def test_down_eight_matches_score_arithmetic_when_the_path_is_forced() -> None:
    """With exactly one more touchdown for us and none for them, the solved game must
    agree with the textbook arithmetic of going for two when down 8."""
    sol = mg.solve(forced_two_touchdowns())
    tv = eq.try_values(sol, [10], [-8], [NONE])  # touchdown with 2 ticks left
    ref = eq.two_try_arithmetic(p_two=0.4, p_xp=0.9)
    assert tv["two"][0, 0, 0] == pytest.approx(ref["go_for_two_now"])
    assert tv["kick"][0, 0, 0] == pytest.approx(ref["kick_now"])
    # By hand: two now 0.4 * 0.95 + 0.6 * 0.4 * 0.5 = 0.50; kick now
    # 0.9 * max(0.45, 0.4) + 0.1 * 0.4 * 0.5 = 0.425.
    assert ref["go_for_two_now"] == pytest.approx(0.50)
    assert ref["kick_now"] == pytest.approx(0.425)
    assert bool(tv["two_better"][0, 0, 0])


def test_score_arithmetic_with_a_coin_flip_try() -> None:
    r = eq.two_try_arithmetic(p_two=0.5, p_xp=1.0)
    assert r["go_for_two_now"] == pytest.approx(0.625)
    assert r["kick_now"] == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# Re-solving pieces
# --------------------------------------------------------------------------------------


def test_last_minutes_solved_alone_equal_the_full_game() -> None:
    model = game(timed_clock(), kernels(RULES, stop=0.5))
    full = mg.solve(model)
    part = mg.solve(eq.truncated(model, 40))
    assert model.clock.n_ticks == 14  # the original is untouched
    np.testing.assert_allclose(part.v["end"], full.v["end"], atol=1e-12)
    n = part.v["mid"].shape[0]
    np.testing.assert_allclose(part.v["mid"], full.v["mid"][:n], atol=1e-12)
    with pytest.raises(ValueError):
        eq.truncated(model, 50)


def test_two_point_override_moves_only_the_copy(one_play: mg.Solution) -> None:
    model = one_play.model
    hi = eq.with_two_point(model, 0.6)
    assert model.p_two == pytest.approx(0.4)
    sol = mg.solve(hi)
    tv = eq.try_values(sol, [0], [-1], [NONE])
    assert tv["two"][0, 0, 0] == pytest.approx(0.6)  # two to win from down 1
    assert bool(tv["two_better"][0, 0, 0])
    one_shot = eq.try_values(one_play, [0], [-1], [NONE], p_two=0.6)
    assert one_shot["two"][0, 0, 0] == pytest.approx(0.6)


# --------------------------------------------------------------------------------------
# The real solved game (needs data/mdp, written by scripts/solve_mdp.py)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def solved() -> mg.Solution:
    if not mg.solution_exists():
        pytest.skip("solved game not built (run scripts/solve_mdp.py)")
    return mg.load_solution()


@pytest.mark.data
def test_real_try_values_match_the_solver(solved: mg.Solution) -> None:
    margins = list(range(-15, 16))  # the score before the touchdown stays inside +-21
    extras = [(3, 3, 0), (1, 2, 0), (0, 0, 1)]
    secs = [30, 240, 300, 305, 600, 900, 1800, 2700]
    tv = eq.try_values(solved, secs, margins, extras)
    m = solved.model
    for i, sec in enumerate(secs):
        t = eq.tick_of(solved, sec)
        p = solved.post(t, m.phase(t))
        for k, ex in enumerate(extras):
            e1, e2 = eq.extra_index(solved, t, ex)
            idx = eq.score_index(solved, np.asarray(margins) - m.td_points)
            best = np.maximum(tv["kick"][i, :, k], tv["two"][i, :, k])
            np.testing.assert_allclose(best, p.td[idx, e1, e2], atol=1e-5)


@pytest.mark.data
def test_real_goal_line_matches_q_values(solved: mg.Solution) -> None:
    for sec, lead, yl, b in ((480, 7, 1, 3), (240, 7, 2, 1), (3300, 0, 3, 3)):
        v = eq.goal_line_values(solved, [sec], [lead], [yl], [(3, b, 0)])
        q = solved.q_values(mg.GameState(sec, lead, 4, yl, yl, 3, b))
        go = max(x for k, x in q.items() if k.startswith("play"))
        assert v["go"][0, 0, 0, 0] == pytest.approx(go, abs=1e-6)
        assert v["field_goal"][0, 0, 0, 0] == pytest.approx(q["field_goal"], abs=1e-6)
