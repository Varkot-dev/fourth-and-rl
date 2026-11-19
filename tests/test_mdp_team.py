"""Tests for the two-team game (`mdp_team.py`).

The first part builds tiny games by hand (reusing the toy games of tests/test_mdp.py) and
runs in seconds without data. The second part, marked `data`, uses the solved
identical-team game under data/mdp and M8's team-strength bundle.

The reference `TeamBrute` is the slow recursive solver of tests/test_mdp.py with every
"other team" value read from a second instance: it shares no code with the fast solver's
two-array backup.
"""

from __future__ import annotations

import dataclasses

import joblib
import numpy as np
import pytest
from test_mdp import Brute, all_values, hand_game, kernels_from_rules, toy_football

from fourth_and_rl import mdp, mdp_team
from fourth_and_rl import mdp_game as mg
from fourth_and_rl.mdp import KERNEL_KEYS, Grid
from fourth_and_rl.paths import DATA

# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------


def toy_pair(seed_a: int = 1, seed_b: int = 5) -> tuple[mg.GameModel, mg.GameModel]:
    """Two teams that play the same toy game (clock, scoring, kicks) with different plays."""
    a = toy_football(seed=seed_a)
    return a, dataclasses.replace(a, kern=toy_football(seed=seed_b).kern)


class TeamBrute(Brute):
    """Two-team reference solver: wherever `Brute` values the other team (a turnover, a
    kickoff or free kick after a score, a return touchdown, the halftime kickoff), this
    one asks `other`, the other team's instance."""

    other: TeamBrute

    def receive(self, t: int, s: int, e: tuple[int, int], phase: str, dist: np.ndarray) -> float:
        # `Brute` calls receive only for the team that gets the ball after a score.
        return Brute.receive(self.other, t, s, e, phase, dist)

    def outcome(self, col: int, t: int, s: int, e: tuple[int, int], phase: str) -> float:
        g = self.g
        if g.nf <= col < g.nf + g.n_yl:
            f = int(self.f10[col - g.nf + 1])
            return 1.0 - self.other.hold(t, f, -s, self.swap(e, phase), phase)
        if col == g.col_opp_td:
            return 1.0 - self.other.after_td(t, -s, self.swap(e, phase), phase)
        return super().outcome(col, t, s, e, phase)

    def _halftime(self, s: int, r: int) -> float:
        th = self.c.th

        def receive(who: TeamBrute, score: int) -> float:
            return sum(
                p * who.hold(th, int(who.f10[y]), score, (0, 0), "mid")
                for y, p in enumerate(who.k.kickoff)
                if p > 0
            )

        return receive(self, s) if r == 1 else 1.0 - receive(self.other, -s)


def team_brutes(a: mg.GameModel, b: mg.GameModel) -> tuple[TeamBrute, TeamBrute]:
    ba, bb = TeamBrute(a), TeamBrute(b)
    ba.other, bb.other = bb, ba
    return ba, bb


def asymmetric_hand_game(p_a: float, p_b: float, n_ticks: int = 2
                         ) -> tuple[mg.GameModel, mg.GameModel]:  # fmt: skip
    """`test_mdp.hand_game` with team-specific scoring chances from the 1 (A: p_a, B: p_b).
    From the 2 both gain a yard half the time. Kickoffs go to the 2; one tick per play."""
    grid = Grid(n_down=1, n_yl=2, dist_exact=1, dist_bins=())
    out = []
    for p in (p_a, p_b):
        rules = {0: [(p, ("td",)), (1 - p, ("opp", 2))], 1: [(0.5, ("keep", 0)), (0.5, ("opp", 1))]}
        kern = kernels_from_rules(grid, rules, {m: 1.0 for m in KERNEL_KEYS})
        kern.kickoff = np.array([0.0, 0.0, 1.0])
        kern.free_kick = kern.kickoff.copy()
        out.append(dataclasses.replace(hand_game(n_ticks), kern=kern))
    return out[0], out[1]


class FakeEffects:
    """Stands in for `RatingEffects` with fixed answers (no M8 models needed)."""

    def __init__(self, drive_shift: float, dlogit: float) -> None:
        self.shift, self.dlogit = drive_shift, dlogit

    def drive_points_shift(self, offense: dict, defense: dict, start: int = 75) -> float:
        return self.shift

    def conversion_dlogit(self, offense: dict, defense: dict) -> float:
        return self.dlogit


# --------------------------------------------------------------------------------------
# 1. Kernels
# --------------------------------------------------------------------------------------


def test_tilt_is_adjusted_play_row_by_row() -> None:
    kern = toy_football(seed=3).kern
    g = kern.grid
    a, b = 0.07, 1.6
    out = mdp_team.tilt_kernels(kern, a, b)
    down, _, yl = g.unpack(np.arange(g.nf))
    for key in ("neutral", "protect"):
        old, new = kern.k_all[key].toarray(), out.k_all[key].toarray()
        assert np.allclose(new.sum(axis=1), old.sum(axis=1))
        for f in np.nonzero(down < g.n_down)[0]:
            row = old[f]
            keep = np.nonzero(row[: g.nf])[0]
            opp = np.arange(g.nf, g.nf + g.n_yl)
            tail = np.concatenate([row[opp], [row[g.col_safety], row[g.col_opp_td]]])
            gain = yl[f] - yl[keep]
            k, td, tl = mdp.adjusted_play(row[keep], row[g.col_td], tail, gain, int(yl[f]), a, b)
            assert np.allclose(new[f, keep], k) and np.isclose(new[f, g.col_td], td)
            assert np.allclose(new[f, opp], tl[:-2])
            assert np.isclose(new[f, g.col_safety], tl[-2]) and np.isclose(
                new[f, g.col_opp_td], tl[-1]
            )
        # The clock-running part moves with its outcome.
        oi, ni = kern.k_in[key].toarray(), out.k_in[key].toarray()
        m = oi > 0
        assert np.allclose(
            ni[m] / oi[m], (new[:, : g.nf] / np.where(old[:, : g.nf] > 0, old[:, : g.nf], 1))[m]
        )
        assert (ni[~m] == 0).all()


def test_conversion_shift_is_in_log_odds_and_the_family_composes() -> None:
    kern = toy_football(seed=3).kern
    g = kern.grid
    last = np.nonzero(g.unpack(np.arange(g.nf))[0] == g.n_down)[0]
    d = 0.4
    out = mdp_team.tilt_kernels(kern, 0.0, 1.0, d)
    made = np.r_[np.arange(g.nf), g.col_td]
    for key in KERNEL_KEYS:
        old, new = kern.k_all[key].toarray(), out.k_all[key].toarray()
        p, q = old[last][:, made].sum(axis=1), new[last][:, made].sum(axis=1)
        ok = (p > 0) & (p < 1)
        assert np.allclose(np.log(q[ok] / (1 - q[ok])) - np.log(p[ok] / (1 - p[ok])), d)
        assert np.allclose(new.sum(axis=1), 1.0)
        # Downs before the last are untouched by the conversion shift.
        rest = np.setdiff1d(np.arange(g.nf), last)
        assert np.allclose(new[rest], old[rest])
    # Tilting twice is one tilt of the summed size and the multiplied turnover scale
    # (scales below 1, so the 0.5 cap on turnovers never binds in this toy).
    two = mdp_team.tilt_kernels(mdp_team.tilt_kernels(kern, 0.05, 0.9), -0.02, 0.8)
    one = mdp_team.tilt_kernels(kern, 0.03, 0.72)
    for key in KERNEL_KEYS:
        assert np.allclose(two.k_all[key].toarray(), one.k_all[key].toarray())
        assert np.allclose(two.k_in[key].toarray(), one.k_in[key].toarray())
    assert mdp_team.tilt_kernels(kern) is kern


def test_drive_points_by_hand() -> None:
    # Hand game, one down: from the 1 a touchdown half the time (7 points): 3.5. From the
    # 2 a yard half the time, then 3.5: 1.75. No kicks are allowed, so the rule goes.
    kern = hand_game().kern
    assert np.allclose(mdp_team.drive_points(kern), [3.5, 1.75])


def test_calibration_hits_its_targets() -> None:
    kern = toy_football(seed=3).kern
    c = mdp_team.calibrate(kern, {}, {}, FakeEffects(0.15, -0.3), start_yardline=4)
    assert c.achieved_shift == pytest.approx(0.15, abs=1e-8)
    assert c.tilt > 0 and c.conv_dlogit == -0.3
    c0 = mdp_team.calibrate(kern, {}, {}, FakeEffects(0.0, 0.0), start_yardline=4)
    assert c0.tilt == 0.0 and c0.achieved_shift == 0.0
    # A weaker offense needs a negative tilt; more drive points need more tilt.
    lo = mdp_team.calibrate(kern, {}, {}, FakeEffects(-0.15, 0.0), start_yardline=4)
    hi = mdp_team.calibrate(kern, {}, {}, FakeEffects(0.30, 0.0), start_yardline=4)
    assert lo.tilt < 0 < hi.tilt and hi.tilt > c.tilt


# --------------------------------------------------------------------------------------
# 2. The two-team solver
# --------------------------------------------------------------------------------------


def test_identical_teams_equal_the_one_array_solution() -> None:
    m = toy_football(seed=1)
    one = mg.solve(m)
    pair = mdp_team.solve_matchup(m, m)
    for sol in pair.sols:
        for ph in one.v:
            assert np.abs(sol.v[ph] - one.v[ph]).max() <= 1e-12
            assert np.array_equal(sol.pol[ph], one.pol[ph])


def test_swapping_the_teams_swaps_the_arrays() -> None:
    a, b = toy_pair()
    ab, ba = mdp_team.solve_matchup(a, b), mdp_team.solve_matchup(b, a)
    for ph in ab.sols[0].v:
        assert np.array_equal(ab.sols[0].v[ph], ba.sols[1].v[ph])
        assert np.array_equal(ab.sols[1].v[ph], ba.sols[0].v[ph])
        assert np.array_equal(ab.sols[0].pol[ph], ba.sols[1].pol[ph])
    # And the teams really differ.
    assert np.abs(ab.sols[0].v["end"] - ab.sols[1].v["end"]).max() > 0.01


def test_two_team_solver_matches_the_reference() -> None:
    a, b = toy_pair()
    pair = mdp_team.solve_matchup(a, b)
    brutes = team_brutes(a, b)
    rng = np.random.default_rng(4)
    for sol, brute in zip(pair.sols, brutes, strict=True):
        states = all_values(sol)
        picks = rng.choice(len(states), size=800, replace=False)
        worst = max(abs(states[i][1] - brute.v(*states[i][0])) for i in picks)
        assert worst < 1e-10


def test_asymmetric_game_by_hand() -> None:
    # A scores from the 1 half the time, B a fifth of the time. Index [t - 1, yl - 1, s + 3].
    pair = mdp_team.solve_matchup(*asymmetric_hand_game(0.5, 0.2))
    va, vb = (s.v["mid"][..., 0, 0] for s in pair.sols)
    z = 3
    # One play left, tied, at the 1: score and win, or the game ends tied (0.5).
    assert va[0, 0, z] == pytest.approx(0.5 * 1 + 0.5 * 0.5)  # 0.75
    assert vb[0, 0, z] == pytest.approx(0.2 * 1 + 0.8 * 0.5)  # 0.60
    # Two plays left, tied, at the 2. A gains a yard (1/2): A at the 1 with one play,
    # 0.75. A loses it (1/2): B at its 1, tied, one play: 0.60 for B, 0.40 for A.
    assert va[1, 1, z] == pytest.approx(0.5 * 0.75 + 0.5 * 0.40)  # 0.575
    # The same for B: 0.5 * 0.60 + 0.5 * (1 - 0.75) = 0.425. With identical teams both
    # would be 0.5; here A is the favourite from the same spot.
    assert vb[1, 1, z] == pytest.approx(0.5 * 0.60 + 0.5 * 0.25)  # 0.425
    # Two plays left, A down 1 at the 1: score (1/2) and B gets one play from its 2, tied
    # (it cannot score: 0.5); fail (1/2) and B leads with the ball: 0.
    assert va[1, 0, z - 1] == pytest.approx(0.5 * 0.5)
    brutes = team_brutes(*(s.model for s in pair.sols))
    for sol, brute in zip(pair.sols, brutes, strict=True):
        for (t, f, s, e1, e2), val in all_values(sol):
            assert val == pytest.approx(brute.v(t, f, s, e1, e2), abs=1e-12)


def test_a_stronger_offense_never_lowers_its_value() -> None:
    base = toy_football(seed=1)
    strong = dataclasses.replace(base, kern=mdp_team.tilt_kernels(base.kern, 0.15, 1.0, 0.3))
    ref = mdp_team.solve_matchup(base, base)
    up = mdp_team.solve_matchup(strong, base)
    for ph in ref.sols[0].v:
        # A with the ball is worth at least as much; B with the ball (facing A's defense,
        # unchanged, but A's offense next) is worth at most as much.
        assert (up.sols[0].v[ph] >= ref.sols[0].v[ph] - 1e-12).all(), ph
        assert (up.sols[1].v[ph] <= ref.sols[1].v[ph] + 1e-12).all(), ph
        assert (up.sols[0].v[ph] - ref.sols[0].v[ph]).max() > 0.01


def test_monte_carlo_matches_two_team_values() -> None:
    a, b = toy_pair()
    pair = mdp_team.solve_matchup(a, b)
    sim = mdp_team.MatchupSimulator(pair)
    g = a.kern.grid
    rng = np.random.default_rng(9)
    n = 30000
    cases = [
        dict(team=0, t=20, s=0, f=int(g.fidx(1, 1, 4)), r=1),
        dict(team=1, t=20, s=0, f=int(g.fidx(1, 1, 4)), r=1),
        dict(team=1, t=13, s=1, f=int(g.fidx(1, 1, 3))),
        dict(team=0, t=9, s=-2, f=int(g.fidx(1, 1, 5)), a=2, b=1),
        dict(team=1, t=7, s=1, f=int(g.fidx(2, 2, 4)), a=0, b=2),
    ]
    for case in cases:
        st = sim.start(n, **case)
        expect = float(sim.values(st)[0])
        out = mg.simulate(sim, st, rng)
        se = max(float(out["result"].std()) / np.sqrt(n), 1e-4)
        assert abs(out["result"].mean() - expect) < 4.5 * se, (case, expect, out["result"].mean())


def test_teams_must_play_the_same_game() -> None:
    a = toy_football(seed=1)
    with pytest.raises(ValueError, match="clock"):
        mdp_team.solve_matchup(a, toy_football(seed=2))
    other = dataclasses.replace(a, kern=dataclasses.replace(a.kern, kickoff=a.kern.free_kick))
    with pytest.raises(ValueError, match="kickoff"):
        mdp_team.check_compatible(a, other)


# --------------------------------------------------------------------------------------
# 3. The real game (needs data/mdp and data/models/team_strength.joblib)
# --------------------------------------------------------------------------------------

BUNDLE = DATA / "models" / "team_strength.joblib"


@pytest.fixture(scope="module")
def real() -> mg.Solution:
    if not mg.solution_exists() or not BUNDLE.exists():
        pytest.skip("solved game or team-strength bundle not built")
    return mg.load_solution()


@pytest.fixture(scope="module")
def effects(real: mg.Solution) -> mdp_team.RatingEffects:
    return mdp_team.RatingEffects.from_bundle(joblib.load(BUNDLE))


@pytest.fixture(scope="module")
def strong_pair(real: mg.Solution, effects: mdp_team.RatingEffects) -> mdp_team.Matchup:
    """End phase: a top-five offense (M8 development groups) against an average defense,
    and an average team on the other side."""
    groups = joblib.load(BUNDLE)["groups"]
    base = mdp_team.endgame_model(real.model)
    strong, _ = mdp_team.matchup_model(base, groups["top"], groups["average"], effects)
    return mdp_team.solve_matchup(strong, base)


@pytest.mark.data
def test_real_identical_teams_equal_the_saved_solution(real: mg.Solution) -> None:
    base = mdp_team.endgame_model(real.model)
    pair = mdp_team.solve_matchup(base, base)
    saved = real.v["end"]
    for sol in pair.sols:
        for t in range(saved.shape[0]):
            assert np.abs(sol.v["end"][t] - np.asarray(saved[t])).max() <= 1e-6, t


@pytest.mark.data
def test_real_tilt_reproduces_the_protect_table(real: mg.Solution) -> None:
    # The protect table is the generic play bent by its fitted (tilt, turnover scale);
    # bending the generic table with the same two numbers must give it back.
    adj = joblib.load(mg.SOLUTION_DIR / "tables.joblib")["tables"]["adjust"]
    kern = real.model.kern
    p = adj["protect"]
    out = mdp_team.tilt_kernels(kern, p["tilt"], p["turnover_scale"], keys=("neutral",))
    diff = abs(out.k_all["neutral"] - kern.k_all["protect"]).max()
    assert diff < 1e-9


@pytest.mark.data
def test_real_calibration(real: mg.Solution, effects: mdp_team.RatingEffects) -> None:
    groups = joblib.load(BUNDLE)["groups"]
    kern = real.model.kern
    up = mdp_team.calibrate(kern, groups["top"], groups["average"], effects)
    down = mdp_team.calibrate(kern, groups["bottom"], groups["best"], effects)
    assert up.achieved_shift == pytest.approx(up.target_drive_points_shift, abs=1e-6)
    assert down.achieved_shift == pytest.approx(down.target_drive_points_shift, abs=1e-6)
    assert up.target_drive_points_shift > 0.2 and up.tilt > 0 and up.conv_dlogit > 0
    assert down.target_drive_points_shift < -0.2 and down.tilt < 0 and down.conv_dlogit < 0
    zero = mdp_team.calibrate(kern, groups["average"], groups["average"], effects)
    assert zero.tilt == 0.0 and zero.conv_dlogit == 0.0


@pytest.mark.data
def test_real_stronger_offense_is_monotone_on_a_sample(
    real: mg.Solution, strong_pair: mdp_team.Matchup
) -> None:
    rng = np.random.default_rng(5)
    saved = real.v["end"]
    n = saved.shape[0]
    for t in rng.choice(np.arange(1, n), size=12, replace=False):
        base = np.asarray(saved[t])
        a, b = strong_pair.sols[0].v["end"][t], strong_pair.sols[1].v["end"][t]
        idx = tuple(rng.integers(0, k, 50000) for k in base.shape)
        assert (a[idx] >= base[idx] - 1e-5).all(), t
        assert (b[idx] <= base[idx] + 1e-5).all(), t


@pytest.mark.data
def test_real_monte_carlo_matches_two_team_values(strong_pair: mdp_team.Matchup) -> None:
    sim = mdp_team.MatchupSimulator(strong_pair)
    g = strong_pair.sols[0].grid
    rng = np.random.default_rng(13)
    n = 6000
    cases = [
        dict(team=0, t=48, s=4, f=int(g.fidx(4, 1, 60)), a=3, b=3),
        dict(team=1, t=48, s=-4, f=int(g.fidx(1, 9, 82)), a=3, b=3),
        dict(team=1, t=30, s=-4, f=int(g.fidx(1, 9, 75)), a=2, b=3),
        dict(team=0, t=60, s=0, f=int(g.fidx(2, 6, 45)), a=3, b=2),
    ]
    for case in cases:
        st = sim.start(n, **case)
        expect = float(sim.values(st)[0])
        out = mg.simulate(sim, st, rng)
        se = max(float(out["result"].std()) / np.sqrt(n), 1e-3)
        assert abs(out["result"].mean() - expect) < 4.5 * se, (case, expect, out["result"].mean())
