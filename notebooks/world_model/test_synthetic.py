"""CPU tests for synthetic.py: synthetic games from the solved game model and ground truth.

The toy tests use the tiny solved game in fourth_and_rl.rl.toy (solved in well under a
second). The tests marked `real` need the solved game under data/mdp and skip without it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import schema as S

pytest.importorskip("fourth_and_rl")
pytest.importorskip("polars")  # records.validate_split's module imports polars

import records  # noqa: E402
import synthetic as syn  # noqa: E402

from fourth_and_rl import mdp_game as mg  # noqa: E402
from fourth_and_rl.rl import toy  # noqa: E402

DATA_MDP = Path(__file__).resolve().parents[2] / "data" / "mdp"
HAVE_REAL = (DATA_MDP / "meta.json").exists() and (DATA_MDP / "tables.joblib").exists()
real = pytest.mark.skipif(not HAVE_REAL, reason="needs the solved game under data/mdp")


@pytest.fixture(scope="module")
def toy_sim():
    sol = toy.toy_game(0)
    return sol, mg.Simulator(sol, coach=toy.toy_coach_table(sol.grid))


@pytest.fixture(scope="module")
def toy_split(toy_sim):
    return syn.simulate_games(toy_sim[1], 400, seed=3, chunk_games=100)


# ---------------------------------------------------------------- schema
def test_appended_actions_do_not_change_existing_codes():
    old = ("PAD", "RUN", "PASS", "PUNT", "FG", "KNEEL", "SPIKE", "KICKOFF", "ONSIDE", "XP",
           "TWO_PT", "NOPLAY")  # fmt: skip
    assert S.ACTIONS[: len(old)] == old
    assert S.ACTIONS[len(old) :] == ("PLAY_NEUTRAL", "PLAY_PROTECT", "PLAY_CHASE")
    assert S.GO_ACTIONS == (S.ACTION["RUN"], S.ACTION["PASS"])
    assert set(S.ANY_GO_ACTIONS) == set(S.GO_ACTIONS) | set(S.PLAY_MODE_ACTIONS)


def test_go_set_is_derived_from_the_split(toy_split):
    from toy import make_toy_split

    assert S.go_actions(toy_split) == S.PLAY_MODE_ACTIONS
    assert S.go_actions(make_toy_split(4, np.random.default_rng(0))) == S.GO_ACTIONS
    assert S.go_actions(None) == S.GO_ACTIONS


# ---------------------------------------------------------------- synthetic games
def test_bookkeeping_identity_on_every_pair(toy_split):
    tm = records.transition_mismatches(toy_split)
    assert tm["pairs"] > 1000 and tm["mismatched_pairs"] == 0


def test_validate_split_passes(toy_split):
    rep = records.validate_split(toy_split, min_transition_match=1.0)
    assert rep["info_apply_transition_match_fraction"] == 1.0


def test_codes_within_vocab_and_expected_actions(toy_split):
    st, tg = toy_split.states, toy_split.targets
    for j, (_, size) in enumerate(S.STATE_FIELDS):
        assert st[:, j].min() >= 0 and st[:, j].max() < size
    for j, (_, size) in enumerate(S.TARGET_FIELDS):
        assert tg[:, j].min() >= 0 and tg[:, j].max() < size
    SI, A, PH = S.STATE_INDEX, S.ACTION, S.PHASE
    ph, act = st[:, SI["phase"]], st[:, SI["action"]]
    allowed_play = set(S.PLAY_MODE_ACTIONS) | {A["FG"], A["PUNT"], A["KNEEL"]}
    assert set(np.unique(act[ph == PH["PLAY"]])) <= allowed_play
    assert set(np.unique(act[ph == PH["KICKOFF"]])) == {A["KICKOFF"]}
    assert set(np.unique(act[ph == PH["TRY"]])) <= {A["XP"], A["TWO_PT"]}
    # every game opens with a kickoff whose receiver is the header's opening receiver
    first = toy_split.offsets[:-1]
    assert (st[first, SI["phase"]] == PH["KICKOFF"]).all()
    assert (st[first, SI["poss"]] == toy_split.header[:, 1]).all()
    assert (toy_split.header[:, 0] == S.spread_code(0)).all()


def test_try_and_kickoff_records_follow_scores(toy_split):
    st, tg = toy_split.states, toy_split.targets
    SI, T, PH = S.STATE_INDEX, S.TARGET_INDEX, S.PHASE
    live = tg[:, T["phase"]] != PH["END"]
    # a try follows every touchdown-sized score on a play, and only those
    td = 2  # the toy game's touchdown is worth 2
    d = S.score_delta_value(tg[:, T["score_delta"]])
    play = st[:, SI["phase"]] == PH["PLAY"]
    assert ((tg[:, T["phase"]] == PH["TRY"]) == (live & play & (np.abs(d) == td))).all()
    # the try's possession is the scoring team: sign of the delta matches HOME/AWAY
    k = np.flatnonzero(live & play & (np.abs(d) == td))
    home_scored = d[k] > 0
    assert (tg[k, T["poss"]] == np.where(home_scored, 1, 2)).all()
    # kickoffs take no clock
    ko = np.flatnonzero((st[:, SI["phase"]] == PH["KICKOFF"]) & live)
    assert (tg[ko, T["elapsed"]] == S.elapsed_code(0)).all()


def test_deterministic_and_independent_of_workers(toy_sim):
    sim = toy_sim[1]
    a = syn.simulate_games(sim, 200, seed=11, chunk_games=100)
    b = syn.simulate_games(sim, 200, seed=11, chunk_games=100, n_workers=2)
    c = syn.simulate_games(sim, 100, seed=11, chunk_games=100)  # the first chunk alone
    d = syn.simulate_games(sim, 200, seed=12, chunk_games=100)
    for f in ("states", "raw", "targets", "offsets", "header", "final_diff"):
        assert np.array_equal(getattr(a, f), getattr(b, f))
    n = int(c.offsets[-1])
    assert np.array_equal(a.states[:n], c.states) and a.game_ids[:100] == c.game_ids
    assert not np.array_equal(a.final_diff, d.final_diff)


def test_final_scores_are_the_simulators(toy_sim):
    """The records' final scores equal what mdp_game.simulate gets with the same draws."""
    sol, sim = toy_sim
    rng = np.random.default_rng(5)
    log = syn._Log()
    st, gid, home_side, _ = syn._kickoff_games(sim, 300, rng, 0, log)
    st2 = {k: v.copy() for k, v in st.items()}
    state = rng.bit_generator.state
    syn._play_out(sim, st, gid, home_side, np.zeros(300, np.int64), rng, log, 4)
    rng2 = np.random.default_rng(0)
    rng2.bit_generator.state = state
    out = mg.simulate(sim, st2, rng2, policy="coach")
    assert np.array_equal(st["pts"], out["points"])


def test_timeout_drop_of_three_is_spread():
    game = np.array([0, 0, 0, 0, 1, 1])
    to = np.array([3, 3, 0, 0, 3, 1])
    assert syn.fix_timeout_drops(game, to).tolist() == [3, 2, 0, 0, 3, 1]


def test_go_forcing_picks_play_modes(toy_split):
    torch = pytest.importorskip("torch")
    import simulate
    from model import WorldModel

    torch.manual_seed(0)
    cfg = {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32, "dec_hidden": 16,
           "dropout": 0.0, "max_len": 32, "sim_batch": 64, "seed": 0}  # fmt: skip
    m = WorldModel(cfg).eval()
    play = np.flatnonzero(toy_split.states[:, S.STATE_INDEX["phase"]] == S.PHASE["PLAY"])[5:7]
    far = time.time() + 600
    r = simulate.rollout(m, toy_split, play, "GO", 8, cfg, far, np.random.default_rng(0))
    assert set(r["start_actions"]) <= {"PLAY_NEUTRAL", "PLAY_PROTECT", "PLAY_CHASE"}
    assert sum(r["start_actions"].values()) == 16


# ---------------------------------------------------------------- ground truth on the toy game
def _toy_states(sol, t: int, scores=(0, -1, -3)) -> dict:
    g = sol.grid
    rows = [(t, s, int(g.fidx(g.n_down, g.dist_bin(1), yl)), 1, 1, 0)
            for s in scores for yl in (2, 4)]  # fmt: skip
    return {k: np.array([r[i] for r in rows]) for i, k in enumerate(("t", "s", "f", "a", "b", "r"))}


def _exact_last_tick(sim, state: dict, code: int) -> float:
    """Win probability of one action with one tick left (every action ends the game),
    under the coach rule for tries: enumerate the action's outcomes."""
    m, c, k, g = sim.model, sim.model.clock, sim.model.kern, sim.grid
    t, s, f = int(state["t"]), int(state["s"]), int(state["f"])
    assert t == 1
    leads = tuple(sim._coach_setting("two_point_leads"))

    def term(x: int) -> float:
        return 1.0 if x > 0 else (0.5 if x == 0 else 0.0)

    def after_td(ss: int) -> float:  # value to the scorer, ss = its score before the try
        two = (ss + m.td_points) in leads  # t1 = 0 is inside the coach's two-point window
        p = m.p_two if two else m.p_xp
        bonus = 2 if two else 1
        return p * term(ss + m.td_points + bonus) + (1 - p) * term(ss + m.td_points)

    def value_of(col: int) -> float:
        if col == g.col_td:
            return after_td(s)
        if col == g.col_opp_td:
            return 1.0 - after_td(-s)
        if col == g.col_safety:
            return term(s - m.safety_points)
        return term(s)  # kept, or the other team takes over: the clock is out either way

    if code == mg.A_FG:
        p = float(k.fg_make[f])
        return p * term(s + m.fg_points) + (1 - p) * term(s)
    if code == mg.A_PUNT:
        row = k.k_punt[f].toarray().ravel()
    else:
        mode = {mg.A_NEUTRAL: "neutral", mg.A_PROTECT: "protect", mg.A_CHASE: "chase"}[code]
        row = k.k_all[c.kernel_key(mode, t)][f].toarray().ravel()
    return float(sum(p * value_of(col) for col, p in enumerate(row) if p > 0))


def test_ground_truth_last_tick_matches_exact_enumeration(toy_sim):
    sol, sim = toy_sim
    states = _toy_states(sol, 1)
    n = 20000
    gt = syn.ground_truth(sim, sol, states, n, seed=1)
    for i in range(len(states["t"])):
        st = {k: v[i] for k, v in states.items()}
        codes = (gt["go_best_code"][i], gt["go_coach_code"][i], mg.A_FG, mg.A_PUNT)
        for j, code in enumerate(codes):
            if not gt["allowed_model"][i, j]:
                assert np.isnan(gt["q_coach"][i, j])
                continue
            exact = _exact_last_tick(sim, st, int(code))
            se = max(gt["se_coach"][i, j], 0.5 / np.sqrt(n) * 0.2)
            assert abs(gt["q_coach"][i, j] - exact) < 4.5 * se, (i, j, exact)


def test_ground_truth_q_star_is_the_solution(toy_sim):
    sol, sim = toy_sim
    states = _toy_states(sol, 9)
    gt = syn.ground_truth(sim, sol, states, 50, seed=2)
    g = sol.grid
    for i in range(len(states["t"])):
        down, dbin, yl = g.unpack(states["f"][i])
        q = sol.q_values(mg.GameState(int(states["t"][i]) * 5, int(states["s"][i]), int(down),
                                      int(min(g.dist_rep(dbin), yl)), int(yl), 1, 1, 0))
        modes = [q.get(k, np.nan) for k in ("play", "play_protect", "play_chase")]
        assert gt["q_star"][i, 0] == pytest.approx(np.nanmax(modes))
        assert gt["q_star"][i, 2] == pytest.approx(q.get("field_goal", np.nan), nan_ok=True)
        assert gt["q_star"][i, 3] == pytest.approx(q.get("punt", np.nan), nan_ok=True)
        assert np.isfinite(gt["q_star"][i, 2]) == gt["allowed_model"][i, 2]


def test_monte_carlo_under_optimal_play_converges_to_q_star(toy_sim):
    """The same Monte Carlo machinery with the solved policy and the solver's timeout and
    try rules estimates Solution.q_values (scored by the model's clipped score)."""
    sol, sim = toy_sim
    states = _toy_states(sol, 9, scores=(0, -1))
    gt = syn.ground_truth(sim, sol, states, 20000, seed=3, policy="optimal", post="model")
    ok = gt["allowed_model"]
    z = (gt["q_coach_model"][ok] - gt["q_star"][ok]) / np.maximum(gt["se_coach"][ok], 1e-3)
    assert np.abs(z).max() < 4.5


def test_ground_truth_rejects_non_fourth_downs(toy_sim):
    sol, sim = toy_sim
    st = _toy_states(sol, 5)
    st["f"] = np.full_like(st["f"], int(sol.grid.fidx(1, sol.grid.dist_bin(1), 3)))
    with pytest.raises(ValueError):
        syn.ground_truth(sim, sol, st, 10, seed=0)


def test_eval_states_and_saving_round_trip(toy_sim, tmp_path):
    sol, sim = toy_sim
    strata = ((30, 60, 0.5), (0, 30, 0.5))
    ev = syn.evaluation_states(sim, 20, seed=4, n_pool_games=300, strata=strata,
                               chunk_games=100, headline=False)  # fmt: skip
    assert len(ev) == 20 and (ev.kind == 0).all()
    records.validate_split(ev.split, min_transition_match=1.0)
    SI = S.STATE_INDEX
    st = ev.split.states[ev.idx]
    assert (st[:, SI["down"]] == sol.grid.n_down).all()
    assert (ev.split.raw[ev.idx, 0] == ev.t * 5).all()
    assert (st[:, SI["yardline"]] == sol.grid.unpack(ev.f)[2]).all()
    syn.save_split(tmp_path / "s.npz", ev.split, idx=ev.idx, kind=ev.kind,
                   **{f"sim_{k}": getattr(ev, k) for k in ("t", "s", "f", "a", "b", "r")})
    back = syn.load_eval_states(tmp_path / "s.npz")
    assert np.array_equal(back.idx, ev.idx) and np.array_equal(back.split.states, ev.split.states)


# ---------------------------------------------------------------- the real solved game
@pytest.fixture(scope="module")
def real_sim():
    if not HAVE_REAL:
        pytest.skip("needs the solved game under data/mdp")
    return syn.load_simulator(DATA_MDP)


@real
def test_real_games_are_valid(real_sim):
    sp = syn.simulate_games(real_sim[1], 300, seed=21)
    rep = records.validate_split(sp, min_transition_match=1.0)
    assert rep["info_apply_transition_match_fraction"] == 1.0
    assert set(np.unique(sp.states[:, S.STATE_INDEX["ydstogo"]])) <= set(range(0, 21))


@real
def test_real_headline_state_and_its_solved_values(real_sim):
    sol, sim = real_sim
    ev = syn.evaluation_states(sim, 12, n_pool_games=1000)
    h = int(np.flatnonzero(ev.kind == 1)[0])
    rec = records.decode_record(ev.split, int(ev.idx[h]))
    assert (rec["down"], rec["ydstogo"], rec["yardline_100"], rec["clock_seconds"]) == (4, 2, 60,
                                                                                       240)
    sign = 1 if rec["poss"] == "HOME" else -1
    assert sign * rec["score_diff_home"] == 4
    assert (rec["timeouts_home"], rec["timeouts_away"]) == (3, 3)
    records.validate_split(ev.split, min_transition_match=1.0)
    one = {k: v[h : h + 1] for k, v in ev.sim_state().items()}
    gt = syn.ground_truth(sim, sol, one, 200, seed=0)
    assert gt["q_star"][0, 0] == pytest.approx(0.778, abs=0.002)  # docs/notes/m4_mdp.md
    assert gt["q_star"][0, 3] == pytest.approx(0.714, abs=0.002)
    assert gt["go_coach_code"][0] == mg.A_PROTECT


@real
def test_saved_ground_truth_matches_this_solution(real_sim):
    path = syn.GT_DIR / f"{syn.GT_NAME}.json"
    if not path.exists():
        pytest.skip("no saved ground truth")
    res = syn.verify_ground_truth(real_sim[0].meta, path, sol=real_sim[0])
    assert res["ok"], res["mismatches"]
