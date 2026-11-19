"""Tests for the M6 offline-RL code.

Everything except the `data` section runs on the tiny hand-made drive models of
`rl.toy`, so CI needs neither the play-by-play cache nor the solved arrays (it needs
gymnasium, which `rl.envs` imports). The d3rlpy test is skipped without d3rlpy and torch.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("gymnasium")

from fourth_and_rl.offline import datasets as ds  # noqa: E402
from fourth_and_rl.offline import learners as L  # noqa: E402
from fourth_and_rl.offline import measure as M  # noqa: E402
from fourth_and_rl.offline import real  # noqa: E402
from fourth_and_rl.paths import DATA  # noqa: E402
from fourth_and_rl.rl import evaluate as ev  # noqa: E402
from fourth_and_rl.rl import toy  # noqa: E402
from fourth_and_rl.rl.envs import FIELD_GOAL, GO, PUNT, DriveModel  # noqa: E402


@pytest.fixture(scope="module")
def drive() -> DriveModel:
    return toy.toy_drive(0)


def exact_counts(model: DriveModel, per_cell: float = 1e6, actions=(0, 1, 2)) -> np.ndarray:
    """'Infinite data': every allowed cell logged per_cell times, in exact proportions."""
    c = np.zeros((3, model.n, model.n + model.n_term))
    for a in actions:
        c[a] = np.where(model.allowed[:, a][:, None], per_cell * model.dist[a], 0.0)
    return c


def uniform(model: DriveModel) -> np.ndarray:
    return model.allowed / model.allowed.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------


def test_generation_is_seed_deterministic(drive: DriveModel) -> None:
    mu = uniform(drive)
    a = ds.generate(drive, mu, 5000, seed=3, n_parallel=64)
    b = ds.generate(drive, mu, 5000, seed=3, n_parallel=64)
    c = ds.generate(drive, mu, 5000, seed=4, n_parallel=64)
    assert a.size == 5000
    assert np.array_equal(a.s, b.s) and np.array_equal(a.a, b.a) and np.array_equal(a.c, b.c)
    assert not np.array_equal(a.c, c.c)
    # Episodes are contiguous: a transition that does not end its drive leads to the
    # state of the next transition.
    cont = ~a.done[:-1]
    assert np.array_equal(a.c[:-1][cont], a.s[1:][cont])
    counts = ds.counts(a, drive.n + drive.n_term)
    assert counts.sum() == 5000
    nested = ds.counts_by_size(a, drive.n + drive.n_term, [1000, 5000])
    assert np.array_equal(nested[5000], counts)
    assert np.array_equal(nested[1000], ds.counts(a.head(1000), drive.n + drive.n_term))


def test_generated_data_follow_the_model(drive: DriveModel) -> None:
    """Empirical outcome frequencies match model.dist (a check of the sampler)."""
    mu = uniform(drive)
    data = ds.generate(drive, mu, 200_000, seed=1)
    p, n_sa = L.empirical(ds.counts(data, drive.n + drive.n_term))
    busy = n_sa > 5000
    for a in range(3):
        rows = np.nonzero(busy[:, a])[0]
        assert np.abs(p[a, rows] - drive.dist[a, rows]).sum(axis=1).max() < 0.08


# --------------------------------------------------------------------------------------
# Learners on a toy MDP with a known optimum
# --------------------------------------------------------------------------------------


def test_fqi_with_full_coverage_recovers_q_star(drive: DriveModel) -> None:
    fit = L.fqi_tabular(drive, exact_counts(drive))
    ok = drive.allowed
    assert np.allclose(fit.q[ok], drive.q_star[ok], atol=1e-6)
    assert ev.drive_regret(drive, fit.policy) == pytest.approx(0.0, abs=1e-9)
    # And from sampled data with full coverage (uniform behavior), approximately.
    data = ds.generate(drive, uniform(drive), 300_000, seed=2)
    fit = L.fqi_tabular(drive, ds.counts(data, drive.n + drive.n_term))
    reach = M.reachable(drive)
    sel = reach[:, None] & ok
    err = np.abs(fit.q[sel] - drive.q_star[sel])
    assert err.mean() < 0.05
    assert ev.drive_regret(drive, fit.policy) < 0.02


def test_unlogged_action_fqi_keeps_init_cql_pushes_it_down(drive: DriveModel) -> None:
    c = exact_counts(drive, per_cell=200.0)
    s0 = int(np.nonzero(drive.allowed[:, FIELD_GOAL])[0][0])
    c[GO, s0] = 0.0  # go is never logged in state s0
    for q_init in (0.0, 5.0):
        fqi = L.fqi_tabular(drive, c, q_init=q_init)
        assert fqi.q[s0, GO] == q_init  # whatever the initialization says
    cql = L.cql_tabular(drive, c, alpha=0.5, q_init=5.0)
    logged = [a for a in (FIELD_GOAL, PUNT) if drive.allowed[s0, a]]
    assert all(cql.q[s0, GO] < cql.q[s0, a] for a in logged)
    assert cql.policy[s0] != GO
    # The logged cells sit at the CQL stationary point q = T - alpha (pi / mu - 1).
    assert cql.info["stationarity_residual_logged"] < 1e-4


def test_cql_matches_its_closed_form_on_one_state() -> None:
    """One state, actions logged in proportions mu, fixed targets: the minimizer of
    sum_a mu_a (q_a - T_a)^2 / 2 + alpha (lse(q) - sum_a mu_a q_a) satisfies
    q_a = T_a - alpha (softmax(q)_a / mu_a - 1). Checked with the chain model (its yard
    line 1 state has go and a field goal, both terminal on every outcome)."""
    m = toy.chain_drive()
    c = np.zeros((3, m.n, m.n + m.n_term))
    c[GO, 0] = 30 * m.dist[GO, 0]
    c[FIELD_GOAL, 0] = 70 * m.dist[FIELD_GOAL, 0]
    c[PUNT, 0] = 0.0
    c[:, 1] = 100 * m.dist[:, 1] * m.allowed[1][:, None]
    alpha = 0.3
    fit = L.cql_tabular(m, c, alpha, outer_iter=200, inner_steps=50)
    q = fit.q[0]
    t = np.array([m.q_star[0, GO], m.q_star[0, FIELD_GOAL]])  # targets are terminal here
    mu = np.array([0.3, 0.7])
    pi = np.exp(q - q[np.isfinite(q)].max())
    pi = pi / pi[np.isfinite(pi)].sum()
    expect = t - alpha * (pi[:2] / mu - 1.0)
    assert q[:2] == pytest.approx(expect, abs=1e-6)
    # An allowed action that is never logged ends below every logged one.
    assert q[PUNT] < q[:2].min()


def test_behavior_cloning_recovers_the_behavior(drive: DriveModel) -> None:
    rng = np.random.default_rng(0)
    mu = ev.as_probs(rng.dirichlet(np.ones(3), drive.n), drive.allowed)
    data = ds.generate(drive, mu, 300_000, seed=5)
    c = ds.counts(data, drive.n + drive.n_term)
    fit = L.behavior_cloning(drive, c)
    busy = L.cell_counts(c).sum(axis=1) > 3000
    assert busy.sum() >= 5
    assert np.abs(fit.policy[busy] - mu[busy]).max() < 0.03
    assert abs(ev.drive_regret(drive, fit.policy) - ev.drive_regret(drive, mu)) < 0.02


def test_plugin_with_full_coverage_recovers_the_optimum(drive: DriveModel) -> None:
    fit = L.plugin(drive, exact_counts(drive, per_cell=1e9))
    ok = drive.allowed
    assert np.allclose(fit.q[ok], drive.q_star[ok], atol=1e-6)
    assert ev.drive_regret(drive, fit.policy) == pytest.approx(0.0, abs=1e-9)


def test_shift_columns_moves_states_and_takeovers(drive: DriveModel) -> None:
    g = drive.grid
    cols = L.shift_columns(drive, 1)
    _, dbin, yl = g.unpack(drive.states)
    pos = {int(f): i for i, f in enumerate(drive.states)}
    for i in range(drive.n):
        if yl[i] < g.n_yl:
            assert cols[i] == pos[int(g.fidx(g.n_down, dbin[i], yl[i] + 1))]
        else:
            assert cols[i] == drive.n + g.n_yl + 1  # pushed behind our goal: a safety
    # "They take over at their y" moves one yard closer to our end zone.
    assert cols[drive.n + 3] == drive.n + 2
    assert cols[drive.n + g.n_yl] == drive.n + g.n_yl  # a touchdown stays a touchdown


def test_fqe_evaluates_a_fixed_policy(drive: DriveModel) -> None:
    pol = np.where(drive.allowed[:, PUNT], PUNT, GO)
    q = L.fqe_tabular(drive, exact_counts(drive), pol)
    v = ev.policy_value(drive, pol)
    assert np.allclose(q[np.arange(drive.n), pol], v, atol=1e-6)


def test_measure_optimizers_curse_is_zero_for_exact_values(drive: DriveModel) -> None:
    c = exact_counts(drive, per_cell=100.0)
    fit = L.fqi_tabular(drive, c)
    res = M.evaluate_fit(drive, fit, c, uniform(drive), M.reachable(drive))
    assert res["regret"] == pytest.approx(0.0, abs=1e-9)
    assert res["optimizers_curse"]["all"]["mean_claimed_minus_true"] == pytest.approx(0, abs=1e-6)


# --------------------------------------------------------------------------------------
# Off-policy evaluation
# --------------------------------------------------------------------------------------


def test_importance_sampling_identities(drive: DriveModel) -> None:
    mu = uniform(drive)
    data = ds.generate(drive, mu, 20_000, seed=9)
    eps = real.episodes(data, drive)
    zero = np.zeros((drive.n, 3))
    on = real.ope_per_episode(eps, drive, mu, mu, zero)
    assert np.allclose(on["ratio"], 1.0)
    assert np.allclose(on["is"], on["return"])
    assert np.allclose(on["dr"], on["is"])  # DR with a zero model is plain IS
    # Off-policy: IS is unbiased for the target's value (within Monte Carlo error), and DR
    # with the exact Q of the target is unbiased with a smaller spread.
    target = drive.policy_star
    q_exact = ev.policy_q(drive, target)
    off = real.ope_per_episode(eps, drive, target, mu, q_exact)
    start = ds.start_states(data)
    truth = float(start @ ev.policy_value(drive, target))
    se = off["is"].std() / np.sqrt(len(eps))
    assert abs(off["is"].mean() - truth) < 4 * se
    assert off["dr"].std() < off["is"].std()


# --------------------------------------------------------------------------------------
# Real-data transition builder on a hand-made two-game frame
# --------------------------------------------------------------------------------------


def _frame() -> pl.DataFrame:
    cols = ("game_id", "play_id", "posteam", "game_half", "qtr", "down", "ydstogo",
            "yardline_100", "play_type", "fourth_down_decision", "touchdown", "td_team",
            "safety", "field_goal_result", "game_seconds_remaining")
    rows = [
        # Game 1. A goes on 4th and 2 at the 40, converts; a penalty no-play; then a made
        # field goal on 4th and 7 at the 25: the go leads to that next decision, the field
        # goal ends the drive.
        ("g1", 1, "A", "Half1", 1, 4, 2, 40, "pass", "go", 0, None, 0, None, 3400),
        ("g1", 2, "A", "Half1", 1, 1, 10, 34, "run", None, 0, None, 0, None, 3360),
        ("g1", 3, "A", "Half1", 1, 2, 15, 39, "no_play", None, 0, None, 0, None, 3330),
        ("g1", 4, "A", "Half1", 1, 4, 7, 25, "field_goal", "field_goal", 0, None, 0, "made",
         3300),
        ("g1", 5, "B", "Half1", 1, None, None, 35, "kickoff", None, 0, None, 0, None, 3295),
        # B punts on 4th and 9 at its 70; A's first snap is at A's 85.
        ("g1", 6, "B", "Half1", 1, 4, 9, 70, "punt", "punt", 0, None, 0, None, 3200),
        ("g1", 7, "A", "Half1", 1, 1, 10, 85, "run", None, 0, None, 0, None, 3190),
        # A goes on 4th and 1 at the 50 and fails; B's first snap is at B's 50.
        ("g1", 8, "A", "Half1", 2, 4, 1, 50, "run", "go", 0, None, 0, None, 3000),
        ("g1", 9, "B", "Half1", 2, 1, 10, 50, "pass", None, 0, None, 0, None, 2990),
        # Game 2. C scores on 4th and goal at the 3.
        ("g2", 1, "C", "Half1", 1, 4, 3, 3, "pass", "go", 1, "C", 0, None, 3500),
        ("g2", 2, "C", "Half1", 1, None, None, 15, "extra_point", None, 0, None, 0, None,
         3500),
        ("g2", 3, "D", "Half1", 1, None, None, 35, "kickoff", None, 0, None, 0, None, 3500),
        # D punts and C returns it for a touchdown.
        ("g2", 4, "D", "Half1", 1, 4, 10, 80, "punt", "punt", 1, "C", 0, None, 3400),
        ("g2", 5, "C", "Half1", 1, None, None, 15, "extra_point", None, 0, None, 0, None,
         3400),
        # D punts from its 95; C muffs and D recovers (D's snap follows); D's next fourth
        # down is a punt at the 40, and the half ends before anyone snaps again.
        ("g2", 6, "D", "Half1", 2, 4, 15, 95, "punt", "punt", 0, None, 0, None, 2000),
        ("g2", 7, "D", "Half1", 2, 1, 10, 45, "run", None, 0, None, 0, None, 1990),
        ("g2", 8, "D", "Half1", 2, 4, 5, 40, "punt", "punt", 0, None, 0, None, 1950),
        ("g2", 9, "C", "Half2", 3, None, None, 35, "kickoff", None, 0, None, 0, None, 1800),
        # A late fourth down when trailing (not the neutral situation): left out.
        ("g2", 10, "C", "Half2", 4, 4, 3, 30, "pass", "go", 0, None, 0, None, 100),
    ]
    df = pl.DataFrame(rows, schema=list(cols), orient="row")
    return df.with_columns(
        pl.lit(2015).alias("season"),
        pl.col("play_id").alias("order_sequence"),
        pl.when(pl.col("game_half") == "Half1").then(pl.col("game_seconds_remaining") - 1800)
        .otherwise(pl.col("game_seconds_remaining")).alias("half_seconds_remaining"),
        pl.when(pl.col("play_id") == 10).then(-3).otherwise(0).alias("score_differential"),
    )


def test_real_transition_builder_on_two_games(drive: DriveModel) -> None:
    from fourth_and_rl.mdp import Grid

    grid = Grid()
    out = real.decision_outcomes(_frame(), grid).sort("game_id", "play_id")
    got = [(r["game_id"], int(r["play_id"]), real.ACTIONS[r["action"]], r["kind"])
           for r in out.iter_rows(named=True)]
    assert got == [
        ("g1", 1, "go", "next"),
        ("g1", 4, "field_goal", "fg"),
        ("g1", 6, "punt", "opp_ball"),
        ("g1", 8, "go", "opp_ball"),
        ("g2", 1, "go", "td"),
        ("g2", 4, "punt", "opp_td"),
        ("g2", 6, "punt", "next"),
        ("g2", 8, "punt", "truncated"),
    ]
    by = {(r["game_id"], int(r["play_id"])): r for r in out.iter_rows(named=True)}
    assert by[("g1", 1)]["next_yl"] == 25
    assert by[("g1", 1)]["next_dbin"] == int(grid.dist_bin(7))
    assert by[("g1", 6)]["opp_yl"] == 85
    assert by[("g1", 8)]["opp_yl"] == 50
    assert by[("g2", 6)]["next_yl"] == 40
    assert by[("g1", 1)]["dbin"] == int(grid.dist_bin(2)) and by[("g1", 1)]["yl"] == 40
    # Including every situation adds the late fourth down.
    assert real.decision_outcomes(_frame(), grid, neutral_only=False).height == 9


def test_real_transitions_map_to_model_columns() -> None:
    """On the full football grid (no solved model needed): build a drive model shell
    with the real grid's layout and check columns and episode chaining."""
    from fourth_and_rl.mdp import Grid

    grid = Grid()
    out = real.decision_outcomes(_frame(), grid)

    class Shell:  # just what to_logged needs
        pass

    m = Shell()
    m.grid = grid
    downs = grid.unpack(np.arange(grid.nf))[0]
    m.states = np.nonzero(downs == grid.n_down)[0]
    m.n = len(m.states)
    m.allowed = np.ones((m.n, 3), dtype=bool)
    data, kept, dropped = real.to_logged(out, m)
    assert dropped["truncated_end_of_half"] == 1
    assert data.size == 7
    n, n_yl = m.n, grid.n_yl
    pos = {int(f): i for i, f in enumerate(m.states)}
    cols = {(r["game_id"], int(r["play_id"])): int(r["c"]) for r in kept.iter_rows(named=True)}
    assert cols[("g1", 1)] == pos[int(grid.fidx(4, grid.dist_bin(7), 25))]
    assert cols[("g1", 4)] == n + n_yl + 3
    assert cols[("g1", 6)] == n + 85 - 1
    assert cols[("g2", 1)] == n + n_yl
    assert cols[("g2", 4)] == n + n_yl + 2
    # The go at g1 and the field goal it led to form one episode; g2's muffed punt leads to
    # a decision that was dropped (truncated), so its episode is cut after one step.
    ep = {(r["game_id"], int(r["play_id"])): int(r["episode"]) for r in kept.iter_rows(named=True)}
    assert ep[("g1", 1)] == ep[("g1", 4)]
    assert len(set(ep.values())) == 6
    assert int(data.first.sum()) == 6


# --------------------------------------------------------------------------------------
# d3rlpy adapter (skipped without d3rlpy and torch)
# --------------------------------------------------------------------------------------


def test_d3rlpy_dataset_and_short_fit(drive: DriveModel) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("d3rlpy")
    from fourth_and_rl.offline import deep

    data = ds.generate(drive, uniform(drive), 3000, seed=11, n_parallel=32)
    mdp = deep.to_mdp_dataset(drive, data, 0.1)
    assert mdp.transition_count == data.size
    cfg = {"hidden_units": [16], "learning_rate": 1e-3, "batch_size": 64,
           "target_update_interval": 50, "n_steps": 200, "reward_scale": 0.1, "alpha": 1.0}
    for algo in ("dqn", "cql"):
        fit = deep.fit_d3rlpy(drive, data, algo, 0, cfg)
        assert fit.q.shape == (drive.n, 3)
        assert np.isfinite(fit.q[drive.allowed]).all()


# --------------------------------------------------------------------------------------
# Real data and the solved model (skipped when absent, as in CI)
# --------------------------------------------------------------------------------------


@pytest.mark.data
def test_real_dev_transitions_match_the_configured_size() -> None:
    from fourth_and_rl import splits
    from fourth_and_rl.rl import runner

    if not (DATA / "mdp" / "meta.json").exists() or not splits.dev_path().exists():
        pytest.skip("solved model or development plays not available")
    cfg = runner.load_config(runner.resolve("configs/offline.yaml"))
    snap = runner.Snapshot(cfg)
    data, _, info = real.build_logged(splits.load_dev_plays(), snap.drive, "test")
    assert data.size == cfg["known_truth"]["real_size"]
    assert info["dropped"]["action_not_allowed_in_model"] == 0
    # The coach-like behavior is the coach table's neutral slice.
    mu = ds.coach_policy(snap.drive, snap.coach["table"])
    assert np.allclose(mu, ev.coach_drive_policy(snap.drive, snap.coach["table"]))
