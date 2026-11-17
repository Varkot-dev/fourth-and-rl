"""Tests for the M5 environments, agents and evaluation.

Everything except the `data` section runs on tiny hand-made models (`rl.toy`), so CI
needs neither the play-by-play cache nor the solved arrays. It does need gymnasium (the
`rl` extra); without it the module is skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from fourth_and_rl import mdp_game as mg  # noqa: E402
from fourth_and_rl.mdp_drive import solve_drive  # noqa: E402
from fourth_and_rl.paths import DATA  # noqa: E402
from fourth_and_rl.rl import envs, toy  # noqa: E402
from fourth_and_rl.rl import evaluate as ev  # noqa: E402
from fourth_and_rl.rl.reinforce import (  # noqa: E402
    ReinforceConfig,
    gradient_variance,
    reinforce_tabular,
)
from fourth_and_rl.rl.tabular import Schedule, td_control  # noqa: E402

SNAPSHOT = DATA / "rl_snapshot" / "mdp"


@pytest.fixture(scope="module")
def drive() -> envs.DriveModel:
    return toy.toy_drive(0)


@pytest.fixture(scope="module")
def game() -> mg.Solution:
    return toy.toy_game(0)


@pytest.fixture(scope="module")
def sim(game: mg.Solution) -> mg.Simulator:
    return mg.Simulator(game, coach=toy.toy_coach_table(game.grid))


# --------------------------------------------------------------------------------------
# 1. The drive environment's optimum is the stage 1 optimum
# --------------------------------------------------------------------------------------


def test_chain_drive_by_hand() -> None:
    # Yard line 2: go reaches the 1 or hands the other team the ball at THEIR 1, worth
    # 0.5 V(1) - 0.5 V(1) = 0; punting hands them their 2, worth -V(2). So V(2) = 0.
    # Yard line 1: go = 0.5 * 7 + 0.5 * (-V(2)) = 3.5; field goal = 0.9 * (3 - 0) = 2.7.
    m = toy.chain_drive()
    assert m.v_star == pytest.approx([3.5, 0.0], abs=1e-8)
    assert m.q_star[0] == pytest.approx([3.5, 2.7, 0.0], abs=1e-8)
    assert m.policy_star[0] == envs.GO
    assert ev.drive_regret(m, m.policy_star) == pytest.approx(0.0, abs=1e-8)


def test_drive_model_optimum_is_stage_one(drive: envs.DriveModel) -> None:
    # The stage 1 values satisfy the environment's Bellman optimality equation ...
    best = np.where(drive.allowed, drive.q_star, -np.inf).max(axis=1)
    assert np.allclose(best, drive.v_star, atol=1e-8)
    # ... the environment's optimal policy, found from scratch by policy iteration on the
    # environment alone, has exactly the stage 1 values ...
    pi = ev.best_restricted_policy(drive, (envs.GO, envs.FIELD_GOAL, envs.PUNT))
    assert np.allclose(ev.policy_value(drive, pi), drive.v_star, atol=1e-8)
    assert ev.drive_regret(drive, pi) == pytest.approx(0.0, abs=1e-8)
    # ... and fixing the fourth-down policy in stage 1 gives the same value as evaluating
    # that policy in the environment (two independent computations).
    rng = np.random.default_rng(1)
    mix = ev.as_probs(rng.dirichlet(np.ones(3), size=drive.n), drive.allowed)
    full = np.tile([1.0, 0.0, 0.0], (drive.grid.nf, 1))  # rows of downs 1-3 are not used
    full[drive.states] = mix
    stage1 = solve_drive(
        drive.kern, 7.0, 3.0, 2.0, fourth_policy=full, tolerance=1e-12, max_iterations=20000
    )
    # Stage 1 lets the continuation follow the same fixed policy, the environment freezes
    # it at the optimum, so compare the environment against itself with V* replaced:
    model_pi = envs.build_drive_model(drive.kern, 7.0, 3.0, 2.0, v_field=stage1["v"])
    assert np.allclose(ev.policy_value(model_pi, mix), stage1["v"][drive.states], atol=1e-7)


def test_exact_policy_evaluation_matches_monte_carlo(
    drive: envs.DriveModel, sim: mg.Simulator
) -> None:
    env = envs.DriveEnv(drive, sim.play["neutral"], sim.punt)
    rng = np.random.default_rng(0)
    pol = ev.as_probs(rng.dirichlet(np.ones(3), size=drive.n), drive.allowed)
    exact = float(drive.start @ ev.policy_value(drive, pol))
    returns = []
    obs, info = env.reset(seed=5)
    for _ in range(6000):
        while True:
            a = int(rng.choice(3, p=pol[int(obs)]))
            obs, r, term, trunc, info = env.step(a)
            if term:
                returns.append(r)
                obs, info = env.reset()
                break
    se = np.std(returns) / np.sqrt(len(returns))
    assert abs(np.mean(returns) - exact) < 4 * se
    # The batch sampler draws from the same model.
    batch = envs.DriveBatch(drive, 20000, seed=3)
    total, alive = np.zeros(batch.n), np.ones(batch.n, dtype=bool)
    cum = np.cumsum(pol, axis=1)
    while alive.any():
        a = np.minimum((rng.random(batch.n)[:, None] > cum[batch.s]).sum(axis=1), 2)
        r, d = batch.step(a)
        total += np.where(alive, r, 0.0)
        alive &= ~d
    assert abs(total.mean() - exact) < 4 * total.std() / np.sqrt(batch.n)


def test_drive_baselines_and_agreement(drive: envs.DriveModel) -> None:
    base = ev.drive_baselines(drive)
    assert base["never_go"] > 0 and base["random"] > 0
    agree = ev.drive_agreement(drive, drive.policy_star)
    assert agree["weighted_by_optimal_visits"] == pytest.approx(1.0)
    by_state = ev.regret_by_state(drive, np.full(drive.n, envs.PUNT))
    assert by_state.sum() == pytest.approx(ev.drive_regret(drive, np.full(drive.n, envs.PUNT)))


# --------------------------------------------------------------------------------------
# 2. Gymnasium conformance and seeding
# --------------------------------------------------------------------------------------


def test_check_env(drive: envs.DriveModel, sim: mg.Simulator) -> None:
    for mode in ("index", "features", "onehot"):
        check_env(envs.DriveEnv(drive, sim.play["neutral"], sim.punt, obs_mode=mode))
    for mode in ("features", "index"):
        check_env(envs.EndgameEnv(sim, obs_mode=mode))
    check_env(envs.EndgameEnv(sim, opponent="coach"))


def _rollout(env, seed: int, actions: list[int]) -> list:
    out = []
    obs, _ = env.reset(seed=seed)
    for a in actions:
        obs, r, term, trunc, _ = env.step(a)
        out.append((np.asarray(obs).tolist(), r, term))
        if term:
            obs, _ = env.reset()
    return out


def test_seeding_is_deterministic(drive: envs.DriveModel, sim: mg.Simulator) -> None:
    acts = [0, 2, 1, 0, 0, 2, 0, 1] * 10
    d1 = envs.DriveEnv(drive, sim.play["neutral"], sim.punt)
    d2 = envs.DriveEnv(drive, sim.play["neutral"], sim.punt)
    assert _rollout(d1, 11, acts) == _rollout(d2, 11, acts)
    assert _rollout(d1, 11, acts) != _rollout(d1, 12, acts)
    acts = [0, 1, 2, 3, 4, 5] * 10
    e1, e2 = envs.EndgameEnv(sim), envs.EndgameEnv(sim)
    assert _rollout(e1, 4, acts) == _rollout(e2, 4, acts)
    b1, b2 = envs.EndgameBatch(sim, 50, seed=9), envs.EndgameBatch(sim, 50, seed=9)
    b1.reset()
    b2.reset()
    for _ in range(5):
        a = np.arange(50) % 6
        assert np.array_equal(b1.step(a)[0], b2.step(a)[0])


# --------------------------------------------------------------------------------------
# 3. End game: the optimal policy earns the solved value; the regret estimator is right
# --------------------------------------------------------------------------------------


def test_endgame_optimal_policy_earns_solved_value(sim: mg.Simulator) -> None:
    starts = ev.sample_starts(sim, 30000, seed=2, spec=envs.StartSpec(seconds_min=10))
    out = ev.evaluate_endgame(sim, ev.optimal_policy, starts, seed=1)
    diff = out["win_rate"] - out["optimal_win_rate_of_starts"]
    assert abs(diff) < 4 * out["win_rate_se"]
    assert out["same_action_as_optimal"] == 1.0


def test_endgame_regret_estimator(sim: mg.Simulator, tmp_path: Path) -> None:
    loss = ev.load_loss_table(sim.sol, tmp_path / "loss.npy")
    starts = ev.sample_starts(sim, 30000, seed=3, spec=envs.StartSpec(seconds_min=10))
    opt = ev.evaluate_endgame(sim, ev.optimal_policy, starts, seed=1, loss=loss)
    assert opt["regret"] == pytest.approx(0.0, abs=1e-3)
    rnd = ev.evaluate_endgame(sim, ev.random_policy, starts, seed=1, loss=loss)
    # V*(start) - win rate and the summed per-decision losses estimate the same number.
    gap = rnd["optimal_win_rate_of_starts"] - rnd["win_rate"]
    assert rnd["regret"] > 0.01
    assert abs(rnd["regret"] - gap) < 4 * rnd["win_rate_se"]
    # The loss-based estimate is the less noisy of the two.
    assert rnd["regret_se"] < rnd["win_rate_se"]


# --------------------------------------------------------------------------------------
# 4. The hand-written learners find the known optimum
# --------------------------------------------------------------------------------------


def _regret_cb(model: envs.DriveModel):
    return lambda q, t, e: ev.drive_regret(model, ev.greedy(q, model.allowed))


@pytest.mark.parametrize("algo", ["q_learning", "sarsa", "expected_sarsa"])
def test_td_learners_converge(drive: envs.DriveModel, algo: str) -> None:
    env = envs.CompiledDrive(drive)
    out = td_control(
        env, 300_000, seed=0, algo=algo, schedule=Schedule(eps_decay_frac=0.6),
        checkpoints=[300_000], on_checkpoint=_regret_cb(drive),
    )
    assert out["records"][-1] < 0.01
    # Q of the optimal actions is close to the true values in states visited often.
    rows = np.arange(drive.n)
    often = out["counts"][rows, drive.policy_star] > 5000
    q_best = out["q"][rows, drive.policy_star]
    assert np.allclose(q_best[often], drive.v_star[often], atol=0.2)


def test_q_learning_on_chain_finds_hand_solution() -> None:
    m = toy.chain_drive()
    out = td_control(envs.CompiledDrive(m), 50_000, seed=1)
    assert out["q"][0] == pytest.approx([3.5, 2.7, 0.0], abs=0.1)
    assert ev.greedy(out["q"], m.allowed)[0] == envs.GO


def test_reinforce_converges_and_baseline_reduces_variance(drive: envs.DriveModel) -> None:
    env = envs.CompiledDrive(drive)
    out = reinforce_tabular(
        env, 300_000, seed=0, cfg=ReinforceConfig(lr=0.1),
        checkpoints=[300_000], on_checkpoint=lambda p, t, e: ev.drive_regret(drive, p),
    )
    assert out["records"][-1] < 0.02
    uniform = drive.allowed / drive.allowed.sum(axis=1, keepdims=True)
    v = ev.policy_value(drive, uniform)
    var = gradient_variance(drive, env, uniform, v, 4000, seed=0)
    assert var["variance_ratio"] < 0.8


def test_reinforce_network_learns(drive: envs.DriveModel) -> None:
    torch = pytest.importorskip("torch")
    from fourth_and_rl.rl.reinforce import NetReinforceConfig, reinforce_network

    torch.set_num_threads(1)

    out = reinforce_network(
        drive, 100_000, seed=0, cfg=NetReinforceConfig(hidden=32, lr=1e-2, episodes_per_update=512)
    )
    uniform = drive.allowed / drive.allowed.sum(axis=1, keepdims=True)
    assert ev.drive_regret(drive, out["probs"]) < 0.5 * ev.drive_regret(drive, uniform)


def test_sb3_adapter_runs(sim: mg.Simulator, drive: envs.DriveModel) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import DQN, PPO

    from fourth_and_rl.rl.sb3_utils import BatchVecEnv, masked_greedy

    torch.set_num_threads(1)
    venv = BatchVecEnv(envs.EndgameBatch(sim, 8, seed=0))
    model = PPO("MlpPolicy", venv, n_steps=32, batch_size=64, n_epochs=1, seed=0, device="cpu")
    model.learn(512)
    assert venv.games_finished > 0
    b = envs.EndgameBatch(sim, 4, seed=1)
    b.reset()
    a = masked_greedy(model, b.observe(), b.mask())
    assert b.mask()[np.arange(4), a].all()
    venv = BatchVecEnv(envs.DriveBatch(drive, 8, seed=0), obs_mode="onehot")
    DQN("MlpPolicy", venv, learning_starts=50, seed=0, device="cpu").learn(400)


# --------------------------------------------------------------------------------------
# 5. On the frozen snapshot of the real solution
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real() -> tuple[mg.Solution, mg.Simulator, envs.DriveModel]:
    if not mg.solution_exists(SNAPSHOT):
        pytest.skip("solved snapshot not present under data/rl_snapshot/mdp")
    sol = mg.load_solution(SNAPSHOT)
    m = sol.model
    dm = envs.build_drive_model(m.kern, m.td_points + m.p_xp, m.fg_points, m.safety_points)
    return sol, mg.Simulator(sol), dm


@pytest.mark.data
def test_real_drive_env_optimum(real) -> None:
    sol, sim, dm = real
    best = np.where(dm.allowed, dm.q_star, -np.inf).max(axis=1)
    assert np.allclose(best, dm.v_star, atol=1e-7)
    assert int((dm.policy_star == envs.GO).sum()) > 0
    env = envs.DriveEnv.from_simulator(sim, dm)
    obs, _ = env.reset(seed=0)
    returns = []
    while len(returns) < 4000:
        obs, r, term, _, _ = env.step(int(dm.policy_star[int(obs)]))
        if term:
            returns.append(r)
            obs, _ = env.reset()
    se = np.std(returns) / np.sqrt(len(returns))
    assert abs(np.mean(returns) - dm.start @ dm.v_star) < 4 * se


@pytest.mark.data
def test_real_endgame_optimal_policy(real) -> None:
    sol, sim, _ = real
    starts = ev.sample_starts(sim, 20000, seed=7)
    out = ev.evaluate_endgame(sim, ev.optimal_policy, starts, seed=8)
    assert abs(out["win_rate"] - out["optimal_win_rate_of_starts"]) < 4 * out["win_rate_se"]
