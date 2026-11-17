"""M5 part 2: deep agents (hand-written REINFORCE network, Stable-Baselines3 DQN and PPO).

Run:  .venv/bin/python scripts/rl_deep.py               (about 30 to 40 minutes, 2 processes)
      .venv/bin/python scripts/rl_deep.py --only drive
          (one of drive | endgame | sparsity | coach | approx)
      .venv/bin/python scripts/rl_deep.py --force       (rerun everything)

Every run is saved under data/rl/runs/deep/ when it finishes (resumable); trained networks
go to data/models/rl/. Built from the frozen snapshot named in configs/rl.yaml.

Measurement
  drive    exact regret in expected points (linear solve) of the greedy policy over all
           1,188 fourth-down states, and for policies that are stochastic, of the policy
  endgame  regret in win probability on a fixed set of start states (the same games for
           every agent and checkpoint): the sum over the agent's decisions of the exact loss
           V*(s) - Q*(s, a), read from a table built with the solver's own backup

Writes results/m5_rl_deep.json and redraws the three M5 figures.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from fourth_and_rl import mdp_game as mg  # noqa: E402
from fourth_and_rl.io import write_json  # noqa: E402
from fourth_and_rl.paths import RESULTS  # noqa: E402
from fourth_and_rl.rl import evaluate as ev  # noqa: E402
from fourth_and_rl.rl import figures, runner  # noqa: E402
from fourth_and_rl.rl.envs import DriveBatch, EndgameBatch, StartSpec  # noqa: E402

GROUP = "deep"


def _torch(cfg: dict) -> Any:
    import torch

    torch.set_num_threads(cfg["deep"]["torch_threads"])
    return torch


def _spec(cfg: dict) -> StartSpec:
    return StartSpec(**cfg["endgame"]["start"])


_EVAL: dict[str, Any] = {}


def endgame_eval_set(cfg: dict) -> tuple[dict, np.ndarray]:
    """The fixed evaluation start states and the loss table (built once, then memory-mapped)."""
    if not _EVAL:
        snap = runner.snapshot(cfg)
        ec = cfg["endgame"]
        _EVAL["starts"] = ev.sample_starts(snap.sim, ec["eval_games"], ec["eval_seed"], _spec(cfg))
        _EVAL["loss"] = ev.load_loss_table(snap.sol, runner.resolve(cfg["loss_table"]))
    return _EVAL["starts"], _EVAL["loss"]


def sb3_policy(model: Any) -> ev.Policy:
    from fourth_and_rl.rl.sb3_utils import masked_greedy

    def pol(batch: EndgameBatch, idx: np.ndarray) -> np.ndarray:
        return masked_greedy(model, batch.observe(idx), batch.mask(idx))

    return pol


def _sb3(algo: str, venv: Any, cfg: dict, seed: int) -> Any:
    from stable_baselines3 import DQN, PPO

    if algo == "dqn":
        c = dict(cfg["deep"]["dqn"])
        arch = c.pop("net_arch")
        c["train_freq"] = (c["train_freq"], "step")
        return DQN("MlpPolicy", venv, policy_kwargs={"net_arch": arch}, seed=seed,
                   device="cpu", **c)
    c = dict(cfg["deep"]["ppo"])
    arch = c.pop("net_arch")
    return PPO("MlpPolicy", venv, policy_kwargs={"net_arch": arch}, seed=seed, device="cpu", **c)


# --------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------


def run_reinforce_net(cfg: dict, seed: int) -> dict:
    torch = _torch(cfg)
    from fourth_and_rl.rl.reinforce import NetReinforceConfig, reinforce_network

    snap = runner.snapshot(cfg)
    model = snap.drive
    rc = cfg["deep"]["reinforce_net"]
    conf = NetReinforceConfig(hidden=rc["hidden"], lr=rc["lr"], baseline_lr=rc["baseline_lr"],
                              episodes_per_update=rc["episodes_per_update"])
    curve: list[dict] = []

    def on_cp(probs: np.ndarray, t: int, episodes: int) -> None:
        curve.append({"steps": t, "episodes": episodes, "regret": ev.drive_regret(model, probs),
                      "regret_greedy": ev.drive_regret(model, probs.argmax(axis=1))})

    t0 = time.time()
    out = reinforce_network(model, rc["n_steps"], seed, conf,
                            ev.log_checkpoints(rc["n_steps"], cfg["drive"]["n_checkpoints"]), on_cp)
    path = runner.resolve(cfg["models_dir"]) / f"reinforce_net_drive_{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": out["policy_net"].state_dict(), "value": out["value_net"].state_dict()},
               path)
    pol = out["probs"].argmax(axis=1)
    return {"algo": "reinforce_net", "env": "drive", "seed": seed, "config": conf.__dict__,
            "curve": curve, "final_policy": pol.tolist(),
            "agreement": ev.drive_agreement(model, pol), "seconds": time.time() - t0}


def run_sb3_drive(cfg: dict, algo: str, seed: int) -> dict:
    _torch(cfg)
    from fourth_and_rl.rl.sb3_utils import BatchVecEnv, CheckpointCallback, action_scores

    snap = runner.snapshot(cfg)
    model = snap.drive
    dc = cfg["deep"]["drive"]
    n_steps = dc[f"{algo}_steps"]
    venv = BatchVecEnv(DriveBatch(model, dc["n_envs"], seed=seed), obs_mode="features")
    agent = _sb3(algo, venv, cfg, seed)
    feats = model.features()

    def on_cp(agent: Any, t: int, games: int) -> dict:
        scores = action_scores(agent, feats)
        pol = ev.greedy(scores, model.allowed)
        rec = {"steps": t, "episodes": games, "regret": ev.drive_regret(model, pol)}
        if algo == "ppo":
            rec["regret_stochastic"] = ev.drive_regret(model, scores)
        return rec

    cb = CheckpointCallback(ev.log_checkpoints(n_steps, cfg["drive"]["n_checkpoints"]), on_cp)
    t0 = time.time()
    agent.learn(n_steps, callback=cb)
    path = runner.resolve(cfg["models_dir"]) / f"{algo}_drive_{seed}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(path)
    pol = ev.greedy(action_scores(agent, feats), model.allowed)
    return {"algo": algo, "env": "drive", "seed": seed, "n_steps": n_steps, "curve": cb.records,
            "final_policy": pol.tolist(), "agreement": ev.drive_agreement(model, pol),
            "hyperparameters": cfg["deep"][algo], "seconds": time.time() - t0}


def run_sb3_endgame(
    cfg: dict, algo: str, seed: int, n_steps: int, shaping: str = "none",
    opponent: str = "optimal", tag: str = "",
) -> dict:
    _torch(cfg)
    from fourth_and_rl.rl.sb3_utils import BatchVecEnv, CheckpointCallback

    snap = runner.snapshot(cfg)
    sim = snap.sim
    starts, loss = endgame_eval_set(cfg)
    ec = cfg["deep"]["endgame"]
    batch = EndgameBatch(sim, ec["n_envs"], seed=seed, opponent=opponent, start=_spec(cfg),
                         shaping=shaping)
    venv = BatchVecEnv(batch)
    agent = _sb3(algo, venv, cfg, seed)

    def on_cp(agent: Any, t: int, games: int) -> dict:
        if opponent == "optimal":
            o = ev.evaluate_endgame(sim, sb3_policy(agent), starts,
                                    seed=cfg["endgame"]["eval_seed"], loss=loss)
            return {"steps": t, "episodes": games, "regret": o["regret"],
                    "regret_se": o["regret_se"], "win_rate": o["win_rate"],
                    "zero_loss_fourth_downs": o["zero_loss_fourth_downs"],
                    "same_action_as_optimal": o["same_action_as_optimal"]}
        o = ev.evaluate_endgame(sim, sb3_policy(agent), starts, seed=cfg["endgame"]["eval_seed"],
                                opponent="coach")
        return {"steps": t, "episodes": games, "win_rate": o["win_rate"],
                "win_rate_se": o["win_rate_se"]}

    cb = CheckpointCallback(ev.log_checkpoints(n_steps, cfg["endgame"]["n_checkpoints"], 10_000),
                            on_cp)
    t0 = time.time()
    agent.learn(n_steps, callback=cb)
    name = f"{algo}_endgame{tag}_{seed}"
    path = runner.resolve(cfg["models_dir"]) / f"{name}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(path)
    out = {"algo": algo, "env": "endgame", "seed": seed, "n_steps": n_steps, "shaping": shaping,
           "opponent": opponent, "curve": cb.records, "hyperparameters": cfg["deep"][algo],
           "opponent_plays": batch.opponent_plays, "seconds": time.time() - t0}
    if opponent == "optimal":
        out["final_eval"] = ev.evaluate_endgame(sim, sb3_policy(agent), starts,
                                                seed=cfg["endgame"]["eval_seed"], loss=loss)
        out["fourth_down_chart"] = endgame_chart(cfg, sb3_policy(agent))
    return out


def endgame_chart(cfg: dict, policy: ev.Policy | None) -> dict:
    """Fourth-down decisions in a fixed slice of the end game: up 4, 4:00 left (as question
    1), opponent with three timeouts, own three; every distance and yard line. Codes are
    0 go (any play mode), 1 field goal, 2 punt."""
    snap = runner.snapshot(cfg)
    sim, g = snap.sim, snap.sim.grid
    dist = np.arange(g.n_dist)
    yl = np.arange(1, g.n_yl + 1)
    d, y = np.meshgrid(dist, yl, indexing="ij")
    f = g.fidx(4, d.ravel(), y.ravel())
    n = len(f)
    batch = EndgameBatch(sim, n, seed=0)
    batch.reset(states={"t": np.full(n, 48), "s": np.full(n, 4), "f": f,
                        "a": np.full(n, 3), "b": np.full(n, 3)})
    idx = np.arange(n)
    act = sim.optimal_actions(batch.st, idx) if policy is None else policy(batch, idx)
    code = np.where(act == mg.A_FG, 1, np.where(act == mg.A_PUNT, 2, 0))
    return {"seconds": 240, "score": 4, "timeouts": [3, 3], "codes": code.reshape(d.shape).tolist()}


def run_baselines(cfg: dict) -> dict:
    snap = runner.snapshot(cfg)
    sim = snap.sim
    starts, loss = endgame_eval_set(cfg)
    seed = cfg["endgame"]["eval_seed"]
    t0 = time.time()
    out = {}
    for name, pol in (("optimal", ev.optimal_policy), ("coach", ev.coach_policy),
                      ("never_go", ev.make_never_go(loss)), ("random", ev.random_policy)):
        out[name] = ev.evaluate_endgame(sim, pol, starts, seed=seed, loss=loss)
    # Check of the regret estimator on the real model: the summed losses and the drop in
    # win rate measure the same thing (the latter with far more noise).
    out["estimator_check"] = {
        k: {"regret_from_losses": out[k]["regret"],
            "win_rate_gap": out[k]["optimal_win_rate_of_starts"] - out[k]["win_rate"],
            "win_rate_se": out[k]["win_rate_se"]}
        for k in ("coach", "never_go", "random")
    }
    out["against_coach_opponent"] = {
        name: ev.evaluate_endgame(sim, pol, starts, seed=seed, opponent="coach")
        for name, pol in (("optimal", ev.optimal_policy), ("coach", ev.coach_policy))
    }
    out["optimal_chart"] = endgame_chart(cfg, None)
    out["coach_chart"] = endgame_chart(cfg, ev.coach_policy)
    m = snap.model
    out["dp_cost"] = {
        "method": "backward induction, last 5:00 (60 slices) of mdp_game.solve",
        "slices": int(m.clock.endgame_ticks),
        "state_backups": int(m.n_states()["end"]),
        "seconds_whole_game_solve": snap.meta.get("solve_seconds"),
        "seconds_end_phase_estimate": snap.meta.get("solve_seconds", 0) * m.n_states()["end"]
        / m.n_states()["total"],
    }
    out["seconds"] = time.time() - t0
    return out


def run_approx(cfg: dict) -> dict:
    """Function-approximation floor: train the DQN architecture by supervised regression on
    the exact Q* (the best targets any learner could have), then measure its greedy policy.
    Whatever regret is left is the error of the representation and the fit, not of
    exploration or bootstrapping."""
    torch = _torch(cfg)
    ac = cfg["deep"]["approximation"]
    torch.manual_seed(ac["seed"])
    snap = runner.snapshot(cfg)
    arch = cfg["deep"]["dqn"]["net_arch"]
    t0 = time.time()

    def mlp(d_in: int, d_out: int) -> Any:
        layers, d = [], d_in
        for h in arch:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            d = h
        return torch.nn.Sequential(*layers, torch.nn.Linear(d, d_out))

    def fit(x: np.ndarray, y: np.ndarray, mask: np.ndarray, w: np.ndarray, epochs: int,
            batch: int) -> Any:
        net = mlp(x.shape[1], y.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        xt, yt = torch.as_tensor(x), torch.as_tensor(np.where(mask, y, 0.0), dtype=torch.float32)
        mt, wt = torch.as_tensor(mask, dtype=torch.float32), torch.as_tensor(w, dtype=torch.float32)
        n = len(x)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch):
                j = perm[i : i + batch]
                err = ((net(xt[j]) - yt[j]) ** 2 * mt[j]).sum(dim=1)
                loss = (err * wt[j]).sum() / wt[j].sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        return net

    # Drive: every state, weighted by how often the optimal policy visits it (plus a floor).
    model = snap.drive
    occ = ev.occupancy(model, model.policy_star)
    w = occ / occ.sum() + 1e-4
    net = fit(model.features(), np.where(model.allowed, model.q_star, 0.0), model.allowed, w,
              ac["drive_epochs"], len(w))
    with torch.no_grad():
        q_hat = net(torch.as_tensor(model.features())).numpy()
    pol = ev.greedy(q_hat, model.allowed)
    drive_out = {"regret": ev.drive_regret(model, pol), "agreement": ev.drive_agreement(model, pol),
                 "rmse_points_on_optimal_actions": float(np.sqrt(np.average(
                     (q_hat[np.arange(model.n), model.policy_star] - model.v_star) ** 2,
                     weights=w)))}

    # End game: states visited by the optimal policy from the start distribution.
    sim = snap.sim
    starts, loss = endgame_eval_set(cfg)
    m = snap.model
    n_want = ac["endgame_states"]
    batch = EndgameBatch(sim, 4000, seed=ac["seed"], start=_spec(cfg))
    feats, targets, masks = [], [], []
    have = 0
    batch.reset()
    while have < n_want:
        idx = np.nonzero(~batch.finished)[0]
        if len(idx) == 0:
            batch.reset()
            continue
        st = batch.st
        rows = np.asarray(loss[st["t"][idx] - 1, st["f"][idx], st["s"][idx] + m.score_max,
                               st["a"][idx], st["b"][idx]]).astype(np.float32)
        v = sim.values(st, idx).astype(np.float32)
        ok = batch.mask(idx) & np.isfinite(rows)
        feats.append(batch.observe(idx))
        targets.append(np.where(ok, v[:, None] - rows, 0.0))
        masks.append(ok)
        have += len(idx)
        act = np.zeros(batch.n, dtype=np.int64)
        act[idx] = sim.optimal_actions(st, idx)
        batch.step(act)
    x, y, mk = np.concatenate(feats), np.concatenate(targets), np.concatenate(masks)
    net_e = fit(x, y, mk, np.ones(len(x)), ac["endgame_epochs"], 4096)

    def pol_e(b: EndgameBatch, idx: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            q = net_e(torch.as_tensor(b.observe(idx))).numpy()
        return np.where(b.mask(idx), q, -np.inf).argmax(axis=1)

    end_out = ev.evaluate_endgame(sim, pol_e, starts, seed=cfg["endgame"]["eval_seed"], loss=loss)
    end_out["n_training_states"] = int(len(x))
    end_out["fourth_down_chart"] = endgame_chart(cfg, pol_e)
    return {"drive": drive_out, "endgame": end_out, "architecture": arch,
            "seconds": time.time() - t0}


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def drive_diagnostics(cfg: dict) -> dict:
    """Why the deep agents stall on the drive: reload the saved networks and compare them
    with the exact answer. For DQN: error of its Q-values against Q* and against the action
    gaps it must resolve. For the REINFORCE network: how deterministic its policy became."""
    torch = _torch(cfg)
    from fourth_and_rl.rl.reinforce import NetReinforceConfig
    from fourth_and_rl.rl.sb3_utils import action_scores

    snap = runner.snapshot(cfg)
    model = snap.drive
    occ = ev.occupancy(model, model.policy_star)
    w = occ / occ.sum()
    srt = np.sort(np.where(model.allowed, model.q_star, -np.inf), axis=1)
    gap = srt[:, -1] - srt[:, -2]
    feats = model.features()
    mdir = runner.resolve(cfg["models_dir"])
    out: dict[str, Any] = {"weighted_median_action_gap_points": float(
        np.median(np.repeat(gap, np.maximum((w * 1e5).astype(int), 0))))}
    from stable_baselines3 import DQN

    dqn = []
    for seed in cfg["deep"]["seeds"]:
        path = mdir / f"dqn_drive_{seed}.zip"
        if not path.exists():
            continue
        q = action_scores(DQN.load(path, device="cpu"), feats)
        err = np.where(model.allowed, q - model.q_star, 0.0)
        n_ok = model.allowed.sum(axis=1)
        rmse = np.sqrt((err**2).sum(axis=1) / n_ok)
        centered = err - (err.sum(axis=1, keepdims=True) / n_ok[:, None])
        rel = np.sqrt(np.where(model.allowed, centered**2, 0.0).sum(axis=1) / n_ok)
        dqn.append({
            "seed": seed,
            "rmse_q_points_visit_weighted": float(np.sqrt(np.average(rmse**2, weights=w))),
            "mean_error_q_points_visit_weighted": float(np.average(
                (err.sum(axis=1) / n_ok), weights=w)),
            "rmse_of_action_differences_points": float(np.sqrt(np.average(rel**2, weights=w))),
            "share_of_visits_where_gap_below_that_error": float(
                w[gap < np.sqrt(np.average(rel**2, weights=w))].sum()),
        })
    out["dqn"] = dqn
    rnet = []
    for seed in cfg["deep"]["seeds"]:
        path = mdir / f"reinforce_net_drive_{seed}.pt"
        if not path.exists():
            continue
        rc = cfg["deep"]["reinforce_net"]
        h = NetReinforceConfig(hidden=rc["hidden"]).hidden
        net = torch.nn.Sequential(
            torch.nn.Linear(feats.shape[1], h), torch.nn.Tanh(), torch.nn.Linear(h, h),
            torch.nn.Tanh(), torch.nn.Linear(h, 3))
        net.load_state_dict(torch.load(path)["policy"])
        with torch.no_grad():
            logits = net(torch.as_tensor(feats)).masked_fill(
                ~torch.as_tensor(model.allowed), -1e9)
            p = torch.softmax(logits, dim=1).numpy().astype(float)
        ent = -(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0)).sum(axis=1)
        wrong = ~(model.q_star[np.arange(model.n), p.argmax(axis=1)] >= model.v_star - 1e-9)
        rnet.append({
            "seed": seed,
            "mean_entropy_nats_visit_weighted": float(np.average(ent, weights=w)),
            "share_of_visits_with_max_prob_above_0.999": float(w[p.max(axis=1) > 0.999].sum()),
            "share_of_visits_wrong_and_max_prob_above_0.999": float(
                w[wrong & (p.max(axis=1) > 0.999)].sum()),
            "share_of_visits_wrong": float(w[wrong].sum()),
        })
    out["reinforce_net"] = rnet
    return out


def summarize(cfg: dict, results: dict) -> dict:
    out: dict[str, Any] = {"drive": {}, "endgame": {}, "sparsity": {}, "coach_opponent": {}}
    groups: dict[tuple[str, str], list[dict]] = {}
    for (_, key), r in results.items():
        if key.startswith(("drive_", "endgame_", "sparsity_", "coach_")):
            fam = key.split("__")[0]
            groups.setdefault((fam, r["algo"] + (f"_{r.get('shaping')}" if fam == "sparsity"
                                                  else "")), []).append(r)
    for (fam, name), runs in sorted(groups.items()):
        runs = sorted(runs, key=lambda r: r["seed"])
        steps = [c["steps"] for c in runs[0]["curve"]]
        n = min(len(r["curve"]) for r in runs)
        steps = steps[:n]
        key = "win_rate" if fam == "coach" else "regret"
        thr = cfg["drive"]["thresholds"] if fam == "drive" else cfg["endgame"]["thresholds"]
        curves = np.array([[c[key] for c in r["curve"][:n]] for r in runs])
        s = ev.summarize_curves(steps, curves, thr)
        s["episodes_mean"] = np.mean([[c["episodes"] for c in r["curve"][:n]] for r in runs],
                                     axis=0).tolist()
        s["seeds"] = [r["seed"] for r in runs]
        s["seconds_per_run"] = float(np.mean([r["seconds"] for r in runs]))
        if "regret_greedy" in runs[0]["curve"][0]:
            g = np.array([[c["regret_greedy"] for c in r["curve"][:n]] for r in runs])
            s["greedy"] = ev.summarize_curves(steps, g, thr)
        if "regret_stochastic" in runs[0]["curve"][0]:
            g = np.array([[c["regret_stochastic"] for c in r["curve"][:n]] for r in runs])
            s["stochastic"] = ev.summarize_curves(steps, g, thr)
        if "regret_se" in runs[0]["curve"][0]:
            s["eval_se_final_per_seed"] = [r["curve"][n - 1]["regret_se"] for r in runs]
        if "agreement" in runs[0]:
            s["agreement_per_seed"] = [r["agreement"] for r in runs]
        if "final_eval" in runs[0]:
            s["final_eval_per_seed"] = [r["final_eval"] for r in runs]
            s["fourth_down_chart_seed0"] = runs[0]["fourth_down_chart"]
        if "hyperparameters" in runs[0]:
            s["hyperparameters"] = runs[0]["hyperparameters"]
        if "config" in runs[0]:
            s["config"] = runs[0]["config"]
        target = {"drive": "drive", "endgame": "endgame", "sparsity": "sparsity",
                  "coach": "coach_opponent"}[fam]
        out[target][name] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=["drive", "endgame", "sparsity", "coach", "approx"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny budgets, separate runs directory, results not written")
    args = ap.parse_args()
    cfg = runner.load_config()
    dc = cfg["deep"]
    if args.smoke:
        cfg["runs_dir"] = cfg["runs_dir"] + "_smoke"
        cfg["models_dir"] = cfg["models_dir"] + "_smoke"
        cfg["endgame"]["eval_games"] = 300
        dc["reinforce_net"]["n_steps"] = 20000
        for part in ("drive", "endgame"):
            for k in ("dqn_steps", "ppo_steps"):
                dc[part][k] = 20000
        dc["dqn"]["learning_starts"] = 1000
        dc["ppo"]["n_steps"] = 64
        dc["sparsity"]["steps"] = dc["coach_opponent"]["steps"] = 15000
        dc["approximation"].update(drive_epochs=50, endgame_states=20000, endgame_epochs=2)
    t0 = time.time()
    # Build the loss table once in this process before the workers need it.
    endgame_eval_set(cfg)
    jobs: list = []
    want = (lambda part: args.only in (None, part))  # noqa: E731
    if want("endgame"):
        jobs.append((GROUP, "baselines", run_baselines, {}))
        for seed in dc["seeds"]:
            for algo in ("ppo", "dqn"):
                jobs.append((GROUP, f"endgame__{algo}_{seed}", run_sb3_endgame,
                             dict(algo=algo, seed=seed, n_steps=dc["endgame"][f"{algo}_steps"])))
    if want("coach"):
        for seed in dc["coach_opponent"]["seeds"]:
            jobs.append((GROUP, f"coach__dqn_{seed}", run_sb3_endgame,
                         dict(algo="dqn", seed=seed, n_steps=dc["coach_opponent"]["steps"],
                              opponent="coach", tag="_vs_coach")))
    if want("sparsity"):
        for shaping in dc["sparsity"]["shaping"]:
            for seed in dc["sparsity"]["seeds"]:
                jobs.append((GROUP, f"sparsity__dqn_{shaping}_{seed}", run_sb3_endgame,
                             dict(algo="dqn", seed=seed, n_steps=dc["sparsity"]["steps"],
                                  shaping=shaping, tag=f"_shaping_{shaping}")))
    if want("drive"):
        for seed in dc["seeds"]:
            jobs.append((GROUP, f"drive__reinforce_net_{seed}", run_reinforce_net,
                         dict(seed=seed)))
            for algo in ("dqn", "ppo"):
                jobs.append((GROUP, f"drive__{algo}_{seed}", run_sb3_drive,
                             dict(algo=algo, seed=seed)))
    if want("approx"):
        jobs.append((GROUP, "approximation", run_approx, {}))
    results = runner.run_all(jobs, cfg, cfg["workers"], force=args.force,
                             threads=dc["torch_threads"])
    for p in sorted(runner.resolve(cfg["runs_dir"]).joinpath(GROUP).glob("*.json")):
        old = runner.load_run(cfg, GROUP, p.stem)
        if old is not None:
            results.setdefault((GROUP, p.stem), old)

    snap = runner.snapshot(cfg)
    summary = summarize(cfg, results)
    out = {
        "what": "deep agents: hand-written REINFORCE network and Stable-Baselines3 DQN and PPO, "
        "on the drive (exact regret, expected points) and the last five minutes (regret in win "
        "probability from summed exact per-decision losses on a fixed set of start states)",
        "endgame_setup": {
            "start_distribution": cfg["endgame"]["start"],
            "eval_games": cfg["endgame"]["eval_games"],
            "eval_seed": cfg["endgame"]["eval_seed"],
            "opponent": "solved optimal policy (variant: coach-like policy of the simulator)",
            "reward": "1 win, 0.5 tie after regulation, 0 loss, at the end only",
            "step": "one play call of the agent's team; the opponent's plays happen inside step",
            "action_mapping": "actions the model does not allow are played as the neutral play; "
            "evaluation takes the greedy action among allowed actions",
        },
        "thresholds": {"drive_points": cfg["drive"]["thresholds"],
                       "endgame_win_probability": cfg["endgame"]["thresholds"]},
        **summary,
        "baselines": results.get((GROUP, "baselines")),
        "approximation_floor": results.get((GROUP, "approximation")),
        "drive_diagnostics": drive_diagnostics(cfg),
        "config": cfg["deep"],
        "snapshot": snap.describe(),
        "script_seconds_this_invocation": time.time() - t0,
        "compute_seconds_sum_over_runs": float(
            sum(r.get("seconds", 0.0) for r in results.values())
        ),
        "note_on_time": "runs execute in `workers` parallel processes on a shared machine; the "
        "wall time of a full run is about the sum over runs divided by the number of workers",
    }
    if args.smoke:
        print("smoke run: results not written")
        return
    write_json(out, RESULTS / "m5_rl_deep.json")
    figures.make_all()
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
