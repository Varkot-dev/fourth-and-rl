"""M5 part 1: hand-written tabular agents on the drive problem, measured exactly.

Run:  .venv/bin/python scripts/rl_tabular.py            (about 10 minutes on 2 processes)
      .venv/bin/python scripts/rl_tabular.py --force    (rerun every run, ignore saved ones)
      .venv/bin/python scripts/rl_tabular.py --only main   (main | ablations | variance)

Every run is saved under data/rl/runs/tabular/ as soon as it finishes, so an interrupted
script resumes where it stopped. The environment is built from the frozen snapshot named
in configs/rl.yaml (`snapshot_dir`).

Writes
  results/m5_rl_tabular.json     curves, final regret over seeds, sample efficiency,
                                 agreement with the DP policy, ablations, baselines, DP cost,
                                 hyperparameters, seeds, snapshot meta
  assets/m5_learning_curves.png  (with the deep results too, once scripts/rl_deep.py has run)
  assets/m5_policy_comparison.png, assets/m5_ablations.png
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from fourth_and_rl.io import write_json  # noqa: E402
from fourth_and_rl.mdp_drive import solve_drive  # noqa: E402
from fourth_and_rl.paths import RESULTS  # noqa: E402
from fourth_and_rl.rl import evaluate as ev  # noqa: E402
from fourth_and_rl.rl import figures, runner  # noqa: E402
from fourth_and_rl.rl.envs import GO, CompiledDrive  # noqa: E402
from fourth_and_rl.rl.reinforce import (  # noqa: E402
    ReinforceConfig,
    gradient_variance,
    reinforce_tabular,
)
from fourth_and_rl.rl.tabular import Schedule, td_control  # noqa: E402

GROUP = "tabular"


# --------------------------------------------------------------------------------------
# One run (executed in a worker process)
# --------------------------------------------------------------------------------------


def _schedule(cfg: dict, overrides: dict | None) -> tuple[Schedule, float]:
    base = dict(cfg["tabular"]["schedule"])
    over = dict(overrides or {})
    q_init = float(over.pop("q_init", 0.0))
    base.update(over)
    return Schedule(**base), q_init


def run_td(cfg: dict, algo: str, seed: int, n_steps: int, overrides: dict | None = None) -> dict:
    snap = runner.snapshot(cfg)
    model = snap.drive
    env = CompiledDrive(model)
    sch, q_init = _schedule(cfg, overrides)
    marks = ev.log_checkpoints(n_steps, cfg["drive"]["n_checkpoints"])
    curve: list[dict[str, Any]] = []

    def on_cp(q: np.ndarray, t: int, episodes: int) -> None:
        pol = ev.greedy(q, model.allowed)
        curve.append(
            {"steps": t, "episodes": episodes, "regret": ev.drive_regret(model, pol),
             "policy": pol.astype(np.int8).tolist()}
        )

    t0 = time.time()
    out = td_control(env, n_steps, seed, algo=algo, schedule=sch, q_init=q_init,
                     checkpoints=marks, on_checkpoint=on_cp)
    return {
        "algo": algo,
        "seed": seed,
        "n_steps": n_steps,
        "schedule": sch.__dict__,
        "q_init": q_init,
        "curve": curve,
        "final_q": np.where(np.isfinite(out["q"]), out["q"], np.nan).tolist(),
        "visits": out["counts"].sum(axis=1).tolist(),
        "seconds": time.time() - t0,
    }


def run_reinforce(
    cfg: dict, seed: int, n_steps: int, baseline: bool = True, overrides: dict | None = None
) -> dict:
    snap = runner.snapshot(cfg)
    model = snap.drive
    env = CompiledDrive(model)
    rc = dict(cfg["tabular"]["reinforce"])
    rc.update(overrides or {})
    conf = ReinforceConfig(lr=rc["lr"], baseline_lr=rc["baseline_lr"], baseline=baseline)
    marks = ev.log_checkpoints(n_steps, cfg["drive"]["n_checkpoints"])
    curve: list[dict[str, Any]] = []

    def on_cp(probs: np.ndarray, t: int, episodes: int) -> None:
        pol = probs.argmax(axis=1)
        curve.append(
            {"steps": t, "episodes": episodes, "regret": ev.drive_regret(model, probs),
             "regret_greedy": ev.drive_regret(model, pol), "policy": pol.astype(np.int8).tolist()}
        )

    t0 = time.time()
    out = reinforce_tabular(env, n_steps, seed, conf, checkpoints=marks, on_checkpoint=on_cp)
    return {
        "algo": "reinforce" if baseline else "reinforce_no_baseline",
        "seed": seed,
        "n_steps": n_steps,
        "config": conf.__dict__,
        "curve": curve,
        "final_probs": out["probs"].tolist(),
        "seconds": time.time() - t0,
    }


def run_variance(cfg: dict, n_episodes: int, seed: int) -> dict:
    """Gradient variance with and without the baseline, at the uniform policy and at a
    policy half way between uniform and optimal (both with the exact value as baseline)."""
    snap = runner.snapshot(cfg)
    model = snap.drive
    env = CompiledDrive(model)
    t0 = time.time()
    uniform = model.allowed / model.allowed.sum(axis=1, keepdims=True)
    opt = ev.as_probs(model.policy_star, model.allowed)
    out = {}
    for name, pol in (("uniform", uniform), ("half_way", 0.5 * uniform + 0.5 * opt)):
        v = ev.policy_value(model, pol)
        out[name] = gradient_variance(model, env, pol, v, n_episodes, seed)
    out["seconds"] = time.time() - t0
    return out


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def curves_of(runs: list[dict], key: str = "regret") -> tuple[list[int], np.ndarray, np.ndarray]:
    steps = [c["steps"] for c in runs[0]["curve"]]
    regret = np.array([[c[key] for c in r["curve"]] for r in runs])
    episodes = np.array([[c["episodes"] for c in r["curve"]] for r in runs])
    return steps, regret, episodes


def learned_last(model: Any, runs: list[dict]) -> dict[str, Any]:
    """For every state: from which checkpoint on its greedy action stays optimal (per
    seed), against how often the state is visited. Rarely visited states are learned last."""
    steps = np.array([c["steps"] for c in runs[0]["curve"]])
    opt_ok = model.q_star >= model.v_star[:, None] - 1e-9  # (n, 3) optimal actions
    occ = ev.occupancy(model, model.policy_star)
    learned = np.full((len(runs), model.n), np.inf)
    for i, r in enumerate(runs):
        pols = np.array([c["policy"] for c in r["curve"]])  # (n_cp, n)
        good = opt_ok[np.arange(model.n)[None, :], pols]
        for s in range(model.n):
            bad = np.nonzero(~good[:, s])[0]
            if len(bad) == 0:
                learned[i, s] = steps[0]
            elif bad[-1] < len(steps) - 1:
                learned[i, s] = steps[bad[-1] + 1]
    reach = occ > 1e-9
    med = np.median(learned, axis=0)
    q = np.where(model.allowed, model.q_star, -np.inf)
    srt = np.sort(q, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    bins = np.quantile(occ[reach], [0.0, 0.25, 0.5, 0.75, 1.0])
    out_bins = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        sel = reach & (occ >= lo) & (occ <= hi)
        fin = np.isfinite(med[sel])
        out_bins.append(
            {
                "visits_per_episode_range": [float(lo), float(hi)],
                "n_states": int(sel.sum()),
                "share_learned_by_end": float(fin.mean()),
                "median_steps_to_learn_if_learned": float(np.median(med[sel][fin]))
                if fin.any() else None,
                "median_margin_points": float(np.median(margin[sel])),
            }
        )
    from scipy.stats import spearmanr

    fin = reach & np.isfinite(med)
    return {
        "by_visit_quartile": out_bins,
        "spearman_steps_to_learn_vs_visits": float(spearmanr(med[fin], occ[fin]).statistic),
        "spearman_steps_to_learn_vs_margin": float(spearmanr(med[fin], margin[fin]).statistic),
        "states_never_learned_by_majority": int((~np.isfinite(med) & reach).sum()),
        "n_reachable": int(reach.sum()),
    }


def majority_policy(runs: list[dict]) -> np.ndarray:
    pols = np.array([r["curve"][-1]["policy"] for r in runs])
    return np.array([np.bincount(col, minlength=3).argmax() for col in pols.T])


def return_sd(model: Any) -> float:
    """Standard deviation of the return of one episode under the optimal policy, exactly:
    rewards come only at the end, so the second moment M solves M = P M + (terminal
    probabilities) @ reward^2, like the value."""
    p, _ = ev.policy_chain(model, model.policy_star)
    probs = ev.as_probs(model.policy_star, model.allowed)
    mixed = np.einsum("sa,asc->sc", probs, model.dist)
    m2 = np.linalg.solve(np.eye(model.n) - p, mixed[:, model.n :] @ model.term_reward**2)
    mean = float(model.start @ model.v_star)
    return float(np.sqrt(model.start @ m2 - mean**2))


def worked_example(model: Any) -> dict[str, Any]:
    """Numbers used in the hand-computed examples of docs/notes/m5_rl.md: fourth and 2 at
    the own 40 (yardline_100 = 60)."""
    g = model.grid
    pos = np.full(g.nf, -1)
    pos[model.states] = np.arange(model.n)
    i = int(pos[g.fidx(4, g.dist_bin(2), 60)])
    f10 = g.first10()
    go = model.dist[GO, i]
    n = model.n
    return {
        "state": "fourth and 2, yardline_100 = 60",
        "q_star_go_fg_punt": [float(x) if np.isfinite(x) else None for x in model.q_star[i]],
        "go_p_other_team_takes_over_at_their_40": float(go[n + 40 - 1]),
        "value_of_that_for_us": float(model.term_reward[40 - 1]),
        "go_p_drive_ends_in_touchdown": float(go[n + g.n_yl]),
        "touchdown_reward_td_value_minus_kickoff": float(model.term_reward[g.n_yl]),
        "td_value": model.td_value,
        "kickoff_value": model.kickoff_value,
        "first_and_10_value_at_yardline_100": {
            str(y): float(model.v_field[f10[y]]) for y in (40, 45, 80, 85)
        },
        "fourth_and_1_at_yardline_45_q_star": [
            float(x) if np.isfinite(x) else None
            for x in model.q_star[int(pos[g.fidx(4, g.dist_bin(1), 45)])]
        ],
    }


def dp_cost(snap: Any) -> dict[str, Any]:
    """What dynamic programming spends on the same problem, in comparable units."""
    m = snap.model
    k = m.kern
    t0 = time.time()
    out = solve_drive(k, snap.td_value, m.fg_points, m.safety_points)
    seconds = time.time() - t0
    nnz = int(k.k_all["neutral"].nnz + k.k_punt.nnz + 2 * k.grid.nf)
    return {
        "method": "value iteration on the stage 1 model (mdp_drive.solve_drive)",
        "sweeps": int(out["iterations"]),
        "states": int(k.grid.nf),
        "state_backups": int(out["iterations"] * k.grid.nf),
        "transition_probabilities_touched": int(out["iterations"] * nnz),
        "seconds": seconds,
        "note": "DP reads the transition probabilities; the learners only see sampled outcomes, "
        "one decision per step",
    }


def summarize(cfg: dict, results: dict, snap: Any) -> dict:
    model = snap.drive
    thr = cfg["drive"]["thresholds"]
    tc = cfg["tabular"]
    out: dict[str, Any] = {}
    groups: dict[str, list[dict]] = {}
    for (_, key), r in results.items():
        if key.startswith("main_"):
            groups.setdefault(r["algo"] if r.get("q_init", 0) == 0 else "q_learning_optimistic",
                              []).append(r)
    algos = {}
    for name, runs in sorted(groups.items()):
        runs = sorted(runs, key=lambda r: r["seed"])
        steps, regret, episodes = curves_of(runs)
        s = ev.summarize_curves(steps, regret, thr)
        s["episodes_mean"] = episodes.mean(axis=0).tolist()
        s["seeds"] = [r["seed"] for r in runs]
        s["seconds_per_run"] = float(np.mean([r["seconds"] for r in runs]))
        maj = majority_policy(runs)
        s["agreement_majority_policy"] = ev.drive_agreement(model, maj)
        s["agreement_per_seed"] = [
            ev.drive_agreement(model, np.array(r["curve"][-1]["policy"]))[
                "weighted_by_optimal_visits"] for r in runs
        ]
        s["majority_policy_regret"] = ev.drive_regret(model, maj)
        if name.startswith("reinforce"):
            _, greedy_regret, _ = curves_of(runs, "regret_greedy")
            s["greedy"] = ev.summarize_curves(steps, greedy_regret, thr)
            s["note"] = ("regret is of the stochastic softmax policy itself; "
                         "'greedy' is of its argmax")
            s["config"] = runs[0]["config"]
        else:
            s["schedule"] = runs[0]["schedule"]
            s["q_init"] = runs[0]["q_init"]
            s["learned_last"] = learned_last(model, runs)
        algos[name] = s
    out["algorithms"] = algos

    abl: dict[str, Any] = {}
    for (_, key), r in results.items():
        if key.startswith("abl_"):
            _, family, variant, _seed = key.split("__")
            abl.setdefault(family, {}).setdefault(variant, []).append(r)
    out["ablations"] = {}
    for family, variants in abl.items():
        out["ablations"][family] = {}
        for variant, runs in variants.items():
            runs = sorted(runs, key=lambda r: r["seed"])
            steps, regret, _ = curves_of(runs)
            s = ev.summarize_curves(steps, regret, thr)
            s["seeds"] = [r["seed"] for r in runs]
            s["settings"] = tc["ablations"][family][variant]
            out["ablations"][family][variant] = s
    var = [r for (g, key), r in results.items() if key == "gradient_variance"]
    if var:
        out["gradient_variance"] = var[0]
    return out


def chart_payload(cfg: dict, snap: Any, summary: dict, results: dict) -> dict:
    """Fourth-down charts (optimal, majority learned policies) for the figure."""
    model = snap.drive
    g = model.grid
    out = {"optimal": ev.chart(model, model.policy_star).tolist()}
    q = np.where(model.allowed, model.q_star, -np.inf)
    srt = np.sort(q, axis=1)
    out["optimal_margin"] = ev.chart(model, srt[:, -1] - srt[:, -2]).tolist()
    out["visits_optimal"] = ev.chart(model, ev.occupancy(model, model.policy_star)).tolist()
    for name in ("q_learning", "reinforce"):
        runs = [r for (gr, k), r in results.items() if k.startswith("main_") and r["algo"] == name
                and r.get("q_init", 0) == 0]
        if runs:
            maj = majority_policy(runs)
            out[name] = ev.chart(model, maj).tolist()
            by_state = ev.regret_by_state(model, maj)
            out[f"{name}_regret_by_state"] = ev.chart(model, by_state).tolist()
    out["dist_reps"] = g.dist_rep(np.arange(g.n_dist)).tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=["main", "ablations", "variance"], default=None)
    args = ap.parse_args()
    cfg = runner.load_config()
    tc = cfg["tabular"]
    t0 = time.time()
    jobs = []
    if args.only in (None, "main"):
        for seed in tc["seeds"]:
            for algo in ("q_learning", "sarsa", "expected_sarsa"):
                jobs.append((GROUP, f"main_{algo}_{seed}", run_td,
                             dict(algo=algo, seed=seed, n_steps=tc["n_steps"])))
            opt = dict(tc["optimistic"]["schedule"], q_init=tc["optimistic"]["q_init"])
            jobs.append((GROUP, f"main_q_learning_optimistic_{seed}", run_td,
                         dict(algo="q_learning", seed=seed, n_steps=tc["n_steps"], overrides=opt)))
            for base in (True, False):
                name = "reinforce" if base else "reinforce_no_baseline"
                jobs.append((GROUP, f"main_{name}_{seed}", run_reinforce,
                             dict(seed=seed, n_steps=tc["n_steps"], baseline=base)))
    if args.only in (None, "ablations"):
        for family, variants in tc["ablations"].items():
            for variant, over in variants.items():
                for seed in tc["ablation_seeds"]:
                    key = f"abl__{family}__{variant}__{seed}"
                    if family == "reinforce_lr":
                        jobs.append((GROUP, key, run_reinforce,
                                     dict(seed=seed, n_steps=tc["ablation_steps"], overrides=over)))
                    else:
                        jobs.append((GROUP, key, run_td,
                                     dict(algo="q_learning", seed=seed,
                                          n_steps=tc["ablation_steps"], overrides=over)))
    if args.only in (None, "variance"):
        jobs.append((GROUP, "gradient_variance", run_variance,
                     dict(n_episodes=tc["gradient_variance_episodes"], seed=tc["seeds"][0])))
    # Longest jobs first so the two workers finish together.
    jobs.sort(key=lambda j: -j[3].get("n_steps", 0))
    results = runner.run_all(jobs, cfg, cfg["workers"], force=args.force)
    # Every saved run, including those of other --only selections.
    for p in sorted(runner.resolve(cfg["runs_dir"]).joinpath(GROUP).glob("*.json")):
        old = runner.load_run(cfg, GROUP, p.stem)
        if old is not None:
            results.setdefault((GROUP, p.stem), old)

    snap = runner.snapshot(cfg)
    model = snap.drive
    summary = summarize(cfg, results, snap)
    coach_table = snap.coach["table"]
    base = ev.drive_baselines(model, coach_table)
    out = {
        "what": "hand-written tabular agents on the single-drive expected-points problem; regret "
        "is exact (policy evaluation by a linear solve), in expected points per drive start",
        "environment": {
            "n_fourth_down_states": int(model.n),
            "n_field_states": int(model.grid.nf),
            "n_reachable_fourth_down_states": ev.drive_agreement(model, model.policy_star)[
                "n_reachable_states"],
            "share_of_drive_starts_reaching_a_fourth_down": model.p_reach,
            "decisions_per_episode_under_optimal_policy": float(
                ev.occupancy(model, model.policy_star).sum()),
            "kickoff_share_of_starts": cfg["drive"]["kickoff_share"],
            "optimal_value_of_start_distribution": float(model.start @ model.v_star),
            "optimal_go_states": int((model.policy_star == GO).sum()),
            "return_sd_points_under_optimal_policy": return_sd(model),
            "step": "one fourth-down decision (downs 1 to 3 are played inside the environment)",
        },
        "baselines_regret": base,
        "worked_example": worked_example(model),
        "dp_cost": dp_cost(snap),
        "thresholds_points": cfg["drive"]["thresholds"],
        **summary,
        "charts": chart_payload(cfg, snap, summary, results),
        "config": cfg["tabular"],
        "snapshot": snap.describe(),
        "script_seconds_this_invocation": time.time() - t0,
        "compute_seconds_sum_over_runs": float(
            sum(r.get("seconds", 0.0) for r in results.values())
        ),
        "note_on_time": "runs execute in `workers` parallel processes on a shared machine; the "
        "wall time of a full run is about the sum over runs divided by the number of workers",
    }
    write_json(out, RESULTS / "m5_rl_tabular.json")
    figures.make_all()
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
