"""M6 Part A: offline RL in a world where the truth is known.

Run:  .venv/bin/python scripts/offline_known_truth.py           (about 25 minutes, 2 processes)
      .venv/bin/python scripts/offline_known_truth.py --force   (ignore saved runs)
      .venv/bin/python scripts/offline_known_truth.py --figures (redraw from the results)

The truth is the single-drive MDP of M4 stage 1 as M5 compiled it (`rl.envs.DriveModel`),
built from the current solved model in data/mdp. Logged datasets are drawn from a family
of behavior policies (coach-like, more conservative, mixed with random actions) at several
sizes and seeds; every learner sees the same logged data, and every learned policy is
scored by exact policy evaluation in the true model.

The hypotheses in configs/offline.yaml are written into the results file before any run.
Every run is saved under data/models/offline/runs/ as soon as it finishes, so an
interrupted script resumes where it stopped.

Writes
  results/m6_offline_known_truth.json
  assets/m6_distribution_shift.png  (headline), assets/m6_regret_vs_data.png,
  assets/m6_optimizers_curse.png
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from fourth_and_rl.io import read_json, write_json  # noqa: E402
from fourth_and_rl.offline import datasets as ds  # noqa: E402
from fourth_and_rl.offline import figures  # noqa: E402
from fourth_and_rl.offline import learners as L  # noqa: E402
from fourth_and_rl.offline import measure as M  # noqa: E402
from fourth_and_rl.paths import CONFIGS, RESULTS  # noqa: E402
from fourth_and_rl.rl import evaluate as ev  # noqa: E402
from fourth_and_rl.rl import runner  # noqa: E402

OUT = RESULTS / "m6_offline_known_truth.json"
GROUP = "known_truth"
T90_4DF = 2.132  # t quantile for a 90% interval of a mean over 5 seeds


def load_cfg(path: str | None = None) -> dict[str, Any]:
    return runner.load_config(runner.resolve(path) if path else CONFIGS / "offline.yaml")


def data_seed(cfg: dict[str, Any], behavior: str, seed: int) -> int:
    """One stream per (behavior, seed): the behaviors' datasets are independent."""
    names = list(cfg["known_truth"]["behaviors"])
    return int(np.random.SeedSequence([seed, names.index(behavior)]).generate_state(1)[0])


def behavior_probs(cfg: dict[str, Any], model: Any, coach: np.ndarray, name: str) -> np.ndarray:
    b = cfg["known_truth"]["behaviors"][name]
    return ds.behavior_policy(model, coach, b.get("epsilon", 0.0), b.get("go_logit_shift", 0.0))


# --------------------------------------------------------------------------------------
# One dataset (a behavior and a seed, every size): run in a worker process
# --------------------------------------------------------------------------------------


def _fqe_value(model: Any, c: np.ndarray, policy: np.ndarray, start_hat: np.ndarray) -> float:
    q = L.fqe_tabular(model, c, policy)
    v = q[np.arange(model.n), policy]
    return float(start_hat @ np.where(np.isfinite(v), v, 0.0))


def run_dataset(cfg: dict[str, Any], behavior: str, seed: int, role: str) -> dict[str, Any]:
    t0 = time.time()
    snap = runner.snapshot(cfg)
    model = snap.drive
    kt, lc = cfg["known_truth"], cfg["learners"]
    mu = behavior_probs(cfg, model, snap.coach["table"], behavior)
    sizes = list(kt["sizes"])
    data = ds.generate(model, mu, max(sizes), data_seed(cfg, behavior, seed))
    n_cols = model.n + model.n_term
    counts = ds.counts_by_size(data, n_cols, sizes)
    reach = M.reachable(model)
    head = kt["headline"]
    out: dict[str, Any] = {"behavior": behavior, "seed": seed, "role": role, "sizes": {}}
    for size in sizes:
        c = counts[size]
        sub = data.head(size)
        start_hat = ds.start_states(sub)
        res: dict[str, Any] = {}

        def ev_fit(fit: L.Fit, key: str | None = None, curse: bool = True, c=c, res=res
                   ) -> None:
            res[key or fit.name] = M.evaluate_fit(model, fit, c, mu, reach, curse=curse)

        bc = L.behavior_cloning(model, c, lc["bc"]["smoothing"])
        ev_fit(bc)
        ev_fit(L.Fit("bc_mode", bc.info["mode"]), curse=False)
        for alpha in lc["cql_tabular"]["alpha_sweep"]:
            f = L.cql_tabular(model, c, alpha, outer_iter=lc["cql_tabular"]["outer_iter"],
                              inner_steps=lc["cql_tabular"]["inner_steps"])
            ev_fit(f, f"cql_tabular@{alpha}")
            res[f"cql_tabular@{alpha}"]["fqe_value"] = _fqe_value(model, c, f.policy, start_hat)
            if role == "eval" and behavior == head["behavior"] and size == head["size"] \
                    and seed == head["seed"]:
                res[f"cql_tabular@{alpha}"]["heatmap"] = M.heatmap_rows(model, f, c)
                res[f"cql_tabular@{alpha}"]["scatter"] = M.error_vs_count_points(
                    model, f, c, reach)
        for beta in lc["fqi_lcb"]["beta_sweep"]:
            f = L.fqi_tabular(model, c, lcb_beta=beta)
            ev_fit(f, f"fqi_lcb@{beta}")
            res[f"fqi_lcb@{beta}"]["fqe_value"] = _fqe_value(model, c, f.policy, start_hat)
        if role == "eval":
            fits = [
                L.fqi_tabular(model, c, q_init=lc["fqi_tabular"]["q_init"]),
                L.plugin(model, c, **{k: (tuple(v) if isinstance(v, list) else v)
                                      for k, v in lc["plugin"].items()}),
                L.fqi_gbt(model, c, lc["fqi_gbt"]["n_iter"], seed, lc["fqi_gbt"]["params"]),
            ]
            for f in fits:
                ev_fit(f)
                if f.name == "fqi_tabular":
                    res[f.name]["fqe_value"] = _fqe_value(model, c, f.policy, start_hat)
            is_head = behavior == head["behavior"] and size == head["size"] \
                and seed == head["seed"]
            if is_head:
                for f in fits:
                    res[f.name]["heatmap"] = M.heatmap_rows(model, f, c)
                    res[f.name]["scatter"] = M.error_vs_count_points(model, f, c, reach)
                res["counts_heatmap"] = {
                    a: ev.chart(model, L.cell_counts(c)[:, i]).tolist()
                    for i, a in enumerate(("go", "field_goal", "punt"))
                }
            ab = lc["abstain"]
            if behavior in ab["behaviors"] and size in ab["sizes"]:
                plug = fits[1]
                rng = np.random.default_rng(seed + 17)
                n_ep = int(sub.first.sum())
                pols = []
                for _ in range(ab["n_boot"]):
                    ids = rng.integers(0, n_ep, n_ep)
                    cb = ds.counts(sub.take_episodes(ids), n_cols)
                    pk = {k: (tuple(v) if isinstance(v, list) else v)
                          for k, v in lc["plugin"].items()}
                    pols.append(L.plugin(model, cb, **pk).policy)
                pols = np.asarray(pols)
                agree = L.bootstrap_agreement(pols, plug.policy)
                pol = L.abstain(plug.policy, agree, bc.info["mode"], ab["threshold"])
                ev_fit(L.Fit("abstain", pol), curse=False)
                bag = np.array([np.bincount(pols[:, s], minlength=3).argmax()
                                for s in range(model.n)])
                ev_fit(L.Fit("plugin_bagged", bag), curse=False)
                w = L.cell_counts(c).sum(axis=1)
                res["abstain"]["share_deferred_data_states"] = float(
                    (w * (agree < ab["threshold"])).sum() / w.sum())
                res["abstain"]["share_deferred_where_plugin_differs_from_mode"] = float(
                    (w * ((agree < ab["threshold"]) & (plug.policy != bc.info["mode"]))).sum()
                    / max((w * (plug.policy != bc.info["mode"])).sum(), 1e-12))
        out["sizes"][str(size)] = res
    out["seconds"] = time.time() - t0
    return out


def run_d3rlpy(cfg: dict[str, Any], behavior: str, seed: int, size: int, algo: str,
               alpha: float | None) -> dict[str, Any]:
    from fourth_and_rl.offline import deep

    t0 = time.time()
    snap = runner.snapshot(cfg)
    model = snap.drive
    mu = behavior_probs(cfg, model, snap.coach["table"], behavior)
    data = ds.generate(model, mu, max(cfg["known_truth"]["sizes"]),
                       data_seed(cfg, behavior, seed)).head(size)
    c = ds.counts(data, model.n + model.n_term)
    dc = dict(cfg["learners"]["d3rlpy"])
    if alpha is not None:
        dc["alpha"] = alpha
    keep = ("hidden_units", "learning_rate", "batch_size", "target_update_interval",
            "n_steps", "reward_scale", "alpha")
    fit = deep.fit_d3rlpy(model, data, algo, seed, {k: dc[k] for k in keep if k in dc})
    reach = M.reachable(model)
    res = M.evaluate_fit(model, fit, c, mu, reach)
    head = cfg["known_truth"]["headline"]
    if behavior == head["behavior"] and size == head["size"] and seed == head["seed"]:
        res["heatmap"] = M.heatmap_rows(model, fit, c)
        res["scatter"] = M.error_vs_count_points(model, fit, c, reach)
    return {"behavior": behavior, "seed": seed, "size": size, "algo": algo, "alpha": alpha,
            "result": res, "seconds": time.time() - t0}


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def seed_stats(x: list[float]) -> dict[str, Any]:
    a = np.asarray([v for v in x if v is not None], dtype=float)
    if len(a) == 0:
        return {"n": 0}
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    half = T90_4DF * sd / np.sqrt(len(a)) if len(a) == 5 else None
    return {"mean": float(a.mean()), "sd": sd, "min": float(a.min()), "max": float(a.max()),
            "ci90_of_mean": [float(a.mean() - half), float(a.mean() + half)] if half is not None
            else None, "n": int(len(a)), "per_seed": a.tolist()}


def method_keys(cfg: dict[str, Any], alpha: float, beta: float) -> dict[str, str]:
    """Reported method name -> key in a run's results."""
    return {
        "bc": "bc", "bc_mode": "bc_mode", "fqi_tabular": "fqi_tabular", "fqi_gbt": "fqi_gbt",
        "fqi_lcb": f"fqi_lcb@{beta}", "cql_tabular": f"cql_tabular@{alpha}",
        "plugin": "plugin", "abstain": "abstain", "plugin_bagged": "plugin_bagged",
    }


def aggregate(cfg: dict[str, Any], runs: dict, d3: dict, model: Any, coach: np.ndarray
              ) -> dict[str, Any]:
    kt, lc = cfg["known_truth"], cfg["learners"]
    behaviors, sizes = list(kt["behaviors"]), [str(s) for s in kt["sizes"]]
    tune = [r for (g, k), r in runs.items() if r["role"] == "tune"]
    evals = [r for (g, k), r in runs.items() if r["role"] == "eval"]

    def sweep(prefix: str, values: list[float], pool: list[dict]) -> dict[str, float]:
        out = {}
        for v in values:
            regs = [r["sizes"][s][f"{prefix}@{v}"]["regret"] for r in pool for s in sizes]
            out[str(v)] = float(np.mean(regs))
        return out

    cql_tune = sweep("cql_tabular", lc["cql_tabular"]["alpha_sweep"], tune)
    lcb_tune = sweep("fqi_lcb", lc["fqi_lcb"]["beta_sweep"], tune)
    alpha = float(min(cql_tune, key=cql_tune.get))
    beta_fixed = float(lc["fqi_lcb"]["beta"])
    keys = method_keys(cfg, alpha, beta_fixed)

    regret: dict[str, Any] = {}
    curse: dict[str, Any] = {}
    shift: dict[str, Any] = {}
    policy: dict[str, Any] = {}
    for b in behaviors:
        rb = [r for r in evals if r["behavior"] == b]
        regret[b], curse[b], shift[b], policy[b] = {}, {}, {}, {}
        for name, key in keys.items():
            regret[b][name], curse[b][name], policy[b][name] = {}, {}, {}
            for s in sizes:
                vals = [r["sizes"][s][key] for r in rb if key in r["sizes"][s]]
                if not vals:
                    continue
                regret[b][name][s] = seed_stats([v["regret"] for v in vals])
                policy[b][name][s] = {
                    k: seed_stats([v[k] for v in vals]) for k in (
                        "go_rate_own_visits", "share_suboptimal_optimal_visits",
                        "share_differs_from_behavior_mode_data_states",
                        "share_go_where_optimum_kicks_data_states",
                        "share_kick_where_optimum_goes_data_states")
                }
                if vals[0].get("optimizers_curse"):
                    oc: dict[str, Any] = {}
                    for part in ("all", "usual", "rare"):
                        oc[part] = {k: seed_stats([v["optimizers_curse"][part][k] for v in vals])
                                    for k in ("mean_claimed_minus_true",
                                              "share_of_logged_decisions",
                                              "mean_claimed", "mean_true")}
                    g = [v["optimizers_curse"].get("gain_over_usual_action") for v in vals]
                    if all(x is not None for x in g):
                        oc["gain_over_usual_action"] = {
                            k: seed_stats([x[k] for x in g]) for k in (
                                "mean_claimed_gain", "mean_true_gain", "share_truly_worse",
                                "slope_true_on_claimed", "n_states")}
                    curse[b][name][s] = oc
                if vals[0].get("q_error_by_count"):
                    qe: dict[str, Any] = {}
                    for act, buckets in vals[0]["q_error_by_count"].items():
                        qe[act] = {}
                        for bk in buckets:
                            qe[act][bk] = {
                                "n_states": seed_stats([v["q_error_by_count"][act][bk]["n_states"]
                                                        for v in vals]),
                                "mean_error": seed_stats([v["q_error_by_count"][act][bk]["mean"]
                                                          for v in vals]),
                                "share_over_1_point": seed_stats(
                                    [v["q_error_by_count"][act][bk]["share_over_1_point"]
                                     for v in vals]),
                            }
                    shift[b].setdefault(name, {})[s] = qe
        # Data-only choice of the CQL weight: the alpha whose policy fitted Q evaluation
        # rates highest on each dataset.
        regret[b]["cql_tabular_fqe_choice"] = {}
        for s in sizes:
            regs, picks = [], []
            for r in rb:
                cand = {a: r["sizes"][s][f"cql_tabular@{a}"] for a in
                        lc["cql_tabular"]["alpha_sweep"]}
                best = max(cand, key=lambda a, cand=cand: cand[a]["fqe_value"])
                regs.append(cand[best]["regret"])
                picks.append(best)
            regret[b]["cql_tabular_fqe_choice"][s] = {**seed_stats(regs), "alphas": picks}
        regret[b]["cql_tabular_sweep"] = {
            str(a): {s: seed_stats([r["sizes"][s][f"cql_tabular@{a}"]["regret"] for r in rb])
                     for s in sizes} for a in lc["cql_tabular"]["alpha_sweep"]}
        regret[b]["fqi_lcb_sweep"] = {
            str(v): {s: seed_stats([r["sizes"][s][f"fqi_lcb@{v}"]["regret"] for r in rb])
                     for s in sizes} for v in lc["fqi_lcb"]["beta_sweep"]}
        # d3rlpy
        for (algo, a_), rows in _d3_groups(d3, b).items():
            name = "dqn_d3rlpy" if algo == "dqn" else f"cql_d3rlpy@{a_}"
            regret[b][name], curse[b][name] = {}, {}
            for s in sizes:
                vals = [r["result"] for r in rows if str(r["size"]) == s]
                if not vals:
                    continue
                regret[b][name][s] = seed_stats([v["regret"] for v in vals])
                oc = {part: {k: seed_stats([v["optimizers_curse"][part][k] for v in vals])
                             for k in ("mean_claimed_minus_true", "share_of_logged_decisions")}
                      for part in ("all", "usual", "rare")}
                curse[b][name][s] = oc
                qe = {}
                for act, buckets in vals[0]["q_error_by_count"].items():
                    qe[act] = {bk: {"mean_error": seed_stats(
                        [v["q_error_by_count"][act][bk]["mean"] for v in vals])}
                        for bk in buckets}
                shift[b].setdefault(name, {})[s] = qe
                policy[b].setdefault(name, {})[s] = {
                    "go_rate_own_visits": seed_stats([v["go_rate_own_visits"] for v in vals])}

    base = ev.drive_baselines(model, coach)
    behavior_regret = {b: ev.drive_regret(model, behavior_probs(cfg, model, coach, b))
                       for b in behaviors}
    return {
        "tuning": {
            "seeds": kt["tuning_seeds"],
            "rule": "mean true regret over every behavior and size, tuning seeds only",
            "cql_alpha_mean_regret": cql_tune, "cql_alpha_chosen": alpha,
            "lcb_beta_mean_regret": lcb_tune,
            "lcb_beta_reported": beta_fixed,
            "lcb_beta_note": "beta was fixed in the config before any run; the tuning-seed "
                             "sweep is reported but not used to pick it",
        },
        "method_keys": keys,
        "behavior_regret": behavior_regret,
        "reference_regret": base,
        "regret": regret,
        "optimizers_curse": curse,
        "q_error_by_count": shift,
        "policy": policy,
    }


def _d3_groups(d3: dict, behavior: str) -> dict[tuple[str, Any], list[dict]]:
    out: dict[tuple[str, Any], list[dict]] = {}
    for r in d3.values():
        if r["behavior"] == behavior:
            out.setdefault((r["algo"], r["alpha"] if r["algo"] == "cql" else None), []).append(r)
    return out


def headline_arrays(cfg: dict[str, Any], runs: dict, d3: dict, alpha: float,
                    model: Any = None) -> dict[str, Any]:
    head = cfg["known_truth"]["headline"]
    r = runs.get((GROUP, f"eval_{head['behavior']}_{head['seed']}"))
    if r is None:
        return {}
    res = r["sizes"][str(head["size"])]
    out = {"config": head, "counts": res["counts_heatmap"], "methods": {}}
    if model is not None:
        # grey in the charts: states no policy can reach (e.g. 4th and 16+ at the 3)
        out["reachable"] = ev.chart(model, M.reachable(model).astype(float)).tolist()
    for name, key in (("fqi_tabular", "fqi_tabular"), ("fqi_gbt", "fqi_gbt"),
                      ("plugin", "plugin"), ("cql_tabular", f"cql_tabular@{alpha}")):
        out["methods"][name] = {"heatmap": res[key]["heatmap"], "scatter": res[key]["scatter"]}
    for v in d3.values():
        if "heatmap" in v["result"]:
            name = "dqn_d3rlpy" if v["algo"] == "dqn" else f"cql_d3rlpy@{v['alpha']}"
            out["methods"][name] = {"heatmap": v["result"]["heatmap"],
                                    "scatter": v["result"]["scatter"]}
    return out


def check_hypotheses(cfg: dict[str, Any], agg: dict[str, Any]) -> dict[str, Any]:
    """The numbers each hypothesis is judged on (the verdicts are read off these)."""
    real = str(cfg["known_truth"]["real_size"])
    big = str(max(cfg["known_truth"]["sizes"]))
    reg, sh, oc = agg["regret"], agg["q_error_by_count"], agg["optimizers_curse"]

    def m(b: str, name: str, s: str) -> float | None:
        return reg.get(b, {}).get(name, {}).get(s, {}).get("mean")

    def diff(b: str, x: str, y: str, s: str) -> dict[str, Any]:
        a = reg[b][x][s]["per_seed"]
        c = reg[b][y][s]["per_seed"]
        return {f"{x}_minus_{y}": seed_stats(list(np.subtract(a, c)))}

    def qerr(name: str, bucket: str) -> float | None:
        try:
            return sh["coach"][name][real]["go"][bucket]["mean_error"]["mean"]
        except KeyError:
            return None

    out: dict[str, Any] = {}
    out["H1"] = {name: {bk: qerr(name, bk) for bk in ("0", "1-9", "10-99", "100-999", "1000+")}
                 for name in ("fqi_tabular", "fqi_gbt", "plugin", "cql_tabular", "fqi_lcb",
                              "dqn_d3rlpy", "cql_d3rlpy@1.0", "cql_d3rlpy@0.1")}
    out["H2"] = {
        name: {part: oc["coach"][name][real][part]["mean_claimed_minus_true"]["mean"]
               for part in ("usual", "rare")} | {
            "slope": oc["coach"][name][real].get("gain_over_usual_action", {}).get(
                "slope_true_on_claimed", {}).get("mean")}
        for name in ("fqi_tabular", "fqi_gbt", "plugin", "cql_tabular", "fqi_lcb")}
    out["H3"] = diff("coach", "cql_tabular", "fqi_tabular", real)
    out["H4"] = {s: diff("very_conservative", "cql_tabular", "plugin", s)
                 for s in (real, "100000", big)}
    out["H4"]["uniform_regret_at_largest_size"] = {
        n: m("uniform", n, big) for n in ("fqi_tabular", "fqi_gbt", "cql_tabular", "plugin",
                                           "fqi_lcb")}
    out["H5"] = {
        "plugin": m("coach", "plugin", real), "bc_mode": m("coach", "bc_mode", real),
        "bc": m("coach", "bc", real), "abstain": m("coach", "abstain", real),
        "behavior": agg["behavior_regret"]["coach"],
        **diff("coach", "bc_mode", "plugin", real), **diff("coach", "abstain", "plugin", real),
    }
    out["H6"] = {b: {"bc": m(b, "bc", big), "behavior": agg["behavior_regret"][b]}
                 for b in agg["behavior_regret"]}
    return out


def worked_example(cfg: dict[str, Any], model: Any, coach: np.ndarray) -> dict[str, Any]:
    """Numbers for the teaching notes: a few fourth downs, their true values, how often the
    coach-like logger tries each action there, and what one logged go would look like."""
    mu = behavior_probs(cfg, model, coach, "coach")
    occ = ev.occupancy(model, mu)
    per_decision = occ / occ.sum()
    g = model.grid
    pos = np.full(g.nf, -1)
    pos[model.states] = np.arange(model.n)
    vals = model.outcome_values(model.v_star)
    size = cfg["known_truth"]["real_size"]
    out = []
    for ytg, yl in ((15, 80), (10, 75), (2, 60), (1, 40), (4, 35)):
        s = int(pos[g.fidx(g.n_down, g.dist_bin(ytg), yl)])
        p = model.dist[0, s]
        nz = np.nonzero(p)[0]
        best_other = float(np.max(np.where(model.allowed[s, 1:], model.q_star[s, 1:], -np.inf)))
        out.append({
            "yards_to_go": ytg, "yardline_100": yl,
            "q_star": {a: (float(model.q_star[s, i]) if model.allowed[s, i] else None)
                       for i, a in enumerate(("go", "field_goal", "punt"))},
            "best_action": ("go", "field_goal", "punt")[int(model.policy_star[s])],
            "coach_probabilities": mu[s].round(5).tolist(),
            "expected_visits_in_real_size_dataset": float(size * per_decision[s]),
            "expected_go_logs_in_real_size_dataset": float(size * per_decision[s] * mu[s, 0]),
            "prob_one_go_sample_target_beats_best_kick": float(
                p[nz][vals[nz] > best_other].sum()),
            "sd_of_one_go_sample_target": float(np.sqrt(
                (p[nz] * (vals[nz] - model.q_star[s, 0]) ** 2).sum())),
        })
    return {"real_size": size, "states": out}


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--figures", action="store_true", help="only redraw the figures")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--config", default=None, help="another config (smoke tests)")
    ap.add_argument("--out", default=None, help="another results path (smoke tests)")
    args = ap.parse_args()
    if args.figures:
        print(figures.make_known_truth())
        return
    cfg = load_cfg(args.config)
    out_path = runner.resolve(args.out) if args.out else OUT
    kt, lc = cfg["known_truth"], cfg["learners"]
    t0 = time.time()
    snap = runner.snapshot(cfg)
    from fourth_and_rl.components import ComponentModels

    components_stamp = getattr(ComponentModels.load(), "stamp", None)
    preamble = {
        "what": "M6 Part A: offline RL with known truth (single-drive MDP)",
        "hypotheses": cfg["hypotheses_known_truth"],
        "hypotheses_recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "running",
        "snapshot": snap.describe(),
        "development_component_bundle_stamp": components_stamp,
        "config": {k: cfg[k] for k in ("known_truth", "learners", "drive")},
    }
    old = read_json(out_path) if out_path.exists() else None
    if old is not None and old.get("hypotheses") == preamble["hypotheses"]:
        preamble["hypotheses_recorded_at"] = old.get("hypotheses_recorded_at")
    write_json(preamble, out_path)

    jobs = []
    for b in kt["behaviors"]:
        for seed in kt["eval_seeds"]:
            jobs.append((GROUP, f"eval_{b}_{seed}", run_dataset,
                         {"behavior": b, "seed": seed, "role": "eval"}))
        for seed in kt["tuning_seeds"]:
            jobs.append((GROUP, f"tune_{b}_{seed}", run_dataset,
                         {"behavior": b, "seed": seed, "role": "tune"}))
    dc = lc["d3rlpy"]
    for b in dc["behaviors"]:
        for seed in kt["eval_seeds"][: dc["n_seeds"]]:
            for size in dc["sizes"]:
                jobs.append(("d3rlpy", f"dqn_{b}_{seed}_{size}", run_d3rlpy,
                             {"behavior": b, "seed": seed, "size": size, "algo": "dqn",
                              "alpha": None}))
                for a in dc["cql_alphas"]:
                    jobs.append(("d3rlpy", f"cql{a}_{b}_{seed}_{size}", run_d3rlpy,
                                 {"behavior": b, "seed": seed, "size": size, "algo": "cql",
                                  "alpha": a}))
    results = runner.run_all(jobs, cfg, args.workers or cfg["workers"], force=args.force)
    runs = {k: v for k, v in results.items() if k[0] == GROUP}
    d3 = {k: v for k, v in results.items() if k[0] == "d3rlpy"}

    model = snap.drive
    agg = aggregate(cfg, runs, d3, model, snap.coach["table"])
    out = {
        **preamble,
        "status": "complete",
        "model": {
            "n_states": model.n,
            "share_of_drives_reaching_a_fourth_down": model.p_reach,
            "optimal_value_of_start_distribution": float(model.start @ model.v_star),
            "units": "expected points per drive (net of the opponent's next possession, as "
                     "valued by the solved model)",
        },
        **agg,
        "headline": headline_arrays(cfg, runs, d3, agg["tuning"]["cql_alpha_chosen"], model),
        "worked_example": worked_example(cfg, model, snap.coach["table"]),
        "hypothesis_numbers": None,
        "runtime_seconds": {
            "total_this_invocation": time.time() - t0,
            "sum_of_run_seconds": float(sum(r.get("seconds", 0) for r in results.values())),
        },
    }
    out["hypothesis_numbers"] = check_hypotheses(cfg, agg)
    write_json(out, out_path)
    print(f"wrote {out_path}")
    if out_path == OUT:
        print(figures.make_known_truth())


if __name__ == "__main__":
    main()
