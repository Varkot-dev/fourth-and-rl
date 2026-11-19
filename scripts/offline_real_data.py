"""M6 Part B: offline RL on real logged fourth-down decisions.

Run:  .venv/bin/python scripts/offline_real_data.py            (about 4 minutes)
      .venv/bin/python scripts/offline_real_data.py --figures  (redraw from the results)

1. Real neutral-situation fourth downs of the development seasons (2010 to 2023, through
   `splits.load_dev_plays`) become transitions of the drive model (`offline.real`).
2. The same learners as Part A run on them. There is no ground truth here: each method's
   claimed values and policy are compared with the solved model's optimum, which is itself
   a model fitted on the same seasons. That is a comparison between two models.
3. Held-out check on 2024 to 2025: the learned policies are evaluated off-policy on the
   fourth downs that actually happened (importance sampling, weighted importance sampling,
   doubly robust with the solved model as the direct method), with the behavior policy
   estimated on development seasons. Test plays are read once, through
   `splits.load_test_plays(purpose=...)`; the decision outcomes built from them are cached
   under data/models/offline/ so a rerun does not read them again. The 2026 holdout is
   never touched.

The CQL weight is the one chosen on the tuning seeds of Part A (known-truth world), read
from results/m6_offline_known_truth.json; nothing is tuned on real data.

Writes results/m6_offline_real_data.json and assets/m6_real_data.png.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from fourth_and_rl import splits  # noqa: E402
from fourth_and_rl.io import read_json, write_json  # noqa: E402
from fourth_and_rl.offline import datasets as ds  # noqa: E402
from fourth_and_rl.offline import figures, real  # noqa: E402
from fourth_and_rl.offline import learners as L  # noqa: E402
from fourth_and_rl.offline import measure as M  # noqa: E402
from fourth_and_rl.paths import CONFIGS, RESULTS  # noqa: E402
from fourth_and_rl.rl import evaluate as ev  # noqa: E402
from fourth_and_rl.rl import runner  # noqa: E402
from fourth_and_rl.rl.envs import GO  # noqa: E402

OUT = RESULTS / "m6_offline_real_data.json"
PART_A = RESULTS / "m6_offline_known_truth.json"
TEST_PURPOSE = ("M6 offline RL: held-out off-policy evaluation of fourth-down policies "
                "learned on 2010-2023, on 2024-2025 neutral fourth downs (evaluation only, "
                "never fitted on)")


def _stamp(cfg: dict[str, Any]) -> str:
    meta = (runner.resolve(cfg["snapshot_dir"]) / "meta.json").read_bytes()
    conf = json.dumps({k: cfg[k] for k in ("real_data", "learners", "drive")}, sort_keys=True)
    return hashlib.sha1(meta + conf.encode()).hexdigest()[:16]


def test_outcomes(cfg: dict[str, Any], grid: Any) -> tuple[pl.DataFrame, dict[str, Any]]:
    """2024 to 2025 decision outcomes, read from the test file once and then cached."""
    cache = runner.resolve(cfg["models_dir"]) / "test_decision_outcomes_2024_2025.parquet"
    meta_p = cache.with_suffix(".json")
    if cache.exists() and meta_p.exists():
        return pl.read_parquet(cache), read_json(meta_p)
    plays = splits.load_test_plays(purpose=TEST_PURPOSE)
    held = splits.seasons_in(plays) & set(splits.holdout_seasons())
    if held:
        raise splits.LeakageError(f"test file holds holdout seasons {sorted(held)}")
    out = real.decision_outcomes(plays, grid, cfg["real_data"]["neutral_only"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(cache)
    meta = {"read_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "purpose": TEST_PURPOSE, "seasons": sorted(splits.seasons_in(plays)),
            "n_play_rows": int(plays.height)}
    write_json(meta, meta_p)
    return out, meta


def d3rlpy_fits(cfg: dict[str, Any], model: Any, data: ds.Logged, alpha_list: list[float]
                ) -> dict[str, L.Fit]:
    """Offline DoubleDQN and DiscreteCQL on the real data, cached by config and model."""
    try:
        from fourth_and_rl.offline import deep
    except ImportError:
        return {}
    dc = dict(cfg["learners"]["d3rlpy"])
    keep = ("hidden_units", "learning_rate", "batch_size", "target_update_interval",
            "n_steps", "reward_scale")
    cache = runner.resolve(cfg["models_dir"]) / f"real_d3rlpy_{_stamp(cfg)}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: L.Fit(k, z[f"{k}__policy"], z[f"{k}__q"]) for k in
                {n.split("__")[0] for n in z.files}}
    out: dict[str, L.Fit] = {}
    seed = cfg["real_data"]["d3rlpy_seeds"][0]
    out["dqn_d3rlpy"] = deep.fit_d3rlpy(model, data, "dqn", seed, {k: dc[k] for k in keep})
    for a in alpha_list:
        f = deep.fit_d3rlpy(model, data, "cql", seed, {**{k: dc[k] for k in keep}, "alpha": a})
        out[f"cql_d3rlpy@{a}"] = f
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{f"{k}__policy": f.policy for k, f in out.items()},
             **{f"{k}__q": f.q for k, f in out.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    if args.figures:
        print(figures.make_real_data())
        return
    t0 = time.time()
    cfg = runner.load_config(CONFIGS / "offline.yaml")
    rd, lc = cfg["real_data"], cfg["learners"]
    snap = runner.snapshot(cfg)
    model = snap.drive
    from fourth_and_rl.components import ComponentModels

    preamble = {
        "what": "M6 Part B: offline RL on real logged fourth downs",
        "hypotheses": cfg["hypotheses_real_data"],
        "hypotheses_recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "running",
        "snapshot": snap.describe(),
        "development_component_bundle_stamp": getattr(ComponentModels.load(), "stamp", None),
        "config": {"real_data": rd, "learners": lc},
    }
    old = read_json(OUT) if OUT.exists() else None
    if old is not None and old.get("hypotheses") == preamble["hypotheses"]:
        preamble["hypotheses_recorded_at"] = old.get("hypotheses_recorded_at")
    write_json(preamble, OUT)

    # ---- 1. development transitions -------------------------------------------------
    dev = splits.load_dev_plays()
    splits.assert_no_test_seasons(dev, "M6 real-data offline learners")
    data, kept, info = real.build_logged(dev, model, "M6 real-data transitions",
                                         rd["neutral_only"])
    fit_stamp = splits.fit_stamp(kept, "M6 real-data offline learners (neutral fourth downs)")
    n_cols = model.n + model.n_term
    c = ds.counts(data, n_cols)
    info["matches_part_a_real_size"] = info["n_transitions"] == cfg["known_truth"]["real_size"]

    # ---- 2. learners ----------------------------------------------------------------
    part_a = read_json(PART_A) if PART_A.exists() else {}
    alpha = part_a.get("tuning", {}).get("cql_alpha_chosen")
    if alpha is None:
        raise RuntimeError("run scripts/offline_known_truth.py first (it picks the CQL alpha)")
    bc = L.behavior_cloning(model, c, lc["bc"]["smoothing"])
    plug_kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in lc["plugin"].items()}
    fits: dict[str, L.Fit] = {
        "bc_mode": L.Fit("bc_mode", bc.info["mode"]),
        "fqi_tabular": L.fqi_tabular(model, c, q_init=lc["fqi_tabular"]["q_init"]),
        "fqi_gbt": L.fqi_gbt(model, c, lc["fqi_gbt"]["n_iter"], rd["seed"],
                             lc["fqi_gbt"]["params"]),
        "fqi_lcb": L.fqi_tabular(model, c, lcb_beta=lc["fqi_lcb"]["beta"]),
        "cql_tabular": L.cql_tabular(model, c, alpha, outer_iter=lc["cql_tabular"]["outer_iter"],
                                     inner_steps=lc["cql_tabular"]["inner_steps"]),
        "plugin": L.plugin(model, c, **plug_kw),
    }
    fits.update(d3rlpy_fits(cfg, model, data, list(lc["d3rlpy"]["cql_alphas"])))
    fits["model_optimum"] = L.Fit("model_optimum", model.policy_star, model.q_star)

    # ---- comparisons with the solved model -------------------------------------------
    n_sa = L.cell_counts(c)
    n_s = n_sa.sum(axis=1)
    w = n_s / n_s.sum()
    reach = M.reachable(model)
    mode = bc.info["mode"]
    rows = np.arange(model.n)
    claims: dict[str, Any] = {}
    agreement: dict[str, Any] = {}
    charts: dict[str, Any] = {
        "counts": {a: ev.chart(model, n_sa[:, i]).tolist()
                   for i, a in enumerate(("go", "field_goal", "punt"))},
        "model_policy": ev.chart(model, model.policy_star).tolist(),
        "reachable": ev.chart(model, M.reachable(model).astype(float)).tolist(),
        "methods": {},
    }
    for name, f in fits.items():
        pol = f.policy
        agreement[name] = {
            "with_model_optimum_logged": float((w * (pol == model.policy_star)).sum()),
            "with_coach_mode_logged": float((w * (pol == mode)).sum()),
            "go_rate_logged_states": float((w * (pol == GO)).sum()),
            "model_regret_of_policy": ev.drive_regret(model, pol),
        }
        if f.q is None or name == "model_optimum":
            continue
        gap = np.where(model.allowed[:, GO], f.q[:, GO] - model.q_star[:, GO], np.nan)
        thin = reach & (n_sa[:, GO] <= 9) & np.isfinite(gap)
        thick = reach & (n_sa[:, GO] >= 10) & np.isfinite(gap)
        q_model_pi = M.exact_q(model, ev.policy_value(model, pol))
        claim = f.q[rows, pol]
        true_m = q_model_pi[rows, pol]
        ok = (n_s > 0) & np.isfinite(claim)
        usual = pol == mode
        top = np.argsort(-np.where(reach & np.isfinite(gap), gap, -np.inf))[:12]
        g = model.grid
        _, dbin, yl = g.unpack(model.states)
        claims[name] = {
            "q_error_by_count_vs_model": M.q_error_by_count(model, f.q, c, reach),
            "go_gap_thin_mean": float(gap[thin].mean()) if thin.any() else None,
            "go_gap_thick_mean": float(gap[thick].mean()) if thick.any() else None,
            "n_states_thin_go_0_9": int(thin.sum()), "n_states_thick_go_10_plus": int(thick.sum()),
            "claimed_minus_model_value_of_chosen_action": {
                part: float((n_s[sel] * (claim - true_m)[sel]).sum() / n_s[sel].sum())
                if n_s[sel].sum() else None
                for part, sel in (("all", ok), ("usual", ok & usual), ("rare", ok & ~usual))
            },
            "largest_go_claims_over_model": [
                {"yards_to_go": figures.DIST_LABELS[int(dbin[i])], "yardline_100": int(yl[i]),
                 "go_logged": int(n_sa[i, GO]), "state_logged": int(n_s[i]),
                 "claimed_q_go": float(f.q[i, GO]), "model_q_go": float(model.q_star[i, GO]),
                 "model_best_action": ("go", "field_goal", "punt")[int(model.policy_star[i])],
                 "method_action": ("go", "field_goal", "punt")[int(pol[i])]} for i in top],
        }
        charts["methods"][name] = {
            "q_go_minus_model": np.where(np.isfinite(ev.chart(model, gap)),
                                         ev.chart(model, gap), None).tolist(),
            "policy": ev.chart(model, pol).tolist(),
        }
    charts["methods"]["model_optimum"] = {"policy": charts["model_policy"]}

    # ---- 3. held-out off-policy evaluation -------------------------------------------
    test_out, test_meta = test_outcomes(cfg, model.grid)
    splits.assert_can_score(fit_stamp, test_out, "M6 held-out off-policy evaluation")
    test_data, test_kept, test_drop = real.to_logged(test_out, model)
    eps = real.episodes(test_data, model)
    games = test_kept["game_id"].to_numpy()[[e["start"] for e in eps]]
    mu = ds.coach_policy(model, snap.coach["table"])
    ope: dict[str, Any] = {}
    o = rd["ope"]
    for name, f in fits.items():
        q_hat = ev.policy_q(model, f.policy)
        parts = real.ope_per_episode(eps, model, f.policy, mu, q_hat)
        summary = real.ope_summary(parts, games, o["n_boot"], o["seed"], o["level"])
        start = np.array([e["s"][0] for e in eps])
        if f.q is not None:
            v = f.q[start, f.policy[start]]
            summary["claimed_value_at_test_starts"] = float(np.nanmean(v))
        ope[name] = summary
    behavior_parts = real.ope_per_episode(eps, model, mu, mu, ev.policy_q(model, mu))
    ope["behavior_on_policy_check"] = {
        "logged_mean_return": float(behavior_parts["return"].mean()),
        "model_value_of_coach_policy_at_test_starts": float(behavior_parts["dm"].mean()),
        "note": "the solved model's value of the coach-like policy at the logged start states "
                "against what the logged drives returned: a calibration check of the model's "
                "continuation values on held-out drives",
    }
    test_counts = ds.counts(test_data, n_cols)
    go_rate_dev = float(n_sa[:, GO].sum() / n_sa.sum())
    go_rate_test = float(L.cell_counts(test_counts)[:, GO].sum() / test_counts.sum())

    out = {
        **preamble,
        "status": "complete",
        "data": {**info, "fit_stamp": fit_stamp},
        "cql_alpha_from_part_a": alpha,
        "agreement": agreement,
        "claims": claims,
        "charts": {**charts, "dist_labels": figures.DIST_LABELS},
        "held_out": {
            "test_access": test_meta,
            "n_decisions": int(test_out.height),
            "n_transitions": int(test_data.size),
            "dropped": test_drop,
            "n_complete_episodes_used": len(eps),
            "go_rate_dev": go_rate_dev,
            "go_rate_test": go_rate_test,
            "behavior_policy": "neutral slice of the solved model's coach table (2018 to 2023)",
            "direct_method": "exact Q of each policy in the solved model",
            "units": "expected points per drive from the first logged fourth down, with "
                     "endings valued by the solved model",
            "ope": ope,
        },
        "runtime_seconds": time.time() - t0,
    }
    write_json(out, OUT)
    print(f"wrote {OUT}")
    print(figures.make_real_data())


if __name__ == "__main__":
    main()
