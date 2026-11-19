"""M1: tune, fit and evaluate the win-probability model, as four separate steps.

The steps are separate so that no tuning or fitting run can see a test-season number.

  python scripts/fit_wp.py --tune
      Hyperparameter grid, season-blocked cross-validation on DEVELOPMENT seasons
      only (splits.load_dev_plays). Writes results/m1_wp_tuning.json and exits.

  python scripts/fit_wp.py
      Development step. Fits the headline model on the development seasons, saves
      it, cross-validates version 2 and version 1 on development seasons, runs the
      sanity checks. Writes results/m1_wp_dev.json. Reads no test row.

  python scripts/fit_wp.py --preregister
      Writes results/m1_wp_v2_preregistration.json: the configuration, the
      development evidence behind each choice and the evaluation plan. Refuses to
      overwrite. Reads no test row.

  python scripts/fit_wp.py --evaluate
      The one test-season evaluation. Refuses to run unless the preregistration
      exists and matches the configuration and the saved model. Loads 2024 to 2025
      through splits.load_test_plays (logged), scores version 2 and version 1, and
      writes results/m1_wp_metrics.json, data/processed/wp_predictions.parquet and
      the figures.

Outputs:
  data/models/wp.joblib                  headline model (not committed)
  data/processed/wp_predictions.parquet  out-of-fold win probability for every play
  results/m1_wp_tuning.json              hyperparameter grid
  results/m1_wp_dev.json                 development-season evidence
  results/m1_wp_v2_preregistration.json  what was fixed before the test seasons were scored
  results/m1_wp_metrics.json             every number quoted about this model
  assets/m1_wp_reliability.png           reliability curves on the test seasons
  assets/m1_wp_by_quarter.png            log loss and calibration error by quarter
  assets/m1_wp_endgame.png               end-game cells, version 1 against version 2
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from fourth_and_rl import splits, style, wp
from fourth_and_rl.io import read_json, write_json
from fourth_and_rl.paths import ASSETS, DATA, RESULTS

MODELS = {"ours": "wp_ours", "ours_v1": "wp_v1", "nflfastr": "wp", "vegas": "vegas_wp"}
LABELS = {
    "ours": "Ours, version 2 (baseline + trees)",
    "ours_v1": "Ours, version 1",
    "nflfastr": "nflfastR wp (no spread)",
    "vegas": "nflfastR vegas_wp (with spread)",
}
V1_COLOR = style.MUTED
MODEL_PATH = DATA / "models" / "wp.joblib"
DEV_CROSSFIT_PATH = DATA / "processed" / "wp_dev_crossfit.parquet"
PREDICTIONS_PATH = DATA / "processed" / "wp_predictions.parquet"
TUNING_JSON = RESULTS / "m1_wp_tuning.json"
DEV_JSON = RESULTS / "m1_wp_dev.json"
STUDY_JSON = RESULTS / "m1_wp_v2_dev_study.json"
PREREG_JSON = RESULTS / "m1_wp_v2_preregistration.json"
METRICS_JSON = RESULTS / "m1_wp_metrics.json"
V1_METRICS_JSON = RESULTS / "m1_wp_metrics_v1.json"
# One line per --evaluate run, written just before the test seasons are read, so a second
# run for the same preregistration is refused (see `previous_evaluations`).
EVALUATIONS_LOG = RESULTS / "m1_wp_v2_evaluations.jsonl"
RESCORE_FLAG = "--i-know-this-rescores-the-test-seasons"
KEEP = [*wp.STATE_COLUMNS, wp.LABEL, "game_id", "play_id", "season", "play_type", "wp",
        "vegas_wp"]  # fmt: skip
TEST_PURPOSE = (
    "M1 win probability version 2: the single pre-registered evaluation on 2024-2025 "
    "(scripts/fit_wp.py --evaluate), with version 1 scored alongside"
)


def seasons(bounds: list[int] | None) -> list[int]:
    return list(range(bounds[0], bounds[1] + 1)) if bounds else []


def model_definition(cfg: dict) -> dict:
    """Everything that determines the fitted model. Its hash ties the steps together."""
    return {
        "spec": wp.V2.name,
        "features": wp.FEATURES,
        "monotone_constraints": {f: wp.MONOTONE.get(f, 0) for f in wp.FEATURES},
        "baseline": {"used": wp.V2.baseline, "tau": wp.V2.baseline_tau,
                     "logit_cap": wp.BASELINE_LOGIT_CAP},  # fmt: skip
        "constants": {
            "SPREAD_DECAY": wp.SPREAD_DECAY, "TIME_FLOOR_SECONDS": wp.TIME_FLOOR_SECONDS,
            "KNEEL_SECONDS": wp.KNEEL_SECONDS, "PLAY_CLOCK_SECONDS": wp.PLAY_CLOCK_SECONDS,
            "KNEEL_CAP_SECONDS": wp.KNEEL_CAP_SECONDS,
            "KNEEL_SLACK_FLOOR": wp.KNEEL_SLACK_FLOOR,
        },  # fmt: skip
        "lgbm": cfg["lgbm"],
        "calibration": cfg["calibration"],
        "split": cfg["split"],
        "seed": cfg["seed"],
    }


def definition_hash(cfg: dict) -> str:
    text = json.dumps(model_definition(cfg), sort_keys=True, default=float)
    return hashlib.sha256(text.encode()).hexdigest()


def new_model(cfg: dict) -> wp.WPModel:
    return wp.WPModel(cfg["lgbm"], cfg["calibration"], cfg["n_threads"], wp.V2)


def fit_headline(dev: pl.DataFrame, cfg: dict, n_estimators: int | None = None) -> wp.WPModel:
    train = wp.season_filter(dev, seasons(cfg["split"]["train"]))
    calib_seasons = seasons(cfg["split"].get("calib"))
    calib = wp.season_filter(dev, calib_seasons) if calib_seasons else None
    return new_model(cfg).fit(train, calib, seed=cfg["seed"], n_estimators=n_estimators,
                              what="M1 headline model (version 2)")  # fmt: skip


def fit_version_1(dev: pl.DataFrame, cfg: dict) -> wp.WPModel:
    """Version 1 exactly as it was shipped: trees 2010 to 2021, Platt map 2022 to 2023."""
    v1 = cfg["version_1"]
    return wp.WPModel(v1["lgbm"], v1["calibration"], cfg["n_threads"], wp.V1).fit(
        wp.season_filter(dev, seasons(v1["train"])), wp.season_filter(dev, seasons(v1["calib"])),
        seed=cfg["seed"], what="M1 version 1, refitted for comparison",
    )  # fmt: skip


def comparable(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Rows where the label and every listed prediction exist."""
    keep = pl.col(wp.LABEL).is_not_null()
    for col in columns:
        keep = keep & pl.col(col).is_not_null()
    return df.filter(keep)


def score_models(df: pl.DataFrame, n_bins: int, names: tuple[str, ...]) -> dict:
    y = df[wp.LABEL].to_numpy()
    out: dict = {"rows": df.height, "games": df["game_id"].n_unique()}
    for name in names:
        out[name] = wp.all_metrics(y, df[MODELS[name]].to_numpy(), n_bins)
    return out


def late_masks(df: pl.DataFrame) -> dict[str, np.ndarray]:
    late5 = wp.endgame_rows(df).to_numpy()
    return {
        "all_plays": np.ones(df.height, bool),
        "last_5_minutes": late5,
        "last_2_minutes": late5 & (df["game_seconds_remaining"].to_numpy() <= 120),
    }


def by_slice(df: pl.DataFrame, n_bins: int, names: tuple[str, ...]) -> dict:
    return {k: score_models(df.filter(pl.Series(m)), n_bins, names)
            for k, m in late_masks(df).items()}  # fmt: skip


def paired(df: pl.DataFrame, a: str, b: str, n_boot: int, seed: int) -> dict:
    """Game-bootstrap difference a minus b: all plays, last 5:00 and last 2:00."""
    y, games = df[wp.LABEL].to_numpy(), df["game_id"].to_numpy()
    pa, pb = df[MODELS[a]].to_numpy(), df[MODELS[b]].to_numpy()
    return {k: wp.game_bootstrap_diff(y[m], pa[m], pb[m], games[m], n_boot, seed)
            for k, m in late_masks(df).items()}  # fmt: skip


def endgame_tables(df: pl.DataFrame, names: tuple[str, ...], n_boot: int, seed: int) -> dict:
    out = {}
    for name in names:
        p = df[MODELS[name]].to_numpy()
        table = wp.endgame_slice_table(df, p, n_boot, seed)
        first = wp.endgame_slice_table(df, p, n_boot, seed, first_down_only=True)
        out[name] = {
            "summary": wp.endgame_summary(table), "cells": table,
            "summary_first_downs": wp.endgame_summary(first), "cells_first_downs": first,
        }  # fmt: skip
    return out


def diagnosis_cells(tables: dict) -> dict:
    """The three first-down cells that exposed version 1, for each model."""
    wanted = {"lead_1_3_0:41_2:00": ("0:41-2:00", "lead 1-3"),
              "lead_1_3_0:00_0:40": ("0:00-0:40", "lead 1-3"),
              "trail_4_8_0:00_0:40": ("0:00-0:40", "trail 4-8")}  # fmt: skip
    out: dict = {}
    for key, (t, m) in wanted.items():
        out[key] = {}
        for name, tab in tables.items():
            cell = next(c for c in tab["cells_first_downs"] if c["time"] == t and c["margin"] == m)
            out[key][name] = {k: cell.get(k) for k in
                              ("rows", "games", "observed", "predicted", "gap", "gap_ci_low",
                               "gap_ci_high")}  # fmt: skip
    return out


# --------------------------------------------------------------------------- tuning


def tune(cfg: dict) -> dict:
    """Grid over leaf size and tree size, then tree count and learning rate.

    Development seasons only, season-blocked cross-validation (the same folds as the
    design study). Returns the table; the configuration in use and the reason for it
    are recorded next to the winner, so a non-winning choice is visible.
    """
    t = cfg["tuning"]
    dev = splits.load_dev_plays().select(KEEP)
    n_bins = cfg["metrics"]["ece_bins"]

    def run(overrides: dict) -> dict:
        params = {**cfg["lgbm"], **overrides}
        cv = wp.season_blocked_cv(dev, params, cfg["calibration"], wp.V2, t["fold_size"], 0,
                                  cfg["seed"], cfg["n_threads"])  # fmt: skip
        frame, pred = cv["frame"], cv["pred"]
        y, masks = frame[wp.LABEL].to_numpy(), late_masks(frame)
        table = wp.endgame_slice_table(frame, pred, 200, cfg["seed"])
        row = {
            **overrides,
            **wp.all_metrics(y, pred, n_bins),
            "last_5_minutes_log_loss": wp.log_loss(y[masks["last_5_minutes"]],
                                                   pred[masks["last_5_minutes"]]),
            "last_2_minutes_log_loss": wp.log_loss(y[masks["last_2_minutes"]],
                                                   pred[masks["last_2_minutes"]]),
            "endgame_mean_abs_gap": wp.endgame_summary(table)["mean_abs_gap"],
            "memorization_gap": float(np.mean([f["held_out_log_loss"] - f["in_sample_log_loss"]
                                               for f in cv["folds"]])),
            "fit_seconds": round(float(np.mean([f["fit_seconds"] for f in cv["folds"]])), 1),
        }  # fmt: skip
        print(f"  tune {overrides}: log loss {row['log_loss']:.5f}, last 2:00 "
              f"{row['last_2_minutes_log_loss']:.4f}", flush=True)  # fmt: skip
        return row

    rows = [run({"min_child_samples": m, "num_leaves": n})
            for m, n in itertools.product(t["min_child_samples"], t["num_leaves"])]  # fmt: skip
    best = min(rows, key=lambda r: r["log_loss"])
    base = {"min_child_samples": best["min_child_samples"], "num_leaves": best["num_leaves"]}
    in_use = {k: cfg["lgbm"][k] for k in base}
    tree_rows = [run({**in_use, "n_estimators": n, "learning_rate": lr})
                 for n, lr in t["trees_and_learning_rate"]]  # fmt: skip
    return {
        "seasons": sorted(dev["season"].unique().to_list()),
        "fold_size": t["fold_size"],
        "note": "Development seasons only, season-blocked cross-validation. The test "
        "seasons are not read by this step, which exits when the grid is done.",
        "fixed": {k: v for k, v in cfg["lgbm"].items() if k not in base},
        "leaf_grid": rows,
        "best_leaf_setting_by_log_loss": base,
        "leaf_setting_in_use": in_use,
        "tree_count_grid_at_setting_in_use": tree_rows,
        "config_in_use": cfg["lgbm"],
        "how_each_value_was_chosen": cfg["provenance"],
    }


# ---------------------------------------------------------------- development step


def calibration_evidence(cv: dict, n_bins: int) -> dict:
    """Is a calibration map needed? Platt maps fitted to the cross-fitted raw scores.

    Every development season is scored by a model that did not see it. A slope of 1
    and an intercept of 0 mean "leave the model alone". The pooled map's metrics are
    in-sample for the map (two numbers on all rows), which favors the map.
    """
    frame, raw = cv["frame"], cv["raw"]
    y, season = frame[wp.LABEL].to_numpy(), frame["season"].to_numpy()
    pooled = wp.Calibrator("platt").fit(raw, y)
    by_season = {}
    for s in sorted(set(season.tolist())):
        cal = wp.Calibrator("platt").fit(raw[season == s], y[season == s])
        by_season[str(s)] = {"slope": cal.a, "intercept": cal.b}
    # Honest version of the pooled map: for each season, fit it on the other seasons.
    leave_out = np.empty_like(raw)
    for s in sorted(set(season.tolist())):
        cal = wp.Calibrator("platt").fit(raw[season != s], y[season != s])
        leave_out[season == s] = cal.transform(raw[season == s])
    slopes = [v["slope"] for v in by_season.values()]
    return {
        "platt_pooled": {"slope": pooled.a, "intercept": pooled.b},
        "platt_by_season": by_season,
        "season_slope_min": min(slopes),
        "season_slope_max": max(slopes),
        "metrics_no_map": wp.all_metrics(y, wp.Calibrator("none").transform(raw), n_bins),
        "metrics_pooled_map_leave_one_season_out": wp.all_metrics(y, leave_out, n_bins),
        "metrics_pooled_map_in_sample": wp.all_metrics(y, pooled.transform(raw), n_bins),
    }


def side_consistency(df: pl.DataFrame, col: str) -> dict:
    """Calibration in the large, split by which side has the ball.

    Turn each prediction into the home team's win probability (p when the home team
    has the ball, 1 - p otherwise). If the model treated the two sides of a game
    inconsistently, the home team's average would differ between the two groups by
    more than its actual win rate does.
    """
    out = {}
    for name, home in (("home_has_ball", 1), ("away_has_ball", 0)):
        sub = df.filter(pl.col("is_home") == home)
        p, y = sub[col].to_numpy(), sub[wp.LABEL].to_numpy()
        home_p, home_y = (p, y) if home else (1 - p, 1 - y)
        out[name] = {
            "rows": sub.height,
            "mean_predicted_home_wp": float(home_p.mean()),
            "home_win_rate": float(home_y.mean()),
            "gap": float(home_p.mean() - home_y.mean()),
        }
    out["passed"] = all(abs(v["gap"]) < 0.03 for v in out.values())
    return out


def smoothness(model: wp.WPModel) -> dict:
    """How step-like is the model along the field, the spread and the clock?

    For sweeps of one input with the rest fixed: the share of adjacent steps where the
    prediction does not move at all, and the largest single step.
    """
    yards = np.arange(1.0, 100.0)
    sweeps = {
        "yardline_1_to_99": dict(yardline_100=yards, ydstogo=np.minimum(10.0, yards)),
        "spread_minus14_to_14": dict(posteam_spread=np.arange(-14.0, 14.5, 0.5)),
        "clock_last_300_seconds_up_3": dict(game_seconds_remaining=np.arange(0.0, 301.0),
                                            score_differential=3.0),
    }  # fmt: skip
    out = {}
    for name, kwargs in sweeps.items():
        flat, big = [], []
        for secs in ((None,) if "clock" in name else (1800.0, 900.0, 300.0)):
            kw = dict(kwargs) if secs is None else {**kwargs, "game_seconds_remaining": secs}
            step = np.abs(np.diff(model.predict(wp.state_frame(**kw))))
            flat.append(float(np.mean(step < 1e-9)))
            big.append(float(step.max()))
        out[name] = {"share_of_flat_steps": float(np.mean(flat)), "largest_step": max(big)}
    return out


def split_counts(model: wp.WPModel) -> dict:
    """How many of the trees' splits use the timeout and clock-rule features."""
    counts = model.booster.feature_importance(importance_type="split")
    names = model.spec.features
    keys = [f for f in names if "timeout" in f or f == "kneel_slack"]
    return {"total_splits": int(counts.sum()),
            **{k: int(counts[names.index(k)]) for k in keys}}  # fmt: skip


def development_step(cfg: dict) -> None:
    seed, n_bins, boot = cfg["seed"], cfg["metrics"]["ece_bins"], cfg["metrics"]["bootstrap_reps"]
    dev = splits.load_dev_plays().select(KEEP)

    start = time.perf_counter()
    model = fit_headline(dev, cfg)
    fit_seconds = time.perf_counter() - start
    model.save(MODEL_PATH)
    start = time.perf_counter()
    fit_headline(dev, cfg, n_estimators=100)
    fast_seconds = time.perf_counter() - start
    print(f"headline fit: {fit_seconds:.1f}s on {model.n_train_rows:,} rows "
          f"(100 trees: {fast_seconds:.1f}s); baseline {model.baseline}", flush=True)  # fmt: skip

    # Season-blocked cross-validation on development seasons: version 2 and version 1.
    fold = cfg["crossfit"]["fold_size"]
    start = time.perf_counter()
    cv2 = wp.season_blocked_cv(dev, cfg["lgbm"], cfg["calibration"], wp.V2, fold, 0, seed,
                               cfg["n_threads"])  # fmt: skip
    cv_seconds = time.perf_counter() - start
    v1 = cfg["version_1"]
    cv1 = wp.season_blocked_cv(dev, v1["lgbm"], v1["calibration"], wp.V1, fold,
                               v1["n_calib_seasons_in_cv"], seed, cfg["n_threads"])  # fmt: skip
    frame = cv2["frame"].with_columns(pl.Series("wp_ours", cv2["pred"]),
                                      pl.Series("wp_v1", cv1["pred"]))  # fmt: skip
    frame.select("game_id", "play_id", "wp_ours", "wp_v1").write_parquet(DEV_CROSSFIT_PATH)
    ours = ("ours", "ours_v1")
    three = comparable(frame, ["wp_ours", "wp_v1", "wp", "vegas_wp"])
    tables = endgame_tables(frame, ours, boot, seed)

    def memorization(cv: dict) -> dict:
        ins = float(np.mean([f["in_sample_log_loss"] for f in cv["folds"]]))
        out = float(np.mean([f["held_out_log_loss"] for f in cv["folds"]]))
        return {"in_sample_log_loss": ins, "held_out_log_loss": out, "gap": out - ins}

    sanity = wp.sanity_checks(model)
    sanity["side_consistency_development_cross_fitted"] = side_consistency(frame, "wp_ours")
    openers = dev.filter(
        (pl.col("play_type") == "kickoff") & (pl.col("game_seconds_remaining") == 3600)
        & pl.col(wp.LABEL).is_not_null()
    )  # fmt: skip
    sanity["kickoff_mirror"]["development_opening_kickoffs"] = openers.height
    sanity["kickoff_mirror"]["development_receiver_win_rate"] = float(openers[wp.LABEL].mean())
    side_ok = sanity["side_consistency_development_cross_fitted"]["passed"]
    sanity["core_passed"] &= side_ok
    sanity["all_passed"] &= side_ok

    out = {
        "note": "Development seasons only (splits.load_dev_plays). No test-season or "
        "holdout row is read by this step.",
        "definition_sha256": definition_hash(cfg),
        "model_definition": model_definition(cfg),
        "seasons": sorted(dev["season"].unique().to_list()),
        "headline_model": {
            "stamp": model.stamp,
            "baseline": model.baseline,
            "calibration": model.calibrator.describe(),
            "rows_used": model.n_train_rows,
            "feature_importance_gain_share": model.feature_importance(),
            "split_counts": split_counts(model),
            "smoothness": smoothness(model),
        },
        "fit_seconds": {
            "headline_fit": round(fit_seconds, 1),
            "fit_with_100_trees": round(fast_seconds, 1),
            "cross_validation_all_folds": round(cv_seconds, 1),
            "threads": cfg["n_threads"],
        },
        "cross_validation": {
            "folds": [{k: f[k] for k in ("predict", "train", "calib")} for f in cv2["folds"]],
            "version_1_folds": [{k: f[k] for k in ("predict", "train", "calib")}
                                for f in cv1["folds"]],
            "rows": frame.height,
            "games": frame["game_id"].n_unique(),
            "metrics": by_slice(frame, n_bins, ours),
            "version_2_minus_version_1": paired(frame, "ours", "ours_v1", boot, seed),
            "by_season": {
                str(s): score_models(frame.filter(pl.col("season") == s), n_bins, ours)
                for s in sorted(frame["season"].unique().to_list())
            },
            "memorization": {"version_2": memorization(cv2), "version_1": memorization(cv1)},
            "baseline_by_fold": [f["baseline"] for f in cv2["folds"]],
            "against_nflfastr_same_rows": {
                **score_models(three, n_bins, ("ours", "ours_v1", "nflfastr", "vegas")),
                "note": "nflfastR's models were probably fitted on some of these seasons, "
                "so its numbers here may be in-sample. Ours are cross-fitted.",
            },
        },  # fmt: skip
        "endgame": {
            "definition": "Fourth quarter, 5:00 or less, scrimmage downs. Margin is for "
            "the team with the ball. gap = predicted minus observed; the interval "
            "resamples whole games.",
            **tables,
            "diagnosis_cells_first_downs": diagnosis_cells(tables),
        },
        "calibration_evidence": calibration_evidence(cv2, n_bins),
        "sanity_checks": sanity,
    }  # fmt: skip
    write_json(out, DEV_JSON)
    m = out["cross_validation"]["metrics"]
    for k in m:
        print(f"  dev {k:15s} v2 {m[k]['ours']['log_loss']:.4f}  v1 "
              f"{m[k]['ours_v1']['log_loss']:.4f}")  # fmt: skip
    print(f"  end-game mean |gap|: v2 {tables['ours']['summary']['mean_abs_gap']:.4f}  "
          f"v1 {tables['ours_v1']['summary']['mean_abs_gap']:.4f}")  # fmt: skip
    failed = [k for k, v in sanity.items() if isinstance(v, dict) and not v["passed"]]
    print(f"sanity checks: core passed {sanity['core_passed']}, end game passed "
          f"{sanity['endgame_passed']}; failing: {failed or 'none'}")  # fmt: skip
    if not sanity["core_passed"]:
        raise SystemExit(f"core sanity checks failed: {failed}")


# ------------------------------------------------------------------ preregistration


def preregister(cfg: dict) -> None:
    """Freeze the configuration, the development evidence and the plan. Dev files only."""
    if PREREG_JSON.exists():
        raise SystemExit(f"{PREREG_JSON} already exists; a preregistration is written once")
    dev, study = read_json(DEV_JSON), read_json(STUDY_JSON)
    if dev["definition_sha256"] != definition_hash(cfg):
        raise SystemExit("results/m1_wp_dev.json is from another configuration; rerun the "
                         "development step first")  # fmt: skip
    if not dev["sanity_checks"]["core_passed"]:
        raise SystemExit("the development step's core sanity checks did not pass")
    failing = [k for k, v in dev["sanity_checks"].items()
               if isinstance(v, dict) and not v["passed"]]  # fmt: skip

    def brief(name: str) -> dict:
        c = study["candidates"][name]
        return {
            "log_loss": c["overall"]["log_loss"], "ece": c["overall"]["ece"],
            "last_5_minutes_log_loss": c["last_5_minutes"]["log_loss"],
            "last_2_minutes_log_loss": c["last_2_minutes"]["log_loss"],
            "endgame_mean_abs_gap": c["endgame_summary"]["mean_abs_gap"],
            "memorization_gap": c["memorization"]["gap"],
            "pooled_platt_slope": c["pooled_platt_on_held_out_scores"]["slope"],
        }  # fmt: skip

    decisions = [{**d, "development_evidence": {n: brief(n) for n in d["compared"]}}
                 for d in cfg["decisions"]]  # fmt: skip
    cvm = dev["cross_validation"]
    out = {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "statement": [
            "Written before any 2024 to 2025 number for version 2 was computed. Every "
            "choice below was made on development seasons (2010 to 2023) only, with "
            "season-blocked cross-validation.",
            "The 2024 to 2025 seasons are not an untouched test set: they were scored "
            "several times for version 1 and are where the end-game weakness was first "
            "seen. The diagnosis was reproduced on development seasons before anything "
            "was changed (results/m1_wp_v2_dev_study.json, candidate v1).",
            "The test evaluation is run once (scripts/fit_wp.py --evaluate). Whatever "
            "it shows is reported. Nothing about the model changes afterwards.",
            "The 2026 holdout has not been loaded and is not part of this evaluation.",
        ],
        "definition_sha256": definition_hash(cfg),
        "model_definition": model_definition(cfg),
        "headline_model_stamp": dev["headline_model"]["stamp"],
        "how_each_value_was_chosen": cfg["provenance"],
        "decisions": decisions,
        "development_results_of_the_chosen_model": {
            "source": "results/m1_wp_dev.json",
            "metrics": cvm["metrics"],
            "version_2_minus_version_1": cvm["version_2_minus_version_1"],
            "memorization": cvm["memorization"],
            "endgame_summary": {k: dev["endgame"][k]["summary"] for k in ("ours", "ours_v1")},
            "diagnosis_cells_first_downs": dev["endgame"]["diagnosis_cells_first_downs"],
            "calibration_evidence": {k: dev["calibration_evidence"][k] for k in
                                     ("platt_pooled", "season_slope_min", "season_slope_max",
                                      "metrics_no_map",
                                      "metrics_pooled_map_leave_one_season_out")},
            "sanity_checks_core_passed": dev["sanity_checks"]["core_passed"],
            "sanity_checks_failing": failing,
            "sanity_checks_note": "Checks that fail are reported, not hidden; their "
            "thresholds were set before version 2 was chosen and are not changed.",
        },  # fmt: skip
        "evaluation_plan": cfg["evaluation_plan"],
    }
    write_json(out, PREREG_JSON)
    print(f"wrote {PREREG_JSON}")


# --------------------------------------------------------------------------- figures


def reliability_band(df: pl.DataFrame, n_bins: int, n_boot: int, seed: int) -> np.ndarray:
    """95% game-bootstrap band for (observed - predicted) per bin, for our model."""
    p, y = df["wp_ours"].to_numpy(), df[wp.LABEL].to_numpy()
    _, g = np.unique(df["game_id"].to_numpy(), return_inverse=True)
    n_games = g.max() + 1
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    flat = g * n_bins + b
    shape = (n_games, n_bins)
    cnt = np.bincount(flat, minlength=n_games * n_bins).reshape(shape).astype(float)
    sum_y = np.bincount(flat, weights=y, minlength=n_games * n_bins).reshape(shape)
    sum_p = np.bincount(flat, weights=p, minlength=n_games * n_bins).reshape(shape)
    rng = np.random.default_rng(seed)
    gaps = np.empty((n_boot, n_bins))
    for i in range(n_boot):
        draw = rng.integers(0, n_games, n_games)
        c = cnt[draw].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gaps[i] = (sum_y[draw].sum(axis=0) - sum_p[draw].sum(axis=0)) / c
    return np.nanquantile(gaps, [0.025, 0.975], axis=0)


def figure_reliability(df: pl.DataFrame, metrics: dict, cfg: dict) -> None:
    style.apply()
    n_bins = cfg["metrics"]["reliability_bins"]
    y = df[wp.LABEL].to_numpy()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"wspace": 0.3})
    ax.plot([0, 1], [0, 1], color=style.MUTED, linewidth=1, linestyle=(0, (4, 3)),
            label="Perfect calibration")  # fmt: skip
    ax2.axhline(0, color=style.MUTED, linewidth=1, linestyle=(0, (4, 3)))
    low, high = reliability_band(df, n_bins, 500, cfg["seed"])
    for name in ("nflfastr", "vegas", "ours"):
        t = wp.reliability_table(y, df[MODELS[name]].to_numpy(), n_bins)
        x, obs = np.array(t["mean_pred"], float), np.array(t["observed"], float)
        ours = name == "ours"
        kw = dict(color=style.MODEL_COLORS[name], marker="o", markersize=4.5,
                  markeredgecolor=style.SURFACE, markeredgewidth=0.8,
                  linewidth=2.2 if ours else 1.6, zorder=3 if ours else 2)  # fmt: skip
        m = metrics[name]
        ax.plot(x, obs, label=LABELS[name], **kw)
        ax2.plot(x, 100 * (obs - x), **kw,
                 label=f"{LABELS[name].split(' (')[0]}: ECE {100 * m['ece']:.2f} pts")  # fmt: skip
        if name == "ours":
            ax2.fill_between(
                x, 100 * low, 100 * high, color=style.MODEL_COLORS[name], alpha=0.15,
                linewidth=0, label="Ours: 95% band (games resampled)",
            )  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Share of those plays the team went on to win")
    ax.legend(loc="upper left", fontsize=9.5)
    first, last = cfg["split"]["test"]
    style.title(
        ax,
        "Reliability on the test seasons",
        f"Test seasons {first} to {last}, {df.height:,} plays, "
        f"{df['game_id'].n_unique()} games, {n_bins} bins",
    )
    ax2.set_xlim(0, 1)
    span = max(6.0, float(np.nanmax(np.abs(np.concatenate([low, high])))) * 100 * 1.1)
    ax2.set_ylim(-span, span * 2.1)
    ax2.set_xlabel("Predicted win probability")
    ax2.set_ylabel("Observed minus predicted (percentage points)")
    ax2.legend(loc="upper left", fontsize=9.5)
    style.title(
        ax2,
        "Calibration error by bin",
        "Above zero: teams won more often than predicted",
    )
    style.save(fig, ASSETS / "m1_wp_reliability.png")


def figure_by_quarter(by_quarter: dict, cfg: dict) -> None:
    style.apply()
    quarters = [q for q in by_quarter if q != "Overtime"]
    names = ("ours", "vegas", "nflfastr")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"wspace": 0.22})
    width = 0.26
    panels = (("log_loss", 1.0, "Log loss"), ("ece", 100.0, "Expected calibration error (pts)"))
    for ax, (key, scale, ylabel) in zip(axes, panels, strict=True):
        for i, name in enumerate(names):
            vals = [scale * by_quarter[q][name][key] for q in quarters]
            xs = np.arange(len(quarters)) + (i - 1) * width
            ax.bar(xs, vals, width=width * 0.92, color=style.MODEL_COLORS[name],
                   label=LABELS[name])  # fmt: skip
            for x, v in zip(xs, vals, strict=True):
                ax.text(x, v, f"{v:.3f}"[1:] if key == "log_loss" else f"{v:.1f}", ha="center",
                        va="bottom", fontsize=7.5, color=style.INK_2)  # fmt: skip
        ax.set_xticks(np.arange(len(quarters)))
        ax.set_xticklabels(
            [f"{q}\n{by_quarter[q]['rows']:,} plays" for q in quarters], fontsize=9.5
        )
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    first, last = cfg["split"]["test"]
    axes[0].legend(loc="upper right", fontsize=9)
    style.title(
        axes[0],
        "Later quarters are easier to call",
        f"Log loss by quarter, test seasons {first} to {last} (lower is better)",
    )
    style.title(
        axes[1],
        "Calibration error by quarter",
        f"Regulation; overtime ({by_quarter['Overtime']['rows']} plays) is in the metrics json",
    )
    style.save(fig, ASSETS / "m1_wp_by_quarter.png")


def figure_endgame(dev_tables: dict, test_tables: dict, cfg: dict) -> None:
    """Predicted minus observed win rate in each end-game cell, version 1 and version 2."""
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 9.2), sharey=True,
                             gridspec_kw={"wspace": 0.42})  # fmt: skip
    dev_first, dev_last = cfg["split"]["train"][0], (cfg["split"].get("calib")
                                                     or cfg["split"]["train"])[1]  # fmt: skip
    test_first, test_last = cfg["split"]["test"]
    panels = (
        (dev_tables, "Development seasons, cross-fitted",
         f"{dev_first} to {dev_last}. Each season is predicted by models that never saw it"),
        (test_tables, "Test seasons",
         f"{test_first} to {test_last}, scored once; they first showed the problem"),
    )  # fmt: skip
    n_margin = len(wp.ENDGAME_MARGIN_BINS)
    n_cells = len(wp.ENDGAME_TIME_BINS) * n_margin
    rows = np.arange(n_cells)[::-1].astype(float)
    rows -= (np.arange(n_cells) // n_margin) * 1.6  # room for a header above each block
    spans = [abs(100 * c[k]) for t in (dev_tables, test_tables) for n in ("ours", "ours_v1")
             for c in t[n]["cells"] for k in ("gap_ci_low", "gap_ci_high")]  # fmt: skip
    limit = float(min(25.0, max(10.0, np.ceil(max(spans) + 1.0))))
    for ax, (tables, head, sub) in zip(axes, panels, strict=True):
        cells = tables["ours"]["cells"]
        ax.axvline(0, color=style.INK_2, linewidth=1)
        for name, color, shift in (("ours_v1", V1_COLOR, 0.17), ("ours", style.BLUE, -0.17)):
            tab = tables[name]["cells"]
            gap = np.array([100 * c["gap"] for c in tab])
            lo = np.array([100 * c["gap_ci_low"] for c in tab])
            hi = np.array([100 * c["gap_ci_high"] for c in tab])
            ax.hlines(rows + shift, lo, hi, color=color, linewidth=1.6, alpha=0.75)
            label = "Version 2" if name == "ours" else "Version 1"
            mean_gap = 100 * tables[name]["summary"]["mean_abs_gap"]
            ax.plot(gap, rows + shift, "o", color=color, markersize=5.5,
                    markeredgecolor=style.SURFACE, markeredgewidth=0.8,
                    label=f"{label}: mean miss {mean_gap:.1f} pts")  # fmt: skip
        ax.set_xlim(-limit, limit)
        for r, c in zip(rows, cells, strict=True):
            ax.text(1.02, r, f"{100 * c['observed']:.1f}%  ({c['rows']:,})",
                    transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.5,
                    color=style.INK_2)  # fmt: skip
        for i, (name, _, _) in enumerate(wp.ENDGAME_TIME_BINS):
            top = rows[i * n_margin] + 0.95
            ax.text(-limit * 0.97, top, f"{name} left", va="center", ha="left", fontsize=10,
                    color=style.INK, fontweight="bold")  # fmt: skip
            ax.text(1.02, top, "won  (plays)", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8.5, color=style.MUTED)  # fmt: skip
        ax.set_yticks(rows)
        ax.set_yticklabels([c["margin"] for c in cells], fontsize=9.5)
        ax.set_ylim(rows.min() - 0.8, rows.max() + 1.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Predicted minus observed win rate (percentage points)")
        ax.legend(loc="lower left", fontsize=9, bbox_to_anchor=(0.0, -0.16), ncols=2,
                  columnspacing=1.2, handletextpad=0.3)  # fmt: skip
        style.title(ax, head, sub)
    fig.suptitle(
        "Win probability at the end of close games: version 1 against version 2",
        x=0.045, y=1.0, ha="left", fontsize=15, fontweight="bold", color=style.INK,
    )  # fmt: skip
    fig.text(
        0.045, 0.955,
        "Fourth quarter, 5:00 or less, scrimmage downs. Rows are the score from the point "
        "of view of the team with the ball. Bars: 95% intervals, whole games resampled.",
        ha="left", fontsize=10, color=style.INK_2,
    )  # fmt: skip
    fig.subplots_adjust(top=0.86)
    style.save(fig, ASSETS / "m1_wp_endgame.png")


# ------------------------------------------------------------------- test evaluation


def previous_evaluations(digest: str) -> list[str]:
    """Evidence that the test seasons were already scored for the preregistration `digest`:
    a line of EVALUATIONS_LOG with that hash, or a metrics file that records it (the
    evaluation of version 2 predates the log, and its metrics file is the record)."""
    found = []
    if EVALUATIONS_LOG.exists():
        for line in EVALUATIONS_LOG.read_text().splitlines():
            entry = json.loads(line) if line.strip() else {}
            if entry.get("definition_sha256") == digest:
                found.append(f"{EVALUATIONS_LOG.name}: run at {entry.get('at')}")
    if METRICS_JSON.exists():
        saved = read_json(METRICS_JSON).get("preregistration", {})
        if saved.get("definition_sha256") == digest:
            found.append(f"{METRICS_JSON.name}: evaluated at {saved.get('evaluated_at')}")
    return found


def evaluate(cfg: dict, rescore: bool = False) -> None:
    """The one evaluation on the test seasons, for version 2 with version 1 alongside.

    Refuses to run a second time for the same preregistration unless `rescore` (the
    command-line flag RESCORE_FLAG): every rerun is another look at the test seasons."""
    seed, n_bins, boot = cfg["seed"], cfg["metrics"]["ece_bins"], cfg["metrics"]["bootstrap_reps"]
    if not PREREG_JSON.exists():
        raise SystemExit("no preregistration: run --preregister before --evaluate")
    prereg, dev_json = read_json(PREREG_JSON), read_json(DEV_JSON)
    digest = definition_hash(cfg)
    if prereg["definition_sha256"] != digest or dev_json["definition_sha256"] != digest:
        raise SystemExit("the configuration differs from the preregistered one; refusing")
    model = wp.WPModel.load(MODEL_PATH)
    known = {k: v for k, v in (model.stamp or {}).items() if k != "fitted_at"}
    wanted = {k: v for k, v in prereg["headline_model_stamp"].items() if k != "fitted_at"}
    if known != wanted or model.params != {**wp.DEFAULT_PARAMS, **cfg["lgbm"]}:
        raise SystemExit("data/models/wp.joblib is not the preregistered model; refusing")
    done = previous_evaluations(digest)
    if done and not rescore:
        raise SystemExit(
            "the test seasons were already scored for this preregistration ("
            + "; ".join(done) + f"). Pass {RESCORE_FLAG} to score them again on purpose."
        )

    dev = splits.load_dev_plays().select(KEEP)
    version_1 = fit_version_1(dev, cfg)
    EVALUATIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATIONS_LOG, "a") as log:
        entry = {"at": datetime.now(UTC).isoformat(timespec="seconds"),
                 "definition_sha256": digest, "rescore": bool(done)}  # fmt: skip
        log.write(json.dumps(entry) + "\n")
    test = splits.load_test_plays(TEST_PURPOSE).select(KEEP)
    splits.assert_can_score(model.stamp, test, "version 2 on the test seasons")
    splits.assert_can_score(version_1.stamp, test, "version 1 on the test seasons")
    test = test.with_columns(
        pl.Series("wp_ours", model.predict(test)).fill_nan(None),
        pl.Series("wp_v1", version_1.predict(test)).fill_nan(None),
    )
    names = ("ours", "ours_v1", "nflfastr", "vegas")
    ev = comparable(test, list(MODELS.values()))
    y, games = ev[wp.LABEL].to_numpy(), ev["game_id"].to_numpy()
    headline = score_models(ev, n_bins, names)
    for other in ("ours_v1", "nflfastr", "vegas"):
        headline[f"ours_minus_{other}"] = wp.game_bootstrap_diff(
            y, ev["wp_ours"].to_numpy(), ev[MODELS[other]].to_numpy(), games, boot, seed
        )
    headline["reliability"] = {
        name: wp.reliability_table(y, ev[MODELS[name]].to_numpy(),
                                   cfg["metrics"]["reliability_bins"])
        for name in names
    }  # fmt: skip
    labelled = test.filter(pl.col(wp.LABEL).is_not_null())
    test_rows = {
        "all": test.height,
        "tied_game_rows_dropped": test.height - labelled.height,
        "incomplete_state_rows_dropped": int(labelled["wp_ours"].is_null().sum()),
        "missing_nflfastr_rows_dropped": int(
            labelled.filter(pl.col("wp_ours").is_not_null())
            .select((pl.col("wp").is_null() | pl.col("vegas_wp").is_null()).sum())
            .item()
        ),
        "evaluated": ev.height,
    }
    quarter_names = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Overtime"}
    by_quarter = {label: score_models(ev.filter(pl.col("qtr").clip(1, 5) == q), n_bins, names)
                  for q, label in quarter_names.items()}  # fmt: skip
    subsets = {
        "scrimmage_downs": score_models(ev.filter(pl.col("down").is_not_null()), n_bins, names),
        "no_down_plays": score_models(ev.filter(pl.col("down").is_null()), n_bins, names),
        "fourth_downs": score_models(ev.filter(pl.col("down") == 4), n_bins, names),
        **{k: v for k, v in by_slice(ev, n_bins, names).items() if k != "all_plays"},
    }
    test_tables = endgame_tables(ev, ("ours", "ours_v1"), boot, seed)

    # Version 1 must reproduce the numbers it was shipped with (same rows, same model).
    old = read_json(V1_METRICS_JSON)["test"]
    reproduced = {
        "log_loss_then": old["ours"]["log_loss"], "log_loss_now": headline["ours_v1"]["log_loss"],
        "rows_then": old["rows"], "rows_now": ev.height,
        "identical": bool(old["rows"] == ev.height and abs(
            old["ours"]["log_loss"] - headline["ours_v1"]["log_loss"]) < 1e-9),
    }  # fmt: skip

    # Out-of-fold predictions for every season. Development seasons: cross-fitted on
    # the other development seasons. Test seasons: a model fitted on all development
    # seasons, which is the headline model again. No fold is fitted on a test season.
    plays = pl.concat([dev, test.select(KEEP)])
    start = time.perf_counter()
    oof, plan = wp.crossfit_predict(
        plays, cfg["lgbm"], cfg["calibration"], cfg["crossfit"]["fold_size"],
        cfg["crossfit"]["n_calib_seasons"], seed, cfg["n_threads"], wp.V2,
    )  # fmt: skip
    crossfit_seconds = time.perf_counter() - start
    assert oof.height == plays.height
    check = oof.join(test.select("game_id", "play_id", pl.col("wp_ours").alias("head")),
                     on=["game_id", "play_id"], how="inner", validate="1:1")  # fmt: skip
    same_as_headline = float((check["wp_ours"] - check["head"]).abs().max() or 0.0)
    stash = pl.read_parquet(DEV_CROSSFIT_PATH).select("game_id", "play_id",
                                                      pl.col("wp_ours").alias("cv"))  # fmt: skip
    check = oof.join(stash, on=["game_id", "play_id"], how="inner", validate="1:1")
    same_as_dev_step = float((check["wp_ours"] - check["cv"]).abs().max())
    if same_as_headline > 1e-9 or same_as_dev_step > 1e-9:
        raise SystemExit(
            f"cross-fitted predictions disagree with the development step "
            f"({same_as_dev_step}) or the headline model ({same_as_headline})"
        )
    oof.write_parquet(PREDICTIONS_PATH)
    joined = comparable(plays.join(oof, on=["game_id", "play_id"], how="left", validate="1:1"),
                        ["wp_ours", "wp", "vegas_wp"])  # fmt: skip
    three = ("ours", "nflfastr", "vegas")
    oof_metrics = {
        "all_seasons": score_models(joined, n_bins, three),
        "by_season": {str(s): score_models(joined.filter(pl.col("season") == s), n_bins, three)
                      for s in sorted(joined["season"].unique().to_list())},  # fmt: skip
    }

    metrics = {
        "version": 2,
        "seed": seed,
        "split": cfg["split"],
        "preregistration": {
            "file": "results/m1_wp_v2_preregistration.json",
            "written_at": prereg["written_at"],
            "definition_sha256": digest,
            "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "test_purpose_logged": TEST_PURPOSE,
        },
        "honesty_note": "The 2024 to 2025 seasons informed the diagnosis that led to "
        "version 2, so they are not an untouched test set. Every version 2 choice was "
        "made on 2010 to 2023 and frozen before these numbers were computed. The 2026 "
        "holdout has not been loaded.",
        "calibration": {"configured": cfg["calibration"], **model.calibrator.describe()},
        "baseline": model.baseline,
        "hyperparameters": model.params,
        "how_each_value_was_chosen": cfg["provenance"],
        "features": wp.FEATURES,
        "monotone_constraints": {f: wp.MONOTONE.get(f, 0) for f in wp.FEATURES},
        "model_stamp": model.stamp,
        "rows": {"train_used": model.n_train_rows, "train_games": model.stamp["n_games"],
                 "calib_used": model.n_calib_rows, "test": test_rows},  # fmt: skip
        "fit_seconds": {**dev_json["fit_seconds"],
                        "crossfit_all_folds": round(crossfit_seconds, 1)},  # fmt: skip
        "test": headline,
        "test_by_quarter": by_quarter,
        "test_subsets": subsets,
        "test_endgame": {
            "definition": dev_json["endgame"]["definition"],
            **test_tables,
            "diagnosis_cells_first_downs": diagnosis_cells(test_tables),
        },
        "test_side_consistency": side_consistency(ev, "wp_ours"),
        "version_1": {
            "what": "Version 1 refitted by the current code exactly as shipped (trees "
            "2010 to 2021, Platt map 2022 to 2023). Its original results file is "
            "results/m1_wp_metrics_v1.json.",
            "config": cfg["version_1"],
            "stamp": version_1.stamp,
            "calibration": version_1.calibrator.describe(),
            "reproduces_shipped_numbers": reproduced,
        },
        "development": {k: dev_json[k] for k in
                        ("cross_validation", "endgame", "calibration_evidence",
                         "headline_model")},  # fmt: skip
        "feature_importance_gain_share": model.feature_importance(),
        "sanity_checks": dev_json["sanity_checks"],
        "out_of_fold": {
            "plan": plan,
            "plan_note": "For each block of seasons in `predict`: the seasons whose plays "
            "fitted the model that predicted it (`train`) and its calibration seasons "
            "(`calib`, empty because version 2 uses no calibration map). Test seasons "
            "are in no `train` list.",
            "rows_written": oof.height,
            "rows_with_prediction": int(oof["wp_ours"].is_not_null().sum()),
            "max_abs_difference_from_headline_model_on_test": same_as_headline,
            "max_abs_difference_from_development_step": same_as_dev_step,
            **oof_metrics,
        },
        "notes": [
            "ece uses equal-width bins of the prediction (count in metrics.ece_bins).",
            "ours_minus_* are game-level bootstrap differences; negative means ours is better.",
            "The seasons nflfastR fitted its models on are not known to this repo. If they "
            "include a season evaluated here, the nflfastR numbers for it are in-sample.",
            "Sanity checks are run in the development step, on hypothetical states and on "
            "development seasons; no test row can fail the run.",
        ],
    }
    write_json(metrics, METRICS_JSON)
    figure_reliability(ev, headline, cfg)
    figure_by_quarter(by_quarter, cfg)
    figure_endgame(dev_json["endgame"], test_tables, cfg)

    print(f"test rows: {ev.height:,} in {headline['games']} games")
    for name in names:
        m = headline[name]
        print(f"  {name:10s} log loss {m['log_loss']:.4f}  Brier {m['brier']:.4f}  "
              f"ECE {m['ece']:.4f}")  # fmt: skip
    for other in ("ours_v1", "nflfastr", "vegas"):
        d = headline[f"ours_minus_{other}"]["log_loss"]
        print(f"  ours_minus_{other}: log loss {d['diff']:+.4f} "
              f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")  # fmt: skip
    print(f"version 1 reproduced: {reproduced['identical']}")


def figures_only(cfg: dict) -> None:
    """Redraw the end-game figure from the saved JSON files (no data is read)."""
    metrics, dev_json = read_json(METRICS_JSON), read_json(DEV_JSON)
    figure_endgame(dev_json["endgame"], metrics["test_endgame"], cfg)
    figure_by_quarter(metrics["test_by_quarter"], cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--tune", action="store_true",
                      help="hyperparameter grid on development seasons, then exit")  # fmt: skip
    step.add_argument("--preregister", action="store_true",
                      help="freeze configuration, development evidence and plan")  # fmt: skip
    step.add_argument("--evaluate", action="store_true",
                      help="the one test-season evaluation (needs --preregister)")  # fmt: skip
    step.add_argument("--figures", action="store_true",
                      help="redraw figures from the saved JSON files")  # fmt: skip
    parser.add_argument(RESCORE_FLAG, dest="rescore", action="store_true",
                        help="with --evaluate: score the test seasons again for a "
                             "preregistration that was already evaluated")  # fmt: skip
    args = parser.parse_args()
    cfg = wp.load_config()
    if args.tune:
        write_json(tune(cfg), TUNING_JSON)
    elif args.preregister:
        preregister(cfg)
    elif args.evaluate:
        evaluate(cfg, rescore=args.rescore)
    elif args.figures:
        figures_only(cfg)
    else:
        development_step(cfg)


if __name__ == "__main__":
    main()
