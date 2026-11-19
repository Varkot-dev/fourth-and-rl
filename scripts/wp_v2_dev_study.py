"""M1, version 2: diagnosis and model comparison on DEVELOPMENT seasons only.

Nothing in this script reads the 2024 to 2025 test seasons or the 2026 holdout. Plays
come from `splits.load_dev_plays()` (2010 to 2023), and every score is season-blocked:
seven folds of two consecutive seasons, each predicted by a model fitted on the other
twelve. No fold ever splits a season, let alone a game.

Each candidate is one model design. For each the script records
  - log loss, Brier score and calibration error on all held-out plays,
  - the same on the last 5:00 and the last 2:00 of regulation,
  - a table of end-game cells (time left by score margin, team with the ball):
    predicted against observed win rate, with game-bootstrap intervals,
  - the memorization gap: held-out log loss minus log loss on the plays the model
    was fitted on,
  - the Platt slope and intercept of the pooled held-out scores (1 and 0 mean the
    raw model is already calibrated).

Output: results/m1_wp_v2_dev_study.json. Finished candidates are kept, so the script
can be rerun to add candidates; `--only NAME` reruns one; `--final` adds the checks
that need a fit on all development seasons for the named candidates.

Run:  python scripts/wp_v2_dev_study.py            (about a minute per candidate)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import polars as pl

from fourth_and_rl import splits, wp
from fourth_and_rl.io import write_json
from fourth_and_rl.paths import RESULTS

OUT = RESULTS / "m1_wp_v2_dev_study.json"
KEEP = [*wp.STATE_COLUMNS, wp.LABEL, "game_id", "play_id", "season"]
FOLD_SIZE = 2

V1_PARAMS = {
    "n_estimators": 300, "learning_rate": 0.06, "num_leaves": 15, "min_child_samples": 10000,
    "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 10.0,
    "max_bin": 127, "monotone_constraints_method": "intermediate",
}  # fmt: skip

F1 = tuple(wp.FEATURES_V1)
NO_HOME = tuple(f for f in F1 if f != "is_home")
KNEEL = (*NO_HOME, "kneel_slack")
SCORES = (*KNEEL, "possessions", "possessions_per_minute")
TIMEOUTS = (*KNEEL, "posteam_timeout_share", "defteam_timeout_share")


def spec(name: str, features: tuple[str, ...], baseline: bool, floor: float = 30.0,
         tau: float | None = None) -> wp.ModelSpec:  # fmt: skip
    return wp.ModelSpec(name, features, baseline, floor, tau)


def candidates() -> dict[str, dict]:
    """Every design that was tried, in the order it was tried. Nothing is removed."""
    c: dict[str, dict] = {}

    def add(name: str, sp: wp.ModelSpec, why: str, calibration: str = "none",
            n_calib: int = 0, **overrides: object) -> None:  # fmt: skip
        c[name] = {"spec": replace(sp, name=name), "params": {**V1_PARAMS, **overrides},
                   "calibration": calibration, "n_calib": n_calib, "why": why}  # fmt: skip

    # ---- stage 1: reproduce version 1 and take its parts away one at a time
    add("v1", wp.V1, "Version 1 as shipped: 10,000-play leaves, Platt map on two seasons.",
        calibration="platt", n_calib=2)  # fmt: skip
    add("v1_no_platt", wp.V1, "Version 1 without the Platt map: is the map the problem?")
    add("v1_leaf2000", wp.V1, "Version 1, smaller leaves only.", min_child_samples=2000)
    add("v1_leaf400", wp.V1, "Version 1, much smaller leaves only.", min_child_samples=400)
    # ---- stage 2: the structural baseline
    add("base", spec("", F1, True), "Version 1 trees on top of the normal baseline.")
    add("base_no_home", spec("", NO_HOME, True), "The same without is_home.")
    add("base_floor5", spec("", NO_HOME, True, 5.0),
        "Baseline, no is_home, 5-second floor in margin_per_sqrt_time.")  # fmt: skip
    add("base_sharp", spec("", NO_HOME, True, 30.0, 30.0),
        "Baseline with its clock offset fixed at 30 seconds instead of fitted (sharper "
        "at the end).")  # fmt: skip
    # ---- stage 3: the kneel-down rule
    add("kneel", spec("", KNEEL, True), "Baseline, no is_home, plus kneel_slack.")
    add("kneel_with_home", spec("", (*KNEEL, "is_home"), True),
        "As kneel but keeping is_home: what does dropping it cost?")  # fmt: skip
    add("kneel_no_base", spec("", KNEEL, False), "kneel_slack without the baseline.")
    # ---- stage 4: leaf size, with the baseline and the kneel feature in place
    for leaf in (5000, 2000, 1000, 500, 200):
        add(f"kneel_leaf{leaf}", spec("", KNEEL, True),
            f"As kneel with a minimum of {leaf} plays per leaf.",
            min_child_samples=leaf)  # fmt: skip
    # ---- stage 5: more end-game features
    add("scores_leaf2000", spec("", SCORES, True),
        "Adds possessions needed and possessions per minute.", min_child_samples=2000)  # fmt: skip
    add("timeouts_leaf2000", spec("", TIMEOUTS, True),
        "Adds the share of the half's clock each side's timeouts can stop.",
        min_child_samples=2000)  # fmt: skip
    add("kneel_floor5_leaf2000", spec("", KNEEL, True, 5.0),
        "As kneel_leaf2000 with a 5-second floor.", min_child_samples=2000)  # fmt: skip
    # ---- stage 6: two stages. General trees with large leaves, then end-game trees
    # with small leaves that see only the last minutes of regulation.
    def late(seconds: int, leaf: int, trees: int = 150, leaves: int = 15) -> dict:
        return {"seconds": seconds, "n_estimators": trees, "min_child_samples": leaf,
                "num_leaves": leaves}  # fmt: skip

    for seconds, leaf in ((300, 200), (300, 500), (600, 200), (600, 500), (900, 500)):
        add(f"two_stage_{seconds}s_leaf{leaf}", spec("", KNEEL, True),
            f"kneel (10,000-play leaves) plus end-game trees on the last {seconds} seconds "
            f"with {leaf}-play leaves.", endgame=late(seconds, leaf))  # fmt: skip
    # ---- stage 7: combinations around the best of stages 4 to 6
    add("scores_leaf1000", spec("", SCORES, True),
        "scores_leaf2000 with 1,000-play leaves.", min_child_samples=1000)  # fmt: skip
    add("scores_leaf5000", spec("", SCORES, True),
        "scores_leaf2000 with 5,000-play leaves.", min_child_samples=5000)  # fmt: skip
    add("scores_no_base_leaf2000", spec("", SCORES, False),
        "scores_leaf2000 without the baseline: what does the baseline add?",
        min_child_samples=2000)  # fmt: skip
    add("scores_home_leaf2000", spec("", (*SCORES, "is_home"), True),
        "scores_leaf2000 keeping is_home.", min_child_samples=2000)  # fmt: skip
    # ---- stage 8: the rules of the game in the baseline (after the all-season sanity
    # checks of scores_leaf2000 showed the baseline too soft at 0:00: its fitted tau of
    # about 460 seconds leaves 5 points of uncertainty with no time left)
    rules = replace(spec("", SCORES, True), baseline_kneel=True)
    both = replace(rules, baseline_possession=True)
    add("rules_kneel_leaf2000", rules,
        "scores_leaf2000 with a baseline that is certain where the leader can kneel out.",
        min_child_samples=2000)  # fmt: skip
    add("rules_both_leaf2000", both,
        "As rules_kneel_leaf2000, plus a fitted possession value in the baseline.",
        min_child_samples=2000)  # fmt: skip
    add("rules_both_leaf1000", both, "rules_both_leaf2000 with 1,000-play leaves.",
        min_child_samples=1000)  # fmt: skip
    add("rules_poss_leaf2000", replace(spec("", SCORES, True), baseline_possession=True),
        "Possession value in the baseline, no kneel rule.", min_child_samples=2000)  # fmt: skip
    # ---- stage 9: a sharp baseline (tau fixed) with the rules, after stage 8 left the
    # 13-second and two-score checks failing because the fitted tau stays near 360 to 520
    for tau, leaf in ((60.0, 2000), (120.0, 2000), (120.0, 1000), (240.0, 2000)):
        add(f"rules_tau{int(tau)}_leaf{leaf}", replace(both, baseline_tau=tau),
            f"rules_both with the baseline clock offset fixed at {int(tau)} seconds, "
            f"{leaf}-play leaves.", min_child_samples=leaf)  # fmt: skip
    add("scores_two_stage_600s", spec("", SCORES, True),
        "scores_leaf2000 plus end-game trees on the last 600 seconds, 500-play leaves.",
        min_child_samples=2000, endgame=late(600, 500))  # fmt: skip
    return c


def folds(seasons: list[int]) -> list[dict[str, list[int]]]:
    return wp.crossfit_plan(seasons, FOLD_SIZE, 0)


def run_candidate(plays: pl.DataFrame, cand: dict, seed: int, n_threads: int) -> dict:
    """Season-blocked cross-validation of one design. Returns metrics and predictions."""
    usable = plays.filter(pl.Series(wp.scoreable(plays)) & pl.col(wp.LABEL).is_not_null())
    season = usable["season"].to_numpy()
    y = usable[wp.LABEL].to_numpy()
    pred, raw = np.full(usable.height, np.nan), np.full(usable.height, np.nan)
    in_sample, held_out, seconds, baselines, platt = [], [], [], [], []
    for fold in folds(sorted(set(season.tolist()))):
        rest = fold["train"]
        n_calib = cand["n_calib"]
        train = wp.season_filter(usable, rest[: len(rest) - n_calib])
        calib = wp.season_filter(usable, rest[len(rest) - n_calib :]) if n_calib else None
        start = time.perf_counter()
        model = wp.WPModel(cand["params"], cand["calibration"], n_threads, cand["spec"]).fit(
            train, calib, seed=seed, what=f"development study, fold {fold['predict']}"
        )
        seconds.append(time.perf_counter() - start)
        mask = np.isin(season, fold["predict"])
        held = usable.filter(pl.Series(mask))
        splits.assert_can_score(model.stamp, held, "development study, held-out fold")
        raw[mask] = model.raw_score(held)
        pred[mask] = model.predict(held)
        sample = train.sample(min(train.height, 150_000), seed=seed)
        in_sample.append(wp.log_loss(sample[wp.LABEL].to_numpy(), model.predict(sample)))
        held_out.append(wp.log_loss(y[mask], pred[mask]))
        if model.baseline:
            baselines.append(model.baseline)
        if cand["calibration"] == "platt":
            platt.append(model.calibrator.describe())
    late5 = wp.endgame_rows(usable).to_numpy()
    late2 = late5 & (usable["game_seconds_remaining"].to_numpy() <= 120)
    pooled = wp.Calibrator("platt").fit(raw, y)
    # The honest version of a pooled map: each season gets a map fitted on the
    # cross-fitted scores of the other seasons only.
    loso = np.empty_like(raw)
    for s_ in sorted(set(season.tolist())):
        cal = wp.Calibrator("platt").fit(raw[season != s_], y[season != s_])
        loso[season == s_] = cal.transform(raw[season == s_])
    loso_table = wp.endgame_slice_table(usable, loso, 1000, seed)
    table = wp.endgame_slice_table(usable, pred, 1000, seed)
    first = wp.endgame_slice_table(usable, pred, 1000, seed, first_down_only=True)
    out = {
        "why": cand["why"],
        "features": list(cand["spec"].features),
        "baseline": cand["spec"].baseline,
        "time_floor": cand["spec"].time_floor,
        "params": cand["params"],
        "calibration": cand["calibration"],
        "overall": wp.all_metrics(y, pred),
        "last_5_minutes": {"rows": int(late5.sum()), **wp.all_metrics(y[late5], pred[late5])},
        "last_2_minutes": {"rows": int(late2.sum()), **wp.all_metrics(y[late2], pred[late2])},
        "endgame_summary": wp.endgame_summary(table),
        "endgame_summary_first_downs": wp.endgame_summary(first),
        "endgame_cells": table,
        "endgame_cells_first_downs": first,
        "memorization": {
            "in_sample_log_loss": float(np.mean(in_sample)),
            "held_out_log_loss": float(np.mean(held_out)),
            "gap": float(np.mean(held_out) - np.mean(in_sample)),
        },
        "pooled_platt_on_held_out_scores": {
            "slope": pooled.a, "intercept": pooled.b,
            "log_loss_after": wp.log_loss(y, pooled.transform(raw)),
            "ece_after": wp.ece(y, pooled.transform(raw)),
            "note": "Fitted on the same held-out scores it is scored on (two numbers).",
        },  # fmt: skip
        "pooled_platt_leave_one_season_out": {
            "overall": wp.all_metrics(y, loso),
            "last_5_minutes": wp.all_metrics(y[late5], loso[late5]),
            "last_2_minutes": wp.all_metrics(y[late2], loso[late2]),
            "endgame_summary": wp.endgame_summary(loso_table),
            "note": "Each season mapped by a Platt map fitted on the other seasons' "
            "cross-fitted scores. Applied on top of this candidate's own calibration.",
        },
        "fold_log_loss": [float(v) for v in held_out],
        "fit_seconds_mean": float(np.mean(seconds)),
    }
    if baselines:
        out["baseline_parameters_by_fold"] = baselines
    if platt:
        out["platt_by_fold"] = platt
    return out, pred, usable


def paired(a: dict, b: dict, usable: pl.DataFrame, n_boot: int, seed: int) -> dict:
    """Game-bootstrap difference a minus b, overall and in the last 5:00 and 2:00."""
    y, games = usable[wp.LABEL].to_numpy(), usable["game_id"].to_numpy()
    late5 = wp.endgame_rows(usable).to_numpy()
    late2 = late5 & (usable["game_seconds_remaining"].to_numpy() <= 120)
    out = {}
    for name, m in (("overall", np.ones(len(y), bool)), ("last_5_minutes", late5),
                    ("last_2_minutes", late2)):  # fmt: skip
        out[name] = wp.game_bootstrap_diff(y[m], a[m], b[m], games[m], n_boot, seed)
    return out


def smoothness(model: wp.WPModel) -> dict:
    """How step-like is the model along the field, the spread and the clock?

    For sweeps of one input with the rest fixed: the share of adjacent steps where the
    prediction does not move at all, and the largest single step.
    """
    out = {}
    sweeps = {
        "yardline_1_to_99": dict(yardline_100=np.arange(1.0, 100.0),
                                 ydstogo=np.minimum(10.0, np.arange(1.0, 100.0))),
        "spread_minus14_to_14": dict(posteam_spread=np.arange(-14.0, 14.5, 0.5)),
        "clock_last_300_seconds_up_3": dict(game_seconds_remaining=np.arange(0.0, 301.0),
                                            score_differential=3.0),
    }  # fmt: skip
    for name, kwargs in sweeps.items():
        flat, big = [], []
        for secs in ((1800.0, 900.0, 300.0) if "clock" not in name else (None,)):
            kw = dict(kwargs) if secs is None else {**kwargs, "game_seconds_remaining": secs}
            step = np.abs(np.diff(model.predict(wp.state_frame(**kw))))
            flat.append(float(np.mean(step < 1e-9)))
            big.append(float(step.max()))
        out[name] = {"share_of_flat_steps": float(np.mean(flat)), "largest_step": max(big)}
    return out


def final_checks(plays: pl.DataFrame, cand: dict, seed: int, n_threads: int) -> dict:
    """Checks that need one fit on all development seasons: sanity, smoothness, speed."""
    train = plays if cand["n_calib"] == 0 else plays.filter(pl.col("season") <= 2021)
    calib = None if cand["n_calib"] == 0 else plays.filter(pl.col("season") >= 2022)
    start = time.perf_counter()
    model = wp.WPModel(cand["params"], cand["calibration"], n_threads, cand["spec"]).fit(
        train, calib, seed=seed, what="development study, all development seasons"
    )
    seconds = time.perf_counter() - start
    return {
        "fit_seconds_all_development_seasons": round(seconds, 1),
        "baseline": model.baseline,
        "sanity_checks": wp.sanity_checks(model),
        "smoothness": smoothness(model),
        "feature_importance_gain_share": model.feature_importance(),
        "trees_using_timeouts": tree_usage(model),
    }


def tree_usage(model: wp.WPModel) -> dict:
    """How many splits use each timeout and clock-rule feature."""
    names = model.spec.features
    counts = model.booster.feature_importance(importance_type="split")
    total = int(counts.sum())
    keys = [f for f in names if "timeout" in f or f == "kneel_slack"]
    return {"total_splits": total, **{k: int(counts[names.index(k)]) for k in keys}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="rerun these candidates")
    parser.add_argument("--final", nargs="*", default=None,
                        help="add all-development-season checks for these candidates")  # fmt: skip
    parser.add_argument("--compare", nargs=2, action="append", default=[],
                        metavar=("A", "B"), help="game-bootstrap difference A minus B")  # fmt: skip
    args = parser.parse_args()
    cfg = wp.load_config()
    seed, n_threads = cfg["seed"], cfg["n_threads"]
    plays = splits.load_dev_plays().select(KEEP)
    seasons = sorted(plays["season"].unique().to_list())
    assert not set(seasons) & (set(splits.test_seasons()) | set(splits.holdout_seasons()))

    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    done.setdefault("candidates", {})
    done.setdefault("final_checks", {})
    done.setdefault("paired_differences", {})
    done.update(
        seasons=seasons, seed=seed,
        folds=[{"predict": f["predict"], "train": f["train"]} for f in folds(seasons)],
        note="Development seasons only (splits.load_dev_plays). Season-blocked "
        "cross-validation, folds of two consecutive seasons. No test or holdout row is "
        "read by this script.",
    )  # fmt: skip
    cands = candidates()
    cache: dict[str, np.ndarray] = {}
    usable = None
    todo = [n for n in cands if (args.only is None and n not in done["candidates"])
            or (args.only is not None and n in args.only)]  # fmt: skip
    need_pred = {n for pair in args.compare for n in pair}
    for name in dict.fromkeys([*todo, *sorted(need_pred)]):
        start = time.perf_counter()
        result, pred, usable = run_candidate(plays, cands[name], seed, n_threads)
        cache[name] = pred
        done["candidates"][name] = result
        o, e = result["overall"], result["endgame_summary"]
        print(f"{name:24s} ll {o['log_loss']:.5f} ece {o['ece']:.4f} | last2 "
              f"{result['last_2_minutes']['log_loss']:.4f} | cell gap {e['mean_abs_gap']:.4f} "
              f"| mem gap {result['memorization']['gap']:.4f} | "
              f"{time.perf_counter() - start:.0f}s", flush=True)  # fmt: skip
        write_json(done, OUT)
    for a, b in args.compare:
        done["paired_differences"][f"{a}_minus_{b}"] = paired(
            cache[a], cache[b], usable, cfg["metrics"]["bootstrap_reps"], seed
        )
    for name in args.final or []:
        done["final_checks"][name] = final_checks(plays, cands[name], seed, n_threads)
        s = done["final_checks"][name]
        print(name, "sanity all passed:", s["sanity_checks"]["all_passed"],
              "fit seconds:", s["fit_seconds_all_development_seasons"])  # fmt: skip
    write_json(done, OUT)


if __name__ == "__main__":
    main()
