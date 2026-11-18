"""M7 coach evaluation: every real fourth down of 2011 to 2025 graded by models that never
saw its season, the league trend, team and coach tables with intervals and shrinkage, the
coach cost ledger, the malpractice check (part one) and the Lions at 49ers case study.

Run `scripts/m7_penalty_ledger.py` first: the ledger joins its priced penalties.

Stages (resumable; fold models and bootstrap replicates are cached under
data/models/coaches/):
  1. grade    fold models, point values and the fold bootstraps; writes
              data/processed/m7_graded_fourth_downs.parquet
  2. analyze  tables, ledger, malpractice cases, case study and figures; writes
              results/m7_coach_evaluation.json, results/m7_coach_ledger.csv and
              assets/m7_*.png

Usage: python scripts/m7_coach_evaluation.py [--stage grade|analyze|all]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_var, "3")

import argparse  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from fourth_and_rl import bootstrap, coaches, decision, splits  # noqa: E402
from fourth_and_rl import wp as wp_mod  # noqa: E402
from fourth_and_rl.io import read_json, write_json  # noqa: E402
from fourth_and_rl.paths import ASSETS, DATA, RESULTS, ROOT  # noqa: E402

TEST_PURPOSE = (
    "M7 coach evaluation: grade the 2024-2025 fourth downs with the headline development "
    "models, the malpractice cases and the case-study tables"
)
GRADED_PATH = DATA / "processed" / "m7_graded_fourth_downs.parquet"
PRICED_PATH = DATA / "processed" / "m7_penalties_priced.parquet"
M3_DECISIONS = DATA / "processed" / "fourth_down_decisions.parquet"
ID_COLUMNS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
              "desc", "order_sequence"]  # fmt: skip


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fourth_downs(plays: pl.DataFrame, seasons: tuple[int, ...]) -> pl.DataFrame:
    """Real fourth-down decisions (the play ran: go, field goal or punt) in regulation."""
    return plays.filter(
        (pl.col("down") == 4) & pl.col("season").is_in(list(seasons))
        & pl.col("fourth_down_decision").is_not_null() & (pl.col("qtr") <= 4)
    ).sort("game_id", "order_sequence", "play_id")  # fmt: skip


def frame_from(states: pl.DataFrame, values: dict[str, np.ndarray], table: np.ndarray,
               p_spread: np.ndarray, fold: coaches.Fold) -> pl.DataFrame:  # fmt: skip
    g = coaches.grade(table, states["fourth_down_decision"].to_list())
    nn = lambda x: pl.Series(np.asarray(x, dtype=float)).fill_nan(None)  # noqa: E731
    names = np.array([*coaches.ACTIONS, None], dtype=object)
    keep = [c for c in ID_COLUMNS if c in states.columns] + wp_mod.STATE_COLUMNS
    return states.select(*keep, pl.col("fourth_down_decision").alias("actual")).with_columns(
        nn(table[:, 0]).alias("go_wp"), nn(values["go_wp"]).alias("go_wp_even_spread"),
        nn(table[:, 1]).alias("fg_wp"), nn(table[:, 2]).alias("punt_wp"),
        nn(values["p_convert"]).alias("p_convert_even_spread"),
        nn(p_spread).alias("p_convert"),
        pl.Series("best_action", names[np.where(g["best"] < 0, 3, g["best"])].tolist(),
                  dtype=pl.String),
        nn(g["best_wp"]).alias("best_wp"), nn(g["margin"]).alias("margin"),
        nn(g["actual_wp"]).alias("actual_wp"), nn(g["wp_lost"]).alias("wp_lost"),
        pl.lit(fold.name).alias("fold"),
        pl.lit("headline development models" if fold.headline else
               f"fold models fitted on {fold.fit[0]}-{fold.fit[-1]} without "
               f"{fold.grade[0]}-{fold.grade[-1]}").alias("graded_by"),
    )  # fmt: skip


def grade_dev_fold(fold: coaches.Fold, dev: pl.DataFrame, cfg: dict, dcfg: dict, wcfg: dict
                   ) -> tuple[pl.DataFrame, dict[str, Any]]:  # fmt: skip
    """Point values with the fold models and categories from the fold bootstrap."""
    directory = ROOT / cfg["folds"]["models_dir"]
    n_threads = int(cfg["n_threads"])
    t0 = time.perf_counter()
    model, comps = coaches.ensure_fold_models(fold, dev, directory, wcfg, None, n_threads)
    engine = decision.DecisionEngine(model, comps, dcfg["engine"])
    states = fourth_downs(dev, fold.grade)
    engine.assert_can_score(states, f"M7 {fold.name}")
    values = engine.action_values(states)
    p_spread = coaches.p_convert_at_spread(comps, states)
    table = coaches.value_table(values, p_spread)
    out = frame_from(states, values, table, p_spread, fold)
    log(f"{fold.name}: {states.height} fourth downs valued ({time.perf_counter() - t0:.0f}s)")

    # The cheaper bootstrap: whole games of the fit seasons only, both models refitted.
    n_boot = int(cfg["bootstrap"]["n_boot"])
    fit_rows = dev.filter(pl.col("season").is_in(list(fold.fit)))
    prepared = bootstrap.prepare_plays(fit_rows)
    del fit_rows
    plan = coaches.fold_bootstrap_plan(prepared, int(cfg["seed"]))
    rep_dir = directory / f"boot_{fold.name}"
    boot = np.empty((n_boot, states.height, 3), dtype=np.float32)
    tag = f"{fold.name}_{bootstrap._states_tag(states, dcfg['engine'])}"
    for b in range(n_boot):
        t1 = time.perf_counter()
        fitted = bootstrap.ensure_replicate(plan, b, rep_dir, wcfg, None, [], n_threads, "none")
        rep = bootstrap.load_replicate_engine(rep_dir, b, dcfg["engine"])
        rep.assert_can_score(states, f"M7 {fold.name} replicate {b}")
        boot[b] = coaches.fold_replicate_values(states, rep_dir, b, dcfg["engine"], tag)
        if fitted:
            log(f"{fold.name}: replicate {b + 1}/{n_boot} ({time.perf_counter() - t1:.0f}s)")
    del prepared
    th = cfg["bootstrap"]["thresholds"]
    s = bootstrap.summarize(table, boot, th["confident"], th["lean"])
    nn = lambda x: pl.Series(np.asarray(x, dtype=float)).fill_nan(None)  # noqa: E731
    out = out.with_columns(
        nn(s["boot_pct"]).alias("boot_pct"),
        pl.Series("category", s["category"].tolist(), dtype=pl.String),
        nn(s["share_go"]).alias("boot_share_go"),
        nn(s["share_field_goal"]).alias("boot_share_field_goal"),
        nn(s["share_punt"]).alias("boot_share_punt"),
        nn(s["gain_lo"]).alias("go_gain_lo"), nn(s["gain_hi"]).alias("go_gain_hi"),
        pl.lit(n_boot).alias("n_boot"), pl.lit("fold bootstrap").alias("category_source"),
    )  # fmt: skip
    split = bootstrap.split_half_agreement(table, boot, th["confident"], th["lean"])
    diag = {"fold": fold.name, "graded": list(fold.grade), "fit": list(fold.fit),
            "fourth_downs": states.height, "n_boot": n_boot,
            "bootstrap_games": {k: v.n_games for k, v in plan.blocks.items()},
            "category_shares": bootstrap.category_shares(s["category"], s["best"]),
            "split_half": split, "wp_stamp_seasons": model.stamp["seasons"],
            "components_stamp_seasons": comps.stamp["seasons"]}  # fmt: skip
    return out, diag


def grade_test_fold(fold: coaches.Fold, plays: pl.DataFrame, test: pl.DataFrame, cfg: dict,
                    dcfg: dict) -> tuple[pl.DataFrame, dict[str, Any]]:  # fmt: skip
    """Test seasons: the headline engine, and the M3 categories (101 replicates)."""
    engine = decision.load_engine(plays, dcfg)
    states = fourth_downs(test, fold.grade)
    engine.assert_can_score(states, "M7 test seasons")
    values = engine.action_values(states)
    p_spread = coaches.p_convert_at_spread(engine.components, states)
    table = coaches.value_table(values, p_spread)
    out = frame_from(states, values, table, p_spread, fold)
    m3 = pl.read_parquet(M3_DECISIONS).select(
        "game_id", "play_id", pl.col("best_action").alias("m3_best_action"),
        pl.col("go_wp").alias("m3_go_wp"), pl.col("fg_wp").alias("m3_fg_wp"),
        pl.col("punt_wp").alias("m3_punt_wp"), "boot_share_go", "boot_share_field_goal",
        "boot_share_punt", "go_gain_lo", "go_gain_hi", "n_boot",
    )  # fmt: skip
    out = out.join(m3, on=["game_id", "play_id"], how="left")
    if out.height != states.height:
        raise ValueError("the M3 decisions file is not unique per play")
    share = {a: out[f"boot_share_{a}"].cast(pl.Float64).fill_null(np.nan).to_numpy()
             for a in coaches.ACTIONS}  # fmt: skip
    best = out["best_action"].to_list()
    pct = np.array([share[b][i] if b is not None else np.nan for i, b in enumerate(best)])
    th = cfg["bootstrap"]["thresholds"]
    cat = bootstrap.classify(pct, th["confident"], th["lean"])
    changed = (out["best_action"] != out["m3_best_action"]).fill_null(False).to_numpy()
    nn = lambda x: pl.Series(np.asarray(x, dtype=float)).fill_nan(None)  # noqa: E731
    out = out.with_columns(
        nn(pct).alias("boot_pct"), pl.Series("category", cat.tolist(), dtype=pl.String),
        pl.Series("category_source", np.where(
            changed, "M3 replicate shares of the spread-adjusted best action",
            "M3 category").tolist()),
    )  # fmt: skip
    f = lambda c: out[c].cast(pl.Float64).fill_null(np.nan).to_numpy()  # noqa: E731
    even = np.column_stack([f("go_wp_even_spread"), f("fg_wp"), f("punt_wp")])
    m3t = np.column_stack([f("m3_go_wp"), f("m3_fg_wp"), f("m3_punt_wp")])
    diff = np.abs(even - m3t)
    diag = {"fold": fold.name, "graded": list(fold.grade), "fourth_downs": states.height,
            "matched_to_m3": int(np.isfinite(m3t[:, 0]).sum()),
            "max_abs_difference_from_m3_values": float(np.nanmax(diff)),
            "best_action_changed_by_the_spread": int(changed.sum()),
            "category_shares": bootstrap.category_shares(cat)}  # fmt: skip
    out = out.drop("m3_best_action", "m3_go_wp", "m3_fg_wp", "m3_punt_wp")
    return out, diag


def headline_in_sample(plays: pl.DataFrame, dcfg: dict, cfg: dict) -> pl.DataFrame:
    """Diagnostic only: the development seasons graded by the headline models, which were
    fitted on them (in sample). Compared with the fold grades it shows how much of a
    season-to-season change comes from which fold's models graded the season."""
    engine = decision.load_engine(plays, dcfg)
    states = fourth_downs(plays, tuple(s for s in splits.dev_seasons()
                                       if s >= int(cfg["grade_seasons"][0])))
    values = engine.action_values(states)
    table = coaches.value_table(values, coaches.p_convert_at_spread(engine.components, states))
    g = coaches.grade(table, states["fourth_down_decision"].to_list())
    names = np.array([*coaches.ACTIONS, None], dtype=object)
    return states.select("game_id", "play_id").with_columns(
        pl.Series("in_sample_best_action", names[np.where(g["best"] < 0, 3, g["best"])].tolist(),
                  dtype=pl.String),
        pl.Series("in_sample_wp_lost", g["wp_lost"]).fill_nan(None),
    )  # fmt: skip


def stage_grade(cfg: dict) -> None:
    dcfg, wcfg = decision.load_config(), wp_mod.load_config()
    lo, hi = cfg["grade_seasons"]
    plan = coaches.fold_plan(splits.dev_seasons(), splits.test_seasons(), range(lo, hi + 1),
                             int(cfg["folds"]["fold_size"]))  # fmt: skip
    coaches.check_fold_plan(plan, splits.test_seasons(), splits.holdout_seasons())
    dev = splits.load_dev_plays()
    parts, diags = [], []
    for fold in plan:
        if fold.headline:
            continue
        part, diag = grade_dev_fold(fold, dev, cfg, dcfg, wcfg)
        parts.append(part)
        diags.append(diag)
    test = splits.load_test_plays(TEST_PURPOSE)
    plays = pl.concat([dev, test], how="diagonal_relaxed")
    del dev
    for fold in plan:
        if fold.headline:
            part, diag = grade_test_fold(fold, plays, test, cfg, dcfg)
            parts.append(part)
            diags.append(diag)
    graded = pl.concat(parts, how="diagonal_relaxed").sort("season", "game_id",
                                                            "order_sequence")  # fmt: skip
    graded = graded.join(headline_in_sample(plays, dcfg, cfg), on=["game_id", "play_id"],
                         how="left")  # fmt: skip
    GRADED_PATH.parent.mkdir(parents=True, exist_ok=True)
    graded.write_parquet(GRADED_PATH)
    write_json({"plan": [{"name": f.name, "grade": list(f.grade), "fit": list(f.fit),
                          "headline": f.headline} for f in plan], "folds": diags},
               ROOT / cfg["folds"]["models_dir"] / "grade_diagnostics.json")  # fmt: skip
    log(f"graded {graded.height} fourth downs -> {GRADED_PATH}")




# ======================================================================================
# Stage 2: analysis
# ======================================================================================

HISTORIC = {"SD": "LAC", "STL": "LA", "OAK": "LV"}  # schedule codes -> play-by-play codes


def with_flags(graded: pl.DataFrame) -> pl.DataFrame:
    """Per-row quantities that the tables sum. Win probability is kept as a fraction, so a
    sum of fractions is a number of wins."""
    ok = pl.col("wp_lost").is_not_null()
    conf = (pl.col("category") == "confident").fill_null(False)
    lost = pl.col("wp_lost").fill_null(0.0)
    went = pl.col("actual") == "go"
    best_go = pl.col("best_action") == "go"
    return graded.with_columns(
        ok.cast(pl.Float64).alias("n_dec"),
        pl.when(ok & conf).then(lost).otherwise(0.0).alias("lost_conf"),
        pl.when(ok).then(lost).otherwise(0.0).alias("lost_total"),
        pl.when(ok & conf & best_go & ~went).then(lost).otherwise(0.0).alias("lost_conf_timid"),
        pl.when(ok & conf & ~best_go & went).then(lost).otherwise(0.0).alias("lost_conf_bold"),
        (ok & conf & (pl.col("best_action") != pl.col("actual"))).cast(pl.Float64)
        .alias("n_conf_mistakes"),
        (ok & went).cast(pl.Float64).alias("went"),
        (ok & best_go).cast(pl.Float64).alias("engine_go"),
        (ok & conf & best_go).cast(pl.Float64).alias("go_conf"),
        (ok & conf & best_go & went).cast(pl.Float64).alias("went_go_conf"),
        (ok & conf).cast(pl.Float64).alias("n_conf"),
    )  # fmt: skip


SUMS = ["n_dec", "lost_conf", "lost_total", "lost_conf_timid", "lost_conf_bold",
        "n_conf_mistakes", "went", "engine_go", "go_conf", "went_go_conf", "n_conf"]  # fmt: skip


def sides_table(graded: pl.DataFrame, lookup: pl.DataFrame) -> pl.DataFrame:
    """One row per team-game of the graded seasons (zero-decision team-games included),
    with the head coach, the team (play-by-play codes) and the summed quantities."""
    games = graded.select("game_id", "season").unique()
    sides = lookup.join(games, on="game_id", how="inner").select(
        "game_id", "season", pl.col("is_home_s").cast(pl.Int8).alias("is_home"), "coach",
        pl.col("team").replace(HISTORIC).alias("team"),
    )  # fmt: skip
    sums = graded.with_columns(pl.col("is_home").cast(pl.Int8)).group_by("game_id", "is_home").agg(
        [pl.col(c).sum() for c in SUMS])  # fmt: skip
    out = sides.join(sums, on=["game_id", "is_home"], how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in SUMS]).with_columns(pl.lit(1.0).alias("team_games"))
    # Era adjustment: cost relative to the league in the same season, per team-game and
    # per graded fourth down, so a coach is compared with his own seasons' league.
    out = out.with_columns(
        (pl.col("lost_conf") - pl.col("lost_conf").mean().over("season")).alias("excess_game"),
        (pl.col("lost_conf") - pl.col("n_dec") * pl.col("lost_conf").sum().over("season")
         / pl.col("n_dec").sum().over("season")).alias("excess_opp"),
    )  # fmt: skip
    return out.sort("season", "game_id", "is_home")


def summarize_groups(sides: pl.DataFrame, group: str, cfg: dict, seed: int) -> list[dict]:
    """Per group: sums, rates with 90% game-bootstrap intervals."""
    t = cfg["tables"]
    iv, per17 = t["interval"], float(t["games_per_season"])
    names = SUMS + ["team_games", "excess_game", "excess_opp"]
    boot = coaches.group_bootstrap(sides[group].to_numpy(), sides["game_id"].to_numpy(),
                                   {c: sides[c].to_numpy() for c in names}, int(t["n_boot"]),
                                   seed)  # fmt: skip
    rows = []
    for key, b in boot.items():
        p, bb = b["point"], b["boot"]

        def r(num: str, den: str, scale: float = 1.0, p: dict = p, bb: dict = bb) -> dict:
            out = coaches.ratio_summary(p[num], p[den], bb[num], bb[den], iv)
            return {k: v * scale for k, v in out.items()}

        rows.append({
            group: key, "games": int(p["team_games"]), "fourth_downs": int(p["n_dec"]),
            "went": int(p["went"]), "confident_go_calls": int(p["go_conf"]),
            "confident_mistakes": int(p["n_conf_mistakes"]),
            "wins_lost_confident": r("lost_conf", "team_games", p["team_games"]),
            "wins_lost_total": r("lost_total", "team_games", p["team_games"]),
            "wins_lost_confident_timid": p["lost_conf_timid"],
            "wins_lost_confident_bold": p["lost_conf_bold"],
            "wins_lost_confident_per_17": r("lost_conf", "team_games", per17),
            "wins_lost_total_per_17": r("lost_total", "team_games", per17),
            "wp_points_lost_confident_per_game": r("lost_conf", "team_games", 100.0),
            "wp_points_lost_total_per_game": r("lost_total", "team_games", 100.0),
            "wp_points_lost_confident_per_opportunity": r("lost_conf", "n_dec", 100.0),
            "wins_above_season_average_per_17": r("excess_game", "team_games", per17),
            "wins_above_season_average": r("excess_game", "team_games", p["team_games"]),
            "wp_points_above_season_average_per_opportunity": r("excess_opp", "n_dec", 100.0),
            "go_rate": r("went", "n_dec"),
            "go_rate_when_engine_confidently_says_go": r("went_go_conf", "go_conf"),
            "engine_go_share": p["engine_go"] / p["n_dec"] if p["n_dec"] else float("nan"),
            "confident_share": p["n_conf"] / p["n_dec"] if p["n_dec"] else float("nan"),
        })  # fmt: skip
    return rows


def shrink(rows: list[dict], key: str, min_games: int) -> dict[str, Any]:
    """Empirical-Bayes shrinkage of `key` (an interval dict) over rows with enough games,
    and how many differences can be told from noise."""
    use = [r for r in rows if r["games"] >= min_games]
    x = np.array([r[key]["est"] for r in use])
    se = np.array([r[key]["se"] for r in use])
    eb = coaches.dersimonian_laird(x, se**2)
    z = 1.6449
    for r, s, sd, b in zip(use, eb["shrunk"], eb["shrunk_sd"], eb["B"], strict=True):
        r[f"{key}_shrunk"] = {"est": float(s), "lo": float(s - z * sd), "hi": float(s + z * sd),
                              "weight_on_own_data": float(b)}
    mean = eb["mean"]
    outside_raw = int(sum((r[key]["lo"] > mean) or (r[key]["hi"] < mean) for r in use))
    outside_shrunk = int(sum((r[f"{key}_shrunk"]["lo"] > mean) or (r[f"{key}_shrunk"]["hi"] < mean)
                             for r in use))  # fmt: skip
    k = len(use)
    i, j = np.triu_indices(k, 1)
    pair_z = np.abs(x[i] - x[j]) / np.sqrt(se[i] ** 2 + se[j] ** 2)
    return {
        "n": k, "league_mean": mean, "tau": float(np.sqrt(eb["tau2"])),
        "mean_standard_error": float(np.sqrt(np.mean(se**2))),
        "reliability": eb["reliability"],
        "raw_interval_excludes_mean": outside_raw,
        "expected_by_chance_at_90pct": 0.10 * k,
        "shrunk_interval_excludes_mean": outside_shrunk,
        "pairs": int(i.size), "pairs_different_at_90pct": int(np.sum(pair_z > z)),
        "pairs_expected_different_by_chance": 0.10 * float(i.size),
    }  # fmt: skip


def coach_seasons_correlation(graded: pl.DataFrame, lookup: pl.DataFrame, n_boot: int,
                              seed: int) -> dict[str, Any]:  # fmt: skip
    """Year-to-year correlation of a coach's per-opportunity cost (same coach, consecutive
    seasons), the plainest test of whether the ranking repeats."""
    s = sides_table(graded, lookup).group_by("coach", "season").agg(
        pl.col("lost_conf").sum(), pl.col("n_dec").sum(), pl.col("went").sum())
    s = s.filter(pl.col("n_dec") >= 30).with_columns(
        (pl.col("lost_conf") / pl.col("n_dec")).alias("cost"),
        (pl.col("went") / pl.col("n_dec")).alias("go_rate"))  # fmt: skip
    nxt = s.select("coach", (pl.col("season") - 1).alias("season"),
                   pl.col("cost").alias("cost_next"), pl.col("go_rate").alias("go_next"))
    pairs = s.join(nxt, on=["coach", "season"], how="inner")
    return {
        "pairs": pairs.height,
        "cost_per_opportunity": coaches.correlation(pairs["cost"].to_numpy(),
                                                    pairs["cost_next"].to_numpy(), n_boot, seed),
        "go_rate": coaches.correlation(pairs["go_rate"].to_numpy(), pairs["go_next"].to_numpy(),
                                       n_boot, seed),
        "rule": "coach-seasons with at least 30 graded fourth downs; same coach, next season",
    }  # fmt: skip


def exact_values(rows: pl.DataFrame) -> list[dict[str, Any] | None]:
    """Exact game solution (M4) for each fourth down: win probability of going, kicking
    and punting under optimal play for both identical teams."""
    from fourth_and_rl import mdp_game as mg

    try:
        sol = mg.load_solution()
    except (FileNotFoundError, ValueError) as err:
        log(f"exact solution not available: {err}")
        return [None] * rows.height
    out: list[dict[str, Any] | None] = []
    for r in rows.iter_rows(named=True):
        st = mg.GameState(int(r["game_seconds_remaining"]), int(r["score_differential"]), 4,
                          int(r["ydstogo"]), int(r["yardline_100"]),
                          int(r["posteam_timeouts_remaining"]),
                          int(r["defteam_timeouts_remaining"]), int(r["receive_2h_ko"]))
        q = sol.q_values(st)
        go = max((v for k, v in q.items() if k.startswith("play")), default=float("nan"))
        vals = {"go": go, "field_goal": q.get("field_goal", float("nan")),
                "punt": q.get("punt", float("nan"))}  # fmt: skip
        finite = {k: v for k, v in vals.items() if np.isfinite(v)}
        best = max(finite, key=finite.get) if finite else None
        act = r["actual"]
        lost = (finite[best] - vals[act]) if best and np.isfinite(vals.get(act, np.nan)) else None
        out.append({"go_wp": vals["go"], "fg_wp": vals["field_goal"], "punt_wp": vals["punt"],
                    "best_action": best, "wp_lost": lost})  # fmt: skip
    return out


def case_rows(graded: pl.DataFrame, cases: list[dict]) -> list[dict[str, Any]]:
    keys = pl.DataFrame([{"game_id": c["game_id"], "play_id": float(p), "case": c["key"]}
                         for c in cases for p in c["play_ids"]])  # fmt: skip
    rows = graded.join(keys, on=["game_id", "play_id"], how="inner")
    exact = exact_values(rows)
    meta = {c["key"]: c for c in cases}
    out = []
    pct = lambda v: None if v is None or not np.isfinite(v) else round(100 * float(v), 2)  # noqa: E731
    for r, ex in zip(rows.iter_rows(named=True), exact, strict=True):
        c = meta[r["case"]]
        out.append({
            "case": r["case"], "label": c["label"], "outrage": c["outrage"],
            "game_id": r["game_id"], "play_id": int(r["play_id"]), "season": r["season"],
            "team": r["posteam"],
            "situation": {"qtr": int(r["qtr"]), "game_seconds_remaining":
                          int(r["game_seconds_remaining"]), "score_differential":
                          int(r["score_differential"]), "ydstogo": int(r["ydstogo"]),
                          "yardline_100": int(r["yardline_100"]),
                          "timeouts": [int(r["posteam_timeouts_remaining"]),
                                       int(r["defteam_timeouts_remaining"])],
                          "posteam_spread": r["posteam_spread"]},
            "what_happened": r["desc"], "actual": r["actual"],
            "one_step": {"go": pct(r["go_wp"]), "field_goal": pct(r["fg_wp"]),
                         "punt": pct(r["punt_wp"]), "best": r["best_action"],
                         "p_convert": pct(r["p_convert"]), "margin": pct(r["margin"]),
                         "wp_points_given_up": pct(r["wp_lost"]),
                         "boot_pct": pct(r["boot_pct"]), "category": r["category"],
                         "n_boot": r["n_boot"], "graded_by": r["graded_by"]},
            "exact_game_solution": None if ex is None else {
                "go": pct(ex["go_wp"]), "field_goal": pct(ex["fg_wp"]),
                "punt": pct(ex["punt_wp"]), "best": ex["best_action"],
                "wp_points_given_up": pct(ex["wp_lost"])},
        })  # fmt: skip
    return out


def ledger_rows(coach_rows: list[dict], pen: dict, sides: pl.DataFrame, cfg: dict
                ) -> list[dict]:  # fmt: skip
    per17 = float(cfg["tables"]["games_per_season"])
    pen_by = {r["coach"]: r for r in pen["coaches"]}
    supported = bool(pen["verdict"]["assigning_penalties_to_the_coach_is_supported"])
    info = sides.group_by("coach").agg(
        pl.col("team").unique().sort().str.join("/").alias("teams"),
        pl.col("season").min().alias("first_season"), pl.col("season").max().alias("last_season"))
    info_by = {r["coach"]: r for r in info.to_dicts()}
    out = []
    for r in coach_rows:
        p = pen_by.get(r["coach"], {})
        d, t = r["wins_lost_confident"], r["wins_lost_total"]
        s17 = r.get("wins_lost_confident_per_17_shrunk", {})
        e17, e17s = r["wins_above_season_average_per_17"], r.get(
            "wins_above_season_average_per_17_shrunk", {})
        eop = r["wp_points_above_season_average_per_opportunity"]
        opp = r["wp_points_lost_confident_per_opportunity"]
        pw = p.get("cost_wins", 0.0)
        out.append({
            "coach": r["coach"], "teams": info_by[r["coach"]]["teams"],
            "first_season": info_by[r["coach"]]["first_season"],
            "last_season": info_by[r["coach"]]["last_season"], "games": r["games"],
            "ranked": r["games"] >= int(cfg["tables"]["min_games"]),
            "fourth_downs_graded": r["fourth_downs"],
            "go_rate": round(r["go_rate"]["est"], 4),
            "decision_wins_confident": d["est"], "decision_wins_confident_lo": d["lo"],
            "decision_wins_confident_hi": d["hi"],
            "decision_wins_total": t["est"], "decision_wins_total_lo": t["lo"],
            "decision_wins_total_hi": t["hi"],
            "decision_wins_confident_per_17": r["wins_lost_confident_per_17"]["est"],
            "decision_wins_confident_per_17_lo": r["wins_lost_confident_per_17"]["lo"],
            "decision_wins_confident_per_17_hi": r["wins_lost_confident_per_17"]["hi"],
            "decision_wins_confident_per_17_shrunk": s17.get("est"),
            "decision_wins_confident_per_17_shrunk_lo": s17.get("lo"),
            "decision_wins_confident_per_17_shrunk_hi": s17.get("hi"),
            "decision_wins_above_season_average": r["wins_above_season_average"]["est"],
            "decision_wins_above_season_average_per_17": e17["est"],
            "decision_wins_above_season_average_per_17_lo": e17["lo"],
            "decision_wins_above_season_average_per_17_hi": e17["hi"],
            "decision_wins_above_season_average_per_17_shrunk": e17s.get("est"),
            "decision_wins_above_season_average_per_17_shrunk_lo": e17s.get("lo"),
            "decision_wins_above_season_average_per_17_shrunk_hi": e17s.get("hi"),
            "decision_wp_points_above_season_average_per_opportunity": eop["est"],
            "decision_wp_points_above_season_average_per_opportunity_lo": eop["lo"],
            "decision_wp_points_above_season_average_per_opportunity_hi": eop["hi"],
            "decision_wp_points_per_opportunity": opp["est"],
            "decision_wp_points_per_opportunity_lo": opp["lo"],
            "decision_wp_points_per_opportunity_hi": opp["hi"],
            "share_of_confident_cost_from_not_going": (
                r["wins_lost_confident_timid"] / d["est"] if d["est"] > 0 else None),
            "penalty_fouls": p.get("fouls", 0), "penalty_wins": pw,
            "penalty_wins_lo": p.get("cost_wins_lo", 0.0),
            "penalty_wins_hi": p.get("cost_wins_hi", 0.0),
            "penalty_wins_per_17": pw * per17 / r["games"],
            "penalty_wins_attributable_to_coach_effect": p.get("cost_attributable_to_coach_wins"),
            "penalties_assigned_to_coach": supported,
            "ledger_wins_total": d["est"] + (pw if supported else 0.0),
        })  # fmt: skip
    return sorted(out, key=lambda x: -x["decision_wins_confident"])


def stage_analyze(cfg: dict) -> None:
    graded = with_flags(pl.read_parquet(GRADED_PATH))
    if not PRICED_PATH.exists() or not (RESULTS / "m7_penalty_ledger.json").exists():
        raise SystemExit("run scripts/m7_penalty_ledger.py first")
    pen = read_json(RESULTS / "m7_penalty_ledger.json")
    sched = pl.read_parquet(ROOT / cfg["penalties"]["cache_dir"] / "schedules.parquet")
    lookup = coaches.coach_lookup(sched)
    graded = coaches.attach_coaches(graded, lookup)
    sides = sides_table(graded, lookup)
    seed, t = int(cfg["seed"]), cfg["tables"]

    log("league trend")
    trend = sorted(summarize_groups(sides, "season", cfg, seed + 1), key=lambda r: r["season"])
    log("coaches and teams")
    coach_rows = summarize_groups(sides, "coach", cfg, seed + 2)
    noise = {
        "era_adjusted_per_17_games": shrink(coach_rows, "wins_above_season_average_per_17",
                                            int(t["min_games"])),
        "era_adjusted_per_opportunity": shrink(
            coach_rows, "wp_points_above_season_average_per_opportunity", int(t["min_games"])),
        "per_17_games": shrink(coach_rows, "wins_lost_confident_per_17", int(t["min_games"])),
        "per_opportunity": shrink(coach_rows, "wp_points_lost_confident_per_opportunity",
                                  int(t["min_games"])),
        "year_to_year": coach_seasons_correlation(graded, lookup, int(t["n_boot"]), seed),
    }  # fmt: skip
    w0, w1 = t["team_window"]
    team_rows = summarize_groups(sides.filter(pl.col("season").is_between(w0, w1)), "team",
                                 cfg, seed + 3)  # fmt: skip
    team_noise = shrink(team_rows, "wins_lost_confident_per_17", 1)
    team_noise_era = shrink(team_rows, "wins_above_season_average_per_17", 1)
    team_all = summarize_groups(sides, "team", cfg, seed + 4)

    led = ledger_rows(coach_rows, pen, sides, cfg)
    pl.DataFrame(led).write_csv(RESULTS / "m7_coach_ledger.csv", float_precision=5)

    log("malpractice cases and case study")
    mal = cfg["malpractice"]
    cases = case_rows(graded, mal["cases"])
    cs = cfg["case_study"]
    study = case_study(graded, cs, cfg)

    sens = graded.filter(pl.col("in_sample_best_action").is_not_null()).group_by("season").agg(
        (pl.col("best_action") == "go").mean().alias("engine_go_share_fold_models"),
        (pl.col("in_sample_best_action") == "go").mean().alias("engine_go_share_in_sample"),
        (100 * pl.col("wp_lost").mean()).alias("wp_points_lost_per_decision_fold_models"),
        (100 * pl.col("in_sample_wp_lost").mean()).alias("wp_points_lost_per_decision_in_sample"),
        (pl.col("best_action") == pl.col("in_sample_best_action")).mean().alias("same_best"),
    ).sort("season")  # fmt: skip
    overall = summarize_groups(sides.with_columns(pl.lit("all").alias("all")), "all", cfg,
                               seed + 5)[0]  # fmt: skip
    out = {
        "what": "Every real fourth-down decision of 2011 to 2025 graded by the one-step engine "
                "with models that never saw its season; win probability given up against the "
                "engine's best action, counting confident recommendations (and the total).",
        "grading": read_json(ROOT / cfg["folds"]["models_dir"] / "grade_diagnostics.json"),
        "units": "wins = win-probability fractions summed (100 WP points = 1 win); 'per 17' = "
                 "per 17 games coached; intervals are 90% bootstrap intervals over games",
        "overall": overall,
        "league_trend": trend,
        "fold_sensitivity": {
            "what": "Diagnostic: the development seasons also graded by the headline models, "
                    "which were fitted on them (in sample, never reported as a result). "
                    "Differences show how much a season's grade depends on which fold's "
                    "models graded it. All decisions, no bootstrap categories.",
            "by_season": sens.to_dicts()},
        "coaches": {"min_games_ranked": int(t["min_games"]), "noise": noise,
                    "table": sorted(coach_rows, key=lambda r: -r["wins_lost_confident"]["est"])},
        "teams": {"window": [w0, w1], "noise": team_noise, "noise_era_adjusted": team_noise_era,
                  "table": team_rows,
                  "all_seasons": team_all},
        "ledger": {"csv": "results/m7_coach_ledger.csv",
                   "penalties_assigned_to_coach":
                       pen["verdict"]["assigning_penalties_to_the_coach_is_supported"],
                   "assumption_statement": ASSUMPTION},
        "malpractice": {"cases": cases, "notes": mal["notes"],
                        "not_scoreable": mal["not_scoreable"],
                        "pending_2026_holdout": mal["pending_2026"]},
        "case_study": study,
    }  # fmt: skip
    write_json(out, RESULTS / "m7_coach_evaluation.json")
    log("figures")
    draw_all(out, led, pen, cfg)


ASSUMPTION = (
    "Penalties counted: false starts and delays of game by the offense, minus delays taken "
    "on purpose before a punt or field goal. Putting them on the head coach is the owner's "
    "assumption, tested in results/m7_penalty_ledger.json."
)


def case_study(graded: pl.DataFrame, cs: dict, cfg: dict) -> dict[str, Any]:
    """The Lions at 49ers NFC Championship: every Detroit fourth down with the numbers at
    that moment, and Detroit's win probability through the game from the model of the
    fold that never saw the 2023 season."""
    rows = graded.filter((pl.col("game_id") == cs["game_id"]) & (pl.col("posteam") == cs["team"]))
    season = int(rows["season"][0])
    lo, hi = cfg["grade_seasons"]
    plan = coaches.fold_plan(splits.dev_seasons(), splits.test_seasons(), range(lo, hi + 1),
                             int(cfg["folds"]["fold_size"]))  # fmt: skip
    fold = coaches.fold_for_season(plan, season)
    dev = splits.load_dev_plays()
    model, _ = coaches.ensure_fold_models(fold, dev, ROOT / cfg["folds"]["models_dir"],
                                          wp_mod.load_config(), None, int(cfg["n_threads"]))
    game = dev.filter((pl.col("game_id") == cs["game_id"]) & pl.col("posteam").is_not_null()
                      & (pl.col("qtr") <= 4)).sort("order_sequence")  # fmt: skip
    del dev
    splits.assert_can_score(model.stamp, game, "M7 case study path")
    p = model.predict(game)
    mine = (game["posteam"] == cs["team"]).to_numpy()
    path = np.where(mine, p, 1 - p)
    exact = exact_values(rows)
    downs = []
    for r, ex in zip(rows.iter_rows(named=True), exact, strict=True):
        downs.append({
            "play_id": int(r["play_id"]), "qtr": int(r["qtr"]),
            "clock": r["desc"].split(")")[0].strip("(") if r["desc"] else None,
            "elapsed": 3600 - int(r["game_seconds_remaining"]),
            "score_differential": int(r["score_differential"]), "ydstogo": int(r["ydstogo"]),
            "yardline_100": int(r["yardline_100"]), "actual": r["actual"],
            "go_wp": r["go_wp"], "fg_wp": r["fg_wp"], "punt_wp": r["punt_wp"],
            "p_convert": r["p_convert"], "best_action": r["best_action"],
            "wp_lost": r["wp_lost"], "boot_pct": r["boot_pct"], "category": r["category"],
            "go_gain_lo": r["go_gain_lo"], "go_gain_hi": r["go_gain_hi"], "desc": r["desc"],
            "exact": ex,
        })  # fmt: skip
    return {"game_id": cs["game_id"], "team": cs["team"], "season": season,
            "graded_by": rows["graded_by"][0], "n_boot": int(rows["n_boot"][0]),
            "final_score": "SF 34, DET 31", "fourth_downs": downs,
            "path": {"elapsed": (3600 - game["game_seconds_remaining"]).to_list(),
                     "wp": np.round(path, 4).tolist()}}  # fmt: skip


# ----------------------------------------------------------------------------- figures


def _band(ax: Any, x: np.ndarray, rows: list[dict], key: str, color: str, label: str,
          scale: float = 1.0, dashed: bool = False) -> None:  # fmt: skip
    est = np.array([r[key]["est"] for r in rows]) * scale
    lo = np.array([r[key]["lo"] for r in rows]) * scale
    hi = np.array([r[key]["hi"] for r in rows]) * scale
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, est, color=color, label=label, linestyle="--" if dashed else "-", marker="o",
            markersize=3.5)


def figure_trend(out: dict, path: Any) -> None:
    import matplotlib.pyplot as plt

    from fourth_and_rl import style

    style.apply()
    rows = out["league_trend"]
    x = np.array([r["season"] for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True)
    ax = axes[0]
    _band(ax, x, rows, "go_rate_when_engine_confidently_says_go", style.BLUE,
          "Coaches went, when the engine confidently said go", 100)
    _band(ax, x, rows, "go_rate", style.INK_2, "Coaches went, all fourth downs", 100)
    ax.plot(x, [100 * r["engine_go_share"] for r in rows], color=style.BLUE, linestyle=":",
            label="Engine says go, all fourth downs")
    ax.set_ylabel("Share of fourth downs (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=9.5)
    first, last = rows[0], rows[-1]
    style.title(ax, "Coaches go for it more, and give up less by not going",
                f"Real fourth downs {x[0]} to {x[-1]}, each graded by models that never saw its "
                f"season. Bands: 90% bootstrap intervals over games.")
    ax = axes[1]
    _band(ax, x, rows, "wp_points_lost_confident_per_game", style.ORANGE,
          "Confident recommendations only (the headline)")
    _band(ax, x, rows, "wp_points_lost_total_per_game", style.MUTED, "All graded decisions",
          dashed=True)
    ax.set_ylabel("Win probability given up\nper team-game (points)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9.5)
    for a in axes:
        a.axvspan(2023.5, 2025.5, color=style.GRID, alpha=0.6, zorder=0, linewidth=0)
    axes[1].text(2024.5, axes[1].get_ylim()[1] * 0.04, "test seasons:\nheadline models",
                 ha="center", va="bottom", fontsize=8.5, color=style.INK_2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(s) if s % 2 else f"'{str(s)[2:]}" for s in x], fontsize=9)
    axes[1].text(0, -0.2, f"{first['season']}: went {100 * first['go_rate']['est']:.1f}% of "
                 f"fourth downs, gave up {first['wp_points_lost_confident_per_game']['est']:.2f} "
                 f"points a game. {last['season']}: {100 * last['go_rate']['est']:.1f}% and "
                 f"{last['wp_points_lost_confident_per_game']['est']:.2f}.",
                 transform=axes[1].transAxes, fontsize=9, color=style.INK_2)
    fig.tight_layout()
    style.save(fig, path)


def _interval_rows(ax: Any, rows: list[dict], key: str, color: str, shrunk_key: str | None,
                   scale: float = 1.0) -> None:  # fmt: skip
    from fourth_and_rl import style

    y = np.arange(len(rows))
    est = np.array([r[key]["est"] for r in rows]) * scale
    lo = np.array([r[key]["lo"] for r in rows]) * scale
    hi = np.array([r[key]["hi"] for r in rows]) * scale
    ax.hlines(y, lo, hi, color=color, alpha=0.45, linewidth=2.2, label="Raw, 90% interval")
    ax.plot(est, y, "o", color=color, markersize=4.5, markerfacecolor=style.SURFACE)
    if shrunk_key:
        s = np.array([r[shrunk_key]["est"] for r in rows]) * scale
        ax.plot(s, y, "D", color=color, markersize=4.5, label="Shrunk toward the league")


def figure_teams(out: dict, path: Any) -> None:
    import matplotlib.pyplot as plt

    from fourth_and_rl import style

    style.apply()
    rows = sorted(out["teams"]["table"], key=lambda r: r["wins_lost_confident_per_17"]["est"])
    noise = out["teams"]["noise"]
    w0, w1 = out["teams"]["window"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 9.6), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.15]})  # fmt: skip
    y = np.arange(len(rows))
    ax = axes[0]
    _interval_rows(ax, rows, "go_rate_when_engine_confidently_says_go", style.BLUE, None, 100)
    ax.set_xlabel("Went for it when the engine\nconfidently said go (%)")
    ax.set_yticks(y)
    ax.set_yticklabels([r["team"] for r in rows], fontsize=9)
    ax.set_xlim(0, 100)
    ax = axes[1]
    _interval_rows(ax, rows, "wins_lost_confident_per_17", style.ORANGE,
                   "wins_lost_confident_per_17_shrunk")
    ax.axvline(noise["league_mean"], color=style.MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("Wins given up per 17 games\n(confident recommendations)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(left=0)
    style.title(axes[0], f"Team aggressiveness, {w0} to {w1}",
                f"Sorted by fourth-down wins given up. Only {noise['raw_interval_excludes_mean']} "
                f"of {noise['n']} teams' intervals exclude the league mean (about "
                f"{noise['expected_by_chance_at_90pct']:.0f} expected by chance).")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08, wspace=0.08)
    style.save(fig, path)


def figure_ledger(out: dict, led: list[dict], pen: dict, cfg: dict, path: Any) -> None:
    import matplotlib.pyplot as plt

    from fourth_and_rl import style

    style.apply()
    n = int(cfg["tables"]["n_shown"])
    key = "decision_wins_above_season_average_per_17"
    ranked = [r for r in led if r["ranked"]]
    ranked.sort(key=lambda r: r[f"{key}_shrunk"])
    shown = ranked[:n] + ranked[-n:]
    noise = out["coaches"]["noise"]["era_adjusted_per_17_games"]
    supported = pen["verdict"]["assigning_penalties_to_the_coach_is_supported"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 10.2), sharey=True,
                             gridspec_kw={"width_ratios": [1.5, 1]})  # fmt: skip
    y = np.array([i if i < n else i + 1 for i in range(len(shown))])
    ax = axes[0]
    lo = [r[f"{key}_lo"] for r in shown]
    hi = [r[f"{key}_hi"] for r in shown]
    est = [r[key] for r in shown]
    sh = [r[f"{key}_shrunk"] for r in shown]
    ax.hlines(y, lo, hi, color=style.ORANGE, alpha=0.45, linewidth=2.2, label="Raw, 90% interval")
    ax.plot(est, y, "o", color=style.ORANGE, markersize=4.5, markerfacecolor=style.SURFACE)
    ax.plot(sh, y, "D", color=style.ORANGE, markersize=4.5, label="Shrunk toward the league")
    ax.axvline(0, color=style.MUTED, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['coach']} ({r['games']} g, "
                        f"{r['decision_wp_points_above_season_average_per_opportunity']:+.2f}/4th)"
                        for r in shown], fontsize=8.6)  # fmt: skip
    ax.axhline(n, color=style.GRID, linewidth=1)
    ax.text(0.02, n, "above: fewest wins given up;  below: most", va="center",
            transform=ax.get_yaxis_transform(), fontsize=8.5, color=style.MUTED)
    ax.set_xlabel("Fourth-down wins given up per 17 games,\nabove the league average of the "
                  "same seasons (confident calls)")
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=9)
    ax = axes[1]
    pl_lo = [r["penalty_wins_lo"] * 17 / r["games"] for r in shown]  # raw, not era-adjusted
    pl_hi = [r["penalty_wins_hi"] * 17 / r["games"] for r in shown]
    pl_est = [r["penalty_wins_per_17"] for r in shown]
    ax.hlines(y, pl_lo, pl_hi, color=style.VIOLET, alpha=0.45, linewidth=2.2)
    ax.plot(pl_est, y, "o", color=style.VIOLET, markersize=4.5)
    ax.set_xlim(left=0)
    ax.set_xlabel("False starts and delays of game: wins per 17\ngames (raw). "
                  + ("NOT assigned to the coach:\nthe assumption test does not support it"
                     if not supported else "Assigned to the coach"))
    style.title(axes[0], "The coach cost ledger, 2011 to 2025",
                f"{n} lowest and {n} highest of {noise['n']} coaches with "
                f"{out['coaches']['min_games_ranked']}+ games, against their own seasons' "
                "average.\n"
                f"Reliability {noise['reliability']:.2f}; {noise['shrunk_interval_excludes_mean']} "
                "shrunk intervals exclude the average.")
    axes[0].set_title(axes[0].get_title(loc="left"), pad=40)
    fig.text(0.01, 0.005, "Per 4th: win-probability points per graded fourth down above the "
             "season average. Right panel: raw, not era-adjusted. "
             + ASSUMPTION, fontsize=8, color=style.INK_2, wrap=True)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.91, bottom=0.13, wspace=0.08)
    style.save(fig, path)


def figure_case(out: dict, path: Any) -> None:
    import matplotlib.pyplot as plt

    from fourth_and_rl import style

    style.apply()
    cs = out["case_study"]
    downs = cs["fourth_downs"]
    fig = plt.figure(figsize=(12.5, 10.2))
    gs = fig.add_gridspec(2, len(downs), height_ratios=[1.05, 1], hspace=0.42, wspace=0.35,
                          bottom=0.13, top=0.92, left=0.07, right=0.98)
    ax = fig.add_subplot(gs[0, :])
    el = np.array(cs["path"]["elapsed"]) / 60
    ax.step(el, 100 * np.array(cs["path"]["wp"]), where="post", color=style.BLUE, linewidth=1.8)
    for i, d in enumerate(downs):
        m = d["elapsed"] / 60
        ax.axvline(m, color=style.ACTION_COLORS[d["actual"]], linewidth=1, alpha=0.6)
        ax.text(m, 103, str(i + 1), ha="center", va="bottom", fontsize=9, color=style.INK)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 112)
    ax.set_xticks([0, 15, 30, 45, 60])
    ax.set_xticklabels(["Kickoff", "Q2", "Half", "Q4", "End"])
    ax.set_ylabel("Detroit win probability (%)")
    style.title(ax, "Lions at 49ers, NFC Championship (January 2024): the fourth downs",
                f"Detroit's win probability from a model that never saw the 2023 season; "
                f"numbered lines are Detroit's fourth downs. Final: {cs['final_score']}.")
    for i, d in enumerate(downs):
        a = fig.add_subplot(gs[1, i])
        vals = [d["go_wp"], d["fg_wp"], d["punt_wp"]]
        names = ["go", "field_goal", "punt"]
        for j, (v, nme) in enumerate(zip(vals, names, strict=True)):
            if v is None:
                a.text(j, 2, "n/a", ha="center", fontsize=8, color=style.MUTED)
                continue
            chosen = nme == d["actual"]
            a.bar(j, 100 * v, color=style.ACTION_COLORS[nme], alpha=1 if chosen else 0.45,
                  edgecolor=style.INK if chosen else "none", linewidth=1.4)
            a.text(j, 100 * v + 1.5, f"{100 * v:.0f}", ha="center", fontsize=8, color=style.INK)
        a.set_xticks([0, 1, 2])
        a.set_xticklabels(["Go", "FG", "Punt"], fontsize=8.5)
        a.set_ylim(0, 105)
        a.tick_params(axis="y", labelsize=8)
        if i == 0:
            a.set_ylabel("Win probability (%)", fontsize=9)
        side = "own" if d["yardline_100"] > 50 else "SF"
        yl = 100 - d["yardline_100"] if side == "own" else d["yardline_100"]
        lead = d["score_differential"]
        score = f"up {lead}" if lead > 0 else ("tied" if lead == 0 else f"down {-lead}")
        cat = {"confident": "confident", "lean": "lean", "uncertain": "too close"}.get(
            d["category"], "n/a")
        lost = 100 * (d["wp_lost"] or 0)
        a.set_title(f"{i + 1}. Q{d['qtr']} {d['clock']}\n4th & {d['ydstogo']}, {side} {yl}, "
                    f"{score}", fontsize=8.8, fontweight="bold", loc="left", pad=6)
        verdict = "best call" if d["actual"] == d["best_action"] else f"gave up {lost:.1f}"
        a.text(0.0, -0.15, f"{style.ACTION_LABELS[d['actual']]}: {verdict}\n"
               f"engine: {style.ACTION_LABELS[d['best_action']].lower()} ({cat})",
               transform=a.transAxes, fontsize=7.8, color=style.INK_2, va="top")
    fig.text(0.01, 0.01, f"Engine: one play ahead, conversion at the game's spread. Bootstrap: "
             f"{cs['n_boot']} refits of whole games from the other development seasons. Outlined "
             "bar: what Detroit did.", fontsize=8.5, color=style.INK_2)
    style.save(fig, path)


def draw_all(out: dict, led: list[dict], pen: dict, cfg: dict) -> None:
    figure_trend(out, ASSETS / "m7_league_trend.png")
    figure_teams(out, ASSETS / "m7_team_aggressiveness.png")
    figure_ledger(out, led, pen, cfg, ASSETS / "m7_coach_ledger.png")
    figure_case(out, ASSETS / "m7_case_study.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=["grade", "analyze", "all"], default="all")
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw the figures from the saved results")
    args = parser.parse_args()
    cfg = coaches.load_config()
    if args.figures_only:
        led = pl.read_csv(RESULTS / "m7_coach_ledger.csv").to_dicts()
        draw_all(read_json(RESULTS / "m7_coach_evaluation.json"), led,
                 read_json(RESULTS / "m7_penalty_ledger.json"), cfg)
        return
    if args.stage in ("grade", "all"):
        stage_grade(cfg)
    if args.stage in ("analyze", "all"):
        stage_analyze(cfg)


if __name__ == "__main__":
    main()
