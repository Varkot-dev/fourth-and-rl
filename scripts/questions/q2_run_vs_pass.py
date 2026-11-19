"""Burning question 2: why do offenses run when passing gains more?

Steps (each is one block of the results file):
  1. puzzle     run versus pass on EPA, success, yards, first downs, turnovers and spread
                of outcomes: all plays, neutral game script, by down and distance, field
                position and season.
  2. marginal   the same comparison inside fine situation cells, and the gap with the
                situation held fixed. An out-of-sample check of the cell verdicts.
  3. defense    do payoffs depend on how often the offense passes: team tendencies from
                other games, what happened earlier in the game, defenders in the box.
  4. risk       outcome distributions; protecting a late lead; short yardage.
  5. verdict    how much of the gap survives, the trend, and a rough upper bound on the
                points left on the table.
  6. hook       run and pass outcome distributions for the decision model.

EPA, win probability (`wp`, `wpa`) and expected pass probability (`xpass`) are
nflfastR's models, used as given. All intervals resample whole games.

Outputs:
  results/questions/q2_run_vs_pass.json
  assets/questions/q2_run_vs_pass.png           the argument in one look
  assets/questions/q2_defense_adjusts.png       evidence on defensive adjustment
  assets/questions/q2_risk_and_clock.png        outcome spread and the late-lead case
  data/models/run_pass_outcomes.joblib          the hook, fitted on development seasons

Run:  python scripts/questions/q2_run_vs_pass.py                  (under a minute)
      python scripts/questions/q2_run_vs_pass.py --figures-only   (redraw from the JSON)
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")  # the machine is shared
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from fourth_and_rl import data, splits, style  # noqa: E402
from fourth_and_rl import run_pass as rp  # noqa: E402
from fourth_and_rl.io import read_json, write_json  # noqa: E402
from fourth_and_rl.paths import ASSETS, DATA, RESULTS  # noqa: E402

OUT = RESULTS / "questions" / "q2_run_vs_pass.json"
HOOK_PATH = DATA / "models" / "run_pass_outcomes.joblib"
COLORS = {"run": style.ORANGE, "pass": style.BLUE}
LABELS = {"run": "Run", "pass": "Pass"}


def np_col(df: pl.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy()


def mask_group(mask: np.ndarray) -> np.ndarray:
    """Group index 0 where the mask holds, -1 (left out) elsewhere."""
    return np.where(mask, 0, -1)


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "boot"}


# --------------------------------------------------------------------------------------
# 1. The puzzle, measured
# --------------------------------------------------------------------------------------


def named_situations(df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Down-and-distance groups everyone talks about (each play in at most one)."""
    down, togo = np_col(df, "down").astype(int), np_col(df, "ydstogo").astype(int)
    group = np.full(df.height, -1)
    labels = []

    def add(label: str, m: np.ndarray) -> None:
        group[m] = len(labels)
        labels.append(label)

    add("1st and 10", (down == 1) & (togo == 10))
    add("2nd and 8 or more", (down == 2) & (togo >= 8))
    add("2nd and 3 to 7", (down == 2) & (togo >= 3) & (togo <= 7))
    add("2nd and 1 or 2", (down == 2) & (togo <= 2))
    add("3rd and 8 or more", (down == 3) & (togo >= 8))
    add("3rd and 3 to 7", (down == 3) & (togo >= 3) & (togo <= 7))
    add("3rd and 1 or 2", (down == 3) & (togo <= 2))
    add("4th and 1 or 2", (down == 4) & (togo <= 2))
    return group, labels


def puzzle(df: pl.DataFrame, full: pl.DataFrame, w: np.ndarray, cfg: dict) -> dict:
    level = cfg["level"]
    neutral, early = np_col(df, "neutral"), np_col(df, "early")
    score = np_col(df, "score_differential")
    late = np_col(df, "half_seconds_remaining") <= cfg["neutral"]["min_half_seconds"]
    big = cfg["neutral"]["max_abs_score_diff"]
    samples = {
        "all_plays": np.ones(df.height, dtype=bool),
        "neutral": neutral,
        "not_neutral": ~neutral,
        "neutral_early_downs": neutral & early,
        "offense_trails_by_more_than_8": score < -big,
        "offense_leads_by_more_than_8": score > big,
        "last_two_minutes_of_a_half": late,
    }
    out: dict = {"samples": {}}
    for name, m in samples.items():
        c = rp.frame_compare(df, mask_group(m), 1, w, level, sd_of=("epa", "yards"))
        out["samples"][name] = c.rows([name])[0]

    # How much the neutral filter moves the gap (same resamples for both).
    c_all = rp.frame_compare(df, mask_group(samples["all_plays"]), 1, w, level, ["epa"])
    c_neu = rp.frame_compare(df, mask_group(neutral), 1, w, level, ["epa"])
    out["epa_gap_neutral_minus_all"] = rp.summary(
        c_neu.est[0, 2, 0] - c_all.est[0, 2, 0], c_neu.boot[:, 0, 2, 0] - c_all.boot[:, 0, 2, 0],
        level,
    )  # fmt: skip

    nd = df.filter(pl.col("neutral"))
    down = np_col(nd, "down").astype(int)
    nb = len(cfg["dist_edges"])
    out["neutral_by_down"] = rp.frame_compare(nd, down - 1, 4, w, level).rows(
        [1, 2, 3, 4], key="down"
    )
    dd = rp.frame_compare(nd, (down - 1) * nb + np_col(nd, "dist_bin"), 4 * nb, w, level)
    out["neutral_by_down_and_distance"] = dd.rows(
        [{"down": d, "distance": lab} for d in (1, 2, 3, 4) for lab in cfg["dist_labels"]]
    )
    group, labels = named_situations(nd)
    out["neutral_named_situations"] = rp.frame_compare(nd, group, len(labels), w, level).rows(
        labels, key="situation"
    )
    out["neutral_by_field_position"] = rp.frame_compare(
        nd, np_col(nd, "field_bin"), len(cfg["field_edges"]), w, level
    ).rows(cfg["field_labels"], key="yardline_100")

    seasons = sorted(df["season"].unique().to_list())
    code = np_col(df, "season") - seasons[0]
    for name, m in (("all_plays", samples["all_plays"]), ("neutral", neutral),
                    ("neutral_early_downs", neutral & early)):  # fmt: skip
        c = rp.frame_compare(df, np.where(m, code, -1), len(seasons), w, level,
                             ["epa", "success", "yards"])  # fmt: skip
        out[f"by_season_{name}"] = c.rows(seasons, key="season")

    # How the classification choices move the answer.
    sens = {}
    for how in ("run", "drop"):
        alt = rp.prepare(full, cfg, scrambles_as=how)
        for name, m in (("all_plays", np.ones(alt.height, dtype=bool)),
                        ("neutral", np_col(alt, "neutral"))):  # fmt: skip
            c = rp.frame_compare(alt, mask_group(m), 1, w, level, ["epa", "yards", "success"])
            sens[f"scrambles_as_{how}_{name}"] = c.rows([name])[0]
    # nflfastR's own flags, which also count plays wiped out by a penalty (for example
    # defensive pass interference, which our table records as "no_play").
    codes = df.select("game_id", "game_code").unique()
    flags = (
        full.filter(
            ((pl.col("pass") == 1) | (pl.col("rush") == 1))
            & pl.col("epa").is_not_null() & pl.col("down").is_between(1, 4)
        )
        .with_columns(rp.neutral_expr(cfg).alias("neutral"))
        .join(codes, on="game_id", how="inner")
    )  # fmt: skip
    for name, m in (("all_plays", np.ones(flags.height, dtype=bool)),
                    ("neutral", np_col(flags, "neutral"))):  # fmt: skip
        c = rp.compare({"epa": np_col(flags, "epa")}, np_col(flags, "pass").astype(int),
                       mask_group(m), 1, np_col(flags, "game_code"), w, level)  # fmt: skip
        sens[f"nflfastr_flags_with_penalty_plays_{name}"] = c.rows([name])[0]
    out["sensitivity"] = sens
    out["counts"] = {
        "called_plays": df.height,
        "games": int(df["game_code"].n_unique()),
        "scrambles_counted_as_passes": int((df["qb_scramble"] == 1).sum()),
        "sacks": int((df["sack"] == 1).sum()),
        "neutral_share": float(neutral.mean()),
    }
    return out


# --------------------------------------------------------------------------------------
# 2. Average versus marginal
# --------------------------------------------------------------------------------------


def cell_block(d: pl.DataFrame, w: np.ndarray, cfg: dict, top: int = 8) -> dict:
    """Cell comparison on the plays in `d` (already restricted to the sample wanted)."""
    level, min_n = cfg["level"], cfg["cells"]["min_n_per_call"]
    c = rp.frame_compare(d, np_col(d, "cell"), rp.n_cells(cfg), w, level, ["epa", "success"])
    verdict = rp.classify_cells(c, "epa", min_n)
    rows = c.rows(rp.cell_labels(cfg))
    names = {1: "pass_better", 0: "tie", -1: "run_better", -9: "too_few_plays"}
    for r, v in zip(rows, verdict, strict=True):
        r["verdict"] = names[int(v)]
    ok = [r for r in rows if r["verdict"] != "too_few_plays"]
    early = np.repeat([True, True, False, False], rp.n_cells(cfg) // 4)
    by_gap = sorted(ok, key=lambda r: r["epa"]["pass_minus_run"]["value"])
    return {
        "n_plays": d.height,
        "cells_compared": len(ok),
        "share_of_plays": rp.cell_shares(c, verdict),
        "share_of_cells": {v: float((verdict == k).mean()) for k, v in names.items()},
        "gap_situation_fixed_all_downs": strip(rp.standardized_gap(c, "epa", min_n, "all", level)),
        "gap_situation_fixed_all_downs_run_weights": strip(
            rp.standardized_gap(c, "epa", min_n, "run", level)
        ),
        "gap_situation_fixed_early_downs": strip(
            rp.standardized_gap(c, "epa", min_n, "all", level, early)
        ),
        "gap_situation_fixed_early_downs_run_weights": strip(
            rp.standardized_gap(c, "epa", min_n, "run", level, early)
        ),
        "success_gap_situation_fixed_all_downs": strip(
            rp.standardized_gap(c, "success", min_n, "all", level)
        ),
        "largest_pass_advantage": by_gap[::-1][:top],
        "smallest_or_reversed": by_gap[:top],
        "cells": ok,
    }


def out_of_sample_cells(df: pl.DataFrame, cfg: dict) -> dict:
    """Label cells on the development seasons, then measure the gap in each label group
    on the test seasons. Guards against reading noise as "passing is better here"."""
    level, min_n = cfg["level"], cfg["cells"]["min_n_per_call"]
    test_seasons = list(splits.test_seasons())
    dev = df.filter(pl.col("neutral") & ~pl.col("season").is_in(test_seasons))
    splits.assert_no_test_seasons(dev, "q2 cell verdicts (development seasons)")
    # Recorded in the access log; the rows themselves come from the joined table.
    logged = splits.load_test_plays(
        purpose="q2 run versus pass: check cell verdicts labelled on development seasons"
    )
    test = df.filter(pl.col("neutral") & pl.col("season").is_in(test_seasons))
    splits.assert_disjoint_games(dev, test, "q2 cell verdicts")
    assert set(test["game_id"].unique().to_list()) <= set(logged["game_id"].unique().to_list())

    def recode(d: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
        d = d.with_columns(pl.col("game_id").rank("dense").cast(pl.Int64).sub(1).alias("game_code"))
        return d, rp.boot_weights(int(d["game_code"].max()) + 1, cfg["n_boot"], cfg["seed"])

    dev, w_dev = recode(dev)
    test, w_test = recode(test)
    c_dev = rp.frame_compare(dev, np_col(dev, "cell"), rp.n_cells(cfg), w_dev, level, ["epa"])
    verdict = rp.classify_cells(c_dev, "epa", min_n)
    names = {1: "pass_better", 0: "tie", -1: "run_better", -9: "too_few_plays"}
    order = [1, 0, -1, -9]
    group = np.array([order.index(int(v)) for v in verdict])[np_col(test, "cell")]
    c_test = rp.frame_compare(test, group, 4, w_test, level, ["epa"])
    rows = c_test.rows([names[k] for k in order], key="verdict_on_development_seasons")
    for r, k in zip(rows, order, strict=True):
        r["cells"] = int((verdict == k).sum())
    return {
        "development_seasons": sorted(splits.seasons_in(dev)),
        "test_seasons": sorted(splits.seasons_in(test)),
        "test_gap_by_development_verdict": rows,
    }


def marginal(df: pl.DataFrame, w: np.ndarray, cfg: dict) -> dict:
    level = cfg["level"]
    nd = df.filter(pl.col("neutral"))
    out = {
        "all_seasons": cell_block(nd, w, cfg),
        "recent_seasons": cell_block(
            nd.filter(pl.col("season") >= cfg["recent_first_season"]), w, cfg
        ),
        "out_of_sample": out_of_sample_cells(df, cfg),
    }
    for block in ("recent_seasons",):
        out[block].pop("cells")
    # One number for the situation: nflfastR's expected pass probability.
    xp = nd.filter(pl.col("xpass").is_not_null())
    d = cfg["defense"]
    c = rp.frame_compare(xp, rp.bin_index(np_col(xp, "xpass"), d["xpass_edges"]),
                         len(d["xpass_edges"]), w, level, ["epa", "success"])  # fmt: skip
    out["by_nflfastr_expected_pass_probability"] = c.rows(d["xpass_labels"], key="xpass")
    return out


# --------------------------------------------------------------------------------------
# 3. Do defenses adjust?
# --------------------------------------------------------------------------------------


def tendency_regressions(ne: pl.DataFrame, w: np.ndarray, cfg: dict) -> dict:
    """Game-level regressions of an offense's run EPA, pass EPA and box count faced on
    its pass rate in the season's other games (neutral early downs throughout)."""
    level = cfg["level"]
    tg = rp.team_game_table(ne).filter(
        pl.col("loo_pass_rate").is_not_null() & pl.col("loo_run_epa").is_not_null()
        & pl.col("loo_pass_epa").is_not_null()
    )  # fmt: skip
    game = np_col(tg, "game_code")
    rate = (np_col(tg, "loo_pass_rate") - float(tg["loo_pass_rate"].mean())) / 0.10
    season_d = rp.dummies(np_col(tg, "season"))
    team_d = rp.dummies(tg["posteam"].rank("dense").to_numpy())
    ability = np.column_stack([np_col(tg, "loo_pass_epa"), np_col(tg, "loo_run_epa")])
    one = np.ones((tg.height, 1))
    specs = {
        "across_teams_season_effects": np.hstack([one, rate[:, None], season_d]),
        "within_team_across_seasons": np.hstack([one, rate[:, None], season_d, team_d]),
        "across_teams_controlling_for_efficiency_in_other_games": np.hstack(
            [one, rate[:, None], season_d, ability]
        ),
    }
    outcomes = {
        "run_epa": (np_col(tg, "run_epa"), np_col(tg, "n_run").astype(float)),
        "pass_epa": (np_col(tg, "pass_epa"), np_col(tg, "n_pass").astype(float)),
        "defenders_in_box": (np_col(tg, "box"), np_col(tg, "n_box").astype(float)),
    }
    res: dict = {}
    boots: dict = {}
    for oname, (y, n) in outcomes.items():
        ok = np.isfinite(y) & (n > 0)
        res[oname] = {"team_games": int(ok.sum())}
        for sname, x in specs.items():
            xs = x[ok]
            xs = xs[:, np.ptp(xs, axis=0) + (np.arange(xs.shape[1]) == 0) > 0]
            fit = rp.cluster_ols(xs, y[ok], n[ok], game[ok], w, level)
            res[oname][sname] = {"value": float(fit["coef"][1]), "low": float(fit["low"][1]),
                                 "high": float(fit["high"][1])}  # fmt: skip
            boots[oname, sname] = (float(fit["coef"][1]), fit["boot"][:, 1])
    res["unit"] = "change in the outcome per 10 percentage points of pass rate in other games"
    res["pass_rate_other_games"] = {
        "mean": float(tg["loo_pass_rate"].mean()), "sd": float(tg["loo_pass_rate"].std()),
        "p05": float(tg["loo_pass_rate"].quantile(0.05)),
        "p95": float(tg["loo_pass_rate"].quantile(0.95)),
    }  # fmt: skip

    # If an offense sets its pass rate p to maximize p * P(p) + (1 - p) * R(p), where
    # the payoffs P and R depend on p because defenses adjust, the first-order condition
    # is  P - R = -p * P'(p) - (1 - p) * R'(p).  Given the measured gap and the measured
    # slope of run EPA, how steeply would pass EPA have to fall with the pass rate for
    # today's mix to be optimal?  P'(p) = -(gap + (1 - p) * R'(p)) / p.
    p = float(ne["is_pass"].mean())
    gap = rp.frame_compare(ne, np.zeros(ne.height, dtype=int), 1, w, level, ["epa"])
    g, gb = gap.est[0, 2, 0], gap.boot[:, 0, 2, 0]
    res["pass_epa_slope_needed_for_todays_mix_to_be_optimal"] = {
        "pass_rate": p,
        "gap": rp.summary(g, gb, level),
        "unit": "EPA per pass per 10 percentage points of pass rate",
    }
    for sname in specs:
        (ps, pb), (rs, rb) = boots["pass_epa", sname], boots["run_epa", sname]
        res["pass_epa_slope_needed_for_todays_mix_to_be_optimal"][sname] = {
            "needed": rp.summary(-(g / 10 + (1 - p) * rs) / p, -(gb / 10 + (1 - p) * rb) / p,
                                 level),
            "measured_not_causal": rp.summary(ps, pb, level),
        }  # fmt: skip
    return res


def sequential(ne: pl.DataFrame, w: np.ndarray, cfg: dict) -> dict:
    """Does the payoff of a run or a pass depend on what the offense has been doing in
    this game? Neutral early downs."""
    level = cfg["level"]
    tg = rp.team_game_table(ne).select("game_id", "posteam", "loo_pass_rate")
    d = ne.join(tg, on=["game_id", "posteam"], how="left").filter(
        pl.col("recent_pass_share").is_not_null() & pl.col("loo_pass_rate").is_not_null()
    )
    nb = len(cfg["dist_edges"])
    sit = rp.dummies((np_col(d, "down").astype(int) - 1) * nb + np_col(d, "dist_bin"))
    x = np.hstack([
        np.ones((d.height, 1)),
        (np_col(d, "recent_pass_share")[:, None] - 0.5) / 0.10,
        (np_col(d, "loo_pass_rate")[:, None] - 0.5) / 0.10,
        sit, rp.dummies(np_col(d, "field_bin")), rp.dummies(np_col(d, "season")),
    ])  # fmt: skip
    is_pass, epa, game = np_col(d, "is_pass"), np_col(d, "epa"), np_col(d, "game_code")
    out: dict = {
        "window_called_plays": cfg["defense"]["recent_window"],
        "unit": "EPA per play per 10 percentage points of pass share over the window",
    }
    for s, call in enumerate(rp.CALLS):
        m = is_pass == s
        fit = rp.cluster_ols(x[m], epa[m], np.ones(int(m.sum())), game[m], w, level)
        out[f"{call}_epa_slope_on_recent_pass_share"] = {
            "n": int(m.sum()), "value": float(fit["coef"][1]), "low": float(fit["low"][1]),
            "high": float(fit["high"][1]),
        }  # fmt: skip

    # By thirds of recent pass share (no model).
    share = np_col(d, "recent_pass_share")
    edges = [0.0, 0.45, 0.65]
    c = rp.frame_compare(d, rp.bin_index(share, edges), 3, w, level, ["epa", "success"])
    out["by_recent_pass_share"] = c.rows(
        ["40% or less passes", "50% or 60%", "70% or more"], key="recent_pass_share"
    )

    # The call itself: second and 10 after a first-and-10 that gained nothing. Same
    # down and distance, so only the previous call differs.
    s = ne.filter(
        (pl.col("down") == 2) & (pl.col("ydstogo") == 10) & (pl.col("prev_down") == 1)
        & (pl.col("prev_ydstogo") == 10)
        & (((pl.col("prev_is_pass") == 1) & (pl.col("prev_incomplete") == 1))
           | ((pl.col("prev_is_pass") == 0) & (pl.col("prev_yards") == 0)))
    )  # fmt: skip
    prev = np_col(s, "prev_is_pass").astype(int)
    rate = rp.group_means(np_col(s, "is_pass").astype(float), prev, 2, np_col(s, "game_code"), w,
                          level)  # fmt: skip
    both = rp.game_sums(np_col(s, "game_code"), w.shape[1], prev, 2,
                        np.column_stack([np.ones(s.height), np_col(s, "is_pass")]))  # fmt: skip
    tot = both.sum(axis=0)
    bt = (w @ both.reshape(w.shape[1], -1).astype(np.float32)).reshape(-1, 2, 2)
    diff_b = bt[:, 1, 1] / bt[:, 1, 0] - bt[:, 0, 1] / bt[:, 0, 0]
    out["second_and_10_after_no_gain"] = {
        "pass_rate_after_run_for_no_gain": rate[0],
        "pass_rate_after_incomplete_pass": rate[1],
        "after_pass_minus_after_run": rp.summary(
            tot[1, 1] / tot[1, 0] - tot[0, 1] / tot[0, 0], diff_b, level
        ),
        "epa_by_previous_call": rp.frame_compare(s, prev, 2, w, level, ["epa"]).rows(
            ["after run for no gain", "after incomplete pass"], key="previous_play"
        ),
        # Here the "pass" side of the comparison is "the previous play was a pass".
        "epa_after_incomplete_minus_after_run": {
            call: rp.summary(cc.est[k, 2, 0], cc.boot[:, k, 2, 0], level)
            for cc in [rp.compare({"epa": np_col(s, "epa")}, prev,
                                  np_col(s, "is_pass").astype(np.int64), 2,
                                  np_col(s, "game_code"), w, level)]
            for k, call in enumerate(rp.CALLS)
        },  # fmt: skip
    }
    return out


def box(df: pl.DataFrame, w: np.ndarray, cfg: dict) -> dict:
    """Defenders in the box (nflverse participation files, 2016 on)."""
    level, d = cfg["level"], cfg["defense"]
    b = df.filter(pl.col("neutral") & pl.col("defenders_in_box").is_not_null())
    game, boxes = np_col(b, "game_code"), np_col(b, "defenders_in_box")
    out: dict = {
        "seasons": sorted(b["season"].unique().to_list()),
        "n_plays": b.height,
        "share_of_neutral_plays_with_a_count": float(
            b.height / max(df.filter(pl.col("neutral") & (pl.col("season") >= cfg[
                "participation_first_season"])).height, 1)
        ),
    }  # fmt: skip
    group, labels = named_situations(b)
    means = rp.group_means(boxes, group, len(labels), game, w, level)
    heavy = rp.group_means((boxes >= 8).astype(float), group, len(labels), game, w, level)
    out["by_situation"] = [
        {"situation": lab, "mean_box": m, "share_8_or_more": h}
        for lab, m, h in zip(labels, means, heavy, strict=True)
    ]
    xp = rp.bin_index(np.nan_to_num(np_col(b, "xpass"), nan=0.5), d["xpass_edges"])
    out["by_nflfastr_expected_pass_probability"] = [
        {"xpass": lab, "mean_box": m}
        for lab, m in zip(d["xpass_labels"],
                          rp.group_means(boxes, xp, len(d["xpass_labels"]), game, w, level),
                          strict=True)
    ]  # fmt: skip
    seasons = out["seasons"]
    e = b.filter(pl.col("early"))
    out["by_season_neutral_early_downs"] = [
        {"season": s, "mean_box": m}
        for s, m in zip(seasons, rp.group_means(
            np_col(e, "defenders_in_box"), np_col(e, "season") - seasons[0], len(seasons),
            np_col(e, "game_code"), w, level), strict=True)
    ]  # fmt: skip
    bins = rp.bin_index(np_col(e, "defenders_in_box"), d["box_edges"])
    c = rp.frame_compare(e, bins, len(d["box_edges"]), w, level, ["epa", "success", "yards"])
    out["neutral_early_downs_by_box"] = c.rows(d["box_labels"], key="defenders_in_box")
    # Same thing on first and 10 only, where the situation is the same for everyone.
    f = e.filter((pl.col("down") == 1) & (pl.col("ydstogo") == 10))
    c = rp.frame_compare(f, rp.bin_index(np_col(f, "defenders_in_box"), d["box_edges"]),
                         len(d["box_edges"]), w, level, ["epa", "success", "yards"])  # fmt: skip
    out["neutral_first_and_10_by_box"] = c.rows(d["box_labels"], key="defenders_in_box")
    return out


def defense(df: pl.DataFrame, w: np.ndarray, cfg: dict) -> dict:
    ne = df.filter(pl.col("neutral") & pl.col("early"))
    return {
        "team_tendency": tendency_regressions(ne, w, cfg),
        "within_game": sequential(ne, w, cfg),
        "defenders_in_box": box(df, w, cfg),
    }


# --------------------------------------------------------------------------------------
# 4. Risk and the objective
# --------------------------------------------------------------------------------------


def distribution(d: pl.DataFrame, w: np.ndarray, cfg: dict) -> dict:
    """Outcome distributions of runs and passes on the plays in `d`."""
    level = cfg["level"]
    is_pass, yards, epa = np_col(d, "is_pass"), np_col(d, "yards"), np_col(d, "epa")
    events = {
        "loses_yards": (yards < 0).astype(float),
        "no_gain_or_loss": (yards <= 0).astype(float),
        "gains_4_or_more": (yards >= 4).astype(float),
        "gains_10_or_more": (yards >= 10).astype(float),
        "gains_20_or_more": (yards >= 20).astype(float),
        "epa_below_minus_1": (epa < -1).astype(float),
        "epa_above_plus_1": (epa > 1).astype(float),
        "turnover": np_col(d, "turnover"),
        "sack": (np_col(d, "sack") == 1).astype(float),
        "incomplete": (np_col(d, "incomplete_pass") == 1).astype(float),
        "epa": epa,
        "yards": yards,
    }
    c = rp.compare(events, is_pass.astype(int), np.zeros(d.height, dtype=np.int64), 1,
                   np_col(d, "game_code"), w, level, sd_of=("epa", "yards"))  # fmt: skip
    out = c.rows(["plays"])[0]
    grid = np.arange(-10, 41)
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    for s, call in enumerate(rp.CALLS):
        m = is_pass == s
        hist = np.bincount(np.clip(yards[m], -10, 40).astype(int) + 10, minlength=grid.size)
        out[f"{call}_yards_histogram"] = {
            "yards": grid.tolist(), "share": (hist / m.sum()).tolist(),
            "note": "ends are clipped: -10 means 10 or more lost, 40 means 40 or more gained",
        }  # fmt: skip
        out[f"{call}_quantiles"] = {
            "probabilities": q,
            "epa": np.quantile(epa[m], q).tolist(),
            "yards": np.quantile(yards[m], q).tolist(),
        }
    return out


def lead_late(df: pl.DataFrame, w: np.ndarray, cfg: dict, max_seconds: int | None = None) -> dict:
    """Leading by one score late: EPA says one thing, the clock another."""
    level, c = cfg["level"], dict(cfg["lead_late"])
    c["max_game_seconds"] = max_seconds or c["max_game_seconds"]
    d = df.filter(
        (pl.col("qtr") == c["qtr"]) & (pl.col("game_seconds_remaining") <= c["max_game_seconds"])
        & pl.col("score_differential").is_between(c["min_lead"], c["max_lead"])
        & pl.col("down").cast(pl.Int64).is_in(c["downs"])
    )  # fmt: skip
    elapsed = np_col(d, "seconds_elapsed")
    ok = np.isfinite(elapsed) & (elapsed >= 0) & (elapsed <= c["max_elapsed_seconds"])
    d = d.filter(pl.Series(ok)).with_columns(
        pl.col("wpa").fill_null(0.0).alias("wpa_nflfastr"),
        (pl.col("seconds_elapsed") <= 10).cast(pl.Float64).alias("clock_stopped"),
    )
    d = d.with_columns(
        (pl.col("seconds_elapsed") + c["timeout_seconds"] * pl.col("defense_timeout_after"))
        .alias("clock_cost_to_defense")
    )
    metrics = ["epa", "wpa_nflfastr", "seconds_elapsed", "defense_timeout_after",
               "clock_cost_to_defense", "clock_stopped", "first_down", "turnover"]  # fmt: skip
    nb = len(cfg["dist_edges"])
    cell = (np_col(d, "down").astype(int) - 1) * nb + np_col(d, "dist_bin")
    comp = rp.frame_compare(d, cell, 4 * nb, w, level, metrics)
    out: dict = {
        "definition": c,
        "overall": rp.frame_compare(d, np.zeros(d.height, dtype=int), 1, w, level, metrics).rows(
            ["all"]
        )[0],
        "same_down_and_distance": {
            m: strip(rp.standardized_gap(comp, m, 30, "all", level)) for m in metrics
        },
    }
    f = d.filter((pl.col("down") == 1) & (pl.col("ydstogo") == 10))
    out["first_and_10"] = rp.frame_compare(f, np.zeros(f.height, dtype=int), 1, w, level,
                                           metrics).rows(["1st and 10"])[0]  # fmt: skip
    # The clean clock comparison: the defense has no timeouts left to stop it.
    z = d.filter((pl.col("defteam_timeouts_remaining") == 0)
                 & (pl.col("game_seconds_remaining") > c["timeout_seconds"]))  # fmt: skip
    out["defense_has_no_timeouts"] = rp.frame_compare(
        z, np.zeros(z.height, dtype=int), 1, w, level, ["seconds_elapsed", "clock_stopped"]
    ).rows(["defense has no timeouts"])[0]
    # Clock by what happened on the play.
    kinds = {
        "run": pl.col("is_pass") == 0,
        "complete_pass": (pl.col("is_pass") == 1) & (pl.col("complete_pass") == 1),
        "incomplete_pass": (pl.col("is_pass") == 1) & (pl.col("incomplete_pass") == 1),
        "sack": pl.col("sack") == 1,
    }
    out["seconds_to_next_snap_by_result"] = {
        k: d.filter(e.fill_null(False)).select(
            pl.len().alias("n"), pl.col("seconds_elapsed").median().alias("median"),
            pl.col("seconds_elapsed").mean().alias("mean"),
        ).row(0, named=True)
        for k, e in kinds.items()
    }  # fmt: skip
    out["note"] = (
        "seconds_elapsed is game clock from this snap to the next snap, including plays "
        "after which the defense called timeout, the two-minute warning fell, the ball "
        "went out of bounds or a team scored. clock_cost_to_defense adds timeout_seconds "
        "for each timeout the defense spent, a rough index and not a model. "
        "docs/research/endgame_clock.md has the cleaner split (in-bounds run by a leader "
        "43 seconds, incompletion 5)."
    )
    return out


def short_yardage(df: pl.DataFrame, w: np.ndarray, cfg: dict) -> dict:
    level, c = cfg["level"], cfg["short_yardage"]
    nd = df.filter(pl.col("neutral"))
    down, togo = np_col(nd, "down").astype(int), np_col(nd, "ydstogo").astype(int)
    yl = np_col(nd, "yardline_100")
    group = np.full(nd.height, -1)
    labels = []
    for dn in c["downs"]:
        for t in range(1, c["max_ydstogo"] + 1):
            group[(down == dn) & (togo == t)] = len(labels)
            labels.append(f"down {dn} and {t}")
    out = {"by_down_and_distance": rp.frame_compare(nd, group, len(labels), w, level).rows(
        labels, key="situation")}  # fmt: skip
    goal = (yl <= c["goal_line_max_yardline"]) & (down <= 3)
    out["goal_line_downs_1_to_3"] = rp.frame_compare(nd, mask_group(goal), 1, w, level).rows(
        [f"inside the {c['goal_line_max_yardline'] + 1}"])[0]  # fmt: skip
    return out


def risk(df: pl.DataFrame, w: np.ndarray, cfg: dict) -> dict:
    f10 = df.filter((pl.col("down") == 1) & (pl.col("ydstogo") == 10)).with_columns(
        pl.col("wpa").fill_null(0.0).alias("wpa_nflfastr")
    )
    first10 = f10.filter(pl.col("neutral"))
    c = cfg["lead_late"]
    lead = (
        (np_col(f10, "qtr") == c["qtr"])
        & (np_col(f10, "score_differential") >= c["min_lead"])
        & (np_col(f10, "score_differential") <= c["max_lead"])
    )
    secs = np_col(f10, "game_seconds_remaining")
    ladder = {
        "Neutral game script": np_col(f10, "neutral"),
        f"Leading by {c['min_lead']} to {c['max_lead']}, 4th quarter": lead,
        f"Leading by {c['min_lead']} to {c['max_lead']}, last "
        f"{c['max_game_seconds'] // 60} minutes": lead & (secs <= c["max_game_seconds"]),
    }
    wpa_rows = [
        rp.frame_compare(f10, mask_group(m), 1, w, cfg["level"], ["epa", "wpa_nflfastr"]).rows(
            [name], key="situation")[0]
        for name, m in ladder.items()
    ]  # fmt: skip
    return {
        "first_and_10_epa_and_win_probability_added": wpa_rows,
        "neutral_first_and_10": distribution(first10, w, cfg),
        "neutral_all": distribution(df.filter(pl.col("neutral")), w, cfg),
        "protecting_a_late_lead": lead_late(df, w, cfg),
        "protecting_a_lead_whole_fourth_quarter": lead_late(df, w, cfg, max_seconds=900),
        "short_yardage_neutral": short_yardage(df, w, cfg),
    }


# --------------------------------------------------------------------------------------
# Points raised in the public argument
# --------------------------------------------------------------------------------------


def debate(df: pl.DataFrame, w: np.ndarray, cfg: dict, res: dict) -> dict:
    """Four claims from the public argument, each with our numbers: the league-wide gap
    and how many teams show it, "the run sets up the pass", and "teams care about
    winning, not EPA" (how much running is late-lead clock play)."""
    level, c = cfg["level"], cfg["debate"]
    last = int(df["season"].max())
    by_season = {r["season"]: r for r in res["puzzle"]["by_season_all_plays"]}
    out: dict = {
        "league_gap": {
            "most_recent_season": last,
            "most_recent_season_all_plays": by_season[last]["epa"],
            "all_seasons_all_plays": res["puzzle"]["samples"]["all_plays"]["epa"],
            "most_recent_season_neutral": next(
                r for r in res["puzzle"]["by_season_neutral"] if r["season"] == last
            )["epa"],
        }
    }
    run, pas = pl.col("is_pass") == 0, pl.col("is_pass") == 1
    teams = df.group_by("season", "posteam").agg(
        (pl.col("epa").filter(pas).mean() - pl.col("epa").filter(run).mean()).alias("gap"),
        pl.len().alias("n"),
    )
    out["teams_with_higher_pass_epa"] = [
        {"season": s, "teams": t.height, "pass_above_run": int((t["gap"] > 0).sum()),
         "by_large_margin": int((t["gap"] >= c["team_gap_large"]).sum()),
         "median_gap": float(t["gap"].median()), "sd_of_team_gaps": float(t["gap"].std())}
        for s in sorted(teams["season"].unique().to_list())
        for t in [teams.filter(pl.col("season") == s)]
    ]  # fmt: skip
    nd_last = df.filter(pl.col("neutral") & (pl.col("season") == last))
    group, labels = named_situations(nd_last)
    out["league_gap"]["most_recent_season_neutral_by_situation"] = rp.frame_compare(
        nd_last, group, len(labels), w, level, ["epa", "success"]
    ).rows(labels, key="situation")

    # "The run sets up the pass."
    ne = df.filter(pl.col("neutral") & pl.col("early"))
    nb = len(cfg["dist_edges"])
    setup: dict = {}
    d = ne.filter(pl.col("prev_is_pass").is_not_null())
    prev_run = 1.0 - np_col(d, "prev_is_pass")
    x = np.hstack([
        np.ones((d.height, 1)), prev_run[:, None],
        rp.dummies((np_col(d, "down").astype(int) - 1) * nb + np_col(d, "dist_bin")),
        rp.dummies(np_col(d, "field_bin")), rp.dummies(np_col(d, "season")),
    ])  # fmt: skip
    for s, call in enumerate(rp.CALLS):
        m = np_col(d, "is_pass") == s
        fit = rp.cluster_ols(x[m], np_col(d, "epa")[m], np.ones(int(m.sum())),
                             np_col(d, "game_code")[m], w, level)  # fmt: skip
        setup[f"{call}_epa_after_a_run_minus_after_a_pass_same_situation"] = {
            "n": int(m.sum()), "value": float(fit["coef"][1]), "low": float(fit["low"][1]),
            "high": float(fit["high"][1]),
        }  # fmt: skip
    setup["note_previous_play"] = (
        "previous called play of the same drive; down, distance bin, field bin and season "
        "held fixed; neutral early downs"
    )
    tg = rp.team_game_table(ne).filter(
        pl.col("loo_run_epa").is_not_null() & pl.col("loo_pass_epa").is_not_null()
        & pl.col("pass_epa").is_not_null()
    )  # fmt: skip
    one, season_d = np.ones((tg.height, 1)), rp.dummies(np_col(tg, "season"))
    run_eff = np_col(tg, "loo_run_epa")[:, None] / 0.10
    run_rate = (1.0 - np_col(tg, "loo_pass_rate"))[:, None] / 0.10
    pass_eff = np_col(tg, "loo_pass_epa")[:, None]
    y, n, game = np_col(tg, "pass_epa"), np_col(tg, "n_pass").astype(float), np_col(tg, "game_code")
    for name, xs in (
        ("pass_epa_per_0.1_run_epa_in_other_games", np.hstack([one, run_eff, season_d])),
        ("same_controlling_for_pass_epa_in_other_games",
         np.hstack([one, run_eff, season_d, pass_eff])),
        ("pass_epa_per_10_points_of_run_rate_in_other_games",
         np.hstack([one, run_rate, season_d])),
        ("same_controlling_for_pass_and_run_epa_in_other_games",
         np.hstack([one, run_rate, season_d, pass_eff, run_eff])),
    ):  # fmt: skip
        fit = rp.cluster_ols(xs, y, n, game, w, level)
        setup[name] = {"value": float(fit["coef"][1]), "low": float(fit["low"][1]),
                       "high": float(fit["high"][1])}  # fmt: skip
    setup["team_games"] = tg.height
    out["run_sets_up_the_pass"] = setup

    # "Teams care about winning, not EPA": how much running is late-lead clock play?
    lead = np_col(df, "score_differential") >= c["clock_min_lead"]
    q4 = np_col(df, "qtr") == c["clock_qtr"]
    secs = np_col(df, "game_seconds_remaining")
    is_run = np_col(df, "is_pass") == 0
    windows = {
        "leading_fourth_quarter": lead & q4,
        "leading_last_5_minutes": lead & q4 & (secs <= 300),
        "leading_last_2_minutes": lead & q4 & (secs <= 120),
    }
    clock: dict = {"windows": {}}
    for name, m in windows.items():
        clock["windows"][name] = {
            "plays": int(m.sum()),
            "run_share_in_window": float(is_run[m].mean()),
            "share_of_all_runs": float((m & is_run).sum() / is_run.sum()),
            "share_of_all_called_plays": float(m.mean()),
        }
    lo, hi = c["check_lead"]
    score = np_col(df, "score_differential")
    chk = (q4 & (secs <= c["check_last_seconds"]) & (np_col(df, "down") <= 3)
           & (score >= lo) & (score <= hi))  # fmt: skip
    clock["check_against_endgame_clock_note"] = {
        "definition": "downs 1 to 3, fourth quarter, last 2:00, leading by 1 to 8",
        "plays": int(chk.sum()),
        "run_share_scrambles_as_passes": float(is_run[chk].mean()),
        "run_share_as_recorded": float((df["play_type"].to_numpy()[chk] == "run").mean()),
        "endgame_clock_note_value": 0.874,
    }
    all_c = rp.frame_compare(df, mask_group(np.ones(df.height, dtype=bool)), 1, w, level, ["epa"])
    clock["epa_gap_all_plays"] = rp.summary(all_c.est[0, 2, 0], all_c.boot[:, 0, 2, 0], level)
    clock["pass_rate_all_plays"] = float(1 - is_run.mean())
    for name, m in windows.items():
        cc = rp.frame_compare(df, mask_group(~m), 1, w, level, ["epa"])
        clock[f"epa_gap_without_{name}"] = rp.summary(cc.est[0, 2, 0], cc.boot[:, 0, 2, 0], level)
        clock[f"change_in_gap_without_{name}"] = rp.summary(
            cc.est[0, 2, 0] - all_c.est[0, 2, 0], cc.boot[:, 0, 2, 0] - all_c.boot[:, 0, 2, 0],
            level)  # fmt: skip
        clock[f"pass_rate_without_{name}"] = float(1 - is_run[~m].mean())
    out["clock_play"] = clock
    return out


# --------------------------------------------------------------------------------------
# 5. The verdict
# --------------------------------------------------------------------------------------


def verdict(df: pl.DataFrame, w: np.ndarray, cfg: dict, res: dict) -> dict:
    level, min_n = cfg["level"], cfg["cells"]["min_n_per_call"]
    ne = df.filter(pl.col("neutral") & pl.col("early"))
    seasons = sorted(df["season"].unique().to_list())
    ns, nb = len(seasons), len(cfg["dist_edges"])
    # Per season, hold down and distance bin fixed (the full cells are too thin by season).
    k = 2 * nb
    coarse = (np_col(ne, "down").astype(int) - 1) * nb + np_col(ne, "dist_bin")
    code = np_col(ne, "season") - seasons[0]
    comp = rp.frame_compare(ne, code * k + coarse, ns * k, w, level, ["epa"])
    raw = rp.frame_compare(ne, code, ns, w, level, ["epa"])
    rows, std_boot = [], []
    for i, s in enumerate(seasons):
        sel = np.zeros(ns * k, dtype=bool)
        sel[i * k : (i + 1) * k] = True
        g = rp.standardized_gap(comp, "epa", 30, "all", level, sel)
        std_boot.append(g["boot"])
        rows.append({
            "season": s, "pass_rate": float(raw.n[i, 1] / raw.n[i].sum()),
            "gap_raw": rp.summary(raw.est[i, 2, 0], raw.boot[:, i, 2, 0], level),
            "gap_same_down_and_distance": g["standardized"],
        })  # fmt: skip
    std_boot = np.array(std_boot).T  # (n_boot, seasons)
    std_point = np.array([r["gap_same_down_and_distance"]["value"] for r in rows])
    t = np.array(seasons, dtype=float) - np.mean(seasons)

    def slope(y: np.ndarray) -> np.ndarray:
        return (y * t).sum(axis=-1) / (t**2).sum() * 10

    rate_b = raw.boot_n[:, :, 1] / raw.boot_n.sum(axis=-1)
    rate = raw.n[:, 1] / raw.n.sum(axis=-1)
    first, last = slice(0, 3), slice(ns - 3, ns)
    out: dict = {
        "neutral_early_downs_by_season": rows,
        "trend_per_decade": {
            "gap_raw": rp.summary(float(slope(raw.est[:, 2, 0])), slope(raw.boot[:, :, 2, 0]),
                                  level),
            "gap_same_down_and_distance": rp.summary(float(slope(std_point)), slope(std_boot),
                                                     level),
            "pass_rate": rp.summary(float(slope(rate)), slope(rate_b), level),
            "run_epa": rp.summary(float(slope(raw.est[:, 0, 0])), slope(raw.boot[:, :, 0, 0]),
                                  level),
            "pass_epa": rp.summary(float(slope(raw.est[:, 1, 0])), slope(raw.boot[:, :, 1, 0]),
                                   level),
        },
        "first_three_seasons_vs_last_three": {
            "seasons": [seasons[:3], seasons[-3:]],
            "gap_same_down_and_distance_first": rp.summary(
                float(std_point[first].mean()), std_boot[:, first].mean(axis=1), level),
            "gap_same_down_and_distance_last": rp.summary(
                float(std_point[last].mean()), std_boot[:, last].mean(axis=1), level),
            "change": rp.summary(
                float(std_point[last].mean() - std_point[first].mean()),
                std_boot[:, last].mean(axis=1) - std_boot[:, first].mean(axis=1), level),
        },
    }  # fmt: skip

    # The decomposition: raw, neutral, situation fixed.
    p, m = res["puzzle"], res["marginal"]["all_seasons"]
    out["how_much_survives"] = {
        "1_raw_all_plays": p["samples"]["all_plays"]["epa"]["pass_minus_run"],
        "2_neutral_game_script": p["samples"]["neutral"]["epa"]["pass_minus_run"],
        "3_neutral_same_down_distance_field": m["gap_situation_fixed_all_downs"]["standardized"],
        "4_neutral_early_downs_same_cell": m["gap_situation_fixed_early_downs"]["standardized"],
        "unit": "nflfastR EPA per play, pass minus run",
    }

    # Rough upper bound: shift a few percent of early-down neutral calls from run to pass.
    recent = ne.filter(pl.col("season") >= cfg["recent_first_season"])
    c = rp.frame_compare(recent, np_col(recent, "cell"), rp.n_cells(cfg), w, level, ["epa"])
    g = rp.standardized_gap(c, "epa", min_n, "run", level)
    games_b = w[:, np.unique(np_col(recent, "game_code"))].sum(axis=1).astype(float)
    n_games = int(recent["game_code"].n_unique())
    per_team_game = recent.height / (2 * n_games)
    per_team_game_b = c.boot_n.sum(axis=(1, 2)) / (2 * games_b)
    share = cfg["shift"]["share_points"]
    out["points_left_on_the_table_upper_bound"] = {
        "seasons": sorted(recent["season"].unique().to_list()),
        "neutral_early_down_calls_per_team_game": per_team_game,
        "neutral_early_down_runs_per_team_game": float(
            (recent["is_pass"] == 0).sum() / (2 * n_games)
        ),
        "pass_rate": float(recent["is_pass"].mean()),
        "share_of_calls_shifted": share,
        "plays_shifted_per_team_game": share * per_team_game,
        "gap_per_shifted_play": g["standardized"],
        "expected_points_per_team_game": rp.summary(
            share * per_team_game * g["standardized"]["value"],
            share * per_team_game_b * g["boot"], level,
        ),
        "caveats": (
            "nflfastR EPA units. Assumes the shifted runs would gain what the average pass "
            "gains in the same down, distance and field cell and that defenses do not "
            "respond; any response lowers it, so read it as an upper bound. The interval "
            "covers sampling noise only. Covers neutral early downs only."
        ),
    }
    return out


# --------------------------------------------------------------------------------------
# 6. The hook
# --------------------------------------------------------------------------------------


def hook(full: pl.DataFrame, cfg: dict) -> dict:
    """Fit the run and pass outcome tables on development seasons, check them on the
    test seasons, and save the bundle."""
    h = cfg["hook"]
    test_seasons = list(splits.test_seasons())
    dev = full.filter(~pl.col("season").is_in(test_seasons))
    splits.assert_no_test_seasons(dev, "q2 run and pass outcome tables")
    model = rp.RunPassOutcomes(h["min_cell_n"], h["min_turnover_n"], h["era_first_season"],
                               cfg["scrambles_as"]).fit(dev)  # fmt: skip
    model.fit_stamp_ = splits.fit_stamp(
        dev.filter(pl.col("season") >= h["era_first_season"]), "q2 run and pass outcome tables"
    )
    HOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, HOOK_PATH)

    splits.load_test_plays(purpose="q2 run versus pass: held-out check of the outcome tables")
    test = rp.prepare(full.filter(pl.col("season").is_in(test_seasons)), cfg).filter(
        pl.col("down") <= 3
    )
    splits.assert_can_score(model.fit_stamp_, test, "q2 run and pass outcome tables")
    cache: dict = {}
    rows = {"p_first": [], "p_to": [], "yards": []}
    for call, dn, tg, yl in zip(test["call"], test["down"], test["ydstogo"],
                                test["yardline_100"], strict=True):  # fmt: skip
        key = (call, int(dn), int(tg), int(yl))
        if key not in cache:
            o = model.outcomes(*key)
            cache[key] = (o["p_first_down"], o["p_turnover"], model.expected_yards(*key))
        for name, v in zip(rows, cache[key], strict=True):
            rows[name].append(v)
    pred = {k: np.array(v) for k, v in rows.items()}
    clean = np_col(test, "turnover") == 0
    converted = ((np_col(test, "first_down") == 1) | (np_col(test, "touchdown") == 1)) & clean
    heldout = {}
    for s, call in enumerate(rp.CALLS):
        m = np_col(test, "is_pass") == s
        heldout[call] = {
            "n": int(m.sum()),
            "first_down_or_touchdown": {"predicted": float(pred["p_first"][m].mean()),
                                        "actual": float(converted[m].mean())},
            "turnover": {"predicted": float(pred["p_to"][m].mean()),
                         "actual": float(np_col(test, "turnover")[m].mean())},
            "yards_without_turnover": {"predicted": float(pred["yards"][m & clean].mean()),
                                       "actual": float(np_col(test, "yards")[m & clean].mean())},
        }  # fmt: skip
    examples = []
    for dn, tg, yl in [(1, 10, 75), (2, 10, 75), (2, 5, 50), (3, 1, 50), (3, 8, 50),
                       (1, 10, 15), (1, 3, 3), (3, 1, 1)]:  # fmt: skip
        row = {"down": dn, "ydstogo": tg, "yardline_100": yl,
               "pass_share": model.pass_share(dn, tg, yl)}  # fmt: skip
        for call in rp.CALLS:
            o = model.outcomes(call, dn, tg, yl)
            row[call] = {
                "p_first_down": o["p_first_down"], "p_touchdown": o["p_touchdown"],
                "p_turnover": o["p_turnover"], "p_safety": o["p_safety"],
                "expected_yards_without_turnover": model.expected_yards(call, dn, tg, yl),
                "sums_to": float(o["p_touchdown"] + o["p_safety"] + o["keep"].sum()
                                 + o["turnover"].sum()),
            }  # fmt: skip
        examples.append(row)
    return {
        "bundle": str(HOOK_PATH.relative_to(DATA.parent)),
        "fit": model.fit_stamp_,
        "plays_fitted": model.n_,
        "heldout_test_seasons": heldout,
        "examples": examples,
        "note": "counts of what happened when teams chose each play type; not what would "
                "happen if a team changed its mix",
    }  # fmt: skip


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _pair(ax: plt.Axes, y: float, row: dict, metric: str = "epa", dy: float = 0.16) -> None:
    """A run dot and a pass dot with their intervals on one horizontal line."""
    for call, off in (("run", dy), ("pass", -dy)):
        v = row[metric][call]
        ax.plot([v["low"], v["high"]], [y + off, y + off], color=COLORS[call], lw=2.2,
                solid_capstyle="butt", zorder=2)  # fmt: skip
        ax.plot(v["value"], y + off, "o", color=COLORS[call], ms=6.5, zorder=3,
                markeredgecolor=style.SURFACE, markeredgewidth=0.8)  # fmt: skip


def _title(ax: plt.Axes, text: str, subtitle: str) -> None:
    """`style.title`, with room for a subtitle of two lines."""
    style.title(ax, text, subtitle)
    if "\n" in subtitle:
        ax.set_title(text, pad=41)


def _legend(ax: plt.Axes, **kw: object) -> None:
    handles = [plt.Line2D([], [], color=COLORS[c], marker="o", lw=2.2, ms=6.5, label=LABELS[c])
               for c in ("pass", "run")]  # fmt: skip
    ax.legend(handles=handles, **{"loc": "lower right", "fontsize": 9.5, **kw})


def figure_main(res: dict) -> None:
    style.apply()
    p, m, v = res["puzzle"], res["marginal"]["all_seasons"], res["verdict"]
    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 1.0], hspace=0.6,
                          wspace=0.62)  # fmt: skip

    # A. the gap survives the filters
    ax = fig.add_subplot(gs[0, 0])
    steps = [
        ("All plays", p["samples"]["all_plays"]),
        ("Neutral game script", p["samples"]["neutral"]),
        ("Neutral, 1st and 2nd down", p["samples"]["neutral_early_downs"]),
    ]
    for i, (_, row) in enumerate(steps):
        _pair(ax, len(steps) - 1 - i, row)
        g = row["epa"]["pass_minus_run"]
        ax.text(0.21, len(steps) - 1 - i, f"gap {g['value']:+.2f}", va="center", ha="right",
                color=style.INK_2, fontsize=9.5)  # fmt: skip
    ax.set_yticks(range(len(steps)), [s[0] for s in steps][::-1])
    ax.set_ylim(-0.6, len(steps) - 0.4)
    ax.set_xlim(-0.12, 0.21)
    ax.axvline(0, color=style.AXIS, lw=1)
    ax.set_xlabel("Expected points added per play (nflfastR EPA)")
    ax.grid(axis="y", visible=False)
    _legend(ax, loc="lower center", bbox_to_anchor=(0.36, -0.02), ncol=2)
    g0 = p["samples"]["all_plays"]["epa"]["pass_minus_run"]["value"]
    g1 = p["samples"]["neutral"]["epa"]["pass_minus_run"]["value"]
    _title(ax, "1. Passes gain more, and game script is not why",
           f"Gap {g0:.2f} points per play on all plays, {g1:.2f} when score and clock are "
           "neutral")

    # B. by down and distance
    ax = fig.add_subplot(gs[:, 1])
    rows = p["neutral_named_situations"] + [res["risk"]["short_yardage_neutral"][
        "goal_line_downs_1_to_3"]]  # fmt: skip
    names = [r.get("situation", r.get("group")) for r in rows]
    names[-1] = "Goal line: " + names[-1] + ",\ndowns 1 to 3"
    for i, row in enumerate(rows):
        y = len(rows) - 1 - i
        _pair(ax, y, row)
        ax.text(1.02, y, f"{row['pass_rate']:.0%}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", color=style.INK_2, fontsize=9.5)  # fmt: skip
    ax.text(1.02, len(rows) - 0.42, "pass share", transform=ax.get_yaxis_transform(), va="bottom",
            ha="left", color=style.MUTED, fontsize=9)  # fmt: skip
    ax.set_yticks(range(len(rows)), names[::-1])
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.axvline(0, color=style.AXIS, lw=1)
    ax.set_xlabel("Expected points added per play (nflfastR EPA), neutral game script")
    ax.grid(axis="y", visible=False)
    _legend(ax, loc="upper left")
    sh = m["share_of_plays"]
    _title(ax, "2. The gap is on 1st and 2nd down, not in short yardage",
           f"Finer cells (down, distance, field position): pass ahead on "
           f"{sh['pass_better']:.0%} of plays,\ntoo close to call on {sh['tie']:.0%}, run "
           f"ahead on {sh['run_better']:.1%} (the rest: too few plays to compare)")

    # C. by season
    ax = fig.add_subplot(gs[1, 0])
    rows = p["by_season_neutral_early_downs"]
    seasons = [r["season"] for r in rows]
    for call in ("pass", "run"):
        val = [r["epa"][call]["value"] for r in rows]
        lo = [r["epa"][call]["low"] for r in rows]
        hi = [r["epa"][call]["high"] for r in rows]
        ax.fill_between(seasons, lo, hi, color=COLORS[call], alpha=0.16, lw=0)
        ax.plot(seasons, val, color=COLORS[call], marker="o", ms=4, label=LABELS[call])
    ax.axhline(0, color=style.AXIS, lw=1)
    ax.set_xticks(seasons[::3])
    ax.set_ylabel("EPA per play (nflfastR)")
    ax.set_xlabel("Season (1st and 2nd down, neutral game script)")
    ax.legend(loc="center left", fontsize=9.5, ncol=2, bbox_to_anchor=(0.0, 0.5))
    fl = v["first_three_seasons_vs_last_three"]
    f0 = fl["gap_same_down_and_distance_first"]["value"]
    f1 = fl["gap_same_down_and_distance_last"]["value"]
    _title(ax, "3. Sixteen seasons on, the gap has not closed",
           f"Gap at the same down and distance: {f0:.2f} in {seasons[0]} to {seasons[2]}, "
           f"{f1:.2f} in {seasons[-3]} to {seasons[-1]}")
    fig.text(0.012, 0.005, "Bars and bands: 95% intervals from resampling whole games. "
             "Scrambles count as passes; kneels and spikes are left out. EPA is nflfastR's "
             "model. 2010 to 2025.", color=style.MUTED, fontsize=9)  # fmt: skip
    style.save(fig, ASSETS / "questions" / "q2_run_vs_pass.png")


def figure_defense(res: dict) -> None:
    style.apply()
    d = res["defense"]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0), gridspec_kw={"wspace": 0.6})

    ax = axes[0]
    rows = d["defenders_in_box"]["neutral_first_and_10_by_box"]
    xs = np.arange(len(rows))
    for call, off in (("run", -0.07), ("pass", 0.07)):
        val = np.array([r["epa"][call]["value"] for r in rows])
        lo = np.array([r["epa"][call]["low"] for r in rows])
        hi = np.array([r["epa"][call]["high"] for r in rows])
        ax.errorbar(xs + off, val, yerr=[val - lo, hi - val], color=COLORS[call], marker="o",
                    ms=6, lw=2, capsize=0, label=LABELS[call])  # fmt: skip
    ax.axhline(0, color=style.AXIS, lw=1)
    ax.set_xticks(xs, [r["defenders_in_box"] for r in rows])
    ax.set_xlabel("Defenders in the box")
    ax.set_ylabel("EPA per play (nflfastR)")
    ax.legend(loc="upper left", fontsize=9.5)
    s = d["defenders_in_box"]["seasons"]
    _title(ax, "Loaded boxes hurt runs, help passes",
           f"1st and 10, neutral game script, {s[0]} to {s[-1]}")

    ax = axes[1]
    t = d["team_tendency"]
    specs = [
        ("across_teams_season_effects", "Comparing teams"),
        ("across_teams_controlling_for_efficiency_in_other_games",
         "Comparing teams of\nequal efficiency"),
        ("within_team_across_seasons", "Same team,\ndifferent seasons"),
    ]  # fmt: skip
    for i, (key, _) in enumerate(specs):
        y = len(specs) - 1 - i
        for call, off in (("run", 0.14), ("pass", -0.14)):
            v = t[f"{call}_epa"][key]
            ax.plot([v["low"], v["high"]], [y + off] * 2, color=COLORS[call], lw=2.2,
                    solid_capstyle="butt")  # fmt: skip
            ax.plot(v["value"], y + off, "o", color=COLORS[call], ms=6.5,
                    markeredgecolor=style.SURFACE, markeredgewidth=0.8)  # fmt: skip
    ax.axvline(0, color=style.AXIS, lw=1)
    ax.set_yticks(range(len(specs)), [s[1] for s in specs][::-1])
    ax.set_ylim(-0.6, len(specs) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in EPA per play for each extra 10 points\nof pass rate in the "
                  "team's other games")  # fmt: skip
    _legend(ax, loc="lower right")
    _title(ax, "Teams that pass more: runs gain little",
           "Neutral 1st and 2nd downs; pass rate measured\nin the team's other games")

    ax = axes[2]
    rows = d["within_game"]["by_recent_pass_share"]
    xs = np.arange(len(rows))
    for call, off in (("run", -0.07), ("pass", 0.07)):
        val = np.array([r["epa"][call]["value"] for r in rows])
        lo = np.array([r["epa"][call]["low"] for r in rows])
        hi = np.array([r["epa"][call]["high"] for r in rows])
        ax.errorbar(xs + off, val, yerr=[val - lo, hi - val], color=COLORS[call], marker="o",
                    ms=6, lw=2, capsize=0, label=LABELS[call])  # fmt: skip
    ax.axhline(0, color=style.AXIS, lw=1)
    ax.set_xticks(xs, [r["recent_pass_share"] for r in rows])
    ax.set_xlabel(f"Passes among the offense's previous "
                  f"{d['within_game']['window_called_plays']} calls in the game")  # fmt: skip
    ax.set_ylabel("EPA per play (nflfastR)")
    ax.legend(loc="center right", fontsize=9.5)
    _title(ax, "After a pass-heavy stretch: no change",
           "Neutral 1st and 2nd downs, 2010 to 2025")
    fig.text(0.012, -0.07, "95% intervals from resampling whole games. Observational: these "
             "show association, not what a defense would do if an offense changed its mix.",
             color=style.MUTED, fontsize=9)  # fmt: skip
    style.save(fig, ASSETS / "questions" / "q2_defense_adjusts.png")


def figure_risk(res: dict) -> None:
    style.apply()
    r = res["risk"]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0),
                             gridspec_kw={"wspace": 0.62, "width_ratios": [1.2, 1, 1]})  # fmt: skip

    ax = axes[0]
    d = r["neutral_first_and_10"]
    for call in ("pass", "run"):
        h = d[f"{call}_yards_histogram"]
        ax.step(h["yards"], np.array(h["share"]) * 100, where="mid", color=COLORS[call],
                label=LABELS[call], lw=2)  # fmt: skip
    ax.set_xlim(-10.6, 40.6)
    ax.set_ylim(0, None)
    ax.set_xlabel("Yards gained (ends hold everything beyond)")
    ax.set_ylabel("Share of plays (%)")
    ax.legend(loc="upper right", fontsize=9.5)
    zero, big = d["no_gain_or_loss"], d["gains_20_or_more"]
    _title(ax, "A pass is a bigger gamble",
           f"1st and 10, neutral. No gain or a loss: {zero['pass']['value']:.0%} of passes, "
           f"{zero['run']['value']:.0%} of runs.\n20 yards or more: "
           f"{big['pass']['value']:.0%} of passes, {big['run']['value']:.1%} of runs")

    ll = r["protecting_a_late_lead"]
    c = ll["definition"]
    ax = axes[1]
    row = ll["defense_has_no_timeouts"]
    for i, call in enumerate(("run", "pass")):
        v = row["seconds_elapsed"][call]
        ax.barh(1 - i, v["value"], height=0.5, color=COLORS[call], label=LABELS[call])
        ax.plot([v["low"], v["high"]], [1 - i] * 2, color=style.INK, lw=1.6)
        ax.text(v["high"], 1 - i, f"  {v['value']:.0f} seconds", va="center", ha="left",
                color=style.INK_2, fontsize=10)  # fmt: skip
    ax.set_yticks([1, 0], ["Run", "Pass"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0, 48)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Game clock used until the next snap (seconds)")
    ax.legend(loc="lower right", fontsize=9.5)
    o = ll["overall"]["defense_timeout_after"]
    _title(ax, "Leading late, a run burns the clock",
           f"Up {c['min_lead']} to {c['max_lead']}, last {c['max_game_seconds'] // 60} "
           "minutes, defense out of timeouts.\nWith timeouts left it spent one after "
           f"{o['run']['value']:.0%} of runs, {o['pass']['value']:.0%} of passes")

    ax = axes[2]
    rows = r["first_and_10_epa_and_win_probability_added"]
    for i, row in enumerate(rows):
        y = len(rows) - 1 - i
        scaled = {"wpa": {k: {a: b * 100 for a, b in row["wpa_nflfastr"][k].items()}
                          for k in ("run", "pass")}}  # fmt: skip
        _pair(ax, y, scaled, "wpa")
    ax.set_yticks(range(len(rows)), [row["situation"].replace(", ", ",\n") for row in rows][::-1])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axvline(0, color=style.AXIS, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Win probability added per play\n(percentage points, nflfastR)")
    _legend(ax, loc="lower right")
    g = ll["same_down_and_distance"]
    wp = g["wpa_nflfastr"]["standardized"]
    _title(ax, "... and the pass loses its edge",
           f"1st and 10. Last {c['max_game_seconds'] // 60} minutes at the same down and "
           f"distance:\npass {g['epa']['standardized']['value']:+.2f} EPA but "
           f"{wp['value'] * 100:+.1f} points of win probability "
           f"({wp['low'] * 100:+.1f} to {wp['high'] * 100:+.1f})")
    fig.text(0.012, -0.07, "95% intervals from resampling whole games. Win probability "
             "added is nflfastR's model, not ours. Which play a team chose is not random, so "
             "the late-lead comparison is suggestive.", color=style.MUTED, fontsize=9)  # fmt: skip
    style.save(fig, ASSETS / "questions" / "q2_risk_and_clock.png")


# --------------------------------------------------------------------------------------


def figures(res: dict) -> None:
    figure_main(res)
    figure_defense(res)
    figure_risk(res)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw the figures from the saved results file")  # fmt: skip
    if parser.parse_args().figures_only:
        figures(read_json(OUT))
        return
    cfg = rp.load_config()
    plays = data.load_plays()
    extra = rp.load_extra(sorted(plays["season"].unique().to_list()), cfg)
    full = rp.with_extra(plays, extra)
    del plays, extra
    df = rp.prepare(full, cfg)
    w = rp.boot_weights(int(df["game_code"].max()) + 1, cfg["n_boot"], cfg["seed"])

    res: dict = {
        "question": "Why do offenses run when passing gains more?",
        "config": cfg,
        "notes": {
            "models_not_ours": "epa, wp, wpa and xpass are nflfastR's models, used as given",
            "intervals": f"{cfg['level']:.0%} percentile intervals from {cfg['n_boot']} "
                         "resamples of whole games",
            "scrambles": f"counted as {cfg['scrambles_as']} (nflverse records them as runs)",
            "seasons": sorted(df["season"].unique().to_list()),
        },
    }  # fmt: skip
    res["puzzle"] = puzzle(df, full, w, cfg)
    res["marginal"] = marginal(df, w, cfg)
    res["defense"] = defense(df, w, cfg)
    res["risk"] = risk(df, w, cfg)
    res["debate"] = debate(df, w, cfg, res)
    res["verdict"] = verdict(df, w, cfg, res)
    res["hook"] = hook(full, cfg)
    write_json(res, OUT)

    figures(read_json(OUT))  # from the file, so the figures show exactly what was saved

    h = res["verdict"]["how_much_survives"]
    for k, v in h.items():
        if isinstance(v, dict):
            print(f"{k}: {v['value']:+.3f} ({v['low']:+.3f} to {v['high']:+.3f})")
    print("share of neutral plays by cell verdict:", res["marginal"]["all_seasons"][
        "share_of_plays"])  # fmt: skip
    pts = res["verdict"]["points_left_on_the_table_upper_bound"]["expected_points_per_team_game"]
    print(f"upper bound, points per team-game: {pts['value']:.2f} "
          f"({pts['low']:.2f} to {pts['high']:.2f})")  # fmt: skip
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
