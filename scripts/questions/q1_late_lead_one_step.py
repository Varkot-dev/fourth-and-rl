"""Burning question 1, the one-step answer: protecting a late lead.

Leading by 4 (also 1, 3, 7) with about 4:00 left (also 2:00, 6:00), fourth down in your
own half: go for it or punt? This script gives the answer of the one-step engine (M2)
with its bootstrap confidence (M3), nfl4th's numbers for the same situations where the
reference has them, and what coaches actually did in comparable real situations.

A one-step win-probability lookup cannot see clock burn after a conversion, the
opponent's timeouts or who gets the last possession; the full-game dynamic program
(M4) answers that part. See docs/notes/m3_bootstrap.md, section "Burning question 1".

Needs the bootstrap replicates (run scripts/bootstrap_decisions.py first).

Outputs:
  results/questions/q1_late_lead_one_step.json
  assets/questions/q1_late_lead_one_step.png

Run:  python scripts/questions/q1_late_lead_one_step.py     (a few minutes)
      python scripts/questions/q1_late_lead_one_step.py --n-boot 20   (trial, *_trial files)
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from fourth_and_rl import bootstrap, data, decision, style, wp
from fourth_and_rl.io import write_json
from fourth_and_rl.paths import ASSETS, RESULTS, ROOT

DIST_GROUPS = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6-10", 6, 10),
               ("11+", 11, 99)]  # fmt: skip


def clock(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def grid_states(q: dict) -> pl.DataFrame:
    """Every (timeouts, lead, time, yard line, distance) cell, as fourth-down states."""
    yl = np.arange(q["yardlines"][0], q["yardlines"][1] + 1)
    d = np.arange(q["distances"][0], q["distances"][1] + 1)
    to, lead, sec, yy, dd = np.meshgrid(q["defteam_timeouts"], q["leads"], q["seconds"], yl, d,
                                        indexing="ij")  # fmt: skip
    states = wp.state_frame(
        down=4, ydstogo=dd.ravel(), yardline_100=yy.ravel(),
        game_seconds_remaining=sec.ravel(), score_differential=lead.ravel(),
        defteam_timeouts_remaining=to.ravel(),
    )  # fmt: skip
    return states.with_columns(pl.Series("lead", lead.ravel()), pl.Series("seconds", sec.ravel()))


def coach_counts(plays: pl.DataFrame, c: dict) -> dict:
    """What coaches did on comparable real fourth downs (all seasons in the table)."""
    base = plays.filter(
        (pl.col("down") == 4) & (pl.col("qtr") == 4)
        & pl.col("score_differential").is_between(*c["lead"])
        & pl.col("game_seconds_remaining").is_between(*c["seconds"])
        & pl.col("yardline_100").is_between(*c["yardline_100"])
        & pl.col("fourth_down_decision").is_not_null()
    )  # fmt: skip

    def table(frame: pl.DataFrame) -> dict:
        out = {}
        for name, lo, hi in DIST_GROUPS:
            sub = frame.filter(pl.col("ydstogo").is_between(lo, hi))
            went = sub.filter(pl.col("fourth_down_decision") == "go")
            out[name] = {
                "n": sub.height,
                "go": went.height,
                "punt": sub.filter(pl.col("fourth_down_decision") == "punt").height,
                "field_goal": sub.filter(pl.col("fourth_down_decision") == "field_goal").height,
                "converted_when_went": int(
                    ((went["first_down"] == 1) | (went["touchdown"] == 1)).sum()
                ),
            }
        return out

    return {
        "definition": {**c, "quarter": 4, "seasons": [int(plays["season"].min()),
                                                      int(plays["season"].max())]},
        "n": base.height,
        "by_distance": table(base),
        "by_distance_lead_4_to_8": table(base.filter(pl.col("score_differential") >= 4)),
        "by_distance_since_2021": table(base.filter(pl.col("season") >= 2021)),
    }  # fmt: skip


def figure(panels: dict, q: dict, cfg: dict, coaches: dict, nfl4th: pl.DataFrame, path) -> None:
    style.apply()
    leads, secs = q["leads"], sorted(q["seconds"], reverse=True)
    fig = plt.figure(figsize=(13.5, 14.2))
    gs = fig.add_gridspec(len(leads) + 1, len(secs), height_ratios=[1] * len(leads) + [0.95],
                          hspace=0.62, wspace=0.12, top=0.885, bottom=0.085, left=0.065,
                          right=0.985)  # fmt: skip
    for i, lead in enumerate(leads):
        for j, sec in enumerate(secs):
            ax = fig.add_subplot(gs[i, j])
            p = panels[(3, lead, sec)]
            decision.draw_chart(ax, p["best"], p["category"])
            ax.set_xlim(0.5, 50.5)
            ax.set_xticks([10, 20, 30, 40, 50])
            ax.set_xticklabels(["Own 10", "20", "30", "40", "50"])
            ax.set_yticks([1, 3, 5, 7, 9])
            head = (lead, sec) == (q["headline"]["lead"], q["headline"]["seconds"])
            ax.set_title(f"Up {lead}, {clock(sec)} left" + ("  (the question)" if head else ""),
                         fontsize=11.5, pad=6)  # fmt: skip
            if head:
                for side in ("left", "bottom", "top", "right"):
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_color(style.INK)
                    ax.spines[side].set_linewidth(1.6)
            if j == 0:
                ax.set_ylabel("Yards to go")
            else:
                ax.tick_params(labelleft=False)
            ref = nfl4th.filter(
                (pl.col("score_differential") == lead)
                & (pl.col("quarter_seconds_remaining") == sec)
                & (pl.col("defteam_timeouts_remaining") == 3)
                & pl.col("nfl4th_go_wp").is_not_null()
            )
            for r in ref.to_dicts():
                letter = "G" if r["nfl4th_go_wp"] > r["nfl4th_punt_wp"] else "P"
                ax.text(100 - r["yardline_100"], r["ydstogo"], letter, ha="center", va="center",
                        fontsize=7.5, fontweight="bold", color=style.INK, zorder=6,
                        bbox={"boxstyle": "round,pad=0.18", "facecolor": style.SURFACE,
                              "edgecolor": style.INK_2, "linewidth": 0.6})  # fmt: skip

    ax = fig.add_subplot(gs[len(leads), :])
    groups = [g for g in coaches["by_distance"] if coaches["by_distance"][g]["n"] > 0]
    go = np.array([coaches["by_distance"][g]["go"] for g in groups])
    n = np.array([coaches["by_distance"][g]["n"] for g in groups])
    x = np.arange(len(groups))
    ax.bar(x, go / n, color=style.ACTION_COLORS["go"], width=0.6)
    for xi, g_, n_ in zip(x, go, n, strict=True):
        ax.text(xi, g_ / n_ + 0.012, f"went {g_} of {n_}", ha="center", color=style.INK,
                fontsize=10)  # fmt: skip
    ax.set_xticks(x)
    ax.set_xticklabels([f"4th and {g}" for g in groups])
    ax.set_ylim(0, max(0.3, float((go / n).max()) + 0.07))
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels(["0%", "10%", "20%", "30%"])
    ax.set_ylabel("Share that went for it")
    ax.grid(axis="x", visible=False)
    c = coaches["definition"]
    ax.set_title(
        f"What coaches did: {coaches['n']} real fourth downs, {c['seasons'][0]} to "
        f"{c['seasons'][1]}, leading by {c['lead'][0]} to {c['lead'][1]}, "
        f"{clock(c['seconds'][0])} to {clock(c['seconds'][1])} left, own half",
        fontsize=11.5, pad=8)  # fmt: skip

    t = cfg["thresholds"]
    fig.suptitle("Up late, fourth down in your own half: go or punt? The one-step answer",
                 x=0.012, y=0.985, ha="left", fontsize=15, fontweight="bold")  # fmt: skip
    fig.text(0.012, 0.966,
             "Best action one play ahead, by field position and yards to go; even matchup, "
             "three timeouts each.\n"
             f"Full color: {t['confident']:.0%} or more of {cfg['n_boot']} bootstrap refits "
             f"agree; pale: {t['lean']:.0%} to {t['confident']:.0%}; grey: too close to call. "
             "Boxed letters: nfl4th's call for the same situation (G go, P punt).\n"
             "A one-step lookup cannot see clock burn after a conversion or the opponent's "
             "timeouts, and our win-probability model is too cautious about\nsmall leads "
             "inside two minutes, so read the 2:00 column with care.",
             color=style.INK_2, fontsize=10, va="top")  # fmt: skip
    fig.legend(handles=decision.chart_legend_handles(True, ("go", "punt")), loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, 0.012), handlelength=1.4, columnspacing=1.8)
    style.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-boot", type=int, default=None,
                        help="use only the first N replicates and write *_trial files")
    args = parser.parse_args()
    cfg = bootstrap.load_config()
    dcfg = decision.load_config()
    suffix = "_trial" if args.n_boot and args.n_boot != cfg["n_boot"] else ""
    cfg["n_boot"] = args.n_boot or cfg["n_boot"]
    q, t = cfg["q1"], cfg["thresholds"]
    n_boot = int(cfg["n_boot"])
    directory = bootstrap.boot_dir(cfg)
    bootstrap.assert_replicates_current(directory, n_boot, cfg, wp.load_config())

    plays = data.load_plays()
    engine = decision.load_engine(plays, dcfg)
    states = grid_states(q)
    out = engine.evaluate(states)
    point = np.column_stack([out[c].fill_null(np.nan).to_numpy()
                             for c in ("go_wp", "fg_wp", "punt_wp")])  # fmt: skip
    boot = bootstrap.replicate_values(states, directory, range(n_boot), dcfg["engine"], "q1")
    s = bootstrap.summarize(point, boot, t["confident"], t["lean"], tuple(cfg["gain_interval"]))

    table = out.with_columns(
        pl.Series("boot_pct", s["boot_pct"]),
        pl.Series("category", s["category"].tolist(), dtype=pl.String),
        pl.Series("go_gain_lo", s["gain_lo"]),
        pl.Series("go_gain_hi", s["gain_hi"]),
        (100 - pl.col("yardline_100")).alias("own_yardline"),
    )
    n_d = q["distances"][1]
    panels, panel_out = {}, {}
    for to in q["defteam_timeouts"]:
        for lead in q["leads"]:
            for sec in q["seconds"]:
                sel = (
                    (table["defteam_timeouts_remaining"] == to) & (table["lead"] == lead)
                    & (table["seconds"] == sec)
                ).to_numpy()  # fmt: skip
                sub = table.filter(pl.Series(sel))
                yl, d = sub["yardline_100"].to_numpy(), sub["ydstogo"].to_numpy()
                panels[(to, lead, sec)] = {
                    "best": decision.to_grid(yl, d, s["best"][sel].astype(float), n_d),
                    "category": decision.to_grid(yl, d, s["category"][sel], n_d, fill=None),
                }
                by_distance = {}
                for dist in range(q["distances"][0], n_d + 1):
                    row = sub.filter(pl.col("ydstogo") == dist)
                    go = row.filter(pl.col("best_action") == "go")
                    by_distance[str(dist)] = {
                        "cells_go_is_best": go.height,
                        "of_cells": row.height,
                        "go_confident": go.filter(pl.col("category") == "confident").height,
                        "go_lean": go.filter(pl.col("category") == "lean").height,
                        "punt_confident": row.filter((pl.col("best_action") == "punt")
                                                     & (pl.col("category") == "confident")).height,
                        "too_close_to_call": row.filter(pl.col("category") == "uncertain").height,
                        "go_from_own_yardline_at_least": int(go["own_yardline"].min())
                        if go.height else None,
                        "mean_go_gain_points": float(100 * row["go_gain"].mean()),
                    }  # fmt: skip
                panel_out[f"up_{lead}__{sec}s__opp_timeouts_{to}"] = {
                    "lead": lead, "seconds": sec, "defteam_timeouts": to, "cells": sub.height,
                    "category_shares": bootstrap.category_shares(s["category"][sel],
                                                                 s["best"][sel]),
                    "share_go_is_best": float((sub["best_action"] == "go").mean()),
                    "by_distance": by_distance,
                }  # fmt: skip

    # The question as asked: up 4, 4:00, three timeouts each, at a few field positions.
    h = q["headline"]
    head = table.filter(
        (pl.col("lead") == h["lead"]) & (pl.col("seconds") == h["seconds"])
        & (pl.col("defteam_timeouts_remaining") == 3)
        & pl.col("own_yardline").is_in([20.0, 30.0, 40.0, 50.0])
        & pl.col("ydstogo").is_in([1.0, 2.0, 3.0, 5.0, 10.0])
    ).sort("own_yardline", "ydstogo")  # fmt: skip
    headline_rows = [
        {
            "own_yardline": int(r["own_yardline"]), "ydstogo": int(r["ydstogo"]),
            "go_wp": round(r["go_wp"], 4), "punt_wp": round(r["punt_wp"], 4),
            "go_gain_points": round(100 * r["go_gain"], 2),
            "go_gain_90pct_interval_points": [round(100 * r["go_gain_lo"], 2),
                                              round(100 * r["go_gain_hi"], 2)],
            "best_action": r["best_action"], "boot_pct": round(r["boot_pct"], 3),
            "category": r["category"], "p_convert": round(r["p_convert"], 3),
            "wp_if_convert": round(r["wp_if_convert"], 4),
            "wp_if_fail": round(r["wp_if_fail"], 4),
        }
        for r in head.to_dicts()
    ]  # fmt: skip

    # nfl4th's numbers for the same situations, side by side (never mixed with ours).
    ref = pl.read_csv(ROOT / dcfg["compare"]["reference_scenarios"], null_values="NA")
    ours = table.select(
        pl.col("ydstogo").cast(pl.Int64), pl.col("yardline_100").cast(pl.Int64),
        pl.col("lead").cast(pl.Int64).alias("score_differential"),
        pl.col("seconds").cast(pl.Int64).alias("quarter_seconds_remaining"),
        pl.col("defteam_timeouts_remaining").cast(pl.Int64),
        pl.col("go_wp").alias("ours_go_wp"), pl.col("punt_wp").alias("ours_punt_wp"),
        pl.col("best_action").alias("ours_best_action"), pl.col("category").alias("ours_category"),
        pl.col("boot_pct").alias("ours_boot_pct"),
    )  # fmt: skip
    side = ref.join(ours, on=["ydstogo", "yardline_100", "score_differential",
                              "quarter_seconds_remaining", "defteam_timeouts_remaining"],
                    how="left", validate="1:1")  # fmt: skip
    have = side.filter(pl.col("nfl4th_go_wp").is_not_null())
    nfl4th_go = have["nfl4th_go_wp"] > have["nfl4th_punt_wp"]
    ours_go = have["ours_best_action"] == "go"
    conf = have["ours_category"] == "confident"
    side_summary = {
        "source": "nfl4th 1.0.7, results/reference/nfl4th_scenarios.csv; nfl4th's numbers",
        "rows": side.height,
        "rows_with_an_nfl4th_go_value": have.height,
        "note": "nfl4th returned NA for go_wp at yardline_100 50 and 60, so only the own 20 "
                "and own 30 can be compared.",
        "nfl4th_says_go": float(nfl4th_go.mean()),
        "ours_says_go": float(ours_go.mean()),
        "same_call": float((nfl4th_go == ours_go).mean()),
        "same_call_where_ours_is_confident": float((nfl4th_go == ours_go).filter(conf).mean()),
        "n_where_ours_is_confident": int(conf.sum()),
        "rows_table": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for r in side.sort("defteam_timeouts_remaining", "score_differential",
                               "quarter_seconds_remaining", "yardline_100", "ydstogo").to_dicts()
        ],
    }  # fmt: skip

    coaches = coach_counts(plays, q["coaches"])
    key = f"up_{h['lead']}__{h['seconds']}s__opp_timeouts_3"
    result = {
        "question": "Leading late, fourth down in your own half: go for it or punt?",
        "method": "one-step engine (M2) with game-level bootstrap confidence (M3)",
        "seed": int(cfg["seed"]),
        "n_boot": n_boot,
        "thresholds": t,
        "grid": {k: q[k] for k in ("leads", "seconds", "yardlines", "distances",
                                   "defteam_timeouts")},
        "seconds_per_play_used": engine.seconds,
        "headline": {"situation": h, "panel": panel_out[key], "selected_cells": headline_rows},
        "panels": panel_out,
        "nfl4th_side_by_side": side_summary,
        "coaches": coaches,
        "what_one_step_cannot_capture": [
            "Clock burn after a conversion: the engine charges every converted fourth down "
            f"{engine.seconds['go_convert']} seconds and then asks the win-probability model, "
            "which knows only how such leads have turned out on average, not that this team "
            "can now run three more plays and 2 minutes off the clock.",
            "The opponent's timeouts: they enter only through the win-probability model's "
            "timeout features, which the trees barely use. Compare the opp_timeouts_0 and "
            "opp_timeouts_3 panels.",
            "Who gets the last possession, and the opponent scoring quickly and handing the "
            "ball back: nothing is planned beyond the next snap.",
            "The win-probability model itself is too cautious about small leads in the last "
            "two minutes (results/m2_nfl4th_comparison.json, "
            "late_game_win_probability_check_test_seasons), which affects the 2:00 panels.",
            "The bootstrap measures sampling noise only. A bias shared by every refit, such "
            "as the ones above, leaves boot% high and the answer still wrong.",
        ],
    }  # fmt: skip
    write_json(result, RESULTS / "questions" / f"q1_late_lead_one_step{suffix}.json")
    out_png = ASSETS / "questions" / f"q1_late_lead_one_step{suffix}.png"
    figure(panels, q, cfg, coaches, ref, out_png)

    p = panel_out[key]
    print(f"up {h['lead']}, {clock(h['seconds'])} left: go is best in "
          f"{p['share_go_is_best']:.1%} of own-half cells; categories {p['category_shares']}")
    for r in headline_rows:
        print(f"  own {r['own_yardline']:2d}, 4th and {r['ydstogo']:2d}: {r['best_action']:5s} "
              f"gain {r['go_gain_points']:+.1f} {r['go_gain_90pct_interval_points']} "
              f"boot% {r['boot_pct']:.2f} {r['category']}")  # fmt: skip


if __name__ == "__main__":
    main()
