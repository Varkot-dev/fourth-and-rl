"""M3: game-level bootstrap of the fourth-down decisions.

For each of B replicates: draw whole training games with replacement, refit the
win-probability model and the play-outcome models, and re-evaluate (a) every cell of the
decision charts and (b) every real fourth down of the evaluation seasons.

Outputs:
  data/models/boot/seed_<seed>_<within_game>/   refitted models per replicate (not committed)
  data/processed/fourth_down_decisions.parquet  per-play results for the coach evaluation
  results/m3_bootstrap.json                     every number quoted about the bootstrap
  assets/m3_decision_chart_confidence.png       the four chart panels with confidence
  assets/hero_decision_chart.png                the reference chart on a football field

Run:  python scripts/bootstrap_decisions.py              (about 45 minutes the first time;
                                                          resumable, a rerun takes seconds)
      python scripts/bootstrap_decisions.py --n-boot 5   (quick trial, separate outputs)
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Rectangle

from fourth_and_rl import bootstrap, data, decision, style, wp
from fourth_and_rl.io import read_json, write_json
from fourth_and_rl.paths import ASSETS, RESULTS, ROOT

ID_COLUMNS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
              "play_type", "desc"]  # fmt: skip


def chart_frames(dcfg: dict) -> tuple[list[dict], pl.DataFrame]:
    """The states of every chart panel, stacked, plus the panel descriptions."""
    chart = dcfg["chart"]
    yl = range(chart["yardlines"][0], chart["yardlines"][1] + 1)
    dist = range(chart["distances"][0], chart["distances"][1] + 1)
    frames, panels = [], []
    for sit in chart["situations"]:
        frame = decision.chart_states(yl, dist, {**chart["defaults"], **sit["state"]})
        panels.append({**sit, "n": frame.height})
        frames.append(frame)
    return panels, pl.concat(frames)


def real_fourth_downs(plays: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    return plays.filter(
        (pl.col("down") == 4) & pl.col("season").is_between(seasons[0], seasons[1])
    ).sort("game_id", "order_sequence", "play_id")


def point_table(engine: decision.DecisionEngine, states: pl.DataFrame) -> np.ndarray:
    v = engine.action_values(states)
    return np.column_stack([v["go_wp"], v["fg_wp"], v["punt_wp"]])


def gain_summary(gain: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    ok = ~np.isnan(gain)
    width = (hi - lo)[ok]
    return {
        "n": int(ok.sum()),
        "share_abs_gain_under_1_point": float(np.mean(np.abs(gain[ok]) < 0.01)),
        "share_abs_gain_under_2_points": float(np.mean(np.abs(gain[ok]) < 0.02)),
        "share_abs_gain_under_4_points": float(np.mean(np.abs(gain[ok]) < 0.04)),
        "median_interval_width": float(np.median(width)),
        "mean_interval_width": float(np.mean(width)),
        "share_interval_excludes_zero": float(np.mean((lo[ok] > 0) | (hi[ok] < 0))),
    }


def coach_agreement(decisions: pl.DataFrame) -> dict:
    """On confident plays: how often the coach did what the engine recommends."""
    d = decisions.filter(pl.col("actual_decision").is_not_null() & pl.col("category").is_not_null())
    out = {}
    for name, frame in (
        ("confident", d.filter(pl.col("category") == "confident")),
        ("lean", d.filter(pl.col("category") == "lean")),
        ("uncertain", d.filter(pl.col("category") == "uncertain")),
    ):
        should_go = frame.filter(pl.col("best_action") == "go")
        should_kick = frame.filter(pl.col("best_action") != "go")
        out[name] = {
            "n": frame.height,
            "n_should_go": should_go.height,
            "coach_went_when_should_go": float((should_go["actual_decision"] == "go").mean())
            if should_go.height else None,
            "n_should_kick": should_kick.height,
            "coach_kicked_when_should_kick": float((should_kick["actual_decision"] != "go").mean())
            if should_kick.height else None,
        }  # fmt: skip
    return out


def replicate_diagnostics(directory, n_boot: int) -> dict:
    """How much the refitted pieces move across replicates (a few reference numbers)."""
    slope, conv, fg50, punt = [], [], [], []
    for b in range(n_boot):
        eng = bootstrap.load_replicate_engine(directory, b)
        if eng.wp_model.calibration != "none":
            slope.append(eng.wp_model.calibrator.a)
        conv.append(float(eng.components.p_convert(50, 2)))
        fg50.append(float(eng.components.p_field_goal(32)))
        punt.append(float(eng.components.punt_outcomes(70) @ np.arange(100)))
    q = lambda x: {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)),  # noqa: E731
                   "min": float(np.min(x)), "max": float(np.max(x))}  # fmt: skip
    return {
        "platt_slope": q(slope) if slope else "no calibration map (version 2)",
        "p_convert_fourth_and_2_at_midfield": q(conv),
        "p_field_goal_50_yards": q(fg50),
        "expected_opponent_yardline_after_punt_from_own_30": q(punt),
    }


# ---------------------------------------------------------------------------- figures


def figure_confidence(panels: list[dict], grids: list[dict], cfg: dict, path) -> None:
    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex=True, sharey=True)
    for ax, panel, g in zip(axes.ravel(), panels, grids, strict=True):
        decision.draw_chart(ax, g["best"], g["category"])
        sh = g["shares"]
        ax.set_title(panel["label"], pad=22, fontsize=12)
        ax.text(0, 1.015, f"{panel['detail']}.  Confident {sh['confident']:.0%}, lean "
                f"{sh['lean']:.0%}, too close to call {sh['uncertain']:.0%} of cells",
                transform=ax.transAxes, color=style.INK_2, fontsize=9.5, va="bottom")  # fmt: skip
    for ax in axes[:, 0]:
        ax.set_ylabel("Yards to go")
    for ax in axes[1]:
        ax.set_xlabel("Field position (own goal line on the left)")
    t = cfg["thresholds"]
    fig.suptitle("Fourth-down decision chart with bootstrap confidence", x=0.012, y=1.0,
                 ha="left", fontsize=15, fontweight="bold")  # fmt: skip
    fig.text(0.012, 0.955,
             f"Best action one play ahead. Each cell is colored by how many of {cfg['n_boot']} "
             f"game-level bootstrap refits agree with it: full color {t['confident']:.0%} or "
             f"more, pale {t['lean']:.0%} to {t['confident']:.0%}, grey below "
             f"{t['lean']:.0%}.\nEven matchup, three timeouts each. Models fitted on 2010 to "
             "2023.", color=style.INK_2, fontsize=10.5, va="top")  # fmt: skip
    fig.legend(handles=decision.chart_legend_handles(True), loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.035), handlelength=1.4, columnspacing=1.6)  # fmt: skip
    fig.tight_layout(rect=(0, 0.0, 1, 0.935))
    style.save(fig, path)


def figure_hero(panel: dict, g: dict, cfg: dict, path) -> None:
    """The reference chart drawn on a football field."""
    style.apply()
    fig, ax = plt.subplots(figsize=(13, 6.3))
    n_d = g["best"].shape[0]
    decision.draw_chart(ax, g["best"], g["category"])
    ax.set_xlim(-10, 110)
    top = n_d + 0.5
    # End zones and the field outline.
    for x0, label in ((-10, "OWN END ZONE"), (100, "OPPONENT END ZONE")):
        ax.add_patch(Rectangle((x0, 0.5), 10, n_d, facecolor=style.GRID, edgecolor="none",
                               zorder=1))  # fmt: skip
        ax.text(x0 + 5, (0.5 + top) / 2, label, rotation=90 if x0 < 0 else 270, ha="center",
                va="center", color=style.INK_2, fontsize=11, fontweight="bold",
                zorder=4)  # fmt: skip
    ax.add_patch(Rectangle((0, 0.5), 100, n_d, facecolor=style.SURFACE, edgecolor="none",
                           zorder=0))  # fmt: skip
    # Yard lines every 5 yards, heavier every 10, goal lines heaviest.
    for x in range(5, 100, 5):
        ax.axvline(x, color=style.SURFACE, linewidth=1.3 if x % 10 == 0 else 0.6,
                   alpha=0.9 if x % 10 == 0 else 0.55, zorder=3)  # fmt: skip
    for x in (0, 100):
        ax.axvline(x, color=style.INK_2, linewidth=1.6, zorder=5)
    ax.add_patch(Rectangle((-10, 0.5), 120, n_d, fill=False, edgecolor=style.INK_2,
                           linewidth=1.2, zorder=5))  # fmt: skip
    # Field numerals under the chart.
    ax.set_xticks(range(10, 100, 10))
    ax.set_xticklabels(["10", "20", "30", "40", "50", "40", "30", "20", "10"], fontsize=15,
                       fontweight="bold", color=style.INK_2)  # fmt: skip
    ax.tick_params(axis="x", length=0, pad=7)
    ax.set_yticks(range(1, n_d + 1))
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("Yards to go on fourth down")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.text(25, -1.55, "own half", ha="center", va="top", color=style.MUTED, fontsize=10)
    ax.text(75, -1.55, "opponent's half  (offense moves this way  →)", ha="center", va="top",
            color=style.MUTED, fontsize=10)  # fmt: skip
    sh, t = g["shares"], cfg["thresholds"]
    style.title(
        ax,
        "Fourth down: go for it, kick, or punt?",
        f"Best action by field position and yards to go: {panel['label'].lower()}, even "
        f"matchup, one play ahead.\nFull color: {t['confident']:.0%} or more of "
        f"{cfg['n_boot']} bootstrap refits agree ({sh['confident']:.0%} of cells). Pale: "
        f"{t['lean']:.0%} to {t['confident']:.0%} ({sh['lean']:.0%}). Grey: too close to "
        f"call ({sh['uncertain']:.0%}).",
    )
    ax.set_title(ax.get_title(loc="left"), pad=42, fontsize=16)
    fig.legend(handles=decision.chart_legend_handles(True), loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.045), handlelength=1.4, columnspacing=1.6)  # fmt: skip
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    style.save(fig, path)


# ------------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-boot", type=int, default=None,
                        help="replicates (default: config); other values write *_trial files")
    args = parser.parse_args()

    cfg = bootstrap.load_config()
    dcfg = decision.load_config()
    wcfg = wp.load_config()
    trial = args.n_boot is not None and args.n_boot != cfg["n_boot"]
    if trial:
        cfg["n_boot"] = args.n_boot
    suffix = "_trial" if trial else ""
    n_boot, t = int(cfg["n_boot"]), cfg["thresholds"]
    directory = bootstrap.boot_dir(cfg)

    started = time.perf_counter()
    plays = data.load_plays()
    prepared = bootstrap.prepare_plays(plays)
    engine = decision.load_engine(plays, dcfg)
    plan = bootstrap.BootstrapPlan(
        prepared, cfg["train_seasons"], cfg["calib_seasons"], cfg["seed"], cfg["within_game"],
        cfg["game_fraction"],
    )  # fmt: skip

    panels, chart = chart_frames(dcfg)
    real = real_fourth_downs(plays, cfg["evaluate_seasons"])
    states = pl.concat([
        chart.select(wp.STATE_COLUMNS).cast(pl.Float64),
        real.select(wp.STATE_COLUMNS).cast(pl.Float64),
    ])  # fmt: skip
    point = point_table(engine, states)

    # ---- the long part: refit and re-evaluate, one replicate at a time (resumable)
    fit_seconds, eval_seconds, fitted_now = 0.0, 0.0, 0
    boot = np.empty((n_boot, states.height, 3), dtype=np.float32)
    for b in range(n_boot):
        t0 = time.perf_counter()
        fitted = bootstrap.ensure_replicate(
            plan, b, directory, wcfg, None, cfg["kickoff_seasons"], cfg["n_threads"],
            cfg["calibration"],
        )  # fmt: skip
        t1 = time.perf_counter()
        boot[b] = bootstrap.replicate_values(states, directory, [b], dcfg["engine"], "m3_main")[0]
        t2 = time.perf_counter()
        if fitted:
            fit_seconds += t1 - t0
            eval_seconds += t2 - t1
            fitted_now += 1
            print(f"replicate {b + 1}/{n_boot}: fit {t1 - t0:.1f}s, evaluate {t2 - t1:.1f}s",
                  flush=True)  # fmt: skip

    summary = bootstrap.summarize(point, boot, t["confident"], t["lean"],
                                  tuple(cfg["gain_interval"]))  # fmt: skip

    # ---- charts
    grids, chart_out, offset = [], {}, 0
    chart_yl = chart["yardline_100"].to_numpy()
    chart_d = chart["ydstogo"].to_numpy()
    n_d = dcfg["chart"]["distances"][1]
    for panel in panels:
        s = slice(offset, offset + panel["n"])
        offset += panel["n"]
        best = decision.to_grid(chart_yl[s], chart_d[s], summary["best"][s].astype(float), n_d)
        cat = decision.to_grid(chart_yl[s], chart_d[s], summary["category"][s], n_d, fill=None)
        shares = bootstrap.category_shares(summary["category"][s], summary["best"][s])
        grids.append({"best": best, "category": cat, "shares": shares})
        chart_out[panel["key"]] = {
            "label": panel["label"], "detail": panel["detail"], "cells": panel["n"],
            "category_shares": shares,
            "gain": gain_summary(summary["gain"][s], summary["gain_lo"][s], summary["gain_hi"][s]),
            "split_half": bootstrap.split_half_agreement(point[s], boot[:, s], t["confident"],
                                                         t["lean"]),
            "cells_table": {
                "yardline_100": chart_yl[s].astype(int).tolist(),
                "ydstogo": chart_d[s].astype(int).tolist(),
                "best_action": [decision.ACTIONS[i] for i in summary["best"][s]],
                "boot_pct": np.round(summary["boot_pct"][s], 4).tolist(),
                "category": summary["category"][s].tolist(),
                "go_gain": np.round(summary["gain"][s], 5).tolist(),
                "go_gain_lo": np.round(summary["gain_lo"][s], 5).tolist(),
                "go_gain_hi": np.round(summary["gain_hi"][s], 5).tolist(),
            },
        }  # fmt: skip

    # ---- real fourth downs
    r = slice(chart.height, states.height)
    evaluated = engine.evaluate(real, out_of_sample=True, what="M3 real fourth downs")
    nn = lambda x: pl.Series(np.asarray(x, dtype=float)).fill_nan(None)  # noqa: E731
    keep = [c for c in ID_COLUMNS if c in real.columns] + wp.STATE_COLUMNS
    decisions = evaluated.select(
        *keep, "go_wp", "fg_wp", "punt_wp", "best_action", "best_wp", "margin", "go_gain",
        "p_convert", "p_fg", "wp_if_convert", "wp_if_fail",
    ).with_columns(
        nn(summary["boot_pct"][r]).alias("boot_pct"),
        pl.Series(summary["category"][r].tolist(), dtype=pl.String).alias("category"),
        nn(summary["share_go"][r]).alias("boot_share_go"),
        nn(summary["share_field_goal"][r]).alias("boot_share_field_goal"),
        nn(summary["share_punt"][r]).alias("boot_share_punt"),
        nn(summary["gain_lo"][r]).alias("go_gain_lo"),
        nn(summary["gain_median"][r]).alias("go_gain_median"),
        nn(summary["gain_hi"][r]).alias("go_gain_hi"),
        real["fourth_down_decision"].alias("actual_decision"),
        pl.lit(int(cfg["seed"])).alias("boot_seed"),
        pl.lit(n_boot).alias("n_boot"),
    )  # fmt: skip
    played = decisions["actual_decision"].is_not_null().to_numpy()
    scored = decisions["go_wp"].is_not_null().to_numpy()
    use = played & scored
    cat_r, best_r = summary["category"][r], summary["best"][r]
    real_out = {
        "seasons": cfg["evaluate_seasons"],
        "fourth_down_rows": real.height,
        "rows_without_a_real_decision_penalty_or_kneel": int((~played).sum()),
        "rows_not_scored_overtime_or_incomplete_state": int((~scored).sum()),
        "plays_used": int(use.sum()),
        "category_shares": bootstrap.category_shares(cat_r[use], best_r[use]),
        "category_shares_by_actual_decision": {
            a: bootstrap.category_shares(
                cat_r[use & (decisions["actual_decision"] == a).fill_null(False).to_numpy()]
            )
            for a in decision.ACTIONS
        },
        "recommended_action_shares": {
            a: float(np.mean(best_r[use] == i)) for i, a in enumerate(decision.ACTIONS)
        },
        "gain": gain_summary(summary["gain"][r][use], summary["gain_lo"][r][use],
                             summary["gain_hi"][r][use]),
        "coach_agreement": coach_agreement(decisions),
        "split_half": bootstrap.split_half_agreement(point[r][use], boot[:, r][:, use],
                                                     t["confident"], t["lean"]),
    }  # fmt: skip
    if not trial:
        out_path = ROOT / cfg["decisions_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        decisions.write_parquet(out_path)

    total = time.perf_counter() - started
    runtime = {
        "replicates_fitted_in_this_run": fitted_now,
        "mean_fit_seconds": round(fit_seconds / fitted_now, 1) if fitted_now else None,
        "mean_evaluate_seconds": round(eval_seconds / fitted_now, 1) if fitted_now else None,
        "states_evaluated_per_replicate": states.height,
        "total_seconds_this_run": round(total, 1),
        "threads": cfg["n_threads"],
    }
    json_path = RESULTS / f"m3_bootstrap{suffix}.json"
    if fitted_now == 0 and json_path.exists():
        # A rerun from the caches: keep the timings of the run that did the fitting.
        runtime = {**read_json(json_path).get("runtime", runtime),
                   "last_rerun_from_cache_seconds": round(total, 1)}  # fmt: skip
    result = {
        "seed": int(cfg["seed"]),
        "n_boot": n_boot,
        "thresholds": t,
        "gain_interval": cfg["gain_interval"],
        "resampling": {
            "unit": "whole games, with replacement" if cfg["within_game"] == "none"
            else "whole games, then drives within each drawn game",
            "within_game": cfg["within_game"],
            "game_fraction": cfg["game_fraction"],
            "train_seasons": cfg["train_seasons"],
            "train_games": plan.blocks["train"].n_games,
            "calib_seasons": cfg["calib_seasons"],
            "calib_games": plan.blocks["calib"].n_games,
            "calibration": (
                "none: trees grown on all drawn development games (version 2)"
                if cfg["calibration"] == "none"
                else f"{cfg['calibration']} map refitted on the drawn calibration games"
            ),
            "components": "refitted on all drawn games (training and calibration seasons)",
            "kickoff_table_seasons_not_resampled": cfg["kickoff_seasons"],
            "wp_seed_per_replicate": "wp.yaml seed + replicate number",
        },
        "point_estimate": {
            "wp_model": dcfg["models"]["wp_path"],
            "components_fit_seasons": dcfg["models"]["components_fit_seasons"],
            "engine": dcfg["engine"],
        },
        "chart": chart_out,
        "real_fourth_downs": real_out,
        "replicate_diagnostics": replicate_diagnostics(directory, n_boot),
        "runtime": runtime,
        "notes": [
            "boot% is the share of replicates whose best action equals the point estimate's "
            "best action. It measures sampling variability of the fitted models only, not "
            "model bias, so it overstates how sure we can be.",
            "split_half compares categories from the even-numbered and odd-numbered "
            "replicates; each half has about half of B, so it understates the stability of "
            "the full-B categories.",
            "coach_agreement is reported for orientation; the coach evaluation is M7.",
        ],
    }
    write_json(result, json_path)
    figure_confidence(panels, grids, cfg, ASSETS / f"m3_decision_chart_confidence{suffix}.png")
    figure_hero(panels[0], grids[0], cfg, ASSETS / f"hero_decision_chart{suffix}.png")

    print(f"B = {n_boot}, seed {cfg['seed']}, {total / 60:.1f} minutes this run")
    for key, c in chart_out.items():
        s = c["category_shares"]
        print(f"  chart {key:12s} confident {s['confident']:.1%}  lean {s['lean']:.1%}  "
              f"uncertain {s['uncertain']:.1%}")  # fmt: skip
    s = real_out["category_shares"]
    print(f"  real fourth downs ({real_out['plays_used']:,}): confident {s['confident']:.1%}  "
          f"lean {s['lean']:.1%}  uncertain {s['uncertain']:.1%}")  # fmt: skip
    print(f"  split-half category agreement on real plays: "
          f"{real_out['split_half']['category_agreement']:.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
