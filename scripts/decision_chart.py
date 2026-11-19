"""M2: the classic fourth-down decision chart from the one-step engine.

For each game situation in configs/decision.yaml, the best action for every yard line
(1 to 99) and distance (1 to 15).

Leakage discipline: only development seasons are read here (`splits.load_dev_plays`):
the chart states are hypothetical and the speed check uses 2022 to 2023 fourth downs. If
the development component bundle has to be refitted, `decision.load_engine` reads the
test seasons' kickoff rows for the rule tables through `splits.load_test_plays`.

Outputs:
  data/models/components_dev.joblib        outcome models, development seasons (not committed)
  data/processed/decision_chart.parquet    every cell of every panel, all columns
  results/m2_decision_chart.json           summary, the charts as text grids, engine checks
  assets/m2_decision_chart.png             the four panels

Run:  python scripts/decision_chart.py     (under a minute)
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from fourth_and_rl import decision, splits, style
from fourth_and_rl.io import write_json
from fourth_and_rl.paths import ASSETS, DATA, RESULTS

LETTER = {"go": "G", "field_goal": "F", "punt": "P", None: "."}


def text_grid(best: np.ndarray) -> list[str]:
    """One string per distance (1 first); characters run from the own 1 to the opponent 1."""
    letters = np.array(["G", "F", "P"])
    rows = []
    for d in range(best.shape[0]):
        row = best[d, ::-1]
        rows.append("".join("." if np.isnan(v) else letters[int(v)] for v in row))
    return rows


def go_frontier(frame: pl.DataFrame) -> dict[str, int]:
    """Longest distance at which going is the best action, at a few field positions."""
    out = {}
    for name, yl in (("own_20", 80), ("own_35", 65), ("midfield", 50), ("opp_35", 35),
                     ("opp_20", 20), ("opp_5", 5)):  # fmt: skip
        go = frame.filter((pl.col("yardline_100") == yl) & (pl.col("best_action") == "go"))
        out[name] = int(go["ydstogo"].max()) if go.height else 0
    return out


def engine_checks(engine: decision.DecisionEngine, plays: pl.DataFrame, cfg: dict) -> dict:
    """Speed, the cost of thinning the distributions, and the home/away asymmetry.
    `plays` are development plays; nothing is scored against an outcome."""
    fourth = plays.filter((pl.col("down") == 4) & pl.col("season").is_in([2022, 2023]))
    sample = fourth.sample(n=min(10_000, fourth.height), seed=cfg["seed"])
    start = time.perf_counter()
    fast = engine.action_values(sample)
    seconds = time.perf_counter() - start
    stats = dict(engine.last_stats)

    sub = sample.head(2_000)
    exact_cfg = {**cfg["engine"], "compress": {"merge_mass": 0.0, "max_span": 8}}
    exact_engine = decision.DecisionEngine(engine.wp_model, engine.components, exact_cfg)
    start = time.perf_counter()
    exact = exact_engine.action_values(sub)
    exact_seconds = time.perf_counter() - start
    table = lambda v, n: np.column_stack([v[k][:n] for k in ("go_wp", "fg_wp", "punt_wp")])  # noqa: E731
    a, b = table(fast, sub.height), table(exact, sub.height)
    best_a, best_b = decision.summarize_actions(a)[0], decision.summarize_actions(b)[0]

    chart = cfg["chart"]
    ref = {**chart["defaults"], **chart["situations"][0]["state"]}
    grid = lambda home: engine.evaluate(  # noqa: E731
        decision.chart_states(range(1, 100), range(1, 16), {**ref, "is_home": home})
    )
    away, home = grid(0), grid(1)
    return {
        "speed": {
            "states": sample.height,
            "seconds": round(seconds, 2),
            "next_states_built": stats["next_states"],
            "distinct_next_states_scored": stats["unique_next_states"],
            "note": "real fourth downs of 2022-2023, 2 threads, machine shared with other jobs",
        },
        "thinning_error": {
            "states": sub.height,
            "merge_mass": cfg["engine"]["compress"]["merge_mass"],
            "max_abs_wp_difference": float(np.nanmax(np.abs(a - b))),
            "mean_abs_wp_difference": float(np.nanmean(np.abs(a - b))),
            "same_best_action": float(np.mean(best_a == best_b)),
            "exact_seconds": round(exact_seconds, 2),
            "exact_distinct_next_states": exact_engine.last_stats["unique_next_states"],
        },
        "home_away_reference_chart": {
            "cells": away.height,
            "same_best_action": float((away["best_action"] == home["best_action"]).mean()),
            "mean_best_wp_away": float(away["best_wp"].mean()),
            "mean_best_wp_home": float(home["best_wp"].mean()),
            "mean_abs_go_gain_difference": float((away["go_gain"] - home["go_gain"]).abs().mean()),
            "note": "is_home 0 versus 1 at spread 0. The level differs (a known M1 weakness); "
                    "the chart uses is_home = 0.",
        },
    }  # fmt: skip


def figure(panels: list[dict], path: Path) -> None:
    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), panels, strict=True):
        decision.draw_chart(ax, p["best"])
        ax.set_title(p["label"], pad=22, fontsize=12)
        c = p["counts"]
        ax.text(0, 1.015, f"{p['detail']}.  Go {c['go']:.0%}, field goal {c['field_goal']:.0%}, "
                f"punt {c['punt']:.0%} of cells", transform=ax.transAxes, color=style.INK_2,
                fontsize=9.5, va="bottom")  # fmt: skip
    for ax in axes[:, 0]:
        ax.set_ylabel("Yards to go")
    for ax in axes[1]:
        ax.set_xlabel("Field position (own goal line on the left)")
    fig.suptitle("Fourth-down decision chart, one-step engine", x=0.012, y=1.0, ha="left",
                 fontsize=15, fontweight="bold")  # fmt: skip
    fig.text(0.012, 0.955,
             "The action with the highest win probability one play ahead, for every field "
             "position and distance. Point estimates only: no uncertainty shown here.\n"
             "Even matchup, three timeouts each. Models fitted on 2010 to 2023.",
             color=style.INK_2, fontsize=10.5, va="top")  # fmt: skip
    fig.legend(handles=decision.chart_legend_handles(False), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), handlelength=1.4, columnspacing=2.0)  # fmt: skip
    fig.tight_layout(rect=(0, 0.0, 1, 0.935))
    style.save(fig, path)


def main() -> None:
    cfg = decision.load_config()
    plays = splits.load_dev_plays()
    engine = decision.load_engine(plays, cfg)
    fg = engine.components.field_goal
    fg_limit = int(fg.max_kick_distance + fg.tail_yards)
    chart = cfg["chart"]
    yl = range(chart["yardlines"][0], chart["yardlines"][1] + 1)
    dist = range(chart["distances"][0], chart["distances"][1] + 1)

    frames, panels, summary = [], [], {}
    for sit in chart["situations"]:
        state = {**chart["defaults"], **sit["state"]}
        out = engine.evaluate(decision.chart_states(yl, dist, state))
        out = out.with_columns(pl.lit(sit["key"]).alias("situation"))
        frames.append(out)
        idx = out["best_action"].replace_strict(
            {a: i for i, a in enumerate(decision.ACTIONS)}, return_dtype=pl.Float64
        )
        best = decision.to_grid(out["yardline_100"].to_numpy(), out["ydstogo"].to_numpy(),
                                idx.to_numpy(), chart["distances"][1])  # fmt: skip
        counts = {a: float((out["best_action"] == a).mean()) for a in decision.ACTIONS}
        panels.append({**sit, "best": best, "counts": counts})
        close = out.filter(pl.col("margin") < 0.01).height / out.height
        summary[sit["key"]] = {
            "label": sit["label"],
            "detail": sit["detail"],
            "state": state,
            "cells": out.height,
            "share_of_cells_by_best_action": counts,
            "share_of_cells_with_margin_under_1_point": close,
            "median_margin": float(out["margin"].median()),
            "longest_distance_where_go_is_best": go_frontier(out),
            "grid_rows_are_distance_1_to_15_columns_are_own_1_to_opp_1": text_grid(best),
        }

    table = pl.concat(frames)
    out_path = DATA / "processed" / "decision_chart.parquet"
    table.write_parquet(out_path)

    ref = frames[0]
    pick = lambda y, d: ref.filter(  # noqa: E731
        (pl.col("yardline_100") == y) & (pl.col("ydstogo") == d)
    ).select("go_wp", "fg_wp", "punt_wp", "best_action", "margin", "p_convert", "p_fg",
             "wp_if_convert", "wp_if_fail").to_dicts()[0]  # fmt: skip
    result = {
        "seed": cfg["seed"],
        "models": cfg["models"],
        "engine": cfg["engine"],
        "seconds_per_play_used": engine.seconds,
        "p_extra_point": engine.p_xp,
        "kickoff": engine.components.kickoff_info,
        "situations": summary,
        "reference_examples": {
            "fourth_and_1_at_midfield": pick(50, 1),
            "fourth_and_3_at_opp_40": pick(40, 3),
            "fourth_and_4_at_own_20": pick(80, 4),
            "fourth_and_goal_from_2": pick(2, 2),
            "fourth_and_10_at_opp_30": pick(30, 10),
        },
        "checks": engine_checks(engine, plays, cfg),
        "full_table": str(out_path.relative_to(DATA.parent)),
        "notes": [
            "One-step lookahead: each action's value is the expected win probability of the "
            "next state. Nothing after the next snap is planned.",
            f"fg_wp is null where P(make) is 0 (kicks longer than {fg_limit} yards: the "
            f"longest made development kick, {int(fg.max_kick_distance)}, plus a "
            f"{int(fg.tail_yards)}-yard fade) and punt_wp is null inside the opponent's 31.",
            "Point estimates. Many cells are decided by less than one point of win "
            "probability; see results/m3_bootstrap.json for which calls are robust.",
        ],
    }
    write_json(result, RESULTS / "m2_decision_chart.json")
    figure(panels, ASSETS / "m2_decision_chart.png")

    for key, s in summary.items():
        c = s["share_of_cells_by_best_action"]
        print(f"{key:12s} go {c['go']:.1%}  fg {c['field_goal']:.1%}  punt {c['punt']:.1%}  "
              f"margin < 1 point: {s['share_of_cells_with_margin_under_1_point']:.1%}")  # fmt: skip
    ch = result["checks"]
    print(f"speed: {ch['speed']['states']} states in {ch['speed']['seconds']}s; thinning max "
          f"error {ch['thinning_error']['max_abs_wp_difference']:.5f}")  # fmt: skip


if __name__ == "__main__":
    main()
