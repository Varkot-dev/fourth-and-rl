"""M13: goal-line and two-point decisions, read from the exactly solved game (M4).

Three questions argued on X with score arithmetic, never with a model:

  1. Leading late (by 3, 4, 7 or 14), fourth and goal from the 1, 2 or 3 with 2:00 to
     12:00 left: field goal or go? (Daboll kicked up 7; Bowles went for it up 14.)
  2. After a touchdown: kick the extra point or go for two, by the margin before the try
     ("down 8 after a touchdown", that is a team that was down 14 and scored; also down
     2, 5, 9 and 10, and two leads), by time left.
  3. Early in a close game (first half, within 3), fourth and goal: points or go?

For each: a map of the best action and its margin in win-probability points, the headline
case taken apart in plain numbers, and sensitivity rows (conversion probability and
two-point success probability plus or minus 5 points, the defense using timeouts at the
measured rates). Each sensitivity row re-solves the whole game in memory (question 3 is in
the first half, which depends on everything after it); data/mdp is never written.

Run:  .venv/bin/python scripts/questions/q5_goal_line_and_two_point.py
      (needs scripts/solve_mdp.py first; about 5 minutes and 2.5 GB on 2 threads)

Writes results/questions/q5_goal_line_and_two_point.json, assets/questions/q5_goal_line.png
and assets/questions/q5_two_point.png.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import gc  # noqa: E402
import resource  # noqa: E402
from typing import Any  # noqa: E402

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from fourth_and_rl import endgame_questions as eq  # noqa: E402
from fourth_and_rl import mdp, splits, style  # noqa: E402
from fourth_and_rl import mdp_game as mg  # noqa: E402
from fourth_and_rl.components import ComponentModels  # noqa: E402
from fourth_and_rl.io import write_json  # noqa: E402
from fourth_and_rl.paths import ASSETS, CONFIGS, RESULTS  # noqa: E402

OUT = RESULTS / "questions" / "q5_goal_line_and_two_point.json"
FIG_GOAL = ASSETS / "questions" / "q5_goal_line.png"
FIG_TWO = ASSETS / "questions" / "q5_two_point.png"

# Diverging categories: orange = kick (field goal, extra point), blue = go (or go for two).
ORANGES = ("#f6cbb5", "#f0915f", "#c24d1c")
BLUES = (style.SEQUENTIAL[1], style.SEQUENTIAL[3], style.SEQUENTIAL[5])
GREY = "#d8d7d0"


def load_config() -> dict:
    with open(CONFIGS / "q5.yaml") as f:
        return yaml.safe_load(f)


def span(spec: dict) -> list[int]:
    return list(range(spec["first"], spec["last"] + 1, spec.get("step", 1)))


def clock(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def game_clock(seconds: int) -> str:
    """Quarter and clock for game seconds left (3600 = kickoff)."""
    q = 4 - (seconds - 1) // 900
    return f"Q{q} {clock(seconds - (4 - q) * 900)}"


def r4(x: Any) -> Any:
    return np.round(np.asarray(x, dtype=float), 4).tolist()


# --------------------------------------------------------------------------------------
# The three maps
# --------------------------------------------------------------------------------------


def late_map(sol: mg.Solution, q: dict) -> dict[str, np.ndarray]:
    """[time, lead, opponent timeouts, yard line] values of going and kicking."""
    c = q["goal_line_late"]
    extras = [(c["offense_timeouts"], b, 0) for b in c["opponent_timeouts"]]
    return eq.goal_line_values(sol, span(c["seconds"]), c["leads"], c["yardlines"], extras)


def early_map(sol: mg.Solution, q: dict) -> dict[str, np.ndarray]:
    """[time, margin, yard line], averaged over who receives the second-half kickoff."""
    c = q["goal_line_early"]
    v = eq.goal_line_values(
        sol, span(c["seconds"]), c["margins"], c["yardlines"], [(3, 3, 0), (3, 3, 1)]
    )
    out = {k: v[k].mean(axis=2) for k in ("go", "field_goal")}
    out["flag_gap"] = np.abs(
        (v["go"] - v["field_goal"])[:, :, 0] - (v["go"] - v["field_goal"])[:, :, 1]
    )
    out["by_flag"] = {k: v[k] for k in ("go", "field_goal")}
    return out


def try_map(sol: mg.Solution, q: dict, all_timeouts: bool = False) -> dict[str, np.ndarray]:
    """[time, margin] (or [time, margin, 16 timeout pairs]) values of kick and two."""
    c = q["two_point"]
    tos = c["timeouts"]
    extras = (
        [(a, b, 0) for a in range(4) for b in range(4)]
        if all_timeouts
        else [(tos["scorer"], tos["receiver"], 0)]
    )
    tv = eq.try_values(sol, span(c["seconds"]), span(c["margins"]), extras)
    if not all_timeouts:
        tv = {k: x[:, :, 0] for k, x in tv.items()}
    return tv


# --------------------------------------------------------------------------------------
# Headline cases
# --------------------------------------------------------------------------------------


def late_case(sol: mg.Solution, q: dict, which: str) -> dict:
    c = q["goal_line_late"]
    h = c[which]
    state = mg.GameState(
        h["seconds"],
        h["lead"],
        4,
        h["yardline"],
        h["yardline"],
        c["offense_timeouts"],
        h["opponent_timeouts"],
    )
    return {"situation": h, **eq.goal_line_case(sol, state)}


def early_case(sol: mg.Solution, q: dict) -> dict:
    h = q["goal_line_early"]["headline"]
    cases = [
        eq.goal_line_case(
            sol, mg.GameState(h["seconds"], h["margin"], 4, h["yardline"], h["yardline"], 3, 3, r)
        )
        for r in (0, 1)
    ]
    keys = [k for k, v in cases[0].items() if isinstance(v, float)]
    avg = {k: 0.5 * (cases[0][k] + cases[1][k]) for k in keys}
    avg["best"] = "go" if avg["go_minus_field_goal"] > 0 else "field_goal"
    return {
        "situation": {**h, "game_clock": game_clock(h["seconds"])},
        "averaged_over_second_half_kickoff": avg,
        "receives_second_half_kickoff": {"no": cases[0], "yes": cases[1]},
    }


def try_headline(sol: mg.Solution, q: dict) -> dict:
    c = q["two_point"]
    h, tos = c["headline"], c["timeouts"]
    case = eq.try_case(sol, h["seconds"], h["margin"], (tos["scorer"], tos["receiver"], 0))
    m = sol.model
    one_shot = {}
    for shift in q["sensitivity"]["two_point_shifts"]:
        tv = eq.try_values(
            sol,
            [h["seconds"]],
            [h["margin"]],
            [(tos["scorer"], tos["receiver"], 0)],
            p_two=m.p_two + shift,
        )
        one_shot[f"{shift:+.2f}"] = float(tv["two"][0, 0, 0] - tv["kick"][0, 0, 0])
    arith = eq.two_try_arithmetic(m.p_two, m.p_xp)
    return {
        "situation": {**h, **tos},
        **case,
        "one_shot_two_point_shift": {
            "note": "only this try's success probability changed, later tries as in the base "
            "model (the sensitivity rows re-solve the whole game instead)",
            "two_minus_kick": one_shot,
        },
        "score_arithmetic_on_its_own_path": {
            **arith,
            "assumes": "the trailing team scores exactly one more touchdown, the leader "
            "scores nothing more, a tie is worth 0.5 (the model has no overtime)",
        },
    }


def path_shares(sol: mg.Solution, q: dict, seconds: int, margin: int) -> dict:
    """How often the score-arithmetic path happens: simulate the solved game from the
    kickoff after the try (trailing by margin, margin + 1, margin + 2) and count games in
    which the trailing team scores exactly one more touchdown and nothing else, and the
    leader scores nothing more."""
    sim = mg.Simulator(sol)
    m, g = sol.model, sol.grid
    n = q["two_point"]["path_games"]
    tos = q["two_point"]["timeouts"]
    t = eq.tick_of(sol, seconds)
    kick = np.asarray(m.kern.kickoff[1:], dtype=float)
    out = {}
    for d in (0, 1, 2):
        rng = np.random.default_rng(q["seed"] + d)
        yl = 1 + rng.choice(len(kick), size=n, p=kick / kick.sum())
        lead = -(margin + d)
        st = sim.start(n, t=t, s=lead, f=g.first10()[yl], a=tos["receiver"], b=tos["scorer"])
        res = mg.simulate(sim, st, rng, policy="optimal", log_drives=True)
        dr = res["drives"]
        trail = dr["side"] == 1
        td = dr["result"] == mg.DRIVE_RESULTS.index("touchdown")
        fg = dr["result"] == mg.DRIVE_RESULTS.index("field_goal")
        n_td = np.bincount(dr["game"][trail & td], minlength=n)
        n_fg = np.bincount(dr["game"][trail & fg], minlength=n)
        pts = res["points"]
        path = (n_td == 1) & (n_fg == 0) & (pts[:, 0] == 0) & np.isin(pts[:, 1], (6, 7, 8))
        win_trailing = 1.0 - res["result"]
        out[f"trailing_by_{-(margin + d)}"] = {
            "share_of_games_on_the_arithmetic_path": float(path.mean()),
            "trailing_team_wins_simulated": float(win_trailing.mean()),
            "trailing_team_wins_exact": float(
                eq.try_values(sol, [seconds], [margin + d], [(tos["scorer"], tos["receiver"], 0)])[
                    "after"
                ][0, 0, 0, 0]
            ),
            "standard_error": float(win_trailing.std() / np.sqrt(n)),
            "trailing_team_wins_on_the_path": float(win_trailing[path].mean())
            if path.any()
            else None,
            "trailing_team_wins_off_the_path": float(win_trailing[~path].mean()),
        }
    return {"seconds": seconds, "games_per_start": n, **out}


# --------------------------------------------------------------------------------------
# Summaries of the maps
# --------------------------------------------------------------------------------------


def late_summary(v: dict, q: dict) -> dict:
    c = q["goal_line_late"]
    secs = span(c["seconds"])
    d = v["go"] - v["field_goal"]
    out = {}
    for j, lead in enumerate(c["leads"]):
        for y, yl in enumerate(c["yardlines"]):
            for k, b in enumerate(c["opponent_timeouts"]):
                col = d[:, j, k, y]
                go_times = [s for s, x in zip(secs, col, strict=True) if x > 0]
                out[f"up_{lead}__from_{yl}__opp_timeouts_{b}"] = {
                    "go_minus_field_goal_by_seconds": dict(
                        zip(map(str, secs), r4(col), strict=True)
                    ),
                    "share_of_times_go_better": float((col > 0).mean()),
                    "go_better_from_seconds": min(go_times) if go_times else None,
                    "largest_go_edge": float(col.max()),
                    "largest_field_goal_edge": float(-col.min()),
                }
    return out


def try_summary(tv: dict, tv_all: dict, q: dict) -> dict:
    c = q["two_point"]
    secs, margins = span(c["seconds"]), span(c["margins"])
    d = tv["two"] - tv["kick"]
    dall = tv_all["two"] - tv_all["kick"]
    flips = (dall > 0) != (dall[..., :1] > 0)  # against scorer 0 receiver 0
    named = {}
    for mgn in c["named_margins"]:
        i = margins.index(mgn)
        col = d[:, i]
        named[str(mgn)] = {
            "two_minus_kick_by_seconds": dict(zip(map(str, secs), r4(col), strict=True)),
            "share_of_times_two_better": float((col > 0).mean()),
            "largest_two_edge": float(col.max()),
            "largest_kick_edge": float(-col.min()),
            "two_better_at_every_time_for_every_timeout_pair": bool((dall[:, i] > 0).all()),
            "kick_better_at_every_time_for_every_timeout_pair": bool((dall[:, i] <= 0).all()),
        }
    return {
        "margins_where_two_is_better_at_every_listed_time": [
            mgn for i, mgn in enumerate(margins) if (d[:, i] > 0).all()
        ],
        "margins_where_kick_is_better_at_every_listed_time": [
            mgn for i, mgn in enumerate(margins) if (d[:, i] <= 0).all()
        ],
        "share_of_cells_two_better": float((d > 0).mean()),
        "cells_whose_call_depends_on_timeouts": int(flips.any(axis=-1).sum()),
        "cells": int(d.size),
        "named_margins": named,
    }


def early_summary(v: dict, q: dict) -> dict:
    c = q["goal_line_early"]
    secs = span(c["seconds"])
    d = v["go"] - v["field_goal"]
    out: dict[str, Any] = {
        "share_of_cells_go_better": float((d > 0).mean()),
        "largest_gap_between_kickoff_flags": float(v["flag_gap"].max()),
    }
    for y, yl in enumerate(c["yardlines"]):
        sub = d[:, :, y]
        kick_cells = [
            {
                "game_clock": game_clock(secs[i]),
                "margin": c["margins"][j],
                "go_minus_fg": r4(sub[i, j]),
            }
            for i, j in zip(*np.nonzero(sub <= 0), strict=True)
        ]
        out[f"from_{yl}"] = {
            "share_go_better": float((sub > 0).mean()),
            "go_minus_field_goal_min": float(sub.min()),
            "go_minus_field_goal_max": float(sub.max()),
            "go_minus_field_goal_median": float(np.median(sub)),
            "cells_where_field_goal_is_better": kick_cells,
        }
    return out


# --------------------------------------------------------------------------------------
# Sensitivity: re-solve the whole game with one assumption changed
# --------------------------------------------------------------------------------------


def scenario_numbers(sol: mg.Solution, q: dict, base: dict | None) -> dict:
    late, early, tv = late_map(sol, q), early_map(sol, q), try_map(sol, q)
    hl, cmp_ = late_case(sol, q, "headline"), late_case(sol, q, "comparison")
    th = try_headline(sol, q)
    ec = early_case(sol, q)["averaged_over_second_half_kickoff"]
    dl, de = late["go"] - late["field_goal"], early["go"] - early["field_goal"]
    dt = tv["two"] - tv["kick"]
    row = {
        "q1_headline_go_minus_field_goal": hl["go_minus_field_goal"],
        "q1_headline_break_even_p_touchdown": hl["break_even_p_touchdown"],
        "q1_headline_p_touchdown": hl["p_touchdown"],
        "q1_comparison_go_minus_field_goal": cmp_["go_minus_field_goal"],
        "q1_share_of_map_go_better": float((dl > 0).mean()),
        "q2_headline_two_minus_kick": th["two_minus_kick"],
        "q2_headline_break_even_p_two": th["break_even_p_two_point"],
        "q2_p_two_point": sol.model.p_two,
        "q2_share_of_map_two_better": float((dt > 0).mean()),
        "q3_headline_go_minus_field_goal": ec["go_minus_field_goal"],
        "q3_share_of_map_go_better": float((de > 0).mean()),
    }
    if base is not None:
        for name, now, ref in (
            ("q1", dl, base["dl"]),
            ("q2", dt, base["dt"]),
            ("q3", de, base["de"]),
        ):
            row[f"{name}_cells_whose_call_differs_from_base"] = int(((now > 0) != (ref > 0)).sum())
            row[f"{name}_cells"] = int(now.size)
    return {"row": row, "dl": dl, "dt": dt, "de": de}


def sensitivity(q: dict, base: dict) -> tuple[list[dict], dict]:
    mcfg = mdp.load_config()
    models = ComponentModels.load()  # the development bundle the game was solved with
    tables = joblib.load(mg.SOLUTION_DIR / "tables.joblib")["tables"]
    s = q["sensitivity"]
    scenarios: list[tuple[str, dict, float]] = [("base model, re-solved here", {}, 0.0)]
    for shift in s["convert_shifts"]:
        scenarios.append(
            (f"fourth-down conversion probability {shift:+.2f}", {"go_convert_shift": shift}, 0.0)
        )
    for shift in s["two_point_shifts"]:
        scenarios.append((f"two-point success probability {shift:+.2f}", {}, shift))
    rates = mcfg["clock"]["defense_timeout_rate_before_warning"][1:]
    rates += mcfg["clock"]["defense_timeout_rate_after_warning"][1:]
    scenarios.append(
        (
            f"defense calls timeouts at measured rates ({100 * min(rates):.0f}% to "
            f"{100 * max(rates):.0f}%), not optimally",
            {"defense_timeout": s["defense_timeout"]},
            0.0,
        )
    )
    rows, check = [], {}
    for label, kw, two_shift in scenarios:
        model = mg.build_model(mcfg, models, tables, horizon="game", **kw)
        if two_shift:
            model = eq.with_two_point(model, model.p_two + two_shift)
        sol = mg.solve(model)
        res = scenario_numbers(sol, q, base)
        if label.startswith("base model"):
            check = {
                "max_abs_difference_from_saved_solution_on_the_three_maps": float(
                    max(np.abs(res[k] - base[k]).max() for k in ("dl", "dt", "de"))
                )
            }
        rows.append({"scenario": label, **res["row"]})
        print(
            f"  {label[:58]:58s} q1 {100 * res['row']['q1_headline_go_minus_field_goal']:+.2f} "
            f"q2 {100 * res['row']['q2_headline_two_minus_kick']:+.2f} "
            f"q3 {100 * res['row']['q3_headline_go_minus_field_goal']:+.2f}",
            flush=True,
        )
        del sol, res, model
        gc.collect()
    return rows, check


# --------------------------------------------------------------------------------------
# What coaches did (development seasons only)
# --------------------------------------------------------------------------------------


def coach_counts(q: dict) -> dict:
    d = splits.load_dev_plays()
    seasons = sorted(splits.seasons_in(d))
    fg = d.filter((pl.col("down") == 4) & (pl.col("goal_to_go") == 1))
    fg = fg.filter(pl.col("yardline_100") <= 3).filter(
        pl.col("fourth_down_decision").is_in(["go", "field_goal"])
    )

    def tally(df: pl.DataFrame, by: list[str]) -> list[dict]:
        out = (
            df.group_by(by)
            .agg(
                pl.len().alias("n"),
                (pl.col("fourth_down_decision") == "go").mean().alias("share_go"),
            )
            .sort(by)
        )
        return out.to_dicts()

    c = q["goal_line_late"]
    late = fg.filter(
        (pl.col("qtr") == 4)
        & (pl.col("game_seconds_remaining") >= c["seconds"]["first"])
        & (pl.col("game_seconds_remaining") <= c["seconds"]["last"])
        & pl.col("score_differential").is_in([float(x) for x in c["leads"]])
    )
    early = fg.filter(
        pl.col("qtr").is_in([1.0, 2.0]) & (pl.col("score_differential").abs() <= 3)
    ).with_columns(
        pl.when(pl.col("season") <= 2017)
        .then(pl.lit("2010-2017"))
        .otherwise(pl.lit("2018-2023"))
        .alias("era")
    )
    first = q["coach_counts"]["two_point_first_season"]
    tries = d.filter(
        (pl.col("season") >= first)
        & (pl.col("qtr") == 4)
        & ((pl.col("two_point_attempt") == 1) | (pl.col("play_type") == "extra_point"))
        & pl.col("score_differential").is_in([float(x) for x in q["two_point"]["named_margins"]])
    )
    two = (
        tries.group_by("score_differential")
        .agg(pl.len().alias("n"), (pl.col("two_point_attempt") == 1).mean().alias("share_two"))
        .sort("score_differential")
        .to_dicts()
    )
    return {
        "source": f"splits.load_dev_plays(), seasons {seasons[0]} to {seasons[-1]}; no test "
        "or holdout season is read",
        "late_goal_line": {
            "what": "fourth and goal from the 1 to 3, fourth quarter, 2:00 to 12:00 left, "
            "leading by the listed amounts; go or field goal only",
            "by_lead_and_yardline": tally(late, ["score_differential", "yardline_100"]),
            "by_lead": tally(late, ["score_differential"]),
        },
        "early_goal_line": {
            "what": "fourth and goal from the 1 to 3, first half, within 3 points",
            "by_yardline": tally(early, ["yardline_100"]),
            "by_era_and_yardline": tally(early, ["era", "yardline_100"]),
        },
        "tries_fourth_quarter": {
            "what": f"tries in the fourth quarter, seasons {first} to {seasons[-1]}, by the "
            "margin before the try; share that went for two (penalty-replayed tries count "
            "once per snap)",
            "by_margin": two,
        },
    }


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def category_cmap(edges: list[float]) -> tuple[ListedColormap, BoundaryNorm, list[str]]:
    e = list(edges)  # already in win-probability points, like the plotted margins
    bounds = [-1e9, -e[2], -e[1], -e[0], e[0], e[1], e[2], 1e9]
    colors = [ORANGES[2], ORANGES[1], ORANGES[0], GREY, BLUES[0], BLUES[1], BLUES[2]]
    labels = [
        f"by {e[2]:g}+",
        f"by {e[1]:g} to {e[2]:g}",
        f"by {e[0]:g} to {e[1]:g}",
        f"within {e[0]:g}",
        f"by {e[0]:g} to {e[1]:g}",
        f"by {e[1]:g} to {e[2]:g}",
        f"by {e[2]:g}+",
    ]
    return ListedColormap(colors), BoundaryNorm(bounds, len(colors)), labels


def category_legend(fig: plt.Figure, labels: list[str], kick: str, go: str, y: float) -> None:
    cmap_colors = [ORANGES[2], ORANGES[1], ORANGES[0], GREY, BLUES[0], BLUES[1], BLUES[2]]
    names = [
        f"{kick} {lab}" if i < 3 else (f"{go} {lab}" if i > 3 else f"Even ({lab})")
        for i, lab in enumerate(labels)
    ]
    handles = [
        Patch(facecolor=c, edgecolor="none", label=n)
        for c, n in zip(cmap_colors, names, strict=True)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.5, y),
        fontsize=9.5,
        handlelength=1.4,
        columnspacing=1.2,
        labelcolor=style.INK,
    )


def bars(ax: plt.Axes, rows: list[tuple[str, float, str]], xlabel: str, xmax: float = 112) -> None:
    ypos = np.arange(len(rows))[::-1]
    ax.barh(ypos, [100 * v for _, v, _ in rows], color=[c for _, _, c in rows], height=0.62)
    for yv, (_, v, _) in zip(ypos, rows, strict=True):
        ax.text(100 * v + 1.0, yv, f"{100 * v:.1f}%", va="center", fontsize=9.5, color=style.INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for lab, _, _ in rows], fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def sens_bars(ax: plt.Axes, sens: list[dict], key: str, go: str, kick: str) -> None:
    labels = []
    for r in sens:
        s = r["scenario"]
        if s.startswith("base"):
            labels.append("Base model")
        elif s.startswith("fourth-down"):
            labels.append(f"Conversion chance {s.split()[-1]}")
        elif s.startswith("two-point"):
            labels.append(f"Two-point success {s.split()[-1]}")
        else:
            labels.append("Defense uses timeouts\nas real teams do")
    vals = [100 * r[key] for r in sens]
    ypos = np.arange(len(vals))[::-1]
    colors = [BLUES[1] if v > 0 else ORANGES[1] for v in vals]
    ax.barh(ypos, vals, color=colors, height=0.6)
    ax.axvline(0, color=style.INK, linewidth=1)
    spanv = max(max(abs(v) for v in vals), 0.5)
    for yv, v in zip(ypos, vals, strict=True):
        ax.text(
            v + (0.04 * spanv if v >= 0 else -0.04 * spanv),
            yv,
            f"{v:+.1f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9.5,
            color=style.INK,
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    lo, hi = min(min(vals), 0), max(max(vals), 0)
    ax.set_xlim(lo - 0.35 * spanv, hi + 0.35 * spanv)
    ax.set_xlabel(f"{go} minus {kick} (win-probability points)")
    ax.grid(axis="y", visible=False)


def figure_goal_line(late: dict, early: dict, cases: dict, sens: list[dict], q: dict) -> None:
    style.apply()
    cmap, norm, labels = category_cmap(q["margin_edges"])
    c, ce = q["goal_line_late"], q["goal_line_early"]
    secs = np.asarray(span(c["seconds"]))
    d = 100 * (late["go"] - late["field_goal"])
    fig = plt.figure(figsize=(19, 13.5))
    outer = fig.add_gridspec(
        2, 2, width_ratios=[2.0, 0.85], height_ratios=[1.55, 1.0], wspace=0.32, hspace=0.32
    )
    grid = outer[0, 0].subgridspec(len(c["leads"]), len(c["yardlines"]), hspace=0.22, wspace=0.06)
    step = c["seconds"]["step"]
    for j, lead in enumerate(c["leads"]):
        for y, yl in enumerate(c["yardlines"]):
            ax = fig.add_subplot(grid[j, y])
            z = d[:, j, :, y].T  # [opp timeouts, time]
            ax.imshow(
                z[:, ::-1],
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
                extent=(-(secs[-1] + step / 2), -(secs[0] - step / 2), -0.5, 3.5),
                interpolation="nearest",
            )
            ax.grid(False)
            ticks = [660, 480, 300, 120]
            ax.set_xticks([-t for t in ticks])
            ax.set_xticklabels([clock(t) for t in ticks] if j == len(c["leads"]) - 1 else [])
            ax.set_yticks([0, 1, 2, 3])
            if y == 0:
                ax.set_ylabel(f"Up {lead}\nopp. timeouts", color=style.INK)
            else:
                ax.set_yticklabels([])
            if j == 0:
                ax.set_title(f"Fourth and goal from the {yl}", fontsize=11.5, pad=6)
            if j == len(c["leads"]) - 1:
                ax.set_xlabel("Time left in the fourth quarter")
            h = c["headline"]
            for key, mk in (("headline", "o"), ("comparison", "s")):
                hh = c[key]
                if hh["lead"] == lead and hh["yardline"] == yl:
                    ax.scatter(
                        [-hh["seconds"]],
                        [hh["opponent_timeouts"]],
                        s=60,
                        marker=mk,
                        facecolor="none",
                        edgecolor=style.INK,
                        linewidths=1.6,
                        zorder=5,
                    )
            del h
    fig.text(
        0.125,
        0.965,
        "Fourth and goal: kick the field goal or go for the touchdown?",
        fontsize=17,
        fontweight="bold",
        color=style.INK,
    )
    fig.text(
        0.125,
        0.947,
        "Exact solution of the game model, identical teams. Blue: going wins more often. "
        "Orange: the field goal does. Circle: up 7 from the 1 (Daboll's call, at 8:00). "
        "Square: up 14 from the 1 (Bowles's, at 8:00).",
        fontsize=10.5,
        color=style.INK_2,
    )
    fig.text(
        0.125,
        0.928,
        "Top: leading late (timeouts only count inside 5:00). "
        "Bottom: the first half of a close game.",
        fontsize=10.5,
        color=style.INK_2,
    )

    # Early game: margin by time, one panel per yard line.
    secs_e = np.asarray(span(ce["seconds"]))
    de = 100 * (early["go"] - early["field_goal"])
    eg = outer[1, 0].subgridspec(1, len(ce["yardlines"]), wspace=0.06)
    st = ce["seconds"]["step"]
    for y, yl in enumerate(ce["yardlines"]):
        ax = fig.add_subplot(eg[0, y])
        z = de[:, :, y].T  # [margin, time]
        ax.imshow(
            z[:, ::-1],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=(
                -(secs_e[-1] + st / 2),
                -(secs_e[0] - st / 2),
                ce["margins"][0] - 0.5,
                ce["margins"][-1] + 0.5,
            ),
            interpolation="nearest",
        )
        ax.grid(False)
        ticks = [3300, 2700, 2100]
        ax.set_xticks([-t for t in ticks])
        ax.set_xticklabels([game_clock(t) for t in ticks], fontsize=9)
        ax.set_yticks(ce["margins"])
        if y == 0:
            ax.set_ylabel("Score margin of the team at the goal line", color=style.INK)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"From the {yl}, first half", fontsize=11.5, pad=6)
        ax.set_xlabel("Game clock")
        h = ce["headline"]
        if h["yardline"] == yl:
            ax.scatter(
                [-h["seconds"]],
                [h["margin"]],
                s=60,
                marker="D",
                facecolor="none",
                edgecolor=style.INK,
                linewidths=1.6,
                zorder=5,
            )

    # Right column: the two headline cases and the sensitivity rows.
    right = outer[:, 1].subgridspec(3, 1, hspace=0.62, height_ratios=[1, 1, 1.15])
    for k, (key, title) in enumerate(
        (
            ("q1", "Up 7, fourth and goal at the 1, 8:00 left"),
            ("q3", "Tied, fourth and goal at the 1, Q1 10:00"),
        )
    ):
        ax = fig.add_subplot(right[k])
        cs = cases[key]
        rows = [
            ("Field goal", cs["field_goal"], ORANGES[1]),
            (
                f"  good ({100 * cs['p_field_goal_good']:.1f}%)",
                cs["win_probability_if_field_goal_good"],
                ORANGES[0],
            ),
            ("Go for it", cs["go"], BLUES[1]),
            (
                f"  touchdown ({100 * cs['p_touchdown']:.0f}%)",
                cs["win_probability_if_touchdown"],
                BLUES[0],
            ),
            (
                f"  stopped ({100 * (1 - cs['p_keep_or_score']):.0f}%)",
                cs["win_probability_if_stopped"],
                BLUES[0],
            ),
        ]
        bars(ax, rows, "Win probability of the team at the goal line (%)")
        style.title(
            ax,
            title,
            f"Go minus field goal {100 * cs['go_minus_field_goal']:+.1f} points; "
            f"break-even touchdown chance {100 * cs['break_even_p_touchdown']:.0f}%",
        )
    ax = fig.add_subplot(right[2])
    labels_s = []
    vals = []
    for r in sens:
        vals.append(
            (100 * r["q1_headline_go_minus_field_goal"], 100 * r["q3_headline_go_minus_field_goal"])
        )
        labels_s.append(r["scenario"])
    ypos = np.arange(len(sens))[::-1]
    hgt = 0.36
    for off, idx in ((hgt / 2, 0), (-hgt / 2, 1)):
        v = [x[idx] for x in vals]
        ax.barh(
            ypos + off,
            v,
            height=hgt,
            color=[BLUES[1] if x > 0 else ORANGES[1] for x in v],
            edgecolor=style.SURFACE,
            hatch=None if idx == 0 else "///",
        )
        for yv, x in zip(ypos + off, v, strict=True):
            ax.text(
                x + (0.12 if x >= 0 else -0.12),
                yv,
                f"{x:+.1f}",
                va="center",
                ha="left" if x >= 0 else "right",
                fontsize=8.5,
                color=style.INK,
            )
    ax.axvline(0, color=style.INK, linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([short_label(s) for s in labels_s], fontsize=9)
    allv = [x for p in vals for x in p]
    ax.set_xlim(min(min(allv), 0) - 1.2, max(allv) + 1.6)
    ax.set_xlabel("Go minus field goal (win-probability points)")
    ax.grid(axis="y", visible=False)
    style.title(
        ax,
        "How the two headline answers move",
        "Re-solved each time. Solid: up 7 at the 1, 8:00. Hatched: tied, Q1",
    )
    category_legend(fig, labels, "Field goal", "Go", 0.035)
    style.save(fig, FIG_GOAL)


def short_label(s: str) -> str:
    if s.startswith("base"):
        return "Base model"
    if s.startswith("fourth-down"):
        return f"Conversion chance {s.split()[-1]}"
    if s.startswith("two-point"):
        return f"Two-point success {s.split()[-1]}"
    return "Defense uses timeouts\nas real teams do"


def figure_two_point(tv: dict, head: dict, paths: dict, sens: list[dict], q: dict) -> None:
    style.apply()
    cmap, norm, labels = category_cmap(q["margin_edges"])
    c = q["two_point"]
    secs, margins = np.asarray(span(c["seconds"])), span(c["margins"])
    d = 100 * (tv["two"] - tv["kick"])
    fig = plt.figure(figsize=(18, 11))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.3)
    ax = fig.add_subplot(outer[0])
    step = c["seconds"]["step"]
    ax.imshow(
        d.T[:, ::-1],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=(-(secs[-1] + step / 2), -(secs[0] - step / 2), margins[0] - 0.5, margins[-1] + 0.5),
        interpolation="nearest",
    )
    ax.grid(False)
    ticks = [900, 720, 540, 360, 180, 30]
    ax.set_xticks([-t for t in ticks])
    ax.set_xticklabels([clock(t) for t in ticks])
    ax.set_yticks(margins)
    ax.set_yticklabels([f"{m:+d}" if m else "0" for m in margins], fontsize=8.5)
    ax.set_xlabel("Time left in the fourth quarter when the touchdown is scored")
    ax.set_ylabel("Scoring team's margin after the touchdown, before the try", color=style.INK)
    for m in c["named_margins"]:
        ax.annotate(
            f"{d[list(secs).index(c['headline']['seconds']), margins.index(m)]:+.1f}",
            xy=(-c["headline"]["seconds"], m),
            ha="center",
            va="center",
            fontsize=8,
            color=style.INK,
        )
    ax.axvline(-c["headline"]["seconds"], color=style.INK_2, linewidth=0.8, linestyle=":")
    style.title(
        ax,
        "After a touchdown: kick the extra point or go for two?",
        f"Exact game model, both teams with {c['timeouts']['scorer']} timeouts. Numbers: "
        f"two minus kick at {clock(c['headline']['seconds'])} left, in points",
    )

    right = outer[1].subgridspec(3, 1, hspace=0.75, height_ratios=[1.0, 0.75, 1.05])
    ax = fig.add_subplot(right[0])
    w = head["win_probability_kicking_off_at_margin"]
    mm = head["margin_before_try"]
    rows = [
        ("Kick the extra point", head["kick"], ORANGES[1]),
        (
            f"  good ({100 * head['p_extra_point']:.0f}%): down {-(mm + 1)}",
            w[str(mm + 1)],
            ORANGES[0],
        ),
        ("Go for two", head["two"], BLUES[1]),
        (f"  good ({100 * head['p_two_point']:.0f}%): down {-(mm + 2)}", w[str(mm + 2)], BLUES[0]),
        (f"  either try fails: down {-mm}", w[str(mm)], GREY),
    ]
    top = 100 * max(v for _, v, _ in rows)
    bars(ax, rows, "Win probability of the trailing team (%)", xmax=1.3 * top)
    ax.set_xticks(np.arange(0, top + 1, 5 if top < 40 else 10))
    ax.set_xlim(0, 1.3 * top)  # set after the ticks, which would widen the view
    style.title(
        ax,
        f"Down {-mm} after a touchdown, {clock(head['situation']['seconds'])} left",
        f"Two minus kick {100 * head['two_minus_kick']:+.1f} points; break-even two-point "
        f"rate {100 * head['break_even_p_two_point']:.0f}%",
    )

    ax = fig.add_subplot(right[1])
    ar = head["score_arithmetic_on_its_own_path"]
    share = paths[f"trailing_by_{-(mm + 1)}"]["share_of_games_on_the_arithmetic_path"]
    vals = [100 * ar["difference"], 100 * head["two_minus_kick"]]
    names = [
        "Score arithmetic:\none more TD for you,\nnone for them",
        "Whole game\n(exact solution)",
    ]
    ypos = np.array([1, 0])
    ax.barh(ypos, vals, color=[BLUES[0], BLUES[1]], height=0.55)
    for yv, v in zip(ypos, vals, strict=True):
        ax.text(v + 0.3, yv, f"{v:+.1f}", va="center", fontsize=9.5, color=style.INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("Two minus kick (win-probability points)")
    ax.grid(axis="y", visible=False)
    style.title(
        ax,
        "Why the edge is small",
        f"That path happens in {100 * share:.0f}% of simulated games from down {-(mm + 1)}",
    )

    ax = fig.add_subplot(right[2])
    sens_bars(ax, sens, "q2_headline_two_minus_kick", "Two", "kick")
    style.title(ax, f"Down {-mm}: how the answer moves", "Each row re-solves the whole game")
    category_legend(fig, labels, "Kick", "Two", 0.0)
    style.save(fig, FIG_TWO)


# --------------------------------------------------------------------------------------


def main() -> None:
    q = load_config()
    sol = mg.load_solution()
    m = sol.model
    print("maps from the saved solution", flush=True)
    late, early = late_map(sol, q), early_map(sol, q)
    tv, tv_all = try_map(sol, q), try_map(sol, q, all_timeouts=True)
    cases = {
        "q1": late_case(sol, q, "headline"),
        "q1_comparison": late_case(sol, q, "comparison"),
        "q2": try_headline(sol, q),
        "q3": early_case(sol, q),
    }
    h2 = q["two_point"]["headline"]
    print("simulating the score-arithmetic path", flush=True)
    paths = path_shares(sol, q, h2["seconds"], h2["margin"])
    base = {
        "dl": late["go"] - late["field_goal"],
        "dt": tv["two"] - tv["kick"],
        "de": early["go"] - early["field_goal"],
    }
    summaries = {
        "q1": late_summary(late, q),
        "q2": try_summary(tv, tv_all, q),
        "q3": early_summary(early, q),
    }
    del sol
    gc.collect()
    print("sensitivity re-solves", flush=True)
    sens, check = sensitivity(q, base)
    coaches = coach_counts(q)
    cl, ce, c2 = q["goal_line_late"], q["goal_line_early"], q["two_point"]
    out = {
        "question": "Goal-line field goal or go (late with a lead, early in a close game), and "
        "the extra point or two after a touchdown, from the exact game solution",
        "method": "exact backward induction on the M4 game model; see docs/notes/m4_mdp.md and "
        "docs/notes/m13_goal_line_and_two_point.md",
        "seed": q["seed"],
        "assumptions": {
            "teams": "identical league-average teams; both play the model's optimal policy",
            "p_extra_point": m.p_xp,
            "p_two_point": m.p_two,
            "two_point_rate_source": "share of two-point attempts that succeeded in the "
            "development seasons (components bundle); one rate for every team and situation",
            "touchdown_chance_source": "the fourth-down play-outcome model (M1) at that yard "
            "line; one rate for every team and situation",
            "no_overtime": "a tie after regulation is worth 0.5",
            "hypothetical": "the 2026 cases quoted on X are in the sealed holdout season and "
            "are rebuilt here as hypothetical states; no 2026 data is read",
        },
        "q1_goal_line_late": {
            "headline": cases["q1"],
            "comparison_case": cases["q1_comparison"],
            "map_summary": summaries["q1"],
            "grid": {
                "seconds": span(cl["seconds"]),
                "leads": cl["leads"],
                "opponent_timeouts": cl["opponent_timeouts"],
                "yardlines": cl["yardlines"],
                "offense_timeouts": cl["offense_timeouts"],
                "axes": "[seconds][lead][opponent timeouts][yardline]",
                "go_wp": r4(late["go"]),
                "field_goal_wp": r4(late["field_goal"]),
            },
        },
        "q2_two_point": {
            "headline": cases["q2"],
            "arithmetic_path_simulated": paths,
            "map_summary": summaries["q2"],
            "grid": {
                "seconds": span(c2["seconds"]),
                "margins_before_try": span(c2["margins"]),
                "timeouts": c2["timeouts"],
                "axes": "[seconds][margin before the try]",
                "kick_wp": r4(tv["kick"]),
                "two_wp": r4(tv["two"]),
            },
        },
        "q3_goal_line_early": {
            "headline": cases["q3"],
            "map_summary": summaries["q3"],
            "grid": {
                "seconds": span(ce["seconds"]),
                "margins": ce["margins"],
                "yardlines": ce["yardlines"],
                "axes": "[seconds][margin][yardline], averaged over who receives the "
                "second-half kickoff",
                "go_wp": r4(early["go"]),
                "field_goal_wp": r4(early["field_goal"]),
            },
        },
        "sensitivity": sens,
        "sensitivity_check": check,
        "sensitivity_note": "each row re-solves the whole game in memory with one assumption "
        "changed (data/mdp is not written); values are exact for that changed model",
        "coach_counts_development_seasons": coaches,
        "public_numbers_quoted_not_ours": {
            "source": "research-kit/x/2_twitter_findings.md, decision 6 (posts of 2026-10-09)",
            "ryan_paganetti_bowles_case": "conversion at the 1 about 57.6% (200 of 347), about "
            "54-55% without a sneak; a touchdown to go up 21 wins over 99%; a failure, up 14, "
            "about 96%; a made field goal (over 99% likely) about 98.5%; he would have kicked",
            "paganetti_go_for_two_down_8": "used 53 times by 23 franchises since 2018; in 2025 "
            "teams went for two in this situation in the fourth quarter roughly half the time",
        },
        "peak_memory_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }
    write_json(out, OUT)
    figure_goal_line(
        late,
        early,
        {"q1": cases["q1"], "q3": cases["q3"]["averaged_over_second_half_kickoff"]},
        sens,
        q,
    )
    figure_two_point(tv, cases["q2"], paths, sens, q)
    print("wrote", OUT, FIG_GOAL, FIG_TWO)
    for k in ("q1", "q1_comparison"):
        c = cases[k]
        print(
            f"{k}: go {c['go']:.4f} fg {c['field_goal']:.4f} "
            f"diff {100 * c['go_minus_field_goal']:+.2f}"
        )
    c = cases["q2"]
    print(f"q2: two {c['two']:.4f} kick {c['kick']:.4f} diff {100 * c['two_minus_kick']:+.2f}")
    c = cases["q3"]["averaged_over_second_half_kickoff"]
    print(
        f"q3: go {c['go']:.4f} fg {c['field_goal']:.4f} diff {100 * c['go_minus_field_goal']:+.2f}"
    )


if __name__ == "__main__":
    main()
