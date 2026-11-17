"""M4: solve the single-drive problem and the whole-game problem, check them, save them.

Run:  .venv/bin/python scripts/solve_mdp.py            (everything, about 3 minutes)
      .venv/bin/python scripts/solve_mdp.py --reuse    (keep the saved game, redo the rest)

Writes
  data/mdp/                         value and policy arrays, model, count tables (gitignored)
  results/m4_mdp_summary.json       sizes, timings, stage 1 numbers, checks
  assets/m4_expected_points.png     stage 1: expected points and the fourth-down chart
  assets/m4_value_function.png      stage 2: win probability across the field as time runs out
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from fourth_and_rl import data, mdp, style  # noqa: E402
from fourth_and_rl import mdp_game as mg  # noqa: E402
from fourth_and_rl.components import ComponentModels  # noqa: E402
from fourth_and_rl.io import write_json  # noqa: E402
from fourth_and_rl.mdp_drive import (  # noqa: E402
    DRIVE_ACTIONS,
    first_and_ten_curve,
    fourth_down_chart,
    solve_drive,
)
from fourth_and_rl.paths import ASSETS, RESULTS, ROOT  # noqa: E402

TABLES_PATH = mg.SOLUTION_DIR / "tables.joblib"


def load_tables(
    cfg: dict, plays: pl.DataFrame | None = None, models: ComponentModels | None = None
) -> dict[str, Any]:
    """Count tables and the coach table, fitted once and cached next to the solution."""
    if plays is None and TABLES_PATH.exists():
        return joblib.load(TABLES_PATH)
    plays = data.load_plays() if plays is None else plays
    models = ComponentModels.load() if models is None else models
    out = {
        "tables": mdp.fit_tables(plays, cfg),
        "coach": mdp.fit_coach_table(plays, mdp.grid_from_config(cfg), cfg),
    }
    out["tables"]["adjust"] = mdp.fit_mode_adjustments(plays, models, cfg)
    out["coach"]["end_fg_max_yardline"] = cfg["coach"]["end_fg_max_yardline"]
    out["coach"]["end_fg_ticks"] = cfg["coach"]["end_fg_ticks"]
    TABLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, TABLES_PATH)
    return out


# --------------------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------------------


def nflfastr_ep_curve(plays: pl.DataFrame, first_season: int) -> np.ndarray:
    """Mean of nflfastR's `ep` on first and 10 (or goal) by yard line, away from the end
    of a half. Index 0 unused. This is nflfastR's number, quoted for comparison only."""
    d = plays.filter(
        (pl.col("season") >= first_season)
        & (pl.col("down") == 1)
        & ((pl.col("ydstogo") == 10) | (pl.col("goal_to_go") == 1))
        & (pl.col("half_seconds_remaining") > 300)
        & pl.col("ep").is_not_null()
        & pl.col("yardline_100").is_between(1, 99)
    )
    g = d.group_by("yardline_100").agg(pl.col("ep").mean(), pl.len().alias("n"))
    out = np.full(100, np.nan)
    out[g["yardline_100"].to_numpy().astype(int)] = g["ep"].to_numpy()
    return out


def stage_one(cfg: dict, models: ComponentModels, bundle: dict, plays: pl.DataFrame) -> dict:
    model = mg.build_model(cfg, models, bundle["tables"], horizon="endgame")
    kern, grid = model.kern, model.kern.grid
    dcfg = cfg["drive"]
    td_value = cfg["scoring"]["touchdown"] + models.p_extra_point()
    common = dict(
        td_value=td_value,
        fg_points=cfg["scoring"]["field_goal"],
        safety_points=cfg["scoring"]["safety"],
        tolerance=dcfg["tolerance"],
        max_iterations=dcfg["max_iterations"],
    )
    coach_pol = np.zeros((grid.nf, 3))
    downs, dbins, yls = grid.unpack(np.arange(grid.nf))
    coach_pol[:] = bundle["coach"]["table"][0, dbins, yls - 1]

    t0 = time.time()
    opt = solve_drive(kern, **common)
    next_score = solve_drive(kern, after_score="absorb", **common)
    coach = solve_drive(kern, fourth_policy=coach_pol, **common)
    seconds = time.time() - t0
    # One-step choice when every LATER fourth down follows the coaches: compare the three
    # actions using the coach-policy values as the continuation. `coach["q"]` is exactly that.
    chart_opt = fourth_down_chart(kern, opt["q"], opt["allowed"])
    chart_coach = fourth_down_chart(kern, coach["q"], coach["allowed"])
    differ = chart_opt["best"] != chart_coach["best"]
    ep_ref = nflfastr_ep_curve(plays, cfg["tables"]["first_season"])
    ours_next = first_and_ten_curve(kern, next_score["v"])
    ours_inf = first_and_ten_curve(kern, opt["v"])
    ours_coach = first_and_ten_curve(kern, coach["v"])
    ok = ~np.isnan(ep_ref[1:])
    marks = [10, 25, 50, 75, 90]
    go_cells = int((chart_opt["best"] == 0).sum())
    summary = {
        "what": "expected net points over endless possessions; fourth-down actions go, field "
        "goal, punt; downs 1 to 3 a generic play",
        "n_states": int(grid.nf),
        "touchdown_value": td_value,
        "solve_seconds_all_three": seconds,
        "value_iteration": {
            name: {
                "iterations": r["iterations"],
                "residual": r["residual"],
                "converged": r["converged"],
            }
            for name, r in (("optimal", opt), ("next_score", next_score), ("coach_future", coach))
        },
        "kickoff_value_to_receiver": {
            "optimal": opt["kickoff_value"],
            "coach_future": coach["kickoff_value"],
        },
        "first_and_10_value_by_yardline_100": {
            str(y): {
                "optimal_endless": float(ours_inf[y - 1]),
                "coach_future_endless": float(ours_coach[y - 1]),
                "optimal_next_score": float(ours_next[y - 1]),
                "nflfastr_ep_mean": float(ep_ref[y]),
            }
            for y in marks
        },
        "next_score_vs_nflfastr_ep": {
            "correlation": float(np.corrcoef(ours_next[ok], ep_ref[1:][ok])[0, 1]),
            "mean_difference_ours_minus_nflfastr": float((ours_next[ok] - ep_ref[1:][ok]).mean()),
            "rmse": float(np.sqrt(((ours_next[ok] - ep_ref[1:][ok]) ** 2).mean())),
            "note": "nflfastR ep is a fitted next-score model on real play with real coaching; "
            "ours is the optimal-policy next-score value of our transition model",
        },
        "fourth_down_chart": {
            "cells": int(differ.size),
            "optimal_go_cells": go_cells,
            "coach_future_go_cells": int((chart_coach["best"] == 0).sum()),
            "cells_where_the_two_charts_differ": int(differ.sum()),
            "largest_margin_points_in_differing_cells": float(
                np.maximum(chart_opt["margin"], chart_coach["margin"])[differ].max()
                if differ.any()
                else 0.0
            ),
            "mean_gain_from_optimal_future_first_and_10_points": float(
                (ours_inf - ours_coach).mean()
            ),
            "note": "coach_future: every later fourth down is played with the observed "
            "neutral-situation coach frequencies; only the current one is optimized",
        },
    }
    figure_expected_points(grid, ours_inf, ours_coach, ours_next, ep_ref, chart_opt, chart_coach)
    return summary


def _chart_axes(ax: plt.Axes, grid: mdp.Grid, best: np.ndarray, differ: np.ndarray | None) -> None:
    colors = [style.ACTION_COLORS[a] for a in DRIVE_ACTIONS]
    ax.imshow(
        best,
        origin="lower",
        aspect="auto",
        cmap=ListedColormap(colors),
        vmin=0,
        vmax=2,
        extent=(0.5, grid.n_yl + 0.5, 0.5, grid.n_dist + 0.5),
        interpolation="nearest",
    )
    if differ is not None and differ.any():
        d, y = np.nonzero(differ)
        ax.scatter(y + 1, d + 1, s=5, color=style.INK, marker="s", linewidths=0)
    ax.set_yticks(range(1, grid.n_dist + 1))
    ax.set_yticklabels([str(i) for i in range(1, 11)] + ["11-15", "16+"])
    ax.set_xticks([1, 20, 40, 60, 80, 99])
    ax.set_xticklabels(["goal", "opp 20", "opp 40", "own 40", "own 20", "own 1"])
    ax.set_xlabel("Field position")
    ax.set_ylabel("Yards to go")
    ax.grid(False)


def figure_expected_points(
    grid, ours_inf, ours_coach, ours_next, ep_ref, chart_opt, chart_coach
) -> None:
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    y = np.arange(1, grid.n_yl + 1)
    ax.plot(y, ours_inf, color=style.BLUE, label="Ours: endless game, optimal fourth downs")
    ax.plot(y, ours_coach, color=style.AQUA, label="Ours: endless game, coaches' fourth downs")
    ax.plot(y, ours_next, color=style.VIOLET, label="Ours: next score only")
    ax.plot(y, ep_ref[1:], color=style.ORANGE, linestyle="--", label="nflfastR ep (mean)")
    ax.axhline(0, color=style.AXIS, linewidth=1)
    ax.set_xlim(99, 1)
    ax.set_xticks([99, 80, 60, 40, 20, 1])
    ax.set_xticklabels(["own 1", "own 20", "own 40", "opp 40", "opp 20", "goal"])
    ax.set_xlabel("Field position, first and 10")
    ax.set_ylabel("Expected points")
    ax.legend(loc="upper left", fontsize=8.5)
    style.title(ax, "Value of a first down", "Solved from the play models alone")
    differ = chart_opt["best"] != chart_coach["best"]
    _chart_axes(axes[1], grid, chart_opt["best"], None)
    style.title(
        axes[1],
        "Fourth-down chart, expected points",
        "Every future fourth down also played optimally",
    )
    _chart_axes(axes[2], grid, chart_coach["best"], differ)
    style.title(
        axes[2],
        "Same, future fourth downs as coaches play",
        "Dots: cells where the two charts disagree",
    )
    handles = [
        Patch(color=style.ACTION_COLORS[a], label=style.ACTION_LABELS[a]) for a in DRIVE_ACTIONS
    ]
    axes[1].legend(
        handles=handles,
        loc="upper left",
        fontsize=8.5,
        frameon=True,
        facecolor=style.SURFACE,
        edgecolor=style.GRID,
    )
    style.save(fig, ASSETS / "m4_expected_points.png")


# --------------------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------------------


def monte_carlo_check(sol: mg.Solution, cfg: dict) -> dict:
    """Random states; simulated win frequency under the solved policy against the value."""
    mc = cfg["monte_carlo"]
    rng = np.random.default_rng(mc["seed"])
    g, c = sol.grid, sol.model.clock
    k, n = mc["n_states"], mc["games_per_state"]
    t = np.where(
        rng.random(k) < 0.5,
        rng.integers(1, c.endgame_ticks + 1, k),
        rng.integers(c.endgame_ticks + 1, c.n_ticks + 1, k),
    )
    score = rng.integers(-10, 11, k)
    yl = rng.integers(1, 100, k)
    down = rng.integers(1, 5, k)
    dist = np.minimum(rng.integers(1, 16, k), yl)
    f = g.fidx(down, g.dist_bin(dist), yl)
    a, b, r = rng.integers(0, 4, k), rng.integers(0, 4, k), rng.integers(0, 2, k)
    sim = mg.Simulator(sol)
    rep = lambda x: np.repeat(x, n)  # noqa: E731
    st = sim.start(k * n, t=rep(t), s=rep(score), f=rep(f), a=rep(a), b=rep(b), r=rep(r))
    expect = sim.values(st)[::n].astype(float)
    t0 = time.time()
    res = mg.simulate(sim, st, rng)["result"].reshape(k, n)
    mean = res.mean(axis=1)
    se = res.std(axis=1) / np.sqrt(n)
    # z scores only where the outcome is really uncertain; near 0 or 1 the sample standard
    # error is itself unreliable, and those states are judged by the raw difference.
    decided = (expect > 0.02) & (expect < 0.98) & (se > 0)
    z = (mean - expect) / np.where(se > 0, se, 1.0)
    rows = [
        {
            "seconds": int(t[i] * c.tick_seconds),
            "score": int(score[i]),
            "down": int(down[i]),
            "ydstogo": int(dist[i]),
            "yardline_100": int(yl[i]),
            "off_timeouts": int(a[i]),
            "def_timeouts": int(b[i]),
            "dp_value": float(expect[i]),
            "simulated": float(mean[i]),
            "standard_error": float(se[i]),
            "z": float(z[i]) if decided[i] else None,
        }
        for i in range(k)
    ]
    zz = z[decided]
    return {
        "n_states": k,
        "games_per_state": n,
        "seed": mc["seed"],
        "simulate_seconds": time.time() - t0,
        "states_with_value_between_0.02_and_0.98": int(decided.sum()),
        "max_abs_z": float(np.abs(zz).max()),
        "share_abs_z_below_2": float((np.abs(zz) < 2).mean()),
        "mean_z": float(zz.mean()),
        "rms_z": float(np.sqrt((zz**2).mean())),
        "max_abs_difference": float(np.abs(mean - expect).max()),
        "max_abs_difference_in_near_certain_states": float(
            np.abs(mean - expect)[~decided].max() if (~decided).any() else 0.0
        ),
        "expected_if_correct": "z scores behave like standard normal draws: rms near 1, about "
        "95% below 2 in absolute value",
        "states": rows,
    }


def describe_solution(sol: mg.Solution) -> dict:
    m, g = sol.model, sol.grid
    z = m.score_max
    top = sol.post(m.clock.n_ticks, "first").ko  # receiving the opening kickoff, tied
    opening = float(top[z, 0, 0])
    share = {}
    for name, arr in sol.pol.items():
        pol = np.asarray(arr[1:] if name == "end" else arr)
        downs = g.unpack(np.arange(g.nf))[0]
        four = pol[:, downs == 4]
        share[name] = {
            "fourth_down_go": float(np.isin(four, (mg.A_NEUTRAL, mg.A_PROTECT, mg.A_CHASE)).mean()),
            "fourth_down_field_goal": float((four == mg.A_FG).mean()),
            "fourth_down_punt": float((four == mg.A_PUNT).mean()),
            "note": "shares of ALL grid states, most of which never occur in a game",
        }
    return {
        "value_of_receiving_the_opening_kickoff": opening,
        "examples": {
            "up_4_first_and_10_own_25_4_00_left_def_3_timeouts": sol.value(
                mg.GameState(240, 4, 1, 10, 75, 3, 3)
            ),
            "down_4_first_and_10_own_25_4_00_left_3_timeouts_each": sol.value(
                mg.GameState(240, -4, 1, 10, 75, 3, 3)
            ),
            "tied_first_and_10_own_25_start_of_fourth_quarter": sol.value(
                mg.GameState(900, 0, 1, 10, 75)
            ),
            "up_7_first_and_10_own_25_start_of_second_half": sol.value(
                mg.GameState(1800, 7, 1, 10, 75)
            ),
        },
        "policy_shares": share,
    }


def figure_value_function(sol: mg.Solution) -> None:
    style.apply()
    m, g, c = sol.model, sol.grid, sol.model.clock
    f10 = g.first10()[1:]
    times = [
        (1800, "30:00"),
        (900, "15:00"),
        (300, "5:00"),
        (180, "3:00"),
        (120, "2:00"),
        (60, "1:00"),
        (30, "0:30"),
    ]
    scores = [(-7, "Trailing by 7"), (-3, "Trailing by 3"), (0, "Tied"), (3, "Leading by 3")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(1, g.n_yl + 1)
    for ax, (score, label) in zip(axes, scores, strict=True):
        for (sec, name), color in zip(times, style.SEQUENTIAL, strict=True):
            t = sec // c.tick_seconds
            ph = m.phase(t)
            v = sol.v[ph][t - m.phase_range(ph)[0]][f10, score + m.score_max]
            v = v[:, 3, 3] if ph == "end" else v[:, 0, 0]
            ax.plot(y, v, color=color, label=f"{name} left", linewidth=1.8)
        ax.set_xlim(99, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([80, 50, 20])
        ax.set_xticklabels(["own 20", "50", "opp 20"])
        ax.set_xlabel("Field position, first and 10")
        ax.text(
            0.5,
            1.02,
            label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color=style.INK,
            va="bottom",
            ha="center",
        )
    axes[0].set_ylabel("Win probability of the team with the ball")
    axes[-1].legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, title="Clock", title_fontsize=9
    )
    fig.suptitle(
        "Field position matters more as the clock runs down",
        x=0.125,
        y=1.06,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=style.INK,
    )
    fig.text(
        0.125,
        0.975,
        "Exact solution of the game model; win probability of the team with "
        "the ball; all timeouts in hand inside 5:00",
        color=style.INK_2,
        fontsize=10,
    )
    style.save(fig, ASSETS / "m4_value_function.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="load the saved game, do not solve")
    args = parser.parse_args()
    cfg = mdp.load_config()
    plays = data.load_plays()
    models = ComponentModels.load()
    bundle = load_tables(cfg, plays, models)
    tables = bundle["tables"]

    print("stage 1: single drive")
    drive = stage_one(cfg, models, bundle, plays)

    print("stage 2: whole game")
    if args.reuse and mg.solution_exists():
        sol = mg.load_solution()
    else:
        t0 = time.time()
        model = mg.build_model(cfg, models, tables, horizon="game")
        build_seconds = time.time() - t0
        sol = mg.solve(model, progress=True)
        sol.meta["kernel_build_seconds"] = build_seconds
        sol.meta["seed"] = cfg["seed"]
        t0 = time.time()
        mg.save_solution(sol, ROOT / cfg["solve"]["output_dir"])
        sol.meta["save_seconds"] = time.time() - t0
    m, c = sol.model, sol.model.clock
    figure_value_function(sol)
    print("monte carlo check")
    mc = monte_carlo_check(sol, cfg)
    print(f"  max |z| = {mc['max_abs_z']:.2f}, rms z = {mc['rms_z']:.2f}")
    files = sorted((ROOT / cfg["solve"]["output_dir"]).glob("*.npy"))
    summary = {
        "seed": cfg["seed"],
        "stage_1_single_drive": drive,
        "stage_2_game": {
            "state": "ticks left, field state (4 downs x 12 distance bins x 99 yard lines), "
            "score differential clipped to +-21, and per phase: both teams' timeouts "
            "(last 5:00), nothing (rest of second half), second-half kickoff flag "
            "(first half)",
            "tick_seconds": c.tick_seconds,
            "ticks": {"game": c.n_ticks, "end_phase": c.endgame_ticks, "halftime_at": c.th},
            "field_states": int(sol.grid.nf),
            "score_states": m.n_score,
            "n_states": sol.meta["n_states"],
            "n_states_note": "the end phase counts 61 time slices including the terminal one",
            "solve_seconds": sol.meta["solve_seconds"],
            "kernel_build_seconds": sol.meta.get("kernel_build_seconds"),
            "peak_memory_gb_whole_script_at_end_of_solve": sol.meta["peak_memory_gb"],
            "threads": int(os.environ["OMP_NUM_THREADS"]),
            "dtype": m.dtype,
            "convergence": "none needed: one backward pass, each slice computed once from "
            "slices closer to the end of the game",
            "saved_arrays_gb": sum(p.stat().st_size for p in files) / 1024**3,
            "kernel_nonzeros": {
                "play": int(m.kern.k_all["neutral"].nnz),
                "in_bounds_part": int(m.kern.k_in["neutral"].nnz),
                "punt": int(m.kern.k_punt.nnz),
            },
            "kneel_thresholds_seconds_by_defense_timeouts_first_down": (
                c.kneel_ticks[:, 0] * c.tick_seconds
            ).tolist(),
            "solution": describe_solution(sol),
        },
        "count_tables": {k: v for k, v in tables.items() if k != "free_kick"},
        "coach_table": {k: bundle["coach"][k] for k in ("n", "n_by_situation", "first_season")}
        | {"situations": list(mdp.SITUATIONS)},
        "monte_carlo_check": mc,
        "unit_tests": "tests/test_mdp.py: hand-solved game, random walk football (arXiv "
        "2406.16171), an independent recursive solver on a toy game with every "
        "rule, invariants, Monte Carlo on the toy and on the saved solution",
    }
    write_json(summary, RESULTS / "m4_mdp_summary.json")
    print("wrote", RESULTS / "m4_mdp_summary.json")


if __name__ == "__main__":
    main()
