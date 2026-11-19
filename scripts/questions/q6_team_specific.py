"""Question 6: "What should THIS team do against THAT opponent?", solved exactly.

The late-lead question of question 1 (up 4, 4:00 left, fourth and 2 at the own 40,
3 timeouts each) with two different teams instead of two identical league-average ones.
Each team's offense gets its own play kernels against the other team's defense
(`mdp_team.matchup_kernels`): a yards tilt calibrated so expected points per drive from
the own 25 move by what M8's drive model implies, and M8's shift in conversion log-odds.
The game is then solved with two value arrays (`mdp_team.solve_matchup`).

Ratings: end-of-2025-season offense and defense ratings. They exist only as the frozen
output of M8's test stage (`results/m8_team_strength.json`, written by
scripts/m8_team_strength.py, which ran the M8 filter through the 2024 and 2025 test
seasons). They are read from that file, not recomputed; no play-by-play is loaded and the
2026 holdout is never touched. Groups (top-five and bottom-five units of 2025) come from
`team_strength.group_ratings`, the effect models from data/models/team_strength.joblib.

Only the last 5:00 is solved for the matchups (the end phase never reads later times, so
its values equal a whole-game solve's; the script checks this on one matchup). One
whole-game two-team solve runs in a child process to measure time and peak memory.

No bootstrap: these values have no uncertainty interval yet.

Run:  python scripts/questions/q6_team_specific.py          (about 8 minutes, 3 threads)
Writes results/questions/q6_team_specific.json and assets/questions/q6_team_specific.png.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("POLARS_MAX_THREADS", "3")

import argparse  # noqa: E402
import gc  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from fourth_and_rl import mdp, style  # noqa: E402
from fourth_and_rl import mdp_game as mg  # noqa: E402
from fourth_and_rl import mdp_team as mt  # noqa: E402
from fourth_and_rl import team_strength as ts  # noqa: E402
from fourth_and_rl.io import read_json, write_json  # noqa: E402
from fourth_and_rl.paths import ASSETS, DATA, RESULTS  # noqa: E402

OUT = RESULTS / "questions" / "q6_team_specific.json"
FIG = ASSETS / "questions" / "q6_team_specific.png"
M8_RESULT = RESULTS / "m8_team_strength.json"
BUNDLE = DATA / "models" / "team_strength.joblib"
TEAM, SEASON = "DET", 2025
PLAY_CODES = (mg.A_NEUTRAL, mg.A_PROTECT, mg.A_CHASE)
AVERAGE_OFF = {"conv_off": 0.0, "drive_off": 0.0}
AVERAGE_DEF = {"conv_def": 0.0, "drive_def": 0.0}
CHECK_STARTS = (90, 75, 50, 30)  # drive starts where the calibration is reported


def season_end_2025() -> pl.DataFrame:
    """End-of-2025 ratings, frozen output of M8's test stage (see the module docstring)."""
    rows = read_json(M8_RESULT)["test"]["ratings_season_end"][str(SEASON)]
    return pl.DataFrame(rows)


def matchups(season_end: pl.DataFrame) -> list[dict[str, Any]]:
    """A = the team with the ball at the decision (leading by 4); B = the opponent."""
    g = ts.group_ratings(season_end, [SEASON], k=5)
    det = season_end.filter(pl.col("team") == TEAM).to_dicts()[0]
    det_off = {"conv_off": det["conv_off"], "drive_off": det["drive_off"]}
    det_def = {"conv_def": det["conv_def"], "drive_def": det["drive_def"]}
    top5_def, bot5_def, bot5_off = g["best"], g["worst"], g["bottom"]
    return [
        {"key": "identical", "label": "Identical average teams",
         "a_off": AVERAGE_OFF, "a_def": AVERAGE_DEF, "b_off": AVERAGE_OFF, "b_def": AVERAGE_DEF},
        {"key": "det_vs_top5_def", "label": "Detroit vs a top-5 defense",
         "a_off": det_off, "a_def": det_def, "b_off": AVERAGE_OFF, "b_def": top5_def},
        {"key": "det_vs_average", "label": "Detroit vs an average team",
         "a_off": det_off, "a_def": det_def, "b_off": AVERAGE_OFF, "b_def": AVERAGE_DEF},
        {"key": "det_vs_bottom5_def", "label": "Detroit vs a bottom-5 defense",
         "a_off": det_off, "a_def": det_def, "b_off": AVERAGE_OFF, "b_def": bot5_def},
        {"key": "bottom5_off_vs_top5_def", "label": "Bottom-5 offense vs a top-5 defense",
         "a_off": bot5_off, "a_def": AVERAGE_DEF, "b_off": AVERAGE_OFF, "b_def": top5_def},
    ]  # fmt: skip


def calibration_record(base: mg.GameModel, off: dict, deff: dict, eff: mt.RatingEffects
                       ) -> tuple[mg.GameModel, dict[str, Any]]:  # fmt: skip
    model, c = mt.matchup_model(base, off, deff, eff)
    g = base.kern.grid
    f10 = g.first10()
    rule = mt.fourth_down_rule(base.kern)
    d0 = mt.drive_points(base.kern, rule=rule)
    d1 = mt.drive_points(model.kern, rule=rule)
    f42 = int(g.fidx(4, g.dist_bin(2), 60))
    f32 = int(g.fidx(3, g.dist_bin(2), 60))
    third0 = base.kern.k_all["neutral"][f32]
    third1 = model.kern.k_all["neutral"][f32]

    def made(row: Any) -> float:
        r = row.toarray().ravel()
        _, _, yl = g.unpack(np.arange(g.nf))
        keep = np.nonzero(r[: g.nf])[0]
        return float(r[keep][yl[keep] <= 58].sum() + r[g.col_td])

    rec = {
        **c.as_dict(),
        "offense": off,
        "defense": deff,
        "by_start": {
            str(y): {
                "m8_shift": eff.drive_points_shift(off, deff, y),
                "model_shift": float(d1[f10[y]] - d0[f10[y]]),
            }
            for y in CHECK_STARTS
        },
        "p_convert_4th_and_2_own_40": {"base": float(base.kern.p_convert[f42]),
                                        "matchup": float(model.kern.p_convert[f42])},
        "p_first_down_3rd_and_2_own_40": {"base": made(third0), "matchup": made(third1)},
    }  # fmt: skip
    return model, rec


def maps_at(sol: mg.Solution, q: dict, head: dict) -> dict[str, Any]:
    """go (best play mode) and punt values over [distance, yard line] at the headline time,
    lead and timeouts, plus the headline cell."""
    g, m = sol.grid, sol.model
    yls = np.arange(q["yardlines"][0], q["yardlines"][1] + 1)
    ds = np.arange(q["distances"][0], q["distances"][1] + 1)
    f = g.fidx(4, g.dist_bin(ds)[:, None], yls[None, :])
    qs = sol.q_slice(head["seconds"] // m.clock.tick_seconds)
    s = head["lead"] + m.score_max
    a, b = q["offense_timeouts"], head["defense_timeouts"]
    go = np.max([qs[c][f, s, a, b] for c in PLAY_CODES], axis=0)
    punt = qs[mg.A_PUNT][f, s, a, b]
    fg = qs[mg.A_FG][f, s, a, b]
    i = list(ds).index(head["ydstogo"])
    j = list(yls).index(head["yardline_100"])
    return {"go": go, "punt": punt, "field_goal": fg, "head": (i, j)}


def frontier(diff: np.ndarray, ds: np.ndarray) -> list[int]:
    """Longest distance at which going beats punting, per yard line (0 if never)."""
    out = []
    for j in range(diff.shape[1]):
        win = np.nonzero(diff[:, j] > 0)[0]
        out.append(int(ds[win].max()) if len(win) else 0)
    return out


# --------------------------------------------------------------------------- benchmark


def benchmark_child(out_dir: str) -> None:
    """Whole-game two-team solve (Detroit vs an average team), run in its own process so
    its peak memory is its own. Saves timing, memory and two end-phase slices."""
    base = mg.load_solution().model
    eff = mt.RatingEffects.from_bundle(joblib.load(BUNDLE))
    mu = [x for x in matchups(season_end_2025()) if x["key"] == "det_vs_average"][0]
    t0 = time.time()
    ma, _ = mt.matchup_model(base, mu["a_off"], mu["b_def"], eff)
    mb, _ = mt.matchup_model(base, mu["b_off"], mu["a_def"], eff)
    kernel_seconds = time.time() - t0
    pair = mt.solve_matchup(ma, mb)
    np.save(Path(out_dir) / "end_slice_a.npy", np.asarray(pair.sols[0].v["end"][48]))
    np.save(Path(out_dir) / "end_slice_b.npy", np.asarray(pair.sols[1].v["end"][48]))
    write_json({
        "matchup": mu["key"],
        "kernel_seconds": kernel_seconds,
        "solve_seconds": pair.meta["solve_seconds"],
        "peak_memory_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "states_per_team": pair.meta["n_states_per_team"]["total"],
        "value_array_gb_per_team": sum(v.nbytes for v in pair.sols[0].v.values()) / 1024**3,
        "policy_array_gb_per_team": sum(v.nbytes for v in pair.sols[0].pol.values()) / 1024**3,
        "threads": int(os.environ.get("OMP_NUM_THREADS", "0")),
    }, Path(out_dir) / "benchmark.json")  # fmt: skip


def run_benchmark() -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, __file__, "--benchmark-child", tmp], check=True)
        bench = read_json(Path(tmp) / "benchmark.json")
        return bench, np.load(Path(tmp) / "end_slice_a.npy"), np.load(Path(tmp) / "end_slice_b.npy")


# --------------------------------------------------------------------------- figure


def figure(res: dict[str, Any]) -> None:
    style.apply()
    q = res["map"]
    yls = np.arange(q["yardlines"][0], q["yardlines"][1] + 1)
    ds = np.arange(q["distances"][0], q["distances"][1] + 1)
    keys = [m["key"] for m in res["matchups"]]
    cmap = LinearSegmentedColormap.from_list("go_punt", [style.AQUA, "#ecebe6", style.BLUE])
    lim = 12
    fig, axes = plt.subplots(1, len(keys), figsize=(17, 4.4), sharey=True)
    xs = 100 - yls[::-1]
    extent = (xs[0] - 0.5, xs[-1] + 0.5, ds[0] - 0.5, ds[-1] + 0.5)
    base = np.asarray(res["matchups"][0]["map_go_minus_punt_points"])[:, ::-1]
    im = None
    for ax, mu in zip(axes, res["matchups"], strict=True):
        z = np.asarray(mu["map_go_minus_punt_points"])[:, ::-1]
        im = ax.imshow(z, origin="lower", aspect="auto", cmap=cmap, vmin=-lim, vmax=lim,
                       extent=extent, interpolation="nearest")  # fmt: skip
        ax.contour(xs, ds, z, levels=[0.0], colors=[style.INK], linewidths=1.4)
        if mu["key"] != "identical":
            ax.contour(xs, ds, base, levels=[0.0], colors=[style.INK], linewidths=1.0,
                       linestyles="dashed")  # fmt: skip
        h = res["headline"]
        ax.scatter([100 - h["yardline_100"]], [h["ydstogo"]], s=60, facecolor="none",
                   edgecolor=style.INK, linewidths=1.5, zorder=5)  # fmt: skip
        ax.grid(False)
        ax.set_xticks([20, 30, 40, 50])
        ax.set_yticks([1, 4, 7, 10])
        ax.set_xlabel("Own yard line")
        hd = mu["headline"]
        everywhere = " (all)" if mu["go_cells"] == mu["n_cells"] else ""
        ax.text(0, 1.17, mu["label"], transform=ax.transAxes, fontsize=11, fontweight="bold",
                color=style.INK, va="bottom")  # fmt: skip
        ax.text(0, 1.02, f"4th and 2, own 40: go by {hd['go_minus_punt_points']:+.1f} pts\n"
                f"go wins in {mu['go_cells']} of {mu['n_cells']} cells{everywhere}",
                transform=ax.transAxes, fontsize=9, color=style.INK_2, va="bottom")  # fmt: skip
    axes[0].set_ylabel("Yards to go")
    cax = fig.add_axes([0.30, -0.08, 0.40, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-12, -8, -4, 0, 4, 8, 12])
    cb.set_ticklabels(["punt +12", "+8", "+4", "even", "+4", "+8", "go +12"])
    cb.set_label("Win probability of going minus punting (points). Solid line: break-even; "
                 "dashed: break-even for identical teams", color=style.INK_2)  # fmt: skip
    cb.outline.set_visible(False)
    fig.text(0.125, 1.16, "Up 4 with 4:00 left, 3 timeouts each: go or punt, by matchup",
             fontsize=14, fontweight="bold", color=style.INK)  # fmt: skip
    fig.text(0.125, 1.105, "Exact two-team solution. End-of-2025 M8 ratings; Detroit's offense "
             "and defense, opponents average except where named. No uncertainty interval yet.",
             fontsize=10, color=style.INK_2)  # fmt: skip
    style.save(fig, FIG)


# --------------------------------------------------------------------------- main


def main() -> None:
    cfg = mdp.load_config()
    q = cfg["question1"]
    head = dict(q["headline"])
    saved = mg.load_solution()
    full_model = saved.model
    base = mt.endgame_model(full_model)
    bundle = joblib.load(BUNDLE)
    eff = mt.RatingEffects.from_bundle(bundle)
    season_end = season_end_2025()
    mus = matchups(season_end)
    print("whole-game benchmark (child process) ...", flush=True)
    bench, slice_a, slice_b = run_benchmark()
    print(f"  {bench['solve_seconds']:.1f} s, {bench['peak_memory_gb']:.2f} GB", flush=True)

    ds = np.arange(q["distances"][0], q["distances"][1] + 1)
    yls = np.arange(q["yardlines"][0], q["yardlines"][1] + 1)
    out_mus = []
    for mu in mus:
        t0 = time.time()
        ma, rec_a = calibration_record(base, mu["a_off"], mu["b_def"], eff)
        mb, rec_b = calibration_record(base, mu["b_off"], mu["a_def"], eff)
        pair = mt.solve_matchup(ma, mb)
        mp = maps_at(pair.sols[0], q, head)
        diff = 100.0 * (mp["go"] - mp["punt"])
        i, j = mp["head"]
        row: dict[str, Any] = {
            "key": mu["key"],
            "label": mu["label"],
            "a_offense_vs_b_defense": rec_a,
            "b_offense_vs_a_defense": rec_b,
            "solve_seconds_end_phase": time.time() - t0,
            "headline": {
                "go": float(mp["go"][i, j]),
                "punt": float(mp["punt"][i, j]),
                "field_goal": float(mp["field_goal"][i, j]),
                "go_minus_punt_points": float(diff[i, j]),
            },
            "map_go_minus_punt_points": np.round(diff, 3).tolist(),
            "go_cells": int((diff > 0).sum()),
            "n_cells": int(diff.size),
            "go_frontier_by_yardline": dict(
                zip([int(y) for y in yls], frontier(diff, ds), strict=True)
            ),  # fmt: skip
        }
        if mu["key"] == "identical":
            row["max_abs_diff_vs_saved_solution"] = {
                name: float(max(np.abs(s.v["end"][t] - np.asarray(saved.v["end"][t])).max()
                                for t in range(saved.v["end"].shape[0])))
                for name, s in zip(("A", "B"), pair.sols, strict=True)
            }  # fmt: skip
        if mu["key"] == bench["matchup"]:
            row["max_abs_diff_vs_whole_game_solve_at_4_00"] = {
                "A": float(np.abs(pair.sols[0].v["end"][48] - slice_a).max()),
                "B": float(np.abs(pair.sols[1].v["end"][48] - slice_b).max()),
            }
        print(f"  {mu['key']:26s} go-punt {diff[i, j]:+.2f} pts, go cells {row['go_cells']}, "
              f"{row['solve_seconds_end_phase']:.0f} s", flush=True)  # fmt: skip
        out_mus.append(row)
        del pair, ma, mb
        gc.collect()

    base_row = out_mus[0]
    base_diff = np.asarray(base_row["map_go_minus_punt_points"])
    for row in out_mus[1:]:
        d = np.asarray(row["map_go_minus_punt_points"])
        fr0 = np.array(list(base_row["go_frontier_by_yardline"].values()))
        fr1 = np.array(list(row["go_frontier_by_yardline"].values()))
        row["versus_identical"] = {
            "headline_change_points": row["headline"]["go_minus_punt_points"]
            - base_row["headline"]["go_minus_punt_points"],
            "go_cells_change": row["go_cells"] - base_row["go_cells"],
            "cells_flipped_to_go": int(((d > 0) & (base_diff <= 0)).sum()),
            "cells_flipped_to_punt": int(((d <= 0) & (base_diff > 0)).sum()),
            "mean_change_go_minus_punt_points": float((d - base_diff).mean()),
            "frontier_change_yards_mean": float((fr1 - fr0).mean()),
            "frontier_change_yards_range": [int((fr1 - fr0).min()), int((fr1 - fr0).max())],
        }

    res = {
        "what": "Question 6: the late-lead decision with two different teams, solved exactly "
        "with two value arrays (src/fourth_and_rl/mdp_team.py)",
        "headline": head | {"offense_timeouts": q["offense_timeouts"]},
        "map": {
            "yardlines": q["yardlines"],
            "distances": q["distances"],
            "seconds": head["seconds"],
            "lead": head["lead"],
            "offense_timeouts": q["offense_timeouts"],
            "defense_timeouts": head["defense_timeouts"],
        },  # fmt: skip
        "ratings": {
            "source": "results/m8_team_strength.json test.ratings_season_end['2025']: frozen "
            "output of M8's test stage (the M8 filter run through the 2024 and 2025 test "
            "seasons with constants tuned on development seasons). Read, not recomputed. "
            "Groups by team_strength.group_ratings over 2025 (k = 5, ranked by drive rating).",
            "season": SEASON,
            "detroit": season_end.filter(pl.col("team") == TEAM).to_dicts()[0],
            "groups_2025": ts.group_ratings(season_end, [SEASON], k=5),
        },
        "kernel_method": {
            "effect_models": "data/models/team_strength.joblib models conv_no_spread and "
            "drive_no_spread (ratings without the spread, fitted on 2012 to 2023): the "
            "exact game has no spread, so the whole effect of the ratings is used",
            "calibration_start_yardline_100": mt.CALIBRATION_START,
            "fourth_down_rule_for_drive_points": "field goal if P(make) >= 0.6, else go on "
            "fourth and 1 from the own 40 on, else punt",
            "turnover_scale": 1.0,
            "kicking": "field goals, punts, kickoffs and tries league average for both teams",
        },
        "benchmark_whole_game": bench,
        "matchups": out_mus,
        "caveats": [
            "No bootstrap: no uncertainty interval on any value here yet. M8's rating SDs "
            "(Detroit offense 0.40 +/- 0.31 points per drive) are large.",
            "Ratings move plays only (yards tilt and conversion log-odds); turnovers, "
            "kicking and clock behavior are league average.",
            "End-of-2025 ratings come from M8's test-season stage (frozen artifact).",
            "Only the end phase is solved per matchup; it equals a whole-game solve's end "
            "phase (checked on one matchup above).",
        ],
    }
    write_json(res, OUT)
    figure(res)
    print(f"wrote {OUT} and {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-child", default=None)
    ap.add_argument("--figure-only", action="store_true")
    args = ap.parse_args()
    if args.benchmark_child:
        benchmark_child(args.benchmark_child)
    elif args.figure_only:
        figure(read_json(OUT))
    else:
        main()
