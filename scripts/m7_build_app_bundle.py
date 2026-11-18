"""M7: build the precomputed app bundle under app/data/ (committed).

The app, the static page and the notebook cannot use the fitted models or the solved game
(gitignored, about 2 GB), so this script evaluates a fixed grid of fourth-down situations
once and saves compact arrays (`fourth_and_rl.app_bundle` holds the schema):

  * the headline one-step engine (decision.load_engine): go, field goal and punt win
    probability, conversion and field-goal chances;
  * the game-level bootstrap (the 101 replicates of M3 under data/models/boot/): for each
    cell the 5th and 95th percentile of each action's win probability and of the go gain
    (go minus the better kick), and how many replicates prefer each action;
  * the exactly solved game (M4, mdp_game.load_solution): go (best play mode), field goal
    and punt values;
  * the question 1 and question 5 maps, copied from their saved results.

The grid is in configs/app.yaml (`grid`). Nothing is interpolated. Checks written into the
bundle: the exact values equal the saved question 1 map; how much a +-7 spread moves the
one-step values late in the game (why the spread is a grid axis at every clock).

No test-season or holdout play is read. `decision.load_engine` reads the development file
(and would read only kickoff rows of the test seasons if it had to refit its bundle).

Run:  python scripts/m7_build_app_bundle.py                  (about 4.5 hours the first time
                                                              on 3 threads; replicate values
                                                              are cached and resumable)
      python scripts/m7_build_app_bundle.py --n-boot 5 --out data/app_build/trial
                                                             (quick trial bundle elsewhere)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

from fourth_and_rl import app_bundle as ab
from fourth_and_rl import bootstrap, decision, fingerprint, splits, wp
from fourth_and_rl import endgame_questions as eq
from fourth_and_rl import mdp_game as mg
from fourth_and_rl.io import read_json
from fourth_and_rl.paths import ROOT

ONE_KEYS = ("go", "fg", "punt")


# ----------------------------------------------------------------------------- states


def grid_axes(grid: dict) -> dict:
    return {
        "scores": ab.axis(grid["scores"]),
        "distances": ab.axis(grid["distances"]),
        "yardlines": ab.axis(grid["yardlines"]),
    }


def grid_states(grid: dict, ctxs: list[ab.Context]) -> tuple[pl.DataFrame, np.ndarray]:
    """Fourth-down states of every valid cell, in the bundle's dense C order
    [context, score, distance, yard line], and the flat positions of those cells."""
    ax = grid_axes(grid)
    c, s, d, y = np.meshgrid(np.arange(len(ctxs)), ax["scores"], ax["distances"],
                             ax["yardlines"], indexing="ij")  # fmt: skip
    c, s, d, y = (a.ravel() for a in (c, s, d, y))
    keep = d <= y
    pos = np.flatnonzero(keep)
    c, s, d, y = c[keep], s[keep], d[keep], y[keep]
    sec = np.array([x.seconds for x in ctxs])[c]
    opp = np.array([x.opp_timeouts for x in ctxs])[c]
    spread = np.array([x.spread for x in ctxs])[c]
    states = wp.state_frame(
        down=4, ydstogo=d, yardline_100=y, game_seconds_remaining=sec, score_differential=s,
        posteam_timeouts_remaining=grid["own_timeouts"], defteam_timeouts_remaining=opp,
        posteam_spread=spread, receive_2h_ko=grid["receive_2h_ko"], is_home=0,
    )  # fmt: skip
    return states, pos


def dense(values: np.ndarray, pos: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    out = np.full(int(np.prod(shape)), np.nan)
    out[pos] = values
    return out.reshape(shape)


def states_tag(states: pl.DataFrame, engine_config: dict) -> str:
    cols = states.select(wp.STATE_COLUMNS).cast(pl.Float64).to_numpy()
    h = hashlib.sha1(cols.tobytes(), usedforsecurity=False)
    h.update(json.dumps(engine_config, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


# -------------------------------------------------------------------------- bootstrap


def replicate_table(
    states: pl.DataFrame, directory: Path, n_boot: int, engine_config: dict, cache: Path,
    threads: int,
) -> list[np.ndarray]:
    """Go, field goal and punt values (n_states, 3) under every replicate, memory-mapped.

    Each replicate's values are cached under `cache` with the hash of the replicate's key
    file in the name (as bootstrap.replicate_values does), so a rerun reuses only values
    made by exactly those models, and a stopped build resumes where it stopped. The arrays
    come back memory-mapped (101 x 3.5 million x 3 floats do not fit the memory budget)."""
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for b in range(n_boot):
        path = cache / f"rep_{b:03d}_{bootstrap.replicate_tag(directory, b)}.npy"
        if not path.exists():
            t0 = time.perf_counter()
            eng = bootstrap.load_replicate_engine(directory, b, engine_config)
            eng.wp_model.n_threads = threads
            v = eng.action_values(states)
            vals = np.column_stack([v["go_wp"], v["fg_wp"], v["punt_wp"]]).astype(np.float32)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, vals)
            tmp.replace(path)
            print(f"replicate {b + 1}/{n_boot}: {time.perf_counter() - t0:.0f}s", flush=True)
        arr = np.load(path, mmap_mode="r")
        if arr.shape != (states.height, 3):
            raise RuntimeError(f"{path} has shape {arr.shape}, expected {(states.height, 3)}")
        out.append(arr)
    return out


def boot_summaries(
    point: np.ndarray, boot: list[np.ndarray], cfg: dict, chunk: int = 100_000
) -> dict[str, np.ndarray]:
    """bootstrap.summarize in chunks of states (memory), plus each action's percentiles."""
    t = cfg["thresholds"]
    lo_q, hi_q = cfg["gain_interval"]
    n_boot, n = len(boot), point.shape[0]
    lo_i, hi_i = int(round(lo_q * (n_boot - 1))), int(round(hi_q * (n_boot - 1)))
    keys = ["gain_lo", "gain_hi", "share_go", "share_field_goal", "share_punt"]
    out = {k: np.empty(n) for k in keys}
    for k in ONE_KEYS:
        out[f"{k}_lo"], out[f"{k}_hi"] = np.empty(n), np.empty(n)
    for start in range(0, n, chunk):
        s = slice(start, min(start + chunk, n))
        block = np.stack([np.asarray(r[s]) for r in boot])  # (n_boot, chunk, 3)
        summ = bootstrap.summarize(point[s], block, t["confident"], t["lean"], (lo_q, hi_q))
        for k in keys:
            out[k][s] = summ[k]
        for i, k in enumerate(ONE_KEYS):
            # NaN (action not available) sorts last; such cells are NaN at every replicate.
            ordered = np.sort(block[:, :, i].astype(float), axis=0)
            out[f"{k}_lo"][s], out[f"{k}_hi"][s] = ordered[lo_i], ordered[hi_i]
    return out


# ------------------------------------------------------------------------------ exact


def exact_values(
    sol: mg.Solution, grid: dict, ectxs: list[ab.Context]
) -> dict[str, np.ndarray]:
    """Go (best of the play modes), field goal and punt values of the solved game, dense
    [exact context, score, distance, yard line]. NaN where an action is not allowed."""
    ax = grid_axes(grid)
    g = sol.grid
    dd, yy = np.meshgrid(ax["distances"], ax["yardlines"], indexing="ij")
    ok = dd <= yy
    f = np.asarray(g.fidx(4, g.dist_bin(np.minimum(dd, yy)), yy))
    si = eq.score_index(sol, ax["scores"])
    shape = (len(ectxs), len(si), *dd.shape)
    out = {k: np.full(shape, np.nan) for k in ONE_KEYS}
    order = sorted(range(len(ectxs)), key=lambda i: ectxs[i].seconds)  # one q_slice per clock
    for i in order:
        c = ectxs[i]
        t = eq.tick_of(sol, c.seconds)
        q = sol.q_slice(t)
        e1, e2 = eq.extra_index(sol, t, (grid["own_timeouts"], c.opp_timeouts,
                                         grid["receive_2h_ko"]))  # fmt: skip
        cell = (f[None], si[:, None, None], e1, e2)
        go = np.stack([q[code][cell].astype(float) for code in eq.GO_CODES if code in q])
        vals = {"go": go.max(axis=0)}
        for k, code in (("fg", mg.A_FG), ("punt", mg.A_PUNT)):
            vals[k] = q[code][cell].astype(float) if code in q else np.full(go.shape[1:], np.nan)
        for k, v in vals.items():
            v = np.where(np.isfinite(v), v, np.nan)
            out[k][i] = np.where(ok[None], v, np.nan)
    return out


def check_question1(exact: dict, ectxs: list[ab.Context], grid: dict, path: Path,
                    tol: float) -> dict:  # fmt: skip
    """The bundle's exact go and punt values against the saved question 1 map."""
    q1 = read_json(path)["grid"]
    ax = grid_axes(grid)
    worst = 0.0
    cells = 0
    for a, sec in enumerate(q1["seconds"]):
        for b, lead in enumerate(q1["leads"]):
            for c, to in enumerate(q1["opponent_timeouts"]):
                i = ectxs.index(ab.Context(int(sec), int(to), 0.0))
                j = ax["scores"].index(int(lead))
                for d_i, d in enumerate(q1["ydstogo"]):
                    for y_i, y in enumerate(q1["yardline_100"]):
                        mine = (exact["go"][i, j, d - 1, y - 1], exact["punt"][i, j, d - 1, y - 1])
                        saved = (q1["go_wp"][a][b][c][d_i][y_i], q1["punt_wp"][a][b][c][d_i][y_i])
                        for m, s in zip(mine, saved, strict=True):
                            worst = max(worst, abs(float(m) - float(s)))
                        cells += 1
    if worst > tol:
        raise SystemExit(f"exact values differ from the saved question 1 map by {worst}")
    return {"cells": cells, "max_abs_difference": worst, "tolerance": tol,
            "source": str(path.relative_to(ROOT))}  # fmt: skip


# -------------------------------------------------------------------------- questions


def question_extracts(cfg: dict) -> dict:
    """The maps and headline numbers the question tabs show, copied from saved results."""
    q = cfg["questions"]
    q1 = read_json(ROOT / q["q1_dp"])
    q1s = read_json(ROOT / q["q1_one_step"])
    q5 = read_json(ROOT / q["q5"])
    head = q1["headline"]
    return {
        "q1": {
            "source": [q["q1_dp"], q["q1_one_step"]],
            "question": q1["question"],
            "situation": head["situation"],
            "exact": {k: head["exact"][k] for k in (
                "go", "punt", "go_minus_punt", "best_action", "go_mode", "p_convert",
                "win_probability_if_converted", "win_probability_if_failed",
                "break_even_p_convert")},  # fmt: skip
            "sensitivity": [{k: r.get(k) for k in ("scenario", "headline_go_minus_punt",
                                               "share_of_map_where_go_beats_punt")}
                            for r in q1["sensitivity"]],  # fmt: skip
            "dp_share_go": q1["one_step_comparison"]["dp_share_go"],
            "one_step_share_go": q1["one_step_comparison"]["one_step_share_go"],
            "gap_accounting": q1["one_step_comparison"]["headline_gap_accounting"],
            "coaches": {"definition": q1s["coaches"]["definition"], "n": q1s["coaches"]["n"],
                        "by_distance": q1s["coaches"]["by_distance"]},  # fmt: skip
        },
        "q5": {
            "source": q["q5"],
            "assumptions": q5["assumptions"],
            "goal_line_late": {"headline": q5["q1_goal_line_late"]["headline"],
                               "comparison_case": q5["q1_goal_line_late"]["comparison_case"],
                               "grid": q5["q1_goal_line_late"]["grid"]},  # fmt: skip
            "two_point": {"headline": q5["q2_two_point"]["headline"],
                          "grid": q5["q2_two_point"]["grid"]},  # fmt: skip
            "goal_line_early": {"headline": q5["q3_goal_line_early"]["headline"],
                                "grid": q5["q3_goal_line_early"]["grid"]},  # fmt: skip
            "sensitivity": q5["sensitivity"],
        },
    }


# ----------------------------------------------------------------------------- checks


def spread_check(engine: decision.DecisionEngine, cfg: dict) -> dict:
    """How far a +-7 spread moves the one-step values late in the game (a diagnostic: it is
    why the spread is a grid axis at every clock)."""
    sc, grid = cfg["spread_check"], cfg["grid"]
    rows = []
    for sec in sc["seconds"]:
        for score in sc["scores"]:
            frames = {}
            for sp in (-7.0, 0.0, 7.0):
                st = decision.chart_states(ab.axis(grid["yardlines"]), ab.axis(grid["distances"]),
                                           {"game_seconds_remaining": sec,
                                            "score_differential": score, "posteam_spread": sp,
                                            "posteam_timeouts_remaining": grid["own_timeouts"],
                                            "defteam_timeouts_remaining": sc["opp_timeouts"],
                                            "receive_2h_ko": 0, "is_home": 0})  # fmt: skip
                v = engine.action_values(st)
                frames[sp] = np.column_stack([v["go_wp"], v["fg_wp"], v["punt_wp"]])
            base = frames[0.0]
            best0 = decision.summarize_actions(base)[0]
            for sp in (-7.0, 7.0):
                diff = np.abs(frames[sp] - base)
                rows.append({
                    "seconds": sec, "score": score, "spread": sp,
                    "max_abs_change": float(np.nanmax(diff)),
                    "mean_abs_change": float(np.nanmean(diff)),
                    "share_best_action_changes": float(np.mean(
                        decision.summarize_actions(frames[sp])[0] != best0)),
                })  # fmt: skip
    return {
        "what": "one-step values with spread -7 or +7 minus spread 0, late in the game "
                "(why the spread is a grid axis at every clock)",
        "max_abs_change": max(r["max_abs_change"] for r in rows),
        "max_share_best_action_changes": max(r["share_best_action_changes"] for r in rows),
        "rows": rows,
    }  # fmt: skip


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None


def stamp_summary(stamp: dict | None) -> dict | None:
    """A model stamp without its bulky data fingerprint (its hash is kept)."""
    if stamp is None:
        return None
    out = {}
    for k, v in stamp.items():
        if k == "data_fingerprint" and isinstance(v, dict):
            out["data_fingerprint_sha1"] = hashlib.sha1(
                json.dumps(v, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:12]
        elif isinstance(v, dict):
            out[k] = stamp_summary(v)
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-boot", type=int, default=None, help="replicates (default: config)")
    parser.add_argument("--out", default=None, help="bundle folder (default: configs/app.yaml)")
    args = parser.parse_args()

    started = time.perf_counter()
    cfg = ab.load_config()
    grid = cfg["grid"]
    bcfg = bootstrap.load_config()
    dcfg = decision.load_config()
    n_boot = int(args.n_boot or bcfg["n_boot"])
    if n_boot != int(bcfg["n_boot"]) and args.out is None:
        raise SystemExit("a trial with fewer replicates needs --out (never the committed bundle)")
    out_dir = ROOT / (args.out or cfg["bundle_dir"])

    plays = splits.load_dev_plays()
    directory = bootstrap.boot_dir(bcfg)
    bootstrap.assert_replicates_current(directory, int(bcfg["n_boot"]), bcfg, wp.load_config(),
                                        data_fingerprint=fingerprint.data_fingerprint(plays))
    engine = decision.load_engine(plays, dcfg)
    del plays
    engine.wp_model.n_threads = int(cfg["threads"])

    ctxs, ectxs = ab.contexts(grid), ab.exact_contexts(grid)
    ax = grid_axes(grid)
    shape = (len(ctxs), len(ax["scores"]), len(ax["distances"]), len(ax["yardlines"]))
    states, pos = grid_states(grid, ctxs)
    print(f"{len(ctxs)} contexts, {states.height:,} valid cells", flush=True)

    t0 = time.perf_counter()
    ev = engine.evaluate(states)
    point = np.column_stack([ev[c].fill_null(np.nan).to_numpy() for c in
                             ("go_wp", "fg_wp", "punt_wp")])  # fmt: skip
    point_seconds = time.perf_counter() - t0
    print(f"point estimates: {point_seconds:.0f}s", flush=True)

    t0 = time.perf_counter()
    tag = states_tag(states, dcfg["engine"])
    boot = replicate_table(states, directory, n_boot, dcfg["engine"],
                           ROOT / cfg["cache_dir"] / f"values_{tag}", int(cfg["threads"]))
    boot_seconds = time.perf_counter() - t0
    summ = boot_summaries(point, boot, bcfg)

    es = cfg["encoding"]
    ws, gs = int(es["wp_scale"]), int(es["gain_scale"])
    one = {}
    for i, k in enumerate(ONE_KEYS):
        one[k] = ab.encode_wp(dense(point[:, i], pos, shape), ws)
        one[f"{k}_lo"] = ab.encode_wp(dense(summ[f"{k}_lo"], pos, shape), ws)
        one[f"{k}_hi"] = ab.encode_wp(dense(summ[f"{k}_hi"], pos, shape), ws)
    one["gain_lo"] = ab.encode_gain(dense(summ["gain_lo"], pos, shape), gs)
    one["gain_hi"] = ab.encode_gain(dense(summ["gain_hi"], pos, shape), gs)
    for key, share in (("n_go", "share_go"), ("n_fg", "share_field_goal"),
                       ("n_punt", "share_punt")):  # fmt: skip
        n = np.round(np.nan_to_num(dense(summ[share], pos, shape)) * n_boot)
        one[key] = n.astype(np.uint8)
    one["p_convert"] = ab.encode_wp(dense(ev["p_convert"].fill_null(np.nan).to_numpy(), pos,
                                          shape), ws)  # fmt: skip
    one["p_fg"] = ab.encode_wp(dense(ev["p_fg"].fill_null(np.nan).to_numpy(), pos, shape), ws)

    t0 = time.perf_counter()
    sol = mg.load_solution()
    exact_f = exact_values(sol, grid, ectxs)
    exact = {k: ab.encode_wp(v, ws) for k, v in exact_f.items()}
    exact_seconds = time.perf_counter() - t0
    q1_check = check_question1(exact_f, ectxs, grid, ROOT / cfg["questions"]["q1_dp"],
                               float(cfg["questions"]["q1_check_tolerance"]))  # fmt: skip
    spreads = spread_check(engine, cfg)

    # Round-trip check of the encodings on the stored point values.
    dec = ab.decode_wp(one["go"], ws).ravel()[pos]
    roundtrip = float(np.nanmax(np.abs(dec - point[:, 0])))

    sol_stamp = sol.meta.get("stamp", {})
    meta = {
        "schema_version": ab.SCHEMA_VERSION,
        "synthetic": False,
        "built_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "built_by": "scripts/m7_build_app_bundle.py",
        "git_commit_at_build": git_commit(),
        "config": "configs/app.yaml",
        "axes": {
            "contexts": [c.__dict__ for c in ctxs],
            "exact_contexts": [c.__dict__ for c in ectxs],
            "scores": ax["scores"], "distances": ax["distances"], "yardlines": ax["yardlines"],
            "clocks": grid["clocks"],
            "array_axes": "[context, score, distance - 1, yardline_100 - 1]",
        },  # fmt: skip
        "fixed": {
            "own_timeouts": grid["own_timeouts"],
            "opp_timeouts_before_endgame": 3,
            "receive_2h_ko": grid["receive_2h_ko"],
            "is_home": 0,
            "endgame_seconds": grid["endgame_seconds"],
            "spread_min_seconds": grid["spread_min_seconds"],
        },
        "encoding": {
            "wp_scale": ws, "gain_scale": gs, "wp_not_available": int(ab.WP_NA),
            "gain_not_available": int(ab.GAIN_NA), "max_roundtrip_error": roundtrip,
        },  # fmt: skip
        "one_step": {
            "what": "headline one-step engine (decision.load_engine), point estimates",
            "wp_model": dcfg["models"]["wp_path"],
            "wp_model_stamp": stamp_summary(getattr(engine.wp_model, "stamp", None)),
            "components": dcfg["models"]["components_cache"],
            "components_stamp": stamp_summary(engine.stamp),
            "kickoff_rule": "latest rule era's kickoff table (hypothetical states)",
            "punt_not_available_inside_yardline_100": dcfg["engine"]["punt_min_yardline_100"],
            "seconds": round(point_seconds, 1),
        },
        "bootstrap": {
            "what": "game-level bootstrap replicates of M3, each refitted on resampled "
                    "development games and evaluated on every cell",
            "n_boot": n_boot,
            "seed": int(bcfg["seed"]),
            "directory": str(directory.relative_to(ROOT)),
            "thresholds": bcfg["thresholds"],
            "interval": bcfg["gain_interval"],
            "interval_note": "order statistics of the replicate values (6th and 96th of 101): "
                             "a 90% interval; under-covers for tree models (M3b, about 77%)",
            "replicate_key_tags": [bootstrap.replicate_tag(directory, b) for b in range(n_boot)],
            "seconds_this_run": round(boot_seconds, 1),
        },  # fmt: skip
        "exact": {
            "what": "exactly solved game (mdp_game.load_solution, data/mdp)",
            "stamp": stamp_summary(mg.comparable(sol_stamp)),
            "solve_seed": sol.meta.get("seed"),
            "tick_seconds": sol.model.clock.tick_seconds,
            "dist_exact": sol.grid.dist_exact,
            "dist_bins": [list(b) for b in sol.grid.dist_bins],
            "score_max": sol.model.score_max,
            "go": "best of the play modes (neutral, protect, chase) on fourth down",
            "seconds": round(exact_seconds, 1),
        },
        "coverage": {
            "interpolated": "nothing; every number is a computed grid cell and the app "
                            "snaps its controls to the grid",
            "not_covered": [
                "clocks other than the listed ones (for example the end of the first half)",
                "score differentials beyond -14..+14",
                "the team with the ball holding fewer than 3 timeouts",
                "opponent timeouts other than 3 before the last 5:00",
                "spreads other than -7, 0 and +7, and any spread in the exact game "
                "(identical teams)",
                "receiving the second-half kickoff (both engines assume the team with the "
                "ball does not)",
                "overtime",
            ],
            "exact_distance_bins": "the exact game treats 11 to 15 yards to go as one bin "
                                   "valued at 13 yards",
        },  # fmt: skip
        "checks": {"question1_exact_map": q1_check, "spread_check": spreads},
        "limits": [
            "The exact game has identical, league-average teams; it must not be read for "
            "anything that depends on team strength.",
            "Conversion chances where coaches almost never go for it are extrapolated from "
            "third downs (the central caveat of M4 and question 1).",
            "The one-step engine trusts a win-probability model fitted on real games for "
            "everything after the next play; the exact game assumes optimal play by both "
            "teams afterwards.",
            "Bootstrap intervals cover refitting on resampled games, not model form; they "
            "are somewhat too narrow (M3b).",
        ],
        "seasons": {"fit": "development seasons 2010 to 2023 for every model", "test": "none read"},
        "runtime": {
            "total_seconds": round(time.perf_counter() - started, 1),
            "threads": int(cfg["threads"]),
            "peak_memory_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        },
    }
    bundle = ab.Bundle(meta, one, exact, question_extracts(cfg))
    sizes = ab.save_bundle(bundle, out_dir, bool(es.get("delta", True)))
    ab.load_bundle(out_dir)  # round trip
    total = sum(sizes.values())
    print(json.dumps({k: f"{v / 1e6:.2f} MB" for k, v in sizes.items()}, indent=1))
    print(f"bundle {total / 1e6:.2f} MB in {out_dir}; spread check max change "
          f"{spreads['max_abs_change']:.4f}; q1 check {q1_check['max_abs_difference']:.2e}")


if __name__ == "__main__":
    main()
