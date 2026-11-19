"""Many two-team matchups on a GPU: the code behind notebooks/team_matchups_colab.ipynb.

The notebook runs on Colab from a fresh clone. It has no data/ folder, so it rebuilds the
exact inputs of the solved game from public nflverse play-by-play with the repo's own
guarded loaders, in the order the repo's scripts use:

  1. `data.build`               2010 to 2025 play-by-play (configs/data.yaml); the 2026
                                holdout is never downloaded or read (checked here)
  2. `splits.write_split_files` the development (2010 to 2023) and test files
  3. `components.fit_dev_bundle` play-outcome models on development seasons, plus the
                                2024 and 2025 kickoff rule tables (the repo's one logged
                                exception, read through `splits.load_test_plays`)
  4. scripts/solve_mdp.py `load_tables`        count, mode and coach tables (development)
  5. `mdp_game.build_model(..., horizon="endgame")`
  6. scripts/m8_team_strength.py `stage_dev`   M8's effect models (development only; the
                                test stage is not run)

Team ratings are not recomputed: they are the end-of-2025 ratings saved in
results/m8_team_strength.json (test.ratings_season_end["2025"]), the frozen output of M8's
test stage, and are labelled as such.

Each matchup is solved over the end phase with `mdp_team_jax`, and the late-lead decision
of question 1 (up 4, 4:00 left, fourth and 2 at the own 40, three timeouts each) is read
off the action values: go minus punt at that state and the go region over distances 1 to
10 and the own 20 to midfield. Finished matchups are written one file each, so a rerun
skips them (resume). No bootstrap: none of these values has an uncertainty interval.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from fourth_and_rl import mdp, splits, style
from fourth_and_rl import mdp_game as mg
from fourth_and_rl import mdp_team as mt
from fourth_and_rl.io import read_json, write_json
from fourth_and_rl.paths import RESULTS, ROOT

TEAM = "DET"
SEASON = 2025
HOLDOUT_SEASON = 2026
PLAY_CODES = (mg.A_NEUTRAL, mg.A_PROTECT, mg.A_CHASE)
AVERAGE_OFF = {"conv_off": 0.0, "drive_off": 0.0}
AVERAGE_DEF = {"conv_def": 0.0, "drive_def": 0.0}
RATINGS_LABEL = (
    "End-of-2025 team ratings: the frozen output of M8's test stage "
    "(results/m8_team_strength.json, test.ratings_season_end['2025']; the M8 filter run "
    "through the 2024 and 2025 test seasons with constants tuned on 2010 to 2023). "
    "Test-season quantities, used here as inputs to a hypothetical game; nothing is scored."
)
CODE_VERSION = "team-matchups-v1"  # part of each matchup's cache key


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


@dataclass
class Inputs:
    base: mg.GameModel  # end-phase model with the generic (league-average) kernels
    effects: mt.RatingEffects
    provenance: dict[str, Any] = field(default_factory=dict)


def _load_script(name: str) -> Any:
    """Import one of the repo's scripts as a module (scripts/ is not a package)."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_no_holdout(seasons: Sequence[int]) -> None:
    """Refuse any season of the sealed holdout (configs/splits.yaml; 2026)."""
    sealed = set(splits.holdout_seasons()) | {HOLDOUT_SEASON}
    bad = sorted(set(int(s) for s in seasons) & sealed)
    if bad:
        raise splits.LeakageError(f"seasons {bad} are the sealed holdout and are never loaded")


def committed_dev_fingerprint() -> str | None:
    """The development-data hash the committed solved game was built from."""
    path = RESULTS / "m4_mdp_summary.json"
    if not path.exists():
        return None
    try:
        return read_json(path)["fitted_on"]["components"]["data_fingerprint"]["hash"]
    except KeyError:
        return None


def _access_log_lines() -> list[str]:
    log = splits.ACCESS_LOG
    return log.read_text().splitlines() if log.exists() else []


def rebuild_inputs(cache_dir: Path, log: Callable[[str], None] = print) -> Inputs:
    """Rebuild the end-phase game model and M8's effect models from public data (see the
    module docstring). The result is cached in `cache_dir` under the development data's
    fingerprint, so a resumed run does not refit."""
    from fourth_and_rl import components as cm
    from fourth_and_rl import data
    from fourth_and_rl import team_strength as ts
    from fourth_and_rl.fingerprint import data_fingerprint

    cache_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    log_before = _access_log_lines()
    dcfg = data.load_config()
    seasons = data.seasons_from_config(dcfg)
    check_no_holdout(seasons)
    t0 = time.time()
    log(f"play-by-play {seasons[0]} to {seasons[-1]} (nflverse via nflreadpy) ...")
    data.build(dcfg)
    plays = data.load_plays(dcfg)
    check_no_holdout(splits.seasons_in(plays))
    splits.write_split_files(plays)
    del plays
    timings["data_seconds"] = time.time() - t0
    dev = splits.load_dev_plays()
    fp = data_fingerprint(dev)
    committed = committed_dev_fingerprint()
    log(f"development plays: {dev.height:,} rows, fingerprint {fp['hash']} "
        f"(committed solution: {committed}, {'same' if fp['hash'] == committed else 'DIFFERENT'})")
    cache = cache_dir / f"inputs_{fp['hash']}.joblib"
    if cache.exists():
        log(f"reusing fitted inputs from {cache}")
        obj = joblib.load(cache)
        obj.provenance["reused_from_cache"] = True
        return obj
    t0 = time.time()
    ccfg = cm.load_config()
    log("fitting the play-outcome models (development seasons) ...")
    models = cm.fit_dev_bundle(ccfg, dev_plays=dev, seed=ccfg["seed"])
    timings["components_seconds"] = time.time() - t0
    t0 = time.time()
    log("fitting the count, mode and coach tables (development seasons) ...")
    mcfg = mdp.load_config()
    solve_mdp = _load_script("solve_mdp")
    tables = solve_mdp.load_tables(mcfg, plays=dev, models=models)
    model = mg.build_model(mcfg, models, tables["tables"], horizon="endgame")
    timings["tables_seconds"] = time.time() - t0
    t0 = time.time()
    log("fitting M8's effect models (development stage only) ...")
    m8 = _load_script("m8_team_strength")
    tcfg = ts.load_config()
    m8.BUNDLE_PATH[0] = cache_dir / "team_strength_rebuilt.joblib"
    blocks = [list(b) for b in tcfg["filter"]["validation_blocks"]]
    m8.stage_dev(dev, tcfg, blocks)
    effects = mt.RatingEffects.from_bundle(joblib.load(m8.BUNDLE_PATH[0]))
    timings["m8_seconds"] = time.time() - t0
    new_log = _access_log_lines()[len(log_before):]
    prov = {
        "seasons_downloaded": [int(seasons[0]), int(seasons[-1])],
        "holdout_seasons_never_loaded": sorted(set(splits.holdout_seasons())),
        "dev_fingerprint": fp["hash"],
        "dev_rows": int(dev.height),
        "committed_dev_fingerprint": committed,
        "same_data_as_committed_solution": fp["hash"] == committed,
        "test_season_reads_this_run": [json.loads(x) for x in new_log],
        "timings": timings,
        "reused_from_cache": False,
    }
    obj = Inputs(base=model, effects=effects, provenance=prov)
    joblib.dump(obj, cache)
    return obj


def load_local_inputs() -> Inputs:
    """The same inputs from a local checkout that already has data/mdp and the M8 bundle
    (for tests and for comparing with a Colab run)."""
    from fourth_and_rl.paths import DATA

    saved = mg.load_solution()
    eff = mt.RatingEffects.from_bundle(joblib.load(DATA / "models" / "team_strength.joblib"))
    return Inputs(mt.endgame_model(saved.model), eff, {"source": "local data/mdp"})


# --------------------------------------------------------------------------------------
# Ratings and the grid of matchups
# --------------------------------------------------------------------------------------


def season_end_ratings(path: Path | None = None) -> list[dict[str, Any]]:
    """End-of-2025 ratings (see RATINGS_LABEL)."""
    res = read_json(path or RESULTS / "m8_team_strength.json")
    return list(res["test"]["ratings_season_end"][str(SEASON)])


def _off(r: dict[str, Any]) -> dict[str, float]:
    return {"conv_off": float(r["conv_off"]), "drive_off": float(r["drive_off"])}


def _def(r: dict[str, Any]) -> dict[str, float]:
    return {"conv_def": float(r["conv_def"]), "drive_def": float(r["drive_def"])}


def _group(rows: list[dict[str, Any]], keys: tuple[str, ...], k: int, worst_first: bool,
           col: str) -> dict[str, float]:  # fmt: skip
    sel = sorted(rows, key=lambda r: r[col], reverse=worst_first)[:k]
    return {c: float(np.mean([r[c] for r in sel])) for c in keys}


def matchup_grid(ratings: list[dict[str, Any]], team: str = TEAM) -> list[dict[str, Any]]:
    """The matchups, most important first. A is the leading team with the ball.

    - q6: the five matchups of scripts/questions/q6_team_specific.py (a reproduction
      check against results/questions/q6_team_specific.json);
    - vs_def: `team`'s offense and defense against an opponent with an average offense
      and each of the 32 defenses (q6's convention: only the named unit differs);
    - top5/bottom5 offenses: each of the five best and five worst 2025 offenses (by drive
      rating), with an average defense, against the top-5 defense group average.
    Defenses are ranked by drive rating (lower allows fewer points)."""
    by_team = {r["team"]: r for r in ratings}
    me = by_team[team]
    best_def = _group(ratings, ("conv_def", "drive_def"), 5, False, "drive_def")
    worst_def = _group(ratings, ("conv_def", "drive_def"), 5, True, "drive_def")
    bottom_off = _group(ratings, ("conv_off", "drive_off"), 5, False, "drive_off")
    det = {"a_off": _off(me), "a_def": _def(me)}

    def spec(key: str, label: str, group: str, a_off: dict, a_def: dict, b_off: dict,
             b_def: dict, **extra: Any) -> dict[str, Any]:  # fmt: skip
        return {"key": key, "label": label, "group": group, "a_off": a_off, "a_def": a_def,
                "b_off": b_off, "b_def": b_def, **extra}  # fmt: skip

    out = [
        spec("identical", "Identical average teams", "q6", AVERAGE_OFF, AVERAGE_DEF,
             AVERAGE_OFF, AVERAGE_DEF),
        spec("det_vs_top5_def", "Detroit vs a top-5 defense", "q6", **det,
             b_off=AVERAGE_OFF, b_def=best_def),
        spec("det_vs_average", "Detroit vs an average team", "q6", **det,
             b_off=AVERAGE_OFF, b_def=AVERAGE_DEF),
        spec("det_vs_bottom5_def", "Detroit vs a bottom-5 defense", "q6", **det,
             b_off=AVERAGE_OFF, b_def=worst_def),
        spec("bottom5_off_vs_top5_def", "Bottom-5 offense vs a top-5 defense", "q6",
             bottom_off, AVERAGE_DEF, AVERAGE_OFF, best_def),
    ]  # fmt: skip
    ranked_def = sorted(ratings, key=lambda r: r["drive_def"])
    for rank, r in enumerate(ranked_def, start=1):
        out.append(spec(f"{team.lower()}_vs_{r['team'].lower()}_def",
                        f"{team} vs the {r['team']} defense", "vs_def", **det,
                        b_off=AVERAGE_OFF, b_def=_def(r), opponent=r["team"],
                        defense_rank=rank))  # fmt: skip
    ranked_off = sorted(ratings, key=lambda r: r["drive_off"], reverse=True)
    for name, rows in (("top5_off", ranked_off[:5]), ("bottom5_off", ranked_off[-5:])):
        for r in rows:
            rank = ranked_off.index(r) + 1
            out.append(spec(f"{r['team'].lower()}_off_vs_top5_def",
                            f"{r['team']} offense vs a top-5 defense", name, _off(r),
                            AVERAGE_DEF, AVERAGE_OFF, best_def, offense=r["team"],
                            offense_rank=rank))  # fmt: skip
    return out


def spec_key(spec: dict[str, Any], inputs_key: str) -> str:
    """Cache key of one matchup: its ratings, the data and the code version."""
    blob = json.dumps({k: spec[k] for k in ("a_off", "a_def", "b_off", "b_def")},
                      sort_keys=True) + inputs_key + CODE_VERSION  # fmt: skip
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Solving
# --------------------------------------------------------------------------------------


class Kernels:
    """Matchup models, with each (offense, defense) calibration done once."""

    def __init__(self, inputs: Inputs) -> None:
        self.inputs = inputs
        self._cal: dict[str, mt.Calibration] = {}

    def calibration(self, off: dict, deff: dict) -> mt.Calibration:
        key = json.dumps([off, deff], sort_keys=True)
        if key not in self._cal:
            self._cal[key] = mt.calibrate(self.inputs.base.kern, off, deff, self.inputs.effects)
        return self._cal[key]

    def model(self, off: dict, deff: dict) -> tuple[mg.GameModel, mt.Calibration]:
        c = self.calibration(off, deff)
        kern = mt.tilt_kernels(self.inputs.base.kern, c.tilt, c.turnover_scale, c.conv_dlogit)
        return dataclasses.replace(self.inputs.base, kern=kern), c

    def pair(self, spec: dict[str, Any]) -> tuple[mg.GameModel, mg.GameModel, dict[str, Any]]:
        ma, ca = self.model(spec["a_off"], spec["b_def"])
        mb, cb = self.model(spec["b_off"], spec["a_def"])
        return ma, mb, {"a_offense_vs_b_defense": ca.as_dict(),
                        "b_offense_vs_a_defense": cb.as_dict()}  # fmt: skip


def map_cells(model: mg.GameModel, qcfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray]:  # fmt: skip
    g = model.kern.grid
    yls = np.arange(qcfg["yardlines"][0], qcfg["yardlines"][1] + 1)
    ds = np.arange(qcfg["distances"][0], qcfg["distances"][1] + 1)
    return ds, yls, g.fidx(4, g.dist_bin(ds)[:, None], yls[None, :])


def frontier(diff: np.ndarray, ds: np.ndarray) -> list[int]:
    """Longest distance at which going beats punting, per yard line (0 if never)."""
    out = []
    for j in range(diff.shape[1]):
        win = np.nonzero(diff[:, j] > 0)[0]
        out.append(int(ds[win].max()) if len(win) else 0)
    return out


def decision_record(q: dict[int, np.ndarray], model: mg.GameModel, qcfg: dict[str, Any]
                    ) -> dict[str, Any]:  # fmt: skip
    """Go, punt and field goal at the headline and the go-minus-punt map, from team A's
    action values at the headline tick (`q[code]`: (2, nf, S, E, E))."""
    head = qcfg["headline"]
    ds, yls, f = map_cells(model, qcfg)
    s = head["lead"] + model.score_max
    a, b = qcfg["offense_timeouts"], head["defense_timeouts"]
    go = np.max([q[c][0][f, s, a, b] for c in PLAY_CODES], axis=0)
    punt = q[mg.A_PUNT][0][f, s, a, b]
    fg = q[mg.A_FG][0][f, s, a, b]
    diff = 100.0 * (go - punt)
    i, j = list(ds).index(head["ydstogo"]), list(yls).index(head["yardline_100"])
    return {
        "headline": {"go": float(go[i, j]), "punt": float(punt[i, j]),
                     "field_goal": float(fg[i, j]), "go_minus_punt_points": float(diff[i, j])},
        "map_go_minus_punt_points": np.round(diff, 3).tolist(),
        "go_cells": int((diff > 0).sum()),
        "n_cells": int(diff.size),
        "go_frontier_by_yardline": dict(zip([int(y) for y in yls], frontier(diff, ds),
                                            strict=True)),
    }  # fmt: skip


def _write_atomic(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(obj, tmp)
    os.replace(tmp, path)


def solve_grid(
    inputs: Inputs,
    specs: list[dict[str, Any]],
    work: Path,
    deadline: float,
    backend: str = "dense",
    batch_size: int = 1,
    reserve_seconds: float = 120.0,
    first_guess_seconds: float = 30.0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Solve every spec not already in `work/matchups`, in order, `batch_size` at a time,
    while the wall clock allows: a batch starts only if the slowest batch so far (or a
    `first_guess_seconds` per matchup) still fits before `deadline` (a `time.time()` value) minus
    `reserve_seconds`. Each finished matchup is written to its own file at once."""
    from fourth_and_rl import mdp_team_jax as mj

    qcfg = mdp.load_config()["question1"]
    tick = qcfg["headline"]["seconds"] // inputs.base.clock.tick_seconds
    out_dir = work / "matchups"
    out_dir.mkdir(parents=True, exist_ok=True)
    ikey = str(inputs.provenance.get("dev_fingerprint", "local"))
    kern = Kernels(inputs)
    todo, done = [], []
    for sp in specs:
        path = out_dir / f"{sp['key']}.json"
        key = spec_key(sp, ikey)
        if path.exists() and read_json(path).get("cache_key") == key:
            done.append(sp["key"])
        else:
            todo.append(sp)
    log(f"{len(done)} matchups already done (resumed), {len(todo)} to solve")
    slowest = 0.0
    stopped = None
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        guess = slowest if slowest > 0 else first_guess_seconds * len(batch)
        left = deadline - time.time() - reserve_seconds
        if guess > left:
            stopped = (f"deadline: {left:.0f} s left before the reserve, a batch takes "
                       f"about {guess:.0f} s; {len(todo) - i} matchups not solved")  # fmt: skip
            log("stopping: " + stopped)
            break
        t0 = time.time()
        built = [kern.pair(sp) for sp in batch]
        t_kern = time.time() - t0
        res = mj.solve_end_phases([(a, b) for a, b, _ in built], backend=backend,
                                  batch_size=len(batch), q_ticks=(tick,), keep_arrays=False)
        for sp, (ma, _, cal), r in zip(batch, built, res, strict=True):
            rec = {
                "key": sp["key"], "label": sp["label"], "group": sp["group"],
                **{k: sp[k] for k in ("opponent", "defense_rank", "offense", "offense_rank")
                   if k in sp},
                "ratings": {k: sp[k] for k in ("a_off", "a_def", "b_off", "b_def")},
                "kernels": cal,
                **decision_record(r.q[tick], ma, qcfg),
                "solver": {**r.meta, "kernel_seconds_batch": t_kern},
                "cache_key": spec_key(sp, ikey),
            }  # fmt: skip
            _write_atomic(rec, out_dir / f"{sp['key']}.json")
            done.append(sp["key"])
            h = rec["headline"]
            log(f"  {sp['key']:28s} go - punt {h['go_minus_punt_points']:+6.2f} pts, "
                f"go cells {rec['go_cells']:3d}/{rec['n_cells']}")  # fmt: skip
        slowest = max(slowest, time.time() - t0)
    return {"done": done, "planned": [sp["key"] for sp in specs], "stopped": stopped,
            "slowest_batch_seconds": slowest}  # fmt: skip


def cpu_vs_gpu(inputs: Inputs, spec: dict[str, Any], backend: str) -> dict[str, Any]:
    """Wall time of one matchup's end phase on the CPU solver and with JAX (kernels built
    once, outside both timings), and the largest value difference between the two."""
    import jax

    from fourth_and_rl import mdp_team_jax as mj

    ma, mb, _ = Kernels(inputs).pair(spec)
    mj.warmup([(ma, mb)], backend=backend)  # compile time is not counted
    t0 = time.time()
    res = mj.solve_end_phases([(ma, mb)], backend=backend)[0]
    t_jax = time.time() - t0
    t0 = time.time()
    cpu = mt.solve_matchup(ma, mb)
    t_cpu = time.time() - t0
    diff = max(float(np.abs(res.v[k] - cpu.sols[k].v["end"]).max()) for k in range(2))
    return {
        "matchup": spec["key"],
        "cpu_seconds": t_cpu,
        "cpu_threads": os.cpu_count(),
        "jax_seconds": t_jax,
        "jax_device": str(jax.devices()[0]),
        "jax_backend": backend,
        "speedup": t_cpu / t_jax if t_jax > 0 else None,
        "max_abs_value_diff": diff,
        "note": "end phase only (61 slices), both teams; JAX time includes copying every "
        "value and policy slice back to the host, compile time excluded",
    }


# --------------------------------------------------------------------------------------
# Results, figure, zip
# --------------------------------------------------------------------------------------


def collect(work: Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for sp in specs:
        p = work / "matchups" / f"{sp['key']}.json"
        if p.exists():
            out.append(read_json(p))
    return out


def q6_reproduction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """This run's q6 rows against the committed results/questions/q6_team_specific.json."""
    path = RESULTS / "questions" / "q6_team_specific.json"
    if not path.exists():
        return {}
    ref = {m["key"]: m for m in read_json(path)["matchups"]}
    out = {}
    for r in rows:
        if r["group"] != "q6" or r["key"] not in ref:
            continue
        a = np.asarray(r["map_go_minus_punt_points"])
        b = np.asarray(ref[r["key"]]["map_go_minus_punt_points"])
        out[r["key"]] = {
            "headline_points_this_run": r["headline"]["go_minus_punt_points"],
            "headline_points_committed": ref[r["key"]]["headline"]["go_minus_punt_points"],
            "map_max_abs_diff_points": float(np.abs(a - b).max()),
            "go_cells_this_run": r["go_cells"],
            "go_cells_committed": ref[r["key"]]["go_cells"],
        }
    return out


def figure(res: dict[str, Any], path: Path) -> Path:
    """Left: go minus punt at the headline by opponent defense, ranked. Right: go regions
    for a few matchups (small multiples), in the style of the q6 figure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    style.apply()
    rows = res["matchups"]
    qm = res["map"]
    yls = np.arange(qm["yardlines"][0], qm["yardlines"][1] + 1)
    ds = np.arange(qm["distances"][0], qm["distances"][1] + 1)
    vs = sorted([r for r in rows if r["group"] == "vs_def"],
                key=lambda r: r["headline"]["go_minus_punt_points"])  # fmt: skip
    base = next((r for r in rows if r["key"] == "identical"), None)
    picks = [r for r in rows if r["key"] in ("identical", "det_vs_average")]
    if vs:
        picks += [vs[0], vs[-1]]
    offs = sorted([r for r in rows if r["group"] in ("top5_off", "bottom5_off")],
                  key=lambda r: r["headline"]["go_minus_punt_points"])  # fmt: skip
    if offs:
        picks += [offs[-1], offs[0]]
    picks += rows  # fill any free panels with the other solved matchups, in grid order
    seen, small = set(), []
    for r in picks:
        if r["key"] not in seen:
            seen.add(r["key"])
            small.append(r)
    small = small[:6]
    fig = plt.figure(figsize=(17, 7.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.28, hspace=0.62)
    ax = fig.add_subplot(gs[:, 0])
    if vs:
        y = np.arange(len(vs))
        val = [r["headline"]["go_minus_punt_points"] for r in vs]
        ax.barh(y, val, color=[style.BLUE if v > 0 else style.AQUA for v in val], height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['opponent']} (#{r['defense_rank']})" for r in vs], fontsize=8)
        if base is not None:
            b0 = base["headline"]["go_minus_punt_points"]
            ax.axvline(b0, color=style.INK, linewidth=1.0, linestyle="dashed")
            ax.text(b0, len(vs) - 0.2, " identical teams", fontsize=8, color=style.INK_2,
                    va="bottom")  # fmt: skip
        ax.axvline(0, color=style.AXIS, linewidth=1.0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Go minus punt (win-probability points)")
    else:
        ax.text(0.5, 0.5, "no defense matchups solved in this run", ha="center",
                transform=ax.transAxes, color=style.INK_2)  # fmt: skip
        ax.set_axis_off()
    ax.set_title(f"{TEAM} with the ball vs each defense\n(rank 1 = best 2025 defense)",
                 loc="left", fontsize=11, fontweight="bold", color=style.INK)  # fmt: skip
    cmap = LinearSegmentedColormap.from_list("go_punt", [style.AQUA, "#ecebe6", style.BLUE])
    xs = 100 - yls[::-1]
    extent = (xs[0] - 0.5, xs[-1] + 0.5, ds[0] - 0.5, ds[-1] + 0.5)
    b_map = None if base is None else np.asarray(base["map_go_minus_punt_points"])[:, ::-1]
    im = None
    h = res["headline"]
    for k, r in enumerate(small):
        a2 = fig.add_subplot(gs[k // 3, 1 + k % 3])
        z = np.asarray(r["map_go_minus_punt_points"])[:, ::-1]
        im = a2.imshow(z, origin="lower", aspect="auto", cmap=cmap, vmin=-12, vmax=12,
                       extent=extent, interpolation="nearest")  # fmt: skip
        if (z > 0).any() and (z <= 0).any():
            a2.contour(xs, ds, z, levels=[0.0], colors=[style.INK], linewidths=1.4)
        if b_map is not None and r["key"] != "identical" and (b_map > 0).any() and (
                b_map <= 0).any():  # fmt: skip
            a2.contour(xs, ds, b_map, levels=[0.0], colors=[style.INK], linewidths=1.0,
                       linestyles="dashed")  # fmt: skip
        a2.scatter([100 - h["yardline_100"]], [h["ydstogo"]], s=50, facecolor="none",
                   edgecolor=style.INK, linewidths=1.4, zorder=5)  # fmt: skip
        a2.grid(False)
        a2.set_xticks([20, 30, 40, 50])
        a2.set_yticks([1, 4, 7, 10])
        a2.set_xlabel("Own yard line", fontsize=9)
        if k % 3 == 0:
            a2.set_ylabel("Yards to go", fontsize=9)
        hd = r["headline"]["go_minus_punt_points"]
        a2.set_title(f"{r['label']}\n4th and 2, own 40: {hd:+.1f}; go in {r['go_cells']}"
                     f"/{r['n_cells']}", loc="left", fontsize=9, color=style.INK)  # fmt: skip
    if im is not None:
        cax = fig.add_axes([0.45, -0.02, 0.35, 0.022])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([-12, -8, -4, 0, 4, 8, 12])
        cb.set_ticklabels(["punt +12", "+8", "+4", "even", "+4", "+8", "go +12"])
        cb.set_label("Go minus punt (points). Solid: break-even; dashed: identical teams",
                     color=style.INK_2)  # fmt: skip
        cb.outline.set_visible(False)
    fig.text(0.125, 1.02, "Up 4 with 4:00 left, 3 timeouts each: go or punt, by matchup",
             fontsize=14, fontweight="bold", color=style.INK)  # fmt: skip
    fig.text(0.125, 0.975, "Exact two-team end-phase solutions (JAX). End-of-2025 M8 "
             "ratings (test-season numbers). No uncertainty interval.", fontsize=10,
             color=style.INK_2)  # fmt: skip
    return style.save(fig, path)


def make_zip(work: Path, name: str = "team_matchups_results.zip") -> Path:
    """results.json, the figure, every matchup file and the side files; no data."""
    zp = work / name
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(work.glob("*.json")) + sorted(work.glob("*.png")):
            z.write(p, p.name)
        for p in sorted((work / "matchups").glob("*.json")):
            z.write(p, f"matchups/{p.name}")
        if (work / "run_log.txt").exists():
            z.write(work / "run_log.txt", "run_log.txt")
    return zp


def link_data_dir(work: Path) -> Path:
    """Point the clone's data/ folder at `work/data`, so downloads and fits survive a
    disconnect when `work` is on Drive. An existing real data/ folder is kept."""
    target = work / "data"
    target.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data"
    if data_dir.is_symlink():
        if data_dir.resolve() == target.resolve():
            return target
        data_dir.unlink()
    elif data_dir.exists():
        if any(data_dir.iterdir()):
            return data_dir
        shutil.rmtree(data_dir)
    data_dir.symlink_to(target, target_is_directory=True)
    return target
