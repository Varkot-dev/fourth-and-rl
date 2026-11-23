"""Research milestone R1, day 6 (run A): the seeded grid of overoptimization cells.

A cell is one proxy: LightGBM of a given capacity ("small" = 8 leaves, "large" = 64
leaves, configs/overopt.yaml) trained on n synthetic coach-vs-coach games drawn with one
seed, graded by one exact pass (scripts/research/overopt_curves.py, `exact_pass`). Its key
is n<games>_<capacity>_s<seed>, for example n250_large_s101; the oracle cell (the exact
coach values, no proxy) is "oracle".

This module holds the parts that need neither the solved game nor LightGBM, so they can be
tested quickly and shared by the local runner (scripts/research/overopt_grid.py) and the
Colab notebook (notebooks/overopt_grid_colab.ipynb):

- planning: the cells in priority order, and disjoint slices of that order ("parts");
- resume: a cell is done when its result file holds the same settings key;
- resources: how many cells to run at once without exceeding RAM;
- the scheduler: run cells as subprocesses until a deadline;
- collect: merge result folders, summarize the drop from peak gain per cell and per
  (games, capacity) across seeds, the figure and the zip.

Everything is exact for the fitted game MDP, not real football.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LABEL = "Exact for the fitted game MDP, not real football."
ORACLE = "oracle"
_KEY = re.compile(r"^n(\d+)_([a-z]+)_s(\d+)$")


# --------------------------------------------------------------------------------------
# Cells and the plan
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One grid cell. `block` is the priority block it belongs to (0 is run first); the
    oracle cell has n_games, capacity and seed None."""

    key: str
    n_games: int | None
    capacity: str | None
    seed: int | None
    block: int

    @property
    def is_oracle(self) -> bool:
        return self.key == ORACLE


def cell_key(n_games: int, capacity: str, seed: int) -> str:
    return f"n{int(n_games)}_{capacity}_s{int(seed)}"


def parse_key(key: str) -> tuple[int, str, int] | None:
    """(n_games, capacity, seed) of a cell key, or None for the oracle."""
    if key == ORACLE:
        return None
    m = _KEY.match(key)
    if m is None:
        raise ValueError(f"not a grid cell key: {key!r} (expected n<games>_<capacity>_s<seed>)")
    return int(m.group(1)), m.group(2), int(m.group(3))


def plan_cells(blocks: Sequence[Sequence[int]], capacities: Sequence[str],
               seeds: Sequence[int], oracle_after_block: int | None = None
               ) -> list[Cell]:  # fmt: skip
    """Every cell in priority order: block by block; within a block seed by seed, then
    games, then capacity. A finished prefix of the plan therefore holds whole replicates
    (every size and capacity of the block for the first seeds), which is what a short
    session should leave behind. The oracle cell follows block `oracle_after_block`."""
    out: list[Cell] = []
    for b, sizes in enumerate(blocks):
        for seed in seeds:
            for n in sizes:
                for cap in capacities:
                    out.append(Cell(cell_key(n, cap, seed), int(n), cap, int(seed), b))
        if oracle_after_block is not None and b == oracle_after_block:
            out.append(Cell(ORACLE, None, None, None, b))
    keys = [c.key for c in out]
    if len(set(keys)) != len(keys):
        raise ValueError("the plan repeats a cell: a size, capacity or seed is listed twice")
    return out


def plan_from_config(grid: dict[str, Any], sizes: Iterable[int] | None = None,
                     capacities: Iterable[str] | None = None,
                     seeds: Iterable[int] | None = None,
                     include_oracle: bool = True) -> list[Cell]:  # fmt: skip
    """The plan from configs/overopt.yaml (section `grid`), optionally narrowed. `sizes`
    keeps the config's blocks for the sizes it lists; a size in no block becomes a last
    block of its own, so any list of sizes is accepted."""
    blocks = [list(b) for b in grid["blocks"]]
    if sizes is not None:
        want = [int(n) for n in sizes]
        blocks = [[n for n in b if n in want] for b in blocks]
        extra = [n for n in want if not any(n in b for b in blocks)]
        if extra:
            blocks.append(extra)
    keep = [i for i, b in enumerate(blocks) if b]
    after = grid.get("oracle_after_block") if include_oracle else None
    if after is not None:  # follow the same block after empty blocks are dropped
        after = max([j for j, i in enumerate(keep) if i <= after], default=0)
    blocks = [blocks[i] for i in keep]
    caps = list(capacities) if capacities is not None else list(grid["capacities"])
    sds = [int(s) for s in seeds] if seeds is not None else [int(s) for s in grid["seeds"]]
    return plan_cells(blocks, caps, sds, after if blocks else None)


def parse_part(part: str) -> tuple[int, int]:
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(part))
    if m is None:
        raise ValueError(f"a part is K/M, for example 1/3, not {part!r}")
    k, n = int(m.group(1)), int(m.group(2))
    if not 1 <= k <= n:
        raise ValueError(f"part {part!r}: K must be between 1 and M")
    return k, n


def select_parts(cells: Sequence[Cell], parts: Iterable[str] | None) -> list[Cell]:
    """The cells of the given parts of the plan. Part K/M takes every M-th cell of the
    full plan starting at the K-th, so each part keeps the priority order, the M parts
    are disjoint and together they are the whole plan. All parts must share M."""
    if not parts:
        return list(cells)
    parsed = [parse_part(p) for p in parts]
    ms = {m for _, m in parsed}
    if len(ms) != 1:
        raise ValueError(f"parts must share the same M: {list(parts)}")
    m = ms.pop()
    ks = {k for k, _ in parsed}
    return [c for i, c in enumerate(cells) if (i % m) + 1 in ks]


# --------------------------------------------------------------------------------------
# Identity of a result: the game and the settings
# --------------------------------------------------------------------------------------


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()[:16]


def game_key(meta: dict[str, Any]) -> str:
    """Identity of a solved game from its meta.json: the fit stamp, the tables key (with
    the data fingerprint) and the solve seed, without fit timestamps. A rebuild from the
    same data gives the same key; a rebuild from revised nflverse data does not."""
    keep = {k: meta.get(k) for k in ("stamp", "tables_key", "seed")}

    def strip(y: Any) -> Any:
        if isinstance(y, dict):
            return {k: strip(w) for k, w in y.items() if k != "fitted_at"}
        if isinstance(y, list):
            return [strip(w) for w in y]
        return y

    return _hash(strip(json.loads(json.dumps(keep, default=float))))


def game_key_of_dir(folder: Path | str) -> str:
    return game_key(json.loads((Path(folder) / "meta.json").read_text()))


def cell_settings(cfg: dict[str, Any], cell: Cell, betas: Sequence[float] | None,
                  game: str) -> dict[str, Any]:  # fmt: skip
    """Everything a cell's numbers depend on. Threads are not here: the pass gives the
    same numbers with any thread count, and the proxy always trains with
    `proxy_threads`."""
    val = cfg["games"][cfg["grid"]["val_games"]]
    out: dict[str, Any] = {
        "betas": [float(b) for b in (betas if betas is not None else cfg["betas"])],
        "post": cfg["post"], "ties": cfg["ties"], "seed": cfg["seed"],
        "val_games": {"n": int(val["n"]), "seed": int(val["seed"])}, "game_key": game,
    }  # fmt: skip
    if not cell.is_oracle:
        out.update(n_games=cell.n_games, games_seed=cell.seed, capacity=cell.capacity,
                   proxy_params=cfg["proxy_sizes"][cell.capacity],
                   proxy_random_state=cfg["proxy_random_state"],
                   proxy_threads=cfg["proxy_threads"])  # fmt: skip
    return out


def settings_key(settings: dict[str, Any]) -> str:
    return _hash(settings)


def read_cell(path: Path) -> dict[str, Any] | None:
    """A cell result file, or None if it is missing, unreadable or not a grid cell."""
    try:
        rec = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) and isinstance(rec.get("grid"), dict) else None


def is_done(path: Path, skey: str) -> bool:
    rec = read_cell(path)
    return rec is not None and rec["grid"].get("settings_key") == skey and bool(rec.get("rows"))


def split_done(cells: Sequence[Cell], dirs: Sequence[Path | str],
               skey: Callable[[Cell], str]) -> tuple[list[Cell], dict[str, Path]]:  # fmt: skip
    """(cells still to run, {key: result file} of cells already done in any of `dirs`).
    A result with other settings (other betas, another game) does not count as done."""
    todo, done = [], {}
    for c in cells:
        k = skey(c)
        hit = next((Path(d) / f"{c.key}.json" for d in dirs
                    if is_done(Path(d) / f"{c.key}.json", k)), None)  # fmt: skip
        if hit is None:
            todo.append(c)
        else:
            done[c.key] = hit
    return todo, done


def write_atomic(obj: Any, path: Path) -> Path:
    """Write JSON through a temporary file and one rename, so a disconnect or a parallel
    reader never sees half a file (Drive included)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp.write_text(json.dumps(obj, indent=2, default=float) + "\n")
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------------------
# Resources: how many cells at once
# --------------------------------------------------------------------------------------


def system_resources() -> dict[str, float]:
    """Usable cores (the process's CPU affinity) and total and available RAM in GB."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cores = os.cpu_count() or 1
    total = avail = None
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            info[name] = float(value.split()[0]) / 2**20  # kB to GB
        total, avail = info.get("MemTotal"), info.get("MemAvailable")
    except OSError:
        pass
    if total is None:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return {"cores": int(cores), "ram_gb": round(total, 2),
            "available_gb": round(avail if avail is not None else total, 2)}  # fmt: skip


def choose_parallel(cores: int, available_gb: float, mem_gb_per_cell: float,
                    threads_per_cell: int, reserve_gb: float = 2.0,
                    max_parallel: int | None = None,
                    ram_gb: float | None = None) -> dict[str, Any]:  # fmt: skip
    """How many cells to run at once, and with how many threads each.

    by_ram = floor((available RAM - reserve) / memory per cell): never more cells than fit
    in memory at their measured peak. by_cpu = cores // threads per cell (at least 1).
    n = max(1, min(by_ram, by_cpu, max_parallel)); one cell always runs, with a warning
    when even one does not fit. Threads per cell = min(threads_per_cell, cores // n), at
    least 1, so a 2-core machine runs one cell with 2 threads."""
    by_ram = int(math.floor((available_gb - reserve_gb) / mem_gb_per_cell))
    by_cpu = max(1, int(cores) // int(threads_per_cell))
    n = max(1, min(by_ram, by_cpu, max_parallel if max_parallel else by_cpu))
    threads = max(1, min(int(threads_per_cell), int(cores) // n))
    warnings = []
    total = ram_gb if ram_gb is not None else available_gb
    if total < 12:
        warnings.append(f"this machine has {total:.1f} GB of RAM, under 12 GB: one cell peaks at "
                        f"about {mem_gb_per_cell:.0f} GB, so the run may be killed. On Colab "
                        "choose Runtime > Change runtime type > CPU with High-RAM.")  # fmt: skip
    if by_ram < 1:
        warnings.append(f"only {available_gb:.1f} GB available; one cell needs about "
                        f"{mem_gb_per_cell:.0f} GB plus {reserve_gb:.0f} GB reserve")  # fmt: skip
    limit = "RAM" if by_ram <= min(by_cpu, n) else ("cores" if by_cpu == n else "max_parallel")
    return {"n_parallel": n, "threads_per_cell": threads, "by_ram": by_ram, "by_cpu": by_cpu,
            "limited_by": limit, "warnings": warnings,
            "cores": int(cores), "available_gb": available_gb,
            "mem_gb_per_cell": mem_gb_per_cell, "reserve_gb": reserve_gb}  # fmt: skip


def block_hours(plan: Sequence[Cell], todo: Sequence[Cell], n_parallel: int,
                seconds_per_cell: float) -> list[dict[str, Any]]:  # fmt: skip
    """Cells and estimated wall-clock hours left per priority block, at `n_parallel` cells
    at a time and `seconds_per_cell` each (blocks run back to back, so a block's hours are
    its share of the total, not a rounded-up number of rounds)."""
    left = {c.key for c in todo}
    out = []
    for b in sorted({c.block for c in plan}):
        cells = [c for c in plan if c.block == b]
        sizes = sorted({c.n_games for c in cells if not c.is_oracle})
        label = ", ".join(f"{n:,}" for n in sizes) + " games"
        if any(c.is_oracle for c in cells):
            label += " + oracle"
        n_left = sum(c.key in left for c in cells)
        out.append({"block": b, "label": label, "cells": len(cells), "todo": n_left,
                    "hours": n_left * seconds_per_cell / max(1, n_parallel) / 3600})  # fmt: skip
    return out


# --------------------------------------------------------------------------------------
# The scheduler
# --------------------------------------------------------------------------------------


def run_grid(cells: Sequence[Cell], launch: Callable[[Cell, int], Any], *, n_parallel: int,
             threads: int, deadline: float, seconds_per_cell: float,
             is_finished: Callable[[Cell], bool], log: Callable[[str], None] = print,
             poll: float = 5.0, clock: Callable[[], float] = time.time,
             sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:  # fmt: skip
    """Run `cells` in order, `n_parallel` at a time, as subprocesses from `launch(cell,
    threads)` (anything with .poll() and .terminate()). A cell starts only if it should
    finish before `deadline` (a time.time() value): the estimate is `seconds_per_cell`
    until cells finish here, then 1.2 times the longest of them. A cell counts as done
    only if `is_finished(cell)` (its result file is complete). A cell killed by the system
    (negative return code, usually out of memory) is retried once with one fewer cell in
    parallel. Interrupting (the stop button) terminates the running cells."""
    queue = list(cells)
    running: dict[str, tuple[Cell, Any, float]] = {}
    done, failed, durations = [], [], {}
    retried: set[str] = set()
    stopped = None

    def estimate() -> float:
        return 1.2 * max(durations.values()) if durations else float(seconds_per_cell)

    try:
        while queue or running:
            while queue and len(running) < n_parallel:
                if clock() + estimate() > deadline:
                    stopped = (f"deadline: a cell takes about {estimate() / 60:.0f} min and "
                               f"{max(0.0, deadline - clock()) / 60:.0f} min are left")
                    break
                c = queue.pop(0)
                running[c.key] = (c, launch(c, threads), clock())
                log(f"start {c.key} ({len(running)} running, {len(queue)} queued)")
            if stopped and not running:
                break
            if not running:
                continue
            sleep(poll)
            for key in list(running):
                c, proc, t0 = running[key]
                rc = proc.poll()
                if rc is None:
                    continue
                del running[key]
                sec = clock() - t0
                if rc == 0 and is_finished(c):
                    done.append(key)
                    durations[key] = sec
                    log(f"done  {key} in {sec / 60:.1f} min")
                elif rc is not None and rc < 0 and key not in retried:
                    retried.add(key)
                    n_parallel = max(1, n_parallel - 1)
                    queue.insert(0, c)
                    log(f"{key} was killed (signal {-rc}, usually out of memory): retrying it "
                        f"with {n_parallel} cell(s) at a time")  # fmt: skip
                else:
                    failed.append(key)
                    log(f"FAILED {key} (exit code {rc}); see its log")
    except KeyboardInterrupt:
        for _c, proc, _t in running.values():
            proc.terminate()
        log(f"interrupted: stopped {', '.join(running) or 'nothing'}")
        raise
    return {"done": done, "failed": failed, "not_started": [c.key for c in queue],
            "stopped": stopped, "seconds": durations, "n_parallel_end": n_parallel}


# --------------------------------------------------------------------------------------
# Collect: merge, summarize, plot, zip
# --------------------------------------------------------------------------------------


def cell_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Peak beta, peak gain, final gain (at the largest beta) and the drop from peak to
    final, in points of win probability (100 times the probability)."""
    rows = sorted(rows, key=lambda r: r["beta"])
    gains = np.array([100 * r["true_gain"] for r in rows])
    i = int(gains.argmax())
    return {"peak_beta": rows[i]["beta"], "peak_gain_points": float(gains[i]),
            "final_beta": rows[-1]["beta"], "final_gain_points": float(gains[-1]),
            "drop_points": float(gains[i] - gains[-1])}  # fmt: skip


def _same_numbers(a: dict, b: dict, tol: float = 1e-9) -> tuple[bool, float]:
    ra = {r["beta"]: r["true_gain"] for r in a["rows"]}
    rb = {r["beta"]: r["true_gain"] for r in b["rows"]}
    if set(ra) != set(rb):
        return False, float("inf")
    diff = max(abs(ra[k] - rb[k]) for k in ra)
    return diff <= tol, diff


def merge_cells(dirs: Sequence[Path | str], game: str | None = None
                ) -> tuple[dict[str, dict], dict[str, Any]]:  # fmt: skip
    """Merge the cell result files of several folders (for example this machine's and an
    unzipped Colab run). Returns ({key: record}, report).

    A key found twice is kept once: identical settings and numbers (to 1e-9) are a
    duplicate; anything else is a conflict, reported, and the first folder wins. Cells of
    different games (game keys) are never pooled: only `game` is kept, by default the
    game key with the most cells, and the others are listed as excluded."""
    found: dict[str, dict] = {}
    report: dict[str, Any] = {"folders": [str(d) for d in dirs], "duplicates": [],
                              "conflicts": [], "excluded_other_game": [], "per_folder": {}}
    for d in dirs:
        n = 0
        for p in sorted(Path(d).glob("*.json")):
            rec = read_cell(p)
            if rec is None or not rec.get("rows"):
                continue
            n += 1
            rec["grid"]["source"] = str(p)
            key = rec["grid"]["key"]
            if key not in found:
                found[key] = rec
                continue
            old = found[key]
            same, diff = _same_numbers(old, rec)
            entry = {"key": key, "kept": old["grid"]["source"], "other": str(p),
                     "max_abs_true_gain_diff": diff}  # fmt: skip
            if same and old["grid"]["settings_key"] == rec["grid"]["settings_key"]:
                report["duplicates"].append(entry)
            else:
                report["conflicts"].append(entry)
        report["per_folder"][str(d)] = n
    keys = Counter(r["grid"]["settings"]["game_key"] for r in found.values())
    report["game_keys"] = dict(keys)
    chosen = game if game is not None else (keys.most_common(1)[0][0] if keys else None)
    report["game_key"] = chosen
    cells = {}
    for k, r in found.items():
        if r["grid"]["settings"]["game_key"] == chosen:
            cells[k] = r
        else:
            report["excluded_other_game"].append(k)
    return cells, report


def _sd(x: np.ndarray) -> float | None:
    return float(np.std(x, ddof=1)) if len(x) > 1 else None


def group_summary(cells: dict[str, dict]) -> list[dict[str, Any]]:
    """Per (games, capacity), across seeds: the drop from peak gain to final gain (mean,
    SD with ddof 1, and whether the mean exceeds 2 SD, the falsifiable claim's test), the
    peak betas, and the mean and SD of the true gain and of sqrt(KL) at every beta. Only
    the betas every seed of the group has are used."""
    groups: dict[tuple[int, str], list[str]] = {}
    for k in cells:
        p = parse_key(k)
        if p is not None:
            groups.setdefault((p[0], p[1]), []).append(k)
    out = []
    for (n, cap), keys in sorted(groups.items()):
        keys = sorted(keys, key=lambda k: parse_key(k)[2])
        betas = sorted(set.intersection(*({r["beta"] for r in cells[k]["rows"]} for k in keys)))
        rows = [{r["beta"]: r for r in cells[k]["rows"]} for k in keys]
        gain = np.array([[100 * r[b]["true_gain"] for b in betas] for r in rows])
        skl = np.array([[r[b]["sqrt_kl"] for b in betas] for r in rows])
        claim = np.array([[100 * r[b]["claimed_gain"] for b in betas] for r in rows])
        peak = gain.max(1)
        drop = peak - gain[:, -1]
        sd = _sd(drop)
        diag = [cells[k].get("diagnostics") or {} for k in keys]
        rmse = [d["value_rmse"] for d in diag if "value_rmse" in d]
        dis = [d["greedy_uniform"]["disagree"] for d in diag
               if "disagree" in (d.get("greedy_uniform") or {})]
        out.append({
            "n_games": n, "capacity": cap, "seeds": [parse_key(k)[2] for k in keys],
            "n_seeds": len(keys), "betas": betas,
            "peak_beta": [betas[i] for i in gain.argmax(1)],
            "peak_gain_points": peak.tolist(), "final_gain_points": gain[:, -1].tolist(),
            "drop_points": drop.tolist(), "drop_mean_points": float(drop.mean()),
            "drop_sd_points": sd,
            "drop_exceeds_2sd": (bool(drop.mean() > 2 * sd) if sd is not None else None),
            "gain_mean_points": gain.mean(0).tolist(),
            "gain_sd_points": [_sd(gain[:, j]) for j in range(len(betas))],
            "sqrt_kl_mean": skl.mean(0).tolist(),
            "sqrt_kl_sd": [_sd(skl[:, j]) for j in range(len(betas))],
            "claimed_gain_mean_points": claim.mean(0).tolist(),
            "value_rmse_mean": float(np.mean(rmse)) if rmse else None,
            "greedy_disagreement_mean": float(np.mean(dis)) if dis else None,
        })  # fmt: skip
    return out


def day5_reproduction(cells: dict[str, dict], day5_dir: Path, reference_game: str | None
                      ) -> list[dict[str, Any]]:  # fmt: skip
    """n1000_<capacity>_s101, s102 and s103 train on the same games as day 5's
    1k_<capacity>, 1k_<capacity>_b and _c. On the same game their true gains must agree
    (a check of the grid runner against the day 5 pass); other games are skipped."""
    out = []
    for cap in ("small", "large"):
        for seed, suffix in ((101, ""), (102, "_b"), (103, "_c")):
            key, old = cell_key(1000, cap, seed), Path(day5_dir) / f"1k_{cap}{suffix}.json"
            if key not in cells or not old.exists():
                continue
            if cells[key]["grid"]["settings"]["game_key"] != reference_game:
                continue
            ref = json.loads(old.read_text())
            new = {r["beta"]: r["true_gain"] for r in cells[key]["rows"]}
            common = [r for r in ref["rows"] if r["beta"] in new]
            if common:
                diff = max(abs(new[r["beta"]] - r["true_gain"]) for r in common)
                out.append({"grid_cell": key, "day5_cell": old.stem, "betas": len(common),
                            "max_abs_true_gain_diff": diff})  # fmt: skip
    return out


def collect(dirs: Sequence[Path | str], out_json: Path, out_fig: Path | None = None,
            game: str | None = None, plan: Sequence[Cell] | None = None,
            reference_game: str | None = None, day5_dir: Path | None = None,
            extra: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """Merge the folders, summarize, write `out_json` and (if any cell) the figure."""
    cells, report = merge_cells(dirs, game)
    per_cell = {}
    for k, r in sorted(cells.items()):
        p = parse_key(k)
        d = r.get("diagnostics") or {}
        gu = d.get("greedy_uniform") if isinstance(d.get("greedy_uniform"), dict) else {}
        per_cell[k] = {
            "n_games": p[0] if p else None, "capacity": p[1] if p else None,
            "leaves": (r["grid"]["settings"].get("proxy_params") or {}).get("num_leaves"),
            "seed": p[2] if p else None, **cell_summary(r["rows"]),
            "value_rmse": d.get("value_rmse"), "greedy_disagreement": gu.get("disagree"),
            "rows": [{"beta": x["beta"], "true_gain": x["true_gain"], "sqrt_kl": x["sqrt_kl"],
                      "claimed_gain": x["claimed_gain"]} for x in r["rows"]],
            "best_response_gain": r.get("best_response_gain"),
            "seconds": (r.get("timing") or {}).get("total_seconds"),
            "peak_rss_gb": (r.get("timing") or {}).get("peak_rss_gb"),
            "host": r["grid"].get("host"), "source": r["grid"]["source"],
        }  # fmt: skip
    out: dict[str, Any] = {
        "what": "R1 day 6, run A: the seeded grid. Exact true gain over coach-vs-coach play of "
                "KL-tilted fourth-down policies built on LightGBM win-probability proxies, by "
                "training games, capacity and data seed, with the exact sequence KL",
        "label": LABEL, "game_key": report["game_key"], "merge": report,
        "cells": per_cell, "groups": group_summary(cells),
    }  # fmt: skip
    if plan is not None:
        missing = [c.key for c in plan if c.key not in cells]
        out["plan"] = {"planned": len(plan), "done": len(plan) - len(missing),
                       "missing": missing}  # fmt: skip
    if reference_game is not None:
        out["same_game_as_reference"] = report["game_key"] == reference_game
        if day5_dir is not None:
            out["day5_reproduction"] = day5_reproduction(cells, day5_dir, reference_game)
    if extra:
        out.update(extra)
    write_atomic(out, out_json)
    if out_fig is not None and per_cell:
        plot(out, out_fig)
    return out


def plot(out: dict[str, Any], path: Path) -> Path:
    """True gain against exact sqrt(KL): one panel per capacity, one line per training-set
    size (the mean over seeds at each beta, with a band of plus or minus one seed SD of
    the gain), and the oracle (exact coach values) as the reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fourth_and_rl import style

    style.apply()
    groups = out["groups"]
    caps = sorted({g["capacity"] for g in groups}, key=lambda c: (c != "small", c))
    caps = caps or ["small"]
    sizes = sorted({g["n_games"] for g in groups})
    ramp = style.SEQUENTIAL[1:]
    colors = {n: ramp[round(i * (len(ramp) - 1) / max(1, len(sizes) - 1))]
              for i, n in enumerate(sizes)}  # fmt: skip
    oracle = out["cells"].get(ORACLE)
    fig, axs = plt.subplots(1, len(caps), figsize=(6.4 * len(caps), 5.0), sharey=True,
                            squeeze=False)  # fmt: skip
    for ax, cap in zip(axs[0], caps, strict=True):
        ax.axhline(0, color=style.MUTED, lw=1)
        if oracle is not None:
            x = [r["sqrt_kl"] for r in oracle["rows"]]
            ax.plot(x, [100 * r["true_gain"] for r in oracle["rows"]], "-", color=style.INK,
                    lw=1.6, label="exact coach value (no proxy error)")  # fmt: skip
            if oracle.get("best_response_gain") is not None:
                ax.axhline(100 * oracle["best_response_gain"], color=style.MUTED, lw=1, ls=":")
        leaves = next((c["leaves"] for c in out["cells"].values()
                       if c["capacity"] == cap and c["leaves"]), None)  # fmt: skip
        for g in [g for g in groups if g["capacity"] == cap]:
            x = np.array(g["sqrt_kl_mean"])
            y = np.array(g["gain_mean_points"])
            col = colors[g["n_games"]]
            if g["n_seeds"] > 1:
                sd = np.array(g["gain_sd_points"], dtype=float)
                ax.fill_between(x, y - sd, y + sd, color=col, alpha=0.22, lw=0)
            flag = ""
            if g["drop_exceeds_2sd"]:
                flag = ", drop > 2 SD"
            ax.plot(x, y, "-o", ms=3.5, color=col,
                    label=f"{g['n_games']:,} games ({g['n_seeds']} seeds{flag})")  # fmt: skip
        name = f"{leaves} leaves" if leaves else cap
        style.title(ax, f"True gain, {name} proxies",
                    "mean over data seeds; band: plus or minus one seed SD")  # fmt: skip
        ax.set_xlabel("sqrt of the exact sequence KL(pi_beta || coach) per game")
        ax.legend(fontsize=8, loc="lower right")
    axs[0][0].set_ylabel("points of win probability over the coach")
    fig.text(0.0, -0.04, f"{LABEL} Team A tilts its fourth-down choice toward the proxy's "
             "values; team B and every other decision follow the coach. Dotted: exact best "
             "fourth-down response.", fontsize=8.5, color=style.INK_2)  # fmt: skip
    style.save(fig, path)
    return Path(path)


def make_zip(work: Path, name: str, members: Sequence[Path]) -> Path:
    """Zip `members` (files or folders under `work`) into work/name, atomically."""
    work = Path(work)
    path = work / name
    tmp = work / f".{name}.tmp-{os.getpid()}"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for m in members:
            m = Path(m)
            files = [m] if m.is_file() else sorted(f for f in m.rglob("*") if f.is_file())
            for f in files:
                if not f.name.startswith("."):
                    z.write(f, f.relative_to(work))
    tmp.replace(path)
    return path
