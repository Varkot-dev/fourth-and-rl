"""Plumbing shared by the M5 scripts: config, the frozen model snapshot, resumable runs."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from fourth_and_rl.paths import CONFIGS, ROOT


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read configs/rl.yaml."""
    with open(path or CONFIGS / "rl.yaml") as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


class Snapshot:
    """The frozen solved model and everything built from it, loaded once per process."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        from fourth_and_rl import mdp_game as mg
        from fourth_and_rl.rl.envs import build_drive_model, default_start_yardline

        self.dir = resolve(cfg["snapshot_dir"])
        self.sol = mg.load_solution(self.dir)
        self.model = self.sol.model
        tables = joblib.load(self.dir / "tables.joblib")
        self.coach = tables["coach"]
        self.sim = mg.Simulator(self.sol, coach=self.coach)
        m = self.model
        self.td_value = m.td_points + m.p_xp
        self.drive = build_drive_model(
            m.kern,
            self.td_value,
            m.fg_points,
            m.safety_points,
            start_yardline=default_start_yardline(m.kern, cfg["drive"]["kickoff_share"]),
        )
        with open(self.dir / "meta.json") as f:
            self.meta = json.load(f)

    def describe(self) -> dict[str, Any]:
        """What the results were computed from (goes into every results file)."""
        files = sorted(p.name for p in self.dir.iterdir())
        stats = {p: os.stat(self.dir / p) for p in files}
        return {
            "snapshot_dir": str(self.dir.relative_to(ROOT)) if self.dir.is_relative_to(ROOT)
            else str(self.dir),
            "meta_json": self.meta,
            "files": {p: {"bytes": s.st_size, "mtime": s.st_mtime} for p, s in stats.items()},
        }


# Snapshots of this process, keyed by what they are built from: the snapshot folder and
# the drive's kickoff share (the only parts of the config a Snapshot reads).
_SNAPS: dict[tuple[str, float], Snapshot] = {}


def _snapshot_key(cfg: dict[str, Any]) -> tuple[str, float]:
    return str(resolve(cfg["snapshot_dir"]).resolve()), float(cfg["drive"]["kickoff_share"])


def snapshot(cfg: dict[str, Any]) -> Snapshot:
    """The snapshot for `cfg` in this process (loaded on first use for that snapshot folder
    and kickoff share, then reused)."""
    key = _snapshot_key(cfg)
    if key not in _SNAPS:
        _SNAPS[key] = Snapshot(cfg)
    return _SNAPS[key]


def run_path(cfg: dict[str, Any], group: str, key: str) -> Path:
    return resolve(cfg["runs_dir"]) / group / f"{key}.json"


def stamp(cfg: dict[str, Any]) -> str:
    """Fingerprint of what a run depends on: the snapshot's meta.json (it changes with every
    solve) and the whole config except the number of workers. A saved run with another
    stamp is treated as missing, so a rerun on a new solution never reuses old runs."""
    import hashlib

    meta = (resolve(cfg["snapshot_dir"]) / "meta.json").read_bytes()
    conf = {k: v for k, v in cfg.items() if k != "workers"}
    return hashlib.sha1(meta + json.dumps(conf, sort_keys=True).encode()).hexdigest()[:16]


def load_run(cfg: dict[str, Any], group: str, key: str) -> dict[str, Any] | None:
    p = run_path(cfg, group, key)
    if p.exists():
        with open(p) as f:
            out = json.load(f)
        if out.get("_stamp") == stamp(cfg):
            return out
    return None


def save_run(cfg: dict[str, Any], group: str, key: str, out: dict[str, Any]) -> None:
    p = run_path(cfg, group, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({**out, "_stamp": stamp(cfg)}, f, default=_json_default)
    tmp.replace(p)


def _json_default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def _init_worker(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass


def run_all(
    jobs: list[tuple[str, str, Callable[..., dict], dict]],
    cfg: dict[str, Any],
    workers: int,
    force: bool = False,
    threads: int = 1,
) -> dict[tuple[str, str], dict]:
    """Run jobs (group, key, function, keyword arguments) in `workers` processes, skipping
    those with a saved result unless `force`. Each function is called as fn(cfg, **kw) and
    must return a JSON-friendly dict. Returns every result, old and new."""
    out: dict[tuple[str, str], dict] = {}
    todo = []
    for group, key, fn, kw in jobs:
        old = None if force else load_run(cfg, group, key)
        if old is not None:
            out[(group, key)] = old
        else:
            todo.append((group, key, fn, kw))
    print(f"{len(out)} runs reused, {len(todo)} to run", flush=True)
    if not todo:
        return out
    if workers <= 1:
        _init_worker(threads)
        for group, key, fn, kw in todo:
            res = fn(cfg, **kw)
            save_run(cfg, group, key, res)
            out[(group, key)] = res
            print(f"  done {group}/{key}  {res.get('seconds', 0):.0f} s", flush=True)
        return out
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(threads,)) as ex:
        futs = {ex.submit(fn, cfg, **kw): (group, key) for group, key, fn, kw in todo}
        for fut in as_completed(futs):
            group, key = futs[fut]
            res = fut.result()
            save_run(cfg, group, key, res)
            out[(group, key)] = res
            print(f"  done {group}/{key}  {res.get('seconds', 0):.0f} s", flush=True)
    return out
