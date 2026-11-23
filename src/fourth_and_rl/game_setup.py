"""The solved game (data/mdp) on a fresh machine: reuse it, restore it from a cache folder,
or rebuild it from public data with the repository's own pipeline, then cache it when the
cache folder has room.

Used by notebooks/overopt_grid_colab.ipynb. The rebuild follows the same steps as the
world-model ground-truth notebook (notebooks/world_model/ground_truth_run.py, build_game),
which ran them on Colab:

  1 data.build: download nflverse play-by-play for the configured seasons (2010 to 2025)
    with nflreadpy and build the state table. The 2026 holdout is refused before anything
    is downloaded.
  2 splits.write_split_files: the development and test files.
  3 components.fit_dev_bundle: the development component bundle (fitted on development
    seasons; the kickoff tables of the 2024 and 2025 rules are the repository's one logged
    exception, read through splits.load_test_plays with a written purpose).
  4 scripts/solve_mdp.py: count, mode and coach tables on development seasons, the game
    model and backward induction, saved with the same meta as the script.

A rebuild from a fresh download can differ from an older local copy, because nflverse
revises its data; overopt_grid.game_key tells the two apart.
"""

from __future__ import annotations

import importlib.util
import os
import resource
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fourth_and_rl.paths import ROOT


def has_game(folder: Path | str) -> bool:
    """A solved game with its coach tables (what overopt.load_game reads)."""
    from fourth_and_rl import mdp_game as mg

    return mg.solution_exists(folder) and (Path(folder) / "tables.joblib").exists()


def dir_bytes(folder: Path | str) -> int:
    return sum(f.stat().st_size for f in Path(folder).rglob("*") if f.is_file())


def copy_game(src: Path | str, dst: Path | str) -> float:
    """Copy a solved game folder atomically (into a temporary folder next to dst, then one
    rename). Returns seconds."""
    t0 = time.time()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.tmp-{os.getpid()}-{time.time_ns()}"
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return round(time.time() - t0, 1)


def cache_decision(need_bytes: int, free_bytes: int | None, margin: float = 1.1,
                   slack_bytes: int = 2**28) -> tuple[bool, str]:  # fmt: skip
    """Whether a cache copy of `need_bytes` fits in `free_bytes` with a margin. Unknown
    free space (None) is tried; the copy itself then fails cleanly if it does not fit."""
    if free_bytes is not None and free_bytes < need_bytes * margin + slack_bytes:
        return False, (f"not caching the solved game: it needs {need_bytes / 2**30:.1f} GB and "
                       f"the cache folder has {free_bytes / 2**30:.1f} GB free; a later session "
                       "rebuilds it")  # fmt: skip
    return True, f"caching the solved game ({need_bytes / 2**30:.1f} GB)"


def try_cache(src: Path, cache: Path, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Copy the solved game to `cache` when there is room. The cache only saves rebuild time
    in a later session, so a full Drive, or a copy that fails, skips it instead of failing."""
    need = dir_bytes(src)
    parent = Path(cache).parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        free = shutil.disk_usage(parent).free
    except OSError:
        free = None
    ok, msg = cache_decision(need, free)
    log(msg)
    if not ok:
        return {"cache_skipped": msg}
    try:
        return {"cache_seconds": copy_game(src, cache)}
    except OSError as e:
        for p in parent.glob(f".{Path(cache).name}.tmp-*"):
            shutil.rmtree(p, ignore_errors=True)
        msg = f"caching the solved game failed ({type(e).__name__}: {e}); continuing without it"
        log(msg)
        return {"cache_skipped": msg}


def _peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def _solve_mdp_module() -> Any:
    path = ROOT / "scripts" / "solve_mdp.py"
    spec = importlib.util.spec_from_file_location("overopt_solve_mdp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_seasons() -> list[int]:
    """The seasons data.build would download; refuses if any is a holdout season."""
    from fourth_and_rl import data, splits

    seasons = data.seasons_from_config(data.load_config())
    holdout = set(splits.holdout_seasons())
    if holdout & set(seasons):
        raise RuntimeError(f"refusing to download holdout seasons {sorted(holdout & set(seasons))}")
    return seasons


def build_game(log: Callable[[str], None] = print) -> dict[str, Any]:
    """Rebuild data/mdp of this checkout from public data (steps 1 to 4 of the module
    docstring). Refuses when this checkout's data folder links outside it, so a test can
    never overwrite another checkout's data."""
    from fourth_and_rl import components as cm
    from fourth_and_rl import data, mdp, paths, splits
    from fourth_and_rl import mdp_game as mg

    if paths.DATA.is_symlink() or not paths.DATA.resolve().is_relative_to(paths.ROOT.resolve()):
        raise RuntimeError(f"refusing to build into {paths.DATA}: it links outside this checkout")
    seasons = check_seasons()
    sec: dict[str, float] = {}
    t = time.time()

    def lap(name: str) -> None:
        nonlocal t
        sec[name] = round(time.time() - t, 1)
        log(f"[game] {name}: {sec[name]:.0f} s (peak memory so far {_peak_rss_gb()} GB)")
        t = time.time()

    log(f"[game] downloading play-by-play {seasons[0]} to {seasons[-1]} (never 2026)")
    data.build()
    lap("download play-by-play and build the state table")
    plays = data.load_plays()
    n_plays = plays.height
    splits.write_split_files(plays)
    del plays
    lap("development and test files")
    ccfg = cm.load_config()
    dev = cm.add_next_state(splits.load_dev_plays())
    test = cm.add_next_state(splits.load_test_plays(
        "overopt grid notebook setup: kickoff start tables of the 2024 and 2025 rules for the "
        "development component bundle, exactly as scripts/fit_components.py"))  # fmt: skip
    models = cm.fit_dev_bundle(ccfg, dev_plays=dev, test_plays=test, seed=ccfg["seed"])
    models.save()
    del dev, test
    lap("development component bundle")
    sm = _solve_mdp_module()
    cfg = mdp.load_config()
    bundle = sm.load_tables(cfg, splits.load_dev_plays(), models)
    stamp = sm.combined_stamp(bundle, models)
    lap("count, mode and coach tables")
    t_kernel = time.time()
    model = mg.build_model(cfg, models, bundle["tables"], horizon="game")
    sec_kernel = time.time() - t_kernel
    lap("game model (kernels)")
    sol = mg.solve(model, progress=True)
    sol.meta["kernel_build_seconds"] = sec_kernel
    sol.meta["seed"] = cfg["seed"]
    sol.meta["stamp"] = stamp
    sol.meta["tables_key"] = bundle["key"]
    lap("backward induction")
    out = paths.ROOT / cfg["solve"]["output_dir"]
    mg.save_solution(sol, out)
    lap("save the solved game")
    return {"seconds": sec, "total_seconds": round(sum(sec.values()), 1),
            "peak_rss_gb_process": _peak_rss_gb(), "plays": int(n_plays),
            "seasons": [seasons[0], seasons[-1]], "solution_dir": str(out),
            "saved_gb": round(dir_bytes(out) / 2**30, 2)}  # fmt: skip


def ensure_game(game_dir: Path | str | None, cache: Path | str | None, use_cache: bool = True,
                log: Callable[[str], None] = print) -> dict[str, Any]:  # fmt: skip
    """Make the solved game available and say where it came from.

    game_dir given: used as is (read only). Otherwise data/mdp of this checkout: reused if
    present, else restored from `cache` (if `use_cache`), else rebuilt from public data.
    After a rebuild or restore the game is cached when `use_cache` and the cache folder
    has room (try_cache)."""
    from fourth_and_rl import mdp_game as mg

    t0 = time.time()
    if game_dir is not None:
        if not has_game(game_dir):
            raise FileNotFoundError(f"GAME_DIR {game_dir} holds no solved game with tables.joblib")
        return {"game_dir": str(game_dir), "source": f"given folder {game_dir} (read only)"}
    d = mg.SOLUTION_DIR
    out: dict[str, Any] = {"game_dir": str(d)}
    if has_game(d):
        out["source"] = f"already in this checkout ({d})"
    elif use_cache and cache is not None and has_game(cache):
        log(f"[game] restoring the solved game from the cache {cache}")
        out["restore_seconds"] = copy_game(cache, d)
        out["source"] = f"restored from the cache {cache}"
    else:
        log("[game] building the solved game from public data (download, fit, solve)")
        out["build"] = build_game(log)
        out["source"] = "rebuilt from public data in this session"
    if use_cache and cache is not None and not has_game(cache):
        out.update(try_cache(d, Path(cache), log))
    out["seconds"] = round(time.time() - t0, 1)
    return out
