"""Ground-truth experiment: how much data does a learned simulator need to recover the optimal
fourth-down policy?

The generator is the repo's exact game model: the solved MDP and its Simulator, with both teams
playing the coach-like policy (synthetic.py explains how a simulator step becomes records). Its
true action values at 300 held-out fourth downs are known (synthetic.ground_truth, saved as
ground_truth/gt_v1). The world model is trained on 1k, 4k, 16k and 64k synthetic games; each
model plans the same states inside itself (GO forced, resolved to a play mode by the model's own
go-set probabilities, which for these games is the coach's mode; FG; PUNT; actions the true model
does not allow are masked) and is graded against the truth.

Five stages, each idempotent, with done-markers, a time ledger and Drive storage as in run.py
(the same Run class, stage runner, deadline guard, chunk saving and JSON helpers):

  1 setup     the exact game: a given folder, the Drive cache, or built from public data with
              the repo's own pipeline (nflreadpy play-by-play, state table, development
              component bundle, count and coach tables, backward induction; about as long as
              scripts/solve_mdp.py). Then synthetic.verify_ground_truth against gt_v1; if it
              fails, a smaller ground truth is recomputed here and the results say so.
  2 generate  the largest training set once (smaller sizes are its prefixes) and a separate
              validation set (half for early stopping, half for the reported NLL)
  3 train     one world model per size with the same architecture and settings, smallest
              first, each with a time budget proportional to its size
  4 evaluate  per size: next-transition NLL on synthetic validation; plan the ground-truth
              states in two phases (every state with n0 rollouts per action, then the closest
              calls refined toward a target standard error of the go-minus-best-kick
              difference); decisions, regret and calibration against the truth
  5 package   gt_results.json (every number and the scaling table), figures, a zip

Run here (CPU smoke test, reusing a solved game so nothing is re-solved):

    python ground_truth_run.py --preset smoke --game-dir ../../data/mdp
"""

from __future__ import annotations

import copy
import json
import os
import resource
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import run as R
import schema as S
import synthetic as Y

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STAGES = ("setup", "generate", "train", "evaluate", "package")
# evaluate depends on generate only: it evaluates every size whose training is done, so a run
# whose largest size is still training (or was cut) still reports the smaller ones.
UPSTREAM = {"setup": (), "generate": ("setup",), "train": ("generate",),
            "evaluate": ("generate",), "package": ()}  # fmt: skip
STAGE_SOURCES = {"setup": ["synthetic"], "generate": ["synthetic", "schema"],
                 "train": ["model", "train"], "evaluate": ["simulate"], "package": []}  # fmt: skip
SEED = 20261121  # model initialisation, data order and rollout draws
VAL_SEED = 20261122  # synthetic validation games (training games use synthetic.TRAIN_SEED)
DEFAULT_SIZES = (1000, 4000, 16000, 64000)
CHOICES = ("GO", "FG", "PUNT")
# Coverage buckets: how often the coach-like policy takes the action in that state.
COVERAGE_EDGES = (0.0, 0.02, 0.10, 0.50, 1.0)
COVERAGE_LABELS = ("under 2%", "2 to 10%", "10 to 50%", "50% or more")
TRAIN_FILE, VAL_FILE = "train_games", "val_games"
RESULTS_NAME, ZIP_NAME = "gt_results.json", "world_model_gt_results.zip"
GT_PRESET_VERSION = "gt-1 (2026-11-19)"


# ---------------------------------------------------------------- presets
def gt_preset(name: str, sizes=None, **overrides: Any) -> dict:
    """The experiment's settings for a GPU class (model and optimizer from run.PRESETS[name],
    so the architecture follows the GPU as in run.py) or the CPU smoke test."""
    base = R.PRESETS[name]
    smoke = name == "smoke"
    p = {
        "name": name,
        "device": base["device"],
        "model": copy.deepcopy(base["model"]),
        # epoch_games: an epoch is at least this many games (a small set is repeated), so
        # every size takes the same number of optimizer steps per epoch at the start and the
        # warmup, patience and validation cadence mean the same thing for all sizes.
        "train": {**copy.deepcopy(base["train"]), "epoch_games": 400 if smoke else 4000},
        "sizes": sorted(int(s) for s in (sizes or ((200, 800) if smoke else DEFAULT_SIZES))),
        "val_games": {"early_stopping": 100 if smoke else 1000, "report": 100 if smoke else 1000},
        "generate": {"chunk_games": 1000, "workers": None},
        "eval": {"n_states": 30 if smoke else 300, "n0": 32 if smoke else 512,
                 "n_max": 128 if smoke else 8192, "target_se": 0.01,
                 "chunk_states": 10 if smoke else 20, "bootstrap": 2000,
                 "nll_games": 100 if smoke else 1000},
        # Only used when the solved game here differs from the one gt_v1 was computed with.
        "gt_recompute": {"n_states": 30 if smoke else 300, "n_rollouts": 2000 if smoke else 20000,
                         "n_pool_games": 1000 if smoke else 4000, "strata": None,
                         "headline": True, "max_abs_diff": 16},
        "budget_min": 25.0 if smoke else 90.0,
        "train_share": 0.45,  # of the time left after setup and generation
        "min_train_s": 15 if smoke else 120,
        "stage_min": {"setup": 40, "generate": 15, "train": 600, "evaluate": 600, "package": 3},
        "min_start_s": {"setup": 20, "generate": 10, "train": 30, "evaluate": 30, "package": 0},
        "reserve_package_s": 60 if smoke else 120,
        "assumed": dict(base["assumed"]),
        "benchmark_s": 10 if smoke else 30,
    }  # fmt: skip
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k].update(v)
        else:
            p[k] = v
    p["sizes"] = sorted(int(s) for s in p["sizes"])
    return p


def gt_definition(name: str) -> dict:
    return {"name": name, "version": GT_PRESET_VERSION,
            "hash": R.config_hash(gt_preset(name, sizes=DEFAULT_SIZES))}  # fmt: skip


def default_workdir() -> Path:
    if os.environ.get("GT_WORKDIR"):
        return Path(os.environ["GT_WORKDIR"])
    drive = Path("/content/drive/MyDrive")
    if drive.is_dir():
        return drive / "fourth_and_rl_world_model_gt"
    if Path("/content").is_dir():
        return Path("/content/wm_gt")
    return REPO / "results" / "world_model_gt_local"


def n_workers(preset: dict) -> int:
    w = preset["generate"].get("workers")
    return int(w) if w else max(1, min(os.cpu_count() or 1, 8))


# ---------------------------------------------------------------- run context
class GTRun(R.Run):
    """run.Run with this experiment's stages. Stage hashes are keyed on configuration and data
    content, never on code, as in run.py; per-size hashes make each trained model and its
    evaluation reusable when the list of sizes changes."""

    upstream = UPSTREAM
    stage_sources = STAGE_SOURCES

    def __init__(self, preset: dict, workdir: Path | None = None,
                 budget_min: float | None = None, seed: int = SEED, quiet: bool = False,
                 force: tuple = (), game_dir: str | Path | None = None,
                 cache_game: bool | None = None) -> None:  # fmt: skip
        super().__init__(preset, Path(workdir or default_workdir()), budget_min=budget_min,
                         seed=seed, quiet=quiet, force=force)  # fmt: skip
        self.game_dir_given = Path(game_dir).resolve() if game_dir else None
        on_drive = str(self.workdir).startswith("/content/drive")
        self.cache_game = on_drive if cache_game is None else bool(cache_game)
        self._game: Any = None
        self._gt: Any = None

    def stage_src(self, stage: str) -> str:
        return R._source_hash(STAGE_SOURCES[stage])

    def _setup_field(self, key: str) -> Any:
        p = self.output_path("setup")
        if not self.marker("setup").exists() or not p.exists():
            return "missing"
        return R.read_json(p).get(key, "missing")

    def size_hash(self, n: int) -> str:
        p = self.preset
        return R.config_hash({
            "stage": "train", "size": int(n), "seed": self.seed, "model": p["model"],
            "train": p["train"], "val_es": p["val_games"]["early_stopping"],
            "data_train": self.content_hash(TRAIN_FILE), "data_val": self.content_hash(VAL_FILE),
        })  # fmt: skip

    def eval_hash(self, n: int) -> str:
        return R.config_hash({"stage": "evaluate", "size": int(n), "train": self.size_hash(n),
                              "eval": self.preset["eval"], "val": self.preset["val_games"],
                              "gt": self._setup_field("gt_key")})  # fmt: skip

    def stage_hash(self, stage: str) -> str:
        p = self.preset
        if stage == "setup":
            parts: dict[str, Any] = {
                "game": str(self.game_dir_given) if self.game_dir_given else "repo pipeline",
                "gt_name": Y.GT_NAME, "gt_recompute": p["gt_recompute"]}  # fmt: skip
        elif stage == "generate":
            parts = {"solution": self._setup_field("solution_key"), "n_games": max(p["sizes"]),
                     "val_games": p["val_games"], "chunk_games": p["generate"]["chunk_games"],
                     "seeds": [Y.TRAIN_SEED, VAL_SEED]}  # fmt: skip
        elif stage == "train":
            parts = {"sizes": {str(n): self.size_hash(n) for n in p["sizes"]}}
        elif stage == "evaluate":
            parts = {"sizes": {str(n): self.eval_hash(n) for n in p["sizes"]}}
        else:
            parts = {}
        return R.config_hash({"stage": stage, **parts})

    def model_cfg(self) -> dict:
        cfg = super().model_cfg()
        cfg.pop("epoch_games", None)
        return cfg

    def data_path(self, name: str) -> Path:
        return self.workdir / "data" / f"{name}.npz"


# ---------------------------------------------------------------- the exact game
def _mg():
    from fourth_and_rl import mdp_game

    return mdp_game


def repo_game_dir() -> Path:
    """Where the repo's pipeline writes the solved game (configs/mdp.yaml, data/mdp)."""
    return _mg().SOLUTION_DIR


def game_dir(run: GTRun) -> Path:
    return run.game_dir_given or repo_game_dir()


def _has_game(d: Path) -> bool:
    return _mg().solution_exists(d) and (Path(d) / "tables.joblib").exists()


def _dir_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file())


def _try_cache_game(run: Any, src: Path, cache: Path) -> dict:
    """Cache the solved game on Drive when there is room. The cache only saves rebuild time
    on a later session, so a full Drive (or a failed copy) skips it instead of failing."""
    need = _dir_bytes(src)
    parent = Path(cache).parent
    try:
        free = shutil.disk_usage(parent if parent.exists() else run.workdir).free
    except OSError:
        free = None
    if free is not None and free < need * 1.1 + 2**28:
        msg = (f"not caching the solved game: it needs {need / 2**30:.1f} GB and the work "
               f"folder has {free / 2**30:.1f} GB free; a later session rebuilds it")
        run.log(f"[setup] {msg}")
        return {"cache_skipped": msg}
    run.log(f"[setup] caching the solved game in {cache} ({need / 2**30:.1f} GB)")
    try:
        return {"cache_seconds": copy_game(src, cache)}
    except OSError as e:
        for p in parent.glob(f".{Path(cache).name}.tmp-*"):
            shutil.rmtree(p, ignore_errors=True)
        msg = f"caching the solved game failed ({type(e).__name__}: {e}); continuing without it"
        run.log(f"[setup] {msg}")
        return {"cache_skipped": msg}


def copy_game(src: Path, dst: Path) -> float:
    """Copy a solved game folder (value and policy arrays, model, meta, tables), atomically:
    into a temporary folder next to dst, then one rename. Returns seconds."""
    t0 = time.time()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.tmp-{os.getpid()}-{time.time_ns()}"
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return round(time.time() - t0, 1)


def ensure_game(run: GTRun) -> Path:
    """The folder of the solved game, restoring it from the Drive cache when this machine is
    new (a resumed Colab session starts with an empty disk)."""
    d = game_dir(run)
    if _has_game(d):
        return d
    cache = run.workdir / "game"
    if run.game_dir_given is None and run.cache_game and _has_game(cache):
        run.log(f"[setup] restoring the solved game from the Drive cache {cache} to {d}")
        copy_game(cache, d)
        return d
    raise FileNotFoundError(f"no solved game under {d}; run the setup stage (FORCE=('setup',))")


def load_game(run: GTRun):
    """(solution, simulator with the coach table), loaded once per session."""
    if run._game is None:
        run._game = Y.load_simulator(ensure_game(run))
    return run._game


def _peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def _solve_mdp_module():
    import importlib.util

    path = REPO / "scripts" / "solve_mdp.py"
    spec = importlib.util.spec_from_file_location("gt_solve_mdp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_game(log=print) -> dict:
    """Build the exact game from public data with the repo's own pipeline, writing where the
    scripts write (data/raw, data/processed, data/models, data/mdp of this checkout):

      1 scripts/build_data.py: download nflverse play-by-play 2010-2025 with nflreadpy and
        build the state table (fourth_and_rl.data.build)
      2 the development and test files (splits.write_split_files)
      3 scripts/fit_components.py: the development component bundle (fit_dev_bundle, with
        the kickoff tables of the 2024-2025 rules as its one logged exception)
      4 scripts/solve_mdp.py: count, mode and coach tables on development seasons
        (load_tables), the game model, backward induction, saved with the same meta

    Refuses to run when this checkout's data folder is a link to another checkout's data, so
    a test can never overwrite real cached data."""
    from fourth_and_rl import components as cm
    from fourth_and_rl import data, mdp, paths, splits

    mg = _mg()
    if paths.DATA.is_symlink() or not paths.DATA.resolve().is_relative_to(paths.ROOT.resolve()):
        raise RuntimeError(f"refusing to build into {paths.DATA}: it links outside this "
                           "checkout (pass a solved game folder instead)")  # fmt: skip
    sec: dict[str, float] = {}
    t = time.time()

    def lap(name: str) -> None:
        nonlocal t
        sec[name] = round(time.time() - t, 1)
        log(f"[setup] {name}: {sec[name]:.0f}s (peak memory so far {_peak_rss_gb()} GB)")
        t = time.time()

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
        "ground-truth experiment setup: kickoff start tables of the 2024 and 2025 rules for the "
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
    model = mg.build_model(cfg, models, bundle["tables"], horizon="game")
    sec_kernel = time.time() - t
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
            "n_states": sol.meta.get("n_states"), "solution_dir": str(out),
            "saved_gb": round(sum(f.stat().st_size for f in out.iterdir()) / 2**30, 2)}  # fmt: skip


def _file_hash(paths_: list[Path]) -> str:
    import hashlib

    h = hashlib.sha256()
    for p in paths_:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def recompute_ground_truth(run: GTRun, sol, sim, reasons: list[str]) -> dict:
    """A smaller ground truth for the solved game here, saved in the work folder."""
    c = run.preset["gt_recompute"]
    out_dir, name = run.workdir / "ground_truth", "gt_colab"
    meta_path = out_dir / f"{name}.json"
    stamp = Y.solution_stamp(sol.meta)
    if meta_path.exists():
        old = R.read_json(meta_path)
        if old.get("solution") == stamp and old.get("recompute_cfg") == c:
            run.log(f"[setup] reusing the ground truth recomputed earlier ({meta_path})")
            return old
    run.log(f"[setup] recomputing a ground truth: {c['n_states']} states x {c['n_rollouts']} "
            f"rollouts per action ({n_workers(run.preset)} processes)")
    meta = Y.precompute(c["n_states"], c["n_rollouts"], n_workers(run.preset), out_dir, name,
                        game=(sol, sim),
                        strata=tuple(tuple(s) for s in c["strata"]) if c["strata"]
                        else Y.DEFAULT_STRATA,
                        n_pool_games=c["n_pool_games"], max_abs_diff=c["max_abs_diff"],
                        headline=c["headline"], bench=False)  # fmt: skip
    meta.update(version=name, recompute_cfg=c, recomputed_because=reasons)
    R.write_json(meta_path, meta)
    return meta


def stage_setup(run: GTRun, deadline: float) -> dict:
    out: dict[str, Any] = {}
    d = game_dir(run)
    cache = run.workdir / "game"
    t0 = time.time()
    if run.game_dir_given is not None:
        if not _has_game(d):
            raise FileNotFoundError(f"GAME_DIR {d} holds no solved game with tables.joblib")
        out["source"] = f"given folder {d} (read only)"
    elif _has_game(d):
        out["source"] = f"solved game already in this checkout ({d}), reused"
    elif run.cache_game and _has_game(cache):
        out["restore_seconds"] = copy_game(cache, d)
        out["source"] = f"Drive cache {cache}"
    else:
        run.log("[setup] building the exact game from public data (download, fit, solve)")
        out["build"] = build_game(run.log)
        out["source"] = "built from public data in this session"
    if run.game_dir_given is None and run.cache_game and not _has_game(cache):
        out.update(_try_cache_game(run, d, cache))
    out["game_dir"] = str(d)
    run._game = None
    sol, sim = load_game(run)
    out["load_seconds"] = round(time.time() - t0, 1)
    out["solution_stamp"] = Y.solution_stamp(sol.meta)
    out["solution_key"] = R.config_hash(out["solution_stamp"])
    v = Y.verify_ground_truth(sol.meta)
    if v["ok"]:  # same stamp: also compare the solved values at every evaluation state
        v = Y.verify_ground_truth(sol.meta, sol=sol)
    out["verify"] = v
    if v["ok"]:
        gdir, name = Y.GT_DIR, Y.GT_NAME
        out["ground_truth"] = {"name": name, "recomputed": False,
                               "note": "the saved ground truth matches this solved game"}
        run.log(f"[setup] ground truth {name} verified against this solved game")
    else:
        run.log(f"[setup] the saved ground truth does not match this game: {v['mismatches']}")
        meta = recompute_ground_truth(run, sol, sim, v["mismatches"])
        gdir, name = run.workdir / "ground_truth", "gt_colab"
        out["ground_truth"] = {"name": name, "recomputed": True, "reasons": v["mismatches"],
                               "n_states": meta["n_states"], "n_rollouts": meta["n_rollouts"],
                               "se_coach": meta["se_coach"]}  # fmt: skip
    files = [gdir / f"{name}.npz", gdir / f"{name}_states.npz", gdir / f"{name}.json"]
    out["ground_truth"]["files"] = [str(f) for f in files]
    out["gt_key"] = _file_hash(files[:2])
    out["peak_rss_gb_process"] = _peak_rss_gb()
    return out


def load_gt(run: GTRun):
    """(ground truth arrays, its meta, evaluation states) chosen by the setup stage."""
    if run._gt is None:
        files = [Path(f) for f in R.read_json(run.output_path("setup"))["ground_truth"]["files"]]
        gt, meta = Y.load_ground_truth(files[0])
        run._gt = (gt, meta, Y.load_eval_states(files[1]))
    return run._gt


# ---------------------------------------------------------------- stage 2: generate
def stage_generate(run: GTRun, deadline: float) -> dict:
    _sol, sim = load_game(run)
    p = run.preset
    n, w, chunk = max(p["sizes"]), n_workers(p), int(p["generate"]["chunk_games"])
    nv = int(p["val_games"]["early_stopping"]) + int(p["val_games"]["report"])
    t0 = time.time()
    tr = Y.simulate_games(sim, n, seed=Y.TRAIN_SEED, chunk_games=chunk, n_workers=w)
    t_tr = time.time() - t0
    R.save_split(run.data_path(TRAIN_FILE), tr)
    run.log(f"[generate] {n:,} training games, {R.n_records(tr):,} records in {t_tr:.0f}s")
    t1 = time.time()
    va = Y.simulate_games(sim, nv, seed=VAL_SEED, chunk_games=chunk, n_workers=w,
                          id_prefix="synval")  # fmt: skip
    R.save_split(run.data_path(VAL_FILE), va)
    run.log(f"[generate] {nv:,} validation games in {time.time() - t1:.0f}s")
    return {
        "train_games": n, "train_records": R.n_records(tr),
        "records_per_game": round(R.n_records(tr) / n, 1),
        "val_games": nv, "val_records": R.n_records(va),
        "seconds_train_games": round(t_tr, 1), "games_per_second": round(n / max(t_tr, 1e-6), 1),
        "workers": w, "chunk_games": chunk, "seeds": {"train": Y.TRAIN_SEED, "val": VAL_SEED},
        "league_val": R.real_league_stats(va),
        "file_mb": {k: round(run.data_path(k).stat().st_size / 1e6, 1)
                    for k in (TRAIN_FILE, VAL_FILE)},
        "prefix_note": "the first k games of the training set are the k-game set "
                       "(chunk seeds), so every size is a prefix of the largest",
    }  # fmt: skip


# ---------------------------------------------------------------- stage 3: train
def train_state_path(run: GTRun, n: int) -> Path:
    return run.workdir / "outputs" / f"train_size_{n}.json"


def ckpt_path(run: GTRun, n: int) -> Path:
    return run.workdir / "ckpt" / f"size_{n}_{run.size_hash(n)}.pt"


def train_state(run: GTRun, n: int) -> dict:
    path = train_state_path(run, n)
    st = R.read_json(path) if path.exists() else {}
    return st if st.get("hash") == run.size_hash(n) else {}


def size_trained(run: GTRun, n: int) -> bool:
    return bool(train_state(run, n).get("done")) and ckpt_path(run, n).exists()


def training_subset(split: S.Split, n: int, epoch_games: int = 0) -> S.Split:
    """The first n games, repeated so that one pass has at least epoch_games games."""
    reps = max(1, -(-int(epoch_games) // int(n))) if epoch_games else 1
    return Y.subset(split, np.tile(np.arange(int(n)), reps))


def allocate_budget(sizes_left: list[int], seconds_left: float, floor_s: float) -> float:
    """Seconds for the first (smallest) of the sizes still to train: its share of the time
    left in proportion to its number of games, at least floor_s."""
    return max(float(floor_s), max(seconds_left, 0.0) * sizes_left[0] / float(sum(sizes_left)))


def plan_epochs(budget_s: float, records_per_epoch: int, records_per_s: float,
                max_epochs: int, min_epochs: int = 3) -> int:  # fmt: skip
    """The cosine schedule's length in epochs: what the time budget fits at the measured
    training speed (with a fifth kept for validation and checkpoints), within
    [min_epochs, max_epochs]. Early stopping still decides when training ends."""
    fit = 0.8 * budget_s * records_per_s / max(records_per_epoch, 1)
    return int(np.clip(np.floor(fit), min_epochs, max_epochs))


def train_speed(preset: dict) -> tuple[float, str]:
    b = preset.get("benchmark") or {}
    if b.get("train_records_per_s"):
        return float(b["train_records_per_s"]), "measured by the preflight benchmark"
    return float(preset["assumed"]["train_rec_per_s"]), "the preset's assumed speed"


def stage_train(run: GTRun, deadline: float) -> dict:
    import torch

    model_mod, train_mod = R._mod("model"), R._mod("train")
    p = run.preset
    sizes = sorted(p["sizes"])
    out: dict[str, Any] = {"sizes": {}}
    pending = [n for n in sizes if not size_trained(run, n)]
    if pending:
        tr_all, _ = R.load_split(run.data_path(TRAIN_FILE))
        va, _ = R.load_split(run.data_path(VAL_FILE))
        va_es = Y.subset(va, np.arange(int(p["val_games"]["early_stopping"])))
        pool_end = time.time() + max(deadline - time.time(), 0.0) * float(p["train_share"])
        speed, basis = train_speed(p)
        out["planning"] = {"train_share": p["train_share"], "records_per_s": speed,
                           "speed_basis": basis,
                           "pool_minutes": round((pool_end - time.time()) / 60, 1)}
    for n in pending:
        if time.time() >= deadline:
            break
        st = train_state(run, n) or {"hash": run.size_hash(n), "size": n, "used_s": 0.0}
        sub = training_subset(tr_all, n, int(p["train"]["epoch_games"]))
        if "budget_s" not in st:
            left = [m for m in pending if not size_trained(run, m) and m >= n]
            st["budget_s"] = round(allocate_budget(left, pool_end - time.time(),
                                                   p["min_train_s"]), 1)
            st["epochs"] = plan_epochs(st["budget_s"], R.n_records(sub), speed,
                                       int(p["train"]["epochs"]))
            st["epoch_games_effective"] = len(sub.game_ids)
            R.write_json(train_state_path(run, n), st)
        remaining = float(st["budget_s"]) - float(st["used_s"])
        ck = ckpt_path(run, n)
        cfg = {**run.model_cfg(), "epochs": int(st["epochs"])}
        torch.manual_seed(run.seed)
        np.random.seed(run.seed % 2**32)
        model = model_mod.WorldModel(cfg)
        stale = R._check_checkpoint(ck, model)
        if stale:
            ck.replace(ck.with_name(ck.name + f".stale_{int(time.time())}"))
            run.log(f"[train {n}] checkpoint does not fit the model ({stale}); starting fresh")
        if remaining > 1.0 or not ck.exists():
            own_end = time.time() + max(remaining, float(p["min_start_s"]["train"]))
            dl = min(own_end, deadline)
            t0, base = time.time(), float(st["used_s"])

            def log(*msg: Any, _st=st, _t0=t0, _base=base, _n=n) -> None:
                _st["used_s"] = round(_base + time.time() - _t0, 1)
                R.write_json(train_state_path(run, _n), _st)
                run.log(f"[train {_n}]", *msg)

            run.log(f"[train {n}] {model.count_params():,} parameters; {n:,} games "
                    f"({len(sub.game_ids):,} per epoch, {R.n_records(sub):,} records); budget "
                    f"{st['budget_s'] / 60:.1f} min ({remaining / 60:.1f} left); cosine over "
                    f"{st['epochs']} epochs; device {cfg['device']}")
            hist = train_mod.train_model(model, sub, va_es, cfg, str(ck), dl, log)
            st["used_s"] = round(base + time.time() - t0, 1)
            st["params"] = int(model.count_params())
            st["history"] = R._jsonable(hist)
            stopped = hist.get("stopped")
            if stopped in ("early_stopping", "finished", "epochs"):
                st["done"], st["stopped"] = True, "converged" if stopped == "early_stopping" \
                    else "epoch limit"
            elif dl >= own_end - 1.0 and time.time() >= own_end - 2.0:
                st["done"], st["stopped"] = True, "its time budget"
            else:
                st["stopped"] = "the run's deadline (resumes on the next run)"
        else:
            st["done"], st["stopped"] = True, "its time budget"
        R.write_json(train_state_path(run, n), st)
        h = st.get("history") or {}
        run.log(f"[train {n}] stopped by {st['stopped']}: {h.get('epochs_done')} epochs, best "
                f"validation NLL {h.get('best_val_nll')}, {st['used_s']:.0f}s")
    for n in sizes:
        st = train_state(run, n)
        h = st.get("history") or {}
        out["sizes"][str(n)] = {
            "done": bool(st.get("done")), "stopped": st.get("stopped"),
            "budget_s": st.get("budget_s"), "used_s": st.get("used_s"),
            "cosine_epochs": st.get("epochs"), "epoch_games": st.get("epoch_games_effective"),
            "params": st.get("params"), "epochs_done": h.get("epochs_done"),
            "best_epoch": h.get("best_epoch"), "best_val_nll": h.get("best_val_nll"),
            "records_per_s": h.get("train_records_per_s"), "device": h.get("device"),
            "val_curve": [e.get("val_total") for e in h.get("history") or []],
        }  # fmt: skip
    if not all(size_trained(run, n) for n in sizes):
        out["status"] = "partial"
        out["reason"] = "not every size has finished training; the next run continues"
    return out


# ---------------------------------------------------------------- stage 4: evaluate
def state_order(kind: np.ndarray, n_states: int, seed: int = SEED) -> np.ndarray:
    """Positions of the evaluation states in planning order: the headline state first, then a
    fixed random permutation, so any prefix is a random subsample (a deadline that cuts the
    list keeps a representative set)."""
    kind = np.asarray(kind)
    head = np.flatnonzero(kind == 1)
    rest = np.flatnonzero(kind != 1)
    rest = rest[np.random.default_rng(seed).permutation(len(rest))]
    return np.r_[head, rest][: int(n_states)].astype(np.int64)


def rollouts_needed(se_diff: np.ndarray, n0: int, target: float, n_max: int) -> np.ndarray:
    """Total rollouts per action so the go-minus-best-kick standard error, measured with n0
    rollouts, reaches the target (it falls as 1/sqrt(n)): 10% margin, rounded up to a
    multiple of n0, capped at n_max."""
    se = np.nan_to_num(np.asarray(se_diff, dtype=float), nan=0.0)
    need = n0 * (se / target) ** 2 * 1.1
    return np.minimum(n0 * np.ceil(np.maximum(need, n0) / n0), n_max).astype(np.int64)


def refine_mask(wp: np.ndarray, se: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Actions to top up in phase B: GO, the best kick by the estimate, and the other kick
    only when it is within two standard errors of the best kick (the choice between kicks is
    then still open)."""
    kick = np.where(allowed[:, 1:] & np.isfinite(wp[:, 1:]), wp[:, 1:], -np.inf)
    k = np.argmax(kick, axis=1)
    r = np.arange(len(wp))
    other = 1 - k
    gap = kick[r, k] - kick[r, other]
    se_gap = np.sqrt(np.nan_to_num(se[:, 1:][r, k]) ** 2 + np.nan_to_num(se[:, 1:][r, other]) ** 2)
    m = np.zeros_like(allowed, dtype=bool)
    m[:, 0] = True
    m[r, 1 + k] = np.isfinite(kick[r, k])
    m[r, 1 + other] = np.isfinite(kick[r, other]) & (gap < 2 * se_gap)
    return m & allowed


def combine(wp_a, se_a, n_a, wp_b, se_b, n_b):
    """Pool two independent batches of rollouts: weighted mean, and the standard error of
    that weighted mean (NaN entries of the second batch leave the first unchanged)."""
    wp_a, se_a, wp_b, se_b = (np.asarray(x, dtype=float) for x in (wp_a, se_a, wp_b, se_b))
    na, nb = np.asarray(n_a, dtype=float), np.asarray(n_b, dtype=float)
    na = na if na.ndim == 2 else na[:, None]
    nb = (nb if nb.ndim == 2 else nb[:, None]) * np.isfinite(wp_b)
    tot = na + nb
    wp = (na * wp_a + nb * np.nan_to_num(wp_b)) / tot
    se = np.sqrt(na**2 * se_a**2 + nb**2 * np.nan_to_num(se_b) ** 2) / tot
    return wp, se


def _go_kick(wp: np.ndarray, se: np.ndarray, allowed: np.ndarray):
    """Go-minus-best-kick difference and its standard error (best kick by the estimate)."""
    kick = np.where(allowed[:, 1:] & np.isfinite(wp[:, 1:]), wp[:, 1:], -np.inf)
    k = 1 + np.argmax(kick, axis=1)
    r = np.arange(len(wp))
    has_kick = np.isfinite(kick).any(axis=1)
    diff = np.where(has_kick, wp[:, 0] - wp[r, k], np.nan)
    se_diff = np.where(has_kick, np.sqrt(se[:, 0] ** 2 + se[r, k] ** 2), np.nan)
    return diff, se_diff


_WP_COLS = [f"wp_{c.lower()}" for c in CHOICES]
_SE_COLS = [f"se_{c.lower()}" for c in CHOICES]


def plan_states(run: GTRun, model, cfg: dict, ev, allowed_all: np.ndarray, order: np.ndarray,
                pdir: Path, deadline: float, seed: int) -> dict:  # fmt: skip
    """Learned action values at the states `order` (positions in ev), in two phases saved in
    chunks: (A) every state with n0 rollouts per allowed action; (B) states whose go-minus-
    best-kick standard error is above the target, closest calls first, with GO and the best
    kick (and the other kick when it is tied with the best) topped up to the rollouts that
    target needs (at most n_max). A deadline cuts phase A (fewer states) or phase B (fewer
    refined states); both are recorded. n is per action: [states, 3]."""
    sim = R._mod("simulate")
    e = run.preset["eval"]
    n0, chunk = int(e["n0"]), int(e["chunk_states"])
    pos_of = {int(ev.idx[i]): i for i in range(len(ev))}

    def mask(part) -> np.ndarray:
        return allowed_all[[pos_of[int(i)] for i in part]]

    def fn_a(part, rng):
        return sim.plan_fourth_downs(model, ev.split, part, n0, cfg, deadline, rng,
                                     action_mask=mask(part))  # fmt: skip

    df_a, info_a = R.chunked(run, pdir / "base", fn_a, ev.idx[order], chunk, deadline, seed)
    info: dict[str, Any] = {"phase_a": info_a, "n0": n0, "target_se": e["target_se"],
                            "n_max": int(e["n_max"])}  # fmt: skip
    if len(df_a) == 0:
        info["phase_b"] = {"states_wanting_more": 0, "complete": False}
        return {"pos": np.zeros(0, np.int64), "info": info}
    df_a = df_a[df_a["done"].astype(bool)].reset_index(drop=True)
    pos = np.array([pos_of[int(i)] for i in df_a["idx"]], dtype=np.int64)
    allowed = allowed_all[pos]
    wp, se = df_a[_WP_COLS].to_numpy(float), df_a[_SE_COLS].to_numpy(float)
    n = np.where(allowed, n0, 0).astype(np.int64)
    counters_b = None
    _, se_diff = _go_kick(wp, se, allowed)
    total = rollouts_needed(se_diff, n0, float(e["target_se"]), int(e["n_max"]))
    extra = np.where(info_a["complete"], total - n0, 0)
    if (extra > 0).any():
        diff, _ = _go_kick(wp, se, allowed)
        closeness = np.abs(diff) / np.maximum(se_diff, 1e-9)
        todo = np.flatnonzero(extra > 0)
        todo = todo[np.argsort(closeness[todo], kind="stable")]
        extra_of = {int(df_a["idx"][j]): int(extra[j]) for j in todo}
        rmask = refine_mask(wp, se, allowed)
        rmask_of = {int(df_a["idx"][j]): rmask[j] for j in todo}

        def fn_b(part, rng):
            import pandas as pd

            frames = []
            for ex in sorted({extra_of[int(i)] for i in part}):
                sub = np.array([i for i in part if extra_of[int(i)] == ex], dtype=np.int64)
                df = sim.plan_fourth_downs(model, ev.split, sub, ex, cfg, deadline, rng,
                                           action_mask=np.stack([rmask_of[int(i)] for i in sub]))
                df["n_extra"] = ex
                frames.append(df)
            return pd.concat(frames, ignore_index=True)

        df_b, info_b = R.chunked(run, pdir / "refine", fn_b, df_a["idx"].to_numpy()[todo],
                                 chunk, deadline, seed + 1)  # fmt: skip
        counters_b = info_b.get("rollout_counters")
        info["phase_b"] = {k: v for k, v in info_b.items() if k != "rollout_counters"}
        info["phase_b"]["states_wanting_more"] = int(len(todo))
        if len(df_b):
            df_b = df_b[df_b["done"].astype(bool)]
            row_of = {int(i): j for j, i in enumerate(df_a["idx"])}
            rows = np.array([row_of[int(i)] for i in df_b["idx"]], dtype=np.int64)
            wp_b = df_b[_WP_COLS].to_numpy(float)
            nb = df_b["n_extra"].to_numpy(np.int64)[:, None] * np.isfinite(wp_b)
            wp[rows], se[rows] = combine(wp[rows], se[rows], n[rows], wp_b,
                                         df_b[_SE_COLS].to_numpy(float), nb)  # fmt: skip
            n[rows] += nb.astype(np.int64)
    else:
        info["phase_b"] = {"states_wanting_more": 0, "complete": bool(info_a["complete"])}
    _, se_diff = _go_kick(wp, se, allowed)
    ok = np.isfinite(se_diff)
    info["se_go_minus_best_kick"] = {
        "mean": float(np.mean(se_diff[ok])) if ok.any() else None,
        "max": float(np.max(se_diff[ok])) if ok.any() else None,
        "share_at_or_below_target": float(np.mean(se_diff[ok] <= e["target_se"] * 1.0001))
        if ok.any() else None,
        "share_at_the_rollout_cap": float(np.mean(n[:, 0] >= int(e["n_max"]))),
    }  # fmt: skip
    info["rollouts_go"] = {"mean": float(n[:, 0].mean()), "max": int(n[:, 0].max()),
                           "min": int(n[:, 0].min())}  # fmt: skip
    info["rollout_counters"] = _sum_counters(info_a.get("rollout_counters"), counters_b)
    return {"pos": pos, "wp": wp, "se": se, "n": n, "info": info}


def _sum_counters(*lists) -> dict:
    """Rollout counters (simulate.COUNTER_KEYS) summed over chunks and actions: transitions
    simulated, truncated rollouts, inconsistent transitions."""
    tot: dict[str, int] = {}
    for lst in lists:
        for c in lst or []:
            for per in (c or {}).values():
                for k, v in (per or {}).items():
                    tot[k] = tot.get(k, 0) + int(v)
    return tot


def _boot_ci(values_fn, n: int, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    vals = np.array([values_fn(rng.integers(0, n, n)) for _ in range(n_boot)], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _stat(values: np.ndarray, fn, rng, n_boot: int, mask: np.ndarray | None = None) -> dict:
    """fn over values (optionally restricted by mask), with a bootstrap over states."""
    values = np.asarray(values, dtype=float)
    m = np.ones(len(values), bool) if mask is None else np.asarray(mask, bool)

    def f(ix):
        v = values[ix][m[ix]]
        return fn(v) if len(v) else np.nan

    lo, hi = _boot_ci(f, len(values), rng, n_boot)
    v = values[m]
    return {"value": float(fn(v)) if len(v) else None, "lo95": lo, "hi95": hi, "n": int(m.sum())}


def _p90(v: np.ndarray) -> float:
    return float(np.percentile(v, 90))


def score_decisions(wp: np.ndarray, se: np.ndarray, q_coach: np.ndarray, q_star: np.ndarray,
                    allowed: np.ndarray, coach_probs: np.ndarray, rng: np.random.Generator,
                    n_boot: int = 2000) -> dict:  # fmt: skip
    """Grade learned action values against the truth. Columns are GO, FG, PUNT; q_coach is
    the true value with coach-like play afterwards (GO in the coach's mode), q_star the
    solved value (GO in the best mode, optimal play afterwards). Values are win
    probabilities; regrets and errors are reported in win-probability points (x 100)."""
    N = len(wp)
    r = np.arange(N)
    A = np.asarray(allowed, bool) & np.isfinite(q_coach)
    qa = np.where(A, q_coach, np.nan)
    wa = np.where(A & np.isfinite(wp), wp, np.nan)
    best = np.nanargmax(np.where(A, q_coach, -np.inf), axis=1)
    chosen = np.nanargmax(np.where(np.isfinite(wa), wa, -np.inf), axis=1)
    srt = np.sort(np.where(A, q_coach, -np.inf), axis=1)
    margin = 100 * (srt[:, -1] - srt[:, -2])  # inf when only one action is allowed
    regret = 100 * (qa[r, best] - qa[r, chosen])
    correct = (chosen == best).astype(float)
    m1, m3 = margin > 1, margin > 3
    out: dict[str, Any] = {"n_states": int(N)}
    out["accuracy"] = {"all": _stat(correct, np.mean, rng, n_boot),
                       "true_margin_over_1_point": _stat(correct, np.mean, rng, n_boot, m1),
                       "true_margin_over_3_points": _stat(correct, np.mean, rng, n_boot, m3)}
    out["regret_points"] = {"mean": _stat(regret, np.mean, rng, n_boot),
                            "p90": _stat(regret, _p90, rng, n_boot),
                            "share_zero": float(np.mean(regret <= 1e-12))}  # fmt: skip
    # The coach-like policy itself (it randomises): expected regret and P(best action).
    cp = np.where(A, np.nan_to_num(coach_probs), 0.0)
    cp = cp / np.maximum(cp.sum(axis=1, keepdims=True), 1e-12)
    coach_regret = 100 * (qa[r, best] - np.sum(cp * np.nan_to_num(qa), axis=1))
    out["coach_policy"] = {"regret_points_mean": _stat(coach_regret, np.mean, rng, n_boot),
                           "regret_points_p90": float(_p90(coach_regret)),
                           "p_best_action": _stat(cp[r, best], np.mean, rng, n_boot)}
    # Against the solved values (optimal play afterwards).
    sa = np.where(A & np.isfinite(q_star), q_star, np.nan)
    sbest = np.nanargmax(np.where(np.isfinite(sa), sa, -np.inf), axis=1)
    out["q_star"] = {
        "agreement": _stat((chosen == sbest).astype(float), np.mean, rng, n_boot),
        "regret_points_mean": _stat(100 * (sa[r, sbest] - sa[r, chosen]), np.mean, rng, n_boot),
        "true_coach_best_equals_star_best": float(np.mean(best == sbest)),
        "note": "GO is valued in the best play mode here (q_star go_best); the learned "
                "planner's GO is the coach's mode",
    }  # fmt: skip
    # Calibration of the learned values against q_coach (points, learned minus true).
    err = 100 * (wa - qa)
    fin = np.isfinite(err)
    by_action = {}
    for j, c in enumerate(CHOICES):
        m = fin[:, j]
        by_action[c] = {"mean_error": float(err[m, j].mean()) if m.any() else None,
                        "mean_abs_error": float(np.abs(err[m, j]).mean()) if m.any() else None,
                        "n": int(m.sum())}  # fmt: skip
    share = np.asarray(coach_probs, float)
    bucket = np.digitize(share, COVERAGE_EDGES[1:-1])
    cov = []
    for b, label in enumerate(COVERAGE_LABELS):
        m = fin & (bucket == b)
        v = err[m]
        se_b = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
        cov.append({"bucket": label,
                    "coach_share_range": [COVERAGE_EDGES[b], COVERAGE_EDGES[b + 1]],
                    "n": int(m.sum()), "mean_error": float(v.mean()) if len(v) else None,
                    "lo95": float(v.mean() - 1.96 * se_b) if se_b is not None else None,
                    "hi95": float(v.mean() + 1.96 * se_b) if se_b is not None else None,
                    "by_action_n": {c: int((m[:, j]).sum()) for j, c in enumerate(CHOICES)}})
    row_err = np.where(fin, err, np.nan)
    out["calibration"] = {
        "mean_error_points": _stat(np.nanmean(row_err, axis=1), np.mean, rng, n_boot),
        "mean_abs_error_points": float(np.abs(err[fin]).mean()),
        "by_action": by_action,
        "by_coach_share": cov,
        "chosen_action_mean_error_points": float(np.nanmean(err[r, chosen])),
        "mean_rollout_se_points": float(100 * np.nanmean(np.where(fin, se, np.nan))),
        "note": "error = learned value minus true value (q_coach), win-probability points; "
                "coach share = how often the coach-like policy takes that action in that "
                "state (synthetic.coach_choice_probs)",
    }  # fmt: skip
    out["per_state"] = {"chosen": [CHOICES[i] for i in chosen],
                        "true_best": [CHOICES[i] for i in best],
                        "regret_points": regret.round(3).tolist(),
                        "true_margin_points": np.where(np.isfinite(margin), margin, None).tolist()}
    return out


def evaluate_size(run: GTRun, n: int, deadline: float) -> dict:
    train_mod, sim, model_mod = R._mod("train"), R._mod("simulate"), R._mod("model")
    p, e = run.preset, run.preset["eval"]
    pdir = run.workdir / "partial" / f"eval_{n}_{run.eval_hash(n)}"
    pdir.mkdir(parents=True, exist_ok=True)
    done_path = pdir / "result.json"
    if done_path.exists():
        return R.read_json(done_path)
    gt, gmeta, ev = load_gt(run)
    order = state_order(ev.kind, int(e["n_states"]))
    model, _ = train_mod.load_checkpoint(str(ckpt_path(run, n)), device=run.device())
    cfg = run.model_cfg()
    out: dict[str, Any] = {"size": int(n)}

    def nll():
        va, _ = R.load_split(run.data_path(VAL_FILE))
        a = int(p["val_games"]["early_stopping"])
        b = a + min(int(p["val_games"]["report"]), int(e["nll_games"]))
        rep = Y.subset(va, np.arange(a, b))
        tr_all, _ = R.load_split(run.data_path(TRAIN_FILE))
        tables = R.fit_frequency_baseline(training_subset(tr_all, n))
        return {"games": b - a, "model": train_mod.evaluate_nll(model, rep, cfg),
                "baseline_freq_phase_down": R.baseline_nll(tables, rep),
                "note": "validation games never seen in training or early stopping"}

    out["nll"] = R._cached_json(pdir / "nll.json", nll)
    run.log(f"[evaluate {n}] NLL per transition: model {out['nll']['model']['total']:.4f}, "
            f"frequency baseline "
            f"{out['nll']['baseline_freq_phase_down']['total_per_transition']:.4f}")

    def go_modes():
        mc = model_mod.merge_cfg({**model.cfg, **cfg})
        probs = sim._go_probs(model, ev.split, ev.idx[order], mc, S.PLAY_MODE_ACTIONS)
        coach = Y.action_table()[gt["go_coach_code"][order]]
        j = np.searchsorted(np.asarray(S.PLAY_MODE_ACTIONS), coach)
        pc = probs[np.arange(len(order)), j]
        return {"mean_p_coach_mode": float(pc.mean()), "min_p_coach_mode": float(pc.min()),
                "share_argmax_is_coach_mode": float(np.mean(probs.argmax(1) == j)),
                "note": "forced GO samples a play mode from the model's go-set probabilities; "
                        "the coach-like policy's mode is deterministic, so a model that learned "
                        "it puts this probability near 1"}  # fmt: skip

    out["go_mode"] = R._cached_json(pdir / "go_mode.json", go_modes)
    allowed_all = np.asarray(gt["allowed_model"][:, 1:4], bool)
    t0 = time.time()
    plan = plan_states(run, model, cfg, ev, allowed_all, order, pdir, deadline,
                       run.seed + int(n))  # fmt: skip
    out["plan"] = plan["info"]
    out["plan_seconds_this_session"] = round(time.time() - t0, 1)
    pos = plan["pos"]
    out["n_states_planned"] = int(len(order))
    out["n_states_evaluated"] = int(len(pos))
    if len(pos):
        rng = np.random.default_rng(run.seed + 1000 + int(n))
        sc = score_decisions(plan["wp"], plan["se"], gt["q_coach"][pos][:, 1:4],
                             gt["q_star"][pos][:, [0, 2, 3]], allowed_all[pos],
                             gt["coach_probs"][pos], rng, int(e["bootstrap"]))  # fmt: skip
        out.update({k: v for k, v in sc.items() if k != "per_state"})
        ps = sc["per_state"]
        ps.update(position=pos.tolist(), wp=plan["wp"].round(5).tolist(),
                  se=plan["se"].round(5).tolist(), n_rollouts=plan["n"].tolist(),
                  q_coach=gt["q_coach"][pos][:, 1:4].round(5).tolist(),
                  q_star=gt["q_star"][pos][:, [0, 2, 3]].round(5).tolist(),
                  coach_probs=gt["coach_probs"][pos].round(4).tolist(),
                  t_ticks=ev.t[pos].tolist(), score=ev.s[pos].tolist())
        out["per_state"] = ps
        hl = np.flatnonzero(ev.kind[pos] == 1)
        if len(hl):
            i, k = int(hl[0]), int(pos[hl[0]])
            out["headline"] = {
                "state": gmeta.get("headline"),
                "learned": dict(zip(CHOICES, _nan_none(plan["wp"][i]), strict=True)),
                "learned_se": dict(zip(CHOICES, _nan_none(plan["se"][i]), strict=True)),
                "true_q_coach": dict(zip(CHOICES, _nan_none(gt["q_coach"][k, 1:4]), strict=True)),
                "true_q_star": dict(zip(CHOICES, _nan_none(gt["q_star"][k, [0, 2, 3]]),
                                        strict=True)),
                "learned_choice": ps["chosen"][i], "true_best": ps["true_best"][i],
            }  # fmt: skip
    complete = bool(out["plan"]["phase_a"]["complete"])
    if complete and len(pos) == 0:
        raise RuntimeError("no evaluation state could be planned (are they fourth downs?)")
    out["status"] = "done" if complete else "partial"
    if not complete:
        out["reason"] = (f"time budget: {len(pos)} of {len(order)} states planned; the next run "
                         "continues from the saved chunks")
    elif not out["plan"]["phase_b"].get("complete", True):
        out["note_rollouts"] = ("time budget: not every state reached the target standard "
                                "error; see plan.se_go_minus_best_kick")
    out = R._jsonable(out)
    if complete:
        R.write_json(done_path, out)
    return out


def _nan_none(a) -> list:
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(a, dtype=float)]


def eval_done(run: GTRun, n: int) -> bool:
    return (run.workdir / "partial" / f"eval_{n}_{run.eval_hash(n)}" / "result.json").exists()


def stage_evaluate(run: GTRun, deadline: float) -> dict:
    sizes = sorted(run.preset["sizes"])
    out: dict[str, Any] = {"sizes": {}}
    for n in sizes:
        if not size_trained(run, n):
            out["sizes"][str(n)] = {"status": "not trained yet"}
            continue
        left = [m for m in sizes if m >= n and size_trained(run, m) and not eval_done(run, m)]
        share_end = time.time() + max(deadline - time.time(), 0.0) / max(len(left), 1)
        too_short = share_end - time.time() < run.preset["min_start_s"]["evaluate"]
        if not eval_done(run, n) and too_short:
            out["sizes"][str(n)] = {"status": "skipped", "reason": "time budget"}
            continue
        run.log(f"[evaluate {n}] {(share_end - time.time()) / 60:.1f} min for this size")
        res = evaluate_size(run, n, share_end)
        out["sizes"][str(n)] = res
        if "accuracy" in res:
            a = res["accuracy"]["all"]
            rg = res["regret_points"]["mean"]
            run.log(f"[evaluate {n}] best action in {a['value']:.0%} of {res['n_states_evaluated']}"
                    f" states; mean regret {rg['value']:.2f} points "
                    f"(95% {rg['lo95']:.2f} to {rg['hi95']:.2f})")
    if not all(eval_done(run, n) for n in sizes):
        out["status"] = "partial"
        out["reason"] = "not every size is evaluated (time budget or untrained sizes)"
    return out


# ---------------------------------------------------------------- stage 5: package
def scaling_table(evals: dict, trains: dict) -> list[dict]:
    rows = []
    for key in sorted(evals, key=int):
        e, t = evals[key], trains.get(key, {})
        if "accuracy" not in e:
            rows.append({"games": int(key), "status": e.get("status")})
            continue
        acc, rg = e["accuracy"], e["regret_points"]
        hl = e.get("headline") or {}
        rows.append({
            "games": int(key), "status": e.get("status"), "n_states": e["n_states_evaluated"],
            "params": t.get("params"), "epochs_done": t.get("epochs_done"),
            "train_stopped_by": t.get("stopped"), "train_minutes": _min(t.get("used_s")),
            "val_nll_model": e["nll"]["model"]["total"],
            "val_nll_freq_baseline": e["nll"]["baseline_freq_phase_down"]["total_per_transition"],
            "accuracy": acc["all"], "accuracy_margin_over_1": acc["true_margin_over_1_point"],
            "accuracy_margin_over_3": acc["true_margin_over_3_points"],
            "regret_mean_points": rg["mean"], "regret_p90_points": rg["p90"],
            "calibration_mean_error_points": e["calibration"]["mean_error_points"],
            "mean_error_rare_actions_points": e["calibration"]["by_coach_share"][0]["mean_error"],
            "q_star_agreement": e["q_star"]["agreement"],
            "q_star_regret_mean_points": e["q_star"]["regret_points_mean"],
            "coach_policy_regret_mean_points": e["coach_policy"]["regret_points_mean"],
            "se_go_minus_best_kick_mean": e["plan"]["se_go_minus_best_kick"]["mean"],
            "share_states_at_target_se": e["plan"]["se_go_minus_best_kick"][
                "share_at_or_below_target"],
            "p_coach_mode_mean": e["go_mode"]["mean_p_coach_mode"],
            "headline_learned": hl.get("learned"), "headline_true": hl.get("true_q_coach"),
        })  # fmt: skip
    return rows


def _min(s) -> float | None:
    return None if s is None else round(float(s) / 60, 1)


def stage_package(run: GTRun, deadline: float) -> dict:
    st = {s: run.current_output(s) for s in STAGES if s != "package"}
    evals = (st["evaluate"].get("sizes") or {}) if st["evaluate"].get("status") != "missing" \
        else {}
    trains = (st["train"].get("sizes") or {}) if st["train"].get("status") != "missing" else {}
    results: dict[str, Any] = {
        "experiment": "How much data does a learned simulator need to recover the optimal "
                      "fourth-down policy? World model trained on synthetic games from the "
                      "solved game model, graded against that model's true action values",
        "preset": run.preset, "seed": run.seed, "sizes": run.preset["sizes"],
        "stage_hashes": {s: run.stage_hash(s) for s in STAGES},
        "versions": R.versions(), "workdir": str(run.workdir),
        "budget_s": run.total_s, "consumed_s_before_this_session": run.consumed_before,
        "session_seconds": round(time.time() - run.session_start, 1),
        "stages": st,
        "scaling_table": scaling_table(evals, trains),
        "limits": LIMITS,
        "honesty": "Every number here was produced by this run. The generator is the repo's "
                   "own game model, so this measures how learnable OUR simulator is, not real "
                   "football.",
    }  # fmt: skip
    try:
        results["figures"] = make_figures(results, run.workdir / "figures")
    except Exception as ex:  # recorded; the results file is still written
        results["figures"] = {"error": f"{type(ex).__name__}: {ex}"}
    R.write_json(run.workdir / RESULTS_NAME, results)
    zpath = run.workdir / ZIP_NAME
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(run.workdir / RESULTS_NAME, RESULTS_NAME)
        for sub in ("outputs", "figures"):
            for f in sorted((run.workdir / sub).glob("*")):
                z.write(f, f"{sub}/{f.name}")
        for f in sorted((run.workdir / "ground_truth").glob("*.json")):
            z.write(f, f"ground_truth/{f.name}")
        for f in ("run.log", "ledger.json", "preset.json"):
            if (run.workdir / f).exists():
                z.write(run.workdir / f, f)
    run.log(f"[package] results at {run.workdir / RESULTS_NAME}; zip at {zpath}")
    return {"results": str(run.workdir / RESULTS_NAME), "zip": str(zpath),
            "figures": results["figures"]}


LIMITS = [
    "The generator is the repo's own game model (identical league-average teams, its clock and "
    "kicking models), so this measures whether a world model can learn OUR simulator from its "
    "games, not whether it can learn real football.",
    "The headline state's game is spliced: a real synthetic game up to a nearby third down, "
    "then the headline record, then the game played on from it (synthetic.py).",
    "GO is graded in the coach's play mode only (the mode the coach-like policy uses, which "
    "the model's go-set probabilities resolve to); q_star values GO in the best mode.",
    "Actions are masked with the TRUE model's allowed actions (field goal range and punt "
    "rule of the game model), not with rules learned from data.",
    "Every size uses the same architecture and optimizer; schedule length follows each "
    "size's time budget, so the largest sizes may be undertrained in a short session (each "
    "size's stopping reason is recorded).",
]


# ---------------------------------------------------------------- figures
def _size_colors(k: int) -> list[str]:
    import matplotlib.colors as mc

    lo, hi = np.array(mc.to_rgb("#9ec5f0")), np.array(mc.to_rgb("#123d73"))
    return [mc.to_hex(lo + (hi - lo) * (i / max(k - 1, 1))) for i in range(k)]


def make_figures(results: dict, fdir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401  (matplotlib.ticker.NullLocator below)

    R._apply_style(plt)
    s = R._STYLE
    fdir.mkdir(parents=True, exist_ok=True)
    evals = {int(k): v for k, v in (results["stages"]["evaluate"].get("sizes") or {}).items()
             if "accuracy" in v}
    made: dict[str, str] = {}
    if not evals:
        return {"skipped": "no evaluated size yet"}
    sizes = sorted(evals)
    x = np.array(sizes, dtype=float)

    def band(key_fn):
        v = np.array([key_fn(evals[n])["value"] for n in sizes], dtype=float)
        lo = np.array([key_fn(evals[n])["lo95"] for n in sizes], dtype=float)
        hi = np.array([key_fn(evals[n])["hi95"] for n in sizes], dtype=float)
        return v, np.vstack([v - lo, hi - v])

    # 1. regret and accuracy versus number of games
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    v, err = band(lambda e: e["regret_points"]["mean"])
    a1.errorbar(x, v, yerr=err, color=s["BLUE"], marker="o", capsize=3, label="learned planner")
    v90, err90 = band(lambda e: e["regret_points"]["p90"])
    a1.errorbar(x * 1.04, v90, yerr=err90, color=s["BLUE"], marker="o", mfc=s["SURFACE"],
                ls=":", capsize=3, label="learned planner, 90th percentile")  # fmt: skip
    coach = np.mean([evals[n]["coach_policy"]["regret_points_mean"]["value"] for n in sizes])
    a1.axhline(coach, color=s["ORANGE"], ls="--", lw=1.5,
               label=f"coach-like policy itself ({coach:.2f})")  # fmt: skip
    a1.axhline(0, color=s["INK_2"], lw=1)
    a1.set_xlabel("Synthetic training games")
    a1.set_ylabel("Regret (win-probability points)")
    a1.set_title("Regret against the true values")
    a1.legend(fontsize=8.5)
    for key, lab, mk, col in (
            ("all", "all states", "o", s["BLUE"]),
            ("true_margin_over_1_point", "true margin over 1 point", "s", s["VIOLET"]),
            ("true_margin_over_3_points", "true margin over 3 points", "^", s["AQUA"])):
        v, err = band(lambda e, k=key: e["accuracy"][k])
        a2.errorbar(x, 100 * v, yerr=100 * err, marker=mk, capsize=3, label=lab, color=col)
    pc = np.mean([evals[n]["coach_policy"]["p_best_action"]["value"] for n in sizes])
    a2.axhline(100 * pc, color=s["ORANGE"], ls="--", lw=1.5,
               label=f"coach-like policy P(best) ({100 * pc:.0f}%)")  # fmt: skip
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n:,}" for n in sizes])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a2.set_ylim(0, 102)
    a2.set_xlabel("Synthetic training games")
    a2.set_ylabel("Picks the truly best action (%)")
    a2.set_title("Decision accuracy")
    a2.legend(fontsize=8.5, loc="lower right")
    n_st = ", ".join(f"{n:,}: {evals[n]['n_states_evaluated']}" for n in sizes)
    fig.text(0.01, -0.04, f"Intervals: 95% bootstrap over states. States evaluated per size "
             f"({n_st}). True values by Monte Carlo in the exact game model.",
             fontsize=8.5, color=s["MUTED"])  # fmt: skip
    made["regret_accuracy"] = R._save(fig, fdir / "gt_regret_accuracy.png", plt)

    # 2. learned versus true Q, per size
    k = len(sizes)
    fig, axes = plt.subplots(1, k, figsize=(3.4 * k + 0.6, 3.6), squeeze=False)
    for ax, n in zip(axes[0], sizes, strict=True):
        ps = evals[n]["per_state"]
        wp = np.array(ps["wp"], dtype=float)
        q = np.array(ps["q_coach"], dtype=float)
        for j, c in enumerate(CHOICES):
            ok = np.isfinite(wp[:, j]) & np.isfinite(q[:, j])
            ax.scatter(q[ok, j], wp[ok, j], s=9, alpha=0.65, color=R._ACTION_COLORS[c],
                       label=c, linewidths=0)  # fmt: skip
        ax.plot([0, 1], [0, 1], color=s["INK_2"], lw=1)
        me = evals[n]["calibration"]["mean_error_points"]["value"]
        ax.set_title(f"{n:,} games")
        ax.text(0.03, 0.93, f"mean error {me:+.2f} pts", transform=ax.transAxes, fontsize=8.5,
                color=s["INK_2"])  # fmt: skip
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("True value")
    axes[0][0].set_ylabel("Learned value (world model)")
    axes[0][0].legend(fontsize=8, loc="lower right")
    made["q_scatter"] = R._save(fig, fdir / "gt_q_scatter.png", plt)

    # 3. overestimation by coverage
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cols = _size_colors(k)
    xb = np.arange(len(COVERAGE_LABELS))
    for i, n in enumerate(sizes):
        cov = evals[n]["calibration"]["by_coach_share"]
        v = np.array([c["mean_error"] if c["mean_error"] is not None else np.nan for c in cov])
        lo = np.array([c["lo95"] if c["lo95"] is not None else np.nan for c in cov])
        hi = np.array([c["hi95"] if c["hi95"] is not None else np.nan for c in cov])
        off = (i - (k - 1) / 2) * 0.08
        ax.errorbar(xb + off, v, yerr=np.vstack([v - lo, hi - v]), color=cols[i], marker="o",
                    capsize=3, label=f"{n:,} games")  # fmt: skip
    ax.axhline(0, color=s["INK_2"], lw=1)
    ax.set_xticks(xb)
    ax.set_xticklabels(COVERAGE_LABELS)
    ax.set_xlabel("How often the coach-like policy takes the action in that state")
    ax.set_ylabel("Learned minus true value (points)")
    ax.set_title("Over- or underestimation by how often the action is logged")
    ax.legend(fontsize=8.5)
    made["coverage"] = R._save(fig, fdir / "gt_overestimation_by_coverage.png", plt)
    return made


# ---------------------------------------------------------------- estimates and preflight
def step_estimate(gt: dict, ev, preset: dict) -> dict:
    """Rollout steps the evaluation of ONE size needs: phase A (n0 per allowed action) and up
    to the target standard error, using p(1 - p) of the true values as the per-rollout
    variance (ties make the real variance a little smaller) and the records left in each
    state's game as the steps per rollout."""
    e = preset["eval"]
    order = state_order(ev.kind, int(e["n_states"]))
    ends = ev.split.offsets[np.searchsorted(ev.split.offsets, ev.idx, side="right")]
    steps = (ends - ev.idx)[order].astype(float)
    allowed = np.asarray(gt["allowed_model"][order][:, 1:4], bool)
    q = np.asarray(gt["q_coach"][order][:, 1:4], float)
    var = np.where(allowed, q * (1 - q), np.nan)
    n0 = int(e["n0"])
    qa, se0 = np.where(allowed, q, np.nan), np.sqrt(var / n0)
    _, se_diff = _go_kick(qa, se0, allowed)
    total = rollouts_needed(se_diff, n0, float(e["target_se"]), int(e["n_max"]))
    na = allowed.sum(axis=1)
    nr = refine_mask(qa, se0, allowed).sum(axis=1)
    return {"phase_a_steps": float((na * n0 * steps).sum()),
            "to_target_steps": float(((na * n0 + nr * (total - n0)) * steps).sum()),
            "mean_steps_per_rollout": float(steps.mean())}  # fmt: skip


def estimate_minutes(preset: dict, gt: dict | None = None, ev=None,
                     records_per_game: float = 157.3) -> dict:  # fmt: skip
    """Minutes per size at the benchmark's measured speeds (or the preset's assumed ones)."""
    if gt is None:
        gt, _ = Y.load_ground_truth()
        ev = Y.load_eval_states()
    b = preset.get("benchmark") or {}
    rec_s = float(b.get("train_records_per_s") or preset["assumed"]["train_rec_per_s"])
    steps_s = float(b.get("rollout_steps_per_s") or preset["assumed"]["rollout_units_per_s"])
    st = step_estimate(gt, ev, preset)
    ep = int(preset["train"]["epoch_games"])
    rows = {}
    for n in preset["sizes"]:
        rows[str(n)] = {
            "train_minutes_per_epoch": round(max(n, ep) * records_per_game / rec_s / 60, 2),
            "eval_minutes_phase_a": round(st["phase_a_steps"] / steps_s / 60, 1),
            "eval_minutes_to_target_se": round(st["to_target_steps"] / steps_s / 60, 1),
        }
    return {"speeds": {"train_records_per_s": rec_s, "rollout_steps_per_s": steps_s,
                       "basis": "benchmark" if b.get("rollout_steps_per_s") else "assumed"},
            "per_size": rows, "steps": st}  # fmt: skip


def prepare(preset_name: str | None = None, sizes=None, workdir: Path | None = None,
            budget_min: float | None = None, game_dir: str | Path | None = None,
            force: tuple = (), benchmark: bool = True, cache_game: bool | None = None,
            overrides: dict | None = None, quiet: bool = False) -> GTRun:  # fmt: skip
    """Pick (or reload) the preset, benchmark once, print the status, return a GTRun. The
    preset is saved in the work folder and reused on every resume, so the stage hashes stay
    the same; SIZES and BUDGET_MIN may change between sessions (per-size work is kept)."""
    wd = Path(workdir or default_workdir())
    wd.mkdir(parents=True, exist_ok=True)
    saved = wd / "preset.json"
    name = preset_name or R.pick_preset()
    preset, notes = None, []
    if saved.exists():
        old = R.read_json(saved)
        if old.get("definition") == gt_definition(name) and old.get("overrides") == (overrides
                                                                                     or {}):
            preset = old
            notes.append(f"reusing the saved preset '{name}' from {saved}")
        else:
            dest = R.archive_previous_run(wd, f"new preset '{name}' (saved one differs)")
            notes.append(f"the saved preset differs; earlier run files moved to {dest}")
    if preset is None:
        preset = gt_preset(name, **(overrides or {}))
        preset["definition"] = gt_definition(name)
        preset["overrides"] = overrides or {}
        if benchmark:
            try:
                bench = R.micro_benchmark(R.get_preset(name), seconds=preset["benchmark_s"])
                preset["benchmark"] = {k: bench[k] for k in (
                    "device", "train_records_per_s", "train_speed_basis", "rollout_steps_per_s",
                    "rollout_speed_basis", "rollout_batch", "assumed")}  # fmt: skip
                notes.append(f"benchmark: {bench['train_records_per_s']:,.0f} training records/s"
                             f", {bench['rollout_steps_per_s']:,.0f} rollout steps/s")
            except Exception as ex:
                preset["benchmark"] = {"error": f"{type(ex).__name__}: {ex}"}
                notes.append(f"benchmark failed ({ex}); planning with the assumed speeds")
    if sizes is not None:
        preset["sizes"] = sorted(int(s) for s in sizes)
    R.write_json(saved, preset)
    run = GTRun(preset, wd, budget_min=budget_min, force=force, game_dir=game_dir,
                cache_game=cache_game, quiet=quiet)  # fmt: skip
    for n_ in notes:
        run.log(f"[preflight] {n_}")
    try:
        est = estimate_minutes(preset)
        run.log("[preflight] estimated minutes per size at "
                f"{est['speeds']['rollout_steps_per_s']:,.0f} rollout steps/s and "
                f"{est['speeds']['train_records_per_s']:,.0f} training records/s "
                f"({est['speeds']['basis']}): {json.dumps(est['per_size'])}")
    except Exception as ex:
        run.log(f"[preflight] no time estimate: {ex}")
    print(status_table(run), flush=True)
    return run


def status_table(run: GTRun) -> str:
    lines = [f"Work folder {run.workdir}", f"{'stage':<9} state"]
    for s in STAGES[:-1]:
        state = "done" if run.is_done(s) else run.status(s)
        lines.append(f"{s:<9} {'not started' if state == 'missing' else state}")
    if run.content_hash(TRAIN_FILE) != "missing":
        for n in run.preset["sizes"]:
            st = train_state(run, n)
            lines.append(f"  size {n:>6,}: training "
                         + ("done" if size_trained(run, n) else
                            f"{st.get('used_s', 0) / 60:.1f} min used" if st else "not started")
                         + ("; evaluated" if eval_done(run, n) else ""))
    lines.append(f"minutes used so far {run.consumed_before / 60:.1f} of {run.total_s / 60:.0f}; "
                 f"{run.remaining() / 60:.1f} left this session")
    return "\n".join(lines)


STAGE_FUNCS = {"setup": stage_setup, "generate": stage_generate, "train": stage_train,
               "evaluate": stage_evaluate, "package": stage_package}  # fmt: skip


def _beyond(stage: str, stop_after: str | None) -> bool:
    if stop_after is None or stage == "package":
        return False
    if stop_after not in STAGES:
        raise ValueError(f"STOP_AFTER must be None or one of {STAGES[:-1]}")
    return STAGES.index(stage) > STAGES.index(stop_after)


def run_stage(run: GTRun, stage: str) -> dict:
    return R.run_stage(run, stage, STAGE_FUNCS[stage])


def notebook_stage(run: GTRun, stage: str, stop_after: str | None = None) -> dict:
    """One notebook cell. Honors STOP_AFTER; on the stop button, packages what exists and
    re-raises so Run all stops there."""
    if _beyond(stage, stop_after):
        msg = f"[{stage}] not run: STOP_AFTER = {stop_after!r}"
        print(msg)
        return {"status": "not_run", "reason": msg}
    try:
        return run_stage(run, stage)
    except KeyboardInterrupt:
        try:
            run_stage(run, "package")
        except BaseException as ex:
            run.log(f"[package] could not package after the stop: {ex}")
        print(R.stopped_message(stage), flush=True)
        raise


def run_all(run: GTRun, stop_after: str | None = None) -> dict:
    out: dict[str, Any] = {}
    for s in STAGES:
        if s != "package" and not _beyond(s, stop_after):
            out[s] = run_stage(run, s)
    out["package"] = run_stage(run, "package")
    return out


def summary(run: GTRun) -> str:
    """Plain-language summary of gt_results.json."""
    path = run.workdir / RESULTS_NAME
    if not path.exists():
        return "No results yet."
    r = R.read_json(path)
    st = r["stages"]
    lines = [f"Preset {r['preset']['name']} on {r['versions'].get('gpu', 'CPU')}; sizes "
             f"{', '.join(f'{n:,}' for n in r['sizes'])} games."]
    for s in ("setup", "generate", "train", "evaluate"):
        o = st.get(s, {})
        lines.append(f"- {s}: {o.get('status')}"
                     + (f" ({o.get('reason') or o.get('error')})" if o.get("status") != "done"
                        else f" in {o.get('seconds')}s"))
    su = st.get("setup", {})
    if su.get("status") == "done":
        g = su["ground_truth"]
        lines.append(f"  exact game: {su['source']}; ground truth {g['name']}"
                     + (" (recomputed here: the saved one did not match this game)"
                        if g["recomputed"] else " (verified against this game)"))
        if su.get("build"):
            b = su["build"]
            lines.append(f"  built from public data in {b['total_seconds'] / 60:.1f} min, peak "
                         f"memory {b['peak_rss_gb_process']} GB")
    rows = [x for x in r["scaling_table"] if "accuracy" in x]
    if rows:
        c = rows[0]["coach_policy_regret_mean_points"]["value"]
        lines.append(f"  the coach-like policy itself loses {c:.2f} points of win probability "
                     "per decision on average against the best action")
    for x in r["scaling_table"]:
        if "accuracy" not in x:
            lines.append(f"  {x['games']:,} games: {x.get('status')}")
            continue
        a, rg = x["accuracy"], x["regret_mean_points"]
        lines.append(
            f"  {x['games']:,} games ({x['n_states']} states): picks the truly best action in "
            f"{a['value']:.0%} (95% {a['lo95']:.0%} to {a['hi95']:.0%}); "
            f"{x['accuracy_margin_over_3']['value']:.0%} where the true margin is over 3 points;"
            f" mean regret {rg['value']:.2f} points (95% {rg['lo95']:.2f} to {rg['hi95']:.2f}), "
            f"90th percentile {x['regret_p90_points']['value']:.2f}; learned values off by "
            f"{x['calibration_mean_error_points']['value']:+.2f} points on average; NLL "
            f"{x['val_nll_model']:.3f} (frequency baseline {x['val_nll_freq_baseline']:.3f})")
        hl, ht = x.get("headline_learned"), x.get("headline_true")
        if hl and ht:
            lines.append("    headline state (up 4, 4:00, 4th and 2 at own 40): learned "
                         + ", ".join(f"{k} {_fmt(hl[k])}" for k in CHOICES)
                         + "; true " + ", ".join(f"{k} {_fmt(ht[k])}" for k in CHOICES))
    return "\n".join(lines)


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=sorted(R.PRESETS), default=None)
    ap.add_argument("--sizes", type=int, nargs="*", default=None)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--budget-min", type=float, default=None)
    ap.add_argument("--game-dir", type=Path, default=None)
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument("--stop-after", choices=STAGES[:-1], default=None)
    ap.add_argument("--force", nargs="*", default=[], choices=list(STAGES[:-1]))
    a = ap.parse_args(argv)
    run = prepare(a.preset, a.sizes, a.workdir, a.budget_min, a.game_dir, tuple(a.force),
                  benchmark=not a.no_benchmark)  # fmt: skip
    run_all(run, a.stop_after)
    print(summary(run))


if __name__ == "__main__":
    sys.exit(main())
