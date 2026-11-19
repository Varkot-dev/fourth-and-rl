"""Runner for the NFL world-model experiment, built to finish inside one Colab GPU session.

Six stages, each idempotent. A finished stage writes a done-marker holding its config hash,
so re-running everything skips finished work and training resumes from its checkpoint:

  1 data     download play-by-play, build and validate train/validation/test splits
  2 train    fit the world model (checkpointed, deadline-bound)
  3 eval     validation only: per-field NLL vs a frequency baseline, simulated league
             stats vs real, win probability by simulation vs nflverse on the same states
  4 test     write prereg/<config hash>.json (this configuration's exact plan) FIRST, then
             score the test seasons once per configuration; a second scoring of the same
             configuration is refused unless allow_rescore is set
  5 plan     fourth-down recommendations on sampled test fourth downs (under the same
             preregistration), one constructed hypothetical state, and agreement with the
             real coaches
  6 package  results.json (every number), figures, a zip to download

A global deadline guard gives every stage a deadline; stages scale down (fewer states or
rollouts) or are skipped with a recorded reason, and packaging always runs, so a run
always ends with a results file. Numbers from nflverse (`home_wp`, `vegas_home_wp`) are
kept in their own labelled fields and never blended with ours.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import platform
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import schema as S

STEPS_PER_ROLLOUT = 90  # mean transitions from a random state to the end of regulation
MAX_SEASON = 2025  # never download a season after this one (2026 is in progress)
SEED = 20261118
STAGES = ("data", "train", "eval", "test", "plan", "package")
UPSTREAM = {
    "data": (),
    "train": ("data",),
    "eval": ("train",),
    "test": ("train",),
    "plan": ("train",),
    "package": (),
}
# Module sources recorded per stage (metadata and warnings only; hashes never use them,
# except that a changed records/schema makes the data stage rebuild in memory and compare).
STAGE_SOURCES = {
    "data": ["schema", "records"],
    "train": ["model", "train"],
    "eval": ["simulate"],
    "test": ["simulate"],
    "plan": ["simulate", "support"],
    "package": [],
}
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- presets
_FULL_SEASONS = {
    "train": list(range(2010, 2022)),
    "val": [2022, 2023],
    "test": [2024, 2025],
}


def _gpu_preset(name: str, model: dict, train: dict, eval_: dict, plan: dict, assumed: dict):
    return {
        "name": name,
        "device": "cuda",
        "seasons": copy.deepcopy(_FULL_SEASONS),
        "model": model,
        "train": train,
        "eval": eval_,
        "test": dict(eval_),
        "plan": plan,
        "budget_min": 55.0,
        # Per-stage ceilings in minutes. They sum to more than the budget on purpose: a
        # stage that finishes early hands its time on; the global deadline is the hard cap.
        # Plan is the last compute stage, so its ceiling is high: it may use whatever the
        # earlier stages left (training on the GPU usually takes well under its 20 minutes).
        "stage_min": {"data": 8, "train": 20, "eval": 9, "test": 7, "plan": 20, "package": 2},
        # Minimum seconds a stage needs to be worth starting; below it, it is skipped.
        "min_start_s": {"data": 60, "train": 60, "eval": 45, "test": 45, "plan": 45,
                        "package": 0},
        "reserve_package_s": 90,
        "assumed": assumed,
    }


PRESETS: dict[str, dict] = {
    "smoke": {
        "name": "smoke",
        "device": "cpu",
        "seasons": {"train": [2021], "val": [2022], "test": [2023]},
        "model": {"d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 256, "dec_hidden": 64,
                  "max_len": 256, "dropout": 0.1},
        "train": {"batch_games": 16, "lr": 1e-3, "weight_decay": 0.01, "epochs": 4,
                  "patience": 2, "warmup_steps": 50, "max_train_games": None,
                  "sim_batch": 256, "sim_max_live": 1024, "score_on_end": True},
        "eval": {"wp_states": 16, "wp_rollouts": 16, "sim_games": 4, "bootstrap": 200,
                 "nll_max_games": 60},
        "test": {"wp_states": 16, "wp_rollouts": 16, "sim_games": 0, "bootstrap": 200,
                 "nll_max_games": 60},
        "plan": {"n_fourth": 8, "n_rollouts": 16, "hypo_rollouts": 64,
                 "support": {"diag_states_per_band": 4, "diag_draws": 64}},

        "budget_min": 12.0,
        "stage_min": {"data": 5, "train": 2.5, "eval": 2, "test": 1.5, "plan": 1.5,
                      "package": 1},
        "min_start_s": {"data": 20, "train": 10, "eval": 10, "test": 10, "plan": 10,
                        "package": 0},
        "reserve_package_s": 20,
        "assumed": {"train_rec_per_s": 3_000, "rollout_units_per_s": 2_000},
    },
    "t4": _gpu_preset(
        "t4",
        model={"d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 1024, "dec_hidden": 256,
               "max_len": 256, "dropout": 0.1},
        train={"batch_games": 64, "lr": 8.5e-4, "weight_decay": 0.1, "epochs": 40,
               "patience": 4, "warmup_steps": 300, "max_train_games": None,
               "sim_batch": 1024, "sim_max_live": 8192, "score_on_end": True},
        eval_={"wp_states": 600, "wp_rollouts": 256, "sim_games": 128, "bootstrap": 1000,
               "nll_max_games": None},
        plan={"n_fourth": 300, "n_rollouts": 512, "hypo_rollouts": 4096},
        assumed={"train_rec_per_s": 60_000, "rollout_units_per_s": 40_000},
    ),
    "l4": _gpu_preset(
        "l4",
        model={"d_model": 320, "n_layers": 6, "n_heads": 8, "d_ff": 1280, "dec_hidden": 320,
               "max_len": 256, "dropout": 0.1},
        train={"batch_games": 64, "lr": 7e-4, "weight_decay": 0.1, "epochs": 50,
               "patience": 4, "warmup_steps": 300, "max_train_games": None,
               "sim_batch": 2048, "sim_max_live": 16384, "score_on_end": True},
        eval_={"wp_states": 1000, "wp_rollouts": 384, "sim_games": 256, "bootstrap": 1000,
               "nll_max_games": None},
        plan={"n_fourth": 400, "n_rollouts": 512, "hypo_rollouts": 8192},
        assumed={"train_rec_per_s": 120_000, "rollout_units_per_s": 80_000},
    ),
    "a100": _gpu_preset(
        "a100",
        model={"d_model": 384, "n_layers": 8, "n_heads": 8, "d_ff": 1536, "dec_hidden": 384,
               "max_len": 256, "dropout": 0.1},
        train={"batch_games": 128, "lr": 7e-4, "weight_decay": 0.1, "epochs": 60,
               "patience": 5, "warmup_steps": 300, "max_train_games": None,
               "sim_batch": 4096, "sim_max_live": 32768, "score_on_end": True},

        eval_={"wp_states": 1500, "wp_rollouts": 512, "sim_games": 512, "bootstrap": 1000,
               "nll_max_games": None},
        plan={"n_fourth": 600, "n_rollouts": 1024, "hypo_rollouts": 16384},
        assumed={"train_rec_per_s": 300_000, "rollout_units_per_s": 200_000},
    ),
}
DOWNGRADE = {"a100": "l4", "l4": "t4", "t4": "t4", "smoke": "smoke"}
# Bump when what a preset means changes without its values changing (for example a new rule
# in how the benchmark adjusts it). preset.json stores this with a hash of the PRESETS entry;
# a saved preset whose definition differs from the code is replaced by a fresh pick.
PRESET_VERSION = "2 (2026-11-18: GPU training, KV-cache rollouts, action support)"
# The preflight benchmark scales a preset down only when the measured rollout or training
# speed is below this share of the preset's assumed speed, and never below these floors
# (or the preset's own value, if that is smaller).
BENCH_SLOW_SHARE = 0.5
BENCH_FLOORS = {"n_fourth": 200, "wp_states": 300, "sim_games": 32, "hypo_rollouts": 1024}


def preset_definition(name: str) -> dict:
    """What identifies a preset's definition: its name, PRESET_VERSION and a hash of its
    PRESETS entry (before any benchmark adjustment)."""
    return {"name": name, "version": PRESET_VERSION, "hash": config_hash(PRESETS[name])}


def pick_preset(device_name: str | None = None, mem_gb: float | None = None) -> str:
    """Preset key from the CUDA device name and memory; "smoke" when there is no GPU."""
    if device_name is None:
        try:
            import torch

            if not torch.cuda.is_available():
                return "smoke"
            props = torch.cuda.get_device_properties(0)
            device_name, mem_gb = props.name, props.total_memory / 2**30
        except Exception:
            return "smoke"
    name = device_name.upper()
    mem = mem_gb or 0.0
    if "A100" in name or "H100" in name or "H200" in name or "B200" in name:
        return "a100"
    if "L4" in name or "A10" in name or "L40" in name or "A6000" in name:
        return "l4"
    if "T4" in name or "P100" in name or "V100" in name:
        return "t4"
    if mem >= 38:
        return "a100"
    if mem >= 20:
        return "l4"
    return "t4"


def get_preset(name: str, **overrides: Any) -> dict:
    p = copy.deepcopy(PRESETS[name])
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k].update(v)
        else:
            p[k] = v
    return p


def check_seasons(seasons) -> list[int]:
    seasons = [int(s) for s in seasons]
    assert seasons, "no seasons given"
    assert max(seasons) <= MAX_SEASON, f"refusing to download a season after {MAX_SEASON}"
    assert min(seasons) >= 1999, "nflverse play-by-play starts in 1999"
    return seasons


# ---------------------------------------------------------------- storage
def default_workdir() -> Path:
    if os.environ.get("WM_WORKDIR"):
        return Path(os.environ["WM_WORKDIR"])
    drive = Path("/content/drive/MyDrive")
    if drive.is_dir():
        return drive / "fourth_and_rl_world_model"
    if Path("/content").is_dir():
        return Path("/content/wm_run")
    return HERE.parent.parent / "results" / "world_model_local"


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if hasattr(x, "to_dict") and callable(x.to_dict):  # pandas objects
        try:
            return _jsonable(x.to_dict(orient="list"))
        except TypeError:
            return _jsonable(x.to_dict())
    if isinstance(x, Path):
        return str(x)
    if x is None or isinstance(x, (str, int, bool)):
        return x
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2))
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def config_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(obj), sort_keys=True).encode()).hexdigest()[:16]


def _source_hash(module_names) -> str:
    h = hashlib.sha256()
    for m in module_names:
        p = HERE / f"{m}.py"
        mod = sys.modules.get(m)
        if mod is not None and getattr(mod, "__file__", None):
            p = Path(mod.__file__)
        h.update(p.read_bytes() if p.exists() else m.encode())
    return h.hexdigest()[:16]


def _mod(name: str):
    """Import a sibling module lazily, so tests can put stubs in sys.modules."""
    return sys.modules.get(name) or importlib.import_module(name)


# ---------------------------------------------------------------- split utilities
_SPLIT_ARRAYS = ("states", "raw", "targets", "offsets", "header", "final_diff", "seasons",
                 "play_ids")


def save_split(path: Path, split: S.Split, **extra: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(getattr(split, k)) for k in _SPLIT_ARRAYS}
    arrays["game_ids"] = np.asarray(list(split.game_ids), dtype=str)
    arrays.update({f"extra_{k}": np.asarray(v) for k, v in extra.items()})
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_split(path: Path) -> tuple[S.Split, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        kw = {k: z[k] for k in _SPLIT_ARRAYS}
        kw["game_ids"] = [str(g) for g in z["game_ids"]]
        extra = {k[6:]: z[k] for k in z.files if k.startswith("extra_")}
    return S.Split(**kw), extra


def concat_splits(splits: list[S.Split]) -> S.Split:
    offs, base = [np.zeros(1, dtype=np.int64)], 0
    for s in splits:
        offs.append(np.asarray(s.offsets[1:], dtype=np.int64) + base)
        base += int(s.offsets[-1])
    return S.Split(
        states=np.concatenate([s.states for s in splits]),
        raw=np.concatenate([s.raw for s in splits]),
        targets=np.concatenate([s.targets for s in splits]),
        offsets=np.concatenate(offs),
        header=np.concatenate([s.header for s in splits]),
        final_diff=np.concatenate([s.final_diff for s in splits]),
        game_ids=[g for s in splits for g in s.game_ids],
        seasons=np.concatenate([s.seasons for s in splits]),
        play_ids=np.concatenate([s.play_ids for s in splits]),
    )


def subset_split(split: S.Split, games) -> S.Split:
    """A split holding only the listed games (in the given order)."""
    games = np.asarray(games, dtype=np.int64)
    rec = [np.arange(split.offsets[g], split.offsets[g + 1]) for g in games]
    lens = np.array([len(r) for r in rec], dtype=np.int64)
    idx = np.concatenate(rec) if rec else np.zeros(0, dtype=np.int64)
    return S.Split(
        states=split.states[idx],
        raw=split.raw[idx],
        targets=split.targets[idx],
        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        header=split.header[games],
        final_diff=split.final_diff[games],
        game_ids=[split.game_ids[g] for g in games],
        seasons=split.seasons[games],
        play_ids=split.play_ids[idx],
    )


def record_game(split: S.Split) -> np.ndarray:
    """Game index of every record."""
    return np.repeat(np.arange(len(split.game_ids)), np.diff(split.offsets))


def n_records(split: S.Split) -> int:
    return int(split.offsets[-1])


def synthetic_split(n_games: int = 8, seed: int = 0) -> S.Split:
    """A schema-valid toy split (drives in which the clock runs down), for tests and the
    micro-benchmark only. It is not football data."""
    rng = np.random.default_rng(seed)
    states, raws, targets, offsets, final = [], [], [], [0], []
    for _g in range(n_games):
        clock, diff, poss = S.GAME_SECONDS, 0, 1
        rows, raw = [], []
        down, togo, yl = 0, 0, 65
        phase = S.PHASE["KICKOFF"]
        while clock > 0:
            act = S.ACTION["KICKOFF"] if phase == S.PHASE["KICKOFF"] else int(
                rng.choice([S.ACTION["RUN"], S.ACTION["PASS"]]))
            if phase == S.PHASE["PLAY"] and down == 4:
                act = S.ACTION["PUNT"]
            rows.append([phase, poss, down, togo, yl, int(S.clock_code(clock)),
                         int(S.score_code(diff)), 4, 4, act])
            raw.append([clock, diff, 3, 3])
            clock = max(clock - int(rng.integers(5, 45)), 0)
            if phase == S.PHASE["KICKOFF"] or act == S.ACTION["PUNT"]:
                poss = 3 - poss
                phase, down, togo, yl = S.PHASE["PLAY"], 1, 10, int(rng.integers(50, 80))
            else:
                yl = max(1, yl - int(rng.integers(0, 8)))
                down = down + 1 if down < 4 else 1
                togo = 10
                if yl <= 2:
                    diff += 7 if poss == 1 else -7
                    phase, down, togo, yl = S.PHASE["KICKOFF"], 0, 0, 65
                    poss = 3 - poss
        states.extend(rows)
        raws.extend(raw)
        tg = np.zeros((len(rows), len(S.TARGET_FIELDS)), dtype=np.int16)
        r = np.asarray(rows)
        for name in ("phase", "poss", "down", "ydstogo", "yardline", "action"):
            tg[:-1, S.TARGET_INDEX[name]] = r[1:, S.STATE_INDEX[name]]
        tg[:-1, S.TARGET_INDEX["score_delta"]] = [
            S.SCORE_DELTA_CODE.get(int(b[1] - a[1]), S.SCORE_DELTA_CODE[0])
            for a, b in zip(raw[:-1], raw[1:], strict=True)]
        tg[:-1, S.TARGET_INDEX["elapsed"]] = S.elapsed_code(
            np.asarray(raw)[:-1, 0] - np.asarray(raw)[1:, 0])
        tg[:-1, S.TARGET_INDEX["to_used_home"]] = 1
        tg[:-1, S.TARGET_INDEX["to_used_away"]] = 1
        tg[-1, S.TARGET_INDEX["phase"]] = S.PHASE["END"]
        targets.append(tg)
        offsets.append(offsets[-1] + len(rows))
        final.append(diff)
    n = offsets[-1]
    return S.Split(
        states=np.asarray(states, dtype=np.int16),
        raw=np.asarray(raws, dtype=np.int32),
        targets=np.concatenate(targets),
        offsets=np.asarray(offsets, dtype=np.int64),
        header=np.tile(np.array([[S.SPREAD_CLIP + 1, 1]], dtype=np.int16), (n_games, 1)),
        final_diff=np.asarray(final, dtype=np.int16),
        game_ids=[f"SYN_{g:04d}" for g in range(n_games)],
        seasons=np.full(n_games, 2000, dtype=np.int16),
        play_ids=np.arange(n, dtype=np.int64),
    )


# ---------------------------------------------------------------- run context
class Run:
    """Everything a stage needs: preset, folders, the deadline guard and the log.

    Stage hashes are keyed on CONFIG and DATA CONTENT, never on source code: the train
    checkpoint and done-marker depend on the model and train config plus a content hash of
    the saved train and validation arrays, so an edited notebook resumes training as long as
    the data and config are the same. Module source hashes are recorded in each done-marker
    as metadata and produce a warning when they change.
    """

    def __init__(self, preset: dict, workdir: Path | None = None,
                 budget_min: float | None = None, allow_rescore: bool = False,
                 seed: int = SEED, quiet: bool = False, force: tuple = ()) -> None:
        self.preset = preset
        self.workdir = Path(workdir or default_workdir())
        self.workdir.mkdir(parents=True, exist_ok=True)
        for sub in ("data", "stages", "outputs", "figures", "ckpt", "partial"):
            (self.workdir / sub).mkdir(exist_ok=True)
        self.allow_rescore = allow_rescore
        self.force = set(force)
        self.seed = seed
        self.quiet = quiet
        self.session_start = time.time()
        self.ledger_path = self.workdir / "ledger.json"
        ledger = read_json(self.ledger_path) if self.ledger_path.exists() else {}
        self.consumed_before = float(ledger.get("consumed_s", 0.0))
        total = 60.0 * float(budget_min if budget_min is not None else preset["budget_min"])
        self.total_s = total
        left = total - self.consumed_before
        # Never less than the packaging reserve: packaging must always be able to run.
        self.deadline = self.session_start + max(left, preset["reserve_package_s"] + 5)
        self._models: dict[str, Any] = {}

    # time ------------------------------------------------------------
    def remaining(self) -> float:
        return self.deadline - time.time()

    def stage_deadline(self, stage: str) -> float:
        ceiling = time.time() + 60.0 * self.preset["stage_min"][stage]
        reserve = 0.0 if stage == "package" else float(self.preset["reserve_package_s"])
        return min(ceiling, self.deadline - reserve)

    def add_consumed(self, seconds: float, stage: str) -> None:
        ledger = read_json(self.ledger_path) if self.ledger_path.exists() else {}
        ledger["consumed_s"] = float(ledger.get("consumed_s", 0.0)) + float(seconds)
        ledger.setdefault("stages", []).append(
            {"stage": stage, "seconds": round(float(seconds), 1), "at": time.time()})
        write_json(self.ledger_path, ledger)

    def log(self, *msg: Any) -> None:
        line = " ".join(str(m) for m in msg)
        with open(self.workdir / "run.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {line}\n")
        if not self.quiet:
            print(line, flush=True)

    # config ----------------------------------------------------------
    def model_cfg(self) -> dict:
        """The cfg handed to model, train and simulate."""
        p = self.preset
        cfg = {**p["model"], **p["train"], "device": self.device(), "seed": self.seed}
        cfg.pop("max_train_games", None)
        return cfg

    def device(self) -> str:
        if self.preset["device"] == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
        return "cpu"

    def content_hash(self, name: str) -> str:
        """Content hash of data/<name>.npz, cached by file size and mtime."""
        path = self.workdir / "data" / f"{name}.npz"
        if not path.exists():
            return "missing"
        cache_path = self.workdir / "data" / "content_hashes.json"
        cache = read_json(cache_path) if cache_path.exists() else {}
        st = path.stat()
        key = f"{st.st_size}:{st.st_mtime_ns}"
        if cache.get(name, {}).get("key") == key:
            return cache[name]["hash"]
        sp, extra = load_split(path)
        h = split_arrays_hash(sp, extra)
        cache[name] = {"key": key, "hash": h}
        write_json(cache_path, cache)
        return h

    def stage_src(self, stage: str) -> str:
        return _source_hash(STAGE_SOURCES[stage])

    def stage_hash(self, stage: str) -> str:
        p = self.preset
        parts: dict[str, Any] = {"stage": stage, "seed": self.seed}
        if stage == "data":
            parts["seasons"] = p["seasons"]
        elif stage == "train":
            parts.update(model=p["model"], train=p["train"],
                         data_train=self.content_hash("train"),
                         data_val=self.content_hash("val"))
        elif stage in ("eval", "test", "plan"):
            parts[stage] = p[stage]
            parts["up_train"] = self.stage_hash("train")
            if stage == "plan":
                # Defaults included, so a resumed run with an older saved preset still
                # recomputes the plan with the support rules.
                parts["support"] = _mod("support").config(p["plan"])
            if stage in ("test", "plan"):
                parts["data_test"] = self.content_hash("test")
        return config_hash(parts)

    def full_config_hash(self) -> str:
        return config_hash({k: self.stage_hash(k) for k in STAGES if k != "package"})

    # files -------------------------------------------------------------
    def marker(self, stage: str) -> Path:
        return self.workdir / "stages" / f"{stage}.done"

    def output_path(self, stage: str) -> Path:
        return self.workdir / "outputs" / f"{stage}.json"

    def partial_dir(self, stage: str) -> Path:
        d = self.workdir / "partial" / f"{stage}_{self.stage_hash(stage)}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def is_done(self, stage: str) -> bool:
        if stage in self.force:
            return False
        m = self.marker(stage)
        if not m.exists():
            return False
        mk = read_json(m)
        if mk.get("hash") != self.stage_hash(stage):
            return False
        # Data is the one stage whose code is re-checked: a changed records.py is rebuilt
        # in memory and compared (see stage_data).
        return not (stage == "data" and mk.get("src") != self.stage_src("data"))

    def status(self, stage: str) -> str:
        """Status of this configuration's output of a stage; "missing" when there is none
        and "other config" when the file was written under a different stage hash."""
        p = self.output_path(stage)
        if not p.exists():
            return "missing"
        out = read_json(p)
        if stage != "package" and out.get("hash") != self.stage_hash(stage):
            return "other config"
        return out.get("status", "missing")

    def current_output(self, stage: str) -> dict:
        """The stage's output file if it belongs to this configuration; otherwise a stub
        saying why not (an output from an earlier configuration is never reported as
        this run's)."""
        p = self.output_path(stage)
        if not p.exists():
            return {"status": "missing"}
        out = read_json(p)
        h = self.stage_hash(stage)
        if out.get("hash") != h:
            return {"status": "missing",
                    "reason": f"the saved output belongs to another configuration (stage hash "
                              f"{out.get('hash')}, this run {h}); not reported"}
        return out

    def split(self, name: str) -> tuple[S.Split, dict]:
        return load_split(self.workdir / "data" / f"{name}.npz")

    def ckpt_path(self) -> Path:
        return self.workdir / "ckpt" / f"model_{self.stage_hash('train')}.pt"

    def load_model(self):
        key = self.stage_hash("train")
        if key in self._models:
            return self._models[key]
        model, _ck = _mod("train").load_checkpoint(str(self.ckpt_path()), device=self.device())
        model.eval()
        self._models[key] = model
        return model


def run_stage(run: Run, stage: str, fn=None) -> dict:
    """Run one stage under the deadline guard. Finished stages are skipped by hash.

    A stage may return status "partial" (it ran out of time but saved its progress); it gets
    no done-marker, so the next run continues it. A KeyboardInterrupt is recorded as
    "interrupted" and re-raised after the stage's output file is written.
    """
    fn = fn or STAGE_FUNCS[stage]
    # A Run subclass with its own stages (ground_truth_run.GTRun) supplies its own graph.
    upstream = getattr(run, "upstream", UPSTREAM)
    sources = getattr(run, "stage_sources", STAGE_SOURCES)
    h = run.stage_hash(stage)
    if stage != "package" and run.is_done(stage):
        out = read_json(run.output_path(stage))
        mk = read_json(run.marker(stage))
        run.log(f"[{stage}] already done (hash {h}); skipping")
        if mk.get("src") and mk["src"] != run.stage_src(stage):
            run.log(f"[{stage}] note: {', '.join(sources.get(stage, []))} changed since this "
                    "stage finished; its results are kept (stages are keyed on config and "
                    f"data). Pass force=('{stage}',) to recompute.")
        return out
    for up in upstream[stage]:
        if not run.is_done(up):
            out = {"status": "skipped", "reason": f"upstream stage '{up}' has not finished",
                   "hash": h}
            write_json(run.output_path(stage), out)
            run.log(f"[{stage}] skipped: {out['reason']}")
            return out
    available = run.stage_deadline(stage) - time.time()
    if available < run.preset["min_start_s"][stage]:
        out = {"status": "skipped", "hash": h,
               "reason": f"time budget: {available:.0f}s left for this stage, needs "
                         f"{run.preset['min_start_s'][stage]}s"}
        write_json(run.output_path(stage), out)
        run.log(f"[{stage}] skipped: {out['reason']}")
        return out
    run.log(f"[{stage}] starting ({available / 60:.1f} min available, "
            f"{run.remaining() / 60:.1f} min left in the run)")
    t0 = time.time()
    interrupted = False
    try:
        out = fn(run, run.stage_deadline(stage))
        out.setdefault("status", "done")
    except KeyboardInterrupt:
        interrupted = True
        out = {"status": "interrupted",
               "reason": "stopped by the user; saved progress is resumed on the next run"}
    except Exception as e:  # recorded, not swallowed silently: packaging reports it
        out = {"status": "error", "error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc()}
        run.log(f"[{stage}] ERROR {out['error']}\n{out['traceback']}")
    out["seconds"] = round(time.time() - t0, 1)
    out["hash"] = h
    out["src"] = run.stage_src(stage)
    write_json(run.output_path(stage), out)
    if stage != "package":
        run.add_consumed(out["seconds"], stage)
    if out["status"] == "done" and stage != "package":
        # Re-hash: the data stage changes the content hashes the other stages depend on.
        write_json(run.marker(stage), {"hash": run.stage_hash(stage), "src": out["src"],
                                       "finished": time.time()})
    run.log(f"[{stage}] {out['status']} in {out['seconds']:.0f}s")
    if interrupted:
        raise KeyboardInterrupt
    return out


# ---------------------------------------------------------------- stage 1: data
def _pbp_wp_columns(pbp) -> Any:
    """game_id, play_id, home_wp, vegas_home_wp from a polars or pandas frame, as pandas."""
    cols = [c for c in ("game_id", "play_id", "home_wp", "vegas_home_wp") if c in pbp.columns]
    if hasattr(pbp, "to_pandas"):
        df = pbp.select(cols).to_pandas()
    else:
        df = pbp[cols].copy()
    for c in ("home_wp", "vegas_home_wp"):
        if c not in df.columns:
            df[c] = np.nan
    return df


def align_nflverse_wp(split: S.Split, wp_df) -> dict[str, np.ndarray]:
    """nflverse home_wp and vegas_home_wp aligned to every record (NaN when missing)."""
    import pandas as pd

    rg = record_game(split)
    keys = pd.DataFrame({
        "game_id": np.asarray(split.game_ids, dtype=object)[rg],
        "play_id": np.asarray(split.play_ids, dtype=np.float64),
    })
    wp = wp_df.copy()
    wp["play_id"] = wp["play_id"].astype(np.float64)
    wp = wp.drop_duplicates(["game_id", "play_id"])
    merged = keys.merge(wp, on=["game_id", "play_id"], how="left")
    return {
        "nflverse_home_wp": merged["home_wp"].to_numpy(dtype=np.float64),
        "nflverse_vegas_home_wp": merged["vegas_home_wp"].to_numpy(dtype=np.float64),
    }


def stage_data(run: Run, deadline: float) -> dict:
    """Per season: reuse the saved arrays when they exist. If records.py or schema.py
    changed since they were built, rebuild the season in memory and keep the saved file
    when the arrays are identical (a cosmetic change); replace it when they differ, which
    changes the data content hash and so restarts training."""
    rec = _mod("records")
    p = run.preset
    ddir = run.workdir / "data"
    src_now = run.stage_src("data")
    out: dict[str, Any] = {"splits": {}, "source_hash": src_now, "seasons_rebuilt": {}}
    built: dict[str, S.Split] = {}
    for split_name in ("train", "val", "test"):
        seasons = check_seasons(p["seasons"][split_name])
        parts, reports, wp_parts = [], {}, []
        for season in seasons:
            cache = ddir / f"season_{season}.npz"
            rep_path = ddir / f"season_{season}_report.json"
            meta_path = ddir / f"season_{season}_meta.json"
            meta = read_json(meta_path) if meta_path.exists() else {}
            if time.time() > deadline and not cache.exists():
                raise TimeoutError(f"data stage ran out of time before season {season}")
            if cache.exists() and rep_path.exists():
                sp, extra = load_split(cache)
                report = read_json(rep_path)
                if meta.get("src") != src_now:
                    run.log(f"[data] records/schema code changed: rebuilding {season} in "
                            "memory to compare")
                    new_sp, new_report, new_extra = _build_season(rec, season)
                    same = split_arrays_hash(new_sp, new_extra) == split_arrays_hash(sp, extra)
                    out["seasons_rebuilt"][season] = "identical, kept" if same else "replaced"
                    if not same:
                        sp, report, extra = new_sp, new_report, new_extra
                        save_split(cache, sp, **extra)
                        write_json(rep_path, report)
                    write_json(meta_path, {"src": src_now})
            else:
                run.log(f"[data] downloading and building season {season}")
                sp, report, extra = _build_season(rec, season)
                save_split(cache, sp, **extra)
                write_json(rep_path, report)
                write_json(meta_path, {"src": src_now})
            reports[season] = _jsonable(report)
            parts.append(sp)
            wp_parts.append(extra)
        split = concat_splits(parts)
        extra = {k: np.concatenate([w[k] for w in wp_parts]) for k in wp_parts[0]}
        validation = rec.validate_split(split)
        built[split_name] = split
        target = ddir / f"{split_name}.npz"
        new_hash = split_arrays_hash(split, extra)
        if not target.exists() or run.content_hash(split_name) != new_hash:
            save_split(target, split, **extra)  # unchanged arrays keep their file (and mtime)
        write_json(ddir / f"{split_name}_validation.json", validation)
        write_json(ddir / f"{split_name}_build_report.json", reports)
        out["splits"][split_name] = {
            "seasons": seasons,
            "games": len(split.game_ids),
            "records": n_records(split),
            "content_hash": run.content_hash(split_name),
            "nflverse_wp_coverage": float(np.mean(np.isfinite(extra["nflverse_home_wp"]))),
            "validation": _jsonable(validation),
            "build_report": reports,
        }
        run.log(f"[data] {split_name}: {len(split.game_ids)} games, "
                f"{n_records(split)} records, seasons {seasons}")
    if hasattr(rec, "validate_disjoint"):
        out["disjoint"] = _jsonable(rec.validate_disjoint(built))
    return out



# ---------------------------------------------------------------- frequency baseline
def target_mask(split: S.Split) -> np.ndarray:
    """[N, T] bool: which target fields are scored, the same rule as model.WorldModel.loss.
    Phase is always scored. Down and ydstogo (where 0 means "not a down") are scored on
    every record except a game's last; every other field is scored where its code is not
    padding (0)."""
    t = np.asarray(split.targets)
    last = np.zeros(len(t), dtype=bool)
    last[np.asarray(split.offsets[1:]) - 1] = True
    m = t != 0
    for name in ("down", "ydstogo"):
        m[:, S.TARGET_INDEX[name]] = ~last
    m[:, S.TARGET_INDEX["phase"]] = True
    return m



def fit_frequency_baseline(split: S.Split, alpha: float = 0.5) -> list[np.ndarray]:
    """Per target field, P(field | current phase, current down) with add-alpha smoothing.
    Pad code 0 gets no mass except for fields where 0 is a real value (down, ydstogo)."""
    st, tg = split.states, split.targets
    ph = st[:, S.STATE_INDEX["phase"]].astype(np.int64)
    dn = st[:, S.STATE_INDEX["down"]].astype(np.int64)
    mask = target_mask(split)
    tables = []
    for f, (name, size) in enumerate(S.TARGET_FIELDS):
        counts = np.zeros((len(S.PHASES), 5, size))
        m = mask[:, f]
        np.add.at(counts, (ph[m], dn[m], tg[m, f].astype(np.int64)), 1.0)
        prior = np.full(size, alpha)
        if name not in ("down", "ydstogo"):
            prior[0] = 0.0
        counts += prior
        tables.append(counts / counts.sum(axis=2, keepdims=True))
    return tables


def baseline_nll(tables: list[np.ndarray], split: S.Split) -> dict:
    st, tg = split.states, split.targets
    ph = st[:, S.STATE_INDEX["phase"]].astype(np.int64)
    dn = st[:, S.STATE_INDEX["down"]].astype(np.int64)
    mask = target_mask(split)
    per_field, total, n_rows = {}, 0.0, int(mask[:, 0].sum())
    for f, (name, _size) in enumerate(S.TARGET_FIELDS):
        m = mask[:, f]
        p = tables[f][ph[m], dn[m], tg[m, f].astype(np.int64)]
        nll = -np.log(np.maximum(p, 1e-12))
        per_field[name] = float(nll.mean()) if m.any() else None
        total += float(nll.sum())
    return {"per_field": per_field, "total_per_transition": total / max(n_rows, 1),
            "n_transitions": n_rows}


def _per_field(d: dict) -> dict:
    src = d.get("fields", d.get("per_field", d)) if isinstance(d, dict) else {}
    return {name: float(src[name]) for name, _ in S.TARGET_FIELDS
            if name in src and src[name] is not None}


def nll_comparison(run: Run, split: S.Split, cfg_key: str) -> dict:
    """Model per-field NLL (via train.evaluate_nll) next to the frequency baseline."""
    train_mod = _mod("train")
    tr, _ = run.split("train")
    tables = fit_frequency_baseline(tr)
    max_games = run.preset[cfg_key].get("nll_max_games")
    if max_games and len(split.game_ids) > max_games:
        split = subset_split(split, np.arange(max_games))
    base = baseline_nll(tables, split)
    model_d = train_mod.evaluate_nll(run.load_model(), split, run.model_cfg())
    model_pf = _per_field(model_d)
    return {
        "games": len(split.game_ids),
        "model": _jsonable(model_d),
        "baseline_freq_phase_down": base,
        "per_field": {name: {"model": model_pf.get(name),
                             "baseline": base["per_field"].get(name)}
                      for name, _ in S.TARGET_FIELDS},
        "note": "Model per-field NLL is conditional on the fields sampled before it in the "
                "same transition; the baseline is the marginal given phase and down. The "
                "totals per transition are the comparable numbers.",
    }


# ---------------------------------------------------------------- league stats
def real_league_stats(split: S.Split) -> dict:
    st, tg = split.states, split.targets
    g = len(split.game_ids)
    phase = st[:, S.STATE_INDEX["phase"]]
    down = st[:, S.STATE_INDEX["down"]]
    act = st[:, S.STATE_INDEX["action"]]
    pts = np.abs(S.score_delta_value(tg[:, S.TARGET_INDEX["score_delta"]]))
    fourth = (phase == S.PHASE["PLAY"]) & (down == 4)
    go = np.isin(act, S.ANY_GO_ACTIONS)
    fd = fourth & (go | (act == S.ACTION["PUNT"]) | (act == S.ACTION["FG"]))
    final = np.asarray(split.final_diff, dtype=float)
    return {
        "games": g,
        "plays_per_game": float((phase == S.PHASE["PLAY"]).sum() / g),
        "points_per_game_both_teams": float(pts.sum() / g),
        "touchdowns_per_game": float(np.isin(pts, (6, 7, 8)).sum() / g),
        "field_goals_per_game": float((pts == 3).sum() / g),
        "punts_per_game": float((act == S.ACTION["PUNT"]).sum() / g),
        "fourth_down_go_rate": float((fd & go).sum() / max(fd.sum(), 1)),
        "fourth_down_punt_rate": float((fd & (act == S.ACTION["PUNT"])).sum() / max(fd.sum(), 1)),
        "fourth_down_fg_rate": float((fd & (act == S.ACTION["FG"])).sum() / max(fd.sum(), 1)),
        "pass_rate": float((act == S.ACTION["PASS"]).sum()
                           / max(np.isin(act, S.GO_ACTIONS).sum(), 1)),
        "home_win_rate_regulation": float(np.mean(S.outcome(final))),
        "regulation_tie_rate": float(np.mean(final == 0)),
        "mean_abs_final_margin": float(np.mean(np.abs(final))),
    }


# ---------------------------------------------------------------- WP calibration
def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def game_bootstrap(values: np.ndarray, groups: np.ndarray, n_boot: int,
                   rng: np.random.Generator) -> dict:
    """Mean of per-state values with a 95% interval from resampling whole games."""
    values = np.asarray(values, dtype=float)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[draws].sum(axis=1) / np.maximum(cnts[draws].sum(axis=1), 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(values.mean()), "lo95": float(lo), "hi95": float(hi),
            "n_states": int(len(values)), "n_games": int(len(uniq))}


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    out = {"bin_mean_pred": [], "bin_mean_outcome": [], "bin_count": []}
    for k in range(n_bins):
        m = b == k
        out["bin_mean_pred"].append(float(p[m].mean()) if m.any() else None)
        out["bin_mean_outcome"].append(float(y[m].mean()) if m.any() else None)
        out["bin_count"].append(int(m.sum()))
    return out


def sample_wp_states(split: S.Split, extra: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random PLAY records that also have both nflverse probabilities, so all three
    estimates are scored on exactly the same states."""
    phase = split.states[:, S.STATE_INDEX["phase"]]
    play = phase == S.PHASE["PLAY"]
    ok = play.copy()
    for k in ("nflverse_home_wp", "nflverse_vegas_home_wp"):
        if k in extra:
            ok &= np.isfinite(extra[k])
    cand = np.flatnonzero(ok)
    if len(cand) == 0:  # nflverse missing everywhere: score ours alone, flagged later
        cand = np.flatnonzero(play)
    n = min(n, len(cand))
    return np.sort(rng.choice(cand, size=n, replace=False)) if n else cand[:0]


def wp_calibration(run: Run, split: S.Split, extra: dict, cfg_key: str, deadline: float,
                   seed: int, pdir: Path) -> dict:
    """Win probability by simulation on a fixed random sample of PLAY records, scored next
    to nflverse's home_wp and vegas_home_wp on the same states (theirs, labelled)."""
    sim = _mod("simulate")
    c = run.preset[cfg_key]
    n_states, n_roll = int(c["wp_states"]), int(c["wp_rollouts"])
    idx = sample_wp_states(split, extra, n_states, np.random.default_rng(seed))
    model, cfg = run.load_model(), run.model_cfg()

    def fn(part, rng):
        return sim.wp_by_simulation(model, split, part, n_roll, cfg, deadline, rng)

    df, info = chunked(run, pdir, fn, idx, _chunk_size(run, n_roll, cfg_key), deadline, seed)
    rng = np.random.default_rng(seed + 1)
    used = idx[: len(df)]
    col = "home_wp_sim" if "home_wp_sim" in df.columns else "sim_wp"
    if len(df) and "idx" in df.columns:
        assert np.array_equal(df["idx"].to_numpy(), used), "simulate returned rows out of order"
    ok = np.isfinite(df[col].to_numpy(dtype=float)) if len(df) else np.zeros(0, dtype=bool)
    if "done" in df.columns:
        ok &= df["done"].to_numpy(dtype=bool)
    n_unfinished = int((~ok).sum())
    df, used = df[ok], used[ok]
    base = {"n_states": int(len(used)), "n_unfinished_dropped": n_unfinished,
            "n_rollouts_per_state": n_roll, "progress": info,
            "complete": info["complete"]}
    if len(used) == 0:
        return {**base, "metrics": {}, "paired_differences": {},
                "note": "no finished rollouts inside the time budget"}
    y = S.outcome(np.asarray(split.final_diff)[record_game(split)[used]])
    games = record_game(split)[used]
    preds = {"ours_simulation": df[col].to_numpy(dtype=float)}
    if "nflverse_home_wp" in extra:
        theirs = {"nflverse_home_wp (theirs)": extra["nflverse_home_wp"][used],
                  "nflverse_vegas_home_wp (theirs)": extra["nflverse_vegas_home_wp"][used]}
        if all(np.isfinite(v).all() for v in theirs.values()):
            preds.update(theirs)
    n_boot = int(c["bootstrap"])
    metrics, diffs = {}, {}
    for k, p in preds.items():
        metrics[k] = {
            "log_loss": game_bootstrap(log_loss(p, y), games, n_boot, rng),
            "brier": game_bootstrap(brier(p, y), games, n_boot, rng),
            "reliability": reliability(p, y),
        }
    for k, p in preds.items():
        if k == "ours_simulation":
            continue
        diffs[f"ours minus {k}"] = {
            "log_loss": game_bootstrap(log_loss(preds["ours_simulation"], y) - log_loss(p, y),
                                       games, n_boot, rng),
            "brier": game_bootstrap(brier(preds["ours_simulation"], y) - brier(p, y),
                                    games, n_boot, rng),
        }
    label_check = None
    lab = "home_result" if "home_result" in df.columns else "actual"
    if lab in df.columns:
        label_check = float(np.mean(np.isclose(df[lab].to_numpy(dtype=float), y)))
    return {
        **base,
        "mean_steps_per_rollout": float(df["mean_steps"].mean())
        if "mean_steps" in df.columns and len(df) else None,
        "outcome_label": "home result at end of regulation: 1 win, 0.5 tie, 0 loss",
        "simulate_actual_matches_label": label_check,
        "metrics": metrics,
        "paired_differences": diffs,
        "source_note": "nflverse home_wp and vegas_home_wp are nflverse's published "
                       "estimates on the same plays, not ours.",
    }


# ---------------------------------------------------------------- stage 2: train
def stage_train(run: Run, deadline: float) -> dict:
    import torch

    model_mod, train_mod = _mod("model"), _mod("train")
    torch.manual_seed(run.seed)
    np.random.seed(run.seed % 2**32)
    tr, _ = run.split("train")
    va, _ = run.split("val")
    mg = run.preset["train"].get("max_train_games")
    if mg and len(tr.game_ids) > mg:
        tr = subset_split(tr, np.arange(mg))
        va = subset_split(va, np.arange(min(len(va.game_ids), max(mg // 4, 8))))
    cfg = run.model_cfg()
    model = model_mod.WorldModel(cfg)
    ck_path = run.ckpt_path()
    fresh_reason = _check_checkpoint(ck_path, model)
    if fresh_reason:
        stale = ck_path.with_name(ck_path.name + f".stale_{int(time.time())}")
        ck_path.replace(stale)
        run.log(f"[train] checkpoint does not fit the current model ({fresh_reason}); "
                f"moved it to {stale.name} and starting fresh")
    progress = run.workdir / "outputs" / "train_progress.json"
    prev = read_json(progress) if progress.exists() else {}
    if prev.get("ckpt") != str(ck_path) or fresh_reason:
        prev = {"seconds": 0.0, "lines": [], "ckpt": str(ck_path)}
    budget_s = 60.0 * run.preset["stage_min"]["train"] - float(prev.get("seconds", 0.0))
    deadline = min(deadline, time.time() + max(budget_s, run.preset["min_start_s"]["train"]))
    t0, base = time.time(), float(prev.get("seconds", 0.0))

    def log(*msg: Any) -> None:
        # Keep the training time used current, so a disconnect mid-training does not hand
        # the resumed run a fresh training budget.
        prev["seconds"] = base + time.time() - t0
        prev["lines"] = (prev.get("lines", []) + [" ".join(str(m) for m in msg)])[-200:]
        write_json(progress, prev)
        run.log("[train]", *msg)

    run.log(f"[train] {model.count_params():,} parameters; {len(tr.game_ids)} train games, "
            f"{n_records(tr):,} records; device {cfg['device']}; checkpoint {ck_path}")
    history = train_mod.train_model(model, tr, va, cfg, str(ck_path), deadline, log)
    if not ck_path.exists():
        raise RuntimeError("train_model returned without writing a checkpoint")
    return {
        "params": int(model.count_params()),
        "train_games": len(tr.game_ids),
        "train_records": n_records(tr),
        "val_games_for_early_stopping": len(va.game_ids),
        "history": _jsonable(history),
        "train_seconds_total": round(base + time.time() - t0, 1),
        "checkpoint": str(ck_path),
        "fresh_start_reason": fresh_reason,
    }


# ---------------------------------------------------------------- stage 3: eval (val)
def stage_eval(run: Run, deadline: float) -> dict:
    sim = _mod("simulate")
    pdir = run.partial_dir("eval")
    va, extra = run.split("val")
    out: dict[str, Any] = {"split": "validation", "seasons": run.preset["seasons"]["val"]}
    out["nll"] = _cached_json(pdir / "nll.json", lambda: nll_comparison(run, va, "eval"))
    run.log(f"[eval] NLL per transition: model "
            f"{_total_nll(out['nll']['model'])}, baseline "
            f"{out['nll']['baseline_freq_phase_down']['total_per_transition']:.3f}")
    out["league_real"] = real_league_stats(va)
    n_games = int(run.preset["eval"]["sim_games"])
    # Simulated games get at most a third of the stage; the rest goes to WP calibration.
    sim_deadline = time.time() + (deadline - time.time()) / 3
    league_path = pdir / "league_sim.json"
    if n_games <= 0:
        out["league_sim"] = {"skipped": "sim_games is 0 in this preset"}
    elif league_path.exists():
        out["league_sim"] = read_json(league_path)
    else:
        res = sim.simulate_games(run.load_model(), va, n_games, run.model_cfg(), sim_deadline,
                                 np.random.default_rng(run.seed + 30))
        res.pop("table", None)  # the same numbers as "sim" and "real"
        out["league_sim"] = _jsonable(res)
        if res.get("n_games", 0) >= n_games:
            write_json(league_path, out["league_sim"])
        run.log(f"[eval] league stats over {res.get('n_games')} simulated games: "
                f"sim {res.get('sim')} real {res.get('real')}")
    out["wp"] = wp_calibration(run, va, extra, "eval", deadline, run.seed + 3, pdir / "wp")
    _log_wp(run, "eval", out["wp"])
    if not out["wp"]["complete"]:
        out["status"] = "partial"
        out["reason"] = (f"time budget: {out['wp']['n_states']} of "
                         f"{out['wp']['progress']['planned_states']} states; the next run "
                         "continues from the saved chunks")
    return out


def _total_nll(d: dict) -> str:
    for k in ("total", "total_per_transition", "nll", "mean"):
        if isinstance(d, dict) and isinstance(d.get(k), (int, float)):
            return f"{d[k]:.3f}"
    return "?"


def _log_wp(run: Run, stage: str, wp: dict) -> None:
    for k, m in wp.get("metrics", {}).items():
        ll, br = m["log_loss"], m["brier"]
        run.log(f"[{stage}] {k}: log loss {ll['mean']:.4f} [{ll['lo95']:.4f}, "
                f"{ll['hi95']:.4f}], Brier {br['mean']:.4f} on {ll['n_states']} states")


# ---------------------------------------------------------------- stage 4: prereg + test
def test_plan(run: Run) -> dict:
    p = run.preset
    sc = _mod("support").config(p["plan"])
    return {
        "config_hash": run.full_config_hash(),
        "train_hash": run.stage_hash("train"),
        "test_hash": run.stage_hash("test"),
        "test_seasons": p["seasons"]["test"],
        "seed": run.seed + 4,
        "metrics": [
            "per-field and total NLL per transition, model vs frequency baseline fitted on "
            "the training split (P(field | phase, down), add-0.5 smoothing)",
            f"win probability by simulation on {p['test']['wp_states']} random PLAY records "
            f"with both nflverse probabilities present, {p['test']['wp_rollouts']} rollouts "
            "each; log loss and Brier with a 95% game-bootstrap interval "
            f"({p['test']['bootstrap']} resamples) for ours, nflverse home_wp and nflverse "
            "vegas_home_wp on the same states, and paired differences",
            f"fourth-down plan on {p['plan']['n_fourth']} sampled test fourth downs, "
            f"{p['plan']['n_rollouts']} rollouts per action; field goals beyond the longest "
            f"made in the training split plus {sc['fg_margin_yards']} yards and punts inside "
            f"the opponent's {sc['punt_min_yardline']} are not simulated; agreement of the "
            "recommended action with the real coach on decisions not flagged too_close, also "
            "split by training support (low support: the recommended call made fewer than "
            f"{sc['min_support']} times by coaches in comparable training states)",
        ],
        "outcome_label": "home result at end of regulation: 1 win, 0.5 tie, 0 loss",
        "preset": p["name"],
        "model": p["model"],
        "rule": "each configuration (full config hash: data, model, training, evaluation, "
                "test and plan settings) preregisters its own plan in prereg/<hash>.json "
                "before it reads the test seasons, and scores them once; a second scoring of "
                "the same configuration needs allow_rescore=True and is recorded as such. "
                "Every preregistration in the work folder is kept and listed in results.json.",
    }


# ---------------------------------------------------------------- preregistration files
# prereg/<config hash>.json           the plan, written once before any test data is read,
#                                     never overwritten
# prereg/<config hash>.attempts.json  every test or plan stage start (a log; appended)
# prereg/<config hash>.scored.json    written once, when that configuration's test scoring
#                                     completed
def prereg_dir(run: Run) -> Path:
    d = run.workdir / "prereg"
    d.mkdir(exist_ok=True)
    return d


def prereg_path(run: Run, config_hash_: str | None = None) -> Path:
    return prereg_dir(run) / f"{config_hash_ or run.full_config_hash()}.json"


def _write_once(path: Path, obj: Any) -> bool:
    """Write path only if it does not exist yet. Returns True if this call wrote it."""
    if path.exists():
        return False
    write_json(path, obj)
    return True


def migrate_legacy_prereg(run: Run) -> list[str]:
    """Copy the first notebook's single prereg.json (and any earlier plans chained under
    "previous") and outputs/test_scored.json into the per-configuration files. The legacy
    files are left where they are. Returns the hashes imported."""
    imported = []
    legacy = run.workdir / "prereg.json"
    d = prereg_dir(run)
    p = read_json(legacy) if legacy.exists() else None
    while isinstance(p, dict):
        h = p.get("config_hash")
        rec = {k: v for k, v in p.items() if k != "previous"}
        rec["migrated_from"] = "prereg.json (single-file format of the first notebook)"
        if h and _write_once(d / f"{h}.json", rec):
            imported.append(h)
        p = p.get("previous")
    scored = run.workdir / "outputs" / "test_scored.json"
    if scored.exists():
        s = read_json(scored)
        if s.get("config_hash"):
            _write_once(d / f"{s['config_hash']}.scored.json",
                        {**s, "scored_at": s.get("at"),
                         "migrated_from": "outputs/test_scored.json"})
    return imported


def _migrate_and_log(run: Run, stage: str) -> None:
    for h in migrate_legacy_prereg(run):
        run.log(f"[{stage}] imported the earlier single-file preregistration of config {h} "
                "into prereg/ (prereg.json is left in place)")


def ensure_prereg(run: Run, stage: str, rescore: bool = False) -> Path:
    """This configuration's preregistration: written now if it does not exist (BEFORE the
    caller reads any test data), never overwritten. Every call is logged as an attempt."""
    _migrate_and_log(run, stage)
    plan = test_plan(run)
    path = prereg_path(run, plan["config_hash"])
    if _write_once(path, {**plan, "written_at": time.time(), "written_by_stage": stage}):
        run.log(f"[{stage}] preregistered this configuration's test plan: {path}")
    elif read_json(path).get("test_hash") != plan["test_hash"]:  # cannot happen by design
        raise RuntimeError(f"{path} does not match this configuration's test plan")
    att_path = path.with_name(f"{plan['config_hash']}.attempts.json")
    attempts = read_json(att_path) if att_path.exists() else []
    attempts.append({"stage": stage, "started_at": time.time(), "rescore": rescore})
    write_json(att_path, attempts)
    return path


def list_preregistrations(run: Run) -> list[dict]:
    """Every preregistration in the work folder, oldest first, with whether and when that
    configuration scored the test seasons."""
    migrate_legacy_prereg(run)
    current = run.full_config_hash()
    rows = []
    for f in prereg_dir(run).glob("*.json"):
        if f.name.endswith((".attempts.json", ".scored.json")):
            continue
        p = read_json(f)
        h = p.get("config_hash", f.stem)
        sc = f.with_name(f"{h}.scored.json")
        att = f.with_name(f"{h}.attempts.json")
        attempts = read_json(att) if att.exists() else p.get("attempts", [])
        scored = read_json(sc) if sc.exists() else None
        rows.append({
            "config_hash": h,
            "test_hash": p.get("test_hash"),
            "train_hash": p.get("train_hash"),
            "preset": p.get("preset"),
            "written_at": p.get("written_at"),
            "test_scored": scored is not None,
            "test_scored_at": (scored or {}).get("scored_at"),
            "attempts": len(attempts),
            "rescores": sum(bool(a.get("rescore")) for a in attempts),
            "this_run": h == current,
            "file": f"prereg/{f.name}",
            "migrated_from": p.get("migrated_from"),
        })
    return sorted(rows, key=lambda r: r["written_at"] or 0)


def _scorings_note(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["test_scored"]]
    note = (f"The test seasons in this work folder have been scored by {len(scored)} "
            "configuration(s)")
    if len(scored) > 1:
        note += ("; each is a separate experiment on the same test seasons, and only the "
                 "first was chosen before any test-season result existed")
    return {"n_configurations_scored": len(scored),
            "config_hashes_scored": [r["config_hash"] for r in scored],
            "note": note + "."}


def stage_test(run: Run, deadline: float) -> dict:
    plan = test_plan(run)
    h = plan["config_hash"]
    scored_path = prereg_dir(run) / f"{h}.scored.json"
    _migrate_and_log(run, "test")
    rescore = scored_path.exists()
    if rescore and not run.allow_rescore:
        raise PermissionError(
            f"the test seasons were already scored by this configuration (config hash {h}); "
            "refusing a second scoring of the same configuration. Set allow_rescore=True to "
            "override; the rescore is recorded in prereg/.")
    path = ensure_prereg(run, "test", rescore=rescore)  # BEFORE any test data is loaded

    pdir = run.partial_dir("test")
    te, extra = run.split("test")
    out: dict[str, Any] = {"split": "test", "seasons": run.preset["seasons"]["test"],
                           "prereg": str(path), "config_hash": h,
                           "test_hash": plan["test_hash"], "rescore": rescore}
    out["nll"] = _cached_json(pdir / "nll.json", lambda: nll_comparison(run, te, "test"))
    out["league_real"] = real_league_stats(te)
    out["wp"] = wp_calibration(run, te, extra, "test", deadline, run.seed + 4, pdir / "wp")
    _log_wp(run, "test", out["wp"])
    if out["wp"]["complete"]:
        _write_once(scored_path, {"config_hash": h, "test_hash": plan["test_hash"],
                                  "scored_at": time.time()})
    else:
        out["status"] = "partial"
        out["reason"] = "time budget; the next run continues the same preregistered plan"
    return out


# ---------------------------------------------------------------- stage 5: plan
HYPOTHETICAL = {
    "description": "offense up 4, 4:00 left in the 4th quarter, 4th and 2 at its own 40",
    "lead_for_offense": 4,
    "clock_seconds": 240,
    "down": 4,
    "ydstogo": 2,
    "yardline_100": 60,
}


def build_hypothetical(split: S.Split) -> tuple[S.Split, dict]:
    """Edit the codes of a real late-game record into the hypothetical state.

    The base record is the first PLAY record, in game order of the split, with 3:20 to
    5:00 left in the game and a real down. Its game is cut after that record; score,
    clock, down, distance and yard line are overwritten (score from the offense's side),
    while possession, timeouts, the pregame spread and the earlier records stay as they
    were in the real game. The record's own action field is left as recorded; the planner
    replaces it with each candidate action.
    """
    st, raw = split.states, split.raw
    phase = st[:, S.STATE_INDEX["phase"]]
    down = st[:, S.STATE_INDEX["down"]]
    poss = st[:, S.STATE_INDEX["poss"]]
    clock = raw[:, 0]
    cand = np.flatnonzero((phase == S.PHASE["PLAY"]) & (down >= 1) & (clock >= 200)
                          & (clock <= 300) & (poss > 0))
    if len(cand) == 0:
        raise ValueError("no late-game PLAY record to edit")
    i = int(cand[0])
    g = int(record_game(split)[i])
    sub = subset_split(split, [g])
    j = i - int(split.offsets[g])
    sub = S.Split(
        states=sub.states[: j + 1].copy(), raw=sub.raw[: j + 1].copy(),
        targets=sub.targets[: j + 1].copy(),
        offsets=np.array([0, j + 1], dtype=np.int64), header=sub.header.copy(),
        final_diff=sub.final_diff.copy(), game_ids=[sub.game_ids[0] + "_HYPOTHETICAL"],
        seasons=sub.seasons.copy(), play_ids=sub.play_ids[: j + 1].copy())
    h = HYPOTHETICAL
    home_has_ball = int(sub.states[j, S.STATE_INDEX["poss"]]) == S.POSS_CODE["HOME"]
    diff = h["lead_for_offense"] if home_has_ball else -h["lead_for_offense"]
    before = {name: int(sub.states[j, S.STATE_INDEX[name]]) for name, _ in S.STATE_FIELDS}
    raw_before = [int(v) for v in sub.raw[j]]
    sub.raw[j, 0] = h["clock_seconds"]
    sub.raw[j, 1] = diff
    sub.states[j, S.STATE_INDEX["clock"]] = S.clock_code(h["clock_seconds"])
    sub.states[j, S.STATE_INDEX["score"]] = S.score_code(diff)
    sub.states[j, S.STATE_INDEX["down"]] = h["down"]
    sub.states[j, S.STATE_INDEX["ydstogo"]] = h["ydstogo"]
    sub.states[j, S.STATE_INDEX["yardline"]] = h["yardline_100"]
    sub.targets[j] = 0
    sub.targets[j, S.TARGET_INDEX["phase"]] = S.PHASE["END"]
    doc = {
        **h,
        "base_game_id": split.game_ids[g],
        "base_play_id": int(split.play_ids[i]),
        "base_record_index": i,
        "base_codes_before_edit": before,
        "base_raw_before_edit": dict(zip(S.RAW_COLUMNS, raw_before, strict=True)),
        "codes_after_edit": {name: int(sub.states[j, S.STATE_INDEX[name]])
                             for name, _ in S.STATE_FIELDS},
        "raw_after_edit": dict(zip(S.RAW_COLUMNS, [int(v) for v in sub.raw[j]], strict=True)),
        "offense": "HOME" if home_has_ball else "AWAY",
        "kept_from_real_game": "possession, timeouts, pregame spread, opening receiver, "
                               "all earlier records",
        "limitation": "earlier records are the real game's, so the history does not lead "
                      "exactly to the edited score and clock",
    }
    return sub, doc


def _agreement(df) -> dict:
    if df is None or len(df) == 0 or "best" not in df.columns:
        return {"n": 0}
    n_unfinished = 0
    if "done" in df.columns:
        n_unfinished = int((~df["done"].astype(bool)).sum())
        df = df[df["done"].astype(bool)]
    if len(df) == 0:
        return {"n": 0, "n_unfinished_dropped": n_unfinished}
    col = "real_choice" if "real_choice" in df.columns else "real_action"
    real = df[col].astype(str).str.upper()
    best = df["best"].astype(str).str.upper()
    close = df["too_close"].astype(bool) if "too_close" in df.columns else best.isna()
    agree = real == best
    confusion: dict[str, dict[str, int]] = {}
    for b, r in zip(best, real, strict=True):
        confusion.setdefault(b, {}).setdefault(r, 0)
        confusion[b][r] += 1
    return {
        "n": int(len(df)),
        "n_unfinished_dropped": n_unfinished,
        "real_column": col,
        "n_clear": int((~close).sum()),
        "agreement_all": float(agree.mean()),
        "agreement_clear": float(agree[~close].mean()) if (~close).any() else None,
        "recommended_counts": best.value_counts().to_dict(),
        "real_counts": real.value_counts().to_dict(),
        "confusion_recommended_by_real": confusion,
    }


SUPPORT_NOTE = ("The hard action rules and the support flags were added after the first Colab "
                "T4 run's test-season fourth-down results had been seen (see "
                "results/world_model/colab_t4_run_summary.md). They are a post hoc change to "
                "the planner, not part of that run's preregistered test plan.")


def _agreement_by_support(df, min_support: int) -> dict:
    """Agreement with the coach, separately for well-supported and low-support
    recommendations (finished decisions only)."""
    if df is None or len(df) == 0 or "low_support" not in df.columns:
        return {"min_support": min_support, "well_supported": {"n": 0},
                "low_support": {"n": 0}}
    d = df[df["done"].astype(bool)] if "done" in df.columns else df
    low = d["low_support"].astype(bool)
    return {
        "min_support": min_support,
        "rule": f"low support: coaches made the recommended call fewer than {min_support} "
                "times in comparable training fourth downs (yard line within 3, same "
                "yards-to-go bucket)",
        "n_low_support": int(low.sum()),
        "well_supported": _agreement(d[~low]),
        "low_support": _agreement(d[low]),
    }


def _fg_recommendations(df) -> dict:
    """The field-goal calls: the simulator's one-step make rate against the training make
    rate at that yard line, and what the coach did."""
    if df is None or len(df) == 0 or "best" not in df.columns:
        return {"n": 0}
    d = df[df["best"] == "FG"]
    if "done" in d.columns:
        d = d[d["done"].astype(bool)]
    out: dict[str, Any] = {"n": int(len(d))}
    if len(d) == 0:
        return out
    out["coach_choice_counts"] = d["real_choice"].astype(str).value_counts().to_dict()
    if "low_support" in d.columns:
        out["n_low_support"] = int(d["low_support"].astype(bool).sum())
    for k in ("sim_fg_make", "emp_fg_make"):
        if k in d.columns:
            v = d[k].to_numpy(dtype=float)
            out[f"mean_{k}"] = float(np.nanmean(v)) if np.isfinite(v).any() else None
    return out


def _support_diagnostic(run: Run, sp, model, te: S.Split, support, rules: dict, cfg: dict,
                        pool: np.ndarray, decision_idx: np.ndarray, hyp_split: S.Split,
                        hyp_idx: np.ndarray, pdir: Path, deadline: float) -> dict:
    """Simulated versus empirical make rate and punt net yards (support.sim_vs_empirical),
    plus the one-step field-goal make rate of every planned state and of the hypothetical.
    One encoder pass per state, so it is cheap; it gets at most an eighth of the time left
    and is saved once complete. A failure is recorded, never fatal to the plan."""
    path = pdir / "support_diagnostic.json"
    if path.exists():
        return read_json(path)
    dl = time.time() + max(deadline - time.time(), 0.0) / 8
    rng = np.random.default_rng(run.seed + 70)
    n = int(support.cfg["diag_draws"])
    try:
        d = sp.sim_vs_empirical(model, te, support, rules, cfg, rng, pool, dl)
        d["fg_headline"] = sp.fg_gap_summary(d, int(support.cfg["min_support"]))
        d["decision_states"] = {
            "idx": decision_idx,
            "sim_fg_make": sp.sim_fg_make(model, te, decision_idx, n, cfg, rng, dl)}
        hyp = sp.sim_fg_make(model, hyp_split, hyp_idx, n, cfg, rng, dl)
        yl = int(hyp_split.states[int(hyp_idx[0]), S.STATE_INDEX["yardline"]])
        emp, att = sp.empirical_fg_rate(support, [yl])
        d["hypothetical_fg"] = {"yardline_100": yl,
                                "distance_yards": yl + sp.KICK_DISTANCE_OFFSET,
                                "sim_make_rate": float(hyp[0]),
                                "train_attempts_within_2_yards": int(att[0]),
                                "empirical_make_rate": float(emp[0])}
    except Exception as e:  # recorded in results.json; the plan itself goes on
        run.log(f"[plan] support diagnostic failed: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}"}
    d = _jsonable(d)
    write_json(path, d)
    return d


def stage_plan(run: Run, deadline: float) -> dict:
    rec, sim, sp = _mod("records"), _mod("simulate"), _mod("support")
    # The plan also reads the test seasons, so it needs this configuration's preregistration
    # too (written here if the test stage was skipped or failed).
    ensure_prereg(run, "plan")
    pdir = run.partial_dir("plan")
    p = run.preset["plan"]
    scfg = sp.config(p)
    # Support tables and hard rules: from the TRAINING split only, never validation or test.
    tr, _ = run.split("train")
    support = sp.build_support(tr, scfg)
    del tr
    rules = sp.hard_rules(support)
    run.log(f"[plan] action rules: {rules['text']}")
    te, _ = run.split("test")
    model, cfg = run.load_model(), run.model_cfg()
    out: dict[str, Any] = {"split": "test", "seasons": run.preset["seasons"]["test"]}
    out["support"] = {"config": scfg, "rules": rules, "note": SUPPORT_NOTE,
                      "training_tables": support.to_json()}
    # The hypothetical first (one state, many rollouts), with up to a quarter of the time.
    hyp_split, doc = build_hypothetical(te)
    hyp_deadline = time.time() + (deadline - time.time()) / 4
    hyp_idx = np.array([n_records(hyp_split) - 1])

    def hyp_fn(part, rng):
        return sim.plan_fourth_downs(model, hyp_split, part, int(p["hypo_rollouts"]), cfg,
                                     hyp_deadline, rng,
                                     action_mask=sp.action_mask(hyp_split, part, rules))

    hdf, hinfo = chunked(run, pdir / "hypothetical", hyp_fn, hyp_idx, 1, hyp_deadline,
                         run.seed + 50)
    if len(hdf):
        hdf = sp.add_support_columns(hdf, support, hyp_split, rules)
    out["hypothetical"] = {"state": doc, "result": _jsonable(hdf), "progress": hinfo,
                           "n_rollouts_per_action": int(p["hypo_rollouts"])}
    if len(hdf):
        run.log(f"[plan] hypothetical: {hdf.to_dict(orient='records')}")
    # Sampled real fourth downs (fixed sample; chunks saved as they finish).
    idx_all = np.asarray(rec.fourth_down_index(te))
    n = min(int(p["n_fourth"]), len(idx_all))
    rng = np.random.default_rng(run.seed + 5)
    idx = np.sort(rng.choice(idx_all, size=n, replace=False)) if n else idx_all[:0]
    masked = (sp.masked_actions(hyp_split, hyp_idx, rules, "hypothetical")
              + sp.masked_actions(te, idx, rules, "test"))
    out["masked_actions"] = masked
    test_masked = [m for m in masked if m["set"] == "test"]
    out["masked_summary"] = {
        "planned_states": int(n),
        "states_with_a_masked_action": len({m["idx"] for m in test_masked}),
        "fg_masked": sum(m["masked"] == "FG" for m in test_masked),
        "punt_masked": sum(m["masked"] == "PUNT" for m in test_masked),
        "hypothetical_masked": [m["masked"] for m in masked if m["set"] == "hypothetical"],
    }
    # Simulated versus empirical kicking (cheap), before the long planning loop.
    diag = _support_diagnostic(run, sp, model, te, support, rules, cfg, idx_all, idx,
                               hyp_split, hyp_idx, pdir, deadline)
    out["support"]["diagnostic"] = diag
    if diag.get("fg_headline"):
        run.log(f"[plan] support diagnostic: {diag['fg_headline']}")

    def fn(part, rng):
        return sim.plan_fourth_downs(model, te, part, int(p["n_rollouts"]), cfg, deadline, rng,
                                     action_mask=sp.action_mask(te, part, rules))

    chunk = max(1, _chunk_size(run, 3 * int(p["n_rollouts"]), "plan"))
    df, info = chunked(run, pdir / "fourth", fn, idx, chunk, deadline, run.seed + 5)
    if len(df):
        df = sp.add_support_columns(df, support, te, rules)
        yl = te.states[df["idx"].to_numpy(dtype=np.int64), S.STATE_INDEX["yardline"]]
        emp, att = sp.empirical_fg_rate(support, yl)
        df["emp_fg_make"], df["emp_fg_attempts"] = emp, att
        ds = diag.get("decision_states") or {}
        sim_fg = dict(zip(ds.get("idx", []), ds.get("sim_fg_make", []), strict=True))
        df["sim_fg_make"] = [sim_fg.get(int(i), np.nan) for i in df["idx"]]
    out["progress"] = info
    out["n_fourth_downs_available"] = int(len(idx_all))
    out["n_rollouts_per_action"] = int(p["n_rollouts"])
    out["decisions"] = _jsonable(df)
    out["agreement"] = _agreement(df)
    out["agreement_by_support"] = _agreement_by_support(df, int(scfg["min_support"]))
    out["fg_recommendations"] = _fg_recommendations(df)
    run.log(f"[plan] {out['agreement']}")
    run.log(f"[plan] by support: well supported {out['agreement_by_support']['well_supported']}"
            f"; low support {out['agreement_by_support']['low_support']}")
    if not (info["complete"] and hinfo["complete"]):
        out["status"] = "partial"
        out["reason"] = (f"time budget: {info['reused_states'] + info['computed_states']} of "
                         f"{n} fourth downs; the next run continues from the saved chunks")
    return out


# ---------------------------------------------------------------- stage 6: package
def versions() -> dict:
    v: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "numpy", "polars", "pandas", "nflreadpy", "matplotlib"):
        try:
            mod = importlib.import_module(name)
            v[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            v[name] = None
    try:
        import torch

        v["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            v["gpu"] = props.name
            v["gpu_mem_gb"] = round(props.total_memory / 2**30, 1)
            v["cuda"] = torch.version.cuda
    except Exception:
        pass
    return v


def stage_package(run: Run, deadline: float) -> dict:
    results: dict[str, Any] = {
        "experiment": "NFL world model: next-play transformer used as a simulator",
        "preset": run.preset,
        "seed": run.seed,
        "stage_seeds": {"train": run.seed, "eval": run.seed + 3, "test": run.seed + 4,
                        "plan": run.seed + 5},
        "config_hash": run.full_config_hash(),
        "stage_hashes": {s: run.stage_hash(s) for s in STAGES},
        "versions": versions(),
        "workdir": str(run.workdir),
        "budget_s": run.total_s,
        "consumed_s_before_this_session": run.consumed_before,
        "session_seconds": round(time.time() - run.session_start, 1),
        "stages": {},
        "honesty": "Every number here was produced by this run. Fields labelled 'theirs' or "
                   "'nflverse' are nflverse's published estimates on the same plays.",
    }
    for s in STAGES:
        if s == "package":
            continue
        results["stages"][s] = run.current_output(s)
    results["benchmark"] = run.preset.get("benchmark")
    results["preset_choice"] = run.preset.get("picked")
    mine = prereg_path(run)
    if mine.exists():
        results["prereg"] = read_json(mine)
    results["preregistrations"] = list_preregistrations(run)
    results["test_scorings"] = _scorings_note(results["preregistrations"])
    figs: dict[str, Any] = {}
    try:
        figs = make_figures(run, results)
    except Exception as e:
        figs = {"error": f"{type(e).__name__}: {e}"}
    results["figures"] = figs
    write_json(run.workdir / "results.json", results)
    zpath = run.workdir / "world_model_results.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(run.workdir / "results.json", "results.json")
        for sub in ("outputs", "figures"):
            for f in sorted((run.workdir / sub).glob("*")):
                z.write(f, f"{sub}/{f.name}")
        for f in sorted((run.workdir / "data").glob("*.json")):
            z.write(f, f"data_reports/{f.name}")
        for f in sorted(prereg_dir(run).glob("*.json")):
            z.write(f, f"prereg/{f.name}")
        for f in ("prereg.json", "run.log", "ledger.json", "preset.json"):
            if (run.workdir / f).exists():
                z.write(run.workdir / f, f)
    run.log(f"[package] results at {run.workdir / 'results.json'}; zip at {zpath}")
    return {"results": str(run.workdir / "results.json"), "zip": str(zpath), "figures": figs}


# ---------------------------------------------------------------- figures
_STYLE = {
    "SURFACE": "#fcfcfb", "INK": "#0b0b0b", "INK_2": "#52514e", "MUTED": "#898781",
    "GRID": "#e1e0d9", "AXIS": "#c3c2b7", "BLUE": "#2a78d6", "ORANGE": "#eb6834",
    "AQUA": "#1baf7a", "VIOLET": "#6250d6", "NEUTRAL": "#c3c2b7",
}
_PRED_COLORS = {"ours_simulation": _STYLE["BLUE"], "nflverse_home_wp (theirs)": _STYLE["ORANGE"],
                "nflverse_vegas_home_wp (theirs)": _STYLE["VIOLET"]}
_ACTION_COLORS = {"GO": _STYLE["BLUE"], "FG": _STYLE["ORANGE"], "PUNT": _STYLE["AQUA"]}


def _apply_style(plt) -> None:
    s = _STYLE
    plt.rcParams.update({
        "figure.facecolor": s["SURFACE"], "axes.facecolor": s["SURFACE"],
        "savefig.facecolor": s["SURFACE"], "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"], "font.size": 10.5,
        "text.color": s["INK"], "axes.edgecolor": s["AXIS"], "axes.labelcolor": s["INK_2"],
        "axes.titlecolor": s["INK"], "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": s["GRID"],
        "xtick.color": s["MUTED"], "ytick.color": s["MUTED"], "legend.frameon": False,
        "lines.linewidth": 2.0, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def make_figures(run: Run, results: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style(plt)
    fdir = run.workdir / "figures"
    made: dict[str, str] = {}
    st = results["stages"]

    hist = (st.get("train", {}).get("history") or {}).get("history") or []
    series = {
        "train (nats per record)": [h.get("train_loss") for h in hist],
        "validation (nats per record)": [h.get("val_total") for h in hist],
    }
    if hist:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = (_STYLE["NEUTRAL"], _STYLE["BLUE"])
        for (k, v), c in zip(series.items(), colors, strict=True):
            ep = [h.get("epoch", i) + 1 for i, h in enumerate(hist)]
            ax.plot(ep, v, label=k, color=c, marker="o", ms=3)
        base = (st.get("eval", {}).get("nll") or {}).get("baseline_freq_phase_down", {})
        if base.get("total_per_transition"):
            ax.axhline(base["total_per_transition"], color=_STYLE["ORANGE"], lw=1.2, ls="--",
                       label="frequency baseline, validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel("NLL, nats per record (sum over fields)")
        ax.set_title("Training curves")
        ax.legend()
        made["training_curves"] = _save(fig, fdir / "training_curves.png", plt)

    panels = [(name, st[name]["wp"]) for name in ("eval", "test")
              if st.get(name, {}).get("status") in ("done", "partial") and "wp" in st[name]]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5), squeeze=False)
        for ax, (name, wp) in zip(axes[0], panels, strict=True):
            ax.plot([0, 1], [0, 1], color=_STYLE["AXIS"], lw=1, ls="--")
            for k, m in wp["metrics"].items():
                r = m["reliability"]
                xs = [a for a, b in zip(r["bin_mean_pred"], r["bin_mean_outcome"], strict=True)
                      if a is not None and b is not None]
                ys = [b for a, b in zip(r["bin_mean_pred"], r["bin_mean_outcome"], strict=True)
                      if a is not None and b is not None]
                ll = m["log_loss"]["mean"]
                ax.plot(xs, ys, marker="o", color=_PRED_COLORS.get(k, _STYLE["INK_2"]),
                        label=f"{k}, log loss {ll:.3f}")
            label = "validation" if name == "eval" else "test"
            ax.set_title(f"Home win probability calibration ({label}, {wp['n_states']} states)")
            ax.set_xlabel("predicted")
            ax.set_ylabel("observed (regulation result)")
            ax.legend(fontsize=8.5, loc="upper left")
        made["calibration"] = _save(fig, fdir / "calibration.png", plt)

    nll = st.get("eval", {}).get("nll")
    if nll:
        names = [n for n, v in nll["per_field"].items()
                 if v["model"] is not None and v["baseline"] is not None]
        if names:
            x = np.arange(len(names))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x - 0.2, [nll["per_field"][n]["baseline"] for n in names], 0.4,
                   color=_STYLE["NEUTRAL"], label="frequency baseline (phase, down)")
            ax.bar(x + 0.2, [nll["per_field"][n]["model"] for n in names], 0.4,
                   color=_STYLE["BLUE"], label="world model")
            ax.set_xticks(x, names, rotation=30, ha="right")
            ax.set_ylabel("NLL (nats), validation")
            ax.set_title("Per-field NLL: world model vs frequency baseline")
            ax.legend()
            made["nll_vs_baseline"] = _save(fig, fdir / "nll_vs_baseline.png", plt)

    plan = st.get("plan", {})
    if plan.get("status") in ("done", "partial") and plan.get("hypothetical"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        res = plan["hypothetical"]["result"]
        acts = [("GO", "wp_go", "se_go"), ("FG", "wp_fg", "se_fg"),
                ("PUNT", "wp_punt", "se_punt")]
        ax = axes[0]
        for k, (a, w, s) in enumerate(acts):
            v = (res.get(w) or [None])[0]
            if v is None:  # masked by a hard rule, or not finished in time
                allowed = (res.get(f"allowed_{a.lower()}") or [True])[0]
                ax.text(k, 0.02, "not allowed\n(hard rule)" if allowed is False else "not run",
                        ha="center", va="bottom", fontsize=8.5, color=_STYLE["MUTED"])
                continue
            se = (res.get(s) or [0])[0]
            ax.bar(k, v, yerr=1.96 * (se or 0), color=_ACTION_COLORS[a], capsize=4)
            ax.text(k, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(acts)), [a for a, _, _ in acts])
        ax.set_ylabel("simulated win probability, offense")
        ax.set_title("Up 4, 4:00 left, 4th and 2 at own 40", fontsize=11)
        ag = plan.get("agreement", {})
        ax = axes[1]
        conf = ag.get("confusion_recommended_by_real", {})
        reals = sorted({r for v in conf.values() for r in v})
        recs = sorted(conf)
        bottom = np.zeros(len(recs))
        for r in reals:
            h = np.array([conf[b].get(r, 0) for b in recs], dtype=float)
            ax.bar(recs, h, bottom=bottom, label=f"coach: {r}",
                   color=_ACTION_COLORS.get(r, _STYLE["NEUTRAL"]))
            bottom += h
        a_all = ag.get("agreement_all")
        ax.set_title(f"Model recommendation vs coach ({ag.get('n', 0)} test 4th downs"
                     + (f", agree {a_all:.0%})" if a_all is not None else ")"), fontsize=11)
        ax.set_xlabel("model recommends")
        ax.set_ylabel("fourth downs")
        ax.legend()
        made["fourth_down"] = _save(fig, fdir / "fourth_down.png", plt)
    diag = (plan.get("support") or {}).get("diagnostic") or {}
    if plan.get("status") in ("done", "partial") and diag.get("fg"):
        made["fg_support"] = _fg_support_figure(diag, plan["support"]["rules"],
                                                fdir / "fg_support.png", plt)
    return made


def _band_series(rows: list[dict], key: str) -> np.ndarray:
    return np.array([np.nan if r.get(key) is None else float(r[key]) for r in rows])


def _fg_support_figure(diag: dict, rules: dict, path: Path, plt) -> str:
    """Simulated vs training make rate by distance (left) and punt net yards by yard line
    (right), each over the number of training attempts in the band. Shaded: masked."""
    s = _STYLE
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.08,
                                          "wspace": 0.22})
    panels = (
        ("fg", "distance_yards", "train_attempts", "empirical_make_rate", "sim_make_rate",
         "Field goals: simulated vs training make rate", "make rate",
         "field goal distance (yards) = yard line + 18", "training\nattempts"),
        ("punt", "yardline", "train_punts", "empirical_net_yards", "sim_net_yards",
         "Punts: simulated vs training net yards", "net yards",
         "yard line of the punt (yards from the opponent's end zone)", "training\npunts"),
    )
    for col, (key, xkey, nkey, ekey, skey, title, ylab, xlab, nlab) in enumerate(panels):
        rows = diag.get(key) or []
        if not rows:
            continue
        x = np.array([(r[xkey][0] + r[xkey][1]) / 2 for r in rows])
        width = rows[0][xkey][1] - rows[0][xkey][0] + 1
        emp, sim, se = (_band_series(rows, k) for k in (ekey, skey, "sim_se"))
        ax = axes[0, col]
        if key == "fg":
            lo, hi = rules["fg_max_distance_yards"] + 0.5, x.max() + width / 2
            label = f"not allowed:\nbeyond {rules['fg_max_distance_yards']} yd"
            ax.set_ylim(0, 1.04)
        else:
            lo, hi = x.min() - width / 2, rules["punt_min_yardline"] - 0.5
            label = f"not allowed:\ninside the\nopponent's {rules['punt_min_yardline']}"
        if hi > lo:
            ax.axvspan(lo, hi, color=s["GRID"], alpha=0.7, lw=0)
            ax.text(lo + 0.6, 0.97, label, transform=ax.get_xaxis_transform(), va="top",
                    fontsize=8.5, color=s["INK_2"])
        ok = np.isfinite(emp)
        ax.plot(x[ok], emp[ok], color=s["ORANGE"], marker="o", ms=6,
                label="training seasons: what coaches' kicks did")
        ok = np.isfinite(sim)
        ax.errorbar(x[ok], sim[ok], yerr=1.96 * np.nan_to_num(se[ok]), color=s["BLUE"],
                    marker="s", ms=6, capsize=3,
                    label="world model: forced kick, simulated next play (95% interval)")
        ax.set_title(title, pad=8)
        ax.set_ylabel(ylab)
        ax = axes[1, col]
        n = np.array([r[nkey] or 0 for r in rows])
        ax.bar(x, n, width=width * 0.85, color=s["NEUTRAL"])
        for xi, ni in zip(x, n, strict=True):
            ax.text(xi, ni, f"{ni}", ha="center", va="bottom", fontsize=8, color=s["INK_2"])
        ax.set_ylim(0, max(int(n.max()), 1) * 1.35)
        ax.set_ylabel(nlab)
        ax.set_xlabel(xlab)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=2,
               fontsize=9)
    fig.text(0.01, -0.03,
             f"Simulated: up to {diag.get('states_per_band')} test-season fourth downs per band "
             f"(whatever the coach did), the kick forced, {diag.get('draws_per_state')} sampled "
             "next plays each.\nTraining: every attempt in the training seasons. Coaches pick "
             "when to kick, so long-range training rates are for selected attempts.",
             fontsize=8.5, color=s["MUTED"], va="top")
    return _save(fig, path, plt)


def _save(fig, path: Path, plt) -> str:
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def split_arrays_hash(split: S.Split, extra: dict | None = None) -> str:
    """Content hash of a split's arrays (and any extra aligned arrays)."""
    h = hashlib.sha256()
    for k in _SPLIT_ARRAYS:
        a = np.ascontiguousarray(getattr(split, k))
        h.update(f"{k}{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    h.update("\x1f".join(split.game_ids).encode())
    for k in sorted(extra or {}):
        a = np.ascontiguousarray(extra[k])
        h.update(f"{k}{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _build_season(rec, season: int) -> tuple[S.Split, dict, dict]:
    pbp = rec.load_pbp(check_seasons([season]))
    sched = rec.load_schedules(check_seasons([season]))
    sp, report = rec.build_split(pbp, sched, [season])
    extra = align_nflverse_wp(sp, _pbp_wp_columns(pbp))
    del pbp
    return sp, report, extra


def chunked(run: Run, pdir: Path, fn, idx: np.ndarray, chunk: int, deadline: float,
            seed: int) -> tuple[Any, dict]:
    """Run fn(idx_part, rng) -> DataFrame over fixed chunks of idx, saving each finished
    chunk under pdir and reusing saved chunks on the next run. Each chunk has its own seed,
    so a resumed result is the same as an uninterrupted one. Stops before a chunk that
    would not finish by the deadline."""
    import pandas as pd

    pdir.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(idx, dtype=np.int64)
    n_chunks = int(np.ceil(len(idx) / max(chunk, 1)))
    frames, counters = [], []
    reused = computed = 0
    per_state, stopped = None, None
    for k in range(n_chunks):
        part = idx[k * chunk:(k + 1) * chunk]
        f = pdir / f"chunk_{k:04d}.pkl"
        if f.exists():
            frames.append(pd.read_pickle(f))
            cf = f.with_suffix(".json")
            counters.append(read_json(cf) if cf.exists() else None)
            reused += len(part)
            continue
        left = deadline - time.time()
        if left <= 0 or (per_state is not None and left < 1.1 * per_state * len(part)):
            stopped = "time budget"
            break
        t0 = time.time()
        df = fn(part, np.random.default_rng([seed, k]))
        per_state = (time.time() - t0) / max(len(part), 1)
        complete = "done" not in df.columns or bool(df["done"].astype(bool).all())
        frames.append(df)
        counters.append(_jsonable(df.attrs.get("counters")))
        computed += len(part)
        if complete:
            tmp = f.with_suffix(".tmp")
            df.to_pickle(tmp)
            tmp.replace(f)
            write_json(f.with_suffix(".json"), counters[-1])
        else:
            stopped = "deadline reached inside a chunk"
            break
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    info = {"planned_states": int(len(idx)), "chunk_states": int(chunk),
            "chunks": n_chunks, "reused_states": reused, "computed_states": computed,
            "stopped": stopped, "complete": stopped is None,
            "seconds_per_state": None if per_state is None else round(per_state, 4),
            "rollout_counters": counters}
    return df, info


def _chunk_size(run: Run, n_rollouts: int, key: str) -> int:
    return _chunk_states(run.preset, n_rollouts, key)


def _chunk_states(preset: dict, n_rollouts: int, key: str) -> int:
    """States per saved chunk of a stage (n_rollouts = rollouts per state per chunk call)."""
    c = preset[key]
    if c.get("chunk_states"):
        return int(c["chunk_states"])
    live = int(preset["train"].get("sim_max_live", 8192))
    return max(4, live // max(int(n_rollouts), 1))


def _cached_json(path: Path, make):
    if path.exists():
        return read_json(path)
    val = _jsonable(make())
    write_json(path, val)
    return val


def _check_checkpoint(path: Path, model) -> str | None:
    """None if there is no checkpoint or it fits the model; else why it does not."""
    if not path.exists():
        return None
    try:
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict):
            return "not a checkpoint dict"
        saved = ck.get("current_model") or ck.get("model")
        if not isinstance(saved, dict):
            return None  # not ours to judge (a stub or another format)
        cur = model.state_dict()
        missing = [k for k in cur if k not in saved]
        extra = [k for k in saved if k not in cur]
        shape = [k for k in cur if k in saved and tuple(saved[k].shape) != tuple(cur[k].shape)]
        if missing or extra or shape:
            return (f"{len(missing)} missing, {len(extra)} unexpected, {len(shape)} "
                    "shape-mismatched tensors")
    except Exception as e:
        return f"unreadable: {type(e).__name__}: {e}"
    return None


def checkpoint_info(run: Run) -> dict:
    path = run.ckpt_path()
    info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        try:
            import torch

            ck = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ck, dict):
                info["epochs_done"] = ck.get("epoch")
                info["best_val_nll"] = ck.get("best_val")
                info["finished"] = ck.get("finished")
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
    return info


def status_table(run: Run) -> str:
    """Which stages are done, the checkpoint, its last epoch and best validation NLL, and
    minutes used so far."""
    lines = [f"Work folder {run.workdir}", f"{'stage':<8} {'state':<12} seconds"]
    for s in STAGES[:-1]:
        if run.is_done(s):
            state = "done"
        else:
            state = run.status(s)
            state = "not started" if state == "missing" else state
        secs = read_json(run.output_path(s)).get("seconds", "") \
            if run.output_path(s).exists() else ""
        lines.append(f"{s:<8} {state:<12} {secs}")
    if run.content_hash("train") != "missing":
        ck = checkpoint_info(run)
        if ck["exists"]:
            best = ck.get("best_val_nll")
            best_s = f"{best:.4f}" if isinstance(best, (int, float)) else "n/a"
            lines.append(f"checkpoint {ck['path']}: {ck.get('epochs_done')} epochs done, best "
                         f"validation NLL {best_s} nats/record"
                         + (", training finished" if ck.get("finished") else ""))
        else:
            lines.append(f"checkpoint {ck['path']}: none yet")
    lines.append(f"minutes used so far {run.consumed_before / 60:.1f} of "
                 f"{run.total_s / 60:.0f}; {run.remaining() / 60:.1f} left this session")
    return "\n".join(lines)


def stopped_message(stage: str) -> str:
    return f"Stopped safely at {stage}. Run all again to continue from here."


def _beyond(stage: str, stop_after: str | None) -> bool:
    if stop_after is None or stage == "package":
        return False
    if stop_after not in STAGES:
        raise ValueError(f"STOP_AFTER must be None or one of {STAGES[:-1]}")
    return STAGES.index(stage) > STAGES.index(stop_after)


def notebook_stage(run: Run, stage: str, stop_after: str | None = None) -> dict:
    """One notebook cell's worth of work. Honors STOP_AFTER; on the stop button, saves what
    exists, writes results.json and re-raises so Colab's Run all stops here."""
    if _beyond(stage, stop_after):
        msg = (f"[{stage}] not run: STOP_AFTER = {stop_after!r}. Set STOP_AFTER = None and "
               "Run all to continue.")
        print(msg)
        return {"status": "not_run", "reason": msg}
    try:
        return run_stage(run, stage)
    except KeyboardInterrupt:
        _safe_package(run)
        print(stopped_message(stage), flush=True)
        raise


def _safe_package(run: Run) -> None:
    try:
        run_stage(run, "package")
    except BaseException as e:  # never let packaging hide the original stop
        run.log(f"[package] could not package after the stop: {e}")



STAGE_FUNCS = {
    "data": stage_data,
    "train": stage_train,
    "eval": stage_eval,
    "test": stage_test,
    "plan": stage_plan,
    "package": stage_package,
}


# ---------------------------------------------------------------- preflight
def system_info() -> dict:
    info = versions()
    try:
        import shutil

        du = shutil.disk_usage(default_workdir().parent if default_workdir().parent.exists()
                               else "/")
        info["disk_free_gb"] = round(du.free / 2**30, 1)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            kb = int(next(line for line in f if line.startswith("MemTotal")).split()[1])
        info["ram_gb"] = round(kb / 2**20, 1)
    except Exception:
        pass
    return info


def _bench_states(syn: S.Split, n: int) -> np.ndarray:
    """n start records near the middle of synthetic games (cycling through the games), so
    the shared history is about as long as at a real mid-game fourth down."""
    lens = np.diff(syn.offsets)
    g = np.arange(n) % len(lens)
    return np.asarray(syn.offsets[g] + lens[g] // 2, dtype=np.int64)


def apply_benchmark(preset: dict, rec_per_s: float, units: float):
    """The benchmark's rule, as a pure function of the measured speeds. Returns (adjusted
    preset, list of changes, decision text, train speed share, rollout speed share)."""
    a = preset["assumed"]
    tr_ratio = rec_per_s / a["train_rec_per_s"]
    ro_ratio = units / a["rollout_units_per_s"]
    adjusted = copy.deepcopy(preset)
    changes: list[str] = []
    downgraded = None
    if preset["device"] == "cuda" and tr_ratio < BENCH_SLOW_SHARE \
            and DOWNGRADE[preset["name"]] != preset["name"]:
        smaller = PRESETS[DOWNGRADE[preset["name"]]]
        adjusted["model"] = copy.deepcopy(smaller["model"])
        adjusted["train"]["batch_games"] = smaller["train"]["batch_games"]
        downgraded = (f"model downgraded to the {smaller['name']} size (training at "
                      f"{tr_ratio:.0%} of the assumed speed)")
        changes.append(downgraded)
    if ro_ratio < BENCH_SLOW_SHARE:
        f = ro_ratio
        for block, key in (("eval", "wp_states"), ("eval", "sim_games"), ("test", "wp_states"),
                           ("test", "sim_games"), ("plan", "n_fourth"),
                           ("plan", "hypo_rollouts")):
            old = int(adjusted[block][key])
            new = max(int(old * f), min(old, BENCH_FLOORS[key]))
            if new != old:
                adjusted[block][key] = new
                changes.append(f"{block}.{key} {old} -> {new}")
        decision = (f"scaled down: rollouts at {units:,.0f} steps/s are {ro_ratio:.0%} of the "
                    f"{a['rollout_units_per_s']:,} the preset assumes (below the "
                    f"{BENCH_SLOW_SHARE:.0%} threshold), so sizes were multiplied by {f:.2f} "
                    f"with floors {BENCH_FLOORS}")
    else:
        decision = (f"kept the preset sizes: rollouts at {units:,.0f} steps/s are "
                    f"{ro_ratio:.0%} of the {a['rollout_units_per_s']:,} the preset assumes "
                    f"(it is scaled only below {BENCH_SLOW_SHARE:.0%})")
    if downgraded:
        decision += "; " + downgraded
    return adjusted, changes, decision, tr_ratio, ro_ratio


def micro_benchmark(preset: dict, seconds: float = 30.0) -> dict:
    """Measure training and rollout speed on this machine with the code the stages use, and
    scale the preset DOWN only if this machine is much slower than the preset assumes.

    Training: two short epochs on a synthetic split (2 x batch_games games on a GPU); the
    speed is read from the second epoch, after warm-up. Rollouts: the current rollout path
    (the key/value cache) at the batch one plan-stage call uses on this GPU (states per
    chunk x n_rollouts), from mid-game states. The model is untrained, so its END phase is
    switched off and max_steps is STEPS_PER_ROLLOUT: every rollout runs about as long as a
    real one, and the speed is rollout steps (transitions) per second.

    Rule: a size is scaled only when the measured speed is below BENCH_SLOW_SHARE of the
    preset's assumed speed, by the measured share, never below BENCH_FLOORS (or the preset's
    own value if smaller). Otherwise the preset is kept as written; every stage still runs
    against its own deadline and saves partial progress. The synthetic split is not
    football; it only fixes tensor shapes."""
    import torch

    model_mod, train_mod, sim = _mod("model"), _mod("train"), _mod("simulate")
    device = "cuda" if preset["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    cfg = {**preset["model"], **preset["train"], "device": device, "seed": SEED, "epochs": 2}
    cfg.pop("max_train_games", None)
    n_games = 2 * int(cfg["batch_games"]) if device == "cuda" else 6
    syn = synthetic_split(n_games=n_games, seed=1)
    model = model_mod.WorldModel(cfg)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"wm_bench_{os.getpid()}_{time.time_ns()}.pt"
    tmp.unlink(missing_ok=True)
    t0 = time.time()
    try:
        hist = train_mod.train_model(model, syn, subset_split(syn, [0, 1]), cfg, str(tmp),
                                     time.time() + seconds / 2, lambda *a: None)
    finally:
        tmp.unlink(missing_ok=True)
    train_s = time.time() - t0
    warm = [h for h in hist.get("history") or [] if h.get("epoch", 0) >= 1]
    rec_per_s = (warm[-1].get("timing") or {}).get("train_records_per_s") if warm else None
    train_basis = "second epoch (after warm-up)"
    if not rec_per_s or not np.isfinite(rec_per_s):
        rec_per_s = hist.get("train_records_per_s")
        train_basis = "all steps, including warm-up"
    if not rec_per_s or not np.isfinite(rec_per_s):
        rec_per_s = n_records(syn) / max(train_s, 1e-6)
        train_basis = "wall time of the whole call"
    rec_per_s = float(rec_per_s)

    # ---- rollouts: the current path, at one plan-stage call's batch
    if hasattr(model, "eval"):
        model.eval()
    heads = getattr(model, "heads", None)
    if heads is not None:
        with torch.no_grad():  # untrained: never end the game early by sampling END
            heads[S.TARGET_INDEX["phase"]][-1].bias[S.PHASE["END"]] = -1e4
    rcfg = {**cfg, "kv_cache": True, "max_steps": STEPS_PER_ROLLOUT}
    n_roll = int(preset["plan"]["n_rollouts"])
    live = int(sim.live_rows(model, rcfg)) if hasattr(sim, "live_rows") else n_roll
    states = max(1, min(_chunk_states(preset, 3 * n_roll, "plan"), live // max(n_roll, 1)))
    idx = _bench_states(syn, states)
    rng = np.random.default_rng(0)
    sim.rollout(model, syn, idx[:1], None, 16, {**rcfg, "max_steps": 3}, time.time() + 30,
                rng)  # warm-up: first-call kernel and allocator setup
    planned = states * n_roll * STEPS_PER_ROLLOUT
    t0 = time.time()
    res = sim.rollout(model, syn, idx, None, n_roll, rcfg, t0 + seconds / 2, rng)
    roll_s = time.time() - t0
    finished = bool(np.all(res["done"]))
    if finished:
        units = float(res["counters"]["transitions"]) / max(roll_s, 1e-6)
        speed_note = "measured"
    else:  # did not finish in its time: at most this fast
        units = planned / max(roll_s, 1e-6)
        speed_note = "upper bound (the batch did not finish in its time)"
    adjusted, changes, decision, tr_ratio, ro_ratio = apply_benchmark(preset, rec_per_s, units)

    def minutes(p: dict) -> dict:
        out = {}
        for key in ("eval", "test"):
            c = p[key]
            u = c["wp_states"] * c["wp_rollouts"] * STEPS_PER_ROLLOUT \
                + c["sim_games"] * 2 * STEPS_PER_ROLLOUT
            out[key] = round(u / max(units, 1e-6) / 60, 1)
        c = p["plan"]
        u = 3 * (c["n_fourth"] * c["n_rollouts"] + c["hypo_rollouts"]) * STEPS_PER_ROLLOUT
        out["plan"] = round(u / max(units, 1e-6) / 60, 1)
        return out

    timing = {k: (round(v, 3) if isinstance(v, float) else v)
              for k, v in (res.get("timing") or {}).items() if k != "method"}
    return {
        "device": device,
        "train_records_per_s": round(rec_per_s, 1),
        "train_speed_basis": train_basis,
        "rollout_steps_per_s": round(units, 1),
        "rollout_speed_basis": speed_note,
        "rollout_batch": {"states": int(states), "rollouts_per_state": n_roll,
                          "rows": int(states * n_roll), "live_rows_cap": live,
                          "max_steps": STEPS_PER_ROLLOUT, "path": "kv_cache",
                          "seconds": round(roll_s, 2), "timing": timing},
        "assumed": dict(preset["assumed"]),
        "train_speed_vs_assumed": round(tr_ratio, 2),
        "rollout_speed_vs_assumed": round(ro_ratio, 2),
        "threshold_share": BENCH_SLOW_SHARE,
        "floors": dict(BENCH_FLOORS),
        "decision": decision,
        "changes": changes,
        "estimated_minutes_at_measured_speed": {"preset": minutes(preset),
                                                "after_adjustment": minutes(adjusted)},
        "stage_ceilings_min": dict(preset["stage_min"]),
        "note": "rollout speed from an untrained model with END switched off on synthetic "
                "games; each stage also times its own chunks and stops at its deadline, "
                "saving what it finished",
        "adjusted_preset": adjusted,
    }


def estimate_minutes(preset: dict) -> dict:
    sm = preset["stage_min"]
    return {"budget_min": preset["budget_min"], "stage_ceilings_min": sm}


# Files of a run that a fresh preset pick starts over: moved into archive/ (results.json,
# the zip, the time ledger, preset.json, figures) or copied there (stage outputs, the legacy
# preregistration and the log, which stay in place: finished data is reused by hash).
_ARCHIVE_MOVE = ("results.json", "world_model_results.zip", "ledger.json", "preset.json")
_ARCHIVE_COPY = ("prereg.json", "run.log")


def archive_previous_run(wd: Path, reason: str) -> Path | None:
    """Keep the previous run's results before a fresh preset pick. Moves results.json, the
    zip, figures/, preset.json and ledger.json (so the time budget starts again) into
    archive/run_<time>/, and copies outputs/, prereg.json and run.log there. Data, the
    checkpoints and the per-configuration preregistrations stay where they are."""
    import shutil

    wd = Path(wd)
    present = [n for n in _ARCHIVE_MOVE + _ARCHIVE_COPY if (wd / n).exists()]
    figs = sorted((wd / "figures").glob("*")) if (wd / "figures").is_dir() else []
    outs = sorted((wd / "outputs").glob("*")) if (wd / "outputs").is_dir() else []
    if not (present or figs or outs):
        return None
    dest = wd / "archive" / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    k = 1
    while dest.exists():
        dest = dest.with_name(dest.name.split("__")[0] + f"__{k}")
        k += 1
    dest.mkdir(parents=True)
    for n in present:
        (shutil.move if n in _ARCHIVE_MOVE else shutil.copy2)(str(wd / n), str(dest / n))
    if figs:
        (dest / "figures").mkdir()
        for f in figs:
            shutil.move(str(f), str(dest / "figures" / f.name))
    if outs:
        shutil.copytree(wd / "outputs", dest / "outputs")
    write_json(dest / "ARCHIVED.json", {"archived_at": time.time(), "reason": reason})
    return dest


def prepare(preset_name: str | None = None, workdir: Path | None = None, benchmark: bool = True,
            budget_min: float | None = None, force_new_preset: bool = False,
            allow_rescore: bool = False, force: tuple = ()) -> Run:
    """Pick (or reload) the preset, benchmark it once, print the status table, return a Run.

    The (possibly benchmark-adjusted) preset is saved in the work folder with its definition
    (PRESET_VERSION and a hash of its PRESETS entry) and reused on every resume, so a
    resumed run keeps the same config hashes and skips finished stages. When the saved
    definition differs from the code (or force_new_preset), the preset is picked afresh:
    the previous run's results are archived and the time ledger starts again.
    """
    wd = Path(workdir or default_workdir())
    wd.mkdir(parents=True, exist_ok=True)
    saved = wd / "preset.json"
    notes: list[str] = []
    preset = None
    if saved.exists():
        old = read_json(saved)
        name_old = old.get("name")
        cur = preset_definition(name_old) if name_old in PRESETS else None
        if force_new_preset:
            reason = "force_new_preset was set"
        elif cur is None:
            reason = f"its preset name {name_old!r} is not in this code"
        elif old.get("definition") is None:
            reason = "it has no definition version (saved by the first notebook)"
        elif old["definition"] != cur:
            reason = (f"its definition (version {old['definition'].get('version')!r}, hash "
                      f"{old['definition'].get('hash')}) differs from the code's (version "
                      f"{cur['version']!r}, hash {cur['hash']})")
        else:
            reason = None
            preset = old
            notes.append(f"Reusing the saved preset '{name_old}' from {saved} (definition "
                         f"hash {cur['hash']})")
            want = preset_name or pick_preset()
            if want != name_old:
                notes.append(f"Note: this machine would get preset '{want}'; the saved "
                             f"'{name_old}' is kept so finished stages resume. Set "
                             "FORCE_NEW_PRESET = True to start over with a fresh pick.")
        if reason:
            dest = archive_previous_run(wd, f"fresh preset pick: {reason}")
            notes.append(f"The saved preset in {saved} is replaced: {reason}. Picking the "
                         "preset afresh"
                         + (f"; the previous run's results, figures and time ledger were "
                            f"moved to {dest}" if dest else ""))
    elif (wd / "ledger.json").exists() or (wd / "results.json").exists():
        dest = archive_previous_run(wd, "fresh preset pick: no preset.json, but an earlier "
                                        "run's ledger or results")
        notes.append(f"No saved preset, but an earlier run's files were found; moved them to "
                     f"{dest} so this run starts with the full time budget")
    if preset is None:
        name = preset_name or pick_preset()
        preset = get_preset(name)
        preset["definition"] = preset_definition(name)
        if benchmark:
            try:
                bench = micro_benchmark(preset)
                preset = bench.pop("adjusted_preset")
                preset["benchmark"] = bench
                notes.append(f"Micro-benchmark: {bench['decision']}")
                print("Micro-benchmark:", json.dumps(
                    {k: v for k, v in bench.items() if k != "rollout_batch"}, indent=1))
            except Exception as e:
                preset["benchmark"] = {"error": f"{type(e).__name__}: {e}",
                                       "decision": "kept the preset as is (benchmark failed)"}
                notes.append(f"Micro-benchmark failed, keeping the preset as is: {e}")
        preset["picked"] = {"at": time.time(), "notes": list(notes)}
        write_json(saved, preset)
    if budget_min is not None:
        preset["budget_min"] = budget_min
    run = Run(preset, wd, budget_min=budget_min, allow_rescore=allow_rescore, force=force)
    for n in notes:
        run.log(f"[preflight] {n}")
    print(status_table(run), flush=True)
    return run


def run_all(run: Run, stages=STAGES, stop_after: str | None = None) -> dict:
    """Run the stages in order (up to stop_after), then package. A KeyboardInterrupt or an
    exception in a stage still ends with results.json; an interrupt prints where it
    stopped."""
    out: dict[str, Any] = {}
    current = None
    try:
        for s in stages:
            if s == "package" or _beyond(s, stop_after):
                continue
            current = s
            out[s] = run_stage(run, s)
    except KeyboardInterrupt:
        _safe_package(run)
        out["stopped_at"] = current
        msg = stopped_message(current)
        run.log(msg)
        if run.quiet:
            print(msg, flush=True)
        return out
    out["package"] = run_stage(run, "package")
    return out


def summary(run: Run) -> str:
    """Plain-language summary of results.json."""
    path = run.workdir / "results.json"
    if not path.exists():
        return "No results.json yet."
    r = read_json(path)
    st = r["stages"]
    lines = [f"Preset {r['preset']['name']}, config {r['config_hash']}, "
             f"GPU {r['versions'].get('gpu', 'none')}."]
    bench = r.get("benchmark") or {}
    if bench.get("decision"):
        lines.append(f"  preflight benchmark: {bench['decision']}")
    ts = r.get("test_scorings") or {}
    if ts.get("n_configurations_scored", 0) > 1:
        lines.append(f"  {ts['note']} (config hashes {', '.join(ts['config_hashes_scored'])}; "
                     "see results.json 'preregistrations')")
    for s in ("data", "train", "eval", "test", "plan"):
        o = st.get(s, {})
        lines.append(f"- {s}: {o.get('status')}"
                     + (f" ({o.get('reason') or o.get('error')})"
                        if o.get("status") != "done" else f" in {o.get('seconds')}s"))
    d = st.get("data", {})
    if d.get("status") == "done":
        for k, v in d["splits"].items():
            lines.append(f"  {k}: {v['games']} games, {v['records']} records, "
                         f"seasons {v['seasons'][0]}-{v['seasons'][-1]}")
    t = st.get("train", {})
    if t.get("status") == "done":
        h = t.get("history") or {}
        speed = h.get("train_records_per_s")
        lines.append(f"  model: {t['params']:,} parameters, trained {t['train_seconds_total']}s"
                     + (f" on {h['device']}" if h.get("device") else "")
                     + (f" ({h.get('epochs_done')} epochs, {speed:,.0f} records/s)"
                        if isinstance(speed, (int, float)) and np.isfinite(speed) else ""))
    for s, label in (("eval", "validation"), ("test", "test")):
        o = st.get(s, {})
        if o.get("status") not in ("done", "partial") or not o.get("wp", {}).get("metrics"):
            continue
        b = o["nll"]["baseline_freq_phase_down"]["total_per_transition"]
        lines.append(f"  {label} NLL per transition: model {_total_nll(o['nll']['model'])}"
                     f" vs frequency baseline {b:.3f} (lower is better)")
        for k, m in o["wp"]["metrics"].items():
            ll = m["log_loss"]
            lines.append(f"  {label} win probability, {k}: log loss {ll['mean']:.4f} "
                         f"(95% game bootstrap {ll['lo95']:.4f} to {ll['hi95']:.4f}), "
                         f"Brier {m['brier']['mean']:.4f}, {ll['n_states']} states")
    p = st.get("plan", {})
    if p.get("status") in ("done", "partial") and p.get("agreement", {}).get("n"):
        ag = p["agreement"]
        lines.append(f"  fourth downs: model agrees with the coach on "
                     f"{ag.get('agreement_all', 0):.0%} of {ag.get('n')} test decisions"
                     + (f", {ag['agreement_clear']:.0%} of the {ag['n_clear']} that are not "
                        "too close to call" if ag.get("agreement_clear") is not None else ""))
        res = p["hypothetical"]["result"]
        bits = [f"{a} {res[k][0]:.3f}" for a, k in (("go", "wp_go"), ("field goal", "wp_fg"),
                                                     ("punt", "wp_punt"))
                if res.get(k) and res[k][0] is not None]
        bits += [f"{a} not allowed (hard rule)" for a, k in (("go", "allowed_go"),
                                                           ("field goal", "allowed_fg"),
                                                           ("punt", "allowed_punt"))
                 if (res.get(k) or [True])[0] is False]
        lines.append(f"  hypothetical (up 4, 4:00 left, 4th and 2 at own 40): "
                     f"{', '.join(bits)}; best {res.get('best', ['?'])[0]}")
    if p.get("status") in ("done", "partial"):
        lines += _support_summary_lines(p)
    return "\n".join(lines)


def _fmt(v, spec: str = ".2f") -> str:
    return "none" if v is None else format(v, spec)


def _pct(a: dict) -> str:
    v = a.get("agreement_all")
    return "n/a" if v is None else f"{v:.0%}"


def _support_summary_lines(p: dict) -> list[str]:
    """The action-support part of the summary: rules, masked actions, low-support flags and
    the simulated-versus-training kicking diagnostic, in plain language."""
    sup = p.get("support") or {}
    rules = sup.get("rules")
    if not rules:
        return []
    lines = [f"  action rules (from the training seasons only): {rules['text']}"]
    ms = p.get("masked_summary") or {}
    if ms.get("planned_states"):
        lines.append(f"  masked: {ms['states_with_a_masked_action']} of {ms['planned_states']} "
                     f"sampled test fourth downs had an action ruled out (field goal on "
                     f"{ms['fg_masked']}, punt on {ms['punt_masked']}); each is listed under "
                     "stages.plan.masked_actions")
    ab = p.get("agreement_by_support") or {}
    ws, ls = ab.get("well_supported") or {}, ab.get("low_support") or {}
    n_ws, n_ls = ws.get("n", 0), ls.get("n", 0)
    if n_ws or n_ls:
        lines.append(f"  support: {n_ls} of {n_ws + n_ls} recommendations have low support (the "
                     f"coaches made that call fewer than {ab.get('min_support')} times in "
                     "comparable training fourth downs); agreement with the coach "
                     f"{_pct(ws)} on the {n_ws} well-supported, {_pct(ls)} on the {n_ls} "
                     "low-support")
    fr = p.get("fg_recommendations") or {}
    if fr.get("n"):
        bit = ""
        if fr.get("mean_sim_fg_make") is not None and fr.get("mean_emp_fg_make") is not None:
            bit = (f"; on them the simulator's make rate averages {fr['mean_sim_fg_make']:.2f} "
                   f"against {fr['mean_emp_fg_make']:.2f} for training kicks from the same "
                   "yard lines (within 2)")
        lines.append(f"  field-goal calls: {fr['n']} (coach: {fr.get('coach_choice_counts')})"
                     + bit)
    diag = sup.get("diagnostic") or {}
    if diag.get("error"):
        lines.append(f"  kicking diagnostic failed: {diag['error']}")
        return lines
    rows = [r for r in diag.get("fg") or [] if r.get("sim_make_rate") is not None]
    if rows:
        bits = [f"{r['distance_yards'][0]}-{r['distance_yards'][1]} yd {r['sim_make_rate']:.2f}"
                f" vs {_fmt(r['empirical_make_rate'])} ({r['train_attempts']})" for r in rows]
        lines.append("  field goals, simulated vs training make rate (training attempts): "
                     + "; ".join(bits))
        h = diag.get("fg_headline") or {}
        parts = []
        if h.get("weighted_mean_gap_with_data") is not None:
            a, b = h["largest_gap_band"]
            parts.append(f"where training has at least {h['min_attempts']} attempts the "
                         f"simulator is off by {h['weighted_mean_gap_with_data']:+.2f} on "
                         f"average (attempt-weighted), at most {h['largest_gap']:+.2f} at "
                         f"{a}-{b} yd")
        if h.get("sim_make_rate_thin_bands") is not None:
            parts.append(f"in thin bands it makes {h['sim_make_rate_thin_bands']:.2f}")
        if h.get("sim_make_rate_beyond_limit") is not None:
            parts.append(f"beyond the {rules['fg_max_distance_yards']}-yd limit it still makes "
                         f"{h['sim_make_rate_beyond_limit']:.2f}")
        if parts:
            lines.append("  " + "; ".join(parts))
    hf = diag.get("hypothetical_fg")
    if hf:
        lines.append(f"  the hypothetical's {hf['distance_yards']}-yard field goal: the simulator "
                     f"makes {hf['sim_make_rate']:.2f} of them (masked by the hard rule)")
    prow = [r for r in diag.get("punt") or [] if r.get("sim_net_yards") is not None]
    if prow:
        bits = [f"{r['yardline'][0]}-{r['yardline'][1]} {r['sim_net_yards']:.1f} vs "
                f"{_fmt(r['empirical_net_yards'], '.1f')} ({r['train_punts']})" for r in prow]
        lines.append("  punts, simulated vs training net yards by yard line (training punts): "
                     + "; ".join(bits))
    return lines


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--budget-min", type=float, default=None)
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument("--allow-rescore", action="store_true")
    ap.add_argument("--force-new-preset", action="store_true")
    ap.add_argument("--stop-after", choices=[s for s in STAGES if s != "package"],
                    default=None)
    ap.add_argument("--force", nargs="*", default=[], choices=list(STAGES[:-1]),
                    help="stages to recompute even if done")
    a = ap.parse_args(argv)
    run = prepare(a.preset, a.workdir, benchmark=not a.no_benchmark, budget_min=a.budget_min,
                  force_new_preset=a.force_new_preset, allow_rescore=a.allow_rescore,
                  force=tuple(a.force))
    out = run_all(run, stop_after=a.stop_after)
    print(summary(run))
    if out.get("stopped_at"):
        raise SystemExit(130)


if __name__ == "__main__":
    main()
