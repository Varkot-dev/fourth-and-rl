"""Research milestone R1, day 6 (run A): run the seeded grid of overoptimization cells.

A cell (key n<games>_<capacity>_s<seed>) draws n synthetic coach-vs-coach games with that
seed from the solved game, fits a LightGBM proxy of that capacity (8 or 64 leaves) and
grades it with one exact pass (overopt_curves.exact_pass): the exact true gain, sequence
KL and claimed gain at every beta. The cell "oracle" uses the exact coach values instead.
Planning, resume, parallelism and collect live in src/fourth_and_rl/overopt_grid.py.

The same code runs here and in notebooks/overopt_grid_colab.ipynb, with the same cell keys
and seeds, so results from both merge in `collect`. Split the work with --part: part K/M
takes every M-th cell of the priority-ordered plan, so each part keeps the priority order
and the parts are disjoint. Example (Colab runs two cells at a time, this machine one):

    Colab notebook:  PARTS = ["1/3", "2/3"]
    this machine:    .venv/bin/python scripts/research/overopt_grid.py run --part 3/3

The results merge only if both used the same solved game: upload this machine's data/mdp
to the notebook's Drive cache folder (MyDrive/fourth_and_rl_overopt_grid/game) first.

Run:  python scripts/research/overopt_grid.py plan [--part 3/3]
      python scripts/research/overopt_grid.py run --part 3/3 [--deadline-hours 5]
      python scripts/research/overopt_grid.py cell n250_large_s101
      python scripts/research/overopt_grid.py collect [--dirs DIR ...]

Writes results/research/overopt_grid/cells/<key>.json (one per cell),
results/research/overopt_grid.json and assets/research/overopt_grid.png. Games, proxies and
per-cell logs go to data/research/overopt/grid (gitignored). Exact for the fitted game MDP,
not real football; no play-by-play is loaded.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import overopt_curves as curves  # noqa: E402

from fourth_and_rl import overopt as oo  # noqa: E402
from fourth_and_rl import overopt_grid as og  # noqa: E402
from fourth_and_rl.paths import ASSETS, RESULTS, ROOT  # noqa: E402

OUT_JSON = RESULTS / "research" / "overopt_grid.json"
OUT_FIG = ASSETS / "research" / "overopt_grid.png"
DAY5_CELLS = RESULTS / "research" / "overopt_cells"


def default_results_dir(cfg: dict) -> Path:
    return ROOT / cfg["grid"]["results_dir"]


def default_work_dir(cfg: dict) -> Path:
    return ROOT / cfg["work_dir"] / "grid"


def game_folder(game_dir: Path | str | None) -> Path:
    from fourth_and_rl import mdp_game as mg

    return Path(game_dir) if game_dir is not None else mg.SOLUTION_DIR


def cell_of(key: str, cfg: dict) -> og.Cell:
    """The Cell for a key (its block is looked up in the config's plan when listed)."""
    for c in og.plan_from_config(cfg["grid"]):
        if c.key == key:
            return c
    p = og.parse_key(key)
    if p is None:
        return og.Cell(og.ORACLE, None, None, None, 0)
    if p[1] not in cfg["proxy_sizes"]:
        raise SystemExit(f"unknown capacity {p[1]!r}; configs/overopt.yaml has "
                         f"{sorted(cfg['proxy_sizes'])}")  # fmt: skip
    return og.Cell(key, p[0], p[1], p[2], -1)


# --------------------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------------------


@contextmanager
def locked(path: Path):
    """An exclusive lock on path (a lock file), so two cells running at once never build
    the same games file twice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def ensure_games(work: Path, n: int, seed: int, game_dir: Path, post: str) -> Path:
    """n synthetic coach-vs-coach games with this seed (overopt.coach_games), saved once."""
    path = work / f"games_n{n}_s{seed}.npz"
    with locked(work / f".games_n{n}_s{seed}.lock"):
        if path.exists():
            return path
        t0 = time.time()
        sol, coach = oo.load_game(game_dir)
        d = oo.coach_games(sol, coach, n, seed, post=post)
        tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
        np.savez_compressed(tmp, X=d["X"], y=d["y"], game=d["game"], seed=seed)
        tmp.replace(path)
        print(f"games n={n} seed={seed}: {len(d['y'])} rows in {time.time() - t0:.1f} s",
              flush=True)  # fmt: skip
    return path


def ensure_proxy(work: Path, cfg: dict, cell: og.Cell, games: Path, val: Path
                 ) -> tuple[Path, dict]:  # fmt: skip
    txt, meta_path = work / f"proxy_{cell.key}.txt", work / f"proxy_{cell.key}.json"
    with locked(work / f".proxy_{cell.key}.lock"):
        if txt.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("params") == cfg["proxy_sizes"][cell.capacity]:
                return txt, meta
        d, v = np.load(games), np.load(val)
        tmp = txt.with_name(f".{txt.name}.tmp-{os.getpid()}")
        meta = {"cell": cell.key, "n_games": cell.n_games, "games_seed": cell.seed,
                "games_file": games.name} | curves.fit_proxy(cfg, cell.capacity, d, v, tmp)
        tmp.replace(txt)
        og.write_atomic(meta, meta_path)
        print(f"proxy {cell.key}: {meta['rows']} rows, fit {meta['fit_seconds']:.1f} s, "
              f"val logloss {meta['val_logloss']:.4f}", flush=True)  # fmt: skip
    return txt, meta


def versions() -> dict[str, str]:
    import lightgbm
    import scipy

    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "lightgbm": lightgbm.__version__}  # fmt: skip


def run_cell(key: str, results_dir: Path, threads: int | None = None,
             betas: list[float] | None = None, game_dir: Path | str | None = None,
             work: Path | None = None, force: bool = False) -> Path:  # fmt: skip
    """Games, proxy and exact pass for one cell; writes results_dir/<key>.json."""
    cfg = copy.deepcopy(oo.load_config())
    if threads is not None:
        cfg["threads"] = int(threads)
    if betas:
        cfg["betas"] = [float(b) for b in betas]
    gdir = game_folder(game_dir)
    work = work or default_work_dir(cfg)
    cell = cell_of(key, cfg)
    gkey = og.game_key_of_dir(gdir)
    settings = og.cell_settings(cfg, cell, betas or None, gkey)
    skey = og.settings_key(settings)
    out_path = Path(results_dir) / f"{key}.json"
    if not force and og.is_done(out_path, skey):
        print(f"{key}: already done ({out_path})", flush=True)
        return out_path
    t0 = time.time()
    val_spec = cfg["games"][cfg["grid"]["val_games"]]
    val = ensure_games(work, val_spec["n"], val_spec["seed"], gdir, cfg["post"])
    proxy = None
    if not cell.is_oracle:
        games = ensure_games(work, cell.n_games, cell.seed, gdir, cfg["post"])
        proxy = ensure_proxy(work, cfg, cell, games, val)
    prep_s = time.time() - t0
    print(f"{key}: data and proxy ready in {prep_s:.0f} s; exact pass with {cfg['threads']} "
          f"threads, betas {cfg['betas']}", flush=True)  # fmt: skip
    rec = curves.exact_pass(cfg, key, proxy, val, game_dir=gdir)
    res = og.system_resources()
    rec["timing"]["prep_seconds"] = prep_s
    rec["grid"] = {"key": key, "n_games": cell.n_games, "capacity": cell.capacity,
                   "seed": cell.seed, "settings": settings, "settings_key": skey,
                   "threads": cfg["threads"], "versions": versions(),
                   "host": {"name": platform.node(), "cores": res["cores"],
                            "ram_gb": res["ram_gb"]},
                   "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}  # fmt: skip
    og.write_atomic(rec, out_path)
    print(f"{key}: wrote {out_path} after {time.time() - t0:.0f} s", flush=True)
    return out_path


# --------------------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------------------


def make_plan(cfg: dict, sizes=None, capacities=None, seeds=None, include_oracle: bool = True,
              parts=None) -> list[og.Cell]:  # fmt: skip
    return og.select_parts(og.plan_from_config(cfg["grid"], sizes, capacities, seeds,
                                               include_oracle), parts)  # fmt: skip


def settings_fn(cfg: dict, betas: list[float] | None, gkey: str):
    def skey(c: og.Cell) -> str:
        return og.settings_key(og.cell_settings(cfg, c, betas or None, gkey))

    return skey


def recorded_seconds(paths: list[Path]) -> list[float]:
    out = []
    for p in paths:
        rec = og.read_cell(p) or {}
        t = rec.get("timing") or {}
        if t.get("total_seconds") is not None:
            out.append(float(t["total_seconds"]) + float(t.get("prep_seconds") or 0))
    return out


def launcher(results_dir: Path, log_dir: Path, betas: list[float] | None,
             game_dir: Path | str | None, work: Path | None):  # fmt: skip
    """launch(cell, threads) -> Popen of `overopt_grid.py cell <key>`, output to a log."""

    def launch(c: og.Cell, threads: int) -> subprocess.Popen:
        cmd = [sys.executable, str(Path(__file__).resolve()), "cell", c.key,
               "--results-dir", str(results_dir), "--threads", str(threads)]  # fmt: skip
        if betas:
            cmd += ["--betas", *(str(b) for b in betas)]
        if game_dir is not None:
            cmd += ["--game-dir", str(game_dir)]
        if work is not None:
            cmd += ["--work-dir", str(work)]
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                   MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")  # fmt: skip
        log_dir.mkdir(parents=True, exist_ok=True)
        f = open(log_dir / f"{c.key}.log", "a")
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT)

    return launch


def run(*, sizes=None, capacities=None, seeds=None, include_oracle: bool = True, parts=None,
        results_dir: Path | None = None, skip_dirs=(), log_dir: Path | None = None,
        deadline: float | None = None, max_parallel: int | None = None,
        threads_per_cell: int | None = None, betas: list[float] | None = None,
        game_dir: Path | str | None = None, work: Path | None = None,
        log=print, dry: bool = False) -> dict[str, Any]:  # fmt: skip
    """Plan, skip finished cells, choose the parallelism and run until the deadline."""
    cfg = oo.load_config()
    g = cfg["grid"]
    results_dir = Path(results_dir) if results_dir else default_results_dir(cfg)
    work = Path(work) if work else default_work_dir(cfg)
    log_dir = Path(log_dir) if log_dir else work / "logs"
    gkey = og.game_key_of_dir(game_folder(game_dir))
    plan = make_plan(cfg, sizes, capacities, seeds, include_oracle, parts)
    todo, done = og.split_done(plan, [results_dir, *skip_dirs], settings_fn(cfg, betas, gkey))
    res = og.system_resources()
    par = og.choose_parallel(res["cores"], res["available_gb"], g["mem_gb_per_cell"],
                             threads_per_cell or g["threads_per_cell"], g["reserve_gb"],
                             max_parallel, ram_gb=res["ram_gb"])  # fmt: skip
    prev = recorded_seconds(list(done.values()))
    est = max(prev) if prev else float(g["seconds_per_cell"])
    log(f"game key {gkey} ({'same as' if gkey == g['reference_game_key'] else 'differs from'} "
        f"the reference data/mdp {g['reference_game_key']})")
    log(f"plan: {len(plan)} cells, {len(done)} done, {len(todo)} to run; "
        f"{res['cores']} cores, {res['ram_gb']:.1f} GB RAM ({res['available_gb']:.1f} GB "
        f"available): {par['n_parallel']} at a time, {par['threads_per_cell']} threads each "
        f"(limited by {par['limited_by']})")
    for w in par["warnings"]:
        log(f"WARNING: {w}")
    for row in og.block_hours(plan, todo, par["n_parallel"], est):
        log(f"  block {row['block'] + 1} ({row['label']}): {row['todo']} of {row['cells']} "
            f"cells to run, about {row['hours']:.1f} h at {est / 60:.0f} min per cell")
    deadline = deadline if deadline is not None else time.time() + 3600 * 5
    out = {"plan": [c.key for c in plan], "already_done": sorted(done), "parallel": par,
           "resources": res, "game_key": gkey, "seconds_per_cell_estimate": est}  # fmt: skip
    if dry or not todo:
        return out | {"run": None}
    status = og.run_grid(
        todo, launcher(results_dir, log_dir, betas, game_dir, work),
        n_parallel=par["n_parallel"], threads=par["threads_per_cell"], deadline=deadline,
        seconds_per_cell=est, log=log,
        is_finished=lambda c: og.is_done(results_dir / f"{c.key}.json",
                                         settings_fn(cfg, betas, gkey)(c)))  # fmt: skip
    log(f"finished {len(status['done'])}, failed {len(status['failed'])}, not started "
        f"{len(status['not_started'])}; stopped: {status['stopped']}")
    return out | {"run": status}


def collect(dirs: list[Path] | None = None, out_json: Path = OUT_JSON,
            out_fig: Path | None = OUT_FIG, game: str | None = None,
            plan_kwargs: dict | None = None, extra: dict | None = None
            ) -> dict[str, Any]:  # fmt: skip
    cfg = oo.load_config()
    dirs = dirs or [default_results_dir(cfg)]
    plan = make_plan(cfg, **(plan_kwargs or {}))
    return og.collect(dirs, out_json, out_fig, game=game, plan=plan,
                      reference_game=cfg["grid"]["reference_game_key"], day5_dir=DAY5_CELLS,
                      extra=extra)  # fmt: skip


def print_summary(out: dict[str, Any], log=print) -> None:
    log(f"{len(out['cells'])} cells, game key {out['game_key']}; merge: "
        f"{len(out['merge']['duplicates'])} duplicates, {len(out['merge']['conflicts'])} "
        f"conflicts, {len(out['merge']['excluded_other_game'])} from another game")
    log(f"{'cell':22s} {'peak beta':>9s} {'peak':>7s} {'final':>7s} {'drop':>7s}  (points)")
    for k, c in out["cells"].items():
        log(f"{k:22s} {c['peak_beta']:9.0f} {c['peak_gain_points']:7.3f} "
            f"{c['final_gain_points']:7.3f} {c['drop_points']:7.3f}")
    for gr in out["groups"]:
        head = f"{gr['n_games']:>6,} games, {gr['capacity']:5s}: drop {gr['drop_mean_points']:.3f}"
        sd = gr["drop_sd_points"]
        if sd is None:
            log(f"{head} (1 seed, no SD)")
        else:
            log(f"{head} (SD {sd:.3f} over {gr['n_seeds']} seeds), exceeds 2 SD: "
                f"{gr['drop_exceeds_2sd']}")
    for r in out.get("day5_reproduction") or []:
        log(f"day 5 check: {r['grid_cell']} vs {r['day5_cell']}: max |true gain diff| "
            f"{r['max_abs_true_gain_diff']:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def filters(p: argparse.ArgumentParser) -> None:
        p.add_argument("--sizes", type=int, nargs="+")
        p.add_argument("--capacities", nargs="+")
        p.add_argument("--seeds", type=int, nargs="+")
        p.add_argument("--no-oracle", action="store_true")
        p.add_argument("--part", action="append", help="K/M, repeatable (e.g. --part 3/3)")

    p = sub.add_parser("plan", help="print the plan and what is done")
    filters(p)
    p.add_argument("--results-dir", type=Path)
    p = sub.add_parser("run", help="run the planned cells until the deadline")
    filters(p)
    p.add_argument("--results-dir", type=Path)
    p.add_argument("--skip-dir", type=Path, action="append", default=[],
                   help="also skip cells already done in this folder (e.g. unzipped Colab)")
    p.add_argument("--deadline-hours", type=float, default=5.0)
    p.add_argument("--parallel", type=int, help="at most this many cells at once")
    p.add_argument("--threads", type=int, help="threads per cell (default from the config)")
    p.add_argument("--betas", type=float, nargs="+", help="override (for a smoke test)")
    p.add_argument("--game-dir", type=Path)
    p.add_argument("--work-dir", type=Path)
    p = sub.add_parser("cell", help="run one cell (used by run)")
    p.add_argument("key")
    p.add_argument("--results-dir", type=Path)
    p.add_argument("--threads", type=int)
    p.add_argument("--betas", type=float, nargs="+")
    p.add_argument("--game-dir", type=Path)
    p.add_argument("--work-dir", type=Path)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("collect", help="merge result folders, summary, figure")
    p.add_argument("--dirs", type=Path, nargs="+")
    p.add_argument("--game-key")
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-fig", type=Path, default=OUT_FIG)
    a = ap.parse_args()
    cfg = oo.load_config()
    if a.cmd in ("plan", "run"):
        kw = dict(sizes=a.sizes, capacities=a.capacities, seeds=a.seeds,
                  include_oracle=not a.no_oracle, parts=a.part)  # fmt: skip
    if a.cmd == "plan":
        rd = a.results_dir or default_results_dir(cfg)
        out = run(**kw, results_dir=rd, dry=True)
        for i, key in enumerate(out["plan"]):
            print(f"{i + 1:3d}  {key:22s} {'done' if key in out['already_done'] else ''}")
    elif a.cmd == "run":
        run(**kw, results_dir=a.results_dir, skip_dirs=a.skip_dir,
            deadline=time.time() + 3600 * a.deadline_hours, max_parallel=a.parallel,
            threads_per_cell=a.threads, betas=a.betas, game_dir=a.game_dir,
            work=a.work_dir)  # fmt: skip
    elif a.cmd == "cell":
        run_cell(a.key, a.results_dir or default_results_dir(cfg), a.threads, a.betas,
                 a.game_dir, a.work_dir, a.force)  # fmt: skip
    else:
        out = collect(a.dirs, a.out_json, a.out_fig, a.game_key)
        print_summary(out)
        print(f"wrote {a.out_json} and {a.out_fig}")


if __name__ == "__main__":
    main()
