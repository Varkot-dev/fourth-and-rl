"""Build notebooks/team_matchups_colab.ipynb. The notebook is generated, never hand-edited:
change this file and run it again.

    python notebooks/build_team_matchups_notebook.py

The notebook clones the public repository (main), installs it with the `jax` extra, and
calls `fourth_and_rl.team_matchups` and `fourth_and_rl.mdp_team_jax`; all logic lives in
those tested modules, the cells only set parameters, print and save.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "team_matchups_colab.ipynb"

INTRO = """# Fourth down by matchup: the exact two-team end game on a GPU

**What this is.** The repository solves the end of an NFL game exactly: every state of the
last five minutes (time, field position, score, timeouts) gets a win probability and a best
call, by backward induction. With two different teams the game needs two value tables, one
per team, and every change of possession reads the other team's table. On a CPU one matchup
takes about a minute. This notebook runs the same calculation on a GPU with JAX, so it can
solve dozens of matchups in one session.

**The question.** Up 4 with 4:00 left, fourth and 2 at your own 40, three timeouts each:
go for it or punt? And how far does the go region (distances 1 to 10, own 20 to midfield)
move with the matchup? The grid: Detroit's offense and defense against an opponent with an
average offense and each of the 32 defenses, then each of the five best and five worst
offenses against a top-5 defense, plus the five matchups the repository already solved on
the CPU (a reproduction check).

**Where the numbers come from.** Nothing is uploaded. The notebook downloads public nflverse
play-by-play for 2010 to 2025 and refits the game's inputs with the repository's own guarded
loaders (development seasons 2010 to 2023; the 2024 and 2025 kickoff tables are the
repository's one logged exception). The 2026 season is a sealed holdout and is never
downloaded. Team ratings are read from `results/m8_team_strength.json`: they are the
end-of-2025 ratings from M8's test stage, test-season numbers used here as inputs to a
hypothetical game. No bootstrap: nothing here has an uncertainty interval.

**How long it takes.** A few minutes to download and refit the inputs, then the
matchups until the deadline (60 minutes by default; it stops cleanly and saves what it has).

**Choosing a GPU.** Runtime > Change runtime type > GPU (a T4 is enough). Then Runtime >
Run all. Without a GPU it still runs, on the CPU, slowly: set `N_MATCHUPS = 2` for a test.

**If you are disconnected.** Run all again. With Google Drive mounted, the downloaded
data, the refitted inputs and every finished matchup are kept in
`MyDrive/fourth_and_rl_team_matchups` and are not redone.
"""

BRING_BACK = """## What to bring back

The last cell downloads `team_matchups_results.zip` (no data inside). It holds:

- `results.json`: every matchup's go, punt and field-goal win probabilities at the headline
  state, its go-minus-punt map, the kernels' calibration, the timing and backend benchmark,
  the reproduction check against the committed CPU results, and the inputs' provenance
  (including whether the rebuilt development data has the same fingerprint as the committed
  solution);
- `team_matchups.png`: go minus punt by opponent defense, ranked, and go regions for a few
  matchups;
- `matchups/*.json`: one file per finished matchup; `run_log.txt`.

Unzip it into the repository under `results/questions/q6_team_matchups_gpu/` and commit it
together with the figure. Also send the printed CPU versus GPU timing and the backend
benchmark (both are in `results.json`).
"""

PARAMS = """# Parameters. The defaults are the real run: leave them as they are for Run all.
import time
NB_START = time.time()
# N_MATCHUPS: None solves the whole grid (as far as the deadline allows); a small number
# solves only the first few matchups (for a quick test).
N_MATCHUPS = None
# DEADLINE_MIN: minutes from the start of this session after which no new batch starts.
DEADLINE_MIN = 60
# USE_DRIVE: keep data, inputs and finished matchups on Google Drive (resume after a
# disconnect).
USE_DRIVE = True
# BACKEND: "auto" times dense, bcoo and bcoo_cusparse on this GPU and uses the fastest;
# or force one of them.
BACKEND = "auto"
# BATCH_SIZE: matchups solved together on the GPU; None picks from the GPU's memory.
BATCH_SIZE = None
# RUN_CPU_TIMING: time one matchup on the CPU solver as well (about 1 to 2 minutes).
RUN_CPU_TIMING = True
REPO_URL = "https://github.com/Varkot-dev/fourth-and-rl.git"
BRANCH = "main"
# SOURCE_DIR: a local checkout to copy instead of cloning (for testing the notebook).
SOURCE_DIR = None
# WORK_DIR: where outputs go; None means Drive (if mounted) or /content/team_matchups_run.
WORK_DIR = None"""

SETUP = """# Get the code: clone the public repository (or copy SOURCE_DIR), install it with the jax
# extra. On Colab the preinstalled GPU build of JAX satisfies the extra and is kept.
import os, shutil, subprocess, sys
from pathlib import Path

BASE = Path('/content') if Path('/content').exists() else Path.cwd()
REPO = BASE / 'fourth-and-rl'
if not (REPO / 'pyproject.toml').exists():
    if SOURCE_DIR:
        skip = shutil.ignore_patterns('.git', '.venv', '__pycache__', '.ipynb_checkpoints',
                                      '*.egg-info')
        def ignore(d, names):  # the top-level data/ folder is not part of the code
            out = set(skip(d, names))
            return out | ({'data'} if Path(d).resolve() == Path(SOURCE_DIR).resolve() else set())
        shutil.copytree(SOURCE_DIR, REPO, symlinks=True, ignore=ignore)
    else:
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', BRANCH, REPO_URL,
                        str(REPO)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', f'{REPO}[jax]'],
               check=True)
sys.path.insert(0, str(REPO / 'src'))  # the editable install is not seen until a restart
os.chdir(REPO)
rev = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout
print('code:', REPO, rev.strip() or '(copied from ' + str(SOURCE_DIR) + ')')"""

DRIVE = """# Work folder: Google Drive if mounted (USE_DRIVE), so a disconnect loses nothing.
WORK = Path(WORK_DIR) if WORK_DIR else None
if WORK is None and USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        WORK = Path('/content/drive/MyDrive/fourth_and_rl_team_matchups')
    except Exception as e:  # not on Colab, or the mount was declined
        print('Drive not mounted, outputs are lost on disconnect:', e)
if WORK is None:
    WORK = BASE / 'team_matchups_run'
WORK.mkdir(parents=True, exist_ok=True)
LOG = open(WORK / 'run_log.txt', 'a')

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    LOG.write(line + '\\n'); LOG.flush()

log(f'work folder: {WORK}')"""

GPU = """# Check for a GPU.
import jax
devices = jax.devices()
ON_GPU = devices[0].platform == 'gpu'
log(f'JAX {jax.__version__}, devices: {devices}')
if not ON_GPU:
    log('No GPU: this runs on the CPU (fine for a test with a small N_MATCHUPS, slow for the '
        'grid). For the real run: Runtime > Change runtime type > GPU, then Run all.')"""

INPUTS = """# Rebuild the exact game inputs from public nflverse data with the repository's own
# guarded loaders (see the intro). The 2026 holdout is refused by name before anything is
# downloaded. The result is cached in the work folder.
from fourth_and_rl import team_matchups as tm
from fourth_and_rl import mdp_team_jax as mj
from fourth_and_rl.io import write_json, read_json
t0 = time.time()
tm.link_data_dir(WORK)
INPUTS = tm.rebuild_inputs(WORK / 'inputs', log=log)
p = INPUTS.provenance
log(f'inputs ready in {time.time() - t0:.0f} s; same development data as the committed '
    f'solution: {p.get("same_data_as_committed_solution")}')
for e in p.get('test_season_reads_this_run', []):
    log(f'  logged test-season read: {e["seasons"]} for: {e["what"]}')"""

GRID = """# Team ratings and the grid of matchups.
RATINGS = tm.season_end_ratings()
log(tm.RATINGS_LABEL)
SPECS = tm.matchup_grid(RATINGS)
if N_MATCHUPS is not None:
    SPECS = SPECS[:int(N_MATCHUPS)]
log(f'{len(SPECS)} matchups planned: ' + ', '.join(s['key'] for s in SPECS[:8])
    + (' ...' if len(SPECS) > 8 else ''))"""

BENCH = """# Sparse or dense products? Time each backend for a few ticks of one real matchup on this
# device (compile time excluded) and use the fastest. On the CPU the sparse backends are not
# timed: their generic JAX path ran out of 15 GB of host memory on the real game, so the
# dense backend is used.
kern = tm.Kernels(INPUTS)
bench_path = WORK / 'benchmark.json'
BENCH = read_json(bench_path) if bench_path.exists() else None
if BENCH is None or BENCH.get('device') != str(devices[0]):
    if ON_GPU and BACKEND == 'auto':
        a, b, _ = kern.pair(SPECS[0])
        BENCH = mj.benchmark_backends((a, b), ticks=4)
    else:
        BENCH = {'device': str(devices[0]), 'choice': 'dense' if BACKEND == 'auto' else BACKEND,
                 'note': 'not benchmarked (CPU, or BACKEND forced)'}
    write_json(BENCH, bench_path)
CHOSEN = BENCH['choice'] if BACKEND == 'auto' else BACKEND
log(f'benchmark: {BENCH}')
mem = mj.device_memory_bytes()
per = mj.bytes_per_matchup(INPUTS.base, CHOSEN)
if BATCH_SIZE is None:
    BATCH_SIZE = max(1, min(4, int(0.6 * mem // per))) if (ON_GPU and mem) else 1
log(f'backend {CHOSEN}, batch size {BATCH_SIZE} (device memory {mem}, about {per / 1e9:.1f} GB '
    'per matchup)')"""

TIMING = """# CPU versus GPU for one matchup (Detroit vs an average team if it is in the grid).
timing_path = WORK / 'cpu_vs_gpu.json'
TIMING = read_json(timing_path) if timing_path.exists() else None
spec_t = next((s for s in SPECS if s['key'] == 'det_vs_average'), SPECS[0])
if RUN_CPU_TIMING and (TIMING is None or TIMING.get('jax_device') != str(devices[0])):
    log(f'timing {spec_t["key"]} on the CPU solver and with JAX ...')
    TIMING = tm.cpu_vs_gpu(INPUTS, spec_t, CHOSEN)
    write_json(TIMING, timing_path)
if TIMING:
    log(f'CPU {TIMING["cpu_seconds"]:.1f} s ({TIMING["cpu_threads"]} threads) vs JAX '
        f'{TIMING["jax_seconds"]:.1f} s on {TIMING["jax_device"]} ({TIMING["jax_backend"]}); '
        f'speedup {TIMING["speedup"]:.1f}x; largest value difference '
        f'{TIMING["max_abs_value_diff"]:.1e}')"""

SOLVE = """# Solve the grid until the deadline. Every finished matchup is saved at once; a
# rerun skips them. A batch starts only if it should finish before the deadline (minus 2
# minutes for the results).
DEADLINE = NB_START + 60 * DEADLINE_MIN
STATUS = tm.solve_grid(INPUTS, SPECS, WORK, DEADLINE, backend=CHOSEN, batch_size=BATCH_SIZE,
                       log=log)
log(f'{len(STATUS["done"])} of {len(SPECS)} matchups done; stopped: {STATUS["stopped"]}')"""

RESULTS = """# results.json, the figure and the zip.
import platform
from fourth_and_rl import mdp
qcfg = mdp.load_config()['question1']
ROWS = tm.collect(WORK, SPECS)
RES = {
    'what': 'Question 6 on a GPU: the late-lead fourth down by matchup, exact two-team '
            'end-phase solutions with JAX (src/fourth_and_rl/mdp_team_jax.py, '
            'notebooks/team_matchups_colab.ipynb)',
    'ratings_label': tm.RATINGS_LABEL,
    'ratings_2025': RATINGS,
    'inputs': INPUTS.provenance,
    'headline': qcfg['headline'] | {'offense_timeouts': qcfg['offense_timeouts']},
    'map': {k: qcfg[k] for k in ('yardlines', 'distances', 'offense_timeouts')},
    'benchmark': BENCH,
    'backend': CHOSEN,
    'batch_size': BATCH_SIZE,
    'cpu_vs_gpu': TIMING,
    'grid': {'planned': len(SPECS), 'done': len(ROWS), 'stopped': STATUS['stopped'],
             'deadline_min': DEADLINE_MIN, 'n_matchups_param': N_MATCHUPS},
    'q6_reproduction': tm.q6_reproduction(ROWS),
    'environment': {'python': platform.python_version(), 'jax': jax.__version__,
                    'device': str(devices[0]), 'code': rev.strip() or str(SOURCE_DIR)},
    'matchups': ROWS,
    'caveats': [
        'No bootstrap: no value here has an uncertainty interval.',
        'Ratings are end-of-2025 numbers from M8 test stage (test-season quantities), used '
        'as inputs to a hypothetical game; nothing is scored.',
        'Ratings move plays only (a yards tilt and a conversion shift); turnovers, kicking '
        'and clock behaviour are league average.',
        'Only the end phase (last 5:00) is solved; it equals a whole-game solve there.',
        'JAX computes in float32; it agrees with the CPU solver to 1e-5 (tests).',
    ],
}
write_json(RES, WORK / 'results.json')
if ROWS:
    tm.figure(RES, WORK / 'team_matchups.png')
ZIP = tm.make_zip(WORK)
log(f'wrote {WORK / "results.json"}, the figure and {ZIP}')
for k, v in RES['q6_reproduction'].items():
    log(f'  reproduction {k}: {v["headline_points_this_run"]:+.2f} here vs '
        f'{v["headline_points_committed"]:+.2f} committed; map max diff '
        f'{v["map_max_abs_diff_points"]:.3f} pts')
vs = sorted([r for r in ROWS if r['group'] == 'vs_def'],
            key=lambda r: r['headline']['go_minus_punt_points'])
for r in vs:
    log(f'  {r["label"]:28s} (defense #{r["defense_rank"]:2d}) go - punt '
        f'{r["headline"]["go_minus_punt_points"]:+.2f} pts, go cells {r["go_cells"]}')
if ROWS:
    from IPython.display import Image, display
    display(Image(filename=str(WORK / 'team_matchups.png')))"""

DOWNLOAD = """# Download the results zip (see "What to bring back").
print(ZIP, f'{ZIP.stat().st_size / 1e6:.2f} MB')
try:
    from google.colab import files
    files.download(str(ZIP))
except Exception as e:
    print('Download it from the file browser instead:', e)"""


def build() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(INTRO),
        new_markdown_cell(BRING_BACK),
        new_code_cell(PARAMS),
        new_markdown_cell("Get the code and install it."),
        new_code_cell(SETUP),
        new_markdown_cell("Work folder (Google Drive keeps progress across disconnects)."),
        new_code_cell(DRIVE),
        new_markdown_cell("Check for a GPU."),
        new_code_cell(GPU),
        new_markdown_cell("Rebuild the game's inputs from public data (a few minutes the "
                          "first time, seconds on a resume)."),
        new_code_cell(INPUTS),
        new_markdown_cell("Team ratings and the matchups."),
        new_code_cell(GRID),
        new_markdown_cell("Choose sparse or dense products for this GPU."),
        new_code_cell(BENCH),
        new_markdown_cell("CPU versus GPU on one matchup."),
        new_code_cell(TIMING),
        new_markdown_cell("Solve the matchups (deadline-guarded, checkpointed, resumable)."),
        new_code_cell(SOLVE),
        new_markdown_cell("Results, figure and zip."),
        new_code_cell(RESULTS),
        new_markdown_cell("Download."),
        new_code_cell(DOWNLOAD),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    return nb


def main() -> None:
    nb = build()
    nbformat.validate(nb)
    OUT.write_text(nbformat.writes(nb))
    print(f"wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
