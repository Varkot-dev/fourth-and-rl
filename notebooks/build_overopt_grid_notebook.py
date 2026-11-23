"""Build notebooks/overopt_grid_colab.ipynb (R1 day 6, run A: the seeded grid on Colab).
The notebook is generated, never hand-edited: change this file and run it again.

    python notebooks/build_overopt_grid_notebook.py

The notebook clones the public repository (main), installs it, and calls the tested code:
`fourth_and_rl.game_setup` (the solved game: reuse, restore from Drive, or rebuild from
public data), `fourth_and_rl.overopt_grid` (plan, resume, parallelism, collect) and
scripts/research/overopt_grid.py (one subprocess per cell). The cells only set parameters,
print and save.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "overopt_grid_colab.ipynb"
BRANCH = "main"
REPO_URL = "https://github.com/Varkot-dev/fourth-and-rl.git"
COLAB_URL = ("https://colab.research.google.com/github/Varkot-dev/fourth-and-rl/blob/"
             f"{BRANCH}/notebooks/overopt_grid_colab.ipynb")

INTRO = f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

# Overoptimization of a learned value model, graded exactly: the seeded grid

**What this runs.** A learned win-probability model (a "proxy") is trained on synthetic
games, and a fourth-down policy leans on it harder and harder: the coach's choice tilted by
exp(beta Q_proxy), for beta from 10 to 10,000. The repository's solved game MDP grades every
policy exactly: its true gain over coach-vs-coach play, the gain the proxy claims, and the
exact sequence KL from the coach. Day 5 found that the true gain rises and then falls only
for the highest-error proxy tried (1,000 games, 64 leaves). This notebook runs the seeded
grid that tests where that happens: training sets of 250, 500, 1,000, 4,000, 16,000 and
64,000 games, two capacities (8 and 64 leaves) and five data seeds (101 to 105), one exact
pass per cell with all betas at once.

**What it does not answer.** Everything is exact for the fitted game MDP, not real football.
The proxies are trained on synthetic games from that MDP only: no play-by-play is used for
any proxy, no test season is fitted, and the 2026 season is never downloaded.

**Order.** Cells run in priority order, so a short session still yields the most informative
ones: first the turnover region (250, 500 and 1,000 games, both capacities, seed by seed),
then the oracle (the exact coach values, the reference curve), then 4,000 and 16,000 games,
then 64,000 if time remains.

**Runtime.** Runtime > Change runtime type > CPU, with High-RAM. One cell takes about 12 to
16 minutes on 3 threads and peaks at about 7.6 GB, so the notebook runs as many cells at once
as fit in RAM and cores, and warns below 12 GB. No GPU is used. Then Runtime > Run all.

**Stopping and resuming.** With Drive mounted, every finished cell is saved at once in
`MyDrive/fourth_and_rl_overopt_grid/cells`. A cell starts only if it should finish before the
deadline (5 hours by default). Run all again in a new session and finished cells are skipped.

**What to bring back.** The last cell downloads `overopt_grid_results.zip` (no data inside):
the cell files, `overopt_grid.json` (per cell: peak beta, peak gain, final gain, drop; per
training-set size and capacity: the mean and SD of the drop across seeds and whether it
exceeds 2 SD), the figure and the logs. Unzip `cells/` into the repository under
`results/research/overopt_grid/cells/`, then run
`python scripts/research/overopt_grid.py collect` there.
"""

GAME_NOTES = """### The solved game, and merging with a local run

The exact grader needs the solved game `data/mdp` (about 2 GB, not in the repository). The
notebook uses, in this order: `GAME_DIR` if set; the copy in this runtime; the Drive cache
`MyDrive/fourth_and_rl_overopt_grid/game`; or a rebuild from public nflverse data with the
repository's own pipeline (download 2010 to 2025, the development component bundle, count and
coach tables on development seasons, backward induction; a few minutes, about 9 GB peak).
After a rebuild it caches the game on Drive when Drive has room, and skips the cache
otherwise.

A rebuild from a fresh download can differ from the owner's local copy (nflverse revises its
data), and cells graded on different games are never pooled. To split the grid between this
notebook and a local machine, first upload the local `data/mdp` folder (all of its files) to
`MyDrive/fourth_and_rl_overopt_grid/game`. The notebook prints whether its game matches the
reference (`reference_game_key` in `configs/overopt.yaml`). Then give each machine its own
parts of the same plan, for example `PARTS = ["1/3", "2/3"]` here (two cells at a time) and
`python scripts/research/overopt_grid.py run --part 3/3` locally (one at a time). Part K/M is
every M-th cell of the priority order, so both machines work through the turnover region
first and never run the same cell.
"""

PARAMS = f"""# Parameters. The defaults are the real run: leave them as they are for Run all.
import time
NB_START = time.time()
SIZES = None              # training games per proxy; None: 250, 500, 1000, 4000, 16000, 64000
CAPACITIES = None         # None: ["small", "large"] (8 and 64 leaves)
SEEDS = None              # data seeds; None: [101, 102, 103, 104, 105]
INCLUDE_ORACLE = True     # the oracle cell (exact coach values), after the first block
PARTS = None              # e.g. ["1/3", "2/3"]: only these parts of the plan (see above)
BETAS = None              # None: the config's betas; a shorter list only for a test run
DEADLINE_HOURS = 5.0      # no cell starts unless it should finish before this
MAX_PARALLEL = None       # None: as many cells at once as RAM and cores allow
THREADS_PER_CELL = None   # None: 3, as measured
USE_DRIVE = True          # keep finished cells (and the game cache) on Google Drive
CACHE_GAME = True         # cache the solved game on Drive when there is room
GAME_DIR = None           # a folder with a solved game to use as is (read only)
RUN_TESTS = True          # the grid's unit tests first (seconds)
REPO_URL = "{REPO_URL}"
BRANCH = "{BRANCH}"
SOURCE_DIR = None         # a local checkout to copy instead of cloning (testing the notebook)
WORK_DIR = None           # None: the Drive folder, or /content/overopt_grid without Drive"""

SETUP = """# Get the code: clone the public repository (or copy SOURCE_DIR) and install it.
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
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', f'{REPO}[dev]'],
               check=True)
for p in (REPO / 'src', REPO / 'scripts' / 'research'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))  # the editable install is not seen until a restart
os.chdir(REPO)
rev = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout
print('code:', REPO, rev.strip() or '(copied from ' + str(SOURCE_DIR) + ')')"""

DRIVE = """# Work folder: Google Drive if mounted (USE_DRIVE): a disconnect loses no finished cell.
WORK = Path(WORK_DIR) if WORK_DIR else None
ON_DRIVE = False
if WORK is None and USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        WORK = Path('/content/drive/MyDrive/fourth_and_rl_overopt_grid')
        ON_DRIVE = True
    except Exception as e:  # not on Colab, or the mount was declined
        print('Drive not mounted, finished cells are lost when the runtime ends:', e)
if WORK is None:
    WORK = BASE / 'overopt_grid'
CELLS = WORK / 'cells'
CELLS.mkdir(parents=True, exist_ok=True)
LOG = open(WORK / 'run_log.txt', 'a')

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    LOG.write(line + '\\n'); LOG.flush()

log(f'work folder: {WORK}')"""

RESOURCES = """# Cores and RAM, and how many cells run at once. One cell peaks at about 7.6 GB.
from fourth_and_rl import overopt as oo
from fourth_and_rl import overopt_grid as og
CFG = oo.load_config()
GRID = CFG['grid']
RES = og.system_resources()
PAR = og.choose_parallel(RES['cores'], RES['available_gb'], GRID['mem_gb_per_cell'],
                         THREADS_PER_CELL or GRID['threads_per_cell'], GRID['reserve_gb'],
                         MAX_PARALLEL, ram_gb=RES['ram_gb'])
log(f'{RES["cores"]} cores, {RES["ram_gb"]:.1f} GB RAM ({RES["available_gb"]:.1f} GB available): '
    f'{PAR["n_parallel"]} cell(s) at a time, {PAR["threads_per_cell"]} threads each '
    f'(limited by {PAR["limited_by"]})')
for w in PAR['warnings']:
    log('WARNING: ' + w)
if RES['ram_gb'] < 12:
    print('\\n*** Less than 12 GB of RAM. Runtime > Change runtime type > CPU, High-RAM, then '
          'Run all again. ***\\n')"""

TESTS = """# The grid's unit tests (no data needed). The notebook stops here if any fail.
if RUN_TESTS:
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                        'tests/test_overopt_grid.py'], capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-1500:])
    if r.returncode != 0:
        raise RuntimeError('Tests failed: stop here and send the output above.')"""

GAME = """# The solved game: GAME_DIR, this runtime copy, the Drive cache, or a rebuild from public
# data (see above). Its key tells whether it is the same game as the owner's local copy.
from fourth_and_rl import game_setup as gs
GAME_INFO = gs.ensure_game(GAME_DIR, (WORK / 'game') if ON_DRIVE else None,
                           use_cache=CACHE_GAME, log=log)
GAME_KEY = og.game_key_of_dir(GAME_INFO['game_dir'])
GAME_INFO['game_key'] = GAME_KEY
GAME_INFO['same_as_reference'] = GAME_KEY == GRID['reference_game_key']
log(f'solved game: {GAME_INFO["source"]}; key {GAME_KEY}; same as the reference local game: '
    f'{GAME_INFO["same_as_reference"]}')
if not GAME_INFO['same_as_reference']:
    log('  cells from this game will not merge with local cells (see "The solved game" above)')"""

PLAN = """# The plan in priority order, what is already done, and the hours left per block.
import overopt_grid as G
PLAN_KW = dict(sizes=SIZES, capacities=CAPACITIES, seeds=SEEDS, include_oracle=INCLUDE_ORACLE,
               parts=PARTS)
PLAN = G.make_plan(CFG, **PLAN_KW)
TODO, DONE = og.split_done(PLAN, [CELLS], G.settings_fn(CFG, BETAS, GAME_KEY))
prev = G.recorded_seconds(list(DONE.values()))
EST = max(prev) if prev else float(GRID['seconds_per_cell'])
log(f'{len(PLAN)} cells planned, {len(DONE)} done, {len(TODO)} to run')
for row in og.block_hours(PLAN, TODO, PAR['n_parallel'], EST):
    log(f'  block {row["block"] + 1} ({row["label"]}): {row["todo"]} of {row["cells"]} to run, '
        f'about {row["hours"]:.1f} h at {EST / 60:.0f} min per cell, {PAR["n_parallel"]} at a time')
print('first cells:', ', '.join(c.key for c in TODO[:8]) + (' ...' if len(TODO) > 8 else ''))"""

RUN = """# Run the grid until the deadline. Each cell is a separate process (its log is in
# WORK/logs); a finished cell is saved at once and skipped on a rerun. A cell starts only if
# it should finish before the deadline, less 5 minutes for the results below.
STATUS = G.run(**PLAN_KW, results_dir=CELLS, log_dir=WORK / 'logs',
               deadline=NB_START + 3600 * DEADLINE_HOURS - 300, max_parallel=MAX_PARALLEL,
               threads_per_cell=THREADS_PER_CELL, betas=BETAS, game_dir=GAME_DIR, log=log)"""

COLLECT = """# Collect: per cell peak beta, peak gain, final gain, drop; per size and capacity the
# mean and SD of the drop across seeds and whether it exceeds 2 SD; and the figure.
import platform
ENV = {'python': platform.python_version(), 'code': rev.strip() or str(SOURCE_DIR),
       'resources': RES, 'parallel': PAR}
OUT = G.collect([CELLS], WORK / 'overopt_grid.json', WORK / 'overopt_grid.png',
                plan_kwargs=PLAN_KW,
                extra={'notebook': {'game': GAME_INFO, 'environment': ENV,
                                    'run': STATUS.get('run'), 'betas_param': BETAS,
                                    'deadline_hours': DEADLINE_HOURS}})
G.print_summary(OUT, log=log)
if OUT['cells']:
    from IPython.display import Image, display
    display(Image(filename=str(WORK / 'overopt_grid.png')))"""

ZIP = """# Package the results (no data inside) and download them (see "What to bring back").
members = [p for p in (CELLS, WORK / 'logs', WORK / 'overopt_grid.json',
                       WORK / 'overopt_grid.png', WORK / 'run_log.txt') if p.exists()]
ZIP_PATH = og.make_zip(WORK, 'overopt_grid_results.zip', members)
log(f'wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.2f} MB)')
try:
    from google.colab import files
    files.download(str(ZIP_PATH))
except Exception as e:
    print('Download it from the file browser instead:', e)"""


def build() -> nbformat.NotebookNode:
    params = new_code_cell(PARAMS)
    params.metadata["tags"] = ["parameters"]
    cells = [
        new_markdown_cell(INTRO),
        new_markdown_cell(GAME_NOTES),
        params,
        new_markdown_cell("Get the code and install it."),
        new_code_cell(SETUP),
        new_markdown_cell("Work folder (Google Drive keeps finished cells across sessions)."),
        new_code_cell(DRIVE),
        new_markdown_cell("Cores, RAM and how many cells run at once."),
        new_code_cell(RESOURCES),
        new_markdown_cell("The grid's unit tests."),
        new_code_cell(TESTS),
        new_markdown_cell("The solved game (seconds if cached, a few minutes to rebuild)."),
        new_code_cell(GAME),
        new_markdown_cell("The plan."),
        new_code_cell(PLAN),
        new_markdown_cell("Run the cells (deadline-guarded, saved as they finish, resumable)."),
        new_code_cell(RUN),
        new_markdown_cell("Results and the figure."),
        new_code_cell(COLLECT),
        new_markdown_cell("Download."),
        new_code_cell(ZIP),
    ]
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"  # stable ids, so a rebuild changes only what changed
    nb = new_notebook(cells=cells)
    nb.metadata["colab"] = {"provenance": []}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    return nb


def main() -> None:
    nb = build()
    nbformat.validate(nb)
    OUT.write_text(nbformat.writes(nb))
    print(f"wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
