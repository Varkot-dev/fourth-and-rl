"""Build notebooks/world_model_ground_truth.ipynb (the ground-truth experiment on Colab).

Unlike world_model_colab.ipynb, this notebook embeds no code: it clones the public repo
branch, installs it, runs the notebook tests that need no data, then the stages of
ground_truth_run.py. Re-run this script whenever the cells below change:

    python notebooks/world_model/build_gt_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "world_model_ground_truth.ipynb"
BRANCH = "claude/world-model-colab"
REPO_URL = "https://github.com/Varkot-dev/fourth-and-rl.git"
COLAB_URL = ("https://colab.research.google.com/github/Varkot-dev/fourth-and-rl/blob/"
             f"{BRANCH}/notebooks/world_model_ground_truth.ipynb")

INTRO = f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

# How much data does a learned simulator need to recover the optimal fourth-down policy?

**What this answers.** The repository has an exact model of an NFL game: a Markov decision
process solved by backward induction, with a simulator that plays it. In that model the true
value of going for it, kicking a field goal or punting is known for any fourth down. This
notebook generates games from the exact model with both teams playing like real coaches, trains
the world model (the next-play transformer) on 1,000, 4,000, 16,000 and 64,000 of those games,
and asks each trained model to plan 300 held-out fourth downs inside itself. Because the truth
is known, every recommendation can be graded: how often the learned planner picks the truly best
action, how much win probability its choices lose (regret), and whether its value estimates are
too optimistic for actions the coaches rarely take. The real-data world model trains on about
3,200 games (2010 to 2021), so the 4,000-game model is the closest match to it.

**What it does not answer.** The games come from our own model of football, so this measures
how learnable our simulator is, not real football. GO is evaluated in the play mode the coaches
use, and actions the exact model does not allow are masked out.

**GPU and time.** Runtime > Change runtime type > GPU. An A100 is recommended; an L4 works with
fewer precise states; a T4 finishes the small sizes and records what it had to cut. The default
budget is 90 minutes. Building the exact model from public data takes a few minutes of CPU time
at the start (it is cached in Google Drive, so a resumed run skips it). Then Runtime > Run all.

**What to bring back.** The last cell downloads `world_model_gt_results.zip`. Unzip it into the
repository under `results/world_model/ground_truth/` on the branch `{BRANCH}`.
"""

RESUME = """### Stopping and resuming

- Every stage writes a done-marker in the work folder (`MyDrive/fourth_and_rl_world_model_gt`
  when Drive is mounted). Run all again after a disconnect and finished stages are skipped: the
  solved game is restored from the Drive copy, the synthetic games are reloaded, each size's
  training resumes from its checkpoint, and planning resumes from its saved chunks of states.
- The time budget is shared across sessions. Each size gets training time in proportion to its
  number of games, smallest first, so a short session still finishes the small sizes. Planning
  splits what is left between the trained sizes.
- Under the deadline the run plans fewer states, or refines fewer states to the target precision,
  and says so in the results. Press the stop button at any time: results so far are packaged.
- Changing `SIZES` keeps every finished size. To start completely fresh, delete the work folder.
"""

PARAMS = f"""# Parameters. The defaults are the real run: leave them as they are for Run all.
BUDGET_MIN = 90                          # total minutes, summed over sessions of this run
SIZES = [1000, 4000, 16000, 64000]       # synthetic training games per model
STOP_AFTER = None                        # or "setup", "generate", "train", "evaluate"
USE_DRIVE = True                         # keep the work folder on Google Drive
PRESET = None                            # None: chosen from the GPU; "smoke": small CPU test
GAME_DIR = None                          # a folder with a solved game; None builds it from data
FORCE = ()                               # stages to recompute, e.g. ("evaluate",)
REPO_URL = "{REPO_URL}"
BRANCH = "{BRANCH}"
REPO_DIR = "/content/fourth-and-rl"      # where the repository is cloned
WORKDIR = None                           # None: Drive folder, or /content/wm_gt without Drive"""

CLONE = """import os, subprocess, sys
if not os.path.isdir(REPO_DIR):
    subprocess.run(['git', 'clone', '--depth', '1', '-b', BRANCH, REPO_URL, REPO_DIR], check=True)
else:
    print('Using the existing checkout in', REPO_DIR)
print(subprocess.run(['git', '-C', REPO_DIR, 'log', '-1', '--format=%h %ci %s'],
                     capture_output=True, text=True).stdout)
# Install the repository (its dependencies include nflreadpy), plus pandas (preinstalled on
# Colab) and gymnasium (the toy game used by the tests imports it). Colab's torch is kept.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', REPO_DIR, 'nflreadpy',
                'pandas', 'gymnasium', 'pytest'], check=True)
NB_DIR = os.path.join(REPO_DIR, 'notebooks', 'world_model')
for p in (os.path.join(REPO_DIR, 'src'), NB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
print('Installed.')"""

DRIVE = """if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print('Drive mounted: the work folder is MyDrive/fourth_and_rl_world_model_gt')
    except Exception as e:  # not on Colab, or the mount was declined
        print('Drive not mounted; the work folder is lost when the runtime ends:', e)
else:
    print('USE_DRIVE is False: the work folder is lost when the runtime ends')"""

TESTS = """# The notebook tests that need no data (under two minutes on the CPU).
env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(REPO_DIR, 'src'), NB_DIR]))
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', NB_DIR],
                   cwd=NB_DIR, env=env, capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:])
if r.returncode != 0:
    raise RuntimeError('Tests failed: stop here and send the output above.')
print('Tests passed.')"""

PREFLIGHT = """import json
import ground_truth_run as G
import run as R

print(json.dumps(R.system_info(), indent=1))
# Picks the preset from the GPU (model size as in the real-data notebook), measures training and
# rollout speed for about 30 seconds on a new run, and prints minutes per size at that speed.
RUN = G.prepare(PRESET, SIZES, WORKDIR, BUDGET_MIN, GAME_DIR, tuple(FORCE))
print('Preset:', RUN.preset['name'], ' sizes:', RUN.preset['sizes'])
print('Work folder:', RUN.workdir)
print(json.dumps(G.estimate_minutes(RUN.preset)['per_size'], indent=1))
print(f'Minutes left in the budget: {RUN.remaining() / 60:.1f}')"""

STAGE_TEXT = {
    "setup": "Stage 1: the exact game model. Built from public play-by-play with the repository's "
             "own pipeline (download, fit, backward induction), or restored from Drive; then the "
             "saved ground truth is checked against it (recomputed here if it does not match).",
    "generate": "Stage 2: synthetic games from the exact model (the largest set once; smaller "
                "sizes are its first games) and a separate validation set.",
    "train": "Stage 3: one world model per size, smallest first, same architecture and settings, "
             "early stopping on the synthetic validation games.",
    "evaluate": "Stage 4: for each size, next-play NLL on validation games, then plan the 300 "
                "ground-truth fourth downs inside the learned model and grade the decisions.",
    "package": "Stage 5: results file, figures and the zip.",
}

SUMMARY = """print(G.summary(RUN))
from IPython.display import Image, display
for f in sorted((RUN.workdir / 'figures').glob('gt_*.png')):
    print(f.name)
    display(Image(filename=str(f)))"""

READING = """### Reading the results

- **Picks the truly best action**: the share of the 300 states where the action with the highest
  learned value is also the best by the true values. The margin columns repeat it for states
  where the best action beats the next one by more than 1 or 3 points of win probability, where
  a wrong pick is a real mistake rather than a near tie.
- **Regret**: true value of the best action minus true value of the chosen one, in points of win
  probability, averaged over states (and its 90th percentile). The dashed line is the regret of
  the coach-like policy that generated the training games; zero is perfect play.
- **Calibration**: learned value minus true value. The coverage figure splits it by how often the
  coaches take that action in that state. A model that overrates actions it has rarely seen
  shows positive errors on the left of that figure, and a planner that maximises over actions is
  drawn to exactly those errors.
- The intervals are 95% bootstrap intervals over states. The learned values have their own
  simulation noise, targeted at a standard error of 0.01 for the go-minus-best-kick difference;
  the results record how many states reached it."""

DOWNLOAD = """zip_path = RUN.workdir / G.ZIP_NAME
print(zip_path, f'{zip_path.stat().st_size / 1e6:.1f} MB')
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as e:
    print('Download it from the file browser instead:', e)"""


def build() -> nbformat.NotebookNode:
    cells = [new_markdown_cell(INTRO), new_markdown_cell(RESUME)]
    params = new_code_cell(PARAMS)
    params.metadata["tags"] = ["parameters"]
    cells += [params, new_code_cell(CLONE), new_code_cell(DRIVE),
              new_markdown_cell("Run the tests that need no data. The notebook stops here if any "
                                "fail."),
              new_code_cell(TESTS),
              new_markdown_cell("Preflight: hardware, preset, a short speed test and time "
                                "estimates."),
              new_code_cell(PREFLIGHT)]  # fmt: skip
    for stage in ("setup", "generate", "train", "evaluate", "package"):
        cells.append(new_markdown_cell(f"**{STAGE_TEXT[stage]}**"))
        cells.append(new_code_cell(
            f"out = G.notebook_stage(RUN, '{stage}', STOP_AFTER)\n"
            f"print('status:', out['status'], out.get('reason') or out.get('error') or '')"))
    cells += [new_markdown_cell("Results in plain language, and the figures."),
              new_code_cell(SUMMARY), new_markdown_cell(READING),
              new_markdown_cell("Download the results zip."), new_code_cell(DOWNLOAD)]
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"  # stable ids, so a rebuild changes only what changed
    nb = new_notebook(cells=cells)
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": [], "gpuType": "A100"}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    return nb


def main() -> None:
    nb = build()
    nbformat.validate(nb)
    OUT.write_text(nbformat.writes(nb))
    print(f"wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
