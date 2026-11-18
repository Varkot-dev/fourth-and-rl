"""Build notebooks/world_model_colab.ipynb from the module files in this folder.

The notebook is self-contained: every module and test file is embedded in a %%writefile
cell, so it needs nothing from the repo. Re-run this script whenever a module changes:

    python notebooks/world_model/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "world_model_colab.ipynb"
CODE_DIR = "/content/wm_code"
MODULES = ("schema", "records", "model", "train", "simulate", "toy", "run")

INTRO = """# NFL world model on Colab: next-play transformer as a fourth-down simulator

**What this is.** A causal transformer is trained on real NFL play-by-play (nflverse, via
`nflreadpy`): training seasons 2010 to 2021, validation 2022 to 2023, test 2024 to 2025. It
reads a game's records so far and predicts the next play's transition. It is then used as a
simulator: from any game state it plays the rest of the game many times.

**What it answers.**
1. Does the model predict the next play better than a frequency baseline (per-field NLL)?
2. Do simulated games look like real games (points, plays, ties, home win share)?
3. Is win probability by simulation calibrated, compared with nflverse's own `home_wp` and
   `vegas_home_wp` on the same states (log loss, Brier, game-bootstrap intervals)?
4. On test-season fourth downs, what does the simulator recommend (go, field goal, punt), and
   how often does that match the real coach? Plus one constructed state: up 4, 4:00 left,
   fourth and 2 at the offense's own 40.

**How long it takes.** About 55 minutes on a T4, L4 or A100 GPU. The run sizes itself to the
GPU it finds and to a fixed time budget; each stage checks the clock and scales down or skips
(with the reason recorded), so it always ends with a results file.

**Choosing a GPU.** Runtime > Change runtime type > Hardware accelerator: GPU (T4 is fine;
L4 or A100 run larger samples). Then Runtime > Run all.

**If you are disconnected.** Reconnect and Run all again. With Google Drive mounted, every
finished stage and the training checkpoint are kept in
`MyDrive/fourth_and_rl_world_model`; finished stages are skipped and training resumes from
its checkpoint. The test seasons are scored only once (preregistered in `prereg.json`).

**What to bring back.** The last cell downloads `world_model_results.zip`. Unzip it into the
repo under `results/world_model/` on the branch `claude/world-model-colab`.
"""

RESUME = """### Stopping and resuming

- Press the stop button at any time. The current stage saves what it has (training keeps a
  checkpoint after every epoch; evaluation and planning save each finished batch of states),
  `results.json` is written with every stage's status, and the cell prints
  "Stopped safely at <stage>. Run all again to continue from here."
- Run all again later, even after a disconnect or with an updated notebook: finished stages
  are skipped and training continues from its checkpoint. Stages are keyed on the
  configuration and the data, not on the code, so an edited notebook still resumes.
- Keep Google Drive mounted, and do not delete the `fourth_and_rl_world_model` folder in
  Drive: that folder is the saved progress.
- To train tonight and evaluate tomorrow, set `STOP_AFTER = "train"` in the next cell, Run
  all, and tomorrow set it back to `None` and Run all again.
"""

PARAMS = """# Parameters.
# STOP_AFTER: None runs everything; "data", "train", "eval", "test" or "plan" stops after
# that stage (the results file is still written).
STOP_AFTER = None
# FORCE: stages to recompute even if done, e.g. ("eval",) after a fix in simulate.py.
FORCE = ()"""

SETUP = """# Colab already has torch, numpy, pandas, polars and matplotlib; only nflreadpy is added.
# Nothing is pinned so Colab's own torch build is kept.
!pip install -q nflreadpy pytest"""

DRIVE = """# Optional: keep checkpoints and outputs on Google Drive so a disconnect loses nothing.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted: outputs go to MyDrive/fourth_and_rl_world_model')
except Exception as e:  # not on Colab, or the mount was declined
    print('Drive not mounted, outputs go to /content/wm_run (lost on disconnect):', e)"""

TESTS = f"""import subprocess, sys
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider',
                    '{CODE_DIR}'], cwd='{CODE_DIR}', capture_output=True, text=True)
print(r.stdout[-4000:], r.stderr[-2000:])
if r.returncode != 0:
    raise SystemExit('Unit tests failed: stop here and send the output above.')
print('Unit tests passed.')"""

PREFLIGHT = f"""import json, os, sys, time
sys.path.insert(0, '{CODE_DIR}')
os.chdir('{CODE_DIR}')
import run as R

info = R.system_info()
print(json.dumps(info, indent=1))
name = R.pick_preset()
print('Chosen preset:', name)
print('Time budget (minutes):', R.PRESETS[name]['budget_min'],
      ' stage ceilings:', R.PRESETS[name]['stage_min'])
if name == 'smoke':
    print('No GPU found: this runs the small CPU smoke preset (2021/2022/2023 seasons only).')
    print('For the real run: Runtime > Change runtime type > GPU, then Run all again.')
# About 30 seconds: measures training and rollout speed and scales the preset if needed.
# On a resume the saved preset is reused, so finished stages keep their hashes.
RUN = R.prepare(name, force=tuple(FORCE))
print('Work folder:', RUN.workdir)
print('Preset in use:', json.dumps({{k: RUN.preset[k] for k in
      ('name', 'model', 'train', 'eval', 'test', 'plan')}}, indent=1))
print(f'Minutes left in the budget: {{RUN.remaining() / 60:.1f}}')"""

STAGE_TEXT = {
    "data": "Stage 1: download play-by-play season by season, build the splits, validate them.",
    "train": "Stage 2: train the world model (checkpointed every epoch; resumes on re-run).",
    "eval": "Stage 3: validation seasons only: NLL vs baseline, league stats, WP calibration.",
    "test": "Stage 4: write prereg.json, then score the test seasons once.",
    "plan": "Stage 5: fourth-down recommendations on test-season fourth downs.",
    "package": "Stage 6: results.json, figures and the zip.",
}

SUMMARY = """print(R.summary(RUN))
from IPython.display import Image, display
for f in sorted((RUN.workdir / 'figures').glob('*.png')):
    print(f.name)
    display(Image(filename=str(f)))"""

DOWNLOAD = """# Download the results. Unzip into the repo under results/world_model/ on the branch
# claude/world-model-colab, then commit. The zip holds results.json, figures, stage outputs,
# the data reports, prereg.json and the run log (no data or checkpoints).
zip_path = RUN.workdir / 'world_model_results.zip'
print(zip_path, f'{zip_path.stat().st_size / 1e6:.1f} MB')
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as e:
    print('Download it from the file browser instead:', e)"""


def build() -> nbformat.NotebookNode:
    cells = [new_markdown_cell(INTRO), new_markdown_cell(RESUME), new_code_cell(PARAMS),
             new_code_cell(SETUP), new_code_cell(DRIVE)]
    cells.append(new_code_cell(f"import os\nos.makedirs('{CODE_DIR}', exist_ok=True)"))
    files = [HERE / f"{m}.py" for m in MODULES]
    files += sorted(HERE.glob("test_*.py"))
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        cells.append(new_code_cell(f"%%writefile {CODE_DIR}/{path.name}\n{path.read_text()}"))
    cells.append(new_markdown_cell("Run the unit tests (CPU, under a minute). "
                                   "The notebook stops here if any fail."))
    cells.append(new_code_cell(TESTS))
    cells.append(new_markdown_cell("Preflight: hardware, chosen preset and a short benchmark."))
    cells.append(new_code_cell(PREFLIGHT))
    for stage in ("data", "train", "eval", "test", "plan", "package"):
        cells.append(new_markdown_cell(f"**{STAGE_TEXT[stage]}**"))
        cells.append(new_code_cell(
            f"out = R.notebook_stage(RUN, '{stage}', STOP_AFTER)\n"
            f"print('status:', out['status'], out.get('reason') or out.get('error') or '')"))

    cells.append(new_markdown_cell("Results in plain language."))
    cells.append(new_code_cell(SUMMARY))
    cells.append(new_markdown_cell("Download the results zip."))
    cells.append(new_code_cell(DOWNLOAD))
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
