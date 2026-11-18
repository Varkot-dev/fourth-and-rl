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
MODULES = ("schema", "records", "model", "train", "simulate", "support", "toy", "run")
# Test files that must be in the notebook (all test_*.py files are embedded; these are
# checked by name so a renamed file cannot silently drop out).
REQUIRED_TESTS = ("test_amp_cpu.py", "test_model.py", "test_records.py", "test_run.py",
                  "test_speed.py", "test_support.py")

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
its checkpoint. Each configuration preregisters its test plan in
`prereg/<config hash>.json` before it reads the test seasons, and scores them once.

**What to bring back.** The last cell downloads `world_model_results.zip`. Unzip it into the
repo under `results/world_model/` on the branch `claude/world-model-colab`.
"""

CHANGES = """### What changed since the first run

The first run (Tesla T4, 18 November 2026) finished every stage, but its training ran on the
CPU and its fourth-down planning was cut to 67 decisions, 56 of them too close to call. This
version changes five things.

- **Training runs on the GPU.** The first run never moved the model to the GPU, so its 20
  minutes of training happened on the CPU. Training now moves the model to the GPU before the
  optimizer is built, keeps the training data on the GPU as ready-made tensors, and uses mixed
  precision (float16 with loss scaling on a T4, bfloat16 on newer GPUs) and a fused AdamW
  optimizer. Training should be many times faster; the training log shows the device,
  records per second and how each epoch's time splits between batching, the forward pass,
  the backward pass, validation and checkpointing.
- **Faster rollouts.** Each simulated play now encodes only the new record and reuses a
  key/value cache of everything before it, and the real game so far is stored once for all
  rollouts of a state. The preflight speed test measures this path at the batch size the
  planning stage uses, and scales the run down only if the GPU is less than half as fast as
  the preset assumes, never below 200 fourth downs. Sample sizes are larger: on a T4, 600
  win-probability states with 256 rollouts each, and 300 fourth downs with 512 rollouts per
  action.
- **Action-support guards and a field-goal check.** The planner no longer simulates field
  goals longer than the longest made field goal in the training seasons plus 3 yards, or
  punts from inside the opponent's 30. Every recommendation reports how often coaches made
  that call in comparable training situations; a call made fewer than 10 times is flagged as
  low support, and agreement with the coaches is reported separately for well-supported and
  low-support calls. A new figure, `fg_support.png`, compares the simulator's field-goal make
  rate and punt net yards with the training seasons. These rules were designed after the
  first run's test-season results had been seen, so they are a post hoc change and are
  labelled as such in the results.
- **Preregistration per model.** Each configuration writes its own
  `prereg/<config hash>.json` before it reads the test seasons, and scores them once. The
  first run's preregistration is kept. `results.json` lists every preregistration in the
  work folder, so it is visible that the test seasons have now been scored by more than one
  model.
- **A clean restart in the same Drive folder.** The preset saved by the first run is
  recognised as out of date and replaced automatically. The first run's results, figures and
  time ledger are moved to `archive/` in the Drive folder, so nothing from it is overwritten
  and the new run gets the full time budget. The downloaded data is reused.
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

PARAMS = """# Parameters. The defaults are the real run: leave them as they are for Run all.
# BUDGET_MIN: total minutes for all stages, summed over sessions of the same run.
BUDGET_MIN = 55
# STOP_AFTER: None runs everything; "data", "train", "eval", "test" or "plan" stops after
# that stage (the results file is still written).
STOP_AFTER = None
# USE_DRIVE: keep checkpoints and outputs on Google Drive, so a disconnect loses nothing.
USE_DRIVE = True
# FORCE: stages to recompute even if done, e.g. ("eval",) after a fix in simulate.py.
FORCE = ()
# FORCE_NEW_PRESET: True archives the saved run in the work folder and starts over.
FORCE_NEW_PRESET = False
# ALLOW_RESCORE: True lets this same configuration score the test seasons a second time
# (recorded). A new model or test plan never needs it.
ALLOW_RESCORE = False"""

SETUP = """# Colab already has torch, numpy, pandas, polars and matplotlib; only nflreadpy is added.
# Nothing is pinned so Colab's own torch build is kept.
!pip install -q nflreadpy pytest"""

DRIVE = """# Keep checkpoints and outputs on Google Drive (USE_DRIVE above) so a disconnect loses
# nothing. Accept the prompt that appears.
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print('Drive mounted: outputs go to MyDrive/fourth_and_rl_world_model')
    except Exception as e:  # not on Colab, or the mount was declined
        print('Drive not mounted, outputs go to /content/wm_run (lost on disconnect):', e)
else:
    print('USE_DRIVE is False: outputs go to /content/wm_run (lost on disconnect)')"""

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
print('Time budget (minutes):', BUDGET_MIN, ' stage ceilings:', R.PRESETS[name]['stage_min'])
if name == 'smoke':
    print('No GPU found: this runs the small CPU smoke preset (2021/2022/2023 seasons only).')
    print('For the real run: Runtime > Change runtime type > GPU, then Run all again.')
# About 30 seconds on a new run: measures training and rollout speed with the code the stages
# use, and scales the preset down only if this GPU is much slower than the preset assumes.
# On a resume the saved preset is reused, so finished stages keep their hashes; a preset
# saved by an older notebook is replaced (the old run is archived first).
RUN = R.prepare(name, budget_min=BUDGET_MIN, force=tuple(FORCE),
                force_new_preset=FORCE_NEW_PRESET, allow_rescore=ALLOW_RESCORE)
print('Work folder:', RUN.workdir)
print('Benchmark decision:', (RUN.preset.get('benchmark') or {{}}).get('decision', 'none'))
print('Preset in use:', json.dumps({{k: RUN.preset[k] for k in
      ('name', 'model', 'train', 'eval', 'test', 'plan')}}, indent=1))
print(f'Minutes left in the budget: {{RUN.remaining() / 60:.1f}}')"""

STAGE_TEXT = {
    "data": "Stage 1: download play-by-play season by season, build the splits, validate them.",
    "train": "Stage 2: train the world model (checkpointed every epoch; resumes on re-run).",
    "eval": "Stage 3: validation seasons only: NLL vs baseline, league stats, WP calibration.",
    "test": "Stage 4: write prereg/<config hash>.json, then score the test seasons once.",
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
# the data reports, every preregistration (prereg/) and the run log (no data, checkpoints
# or archived earlier runs).
zip_path = RUN.workdir / 'world_model_results.zip'
print(zip_path, f'{zip_path.stat().st_size / 1e6:.1f} MB')
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as e:
    print('Download it from the file browser instead:', e)"""


def build() -> nbformat.NotebookNode:
    cells = [new_markdown_cell(INTRO), new_markdown_cell(CHANGES), new_markdown_cell(RESUME),
             new_code_cell(PARAMS), new_code_cell(SETUP), new_code_cell(DRIVE)]
    cells.append(new_code_cell(f"import os\nos.makedirs('{CODE_DIR}', exist_ok=True)"))
    files = [HERE / f"{m}.py" for m in MODULES]
    files += sorted(HERE.glob("test_*.py"))
    missing = set(REQUIRED_TESTS) - {f.name for f in files}
    if missing:
        raise FileNotFoundError(f"test files missing from the notebook: {sorted(missing)}")
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
