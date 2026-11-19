# Running the world-model experiment on Google Colab

This runs the NFL world-model experiment end to end in one Colab GPU session of about 55
minutes: it downloads nflverse play-by-play for 2010 to 2025, trains a next-play transformer,
evaluates it on the validation seasons, scores the test seasons once, and produces
fourth-down recommendations. Everything it needs is inside the notebook.

## 0. What is different from the first run

The first run (T4, 18 November 2026) is summarised in
`results/world_model/colab_t4_run_summary.md`; every change since then, and why, is in
`results/world_model/CHANGES_v2.md`. In short:

- **Training is on the GPU now.** The first run trained on the CPU by mistake (the model was
  never moved to the GPU). Expect training to be much faster. Each epoch's log line shows
  records per second and how the time splits between batching, forward, backward plus
  optimizer, validation and checkpointing; the train stage in `results.json` records the
  device, the precision (float16 with loss scaling on a T4) and the optimizer.
- **Rollouts use a key/value cache**, so simulations are much cheaper, and the samples are
  larger (on a T4: 600 win-probability states with 256 rollouts each, 300 fourth downs with
  512 rollouts per action). The planning stage may now use whatever time the earlier stages
  leave (up to 20 minutes).
- **The preflight speed test no longer cuts the run on a rough guess.** It measures the
  current rollout path at the planning stage's batch size and scales sizes down only if the
  GPU is below half the speed the preset assumes, never below 200 fourth downs (300
  win-probability states). Its measurements and decision are printed, and saved in
  `results.json` under `benchmark`.
- **Action-support guards** for the fourth-down planner, a field-goal and punt diagnostic,
  and a new figure, `fg_support.png` (see "Reading the fourth-down results" below).
- **Preregistration is per model** (section 4).
- **The same Drive folder is reused cleanly.** The preset saved there by the first run is
  recognised as out of date and replaced. The first run's `results.json`, zip, figures and
  time ledger are moved to `archive/run_<date>_<time>/` in the Drive folder (its stage
  outputs are copied there), so nothing is overwritten and the new run gets the full 55
  minutes. The downloaded seasons are reused, so the data stage takes seconds.

## 1. Open the notebook

Use either route:

- From GitHub: in Colab choose File > Open notebook > GitHub, sign in if asked, pick this
  repository, switch the branch to `claude/world-model-colab` and open
  `notebooks/world_model_colab.ipynb`. While the repository is private, Colab needs the
  "Include private repos" box ticked.
- By upload: download `notebooks/world_model_colab.ipynb` from the branch, then in Colab
  choose File > Upload notebook.

## 2. Choose a GPU runtime

Runtime > Change runtime type > Hardware accelerator: GPU. A T4 is enough. An L4 or A100 runs
larger samples in the same time. The notebook picks its settings from the GPU it finds; with
no GPU it runs a small CPU smoke test instead, which is not the real experiment.

## 3. Run all

The parameters cell near the top already holds the right defaults for the real run:
`BUDGET_MIN = 55`, `STOP_AFTER = None`, `USE_DRIVE = True`, no forced stages, no fresh preset,
no rescoring. Leave it as it is.

Runtime > Run all. In order, the notebook:

1. installs `nflreadpy` (everything else is already on Colab);
2. asks to mount Google Drive. Accept it: checkpoints and stage outputs are then saved in
   `MyDrive/fourth_and_rl_world_model`, so a disconnect loses nothing;
3. writes the code files and runs the unit tests, and stops if any fail;
4. prints the GPU, memory, disk and chosen preset, and on a new run a speed test of about
   30 seconds that shrinks the sample sizes only if the GPU is much slower than expected
   (the decision is printed as "Benchmark decision: ...");
5. runs the six stages, each printing its progress: data, train, eval, test, plan, package;
6. prints a plain-language summary and the figures, and downloads the results zip.

Every stage watches the clock. If time runs short it uses fewer states or rollouts, or skips
itself and records why, so the run always ends with a results file.

## 4. Stopping, resuming and disconnects

- Press the stop button at any time. The current stage saves what it has (a checkpoint after
  every training epoch; each finished batch of evaluation and planning states), writes
  `results.json`, and prints "Stopped safely at <stage>. Run all again to continue from here."
- Later, reconnect, choose the same GPU runtime, and Run all again, even with an updated
  notebook. Mount Drive again when asked. Finished stages are skipped and training continues
  from its last checkpoint. Stages are keyed on the configuration and the data, not on the
  code, so a corrected notebook still resumes. The time already used is remembered.
- Keep Google Drive mounted, and do not delete the `fourth_and_rl_world_model` folder in
  Drive: it holds all saved progress.
- To train tonight and evaluate tomorrow, set `STOP_AFTER = "train"` in the parameters cell
  near the top, Run all, and the next day set it back to `None` and Run all again.
- At the start of every run the preflight cell prints a status table: which stages are done,
  where the checkpoint is, its last epoch and best validation NLL, and minutes used so far.


The test seasons (2024 and 2025) are scored once per configuration. Before a configuration
(model, training, evaluation, test and planning settings, summarised in one config hash)
reads the test seasons, the notebook writes `prereg/<config hash>.json` with the exact test
plan; that file is never overwritten. A second scoring of the same configuration is refused
unless `ALLOW_RESCORE = True` (which is recorded). A new model has a new hash, so it gets
its own preregistration and its own scoring; the first run's `prereg.json` is kept and
imported under its own hash. `results.json` lists every preregistration in the work folder
under `preregistrations`, and `test_scorings` says how many configurations have scored the
test seasons. After this run that will be two: the first run's model and this one. Only the
first was chosen before any test-season result existed.

Without Drive the outputs live in `/content/wm_run` and are lost when the runtime ends. To
start over in the same folder (keeping the data), set `FORCE_NEW_PRESET = True` once; to
start completely fresh, delete the `fourth_and_rl_world_model` folder on Drive.

## 5. What to bring back

The last cell downloads `world_model_results.zip` (if the browser blocks it, find it in the
Colab file browser under the work folder). It contains `results.json` with every number,
the figures, each stage's output, the data reports, every preregistration (`prereg/`) and
the run log. It does not contain data, model checkpoints or the archived first run (that
stays in the Drive folder under `archive/`).

Unzip it into the repository under `results/world_model/` on the branch
`claude/world-model-colab`, then commit it or hand it back for review.

## 6. Reading the fourth-down results

The planner simulates go, field goal and punt from each sampled test fourth down and picks
the call with the highest simulated win probability. A simulator trained on coaches'
choices knows little about calls coaches almost never make, and a planner that maximises
over actions is drawn to exactly those. So:

- **Hard rules.** Field goals longer than the longest made field goal in the training
  seasons plus 3 yards, and punts from inside the opponent's 30, are not simulated. Each
  ruled-out action is listed in `stages.plan.masked_actions`, and the summary says how many
  states had one. The hypothetical (4th and 2 at the offense's own 40) is a 78-yard field
  goal, so its field goal is ruled out and its figure bar reads "not allowed (hard rule)".
- **Low-support flags.** Every decision row has `sup_n` (comparable training fourth downs:
  yard line within 3, same yards-to-go bucket), `sup_go`, `sup_fg`, `sup_punt` (what coaches
  did there), and `low_support`, which is true when coaches made the recommended call fewer
  than 10 times in those states. A low-support recommendation rests on the simulator's
  extrapolation rather than on observed outcomes: read it as a hypothesis, not a finding.
  `agreement_by_support` gives agreement with the coaches separately for well-supported and
  low-support recommendations; the well-supported number is the one to quote.
- **`fg_support.png`.** Left: for each 5-yard band of field-goal distance, the simulator's
  make rate (blue, with a 95% interval) when a kick is forced from real test fourth downs,
  against what kicks in the training seasons did (orange), with the number of training
  attempts underneath. Right: the same for punt net yards by yard line. Shaded areas are
  ruled out by the hard rules. Where the blue line sits well above orange, the simulator is
  optimistic about kicking and its field-goal recommendations there deserve doubt; thin
  training bars mean little data in that band. Training rates at long range are for the
  kicks coaches chose to attempt, so they are not a clean benchmark. The summary also gives
  the simulator's make rate on the recommended field goals against the training rate at
  the same yard lines.
- These rules and flags were designed after the first run's test-season results had been
  seen, so they are a post hoc change; `results.json` says so in `stages.plan.support.note`.

## 7. If something fails

If the unit tests fail, or a stage prints `status: error`, copy the printed error and the
end of `run.log` from the work folder and send them back. The other stages still run where
they can, and the package stage still writes `results.json` recording the error.

## 8. The second notebook: the world model against a known truth

`notebooks/world_model_ground_truth.ipynb` asks how much data the world model needs to recover
the optimal fourth-down policy, using games from the repo's exact game model, where the true
value of every action is known. Design, metrics and limits:
`results/world_model/GROUND_TRUTH_DESIGN.md`. It is independent of the first notebook: its own
Drive folder (`MyDrive/fourth_and_rl_world_model_gt`), no shared files.

- **Open it.** [Open in Colab](https://colab.research.google.com/github/Varkot-dev/fourth-and-rl/blob/claude/world-model-colab/notebooks/world_model_ground_truth.ipynb)
  (the badge at the top of the notebook is the same link). Unlike the first notebook it embeds
  no code: it clones the branch `claude/world-model-colab` from GitHub and installs it, so it
  always runs the code that is pushed to that branch.
- **GPU.** An A100 is recommended. At the presets' assumed speeds (not yet measured) an A100
  plans all four sizes and refines most states to the target precision within the budget; an L4
  plans every state but refines fewer; a T4 completes every size with coarser estimates and
  records what it cut. The preflight cell measures the GPU and prints minutes per size.
- **Budget.** `BUDGET_MIN = 90` by default, shared across sessions. The first few minutes rebuild
  the exact game from public data on the CPU (download, fit, solve; about 2 GB, cached in the
  Drive folder so a resumed run restores it instead). Training gets 45% of what is left, in
  proportion to each size's number of games, smallest first; planning splits the rest between
  the trained sizes. Leave the other parameters as they are.
- **Resuming.** Run all again after a disconnect: finished stages are skipped, training resumes
  from each size's checkpoint and planning from its saved chunks. `STOP_AFTER = "train"` stops
  after training; changing `SIZES` keeps every finished size.
- **What to send back.** The last cell downloads `world_model_gt_results.zip`
  (`gt_results.json` with every number and the scaling table, the three figures
  `gt_regret_accuracy.png`, `gt_q_scatter.png` and `gt_overestimation_by_coverage.png`, every
  stage's output, the run log and the preset). Unzip it into `results/world_model/ground_truth/`
  on the branch. If a cell fails, send its output and the end of `run.log` from the work folder.
- Expect the setup stage to say that the saved ground truth does not match the game built in
  Colab. nflverse has revised its data since `gt_v1` was computed (on 19 November 2026 a fresh
  download had two fewer 2011 plays and a few more 2018 to 2022 kickoffs than the cached copy),
  so the rebuilt game differs slightly from the one `gt_v1` used. The stage then recomputes the
  ground truth for the game it built (300 states, 20,000 rollouts per action, a few minutes on
  the Colab CPUs) and the results say which ground truth was used.
