# Running the world-model experiment on Google Colab

This runs the NFL world-model experiment end to end in one Colab GPU session of about 55
minutes: it downloads nflverse play-by-play for 2010 to 2025, trains a next-play transformer,
evaluates it on the validation seasons, scores the test seasons once, and produces
fourth-down recommendations. Everything it needs is inside the notebook.

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

Runtime > Run all. In order, the notebook:

1. installs `nflreadpy` (everything else is already on Colab);
2. asks to mount Google Drive. Accept it: checkpoints and stage outputs are then saved in
   `MyDrive/fourth_and_rl_world_model`, so a disconnect loses nothing;
3. writes the code files and runs the unit tests, and stops if any fail;
4. prints the GPU, memory, disk and chosen preset, and runs a 30 second speed test that
   shrinks the sample sizes if the GPU is slower than expected;
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


The test seasons (2024 and 2025) are scored only once. Before they are scored the notebook
writes `prereg.json` with the configuration hash and the exact test plan. If the
configuration changes after scoring, the test stage refuses to run again; that is deliberate.

Without Drive the outputs live in `/content/wm_run` and are lost when the runtime ends. To
start completely fresh, delete the `fourth_and_rl_world_model` folder on Drive.

## 5. What to bring back

The last cell downloads `world_model_results.zip` (if the browser blocks it, find it in the
Colab file browser under the work folder). It contains `results.json` with every number,
the figures, each stage's output, the data reports, `prereg.json` and the run log. It does
not contain data or model checkpoints.

Unzip it into the repository under `results/world_model/` on the branch
`claude/world-model-colab`, then commit it or hand it back for review.

## 6. If something fails

If the unit tests fail, or a stage prints `status: error`, copy the printed error and the
end of `run.log` from the work folder and send them back. The other stages still run where
they can, and the package stage still writes `results.json` recording the error.
