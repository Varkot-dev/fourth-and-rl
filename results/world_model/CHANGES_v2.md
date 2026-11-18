# World model notebook: changes since the first Colab run

The first run is `colab_t4_run_summary.md` (T4, 18 November 2026, commit 812b7e6). This note
lists every change made for the second run and why. None of it has run on a GPU yet: it was
tested on the CPU (unit tests, a CPU bfloat16 autocast test, and the smoke preset end to
end), and the GPU-only code paths were reviewed by reading.

## Speed

1. **Training on the GPU.** `train_model` now moves the model to `cfg["device"]` before the
   optimizer is built. Why: the first run never moved it, so its 20 minutes of training ran
   on the CPU.
2. **Pre-built tensor batches.** Each split is built once as padded int16 tensors and kept on
   the GPU when it fits; a batch is an index operation, converted to int64 for the embedding
   lookups. Why: the old per-batch Python collate was slow. The new batches are tested to be
   identical to the reference ones.
3. **No per-step host synchronisation.** Losses are summed on the device and read once per
   epoch, record counts come from host-side lengths, and on CUDA AdamW is fused, which keeps
   the float16 loss scaler's overflow check on the device. If a torch build rejects fused
   AdamW, the default AdamW is used and logged.
4. **Batched field decoder for the training loss.** All ten heads run in one pass, with the
   same parameters and the same loss as the one-head-at-a-time reference (tested, including
   gradients).
5. **Mixed precision.** float16 with a GradScaler on a T4, bfloat16 without a scaler on
   compute capability 8 and up. The sampler (field decoder) still runs in float32 inside
   autocast, as before.
6. **Timing.** Every epoch logs records per second and splits its time into batching,
   host-to-device, forward, backward plus optimizer, validation and checkpointing.
7. **Key/value cache for rollouts.** Each simulated play encodes one new record per rollout.
   The real game history, shared by all rollouts of a state, is cached once and attended to
   with "cascade attention" (one softmax over the shared prefix and the rollout's own keys).
   Rollouts whose window outgrows 256 records re-encode exactly. Why: the first run's
   planning was limited by rollout speed (67 fourth downs). Tested to give the same random
   draws as re-encoding the whole window every play.
8. **Sampler.** Gumbel-max with every field's noise drawn in one call. Same distributions
   (tested), but a different random stream than the first run.
9. **Memory.** The cache is sized from 40% of total GPU memory, now also capped at 70% of
   the memory actually available (`torch.cuda.mem_get_info` plus memory PyTorch holds but
   does not use). A rollout group that runs out of GPU memory is retried in halves. Why:
   both are guards, so a shared or fragmented GPU cannot end the plan stage with an error.
10. **Presets.** On the T4, games per step go from 32 to 64 and the learning rate from 6e-4
    to 8.5e-4 (L4: 48 to 64 games, 7e-4; A100: 64 to 128 games, 7e-4). Each new rate is
    about the old one times the square root of the batch-size ratio. It was not tuned, so
    the second model differs from the first in batch size and learning rate as well as in
    device. Sample sizes on the T4: 256 rollouts per win-probability state (was 64); 300
    fourth downs (was 150) with 512 rollouts per action (was 128); 4,096 rollouts per
    action for the hypothetical (was 1,024). The plan stage's ceiling goes from 8 to 20
    minutes, because GPU training should leave time over and planning is the last
    compute stage.

## The preflight benchmark (fixes the first run's cut from 150 to 67 fourth downs)

11. The benchmark measures the current rollout path (the key/value cache) at the batch one
    plan-stage call uses on this GPU, from mid-game states. The untrained model has its END
    phase switched off and runs 90 steps per rollout, about as long as a real one. Training
    speed is read from a second epoch, after warm-up. Why: the first run's benchmark timed
    1,024 rollouts of an untrained model on the old path, and untrained rollouts end after a
    few plays (the old code's own comment said it undercounted).
12. Sizes are scaled down only when the measured speed is below half the preset's assumed
    speed, by the measured share, and never below 200 fourth downs, 300 win-probability
    states, 32 simulated games or 1,024 hypothetical rollouts (or the preset's own value if
    smaller). Otherwise the preset is kept, and each stage still stops at its deadline with
    its saved progress. The measurements, the decision and the minutes estimated at the
    measured speed are saved in `preset.json` and in `results.json` under `benchmark`.

## Resuming in the first run's Drive folder

13. **Preset definition version.** `preset.json` now stores a definition (a version string and
    a hash of the PRESETS entry). When the saved definition differs from the code, or is
    missing as in the first run's file, the preset is picked afresh and the reason is logged.
    Why: the first run's saved preset would otherwise be reused, with the old sizes.
14. **The previous run is archived on a fresh pick.** Its `results.json`, zip, figures,
    `preset.json` and time ledger move to `archive/run_<date>_<time>/`; its stage outputs,
    `prereg.json` and log are copied there. Why: the first run's full results are only in
    that Drive folder and would have been overwritten. Also, its ledger records about 41
    minutes used, which would have left the new run 14 of its 55 minutes.
15. **Stale outputs are not reported.** `results.json` and the status table only report a
    stage output whose stage hash matches the current configuration. Why: a stage that did
    not run in the new configuration (for example, stopped after training) would otherwise
    have shown the first run's results as if they were the new run's.

## Preregistration per model (fixes the refused test stage)

16. Each configuration (the full config hash: data, model, training, evaluation, test and
    plan settings) writes `prereg/<hash>.json` once, before it reads the test seasons, and
    never overwrites it. Each stage start is logged in `prereg/<hash>.attempts.json`, and a
    completed scoring in `prereg/<hash>.scored.json`. A second scoring of the same hash is
    refused unless `allow_rescore` is set, in which case it is recorded. Why: the first run's
    single `prereg.json` made the test stage refuse any new model.
17. The plan stage, which also reads the test seasons, writes the preregistration too if the
    test stage has not run.
18. The first run's `prereg.json` and `outputs/test_scored.json` are imported under its own
    hash (4bf68c37adae789a) and left in place. `results.json` lists every preregistration in
    the work folder (`preregistrations`) and how many configurations have scored the test
    seasons (`test_scorings`). After the second run that is two, and only the first model
    was chosen before any test-season result existed.

## Action support for the fourth-down planner (post hoc)

19. **These rules were designed after the first run's test-season fourth-down results had
    been seen**, and are not part of that run's preregistered plan. `results.json` repeats
    this in `stages.plan.support.note`. Why add them: the first run's planner can recommend
    calls coaches almost never make, where the simulator has little data to be right about
    (the offline reinforcement learning problem of distribution shift).
20. From the training seasons only: support tables of what coaches did on fourth down, by
    yard line (within 3) and yards-to-go bucket, plus empirical field-goal make rates and
    punt net yards by yard line.
21. Hard rules, passed to `plan_fourth_downs` as an action mask. No field goal beyond the
    longest made one in training plus 3 yards; no punt inside the opponent's 30; going for
    it is always allowed. Masked actions are listed in `stages.plan.masked_actions`.
22. Soft flags. Every recommendation records how often coaches made that call in comparable
    training states, and "low support" means fewer than 10 times. Agreement with coaches is
    reported separately for well-supported and low-support recommendations
    (`agreement_by_support`). Field-goal recommendations get their own summary
    (`fg_recommendations`).
23. Diagnostic and figure. The simulator's one-play field-goal make rate and punt net yards
    with the kick forced from real test fourth downs, against the training seasons by
    distance band (`stages.plan.support.diagnostic`, `fg_support.png`).
24. The plan stage hash includes the support settings, so a resumed run recomputes the plan
    with the rules.

## Notebook

25. The parameters cell now holds `BUDGET_MIN = 55`, `STOP_AFTER = None`, `USE_DRIVE = True`,
    `FORCE = ()`, `FORCE_NEW_PRESET = False` and `ALLOW_RESCORE = False`. A new markdown cell,
    "What changed since the first run", summarises this note. The notebook embeds
    `support.py` and every test file. The build fails if `test_speed.py`, `test_support.py`
    or `test_amp_cpu.py` is missing.
26. Tests added: `test_speed.py` (fast paths against their references; out-of-memory retry;
    fused-AdamW fallback; the cache memory plan with a simulated GPU), `test_support.py`,
    `test_amp_cpu.py` (a training step, validation, cascade attention with bfloat16 caches
    and rollouts under CPU bfloat16 autocast), and in `test_run.py` (per-model
    preregistration, import of the first run's files, stale outputs, preset versioning and
    archiving, the benchmark rule, and the benchmark with real modules).

## Not verified

- Nothing above has run on a GPU. Untested there: the float16 GradScaler with fused AdamW,
  the CUDA memory planning, pinned-memory transfers, and the actual speeds.
- The preset's assumed speeds (T4: 60,000 training records/s, 40,000 rollout steps/s) are
  estimates. The T4 plan needs about 42 million rollout steps: about 18 minutes at the
  assumed speed, so it fits the 20-minute ceiling only at 36,000 steps/s or more. Below
  that the plan stage ends partial with its finished fourth downs saved; the benchmark
  scales it down only below 20,000 steps/s.
