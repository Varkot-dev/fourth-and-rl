# World model: first Colab run (summary as printed by run.py)

Run by the owner on Google Colab on 2026-11-18 with `notebooks/world_model_colab.ipynb`
(commit 812b7e6 on this branch). Pasted verbatim from the notebook's final summary cell.
The full `results.json` and figures (calibration.png, fourth_down.png, nll_vs_baseline.png,
training_curves.png) are in the owner's Google Drive folder `fourth_and_rl_world_model` and
are not yet in the repo. Numbers labelled "nflverse ... (theirs)" are nflfastR's models,
scored on the same states, never blended with ours.

```
Preset t4, config 4bf68c37adae789a, GPU Tesla T4.
- data: done in 31.9s
- train: done in 1204.1s
- eval: done in 438.8s
- test: done in 342.4s
- plan: done in 422.9s
  train: 3224 games, 529513 records, seasons 2010-2021
  val: 569 games, 91412 records, seasons 2022-2023
  test: 570 games, 90594 records, seasons 2024-2025
  model: 6,495,984 parameters, trained 1203.8s
  validation NLL per transition: model 9.466 vs frequency baseline 12.775 (lower is better)
  validation win probability, ours_simulation: log loss 0.5167 (95% game bootstrap 0.4477 to 0.5863), Brier 0.1564, 478 states
  validation win probability, nflverse_home_wp (theirs): log loss 0.4821 (95% game bootstrap 0.4370 to 0.5347), Brier 0.1490, 478 states
  validation win probability, nflverse_vegas_home_wp (theirs): log loss 0.4542 (95% game bootstrap 0.4059 to 0.5046), Brier 0.1379, 478 states
  test NLL per transition: model 9.654 vs frequency baseline 12.998 (lower is better)
  test win probability, ours_simulation: log loss 0.5252 (95% game bootstrap 0.4616 to 0.5868), Brier 0.1655, 372 states
  test win probability, nflverse_home_wp (theirs): log loss 0.5058 (95% game bootstrap 0.4563 to 0.5593), Brier 0.1620, 372 states
  test win probability, nflverse_vegas_home_wp (theirs): log loss 0.4628 (95% game bootstrap 0.4087 to 0.5169), Brier 0.1427, 372 states
  fourth downs: model agrees with the coach on 45% of 67 test decisions, 73% of the 11 that are not too close to call
  hypothetical (up 4, 4:00 left, 4th and 2 at own 40): go 0.898, field goal 0.780, punt 0.790; best GO
```

## Reading (written in the repo session, not by the script)

- Next-play prediction generalizes: 26% lower negative log likelihood than a frequency
  baseline on validation, and the same on the test seasons.
- Win probability by simulation, a quantity the model was never trained on directly, is
  within about 0.02 log loss of nflfastR without the spread on the test seasons (intervals
  overlap) and about 0.06 behind nflfastR with the spread. Few states were scored (372 to
  478), so the intervals are wide.
- Fourth-down planning was limited by rollout speed on a T4: 67 test decisions, 56 of them
  too close to call. The hypothetical late-lead case agrees in direction with the exact
  game solution (go), with higher levels and a larger margin; its standard errors are in
  results.json, not in this summary.
- Next: faster rollouts (a key-value cache or a stronger GPU) for many more simulations
  per decision, then the synthetic ground-truth experiment against the exact solution.
