# Leakage audit

Audited on 2026-11-17, read-only. Checks: `scripts/research/leakage_checks.py`, output in
`results/research/leakage_checks.json` (one thread, about five minutes). Every number
below is from that file unless a source is named.

**State of the code when read.** Committed and finished: `data.py`, `wp.py`,
`components.py`, `blown_leads.py`, their scripts and tests, `splits.py`. Unfinished and
uncommitted, read as they stood: `decision.py` (670 lines), `wp_nn.py` and
`scripts/fit_wp_nn.py` (no results file yet), `bootstrap.py` and
`scripts/bootstrap_decisions.py`, `mdp.py`, `mdp_drive.py`, `mdp_game.py` (grepped for
data access only). `scripts/questions/` does not exist. No guard from `splits.py` is
called anywhere yet.

**Verdict.** The headline win-probability numbers are clean: the saved model is exactly
the declared fit. One reported number is contaminated by a constant taken from a test
season. Seven further issues are design choices influenced by test data, unfair
comparisons, or hazards that are safe today only because of a file name or a config
value.

Counts: 1 critical, 7 moderate, 7 minor.

## Findings

| ID | Severity | Where | What leaks into what | Result affected, direction | Fix |
|---|---|---|---|---|---|
| C1 | Critical | `configs/components.yaml:31`; used at `components.py:601`, `fit_components.py:138-141` | `max_kick_distance: 68` is the longest kick of 2010 to 2025. The longest made kick in 2010 to 2023 is 66; the 68 comes from the test seasons. The "fit on 2010 to 2023" field-goal model therefore carries a test-season fact. 6 held-out attempts and 2 makes lie beyond 66 yards. | Held-out field-goal log loss in `results/m1_components.json`: 0.3695 as reported, 0.3798 with the development cap. Optimistic. The gap is larger than the differences between the three functional forms (0.001 to 0.004). Also in `components_2010_2023.joblib`, so `fg_wp` on 67 and 68 yard test attempts. | Derive the cap inside `FieldGoalModel.fit` from the rows it receives (`longest_made_` already exists), or a smooth tail. Rerun and restate the number. |
| M1 | Moderate | `fit_components.py:75-88, 91-151`; `docs/notes/m1_components.md:190-191` | Seven conversion variants and three field-goal variants are scored on 2024 to 2025. The field-goal default is the best of its three on those seasons and was kept. | Every `p_fg` and `p_convert` used to judge 2024 to 2025 decisions. Optimistic, small. The conversion default is not the best of its seven, so no selection is visible there. | Compare variants on 2022 to 2023 after fitting through 2021. Keep 2024 to 2025 for one final score of the chosen form. |
| M2 | Moderate, needs a decision | `decision.py:599-623`, `configs/decision.yaml:8`, `bootstrap.py:203`, `configs/bootstrap.yaml:21` | The kickoff start table is fitted on 2025 and used by the engine that evaluates 2024 and 2025 fourth downs. | Every action value after a score, on test plays. For: it encodes a rule, not skill, and earlier seasons are wrong for it. Against: it is fitted on returns from the games being judged, and it is the 2025 rule applied to 2024 plays (touchback at the 30 that year). Size small, direction unclear. | Use a fixed touchback spot per season from the rule book plus the 2010 to 2023 return shape, or log the exception through `assert_no_test_seasons(..., allow_test_seasons=True, reason=...)`. |
| M3 | Moderate | `scripts/r/nfl4th_reference.R:27`; `configs/decision.yaml` compare block | The nfl4th reference exists only for 2024 to 2025, and M2 is "done when" differences from it are explained. Any engine setting adjusted while reading that comparison is tuned on test plays. | Coach evaluation and nfl4th agreement on 2024 to 2025. Optimistic agreement. No outcome labels involved. | Produce the reference for 2022 to 2023, iterate there, and run 2024 to 2025 once. |
| M4 | Moderate, needs a decision | `fit_wp.py:340-379`; `configs/wp.yaml:22`; `configs/wp_nn.yaml:31`; `fit_wp_nn.py:545` | "Test is touched once" is not enforced. `--tune` does not exit, so every tuning run also prints test metrics. `num_leaves: 15` is not the tuning winner (31: 0.42819 against 0.42827), with no recorded reason. `SPREAD_DECAY`, the feature list and the learning rate have no provenance, and module, results and notes share one commit second, so history cannot show the order. The MLP headline uses `calibration: none` after the tree result showed Platt hurting on test, and four MLP variants are scored on test. | Headline log loss 0.4490 and the coming MLP comparison. Cannot be shown either way. For: all effects seen are within noise. Against: unverifiable is not clean. | Split `fit_wp.py` into tune, fit and evaluate steps. Evaluation loads test rows only through `load_test_plays` and appends to the access log. Name the MLP headline before the first test run. |
| M5 | Moderate | `fit_wp.py:29, 51-56, 434-441`; `scripts/blown_leads.py:126-135` | nflfastR `wp` and `vegas_wp` come from models whose training seasons are unknown to this repo (the nflfastR article could not be fetched from this machine). Ours is out-of-fold everywhere. | If their training covers a season, they are in-sample there and look better than they are. That biases the all-season and blown-lead comparisons against us, and makes the M3b statement that only our model departs asymmetric. On 2024 to 2025 they are almost certainly out-of-sample, but `wp` lacks the spread (favors us, disclosed). | Record the nflfastR training seasons with a source. Report comparisons separately inside and outside them. |
| M6 | Moderate (prospective) | `mdp.py:407, 499`; `scripts/research/endgame_clock_stats.py:661`; `configs/mdp.yaml` "measured" values | Clock tables, the coach table and the hard-coded timeout shares use 2018 to 2025 with no upper bound. | Any MDP or simulator result compared with real 2024 to 2025 coaching or outcomes would be in-sample. Fine for forward-looking artifacts. | Add a last-season bound, fit a development copy for anything scored on test, guard both fits. |
| M7 | Moderate (hazard) | `components.py:38, 1151`; `decision.py:640-643`; `bootstrap.py:235`; `wp.py:417-449` | No saved model records its seasons. `ComponentModels.load()` with no path returns the 2010 to 2025 bundle. `load_engine` and `ensure_replicate` reuse a file because it exists, not because it matches the config. | None today (verified below). One changed config or one default-path load silently contaminates every decision number. | Stamp on save, check on load, key caches by a hash of the config. Remove the default path. |
| m1 | Minor | `data.py:83` | The last row of 4,362 of 4,363 games is "END GAME" with the final score as its pre-play score: the sign of `score_differential` equals the label on 100% of them. They are never scored only because `yardline_100` is null. | None today (0 scoreable). | Drop rows with null `play_type` in `build_state_table`, and test it. |
| m2 | Minor | `wp_nn.py:120-133, 204-212` | `resample_games` gives copies new game ids, so with `calib=None` copies of one game can sit on both sides of the early-stopping split. | None today: the script always passes `calib`. | Split on the original id, or forbid the combination. |
| m3 | Minor | `wp.py:543-566` | Cross-fit folds 1 to 3 calibrate on 2024 to 2025; 2019 to 2021 are predicted with hyperparameters tuned on them; the out-of-fold 2024 and 2025 numbers come from a different model (trees 2010 to 2019) than the headline. | Blown-lead input for 2010 to 2021, slightly optimistic for 2019 to 2021. | Calibrate each fold on development seasons only. Label the two sets of test numbers. |
| m4 | Minor | `scripts/blown_leads.py:86-114, 160-165` | The discrete reference simulator's gap shape and starting probabilities are fitted to the same paths it then tests. | Pulls the reference toward the data: fewer departures found. | Fit the shape on half the seasons, test on the other half. |
| m5 | Minor | `fit_wp_nn.py:519` | The MLP is early-stopped and Platt-fitted on the same seasons. | `mlp_platt` slope slightly optimistic. | Note it, or stop on a held-out tenth of training games. |
| m6 | Minor | `fit_components.py:364-390, 393-447`; `configs/decision.yaml:28` | Structural constants (18-yard snap, miss spot, punt cutoff "71 punts in 16 seasons", era start) were read off all 16 seasons. | Negligible: rules and counts, not outcomes of judged plays. | Recompute on development seasons and confirm they do not move. |
| m7 | Minor | `fit_wp.py:181-183, 410-419, 496-498`; `tests/test_decision.py:370` | A test-row sanity check can fail the run; a pooled Platt map is scored on test after the fact (disclosed); a unit test reads 2024 states. | None on reported headline numbers. | Move the gate to calibration seasons; keep the diagnostic labelled. |

## Checked and found clean

- **Saved win-probability model is the declared fit.** Refit on 2010 to 2021 with Platt
  on 2022 to 2023, seed 20261117: 531,244 and 91,663 rows, the same as the saved file;
  largest prediction difference 1.1e-16 on all rows. A contrast model that also trained
  on 2024 to 2025 differs by up to 0.079 and would score 0.4443 against 0.4490, so the
  test is able to see a leak.
- **Cross-fit fold that predicts 2022 to 2025** (trees 2010 to 2019, Platt 2020 to 2021):
  refit matches `wp_predictions.parquet` to 2.2e-16 on 183,173 rows.
- **In-sample against out-of-fold.** The headline model is better in-sample by 0.0055
  log loss on average over 2010 to 2021, and test seasons (0.4228, 0.4758) are not
  better than training seasons. The gap is small because leaves hold 10,000 plays, so
  this check alone has little power; the refit check is the evidence.
- **Evaluation bundle.** `components_2010_2023.joblib` equals a fresh 2010 to 2023 fit
  with 2025 kickoffs (differences 0 to 4e-14) and differs from `components.joblib`
  (up to 0.065 in field-goal probability). The all-seasons bundle is loaded by no code
  path that scores plays (`decision.py:643`, `bootstrap.py:252` load named files).
- **Splits.** 3,224, 569 and 570 games; no game in two splits; one season, one result and
  one spread per game; no duplicate (game_id, play_id); labels of the two teams sum to 1.
- **Pre-play columns.** Score on 57,411 scoring rows equals the previous row and changes
  on the next (100%); every first score is at 0 to 0; touchdown rows show a clock above
  the next row (99.6%); opening kickoffs at 3600 s (99.8%); across 31,452 timeout rows
  the count drops by one on the following play (95.1%) and is unchanged on 0.2%.
- **Features.** `wp.FEATURES` (`wp.py:46-64`) and `_feature_exprs` (`wp.py:108-140`)
  read only the pre-play state, the pregame spread, home and the opening-kickoff
  receiver. No outcome column, no next-play join, no game or season aggregate.
  `add_next_state` (`components.py:68-96`) is used for targets only.
- **Within-game grouping.** Tuning scores whole seasons (`fit_wp.py:62-68`); calibration
  choice folds by game (`fit_wp.py:112-116`); the metric bootstrap, the MLP early-stop
  split and the M3 bootstrap all draw whole games (`wp.py:518-525`, `wp_nn.py:207-212`,
  `bootstrap.py:141-173`). No play-level split exists outside the simulation that
  demonstrates it (`wp_sample_size_sim.py:337`).
- **Preprocessing.** The MLP scaler is fitted on training rows after the split
  (`wp_nn.py:215`); spline knots are fixed constants (`components.py:434-443`).
- **M3 bootstrap** draws only 2010 to 2023, train and calibration groups separately.

## Recommended physical barriers

1. **Fit guards.** Call `splits.assert_no_test_seasons(df, what)` as the first line of
   `WPModel.fit` (`wp.py:341`, on `train` and `calib`), `NNWPModel.fit` (`wp_nn.py:185`),
   `ComponentModels.fit` (`components.py:1032`) and each sub-model `fit`
   (`components.py:424, 580, 659, 775`), `fit_kickoff` (`components.py:806`),
   `mdp.fit_tables` (`mdp.py:391`) and `mdp.fit_coach_table` (`mdp.py:488`). Thread
   `allow_test_seasons` and `reason` through as keyword arguments. `crossfit_predict`
   (`wp.py:569`) and the final bundle in `fit_components.py:598` are then the visible,
   logged exceptions.
2. **Stamps.** Store `splits.fit_stamp(...)` in `WPModel.save` (`wp.py:417`),
   `NNWPModel._blob` (`wp_nn.py:308`) and on `ComponentModels` before `joblib.dump`
   (`components.py:1147`). Add a `stamp` attribute to `DecisionEngine`. Call
   `splits.assert_can_score(stamp, states, what)` in `DecisionEngine.evaluate`
   (`decision.py:541`) whenever `states` has a `season` column, in `load_engine` after
   line 643, in `fit_wp.py` before line 362 and in `fit_wp_nn.py` before line 522.
3. **Two bundles, two names.** `components_dev.joblib` (2010 to 2023) for anything scored
   on test and `components_all.joblib` for forward-looking artifacts. Delete the default
   `MODEL_PATH` so `ComponentModels.load()` needs a path, and make loading the all-seasons
   file require a stated purpose.
4. **Loads.** Replace `data.load_plays()` with `splits.load_dev_plays()` in tuning and
   fitting code; allow `load_test_plays(purpose)` only in evaluation steps. Add a pytest
   that scans `src/` and `scripts/` for `load_plays(` and `read_parquet(` outside
   `data.py`, `splits.py`, `build_data.py` and an explicit allow-list.
5. **No test-derived constants.** A pytest that refits `FieldGoalModel` on development
   rows and asserts `max_kick_distance <= longest_made_`, and one that asserts no
   scoreable row has a null `play_type`.
6. **Leak canary in CI (data-marked).** Keep `leakage_checks.py` as a test: the saved
   model must match the refit, and a model trained with test seasons must not.
