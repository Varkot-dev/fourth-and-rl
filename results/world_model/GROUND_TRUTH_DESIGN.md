# World model against a known truth: design of the ground-truth experiment

Question: how much data does a learned simulator need to recover the optimal fourth-down
policy? On real data the right answer is unknown, so the real-data notebook can only compare the
world model with coaches and with nflverse. Here the games come from a model whose answer is
known: the repo's exact game model (the solved MDP and its `Simulator`, M4). The world model is
trained on its games and graded against its true values.

Code: `notebooks/world_model/ground_truth_run.py` (stages), `synthetic.py` (games, evaluation
states, ground truth), `build_gt_notebook.py` (the notebook),
`notebooks/world_model_ground_truth.ipynb`. Tests: `test_ground_truth_run.py`. No result from
this experiment exists yet: nothing below is a finding, and only the CPU smoke test has run.

## Design

1. **Generator.** `synthetic.simulate_games`: regulation games from the opening kickoff, both
   teams on the coach-like policy (`Simulator.coach_actions`, coach rules for timeouts and
   tries), written in the world model's record format. Plays are play modes (neutral, protect,
   chase) instead of run or pass, so "go" means a play mode.
2. **Ground truth.** `ground_truth/gt_v1`: 300 held-out fourth downs (stratified by clock,
   within 16 points, plus the late-lead headline state), each with
   - `q_coach`: win probability of each action when both teams then play the coach-like policy,
     by Monte Carlo in the exact model (40,000 rollouts per action, standard error at most
     0.0025); this is the target, because the world model learns from coach-like games;
   - `q_star`: the solved value (optimal play afterwards), reported as a second comparison;
   - `allowed_model`: whether the exact model allows the field goal and the punt there;
   - `coach_probs`: how often the coach-like policy goes, kicks or punts there.
3. **Data sizes.** 1,000, 4,000, 16,000 and 64,000 games. The real development data
   (2010 to 2021 training seasons) is about 3,200 games, so 4,000 is the realistic size. The
   64,000-game set is generated once; each smaller set is its first games (chunk seeds make
   that exact). A separate validation set (seed 20261122) has 1,000 games for early stopping
   and 1,000 for the reported NLL.
4. **Same model for every size.** Architecture and optimizer come from the GPU preset of the
   real-data notebook. One epoch is at least 4,000 games (a small set is repeated), so warmup,
   patience and validation cadence mean the same thing for every size. Each size gets training
   time in proportion to its number of games (at least 2 minutes), smallest first; the cosine
   schedule's length is what that budget fits at the measured speed, and early stopping on the
   synthetic validation games decides when to stop. Every size's stopping reason is recorded.
5. **Planning.** For each state the learned model simulates GO, FG and PUNT from the state's
   own game history. GO samples a play mode from the model's go-set probabilities; the coach's
   mode is deterministic, so a model that learned the data plays the coach's mode (the results
   report the model's mean probability of it). Actions the exact model does not allow are
   masked. Rollouts use the key/value cache. Two phases: every state with 512 rollouts per
   action, then the closest calls first, GO and the best kick (and the other kick when it is
   tied with the best) topped up to the rollouts that a standard error of 0.01 for the
   go-minus-best-kick difference needs (at most 8,192). A deadline cuts phase A (fewer states,
   a random subset because the order is a fixed permutation with the headline state first) or
   phase B (fewer refined states); both are recorded.
6. **Metrics** (each with a 95% bootstrap over states):
   - share of states where the learned planner picks the truly best action by `q_coach`,
     overall and where the true margin exceeds 1 and 3 points;
   - regret = `q_coach`(true best) minus `q_coach`(chosen), in win-probability points: mean and
     90th percentile; the coach-like policy's own regret is the reference line;
   - calibration: learned minus true value, overall, by action, and by how often the coaches
     take that action in that state (under 2%, 2 to 10%, 10 to 50%, 50% or more). Positive errors
     on rarely taken actions are the offline-RL failure the real-data notebook guards against;
   - next-transition NLL on synthetic validation against a frequency baseline;
   - the headline state's three values against the truth;
   - agreement with `q_star` and regret measured on `q_star`.
7. **Exact game on Colab.** The setup stage rebuilds the exact game from public data with the
   repo's own code (nflreadpy download, state table, development and test files, the
   development component bundle, count and coach tables on development seasons, backward
   induction), caches it in the Drive work folder, and checks the saved ground truth against
   it (`verify_ground_truth`: same fit stamp, data fingerprint and solve seed, and the same
   solved value at all 300 states). If they differ, a ground truth with fewer rollouts (300
   states, 20,000 rollouts per action, standard error at most about 0.0035) is recomputed in
   Colab for the game built there, and the results say so. They will differ: a build from a
   fresh download on 19 November 2026 (`test_full_build_from_public_data`, 82 seconds, peak
   memory 6.0 GB, most of it the download) produced a game whose data fingerprint differs from
   `gt_v1`'s, because nflverse revised its data after the cached copy was taken (two 2011 plays
   removed; a few 2018 to 2022 plays now count as kickoffs). The Colab run therefore grades
   against the recomputed ground truth, not `gt_v1`.

## Honest limits

- **The generator is our model.** Identical league-average teams, our clock and kicking models,
  play modes instead of run or pass. The experiment measures whether a world model can learn
  OUR simulator from its games, not whether it can learn real football. A good score here is
  necessary for trusting the real-data planner, not sufficient.
- **The headline state is spliced.** Its game is a synthetic game up to a nearby third down,
  then the headline record (4th and 2 at the own 40, up 4, 4:00, three timeouts each), then the
  game played on from it. The world model therefore sees a history that was not generated in
  one piece.
- **GO is one mode.** GO is graded in the coach's play mode only. `q_star` values GO in the best
  mode, so part of any gap to `q_star` is the mode, not the go-or-kick choice.
- **True masks.** Actions are masked with the exact model's own field-goal range and punt rule,
  which the real-data planner does not have (it uses rules learned from training data).
- **Precision.** The truth has standard errors up to 0.0025. The learned values target 0.01 on
  the go-minus-best-kick difference; states that did not reach it are counted in the results.
- **Training time, not only data, can limit the large sizes.** Within a 90-minute budget the
  64,000-game model may stop on its time budget rather than by early stopping; the results
  record which.

## Time estimates (not measurements)

From `estimate_minutes` with the presets' assumed speeds (T4: 60,000 training records/s and
40,000 rollout steps/s; L4: 120,000 and 80,000; A100: 300,000 and 200,000), which have not been
measured on these GPUs yet. Planning one size: 20 million rollout steps for phase A and 168
million to reach the target on every state (54 steps per rollout on average).

| GPU | phase A per size | every state at target, per size | one epoch at 64,000 games |
|---|---|---|---|
| T4 | 8.5 min | 70 min | 2.8 min |
| L4 | 4.2 min | 35 min | 1.4 min |
| A100 | 1.7 min | 14 min | 0.6 min |

So with the 90-minute default: an A100 should plan all four sizes and refine most states to the
target; an L4 plans every state and refines the closest calls; a T4 plans every state with 512
rollouts per action and refines only some. The preflight cell prints these estimates at the
speeds it measures.
