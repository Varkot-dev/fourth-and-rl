# R1: exact overoptimization of a learned value model (days 1 to 5, and the day 6 grid runner)

Plan, falsifiable claim and pilot critiques: `docs/plans/fellowship_research_directions.md`, sections 4 to 7. Milestone and "done when" criteria: `docs/PLAN.md`, section 4, research milestone R1. Settings and seeds: `configs/overopt.yaml`.

Everything here is exact for the fitted game MDP (`data/mdp`), not real football. No play-by-play was loaded; the synthetic games come from the solved MDP's simulator, so no season (2024, 2025 or 2026) was touched.

## The question in one paragraph

A learned value model (a "proxy") estimates win probability. A policy that leans on the proxy harder and harder (larger beta) gains what the proxy claims, but how much does it gain in truth? In RLHF the truth is a human or a stronger model and is itself noisy. Here the truth is the solved game MDP, so the true gain of every policy can be computed exactly. The pilot found that for proxies trained on 1,000 games, the true gain rose and then fell while the proxy's claimed gain kept rising. Days 1 to 5 rebuild that pipeline with tests and check whether the turnover survives the critic's fixes.

## Concepts

### 1. Exact evaluation of a fixed policy (day 1)

The solver computes the best value with a max over actions:

    V*(s) = max over a of Q*(s, a)

A fixed stochastic policy pi has its own value, with an average in place of the max:

    V_pi(s) = sum over a of pi(a | s) * Q_pi(s, a)
    Q_pi(s, a) = sum over outcomes o of P(o | s, a) * V_pi(next state)

This is the Bellman expectation equation instead of the optimality equation. Because every play uses at least one 5-second tick, V_pi at time t only needs V_pi at earlier times, so one sweep t = 1, ..., 720 evaluates the policy exactly, the same way the solver is exact.

Worked examples:
- If pi always picks the solver's own action, V_pi = V* exactly. The real-game test feeds the solver's stored policy back into the evaluator and gets every one of the 408 million values back bit for bit (the test asserts exact equality at every tick).
- If both teams play the same policy (coach against coach), the game is symmetric and a coin flip decides who receives the opening kickoff, so team A's win probability is exactly 0.5. The evaluator gets 0.5 on the real game.
- A fourth down where the coach goes 40%, kicks 50% and punts 10%, with Q = 0.30, 0.34, 0.20: the coach's value there is 0.4 * 0.30 + 0.5 * 0.34 + 0.1 * 0.20 = 0.31, while the greedy choice is worth 0.34.

Two teams. Team A plays pi and team B plays the coach. The game is zero-sum, so each team's solution holds its own win probability when it has the ball, and a change of possession reads the other team's array (1 minus its value, from its side). This is the same linking `mdp_team.link` uses for two different teams.

Ring buffer. A backup at tick t reads values at most 10 ticks back (the longest play-length histogram; the clock-burning runoff is 9, the punt 2) plus the halftime slice. `RingSolution` keeps only those slices, so one solution needs 0.6 GB at peak instead of the 1.6 GB of a full value array, and many policies fit in one pass.

### 2. Post-decisions (day 2)

After a play, other decisions follow: the offense's timeout, the defense's timeout, and the try after a touchdown. The solver makes them optimally with its own values (a max or min). For a fixed-policy evaluation that is wrong in two ways: (a) it mixes an optimizing component into "coach" play, so "coach versus coach" is not what the synthetic games contain, and (b) a max is not linear, which a forward pass cannot replay. The new flag `post="coach"` uses the simulator's coach rules instead: a chasing offense always stops the clock inside the two-minute warning when it has a timeout, the defense calls one at the measured rates when the offense leads, and the scorer goes for two only in the fourth quarter at leads -5, -2, 1 or 5 after the touchdown. The default `post="model"` keeps the old code path, so the solver's outputs are bit-identical (tested on the toy game and on the real game through the solver-policy test).

Worked example of the try rule: a team down 8 scores a touchdown with 6:00 left. The lead after the touchdown is -2, which is in the list, and 360 seconds is inside the last 900, so the coach goes for two. The same touchdown with 20:00 left takes the kick.

Monte Carlo check. With n independent games from a state, the simulated win rate has standard error sqrt(p (1 - p) / n). At p = 0.5 and n = 62,500 that is sqrt(0.25 / 62,500) = 0.002. A z-score (simulated minus exact, divided by its SE) beyond 2 happens about 5% of the time by chance, so among 24 states about one is expected beyond 2.

### 3. The grid predictor and float32 action values (day 3)

A LightGBM model is a sum of trees, each splitting one feature at a threshold. The states of one tick form a product grid (score by down by distance bin by yard line by timeouts by flag), so a tree can be walked over boxes of grid indices: a split cuts a box in two along one axis, and a leaf adds its value to the whole box at once. This gives the same numbers as `Booster.predict` row by row, in a fraction of the time.

Why float32. The pilot stored the fourth-down action values Q in float16, which has about 3 decimal digits. Two actions with values 0.5123 and 0.5127 round to the same float16 number, so the argmax and the tilt saw false ties. Float32 keeps about 7 digits. On the toy game the float32 storage error in an exact identity (below) stays inside the test's 1e-9 tolerance.

Ties made explicit. When several actions are within `atol` of the best, `greedy` either spreads the mass evenly over them ("uniform") or the diagnostics drop the state ("exclude"). The pilot's argmax silently broke ties toward "go".

### 4. The KL tilt and exact sequence KL (day 4)

At a fourth down with coach distribution P(a) and value estimates Q(a):

    pi_beta(a) = P(a) exp(beta Q(a)) / sum over b of P(b) exp(beta Q(b))

This maximizes sum_a pi(a) Q(a) - KL(pi || P) / beta at that one state: one step of soft policy improvement, applied independently at every fourth down. It is not the KL-regularized optimum of the whole game (day 6 builds that with soft value iteration).

Worked example: P = (0.6, 0.3, 0.1), Q = (0.40, 0.45, 0.41). At beta = 10 the weights are 0.6 e^4.0, 0.3 e^4.5, 0.1 e^4.1, which normalize to about (0.50, 0.41, 0.09): the field goal gains mass. At beta = 1000 almost all mass is on the field goal.

The per-game sequence KL is the expected sum, over the fourth downs team A faces, of KL(pi_beta(. | s) || P(. | s)). It needs to know how often each state is visited. That is the occupancy d_A(s), the probability that the game passes through s with A on offense:

    sequence KL = sum over s of d_A(s) * KL(pi_beta(. | s) || P(. | s))

The forward pass computes d_A exactly. Probability mass starts at the opening kickoff (half on each team) and flows down the clock through the same transitions the backup uses. With post-decisions fixed to the coach rules, the backup at tick t is linear in earlier values (V_t = L_t V_earlier), and the forward pass applies the transposes term by term (mass_earlier += L_t^T mass_t). The mass that reaches t = 0 by score, plus kneel-outs, gives the win probability, which must equal the backward start value; on the toy game the test requires agreement within 1e-12, for four policy pairs and both defense settings.

Performance-difference identity. With Q_coach the exact coach action values (A plays the coach, B plays the coach),

    V_pi(start) - V_coach(start) = sum over s of d_A^pi(s) * sum over a of (pi(a | s) - P(a | s)) Q_coach(s, a)

The same sum with the proxy's Q is the gain the proxy believes in (its "claimed gain"). With the exact Q it must equal the true gain, which is a third independent check of the pipeline.

### 5. Overoptimization and the optimizer's curse

If the proxy's Q has errors e(s, a), the greedy choice picks the action whose error happens to be most positive, so the expected claimed value of the chosen action exceeds its true value (the optimizer's curse). At small beta the policy moves only where the proxy's advantage is large, which is usually real; at large beta it moves wherever the proxy prefers anything at all, including where the preference is noise. True gain can then fall while the claimed gain keeps rising: Goodhart's law in a form that can be measured exactly.

## Decisions and why

1. **Post-decisions fixed to the coach rules everywhere (gold, data, Monte Carlo).** The pilot let each team's timeouts and tries be optimal for its own value inside the exact pass, while its synthetic games and Monte Carlo used the solved game's optimal choices: three different "coaches". Now the synthetic games, the exact evaluator and every Monte Carlo check use the simulator's coach rules (`post="coach"`), which is also what the world-model ground truth (`gt_v1`) uses. It makes the backup linear, which the forward pass needs. The solver's default stays `post="model"`, bit-identical to before.
2. **The oracle is computed inside the pass.** The exact coach action values come from a coach-vs-coach solution stepped alongside the others, so no 1.6 GB value array is written to a nearly full disk.
3. **Fourth-down action values stored as float32,** never float16: 1.2 GB per set in memory, two sets per pass (proxy and exact).
4. **Ties explicit.** The default rule is "uniform" (`configs/overopt.yaml`); the diagnostics report both rules and the number of exact ties.
5. **Exact KL from a forward pass,** not Monte Carlo. Monte Carlo is kept only as a check. The forward pass runs in float64: a float32 trial (a scratch run, not saved) visibly leaked probability mass over a game, which defeats a check that should be exact.
6. **The performance-difference sum with the exact coach values is computed for every cell.** It must equal the true gain; it is a third, independent check (backward value, forward terminal mass, performance-difference sum).
7. **Seeds.** LightGBM without bagging or feature subsampling is deterministic, so `random_state` (7, as in the pilot) changes nothing and the seed of a proxy is the seed of its games: 101, 102 and 103 for the 1k proxies, 116 for the 16k proxies (pilot seeds 101 and 116 kept), 999 for the validation games.
8. **Day 2 states.** 24 pre-snap states drawn from 300 synthetic coach games (seed 20261123), 8 from each phase, keeping only states whose solved value is between 0.15 and 0.85 so that 62,500 games give a standard error near 0.002 (nearly decided states have a standard error near zero and say little). The 6 states with the largest |z| were then rerun with 4 times the games and fresh random numbers.
9. **Threads.** Within a tick the policy pairs are independent, and numpy and scipy release the GIL, so 3 threads step them in parallel (much faster in a scratch timing, not saved; the saved pass timings are in the results). The six forward passes run 3 at a time. Still one job at a time on the machine.
10. **Library checks** (CLAUDE.md asks for current documentation): LightGBM 4.7.0 is installed; the `dump_model` fields the grid predictor reads (`tree_info`, `tree_structure`, `split_feature`, `threshold`, `decision_type`, `missing_type`, `left_child`, `right_child`, `leaf_value`) were checked against the installed package by the tests, which compare with `Booster.predict` directly. The predictor refuses categorical and zero-as-missing splits rather than guess.

## Results

All numbers below are read from files under `results/research/` (named in each paragraph). Gains are in points of win probability (100 times the probability), team A against a coach team B, from the opening kickoff with a coin flip.

### Day 1: the evaluator (`overopt_day1_timing.json`)

| Pass (whole game, 720 ticks) | Seconds | Peak memory (GB) |
|---|---|---|
| One solution (both teams play the coach), coach post-decisions | 62.9 | 0.60 |
| One solution, model post-decisions | 63.4 | 0.58 |
| Two linked solutions (A and B), coach post-decisions | 133.0 | 0.88 |
| Two linked solutions, model post-decisions | 124.3 | 0.88 |
| Forward occupancy pass, two teams, float64 | 179.6 | 0.97 |

Coach against coach is 0.5 at kickoff in every run; the forward pass gives 0.5000000000000012 with total mass 1.0000000000000024. Tests: the solver's own policy reproduces the solver bit for bit (real game and toy game), and a separate recursive two-team evaluator (`TwoTeamBrute` in `tests/test_policy_eval.py`) agrees within 1e-10 on 1,200 random toy states, for both defense-timeout settings.

### Day 2: coach post-decisions against the simulator (`overopt_day2_post_mc.json`)

24 states (8 per phase), 62,500 games each, mean standard error 0.00168. 23 of 24 within 2 standard errors; the largest |z| is 2.40 (a 4-point lead at 16:05 left, third down). The sum of z squared is 34.9 on 24 states. The 6 states with the largest |z| rerun with 250,000 games each and fresh random numbers all land within 1.04 standard errors, and the worst state moves from z = -2.40 to -0.61, so the excess was chance. The criterion asked for every state within 2 standard errors; one state at 2.40 among 24 is what chance gives about a third of the time, and the follow-up shows no systematic error.

### Day 3: the grid predictor (`overopt_day3_grid.json`)

On every state of every tick (404,585,280 states, ticks 1 to 720) the grid predictor equals `Booster.predict` for the `1k_large` proxy to 2.2e-16 in float64, and to 3.0e-8 after storing as float32. It took 60.4 s against 361.1 s for `Booster.predict` on 4 threads. Only one proxy was checked on the real game (the toy test covers a second model on every tick). Exact float32 ties at the top of the exact coach action values: 28 of 10,067 held-out fourth downs (the pilot's float16 had 2.4%, per the plan document); 782 have a top-two gap below 0.001, which is a real property of the game, not storage.

### Day 4: exact KL against Monte Carlo (`overopt_day5.json`, `monte_carlo_checks` and `extra_monte_carlo`)

At the oracle and `1k_large` cells, 20,000 games per beta (12 runs): every exact KL is within 1.47 standard errors of Monte Carlo, and every exact win probability within 1.60. The large runs at `1k_large` (200,000 games, beta 100 and 10,000, one shared seed): KL z = -0.76 and -2.57, win-probability z = -2.12 and -1.61. Because those two disagreements had the same sign and shared random numbers, an independent run with 1,000,000 games (seed 2029) was added: win-probability z = 1.26 (beta 0, coach against coach), -0.45 (beta 100) and 0.24 (beta 10,000); KL z = -0.20 and -0.58. Per-decision KL at large beta is heavy-tailed (a decision the coach almost never makes costs many nats), which makes a 200,000-game standard error less reliable than it looks.

Internal exactness over all 54 cell-beta pairs: forward terminal mass minus backward value at most 1.03e-7, performance-difference sum minus backward value at most 1.04e-7. Both are float32 rounding of the backward values; on the toy game in float64 the tests hold to 1e-12.

### Day 5: the curves (`overopt_day5.json`, figure `assets/research/overopt_day5.png`)

True gain over the coach at beta = 10 / 30 / 100 / 300 / 1,000 / 10,000:

| Cell | Value RMSE vs exact coach value | Greedy disagreement | True gain | Exact sqrt(KL) at beta 10,000 | Claimed gain at beta 10,000 |
|---|---|---|---|---|---|
| oracle | 0 | 0 | 0.58 / 1.37 / 2.78 / 4.01 / 4.57 / 4.68 | 2.83 | 4.68 |
| 16k_large | 0.018 | 16.4% | 0.51 / 1.24 / 2.55 / 3.54 / 3.85 / 3.85 | 2.91 | 5.27 |
| 16k_small | 0.024 | 19.5% | 0.42 / 1.08 / 2.32 / 3.33 / 3.65 / 3.64 | 2.97 | 4.74 |
| 1k_small (seed 101) | 0.035 | 24.4% | 0.41 / 1.00 / 2.01 / 2.70 / 2.84 / 2.76 | 2.92 | 5.50 |
| 1k_small_b (102) | 0.035 | 25.8% | 0.35 / 0.89 / 1.89 / 2.63 / 2.80 / 2.77 | 2.86 | 4.95 |
| 1k_small_c (103) | 0.038 | 23.4% | 0.46 / 1.14 / 2.25 / 2.89 / 2.95 / 2.84 | 3.09 | 6.72 |
| 1k_large (seed 101) | 0.051 | 30.5% | 0.49 / 1.11 / 1.89 / 1.99 / 1.74 / 1.53 | 3.19 | 9.30 |
| 1k_large_b (102) | 0.052 | 31.6% | 0.48 / 1.07 / 1.78 / 1.91 / 1.68 / 1.47 | 3.16 | 9.20 |
| 1k_large_c (103) | 0.052 | 29.1% | 0.49 / 1.13 / 1.91 / 2.02 / 1.76 / 1.55 | 3.36 | 10.85 |

RMSE is on 100,000 held-out synthetic states (games seed 999); disagreement is the uniform-tie rule at 10,067 held-out unforced fourth downs. The exact best fourth-down response to the coach is worth 4.69 points; the oracle tilt reaches 4.68.

Seed summary (3 data seeds each):
- **1k_large:** every seed peaks at beta 300 (1.99, 1.91, 2.02) and ends at 1.53, 1.47, 1.55. Peak-to-final drop 0.457 points, seed SD 0.015. With the peak fixed in advance at the pilot's beta 100: drop 0.346, SD 0.030. Both exceed 2 SD by a wide margin.
- **1k_small:** every seed peaks at beta 1,000 (2.84, 2.80, 2.95) and ends at 2.76, 2.77, 2.84. Drop 0.070, SD 0.041: below 2 SD. At the pilot's peak beta 300 the "drop" is -0.055 (the gain is still rising there). This is a plateau, not a turnover.
- **16k proxies:** plateaus (16k_large 3.85 to 3.85; 16k_small 3.65 to 3.64), one seed each.
- **Claimed against true:** every proxy claims more than it delivers, and the gap grows with beta and with proxy error: at beta 10,000 the 1k 64-leaf proxies claim 9.2 to 10.9 points and deliver 1.5.

Against the pilot (numbers quoted from `docs/plans/fellowship_research_directions.md`, section 4, not recomputed): the pilot's 1k_large peaked at 1.71 (beta 100) and fell to 0.94; its 1k_small fell from 2.64 to 2.24. With the fixes (coach post-decisions everywhere, float32, exact KL) 1k_large peaks higher and falls less, and the 1k_small dip is gone. Which fix moved which number was not isolated: no ablation was run.

## Go/no-go (day 5 gate)

**Go, on a narrower claim.** The true-gain turnover survives the fixes for the 1k-game 64-leaf proxies: a drop of 0.46 points from peak to beta 10,000 (0.35 from the pilot's peak beta), against a seed SD of 0.015 (0.030), across 3 data seeds. It does not survive for the 1k-game 8-leaf proxies (drop 0.07, SD 0.04: a plateau), and the 16k proxies plateau. So in this testbed overoptimization appears only for the highest-error proxy tried (value RMSE about 0.052), and it gives back about a quarter of its peak, not the pilot's 45%.

Caveats that bound the claim:
- 3 seeds, not the 5 the falsifiable claim asks for; an SD from 3 values is itself uncertain.
- The proxy seed is the data seed (LightGBM here is deterministic), and the validation set is one seed.
- Ordering by proxy error holds within these cells (peak gain falls as RMSE rises: oracle, 16k_large, 16k_small, 1k_small, 1k_large), but that is 5 points, not a test.
- The tilt is one step of soft improvement per fourth down, not the KL-regularized optimum of the game.
- Exact for the fitted game MDP, not real football.

What this means for days 6 to 10: the turnover regime is real but sits at high proxy error, so the seeded grid (day 6) should include smaller data sets (for example 250 and 500 games) and higher-capacity proxies, where the falsifiable claim can be tested with room on both sides; and the 1k_small result should be reported as a plateau.

## For the Colab grid (day 6)

Per cell on this CPU (3 threads), from `timing` in each cell file: backward pass 298 to 451 s (14 or 15 solutions), forward passes 383 to 468 s (six betas, 3 at a time), whole pass 718 to 947 s, peak memory 7.06 to 7.64 GB. Synthetic games take seconds to generate (not timed in a saved file) and proxies train in under 7 s (`fit_seconds` in each cell's proxy record). Files needed: `src/`, `configs/overopt.yaml`, `scripts/research/overopt_curves.py`, and the solved game `data/mdp` (about 2.0 GB, gitignored: upload it, or re-solve with `scripts/solve_mdp.py`). The synthetic games and proxies in `data/research/overopt` are regenerated from the config's seeds. A high-RAM runtime is needed (at least 8 GB free per process); two processes at once need about 16 GB.

## Day 6, run A: the seeded grid (runner and notebook built, grid not yet run)

Code: `src/fourth_and_rl/overopt_grid.py` (plan, parts, resume, parallelism, scheduler, collect), `src/fourth_and_rl/game_setup.py` (the solved game on a fresh machine), `scripts/research/overopt_grid.py` (one cell, the local runner, collect) and `notebooks/overopt_grid_colab.ipynb`, generated by `notebooks/build_overopt_grid_notebook.py`. Settings: `configs/overopt.yaml`, section `grid`. Tests: `tests/test_overopt_grid.py`. Soft value iteration, the other half of day 6, is not started.

### The grid

A cell is one proxy: n synthetic coach-vs-coach games drawn with one seed, a LightGBM proxy of one capacity fitted on them, and one exact pass over all six betas. Its key is `n<games>_<capacity>_s<seed>`, for example `n250_large_s101`. Sizes 250, 500, 1,000, 4,000, 16,000 and 64,000 games; capacities small (8 leaves) and large (64 leaves), the day 5 settings; seeds 101 to 105; plus the oracle cell. That is 61 cells.

Priority order (`plan_cells`): block 1 is the turnover region (250, 500 and 1,000 games), then the oracle, then block 2 (4,000 and 16,000), then block 3 (64,000). Inside a block the loop is seed, then size, then capacity, so the first 6 cells are seed 101 at every size and capacity of block 1, the next 6 are seed 102, and so on. Why: if a session ends early, what it leaves is complete replicates across the region that matters, not five seeds of one size and nothing of the others. Worked example: after 18 cells, every (size, capacity) of block 1 has seeds 101, 102 and 103, enough for a first SD; after 30, all five.

Same games as day 5: `coach_games(n, seed)` is deterministic, so `n1000_large_s101` trains on exactly day 5's `1k_large` games (seed 101), and s102 and s103 match `1k_large_b` and `_c`. Collect compares them (`day5_reproduction`): on the same solved game the true gains must agree, a free check of the new runner. The validation games are day 5's (4,000 games, seed 999; the regenerated file was checked equal to day 5's).

### The test of a turnover

For each cell: the peak beta (the beta of the largest true gain), the peak gain, the final gain (at beta 10,000) and the drop, peak minus final. For each (size, capacity), across seeds: the mean drop and its SD (ddof 1), and whether the mean exceeds 2 SD, the falsifiable claim's first criterion. Worked example from day 5: the 1k 64-leaf drops were 0.457, 0.443 and 0.473 points, mean 0.457, SD 0.015, and 0.457 > 2 x 0.015 = 0.030, so it passes; the 1k 8-leaf drops were 0.080, 0.024 and 0.105, mean 0.070, SD 0.041, and 0.070 < 0.082, so it does not. Note that the drop from the observed peak is never negative, so with noise alone its mean is positive; the 2 SD rule guards against calling that a turnover, but it is a heuristic, not a calibrated test, and with 5 seeds the SD is itself uncertain.

The figure (`assets/research/overopt_grid.png` from collect) has one panel per capacity, one line per size: at each beta, the mean over seeds of sqrt(KL) on the x axis and of the true gain on the y axis, with a band of plus or minus one seed SD of the gain, and the oracle curve as the reference. Labelled exact for the fitted game MDP, not real football.

### Decisions and why

1. **One process per cell.** A pass peaks at about 7.6 GB and memory is returned to the system only when a process ends, so each cell runs as `overopt_grid.py cell <key>` in its own process. A crash or an out-of-memory kill loses one cell, not the session.
2. **How many at once (`choose_parallel`).** n = max(1, min(floor((available RAM - 2 GB) / 8 GB), cores // 3)), threads per cell = min(3, cores // n). 8 GB is the measured peak (7.64 GB) rounded up, 3 threads is what was measured. Worked examples: this machine (4 cores, 15.1 GB available) gives floor(13.1 / 8) = 1 and 4 // 3 = 1, so one cell with 3 threads; a Colab high-RAM CPU runtime with 8 vCPUs and about 49 GB available gives min(5, 2) = 2 cells with 3 threads; Colab's standard CPU runtime (2 cores, 12.7 GB) gives 1 cell with 2 threads. Below 12 GB of RAM the notebook warns. A cell killed by a signal (usually out of memory) is retried once with one fewer cell in parallel.
3. **Resume by settings key.** A cell counts as done only if its result file exists, parses, has rows, and carries the same settings key: a hash of the betas, the post-decision and tie rules, the validation games, the proxy parameters, the size, the seed and the solved game's key. A smoke run with two betas therefore never passes for a real cell. Threads are left out because they do not change the numbers (the pairs within a tick are independent, and the proxy always trains with `proxy_threads`). Files are written through a temporary file and one rename, so a disconnect never leaves half a file on Drive.
4. **Deadline guard.** A cell starts only if now plus the estimate is before the deadline (default 5 hours from the notebook's start, less 5 minutes for collect). The estimate is 1,000 s from the config until cells finish, then 1.2 times the longest finished cell. Worked example: with 30 minutes left and a longest cell of 15 minutes, the estimate is 18 minutes, so one more round starts; with 15 minutes left, none does.
5. **The game key.** `game_key` hashes the solved game's fit stamp, tables key (with the data fingerprint) and solve seed, without fit timestamps. The local `data/mdp` has key `ea0c61dd52ab0fe7`, recorded in the config as `reference_game_key`. A rebuild from a fresh nflverse download is expected to differ (the world-model notes record that nflverse revised 2011 and 2018 to 2022 plays after our copy was taken), so collect never pools cells from different games: it keeps the most common game key and lists the others as excluded. To merge Colab and local cells, upload the local `data/mdp` to the Drive cache folder before the first session.
6. **Parts.** Part K/M is every M-th cell of the priority-ordered plan from the K-th. Each part keeps the priority order, the parts are disjoint, and together they are the whole plan (all tested). Colab with two cells at a time runs about twice as fast as this machine, so the suggested split is `PARTS = ["1/3", "2/3"]` in the notebook and `--part 3/3` here.
7. **Merging.** A key found in two folders is a duplicate if settings and true gains agree to 1e-9, otherwise a conflict, reported, with the first folder kept.
8. **The game on Colab.** `game_setup.ensure_game` uses a given folder, the runtime's copy, the Drive cache, or rebuilds from public data with the same steps as the world-model ground-truth notebook (download 2010 to 2025 with nflreadpy, development and test files, the development component bundle with the 2024 and 2025 kickoff tables as the one logged exception, count and coach tables on development seasons, backward induction). It refuses holdout seasons before downloading. The Drive cache is skipped when Drive lacks space, as in the ground-truth notebook. The world-model code was left unchanged, so the build steps exist twice; the duplication is about 60 lines.

### How to run and split

Here, one slice (writes `results/research/overopt_grid/cells/`, logs in `data/research/overopt/grid/logs`):

    .venv/bin/python scripts/research/overopt_grid.py plan --part 3/3
    .venv/bin/python scripts/research/overopt_grid.py run --part 3/3 --deadline-hours 5

On Colab: open `notebooks/overopt_grid_colab.ipynb`, Runtime > Change runtime type > CPU with High-RAM, set `PARTS = ["1/3", "2/3"]`, Run all; run all again in later sessions. Bring back the zip, unzip its `cells/` into `results/research/overopt_grid/cells/`, then:

    .venv/bin/python scripts/research/overopt_grid.py collect

which writes `results/research/overopt_grid.json` and `assets/research/overopt_grid.png`.

### Time estimates (not measurements of the grid)

Per cell on this machine: 718 to 947 s (day 5, 3 threads, one at a time); data and proxy for 250 to 1,000 games take seconds to about a minute (6 s in a local smoke run that a container restart interrupted, 76 s in the dry run below, which shared the machine with the test suite). Assumed for a Colab high-RAM CPU runtime: 8 vCPUs (hyperthreads, so about 4 physical cores) and about 51 GB, so 2 cells at a time, each 14 to 19 minutes (this machine's range, up to 1.2 times slower because two cells share 4 physical cores). Then block 1 plus the oracle (31 cells) takes about 3.6 to 4.9 hours, block 2 (20 cells) 2.3 to 3.2 hours, and block 3 (10 cells) 1.2 to 1.6 hours plus a few minutes of data and fitting for 64,000 games: about 7 to 10 hours in all, two 5-hour sessions. Split 2:1 with this machine, the Colab parts (41 cells) take about 4.8 to 6.5 hours and the local part (20 cells, one at a time) about 4 to 5.3 hours. These are estimates; the notebook prints its own from the cells it finishes.

### Dry run of the notebook (a smoke test, not a result)

`notebooks/overopt_grid_colab.ipynb` was executed end to end with nbclient 0.11.0 in a fresh virtual environment, with the repository cloned from the local checkout (the notebook's own clone and `pip install -e` cells), the local `data/mdp` as `GAME_DIR`, Drive unavailable (the notebook fell back to a local work folder), and a tiny grid: one cell, `n250_large_s101`, with betas 10 and 10,000 only. All 21 cells ran with 0 errors in 695 s (11.6 minutes, including the install). The cell itself took 10.2 minutes: 76 s of data and proxy, 221 s backward (6 solutions), 58 s grid prediction, 248 s forward (2 betas), peak memory 5.0 GB; collect wrote the JSON, the figure and a 0.15 MB zip. Its numbers (true gain 0.48 points at beta 10 and -0.54 at beta 10,000, one seed) are a check that the pipeline runs, not a grid result, and are not saved under `results/`.
