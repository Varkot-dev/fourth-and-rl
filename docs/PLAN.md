# fourth-and-rl: plan

NFL fourth-down and play-calling decisions as a decision process: statistical models, exact dynamic programming and reinforcement learning.

## 1. The question

On fourth down a coach chooses: go for it, kick a field goal, or punt. Analysts have argued for years that coaches are too conservative. This project:

- (a) rebuilds the standard analysis from raw play-by-play data,
- (b) adds honest uncertainty to it,
- (c) uses the problem as a real-data setting to learn reinforcement learning, because it is a true sequential decision problem with a clear reward (winning).

It also works as a research topic. The project is built around the work of Abraham Wyner's group at Penn (Wharton Statistics): how much win-probability models can be trusted, what a "blown lead" really tells you, and what that means for fourth-down recommendations. The RL part then asks the follow-up question: what should a decision-maker do when the win-probability estimates it relies on are that uncertain?

### 1.1 Burning questions (the public framing)

The project is presented as answers to concrete questions fans and coaches argue about. Each question gets the same treatment: what the standard one-step analysis says, how confident that answer is under the game-level bootstrap, what the exact full-game solution says, and what coaches actually do in that spot. Every answer is produced by a script and saved under `results/questions/`, with one figure each under `assets/questions/`.

1. **Protecting a late lead (the first question to answer).** Up 4 with about 4:00 left, fourth down in your own half. Go for it or punt? Converting lets you burn the clock. Failing gives the opponent a short field, but a quick score also hands you the ball back with time to answer. The answer should be a map over yards to go, field position, time left and the opponent's timeouts, not a single number.
2. **Why do offenses run when passing gains more?** Passes gain more per play on average, yet teams run about 40% of the time. Measure the gap by down, distance and situation, then test the game-theory explanation (what matters is the value of the next run versus the next pass, and defenses adjust) with the run-or-pass version of the decision model.
3. onward: **chosen by the owner**, from the popular explainer videos he has in mind (the kind that take one question fans argue about and answer it with data). He supplies the links or titles; each becomes one question with its own script and figure.
4. Was that famous collapse really historic, or is a 95% peak for a loser ordinary? (Answered by the blown-lead analysis.)

Kept as a "maybe later" list, because they need the full-game solution and extra actions: going for two when down 8, the intentional safety, letting the opponent score, and overtime strategy under the 2025 rules (see `docs/research/landscape.md`).

## 2. What exists today (verify before building)

**Data: nflverse.** Free play-by-play for every game since 1999. Python access is the `nflreadpy` package (`pip install nflreadpy`, then `nfl.load_pbp(...)`). The older `nfl_data_py` package is deprecated in favor of it. Each play already carries the nflfastR expected-points and win-probability estimates, which make a useful baseline.

**nfl4th** (R package from the nflverse group, by Ben Baldwin). The well-known model behind the "4th down decision bot". It estimates win probability for going, kicking and punting and recommends the highest. Use its outputs as the sanity check for this project's decision engine.

**The anchor papers (Wyner group, Penn). Read these first and build the project around them:**
- Pipping-Gamon and Wyner, "The Blown Lead Paradox: Conditional Laws for the Running Maximum of Binary Doob Martingales" (arXiv 2601.18774, January 2026), plus the shorter Wharton research note "A Paradox of Blown Leads: Rethinking Win Probability in Football" (July 2025, on the Wharton Sports Analytics and Business Initiative site). Core idea: commentators cite the highest win probability reached by the team that lost as proof of a historic collapse or a bad model. But even a perfectly calibrated model produces high peaks for losers often, so a single peak means little without the reference distribution of that peak. The paper derives that distribution and proposes it as a calibration check for live forecasts.
- Brill, Yurko and Wyner, "Exploring the Difficulty of Estimating Win Probability: A Simulation Study" (arXiv 2406.16171; published in the Journal of Quantitative Analysis in Sports, 2025). Every play in a game shares the same win/loss label, so the effective sample size is closer to the number of games than the number of plays, and honest confidence intervals are wide.
- Brill, Yurko and Wyner, "Analytics, have some humility: a statistical view of fourth-down decision making" (arXiv 2311.03490). Carries that uncertainty into the fourth-down decision.

**Supporting papers:**
1. Romer (2006), "Do Firms Maximize? Evidence from Professional Football". The original conservative-coaches result.
2. Sandholtz and coauthors, "Learning Risk Preferences in Markov Decision Processes: an Application to the Fourth Down Decision in the National Football League" (arXiv 2309.00756). Models the game as an MDP estimated from 2014 to 2022 play-by-play and infers how risk-averse coaches behave.

**Tracking data.** The NFL Big Data Bowl on Kaggle releases player-tracking data each year. Out of scope for the core; a good stretch for visuals.

## 3. Design

### 3.1 The MDP
- **State:** down, yards to go, yard line, seconds remaining, score differential, timeouts for each side. Start smaller with (down, distance, yard line) for a single drive, then add clock and score.
- **Actions:** on fourth down: go, field goal, punt. Extended version: run or pass on every down.
- **Transitions:** learned from data. Conversion probability by distance and field position, yards-gained distributions for run and pass, field-goal make probability by distance, net punt distance, turnover rates.
- **Reward:** 1 for a win and 0 for a loss at the end of the game. For the single-drive version: points scored minus the opponent's expected points from the resulting field position.

### 3.2 Three ways to get a policy, compared on one problem
1. **One-step lookahead with a win-probability model.** The standard analytics approach. Fit a calibrated win-probability model, compute each action's expected win probability, pick the best.
2. **Exact dynamic programming.** Discretize the state, build the transition model, run value iteration. This gives the true optimum of the model.
3. **Reinforcement learning in a simulator.** Wrap the fitted transition model as a Gymnasium environment. Write tabular Q-learning and REINFORCE by hand, then train DQN and PPO with a library.

Because method 2 gives the exact answer, you can measure how close each learned policy gets and how many simulated games it needs. That comparison is the main learning value: sample efficiency, exploration and function-approximation error, all measured against known ground truth.

### 3.3 Offline RL (the research-flavored part)
Real data comes from coaches' choices, not from exploration. Almost nobody goes for it on fourth and 15 from their own 10, so the data says very little there. Steps:
- Run fitted Q iteration directly on logged plays.
- Show where it gives overconfident values for rarely taken actions (distribution shift).
- Apply a conservative method such as CQL (the d3rlpy library has one) and show the difference.

### 3.4 Uncertainty
Bootstrap over games (not plays, because plays within a game are correlated), refit the models, and report for each situation the fraction of refits in which each action is best. The decision chart gets a third category: "too close to call".

### 3.5 Blown leads as a calibration test (the centerpiece)
The math to understand before coding, with intuition:

- A calibrated win probability is a **martingale**: its expected future value equals its current value. If it did not, the model would be predictably wrong and you could improve it.
- For a martingale that starts at p0 and moves continuously until it ends at 0 or 1, the chance it ever reaches a level x (with x above p0) is **p0 / x**. Intuition: of all the paths that reach x, a fraction x go on to win, and the total winning fraction must equal p0.
- Among paths that reach x, a fraction (1 - x) go on to lose. So the chance that a team reaches x and then loses is (p0 / x) * (1 - x), and given that the team lost:

**P(loser's peak win probability >= x) = p0 * (1 - x) / ( x * (1 - p0) )**

- For an even game (p0 = 0.5) this is **(1 - x) / x**. Worked examples: x = 0.67 gives about 0.5, so half of all losing teams were at least 67% favorites at some point. x = 0.9 gives about 0.11. x = 0.99 gives about 0.01, so in a 272-game season you expect roughly 2 or 3 losers who peaked at 99% from even games alone.

This derivation is mine from first principles for the continuous case. Check it against the paper, which also handles the corrections for real games, where win probability jumps play to play.

What to build:
1. For every game, extract the losing team's peak win probability under each model (ours, and the nflfastR column).
2. Compare the empirical distribution of those peaks with the theoretical one given each game's starting probability. A calibrated model matches; an overconfident model shows too many extreme peaks.
3. Rank the "biggest collapses" two ways: by raw peak, and by how surprising the peak is under the reference distribution. Show how the lists differ.
4. Check the continuous-path assumption: measure the jump sizes in real win-probability paths and show how discreteness shifts the curve, by simulation.
5. Reproduce the group's sample-size point with a simulation: fit a win-probability model on simulated games where the truth is known and show how wide honest intervals must be.

### 3.6 Coach evaluation
For each real fourth-down decision, compute the win probability given up relative to the model's choice, counting only confident recommendations. Aggregate by team and season, and show the league trend over time.

### 3.7 Pitfalls to design against
- **Leakage:** split train and test by season, never by play.
- **Calibration:** matters more than accuracy for win probability. Show reliability curves.
- **Selection bias:** teams that go for it are not a random sample. Include team-strength features (the pregame spread is in the data) and state what remains uncontrolled.
- **Rule changes across eras:** restrict the seasons used or model the era.
- **Simulator validity:** simulated scores and drive outcomes should match the real distributions. Test this before trusting any RL result.

## 4. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Data pipeline: load, clean, build the state table, cache to Parquet, data dictionary | Row counts and basic league rates match published figures |
| M1 | Component models: win probability (gradient boosting, calibrated), conversion, field goal, punt | Season-based test split, reliability plots, comparison with the win-probability column already in the data |
| M2 | Decision engine and the classic decision chart (yard line by distance) | Recommendations on several well-known plays agree with nfl4th, or the differences are explained |
| M3 | Uncertainty by game-level bootstrap | Chart shows confident vs too-close-to-call regions |
| M3b | Blown-lead analysis: losers' peak win probability vs the theoretical reference, for our model and the nflfastR one; discreteness simulation; effective-sample-size simulation | Calibration figure and the two collapse rankings are generated by scripts |
| M4 | MDP and value iteration: single drive, then full game | Every disagreement between the DP policy and the one-step engine is explained |
| M5 | RL agents: Gymnasium environment, hand-written tabular Q-learning and REINFORCE, then DQN and PPO | Learning curves plotted against the DP optimum |
| M6 | Offline RL on logged decisions | Distribution-shift demonstration with and without the conservative method |
| M7 | Coach evaluation, interactive app, write-up | README complete, repo ready to make public |

RL tie-in for M5 and M6: train and evaluate agents under model uncertainty. Give the agent a win-probability estimate drawn from the bootstrap ensemble instead of the truth, and compare a policy that maximizes the point estimate with one that acts only when the ensemble agrees.

Stretch: play calling on all downs; inferring coach risk preferences following the Sandholtz paper; opponent-specific models; tracking-data visuals.

## 4.1 Machine-learning methods map

Each method is used where it answers a real question in the project, and each is compared against a simpler baseline so its value is measured, not assumed.

| Method | Where | What it is compared against |
|---|---|---|
| Gradient boosting (LightGBM) with monotone constraints | M1 win probability | nflfastR `wp` and `vegas_wp` |
| Probability calibration (isotonic or Platt) | M1 | the uncalibrated model |
| Logistic regression with splines | M1 conversion, field goal, punt | raw empirical rates |
| Neural win-probability model (PyTorch MLP) | M1 extension | the gradient-boosted model, on the same test seasons |
| Deep ensemble (several PyTorch networks with different seeds) | M3 extension | the game-level bootstrap, as a second way to measure uncertainty |
| Game-level bootstrap | M3 | naive play-level intervals |
| Exact dynamic programming (backward induction) | M4 | the one-step engine |
| Tabular Q-learning and REINFORCE, written by hand | M5 | the DP optimum |
| DQN and PPO (PyTorch, Stable-Baselines3) | M5 | the DP optimum and the tabular agents |
| Fitted Q iteration and conservative Q-learning (d3rlpy) | M6 | each other, on logged coach decisions |
| Behavior cloning of coaches (a classifier predicting go, kick or punt) | M6 and M7 | the optimal policy, to locate where coaches differ most |

Added on 2026-11-17 at the owner's request: the neural win-probability model, the deep ensemble and behavior cloning. Considered and left as stretch because they need data or scope the core does not have: sequence models over play history (the state is already Markov by construction, so the gain should be small and is worth one honest test at most), and anything on player-tracking data.

## 5. Visuals

This should be the most visual project of the set.

- **Hero:** the decision chart drawn on a football field, colored by best action, with opacity showing confidence.
- **Blown-lead figure:** histogram of losers' peak win probability with the theoretical curve overlaid, plus a gallery of win-probability paths for the most and least surprising collapses.
- **Interactive app** (Streamlit, or a static Plotly page on GitHub Pages): set the game situation with sliders, see each action's win probability with intervals and the recommendation.
- **Animation:** the value function across the field as the clock runs down in a close game.
- **RL learning curves** converging toward the DP optimum line, one per algorithm.
- **Team aggressiveness ranking** and the league trend by season.
- **Offline RL heatmap:** how often each state-action pair appears in the data, next to the naive Q-values, showing overestimation where data is thin.
- **Case study:** one famous real decision with the numbers at that moment.

## 6. Stack
nflreadpy, polars or pandas, scikit-learn, XGBoost or LightGBM, NumPy for dynamic programming, Gymnasium, PyTorch, Stable-Baselines3, d3rlpy, Plotly, Streamlit, pytest.

## 7. Questions you should be able to answer
1. Define the MDP: state, action, transition, reward. Why is the problem Markov only if the state includes clock and score?
2. Write the Bellman optimality equation for this problem and explain value iteration.
3. Why does one-step lookahead with a win-probability model differ from the full DP solution?
4. Q-learning vs policy gradient: what does each estimate, and which is on-policy?
5. Why does naive RL on logged coach decisions fail, and what does a conservative method change?
6. Why bootstrap over games and not plays?
7. What is calibration, and why does it matter more than accuracy here?
8. What selection bias hides in "teams that go for it convert X% of the time"?
9. How did you validate the simulator?
10. If coaches are not maximizing win probability, what might they be maximizing?
11. What is a martingale, and why must a calibrated win probability be one?
12. Derive the chance that a team reaches win probability x and still loses. Why is a 90% peak for a loser unremarkable?
13. Why is the effective sample size for a win-probability model closer to games than plays?
14. How does the blown-lead distribution act as a calibration test, and what would an overconfident model look like under it?

## Decision log

- **2026-11-17. Working mode: build first, teach at the end.** The owner asked for the milestones to be built autonomously, with the teaching pass and the quiz moved to the end. `CLAUDE.md` was updated to match. To keep the teaching pass honest, each milestone records its concepts, decisions and results in `docs/NOTES.md` as it is built.
- **2026-11-17. Blown-lead analysis (M3b) moves to right after M1.** It needs only the data pipeline, the nflfastR win-probability column and our own win-probability model, and it is the most original result, so it should exist early. Build order is now M0, M1, M3b, M2, M3, M4, M5, M6, M7.
- **2026-11-17. Full-game MDP discretization.** Yard line exact (1 to 99). Distance 1 to 10 exact, then bins 11 to 15 and 16+. Score differential clipped to -21..+21. Clock measured in plays remaining, not seconds. Timeouts left out of the first version. Because the clock only decreases, the model is solved by one backward pass (finite-horizon backward induction) instead of iterating to convergence.
- **2026-11-17. The opponent in the full-game MDP.** The plan did not say how the other team behaves. Decision: both teams follow the same policy and the game is zero-sum, so the value of a state for the team without the ball is 1 minus the value for the team with it. Backward induction then solves both sides at once.
- **2026-11-17. nfl4th comparison (M2).** nfl4th is an R package. It is used once, from a short R script, to produce a reference table of well-known fourth downs saved under `results/` and labelled as nfl4th output. The project itself stays in Python.
- **2026-11-17. Burning questions become the framing.** The owner wants the project to answer the questions people actually argue about, starting with: up 4, about 4:00 left, fourth down in your own half, go or punt? Section 1.1 lists them. Each is answered by the one-step engine, the bootstrap and the full-game DP, with coach behavior alongside.
- **2026-11-17. Timeouts are in the full-game MDP from the start.** This reverses part of the discretization entry above. The first burning question turns on whether a conversion lets the leading team run out the clock, which depends on the opponent's timeouts, so the end-game model tracks the trailing team's timeouts and uses real seconds (not plays remaining) in the final minutes.
- **2026-11-17. Citations checked against arXiv (details in `docs/PAPERS.md`).** Three of the four arXiv papers were read in full and their titles, authors and IDs match the plan. Corrections: (1) arXiv lists 2406.16171 as "Accepted to JQAS" with no year or volume, so the plan's "published 2025" is unconfirmed. (2) The Sandholtz MDP has no clock, score or timeouts and values the decision by expected points under league-average future play, so it is a shape check for our single-drive stage only, not a full-game reference. (3) The humility paper resamples games and then drives within games, with B = 101, and classifies a decision as confident (83% or more of replicates agree), lean (67% to 83%) or uncertain (below 67%). M3 adopts that rule, with thresholds in config.
- **2026-11-17. nfl4th reference is broader than planned.** Instead of a handful of famous plays, `scripts/r/nfl4th_reference.R` (nfl4th 1.0.7, R 4.3.3) saves nfl4th's numbers for all 8,465 real fourth downs of 2024 and 2025 and for a 480-row grid of late-lead scenarios under `results/reference/`. M2 compares against all of them.
- **2026-11-17. Environment.** `nflreadpy` 0.1.5 works as the plan describes (`load_pbp(season)` returns a polars DataFrame). Python 3.12 is used for the virtual environment for library compatibility.
- **2026-11-17. Novelty stated honestly (from `docs/research/landscape.md`).** (1) The development version of nfl4th on GitHub already has a game-level bootstrap, timeouts in win probability and kneel rules; our reference uses the CRAN release 1.0.7, which does not. So M3 is a rebuild, not a new result. (2) arXiv 2601.18774 (latest version retitled "A Pathwise Calibration Benchmark for Win Probability Forecasts") already runs the blown-lead test on ESPN's NFL win probability for 2018 to 2024 and reports no departure. Our M3b is a replication on open models (nflfastR and ours) plus the discreteness and sample-size simulations. (3) Biro and Walker (arXiv 2103.06939) built a final-five-minutes MDP with score and time but without timeouts or kneels and with an approximate solve. What appears not to exist in public: an exact full-game solve with clock, score, timeouts and kneels set beside the one-step answer, with the bootstrap carried through it, and end-game actions such as the intentional safety and conceding a score. That is where the project should spend its effort. The search could not reach GitHub search or several sites, so "nobody has done this" is unconfirmed and the README must not claim it.
- **2026-11-17. Burning questions are the owner's choice.** A first revision replaced the list with end-game edge cases (intentional safety and similar). The owner rejected that as not the point: the questions should be the ones popular explainer videos ask, such as why offenses run when passing gains more. Section 1.1 now has the late-lead question first, run versus pass second, and leaves the rest for him to supply. The edge cases are a "maybe later" list.
- **2026-11-17. The uncertainty research question is reframed (from `docs/research/uncertainty_and_rl.md`).** The plan's "act only when the ensemble agrees" treats not going for it as inaction, but punting is a decision too, and for a win-or-lose objective the best action maximizes the average estimate regardless of its spread. The question becomes: given limited data logged from conservative coaches, how much true win probability does each decision rule lose, and how much can honestly be claimed about coaches' errors? It is answered inside our own solved MDP, where the truth is known. Experiments E1 to E5 in that document replace the "RL tie-in" paragraph of section 4.
- **2026-11-17. M1 win-probability result, stated as it came out.** On the 2024 to 2025 test seasons our model (log loss 0.4490) clearly beats nflfastR `wp` (0.4845), which is not a fair comparison because ours sees the point spread, and is statistically tied with but not better than nflfastR `vegas_wp` (0.4466). The calibration step made our model slightly worse on the test seasons: the Platt map fitted on 2022 to 2023 shrinks predictions toward 50% (slope 0.867) because 2022 was an unusually unpredictable season, and two seasons are too few to estimate that slope. The pre-specified split stays as the headline. A pooled calibration over all non-test seasons was computed afterwards and is reported only as a diagnostic, because it was chosen after seeing test results. Hyperparameters were tuned on 2010 to 2018 against 2019 to 2021 only; the main finding was that very large leaves (min 10,000 plays) guard against memorizing individual games.
