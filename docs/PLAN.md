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
2. Fourth and goal from the 1 early in a close game: take the points or go?
3. Down 3 late, fourth and short inside field-goal range: kick to tie or go for the win?
4. Up 7 late in field-goal range: kick to go up 10 or go for it to end the game?
5. Fourth and 1 near midfield in the first quarter of a tied game: is it really an obvious go?
6. Was that famous collapse really historic, or is a 95% peak for a loser ordinary? (Answered by the blown-lead analysis.)

Questions 2 to 5 are candidates and can be swapped for better ones. Question 1 is fixed.

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
