# fourth-and-rl: plan

NFL fourth-down and play-calling decisions as a decision process: statistical models, exact dynamic programming and reinforcement learning.

## 1. The question

On fourth down a coach chooses: go for it, kick a field goal, or punt. Analysts have argued for years that coaches are too conservative. This project:

- (a) rebuilds the standard analysis from raw play-by-play data,
- (b) adds honest uncertainty to it,
- (c) uses the problem as a real-data setting to learn reinforcement learning, because it is a true sequential decision problem with a clear reward (winning).

It also works as a research topic. The project is built around the work of Abraham Wyner's group at Penn (Wharton Statistics): how much win-probability models can be trusted, what a "blown lead" really tells you, and what that means for fourth-down recommendations. The RL part then asks the follow-up question: what should a decision-maker do when the win-probability estimates it relies on are that uncertain?

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

### 3.8 Research questions
The method sections above build the tools. This section is what the tools are for: a small set of specific, answerable questions in the style of the Wyner group's work, each answered so that the result survives a skeptical statistician or interviewer. A result that cannot be defended does not go in the README.

**Standard of evidence for every question.** Each answer must report:
1. **The estimand**, stated in one sentence before any code is written (what number, for which population of plays or games).
2. **The estimate with an interval**, from the game-level bootstrap or a stated Bayesian model, plus the sample size in games, not just plays.
3. **An out-of-sample check**: models fit on earlier seasons, evaluated on later ones.
4. **A falsification or power check**: a test the result could have failed, or a simulation showing the method can detect an effect of the size claimed.
5. **The strongest remaining objection**, written down, with what was done about it and what was not.

The analysis for each question is written down in `docs/notes/` before the final run (features, test, threshold), so the result cannot be tuned after seeing it. Each finished question becomes a short technical note in the same folder: question, method, result, limitations, one figure.

| # | Question | Method | Main objection and how it is handled | Milestone |
|---|---|---|---|---|
| R1 | **Break-even conversion rates.** For each fourth-down situation, what conversion probability makes going for it exactly as good as kicking or punting, and is the estimated conversion rate confidently above or below it? | Solve for the break-even rate from the decision engine, then compare with the bootstrap interval on the estimated rate. | The decision depends on several fitted models. Isolating the one most uncertain input (conversion rate) makes the claim easy to check and hard to argue with. | M2, M3 |
| R2 | **Are coaches too conservative, once uncertainty is honest?** How much win probability per season is lost on fourth downs where the recommendation is confident? | Coach evaluation from 3.6, counting only decisions where at least, say, 90% of bootstrap refits agree. Revisits Romer (2006) and Brill, Yurko and Wyner (arXiv 2311.03490) with our own models. | Selection on what the coach knows and the model does not (injuries, matchups). Sensitivity analysis: how large a hidden conversion-rate advantage would a team need for its choices to be correct? If the answer is implausibly large, the conclusion stands. | M3, M7 |
| R3 | **Is aggressiveness a stable coach trait?** Do coaches have consistent, measurable risk preferences, or is "aggressive coach" mostly noise? | Choice model: probability of going for it as a function of the model's win-probability gain from going, with a coach-level effect and hierarchical shrinkage. Test stability with split-half reliability across seasons. Related to Sandholtz and coauthors (arXiv 2309.00756). | Coach and roster are confounded (a coach with a great offense goes for it more). Include offensive strength features and report how much of the coach effect survives. | M7 |
| R4 | **Can the blown-lead test detect a miscalibrated model with the data we have?** | Apply the 3.5 reference distribution to the nflfastR win-probability column and to our model. Then, by simulation, deliberately make a model overconfident by known amounts and find the smallest miscalibration the test detects with 80% power over the available seasons. | "The test passed" means little if the test has no power. The power curve answers that directly. Discreteness of real paths is handled by the simulation in 3.5. | M3b |
| R5 | **How much data does a win-probability model need before its fourth-down advice is reliable?** | Extend the Brill, Yurko and Wyner simulation (arXiv 2406.16171): in simulated leagues with known truth, measure how often the recommended action is wrong as a function of seasons of training data. | Results depend on the simulator. Check that the simulator reproduces real score and drive-outcome distributions first (3.7), and report results for more than one simulator setting. | M3b |
| R6 | **Is the game Markov in the state we use?** Does a recent swing play (turnover, failed fourth down) predict the next drive's outcome beyond what the state already explains? This is the formal version of "momentum". | Add a short, pre-specified list of lagged event features to a next-drive outcome model and test them jointly, with a permutation test and a multiple-testing correction. | If momentum exists, the MDP in 3.1 is missing state, which matters for M4 and M5. If the test finds nothing, report its power, so "no effect" is not overclaimed. | M3b |
| R7 | **When is going for two correct, and how sure can we be?** | Two-point attempts are relatively few, so estimate the conversion rate with a beta-binomial model pooled across teams, use only seasons after the 2015 extra-point rule change, and run the decision engine after touchdowns. Includes the down-8 question (go for two after the first touchdown). | Small samples. Show the score-and-time regions where the recommendation flips across posterior draws and label them too close to call. | M2, M4 |
| R8 | **Does icing the kicker change field-goal success?** | Compare make rates after an opponent timeout right before the kick with similar kicks without one, adjusting for distance, kicker, weather, surface and game situation (matching or regression adjustment). | Teams choose when to ice, which is the main threat. State it plainly, report a sensitivity analysis, and report the smallest effect the data could detect. Without that, "no effect" is not a finding. | M1 |

**Resume and interview framing.** The strongest single result, most likely R2 or R4, becomes the headline: one sentence with the number and its interval, for example "Showed that X% of fourth-down recommendations from a standard win-probability model are not robust to game-level resampling" (filled in only once a script produces the number). The rest are supporting notes. In an interview, each question should be explainable in two minutes: the question, why the naive answer is wrong, what was done, and the remaining weakness.

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

Each milestone is also done only when the research questions tagged to it in 3.8 are answered to the standard of evidence there, each with a figure and a technical note in `docs/notes/`.

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
- **Research-note figures:** one figure per question in 3.8, each showing the estimate with its interval, not just a point.
- **Power curve** for the blown-lead test (R4): detection probability against the amount of overconfidence.
- **Two-point chart:** score differential by time remaining, colored by go for two vs kick.

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

**Defend it** (answer as if a skeptical reviewer asked):
15. A reviewer says your coach-evaluation result is just selection bias. What exactly is the bias, and what does your sensitivity analysis show?
16. Your blown-lead test found no miscalibration. Why is that not yet evidence the model is calibrated?
17. Why report a break-even conversion rate instead of only the recommended action?
18. How did you make sure you did not tune the momentum test until it found something?
19. Your two-point recommendation is based on a few hundred attempts per season. Why pool across teams, and what does pooling cost you?
20. What is the single weakest assumption in your headline result?

## Decision log
- 2026-11-17: Added section 3.8, first as fan-style questions, then reworked the same day into research questions with a fixed standard of evidence (estimand, interval, out-of-sample check, falsification or power check, stated remaining objection). Reason: the owner wants results that are defensible to a statistician and strong enough for a resume, in the style of the Wyner group's work, rather than casual fan answers. Each question is tied to an existing milestone, so the build order does not change. Section 7 gained a "defend it" block of reviewer-style questions.
