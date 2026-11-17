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

### 3.8 Fan questions
The method sections above are the engine. This section is what the engine is for: the arguments fans actually have on Sunday, answered with numbers. Each question gets a plain-English answer (a few sentences, no jargon), one figure, and an honest note on what the data cannot settle. They are collected in `docs/fan-questions.md` and linked from the README.

Rule for every answer: judge the decision, not the outcome. A good call that fails is still a good call, and each write-up should say which one it was.

**Tier 1: core, answered by the main milestones**

| # | Question | What we compute | Milestone |
|---|---|---|---|
| F1 | "Should they have gone for it?" | For any fourth down: win probability for go, kick and punt, with intervals and a verdict of go, kick, punt or too close to call. Available in the app for any situation. | M2, M3 |
| F2 | "Is my coach a coward?" | Per team and season: how often they went for it when the model confidently said go, and the total win probability given up. Ranked across the league. | M7 |
| F3 | "Are coaches actually getting braver?" | League go rate in confident-go situations, by season since 1999, with the 2018 to present analytics era marked. | M7 |
| F4 | "Was that the worst collapse ever?" | The biggest blown leads ranked two ways: by raw peak win probability, and by how surprising that peak is given the pregame odds. Show where famous collapses land on each list. | M3b |
| F5 | "Is the win probability on TV lying to me?" | Plain-language version of the blown-lead calibration test: how often a 90% team should lose, and how often it really does. | M3b |
| F6 | "The call failed, so the analytics were wrong, right?" | For confident go recommendations, split by whether the attempt converted. Show that failed attempts were still the right call on average, and how much a single attempt is luck. | M3 |

**Tier 2: extensions, each a small add-on to an existing milestone**

| # | Question | What we compute | Milestone |
|---|---|---|---|
| F7 | "Go for two or kick the extra point?" | Same decision engine applied after a touchdown: two-point rate vs extra-point rate (post-2015 rule, kick from the 15), and the score-and-time chart of when to go for two. | M2 |
| F8 | "Down 8 late, go for two after the first touchdown?" | The well-known argument for going for two early when down 8 (or 14). Compute it with our own conversion rates and show the margin and its uncertainty. | M4 |
| F9 | "Kick the field goal down 4 with two minutes left, or go for it?" | Late-game decisions where clock and timeouts matter, which only the full-game DP can handle properly. A few classic spots worked through. | M4 |
| F10 | "Does icing the kicker work?" | Field-goal make rate after an opponent timeout right before the kick vs no timeout, controlling for distance, weather and kicker. Selection caveat: teams ice on harder kicks. | M1 |
| F11 | "Is momentum real?" | After a big swing play (turnover, fourth-down stop), does the next drive go better than the game state alone predicts? If the state is truly Markov, it should not. Links directly to the martingale idea in 3.5. | M3b |
| F12 | "Kick or receive in overtime?" | Overtime decisions under the current rules. The rules changed in 2022 (playoffs) and again for the regular season; verify the current rules and seasons during M0, and only use games played under the same rules. Small sample, so the answer may be "not enough data", which is a fine answer. | M4 |

**Tier 3: case studies** (one famous real decision each, with the numbers at that moment; pick three to five once the engine works, favoring fourth downs that fans still argue about, and include at least one where the coach was right and the critics were wrong).

What fans should come away with, in one line each:
- Most fourth-down calls are not close, and coaches still get many of the clear ones wrong.
- Some calls really are close, and the honest answer is "too close to call".
- A 90% favorite losing is not a scandal; it should happen about one time in ten.

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

Each milestone is also done only when the fan questions tagged to it in 3.8 are answered, each with a figure and a short plain-English write-up in `docs/fan-questions.md`.

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
- **Fan-question cards:** one shareable figure per fan question in 3.8, each with the question as its title and the answer in one sentence underneath.
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

**Explain it to a fan** (no jargon allowed in the answer):
15. A friend says "they went for it and failed, so it was a dumb call." What do you say?
16. Why is a team that was a 95% favorite and lost not necessarily a historic choke?
17. Why should a team down 8 late go for two after the first touchdown?
18. Your coach is ranked least aggressive. What would you need to see before calling that a mistake rather than a smart read of his own team?
19. Does icing the kicker work, and why is the raw comparison misleading?

## Decision log
- 2026-11-17: Added section 3.8, fan questions. Reason: the plan covered the method but not the questions a general football audience asks. Each fan question is tied to an existing milestone rather than added as a new milestone, so the build order does not change. Added an "explain it to a fan" block to section 7.
