# My questions

This branch is for the questions I want answered for myself. It is built on top of the core branch (`claude/optimistic-mendel-v2267n`, later `main`), which keeps the research and reinforcement-learning milestones. The core plan in `docs/PLAN.md` still applies here for code quality, honesty rules and commits.

## How this branch works

- Bring core work in regularly: `git merge <core branch>`. Never merge this branch back into the core unless I decide a finished result belongs in the main README.
- Reuse the core's data pipeline, win-probability model, play-outcome models, bootstrap and, once built, the full-game solver. Do not fork them; if one needs a change, make it on the core branch first.
- Each question gets a script under `scripts/questions/`, results under `results/questions/`, a figure under `assets/questions/`, and a short write-up in `docs/questions/<name>.md` with the answer, how confident it is, and what would change it.
- New ideas go to `docs/research/inbox/` on this branch.
- The X research kit (`research-kit/x/`) and its findings feed this branch.

## Standard of evidence (applies to every question below)

The goal is answers that hold up when a statistician or an interviewer pushes on them, in the style of the Wyner group's papers. Each write-up in `docs/questions/` must report:

1. **The estimand**, stated in one sentence before any code is written: which number, for which plays or games.
2. **The estimate with an interval**, from the game-level bootstrap (`src/fourth_and_rl/bootstrap.py`) or a stated Bayesian model, with the sample size in games, not only plays.
3. **An out-of-sample check** using the season splits and leakage guards already in `src/fourth_and_rl/splits.py`. The 2026 holdout stays untouched until a result is final.
4. **A falsification or power check**: a test the result could have failed, or a simulation showing the method can detect an effect of the size claimed. "No effect" is only reported together with the smallest effect the data could have detected.
5. **The strongest remaining objection**, written down, with what was done about it and what was not.

The analysis for each question (features, test, threshold) is written into its `docs/questions/<name>.md` before the final run, so the result cannot be tuned after seeing it.

## The questions, in the order I want them

1. **Why do offenses run when passing gains more?** Passes gain more per play on average, yet teams run about 40% of the time. Measure the gap by down, distance and situation, then test the game-theory explanation (what matters is the value of the next run versus the next pass, and defenses adjust) with the run-or-pass version of the decision model.
2. **Are the Lions right to go for it so often?** Detroit has the highest fourth-down go rate in the league since 2021. Does a strong offense justify more aggression than a league-average model recommends, and by how much? Needs a team-strength adjustment built only from games before the one being judged.
3. **Does the model know my team?** "Does the analytics take into account their defense is terrible?" Asked constantly, answered informally. Measured by how far the recommendation moves with offense and defense strength, built from prior games only (overlaps question 3).
4. **Was it really fourth and 1?** Next Gen Stats called the Lions' fourth and 1 from their own 24 "closer to a 4th and 2", turning a very strong go into a toss-up. How often does half a yard of hidden distance flip the call?
5. **Case studies from the public debate** (`research-kit/x/2_twitter_findings.md`): the ten most argued decisions, each scored by the one-step engine, the bootstrap and the full-game solution, next to what the public models said at the time. Several are from the 2026 season, which no model in this repo has seen, so they are a clean out-of-sample check.
6. **Which coach has cost his team the most wins, and does the data back up "malpractice"?** (The owner's idea, `docs/research/inbox/coach_cost_idea.md`.) Take the calls the public labelled malpractice and show what each one cost by the models. Then build a per-coach ledger in wins: fourth-down, field-goal and clock decisions, plus the procedure penalties the owner assigns to the coach (false start, delay of game), and explicitly not encroachment, defensive offside or neutral-zone infraction. See section 3.6.
7. Further questions from the owner's video list, when supplied.

### Research questions added 2026-11-17

These fill gaps the core does not cover. Checked against `docs/research/landscape.md` and the core notes first; questions already answered there were left out (break-even conversion rates are in `docs/notes/m2_decision.md`; coach aggressiveness rankings and risk-preference inference already exist in published work).

8. **Is momentum real, or is the game Markov in our state?** After a swing play (turnover, failed fourth down, return touchdown), does the next drive go better or worse than the state alone predicts? Method: a short, pre-specified list of lagged event features added to a next-drive outcome model, tested jointly with a permutation test and a multiple-testing correction. Objection handled: if the test finds nothing, report its power. Why it matters beyond the argument on X ("analytics ignores momentum", `docs/research/public_debate_and_teams.md`): if momentum exists, the MDP state in the core is missing something, which affects the full-game solver and the RL milestones.
9. **Can the blown-lead test actually catch a bad model?** The core ran the test (M3b). This asks whether passing it means anything. Method: make a win-probability model overconfident by known amounts in simulation and find the smallest miscalibration the test detects with 80% power over the seasons available. Deliverable: a power curve. Reuses `src/fourth_and_rl/blown_leads.py`.
10. **How many seasons of data before fourth-down advice is reliable?** The core's sample-size simulation (`scripts/wp_sample_size_sim.py`) measures interval width for win probability. This carries it one step further, to the decision: in simulated leagues with known truth, how often is the recommended action wrong, as a function of seasons of training data? Objection handled: results depend on the simulator, so report more than one simulator setting.
11. **Does icing the kicker work?** Make rate after an opponent timeout right before the kick against similar kicks with no timeout, adjusting for distance, kicker, weather, surface and game situation (matching or regression adjustment). Main objection: teams choose when to ice, and they ice harder kicks. State it, run a sensitivity analysis, and report the smallest detectable effect. Reuses the field-goal model in `src/fourth_and_rl/components.py`.
12. **Is "aggressive coach" a stable trait or mostly noise?** Feeds questions 2 (the Lions) and 6 (the coach ledger). Method: a choice model for going for it as a function of the engine's win-probability gain, with a coach effect and hierarchical shrinkage; stability by split-half reliability across seasons and by whether the effect follows a coach who changes teams. Objection handled: coach and roster are confounded, so include offensive strength built from prior games only and report how much of the coach effect survives.
13. **Down 8 late: go for two after the first touchdown?** Two-point attempts are few, so pool the conversion rate across teams with a beta-binomial model, seasons after the 2015 extra-point change only. Waits on the core: it needs the two-point action in the full-game solver (landscape gap G2), because the answer depends on the later two-point choice.

### Defend it (interview practice for these questions)

- A reviewer says the coach ledger is just selection bias. What exactly is the bias, and what does the sensitivity analysis show?
- The blown-lead test found nothing wrong with the model. Why is that not yet evidence the model is calibrated?
- How did you make sure you did not tune the momentum test until it found something?
- Why pool two-point attempts across teams, and what does pooling cost you?
- Icing showed no effect. What is the smallest effect you could have detected?
- What is the single weakest assumption in your headline result?

## Ideas in the inbox

- `docs/research/inbox/coach_cost_idea.md`: which coach has cost his team the most wins; false starts and delay of game count against the coach, encroachment and offside do not.
- `docs/research/inbox/referee_and_coach_prior_work.md`: referee crews and existing work on coach rankings.
- From the X findings: the Surrender Index, penalties as coaching decisions, named coaches (Todd Bowles, Sean Payton reply thread).

## Order of work

Questions that need only what the core has already built come first (run versus pass rates, the Lions go rate, the hidden half yard, the coach ledger using the one-step engine). Questions that need the full-game solver or the bootstrap wait until the core has them. Of the research questions, 9, 10 and 11 need only what is built now; 8 and 12 need the state table and engine; 13 waits for the two-point action on the core.
