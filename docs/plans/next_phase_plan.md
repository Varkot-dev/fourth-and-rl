# Next phase plan

Prepared 2026-11-19 for Varshith. Every number below comes from scripts that the research agents ran today on development seasons only, or from a cited source. No 2024, 2025 or 2026 outcomes were explored. Where two agents used different sample definitions, the definition is stated next to the number.

---

## 1. What to run tomorrow (just click Run all)

### Already done
The world-model v3 notebook is finished, tested (134 tests passed, 2 skipped; `ruff` clean; a CPU dry run of all 44 cells finished with 0 errors) and pushed to branch `claude/world-model-colab`. All three links below open now.

### Order

| # | Notebook | Link | Time | Runtime |
|---|---|---|---|---|
| 1 | Ground truth | https://colab.research.google.com/github/Varkot-dev/fourth-and-rl/blob/claude/world-model-colab/notebooks/world_model_ground_truth.ipynb | about 90 min | GPU (Runtime > Change runtime type) |
| 2 | Team matchups | https://colab.research.google.com/github/Varkot-dev/fourth-and-rl/blob/main/notebooks/team_matchups_colab.ipynb | about 60 min | GPU |
| 3 | World model v3 (after the push) | https://colab.research.google.com/github/Varkot-dev/fourth-and-rl/blob/claude/world-model-colab/notebooks/world_model_colab.ipynb | 55 min budget; planning plus close-call refinement could take about 25 to 30 min of that | **T4**, the same GPU type as the v2 run |

Why this order: the ground-truth run is the longest, so start it first. Run the notebooks one at a time unless your Colab plan allows two GPU sessions at once.

### Steps for each notebook
1. Open the link (the repo is public, so no GitHub sign-in is needed).
2. Runtime > Change runtime type > GPU.
3. Leave the parameters cell as it is. For v3 that means `BUDGET_MIN = 55`, `STOP_AFTER = None`, `USE_DRIVE = True`.
4. Runtime > Run all, and accept the Drive mount.

### What to expect from v3
- The preflight should say the saved preset was replaced (version 3) and the v2 results were archived under `archive/` in the Drive folder.
- On a T4, where the benchmark keeps the same sizes, the v2 data, model, eval and test-scoring stages are reused. Only the plan stage reruns. On a different GPU type everything reruns, so pick T4.
- v3 writes its own preregistration (`prereg/<hash>.json`) before its plan stage reads the test seasons. The v2 preregistration (477f07d599027e39) stays.
- What changed in v3: the field-goal limit is now a 60-yard attempt (yardline_100 <= 42), labelled post hoc and citing the kicker selection-bias analysis; and a second planning pass gives more rollouts to the two best actions on calls flagged too close (cap 4096 rollouts on T4).
- If refinement runs out of time, the plan is marked partial and the next session continues it. Just run it again.

### After the runs
- Download `world_model_results.zip` and unzip it into `results/world_model/` on the branch.
- In the v3 summary read three things: the "close calls:" line, `stages.plan.refinement` (how many close calls were resolved, still close, or changed their best call), and the agreement before and after refinement. Quote the agreement on well-supported calls.
- Claude can read the results straight from your Google Drive afterward, so you do not need to copy numbers by hand.
- The CPU dry-run numbers (8 too close, 2 resolved) came from a toy model and mean nothing.

---

## 2. What we learned today, in plain language

### The data that exists
Sources: nflreadpy (https://github.com/nflverse/nflreadpy); field dictionaries at https://nflreadr.nflverse.com/articles/dictionary_participation.html and https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html.

- **Participation data** (formation, personnel, box count) starts in **2016**. **Man/zone coverage labels** start in **2018**. **FTN charting** (QB sneak, play action, blitzers, box) starts in **2022**. So for anything coverage-aware, development data is 2018 to 2023, and for FTN features only 2022 to 2023.
- **Coverage is labelled on passes, essentially never on runs.** In 2022 the label is on 0 of 15,037 runs; in 2023 on 1,098 of 14,877. On short-yardage runs in 2022 to 2023 it is on 47 of 2,409. So "run against man" cannot be measured.
- **The labels have holes that depend on the outcome.** In 2018 to 2022, sacks and scrambles were never given a coverage label. Sacks almost never convert (0.2%), scrambles usually do (87% on 296 short-yardage plays). Ordinary throws are 99.4% labelled. In 2023 sacks and scrambles are labelled. Any model fitted on "labelled passes" quietly drops both groups.
- **The 2023 data changed its coding.** Formation labels went from seven values to three (pistol, shotgun, under center), new coverage labels appeared, missing labels became empty strings (23,252 of 46,168 rows in 2023, which a simple null check counts as filled), and the share of passes labelled man went from 0.287 (2022) to 0.415 (2023).
- **Sample sizes.** Third or fourth down, 1 to 3 yards to go, run or pass, 2016 to 2023: **18,957 plays, of which 3,245 are fourth downs**. That is a median of 74 plays per offense-season, and 12 on fourth down. An offense-versus-defense pairing gets a median of 5 plays per season.
- **Team effects are real but small and hard to see in one season.** Teams differ by about 2 points of conversion rate (1 SD, adjusted for down, distance and call mix), and that difference persists. But one season of about 74 short-yardage plays gives a team's own record only about 15% weight. The repo's M8 work already found that the pregame spread captures almost all of this.
- **Coverage associations** (not causes; the defense picks coverage, and it is charted after the snap):
  - Like for like (sacks and scrambles removed): passes convert 0.557 against man and 0.619 against zone in 2018 to 2022, and 0.536 against 0.637 in 2023.
  - **At the goal line it flips:** inside the 5, man 0.504 (746 plays) against zone 0.402 (97 plays). Defenses play man 0.821 of the time inside the 10 against 0.548 outside it (2022 to 2023 labelled short-yardage throws). So one league-wide "zone is easier" number is wrong.
  - QB sneak: 0.819 (487 plays) against 0.641 for other runs (2,107 plays), 2022 to 2023.
- **Defense coverage tendency is a real trait.** A defense's all-downs man share agrees 0.848 between odd and even weeks. Its short-yardage man share is much noisier (0.489), so it needs heavy shrinkage.
- **The Ja'Marr Chase idea cannot be measured with public data.** Receiver man-minus-zone success agrees only 0.089 between halves of a season (rough 95% interval -0.03 to 0.21, 249 receiver-seasons) and 0.003 season to season. Chase himself had 5 short-yardage targets against man in 2023. Cincinnati had 4 fourth-and-1-to-2 passes against man in 2022 and 1 in 2023. This means "not detectable here", not "false": the test had almost no power, and target-based numbers are biased because QBs throw to whoever is open.

### What the literature says is solved and what is open
Most sources below were seen only through search summaries because arxiv.org, nber.org and some other hosts were blocked today. They are marked unverified until checked.

**Solved or well known (do not claim as ours):**
- Public fourth-down tools use team strength only through the spread and total (nfl4th, read directly at https://github.com/nflverse/nfl4th/blob/master/data-raw/_go_for_it_model.R) or proprietary team and QB strength (NGS Decision Guide).
- NFL offenses do not play run/pass minimax (Kovash and Levitt, https://www.nber.org/papers/w15347, unverified).
- Coaches are too conservative (Romer 2006, https://doi.org/10.1086/501171).
- Bootstrap uncertainty for fourth downs (Brill, Yurko and Wyner, https://arxiv.org/abs/2311.03490).
- Man/zone identification from tracking data (Dutta, Yurko and Ventura, https://arxiv.org/abs/1906.11373).
- Theory for caution under model uncertainty: robust MDPs, percentile optimization, CQL, pessimistic value iteration, SPIBB (https://proceedings.mlr.press/v97/laroche19a.html).

**Open, as far as public sources show:**
- No public fourth-down model conditions on coverage, box count or personnel.
- No public payoff table of offensive call against defensive coverage estimated from charting data, with an equilibrium solved over it.
- No public fourth-down tool turns its uncertainty into the decision rule itself, and no comparison of what each cautious rule costs in win probability.

The honest headline may turn out to be "matchup changes the fourth-down call only in a narrow band of states". Measuring how narrow is itself a result.

---

## 3. The next phase, step by step

### Ground rules for all three steps
- **Branch:** the two-team solver (`mdp_team.py`, `mdp_team_jax.py`) is merged into main and into the working branch. Team-specific results stay out of the main README headline unless promoted, as `docs/PLAN.md` says.
- **Seasons:** development is 2010 to 2023 (train 2010 to 2021, calibration 2022 to 2023). 2024 to 2025 is scored once, after a preregistration is committed. 2026 is never loaded.
- **Already explored:** 2022 to 2023 coverage and cell rates were looked at while designing this, so they are no longer a clean check. The 2024 to 2025 test is the only clean out-of-sample check. Record this in the decision log.
- **Known leak to fix first:** the repo's B2 team-effects finding was computed on 2011 to 2025, which includes the test seasons. Rerun it on 2011 to 2023 before using it as a design input, and log the leak.
- **Same definitions as the repo:** converted means `(first_down | off_td) & ~turnover`, as in `components.GoModel`. Decide explicitly whether penalty first downs count.

### Step 0. A clean data loader (one or two commits)
- Load participation and FTN, join on (game_id, play_id), cast `play_id` (it is Float64 in some seasons and Int32 in others).
- Treat empty strings as missing, and a box count of 0 as missing.
- Flag sacks and scrambles (nflverse codes scrambles as runs).
- Parse personnel strings across the 2023 format change; harmonize or add an era indicator.
- Record library version and download date next to the files. Check the FTN license and attribution terms before anything goes public.
- Tests: 2018 to 2022 sacks and scrambles are unlabelled; 2023 empty strings count as missing; join coverage and value ranges.

### Step 1. Cautious decision rule ("punt and trust the defense when on the fence")
Do this first: it builds on M3 and M8 and needs no new data.

**What it is.** For each fourth-down state, draw many plausible versions of the model (M3's resampled games for the play models, M8's effect models refit with the same game weights, team ratings drawn from the rating filter), solve each one, and read the gain of going over the best kick inside each draw. Then apply a rule:
- Headline: the expected-value call (the action with the best average win probability).
- Opt-in "fence" mode: go only if the average go value beats the average best kick by more than a margin delta fixed in advance; otherwise kick. Defined on the difference of averages, its cost is at most delta per decision.

**Why it matters.** It turns "on the fence" from a feeling into a rule with a visible price. Win probability is already a probability, so the expected-value call maximizes wins if the model is right. Caution only pays when the model is wrong in a particular direction, and it costs when the model is wrong the other way. Punting is not automatically safe: the punt's value depends on the opponent's offense rating and our defense rating, both uncertain.

**ML or data problem.** Mostly a decision and statistics problem. The data are already there.

**What today's demo showed** (one-step engine, 5,520 synthetic chart cells, a stand-in for the exact solver):
- "Go only if the lower bound beats the kick" moved 1,006 cells from go to kick and raised held-out regret from 0.094 to 0.303 win-probability points per cell. It protected well if conversion was 0.4 logits too optimistic (0.020 against 0.236) and cost heavily if 0.4 too pessimistic (0.899 against 0.266).
- A symmetric robust rule moved more cells toward going (308) than toward kicking (51).
- The fence rule's cost figure (0.102 against 0.094 at delta 0.5) was in-sample and used a slightly different definition. It must be re-measured held out before you pick delta.

**How we test it.**
- Score every rule held out: build it on even-numbered draws, score on odd, and the reverse.
- Stress worlds that err in both directions: conversion, field-goal make rate, team ratings shrunk to average, opponent offense better or worse, punt net distance. **Set their sizes from calibration-season (2022 to 2023) prediction errors, not guesses**, and write the world list, weights and pass criterion to a preregistration file before running.
- Add a variant with the spread widened, because the plain game bootstrap undercovers (77% coverage for a nominal 90% in M3b).
- Two named variants: "kick as default" and "coach's usual call as default". They differ in states where coaches usually go.

**Critics' fixes folded in.**
- Call the draws an approximate predictive distribution, not a posterior. Treat "average over draws beats the point estimate" as a sensitivity check, not a finding.
- Tune on end-of-2023 ratings and development states only. End-of-2025 ratings include test-season outcomes, so use them only on final display cards, labelled as test-stage. Add a guard that refuses ratings past 2023 unless a display flag is set.
- Hold the clock, kickoff and try tables fixed across draws (required for GPU batching, since `solve_end_phases` checks every pair against the first).
- Every draw must evaluate the same available actions (the demo had 180 cells where a missing field goal silently tilted toward go).

**What could make us drop it.** If the held-out cost of the fence rule is large at every delta you find acceptable, and the stress worlds show no plausible direction of model error where it protects, keep only the expected-value call plus the uncertainty labels.

**Scope for tomorrow.** `decision_rules.py` (pure functions with hand-checkable tests) and one notebook cell that runs draws for a single matchup after a GPU timing benchmark. The whole-game version (forced-action evaluation in the solver) comes later. No GPU timing exists yet; on the CPU, 101 draws would take about 94 minutes per matchup (arithmetic from the measured 56-second end-phase solve).

### Step 2. Conversion model with matchup and coverage
**What it is.** A short-yardage conversion model that starts from the repo's existing curve (`GoModel`) and learns only how far each situation moves from it, using:
- the call (run, pass, QB sneak) by distance;
- pre-snap features available on every play (box count, personnel; FTN sneak and play action from 2022);
- shrunk offense and defense effects from the past-only M8 rating filter;
- for passes only, the defense's expected man share in that situation, times a league man-versus-zone term (with a shrunk offense term that the data are expected to push to zero).

A called pass is split into three outcomes: sack, scramble and throw. The coverage term is fitted on throws only, and mixing over coverage is done on the probability scale: p = (1 - m) times P(zone) plus m times P(man).

**Why it matters.** It is the direct test of your idea that matchup and coverage should change fourth-down advice. Nothing public does this.

**ML or data problem.** Mostly a data problem. The model itself is an ordinary shrunk logistic regression in the style of `kicker.py`. The hard parts are label holes, the 2023 coding change, small samples and the fact that the defense chooses its coverage.

**How it plugs into the solver.** Generalize `mdp_team.tilt_kernels` so the conversion shift can differ by field state (one line changes; a test checks a constant array matches the old scalar). States with 4 or more to go keep M8's shift.

**How we test it.** One preregistered test on 2024 to 2025, committed before test data is read:
- Score the probability the solver actually uses: one go probability averaged over the call mix, before any play is called. Scoring "given the actual call" is secondary, because calls are often made after seeing the box.
- Compare against league average and against the current M8 tilt, on log loss with a 95% interval from 2,000 resamples of whole games.
- Verdicts: PASS, FAIL or INCONCLUSIVE. **Inconclusive is the most likely**, because M8's test interval was about plus or minus 0.65 per 1,000 against an estimated ceiling of about 0.5 per 1,000 for any team effect. The rule written in advance: if inconclusive or fail, keep M8's tilt.
- Before reading any test outcome, check fill rates on 2024 to 2025 and record them.
- Secondary: does the defense man-share model predict better than league rates; does knowing the realized coverage add signal (diagnostic only); is the offense-against-man variance distinguishable from zero.

**Critics' fixes folded in.**
- Fit coverage effects separately for goal-to-go and open field, with an era term.
- Calibration bounds as first drafted would fail a perfectly calibrated model up to three quarters of the time when predictions are narrow. Set the bounds by simulation before preregistering.
- Report all man/zone comparisons on a like-for-like base.
- Train coverage terms on 2018 to 2021 and calibrate on 2022 to 2023 to match `configs/splits.yaml`. Fitting FTN features on 2022 to 2023 is allowed by the split rules, but then those seasons can no longer check those features.
- Do not quote the offense man-minus-zone figure of -0.20 until it is rechecked; it is suspiciously far below zero for pure noise.
- No "WR1" feature built from full-season target counts (that looks ahead). If kept, define it from earlier games only.

**What could make us drop it.** An inconclusive or failing test means the matchup features are "not established" and the solver keeps M8's tilt. The README says so.

### Step 3. Play call against coverage as a game (where RL truly fits)
**What it is.** On third and fourth and short, the offense picks a play family and the defense picks a coverage. Each such state becomes a small zero-sum game instead of a single max. The exact solver can still solve the whole game backward, because time always runs down: at each tick each short-yardage state solves one small game. Because the answer is exact, it can grade multi-agent RL (minimax-Q, self-play) the same way M5 graded single-agent RL.

**Why it matters.** It is the honest way to say "the defense adapts". It answers the Chase question in the right form: a quick out against man is strong only until the defense shifts its mix, and the equilibrium says how often to call it, not "always".

**ML or data problem.** Both, and the data side is the hard part.

**Structure (critics' fixes folded in).**
- Model it in order: offense formation and personnel, then the defense's box (which the offense sees), then the call and the coverage chosen at the same time.
- Label calls only from what was known before the snap: run or dropback, QB sneak, play action, screen, RPO. **Do not** use "quick pass to WR1" or where a run ended up; those labels already contain the defense's choice and the result.
- Treat "WR1 beats man" as a receiver-level shift on passes against man, estimated from all dropbacks.
- Payoffs should come from a small outcome distribution per call (gain, turnover, sack, touchdown), not one conversion rate, because a sneak and a deep shot with the same conversion rate are worth different amounts.
- Build payoffs per situation band (at least goal-to-go against open field), with shrinkage toward a league table and an interval on every cell. Team-specific cells would be fitted to noise (median 59 short-yardage plays per team-season in 2022 to 2023).
- Solve small games exactly (closed form or a linear program). Plain regret matching would need on the order of 100 million iterations per state for the target accuracy.
- Offline Markov-game learning from logs is not feasible here (about 330 labelled fourth-and-short passes over two seasons and almost no labelled runs). Say that in one sentence; it is not a research project.

**An important correction.** The equilibrium value is **not automatically the cautious number**. It is what the offense can guarantee if it mixes well, and it can be above or below today's go value. The value that is cautious by construction is the defense's best response to the offense's current mix. Report all four go values side by side: historical mix, equilibrium, offense best response, defense best response.

**How we test it.**
- Known-truth simulation first: plant a payoff table, generate plays where defenses choose by situation, check that the estimator recovers it, and measure the bias from missing run coverage.
- Solver tests: matching pennies and rock-paper-scissors values; LP and iterative answers agree; with one call and one defense the go value equals the current solver exactly; the historical-mix value reproduces the current go value within bootstrap error.
- A sensitivity analysis for unobserved confounding (how much the game value moves if selection on observables is wrong). The minimax indifference test is descriptive only.
- Power simulations before any team-level test.
- Headline output: the number of fourth-down chart states where go, field goal or punt flips when the defense adapts, with an interval and a play count beside every cell.

**What could make us drop it.** If the known-truth simulation shows the payoffs cannot be recovered at realistic sample sizes, or the flipped-state count is zero within its interval, keep it as an explanatory module labelled "untested for decisions".

---

## 4. Decisions only you can make

1. **What "on the fence" means.** Distrust of the model's numbers (ambiguity, which the literature supports and we can measure), or a wish for less variance in outcomes (risk aversion, which costs wins when the goal is win probability)? Or avoiding blame for a visible failed fourth down, which would need its own labelled penalty?
2. **How much caution costs you.** Which fence margin delta, fixed before the exact-solver run: 0.5 or 1.0 win-probability points, or a margin tied to each state's spread? Should cautious mode ever be the headline, or only an opt-in shown next to the expected-value call with its cost?
3. **Which way you think the model is wrong late in games:** conversion too optimistic, field goals too optimistic, or our defense rated too well? The stress-world weights are written down before the run.
4. **Who moves last on fourth and short?** How often does a short-yardage call change at the line after the offense sees the box and shell? This decides whether the game is simultaneous or the offense moves second.
5. **Play families.** Which calls make football sense at fourth and short: inside run, outside run, QB sneak, quick pass, play action, screen, deep shot? Fewer categories mean more data per cell. Would you merge play action into other passes, or sneak into inside run?
6. **Defense actions.** Man or zone only, or also box and blitz? Each extra dimension thins the cells.
7. **FTN features** (sneak, play action, blitzers) exist only from 2022: fit them on 2022 to 2023 and lose an independent check, build a play-by-play sneak proxy checked against FTN labels, or leave them out?
8. **The 2023 coding change:** harmonize to a common vocabulary (under center against shotgun, man against zone, coverage family) or keep an era indicator?
9. **The Chase idea:** keep it only as a preregistered hypothesis expected to come out null, or look for better data (NGS separation, not yet checked)?
10. **Third and short:** should the new conversion model and the game cover third downs too, or only the fourth-down node first?
11. **Playoffs:** keep the 880 playoff plays (of 18,957) in training?
12. **Named teams on cards:** say "Detroit", or "an offense rated +0.40, SD 0.31", given that the ratings are uncertain and come from the test stage?
13. **Are you comfortable** with "inconclusive means keep M8's tilt", and with a headline that may be "matchup changes the call only in a narrow band"?

### Candidate features to vet with football judgment
Mark each keep, drop or change. Raw rates are descriptive and confounded by situation.

| # | Feature | Who chooses it | Data fact |
|---|---|---|---|
| 1 | Call: run, pass, QB sneak | Offense | Short yardage 2018 to 2023: run 0.686, pass 0.541. Sneak 0.819 (487 plays) in 2022 to 2023. Should sneak be its own call? |
| 2 | Defensive box count, heavy front | Defense, seen by offense | Runs into 9 or more in the box 0.635 against 5 in the box 0.785. Can the offense audible after seeing it? |
| 3 | Defense man share in this situation | Defense trait | 63% man on short-yardage throws against 34.5% overall. All-downs share stable (0.848), short-yardage share noisy (0.489). |
| 4 | Offense and defense run strength on all downs (line quality) | Unit trait | Not yet measured. Many more plays than short-yardage ratings. |
| 5 | Offensive personnel weight (11, 12, 13, 22, jumbo) | Offense | 2018 to 2023: 11 personnel 7,280 plays, 12: 1,496, 13: 634, 22: 612. |
| 6 | QB rushing threat (designed QB runs) | Offense trait | Not yet measured. |
| 7 | Play action on passes | Offense | 0.562 (372) against 0.510 (2,158), 2022 to 2023 only. |
| 8 | Defensive pass rush and blitz tendency | Defense | FTN blitzers from 2022; sacks from play-by-play for all years. |
| 9 | Offense passing edge against man (team) | Unit trait | Not detectable at public sample sizes. |
| 10 | WR1 against man (Chase quick out) | Player trait | Not detectable (0.089 between halves, 0.003 season to season). Heavily shrunk, tested term only. |
| 11 | Motion, RPO, screens | Offense | FTN, 2022 onward. Low priority. |

**Never inputs:** who was targeted, realized coverage, pressure, time to throw, `read_thrown`. All are known only after the snap.

---

## 5. Risks and honest limits

- **Coverage is chosen by the defense and charted after the snap.** Every man-against-zone number is an association. An equilibrium solved on confounded payoffs can be confidently wrong.
- **Coverage on runs is missing.** Run-against-coverage payoffs are not identified without an imputation assumption or bounds.
- **Outcome-dependent label holes.** In 2018 to 2022, sacks and scrambles have no coverage label; fitting on labelled plays biases pass conversion in an unknown direction unless they are modelled separately.
- **The 2023 coding change** could look like a trend, and 2024 to 2025 may follow 2023's coding rather than 2018 to 2022's. Check fill rates on the test seasons, blind to outcomes, before scoring.
- **Low power.** The plausible gain from matchup features is at or below what the test can resolve. The likely headline is "not established", and the README must say so rather than present point estimates as findings.
- **Small samples at team, matchup and receiver level.** Any team-specific coverage advice would mostly reflect noise. "Not detectable" is not "zero".
- **Caution has a price.** A rule that defaults to punting can look prudent while giving back wins. In development data coaches went on 8,202 of 55,328 fourth-down decisions, punted on 33,708 and kicked 13,418 field goals, so a punt-leaning rule moves toward the status quo and needs evidence.
- **The uncertainty draws are narrow.** The game bootstrap undercovers (77% for a nominal 90%), and it misses model-form error (yards tilt, league-average kicking, fixed clock tables).
- **The demo numbers in Step 1** come from the one-step engine on synthetic cells. They show the method, not the size of the effect in the exact solver.
- **Leakage traps:** end-of-2025 ratings include test outcomes; B2 included test seasons; 2022 to 2023 have been explored. Each needs the fix listed above and a decision-log entry.
- **Unverified sources.** Most literature items were seen only in search summaries, including the conflicting Kovash and Levitt figures ("about half a win per year" against "about one point per game"). Some DOIs are from memory. Verify before anything goes into notes or the README, and claim no firsts.
- **Licensing.** FTN and participation data carry their own terms; check attribution before publishing figures.
- **Compute.** No GPU timing exists for the uncertainty runs yet; a benchmark comes first.
- **Branch.** The solver pieces live on `claude/team-specific-solver`; the v3 notebook is uncommitted in `/home/user/fourth-and-rl-wm`. Nothing from today is pushed.
