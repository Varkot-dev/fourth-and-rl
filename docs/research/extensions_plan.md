# Research extensions: detailed plan

Written 2026-11-17 on the `research-questions` branch, after merging the core branch
(`claude/optimistic-mendel-v2267n`). Goal: turn the strongest finished result (M3b, the
blown-lead calibration test) into work that a statistics researcher would find new and
defensible, then extend the same standard to the other questions in
`docs/MY_QUESTIONS.md`.

Every task follows the standard of evidence in `docs/MY_QUESTIONS.md`: estimand first,
interval, out-of-sample or simulation check, power, and the strongest remaining objection.
Each task lists its files, its "done when" and how long it should take.

## Order and why

Track A comes first because it extends the paper the email to Professor Wyner is about, it
needs no new data (pure simulation plus results already saved), and each step produces a
figure. Tracks B to E follow in order of how much existing code they reuse.

## Track A. How good is the blown-lead test? (question 9)

The M3b result says the models pass, or nearly pass, the first-crossing test. That is only
informative if the test could have failed. Track A measures that.

**A1. Power of the tests against known overconfidence.** (this session)
- Estimand: probability that each test rejects at the 5% level, as a function of how
  overconfident the model is and how many games are available.
- Miscalibration model: a feed reports p' = expit(k * logit(p)), where p is the true,
  calibrated win probability. k = 1 is calibrated, k > 1 overconfident (pushed toward 0
  and 1), k < 1 underconfident. This is temperature scaling, the standard one-parameter
  distortion. Grid: k from 0.8 to 1.5.
- Simulated league: the M3b simulator (Brownian lead, 163 reported plays per game, Gamma
  gaps with shape 1.2, both fitted to real nflfastR paths in M3b). Pregame probabilities
  p0 = Phi(Z), Z ~ Normal(0, 0.55), which gives favorites mostly between 25% and 75%.
  Assumption, stated in the note.
- Tests compared:
  1. Continuous-formula one-sided KS (d_upper) on the loser's peak PIT. This is the test
     that looks for "too many extreme peaks".
  2. First-crossing test, combined over thresholds 0.6, 0.7, 0.8, 0.9, 0.95: the largest
     |z| across thresholds, with its 5% critical value set by simulation under k = 1, so
     the size is exactly 5% under the simulator.
- League sizes: 1, 4, 8 and 16 seasons (272, 1,088, 2,176 and 4,350 games).
- Reps: 400 simulated leagues per (k, size). Monte Carlo standard error on a power near
  50% is 2.5 points; reported.
- Files: `src/fourth_and_rl/blown_power.py`, `tests/test_blown_power.py`,
  `configs/blown_power.yaml`, `scripts/research/blown_power.py`,
  `results/research/blown_power.json`, `assets/research/blown_power.png`.
- Done when: the size at k = 1 is within Monte Carlo error of 5% for the first-crossing
  test; tests pass; the figure shows power curves for both tests at each league size;
  the JSON records seeds and settings.
- Expected finding to check, not assume: discreteness makes the continuous-formula test
  conservative, so it should have low power against overconfidence.

**A2. What does the real result rule out?** Invert A1: the largest k consistent with the
M3b first-crossing results for each model (a confidence interval for k). Turns "no
detected departure" into "overconfidence larger than k = ... is ruled out". Reuses
`results/m3b_blown_leads.json`. Needs the cached data to recompute crossings; until then,
use the saved crossing counts.

**A3. Where is the mid-range overconfidence?** M3b found our model about 2 points
overconfident at first crossings of 0.6 to 0.8 (z up to 3.7). Split crossings by quarter,
score state and spread, with the team bootstrap. Then recalibrate on development seasons
only and retest on later seasons. Done when: either the fault is located and removed out
of sample, or it is shown to be spread evenly.

**A4. Dependence-correct intervals.** M3b's crossing intervals assume independent games.
Redo them with the season-stratified team bootstrap already in `blown_leads.py`
(`pigeonhole_weights`) and report how much wider they get.

**A5. Write-up.** A four-page technical note `docs/research/blown_lead_power.md`: question,
method, power curves, what the real data rules out, limits. This is the "two-page
summary" offered in the email.

## Track B. How much data before advice is reliable? (question 10)
- B1. In the random-walk football setup already solved exactly in M4 tests, fit win
  probability on 1, 2, 4, 8 and 16 simulated seasons, run the one-step decision on a fixed
  set of fourth-down states, and record the share of wrong recommendations and the win
  probability lost.
- B2. Same with the bootstrap rule (act only when confident) to see whether the rule
  lowers loss or only lowers claims, as `docs/research/uncertainty_and_rl.md` predicts.

## Track C. Is momentum real? (question 8)
- C1. Pre-register features (turnover on the previous drive, failed fourth down on the
  previous drive, return touchdown) in `docs/questions/momentum.md`.
- C2. Next-drive points model with and without them; permutation test that shuffles the
  features within score-and-time cells; Holm correction.
- C3. Power by simulation: inject a known momentum effect into the simulator.

## Track D. Icing the kicker (question 11)
- D1. Identify ices from play-by-play (defensive timeout immediately before a field goal).
- D2. Matched comparison on distance, kicker, weather, surface, time and score; logistic
  adjustment as a second method.
- D3. Sensitivity analysis and smallest detectable effect.

## Track E. Is aggressiveness a stable trait? (question 12)
- E1. Choice model with coach effects, hierarchical shrinkage.
- E2. Split-half reliability across seasons; coaches who change teams.

## Status
- [x] A1 (2026-11-17): power curves saved; write-up in `docs/research/blown_lead_power.md`. Size at 16 seasons rechecked with 1,500 leagues: 0.063 (SE about 0.008), consistent with 5%.
- [x] A2 (2026-11-17): overconfidence or underconfidence of 5% or more in log-odds ruled out for both nflfastR feeds; no single temperature passes, so the remaining error has a shape (see `docs/research/blown_lead_power.md`). Our own model still to add.
- [ ] A3 to A5
- [ ] B, C, D, E
