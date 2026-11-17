# M1 notes: the win-probability model

Code: `src/fourth_and_rl/wp.py`. Config: `configs/wp.yaml`. Script: `scripts/fit_wp.py`.
Worked examples marked "by hand" are arithmetic, not results.

**There are two versions.** Sections 1 to 5 describe version 1, the model as first
shipped; their numbers are from `results/m1_wp_metrics_v1.json` (a copy of the original
results file) and `results/m1_wp_tuning_v1.json`. Version 1 was badly wrong late in
close games. Section 6 describes version 2, which is now the headline model, how the
problem was diagnosed on development seasons only, and what changed. Its numbers are
from `results/m1_wp_v2_dev_study.json`, `results/m1_wp_dev.json`,
`results/m1_wp_v2_preregistration.json` and `results/m1_wp_metrics.json`.

## 1. Concepts

### Win probability
Win probability (WP) is a number attached to a game state: of all the times a team
has been in this situation, what share went on to win? "Situation" here means the state
before the snap: down, distance, field position, clock, score, timeouts, and how strong
the team was thought to be before kickoff (the point spread). The model is always asked
about the team with the ball (the posteam), so one model serves both teams.

WP is a prediction about the whole rest of the game, and the label that teaches it is
a single bit per game: won or lost. A 150-play game gives 150 rows that all carry the
same label. That one fact drives most of the decisions below.

### Gradient boosting in a paragraph
A decision tree splits the data with yes/no questions ("is the lead above 7?") and
predicts a constant in each final box (leaf). One tree is crude. Gradient boosting
builds hundreds of small trees in sequence. Each new tree is fitted to the errors the
current sum of trees still makes, and its output is added in with a small weight (the
learning rate). The model works on the log-odds scale: the sum of the trees is a score
z, and the probability is sigmoid(z) = 1 / (1 + exp(-z)). LightGBM is a fast
implementation that first buckets every feature into a small number of bins.

### Calibration versus accuracy
Accuracy asks "did the side you favored win?". Calibration asks "when you say 70%, does
it happen 70% of the time?". A model can be accurate and badly calibrated: saying 99%
for every favorite gets the same accuracy as saying 65%, but the 99% is a lie.

Later milestones use the probabilities as numbers, not as picks. The decision engine
compares "go for it: 0.54" with "punt: 0.52". The blown-lead analysis asks how often a
team that reached 90% still lost. Both are only meaningful if 0.54 and 0.90 mean what
they say. So calibration is the property we need, and accuracy is a by-product.

### Log loss
For one play with outcome y (1 or 0) and prediction p:

    loss = -[ y * ln(p) + (1 - y) * ln(1 - p) ]

It is the negative log of the probability you gave to what actually happened. Average
it over plays. A coin flip (p = 0.5 always) scores ln(2) = 0.693. Lower is better.
It punishes confident mistakes very hard, and it is the loss the trees minimize.

Worked example (by hand). Two plays, the first team won and the second lost.
- Predictions 0.9 and 0.2: losses are -ln(0.9) = 0.105 and -ln(0.8) = 0.223. Mean 0.164.
- Predictions 0.99 and 0.99: losses are -ln(0.99) = 0.010 and -ln(0.01) = 4.605. Mean 2.308.
One confident miss costs more than twenty coin flips' worth of caution.

### Brier score
The mean of (p - y)^2. A coin flip scores 0.25. Same two plays (by hand): predictions
0.9 and 0.2 give (0.01 + 0.04) / 2 = 0.025. It is bounded (the worst single-play score
is 1), so it is gentler than log loss on confident misses. We report both.

Both log loss and Brier are "proper" scores: the way to get the best expected score is
to report your honest probability. That is why they reward calibration.

### Reliability curve and expected calibration error (ECE)
Sort the predictions into bins (0 to 0.05, 0.05 to 0.10, and so on). In each bin plot
the mean prediction on the x-axis against the share of those plays the team actually
won on the y-axis. A calibrated model sits on the diagonal.

ECE collapses the curve to one number: the average distance from the diagonal, weighted
by how many plays are in each bin.

Worked example (by hand). 1,000 plays in three bins:

| bin | plays | mean prediction | actual win share | gap |
|---|---|---|---|---|
| low | 300 | 0.20 | 0.23 | 0.03 |
| middle | 500 | 0.50 | 0.49 | 0.01 |
| high | 200 | 0.80 | 0.74 | 0.06 |

ECE = (300 * 0.03 + 500 * 0.01 + 200 * 0.06) / 1000 = (9 + 5 + 12) / 1000 = 0.026,
which reads as "off by 2.6 percentage points on a typical play". We use 10 equal-width
bins for ECE and 20 for the plotted curve.

ECE has a weakness you should be able to state: it depends on the binning, and it is
noisy when games are few, because all plays of a game move together.

### Platt scaling versus isotonic regression
Both take the model's output and learn a correction on data the trees have not seen.

- **Platt scaling** fits two numbers, a and b, and outputs sigmoid(a * z + b), where z
  is the tree score (log-odds). a < 1 pulls every prediction toward 50% (the model was
  overconfident), a > 1 pushes them away (it was underconfident), and b shifts them all.
- **Isotonic regression** fits any non-decreasing step-like function. It can fix any
  shape of miscalibration, but it has as many free values as it likes.

Worked example (by hand). The trees say z = 2.0, which is p = 0.881. With the slope we
actually fitted, a = 0.867 and b = 0.004: a * z + b = 1.738, so p = 0.850. A prediction
of 0.5 (z = 0) barely moves: sigmoid(0.004) = 0.501.

### Why split by season
If plays were split at random, plays from the same game would land in both the training
and the test set. They share one label, so the model could score well by recognizing
the game (this spread, this score sequence) and not by understanding football. That is
leakage. Splitting by whole seasons keeps every game on one side, and also tests the
real use: predicting seasons the model has never seen.

### Monotone constraints
Some things are true by the logic of the game: with everything else fixed, more points
is never worse. Data is noisy, and an unconstrained tree model will happily learn that
being up 11 is worse than being up 10 in some corner of the state space, because of
which few games happened to land there. A monotone constraint forbids that. LightGBM
enforces it while growing each tree: a split on a constrained feature is only allowed
if the "higher" side predicts at least as much as the "lower" side.

### Cross-fitting
The blown-lead analysis needs a WP for every play of every season, and it must be an
honest forecast: a model that trained on a game already "knows" who won it and will be
too confident on it. Cross-fitting: cut the seasons into folds, and predict each fold
with a model fitted only on the other folds. Every play gets a prediction from a model
that never saw its game.

## 2. Design decisions and why

**The model.** LightGBM, binary log loss, 300 trees, learning rate 0.06, 15 leaves,
at least 10,000 plays per leaf, 80% row and column subsampling, seed 20261117.

**Features (17).** The raw state: down, ydstogo, yardline_100, game_seconds_remaining,
half_seconds_remaining, qtr, score_differential, both timeout counts, posteam_spread,
receive_2h_ko, is_home. Then engineered ones, all computed from the raw state so that
hypothetical states can be scored:
- `goal_to_go`: ydstogo >= yardline_100 on a scrimmage down. A tree cannot easily learn
  a diagonal rule from two columns, so we hand it over.
- `is_overtime`: qtr is 5 or more.
- `spread_time` = spread * exp(-4 * share of regulation played). The spread matters a
  lot at kickoff and little late. This is the form nflfastR uses.
- `expected_margin` = score_differential + spread * (share of regulation remaining).
  The lead now, plus the part of the spread the favorite has not yet had time to earn.
- `margin_per_sqrt_time` = expected_margin / sqrt((seconds remaining + 30) / 3600).
  The score behaves roughly like a random walk, and a random walk's spread grows with
  the square root of time. So this is "how many standard deviations of remaining
  randomness is the expected lead". The 30 seconds stop a division by zero.

  Worked example (by hand). Up 3, favored by 4 before the game, 15:00 left (900 s).
  Remaining share = 0.25. expected_margin = 3 + 4 * 0.25 = 4.0.
  margin_per_sqrt_time = 4.0 / sqrt(930 / 3600) = 4.0 / 0.508 = 7.87.
  Same lead with 2:00 left: (3 + 4 * 0.033) / sqrt(150 / 3600) = 3.13 / 0.204 = 15.3.
  The same 3 points are worth about twice as much because less can still happen.

  These two features carry 93% of the model's total gain (0.58 and 0.35). In effect the
  trees learned a classic formula (lead over the square root of time) and use the other
  15 features for corrections.

**Monotone constraints.** Increasing in score_differential, posteam_spread,
spread_time, expected_margin, margin_per_sqrt_time and posteam timeouts. Decreasing in
defteam timeouts. The three engineered features each rise when the score or the spread
rises, so constraining all of them makes the whole model monotone in the raw score and
the raw spread. If only the raw columns were constrained, the engineered ones could
undo it. Field position is deliberately not constrained: it is not strictly true (a
team killing the clock does not care), and on kickoff rows `yardline_100` is the
kicking spot, which means something different.

**Rows with no down (kickoffs, extra points, two-point tries).** Kept, with down coded
as 0. Reason: the blown-lead analysis follows WP through a whole game, and the moments
right after a score are where WP jumps. Dropping them would leave holes exactly there.
On the test seasons these rows score about the same as scrimmage plays (log loss 0.449
for both). Rows with no field position or no clock (end of quarter, timeouts, about 1.9%
of rows) cannot be scored: `predict` returns NaN for them and the saved predictions are
null.

**Overtime.** Kept. `game_seconds_remaining` restarts in overtime in the source data,
so qtr = 5 tells the model which clock it is reading. There are only 486 overtime plays
in the test seasons, so nothing reliable can be said about overtime quality.

**Tied games.** Dropped from fitting and scoring (no winner to predict).

**The split.** Trees on 2010 to 2021 (531,244 plays, 3,224 games). Calibration map on
2022 to 2023 (91,663 plays). Test on 2024 to 2025, used once for the reported numbers.

**Hyperparameters.** Chosen inside the training seasons only: fit on 2010 to 2018,
score on 2019 to 2021 (`results/m1_wp_tuning.json`). The important finding is the leaf
size. With at least 400 plays per leaf, validation log loss was 0.4318. With at least
10,000 it was 0.4283. With 20,000 it was 0.4291. Small leaves let the trees isolate
individual games (a leaf of 400 plays can be three games) and memorize their results.
Forcing each leaf to span many games is the main guard against that. 31 leaves scored
0.4282 against 0.4283 for 15, which is a tie, so we kept the smaller trees. 300 trees
beat 150 and 600 by less than 0.0004.

**Calibration map: Platt.** Three reasons, in order of importance.
1. The decision engine subtracts WPs of nearby states. Isotonic regression is flat in
   places, so two different states can get exactly the same WP and their difference
   vanishes. Platt is smooth and strictly increasing, so it never reorders or merges.
2. Two parameters. The calibration seasons hold 569 games, and games are the real
   sample size. A flexible map fitted to that will chase noise.
3. Evidence from the calibration seasons alone (five folds of games, map fitted on four
   and scored on the fifth): log loss 0.4843 for Platt, 0.4856 for no map, 0.4863 for
   isotonic. Isotonic had the best ECE there (0.005 against 0.015), but worse log loss.
Both maps are non-decreasing, so the monotone guarantees survive calibration.

**`fit(train, calib=None)`.** With no calibration frame the model returns the trees' own
probabilities. One fit of the headline model takes about 8.5 seconds on 2 threads, and
about 3.3 seconds with `n_estimators=100`, so 100 bootstrap refits are feasible.

**Out-of-fold predictions.** Four folds of four consecutive seasons. For each fold the
other 12 seasons are split into 10 for the trees and the most recent 2 for the Platt
map. Saved to `data/processed/wp_predictions.parquet` (729,962 rows, 716,459 with a
prediction). Note that early seasons are predicted by models trained on later seasons.
That is fine for a calibration study, and it would not be fine for a claim like "this
is what a forecaster could have known in 2011".

## 3. What the results showed

Test seasons 2024 to 2025, 90,931 plays in 569 games, all models on the same rows.

| model | log loss | Brier | ECE |
|---|---|---|---|
| Ours (trees + Platt) | 0.4490 | 0.1484 | 0.0217 |
| Ours, trees only (no Platt) | 0.4469 | 0.1478 | 0.0084 |
| nflfastR `wp` (no spread) | 0.4845 | 0.1642 | 0.0123 |
| nflfastR `vegas_wp` (with spread) | 0.4466 | 0.1478 | 0.0095 |

Read this honestly:

1. **We beat nflfastR `wp` clearly, but that is not a fair fight.** Our model sees the
   point spread and `wp` does not. Log loss difference -0.0355, with a 95% interval of
   -0.0525 to -0.0192 from resampling games.
2. **Against the fair comparison, `vegas_wp`, we are slightly worse.** Log loss +0.0025,
   interval -0.0018 to +0.0064. The interval includes zero, so the data cannot tell the
   two apart, but we did not beat it. On ECE we are worse than both nflfastR models
   (2.2 points against 1.0 and 1.2).
3. **The calibration step made our model worse on the test seasons.** ECE went from
   0.0084 to 0.0217 and log loss from 0.4469 to 0.4490. This is the most instructive
   result of the milestone, so understand it.

**Why calibration backfired.** The Platt slope fitted on 2022 to 2023 was 0.867, which
shrinks every prediction toward 50%. The 2022 season was unusually unpredictable (every
model's log loss that year is above 0.51, against about 0.40 to 0.48 in other years),
so on those seasons the trees looked overconfident. In 2024 to 2025 they were not, and
the shrinkage made the model underconfident: on the reliability figure our curve sits
above the diagonal for predictions over 0.6 and below it under 0.4.

The script measures how unstable this is, using only the non-test seasons
(`calibration_stability` in the metrics file). A Platt slope fitted to one season
ranges from 0.79 (2022) to 1.22 (2017). Fitted to pairs of consecutive seasons it
ranges from 0.87 to 1.13, and 2022 to 2023 is the lowest pair of all. Pooled over all
14 non-test seasons it is 0.963, close to "leave the model alone".

The lesson is the one the Wyner group's papers make: the sample size for anything about
WP is the number of games, not plays. 91,663 calibration plays sounds like plenty, but
it is 569 games, and two seasons can simply be unusual together. A two-season
calibration set is too small to estimate even two parameters well.

**A post hoc check, labelled as such.** Applying the pooled slope (0.963) to the
headline trees gives test log loss 0.4471 and ECE 0.0106. This was computed after the
headline result was known, so it is a diagnostic and not the reported model. It
suggests that calibrating on cross-fitted scores from all 14 seasons would be the
better design. Whether to switch is recorded as an open decision (see the end).

**By quarter.** Log loss falls from 0.58 in the first quarter to 0.30 in the fourth for
both spread-aware models: the outcome is easier to call late. We match `vegas_wp`
in the first half and lose a little in the second half (0.304 against 0.298 in the
fourth quarter).

**Out-of-fold, all 16 seasons** (713,837 plays, 4,350 games): ours 0.4496 log loss and
ECE 0.0079, `vegas_wp` 0.4451 and 0.0034, `wp` 0.4820 and 0.0059. `vegas_wp` is ahead in
14 of 16 seasons. One caveat may flatter the nflfastR numbers: we do not know which seasons
nflfastR's models were fitted on. Where they include a season in this table, the
nflfastR number is in-sample and ours is not. We could not verify this either way.

**Sanity checks** (all pass on the headline model):
- WP never falls when score differential or spread rises, over a grid of states.
- First and 10, tied: the opponent 20 is worth 0.18 more WP than your own 20, and the
  curve along the field has no reversal larger than 0.007.
- Up 30 with two minutes left: 0.990. Down 30: 0.006.
- Calibration by side: turning each prediction into the home team's WP, the average is
  within 1.4 points of the home team's actual win rate whether the home team has the
  ball or not.
- Kickoff mirror. The model only speaks for the team with the ball, so a state cannot
  be flipped directly. Take one matchup where the home team is favored by s. If the home
  team receives the opening kickoff its WP is model(spread s, home). If the away team
  receives, its WP is model(spread -s, away). Those are the two opponents of one
  matchup, so the two numbers should sum to about 1. They sum to 0.974 on average, and
  the worst case over spreads from -10 to 10 is 0.063 away from 1. The sum is a bit
  under 1 because receivers of the opening kickoff won only 47.6% of the 3,209 training
  games (the other team gets the ball to start the second half).

## 4. Known weaknesses

- **Platt on two seasons is noisy**, as shown above. The headline model is somewhat
  underconfident on 2024 to 2025.
- **`is_home` near even spreads.** At a spread of 0 the model gives the away team a
  higher WP than the home team at the opening kickoff (0.545 against 0.499). The spread
  already contains home-field advantage, so this gap should be about zero. It is noise
  from the few hundred near-even games in the training data, absorbed by the trees. It
  shifts both options of a fourth-down decision almost equally, so it matters little
  for decisions, but it is wrong as a level.
- **The model is a step function.** Trees give piecewise-constant output. Along the
  spread axis at the kickoff, WP is almost flat from 7 to 9 points and then jumps. Small
  differences between nearby states can be exactly zero or oddly large.
- **No ablation without the spread** was run, so we cannot say how our model compares
  with nflfastR `wp` on equal information.
- **Eras are pooled.** Rule changes (kickoff spot, overtime rules) are not modelled.

## 5. Questions you should be able to answer

1. Why is 91,663 calibration plays a small sample? What is the real sample size?
2. The Platt slope was 0.867. What does a slope below 1 do to a prediction of 0.9?
3. Why did we raise the minimum leaf size to 10,000 plays, and what was going wrong at 400?
4. Why must the engineered features be constrained too, and not just score_differential?
5. Why Platt and not isotonic, given isotonic had the lower ECE on the calibration seasons?
6. Our model beats `wp` by 0.0355 log loss. Why is that not evidence that it is a better model?
7. Compute by hand the log loss of predicting 0.8 for a team that lost.
8. Why can the blown-lead analysis not use predictions from a model that trained on those games?
9. The kickoff mirror sums to 0.974, not 1. Is that a bug?

## 6. Version 2: fixing the end game

### 6.1 What was wrong

Using the model in the decision engine (M2) exposed a fault. On first downs of the 2024
and 2025 seasons, teams leading by 1 to 3 with 41 to 120 seconds left won 95.9% of the
time and version 1 said 86.2%. Teams trailing by 4 to 8 with 40 seconds or less won 8.1%
and it said 16.6%. Up 1 with the ball at the own 25 and 13 seconds left, it said 0.72.
Anyone who has watched football knows that last team has all but won: it kneels.

Four suspects were named in advance: the 10,000-play minimum leaf, the Platt map fitted
on two seasons (slope 0.867, which shrinks everything toward 50%), a 30-second floor in
`margin_per_sqrt_time`, and timeouts that the trees almost never used (25 of about
4,150 splits). A fifth was a side issue: `is_home` gave the two teams of an even game
different win probabilities at the opening kickoff.

### 6.2 The honesty problem, and how it was handled

The 2024 and 2025 seasons had now been looked at several times, and they are where the
fault was seen. Any fix tuned until it looks good on those seasons would be tuned on the
test set. So the rule for version 2 was:

1. Every choice is made on the development seasons (2010 to 2023) only, loaded through
   `splits.load_dev_plays()`, with season-blocked cross-validation: seven folds of two
   consecutive seasons, each predicted by a model fitted on the other twelve. A season
   is never split, so a game is never on both sides.
2. The diagnosis is redone on development seasons first. If the fault did not show
   there, there would be nothing to fix without peeking.
3. The chosen configuration, the development evidence for each choice and the
   evaluation plan are written to `results/m1_wp_v2_preregistration.json` before any
   2024 to 2025 number for version 2 exists. The test seasons are then scored once
   (`scripts/fit_wp.py --evaluate`, which refuses to run if the configuration or the
   saved model differ from what was preregistered), and the result is reported as it
   came out.
4. The 2026 season is the final holdout and was not touched.

The 2024 to 2025 numbers below are still not those of an untouched test set: those
seasons informed the diagnosis. They are reported, labelled that way, next to version 1.

### 6.3 The diagnosis on development seasons

Version 1, refitted exactly as designed inside each fold, shows the same fault on
development seasons (study candidate `v1`, first downs, last 5:00):

| situation | plays | won | version 1 says |
|---|---|---|---|
| lead 1 to 3, 0:41 to 2:00 | 367 | 95.4% | 87.4% |
| lead 1 to 3, 0:40 or less | 189 | 100.0% | 89.3% |
| trail 9 or more, 0:40 or less | 830 | 0.0% | 2.5% |
| trail 4 to 8, 0:40 or less | 736 | 16.8% | 15.4% |

The first three are clear misses with intervals far from zero. The fourth, which was
the second headline symptom on the test seasons, is fine on development seasons. That
matters later.

Taking version 1 apart one piece at a time:
- Removing the Platt map helped everywhere: overall log loss 0.45104 to 0.44973, last
  two minutes 0.2480 to 0.2419. The two-season map is simply worse than no map.
- Smaller leaves helped the end game most: 2,000-play leaves cut the average end-game
  miss from 1.4 to 0.9 points.
- The memorization gap (held-out log loss minus in-sample log loss) of version 1 is
  0.0126, of which about 0.005 is the calibration map; with smaller leaves it rose only
  slightly. So the 10,000-play leaf was buying less protection than it cost.

### 6.4 What was tried

Every candidate is in `results/m1_wp_v2_dev_study.json`, with why it was tried. Nothing
was removed. Numbers are season-blocked cross-validation on 2010 to 2023 (622,907 plays).
"End-game miss" is the play-weighted average of |predicted minus observed| over 21
cells: three time bins under 5:00 (0:00 to 0:40, 0:41 to 2:00, 2:01 to 5:00) by seven
score bins for the team with the ball (trail 9+, 4 to 8, 1 to 3, tied, lead 1 to 3, 4 to
8, 9+), scrimmage downs.

| candidate | log loss | last 2:00 | end-game miss | memorization gap |
|---|---|---|---|---|
| v1 (as shipped) | 0.45104 | 0.2480 | 1.62 pts | 0.0126 |
| v1 without Platt | 0.44973 | 0.2419 | 1.38 | 0.0075 |
| v1, 2,000-play leaves | 0.44965 | 0.2370 | 0.87 | 0.0093 |
| baseline + v1 trees | 0.44913 | 0.2482 | 1.58 | 0.0051 |
| baseline, no is_home | 0.44930 | 0.2480 | 1.59 | 0.0044 |
| + kneel_slack | 0.44932 | 0.2468 | 1.89 | 0.0043 |
| + kneel_slack, 2,000-play leaves | 0.44877 | 0.2348 | 0.79 | 0.0057 |
| + possessions, 2,000 | 0.44863 | 0.2341 | 0.51 | 0.0057 |
| + timeout shares, 2,000 | 0.44888 | 0.2350 | 0.77 | 0.0058 |
| two-stage (end-game trees, last 10:00) | 0.44869 | 0.2333 | 0.51 | 0.0065 |
| **chosen: + kneel rule in the baseline** | **0.44863** | **0.2338** | **0.48** | **0.0057** |

Leaf sizes 5,000, 1,000, 500 and 200, a 5-second time floor, keeping `is_home`, a fitted
value of possession in the baseline, and a baseline with its clock offset fixed at 60,
120 or 240 seconds were also tried; see the study file and `configs/wp.yaml`
(`provenance` and `decisions`), which give the comparison behind every choice.

### 6.5 The chosen design, piece by piece

**1. A structural baseline.** Before any tree, the model starts from the classic normal
approximation. Treat the rest of the game as a random walk in the score. If the
expected final margin is m and the remaining randomness has standard deviation s, the
chance of finishing ahead is Phi(m / s), where Phi is the standard normal cumulative
distribution. Randomness grows with the square root of time, so

    WP_baseline = Phi( expected_margin / (sigma * sqrt((seconds_left + tau) / 3600)) )

`expected_margin` is the lead plus the share of the spread still to be earned (section
2). Two numbers are fitted by maximum likelihood on the training rows: sigma, the
standard deviation of the margin over a full game, and tau, seconds added to the clock.
For the headline model sigma = 13.8 points and tau = 518 seconds.

The baseline goes to LightGBM as `init_score`: the trees start from its log-odds and
learn corrections to it, on the log-odds scale. Think of it as boosting with the first
"tree" replaced by a two-parameter formula. Why it helps: a tree has to learn a smooth
curve as a staircase, and it needs many rows on each step. The formula already has the
right shape in the score, the spread and the clock, so the trees only learn what the
formula misses (field position, down and distance, timeouts, the end game). The model
is now smooth in the spread (no flat steps in a sweep from -14 to +14, against 36% flat
steps for version 1) and in the clock (2% flat steps in the last 5:00 up 3, against
84%). Removing the baseline from the design one step before the final one
(`scores_leaf2000` against `scores_no_base_leaf2000`) costs 0.0008 of log loss, with a
game-bootstrap interval from -0.0015 to -0.0002 that excludes zero.

Worked example (by hand). Tied, favored by 3, kickoff: expected margin 3, s = 13.8 *
sqrt((3600 + 518) / 3600) = 14.8, z = 0.20, WP = 0.58. Up 7 with 5:00 left and an even
spread: s = 13.8 * sqrt(818 / 3600) = 6.6, z = 1.06, WP = 0.86.

**Why tau is so large, and what that costs.** At 0:00 the baseline still has 13.8 *
sqrt(518 / 3600) = 5.2 points of uncertainty. That sounds wrong, but the rows decide
it: in the last seconds the team with the ball can still score, and those rows are the
most common late rows (trailing teams run many plays). The formula has no notion of
possession, so a single tau has to serve both the leader with the ball (nearly certain)
and the trailer with the ball (about 35% when down 1 to 3). Maximum likelihood picks the
compromise that the plentiful trailing rows want. Fixing tau smaller made things worse
(the trees then over-corrected the leader states), and a fitted value of possession did
not help once the trees were there.

**2. A rule of the game in the baseline: the kneel-down.** A kneel takes about 2
seconds and lets the clock run, so the next snap can wait 40 more unless the defense
calls timeout. With k kneels available (3 on first down, 2 on second, 1 on third) and
the defense holding n timeouts, the leader can burn T = 40 * max(k - n, 0) + 2 * k
seconds, at most 120 because the two-minute warning stops the clock once
(`docs/research/endgame_clock.md`). On first down that is 120, 86, 46 and 6 seconds for
0 to 3 defensive timeouts.

`kneel_slack` = T minus the seconds left, for the team with the ball and a lead, on a
scrimmage down in regulation (a floor value of -600 everywhere else). Zero or more means
the game can be ended by kneeling. On development seasons those rows were won 3,313
times out of 3,314. So where `kneel_slack` is zero or more, the baseline says the game
is won (log-odds 9, about 1 in 8,000). This is a fact about the rules, applied before
any fitting, not a pattern learned from outcomes. It also makes the defense's timeouts
matter: they change T.

Worked example (by hand). Up 2, first down, 70 seconds left. The defense has 1 timeout:
T = 40 * 2 + 6 = 86, slack = 16, game over. The defense has 2: T = 40 + 6 = 46, slack =
-24; the leader must still run real plays or punt.

**3. Two more end-game features.**
- `possessions` = the deficit or lead in scores of up to 8 points, signed: down 9 is -2
  because one touchdown and conversion cannot catch up.
- `possessions_per_minute` = possessions / ((seconds left + 30) / 60). Down two scores
  with a minute left is very different from down two scores with ten.

Timeout shares (the share of the half's clock each side's timeouts could stop) were
tried and not kept: no gain and a fit three times slower.

**4. `is_home` dropped.** Home-field advantage is already in the point spread. Keeping
`is_home` is worth 0.00014 of log loss on development seasons, with an interval from
-0.0008 to +0.0010, so the data cannot see it. Dropping it makes the two sides of a game
symmetric by construction: at an even spread the receiving team now has 0.501 whether
it is home or away (version 1: 0.499 and 0.545).

**5. Leaves of at least 2,000 plays, 15 leaves, 300 trees at 0.06.** From the leaf sweep
and the tuning grid (`results/m1_wp_tuning.json`). 2,000 is not the overall winner of
the grid: 5,000 scored 0.44852 against 0.44863, a difference of 0.0001, but its last
two minutes are clearly worse (0.2362 against 0.2338). Version 2 exists for the end
game, so 2,000 was chosen and the reason is recorded in `configs/wp.yaml`. The
memorization gap is 0.0057, less than half of version 1's 0.0126: the baseline does the
work that large leaves used to do, so smaller leaves no longer memorize games.

**6. No calibration map.** Every development season was predicted by a model that did
not see it, then a Platt map was fitted to those predictions, and each season was mapped
by a map fitted on the other seasons only. The pooled slope is 0.934 (single seasons
range from 0.75 to 1.20), so the raw model is slightly overconfident in the middle. The
map would gain 0.0003 of overall log loss (0.44863 to 0.44833) and cut ECE from 0.008 to
0.003, but it makes the end game worse (last two minutes 0.2338 to 0.2344, end-game miss
0.48 to 0.57 points). The end game is what version 2 is for, so there is no map, and the
mild mid-range overconfidence is a recorded weakness. A side benefit: the model is one
fit, with no calibration seasons, which simplifies the M3 bootstrap.

**7. No separate end-game model.** A second stage of small-leaf trees on the last 5 to
15 minutes improved the last two minutes by about 0.001 but did not improve the
end-game cells, and it is a second model to explain.

**Hyperparameters and constants.** Every value and how it was chosen (a comparison, or a
stated reason when it was not tuned) is in `configs/wp.yaml` under `provenance`, and is
copied into the preregistration and the metrics file. This answers audit finding M4.
`SPREAD_DECAY = 4` is one value that was never compared with anything, in either
version; it is stated as such.

**Fit time.** 10.6 seconds on 2 threads for the headline fit (622,907 plays), 6.1 seconds
with 100 trees, so the 101 bootstrap refits stay feasible.

### 6.6 Results on development seasons (cross-fitted)

| | plays | version 2 | version 1 | difference (95% interval, games resampled) |
|---|---|---|---|---|
| log loss, all plays | 622,907 | 0.4486 | 0.4510 | -0.0024 (-0.0044 to -0.0005) |
| log loss, last 5:00 | 63,102 | 0.2454 | 0.2550 | -0.0096 (-0.0113 to -0.0077) |
| log loss, last 2:00 | 28,918 | 0.2338 | 0.2480 | -0.0142 (-0.0170 to -0.0114) |
| ECE, all plays | | 0.0080 | 0.0083 | |
| end-game miss (21 cells) | | 0.48 pts | 1.62 pts | |

The diagnosis cells: lead 1 to 3 with 0:41 to 2:00, 95.4% won, version 2 says 96.1%;
lead 1 to 3 with 0:40 or less, 100% won, version 2 says 99.8%. Trailing by 4 to 8 with
0:40 or less, 16.8% won, version 2 says 15.4%, the same as version 1.

### 6.7 Results on the test seasons, scored once

Preregistered at 2026-11-17 07:25 UTC; evaluated at 07:33 UTC, one load of the test
plays, recorded in the access log. Version 1 was refitted by the new code and reproduced
its shipped numbers exactly. Same 90,931 plays in 569 games for every model.

| model | log loss | Brier | ECE |
|---|---|---|---|
| Ours, version 2 | 0.4450 | 0.1476 | 0.0085 |
| Ours, version 1 | 0.4490 | 0.1484 | 0.0217 |
| nflfastR `vegas_wp` (with spread) | 0.4466 | 0.1478 | 0.0095 |
| nflfastR `wp` (no spread) | 0.4845 | 0.1642 | 0.0123 |

- Version 2 minus version 1: -0.0041 log loss, interval -0.0080 to +0.0001. Better, but
  the interval just touches zero, so on all plays the improvement is not established by
  these two seasons alone. About half of the overall gain is the two-season Platt map
  going away: version 1's own trees without the map scored 0.4469 on the same rows
  (`results/m1_wp_metrics_v1.json`, `ours_uncalibrated`).
- Version 2 minus `vegas_wp`: -0.0016, interval -0.0051 to +0.0019. Tied, now slightly
  ahead rather than slightly behind.
- Last 5:00: 0.2463 against 0.2571 for version 1 and 0.2477 for `vegas_wp`. Last 2:00:
  0.2372 against 0.2523 and 0.2386.
- End-game miss over the 21 cells: 2.7 points against 3.2 for version 1. The test cells
  are small (65 to 1,689 plays), so most intervals are wide.

The three cells that exposed version 1 (first downs):

| situation | plays | won | version 1 | version 2 |
|---|---|---|---|---|
| lead 1 to 3, 0:41 to 2:00 | 73 | 95.9% | 86.2% | 94.9% |
| lead 1 to 3, 0:40 or less | 18 | 100.0% | 89.0% | 100.0% |
| trail 4 to 8, 0:40 or less | 74 | 8.1% | 16.6% | 16.6% |

The leader's side is fixed. The trailing-by-4-to-8 cell is not: version 2 says the same
as version 1. On development seasons that cell was well calibrated (16.8% won, 15.4%
predicted), so either 2024 and 2025 differ there (74 plays in a few dozen games) or the
model misses something those seasons have. Nothing was changed after seeing it. One new
miss appears: trailing by 1 to 3 with 0:40 or less, all downs, 44.1% won and version 2
says 33.1% (interval excludes zero; version 1 said 39.7%). The figure
`assets/m1_wp_endgame.png` shows every cell for both versions, development seasons on
the left and test seasons on the right.

### 6.8 Sanity checks

The five original checks pass (monotone in score and spread, field position, blowout,
kickoff mirror, now with a worst deviation of 0.041 against 0.063), and so does
calibration by side on cross-fitted development seasons. Four end-game checks were
added, with thresholds set before version 2 was chosen:

| check | bar | version 1 | version 2 |
|---|---|---|---|
| lead 1 to 3, first down, defense out of timeouts, under 1:00 (worst state) | above 0.95 | 0.669 | 1.000, passes |
| down 9 or more, under 0:30 (worst state) | below 0.02 | 0.113 | 0.063, fails |
| up 3, own 40, 1:20, defense timeouts 0 against 3 | WP falls by more than 0.01 | 0.002, fails | 0.069, passes |
| up 1, own 25, 13 seconds, all timeouts | above 0.90 | 0.717 | 0.876, fails |

Two still fail and are reported as failing; their thresholds are not changed, and the
tests mark them as strict expected failures, so a future fix has to remove the mark.
Both failures come from the soft baseline at 0:00 (tau, section 6.5): the 13-second state
is not a kneel-out (three defensive timeouts leave 7 seconds), and down 9 at the
opponent's 20 with 29 seconds is not covered by any rule, so the trees must correct a
starting point of about 0.04 from rare rows. In a fit on 2010 to 2021 only (used in a
dry run of the pipeline), the field-position check also failed narrowly, so that check
is close to its limit.

Timeouts are used more: 53 and 29 splits for the two timeout counts and 58 for
`kneel_slack`, out of 4,184 (version 1: 25 in all).

### 6.9 Leakage guards added

- `WPModel.fit` and `NNWPModel.fit` call `splits.assert_no_test_seasons` on the training
  and calibration frames and refuse frames without `season` and `game_id`. The only way
  past is `allow_test_seasons=True` with a written `reason`, which is logged. The
  out-of-fold predictions no longer need it: test seasons are in no fold's training set
  (they are predicted by the model fitted on all development seasons, which is the
  headline model itself).
- Every fit stores `splits.fit_stamp(...)` as `model.stamp` (seasons, games, rows, which
  seasons trained and which calibrated, and whether an exception was used), and `save`
  and `load` keep it. `splits.assert_can_score(model.stamp, rows, what)` is called before
  every out-of-sample number: each cross-fitted fold, the development CV, the test
  evaluation, and the neural model script.
- `scripts/fit_wp.py --tune` reads only development plays and exits. The development
  step and `--preregister` read no test row. `--evaluate` is the only step that loads
  the test seasons, through `splits.load_test_plays(purpose=...)`.
- `results/m1_wp_metrics.json` records the fold plan of the out-of-fold predictions:
  which seasons trained the model that predicted each season.

### 6.10 Known weaknesses of version 2

- **The baseline is soft at 0:00** (tau = 518 seconds), so states that no rule covers
  and the data rarely show (a big deficit near the opponent's goal with seconds left; a
  narrow lead the defense can still stop the clock against) are corrected only partly by
  the trees. Two sanity checks fail because of this.
- **Mild mid-range overconfidence** (pooled slope 0.934 on cross-fitted development
  seasons), left in on purpose, see 6.5.
- **The trailing-by-4-to-8 cell in the last 40 seconds** is still 8 points high on the
  test seasons.
- **Steps along the field.** Field position still comes only from the trees: 21% of
  one-yard steps are flat, with a largest single step of 3.2 points.
- **The kneel rule is a step.** WP jumps where `kneel_slack` crosses zero. That is real
  (one second can decide whether the defense gets the ball back), but the decision
  engine subtracts nearby states, so differences near the threshold should be read with
  that in mind.

### 6.11 Questions on version 2

1. Why could the test seasons not be used to choose between candidate fixes, and what
   did we do instead?
2. What does `init_score` do in LightGBM, and why does a structural baseline let us use
   smaller leaves without memorizing games?
3. Up 3, first down, 1:00 left, defense with 2 timeouts. Compute T and `kneel_slack`. Is
   the baseline certain?
4. Why is the fitted tau about 500 seconds, and what goes wrong at the end of a game
   because of it?
5. The pooled Platt map improves overall log loss. Why was it not used?
6. Version 2 minus version 1 on the test seasons is -0.0041 with an interval from
   -0.0080 to +0.0001. What can and cannot be claimed?
7. Why was `is_home` dropped although keeping it scored slightly better?
8. Two sanity checks fail. Why not lower their thresholds?
