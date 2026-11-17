# M1 notes: the win-probability model

Code: `src/fourth_and_rl/wp.py`. Config: `configs/wp.yaml`. Script: `scripts/fit_wp.py`.
Every number below is from `results/m1_wp_metrics.json` or `results/m1_wp_tuning.json`,
both written by that script. Worked examples marked "by hand" are arithmetic, not results.

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
