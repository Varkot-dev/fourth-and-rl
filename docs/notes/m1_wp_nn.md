# M1 extension notes: a neural win-probability model and a deep ensemble

Code: `src/fourth_and_rl/wp_nn.py`. Config: `configs/wp_nn.yaml`. Script:
`scripts/fit_wp_nn.py`. Tests: `tests/test_wp_nn.py`.
Every number below is from `results/m1_wp_nn.json` or `results/m1_wp_nn_tuning.json`,
both written by that script. Worked examples marked "by hand" are arithmetic, not results.
Read `docs/notes/m1_wp.md` first: this note assumes log loss, Brier score, ECE, Platt
scaling and the season split from there.

The short version of the results, so you know where this is going:

- A network with one hidden layer of 32 units (737 numbers in total) matches the
  300-tree model on the test seasons. It is not measurably better. Log loss 0.4461
  against 0.4469 for the trees without their Platt map, difference -0.0008 with a 95%
  interval of -0.0036 to +0.0020.
- Its curves are smooth where the trees are a staircase, but it has two defects of its
  own that the trees do not have.
- Averaging ten networks does not improve log loss in any measurable way.
- The spread of a seed-only ensemble is about 0.64 of the spread of a game bootstrap.
  It is too narrow to be used as the uncertainty of a win probability.

## 1. Concepts

### What an MLP is
A multilayer perceptron (MLP) is a function built from two kinds of step, repeated:

1. A linear step: multiply the inputs by weights, add them up, add a bias. One such sum
   is computed for each "hidden unit".
2. A nonlinear step: pass each sum through a simple fixed function. We use ReLU, which
   is relu(a) = max(0, a): negative sums become zero, positive ones pass through.

Our network does this once (17 game-state inputs, expanded to 21 network inputs, into 32
hidden units), and then takes one more weighted sum of the 32 hidden values to produce a
single number z. As with the trees, z is a log-odds, and the win probability is
sigmoid(z) = 1 / (1 + exp(-z)).

Why the nonlinear step matters: without it, a weighted sum of weighted sums is just
another weighted sum, and the whole network would be plain logistic regression. With
ReLU, each hidden unit is a hinge: flat on one side of a line through input space,
rising on the other. The output is a sum of 32 hinges, which is a surface made of flat
facets joined without gaps. That is the key structural difference from a tree model,
whose surface is made of flat boxes with jumps between them.

Counting the parameters (by hand): 21 inputs times 32 units = 672 weights, plus 32
biases, plus 32 output weights, plus 1 output bias = 737.

### How it is trained
Training means finding the 737 numbers that make the log loss on the training plays
small. The loop in `NNWPModel.fit` does four things for each batch of plays.

1. **Forward pass.** Run the batch through the network to get a log-odds z for each
   play, hence a probability p.
2. **Loss.** Compute the mean log loss of the batch, -[y ln(p) + (1 - y) ln(1 - p)],
   the same loss the trees minimize.
3. **Backpropagation.** Compute, for every weight, how much the loss would change if
   that weight were nudged up (its gradient). This is the chain rule from calculus,
   applied from the output backwards through the layers, so that each layer reuses the
   work done for the layer after it. PyTorch does it for us in `loss.backward()`.
4. **Optimizer step.** Move every weight a small step in the direction that lowers the
   loss. Plain gradient descent is: weight = weight - learning_rate * gradient. We use
   Adam (in its AdamW form), which keeps a running average of each weight's recent
   gradients and of their squares and scales each weight's step by them, so that weights
   with consistently small gradients still make progress.

One pass over all the training plays is an epoch. With 531,244 training plays and
batches of 4,096, an epoch is 130 steps.

### A worked example (by hand)
A network with 2 inputs, 2 hidden units and 1 output, and one play that the team won
(y = 1). Inputs: x1 = 1.0, x2 = -0.5.

Weights before the step:
- hidden unit 1: weights 0.8 and 0.2, bias 0.1
- hidden unit 2: weights -0.5 and 0.4, bias 0.0
- output: weights 1.5 and -1.0, bias 0.2

Forward pass:
- unit 1 sum = 0.8 * 1.0 + 0.2 * (-0.5) + 0.1 = 0.8, so h1 = relu(0.8) = 0.8
- unit 2 sum = -0.5 * 1.0 + 0.4 * (-0.5) + 0.0 = -0.7, so h2 = relu(-0.7) = 0
- z = 1.5 * 0.8 + (-1.0) * 0 + 0.2 = 1.4, so p = sigmoid(1.4) = 0.802

Loss: -ln(0.802) = 0.221.

Backpropagation. A convenient fact: for the log loss with a sigmoid, the gradient of
the loss with respect to z is simply p - y = 0.802 - 1 = -0.198. Everything else is the
chain rule.
- output weight on h1: (p - y) * h1 = -0.198 * 0.8 = -0.158
- output weight on h2: (p - y) * h2 = 0, because h2 was 0
- output bias: p - y = -0.198
- signal reaching unit 1: (p - y) * 1.5 = -0.297. Its ReLU was active, so it passes.
  - weight on x1: -0.297 * 1.0 = -0.297
  - weight on x2: -0.297 * (-0.5) = +0.149
  - bias: -0.297
- unit 2: its ReLU was off (the sum was negative), so all of its gradients are 0. A
  unit that is off for a play learns nothing from that play.

One plain gradient-descent step with learning rate 0.1 (subtract 0.1 times each
gradient): output weight on h1 becomes 1.516, output bias 0.220, unit 1 weights 0.830
and 0.185, unit 1 bias 0.130.

Forward pass again: unit 1 sum = 0.830 - 0.093 + 0.130 = 0.867, z = 1.516 * 0.867 +
0.220 = 1.534, p = 0.823, loss = 0.195. The loss fell from 0.221 to 0.195, and the
prediction moved toward the outcome. Training is this, repeated a few thousand times
on batches of 4,096 plays.

### Standardizing the inputs
Trees only ask "is this feature above a threshold?", so the scale of a feature does not
matter to them. A network multiplies features by weights, so scale matters a great
deal: `game_seconds_remaining` runs to 3,600 and `is_home` to 1, and with raw inputs
the gradient steps for the two would be wildly different sizes. So each input is
standardized: subtract its mean and divide by its standard deviation.

The mean and standard deviation are computed from the training seasons only and then
reused unchanged for every other row. Computing them on all the data would let
information from the test seasons into the model. The leak would be tiny here, but the
rule costs nothing and it is the kind of thing a reviewer checks.

### Why trees usually win on tabular data, and when a network can help
"Tabular data" means a table whose columns each mean something different (down, clock,
score). Trees have three built-in advantages there:

- They do not care about scale, units or outliers.
- Jumps are natural for them. Football has real jumps: a 3-point lead is different in
  kind from a 4-point lead, and a tree can put a split exactly there.
- Interactions are natural for them. "Late in the game AND down by one score" is just
  two splits in a row.

A network has to build all of that out of hinges, and it tends to smooth over real
jumps. What it offers in return is the mirror image: where the truth is smooth, it is
smooth. Win probability should rise steadily with the pregame spread; there is no
reason for it to be flat between 6.5 and 7.5 and then jump. A tree model can only
represent that curve as a staircase. That matters to the decision engine, which
subtracts the win probabilities of nearby states: on a flat step, two different states
get exactly the same value and the difference vanishes.

So the test of whether the network earns its place is not only "is the log loss lower"
(it is rarely lower on a table this simple) but "are the curves better where the engine
uses them".

### Overfitting here means memorizing games
Every play of a game carries the same label. A flexible model can lower its training
loss by learning to recognize a game (this spread, this score path, this clock) and
attaching the result, which teaches it nothing about football. The M1 trees showed
this with small leaves. A network does it with too many weights trained for too long.

The tuning run includes one deliberate demonstration: three layers of 128 units, small
batches, no weight decay, 40 epochs, no early stopping. Its training log loss fell to
0.337, while its log loss on held-out games from the same seasons rose to 0.693 (the
score of a coin flip) and on the validation seasons to 0.654. The network had learned
the games, not the game.

### The guards, and what each one did
All of these were tried on non-test seasons (section 2). Results, in order of how much
they mattered:

- **Early stopping on held-out games.** After every epoch, score the network on games
  it is not being trained on, and keep the weights from the epoch where that score was
  best. Stop when it has not improved for 6 epochs. The held-out set must be whole
  games: a held-out play from a game whose other plays are in training would be scored
  by a network that already knows that game's result, and the warning would never fire.
  This was the guard that mattered most. Without it the two-layer starting network
  went from 0.4286 to 0.4408 on the validation seasons.
- **A small network.** Capacity is the root cause, so the simplest guard is to have
  less of it. One layer of 32 units did as well as anything larger, and with it even
  40 epochs with no early stopping cost little (0.4294 against 0.4279).
- **Weight decay.** Shrink every weight slightly toward zero at each step, which
  favors smoother functions. Values of 0, 0.0001, 0.01 and 0.1 gave the same validation
  log loss to four decimals. It does nothing measurable here.
- **Dropout.** While training, zero a random share of the hidden units on each batch,
  so the network cannot rely on any single unit. No measurable effect at 0.1 or 0.3.
- **Large batches.** A batch of 4,096 plays mixes about 25 plays from each of many
  games, so no single game dominates a step, and there are fewer steps per epoch in
  which to overfit. Batches of 512 were slightly worse and peaked after 2 epochs;
  16,384 was no better and noisier across seeds. 4,096 was kept.
- **Game-level label noise.** Each epoch, flip the result of a random few percent of
  games (all the plays of a game together), so that no single game's label is worth
  memorizing. It made things worse: 0.4308 at 2%, 0.4373 at 5%, 0.4512 at 10%, against
  0.4286 with none. Part of that is by construction, because training on flipped labels
  pulls every prediction toward 50%. But it was still worse after a Platt map that
  undoes the pull (0.4305, 0.4323, 0.4365 against 0.4292). It is not used.

The honest summary: with a network this small and early stopping in place, the other
guards have nothing left to do.

### What a deep ensemble is
Train the same network K times, changing only the random seed. The seed sets the
starting weights and the order in which plays are visited, and because training stops
at a slightly different place each time, the K networks disagree a little. A deep
ensemble predicts with their average, and reads their disagreement (the standard
deviation across members) as a measure of uncertainty.

### What its spread does and does not measure
There are two different reasons a fitted win probability could have come out
differently.

- **Optimization noise.** Same data, different seed. The network lands somewhere else.
- **Sampling noise.** Same procedure, different data. The 3,224 training games are one
  draw of the games that could have been played. Another draw would give another model.

Sampling noise is the one that answers "how sure can we be about this number given how
few games there are", and it is the uncertainty the Wyner group's papers are about. A
seed-only ensemble never changes the data, so it cannot see sampling noise at all. Its
spread measures optimization noise only.

The tool that does measure sampling noise is the game-level bootstrap: draw 3,224 games
with replacement from the training games, refit, repeat. If the bootstrap members also
get different seeds, their spread contains both kinds of noise.

Variances add when the sources are independent, so (by hand, with the measured values
from section 3):

    total variance      = sampling variance + optimization variance
    0.02414^2           = sampling variance + 0.01540^2
    sampling variance   = 0.000583 - 0.000237 = 0.000345
    sampling SD         = 0.0186, or about 1.9 points of win probability

## 2. Design decisions and why

**Same rows, same split, same inputs as the trees.** Train 2010 to 2021 (531,244
plays, 3,224 games), 2022 to 2023 for early stopping (91,663 plays, 569 games), test
2024 to 2025. The 17 features come from `wp.make_features`, unchanged. The script
checks that the test rows and the tree model's test log loss are identical to
`results/m1_wp_metrics.json` and stops if they are not.

**Two fixed input transforms before standardizing.** `down` (0 to 4, where 0 means a
kickoff or try) becomes five 0/1 indicators, because "no down" is not one less than
first down. `margin_per_sqrt_time` is passed through asinh(x / 10): it divides by the square root
of the time left, so it grows very large in the last seconds of decided games, and
those rows would otherwise set the scale for everything else.
Be ready to be challenged on this: in tuning the plain version (no indicators, no
asinh) was 0.0005 worse with the two-layer network and 0.0002 better with the final
one-layer network, both within seed noise. The transforms are defensible on principle
but the data did not show that they help.

**Architecture: one hidden layer of 32 ReLU units.** Chosen by the tuning below.

**Optimizer.** AdamW, learning rate 0.002, weight decay 0.0001, batch 4,096, at most 40
epochs, patience 6. One fit takes about 5 seconds on one CPU thread.

**Tuning used only 2010 to 2021.** Fit on 2010 to 2018 with a random tenth of those
games (240 of 2,403) held out for early stopping, score on 2019 to 2021 (134,072 plays,
818 games), as the M1 tree tuning did. One setting was changed at a time, and every
variant was trained with three seeds, because one seed's result moves by up to 0.0008
on its own. The adoption rule was fixed in advance: a variant replaces the starting
point only if its worst seed beats the starting point's best seed.

- Round 1 started from two layers of 64 units (0.4286). Only one variant cleared the
  rule: one layer of 32 units (0.4279, all three seeds between 0.4278 and 0.4280).
- Round 2 started from there and tried 8, 16 and 64 units, two layers of 32, and the
  regularization settings again. Nothing cleared the rule.

The validation log loss of the tuned trees on the same seasons was 0.4282
(`results/m1_wp_tuning.json`). So the two models were already tied before the test
seasons were touched.

**What `calib` means for this model: early stopping, not Platt.** `fit(train, calib)`
uses the calibration seasons to choose the stopping epoch. With `calib=None` it holds
out a tenth of the training games instead. A Platt map is available
(`calibration: platt`) but is off in the headline model. The reason is the M1 finding
that a Platt slope fitted on two seasons is unstable, which was measured on non-test
seasons. But be honest about the order of events: M1 had also already shown that the
2022 to 2023 map hurt the trees on the test seasons, so this choice was made knowing
that. To keep the comparison fair, the script reports all four combinations: MLP and
trees, each with and without the 2022 to 2023 Platt map.

**No monotone constraints.** The trees are guaranteed never to give a lower win
probability for more points or a better spread. The network has no such guarantee. On
every grid we checked it happens to be monotone (section 3), but that is an observation
about this fit, not a property of the model.

**Ensembles.** The deep ensemble is the mean of 10 networks with seeds 20261117 to
20261126. For the spread comparison there are 30 networks of each of three kinds:
seeds only; game bootstrap with a new seed each; and the same 30 bootstrap resamples
with one seed for all. The third kind was added to see how much of the bootstrap spread
remains when the seed is held still.

**What was subsampled.** The spreads are compared on 2,000 plays drawn at random (with
a fixed seed) from the 90,931 evaluated test-season plays. Only their game states are
used, never their outcomes. The early-stopping seasons are not resampled in any
bootstrap.

## 3. What the results showed

### The comparison
Test seasons 2024 to 2025, 90,931 plays in 569 games, the same rows as M1.

| model | log loss | Brier | ECE |
|---|---|---|---|
| MLP, one network | 0.4461 | 0.1475 | 0.0106 |
| MLP + Platt (2022 to 2023) | 0.4481 | 0.1480 | 0.0234 |
| MLP deep ensemble (10 seeds) | 0.4460 | 0.1474 | 0.0088 |
| MLP bagged ensemble (10 game resamples) | 0.4466 | 0.1475 | 0.0066 |
| Trees + Platt (the M1 model) | 0.4490 | 0.1484 | 0.0217 |
| Trees, no Platt | 0.4469 | 0.1478 | 0.0084 |
| nflfastR `wp` (no spread) | 0.4845 | 0.1642 | 0.0123 |
| nflfastR `vegas_wp` (with spread) | 0.4466 | 0.1478 | 0.0095 |

Log loss differences, with 95% intervals from resampling games (negative means the
first model is better):

| comparison | difference | interval |
|---|---|---|
| MLP minus trees + Platt | -0.0029 | -0.0061 to +0.0005 |
| MLP minus trees, no Platt | -0.0008 | -0.0036 to +0.0020 |
| MLP + Platt minus trees + Platt | -0.0009 | -0.0032 to +0.0013 |
| MLP minus `vegas_wp` | -0.0004 | -0.0035 to +0.0026 |
| Deep ensemble minus one MLP | -0.0001 | -0.0010 to +0.0008 |
| Bagged ensemble minus deep ensemble | +0.0006 | -0.0003 to +0.0015 |

Read this honestly:

1. **The MLP and the trees are tied.** The like-for-like comparisons are the second and
   third rows (both without Platt, both with). Each favors the MLP by less than 0.001
   and each interval comfortably includes zero.
2. **The first row is not like for like.** Most of the 0.0029 is the Platt map hurting
   the trees, which M1 already knew. Even so the interval includes zero.
3. **The same Platt map hurts the MLP by the same amount** (0.4461 to 0.4481, ECE from
   0.011 to 0.023; fitted slope 0.911). That confirms the M1 diagnosis from a second,
   unrelated model: the problem is the two calibration seasons, not the trees.
4. **On calibration the MLP is slightly worse than the uncalibrated trees** (ECE 0.0106
   against 0.0084) and slightly better than the trees with Platt (0.0217).
5. **One network's result depends on its seed.** The ten seeds score between 0.4459 and
   0.4476 on the test seasons. The headline network (0.4461) happens to be the second
   best of the ten; the mean of the ten is 0.4466. So "the MLP" is anywhere from a
   shade better than the trees to a shade worse, depending on the seed.
6. **By quarter and on fourth downs the picture is the same:** differences of 0.000 to
   0.002 in log loss, in the MLP's favor against the uncalibrated trees in every
   quarter, never clearly so.

The two models agree in aggregate but not play by play: the mean absolute difference
between the MLP and the uncalibrated trees on the test plays is 2.4 points of win
probability, and on 11.6% of plays they differ by more than 5 points. Equal log loss
does not mean equal answers.

### Smoothness (figure `assets/m1_wp_nn_comparison.png`)
Three slices that the decision engine cares about. "Roughness" is the sum of absolute
second differences along the curve: how much the slope changes from one grid point to
the next, added up. A straight line scores 0 and a staircase scores high.

| slice | model | distinct values / grid points | flat steps | roughness | steps in the wrong direction |
|---|---|---|---|---|---|
| WP against pregame spread, opening kickoff | MLP | 57 / 57 | 0% | 0.07 | 0 |
| | Trees + Platt | 45 / 57 | 21% | 0.72 | 0 |
| WP against yard line, first and 10, tied, second-half start | MLP | 99 / 99 | 0% | 0.12 | 0 |
| | Trees + Platt | 81 / 99 | 18% | 0.40 | 22 (largest 0.3 points) |
| WP against score, 5:00 left | MLP | 43 / 43 | 0% | 0.17 | 0 |
| | Trees + Platt | 40 / 43 | 7% | 0.54 | 0 |

The network delivers what it was built for: no flat stretches, no reversals along the
field, and curves three to ten times less rough. But the figure also shows two defects
that you must be able to describe.

- **A cliff at the goal-to-go boundary.** Along the field the MLP drops 5.9 points
  between first and goal at the 10 and first and 10 at the 11, where the trees move by
  under 1 point. The `goal_to_go` input flips from 1 to 0 there, the network has
  attached a large weight to it, and nothing in a 32-unit network limits that to the
  situations where it belongs. Averaging ten networks halves the cliff (2.8 points) but
  does not remove it. Smooth in the continuous inputs does not mean smooth everywhere.
- **The opening kickoff is off, and it varies by seed.** For an even game with the home
  team receiving, the headline network says 45.0%. The trees say 49.9% and the
  ten-network ensemble 47.6%. The network fails the kickoff mirror check from M1 (the
  two sides of the same matchup should sum to about 1; the largest deviation is 0.117
  against a tolerance of 0.08, and the average sum is 0.92), which the trees and the ensemble pass.
  Kickoffs are a thin, unusual corner of the data, and a single small network is not
  reliable there.

The levels differ too. Along the field the MLP sits 3.5 points below the trees on
average (largest gap 7.8 points); against the spread 3.4 points on average. This file
cannot say which is right: those are single hypothetical states, and the spread of
either model across refits at such states is 2 to 3 points (next section).

### Does averaging help?
Barely. The ten single networks average 0.4466 log loss on the test seasons and their
mean prediction scores 0.4460, a gain of 0.0006. Thirty networks give 0.4460 as well.
ECE moves from 0.0090 (average member) to 0.0088. Against the headline network alone
the difference is -0.0001 with an interval of -0.0010 to +0.0008.

Why so little: averaging helps when the members make different errors. These members
are small, see identical data and stop early, so they are close to each other (1.5
points apart, below) compared with the 30-point-scale noise in a single win or loss.

What averaging does buy is stability, which log loss does not show: the kickoff value
moved halfway back to the trees, the goal-to-go cliff halved, and the ensemble passes
the sanity check that the single network fails.

Bagging (each member on its own resample of games) gives members that are individually
worse (0.4480 on average, because a resample contains only about 63% of the distinct
games) and an ensemble that is no better than the plain one (0.4466 against 0.4460).

### Is the ensemble spread a meaningful uncertainty? (figure `assets/m1_wp_ensemble_spread.png`)
Standard deviation of win probability across refits, on 2,000 test-season plays, as a
root mean square over the plays:

| what varies | members | RMS spread (points of WP) | relative to MLP bootstrap |
|---|---|---|---|
| MLP, seed only | 30 | 1.54 | 0.64 |
| MLP, game bootstrap, one seed for all | 30 | 2.20 | 0.91 |
| MLP, game bootstrap + new seed (total) | 30 | 2.41 | 1.00 |
| Trees, game bootstrap, no Platt | 101 | 2.75 | 1.14 |
| Trees, game bootstrap, Platt refitted | 101 | 2.73 | 1.13 |

1. **The seed-only ensemble is too narrow, as expected: 0.64 of the bootstrap spread.**
   It is smaller on 99% of the 2,000 plays. In variance terms, seed noise is 41% of the
   total and sampling noise is the other 59% (an implied sampling SD of 1.86 points).
   An interval built from the seed ensemble would be about two thirds as wide as it
   should be. Against the tree bootstrap the ratio is 0.56.
2. **It is narrower than expected in a useful sense, though: it is not negligible.**
   A reasonable guess beforehand would have been that seed noise is a small fraction.
   It is 1.5 points, the size of many go-or-punt margins. A single network's answer to
   a close fourth-down question can flip with the seed. The trees have no comparable
   issue. That is a reason to use the ensemble mean (seed noise then falls by a factor
   of about sqrt(10) to 0.5 points) and never one network.
3. **The seed spread knows where the uncertainty is, even though it gets the size
   wrong.** Across plays its rank correlation with the bootstrap spread is 0.90. Both
   are largest for toss-up states (2.0 against 3.2 points where WP is 40% to 60%) and
   smallest for decided ones (0.3 against 0.6 points where WP is below 5%). The seed
   spread is closest to a rescaled bootstrap spread, and the scale factor is not
   something the ensemble can tell you.
4. **Holding the seed fixed does not isolate sampling noise.** The fixed-seed bootstrap
   spread (2.20) is larger than the 1.86 implied by subtraction. A different resample
   changes the batches and the stopping epoch, so optimization noise comes back in
   through the data. Do not present the fixed-seed bootstrap as "pure" sampling noise.
5. **The two model families roughly agree on the total.** The tree bootstrap spread is
   14% wider than the MLP's (2.75 against 2.41 points) and the two rank plays similarly
   (rank correlation 0.86). The tree bootstrap comes from `scripts/bootstrap_decisions.py`
   (101 refits, 300 trees each). Two caveats on that comparison: the tree bootstrap
   resamples the calibration seasons and ours does not, which is why the no-Platt row
   is the like-for-like one; and in the right panel of the figure the trees' spread
   peaks at 4 points in a tied game, well above the MLP's 2.5, because a staircase moves
   its steps from refit to refit.

Bottom line for M3: the game bootstrap stays the uncertainty method. A deep ensemble is
cheaper (no resampling) but understates the uncertainty by about a third here, and by
an amount you could not know without running the bootstrap anyway.

### Training time
One CPU thread throughout (`torch.set_num_threads(1)`). One network: about 5 seconds
(the headline one stopped after 17 epochs and kept epoch 11). The 90 networks of the
spread study: 368 seconds. The whole script without `--tune`: 382 seconds. The tuning
sweeps add roughly 15 minutes; the slowest single run in them is the memorization
demonstration at 75 seconds.

## 4. Things to be able to say out loud

- The network is tied with the trees on every aggregate score. It is not the better
  model, and the trees are not either.
- Its advantage is smooth curves in the continuous inputs. Its disadvantages are no
  monotone guarantee, a cliff wherever a 0/1 input flips, weak behavior in thin corners
  such as kickoffs, and a result that depends on the seed.
- The early-stopping signal is weak. On the 2022 to 2023 seasons the headline network's
  held-out loss sits between 0.4841 and 0.4851 from epoch 2 to epoch 17, so "the best
  epoch" is close to a coin toss among them (the 30 seed members kept anything from
  epoch 3 to epoch 22). This is the same lesson as the Platt slope: 569 games are a
  small sample for choosing anything.
- The test seasons were used once. Architecture, regularization and the adoption rule
  were settled on 2010 to 2021. One choice (no Platt map) was made with knowledge of
  M1's test result, and both variants are reported for that reason.
- A seed-only ensemble measures optimization noise. It cannot see sampling noise,
  because every member saw the same games. Measured here: 0.64 of the bootstrap spread.

## 5. Open points

- Not done: monotone constraints for the network (possible with constrained weights,
  at a cost in fit), and a fix for the goal-to-go cliff (drop the indicator for the
  network, or regularize its weight). Neither was tried.
- Not done: using the MLP or the ensemble inside the decision engine. Nothing
  downstream reads these models yet. Given the tie on accuracy and the defects above,
  the case for switching is not made.
- The seasons nflfastR fitted its models on are not known to this repo, as in M1.
- The bootstrap in this file does not resample the early-stopping seasons, so the MLP
  "total" spread leaves that source of noise out.
