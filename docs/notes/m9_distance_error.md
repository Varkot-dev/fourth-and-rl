# M9 notes: "Was it really fourth and 1?"

Code: `src/fourth_and_rl/distance_error.py`. Config: `configs/distance_error.yaml`.
Script: `scripts/m9_distance_error.py` (stages `dev`, `test`, `figures`).
Tests: `tests/test_distance_error.py`.
Every number below is from `results/m9_distance_error.json`, written by that script.
Numbers from the public debate (the nfl4th bot, Next Gen Stats, ESPN) are theirs, quoted
from `research-kit/x/2_twitter_findings.md`, and are never mixed with ours. Worked
examples marked "by hand" are arithmetic.
Figures: `assets/m9_distance_error.png`, `assets/questions/q4_really_fourth_and_one.png`.

## 1. The question

Play-by-play records yards to go as a whole number. "Fourth and inches" is stored as 1.
The real distance from the ball to the line to gain is a continuous number that nobody
writes down. After the Lions went for it on fourth and 1 from their own 24 (Packers at
Lions, 2026-10-25), Next Gen Stats posted that tracking data put the real distance
"closer to a 4th and 2", turning their model's go into a toss-up. The nfl4th bot, working
from the recorded 1, had posted "Go for it (+8.1 WP)", rated VERY STRONG.

So: how often does the hidden fraction of a yard change the recommended call, and how much
win probability rides on it? We answer it with play-by-play data alone.

## 2. Measurement error, and why rounding biases a curve

### The recording rule

Every spot on the field is written down as a whole yard. Call the yard a spot falls in its
"cell". The line to gain sits somewhere inside its own cell; call that position v, a number
between 0 (the start of the cell) and 1 (the end). The record is kept consistent: a ball
short of the line is always written at least one yard short. Put together, the recorded
distance d and the true distance D are related by

    d = 1          if 0 < D <= 1 + v
    d = k (k >= 2) if k - 1 + v < D <= k + v

In words: subtract the hidden offset v from the true distance and round up, with 1 as the
smallest value. The offset v changes from series to series because it is wherever the ball
happened to be when the series began (the line is that spot plus 10).

Worked example (by hand). Suppose v = 0.5. Then a recorded 1 means any true distance up to
1.5 yards, a recorded 2 means 1.5 to 2.5, a recorded 3 means 2.5 to 3.5. Note that the
recorded 1 is wider than the others: it holds everything from inches to a yard and a half.

Worked example (by hand). If v were spread evenly between 0 and 1 (every position equally
likely), a recorded 2 means a true distance between 1 and 3 yards, most likely near 2, with
a triangle shape: 2 is twice as likely as 1.5, and 1 and 3 are the extremes. A recorded 1
means anything from 0 to 2 yards: the density is flat from 0 to 1 and falls linearly to 0
at 2. Its average is 7/9 of a yard, about 0.78, not 1.

### Why rounding biases a curve

If you measure conversion rates at each recorded distance and draw them at x = 1, 2, 3,
you have drawn each rate at the wrong place. The rate at "1" is the average conversion over
true distances from 0 to about 1.5, centred near 0.78. Conversion falls with distance, so
that average is higher than the true chance at exactly 1 yard.

Worked example (by hand). Say the true chance is 80% at 0.25 yards, 70% at 0.75 and 62% at
1.25, and a recorded 1 is equally likely to be any of the three. The recorded rate is
(80 + 70 + 62) / 3 = 70.7%. Plotted at 1 yard, it overstates the chance at exactly 1 yard
(about 66% if you interpolate between 0.75 and 1.25) by about 5 points. The whole-yard
rate is right on average, but it is not the chance at any particular true distance.

Whole-yard models (ours from M1, and as far as we know every public bot) are fitted on
recorded distances, so they answer "what happens on a recorded fourth and 1", which is an
average over hidden distances. That is the right answer if you cannot see the ball. It is
the wrong answer if you can, as Next Gen Stats can.

## 3. What the data can tell us about the hidden offset

The brief listed four possible signals. Two of them turn out to carry nothing, and saying so
is part of the answer.

**The integer record is consistent by construction** (key `evidence.integer_consistency`).
Of 9,236 plays that gained exactly the recorded distance, 0.03% were not first downs. Of
23,470 plays one yard short, 0.07% were. In 296,307 consecutive downs, the line to gain
moved 0.002% of the time, and the spot moved by something other than the recorded gain
0.3% of the time (penalties and laterals, mostly). So "a play gaining exactly the distance
that failed" almost never happens: the scorer writes the gain so that it agrees with the
first-down outcome. Signals (b) and (d) of the brief are therefore uninformative in their
direct form. Play-by-play has no "inches" marker either: ydstogo is 1 and the descriptions
do not say (one description in fourteen seasons contains the word).

**The information is in how often each gain occurs.** Look at plays by k = recorded gain
minus recorded distance. A play that ends past the line but still inside the line's own
cell is recorded as k = 0. That cell is only (1 - v) yards wide, so k = 0 should be rarer
than its neighbours. The yard just short of the line also holds the balls inside the line's
cell but short of it, so it is (1 + v) yards wide and k = -1 should be too common. That is
exactly what the data show (figure, top left): on second to fourth downs with 2 to 10 to go,
runs and completed passes, the k = 0 bar is about half the height the smooth trend predicts
and the k = -1 bar is well above it.

The estimator (`fit_line_offset`) is a Poisson regression: count at k = exp(a + b*k +
c*k^2) * width(k), with width 1 everywhere except (1 + m) at k = -1 and (1 - m) at k = 0,
where m is the mean offset. Results:

- Raw pooled estimate m = 0.443 (standard error 0.007; game bootstrap 0.432 to 0.449). By
  down: 0.45, 0.44, 0.46 for downs 2, 3 and 4.
- On simulated plays shaped like the real ones with a known uniform offset (true mean 0.5),
  the estimator returns 0.550: it is biased up by 0.050 because the real gain distribution
  is not exactly quadratic on the log scale over nine yards. Corrected: m = 0.393 (0.382 to
  0.399).
- So the line to gain sits, on average, a little before the middle of its recorded yard.
  This rules out "the line always sits on a yard boundary" (m = 0) and "always at the far
  end" (m = 1).

**A test that the fraction is a real, carried position** (key `evidence.offset.history`).
If the fraction is the ball's actual position, a first down gained by a play that barely
crossed the line (k = 0) starts the next series with the ball late in its yard, so the next
line to gain also sits late in its yard. If the scorer re-rounded every spot independently,
there would be no such memory. On first and 10 after a barely converting play the estimate
is 0.518; after a clear conversion 0.378. Difference 0.139 (standard error 0.045, z about
3). The simulation predicts 0.22 if positions are carried and uniform, 0 if the fraction
resets. The data sit between: the memory is real, but weaker than the pure model says
(spotting noise or some heaping at yard lines would do that). This is why we report a range
of error models rather than one.

What the data cannot tell us: the spread of v around its mean. The dip identifies only the
mean. So the error models (config `error_models`) are:

| Model | Offset v | Spot error | Goal to go |
|---|---|---|---|
| uniform (headline) | uniform 0 to 1 | none | 0.5 |
| fitted_mean | beta, mean 0.39 (corrected estimate), same spread as uniform | none | 0.5 |
| narrow | fixed at 0.5 (a recorded d is a 1-yard window) | none | 0.5 |
| wide | uniform | 0.25 yards (an assumption, not measured) | 0.5 |
| goal_line_edge | uniform | none | 0.0 |

The "plausible range" of a recorded distance is the central 90% of the true distance under
the model. Under the uniform model, a recorded 1 ranges from 0.07 to 1.61 yards, a recorded
2 from 1.31 to 2.69. The NGS description of the Lions play ("closer to a 4th and 2") is
inside the range of a recorded 1 under every model except the narrow one.

Assumption to remember: within one recorded yard, every true distance is equally likely a
priori (flat density). The goal-to-go convention (where the goal line sits inside its
recorded yard) cannot be measured from these plays because goal-line stands pile up just
short of the goal line; it is a sensitivity run.

## 4. Deconvolution in plain terms

We want q(D), the chance of converting at true distance D. We observe only the rate at each
recorded distance, which is q averaged over the true distances that recorded value can
hide:

    P(convert | recorded d) = sum over true distances D of  w(D | d) * q(D)

where w(D | d) is the distance kernel from section 3 (it sums to 1). Going from the observed
averages back to q is called deconvolution: undoing a blur.

Worked example (by hand). Two recorded values, each hiding two equally likely true
distances: recorded 1 hides 0.5 and 1.0; recorded 2 hides 1.0 and 2.0. Observed rates: 72%
at recorded 1, 61% at recorded 2. Suppose q is a straight line on these three points with
q(0.5) = x, q(1.0) = y, q(2.0) = z. Then (x + y) / 2 = 72 and (y + z) / 2 = 61. With y = 66,
x = 78 and z = 56. Without the deconvolution, you would have said "72% at one yard"; the
deconvolved curve says 66% at one yard and 78% at half a yard. But notice the problem: y is
not pinned down by the two equations. y = 68 gives x = 76 and z = 54, which fits equally
well. Something else has to choose, and here it is smoothness.

That is the honest core of this milestone: **whole-yard data identify only the average of
the curve over each recorded yard.** The shape inside a yard comes from an assumption. We
use the smoothest non-increasing curve that fits (`fit_latent_curve`):

    logit q(D) = offset(yard line, down) + h(D)

with h piecewise linear between knots (every quarter yard up to 2 yards, then wider), never
increasing, and a penalty on its bends. The offset carries field position and down from the
existing whole-yard go model at a fixed distance, so only the distance shape is new. The fit
maximizes the likelihood of the recorded outcomes of 106,962 third- and fourth-down runs and
passes from 2010 to 2023, each averaged over its kernel.

The penalty was chosen by season-blocked cross-validation (seven folds of two seasons):
held-out log loss 0.617446, 0.617433, 0.617426, 0.617428, 0.617460 and 0.617813 for
penalties 10, 100, 1000, 3000, 10000 and 100000. 1000 is the minimum. The differences are
tiny precisely because the within-yard shape barely affects how well recorded outcomes are
predicted. The sensitivity runs use 100 and 10000.

**Checks on simulated data with a known curve** (key `simulation_checks.deconvolution`):
300,000 attempts with a curve that has a short-yardage cliff, recorded by the rule. The
fitted curve is within 1 point of the truth from 1.5 yards on, and within about 3
points everywhere. The largest errors are below one yard, where a sharp cliff is flattened. So the
estimate of the chance at inches is the least reliable number in this milestone.

## 5. The continuous curve against the whole-yard model

Fourth down at midfield (key `comparison_whole_yard.midfield_fourth_down`):

| Recorded | Whole-yard model | Mean true distance | 90% range (yards) | Curve at range ends |
|---|---|---|---|---|
| 1 | 71.0% | 0.78 | 0.07 to 1.61 | 78.1% to 62.7% |
| 2 | 60.2% | 2.00 | 1.31 to 2.69 | 65.4% to 56.2% |
| 3 | 54.9% | 3.00 | 2.31 to 3.69 | 58.2% to 52.0% |
| 4 | 50.6% | 4.00 | 3.31 to 4.69 | 53.2% to 48.2% |

The curve at exactly one yard is 68.6%, against 71.0% for a recorded 1: reading the
recorded 1 as exactly one yard overstates conversion by about 2.4 points, the rounding bias
of section 2. For recorded 2 and longer, the mean true distance equals the recorded one and
the bias is near zero. The swing inside a recorded 1 is about 15 points of conversion
probability; inside a recorded 2, about 9.

Fit checks: averaged over its kernels, the curve reproduces the observed third- and
fourth-down rates within 1.2 points for every distance from 1 to 10 (worst gap 0.012), and
the curve's offset needed to reproduce the whole-yard model at distances 1 to 5 varies by
only 0.01 on the log-odds scale at three field positions, so the additive structure fits.

On the log-odds scale relative to one yard, the headline curve gives: inches (0.1 yards)
+0.47, half a yard +0.26, 1.5 yards -0.22, 2 yards -0.38, 3 yards -0.59.

## 6. How the error propagates into a decision

The one-step go value is linear in the conversion probability p:

    go(p) = p * WP(if converted) + (1 - p) * WP(if failed)

Kicks do not depend on the distance. So for each real fourth down we take the engine's
conditional values once, and evaluate go at the curve's p across the plausible distances.
Each play's curve is anchored so that its kernel average equals the engine's whole-yard p;
integrated over the hidden distance, the decision is then exactly the whole-yard decision.
The hidden fraction does not bias the average call. It adds uncertainty to each one.

Worked example (by hand). WP if converted 0.60, WP if failed 0.20, best kick 0.44. Going is
better when 0.60p + 0.20(1 - p) > 0.44, that is p > 0.60. If a recorded 1 has p = 0.66 at
the short end of its range and p = 0.57 at the long end, the call is go at the short end
and kick at the long end: it flips within the range. If p stays above 0.60 across the range,
it does not.

We report four things per play (`propagate`):

- **flips within range**: the call at the short end of the 90% range differs from the call
  at the long end.
- **call differs at the true distance**: the probability, under the kernel, that the call
  at the true distance differs from the whole-yard call. This is the expected share.
- **swing**: go-minus-kick at the short end minus at the long end.
- **value of knowing the distance** (expected value of perfect information): the expected
  win probability gained by seeing the true distance before choosing, average of
  max(go, kick) over the kernel minus max of the averages.

The linear shortcut was checked against the real engine with the probability substituted
(`ConversionOverride`, which rescales the engine's outcome arrays): the go values agree to
0.00001 (key `lions_case.one_step.engine_substitution_check`), and a unit test proves it
exactly on a toy engine. The assumption it carries: where the ball ends up given success or
failure does not change with the true distance.

## 7. Results

### Development seasons (design, engine in sample)

54,927 real fourth downs from 2010 to 2023 with a recorded decision (key `development`).

- The call flips within the plausible range on **10.4%** of fourth downs (game bootstrap
  10.1% to 10.7%). Across error models and smoothness: **7.1% to 12.1%** (narrow model
  lowest, wide model highest).
- The expected share whose call at the true distance differs from the whole-yard call is
  **2.5%** (models 2.0% to 3.0%).
- Mean swing 0.72 win-probability points (median 0.56).
- Value of knowing the true distance: **0.003 win-probability points per fourth down**
  (models 0.002 to 0.004), 0.03 on the plays that flip.

### Test seasons (2024 to 2025, scored once)

The evaluation plan, including the frozen curve values and a hash, was written to the
results file before the test seasons were read; the script refuses to score them twice and
the read is logged in `results/test_access_log.jsonl`. 7,955 fourth downs (key `test`):

- Flips within range: **10.3%** (bootstrap 9.9% to 10.9%; models 7.2% to 12.1%).
- Call differs at the true distance: **2.6%** (2.4% to 2.7%; models 2.0% to 3.0%).
- Mean swing 0.74 points. Value of knowing the distance: 0.003 points per fourth down.

### Which situations are fragile

The hidden fraction only matters where the call was already close (key
`by_whole_yard_margin`). On test fourth downs whose whole-yard margin was under 1 point,
23.3% flip; between 1 and 2 points, 0.4%; above 2 points, none. Development: 22.3%, 0.7%,
0.1%. **If the engine's margin is above 2 points, the hidden fraction does not matter.**

By recorded distance (development / test): 1 yard 9.5% / 6.4%; 2 yards 10.6% / 10.1%;
3 yards 13.7% / 8.8%; 4 to 5 yards 16.8% / 14.5%; 6 to 10 yards 13.2% / 16.0%; 11 and more
2.1% / 2.6%. Two things are counterintuitive here. First, fourth and 1 is not the most
fragile distance; fourth and 4 to 10 is. The swing in conversion is largest at 1, but many
fourth-and-1 calls are clear goes, while at 4 to 10 the engine is often within a point of
kicking. Second, fourth and 1 near midfield is among the most stable cells (4% development,
2% test), because going there is usually a clear win. The fragile fourth and 1s are deep in
the opponent's end (14% to 18% in development, where going competes with a short field
goal) and backed up in one's own end (14%).

Direction: flips are split evenly between "whole-yard go, kick at the long end" (2,934 on
development) and "whole-yard kick, go at the short end" (2,767).

### The Lions' fourth and 1 from their own 24

The 2026 game is in the sealed holdout and was not loaded. The state was rebuilt from the
posts: tied, 4:53 left in the fourth quarter, fourth and 1 at the own 24, even spread, 3
timeouts each (not stated in the posts; assumed). Key `lions_case`.

One-step engine (ours): go minus punt is +6.0 points at a recorded 1 and +3.4 at a recorded
2. Across true distances with the continuous curve: +8.3 at inches (0.05 yards), +7.7 at a
quarter yard, +5.2 at one yard, +4.1 at 1.5, +3.3 at 2 yards. It turns into a punt only at
about **4.7 yards** of true distance (4.5 to 5.1 across models). Timeouts barely matter:
at 2 yards the margin is +3.3 (3 each), +3.3 (2 each), +3.1 (3 and 1).

Exact game solution (ours, M4): +12.6 at a recorded 1, +9.8 at a recorded 2; with the curve,
+14.9 at inches falling to +9.7 at 2 yards, and it never turns into a punt within 8 yards
(beyond 4 yards the solution's conditional values are held at their 4-yard values).

Their numbers, as posted: nfl4th bot "Go for it (+8.1 WP)", VERY STRONG; Next Gen Stats
"We had this as a toss-up. The actual yards to go using tracking data made it closer to a
4th and 2 situation. If it was one yard or less, it would have been a go by our model."; ESPN
reported by Solak as strongly recommending go, no number posted.

Our reading: in our models, the hidden fraction moves this call by about 5 points between
inches and 2 yards, but the call stays go. The NGS flip to a toss-up must come from their
model having a smaller margin at 2 yards than ours, not from the distance error alone. We
cannot check their model.

## 8. What this means for "was it really fourth and 1?"

1. Yes, the recorded distance hides up to a yard and a half, and the data prove the hidden
   fraction is real (the dip at k = 0 and its memory across series).
2. About one fourth down in ten has a call that could go either way within the plausible
   true distance; on average about 2.5% would actually be called differently if the true
   distance were known. These are almost all calls the whole-yard engine already rated
   within one point.
3. The win probability at stake is small: 0.003 points per fourth down from knowing the
   true distance. Tracking data would refine close calls; it would not overturn clear ones.
4. For famous arguments: if a model says go by 6 points or more on a recorded fourth and 1,
   the hidden fraction will not turn it into a punt. If it says go by 1 point, it might.

## 9. Design decisions someone could challenge

- **The recording rule.** We assume spots are recorded by cell with the consistency rule.
  The dip and the memory test support it; the official scoring manual was not consulted.
- **Flat density inside a recorded yard.** Simple and testable; the simulation used a
  decreasing density and still recovered the curve beyond 1.5 yards.
- **Smoothness decides the shape inside a yard.** Not identified by the data. Cross-
  validation picks the penalty but is nearly flat. The chance at inches could be several
  points higher than our curve says (a sharp quarterback-sneak cliff would be flattened).
  That would raise flip shares on recorded 1s.
- **Offset from the whole-yard model.** Field position and down effects are borrowed, not
  re-estimated. The anchor check shows the additive structure holds within 0.01 log-odds.
- **The bias correction of the offset estimate** depends on simulated gain shapes. Corrected
  or not (0.39 or 0.44), the results barely move (the fitted_mean model gives 10.3%).
- **Conditional values fixed across the true distance.** Second order: a longer true
  distance also moves where the ball ends up after converting.
- **Uncertainty in the engine itself is not combined here.** The bootstrap intervals cover
  the curve and which games are in the sample, not the win-probability model (M3 does that).
- **The development numbers use the engine in sample.** The test numbers are the
  out-of-sample answer, and they match.
