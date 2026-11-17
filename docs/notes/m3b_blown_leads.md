# M3b: blown leads as a calibration test

Teaching notes for milestone M3b. Code: `src/fourth_and_rl/blown_leads.py`,
`scripts/blown_leads.py`, `scripts/wp_sample_size_sim.py`. Config:
`configs/blown_leads.yaml`. Every number below that is ours comes from
`results/m3b_blown_leads.json` or `results/m3b_sample_size_sim.json`. Numbers from the
papers are marked as theirs.

## 0. The papers, as actually read

Both papers were downloaded from arXiv and read on 2026-11-17.

**Pipping-Gamon and Wyner, arXiv 2601.18774.** It exists. First submitted 26 January 2026.
The plan cites the title of version 1, "The Blown Lead Paradox: Conditional Laws for the
Running Maximum of Binary Doob Martingales". The current version (v4, 25 August 2026) is
titled "The Blown Lead Paradox: A Pathwise Calibration Benchmark for Win Probability
Forecasts". Cite the current title and mention the old one.

What it says, compared with the plan:

- The plan's formula is the paper's Theorem 2, word for word:
  P(M >= x | lost) = p0 (1 - x) / (x (1 - p0)) for x between p0 and 1, for a calibrated
  path that moves continuously. The proof is the optional-stopping argument in section 2
  below.
- For paths that are reported at discrete times, the paper proves (Theorem 3) that the
  formula is an upper bound on the tail: P(M_N >= x | lost) <= p0 (1 - x) / (x (1 - p0)).
  Discrete reporting can only hide peaks. It also gives the exact gap (Remark 4):
  the tail equals the formula minus C1(x) minus C2(x), where
  C1 = E[(p_tau - x) 1{tau < N}] / (x (1 - p0)) is the overshoot at the first crossing and
  C2 = (1 - x) P(tau = N) / (x (1 - p0)) covers paths that first reach x only when the
  result is revealed. tau is the first time the path is at or above x.
- It combines the two possible losers into one game-level distribution (their section 3):
  with f the favorite's starting value, the CDF of the eventual loser's peak is 0 below
  1 - f, then 1 - (1 - f)/x up to f, then 2 - 1/x. Above f the tail is 1/x - 1 whatever
  the starting value. So "the loser reached 90%" has probability 1/9, about 11%, in any
  game whose favorite started at or below 90%.
- The calibration test is the PIT: U = F(peak; p0) per game, uniform under the
  continuous reference. The test statistic is one-sided,
  D_upper = sup over t of (t - empirical CDF just below t), which is large when too many
  games have U near 1. The p-value comes from a season-stratified team ("pigeonhole")
  bootstrap, because games that share a team are not independent.
- It adds a first-crossing check (their section 5.3): at the first time a team is at or
  above 0.95, compare the observed loss rate with the implied one, 1 - p. Optional
  stopping makes this exact in discrete time.
- Their data: ESPN's public feeds, regular seasons 2018 to 2024. Their NFL result
  (1,848 games): D_upper = 0.015, p = 0.736, "no detected departure"; the share of games
  with U >= 0.90, 0.95, 0.99 was 8.5%, 4.5%, 1.5% against 10%, 5%, 1%. Their NBA result:
  a systematic excess of extreme losing peaks (6.3% of games in the 5% tail). These are
  the paper's numbers, on a different model and sample from ours.
- What the plan got slightly wrong: it expected the paper to supply a correction formula
  to apply to real games. The paper supplies a bound and an identity, and shows the
  direction by simulation (their section 6); it does not give a ready-made corrected
  reference. We built one by simulation and say below how far to trust it.

**Brill, Yurko and Wyner, arXiv 2406.16171.** It exists, title as in the plan, submitted
23 June 2024. They simulate "random walk football" (56 plays, a 4-yard field, true win
probability by dynamic programming), fit XGBoost, and compare training sets with the same
number of rows but different dependence. What it says, compared with the plan:

- The plan says the effective sample size is "closer to the number of games than the
  number of plays". That is not what the paper reports. Its effective sample size is the
  number of independent plays that gives the same error, and for 4,101 games it is 56% of
  the plays ("we have half as much data as we think"), 31% for half as many games and 84%
  for twice as many. That is far more than one observation per game.
- On intervals, the paper is more pessimistic than the plan. For nominal 90% intervals
  their coverage was 0.60 for a bootstrap over plays, 0.71 for a bootstrap over games and
  0.76 for games plus plays within games. None reached 90%. They then shrink the
  resample size (a "fractional" bootstrap) until coverage is 90%, at a mean width of 6.3
  points of win probability, and note that this tuning needs the truth and so cannot be
  done on real data.

## 1. Martingales, and why a calibrated win probability is one

A sequence p_0, p_1, p_2, ... is a martingale if the expected next value, given
everything known so far, equals the current value:

    E[p_{k+1} | everything known at step k] = p_k

A fair gambler's fortune is the standard example. Up moves and down moves cancel on
average, at every moment, whatever has happened before.

A win probability that uses all the available information is
p_k = P(team wins | information at play k). The tower rule of conditional expectation
says E[p_{k+1} | information at k] = p_k: today's probability is the average of
tomorrow's probabilities. This is the definition of "sequentially calibrated", and it is
stronger than the calibration shown in a reliability plot.

Why it must hold: suppose it did not. Say that whenever the model shows 70% after a
turnover, the next number is 74% on average. Then you could improve the model by
printing 74% right away. A model with a predictable drift is leaving information unused,
and a bettor could profit from the drift. So calibrated means no predictable drift, which
means martingale. The path also has to end at 0 or 1, because at the end the result is
known.

Two things this is not. It does not say the path is smooth or that moves are small. And
it does not say the model is good: p_k = 0.5 until the final whistle is a martingale too.
It is a consistency property.

Worked example. A team is at 60%. On the next play there are two outcomes: a big gain
(probability 0.3) after which it would be at 80%, or nothing much (probability 0.7).
Calibration forces the second value: 0.3 * 0.80 + 0.7 * q = 0.60, so q = 0.514. If the
model showed 55% there instead, its average next value would be 0.625, not 0.60, and it
would be drifting.

## 2. The chance of reaching x: p0 / x

Take a calibrated path that starts at p0, moves continuously, and ends at 0 or 1. Pick a
level x above p0. What is the chance the path ever touches x?

**Optional stopping.** For a bounded martingale, the expected value at a stopping time
equals the starting value. A stopping time is a rule for when to stop that only uses the
past, such as "the first time the path touches x, or the end of the game if it never
does". You cannot beat a fair game by choosing when to quit.

Apply it. Stop at the first touch of x, or at the end. Two things can happen:

- The path touches x. Because it moves continuously it is exactly at x when we stop.
- The path never touches x. Then it cannot have ended at 1 (to get to 1 it would have to
  pass through x), so it ended at 0.

So the expected stopped value is x * P(touch x) + 0 * P(never touch) and optional
stopping says this equals p0:

    P(touch x) = p0 / x

Intuition without the theorem: of all the paths that reach x, a fraction x go on to win
(that is what "x" means for a calibrated model). Every winning path reaches x. So the
winners are (fraction that reach x) * x, and the winners are p0 of all paths.

Worked examples. An even team (p0 = 0.5) reaches 75% with probability 0.5 / 0.75 = 2/3.
A 20% underdog reaches 50% with probability 0.4, and reaches 90% with probability 0.22.
A 60% favorite reaches 90% with probability 2/3.

## 3. The loser's peak

Among paths that reach x, a fraction 1 - x go on to lose. So

    P(reach x and then lose) = (p0 / x) * (1 - x)

and dividing by the chance of losing, 1 - p0:

    P(loser's peak >= x | lost) = p0 (1 - x) / (x (1 - p0))     for p0 <= x < 1

It is 1 at x = p0 (every path is at least at its start) and falls to 0 at x = 1 (a loser
never gets to certainty).

For an even game this is (1 - x) / x:

| x | (1 - x) / x | meaning |
|---|---|---|
| 0.67 | 0.50 | half of all losers were at least 2 to 1 favorites at some point |
| 0.75 | 0.33 | a third reached 75% |
| 0.90 | 0.11 | one in nine reached 90% |
| 0.95 | 0.053 | one in nineteen |
| 0.99 | 0.010 | one in ninety-nine |

With about 272 games a season, the formula predicts about 30 losers a season who reached
90% and 2 or 3 who reached 99%. That is the "paradox": the number quoted on television
after a collapse, "they had a 90% chance to win", describes something that should happen
in one game in nine. The 10% belongs to the question "given a team is at 90% now, how
often does it lose". The post-game number answers a different question: "among teams that
lost, how high did they get", and the team was picked because it lost.

The starting value matters. For a 20% underdog the chance of peaking at 90% or more,
given it lost, is 0.2 * 0.1 / (0.9 * 0.8) = 0.028. For an 80% favorite it is
0.8 * 0.1 / (0.9 * 0.2) = 0.44. A favorite that loses was usually far ahead on the
probability scale at some point, simply because it started there.

The quantile function inverts the formula. The peak that a fraction u of losers stay
below is x = p0 / (p0 + (1 - u)(1 - p0)). For p0 = 0.5 the median loser's peak is 2/3.

## 4. The PIT: putting all games on one scale

Games start at different p0, so their peaks are not comparable directly. The probability
integral transform fixes that. If a random number X has continuous CDF F, then F(X) is
Uniform(0, 1). So for each game compute

    u = F(peak | p0) = 1 - p0 (1 - peak) / (peak (1 - p0))

If the model is calibrated and paths are continuous, the u values of all games are
uniform on (0, 1) whatever the mix of favorites and underdogs. u = 0.97 means "this peak
was higher than 97% of what calibrated losers with this start would show".

Reading the histogram of u:

- flat: calibrated;
- piled up near 1: too many extreme peaks, the model gets too sure too early
  (overconfident);
- thin near 1: too few extreme peaks, which means either an underconfident model or, as
  it turns out, peaks hidden by discreteness.

The formal test is Kolmogorov-Smirnov: the largest gap between the empirical CDF of the
u values and the diagonal. Its textbook p-value assumes the games are independent and
that the reference is exact. Neither is quite true (teams recur, and real paths jump), so
we also report the paper's one-sided statistic with the team bootstrap, and we do not
lean on the KS p-value alone.

We compute two versions of u. `pit` conditions on which team lost and uses that team's
p0; it uses the most information. `pit_game` is the paper's version, which does not
condition on the loser. Both are uniform under the reference.

## 5. Discreteness: where the continuous formula fails

The derivation used continuity twice: "the path is exactly at x when it first reaches x",
and "a winner must pass through x before the end". Real paths move in jumps, one per
play. Two things change:

- **Overshoot.** The first value at or above 0.90 is typically 0.92, not 0.90. Optional
  stopping still holds, so p0 = E[value at first crossing] * P(cross), and with a bigger
  value at the crossing there must be fewer crossings, and those that cross lose less
  often (8%, not 10%).
- **Last-moment winners.** Some winners are below x until the final play decides the
  game. In the continuous world every winner passes x early; here some of the "reach x"
  probability is used up by wins that never showed x on the feed.

Both effects cut the number of losers with a high peak. The paper's Theorem 3 says
exactly this: for a calibrated discrete path the formula is an upper bound.

What our real paths look like (nflfastR `wp`, 709,242 play-to-play changes): the median
change is 1.4 points of win probability, one play in ten moves it by more than 5.4
points, one in a hundred by more than 15.4, and the largest change in a typical game is
20 points. The loser never rises above its starting value in 8.0% of games, which has
probability zero for a continuous path.

**The simulator.** A simulated game in which the truth is known: the lead is a Brownian
motion W on game time 0 to 1, the team starts with a handicap x0 = Phi^-1(p0) and wins if
x0 + W(1) > 0. Its exact win probability is p(t) = Phi((x0 + W(t)) / sqrt(1 - t)). This
path is continuous and calibrated, so the formula holds for it exactly. Then we look at
it only at n play times, as a feed would. The reported values are still an exact
martingale, but peaks between plays are missed.

How much is missed, for an even game with equally spaced plays (share of losers at or
above x, as a fraction of the formula):

| plays | x = 0.90 | x = 0.95 | x = 0.99 |
|---|---|---|---|
| 20 | 0.49 | 0.40 | 0.25 |
| 163 | 0.78 | 0.71 | 0.59 |
| 650 | 0.86 | 0.81 | 0.70 |
| 2,600 | 0.94 | 0.91 | 0.87 |
| 10,400 | 0.98 | 0.94 | 0.91 |

(Monte Carlo standard errors are 0.005 to 0.06, largest in the last row and column.)
At the resolution of an NFL game, 163 plays, a perfectly calibrated model shows only
about three quarters of the 90% peaks the formula predicts, and about 60% of the 99%
peaks. The convergence is slow because the last few plays always carry a large share of
the remaining uncertainty: near the end the scale 1 / sqrt(1 - t) blows up, so the final
steps are big however many plays there are.

To make the jumps realistic the play times are drawn with Gamma-distributed gaps, so a
few plays carry much more game time than others. The Gamma shape is chosen so that four
quantiles of the simulated jump sizes (50%, 90%, 99%, 99.9%) are as close as possible to
the real ones. The fit is decent but not perfect: the simulator has slightly too few
jumps of 10 to 40 points (see `assets/m3b_discreteness.png`), and it predicts 8.9% of
losers never rising above their start against 8.0% observed. The chosen shape differs by
model (1.2, 1.5, 1.5) because the fit is flat in that range; expected counts for equal
spacing and for a much rougher shape of 0.5 are saved as a sensitivity range.

## 6. Results

4,350 non-tied games, 2010 to 2025, for three models: nflfastR `wp` (no point spread),
nflfastR `vegas_wp` (with spread), and ours (`wp_ours`, the out-of-fold predictions from
M1). The nflfastR models were fitted by their authors on data that overlaps these
seasons, so for them this is not an out-of-sample test.

Losers who peaked at or above x, observed against three references:

| model | x | observed | formula | simulated with jumps | first crossing |
|---|---|---|---|---|---|
| nflfastR (no spread) | 0.90 | 312 | 482.9 | 360.2 | 354.4 |
| | 0.95 | 113 | 228.7 | 154.3 | 145.2 |
| | 0.99 | 11 | 43.9 | 23.5 | 16.7 |
| nflfastR with spread | 0.90 | 320 | 479.0 | 372.1 | 356.7 |
| | 0.95 | 136 | 227.6 | 160.9 | 148.8 |
| | 0.99 | 12 | 43.7 | 25.0 | 19.6 |
| Ours | 0.90 | 359 | 489.2 | 381.5 | 347.9 |
| | 0.95 | 160 | 234.7 | 167.8 | 141.2 |
| | 0.99 | 17 | 45.3 | 26.4 | 18.9 |

- **Formula:** sum over games of p0 (1 - x) / (x (1 - p0)).
- **Simulated with jumps:** the same sum using the simulated discrete reference.
- **First crossing:** sum over teams of 1 - p at the first value at or above x. This
  one needs no continuity and no simulator. It is exact for any calibrated model, and the
  number of losers who peaked at or above x is exactly the number of first crossings that
  ended in a loss, so the observed column is the same.

What this shows:

1. **No model shows too many extreme peaks.** Against the continuous formula every model
   has far too few: about two thirds of the expected count at 90% and a quarter to a
   third at 99%. The KS statistics against the formula are 0.085, 0.067 and 0.064 (all
   with p far below 0.001), and all of the departure is on the "too few high peaks" side;
   the one-sided statistic that looks for overconfidence is 0.000 for both nflfastR
   models and 0.0002 for ours.
2. **Most of that gap is discreteness, not miscalibration.** The first-crossing
   reference, which is exact for jumpy paths, predicts 354 where the formula predicts
   483. The simulator independently lands in the same place (360). Taking the formula at
   face value would have led to the wrong conclusion that these models are badly
   underconfident.
3. **After the correction, small departures remain, and they differ by model.** Each
   result below gives two z values: the first treats every first crossing as an
   independent episode, the second (`team_season_bootstrap` in the results file) uses the
   same season-stratified team bootstrap weights as the PIT test, so both sides of a game,
   and games that share a team-season, are resampled together.
   - nflfastR without spread: at first crossings of 0.90 and 0.95 teams lost less often
     than implied (7.3% against 8.3%, and 3.1% against 3.9%; z = -2.4 and -2.7 assuming
     independence, -1.5 and -1.8 with the team bootstrap, two-sided p = 0.12 and 0.07).
     That is mild underconfidence at the top, and it is not clearly there once the
     dependence is allowed for.
   - Ours: at 0.90 and 0.99 observed and implied agree (359 against 348, 17 against
     19); at 0.95 there are a few more losses than implied (160 against 141, z = 1.6, or
     0.9 with the team bootstrap). At first crossings of 0.60, 0.70 and 0.80 teams lost
     more often than our model implied: 33.6% against 32.7%, 26.5% against 25.3%, and
     18.3% against 16.9% (z = 1.5, 2.1, 2.7 assuming independence; 1.6, 1.7, 1.8 with
     the team bootstrap, two-sided p = 0.11, 0.08, 0.07). That is mild overconfidence in
     the middle of the range, about one to one and a half points, and it is the one place
     where this analysis finds a possible fault in our model. The cross-fitted development
     seasons show the same thing in M1 (a pooled Platt slope of 0.93,
     `docs/notes/m1_wp.md` section 6.5).
     The spread version of nflfastR shows the same pattern (z = 2.2, 2.1, 2.4 assuming
     independence; 2.4, 1.7, 1.6 with the team bootstrap, p = 0.02 at 0.60).
   - At 0.99 the two nflfastR models have fewer losses than implied (11 and 12 observed
     against 17 and 20), within or at the edge of their 95% intervals.
   The team bootstrap mostly shrinks the z values, by about a third at 0.80 and above,
   because games that share a team carry correlated surprises. At 0.60 it can widen them
   slightly: both teams of a game can cross 0.60, and since one of them must lose, the two
   episodes' surprises tend to cancel, which the independence count does not credit.
   There are 18 crossing tests here, so a single z of 2 means little. With the team
   bootstrap no test reaches the bar of a Bonferroni correction for 18 tests (two-sided
   p below 0.0028, or |z| above about 3.0); the largest is nflfastR with spread at 0.60
   (z = 2.4, p = 0.02), and ours is z = 1.8 at 0.80. Even under the independence
   assumption our z = 2.7 at 0.80 only sat at the edge of that bar. So the mid-range
   overconfidence is suggestive, not established.
4. **Against the simulated reference the PIT is much flatter** (KS 0.036, 0.051, 0.042
   instead of 0.085, 0.067, 0.064) but not flat: there are somewhat too many moderately
   high peaks and too few very high ones, and the one-sided bootstrap test rejects for
   the two spread-aware models (p = 0.0005 and 0.0025; 0.02 for nflfastR without spread).
   Part of that is the mid-range overconfidence in point 3, which the exact test
   confirms, and part is the simulator not being football. This test cannot separate the
   two, so the first-crossing test is the one to trust.
5. For comparison, the paper's NFL feed (ESPN, their numbers) put 8.5%, 4.5% and 1.5% of
   games above the 90%, 95% and 99% points of the continuous game-level reference. Ours
   are lower: 6.1%, 2.4% and 0.25% for nflfastR without spread, and 7.5%, 3.4% and 0.37%
   for our model. The nflfastR-style paths reach extreme values less often than ESPN's.

**Answer to "does discreteness change the conclusion": yes, completely.** With the
continuous formula the conclusion would be "strongly underconfident". With the correct
discrete references it is "close to calibrated in the tail, with a possible small
mid-range overconfidence in the spread-aware models and a possible small underconfidence
at the top for nflfastR without spread", neither of which survives the correction for 18
tests once the dependence between crossings is allowed for.

### Collapse rankings

Two top-25 lists per model are saved: `results/m3b_collapses_by_peak.csv` and
`results/m3b_collapses_by_surprise.csv`.

- By raw peak: the highest win probability the loser reached.
- By surprise: the smallest tail probability p0 (1 - peak) / (peak (1 - p0)).

For nflfastR without spread every team starts near 50%, so the lists nearly coincide (20
of 25 shared, rank correlation over all games 0.93). With the spread in the model they
differ a lot: 14 of 25 shared for nflfastR with spread and 14 of 25 for ours, rank
correlation 0.59 and 0.60. Favorites that peaked high drop out of the surprise list, and
underdogs with slightly lower peaks enter. Example from the spread model: Jacksonville,
a 6% pregame underdog at Houston in 2012, reached 92% and lost in overtime. That peak is
not in the raw top 25 but is a 1 in 180 event for a team starting at 6%. In the other
direction, New England started at 92% against Miami in 2019 week 17 and peaked at 97%:
about one calibrated loser in three with that start gets that high.

The most surprising game under both nflfastR models is Indianapolis at Minnesota, 2022
week 15 (Indianapolis led 33-0 and lost 39-36). Under our model it is Minnesota at
Buffalo, 2022 week 10: Buffalo led 27-23 with the ball at its own 1 and 49 seconds left,
and fumbled the snap into a Minnesota touchdown. Our version 2 model treats that state as
all but won because the leader could have run out the clock (the kneel rule in
`docs/notes/m1_wp.md`, section 6.5), so it ranks first at 1 in 17,600 by the formula.
Super Bowl LI (Atlanta led 28-3) is in every list, between 8th and 19th.

Surprise is computed from the continuous formula because it is a closed form that can be
explained in one line. Its absolute "1 in N" values are therefore too small by the
discreteness factor (roughly 1.3 at 90% and 2 at 99%); the ordering is hardly affected.

## 7. Why plays within a game are correlated, and effective sample size

A win-probability model is fitted to rows of (game state, did this team win). A game
contributes about 160 rows and all of them carry the same answer. If the home team won on
a last-second field goal, 160 rows say "win" for the home side. The rows are 160 looks at
one coin flip, not 160 coin flips.

**Effective sample size** is the number of independent observations that would give the
same precision. If n rows have a "design effect" D (true variance divided by the variance
you would compute assuming independence), they are worth n / D independent rows. For
m equally correlated rows per cluster with correlation r, D = 1 + (m - 1) r. With r = 1
each cluster is worth one row; with r = 0 it is worth m.

Our simulation (`scripts/wp_sample_size_sim.py`): a game is 50 plays, the lead is a
Gaussian random walk, and the true win probability is Phi(lead / sqrt(50 - t)). Two
estimators: a three-parameter probit that contains the truth, and LightGBM on (t, lead).

**(a) Error against number of games.** For LightGBM the error (RMSE against the true win
probability) falls from 8.7 points with 100 games to 1.7 points with 3,200 games. The
same number of plays taken from separate games gives 5.7 and 1.2 points. One play per
game is far worse (28 and 6.9 points): keeping all the plays is clearly better than
throwing them away, even though they are dependent.

**(b) Intervals.** Nominal 90% intervals for the true win probability, 400 games per fit:

| estimator | interval | coverage | mean width |
|---|---|---|---|
| probit | formula, plays independent | 26% | 1.4 points |
| probit | bootstrap over plays | 25% | 1.4 points |
| probit | bootstrap over games | 90% | 7.1 points |
| LightGBM | bootstrap over plays | 62% | 8.0 points |
| LightGBM | bootstrap over games | 77% | 11.8 points |

Treating plays as independent gives intervals five times too narrow for the probit, and
resampling plays reproduces exactly that mistake, because resampling rows keeps the
false assumption that rows are independent. Resampling whole games gives 90% for the
probit. For LightGBM the game bootstrap is better than the play bootstrap but still
short (77%), because a flexible model is also biased and the bootstrap only measures
variance. The paper found the same ordering with XGBoost (their numbers: 0.60 for plays,
0.71 for games). So "bootstrap over games" is necessary but, for a machine-learning
model, not sufficient; the "too close to call" regions in M3 should be read as a lower
bound on the uncertainty. LightGBM coverage here rests on 8 simulated data sets with 40
resamples each, so it has a standard error of about 3 points.

**(c) Effective sample size.** It depends on what is being estimated.

- Probit: the design effect of the slope is 11.7, so 20,000 plays from 400 games are
  worth about 1,700 independent plays, roughly 4 per game. Matching errors instead
  (how many independent plays give the same RMSE) gives about 2 per game across all
  sizes. For this model the plan's statement holds: the data are worth a small multiple
  of the number of games, nowhere near the number of plays.
- LightGBM, by the paper's definition: the dependent plays are worth 37% to 50% of their
  number, about 20 independent plays per game. The paper's own figure for XGBoost is 56%
  (theirs). So for a flexible model the loss is "about half", not "down to the number of
  games".

Why the difference: the probit pools all plays into three numbers, so the shared label
hurts it fully. A tree model estimates each region of (time, lead) mostly from the plays
that land there, and a game puts at most one play at each time, so within a region the
plays come from different games and are nearly independent. The flexible model is also
limited by bias, which dependence does not change. The honest summary is: the effective
sample size is somewhere between the number of games and half the number of plays,
depending on how much the model pools, and intervals must be built by resampling games
either way.

## 8. Design decisions, and why

1. **Rows with a null `play_type` are dropped** (timeouts, end of quarter, about 2% of
   rows). Their `posteam` is unreliable: in some games the first rows name the wrong team
   and would flip the path. A test checks that after this filter our home-team path
   equals the `home_wp` column nflverse ships, to 1e-9.
2. **A team's value without the ball is 1 minus the posteam value.** The posteam
   column is turned into a home-team path once, and the loser's path is that or its
   complement.
3. **p0 is the first value of the game** (the opening kickoff row), as in the paper. For
   models without the spread it is always near 0.5.
4. **Values are from before each play, so no observed value is 0 or 1.** The path is
   treated as ending at 0 for the loser after the last play. That final value cannot be
   the peak, and the final drop is not counted as a jump.
5. **Tied games are dropped** (13 of 4,363): there is no loser, and the theory needs a
   0 or 1 ending. Games with fewer than 50 usable values would be dropped; none are.
6. **Two PITs.** The one conditioned on the loser is the primary one; the paper's
   game-level one is kept for comparison with the paper.
7. **Three references are reported side by side** (formula, simulation, first crossing)
   instead of one, because they fail in different ways: the formula ignores jumps, the
   simulator is a stylized game, and the first-crossing test is exact but only tests one
   threshold at a time.
8. **Simulator.** Brownian lead observed at play times. It was chosen because the win
   probability is exact in closed form, it stays an exact martingale under any reporting
   grid, and the continuous formula is its exact limit, so any gap is discreteness and
   nothing else. One free parameter (the Gamma shape of the gaps) is fitted to the real
   jump sizes. The number of plays is the real median, 163.
9. **Team bootstrap** for the PIT test follows the paper (season-stratified pigeonhole
   bootstrap, 2,000 draws). The first-crossing test reports both the independence z and
   an interval and p-values from the same bootstrap weights, each crossing taking its
   game's weight. The peak-count intervals use the simpler assumption of independent
   games and say so.
10. **Ranking ties.** Raw peaks are compared after rounding to three decimals, because
    differences beyond a tenth of a point are below what any model can support; ties go
    to the later peak. Without rounding the order among the 11 to 17 games above 99%
    would be decided by noise.
11. **Sample-size simulation.** A Gaussian random walk instead of the paper's random-walk
    football, because the truth is a closed form and a correctly specified model exists,
    which separates the dependence problem from the bias problem. Sizes were kept small
    (400 games for coverage) to run in about five minutes on two threads.

## 9. Limits

- The simulated reference is an approximation. It matches the jump-size quantiles and the
  share of losers who never rise above their start, but it is not football, and the PIT
  against it is not flat.
- The peak-count intervals assume independent games. The team bootstrap is used for the
  PIT statistic, its tail shares and the first-crossing test (reported next to the
  independence z, which is kept for comparison).
- The nflfastR models are tested partly in-sample. Ours is tested on out-of-fold
  predictions.
- The crossing tests at six thresholds and three models are 18 looks at the data.
- The sample-size simulation is a toy game with 50 plays and two state variables. Real
  football has a much larger state space; the paper expects the effective sample size to
  be smaller there.
- Wall-clock time for `scripts/wp_sample_size_sim.py` was about 5 minutes on a shared
  machine with 2 threads, and about 2 minutes for `scripts/blown_leads.py`.

## 10. Answers to the plan's questions 11 to 14

**11. What is a martingale, and why must a calibrated win probability be one?** A
sequence whose expected next value equals its current value given the past. A win
probability that uses its information fully is a conditional probability of a fixed
event, and conditional probabilities average to themselves as information arrives. If it
drifted predictably you could fix the model by adding the drift.

**12. Derive the chance that a team reaches x and still loses. Why is a 90% peak for a
loser unremarkable?** Stop the path at the first touch of x or at the end. Optional
stopping gives x * P(touch) = p0, so P(touch) = p0 / x. Of those, 1 - x lose, so
P(reach x and lose) = p0 (1 - x) / x, and given a loss, p0 (1 - x) / (x (1 - p0)). For an
even game at x = 0.9 that is 1/9: one loser in nine was a 90% favorite at some point,
about 30 games a season by the formula. With real play-to-play jumps it is nearer
one in fourteen (312 of 4,350 for nflfastR, 7.2%), still a weekly event.

**13. Why is the effective sample size closer to games than plays?** Every play of a
game carries the same win or loss, so the labels are one draw per game. For a model that
pools plays (our probit) 20,000 plays were worth about 1,700 independent ones. Be
precise, though: for a flexible tree model the simulation gives 37% to 50% of the plays,
and the paper reports 56%. What is always true is that intervals that treat plays as
independent are far too narrow (26% coverage instead of 90%) and that you must resample
games.

**14. How does the blown-lead distribution act as a calibration test, and what would an
overconfident model look like?** Turn each game's loser's peak into u = F(peak | p0).
Calibrated continuous paths give uniform u. An overconfident model pushes probabilities
toward 0 and 1 too early, so teams "reach 95%" too often and then lose more than 5% of
the time: the histogram of u piles up near 1 and there are more losers above 90%, 95%
and 99% than the reference predicts. We did not see that in the tail for any of the
three models. The catch is that real paths jump, which removes extreme peaks even for a
perfect model, so the comparison has to be made against a discrete reference or with the
first-crossing version (observed losses against the sum of 1 - p at first crossings).
That version did find our model about one to one and a half points overconfident at
first crossings of
60% to 80%.
