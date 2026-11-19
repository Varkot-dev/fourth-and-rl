# M10: how much does being predictable cost an offense?

Code: `src/fourth_and_rl/predictability.py`. Config: `configs/predictability.yaml`.
Script: `scripts/m10_predictability.py`. Tests: `tests/test_predictability.py`.
Every number below is from `results/m10_predictability.json`, written by that script,
unless it is marked "by hand" (arithmetic on a made-up example) or attributed to
someone else. Figures: `assets/m10_predictability.png` (the headline) and
`assets/m10_team_predictability.png`.

EPA is nflfastR's expected points added, used as given and quoted as theirs. Play
classification (scrambles count as passes), the neutral game-script filter and the
situation cells are question 2's (`run_pass.py`, `configs/run_pass.yaml`), unchanged.
Intervals in brackets are 95% percentile intervals from resampling whole games.

## 0. The answer in short

Data: 476,735 called runs and passes in 3,793 development games (2010 to 2023); the cost
analysis uses neutral plays on downs 1 to 3 (189,873 plays scored by models fitted only
on earlier seasons, 2013 to 2023), and is repeated once on 2024 to 2025 (35,876 plays)
after the evaluation plan was written (plan hash e080a1e2fc6bd6da).

1. Play history makes calls more predictable, but not by much. Adding the calls and
   results earlier in the game and drive lowers the log loss of a situation-only
   gradient-boosting model by 10.7 nats per 1,000 calls on 2020 to 2023 validation
   [9.8, 11.8] and by 9.7 on 2024 to 2025 [8.2, 11.2] (0.525 to 0.515 per call on the
   test seasons). The situation already explains 0.147 nats per call over the base rate;
   history adds about 7% on top. A GRU over the game's play sequence adds 14.7 [12.6,
   16.5] over the same network without it, but the network as a whole stays 3.2 [2.0, 4.6]
   per 1,000 worse than boosting with hand-made history features. Shotgun and no-huddle,
   which the defense sees before the snap, add twice as much as all the history (19.9
   [17.5, 22.4]): our predictability is a lower bound.
2. Expected runs gain less; expected passes do not, as far as our model can see. Same
   offense-season, same situation group, same opponent-season: each 10 points of P(run)
   costs a run 0.015 EPA (nflfastR's) [0.011, 0.024] on 2013 to 2023 and 0.013 [0.003,
   0.031] on 2024 to 2025. From the most surprising runs (P(run) under 0.2) to the most
   expected (over 0.8) that is about 0.10 EPA per run. For passes the slope is +0.005
   per 10 points [-0.004, +0.012], no cost; on the test seasons -0.004 [-0.029, +0.016].
   With shotgun and no-huddle in the model, expected passes do lose (-0.012 per 10 points
   [-0.018, -0.006] on development seasons, -0.005 [-0.023, +0.007] on test), so part of
   the pass cost may hide in information we do not have.
3. Offenses barely differ in how predictable they are once the situation is held fixed,
   and a third of what difference there is, is noise. The spread of team-season excess
   predictability is 0.7 points of P(call) (standard deviation); 39% of that spread is
   sampling noise (36% on the test seasons) and it carries over to the next season with
   a correlation of 0.38. Priced with the league slopes, team-season costs spread with a
   standard deviation of about 0.10 EPA per game (0.03 on the test seasons), but this
   ranking mostly restates pass rate (correlation -0.72): a pass-heavy offense's runs come
   when the model expects a pass, and surprising runs do better. It is not a clean
   measure of a coaching skill. Whether the cost itself differs by offense cannot be
   told apart from noise (88% of the spread of team-specific slopes is noise).
4. Being predictable is not related to offensive success once offense quality is held
   fixed: per 10 points of excess P(call) in a game, EPA per play changes by -0.019
   [-0.052, +0.016] with prior-games passing and running efficiency, opponent defense
   and tendency controlled, -0.013 [-0.049, +0.021] within team-season, and +0.007
   [-0.067, +0.093] on the test seasons.
5. First-down runs. On neutral first and 10 offenses run 53% of the time, and the model
   gives their runs an average P(run) of 0.555 (0.513 for runs on other downs): first-down
   runs are only a little more predictable than a coin flip from what a defense knows
   before the snap, without formation. 18% of them have P(run) of 0.65 or more. The cost
   per 10 points of P(run) is 0.003 EPA [-0.002, +0.014] on development seasons and 0.022
   [0.005, 0.045] on the test seasons: somewhere between nothing and 0.02. Compare
   question 2's pass-minus-run gap on neutral first and 10, 0.205 [0.191, 0.217] (its
   number, all seasons): the cost of predictability is a tenth or less of it. Team
   differences in first-down run predictability are mostly noise (54% on development
   seasons, 80% on test; season-to-season correlation 0.24).

Answer to "how much does being predictable cost an offense?": once the situation, the
offense and the opponent are held fixed, little. Runs the defense could see coming do
gain less, about 0.015 EPA per run per 10 points of predictability; passes show no
detectable cost with the information we have. This does not settle the run-pass debate
either way, and no balance point is claimed (Decision log, "M10 narrowed").

## 1. Concepts

### What "predictable" means, as a probability

Before the snap a defense has a belief about what is coming. We model that belief as a
probability, P(pass), computed only from what the defense could know: down, distance,
field position, score, clock, timeouts, the pregame spread and total, weather, the
offense's tendencies in its earlier games, and the calls and results earlier in the same
game and drive. For the play that was actually called, the number that matters is

    P(call) = P(pass)      if the offense passed
    P(call) = 1 - P(pass)  if it ran

A call with P(call) = 0.85 was telegraphed; a call with P(call) = 0.25 was a surprise.
"Surprise" in information-theory units is -log P(call), measured in nats (natural log).

Worked examples (by hand):

1. Second and 1, the model says P(pass) = 0.25. The offense runs. P(call) = 0.75,
   surprise = -log 0.75 = 0.29 nats. A predictable call.
2. Same spot, the offense passes. P(call) = 0.25, surprise = -log 0.25 = 1.39 nats.
   A surprise.
3. First and 10, P(pass) = 0.50. Either call has P(call) = 0.50 and surprise
   0.69 nats, which is the most surprise a two-way call can carry on average.

### Log loss and calibration of the classifier

Log loss is the average surprise over all calls: mean of -log P(call). Lower is better:
a model that always says 0.5 scores 0.693; a perfect model scores 0. It punishes
confident mistakes heavily (P(call) = 0.05 costs 3.0 nats), which is why it is the
right score for a probability, not just for a yes-or-no guess.

Example (by hand): four calls with P(call) 0.8, 0.6, 0.9, 0.3. Log loss =
(0.223 + 0.511 + 0.105 + 1.204) / 4 = 0.511 nats.

Calibration asks whether the probabilities mean what they say: of all plays where the
model said 70% pass, did about 70% turn out to be passes? We group predictions into ten
bins and compare the predicted with the observed pass share. The expected calibration
error (ECE) is the play-weighted average gap. A model can rank plays well (high AUC) and
still be badly calibrated; for this question calibration matters, because P(call) is
used as a measure, not just a ranking.

Example (by hand): 1,000 plays predicted at 0.70 of which 640 were passes; 1,000 at 0.30
of which 330 were passes. ECE = 0.5 * |0.64 - 0.70| + 0.5 * |0.33 - 0.30| = 0.045.

The Brier score is the mean squared error of the probability, (p - y)^2; AUC is the
chance that a random pass got a higher P(pass) than a random run. Both are reported.

### What a sequence model adds over a situation-only model

A situation-only model sees each snap on its own. A defense also remembers what the
offense just did. Some offenses alternate (run, then pass), some stay with what worked,
some go to a habit after a negative play. A sequence model reads the earlier plays of the
game in order and can learn such patterns.

Tiny example (by hand). An offense on second and 7 passes 50% of the time overall. But
it passed on first down in half of those drives and ran in the other half, and it
alternates: after a first-down run it passes 80% of the time, after a first-down pass
20%. The situation-only model says 0.50 on every second and 7 and scores log loss
0.693. A model that knows the previous call says 0.8 or 0.2 and scores
-(0.8 log 0.8 + 0.2 log 0.2) = 0.500. The 0.193 nats per call is the information the
history carries about the call.

Two ways to give a model the history are compared:

- hand-made features (previous call, its yards, whether it was incomplete, the pass
  share so far in the game and drive, EPA of passes and runs so far, the drive number)
  added to gradient boosting;
- a GRU (a gated recurrent network) that reads up to 40 earlier plays of the offense in
  the same game, oldest first, each as a short vector (call, yards, success, first
  down, incompletion, sack, turnover, down, distance, field position, EPA, same drive or
  not, how many plays ago). Its final hidden state is joined to the situation features
  and passed to a small multilayer network. The same network without the GRU is the
  fair baseline: the difference between the two is what the sequence adds.

A GRU keeps a running memory vector h. At each earlier play it computes an update gate
z (how much to overwrite memory) and a candidate memory from the new play, and sets
h_new = (1 - z) * h + z * candidate. It can learn "the last call matters most" or "the
whole game's mix matters" from data, where the hand-made features fix that choice.

### The confounding problem

The naive plot is EPA against P(call). It is not a cost of predictability, for three
reasons:

1. The expected call is often the better call. On third and 1 the run is expected and
   it also converts; on third and 12 the pass is expected and it is the only real try.
   Situations differ both in what is expected and in what each play is worth.
2. Good offenses differ from bad ones. A team with an elite quarterback passes more and
   its passes are both more expected and better. Across teams, predictable passes then
   look good.
3. Opponents differ. A team facing a weak run defense runs more and gains more.

Worked example of the bias (by hand, and it is the synthetic data in
`test_within_team_fixed_effects_recover_a_known_cost_that_the_raw_slope_misses`):
every play loses 0.2 EPA per unit of P(call), but good offenses are more predictable and
situations where the call is obvious are also worth more. The raw slope comes out
positive ("predictable calls do better") although the true effect is negative. The test
checks that the within design below recovers -0.2.

### The within-team design

The fix is to compare plays only with plays that share the confounders. We estimate,
separately for runs and for passes,

    EPA = effect of P(call) + situation effect + offense-season effect
          + defense-season effect + noise

where "situation" is a group of question 2's cell (down, distance bin, field-position
bin) by score bucket (within the neutral range) by half, "offense-season" is one effect
per offense per season and "defense-season" one per defense per season. The P(call)
effect is read either as a curve (one effect per bin of P(call)) or as a straight-line
slope. In words: the same offense, in the same season, in the same kind of situation,
against an opponent of the same quality: did its more expected calls gain less?

With hundreds of effects per set, we solve by alternating projections: start every
effect at zero; replace the situation effects by the average of what the other effects
leave unexplained; then the offense effects; then the defense effects; repeat until
nothing moves. For a slope we use the Frisch-Waugh-Lovell theorem: remove the fixed
effects from EPA and from P(call) separately, then regress residual on residual. That
gives exactly the same slope as the full regression and converges much faster (the
direct sweep needed 765 rounds on the real data, the residual version about 60).

Intervals: whole games are resampled 200 times and the entire fit is redone on each
resample (warm-started from the full-sample fit), so the uncertainty in the fixed effects
is included.

What this identifies: within an offense-season, situation group and opponent quality, the
association between how expected a call was and what it gained.

What it does not identify:

- Information the defense has and we do not: formation, personnel, motion, the play
  clock. Our P(call) is a lower bound on how predictable the call really was, and the
  part of predictability that only formation reveals is missing (the shotgun and
  no-huddle variant gives a feel for its size).
- Information the offense has and we do not: the front it sees at the line. If an
  offense checks out of the expected call when the defense is set up for it, the
  surprising calls are selected to be good ones. That inflates the measured cost.
- Within-cell differences we absorb only partly: a cell is a range of distances and
  yard lines; the clock inside a half and the exact score inside a bucket still vary.
- It is a description of today's range of behavior. It does not say what would happen
  if an offense became much less predictable, because defenses would respond.

### Shrinkage of team estimates

Each offense-season gets two numbers: its excess predictability (its average P(call)
minus the league average for the same call in the same situation group) and the cost of
that excess in EPA per game (minus the league slope times its summed excess, divided by
games). Each comes with a standard error from resampling its games.

Raw team estimates spread out more than the true values, because each carries its own
noise. Empirical Bayes shrinkage assumes the true values come from a normal
distribution with mean mu and spread tau, and each estimate is the true value plus
noise with standard error se. Then

    tau^2 = variance of the estimates - average se^2       (floored at 0)
    shrunk = mu + tau^2 / (tau^2 + se^2) * (estimate - mu)

The factor tau^2 / (tau^2 + se^2) is the reliability: how much of an estimate's distance
from the average is signal. One minus the average se^2 over the variance of estimates is
the noise share: how much of the spread across teams is sampling noise.

Worked examples (by hand): estimates spread with variance 0.0100; average se^2 0.0064.
tau^2 = 0.0036, tau = 0.06, noise share 64%. A team at mu + 0.20 with se 0.08 has
reliability 0.0036 / (0.0036 + 0.0064) = 0.36 and is shrunk to mu + 0.072. A team with
se 0.03 has reliability 0.80 and keeps most of its distance (mu + 0.16).

Season-to-season persistence (the correlation of a team's value with its own value the
next season) is a second check that does not depend on the normal model.

## 2. How predictable are the calls?

Five call models, all using only pre-snap information. Training starts in 2011 (2010 is
history only, so tendencies exist); each model early-stops on the last training season.
Validation blocks: 2020 to 2021 (trained 2011 to 2018, stopped on 2019) and 2022 to 2023
(trained 2011 to 2020, stopped on 2021). Test: trained 2011 to 2022, stopped on 2023,
scored once on 2024 to 2025.

| Model (2024 to 2025 test) | Log loss | Brier | AUC | Calibration error |
|---|---|---|---|---|
| Base rate only | 0.672 | | | |
| Boosting, situation and earlier-games tendencies | 0.525 | 0.179 | 0.795 | 0.009 |
| Boosting, plus hand-made play history | 0.515 | 0.174 | 0.806 | 0.008 |
| Boosting, plus history, shotgun and no-huddle | 0.495 | 0.166 | 0.827 | 0.011 |
| Network, situation only | 0.533 | 0.181 | 0.789 | 0.007 |
| Network plus GRU over the game | 0.519 | 0.175 | 0.805 | 0.012 |

Paired differences in log loss per 1,000 calls (negative = the first model is better):

| Comparison | 2020 to 2023 validation | 2024 to 2025 test |
|---|---|---|
| Boosting + history minus boosting, situation | -10.7 [-11.8, -9.8] | -9.7 [-11.2, -8.2] |
| GRU network minus same network without history | -16.0 [-17.3, -14.7] | -14.7 [-16.5, -12.6] |
| GRU network minus boosting + history | +5.3 [4.2, 6.3] | +3.2 [2.0, 4.6] |
| Network minus boosting, both situation only | +10.5 [9.5, 11.5] | +8.3 [7.0, 9.5] |
| Shotgun and no-huddle added to boosting + history | -21.5 [-23.0, -19.9] | -19.9 [-22.4, -17.5] |

Reading. History adds real but modest information: the defense learns roughly 0.01
nats per call from what happened earlier, against 0.15 from the situation. The GRU
gains more over its own baseline than boosting does over its own, but its baseline is
weaker (the network is worse than boosting on the situation alone), and it ends up
behind. With a few hundred thousand rows and tabular features this is the usual result:
boosting is hard to beat, and a sequence model earns its place only if the order of
events carries information that simple summaries miss. Here the previous play, its
yards and the pass share so far carry most of it. All models are well calibrated
(errors around 1 point), which matters because P(call) is used below as a measurement.
History helps most on second down (0.553 to 0.535 nats on validation), where the
first-down result changes the call; on third down the distance already decides.

The most important features of the history model (share of gain): yards to go 0.20,
down 0.16, score 0.10, seconds left in the half 0.10 and in the game 0.09, yards on
the previous play of the drive 0.05, the drive's pass share so far 0.04, field position
0.04.

## 3. The cost of predictability

Neutral plays on downs 1 to 3, P(call) from the history model fitted on earlier seasons
only. EPA by bin of P(call), development seasons 2013 to 2023:

| P(call) | under 0.2 | 0.2-0.35 | 0.35-0.5 | 0.5-0.65 | 0.65-0.8 | over 0.8 |
|---|---|---|---|---|---|---|
| Runs, share of runs | 4% | 9% | 21% | 40% | 22% | 4% |
| Runs, raw EPA | -0.087 | -0.091 | -0.077 | -0.094 | -0.061 | -0.000 |
| Runs, within situation | -0.020 | -0.064 | -0.062 | -0.088 | -0.097 | -0.086 |
| Runs, within team | -0.018 | -0.064 | -0.061 | -0.084 | -0.099 | -0.114 |
| Passes, share of passes | 1% | 7% | 22% | 19% | 16% | 36% |
| Passes, raw EPA | -0.050 | +0.070 | +0.099 | +0.115 | +0.087 | -0.035 |
| Passes, within situation | -0.116 | -0.022 | -0.001 | +0.028 | +0.047 | +0.108 |
| Passes, within team | +0.017 | +0.061 | +0.035 | +0.026 | +0.045 | +0.070 |

"Within team" holds the situation group, the offense-season and the defense-season
fixed. The values are the bin effects plus the overall mean, so they read as EPA at the
average situation, offense and opponent. Intervals are in the JSON and the headline
figure.

Slopes (EPA per unit of P(call); divide by 10 for "per 10 points"):

| | Raw | Within situation | Within team | Test, within team |
|---|---|---|---|---|
| Runs, EPA | +0.062 [0.020, 0.113] | -0.131 [-0.198, -0.072] | -0.149 [-0.244, -0.109] | -0.130 [-0.312, -0.031] |
| Runs, success | +0.176 | -0.099 | -0.098 [-0.150, -0.074] | -0.109 [-0.205, -0.048] |
| Passes, EPA | -0.242 [-0.280, -0.208] | +0.262 [0.188, 0.348] | +0.052 [-0.036, 0.115] | -0.041 [-0.289, 0.165] |
| Passes, success | -0.197 | +0.085 | +0.016 [-0.017, 0.043] | -0.030 [-0.108, 0.041] |

The table is a lesson in confounding. Raw, expected runs look good (+0.062) and expected
passes look bad (-0.242). Both come from situations: expected runs are short-yardage
runs, which convert; expected passes are third and long, where every play is bad. Within
the situation the signs flip. Then, for passes, holding the offense fixed takes most of
the remaining slope away (+0.262 to +0.052): good passing offenses are both more
predictable passers and better passers, which is reason 2 of section 1. For runs the
team adjustment changes little, and the within-team cost is the most robust number here.

Sensitivity to the predictor (within team, EPA slope, development seasons): situation
only model, runs -0.236 [-0.341, -0.153], passes +0.047 [-0.057, +0.166]; with shotgun
and no-huddle, runs -0.131 [-0.179, -0.105], passes -0.119 [-0.181, -0.059]. The run
cost holds whichever predictor is used. A pass cost appears only when the predictor knows
the formation, which is consistent with the defense reading passes partly from
formation.

## 4. By team

Per offense-season (neutral downs 1 to 3; team-seasons with at least 150 such plays):

- Log loss of the model on the team's calls ranges from 0.49 to 0.65 nats per call on
  development seasons. Most of that range is the situations the team was in and its pass
  rate, not predictability: a team that passes 68% of the time is "predictable" on
  passes simply by passing a lot.
- Excess predictability (average P(call) minus the league average for the same call in
  the same situation group) has a standard deviation of 0.0074 across team-seasons, under
  one point of probability. Shrinkage: tau = 0.0058, noise share 39%, mean reliability
  0.63; season-to-season correlation 0.38. Most predictable after shrinkage: Kansas City
  2022 (+2.0 points) and 2021, Detroit 2016; least: New England 2019 (-1.5), Kansas City
  2014, Philadelphia 2022. On the test seasons: Dallas, Miami and Cincinnati 2025 most,
  Indianapolis, Philadelphia and Atlanta 2025 least; noise share 36%.
- Cost in EPA per game, priced with the league slopes: shrunk values from -0.27 to +0.22
  on development seasons (standard deviation 0.10; tau 0.10), from -0.07 to +0.06 on the
  test seasons (standard deviation 0.03). Only 1 of 64 test team-seasons has an interval
  that excludes zero. The ranking correlates -0.72 with pass rate (-0.60 on test): the
  pass-heavy offenses (Kansas City 2021 and 2022) come out as "gaining" from their
  predictability because their runs are surprises, and the run cost is the only firmly
  measured slope. Read the figure as "who runs when the defense does not expect it", not
  as a coaching grade.
- Does the cost differ by offense? Team-season slopes of EPA on P(call) (within team,
  pooled over runs and passes) spread with a standard deviation of 0.55 per unit, but the
  average standard error is 0.50: 88% of the spread is noise and the estimated true spread
  is tau = 0.19 per unit (0.019 EPA per 10 points). We cannot say which offenses pay more
  per unit of predictability.
- Predictability and success: see section 0, point 4. Prior-games passing efficiency
  (+0.47 EPA per play per unit [0.40, 0.55]), running efficiency (+0.24) and the
  opponent's prior-games EPA allowed (+0.50) predict a game's EPA per play; excess
  predictability adds nothing detectable.

## 5. "If everyone knows you run on first down, why keep doing it?"

On neutral first and 10, development seasons (77,581 plays): run share 53%; average
P(run) of the runs 0.555; average P(pass) of the passes 0.489. So the premise is weak
at the level we can measure: a defense with our information is only a little better than
a coin flip on first down (log loss 0.673 nats, against 0.693 for a coin). 18% of
first-down runs are clearly expected (P(run) of 0.65 or more).

The cost: within team, EPA per first-down run changes by -0.003 per 10 points of P(run)
[-0.014, +0.002] on development seasons and -0.022 [-0.045, -0.005] on 2024 to 2025.
Raw, expected first-down runs (P(run) 0.65 or more) averaged -0.103 EPA and surprising
ones (under 0.35) -0.132 on development seasons; -0.126 and -0.016 on the test seasons
(only 236 surprising runs). A first-down run is a below-average play whether or not the
defense expects it; predictability explains at most a tenth of question 2's 0.205 gap.

By team: the offenses whose first-down runs were most expected (average P(run) of those
runs) were Pittsburgh (0.635) and Cleveland (0.628) in 2023, and Cleveland (0.623),
Atlanta, Miami and New Orleans (about 0.59) in 2025; the least, Kansas City (0.472 in
2025), New England and the Chargers (about 0.50). Teams that pass more on first down
have less expected runs (correlation of first-down pass rate with average P(run) of the
runs: -0.60 on development seasons). Team differences in first-down run excess
predictability are mostly noise (54% development, 80% test; season-to-season
correlation 0.24 and 0.07), and the cost per game is a few hundredths of a point at most.

So the honest answer to the fan question: first-down runs are not as obvious as the
complaint assumes once the defense cannot see the formation, and being expected costs a
first-down run little. The case against first-down runs is question 2's: they gain less
than passes in the same spot, expected or not.

## 6. What this means for the debate

- "The run sets up the pass" (Kollmann and others, `research-kit/x/2_twitter_findings.md`
  section 7, quoted from posts) predicts that passes lose value when they become
  predictable. Within team, with the information we have, they do not; with formation
  they lose about 0.012 EPA per 10 points. Either way it is small next to the 0.18 to
  0.20 early-down pass advantage of question 2.
- "Pass more, defenses will adjust eventually" (Walder) is not tested here: we measure
  the cost of being predictable at today's mix, call by call. Nothing here says what
  happens at an 80% pass rate. No balance point is claimed.
- The cleanest finding is the run cost: the same offense loses about 0.10 EPA per run
  from its least to its most expected runs. That fits question 2's one clean sign (a
  predictable second and 10 pass after a no-gain run lost 0.071) in spirit, but here it
  appears for runs, not passes.

## 7. Design decisions someone could challenge

1. P(call) comes from a model, not from the defense. It is a stand-in for what a
   well-informed defense believes. Without formation and personnel it understates
   predictability; with a model fitted on many seasons it may overstate what a defense
   knows about one offense.
2. The primary predictor is boosting with hand-made history, fixed before the test
   seasons were read, because it was the best on validation and can be refitted
   season by season. The GRU is reported, not used, for the cost analysis.
3. The cost analysis uses forward predictions: each season's P(call) from a model fitted
   on earlier seasons only. A cross-fitted model would use more data per season but
   would let later seasons inform earlier "defenses".
4. Tendencies decay by game (0.92 per game, half weight across a season boundary) and
   are shrunk toward fixed priors with 40 pseudo-plays. These constants were set, not
   tuned.
5. The situation group (question 2's cell by score bucket by half) is coarse within a
   cell; residual differences in distance, yard line and clock could still move both
   P(call) and EPA.
6. Within-team means offense-season and defense-season fixed effects. Changes within a
   season (injuries, a new quarterback) are not absorbed.
7. Team cost = league slope times the team's excess. Run and pass slopes have opposite
   signs (or one is zero), so the team ranking is mostly about runs and correlates with
   pass rate. A team-specific slope was estimated as a check and is mostly noise.
8. Offense quality is controlled with prior-games EPA per pass and per run, our stand-in
   for the quarterback: quarterback identity is not in our data.
9. Bootstrap intervals resample games and refit every fixed effect, but they hold the
   call model fixed; uncertainty in P(call) itself is not propagated (the sensitivity
   runs show how much the predictor matters).
10. The first full run was stopped during the test stage, before any test number was
   computed or read, to fix a bug: with a single set of fixed effects the residuals
   were not centered, which biased the "within situation" slopes (not the within-team
   ones). The rerun's test load is logged with that reason in
   `results/test_access_log.jsonl`; the first load is also there.

## 8. Limits and what is unfinished

- Observational: P(call) is not assigned at random. The offense sees the front and can
  check out of an expected call into a surprising one when the surprise is likely to
  work. That makes surprising calls look better and overstates the cost.
- No formation or personnel beyond nflverse's shotgun and no-huddle flags, used only as a
  sensitivity variant. Real predictability is higher than ours.
- Our EPA is nflfastR's model; success is EPA above zero.
- The test seasons are only two, so team-level test results are noisy (1 of 64 cost
  intervals excludes zero).
- Not done: a sequence model refitted season by season for the cost analysis; opponent
  (defense) tendencies as features; win probability instead of EPA as the outcome.
