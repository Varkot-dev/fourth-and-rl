# A1: can the blown-lead test catch an overconfident model?

Code: `src/fourth_and_rl/blown_power.py`, `scripts/research/blown_power.py`.
Config: `configs/blown_power.yaml`. Tests: `tests/test_blown_power.py`.
Every number below is from `results/research/blown_power.json`.
Figure: `assets/research/blown_power.png`.

## Question
M3b found real feeds pass, or nearly pass, the blown-lead tests. Passing means little if
the test could not have failed. Estimand: the probability that each test rejects at the
5% level, for a feed with known overconfidence, at league sizes of 1, 4, 8 and 16 seasons.

## Method
- True paths: the M3b simulator (163 reported plays per game, Gamma gaps with shape 1.2,
  both fitted to real nflfastR paths). Pregame value Phi(Z), Z ~ Normal(0, 0.55).
- Miscalibrated feed: q = expit(k * logit(p)). k = 1 calibrated, k = 1.1 means every
  log-odds is 10% too far from even.
- Tests: (1) one-sided KS against the continuous formula; (2) first-crossing test at
  0.6, 0.7, 0.8, 0.9 and 0.95, statistic max |z|, critical value from 600 simulated
  calibrated leagues per size. Power from 300 leagues per size, all k on the same leagues.

## Results
Rejection rate (first-crossing / continuous KS):

| k | 1 season | 4 seasons | 8 seasons | 16 seasons |
|---|---|---|---|---|
| 0.9 (under) | 0.22 / 0.00 | 0.71 / 0.00 | 0.97 / 0.00 | 1.00 / 0.00 |
| 1.0 (size) | 0.05 / 0.00 | 0.05 / 0.00 | 0.05 / 0.00 | 0.08 / 0.00 |
| 1.05 | 0.10 / 0.01 | 0.23 / 0.00 | 0.43 / 0.00 | 0.80 / 0.00 |
| 1.1 | 0.28 / 0.01 | 0.67 / 0.00 | 0.97 / 0.00 | 1.00 / 0.00 |
| 1.2 | 0.73 / 0.09 | 1.00 / 0.24 | 1.00 / 0.43 | 1.00 / 0.73 |
| 1.3 | 0.95 / 0.37 | 1.00 / 0.89 | 1.00 / 1.00 | 1.00 / 1.00 |

1. **The continuous-formula KS test is nearly blind to mild overconfidence.** Its size is
   about 0, not 5%, because discrete paths hide peaks (M3b), and that slack absorbs
   overconfidence up to about k = 1.1 at every league size. It never detects
   underconfidence, by design.
2. **The first-crossing test is far more powerful.** With 16 seasons it detects k = 1.05
   80% of the time and k = 1.1 always; with 4 seasons it catches k = 1.1 two times in three.
   It also detects underconfidence.
3. **Reading the real result.** M3b's first-crossing test on 16 seasons found only small
   departures. Under this simulator, a model 10% overconfident in log-odds would almost
   surely have been caught. The next step (A2) turns this into an interval for k for each
   real model.

## Limits
- The size at 16 seasons was 0.077 in the main run (600 null leagues). A recheck with
  1,500 null and 1,500 test leagues (`results/research/blown_power_size_check.json`)
  gives 0.063, Monte Carlo SE 0.006. The critical value is estimated from 1,500 leagues,
  which adds about 0.006 more, so the combined SE is about 0.008 and 0.063 is within
  about 1.5 SE of 0.05. Read the size as close to 5%, possibly slightly above.
- Temperature scaling is one kind of miscalibration. A model that is wrong only in some
  game states (M3b's mid-range fault) may be harder to detect; A3 addresses that.
- Games are independent here; real games share teams. The simulator is a Brownian lead,
  not football.

# A2: what the real result rules out

Code: `scripts/research/blown_power_invert.py`. Output: `results/research/blown_power_invert.json`,
`assets/research/blown_power_invert.png`. Data: `data/processed/plays.parquet`, 4,350
non-tied games, 2010 to 2025.

## Method
Test inversion. Rescale each real feed q to expit(c * logit(q)) for c from 0.70 to 1.30
and run the first-crossing test (max |z| over 0.6, 0.7, 0.8, 0.9, 0.95) on the rescaled
feed. A feed overconfident by k is calibrated after c = 1 / k, so the c values the test
accepts give a confidence set for k. Critical value: 2.13, simulated for 4,350
independent games in A1. Our own model is not included yet (it needs the M1 out-of-fold
predictions, which were not rebuilt in this session).

Check: at c = 1 the z values reproduce M3b (nflfastR without spread: -2.35 at 0.90 and
-2.73 at 0.95, against M3b's -2.4 and -2.7).

## Results
| feed | z at c = 1 (0.6, 0.7, 0.8, 0.9, 0.95) | best k | max abs z at best | max abs z at k = 1.05 | at k = 0.95 |
|---|---|---|---|---|---|
| nflfastR, no spread | 1.69, 1.80, 0.23, -2.35, -2.73 | 0.99 | 2.29 | 4.67 | 3.38 |
| nflfastR, with spread | 2.18, 2.11, 2.36, -2.03, -1.07 | 1.00 | 2.36 | 3.48 | 4.20 |

1. **Overconfidence of 5% or more in log-odds is ruled out for both feeds**, and
   underconfidence of 5% or more as well: every such k gives max |z| of 3.4 or more,
   far above 2.13.
2. **No single temperature passes.** The best k for each feed (about 1) still has max |z|
   just above the critical value. The signs say why: teams lose slightly more often than
   implied at first crossings of 0.6 to 0.8 (positive z, mild overconfidence in the
   middle) and slightly less often at 0.9 to 0.95 (negative z, mild underconfidence near
   the top). One temperature cannot fix errors in opposite directions, so the remaining
   miscalibration has a shape, not a scale. That is the target of A3.
3. **How seriously to take the rejection at k = 1.** The critical value assumes
   independent games. Real games share teams, which widens the null spread of z, so the
   true critical value is somewhat higher and the marginal rejections (2.29 and 2.36
   against 2.13) may not survive. A4 (team bootstrap) settles this. The 5% statement in
   point 1 is far enough from the line that it should.
