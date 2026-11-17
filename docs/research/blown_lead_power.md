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
- The size at 16 seasons is 0.077 against 0.05 (Monte Carlo SE about 0.015, and the
  critical value is itself estimated from 600 leagues). Rerun with more null leagues
  before quoting it.
- Temperature scaling is one kind of miscalibration. A model that is wrong only in some
  game states (M3b's mid-range fault) may be harder to detect; A3 addresses that.
- Games are independent here; real games share teams. The simulator is a Brownian lead,
  not football.
