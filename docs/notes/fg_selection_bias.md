# Field-goal selection bias: who kicks, in what conditions, and what an average kicker makes

Code: `src/fourth_and_rl/kicker.py` (model, decomposition, support rule, planner interface),
`scripts/questions/fg_selection_bias.py` (the three steps). Config: `configs/kicker.yaml`.
Tests: `tests/test_kicker.py`. Figure: `assets/questions/fg_selection_bias.png`.

Every number below comes from one of three files written by that script:
- `results/questions/fg_selection_bias.json` (development: training fit, calibration check),
- `results/questions/fg_selection_bias_preregistration.json` (written before the test check),
- `results/questions/fg_selection_bias_test.json` (the one check on 2024 to 2025).

The world-model numbers in section 1 are not ours. They are quoted from
`results/world_model/colab_t4_v2/outputs/plan.json` on branch `claude/world-model-colab`
and are never mixed with this model's numbers.

**Post hoc.** This model, its support distance and the cap or override it offers the
planner were designed after seeing the world-model run. They are not part of that run's
preregistered plan.

## 1. Why this exists

The world-model run compared its simulated field-goal make rate with the training rate by
band of kick distance (distance = yardline_100 + 18 in that file):

| Band (yards) | Training attempts | Training rate | Simulated rate |
|---|---|---|---|
| 54 to 58 | 541 | 0.566 | 0.676 |
| 59 to 63 | 82 | 0.378 | 0.521 |

(Source: world-model branch, file above. The bands in that file are 54 to 58 and 59 to 63,
not 56 to 60 and 61 to 65 as they were described in the request.)

The simulator looks too optimistic. But the training rate is itself a selected number.
Coaches send strong-legged kickers out for long tries, mostly indoors or in calm weather,
which pushes the raw rate up. End-of-half heaves are kicked only because time ran out,
which pushes it down. So "simulated 0.68 against raw 0.57" could be simulator error, or it
could be the raw rate understating or overstating what an average kicker makes. This
work measures which.

## 2. Concepts

### 2.1 Selection bias in a make rate

The raw make rate at 58 yards is the mean of `made` over the attempts that coaches chose to
take from 58 yards. It answers "how often did the kicks that were tried go in", not "how
often would an average kicker make a 58-yarder". The two differ whenever the choice to
kick depends on something that also affects the make: kicker strength, roof, wind, or
the clock.

Worked example 1 (toy numbers). Two kickers, A with true 60-yard make rate 0.50 and B with
0.30. A league-average kicker sits at 0.40. If coaches let A try 9 of every 10 sixty-yard
kicks, the raw rate is 0.9 x 0.50 + 0.1 x 0.30 = 0.48, eight points above the average
kicker, with no error anywhere. The data are right; the question is wrong.

Worked example 2 (toy numbers). Add heaves: one in three 60-yard tries is an end-of-half
heave by whichever kicker the team has, made 0.25 of the time. Then the raw rate is
(2/3) x 0.48 + (1/3) x 0.25 = 0.403. The two biases almost cancel, and the raw rate looks
"right" for the wrong reason. Section 4 shows the real data come close to this.

### 2.2 A hierarchical (mixed-effects) logistic regression

For attempt i by kicker k, with z = (distance - 45) / 10:

    log-odds(make) = s(distance)
                     + b_out * outdoor + b_open * open_roof
                     + outdoor * (b_temp * temp_c + b_wind * wind_c + interactions with z)
                     + b_season * season_t + b_sz * season_t * z
                     + b_heave * heave
                     + gamma * (usage_k - usage_league) / 10
                     + u_k + v_k * z
    (u_k, v_k) ~ Normal(0, Sigma)

- s(distance) is a natural cubic spline: cubic pieces joined smoothly at knots, and a
  straight line in log-odds beyond the outer knots. The straight-line tail matters at long
  range, where data are thin: the curve cannot bend back up.
- u_k is kicker k's own shift in log-odds; v_k would let his curve be steeper or flatter.
- Sigma is how much kickers differ. It is estimated from the data, and it decides how far
  each kicker's estimate is pulled toward the league average (shrinkage).

**Shrinkage, in one formula.** For one kicker with information I (roughly the number of
attempts times p(1 - p)) and between-kicker variance s^2, the estimate of u_k is about
(I s^2 / (I s^2 + 1)) times his raw deviation. Worked example 3: s = 0.15 (our estimate) and
300 attempts at p about 0.85 give I about 300 x 0.13 = 38, so the factor is
38 x 0.0225 / (38 x 0.0225 + 1) = 0.46. Even a kicker with 300 attempts keeps only about
half of his raw deviation. A kicker with 30 attempts keeps about 8%.

**Fitting (empirical Bayes).** For a given Sigma, the fixed effects and all kicker effects
are found together by maximizing the penalized log likelihood

    sum_i [y_i * eta_i - log(1 + exp(eta_i))] - 0.5 * sum_k b_k' Sigma^-1 b_k

by Newton's method. Each attempt has one kicker, so the kicker part of the Hessian is block
diagonal (2 by 2 per kicker) and the fixed effects are solved with a Schur complement; a
fit takes about 2 seconds. Sigma is then chosen to maximize the Laplace approximation of
the marginal likelihood (the likelihood with the kicker effects integrated out):

    log L(Sigma) ~= l(beta, b) - 0.5 sum_k b_k' Sigma^-1 b_k - (K/2) log|Sigma|
                    - 0.5 sum_k log|H_k|

where H_k is kicker k's 2 by 2 block of the negative Hessian. Only numpy and scipy are used.

**Intervals.** The inverse of the joint negative Hessian at the mode is the approximate
posterior covariance of (beta, b) given Sigma. Curve intervals and the 1,000 draws behind
the decomposition intervals come from it. It ignores the uncertainty in Sigma itself, so
the intervals are somewhat too narrow.

### 2.3 Why random effects alone are not enough here: the Mundlak term

A random-effects model assumes a kicker's effect is unrelated to the covariates. Selection
breaks exactly that assumption: strong kickers are the ones given long kicks, so u_k is
correlated with distance. The fit then explains part of "strong kickers make long kicks"
with a flatter distance curve instead of with kicker strength, and the average-kicker
curve comes out too high at long range.

The fix is the Mundlak (correlated random effects) device: add each kicker's mean attempt
distance (his "usage") as a fixed covariate. Usage absorbs the part of strength that goes
with being sent out long, and the distance curve is then learned from comparisons within
kickers. The league-average kicker has league usage (37.9 yards, the mean over training
attempts) and zero random effects.

Worked example 4 (synthetic, from `synthetic_selection_world` in the results file): 80
kickers, 150 attempts each, strength u ~ Normal(0, 0.5), attempt distance
Normal(38 + 12u, 9). On attempts of 55+ yards over five seeds:

| Quantity | Range over 5 seeds |
|---|---|
| Raw rate | 0.572 to 0.602 |
| True average-kicker rate | 0.462 to 0.491 |
| Model with usage term | 0.452 to 0.498 (within 0.016 of the truth) |
| Plain random effects | 0.506 to 0.544 (0.03 to 0.07 too high) |

The raw rate is 10 to 13 points too high; plain random effects remove only part of it;
the usage term removes nearly all of it. In the real data the term changes little
(section 4.4), because real kickers' usage differs by only about 1 yard.

### 2.4 Shapley decomposition

To say how much of the raw rate is "who kicks" and how much is "conditions", define for any
set S of factors

    v(S) = mean predicted P(make) on the same attempts, with factors in S at their observed
           values and the others set to neutral.

Factors: kicker (own usage and effects, or the average kicker), conditions (observed roof
and weather, or outdoors at median weather), heave (observed, or not a heave), era (own
season, or the mean training season). v(all) is the model's fit and v(none) is the
"selection-free" rate: an average kicker, neutral conditions, not a heave, mid-era, from the
same distances. Switching factors off one at a time gives answers that depend on the order,
so each factor gets its Shapley value: its marginal effect averaged over every order. The
four values add up exactly to v(all) - v(none).

Worked example 5 (two factors, toy numbers). v(none) = 0.40, v(kicker) = 0.46,
v(heave) = 0.36, v(both) = 0.43. Kicker: in the order (kicker, heave) it adds 0.06; in the
order (heave, kicker) it adds 0.43 - 0.36 = 0.07; Shapley value 0.065. Heave: -0.04 and
0.43 - 0.46 = -0.03; Shapley value -0.035. Sum 0.03 = 0.43 - 0.40.

### 2.5 The heave flag

A heave is a field-goal attempt with at most 10 seconds left in the first half or in
regulation (never overtime), with no later snap or kickoff in that half: the kick was the
last play, taken because time ran out. 916 of the 12,409 training attempts qualify. The
flag includes short end-of-half kicks too; its coefficient measures the cost of a
time-forced kick at any range.

## 3. Decisions and why

1. **Training seasons only (2010 to 2021).** `FGMakeModel.fit` raises `LeakageError` for
   any season outside the training split, so calibration seasons (2022 to 2023), test
   seasons and the 2026 holdout are all refused, through `splits.assert_no_test_seasons`
   plus a training-season check. The kicker id download refuses the holdout.
2. **Kicker ids from nflreadpy.** The state table has no kicker column. `kicker_player_id`
   and `kicker_player_name` are cached per season under `data/kicker/` and joined by game
   and play id (the pattern of `coaches.download_penalties`). nflreadpy 0.1.5 is still the
   current release, and the nflverse data dictionary confirms the columns used:
   `kicker_player_id`, `kick_distance`, `field_goal_result`, `roof`, `temp` and `wind`
   (temp and wind only for "outdoors" or "open"), `game_seconds_remaining` and
   `half_seconds_remaining`.
3. **Distance is `kick_distance`**, with yardline_100 + 18 when it is missing. The two agree
   on 91.0% of training attempts. The planner should pass yardline_100 + 18, the repo
   convention.
4. **Roof has three levels.** Every "open" (retractable roof left open) attempt has no
   weather recorded, so a "weather missing" flag turned out to be an open-roof flag. The
   model has an outdoor flag and a separate open-roof flag. Weather terms apply to fully
   outdoor games. The 22 outdoor training attempts with no weather get the training median
   (60 F, 8 mph). In 2022 to 2023, 441 outdoor attempts have no weather; they get the same
   fill.
5. **Knots chosen by AIC on the training seasons.** The first knot set (23, 32, 40, 47, 55)
   predicted 0.365 on the 78 training attempts of 60+ yards, against a raw 0.256. That is
   an in-sample check, so a short list of five candidate knot sets was added and the set
   with the lowest AIC (-2 x Laplace log marginal + 2 x fixed effects) is used: (23, 32, 40,
   47, 55, 60), AIC 9272.7 against 9278.7 for the first choice, predicting 0.277 on those 78
   attempts. Calibration-season log loss for every candidate is reported but was not used
   to choose. This change came after the first fit and is recorded as such in the config.
6. **The usage (Mundlak) term was added after the synthetic recovery test failed without
   it** (section 2.3). It is the textbook fix for the failure the test exposed, not a
   tuning choice.
7. **The kicker slope is in the model but the data do not support it.** Its standard
   deviation is 0.0035 log-odds per 10 yards, effectively zero: kickers differ by a
   constant shift in log-odds, not by how steeply their curves fall. The correlation
   between intercept and slope is therefore not identified and is reported as null.
8. **Monotone by construction for the planner.** `p_make` caps each value by the smallest
   value at any shorter whole-yard distance. On the training fit this changes nothing (the
   largest change is 2e-16); it guards against a spline wiggle at short range in other
   fits, which the synthetic world does produce.
9. **Planner defaults.** Season 2021 (the last training season, matching the M1 field-goal
   model's convention), and when the weather is not given, an average over the weather of
   every training game. A planner that does not know the weather should average the make
   probability over weather, not plug in the median weather: P is not linear in wind.

## 4. Results (training seasons 2010 to 2021)

12,409 attempts, 114 kickers (47 with 100 or more attempts), 916 heaves.

### 4.1 What matters for a make

Fixed effects in log-odds (standard errors in brackets):
- Wind outdoors: -0.211 (0.058) per 10 mph. A 20 mph wind takes a 55-yard kick for an
  average kicker from 0.587 (weather average) to 0.512.
- Heave: -0.234 (0.093). At 55 yards, 0.530 against 0.587.
- Open retractable roof: +0.596 (0.265). Outdoor against dome: -0.012 (0.079), no
  difference. Temperature: 0.010 (0.019) per 10 F, no detectable effect.
- Season: +0.151 (0.085) per decade at 45 yards, and -0.215 (0.096) per decade per 10 yards
  of distance. Inside 2010 to 2021, short and mid-range kicks improved, but long kicks did
  not improve once kicker, conditions and heaves are held fixed.
- Usage: +0.441 (0.233) per 10 yards of a kicker's mean attempt distance.
- Kicker spread: standard deviation of the random intercept 0.150; of total 50-yard
  strength (usage part plus random part) 0.158. Likelihood ratio for the kicker effects
  5.85 (p = 0.035 against the boundary mixture of chi-squares). Kickers differ, but less
  than commentary suggests: at 55 yards a top-decile kicker makes 0.623 and a bottom-decile
  kicker 0.524 (neutral conditions, 2021).

### 4.2 Curves: raw against average kicker

"Raw" is the training make rate within 2 yards of the distance. "Selection-free, same
attempts" is v(none) on exactly those attempts. "Average kicker" is the curve at that exact
distance, neutral conditions, 2021.

| Distance | Raw (attempts) | Model fit, same attempts | Selection-free, same attempts | Average kicker, neutral, 2021 (90% interval) | Top decile | M1 model |
|---|---|---|---|---|---|---|
| 50 | 0.692 (1,726) | 0.698 | 0.681 | 0.684 (0.657 to 0.710) | 0.726 | 0.721 |
| 55 | 0.629 (797) | 0.620 | 0.600 | 0.574 (0.532 to 0.615) | 0.623 | 0.608 |
| 60 | 0.415 (118) | 0.407 | 0.404 | 0.352 (0.285 to 0.425) | 0.400 | 0.448 |
| 65 | 0.214 (28) | 0.206 | 0.216 | 0.164 (0.099 to 0.259) | 0.194 | 0.257 |

Named kickers (shrunken, neutral conditions, 2021) at 50, 55, 60 and 65 yards: J.Tucker
(strongest) 0.766, 0.672, 0.453, 0.231; G.Zuerlein (most 55+ attempts, 35) 0.700, 0.593,
0.369, 0.174; D.Akers (weakest ranked) 0.644, 0.529, 0.311, 0.140.

Two gaps in the table have different causes. Raw against selection-free on the same
attempts is selection (section 4.3): small. Selection-free against the 2021 curve is the
era term and the shape of the curve inside the 4-yard window: long kicks are slightly
lower at 2021 than mid-era in this model.

### 4.3 The selection-bias decomposition

On the 443 training attempts of 55+ yards:

| Source | Contribution to the raw rate (90% interval) |
|---|---|
| Raw rate | 0.497 |
| Model fit | 0.504 (fit error -0.007) |
| Who kicks | +0.016 (0.007 to 0.026) |
| Roof and weather | +0.017 (0.005 to 0.028) |
| Heaves | -0.019 (-0.032 to -0.008) |
| Era | -0.002 (-0.006 to 0.002) |
| Selection-free rate | 0.492 (0.456 to 0.528) |
| Total selection | +0.012 (-0.008 to 0.031) |

Who kicks and the conditions together push the raw 55+ rate up by about 3.3 points; heaves
(36.3% of these attempts) push it down by about 1.9. The net selection bias is about 1
point, and its interval includes zero. The 60+ band shows the same pattern more strongly:
70.5% heaves, kicker +0.019, conditions +0.011, heaves -0.031, net -0.002.

**Consequence for the world model.** On the world-model bands (yardline_100 + 18), the
selection-free rate is 0.569 for 54 to 58 and 0.344 for 59 to 63, and the planner's
average-kicker rate (weather average, 2021) is 0.545 and 0.323; even a top-decile kicker is
0.594 and 0.369. The simulated rates in that run (0.676 and 0.521) are above every one of
these. Selection does not explain the gap: the simulator is too optimistic at long range.

### 4.4 Usage term on the real data

Without the usage term the selection-free 55+ rate is 0.494 (with it, 0.492) and the
average kicker at 55 yards is 0.580 (with it, 0.574). Usage across established kickers has
a standard deviation of only 0.96 yards, so real selection by kicker is mild and the term
matters little here. It is kept because it is what makes the model correct when selection
is strong (section 2.3).

### 4.5 Support distance

Rule, fixed in the config before the fit: the support distance is the largest whole-yard
distance D such that every distance d from 40 to D satisfies both
(a) the top-decile kickers (top 10% by estimated 50-yard strength among kickers with at
least 100 training attempts) made at least 20 training attempts within 2 yards of d, and
(b) the median over those kickers of the width of the 90% interval of their own predicted
make rate at d (neutral conditions, 2021) is at most 0.20.

Result: **60 yards.** The six top-decile kickers are J.Tucker, H.Butker, C.Boswell,
M.Prater, R.Gould and M.Bryant. At 61 yards they made only 14 attempts within the window,
so condition (a) fails. Condition (b) never fails up to 75 yards (largest median width
0.191), because as P(make) falls toward 0 the interval narrows in probability even as it
widens in log-odds. In practice the attempt-count condition is the rule. `in_support`
returns False beyond 60 yards and for any season after 2021.

### 4.6 Calibration seasons (2022 to 2023, checks only)

2,212 attempts, 1,921 by kickers seen in training. Log loss: hierarchical model with own
kicker effects 0.3538, with average-kicker effects 0.3551, M1 distance-and-season model
0.3546. On 50+ yard attempts: observed 0.695, hierarchical 0.623, M1 0.648. Both models
under-predict long kicks after 2021, and ours more, because its negative season-by-distance
term extrapolates the 2010 to 2021 pattern.

### 4.7 The preregistered test check (2024 to 2025, run once)

The preregistration was written before any test number was computed. 2,306 attempts,
1,423 by kickers seen in training.

| Hypothesis | Result |
|---|---|
| H1: lower log loss than the M1 model | **Failed.** 0.3751 against 0.3702 |
| H2: kickers on 55+ attempts are stronger than average (training estimates) | **Passed.** +0.010 log-odds (123 attempts by known kickers) |
| H3: calibrated within 5 points on 50+ attempts | **Failed.** Observed 0.692, predicted 0.593 (gap 0.099, 565 attempts) |

By band, the gap grows with distance: 50 to 54 observed 0.740 against 0.669; 55 to 59
0.653 against 0.519; 60+ 0.421 against 0.223 (38 attempts). Long-range kicking improved
after 2021 far faster than the model's season terms allow; on the 2024 to 2025 55+
attempts the fit error is 0.145, which dwarfs every selection term. The selection terms
themselves keep their sign there (kicker +0.009, conditions +0.019, heaves -0.011).

**What this means.** The model is a good description of 2010 to 2021, which is the era the
world model was trained on, and it answers the selection question for that era. It is not
a forecast of current kickers. For any planner meant to reflect 2024 or later kicking,
neither this model nor the M1 model is adequate at long range, and the season argument of
`p_make` must not be trusted beyond 2021 (`in_support` says so).

## 5. Planner interface

```python
from fourth_and_rl.kicker import FGMakeModel
m = FGMakeModel.load()          # data/models/kicker_fg.joblib, written by the script
m.p_make(distance, strength=None, roof=None, temp=None, wind=None, season=None, heave=False)
m.in_support(distance, season=None)
m.capped_rate(simulated, distance, strength=0.9, level=0.9)
```

- `distance`: yards, scalar or array; pass yardline_100 + 18.
- `strength`: None or 0.5 is the league-average kicker; a float q in (0, 1) is a kicker at
  population quantile q of 50-yard strength (0.9 = top decile), with the usage and slope
  the population pairs with that strength; a string is a kicker id seen in training.
- `roof`: None averages over the weather of every training game; "dome" or "closed",
  "open", "outdoors" (with `temp` in F and `wind` in mph, training medians if omitted).
- `season`: default 2021. Later seasons extrapolate a trend that failed the test check.
- `heave`: True for an end-of-half last-play kick.
- Returns P(make), non-increasing in distance.
- `in_support`: True up to the support distance (60) and for seasons up to 2021.
- `capped_rate` (post hoc): the simulator's rate, capped at the upper end of this model's
  90% interval for a kicker of the given strength, weather-averaged. The planner can
  instead override its rate with `p_make` directly. Both are post hoc changes designed after
  seeing the world-model run.

Examples (weather average, 2021): 55 yards, average kicker 0.587, top decile 0.635; 60
yards 0.365 and 0.414; 65 yards 0.173 and 0.205 (beyond support).

The integration into the world-model branch is not done here.

## 6. What is uncertain or unfinished

- Intervals hold Sigma fixed, so they are somewhat too narrow; there is no bootstrap over
  kickers or games.
- Kicker ability is constant across a career in this model. A kicker who improved or
  declined is averaged.
- Usage can reflect the coach as well as the kicker (an aggressive coach sends any kicker
  out long). The usage coefficient (0.441, standard error 0.233) is not cleanly a
  kicker-strength measure.
- Selection on things not in the data (a kicker's form that week, the holder, the snap,
  the rush) is not handled.
- The season terms describe 2010 to 2021 and fail after it (section 4.7).
- The support rule's interval condition never binds with the 0.20 threshold; the support
  distance is set by the attempt count.
