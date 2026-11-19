# M8 notes: "Does the model know our defense is terrible?"

Code: `src/fourth_and_rl/team_strength.py`. Config: `configs/team_strength.yaml`.
Scripts: `scripts/m8_team_strength.py` (ratings and prediction tests),
`scripts/questions/q3_team_aware.py` (how much the call moves).
Tests: `tests/test_team_strength.py`.
Every number below is from `results/m8_team_strength.json` or
`results/questions/q3_team_aware.json`, written by those scripts. Worked examples marked
"by hand" are arithmetic. Figures: `assets/m8_team_ratings.png`,
`assets/questions/q3_team_aware.png`.

## 1. The question

Every public fourth-down tool (nfl4th, ESPN's model, our M2 engine) adjusts for team
strength only through the pregame point spread. Fans keep asking whether that is enough:
should the call change for a great offense facing a bad defense, or for a team whose own
defense cannot stop anyone, so that handing the ball back is worse than usual? Ben Baldwin
answered one such case informally ("their defense is Bad Bad Bad"); nobody has put a number
with honest uncertainty on it. This milestone does.

Short answer, in three parts:

1. Team ratings built only from earlier games DO know something the spread does not about
   how drives end (touchdown, field goal or nothing). The gain is small but clear on
   development seasons and on the 2024 to 2025 test seasons.
2. For short-yardage conversion, the gain over the spread is too small to establish. On
   conversion, the spread already knows nearly everything the ratings can measure.
3. Carried through the decision engine, team strength moves the go-minus-kick margin by
   well under 1.5 win-probability points in a tied first quarter, even for extreme teams.
   That flips only calls the spread-only engine had within about one point. On real
   2024 to 2025 fourth downs it changes 4.3% of calls, half toward going and half away.
   The answer to "does the model know our defense is terrible?" is: the spread knows most
   of it, and the rest can only tip coin flips.

## 2. Concepts

### Why team strength is hard to measure: most of what you see is luck

Take one team-season of short-yardage plays (third and fourth down, 1 to 3 to go). A team
gets about 70 of them. Each one converts or not, with probability about 0.59. Even if the
team were exactly average, its conversion rate over 70 plays would wander by luck alone:

    standard deviation of a rate = sqrt(p (1 - p) / n) = sqrt(0.59 * 0.41 / 70) = 0.059

So pure luck moves a team-season's conversion rate by about 6 points either way. The
research note (`docs/research/public_debate_and_teams.md`, B2) measured the observed spread
of team-season rates at 6.3 points: almost all of it (88%) is that luck. The true
differences between offenses are small, and buried in noise.

Drives are better. A team-season has about 170 drives (70,027 drives over 13 seasons of 32 teams); points per drive vary a lot per
drive (variance about 8.8 points squared, from the drive base fits), but the signal is
larger relative to the noise, as the worked example below shows.

### Regression to the mean and shrinkage

If a team converts 65% when 59% was expected, the best guess of its true rate is not 65%.
The team that looks best in a noisy measurement is usually partly good and partly lucky,
and next time the luck does not repeat. The fix is shrinkage: pull the raw number back
toward the league average by an amount set by how noisy it is.

    shrinkage factor B = tau^2 / (tau^2 + sigma^2 / n)
    estimate = average + B * (raw - average)

tau is the spread of TRUE team effects; sigma^2 / n is the luck variance of the raw number.

Worked example 1 (by hand, conversion). tau = 0.03 (our tuned offense value, 3 points),
70 plays, raw rate 6 points above expected.
sigma^2 / n = 0.242 / 70 = 0.00346; tau^2 = 0.0009; B = 0.0009 / (0.0009 + 0.00346) = 0.21.
Estimate: 0.21 * 6 = 1.2 points above average. A full season of short-yardage plays tells
you only a fifth of what the raw rate seems to say.

Worked example 2 (by hand, drives). tau = 0.30 points per drive (our tuned offense value),
90 drives (about half a season), raw +0.8 points per drive.
sigma^2 / n = 8.8 / 90 = 0.098; tau^2 = 0.09; B = 0.09 / (0.09 + 0.098) = 0.48.
Estimate: 0.48 * 0.8 = +0.38. Half a season of drives already carries half of its face
value. This difference between the two examples is the main reason drive ratings beat the
spread and conversion ratings do not.

Worked example 3 (by hand, across seasons). Carry-over from one season to the next is
rho = 0.5 for drive ratings (tuned). A team that ends a season at +0.8 points per drive
starts the next at 0.5 * 0.8 = +0.4, with its uncertainty mostly restored:
variance = 0.25 * (end-of-season variance) + 0.75 * tau^2.

### The rating method: a Kalman filter (empirical Bayes, one week at a time)

Each game gives one noisy look at "this offense plus that defense": the mean residual of
the offense's plays (or drives) in the game. The filter keeps, for every team, an offense
rating and a defense rating, each with a variance, plus one league-wide level.

    observation y = offense rating + opposing defense rating + league level + noise
    noise variance r = (sum of per-row variances) / n^2

For each observation the filter predicts y, sees the surprise (y minus the prediction),
and moves each rating by its share of the predicted variance. With h the sum of the
covariance columns of the three states involved:

    S = h[off] + h[def] + h[league] + r       (how uncertain the prediction of y is)
    K = h / S                                 (gain: share of the surprise each state takes)
    x <- x + K * surprise,   P <- P - K h'

Worked example 4 (by hand, one drive observation, league level ignored). Our offense
rating 0 with variance 0.09 (tau 0.30); their defense +0.10 with variance 0.04 (tau 0.20);
11 drives with mean residual +1.5, so r = 8.8 / 11 = 0.80.
Prediction 0 + 0.10 = 0.10. Surprise 1.40. S = 0.09 + 0.04 + 0.80 = 0.93.
Offense moves by 0.09 / 0.93 * 1.40 = +0.135; defense by 0.04 / 0.93 * 1.40 = +0.060.
One good game moves the rating by about a tenth of the surprise. That is shrinkage done
game by game: the filter is the sequential form of the empirical-Bayes formula above.

Why this method: it shrinks by exactly the right amount for each team's sample size (a
team with few short-yardage plays moves less); it adjusts for schedule (a big game against
a bad defense is partly credited to the defense); it is online, so a rating can only be
built from games already played; and it has four constants per metric (tau for offense,
tau for defense, carry-over rho, weekly drift), few enough to tune honestly.

Defense ratings are what the defense ALLOWS above average: a bad defense has a POSITIVE
rating. The league level (fixed prior SD 0.02 for conversion, 0.3 points for drives, not
tuned) absorbs league-wide drift, for example scoring rising from one season to the next,
so that it is not credited to teams (`test_league_level_absorbs_a_common_offset`).

### What the ratings are measured against (residuals)

- Conversion: converted (1 or 0) minus the M1 conversion curve for the play's down,
  distance and field position, evaluated at an even spread. Distance and field position
  are therefore already accounted for.
- Drives: offense points on the drive (touchdown 7, field goal 3, returned for a touchdown
  -7, safety -2) minus a cubic curve of expected points by start yard line. Drives that
  start with two minutes or less in a half, and overtime, are left out.

The league curves themselves are fitted on earlier seasons only: the curve for season s
uses seasons 2010 to s - 1, and both test seasons use the curve fitted on 2010 to 2023.
2010 only fits the 2011 curve, so ratings start in 2011 (from zero).

### Why ratings must use only past games

A rating that includes the game being judged leaks that game's outcome into its own
prediction: a team that converted its fourth down today looks like a better converter
today. Season-total ratings joined to every week leak later games too (the research note
measured the "full-season" version inflating the top-versus-bottom gap from 4.8 to 5.4
points). The filter reads every rating for week w before any game of week w updates it.
The tests prove it on synthetic seasons: scrambling every outcome from a cutoff week on
leaves every rating up to that week unchanged
(`test_ratings_never_use_their_own_game_or_a_later_one`, at five cutoffs), and flipping
one game's outcomes leaves that game's own rating and prediction unchanged
(`test_a_games_own_outcome_never_reaches_its_rating`). The script also checks that
running the filter forward through the test seasons leaves every development-season
rating identical.

### What the spread already contains

The spread is a market forecast built from everything bettors and bookmakers know:
injuries, quarterbacks, home field and recent form. Our ratings are a narrow summary of
past results. Regressing the spread on both teams' pre-game ratings explains 64% of its
variance (uncentered R squared, 6,518 team-games, 2012 to 2023): one point per drive of
offense is worth about 11.5 points of spread, and one point per drive allowed about 12.7.
So most of what a rating says is already priced in. What remains is the part of the
rating that disagrees with the market.

The fitted models show how much the spread soaks up. Comparing the top five and bottom
five units of each season (by drive rating), at an even spread:

| Implied gap | Ratings without the spread | Ratings with the spread |
|---|---|---|
| Short-yardage conversion, top minus bottom offense | 7.3 points | 5.4 points |
| Short-yardage conversion, worst minus best defense | 5.5 points | 4.3 points |
| Points per drive from the own 25, top minus bottom offense | 1.02 | 0.78 |
| Points per drive from the own 25, worst minus best defense | 0.57 | 0.42 |

(The conversion rows come from a model with all four ratings, a diagnostic: its held-out
gain is not established, see section 4.) Roughly a quarter of the apparent team effect is
in the spread. The research note found about a third with a cruder rating.

## 3. How the ratings enter the decision

`TeamAwareEngine` extends the M2 engine without changing it (the test
`test_extension_leaves_the_engine_values_unchanged` checks the parent's numbers are
identical). Both engines get the same spread; the conversion probability of the
spread-only engine is evaluated at the play's spread, as the decision log asked for real
plays.

1. Conversion. log-odds of converting = spread-only model + (team model minus spread-only
   model). The go value is re-weighted: go = p' * WP(if converted) + (1 - p') * WP(if failed).
2. Giving the ball back, and keeping it. Every next state in which a team starts with the
   ball (after a punt, a failed attempt, a kickoff, or our own first down after a
   conversion) gets that team's change in next-drive touchdown and field-goal chances:
   team drive model minus spread-only drive model, using the offense rating of the team
   with the ball against the defense rating of the other. The change is valued through the
   win-probability model:

       change in value = sum over next states of P(state) * [dP(touchdown) * (WP after a
                         touchdown - WP after nothing) + dP(field goal) * (WP after a field
                         goal - WP after nothing)]

   from that team's point of view, with the sign flipped for the opponent.

Why through the drive and the win-probability input, not by changing the WP model: the
WP model is fitted with the spread only and cannot take ratings without being refitted
(and re-validated end to end). The next drive is the one thing a fourth-down choice
changes; everything after it is similar across the three actions. Adjusting only the next
drive keeps the spread-only engine as the exact baseline (with zero rating effects the two
tables are identical) and makes the adjustment testable by hand
(`test_opponent_scoring_more_lowers_the_punt_by_hand`: their touchdown chance up 0.10 lowers
a punt from our own 40 by 0.10 * 0.07 = 0.007 with the toy formula).

Worked example 5 (by hand, same toy formula). Fourth and 2 at the opponent's 40, 70%
conversion. If our drive's touchdown chance after a first down rises by 0.10, going gains
0.7 (we keep the ball) * 0.10 * 0.07 (swing between our touchdown and nothing) = 0.0049.
A punt gains nothing from our offense being good. That is the "possessions are worth more
to a great offense" mechanism, and the bad-defense mechanism is its mirror: a punt hands
the ball to an offense that will score more often than the spread assumes.

Simplifications: drive outcomes are three classes (touchdown, field goal, nothing); after
a scoreless drive the other team starts at a fitted spot (96.6 - 0.36 * start yard line);
drive times are their development means (touchdown 217 s, field goal 249 s, nothing
132 s); no adjustment is made where 360 s or less are left in the half (late-game drives
behave differently, and the drive model excludes the last two minutes); a first down in
mid-drive is treated like a drive starting there.

## 4. Results: do ratings predict beyond the spread?

Constants tuned on 2011 to 2023 (development seasons only) by season-blocked validation,
four blocks of three seasons, with the nested choice per block for the out-of-fold numbers.

| Filter | tau offense | tau defense | rho | weekly drift (share of tau^2) |
|---|---|---|---|---|
| Conversion | 0.03 | 0.02 | 0.75 | 0 |
| Drive points | 0.30 | 0.20 | 0.50 | 0.06 |

The filter's own pre-game predictions beat "no team effect" in every held-out block for
both metrics (for example drives: 0.08 to 0.12 points squared less error per drive, from
9.05). The drive grid was widened once after a first look on development seasons put two
constants on its edge; rho = 0.5 is still at the bottom of the drive grid, so true
carry-over may be lower.

The question proper: does adding ratings to the conversion and drive-outcome models,
which already have the spread, lower held-out log loss? Differences are per row (play or
drive), ratings model minus spread-only, with 95% intervals from 2,000 resamples of games.
Negative is better.

| Model | Development, out of fold (2012 to 2023) | Test, 2024 to 2025 (scored once) |
|---|---|---|
| Conversion (primary: conversion ratings), 29,816 / 5,355 plays | -0.20 per 1,000 (-0.47 to +0.06) | -0.21 per 1,000 (-0.84 to +0.45) |
| Drive outcome (primary: drive ratings), 70,027 / 10,500 drives | -2.17 per 1,000 (-2.71 to -1.61) | -3.46 per 1,000 (-5.03 to -1.89) |

The primary variant of each was chosen on development seasons by a rule written in the
config (lowest out-of-fold loss, simpler variant preferred within 0.0002) and recorded in
the evaluation plan before any test row was read. Other development variants: conversion
with drive ratings -0.25 (-0.66 to +0.14), with all ratings -0.30 (-0.72 to +0.10); ratings
without the spread -0.18 for conversion and -1.36 for drives (both worse than ratings with
the spread: the market knows things the ratings do not, and vice versa).

Reading. For drive outcomes the ratings add real information beyond the spread; the gain
is small in absolute terms (about 0.4% of a log loss of 0.92) but survives a fresh pair of
seasons. For conversion, "the spread already knows": the point estimate is a gain, but
neither the development nor the test interval excludes zero.

Why the conversion gain is so small even though the implied top-versus-bottom gap is 5
points (by hand): a prediction off by d in probability costs about d^2 / (2 p (1 - p)) in
log loss. If the part of team strength the spread misses has an SD of about 1.5 points,
the most any rating could save is 0.015^2 / (2 * 0.24) = 0.0005 per play, and a noisy
rating recovers only part of that. That is the size we see (0.2 per 1,000), and detecting it
would need far more than 30,000 plays.

## 5. Results: how much does the call move?

Charts in a tied first quarter (10:00 left, 3 timeouts each), 945 cells (yard lines 1 to 99,
distances 1 to 10). "Top" and "bottom" are the average end-of-season ratings of the top
five and bottom five units of each development season. The spread of each pairing is the
one the ratings imply. A flip "survives" if at least 83% of 200 resamples (effect models
refitted on resampled development games; Detroit's ratings also drawn from the filter's
posterior) flip it the same way. The base engine is held fixed; its own uncertainty is M3's.

| Pairing | Implied spread | Cells that flip | Survive | Go-gain shift, mean (range) | Direction |
|---|---|---|---|---|---|
| Strong offense vs weak defense | +9.7 | 100 | 92 | +0.61 (+0.15 to +1.11) | all toward go |
| Weak offense vs strong defense | -10.1 | 130 | 122 | -0.40 (-0.64 to -0.08) | all away from go |
| Average vs average | 0.0 | 1 | 1 | +0.02 | (rounding) |
| Great offense, terrible defense, vs average | +2.5 | 116 | 110 | +0.87 (+0.21 to +1.45) | all toward go |
| Detroit, end of 2023, vs average | +3.0 | 104 | 52 | +0.73 (+0.05 to +1.39) | all toward go |
| Detroit, end of 2024, vs average | +10.3 | 101 | 58 | +0.64 (+0.08 to +1.27) | all toward go |
| Detroit, end of 2025, vs average | +3.5 | 81 | 5 | +0.40 (+0.03 to +0.77) | all toward go |

Every flipped cell had a spread-only margin of at most 1.04 win-probability points; in
the first quarter, at most 0.96. In M3's terms these are "too close to call" cells: the
team adjustment moves the boundary between go and kick (in the figure, mostly fourth and
5 to 10 and the edges of the punt and field-goal regions), never the clear calls.

Trailing by 7 with 8:00 left the spread-only engine already says go in 96% to 99% of
cells, so few cells can flip toward go (7 to 20); the weak-offense pairing flips 178
toward kicking. Shifts there are larger (up to 3.5 points for Detroit 2024), because
possessions are scarcer late.

"Our defense is terrible." The pairing that isolates it (a top-five offense with a
bottom-five defense, which nets to a spread of only +2.5) moves the call toward going by
0.87 points on average, a little more than a top offense facing a bad defense (0.61). Both
mechanisms point the same way: when the offense is the strong unit, possessions are worth
more than field position. Baldwin's informal argument is right in sign; its size is under
a point and a half.

Detroit. Its ratings at the end of 2023 match the "great offense, bad defense" profile
(offense +0.58 points per drive, SD 0.29; defense allows +0.29, SD 0.23). By 2024 the
offense was +0.96 and the defense about average (+0.05); in 2025 +0.40 and +0.09. Its
conversion ratings were near zero every year (+0.1, +1.2, +0.1 points): the data do not
say Detroit converts unusually well beyond what drive quality implies. For 2025 only 5 of
81 flips survive, because the offense rating is uncertain (SD 0.31 against a mean of 0.40).

## 6. Results: real 2024 to 2025 fourth downs (scored once)

7,955 regulation fourth-down decisions (all of the 7,994 coach decisions except 39 in
overtime, which the engine does not value).

- 344 calls change (4.3%); 55 of them survive 83% of resamples. 174 move away from going
  (124 to a punt, 50 to a field goal) and 170 toward it (111 from a punt, 59 from a field
  goal), so the overall go share is unchanged (46.4% spread-only, 46.3% team-aware).
- The changed calls were close already: median spread-only margin 0.16 points, largest
  2.55. Across all plays the go gain moves by 0.26 points on average (absolute), 0.68 at
  the 90th percentile, 4.15 at most.
- Agreement with what coaches actually did does not improve (67.2% spread-only, 67.1%
  team-aware). Team strength beyond the spread does not explain coaches' choices.
- Most affected, by mean shift of the go gain: toward going Buffalo (+0.44), Philadelphia
  (+0.37), Dallas (+0.32), Washington and Baltimore (+0.29), Detroit (+0.27, 12 changed
  calls); toward kicking the Jets (-0.34), Cleveland (-0.33), Las Vegas (-0.31), Miami
  (-0.26), New England (-0.21).
- Detroit: 223 decisions, 66 goes. The spread-only engine endorses 58 of those goes, the
  team-aware engine 59. The average go gain on Detroit's goes rises from +2.32 to +2.68
  points. All 12 changed calls move toward going, and 11 of them were plays where Detroit
  kicked or punted. So Detroit's aggressiveness is slightly more justified once its
  offense is counted, but it was already justified without it: the spread-only engine
  agreed with 88% of its goes. If anything the team-aware engine says Detroit should have
  gone a little more often.

## 7. What it means

"Does the model know our defense is terrible?" Mostly yes, through the spread. Ratings
built honestly from past games carry a small amount of extra information about how drives
end, none that can be established about short-yardage conversion, and when that extra
information is pushed through the decision it moves the go-or-kick margin by a fraction of
a point for typical teams and by about one to one and a half points for extreme ones. It
changes calls that were already coin flips. It never turns a clear go into a kick or the
reverse. For the public argument, this is a strong version of "the spread already knows":
a coach who says "the model doesn't know my team" is right about the sign when his
offense is the strength of the team, and almost always wrong about the size.

## 8. Design decisions someone could challenge

1. Ratings measure results (conversion residuals, drive points), not play-level
   efficiency like EPA per play, which might be less noisy. Drive points were chosen
   because they are the decision-relevant outcome; an EPA-based rating is a natural
   alternative not tried.
2. Conversion residuals are against an even-spread curve, so ratings contain full team
   strength and the effect models decide what is beyond the spread. Residuals against the
   spread-adjusted curve would give "beyond the market" ratings directly.
3. Pairings use the spread the ratings imply. A pairing such as "top offense with an
   average defense" is then judged against a spread that also expects a better defense,
   so part of the shift comes from the defense being worse than the spread assumes. This
   is deliberate (it is exactly the "beyond the spread" question) but must be read that way.
4. Only the next drive is adjusted, and only with more than 360 s left in the half.
   End-game effects of team strength are not modelled.
5. The bootstrap refits the effect models and draws Detroit's and real plays' ratings,
   but holds the base engine (WP and play-outcome models) fixed and does not re-run the
   filter on resampled games.
6. Field-goal kicking ratings (optional in the brief) were not built.
7. The filter's noise model treats the per-game mean residual as Gaussian; with about four
   short-yardage plays per team-game this is rough. Tuning is by actual play-level log loss,
   so the constants are judged on the right scale.
8. The underlying engine's own weak spots apply: in the first-quarter chart it still says
   go on fourth and 4 inside the own 10 (decision log), which the team adjustment does not
   touch.
