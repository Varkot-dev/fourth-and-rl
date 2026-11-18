# M7 notes: grading coaches, and the coach cost ledger

Code: `src/fourth_and_rl/coaches.py`. Config: `configs/coaches.yaml`.
Scripts: `scripts/m7_penalty_ledger.py` (run first), `scripts/m7_coach_evaluation.py`.
Tests: `tests/test_coaches.py`.
Every number below is from `results/m7_coach_evaluation.json`, `results/m7_coach_ledger.csv`
or `results/m7_penalty_ledger.json`, written by those scripts. Worked examples marked "by
hand" are arithmetic. Labels quoted from posts on X are the posters' words, unverified, and
come from `research-kit/x/2_twitter_findings.md`.
Figures: `assets/m7_league_trend.png`, `assets/m7_team_aggressiveness.png`,
`assets/m7_coach_ledger.png`, `assets/m7_case_study.png`.

## 1. What was built, in one paragraph

Every real fourth-down decision from 2011 to 2025 (58,929 of them, in 8,192 team-games) was
valued by the one-step engine of M2: the win probability of going, kicking and punting. The
decision's cost is the win probability the coach gave up against the engine's best action.
Only decisions whose best action the game-level bootstrap calls confident count toward the
headline cost; the total over all decisions is reported next to it. The costs are summed by
season, team and coach, with intervals from resampling games and with shrinkage toward the
league. A second column of the ledger prices the owner's "coach" penalties (false starts and
delays of game), and a statistical test asks whether those penalties really belong to the
coach. Finally, the calls that were called malpractice on X before the 2026 season are
rebuilt and scored, and the Lions at 49ers NFC Championship is drawn as a case study.

## 2. How a decision is graded

### The arithmetic

For a fourth down the engine returns three win probabilities (WP), for go, field goal and
punt. The best action is the largest. The coach chose one action. Then

    WP given up = WP(best action) - WP(chosen action)     (0 when the coach chose the best)

Win probability is a fraction, so these numbers add up to wins: 100 WP points is one win,
and a coach who gives up 2 points on each of 50 decisions has given up 1 win.

Worked example 1 (from the results, Bears at the start of the fourth quarter of the
2025 divisional round, 4th and 2 at their own 45, tied). Go 46.39%, punt 41.29%. Chicago
punted. WP given up = 46.39 - 41.29 = 5.10 points = 0.051 wins (by hand). The bootstrap
put go first in 100% of refits, so the decision counts toward the headline.

Worked example 2 (by hand). A coach faces three fourth downs. On the first the engine says
punt by 3 points and he punts: 0. On the second the engine says go by 4 points (confident)
and he punts: 4. On the third the engine says go by 0.5 points but only 55% of refits agree
(too close to call) and he punts: 0.5. Headline cost 4 points; total cost 4.5 points.

### The play's own spread enters the conversion chance

The M2 engine evaluates the conversion probability for an evenly matched game (the right
choice for charts). A real play has a pregame spread, and the decision log said real plays
would use it. The engine already gives the two conditional values: WP if converted (a) and
WP if not (b). Going is worth p * a + (1 - p) * b, so the spread is put in by recomputing p
at the play's spread and reweighting, leaving a and b alone (`spread_adjusted_go`).

Worked example 3 (by hand). a = 0.70, b = 0.40. At an even spread p = 0.60, so go = 0.42 +
0.16 = 0.58. The offense is favoured by 7 and p rises to 0.63: go = 0.441 + 0.148 = 0.589.
If punting is worth 0.585, the call flips from punt to go. On 2024 to 2025 the spread changed
the best action on 305 of 7,955 plays.

### Confident only, and why

A model that is unsure should not be allowed to fine a coach. Using the M3 rule (Brill,
Yurko and Wyner): a decision is "confident" when at least 83% of game-level bootstrap refits
pick the same best action, "lean" from 67% to 83%, and "too close to call" below that.
About 70% of real fourth downs are confident in every fold (69% to 72%). Over 2011 to 2025
the headline cost is 246.5 wins (90% interval 240.8 to 252.4) and the total 297.9 wins:
the confident rule removes about a sixth of the cost. Of the confident cost, 227.7 wins
(92%) come from not going when the engine said go, and 10.6 from going when it said kick.

Limit, stated plainly: the bootstrap measures how much the answer moves when the data are
resampled; it does not measure model bias, and for tree models it under-covers (M3b). The
case study shows a "confident" label on a 0.12-point edge.

## 3. Why the grading model must never have seen the season it grades

If the model that grades 2019 was fitted on 2019, then every 2019 game's outcome helped
shape the win probabilities used to judge 2019's decisions. The bias goes in a sneaky
direction: a team that went for it and then won pushes the model, in that region of the
state space, toward "going wins", which grades its own decision kindly. The grade is then
partly a record of what happened, not of what was knowable.

So each season is graded by models fitted without it (`fold_plan`, `check_fold_plan`):

| Seasons graded | Models fitted on |
|---|---|
| 2011 | 2012 to 2023 (2010 and 2011 held out together) |
| 2012-13, 2014-15, ..., 2022-23 | the other development seasons |
| 2024-25 (test seasons) | all development seasons (the headline models) |

Both the win-probability model (version 2 spec, no calibration map) and the play-outcome
bundle are refitted per fold, and every fit passes the leakage guard. The 2026 holdout is
never loaded. A test proves no fold grades a season it was fitted on, and a data test reads
the saved models' stamps to check the same thing after the run.

Each development fold also gets its own cheaper bootstrap: 31 refits of whole games drawn
from the fold's fit seasons only (M3 used 101). With 31 refits the confident threshold means
26 or more agree. Split-half agreement of the categories is 79% to 82% per fold (M3 had 89%
with 101 refits). The test seasons reuse M3's 101-refit categories; where the spread
changed the best action, the share of M3's refits that picked the new best action is used.

### A decision made after seeing a result

The first run used folds of four seasons. The league trend then jumped exactly at fold
boundaries: the engine said go on 40% of 2018 to 2021 fourth downs and 47% in 2022. The
cause was mechanical: the play-outcome count tables use 2018 onward only, so the fold that
held out 2018 to 2021 had just 2022 and 2023 to fit them on. The folds were changed to two
seasons (the size the win-probability cross-fit already used). To keep the effect of the fold
visible, the results also hold every development season graded in sample by the headline
models (`fold_sensitivity`, a diagnostic never reported as a result). With two-season folds
the two agree on the best action for 91% to 94% of plays a season and on the go share within
3.4 points; the largest gap is 2022 (0.53 against 0.47 WP points per decision).

## 4. Per opportunity versus total, per game, and the era

Three ways to measure one coach, and why all are needed:

- **Total cost** grows with tenure. Mike Tomlin has the largest total (9.9 wins over 258
  games) partly because he coached the most games.
- **Per game** (shown per 17 games, one season) fixes tenure.
- **Per opportunity** (WP points per graded fourth down) fixes how many fourth downs a team
  faces. A team with a bad offense faces more fourth downs and has more chances to err.

Worked example 4 (by hand). Coach A: 60 fourth downs in 8 games, 3.0 points given up. Coach
B: 90 fourth downs in 8 games, 3.6 points. Per game B is worse (0.45 against 0.375); per
opportunity B is better (0.040 against 0.050). B's offense simply stalled more.

**The era.** The whole league gave up 3.8 WP points per team-game in 2011 and 2.3 in 2025.
A raw coach table therefore mostly ranks eras: on raw per-game cost the best coaches were all
recent and the worst all from the early 2010s. The ledger ranks on cost above the league
average of the same seasons (each team-game's cost minus that season's league mean), and
keeps the raw numbers in the CSV.

## 5. Shrinkage, and how many differences are real

A coach's per-game cost is a noisy estimate. The empirical-Bayes estimate pulls it toward
the league by an amount set by its own noise (`dersimonian_laird`):

    tau^2 = spread of the TRUE coach effects (method of moments)
    B     = tau^2 / (tau^2 + se^2)            (weight on the coach's own data)
    shrunk = mean + B * (raw - mean)

For era-adjusted wins per 17 games among the 83 coaches with 32 or more games: tau = 0.066,
the average standard error 0.071, mean -0.012.

Worked example 5 (by hand, Tomlin). Raw +0.14, 90% interval 0.077 to 0.203, so se is about
0.126 / 3.29 = 0.038. B = 0.0044 / (0.0044 + 0.0015) = 0.75. Shrunk = -0.012 + 0.75 * 0.152
= +0.102. The script gives +0.10.

Worked example 6 (by hand, Norv Turner, 32 games). Raw -0.19, 90% interval -0.36 to -0.02,
so se is about 0.34 / 3.29 = 0.104. B = 0.0044 /
(0.0044 + 0.011) = 0.29. Shrunk = -0.012 + 0.29 * (-0.178) = -0.064. One short tenure that
looks brilliant is mostly noise and is pulled most of the way back.

How much of the coach table is signal:

| Measure | Reliability | Raw 90% intervals excluding the mean | Expected by chance | Shrunk intervals excluding it |
|---|---|---|---|---|
| Era-adjusted, per 17 games | 0.46 | 20 of 83 | 8.3 | 10 |
| Era-adjusted, per opportunity | 0.41 | 19 of 83 | 8.3 | 8 |
| Raw, per 17 games (era confounded) | 0.72 | 37 of 83 | 8.3 | 31 |

Reliability 0.46 means a bit under half of the spread you see between coaches is real. Of
3,403 coach pairs, 789 differ at the 90% level against 340 expected by chance. Year to year,
a coach's cost per opportunity correlates 0.27 (0.20 to 0.34) with his next season, while
his go rate correlates 0.62 (0.56 to 0.67): aggressiveness is a stable trait, the cost of it
is mostly noise around a smaller stable part.

## 6. Results

### League trend (`assets/m7_league_trend.png`)

| Season | Went for it | Went when the engine confidently said go | WP points given up per team-game (confident), 90% interval |
|---|---|---|---|
| 2011 | 11.2% | 25.0% | 3.83 (3.50 to 4.17) |
| 2014 | 12.0% | 27.3% | 3.49 (3.22 to 3.75) |
| 2017 | 12.8% | 30.4% | 3.37 (3.08 to 3.65) |
| 2018 | 14.9% | 36.9% | 2.63 (2.39 to 2.87) |
| 2020 | 19.2% | 46.6% | 1.96 (1.78 to 2.16) |
| 2021 | 20.8% | 48.0% | 2.01 (1.82 to 2.22) |
| 2022 | 19.0% | 41.7% | 3.16 (2.88 to 3.47) |
| 2024 | 20.4% | 47.3% | 2.49 (2.25 to 2.75) |
| 2025 | 23.2% | 51.0% | 2.27 (2.06 to 2.50) |

Coaches roughly doubled their go rate and now take about half of the engine's confident go
calls, against a quarter in 2011. The cost fell from about 4 points per team-game to about 2
by 2020 and 2021, rose in 2022 (the go rate dipped that season too, and the in-sample
headline grade also rises, so this is mostly real; the fold models add about 0.06 points per
decision), and fell again. In the whole period the league gave up 0.51 wins per team per 17
games (0.50 to 0.52) on confident calls.

### Teams, 2021 to 2025 (`assets/m7_team_aggressiveness.png`)

Pittsburgh gave up the most (0.71 wins per 17 games raw, 0.60 shrunk) and took the engine's
confident go calls least often (33%); Green Bay the least (0.25 raw, 0.27 shrunk). Detroit
took 63% of confident go calls, the highest. Reliability 0.70; 10 of 32 teams' intervals
exclude the league mean against 3.2 expected by chance.

### Coaches (`assets/m7_coach_ledger.png`, `results/m7_coach_ledger.csv`)

Era-adjusted wins given up per 17 games, shrunk, with 90% intervals:

- Fewest: Matt LaFleur -0.15 (-0.19 to -0.10), Nick Sirianni -0.12 (-0.18 to -0.07), Dan
  Campbell -0.11 (-0.17 to -0.06), Sean McDermott -0.10 (-0.16 to -0.05), Dave Canales and
  Jonathan Gannon about -0.09.
- Most: Mike Tomlin +0.10 (+0.05 to +0.16), Doug Marrone +0.08 (+0.01 to +0.15), Matt
  Patricia +0.06, Ken Whisenhunt +0.05, Mike Mularkey +0.05, Anthony Lynn +0.04; from
  Patricia on, the shrunk intervals include zero. Todd Bowles, the coach named on X, is
  +0.04 shrunk (-0.03 to +0.11; raw +0.07, -0.01 to +0.16): not distinguishable from his
  seasons' average.
- In total over his tenure Tomlin gave up 2.1 wins more than an average coach of the same
  seasons would have on the same number of games, and 9.9 confident wins in all.

The intervals at the two ends do not overlap, but for most of the middle of the table they
do. A quoted "worst coach" list should carry the intervals, the per-opportunity rate and the
statement that only about half of the visible spread is real.

## 7. Pricing a procedure penalty

A false start is a play with a state before and after. With the foul, the offense lines up
5 yards back, with 5 more yards to go, same down, same clock. Without it, the snap would
have happened at the original spot. So (`penalty_states`, `price_states`)

    price = WP(down, distance, spot as they were) - WP(same down, distance + yards, spot - yards)

using the version 2 model that never saw the season (the same folds as the decisions). In
goal to go the distance is the yard line; the nflverse yardage already applies the half-the-
distance rule. The 10-second runoff option after a late false start is not priced (632
fouls came in the last minute of a half), nor overtime fouls (58).

Worked example 7 (by hand, with the toy model of the tests: WP = 0.5 + 0.01 * lead - 0.002 *
(yards from the end zone - 50) - 0.004 * distance). First and 10 at the opponent's 40 (60
yards from the end zone): the foul moves it to 65 yards and 15 to go. Price = 0.002 * 5 +
0.004 * 5 = 0.030, 3 points. The test checks exactly this.

What counts: false start or delay of game by the offense, enforced before the snap (no-play,
yards above zero, a down). Not counted: extra-point tries (168, no down), delays called after
a play ran (42), zero-yard fouls (56), and defensive fouls (encroachment, defensive offside,
neutral zone infraction, as the owner said). **Deliberate delays** are excluded: lined up in
punt or field-goal formation (464), or a delay on fourth down followed by the offense's punt
or field goal (357 more). 100 of the excluded were in field-goal formation; some of those
may be real mistakes, and they are counted in the JSON as a sensitivity.

Results, 2011 to 2025: 10,104 priced coach fouls. False starts: 8,855, mean price 1.40 WP
points (median 1.12), 124.4 wins in all. Delays: 1,249, mean 1.30, 16.2 wins. About 9.4
wins a season league-wide, 0.29 wins per team per season, roughly half the size of the
fourth-down cost (0.51).

## 8. The assumption test, and what it found

The owner's rule ("false starts and delays of game are on the coach") is an assumption, so
it was tested three ways, under a verdict rule written into the config before the penalty
data was read (penalties enter the coach's total only if all three checks pass).

**Stability.** Rates are season-adjusted: log((fouls + 0.5) / (expected + 0.5)), where
expected counts come from league rates by season, home or road, dome and third-or-fourth-
and-long. Consecutive seasons of the same coach at the same team correlate 0.15 (0.07 to
0.24); split halves of a coach's career correlate 0.28 (Spearman-Brown reliability 0.44).
Weak.

**Does it follow the coach?** The pre-registered version (consecutive seasons only) found
just 7 moves, too few to say anything (r = -0.51, interval -0.98 to 0.25). A stint-based
version, added after seeing that and outside the verdict rule, uses every coach who moved
(31 moves: Reid, Carroll, Payton, Harbaugh, Bowles and others). It compares the coach's rate
at his old team with his rate at the new one, and the new team's rate under its previous
coach with his rate there:

| | Correlation | 90% interval |
|---|---|---|
| Follows the coach (old team to new team) | -0.13 | -0.50 to 0.27 |
| Stays with the team (previous coach to new coach) | 0.41 | 0.19 to 0.61 |

The rate stays with the team, not the coach.

**How much of the variance is the coach?** A Poisson model of fouls per offense team-game
with expected counts as the offset and crossed random effects for coach, franchise and
team-season (`fit_poisson_crossed`, Laplace approximation; statsmodels is not installed,
so it was written here and checked on simulated data in the tests):

    fouls ~ Poisson(expected * exp(b0 + coach + franchise + team_season))

| Effect | Standard deviation (log scale) | Share of systematic variance |
|---|---|---|
| Coach | 0.066 (95% profile interval reaches zero, upper 0.105) | 21% |
| Franchise | 0.061 | 18% |
| Team-season | 0.114 | 61% |

Worked example 8 (by hand). Variances: 0.066^2 = 0.0044, 0.061^2 = 0.0037, 0.114^2 = 0.0130,
total 0.0211, coach share 0.21. A team averages 1.23 expected fouls a game, and Poisson
noise adds about log(1 + 1 / 1.23) = 0.59 of variance on the same scale, so the coach
explains under 1% of the game-to-game variation. A coach one standard deviation above average
commits exp(0.066) = 1.07 times as many fouls: about 1.4 extra fouls a season.

Fixed situations matter more: road teams commit 1.06 times the home rate (1.11 in domes,
1.04 outdoors), and third or fourth and long draws 1.42 times the rate of other snaps.

**Verdict:** all three checks fail (interval reaches zero; coach share 0.21 under 0.33; the
rate does not follow the coach). Assigning these penalties to the coach is not supported.
The ledger reports them in a separate column and does not add them to the coach's total.

One part does look like the coach: **delays of game alone** have a coach standard deviation
of 0.26 (0.17 to 0.37), 54% of their systematic variance. But delays are 12% of the fouls and
16 of the 141 penalty wins, the mover test does not support them either (follows -0.08,
stays 0.37), and this split was looked at after the main result. It is reported as a lead for
later work, not used.

## 9. The malpractice check, part one

Calls from `research-kit/x/2_twitter_findings.md` (sections 1, 8.7 and 9.2) before the 2026
season, rebuilt from play-by-play and scored by the one-step engine (with its category) and by
the exact game solution of M4 (identical teams, spread 0). WP points; positive = given up.

| Call (label on X) | Coach did | One step: best, given up, category | Exact game: best, given up |
|---|---|---|---|
| Bears punt 4th and 2 at own 45, tied, start of Q4, divisional ("one of the dumbest punts") | punt | go, 5.1, confident | go, 4.6 |
| Tomlin punts 4th and 9 at own 23, down 3, 2:01 ("cowardly") | punt | go, 2.7, confident | go, 8.1 |
| Campbell goes for the win, 4th and 1 at GB 21, tied, 0:43 (praised) | go | field goal, 4.0, confident | field goal, 7.2 |
| Harbaugh punts 4th and 3 at own 38, up 2, 1:33 ("the game was over the moment they punted") | punt | go, 4.9, confident | punt, 0 (punt by 4.4) |
| Daboll kicks from the 1, up 7, 10:21 (debated) | field goal | go, 2.4, confident | go, 2.0 |
| McDermott punt, 4th and 3 at midfield, down 14, Q2 ("cowardly", play not named) | punt | go, 3.8, confident | go, 2.1 |
| McDermott punt, 4th and 5 at own 44, down 7, 10:24 Q4 | punt | go, 1.7, confident | go, 3.1 |
| LaFleur punts, 4th and 5 at CIN 42 or 4th and 6 at own 20 ("cowardly", play not named) | punt | go, 0.6 and 0.9, too close | go 1.5; punt 0 |
| Lions go 4th and 2 at SF 28, up 14 (NFC title, 2023 season) | go | field goal, 0.12, "confident" | go, 0 |
| Lions go 4th and 3 at SF 30, down 3 (NFC title) | go | go, 0 | go, 0 |
| Payton goes 4th and 1 at NE 14, up 7, AFC title ("recklessly ignoring a comfortable field goal") | go | go, 0, confident | go, 0 |
| Lions go 4th and 3 at GB 21, down 10, Thanksgiving | go | go, 0, confident | go, 0 |
| Eagles' last fourth down, 4th and 11 at SF 21, down 4, 0:43 ("Malpractice", about the play design) | go | go, 0 | go, 0 |

Read: the calls labelled reckless for going (Payton, the Lions twice) were the models' calls.
The real costs were punts that drew milder words (Bears, Tomlin, McDermott) and one praised
go (Campbell). Harbaugh's punt is the one case where the two models split, by about 4.5
points each way, and it is exactly the late-lead clock case where the one-step engine is
weakest (its fixed 27-second charge for a conversion) and the exact solution, which plans the
clock, is the better guide. Notes on the rebuilds: the Bears post says "last regulation
possession" but the only 4th and 2 punt was the first play of the fourth quarter; the Eagles
post says 4th and 8 but the play-by-play has 4th and 11; the Harbaugh 40-25 lead was at 4:48
and the lead was 2 at the punt. Not scoreable here: timeout decisions (Bowles's kneel,
Sirianni's timeout), play design and personnel. Pending the 2026 holdout: the Lions' 4th and
1 at their own 24, Bowles's 4th and goal up 14, the Chiefs' long field goal, Quinn's clock
and field-goal calls, Schottenheimer's punt.

## 10. The case study: Lions at 49ers, NFC Championship (`assets/m7_case_study.png`)

Graded by the 2022-23 fold models (fitted on 2010 to 2021), 31 refits, game spread used.

| # | Situation | Detroit did | Go / FG / punt (%) | Engine | Given up | Exact game |
|---|---|---|---|---|---|---|
| 1 | Q2 10:23, 4th and 6 at SF 46, up 7 | punt | 43.8 / 41.6 / 40.9 | go (confident) | 2.9 | go, 1.1 |
| 2 | Q2 0:10, 4th and 3 at SF 3, up 14 | field goal | 77.9 / 78.9 / - | field goal (confident) | 0 | field goal |
| 3 | Q3 7:03, 4th and 2 at SF 28, up 14 | go (failed) | 86.5 / 86.6 / - | field goal, labelled confident | 0.12 | go by 0.2 |
| 4 | Q3 2:13, 4th and 9 at own 26, tied | punt | 26.5 / - / 27.6 | punt (confident) | 0 | punt |
| 5 | Q4 7:38, 4th and 3 at SF 30, down 3 | go (failed) | 26.9 / 25.0 / - | go (confident) | 0 | go by 1.9 |
| 6 | Q4 1:00, 4th and 3 at SF 3, down 10 | go (touchdown) | 1.8 / 0.8 / - | go | 0 | lost either way |

The famous fourth and 2 was a coin flip: one step says field goal by 0.12 points, the exact
solution says go by 0.24. It is labelled "confident" only because 26 of 31 refits put the
field goal first; the 90% interval of the go-minus-kick gain is -0.86 to +0.43 points, which
straddles zero. That is the limit of a bootstrap category with few refits on a tiny edge, and
the right reading is "too close to call". The fourth and 3 when trailing was the right call
by both models. The decision that cost Detroit something was a punt: 4th and 6 at midfield
in the second quarter (2.9 points one step, 1.1 exact). The game turned on what happened
after the calls (two failed conversions, a fumble, a dropped pass off a helmet), not on the
calls.

## 11. Limits

- One-step engine: it charges 27 seconds for a converted fourth down and assumes average
  play afterwards. Late in close games (Harbaugh) the exact solution can disagree.
- Older seasons are graded by models whose play-outcome count tables come from 2018 onward
  and whose kickoff table is the 2018 to 2023 rule; 2011 to 2017 are graded "by a later game".
- Bootstrap categories measure sampling noise, not model bias, and under-cover for tree
  models; development folds have 31 refits, so the categories are coarser than M3's.
- The coach is the head coach from the schedule. Play-callers and coordinators are not
  separated; the penalty "coach effect" includes the staff.
- Only the first penalty on a play is read; runoffs and overtime fouls are not priced.
- The assumption test is descriptive over all of 2011 to 2025, including the test seasons;
  nothing in it is a prediction scored on them.
- The stint mover test and the delay-of-game split were added after seeing results, and the
  fold size was changed after seeing the trend; each is recorded as such.

## 12. Questions you should be able to answer

1. Why must the model that grades 2019 not have seen 2019? Which way does the bias go?
2. A coach gives up 5 points on a confident call and 1 point on a too-close-to-call one.
   What are his headline and total costs, in wins? (0.05 and 0.06.)
3. Why rank coaches against their own seasons' average? What did the raw table rank?
4. Tomlin has the largest total cost. Give two reasons that is not the same as "worst".
5. Compute the shrunk estimate for a coach with raw +0.30, se 0.10, tau 0.066, mean 0.
   (B = 0.30, shrunk +0.09.)
6. How is a false start priced? What is not priced, and why does a delay before a punt not
   count?
7. The coach explains 21% of systematic variance in procedure fouls and under 1% of all
   variation. Which number answers "is it on the coach"? What did the movers show?
8. Why can the one-step engine and the exact solution disagree on Harbaugh's punt, and which
   one would you trust there?
9. The NFC title 4th and 2 is labelled confident at a 0.12-point edge. What is wrong with
   reading that label at face value?
10. What selection bias hides in "coach X gives up the most per game"? (Bad offenses face
    more fourth downs; long tenures accumulate; eras differ.)
