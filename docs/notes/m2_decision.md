# M2 notes: the one-step decision engine

Code: `src/fourth_and_rl/decision.py`. Config: `configs/decision.yaml`.
Scripts: `scripts/decision_chart.py`, `scripts/compare_nfl4th.py`.
Tests: `tests/test_decision.py`.
Every number below is from `results/m2_decision_chart.json` or
`results/m2_nfl4th_comparison.json`, both written by those scripts. Numbers labelled
"nfl4th" are nfl4th's own (version 1.0.7, saved under `results/reference/`), never ours.
Worked examples marked "by hand" are arithmetic. Figures: `assets/m2_decision_chart.png`,
`assets/m2_nfl4th_comparison.png`.

All results are from the final rerun: win-probability version 2 (`results/m1_wp_metrics.json`),
the development component bundle (2010 to 2023, with the kickoff tables of each kickoff
rule era) and a field-goal limit taken from the development seasons. Findings from the
version 1 era are kept where they explain why version 2 exists, and are labelled as such.

## 1. Concepts

### Expected value over outcomes

A fourth-down action does not have one result. Going for it can end in a first down at
many different spots, a touchdown, a stop, an interception. Each result leaves the game
in some next state, and each next state has a win probability (WP). The value of the
action is the average of those win probabilities, weighted by how likely each result is:

    value(state, action) = sum over outcomes o of  P(o | state, action) * V(next state after o)

The engine computes this for go, field goal and punt and recommends the largest. P comes
from the M1 component models, V from the M1 win-probability model.

Why the full distribution and not the average outcome. V is not a straight line in field
position or score, so V(average next state) is not the average of V. The humility paper
(arXiv 2311.03490) evaluates WP at the expected next yard line and lists that as a
limitation; we average over the whole distribution.

Worked example (numbers from `reference_examples.fourth_and_3_at_opp_40`). First quarter,
tied, fourth and 3 at the opponent's 40.

- Go. P(convert) = 0.541. Averaged over where a conversion leaves the ball (and the small
  chance of a touchdown), our WP is 0.575. Averaged over the ways it fails, 0.417.
  go = 0.541 * 0.575 + 0.459 * 0.417 = 0.311 + 0.191 = 0.503 (the terms are rounded).
- Field goal (58 yards). P(make) = 0.551. The engine's value is 0.470.
- Punt. 0.461.

Go wins by 0.503 - 0.470 = 0.033, about 3 points of win probability. That 0.033 is the
`margin`. The break-even conversion probability is the p at which go equals the best
alternative: p * 0.575 + (1 - p) * 0.417 = 0.470 gives p = (0.470 - 0.417) / (0.575 -
0.417) = 0.053 / 0.158, about 0.34 (by hand). We estimate 0.54, so going is right unless
our conversion estimate is badly off.

### The perspective flip

The WP model answers one question only: "what is the probability that the team with the
ball wins?" After a punt, a made or missed field goal, a failed fourth down or a kickoff,
the other team has the ball. To get OUR win probability we must (1) describe the next
state from THEIR point of view and (2) take one minus the answer:

    our value = 1 - WP(their state)

"From their point of view" means every team-specific input is swapped or negated:

| Input | Ours | Theirs |
|---|---|---|
| score differential | +5 | -5 |
| pregame spread | +3.5 | -3.5 |
| posteam timeouts / defteam timeouts | 2 / 1 | 1 / 2 |
| is_home | 1 | 0 |
| receive_2h_ko (first half only) | 1 | 0 |
| yard line | measured to their target end zone, straight from the component model |

Version 2 of the WP model no longer uses `is_home` (home advantage is already in the
spread), but the engine still flips it, so any model that does use it is asked the right
question. Forgetting one of these is the classic bug in a decision engine, so the test
`test_change_of_possession_flips_every_team_specific_input` builds exactly this state
and inspects what the engine asks the WP model.

Worked example (by hand, with the toy WP of the tests: WP = 0.55 + 0.01 * lead - 0.002 *
(yards from the end zone - 50)). We punt and they start 80 yards from their end zone in a
tied game: their WP = 0.55 - 0.002 * 30 = 0.49, so the punt is worth 1 - 0.49 = 0.51 to us.

### Scores, kickoffs and the clock

- Touchdown: +6, then the extra point as two branches (good with the fitted probability,
  0.942 in the development bundle, giving +7; missed, giving +6), then a kickoff to the
  other team. A fractional
  "6.94 points" is never used: the WP model splits on whole scores, so an expected score
  would be read as either 6 or 7.
- Made field goal: +3, then a kickoff. Missed: the other team takes over at the spot of
  the kick (or their 20).
- Kickoff: the receiving team gets first and 10 at a spot drawn from the kickoff table of
  the kickoff rule in force: a real play uses the table of its own season's rule era
  (before 2024, the 2024 rule, the 2025 rule), and a hypothetical state without a season
  uses the latest rule (2025). The return touchdown (about 0.2% of kickoffs under the
  2025 rule) is closed off with a shortcut: 7 points to the receiving team, then the other team at the
  mean starting spot.
- Punt: the fitted distribution of the receiving team's starting spot, including the
  punts the kicking team recovers (about 1%) and those returned for a touchdown (about
  0.5%), from the development bundle (`results/m1_components.json`, `fitted.punt_rates`).
- Safety on a failed attempt: -2 and we kick to them (the ordinary kickoff distribution
  stands in for the free kick).
- Clock. Each play takes whole seconds off: converted fourth down 27, failed attempt 5,
  scoring play 5, field goal 4, punt 9, kickoff 2 (the development bundle's fitted
  values, rounded; the rounded numbers are also the config's fallbacks). These come from
  `components.seconds_per_play()` (mean snap-to-snap time of all scrimmage plays; an
  incomplete pass with the clock stopped; the kick means), rounded.
- End of game: if the play takes at least as many seconds as the half has left and it
  is the second half, the game is over with the score as it stands after the play:
  1 for a lead, 0 for a deficit, 0.5 for a tie. A field goal or touchdown on the last
  play counts. No WP model is asked.
- End of first half: the next state is the second-half kickoff at 1800 seconds, received
  by us if `receive_2h_ko` is 1 and by them otherwise, with timeouts reset to 3 each.

Worked example (it is a test). Down 3, one second left, a kick we never miss: field goal
= tie = 0.5. Going: a first down ends the game and we lose, so 0. Punt: 0. Field goal is
best with margin 0.5. Down 2 instead: field goal = 1.0.

## 2. Design decisions and why

1. **Full distributions, thinned for speed.** A typical real fourth down has about 850 possible next
   states. Scoring all of them for 10,000 states is about 8 million WP calls. So each
   distribution is thinned: yard lines carrying at least 0.4% of the action's probability
   stay exact, thinner neighbours are merged into one point at their mean (at most 8 yards
   wide). This leaves about 220 next states per play. Measured against the exact
   calculation on 2,000 real fourth downs: largest change in any action's WP 0.0002,
   mean 0.00003, same best action on 99.95% (`checks.thinning_error`). `merge_mass: 0` in
   the config turns thinning off.
2. **One batch, duplicates scored once.** All next states of all rows are built as numpy
   arrays, packed into one integer key each, de-duplicated, and sent to the WP model in
   one call. A chart panel (1,380 cells) builds 290,000 next states of which only about
   700 are distinct, so it takes a fraction of a second. Real plays share little:
   8,790 real fourth downs of 2022 and 2023 took 7.2 seconds on two threads on a machine
   shared with other jobs (`checks.speed`; the time varies with what else is running). The WP model's trees are the cost: about 1.7 million distinct states are
   scored. This is slower than the "few seconds" target for real plays.
3. **Which models.** WP: the M1 headline model, version 2 (trees on 2010 to 2023 that
   start from a structural baseline with the kneel-out rule, no calibration map;
   `docs/notes/m1_wp.md`). Components: the development bundle
   (`data/models/components_dev.joblib`, fitted on 2010 to 2023 only), so no 2024 or 2025
   play informs the engine that is compared with nfl4th on 2024 and 2025 and that the
   bootstrap resamples. One exception, stated plainly and logged: the kickoff starting
   distribution is a rule, not a skill, so the bundle holds one kickoff table per rule
   era, the 2024 and 2025 tables fitted on that season's own kickoffs. The bundle is
   reused only while its stamp matches the components config and the current development
   plays (a fingerprint of the play table); otherwise `decision.load_engine` refits it.
   The all-seasons bundle is not used here. Scripts read development seasons through
   `splits.load_dev_plays` and score 2024 to 2025 only through `splits.load_test_plays`
   with a logged purpose.
4. **Evenly matched conversion and kicking.** `p_convert` and `p_fg` are for an average
   offense, defense and kicker. The spread enters only through the WP model.
5. **Field goal out of range is null.** The component model's distance limit is the
   longest made kick in its fitted rows, 66 yards on the development seasons, after which
   P(make) fades to 0 over 3 yards: it is 0 for kicks of 70 yards or more (a line of
   scrimmage at the opponent's 52 or beyond), and `fg_wp` is null there. (The earlier
   68-yard cap came from a test-season kick and was replaced; see the leakage entry of the
   decision log.) **Punt is null inside the opponent's 31**: there
   are 71 punts in 16 seasons from there, so the model would be extrapolating.
6. **Overtime is not modelled.** Overtime states get nulls (39 of the 8,465 reference
   plays are dropped for this or for a missing input). A regulation tie is worth 0.5.
7. **Seconds per play are averages.** A team protecting a lead would take 40 seconds a
   play and a trailing team would hurry. The engine does not know that. This matters
   late in games and is one reason the full-game DP exists.
8. **`is_home = 0` in the charts.** Version 1 used `is_home` and, at a spread of 0, gave
   the away team a higher WP than the home team: on the reference chart the mean best WP
   was 0.496 with `is_home = 0` and 0.447 with 1, and the best action the same in 92.4% of
   cells (`checks.home_away_reference_chart`, version 1). Version 2 drops `is_home`, and
   the rerun confirms it: the best action is the same in every cell and the mean best WP
   is 0.470 either way. The charts keep `is_home = 0`.

## 3. Results

Every number in this section is from the final rerun of `scripts/decision_chart.py` and
`scripts/compare_nfl4th.py` (win-probability version 2, development-only outcome models,
the 66-yard field-goal limit). Version 1 findings are kept, labelled, because they are why
version 2 exists.

### The chart (`assets/m2_decision_chart.png`)

Share of the 1,380 cells (yard line 1 to 99, distance 1 to 15) by best action:

| Situation | Go | Field goal | Punt | Cells decided by under 1 point |
|---|---|---|---|---|
| First quarter, tied | 45% | 18% | 37% | 29% |
| Trailing by 7, 8:00 left | 73% | 12% | 16% | 26% |
| Leading by 4, 4:00 left | 35% | 24% | 41% | 16% |
| Trailing by 3, 2:00 left | 65% | 27% | 8% | 16% |

In the reference situation, going is best up to fourth and 3 at the own 20, fourth and 6
at the own 35, fourth and 9 at midfield, fourth and 7 at the opponent's 35 and fourth
and 4 at the opponent's 5. Almost 3 in 10 cells are decided by less than one point of
win probability, which is why M3 exists.

Two things on the chart should not be believed as drawn. First, going on fourth and 4 at
every spot inside your own 10, and on fourth and 5 or 6 at several of them (the blue patch
at the left edge): it rests on conversion probabilities extrapolated into situations
coaches never try and on the punt model near the own goal line, where data is thinnest,
and the margins are tiny. It is a model result, not a recommendation. Second, the
trailing-by-7 panel says go in 73% of cells, including fourth and 10 from your own 10 with
8:00 left. With version 1 it said go in 87%, which we put down to a WP model that gave a
team trailing by 7 little credit for field position; version 2 cut the share but not the
fourth-and-10 calls. Treat it as a known weakness of the one-step engine, not a finding.

### Comparison with nfl4th (`assets/m2_nfl4th_comparison.png`)

8,426 real fourth downs of 2024 and 2025 (the WP model's test seasons).

- Same recommended action (go, field goal or punt): **79.5%**. Same go-or-kick call: 80.6%.
- By how decisive nfl4th is (its |gain from going|, in WP points): under 0.5: 64%;
  0.5 to 1: 72%; 1 to 2: 80%; 2 to 4: 90%; 4 to 8: 97%; 8 or more: 97%.
  At 2 points or more: **93.0%** of 3,221 plays.
- Correlation of the gain from going: 0.84 (Spearman 0.81).
- We recommend going more often: 47.3% of plays against nfl4th's 41.1% (coaches went on
  21.3%).
- Of the 1,728 disagreements, 79% are plays where both models have the gain under 2
  points. The median nfl4th gain on a disagreement is 0.5 points.

With version 1 the agreement was 74.2% overall, 90.7% where nfl4th is decisive, and the
correlation 0.77. Version 2 closed about a fifth of the disagreement.

Where we disagree most (groups with at least 150 plays): distance 4 to 6 (66% agreement),
distance 3 (73%), between the opponent's 36 and midfield (75%), inside the opponent's 10
(75%), distance 7 to 10 (75%). Agreement is highest at the extremes: fourth and 1 (91%)
and 11 or more (92%).

### Why we disagree, with evidence

1. **Our conversion probability is higher.** Mean 0.399 against nfl4th's 0.375; the gap is
   1 point at fourth and 1 and 2 to 4 points from 2 to 10 yards. Swap diagnostic: put
   nfl4th's conversion probability into our engine (keeping our WP for each outcome) and
   agreement rises from 79.5% to 81.3%, and our share of "go" drops from 47.3% to 41.6%,
   almost exactly nfl4th's 41.1%. So the conversion model explains nearly all of our
   extra aggressiveness but only 2 points of the 21 points of disagreement. Note that our
   own held-out check found our conversion model too LOW for 2024 and 2025 (53.8%
   predicted, 56.1% actual), so it is not obvious that nfl4th's lower number is the
   better one.
2. **Field goal probability is not the cause.** In range, ours averages 0.699 and nfl4th's
   0.696; swapping it as well moves agreement by 0.3 points.
3. **The win-probability models differ, and that is most of it.** After both swaps 18.5%
   of plays still disagree. With version 1 the slope of our gain on nfl4th's was 0.73: our
   engine saw smaller differences between going and kicking than nfl4th did. Version 1 of
   our WP model had 15-leaf trees with at least 10,000 plays per leaf and a Platt map that
   shrinks toward 50%, which made it smooth and flat; its clock splits were about 30
   seconds apart, so taking 5 or 27 seconds off often changed nothing. Version 2 uses
   2,000-play leaves, a structural baseline and no map, and the slope is now 0.89. Our
   gains now spread slightly more than nfl4th's (standard deviation 3.39 against 3.21
   points), so the remaining slope below 1 comes from the imperfect correlation (slope =
   0.84 * 3.39 / 3.21), not from a flatter model.
4. **Late-game win probability: fixed on the leading side, still off on the trailing
   side.** First downs in the last 5:00 of the fourth quarter of 2024 and 2025 (the script
   quotes the preregistered M1 evaluation, `results/m1_wp_metrics.json`, `test_endgame`,
   as `causes.late_game_win_probability_from_m1_metrics`): teams leading by 1 to 3 with 41
   to 120 seconds left won 95.9% (n = 73) and version 2 says 94.9%; with 40 seconds or
   less they won all 18 and version 2 says 99.96%. Version 1 had said 86.2% and 89.0%, and
   a version 1 model that thought a 1-point lead with the ball and 13 seconds left was
   worth 72% valued a punt when trailing by 1 with 22 seconds left at 46% (nfl4th: 0.1%).
   Version 2 values that punt (the SEA v IND play below) at 0.9%. The trailing side is
   not fixed: teams trailing by 4 to 8 with 40 seconds or less won 8.1% (n = 74) and
   version 2 still says 16.6%; teams trailing by 1 to 3 with 40 seconds or less won 40.2%
   (n = 112) and version 2 says 29.7%. The engine's bookkeeping is right there (the
   end-of-game tests pass); the input is not.
5. **Clock handling.** We charge fixed average seconds per play. Agreement by clock:
   79.7% in the bulk of the game, 77.6% in the fourth quarter with 2:00 to 5:00 left,
   79.5% in the last 2:00 of a half. In the last two minutes we recommend going less
   often than nfl4th (35.1% against 40.3%), the reverse of the rest of the game (49.8%
   against 40.7%). That is consistent with the fixed 27-second charge for a converted
   fourth down (design decision 7), which the showcase plays below make visible.

### The late-lead scenario grid

On the 240 hypothetical late-lead scenarios where nfl4th returned a go value (own 20 and
own 30 only; nfl4th returned NA at the own 40 and midfield), the two models agree on
go-or-punt in only 43%, and the gains correlate at -0.12. We recommend going in 59% of
them and nfl4th in 16%. Our mean punt WP is 0.698 and nfl4th's 0.633. With version 1 the
agreement was 53%; version 2 made our engine much more willing to go here, which moves it
toward the exact solution of M4 and away from nfl4th. The two tools see this situation
very differently, and neither is a full-game solution. This is burning question 1 and it
is taken up in `docs/notes/m3_bootstrap.md` and `docs/notes/m4_mdp.md`.

### Ten high-leverage plays, both sets of numbers

Chosen by a fixed rule in the script (fourth quarter, one-score games, largest gain in
either model, one play per game: four playoff plays, three regular-season agreements,
three disagreements). WP in percent. "nfl4th" columns are nfl4th's numbers.

| Play | Situation | Coach | Ours: go / FG / punt | nfl4th: go / FG / punt | Ours | nfl4th |
|---|---|---|---|---|---|---|
| 2025 wild card, BUF at JAX | down 4, 1:10, 4th and 1 at the 11 | go | 41.9 / 12.3 / - | 40.8 / 18.0 / - | go | go |
| 2024 AFC title, KC v BUF | tied, 3:37, 4th and 17 at the 17 | FG | 52.8 / 62.8 / - | 53.5 / 69.8 / - | FG | FG |
| 2025 wild card, GB at CHI | down 4, 1:05, 4th and 7 at own 31 | go | 9.4 / - / 0.5 | 13.1 / - / 0.7 | go | go |
| 2025 NFC title, LA at SEA | down 4, 8:16, 4th and 1 at the 26 | go | 40.1 / 30.9 / - | 36.1 / 30.8 / - | go | go |
| 2024 wk 7, DET at MIN | down 1, 0:19, 4th and 14 at the 26 | FG | 3.8 / 75.6 / - | 13.6 / 72.8 / - | FG | FG |
| 2025 wk 18, HOU v IND | down 1, 0:17, 4th and 7 at the 25 | FG | 18.0 / 80.8 / - | 45.2 / 80.6 / - | FG | FG |
| 2025 wk 15, SEA v IND | down 1, 0:22, 4th and 2 at the 38 | FG | 3.0 / 57.9 / 0.9 | 38.3 / 50.5 / 0.1 | FG | FG |
| 2024 wk 18, IND v JAX | tied, 0:22, 4th and 7 at own 46 | punt | 36.4 / - / 48.9 | 52.3 / 26.6 / 52.0 | punt | go |
| 2025 wk 18, BAL at PIT | down 2, 0:21, 4th and 7 at midfield | go | 1.4 / 8.0 / 0.7 | 21.1 / 9.8 / 0.6 | FG | go |
| 2025 wk 12, PIT at CHI | down 3, 0:21, 4th and 6 at the 47 | go | 1.3 / 10.5 / 0.6 | 11.1 / 7.8 / 0.2 | FG | go |

The three disagreements are all in the last 22 seconds, and the cause is the clock, not
the win-probability model. The engine charges a converted fourth down 27 seconds (design
decision 7). With 21 or 22 seconds left that is more than the clock holds, so the engine
ends the game at the spot of any conversion short of the end zone: going is then worth
little more than the chance of a touchdown on the play (1.4% and 1.3% in the BAL and PIT
plays), and in the tied IND game a conversion is worth only a tie. nfl4th lets the offense
keep playing after a first down. The same charge explains our very low go values on the
DET and SEA field goal plays, where it does not change the call. In the IND play nfl4th's
own margin is 0.3 points, a coin flip.

With version 1 the regular-season plays chosen by the rule were different, and our three
disagreements were wrong because of the late-game win probability (point 4): a punt when
trailing by 1 with 40 seconds left was worth 28.5% to us, and a field goal when trailing
by 4 with 47 seconds left 19.2%. Version 2 removed that failure; the fixed clock charge is
what remains. Outside the last half minute (the four playoff plays) the two tools give
the same call with similar numbers.

## 4. Limits, stated plainly

- One step only. Nothing after the next snap is planned.
- Version 1 of the WP model was flat in places and too cautious late in close games
  (point 4). Version 2 fixes the leading side of the end game but not the trailing side:
  on the 2024 to 2025 first downs of the fourth quarter, trailing by 4 to 8 in the last
  40 seconds is predicted at 16.6% against 8.1% observed (74 plays), and trailing by 1 to
  3 in the last 40 seconds at 29.7% against 40.2% (112 plays)
  (`results/m1_wp_metrics.json`, `test_endgame.ours.cells_first_downs`). Recommendations for a
  trailing team inside the last minute should still not be trusted.
- With less than 27 seconds left, the fixed charge for a converted fourth down ends the
  game after any conversion short of the end zone, so going is undervalued there (the
  three showcase disagreements).
- The chart panels for trailing teams over-recommend going.
- Conversion and kicking are league-average; no team strength, weather or kicker.
- Overtime is not evaluated. Ties are worth 0.5.
- The comparison with nfl4th is against the CRAN release 1.0.7.
- 79% agreement means two models, not a model and the truth. Neither is ground truth.

## 5. Questions you should be able to answer

1. Write the value of "go" as a sum. What are the outcomes and where do the
   probabilities come from?
2. After a punt, why is our value 1 minus a win probability, and which five inputs must
   be flipped first?
3. Compute by hand: P(convert) 0.6, WP 0.70 if converted, 0.40 if not, punt worth 0.52.
   Go or punt, and what is the break-even conversion probability? (0.58 against 0.52;
   break-even 0.40.)
4. Why does the engine use two branches for the extra point and not 6.94 points?
5. Why average over the whole punt distribution and not use the mean landing spot?
6. What does the engine do when a play runs out the first half? The second half?
7. We agree with nfl4th on 79% of plays overall and 93% when it is decisive. Why is the
   second number the more meaningful one?
8. Swapping in nfl4th's conversion probability closes our go-rate gap but barely changes
   agreement. What does that tell you about where the disagreement comes from?
9. With version 1, our engine valued a punt when trailing by 1 with 22 seconds left at
   46%. Was that an engine bug? How do you know? What did version 2 change about it?
10. Why are the outcome models fitted without 2024 and 2025, and why is the kickoff table
    the one exception?
