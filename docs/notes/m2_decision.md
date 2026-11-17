# M2 notes: the one-step decision engine

Code: `src/fourth_and_rl/decision.py`. Config: `configs/decision.yaml`.
Scripts: `scripts/decision_chart.py`, `scripts/compare_nfl4th.py`.
Tests: `tests/test_decision.py`.
Every number below is from `results/m2_decision_chart.json` or
`results/m2_nfl4th_comparison.json`, both written by those scripts. Numbers labelled
"nfl4th" are nfl4th's own (version 1.0.7, saved under `results/reference/`), never ours.
Worked examples marked "by hand" are arithmetic. Figures: `assets/m2_decision_chart.png`,
`assets/m2_nfl4th_comparison.png`.

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
  chance of a touchdown), our WP is 0.625. Averaged over the ways it fails, 0.435.
  go = 0.541 * 0.625 + 0.459 * 0.435 = 0.338 + 0.200 = 0.538.
- Field goal (58 yards). P(make) = 0.551. The engine's value is 0.498.
- Punt. 0.485.

Go wins by 0.538 - 0.498 = 0.039, about 4 points of win probability. That 0.039 is the
`margin`. The break-even conversion probability is the p at which go equals the best
alternative: p * 0.625 + (1 - p) * 0.435 = 0.498 gives p = (0.498 - 0.435) / (0.625 -
0.435) = 0.33 (by hand). We estimate 0.54, so going is right unless our conversion
estimate is badly off.

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

Forgetting one of these is the classic bug in a decision engine, so the test
`test_change_of_possession_flips_every_team_specific_input` builds exactly this state
and inspects what the engine asks the WP model.

Worked example (by hand, with the toy WP of the tests: WP = 0.55 + 0.01 * lead - 0.002 *
(yards from the end zone - 50)). We punt and they start 80 yards from their end zone in a
tied game: their WP = 0.55 - 0.002 * 30 = 0.49, so the punt is worth 1 - 0.49 = 0.51 to us.

### Scores, kickoffs and the clock

- Touchdown: +6, then the extra point as two branches (good with the fitted probability
  0.942, giving +7; missed, giving +6), then a kickoff to the other team. A fractional
  "6.94 points" is never used: the WP model splits on whole scores, so an expected score
  would be read as either 6 or 7.
- Made field goal: +3, then a kickoff. Missed: the other team takes over at the spot of
  the kick (or their 20).
- Kickoff: the receiving team gets first and 10 at a spot drawn from the fitted kickoff
  distribution. The return touchdown (0.2% of kickoffs) is closed off with a shortcut:
  7 points to the receiving team, then the other team at the mean starting spot.
- Punt: the fitted distribution of the receiving team's starting spot, including the 1%
  of punts the kicking team recovers and the 0.5% returned for a touchdown.
- Safety on a failed attempt: -2 and we kick to them (the ordinary kickoff distribution
  stands in for the free kick).
- Clock. Each play takes whole seconds off: converted fourth down 27, failed attempt 5,
  scoring play 5, field goal 4, punt 9, kickoff 2. These come from
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
   calculation on 2,000 real fourth downs: largest change in any action's WP 0.0005, mean
   0.00002, same best action on 99.95% (`checks.thinning_error`). `merge_mass: 0` in the
   config turns thinning off.
2. **One batch, duplicates scored once.** All next states of all rows are built as numpy
   arrays, packed into one integer key each, de-duplicated, and sent to the WP model in
   one call. A chart panel (1,380 cells) builds 290,000 next states of which only about
   700 are distinct, so it takes a fraction of a second. Real plays share little:
   8,790 real fourth downs took 8.8 seconds on two threads on a machine shared with
   other jobs (`checks.speed`; the time varies with what else is running). The WP model's trees are the cost: about 1.7 million distinct states
   are scored. This is slower than the "few seconds" target for real plays.
3. **Which models.** WP: the M1 headline model (trees 2010 to 2021, Platt 2022 to 2023).
   Components: refitted on 2010 to 2023 only, so no 2024 or 2025 play informs the
   engine that is compared with nfl4th on 2024 and 2025 and that the bootstrap
   resamples. One exception, stated plainly: the kickoff starting distribution comes
   from 2025, because it is a rule (touchback to the 35), not a skill, and earlier
   seasons used other rules. The M1 bundle fitted on all seasons is not used here.
4. **Evenly matched conversion and kicking.** `p_convert` and `p_fg` are for an average
   offense, defense and kicker. The spread enters only through the WP model.
5. **Field goal out of range is null.** The component model returns P(make) = 0 beyond a
   68-yard kick; `fg_wp` is null there. **Punt is null inside the opponent's 31**: there
   are 71 punts in 16 seasons from there, so the model would be extrapolating.
6. **Overtime is not modelled.** Overtime states get nulls (39 of the 8,465 reference
   plays are dropped for this or for a missing input). A regulation tie is worth 0.5.
7. **Seconds per play are averages.** A team protecting a lead would take 40 seconds a
   play and a trailing team would hurry. The engine does not know that. This matters
   late in games and is one reason the full-game DP exists.
8. **`is_home = 0` in the charts.** At a spread of 0 the M1 model gives the away team a
   higher WP than the home team. On the reference chart the mean best WP is 0.496 with
   `is_home = 0` and 0.447 with 1, and the best action is the same in 92.4% of cells
   (`checks.home_away_reference_chart`). The level shift is an M1 weakness; the 7.6% of
   cells that change are a warning that close calls depend on such details.

## 3. Results

### The chart (`assets/m2_decision_chart.png`)

Share of the 1,380 cells (yard line 1 to 99, distance 1 to 15) by best action:

| Situation | Go | Field goal | Punt | Cells decided by under 1 point |
|---|---|---|---|---|
| First quarter, tied | 41% | 23% | 36% | 25% |
| Trailing by 7, 8:00 left | 87% | 4% | 9% | 24% |
| Leading by 4, 4:00 left | 23% | 24% | 53% | 17% |
| Trailing by 3, 2:00 left | 66% | 20% | 14% | 16% |

In the reference situation, going is best up to fourth and 4 at the own 20, fourth and 6
at the own 35, fourth and 10 at midfield, fourth and 7 at the opponent's 35 and fourth
and 2 at the opponent's 5. A quarter of the cells are decided by less than one point of
win probability, which is why M3 exists.

Two things on the chart should not be believed as drawn. First, "go on fourth and 5 to 7
from inside your own 10" (the blue patch at the left edge): punts from the goal line are
poor and the WP model is nearly flat there, and the margins are tiny. Second, the
trailing-by-7 panel says go on almost everything, including fourth and 10 from your own
10 with 8:00 left. That is the one-step engine leaning on a WP model that gives a team
trailing by 7 little credit for field position. Treat it as a known weakness, not a
finding.

### Comparison with nfl4th (`assets/m2_nfl4th_comparison.png`)

8,426 real fourth downs of 2024 and 2025 (the WP model's test seasons).

- Same recommended action (go, field goal or punt): **74.2%**. Same go-or-kick call: 75.6%.
- By how decisive nfl4th is (its |gain from going|, in WP points): under 0.5: 55%;
  0.5 to 1: 65%; 1 to 2: 75%; 2 to 4: 87%; 4 to 8: 96%; 8 or more: 95%.
  At 2 points or more: **90.7%** of 3,221 plays.
- Correlation of the gain from going: 0.77 (Spearman 0.75).
- We recommend going more often: 46.7% of plays against nfl4th's 41.1% (coaches went on
  21.3%).
- Of the 2,177 disagreements, 77% are plays where both models have the gain under 2
  points. The median nfl4th gain on a disagreement is 0.5 points.

Where we disagree most (groups with at least 150 plays): distance 4 to 6 (63% agreement),
trailing by 9 or more (66%), distance 3 (68%), distance 7 to 10 (69%), between the
opponent's 36 and midfield (69%). Agreement is highest at the extremes: fourth and 1
(85%) and 11 or more (87%).

### Why we disagree, with evidence

1. **Our conversion probability is higher.** Mean 0.399 against nfl4th's 0.375; the gap is
   1 point at fourth and 1 and 2 to 4 points from 2 to 10 yards. Swap diagnostic: put
   nfl4th's conversion probability into our engine (keeping our WP for each outcome) and
   agreement rises from 74.2% to 76.1%, and our share of "go" drops from 46.7% to 41.8%,
   almost exactly nfl4th's 41.1%. So the conversion model explains nearly all of our
   extra aggressiveness but only 2 points of the 26 points of disagreement. Note that our
   own held-out check found our conversion model too LOW for 2024 and 2025 (53.8%
   predicted, 56.1% actual), so it is not obvious that nfl4th's lower number is the
   better one.
2. **Field goal probability is not the cause.** In range, ours averages 0.719 and nfl4th's
   0.711; swapping it as well moves agreement by 0.2 points.
3. **The win-probability models differ, and that is most of it.** After both swaps 23.7%
   of plays still disagree. The slope of our gain on nfl4th's is 0.73: our engine sees
   smaller differences between going and kicking than nfl4th does. Our WP model has
   15-leaf trees with at least 10,000 plays per leaf and a Platt map that shrinks toward
   50%, which makes it smooth and flat; its clock splits are about 30 seconds apart, so
   taking 5 or 27 seconds off often changes nothing.
4. **Late-game win probability is too cautious in ours.** First downs in the fourth
   quarter of 2024 and 2025 (`causes.late_game_win_probability_check_test_seasons`):
   teams leading by 1 to 3 with 41 to 120 seconds left won 95.9% (n = 73) and our model
   says 86.2%; with 40 seconds or less they won all 18 and we say 89.0%. Teams trailing
   by 4 to 8 with 40 seconds or less won 8.1% (n = 74) and we say 16.6%. A model that
   thinks a 1-point lead with the ball and 13 seconds left is worth 72% will value a
   punt when trailing by 1 with 22 seconds left at 46% (nfl4th: 0.1%). That is wrong, and
   it shows in the showcase plays below. The engine's bookkeeping is right there (the
   end-of-game tests pass); the input is not.
5. **Clock handling.** We charge fixed average seconds per play. Agreement by clock:
   74.2% in the bulk of the game, 72.4% in the fourth quarter with 2:00 to 5:00 left,
   74.8% in the last 2:00 of a half. In the last two minutes we recommend going less
   often than nfl4th (35.7% against 40.3%), the reverse of the rest of the game, which
   is consistent with point 4.

### The late-lead scenario grid

On the 240 hypothetical late-lead scenarios where nfl4th returned a go value (own 20 and
own 30 only; nfl4th returned NA at the own 40 and midfield), the two models agree on
go-or-punt in only 53%, and the gains correlate at -0.32. nfl4th's mean gain from going
is -10.3 points and ours is -1.7. Our mean punt WP is 0.709 and nfl4th's 0.633. The two
tools see this situation very differently, and neither is a full-game solution. This is
burning question 1 and it is taken up in `docs/notes/m3_bootstrap.md`.

### Ten high-leverage plays, both sets of numbers

Chosen by a fixed rule in the script (fourth quarter, one-score games, largest gain in
either model, one play per game: four playoff plays, three regular-season agreements,
three disagreements). WP in percent. "nfl4th" columns are nfl4th's numbers.

| Play | Situation | Coach | Ours: go / FG / punt | nfl4th: go / FG / punt | Ours | nfl4th |
|---|---|---|---|---|---|---|
| 2025 wild card, BUF at JAX | down 4, 1:10, 4th and 1 at the 11 | go | 39.4 / 27.1 / - | 40.8 / 18.0 / - | go | go |
| 2024 AFC title, KC v BUF | tied, 3:37, 4th and 17 at the 17 | FG | 51.3 / 63.6 / - | 53.5 / 69.8 / - | FG | FG |
| 2025 wild card, GB at CHI | down 4, 1:05, 4th and 7 at own 31 | go | 13.9 / - / 9.9 | 13.1 / - / 0.7 | go | go |
| 2025 NFC title, LA at SEA | down 4, 8:16, 4th and 1 at the 26 | go | 43.5 / 33.2 / - | 36.1 / 30.8 / - | go | go |
| 2024 wk 7, DET at MIN | down 1, 0:19, 4th and 14 at the 26 | FG | 28.7 / 63.1 / - | 13.6 / 72.8 / - | FG | FG |
| 2025 wk 18, KC at LV | down 2, 1:05, 4th and 10 at the 23 | FG | 38.7 / 54.9 / - | 18.7 / 63.5 / - | FG | FG |
| 2025 wk 15, SEA v IND | down 1, 0:22, 4th and 2 at the 38 | FG | 18.2 / 62.6 / 46.5 | 38.3 / 50.5 / 0.1 | FG | FG |
| 2024 wk 5, ARI at SF | down 2, 1:40, 4th and 3 at the 16 | FG | 42.9 / 39.3 / - | 42.0 / 58.0 / - | go | FG |
| 2024 wk 13, ARI at MIN | down 1, 0:40, 4th and 10 at own 30 | go | 25.6 / - / 28.5 | 15.2 / - / 1.0 | punt | go |
| 2024 wk 13, ATL v LAC | down 4, 0:47, 4th and 12 at the 35 | go | 14.6 / 19.2 / 7.6 | 12.1 / - / 0.1 | FG | go |

The three disagreements are all in the last two minutes and all ours are wrong for the
reason in point 4. Punting when trailing by 1 with 40 seconds left is not worth 28.5%,
and a field goal when trailing by 4 with 47 seconds left is not worth 19.2%. A field goal
down 2 with 1:40 left should be worth much more than our 39.3%. Where the model is not
in its weak region (the four playoff plays) the two tools give the same call with
similar numbers.

## 4. Limits, stated plainly

- One step only. Nothing after the next snap is planned.
- The WP model is flat in places and too cautious late in close games (point 4). Any
  recommendation inside the last two minutes of a one-score game should not be trusted.
- The chart panels for trailing teams over-recommend going.
- Conversion and kicking are league-average; no team strength, weather or kicker.
- Overtime is not evaluated. Ties are worth 0.5.
- The comparison with nfl4th is against the CRAN release 1.0.7.
- 74% agreement means two models, not a model and the truth. Neither is ground truth.

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
7. We agree with nfl4th on 74% of plays overall and 91% when it is decisive. Why is the
   second number the more meaningful one?
8. Swapping in nfl4th's conversion probability closes our go-rate gap but barely changes
   agreement. What does that tell you about where the disagreement comes from?
9. Our engine values a punt when trailing by 1 with 22 seconds left at 46%. Is that an
   engine bug? How do you know? What would fix it?
10. Why were the outcome models refitted without 2024 and 2025 for this milestone?
