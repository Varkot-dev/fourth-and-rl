# M1 notes: play-outcome ("component") models

Code: `src/fourth_and_rl/components.py`. Config: `configs/components.yaml`.
Script: `scripts/fit_components.py`. Every number below comes from
`results/m1_components.json`, written by that script (design-decision numbers are under
its `diagnostics` key). Figure: `assets/m1_components.png`.

## 1. What this milestone is for

A fourth-down decision compares three actions. To compare them you need to know, for
each action, what can happen next and how likely each outcome is. These models are that
knowledge. In the language of a Markov decision process (MDP) they are the
**transition model**:

    P(next state | current state, action)

Example. State: fourth and 3 at the opponent's 40 (yardline_100 = 40). Action: go.
The fitted model says (file key `go_outcomes_example_fourth_and_3_at_yardline_100_40`):

- 54.7% convert. Of that, 2.4% is a touchdown on the play and 52.3% is a first down,
  on average at yardline_100 = 29.5.
- 45.3% the other team gets the ball, on average 59.4 yards from their end zone,
  including 0.3% returned for a touchdown.

Those four numbers hide two arrays of length 100: one probability per yard line for
"first down here" and one for "opponent takes over here". The arrays are what later
milestones use.

### Why distributions and not averages

Value is not a straight line in field position. A small worked example with made-up
values: suppose the opponent's expected points are 0.5 from their own 10, 1.5 from
their own 40 and 3.0 from our 30. A punt that lands them on their 10 half the time and
on our 30 half the time has an "average" landing spot of the 40, value 1.5. The true
expected value is 0.5 * 0.5 + 0.5 * 3.0 = 1.75. Using the average spot is wrong by a
quarter of a point, and the error always has the same sign when the curve bends one
way. So every model returns the whole distribution and the value function is averaged
over it (that averaging is the "expectation" in the Bellman equation).

## 2. Concepts

### Logistic regression, and why smooth at all

There are 99 yard lines and about 30 distances: roughly 3,000 cells. Fourth-down
attempts number 9,948 in 16 seasons and most sit in a few cells (fourth and 1 has
3,836). Most cells are empty or have a handful of plays, so the raw rate in a cell is
mostly noise. Fourth and 12 "converts" 32% in the raw data and fourth and 9 only 31%
(between the 20s), which nobody believes.

Smoothing means assuming neighbouring situations are similar and letting them share
data. Logistic regression does it with a formula. It models the log-odds:

    log( p / (1 - p) ) = a + b1*x1 + b2*x2 + ...
    p = 1 / (1 + exp(-(a + b1*x1 + ...)))

Log-odds can be any number, so the right-hand side can be a plain sum, and p always
lands between 0 and 1. Worked example: log-odds 0 is p = 0.5; log-odds 1 is
1 / (1 + exp(-1)) = 0.73; log-odds -1 is 0.27. A coefficient is a shift in log-odds:
near p = 0.5, a shift of 0.1 moves p by about 0.1 / 4 = 2.5 points.

A **spline** is how a smooth curve enters that sum: the x axis gets a few "knots" and
the curve is a cubic between knots, joined smoothly. More knots where the curve bends
fast (1, 2, 3, 4 yards to go; inside the 20) and few elsewhere.

### Selection bias in fourth-down attempts

Coaches choose when to go. That makes the fourth-down sample unrepresentative in at
least three ways:

1. Distance: they go when it is short. Long attempts are mostly desperate (late,
   trailing, against a defense happy to give up 8 yards).
2. Team and score: in our data teams that go on fourth and 3 or less were on average
   0.4-point underdogs before the game and trailing by 4.2 points at the time; teams
   that punt were ahead by 0.7 and teams that kick by 1.6.
3. Unobserved: a coach goes when he likes the matchup or has a play he trusts. Nothing
   in the data records that.

So "teams convert X% on fourth and 2" answers "how do the teams that chose to go do",
not "what would happen if this team went now". The decision engine needs the second.

### Censoring at the goal line and the Kaplan-Meier estimator

From the 3-yard line a play cannot gain 12. Any play "good for" 3 or more is recorded
as a 3-yard touchdown. The long gains are not missing, they are **censored**: we know
only "at least 3". If you histogram recorded gains for plays between the 11 and the 20
and then ask "from the 20, how often does a play gain 20 or more?", you get about a
tenth of the right answer, because only the plays that started at the 20 could ever
show a 20-yard gain.

Kaplan-Meier fixes this by working one yard at a time. For each gain g:

    h(g) = P(gain is exactly g | gain is at least g)

estimated only from plays that could have shown exactly g: they gained at least g and
started more than g yards out. Then P(gain >= g) is the product of (1 - h) over all
smaller gains.

Worked example (it is a unit test). True gain is uniform on 0 to 9, every play starts
at the 1, 2, 3, 4 or 5. The share of plays seen to gain 5 or more is about 10% (only
those from the 5), but the truth is 50%. Kaplan-Meier: h(0) = 1/10, h(1) = 1/9, ...,
h(4) = 1/6 from the plays still able to show each gain, so
P(gain >= 5) = (9/10)(8/9)(7/8)(6/7)(5/6) = 0.5. Correct.

### Log loss and calibration

Log loss is the average of -log(probability the model gave to what happened). Saying
0.9 and being right costs 0.105, saying 0.9 and being wrong costs 2.3. Lower is better
and a model cannot game it by being overconfident. The baseline is a model that always
says the training-season average. Calibration is the direct check: among plays where
the model said about 60%, did about 60% happen?

## 3. Design decisions

### 3.1 Conversion probability (go)

**Model.** Logistic regression on third and fourth downs together (113,069 third downs,
9,948 fourth downs, 2010 to 2025):

    log-odds = a + spline(yards to go) + spline(yardline_100)
               + b1*fourth + b2*season + b3*spread

Predictions use fourth = 1, the last season fitted (2025) and spread 0.

**Why third downs.** Third down is the closest thing to a fourth-down attempt that is
not chosen: every team must run a play on third and 4. It has 11 times the data and
covers every distance and field position. The fourth-down term then asks "at the same
distance, field position and team strength, how different is fourth down?". Answer:
b1 = +0.059 log-odds, about 1.5 percentage points. Third and fourth down convert at
almost the same rate.

**Why the spread.** It stands in for team strength (selection problem 2). Its
coefficient is 0.155 per 10 points: a 7-point favorite converts about 2.5 points more
often than an even team. Without it the fourth-down term is 0.037, with it 0.059,
because teams that go are slightly weaker than average. Predictions are for an even
matchup; `GoModel.p_convert(..., spread=...)` gives other matchups.

**What is still uncontrolled.** Selection problem 3 (the coach's private information),
and the fact that defenses call fourth and 2 differently from third and 2. We cannot
test the model where coaches never go (fourth and 12 from your own 20); there the
number is the third-down shape plus a constant, an assumption.

**Definition of "convert".** First down or touchdown, and the offense still has the
ball. A first down by penalty on a play that counted is included. Plays wiped out by a
penalty are excluded, so a defensive offside on fourth and 1 does not count for us.

**Monotone in distance.** After fitting, each field position's curve is passed through
a running minimum over distance, so it can never go up. The largest change this made
anywhere was 3.0 points, out in the 23+ yards-to-go tail where the spline has almost
no data. Distances beyond 30 are treated as 30.

**Season trend.** One linear term. It is small (+0.026 log-odds per decade). Kept
because the plan says to model eras, but it barely matters.

**Fitted values (yardline_100 = 50).** Fourth and 1: 71.4%. And 2: 60.4%. And 3: 55.3%.
And 5: 47.6%. And 10: 31.9%. And 15: 18.0%. At the opponent's 5 it is harder (less
field to defend): fourth and 1 is 62.0%, fourth and goal from the 5 is 37.3%.

**Where the ball goes.** The yards-gained distribution for third and fourth downs
(2018 on, Kaplan-Meier, by distance bin and field-position bin) is cut at the line to
gain. The part at or beyond the line is rescaled to total P(convert); the part short
of it is rescaled to total 1 - P(convert). Gains reaching the goal line are the
touchdown. Of failures, a share is an interception or lost fumble, taken from
fourth downs only: 7.8% at 1 to 2 yards to go, 7.9% at 3 to 6, 10.8% at 7 or more
(third-down failures are turnovers 4.6% of the time; on fourth down an
interception costs little more than an incompletion, so quarterbacks take more risk).
Turnovers are placed using the observed "yards downfield where the other team took
over", and `turnover[0]` holds a return touchdown.

Interceptions and fumbles are not separate keys. Both end with the other team holding
the ball at some yard line, which is all the MDP state records, so they are merged
into the one `turnover` array.

### 3.2 Field goal

**Kick distance.** yardline_100 + 18, not 17. In the data `kick_distance` minus
`yardline_100` is 18 for 90.1% of attempts, 19 for 7.7%, 17 for 2.1%.

**Model.** With x = (distance - 40) / 10 and t = (season - 2018) / 10:

    log-odds = a + b1*x + b2*x^2 + b3*x^3 + b4*t

fitted on all 16,927 attempts, evaluated at 2025. Blocked kicks are misses.

**Why a cubic.** The log-odds fall steadily to about 55 yards and then drop off a
cliff. A straight line misses the cliff, and a quadratic bent the wrong way at long
range (it flattened out exactly where the data drop). The fitted cubic is decreasing everywhere, so the monotone step changed
nothing (largest change 0.0).

**Why a trend and not "recent seasons only".** Kickers improved: b4 = +0.31 log-odds
per decade. At 2010 level a 50-yarder is 65.6%, at 2025 level 75.3%. Three versions
were scored on 2024 and 2025 after fitting through 2023: all seasons with trend
0.3695, recent seasons only (2018 on) 0.3708, all seasons no trend 0.3734. The trend
version wins and uses all the data. I also tried letting the trend differ by distance;
it did not help out of sample and was dropped.

**Fitted values (2025 level).** 30 yards: 96.0%. 40: 87.9%. 50: 75.3%. 55: 65.9%.
60: 52.7%. 65: 35.7%. Beyond 68 yards (the longest make in the data) the model
returns 0.

**Caution at long range.** Kicks from 58+ are attempted only when the coach thinks his
kicker can make it (strong leg, dome, end of half). The 60-yard number is for kicks
like those, not for any kicker on any day. This is the same selection problem as on
fourth-down attempts and there is no third-down trick to fix it.

**After a miss.** The rule: the other team takes over at the spot of the kick, or at
its own 20 if the kick was from inside the 20. The spot is 8 yards behind the line of
scrimmage, so their yardline_100 is min(92 - ours, 80). Checked in the data: after
2,003 misses, from outside the 12 the other team started 7 to 9 yards behind the old
line of scrimmage 99.4% of the time (exactly 8: 88.2%); from the 12 or closer it
started at its own 20 97.4% of the time. Worked example: miss from the 30
(a 48-yard try) gives them the ball 62 yards from scoring, their own 38. Returns of
blocked or short kicks are ignored.

### 3.3 Punt

**Target.** The receiving team's yardline_100 at its next snap. This one number already
includes the return, touchbacks, blocks, shanks and penalties on the play.

**Smoothing.** For each field position, punts from nearby yard lines are pooled with a
Gaussian weight (2 yards wide, widened until it covers 400 punts). What is pooled
depends on where you are:

- yardline_100 of 50 or more: the **net yards**. With room, a punt nets about 43.7
  from anywhere, so a punt from the 68 is a good stand-in for one from the 70 after
  shifting by 2 yards. Touchbacks are pooled as their own outcome and placed on the 20.
- Closer than 50: the **landing spot**. The punter aims at the same patch inside the
  10 from the 42 as from the 44, so the landing spot is what is shared.

**Seasons.** 2018 on (17,551 punts). Mean net was 37.5 yards in 2010, 40.4 in 2018 and
41.4 in 2025, so the early seasons would bias it down.

**Indexing.** `punt_outcomes(y)[k]` for k = 1..99 is P(receiving team starts at their
yardline_100 = k). Index 0 is a punt returned for a touchdown (0.5% overall). The array
sums to 1 given that the receiving team gets the ball. In 1.0% of punts the kicking
team recovers a muff or fumble; `punt_outcomes_full` adds those as a second array.

**Fitted values.** Expected net from own 20: 43.7 yards. Own 30: 43.8. Own 40: 42.2.
Midfield: 36.7. Opponent's 40: 28.9. From midfield 15% are touchbacks and 39% pin the
opponent inside the 10.

**Limit.** Almost nobody punts from inside the opponent's 35 (71 punts in 16 seasons,
5 of them from inside the 30), so the model there is the nearest data stretched. No sensible
policy punts there, but the numbers should not be quoted.

### 3.4 Generic play, downs 1 to 3

One "typical play" per context, runs and passes mixed as teams actually call them
(the model never chooses run or pass). Context is down (1, 2, 3), distance bin
(1, 2, 3, 4-5, 6-7, 8-9, 10, 11-15, 16+) and field bin (yardline_100 1-4, 5-10, 11-20,
21-40, 41-75, 76-95, 96-99). Seasons 2018 on, 270,393 plays.

Per cell: the turnover share, and the Kaplan-Meier distribution of yards gained on the
other plays. To use it at a yard line y: gains of y or more are a touchdown, gains that
would put the ball behind the offense's own goal line are a safety, everything else
moves the ball. A cell needs 300 plays to stand alone (102 of 189 do); smaller cells
borrow from the same down and distance over all field positions.

**Touchdown calibration.** Kaplan-Meier assumes a play's potential gain does not depend
on where in the bin it started. Near the goal line that is slightly false: the defense
has less field to cover, so a 12-yard gain is harder from the 12 than from the 20. The
uncorrected cells predicted 23.2% touchdowns per play from the 5 to 10 when 22.0%
happened, and 9.8% from the 11 to 20 when 9.5% happened (in sample). So each exact yard line gets a ratio, observed touchdowns over
predicted touchdowns (pulled toward 1 by 30 pseudo-touchdowns so it does not chase
noise), and the touchdown share is multiplied by it.

**Left out, on purpose, and to be tested in M4:**
- Penalties. Plays wiped out by a penalty are not in the model at all, and penalty
  yards on plays that counted are not added. Real drives get automatic first downs
  that simulated drives will not.
- Clock, score and timeouts do not change the play mix. Real teams pass more when
  trailing and kneel when leading.
- One average offense against one average defense.

**Smaller pieces.**
- Seconds per play (2018 on): 27.0 on average. 34.7 when the clock keeps running, 12.4
  when it stops (incomplete, out of bounds, score, turnover). The JSON has the table by
  play type and by whether it is the last two minutes of a half. Timeouts are not
  removed.
- Kickoff: 2025 only (2,927 kickoffs). The touchback spot moved to the 30 in 2024 and
  the 35 in 2025, and the data show it: the most common start was the 25 from 2016
  to 2023 (the 20 before that), the 30 in 2024 and the 35 in 2025. Mean start in 2025: 69.1 yards from the end zone
  (own 31). Onside kicks and other kicking-team recoveries (0.7%) are left out.
- Extra point 94.6% (10,378 kicks, 2018 on). Two-point try 49.0% (1,083 tries).

### 3.5 Drive-level targets

Computed from real drives, not from the models, and saved for M4 to compare the
simulator against. Seasons 2021 to 2025, 30,731 drives: touchdown 22.1%, field goal
15.5%, missed field goal 2.6%, punt 35.6%, turnover 10.2% (plus 1.1% returned for a
touchdown), downs 5.8%, safety 0.2%, end of half 6.8%. Points per drive 2.01. Plays per
drive 5.80. By starting position, points per drive run from 1.46 (own 1 to 20) to 4.71
(inside the opponent's 20). All-season versions are in the JSON too.

## 4. What the held-out check showed

Fit on 2010 to 2023, score on 2024 and 2025. The saved bundle is then refitted on all
seasons.

| Model | Held-out n | Log loss, model | Log loss, constant baseline |
|---|---|---|---|
| Conversion, fourth downs | 1,746 | 0.6434 | 0.6917 |
| Conversion, third downs (same model) | 14,309 | 0.6109 | 0.6742 |
| Field goal | 2,306 | 0.3695 | 0.4244 |
| Punt (100 outcomes) | 4,112 | 3.529 | 3.760 (one net distribution), 3.926 (one landing distribution) |
| Play model, first down or not | 67,658 | 0.5448 | 0.5982 |

Honest reading:

- **Conversion under-predicted 2024 and 2025.** Predicted 53.8% overall, actual 56.1%.
  Worst at fourth and 2 (56.0% predicted, 61.2% actual, n = 258) and 3 to 4 (49.8% vs
  53.6%, n = 347). Fourth and 1 was close (67.8% vs 69.1%). With 1,746 plays the
  standard error of the overall rate is about 1.2 points, so a 2.3-point gap is about
  two standard errors: suggestive, not proven. Possible reasons: teams got better at
  short-yardage plays, or coaches got better at picking spots (selection again). The
  final fit includes those two seasons.
- **The variants cannot be told apart on fourth downs.** Default 0.6434, fourth downs
  only 0.6423, third downs only 0.6445, no trend 0.6439, no spread 0.6438. Differences
  of 0.001 to 0.002 on 1,746 plays are noise. The case for the pooled model is not that
  it scores better on attempted fourth downs; it is that it has data where coaches
  rarely go. The held-out fourth downs cannot test that, because they are also plays
  coaches chose.
- **Field goals from 55+ were under-predicted** (54.9% predicted, 61.8% actual, n =
  207). Long kicks kept improving faster than a trend frozen at 2023.
- **Punts** are calibrated within about 1 yard of expected landing spot in every field
  bin.
- **Play model** runs slightly hot on held-out seasons: first-down rate 29.5% predicted
  vs 28.6% actual, touchdowns from the 1 to 4 45.3% vs 42.9%, from the 11 to 20 9.6% vs
  8.8%. In sample the touchdown rates match, so this is mostly 2024 and 2025 being a
  little lower-scoring per play than 2018 to 2023, not a flaw in the method. It is a
  warning for M4: simulated drives may score a bit too easily.

## 5. Things to be able to say without notes

1. Why are fourth-down conversion rates biased, and what did you do? (Selection on
   distance, team and private information; pooled with third downs, controlled for the
   spread, said what remains.)
2. How big is the fourth-down effect once you control for distance, field position and
   team strength? (About 1.5 points; essentially the same as third down.)
3. Why does a 60-yard field goal probability deserve less trust than a 40-yard one?
4. Why smooth net yards from deep but landing spot near midfield?
5. What is censoring, and what would go wrong at the goal line without Kaplan-Meier?
6. Why does the MDP need distributions when the decision engine reports one number?
7. What does the play model leave out, and how will you know if it matters? (Penalties,
   game situation, team quality; the drive-level targets in M4.)
8. What did the held-out seasons show that you did not expect?
