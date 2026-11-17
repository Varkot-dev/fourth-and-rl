# End-game clock model: rules, measurements and a recommendation for the DP

Research note for M4 and burning question 1 (up 4, about 4:00 left, fourth down in your own half).
Every measured number comes from `scripts/research/endgame_clock_stats.py` and is saved in
`results/research/endgame_clock_stats.json` (2010 to 2025, 729,962 rows). "n" is always a sample size.
Rules are from memory of the NFL rulebook; each carries a certainty mark. Items marked "checked"
were confirmed by a web search on 2026-11-17 against secondary sources (press and officiating
sites), not against the rulebook text itself.

## 1. The rules

- **Play clock** (certain): 40 seconds from the end of the previous play; 25 seconds after an
  administrative stoppage (timeout, change of possession, penalty, two-minute warning).
- **The game clock stops for** (certain): an incomplete pass (restarts at the snap), a change of
  possession, a score (stays stopped through the try and until the kickoff is touched in the field
  of play), a timeout, a penalty, and the two-minute warning (at 2:00 of each half, or when the
  play in progress at 2:00 ends). Each team has 3 timeouts per half.
- **A first down does not stop the clock** in the NFL (certain).
- **Out of bounds** (checked): the clock stops, but restarts when the ball is made ready, except in
  the last 2 minutes of the first half and the last 5 minutes of the second half, where it stays
  stopped until the snap. The data show this clearly (section 2.1).
- **10-second runoff** (checked): after the two-minute warning of either half (since 2017; before
  that, the last minute), if the offense commits a clock-stopping foul while the clock is running
  (false start, intentional grounding, illegal forward pass and similar), 10 seconds are run off in
  addition to the yardage. The same runoff applies to a replay reversal that turns a stopped clock
  into a running one. A team can use a timeout to avoid it (unverified in detail).
- **Overtime** (checked for 2025; older history from memory): regular season is one 10-minute
  period (15 minutes before 2017) and a game still tied is a tie. Until 2024 a touchdown on the
  opening possession ended the game. From 2025 both teams are guaranteed a possession in the
  regular season as well, as in the playoffs since 2022, then sudden death. Playoff overtime uses
  15-minute periods and continues until there is a winner. Each team has 2 timeouts in regular
  season overtime (certain); 3 per two periods in the playoffs (unverified).

### Kneel-down arithmetic

A kneel takes k seconds and leaves the clock running, so the next snap can wait a further 40
seconds unless the defense spends a timeout. With m kneels available (m = 3 on first down, 2 on
second, 1 on third) and the defense holding n timeouts:

    T(n, m) = 40 * max(m - n, 0) + k * m,  capped at 120

The game is over if the leading team snaps with T or fewer seconds left. The cap is the two-minute
warning: it acts as one free timeout, so a series that starts above 2:00 loses most of its first
runoff and cannot finish. We measured k = 2 seconds (median gap from a kneel to the next snap when
the defense called timeout, n = 183). First-down thresholds:

| Defense timeouts | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| T on first down (seconds) | 120 | 86 | 46 | 6 |
| T on second down | 84 | 44 | 4 | 4 |

If the first down is gained on a play that ends in bounds, the clock is already running, so that
play earns one more 40-second runoff unless the defense spends a timeout on it.

## 2. Measured from the data

### 2.1 Seconds from one snap to the next

Plays followed by a timeout, crossing a two-minute warning, with a penalty, or ending in a score or
turnover are excluded; 556 plays with gaps outside 0 to 60 seconds were dropped as timestamp errors.
Cells are median (25th to 75th percentile), n.

| Play | All regulation | Last 5:00, offense leads by 1 to 8 | Last 5:00, offense trails by 1 to 8 |
|---|---|---|---|
| Run, in bounds | 39 (35 to 42), 172,101 | 43 (41 to 45), 2,098 | 36 (27 to 41), 1,745 |
| Complete pass, in bounds | 38 (31 to 42), 119,153 | 43 (38 to 46), 659 | 25 (20 to 35), 3,673 |
| Run, out of bounds | 33 (27 to 38), 13,636 | 7 (6 to 8), 124 | 7 (6 to 8), 270 |
| Complete pass, out of bounds | 29 (16 to 35), 33,218 | 6 (5 to 8), 183 | 6 (5 to 7), 1,461 |
| Incomplete pass | 5 (4 to 6), 94,607 | 5 (4 to 6), 792 | 5 (4 to 6), 4,204 |
| Sack | 39 (33 to 43), 15,330 | 42 (38 to 44), 73 | 30 (23 to 37), 368 |
| Kneel | 39 (31 to 41), 2,262 | 38 (30 to 41), 859 | too few (14) |
| Punt | 9 (8 to 11), 33,360 | 9 (7 to 11), 940 | 9 (8 to 11), 237 |
| Field goal | 4 (4 to 5), 15,177 | 4 (4 to 5), 489 | 4 (4 to 5), 371 |

Over the whole fourth quarter the same pattern is weaker: a leading team's in-bounds run takes 42
(39 to 44), n = 22,155; a trailing team's in-bounds completion takes 30 (23 to 38), n = 22,479.
A spike takes 1 second (n = 974). When the defense calls timeout after an in-bounds run, the gap is
6 seconds (5 to 8), n = 8,479: that is the length of the play itself.

Whole drives by a team down 4 to 8 with 1:00 to 5:00 left run at 13.7 seconds per play
(10.5 to 17.3), n = 928 drives, because most of their plays stop the clock.

### 2.2 Timeouts in practice

Leading by 1 to 8 in the last 5:00, after a play that kept the clock running and was not a first
down, the defense called timeout:

| Defense holds | Before the warning | After the warning |
|---|---|---|
| 1 | 66.9% (n = 708) | 80.6% (n = 723) |
| 2 | 56.3% (n = 1,256) | 93.8% (n = 451) |
| 3 | 40.3% (n = 1,401) | 93.9% (n = 212) |

At the first snap inside 5:00 of a one-score game (n = 2,521 games) the trailing team holds
3, 2, 1, 0 timeouts in 55.0%, 29.6%, 9.4%, 6.0% of games; the leading team in 49.2%, 34.0%, 14.0%,
2.9%.

### 2.3 Play calling (run share on downs 1 to 3, kneels and spikes excluded)

| Score | Quarters 1 to 3 | Q4 10:00 to 5:00 | Q4 5:00 to 2:00 | Last 2:00 |
|---|---|---|---|---|
| Lead 1 to 8 | 43.0% (82,920) | 50.7% (8,192) | 66.0% (6,556) | 87.4% (2,274) |
| Tied | 44.6% (92,148) | 42.2% (2,057) | 41.4% (1,462) | 32.4% (2,059) |
| Trail 1 to 8 | 41.7% (103,798) | 36.8% (8,659) | 27.3% (7,348) | 13.6% (8,941) |

### 2.4 The situation of interest: trailing by 4 to 8, drive starts with 1:00 to 5:00 left

1,131 drives in 1,024 games. The trailing team scored a touchdown on **31.2%** and won 19.0% of the
games.

| Start | TD rate (n) | | Time left at start | TD rate (n) |
|---|---|---|---|---|
| Own 1 to 20 | 23.2% (431) | | 1:00 to 2:00 | 20.7% (357) |
| Own 21 to 30 | 32.6% (389) | | 2:00 to 3:00 | 36.5% (323) |
| Own 31 to 50 | 34.7% (242) | | 3:00 to 4:00 | 38.6% (246) |
| Opponent's half | 60.9% (69) | | 4:00 to 5:00 | 32.2% (205) |

Field position and time interact: from the own 1 to 20 with 1:00 to 2:00 left the rate is 8.3%
(n = 132), with 3:00 to 4:00 it is 36.0% (n = 86). From the own 21 to 30 with 3:00 to 4:00 it is
38.6% (n = 88). By the offense's timeouts the raw rates are flat: 30.3% (0 or 1, n = 555), 33.5%
(2, n = 260), 31.0% (3, n = 316). That is confounded (teams with few timeouts spent them getting
the ball back), so the value of a timeout should come from the clock mechanics, not this table.
The three-way table with every cell's n is in the JSON (`by_field_time_timeouts`); most cells
have fewer than 50 drives.

Touchdown drives took a median of 106 seconds (74 to 151) and 8 plays (6 to 10), n = 342.

**Does the leader get the ball back?** After the 353 touchdowns, the team that had led got a
possession in regulation 92.1% of the time, with a median of 53 seconds left (24 to 96); 65.2% had
30 seconds or more and 42.5% had 60 or more. That team still won 42.6% of those games (this
includes deficits of 6 to 8, where the touchdown may only tie). When it was tied or behind and had
30 seconds or more (n = 210), its answering drive produced a field goal or touchdown 35.2% of the
time and it won 40.9%. When the trailing drive did not score a touchdown (n = 778), the leader won
98.1%.

### 2.5 Kneels against the arithmetic

Game-ending kneel series by a team leading by 1 to 8, first kneel on first down:

| Defense timeouts | n | Median clock | 95th percentile | Max | Formula T | Above T |
|---|---|---|---|---|---|---|
| 0 | 861 | 49 | 120 | 120 | 120 | 0 |
| 1 | 248 | 44 | 82 | 120 | 86 | 8 |
| 2 | 62 | 33.5 | 50 | 57 | 46 | 6 |
| 3 | 4 | 5.5 | 12.9 | 14 | 6 | 2 |

Over all downs, 1.74% of the 1,204 series started above the formula threshold and 0.42% more than
10 seconds above it (unused timeouts, slow kneels, or timeout-count errors). In the other
direction, on first downs inside the threshold with a lead of 1 to 8 (n = 1,217) teams knelt 96.1%
of the time and won 99.92%; on first downs 5 to 45 seconds outside it (n = 421) they knelt 4.8% and
won 98.1%. Teams leading by 9 or more are not a valid test: the defense concedes with timeouts in
hand (median first kneel at 76 seconds with 3 defensive timeouts, n = 197).

### 2.6 Overtime

262 of 4,363 games (6.01%) reached overtime (247 regular season, 15 playoff, 13 ties, 16 in the
2025 season). Counting a tie as half a win, the home team's value was 0.544 and the pregame
favorite's 0.582. By size of spread the favorite's value was 0.629 (0.5 to 2.5 points, n = 62),
0.558 (3 to 6.5, n = 138) and 0.589 (7+, n = 62). A least-squares line through 0.5 gives 0.0157
per point of spread, but the bins are not monotone, so this slope is weakly determined.

## 3. Recommended clock model

**Time step.** Seconds on a 5-second grid for the last 5:00 of the game (60 steps), which fits the
data: stopped-clock plays take about 5 seconds and the runoff is 40. Before that, a 30-second grid
with expectation-preserving random rounding (a 39-second play moves one step with probability 0.7
and two with probability 0.3) and no timeout dimension. At the 5:00 boundary, average the end-game
values over the timeout shares in section 2.2.

**Seconds per play** (ticks of 5 seconds):

| Result | Ticks |
|---|---|
| Incomplete pass, spike, field goal, any play ending out of bounds, score, turnover | 1 |
| Punt | 2 |
| Kickoff (measured median 0 to 1 second) | 0, folded into the scoring transition |
| In-bounds play, offense burning clock | 1 for the play plus 8 of runoff |
| In-bounds play, offense hurrying | the empirical 5-second histograms saved under `five_second_grid` (in-bounds completion: median 25 seconds) |
| In-bounds play followed by a timeout by either side | 1, and that side loses a timeout |

Tempo is a choice: the offense takes the better of "burn" and "hurry" in each state, so no rule
about who is protecting a lead is needed. After an in-bounds play the defense chooses whether to
spend a timeout (the value is the minimum for the offense), and a hurrying offense may spend its
own. Every play costs at least one tick, so one backward pass still solves the model.

**Two-minute warning.** If a play starts above 120 seconds and its runoff would cross 120, the next
state is at exactly 120. No timeout is spent.

**Terminal rule.** If the team in front has first, second or third down and seconds left are at
most T(n, m) from section 1, the game is over and it wins. On the grid use 120, 85, 45, 5 on first
down for n = 0, 1, 2, 3. Apply this in closed form instead of a kneel action, because a one-tick
kneel would overstate each kneel by 3 seconds.

**Overtime.** Value a tie at the end of regulation as 0.5. The coin toss makes it independent of
who had the ball last, and the league-average DP has no favorite. A spread-aware variant can use
0.5 + 0.0157 * spread, with the caveat in 2.6. Sixteen overtime games under the 2025 rule are too
few to estimate anything new.

**State space** (upper bounds): 99 yard lines x 4 downs x 12 distance bins = 4,752 field states;
x 43 score differentials x 16 timeout pairs = 3,269,376 states per time slice. Last 5:00 at
5 seconds: 196,162,560 states (8:00 would be 313,860,096; a 10-second grid halves either). Early
game at 30 seconds without timeouts: 22,476,960. Whole game: 218,639,520 states, about 0.87 GB as
float32, against 11,769,753,600 at one-second resolution with timeouts throughout.

**Main simplifications and their likely effect on question 1.**

1. Runoff fixed at 45 seconds per in-bounds play against a measured median of 43: slightly
   favors going for it (about 2 seconds per play of extra burn).
2. The defense uses timeouts optimally, while real defenses used one only 40% to 67% of the time
   before the warning: the model understates what a conversion is worth against real opponents,
   so it is conservative about going.
3. Downs 1 to 3 need context-specific transitions. Run share for a leader goes from 43.0% to 87.4%
   and for a trailer from 41.7% to 13.6%, so league-average yardage and clock use on those downs
   would be wrong for both sides. Either make run or pass a choice, or fit separate "protect" and
   "chase" transition tables.
4. No penalties, 10-second runoffs, injuries, replay or onside kicks: small for this question.

## 4. What could go wrong, and the test for each

| Risk | Number the solved model should reproduce |
|---|---|
| Trailing offense too slow or too fast, which biases the value of punting | Touchdown rate 31.2% overall for drives down 4 to 8 with 1:00 to 5:00 left; 20.7%, 36.5%, 38.6%, 32.2% by minute; 8.3% from the own 1 to 20 with under 2:00 |
| Scoring drives take the wrong amount of time, which biases the "ball back" branch | Touchdown drives use a median of 106 seconds and 8 plays |
| Leader's answer drive mis-valued, which biases the cost of a failed fourth down | After a touchdown the leader has 30 seconds or more 65.2% of the time; its answering drive scores 35.2%; it wins 42.6% overall |
| Terminal rule too generous or too strict, which biases the value of converting | First down inside T: win rate 99.92%; 5 to 45 seconds outside: 98.1%; no kneel series starts above 120 with 0 timeouts |
| Burn rate wrong | A leader's in-bounds run takes 43 seconds snap to snap (41 to 45) |
| Timeout policy unrealistic | Defensive timeout use of 40.3% to 66.9% before the warning and 80.6% to 93.9% after; report the model with optimal and with empirical timeout use |

Caveats on the data: timeouts and the warning are inferred from changes in timeout counts, because
those rows are not in the state table; some early-season games have broken timestamps (kneel series
with a single timestamp were removed); all rates here are raw averages over mixed teams and
deficits, with no adjustment.
