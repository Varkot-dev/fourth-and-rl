# M13 notes: goal-line field goals and two-point tries, read from the solved game

Code: `src/fourth_and_rl/endgame_questions.py` (helpers), `scripts/questions/q5_goal_line_and_two_point.py`
(the answers). Config: `configs/q5.yaml`. Tests: `tests/test_endgame_questions.py`.
Every number below comes from `results/questions/q5_goal_line_and_two_point.json`.
Figures: `assets/questions/q5_goal_line.png`, `assets/questions/q5_two_point.png`.

## 1. The questions

These three calls are argued on X with score arithmetic, never with a model
(`research-kit/x/2_twitter_findings.md`, decisions 6 and 7, question 7, section 8.1):

1. **Up 7 late at the goal line: kick or go?** Brian Daboll kicked from the 1 to go up 10
   in the fourth quarter ("take the points, get a 10-point lead"). Todd Bowles went for it
   up 14 with about 8 minutes left and failed. Also up 3 and up 4.
2. **Down 8 after a touchdown: go for two now, or kick and go for two later?** That is a
   team that was down 14 and just scored. Kicking leaves it down 7; a successful two leaves
   it down 6, a failed one down 8. "Down 14, go for two on the first touchdown" and "down 8
   after the touchdown" are the same decision described before and after the six points.
   Also the margins down 2, 5, 9 and 10 after a touchdown, and two leads.
3. **Fourth and goal early in a close game: take the points or go?**

The Bowles case is from October 2026, which is in the sealed holdout season, so it is
rebuilt only as a hypothetical state (up 14, fourth and goal at the 1, 8:00 left). No 2026
data is read.

## 2. Why score arithmetic misleads

Score arithmetic treats points as if each one were worth the same: "a field goal makes it
a two-score game", "only the third point matters". The value of a point depends on the
score and the clock. A concrete way to see it, from the solved game, for a trailing team
kicking off after its try with 8:00 left (its win probability):

    down 8   7.6%
    down 7  10.3%
    down 6  14.7%

The seventh point (from down 8 to down 7) is worth 2.7 win-probability points; the
eighth (from down 7 to down 6) is worth 4.4. Same one point, different value, because
down 6 means one touchdown and the extra point wins, down 7 means it ties, down 8 means it
needs another two-point try just to tie. Every argument in this milestone is a
comparison of such values, weighted by the chance of each outcome.

The second way arithmetic misleads is that it follows one path ("we score once more and
they do not") and ignores how often that path happens. Section 4 measures both.

## 3. How the solved game answers each question

The game model of M4 (`docs/notes/m4_mdp.md`) is solved for every state: time in 5-second
ticks, score, down, distance, yard line, both teams' timeouts in the last 5:00. For any
state it returns the win probability of each action, with both teams playing optimally
afterwards. Three pieces are used.

- **Fourth and goal.** The value of each action on fourth down is one row of the Bellman
  backup (`Solution.q_slice`). Going for it from the y-yard line means yards to go = y;
  the best of the allowed play modes (protect, neutral) is the "go" value. The field goal
  value is the other row. `endgame_questions.goal_line_values` reads both for a grid of
  times, scores, timeouts and yard lines.
- **The try.** After a touchdown the solver takes the better of the kick and the two-point
  try: each is a mix of "kick off at margin + 1 (or + 2)" and "kick off at the margin".
  The solver keeps only the better value (`_Post.td`) and a flag (`_Post.two`), so
  `endgame_questions.try_values` rebuilds both values from the public kickoff values
  (`Solution.post(t).ko`, flipped to the scorer's side with `Solution.opp`). A test checks
  that the better of the two equals the solver's own array at every time, margin and
  timeout state, on a toy game and on the real solution.
- **Taking a case apart.** `Solution.play_split` splits going for it into "touchdown" and
  "stopped"; `endgame_questions.field_goal_split` does the same for the kick (the value
  after a miss is private to the solver, so it is recovered from the action value:
  if_missed = (value - make * if_made) / (1 - make); a test checks it against an
  independent lookup).

The answer to "kick or go?" is then: go is better when

    P(touchdown) * WP(touchdown) + (1 - P(touchdown)) * WP(stopped)
        >  P(good) * WP(good) + (1 - P(good)) * WP(missed)

and the break-even touchdown chance is (field goal value - WP(stopped)) / (WP(touchdown) -
WP(stopped)). The win probabilities on the right of each "*" are where the score and the
clock enter; they come from the solved game, not from arithmetic.

## 4. Worked decompositions

### Question 1: up 7, fourth and goal at the 1, 8:00 left

    Go:          P(touchdown) 57.8%
                 WP if it scores (up 14 after the kick)   97.2%
                 WP if stopped (they take over at their 1) 87.0%
                 go = 0.578 * 0.972 + 0.422 * 0.870 = 0.929
    Field goal:  P(good) 99.6%,  WP if good (up 10) 92.1%,  if missed 83.6%
                 field goal = 0.921
    Go minus field goal: +0.8 points. Break-even touchdown chance: 49.8%.

In words: the field goal buys a 10-point lead worth 92.1%. Going is a 58% shot at a 14-point
lead worth 97.2%; if it fails, the opponent is still down 7 but starts at its own 1, and
that is worth 87.0% to the leader, only 5 points less than the made field goal. The failed
attempt is a disguised punt to the 1. That is why going is close even though "up 10 is a
two-score game".

The same call later flips: at 4:00 the field goal wins by 1.7 points (the leader's
remaining time is short, so a 10-point lead is nearly as safe as 14 and failing hurts
more). From the 1, going is better with 7:00 or more left and the field goal with less;
from the 2 the switch is at 9:00, from the 3 at 11:00. So the Daboll call (clock not stated
in the posts) was right in the model if it came inside about 7:00, and wrong by up to 2.4
points if it came with 12:00 left.

### Question 1 comparison: up 14, fourth and goal at the 1, 8:00 left (Bowles)

    Go:  57.8% * 99.9% (up 21)  +  42.2% * 98.4% (up 14, they start at their 1) = 99.29%
    FG:  99.6% * 99.65% (up 17) +  0.4% * 97.7%                               = 99.64%
    Field goal by 0.35 points; break-even touchdown chance 82%.

The field goal is better, by a third of a point: a coin flip graded as a decision. Every
cell of the up-14 map is within 0.7 points. For comparison, these are Ryan Paganetti's
numbers, quoted as his and not blended with ours (decision 6 of the X findings): conversion
at the 1 about 57.6% (54 to 55% without a sneak); a touchdown wins over 99%, a failure up
14 about 96%, a made field goal about 98.5%; he would have kicked. Our model has the same
touchdown chance (57.8%) and the same answer, but higher win probabilities after a failure
(98.4% against his 96%), because it pins the opponent at its own 1 and has identical teams.

### Question 2: down 8 after a touchdown, 8:00 left

    Kick:  94.2% good -> down 7, WP 10.3%;  5.8% miss -> down 8, WP 7.6%    = 10.1%
    Two:   50.9% good -> down 6, WP 14.7%; 49.1% fail -> down 8, WP 7.6%    = 11.2%
    Two minus kick: +1.1 points. Break-even two-point success rate: 35.7%.

The score-arithmetic argument, on its own path (you score exactly one more touchdown,
they score nothing more, a tie is worth 0.5 because the model has no overtime):

    go for two now:  0.509 * (win with the next kick, 0.971)
                   + 0.491 * (need two to tie: 0.509 * 0.5)       = 0.619
    kick now:        0.942 * (next try: go for two to win, 0.509)
                   + 0.058 * (need two to tie: 0.509 * 0.5)       = 0.494
    difference on that path: +12.5 points

The arithmetic is right about the direction and about the reason (after a failed two you
still have the second touchdown to fix it, and you know what you need). It is wrong about
the size. Simulating the solved game from the kickoff after the kick (100,000 games), the
path "exactly one more touchdown for the trailing team, nothing else for either side"
happens in 10.0% of games. Off that path the try barely matters: the trailing team loses
anyway, or the leader scores again, or a field goal changes the arithmetic. Roughly
0.10 * 12.5 = 1.25 points, close to the exact +1.1. (The simulated win rates from down 6,
7 and 8 match the exact values within two standard errors: 14.8% against 14.7%, 10.4%
against 10.3%, 7.5% against 7.6%.)

So the honest answer to "why not go for two every time?" is: the reason to go for two down
8 is specific to the score (it only pays because the next touchdown's try can fix a
failure), and the gain is about one win-probability point, not the 12 the arithmetic
suggests.

### Question 3: tied, fourth and goal at the 1, first quarter 10:00

    Go:  57.8% touchdown -> WP 65.0%;  42.2% stopped -> WP 51.1%        = 59.1%
    FG:  99.6% good -> WP 54.7%                                          = 54.7%
    Go minus field goal: +4.4 points. Break-even touchdown chance: 26%.

Early in the game a failed attempt costs little: the opponent starts at its own 1 and
the stopped team is still favoured (51.1%, above 50%), because the opponent is likely to
punt from deep in its end. The touchdown is worth 4 more points of score than the field
goal, and early on every point is worth roughly the same. So going needs only a 26% chance.

## 5. Results

**Question 1 (leading late, 2:00 to 12:00 left, own team with 3 timeouts).**

- Up 3 or up 4: go everywhere on the map, from the 1, 2 and 3. Up 3 from the 1: by 7.0 to
  13.0 points. Up 4 from the 3 is the closest: within 0.25 points around 5:00 to 7:00.
- Up 7: from the 1, go with 7:00 or more left (up to +2.4 at 12:00), field goal with less
  (up to -2.3). From the 2 the switch is at 9:00, from the 3 at 11:00; inside 4:00 the field
  goal wins by 1.7 to 3.9 points. Opponent timeouts move the edge only inside 5:00 and never
  flip the call from the 1.
- Up 14: field goal everywhere, by at most 0.7 points. A toss-up by any reasonable standard.
- Coaches (development seasons 2010 to 2023, fourth and goal from the 1 to 3, fourth
  quarter, 2:00 to 12:00 left): up 3, 5 of 7 went; up 4, 2 of 7; up 7, 0 of 9; up 14, 0 of
  4. The samples are tiny.

**Question 2 (after a touchdown, fourth quarter, both teams with 3 timeouts).**

- Two is better at every time from 15:00 to 0:30 for margins (before the try) of -10, -8,
  -5, -4, -2, +1, +4, +5, +7, +12, +13 and +15; the kick for 0, +3, +6, +8, +9, +11 and +14.
- Down 8: two by 0.7 to 1.2 points; down 10 (two makes it down 8, one score): up to 1.5;
  down 5 (two makes it down 3, a field goal ties): up to 2.2; down 2: up to 16.4 (tie the
  game); down 9: mixed, within 0.25 points; up 1 (to lead by 3): up to 8.6; up 5 (to lead by
  7): up to 4.1. Down 2, up 1 and up 5 are two-point calls for every timeout pair.
- Timeouts matter only near the end: 66 of 930 cells change their call under some pair of
  timeouts.
- Coaches (2015 to 2023, fourth quarter): down 8, 17% went for two (n = 156); down 9, 24%
  (n = 66); down 10, 100% (n = 56); down 5, 99%; down 2, 98%; up 1, 100%; up 5, 94%. For
  comparison, Paganetti's count (his numbers, posted 2026-10-09): the down-8 two-point try
  has been used 53 times by 23 franchises since 2018, and in 2025 teams went for two in this
  spot in the fourth quarter roughly half the time.

**Question 3 (first half, within 3 points).**

- From the 1: go at every time and margin, by 2.5 to 6.2 points.
- From the 2: go everywhere, by 0.1 to 4.0.
- From the 3: go everywhere except with 1:00 left in the half, by up to 3.1; the field goal
  is better by up to 0.8 with 1:00 left (no time left to use the field position after a
  stop).
- Coaches (2010 to 2023): from the 1, 76% went (65% in 2010 to 2017, 89% in 2018 to 2023);
  from the 2, 24% (8% then 47%); from the 3, 10% (0% then 21%).

**Sensitivity** (each row re-solves the whole game; headline margins in points):

| Change | Up 7 at the 1, 8:00 | Up 14 at the 1, 8:00 | Down 8, two minus kick | Tied at the 1, Q1 |
|---|---|---|---|---|
| Base model | +0.8 | -0.35 | +1.1 | +4.4 |
| Conversion chance -0.05 | +0.4 | -0.39 | +1.1 | +3.8 |
| Conversion chance +0.05 | +1.2 | -0.30 | +1.1 | +5.0 |
| Two-point success -0.05 | +0.7 | -0.32 | +0.8 | +4.4 |
| Two-point success +0.05 | +0.9 | -0.38 | +1.2 | +4.5 |
| Defense timeouts at measured rates | +0.8 | -0.34 | +1.0 | +4.4 |

No headline flips. The maps move more than the headlines: the share of the late goal-line
map where going wins runs from 52% to 61%; the share of the try map where two wins runs
from 42% (two-point success 46%) to 75% (56%), because many cells are within a fraction of
a point. The re-solved base model equals the saved solution exactly on all three maps.

## 6. Limits

- **Identical teams.** Both teams are league average. A strong offense or a weak defense
  changes every value; M8 showed the spread captures most of that for the one-step engine,
  but the game model has no team strength at all.
- **The touchdown chance comes from the fourth-down play-outcome model** (57.8% from the 1,
  46.0% from the 2, 41.5% from the 3), the same for every team and situation. It agrees
  with Paganetti's 57.6% at the 1, but sneaks, personnel and the defense's goal-line package
  are not in it.
- **The two-point rate is the raw share of successful two-point attempts in the
  development seasons**: 50.9% of 805 attempts. Teams choose when to go for two, so this is
  a selected sample. It is also inconsistent with the model's own fourth and goal from the
  2 (46.0%), though both are plays from the 2. The early-game result that two is slightly
  better at almost every margin (by 0.1 to 0.3 points) follows directly from 2 * 0.509 >
  0.942; at 46% it would not. The sensitivity rows show how much depends on it.
- **No overtime**: a tie after regulation is worth 0.5. Under real overtime rules the team
  that ties late is not exactly even, which moves the "kick to tie" values.
- **First half simplifications**: no timeouts and no two-minute warning in the first half
  of the model, which matters for question 3 near halftime.
- **The extra point is 94.2% everywhere**, from the same development seasons, and kick
  distance, weather and kicker are not in the model.

## 7. Design decisions someone could challenge

1. **Full-game re-solves for sensitivity.** Question 3 is in the first half, which depends
   on everything after it, so each sensitivity row re-solves all 60 minutes (about 40
   seconds and 2.2 GB on 2 threads). Questions 1 and 2 alone would need only the last 12 to
   15 minutes; `endgame_questions.truncated` does that, and a test proves it gives the same
   values.
2. **Headline clock 8:00** for both late questions, because the Bowles case is at about
   8:00 and Daboll's clock is not in the posts. The map shows every time.
3. **Question 3 is averaged over who receives the second-half kickoff**; the two values
   differ by at most 0.7 points.
4. **The score-arithmetic path** is defined as: the trailing team scores exactly one more
   touchdown and nothing else, the leader scores nothing more. Other definitions (allowing a
   field goal) would give other shares; the point is the order of magnitude.
5. **Margins limited to plus or minus 15 before the try**, because the model clips the score
   at plus or minus 21 and a wider range would read clipped values.

## 8. Things to be able to say without notes

1. Why one point is worth different amounts at different scores, with the down 6, 7, 8
   numbers.
2. The decomposition of up 7 at the 1, and why a failed attempt is "a punt to the 1".
3. Why the down-8 two-point gain is about 1 point when the arithmetic says 12.
4. Why the early-game two-point result depends on 2 * p(two) against p(kick).
5. Which assumption moves each answer most (conversion chance for questions 1 and 3,
   two-point rate for question 2).
