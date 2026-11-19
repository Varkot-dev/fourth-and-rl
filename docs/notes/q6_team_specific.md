# Question 6 notes: the exact game with two different teams

Code: `src/fourth_and_rl/mdp_team.py` (matchup kernels, two-team solve, two-team
simulator), with two small hooks in `src/fourth_and_rl/mdp_game.py` (`Solution.other`,
`backward_induction`). Script: `scripts/questions/q6_team_specific.py`.
Tests: `tests/test_mdp_team.py`. Every number below is from
`results/questions/q6_team_specific.json`; the figure is
`assets/questions/q6_team_specific.png`. Worked examples marked "by hand" are arithmetic.

## 1. The question

The M4 solver answers "go or punt?" for two identical league-average teams. Coaches ask
something narrower: should Detroit, with its offense, go for it against this defense? M8
measured team strength (offense and defense ratings for drive points and short-yardage
conversion) and pushed it through the one-step engine, which only adjusts the next drive.
This note does it inside the exact game: both teams' offenses and defenses enter every play
of the rest of the game, and the game is solved again by backward induction.

## 2. Why the symmetry trick breaks

The M4 solver stores one array V(t, f, s, e): the win probability of the team with the
ball. When the ball changes hands it needs the other team's value, and it gets it from the
same array:

    value to us when they have the ball = 1 - V(t, f', -s, e swapped)

That is `Solution.opp`. It is correct only because the other team, put in the mirror
state, is in exactly our situation: same plays, same kicker, same everything. Once team A's
offense is better than team B's, "B has the ball at its 30, down 4" is not the same
situation as "A has the ball at its 30, down 4", so no single array can hold both.

Worked example 1 (by hand, the test `test_asymmetric_game_by_hand`). Two yard lines, one
down, a touchdown is 1 point. From the 1, A scores half the time and B a fifth of the time.
From the 2 both gain a yard half the time; otherwise the other team takes over at its 1.
One play left, tied, at the 1: A wins 0.5 * 1 + 0.5 * 0.5 = 0.75, B wins
0.2 * 1 + 0.8 * 0.5 = 0.60. Two plays left, tied, at the 2:

    A: gain (1/2) -> A at the 1, one play: 0.75
       lose (1/2) -> B at its 1, one play: B 0.60, so A 0.40
       VA = 0.5 * 0.75 + 0.5 * 0.40 = 0.575
    B: 0.5 * 0.60 + 0.5 * (1 - 0.75) = 0.425

With identical teams (both scoring half the time from the 1) both numbers would be 0.5:
the same spot, the same chances. Here A is the favourite from the spot where B is the
underdog, and each number needed the other team's own values (0.60 inside A's, 0.75 inside
B's). No relabelling of one array produces both.

## 3. The two-array Bellman equation

Two arrays, both from the side of the team with the ball:

    VA(t, f, s, e)   A's win probability when A has the ball
    VB(t, f, s, e)   B's win probability when B has the ball

A's backup, with PA the kernels of A's offense against B's defense:

    VA(t, f, s, e) = max over actions a of
                     sum over outcomes o of PA(o | f, a) * WA(o, t - duration(o))

    keep the ball in f'             VA(t', f', s, e)
    B takes over at its y           1 - VB(t', first and 10 at y, -s, e swapped)
    A touchdown                     better of kick and two-point try, each a mix of
                                    1 - KB(t', -(s + points), e swapped)
    A field goal                    1 - KB(t', -(s + 3), e swapped)
    safety against A                1 - FB(t', -(s - 2), e swapped)
    B returns it for a touchdown    1 - TDB(t', -s, e swapped)
    clock at 0                      1, 0.5 or 0 by s

KB and FB are B's values of receiving a kickoff or free kick (VB averaged over the kick
distribution); TDB is B's value just after B scored a touchdown (B choosing its own try).
B's equation is the same with A and B exchanged. The halftime kickoff uses the receiving
team's array. Everything that does not change possession (the defense's timeout, which is a
min inside A's value; the offense's own timeout; protect and chase; the kneel rule) uses A's
own array, exactly as before.

Each action still uses at least one tick, so VA at t needs VA and VB only at earlier times.
Backward induction still works: at each t solve A's slice and B's slice (in either order),
then move to t + 1. No iteration, no convergence question.

How it is coded without forking the rules. Every place the old solver used `opp` on its own
array now asks `self.other`: `_aug` (turnovers and return touchdowns), `_derive` (the
kickoff and free-kick values after a score), and the halftime kickoff. By default a
solution's `other` is itself, which is the identical-team game; `mdp_team.link(a, b)` points
two solutions at each other. `mdp_game.backward_induction` loops over the clock once and
solves every linked solution at each t. With `other = self` the arithmetic is the old code
operation for operation: the refactored solver reproduces the saved solution bit for bit
(maximum difference 0.0 on both toy games and on the real end phase).

## 4. Team-specific kernels

### The family

The protect and chase modes of M4 already bend the generic play with two numbers. On downs 1
to 3 every kept-ball outcome that gained g yards is reweighted by exp(a * g) (a touchdown
counts as a gain of the whole yard line), the turnover probability is multiplied by b
(capped at 0.5), the safety is left alone, and the kept outcomes are rescaled so the row
sums to 1. On fourth down, going for it is a separate model, so team strength enters it as a
shift d of the conversion log-odds:

    p' = 1 / (1 + exp(-(log(p / (1 - p)) + d)))

converted outcomes are scaled by p'/p and failed ones by (1 - p')/(1 - p).
`tilt_kernels` applies both to every play table (neutral, protect, chase and their
end-game versions) directly on the sparse matrices, row by row, and scales the
clock-running part of each entry with it. The family composes: tilting the protect table
(itself the generic play tilted by -0.036) by +0.01 gives the generic play tilted by
-0.026, so a strong offense that protects a lead is the protect table shifted toward more
yards.

Worked example 2 (by hand). A row keeps the ball with a gain of 2 yards (probability 0.5)
or 6 yards (0.3), and turns it over 0.2. Tilt a = 0.05, b = 1: weights exp(0.1) = 1.105
and exp(0.3) = 1.350; unscaled kept mass 0.5 * 1.105 + 0.3 * 1.350 = 0.958; it must stay
0.8, so the scale is 0.8 / 0.958 = 0.835. New row: 2 yards 0.461, 6 yards 0.338, turnover
0.2. Same turnover rate, more yards.

Worked example 3 (by hand, conversion). Base conversion 0.603, d = +0.098 (Detroit's
offense against an average defense). log-odds 0.419 + 0.098 = 0.517, p' = 0.626. The
result file has 0.6263.

### Where the numbers come from

Each matchup has three numbers: a (tilt), b (turnover scale), d (conversion shift).

- d is M8's conversion model's change in log-odds for this offense against this defense,
  relative to average units.
- a is calibrated: with d in place, a is the root of

      drive points(tilted kernels, first and 10 at the own 25)
        - drive points(generic kernels, same) = M8's implied change

  found with Brent's method on the bracket the mode fit uses (-0.3 to 0.3). Drive points
  are counted as M8 counts them (touchdown 7, field goal 3, returned for a touchdown -7,
  safety -2) and computed exactly: one drive is an absorbing Markov chain over field
  states, so D = r + P D is a 4,752-row sparse linear system, which is what infinitely many
  simulated drives would average to. Fourth downs follow a fixed rule (field goal where
  it is good 60% of the time, otherwise go on fourth and 1 from the own 40 on, otherwise
  punt), so the number measures the offense, not a re-optimized policy.
- b is held at 1. M8 has no turnover rating, and drive points alone cannot separate "more
  yards" from "fewer turnovers". The function takes a turnover scale for a sensitivity.
- Field goals, punts, kickoffs and the try stay league average. M8 built no kicking ratings
  (M8 notes, design decision 6), and there is no other defensible rating to use.

Which M8 models. M8's primary models answer "what do ratings add beyond the pregame
spread". The exact game has no spread, so what it needs is the whole effect of the ratings
relative to an average matchup. The team-strength bundle holds exactly that: M8's
ratings-without-the-spread models (`conv_no_spread`, `drive_no_spread`), fitted on the
development seasons 2012 to 2023. M8 used them as a diagnostic; here they are the right
object. One consequence: `matchup_kernels(base, offense, defense)` depends only on that
offense and that defense, never on the other half of the game, so A's kernels do not change
when B's offense changes.

### The calibration, with numbers

Generic kernels: 1.816 points per drive from the own 25 (M8's base curve says 1.93).

| Offense vs defense (end of 2025) | M8 drive-points change | tilt a | conversion d | P(convert), 4th and 2, own 40 |
|---|---|---|---|---|
| Detroit vs average | +0.408 | +0.0084 | +0.098 | 0.603 to 0.626 |
| Detroit vs top-5 defense | -0.066 | -0.0014 | -0.082 | 0.603 to 0.583 |
| Detroit vs bottom-5 defense | +0.839 | +0.0161 | +0.243 | 0.603 to 0.660 |
| Bottom-5 offense vs top-5 defense | -0.851 | -0.0231 | -0.358 | 0.603 to 0.515 |
| Average vs Detroit's defense | +0.096 | +0.0021 | +0.028 | 0.603 to 0.610 |

The own 25 is matched exactly by construction (to 1e-10). Other starting points show the
limits of one parameter. For Detroit against an average defense the model's change against
M8's: own 10, +0.41 against +0.35; midfield, +0.34 against +0.44; opponent's 30, +0.20
against +0.36. A yards tilt helps most where there are many yards to gain, while M8's
drive model says strong offenses also gain most of their edge in the red zone. The tilt
therefore slightly overstates the effect of team strength on long drives and understates
it near the goal line.

## 5. The tests

CPU, seconds, no data (`tests/test_mdp_team.py`):

1. `tilt_kernels` equals `mdp.adjusted_play` row by row, keeps rows summing to 1, moves
   conversion by exactly d in log-odds, leaves downs 1 to 3 alone under a pure conversion
   shift, and composes (two tilts equal one).
2. Drive points by hand: 3.5 and 1.75 in the two-yard-line game.
3. The calibration hits its target to 1e-8; a weaker offense gets a negative tilt.
4. Identical teams: the two-team solution equals `mdp_game.solve` in both arrays on the toy
   game that exercises every rule (difference 0, policies identical).
5. Swapping the teams swaps the arrays exactly (bitwise).
6. A two-team reference solver, the slow recursive `Brute` of `test_mdp.py` with every
   other-team value read from a second instance, agrees with the fast solver to 1e-10 on
   800 random states per team, and on every state of the hand game to 1e-12.
7. The hand-checkable asymmetric game (worked example 1).
8. Monotonicity: a stronger offense for A (tilt +0.15, conversion +0.3) never lowers A's
   value in any state, and never raises B's.
9. Monte Carlo with the two-team simulator, five states with either team on the ball,
   30,000 games each, within 4.5 standard errors.
10. Models that differ in clock or kicks are refused.

Data (marked `data`, about two minutes):

1. Identical league-average teams, two-array end-phase solve against the saved M4 solution:
   maximum difference 0.0 in both arrays over all 61 slices (tolerance 1e-6).
2. Tilting the generic table by the fitted protect numbers reproduces the protect table to
   1e-9.
3. The calibration on the real kernels for M8's top and bottom groups.
4. Monotonicity on 600,000 sampled end-phase states (top-five offense against an average
   defense): A's value never lower than the identical-team value and B's never higher, to
   1e-5.
5. Monte Carlo on the real two-team game, four states, 6,000 games each.

## 6. Results

Up 4, 4:00 left, fourth and 2 at the own 40, three timeouts each. A is the leading team
with the ball. Detroit means Detroit's offense and defense at the end of 2025; the opponent
has an average offense and the named defense. The map covers distances 1 to 10 and the own
20 to midfield (310 cells).

| Matchup | Go | Punt | Go minus punt | Cells where go wins | Change in the go region |
|---|---|---|---|---|---|
| Identical average teams | 0.778 | 0.714 | +6.5 | 246 | |
| Detroit vs top-5 defense | 0.759 | 0.696 | +6.4 | 233 | 13 cells to punt; up to 1 yard shorter |
| Detroit vs average | 0.800 | 0.719 | +8.1 | 310 | 64 cells to go; 2 to 3 yards longer |
| Detroit vs bottom-5 defense | 0.831 | 0.739 | +9.2 | 310 | 64 cells to go; 2 to 3 yards longer |
| Bottom-5 offense vs top-5 defense | 0.694 | 0.671 | +2.3 | 93 | 153 cells to punt; 4 to 5 yards shorter |

(Win-probability points. For Detroit against average and bottom-5 defenses going wins in
every cell of the map, so the edge is beyond fourth and 10 and "2 to 3 yards" is a lower
bound.) The identical baseline reproduces question 1 (+6.5, go to fourth and 7 or 8).

Reading.

- Detroit's 2025 offense (+0.40 points per drive) against a top-five defense (-0.48) nets
  out close to an average matchup, and the answer is question 1's: go on fourth and 2 by
  6.4 points, go out to fourth and 7 or 8.
- Against an average or a bad defense Detroit should go for it at every distance on the map;
  the headline margin rises by 1.6 and 2.7 points. Both the go and the punt get better (a
  better offense also wins more after a punt), but going gains more: conversion rises and
  every later possession is worth more.
- A bottom-five offense against a top-five defense should still go on fourth and 2 (by 2.3
  points) and on fourth and 3, but punt from fourth and 4 on. The go region shrinks from 246
  to 93 cells. This is the largest move, because the weak offense loses on both channels:
  conversion falls to 0.515 and keeping the ball is worth less.
- Compared with M8's one-step result (team strength moved the first-quarter call by under
  1.5 points), the late-game effects here are larger (up to 4.1 points at the headline),
  as M8 expected when possessions are scarce: M8 did not adjust anything with 6:00 or less
  left in a half, and this is the first estimate that does.

## 7. Cost and what a GPU port needs

One whole-game two-team solve (Detroit against an average team, 407.9 million states per
team, 3 threads): 81.4 seconds and 4.23 GB peak (1.52 GB of values and 0.38 GB of policy per
team, float32 and uint8). The identical-team solve takes 45 to 53 seconds and 2.8 GB, so two
teams cost about 1.6 times the time (the kernels and kick values are shared work) and 1.5
times the memory. An end-phase-only matchup solve takes about 56 seconds; it is all the
question needs, and its values equal the whole-game solve's (difference 0.0 at 4:00).

Where the time goes, per team and per tick (end phase): three sparse products K @ W, with K
the play table (4,752 by 4,854, 550,305 non-zeros) and W the outcome values (4,854 by 688,
the 43 scores times 16 timeout pairs), one more for the punt; the clock-running correction
K_in @ (U - V); building U, a weighted sum over the tempo histogram of up to a dozen earlier
slices, then elementwise max and min for the offense's and defense's timeouts; and the max
over actions with the argmax for the policy. Everything is dense elementwise work on
(4,752, 43, 4, 4) arrays plus SpMM with a fixed sparsity pattern. A JAX port keeps the
kernels as BCOO (or dense blocks by down, since each row has about 115 entries), stacks A and
B on a leading axis of 2 so both teams' products run in one batched call, and keeps a ring
buffer of the last max-play-ticks slices on device, so only the policy needs copying back.

## 8. Limits

- No bootstrap: nothing here has an uncertainty interval yet. M8's rating SDs are large
  (Detroit's offense +0.40, SD 0.31 points per drive), so the Detroit rows should be read as
  "an offense of that rating", not as a statement about Detroit.
- Team strength moves plays only: a one-parameter yards tilt plus a conversion shift.
  Turnover rates, kicking, clock behaviour and the protect and chase fits stay league
  average. The tilt matches drive points from the own 25 exactly but not the shape by field
  position (section 4).
- The end-of-2025 ratings are the frozen output of M8's test stage (the filter run through
  the 2024 and 2025 seasons with constants tuned on development seasons). Using them as
  inputs describes a hypothetical game; it scores nothing, but they are test-season
  quantities and are labelled as such.
- Everything the M4 notes list for the identical-team model still applies, including the
  extrapolated conversion probabilities in the own half.

## 9. Things to be able to say without notes

1. Why one array is enough for identical teams and why it is not for two different ones.
2. The two-array Bellman equation, and which terms read the other team's array.
3. Why backward induction still needs only one pass.
4. The tilt family, why it composes, and why the turnover scale is held at 1.
5. How the tilt is calibrated (exact drive points, Brent's method, the own 25) and where
   one parameter fits badly.
6. Why the no-spread M8 models are the right ones here and not M8's primary models.
7. The identical-team equality test and why it is the most important one.
8. Which matchup moves the go region most, and why.
