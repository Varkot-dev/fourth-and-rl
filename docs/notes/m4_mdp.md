# M4 notes: the game as a Markov decision process, solved exactly

Code: `src/fourth_and_rl/mdp.py` (grid, kernels, count tables), `mdp_drive.py` (stage 1),
`mdp_game.py` (stage 2, simulator, loader). Config: `configs/mdp.yaml`.
Scripts: `scripts/solve_mdp.py`, `scripts/validate_simulator.py`,
`scripts/questions/q1_late_lead_dp.py`. Tests: `tests/test_mdp.py`.
Every number below comes from `results/m4_mdp_summary.json`,
`results/m4_simulator_validation.json` or `results/questions/q1_late_lead_dp.json`.

## 1. Concepts

### What an MDP is

A Markov decision process has four parts.

- **State**: everything you need to know to decide. Here: time left, score differential,
  down, yards to go, yard line, timeouts.
- **Action**: what the decision-maker can choose. Here: on fourth down go, field goal or
  punt; on every down how to play (protect, chase, neutral), kneel, or kick.
- **Transition**: P(next state | state, action). Here: the M1 component models (yards
  gained, conversion, field goal, punt, kickoff) plus clock rules.
- **Reward**: what you are trying to collect. Here: 1 for a win, 0.5 for a tie after
  regulation, 0 for a loss, paid once at the end. (Stage 1 uses points instead.)

A **policy** maps each state to an action. The **value** V(s) of a state is the expected
total reward from s when you follow the best policy.

### The Markov property, and why clock and score must be in the state

"Markov" means the future depends on the past only through the current state. If the
state were only (down, distance, yard line), the best action could not be Markov: fourth
and 2 at your own 40 is a punt when you lead by 4 with 0:40 left and a go when you trail by
4 with 0:40 left. Two situations with the same "state" would need different actions, which
means the state is missing something. Put clock and score in the state and the problem is
Markov again: given (time, score, down, distance, yard line, timeouts), nothing about how
the game got there changes what can happen next in our model. (In real football momentum,
injuries and who is hot would matter; the model says they do not.)

### Bellman optimality equation for this problem

Let V(t, f, s, e) be the win probability of the team with the ball, with t ticks left
(one tick = 5 seconds), field state f = (down, distance, yard line), score differential s
from its side, and extra state e (timeouts in the last 5:00). Then

    V(t, f, s, e) = max over actions a of
                    sum over outcomes o of  P(o | f, a) * value(o, at time t - duration(o))

with the value of each kind of outcome:

    keep the ball in field state f'        V(t', f', s, e)
    other team takes over at their y       1 - V(t', first and 10 at y, -s, e swapped)
    touchdown                              the better of kick and two-point try, each a
                                           mix of  1 - K(t', -(s + points), e swapped)
    field goal made                        1 - K(t', -(s + 3), e swapped)
    safety against us                      1 - F(t', -(s - 2), e swapped)
    clock at 0                             1 if s > 0, 0.5 if s = 0, 0 if s < 0

K is the value of receiving a kickoff (V averaged over the kickoff distribution), F the
same for a free kick. "max over actions" is what makes it an optimality equation: the value
of a state assumes the best action now and the best actions ever after.

### Zero-sum and the "1 minus V" trick

One team's win is the other's loss, and a tie is worth 0.5 to both. With identical teams,
the value of a state to the team WITHOUT the ball is 1 minus the value to the team with it,
after re-describing the state from the other side (negate the score, swap the timeouts). So
one array holds both teams' values, and a change of possession is "1 - V(flipped state)".
In `mdp_game.py` this is the function `Solution.opp`.

The defense also makes decisions in our model (when to call a timeout). Because the game is
zero-sum, the defense choosing what is worst for the offense is a **min** inside the
offense's max. Alternating max and min over a finite game tree is exactly minimax, and
backward induction solves it.

### Value iteration versus backward induction

- **Stage 1 (no clock)** has no last state. Possessions go on forever, so V appears on both
  sides of the Bellman equation. You solve it by **value iteration**: start with V = 0,
  apply the right-hand side to get a new V, repeat until it stops changing (ours stops
  after 272 sweeps at a change below 1e-10).
- **Stage 2 (with a clock)** has an end. Every action takes at least one tick, so V at
  t ticks depends only on V at fewer ticks. Fill in t = 0 (game over), then t = 1 from
  t = 0, then t = 2, and so on. Each slice is computed once. This is **backward
  induction**: one pass, no convergence question, exact for the model.

A detail in stage 1: because possession changes flip the sign of the value, plain value
iteration can oscillate forever (the one-state unit test jumps between 0 and 3.5). The fix
is damping: V <- 0.5 V + 0.5 T(V). Same fixed point, no oscillation.

### Why one-step lookahead differs from the DP

The one-step engine (M2) computes, for each action,
sum over outcomes of P(outcome) * WP_model(next state), and trusts a fitted win-probability
model for everything after the next snap. The DP replaces WP_model by the model's own
solved V. They differ for three reasons:

1. **Who plays the rest of the game.** A win-probability model fitted on real games
   contains real coaches, who punt more than the optimum. The DP assumes both teams play
   optimally from here. For the headline case below this accounts for 2.5 of the 5.5
   points of difference.
2. **Resolution.** The DP knows the kneel-down threshold to the tick and what three
   timeouts can and cannot stop. A statistical model sees a few hundred similar games.
3. **Model error on either side.** The DP is exact only for its own simplified football.
   A fitted model is noisy but is at least fitted to real outcomes.

## 2. The model, decision by decision

### State space (what was solved)

| Part | Choice |
|---|---|
| Time | 5-second ticks for the whole game: 720 slices |
| Field | 4 downs x 12 distance bins (1 to 10 exact, 11 to 15, 16+) x 99 yard lines = 4,752 |
| Score | differential clipped to -21..+21 (43 values) |
| Last 5:00 | timeouts of both teams (4 x 4) |
| Rest of second half | no extra state |
| First half | one flag: does the team with the ball receive the second-half kickoff |

Total 407,854,656 states. The whole game solves in 43 seconds on 2 threads, peak memory
2.9 GB for the script, 1.9 GB of saved arrays. Nothing had to be reduced. The research
note suggested a 30-second grid before the last 5:00; I used 5 seconds everywhere because
on a 30-second grid every play would cost at least 30 seconds (real plays average 27), so
games would have about 20% too few plays.

Both teams' timeouts are tracked (not only the defense's) because possession changes: the
leader's timeouts become the defense's timeouts on the trailing team's drive, and the
trailing team's own timeouts are how it stops the clock in a hurry-up drive.

### The trick that makes it fast

The outcome of a play depends only on the field state, not on clock or score. So "what a
play does" is one sparse matrix K with 4,752 rows (field states) and 4,854 columns (next
field state, or "opponent's ball at y", touchdown, safety, return touchdown), with 550,305
non-zeros. For one time slice the values of all outcomes form a matrix W (4,854 rows,
688 columns for score x timeouts). The value of running a play in every state of the
slice is then one product K @ W. A slice of 3.3 million states costs three such products.

### Clock (follows docs/research/endgame_clock.md)

- Stopped-clock result (incomplete, out of bounds inside 5:00, score, change of
  possession): 1 tick. Punt: 2 ticks. Kickoff: 0.
- In-bounds play in the last 5:00: the offense chose a mode with the play.
  **Protect** lets the clock run: 1 + 8 ticks = 45 seconds (measured median 44).
  **Chase** hurries: the measured histogram (median 37 seconds between 5:00 and 2:00,
  21.5 seconds inside 2:00), or spends its own timeout for 1 tick. **Neutral** is normal
  tempo (median 38 seconds). After any in-bounds play the defense may spend a timeout to
  make it 1 tick, and does when that hurts the offense (the min).
- Two-minute warning: a runoff that would cross 2:00 stops at 2:00.
- Kneel rule (closed form): leading, on downs 1 to 3, with at most T seconds left, the
  game is over. The data say 99.92% of such teams won.
- Before the last 5:00: no timeouts, no modes. An in-bounds play lasts a number of ticks
  drawn from the measured histogram. On entering the last 5:00, timeouts are drawn from
  the measured shares (55% of trailing teams hold three).
- Tie after regulation: 0.5.

### Protect and chase tables (a change from the brief, and why)

The brief said downs 1 to 3 use the generic M1 play. I built that first and validated it.
It failed exactly where question 1 lives: simulated teams down 4 to 8 with 1:00 to 5:00
left scored a touchdown on 21% of drives against 31.2% in the data, and leads of 1 to 3
with 2:00 left held 78% of the time against 66%.

The cause is measurable. On downs 1 to 3 in the last 5:00 of one-score games:

| | First-down rate | Turnover rate |
|---|---|---|
| Generic model, same situations | 29.8% (leading) / 28.7% (trailing) | 2.0% / 1.9% |
| Leading teams, real | 22.3% (n = 4,539) | 1.0% |
| Trailing teams, real | 31.8% (n = 7,488) | 2.8% |

Leading teams run into loaded fronts; trailing teams throw against soft coverage. The
research note recommended "separate protect and chase transition tables" for this reason.
A full table per context would be too thin, so each mode is the generic play bent by two
fitted numbers: a **tilt** a (yards gained reweighted by exp(a * yards), found by
bisection so the first-down rate matches) and a **turnover scale**. Fitted: protect
a = -0.036, turnovers x 0.51; chase a = +0.013, turnovers x 1.44.

Rules: protect is allowed unless trailing, chase unless leading, neutral (the generic
play) always. A leader can therefore still throw on third and 8 by choosing neutral, at
the price of more clock stops. A run-or-pass choice later replaces these three modes by
two; the kernel dictionary already takes any number of tables.

Honest caveats: (1) the yardage a team gets is partly the defense's reaction, so letting
the optimizer pick the table is only an approximation of a real choice; (2) the chase
sample leaves out the last 30 seconds (desperation heaves); (3) because a tied team may
chase and a team up 1 may not, 0.0001% of states are worth up to 0.0008 more when tied
than when ahead by a point. The test allows for it.

### Other simplifications and their likely effect

| Simplification | Likely effect |
|---|---|
| Identical league-average teams | Final margins too narrow: SD 12.1 against 14.1 real |
| No penalties, injuries, replay, onside kicks | Small for question 1; drives slightly short |
| Kickoff and free-kick return touchdowns dropped (0.24% of kickoffs) | Negligible |
| Distance bins beyond 10 (13 and 20 stand for 11 to 15 and 16+) | Small, long-yardage only |
| First half has no timeouts, modes or two-minute warning | Slightly too few points before halftime |
| Kneel action costs 45 seconds, real 42; closed-form rule is exact | Tiny |
| Fourth-down conversion model is the same for everyone | See sensitivity: the largest lever |
| Field goal allowed on any down; two-point try chosen optimally | Matters only at the end of halves |
| Score clipped at 21 | A 21-point lead is treated as any bigger one |

## 3. Results

### Stage 1: expected points

- Our "next score" value of a first and 10 against nflfastR's `ep`: correlation 0.9993,
  RMSE 0.09 points (own 25: 1.19 against 1.18; midfield 2.71 against 2.75). Our transition
  model alone reproduces the standard expected-points curve.
- Counting everything after the score too (the endless game), a first and 10 at the own 25
  is worth 0.93 points and receiving a kickoff 1.21.
- Fourth-down chart: go is best in 631 of 1,188 cells. If every later fourth down is played
  as coaches play, go is best in 589 cells; 44 cells change, none by more than 0.10 points.
  Assuming league-average future play makes the chart slightly more conservative, and that
  is all it does at the expected-points level. The chart is aggressive (go on fourth and 4
  or less nearly everywhere); the same known caveats as Romer apply, and conversion
  probabilities deep in your own half are extrapolated from third downs.

### Correctness checks

| Check | Result |
|---|---|
| Hand-solved game (two yard lines, two plays) | exact |
| Random walk football, arXiv 2406.16171 (L = 4, T = 56; and L = 6, T = 30) | max error below 1e-12 |
| Independent recursive solver on a toy game with every rule, 1,500 random states, optimal and empirical defense | max error below 1e-10 |
| Invariants: range, monotone in score, terminal values, kneel region, timeouts help their owner | pass |
| Monte Carlo on the real solution: 60 random states x 4,000 games | 52 uncertain states: rms z 1.03, max 2.67, 92% below 2 |

### Simulator against real football (coach-like policy, 40,000 games)

Matches: points per drive 2.014 (real 2.013), plays per drive 5.81 (5.80), touchdown share
22.5% (22.1%), punts 35.4% (35.6%), turnovers 9.9% (10.2%), regulation ties 5.4% (6.2%).

Mismatches:

- Points per game 46.5 against 44.9, drives per game 22.1 against 21.6.
- Margin SD 12.1 against 14.1; games decided by 17 or more 17.6% against 26.3%. Identical
  teams cannot produce blowouts at the real rate.
- Points per drive from good field position are 0.2 to 0.3 too high (opp 20 to goal: 4.97
  against 4.71), slightly low from the own 1 to 20 (1.40 against 1.46).
- "On downs" 6.5% against 5.8%.
- Leads: mean absolute gap 1.2 points over 24 cells. Worst: lead of 1 to 3 with 2:00 left
  holds 72.2% simulated, 66.3% real (n = 245, standard error 3.0).
- Late trailing drives: touchdown rate 28.8% against 31.2%; after the touchdown the leader
  has 30 seconds or more 80% of the time against 65% (the coach-like trailing team scores
  too quickly or the leader is left too much time); leader then wins 43.8% against 42.6%.

Reading: trust the simulator for drive-level behavior and for close games; do not use it
for anything that depends on team strength or on the size of blowouts.

### Burning question 1

Headline case: up 4, 4:00 left, fourth and 2 at the own 40, opponent has three timeouts.

    go   0.780     punt 0.715     go minus punt +6.5 points

The map (up 4, opponent with three timeouts): at 4:00 going beats punting up to
fourth and 7 to 9 depending on field position; at 3:00 up to 6 or 7; at 2:00 only fourth and 2 or shorter, and with no
opponent timeouts at 2:00 only fourth and 1 (by 1 point). Beyond fourth and 5 the margin
is under 3 points and should be read as a toss-up. A smaller lead means go more (up 1 at
4:00: go at every distance to 10), a bigger one less (up 7: to about 6).

Sensitivity (headline case, go minus punt, points): base +6.5; conversion probability
minus or plus 0.05: +4.3 / +8.9; defense using timeouts at measured rates: +6.8; trailing
offense worse or better: +2.9 to +11.0; generic plays only: +4.7; both teams playing like
real coaches afterwards: +4.0. The sign never changes for fourth and 2. The share of the
whole map where go wins runs from 39% to 97%, so the edge of the go region is not known.
The break-even conversion probability is 38%; the model says 60%.

Against other methods: the one-step engine gives +1.0 for the headline case and says go in
24% of the map's cells against the DP's 68%. nfl4th (its numbers) says go in 24% of the
160 shared cells against the DP's 76%.

Unexpected: more opponent timeouts make going MORE attractive (go wins in 61% of cells with
none, 74% with three). Timeouts help the trailing team's drive after a punt more than they
hurt the leader's attempt to burn clock after a conversion.

## 4. Worked examples

### Kneel-down arithmetic

A kneel takes 2 seconds and the play clock allows 40 more unless the defense calls
timeout. With m kneels available and the defense holding n timeouts:

    T(n, m) = 40 * max(m - n, 0) + 2 * m     (capped at 120 by the two-minute warning)

First down (m = 3), defense has one timeout: 40 * 2 + 2 * 3 = 86 seconds. Check by playing
it: kneel (2 s), they call timeout; kneel (2 s), 40 runs; kneel (2 s), 40 runs.
2 + 2 + 40 + 2 + 40 = 86. On the 5-second grid that is 85. With two timeouts
40 + 6 = 46, with three 6, with none 126 capped at 120.

### One Bellman backup by hand (the test `test_hand_solved_game`)

Tiny game: yard lines 1 and 2, one down. From the 1 you score (1 point) half the time,
otherwise they take over at their 2. Tied, one play left, at the 1:

    V = 0.5 * (lead by 1, game over: 1) + 0.5 * (still tied, game over: 0.5) = 0.75

Now two plays left, tied, at the 1. Score (0.5): they get the ball at their 2, down 1,
with one play; from the 2 they cannot score, so their value is 0 and ours is 1 - 0 = 1.
Fail (0.5): they have the ball at their 2, tied, one play: their value is 0.5, ours
1 - 0.5 = 0.5.

    V = 0.5 * 1 + 0.5 * 0.5 = 0.75

The "1 -" each time the ball changes hands is the zero-sum trick.

### The headline decision taken apart

    P(convert) = 0.605
    win probability if converted = 0.895
    win probability if failed    = 0.604
    go   = 0.605 * 0.895 + 0.395 * 0.604 = 0.780
    punt = 0.715

Punt, from 200,000 simulated games: the opponent starts at its own 18 with 3:50 and scores
a touchdown on 35.4% of those drives (real, own 1 to 20 with 3:00 to 4:00: 36.0%, n = 86).
If it does not, the leader wins 95.6%. If it does, the leader gets the ball back 97.7% of
the time, with a median of 65 seconds, and wins 27.1%.

    punt = 0.646 * 0.956 + 0.354 * 0.271 = 0.714

Failing: the opponent starts at the leader's 40 and scores a touchdown 54.8% (real,
drives starting in the leader's half: 60.9%, n = 69); the leader then has a median of 115
seconds and wins 36.1%. Converting: the leader keeps the ball to the end only 12.7% of the
time against three timeouts, but the opponent's next drive starts at its own 24 with a
median of 130 seconds and scores a touchdown 18.3%.

Break-even: p * 0.895 + (1 - p) * 0.604 = 0.715 gives p = 0.381.

So the logic is: a punt does not buy safety (the opponent still scores a third of the
time and then usually wins), a failure costs 11 points of win probability relative to the
punt, and a conversion gains 18.

## 5. Things to be able to say without notes

1. State, action, transition, reward for this problem, and why (down, distance, yard
   line) alone is not Markov.
2. The Bellman equation above, and where the max, the min and the "1 minus" come from.
3. Why stage 1 needs iteration and stage 2 does not.
4. Why the DP says go more often than the one-step engine, with the three reasons.
5. Why protect and chase tables were added, and what would change with a run or pass
   choice.
6. What the simulator gets right and wrong, and what that means for the RL milestone.
7. Why the answer to question 1 is a range, and which assumption moves it most.
8. The kneel arithmetic for any number of timeouts.

## 6. For the next milestone

    from fourth_and_rl import mdp_game as mg
    sol = mg.load_solution()                       # memory-mapped arrays under data/mdp
    s = mg.GameState(seconds=240, score=4, down=4, ydstogo=2, yardline_100=60,
                     off_timeouts=3, def_timeouts=3)
    sol.value(s); sol.policy(s); sol.q_values(s)   # also module-level mg.value, mg.policy
    nxt, info = mg.step(s, "play_protect", rng)    # one play, same transition model
    sim = mg.Simulator(sol); mg.simulate(sim, sim.start(1000, t=48, s=4, f=...), rng)

`step` returns the next state from the side of whoever then has the ball, with
`possession_changed`, `points`, `done` and `result`. Decisions that are not the offense's
play call (defensive timeouts, the try after a touchdown) are made as the solver assumed.
Action names: play, play_protect, play_chase, field_goal, punt, kneel.
