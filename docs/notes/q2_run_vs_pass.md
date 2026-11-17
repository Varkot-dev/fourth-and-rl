# Burning question 2: why do offenses run when passing gains more?

Code: `src/fourth_and_rl/run_pass.py`. Config: `configs/run_pass.yaml`.
Script: `scripts/questions/q2_run_vs_pass.py`. Tests: `tests/test_run_pass.py`.
Every number below is from `results/questions/q2_run_vs_pass.json`, written by that
script, unless it is marked "by hand" (arithmetic on a made-up example) or attributed to
someone else. Figures: `assets/questions/q2_run_vs_pass.png` (the main one),
`q2_defense_adjusts.png`, `q2_risk_and_clock.png`.

EPA, win probability (`wp`, `wpa`) and expected pass probability (`xpass`) are nflfastR's
models. They are used as given and never mixed with our own win-probability model.
Intervals in brackets are 95% percentile intervals from 1,000 resamples of whole games.
Data: 546,139 called runs and passes in 4,363 games, 2010 to 2025, regular season and
playoffs.

## 0. The answer in short

1. The gap is real and it is not a game-script illusion. A called pass is worth 0.111 EPA
   per play more than a called run on all plays [0.104, 0.118], 0.118 in neutral game
   script, and more, not less, once the situation is held fixed: 0.166 at the same down,
   distance and field position, 0.193 on first and second down.
2. It is an early-down gap. On third and 1 or 2, fourth and 1 and inside the 3-yard line
   the run is as good or better, and when a team protects a late lead the pass's EPA
   advantage produces no advantage in nflfastR's win probability.
3. Defenses do adjust (lighter boxes against pass-heavy teams and in passing situations),
   but every response we can measure is several times too small to make today's mix
   optimal, and the gap in 2023 to 2025 is the same size as in 2010 to 2012. Our data
   cannot say what happens at pass rates nobody has tried.

A rough upper bound: moving 5 percentage points of neutral early-down calls from run to
pass (about 1.25 plays per team per game) is worth at most 0.21 expected points per team
per game [0.19, 0.23], before any defensive response.

## 1. Concepts

### Expected points added (EPA)

Expected points (EP) is the average net points of the next score, from the offense's
side, given down, distance, field position and a few other things. First and 10 at your
own 25 is worth about +1 point; first and goal at the 1 about +6. A play's EPA is EP
after the play minus EP before it. A 4-yard run on first and 10 gains yards but leaves
second and 6, which is worth slightly less than first and 10 was, so its EPA is slightly
negative. EPA puts yards, first downs, turnovers and field position in one unit (points).
Two things it does not do: it treats a point as a point whatever the score and clock
(it is not win probability), and it is an average, so it says nothing about risk. The
EP model here is nflfastR's, not ours.

### Average value and marginal value

The average value of passes is what the passes that were called gained. The marginal
value is what one more pass would gain. A team deciding whether to pass more needs the
second, and the two differ for two separate reasons.

Reason 1: runs and passes are called in different situations. Example (by hand, and it
is the toy frame in `test_cell_comparison_recovers_a_known_answer`). Two situations:

    situation           runs   run EPA   passes   pass EPA
    first and 10         90     +0.30      10      +0.50
    third and 12         10     -1.00      90      -0.80

In each situation a pass is worth 0.20 more than a run. But the plain averages are
run (90 * 0.30 + 10 * -1.00) / 100 = +0.17 and pass (10 * 0.50 + 90 * -0.80) / 100 =
-0.67. The average says running is better by 0.84 when passing is better everywhere,
because passes are called where every play is bad. This is Simpson's paradox. The cure
is to compare inside situations and then average the differences ("standardizing").
In the real data it goes the other way from what the popular reply claims: holding the
situation fixed makes the pass advantage bigger (section 3).

Reason 2: the payoff of a play type depends on how often it is used, because the defense
sets up for what it expects. Example (by hand). Let p be the pass rate and suppose

    pass EPA  P(p) =  0.30 - 0.40 * p      (passing more makes each pass worse)
    run EPA   R(p) = -0.10 + 0.10 * p      (and each run better)

Points per play are V(p) = p * P(p) + (1 - p) * R(p) = -0.10 + 0.50 * p - 0.50 * p^2,
which is largest at p = 0.50. At that best mix P = +0.10 and R = -0.05. The average pass
beats the average run by 0.15 and the team should still not pass more: one more pass
adds 0.10 itself but lowers the value of the other passes and raises the value of the
runs, and the total change is zero. In general the best mix satisfies

    P - R = -p * P'(p) - (1 - p) * R'(p)

so an average gap is consistent with good play calling only if pass efficiency falls
fast enough as the pass rate rises. This is the honest form of "marginal, not average",
and it gives a number to test: with the measured gap (0.178 on neutral early downs) and
pass rate (0.514), today's mix is optimal only if each extra 10 points of pass rate cost
about 0.04 EPA on every pass (section 4).

### Mixed-strategy equilibrium, worked two-by-two

Offense calls run or pass; defense sets up for the run or for the pass. Payoffs to the
offense in EPA (by hand):

                      defense plays run   defense plays pass
    offense runs           -0.30               +0.20
    offense passes         +0.50               -0.10

No pure strategy is stable: whatever the offense always does, the defense sets up for
it. Let the defense play pass with probability q and the offense pass with probability p.

- The offense is willing to mix only if both calls are worth the same:
  run value = -0.30 * (1 - q) + 0.20 * q = -0.30 + 0.50 * q,
  pass value = 0.50 * (1 - q) - 0.10 * q = 0.50 - 0.60 * q.
  Equal when 1.10 * q = 0.80, q = 0.727. Both are then worth +0.064.
- The defense is willing to mix only if both setups give up the same:
  against run defense the offense gets 0.50 * p - 0.30 * (1 - p) = 0.80 * p - 0.30,
  against pass defense -0.10 * p + 0.20 * (1 - p) = 0.20 - 0.30 * p.
  Equal when 1.10 * p = 0.50, p = 0.455. Value +0.064 again.

Three lessons. (1) The offense passes only 45% of the time although the pass has the
bigger upside (+0.50 against +0.20), because the defense plays pass 73% of the time.
(2) In equilibrium the average run and the average pass are worth the same, +0.064,
within one situation. So in this simple game a measured gap inside a situation is
evidence against equilibrium, or evidence that EPA is not what teams maximize. (3) If
the offense moves to p = 0.60 and the defense does not react, nothing changes (the
offense was indifferent). If the defense reacts by always playing pass, the offense gets
0.20 - 0.30 * 0.60 = +0.02, less than +0.064. Deviating only pays if the defense fails
to adjust. The two-by-two game assumes the defense cannot see the offense's mix before
it commits on each play; the P(p), R(p) model above assumes it reacts to the offense's
known tendency. Real football is between the two, which is why both tests are run.

### Selection and game-script confounding

A team that leads runs to burn clock; a team that trails passes against a defense that
gives up short gains. So run plays are over-represented in comfortable situations and
passes in desperate ones. That is selection on things we observe (score, clock, down,
distance, field position) and the fix is to restrict and to compare within cells. What
remains is selection on things we do not observe: the offense sees the defensive front
and can check to a run against a light box or a pass against a heavy one, a draw on
third and 15 is a surrender, a quarterback sneak is only called when it will work. Within
a cell, each play type is therefore observed mostly where it looked good, and the
comparison is still not an experiment.

### Game-clustered bootstrap

Plays in a game share a quarterback, an opponent and the weather, so they are not
independent. Whole games are resampled with replacement (1,000 times, seed 20261117),
every statistic is recomputed, and the middle 95% of the results is the interval.
Implementation detail worth knowing: drawing games with replacement is the same as
giving each game a multinomial count, so per-game sums are computed once and each
resample is one matrix product. One weight matrix is shared by every statistic.

## 2. The puzzle, measured

Run versus pass, all plays and neutral game script:

| | Run | Pass | Pass minus run |
|---|---|---|---|
| EPA per play, all plays | -0.070 | +0.041 | +0.111 [0.104, 0.118] |
| EPA per play, neutral | -0.069 | +0.048 | +0.118 [0.108, 0.128] |
| EPA per play, not neutral | -0.072 | +0.033 | +0.105 [0.095, 0.114] |
| Neutral: success rate (EPA above 0) | 39.8% | 45.9% | +6.1 points [5.7, 6.4] |
| Neutral: yards per play | 4.28 | 6.49 | +2.20 [2.14, 2.27] |
| Neutral: first-down rate | 23.3% | 33.4% | +10.1 points [9.8, 10.4] |
| Neutral: turnover rate | 0.7% | 2.7% | +2.0 points [1.9, 2.1] |
| Neutral: SD of EPA on one play | 0.98 | 1.62 | +0.64 [0.63, 0.65] |
| Neutral: SD of yards on one play | 6.3 | 10.1 | +3.8 [3.7, 3.9] |

Neutral means: score within 8, nflfastR win probability between 20% and 80%, more than
2:00 left in the half, no overtime. 51% of called plays qualify. The neutral filter moves
the gap by +0.007 [0.000, 0.013]: game script is not the explanation. The parts of the
non-neutral sample pull in opposite directions: trailing by more than 8 the gap is only
0.034 (passes against soft coverage are not worth much), leading by more than 8 it is
0.199.

By down, neutral:

| Down | Pass share | Run EPA | Pass EPA | Gap |
|---|---|---|---|---|
| 1 | 46% | -0.083 | +0.107 | +0.190 [0.177, 0.202] |
| 2 | 58% | -0.073 | +0.090 | +0.163 [0.149, 0.179] |
| 3 | 84% | -0.028 | -0.074 | -0.046 [-0.081, -0.014] |
| 4 | 49% | +0.340 | +0.307 | -0.033 [-0.231, 0.173] |

By down and distance, neutral:

| Situation | Pass share | Gap in EPA | Gap in success rate |
|---|---|---|---|
| 1st and 10 | 46% | +0.205 [0.191, 0.217] | +15.0 points |
| 2nd and 8 or more | 68% | +0.237 [0.217, 0.260] | +14.7 |
| 2nd and 3 to 7 | 52% | +0.158 [0.134, 0.184] | +7.0 |
| 2nd and 1 or 2 | 26% | +0.040 [-0.018, 0.098] | -0.3 |
| 3rd and 8 or more | 95% | +0.215 [0.144, 0.288] | +10.3 |
| 3rd and 3 to 7 | 93% | -0.046 [-0.135, 0.031] | -2.0 |
| 3rd and 1 or 2 | 43% | -0.054 [-0.119, 0.006] | -12.6 |
| 4th and 1 or 2 | 35% | -0.039 [-0.279, 0.212] | -8.6 |

The classic claim holds: the gap is largest on first and 10 and second and long, and it
is gone in short yardage. By field position the gap is 0.10 to 0.19 everywhere outside
the 10 and reverses inside the 5 (run better by 0.131 [0.049, 0.205]).

Third and 8 or more needs care: the 5% of plays that are runs there are mostly draws
that concede the down, so "pass better by 0.215" compares a real attempt with a give-up.

By season (all plays): the gap was 0.116 in 2010, peaked near 0.146 in 2015 and 2016,
and was 0.088 [0.060, 0.118] in 2025. The league pass rate under our definition stayed
between 60% and 62% in every season.

### How the classification changes the answer

This matters more than any filter and should be said whenever the headline is quoted.

| Definition | Gap, all plays | Gap, neutral |
|---|---|---|
| Scrambles are passes (ours; 13,711 scrambles) | +0.111 | +0.118 |
| Scrambles left out | +0.093 | +0.099 |
| Scrambles are runs (as nflverse records them) | +0.061 | +0.068 |
| nflfastR `pass`/`rush` flags, penalty plays included | +0.146 | +0.153 |

A scramble is a pass call that ended with the quarterback running, usually for a good
gain. Recorded as a run it moves about half the gap to the wrong side. Our table drops
plays wiped out by a penalty (`play_type` "no_play"), which removes defensive pass
interference and so understates passing; with nflfastR's own flags, which keep those
plays, the gap is 0.146. Our main numbers are therefore the conservative middle.

## 3. Average versus marginal: inside situation cells

Cells are down (4) by distance bin (1, 2, 3-4, 5-7, 8-10, 11+) by field-position bin
(1-4, 5-10, 11-20, 21-40, 41-60, 61-80, 81-99 yards from the end zone), neutral plays
only. A cell is compared when it has at least 100 runs and 100 passes: 70 of the 168
cells, holding 90% of neutral plays.

| | Share of neutral plays |
|---|---|
| Cells where the pass is better (interval above zero) | 71.8% |
| Cells too close to call | 17.9% |
| Cells where the run is better (interval below zero) | 0.1% |
| Cells with too few plays to compare | 10.2% |

By down: every compared first-down and second-down play is in a "pass better" or "tie"
cell (first down 114,441 pass-better to 9,591 tie; second down 78,455 to 14,258). On
third down most compared plays are in tie cells (24,948), with 8,215 in pass-better
cells (long yardage). The one "run better" cell is third and 2 inside the 5. The too-small
cells are mostly third and medium or long, where teams almost never run.

The gap with the situation held fixed (difference inside each cell, averaged with the
cell's play count as weight):

| Sample | Plain gap, same plays | Situation held fixed |
|---|---|---|
| Neutral, all downs | +0.138 | +0.166 [0.154, 0.178] |
| Neutral, first and second down | +0.180 | +0.193 [0.184, 0.203] |
| Neutral, 2021 to 2025, all downs | +0.148 | +0.157 [0.140, 0.175] |
| Neutral, 2021 to 2025, first and second down | +0.161 | +0.172 [0.155, 0.188] |

Holding the situation fixed raises the gap, as in the toy example: passes are called
disproportionately on third and long, where every play has low EPA.

Largest gaps: second and 8 to 10 (0.27 to 0.33 depending on field position) and third
and 11 or more. Smallest: third and 2 inside the 5 (-0.79 [-1.17, -0.40]), fourth and 1
(-0.30 to -0.40, wide intervals), third and 2 to 4 near midfield (-0.12 to -0.20, wide).

Out-of-sample check, because sorting 70 cells by an estimate invites reading noise as
signal: cells were labelled on the development seasons (2010 to 2023) and the gap was
then measured on the test seasons (2024 and 2025). In the 25 cells labelled "pass better"
the test-season gap is +0.155 [0.128, 0.181]. In the 41 "tie" cells it is -0.029
[-0.100, 0.037]. The labels hold up.

One number for the situation: nflfastR's expected pass probability (`xpass`). The gap is
+0.19 to +0.21 where a pass is expected 40% to 60% of the time, +0.14 at 70% to 85%,
+0.09 at 85% or more, and it is gone (-0.033 [-0.094, 0.031]) only where a pass is
expected less than 30% of the time.

What this does and does not show. It rejects the simple version of the equilibrium
story, in which the average run and the average pass are equally good once you look
inside a situation. It does not by itself show that teams should pass more, because of
the second reason in section 1 (payoffs that depend on the mix) and because of what EPA
leaves out (section 5).

## 4. Do defenses adjust?

### (a) Team tendencies, measured in other games

For each offense and game, the tendency is the team's neutral early-down pass rate in its
other games that season (leave one game out), so a game's own plays never enter its own
predictor. The outcome is the team's run EPA, pass EPA and box count faced in that game.
8,693 team-games. Pass rate in other games has mean 51.3%, SD 5.5 points, and 90% of
team-games lie between 42% and 61%. Slopes are per 10 points of pass rate.

| Outcome | Comparing teams | Teams of equal efficiency in other games | Same team across seasons |
|---|---|---|---|
| Run EPA | +0.010 [0.000, 0.020] | +0.003 [-0.008, 0.014] | +0.004 [-0.007, 0.016] |
| Pass EPA | +0.053 [0.038, 0.068] | +0.039 [0.023, 0.054] | +0.040 [0.022, 0.057] |
| Defenders in the box (2016 on) | -0.20 [-0.22, -0.18] | -0.20 [-0.22, -0.18] | -0.15 [-0.17, -0.13] |

Reading it:

- Defenses clearly respond in how they line up: a team that passes 10 points more faces
  0.15 to 0.20 fewer defenders in the box on neutral early downs.
- Runs gain a little from it, about 0.003 to 0.010 EPA per 10 points, not clearly
  different from zero once efficiency or the team is held fixed.
- Pass EPA rises with pass rate in every version. That is not a causal effect of passing
  more. Teams with good quarterbacks pass more, and the controls (efficiency in other
  games, team fixed effects) only partly remove that, because a team's passing quality
  changes from season to season with its quarterback. This slope is not identified.

How steep would it have to be? From section 1, today's mix is optimal only if
P'(p) = -(gap + (1 - p) * R'(p)) / p. With the neutral early-down gap 0.178, p = 0.514
and the measured run slopes, pass EPA would have to fall by 0.038 to 0.045 per 10 points
of pass rate [about -0.054, -0.027]. The measured, confounded slope is +0.04.

A back-of-envelope through the box channel (by hand, from the numbers above and in (c)):
10 more points of pass rate remove about 0.2 defenders from the box; one defender is
worth about +0.04 EPA to a run and -0.03 to a pass on first and 10; so the box channel
predicts roughly +0.008 for runs and -0.006 for passes per 10 points. That is the right
sign and about one sixth of the size needed. Coverage, pass rush and personnel changes
are not in the box count, so this is a floor on the response, not a measurement of it.

### (b) Within a game

Pass share over the offense's previous 10 called plays in the game, neutral early downs,
with down, distance bin, field bin, season and the team's pass rate in other games held
fixed. Slope per 10 points of recent pass share: run EPA +0.002 [-0.002, 0.006]
(n = 72,593), pass EPA -0.003 [-0.010, 0.003] (n = 79,494). The signs match "defenses
adjust" and both are indistinguishable from zero. The gap is 0.157 after a run-heavy
stretch (40% or fewer passes), 0.181 in the middle and 0.170 after a pass-heavy one.

The call itself is predictable. On second and 10 after a first and 10 that gained
nothing, offenses pass 79.6% of the time when the first-down play was a run for no gain
and 50.5% when it was an incomplete pass, a difference of 29.1 points [27.6, 30.7] at the
same down and distance. Offenses alternate, which a truly random mixer would not do
(Kovash and Levitt reported the same pattern; that is their finding, quoted from
`docs/research/public_debate_and_teams.md`, and this is ours). And the predictable pass
pays less: pass EPA on that second and 10 is -0.013 after a run and +0.059 after an
incompletion, a difference of 0.071 [0.012, 0.135]. This is the clearest sign in our
data that predictability is punished. It is one situation, and the two groups differ in
ways we do not control (a team that just threw incomplete may be a better passing team).

### (c) Defenders in the box

From the nflverse participation files, 2016 to 2025, 99.9% of neutral plays have a
count. The counts come from a different charting source from 2023 on, and the mean on
neutral early downs is 6.55 to 6.74 in 2016 to 2022 and 6.26 to 6.43 in 2023 to
2025; part of that may be the change of source, so no trend is claimed.

The box follows the situation: 6.59 on first and 10, 6.18 on second and 8 or more, 5.45
on third and 8 or more, 6.91 on third and 1 or 2, 7.31 on fourth and 1 or 2. It falls
steadily from 7.24 where nflfastR expects a pass less than 30% of the time to 5.68 where
it expects one 85% or more.

First and 10, neutral:

| Defenders in the box | Pass share | Run EPA | Pass EPA | Gap |
|---|---|---|---|---|
| 5 or fewer | 78% | -0.047 | +0.059 | +0.106 [0.041, 0.170] |
| 6 | 50% | -0.065 | +0.104 | +0.169 [0.145, 0.193] |
| 7 | 42% | -0.104 | +0.136 | +0.240 [0.214, 0.266] |
| 8 or more | 35% | -0.152 | +0.174 | +0.326 [0.273, 0.381] |

Payoffs do depend on what the defense does: runs lose about 0.10 EPA from a light box to
a loaded one and passes gain about as much. Two things stand out. Offenses still run 65%
of the time into 8 or more defenders, where the gap is largest. And even against 5 or
fewer in the box the pass is ahead. The box is partly a response to the offense's
personnel and formation (a heavy formation draws a heavy box and also signals run), so
this is not the effect of the defense's choice alone.

### What this evidence identifies

It identifies that defenses line up differently against pass-heavy teams and in passing
situations, that play-type payoffs move with the front, and that offenses are
predictable in at least one clean situation and pay for it. It does not identify the
effect on efficiency of a team choosing to pass more: every comparison across teams,
seasons or game stretches mixes that effect with who the team is. And all of it lies
inside the range teams actually use (42% to 61% on neutral early downs). Nothing here
can say what would happen at 70% or 80%.

## 5. Risk and the objective

### Outcome distributions

First and 10, neutral (60,639 runs, 52,173 passes):

| | Run | Pass |
|---|---|---|
| No gain or a loss | 17.3% | 37.6% |
| Gains 4 or more | 46.7% | 54.9% |
| Gains 10 or more | 11.2% | 29.8% |
| Gains 20 or more | 2.6% | 10.2% |
| Turnover | 0.6% | 2.4% |
| EPA below -1 | 4.3% | 6.8% |
| EPA above +1 | 5.2% | 16.8% |
| EPA, 5th / 50th / 95th percentile | -0.96 / -0.22 / +1.02 | -1.29 / +0.01 / +1.96 |
| SD of EPA | 0.80 | 1.31 |

28% of the passes are incomplete and 4.6% are sacks. A pass is the bigger gamble: twice
as likely to gain nothing, four times as likely to turn the ball over, four times as
likely to gain 20. But note the median: the typical pass (+0.01) is also better than the
typical run (-0.22). The pass is not a lottery ticket with a bad median and a fat tail.
Risk aversion alone has to be strong to prefer the run on first and 10.

### Protecting a late lead

Leading by 1 to 8 in the last 5:00, downs 1 to 3 (6,161 runs, 2,666 passes), compared at
the same down and distance bin:

| Pass minus run | |
|---|---|
| EPA per play | +0.281 [0.208, 0.350] |
| nflfastR win probability added | -0.19 points [-0.44, +0.05] |
| Game clock to the next snap | -4.1 seconds [-5.2, -3.1] |
| Defense spends a timeout after the play | -23 points [-25, -21] (44% after runs, 20% after passes) |
| Clock plus 40 seconds per timeout spent | -13.4 seconds [-14.5, -12.4] |
| Turnover rate | +1.8 points [1.1, 2.7] |

The EPA gap is as large here as anywhere and it buys nothing in win probability. The
reason is the clock, but it does not show up as raw seconds, because the trailing defense
answers a run with a timeout. When the defense has no timeouts left the clean numbers
appear: 35 seconds to the next snap after a run, 22 after a pass (a difference of 13
[11, 15]; 13% of runs and 47% of passes leave the clock stopped). That matches
`docs/research/endgame_clock.md`: an in-bounds run by a leader takes 43 seconds snap to
snap and an incompletion 5. Charging 40 seconds for each timeout the defense burns gives
the "clock plus timeouts" row. That index is an argument, not a model result.

The same thing on first and 10 as the clock becomes the objective (win probability added
per play, nflfastR):

| First and 10 | Pass minus run, EPA | Pass minus run, win probability |
|---|---|---|
| Neutral game script | +0.205 | +0.66 points [0.62, 0.70] |
| Leading by 1 to 8, fourth quarter | +0.306 | +0.25 points [0.09, 0.41] |
| Leading by 1 to 8, last 5:00 | +0.334 | -0.03 points [-0.52, +0.50] |

Limits: `wpa` is nflfastR's model and inherits its treatment of the clock and timeouts;
the last row has 301 passes; and the choice was not random (teams that pass on first
down while leading late may be the ones with a reason to).

How much of all running is this? Plays by a team leading in the fourth quarter are 10.0%
of called plays and 15.5% of all runs (run share 60.8% there); leading in the last 5:00,
6.3% of all runs (run share 76.7%); leading in the last 2:00, 1.7% (run share 87.8%). The
check against the endgame note reproduces exactly: downs 1 to 3, last 2:00, leading by 1
to 8, run share 87.4% as recorded (86.5% with scrambles as passes), 2,274 plays. Removing
every play by a team leading in the fourth quarter lowers the all-plays gap from 0.111 to
0.097 (change -0.015 [-0.017, -0.012]). Clock play is real and correct, and it explains
about an eighth of the gap.

### Short yardage

Neutral game script:

| Situation | Pass share | Run converts | Pass converts | EPA gap (pass minus run) |
|---|---|---|---|---|
| 3rd and 1 | 25% | 71.8% | 58.5% | -0.096 [-0.201, 0.008] |
| 3rd and 2 | 68% | 60.9% | 55.5% | -0.121 [-0.223, -0.023] |
| 4th and 1 | 24% | 70.2% | 60.5% | -0.366 [-0.685, -0.023] |
| 4th and 2 | 78% | 54.1% | 60.3% | +0.580 [-0.071, 1.282] |
| Inside the 3, downs 1 to 3 | 32% | touchdown 52.4% | touchdown 48.6% | -0.133 [-0.228, -0.032] |

With 1 yard to go the run converts 10 to 13 points more often and is at least as good in
EPA. This is where the equilibrium story fits: teams run about 75% of the time, the
defense knows it, and the pass still is not better. Runs here include quarterback sneaks.

## 6. Points from the public argument, with our numbers

The arguments are described in general terms; the source notes are in
`research-kit/x/2_twitter_findings.md`, section 7.

1. "The average pass is about 0.11 EPA better than the average run, and most teams show
   it." Our all-seasons figure is 0.111 [0.104, 0.118]. For 2025, the most recent
   complete season, it is 0.088 [0.060, 0.118] on all plays and 0.084 [0.047, 0.126] in
   neutral script. In 2025, 25 of 32 teams had higher pass EPA than run EPA, 4 of them by
   0.2 or more; across seasons the count ranges from 21 (2022) to 30 (2014, 2015). A
   single team's season gap is noisy (the SD across teams is about 0.11), so "this team
   should pass more" needs more than its own split. The 2025 neutral gap by situation:
   first and 10 +0.186 [0.135, 0.234], second and 8 or more +0.167 [0.086, 0.247], second
   and 3 to 7 +0.084 [-0.006, 0.183], and nothing distinguishable from zero in short
   yardage or third and medium. The public figure's exact definitions are not known to
   us; ours counts scrambles as passes and drops penalty no-plays.
2. "Averages at the current mix say nothing about a different mix." Correct, and it is
   the main limit of this analysis (section 4, "What this evidence identifies"). What the
   within-team and sequential evidence adds: the response we can see is small inside the
   observed range, and far below what would make the current mix optimal. What it cannot
   add: anything about pass rates outside 42% to 61%.
3. "The run sets up the pass." Three tests, neutral early downs. (i) Pass EPA after a run
   versus after a pass, same drive, same down, distance bin, field bin and season: +0.008
   [-0.010, 0.028]. No effect. (ii) Teams that run more in their other games do not pass
   better: -0.053 per 10 points of run rate [-0.068, -0.038], and -0.039 [-0.054, -0.023]
   with efficiency held fixed (the same confounded slope as in section 4, seen from the
   other side). (iii) Teams that run efficiently in other games do pass better: +0.051
   EPA per pass for each 0.1 of run EPA [0.039, 0.061], +0.035 [0.023, 0.046] holding
   pass EPA in other games fixed. So running volume shows no benefit to the pass, while
   running quality goes with passing quality. The second fact fits "a run game defenses
   must respect helps the pass" and fits equally well "good lines and good quarterbacks
   are good at both". We cannot separate them. Play-action is not in our columns, so the
   specific play-action claim is not tested.
4. "Teams care about winning, not EPA." True where the clock is the objective, and
   quantified in section 5: late-lead running is 6% to 16% of all runs depending on the
   window, it is justified in win-probability terms, and removing it shrinks the gap from
   0.111 to 0.097. It does not touch the neutral early-down gap, where win probability
   and EPA agree (+0.66 points of nflfastR win probability per first-and-10 pass over a
   run).

## 7. The verdict

How much of the raw gap survives (EPA per play, pass minus run):

| Step | Gap |
|---|---|
| 1. All plays | +0.111 [0.104, 0.118] |
| 2. Neutral game script | +0.118 [0.108, 0.128] |
| 3. Neutral, same down, distance and field position | +0.166 [0.154, 0.178] |
| 4. Neutral, first and second down, same cell | +0.193 [0.184, 0.203] |

All of it survives, and the adjustments raise it. Situation and game script are not the
explanation. What does change the picture: the gap belongs to first and second down (it
is zero or negative on third and short, fourth and 1 and at the goal line), it produces
no win-probability gain when protecting a late lead, and its size depends on counting
scrambles as passes (0.061 if they are counted as runs).

The trend, neutral early downs at the same down and distance: 0.176 [0.153, 0.199] in
2010 to 2012 and 0.173 [0.152, 0.194] in 2023 to 2025, a change of -0.003 [-0.033,
0.026]. In between it rose to about 0.24 in 2015 to 2017 and came back down. The linear
trend is -0.014 per decade [-0.035, 0.006]. The neutral early-down pass rate rose from
49.5% to about 52% (+1.5 points per decade [1.0, 2.0]). Teams have moved very little on
these downs, and the gap has not closed. The decline from the 2015 to 2017 peak fits
"defenses adjusted to the passing boom", but with one peak in 16 seasons that is a
description, not a test.

Is the remaining gap evidence that teams run too much on neutral early downs? It is
evidence in that direction and not proof. For: the gap is large, stable for 16 seasons,
present in win probability as well as EPA on these downs, largest exactly where offenses
run into loaded boxes, and the measurable defensive response is too small to offset it.
Against: the response to a large change in mix is not observed; EPA ignores what runs
may buy that we cannot see (pass-rusher fatigue, quarterback hits, the credibility of
play-action); and within a cell the plays are not randomly assigned.

A rough estimate of what is at stake, 2021 to 2025. An average team has 25.0 neutral
early-down calls per game, 12.0 of them runs. Shifting 5 percentage points of those
calls from run to pass is 1.25 plays per game. Valuing each at the cell-matched gap,
weighted by where runs are called (0.168 [0.151, 0.184]), gives

    0.05 * 25.0 * 0.168 = 0.21 expected points per team per game   [0.19, 0.23]

This is an upper bound in nflfastR EPA units: it assumes the shifted runs would do as
well as the average pass in the same cell and that defenses do not respond. The interval
is sampling noise only and says nothing about those assumptions. For scale, Kovash and
Levitt's estimate (theirs, as reported in `docs/research/public_debate_and_teams.md`,
from a search summary) was about one point per game for a larger change.

## 8. The hook for the decision model

`RunPassOutcomes` fits the same count-based tables as `components.PlayModel`
(Kaplan-Meier yards gained by down, distance bin and field bin, small cells borrowing
from wider groups, separate turnover share and takeover spot), once on called runs and
once on called passes. `outcomes(call, down, ydstogo, yardline_100)` returns the keys and
array shapes of `ComponentModels.play_outcomes` (`p_touchdown`, `p_safety`, `keep`,
`turnover`, `p_first_down`, `p_turnover`), which sum to 1. `pass_share` gives the
observed mix in the cell and `expected_yards` a summary. `components.py` is not modified;
its private helpers are imported.

The saved bundle (`data/models/run_pass_outcomes.joblib`) is fitted on development
seasons only, 2018 to 2023 (78,532 runs, 124,203 passes), and checked on 2024 and 2025:

| Held-out seasons | Run: predicted / actual | Pass: predicted / actual |
|---|---|---|
| First down or touchdown | 24.3% / 24.1% | 32.7% / 32.3% |
| Turnover | 0.62% / 0.67% | 2.82% / 2.58% |
| Yards without a turnover | 4.32 / 4.35 | 6.50 / 6.43 |

Example, first and 10 at the own 25: run first down 11.7%, turnover 0.6%, 4.8 yards;
pass first down 29.7%, turnover 2.7%, 7.8 yards. Third and 1 at midfield: run converts
73.3%, pass 63.3%.

A warning for whoever wires this into the dynamic program. These tables describe what
happened at the mix teams used. A dynamic program given a free run-or-pass choice on top
of fixed tables will pass on almost every early down, because nothing in the tables
makes passing worse as it becomes predictable. That would be the model's assumption
talking, not a finding. Either constrain the mix or add a response.

## 9. Design decisions someone could challenge

1. Scrambles are passes. Defensible (the call was a pass) and standard in public work, but
   it is half the raw gap. Both alternatives are reported.
2. Penalty no-plays are dropped, because our state table does not classify them. This
   leaves out defensive pass interference and understates passing. Shown as a sensitivity.
3. The neutral filter uses nflfastR's win probability (20% to 80%), a score margin of 8
   and the 2:00 mark. Other reasonable thresholds were not tried; the result that the
   filter barely matters makes this low stakes.
4. Cell edges and the 100-play minimum were fixed before looking at results and not
   tuned. Ten percent of neutral plays fall in cells too small to compare.
5. "Pass better" is defined by a 95% interval per cell with no multiple-comparison
   correction. The out-of-sample check is the guard.
6. Standardizing weights by all plays in the cell (the average situation). Weighting by
   runs (the situations where runs are called) gives 0.167 instead of 0.166.
7. Descriptive statistics use all seasons, including 2024 and 2025. Only the two things
   evaluated out of sample (cell labels, the outcome tables) are fitted on development
   seasons, with the split guards in `splits.py` and logged test access.
8. Team tendency is leave-one-game-out within a season, not prior games only. It never
   uses the game being evaluated, but it does use later games, so it is a description of
   the team that season and not something a defense could have known that week.
9. In the regressions the leave-one-out predictors are computed once and not recomputed
   inside each bootstrap resample, so those intervals are a little too narrow.
10. Games are resampled as independent. A team's games within a season are not, so
    intervals on team-level slopes are somewhat optimistic.
11. The points-per-game figure uses a 5-point shift because the brief asked for "a few
    percent". It scales linearly by construction, which is exactly the assumption that
    fails for large shifts.
12. The "clock plus 40 seconds per timeout" index is ours and rough.

## 10. Limits and what is not done

- No causal estimate of the defensive response. Nothing in public play-by-play can give
  one; it would need a team that changed its mix for reasons unrelated to its talent.
- No play-action, personnel or formation columns, so "run sets up play-action" and
  "runs from passing formations" are untested. `shotgun` and `no_huddle` are cached but
  unused.
- The run-or-pass action is not in the dynamic program yet. Section 8 is the hook only,
  so the plan's "test with the run-or-pass version of the decision model" is still open.
- Win-probability results use nflfastR's `wpa`, not our model.
- EPA counts a point as a point. Fatigue, injuries and hits on the quarterback are not
  in any number here.
- Fourth-down rows are a selected sample (teams that chose to go).
