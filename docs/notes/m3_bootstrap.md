# M3 notes: uncertainty by game-level bootstrap

Code: `src/fourth_and_rl/bootstrap.py`. Config: `configs/bootstrap.yaml`.
Scripts: `scripts/bootstrap_decisions.py`, `scripts/questions/q1_late_lead_one_step.py`.
Tests: `tests/test_bootstrap.py`.
Every number below is from `results/m3_bootstrap.json` or
`results/questions/q1_late_lead_one_step.json`, written by those scripts. Numbers quoted
from the humility paper (Brill, Yurko and Wyner, arXiv 2311.03490) are the authors' and
are labelled as such. Worked examples marked "by hand" are arithmetic.
Figures: `assets/hero_decision_chart.png`, `assets/m3_decision_chart_confidence.png`,
`assets/questions/q1_late_lead_one_step.png`.

## 1. Concepts

### The problem

M2 says "punt" on fourth and 4 from your own 20 in a tied first quarter, by 0.06 points of
win probability (go 0.4270, punt 0.4276). Nobody should act on 0.06 points. (Before the
final rerun, with win-probability version 1, the same cell said "go" by 0.3 points: the
call flipped with the model, which is the point of this milestone.) The question is: how
much of a recommendation would survive if we had been handed a slightly different
history of NFL games to learn from? If a different, equally plausible set of seasons would have
flipped the call, the call is noise.

### What the bootstrap is

We have one data set. We cannot replay 2010 to 2023 to see how much our fitted models
would change. The bootstrap fakes the replay: draw a new data set of the same size from
the one we have, WITH replacement, and refit everything on it. Some games appear twice,
some not at all (on average a draw contains about 63% of the distinct games, since the
chance a given game is never drawn in n draws is (1 - 1/n)^n, about 0.37). Do this B
times. The spread of the B refitted answers is an estimate of how much the answer
depends on which games happened to be in the data.

Worked example (by hand). Four games A, B, C, D. One draw of four with replacement might
be A, A, C, D. B is left out and A counts twice. A model fitted on that draw leans more
on game A and knows nothing of B. Another draw, B, C, C, D, gives a different model.

### Why games are the unit, not plays

Every play of a game shares one win or loss label. A game with 150 plays is not 150
independent observations of "who wins from here"; it is one outcome seen 150 times. If
we resampled plays, each resample would still contain a piece of nearly every game, the
refits would barely differ, and the uncertainty would look far smaller than it is. The
real sample size for win probability is the number of games: 3,793 development games
here, not 637,000 plays. Resampling whole games keeps each game's plays together so the
draw-to-draw variation is the honest one.

In code this needs one piece of care: a game drawn twice must count as two games, so
each draw gets its own game id, and the "what is the next state" lookup
(`components.add_next_state`) is done once on the real games before any resampling.

### boot%, and the three categories

For a state x, let d(x) be the best action from the models fitted on the real data (the
point estimate), and d_b(x) the best action from replicate b.

    boot%(x) = (number of replicates b with d_b(x) = d(x)) / B

Rule (from the humility paper, Section 4.2): confident if boot% >= 83%, lean if 67% <=
boot% < 83%, uncertain ("too close to call") below 67%. With B = 101: 84 or more
agreeing replicates is confident (84/101 = 0.832), 68 to 83 is lean, 67 or fewer is
uncertain.

The gain interval: for each replicate compute gain_b = go_wp - max(fg_wp, punt_wp), sort
the 101 values, and report the 6th and the 96th. That is a 90% interval, and B is odd so
those are whole positions in the sorted list.

Worked example (reference chart, `chart.reference.cells_table`). Fourth and 1 at
midfield: gain +5.5 points, interval +2.6 to +7.5, boot% 100%: confident go. Fourth and
goal from the 2: gain +1.6 points, interval -0.03 to +3.6, boot% 94%: confident go, even
though the interval for the gain just touches zero (the category counts how often the
best action agrees, not whether the gain is clearly positive). Fourth and 4 at the own
20: punt by 0.06 points, interval for going -0.7 to +2.6, boot% 22%: too close to call
(only 22 of 101 replicates agree with the punt; the rest prefer going).

### What boot% means and does not mean

It means: "the share of plausible alternative data sets on which this pipeline would
make the same call". It does NOT mean "the probability that this is the right call".
Three reasons.

1. It measures variance, not bias. Every replicate uses the same model family, the same
   features, the same clock assumptions. If all of them are wrong in the same direction
   they agree with each other and boot% is 100%. M2 found such biases: version 1 of our
   WP model was too cautious with small late leads, and version 2 still over-rates a team
   trailing by 4 to 8 in the last 40 seconds. The bootstrap cannot see them.
2. It is not a posterior probability. No prior was written down.
3. The plain game bootstrap is known to be too narrow. In the authors' simulation study
   (arXiv 2406.16171, Table 5, their numbers) a plain game-level bootstrap interval with
   nominal 90% coverage covered 71% of the time. Our own simulation in M3b found 77% for a
   LightGBM model (`results/m3b_sample_size_sim.json`). So our "confident" is, if
   anything, over-confident.

### The humility paper's point

Fourth-down bots report "go: +2.1%" as if the 2.1 were known. It is an estimate from a
few thousand games. When the uncertainty is carried through, a large share of
recommendations cannot be told from a coin flip, so criticism of a coach is only fair
where the recommendation is robust. Their numbers: confident on 48% of 2018 to 2022
fourth downs, uncertain on 27%.

## 2. Design decisions and why

1. **What is resampled.** The games of 2010 to 2023 only, drawn in two groups of seasons:
   3,224 games of 2010 to 2021 and 569 of 2022 to 2023, each group with replacement to
   its own size. The groups are those of the version 1 fit (training and calibration);
   with version 2 the trees grow on both groups together. 2024 and 2025 are never drawn,
   so the 2024 to 2025 decisions are judged by models that never saw them.
2. **Both models are refitted in every replicate.** WP trees on all drawn development
   games with the same hyperparameters as the headline model (300 trees); every component
   model on all drawn games. Refitting only the WP model would understate uncertainty in
   the conversion and kicking probabilities.
3. **No calibration map (version 2).** The headline model has no calibration map, so
   neither do the refits (`calibration: none` in `configs/bootstrap.yaml`). Version 1 had
   a Platt map fitted on two seasons, and the bootstrap refitted it on the drawn
   calibration games in every replicate, because M1 showed that a two-season Platt slope
   is noisy and that noise was a real part of the pipeline's uncertainty (in the version 1
   run the slope averaged 0.861 with standard deviation 0.066 and ranged from 0.72 to
   1.04). The code path is kept for version 1; the version 2 run records "no calibration
   map" in `replicate_diagnostics`.
4. **The point estimate is the M2 engine.** boot% compares each replicate with the
   headline engine, which was fitted on the same seasons by the same procedure.
5. **Games only, not games then drives.** The humility paper resamples games and then
   drives within each drawn game. We use whole games as the default, because it is the
   textbook cluster bootstrap and the one the plan names; the drives option is
   implemented (`within_game: drives`) and unit-tested but has not been run on the full
   data, so we cannot say how much it would change the categories. `game_fraction`
   (draw fewer games, the "fractional bootstrap") is also implemented and not run.
6. **The kickoff-rule tables are fixed.** The tables of the 2024 and 2025 kickoff rules
   are fitted on those seasons' own kickoffs and are not resampled (see M2).
7. **A different LightGBM seed per replicate** (base seed plus the replicate number), so
   the random row and column subsampling of the trees is part of the variation too.
8. **Evaluation states are fixed**: the four chart panels (5,520 cells) and all 8,569
   real fourth-down rows of 2024 to 2025, the same for every replicate.
9. **Saved and resumable.** Each replicate's WP model, component bundle and drawn game
   list are saved under `data/models/boot/seed_20261117_none/` (128 MB, not committed),
   and each replicate's action values are cached, so a rerun takes seconds and later
   milestones can reuse the refits.

Runtime: 12.7 seconds to refit and 7.2 seconds to evaluate 14,089 states per replicate,
34 minutes in total on 2 threads.

How much the refitted pieces move (`replicate_diagnostics`): P(convert) on fourth and 2
at midfield 0.602 with standard deviation 0.008; a 50-yard field goal 0.738, sd 0.010;
the expected spot after a punt from the own 30, sd 0.2 yards. The component models are
stable. Most of the variation in decisions comes from the WP model.

## 3. Results

### Chart cells (`assets/m3_decision_chart_confidence.png`)

| Situation | Confident | Lean | Too close to call |
|---|---|---|---|
| First quarter, tied (the hero chart) | 70.1% | 14.9% | 14.9% |
| Trailing by 7, 8:00 left | 81.4% | 9.9% | 8.8% |
| Leading by 4, 4:00 left | 77.5% | 11.5% | 11.0% |
| Trailing by 3, 2:00 left | 79.2% | 9.5% | 11.3% |

(With version 1 the reference chart was 63.7% confident.)

On the reference chart the confident regions are the ones a fan would guess: go on
fourth and 1 to 3 almost everywhere, punt on long yardage in your own half, kick on long
yardage in range. The grey band is the boundary between them: fourth and 5 to 7 in your
own half, fourth and 5 to 7 inside the opponent's 20, and the long-yardage zone around
the opponent's 40 where punt, long field goal and go are all close. The 90% interval of
the gain from going is typically 3.4 points wide and excludes zero in 51% of cells.

The late-game panels look MORE confident than the first-quarter one. Do not read that as
"we know more late in games". It is the bias point from section 1: all 101 refits share
the same model family and the same end-game errors, so they agree with each other
whether or not they are right.

### Real fourth downs, 2024 to 2025

7,955 plays with a real decision (575 penalty or kneel rows and 43 overtime or
incomplete rows are in the saved file but not counted here).

- Confident **72.3%**, lean **13.5%**, too close to call **14.2%**. (With version 1:
  62.0%, 17.7% and 20.3%.)
- By recommended action: go 72% confident and 13% uncertain; field goal 75% and 14%;
  punt 71% and 17%.
- The estimated gain from going is under 1 point on 44% of plays, under 2 points on 64%
  and under 4 points on 86%. (The paper reports 44% under 1 and 87% under 4 for their
  model on 2018 to 2022; their numbers.)
- The 90% interval for the gain is 2.8 points wide at the median and excludes zero on
  58% of plays.
- For orientation only (the coach evaluation is M7): on confident plays coaches went
  49.2% of the time when the engine says go (n = 2,689) and kicked 95.3% of the time
  when it says kick (n = 3,060). The paper reports 49% and 91% for its own model and
  seasons. On too-close-to-call plays where the engine leans go, coaches went 12%.

We are confident more often than the paper (72% against their 48%). We cannot say why
from here. Candidates: they also resample drives within games, their model and seasons
differ, and the plain game bootstrap is too narrow for tree models (77% coverage for a
nominal 90% in M3b), which makes refits agree more than they should. Version 1 had
explained the gap partly by its very large leaves; version 2 has 2,000-play leaves and is
confident even more often, so that explanation does not hold.

### Stability of the categories

We did not run a second seed. Instead the 101 replicates were split into even-numbered
(51) and odd-numbered (50) and each half classified every state on its own. The two
halves give the same category on **88.7%** of real plays (83% on the reference chart).
Almost all changes are between neighbouring categories: only 13 of 7,955 plays move
between confident and uncertain. Each half has half the replicates, so this understates
the stability of the full B = 101 result. The paper reports about 0.9 agreement across
repeated full bootstraps (their number, a different check).

### Saved for the coach evaluation

`data/processed/fourth_down_decisions.parquet`: 8,569 rows, one per fourth-down row of
2024 to 2025, with the state, the three action values, best action, margin, gain,
boot%, category, the share of replicates preferring each action, the gain interval and
median, and `actual_decision` (null for penalty no-plays and kneels).

## 4. Burning question 1, the one-step answer

Up 4, 4:00 left, fourth down in your own half, three timeouts each, even matchup
(`results/questions/q1_late_lead_one_step.json`, figure
`assets/questions/q1_late_lead_one_step.png`).

**The one-step answer: go on fourth and 1 to 3 everywhere in your own half; fourth and 4
is mostly too close to call; punt from fourth and 8 on.** Going is the best action in 47%
of the 500 own-half cells (50 field positions by 10 distances). 65% of the cells are
confident, 19% lean and 16% too close to call.

| Yards to go | What the engine says across the own half |
|---|---|
| 1 | Go everywhere, confident on all 50 yard lines |
| 2 | Go everywhere: confident on 47 of 50 yard lines, lean on 3 |
| 3 | Go everywhere: confident on 16, lean on 26, too close to call on 8 |
| 4 | Go on 38 of 50 yard lines, but too close to call on 34 (confident go on 10) |
| 5 and 6 | Go on 21 and 17 of 50 yard lines; most cells lean or too close to call |
| 7 | Go on 10 of 50 yard lines; punt confident on 34 |
| 8 or more | Punt, confident nearly everywhere (45 of 50 at fourth and 8, all 50 at 9 and 10) |

The earlier version of this answer (win-probability version 1) was "punt, unless it is
fourth and 1": go in 18% of the cells. It changed with the win-probability model. The
most likely reason is the one version 2 was built to fix: version 1 was too cautious
about small late leads (M2 notes, point 4), so it under-valued the state a conversion
produces (a lead with the ball and the clock running), and going looked worse than it is.
The rerun did not isolate this cause; it is an inference.

Selected cells (gain from going in WP points, 90% interval, boot%; cells without a label
are confident):

| Spot | 4th and 1 | 4th and 2 | 4th and 3 | 4th and 5 |
|---|---|---|---|---|
| Own 20 | go +5.6 (+2.1, +8.6) 100% | go +2.5 (-0.6, +5.0) 87% | go +0.7 (-2.1, +3.3) 67% lean | punt -1.7 (-4.8, +1.2) 81% lean |
| Own 30 | go +5.6 (+2.6, +8.3) 100% | go +3.1 (+0.1, +5.8) 96% | go +1.6 (-1.3, +4.1) 81% lean | punt -0.5 (-3.1, +1.9) 67% lean |
| Own 40 | go +7.2 (+3.2, +9.0) 100% | go +4.8 (+0.8, +6.4) 99% | go +3.3 (-0.3, +5.2) 91% | go +1.3 (-2.1, +3.3) 66% uncertain |
| Midfield | go +8.8 (+5.0, +11.2) 100% | go +6.1 (+2.7, +8.5) 100% | go +4.6 (+1.5, +7.2) 100% | go +2.7 (-0.3, +5.2) 88% |

Other leads and clocks. With 6:00 left the engine looks like 4:00 (47% of cells up 4,
35% up 7). With 2:00 left it goes much less (18% of cells up 4). Up 3 behaves like up 4
(53% at 4:00); up 1 goes more (69% at 4:00, 68% at 6:00, 34% at 2:00). Across the whole
question 1 grid of the exact solution (M4), the one-step engine says go in 43% of the
cells and the exact solution in 67% (`results/questions/q1_late_lead_dp.json`,
`one_step_comparison`).

**nfl4th, same situations.** The reference grid has nfl4th go values only at the own 20
and own 30 (it returned NA at the own 40 and midfield). There, nfl4th says go on 16% of
the 240 scenarios and we say go on 59%; the two tools make the same call on 43%, and on
48% of the 172 where ours is "confident". With three opponent timeouts nfl4th says punt
on every one of them, including fourth and 1. Its go recommendations are all in the
rows where the opponent has no timeouts. Our "confident" label does not protect against
this disagreement, which is the variance-not-bias point again. (With version 1 we said go
on 39% and the two tools agreed on 53%: version 2 moved our engine away from nfl4th.)

**Coaches.** 774 real fourth downs from the development seasons 2010 to 2023, leading by
1 to 8 with 1:30 to 7:00 left in the fourth quarter, in their own half. They went on 16 of
91 at fourth and 1 (10 converted), 2 of 55 at fourth and 2, 1 of 62 at fourth and 3, and 3
of 566 beyond that. From 2021 to 2023: 6 of 14 at fourth and 1, 1 of 11 at fourth and 2.
Coaches punt almost always; the engine now disagrees with them on fourth and 1 to 3
everywhere in the own half. (The earlier version of this paragraph counted 2010 to 2025
and said the engine agreed with coaches except at fourth and 1 and at fourth and 2 near
midfield.)

### What a one-step lookup cannot capture here

Say this clearly whenever the one-step answer is quoted.

1. **Clock burn after a conversion.** The real prize of converting with 4:00 left is that
   the offense can run three more plays and take about two more minutes, or all of it if
   the opponent is out of timeouts. The engine charges 27 seconds for the conversion and
   then asks the WP model, which knows how such leads turned out on average under
   ordinary play calling, not what this team can now do on purpose.
2. **The opponent's timeouts.** They enter only through two WP features that the trees
   barely use. Changing the opponent from three timeouts to none moves the share of "go"
   cells from 47.4% to 46.8%. In a real end-game that number decides whether a first
   down ends the game. nfl4th, in contrast, flips from punt to go when the timeouts go.
3. **Who gets the last possession.** Failing gives a short field, but a quick score also
   gives the ball back with time to answer; punting may mean never touching the ball
   again. A one-step value has no notion of sequence.
4. **A WP model with its own end-game errors.** Version 1 under-valued small late leads;
   version 2 fixed that, but it still over-rates a team trailing by 4 to 8 in the last 40
   seconds (16.6% predicted against 8.1% observed) and under-rates one trailing by 1 to 3
   (29.7% against 40.2%) (M2 notes, point 4). After a punt or a failed attempt the
   opponent is the trailing team, so this contaminates the 2:00 panels most.
5. **Bootstrap confidence is about noise.** All 101 refits share points 1 to 4.

The full-game dynamic program (M4) is the tool that answers the question properly: it
models the clock in seconds, kneel-downs and the opponent's timeouts. Its headline (up 4,
4:00, fourth and 2 at the own 40) is go 0.778 against punt 0.714, and the one-step answer
above has moved most of the way toward it. The one-step answer here is the standard
analytics answer, with honest error bars on the part of the error that is sampling noise.

## 5. Limits, stated plainly

- boot% is not the probability of being right, and the plain game bootstrap is
  probably too narrow.
- No second seed was run; stability is from a split of the replicates.
- The drives-within-games and fractional options are untested on real data.
- 2024 to 2025 were not used to tune anything, but the thresholds (83%, 67%) and B are
  the paper's, not validated by us.
- Everything inherits M2's limits: one step, league-average teams, no overtime, a fixed
  charge of seconds per play, and a WP model that still errs for trailing teams in the
  last 40 seconds.

## 6. Questions you should be able to answer

1. What is a bootstrap replicate, step by step, in this project?
2. Why whole games? What would go wrong, and in which direction, if you resampled plays?
3. What is the effective sample size for the WP model, and what was it for version 1's
   calibration map?
4. A cell has boot% = 71%. Which category is it, and what exactly is 71% the share of?
5. boot% is 100% for "punt when trailing by 1 with 22 seconds left" if every refit says
   so. Why does that not make it right?
6. Version 1 refitted the Platt map in each replicate instead of reusing the headline
   one. Why, and why does version 2 not need to?
7. Why is B odd, and which sorted values bound the 90% interval when B = 101?
8. Why must a game that is drawn twice get two different ids?
9. Why were 2024 and 2025 kept out of the resampling?
10. The paper is confident on 48% of plays and we are on 72%. Give two possible reasons
    and say what you would run to find out.
11. Up 4 with 4:00 left at your own 30, fourth and 1: what does the one-step engine say,
    with what confidence, and name three things it cannot see.
12. Split-half agreement is 89%. Why does that understate the stability of the B = 101
    categories?
13. With version 1 the one-step answer to question 1 was "punt unless fourth and 1"; with
    version 2 it is "go on fourth and 1 to 3". Which error in version 1 most likely caused
    the difference, why does it push toward punting, and what would you run to confirm it?
