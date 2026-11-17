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

M2 says "go" on fourth and 4 from your own 20 in a tied first quarter, by 0.3 points of
win probability. Nobody should act on 0.3 points. The question is: how much of a
recommendation would survive if we had been handed a slightly different history of NFL
games to learn from? If a different, equally plausible set of seasons would have
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
real sample size for win probability is the number of games: 3,224 training games and
569 calibration games here, not 623,000 plays. Resampling whole games keeps each game's
plays together so the draw-to-draw variation is the honest one.

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
midfield: gain +6.5 points, interval +3.1 to +8.5, boot% 100%: confident go. Fourth and
4 at the own 20: gain +0.3 points, interval -1.5 to +3.1, boot% 71%: lean. Fourth and
goal from the 2: field goal by 0.04 points, interval for going -2.3 to +3.8, boot% 35%:
too close to call (more replicates preferred going than kicking).

### What boot% means and does not mean

It means: "the share of plausible alternative data sets on which this pipeline would
make the same call". It does NOT mean "the probability that this is the right call".
Three reasons.

1. It measures variance, not bias. Every replicate uses the same model family, the same
   features, the same clock assumptions. If all of them are wrong in the same direction
   they agree with each other and boot% is 100%. M2 found exactly such a bias: our WP
   model is too cautious in the last two minutes. The bootstrap cannot see it.
2. It is not a posterior probability. No prior was written down.
3. The plain game bootstrap is known to be too narrow. In the authors' simulation study
   (arXiv 2406.16171, Table 5, their numbers) a plain game-level bootstrap interval with
   nominal 90% coverage covered 71% of the time. So our "confident" is, if anything,
   over-confident.

### The humility paper's point

Fourth-down bots report "go: +2.1%" as if the 2.1 were known. It is an estimate from a
few thousand games. When the uncertainty is carried through, a large share of
recommendations cannot be told from a coin flip, so criticism of a coach is only fair
where the recommendation is robust. Their numbers: confident on 48% of 2018 to 2022
fourth downs, uncertain on 27%.

## 2. Design decisions and why

1. **What is resampled.** The games of 2010 to 2023 only, in the same two groups as the
   headline WP fit: 3,224 training games (2010 to 2021) and 569 calibration games (2022
   to 2023), each group drawn with replacement to its own size. 2024 and 2025 are never
   drawn, so the 2024 to 2025 decisions are judged by models that never saw them.
2. **Both models are refitted in every replicate.** WP trees on the drawn training games
   with the same hyperparameters as the headline model (300 trees); every component model
   on all drawn games. Refitting only the WP model would understate uncertainty in the
   conversion and kicking probabilities.
3. **Calibration is refitted too.** The Platt map is refitted on the drawn calibration
   games. M1 showed that a two-season Platt slope is noisy. That noise is a real part of
   our pipeline's uncertainty, so it belongs in the bootstrap and is not frozen. Across
   the 101 replicates the slope averaged 0.861 with standard deviation 0.066 and ranged
   from 0.72 to 1.04 (`replicate_diagnostics`). The alternatives were to freeze the
   headline map (hides the noise) or drop calibration in refits (then the replicates
   would differ systematically from the point estimate).
4. **The point estimate is the M2 engine.** boot% compares each replicate with the
   headline engine, which was fitted on the same seasons by the same procedure.
5. **Games only, not games then drives.** The humility paper resamples games and then
   drives within each drawn game. We use whole games as the default, because it is the
   textbook cluster bootstrap and the one the plan names; the drives option is
   implemented (`within_game: drives`) and unit-tested but has not been run on the full
   data, so we cannot say how much it would change the categories. `game_fraction`
   (draw fewer games, the "fractional bootstrap") is also implemented and not run.
6. **The kickoff table is fixed.** It comes from 2025 and is not resampled (see M2).
7. **A different LightGBM seed per replicate** (base seed plus the replicate number), so
   the random row and column subsampling of the trees is part of the variation too.
8. **Evaluation states are fixed**: the four chart panels (5,520 cells) and all 8,569
   real fourth-down rows of 2024 to 2025, the same for every replicate.
9. **Saved and resumable.** Each replicate's WP model, component bundle and drawn game
   list are saved under `data/models/boot/seed_20261117_none/` (128 MB, not committed),
   and each replicate's action values are cached, so a rerun takes seconds and later
   milestones can reuse the refits.

Runtime: 12.0 seconds to refit and 9.1 seconds to evaluate 14,089 states per replicate,
35 minutes in total on 2 threads.

How much the refitted pieces move (`replicate_diagnostics`): P(convert) on fourth and 2
at midfield 0.602 with standard deviation 0.008; a 50-yard field goal 0.738, sd 0.010;
the expected spot after a punt from the own 30, sd 0.2 yards. The component models are
stable. Most of the variation in decisions comes from the WP model.

## 3. Results

### Chart cells (`assets/m3_decision_chart_confidence.png`)

| Situation | Confident | Lean | Too close to call |
|---|---|---|---|
| First quarter, tied (the hero chart) | 63.7% | 18.5% | 17.8% |
| Trailing by 7, 8:00 left | 73.3% | 10.3% | 16.4% |
| Leading by 4, 4:00 left | 73.6% | 13.6% | 12.8% |
| Trailing by 3, 2:00 left | 78.2% | 11.9% | 9.9% |

On the reference chart the confident regions are the ones a fan would guess: go on
fourth and 1 to 3 almost everywhere, punt on long yardage in your own half, kick on long
yardage in range. The grey band is the boundary between them: fourth and 5 to 7 in your
own half, fourth and 4 to 7 inside the opponent's 20, and the long-yardage zone around
the opponent's 40 where punt, long field goal and go are all close. The 90% interval of
the gain from going is typically 4.5 points wide and excludes zero in 42% of cells.

The late-game panels look MORE confident than the first-quarter one. Do not read that as
"we know more late in games". It is the bias point from section 1: all 101 refits share
the same flat late-game win probability.

### Real fourth downs, 2024 to 2025

7,955 plays with a real decision (575 penalty or kneel rows and 43 overtime or
incomplete rows are in the saved file but not counted here).

- Confident **62.0%**, lean **17.7%**, too close to call **20.3%**.
- By recommended action: go 59% confident and 21% uncertain; field goal 53% and 26%;
  punt 70% and 17%.
- The estimated gain from going is under 1 point on 41% of plays, under 2 points on 63%
  and under 4 points on 85%. (The paper reports 44% under 1 and 87% under 4 for their
  model on 2018 to 2022; their numbers.)
- The 90% interval for the gain is 4.1 points wide at the median and excludes zero on
  44% of plays.
- For orientation only (the coach evaluation is M7): on confident plays coaches went
  47.8% of the time when the engine says go (n = 2,159) and kicked 90.4% of the time
  when it says kick (n = 2,770). The paper reports 49% and 91% for its own model and
  seasons. On too-close-to-call plays where the engine leans go, coaches went 14%.

We are confident more often than the paper (62% against their 48%). We cannot say why
from here. Candidates: they also resample drives within games, their model and seasons
differ, and our WP model is smoother (very large leaves), which makes refits agree.

### Stability of the categories

We did not run a second seed. Instead the 101 replicates were split into even-numbered
(51) and odd-numbered (50) and each half classified every state on its own. The two
halves give the same category on **85.2%** of real plays (87% on the reference chart).
Almost all changes are between neighbouring categories: only 22 of 7,955 plays move
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

**The one-step answer: punt, unless it is fourth and 1.** Going is the best action in
18% of the 500 own-half cells (50 field positions by 10 distances). 79% of the cells are
confident, 11% lean and 10% too close to call.

| Yards to go | What the engine says across the own half |
|---|---|
| 1 | Go everywhere. Confident on 35 of 50 yard lines; too close to call on 10, around the own 15 to 30 |
| 2 | Go on 22 of 50 yard lines (8 confident): from about the own 40 out, and a suspect patch inside the own 5 where the punt model is weak. Too close to call on 23 |
| 3 | Punt on most (39 of 50); go confident on only 2 |
| 5 or more | Punt, confident nearly everywhere |

Selected cells (gain from going in WP points, 90% interval, boot%):

| Spot | 4th and 1 | 4th and 2 | 4th and 3 | 4th and 5 |
|---|---|---|---|---|
| Own 20 | go +0.9 (-2.5, +4.7) 66% uncertain | punt -2.2 (-5.3, +1.7) 77% lean | punt -4.0 (-6.5, +0.2) 94% | punt -6.5 (-9.3, -2.2) 100% |
| Own 30 | go +2.2 (-1.3, +4.6) 84% confident | punt -0.4 (-3.7, +2.1) 70% lean | punt -1.8 (-5.2, +0.8) 88% | punt -3.8 (-7.1, -1.2) 100% |
| Own 40 | go +3.2 (-0.1, +5.3) 93% confident | go +1.0 (-2.3, +3.1) 55% uncertain | punt -0.5 (-3.7, +1.4) 72% lean | punt -2.2 (-5.6, -0.3) 96% |
| Midfield | go +5.5 (+2.5, +7.8) 100% | go +3.5 (+0.4, +5.3) 96% | go +2.2 (-0.9, +3.9) 89% | go +0.7 (-2.2, +2.2) 52% uncertain |

Other leads and clocks. With 6:00 left the engine goes far more often (40% of cells up
4, 66% up 7) and is less sure. With 2:00 left it looks like 4:00 (21% of cells up 4).
Up 3 behaves like up 4; up 1 goes more (30 to 42% of cells).

**nfl4th, same situations.** The reference grid has nfl4th go values only at the own 20
and own 30 (it returned NA at the own 40 and midfield). There, nfl4th says go on 16% of
the 240 scenarios and we say go on 39%; the two tools make the same call on 53%, and on
64% of the 161 where ours is "confident". With three opponent timeouts nfl4th says punt
on every one of them, including fourth and 1. Its go recommendations are all in the
rows where the opponent has no timeouts. Our "confident" label does not protect against
this disagreement, which is the variance-not-bias point again.

**Coaches.** 896 real fourth downs from 2010 to 2025, leading by 1 to 8 with 1:30 to
7:00 left in the fourth quarter, in their own half. They went on 20 of 100 at fourth and
1 (12 converted), 4 of 68 at fourth and 2, 1 of 68 at fourth and 3, and 3 of 660 beyond
that. Since 2021: 10 of 23 at fourth and 1, 3 of 24 at fourth and 2. Coaches punt almost
always; the engine agrees with them except at fourth and 1, and at fourth and 2 near
midfield.

### What a one-step lookup cannot capture here

Say this clearly whenever the one-step answer is quoted.

1. **Clock burn after a conversion.** The real prize of converting with 4:00 left is that
   the offense can run three more plays and take about two more minutes, or all of it if
   the opponent is out of timeouts. The engine charges 27 seconds for the conversion and
   then asks the WP model, which knows how such leads turned out on average under
   ordinary play calling, not what this team can now do on purpose.
2. **The opponent's timeouts.** They enter only through two WP features that the trees
   barely use. Changing the opponent from three timeouts to none moves the share of "go"
   cells from 18.4% to 17.6%. In a real end-game that number decides whether a first
   down ends the game. nfl4th, in contrast, flips from punt to go when the timeouts go.
3. **Who gets the last possession.** Failing gives a short field, but a quick score also
   gives the ball back with time to answer; punting may mean never touching the ball
   again. A one-step value has no notion of sequence.
4. **A WP model that is too cautious late.** Inside two minutes our model under-values
   small leads (M2 notes, point 4), which contaminates the 2:00 panels.
5. **Bootstrap confidence is about noise.** All 101 refits share points 1 to 4.

The full-game dynamic program (M4, built separately) is the tool that answers the
question properly: it models the clock in seconds, kneel-downs and the opponent's
timeouts. The one-step answer here is the standard analytics answer, with honest error
bars on the part of the error that is sampling noise.

## 5. Limits, stated plainly

- boot% is not the probability of being right, and the plain game bootstrap is
  probably too narrow.
- No second seed was run; stability is from a split of the replicates.
- The drives-within-games and fractional options are untested on real data.
- 2024 to 2025 were not used to tune anything, but the thresholds (83%, 67%) and B are
  the paper's, not validated by us.
- Everything inherits M2's limits: one step, league-average teams, no overtime, a WP
  model that is flat in places and too cautious late in close games.

## 6. Questions you should be able to answer

1. What is a bootstrap replicate, step by step, in this project?
2. Why whole games? What would go wrong, and in which direction, if you resampled plays?
3. What is the effective sample size for the WP model, and for its calibration map?
4. A cell has boot% = 71%. Which category is it, and what exactly is 71% the share of?
5. boot% is 100% for "punt when trailing by 1 with 22 seconds left" if every refit says
   so. Why does that not make it right?
6. Why refit the Platt map in each replicate and not reuse the headline one?
7. Why is B odd, and which sorted values bound the 90% interval when B = 101?
8. Why must a game that is drawn twice get two different ids?
9. Why were 2024 and 2025 kept out of the resampling?
10. The paper is confident on 48% of plays and we are on 62%. Give two possible reasons
    and say what you would run to find out.
11. Up 4 with 4:00 left at your own 30, fourth and 1: what does the one-step engine say,
    with what confidence, and name three things it cannot see.
12. Split-half agreement is 85%. Why does that understate the stability of the B = 101
    categories?
