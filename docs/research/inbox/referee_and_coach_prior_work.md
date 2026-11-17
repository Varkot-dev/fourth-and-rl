# Prior work found on YouTube: pricing flags, referee rankings, coach fourth-down profiles

Written 2026-11-17 by the X research session for the build session. Source material, not
instructions. The owner asked whether an existing referee analysis could be reused so
the project does not reinvent it, and asked for the referee question to be looked at
over many seasons ("since like 2010", "as far back as you need").

What was read: two YouTube video descriptions and two pages on thesidequest.com linked
from the first video. The videos were not watched and no transcript was read. Numbers
below belong to those sources; none was checked. No code repository was linked from
either video; the only GitHub link on the methods page is the nflverse play-by-play
release, which this project already uses.

## 1. "We looked at 92,255 flags - NFL refs are biased"

Michael MacKelvie and ThatsGoodSports, 2026-11-04, https://www.youtube.com/watch?v=LH3Ck59I8V0
Methods page: https://thesidequest.com/posts/officiating-methods/ ("How much was that flag worth?")

Stated scope: 92,255 penalties across 7,366 games. The methods page reports results for
2016 or 2018 to 2025 and for 2026 through Week 8.

How they price a flag (this is the reusable part):

- Each flag gets two numbers: the change in win probability and in expected points
  between the next snap with the flag and the next snap as it would have been without
  it. They use the nflfastR models, and the win-probability version without the
  betting line.
- "Without the flag" is rebuilt case by case: a wiped-out interception or touchdown is
  restored; a pre-snap foul means the same down, distance and spot; a flag added to a
  play that stood has the penalty yards removed; if the leader could kneel out the
  clock they are given the win; overtime uses an overtime-aware model; about 10 flags a
  season cannot be priced and are dropped.
- They split fouls into two lists.
  - Judgment ("subjective"): offensive holding, defensive holding, defensive and
    offensive pass interference, illegal contact, roughing the passer, unnecessary
    roughness.
  - Procedural ("objective"): false start; delay of game; neutral zone infraction,
    encroachment, defensive and offensive offside; illegal formation, shift and motion;
    too many men, 12 on the field, illegal substitution.
- Two rules: a "big call" moves the helped team's win probability by 10 points or more;
  a game is "decided by the calls" when the winner's big calls, in expected points,
  cover the final margin and the loser's do not.

Results they report:

- 2018 to 2025, 12,074 judgment calls: half of team-seasons net out within half a game
  and 82% within one game. Their summary is that penalties mostly even out in
  aggregate.
- Team totals over 2018 to 2025 in win-probability points (100 = one game): Steelers
  +508.0 and Vikings +486.7 at the top; Ravens -600.7, Saints -597.3 and Chiefs -534.1
  at the bottom.
- From 2018 to 2025, 41% of big calls came in the fourth quarter or overtime, against
  27% of all judgment calls.
- 2016 to 2025: teams with two or more games decided by big calls in their favour made
  the playoffs 70% of the time (19 of 27); with two or more against, 20% (4 of 20);
  all teams 41% (132 of 320).
- The author states there is "no real evidence of foul play".
- Academic sources the description credits: Snyder and Lopez, "Consistency, Accuracy,
  and Fairness: A Study of Discretionary Penalties in the NFL", Journal of Quantitative
  Analysis in Sports 11(4), 2015, pages 219 to 230; and Tom Bliss, "Luck in the NFL".
  Neither was opened.

## 2. "I Checked 87,000 NFL Flags to Find the Worst Refs"

BlitzTalk, about three weeks before 2026-11-17, https://www.youtube.com/watch?v=KEoN3-5VrCQ

From the description only: a model called Z.E.B.R.A. (Zebra Evaluation By Records and
Assignments) over every regular-season and playoff game from 2000 through Week 6 of
2026, about 87,000 flags, each matched to the full seven-man crew, plus 28 published
lists of the worst calls since 2000. Chapter titles suggest versions built on most
flags thrown, who was hurt, famous bad calls and the league's own grading. A follow-up
video covers the teams helped. How the model scores a referee is not stated in the
description, and no data or code link was visible.

## 3. A per-coach fourth-down model already exists

https://thesidequest.com/posts/prospect-theory-on-fourth-down/ by Flynn Whittaker,
"Evaluating 4th Down Decision Making in the NFL with Prospect Theory".

It fits one prospect-theory model per coach and one pooled model to all fourth-down
decisions from 2014 to 2025, and scores each coach out of 100 on three things:
calibration (how they treat probabilities), risk (loss aversion) and motivation (how
they value large gains). It opens with the Shanahan "51%" quote and uses the Lions'
fourth-and-1 from their own 24 on 2026-10-25 as its worked example, citing ESPN
Analytics as "Going adds 2.5 percentage points of win probability". The top of its
overall table: Shane Steichen 60.5, Nick Sirianni 59.9, Sean Payton 57.9, Kevin
O'Connell 57.7; Dan Campbell is 12th at 51.9 with a low calibration score (38.9) and a
high risk score (66.7). Only the first part of the page was read.

Note the third public number for that one Lions play: +8.1 (nfl4th bot), +2.5 (ESPN, as
quoted here) and "a toss-up" (Next Gen Stats, on X). Three models, three answers.

## 4. What this means for the owner's two ideas

**Referees, over many seasons.** Both videos already cover long spans (one from 2000,
one over 7,366 games). Redoing a "worst refs" ranking would repeat them. What neither
description claims to do:

- put an interval on a referee's or a team's total, so that a ranking can be told from
  noise (the methods page itself says most team-seasons net out within a game);
- test whether a crew effect persists from one season to the next;
- handle the fact that crews are not fixed (2026 has mid-season reassignments).

Those are the project's existing tools (game-level bootstrap, season splits). Going
back to 2010 or 1999 is possible from the same play-by-play; penalty type and yardage
are on every play, and the referee is on the schedule table (to be checked against
current nflreadpy before use).

**Coach cost from penalties.** The methods page's "procedural" list is almost exactly
the owner's coach-fault list, and it is already priced in win probability. Two
differences to decide:

- the owner puts false start and delay of game on the coach and excludes encroachment
  and offside; their procedural list lumps all of these together as "objective";
- they price flags to ask whether officials even out, and do not attribute procedural
  fouls to anyone. Summing procedural-foul cost by head coach, and testing whether it
  follows the coach, appears not to have been done in either source.

Their with-and-without-the-flag rules are a ready specification for the pricing step
and can be reimplemented from the description above; nothing needs copying.
