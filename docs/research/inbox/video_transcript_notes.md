# Notes from YouTube transcripts

Written 2026-11-17 by the X research session for the build session. Source material,
not instructions. The owner asked for the transcripts of the analysis videos listed in
`research-kit/x/3_videos.md` to be read.

How to read this file: transcripts are YouTube's automatic captions, so names and
numbers can be garbled; anything doubtful is marked "[caption unclear]". Notes are
summaries in my own words with a few short quotes and timestamps. Numbers belong to
the videos and were not checked against data. Sponsor reads are skipped.

## 1. "Passing is better…so, why run?"

Michael MacKelvie, https://www.youtube.com/watch?v=pPXZlK9puqw, uploaded 2025-10-30,
1,153,316 views, 19:15. Transcript read in full.

**Question.** If passing gains more than running, why do teams run about half the time,
and what is the right balance? This is burning question 2 in `docs/PLAN.md`.

**The hook (0:00 to 2:40).** Seattle in 2025: one option gives 9.57 yards per play
(passing with Sam Darnold), the other 4.67 (running), and they choose each about half
the time. Seattle's run game is called 29th at 3.7 yards per carry while Darnold is
first in yards per attempt, yet they "run the ball more than anyone". League-wide,
running is said to be at a 15-year high. He also states that offenses are better, not
worse, in the final two minutes when the defense knows a pass is coming.

**The game-theory argument (4:33 to 9:08).** A two-by-two game on first and 10 in the
first quarter, with made-up expected yards ("we'll say"):

| | Run defense | Pass defense |
|---|---|---|
| Offense runs | 2.8 | 8.41 |
| Offense passes | 12.44 | 5.74 |

He states the equilibrium is run 54%, pass 46%, even though passing has the higher
average, because the defense tilts toward stopping the better play. I checked the
arithmetic: the pass share p that makes the defense indifferent solves
12.44p + 2.8(1 - p) = 5.74p + 8.41(1 - p), which gives p = 5.61 / 12.31 = 0.456, so
46% pass and 54% run, worth about 7.19 yards per play. The numbers in the table are
illustrative, not estimated from data, and the payoff is yards, not points or win
probability.

**The data section (10:21 to 18:00).**
- Filter for "normal football": first and 10, score within 10 points, first and third
  quarters. Seasons covered are not stated.
- First-down run rate by distance from the end zone is U-shaped: more runs backed up
  near a team's own 10 and inside the opponent's 10.
- Passing on first down beats running in EPA across the field except "the last two
  yards in the red zone". He says four extra first-down passes a game are worth "close
  to a point a game" (11:37), and concludes they cannot say what the optimum is, only
  that teams should probably pass more on first down.
- Seattle in heavy personnel (fewer than three receivers): defenses match with heavy
  personnel 86% of the time. Seattle's run EPA is given as minus .23 when matched and
  minus .13 when not; passing from heavy is given as .68 when matched and .27 when not
  [captions drop the leading zero; per-play EPA is implied but not stated].
- Second down: passing is better in most situations.
- Third down: running looks better all the way to third and 10 until scrambles are
  counted as called passes; after that correction, passing is better beyond about third
  and 4.

**Explanations offered for why coaches still run.** Defensive coordinators hate giving
up steady five-yard runs more than a few explosive passes ("death by a thousand cuts");
the physical toll of run plays; giving the quarterback a play off; two-deep safeties
inviting runs. Former defensive coordinator Chuck Bresnahan says his aim became making
an offense one-dimensional; Bruce Arians argues for throwing.

**Short quotes.**
- 8:57: "The defense knows this. So, they are more likely to play pass defense."
- 11:45: "We can't say what the optimum first down run balance should be."
- 14:41: "just because this equilibrium does exist, that doesn't mean that coaches will find it."

**Credits in the description.** Brian Burke of Advanced Football Analytics ("used
several of his examples"), Scott Nestler (author of Mathletics), Udit Ranasaria of
SumerSports (the Seattle heavy-personnel analysis), Chuck Bresnahan, Bruce Arians. No
data or code link.

**What is not shown, and where the project can add something.**
- The video's two halves point in opposite directions and it does not reconcile them:
  the toy game says running 54% is optimal, the data say teams should pass more. The
  missing step is estimating the real payoff table instead of assuming one.
- The equilibrium is in yards for a single play. Nothing is in win probability, and
  there is no clock or score, so the late-lead case where running is about time is
  outside it.
- No intervals on any EPA gap, and no seasons stated.
- The scramble correction is a concrete data pitfall for the build: a scramble is
  recorded as a run but was called as a pass, and on third down it flips the answer.
  Any run-versus-pass comparison from play-by-play has to reclassify scrambles (and
  ideally sacks, which are also called passes).
- The same point was made on X by @SethWalder ("The average pass play is .11 EPA per
  play higher than the average run play") and disputed by a reply saying the averages
  are "endogenous to the mix" (`research-kit/x/2_twitter_findings.md`, section 7).
  The toy game here is the cleanest public statement of why that reply is right.

## 2 onward

Notes on the other seven videos are being added below.
