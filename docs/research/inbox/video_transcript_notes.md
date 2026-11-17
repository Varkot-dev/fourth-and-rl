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

## How sections 2 to 10 were produced

Sections 2 to 9 were written by a helper agent that read each transcript in the
browser to its last caption line; all eight were found and read in full. I reviewed
sections 2 and 9 and the cross-video notes line by line and did not line-check the
rest against the transcripts. Inside sections 2 to 10 the phrase "video N" uses the
helper's own numbering, which is one lower than the section number here (its "video 1"
is section 2, the flags video; its "video 8" is section 9). On-screen tables (the
32-team table, the referee tables, the 39-coach table) are not in captions and were
not captured.

Two points where the flags video and its written methods page differ, found by
comparing section 2 with `referee_and_coach_prior_work.md`:

- The transcript defines a "big call" as a flag whose expected-points swing covers the
  final margin. The methods page defines a big call as a win-probability move of 10
  points or more, and uses the margin rule for "decided by the calls". Use the page.
- The transcript gives the Vikings 124 points of win probability from penalties; the
  page's table, through Week 8, gives 114.0. The video says it was made seven weeks in.

## 2. We looked at 92,255 flags - NFL refs are biased

- **Channel:** Michael MacKelvie and ThatsGoodSports (MacKelvie channel shows 511K subscribers)
- **URL:** https://www.youtube.com/watch?v=LH3Ck59I8V0
- **Upload date / views / length:** Nov 4, 2026 as shown; 1,582,741 views; 22,456 likes; 31:52
- **Transcript status:** read fully (0:00 to 31:52)
- **Chapters:** 0:00 it all "evens out"; 1:23 Objective and Subjective Penalties; 3:45 Frequency Fallacies; 8:21 Timing Impact; 13:41 Ad; 14:28 Playoff Officiating; 16:30 Referee Bias; 23:45 Penalty Luck Team Rankings; 24:22 The Side Quest; 24:59 Luck Team Rankings

### Question asked
Do penalties "even out" over a game, season or postseason, and is there measurable referee bias (toward teams, players, home teams, sidelines, trailing teams)?

### Data and method
- **Sample:** 92,255 penalties across 7,366 games (1:04; caption reads "7366"; description says 7,366). Seasons are not stated in the transcript. The combined luck analysis starts in 2018 because that is "the first year of data we have for player tracking in this context" (24:59). The current season is described as "only 7 weeks into the season" (28:48).
- **Penalty taxonomy (2:19 to 3:10):**
  - *Objective*: mostly pre-snap; named examples are illegal motion (caption: "a legal motion" [caption unclear]), false start, defensive offside.
  - *Subjective*: defensive holding, offensive holding, defensive pass interference, offensive pass interference, roughing the passer, illegal contact (caption: "roughing the pass or legal contact" [caption unclear]).
  - Later groupings: "coverage" flags/calls (not itemised, DPI given as the example at 21:12); "defensive aggression penalties" = unnecessary roughness, personal foul, unsportsmanlike conduct, horse collar (21:29).
- **Metrics:** win probability added (WPA) summed in "percentage points"; expected points added (EPA, called "the nerds gold standard", 10:13). Model source is not named in the transcript.
- **"Big call" definition (12:25 to 12:40):** a single flag that swung expected points enough to cover the final margin of victory of that game. A "looser definition" is used for the playoff follow-up (15:17) but is not spelled out.
- **Team quality buckets:** "great teams" vs "bad teams" (4:27); cut-offs not stated.
- **Stability check described (6:12 to 6:35):** claims penalties are not correlated game-to-game or first-half-of-season to second-half-of-season within a team. No coefficient given.
- **Penalty luck + other luck (23:50 to 25:12):** adds penalty luck to four "chance event" categories from an earlier video: opponent dropped passes, dropped opponent interceptions, fumble recoveries, and (opponent) field goals, "once we control for blocks".
- **Situation control (27:43):** trailing-team penalty difference disappears "when we control for the situation"; method not described.
- **Credits (description):** Michael Lopez, "Discretionary Penalties in the NFL"; Kevin Snyder and Michael Lopez, "Consistency, Accuracy, and Fairness: A Study of Discretionary Penalties in the NFL", Journal of Quantitative Analysis in Sports 11, no. 4 (2015): 219-230; Tom Bliss, "Luck in the NFL". Rules analyst Chuck Russo (ThatsGoodSports). Methods post linked at thesidequest.com/posts/offici... (URL truncated on the page; I did not open it). The video says data is available via thesidequest.com (24:39).

### Findings with numbers
**Timing within a game**
- Early in games officials call holding "about half as often on both runs and passes" (2:04). Objective penalties are called "just about as often in the first quarter" (2:40).

**Frequency (Frequency Fallacies, 3:45)**
- Steelers-Falcons game: Falcons flagged for holding four times, Steelers zero (3:44).
- "Last year": Bills and Cowboys each had 31 accepted holding calls; the 49ers (caption "ners") had 11 (3:57).
- Great teams flagged 6.2 times a game (caption "6.2 two"), bad teams 6.4 (4:27). The gap is described as entirely objective flags, apparently just false starts: 2.35 per game for bad teams, 2.04 for great teams (4:47).
- Belichick-era Patriots: 2.7 subjective penalties a game vs NFL average 2.71 in that era; about half a flag a game fewer objective penalties (5:08 to 5:26).
- Championship teams commit about as many penalties as very bad teams, "maybe about a half a false start less" (5:28).
- Stated conclusion: frequency of subjective penalties on a team is "completely random" (6:44).

**Timing impact (8:21)**
- Rams-Broncos DPI (overturned interception by Josh Wallace; QB Bo Nix, caption "Bo Knicks"): Broncos win probability went from 12% to 69%, a 57-point swing (8:27 to 8:44). Video says the league later told McVay the call was wrong (8:14).
- Season WPA-from-penalties leader at time of video: Vikings, 124 percentage points of WPA (9:35). Example: a defensive holding on "Xavier McKenna" [caption unclear] that wiped out a possible game-ending interception.
- On WPA alone, as many playoff teams were above average as below (9:53 to 10:04).
- On EPA "big calls": teams with two or more games with a big call in their favour made the playoffs 70% of the time; teams with two or more big calls going to the opponent made the playoffs 20% of the time (10:19 to 10:42; repeated 12:49). Framed as "about four teams every year".
- Big calls decided about 6.7% of regular-season games under the definition, "about one game a week or 18 games a season" (12:25).
- Titans example under coach "Big Mike" (surname not given in captions): 6 and 11 in his last season; week 16 vs Seattle, leading 17-13, DPI with "114" left in the fourth (i.e. 1:14 [caption unclear]) moved Seattle to the Tennessee 5, Titans lost by three; second example vs Colts, third quarter, second and 10 at Tennessee 40, Titans leading 13-10, deep DPI (11:03 to 12:00).
- Teams that got these big flags and made the playoffs went 5 and 12 once there; every other playoff team "effectively 500" (13:07).

**Playoff officiating (14:28)**
- Saints no-call: a penalty would have given the Saints "at least an 85% chance of making the Super Bowl" (14:41).
- Seven big calls (original definition) in the last 10 playoff seasons; three of those seven beneficiaries won the Super Bowl (14:50 to 15:07).
- Under the looser definition, teams that won a playoff game with a big call won their next game 37% of the time (15:17).
- Rams vs Cincinnati Super Bowl holding call cited as the biggest (15:33).
- 27% fewer subjective penalties in the playoffs (15:58 to 16:07).

**Referee bias (16:30 to 23:45)**
- Mahomes, regular season: about 5.6 roughing-the-passer or unnecessary-roughness calls "per 1,000" (denominator given later as dropbacks) vs league 6.2 (17:12).
- Coverage flags, regular season: Mahomes 26.4 vs league 28.3 (17:29). Unit not restated; presumably per 1,000 dropbacks, but the transcript does not say.
- At Arrowhead, Chiefs opponents are flagged 12% more, described as a home benefit double the league average (17:49).
- Visitors at Arrowhead commit "almost three times as many penalties as the average visiting opponent leaguewide", attributed to crowd noise (18:20). The penalty type is not specified in the captions, and this sits awkwardly beside the 12% figure; treat as [caption unclear / unspecified category].
- Playoffs, roughing the passer or unnecessary roughness: 4.5 per 1,000 dropbacks for NFL teams; Mahomes 9.7 (18:59 to 19:13). Next line says he has drawn "nearly double the calls at Arrowhead" (19:15); unclear whether 9.7 is all playoffs or Arrowhead only.
- Playoffs, coverage calls: Chiefs drew 28.1 vs league average 18 (20:40).
- Sideline effect (21:12 to 21:46): defensive pass interference happens "about twice as frequently" as a team approaches midfield on the offense's own sideline; the same roughly 2x effect for defensive aggression penalties (unnecessary roughness, personal foul, unsportsmanlike conduct, horse collar) drawn by the offense near midfield on its own sideline. Described as league-wide. Illustrated with a Bengals DPI on Mike Hilton (#21).
- Mahomes at Arrowhead in the playoffs: the defense is flagged for coverage "more than twice as often" as when his opponents drop back in the same game; Chiefs get "around 60% more coverage calls than the average home playoff team" (22:18 to 22:33).
- Chiefs playoff penalty WPA by run, out of 108 playoff teams in the Mahomes era (22:43 to 23:31): seven playoff appearances with Mahomes; 2018 run: 57 points of win probability from two playoff games, second overall of 108; next year: 43 points, won the Super Bowl, penalty luck fourth "for that season"; 2022 and 2023: fifth and 12th, both positive. Significant negative penalty luck only once in seven trips. Number one overall for subjective calls in the Mahomes era. Least lucky: the Bills (23:31).

**Penalty luck / luck team rankings (23:45, 24:59)**
- Combined luck (four chance categories + penalty luck), 2018 onward: Seahawks third, Texans second, Steelers first with "over 1,000 WPA, about twice that of second place" (25:03 to 25:24). (The sentence order is confusing: the captions name the Texans as second, then say the top team has twice the WPA of second place and name the Pittsburgh Steelers; the following passage makes clear the Steelers are the luckiest.)
- Steelers won about 10 more games than point differential implies over the period; their summed luck was worth 10.3 wins (25:36 to 26:07). Correlation of scoring differential with wins across a season "about .91" (caption "about.91", 25:46).
- Luckiest quarter of teams made the playoffs 55% of the time; unluckiest quarter 33% (26:07 to 26:23).
- Chiefs regular season, Mahomes era: third to last in penalty luck (26:23).
- Postseason penalties combined with luck score: Patriots dead last; Packers third; Eagles second at 83.7 WPA; Chiefs first at 250 WPA, "three times the luck score" of second (26:37 to 27:06).

**Debunks (27:15 to 28:54)**
- Trailing teams do receive more penalties, but "entirely" because they throw more; no difference in subjective penalties after controlling for situation (27:36 to 27:49).
- Penalties this season are up "half a penalty a game in terms of subjective calls" (28:37).
- Nullified touchdowns are not at a record: more in 2024 and in 2019 (28:42).

### Caveats the video itself states
- Not all flags are random; holding has a gradient of severity (5:50).
- The big-call measure "is not a perfect way to measure all flags because that's impossible" (12:40); WPA "has its limitations" (10:04).
- Playoff findings are repeatedly called small sample (15:24, 20:25, 20:47, 23:41).
- Many of the big calls shown were correct calls; the point is that they could have gone uncalled (11:35, 12:00, 15:41).
- Luck "does not mean rigged" (27:06). Bias "gets murky when it's applied to a player or team specifically" (29:30).
- Mahomes' playoff roughing numbers may reflect him extending plays more in the playoffs, not officiating (20:25).
- Officiating alone will not get a team into the playoffs (13:15).

### What is NOT shown (from the transcript)
- No seasons/date range for the 92,255 flags; no named data source or WP/EP model.
- No confidence intervals, p-values or sample counts behind any rate (sideline 2x, Arrowhead 12%, playoff 27%, Mahomes 9.7 vs 4.5, 70% vs 20%).
- The 70%/20% playoff split is not adjusted for team quality or for the fact that winning a close game mechanically helps make the playoffs (the "big call" is defined using the final margin, i.e. conditioned on outcome).
- No denominator for how many team-seasons had two or more big calls each way.
- The within-season non-persistence claim is asserted without a correlation value; no season-to-season repeat test for team penalty luck, and no per-referee analysis at all (no individual referee or crew is named or ranked, despite the title).
- Whether declined/offsetting penalties are included is not stated (only "accepted holding calls" for one stat).
- The sideline effect has no stated control for where plays/targets actually occur.
- "Great" and "bad" team thresholds not defined; "looser definition" of big call not defined.
- Code and data are said to be available via a linked post, but nothing is shown in the video itself.

### Short verbatim quotes
- "we looked at 92,255 penalties across 7366 games" (1:04)
- "The frequency of subjective penalties called on a team is completely random" (6:44)
- "Overall, big calls decided about 6.7% of regular season games under this definition." (12:25)
- "The offense will get the penalty about twice as frequently when it happens around midfield on their own sideline." (21:38)

---

## 3. Why Teams Come Back - The Rubber Band Effect

- **Channel:** Michael MacKelvie (511K subscribers)
- **URL:** https://www.youtube.com/watch?v=8By4b6cTVFU
- **Upload date / views / length:** Nov 26, 2025; 733,814 views; 17,625 likes; 18:09
- **Transcript status:** read fully (0:00 to 18:09)
- **Chapters:** 0:00 Losing Leads to Winning; 1:47 The Rubber Band Effect; 2:55 "Prevent" Analysis; 8:45 Ad; 9:44 Coaches Corner - Possession Math; 10:07 Why'd they wait so long? An analysis; 14:52 "Just Don't Get Beat Deep"

### Question asked
Why do trailing teams seem to play better late in games? Is the "rubber band effect" caused by prevent defense, by the offense's situation, or by leading teams being too conservative?

### Data and method
- Seasons and sample sizes are not stated anywhere in the transcript.
- Metrics: scoring rate per possession (field goal rate, touchdown rate), EPA per play by score state and quarter ("Adjusted EPA Per Play Rubber Band graph", per description), completion percentage by target depth, deep-pass completion vs coverage type, safety depth over game time (tracking-derived, "with the help of a friend").
- Definitions: "normal football scenario" = first or second quarter, tied (1:58). "Two-minute drill" = final two minutes. Score buckets: down 1 to 3; down 4 to 7. "Deep" = over 20 yards (15:26). Prevent treated as a gradient from "drop 11" to "drop eight" (eight in coverage, three rushers) (5:26 to 6:04).
- Credits (description): Lau Sze Yui (built the adjusted EPA rubber-band graph and safety depth graph); Sam Schwartzstein, Amazon TNF (caption "Sam Schwarzene"); Ben Baldwin (earlier look at late-game offenses); Udit Ranasaria (the friend who raised the defensive point). Unnamed coaches speak in the "Coaches Corner" clip.

### Findings with numbers
- Leading team in the fourth quarter scores on about 29% of possessions; trailing team in the final two minutes scores on 49% (0:00 to 0:15).
- Normal scenario: field goal rate 17%, touchdown rate 23%, about 40% scoring per possession (1:58 to 2:15).
- Down 1 to 3 in the two-minute drill: field goal 26% of the time, touchdown 23% (2:15 to 2:22).
- Down 4 to 7 (needing a touchdown): touchdown 37% of the time (2:22 to 2:30).
- Bills vs Vikings example: Bills trail by a field goal with 36 seconds left; Vikings safeties about 20 then 25 yards deep with five other DBs about 10 yards off; gains of 12, 8, 20 and 15 yards; four plays in 19 seconds to reach field goal range and force overtime (2:56 to 4:32).
- In the first three quarters there "isn't much difference in EPA per play" by score state; the gap opens in the fourth (4:48).
- Rams dropped all 11 against the Seahawks on one play and it still failed (5:34 to 5:56).
- Bengals used drop eight on roughly one of every three plays vs Mahomes in the 2022 AFC Championship (from re-watching film); Mahomes threw for 55 yards in the second half (6:30 to 6:42).
- Versus drop eight: quarterbacks are "slightly better"; completion percentage and interceptions "relatively stable"; deep passes completed 36.8% vs drop eight against 35.5 for all other coverage types (7:32 to 8:05).
- Safety starting depth increases as the game goes on, by "only a few yards" (8:05 to 8:18).
- Coach's rule of thumb: with a two-score lead, ask whether the opponent has time for two 10-play drives; rush three to defend the sidelines and keep the ball in bounds (9:44 to 10:07).
- Average fourth-quarter drive takes about 3 and 1/2 minutes; with a three-possession lead give up 5 to 10 yards a play and tackle in bounds (10:10 to 10:19).
- Key correction: trailing quarterbacks' EPA in the fourth quarter "plummets" even though scoring rate rises (12:05 to 12:44). Explanation: must-go situations (e.g. Buccaneers fourth and 10 at own 32, down five to the Texans with 1:24 left, converted by a 15-yard Baker Mayfield scramble) produce more scores but also far more turnovers/turnovers on downs, so the same play style would be bad in neutral situations (12:44 to 13:52).
- Claims teams still run "far too frequently on first down"; frames run/pass as a Nash equilibrium where the offense should keep choosing the more lucrative option (passing) until the defense adjusts (14:01 to 14:27). Rams vs Seahawks cited: Rams used a pass-oriented defense instead of matching heavy personnel and Darnold threw four interceptions (14:27 to 14:46).
- Completion percentage by target depth: normal football, 36% of passes over 20 yards completed (15:18 to 15:34). With 2 minutes or less, down by three: stable under 20 yards, 17.7% over 20 yards (15:42 to 15:57). Down 4 to 7: lowest completion under 20 yards, but 31% over 20 yards (16:04 to 16:22).
- Hypothesis: defenses leading by three or fewer over-protect against the deep ball and concede field-goal drives, a loss-aversion style miscalculation (15:03, 16:26 to 17:05).

### Caveats the video itself states
- Drop eight is not purely a prevent defense; it can also be a timing-disruption call (6:20, 6:58).
- Higher late scoring rate does not mean offenses are more efficient; EPA says the opposite (12:29).
- The deep-ball loss-aversion explanation is offered as "my belief" (16:22).
- The psychological component cannot be quantified (17:10 to 17:29).

### What is NOT shown (from the transcript)
- No seasons, play counts or possession counts for any rate.
- No confidence intervals on 36.8 vs 35.5 or on 17.7% vs 31% (the latter are narrow situations, likely small samples, but sizes are not given).
- "Adjusted EPA" adjustment is not explained in the video.
- No control for which teams trail (team quality) or for clock/timeouts in scoring-rate comparisons.
- No estimate of the win-probability cost of prevent defense, and no coach-level breakdown.
- The first-down run/pass claim cites "previous videos"; no numbers are given here.
- No code or data link in the transcript.

### Short verbatim quotes
- "In the final two minutes, they'll score on 49% of all possessions." (0:15)
- "Trailing quarterbacks in the fourth quarter, their EPA plummets." (12:13)
- "The hit rate is only 17.7%." (15:49)
- "Coaches might lean a little too conservatively with their play calling when they establish a lead." (17:36)

---

## 4. I Checked 87,000 NFL Flags to Find the Worst Refs

- **Channel:** BlitzTalk (5.75K subscribers)
- **URL:** https://www.youtube.com/watch?v=KEoN3-5VrCQ
- **Upload date / views / length:** Oct 26, 2026; 128,303 views; 1,882 likes; length not read directly (an ad was occupying the player timer); last caption line is at 17:14, final chapter starts 17:03
- **Transcript status:** read fully (0:00 to 17:14)
- **Chapters:** 0:00 Cold Open; 0:42 The Model: Z.E.B.R.A.; 2:42 v0.1: Most Flags Thrown; 3:57 v0.2: Who Got Hurt; 6:37 v0.3: The Most Famous Bad Calls; 10:14 v0.4: The League's Report Card; 13:07 Who Gets Sent Where; 15:22 The Team The Refs Hurt Most; 17:03 Next: Who The Refs Helped

### Question asked
Which NFL referees are the "worst" over the past 25 years, and which team have the officials hurt the most since 2000?

### Data and method
- **Model name:** Z.E.B.R.A. = "Zebra Evaluation By Records and Assignments" (1:09).
- **Data (description):** nflverse play-by-play and game officials, 2000 to 2025 seasons plus 2026 through Week 6; regular season and playoffs; about 87,000 flags, each matched to the full seven-man crew for that game. Plus 28 published "worst calls since 2000" lists merged into one consensus top 50; NFL officiating statements and Football Zebras for officials' moves and discipline. Earned vs judgment split credited to commenter @CCulliton. "Format inspired by HQE's I Built a Model series."
- **Key limitation stated up front (1:38):** play-by-play does not say which of the seven officials threw the flag, so grading a referee is really grading the crew he runs. Play-by-play also contains only flags actually thrown (no no-calls) (6:47).
- **Earned vs judgment split (1:46 to 2:07):** *earned* (charged to the team) = false starts, offside, 12 men on the field. *Judgment* (charged to refs) = holding, pass interference, roughing the passer. (Video 4 clarifies that illegal contact, defensive holding and offensive pass interference were also in the judgment set, and that ineligible man downfield was counted as earned in this version.)
- **Team "hurt" measure (2:07 to 2:38):** judgment calls a team drew minus judgment calls against it, each valued "in points", including touchdowns and turnovers wiped out. Split into two halves: (a) the team's own flags (could be refs or the team); (b) whether the team's opponents are flagged less against it than those same opponents are flagged in their other games. Half (b) is treated as the part that "really points to the refs". How points are assigned is not specified (no EPA model named in the transcript).
- **Eras (3:58 to 4:24):** regular season split into three eras at breaks in flag rates: 2000 to 2011, 2012 to 2019, 2020 on. Rule changes cited: 2011 to 2013 replay on every scoring play and the crown-of-the-helmet rule; in 2020 offensive holding fell from about 2.7 flags a game to 1.8 in one year.
- **Playoff version (5:24, 15:32):** playoff opponents' judgment calls compared with how often those opponents are flagged in their own regular seasons, "with the playoffs running a little lighter on flags".

**How each version scores a referee**
- **v0.1 "Most Flags Thrown" (2:42):** worst ref = crew that throws the most flags (extra flags per game vs average; current season "blended" with each referee's history; blending weights not given).
- **v0.2 "Who Got Hurt" (3:57):** does not score referees directly; builds the team-level points measure above, then tests every referee-team pair and every on-field official-team pair (about 17,000 pairs; caption "17,000 parents" [caption unclear]) plus coaches and players for a persistent lean.
- **v0.3 "The Most Famous Bad Calls" (6:37):** counts each referee's appearances in the consensus top 50 worst calls vs the number expected from his workload; a stricter cut keeps only calls that decided the game and were admitted or widely called wrong; a separate list of 41 calls since 2010 that the NFL or nearly everybody called wrong.
- **v0.4 "The League's Report Card" (10:14):** first tries "mistakes" = replay overturns and big flags late in close games, finds them to be noise, then falls back to the league's own playoff assignments: playoff selections per season as referee, excluding the rookie season (no referee in the data has worked a playoff game in year one), compared with what a "normal ref" of the same tenure gets.

### Findings with numbers
**v0.1**
- Shawn Hochuli (caption "Shawn Hawkley, Ed Hawk's son") near the top for flags since 2000; picked for the playoffs in seven of his eight seasons as referee (2:43 to 3:09).
- Scott Novak's crew: almost eight extra flags a game this season, over only six games (2:54). After blending history, Alex Moore's crew is the heaviest (3:01).
- Raiders: most penalized team since 2000, about one extra flag a game, half of it false starts and offside; 488 more flags than the average team since 2000, only about seven of those attributable to the crews they drew (3:09 to 3:27).
- Crews a team draws explain under 1% of how many flags it gets in a season (3:27).
- Crews differ and "a heavy crew stays heavy from year to year"; lightest-to-heaviest gap is about two flags in the same game, while a normal game swings about four on its own (3:35 to 3:43).
- Raiders' opponents were flagged slightly more against them than in other games (3:51).

**v0.2**
- 2000 to 2011: Raiders most hurt, about a point a game, their own flags (4:24).
- 2012 to 2019: Seahawks, with Rams and Cowboys basically tied; Seattle about half own flags, half calls not received (4:24 to 4:32).
- Since 2020: Chiefs most hurt in the regular season, almost all their own flags; calls on their opponents about where expected (4:42 to 4:50).
- Chiefs playoffs: in an 11-game stretch from January 2022 (per ESPN) opponents were flagged 66 times to the Chiefs' 35, the most lopsided 11-game playoff run since 2000; in Mahomes' first three playoff runs (caption "2018 and 2020") the Chiefs were behind by two flags (5:00 to 5:15). Since 2020, Chiefs playoff opponents drew about one more judgment call a game than expected, offset in points by the Chiefs' own flags (5:24 to 5:42).
- All 27 seasons: Ravens most hurt, with Raiders and Bills close enough for a "three-way tie"; about two-thirds of the Ravens' figure is their own flags (roughing the passer and personal fouls on defense); Ravens come out slightly ahead in the playoffs (5:42 to 6:06).
- Referee-vs-team grudges: about 17,000 pairs tested, "none of them held up"; same null for every coach and player tested (6:06 to 6:22).
- Star protection: top quarterbacks get roughing calls at about the same rate per hit as others; Tom Brady is "34th or 41" [caption unclear, likely "34th of 41"] since 2009; Josh Allen third (6:22 to 6:31).

**v0.3**
- Consensus top 5 worst calls: 1 Tuck Rule (referee Walt Coleman; caption "Coleman"); 2 Fail Mary (replacement ref Wayne Elliott, 2012 lockout); 3 Rams-Saints NFC title no-call, 2019 (referee Bill Vinovich; nearest officials down judge Patrick Turner and side judge Gary Cavaletto, caption "Gary Cavalto"); 4 Super Bowl 40 (Bill Leavy, caption "Bill Levy"); 5 Dez Bryant catch (Gene Steratore, caption "Jean Territory"). Number 9: picked-up flag in Dallas vs Lions, Pete Morelli's crew (6:54 to 7:29).
- Gene Steratore appears in six of the top 50 vs about two and a half expected from workload; he also refereed Super Bowl 52; three of his six were the old catch rule applied as written or rulings that proved right (7:33 to 7:49).
- 31 of the 50 were playoff games (playoffs are about 4% of all games); 12 were Super Bowls (7:49 to 8:01). 12 involve pass interference, 11 the catch rule (8:04). 13 of the 50 were the rule as written or a call that held up (8:23).
- Biggest swings since 2006 among decisive, admitted/widely-wrong calls: Fail Mary; then Rams at Broncos this season (crew name dropped in captions, "where's crew" [caption unclear]; video 4 attributes it to Vinovich's crew), Josh Wallace DPI wiping out an interception, McVay says the NFL told him it should not have been thrown; then two Lions games: illegal bat in Seattle 2015 (NFL admitted) and the Taylor Decker two-point play in 2023 (league backed it; Brad Allen's crew reportedly kept out of the playoffs) (8:23 to 9:05).
- List of 41 calls since 2010: Lions on the wrong end of five and the right end of none, most of any team (9:05 to 9:22). But Lions' opponents were flagged slightly more against Detroit than elsewhere; over 27 regular seasons the Lions rank "12" [caption: "came out 12"]; most of their hurt is their own flags, mostly pass interference; only seven playoff games, too few to rank (9:22 to 9:45).
- Saints: ninth over 27 regular seasons; about normal in the playoffs (9:45).
- Consequences: after a clear error the referee worked the next playoffs 53% of the time vs a normal rate of about 60%. Vinovich refereed Super Bowl 54 the season after the no-call. Coleman, "Ron Wy" [caption unclear] and Morelli sat out the next postseason (9:54 to 10:11).

**v0.4**
- Flag volume separates referees well; replay overturns are "mostly noise"; in the playoffs every measure tried is "pure noise" (10:23 to 10:41).
- 17 referees working this season. Shawn Hochuli and Shawn Smith picked every year since their first; Ron Torbert missed once in 11 with two Super Bowls; Vinovich picked 14 times in 16 with three Super Bowls; Clete Blakeman high too despite four public bad calls (tied for most), two in AFC title games (10:56 to 11:19).
- **Named worst by playoff selection:** Brad Rogers, Scott Novak, Land Clark (caption "Lan Clark"): each a referee for six or seven seasons, each picked for the playoffs once, vs about three or four for a normal referee of that tenure (11:19 to 11:42).
- Adrian Hill: two calls the NFL or a team said were wrong (one a DPI on Cooper DeJean, caption "Cooper Deene", Commanders at Eagles 2024, which Walt Anderson said should not have been called); a one-game suspension this season with line judge Jeff Seeman (caption "Seaman"); playoff picks about normal; his crew is second heaviest this year after blending; none of the top 50 are his (11:42 to 12:23).
- Alex Moore: referee only since last season, too early. Tra Blake (caption "Trey Blake") refereed 2022 to 2024, moved back to umpire April 2025. Brad Allen: four public bad calls, still picked about normally (12:23 to 12:46).
- Could not find any way the most-trusted referees call playoff games differently (12:54).

**Who gets sent where (13:07)**
- After the Tuck Rule, Walt Coleman (caption "Paul Coleman" [caption unclear]) refereed 253 more regular-season games over 17 seasons; captions say "now one of them was a Raiders game" (likely "not one" [caption unclear]) when about 16 expected; longest ref-team gap since 2000; next longest 146 games. Rest of that crew got Raiders games at a normal rate; Walt Coleman IV worked them nine times as an on-field official (13:11 to 13:34).
- Vinovich went 116 regular-season games (caption "won 116") without a Saints game until Steelers-Saints (caption "steals and Saints") in Paris; about seven expected; saw the Rams 10 times, about normal (13:45 to 14:01).
- Home-state avoidance: referees got teams from their home state 577 times vs about 634 expected; Jeff Triplette had the Panthers five times vs about 18 expected (14:01 to 14:16).
- Home teams get about one extra judgment call every seven games; no individual referee leans home more than others (14:16 to 14:23).
- From week 13 on, games between two winning teams get more trusted referees. The five most-trusted referees each season work about 29% of all games, but since 2021 got eight of the 14 week-18 games between winning teams (14:23 to 14:47).
- This season: 14.8 flags a game through week six, more than any full season since 2000; offensive holding alone over three a game. Four heaviest crews after blending: Moore, Hill, Novak (caption "Novag"), Blakeman (14:47 to 15:04).
- Adding crew to a flag prediction improved it by only about 1% (15:04).

**Team the refs hurt most (15:22)**
- Regular season: three-way tie of Ravens, Raiders, Bills. Playoffs: Bills. In 16 playoff games the Bills drew 27 judgment calls vs about 46 expected for their opponents; the Bills themselves were flagged slightly less than their regular-season rate (15:24 to 15:55).
- Pass interference: a normal playoff team is called about six times in 16 games; Bills eight, their opponents three (16:03). Spread across seven referees; Brad Allen had four of the games (all wild cards) and they were the worst; excluding him opponents were still called about one fewer time a game than normal; gap is similar vs non-Chiefs opponents (16:11 to 16:28).
- Three Bills calls on the top 50: Music City Miracle; Josh Allen fourth-down sneak in the AFC title game (Blakeman); Brandin Cooks overtime interception ruling in Denver, January 2026 (16:28 to 16:44).
- Bills are the only team in the top three in both regular season and playoffs (16:52).

### Caveats the video itself states
- Grading a referee is grading his crew (1:46).
- No-calls are invisible to the model (6:47).
- This-season figures are only six games (2:54).
- Famous-call lists are biased toward playoff games, which go to the most trusted referees (7:57).
- Playoff selections are "the NFL's opinion, not concrete proof" (12:38).
- 16 Bills playoff games is "a small sample"; wants more playoff runs (16:44). Lions' seven playoff games are too few to rank (9:36).
- Does not have data to say whether anything is "rigged" (15:12).
- The Chiefs playoff flag gap depends on where you start counting (5:15).

### What is NOT shown (from the transcript)
- No explicit formula for points per flag, for the "expected" counts, or for blending current season with history; no named EPA/WP model in this video.
- "None of them held up" for 17,000 pairs: no test statistic, threshold or multiple-comparison method stated.
- "Heavy crew stays heavy year to year" and "under 1%" are stated without a correlation or R-squared definition.
- No confidence intervals for team rankings; the three-way tie is asserted qualitatively.
- No full list of the 17 referees' scores in the transcript (any on-screen table was not captured).
- Declined/offsetting penalty handling not stated.
- No code link; sources are listed in the description (nflverse, Football Zebras, etc.).

### Short verbatim quotes
- "when I grade a referee, I'm kind of grading the crew he runs" (1:46)
- "the crews a team draws explains under 1% of how many flags it gets in a season" (3:27)
- "replay overturns are mostly noise. And in the playoffs, every single measure I tried comes out as pure noise" (10:32)
- "When I tried predicting flags with the crew included though, it only got about 1% better." (15:04)

---

## 5. I Found The Teams Helped Most By The Refs

- **Channel:** BlitzTalk (5.75K subscribers)
- **URL:** https://www.youtube.com/watch?v=OX6tCqzOlaY (found via YouTube search)
- **Upload date / views / length:** Nov 2, 2026; 44,840 views; 491 likes; 16:08
- **Transcript status:** read fully (0:00 to 15:53 last caption)
- **Chapters:** 0:00 Who Have The Refs Helped?; 0:31 The Model: Z.E.B.R.A. v2; 1:38 v0.1: The Team Everybody Says; 3:43 v0.2: Who The NFL Trusts; 4:28 v0.3: The Same 50 Calls; 5:48 v0.35: The Flags Not Thrown; 8:06 v0.4: Your Theories; 10:58 v0.5: Who The Refs Helped; 12:26 The Playoff Answer; 14:28 The Most Impactful Call Since 2000

### Question asked
Flip of video 3: which teams have officials helped the most since 2000, which referees does the league trust most and which stay off the wrong-call lists, and what can be learned about flags that were never thrown?

### Data and method
- **Data (description):** nflverse play-by-play and game officials, 2000 to 2025 plus 2026 through Week 7 (107 games this year, about 87,000 flags since 2000, 0:55 to 1:14); NFL playoff referee assignments; NFL on-field fines 2023 to 2026; the consensus top 50 worst calls from the 28 lists; Football Zebras. "Win-chance swings from nflverse win probability."
- **Z.E.B.R.A. v2 change (0:47 to 1:02):** judgment set confirmed to include illegal contact, defensive holding and offensive pass interference (already in v1); ineligible man downfield moved from earned to judgment; everything re-run.
- **Team "helped" measure (1:14 to 1:31):** mirror of v1. Half one: team flagged less than the league (refs or discipline). Half two: team's opponents flagged more against it than in their other games (the half that "points to the refs"). Every call counted in points.
- **Referee scoring by version:**
  - **v0.2 "Who The NFL Trusts" (3:43):** playoff selections per season as referee, rookie year excluded.
  - **v0.3 "The Same 50 Calls" (4:28):** each referee's count on the top 50 plus the 41 admitted/near-consensus wrong calls since 2010, compared with the count expected from how many big games he worked. Also tested "fairness" (lean to home team or betting favourite) year over year, and replay overturns.
  - **v0.35 "The Flags Not Thrown" (5:48):** three partial windows on no-calls: (1) NFL weekly in-game fines since 2023 (with quarter and clock) matched to whether a flag was thrown; (2) 2019, the one year replay could add pass interference; (3) holding rates by crew and regular season vs playoffs.
  - **Final "best referee" pick:** top of both the playoff-selection list and the wrong-call list, with a check that the crew calls holding at or above league rate.
- **Win-chance scoring / "ref points" (13:12 to 13:29):** every call scored by change in win probability (term credited to a commenter, caption "Bwee man"). Used for team rankings and for the most impactful call list (flags plus no-calls "on the record").
- **Betting tests (10:13 to 10:40):** second-half flags vs the spread; games vs a "fake line" (placebo); referee-level results on totals for referees with 50 games.

### Findings with numbers
**Chiefs (1:38 to 3:43)**
- Playoffs since 2020: about even; opponents get about one extra judgment call a game, offset by the Chiefs' own flags (1:40 to 1:57).
- Starting in 2018 it leans to the Chiefs by about a point and a half a game; since 2000 they are fourth in the playoffs (1:57 to 2:04).
- Super Bowls: in their three wins the only hold on the Chiefs came on a kickoff return; in the two losses the line was called five times (counting one declined). 21 of the 52 Super Bowl teams since 2000 had zero holds, and 10 of those lost; in two of the three Chiefs wins the opponent also had zero (2:04 to 2:36).
- All Chiefs playoff games: 24 holds called vs about 24 expected (2:36).
- Pass interference drawn in the playoffs since 2018: 11 vs about five expected (2:43 to 2:51).
- Since 2018, judgment flags have cost the Chiefs more touchdowns and turnovers than they saved, by more than any team; "a gap that size could just be chance" (2:58 to 3:07).
- Referee Carl Cheffers (captions "Carl Jeffers"/"Sheffer's"): a commenter's test on 14 Chiefs games found about a flag and a half more than expected per game; rerunning that method gave under half a flag; under the video's method his crews flag both the Chiefs and their opponents more, cancelling out (3:07 to 3:40).

**Referees (3:43 to 8:03)**
- Most trusted: Bill Vinovich, picked 14 times in 16 seasons, three Super Bowls; Shawn Hochuli and Shawn Smith picked every year since their first (4:00 to 4:09). Vinovich's crew threw the Josh Wallace flag this season (4:17).
- Vinovich's wrong-call count is about what workload predicts (4:46).
- Shawn Hochuli: zero of the top 50 and zero of the 41, vs about two expected (4:53 to 5:07).
- Ron Torbert: playoffs 10 times in 11 seasons, two Super Bowls; zero of the 41 vs about two expected; two on the top 50, equal to workload (5:07 to 5:27).
- No referee leans to the home team or the betting favourite two years in a row; replay overturns do not separate referees; "the only thing a crew really keeps from one season to the next is how many flags it throws" (5:27 to 5:48).
- Fines since 2023: about 1,200 physical plays fined, about 600 never flagged; excluding hip-drop and helmet hits still about one in three (caption "1 and three"). No referee stands out (5:58 to 6:34).
- 2019: New York added 21 pass interference flags via replay (6:34).
- Holding: since 2021 Adrian Hill's crew calls about one more hold a game than Brad Allen's. In the playoffs teams are called for about a third fewer holds than in their own regular seasons, while false starts drop only about 10% (6:42 to 7:05).
- Bills: eight holds in 16 playoff games vs about 15 expected; their opponents four vs about 10 expected (7:05 to 7:14).
- **Best referees named:** Shawn Hochuli and Ron Torbert (cannot be split); both call holding at or above league rate. Torbert preferred for a Super Bowl. All-time most trusted: Ed Hochuli, picked 16 times in 18 seasons in the data (7:14 to 7:55).

**Viewer theories (8:06 to 10:58)**
- 49ers (caption "ners"): since 2000 about normal; since 2020 their opponents are called for holding less than against any other team, about 20 fewer holds than expected; "could just be chance"; normal in the playoffs (8:08 to 8:33).
- 2009 NFC title game (Morelli): two overtime flags, both on Minnesota on the winning drive; over the whole game the Saints were flagged almost twice as much (8:33 to 8:54).
- Across every regular-season game since 2006, the gap in judgment calls between the two teams is about two, "what you'd get flipping a coin on every flag" (8:54 to 9:11).
- Packers-Lions week seven (Alex Moore's crew): 13 accepted flags (below this year's average); judgment calls six on Green Bay to three on Detroit; nearly half of Detroit's cost from one pass interference on "Cece" [caption unclear] that wiped out his own interception; one of the seven worst team-games through week seven out of 214; crew moved off Sunday Night Football the next week per Football Zebras (9:11 to 10:03).
- Betting: second-half flags do not lean toward the spread in a steady way; games land just as near a fake line; captions say "now one ref with 50 games beats chance on the total" (likely "not one" [caption unclear]) (10:23 to 10:48).
- Patriots: over 27 regular seasons and 46 playoff games, about normal both ways (10:48).

**Who was helped (10:58 to 14:20)**
- Regular season, 27 years: Colts, about 7/10 of a point a game, described as the biggest help number and "too big to be luck"; about 87% of it is the Colts not getting flagged; calls on their opponents about normal (11:00 to 11:21).
- 2004 Colts drew more pass interference and defensive holding than anybody (the "Ty Law rule", Bill Polian), but opponents were flagged less against the Polian Colts than elsewhere. The ref-side help came after Polian left; 2012 is the most-helped team season in the data (11:21 to 11:48).
- The Colts' not-getting-flagged portion is kept out of the creator's separate luck model (11:48 to 12:07).
- In the playoffs the Colts are the second most hurt team on the ref-pointing half (12:07 to 12:14).
- **Playoffs: Arizona Cardinals most helped.** In 10 playoff games opponents drew 42 judgment calls vs about 26 expected; biggest piece is offensive holding (refs or pass rush, cannot be separated); 5 and 5 record in those games; still first with the 2021 wild card removed; also first in the regular season on the ref-pointing half. Super Bowl 43: Arizona nine judgment calls to Pittsburgh's six (12:27 to 13:12).
- Since 2006 about 4,600 drives were extended by a judgment flag after a failed third or fourth down; 42% ended in a touchdown (13:21 to 13:29).
- Win-chance ranking: regular season Colts first, Cardinals third; playoffs Cardinals 17th, Eagles first, Chiefs third, "close enough to call a chance" (13:29 to 13:51).
- Cardinals' edge in calls drawn is about double the next team; a gap that size "shows up about one time in 75" (13:51 to 14:07).
- A 32-team table is shown on screen (14:07); not in the captions.

**Most impactful calls since 2000 by win-probability swing (14:28 to 15:31)**
- 5: Bowman no-call on Roddy White, NFC title game, January 2013. 4: botched field goal, Giants-49ers, admitted by the league next day. 3: replay wiping out Derrick Johnson's fumble return, Titans-Chiefs. 2: Music City Miracle. 1: Fail Mary.
- Saints no-call does not make the top five because the model cannot value the clock the Saints would have run. Myles Jack play moved win chance only about seven points (Jacksonville already up 10).

### Caveats the video itself states
- No-calls remain mostly invisible; nobody outside the league can count them fully (5:50).
- A fine is not the league saying the referee erred (6:22).
- Cannot separate referees letting holds go in the playoffs from players holding less (6:57).
- Hochuli/Torbert evidence is not "big enough to call proof" (7:37).
- Several gaps "could just be chance" (Chiefs TD/turnover gap, 49ers holding, playoff win-chance ranks).
- Cardinals: only 10 games, "take that one with a grain of salt"; offensive holding drawn may reflect pass rush (12:43, 13:51).
- Carl Cheffers: only 14 games, two or three big ones swing each side (3:32).
- Betting tests only see flags thrown (10:40).
- Chiefs conclusions depend on the starting year (2:04).

### What is NOT shown (from the transcript)
- No formula for expected counts or points per call; the "one time in 75" and "too big to be luck" claims have no stated test or multiple-comparison adjustment (32 teams, two halves, several eras are all being searched).
- The fairness-does-not-repeat and flag-volume-does-repeat claims have no correlations reported.
- The 32-team ranking table and any on-screen referee tables are not in the captions.
- No link to code; data sources are named in the description.
- Fines matching method (how a fine is matched to a flag) not described.
- No out-of-sample check of the Cardinals/Colts/Bills results.

### Short verbatim quotes
- "The only thing a crew really keeps from one season to the next is how many flags it throws." (5:39)
- "On about 1,200 physical plays, the league find a guy, and about 600 of them never got flagged." (6:05)
- "about 87% of it is the Colts just not getting flagged" (11:12)
- "a gap that size for anybody shows up about one time in 75" (13:59)

---

## 6. The NFL's Fourth-Down Revolution

- **Channel:** NFL Films (1.4M subscribers)
- **URL:** https://www.youtube.com/watch?v=d60F61zVOJY (found via YouTube search)
- **Upload date / views / length:** Sep 18, 2026; 257,129 views; 1,112 likes; 7:24
- **Transcript status:** read fully (0:00 to 7:23)
- **Chapters:** 0:00 Rise of fourth down plays; 1:19 Shifting team mindsets; 2:52 The art of decision-making; 4:33 The impact of aggressive play; 6:34 Commitment to the aggressive
- **Note:** this is an interview/highlight feature, not an analysis. Speakers are not identified in the captions (names would be on-screen graphics), so quotes below cannot be attributed to a named coach from the transcript alone.

### Question asked
Why are NFL teams going for it on fourth down so much more often, and how do coaches actually make the decision?

### Data and method
- No model or dataset. Two league-level numbers are stated by the narrator; the rest is coach testimony and game clips.
- People named in speech: Joe Philbin (caption "Joe Filman") as Dolphins head coach, Dan Campbell as tight ends coach, Zac Taylor (caption "Zack Taylor") as quarterbacks coach on the 2014-15 Dolphins staff; Doug Pederson (caption "Doug Peterson"); Nick Foles (caption "Nick Fos"); Matt LaFleur (caption "Matt Lur"); Sean Payton (caption "Shawn Payton"); Jordan Love, Tucker Kraft (caption "Tucker Craft"), Stafford, Puka; defenders Micah Parsons (caption "Michael Parsons") and Aaron Donald as examples.

### Findings with numbers
- 886 fourth-down conversion attempts "last season" (0:05 to 0:33). Coaches' guesses ranged from 250 to "6 to 700"; one said "558".
- Going for it on fourth down is up 86% compared with 10 years ago (1:19).
- An unnamed coach says the Miami Dolphins did a large offseason analytics study between the 2014 and 2015 seasons concluding teams should go for it more; it was not implemented because the head coach was let go after four games (1:25 to 1:59).
- The Eagles' Super Bowl run (fourth and goal from the one, the "Philly Special") is credited by speakers as the first visible success that spread the practice (1:59 to 2:50).
- Decision inputs coaches list: game complexion, score, how the defense and offense are playing, trust in the kicker, flow, time left, one-possession lead vs trailing (3:09 to 3:44).
- One coach's rule: "once we get on the plus side of the 50, my mind kind of just goes to four down mode" (6:42).
- Examples: Packers fourth and two, a field goal would have tied, opponent Arizona would take over with under two and a half minutes; converted (Love to Kraft), Packers won 27-23 (4:39 to 5:26). Broncos fourth and one at the Patriots 14 in the AFC Championship: incompletion; the speaker frames it as "you go up 10" with a field goal and then losing by three (5:34 to 6:12). NFC Championship: Rams fourth and four, down four, incomplete vs Seattle; coach says he would go for it again (6:12 to 6:30).

### Caveats the video itself states
- Coaches say the model "is not perfect", "it's a tool", and does not say whether you will convert (3:58 to 4:21).
- Outcome bias is acknowledged: convert and you are a genius, fail and you are an idiot (4:21); one coach says the staff tries not to let the outcome dictate the evaluation (6:12).
- "Momentum" is asserted as real by a coach (5:06) with no evidence offered.

### What is NOT shown (from the transcript)
- No source or definition for the 886 or 86% figures (regular season only? includes fakes/penalties?).
- No win probability or EPA numbers, no conversion rates, no team or coach rankings.
- No test of the momentum claim.
- Speaker identities are not in the captions.

### Short verbatim quotes
- "Compared to 10 years ago, going for it on fourth down has increased 86%." (1:19)
- "The overriding philosophy is I go for it if I think we're going to get it." (3:58)
- "the chart doesn't have to block uh Michael Parsons" (3:44)
- "not let the outcome dictate whether we would or wouldn't" (6:12)

---

## 7. Ep82 "Should NFL Teams Be More Aggressive on Fourth Down?" With David Romer

- **Channel:** Stanford Graduate School of Business (2.82M subscribers); All Else Equal podcast, hosts Jonathan Berk (Stanford GSB) and Jules van Binsbergen (Wharton / Lauder Institute)
- **URL:** https://www.youtube.com/watch?v=oaK-LTlADEY (found via YouTube search)
- **Upload date / views / length:** Nov 6, 2026; 4,319 views; 37:13
- **Transcript status:** read fully; the single get_page_text call returned the whole transcript (0:00 to 36:59), no cut-off. Captions here are speaker-labelled and look human-edited rather than auto-generated.
- **Chapters:** Introduction to the topic; Basic rules of football (2:22); The fourth down dilemma (5:50); Strategies and incentives (9:40); Meet guest David Romer (12:59); The impact of research (16:53); Rationality vs agency issues (21:36); Lessons from football (25:55); Closing thoughts (29:58)

### Question asked
If competition forces firms to optimize, why did NFL teams persistently fail to go for it on fourth down when the analysis said they should, and what finally changed?

### Data and method
- Discussion of Romer's fourth-down paper (title, journal and year are not stated in the transcript beyond "early 2000s"; editor was Steven Levitt, 32:36). No new data analysis in the episode.
- Method as described by Berk (5:50 to 6:51): dynamic programming to compute the expected scoring value of having the ball at any yard line and down; value is negative near your own goal line (safety risk and opponent field position). Romer also mentions "dynamic programming and linear algebra" (33:00).
- Earlier work cited by Romer: a 1971 paper in Operations Research, one author a backup NFL quarterback, covered in Sports Illustrated (he thinks) (16:01 to 16:16). Authors not named.
- Romer says he has not redone the analysis with modern data (18:53, 25:47).

### Findings with numbers
(All figures are conversational approximations by the speakers.)
- Berk's fourth-and-goal-from-the-one illustration: field goal is three "for sure"; touchdown worth seven with conversion probability "about 40% or something", so roughly equal in expected points; the tiebreak is field position, since a failure leaves the opponent at its own one, which has negative value for them (8:37 to 9:36). Berk then says "even today, we don't see teams do that" (9:36) [ambiguous in context whether "that" means going for it].
- Berk's midfield illustration: at the 50, too far for a field goal, a punt nets about 20 or 30 yards, so going for it dominates; teams now generally go for it there (10:30 to 11:17).
- Romer's surprise from the data: fourth and one converts at "70% or 75 or 65, but it's not 90", and fourth and ten is "30% or something", not 5% (17:35 to 18:00). Implication: in the part of the field where neither punt nor field goal is attractive, going for it is right even with substantial yards to go (18:00).
- Romer: fourth-down go rates were essentially flat from before his paper until 2017; a discrete step up starting in 2018 and a rising trend since (20:09 to 20:29).
- Precipitating event per Romer: Super Bowl 52, February 2018, Eagles vs Patriots. Eagles went for it twice in unconventional spots: the trick play on fourth and goal, and a fourth and one deep in their own territory while trailing, which led to the touchdown that was the margin of victory (20:29 to 21:04).
- Three causes Romer offers for the change: competition eventually working; the general rise of analytics staffs; the Super Bowl 52 event (19:34 to 20:29).
- Romer's impression (not re-analysed): teams are still not aggressive enough, "particularly in their own territory" (18:53 to 19:07). He does not know whether teams have captured most of the available gain (deadweight-loss triangle analogy, 25:24 to 25:47).
- Reception: within months every team knew of the paper via journalists; behaviour did not change (14:13 to 14:53). The furthest any team went was finding something the paper left out and then dismissing it, without asking which direction the omission cuts or whether it could overturn the result (35:28 to 36:11).
- Explanations debated: agency/reputation (coach is pilloried for unconventional failures) vs behavioural bias/technology. Romer is sceptical of the agency story (coaches of bad teams or ambitious college coaches would gain from defying convention; pressure to change did not come from owners) and leans to strong behavioural biases toward caution, but says repeatedly that he does not know (22:47 to 24:47).
- Cross-sport parallels Romer cites for conservative bias: slow adoption of three-point shots, baseball playing for one run, hockey pulling the goalie too late (28:19 to 28:43).
- Anecdote: a high-school coach (Korean War veteran) who went for it often and rejected parents' "life-or-death" framing (28:57 to 29:23).

### Caveats the video itself states
- Romer: "my best answer is I don't know" on agency vs behavioural explanations (22:41); "this is not strong evidence" (23:11).
- Romer has not redone the analysis; someone would need to redo the paper with modern data to say how far teams are from optimal (25:47).
- Both Berk and Romer say they do not watch much football (24:47 to 25:02). Romer adds a personal reservation about the sport's cost to players (25:02 to 25:16).
- Hosts' rule explanations contain simplifications (e.g. field goal "impossible" at about 45 yards, 5:16; opponents "always take over at the 20", 7:38); these are the hosts' statements, not findings.

### What is NOT shown (from the transcript)
- No actual tables, break-even yardages, sample sizes, seasons or data source from the paper.
- No statement of the paper's objective function (points vs win probability) or of its restriction to particular game states; none of that is discussed.
- The 2018 step-up is described verbally with no numbers.
- No per-team or per-coach evidence; no test of the agency vs behavioural hypotheses.
- The 1971 paper is not named.

### Short verbatim quotes
- "I had two predictions. Both of them were completely wrong." (14:10, Romer)
- "Starting in 2018, there's a discrete step up, and there's been a trend ever since then" (20:22, Romer)
- "my impression is they're not as aggressive as they should be, particularly in their own territory" (18:53, Romer)
- "Oh, the analysis isn't perfect, therefore we can ignore it." (36:01, Romer describing teams' reaction)

---

## 8. I Checked 6,340 NFL 4th Downs to Find the Most Cautious Coach

- **Channel:** Flare Shorts (13.1K subscribers). Description discloses an AI-generated voiceover.
- **URL:** https://www.youtube.com/watch?v=FGRj-n9YSXE (found via YouTube search)
- **Upload date / views / length:** Nov 1, 2026; 859 views; 1 like; 8:37 per the search listing (an ad occupied the player timer; last caption at 8:28)
- **Transcript status:** read fully (0:00 to 8:28)
- **Chapters:** 0:00 The fourth-down question; 0:55 How aggressiveness is measured; 1:45 The league-wide picture; 3:01 The most aggressive coaches; 4:06 The coach who flipped; 5:21 The most cautious coaches; 6:55 Aggressive isn't the same as right; 7:50 Reality check; 8:20 Next video

### Question asked
Across the 2024 and 2025 seasons, which NFL head coach is the most cautious on fourth down, and is being aggressive the same as deciding well?

### Data and method
- **Not original data.** Uses FTN's published Aggressiveness Index tables for 2024 and 2025 (Aaron Schatz, calculated by Jim Armstrong; caption "Aaron Shots"). The creator combined the two seasons and computed totals, rates and two-season averages himself (0:57 to 1:10; description).
- **Sample:** 6,340 qualifying fourth downs over two seasons (title/description; caption reads "6,344th downs" [caption unclear]). Go-for-it count is garbled: "1,05 of them. That's 15.9%" (0:20) [caption unclear; number not recoverable from captions].
- **Index definition as explained (1:10 to 1:42):** take the fourth downs a coach faced, drop situations excluded because of score and clock (exclusion rules not specified), and compare how often the coach went for it with how often coaches normally go in that spot. Baseline = league behaviour since 2018. 1.0 = average; 2.0 = twice as aggressive; captions say "Five means half" (i.e. 0.5 [caption unclear]). It compares coaches with other coaches, not with an optimal model.
- **"Typical coach would have gone X times"** = expected attempts implied by the index on that coach's qualifying chances.
- **Other sources:** Boston Globe (Oct 29, 2026) for this season's league totals; a 2024 Annals of Applied Statistics study led by Nathan Sandholtz (85 coaches, 2014 to 2022), read via a StudyFinds summary per the description; a Pro Football Focus team-level article (2021 to 2025) on how often calls matched PFF's decision model (model unexplained).
- **Persistence check:** correlation of 2024 vs 2025 index for the 25 coaches present in both seasons.

### Findings with numbers
**League picture (1:45 to 3:00)**
- Go rate on qualifying fourth downs: 14.8% in 2024 (caption "14.8% 8%"), 16.9% in 2025 (creator's sums of FTN tables).
- All fourth-down attempts: 766 in 2024, converted 56.8%, "an all-time high at the time" (1:58).
- This season through week 7: 304 attempts, second most ever at that point; same point in 2016: 187 (2:06 to 2:22).
- Conversion rate this season 53.6%, "down from 59% last year" (2:22). (Note: 59% is for 2025; 56.8% was 2024.)
- Sandholtz et al.: early on coaches were overwhelmingly conservative, especially in their own territory; by 2021 and 2022 more aggressive but still short of what win probability models wanted (2:37 to 2:54).
- League got about 14% more aggressive from 2024 to 2025 "by my math" (4:25).

**Most aggressive (3:00 to 4:06)**
- Dan Campbell: 1.95 (2024), 2.02 (2025); 56 attempts on 195 qualifying chances = 28.7%; went on all nine qualifying fourth-and-ones in 2025.
- Sean McDermott: 1.65 over both years; 48 attempts on 186 chances.
- Nick Sirianni: 1.35; 46 attempts on 229 chances. The Sandholtz study ranked him sixth of 85 for risk-taking.
- Ben Johnson: as of October 30 has gone for it more than any coach this season; third in the NFL last year at 1.78 with Chicago, though the Bears converted only 15 of 29.

**Year-to-year changes (4:06 to 5:21)**
- Andy Reid: 0.84 in 2024 (29th of 33) to 1.41 in 2025 (7th of 34), +0.57, the biggest jump. Kansas City converted 24 of 32 that year.
- Jonathan Gannon: 0.85 to 1.39. Sean McVay: 0.99 to 1.32.
- Zac Taylor: 1.32 to 0.82 (caption "0.82, 82"), the biggest drop.
- 25 coaches appear in both seasons; 2024 vs 2025 correlation 0.66; 18 of 25 stayed on the same side of league average; seven crossed (5:05 to 5:14).

**Most cautious, both seasons (5:21 to 6:55)**
- Excluded: Brian Callahan (part of 2025 only, 36 qualifying chances).
- 4. Mike Tomlin: 0.88; 29 attempts on 220 chances; typical coach about 33.
- 3. Mike Macdonald: 0.83; 26 on 204; about five fewer than typical; 0.94 in 2024, 0.75 in 2025.
- 2. DeMeco Ryans (caption "Demo Ryan"): 0.80; 26 on 247 chances (most qualifying chances of any coach); about six and a half fewer than typical.
- 1. Jim Harbaugh: 0.70 then 0.71; 19 attempts on 225 chances = 8.4%; typical coach about 27.
- Framing: the most cautious coach went about eight fewer times than typical over two seasons (four decisions a year), while Campbell went roughly 28 more times; the outliers are on the aggressive side (6:37 to 6:50).

**Aggressive vs right (6:55 to 7:50), PFF 2021 to 2025**
- Most attempts: Lions 198. Fewest: Seahawks 90.
- Agreement with PFF's model: Buffalo 81.2% (highest); Seattle 79.8% (second); Chicago 73.9% (fourth lowest) despite 170 attempts (fifth most); Dallas 71.6% (lowest).

### Caveats the video itself states
- The index compares coaches with coaches, not with an optimal model or with wins; low does not mean wrong (1:34, 7:50).
- "I haven't shown that aggression wins games" (7:58).
- Only two seasons, and the league is drifting upward, so some flips are the tide (8:04).
- PFF does not explain its model; the academic study left out injuries, weather and talent; a coach whose offense cannot get two yards may be right to punt (8:04 to 8:11).
- The explanation for the falling conversion rate (easy calls already taken) is "a guess, not a proven fact" (2:30).

### What is NOT shown (from the transcript)
- No win probability or EPA cost of any coach's decisions; no measure of decision quality per coach (PFF figures are team-level over a different period).
- No confidence intervals: with roughly 100 qualifying chances per coach-season and gaps of about four attempts a year, no test of whether Harbaugh differs from average.
- The 0.66 correlation is a single pair of seasons, n = 25, no interval.
- FTN's exact situation filters are not given; attempts counted do not distinguish converted vs failed for the index.
- Coach and team are conflated in places (PFF team data vs coach index; coaches who changed teams).
- No code or data file.

### Short verbatim quotes
- "This compares a coach to other coaches, not to a perfect math model." (1:34)
- "their 2024 and 2025 numbers have a correlation of 0.66" (5:05)
- "Going for it a lot and going for it at the right time are two different skills." (7:45)
- "I haven't shown that aggression wins games." (7:58)

---

## 9. I Built a Model to Find the NFL's Most Overrated Coach

- **Channel:** Flare Shorts (13.1K subscribers). Description discloses an AI-generated voiceover.
- **URL:** https://www.youtube.com/watch?v=ch3pd2E62io (found via YouTube search)
- **Upload date / views / length:** Nov 3, 2026; 471 views; 2 likes; 7:49
- **Transcript status:** read fully (0:00 to 7:39 last caption)
- **Chapters:** 0:00 Hook; 0:23 Setup; 0:57 V0.1 Win Percentage; 1:26 V0.2 Points Against the Spread; 2:07 V0.3 Reputation vs the Market; 2:48 V0.4 The Quarterback Test; 3:41 V0.5 Underdog vs Favorite; 4:17 V0.6 Shrinking the Noise; 5:18 V0.7 Does It Repeat?; 5:43 V0.8 The Chance Test; 6:31 The Verdict; 7:14 What the Model Can't See; 7:34 Close

### Question asked
Measured against the Vegas point spread rather than win-loss record, which NFL head coaches actually beat expectations, and can any coach be shown to be over- or underrated beyond chance?

### Data and method
- **Model name:** BENCH = "Beating Expectations, Not Credit or Hype".
- **Data:** public nflverse game data ("nfldata") with listed point spread; NFL regular season 2002 to 2025; 6,223 games with a spread; 39 head coaches with at least 100 regular-season games since 2002 (0:00 to 0:57; description). (One video frame I saw during loading displayed "5,849 games, regular season 2002-2025", which differs from the spoken and written 6,223; I could not tell from the transcript which is used where.)
- **Metric:** average points by which the coach's team beats the closing/listed spread per game (cover margin).
- **Versions:**
  - V0.1 win percentage.
  - V0.2 mean points against the spread (ATS margin).
  - V0.3 rank by win percentage vs rank by ATS margin (reputation vs market).
  - V0.4 split by star quarterback (with vs without).
  - V0.5 split by underdog vs favourite.
  - V0.6 shrinkage toward league average, stronger for small samples; single-game spread error about 13 points; estimated true between-coach spread about 0.44 points per game (4:17 to 4:39). Shrinkage formula not given (reads like empirical Bayes).
  - V0.7 repeatability: split-half correlation of careers, and odd vs even seasons.
  - V0.8 permutation test: coach labels shuffled across games 3,000 times; compare the spread of coach means in fake vs real leagues; then per-coach tests with a multiple-comparison correction for 39 coaches (correction method not named).

### Findings with numbers
**V0.1 win percentage (0:57):** Tony Dungy 75.9%; Bill Belichick 70.4; Sean McDermott 66.2; bottom: Ken Whisenhunt 39.1, Herm Edwards 39.3.

**V0.2 points against the spread per game (1:26 to 2:07):** Belichick +2.93 (355 games); McDermott +2.44; John Harbaugh +2.27; Dungy +2.21; Pete Carroll and Kyle Shanahan +1.0 each. Bottom: Whisenhunt -2.38; Kevin Stefanski -1.97; Mike Shanahan -1.28.

**V0.3 rank by wins vs rank ATS (2:07 to 2:48):** Mike Tomlin 7th to 16th; Matt LaFleur 4th to 13th; Andy Reid 5th to 10th; Gary Kubiak 20th to 33rd; Mike Shanahan 25th to 37th. Other direction: Lovie Smith 33rd to 18th; Mike Vrabel 13th to 7th; John Harbaugh 10th to 3rd.

**V0.4 quarterback split (2:48 to 3:41):**
- Belichick with Brady: 267 games, +3.64; without: 88 games, +0.75 (drop "not statistically proven" at 88 games).
- John Harbaugh with Lamar Jackson +3.36; without +1.64.
- Andy Reid with Mahomes +1.12; without +1.31.
- Sean Payton with Brees +1.87; without +1.77.
- Mike McCarthy with Rodgers +1.57; without +1.21.

**V0.5 underdog vs favourite (3:41 to 4:17):** Belichick as underdog +4.67 over 77 games, as favourite +2.44. Tomlin as underdog +2.23 over 109 games, as favourite +0.21 over 200 games.

**V0.6 shrunk estimates, points per game (4:17 to 5:18):** Belichick +0.83; Harbaugh +0.56; Payton +0.45; Reid +0.38; McDermott +0.34. Bottom: Whisenhunt -0.30; Mike Shanahan -0.21; Stefanski -0.20 (raw -1.97). In wins: Belichick raw about 1.3 wins a season above Vegas expectation, shrunk about 4/10 of a win.

**V0.7 repeatability (5:18 to 5:43):** first-half vs second-half of career correlation 0.42; odd vs even seasons correlation 0.09.

**V0.8 chance test (5:43 to 6:31):** spread between coaches in the shuffled league 1.06 points vs 1.15 in the real league; p-value 0.21, so as a group coaches are "not clearly different from random". After correcting for 39 comparisons only Belichick survives; Harbaugh is at 2.9 standard errors and does not clear the bar; Payton, McDermott, Reid, Dungy do not; no negative coach (Whisenhunt, Stefanski, Shanahan, Gruden) is provably below market.

**Verdict (6:31 to 7:14):** "most overrated" among big reputations: Tomlin and LaFleur (not proven). Most underrated: Harbaugh (not proven). A typical coach is worth about half a point per game above or below average; home field "was worth almost two".

### Caveats the video itself states
- The spread already includes the market's opinion of the coach, so "underrated" means underrated by the market, not by fans (7:16).
- Cannot separate play-calling from staff hiring and roster building (7:23).
- 100 games is only a few seasons; one injured quarterback moves the number (7:23 to 7:31).
- Coach and quarterback effects cannot be cleanly separated for Belichick and Harbaugh (3:35).
- Tomlin's 109 underdog games is a small sample (4:08).
- The repeat signal is "there, but fragile" (5:30).

### What is NOT shown (from the transcript)
- Shrinkage formula, variance estimates and the specific multiple-comparison correction are not stated.
- No explanation of why split-half gives 0.42 but odd/even gives 0.09 (the first is confounded with era/tenure blocks; the video only says "fragile").
- Which "Shanahan" in the unprovable-negatives list at 6:20 is not specified (Mike Shanahan from context).
- No playoff games; no decision-level evaluation (fourth downs, timeouts, clock), only game-level cover margin.
- No adjustment for the spread already pricing the coach (acknowledged), so a consistently good coach should regress toward zero ATS by construction.
- No full 39-coach table in the captions; no code link.

### Short verbatim quotes
- "out of 39, only one coach can be proven to beat the market" (0:11)
- "A typical coach sits about 0.44 points per game above or below average." (4:30)
- "The correlation was 0.42 ... when I split by odd and even seasons, it fell to 0.09." (5:23 to 5:30, two adjacent sentences joined by ellipsis)
- "The p value is 0.21. As a group, coaches are not clearly different from random." (5:51)

---

## 10. Cross-video notes

- **Shared taxonomy:** videos 1, 3 and 4 all split penalties into team-controlled ("objective"/"earned": false start, offside, illegal motion, 12 men) vs official-controlled ("subjective"/"judgment": holding both ways, DPI/OPI, roughing the passer, illegal contact; video 4 also moves ineligible man downfield there). Videos 1 and 3 both report that the between-team quality difference lives almost entirely in the pre-snap/procedural bucket (video 1: 2.35 vs 2.04 false starts per game; Belichick-era Patriots about half a flag a game fewer objective penalties; video 3: half of the Raiders' extra flag a game is false starts and offside).
- **What repeats and what does not, as claimed:** crew flag volume repeats year to year (videos 3, 4), but crew identity explains under 1% of a team's season flag count and improves flag prediction by about 1% (video 3). Team subjective-penalty counts do not carry over within a season (video 1, no number). Referee home/favourite lean and replay overturns do not repeat (video 4). Coach fourth-down aggressiveness correlates 0.66 across two seasons (video 7). Coach ATS margin: 0.42 split-half but 0.09 odd/even; permutation p = 0.21 (video 8).
- **Agreement across independent creators:** Chiefs are unremarkable or unlucky on penalties in the regular season and favoured in the playoffs mainly via coverage/pass-interference calls (videos 1, 3, 4); Bills are the least lucky/most hurt (videos 1, 3); playoffs see far fewer subjective/holding calls (video 1: 27% fewer subjective; video 4: about a third fewer holds vs about 10% fewer false starts).
- **Same event, three sources:** the Rams-Broncos DPI on Josh Wallace appears in videos 1, 3 and 4 (video 1 gives the win-probability swing: 12% to 69%).

