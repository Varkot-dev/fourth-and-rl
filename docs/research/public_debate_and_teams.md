# Public debate and team effects

Researched 2026-11-17. Part A is web research; Part B is our data.

Evidence labels:

- **[search]** Taken from a web-search result summary. The page itself could not be opened: this machine's network allowlist blocked every news host tried (sports.yahoo.com, si.com, footballperspective.com). Re-check against the page before quoting publicly.
- **[unverified]** Recalled from background knowledge, not checked.
- **[ours]** Produced by `scripts/research/team_aggressiveness_stats.py`, saved in `results/research/team_aggressiveness_stats.json`.

Twitter/X and YouTube were not reachable, so the explainer videos and threads themselves were not seen. No direct quotes are reproduced below, because none could be checked at the source.

## Part A. What people argue about

### A1. Detroit under Dan Campbell

**What they do.** Reporting describes Detroit as the most aggressive fourth-down team since Campbell arrived in 2021: a league-high 147 attempts with a 55.8% conversion rate as of January 2025 [search: https://www.startribune.com/detroit-lions-minnesota-vikings-dan-campbell-fourth-down-ben-johnson/601201793], and the highest go rate in the league, 31.3%, as of December 2024 [search: https://abc30.com/post/dan-campbells-aggressiveness-has-detroit-lions/15727402/]. Our own counts are in B5; the numbers are not blended.

**NFC Championship at San Francisco, 28 January 2024 (lost 34-31 after leading by 17).** Two decisions carry the argument [search: https://www.cbsnews.com/amp/detroit/news/detroit-lions-dan-campbell-san-francisco-49ers-loss-superbowl-nfc-championship-game-football and https://www.theringer.com/nfl/2024/1/29/24054043/nfl-conference-championship-hot-read-detroit-lions-kansas-city-chiefs]:

- Third quarter, leading by 14, fourth and 2: went for it instead of a 46-yard field goal; the pass was dropped. Next Gen Stats reportedly had it as a lean to go, 86.8% win probability going versus 85.8% kicking. Another reported figure is 90.5% versus 90.3%.
- Fourth quarter, trailing by 3, fourth and 3: went for it instead of a tying kick; incomplete. Next Gen Stats reportedly had 32.7% going versus 30.2% kicking. ESPN's model reportedly rated both decisions as nearly neutral.

The two sides: critics (television analysts, an anonymous coach quoted by Bleacher Report, many fans) said take the points on the road with a lead, and that the failures swung momentum [search: https://bleacherreport.com/articles/10107026-nfl-coach-rips-lions-hc-dan-campbells-ridiculous-decision-not-to-kick-fg-vs-49ers, https://awfulannouncing.com/nfl/booger-mcfarland-espn-dan-campbell-outcoached-lions.html]. Defenders said the calls were consistent with what got Detroit there, the models leaned go, and a dropped pass is not a decision error [search: https://www.nbcsports.com/fantasy/football/news/dan-campbell-did-nothing-wrong, https://www.theringer.com link above]. Campbell said he did not regret them and understood the scrutiny [search: https://www.nbcsports.com/nfl/profootballtalk/rumor-mill/news/dan-campbell-i-understand-the-scrutiny-ill-get-for-fourth-down-decisions].

What matters for this project: the model margins were 1 to 2.5 points of win probability. That is exactly the "too close to call" zone of the humility paper, yet the public argued as if one side was plainly right.

**Green Bay, 5 December 2024 (won 34-31).** Tied with 43 seconds left, fourth and 1 at the Packers 21: went for it instead of kicking the go-ahead field goal, converted, then kicked the winner as time expired. Baldwin's bot reportedly called it close with a slight lean to kick; Next Gen Stats reportedly called it a toss-up and measured the distance as nearer 2 yards [search: https://www.si.com/nfl/lions/dan-campbell-puzzling-fourth-down-calls-win-packers, https://global.espn.com/nfl/story/_/id/42808088/lions-all-campbell-gamble-walk-win-vs-packers]. The argument for going was denying Green Bay a final possession, which a one-step model handles poorly; it also shows the integer-distance problem (Lopez) live.

**Efficiency versus volume.** One 2025 report says Detroit ranked 31st in fourth-down decision EPA at that point, and an ESPN coach ranking put Detroit last in actual minus expected conversions for one period [search: https://heavy.com/sports/detroit-lions/dan-campbell-fourth-down-ranking-espn, https://www.si.com/nfl/lions/news/dan-campbell-examines-why-more-teams-not-aggressive]. So one live argument is "do they convert enough to justify it".

### A2. Other recurring arguments

- **Sean Payton, AFC Championship, 25 January 2026 (Denver lost 10-7).** Leading 7-0, fourth and 1 at the New England 14, went for it and failed. Baldwin's bot reportedly preferred the kick, 66% versus 64%; a second search summary says ESPN's model preferred going, 69.3% versus 67.2%. Critics said he leaned on analytics too much; Payton later said the play call bothered him more than the decision [search: https://www.si.com/nfl/fans-crushed-sean-payton-costly-fourth-down-decision-broncos-loss, https://www.nfl.com/news/broncos-hc-sean-payton-laments-first-half-in-afc-title-game-loss, https://nesn.com/2026/01/did-sean-payton-mess-up-with-controversial-decision-vs-patriots]. Two public models on opposite sides by 2 points each is the best advertisement for an uncertainty band.
- **Brandon Staley, Chargers, 2021 to 2023.** Repeated failed fourth downs against Kansas City and a late go deep in his own territory against Cleveland. Analytics writers mostly backed him; players and television critics did not [search: https://nbcsports.com/nfl/profootballtalk/rumor-mill/news/brandon-staley-real-football-people-understand-what-im-doing, https://www.si.com/nfl/2021/10/21/nfl-coaches-fourth-down-decisions-explain-trust].
- **John Harbaugh's two-point tries for the win, 2021.** Twice in three weeks (at Pittsburgh, then against Green Bay) Baltimore went for two in the final minute instead of kicking to tie, and failed both times. Sides: play for the win with a depleted roster, or take overtime [search: https://global.espn.com/nfl/story/_/id/32906780/john-harbaugh-says-2-point-conversion-call-mostly-gut-decision-ravens-fall-short-again].
- **"Are teams now going too often?"** Jason Garrett and Chris Simms argued in 2026 that the movement overshoots and ignores the psychological cost of failure [search: https://www.profootballnetwork.com/former-cowboys-head-coach-ex-nfl-qb-rip-analytics-movement-urging-bold-4th-down-decisions/].
- **Overtime under the 2025 rules** and going for two when down 8 remain open arguments [search: https://www.footballzebras.com/2025/04/everything-you-need-to-know-about-the-new-overtime-rules/; down-8 debate unverified here, see `landscape.md`].

### A3. Run versus pass

- **The claim.** Passing gains more per play and more expected points, so teams run too much. Kovash and Levitt (NBER working paper 15347) estimated teams give up about one point per game by not passing more, and found negative serial correlation in play calls [search: https://www.nber.org/papers/w15347]. FiveThirtyEight ran the same argument for first-down runs [search: https://fivethirtyeight.com/features/you-called-a-run-on-first-down-youre-already-screwed]. Ben Baldwin is the name most attached to early-down pass-rate arguments [search, secondhand].
- **"You must run to set up play action."** Reported studies find play action works regardless of run volume or success [search: https://www.thescore.com/nfl/news/2242190].
- **The game-theory reply.** In a mixed-strategy equilibrium the marginal run and marginal pass should be equally valuable, not the average ones; averages can differ because of risk (sacks, interceptions) and because defenses choose. Defenses did adjust: more two-deep shells and lighter boxes, and league run EPA per play reportedly rose from about -0.08 in 2018 to about -0.03 in 2022 [search: https://www.theringer.com/2023/06/05/nfl/five-games-2022-season-nfl-scheme-wars].
- **Situational running.** Running is competitive in short yardage and near the goal line, and when protecting a lead and the clock [search: https://www.footballperspective.com/why-do-teams-run-the-ball-part-iii/].

This maps to burning question 2: average versus marginal value.

### A4. Criticisms of bots and win-probability graphics

1. **"The model does not know our team."** Coaches cite their own line, quarterback, injuries and matchup. Bill Parcells is reported as saying he went for it because of his players, not the percentages [search: https://www.pressreader.com/usa/the-arizona-republic/20211212/282059100301095]. Early bots used little beyond down, distance and field position [search: https://www.thescore.com/nfl/news/2250074]. Measured in Part B.
2. **"Analytics ignores momentum and feel."** Payton himself cited momentum as a reason to go; broadcasters cite it as a reason not to [search: https://www.foxnews.com/sports/nfl-legend-criticizes-coaches-fourth-down-aggressiveness-after-sean-paytons-costly-decision-broncos-loss.print, https://awfulannouncing.com/college-football/gus-johnson-analytics-joel-klatt-conversion.html]. Testable in play-by-play data; a good later question.
3. **"It said 99% and they lost."** After Super Bowl LI, ESPN's figure for Atlanta was reported as 98.9% and Pro-Football-Reference's as 99.9%; one academic argued such models were overconfident, others that this is what a calibrated model does occasionally [search: https://thescore.com/nfl/news/1226023, https://www.numberfire.com/nfl/news/12861/the-win-probability-changes-from-super-bowl-li-were-mind-blowing]. Answered by M3b.
4. **Results-based judging and asymmetric blame.** A failed go is blamed on the coach; a punt that loses slowly is not [search: https://defector.com/footballs-fight-over-analytics-doesnt-have-to-be-so-exhausting].
5. **False precision.** Public graphics show a recommendation and a margin with no interval.

## Part B. Team effects in our data [ours]

All numbers below are from the results JSON. Regular season unless stated. EPA and EP are nflfastR's columns, not our models.

### B1. Go rates

League go rate on all fourth downs rose from 12.5% (481 of 3,840) in 2010 to 23.3% (882 of 3,793) in 2025. In neutral situations (fourth and 1 to 3, at or beyond the own 40, score within 8, quarters 1 to 3) it rose from 28.4% (98 of 345) to 66.3% (250 of 377).

Detroit's rank among 32 teams:

| Season | Go rate, all (rank) | Neutral go rate (rank) | Neutral decisions |
|---|---|---|---|
| 2021 | 30.8% (2) | 73.3% (4) | 15 |
| 2022 | 31.6% (1) | 66.7% (7) | 18 |
| 2023 | 33.9% (1) | 84.6% (3) | 13 |
| 2024 | 32.0% (1) | 78.6% (3) | 14 |
| 2025 | 26.7% (7) | 84.6% (4) | 13 |

Neutral ranks within one season rest on 13 to 18 decisions per team and are mostly noise. Pooled over 2021 to 2025 Detroit is first on both measures: 31.0% overall (182 of 587) and 76.7% neutral (56 of 73).

Most aggressive 2021 to 2025 by neutral rate: Detroit 76.7%, Green Bay 75.5% (40 of 53), Philadelphia 73.1% (38 of 52), Cleveland 73.0% (46 of 63). Least: Miami 43.8% (21 of 48), New England 44.1% (26 of 59), Pittsburgh 44.3% (31 of 70), New Orleans 44.4% (24 of 54), Houston 45.8% (33 of 72). The overall go rate flatters losing teams, who go because they trail: Carolina (25.5%) and Arizona (25.2%) rank second and third overall but 11th and 10th in neutral spots. Seattle is last overall at 13.6%.

The highest team-season go rate since 2010 is Detroit 2023 (33.9%, 40 of 118); the lowest is Arizona 2014 (3.1%, 4 of 128).

### B2. Does offense quality change conversion?

Sample: third-down runs and passes and fourth-down go plays with 1 to 3 to go, 2011 to 2025 including playoffs (35,171 plays: 29,434 third downs, 5,737 fourth downs). Offense strength for a game is early-down EPA per play from that season's earlier games only, plus 300 pseudo-plays at half of last season's value. Teams are ranked 1 to 32 on that number entering each week.

Top-5 offense minus bottom-5 offense, same down and distance (six cells, weighted), with a game-level bootstrap (2,000 resamples):

| Sample | Top 5 | Bottom 5 | Gap | 95% interval | Plays (top / bottom) |
|---|---|---|---|---|---|
| Third and fourth | 62.7% | 57.6% | 4.8 points | 3.0 to 6.6 | 6,001 / 5,051 |
| Third only | 62.5% | 57.1% | 4.9 points | 3.0 to 6.9 | 5,055 / 4,186 |
| Fourth only | 63.7% | 59.9% | 3.9 points | -0.7 to 7.9 | 946 / 865 |

A logistic model (down, distance, offense strength) gives the same answer, 4.8 points (3.2 to 6.5). Adding the opposing defense's prior-games strength leaves it at 5.0 (3.4 to 6.6). Adding the pregame spread, which public bots already use, cuts the offense effect to 3.0 points (1.1 to 4.7). So about a third of what "our offense" adds is already in the spread.

Ranking offenses by full-season EPA instead (the leaky version, which uses the game being judged and later games) gives 5.4 points (3.7 to 7.3). The leak adds about 0.7 points: small, but in the direction that flatters a team adjustment.

**How much of the team-to-team spread is noise?** For each team-season, take (conversions minus expected conversions at the league rate for that season, down and distance) divided by plays. Binomial luck alone gives that number a large spread.

| Unit | Plays per unit (median) | Observed SD | SD expected from luck alone | Noise share of variance | SD of true effect | Shrinkage factor |
|---|---|---|---|---|---|---|
| Team-season, third/fourth and 1 to 3 | 69 | 6.3 points | 5.9 points | 88% (78 to 99) | 2.2 points | 0.12 |
| Team-season, fourth and 1 to 3 only | 10 | 16.9 | 16.4 | 94% (83 to 100) | 4.2 (0 to 7.5) | 0.07 |
| Team-season, all fourth-down goes | 17 | 11.6 | 12.0 | 100% (94 to 100) | 0 (0 to 3.0) | 0.00 |
| Team, 2021 to 2025 pooled, all goes | 120 | 4.4 | 4.3 | 97% (62 to 100) | 0.8 (0 to 3.4) | 0.03 |

The shrinkage factor is tau^2 / (tau^2 + sampling variance): the weight an empirical-Bayes estimate gives the team's own rate. A team-season's fourth-down conversion rate should be pulled 93% to 100% of the way back to the league rate. With true SD 2.2 points, the gap between the truly best five and worst five converters among 32 teams would be about 6.5 points, consistent with the 4.8 measured through offense strength.

### B3. What that implies for decisions

Break-even, in expected points, at yard line y (yards from the opponent's goal) with d to go:

    go if  p * V_success + (1 - p) * V_fail  >  V_alt
    p*  =  (V_alt - V_fail) / (V_success - V_fail)

with V_success = E(y - d), V_fail = -E(100 - y), V_alt the better of punting (minus the opponent's average EP after punts from that spot) and kicking (make rate times (3 - E(70)), plus miss rate times -E(100 - y - 7)), and E the nflfastR expected points for first and 10. The league curve fitted to fourth-down goes outside the last two minutes of a half is logit p(d) = 0.790 - 0.180 * d (n = 7,134), which falls 4.1 to 4.5 points per extra yard. A top-5 offense is p(d) + 2.4 points and a bottom-5 offense p(d) - 2.4.

| Spot | Alternative | p* at 2 to go | Break-even distance: bottom 5 / average / top 5 | Shift | Shift once spread is known |
|---|---|---|---|---|---|
| Own 35 | punt | 0.51 | 3.8 / 4.4 / 5.0 yards | 1.15 | 0.77 |
| Midfield | punt | 0.43 | 5.8 / 6.3 / 7.0 | 1.2 | 0.81 |
| Opponent 35 | field goal (68% make) | 0.41 | 6.4 / 7.0 / 7.7 | 1.35 | 0.72 |
| Opponent 20 | field goal (88% make) | 0.38 | 7.1 / 7.7 / 8.3 | 1.2 | 0.79 |

Rule of thumb: 4.8 points of conversion probability divided by about 4.3 points per yard is a bit over one yard. This is an expected-points, clock-free illustration, and its absolute break-even distances are more aggressive than a win-probability engine will give; only the shift is the point.

Verdict: "the model does not know our offense" is a real but small effect. Between the extremes of the league it moves the break-even by about one yard; for a typical team (half a standard deviation from average) by a fraction of a yard; and for a model that already has the spread, by about three quarters of a yard at the extremes. It can flip only decisions that were already close. It cannot rescue fourth and 8 or condemn fourth and 1.

### B4. Leakage warning

Ways a team-strength adjustment leaks:

1. **Same game.** Rating the offense with plays from the game being judged, including the fourth-down play itself and the drives its outcome produced.
2. **Later games in the season.** Season-to-date tables built after the fact include weeks that had not happened.
3. **Season totals for a week-3 decision.** The most common form: joining a season-level rating to every play.
4. **Fitted constants.** Choosing the prior weight or shrinkage using the seasons later used for testing.

Leak-free design: for game g of team t in season s, features use only games with a date strictly before g. Strength = (K * c * strength in season s-1 + sum over earlier games in s) / (K + plays in earlier games), with K and c fixed in a config file and tuned on training seasons only. This script used K = 300 plays and c = 0.5 without tuning; the measured year-to-year slope is 0.33 (480 team-season pairs), so c = 0.5 is on the generous side. This feature correlates 0.25 with the same game's early-down EPA per play (8,192 team-games), which is the honest ceiling on what it knows.

Tests: (a) invariance: scramble all data at or after a cutoff and assert features at or before it are unchanged. The script does this at four cutoffs and passes (maximum change below 1e-12; all later rows change, as they should). (b) A unit test that week-1 features equal the prior exactly. (c) Season-forward evaluation only.

### B5. Detroit, 2021 to 2025

| Season | Decisions | Go | Go rate (95% interval) | Converted | Conversion rate (95% interval) |
|---|---|---|---|---|---|
| 2021 | 133 | 41 | 30.8% (23.6 to 39.1) | 21 | 51.2% (36.5 to 65.8) |
| 2022 | 117 | 37 | 31.6% (23.9 to 40.5) | 20 | 54.1% (38.4 to 69.0) |
| 2023 | 118 | 40 | 33.9% (26.0 to 42.8) | 21 | 52.5% (37.5 to 67.1) |
| 2024 | 103 | 33 | 32.0% (23.8 to 41.6) | 22 | 66.7% (49.6 to 80.3) |
| 2025 | 116 | 31 | 26.7% (19.5 to 35.4) | 18 | 58.1% (40.8 to 73.6) |
| Total | 587 | 182 | 31.0% (27.4 to 34.9) | 102 | 56.0% (48.8 to 63.1) |

With playoffs: 609 decisions, 189 goes (31.0%), 107 converted (56.6%, 49.5 to 63.5). League conversion on all goes in the same regular seasons: 53.7% (2,129 of 3,967).

Detroit's conversion interval contains the league rate in every season and in total. The data cannot show that Detroit converts better than average on fourth down; it shows clearly that Detroit goes more. No win probability gained is computed here.

## Is a team adjustment worth building?

Yes, as a small, shrunken feature, not as a headline. Use prior-games-only offense and defense strength on top of the spread, expect about 3 points of conversion probability between the extremes, and report recommendations with and without it. Do not use a team's own fourth-down conversion record: at these sample sizes it is noise. The bigger public payoff is elsewhere: the most argued-about calls (San Francisco 2024, Green Bay 2024, Denver 2026) all had model margins of 1 to 2.5 points, which our uncertainty band and clock-aware solve address directly.
