# Landscape: what exists, what is criticized, where the gaps are

Researched 2026-11-17. Evidence labels used throughout:

- **[read]** I read the source itself (arXiv abstract page or PDF, or the raw file on GitHub).
- **[search]** Seen only in a web-search summary; the source itself was not opened. Treat as unverified.
- **[unverified]** Recalled from background knowledge; not checked.

Numbers below belong to their sources, not to this repo.

## 1. Existing public work

**nfl4th and the 4th down bot (Baldwin, nflverse).** One-step lookahead: for go, field goal and punt, average a first-down win-probability (WP) model over the outcome distribution and recommend the highest. The CRAN release (1.0.7, the one our reference script uses) is the old model. The development version on GitHub is a large rebuild [read: `raw.githubusercontent.com/nflverse/nfl4th/master/NEWS.md` and `README.md`]:
- a `bootstrap = TRUE` option: 200 refits of the WP and go models on histories resampled by game, the share of draws agreeing with the call, and an 80% interval on the gain;
- a WP model that uses both teams' timeouts, with explicit end-game rules (kneel-out, last-seconds field goal, tie valued by an overtime model);
- conversion that depends on spread and total, a separate curve for late and desperate fourth downs, and a field-goal curve corrected for selection in long attempts;
- overtime decisions under all three rule sets, including both-teams-possess (regular season since 2025);
- a two-point model (`add_2pt_probs`).

What it does not do: optimize any future decision. The state after the play is valued by a statistical WP model, so the future is "teams play as they historically did". Its stated limits: nothing team-specific beyond spread and total, and the bootstrap covers sampling error only.

**nflfastR EP/WP.** Expected points and two WP models (`wp`, `vegas_wp`) attached to every play. Statistical models of outcomes; no decisions. The nfl4th notes say `vegas_wp` was rebuilt as a probit model of final margin with monotone tree corrections [read: nfl4th NEWS].

**New York Times 4th Down Bot (with Brian Burke).** Expected points early, switching to WP late in the fourth quarter; overtime handled with old sudden-death rules at the time described [search: Nieman Lab 2013 and later summaries]. Proprietary and no longer the reference tool [unverified].

**ESPN Analytics.** Proprietary WP-based recommendations; inputs reported as score, distance, yard line, clock, timeouts, pregame WP and offense/defense strength [search: ESPN explainer 2024]. One-step; no public code or uncertainty.

**NFL Next Gen Stats Decision Guide (AWS).** WP comparison for fourth downs and two-point tries, with conversion models that use tracking data for exact ball position [search: nfl.com and AWS blog]. Proprietary; not reproducible from play-by-play.

**Romer (2006).** Expected-points dynamic programming restricted to first quarters, concluding coaches are too conservative [unverified in detail; citation confirmed in `docs/PAPERS.md`].

**Brill, Yurko, Wyner, arXiv 2311.03490 and 2406.16171.** Game-then-drive bootstrap of a one-step engine; confident in 48% of 2018 to 2022 fourth downs. Details in `docs/PAPERS.md` [read there]. No DP, no optimized future.

**Sandholtz, Wu, Puterman, Chan, arXiv 2309.00756.** Inverse optimization of coach risk preference in a clockless, scoreless expected-points model with league-average future play [read: abstract; details in `docs/PAPERS.md`].

**Chan, Fernandes, Puterman (2021), "Points Gained in Football", Operations Research 69(3), 877-894.** Markov reward process value function for performance assessment, 2013 to 2016 data; not a decision optimizer [search].

**Biro and Walker, arXiv 2103.06939 (JQAS 2022 [search]).** The closest prior work to our M4 [read: PDF]. A run/pass MDP per drive with points utilities, then a late-game version (final five minutes) whose state adds score differential and seconds remaining and whose terminal utility is win, tie (0.5) or loss; fourth down includes punt and field goal. Limits they state: timeouts, kneels and spikes are not modeled; the late-game state space is too large to sweep, so values come from iterated single-state updates, not an exhaustive solve; a tie is worth 0.5; no uncertainty analysis. Data: 2017 and 2018 seasons via nflscrapR. A 2024 follow-up in The American Statistician targets specific outcomes [search].

**Other RL and DP work.** Khandelwal, Harvard senior thesis (2019): off-policy Q-learning for run/pass on simulated drives from 2017 plays [search]. Chebat, Harvard thesis (2025): predictive fourth-down model with RL as future work [search]. Jordan, Melouk and Perry (2009), "Optimizing Football Game Play Calling", JQAS 5 [read: cited in Biro reference list; not opened]. Krasker's Football Commentary site: a backward-induction WP model used for two-point and fourth-down thresholds in the 2000s, with hand-set transition assumptions [search]. ZEUS (Frigo and Bowen): proprietary simulation [unverified]. Kaplan (Yale) lecture slides "Football State Spaces and Decision Making" use a clockless "infinite football" value model [read: first pages only].

**Blown-lead benchmark.** Pipping-Gamon and Wyner, arXiv 2601.18774 v4 (25 Aug 2026), now titled "The Blown Lead Paradox: A Pathwise Calibration Benchmark for Win Probability Forecasts" [read: abstract and PDF]. It gives the continuous-path law, a conservative discrete-time bound, a probability-integral-transform test with a season-stratified dyadic bootstrap, and applies it to ESPN's published NFL feed (1,848 games, 2018 to 2024: no detected departure, p = 0.736) and NBA feed (excess of extreme peaks). It does not test nflfastR `wp`, `vegas_wp`, nfl4th or any open model.

**Causal and conversion-data work.** Yam and Lopez (2019), "What was lost?", Journal of Sports Analytics: matching estimate of about 0.4 wins per team-season from more aggression [search]. Lopez, arXiv 1909.10631: integer yards-to-go hides the true distance, so past pro-aggression findings may be overstated [read: abstract].

**Two-point and end-game.** nfl4th and Next Gen Stats cover two-point tries with one-step WP. ESPN's "down 8, go for two" argument is a short two-possession tree [search]. A dynamic-programming two-point analysis exists on the Punk Rock OR blog (2019) [search; host blocked]. I found no public model of intentional safeties, letting the opponent score, or kneel-versus-score built from play-by-play; absence of search hits is not proof.

**Public GitHub.** `nflverse/nfl4th`, `snoopryan123/fourth_down` (humility paper), `nsandholtz/fourth_down_risk`, `rtelmore/NFLSimulatoR` (drive simulation of fixed strategies) [search]. No public repository was found that solves a full-game NFL MDP exactly or benchmarks RL agents against such a solution. GitHub itself could not be searched from this machine.

## 2. Known criticisms and open problems

1. **Selection bias in conversion data.** Teams that go are a selected group; late desperate attempts differ from normal ones (nfl4th now fits them separately [read]); long field goals are tried only by teams with good kickers [read]. A Wharton note argues team-quality adjustment raises the average team's conversion rate above the raw rate [search].
2. **Unobserved distance and play call.** Integer yards-to-go (Lopez [read]); the humility paper lists both as limitations [`docs/PAPERS.md`].
3. **Team strength and matchup.** Public tools use spread and total only [read: nfl4th README]. Short-yardage strength, kicker and weather are missing.
4. **One-step lookahead.** Every public bot values the next state with a model fitted to historical play, so future decisions are implicitly historical coach decisions. The humility paper names this behavior shift as outside a statistical model. Sandholtz leaves optimizing future decisions to future work.
5. **End-of-game clock.** Statistical WP is weakest where states are rare and rules are sharp. nfl4th now patches this with hand-written kneel and last-second rules [read], which is evidence the fitted model alone was inadequate there.
6. **Model uncertainty.** Game-level bootstraps (Brill; nfl4th dev) capture sampling error only and under-cover even in a toy game (0.71 to 0.76 at nominal 0.90, 2406.16171).
7. **What coaches optimize.** Low-quantile preferences (Sandholtz [read]). Job-security explanations exist in economics (Owens and Roach) [unverified].
8. **Overtime.** Regular-season rules changed in 2025; team analytics staffs reportedly disagree on receive versus kick [search: ESPN survey]. Almost no data exists under the new rules, so a fitted model cannot settle it.

## 3. The gaps

**G1. An exact, open, full-game solution with clock, score and timeouts, set beside the one-step answer.** Closest work: Biro and Walker (late game only, no timeouts or kneels, approximate solve, no uncertainty, 2017 to 2018 data); nfl4th (timeouts, but no optimization). Not done: an exhaustive backward-induction solve, and the three-way decomposition already sketched in `docs/PAPERS.md` (one-step, DP with future fourth downs fixed to coach policy, DP fully optimized). Difficulty: high; clock runoff, timeout use and kneel actions are the hard modeling. Deliverable: policy maps per burning question plus a table attributing each disagreement to optimal future play or to model error.

**G2. Decisions that are not go/kick/punt.** Kneel, intentional safety, letting the opponent score, declining to score, and two-point choices as part of the same solve. One-step tools cannot represent actions with almost no historical data. Difficulty: medium once G1 exists, because these are rule-based transitions. Deliverable: threshold charts (for example, seconds and timeouts at which taking a safety beats punting). Caveat: results depend on modeled transitions that data barely tests; say so.

**G3. Uncertainty carried through the DP.** Bootstraps exist for one-step engines (Brill; nfl4th dev) and for Sandholtz's clockless model. Not done for a clock-and-score DP. Difficulty: medium-high (compute). Deliverable: confident, lean and uncertain categories for DP policies, compared with the one-step categories.

**G4. Blown-lead benchmark on open models and more seasons.** The paper tests ESPN's feed only, 2018 to 2024. Applying it to nflfastR `wp`, `vegas_wp` and our model from 1999 on is an extension, not a new method. Difficulty: low-medium. Deliverable: PIT plots per model and the two collapse rankings. Honest framing: replication plus extension.

**G5. RL and offline RL measured against a known optimum on a real-data MDP.** Prior RL work (Khandelwal, Biro and Walker) has no exact reference. I found no public offline-RL (FQI or CQL) study on logged NFL decisions [absence unverified]. This is a methods benchmark and a teaching result, not football insight. Difficulty: medium.

**G6. Overtime under the 2025 rules by DP.** nfl4th values overtime with a fitted overtime WP model [read]. A rules-based DP needs no overtime data. Difficulty: medium. Deliverable: receive-or-kick value and second-possession two-point thresholds.

Not gaps: a fourth-down bot; a game-level bootstrap of it; team-strength-adjusted conversion; selection-corrected field goals; coach aggressiveness rankings; risk-preference inference. All exist.

## 4. Burning questions

For each: existing tools; whether a full-game DP adds something; extra state needed.

1. **Up 4, about 4:00 left, fourth down in own half.** nfl4th answers one-step. DP is better: the value of converting depends on kneel-out arithmetic and the opponent's optimal hurry-up. Needs seconds, opponent timeouts, kneel action.
2. **Down 8 after a touchdown: go for two now?** nfl4th `add_2pt_probs`, ESPN tree. DP is better because the answer depends on the later two-point choice, which is a sequential problem. Needs two-point action, clock, both teams' timeouts.
3. **Down 3 late, fourth and short in range: tie or win?** One-step tools answer. DP adds an explicit overtime value and time left for the opponent. Needs clock, timeouts, overtime sub-model.
4. **Up 7 late in range: kick for a 10-point lead or go?** One-step tools answer. DP adds the opponent's two-point response (8 versus 7 versus 10). Needs two-point option, clock.
5. **Fourth and goal from the 1 early.** One-step is adequate; DP adds little because the clock is slack. Main issue is uncertainty and integer distance. No extra state.
6. **Fourth and 1 at midfield, first quarter, tied.** Same as 5: an uncertainty question (Brill), not a DP question. No extra state.
7. **Intentional safety when leading late and backed up.** No public tool models the action. DP only. Needs seconds, timeouts, free-kick transition.
8. **Let them score, or decline to score, when the opponent can run out the clock before a chip-shot field goal.** No public tool. DP only. Needs seconds, both teams' timeouts, kneel action, a "concede" action.
9. **Punt or try a 55-plus-yard field goal at the end of a half.** nfl4th answers with a selection-corrected curve. DP gain is small; needs half-end clock.
10. **Overtime under 2025 rules: receive or kick; go for two on the answering touchdown?** nfl4th uses a fitted overtime model. DP is better because data is scarce. Needs possession count, overtime clock.
11. **Was that collapse historic?** Answered by the blown-lead benchmark, not by DP.

## 5. Positioning

The most defensible claim is narrow. Public fourth-down tools, including the rebuilt nfl4th, are one-step: they value the next state with a model of how games have historically gone, and they now do this with timeouts, end-game rules and a bootstrap. This project solves the same game exactly by backward induction with clock, score and timeouts, then shows where and why the exact answer differs from the one-step answer, separating the effect of optimal future play from model error. It carries game-level resampling through that solve, so the dynamic-programming recommendations come with the same confident, lean and uncertain labels as the statistical ones. It answers end-game questions that one-step tools cannot pose at all, such as intentional safeties and conceding a score. It applies a published calibration benchmark for blown leads to open win-probability models that the original paper did not test. The reinforcement-learning work is a benchmark of standard algorithms against a known optimum on a real-data problem, not a claim of better football advice. All of it is reproducible from public play-by-play, and every DP result is conditional on a transition model that end-game data only weakly tests.
