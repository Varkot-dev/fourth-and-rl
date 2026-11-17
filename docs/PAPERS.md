# Papers: verified citations and build notes

Checked on 2026-11-17 against the arXiv abstract pages and the full PDFs of the latest arXiv versions. Every number below was read in the paper and carries its section or table. Numbers here are the authors' numbers, not ours, and must not be mixed with results from this repo.

## Verification summary

| Plan citation | Status | Difference from the plan |
|---|---|---|
| arXiv 2311.03490, Brill, Yurko and Wyner, "Analytics, have some humility..." | Verified (title, authors, ID) | None. The plan gives no journal; see the note below. |
| arXiv 2309.00756, Sandholtz and coauthors, "Learning Risk Preferences in Markov Decision Processes..." | Verified (title, authors, ID, 2014 to 2022 data) | Citation correct. The coauthors are Wu, Puterman and Chan. The paper's MDP is much smaller than the plan's wording suggests: no clock or score in the state, and nothing is solved by optimizing over future decisions. |
| arXiv 2406.16171, Brill, Yurko and Wyner, "Exploring the Difficulty of Estimating Win Probability: A Simulation Study" | Verified (title, authors, ID) | The plan says "published in the Journal of Quantitative Analysis in Sports, 2025". The arXiv page only says "Accepted to JQAS" and lists no journal reference. Publication year, volume and pages are not confirmed. |
| Romer (2006), "Do Firms Maximize? Evidence from Professional Football" | Confirmed from two reference lists | None. Journal of Political Economy 114, pages 340 to 365 (volume and pages from the reference list of 2309.00756; pages also in 2311.03490). Issue number not given in either list. Romer's paper itself was not read. |

## 1. Analytics, have some humility (arXiv 2311.03490)

**As stated on the arXiv page**
- Title: Analytics, have some humility: a statistical view of fourth-down decision making
- Authors: Ryan S. Brill, Ronald Yurko, Abraham J. Wyner
- arXiv ID: 2311.03490 (stat.AP)
- Dates: submitted 6 Nov 2023 (v1); last revised 31 Jan 2025 (v6). Six versions.
- Journal reference: none on the arXiv page. The reference list of 2406.16171 v5 cites it as The American Statistician, 79(3):393-409 (2025). That comes from the other paper, not from this arXiv page.
- Full text read: yes, v6 (40 pages including supplement).

**Summary**

Question. Fourth-down bots recommend the action with the highest estimated win probability (WP). How sure can we be that the recommended action is really the best one, given that WP is fitted from a few thousand games?

Data (Section 2.1). nflFastR play-by-play, 1999 to 2022. The main fitting set is 2006 to 2021: 600,825 plays, 229,635 first-down plays, 4,101 non-tied games. Results are reported on fourth downs from 2018 to 2022.

Decision model (Sections 2.3 to 2.5). One XGBoost model of first-down WP, fitted on first-down plays only: score differential, seconds remaining, point spread, yard line, second-half-kickoff flag, both teams' timeouts, total score, and score differential / (0.01 + seconds remaining), with monotonic constraints. Each action's value is built from that model plus small transition models: punt (linear regression for expected next yard line), field goal (logistic regression, with 500 synthetic misses added from 51 to 99 yards out), and go (logistic conversion probability that borrows third-down plays, plus linear models for expected gain given success and given failure). They evaluate WP at the expected next yard line, not over the full outcome distribution.

Bootstrap (Section 3). A "randomized cluster bootstrap": resample games uniformly with replacement, and within each resampled game resample drives uniformly with replacement. B = 101 replicates. The full decision model is refitted on each replicate. The text says the bootstrap is also tuned with the fractional bootstrap from the simulation paper, but the fraction used is not stated in the paper.

From bootstrap to category. Let d(x) be the point-estimate best action at state x and d_b(x) the best action in replicate b. boot%(x) = 100 * (number of b with d_b(x) = d(x)) / B. Categories (Section 4.2, Appendix C): confident if boot% is in [83, 100], lean if in [67, 83), uncertain if in [0, 67). They also give a 90% interval on the WP gain as the 6th and 96th of the 101 sorted bootstrap gains. B = 101 was chosen by a stability check: over 100 repeated bootstrap draws, the mean share of draws agreeing on a play's category was 0.9 (Appendix C).

Headline numbers.
- Out-of-sample log loss: their model 0.440, Lock and Nettleton random forest 0.446, Baldwin XGBoost 0.476, fair coin 0.693 (Table 1).
- 44% of fourth downs have an estimated gain under 1% WP and 87% under 4% (Section 4.2).
- Confident in 48% of 2018 to 2022 fourth-down decisions; 27% uncertain (Section 4.2).
- On confident plays, coaches chose correctly 91% of the time when they should kick and 49% when they should go (Section 4.2).

Limitations the authors state (Section 5). The bootstrap captures sampling variance only, not model bias, so it understates uncertainty. Expectations are swapped with WP for simplicity. Yards to go is an integer (inches matter on fourth and short). The play call is unobserved. If teams followed the advice, behavior and WP would shift, which a statistical WP model cannot represent.

## 2. Learning Risk Preferences in Markov Decision Processes (arXiv 2309.00756)

**As stated on the arXiv page**
- Title: Learning Risk Preferences in Markov Decision Processes: an Application to the Fourth Down Decision in the National Football League
- Authors: Nathan Sandholtz, Lucas Wu, Martin Puterman, Timothy C. Y. Chan
- arXiv ID: 2309.00756 (stat.AP; math.OC)
- Dates: submitted 1 Sep 2023 (v1); last revised 15 Aug 2024 (v3)
- Journal reference: The Annals of Applied Statistics 2024, Vol. 18, No. 4, 3205-3228. DOI in the page metadata: 10.1214/24-AOAS1933.
- Comments: 22 pages, 12 figures
- Full text read: yes, v3 (main text and Appendices A and B; figures only through their captions).

**Summary**

Question. Assume coaches' fourth-down choices are optimal for some unknown risk preference, and estimate it.

Data (Section 1.1). nflfastR play-by-play, 2014 through 2022 (nine seasons).

State space (Section 3.1.1). Play states are (team with possession, down 1 to 4, yard line in ten 10-yard bins, yards to go 1 to 9 and 10+). Scoring states are (team, touchdown / field goal / safety). There is no clock, score differential or timeouts in the state. Actions exist only on fourth down: GO, FGA, PUNT. Rewards (Section 3.1.4): touchdown 6.95 (six plus the average conversion value), field goal 3, safety -2, negated for the other team.

Not a solved multi-period MDP. All future decisions follow a fixed league-average policy, so the problem is a one-period decision followed by a Markov reward process (Section 3.1). Optimizing future decisions is left as future work.

Transitions (Section 5.1). Plain empirical proportions: counts of (state, action, next state) for fourth downs, and counts of (state, next state) for everything else under the league-average policy. GO is rare in some states, so its estimates are unstable and produce odd "islands". Two fixes (Section 5.3): fourth-down GO data is augmented with third-down plays, and the quantile surfaces are smoothed with a monotone-decreasing shape-constrained additive model over yard line and yards to go (R package scam, k = 4).

Opponent and clock. The two teams are identical league-average teams, so team B's value is the negative of team A's (Appendix A). The value function is the infinite-horizon expected point differential, solved in closed form by a matrix inverse following Chan, Fernandes and Puterman (2021); the text extraction of equation A.2 was garbled, so take the exact formula from that paper. Clock and score are handled only by stratifying the analysis by nflfastR win probability (twenty bins of 0.05, later three bins), by quarter and by season.

Inferring risk (Sections 4 and 5). For each (state, action), the next-state value is a discrete random variable: reward of the next state plus its value, with the transition probabilities as weights. A coach with preference tau picks the action with the highest tau-quantile of it. tau is chosen to minimize the share of observed decisions that differ from the tau-optimal action (average Hamming loss). Two taus are fitted: opponent's half and own half, split at the 50. Three or more regions are not identifiable. Inference is limited to tau in [0.2, 0.8]. Uncertainty: 200 game-level bootstrap samples, recomputing transitions, values and quantiles each time (Section 5.6).

Headline findings (mostly figures). Coaches fit low quantiles, are more risk-tolerant in the opponent's half, and more so over time (Sections 6.1, 6.3). No coach's median own-half tau exceeds the risk-neutral policy's (Section 6.4). In a regression of average points gained on fourth down, the coefficient on tau is 0.769 (standard error 0.138), with partial R-squared 0.048 (Table 1).

Limitations stated (Sections 3.2, 7). Score, time and timeouts are excluded for sparsity and tractability; win probability is only a proxy. Transitions are league-wide, not team-specific.

## 3. Exploring the Difficulty of Estimating Win Probability (arXiv 2406.16171)

**As stated on the arXiv page**
- Title: Exploring the Difficulty of Estimating Win Probability: A Simulation Study
- Authors: Ryan S. Brill, Ronald Yurko, Abraham J. Wyner
- arXiv ID: 2406.16171 (stat.ME; stat.AP)
- Dates: submitted 23 Jun 2024 (v1); last revised 20 Aug 2025 (v5). Five versions.
- Journal reference: none listed. Comments field: "Accepted to JQAS".
- Full text read: yes, v5 (22 pages).

**Summary**

Question. How well can a model recover true win probability when every play of a game carries the same win/loss label?

Simulated game (Section 2.1), "random walk football". The field has yard lines 0 to L. The ball starts at L/2 and moves one yard left or right with probability 1/2 each play. Reaching 0 scores +1, reaching L scores -1, and the ball resets to L/2. The game ends after T plays; a tie is settled by a fair coin. They use L = 4 and T = 56, the average number of first-down plays per game in the companion paper. True WP(t, x, s) is computed exactly by backward dynamic programming: terminal value 1, 1/2 or 0 by the sign of the score, and each earlier value is the average of the two successor values (equations 6 and 7).

Design (Sections 2.2 to 2.4). A dataset has G games; K randomly chosen plays are kept per game. K = 1 gives independent labels and K = T = 56 mimics real data. Row count is held fixed with G = round(zeta * T / K), zeta = 4,101. The only model is XGBoost on (t, x, s). Squared bias, variance and RMSE against true WP are computed over M = 100 training sets, each scored on a test set generated with G = 10,000 and K = 1.

Findings on accuracy and effective sample size.
- Error grows roughly linearly in K (Figure 1).
- Keeping all plays per game still beats one play per game (Figure 2). Bias is worst early in the game (Figure 3).
- Effective sample size (Section 2.5): fit a biexponential curve to RMSE versus zeta for K = 1 and K = T, and find zeta' with RMSE(zeta', K = 1) = RMSE(zeta, K = T). For zeta = 4,101 games, zeta' = 2,291, or 56% of nominal. For 2,050 games, 645 (31%). For 8,202 games, 6,911 (84%).

Findings on intervals (Sections 2.6, 2.7). B = 101; the 90% interval is the 6th to 96th sorted bootstrap estimate. Coverage and mean width (Table 5):
- Standard bootstrap (resample plays): 0.60, width 0.027.
- Cluster bootstrap (resample games, keep all plays): 0.71, width 0.036.
- Randomized cluster bootstrap (resample games, then resample T plays within each): 0.76, width 0.042.

The fractional bootstrap resamples only G * phi games. Coverage by phi (Table 6): 0.75 gives 0.80; 0.5 gives 0.85; 0.35 gives 0.90 with mean width 0.063. Even then, coverage is about 85% where true WP is near 0.3 or 0.7 (Figure 5).

Limitations the authors state. phi cannot be tuned on real data because true WP is never observed; they suggest tuning it on synthetic seasons from a realistic simulator, which is hard to build (Section 2.7). They expect a smaller effective sample size and worse coverage in real football (Sections 2.5, 2.6). State-space models have an effective sample size equal to the number of plays but more model bias (Section 3).

## What this means for our build

### M3: game-level bootstrap for fourth-down decisions

1. **Resample whole games and refit everything.** Each replicate must refit the win-probability model and the conversion, field-goal and punt models, then recompute the best action in every chart cell. Refitting only the win-probability model would understate the spread. Never resample plays: in the simulation that gave 0.60 coverage for a nominal 0.90 interval (2406.16171, Table 5).
2. **Use the published decision rule so results are comparable.** boot% = share of replicates whose best action equals the point-estimate best action. Confident at 83% or more, lean from 67% to under 83%, uncertain below 67% (2311.03490, Section 4.2). Our "too close to call" should be defined as their "uncertain", with "lean" shown as a middle shade. Put the thresholds in config.
3. **Use B = 101 and check it.** An odd B gives clean 6th and 96th order statistics for a 90% interval on the win-probability gain. Repeat their stability check on a sample of plays: rerun the bootstrap with different seeds and report how often the category is unchanged (they got 0.9 on average, Appendix C). Save the seeds.
4. **Say in the README that our intervals are probably too narrow.** A plain game-level bootstrap covered 0.71 and the games-then-plays version 0.76 at nominal 0.90, in a far simpler game (Table 5). Two honest options: (a) report the plain game bootstrap and state this caveat with the source; (b) add the fractional bootstrap as a config option (resample phi * G games) and show the chart at phi = 1 and phi = 0.35 as a sensitivity check. Do not claim phi = 0.35 is calibrated for real data; the authors say it cannot be tuned there.
5. **Decide and log the within-game step.** The fourth-down paper resamples drives within each resampled game; the simulation paper resamples plays within each game. The plan only says "bootstrap over games". Start with the plain cluster bootstrap, offer drives-within-games as an option, and record the choice in the decision log.
6. **Reuse random walk football in M3b.** The effective-sample-size simulation in the plan is exactly Section 2 of 2406.16171: L = 4, T = 56, K from 1 to 56, XGBoost on (t, x, s), M = 100. Their numbers (56% at 4,101 games) are a target to compare against, quoted as theirs.
7. **Judge coaches only on confident plays** (M7), and report the kick and go agreement rates separately, as in Section 4.2 of 2311.03490.
8. **A cheap improvement over the paper.** They evaluate win probability at the expected next yard line and name this as a limitation. Our engine can average win probability over the full distribution of punt and conversion outcomes. If we do, any disagreement with their charts has a known cause.

### M4: MDP solved by dynamic programming

1. **Do not treat Sandholtz as a reference solution for the full-game MDP.** Its state has no clock, score or timeouts, the yard line is in 10-yard bins, the value is expected points over an infinite horizon, and future play is fixed at the league average. It is a good check only for our single-drive stage: under a fixed league-average policy our expected-points values by (down, yard-line bin, distance) should have the same shape as their Figure A.1 (value falls as down, yard line or distance increases).
2. **Use random walk football as the first unit test for the backward-induction code.** It has clock, field position and score, a three-line recursion and an exact answer (2406.16171, equations 6 and 7). If our solver reproduces that table, the indexing of time, score and the terminal tie rule is right before any real data is involved.
3. **Do not use raw empirical transition counts at our resolution.** Sandholtz used empirical proportions on only 10 yard-line bins by 10 distances and still got unstable GO estimates and "islands" of nonsense recommendations where coaches rarely go (Section 5.3). Our grid (exact yard line, 12 distance bins, score, clock) is far finer. Feed the DP from the fitted M1 models, borrow third-down plays for conversion as both papers and Romer do, and add a test that the policy map has no isolated go cells. Where smoothing or monotonicity is imposed, say so.
4. **Keep the symmetric opponent and add the scoring conventions.** Sandholtz treats the game as two identical league-average teams with the opponent's value equal to the negative of ours, which matches our decision-log choice (1 minus value for win probability). For the single-drive version adopt their reward convention: touchdown worth six plus the league-average conversion value computed from our own data (theirs was 6.95 on 2014 to 2022), field goal 3, safety -2.
5. **Name the real reason DP and one-step lookahead disagree.** Statistical win-probability models, and Sandholtz's value function, assume everyone keeps playing like the league average after this play. The DP assumes both teams play optimally from here on. The humility paper lists this behavior shift as something statistical models cannot capture (Section 5). To meet M4's "done when", solve the model twice: once with future fourth downs fixed to the empirical coach policy (policy evaluation) and once fully optimized. The first should sit close to the one-step engine; the gap between the two is the effect of optimal future play, and what remains is model error.
6. **Carry the bootstrap through the DP.** Sandholtz recomputes transitions and values on each of 200 game-level bootstrap samples (Section 5.6). The Brill papers argue transition models have a larger effective sample size than win/loss labels but more model bias, and that their uncertainty still needs propagating (2311.03490, Appendix F). So reuse the M3 resampled game lists: refit transitions, re-solve, and report the share of replicates in which each action is optimal. That gives a DP chart with the same confident, lean and uncertain categories as M3, and the comparison of the two charts is a result worth showing.
7. **Stretch goal sizing.** The risk-preference analysis needs only the single-drive model: next-state value distributions per (state, action), a quantile in place of the mean, two taus split at midfield, and Hamming loss against observed decisions. It does not need the full-game DP.
