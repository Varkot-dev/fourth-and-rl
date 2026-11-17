# Machine-learning questions in NFL analytics that this project could take on

Written 2026-11-17 by the X research session for the build session. Source material,
not instructions. The owner asked for the research to focus on machine-learning
questions for the NFL.

Evidence labels: **[search]** means seen only in a web-search summary on 2026-11-17 and
not opened, so details are unverified. **[repo]** means already in this repository's
docs. **[X]** and **[YT]** point to `research-kit/x/2_twitter_findings.md` and
`docs/research/inbox/video_transcript_notes.md`. Nothing here was reproduced.

Each question is stated as a learning problem: what is predicted or optimized, from
what data, and how you would know the answer is right.

## A. Sequential decisions and reinforcement learning

**A1. Can offline RL recover good decisions from logged coach behavior?**
Coaches are conservative, so the logged data barely covers aggressive actions. The
question is how far algorithms such as fitted Q-iteration, CQL and IQL get from the
optimum when trained only on what coaches did. This repo can answer it unusually well
because the full-game model is solved exactly, so the true optimum is known [repo:
`uncertainty_and_rl.md`, experiments E1 to E5].
- Existing: a Stanford CS224R course project, "PlayGrader", builds a drive-level MDP
  from PFF and All-22 data on about 388k plays and fits IQL and CQL with expected-points
  and win-probability value heads [search:
  https://cs224r.stanford.edu/projects/pdfs/JP%20Paul%20McAnally%20submission_416260391/224R.pdf].
  This contradicts the note in `landscape.md` that no public offline-RL study on NFL
  decisions was found. It uses proprietary data and, from the summary, has no exact
  optimum to measure against.
- Open: regret against a known optimum; how regret changes with how conservative the
  logging policy is.

**A2. Does a one-step recommendation differ from the full-game optimum, and where?**
Already the project's core [repo: G1]. The ML angle is value-function approximation:
can a learned value function match the exact table, and with how much data?

**A3. Run or pass as a two-player game, not a one-player optimization.**
- Existing: a 2026 conference talk builds a zero-sum Markov game from NFL play-by-play
  and trains Nash DQN under three state abstractions [search: JSAI 2026, Fukita, Masaka
  and Iwasaki, https://pub.confit.atlas.jp/en/event/jsai2026/presentation/1M4-GS-5x-04].
  Economics papers tested minimax on run/pass and rejected it: Kovash and Levitt
  (NBER w15347, 2009) report teams "passing less than they should" and negative serial
  correlation in play calls; Emara, Owens, Smith and Wilmer (MPRA 58907, 2014) use
  3,455 games from 2000 to 2012 and find play types alternate too often [search].
- Public demand: the "EPA go brrrr" argument [X, section 7] and the 1.15M-view video
  whose toy game says run 54% while its data says pass more [YT, section 1].
- Open: estimate the payoff table from data instead of assuming it, which needs the
  defense's choice. Play-by-play does not record the defensive call, so the defense's
  action must be proxied (box count, personnel, coverage shell from charting data).
  That missing variable is the real obstacle and should be stated up front.

## B. Uncertainty and calibration

**B1. How much should a recommendation be trusted?**
- Existing: Brill, Yurko and Wyner show win-probability estimates are high-variance and
  many fourth-down calls are uncertain [repo; search confirms arXiv 2311.03490 v4].
- Public demand: three public models gave three answers on one play (+8.1, +2.5,
  "toss-up") [X section 1; YT notes file, prior-work section]; Shanahan's "51%" [X].
- Open, and learnable: predict when models will disagree. Train on the gap between
  nfl4th, ESPN and Next Gen Stats recommendations where all three are public, with
  features such as distance (integer versus tracked), clock and score.

**B2. Do game-level bootstraps cover?** Already found here to under-cover for tree
models (77% at nominal 90%) [repo: decision log]. A conformal or calibrated-interval
method that does cover would be a methods contribution.

**B3. Is a win-probability model calibrated along whole games, not just play by play?**
The blown-lead benchmark [repo: M3b]. Public demand: 99.1% collapses reported as
shocking with no base rate [X section 3].

## C. Causal and selection problems

**C1. Selection in who goes for it.** Teams that attempt fourth downs are not a random
sample. Existing: a Heckman-style correction project
(`danieldalygrafstein/nfl4th-heckman`, no license) and Yam and Lopez (2019), who
estimate about 0.4 wins per team-season from more aggression by matching [search;
repo]. Open: a doubly robust or instrumental estimate with modern ML nuisance models,
compared with the matching answer.

**C2. Hidden distance.** Next Gen Stats said the Lions' fourth-and-1 was "closer to a
4th and 2" from tracking [X]. Open: model the true distance as a latent variable given
the integer in play-by-play (Lopez, arXiv 1909.10631 [repo]) and propagate it into the
recommendation.

**C3. Scrambles and sacks are called passes.** Reclassifying scrambles flips the
third-down run-versus-pass answer [YT section 1]. A classifier for "called pass" from
play-by-play text is a small, checkable ML task that every run/pass analysis needs.

## D. Attribution: how much is the coach, the referee, or luck?

**D1. Coach cost ledger** (the owner's idea, now in the plan). The learning problem is
a multilevel model: decompose a team-season's procedural-penalty cost into coach,
roster and noise, and test whether the coach effect follows him between teams.
- Existing pieces: flag pricing by win probability with and without the flag [YT/prior
  work: thesidequest methods page]; SumerSports' 2022 in-game coaching model covers
  fourth downs, two-point tries, timeouts and delay-of-game penalties with an XGBoost
  model [search: https://sumersports.com/the-zone/sumersports-in-game-coaching-model-an-introduction/];
  `4th-Down-Audit` (MIT) ranks coach-seasons on fourth downs.
- Open: nobody found combines decisions and penalties in wins with intervals, or tests
  persistence. SumerSports is the closest and should be read in full before building.

**D2. Referee effects: real or noise?** Existing: Snyder and Lopez (JQAS 2015) on
discretionary penalties; a 2021 Iowa honors thesis on twenty years of penalties that
"did not find evidence supporting individual referee bias"; a fixed-effects panel
study by Mendez and Dischman reporting that officials favored Kansas City in playoff
penalties, publicized by UTEP in 2026; Eiserloh, Foreman and Heintz (Frontiers in
Sociology, 2020) on holding calls for 2013 to 2016 [all search]. Two YouTube analyses
cover 2000 to 2026 [YT/prior work]. Open: shrinkage estimates per crew with intervals
and a season-to-season persistence test. The published findings disagree with each
other, which is itself evidence the effects are near the noise floor.

**D3. Outrage versus measured cost.** A text task unique to this project's X corpus:
do the plays the public calls "malpractice" or "cowardly" have higher model cost than
other plays? Labels exist at scale from the Surrender Index bot (every punt since
1999, MIT) [X section 10.1].

## E. Tracking data (probably out of scope, listed so the choice is deliberate)

The league's own ML frontier is player-tracking models: the 2026 Big Data Bowl was a
movement-prediction competition, the 2027 edition asks which pre-draft measures
predict NFL success, and a 2026 AWS/NFL paper uses a transformer on 2020 to 2024
tracking data to predict coverage assignments (arXiv 2603.25901) [search]. These need
tracking data released only for the competition, not play-by-play, and do not connect
to decisions. They would be a different project.

## Which of these fit this repo best (my view, for the owner to accept or reject)

1. **A1 with a known optimum** is the strongest ML result available here: most RL work
   on football has no ground truth, and this repo does.
2. **D1 and D2 as one multilevel-model study** ("is it the coach, the crew, or luck?")
   has the largest public audience [X sections 8.4, 9] and reuses the bootstrap.
3. **A3** is the best-known public question but is blocked by the missing defensive
   call; worth doing only with an honest proxy and a clear statement of that limit.
4. **B1's disagreement predictor** and **C3's called-pass classifier** are small,
   self-contained and checkable, and each fixes something people argue about.

## Sources seen (search summaries only)

- https://arxiv.org/pdf/2103.06939 (Biro and Walker)
- https://arxiv.org/pdf/2309.00756 (Sandholtz et al.)
- https://arxiv.org/html/2311.03490v4 (Brill, Yurko, Wyner)
- https://cs224r.stanford.edu/projects/pdfs/JP%20Paul%20McAnally%20submission_416260391/224R.pdf
- https://pub.confit.atlas.jp/en/event/jsai2026/presentation/1M4-GS-5x-04
- https://www.nber.org/papers/w15347 (Kovash and Levitt)
- https://mpra.ub.uni-muenchen.de/58907/ (Emara et al.)
- https://sumersports.com/the-zone/sumersports-in-game-coaching-model-an-introduction/
- https://www.utep.edu/business/about/hunt-insights/articles/2026/05-08-2026-financial-pressure-study-national-football.html
- https://iro.uiowa.edu/esploro/outputs/undergraduate/NFL-Penalty-Analysis-Referee-Influence-and/9984112118002771
- https://www.frontiersin.org/journals/sociology/articles/10.3389/fsoc.2020.00048/full
- https://arxiv.org/html/2603.25901v1 (coverage transformer)
- https://operations.nfl.com/gameday/analytics/big-data-bowl
- https://content.iospress.com/articles/journal-of-sports-analytics/jsa190294 (Yam and Lopez)
