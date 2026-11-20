# Fellowship research directions

Prepared for Varshith, 2026-11-20. All pilot work lives in the scratch directory `/tmp/claude-0/-home-user-fourth-and-rl/f419b03a-4266-50b3-be51-e6683b77e914/scratchpad/fellowship`. No repo files were changed. No 2026 holdout data was loaded, and nothing was fit on the 2024 or 2025 seasons.

---

## 1. What the Anthropic Fellows program looks for

**How this was checked.** The fellowship scout could not fetch alignment.anthropic.com or job-boards.greenhouse.io directly from this session, because both hosts are outside the session's allowed network. Every point below comes from web search summaries restricted to official domains. Treat all of it as official but summary-level, and confirm wording, dates and figures on the live pages before relying on them.

**Format**
- Fellows work full-time for 4 months on empirical research questions tied to Anthropic's priorities, usually aiming for a public output such as a paper submission. The 2024 pilot ran for 6 months. Sources: [2024 announcement](https://alignment.anthropic.com/2024/anthropic-fellows-program/), [2026 program post](https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/).
- Fellows are not employees. They work with Anthropic mentors, who pitch projects that the fellow then shapes. Postings name Ethan Perez and Samuel Marks as possible mentors.
- Work mostly runs on external infrastructure (open models, public APIs). Postings mention compute funding of roughly 15k USD per month.
- Location is Berkeley or London for the safety track, and fellows need work authorization in the US, UK or Canada. UNVERIFIED for the safety track: the "must be in country" and "no visa sponsorship" lines came from the Economics and Policy posting.
- Stipend: UNVERIFIED. Postings disagree, so use the current posting.

**Research areas** (main AI Safety posting, [5023394008](https://job-boards.greenhouse.io/anthropic/jobs/5023394008)): scalable oversight, adversarial robustness and AI control, model organisms of misalignment, mechanistic interpretability, and AI welfare. Separate workstream postings exist for [AI Safety and Security](https://job-boards.greenhouse.io/anthropic/jobs/5183044008), [AI Security](https://job-boards.greenhouse.io/anthropic/jobs/5030244008), [ML Systems and Reinforcement Learning](https://job-boards.greenhouse.io/anthropic/jobs/5183051008) and [Economics and Policy](https://job-boards.greenhouse.io/anthropic/jobs/5183053008). UNVERIFIED: whether the ML Systems and RL workstream is safety-focused or infrastructure-focused. Read that posting first, because it may be the most natural fit for this repo.

**What fellows have published** (credited on official pages): [Subliminal Learning](https://alignment.anthropic.com/2025/subliminal-learning/), [Stress-testing model specs](https://alignment.anthropic.com/2025/stress-testing-model-specs/), [The Hot Mess of AI](https://alignment.anthropic.com/2026/hot-mess-of-ai/), [Diffuse AI Control on Fuzzy Tasks](https://alignment.anthropic.com/2026/diffuse-ai-control/), [TASTE](https://alignment.anthropic.com/2026/taste/) and [Automated Researchers Can Reliably Mitigate Alignment Failures](https://www-cdn.anthropic.com/7b1c44894e980876479947dcdd40716278aeeffd/automated-alignment-researchers-august-2026.pdf). Official pages also name SHADE-Arena, SLEIGHT-Bench, Petri and open-source circuit tracing. Nearly all of this is empirical LLM work, released with code. Program-level outcome figures (share of fellows publishing, share hired) are Anthropic's own and disagree between postings, so they are not quoted here.

**Selection criteria.** A PhD, prior ML experience and publications are not required. Postings ask for strong Python and ML-framework skills ([4823268008](https://job-boards.greenhouse.io/anthropic/jobs/4823268008)), a strong technical background, motivation to reduce catastrophic AI risk, experience with empirical ML projects (a plus), the ability to work independently while taking feedback, and clear communication. The process listed is: application and reference check, technical assessments and interviews, then a research discussion. UNVERIFIED: the exact form fields.

**Timing.** Applications are rolling, with several cohorts a year. The deadlines found for the January 2027 cohort (October 18 on the blog, October 30 on one posting) have passed as of today. UNVERIFIED: whether late applications roll into a later cohort, and the dates of cohorts after January 2027.

**What this implies for an independent project** (inference, not an official statement): finished, public, empirical work with clean code, baselines, fixed seeds and uncertainty estimates, written up like a paper and tied clearly to a safety theme. A project framed as sports analytics will not read as safety research. The write-up has to lead with the safety question and treat the NFL as the testbed.

---

## 2. Why this project is an unusual testbed

Safety research on Goodhart effects, model exploitation, oversight and interpretability usually has to choose between two kinds of setting: toy domains with exact ground truth (gridworlds, Othello, small synthetic MDPs), or realistic domains where "ground truth" is another proxy (a larger reward model, GPT-4 judgments, human labels). This repo sits in between. The MDP is fitted to real nflverse play-by-play and real coaching decisions, and for that fitted model the optimal values Q*, the coach's values and the exact value of any fixed stochastic policy can all be computed by backward induction over about 408M states. A scratch evaluator does one full-game policy evaluation in 72.7 s at 1.84 GB peak (prototype, not yet a tested repo function). The repo also has a real, observed failure to build on: a learned world model that matched nflverse win-probability models on test data, but whose planner exploited the model's own errors and over-recommended long field goals. The caveat that must appear every time: "exact" means exact for the fitted MDP, not for real football.

---

## 3. Ranked ideas

Each of three judges scored every idea from 1 to 10 on fellowship relevance, novelty, two-week feasibility, achievable rigor and impact, so the maximum combined score is 150.

| Rank | Idea | Combined score (of 150) | Needs |
|---|---|---|---|
| 1 (tie) | Exact Goodhart curves for KL-regularized optimization against a learned value model | 107 | CPU now; Colab for the full grid |
| 1 (tie) | Diffuse sabotage with exact stakes: can monitors catch a policy that shaves win probability inside the noise? | 107 | CPU now |
| 3 | Stakes-blind deference: does an LLM advisor's sycophancy scale with how wrong the authority is? | 104 | API key or open model on Colab |
| 4 | Scaling laws of learned-simulator exploitation, and whether pessimism recovers optimal or merely coach-like play | 94 | Colab GPU |
| 5 | Do evaluators know when they do not know? Exact coverage audit of off-policy value claims after policy selection | 93 | CPU now |
| 6 | Detecting reward hacking without the true reward: expected points vs win probability, graded exactly | 92 | CPU now |
| 7 | Diffuse sabotage under weak oversight: can a weak judge catch a strong advisor that quietly loses games? | 91 | API key |
| 8 | Detecting out-of-support predictions from activations, with exact error labels (planted support holes) | 87 | Colab GPU |
| 9 | Does a next-event predictor represent the behavior policy's value or the optimal value? | 85 | Colab GPU |
| 10 (tie) | Can we detect reward hacking before deployment? An exactly labeled benchmark for auditing optimized policies | 83 | CPU, depends on ideas 1 and 4 |
| 10 (tie) | Unfaithful reasoning with exact causal ground truth | 83 | API key |
| 12 | Mechanism of a learned-simulator failure: SAE features explaining over-predicted long field goals | 70 | Colab GPU, trained model first |

### The top five

**Exact Goodhart curves (107).** This is RLHF reward-model overoptimization ([Gao, Schulman, Hilton 2023](https://arxiv.org/abs/2210.10760)) rebuilt on a sequential decision problem. The coach policy plays the role of the reference model, a learned win-probability model trained on noisy game outcomes plays the reward model, and the policy pi_beta(a|s), proportional to pi_coach(a|s) times exp(beta times Q_proxy(s,a)), is tilted toward the proxy at fourth downs. True value is computed exactly. All three judges ranked it the cleanest experiment. Their shared worry was that three actions leave little room to overoptimize, so the curve might never turn over. The pilot (section 4) showed that it does turn over for small-data proxies. The judges also warned that exact-reward Goodhart curves already exist in small MDPs ([Karwowski et al. 2024](https://arxiv.org/abs/2310.09144)), so the novelty rests on scale, a human reference policy and exact evaluation, not on exact gold alone.

**Diffuse sabotage with exact stakes (107).** This is a low-stakes control problem in the spirit of the Fellows paper [Diffuse AI Control on Fuzzy Tasks](https://alignment.anthropic.com/2026/diffuse-ai-control/): a policy makes many small, deniable bad decisions, and every one has an exact cost. Judges rated it the most feasible high-relevance idea, since a drive-level version runs on CPU in minutes. They also warned that "more data catches more" is textbook, that the oracle monitor wins trivially, and that the adaptive attacker is the only distinctive part. The pilot confirmed that warning more strongly than expected (section 4).

**Stakes-blind deference (104).** This is the most directly Anthropic-legible idea. Tell an LLM advisor that a coach, a user or an analytics staffer prefers a different action, and measure how often it flips as a function of the exact cost of flipping ([Sharma et al. 2023](https://arxiv.org/abs/2310.13548) on sycophancy). A model that defers as readily at a 10-point gap as at a 0.5-point gap is stakes-blind. It is blocked on an API key. The main validity threat: an LLM that disagrees with the fitted MDP may be right about real football. Judges recommended restricting it to bootstrap-stable, large-gap states, conditioning on correctness in the neutral condition, and leading with the asymmetry between pressure toward and away from the optimum.

**Scaling of learned-simulator exploitation (94).** The judges considered this the best story: it grows out of the repo's real world-model failure, and "does pessimism find optimal play or just revert to the coach?" can only be answered with exact Q* and Q_coach. It scored low on feasibility. The CPU smoke model's NLL is worse than a frequency baseline (10.77 against 10.34), so any CPU scaling plot would measure undertraining. Only about 300 states are graded, and the 64k-game Colab run cost is an estimate, not a measurement. Judges said to drop "scaling law" language and scope it to one clean comparison: no guard, support guard and ensemble lower confidence bound, at three data sizes.

**Exact coverage audit of off-policy evaluation (93).** Does a nominal 90% interval for the value of the policy an offline learner chose actually cover its exact true value? This was the most rigorous and cheapest idea, because it reuses M6 code and exact drive-model grading at 0.04 s per policy. All three judges called the expected result textbook (optimizer's curse, selective inference) and recommended it as a supporting section, not a headline. Two judges noted that IS, WIS and DR already exist in `src/fourth_and_rl/offline/real.py`, wired to real logged data rather than the known-truth harness.

---

## 4. The two pilots

Both pilots ran on this CPU machine using synthetic data drawn from the cached solved MDP. Every number below is quoted from the pilot or critic reports.

### Pilot 1: exact Goodhart curves (fourth downs only)

**What was run.** Synthetic coach-vs-coach games were generated: 1,000 games (seed 101), 16,000 games (seed 116), and 4,000 validation games (seed 999). Four LightGBM home win-probability proxies were trained on them (1k or 16k games crossed with 8 or 64 leaves, random_state 7), plus an "oracle" proxy equal to the exact Q_coach. For each proxy, one exact backward pass over all 720 ticks computed Q_proxy at every fourth down using true dynamics, then evaluated pi_beta at six betas (10 to 10,000) for team A against a coach team B. Each pass took about 885 to 916 s and 3.7 GB. KL to the coach and the proxy's claimed gain were estimated from 20,000 Monte Carlo games per beta.

**Proxy quality** (RMSE against exact V_coach on 100k held-out states; label-noise RMSE is 0.405):

| Proxy | RMSE | Greedy-action disagreement with exact argmax |
|---|---|---|
| 1k_large | 0.050 | 32.5% |
| 1k_small | 0.032 | 27.9% |
| 16k_small | 0.024 | 21.2% |
| 16k_large | 0.017 | 14.7% |

**Exact true gain** over coach-vs-coach, in win-probability points, at beta = 10 / 30 / 100 / 300 / 1000 / 10000:

| Proxy | True gain | Proxy's claimed gain at beta 10000 |
|---|---|---|
| oracle | 0.55 / 1.30 / 2.67 / 3.89 / 4.46 / 4.58 | +4.53 |
| 16k_large | 0.50 / 1.19 / 2.42 / 3.39 / 3.72 / 3.74 | +5.09 |
| 16k_small | 0.42 / 1.05 / 2.21 / 3.14 / 3.42 / 3.38 | +4.74 |
| 1k_small | 0.43 / 1.05 / 2.13 / 2.64 / 2.50 / 2.24 | +7.22 |
| 1k_large | 0.47 / 1.05 / 1.71 / 1.61 / 1.21 / 0.94 | +10.91 |

The exact best fourth-down response to the coach is worth +4.587 points, and the oracle tilt reaches +4.580. Internal checks passed: coach-vs-coach is exactly 0.5, coach values match the earlier prototype evaluator to within 2.4e-7, and all 30 Monte Carlo checks agree within 2 standard errors (SE 0.35 points).

**What the numbers show.** For the 1k-game proxies, true value rises and then falls as optimization pressure grows. 1k_large peaks at +1.71 (beta 100) and ends at +0.94, giving back 45% of its peak. Over the same range its claimed gain keeps climbing to +10.91. Peak height orders with proxy error. Even the best proxy captures only 3.74 of the 4.58 available points.

**Critic verdict: promising with fixes.** The main points:
- *The optimizer is not the KL-regularized optimum.* pi_beta is one step of soft policy improvement on an estimated Q_coach, applied independently at each fourth down. Fix: describe it that way, and add exact soft value iteration under a KL penalty for the true KL-regularized optimum.
- *The novelty claim against Gao is overstated.* Gao's synthetic gold is ground truth by construction. Fix: claim exact policy evaluation and exact occupancy KL in a large sequential MDP with a human reference policy, positioned against Karwowski et al. 2024 and Laidlaw et al. 2024, "Correlated Proxies" ([arXiv 2403.03185](https://arxiv.org/abs/2403.03185); cited from memory by the critic, unverified).
- *The functional-form fit is mostly an artifact.* b is about 0.009 for every curve, including the oracle, which cannot overoptimize. So the fit mainly captures saturation of KL as the policy becomes deterministic. Two of the implied peaks also lie outside the data. Fix: drop the coefficient-scaling claim for now.
- *Only two turnovers are clear.* The 16k_small dip is 0.04 points from one seed. Fix: use 3 to 5 proxy and data seeds per cell, and call 16k_small a plateau unless the dip survives.
- *Float16 Q storage distorts the KL and the diagnostics.* 2.4% of oracle states have exact float16 ties at the top, and 8.3% have a top-two gap below 1e-3. The argmax breaks ties toward "go". Fix: store Q in float32 or as advantages, handle ties explicitly, and compute KL inside the exact pass with a forward occupancy pass.
- *Post-decisions contaminate the gold.* Timeouts and two-point tries remain model-optimal for each side's own value. Fix: fix them to coach-vs-coach behavior, or measure the effect.
- *Verification is light.* Monte Carlo at SE 0.35 cannot detect errors around 0.7 points, which is the size of the reported effects. Fix: test the grid predictor on all ticks and proxies, and run at least 200k Monte Carlo games at one or two cells.
- *The turnover is the optimizer's curse on a three-action argmax.* Fix: add something the noise model does not imply. Predict each curve's peak from the per-state error distribution, map where the loss arises, and compare mitigations at matched exact KL.

### Pilot 2: diffuse sabotage (drive-level, CPU)

**What was run.** The testbed was the M6 drive MDP (1,188 fourth-down states) with the coach policy as the honest baseline. Harm is exact, in **expected points per drive**, not win probability. Saboteurs were exponential tilts of the coach policy: greedy, fuzzy-only (close calls only), adaptive (exploiting one weak monitor's errors), and a joint attacker that solves for two multipliers so that the expected values of both deployed monitor statistics match the honest values. Monitors were behavior surprisal, plug-in value monitors fitted on 2k, 20k or 200k honest decisions, and an exact Q* oracle, each thresholded at 5% false-positive rate. Total compute was about 17 minutes. pilot.py and analyze.py were outputs inherited from an earlier attempt and were checked, not re-run.

**What the numbers show.**
- The coach's own regret is 0.2244 points per drive, at 1.04 decisions per drive.
- Against a non-adaptive saboteur, weak value monitors need about 1.2x (200k), 1.8x (20k) and 2.9x (2k) as much data as the oracle. A behavior-only monitor is 10 to 60 times worse at small budgets.
- Restricting the saboteur to close calls yields little harm. Up to about 0.0015 to 0.005 points per drive evades every monitor at N = 10k to 100k.
- The joint attacker at lam = 3 removes 0.058 points per drive, about 26% of the coach's own regret. Combined two-sided power is 0.18, 0.08 and 0.10 at N = 1k, 10k and 100k, while the oracle and value_200k monitors reach 1.00. An attacker using only its own Q estimate, fitted on 200k decisions, still removes 0.054.
- A Neyman-Pearson behavior test that knows the saboteur's policy catches it in 29 decisions, faster than the oracle's 45.

**Critic verdict: promising with fixes, but the headline does not hold.**
- *The joint-evasion result is mostly built in by construction.* Matching the means of two linear statistics leaves thousands of free cells. Using the pilot's own setup, the critic found that the variance of the value_20k statistic is 0.122 under the coach and 0.235 under the lam = 3 saboteur. A squared version of the same weak statistic would need about 65 decisions for 50% power at lam = 3 (oracle 45) and about 760 at lam = 1 (oracle 259). These are z approximations, not checked by sampling. Fix: evaluate against monitor families closed under obvious extensions (two-sided, second moment, chi-square or G-test on counts, a classifier), or plot harm against the number of constraints the attacker must match.
- *The saboteur is not statistically indistinguishable.* Its TV distance is 0.081 per decision and its KL is 0.077. Fix: give the attacker a divergence budget and measure harm against it.
- *The oracle's win is close to a tautology.* By the regret identity, the oracle statistic is the harm. Fix: present the oracle as the ceiling.
- *"Monitor diversity matters" rests on one unlucky fit.* value_20k has MAE 0.62, the same as the 2k monitor. Fix: refit monitors over 10 or more seeds.
- *Units and scale.* Results are in points per drive, and 10k to 100k decisions corresponds to many league-seasons. Fix: convert to win probability with the full game, and convert N to seasons using a count from the training seasons.
- *Noise and prior work.* Honest false-positive rates ranged from 0.005 to 0.10 across cells, so use 2,000 or more null datasets. Cite Wen et al., "Adaptive Deployment of Untrusted LLMs Reduces Distributed Threats" (arXiv 2411.17693; cited from memory by the critic, unverified), along with the classical moment-matching and SPRT literature.

**Summary of the two pilots.** Pilot 1 found a real signal that needs hardening. Pilot 2 found a measurement framework whose headline was largely an artifact of mean-only monitors.

---

## 5. Recommendation

### Pursue first: exact overoptimization of a learned value model, with mitigations compared at matched exact KL

Working title: **"Overoptimization of learned value models under exact evaluation: where it happens, how to predict it, and which mitigations actually help."**

**Why this one.**
1. It is the only pilot with a positive signal that survived critique. The critic's fixes are engineering and controls, not a change of question.
2. It maps directly onto a central safety concern, reward-model overoptimization in RLHF, and onto the Fellows "ML Systems and RL" and safety workstreams.
3. Exact gold means the uncertainty in each figure comes only from proxy and data seeds, not from a learned judge. Few overoptimization studies can say that.
4. It needs no API key, and most of it runs on CPU.
5. It sets up the world-model exploitation study (idea 4) as the natural second paper: that study adds dynamics error to the value error isolated here.
6. A scaled-down slice of the sabotage idea can be added later as one section on "can a monitor tell the overoptimized policy from the coach?", using the critic's stronger monitor families.

**The contribution beyond the optimizer's curse** (the critic's main demand) has three parts:
- (a) predict each curve's peak beta and peak loss in advance from the proxy's per-state error distribution, then test the prediction on held-out seeds;
- (b) exact per-state regret maps showing where the loss comes from;
- (c) compare mitigations (ensemble minimum, ensemble lower confidence bound, action-support mask, early stopping in beta) at matched exact KL, and compare the one-step tilt with the exact KL-regularized optimum from soft value iteration.

### Falsifiable claim

Under exact evaluation on the fitted game MDP, using proxies trained on 1k games and with 5 proxy seeds and data seeds per cell:
1. true gain under KL-tilted optimization peaks at a finite beta and then declines, by more than twice the seed-to-seed standard deviation;
2. peak beta and peak loss are ordered by held-out proxy error, and a prediction made from the per-state error distribution lands within its stated interval on held-out seeds;
3. at matched exact KL, an ensemble lower confidence bound proxy loses less true value than the single proxy.

The claim is falsified if:
- the seed spread exceeds the 0.4 to 0.8 point losses seen in the pilot;
- the ordering by proxy error breaks across seeds;
- the exact-KL axis reorders the curves relative to Monte Carlo KL;
- or ensemble pessimism does not reduce the loss at matched KL.

A monotone result at every data size would also be reported. It would be a legitimate finding ("no overoptimization regime at these sizes"), but a different paper.

### Two-week plan (10 working days)

CLAUDE.md asks for milestones to be done in order in `docs/PLAN.md`, so add this as a milestone with "done when" criteria and a decision-log entry before starting, and commit one logical step at a time.

| Day | Milestone | Done when |
|---|---|---|
| 1 | Promote the fixed-policy evaluator from scratch into a repo module, with the ring-buffer pass. | Unit tests pass; coach-vs-coach equals 0.5; matches the existing solver's values. |
| 2 | Add post-decision policy plumbing (timeouts and two-point tries fixed to coach behavior) to `_running_values` and `_derive`. Test against Monte Carlo at 20 or more states, not one. | All states agree within 2 SE, at SE near 0.002 per state. |
| 3 | Promote the grid predictor; store Q as float32 or as advantages; make tie handling explicit (uniform or excluded). | The grid predictor matches `Booster.predict` on all ticks and all proxies. |
| 4 | Build the exact forward occupancy pass, giving exact sequence KL inside the exact pass. | Exact KL matches Monte Carlo KL within its SE at the pilot cells. |
| 5 | Re-run the pilot's 5 curves with the fixed pipeline, plus a 200k-game Monte Carlo check at two cells. **Go/no-go gate:** does the 1k turnover survive the fixes? | A written result in either case. If it fails, switch to the plateau framing or add timeout and play-mode actions to the tilt. |
| 6 | Launch the seeded grid on Colab (run A below). Meanwhile, implement exact soft value iteration under a KL penalty. | Grid running; soft value iteration tested on the drive MDP. |
| 7 | Run the mitigations: ensemble min and LCB (5-member ensembles), action-support mask, early stopping in beta, each at matched exact KL (run B). | One table of true loss at matched KL per mitigation, with seed intervals. |
| 8 | Peak prediction: fit a predictor from per-state error to peak beta and loss on half the seeds, then test it on the other half. Compare the tilt with the KL-regularized optimum. | Prediction error reported with intervals, held out. |
| 9 | Exact regret maps by game situation (clock, score, field position, field goal distance). Make the figures with scripts under `scripts/`, outputs under `assets/` and `results/`. | Every number in the draft traces to a saved file in `results/`. |
| 10 | Draft the write-up, honesty pass, release code. | Draft passes a read-through against the CLAUDE.md honesty rules. |

### Colab runs needed

All cost figures here are from the pilot report or are marked as estimates.
- **Setup:** `data/mdp` (2.0 GB) is gitignored, so it must be uploaded to Drive or re-solved on Colab (a re-solve took 53 s and 2.8 GB on this machine).
- **Run A, the seeded grid:** N in {1k, 4k, 16k, 64k}, 3 to 4 LightGBM capacities plus 2 to 3 MLP widths, 3 to 5 seeds each, all betas in one pass. That is about 100 to 250 passes. At about 15 minutes per pass with two in parallel on this CPU, the pilot report estimates 12 to 30 CPU hours. Options are a long high-RAM Colab CPU session, or porting the per-tick backup to the JAX backend (`mdp_team_jax.py` exists; the GPU speedup is unmeasured). Store only start values and Q at sampled states: the float16 Q tables were about 0.6 GB per proxy, and float32 doubles that.
- **Run B, mitigations:** one extra pass per variant per seed, at a smaller N subset (1k and 4k).
- **Optional run C, MLP proxies:** training the MLP proxies on a T4 if CPU training is slow. Unmeasured.

### Write-up

- **Format:** a short paper of about 8 pages on arXiv, plus a blog post of about 1,500 words that leads with the safety question, plus a public code release. The NFL appears as the testbed in the second paragraph, not the title.
- **Figures:**
  1. True gain and proxy-claimed gain against exact sqrt(KL), with seed bands, for each proxy, with coach and best-response reference lines (the pilot's `goodhart_pilot.png`, rebuilt).
  2. Predicted against observed peak beta and loss on held-out seeds.
  3. Mitigations at matched exact KL.
  4. A regret map of where overoptimized play loses value.
  5. One-step tilt against the exact KL-regularized optimum.
- **Code:** tested evaluator, occupancy pass and grid predictor in `src/`; one config file per run; recorded seeds; CI passing.
- **Framing:** cite Gao et al., Coste et al. ([2310.02743](https://arxiv.org/abs/2310.02743)), Karwowski et al., and Laidlaw et al. (verify first). State the contribution as exact evaluation in a large sequential MDP with a human reference policy, not as fixing a weakness in Gao.

---

## 6. How it connects to work already done

- **World-model exploitation finding.** The v2 world model matched nflverse win-probability models on test data, yet its planner over-recommended 60+ yard field goals. That is optimization exploiting errors in a learned model, with dynamics error and value error mixed together. The recommended project isolates the value-error half: true dynamics, a learned value model, exact grading. The paper can then say that overoptimization of a learned value model already appears when dynamics are perfect, and that the repo's real-data world model shows the same failure when dynamics are learned too.
- **Ground-truth notebook.** `ground_truth_run.py` and `gt_v1` train the world model on synthetic games from the same MDP and grade its planning against Q* and Q_coach. They are the natural second paper (idea 4): the same exact grading with dynamics error added, and the question of whether pessimism finds optimal play or only reverts to coach play. The tested evaluator from days 1 to 4 also makes gt_v1's Q_coach exact rather than Monte Carlo, once the post-decision policy matches gt_v1's `post="coach"` setting.
- **Uncertainty work.** The game-level bootstrap and the "analytics, have some humility" point ([Brill, Yurko, Wyner](https://arxiv.org/abs/2311.03490)) are the human-scale version of the same problem: a decision rule is only as good as the uncertainty in its value estimates. Ensemble lower-confidence-bound proxies are the direct descendant of that work, and the `optimizers_curse` section in `results/m6_offline_known_truth.json` is the drive-scale precursor of the turnover mechanism.
- **Kicker selection-bias model.** Long field goals are where the real world model failed, and probably (unverified) where proxies err most. The day-9 regret maps can test whether overoptimization losses concentrate at long field-goal distances, which would tie the synthetic result to the real failure.

---

## 7. Honest limits

- **Exact for the fitted MDP only.** The model uses identical league-average teams, scores clipped at plus or minus 21, and play modes instead of run or pass, and it is stamped on 2010 to 2023 data. One kickoff table uses 2024 data, a documented rule-constant exception. The results are about optimizing against this model, not about real football.
- **The pilots are small.** Pilot 1 had one seed per proxy, covered fourth downs only, used float16 Q and a Monte Carlo KL, and kept model-optimal post-decisions. Only two turnovers are clear. Pilot 2's headline did not survive critique, and its units are points per drive.
- **Nothing here can go in the README yet.** All pilot outputs live in the scratch directory. Under the repo's honesty rules, a number can be cited only after a repo script reproduces it and saves it under `results/`.
- **The optimizer is not RLHF.** A tilted three-action choice at fourth downs is a much smaller policy space than a language model's. Results speak to the shape of overoptimization under exact evaluation, not to its magnitude in LLM training.
- **The mechanism may be textbook.** If the peak-prediction and mitigation results add nothing beyond the optimizer's curse, the project is a careful replication in a new setting. That is still a reasonable portfolio piece, but a weaker one.
- **Costs are partly unmeasured.** Colab and JAX timings, and MLP proxy costs, are estimates. Disk is tight: the pilot report measured 4.6 GB free, after the q4 tables (about 2.9 GB, deletable).
- **Fellowship facts are summary-level.** The scout could not fetch the official pages directly, the January 2027 deadlines have passed, and later cohort dates are unknown. Check the live postings before applying.
- **Some citations are from memory.** Laidlaw et al. 2024, Wen et al. 2024, Wolfers 2006, Thomas et al. 2015 and Hanna et al. 2017 were cited from memory by subagents and must be verified before use.
- **LLM directions are blocked.** Ideas 3, 7 and 11 need an API key from you, or an open model on Colab. No LLM experiment was run.
