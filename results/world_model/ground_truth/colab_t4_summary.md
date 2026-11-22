# Ground-truth world-model run, Colab T4, 2026-11-22 (summary as printed)

Pasted from the notebook's summary cell (`world_model_ground_truth.ipynb`). The full results zip
(`world_model_gt_results.zip`) is in the owner's Drive and still has to be added here.

Preset t4 on Tesla T4; sizes 1,000, 4,000, 16,000, 64,000 games.
- setup: done in 699.0s
- generate: done in 63.1s
- train: done in 1710.3s
- evaluate: done in 2775.1s
  exact game: built from public data in this session; ground truth gt_colab (recomputed here: the saved one did not match this game)
  built from public data in 2.7 min, peak memory 8.97 GB
  the coach-like policy itself loses 0.81 points of win probability per decision on average against the best action
  1,000 games (300 states): picks the truly best action in 65% (95% 59% to 70%); 76% where the true margin is over 3 points; mean regret 0.71 points (95% 0.54 to 0.91), 90th percentile 2.81; learned values off by +0.05 points on average; NLL 5.205 (frequency baseline 10.330)
    headline state (up 4, 4:00, 4th and 2 at own 40): learned GO 0.722, FG n/a, PUNT 0.756; true GO 0.769, FG n/a, PUNT 0.729
  4,000 games (300 states): picks the truly best action in 66% (95% 60% to 71%); 87% where the true margin is over 3 points; mean regret 0.50 points (95% 0.39 to 0.64), 90th percentile 1.70; learned values off by +0.25 points on average; NLL 4.678 (frequency baseline 10.327)
    headline state (up 4, 4:00, 4th and 2 at own 40): learned GO 0.845, FG n/a, PUNT 0.750; true GO 0.769, FG n/a, PUNT 0.729
  16,000 games (300 states): picks the truly best action in 80% (95% 75% to 84%); 92% where the true margin is over 3 points; mean regret 0.35 points (95% 0.22 to 0.54), 90th percentile 1.01; learned values off by -0.08 points on average; NLL 4.205 (frequency baseline 10.326)
    headline state (up 4, 4:00, 4th and 2 at own 40): learned GO 0.809, FG n/a, PUNT 0.710; true GO 0.769, FG n/a, PUNT 0.729
  64,000 games (300 states): picks the truly best action in 76% (95% 71% to 81%); 93% where the true margin is over 3 points; mean regret 0.26 points (95% 0.19 to 0.34), 90th percentile 0.90; learned values off by +0.05 points on average; NLL 4.145 (frequency baseline 10.326)
    headline state (up 4, 4:00, 4th and 2 at own 40): learned GO 0.747, FG n/a, PUNT 0.724; true GO 0.769, FG n/a, PUNT 0.729

Run facts from results.json (as printed): seed 20261121; exact game built on 2010 to 2023 (637,139 rows, 3,793 games) plus the documented 2024 and 2025 kickoff rule tables; 407,854,656 states; 64,000 synthetic training games (10,068,362 records), 2,000 validation games; ground truth 300 states x 20,000 rollouts per action, coach-value standard error at most 0.0035.
