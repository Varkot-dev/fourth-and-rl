# M7 notes: the interactive pieces (app bundle, Streamlit app, static page, notebook, animation)

Code: `src/fourth_and_rl/app_bundle.py` (bundle schema, loader, lookups),
`src/fourth_and_rl/app_figures.py` (Plotly figures), `app/streamlit_app.py`,
`app/static_page_template.html`. Config: `configs/app.yaml`. Scripts:
`scripts/m7_build_app_bundle.py`, `scripts/m7_build_static_page.py`,
`scripts/m7_value_animation.py`, `scripts/m7_app_screenshot.py` (with `m7_screenshot.js`).
Notebook: `notebooks/fourth_down_demo.ipynb`. Tests: `tests/test_app.py`.
Every number below comes from `app/data/bundle_meta.json` (written by the bundle script)
unless marked otherwise.

## 1. The problem: the models cannot ship

The one-step engine needs the fitted win-probability and play-outcome models, the
bootstrap needs 101 refitted pairs of them (129 MB), and the exact answer needs the solved
game (about 2 GB of value and policy arrays). All of it lives under `data/`, which is not
committed. So the app, the static page and the notebook cannot compute anything; they read a
**precomputed bundle**: every situation of a fixed grid evaluated once, offline, by the same
code that produced the M2, M3 and M4 results.

This is a classic trade of compute for storage. The cost is that the grid is finite: the app
can only show situations that were computed. The rule kept here is that nothing is
interpolated; controls snap to grid values and the app says when it snapped.

## 2. Concepts

### A dense grid and its index arithmetic

The bundle stores arrays shaped [context, score, distance - 1, yardline_100 - 1], where a
context is (game clock, opponent timeouts, spread). Looking up a situation is index
arithmetic, no search: with 87 contexts, 29 scores, 15 distances and 99 yard lines, the
flat position of (c, s, d, y) is ((c * 29 + s) * 15 + d) * 99 + y. The static page's
JavaScript does exactly this.

Worked example: Q4 4:00, up 4, opponent 3 timeouts, even spread, fourth and 2 at the own
40. The context list puts the end-game clocks after the five early ones (each with three
spreads, so 15 contexts), and each end-game clock has 4 timeout values times 3 spreads = 12
contexts. Q4 5:00 takes contexts 15 to 26, Q4 4:00 starts at 27; 3 timeouts is the fourth
timeout value (offset 9), even spread the middle (offset 1): context 37. Score +4 is index
18 (scores start at -14), distance 2 is index 1, yard line 60 is index 59.

Cells that cannot happen (fourth and 8 at the opponent's 5) are stored as "not available",
which costs about 7% of the array but keeps the arithmetic trivial.

### Quantization

A win probability is stored as an unsigned 16-bit integer in units of 1/10,000 (0.01
win-probability points), with 65535 meaning "not available". Gains (go minus the better
kick) are signed 16-bit in the same unit. The decision categories never depend on that
resolution: the smallest distinction the project reports is a tenth of a point. Bootstrap
counts (how many of the 101 replicates prefer each action) fit in 8 bits.

### Delta coding and why it compresses

Neighbouring yard lines have nearly equal win probabilities, so the differences along the
yard-line axis are small numbers clustered near zero, which a general-purpose compressor
(LZMA here) stores in far fewer bits than the raw values. The differences are exact
integers (int32), so a cumulative sum gives back every value, sentinels included.

Worked example: go values 7489, 7495, 7502, 7508 become 7489, 6, 7, 6. In the first trial
the plain zlib file was 11.5 MB for 38 contexts; LZMA alone took it to about 0.58 of that
and LZMA with deltas to about 0.3.

### Bootstrap summaries per cell

For each cell the 101 replicate engines give three win probabilities each. The bundle keeps
(a) the count of replicates whose best action is each of go, field goal and punt, from which
boot% and the confident / lean / too-close-to-call category follow by the M3 rule; (b) the
6th and 96th sorted values of the go gain (a 90% interval, as in M3); (c) the same order
statistics for each action's win probability. The per-action intervals move together
across replicates (a refit that is more pessimistic about the whole situation lowers all
three), so they are much wider than the interval on the difference, and they do not decide
the call; the app says so next to them.

### Two engines side by side

The app shows the one-step answer and the exact answer for every cell, plus a sentence on
why they can differ (see `docs/notes/m4_mdp.md`, "Why one-step lookahead differs from the
DP"): who plays the rest of the game (real coaches inside a fitted win-probability model
versus optimal play), resolution of the clock and timeouts, and model error on either side.
The exact game also ignores the spread (identical teams), and treats 11 to 15 yards to go as
one bin valued at 13.

## 3. Decisions and why

- **Grid.** 11 clocks (Q1, Q2, Q3 at 10:00; Q4 at 15:00 and 10:00; then 5:00, 4:00, 3:00,
  2:00, 1:00 and 0:30), scores -14 to +14 at every point, every yard line, 1 to 15 yards to
  go, opponent timeouts 0 to 3 in the last 5:00 (3 before, as the exact game has no timeouts
  earlier), spreads -7, 0, +7 at every clock. The team with the ball holds 3 timeouts and
  does not receive the second-half kickoff, in both engines. 87 contexts, 3,481,740 cells.
- **The spread is a grid axis at every clock (a corrected assumption).** The first trial
  stored spreads only up to the start of the fourth quarter, assuming that the model's
  spread term decays to nothing late. The build script's spread check showed the opposite: a
  7-point spread still moves one-step values late in the game by up to 11 to 16
  win-probability points (the largest change at each clock from 2:00 to 10:00 left) and
  changes the best action in up to 18% of a chart's cells, because the version 2
  win-probability model carries the spread in its structural baseline (an expected final
  margin), not only in the decaying feature. The check stays in the bundle as a diagnostic.
- **Bootstrap on every cell, not a subset.** A recommendation without its confidence category
  invites exactly the over-reading M3 warns about, so every cell has its 101 replicates. That
  is the cost driver: about 157 seconds per replicate on 3 threads, 4.4 hours in all, cached
  per replicate so a stopped build resumes.
- **NumPy .npz, not parquet.** The grid is dense, so arrays with named axes are the natural
  format, lookups are index arithmetic, and the app needs only NumPy. (The repository's
  leakage test also flags every `read_parquet` call under `scripts/` so that no script reads
  the play table unguarded; a bundle in parquet would have needed an allowlist entry.)
- **The static page is a slice.** All three spreads would make a 19 MB page; it holds the
  even spread only (7.0 MB, every score), says so, and points to the app for the rest. It
  also drops the per-action intervals. Data travel as one gzip block, unpacked in the browser
  with DecompressionStream; plotly comes from its CDN pinned to the installed version.
- **The notebook's RL part uses the toy drive model.** The real environments need the fitted
  tables. The toy has the same structure and an exact optimum, so regret is exact; the
  notebook says it is a toy.

## 4. Results

From the final build of 2026-11-18: `app/data/bundle_meta.json`, the build log
(`data/app_build/full_build.log`, not committed), the static page script's output, the
notebook run and a one-off spot check run by hand (its numbers are below; the check itself
is not a script in the repository).

- **The bundle.** 87 one-step contexts and 29 exact contexts, 29 scores, 15 distances, 99
  yard lines: 3,746,655 cells per array, of which 3,481,740 can occur. Every cell carries the
  one-step values, the 101 bootstrap refits and the exact game's values. Four files, 15.63
  MB in all: `one_step.npz` 14.96 MB, `exact.npz` 0.59 MB, `questions.json` 0.05 MB,
  `bundle_meta.json` 0.03 MB.
- **Build time.** 15,927 seconds (4.4 hours) on 3 threads, peak memory 6.9 GB. Point
  estimates 191 s; the replicates 15,676 s (100 computed in this run, 136 to 185 s each,
  mean 157; the first was cached by an earlier, stopped run); the exact values 8 s. The
  per-replicate caches under `data/app_build/` take 4.0 GB and are not committed.
- **Checks written into the bundle.** Encoding round trip: largest error 5.0e-5, half the
  1/10,000 step. Exact values against the saved question 1 map: 19,840 cells, largest
  difference 5.0e-5 (tolerance 1e-4).
- **Spot check against the sources.** Five cells (Q1 10:00 tied at the own 35, spread -7;
  Q3 10:00 down 3 at the opponent's 38, spread +7; the question 1 headline; Q4 2:00 down 3 at
  the opponent's 30 with the opponent out of timeouts; Q4 0:30 down 7, fourth and goal from
  the 13, spread -7). One-step engine re-evaluated on those states: largest difference
  4.3e-5. Exact game read through `Solution.q_values` (not the build's array path): 4.1e-5.
  Bootstrap counts recomputed from the per-replicate caches: identical; go-gain interval:
  4.5e-5. Question 1 headline from the bundle: go 0.7784, punt 0.7139 (saved 0.778 and
  0.714), one-step go minus punt +4.82 points (saved +4.82).
- **The spread check.** One-step values at spread -7 or +7 minus spread 0, at Q4 10:00, 5:00
  and 2:00 for scores -7, -3, 0, +3, +7 (opponent with 3 timeouts). Largest change 16.3
  points (Q4 10:00, tied, favoured by 7); the largest per clock is 16.3 at 10:00, 13.5 at
  5:00 and 11.4 at 2:00, and the mean change over a chart is 7.1 to 14.0 points at 10:00
  and 1.6 to 8.5 at 2:00. The best action changes in up to 17.8% of a chart's possible cells
  (Q4 10:00, down 3, favoured by 7). This is why the spread is an axis at every clock.
- **The static page.** 7.03 MB (budget 10 MB), holding the even-spread slice: 29 contexts,
  every score. The data block is 5.25 MB of gzip (7.01 MB as base64); the page builds in
  about 5 seconds. All three spreads would make 18.9 MB. Plotly comes from
  `https://cdn.plot.ly/plotly-4.1.1.min.js`, the installed plotly.js version (4.1.1, on npm
  since 2026-09-14; the npm file is byte-identical to the installed one, and the plotly.js
  README gives `cdn.plot.ly/plotly-<exact version>.min.js` as the CDN form). The build
  machine cannot reach the CDN, so the screenshot script serves the local copy.
- **Screenshots.** `assets/m7_app.png` (the app's first tab at the default situation, Q4
  4:00, up 4, fourth and 2 at the own 40: one step ahead go 74.9% and punt 70.1%, 100 of 101
  refits agree, go minus punt +4.8 points with interval +0.8 to +6.4; exact game go 77.8%
  and punt 71.4%) and `assets/m7_static_page.png` (the page at phone width, same
  situation). The action chart beside the answer was changed from bars to dots with their
  intervals: its axis is cropped to the values, and bars on a cropped axis exaggerate the
  gaps.
- **Notebook.** Executed end to end against this bundle by nbclient in 6.8 seconds (three
  Q-learning runs of 400,000 decisions on the toy drive model in 2 s, final regret 0.0 for
  each seed); the outputs and the two figures are saved in the notebook.
