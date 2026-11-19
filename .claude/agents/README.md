# Review agents

Three reviewer definitions used for independent review passes on this project:

- `mle-reviewer.md`: machine-learning review (leakage, point-in-time features,
  reproducibility, evaluation discipline).
- `silent-failure-hunter.md`: code that looks successful while skipping data or
  swallowing errors.
- `python-reviewer.md`: general Python review.

They are copied from the open-source repository `affaan-m/everything-claude-code`
(commit 8abdf329, MIT license, see `LICENSE-everything-claude-code`). The only change
is that the line pinning a specific model was removed so they use the session's model.
Nothing else from that repository (installer, hooks, plugins) is used here.
