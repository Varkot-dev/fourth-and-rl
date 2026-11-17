# My questions

This branch is for the questions I want answered for myself. It is built on top of the core branch (`claude/optimistic-mendel-v2267n`, later `main`), which keeps the research and reinforcement-learning milestones. The core plan in `docs/PLAN.md` still applies here for code quality, honesty rules and commits.

## How this branch works

- Bring core work in regularly: `git merge <core branch>`. Never merge this branch back into the core unless I decide a finished result belongs in the main README.
- Reuse the core's data pipeline, win-probability model, play-outcome models, bootstrap and, once built, the full-game solver. Do not fork them; if one needs a change, make it on the core branch first.
- Each question gets a script under `scripts/questions/`, results under `results/questions/`, a figure under `assets/questions/`, and a short write-up in `docs/questions/<name>.md` with the answer, how confident it is, and what would change it.
- New ideas go to `docs/research/inbox/` on this branch.
- The X research kit (`research-kit/x/`) and its findings feed this branch.

## The questions, in the order I want them

1. **Why do offenses run when passing gains more?** Passes gain more per play on average, yet teams run about 40% of the time. Measure the gap by down, distance and situation, then test the game-theory explanation (what matters is the value of the next run versus the next pass, and defenses adjust) with the run-or-pass version of the decision model.
2. **Are the Lions right to go for it so often?** Detroit has the highest fourth-down go rate in the league since 2021. Does a strong offense justify more aggression than a league-average model recommends, and by how much? Needs a team-strength adjustment built only from games before the one being judged.
3. **Does the model know my team?** "Does the analytics take into account their defense is terrible?" Asked constantly, answered informally. Measured by how far the recommendation moves with offense and defense strength, built from prior games only (overlaps question 3).
4. **Was it really fourth and 1?** Next Gen Stats called the Lions' fourth and 1 from their own 24 "closer to a 4th and 2", turning a very strong go into a toss-up. How often does half a yard of hidden distance flip the call?
5. **Case studies from the public debate** (`research-kit/x/2_twitter_findings.md`): the ten most argued decisions, each scored by the one-step engine, the bootstrap and the full-game solution, next to what the public models said at the time. Several are from the 2026 season, which no model in this repo has seen, so they are a clean out-of-sample check.
6. **Which coach has cost his team the most wins, and does the data back up "malpractice"?** (The owner's idea, `docs/research/inbox/coach_cost_idea.md`.) Take the calls the public labelled malpractice and show what each one cost by the models. Then build a per-coach ledger in wins: fourth-down, field-goal and clock decisions, plus the procedure penalties the owner assigns to the coach (false start, delay of game), and explicitly not encroachment, defensive offside or neutral-zone infraction. See section 3.6.
7. Further questions from the owner's video list, when supplied.

## Ideas in the inbox

- `docs/research/inbox/coach_cost_idea.md`: which coach has cost his team the most wins; false starts and delay of game count against the coach, encroachment and offside do not.
- `docs/research/inbox/referee_and_coach_prior_work.md`: referee crews and existing work on coach rankings.
- From the X findings: the Surrender Index, penalties as coaching decisions, named coaches (Todd Bowles, Sean Payton reply thread).

## Order of work

Questions that need only what the core has already built come first (run versus pass rates, the Lions go rate, the hidden half yard, the coach ledger using the one-step engine). Questions that need the full-game solver or the bootstrap wait until the core has them.
