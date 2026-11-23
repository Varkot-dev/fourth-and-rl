# Teaching notes

Written while building, for the teaching pass at the end. One section per milestone:
the concepts it used, the decisions made and why, and what the results showed.

## M0: data pipeline

**Concepts**
- Play-by-play data: one row per play, with the game state before the play and what happened.
- "Posteam" point of view: every quantity (score differential, spread, win label) is
  expressed for the team with the ball, so one model serves both teams.
- Why a cached, typed table: every later milestone reads the same file, so results are
  comparable and reruns are fast.

**Decisions**
- Seasons 2010 to 2025 (16 seasons, 4,363 games, about 730,000 plays). Earlier seasons
  are a different game (fewer passes, different overtime and kickoff rules).
- Regular season and playoffs are both kept. Playoff overtime rules differ, which only
  matters for the handful of plays in overtime.
- Tied games get a null win label instead of 0.5, and are dropped when fitting win
  probability. There is about one per season.
- A fourth-down "decision" is labelled only when the play actually ran. Penalty
  no-plays are left out, which slightly undercounts decisions.
- The team that receives the second-half kickoff is worked out from the opening kickoff,
  because the raw file does not carry it. It matters for first-half win probability:
  a guaranteed extra possession after halftime is worth something.

**Results** (from `results/m0_data_summary.json`)
- Game counts match the schedule exactly: 267 per season through 2019, 269 in 2020,
  285 from 2021, and 284 in 2022 (one cancelled game).
- Field goals made: 84.5%. Fourth-down conversions when going for it: 51.5%.
- Fourth-down go rate rose from 12.6% in 2010 to 23.3% in 2025.

## Where the rest of the notes are

From M1 on, each milestone's teaching notes have their own file under `docs/notes/`:

| Milestone | Notes |
|---|---|
| M1 win probability (versions 1 and 2) | `docs/notes/m1_wp.md` |
| M1 neural network and deep ensemble | `docs/notes/m1_wp_nn.md` |
| M1 play-outcome models | `docs/notes/m1_components.md` |
| M2 decision engine | `docs/notes/m2_decision.md` |
| M3 bootstrap | `docs/notes/m3_bootstrap.md` |
| M3b blown leads | `docs/notes/m3b_blown_leads.md` |
| M4 MDP and dynamic programming | `docs/notes/m4_mdp.md` |
| M5 reinforcement learning | `docs/notes/m5_rl.md` |
| M6 offline reinforcement learning | `docs/notes/m6_offline.md` |
| Question 2, run versus pass | `docs/notes/q2_run_vs_pass.md` |
| M8 "does the model know our defense is terrible?" (and question 3) | `docs/notes/m8_team_strength.md` |
| M9 "was it really fourth and 1?" | `docs/notes/m9_distance_error.md` |
| M13 goal-line field goals and two-point tries | `docs/notes/m13_goal_line_and_two_point.md` |
| M10 "how much does being predictable cost an offense?" | `docs/notes/m10_predictability.md` |
| M7 coach evaluation, coach cost ledger, penalty assumption test, malpractice check (part one) | `docs/notes/m7_coaches.md` |
| M7 interactive pieces: app bundle, Streamlit app, static page, Colab notebook, value-function animation | `docs/notes/m7_app.md` |
| Field-goal selection bias: kicker model, decomposition, support distance, planner interface (post hoc, after the world-model run) | `docs/notes/fg_selection_bias.md` |
| R1 exact overoptimization of a learned value model (days 1 to 5) | `docs/notes/overopt.md` |

The design decisions and their history are in the Decision log at the bottom of `docs/PLAN.md`.

