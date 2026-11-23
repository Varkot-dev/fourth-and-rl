# Fellowship pilots (exploratory)

Small CPU pilots run on 2026-11-20 to rank research directions. They are scratch-quality: one seed per proxy, light verification, and known issues listed in `docs/plans/fellowship_research_directions.md` (section 4). Nothing here is quoted in the main README.

- `pilot_1/`: exact Goodhart curves. KL-tilted fourth-down policies built on learned win-probability proxies, evaluated exactly against the solved game MDP. Data was synthetic games from the solved MDP; the large intermediate arrays (about 2.9 GB) were not kept.
- `pilot_2/`: diffuse sabotage on the drive MDP. Can monitors detect a policy that quietly loses expected points? The critic found the headline result was largely built in by construction.
- `exact_policy_eval.py`: prototype exact evaluation of a fixed policy over the whole game, to be promoted into `src/` with tests.

Scripts were written against absolute scratch paths; they need path edits to rerun.
