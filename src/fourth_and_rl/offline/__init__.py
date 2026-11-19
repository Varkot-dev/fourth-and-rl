"""M6: offline reinforcement learning on logged fourth-down decisions.

The question: real data comes from coaches' choices, not from exploration. What happens
when a policy is learned from such logs, and does a conservative method fix it?

Modules
-------
datasets   behavior policies, logged datasets generated in the known-truth drive model,
           and the count tensor every tabular learner works from
learners   behavior cloning, fitted Q iteration (tabular and gradient-boosted trees),
           count-based pessimism, conservative Q-learning (hand-written, tabular), the
           model-based plug-in, fitted Q evaluation
deep       adapters for the d3rlpy library (DoubleDQN as naive offline Q-learning,
           DiscreteCQL); imported lazily, so the rest needs neither torch nor d3rlpy
measure    exact measurement in the known-truth model: regret, Q error by data count,
           the optimizer's curse, where policies differ
real       logged fourth-down transitions from real play-by-play, and off-policy
           evaluation (importance sampling, weighted, doubly robust)
figures    the four M6 figures

One color per method, used in every figure of the milestone. Methods that already
appeared in M5 keep their M5 color (DQN, the DP optimum, the coach-like policy).
"""

from __future__ import annotations

from fourth_and_rl import style
from fourth_and_rl.rl import ALGO_COLORS

METHOD_COLORS = {
    "behavior": ALGO_COLORS["coach"],
    "bc": style.INK_2,
    "bc_mode": style.SEQUENTIAL[6],
    "fqi_tabular": ALGO_COLORS["q_learning"],
    "fqi_gbt": style.SEQUENTIAL[2],
    "fqi_lcb": style.YELLOW,
    "cql_tabular": style.ORANGE,
    "plugin": style.AQUA,
    "abstain": style.SEQUENTIAL[4],
    "dqn_d3rlpy": ALGO_COLORS["dqn"],
    "cql_d3rlpy": style.RED,
    "optimal": ALGO_COLORS["optimal"],
}
METHOD_LABELS = {
    "behavior": "Behavior policy (the logger)",
    "bc": "Behavior cloning (stochastic)",
    "bc_mode": "Behavior cloning, most common action",
    "fqi_tabular": "Fitted Q iteration (tabular)",
    "fqi_gbt": "Fitted Q iteration (boosted trees)",
    "fqi_lcb": "FQI with count-based pessimism",
    "cql_tabular": "Conservative Q-learning (tabular)",
    "plugin": "Model-based plug-in",
    "abstain": "Plug-in, else the coach when unsure",
    "dqn_d3rlpy": "Offline DoubleDQN (d3rlpy)",
    "cql_d3rlpy": "Discrete CQL (d3rlpy)",
    "optimal": "DP optimum",
}
