"""M5: reinforcement-learning agents measured against the exact optimum of the solved model.

Modules
-------
envs       the two environments (single drive in expected points, last five minutes in win
           probability), as Gymnasium environments and as fast batch versions
tabular    hand-written Q-learning, SARSA and Expected SARSA (NumPy and plain Python)
reinforce  hand-written REINFORCE: tabular softmax policy and a small PyTorch network
evaluate   exact policy evaluation, regret, agreement with the optimum, sample efficiency
sb3_utils  adapters that let Stable-Baselines3 train on the batch environments
toy        tiny hand-made models with a known optimum, for the tests

One color per algorithm, used in every figure of the milestone.
"""

from __future__ import annotations

from fourth_and_rl import style

ALGO_COLORS = {
    "q_learning": style.BLUE,
    "q_learning_optimistic": style.SEQUENTIAL[2],
    "sarsa": style.SEQUENTIAL[5],
    "expected_sarsa": style.YELLOW,
    "reinforce": style.ORANGE,
    "reinforce_no_baseline": style.INK_2,
    "reinforce_net": style.SEQUENTIAL[6],
    "dqn": style.VIOLET,
    "ppo": style.AQUA,
    "optimal": style.INK,
    "coach": style.MUTED,
    "never_go": style.NEUTRAL,
    "random": style.RED,
}
ALGO_LABELS = {
    "q_learning": "Q-learning (tabular)",
    "q_learning_optimistic": "Q-learning, optimistic start",
    "sarsa": "SARSA (tabular)",
    "expected_sarsa": "Expected SARSA (tabular)",
    "reinforce": "REINFORCE with baseline (tabular)",
    "reinforce_no_baseline": "REINFORCE, no baseline (tabular)",
    "reinforce_net": "REINFORCE, policy network",
    "dqn": "DQN (Stable-Baselines3)",
    "ppo": "PPO (Stable-Baselines3)",
    "optimal": "DP optimum",
    "coach": "Coach-like policy",
    "never_go": "Never go for it",
    "random": "Random actions",
}
