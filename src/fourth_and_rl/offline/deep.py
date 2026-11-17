"""d3rlpy adapters: offline DoubleDQN (naive deep offline Q-learning) and DiscreteCQL.

Checked against the installed d3rlpy 2.8.0 source:
- `d3rlpy.dataset.MDPDataset(observations, actions, rewards, terminals, timeouts=...)`
  takes flat arrays in episode order. Row i holds the observation, the action taken and
  the reward received after it; `terminals[i] = 1` on the last step of an episode that
  ended, `timeouts[i] = 1` on the last row of one that was cut. An episode that was cut
  contributes one transition less than it has rows (the last row has no next
  observation), so a cut episode gets an extra row holding its next observation.
- DoubleDQN's TD error is a Huber loss with threshold 1 (`compute_huber_loss`). One
  drive's return has a standard deviation near 2.8 points, so on raw points most errors
  would fall in the linear part and the fit would drift toward a median, not a mean.
  Rewards are therefore multiplied by `reward_scale` (0.1) for training and the
  predictions divided by it, which keeps nearly every error in the quadratic part.
- DiscreteCQL adds alpha * mean(logsumexp_a Q(s, a) - Q(s, a_data)) to the DoubleDQN
  loss, on the scaled values. Its alpha is in those units and is the library default.
- The target uses max over all three actions of the target network: d3rlpy has no action
  mask, so a field goal that is not allowed in the next state can still feed the target.
  Learned values are read for every state with `predict_value` and the greedy action is
  taken among allowed actions only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fourth_and_rl.offline.datasets import Logged
from fourth_and_rl.offline.learners import Fit, greedy_policy
from fourth_and_rl.rl.envs import DriveModel


def to_mdp_dataset(model: DriveModel, data: Logged, reward_scale: float) -> Any:
    """The logged transitions as a d3rlpy MDPDataset on the M5 feature vector."""
    from d3rlpy.dataset import MDPDataset

    feats = model.features()
    n = model.n
    done = data.done
    first = data.first
    obs, act, rew, term, tout = [], [], [], [], []
    ends = np.nonzero(np.append(first[1:], True))[0]
    begin = 0
    for end in ends:
        sl = slice(begin, end + 1)
        s, a, c = data.s[sl], data.a[sl], data.c[sl]
        r = np.where(c >= n, model.term_reward[np.maximum(c - n, 0)], 0.0)
        obs.append(feats[s])
        act.append(a.astype(np.int64))
        rew.append(r * reward_scale)
        k = len(s)
        t = np.zeros(k)
        o = np.zeros(k)
        if done[end]:
            t[-1] = 1.0
        else:
            # cut short: add the next observation as a row of its own (never trained on)
            obs.append(feats[c[-1:]])
            act.append(np.zeros(1, dtype=np.int64))
            rew.append(np.zeros(1))
            t = np.append(t, 0.0)
            o = np.append(o, 1.0)
        term.append(t)
        tout.append(o)
        begin = end + 1
    return MDPDataset(
        observations=np.concatenate(obs).astype(np.float32),
        actions=np.concatenate(act),
        rewards=np.concatenate(rew).astype(np.float32)[:, None],
        terminals=np.concatenate(term).astype(np.float32),
        timeouts=np.concatenate(tout).astype(np.float32),
        action_size=3,
    )


def fit_d3rlpy(
    model: DriveModel,
    data: Logged,
    algo: str,
    seed: int,
    cfg: dict[str, Any],
    threads: int = 1,
) -> Fit:
    """Train "dqn" (DoubleDQN) or "cql" (DiscreteCQL) on the logged data. `cfg` holds
    hidden_units, learning_rate, batch_size, target_update_interval, n_steps, alpha,
    reward_scale. Returns the learned Q-values in points for every state."""
    import d3rlpy
    import torch

    torch.set_num_threads(threads)
    d3rlpy.seed(seed)
    scale = float(cfg["reward_scale"])
    dataset = to_mdp_dataset(model, data, scale)
    common: dict[str, Any] = {
        "batch_size": int(cfg["batch_size"]),
        "learning_rate": float(cfg["learning_rate"]),
        "gamma": 1.0,
        "target_update_interval": int(cfg["target_update_interval"]),
        "encoder_factory": d3rlpy.models.VectorEncoderFactory(
            hidden_units=list(cfg["hidden_units"])
        ),
    }
    if algo == "dqn":
        learner = d3rlpy.algos.DoubleDQNConfig(**common).create(device="cpu:0")
    elif algo == "cql":
        learner = d3rlpy.algos.DiscreteCQLConfig(alpha=float(cfg["alpha"]), **common).create(
            device="cpu:0"
        )
    else:
        raise ValueError("algo must be 'dqn' or 'cql'")
    n_steps = int(cfg["n_steps"])
    learner.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps,
        show_progress=False,
        save_interval=10**9,
        logger_adapter=d3rlpy.logging.NoopAdapterFactory(),
    )
    feats = model.features().astype(np.float32)
    q = np.column_stack(
        [learner.predict_value(feats, np.full(model.n, a, dtype=np.int64)) for a in range(3)]
    ) / scale
    q = np.where(model.allowed, q, -np.inf)
    name = "dqn_d3rlpy" if algo == "dqn" else "cql_d3rlpy"
    return Fit(name, greedy_policy(q, model.allowed), q, {"n_steps": n_steps, **cfg})
