"""Hand-written REINFORCE on the drive problem.

Policy gradient. For a policy pi_theta and an episode tau = (s_0, a_0, ..., s_T, a_T)
with return G (here: the single terminal reward),

    grad J(theta) = E[ sum over t of  G * grad log pi_theta(a_t | s_t) ]

and any baseline b(s_t) that does not depend on a_t can be subtracted from G without
changing the expectation, because E[grad log pi(a | s)] = sum over a of grad pi(a | s) = 0.
A good baseline (close to the value of s_t) removes most of the variance.

Tabular softmax policy: pi(a | s) = exp(theta[s, a]) / sum over allowed b of exp(theta[s, b]),
for which grad log pi(a | s) with respect to theta[s, .] is onehot(a) - pi(. | s). So one
step of the update after an episode is

    theta[s_t, b] += lr * (G - b(s_t)) * (1{b = a_t} - pi(b | s_t))     for every allowed b
    b(s_t)        += beta * (G - b(s_t))                                (a running value estimate)

The network version replaces theta by a small multilayer perceptron on the state features
and a second network for the baseline; the loss whose gradient is the REINFORCE estimator
is written out by hand and stepped with Adam.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from fourth_and_rl.rl.envs import CompiledDrive, DriveBatch, DriveModel


@dataclass
class ReinforceConfig:
    lr: float = 0.05  # policy step size
    lr_decay: float = 0.0  # lr_t = lr / (1 + lr_decay * episodes / 1e6)
    baseline: bool = True
    baseline_lr: float = 0.05  # beta: step size of the per-state running value
    logit_clip: float = 30.0  # keeps exp() finite once the policy is nearly deterministic


def _softmax(row: list[float], vs: list[int]) -> list[float]:
    m = max(row[a] for a in vs)
    e = [math.exp(row[a] - m) for a in vs]
    z = sum(e)
    return [x / z for x in e]


def reinforce_tabular(
    env: CompiledDrive,
    n_steps: int,
    seed: int,
    cfg: ReinforceConfig | None = None,
    checkpoints: list[int] | None = None,
    on_checkpoint: Callable[[np.ndarray, int, int], Any] | None = None,
) -> dict[str, Any]:
    """Tabular REINFORCE for about `n_steps` decisions (it stops at the end of the episode
    in which the budget runs out). `on_checkpoint(probs, steps, episodes)` receives the
    current policy as an (n, 3) array of probabilities."""
    cfg = cfg or ReinforceConfig()
    rng = np.random.default_rng(seed)
    stream = rng.random(1 << 20).tolist()
    pos = 0

    def nxt() -> float:
        nonlocal stream, pos
        if pos == len(stream):
            stream, pos = rng.random(1 << 20).tolist(), 0
        pos += 1
        return stream[pos - 1]

    n_s, valid = env.n_states, env.valid
    theta = [[0.0, 0.0, 0.0] for _ in range(n_s)]
    base = [0.0] * n_s
    marks = sorted(set(checkpoints or []))
    next_mark = marks.pop(0) if marks else None
    records: list[Any] = []
    clip = cfg.logit_clip

    def probs_array() -> np.ndarray:
        out = np.zeros((n_s, 3))
        for s in range(n_s):
            vs = valid[s]
            for a, p in zip(vs, _softmax(theta[s], vs), strict=True):
                out[s, a] = p
        return out

    t, episodes = 0, 0
    while t < n_steps:
        s = env.reset(nxt())
        traj: list[tuple[int, int, list[float]]] = []
        while True:
            vs = valid[s]
            p = _softmax(theta[s], vs)
            u, k, acc = nxt(), 0, p[0]
            while u > acc and k < len(vs) - 1:
                k += 1
                acc += p[k]
            a = vs[k]
            traj.append((s, k, p))
            t += 1
            s, r, done = env.step(s, a, nxt())
            if done:
                break
        episodes += 1
        lr = cfg.lr / (1.0 + cfg.lr_decay * episodes / 1e6)
        for s_t, k, p in traj:
            adv = r - base[s_t] if cfg.baseline else r
            row = theta[s_t]
            vs = valid[s_t]
            for j, b in enumerate(vs):
                row[b] += lr * adv * ((1.0 if j == k else 0.0) - p[j])
                if row[b] > clip:
                    row[b] = clip
                elif row[b] < -clip:
                    row[b] = -clip
            if cfg.baseline:
                base[s_t] += cfg.baseline_lr * (r - base[s_t])
        while next_mark is not None and t >= next_mark:
            if on_checkpoint is not None:
                records.append(on_checkpoint(probs_array(), t, episodes))
            next_mark = marks.pop(0) if marks else None
    return {
        "probs": probs_array(),
        "baseline": np.array(base),
        "records": records,
        "episodes": episodes,
        "steps": t,
    }


def gradient_variance(
    model: DriveModel,
    env: CompiledDrive,
    probs: np.ndarray,
    baseline: np.ndarray,
    n_episodes: int,
    seed: int,
) -> dict[str, float]:
    """How noisy is the one-episode REINFORCE gradient at a fixed policy, with no baseline
    and with the given state baseline? Returns the total variance (sum over all
    parameters of the variance of the per-episode gradient) for each, and their ratio.
    The mean gradient is the same in both cases (checked: `mean_gap_relative`)."""
    rng = np.random.default_rng(seed)
    n, k = model.n, 3
    g_sum = {"none": np.zeros((n, k)), "base": np.zeros((n, k))}
    g_sq = {"none": np.zeros((n, k)), "base": np.zeros((n, k))}
    cum = np.cumsum(probs, axis=1)
    for _ in range(n_episodes):
        s = env.reset(rng.random())
        steps = []
        while True:
            a = int(min((rng.random() > cum[s]).sum(), 2))
            while not model.allowed[s, a]:
                a -= 1
            steps.append((s, a))
            s, r, done = env.step(s, a, rng.random())
            if done:
                break
        for key, use in (("none", False), ("base", True)):
            g = {}
            for s_t, a_t in steps:
                score = -probs[s_t].copy()
                score[a_t] += 1.0
                adv = r - (baseline[s_t] if use else 0.0)
                g[s_t] = g.get(s_t, 0.0) + adv * score
            for s_t, vec in g.items():
                g_sum[key][s_t] += vec
                g_sq[key][s_t] += vec * vec
    out = {}
    for key in g_sum:
        mean = g_sum[key] / n_episodes
        out[key] = float((g_sq[key] / n_episodes - mean**2).sum())
    m0, m1 = g_sum["none"] / n_episodes, g_sum["base"] / n_episodes
    return {
        "total_variance_no_baseline": out["none"],
        "total_variance_with_baseline": out["base"],
        "variance_ratio": out["base"] / out["none"],
        "mean_gap_relative": float(np.abs(m0 - m1).sum() / max(np.abs(m0).sum(), 1e-12)),
        "n_episodes": int(n_episodes),
    }


# --------------------------------------------------------------------------------------
# Policy network (PyTorch)
# --------------------------------------------------------------------------------------


@dataclass
class NetReinforceConfig:
    hidden: int = 64
    lr: float = 3e-3
    baseline_lr: float = 3e-3
    episodes_per_update: int = 1024
    baseline: bool = True
    entropy_coef: float = 0.0


def reinforce_network(
    model: DriveModel,
    n_steps: int,
    seed: int,
    cfg: NetReinforceConfig | None = None,
    checkpoints: list[int] | None = None,
    on_checkpoint: Callable[[np.ndarray, int, int], Any] | None = None,
) -> dict[str, Any]:
    """REINFORCE with a two-layer policy network and a value network as baseline.

    Each update plays `episodes_per_update` fresh drives to the end with `DriveBatch`,
    then minimizes

        policy loss   = - mean over decisions of  (G - b(s)) * log pi(a | s)   (b detached)
        baseline loss =   mean over decisions of  (G - b(s))^2

    Illegal actions are masked out of the softmax (logit set to -1e9), so the network
    never puts probability on them.
    """
    import torch

    cfg = cfg or NetReinforceConfig()
    torch.manual_seed(seed)
    feats = torch.as_tensor(model.features())
    mask = torch.as_tensor(model.allowed)
    d_in = feats.shape[1]

    def mlp(d_out: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(d_in, cfg.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(cfg.hidden, cfg.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(cfg.hidden, d_out),
        )

    pol, val = mlp(3), mlp(1)
    opt_p = torch.optim.Adam(pol.parameters(), lr=cfg.lr)
    opt_v = torch.optim.Adam(val.parameters(), lr=cfg.baseline_lr)

    def all_probs() -> np.ndarray:
        with torch.no_grad():
            logits = pol(feats).masked_fill(~mask, -1e9)
            return torch.softmax(logits, dim=1).numpy().astype(float)

    batch = DriveBatch(model, cfg.episodes_per_update, seed=seed)
    rng = np.random.default_rng(seed + 1)
    marks = sorted(set(checkpoints or []))
    next_mark = marks.pop(0) if marks else None
    records: list[Any] = []
    t, episodes = 0, 0
    while t < n_steps:
        batch.reset()
        probs = all_probs()
        cum = np.cumsum(probs, axis=1)
        alive = np.ones(batch.n, dtype=bool)
        states, actions, owner = [], [], []
        ret = np.zeros(batch.n)
        while alive.any():
            idx = np.nonzero(alive)[0]
            s = batch.s[idx]
            a = np.minimum((rng.random(len(idx))[:, None] > cum[s]).sum(axis=1), 2)
            states.append(s)
            actions.append(a)
            owner.append(idx)
            full = np.zeros(batch.n, dtype=np.int64)
            full[idx] = a
            sub_s = batch.s.copy()
            r, done = batch.step(full)
            # Slots already finished were stepped too; restore them (their result is kept).
            batch.s = np.where(alive, batch.s, sub_s)
            ret[idx] += r[idx]
            alive[idx[done[idx]]] = False
        s_all = torch.as_tensor(np.concatenate(states))
        a_all = torch.as_tensor(np.concatenate(actions))
        g_all = torch.as_tensor(ret[np.concatenate(owner)], dtype=torch.float32)
        t += len(s_all)
        episodes += batch.n
        x = feats[s_all]
        logits = pol(x).masked_fill(~mask[s_all], -1e9)
        logp = torch.log_softmax(logits, dim=1)
        chosen = logp.gather(1, a_all[:, None]).squeeze(1)
        b = val(x).squeeze(1)
        adv = g_all - b.detach() if cfg.baseline else g_all
        loss = -(adv * chosen).mean()
        if cfg.entropy_coef:
            ent = -(logp.exp() * logp.masked_fill(~mask[s_all], 0.0)).sum(dim=1).mean()
            loss = loss - cfg.entropy_coef * ent
        opt_p.zero_grad()
        loss.backward()
        opt_p.step()
        if cfg.baseline:
            vloss = ((g_all - b) ** 2).mean()
            opt_v.zero_grad()
            vloss.backward()
            opt_v.step()
        while next_mark is not None and t >= next_mark:
            if on_checkpoint is not None:
                records.append(on_checkpoint(all_probs(), t, episodes))
            next_mark = marks.pop(0) if marks else None
    return {"probs": all_probs(), "records": records, "episodes": episodes, "steps": t,
            "policy_net": pol, "value_net": val}
