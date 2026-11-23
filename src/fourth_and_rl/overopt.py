"""Research milestone R1: overoptimization of a learned value model, graded exactly.

Shared pieces of the R1 scripts (scripts/research/overopt_*.py): the solved game and its
coach table, synthetic coach-vs-coach games with every pre-snap state labelled by the
final result, the proxy features, and the policy arithmetic (the KL tilt, per-decision
KL, the greedy choice with explicit tie handling).

The tilted policy at a fourth down s with coach distribution P(a | s) over
(go, field goal, punt) and value estimates Q(s, a) is

    pi_beta(a | s) = P(a | s) exp(beta Q(s, a)) / sum over b of P(b | s) exp(beta Q(s, b))

which maximizes sum_a pi(a) Q(s, a) - KL(pi || P) / beta at that state: one step of soft
policy improvement on Q, applied independently at every fourth down. beta = 0 is the
coach; beta -> infinity is the greedy choice among the actions the coach ever takes.
Everything is exact for the fitted game MDP, not real football.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from fourth_and_rl import mdp_game as mg
from fourth_and_rl.paths import CONFIGS

FEATURES = ("t", "s", "down", "dbin", "yl", "a", "b", "r")


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    with open(path or CONFIGS / "overopt.yaml") as f:
        return yaml.safe_load(f)


def load_game(path: Path | str | None = None) -> tuple[mg.Solution, dict[str, Any]]:
    """The solved game (memory-mapped) and its coach table."""
    folder = Path(path) if path is not None else mg.SOLUTION_DIR
    sol = mg.load_solution(folder, mmap=True)
    return sol, joblib.load(folder / "tables.joblib")["coach"]


# --------------------------------------------------------------------------------------
# Synthetic games and proxy features
# --------------------------------------------------------------------------------------


def features(model: mg.GameModel, t: Any, s: Any, f: Any, a: Any, b: Any, r: Any) -> np.ndarray:
    """Proxy features (FEATURES) from the side of the team with the ball, float32 rows.
    Timeouts exist only in the last 5:00 (3 each before), the second-half flag only in
    the first half (0 after), so every state of the solved game has one feature row."""
    g, c = model.kern.grid, model.clock
    t, s, f, a, b, r = np.broadcast_arrays(*(np.asarray(x) for x in (t, s, f, a, b, r)))
    down, dbin, yl = g.unpack(f)
    end = t <= c.endgame_ticks
    a = np.where(end, a, c.n_timeouts)
    b = np.where(end, b, c.n_timeouts)
    r = np.where(t > c.th, r, 0)
    return np.stack([t, s, down, dbin, yl, a, b, r], -1).reshape(-1, 8).astype(np.float32)


class _Recorder(mg.Simulator):
    """Simulator that logs every snap: (game, t, s, f, a, b, r, side)."""

    def advance(self, st: dict, idx: np.ndarray, act: np.ndarray, rng: np.random.Generator,
                post: str = "model") -> dict:  # fmt: skip
        self.log.append(np.stack([idx, *(st[k][idx] for k in ("t", "s", "f", "a", "b", "r",
                                                               "side"))], 1))  # fmt: skip
        return super().advance(st, idx, act, rng, post)


def kickoff_start(sim: mg.Simulator, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """n games at the opening kickoff, tied: the receiver's field position is drawn from
    the kickoff table, and the receiver kicks off the second half (flag 0)."""
    k = sim.model.kern
    p = k.kickoff[1:] / k.kickoff[1:].sum()
    f = sim.grid.first10()[1:][rng.choice(len(p), size=n, p=p)]
    return sim.start(n, t=sim.model.clock.n_ticks, s=0, f=f, r=0)


def coach_games(sol: mg.Solution, coach: dict[str, Any], n: int, seed: int,
                post: str = "coach") -> dict[str, np.ndarray]:  # fmt: skip
    """n synthetic coach-vs-coach games from kickoff. Returns X (features of every snap),
    y (final result for the team with the ball: 1, 0.5 or 0), game (index), and
    result (for the opening receiver)."""
    sim = _Recorder(sol, coach=coach)
    sim.log = []
    rng = np.random.default_rng(seed)
    st = kickoff_start(sim, n, rng)
    res = mg.simulate(sim, st, rng, policy="coach", post=post)["result"]
    rows = np.concatenate(sim.log)
    g, t, s, f, a, b, r, side = rows.T
    y = np.where(side == 0, res[g], 1.0 - res[g])
    return {"X": features(sol.model, t, s, f, a, b, r), "y": y.astype(np.float32),
            "game": g.astype(np.int32), "result": res}  # fmt: skip


# --------------------------------------------------------------------------------------
# Policy arithmetic over (go, field goal, punt)
# --------------------------------------------------------------------------------------


def tilt(p: np.ndarray, q: np.ndarray, beta: float) -> np.ndarray:
    """pi_beta proportional to p * exp(beta q) over the last axis. Actions with p = 0 get
    0 (whatever q is); q is float32 or float64, never float16."""
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        logp = np.log(np.where(p > 0, p, 1.0))
        z = np.where(p > 0, beta * np.asarray(q, dtype=np.float64) + logp, -np.inf)
        z = z - z.max(-1, keepdims=True)
        w = np.where(p > 0, np.exp(z), 0.0)
    return w / w.sum(-1, keepdims=True)


def kl(pi: np.ndarray, p: np.ndarray) -> np.ndarray:
    """KL(pi || p) over the last axis, in nats (0 log 0 = 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lpi = np.log(np.where(pi > 0, pi, 1.0))
        lp = np.log(np.where(p > 0, p, 1.0))
        x = np.where(pi > 0, pi * (lpi - lp), 0.0)
    return x.sum(-1)


def greedy(q: np.ndarray, allowed: np.ndarray, rule: str = "uniform", atol: float = 0.0
           ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """The greedy choice over the last axis among `allowed` actions, with ties explicit.

    Returns (probabilities, tied): actions within `atol` of the best allowed value are
    tied. rule="uniform" spreads the mass evenly over the tied actions; rule="exclude"
    does the same but callers drop the states where `tied` is True from any diagnostic
    that needs a single best action. A state with no allowed action gets all zeros."""
    if rule not in ("uniform", "exclude"):
        raise ValueError(f"tie rule must be 'uniform' or 'exclude', not {rule!r}")
    qa = np.where(allowed, np.asarray(q, dtype=np.float64), -np.inf)
    top = qa.max(-1, keepdims=True)
    hit = allowed & (qa >= top - atol)
    n = hit.sum(-1, keepdims=True)
    probs = np.where(n > 0, hit / np.maximum(n, 1), 0.0)
    return probs, n[..., 0] > 1


def disagreement(q_true: np.ndarray, q_proxy: np.ndarray, allowed: np.ndarray,
                 rule: str = "uniform", atol: float = 0.0) -> dict[str, float]:  # fmt: skip
    """How often the proxy's greedy action differs from the true greedy action, and the
    true value it gives up, over states with at least two allowed actions.

    rule="uniform": a state counts by the chance that a uniform draw from the proxy's tied
    set is not in the true tied set; regret is the expected true shortfall of that draw.
    rule="exclude": states where either side has a tie are left out."""
    two = allowed.sum(-1) >= 2
    pt, tied_t = greedy(q_true, allowed, rule, atol)
    pp, tied_p = greedy(q_proxy, allowed, rule, atol)
    keep = two & ~(tied_t | tied_p) if rule == "exclude" else two
    miss = (pp * (pt == 0)).sum(-1)
    qt = np.where(allowed, np.asarray(q_true, dtype=np.float64), -np.inf)
    best = qt.max(-1)
    regret = best - np.where(pp > 0, pp * np.where(allowed, q_true, 0.0), 0.0).sum(-1)
    return {"n_states": int(keep.sum()), "n_tied_true": int((two & tied_t).sum()),
            "n_tied_proxy": int((two & tied_p).sum()),
            "disagree": float(miss[keep].mean()) if keep.any() else float("nan"),
            "mean_regret": float(regret[keep].mean()) if keep.any() else float("nan")}  # fmt: skip


# --------------------------------------------------------------------------------------
# Fourth-down action values at every tick, and the policies built on them
# --------------------------------------------------------------------------------------


class FourthQ:
    """Action values of (go in the coach's mode, field goal, punt) at every fourth-down
    state of every tick, float32: one array per phase of shape
    (ticks in phase, n_fourth, n_score, E1, E2, 3). -inf marks an action that is not
    allowed; NaN a tick not filled yet. The full real game takes 1.2 GB."""

    def __init__(self, model: mg.GameModel) -> None:
        self.model = model
        g = model.kern.grid
        self.n4 = g.n_dist * g.n_yl
        self.f0 = (g.n_down - 1) * self.n4
        self.q: dict[str, np.ndarray] = {}
        for ph in ("end", "mid", "first"):
            lo, hi = model.phase_range(ph)
            if hi >= lo:
                shape = (hi - lo + 1, self.n4, model.n_score, *model.extra_shape(ph), 3)
                self.q[ph] = np.full(shape, np.nan, dtype=np.float32)

    def put(self, t: int, q3: np.ndarray) -> None:
        ph = self.model.phase(t)
        self.q[ph][t - self.model.phase_range(ph)[0]] = q3

    def get(self, t: int) -> np.ndarray:
        ph = self.model.phase(t)
        return self.q[ph][t - self.model.phase_range(ph)[0]]

    def lookup(self, t: np.ndarray, f: np.ndarray, s: np.ndarray, a: np.ndarray,
               b: np.ndarray, r: np.ndarray) -> np.ndarray:  # fmt: skip
        """Values (n, 3) at fourth-down states given as arrays (s is the clipped score)."""
        m = self.model
        out = np.full((len(t), 3), np.nan, dtype=np.float64)
        si = s + m.score_max
        for ph, arr in self.q.items():
            lo, hi = m.phase_range(ph)
            sel = (t >= lo) & (t <= hi)
            if not sel.any():
                continue
            e = m.extra_shape(ph)
            if ph == "end":
                e1, e2 = np.minimum(a[sel], e[0] - 1), np.minimum(b[sel], e[1] - 1)
            else:
                e1 = r[sel] if ph == "first" else np.zeros(int(sel.sum()), dtype=np.int64)
                e2 = np.zeros(int(sel.sum()), dtype=np.int64)
            out[sel] = arr[t[sel] - lo, f[sel] - self.f0, si[sel], e1, e2]
        return out


def tilt_policy(coach: Any, q4: FourthQ, beta: float) -> Any:
    """pi_beta at unforced fourth downs (the coach everywhere else): the coach's
    (go, field goal, punt) distribution tilted by exp(beta Q) with Q from `q4`."""

    def fourth(t: int, q: dict) -> np.ndarray:
        p = coach.slice(t).p[coach.f4][:, :, None, None, :]
        return tilt(p, q4.get(t), beta)

    return coach.with_fourth(fourth)


class TiltSimulator(mg.Simulator):
    """Simulator in which team A plays pi_beta at its unforced fourth downs (from a stored
    `FourthQ`) and the coach everywhere else; team B is the coach.

    Per game it accumulates A's sequence KL(pi_beta || coach) (`kl`), the sum over A's
    decisions of sum_a (pi_beta - coach)(a) Q(a) for each `FourthQ` in `claims` (`claim`),
    and the number of A's free fourth downs (`decisions`). Set `a_side` (per game, the
    `side` that is team A) and `beta` before `simulate`, and call `reset(n)`."""

    def __init__(self, sol: mg.Solution, coach: dict[str, Any], q4: FourthQ,
                 claims: dict[str, FourthQ] | None = None) -> None:  # fmt: skip
        super().__init__(sol, coach=coach)
        from fourth_and_rl.policy_eval import CoachPolicy

        self.cp = CoachPolicy(sol.model, coach)
        self.q4 = q4
        self.claims = claims or {}
        self.beta = 0.0
        self.a_side = np.zeros(0, dtype=np.int64)

    def reset(self, n: int) -> None:
        self.kl = np.zeros(n)
        self.decisions = np.zeros(n)
        self.claim = {k: np.zeros(n) for k in self.claims}

    def coach_actions(self, st: dict[str, np.ndarray], idx: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:  # fmt: skip
        act = super().coach_actions(st, idx, rng)
        t, s, f = st["t"][idx], st["s"][idx], st["f"][idx]
        cs = self.cp.at(t, f, s)
        mine = st["side"][idx] == self.a_side[idx]
        sel = np.nonzero(cs.free_fourth & mine)[0]
        if len(sel) == 0:
            return act
        a, b, r = (st[k][idx][sel] for k in ("a", "b", "r"))
        q = self.q4.lookup(t[sel], f[sel], s[sel], a, b, r)
        p = cs.p[sel]
        pi = tilt(p, q, self.beta)
        game = idx[sel]
        np.add.at(self.kl, game, kl(pi, p))
        np.add.at(self.decisions, game, 1.0)
        for name, qc in self.claims.items():
            qq = qc.lookup(t[sel], f[sel], s[sel], a, b, r)
            with np.errstate(invalid="ignore"):
                gain = (np.where(pi > 0, pi * qq, 0.0) - np.where(p > 0, p * qq, 0.0)).sum(-1)
            np.add.at(self.claim[name], game, gain)
        choice = (rng.random(len(sel))[:, None] > np.cumsum(pi, axis=1)).sum(axis=1)
        choice = np.minimum(choice, 2)
        act[sel] = np.where(choice == 1, mg.A_FG, np.where(choice == 2, mg.A_PUNT, cs.base[sel]))
        return act


def tilt_monte_carlo(sim: TiltSimulator, beta: float, n: int, seed: int,
                     post: str = "coach") -> dict[str, float]:  # fmt: skip
    """n games of A (pi_beta) against the coach from the opening kickoff, A receiving it
    in the first half of the batch. Returns means and standard errors of A's result,
    sequence KL, claims and decisions per game."""
    rng = np.random.default_rng(seed)
    st = kickoff_start(sim, n, rng)
    sim.a_side = np.where(np.arange(n) < n // 2, 0, 1)
    sim.beta = beta
    sim.reset(n)
    res = mg.simulate(sim, st, rng, policy="coach", post=post)["result"]
    res_a = np.where(sim.a_side == 0, res, 1.0 - res)

    def ms(x: np.ndarray) -> tuple[float, float]:
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))

    out = {"games": n, "beta": beta}
    for name, x in (("win_a", res_a), ("kl", sim.kl), ("decisions", sim.decisions),
                    *((f"claim_{k}", v) for k, v in sim.claim.items())):  # fmt: skip
        out[name], out[f"{name}_se"] = ms(x)
    return out


def record_fourth_q(pair: Any, coach: Any, q4: FourthQ, team: str = "A") -> Any:
    """An `after(t)` callback for `policy_eval.run` that stores the team's own fourth-down
    action values of each tick in `q4` (float32)."""
    from fourth_and_rl.policy_eval import fourth_q

    m = pair.model

    def after(t: int) -> None:
        shape = (m.kern.grid.nf, m.n_score, *m.extra_shape(m.phase(t)))
        q4.put(t, fourth_q(pair.last_q(team), coach.slice(t), coach.f4, shape))

    return after


class TiltStats:
    """Forward-pass hook: exact expectations over team A's unforced fourth downs under
    pi_beta (built on `q4`): sequence KL from the coach (`kl`), the number of decisions
    (`decisions`), and for each named FourthQ in `claims` the performance-difference sum
    E[sum over decisions of sum_a (pi_beta - coach)(a) Q(a)] (`claim`). With Q the exact
    coach action values this sum is the true gain over the coach exactly; with the
    proxy's own Q it is the gain the proxy believes in."""

    def __init__(self, coach: Any, q4: FourthQ, beta: float,
                 claims: dict[str, FourthQ] | None = None) -> None:  # fmt: skip
        self.coach, self.q4, self.beta = coach, q4, beta
        self.claims = claims or {}
        self.kl = 0.0
        self.decisions = 0.0
        self.claim = {k: 0.0 for k in self.claims}

    def __call__(self, team: str, t: int, mass: np.ndarray) -> None:
        if team != "A" or t == 0:
            return
        f4 = self.coach.f4
        cs = self.coach.slice(t)
        w = mass[f4] * cs.free_fourth[f4][:, :, None, None]
        if not w.any():
            return
        p = cs.p[f4][:, :, None, None, :]
        pi = tilt(p, self.q4.get(t), self.beta)
        self.kl += float((w * kl(pi, p)).sum())
        self.decisions += float(w.sum())
        for name, qc in self.claims.items():
            q = qc.get(t).astype(np.float64)
            with np.errstate(invalid="ignore"):
                gain = np.where(p > 0, (pi - p) * q, 0.0).sum(-1)
            self.claim[name] += float((w * gain).sum())
