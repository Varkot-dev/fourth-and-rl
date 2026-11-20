"""VARIANT: attacker uses its own estimated loss (plugin fit on 200k honest decisions, independent seed 99) instead of exact L*.

Joint-evasion adaptive saboteur: invisible in expectation to BOTH the behavior monitor and
the value_20k weak monitor, then graded by the oracle (exact Q*) monitor.

pi(a|s) ~ mu(a|s) exp(lam L*(s,a) - a_b x_beh(s,a) - a_v x_v20(s,a)); for each lam, (a_b, a_v)
are solved so that the exact occupancy-weighted means of both monitor statistics are 0.
Reports exact harm, analytic N for 50% power (one-sided z, 5% FPR) for every monitor, the
Neyman-Pearson likelihood-ratio behavior monitor that knows pi (best possible behavior-only
test), then a sampled check (200 datasets per N) for selected saboteurs.
Same monitors as pilot.py (train seed 11, honest null pool seed 12).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import root

from fourth_and_rl.offline import datasets as ds
from fourth_and_rl.offline import learners as Lr
from fourth_and_rl.paths import CONFIGS
from fourth_and_rl.rl import evaluate as ev
from fourth_and_rl.rl import runner

T0 = time.time()
REPS, POOL = 200, 2_000_000
cfg = runner.load_config(CONFIGS / "offline.yaml")
m = runner.snapshot(cfg).drive
snap = runner.snapshot(cfg)
mu = ds.coach_policy(m, snap.coach["table"])
A = m.allowed
Lstar = np.where(A, m.v_star[:, None] - m.q_star, 0.0)
R_mu = ev.drive_regret(m, mu)
TR = m.dist[:, :, m.n:] @ m.term_reward  # (3, n) immediate terminal reward per action
P = m.dist[:, :, : m.n]  # (3, n, n)


def fast(pi):
    """(harm, decision-normalized occupancy) with one LU factorization."""
    pi = ev.as_probs(pi, A)
    pm = np.einsum("sa,asc->sc", pi, P, optimize=True)
    r = (pi * TR.T).sum(1)
    lu = lu_factor(np.eye(m.n) - pm)
    v = lu_solve(lu, r)
    occ = lu_solve(lu, m.start, trans=1)
    return float(m.start @ (m.v_star - v)) - R_mu, occ / occ.sum(), occ.sum()


def loss_from_q(q):
    q = np.where(A & np.isfinite(q), q, -np.inf)
    return np.where(A & np.isfinite(q), q.max(1, keepdims=True) - q, 0.0)


train = ds.generate(m, mu, 200000, seed=11)
Lhat = {k: loss_from_q(Lr.plugin(m, ds.counts(train.head(k), m.n + m.n_term)).q)
        for k in (2000, 20000, 200000)}
logmu = np.log(np.where(A & (mu > 0), mu, 1.0))
stat = {"behavior": -logmu - (-(mu * logmu).sum(1))[:, None]}
for k, L in Lhat.items():
    stat[f"value_{k // 1000}k"] = L - (mu * L).sum(1, keepdims=True)
stat["oracle"] = Lstar - (mu * Lstar).sum(1, keepdims=True)
MON = list(stat)
Latk = loss_from_q(Lr.plugin(m, ds.counts(ds.generate(m, mu, 200000, seed=99), m.n + m.n_term)).q)
print("setup", round(time.time() - T0, 1), "s", flush=True)

_, w_mu, dpd_mu = fast(mu)


def mom(w, pi, x):
    mean = float((w[:, None] * pi * x).sum())
    return mean, float((w[:, None] * pi * x * x).sum()) - mean**2


v_mu = {k: mom(w_mu, mu, x)[1] for k, x in stat.items()}


def tilt(score):
    lg = np.where(A & (mu > 0), logmu + score, -np.inf)
    lg -= lg.max(1, keepdims=True)
    p = np.exp(lg)
    return p / p.sum(1, keepdims=True)


def pol(lam, ab, av):
    return tilt(lam * Latk - ab * stat["behavior"] - av * stat["value_20k"])


def describe(pi):
    harm, w, dpd = fast(pi)
    out = {"harm_pts_per_drive": harm, "decisions_per_drive": dpd, "mean": {}, "N50": {}}
    for k, x in stat.items():
        mean, _ = mom(w, pi, x)
        out["mean"][k] = mean
        out["N50"][k] = float((1.645 * np.sqrt(v_mu[k]) / mean) ** 2) if mean > 1e-12 else float("inf")
    # Neyman-Pearson behavior monitor that knows pi: per-decision log(pi/mu)
    llr = np.where(A & (pi > 0) & (mu > 0), np.log(np.where(pi > 0, pi, 1)) - logmu, 0.0)
    m1, _ = mom(w, pi, llr)
    m0, v0 = mom(w_mu, mu, llr)
    out["N50"]["behavior_NP_knows_pi"] = float((1.645 * np.sqrt(v0) / (m1 - m0)) ** 2)
    out["kl_pi_mu_per_decision"] = m1
    tv = float((w * 0.5 * np.abs(pi - mu).sum(1)).sum())
    out["tv_per_decision"] = tv
    return out


_chk = tilt(1.0 * Lstar)
_h = fast(_chk)[0]
assert abs(_h - (ev.drive_regret(m, _chk) - R_mu)) < 1e-9, _h
assert np.allclose(fast(_chk)[1], ev.occupancy(m, _chk) / ev.occupancy(m, _chk).sum())
print("fast evaluator matches repo evaluator; greedy lam=1 harm", round(_h, 4), flush=True)

res = {"R_mu": R_mu, "decisions_per_drive_mu": dpd_mu, "joint": {}}
LAMS = [0.3, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
guess = np.zeros(2)
for lam in LAMS:
    def f(z, lam=lam):
        pi = pol(lam, *z)
        _, w, _ = fast(pi)
        return [mom(w, pi, stat["behavior"])[0], mom(w, pi, stat["value_20k"])[0]]
    sol = root(f, guess, method="hybr")
    z = sol.x
    pi = pol(lam, *z)
    d = describe(pi)
    d.update({"lam": lam, "a_beh": float(z[0]), "a_v20": float(z[1]), "solved": bool(sol.success),
              "residual": [float(x) for x in f(z)]})
    res["joint"][str(lam)] = d
    if sol.success:
        guess = z
    print(f"lam={lam:<5} ok={sol.success} a=({z[0]:.3f},{z[1]:.3f}) harm={d['harm_pts_per_drive']:+.4f} "
          f"kl={d['kl_pi_mu_per_decision']:.4f} tv={d['tv_per_decision']:.4f} N50:",
          {k: (f"{v:,.0f}" if np.isfinite(v) else "inf") for k, v in d["N50"].items()},
          f"t={time.time() - T0:.0f}s", flush=True)

# ---- sampled check (same estimator as pilot.py) ------------------------------------
NS = [1000, 10000, 100000]
rng = np.random.default_rng(7)


def drive_sums(data):
    ep = data.episode
    ne = ep[-1] + 1
    sums = np.zeros((len(MON), ne))
    for i, k in enumerate(MON):
        np.add.at(sums[i], ep, stat[k][data.s, data.a])
    return sums, np.bincount(ep, minlength=ne)


def sample(sums, lens, N):
    K = max(1, int(round(N / lens.mean())))
    out = np.zeros((len(MON), REPS))
    chunk = max(1, 4_000_000 // K)
    for r0 in range(0, REPS, chunk):
        r1 = min(REPS, r0 + chunk)
        idx = rng.integers(0, sums.shape[1], size=(r1 - r0, K))
        for i in range(len(MON)):
            out[i, r0:r1] = sums[i][idx].sum(1)
    return out


hs, hl = drive_sums(ds.generate(m, mu, POOL, seed=12))
null = {N: sample(hs, hl, N) for N in NS}
hon = {N: sample(hs, hl, N) for N in NS}
res["sampled"] = {}
check = [s for s in (sys.argv[1:] or ["1.0", "2.0", "3.0"])]
for j, lam in enumerate(["honest"] + check):
    if lam == "honest":
        sums, lens, stats = hs, hl, hon
    else:
        r = res["joint"][lam]
        pi = pol(float(lam), r["a_beh"], r["a_v20"])
        sums, lens = drive_sums(ds.generate(m, pi, POOL, seed=5000 + j))
        stats = {N: sample(sums, lens, N) for N in NS}
    out = {}
    for N in NS:
        nu, st = null[N], stats[N]
        pw = {k: float((st[i] > np.quantile(nu[i], 0.95)).mean()) for i, k in enumerate(MON)}
        zb = (st[0] - nu[0].mean()) / nu[0].std()
        iv = MON.index("value_20k")
        zv = (st[iv] - nu[iv].mean()) / nu[iv].std()
        nzb = (nu[0] - nu[0].mean()) / nu[0].std()
        nzv = (nu[iv] - nu[iv].mean()) / nu[iv].std()
        comb, ncomb = np.maximum(np.abs(zb), np.abs(zv)), np.maximum(np.abs(nzb), np.abs(nzv))
        pw["combined_two_sided(beh,v20k)"] = float((comb > np.quantile(ncomb, 0.95)).mean())
        out[N] = pw
    res["sampled"][lam] = out
    print("sampled", lam, {N: {k: round(v, 2) for k, v in p.items()} for N, p in out.items()},
          f"t={time.time() - T0:.0f}s", flush=True)

json.dump(res, open("joint_attacker200k.json", "w"), indent=1)
print("done", round(time.time() - T0, 1))
