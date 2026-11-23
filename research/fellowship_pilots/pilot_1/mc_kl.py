"""Monte Carlo over games of team A (pi_beta at fourth downs, from the stored proxy Q) against
coach team B: sequence-level KL(pi_beta || coach) per game, the proxy's claimed gain
(performance-difference sum of proxy advantages), and A's result as a check on the exact value.
usage: mc_kl.py tag n_games seed"""
import sys, json, time, numpy as np
from fourth_and_rl import mdp_game as mg
from common import load, coach_fourth, tilt, kl, BETAS, HERE
from gen_data import kickoff_start
tag, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sol, coach = load(); model = sol.model; g = sol.grid
F4 = np.nonzero(g.unpack(np.arange(g.nf))[0] == g.n_down)[0]
q4 = {ph: np.load(HERE / f"q4_{tag}_{ph}.npy", mmap_mode="r") for ph in ("end", "mid", "first")}

class Sim(mg.Simulator):
    def coach_actions(self, st, idx, rng):
        act = super().coach_actions(st, idx, rng)
        t, s, f = st["t"][idx], st["s"][idx], st["f"][idx]
        base, P, forced, fourth = coach_fourth(model, coach, t, s, f)
        isA = st["side"][idx] == self.a_side[idx]
        sel = np.nonzero(fourth & (forced < 0) & isA)[0]
        if len(sel) == 0:
            return act
        ts, ss, fs = t[sel], s[sel] + model.score_max, f[sel] - F4[0]
        a, b, r = st["a"][idx][sel], st["b"][idx][sel], st["r"][idx][sel]
        Q = np.empty((len(sel), 3))
        ph = self._phase_code(ts)
        for code, name in enumerate(("end", "mid", "first")):
            m_ = ph == code
            if not m_.any(): continue
            lo = model.phase_range(name)[0]
            e1 = np.minimum(a[m_], 3) if name == "end" else (r[m_] if name == "first" else 0)
            e2 = np.minimum(b[m_], 3) if name == "end" else 0
            Q[m_] = q4[name][ts[m_] - lo, fs[m_], ss[m_], e1, e2].astype(np.float64)
        Ps = P[sel]
        pi = tilt(Ps, Q, self.beta)
        gid = idx[sel]
        np.add.at(self.kl, gid, kl(pi, Ps))
        with np.errstate(invalid="ignore"):
            adv = np.where(pi > 0, pi * Q, 0).sum(-1) - np.where(Ps > 0, Ps * Q, 0).sum(-1)
        np.add.at(self.adv, gid, adv); np.add.at(self.nd, gid, 1)
        ch = (rng.random(len(sel))[:, None] > np.cumsum(pi, 1)).sum(1).clip(0, 2)
        act[sel] = np.where(ch == 1, mg.A_FG, np.where(ch == 2, mg.A_PUNT, base[sel]))
        return act

sim = Sim(sol, coach=coach); out = {}
for beta in BETAS:
    rng = np.random.default_rng(seed); t0 = time.time()
    st = kickoff_start(sim, n, rng)
    sim.a_side = np.where(np.arange(n) < n // 2, 0, 1)  # A receives the opening kickoff in half the games
    sim.beta = beta; sim.kl = np.zeros(n); sim.adv = np.zeros(n); sim.nd = np.zeros(n)
    res = mg.simulate(sim, st, rng, policy="coach", post="model")["result"]
    resA = np.where(sim.a_side == 0, res, 1 - res)
    se = lambda x: float(x.std(ddof=1) / np.sqrt(n))
    out[str(beta)] = dict(kl_mean=float(sim.kl.mean()), kl_se=se(sim.kl), claimed_gain=float(sim.adv.mean()), claimed_se=se(sim.adv),
                          mc_winA=float(resA.mean()), mc_winA_se=se(resA), a_decisions_per_game=float(sim.nd.mean()), secs=round(time.time() - t0, 1))
    print(beta, out[str(beta)], flush=True)
json.dump(dict(tag=tag, n=n, seed=seed, by_beta=out), open(HERE / f"mc_{tag}.json", "w"), indent=1)
