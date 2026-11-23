import time, numpy as np
t=time.time()
from fourth_and_rl.rl import runner, evaluate as ev
from fourth_and_rl.offline import datasets as ds
from fourth_and_rl.paths import CONFIGS
cfg = runner.load_config(CONFIGS/"offline.yaml")
snap = runner.snapshot(cfg)
m = snap.drive
print("load", time.time()-t, m.n)
mu = ds.coach_policy(m, snap.coach["table"])
print("regret coach", ev.drive_regret(m, mu), "opt", ev.drive_regret(m, m.policy_star))
L = np.where(m.allowed, m.v_star[:,None]-m.q_star, np.nan)
occ = ev.occupancy(m, mu)
print("decisions/drive", occ.sum(), "p_reach", m.p_reach)
exp_loss = np.nansum(mu*np.nan_to_num(L),1)
print("per-decision mean loss under coach", (occ*exp_loss).sum()/occ.sum())
t=time.time(); d = ds.generate(m, mu, 1_000_000, seed=1); print("gen 1M", time.time()-t)
from fourth_and_rl.offline import learners as Lr
for n in [2000, 20000, 200000]:
    t=time.time(); c = ds.counts(d.head(n), m.n+m.n_term); fit = Lr.plugin(m, c); print(n, "plugin", time.time()-t, fit.info)
gap = np.sort(np.where(m.allowed, m.q_star, -np.inf),1)
g = gap[:,2]-gap[:,1]
w = occ/occ.sum()
for tau in [0.05,0.1,0.2,0.5]: print("share of coach decisions in states with gap<",tau, w[g<tau].sum())
print("coach P(suboptimal action)", (w*(1-mu[np.arange(m.n), m.policy_star])).sum())
