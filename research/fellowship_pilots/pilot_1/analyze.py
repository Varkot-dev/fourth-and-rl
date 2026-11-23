"""Pilot analysis: (1) decision-level proxy error at coach-visited fourth downs (held-out
synthetic seed 999), measured against the exact Q^coach (oracle q4); (2) table of exact gold
vs sqrt(KL); (3) least-squares fit of Gao et al.'s RL form gold = d (alpha - b log d), d = sqrt(KL)."""
import json, numpy as np
from common import load, BETAS, HERE
sol, coach = load(); m = sol.model; g = sol.grid
F4 = np.nonzero(g.unpack(np.arange(g.nf))[0] == g.n_down)[0]
d = np.load("dval.npz"); X = d["X"]; t, s, down, dbin, yl, a, b, r = X.T.astype(int)
sel = down == 4; t, s, dbin, yl, a, b, r = (x[sel] for x in (t, s, dbin, yl, a, b, r))
f = g.fidx(np.full(len(t), 4), dbin, yl) - F4[0]
def lookup(tag):
    out = np.empty((len(t), 3))
    for ph in ("end", "mid", "first"):
        lo, hi = m.phase_range(ph); k = (t >= lo) & (t <= hi)
        q = np.load(HERE / f"q4_{tag}_{ph}.npy", mmap_mode="r")
        e1 = a[k] if ph == "end" else (r[k] if ph == "first" else 0); e2 = b[k] if ph == "end" else 0
        out[k] = q[t[k] - lo, f[k], s[k] + m.score_max, e1, e2]
    return out
Qo = lookup("oracle"); fin = np.isfinite(Qo)
res = {"n_fourth_down_states": int(len(t))}
for tag in ("1k_small", "1k_large", "16k_small", "16k_large"):
    Qp = lookup(tag)
    ok = fin.sum(1) >= 2
    Ao = np.where(fin, Qo, -np.inf); Ap = np.where(fin, Qp, -np.inf)
    io, ip = Ao.argmax(1), Ap.argmax(1)
    regret = Ao.max(1) - Ao[np.arange(len(t)), ip]
    # advantage error: proxy gap between best-by-proxy and true-best action vs true gap
    claimed = Ap.max(1) - Ap[np.arange(len(t)), io]
    res[tag] = dict(argmax_disagree=float(np.mean(io[ok] != ip[ok])), mean_regret_greedy=float(regret[ok].mean()),
                    p95_regret=float(np.quantile(regret[ok], 0.95)), max_regret=float(regret[ok].max()),
                    q_rmse=float(np.sqrt(np.nanmean(np.where(fin, (Qp - Qo) ** 2, np.nan)))))
curves = {}
for tag in ("oracle", "1k_small", "1k_large", "16k_small", "16k_large"):
    try:
        sv = json.load(open(f"pass_{tag}.json"))["start_value"]; mc = json.load(open(f"mc_{tag}.json"))["by_beta"]
    except FileNotFoundError:
        continue
    rows = [dict(beta=bt, sqrt_kl=float(np.sqrt(mc[str(bt)]["kl_mean"])), kl=mc[str(bt)]["kl_mean"], kl_se=mc[str(bt)]["kl_se"],
                 gold_gain=sv[f"beta_{bt}"] - 0.5, proxy_claim=mc[str(bt)]["claimed_gain"], claim_se=mc[str(bt)]["claimed_se"],
                 mc_check=mc[str(bt)]["mc_winA"] - 0.5, mc_check_se=mc[str(bt)]["mc_winA_se"]) for bt in BETAS]
    dd = np.array([x["sqrt_kl"] for x in rows]); y = np.array([x["gold_gain"] for x in rows])
    A = np.stack([dd, -dd * np.log(dd)], 1); coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    curves[tag] = dict(rows=rows, gao_fit=dict(alpha=float(coef[0]), b=float(coef[1]), rmse=float(np.sqrt(np.mean((pred - y) ** 2))),
                       d_peak=float(np.exp(coef[0] / coef[1] - 1)) if coef[1] > 0 else None),
                       peak_beta=rows[int(np.argmax(y))]["beta"], peak_gold=float(y.max()), last_gold=float(y[-1]))
res["curves"] = curves
json.dump(res, open("analysis.json", "w"), indent=1); print(json.dumps(res, indent=1))
