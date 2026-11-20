"""Proxy error against the exact coach value V_coach on held-out synthetic states (seed 999)."""
import json, time, numpy as np, joblib, lightgbm as lgb
from common import load, HERE
sol, coach = load(); m = sol.model; g = sol.grid
vc = joblib.load(HERE.parent / "v_coach.joblib", mmap_mode="r")
d = np.load("dval.npz"); rng = np.random.default_rng(0); i = rng.choice(len(d["y"]), 100000, replace=False)
X = d["X"][i]; t, s, down, dbin, yl, a, b, r = X.T.astype(int)
f = g.fidx(down, dbin, yl)
true = np.empty(len(t))
for ph in ("end", "mid", "first"):
    lo, hi = m.phase_range(ph); sel = (t >= lo) & (t <= hi)
    e1 = a[sel] if ph == "end" else (r[sel] if ph == "first" else 0); e2 = b[sel] if ph == "end" else 0
    true[sel] = vc[ph][t[sel] - lo, f[sel], s[sel] + m.score_max, e1, e2]
out = {}
for name in ("1k_small", "1k_large", "16k_small", "16k_large"):
    bst = lgb.Booster(model_file=f"proxy_{name}.txt")
    t0 = time.time(); p = bst.predict(X, num_threads=1); dt = time.time() - t0
    e = p - true; q4 = down == 4; late = t <= 60
    out[name] = dict(rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.mean(np.abs(e))), rmse_4th=float(np.sqrt(np.mean(e[q4]**2))),
                     rmse_last5min=float(np.sqrt(np.mean(e[late]**2))), pred_s_per_1M_rows_1thread=round(dt * 10, 2))
out["label_noise_rmse_vs_true"] = float(np.sqrt(np.mean((d["y"][i] - true) ** 2)))
json.dump(out, open("diag.json", "w"), indent=1); print(json.dumps(out, indent=1))
