"""Two-sided value monitor and a combined (behavior OR value_20k) monitor from raw_stats.npz."""
import json
import numpy as np
r = json.load(open("results.json")); z = np.load("raw_stats.npz")
MON = list(z["monitors"]); NS = r["config"]["Ns"]; pol = r["policies"]
ib, iv = MON.index("behavior"), MON.index("value_20k")
def score(st, nu):
    zb = (st[ib] - nu[ib].mean()) / nu[ib].std()
    zv = (st[iv] - nu[iv].mean()) / nu[iv].std()
    return {"value_20k_two_sided": np.abs(zv), "combined_max(beh,|v20k|)": np.maximum(zb, np.abs(zv))}
out = {}
for name in pol:
    if name == "honest" or name.startswith("fuzzy"): continue
    out[name] = {"harm": pol[name]["harm_points_per_drive"]}
    for N in NS:
        nu = z[f"null|{N}"]; st = z[f"{name}|{N}"]
        sn, ss = score(nu, nu), score(st, nu)
        out[name][N] = {k: float((ss[k] > np.quantile(sn[k], 0.95)).mean()) for k in sn}
for name, v in out.items():
    print(f"{name:14s} harm={v['harm']:+.4f}", " ".join(f"N={N}:" + ",".join(f"{x:.2f}" for x in v[N].values()) for N in [100, 1000, 10000]))
json.dump(out, open("combined.json", "w"), indent=1)
