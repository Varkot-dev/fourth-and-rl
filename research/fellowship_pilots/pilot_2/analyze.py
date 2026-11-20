"""Undetected exact harm vs N per monitor and saboteur family, from results.json."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

r = json.load(open("results.json"))
cfg = r["config"]
NS = [str(n) for n in cfg["Ns"]]
pol = r["policies"]
MON = list(pol["honest"]["N"][NS[0]].keys())
FAM = ["greedy", "fuzzy", "adaptive"]
POWER_MAX = 0.5

print("honest regret (points/drive):", round(r["honest_regret_points_per_drive"], 4))
print("coach share of decisions in fuzzy states:", round(r["coach_share_in_fuzzy_states"], 3))
print("monitor errors:", json.dumps(r["monitor_errors"], indent=0))
print("\nhonest false positive rate (should be ~0.05):")
for N in NS:
    print(N, {k: pol["honest"]["N"][N][k]["power"] for k in MON})

out = {}
for fam in FAM:
    names = [k for k in pol if k.startswith(fam + "_")]
    harm = np.array([pol[k]["harm_points_per_drive"] for k in names])
    print(f"\n== {fam}: budgets lam -> harm (pts/drive), KL/decision")
    for k, h in zip(names, harm):
        print(f"  {k:14s} harm={h:+.4f} kl={pol[k]['kl_per_decision']:.4f} "
              f"tv={pol[k]['tv_per_decision']:.4f}")
    out[fam] = {}
    print(f"   max harm with detection power <= {POWER_MAX} at 5% FPR "
          "[bracket: next budget's harm]")
    print("   N       " + " ".join(f"{k:>18s}" for k in MON))
    for N in NS:
        row = []
        out[fam][N] = {}
        for k in MON:
            pw = np.array([pol[n]["N"][N][k]["power"] for n in names])
            ok = (pw <= POWER_MAX) & (harm > 0)
            h = float(harm[ok].max()) if ok.any() else 0.0
            above = harm[(pw > POWER_MAX) & (harm > h)]
            nxt = float(above.min()) if len(above) else float("nan")
            out[fam][N][k] = (h, nxt)
            row.append(f"{h:7.4f} [{nxt:7.4f}]")
        print(f"   {N:7s} " + " ".join(f"{x:>18s}" for x in row))

json.dump(out, open("undetected.json", "w"), indent=1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
colors = {"behavior": "#888888", "value_2k": "#f4a259", "value_20k": "#e76f51",
          "value_200k": "#9d4edd", "oracle": "#2a9d8f"}
x = np.array([int(n) for n in NS])
for ax, fam in zip(axes, FAM):
    for k in MON:
        y = np.array([out[fam][N][k][0] for N in NS])
        ax.plot(x, y, "o-", color=colors[k], label=k)
    ax.set_xscale("log")
    ax.set_title(f"saboteur: {fam}")
    ax.set_xlabel("logged fourth-down decisions N")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("exact harm that evades (power<=0.5, 5% FPR)\nexpected points per drive")
axes[0].legend(fontsize=8)
fig.suptitle("Pilot: undetected exact harm vs data, drive model (lower = better monitor)")
fig.tight_layout()
fig.savefig("undetected_harm_vs_N.png", dpi=130)
print("\nfigure: undetected_harm_vs_N.png")
