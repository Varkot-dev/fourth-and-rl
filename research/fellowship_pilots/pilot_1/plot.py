"""Figure: exact gold gain (V_A - 0.5) and proxy-claimed gain against sqrt(KL to coach)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import BETAS
tags = [("oracle", "exact V_coach (no proxy error)", "#2a78d6"), ("16k_large", "LightGBM 64 leaves, 16k games", "#1baf7a"), ("16k_small", "LightGBM 8 leaves, 16k games", "#7a5bd6"), ("1k_small", "LightGBM 8 leaves, 1k games", "#c9a227"), ("1k_large", "LightGBM 64 leaves, 1k games", "#eb6834")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b"})
fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
po = json.load(open("pass_oracle.json"))["start_value"]
br = 100 * (po["br_inf"] - 0.5)
ax.axhline(0, color="#52514e", lw=1); ax.text(0.02, 0.15, "coach (beta = 0)", color="#52514e", fontsize=8)
ax.axhline(br, color="#52514e", lw=1, ls=":"); ax.text(0.02, br + 0.12, "best fourth-down response to coach (exact)", color="#52514e", fontsize=8)
rows = []
for tag, label, col in tags:
    try:
        sv = json.load(open(f"pass_{tag}.json"))["start_value"]; mc = json.load(open(f"mc_{tag}.json"))["by_beta"]
    except FileNotFoundError:
        continue
    x = np.sqrt([mc[str(b)]["kl_mean"] for b in BETAS]); gold = [100 * (sv[f"beta_{b}"] - 0.5) for b in BETAS]
    claim = [100 * mc[str(b)]["claimed_gain"] for b in BETAS]
    ax.plot(x, gold, "-o", color=col, lw=2, ms=6, mec="#fcfcfb", mew=1.5, label=f"{label}: exact gold")
    ax.plot(x, claim, "--", color=col, lw=1.5, alpha=0.8, label=f"{label}: proxy claim")
    for b, xi, gi in zip(BETAS, x, gold):
        rows.append((tag, b, xi, gi))
    ax.annotate(f"beta={int(BETAS[-1])}", (x[-1], gold[-1]), textcoords="offset points", xytext=(4, -10), fontsize=7, color="#52514e")
ax.set_xlabel("sqrt(KL(pi_beta || coach)) per game, nats^0.5 (Monte Carlo, 20k games)")
ax.set_ylabel("team A win probability gain vs coach, points of WP")
ax.set_title("KL-regularized optimization of a learned value model, graded by the exact MDP value", fontsize=10, loc="left")
ax.legend(fontsize=7, frameon=False, loc="lower right"); ax.grid(axis="y", color="#e6e5e0", lw=0.6)
fig.text(0.01, 0.005, "Exact for the fitted game MDP, not real football. Pilot: fourth downs only, coach elsewhere; timeouts and tries model-optimal.", fontsize=6.5, color="#52514e")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("goodhart_pilot.png", dpi=160)
for r in rows: print(r)
