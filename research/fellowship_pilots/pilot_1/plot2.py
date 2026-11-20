"""Two-panel pilot figure: exact gold gain and proxy-claimed gain against sqrt(KL to coach)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import BETAS
tags = [("oracle", "exact V_coach (no proxy error)", "#2a78d6"), ("16k_large", "64 leaves, 16k games", "#1baf7a"),
        ("16k_small", "8 leaves, 16k games", "#7a5bd6"), ("1k_small", "8 leaves, 1k games", "#c9a227"), ("1k_large", "64 leaves, 1k games", "#eb6834")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e"})
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="#fcfcfb", sharex=True)
br = 100 * (json.load(open("pass_oracle.json"))["start_value"]["br_inf"] - 0.5)
for ax in axs:
    ax.set_facecolor("#fcfcfb"); ax.axhline(0, color="#52514e", lw=1); ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.axhline(br, color="#52514e", lw=1, ls=":")
axs[0].text(0.05, br + 0.15, "exact best fourth-down response to coach", fontsize=7, color="#52514e")
for tag, label, col in tags:
    sv = json.load(open(f"pass_{tag}.json"))["start_value"]; mc = json.load(open(f"mc_{tag}.json"))["by_beta"]
    x = np.sqrt([mc[str(b)]["kl_mean"] for b in BETAS])
    axs[0].plot(x, [100 * (sv[f"beta_{b}"] - 0.5) for b in BETAS], "-o", color=col, lw=2, ms=5, mec="#fcfcfb", label=label)
    axs[1].plot(x, [100 * mc[str(b)]["claimed_gain"] for b in BETAS], "--o", color=col, lw=1.5, ms=4, label=label)
axs[0].set_title("Gold: exact win-probability gain of pi_beta vs coach", loc="left", fontsize=9)
axs[1].set_title("Proxy's claimed gain (sum of proxy advantages along pi_beta games)", loc="left", fontsize=9)
for ax in axs: ax.set_xlabel("sqrt(KL(pi_beta || coach)) per game (Monte Carlo, 20k games)")
axs[0].set_ylabel("points of win probability"); axs[0].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.9))
fig.text(0.01, 0.005, "Exact for the fitted game MDP, not real football. Pilot: one proxy seed each; fourth-down choice only, coach elsewhere; timeouts and tries model-optimal. beta in {10,30,100,300,1000,10000}.", fontsize=6.5, color="#52514e")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("goodhart_pilot.png", dpi=160)
