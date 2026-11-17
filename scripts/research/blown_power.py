"""Track A1: power of the blown-lead calibration tests against known overconfidence.

Writes results/research/blown_power.json and assets/research/blown_power.png.
Pure simulation; needs no play-by-play data.
"""

from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml

from fourth_and_rl import style
from fourth_and_rl.blown_power import run_power
from fourth_and_rl.paths import ROOT

CONFIG = ROOT / "configs" / "blown_power.yaml"
OUT_JSON = ROOT / "results" / "research" / "blown_power.json"
OUT_FIG = ROOT / "assets" / "research" / "blown_power.png"


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    by_size = []
    for n_games in cfg["league_sizes"]:
        res = run_power(
            n_games=n_games, ks=cfg["ks"], thresholds=cfg["thresholds"],
            n_null=cfg["n_null"], n_rep=cfg["n_rep"], n_steps=cfg["n_steps"],
            gap_shape=cfg["gap_shape"], p0_sd=cfg["p0_sd"], rng=rng, alpha=cfg["alpha"],
        )
        by_size.append(res)
        print(n_games, "games:", [round(x, 3) for x in res["power_first_crossing"]])
    out = {
        "description": (
            "Rejection rate at the 5% level of two blown-lead tests on simulated leagues "
            "whose feed reports expit(k * logit(p)) for the true win probability p. "
            "k = 1 calibrated, k > 1 overconfident."
        ),
        "config": cfg,
        "results": by_size,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    plot(out)


def plot(out: dict) -> None:
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    seasons = {272: "1 season", 1088: "4 seasons", 2176: "8 seasons", 4350: "16 seasons"}
    colors = [style.SEQUENTIAL[i] for i in (1, 3, 5, 6)]
    for ax, key, name in (
        (axes[0], "power_first_crossing", "First-crossing test (exact for jumpy paths)"),
        (axes[1], "power_continuous_ks", "Continuous-formula KS test"),
    ):
        for res, color in zip(out["results"], colors, strict=True):
            ax.plot(res["ks"], res[key], marker="o", ms=4, color=color,
                    label=seasons.get(res["n_games"], f"{res['n_games']} games"))
        ax.axhline(0.05, color=style.MUTED, lw=1, ls="--")
        ax.axvline(1.0, color=style.AXIS, lw=1)
        ax.set_xlabel("Temperature k (1 = calibrated, above 1 = overconfident)")
        ax.set_title(name, fontsize=11)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Share of simulated leagues where the test rejects")
    axes[0].legend(loc="lower right")
    fig.suptitle("Can the blown-lead test catch an overconfident win-probability model?",
                 x=0.06, y=1.04, ha="left", fontweight="bold", fontsize=13)
    style.save(fig, OUT_FIG)


if __name__ == "__main__":
    import sys

    if "--plot-only" in sys.argv:
        plot(json.loads(OUT_JSON.read_text()))
    else:
        main()
