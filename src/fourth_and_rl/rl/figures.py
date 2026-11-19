"""The three M5 figures, drawn from results/m5_rl_tabular.json and results/m5_rl_deep.json.

Each figure draws whatever results exist, so either script can redraw all three.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from fourth_and_rl import style
from fourth_and_rl.paths import ASSETS, RESULTS
from fourth_and_rl.rl import ALGO_COLORS, ALGO_LABELS

# Variants of one algorithm inside an ablation panel (the default always first, in ink).
VARIANT_SHADES = [style.INK, style.BLUE, style.AQUA, style.ORANGE, style.VIOLET]


def _load(name: str) -> dict[str, Any] | None:
    p = RESULTS / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _curve(ax: plt.Axes, s: dict, color: str, label: str, ls: str = "-") -> None:
    x = np.asarray(s["steps"], dtype=float)
    ax.fill_between(x, s["min"], s["max"], color=color, alpha=0.18, linewidth=0)
    ax.plot(x, s["mean"], color=color, label=label, linestyle=ls)


def _symlog(ax: plt.Axes, linthresh: float, top: float) -> None:
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
    ax.set_ylim(0, top)
    ax.set_xscale("log")
    ax.axhline(0.0, color=style.INK, linewidth=1.2)


def _hline(ax: plt.Axes, y: float, name: str, x_text: float, below: bool = False) -> None:
    ax.axhline(y, color=ALGO_COLORS[name], linewidth=1.4, linestyle="--")
    ax.text(x_text, y * (0.96 if below else 1.04), ALGO_LABELS[name], color=style.INK_2,
            fontsize=8.5, va="top" if below else "bottom", ha="right")


# --------------------------------------------------------------------------------------
# 1. Learning curves (headline)
# --------------------------------------------------------------------------------------


def learning_curves(path: Path = ASSETS / "m5_learning_curves.png") -> Path | None:
    tab, deep = _load("m5_rl_tabular.json"), _load("m5_rl_deep.json")
    if tab is None and deep is None:
        return None
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    xmin = 1e3
    if tab is not None:
        algos = tab["algorithms"]
        for name in ("q_learning", "expected_sarsa", "reinforce"):
            if name in algos:
                _curve(ax, algos[name], ALGO_COLORS[name], ALGO_LABELS[name])
        base = tab["baselines_regret"]
        for name in ("never_go", "coach"):
            _hline(ax, base[name], name, 5.5e6, below=name == "coach")
    if deep is not None:
        for name in ("reinforce_net", "dqn", "ppo"):
            if name in deep.get("drive", {}):
                _curve(ax, deep["drive"][name], ALGO_COLORS[name], ALGO_LABELS[name])
    _symlog(ax, 0.01, 1.0)
    ax.set_yticks([0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8])
    ax.set_yticklabels(["0", "0.005", "0.01", "0.05", "0.1", "0.2", "0.4", "0.8"])
    ax.set_xlim(left=xmin)
    ax.set_xlabel("Fourth-down decisions seen in training (log scale)")
    ax.set_ylabel("Regret: expected points lost per drive")
    ax.text(1.2e3, 0.0006, "DP optimum (regret 0)", color=style.INK, fontsize=8.5, va="bottom")
    style.title(ax, "Single drive, expected points",
                "Exact regret by policy evaluation. Line: mean; band: range over seeds")
    ax.legend(loc="lower left", fontsize=8.5, bbox_to_anchor=(0.0, 0.06))

    ax = axes[1]
    if deep is not None and deep.get("endgame"):
        for name in ("dqn", "ppo"):
            if name in deep["endgame"]:
                _curve(ax, deep["endgame"][name], ALGO_COLORS[name], ALGO_LABELS[name])
        b = deep.get("baselines")
        if b:
            low = min(("never_go", "coach"), key=lambda k: b[k]["regret"])
            for name in ("never_go", "coach"):
                _hline(ax, b[name]["regret"], name, 1.5e6, below=name == low)
        _symlog(ax, 0.002, 0.3)
        ax.set_yticks([0, 0.002, 0.01, 0.03, 0.1, 0.2])
        ax.set_yticklabels(["0", "0.2", "1", "3", "10", "20"])
        ax.set_xlim(left=1e4)
        ax.text(1.2e4, 0.00012, "DP optimum (regret 0)", color=style.INK, fontsize=8.5,
                va="bottom")
        ax.legend(loc="lower left", fontsize=8.5, bbox_to_anchor=(0.0, 0.06))
    else:
        ax.text(0.5, 0.5, "run scripts/rl_deep.py", transform=ax.transAxes, ha="center",
                color=style.INK_2)
    ax.set_xlabel("Play calls seen in training (log scale)")
    ax.set_ylabel("Regret: win-probability points lost per game")
    style.title(ax, "Last five minutes, win probability",
                "Exact per-decision losses summed over 4,000 fixed games. Range over seeds")
    fig.tight_layout(w_pad=3)
    return style.save(fig, path)


# --------------------------------------------------------------------------------------
# 2. Policy comparison
# --------------------------------------------------------------------------------------

_CMAP = ListedColormap([style.ACTION_COLORS[a] for a in ("go", "field_goal", "punt")])


def _chart(ax: plt.Axes, codes: np.ndarray, reps: list[int], differ: np.ndarray | None,
           title: str, subtitle: str | None = None) -> None:
    codes = np.asarray(codes, dtype=float)
    ax.imshow(codes, cmap=_CMAP, vmin=-0.5, vmax=2.5, aspect="auto", origin="lower",
              interpolation="nearest", extent=(0.5, codes.shape[1] + 0.5, -0.5,
                                               codes.shape[0] - 0.5))
    if differ is not None and differ.any():
        r, c = np.nonzero(differ)
        ax.scatter(c + 1, r, s=7, color=style.INK, marker="o", linewidths=0)
    ax.invert_xaxis()
    ax.set_xticks([90, 70, 50, 30, 10])
    ax.set_xticklabels(["own 10", "own 30", "50", "opp 30", "opp 10"])
    ax.set_yticks(range(len(reps)))
    ax.set_yticklabels([str(r) if r <= 10 else ("11-15" if r == 13 else "16+") for r in reps],
                       fontsize=7.5)
    ax.set_ylabel("Yards to go")
    ax.grid(False)
    style.title(ax, title, subtitle)


def policy_comparison(path: Path = ASSETS / "m5_policy_comparison.png") -> Path | None:
    tab, deep = _load("m5_rl_tabular.json"), _load("m5_rl_deep.json")
    if tab is None:
        return None
    style.apply()
    ch = tab["charts"]
    reps = ch["dist_reps"]
    opt = np.asarray(ch["optimal"], dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.6))
    _chart(axes[0, 0], opt, reps, None, "Drive: DP optimum", "Best fourth-down action")
    for ax, name in ((axes[0, 1], "q_learning"), (axes[0, 2], "reinforce")):
        if name in ch:
            learned = np.asarray(ch[name], dtype=float)
            diff = (learned != opt) & np.isfinite(opt)
            reg = tab["algorithms"][name]["majority_policy_regret"]
            _chart(ax, learned, reps, diff, f"Drive: {ALGO_LABELS[name]}",
                   f"Majority of 5 seeds. Dots: differs from DP ({int(diff.sum())} cells, "
                   f"regret {reg:.4f})")
    ax = axes[1, 0]
    visits = np.asarray(ch["visits_optimal"], dtype=float)
    vis = np.where(visits > 0, visits, np.nan)
    im = ax.imshow(vis, cmap=ListedColormap(style.SEQUENTIAL), norm=LogNorm(1e-6, 1e-2),
                   aspect="auto", origin="lower", interpolation="nearest",
                   extent=(0.5, vis.shape[1] + 0.5, -0.5, vis.shape[0] - 0.5))
    if "q_learning" in ch:
        diff = (np.asarray(ch["q_learning"], dtype=float) != opt) & np.isfinite(opt)
        r, c = np.nonzero(diff)
        ax.scatter(c + 1, r, s=9, color=style.ORANGE, linewidths=0, label="Q-learning differs")
        ax.legend(loc="upper left", fontsize=8)
    ax.invert_xaxis()
    ax.set_xticks([90, 70, 50, 30, 10])
    ax.set_xticklabels(["own 10", "own 30", "50", "opp 30", "opp 10"])
    ax.set_yticks(range(len(reps)))
    ax.set_yticklabels([str(r) if r <= 10 else ("11-15" if r == 13 else "16+") for r in reps],
                       fontsize=7.5)
    ax.grid(False)
    ax.set_ylabel("Yards to go")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Visits per drive start", color=style.INK_2)
    style.title(ax, "Where the mistakes are", "How often the optimal policy reaches each state")
    if deep is not None and deep.get("baselines"):
        b = deep["baselines"]
        e_opt = np.asarray(b["optimal_chart"]["codes"], dtype=float)
        _chart(axes[1, 1], e_opt, reps, None, "End game: DP optimum",
               "Up 4, 4:00 left, both teams 3 timeouts")
        dq = deep.get("endgame", {}).get("dqn", {}).get("fourth_down_chart_seed0")
        if dq is not None:
            e = np.asarray(dq["codes"], dtype=float)
            diff = e != e_opt
            _chart(axes[1, 2], e, reps, diff, "End game: DQN (first seed)",
                   f"Same slice. Dots: differs from DP ({int(diff.sum())} of {diff.size} cells)")
        else:
            axes[1, 2].axis("off")
    else:
        axes[1, 1].axis("off")
        axes[1, 2].axis("off")
    for ax in axes[:, :].ravel():
        if ax.get_xlabel() == "" and ax.axison:
            ax.set_xlabel("Field position")
    handles = [Patch(color=style.ACTION_COLORS[a], label=style.ACTION_LABELS[a])
               for a in ("go", "field_goal", "punt")]
    handles.append(Line2D([], [], marker="o", color=style.INK, linestyle="", markersize=4,
                          label="Learned action differs from DP"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=3)
    return style.save(fig, path)


# --------------------------------------------------------------------------------------
# 3. Ablations
# --------------------------------------------------------------------------------------


def _family(ax: plt.Axes, fam: dict, labels: dict[str, str], title: str, sub: str) -> None:
    for (variant, s), color in zip(fam.items(), VARIANT_SHADES * 2, strict=False):
        _curve(ax, s, color, labels.get(variant, variant))
    _symlog(ax, 0.01, 1.0)
    ax.set_yticks([0, 0.01, 0.05, 0.1, 0.2, 0.4])
    ax.set_yticklabels(["0", "0.01", "0.05", "0.1", "0.2", "0.4"])
    ax.set_xlim(left=1e3)
    ax.set_xlabel("Decisions (log scale)")
    ax.set_ylabel("Regret, points per drive")
    ax.legend(loc="upper right", fontsize=8)
    style.title(ax, title, sub)


def ablations(path: Path = ASSETS / "m5_ablations.png") -> Path | None:
    tab, deep = _load("m5_rl_tabular.json"), _load("m5_rl_deep.json")
    if tab is None:
        return None
    style.apply()
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    ab = tab["ablations"]
    if "exploration" in ab:
        fam = dict(ab["exploration"])
        fam.update({k: v for k, v in ab.get("optimistic", {}).items()
                    if k == "optimistic_greedy_const_alpha"})
        _family(axes[0, 0], fam, {
            "linear_decay": "epsilon 1 to 0.05 (default)",
            "constant_0.1": "epsilon 0.1",
            "constant_0.5": "epsilon 0.5",
            "greedy_zero_init": "greedy, Q starts at 0",
            "optimistic_greedy_const_alpha": "greedy, Q starts at 8 (optimistic)",
        }, "Exploration (Q-learning)", "3 seeds, 3M decisions each")
        axes[0, 0].legend(loc="lower left", fontsize=8)
    if "step_size" in ab:
        _family(axes[0, 1], ab["step_size"], {
            "count_omega_0.7": "1 / n^0.7 (default)",
            "count_omega_1.0": "1 / n (running mean)",
            "count_omega_0.5": "1 / n^0.5",
            "constant_0.05": "constant 0.05",
            "constant_0.005": "constant 0.005",
        }, "Step size (Q-learning)", "n = updates of that state and action so far")
    ax = axes[0, 2]
    algos = tab["algorithms"]
    for name in ("q_learning", "sarsa", "expected_sarsa"):
        if name in algos:
            _curve(ax, algos[name], ALGO_COLORS[name], ALGO_LABELS[name])
    _symlog(ax, 0.01, 1.0)
    ax.set_yticks([0, 0.01, 0.05, 0.1, 0.2, 0.4])
    ax.set_yticklabels(["0", "0.01", "0.05", "0.1", "0.2", "0.4"])
    ax.set_xlim(left=1e3)
    ax.set_xlabel("Decisions (log scale)")
    ax.set_ylabel("Regret, points per drive")
    ax.legend(loc="upper right", fontsize=8)
    style.title(ax, "Off-policy versus on-policy", "Greedy policy of each learner, 5 seeds")

    ax = axes[1, 0]
    for name in ("reinforce_no_baseline", "reinforce"):
        if name in algos:
            _curve(ax, algos[name], ALGO_COLORS[name], ALGO_LABELS[name])
    _symlog(ax, 0.01, 1.0)
    ax.set_yticks([0, 0.01, 0.05, 0.1, 0.2, 0.4])
    ax.set_yticklabels(["0", "0.01", "0.05", "0.1", "0.2", "0.4"])
    ax.set_xlim(left=1e3)
    ax.set_xlabel("Decisions (log scale)")
    ax.set_ylabel("Regret, points per drive")
    gv = tab.get("gradient_variance")
    sub = "Same step size, 5 seeds"
    if gv:
        sub += f". Gradient variance with baseline: {gv['uniform']['variance_ratio']:.2f}x"
    ax.legend(loc="upper right", fontsize=8)
    style.title(ax, "REINFORCE: baseline on or off", sub)

    ax = axes[1, 1]
    if deep is not None and deep.get("approximation_floor"):
        ap = deep["approximation_floor"]
        end = deep.get("endgame", {})
        rows = [("Network fit to exact Q*", ap["endgame"]["regret"], style.NEUTRAL)]
        for name in ("dqn", "ppo"):
            if name in end:
                rows.append((f"{ALGO_LABELS[name].split(' (')[0]}, learned (mean of seeds)",
                             end[name]["final_regret"]["mean"], ALGO_COLORS[name]))
        if deep.get("baselines"):
            rows.append(("Coach-like policy", deep["baselines"]["coach"]["regret"],
                         ALGO_COLORS["coach"]))
        y = np.arange(len(rows))
        vals = [100 * r[1] for r in rows]
        ax.barh(y, vals, color=[r[2] for r in rows], height=0.6)
        for yi, v in zip(y, vals, strict=True):
            ax.text(v, yi, f"  {v:.2f}", va="center", color=style.INK, fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel("Regret, win-probability points per game (end game)")
        ax.set_xlim(0, max(vals) * 1.3)
        ax.grid(axis="y", visible=False)
        dq = deep.get("drive", {}).get("dqn", {}).get("final_regret", {}).get("mean")
        sub = (f"2 x {deep['config']['dqn']['net_arch'][0]} network as DQN. On the drive the same "
               f"fit loses only {ap['drive']['regret']:.4f} points")
        if dq is not None:
            sub += f" (DQN: {dq:.2f})"
        style.title(ax, "Function approximation error", sub)
    else:
        ax.axis("off")
    ax = axes[1, 2]
    if deep is not None and deep.get("sparsity"):
        names = {"dqn_none": "sparse: win or loss at the end",
                 "dqn_score": "shaped by the score (no answer used)",
                 "dqn_value": "shaped by the solved value (oracle)"}
        for (name, s), color in zip(deep["sparsity"].items(), VARIANT_SHADES, strict=False):
            _curve(ax, s, color, names.get(name, name))
        _symlog(ax, 0.002, 0.3)
        ax.set_yticks([0, 0.002, 0.01, 0.03, 0.1, 0.2])
        ax.set_yticklabels(["0", "0.2", "1", "3", "10", "20"])
        ax.set_xlim(left=1e4)
        ax.set_xlabel("Play calls (log scale)")
        ax.set_ylabel("Regret, win-probability points")
        ax.legend(loc="upper right", fontsize=8)
        style.title(ax, "Reward sparsity (end game, DQN)", "2 seeds each")
    else:
        ax.axis("off")
    fig.tight_layout(h_pad=3.5, w_pad=2.5)
    return style.save(fig, path)


def make_all() -> list[Path]:
    """Redraw every M5 figure that has results."""
    out = [learning_curves(), policy_comparison(), ablations()]
    return [p for p in out if p is not None]
