"""The four M6 figures, drawn from results/m6_offline_known_truth.json and
results/m6_offline_real_data.json (each function draws only from what exists).

Charts are laid out like M5's policy charts: yards to go up the side, field position
along the bottom with the own goal line on the left. Counts use the one-hue sequential
ramp on a log scale; signed errors use a blue-to-red diverging scale with a grey midpoint
(red: the method claims more than the truth, blue: less).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from fourth_and_rl import style
from fourth_and_rl.io import read_json
from fourth_and_rl.offline import METHOD_COLORS, METHOD_LABELS
from fourth_and_rl.paths import ASSETS, RESULTS

DIST_LABELS = [str(d) for d in range(1, 11)] + ["11-15", "16+"]
DIVERGING = LinearSegmentedColormap.from_list(
    "m6_diverging", [style.SEQUENTIAL[5], style.SEQUENTIAL[1], "#e4e3de", "#f2b8b5",
                     "#b8302f"])
COUNTS = ListedColormap(style.SEQUENTIAL)
ACTIONS = ListedColormap([style.ACTION_COLORS[a] for a in ("go", "field_goal", "punt")])
NEVER = "#ffffff"


def _load(name: str) -> dict[str, Any] | None:
    p = RESULTS / name
    return read_json(p) if p.exists() else None


def _label(name: str) -> str:
    base = name.split("@")[0]
    lab = METHOD_LABELS.get(base, base)
    if "@" in name:
        lab += f", alpha {name.split('@')[1]}"
    return lab


def _color(name: str) -> str:
    return METHOD_COLORS.get(name.split("@")[0], style.INK_2)


def _field(ax: plt.Axes, ylabel: bool = True) -> None:
    ax.set_xlim(99.5, 0.5)
    ax.set_xticks([80, 50, 20])
    ax.set_xticklabels(["own 20", "50", "opp 20"])
    ax.set_yticks(range(len(DIST_LABELS)))
    ax.set_yticklabels(DIST_LABELS, fontsize=7.5)
    ax.set_xlabel("Field position")
    if ylabel:
        ax.set_ylabel("Yards to go")
    ax.grid(False)


def _image(ax: plt.Axes, arr: np.ndarray, **kw: Any) -> Any:
    arr = np.asarray(arr, dtype=float)
    return ax.imshow(arr, aspect="auto", origin="lower", interpolation="nearest",
                     extent=(0.5, arr.shape[1] + 0.5, -0.5, arr.shape[0] - 0.5), **kw)


def _as_float(x: Any) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in x], dtype=float)


def _counts_panel(fig: plt.Figure, ax: plt.Axes, counts: np.ndarray, title: str, sub: str,
                  vmax: float | None = None) -> None:
    counts = np.asarray(counts, dtype=float)
    ax.set_facecolor(NEVER)
    shown = np.where(counts > 0, counts, np.nan)
    _image(ax, np.where(np.isnan(counts), 1.0, np.nan), cmap=ListedColormap(["#ecebe6"]),
           vmin=0, vmax=2)
    top = vmax or max(np.nanmax(shown) if np.isfinite(shown).any() else 10.0, 10.0)
    im = _image(ax, shown, cmap=COUNTS, norm=LogNorm(1, top))
    _field(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Logged go attempts (white: never)", color=style.INK_2)
    style.title(ax, title, sub)


def _error_panel(fig: plt.Figure, ax: plt.Axes, err: np.ndarray, title: str, sub: str,
                 lim: float = 3.0, ylabel: bool = True, label: str | None = None) -> None:
    ax.set_facecolor("#ecebe6")
    im = _image(ax, np.clip(err, -lim, lim), cmap=DIVERGING, norm=Normalize(-lim, lim))
    _field(ax, ylabel)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, extend="both")
    cb.set_label(label or "Learned Q(go) minus true Q*(go), points", color=style.INK_2)
    style.title(ax, title, sub)


def _bucket_mean(res: dict, behavior: str, name: str, size: str, buckets: tuple[str, ...]
                 ) -> float | None:
    try:
        q = res["q_error_by_count"][behavior][name][size]["go"]
    except KeyError:
        return None
    vals = [(q[b]["mean_error"].get("mean"), q[b].get("n_states", {}).get("mean", 1.0))
            for b in buckets if q[b]["mean_error"].get("mean") is not None]
    if not vals:
        return None
    w = np.array([v[1] for v in vals], dtype=float)
    return float(np.average([v[0] for v in vals], weights=w))


# --------------------------------------------------------------------------------------
# 1. Distribution shift (headline)
# --------------------------------------------------------------------------------------


def distribution_shift(path: Path = ASSETS / "m6_distribution_shift.png") -> Path | None:
    res = _load("m6_offline_known_truth.json")
    if res is None or not res.get("headline"):
        return None
    style.apply()
    h = res["headline"]
    size = str(h["config"]["size"])
    alpha = res["tuning"]["cql_alpha_chosen"]
    meth = h["methods"]
    counts = np.asarray(h["counts"]["go"], dtype=float)
    reach = np.asarray(h.get("reachable", np.ones_like(counts)), dtype=float) > 0
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.6))
    n_size = f"{int(size):,}"
    _counts_panel(fig, axes[0, 0], np.where(reach, counts, np.nan),
                  "What coaches log: going for it",
                  f"One coach-like dataset, {n_size} fourth downs (real size)")

    def sub(name: str) -> str:
        thin = _bucket_mean(res, "coach", name, size, ("0", "1-9"))
        thick = _bucket_mean(res, "coach", name, size, ("10-99", "100-999", "1000+"))
        reg = res["regret"]["coach"].get(name, {}).get(size, {}).get("mean")
        parts = []
        if thin is not None:
            parts.append(f"go logged 0-9 times: {thin:+.2f}")
        if thick is not None:
            parts.append(f"10+: {thick:+.2f}")
        if reg is not None:
            parts.append(f"regret {reg:.3f}")
        return "Mean error, " + ", ".join(parts)

    def err(name: str) -> np.ndarray:
        return np.where(reach, _as_float(meth[name]["heatmap"]["error"]), np.nan)

    _error_panel(fig, axes[0, 1], err("fqi_tabular"),
                 "Fitted Q iteration (tabular)", sub("fqi_tabular"))
    _error_panel(fig, axes[0, 2], err("cql_tabular"),
                 f"Conservative Q-learning (tabular, alpha {alpha})",
                 sub("cql_tabular"), ylabel=False)
    # Error against count, every reachable state, for four learners.
    ax = axes[1, 0]
    for name in ("fqi_tabular", "plugin", "cql_tabular"):
        sc = meth.get(name, {}).get("scatter")
        if not sc:
            continue
        x = np.asarray(sc["count"], dtype=float)
        y = np.asarray(sc["error"], dtype=float)
        rng = np.random.default_rng(0)
        xj = np.where(x > 0, x, 0.5) * np.exp(rng.normal(0, 0.08, len(x)))
        ax.scatter(xj, np.clip(y, -6, 6), s=7, alpha=0.5, color=_color(name), linewidths=0,
                   label=_label(name))
    ax.set_xscale("log")
    ax.axhline(0, color=style.INK, linewidth=1.0)
    ax.set_xlim(0.35, None)
    ax.set_ylim(-6.3, 6.3)
    ax.set_xlabel("Times go was logged in that state (0 drawn at 0.5; log scale)")
    ax.set_ylabel("Q(go) minus true Q*(go), points (clipped at 6)")
    ax.legend(loc="upper right", fontsize=8, markerscale=2.5)
    style.title(ax, "Errors live where the data is thin",
                "Every reachable fourth-down state, same dataset")
    d3 = {k: v for k, v in meth.items() if k.startswith(("dqn_d3rlpy", "cql_d3rlpy"))}
    if "dqn_d3rlpy" in d3:
        _error_panel(fig, axes[1, 1], err("dqn_d3rlpy"),
                     "Offline DoubleDQN (d3rlpy)", sub("dqn_d3rlpy"))
    else:
        axes[1, 1].axis("off")
    cql_d3 = sorted(k for k in d3 if k.startswith("cql_d3rlpy"))
    if cql_d3:
        name = "cql_d3rlpy@0.1" if "cql_d3rlpy@0.1" in cql_d3 else cql_d3[0]
        _error_panel(fig, axes[1, 2], err(name),
                     f"Discrete CQL (d3rlpy, alpha {name.split('@')[1]})", sub(name),
                     ylabel=False)
    else:
        axes[1, 2].axis("off")
    fig.suptitle("Learning from coaches' logs: tabular FQI overclaims going for it where "
                 "coaches never go, CQL buries it there, a network extrapolates more gently",
                 x=0.01, ha="left", fontsize=15,
                 fontweight="bold", color=style.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=3, w_pad=2)
    return style.save(fig, path)


# --------------------------------------------------------------------------------------
# 2. Regret versus data
# --------------------------------------------------------------------------------------

REGRET_METHODS = ("fqi_tabular", "fqi_gbt", "fqi_lcb", "cql_tabular", "plugin", "bc_mode", "bc")


def regret_vs_data(path: Path = ASSETS / "m6_regret_vs_data.png") -> Path | None:
    res = _load("m6_offline_known_truth.json")
    if res is None or "regret" not in res:
        return None
    style.apply()
    behaviors = list(res["config"]["known_truth"]["behaviors"])
    titles = {"very_conservative": "More conservative than coaches",
              "coach": "Coach-like (the real logger)", "coach_eps_0.05": "Coach, 5% random",
              "coach_eps_0.2": "Coach, 20% random", "uniform": "Uniformly random"}
    real = res["config"]["known_truth"]["real_size"]
    fig, axes = plt.subplots(1, len(behaviors), figsize=(4.0 * len(behaviors), 6.4),
                             sharey=True)
    for ax, b in zip(np.atleast_1d(axes), behaviors, strict=True):
        rb = res["regret"][b]
        for name in REGRET_METHODS:
            s = rb.get(name)
            if not s:
                continue
            xs = sorted(int(k) for k in s)
            mean = [s[str(x)]["mean"] for x in xs]
            lo = [s[str(x)]["min"] for x in xs]
            hi = [s[str(x)]["max"] for x in xs]
            ls = "--" if name.startswith("bc") else "-"
            ax.fill_between(xs, lo, hi, color=_color(name), alpha=0.15, linewidth=0)
            ax.plot(xs, mean, color=_color(name), linestyle=ls, marker="o", markersize=3,
                    label=_label(name))
        for name, mk in (("dqn_d3rlpy", "s"), ("cql_d3rlpy@1.0", "D"), ("cql_d3rlpy@0.1", "^")):
            s = rb.get(name)
            if s:
                for k, v in s.items():
                    ax.errorbar([int(k) * (1.12 if mk != "s" else 0.9)], [v["mean"]],
                                yerr=[[v["mean"] - v["min"]], [v["max"] - v["mean"]]],
                                color=_color(name), marker=mk, markersize=6, linestyle="",
                                label=_label(name), capsize=2)
        beh = res["behavior_regret"][b]
        ax.axhline(beh, color=METHOD_COLORS["behavior"], linewidth=1.3, linestyle=":")
        ax.text(0.985, beh * 1.05, "logger", color=style.INK_2, fontsize=8, va="bottom",
                ha="right", transform=ax.get_yaxis_transform())
        ax.axvline(real, color=style.AXIS, linewidth=1.0)
        ax.text(real * 1.08, 0.0006, "real data size", color=style.INK_2, fontsize=8,
                rotation=90, va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.01, linscale=0.5)
        ax.set_ylim(0, 0.9)
        ax.set_yticks([0, 0.005, 0.01, 0.03, 0.1, 0.3, 0.6])
        ax.set_yticklabels(["0", "0.005", "0.01", "0.03", "0.1", "0.3", "0.6"])
        ax.set_xlabel("Logged fourth downs (log scale)")
        style.title(ax, titles.get(b, b), f"Logger's own regret {beh:.3f}")
    np.atleast_1d(axes)[0].set_ylabel("True regret: expected points lost per drive")
    seen: dict[str, Any] = {}
    for a in np.atleast_1d(axes):
        handles, labels = a.get_legend_handles_labels()
        for hd, lb in zip(handles, labels, strict=True):
            seen.setdefault(lb, hd)
    seen["Logger's own regret"] = Line2D([], [], color=METHOD_COLORS["behavior"],
                                         linestyle=":", linewidth=1.3)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("True regret of each learned policy, by data size and how conservative the "
                 "logger was (exact policy evaluation; band: range over 5 seeds)", x=0.01,
                 ha="left", fontsize=13.5, fontweight="bold", color=style.INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95), w_pad=1.5)
    return style.save(fig, path)


# --------------------------------------------------------------------------------------
# 3. The optimizer's curse
# --------------------------------------------------------------------------------------

CURSE_METHODS = ("fqi_tabular", "fqi_gbt", "plugin", "fqi_lcb", "cql_tabular", "dqn_d3rlpy",
                 "cql_d3rlpy@1.0", "cql_d3rlpy@0.1")


def optimizers_curse(path: Path = ASSETS / "m6_optimizers_curse.png") -> Path | None:
    res = _load("m6_offline_known_truth.json")
    if res is None or "optimizers_curse" not in res:
        return None
    style.apply()
    real = str(res["config"]["known_truth"]["real_size"])
    oc = res["optimizers_curse"]["coach"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    names = [n for n in CURSE_METHODS if real in oc.get(n, {})]
    for i, name in enumerate(names):
        col = _color(name)
        for part, mk, fill in (("usual", "o", False), ("rare", "o", True)):
            st = oc[name][real][part]["mean_claimed_minus_true"]
            if st.get("mean") is None:
                continue
            ax.errorbar([st["mean"]], [i + (0.13 if part == "rare" else -0.13)],
                        xerr=[[st["mean"] - st["min"]], [st["max"] - st["mean"]]],
                        color=col, marker=mk, markersize=8, capsize=2,
                        markerfacecolor=col if fill else style.SURFACE, linestyle="")
        share = oc[name][real]["rare"]["share_of_logged_decisions"]["mean"]
        ax.text(0.01, i + 0.4, f"differs from the coaches' most common call in {100 * share:.1f}% "
                "of logged decisions", fontsize=7.5, color=style.INK_2, va="center",
                transform=ax.get_yaxis_transform())
    ax.axvline(0, color=style.INK, linewidth=1.0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([_label(n) for n in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Claimed value of the chosen action minus its true value, points")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=style.INK_2, markerfacecolor=style.SURFACE,
               linestyle="", label="chosen action is the coaches' most common call"),
        Line2D([], [], marker="o", color=style.INK_2, linestyle="",
               label="chosen action is another (less often logged) one")],
        loc="lower right", fontsize=8.5)
    style.title(ax, "What each method claims for its own choices",
                f"Coach-like data, {int(real):,} fourth downs; weighted by logged decisions; "
                "range over seeds")
    ax = axes[1]
    for name in ("fqi_tabular", "fqi_gbt", "plugin", "fqi_lcb", "cql_tabular"):
        s = oc.get(name)
        if not s:
            continue
        xs = sorted(int(k) for k in s)
        y = [s[str(x)]["rare"]["mean_claimed_minus_true"].get("mean") for x in xs]
        keep = [(x, v) for x, v in zip(xs, y, strict=True) if v is not None]
        if keep:
            ax.plot([k[0] for k in keep], [k[1] for k in keep], color=_color(name),
                    marker="o", markersize=3.5, label=_label(name))
    ax.axhline(0, color=style.INK, linewidth=1.0)
    ax.axvline(int(real), color=style.AXIS, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Logged fourth downs (log scale)")
    ax.set_ylabel("Overclaim where the method departs from coaches, points")
    ax.legend(loc="upper right", fontsize=8.5)
    style.title(ax, "The curse shrinks slowly with data", "Mean over 5 seeds, coach-like data")
    fig.tight_layout(w_pad=3)
    return style.save(fig, path)


# --------------------------------------------------------------------------------------
# 4. Real data
# --------------------------------------------------------------------------------------


def real_data(path: Path = ASSETS / "m6_real_data.png") -> Path | None:
    res = _load("m6_offline_real_data.json")
    if res is None or "charts" not in res:
        return None
    style.apply()
    ch = res["charts"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.2))
    n = res["data"]["n_transitions"]
    go_n = res["data"]["actions"]["go"]
    reach = np.asarray(ch.get("reachable", np.ones_like(ch["counts"]["go"])), dtype=float) > 0
    _counts_panel(fig, axes[0, 0], np.where(reach, np.asarray(ch["counts"]["go"], float), np.nan),
                  "Real go attempts, 2010 to 2023",
                  f"Neutral fourth downs: {n:,}, of which {go_n:,} go")
    for ax, name, yl in ((axes[0, 1], "fqi_tabular", True), (axes[0, 2], "cql_tabular", False)):
        m = ch["methods"][name]
        st = res["claims"][name]
        _error_panel(fig, ax, np.where(reach, _as_float(m["q_go_minus_model"]), np.nan),
                     _label(name),
                     f"Mean gap, go logged 0-9 times: {st['go_gap_thin_mean']:+.2f}; "
                     f"10+: {st['go_gap_thick_mean']:+.2f}",
                     ylabel=yl, label="Learned Q(go) minus model Q*(go), points")
    for ax, name, yl in ((axes[1, 0], "model_optimum", True), (axes[1, 1], "fqi_tabular", False),
                         (axes[1, 2], "cql_tabular", False)):
        pol = np.asarray(ch["methods"][name]["policy"], dtype=float) if name != "model_optimum" \
            else np.asarray(ch["model_policy"], dtype=float)
        ax.set_facecolor("#ecebe6")
        pol = np.where(reach, pol, np.nan)
        _image(ax, pol, cmap=ACTIONS, vmin=-0.5, vmax=2.5)
        if name != "model_optimum":
            diff = reach & (pol != np.asarray(ch["model_policy"], dtype=float))
            r, c = np.nonzero(diff)
            ax.scatter(c + 1, r, s=6, color=style.INK, linewidths=0)
            agree = res["agreement"][name]
            sub = (f"Dots: differs from the model. Same call as model "
                   f"{100 * agree['with_model_optimum_logged']:.0f}%, as coaches "
                   f"{100 * agree['with_coach_mode_logged']:.0f}%")
        else:
            agree = res["agreement"]["model_optimum"]
            sub = (f"Same call as coaches' most common one in "
                   f"{100 * agree['with_coach_mode_logged']:.0f}% of logged decisions")
        _field(ax, yl)
        style.title(ax, "Solved model's optimum" if name == "model_optimum"
                    else f"{_label(name)}: policy", sub)
    handles = [Patch(color=style.ACTION_COLORS[a], label=style.ACTION_LABELS[a])
               for a in ("go", "field_goal", "punt")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Real coaches' logs, compared with the solved model (a model, not the truth): "
                 "where go is thin, tabular FQI claims more for it, CQL less", x=0.01,
                 ha="left", fontsize=15, fontweight="bold", color=style.INK)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965), h_pad=3, w_pad=2)
    return style.save(fig, path)


def make_known_truth() -> list[Path]:
    out = [distribution_shift(), regret_vs_data(), optimizers_curse()]
    return [p for p in out if p is not None]


def make_real_data() -> list[Path]:
    out = [real_data()]
    return [p for p in out if p is not None]
