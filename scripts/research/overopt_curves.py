"""Research milestone R1, day 5: the pilot's overoptimization curves with the fixed pipeline.

For each cell (a learned win-probability proxy, or the oracle = the exact coach value):

1. Synthetic coach-vs-coach games from the solved MDP (post-decisions by the coach rules)
   label every pre-snap state with the final result; LightGBM proxies are fit on them.
2. One exact backward pass over all 720 ticks computes, at every fourth down, the proxy's
   action values Q_proxy (true dynamics, proxy values one step ahead, float32), the exact
   coach action values Q_coach, and the exact win probability of team A playing pi_beta
   (the coach tilted by exp(beta Q_proxy) at unforced fourth downs) against a coach team
   B, for every beta at once. The oracle cell also evaluates A's exact best fourth-down
   response to the coach.
3. One exact forward pass per beta gives A's state occupancy, and from it the exact
   sequence KL(pi_beta || coach) per game, the proxy's claimed gain
   E[sum over A's decisions of (pi_beta - coach) . Q_proxy] and the same sum with Q_coach,
   which by the performance-difference identity equals the true gain exactly (a check).
4. Optional Monte Carlo (`--mc`): KL, claims and win probability from simulated games.

Exact for the fitted game MDP, not real football. No play-by-play is loaded: no season is
touched. Settings and seeds: configs/overopt.yaml.

Run:  python scripts/research/overopt_curves.py data
      python scripts/research/overopt_curves.py train
      python scripts/research/overopt_curves.py pass oracle --mc     (about 40 minutes)
      python scripts/research/overopt_curves.py pass 1k_large --mc
      python scripts/research/overopt_curves.py pass 16k_large        (and the other cells)
      python scripts/research/overopt_curves.py collect
Writes results/research/overopt_cells/<cell>.json, results/research/overopt_day5.json and
assets/research/overopt_day5.png.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from fourth_and_rl import mdp_game as mg
from fourth_and_rl import overopt as oo
from fourth_and_rl import policy_eval as pe
from fourth_and_rl.io import read_json, write_json
from fourth_and_rl.paths import ASSETS, RESULTS, ROOT

CELLS_DIR = RESULTS / "research" / "overopt_cells"
OUT = RESULTS / "research" / "overopt_day5.json"
FIG = ASSETS / "research" / "overopt_day5.png"
LABEL = "Exact for the fitted game MDP, not real football."


def work_dir(cfg: dict) -> Path:
    d = ROOT / cfg["work_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------------------
# Data and proxies
# --------------------------------------------------------------------------------------


def make_data(args: argparse.Namespace) -> None:
    cfg = oo.load_config()
    sol, coach = oo.load_game()
    for name, g in cfg["games"].items():
        path = work_dir(cfg) / f"games_{name}.npz"
        if path.exists() and not args.force:
            print(f"{name}: exists", flush=True)
            continue
        t0 = time.time()
        d = oo.coach_games(sol, coach, g["n"], g["seed"], post=cfg["post"])
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, X=d["X"], y=d["y"], game=d["game"], seed=g["seed"])
        tmp.replace(path)  # never leave a half-written file behind
        print(f"{name}: {g['n']} games, seed {g['seed']}, {len(d['y'])} rows, "
              f"{time.time() - t0:.1f} s, receiver wins {d['result'].mean():.4f}", flush=True)


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train(args: argparse.Namespace) -> None:
    import lightgbm as lgb

    cfg = oo.load_config()
    wd = work_dir(cfg)
    val = np.load(wd / "games_val.npz")
    for cell, spec in cfg["cells"].items():
        if spec["proxy"] == "oracle":
            continue
        path = wd / f"proxy_{cell}.txt"
        if path.exists() and not args.force:
            print(f"{cell}: exists", flush=True)
            continue
        d = np.load(wd / f"games_{spec['games']}.npz")
        params = cfg["proxy_sizes"][spec["proxy"]]
        reg = lgb.LGBMRegressor(objective="cross_entropy", random_state=cfg["proxy_random_state"],
                                n_jobs=cfg["proxy_threads"], verbose=-1, **params)  # fmt: skip
        t0 = time.time()
        reg.fit(d["X"], d["y"])
        reg.booster_.save_model(str(path))
        meta = {"cell": cell, "games": spec["games"], "n_games": cfg["games"][spec["games"]]["n"],
                "games_seed": cfg["games"][spec["games"]]["seed"], "size": spec["proxy"],
                "params": params, "random_state": cfg["proxy_random_state"],
                "rows": int(len(d["y"])), "fit_seconds": time.time() - t0,
                "train_logloss": _logloss(reg.predict(d["X"]), d["y"]),
                "val_logloss": _logloss(reg.predict(val["X"]), val["y"])}  # fmt: skip
        (wd / f"proxy_{cell}.json").write_text(json.dumps(meta, indent=1))
        print(meta, flush=True)


# --------------------------------------------------------------------------------------
# One exact pass
# --------------------------------------------------------------------------------------


def _val_states(m: mg.GameModel, X: np.ndarray) -> list[tuple[str, tuple[int, ...]]]:
    g = m.kern.grid
    t, s, down, dbin, yl, a, b, r = X.T.astype(np.int64)
    f = g.fidx(down, dbin, yl)
    out = []
    for i in range(len(t)):
        ph = m.phase(int(t[i]))
        e = (a[i], b[i]) if ph == "end" else ((r[i], 0) if ph == "first" else (0, 0))
        out.append(("A", (t[i], f[i], s[i], *e)))
    return out


def _diagnostics(m: mg.GameModel, cfg: dict, cp: pe.CoachPolicy, X: np.ndarray,
                 v_coach: np.ndarray, oracle: oo.FourthQ, q_proxy: oo.FourthQ,
                 bst: Any) -> dict[str, Any]:  # fmt: skip
    """Proxy error on held-out synthetic states (seed of games `val`): value RMSE against
    the exact coach value, and at the coach-visited unforced fourth downs the action-value
    RMSE and the greedy-action disagreement with the exact coach action values."""
    out: dict[str, Any] = {"n_states": int(len(X))}
    if bst is not None:
        pred = bst.predict(X, num_threads=cfg["proxy_threads"])
        e = pred - v_coach
        out.update(value_rmse=float(np.sqrt(np.mean(e**2))), value_mae=float(np.mean(np.abs(e))),
                   value_bias=float(np.mean(e)))  # fmt: skip
    g = m.kern.grid
    t, s, down, dbin, yl, a, b, r = X.T.astype(np.int64)
    f = g.fidx(down, dbin, yl)
    cs = cp.at(t, f, s)
    sel = cs.free_fourth
    qo = oracle.lookup(t[sel], f[sel], s[sel], a[sel], b[sel], r[sel])
    qp = q_proxy.lookup(t[sel], f[sel], s[sel], a[sel], b[sel], r[sel])
    allowed = (cs.p[sel] > 0) & np.isfinite(qo)
    diff = np.where(allowed, qp - qo, np.nan)
    out["n_fourth_down_states"] = int(sel.sum())
    out["q_rmse_fourth"] = float(np.sqrt(np.nanmean(diff**2)))
    ties = cfg["ties"]
    for rule in ("uniform", "exclude"):
        out[f"greedy_{rule}"] = oo.disagreement(qo, qp, allowed, rule, ties["atol"])
    # How often the exact values tie at the top among the coach's actions (float32).
    top2 = np.sort(np.where(allowed, qo, -np.inf), axis=1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    two = allowed.sum(1) >= 2
    out["exact_top_two_gap"] = {"exact_ties": int((two & (gap == 0)).sum()),
                                "below_1e-3": int((two & (gap < 1e-3)).sum()),
                                "n": int(two.sum())}  # fmt: skip
    return out


def run_pass(args: argparse.Namespace) -> None:
    from fourth_and_rl.grid_predict import GridPredictor, slice_values
    from fourth_and_rl.occupancy import ForwardPass

    cfg = oo.load_config()
    cell = args.cell
    spec = cfg["cells"][cell]
    post = cfg["post"]
    betas = [float(b) for b in cfg["betas"]]
    sol, coach = oo.load_game()
    m = sol.model
    g = m.kern.grid
    wd = work_dir(cfg)
    T0 = time.time()

    # Held-out states for the diagnostics (a fixed random subset of the validation games).
    val = np.load(wd / "games_val.npz")
    rng = np.random.default_rng(cfg["seed"])
    pick = rng.choice(len(val["y"]), size=min(100_000, len(val["y"])), replace=False)
    Xv = val["X"][pick]

    cp = pe.CoachPolicy(m, coach)
    oracle = oo.FourthQ(m)
    coach_pair = pe.PolicyPair(m, cp, post=post, coach=coach, record=_val_states(m, Xv))
    is_oracle = spec["proxy"] == "oracle"
    if is_oracle:
        q_proxy, gp, bst, proxy_meta = oracle, None, None, {"proxy": "oracle"}
        proxy_sol = None
    else:
        q_proxy = oo.FourthQ(m)
        gp = GridPredictor(wd / f"proxy_{cell}.txt")
        bst = gp.bst
        proxy_meta = json.loads((wd / f"proxy_{cell}.json").read_text())
        proxy_sol = pe.RingSolution(m, post=post, coach=coach)

    def best_fourth(t: int, q: dict) -> np.ndarray:
        cs = cp.slice(t)
        shape = (g.nf, m.n_score, *m.extra_shape(m.phase(t)))
        q3 = pe.fourth_q(q, cs, cp.f4, shape)
        allowed = np.broadcast_to(cs.p[cp.f4][:, :, None, None, :] > 0, q3.shape)
        return oo.greedy(q3, allowed & np.isfinite(q3), cfg["ties"]["rule"], cfg["ties"]["atol"])[0]

    pairs = {b: pe.PolicyPair(m, oo.tilt_policy(cp, q_proxy, b), cp, post=post, coach=coach,
                              check=False) for b in betas}  # fmt: skip
    if is_oracle:
        pairs["best_response"] = pe.PolicyPair(m, cp.with_fourth(best_fourth), cp, post=post,
                                               coach=coach, check=False)  # fmt: skip
    t_pred = t_back = 0.0
    T = m.clock.n_ticks
    # The pairs are independent within a tick, and numpy and scipy release the GIL in
    # their array work, so a few threads step them in parallel.
    pool = ThreadPoolExecutor(max_workers=cfg["threads"])
    for t in range(1, T + 1):
        t1 = time.time()
        coach_pair.step(t)
        shape = (g.nf, m.n_score, *m.extra_shape(m.phase(t)))
        oracle.put(t, pe.fourth_q(coach_pair.last_q("A"), cp.slice(t), cp.f4, shape))
        if proxy_sol is not None:
            q, _ = proxy_sol.backup(t)
            q_proxy.put(t, pe.fourth_q(q, cp.slice(t), cp.f4, shape))
            t2 = time.time()
            proxy_sol.put(t, slice_values(gp, m, t, dtype=np.dtype(m.dtype)))
            t_pred += time.time() - t2
        list(pool.map(lambda p, t=t: p.step(t), pairs.values()))
        t_back += time.time() - t1
        if t % 60 == 0:
            print(f"  t = {t} / {T}  {time.time() - T0:.0f} s", flush=True)
    t_back -= t_pred
    back_peak = pe.peak_rss_gb()
    start = {str(k): p.start_value() for k, p in pairs.items()}
    coach_start = coach_pair.start_value()
    v_coach = np.array(coach_pair.recorded)
    del pairs, coach_pair, proxy_sol
    gc.collect()
    print(f"backward done {time.time() - T0:.0f} s, peak {back_peak:.2f} GB", flush=True)

    diag = _diagnostics(m, cfg, cp, Xv, v_coach, oracle, q_proxy, bst)
    print("diagnostics", diag, flush=True)

    def forward(b: float) -> dict[str, Any]:
        cpb = pe.CoachPolicy(m, coach)  # one per thread: its slice cache is per tick
        stats = oo.TiltStats(cpb, q_proxy, b, {"oracle": oracle, "proxy": q_proxy})
        fw = ForwardPass(m, oo.tilt_policy(cpb, q_proxy, b), cpb, coach=coach, hooks=[stats])
        fw.run()
        return {"beta": b, "true_gain": start[str(b)] - 0.5, "forward_gain": fw.result_a - 0.5,
               "pdl_gain": stats.claim["oracle"], "claimed_gain": stats.claim["proxy"],
               "kl": stats.kl, "sqrt_kl": float(np.sqrt(stats.kl)),
               "decisions_per_game": stats.decisions, "total_mass": fw.total_mass,
               "forward_seconds": fw.seconds}  # fmt: skip

    t0 = time.time()
    rows = list(pool.map(forward, betas))
    fwd_s = time.time() - t0
    pool.shutdown()
    for row in rows:
        print(row, flush=True)

    out: dict[str, Any] = {
        "cell": cell, "proxy": proxy_meta, "post": post, "betas": betas,
        "coach_start_value": coach_start, "rows": rows, "diagnostics": diag,
        "best_response_gain": (start["best_response"] - 0.5) if is_oracle else None,
        "timing": {"backward_seconds": t_back, "grid_predict_seconds": t_pred,
                   "forward_seconds": fwd_s, "backward_peak_rss_gb": back_peak,
                   "peak_rss_gb": pe.peak_rss_gb(), "n_backward_solutions":
                   2 + 2 * len(betas) + (2 if is_oracle else 0) - (1 if is_oracle else 0)},
        "label": LABEL,
    }  # fmt: skip

    if args.mc:
        checks = cfg["checks"]
        sim = oo.TiltSimulator(sol, coach, q_proxy, {"oracle": oracle, "proxy": q_proxy})
        mc = {}
        if cell in checks["kl_mc"]["cells"]:
            kc = checks["kl_mc"]
            mc["kl_mc"] = []
            for b in betas:
                t0 = time.time()
                r = oo.tilt_monte_carlo(sim, b, kc["games"], kc["seed"], post=post)
                r["seconds"] = time.time() - t0
                mc["kl_mc"].append(r)
                print("mc", r, flush=True)
        if cell in checks["large_mc"]["cells"]:
            lc = checks["large_mc"]
            mc["large_mc"] = []
            for b in lc["betas"]:
                t0 = time.time()
                r = oo.tilt_monte_carlo(sim, float(b), lc["games"], lc["seed"], post=post)
                r["seconds"] = time.time() - t0
                mc["large_mc"].append(r)
                print("large mc", r, flush=True)
        out["monte_carlo"] = mc
    out["timing"]["total_seconds"] = time.time() - T0
    write_json(out, CELLS_DIR / f"{cell}.json")


def _z(mc: float, exact: float, se: float) -> float:
    """(mc - exact) / se; 0 when both are zero up to rounding (the coach's KL is 0)."""
    if abs(mc - exact) < 1e-12:
        return 0.0
    return (mc - exact) / se


def run_mc(args: argparse.Namespace) -> None:
    """Extra Monte Carlo for one cell: rebuild the fourth-down action values (one backward
    pass of the coach and the proxy only), then simulate `games` games at each beta
    (beta 0 is coach against coach, exactly 0.5). The exact values come from the cell's
    saved pass. Appends to results/research/overopt_cells/<cell>_mc.json."""
    from fourth_and_rl.grid_predict import GridPredictor, slice_values

    cfg = oo.load_config()
    spec = cfg["cells"][args.cell]
    post = cfg["post"]
    sol, coach = oo.load_game()
    m = sol.model
    g = m.kern.grid
    cp = pe.CoachPolicy(m, coach)
    oracle = oo.FourthQ(m)
    coach_pair = pe.PolicyPair(m, cp, post=post, coach=coach)
    if spec["proxy"] == "oracle":
        q_proxy, gp, proxy_sol = oracle, None, None
    else:
        q_proxy = oo.FourthQ(m)
        gp = GridPredictor(work_dir(cfg) / f"proxy_{args.cell}.txt")
        proxy_sol = pe.RingSolution(m, post=post, coach=coach)
    t0 = time.time()
    for t in range(1, m.clock.n_ticks + 1):
        shape = (g.nf, m.n_score, *m.extra_shape(m.phase(t)))
        coach_pair.step(t)
        oracle.put(t, pe.fourth_q(coach_pair.last_q("A"), cp.slice(t), cp.f4, shape))
        if proxy_sol is not None:
            q, _ = proxy_sol.backup(t)
            q_proxy.put(t, pe.fourth_q(q, cp.slice(t), cp.f4, shape))
            proxy_sol.put(t, slice_values(gp, m, t, dtype=np.dtype(m.dtype)))
    back_s = time.time() - t0
    exact = {r["beta"]: r for r in read_json(CELLS_DIR / f"{args.cell}.json")["rows"]}
    sim = oo.TiltSimulator(sol, coach, q_proxy, {"oracle": oracle, "proxy": q_proxy})
    path = CELLS_DIR / f"{args.cell}_mc.json"
    out = read_json(path) if path.exists() else {"cell": args.cell, "runs": []}
    for b in args.betas:
        t1 = time.time()
        r = oo.tilt_monte_carlo(sim, b, args.games, args.seed, post=post)
        r["seconds"] = time.time() - t1
        gain = exact[b]["true_gain"] if b in exact else 0.0
        kl_ex = exact[b]["kl"] if b in exact else 0.0
        r.update(exact_gain=gain, exact_kl=kl_ex, seed=args.seed,
                 z_gain=(r["win_a"] - 0.5 - gain) / r["win_a_se"],
                 z_kl=_z(r["kl"], kl_ex, r["kl_se"]))  # fmt: skip
        out["runs"].append(r)
        print(r, flush=True)
    out["rebuild_backward_seconds"] = back_s
    write_json(out, path)


# --------------------------------------------------------------------------------------
# Collect and plot
# --------------------------------------------------------------------------------------


def _mc_checks(cell: dict) -> list[dict[str, Any]]:
    """Exact against Monte Carlo, in standard errors, for every Monte Carlo run of a cell."""
    rows = {r["beta"]: r for r in cell["rows"]}
    out = []
    for kind, runs in (cell.get("monte_carlo") or {}).items():
        for r in runs:
            ex = rows[r["beta"]]
            z = {"kl": (r["kl"] - ex["kl"]) / r["kl_se"],
                 "win_a": (r["win_a"] - 0.5 - ex["true_gain"]) / r["win_a_se"],
                 "claim_proxy": (r["claim_proxy"] - ex["claimed_gain"]) / r["claim_proxy_se"],
                 "claim_oracle": (r["claim_oracle"] - ex["pdl_gain"]) / r["claim_oracle_se"],
                 "decisions": (r["decisions"] - ex["decisions_per_game"]) / r["decisions_se"]}
            out.append({"cell": cell["cell"], "kind": kind, "beta": r["beta"], "games": r["games"],
                        "exact_kl": ex["kl"], "mc_kl": r["kl"], "mc_kl_se": r["kl_se"],
                        "exact_gain": ex["true_gain"], "mc_gain": r["win_a"] - 0.5,
                        "mc_gain_se": r["win_a_se"], "z": z})  # fmt: skip
    return out


def _seed_summary(cells: dict[str, dict], names: list[str], betas: list[float],
                  fixed_peak: float) -> dict[str, Any]:  # fmt: skip
    gains = np.array([[r["true_gain"] for r in cells[n]["rows"]] for n in names])
    peak = gains.max(1)
    final = gains[:, -1]
    j = betas.index(fixed_peak)
    drop = peak - final
    drop_fixed = gains[:, j] - final

    def sd(x: np.ndarray) -> float | None:
        return float(np.std(x, ddof=1)) if len(x) > 1 else None

    out = {"cells": names, "gain_mean": gains.mean(0).tolist(),
           "gain_sd": [sd(gains[:, i]) for i in range(len(betas))],
           "peak_beta": [betas[i] for i in gains.argmax(1)], "peak_gain": peak.tolist(),
           "final_gain": final.tolist(), "drop_peak_to_final": drop.tolist(),
           "drop_mean": float(drop.mean()), "drop_sd": sd(drop),
           "fixed_peak_beta": fixed_peak, "drop_fixed": drop_fixed.tolist(),
           "drop_fixed_mean": float(drop_fixed.mean()),
           "drop_fixed_sd": sd(drop_fixed)}  # fmt: skip
    for key in ("drop", "drop_fixed"):
        s = out[f"{key}_sd"]
        out[f"{key}_survives_2sd"] = (bool(out[f"{key}_mean"] > 2 * s) if s is not None
                                      else None)  # fmt: skip
    return out


def collect(args: argparse.Namespace) -> None:
    cfg = oo.load_config()
    betas = [float(b) for b in cfg["betas"]]
    cells = {p.stem: read_json(p) for p in sorted(CELLS_DIR.glob("*.json"))
             if not p.stem.endswith("_mc")}  # fmt: skip
    extra_mc = [read_json(p) for p in sorted(CELLS_DIR.glob("*_mc.json"))]
    for x in extra_mc:
        for r in x["runs"]:
            r["z_kl"] = _z(r["kl"], r["exact_kl"], r["kl_se"])
    if "oracle" not in cells:
        raise SystemExit("the oracle cell has not been run")
    summary = {}
    for size, peak in (("large", 100.0), ("small", 300.0)):
        names = [n for n in (f"1k_{size}", f"1k_{size}_b", f"1k_{size}_c") if n in cells]
        if names:
            summary[f"1k_{size}"] = _seed_summary(cells, names, betas, peak)
    checks = [c for cell in cells.values() for c in _mc_checks(cell)]
    exactness = [{"cell": n, "beta": r["beta"],
                  "forward_minus_backward": r["forward_gain"] - r["true_gain"],
                  "pdl_minus_backward": r["pdl_gain"] - r["true_gain"]}
                 for n, c in cells.items() for r in c["rows"]]  # fmt: skip
    out = {"what": "R1 day 5: exact true gain of KL-tilted fourth-down policies built on learned "
                   "win-probability proxies, against coach-vs-coach play, with the exact sequence "
                   "KL and the proxy's claimed gain from the forward occupancy pass",
           "label": LABEL, "betas": betas, "post": cfg["post"],
           "best_response_gain": cells["oracle"]["best_response_gain"],
           "cells": {n: {"proxy": c["proxy"], "rows": c["rows"], "diagnostics": c["diagnostics"],
                         "timing": c["timing"]} for n, c in cells.items()},
           "seed_summary": summary, "monte_carlo_checks": checks,
           "extra_monte_carlo": extra_mc,
           "exactness": {"max_abs_forward_minus_backward":
                         max(abs(x["forward_minus_backward"]) for x in exactness),
                         "max_abs_pdl_minus_backward":
                         max(abs(x["pdl_minus_backward"]) for x in exactness)}}  # fmt: skip
    write_json(out, OUT)
    plot(out)
    print(json.dumps(summary, indent=1))


def plot(out: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fourth_and_rl import style

    style.apply()
    order = [("oracle", "exact coach value (no proxy error)", style.BLUE),
             ("16k_large", "64 leaves, 16k games", style.AQUA),
             ("16k_small", "8 leaves, 16k games", style.VIOLET),
             ("1k_small", "8 leaves, 1k games", style.YELLOW),
             ("1k_large", "64 leaves, 1k games", style.ORANGE)]  # fmt: skip
    cells = out["cells"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True)
    br = 100 * out["best_response_gain"]
    for ax in axs:
        ax.axhline(0, color=style.MUTED, lw=1)
    axs[0].axhline(br, color=style.MUTED, lw=1, ls=":")
    axs[0].text(0.03, br + 0.12, "exact best fourth-down response to the coach", fontsize=8.5,
                color=style.INK_2)  # fmt: skip
    for name, label, col in order:
        group = [n for n in (name, f"{name}_b", f"{name}_c") if n in cells]
        for i, n in enumerate(group):
            rows = cells[n]["rows"]
            x = [r["sqrt_kl"] for r in rows]
            lab = label + (f" ({len(group)} data seeds)" if len(group) > 1 else "")
            kw = {"color": col, "lw": 2.2 if i == 0 else 1.2, "alpha": 1.0 if i == 0 else 0.6}
            axs[0].plot(x, [100 * r["true_gain"] for r in rows], "-o", ms=4.5 if i == 0 else 3,
                        label=lab if i == 0 else None, **kw)  # fmt: skip
            axs[1].plot(x, [100 * r["claimed_gain"] for r in rows], "--o",
                        ms=3.5 if i == 0 else 2.5, label=lab if i == 0 else None, **kw)  # fmt: skip
    style.title(axs[0], "True gain of pi_beta over the coach",
                "exact win-probability points, team A against a coach team B")  # fmt: skip
    style.title(axs[1], "Gain the proxy claims", "sum of proxy advantages along A's games (exact)")
    for ax in axs:
        ax.set_xlabel("sqrt of the exact sequence KL(pi_beta || coach) per game")
    axs[0].set_ylabel("points of win probability")
    axs[0].legend(fontsize=8, loc="lower right")
    betas = ", ".join(str(int(b)) for b in out["betas"])
    fig.text(0.0, -0.04, f"{LABEL} Fourth-down choice only, coach elsewhere; timeouts and tries by "
             f"the coach rules. beta in {{{betas}}}. Thin lines: other data seeds.", fontsize=8.5,
             color=style.INK_2)  # fmt: skip
    style.save(fig, FIG)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("data", "train"):
        sub.add_parser(name).add_argument("--force", action="store_true")
    p = sub.add_parser("pass")
    p.add_argument("cell")
    p.add_argument("--mc", action="store_true")
    p = sub.add_parser("mc")
    p.add_argument("cell")
    p.add_argument("--betas", type=float, nargs="+", required=True)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    sub.add_parser("collect")
    args = ap.parse_args()
    {"data": make_data, "train": train, "pass": run_pass, "mc": run_mc,
     "collect": collect}[args.cmd](args)  # fmt: skip


if __name__ == "__main__":
    main()
