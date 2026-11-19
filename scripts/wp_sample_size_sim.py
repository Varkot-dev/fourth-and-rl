"""M3b: how much data is a season of plays really worth? A simulation with known truth.

This reproduces the point of Brill, Yurko and Wyner, "Exploring the Difficulty of
Estimating Win Probability: A Simulation Study" (arXiv 2406.16171), with a simpler game
and our own numbers. It is not a replication of their tables.

The simulated game
  A game has T plays. The lead starts at 0 and each play adds an independent
  Normal(0, 1). The team wins if the final lead is positive. Before play t (t plays
  already run) the true win probability is known exactly:

      WP(t, lead) = Phi( lead / sqrt(T - t) )

  One row is one play: (t, lead, won). Every row of a game carries the same `won`.

Three kinds of training data, as in the paper
  all plays      G games, all T plays of each (what real play-by-play looks like)
  independent    G * T plays, each from its own game (same rows, independent labels)
  one per game   G plays, one random play from each of G games

Two estimators
  probit   Phi(a + b z + c z t/T) with z = lead / sqrt(T - t). The truth is inside
           this family (a = 0, b = 1, c = 0), so it has no bias: the best case.
  gbm      LightGBM on (t, lead), the kind of model people actually fit.

What is measured
  (a) error of the fitted win probability against the truth, by number of games
  (b) coverage of nominal 90% intervals for the true win probability: the textbook
      interval that treats plays as independent, a bootstrap over plays, and a
      bootstrap over games
  (c) effective sample size, two ways: the design effect of the probit slope, and the
      paper's definition (how many independent plays give the same error)

Outputs: results/m3b_sample_size_sim.json, assets/m3b_sample_size.png
"""

from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.special import ndtr, ndtri

from fourth_and_rl import style
from fourth_and_rl.io import read_json, write_json
from fourth_and_rl.paths import ASSETS, CONFIGS, RESULTS

SQRT_2PI = np.sqrt(2 * np.pi)

# ---------------------------------------------------------------------------
# the simulated game
# ---------------------------------------------------------------------------


def simulate_games(n_games: int, n_plays: int, rng: np.random.Generator) -> dict:
    """All plays of `n_games` games. Arrays have shape (n_games, n_plays)."""
    steps = rng.standard_normal((n_games, n_plays))
    lead_after = np.cumsum(steps, axis=1)
    lead = np.concatenate([np.zeros((n_games, 1)), lead_after[:, :-1]], axis=1)
    t = np.broadcast_to(np.arange(n_plays), (n_games, n_plays))
    won = lead_after[:, -1] > 0
    return {
        "t": t,
        "lead": lead,
        "won": np.broadcast_to(won[:, None], (n_games, n_plays)),
        "true_wp": ndtr(lead / np.sqrt(n_plays - t)),
    }


def flatten(games: dict) -> dict:
    return {k: np.ascontiguousarray(v).ravel() for k, v in games.items()}


def one_play_per_game(n_games: int, n_plays: int, rng: np.random.Generator) -> dict:
    """One random play from each of `n_games` independent games (independent labels)."""
    out = {"t": [], "lead": [], "won": [], "true_wp": []}
    for start in range(0, n_games, 20000):
        m = min(20000, n_games - start)
        g = simulate_games(m, n_plays, rng)
        pick = rng.integers(0, n_plays, size=m)
        rows = np.arange(m)
        for k in out:
            out[k].append(g[k][rows, pick])
    return {k: np.concatenate(v) for k, v in out.items()}


def make_dataset(kind: str, n_games: int, n_plays: int, rng: np.random.Generator) -> dict:
    if kind == "all_plays":
        d = flatten(simulate_games(n_games, n_plays, rng))
        d["game"] = np.repeat(np.arange(n_games), n_plays)
        return d
    if kind == "independent":
        return one_play_per_game(n_games * n_plays, n_plays, rng)
    if kind == "one_per_game":
        return one_play_per_game(n_games, n_plays, rng)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------


def probit_design(t: np.ndarray, lead: np.ndarray, n_plays: int) -> np.ndarray:
    z = lead / np.sqrt(n_plays - t)
    return np.column_stack([np.ones_like(z), z, z * t / n_plays])


def probit_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    start: np.ndarray | None = None,
    iters: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted probit by Fisher scoring, for B weight vectors at once.

    `weights` has shape (n, B) (or None for one unweighted fit). Returns the
    coefficients, shape (B, k), and the inverse Fisher information, shape (B, k, k),
    which is the textbook covariance that assumes independent rows.
    """
    n, k = x.shape
    w = np.ones((n, 1)) if weights is None else weights
    n_fits = w.shape[1]
    beta = np.tile(np.array([0.0, 1.0, 0.0]) if start is None else start, (n_fits, 1))
    yy = y.astype(float)[:, None]
    cov = np.zeros((n_fits, k, k))
    for it in range(iters + 1):
        eta = np.clip(x @ beta.T, -8.0, 8.0)
        p = np.clip(ndtr(eta), 1e-12, 1 - 1e-12)
        dens = np.exp(-0.5 * eta * eta) / SQRT_2PI
        h = w * dens * dens / (p * (1 - p))
        info = np.empty((n_fits, k, k))
        for i in range(k):
            for j in range(i, k):
                info[:, i, j] = info[:, j, i] = (x[:, i] * x[:, j]) @ h
        cov = np.linalg.inv(info)
        if it == iters:
            break
        score = (x.T @ (w * dens * (yy - p) / (p * (1 - p)))).T
        beta = beta + np.einsum("bij,bj->bi", cov, score)
    return beta, cov


def gbm_fit(d: dict, params: dict, threads: int, seed: int) -> lgb.Booster:
    x = np.column_stack([d["t"], d["lead"]]).astype(np.float64)
    train = lgb.Dataset(x, label=d["won"].astype(int), weight=d.get("weight"))
    return lgb.train(
        {
            "objective": "binary",
            "learning_rate": params["learning_rate"],
            "num_leaves": params["num_leaves"],
            "min_child_samples": params["min_child_samples"],
            "num_threads": threads,
            "seed": seed,
            "deterministic": True,
            "verbose": -1,
        },
        train,
        num_boost_round=params["n_estimators"],
    )


def gbm_predict(model: lgb.Booster, d: dict) -> np.ndarray:
    return model.predict(np.column_stack([d["t"], d["lead"]]).astype(np.float64))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------------------------------------------------------------------------
# (a) error versus number of games, and (c) the paper's effective sample size
# ---------------------------------------------------------------------------


def error_curves(cfg: dict, test: dict, rng: np.random.Generator) -> dict:
    n_plays = cfg["plays_per_game"]
    x_test = probit_design(test["t"], test["lead"], n_plays)
    kinds = ["all_plays", "independent", "one_per_game"]
    out: dict = {"games_grid": cfg["games_grid"], "probit": {}, "gbm": {}}
    for model, reps in (("probit", cfg["probit_reps"]), ("gbm", cfg["gbm_reps"])):
        for kind in kinds:
            means, ses = [], []
            for g in cfg["games_grid"]:
                errs = []
                for r in range(reps):
                    d = make_dataset(kind, g, n_plays, rng)
                    if model == "probit":
                        beta, _ = probit_fit(probit_design(d["t"], d["lead"], n_plays), d["won"])
                        pred = ndtr(x_test @ beta[0])
                    else:
                        pred = gbm_predict(gbm_fit(d, cfg["gbm"], cfg["threads"], r), test)
                    errs.append(rmse(pred, test["true_wp"]))
                means.append(float(np.mean(errs)))
                ses.append(float(np.std(errs, ddof=1) / np.sqrt(reps)))
            out[model][kind] = {"rmse": means, "se": ses, "reps": reps}
    return out


def matched_independent_rows(curves: dict, n_plays: int) -> list[dict]:
    """The paper's effective sample size: independent plays that give the same error.

    Both the "independent" data (G * T rows) and the "one per game" data (G rows) are
    made of independent plays, so together they trace error against the number of
    independent plays over a wide range. For each number of games G, find the number
    of independent plays whose error equals the error from all G * T dependent plays,
    by straight-line interpolation of log(error) against log(rows) on that curve.
    """
    games = np.array(curves["games_grid"], float)
    out = []
    for model in ("probit", "gbm"):
        rows_all = np.concatenate([games, games * n_plays])
        err_all = np.concatenate(
            [curves[model]["one_per_game"]["rmse"], curves[model]["independent"]["rmse"]]
        )
        order = np.argsort(rows_all)
        log_rows, log_err = np.log(rows_all[order]), np.log(err_all[order])
        # error should fall as rows grow; enforce it so the curve can be inverted
        log_err = np.minimum.accumulate(log_err)
        for g, err in zip(games, curves[model]["all_plays"]["rmse"], strict=True):
            target = np.log(err)
            in_range = bool(log_err.min() <= target <= log_err.max())
            rows = float(np.exp(np.interp(target, log_err[::-1], log_rows[::-1])))
            out.append(
                {
                    "model": model,
                    "games": int(g),
                    "plays": int(g * n_plays),
                    "equivalent_independent_plays": rows,
                    "share_of_plays": rows / (g * n_plays),
                    "multiple_of_games": rows / g,
                    "outside_simulated_range": not in_range,
                }
            )
    return out


# ---------------------------------------------------------------------------
# (b) interval coverage, and (c) the design effect
# ---------------------------------------------------------------------------


def probit_coverage(cfg: dict, test: dict, rng: np.random.Generator) -> dict:
    n_plays, g = cfg["plays_per_game"], cfg["coverage_games"]
    level, n_boot = cfg["nominal_level"], cfg["probit_boot"]
    zcrit = float(ndtri(0.5 + level / 2))
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    x_test = probit_design(test["t"], test["lead"], n_plays)
    truth = test["true_wp"]
    n = g * n_plays
    cover = {"naive": [], "play_bootstrap": [], "game_bootstrap": []}
    width = {k: [] for k in cover}
    slope_cover = {k: [] for k in cover}
    betas, naive_var, naive_eta_var, eta_hat = [], [], [], []
    for _ in range(cfg["probit_coverage_reps"]):
        d = make_dataset("all_plays", g, n_plays, rng)
        x = probit_design(d["t"], d["lead"], n_plays)
        beta, cov = probit_fit(x, d["won"])
        b, v = beta[0], cov[0]
        eta = x_test @ b
        se_eta = np.sqrt(np.einsum("ni,ij,nj->n", x_test, v, x_test))
        intervals = {"naive": (ndtr(eta - zcrit * se_eta), ndtr(eta + zcrit * se_eta))}
        slope_int = {"naive": (b[1] - zcrit * np.sqrt(v[1, 1]), b[1] + zcrit * np.sqrt(v[1, 1]))}
        w_play = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).T.astype(float)
        w_game = rng.multinomial(g, np.full(g, 1.0 / g), size=n_boot).T.astype(float)
        w_game = np.repeat(w_game, n_plays, axis=0)
        for name, w in (("play_bootstrap", w_play), ("game_bootstrap", w_game)):
            bb, _ = probit_fit(x, d["won"], weights=w, start=b, iters=4)
            preds = ndtr(x_test @ bb.T)
            intervals[name] = tuple(np.quantile(preds, [lo_q, hi_q], axis=1))
            slope_int[name] = tuple(np.quantile(bb[:, 1], [lo_q, hi_q]))
        for name, (lo, hi) in intervals.items():
            cover[name].append(float(np.mean((truth >= lo) & (truth <= hi))))
            width[name].append(float(np.mean(hi - lo)))
            slope_cover[name].append(float(slope_int[name][0] <= 1.0 <= slope_int[name][1]))
        betas.append(b)
        naive_var.append(np.diag(v))
        naive_eta_var.append(se_eta**2)
        eta_hat.append(eta)
    betas, naive_var = np.array(betas), np.array(naive_var)
    true_var = betas.var(axis=0, ddof=1)
    deff_slope = float(true_var[1] / naive_var[:, 1].mean())
    # the same ratio for the fitted index at each test state, then averaged
    eta_true_var = np.var(np.array(eta_hat), axis=0, ddof=1)
    eta_naive_var = np.mean(np.array(naive_eta_var), axis=0)
    ok = eta_naive_var > 0
    deff_pred = float(np.median(eta_true_var[ok] / eta_naive_var[ok]))
    reps = cfg["probit_coverage_reps"]

    def summ(v: list[float]) -> dict:
        return {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(reps))}

    return {
        "games": g,
        "plays": n,
        "reps": reps,
        "n_boot": n_boot,
        "nominal_level": level,
        "coverage_of_true_wp": {k: summ(v) for k, v in cover.items()},
        "mean_interval_width": {k: summ(v) for k, v in width.items()},
        "coverage_of_slope": {k: summ(v) for k, v in slope_cover.items()},
        "coefficients_mean": betas.mean(axis=0).tolist(),
        "coefficients_true": [0.0, 1.0, 0.0],
        "design_effect": {
            "slope": deff_slope,
            "effective_plays_from_slope": n / deff_slope,
            "effective_plays_per_game_from_slope": n_plays / deff_slope,
            "prediction_median": deff_pred,
            "effective_plays_from_prediction": n / deff_pred,
            "note": (
                "design effect = true sampling variance / variance claimed by the "
                "independent-rows formula; effective plays = plays / design effect"
            ),
        },
    }


def gbm_coverage(cfg: dict, test: dict, rng: np.random.Generator) -> dict:
    n_plays, g = cfg["plays_per_game"], cfg["coverage_games"]
    level, n_boot = cfg["nominal_level"], cfg["gbm_boot"]
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    truth = test["true_wp"]
    n = g * n_plays
    cover = {"play_bootstrap": [], "game_bootstrap": []}
    width = {k: [] for k in cover}
    for _ in range(cfg["gbm_coverage_reps"]):
        d = make_dataset("all_plays", g, n_plays, rng)
        for name in cover:
            preds = np.empty((n_boot, truth.size))
            for b in range(n_boot):
                if name == "play_bootstrap":
                    idx = rng.integers(0, n, size=n)
                else:
                    games = rng.integers(0, g, size=g)
                    idx = (games[:, None] * n_plays + np.arange(n_plays)[None, :]).ravel()
                boot = {k: d[k][idx] for k in ("t", "lead", "won")}
                preds[b] = gbm_predict(gbm_fit(boot, cfg["gbm"], cfg["threads"], b), test)
            lo, hi = np.quantile(preds, [lo_q, hi_q], axis=0)
            cover[name].append(float(np.mean((truth >= lo) & (truth <= hi))))
            width[name].append(float(np.mean(hi - lo)))
    reps = cfg["gbm_coverage_reps"]

    def summ(v: list[float]) -> dict:
        return {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(reps))}

    return {
        "games": g,
        "plays": n,
        "reps": reps,
        "n_boot": n_boot,
        "nominal_level": level,
        "coverage_of_true_wp": {k: summ(v) for k, v in cover.items()},
        "mean_interval_width": {k: summ(v) for k, v in width.items()},
    }


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def figure(res: dict) -> None:
    style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"wspace": 0.62})
    games = np.array(res["error_curves"]["games_grid"])
    series = [
        ("independent", "Same number of plays, each from its own game", style.ORANGE, "s"),
        ("all_plays", "All plays of G games (real data looks like this)", style.BLUE, "o"),
        ("one_per_game", "One play per game", style.AQUA, "^"),
    ]
    for kind, label, color, marker in series:
        c = res["error_curves"]["gbm"][kind]
        ax1.errorbar(games, 100 * np.array(c["rmse"]), yerr=200 * np.array(c["se"]),
                     color=color, marker=marker, markersize=6, markeredgecolor=style.SURFACE,
                     markeredgewidth=1.2, label=label, capsize=2)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks(games)
    ax1.set_xticklabels([f"{g:,}" for g in games])
    ax1.minorticks_off()
    ax1.set_ylim(0.35, 40)
    yt = [1, 2, 4, 8, 16, 32]
    ax1.set_yticks(yt)
    ax1.set_yticklabels([str(v) for v in yt])
    ax1.set_xlabel("Games in the training data (G)")
    ax1.set_ylabel("Error of fitted win probability (RMSE, points)")
    ax1.legend(loc="lower left", fontsize=9)
    style.title(
        ax1,
        "Shared outcomes make plays worth less",
        "Boosted trees on simulated games with known truth. Bars: 2 SE",
    )

    level = 100 * res["probit_coverage"]["nominal_level"]
    rows = [
        ("Probit, formula interval\n(plays independent)", "probit_coverage", "naive",
         style.NEUTRAL),
        ("Probit, bootstrap plays", "probit_coverage", "play_bootstrap", style.ORANGE),
        ("Probit, bootstrap games", "probit_coverage", "game_bootstrap", style.BLUE),
        ("Boosted trees, bootstrap plays", "gbm_coverage", "play_bootstrap", style.ORANGE),
        ("Boosted trees, bootstrap games", "gbm_coverage", "game_bootstrap", style.BLUE),
    ]
    ypos = [5.2, 4.2, 3.2, 1.6, 0.6]
    for y, (_, block, key, color) in zip(ypos, rows, strict=True):
        c = res[block]["coverage_of_true_wp"][key]
        ax2.barh(y, 100 * c["mean"], height=0.72, color=color)
        ax2.errorbar(100 * c["mean"], y, xerr=200 * c["se"], color=style.INK_2, capsize=2,
                     linewidth=1.2)
        ax2.text(104, y, f"{100 * c['mean']:.0f}%", va="center", ha="right",
                 color=style.INK, fontsize=10.5)
    ax2.axvline(level, color=style.INK, linestyle="--", linewidth=1.2)
    ax2.text(level - 1.5, 6.05, f"nominal {level:.0f}%", ha="right", va="bottom",
             color=style.INK_2, fontsize=9.5)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([r[0] for r in rows])
    ax2.set_xlim(0, 108)
    ax2.set_ylim(0, 6.6)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("Share of game states whose true win probability is inside the interval (%)")
    g = res["probit_coverage"]["games"]
    style.title(
        ax2,
        "Ignoring the shared outcome: intervals too narrow",
        f"Coverage of nominal {level:.0f}% intervals, {g} games per fit",
    )
    style.save(fig, ASSETS / "m3b_sample_size.png")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-only", action="store_true",
                        help="redraw the figure from the saved results without simulating")
    args = parser.parse_args()
    if args.figure_only:
        figure(read_json(RESULTS / "m3b_sample_size_sim.json"))
        return
    with open(CONFIGS / "blown_leads.yaml") as f:
        cfg = yaml.safe_load(f)["sample_size_sim"]
    rng = np.random.default_rng(cfg["seed"])
    n_plays = cfg["plays_per_game"]
    t0 = time.time()
    test = one_play_per_game(cfg["n_test_states"], n_plays, rng)
    curves = error_curves(cfg, test, rng)
    print(f"error curves done ({time.time() - t0:.0f}s)")
    pc = probit_coverage(cfg, test, rng)
    print(f"probit coverage done ({time.time() - t0:.0f}s)")
    gc = gbm_coverage(cfg, test, rng)
    print(f"gbm coverage done ({time.time() - t0:.0f}s)")
    res = {
        "description": (
            "Simulated games: lead is a Gaussian random walk over T plays, true win "
            "probability Phi(lead / sqrt(T - t)). All numbers are from this simulation, "
            "not from Brill, Yurko and Wyner (arXiv 2406.16171), whose design it follows."
        ),
        "config": cfg,
        "error_curves": curves,
        "effective_sample_size_matched_error": matched_independent_rows(curves, n_plays),
        "probit_coverage": pc,
        "gbm_coverage": gc,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    write_json(res, RESULTS / "m3b_sample_size_sim.json")
    figure(res)
    for name, block in (("probit", pc), ("gbm", gc)):
        for k, v in block["coverage_of_true_wp"].items():
            print(f"  {name:6s} {k:15s} coverage {v['mean']:.3f} (se {v['se']:.3f})")
    de = pc["design_effect"]
    print(f"  design effect (slope) {de['slope']:.1f}: {pc['plays']} plays are worth "
          f"{de['effective_plays_from_slope']:.0f} independent ones ({pc['games']} games)")
    for row in res["effective_sample_size_matched_error"]:
        print(f"  {row['model']:6s} G={row['games']:5d}: equal error to "
              f"{row['equivalent_independent_plays']:.0f} independent plays "
              f"({100 * row['share_of_plays']:.0f}% of plays, "
              f"{row['multiple_of_games']:.1f} per game)")


if __name__ == "__main__":
    main()
