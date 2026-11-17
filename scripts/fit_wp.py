"""M1: fit, calibrate and evaluate the win-probability model.

Outputs:
  data/models/wp.joblib                  headline model (not committed)
  data/processed/wp_predictions.parquet  out-of-fold win probability for every play
  results/m1_wp_metrics.json             every number quoted about this model
  results/m1_wp_tuning.json              hyperparameter grid (only with --tune)
  assets/m1_wp_reliability.png           reliability curves on the test seasons
  assets/m1_wp_by_quarter.png            log loss and calibration error by quarter

Run:  python scripts/fit_wp.py          (about two minutes)
      python scripts/fit_wp.py --tune   (also reruns the hyperparameter grid)
"""

from __future__ import annotations

import argparse
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from fourth_and_rl import data, style, wp
from fourth_and_rl.io import write_json
from fourth_and_rl.paths import ASSETS, DATA, RESULTS

MODELS = {"ours": "wp_ours", "nflfastr": "wp", "vegas": "vegas_wp"}
LABELS = {
    "ours": "Ours (LightGBM + Platt)",
    "nflfastr": "nflfastR wp (no spread)",
    "vegas": "nflfastR vegas_wp (with spread)",
}


def seasons(bounds: list[int]) -> list[int]:
    return list(range(bounds[0], bounds[1] + 1))


def comparable(df: pl.DataFrame) -> pl.DataFrame:
    """Rows where the label and all three models' predictions exist."""
    return df.filter(
        pl.col(wp.LABEL).is_not_null()
        & pl.col("wp_ours").is_not_null()
        & pl.col("wp").is_not_null()
        & pl.col("vegas_wp").is_not_null()
    )


def three_way(df: pl.DataFrame, n_bins: int) -> dict:
    y = df[wp.LABEL].to_numpy()
    out: dict = {"rows": df.height, "games": df["game_id"].n_unique()}
    for name, col in MODELS.items():
        out[name] = wp.all_metrics(y, df[col].to_numpy(), n_bins)
    return out


# --------------------------------------------------------------------------- tuning


def tune(plays: pl.DataFrame, cfg: dict) -> dict:
    """Grid over tree size and leaf size, scored on seasons inside the training span."""
    t = cfg["tuning"]
    fit = wp.season_filter(plays, seasons(t["fit_seasons"]))
    val = wp.season_filter(plays, seasons(t["validation_seasons"]))
    val = val.filter(pl.Series(wp.scoreable(val)) & pl.col(wp.LABEL).is_not_null())
    y = val[wp.LABEL].to_numpy()

    def run(overrides: dict) -> dict:
        start = time.perf_counter()
        model = wp.WPModel({**cfg["lgbm"], **overrides}, "none", cfg["n_threads"])
        model.fit(fit, seed=cfg["seed"])
        seconds = time.perf_counter() - start
        row = {**overrides, **wp.all_metrics(y, model.predict(val), cfg["metrics"]["ece_bins"])}
        row["fit_seconds"] = round(seconds, 1)
        print(f"  tune {overrides}: log loss {row['log_loss']:.5f}")
        return row

    rows = [
        run({"min_child_samples": m, "num_leaves": n})
        for m, n in itertools.product(t["min_child_samples"], t["num_leaves"])
    ]
    best = min(rows, key=lambda r: r["log_loss"])
    base = {"min_child_samples": best["min_child_samples"], "num_leaves": best["num_leaves"]}
    tree_rows = [run({**base, "n_estimators": n}) for n in t["n_estimators"]]
    return {
        "fit_seasons": t["fit_seasons"],
        "validation_seasons": t["validation_seasons"],
        "validation_rows": val.height,
        "note": "Uncalibrated booster. Calibration and test seasons are not used here.",
        "fixed": {k: v for k, v in cfg["lgbm"].items() if k not in base},
        "leaf_grid": rows,
        "best_leaf_setting": base,
        "tree_count_grid": tree_rows,
        "config_in_use": cfg["lgbm"],
    }


# ---------------------------------------------------------------------- calibration


def calibration_choice(model: wp.WPModel, calib: pl.DataFrame, cfg: dict) -> dict:
    """Compare calibration maps inside the calibration seasons, five folds of games.

    The trees are fixed. Each map is fitted on four fifths of the calibration games
    and scored on the remaining fifth, so the comparison never touches the test
    seasons and never scores a map on games it was fitted on.
    """
    calib = calib.filter(pl.Series(wp.scoreable(calib)) & pl.col(wp.LABEL).is_not_null())
    score, y = model.raw_score(calib), calib[wp.LABEL].to_numpy()
    games = calib["game_id"].to_numpy()
    unique = np.unique(games)
    rng = np.random.default_rng(cfg["seed"])
    fold_of_game = dict(zip(unique, rng.permutation(len(unique)) % 5, strict=True))
    fold = np.array([fold_of_game[g] for g in games])
    out = {}
    for method in ("none", "platt", "isotonic"):
        pred = np.empty_like(score)
        for k in range(5):
            cal = wp.Calibrator(method).fit(score[fold != k], y[fold != k])
            pred[fold == k] = cal.transform(score[fold == k])
        out[method] = wp.all_metrics(y, pred, cfg["metrics"]["ece_bins"])
    return {"rows": calib.height, "games": len(unique), "folds": 5, "methods": out}


def calibration_stability(
    plays: pl.DataFrame, model: wp.WPModel, test_ev: pl.DataFrame, cfg: dict
) -> dict:
    """How much does a Platt map move with the seasons it is fitted on?

    Uses only the train and calibration seasons. Each block of four consecutive
    seasons is scored by an uncalibrated booster grown on the other non-test
    seasons, then a Platt map is fitted to every single season, to every pair of
    consecutive seasons and to all of them pooled. A slope of 1 and intercept of 0
    means "leave the booster alone".

    The last entry applies the pooled map to the headline booster on the test
    seasons. It is a diagnostic added after the headline result was seen, so it is
    reported next to the headline and never in place of it.
    """
    span = seasons(cfg["split"]["train"]) + seasons(cfg["split"]["calib"])
    pool = wp.season_filter(plays, span)
    pool = pool.filter(pl.Series(wp.scoreable(pool)) & pl.col(wp.LABEL).is_not_null())
    score = np.empty(pool.height)
    season = pool["season"].to_numpy()
    size = cfg["crossfit"]["fold_size"]
    for i in range(0, len(span), size):
        held = span[i : i + size]
        rest = pool.filter(~pl.col("season").is_in(held))
        booster = wp.WPModel(cfg["lgbm"], "none", cfg["n_threads"]).fit(rest, seed=cfg["seed"])
        mask = np.isin(season, held)
        score[mask] = booster.raw_score(pool.filter(pl.Series(mask)))
    y = pool[wp.LABEL].to_numpy()

    def platt(mask: np.ndarray) -> dict:
        cal = wp.Calibrator("platt").fit(score[mask], y[mask])
        return {"slope": cal.a, "intercept": cal.b, "rows": int(mask.sum())}

    by_season = {str(s): platt(season == s) for s in span}
    by_pair = {
        f"{a}-{b}": platt((season == a) | (season == b))
        for a, b in zip(span[:-1], span[1:], strict=True)
    }
    pooled = wp.Calibrator("platt").fit(score, y)
    n_bins = cfg["metrics"]["ece_bins"]
    y_test = test_ev[wp.LABEL].to_numpy()
    pair_slopes = [v["slope"] for v in by_pair.values()]
    return {
        "seasons": [span[0], span[-1]],
        "games": pool["game_id"].n_unique(),
        "platt_by_season": by_season,
        "platt_by_consecutive_pair": by_pair,
        "pair_slope_min": min(pair_slopes),
        "pair_slope_max": max(pair_slopes),
        "platt_pooled": {"slope": pooled.a, "intercept": pooled.b, "rows": pool.height},
        "pooled_out_of_fold_metrics": {
            "none": wp.all_metrics(y, wp.Calibrator("none").transform(score), n_bins),
            "platt_pooled": wp.all_metrics(y, pooled.transform(score), n_bins),
        },
        "post_hoc_test_with_pooled_platt": wp.all_metrics(
            y_test, pooled.transform(model.raw_score(test_ev)), n_bins
        ),
        "note": "post_hoc_test_with_pooled_platt was computed after the headline test "
        "result was known. It is a diagnostic, not the reported model.",
    }


# -------------------------------------------------------------------- symmetry check


def side_consistency(df: pl.DataFrame) -> dict:
    """Calibration in the large, split by which side has the ball.

    Turn each prediction into the home team's win probability (p when the home team
    has the ball, 1 - p otherwise). If the model treated the two sides of a game
    inconsistently, the home team's average would differ between the two groups by
    more than its actual win rate does.
    """
    out = {}
    for name, home in (("home_has_ball", 1), ("away_has_ball", 0)):
        sub = df.filter(pl.col("is_home") == home)
        p, y = sub["wp_ours"].to_numpy(), sub[wp.LABEL].to_numpy()
        home_p, home_y = (p, y) if home else (1 - p, 1 - y)
        out[name] = {
            "rows": sub.height,
            "mean_predicted_home_wp": float(home_p.mean()),
            "home_win_rate": float(home_y.mean()),
            "gap": float(home_p.mean() - home_y.mean()),
        }
    out["passed"] = all(abs(v["gap"]) < 0.03 for v in out.values())
    return out


# --------------------------------------------------------------------------- figures


def reliability_band(df: pl.DataFrame, n_bins: int, n_boot: int, seed: int) -> np.ndarray:
    """95% game-bootstrap band for (observed - predicted) per bin, for our model."""
    p, y = df["wp_ours"].to_numpy(), df[wp.LABEL].to_numpy()
    _, g = np.unique(df["game_id"].to_numpy(), return_inverse=True)
    n_games = g.max() + 1
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    flat = g * n_bins + b
    shape = (n_games, n_bins)
    cnt = np.bincount(flat, minlength=n_games * n_bins).reshape(shape).astype(float)
    sum_y = np.bincount(flat, weights=y, minlength=n_games * n_bins).reshape(shape)
    sum_p = np.bincount(flat, weights=p, minlength=n_games * n_bins).reshape(shape)
    rng = np.random.default_rng(seed)
    gaps = np.empty((n_boot, n_bins))
    for i in range(n_boot):
        draw = rng.integers(0, n_games, n_games)
        c = cnt[draw].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gaps[i] = (sum_y[draw].sum(axis=0) - sum_p[draw].sum(axis=0)) / c
    return np.nanquantile(gaps, [0.025, 0.975], axis=0)


def figure_reliability(df: pl.DataFrame, metrics: dict, cfg: dict) -> None:
    style.apply()
    n_bins = cfg["metrics"]["reliability_bins"]
    y = df[wp.LABEL].to_numpy()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"wspace": 0.3})
    ax.plot([0, 1], [0, 1], color=style.MUTED, linewidth=1, linestyle=(0, (4, 3)),
            label="Perfect calibration")  # fmt: skip
    ax2.axhline(0, color=style.MUTED, linewidth=1, linestyle=(0, (4, 3)))
    low, high = reliability_band(df, n_bins, 500, cfg["seed"])
    for name in ("nflfastr", "vegas", "ours"):
        t = wp.reliability_table(y, df[MODELS[name]].to_numpy(), n_bins)
        x, obs = np.array(t["mean_pred"], float), np.array(t["observed"], float)
        ours = name == "ours"
        kw = dict(color=style.MODEL_COLORS[name], marker="o", markersize=4.5,
                  markeredgecolor=style.SURFACE, markeredgewidth=0.8,
                  linewidth=2.2 if ours else 1.6, zorder=3 if ours else 2)  # fmt: skip
        m = metrics[name]
        ax.plot(x, obs, label=LABELS[name], **kw)
        ax2.plot(x, 100 * (obs - x), **kw,
                 label=f"{LABELS[name].split(' (')[0]}: ECE {100 * m['ece']:.2f} pts")  # fmt: skip
        if name == "ours":
            ax2.fill_between(
                x, 100 * low, 100 * high, color=style.MODEL_COLORS[name], alpha=0.15,
                linewidth=0, label="Ours: 95% band (games resampled)",
            )  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Share of those plays the team went on to win")
    ax.legend(loc="upper left", fontsize=9.5)
    first, last = cfg["split"]["test"]
    style.title(
        ax,
        "Reliability on the test seasons",
        f"Test seasons {first} to {last}, {df.height:,} plays, "
        f"{df['game_id'].n_unique()} games, {n_bins} bins",
    )
    ax2.set_xlim(0, 1)
    span = max(6.0, float(np.nanmax(np.abs(np.concatenate([low, high])))) * 100 * 1.1)
    ax2.set_ylim(-span, span * 2.1)
    ax2.set_xlabel("Predicted win probability")
    ax2.set_ylabel("Observed minus predicted (percentage points)")
    ax2.legend(loc="upper left", fontsize=9.5)
    style.title(
        ax2,
        "Calibration error by bin",
        "Above zero: teams won more often than predicted",
    )
    style.save(fig, ASSETS / "m1_wp_reliability.png")


def figure_by_quarter(by_quarter: dict, cfg: dict) -> None:
    style.apply()
    quarters = [q for q in by_quarter if q != "Overtime"]
    names = ("ours", "vegas", "nflfastr")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"wspace": 0.22})
    width = 0.26
    panels = (("log_loss", 1.0, "Log loss"), ("ece", 100.0, "Expected calibration error (pts)"))
    for ax, (key, scale, ylabel) in zip(axes, panels, strict=True):
        for i, name in enumerate(names):
            vals = [scale * by_quarter[q][name][key] for q in quarters]
            xs = np.arange(len(quarters)) + (i - 1) * width
            ax.bar(xs, vals, width=width * 0.92, color=style.MODEL_COLORS[name],
                   label=LABELS[name])  # fmt: skip
            for x, v in zip(xs, vals, strict=True):
                ax.text(x, v, f"{v:.3f}"[1:] if key == "log_loss" else f"{v:.1f}", ha="center",
                        va="bottom", fontsize=7.5, color=style.INK_2)  # fmt: skip
        ax.set_xticks(np.arange(len(quarters)))
        ax.set_xticklabels(
            [f"{q}\n{by_quarter[q]['rows']:,} plays" for q in quarters], fontsize=9.5
        )
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    first, last = cfg["split"]["test"]
    axes[0].legend(loc="upper right", fontsize=9.5)
    style.title(
        axes[0],
        "Later quarters are easier to call",
        f"Log loss by quarter, test seasons {first} to {last} (lower is better)",
    )
    style.title(
        axes[1],
        "Calibration error by quarter",
        f"Regulation; overtime ({by_quarter['Overtime']['rows']} plays) is in the metrics json",
    )
    style.save(fig, ASSETS / "m1_wp_by_quarter.png")


# ------------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="rerun the hyperparameter grid")
    args = parser.parse_args()
    cfg = wp.load_config()
    seed, n_bins = cfg["seed"], cfg["metrics"]["ece_bins"]
    plays = data.load_plays()

    if args.tune:
        write_json(tune(plays, cfg), RESULTS / "m1_wp_tuning.json")

    split = {k: seasons(v) for k, v in cfg["split"].items()}
    train, calib, test = (wp.season_filter(plays, split[k]) for k in ("train", "calib", "test"))

    # Headline model.
    start = time.perf_counter()
    model = wp.WPModel(cfg["lgbm"], cfg["calibration"], cfg["n_threads"]).fit(
        train, calib, seed=seed
    )
    fit_seconds = time.perf_counter() - start
    model.save(DATA / "models" / "wp.joblib")
    start = time.perf_counter()
    wp.WPModel(cfg["lgbm"], cfg["calibration"], cfg["n_threads"]).fit(
        train, calib, seed=seed, n_estimators=100
    )
    fast_fit_seconds = time.perf_counter() - start
    print(f"fit: {fit_seconds:.1f}s ({model.n_train_rows:,} rows), "
          f"100 trees: {fast_fit_seconds:.1f}s")  # fmt: skip

    # Test-season evaluation, all three models on the same rows.
    test = test.with_columns(
        pl.Series("wp_ours", model.predict(test)).fill_nan(None),
        pl.Series("wp_ours_uncalibrated", model.predict(test, calibrated=False)).fill_nan(None),
    )
    ev = comparable(test)
    y = ev[wp.LABEL].to_numpy()
    headline = three_way(ev, n_bins)
    headline["ours_uncalibrated"] = wp.all_metrics(
        y, ev["wp_ours_uncalibrated"].to_numpy(), n_bins
    )
    games = ev["game_id"].to_numpy()
    boot = cfg["metrics"]["bootstrap_reps"]
    headline["ours_minus_nflfastr"] = wp.game_bootstrap_diff(
        y, ev["wp_ours"].to_numpy(), ev["wp"].to_numpy(), games, boot, seed
    )
    headline["ours_minus_vegas"] = wp.game_bootstrap_diff(
        y, ev["wp_ours"].to_numpy(), ev["vegas_wp"].to_numpy(), games, boot, seed
    )
    headline["reliability"] = {
        name: wp.reliability_table(y, ev[col].to_numpy(), cfg["metrics"]["reliability_bins"])
        for name, col in MODELS.items()
    }
    labelled = test.filter(pl.col(wp.LABEL).is_not_null())
    test_rows = {
        "all": test.height,
        "tied_game_rows_dropped": test.height - labelled.height,
        "incomplete_state_rows_dropped": int(labelled["wp_ours"].is_null().sum()),
        "missing_nflfastr_rows_dropped": int(
            labelled.filter(pl.col("wp_ours").is_not_null())
            .select((pl.col("wp").is_null() | pl.col("vegas_wp").is_null()).sum())
            .item()
        ),
        "evaluated": ev.height,
    }

    quarter_names = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Overtime"}
    by_quarter = {
        label: three_way(ev.filter(pl.col("qtr").clip(1, 5) == q), n_bins)
        for q, label in quarter_names.items()
    }
    subsets = {
        "scrimmage_downs": three_way(ev.filter(pl.col("down").is_not_null()), n_bins),
        "no_down_plays": three_way(ev.filter(pl.col("down").is_null()), n_bins),
        "fourth_downs": three_way(ev.filter(pl.col("down") == 4), n_bins),
    }

    # Sanity checks on hypothetical states and on the test rows.
    sanity = wp.sanity_checks(model)
    sanity["side_consistency_test_seasons"] = side_consistency(ev)
    openers = plays.filter(
        (pl.col("play_type") == "kickoff")
        & (pl.col("game_seconds_remaining") == 3600)
        & pl.col(wp.LABEL).is_not_null()
        & pl.col("season").is_in(split["train"])
    )
    sanity["kickoff_mirror"]["train_opening_kickoffs"] = openers.height
    sanity["kickoff_mirror"]["train_receiver_win_rate"] = float(openers[wp.LABEL].mean())
    sanity["all_passed"] &= sanity["side_consistency_test_seasons"]["passed"]

    # Which calibration map: decided inside the calibration seasons.
    choice = calibration_choice(model, calib, cfg)
    stability = calibration_stability(plays, model, ev, cfg)

    # Out-of-fold predictions for every season.
    start = time.perf_counter()
    oof, plan = wp.crossfit_predict(
        plays, cfg["lgbm"], cfg["calibration"], cfg["crossfit"]["fold_size"],
        cfg["crossfit"]["n_calib_seasons"], seed, cfg["n_threads"],
    )  # fmt: skip
    crossfit_seconds = time.perf_counter() - start
    assert oof.height == plays.height
    oof.write_parquet(DATA / "processed" / "wp_predictions.parquet")
    joined = comparable(plays.join(oof, on=["game_id", "play_id"], how="left", validate="1:1"))
    oof_metrics = {
        "all_seasons": three_way(joined, n_bins),
        "by_season": {
            str(s): three_way(joined.filter(pl.col("season") == s), n_bins)
            for s in sorted(joined["season"].unique().to_list())
        },
    }

    metrics = {
        "seed": seed,
        "split": cfg["split"],
        "calibration": {"configured": cfg["calibration"], **model.calibrator.describe()},
        "hyperparameters": model.params,
        "features": wp.FEATURES,
        "monotone_constraints": wp.MONOTONE,
        "rows": {
            "train_used": model.n_train_rows,
            "train_games": train["game_id"].n_unique(),
            "calib_used": model.n_calib_rows,
            "calib_games": calib["game_id"].n_unique(),
            "test": test_rows,
        },
        "fit_seconds": {
            "headline_fit": round(fit_seconds, 1),
            "fit_with_100_trees": round(fast_fit_seconds, 1),
            "crossfit_all_folds": round(crossfit_seconds, 1),
            "threads": cfg["n_threads"],
        },
        "test": headline,
        "test_by_quarter": by_quarter,
        "test_subsets": subsets,
        "calibration_choice": choice,
        "calibration_stability": stability,
        "feature_importance_gain_share": model.feature_importance(),
        "sanity_checks": sanity,
        "out_of_fold": {
            "plan": plan,
            "rows_written": oof.height,
            "rows_with_prediction": int(oof["wp_ours"].is_not_null().sum()),
            **oof_metrics,
        },
        "notes": [
            "ece uses equal-width bins of the prediction (count in metrics.ece_bins).",
            "ours_minus_* are game-level bootstrap differences; negative means ours is better.",
            "The seasons nflfastR fitted its models on are not known to this repo. If they "
            "include a season evaluated here, the nflfastR numbers for it are in-sample.",
        ],
    }
    write_json(metrics, RESULTS / "m1_wp_metrics.json")
    figure_reliability(ev, headline, cfg)
    figure_by_quarter(by_quarter, cfg)

    print(f"test rows: {ev.height:,} in {headline['games']} games")
    for name in ("ours", "ours_uncalibrated", "nflfastr", "vegas"):
        m = headline[name]
        print(f"  {name:18s} log loss {m['log_loss']:.4f}  Brier {m['brier']:.4f}  "
              f"ECE {m['ece']:.4f}")  # fmt: skip
    for name in ("ours_minus_nflfastr", "ours_minus_vegas"):
        d = headline[name]["log_loss"]
        print(f"  {name}: log loss {d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
    print(f"sanity checks passed: {sanity['all_passed']}")
    if not sanity["all_passed"]:
        failed = [k for k, v in sanity.items() if isinstance(v, dict) and not v["passed"]]
        raise SystemExit(f"sanity checks failed: {failed}")


if __name__ == "__main__":
    main()
