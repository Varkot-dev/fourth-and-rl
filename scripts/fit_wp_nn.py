"""M1 extension: neural win-probability model, deep ensemble, and what its spread means.

Outputs:
  data/models/wp_nn.joblib               headline MLP (not committed)
  data/models/wp_nn_ensemble.joblib      deep ensemble, K seeds (not committed)
  data/models/wp_nn_seeds.joblib         all seed-only members used for the spread study
  data/models/wp_nn_bagged.joblib        game-bootstrap members, a new seed each
  data/models/wp_nn_boot_fixed_seed.joblib  game-bootstrap members, one seed for all
  results/m1_wp_nn.json                  every number quoted about these models
  results/m1_wp_nn_tuning.json           the tuning record (only with --tune)
  assets/m1_wp_nn_comparison.png         reliability and smoothness, MLP versus trees
  assets/m1_wp_ensemble_spread.png       seed spread versus game-bootstrap spread

Run:  python scripts/fit_wp_nn.py           (about 7 minutes on one thread)
      python scripts/fit_wp_nn.py --tune    (also reruns the tuning sweeps, about 15 more)
      python scripts/fit_wp_nn.py --reuse   (load the saved models, redo the evaluation)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr

from fourth_and_rl import data, splits, style, wp, wp_nn
from fourth_and_rl.io import read_json, write_json
from fourth_and_rl.paths import ASSETS, DATA, RESULTS

MODELS_DIR = DATA / "models"
TUNING_PATH = RESULTS / "m1_wp_nn_tuning.json"
KEEP = [*wp.STATE_COLUMNS, wp.LABEL, "game_id", "play_id", "season", "wp", "vegas_wp"]

# Prediction columns on the test frame, and how each model is named in text.
COLUMNS = {
    "mlp": "wp_mlp",
    "mlp_platt": "wp_mlp_platt",
    "mlp_ensemble": "wp_mlp_ensemble",
    "mlp_bagged": "wp_mlp_bagged",
    "trees": "wp_trees",
    "trees_uncalibrated": "wp_trees_raw",
    "nflfastr": "wp",
    "vegas": "vegas_wp",
}
LABELS = {
    "mlp": "MLP (one network)",
    "mlp_platt": "MLP + Platt",
    "mlp_ensemble": "MLP deep ensemble",
    "mlp_bagged": "MLP bagged ensemble",
    "trees": "Trees + Platt (M1 model)",
    "trees_uncalibrated": "Trees, no Platt",
    "nflfastr": "nflfastR wp (no spread)",
    "vegas": "nflfastR vegas_wp",
}
MLP_COLOR = style.AQUA
TREE_COLOR = style.MODEL_COLORS["ours"]


def seasons(bounds: list[int]) -> list[int]:
    return list(range(bounds[0], bounds[1] + 1))


def usable(df: pl.DataFrame) -> pl.DataFrame:
    """Rows with a label and a complete state."""
    return df.filter(pl.Series(wp.scoreable(df)) & pl.col(wp.LABEL).is_not_null())


# --------------------------------------------------------------------------- tuning


def tune(plays: pl.DataFrame, cfg: dict) -> dict:
    """One-setting-at-a-time sweeps in two rounds, scored inside the training span.

    Fit on 2010 to 2018 with a tenth of those games held out for early stopping,
    score on 2019 to 2021. Every variant is trained once per tuning seed. Neither the
    calibration seasons nor the test seasons are read.
    """
    t = cfg["tuning"]
    pool = wp.season_filter(plays, seasons(t["fit_seasons"]))
    games = pool["game_id"].unique(maintain_order=True).to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    n_stop = int(round(t["early_stop_game_fraction"] * len(games)))
    stop_games = rng.permutation(games)[:n_stop]
    stop = pool.filter(pl.col("game_id").is_in(stop_games))
    fit = pool.filter(~pl.col("game_id").is_in(stop_games))
    val = usable(wp.season_filter(plays, seasons(t["validation_seasons"])))
    y = val[wp.LABEL].to_numpy()
    stop_ok = usable(stop)
    y_stop = stop_ok[wp.LABEL].to_numpy()
    n_bins = cfg["metrics"]["ece_bins"]

    def run(name: str, overrides: dict, seeds: list[int]) -> dict:
        per_seed = []
        for seed in seeds:
            model = wp_nn.NNWPModel({**cfg["mlp"], **overrides}, cfg["n_threads"])
            model.fit(fit, stop, seed=seed)
            m = wp.all_metrics(y, model.predict(val), n_bins)
            # The same network after a two-parameter Platt map fitted on the
            # early-stopping games. Needed to judge label noise fairly: flipping
            # labels pulls every prediction toward 50% by construction.
            platt = wp.Calibrator("platt").fit(model.raw_score(stop_ok), y_stop)
            best = min(model.history, key=lambda h: h["heldout_loss"])
            last = model.history[-1]
            per_seed.append(
                {
                    "seed": seed,
                    "log_loss": m["log_loss"],
                    "brier": m["brier"],
                    "ece": m["ece"],
                    "log_loss_after_platt": wp.log_loss(y, platt.transform(model.raw_score(val))),
                    "platt_slope": platt.a,
                    "epochs_run": len(model.history),
                    "epoch_kept": model.best_epoch,
                    "best_heldout_loss": best["heldout_loss"],
                    "last_epoch_train_loss": last["train_loss"],
                    "last_epoch_heldout_loss": last["heldout_loss"],
                    "fit_seconds": round(model.fit_seconds, 1),
                }
            )
        ll = np.array([r["log_loss"] for r in per_seed])
        row = {
            "name": name,
            "overrides": overrides,
            "log_loss_mean": float(ll.mean()),
            "log_loss_min": float(ll.min()),
            "log_loss_max": float(ll.max()),
            "log_loss_after_platt_mean": float(
                np.mean([r["log_loss_after_platt"] for r in per_seed])
            ),
            "ece_mean": float(np.mean([r["ece"] for r in per_seed])),
            "epoch_kept_mean": float(np.mean([r["epoch_kept"] for r in per_seed])),
            "fit_seconds_mean": float(np.mean([r["fit_seconds"] for r in per_seed])),
            "per_seed": per_seed,
        }
        print(f"  tune {name:34s} log loss {row['log_loss_mean']:.5f} "
              f"[{row['log_loss_min']:.5f}, {row['log_loss_max']:.5f}]  "
              f"after Platt {row['log_loss_after_platt_mean']:.5f}", flush=True)  # fmt: skip
        return row

    seeds = list(t["seeds"])

    def sweep(start: dict, sweeps: dict) -> list[dict]:
        """The starting point, then one setting changed at a time from it."""
        rows = [run("start", start, seeds)]
        for key, values in sweeps.items():
            rows += [run(f"{key}={v}", {**start, key: v}, seeds) for v in values]
        raw = {**start, "one_hot_down": False, "squash_margin": False}
        rows.append(run("raw inputs (no one-hot, no asinh)", raw, seeds))
        rows.append(run("no early stopping", {**start, "early_stopping": False}, seeds))
        base = rows[0]
        for r in rows:
            r["log_loss_mean_minus_start"] = r["log_loss_mean"] - base["log_loss_mean"]
            # The adoption rule: every seed of the variant beats every seed of the start.
            r["clears_start_seed_range"] = bool(r["log_loss_max"] < base["log_loss_min"])
        return rows

    rounds = {}
    for name, spec in t["rounds"].items():
        print(f" {name}: start {spec['start']}", flush=True)
        rows = sweep(dict(spec["start"]), spec["sweeps"])
        rounds[name] = {
            "start": {**cfg["mlp"], **spec["start"]},
            "rows": rows,
            "start_seed_range": rows[0]["log_loss_max"] - rows[0]["log_loss_min"],
            "adopted": [r["name"] for r in rows if r["clears_start_seed_range"]],
        }
    demo = run("memorization demo", dict(t["memorization_demo"]), seeds[:1])

    trees_tuning = read_json(RESULTS / "m1_wp_tuning.json")
    trees_best = min(trees_tuning["leaf_grid"] + trees_tuning["tree_count_grid"],
                     key=lambda r: r["log_loss"])  # fmt: skip
    return {
        "fit_seasons": t["fit_seasons"],
        "validation_seasons": t["validation_seasons"],
        "fit_games": int(fit["game_id"].n_unique()),
        "early_stop_games": int(stop["game_id"].n_unique()),
        "validation_rows": val.height,
        "validation_games": int(val["game_id"].n_unique()),
        "seeds": seeds,
        "note": "No calibration map. Calibration and test seasons are not used here. "
        "log_loss_after_platt applies a Platt map fitted on the early-stopping games.",
        "adoption_rule": "A variant replaces the starting point of its round only if its "
        "worst seed has a lower validation log loss than the starting point's best seed.",
        "rounds": rounds,
        "memorization_demo": demo,
        "trees_on_the_same_validation_seasons": {
            "source": "results/m1_wp_tuning.json (uncalibrated booster, best grid row)",
            "log_loss": trees_best["log_loss"],
            "ece": trees_best["ece"],
        },
        "config_in_use": cfg["mlp"],
    }


# -------------------------------------------------------------------------- fitting


def fit_members(
    kind: str, train: pl.DataFrame, calib: pl.DataFrame, cfg: dict, reuse: bool
) -> tuple[wp_nn.WPEnsemble, float]:
    """Train (or load) the members of one kind. Returns the ensemble and seconds spent.

    seeds:       member i has seed + i and sees all training games.
    bagged:      member i has seed + i and sees a game-level resample drawn from seed + i.
    fixed_seed:  member i has the base seed and sees the same resample as bagged member i.
    """
    names = {"seeds": "seeds", "bagged": "bagged", "fixed_seed": "boot_fixed_seed"}
    path = MODELS_DIR / f"wp_nn_{names[kind]}.joblib"
    n, seed = cfg["ensemble"]["n_spread_members"], cfg["seed"]
    if reuse and path.exists():
        ens = wp_nn.WPEnsemble.load(path)
        if len(ens.members) == n:
            return ens, float(sum(m.fit_seconds for m in ens.members))
    start = time.perf_counter()
    if kind == "fixed_seed":
        ens = wp_nn.WPEnsemble(cfg["mlp"], n, True, cfg["n_threads"])
        ens.seed = seed
        for i in range(n):
            sample = wp_nn.resample_games(train, np.random.default_rng(seed + i))
            ens.members.append(
                wp_nn.NNWPModel(cfg["mlp"], cfg["n_threads"]).fit(sample, calib, seed=seed)
            )
    else:
        ens = wp_nn.WPEnsemble(cfg["mlp"], n, kind == "bagged", cfg["n_threads"])
        ens.fit(train, calib, seed=seed)
    ens.save(path)
    seconds = time.perf_counter() - start
    print(f"fitted {n} {kind} members in {seconds:.0f}s", flush=True)
    return ens, seconds


def load_tree_bootstrap() -> tuple[list[wp.WPModel], dict]:
    """Game-bootstrap refits of the tree model, if scripts/bootstrap_decisions.py has run.

    That script writes data/models/boot/<run>/rep_<i>_wp.joblib. Each replicate grows
    the trees on resampled training games and refits the Platt map on resampled
    calibration games. The run folder with the most replicates is used.
    """
    folder = MODELS_DIR / "boot"
    runs = sorted(d for d in folder.iterdir() if d.is_dir()) if folder.exists() else []
    files_by_run = {d: sorted(d.glob("rep_*_wp.joblib")) for d in runs}
    run = max(files_by_run, key=lambda d: len(files_by_run[d]), default=None)
    files = files_by_run[run] if run is not None else []
    models, errors = [], []
    for f in files:
        try:
            models.append(wp.WPModel.load(f))
        except Exception as exc:  # noqa: BLE001  (a file may still be being written)
            errors.append(f"{f.name}: {type(exc).__name__}")
    info = {
        "folder": f"data/models/boot/{run.name}" if run is not None else "data/models/boot",
        "produced_by": "scripts/bootstrap_decisions.py (configs/bootstrap.yaml)",
        "files_found": len(files),
        "models_loaded": len(models),
        "load_errors": errors[:5],
        "trees_per_model": sorted({m.booster.num_trees() for m in models if m.booster}),
        "calibration": sorted({m.calibrator.method for m in models}),
    }
    return models, info


# ----------------------------------------------------------------------- evaluation


def smoothness_curves() -> dict[str, dict]:
    """Three one-dimensional slices through the state space that a decision engine uses."""
    spread = np.arange(-14.0, 14.5, 0.5)
    yards = np.arange(1.0, 100.0)
    diff = np.arange(-21.0, 22.0)
    return {
        "spread_at_kickoff": {
            "x": spread,
            "direction": 1,
            "xlabel": "Pregame spread for the receiving team (points, positive = favored)",
            "description": "Opening kickoff, receiving team at home, half-point steps",
            "frame": wp.state_frame(down=0, ydstogo=0, yardline_100=35,
                                    game_seconds_remaining=3600, posteam_spread=spread,
                                    is_home=1),
        },
        "yardline_at_half": {
            "x": yards,
            "direction": -1,
            "xlabel": "Yards from the opponent's end zone",
            "description": "First and 10 (or goal), tied, even spread, second-half start",
            "frame": wp.state_frame(down=1, ydstogo=np.minimum(10.0, yards), yardline_100=yards,
                                    game_seconds_remaining=1800),
        },
        "score_with_5_min": {
            "x": diff,
            "direction": 1,
            "xlabel": "Score differential for the team with the ball (points)",
            "description": "First and 10 at own 25, 5:00 left, even spread",
            "frame": wp.state_frame(yardline_100=75, game_seconds_remaining=300,
                                    score_differential=diff),
        },
    }  # fmt: skip


def curve_stats(p: np.ndarray, direction: int) -> dict:
    """How smooth is a curve sampled on an even grid?

    roughness is the sum of |second differences|: how much the slope changes from one
    grid step to the next, added up. A straight line scores 0, a gentle S-curve a
    little, and a staircase a lot, because its slope keeps switching between zero
    and a jump.
    """
    step = np.diff(p)
    wrong = step * direction < -1e-9
    return {
        "points": len(p),
        "distinct_values": int(len(np.unique(np.round(p, 9)))),
        "flat_steps_share": float(np.mean(np.abs(step) < 1e-9)),
        "largest_step": float(np.max(np.abs(step))),
        "median_abs_step": float(np.median(np.abs(step))),
        "roughness": float(np.sum(np.abs(np.diff(step)))),
        "wrong_direction_steps": int(wrong.sum()),
        "largest_wrong_direction_step": float(np.max(np.abs(step[wrong]))) if wrong.any() else 0.0,
        "range": [float(p.min()), float(p.max())],
    }


def sd_summary(sd: np.ndarray) -> dict:
    return {
        "mean": float(sd.mean()),
        "median": float(np.median(sd)),
        "p90": float(np.quantile(sd, 0.9)),
        "rms": float(np.sqrt(np.mean(sd**2))),
    }


def spread_comparison(
    members: dict[str, np.ndarray], states: pl.DataFrame, k: int, center: np.ndarray
) -> tuple[dict, dict[str, np.ndarray]]:
    """Member-to-member standard deviation of win probability, state by state.

    `members` maps a kind to an array (members, states). The seed-only spread is
    optimization noise. The bagged spread (new games and a new seed) is sampling
    noise plus optimization noise. Variances add, so sampling variance is estimated
    as bagged variance minus seed variance.
    """
    sd = {kind: p.std(axis=0, ddof=1) for kind, p in members.items()}
    var = {kind: float(np.mean(s**2)) for kind, s in sd.items()}
    out: dict = {
        "states": states.height,
        "members_per_kind": {kind: int(p.shape[0]) for kind, p in members.items()},
        "unit": "standard deviation of win probability across members (0.01 = 1 point)",
        "sd": {kind: sd_summary(s) for kind, s in sd.items()},
        "seed_sd_first_k_members": sd_summary(members["seeds"][:k].std(axis=0, ddof=1)),
        "rms_ratio_seed_over_bagged": float(np.sqrt(var["seeds"] / var["bagged"])),
        "variance_share_from_seed_noise": var["seeds"] / var["bagged"],
        "implied_sampling_sd_rms": float(np.sqrt(max(var["bagged"] - var["seeds"], 0.0))),
        "median_of_per_state_ratio_seed_over_bagged": float(
            np.median(sd["seeds"] / sd["bagged"])
        ),
        "share_of_states_where_seed_sd_is_smaller": float(np.mean(sd["seeds"] < sd["bagged"])),
        "spearman_seed_vs_bagged": float(spearmanr(sd["seeds"], sd["bagged"]).statistic),
    }
    out["rms_sd_relative_to_mlp_bagged"] = {
        kind: float(np.sqrt(v / var["bagged"])) for kind, v in var.items()
    }
    out["rms_ratio_seed_over_fixed_seed_bootstrap"] = float(
        np.sqrt(var["seeds"] / var["fixed_seed"])
    )
    for kind in ("trees_bootstrap", "trees_bootstrap_no_platt"):
        if kind in sd:
            out[f"rms_ratio_seed_over_{kind}"] = float(np.sqrt(var["seeds"] / var[kind]))
            out[f"spearman_mlp_bagged_vs_{kind}"] = float(
                spearmanr(sd["bagged"], sd[kind]).statistic
            )
    # By how lopsided the state is, and by quarter.
    edges = [0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    by_wp = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (center >= lo) & (center < hi) if hi < 1 else (center >= lo)
        if mask.sum() == 0:
            continue
        by_wp.append(
            {
                "wp_from": lo,
                "wp_to": hi,
                "states": int(mask.sum()),
                **{f"rms_sd_{kind}": float(np.sqrt(np.mean(s[mask] ** 2)))
                   for kind, s in sd.items()},
            }
        )  # fmt: skip
    out["by_predicted_wp"] = by_wp
    qtr = states["qtr"].to_numpy().clip(1, 5)
    out["by_quarter"] = {
        name: {
            "states": int((qtr == q).sum()),
            **{f"rms_sd_{kind}": float(np.sqrt(np.mean(s[qtr == q] ** 2)))
               for kind, s in sd.items()},
        }
        for q, name in {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}.items()
    }  # fmt: skip
    return out, sd


# --------------------------------------------------------------------------- figures


def figure_comparison(
    ev: pl.DataFrame, metrics: dict, curves: dict, curve_preds: dict, cfg: dict
) -> Path:
    style.apply()
    n_bins = cfg["metrics"]["reliability_bins"]
    y = ev[wp.LABEL].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.6),
                             gridspec_kw={"wspace": 0.24, "hspace": 0.52})  # fmt: skip
    series = (
        ("trees_uncalibrated", TREE_COLOR, (0, (4, 2)), 1.5, "Trees, no Platt"),
        ("trees", TREE_COLOR, "-", 2.0, "Trees + Platt (M1 model)"),
        ("mlp", MLP_COLOR, "-", 2.2, "MLP (one network)"),
    )

    ax = axes[0, 0]
    ax.axhline(0, color=style.MUTED, linewidth=1, linestyle=(0, (4, 3)))
    for name, color, dash, width, label in series:
        t = wp.reliability_table(y, ev[COLUMNS[name]].to_numpy(), n_bins)
        x, obs = np.array(t["mean_pred"], float), np.array(t["observed"], float)
        ax.plot(x, 100 * (obs - x), color=color, linestyle=dash, linewidth=width, marker="o",
                markersize=4.5, markeredgecolor=style.SURFACE, markeredgewidth=0.8,
                label=f"{label}: ECE {100 * metrics[name]['ece']:.2f} pts, "
                      f"log loss {metrics[name]['log_loss']:.4f}")  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_ylim(-7, 12)
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed minus predicted (percentage points)")
    ax.legend(loc="upper left", fontsize=9.5)
    first, last = cfg["split"]["test"]
    style.title(
        ax,
        "Calibration error by bin, test seasons",
        f"{first} to {last}, {ev.height:,} plays in {ev['game_id'].n_unique()} games, "
        f"{n_bins} bins. Zero is perfect.",
    )

    panels = (
        (axes[0, 1], "spread_at_kickoff", "Win probability against the pregame spread"),
        (axes[1, 0], "yardline_at_half", "Win probability along the field"),
        (axes[1, 1], "score_with_5_min", "Win probability against the score"),
    )
    for ax, key, heading in panels:
        c = curves[key]
        ax.step(c["x"], 100 * curve_preds[key]["trees"], where="mid", color=TREE_COLOR,
                linewidth=1.8, label="Trees + Platt (M1 model)")  # fmt: skip
        ax.plot(c["x"], 100 * curve_preds[key]["mlp"], color=MLP_COLOR, linewidth=2.2,
                label="MLP (one network)")  # fmt: skip
        ax.plot(c["x"], 100 * curve_preds[key]["mlp_ensemble"], color=MLP_COLOR, linewidth=1.5,
                linestyle=(0, (4, 2)), label="MLP deep ensemble (mean of 10)")  # fmt: skip
        ax.set_xlabel(c["xlabel"])
        ax.set_ylabel("Win probability (%)")
        ax.legend(loc="upper left" if c["direction"] > 0 else "upper right", fontsize=9.5)
        style.title(ax, heading, c["description"])
    return style.save(fig, ASSETS / "m1_wp_nn_comparison.png")


def figure_spread(
    sd: dict[str, np.ndarray], summary: dict, curve_sd: dict[str, np.ndarray], curve: dict
) -> Path:
    style.apply()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"wspace": 0.3})
    x, y = 100 * sd["bagged"], 100 * sd["seeds"]
    top = float(np.quantile(np.concatenate([x, y]), 0.995)) * 1.08
    ax.plot([0, top], [0, top], color=style.MUTED, linewidth=1, linestyle=(0, (4, 3)),
            label="Equal spread")  # fmt: skip
    ax.scatter(x, y, s=9, color=MLP_COLOR, alpha=0.35, linewidths=0,
               label=f"One of {len(x):,} test-season plays")  # fmt: skip
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("Game-bootstrap spread: new games and new seed (points of WP)")
    ax.set_ylabel("Seed-only ensemble spread (points of WP)")
    ax.legend(loc="upper left", fontsize=9.5)
    share = summary["share_of_states_where_seed_sd_is_smaller"]
    style.title(
        ax,
        "The seed ensemble is narrower than the bootstrap",
        f"RMS seed spread is {summary['rms_ratio_seed_over_bagged']:.2f} of the bootstrap "
        f"spread; smaller on {100 * share:.0f}% of plays",
    )

    lines = [
        ("seeds", MLP_COLOR, "-", "MLP, seeds only (optimization noise)"),
        ("bagged", style.INK_2, "-", "MLP, game bootstrap + seeds (total)"),
        ("fixed_seed", style.INK_2, (0, (4, 2)), "MLP, game bootstrap, one seed"),
        ("trees_bootstrap_no_platt", TREE_COLOR, "-", "Trees, game bootstrap (no Platt)"),
    ]
    for kind, color, dash, label in lines:
        if kind in curve_sd:
            ax2.plot(curve["x"], 100 * curve_sd[kind], color=color, linestyle=dash,
                     linewidth=2.0, label=label)  # fmt: skip
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.35)
    ax2.set_xlabel(curve["xlabel"])
    ax2.set_ylabel("Spread across refits (points of WP)")
    ax2.legend(loc="upper left", fontsize=9.5)
    style.title(ax2, "Spread by score, 5:00 left", curve["description"])
    return style.save(fig, ASSETS / "m1_wp_ensemble_spread.png")


# ------------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="rerun the tuning sweeps")
    parser.add_argument("--reuse", action="store_true", help="load saved models if present")
    args = parser.parse_args()
    cfg = wp_nn.load_config()
    seed, n_bins, k = cfg["seed"], cfg["metrics"]["ece_bins"], cfg["ensemble"]["k"]
    torch.set_num_threads(cfg["n_threads"])
    started = time.perf_counter()
    plays = data.load_plays().select(KEEP)

    if args.tune:
        write_json(tune(plays, cfg), TUNING_PATH)
    tuning = read_json(TUNING_PATH) if TUNING_PATH.exists() else None
    if tuning is not None and tuning["config_in_use"] != cfg["mlp"]:
        tuning["config_in_use"] = cfg["mlp"]
        write_json(tuning, TUNING_PATH)

    split = {name: seasons(v) for name, v in cfg["split"].items()}
    train, calib, test = (wp.season_filter(plays, split[s]) for s in ("train", "calib", "test"))

    # Members. Seed member 0 is the headline network: same data, same seed.
    seeds_ens, seeds_seconds = fit_members("seeds", train, calib, cfg, args.reuse)
    bagged, bagged_seconds = fit_members("bagged", train, calib, cfg, args.reuse)
    fixed, fixed_seconds = fit_members("fixed_seed", train, calib, cfg, args.reuse)
    model = seeds_ens.members[0]
    model.save(MODELS_DIR / "wp_nn.joblib")
    ensemble = wp_nn.WPEnsemble(cfg["mlp"], k, False, cfg["n_threads"])
    ensemble.members, ensemble.seed = seeds_ens.members[:k], seed
    ensemble.save(MODELS_DIR / "wp_nn_ensemble.joblib")
    trees = wp.WPModel.load(MODELS_DIR / "wp.joblib")
    trees.n_threads = 1

    # The Platt variant: the same network with a map fitted on the calibration seasons.
    calib_ok = usable(calib)
    platt = wp.Calibrator("platt").fit(model.raw_score(calib_ok), calib_ok[wp.LABEL].to_numpy())

    # Test seasons: every model on the rows used in results/m1_wp_metrics.json.
    # Each model must not have been fitted on any season it is about to be scored on.
    for name, fitted in (("network", model), ("trees", trees)):
        splits.assert_can_score(fitted.stamp, test, f"M1 {name} on the test seasons")
    for name, ens in (("seed ensemble", seeds_ens), ("bagged ensemble", bagged)):
        for member in ens.members:
            splits.assert_can_score(member.stamp, test, f"M1 {name} member on the test seasons")
    seed_test = seeds_ens.predict_members(test)
    bag_test = bagged.predict_members(test)
    mlp_score = model.raw_score(test)
    mlp_platt = np.full(test.height, np.nan)
    mlp_platt[~np.isnan(mlp_score)] = platt.transform(mlp_score[~np.isnan(mlp_score)])
    test = test.with_columns(
        pl.Series("wp_trees", trees.predict(test)).fill_nan(None),
        pl.Series("wp_trees_raw", trees.predict(test, calibrated=False)).fill_nan(None),
        pl.Series("wp_mlp", seed_test[0]).fill_nan(None),
        pl.Series("wp_mlp_platt", mlp_platt).fill_nan(None),
        pl.Series("wp_mlp_ensemble", seed_test[:k].mean(axis=0)).fill_nan(None),
        pl.Series("wp_mlp_bagged", bag_test[:k].mean(axis=0)).fill_nan(None),
    )
    keep = (
        pl.col(wp.LABEL).is_not_null()
        & pl.col("wp_trees").is_not_null()
        & pl.col("wp").is_not_null()
        & pl.col("vegas_wp").is_not_null()
    )
    keep_mask = test.select(keep.alias("k"))["k"].to_numpy()
    ev = test.filter(pl.Series(keep_mask))
    y, games = ev[wp.LABEL].to_numpy(), ev["game_id"].to_numpy()
    m1 = read_json(RESULTS / "m1_wp_metrics.json")["test"]
    table = {name: wp.all_metrics(y, ev[col].to_numpy(), n_bins) for name, col in COLUMNS.items()}
    same_rows = {
        "rows": ev.height,
        "games": int(ev["game_id"].n_unique()),
        "rows_in_m1_wp_metrics": m1["rows"],
        "trees_log_loss_here": table["trees"]["log_loss"],
        "trees_log_loss_in_m1_wp_metrics": m1["ours"]["log_loss"],
        "identical": bool(
            ev.height == m1["rows"]
            and abs(table["trees"]["log_loss"] - m1["ours"]["log_loss"]) < 1e-9
            and abs(table["vegas"]["log_loss"] - m1["vegas"]["log_loss"]) < 1e-9
        ),
    }
    if not same_rows["identical"]:
        raise SystemExit(f"test rows differ from results/m1_wp_metrics.json: {same_rows}")

    boot = cfg["metrics"]["bootstrap_reps"]
    pairs = [
        ("mlp", "trees"), ("mlp", "trees_uncalibrated"), ("mlp_platt", "trees"),
        ("mlp", "vegas"), ("mlp", "nflfastr"), ("mlp_ensemble", "mlp"),
        ("mlp_ensemble", "trees"), ("mlp_ensemble", "trees_uncalibrated"),
        ("mlp_ensemble", "vegas"), ("mlp_bagged", "mlp_ensemble"), ("mlp_bagged", "mlp"),
    ]  # fmt: skip
    differences = {
        f"{a}_minus_{b}": wp.game_bootstrap_diff(
            y, ev[COLUMNS[a]].to_numpy(), ev[COLUMNS[b]].to_numpy(), games, boot, seed
        )
        for a, b in pairs
    }
    gap = np.abs(ev["wp_mlp"].to_numpy() - ev["wp_trees_raw"].to_numpy())
    agreement = {
        "compared": "mlp versus trees_uncalibrated, absolute difference in win probability",
        "mean": float(gap.mean()),
        "median": float(np.median(gap)),
        "p90": float(np.quantile(gap, 0.9)),
        "p99": float(np.quantile(gap, 0.99)),
        "share_above_5_points": float(np.mean(gap > 0.05)),
    }
    reliability = {
        name: wp.reliability_table(y, ev[col].to_numpy(), cfg["metrics"]["reliability_bins"])
        for name, col in COLUMNS.items()
    }
    subsets = {
        "Q1": pl.col("qtr") == 1,
        "Q2": pl.col("qtr") == 2,
        "Q3": pl.col("qtr") == 3,
        "Q4": pl.col("qtr") == 4,
        "fourth_downs": pl.col("down") == 4,
        "q4_one_score": (pl.col("qtr") == 4) & (pl.col("score_differential").abs() <= 8),
    }
    by_subset = {}
    for name, expr in subsets.items():
        sub = ev.filter(expr)
        ys = sub[wp.LABEL].to_numpy()
        by_subset[name] = {
            "rows": sub.height,
            **{m: wp.all_metrics(ys, sub[COLUMNS[m]].to_numpy(), n_bins)
               for m in ("mlp", "mlp_ensemble", "trees", "trees_uncalibrated", "vegas")},
        }  # fmt: skip

    # Does averaging help? Members one at a time, then the mean of the first j.
    seed_ev, bag_ev = seed_test[:, keep_mask], bag_test[:, keep_mask]

    def averaging(p: np.ndarray) -> dict:
        singles = [wp.all_metrics(y, p[i], n_bins) for i in range(k)]
        return {
            "single_member_log_loss": [s["log_loss"] for s in singles],
            "single_member_log_loss_mean": float(np.mean([s["log_loss"] for s in singles])),
            "single_member_ece_mean": float(np.mean([s["ece"] for s in singles])),
            "single_member_ece": [s["ece"] for s in singles],
            "mean_of_first_j_members": [
                {"j": j, **wp.all_metrics(y, p[:j].mean(axis=0), n_bins)}
                for j in range(1, p.shape[0] + 1)
            ],
        }

    ensemble_results = {
        "k": k,
        "member_seeds": [seed + i for i in range(k)],
        "seeds_only": averaging(seed_ev),
        "bagged": averaging(bag_ev),
        "epochs_kept": {
            "seeds_only": [m.best_epoch for m in seeds_ens.members],
            "bagged": [m.best_epoch for m in bagged.members],
        },
    }

    # Smoothness on three slices.
    curves = smoothness_curves()
    curve_preds, smoothness = {}, {}
    for key, c in curves.items():
        preds = {
            "mlp": model.predict(c["frame"]),
            "mlp_ensemble": ensemble.predict(c["frame"]),
            "trees": trees.predict(c["frame"]),
            "trees_uncalibrated": trees.predict(c["frame"], calibrated=False),
        }
        curve_preds[key] = preds
        smoothness[key] = {
            "description": c["description"],
            "x": c["x"].tolist(),
            "stats": {name: curve_stats(p, c["direction"]) for name, p in preds.items()},
            "mean_abs_gap_mlp_vs_trees": float(np.mean(np.abs(preds["mlp"] - preds["trees"]))),
            "max_abs_gap_mlp_vs_trees": float(np.max(np.abs(preds["mlp"] - preds["trees"]))),
            "mean_abs_gap_ensemble_vs_trees": float(
                np.mean(np.abs(preds["mlp_ensemble"] - preds["trees"]))
            ),
            "curves": {name: p.tolist() for name, p in preds.items()},
        }

    # wp.sanity_checks only calls .predict on hypothetical states, so it accepts any model.
    sanity = {
        name: wp.sanity_checks(m)  # type: ignore[arg-type]
        for name, m in (("mlp", model), ("mlp_ensemble", ensemble), ("trees", trees))
    }

    # Spread: seed-only versus game bootstrap, on a fixed sample of test-season plays.
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(ev.height, cfg["ensemble"]["n_states"], replace=False))
    states = ev[pick]
    tree_boot, tree_boot_info = load_tree_bootstrap()
    kinds = {
        "seeds": seed_ev[:, pick],
        "bagged": bag_ev[:, pick],
        "fixed_seed": fixed.predict_members(states),
    }
    score_curve = curves["score_with_5_min"]
    curve_members = {
        "seeds": seeds_ens.predict_members(score_curve["frame"]),
        "bagged": bagged.predict_members(score_curve["frame"]),
        "fixed_seed": fixed.predict_members(score_curve["frame"]),
    }
    if len(tree_boot) >= 5:
        for m in tree_boot:
            m.n_threads = 1
        frame = score_curve["frame"]
        kinds["trees_bootstrap"] = np.stack([m.predict(states) for m in tree_boot])
        kinds["trees_bootstrap_no_platt"] = np.stack(
            [m.predict(states, calibrated=False) for m in tree_boot]
        )
        curve_members["trees_bootstrap"] = np.stack([m.predict(frame) for m in tree_boot])
        curve_members["trees_bootstrap_no_platt"] = np.stack(
            [m.predict(frame, calibrated=False) for m in tree_boot]
        )
    spread, sd = spread_comparison(kinds, states, k, seed_ev[:k, pick].mean(axis=0))
    spread["tree_bootstrap"] = tree_boot_info
    spread["tree_bootstrap"]["used"] = "trees_bootstrap" in kinds
    curve_sd = {kind: p.std(axis=0, ddof=1) for kind, p in curve_members.items()}
    spread["on_score_curve_5_min"] = {
        "x": score_curve["x"].tolist(),
        **{f"sd_{kind}": s.tolist() for kind, s in curve_sd.items()},
    }
    spread["on_curves_rms_sd"] = {
        key: {
            "seeds": float(np.sqrt(np.mean(
                seeds_ens.predict_members(c["frame"]).std(axis=0, ddof=1) ** 2))),
            "bagged": float(np.sqrt(np.mean(
                bagged.predict_members(c["frame"]).std(axis=0, ddof=1) ** 2))),
        }
        for key, c in curves.items()
    }  # fmt: skip
    spread["notes"] = [
        "seeds: same training games, seed + i. bagged: games resampled with replacement and "
        "seed + i. fixed_seed: the same resamples as bagged, every member with the base seed.",
        "The early-stopping seasons are not resampled in any of the three.",
        "trees_bootstrap: the tree model refitted on resampled training games, with its "
        "Platt map refitted on resampled calibration games. trees_bootstrap_no_platt: the "
        "same refits without the map, which is the like-for-like comparison with the MLP "
        "bootstrap, where the 2022 to 2023 seasons are not resampled.",
        "The states are a seeded random sample of the evaluated test-season plays. Only "
        "their game states are used here, never their outcomes.",
    ]

    results = {
        "seed": seed,
        "split": cfg["split"],
        "torch_version": torch.__version__,
        "threads": cfg["n_threads"],
        "features": wp.FEATURES,
        "mlp": {
            "params": model.params,
            "network_inputs": model.pre.n_out if model.pre else None,
            "parameters": int(sum(p.numel() for p in model.net.parameters())) if model.net else 0,
            "train_rows": model.n_train_rows,
            "train_games": int(train["game_id"].n_unique()),
            "early_stopping_rows": model.n_stop_rows,
            "early_stopping_games": int(calib["game_id"].n_unique()),
            "epoch_kept": model.best_epoch,
            "epochs_run": len(model.history),
            "history": model.history,
            "platt_variant": platt.describe(),
        },
        "training_seconds": {
            "headline_mlp": round(model.fit_seconds, 1),
            "per_member_mean": round(
                float(np.mean([m.fit_seconds for m in seeds_ens.members])), 1
            ),
            "seed_members": round(seeds_seconds, 1),
            "bagged_members": round(bagged_seconds, 1),
            "fixed_seed_bootstrap_members": round(fixed_seconds, 1),
            "members_per_kind": cfg["ensemble"]["n_spread_members"],
            "models_reused_from_disk": bool(args.reuse),
            "this_run_wall_clock": None,
        },
        "test_rows": same_rows,
        "test": table,
        "test_differences": differences,
        "test_agreement": agreement,
        "test_reliability": reliability,
        "test_subsets": by_subset,
        "ensemble": ensemble_results,
        "smoothness": smoothness,
        "sanity_checks": sanity,
        "spread": spread,
        "tuning": tuning,
        "notes": [
            "Every model is scored on the same rows as results/m1_wp_metrics.json.",
            "a_minus_b entries are game-level bootstrap differences with 95% intervals; "
            "negative means a is better.",
            "mlp is one network (the base seed), early stopped on the calibration seasons, "
            "with no calibration map. mlp_platt adds a Platt map fitted on those same seasons.",
            "mlp_ensemble is the mean of the first k seed-only members; mlp_bagged is the "
            "mean of the first k game-bootstrap members.",
            "ece uses equal-width bins of the prediction (count in metrics.ece_bins).",
        ],
    }
    results["training_seconds"]["this_run_wall_clock"] = round(time.perf_counter() - started, 1)
    write_json(results, RESULTS / "m1_wp_nn.json")
    figure_comparison(ev, table, curves, curve_preds, cfg)
    figure_spread(sd, spread, curve_sd, score_curve)

    print(f"test rows: {ev.height:,} in {same_rows['games']} games")
    for name in COLUMNS:
        m = table[name]
        print(f"  {name:20s} log loss {m['log_loss']:.4f}  Brier {m['brier']:.4f}  "
              f"ECE {m['ece']:.4f}")  # fmt: skip
    for name, d in differences.items():
        d = d["log_loss"]
        print(f"  {name}: log loss {d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
    print(f"seed spread / bootstrap spread (RMS): {spread['rms_ratio_seed_over_bagged']:.2f}")


if __name__ == "__main__":
    main()
