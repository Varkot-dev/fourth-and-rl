"""M8: team ratings from earlier games only, and whether they predict beyond the spread.

Stage 1 (development seasons only, 2010 to 2023):
  * league curves for each season fitted on earlier seasons only (2010 is used only to fit
    the curves of 2011; ratings start in 2011);
  * the rating filter's constants tuned by season-blocked validation (four blocks of three
    seasons), and the nested choice per block for honest out-of-fold numbers;
  * conversion and drive-outcome models with the spread only and with ratings, compared by
    out-of-fold log loss with game-bootstrap intervals; the primary rating variant of each
    is chosen by a rule fixed in the config;
  * the evaluation plan is written to results/m8_team_strength.json BEFORE any test-season
    row is read.
Stage 2 (once): the 2024 to 2025 seasons, read through `splits.load_test_plays`, scored
with the frozen constants and models. A second scoring is refused.

Outputs:
  results/m8_team_strength.json
  data/models/team_strength.joblib      (frozen models for scripts/questions/q3_team_aware.py)
  assets/m8_team_ratings.png

Run:  python scripts/m8_team_strength.py             (about 2 minutes)
      python scripts/m8_team_strength.py --dry-run   (2022 to 2023 as a pseudo test, written
                                                      to the scratch directory; no test data)
      python scripts/m8_team_strength.py --figure-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from fourth_and_rl import splits, style  # noqa: E402
from fourth_and_rl import team_strength as ts  # noqa: E402
from fourth_and_rl.components import config_hash  # noqa: E402
from fourth_and_rl.fingerprint import data_fingerprint  # noqa: E402
from fourth_and_rl.io import read_json, write_json  # noqa: E402
from fourth_and_rl.paths import ASSETS, DATA, RESULTS  # noqa: E402

RESULT = RESULTS / "m8_team_strength.json"
BUNDLE = DATA / "models" / "team_strength.joblib"
FIGURE = ASSETS / "m8_team_ratings.png"
TEST_PURPOSE = (
    "M8 team strength: score the preregistered spread-only and team-rating conversion and "
    "drive models once on 2024-2025 (scripts/m8_team_strength.py)"
)


# --------------------------------------------------------------------------- stage 1


def prepare(dev: pl.DataFrame, cfg: dict) -> dict[str, Any]:
    seasons = sorted(splits.seasons_in(dev))
    start = seasons[0] + 1
    bases = ts.fit_bases(dev, range(start, seasons[-1] + 1), cfg)
    conv, drives = ts.prepare_samples(dev, bases, cfg, start)
    sched = ts.Schedule.from_frame(ts.schedule(dev.filter(pl.col("season") >= start)))
    return {"start": start, "last": seasons[-1], "bases": bases, "conv": conv,
            "drives": drives, "sched": sched}  # fmt: skip


def tune(d: dict, cfg: dict, blocks: list[list[int]]) -> dict[str, Any]:
    """Tune both filters; final constants and the nested per-block choices."""
    out: dict[str, Any] = {}
    fcfg = cfg["filter"]
    for metric, grid_key, sample in (("conv", "conversion", d["conv"]),
                                     ("drive", "drive", d["drives"])):  # fmt: skip
        tuned = ts.tune_filter(metric, sample, d["sched"], fcfg["grids"][grid_key], blocks,
                               d["start"], fcfg["league_sd"][grid_key])  # fmt: skip
        seasons = tuned["seasons"]
        none = ts.no_effect_index(tuned)
        final, fi = ts.choose_params(tuned, seasons)
        total = tuned["season_loss"].sum(axis=1)
        n = int(tuned["n_rows"].sum())
        folds = []
        for blk in blocks:
            inside = [s for s in seasons if blk[0] <= s <= blk[1]]
            others = [s for s in seasons if s not in inside]
            prm, k = ts.choose_params(tuned, others)
            cols = [seasons.index(s) for s in inside]
            n_blk = int(tuned["n_rows"][cols].sum())
            gain = (tuned["season_loss"][none, cols].sum()
                    - tuned["season_loss"][k, cols].sum()) / n_blk  # fmt: skip
            folds.append({"block": blk, "params": prm.as_dict(), "n_rows": n_blk,
                          "held_out_gain_per_row": float(gain)})  # fmt: skip
        ranked = np.argsort(total)[:5]
        out[metric] = {
            "tuned": tuned, "final": final, "fold_params": [ts.FilterParams(**f["params"])
                                                           for f in folds],
            "summary": {
                "score": "log loss per play" if metric == "conv" else
                         "squared error per drive (points squared)",
                "grid_points": len(tuned["params"]),
                "final_params": final.as_dict(),
                "no_team_effect_score_per_row": float(total[none] / n),
                "final_score_per_row": float(total[fi] / n),
                "in_sample_gain_per_row": float((total[none] - total[fi]) / n),
                "best_five": [{**tuned["params"][j].as_dict(),
                               "gain_per_row": float((total[none] - total[j]) / n)}
                              for j in ranked],
                "folds": folds,
            },
        }  # fmt: skip
    return out


def oof_predictions(d: dict, tuned: dict, cfg: dict, blocks: list[list[int]]
                    ) -> dict[str, Any]:  # fmt: skip
    """Out-of-fold predictions of every variant, block by block, with each block's ratings
    computed from constants chosen without that block."""
    mcfg = cfg["models"]
    first = int(mcfg["first_fit_season"])
    conv_rows, drive_rows = [], []
    for b, blk in enumerate(blocks):
        params = {"conv": tuned["conv"]["fold_params"][b],
                  "drive": tuned["drive"]["fold_params"][b]}  # fmt: skip
        ratings, _, _ = ts.compute_ratings(d["conv"], d["drives"], d["sched"], params, d["start"])
        conv = ts.attach_ratings(d["conv"], ratings)
        drv = ts.attach_ratings(d["drives"], ratings)
        inb = pl.col("season").is_between(blk[0], blk[1])
        train_c = conv.filter((pl.col("season") >= first) & ~inb)
        test_c = conv.filter(inb)
        train_d = drv.filter((pl.col("season") >= first) & ~inb)
        test_d = drv.filter(inb)
        cols_c = {"game_id": test_c["game_id"], "season": test_c["season"],
                  "y": test_c["converted"]}  # fmt: skip
        for name, feats in mcfg["conversion_variants"].items():
            m = ts.ConversionModel(list(feats), l2=mcfg["l2"]).fit(train_c)
            cols_c[name] = m.predict(test_c)
        m = ts.ConversionModel(list(mcfg["diagnostic_no_spread"]), use_spread=False,
                               l2=mcfg["l2"]).fit(train_c)  # fmt: skip
        cols_c["diagnostic_no_spread"] = m.predict(test_c)
        conv_rows.append(pl.DataFrame(cols_c))
        cols_d: dict[str, Any] = {"game_id": test_d["game_id"], "season": test_d["season"],
                                  "outcome": test_d["outcome"]}  # fmt: skip
        knots = tuple(mcfg["drive_yardline_knots"])
        for name, feats in mcfg["drive_variants"].items():
            m = ts.DriveModel(list(feats), knots=knots, l2=mcfg["l2"]).fit(train_d)
            cols_d[name] = ts.drive_log_loss(test_d["outcome"].to_list(), m.predict(test_d))
        m = ts.DriveModel(list(mcfg["diagnostic_no_spread"]), knots=knots, l2=mcfg["l2"],
                          use_spread=False).fit(train_d)  # fmt: skip
        cols_d["diagnostic_no_spread"] = ts.drive_log_loss(test_d["outcome"].to_list(),
                                                           m.predict(test_d))  # fmt: skip
        drive_rows.append(pl.DataFrame(cols_d))
    return {"conv": pl.concat(conv_rows), "drive": pl.concat(drive_rows)}


def compare(losses: dict[str, np.ndarray], games: np.ndarray, seasons: np.ndarray,
            base: str, cfg: dict, seed: int, blocks: list[list[int]] | None = None
            ) -> dict[str, Any]:  # fmt: skip
    """Mean log loss of each variant and its difference from `base` (negative = better),
    with a game-bootstrap interval, overall and by block (or by season)."""
    ev = cfg["evaluation"]
    out: dict[str, Any] = {}
    for name, ll in losses.items():
        row = {"log_loss": float(ll.mean()), "n": int(ll.size)}
        if name != base:
            row["minus_spread_only"] = ts.game_bootstrap_mean_diff(
                games, ll - losses[base], ev["n_boot"], seed, ev["interval"]
            )
            groups = blocks or [[int(s), int(s)] for s in np.unique(seasons)]
            row["by_block"] = {
                f"{a}-{b}": float((ll - losses[base])[(seasons >= a) & (seasons <= b)].mean())
                for a, b in groups
            }
        out[name] = row
    return out


def dev_cv(oof: dict, cfg: dict, blocks: list[list[int]]) -> dict[str, Any]:
    seed = int(cfg["seed"])
    c = oof["conv"]
    y = c["y"].to_numpy()
    names = [*cfg["models"]["conversion_variants"], "diagnostic_no_spread"]
    conv_ll = {k: ts.conversion_log_loss(y, c[k].to_numpy()) for k in names}
    d = oof["drive"]
    names_d = [*cfg["models"]["drive_variants"], "diagnostic_no_spread"]
    drive_ll = {k: d[k].to_numpy() for k in names_d}
    return {
        "conversion": compare(conv_ll, c["game_id"].to_numpy(), c["season"].to_numpy(),
                              "spread_only", cfg, seed, blocks),
        "drive": compare(drive_ll, d["game_id"].to_numpy(), d["season"].to_numpy(),
                         "spread_only", cfg, seed + 1, blocks),
    }  # fmt: skip


def implied_effects(models: dict, groups: dict, d_conv: pl.DataFrame) -> dict[str, Any]:
    """What the fitted models imply for top versus bottom units, at an even spread."""
    bl = float(d_conv["base_logit"].mean())
    avg = groups["average"]

    def conv_p(model: ts.ConversionModel, off: dict, dfn: dict) -> float:
        f = {k: np.array([v]) for k, v in ts.unit_features(off, dfn).items()}
        return float(ts._sigmoid(bl + model.linear_predictor(np.zeros(1), f))[0])

    def drive_ep(model: ts.DriveModel, off: dict, dfn: dict, start: int = 75) -> dict:
        f = {k: np.array([v]) for k, v in ts.unit_features(off, dfn).items()}
        p = model.proba(np.array([start]), np.zeros(1), f)[0]
        return {"p_td": float(p[2]), "p_fg": float(p[1]), "points": float(7 * p[2] + 3 * p[1])}

    out: dict[str, Any] = {"average_base_conversion": float(ts._sigmoid(bl))}
    for key in ("conv_with_spread", "conv_no_spread"):
        m = models[key]
        out[key] = {
            "offense_top_minus_bottom_points": 100 * (conv_p(m, groups["top"], avg)
                                                      - conv_p(m, groups["bottom"], avg)),
            "defense_worst_minus_best_points": 100 * (conv_p(m, avg, groups["worst"])
                                                      - conv_p(m, avg, groups["best"])),
        }  # fmt: skip
    for key in ("drive_with_spread", "drive_no_spread"):
        m = models[key]
        top, bot = drive_ep(m, groups["top"], avg), drive_ep(m, groups["bottom"], avg)
        worst, best = drive_ep(m, avg, groups["worst"]), drive_ep(m, avg, groups["best"])
        out[key] = {
            "from_own_25": {"top_offense": top, "bottom_offense": bot,
                            "vs_worst_defense": worst, "vs_best_defense": best},
            "offense_top_minus_bottom_points_per_drive": top["points"] - bot["points"],
            "defense_worst_minus_best_points_per_drive": worst["points"] - best["points"],
        }  # fmt: skip
    return out


def rating_summary(season_end: pl.DataFrame, seasons: list[int]) -> dict[str, Any]:
    df = season_end.filter(pl.col("season").is_in(seasons))
    out = {}
    for c in ("conv_off", "conv_def", "drive_off", "drive_def"):
        out[c] = {"sd_across_team_seasons": float(df[c].std()),
                  "mean_posterior_sd": float(df[f"{c}_sd"].mean())}  # fmt: skip
    return out


def stage_dev(dev: pl.DataFrame, cfg: dict, blocks: list[list[int]]) -> dict[str, Any]:
    d = prepare(dev, cfg)
    print(f"samples: {d['conv'].height} conversion plays, {d['drives'].height} drives")
    tuned = tune(d, cfg, blocks)
    print("final constants:", {k: v["final"].as_dict() for k, v in tuned.items()})
    oof = oof_predictions(d, tuned, cfg, blocks)
    cv = dev_cv(oof, cfg, blocks)
    mcfg = cfg["models"]
    order_c = [k for k in mcfg["conversion_variants"] if k != "spread_only"]
    order_d = [k for k in mcfg["drive_variants"] if k != "spread_only"]
    primary = {
        "conversion": ts.choose_variant({k: cv["conversion"][k]["log_loss"] for k in order_c},
                                        order_c, mcfg["min_improvement"]),
        "drive": ts.choose_variant({k: cv["drive"][k]["log_loss"] for k in order_d},
                                   order_d, mcfg["min_improvement"]),
    }  # fmt: skip
    print("primary variants:", primary)

    final_params = {k: tuned[k]["final"] for k in ("conv", "drive")}
    ratings, season_end, _ = ts.compute_ratings(d["conv"], d["drives"], d["sched"],
                                                final_params, d["start"])  # fmt: skip
    first = int(mcfg["first_fit_season"])
    conv = ts.attach_ratings(d["conv"], ratings).filter(pl.col("season") >= first)
    drv = ts.attach_ratings(d["drives"], ratings).filter(pl.col("season") >= first)
    knots = tuple(mcfg["drive_yardline_knots"])
    l2 = mcfg["l2"]
    models = {
        "conv_spread": ts.ConversionModel([], l2=l2).fit(conv),
        "conv_team": ts.ConversionModel(list(mcfg["conversion_variants"][primary["conversion"]]),
                                        l2=l2).fit(conv),
        "conv_with_spread": ts.ConversionModel(list(mcfg["diagnostic_no_spread"]), l2=l2).fit(conv),
        "conv_no_spread": ts.ConversionModel(list(mcfg["diagnostic_no_spread"]), use_spread=False,
                                             l2=l2).fit(conv),
        "drive_spread": ts.DriveModel([], knots=knots, l2=l2).fit(drv),
        "drive_team": ts.DriveModel(list(mcfg["drive_variants"][primary["drive"]]), knots=knots,
                                    l2=l2).fit(drv),
        "drive_with_spread": ts.DriveModel(list(mcfg["diagnostic_no_spread"]), knots=knots,
                                           l2=l2).fit(drv),
        "drive_no_spread": ts.DriveModel(list(mcfg["diagnostic_no_spread"]), knots=knots, l2=l2,
                                         use_spread=False).fit(drv),
    }  # fmt: skip
    fit_seasons = list(range(first, d["last"] + 1))
    groups = ts.group_ratings(season_end, fit_seasons)
    spread_fit = ts.fit_spread_on_ratings(drv.unique(["game_id", "posteam"]))
    consts = ts.drive_constants(drv, cfg["decision"]["drive_seconds_fallback"])
    stamp = splits.fit_stamp(conv, "M8 effect models")
    stamp.update({"data_fingerprint": data_fingerprint(dev), "config_hash": config_hash(cfg),
                  "primary": primary})  # fmt: skip
    effects = ts.TeamEffects(models["conv_spread"], models["conv_team"], models["drive_spread"],
                             models["drive_team"], final_params, consts, stamp)  # fmt: skip
    drive_coef = {k: m.rating_coef() or None for k, m in models.items() if k.startswith("drive")}
    plan = {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "test_seasons": list(splits.test_seasons()),
        "filter_constants": {k: v.as_dict() for k, v in final_params.items()},
        "residual_bases": "fitted on development seasons 2010-2023 for both test seasons",
        "ratings": "pre-game, from the filter run forward through the test seasons with the "
                   "frozen constants (online updating, no refit)",
        "models_fitted_on": fit_seasons,
        "primary_conversion_variant": primary["conversion"],
        "primary_drive_variant": primary["drive"],
        "comparison": "mean log loss per row, primary minus spread_only, 95% interval from "
                      f"{cfg['evaluation']['n_boot']} resamples of test-season games",
        "conversion_rows": "third and fourth downs, 1 to 3 to go, runs and passes",
        "drive_rows": "regulation drives starting with more than "
                      f"{cfg['samples']['drive_min_half_seconds']} s left in the half",
        "reading": "an interval entirely below 0 is an improvement; an interval that "
                   "includes 0, or a mean within 0.001 of 0, means the spread already carries "
                   "what the ratings know",
    }  # fmt: skip
    plan["hash"] = hashlib.sha1(json.dumps(plan, sort_keys=True, default=str).encode()
                                ).hexdigest()[:16]  # fmt: skip
    joblib.dump({"effects": effects, "models": models, "groups": groups,
                 "spread_fit": spread_fit, "plan_hash": plan["hash"]}, BUNDLE_PATH[0])  # fmt: skip
    result = {
        "what": "M8 team ratings from earlier games only, and whether they predict beyond "
        "the pregame spread (scripts/m8_team_strength.py)",
        "seasons": {
            "ratings_from": d["start"],
            "models_fitted_on": fit_seasons,
            "validation_blocks": blocks,
        },
        "samples": {
            "conversion_plays": d["conv"].height,
            "drives": d["drives"].height,
            "games": len(d["sched"].game_ids),
        },
        "filter": {k: v["summary"] for k, v in tuned.items()},
        "ratings_summary": rating_summary(season_end, fit_seasons),
        "development_cv": cv,
        "primary": primary,
        "final_models": {
            "conversion_coef": {
                k: models[k].coef_
                for k in ("conv_spread", "conv_team", "conv_with_spread", "conv_no_spread")
            },
            "drive_rating_coef_by_class": drive_coef,
        },  # fmt: skip
        "spread_contains": {
            "spread_on_ratings": spread_fit,
            "implied_effects": implied_effects(models, groups, conv),
        },
        "groups": groups,
        "drive_constants": consts,
        "evaluation_plan": plan,
        "test": "pending",
        "_ratings_dev": ratings,
    }
    return result


# --------------------------------------------------------------------------- stage 2


def stage_test(dev: pl.DataFrame, test: pl.DataFrame, cfg: dict, result: dict) -> dict:
    bundle = joblib.load(BUNDLE_PATH[0])
    if bundle["plan_hash"] != result["evaluation_plan"]["hash"]:
        raise RuntimeError("the saved models do not match the written evaluation plan")
    eff: ts.TeamEffects = bundle["effects"]
    test_seasons = sorted(splits.seasons_in(test))
    dev_seasons = sorted(splits.seasons_in(dev))
    start = dev_seasons[0] + 1
    bases = ts.fit_bases(dev, range(start, test_seasons[-1] + 1), cfg)
    plays = pl.concat([dev, test], how="diagonal_relaxed")
    conv, drives = ts.prepare_samples(plays, bases, cfg, start)
    sched = ts.Schedule.from_frame(ts.schedule(plays.filter(pl.col("season") >= start)))
    ratings, season_end, _ = ts.compute_ratings(conv, drives, sched, eff.filter_params, start)
    # Online filtering: adding later seasons cannot change an earlier rating.
    dev_r = result.pop("_ratings_dev", None)
    if dev_r is not None:
        joined = dev_r.join(ratings, on=["game_id", "team"], suffix="_all")
        assert np.allclose(joined["drive_off"], joined["drive_off_all"])
        assert np.allclose(joined["conv_def"], joined["conv_def_all"])
    in_test = pl.col("season").is_in(test_seasons)
    conv_t = ts.attach_ratings(conv.filter(in_test), ratings)
    drv_t = ts.attach_ratings(drives.filter(in_test), ratings)
    splits.assert_can_score(eff.stamp, conv_t, "M8 conversion models on test seasons")
    splits.assert_can_score(eff.stamp, drv_t, "M8 drive models on test seasons")
    y = conv_t["converted"].to_numpy()
    conv_ll = {"spread_only": ts.conversion_log_loss(y, eff.conv_spread.predict(conv_t)),
               result["primary"]["conversion"]: ts.conversion_log_loss(
                   y, eff.conv_team.predict(conv_t))}  # fmt: skip
    out_d = drv_t["outcome"].to_list()
    drive_ll = {
        "spread_only": ts.drive_log_loss(out_d, eff.drive_spread.predict(drv_t)),
        result["primary"]["drive"]: ts.drive_log_loss(out_d, eff.drive_team.predict(drv_t)),
    }
    seed = int(cfg["seed"]) + 7
    lions = cfg["decision"]["lions_team"]
    weekly = (ratings.filter((pl.col("team") == lions) & (pl.col("season") >= 2021))
              .sort("season", "week"))  # fmt: skip
    return {
        "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "plan_hash": result["evaluation_plan"]["hash"],
        "seasons": test_seasons,
        "conversion": compare(conv_ll, conv_t["game_id"].to_numpy(), conv_t["season"].to_numpy(),
                              "spread_only", cfg, seed),
        "drive": compare(drive_ll, drv_t["game_id"].to_numpy(), drv_t["season"].to_numpy(),
                         "spread_only", cfg, seed + 1),
        "observed_conversion_rate": float(y.mean()),
        "predicted_conversion_rate_spread_only": float(eff.conv_spread.predict(conv_t).mean()),
        "ratings_season_end": {
            str(s): season_end.filter(pl.col("season") == s).sort("team").to_dicts()
            for s in test_seasons
        },
        "lions_weekly": weekly.select("season", "week", "drive_off", "drive_off_sd",
                                      "drive_def", "drive_def_sd", "conv_off", "conv_off_sd",
                                      "conv_def", "conv_def_sd").to_dicts(),
    }  # fmt: skip


# --------------------------------------------------------------------------- figure


def draw(result: dict, path: Path) -> Path:
    style.apply()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.7, 1]})
    test = result["test"]
    rows = pl.DataFrame(test["lions_weekly"])
    x = np.arange(rows.height)
    for col, color, label in (("drive_off", style.BLUE, "Offense: points per drive above average"),
                              ("drive_def", style.RED,
                               "Defense: points per drive allowed above average")):  # fmt: skip
        m, sd = rows[col].to_numpy(), rows[f"{col}_sd"].to_numpy()
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, m, color=color, label=label)
    seasons = rows["season"].to_numpy()
    for s in np.unique(seasons):
        idx = np.nonzero(seasons == s)[0]
        ax.axvline(idx[0] - 0.5, color=style.AXIS, linewidth=0.8)
        ax.text(
            idx.mean(),
            0.97,
            str(s),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color=style.INK_2,
            fontsize=10,
        )
    ax.axhline(0, color=style.INK_2, linewidth=0.8)
    ax.set_xticks([])
    ax.set_xlabel("Detroit's games, 2021 to 2025 (rating before each game, band = 1 SD)")
    ax.set_ylabel("Points per drive")
    ax.legend(loc="lower left", fontsize=9.5)
    style.title(
        ax,
        "Detroit, rated only from earlier games",
        "Kalman-filter ratings; a bad defense has a positive rating",
    )

    labels, means, los, his, colors = [], [], [], [], []
    for model, key, primary in (
        ("Conversion", "conversion", result["primary"]["conversion"]),
        ("Drive outcome", "drive", result["primary"]["drive"]),
    ):
        for where, block in (("development, out of fold", result["development_cv"][key]),
                             (f"{test['seasons'][0]}-{test['seasons'][-1]} test",
                              test[key])):  # fmt: skip
            r = block[primary]["minus_spread_only"]
            labels.append(f"{model}\n{where}")
            means.append(1000 * r["mean"])
            los.append(1000 * r["lo"])
            his.append(1000 * r["hi"])
            colors.append(style.BLUE if "test" in where else style.INK_2)
    yy = np.arange(len(labels))[::-1]
    m, lo, hi = np.array(means), np.array(los), np.array(his)
    for i in range(len(labels)):
        bx.errorbar(m[i], yy[i], xerr=[[m[i] - lo[i]], [hi[i] - m[i]]], fmt="o",
                    color=colors[i], elinewidth=2, capsize=4, markersize=6)  # fmt: skip
    bx.axvline(0, color=style.INK_2, linewidth=0.9)
    bx.set_yticks(yy)
    bx.set_yticklabels(labels, fontsize=9.5)
    bx.set_xlabel("Log loss, ratings model minus spread-only model (per 1,000 rows)")
    bx.grid(axis="y", visible=False)
    style.title(bx, "Do ratings beat the spread?", "Below 0 is better; bars are 95% intervals")
    fig.tight_layout()
    return style.save(fig, path)


# --------------------------------------------------------------------------- main

BUNDLE_PATH = [BUNDLE]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="use 2022-2023 as a pseudo test (development data only)",
    )
    ap.add_argument("--figure-only", action="store_true")
    ap.add_argument(
        "--rescore-reason",
        default=None,
        help="score the test seasons again (logged); needs a written reason",
    )
    args = ap.parse_args()
    cfg = ts.load_config()
    result_path, figure_path = RESULT, FIGURE
    if args.dry_run:
        scratch = Path(os.environ.get("M8_SCRATCH", DATA / "scratch_m8"))
        scratch.mkdir(parents=True, exist_ok=True)
        result_path = scratch / "m8_team_strength_dryrun.json"
        figure_path = scratch / "m8_team_ratings_dryrun.png"
        BUNDLE_PATH[0] = scratch / "team_strength_dryrun.joblib"
    if args.figure_only:
        draw(read_json(result_path), figure_path)
        return
    if (not args.dry_run and result_path.exists()
            and isinstance(read_json(result_path).get("test"), dict)
            and not args.rescore_reason):  # fmt: skip
        raise SystemExit(f"{result_path} already holds the test-season scoring; the test "
                         "seasons are scored once. Use --figure-only to redraw.")  # fmt: skip

    dev_all = splits.load_dev_plays()
    blocks = [list(b) for b in cfg["filter"]["validation_blocks"]]
    if args.dry_run:
        pseudo = [2022, 2023]
        dev = dev_all.filter(pl.col("season") < pseudo[0])
        test = dev_all.filter(pl.col("season").is_in(pseudo))
        blocks = [b for b in blocks if b[1] < pseudo[0]]
    else:
        dev, test = dev_all, None
    result = stage_dev(dev, cfg, blocks)
    saved = {k: v for k, v in result.items() if not k.startswith("_")}
    write_json(saved, result_path)  # the plan is on disk before any test row is read
    print(f"evaluation plan written to {result_path}")
    if test is None:
        reason = args.rescore_reason
        purpose = TEST_PURPOSE if not reason else f"{TEST_PURPOSE}; RESCORE: {reason}"
        test = splits.load_test_plays(purpose)
    result["test"] = stage_test(dev, test, cfg, result)
    write_json({k: v for k, v in result.items() if not k.startswith("_")}, result_path)
    draw(result, figure_path)
    t = result["test"]
    for key in ("conversion", "drive"):
        r = t[key][result["primary"][key]]["minus_spread_only"]
        print(
            f"test {key}: {1000 * r['mean']:+.2f} per 1000 ({1000 * r['lo']:+.2f} to "
            f"{1000 * r['hi']:+.2f})"
        )


if __name__ == "__main__":
    main()
