"""Field-goal make rate with the selection bias taken out: who kicks, and in what conditions.

Post hoc. The world-model run on branch claude/world-model-colab
(results/world_model/colab_t4_v2) found its simulated long-range make rate above the raw
training rate. The raw rate is itself selected: coaches send strong kickers out for long
tries in good conditions (pushes it up), and end-of-half heaves are kicked because time
ran out (pushes it down). This script fits the hierarchical make model in
`src/fourth_and_rl/kicker.py` on the training seasons and reads the curve for a stated
kicker in stated conditions.

Three steps, separate so that nothing tuned can see a test-season number:

  .venv/bin/python scripts/questions/fg_selection_bias.py
      Development. Fits on 2010 to 2021 (training seasons only), chooses the knot set
      by AIC on those seasons, checks the model on 2022 to 2023 (calibration seasons),
      and writes results/questions/fg_selection_bias.json and
      assets/questions/fg_selection_bias.png. Saves data/models/kicker_fg.joblib.

  .venv/bin/python scripts/questions/fg_selection_bias.py --preregister
      Writes results/questions/fg_selection_bias_preregistration.json: the model
      fingerprint, the hypotheses and their thresholds. Refuses to overwrite. Reads no
      test row.

  .venv/bin/python scripts/questions/fg_selection_bias.py --evaluate
      The one check on 2024 to 2025, read through splits.load_test_plays (logged).
      Refuses without the preregistration, refuses if the saved model differs from it,
      and refuses a second run. Writes results/questions/fg_selection_bias_test.json.

The 2026 holdout is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from fourth_and_rl import components, splits, style
from fourth_and_rl import kicker as kk
from fourth_and_rl.io import read_json, write_json
from fourth_and_rl.paths import ASSETS, RESULTS

OUT = RESULTS / "questions" / "fg_selection_bias.json"
PREREG = RESULTS / "questions" / "fg_selection_bias_preregistration.json"
TEST_OUT = RESULTS / "questions" / "fg_selection_bias_test.json"
EVAL_LOG = RESULTS / "questions" / "fg_selection_bias_evaluations.jsonl"
FIGURE = ASSETS / "questions" / "fg_selection_bias.png"
TEST_PURPOSE = (
    "fg selection bias: the single preregistered check of the kicker field-goal model on "
    "2024-2025 (scripts/questions/fg_selection_bias.py --evaluate)"
)
POST_HOC = (
    "Post hoc: this model, the support distance and the cap or override it offers the "
    "planner were designed after seeing the world-model run on branch "
    "claude/world-model-colab (results/world_model/colab_t4_v2), whose simulated "
    "long-range make rate was above the raw training rate. They are not part of that "
    "run's preregistered plan."
)


# ------------------------------------------------------------------------------ helpers


def wilson(made: float, n: float, level: float = 0.9) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    zq = stats.norm.ppf(0.5 + level / 2)
    p = made / n
    den = 1 + zq * zq / n
    mid = (p + zq * zq / (2 * n)) / den
    half = zq * np.sqrt(p * (1 - p) / n + zq * zq / (4 * n * n)) / den
    return (float(mid - half), float(mid + half))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def attempts_for(plays: pl.DataFrame, seasons: list[int], cfg: dict) -> pl.DataFrame:
    return kk.field_goal_attempts(plays, kk.load_kicker_ids(seasons), cfg)


def games_frame(plays: pl.DataFrame) -> pl.DataFrame:
    """One row per game with its roof and weather (for the planner's weather average)."""
    return plays.group_by("game_id").agg(pl.col("roof").first(), pl.col("temp").first(),
                                         pl.col("wind").first())  # fmt: skip


def fit_model(train_att: pl.DataFrame, train_plays: pl.DataFrame, cfg: dict
              ) -> tuple[kk.FGMakeModel, list[dict]]:  # fmt: skip
    knots, candidates = kk.select_knots(train_att, cfg)
    c = kk.copy_cfg(cfg)
    c["distance"]["knots"] = knots
    model = kk.FGMakeModel(c).fit(train_att)
    model.set_condition_mix(games_frame(train_plays))
    sup = kk.support_distance(model, train_att)
    model.support_distance = sup["support_distance"]
    model.support = sup
    return model, candidates


def model_fingerprint(model: kk.FGMakeModel) -> str:
    blob = json.dumps({
        "beta": np.round(model.beta, 10).tolist(),
        "b": np.round(model.b, 10).tolist(),
        "sigma": np.round(model.sigma, 12).tolist(),
        "kickers": list(model.kicker_ids),
        "config": model.cfg,
    }, sort_keys=True)  # fmt: skip
    return hashlib.sha256(blob.encode()).hexdigest()


def m1_model(train_plays: pl.DataFrame) -> components.FieldGoalModel:
    """The M1 field-goal model (distance and season trend), refitted on 2010 to 2021."""
    cfg = components.load_config()
    return components.FieldGoalModel(**cfg.get("field_goal", {})).fit(train_plays)


def m1_predict(m1: components.FieldGoalModel, att: pl.DataFrame) -> np.ndarray:
    yl = (att["distance_yl18"].to_numpy() - 18).astype(int)
    seasons = att["season"].to_numpy()
    out = np.empty(att.height)
    for s in np.unique(seasons):
        mask = seasons == s
        out[mask] = m1.p_make(yl[mask], season=int(s))
    return out


def band_table(att: pl.DataFrame, preds: dict[str, np.ndarray], bands: list[list[int]]
               ) -> list[dict]:  # fmt: skip
    d = att["distance"].to_numpy()
    y = att["made"].to_numpy().astype(float)
    rows = []
    for lo, hi in bands:
        m = (d >= lo) & (d <= hi)
        row: dict[str, Any] = {"band": [lo, hi], "attempts": int(m.sum()),
                               "observed": float(y[m].mean()) if m.any() else None}  # fmt: skip
        for k, p in preds.items():
            row[k] = float(p[m].mean()) if m.any() else None
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- development


def development(cfg: dict) -> None:
    dev = splits.load_dev_plays()
    train_s, cal_s = list(splits.train_seasons()), list(splits.calibration_seasons())
    train_plays = dev.filter(pl.col("season").is_in(train_s))
    cal_plays = dev.filter(pl.col("season").is_in(cal_s))
    train_att = attempts_for(train_plays, train_s, cfg)
    cal_att = attempts_for(cal_plays, cal_s, cfg)
    no_id = int(train_att["kicker_id"].null_count())
    train_att = train_att.filter(pl.col("kicker_id").is_not_null())

    model, candidates = fit_model(train_att, train_plays, cfg)
    model.save()
    m1 = m1_model(train_plays)
    rng_seed = int(cfg["seed"])
    beta_d, b_d = model.draws(int(cfg["curves"]["posterior_draws"]), rng_seed)

    # ---- curves
    lo_d, hi_d = cfg["curves"]["distances"]
    grid = np.arange(lo_d, hi_d + 1)
    nc = model.neutral_conditions()
    avg = model.curve_interval(grid)
    curves: dict[str, Any] = {
        "distance": grid.tolist(),
        "average_kicker_neutral": {k: avg[k].tolist() for k in ("p", "lo", "hi")},
        "average_kicker_weather_mix": model.p_make(grid).tolist(),
        "average_kicker_dome": model.p_make(grid, roof="dome").tolist(),
        "m1_distance_and_season_model": m1.p_make(np.clip(grid - 18, 1, 99),
                                                  season=model.reference_season).tolist(),
    }  # fmt: skip
    for q in cfg["curves"]["strength_quantiles"]:
        ci = model.curve_interval(grid, strength=q)
        curves[f"strength_quantile_{q}"] = {k: ci[k].tolist() for k in ("p", "lo", "hi")}

    d_all = train_att["distance"].to_numpy()
    y_all = train_att["made"].to_numpy().astype(float)
    raw_rows = []
    for d in grid:
        m = d_all == d
        n, k = int(m.sum()), float(y_all[m].sum())
        lo, hi = wilson(k, n)
        raw_rows.append({"distance": int(d), "attempts": n, "made": int(k),
                         "rate": k / n if n else None, "lo": lo if n else None,
                         "hi": hi if n else None})  # fmt: skip
    curves["raw"] = raw_rows

    # named kickers: highest and lowest 50-yard strength among kickers with enough
    # attempts, the kicker with the most 55+ attempts, and the median-ranked kicker.
    ranked = model.ranked_kickers(cfg["kickers"]["named_min_attempts"])
    long_counts = (train_att.filter(pl.col("distance") >= 55).group_by("kicker_id").len()
                   .sort("len", descending=True))  # fmt: skip
    long_counts = long_counts.filter(pl.col("kicker_id").is_in(ranked["kicker_id"].to_list()))
    picks = [("strongest", ranked["kicker_id"][0]),
             ("most_55_plus_attempts", long_counts["kicker_id"][0]),
             ("median", ranked["kicker_id"][ranked.height // 2]),
             ("weakest", ranked["kicker_id"][-1])]  # fmt: skip
    named = []
    seen: set[str] = set()
    for role, kid in picks:
        if kid in seen:
            continue
        seen.add(kid)
        ci = model.curve_interval(grid, kicker_id=kid)
        r = ranked.filter(pl.col("kicker_id") == kid).row(0, named=True)
        n55 = long_counts.filter(pl.col("kicker_id") == kid)["len"]
        named.append({"role": role, "kicker_id": kid, "kicker_name": r["kicker_name"],
                      "attempts": r["attempts"], "attempts_55_plus": int(n55[0]) if len(n55)
                      else 0, "shift_50_log_odds": r["shift_50"],
                      "p": ci["p"].tolist(), "lo": ci["lo"].tolist(),
                      "hi": ci["hi"].tolist()})  # fmt: skip
    curves["named_kickers"] = named

    # ---- table at the report distances (raw within +-2 yards)
    at = []
    for d in cfg["curves"]["report_at"]:
        i = int(d - lo_d)
        win = np.abs(d_all - d) <= 2
        same = kk.selection_decomposition(model, train_att.filter(pl.Series(win)))
        at.append({
            "distance": d,
            "raw_within_2_yards": float(y_all[win].mean()) if win.any() else None,
            "raw_attempts_within_2_yards": int(win.sum()),
            "model_fit_same_attempts": same["model_fit"],
            "selection_free_same_attempts": same["selection_free"],
            "average_kicker_neutral": avg["p"][i],
            "average_kicker_neutral_90": [avg["lo"][i], avg["hi"][i]],
            "average_kicker_weather_mix": curves["average_kicker_weather_mix"][i],
            "top_decile_kicker_neutral": curves["strength_quantile_0.9"]["p"][i],
            "bottom_decile_kicker_neutral": curves["strength_quantile_0.1"]["p"][i],
            "m1_distance_and_season_model": curves["m1_distance_and_season_model"][i],
        })  # fmt: skip

    # ---- selection-bias decomposition on training attempts
    decomp = []
    for lo, hi in cfg["decomposition"]["bands"]:
        a = train_att.filter(pl.col("distance").is_between(lo, hi))
        r = kk.selection_decomposition(model, a, beta_d, b_d)
        r["band"] = [lo, hi]
        r["heave_share"] = float(a["heave"].mean())
        r["outdoor_share"] = float((a["outdoor"] == 1).mean())
        r["mean_season"] = float(a["season"].mean())
        r["mean_kicker_shift_50"] = float(np.mean([model.kicker_shift(k)
                                                   for k in a["kicker_id"].to_list()]))
        decomp.append(r)
    all_shift = float(np.mean([model.kicker_shift(k) for k in train_att["kicker_id"].to_list()]))

    wm = []
    for lo, hi in cfg["decomposition"]["world_model_bands"]:
        a = train_att.filter(pl.col("distance_yl18").is_between(lo, hi))
        r = kk.selection_decomposition(model, a, beta_d, b_d)
        dd = np.arange(lo, hi + 1)
        wm.append({
            "band_yardline_plus_18": [lo, hi], "attempts": r["attempts"], "raw": r["raw"],
            "model_fit": r["model_fit"], "selection_free": r["selection_free"],
            "contributions": r["contributions"],
            "planner_average_kicker_weather_mix_2021": float(np.mean(model.p_make(dd))),
            "planner_top_decile_weather_mix_2021": float(np.mean(model.p_make(dd, 0.9))),
        })  # fmt: skip

    # ---- likelihood check of the kicker effects
    plain = _plain_loglik(model, train_att)
    lr = 2 * (model.log_marginal - plain)

    # ---- calibration seasons (2022 to 2023): checks only
    cal_att = cal_att.filter(pl.col("kicker_id").is_not_null())
    known = model.known(cal_att)
    y_cal = cal_att["made"].to_numpy().astype(float)
    p_own = model.predict_attempts(cal_att, "own")
    p_avg = model.predict_attempts(cal_att, "average")
    p_m1 = m1_predict(m1, cal_att)
    cal = {
        "seasons": cal_s,
        "attempts": int(cal_att.height),
        "attempts_by_kickers_seen_in_training": int(known.sum()),
        "outdoor_attempts_without_weather": int(((cal_att["outdoor"] == 1)
                                                 & cal_att["temp"].is_null()).sum()),
        "note": "Season trend extrapolated from 2021. Weather missing outdoors is filled "
        "with the training median. Kickers not seen in training get the average kicker.",
        "metrics": {
            name: {"log_loss": log_loss(y_cal, p), "brier": brier(y_cal, p)}
            for name, p in (("hierarchical_own_kicker", p_own),
                            ("hierarchical_average_kicker", p_avg),
                            ("m1_distance_and_season", p_m1))
        },
        "metrics_known_kickers_only": {
            name: {"log_loss": log_loss(y_cal[known], p[known]),
                   "brier": brier(y_cal[known], p[known])}
            for name, p in (("hierarchical_own_kicker", p_own),
                            ("hierarchical_average_kicker", p_avg),
                            ("m1_distance_and_season", p_m1))
        },
        "by_band": band_table(cal_att, {"hierarchical_own_kicker": p_own,
                                        "hierarchical_average_kicker": p_avg,
                                        "m1_distance_and_season": p_m1},
                              [[18, 29], [30, 39], [40, 49], [50, 54], [55, 59], [60, 75],
                               [50, 75]]),
    }  # fmt: skip
    cand_cal = []
    for c in candidates:
        cc = kk.copy_cfg(cfg)
        cc["distance"]["knots"] = c["knots"]
        mc = kk.FGMakeModel(cc).fit(train_att)
        cand_cal.append({"knots": c["knots"],
                         "calibration_log_loss": log_loss(y_cal, mc.predict_attempts(cal_att))})
    cal["knot_candidates_log_loss_reported_not_used"] = cand_cal

    # ---- the same model without the usage (Mundlak) term, on the real training data
    plain_cfg = kk.copy_cfg(model.cfg)
    plain_cfg["terms"]["kicker_usage"] = False
    plain_model = kk.FGMakeModel(plain_cfg).fit(train_att)
    a55 = train_att.filter(pl.col("distance") >= 55)
    without_usage = {
        "average_kicker_neutral_at": {str(d): float(plain_model.curve_interval(
            np.array([d]))["p"][0]) for d in cfg["curves"]["report_at"]},
        "selection_free_55_plus": kk.selection_decomposition(plain_model, a55)["selection_free"],
        "kicker_sd_intercept_log_odds": float(np.sqrt(plain_model.sigma[0, 0])),
    }  # fmt: skip

    # ---- synthetic world with selection built in (the recovery test, for the record)
    synthetic = [kk.synthetic_recovery(seed, cfg) for seed in range(5)]

    # ---- monotone check and planner examples
    grid75 = np.arange(18, 76)
    raw_curve = 1.0 / (1.0 + np.exp(-model.logit(grid75)))
    pm = model.p_make(grid75, roof="outdoors")
    monotone = bool(np.all(np.diff(raw_curve) <= 1e-12))
    monotone_fix = float(np.max(np.abs(pm - raw_curve)))
    ex = []
    for d in (45, 50, 55, 60, 65):
        ex.append({"distance": d, "average_weather_mix": model.p_make(d),
                   "top_decile_weather_mix": model.p_make(d, 0.9),
                   "average_dome": model.p_make(d, roof="dome"),
                   "average_outdoors_20mph_wind": model.p_make(d, roof="outdoors", wind=20),
                   "average_heave": model.p_make(d, heave=True),
                   "average_season_2025_extrapolated": model.p_make(d, season=2025),
                   "in_support": bool(model.in_support(d))})  # fmt: skip

    sd = np.sqrt(np.diag(model.sigma))
    # A slope SD under 0.01 log-odds per 10 yards moves a 60-yard kick by under 0.2
    # points: effectively zero, and the correlation is then not identified.
    at_bound = bool(model.slope and sd[1] < 0.01)
    rho = float(model.sigma[0, 1] / (sd[0] * sd[1])) if model.slope and not at_bound else None
    out = {
        "question": "How much of the raw long-range field-goal make rate is who kicks and "
        "in what conditions, and what does an average kicker make?",
        "post_hoc": POST_HOC,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": model.cfg,
        "config_sha256_16": kk.config_hash(model.cfg),
        "model_fingerprint": model_fingerprint(model),
        "data": {
            "training_seasons": train_s,
            "attempts": int(train_att.height),
            "attempts_without_kicker_id_dropped": no_id,
            "kickers": int(len(model.kicker_ids)),
            "kickers_ranked": int(model.ranked_kickers().height),
            "heaves": int(train_att["heave"].sum()),
            "outdoor_attempts_without_weather": int(((train_att["outdoor"] == 1)
                                                     & train_att["temp"].is_null()).sum()),
            "open_roof_attempts": int((train_att["outdoor"] == 0.5).sum()),
            "distance_source": "kick_distance (yardline_100 + 18 when missing)",
            "kick_distance_equals_yardline_plus_18_share": float(
                (train_att["distance"] == train_att["distance_yl18"]).mean()),
        },  # fmt: skip
        "knot_selection": {"rule": "lowest AIC on training seasons",
                           "candidates": candidates, "chosen": model.cfg["distance"]["knots"]},
        "model": {
            "converged": model.converged,
            "fixed_effects": [{"name": n, "estimate": float(b), "se": float(s)}
                              for n, b, s in zip(model.feature_names, model.beta,
                                                 np.sqrt(np.diag(model.cov_beta)),
                                                 strict=True)],
            "kicker_sd_intercept_log_odds": float(sd[0]),
            "kicker_sd_slope_log_odds_per_10_yards": float(sd[1]) if model.slope else None,
            "kicker_slope_sd_effectively_zero_below_0_01": at_bound,
            "kicker_intercept_slope_correlation": rho,
            "strength_sd_at_50_yards": model.strength_sd(),
            "usage_coefficient_per_10_yards": model.usage_coef,
            "usage_league_mean_distance": model.usage_league,
            "usage_sd_across_established_kickers_yards": float(10 * np.sqrt(model.usage_var10)),
            "log_marginal_likelihood": float(model.log_marginal),
            "log_likelihood_without_kicker_effects": plain,
            "likelihood_ratio_kicker_effects": float(lr),
            "likelihood_ratio_p_value_boundary_mixture": float(
                0.5 * stats.chi2.sf(lr, 1) + 0.5 * stats.chi2.sf(lr, 2)) if model.slope
            else float(0.5 * stats.chi2.sf(lr, 1)),
            "reference_season": model.reference_season,
            "mean_training_season": model.mean_season,
            "neutral_conditions": {"outdoor": True, "temp_f": nc.temp, "wind_mph": nc.wind,
                                   "heave": False},
            "average_kicker_curve_monotone_18_to_75_before_fix": monotone,
            "monotone_fix_max_change_18_to_75": monotone_fix,
        },  # fmt: skip
        "curves": curves,
        "at_distances": at,
        "decomposition": {
            "method": "Shapley split of model_fit - selection_free over four factors "
            "(kicker, conditions, heave, era); 90% intervals from 1,000 draws of the "
            "Gaussian approximation at the mode (Sigma held fixed).",
            "mean_kicker_shift_50_all_attempts": all_shift,
            "bands": decomp,
            "world_model_bands": wm,
        },
        "support": {
            "rule": (
                "The support distance is the largest integer distance D such that every "
                f"distance d from {cfg['support']['start_distance']} to D satisfies both: "
                f"(a) top-decile kickers (top {100 - 100 * cfg['kickers']['top_quantile']:.0f}% "
                f"by estimated 50-yard strength among kickers with at least "
                f"{cfg['kickers']['min_attempts_ranked']} training attempts) made at least "
                f"{cfg['support']['min_attempts']} training attempts with distance in "
                f"[d - {cfg['support']['half_window']}, d + {cfg['support']['half_window']}]; "
                f"and (b) the median over those kickers of the width of the "
                f"{100 * cfg['support']['interval_level']:.0f}% interval of their own "
                f"predicted make rate at d (neutral conditions, 2021) is at most "
                f"{cfg['support']['max_interval_width']}."
            ),
            **model.support,
        },
        "calibration_check": cal,
        "without_usage_term": without_usage,
        "synthetic_selection_world": {
            "description": "80 kickers, 150 attempts each, strength u ~ Normal(0, 0.5) in "
            "log-odds, attempt distance Normal(38 + 12 u, 9): strong kickers kick longer. "
            "Rates on attempts of 55 yards or more (kicker.simulate_selected_attempts).",
            "seeds": synthetic,
        },
        "planner_examples": ex,
    }
    write_json(out, OUT)
    print(f"wrote {OUT}")
    figure(out)
    print(f"wrote {FIGURE}")


def _plain_loglik(model: kk.FGMakeModel, att: pl.DataFrame) -> float:
    """Log likelihood of the same fixed effects with no kicker effects (plain logistic)."""
    from scipy import optimize

    x = model.design.matrix(**model._arrays(att))
    y = att["made"].to_numpy().astype(float)
    ridge = np.full(x.shape[1], model.cfg["fit"]["ridge"])
    ridge[0] = 0.0

    def f(bt: np.ndarray) -> float:
        eta = x @ bt
        return -float(np.sum(y * eta - np.logaddexp(0.0, eta))) + 0.5 * float(bt @ (ridge * bt))

    def g(bt: np.ndarray) -> np.ndarray:
        return -(x.T @ (y - 1 / (1 + np.exp(-(x @ bt))))) + ridge * bt

    res = optimize.minimize(f, model.beta.copy(), jac=g, method="L-BFGS-B",
                            options={"maxiter": 2000, "gtol": 1e-8})  # fmt: skip
    eta = x @ res.x
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


# -------------------------------------------------------------------------------- figure


def figure(out: dict) -> None:
    style.apply()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.6),
                                 gridspec_kw={"width_ratios": [1.6, 1]})  # fmt: skip
    c = out["curves"]
    d = np.array(c["distance"])
    raw = c["raw"]
    rd = np.array([r["distance"] for r in raw if r["attempts"] >= 5])
    rr = np.array([r["rate"] for r in raw if r["attempts"] >= 5])
    rn = np.array([r["attempts"] for r in raw if r["attempts"] >= 5])
    ax.scatter(rd, rr, s=8 + 60 * np.sqrt(rn / rn.max()), color=style.MUTED, alpha=0.55,
               linewidths=0, label="Raw rate (attempts of 5+ per yard; size = attempts)",
               zorder=2)  # fmt: skip
    avg = c["average_kicker_neutral"]
    ax.fill_between(d, avg["lo"], avg["hi"], color=style.BLUE, alpha=0.15, linewidth=0)
    ax.plot(d, avg["p"], color=style.BLUE, label="Average kicker, neutral (90% band)", zorder=4)
    top, bot = c["strength_quantile_0.9"], c["strength_quantile_0.1"]
    ax.plot(d, top["p"], color=style.ORANGE, lw=1.6, label="Top-decile kicker", zorder=3)
    ax.plot(d, bot["p"], color=style.VIOLET, lw=1.6, label="Bottom-decile kicker", zorder=3)
    ax.plot(d, c["m1_distance_and_season_model"], color=style.INK_2, lw=1.2, ls="--",
            label="M1 model (distance and season only)", zorder=3)  # fmt: skip
    for nk in c["named_kickers"][:1]:
        ax.plot(d, nk["p"], color=style.AQUA, lw=1.2, ls=":",
                label=f"{nk['kicker_name']} (strongest, shrunken)", zorder=3)  # fmt: skip
    sup = out["support"]["support_distance"]
    ax.axvspan(sup + 0.5, d.max(), color=style.GRID, alpha=0.5, zorder=0, linewidth=0)
    ax.text(sup + 1.2, 0.93, f"Beyond the support\ndistance ({sup} yd)", color=style.INK_2,
            fontsize=9, va="top")  # fmt: skip
    ax.set_xlim(d.min(), d.max())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Kick distance (yards)")
    ax.set_ylabel("P(make)")
    ax.legend(loc="lower left", fontsize=8.6)
    style.title(ax, "Field-goal make rate by distance, 2010 to 2021",
                "Hierarchical logistic model with kicker effects; neutral = outdoors, median "
                "weather, not a heave, 2021")  # fmt: skip

    band = next(b for b in out["decomposition"]["bands"]
                if b["band"] == out["config"]["decomposition"]["headline_band"])  # fmt: skip
    labels = ["Raw rate", "Fit error\n(raw - model)", "Who kicks", "Roof and weather",
              "Heaves", "Era",
              "Average kicker,\nneutral"]
    contrib = band["contributions"]
    steps = [band["raw"], -band["fit_error"], -contrib["kicker"], -contrib["conditions"],
             -contrib["heave"], -contrib["era"]]  # fmt: skip
    level = band["raw"]
    ys = np.arange(len(labels))[::-1]
    bx.barh(ys[0], level, color=style.MUTED, height=0.6)
    bx.text(level + 0.004, ys[0], f"{level:.3f}", va="center", fontsize=9, color=style.INK)
    for i, s in enumerate(steps[1:], start=1):
        start = level
        level = level + s
        color = style.ORANGE if s < 0 else style.AQUA
        bx.barh(ys[i], s, left=start, color=color, height=0.6)
        # Label with the contribution to the raw rate (the step's opposite sign).
        bx.text(max(start, level) + 0.004, ys[i], f"{-s:+.3f}", va="center", fontsize=9,
                color=style.INK_2)  # fmt: skip
    bx.barh(ys[-1], band["selection_free"], color=style.BLUE, height=0.6)
    bx.text(band["selection_free"] + 0.004, ys[-1], f"{band['selection_free']:.3f}",
            va="center", fontsize=9, color=style.INK)  # fmt: skip
    bx.set_yticks(ys, labels)
    lo = min(band["selection_free"], band["raw"]) - 0.12
    bx.set_xlim(max(0, lo), max(band["selection_free"], band["raw"]) + 0.08)
    bx.set_xlabel("Mean P(make) on the same attempts")
    bx.grid(axis="y", visible=False)
    b0, b1 = band["band"]
    style.title(bx, f"Where the raw {b0}+ yard rate comes from",
                f"{band['attempts']} training attempts; numbers: how much each source "
                "raised (+) or lowered (-) it")  # fmt: skip
    fig.tight_layout()
    style.save(fig, FIGURE)


# ------------------------------------------------------------------------ preregistration


def preregister(cfg: dict) -> None:
    if PREREG.exists():
        raise SystemExit(f"{PREREG} already exists; a preregistration is written once")
    if not OUT.exists():
        raise SystemExit("run the development step first")
    dev = read_json(OUT)
    model = kk.FGMakeModel.load()
    fp = model_fingerprint(model)
    if dev["model_fingerprint"] != fp:
        raise SystemExit("the saved model is not the one in the development results")
    out = {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "statement": [
            "Written before any 2024 to 2025 field-goal number was computed for this model.",
            "The model was fitted on 2010 to 2021 only; the knot set was chosen by AIC on "
            "those seasons; 2022 to 2023 were used for checks only.",
            "The test check runs once (--evaluate). Whatever it shows is reported, and the "
            "model does not change afterwards.",
            POST_HOC,
            "The 2026 holdout has not been loaded.",
        ],
        "model_fingerprint": fp,
        "config_sha256_16": dev["config_sha256_16"],
        "model_stamp": model.stamp,
        "support_distance": model.support_distance,
        "scoring": {
            "kickers_not_seen_in_training": "average kicker (effects 0)",
            "season": "each attempt's own season (the linear trend extrapolates from 2021)",
            "weather_missing_outdoors": "training median",
            "comparison_model": "M1 FieldGoalModel (distance and season trend), refitted on "
            "2010 to 2021",
        },
        "hypotheses": [
            {"id": "H1", "statement": "The hierarchical model (own kicker effects) has lower "
             "log loss than the M1 distance-and-season model on all 2024 to 2025 attempts.",
             "pass_if": "log_loss(hierarchical_own_kicker) < log_loss(m1_distance_and_season)"},
            {"id": "H2", "statement": "Selection by kicker persists: among attempts by "
             "kickers seen in training, the mean training-estimated 50-yard strength of "
             "kickers on 55+ yard attempts exceeds that on all attempts.",
             "pass_if": "mean_shift_55_plus - mean_shift_all > 0"},
            {"id": "H3", "statement": "The hierarchical model is calibrated within 5 points "
             "on 50+ yard attempts.",
             "pass_if": "abs(observed - predicted_own_kicker) <= 0.05 on 50+ yard attempts",
             "expectation": "May fail toward under-prediction: the season trend is a "
             "straight line fitted through 2021, and long-range kicking may have improved "
             "faster since."},
        ],  # fmt: skip
        "also_reported": [
            "log loss and Brier score for the hierarchical model with average-kicker effects",
            "observed against predicted by distance band",
            "the selection decomposition on 2024 to 2025 55+ yard attempts",
        ],
    }
    write_json(out, PREREG)
    print(f"wrote {PREREG}")


# ------------------------------------------------------------------------------ evaluate


def evaluate(cfg: dict) -> None:
    if not PREREG.exists():
        raise SystemExit("no preregistration: run --preregister before --evaluate")
    if TEST_OUT.exists() or EVAL_LOG.exists():
        raise SystemExit("the test seasons were already scored for this model; refusing")
    prereg = read_json(PREREG)
    model = kk.FGMakeModel.load()
    if model_fingerprint(model) != prereg["model_fingerprint"]:
        raise SystemExit("the saved model is not the preregistered one; refusing")
    dev = splits.load_dev_plays()
    train_plays = dev.filter(pl.col("season").is_in(list(splits.train_seasons())))
    m1 = m1_model(train_plays)
    EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG, "a") as f:
        f.write(json.dumps({"at": datetime.now(UTC).isoformat(timespec="seconds"),
                            "model_fingerprint": prereg["model_fingerprint"]}) + "\n")
    test = splits.load_test_plays(TEST_PURPOSE)
    splits.assert_can_score(model.stamp, test, "kicker field-goal model on test seasons")
    test_s = list(splits.test_seasons())
    att = attempts_for(test, test_s, cfg).filter(pl.col("kicker_id").is_not_null())
    y = att["made"].to_numpy().astype(float)
    p_own = model.predict_attempts(att, "own")
    p_avg = model.predict_attempts(att, "average")
    p_m1 = m1_predict(m1, att)
    known = model.known(att)
    shift = np.array([model.kicker_shift(k) if k in model._kicker_index else np.nan
                      for k in att["kicker_id"].to_list()])  # fmt: skip
    d = att["distance"].to_numpy()
    long = (d >= 55) & known
    h2 = float(np.nanmean(shift[long]) - np.nanmean(shift[known]))
    m50 = d >= 50
    h3_gap = float(y[m50].mean() - p_own[m50].mean())
    metrics = {name: {"log_loss": log_loss(y, p), "brier": brier(y, p)}
               for name, p in (("hierarchical_own_kicker", p_own),
                               ("hierarchical_average_kicker", p_avg),
                               ("m1_distance_and_season", p_m1))}  # fmt: skip
    beta_d, b_d = model.draws(int(cfg["curves"]["posterior_draws"]), int(cfg["seed"]))
    a55 = att.filter(pl.col("distance") >= 55)
    out = {
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "preregistration": "results/questions/fg_selection_bias_preregistration.json",
        "model_fingerprint": prereg["model_fingerprint"],
        "seasons": test_s,
        "attempts": int(att.height),
        "attempts_by_kickers_seen_in_training": int(known.sum()),
        "outdoor_attempts_without_weather": int(((att["outdoor"] == 1)
                                                 & att["temp"].is_null()).sum()),
        "metrics": metrics,
        "hypotheses": {
            "H1": {"hierarchical_log_loss": metrics["hierarchical_own_kicker"]["log_loss"],
                   "m1_log_loss": metrics["m1_distance_and_season"]["log_loss"],
                   "passed": metrics["hierarchical_own_kicker"]["log_loss"]
                   < metrics["m1_distance_and_season"]["log_loss"]},
            "H2": {"mean_shift_55_plus_minus_all": h2, "attempts_55_plus_known": int(long.sum()),
                   "passed": h2 > 0},
            "H3": {"observed_50_plus": float(y[m50].mean()),
                   "predicted_50_plus": float(p_own[m50].mean()),
                   "observed_minus_predicted": h3_gap, "attempts": int(m50.sum()),
                   "passed": abs(h3_gap) <= 0.05},
        },  # fmt: skip
        "by_band": band_table(att, {"hierarchical_own_kicker": p_own,
                                    "hierarchical_average_kicker": p_avg,
                                    "m1_distance_and_season": p_m1},
                              [[18, 29], [30, 39], [40, 49], [50, 54], [55, 59], [60, 75],
                               [50, 75]]),  # fmt: skip
        "decomposition_55_plus": kk.selection_decomposition(model, a55, beta_d, b_d),
    }
    write_json(out, TEST_OUT)
    print(f"wrote {TEST_OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--preregister", action="store_true")
    g.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()
    cfg = kk.load_config()
    if args.preregister:
        preregister(cfg)
    elif args.evaluate:
        evaluate(cfg)
    else:
        development(cfg)


if __name__ == "__main__":
    main()
