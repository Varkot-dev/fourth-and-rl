"""M9 "Was it really fourth and 1?": how often the hidden fraction of a yard flips the call.

Two stages, run in order:

  python scripts/m9_distance_error.py --stage dev
      Development seasons only (splits.load_dev_plays). Measures the evidence for the
      hidden offset, checks the estimators on simulated plays with a known truth, fits
      the continuous conversion curve under every error model, propagates it through the
      decision engine on 2010 to 2023 fourth downs, works the Lions case, and writes
      results/m9_distance_error.json with the evaluation plan for the test seasons.

  python scripts/m9_distance_error.py --stage figures
      Redraws the figures from the saved results without reading the test seasons.

  python scripts/m9_distance_error.py --stage test
      Recomputes the development fits, checks that they match the saved plan exactly,
      then reads 2024 to 2025 through splits.load_test_plays (logged) and scores the
      frozen propagation once. Refuses to run if test results already exist.

The 2026 season (the final holdout) is never loaded. Figures: assets/m9_distance_error.png
and assets/questions/q4_really_fourth_and_one.png. Notes: docs/notes/m9_distance_error.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("POLARS_MAX_THREADS", "2")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from scipy.special import expit, logit  # noqa: E402

from fourth_and_rl import decision, splits, style  # noqa: E402
from fourth_and_rl import distance_error as de  # noqa: E402
from fourth_and_rl import wp as wp_mod  # noqa: E402
from fourth_and_rl.io import read_json, write_json  # noqa: E402
from fourth_and_rl.paths import ASSETS, RESULTS  # noqa: E402

RESULT = RESULTS / "m9_distance_error.json"
FIGURE = ASSETS / "m9_distance_error.png"
FIGURE_Q = ASSETS / "questions" / "q4_really_fourth_and_one.png"


# ------------------------------------------------------------------------- helpers


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def game_index(df: pl.DataFrame, games: np.ndarray) -> np.ndarray:
    """Position of each row's game in `games` (sorted unique game ids)."""
    return np.searchsorted(games, df["game_id"].to_numpy())


def offset_function(go: Any, distance: int):
    """Logit offset by (yard line, down) from the whole-yard go model at one distance:
    field position and down come from the existing model; the distance shape is new."""

    def f(yl: np.ndarray, down: np.ndarray) -> np.ndarray:
        yl, down = np.asarray(yl, dtype=int), np.asarray(down, dtype=int)
        p3 = np.asarray(go.p_convert(yl, distance, down=3), dtype=float)
        p4 = np.asarray(go.p_convert(yl, distance, down=4), dtype=float)
        return logit(np.where(down == 3, p3, p4))

    return f


def percentile_interval(x: list[float] | np.ndarray) -> list[float]:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(a, 5)), float(np.percentile(a, 95))]


# ------------------------------------------------------------------------ evidence


def with_previous(plays: pl.DataFrame) -> pl.DataFrame:
    cols = ["yards_gained", "ydstogo", "yardline_100", "first_down", "penalty", "play_type",
            "posteam", "down", "desc", "goal_to_go"]  # fmt: skip
    return plays.sort("game_id", "order_sequence").with_columns(
        [pl.col(c).shift(1).over("game_id").alias(f"prev_{c}") for c in cols]
    )


def integer_consistency(plays: pl.DataFrame) -> dict[str, Any]:
    """Signals (b) and (d) of the brief: does the integer record contradict itself?"""
    sel = de.line_offset_plays(plays, (1, 4))
    k = sel["k"].to_numpy()
    fd = sel["first_down"].fill_null(0).to_numpy() == 1
    nxt = plays.sort("game_id", "order_sequence").with_columns(
        [pl.col(c).shift(-1).over("game_id").alias(f"next_{c}")
         for c in ("down", "ydstogo", "yardline_100", "posteam")]  # fmt: skip
    )
    same = nxt.filter(
        pl.col("play_type").is_in(["run", "pass"])
        & (pl.col("penalty").fill_null(0) == 0)
        & (pl.col("first_down").fill_null(0) == 0)
        & (pl.col("fumble_lost").fill_null(0) == 0)
        & (pl.col("goal_to_go").fill_null(0) == 0)
        & (pl.col("next_posteam") == pl.col("posteam"))
        & (pl.col("next_down") == pl.col("down") + 1)
    )
    line_now = same["yardline_100"] - same["ydstogo"]
    line_next = same["next_yardline_100"] - same["next_ydstogo"]
    spot_resid = same["yardline_100"] - same["next_yardline_100"] - same["yards_gained"]
    inches = plays.filter(pl.col("desc").fill_null("").str.contains(r"(?i)\binch(es)?\b"))
    return {
        "plays_gaining_exactly_the_distance": int((k == 0).sum()),
        "of_which_not_a_first_down": float(np.mean(~fd[k == 0])),
        "plays_one_yard_short": int((k == -1).sum()),
        "of_which_a_first_down": float(np.mean(fd[k == -1])),
        "consecutive_downs_checked": int(same.height),
        "share_line_to_gain_moves": float((line_now != line_next).mean()),
        "share_spot_change_differs_from_yards_gained": float((spot_resid != 0).mean()),
        "descriptions_mentioning_inches": int(inches.height),
        "reading": "The integer record is kept consistent by construction: a play gaining the "
        "recorded distance is a first down, the line to gain never moves within a series, "
        "and the spot moves by the recorded gain. So signals (b) and (d) carry no direct "
        "information about the fraction, and play-by-play has no 'inches' marker (ydstogo "
        "is 1 and the descriptions do not say). The information is in how often each "
        "recorded gain occurs (the dip at k = 0).",
    }


def offset_evidence(dev: pl.DataFrame, cfg: dict) -> dict[str, Any]:
    lo = cfg["line_offset"]
    downs, dist, win = tuple(lo["downs"]), tuple(lo["distances"]), int(lo["window"])
    pooled = de.pooled_line_offset(dev, dist, downs, win)
    sel = de.line_offset_plays(dev, downs)
    by_down = {}
    for down in range(downs[0], downs[1] + 1):
        k = sel.filter((pl.col("down") == down) & pl.col("ydstogo").is_between(*dist))["k"]
        by_down[str(down)] = de.fit_line_offset(*de.line_offset_counts(k.to_numpy(), win))
    k1 = sel.filter(pl.col("ydstogo") == 1)["k"].to_numpy()
    ks_pool, counts_pool = de.line_offset_counts(
        sel.filter(pl.col("ydstogo").is_between(*dist))["k"].to_numpy(), win
    )
    return {
        "definition": "k = yards gained minus recorded yards to go, on runs and completed "
        f"passes, downs {downs[0]} to {downs[1]}, no penalty, turnover or goal to go. "
        "mean_offset is E[v], the mean position of the line to gain inside its recorded "
        "yard (0.5 if positions are uniform inside a yard).",
        "pooled": pooled,
        "by_down": by_down,
        "distance_1_for_reference": de.fit_line_offset(*de.line_offset_counts(k1, win)),
        "pooled_counts": {"k": ks_pool.tolist(), "count": counts_pool.tolist()},
        "history": history_test(dev, win),
    }


def history_selection(dev: pl.DataFrame) -> pl.DataFrame:
    """First and 10 plays (clean spot) whose previous play was a clean run or completed
    pass by the same team that made the first down, with that play's k."""
    prev = with_previous(dev)
    sel = de.line_offset_plays(prev, (1, 1)).filter(
        (pl.col("ydstogo") == 10)
        & (pl.col("prev_first_down") == 1)
        & (pl.col("prev_penalty").fill_null(0) == 0)
        & pl.col("prev_play_type").is_in(["run", "pass"])
        & (pl.col("prev_posteam") == pl.col("posteam"))
        & (pl.col("prev_goal_to_go").fill_null(0) == 0)
        & ~pl.col("prev_desc").fill_null("").str.contains("(?i)incomplete")
    )
    return sel.with_columns((pl.col("prev_yards_gained") - pl.col("prev_ydstogo")).alias("kp"))


def history_test(dev: pl.DataFrame, window: int) -> dict[str, Any]:
    """If the fraction is a real position carried from play to play, a first down gained
    by a play that barely crossed the line (k = 0) starts its series with the ball late in
    its yard, so the NEXT line to gain sits late in its yard too (larger offset). If the
    record reset the fraction every play, the two groups would not differ."""
    sel = history_selection(dev)
    barely = de.fit_line_offset(*de.line_offset_counts(
        sel.filter(pl.col("kp") == 0)["k"].to_numpy(), window))  # fmt: skip
    clear = de.fit_line_offset(*de.line_offset_counts(
        sel.filter(pl.col("kp") >= 1)["k"].to_numpy(), window))  # fmt: skip
    diff = barely["mean_offset"] - clear["mean_offset"]
    se = float(np.hypot(barely["se"], clear["se"]))
    return {
        "after_barely_converting": barely,
        "after_clear_conversion": clear,
        "difference": float(diff),
        "difference_se": se,
        "z": float(diff / se),
        "predicted_difference_if_positions_carried_and_uniform": 0.25,
        "predicted_difference_if_fraction_resets": 0.0,
    }


# --------------------------------------------------------------- simulation checks


def smoothed_gains(dev: pl.DataFrame, rng: np.random.Generator, n: int, sd: float
                   ) -> np.ndarray:  # fmt: skip
    """True first-down gains for the simulation: recorded gains of first and 10 runs and
    completions, resampled and jittered so their density is smooth."""
    sel = de.line_offset_plays(dev, (1, 1)).filter(pl.col("ydstogo") == 10)
    g = sel["yards_gained"].to_numpy().astype(float)
    return rng.choice(g, n) + rng.normal(0.0, sd, n)


def pooled_ends(dev: pl.DataFrame, cfg: dict, rng: np.random.Generator, n: int, sd: float
                ) -> np.ndarray:  # fmt: skip
    """True end positions relative to the line for the simulation, shaped like the plays of
    the pooled estimate: a recorded gain, jittered, minus a true distance drawn around the
    recorded one (triangular, as under a uniform offset)."""
    lo = cfg["line_offset"]
    sel = de.line_offset_plays(dev, tuple(lo["downs"])).filter(
        pl.col("ydstogo").is_between(*lo["distances"]))
    idx = rng.integers(0, sel.height, n)
    g = sel["yards_gained"].to_numpy().astype(float)[idx] + rng.normal(0.0, sd, n)
    D = sel["ydstogo"].to_numpy().astype(float)[idx] + rng.uniform(0, 1, n) - rng.uniform(0, 1, n)
    return g - D


def offset_simulation(dev: pl.DataFrame, cfg: dict, rng: np.random.Generator) -> dict[str, Any]:
    """The offset estimator on simulated plays whose offsets are known, with end positions
    shaped like the real ones. Its error here is the bias that comes from the shape of the
    gain distribution, used to correct the real estimate."""
    sim = cfg["simulation"]
    n, win, sd = int(sim["n_line"]), int(cfg["line_offset"]["window"]), float(sim["gain_jitter_sd"])
    ends = pooled_ends(dev, cfg, rng, n, sd)
    first = smoothed_gains(dev, rng, n, sd) - 10.0
    out: dict[str, Any] = {"pooled_shape": {}, "first_and_10_shape": {}}
    for label, e in (("pooled_shape", ends), ("first_and_10_shape", first)):
        cases = {
            "uniform_true_mean_0.50": rng.uniform(0, 1, n),
            "point_true_mean_0.50": np.full(n, 0.5),
            "beta_true_mean_0.40": rng.beta(0.8, 1.2, n),
            "carried_after_barely_converting_true_mean_0.75":
                rng.uniform(rng.uniform(0, 1, n), 1.0),
        }  # fmt: skip
        for name, v in cases.items():
            k = de.record_end_spot(e, v)
            out[label][name] = de.fit_line_offset(*de.line_offset_counts(k, win))
    bias = out["pooled_shape"]["uniform_true_mean_0.50"]["mean_offset"] - 0.5
    f10 = out["first_and_10_shape"]
    out["bias_pooled"] = float(bias)
    out["predicted_history_difference_carried_uniform"] = float(
        f10["carried_after_barely_converting_true_mean_0.75"]["mean_offset"]
        - f10["uniform_true_mean_0.50"]["mean_offset"])
    out["reading"] = ("True offsets known; end positions shaped like the real plays. The "
                      "estimate minus the truth is the estimator's bias for that shape.")
    return out


def resolve_penalty(att: pl.DataFrame, go: Any, cfg: dict, model: de.ErrorModel
                    ) -> tuple[dict, dict[str, Any]]:  # fmt: skip
    """Season-blocked cross-validation of the smoothness penalty (headline error model).
    Returns a copy of the config with numeric penalties, and the validation table."""
    import copy

    cc = cfg["curve"]
    rcfg = copy.deepcopy(cfg)
    if cc["penalty"] != "cv":
        rcfg["curve"]["penalty_sensitivity"] = [float(cc["penalty"]) * f
                                                for f in cc["penalty_sensitivity_factors"]]
        return rcfg, {"chosen": float(cc["penalty"]), "by": "config"}
    knots = np.asarray(cc["knots"], dtype=float)
    off = offset_function(go, int(cc["offset_distance"]))
    tab = de.kernel_table(model, int(cfg["kernel"]["max_ydstogo"]), float(cfg["kernel"]["step"]))
    seasons = att["season"].to_numpy()
    table = {}
    for lam in cc["penalty_grid"]:
        losses = []
        for fold in cc["cv_folds"]:
            held = np.isin(seasons, fold)
            fit = de.aggregate_cells(att.filter(pl.Series(~held)), off, tab)
            val = de.aggregate_cells(att.filter(pl.Series(held)), off, tab)
            curve = de.fit_latent_curve(fit, tab, knots, float(lam))
            P = np.clip(de.implied_whole_yard(curve, tab, val.cell, val.offset), 1e-9, 1 - 1e-9)
            losses.append(float(-(val.y * np.log(P) + (val.n - val.y) * np.log1p(-P)).sum()
                                / val.n.sum()))
        table[str(lam)] = {"mean_log_loss": float(np.mean(losses)), "folds": losses}
    chosen = float(min(cc["penalty_grid"], key=lambda x: table[str(x)]["mean_log_loss"]))
    rcfg["curve"]["penalty"] = chosen
    rcfg["curve"]["penalty_sensitivity"] = [chosen * f for f in cc["penalty_sensitivity_factors"]]
    return rcfg, {"chosen": chosen, "by": "season-blocked cross-validation, minimum log loss",
                  "model": model.name, "table": table}  # fmt: skip


def deconvolution_simulation(cfg: dict, rng: np.random.Generator) -> dict[str, Any]:
    """Deconvolution on simulated attempts: a known curve with a short-yardage cliff."""
    sim = cfg["simulation"]

    def truth(D: np.ndarray) -> np.ndarray:
        return expit(1.3 - 0.55 * np.log1p(D) - 0.9 * (1 - np.exp(-D / 0.7)))

    knots = np.asarray(cfg["curve"]["knots"], dtype=float)
    models = {m.name: m for m in de.error_models(cfg, fitted_mean=0.5)}
    grid = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    rows = {}
    for true_name in ("uniform", "narrow"):
        s = de.simulate_recorded(int(sim["n_curve"]), models[true_name], rng, curve=truth)
        att = pl.DataFrame({"ydstogo": s["recorded"], "gtg": np.zeros(s["recorded"].size, int),
                            "yardline_100": np.full(s["recorded"].size, 50),
                            "down": np.full(s["recorded"].size, 4),
                            "converted": s["converted"]})  # fmt: skip
        for fit_name in ("uniform", "narrow"):
            tab = de.kernel_table(models[fit_name], 20, float(cfg["kernel"]["step"]))
            cells = de.aggregate_cells(att, lambda yl, dn: np.zeros(np.size(yl)), tab)
            curve = de.fit_latent_curve(cells, tab, knots, float(cfg["curve"]["penalty"]))
            err = curve.p(grid) - truth(grid)
            rows[f"truth_{true_name}_fitted_with_{fit_name}"] = {
                "true_q": truth(grid).round(4).tolist(),
                "fitted_q": curve.p(grid).round(4).tolist(),
                "max_abs_error": float(np.abs(err).max()),
                "max_abs_error_from_0.5_yards": float(np.abs(err[grid >= 0.5]).max()),
                "max_abs_error_from_1.5_yards": float(np.abs(err[grid >= 1.5]).max()),
            }
    return {"distances": grid.tolist(), "cases": rows, "n_attempts": int(sim["n_curve"]),
            "penalty": float(cfg["curve"]["penalty"]),
            "reading": "Only the average of the curve over each recorded yard is identified; "
            "the shape inside a yard comes from the smoothness penalty. Errors are largest "
            "below one yard, where a sharp cliff is flattened."}  # fmt: skip


# ----------------------------------------------------------------------- the curves


def fit_all_curves(att: pl.DataFrame, go: Any, cfg: dict, models: list[de.ErrorModel],
                   weight: np.ndarray | None = None, only: tuple[str, float] | None = None
                   ) -> dict[tuple[str, float], tuple[de.KernelTable, de.LatentCurve]]:  # fmt: skip
    cc = cfg["curve"]
    knots = np.asarray(cc["knots"], dtype=float)
    off = offset_function(go, int(cc["offset_distance"]))
    lams = [float(cc["penalty"]), *[float(x) for x in cc["penalty_sensitivity"]]]
    out = {}
    for m in models:
        tab = de.kernel_table(m, int(cfg["kernel"]["max_ydstogo"]), float(cfg["kernel"]["step"]))
        cells = None
        for lam in lams:
            if only is not None and (m.name, lam) != only:
                continue
            if cells is None:
                cells = de.aggregate_cells(att, off, tab, weight)
            out[(m.name, lam)] = (tab, de.fit_latent_curve(cells, tab, knots, lam))
    return out


def whole_yard_comparison(att: pl.DataFrame, go: Any, cfg: dict,
                          curves: dict) -> dict[str, Any]:  # fmt: skip
    """The continuous curve against the whole-yard go model and the observed rates."""
    head = (cfg["headline_model"], float(cfg["curve"]["penalty"]))
    tab, curve = curves[head]
    off = offset_function(go, int(cfg["curve"]["offset_distance"]))
    cells = de.aggregate_cells(att, off, tab)
    imp = de.implied_whole_yard(curve, tab, cells.cell, cells.offset)
    d_of = cells.cell // 2 + 1
    gtg = cells.cell % 2 == 1
    by_d = []
    for d in range(1, 11):
        m = (d_of == d) & ~gtg
        obs = cells.y[m].sum() / cells.n[m].sum()
        by_d.append({"recorded": d, "attempts": int(cells.n[m].sum()),
                     "observed": float(obs),
                     "implied_by_curve": float((imp[m] * cells.n[m]).sum() / cells.n[m].sum())})
    worst = max(abs(r["observed"] - r["implied_by_curve"]) for r in by_d)
    # Midfield fourth down: the whole-yard model at d, the curve at D = d, the curve's
    # kernel average (which should reproduce the whole-yard value), and the mean D.
    off50 = float(off(np.array([50]), np.array([4]))[0])
    mid = []
    for d in range(1, 7):
        cell = tab.cell(np.array([d]), np.array([0]))
        p_whole = float(go.p_convert(50, d, down=4))
        c = float(de.anchor_offsets(curve, tab, cell, np.array([p_whole]))[0])
        mid.append({
            "recorded": d,
            "whole_yard_model": p_whole,
            "curve_at_true_distance_equal_to_recorded": float(curve.p(float(d), c)),
            "curve_kernel_average": float(de.implied_whole_yard(curve, tab, cell,
                                                                np.array([c]))[0]),
            "mean_true_distance": float(tab.mean()[cell][0]),
            "plausible_range": [float(de.kernel_quantiles(tab, cell, 0.05)[0]),
                                float(de.kernel_quantiles(tab, cell, 0.95)[0])],
            "curve_at_range_ends": [
                float(curve.p(float(de.kernel_quantiles(tab, cell, 0.05)[0]), c)),
                float(curve.p(float(de.kernel_quantiles(tab, cell, 0.95)[0]), c))],
            "anchor_minus_fitted_offset": c - off50,
        })  # fmt: skip
    # Additivity check: how much the per-distance anchor moves at fixed field position.
    spread = []
    for yl in (20, 50, 76):
        cs = []
        for d in range(1, 6):
            cell = tab.cell(np.array([d]), np.array([0]))
            cs.append(float(de.anchor_offsets(curve, tab, cell,
                                              np.array([float(go.p_convert(yl, d))]))[0]))
        spread.append({"yardline_100": yl, "anchors_d1_to_d5": cs,
                       "range": float(max(cs) - min(cs))})  # fmt: skip
    return {
        "headline": {"model": head[0], "penalty": head[1]},
        "observed_vs_implied_third_and_fourth_downs": by_d,
        "worst_abs_gap_d1_to_d10": float(worst),
        "midfield_fourth_down": mid,
        "anchor_additivity": spread,
    }


def curve_table(curves: dict) -> dict[str, Any]:
    grid = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    out = {}
    for (name, lam), (_, c) in curves.items():
        rel = c.h(np.array(grid)) - c.h(1.0)
        out[f"{name}|penalty={lam:g}"] = {
            "curve": c.to_dict(),
            "logit_relative_to_one_yard": dict(zip([str(g) for g in grid],
                                                   rel.round(4).tolist(), strict=True)),
        }
    return out


# -------------------------------------------------------------------- fourth downs


def real_fourth_downs(plays: pl.DataFrame) -> pl.DataFrame:
    return plays.filter(
        (pl.col("down") == 4) & pl.col("fourth_down_decision").is_not_null()
    ).sort("game_id", "order_sequence", "play_id")


def engine_rows(engine: decision.DecisionEngine, states: pl.DataFrame,
                out_of_sample: bool, what: str) -> dict[str, np.ndarray]:  # fmt: skip
    if out_of_sample:
        engine.assert_can_score(states, what)
    v = engine.action_values(states)
    kick = np.fmax(v["fg_wp"], v["punt_wp"])
    yl = states["yardline_100"].to_numpy().astype(int)
    d = np.minimum(np.minimum(states["ydstogo"].to_numpy(), yl), 40).astype(int)
    return {
        "a": v["wp_if_convert"], "b": v["wp_if_fail"], "k": kick, "p": v["p_convert"],
        "go": v["go_wp"], "d": d, "yl": yl,
        "gtg": states["goal_to_go"].fill_null(0).to_numpy().astype(int),
        "qtr": states["qtr"].to_numpy(),
        "diff": states["score_differential"].to_numpy(),
        "decision": np.array(states["fourth_down_decision"].to_list(), dtype=object),
        "game": states["game_id"].to_numpy(),
        "season": states["season"].to_numpy(),
    }  # fmt: skip


def run_propagation(rows: dict, tab: de.KernelTable, curve: de.LatentCurve, cfg: dict
                    ) -> dict[str, np.ndarray]:  # fmt: skip
    return de.propagate(curve, tab, rows["d"], rows["gtg"], rows["p"], rows["a"], rows["b"],
                        rows["k"], float(cfg["plausible_coverage"]))  # fmt: skip


def group_masks(rows: dict, cfg: dict) -> dict[str, np.ndarray]:
    fr = cfg["fragile"]
    d, yl, gtg = rows["d"], rows["yl"], rows["gtg"] == 1
    masks: dict[str, np.ndarray] = {}
    for lo, hi, label in fr["distance_groups"]:
        masks[f"distance {label}"] = (d >= lo) & (d <= hi)
    for lo, hi, label in fr["field_groups"]:
        masks[f"field {label}"] = (yl >= lo) & (yl <= hi) & ~gtg
    masks["field goal to go"] = gtg
    for lo, hi, dl in fr["distance_groups"][:3]:
        for flo, fhi, fl in fr["field_groups"]:
            masks[f"cell {dl} | {fl}"] = (d >= lo) & (d <= hi) & (yl >= flo) & (yl <= fhi) & ~gtg
    masks["fourth quarter, within 8"] = (rows["qtr"] == 4) & (np.abs(rows["diff"]) <= 8)
    return masks


def gain_bin_masks(rows: dict, cfg: dict) -> dict[str, np.ndarray]:
    gain = 100 * np.abs(de.go_value(rows["p"], rows["a"], rows["b"]) - rows["k"])
    e = cfg["fragile"]["gain_bins"]
    return {f"|gain| {lo:g}-{hi:g}" if hi < 100 else f"|gain| {lo:g}+":
            (gain >= lo) & (gain < hi) for lo, hi in zip(e[:-1], e[1:], strict=True)}  # fmt: skip


def describe(rows: dict, prop: dict, cfg: dict) -> dict[str, Any]:
    """Overall and per-group summaries plus the coaches' choices on flipping plays."""
    out = {"overall": de.summarize_flips(prop)}
    out["groups"] = {k: de.summarize_flips(prop, m) for k, m in group_masks(rows, cfg).items()}
    out["by_whole_yard_margin"] = {k: de.summarize_flips(prop, m)
                                   for k, m in gain_bin_masks(rows, cfg).items()}  # fmt: skip
    flips = prop["flips"]
    whole = de.go_value(rows["p"], rows["a"], rows["b"]) - rows["k"]
    out["flip_direction"] = {
        "whole_yard_go_long_end_kick": int((flips & (whole > 0)).sum()),
        "whole_yard_kick_short_end_go": int((flips & (whole <= 0)).sum()),
    }
    dec = rows["decision"]
    out["coach_choice_on_flipping_plays"] = {
        a: int(((dec == a) & flips).sum()) for a in ("go", "field_goal", "punt")
    }
    return out


def model_range(rows: dict, curves: dict, cfg: dict) -> dict[str, Any]:
    per = {}
    for (name, lam), (tab, c) in curves.items():
        prop = run_propagation(rows, tab, c, cfg)
        per[f"{name}|penalty={lam:g}"] = {
            "overall": de.summarize_flips(prop),
            "by_distance": {k: de.summarize_flips(prop, m)["share_flip_in_range"]
                            for k, m in group_masks(rows, cfg).items()
                            if k.startswith("distance")},
        }  # fmt: skip
    shares = [v["overall"]["share_flip_in_range"] for v in per.values()]
    differs = [v["overall"]["share_call_differs_at_true_distance"] for v in per.values()]
    evpi = [v["overall"]["mean_evpi_wp_points"] for v in per.values()]
    return {
        "per_model": per,
        "share_flip_in_range_min_max": [float(min(shares)), float(max(shares))],
        "share_call_differs_min_max": [float(min(differs)), float(max(differs))],
        "mean_evpi_wp_points_min_max": [float(min(evpi)), float(max(evpi))],
    }


def bootstrap(dev: pl.DataFrame, att: pl.DataFrame, go: Any, cfg: dict,
              head_model: de.ErrorModel, rows_sets: dict[str, dict]) -> dict[str, Any]:  # fmt: skip
    """Resample development games: refit the offset estimate and the headline curve, and
    resample each evaluated set's games. Returns per-replicate summaries."""
    b = cfg["bootstrap"]
    rng = np.random.default_rng(int(b["seed"]))
    lam = float(cfg["curve"]["penalty"])
    games = np.unique(dev["game_id"].to_numpy())
    att_g = game_index(att, games)
    lo = cfg["line_offset"]
    sel = de.line_offset_plays(dev, tuple(lo["downs"])).filter(
        pl.col("ydstogo").is_between(*lo["distances"]))
    sel_g = game_index(sel, games)
    sel_k = sel["k"].to_numpy()
    win = int(lo["window"])
    sets = {}
    for name, rows in rows_sets.items():
        g = np.unique(rows["game"])
        sets[name] = (g, np.searchsorted(g, rows["game"]))
    reps: dict[str, list] = {"offset": [], **{n: [] for n in rows_sets}}
    for _ in range(int(b["n_boot"])):
        mult = rng.multinomial(games.size, np.full(games.size, 1 / games.size))
        # offset estimate on the pooled counts, as the point estimate
        r = de.fit_line_offset(*de.line_offset_counts(sel_k, win, mult[sel_g]))
        reps["offset"].append(float(r["mean_offset"]))
        fits = fit_all_curves(att, go, cfg, [head_model], mult[att_g], (head_model.name, lam))
        tab, curve = fits[(head_model.name, lam)]
        for name, rows in rows_sets.items():
            g, gi = sets[name]
            wmult = rng.multinomial(g.size, np.full(g.size, 1 / g.size))[gi]
            prop = run_propagation(rows, tab, curve, cfg)
            summ = {"overall": de.summarize_flips(prop, weights=wmult)}
            summ["groups"] = {k: de.summarize_flips(prop, m, wmult)
                              for k, m in group_masks(rows, cfg).items()}  # fmt: skip
            reps[name].append(summ)
    return reps


def intervals(reps: list[dict]) -> dict[str, Any]:
    keys = ("share_flip_in_range", "share_call_differs_at_true_distance",
            "mean_swing_wp_points", "mean_evpi_wp_points")  # fmt: skip
    out = {"overall": {k: percentile_interval([r["overall"][k] for r in reps]) for k in keys}}
    out["groups"] = {
        g: {k: percentile_interval([r["groups"][g].get(k, np.nan) for r in reps])
            for k in ("share_flip_in_range", "share_call_differs_at_true_distance")}
        for g in reps[0]["groups"]
    }  # fmt: skip
    return out


# ------------------------------------------------------------------- the Lions case


def lions_states(cfg: dict, yardline: int | np.ndarray, distances: np.ndarray,
                 timeouts: tuple[int, int]) -> pl.DataFrame:  # fmt: skip
    L = cfg["lions"]
    yl = np.broadcast_to(np.asarray(yardline), distances.shape)
    return wp_mod.state_frame(
        down=4, ydstogo=distances, yardline_100=yl,
        game_seconds_remaining=int(L["game_seconds_remaining"]),
        score_differential=int(L["score_differential"]),
        posteam_spread=float(L["posteam_spread"]),
        posteam_timeouts_remaining=int(timeouts[0]),
        defteam_timeouts_remaining=int(timeouts[1]),
        receive_2h_ko=0, is_home=int(L["is_home"]),
    )  # fmt: skip


def margin_curve(D: np.ndarray, whole: dict[int, dict[str, float]], curve: de.LatentCurve,
                 tab: de.KernelTable, anchor_d: int = 1) -> np.ndarray:  # fmt: skip
    """Go minus best kick at true distances D. Conversion: the curve anchored so its
    kernel average at recorded `anchor_d` equals that state's whole-yard p. Conditional
    values (win probability if converted / failed) interpolated between whole-yard states
    (D <= 1 uses the one-yard state)."""
    ds = np.array(sorted(whole))
    cell = tab.cell(np.array([anchor_d]), np.array([0]))
    c = float(de.anchor_offsets(curve, tab, cell, np.array([whole[anchor_d]["p"]]))[0])
    a = np.interp(D, ds, [whole[x]["a"] for x in ds])
    b = np.interp(D, ds, [whole[x]["b"] for x in ds])
    k = np.interp(D, ds, [whole[x]["k"] for x in ds])
    return de.go_value(curve.p(D, c), a, b) - k


def engine_whole(engine: decision.DecisionEngine, cfg: dict, timeouts: tuple[int, int],
                 dmax: int = 8) -> dict[int, dict[str, float]]:  # fmt: skip
    L = cfg["lions"]
    ds = np.arange(1, dmax + 1)
    v = engine.action_values(lions_states(cfg, int(L["yardline_100"]), ds, timeouts))
    return {int(d): {"a": float(v["wp_if_convert"][i]), "b": float(v["wp_if_fail"][i]),
                     "k": float(np.fmax(v["fg_wp"][i], v["punt_wp"][i])),
                     "p": float(v["p_convert"][i]), "go": float(v["go_wp"][i])}
            for i, d in enumerate(ds)}  # fmt: skip


def dp_whole(cfg: dict, timeouts: tuple[int, int], dmax: int = 8) -> dict[int, dict] | None:
    from fourth_and_rl import mdp_game as mg

    try:
        sol = mg.load_solution()
    except (FileNotFoundError, ValueError):
        return None
    L = cfg["lions"]
    out = {}
    for d in range(1, dmax + 1):
        st = mg.GameState(int(L["game_seconds_remaining"]), int(L["score_differential"]), 4, d,
                          int(L["yardline_100"]), int(timeouts[0]), int(timeouts[1]))
        qv = sol.q_values(st)
        split = sol.play_split(st)
        go = max(v for k, v in qv.items() if k.startswith("play"))
        kick = max(qv.get("punt", -np.inf), qv.get("field_goal", -np.inf))
        out[d] = {"a": split["value_if_kept"], "b": split["value_if_lost"], "k": float(kick),
                  "p": split["p_keep_or_score"], "go": float(go), "mode": split["mode"],
                  "split_value": split["value"]}  # fmt: skip
    return out


def crossing(D: np.ndarray, m: np.ndarray) -> float | None:
    """First true distance where the margin turns negative (linear interpolation)."""
    neg = np.nonzero(m < 0)[0]
    if neg.size == 0:
        return None
    i = int(neg[0])
    if i == 0:
        return float(D[0])
    return float(D[i - 1] + (D[i] - D[i - 1]) * m[i - 1] / (m[i - 1] - m[i]))


def lions_case(engine: decision.DecisionEngine, curves: dict, cfg: dict) -> dict[str, Any]:
    L = cfg["lions"]
    head = (cfg["headline_model"], float(cfg["curve"]["penalty"]))
    lo, hi, step = (float(x) for x in L["true_distances"])
    D = np.round(np.arange(lo, hi + step / 2, step), 4)
    D_long = np.round(np.arange(lo, 8.0 + 1e-9, 0.01), 4)
    to = (int(L["posteam_timeouts_remaining"]), int(L["defteam_timeouts_remaining"]))
    eng = engine_whole(engine, cfg, to)
    tab, curve = curves[head]
    out: dict[str, Any] = {
        "situation": {k: L[k] for k in ("game_seconds_remaining", "score_differential",
                                        "yardline_100", "ydstogo", "posteam_spread",
                                        "posteam_timeouts_remaining",
                                        "defteam_timeouts_remaining")},
        "note": "Hypothetical state rebuilt from the posts. The 2026 game itself is in the "
        "sealed holdout and was not loaded. Timeouts were not stated; 3 each assumed.",
        "public_numbers_theirs": L["public_numbers"],
        "true_distances": D.tolist(),
    }  # fmt: skip
    m_eng = margin_curve(D, eng, curve, tab)
    out["one_step"] = {
        "whole_yard_margin_recorded_1": eng[1]["go"] - eng[1]["k"],
        "whole_yard_margin_recorded_2": eng[2]["go"] - eng[2]["k"],
        "whole_yard_p_convert": {str(d): eng[d]["p"] for d in (1, 2, 3)},
        "margin_by_true_distance": m_eng.round(5).tolist(),
        "margin_integrated_over_recorded_1": float(
            (tab.weights[tab.cell(1, 0)] * margin_curve(tab.nodes[tab.cell(1, 0)], eng, curve,
                                                        tab)).sum()),
        "plausible_range_recorded_1": [float(de.kernel_quantiles(tab, tab.cell(np.array([1]),
                                                                              np.array([0])),
                                                             q)[0])
                                       for q in (0.05, 0.95)],
        "break_even_true_distance": crossing(D_long, margin_curve(D_long, eng, curve, tab)),
        "by_model": {},
    }  # fmt: skip
    for (name, lam), (t2, c2) in curves.items():
        mm = margin_curve(D, eng, c2, t2)
        out["one_step"]["by_model"][f"{name}|penalty={lam:g}"] = {
            "margin_at": {str(x): float(np.interp(x, D, mm)) for x in (0.25, 1.0, 1.5, 2.0)},
            "break_even_true_distance": crossing(D_long, margin_curve(D_long, eng, c2, t2)),
        }
    # The linear shortcut against the full engine with the probability substituted.
    cell = tab.cell(np.array([1]), np.array([0]))
    c = float(de.anchor_offsets(curve, tab, cell, np.array([eng[1]["p"]]))[0])
    checks = []
    for x in (0.25, 1.0):
        p_new = float(curve.p(x, c))
        wrapped = de.ConversionOverride(engine.components, lambda yl, d, p=p_new: p)
        e2 = decision.DecisionEngine(engine.wp_model, wrapped, engine.config)
        v = e2.action_values(lions_states(cfg, int(L["yardline_100"]), np.array([1]), to))
        checks.append({"true_distance": x, "p_convert": p_new,
                       "engine_go_wp": float(v["go_wp"][0]),
                       "linear_go_wp": float(de.go_value(p_new, eng[1]["a"], eng[1]["b"]))})
    out["one_step"]["engine_substitution_check"] = checks
    sens = {}
    for pair in L["timeout_sensitivity"]:
        w = engine_whole(engine, cfg, (int(pair[0]), int(pair[1])), 3)
        mm = margin_curve(D, w, curve, tab)
        sens[f"{pair[0]}-{pair[1]}"] = {str(x): float(np.interp(x, D, mm))
                                        for x in (0.25, 1.0, 1.5, 2.0)}  # fmt: skip
    out["one_step"]["timeout_sensitivity_margin"] = sens
    dp = dp_whole(cfg, to, dmax=4)
    if dp is None:
        out["exact_game_solution"] = {"available": False}
    else:
        m_dp = margin_curve(D, dp, curve, tab)
        out["exact_game_solution"] = {
            "available": True,
            "whole_yard_margin": {str(d): dp[d]["go"] - dp[d]["k"] for d in (1, 2, 3)},
            "whole_yard_p_keep_or_score": {str(d): dp[d]["p"] for d in (1, 2, 3)},
            "go_mode": dp[1]["mode"],
            "margin_by_true_distance": m_dp.round(5).tolist(),
            "plain_interpolation_note": "Interpolating the solution's go-minus-punt linearly "
            "between recorded 1 and 2 reads recorded distance as if it were exact; the "
            "curve-based margin above replaces the solution's conversion probability by the "
            "continuous curve anchored to the solution's own probability at recorded 1.",
            "break_even_true_distance": crossing(D_long, margin_curve(D_long, dp, curve, tab)),
        }
    return out


def lions_heatmap(engine: decision.DecisionEngine, curve: de.LatentCurve, tab: de.KernelTable,
                  cfg: dict) -> dict[str, Any]:  # fmt: skip
    L = cfg["lions"]
    y0, y1 = (int(x) for x in L["heatmap_yardlines"])
    d0, d1 = (float(x) for x in L["heatmap_distances"])
    yls = np.arange(y0, y1 + 1)
    D = np.round(np.arange(d0, d1 + 1e-9, 0.05), 4)
    ds = np.arange(1, int(np.ceil(d1)) + 2)
    yy, dd = np.meshgrid(yls, ds, indexing="ij")
    to = (int(L["posteam_timeouts_remaining"]), int(L["defteam_timeouts_remaining"]))
    v = engine.action_values(lions_states(cfg, yy.ravel(), dd.ravel(), to))
    grid = np.zeros((yls.size, D.size))
    for i in range(yls.size):
        rows = slice(i * ds.size, (i + 1) * ds.size)
        whole = {int(d): {"a": float(v["wp_if_convert"][rows][j]),
                          "b": float(v["wp_if_fail"][rows][j]),
                          "k": float(np.fmax(v["fg_wp"][rows][j], v["punt_wp"][rows][j])),
                          "p": float(v["p_convert"][rows][j])}
                 for j, d in enumerate(ds)}  # fmt: skip
        grid[i] = margin_curve(D, whole, curve, tab)
    return {"yardline_100": yls, "true_distance": D, "margin": grid}


# ------------------------------------------------------------------------- figures


def figure_main(res: dict, curves: dict, go: Any, cfg: dict) -> None:
    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    ev = res["evidence"]["offset"]

    # (a) the dip at k = 0
    ax = axes[0, 0]
    k = np.array(ev["pooled_counts"]["k"])
    n = np.array(ev["pooled_counts"]["count"])
    colors = [style.NEUTRAL] * k.size
    colors[int(np.nonzero(k == -1)[0][0])] = style.MUTED
    colors[int(np.nonzero(k == 0)[0][0])] = style.VIOLET
    ax.bar(k, n / 1000, color=colors, width=0.75)
    smooth_k = k[(k != 0) & (k != -1)]
    coef = np.polyfit(smooth_k, np.log(n[(k != 0) & (k != -1)]), 2)
    xx = np.linspace(k.min(), k.max(), 200)
    ax.plot(xx, np.exp(np.polyval(coef, xx)) / 1000, color=style.INK, linewidth=1.4,
            linestyle=(0, (4, 3)), label="Smooth trend from the other yards")  # fmt: skip
    ax.set_xticks(k)
    ax.set_xticklabels([("+" if x > 0 else "") + f"{x}" for x in k])
    ax.set_xlabel("Recorded yards gained minus recorded yards to go")
    ax.set_ylabel("Plays (thousands)")
    m = ev["pooled"]["mean_offset"]
    ax.text(0.0, n[k == 0][0] / 1000 + 0.4, "gained exactly\nthe distance", ha="center",
            va="bottom", color=style.INK, fontsize=9)  # fmt: skip
    ax.text(-1.0, n[k == -1][0] / 1000 + 0.4, "one yard\nshort", ha="center", va="bottom",
            color=style.INK, fontsize=9)  # fmt: skip
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, n.max() / 1000 * 1.28)
    style.title(ax, "Plays that gain exactly the distance are too rare",
                f"2010-2023, second to fourth down, 2-10 to go. Implied mean offset of the "
                f"line inside its yard: {m:.2f}")  # fmt: skip

    # (b) the continuous curve against the whole-yard model at midfield
    ax = axes[0, 1]
    head = (cfg["headline_model"], float(cfg["curve"]["penalty"]))
    tab, curve = curves[head]
    D = np.linspace(0.02, 6.0, 300)
    band = []
    for (_, _), (t2, c2) in curves.items():
        cell = t2.cell(np.array([1]), np.array([0]))
        c = float(de.anchor_offsets(c2, t2, cell, np.array([float(go.p_convert(50, 1))]))[0])
        band.append(c2.p(D, c))
    band = np.array(band)
    ax.fill_between(D, 100 * band.min(0), 100 * band.max(0), color=style.SEQUENTIAL[0],
                    label="Range over error models and smoothness")  # fmt: skip
    cell = tab.cell(np.array([1]), np.array([0]))
    c = float(de.anchor_offsets(curve, tab, cell, np.array([float(go.p_convert(50, 1))]))[0])
    ax.plot(D, 100 * curve.p(D, c), color=style.BLUE, label="Continuous curve (true distance)")
    for row in res["comparison_whole_yard"]["midfield_fourth_down"]:
        lo_, hi_ = row["plausible_range"]
        y = 100 * row["whole_yard_model"]
        ax.plot([lo_, hi_], [y, y], color=style.INK_2, linewidth=1.2, alpha=0.8)
        ax.plot([row["mean_true_distance"]], [y], "o", color=style.INK, markersize=5)
    ax.plot([], [], "o-", color=style.INK, markersize=5, linewidth=1.2,
            label="Whole-yard model at its mean true distance\n(line: 90% plausible range)")
    ax.set_xlabel("True distance to the line to gain (yards)")
    ax.set_ylabel("Conversion probability (%)")
    ax.set_xlim(0, 6)
    ax.legend(loc="upper right", fontsize=9)
    style.title(ax, "Conversion falls continuously inside each recorded yard",
                "Fourth down at midfield. Uniform offset; development seasons.")

    # (c) fragile situations: flip share by distance and field zone (development)
    ax = axes[1, 0]
    groups = res["development"]["headline"]["groups"]
    dlabels = [g[2] for g in cfg["fragile"]["distance_groups"][:3]]
    flabels = [g[2] for g in cfg["fragile"]["field_groups"]]
    mat = np.array([[groups[f"cell {dl} | {fl}"].get("share_flip_in_range", np.nan)
                     for fl in flabels] for dl in dlabels])  # fmt: skip
    cmap = LinearSegmentedColormap.from_list("seq", [style.SURFACE, *style.SEQUENTIAL])
    im = ax.imshow(100 * mat, cmap=cmap, vmin=0, vmax=max(5, 100 * np.nanmax(mat)),
                   aspect="auto")  # fmt: skip
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            nn = groups[f"cell {dlabels[i]} | {flabels[j]}"].get("n", 0)
            val = 100 * mat[i, j]
            ink = style.SURFACE if val > 0.6 * 100 * np.nanmax(mat) else style.INK
            ax.text(j, i, f"{val:.0f}%\n(n={nn:,})", ha="center", va="center", fontsize=9,
                    color=ink)  # fmt: skip
    ax.set_xticks(range(len(flabels)))
    ax.set_xticklabels(flabels)
    ax.set_yticks(range(len(dlabels)))
    ax.set_yticklabels([f"4th and {x}" for x in dlabels])
    ax.grid(False)
    ax.set_xlabel("Field position (own end on the left)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Call flips within range (%)")
    style.title(ax, "Where the hidden fraction flips the call",
                "Share of real fourth downs, 2010-2023, uniform offset, 90% plausible range")

    # (d) by recorded distance: development and test, model range as whiskers
    ax = axes[1, 1]
    dist_keys = [f"distance {g[2]}" for g in cfg["fragile"]["distance_groups"]]
    x = np.arange(len(dist_keys))
    sets = [("development", "2010-2023", style.SEQUENTIAL[2])]
    if "test" in res:
        sets.append(("test", "2024-2025 (scored once)", style.SEQUENTIAL[5]))
    width = 0.38 if len(sets) == 2 else 0.6
    for i, (key, label, color) in enumerate(sets):
        block = res[key]
        vals = np.array([100 * block["headline"]["groups"][k]["share_flip_in_range"]
                         for k in dist_keys])  # fmt: skip
        per = block["model_range"]["per_model"].values()
        lo_ = np.array([100 * min(p["by_distance"][k] for p in per) for k in dist_keys])
        hi_ = np.array([100 * max(p["by_distance"][k] for p in per) for k in dist_keys])
        xs = x + (i - (len(sets) - 1) / 2) * width
        ax.bar(xs, vals, width=width * 0.92, color=color, label=label)
        ax.errorbar(xs, vals, yerr=[vals - lo_, hi_ - vals], fmt="none", ecolor=style.INK,
                    elinewidth=1.0, capsize=3)  # fmt: skip
    ax.set_xticks(x)
    ax.set_xticklabels([g[2] for g in cfg["fragile"]["distance_groups"]])
    ax.set_xlabel("Recorded yards to go")
    ax.set_ylabel("Fourth downs whose call flips within range (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.grid(axis="x", visible=False)
    style.title(ax, "Flips are not confined to fourth and 1",
                "Bars: uniform offset. Whiskers: range over error models and smoothness.")
    fig.tight_layout(h_pad=3.0, w_pad=2.5)
    style.save(fig, FIGURE)


def figure_question(res: dict, heat: dict) -> None:
    style.apply()
    lions = res["lions_case"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    D = np.array(lions["true_distances"])
    m1 = 100 * np.array(lions["one_step"]["margin_by_true_distance"])
    lim = max(10.0, float(np.nanmax(np.abs(m1))) * 1.25)
    series = [m1]
    if lions["exact_game_solution"].get("available"):
        m2 = 100 * np.array(lions["exact_game_solution"]["margin_by_true_distance"])
        series.append(m2)
        lim = max(lim, float(np.nanmax(np.abs(m2))) * 1.25)
    ax.axhspan(0, lim, color=style.BLUE, alpha=0.08, linewidth=0)
    ax.axhspan(-lim, 0, color=style.AQUA, alpha=0.10, linewidth=0)
    ax.axhline(0, color=style.NEUTRAL, linewidth=1.4)
    ax.plot(D, m1, color=style.INK, label="One-step engine")
    if len(series) == 2:
        ax.plot(D, series[1], color=style.INK_2, linestyle=(0, (5, 3)),
                label="Exact game solution")  # fmt: skip
    rng_ = lions["one_step"]["plausible_range_recorded_1"]
    ax.axvspan(rng_[0], rng_[1], ymin=0, ymax=0.04, color=style.INK_2, alpha=0.5, linewidth=0)
    ax.text(rng_[1], -lim * 0.90, " plausible range of\n a recorded 4th and 1", fontsize=8.5,
            color=style.INK_2, va="bottom")  # fmt: skip
    ax.text(0.06, lim * 0.92, "go is better", color=style.INK_2, fontsize=9, va="top")
    ax.text(0.06, -lim * 0.55, "punt is better", color=style.INK_2, fontsize=9, va="top")
    ax.set_xlim(0, D.max())
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("True distance to the line to gain (yards)")
    ax.set_ylabel("Go minus punt (win-probability points)")
    ax.legend(loc="upper right", fontsize=9)
    style.title(ax, "Lions, 4th and 1 at their 24: still a go at 2 yards?",
                "Tied, 4:53 left, even spread, 3 timeouts each (hypothetical rebuild)")

    ax = axes[1]
    grid = 100 * heat["margin"]
    vmax = float(np.nanpercentile(np.abs(grid), 98))
    cmap = LinearSegmentedColormap.from_list("div", [style.AQUA, "#e8e7e1", style.BLUE])
    extent = [heat["true_distance"][0], heat["true_distance"][-1],
              heat["yardline_100"][-1] + 0.5, heat["yardline_100"][0] - 0.5]  # fmt: skip
    im = ax.imshow(grid, aspect="auto", cmap=cmap, extent=extent,
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))  # fmt: skip
    ax.contour(heat["true_distance"], heat["yardline_100"], grid, levels=[0.0],
               colors=[style.INK], linewidths=1.2)  # fmt: skip
    ax.plot([lions["true_distances"][0], 2.0], [76, 76], color=style.INK, linewidth=0.8,
            linestyle=(0, (2, 2)))  # fmt: skip
    ax.text(2.05, 76, "own 24, up to 2 yards", color=style.INK, fontsize=9, va="center")
    ax.set_yticks([10, 30, 50, 70, 90])
    ax.set_yticklabels(["opp 10", "opp 30", "50", "own 30", "own 10"])
    ax.set_xlabel("True distance to the line to gain (yards)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 label="Go minus best kick (win-probability points)")
    style.title(ax, "The same clock and score across the field",
                "One-step engine. Blue: go. Aqua: kick. Black line: break-even.")
    fig.tight_layout(w_pad=2.5)
    style.save(fig, FIGURE_Q)


# --------------------------------------------------------------------------- stages


def development_part(cfg: dict) -> tuple[dict, dict, Any, Any, dict, dict]:
    """Everything fitted and measured on the development seasons. Returns the results,
    the curves, the engine, the fit inputs, the config with the penalty resolved, and the
    evaluated development fourth downs."""
    dev = splits.load_dev_plays()
    de.assert_development_only(dev, "M9 distance-error fits")
    engine = decision.load_engine(dev)
    go = engine.components.go
    rng = np.random.default_rng(int(cfg["seed"]))
    original_hash = config_hash(cfg)
    evidence = {"integer_consistency": integer_consistency(dev),
                "offset": offset_evidence(dev, cfg)}  # fmt: skip
    sim = {"offset_estimator": offset_simulation(dev, cfg, rng)}
    raw = evidence["offset"]["pooled"]["mean_offset"]
    m_hat = raw - sim["offset_estimator"]["bias_pooled"]
    evidence["offset"]["pooled"]["bias_corrected_mean_offset"] = float(m_hat)
    hist = evidence["offset"]["history"]
    hist["predicted_difference_if_positions_carried_and_uniform"] = sim["offset_estimator"][
        "predicted_history_difference_carried_uniform"]
    models = de.error_models(cfg, fitted_mean=m_hat)
    att = de.conversion_attempts(dev, tuple(cfg["curve"]["fit_downs"]))
    head_model = next(m for m in models if m.name == cfg["headline_model"])
    cfg, cv = resolve_penalty(att, go, cfg, head_model)
    sim["deconvolution"] = deconvolution_simulation(cfg, rng)
    curves = fit_all_curves(att, go, cfg, models)
    res: dict[str, Any] = {
        "question": "Was it really fourth and 1? How often the hidden fraction of a yard in "
        "recorded distance flips the recommended fourth-down call, and how much win "
        "probability rides on it.",
        "method": "measurement-error model of the true distance given the recorded one "
        "(evidence: the dip at k = 0), a deconvolved continuous conversion curve, and "
        "propagation through the one-step decision engine; see docs/notes/m9_distance_error.md",
        "seed": int(cfg["seed"]),
        "config_hash": original_hash,
        "penalty_selection": cv,
        "seasons": {"development": list(splits.dev_seasons()),
                    "test": list(splits.test_seasons()),
                    "holdout_not_loaded": list(splits.holdout_seasons())},
        "evidence": evidence,
        "error_models": [m.__dict__ for m in models],
        "simulation_checks": sim,
        "curves": curve_table(curves),
        "comparison_whole_yard": whole_yard_comparison(att, go, cfg, curves),
    }  # fmt: skip
    rows_dev = engine_rows(engine, real_fourth_downs(dev), False, "development fourth downs")
    head = (cfg["headline_model"], float(cfg["curve"]["penalty"]))
    tab, curve = curves[head]
    prop = run_propagation(rows_dev, tab, curve, cfg)
    res["development"] = {
        "note": "Engine is in sample here (its models were fitted on these seasons); these "
        "numbers are for design. The test block is the out-of-sample answer.",
        "n_fourth_downs": int(rows_dev["d"].size),
        "headline": describe(rows_dev, prop, cfg),
        "model_range": model_range(rows_dev, curves, cfg),
    }
    return res, curves, engine, (dev, att, go, head_model), cfg, rows_dev


def evaluation_plan(cfg: dict, curves: dict, original_hash: str) -> dict[str, Any]:
    frozen = {f"{n}|penalty={lam:g}": np.round(c.values, 6).tolist()
              for (n, lam), (_, c) in curves.items()}  # fmt: skip
    plan = {
        "written_before_test": True,
        "config_hash": original_hash,
        "frozen_curve_values": frozen,
        "headline": {"model": cfg["headline_model"], "penalty": cfg["curve"]["penalty"],
                     "plausible_coverage": cfg["plausible_coverage"]},
        "test_rows": "2024-2025 plays with down 4 and a recorded decision (go, field goal, "
        "punt), valued by the headline engine (decision.load_engine) out of sample",
        "metrics": ["share_flip_in_range", "share_call_differs_at_true_distance",
                    "mean_swing_wp_points", "mean_evpi_wp_points", "the same by group"],
        "groups": "configs/distance_error.yaml fragile: distance, field position, their "
        "cross for distances 1 to 3, goal to go, fourth quarter within 8",
        "intervals": "game-level bootstrap: development games refit the curve, test games "
        "resampled for the rows; plus the range over error models and smoothness",
        "scored_once": "the script refuses to score the test seasons a second time",
    }  # fmt: skip
    plan["plan_hash"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
    return plan


def stage_dev(cfg: dict) -> None:
    old = read_json(RESULT) if RESULT.exists() else None
    res, curves, engine, (dev, att, go, head_model), cfg, rows_dev = development_part(cfg)
    plan = evaluation_plan(cfg, curves, res["config_hash"])
    if old and "test" in old:
        if old.get("evaluation_plan", {}).get("plan_hash") != plan["plan_hash"]:
            raise SystemExit("test seasons were already scored under a different plan; "
                             "the development stage would change the frozen plan")
        res["test"] = old["test"]
    res["evaluation_plan"] = plan
    reps = bootstrap(dev, att, go, cfg, head_model, {"development": rows_dev})
    pooled = res["evidence"]["offset"]["pooled"]
    pooled["bootstrap_interval"] = percentile_interval(reps["offset"])
    bias = res["simulation_checks"]["offset_estimator"]["bias_pooled"]
    pooled["bias_corrected_bootstrap_interval"] = [x - bias for x in pooled["bootstrap_interval"]]
    res["development"]["bootstrap"] = {"n_boot": len(reps["development"]),
                                       **intervals(reps["development"])}  # fmt: skip
    res["lions_case"] = lions_case(engine, curves, cfg)
    head = (cfg["headline_model"], float(cfg["curve"]["penalty"]))
    heat = lions_heatmap(engine, curves[head][1], curves[head][0], cfg)
    write_json(res, RESULT)
    figure_main(res, curves, go, cfg)
    figure_question(res, heat)
    print(json.dumps(res["development"]["headline"]["overall"], indent=1))
    print(json.dumps(res["development"]["model_range"]["share_flip_in_range_min_max"]))


def stage_test(cfg: dict) -> None:
    if not RESULT.exists():
        raise SystemExit("run --stage dev first: the evaluation plan must be written first")
    saved = read_json(RESULT)
    if "test" in saved:
        raise SystemExit("the test seasons were already scored for M9; they are scored once")
    res, curves, engine, (dev, att, go, head_model), cfg, rows_dev = development_part(cfg)
    plan = evaluation_plan(cfg, curves, res["config_hash"])
    if plan["plan_hash"] != saved.get("evaluation_plan", {}).get("plan_hash"):
        raise SystemExit("the development fits no longer match the saved evaluation plan")
    test = splits.load_test_plays(cfg["test_purpose"])
    rows_test = engine_rows(engine, real_fourth_downs(test), True, "M9 test fourth downs")
    head = (cfg["headline_model"], float(cfg["curve"]["penalty"]))
    tab, curve = curves[head]
    prop = run_propagation(rows_test, tab, curve, cfg)
    reps = bootstrap(dev, att, go, cfg, head_model, {"test": rows_test})
    saved["test"] = {
        "n_fourth_downs": int(rows_test["d"].size),
        "seasons": sorted({int(s) for s in rows_test["season"]}),
        "headline": describe(rows_test, prop, cfg),
        "model_range": model_range(rows_test, curves, cfg),
        "bootstrap": {"n_boot": len(reps["test"]), **intervals(reps["test"])},
    }
    write_json(saved, RESULT)
    figure_main(saved, curves, go, cfg)
    print(json.dumps(saved["test"]["headline"]["overall"], indent=1))
    print(json.dumps(saved["test"]["model_range"]["share_flip_in_range_min_max"]))


def stage_figures(cfg: dict) -> None:
    """Redraw both figures from the saved results (development fits recomputed, the test
    seasons not read)."""
    saved = read_json(RESULT)
    res, curves, engine, (_, _, go, _), cfg, _ = development_part(cfg)
    if evaluation_plan(cfg, curves, res["config_hash"])["plan_hash"] != saved[
            "evaluation_plan"]["plan_hash"]:
        raise SystemExit("the development fits no longer match the saved evaluation plan")
    head = (cfg["headline_model"], float(cfg["curve"]["penalty"]))
    figure_main(saved, curves, go, cfg)
    figure_question(saved, lions_heatmap(engine, curves[head][1], curves[head][0], cfg))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=["dev", "test", "figures"], required=True)
    args = parser.parse_args()
    cfg = de.load_config()
    {"dev": stage_dev, "test": stage_test, "figures": stage_figures}[args.stage](cfg)


if __name__ == "__main__":
    main()
