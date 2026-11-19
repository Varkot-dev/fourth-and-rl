"""M10: how much does being predictable cost an offense?

Stage 1 (development seasons 2010 to 2023 only):
  1. call models  P(pass) before the snap from (a) the situation and the offense's
                  earlier-games tendencies, (a2) the same plus hand-made features of the
                  calls and results earlier in the game and drive, (a3) plus shotgun and
                  no-huddle (sensitivity: the defense sees these pre-snap), (b0) a small
                  network on the situation, (b) the same network plus a GRU over the
                  game's play sequence. Season-blocked validation.
  2. forward      the primary model refitted season by season, each season scored by a
                  model fitted only on earlier seasons (2013 to 2023).
  3. cost         EPA and success of runs and passes by P(the call actually made), raw,
                  within situation, and within situation, offense-season and
                  defense-season. Slopes. Game-level bootstrap refits everything.
  4. teams        predictability and its cost per team-season, with shrinkage, noise
                  share and persistence; relation to offensive success.
  5. first down   first and 10 runs: how predictable, what it costs.
  The evaluation plan is written to results/m10_predictability.json BEFORE any
  test-season row is read.
Stage 2 (once): 2024 to 2025 through `splits.load_test_plays`, scored by the models
fitted on development seasons; the cost analysis repeated on them.

EPA is nflfastR's model (quoted as theirs). Play classification, the neutral filter and
the situation cells are question 2's (`run_pass`).

Outputs:
  results/m10_predictability.json
  assets/m10_predictability.png          the cost curves and what play history adds
  assets/m10_team_predictability.png     offenses: predictability and its cost

Run:  python scripts/m10_predictability.py                 (about 45 minutes, 2 threads)
      python scripts/m10_predictability.py --figures-only  (redraw from the JSON)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from typing import Any

os.environ.setdefault("POLARS_MAX_THREADS", "2")  # the machine is shared
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from fourth_and_rl import predictability as pr  # noqa: E402
from fourth_and_rl import run_pass as rp  # noqa: E402
from fourth_and_rl import splits, style  # noqa: E402
from fourth_and_rl.io import read_json, write_json  # noqa: E402
from fourth_and_rl.paths import ASSETS, RESULTS  # noqa: E402

OUT = RESULTS / "m10_predictability.json"
FIG = ASSETS / "m10_predictability.png"
FIG_TEAM = ASSETS / "m10_team_predictability.png"
Q2 = RESULTS / "questions" / "q2_run_vs_pass.json"
TEST_PURPOSE = (
    "M10 predictability: score the call models fitted on 2010-2023 on 2024-2025 once, "
    "and repeat the cost-of-predictability analysis on those seasons"
)
COLORS = {"run": style.ORANGE, "pass": style.BLUE}
LABELS = {"run": "Run", "pass": "Pass"}
CALL_CODE = {"run": 0, "pass": 1}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def codes(*cols: np.ndarray) -> np.ndarray:
    """Dense integer codes of the combinations of the given columns."""
    stacked = np.column_stack([np.asarray(c).astype(str) for c in cols])
    _, inv = np.unique(stacked, axis=0, return_inverse=True)
    return inv.ravel().astype(np.int64)


def strip(obj: Any) -> Any:
    """Drop bootstrap arrays before writing JSON."""
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k != "boot"}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- data


def load_table(plays: pl.DataFrame, cfg: dict, rcfg: dict) -> pl.DataFrame:
    extra = rp.load_extra(sorted(splits.seasons_in(plays)), rcfg)
    return pr.build_features(pr.called_plays(plays, extra, rcfg), cfg)


def rows_of(table: pl.DataFrame, seasons: list[int]) -> np.ndarray:
    return np.where(np.isin(table["season"].to_numpy(), seasons))[0]


def season_range(a: int, b: int) -> list[int]:
    return list(range(a, b + 1))


# --------------------------------------------------------------------------- call models


def make_model(name: str, cfg: dict, seed: int) -> Any:
    v = cfg["variants"][name]
    names = pr.feature_names(cfg, v["groups"])
    if v["kind"] == "gbm":
        return pr.GBMModel(names, cfg["gbm"], seed, cfg["threads"])
    return pr.SeqModel(names, cfg["seq"], seed, cfg["threads"], v["use_history"])


def fit_predict(name: str, table: pl.DataFrame, train: list[int], valid: int,
                score_table: pl.DataFrame, score_rows: np.ndarray, cfg: dict,
                seed: int) -> tuple[np.ndarray, Any]:  # fmt: skip
    """Fit variant `name` on `train` seasons (early stopping on `valid`) and return
    P(pass) on `score_table[score_rows]`."""
    m = make_model(name, cfg, seed)
    tr, va = rows_of(table, train), rows_of(table, [valid])
    if isinstance(m, pr.GBMModel):
        m.fit(table[tr], table[va])
        return m.predict(score_table[score_rows]), m
    m.fit(table, tr, va)
    return m.predict(score_table, score_rows), m


def model_info(m: Any) -> dict[str, Any]:
    if isinstance(m, pr.GBMModel):
        imp = m.importance()
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:8]
        return {"best_iteration": m.best_iteration_, "top_gain_share": dict(top)}
    return {"best_epoch": m.best_epoch_, "valid_log_loss_by_epoch": m.curve_}


COMPARISONS = [  # (better candidate, baseline, what the difference answers)
    ("situation_history", "situation", "hand-made play history over situation only"),
    ("nn_sequence", "nn_situation", "GRU over the game's plays over the same network without it"),
    ("nn_sequence", "situation_history", "GRU sequence model over hand-made history (boosting)"),
    ("nn_situation", "situation", "network over boosting, both situation only"),
    ("situation_history_formation", "situation_history",
     "shotgun and no-huddle over history (sensitivity)"),
]  # fmt: skip


def compare_models(y: np.ndarray, preds: dict[str, np.ndarray], game: np.ndarray,
                   down: np.ndarray, cfg: dict, seed: int) -> dict[str, Any]:  # fmt: skip
    """Scores of every variant and paired log-loss differences (candidate minus
    baseline, per 1,000 calls; negative = the candidate predicts better)."""
    gc = pr.game_codes(game)
    w = rp.boot_weights(int(gc.max()) + 1, cfg["n_boot"], seed)
    ll = {k: pr.log_loss_rows(y, p) for k, p in preds.items()}
    out: dict[str, Any] = {"scores": {k: pr.score(y, p) for k, p in preds.items()},
                           "differences_per_1000": []}  # fmt: skip
    for a, b, what in COMPARISONS:
        if a in ll and b in ll:
            d = pr.paired_diff(ll[a], ll[b], gc, w, cfg["level"])
            out["differences_per_1000"].append(
                {"candidate": a, "baseline": b, "what": what,
                 **{k: (None if v is None else 1000 * v) for k, v in d.items()}})  # fmt: skip
    out["log_loss_by_down"] = {
        k: {int(dn): float(v[down == dn].mean()) for dn in np.unique(down)}
        for k, v in ll.items()
    }
    return out


def validation(table: pl.DataFrame, cfg: dict) -> tuple[dict[str, Any], dict]:
    """Season-blocked comparison of all variants on development seasons."""
    first = cfg["first_train_season"]
    blocks, pooled_y, pooled_p, pooled_g, pooled_d = [], [], {}, [], []
    for i, block in enumerate(cfg["validation_blocks"]):
        valid = block[0] - 1
        train = season_range(first, valid - 1)
        rows = rows_of(table, block)
        y = table["is_pass"].to_numpy()[rows]
        preds, info = {}, {}
        for name in cfg["variants"]:
            t0 = time.time()
            preds[name], m = fit_predict(name, table, train, valid, table, rows, cfg,
                                         cfg["seed"] + i)  # fmt: skip
            info[name] = model_info(m)
            log(f"block {block} {name}: log loss {pr.log_loss_rows(y, preds[name]).mean():.4f}"
                f" ({time.time() - t0:.0f}s)")
        game = table["game_id"].to_numpy()[rows]
        down = table["down"].to_numpy()[rows]
        res = compare_models(y, preds, game, down, cfg, cfg["seed"] + 100 + i)
        blocks.append({"block": block, "train": [train[0], train[-1]], "early_stop": valid,
                       "models": info, **res})  # fmt: skip
        pooled_y.append(y)
        pooled_g.append(game)
        pooled_d.append(down)
        for k, v in preds.items():
            pooled_p.setdefault(k, []).append(v)
    y = np.concatenate(pooled_y)
    preds = {k: np.concatenate(v) for k, v in pooled_p.items()}
    pooled = compare_models(y, preds, np.concatenate(pooled_g), np.concatenate(pooled_d), cfg,
                            cfg["seed"] + 199)  # fmt: skip
    return {"blocks": blocks, "pooled": pooled}, preds


def forward_predictions(table: pl.DataFrame, cfg: dict, variants: list[str]
                        ) -> tuple[dict[str, np.ndarray], dict]:  # fmt: skip
    """P(pass) for every play from `first_forward_season` on, each season scored by a
    model fitted on earlier seasons only."""
    first, start = cfg["first_train_season"], cfg["first_forward_season"]
    last = max(splits.seasons_in(table))
    out = {v: np.full(table.height, np.nan) for v in variants}
    info: dict[str, dict] = {v: {} for v in variants}
    for s in range(start, last + 1):
        rows = rows_of(table, [s])
        for v in variants:
            p, m = fit_predict(v, table, season_range(first, s - 2), s - 1, table, rows, cfg,
                               cfg["seed"] + s)  # fmt: skip
            out[v][rows] = p
            info[v][str(s)] = model_info(m).get("best_iteration")
        log(f"forward season {s} done")
    return out, info


# --------------------------------------------------------------------------- cost


def cost_sample(table: pl.DataFrame, p_pass: np.ndarray, cfg: dict, rcfg: dict,
                downs: list[int] | None = None, ydstogo: int | None = None) -> pl.DataFrame:
    """Neutral plays on the cost downs with a prediction, and the codes used below."""
    c = cfg["cost"]
    df = table.with_columns(pl.Series("p_pass", p_pass))
    m = (pl.col("neutral") & pl.col("down").cast(pl.Int64).is_in(downs or c["downs"])
         & pl.col("p_pass").is_not_nan() & pl.col("p_pass").is_not_null())  # fmt: skip
    if ydstogo is not None:
        m = m & (pl.col("ydstogo") == ydstogo)
    df = df.filter(m)
    y = df["is_pass"].to_numpy()
    pc = pr.p_call(y, df["p_pass"].to_numpy())
    score_b = rp.bin_index(df["score_differential"].to_numpy(), c["score_edges"])
    half = (df["game_half"].cast(pl.String) == "Half1").to_numpy().astype(int)
    sit = codes(df["cell"].to_numpy(), score_b, half)
    season = df["season"].to_numpy()
    return df.with_columns(
        pl.Series("p_call", pc),
        pl.Series("surprise", -np.log(pc)),
        pl.Series("pbin", rp.bin_index(pc, c["p_edges"][:-1]).astype(np.int64)),
        pl.Series("sit", sit),
        pl.Series("off_season", codes(df["posteam"].to_numpy(), season)),
        pl.Series("def_season", codes(df["defteam"].to_numpy(), season)),
        pl.Series("gcode", pr.game_codes(df["game_id"].to_numpy())),
    )


def controls_for(sub: pl.DataFrame, spec: str) -> list[np.ndarray]:
    def dense(name: str) -> np.ndarray:
        return codes(sub[name].to_numpy())

    if spec == "raw":
        return []
    if spec == "situation":
        return [dense("sit")]
    if spec == "within_team":
        return [dense("sit"), dense("off_season"), dense("def_season")]
    raise ValueError(spec)


SPECS = ("raw", "situation", "within_team")


def cost_curves(sample: pl.DataFrame, cfg: dict, seed: int, specs: tuple[str, ...] = SPECS,
                outcomes: tuple[str, ...] = ("epa", "success")) -> dict[str, Any]:  # fmt: skip
    """Curves (by bin of P(call)) and slopes for runs and passes. One game-resample
    matrix is shared by every curve so differences between curves are coherent."""
    c = cfg["cost"]
    n_bins = len(c["p_edges"]) - 1
    gcode = sample["gcode"].to_numpy()
    w = rp.boot_weights(int(gcode.max()) + 1, c["n_boot"], seed)
    out: dict[str, Any] = {"p_edges": c["p_edges"], "n_games": int(gcode.max()) + 1}
    for call in ("run", "pass"):
        m = (sample["is_pass"] == CALL_CODE[call]).to_numpy()
        sub = sample.filter(pl.Series(m))
        g = gcode[m]
        share = np.bincount(sub["pbin"].to_numpy(), minlength=n_bins) / sub.height
        res: dict[str, Any] = {"n": sub.height, "mean_p_call": float(sub["p_call"].mean()),
                               "share_by_bin": share.tolist()}  # fmt: skip
        for spec in specs:
            ctrl = controls_for(sub, spec)
            for y_name in outcomes:
                y = sub[y_name].to_numpy().astype(float)
                t0 = time.time()
                curve = pr.bin_curve(y, sub["pbin"].to_numpy(), n_bins, ctrl, g, w, cfg)
                slope = (pr.slope_fit(y, sub["p_call"].to_numpy(), ctrl, g, w, cfg) if ctrl
                         else raw_slope(y, sub["p_call"].to_numpy(), g, w, cfg))  # fmt: skip
                res[f"{spec}_{y_name}"] = {"curve": curve, "slope": slope}
                log(f"  {call} {spec} {y_name}: slope {slope['value']:+.4f} "
                    f"({time.time() - t0:.0f}s)")
        out[call] = res
    return out


def raw_slope(y: np.ndarray, x: np.ndarray, g: np.ndarray, w: np.ndarray, cfg: dict
              ) -> dict[str, Any]:  # fmt: skip
    xm = np.column_stack([np.ones_like(x), x])
    r = rp.cluster_ols(xm, y, np.ones_like(y), g, w, cfg["level"])
    return {"value": float(r["coef"][1]), "low": float(r["low"][1]), "high": float(r["high"][1]),
            "boot": r["boot"][:, 1]}  # fmt: skip


# --------------------------------------------------------------------------- teams


def excess(sample: pl.DataFrame) -> np.ndarray:
    """P(call) minus the league mean P(call) for the same call in the same situation
    group: how much more expected this call was than an average offense's same call."""
    key = codes(sample["sit"].to_numpy(), sample["is_pass"].to_numpy())
    pc = sample["p_call"].to_numpy()
    mean = np.bincount(key, weights=pc) / np.bincount(key)
    return pc - mean[key]


def team_table(sample: pl.DataFrame, curves: dict[str, Any], cfg: dict, seed: int,
               spec: str = "within_team") -> tuple[pl.DataFrame, dict[str, Any]]:  # fmt: skip
    """Per offense-season: predictability (log loss, mean P(call), excess over the league
    in the same situations) and the cost of that excess in EPA per game, with the
    pooled within-team slopes. Bootstrap over games for intervals; empirical-Bayes
    shrinkage across team-seasons."""
    c = cfg["cost"]
    x = excess(sample)
    sample = sample.with_columns(pl.Series("excess", x),
                                 pl.Series("ll", -np.log(sample["p_call"].to_numpy())))  # fmt: skip
    ts_key = codes(sample["posteam"].to_numpy(), sample["season"].to_numpy())
    n_ts = int(ts_key.max()) + 1
    gcode = sample["gcode"].to_numpy()
    n_games = int(gcode.max()) + 1
    is_pass = sample["is_pass"].to_numpy().astype(int)
    beta = {k: curves[k][f"{spec}_epa"]["slope"] for k in ("run", "pass")}
    b_point = np.array([beta["run"]["value"], beta["pass"]["value"]])
    b_boot = np.column_stack([beta["run"]["boot"], beta["pass"]["boot"]])
    w = rp.boot_weights(n_games, c["n_boot"], seed)
    if w.shape[0] != b_boot.shape[0]:
        raise ValueError("team bootstrap must use the same number of resamples as the slopes")
    # Per game sums of the excess by team-season and call, and team-season game counts.
    cols = np.column_stack([np.ones(len(x)), x, sample["ll"].to_numpy()])
    s = rp.game_sums(gcode, n_games, ts_key * 2 + is_pass, n_ts * 2, cols)  # games, ts*2, 3
    s = s.reshape(n_games, n_ts, 2, 3)
    games_ts = (s[..., 0].sum(axis=2) > 0).astype(float)  # games, ts
    point_x = s[..., 1].sum(axis=0)  # ts, 2
    point_g = games_ts.sum(axis=0)
    cost_point = -(point_x @ b_point) / point_g
    bx = np.einsum("bg,gtc->btc", w, s[..., 1])
    bg = w @ games_ts
    cost_boot = -np.einsum("btc,bc->bt", bx, b_boot) / bg
    cost_boot_fixed = -(bx @ b_point) / bg  # team noise only (slope held at its estimate)
    lo, hi = rp._interval(cost_boot, cfg["level"])
    n_plays = s[..., 0].sum(axis=(0, 2))
    ex_mean = point_x.sum(axis=1) / n_plays
    ex_boot = bx.sum(axis=2) / np.einsum("bg,gt->bt", w, s[..., 0].sum(axis=2))
    ll_mean = s[..., 2].sum(axis=(0, 2)) / n_plays
    keys = (sample.select("posteam", "season").with_columns(pl.Series("k", ts_key))
            .unique("k").sort("k"))  # fmt: skip
    tab = keys.with_columns(
        pl.Series("n_plays", n_plays), pl.Series("n_games", point_g),
        pl.Series("pass_rate", s[..., 0][:, :, 1].sum(axis=0) / n_plays),
        pl.Series("log_loss", ll_mean),
        pl.Series("excess", ex_mean), pl.Series("excess_se", ex_boot.std(axis=0, ddof=1)),
        pl.Series("cost_per_game", cost_point),
        pl.Series("cost_low", lo), pl.Series("cost_high", hi),
        pl.Series("cost_se_team", cost_boot_fixed.std(axis=0, ddof=1)),
        pl.Series("excess_run", point_x[:, 0] / np.maximum(s[..., 0].sum(axis=0)[:, 0], 1)),
        pl.Series("excess_pass", point_x[:, 1] / np.maximum(s[..., 0].sum(axis=0)[:, 1], 1)),
    )  # fmt: skip
    tab = tab.filter(pl.col("n_plays") >= c["min_team_plays"])
    summary: dict[str, Any] = {"n_team_seasons": tab.height}
    for col, se in (("excess", "excess_se"), ("cost_per_game", "cost_se_team")):
        eb = pr.shrink(tab[col].to_numpy(), tab[se].to_numpy())
        tab = tab.with_columns(pl.Series(f"{col}_shrunk", eb["shrunk"]),
                               pl.Series(f"{col}_post_sd", eb["post_sd"]))  # fmt: skip
        summary[col] = {k: v for k, v in eb.items()
                        if k not in ("shrunk", "post_sd", "reliability")}  # fmt: skip
        summary[col]["mean_reliability"] = float(np.mean(eb["reliability"]))
        summary[col]["persistence"] = pr.persistence(tab, col)
        summary[col]["observed_sd"] = float(tab[col].std())
    summary["league_cost_per_game_note"] = (
        "costs are relative to an average offense in the same situations, so they average "
        "to about zero across the league by construction")
    summary["slopes_used"] = {k: strip(v) for k, v in beta.items()}
    return tab.drop("k"), summary


def team_slopes(sample: pl.DataFrame, cfg: dict) -> dict[str, Any]:
    """Does the cost of predictability itself differ by offense? Per team-season slope of
    EPA on P(call), after the within-team controls are absorbed (separately for runs and
    passes, then pooled), with game-clustered standard errors and shrinkage."""
    xr, yr = np.zeros(sample.height), np.zeros(sample.height)
    for call in ("run", "pass"):
        m = (sample["is_pass"] == CALL_CODE[call]).to_numpy()
        sub = sample.filter(pl.Series(m))
        ctrl = controls_for(sub, "within_team")
        xr[m] = pr.residualize(sub["p_call"].to_numpy().astype(float), ctrl, cfg)
        yr[m] = pr.residualize(sub["epa"].to_numpy().astype(float), ctrl, cfg)
    key = codes(sample["posteam"].to_numpy(), sample["season"].to_numpy())
    game = sample["game_id"].to_numpy()
    est, se = [], []
    for k in range(int(key.max()) + 1):
        m = key == k
        if m.sum() < cfg["cost"]["min_team_plays"]:
            continue
        b, s = pr.cluster_slope(xr[m], yr[m], game[m])
        est.append(b)
        se.append(s)
    eb = pr.shrink(np.array(est), np.array(se))
    pooled, pooled_se = pr.cluster_slope(xr, yr, game)
    return {
        "n_team_seasons": len(est),
        "pooled_slope": pooled,
        "pooled_se": pooled_se,
        "observed_sd_of_team_slopes": float(np.std(est, ddof=1)),
        "mean_se": float(np.mean(se)),
        "tau": eb["tau"],
        "noise_share": eb["noise_share"],
        "shrunk_range": [float(eb["shrunk"].min()), float(eb["shrunk"].max())],
    }


def success_regression(sample: pl.DataFrame, cfg: dict, seed: int) -> dict[str, Any]:
    """Game level: is an offense's EPA per play in a game related to how predictable its
    calls were in that game, once its earlier-games passing and running efficiency (our
    stand-in for quarterback and offense quality) and the opponent's earlier-games
    defense are controlled for? Second spec: within team-season (demeaned)."""
    x = excess(sample)
    df = sample.with_columns(pl.Series("excess", x))
    g = df.group_by("game_id", "posteam", "season").agg(
        pl.col("epa").mean().alias("epa"), (pl.col("excess").mean() * 10).alias("excess10"),
        pl.len().alias("n"),
        pl.col("tend_pass_epa").first(), pl.col("tend_run_epa").first(),
        pl.col("opp_def_epa_allowed").first(), pl.col("tend_pass_rate_ne").first(),
    ).sort("game_id", "posteam")  # fmt: skip
    gc = pr.game_codes(g["game_id"].to_numpy())
    w = rp.boot_weights(int(gc.max()) + 1, cfg["n_boot"], seed)
    names = ["excess10", "tend_pass_epa", "tend_run_epa", "opp_def_epa_allowed",
             "tend_pass_rate_ne"]  # fmt: skip
    xm = g.select(names).to_numpy().astype(float)
    y = g["epa"].to_numpy().astype(float)
    wt = g["n"].to_numpy().astype(float)
    seasons = rp.dummies(g["season"].to_numpy())
    out: dict[str, Any] = {"n_team_games": g.height, "predictors": names,
                           "unit": "EPA per play in the game; excess10 is the game's mean "
                                   "excess P(call) times 10 (per 10 points)"}  # fmt: skip
    r = rp.cluster_ols(np.column_stack([np.ones(len(y)), xm, seasons]), y, wt, gc, w,
                       cfg["level"])  # fmt: skip
    out["with_controls"] = {n: {"coef": float(r["coef"][i + 1]), "low": float(r["low"][i + 1]),
                                "high": float(r["high"][i + 1])} for i, n in enumerate(names)}
    r0 = rp.cluster_ols(np.column_stack([np.ones(len(y)), xm[:, :1], seasons]), y, wt, gc, w,
                        cfg["level"])  # fmt: skip
    out["no_controls"] = {"coef": float(r0["coef"][1]), "low": float(r0["low"][1]),
                          "high": float(r0["high"][1])}  # fmt: skip
    ts = codes(g["posteam"].to_numpy(), g["season"].to_numpy())

    def demean(v: np.ndarray) -> np.ndarray:
        s = np.bincount(ts, weights=wt * v) / np.bincount(ts, weights=wt)
        return v - s[ts]

    r1 = rp.cluster_ols(np.column_stack([demean(xm[:, 0])]), demean(y), wt, gc, w,
                        cfg["level"])  # fmt: skip
    out["within_team_season"] = {"coef": float(r1["coef"][0]), "low": float(r1["low"][0]),
                                 "high": float(r1["high"][0]),
                                 "note": "team-season means removed from both sides; the "
                                         "prior-games controls are constant-ish within a "
                                         "season and are absorbed"}  # fmt: skip
    return out


# --------------------------------------------------------------------------- first down


def first_down(table: pl.DataFrame, p_pass: np.ndarray, cfg: dict, rcfg: dict, seed: int
               ) -> dict[str, Any]:  # fmt: skip
    """First and 10 in neutral game script: how predictable the runs are, by team, and
    what that predictability costs."""
    s = cost_sample(table, p_pass, cfg, rcfg, downs=[1], ydstogo=cfg["first_down"]["ydstogo"])
    other = cost_sample(table, p_pass, cfg, rcfg)
    runs_other = other.filter((pl.col("is_pass") == 0) & (pl.col("down") != 1))
    curves = cost_curves(s, cfg, seed, specs=("raw", "within_team"), outcomes=("epa", "success"))
    tab, summ = team_table(s, curves, cfg, seed)
    runs = s.filter(pl.col("is_pass") == 0)
    by_team = (runs.with_columns(pl.Series("excess", excess(s)[(s["is_pass"] == 0).to_numpy()]))
               .group_by("posteam", "season").agg(
                   pl.len().alias("n_runs"), pl.col("p_call").mean().alias("mean_p_run"),
                   (pl.col("p_call") >= 0.65).mean().alias("share_runs_p65"),
                   pl.col("excess").mean().alias("excess_run"))
               )  # fmt: skip
    tab = tab.join(by_team, on=["posteam", "season"], how="left")
    b_run = curves["run"]["within_team_epa"]["slope"]
    hi_lo = expected_vs_surprising(runs, cfg)
    return {
        "n_plays": s.height,
        "run_share": float(1 - s["is_pass"].mean()),
        "mean_p_call_runs": float(runs["p_call"].mean()),
        "mean_p_call_passes": float(s.filter(pl.col("is_pass") == 1)["p_call"].mean()),
        "mean_p_call_runs_other_downs": float(runs_other["p_call"].mean()),
        "share_runs_with_p_run_65_plus": float((runs["p_call"] >= 0.65).mean()),
        "log_loss": float(s["surprise"].mean()),
        "curves": curves,
        "run_slope_per_10_points": {k: (None if v is None else v / 10)
                                    for k, v in strip(b_run).items()
                                    if k in ("value", "low", "high")},  # fmt: skip
        "expected_vs_surprising_runs": hi_lo,
        "q2_first_and_10_gap": q2_gap(),
        "teams_summary": summ,
        "_team_table": tab,
    }


def q2_gap() -> dict[str, Any] | None:
    """Question 2's pass-minus-run EPA gap on neutral first and 10 (its own number, from
    its own results file, for scale only)."""
    if not Q2.exists():
        return None
    for r in read_json(Q2)["puzzle"]["neutral_named_situations"]:
        if r["situation"] == "1st and 10":
            return {"source": "results/questions/q2_run_vs_pass.json (question 2, all "
                              "seasons 2010-2025)", **r["epa"]["pass_minus_run"]}  # fmt: skip
    return None


def expected_vs_surprising(runs: pl.DataFrame, cfg: dict) -> dict[str, Any]:
    """Raw mean EPA of first-down runs the model expected (P(run) 0.65 or more) and of
    surprising ones (below 0.35)."""
    hi = runs.filter(pl.col("p_call") >= 0.65)
    lo = runs.filter(pl.col("p_call") < 0.35)
    return {"expected": {"n": hi.height, "epa": float(hi["epa"].mean()),
                         "success": float(hi["success"].mean())},
            "surprising": {"n": lo.height, "epa": float(lo["epa"].mean()),
                           "success": float(lo["success"].mean())}}  # fmt: skip


# --------------------------------------------------------------------------- stages


def cost_block(table: pl.DataFrame, p_pass: np.ndarray, cfg: dict, rcfg: dict, seed: int,
               sensitivity: dict[str, np.ndarray] | None = None) -> dict[str, Any]:  # fmt: skip
    sample = cost_sample(table, p_pass, cfg, rcfg)
    log(f"cost sample: {sample.height} plays, {sample['gcode'].max() + 1} games")
    curves = cost_curves(sample, cfg, seed)
    tab, summ = team_table(sample, curves, cfg, seed + 1)
    out: dict[str, Any] = {
        "seasons": sorted(splits.seasons_in(sample)),
        "n_plays": sample.height,
        "mean_log_loss": float(sample["surprise"].mean()),
        "curves": curves,
        "teams_summary": summ,
        "team_slopes": team_slopes(sample, cfg),
        "success_regression": success_regression(sample, cfg, seed + 2),
        "first_down": first_down(table, p_pass, cfg, rcfg, seed + 3),
        "_team_table": tab,
        "_sample": sample,
    }
    if sensitivity:
        out["sensitivity"] = {}
        for name, p in sensitivity.items():
            s2 = cost_sample(table, p, cfg, rcfg)
            log(f"sensitivity {name}")
            out["sensitivity"][name] = cost_curves(s2, cfg, seed, specs=("within_team",),
                                                   outcomes=("epa",))  # fmt: skip
    return out


def team_rows(tab: pl.DataFrame) -> list[dict[str, Any]]:
    return tab.sort("season", "posteam").to_dicts()


def finish_cost(block: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in block.items() if not k.startswith("_")}
    out["teams"] = team_rows(block["_team_table"])
    fd = dict(out["first_down"])
    fd["teams"] = team_rows(fd.pop("_team_table"))
    out["first_down"] = fd
    return strip(out)


def evaluation_plan(cfg: dict) -> dict[str, Any]:
    plan = {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "test_seasons": list(splits.test_seasons()),
        "models_fitted_on": [cfg["first_train_season"], max(splits.dev_seasons()) - 1],
        "early_stopping_season": max(splits.dev_seasons()),
        "variants": cfg["variants"],
        "primary": cfg["primary"],
        "metrics": "log loss (nats per call), Brier, AUC, expected calibration error; paired "
                   "log-loss differences per 1,000 calls with a 95% interval from "
                   f"{cfg['n_boot']} resamples of test-season games",
        "comparisons": [{"candidate": a, "baseline": b, "what": w} for a, b, w in COMPARISONS],
        "cost_analysis": "repeated on 2024-2025 with the primary model's P(pass): curves by bin "
                         f"{cfg['cost']['p_edges']} of P(call) and slopes, raw, within "
                         "situation, within situation + offense-season + defense-season; "
                         "team-season predictability and cost with shrinkage; first and 10",
        "reading": "history adds predictive power if the interval of situation_history minus "
                   "situation (and nn_sequence minus nn_situation) lies below zero; a cost "
                   "of predictability exists if the within-team slope of EPA on P(call) is "
                   "below zero with an interval that excludes zero",
        "config": cfg,
    }  # fmt: skip
    plan["hash"] = hashlib.sha1(json.dumps(plan, sort_keys=True, default=str).encode()
                                ).hexdigest()[:16]  # fmt: skip
    return plan


def stage_dev(dev_table: pl.DataFrame, cfg: dict, rcfg: dict) -> tuple[dict, dict]:
    t0 = time.time()
    val, _ = validation(dev_table, cfg)
    log(f"validation done ({time.time() - t0:.0f}s)")
    variants = [cfg["primary"], "situation", "situation_history_formation"]
    fwd, fwd_info = forward_predictions(dev_table, cfg, variants)
    primary = fwd[cfg["primary"]]
    m = ~np.isnan(primary)
    y = dev_table["is_pass"].to_numpy()
    forward_scores = {v: pr.score(y[m], fwd[v][m]) for v in variants}
    cost = cost_block(dev_table, primary, cfg, rcfg, cfg["seed"] + 300,
                      sensitivity={v: fwd[v] for v in variants if v != cfg["primary"]})
    result = {
        "what": "M10 cost of predictability (scripts/m10_predictability.py)",
        "data": {"called_plays": dev_table.height,
                 "seasons": sorted(splits.seasons_in(dev_table)),
                 "games": int(dev_table["game_id"].n_unique())},  # fmt: skip
        "validation": val,
        "forward": {"seasons": [cfg["first_forward_season"], max(splits.seasons_in(dev_table))],
                    "best_iterations": fwd_info, "scores": forward_scores},  # fmt: skip
        "dev_cost": finish_cost(cost),
        "evaluation_plan": evaluation_plan(cfg),
        "test": "pending",
    }
    return result, {"cost": cost}


def stage_test(dev_table: pl.DataFrame, full: pl.DataFrame, cfg: dict, rcfg: dict,
               result: dict) -> dict:  # fmt: skip
    first = cfg["first_train_season"]
    last_dev = max(splits.seasons_in(dev_table))
    train = season_range(first, last_dev - 1)
    test_seasons = sorted(set(splits.test_seasons()) & splits.seasons_in(full))
    rows = rows_of(full, test_seasons)
    y = full["is_pass"].to_numpy()[rows]
    preds, info = {}, {}
    for i, name in enumerate(cfg["variants"]):
        preds[name], m = fit_predict(name, dev_table, train, last_dev, full, rows, cfg,
                                     cfg["seed"] + 500 + i)  # fmt: skip
        stamp = {"seasons": train + [last_dev]}
        splits.assert_can_score(stamp, full[rows], f"M10 {name} on test seasons")
        info[name] = model_info(m)
        log(f"test {name}: log loss {pr.log_loss_rows(y, preds[name]).mean():.4f}")
    res = compare_models(y, preds, full["game_id"].to_numpy()[rows],
                         full["down"].to_numpy()[rows], cfg, cfg["seed"] + 600)  # fmt: skip
    p_full = np.full(full.height, np.nan)
    p_full[rows] = preds[cfg["primary"]]
    sens = {}
    for v in ("situation", "situation_history_formation"):
        a = np.full(full.height, np.nan)
        a[rows] = preds[v]
        sens[v] = a
    cost = cost_block(full, p_full, cfg, rcfg, cfg["seed"] + 700, sensitivity=sens)
    return {
        "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "plan_hash": result["evaluation_plan"]["hash"],
        "seasons": test_seasons,
        "models": info,
        **res,
        "cost": finish_cost(cost),
    }


# --------------------------------------------------------------------------- figures


def _mid(edges: list[float]) -> np.ndarray:
    e = np.asarray(edges)
    return (e[:-1] + e[1:]) / 2


def draw_main(result: dict, path: Any) -> None:
    style.apply()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.2),
                                  gridspec_kw={"width_ratios": [1.5, 1]})  # fmt: skip
    dev = result["dev_cost"]["curves"]
    test = result["test"]["cost"]["curves"] if isinstance(result.get("test"), dict) else None
    x = _mid(dev["p_edges"])
    for call in ("run", "pass"):
        c = dev[call]["within_team_epa"]["curve"]
        v, lo, hi = (np.array([np.nan if a is None else a for a in c[k]], float)
                     for k in ("value", "low", "high"))  # fmt: skip
        ok = np.array(c["n"]) >= 200
        off = -0.008 if call == "run" else 0.008
        ax.fill_between(x[ok], lo[ok], hi[ok], color=COLORS[call], alpha=0.15, lw=0)
        b = dev[call]["within_team_epa"]["slope"]
        ax.plot(x[ok] + off, v[ok], "-o", color=COLORS[call], ms=5,
                label=f"{LABELS[call]}, 2013-2023 (95% interval); per 10 points "
                      f"{b['value'] / 10:+.3f} [{b['low'] / 10:+.3f}, {b['high'] / 10:+.3f}]")
        if test is not None:
            t = test[call]["within_team_epa"]["curve"]
            tv, tlo, thi = (np.array([np.nan if a is None else a for a in t[k]], float)
                            for k in ("value", "low", "high"))  # fmt: skip
            tok = np.array(t["n"]) >= 200
            ax.errorbar(x[tok] + 2 * off, tv[tok], yerr=[tv[tok] - tlo[tok], thi[tok] - tv[tok]],
                        fmt="o", mfc=style.SURFACE, mec=COLORS[call], ecolor=COLORS[call],
                        ms=6, lw=1, capsize=0, label=f"{LABELS[call]}, 2024-2025 test")  # fmt: skip
    ax.axhline(0, color=style.AXIS, lw=1)
    ax.set_xlabel("How expected the call was: the model's P(the call actually made)")
    ax.set_ylabel("nflfastR EPA per play")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5)
    style.title(ax, "Runs the defense could see coming gain less; passes show no clear cost",
                "EPA by how expected the call was; same situation, offense and opponent season")
    # Right: what play history adds to predicting the call.
    rows = []
    dv = {(d["candidate"], d["baseline"]): d
          for d in result["validation"]["pooled"]["differences_per_1000"]}  # fmt: skip
    tv = ({(d["candidate"], d["baseline"]): d for d in result["test"]["differences_per_1000"]}
          if test is not None else {})  # fmt: skip
    names = {("situation_history", "situation"): "Boosting: + play history",
             ("nn_sequence", "nn_situation"): "Network: + GRU over the game",
             ("nn_sequence", "situation_history"): "GRU vs hand-made history",
             ("situation_history_formation", "situation_history"):
                 "+ shotgun, no-huddle (sensitivity)"}  # fmt: skip
    for key, label in names.items():
        rows.append((label, dv.get(key), tv.get(key)))
    yy = np.arange(len(rows))[::-1]
    for yi, (_label, d, t) in zip(yy, rows, strict=True):
        for obj, dy, face, name in ((d, 0.13, style.INK, "2020-2023 validation"),
                                    (t, -0.13, style.SURFACE, "2024-2025 test")):  # fmt: skip
            if obj is None:
                continue
            ax2.errorbar(obj["value"], yi + dy, xerr=[[obj["value"] - obj["low"]],
                                                       [obj["high"] - obj["value"]]],
                         fmt="o", color=style.INK, mfc=face, ms=6, lw=1.2,
                         label=name if yi == yy[0] else None)  # fmt: skip
    ax2.axvline(0, color=style.AXIS, lw=1)
    ax2.set_yticks(yy, [r[0] for r in rows])
    ax2.set_xlabel("Change in log loss per 1,000 calls (left = predicts better)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    ax2.grid(axis="y", visible=False)
    style.title(ax2, "What the defense can learn from the drive", "Predicting run or pass "
                "before the snap; 95% intervals over games")  # fmt: skip
    fig.tight_layout(w_pad=3)
    style.save(fig, path)


def label_points(ax: plt.Axes, names: list[str], xs: np.ndarray, ys: np.ndarray) -> None:
    """Label scatter points, trying four spots around each point and taking the first
    whose (approximate) text box does not hit a label already placed."""
    w = 0.055 * (xs.max() - xs.min())
    h = 0.022 * (ys.max() - ys.min())
    spots = [((3, 2), "left", 0.1 * w, 0.1 * h), ((3, -9), "left", 0.1 * w, -1.2 * h),
             ((-4, 2), "right", -1.1 * w, 0.1 * h), ((-4, -9), "right", -1.1 * w, -1.2 * h)]
    boxes: list[tuple[float, float]] = []
    for name, a, b in sorted(zip(names, xs, ys, strict=True), key=lambda r: -r[2]):
        free = [sp for sp in spots
                if not any(abs(a + sp[2] - bx) < w and abs(b + sp[3] - by) < h
                           for bx, by in boxes)]  # fmt: skip
        off, ha, dx, dy = (free or spots)[0]
        boxes.append((a + dx, b + dy))
        ax.annotate(name, (a, b), xytext=off, ha=ha, textcoords="offset points",
                    fontsize=7.5, color=style.INK_2)


def draw_team(result: dict, path: Any) -> None:
    style.apply()
    have_test = isinstance(result.get("test"), dict)
    block = result["test"]["cost"] if have_test else result["dev_cost"]
    teams = pl.DataFrame(block["teams"])
    last = int(teams["season"].max())
    t = teams.filter(pl.col("season") == last).sort("cost_per_game_shrunk")
    fd = pl.DataFrame(block["first_down"]["teams"]).filter(pl.col("season") == last)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 8.2))
    y = np.arange(t.height)
    v = t["cost_per_game"].to_numpy()
    ax.errorbar(v, y, xerr=[v - t["cost_low"].to_numpy(), t["cost_high"].to_numpy() - v],
                fmt="o", color=style.MUTED, mfc=style.SURFACE, ms=4, lw=0.9,
                label="Observed (95% interval)")  # fmt: skip
    ax.plot(t["cost_per_game_shrunk"].to_numpy(), y, "o", color=style.INK, ms=5,
            label="Shrunk toward the league")  # fmt: skip
    ax.axvline(0, color=style.AXIS, lw=1)
    ax.set_yticks(y, t["posteam"].to_list(), fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("EPA per game lost to being more predictable than average (right = costs more)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=9)
    ns = block["teams_summary"]["excess"]["noise_share"]
    style.title(ax, f"{last} offenses: the cost of their predictability",
                f"Neutral downs 1-3, league-wide slopes; {100 * ns:.0f}% of the spread in "
                "team predictability is sampling noise")
    xs = (1 - fd["pass_rate"].to_numpy()) * 100
    ys = fd["mean_p_run"].to_numpy() * 100
    ax2.scatter(xs, ys, s=36, color=style.ORANGE, edgecolor=style.SURFACE, lw=0.6,
                label="One offense, first-and-10 runs")  # fmt: skip
    label_points(ax2, fd["posteam"].to_list(), xs, ys)
    ax2.set_xlabel("Run share on first and 10 (neutral game script, %)")
    ax2.set_ylabel("Average P(run) the model gave its first-and-10 runs (%)")
    ax2.legend(loc="lower right", fontsize=9)
    slope = block["first_down"]["run_slope_per_10_points"]
    style.title(ax2, "First-and-10 runs: who telegraphs them",
                f"EPA of a first-down run per 10 points of P(run): {slope['value']:+.3f} "
                f"[{slope['low']:+.3f}, {slope['high']:+.3f}]")  # fmt: skip
    fig.tight_layout(w_pad=3)
    style.save(fig, path)


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--dev-only", action="store_true", help="stop after writing the plan")
    ap.add_argument("--rescore-reason", default=None)
    args = ap.parse_args()
    cfg, rcfg = pr.load_config(), rp.load_config()
    if args.figures_only:
        res = read_json(OUT)
        draw_main(res, FIG)
        draw_team(res, FIG_TEAM)
        return
    if OUT.exists() and isinstance(read_json(OUT).get("test"), dict) and not args.rescore_reason:
        raise SystemExit(f"{OUT} already holds the test-season scoring; the test seasons are "
                         "scored once. Use --figures-only to redraw.")  # fmt: skip
    import torch

    torch.set_num_threads(cfg["threads"])
    dev = splits.load_dev_plays()
    splits.assert_no_test_seasons(dev, "M10 call models")
    dev_table = load_table(dev, cfg, rcfg)
    log(f"development table: {dev_table.height} called plays")
    result, _ = stage_dev(dev_table, cfg, rcfg)
    write_json(result, OUT)  # the plan is on disk before any test row is read
    log(f"evaluation plan written to {OUT}")
    draw_main(result, FIG)
    draw_team(result, FIG_TEAM)
    if args.dev_only:
        return
    purpose = TEST_PURPOSE if not args.rescore_reason else (
        f"{TEST_PURPOSE}; RESCORE: {args.rescore_reason}")  # fmt: skip
    test = splits.load_test_plays(purpose)
    full = load_table(pl.concat([dev, test], how="diagonal_relaxed"), cfg, rcfg)
    # No look-ahead: adding later seasons must not change any development-season feature.
    names = pr.feature_names(cfg, ["situation", "tendency", "history"])
    a = dev_table.select("game_id", "play_id", *names)
    b = full.filter(pl.col("season").is_in(sorted(splits.seasons_in(dev)))).select(
        "game_id", "play_id", *names)  # fmt: skip
    j = a.join(b, on=["game_id", "play_id"], suffix="_full")
    for n in names:
        d = (j[n] - j[f"{n}_full"]).abs().max()
        if d is not None and d > 1e-9:
            raise RuntimeError(f"feature {n} changed when later seasons were added")
    result["test"] = stage_test(dev_table, full, cfg, rcfg, result)
    write_json(result, OUT)
    draw_main(result, FIG)
    draw_team(result, FIG_TEAM)
    log("done")


if __name__ == "__main__":
    main()
