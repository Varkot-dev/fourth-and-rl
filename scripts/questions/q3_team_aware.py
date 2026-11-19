"""Question 3: "Does the model know our defense is terrible?" How much does the call move?

The one-step engine (M2) adjusts for team strength only through the pregame spread. This
script adds the M8 team ratings (scripts/m8_team_strength.py must have run first) through
`team_strength.TeamAwareEngine`:

  * conversion: the team model's change in conversion log-odds;
  * giving the ball back (and keeping it): every next state in which a team starts with
    the ball gets the change in that team's next-drive touchdown and field-goal chances
    (its offense rating against the other side's defense rating), valued through the
    win-probability model.

Both engines get the same spread. For hypothetical pairings the spread is the one the
ratings imply (a regression of the spread on the ratings), so a "strong offense" pairing is
also a favourite, as it would be in a real game.

Part A: decision charts for pairings of strong and weak units and for the Lions at the end
of 2023, 2024 and 2025 against an average opponent; where the call flips relative to the
spread-only engine, by how many win-probability points, and in what share of resamples the
flip survives (the four effect models refitted on resampled development games; Lions
ratings drawn from the filter's posterior).
Part B (test seasons, scored once): every real 2024 to 2025 fourth down with the
team-aware and spread-only recommendations side by side.

Outputs:
  results/questions/q3_team_aware.json
  assets/questions/q3_team_aware.png

Run:  python scripts/questions/q3_team_aware.py            (about 11 minutes)
      python scripts/questions/q3_team_aware.py --dry-run  (2022 to 2023 as a pseudo test,
                                                            development data only)
"""

from __future__ import annotations

import argparse
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

from fourth_and_rl import decision, splits, style  # noqa: E402
from fourth_and_rl import team_strength as ts  # noqa: E402
from fourth_and_rl.io import read_json, write_json  # noqa: E402
from fourth_and_rl.paths import ASSETS, DATA, RESULTS  # noqa: E402

RESULT = RESULTS / "questions" / "q3_team_aware.json"
FIGURE = ASSETS / "questions" / "q3_team_aware.png"
M8_RESULT = RESULTS / "m8_team_strength.json"
BUNDLE = DATA / "models" / "team_strength.joblib"
TEST_PURPOSE = (
    "Q3 team-aware decisions: spread-only and team-aware recommendations for every real "
    "2024-2025 fourth down, scored once (scripts/questions/q3_team_aware.py)"
)


# --------------------------------------------------------------------------- setup


def ratings_through(dev: pl.DataFrame, later: pl.DataFrame | None, eff: ts.TeamEffects,
                    cfg: dict) -> dict[str, Any]:  # fmt: skip
    """Pre-game and end-of-season ratings from the first rating season through the last
    season of `later` (frozen constants; bases fitted on development seasons only)."""
    seasons = sorted(splits.seasons_in(dev))
    start = seasons[0] + 1
    plays = dev if later is None else pl.concat([dev, later], how="diagonal_relaxed")
    last = max(splits.seasons_in(plays))
    bases = ts.fit_bases(dev, range(start, last + 1), cfg)
    conv, drives = ts.prepare_samples(plays, bases, cfg, start)
    sched = ts.Schedule.from_frame(ts.schedule(plays.filter(pl.col("season") >= start)))
    ratings, season_end, _ = ts.compute_ratings(conv, drives, sched, eff.filter_params, start)
    return {"ratings": ratings, "season_end": season_end, "conv": conv, "drives": drives}


def replicate_effects(eff: ts.TeamEffects, conv: pl.DataFrame, drives: pl.DataFrame,
                      n_boot: int, seed: int) -> list[ts.TeamEffects]:  # fmt: skip
    """Refit the four effect models on resampled development games (one draw of games
    shared by the conversion and drive samples)."""
    games = np.unique(np.concatenate([conv["game_id"].to_numpy(), drives["game_id"].to_numpy()]))
    pos = {g: i for i, g in enumerate(games)}
    ic = np.array([pos[g] for g in conv["game_id"].to_list()])
    idr = np.array([pos[g] for g in drives["game_id"].to_list()])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, games.size, games.size), minlength=games.size)
        wc, wd = counts[ic].astype(float), counts[idr].astype(float)
        cs, ct = eff.conv_spread, eff.conv_team
        ds, dt = eff.drive_spread, eff.drive_team
        out.append(ts.TeamEffects(
            ts.ConversionModel(cs.ratings, l2=cs.l2).fit(conv, wc),
            ts.ConversionModel(ct.ratings, l2=ct.l2).fit(conv, wc),
            ts.DriveModel(ds.ratings, knots=ds.knots, l2=ds.l2).fit(drives, wd, init=ds),
            ts.DriveModel(dt.ratings, knots=dt.knots, l2=dt.l2).fit(drives, wd, init=dt),
            eff.filter_params, eff.drive_consts,
        ))  # fmt: skip
    return out


def p_at_spread(engine: Any, yl: np.ndarray, togo: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """P(convert) at the offense's spread (the spread-only engine), distance capped as the
    engine caps it."""
    d = np.minimum(np.minimum(togo, yl), 40)
    return np.asarray(engine.components.p_convert(yl, d, spread=spread), dtype=float)


def tables_for(eff: ts.TeamEffects, values: dict, sens: np.ndarray, p_sp: np.ndarray,
               ctx: ts.TeamContext) -> dict[str, np.ndarray]:  # fmt: skip
    return ts.team_aware_table(values, sens, p_sp, eff.dlogit(ctx), eff.drive_shift(ctx))


def draw_ctx(ctx: ts.TeamContext, sd_us: dict, sd_them: dict, rng: np.random.Generator
             ) -> ts.TeamContext:  # fmt: skip
    """One draw of the ratings from the filter's posterior (independent normals)."""
    us = {k: v + rng.normal(0, 1, v.size) * sd_us[k] for k, v in ctx.us.items()}
    them = {k: v + rng.normal(0, 1, v.size) * sd_them[k] for k, v in ctx.them.items()}
    return ts.TeamContext(ctx.spread, us, them)


def boot_best(
    reps: list[ts.TeamEffects],
    values: dict,
    sens: np.ndarray,
    p_sp: np.ndarray,
    ctx: ts.TeamContext,
    sd: tuple[dict, dict] | None,
    seed: int,
) -> np.ndarray:
    """Team-aware best action in each replicate (n_boot x n)."""
    rng = np.random.default_rng(seed)
    out = []
    for eff in reps:
        c = ctx if sd is None else draw_ctx(ctx, sd[0], sd[1], rng)
        t = tables_for(eff, values, sens, p_sp, c)
        out.append(decision.summarize_actions(t["team_table"])[0])
    return np.array(out)


# --------------------------------------------------------------------------- part A


def unit(groups: dict, name: str) -> dict[str, float]:
    return groups[name]


def pairing_units(cfg: dict, groups: dict, season_end: pl.DataFrame) -> list[dict[str, Any]]:
    out = []
    for p in cfg["decision"]["pairings"]:
        out.append({"key": p["key"], "label": p["label"],
                    "us_off": unit(groups, p["our_offense"]),
                    "us_def": unit(groups, p["our_defense"]),
                    "them_off": unit(groups, p["their_offense"]),
                    "them_def": unit(groups, p["their_defense"]), "sd": None})  # fmt: skip
    team = cfg["decision"]["lions_team"]
    for s in cfg["decision"]["lions_seasons"]:
        rows = season_end.filter((pl.col("season") == s) & (pl.col("team") == team))
        if rows.height == 0:
            continue
        r = rows.row(0, named=True)
        us_off = {"conv_off": r["conv_off"], "drive_off": r["drive_off"]}
        us_def = {"conv_def": r["conv_def"], "drive_def": r["drive_def"]}
        sd_us = {
            "off_conv": 100 * r["conv_off_sd"],
            "off_drive": r["drive_off_sd"],
            "def_conv": 0.0,
            "def_drive": 0.0,
        }
        sd_them = {"off_conv": 0.0, "off_drive": 0.0, "def_conv": 100 * r["conv_def_sd"],
                   "def_drive": r["drive_def_sd"]}  # fmt: skip
        out.append({"key": f"lions_{s}", "label": f"Detroit at the end of {s} vs average",
                    "us_off": us_off, "us_def": us_def, "them_off": groups["average"],
                    "them_def": groups["average"], "sd": (sd_us, sd_them),
                    "ratings": r})  # fmt: skip
    return out


def part_a(engine: ts.TeamAwareEngine, eff: ts.TeamEffects, reps: list[ts.TeamEffects],
           spread_fit: dict, pairs: list[dict], cfg: dict) -> dict[str, Any]:  # fmt: skip
    ch = cfg["decision"]["chart"]
    yls = np.arange(ch["yardlines"][0], ch["yardlines"][1] + 1)
    ds = np.arange(ch["distances"][0], ch["distances"][1] + 1)
    survive = float(cfg["decision"]["survive_share"])
    out: dict[str, Any] = {}
    for sit in ch["situations"]:
        res_sit = {}
        for k, p in enumerate(pairs):
            spread = ts.implied_spread(spread_fit, p["us_off"], p["us_def"], p["them_off"],
                                       p["them_def"])  # fmt: skip
            states = decision.chart_states(
                yls, ds, {**ch["defaults"], **sit["state"], "posteam_spread": round(spread, 2)}
            )
            n = states.height
            values, sens = engine.sensitivities(states)
            yl, togo = states["yardline_100"].to_numpy(), states["ydstogo"].to_numpy()
            p_sp = p_at_spread(engine, yl, togo, np.full(n, round(spread, 2)))
            ctx = ts.TeamContext.constant(
                n, round(spread, 2), ts.unit_features(p["us_off"], p["them_def"]),
                ts.unit_features(p["them_off"], p["us_def"]))  # fmt: skip
            t = tables_for(eff, values, sens, p_sp, ctx)
            fl = ts.flips(t["spread_table"], t["team_table"])
            bb = boot_best(reps, values, sens, p_sp, ctx, p["sd"], int(cfg["seed"]) + k)
            same_as_team = (bb == fl["best_team"][None, :]).mean(axis=0)
            flips_in_boot = (bb != fl["best_spread"][None, :]).mean(axis=0)
            ch_idx = np.nonzero(fl["changed"])[0]
            go_shift = 100 * (fl["go_gain_team"] - fl["go_gain_spread"])
            flipped = [{
                "yardline_100": int(yl[i]), "ydstogo": int(togo[i]),
                "spread_only": decision.ACTIONS[fl["best_spread"][i]],
                "team_aware": decision.ACTIONS[fl["best_team"][i]],
                "spread_only_margin_points": float(100 * fl["margin_spread"][i]),
                "team_margin_points": float(100 * fl["margin_team"][i]),
                "go_gain_shift_points": float(go_shift[i]),
                "share_of_resamples_flipped": float(same_as_team[i]),
            } for i in ch_idx]  # fmt: skip
            valid = fl["best_spread"] >= 0
            res_sit[p["key"]] = {
                "label": p["label"], "implied_spread": float(spread),
                "features_us": ts.unit_features(p["us_off"], p["them_def"]),
                "features_them": ts.unit_features(p["them_off"], p["us_def"]),
                "n_cells": int(valid.sum()),
                "n_flipped": int(fl["changed"].sum()),
                "n_flipped_surviving": int(sum(f["share_of_resamples_flipped"] >= survive
                                               for f in flipped)),
                "n_cells_flipping_in_25pct_of_resamples": int((flips_in_boot[valid] >= 0.25
                                                               ).sum()),
                "go_gain_shift_points": {
                    "mean": float(np.nanmean(go_shift[valid])),
                    "min": float(np.nanmin(go_shift[valid])),
                    "max": float(np.nanmax(go_shift[valid])),
                    "fourth_and_1_to_3_mean": float(np.nanmean(go_shift[valid & (togo <= 3)])),
                },
                "conversion_shift_points_mean": float(100 * np.nanmean(
                    (t["p_team"] - t["p_spread"])[valid])),
                "go_share_spread_only": float(np.mean(fl["best_spread"][valid] == 0)),
                "go_share_team": float(np.mean(fl["best_team"][valid] == 0)),
                "flipped_cells": flipped,
                "_grid": {"yl": yl, "togo": togo, "best": fl["best_team"],
                          "changed": fl["changed"], "survive": same_as_team},
            }  # fmt: skip
            print(
                f"  {sit['key']} {p['key']}: spread {spread:+.1f}, flips "
                f"{res_sit[p['key']]['n_flipped']}, surviving "
                f"{res_sit[p['key']]['n_flipped_surviving']}"
            )
        out[sit["key"]] = {"label": sit["label"], "pairings": res_sit}
    return out


# --------------------------------------------------------------------------- part B


def part_b(engine: ts.TeamAwareEngine, eff: ts.TeamEffects, reps: list[ts.TeamEffects],
           test: pl.DataFrame, ratings: pl.DataFrame, cfg: dict, dry_run: bool = False
           ) -> dict[str, Any]:  # fmt: skip
    fd = test.filter(pl.col("fourth_down_decision").is_not_null())
    fd = ts.attach_ratings(fd, ratings)
    if not dry_run:  # a dry run's pseudo-test seasons are inside the engine's fit by design
        engine.assert_can_score(fd, "Q3 team-aware decisions on real fourth downs")
    splits.assert_can_score(eff.stamp, fd, "Q3 effect models on real fourth downs")
    values, sens = engine.sensitivities(fd)
    ok = (
        ~np.isnan(values["go_wp"])
        & fd["us_drive_off"].is_not_null().to_numpy()
        & fd["them_drive_off"].is_not_null().to_numpy()
    )
    fd, sens = fd.filter(pl.Series(ok)), sens[ok]
    values = {k: v[ok] for k, v in values.items()}
    yl, togo = fd["yardline_100"].to_numpy(), fd["ydstogo"].to_numpy()
    spread = fd["posteam_spread"].cast(pl.Float64).fill_null(0.0).to_numpy()
    p_sp = p_at_spread(engine, yl, togo, spread)
    ctx = ts.TeamContext.from_frame(fd)
    t = tables_for(eff, values, sens, p_sp, ctx)
    fl = ts.flips(t["spread_table"], t["team_table"])
    valid = (fl["best_spread"] >= 0) & np.isfinite(fl["go_gain_spread"])
    sd_us = {
        "off_conv": 100 * fd["us_conv_off_sd"].to_numpy(),
        "off_drive": fd["us_drive_off_sd"].to_numpy(),
        "def_conv": 100 * fd["them_conv_def_sd"].to_numpy(),
        "def_drive": fd["them_drive_def_sd"].to_numpy(),
    }
    sd_them = {"off_conv": 100 * fd["them_conv_off_sd"].to_numpy(),
               "off_drive": fd["them_drive_off_sd"].to_numpy(),
               "def_conv": 100 * fd["us_conv_def_sd"].to_numpy(),
               "def_drive": fd["us_drive_def_sd"].to_numpy()}  # fmt: skip
    bb = boot_best(reps, values, sens, p_sp, ctx, (sd_us, sd_them), int(cfg["seed"]) + 99)
    share_same = (bb == fl["best_team"][None, :]).mean(axis=0)
    survive = float(cfg["decision"]["survive_share"])
    changed = fl["changed"] & valid
    shift = 100 * (fl["go_gain_team"] - fl["go_gain_spread"])
    acts = np.array(decision.ACTIONS)
    coach = fd["fourth_down_decision"].to_numpy()
    teams = fd["posteam"].to_numpy()

    def rec(best: np.ndarray) -> np.ndarray:
        return np.where(best >= 0, acts[np.maximum(best, 0)], "")

    rs, rt = rec(fl["best_spread"]), rec(fl["best_team"])
    by_team = []
    for team in sorted(set(teams.tolist())):
        m = (teams == team) & valid
        by_team.append({
            "team": team, "fourth_downs": int(m.sum()), "calls_changed": int((changed & m).sum()),
            "mean_go_gain_shift_points": float(np.mean(shift[m])),
            "mean_abs_go_gain_shift_points": float(np.mean(np.abs(shift[m]))),
            "coach_agrees_spread_only": float(np.mean(coach[m] == rs[m])),
            "coach_agrees_team_aware": float(np.mean(coach[m] == rt[m])),
        })  # fmt: skip
    by_team.sort(key=lambda r: -r["mean_abs_go_gain_shift_points"])

    def lions(team: str) -> dict[str, Any]:
        m = (teams == team) & valid
        went = m & (coach == "go")
        return {
            "fourth_downs": int(m.sum()), "went_for_it": int(went.sum()),
            "goes_recommended_by_spread_only": int((went & (rs == "go")).sum()),
            "goes_recommended_by_team_aware": int((went & (rt == "go")).sum()),
            "mean_go_gain_on_their_goes_spread_only_points": float(
                100 * np.mean(fl["go_gain_spread"][went])),
            "mean_go_gain_on_their_goes_team_aware_points": float(
                100 * np.mean(fl["go_gain_team"][went])),
            "calls_changed": int((changed & m).sum()),
            "changed_plays": [
                {"game_id": fd["game_id"][int(i)], "qtr": int(fd["qtr"][int(i)]),
                 "ydstogo": int(togo[i]), "yardline_100": int(yl[i]),
                 "coach": str(coach[i]), "spread_only": str(rs[i]), "team_aware": str(rt[i]),
                 "go_gain_shift_points": float(shift[i]),
                 "share_of_resamples": float(share_same[i])}
                for i in np.nonzero(changed & m)[0]
            ],
            "by_season": {str(s): {
                "mean_go_gain_shift_points": float(np.mean(shift[m & (fd["season"].to_numpy()
                                                                      == s)]))}
                for s in sorted(set(fd["season"].to_list()))},
        }  # fmt: skip

    trans = {}
    for a in decision.ACTIONS:
        for b in decision.ACTIONS:
            n = int((changed & (rs == a) & (rt == b)).sum())
            if n:
                trans[f"{a}_to_{b}"] = n
    return {
        "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seasons": sorted(splits.seasons_in(fd)),
        "fourth_downs_valued": int(valid.sum()),
        "calls_changed": int(changed.sum()),
        "share_changed": float(changed.sum() / valid.sum()),
        "calls_changed_surviving": int((changed & (share_same >= survive)).sum()),
        "changes_by_direction": trans,
        "go_share_spread_only": float(np.mean(rs[valid] == "go")),
        "go_share_team_aware": float(np.mean(rt[valid] == "go")),
        "coach_agrees_spread_only": float(np.mean(coach[valid] == rs[valid])),
        "coach_agrees_team_aware": float(np.mean(coach[valid] == rt[valid])),
        "go_gain_shift_points": {
            "mean_abs": float(np.mean(np.abs(shift[valid]))),
            "p90_abs": float(np.quantile(np.abs(shift[valid]), 0.9)),
            "max_abs": float(np.max(np.abs(shift[valid]))),
        },
        "spread_only_margin_of_changed_calls_points": {
            "median": float(np.median(100 * fl["margin_spread"][changed])) if changed.any()
            else None,
            "max": float(np.max(100 * fl["margin_spread"][changed])) if changed.any() else None,
        },
        "by_team": by_team,
        "lions": lions(cfg["decision"]["lions_team"]),
        "_by_team_shift": {r["team"]: r["mean_go_gain_shift_points"] for r in by_team},
    }  # fmt: skip


# --------------------------------------------------------------------------- figure


def draw(res: dict, grids: dict, path: Path) -> Path:
    style.apply()
    sit = res["part_a"]["q1_tied"]
    keys = list(sit["pairings"])
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.6))
    axes = axes.ravel()
    for ax, key in zip(axes, keys, strict=False):
        p = sit["pairings"][key]
        g = grids[key]
        best = decision.to_grid(g["yl"], g["togo"], g["best"].astype(float), n_distances=10)
        decision.draw_chart(ax, best)
        ch = np.asarray(g["changed"])
        for i in np.nonzero(ch)[0]:
            strong = g["survive"][i] >= 0.83
            ax.scatter(100 - g["yl"][i], g["togo"][i], marker="o", s=26, zorder=5,
                       facecolor=style.INK if strong else "none", edgecolor=style.INK,
                       linewidth=1.1)  # fmt: skip
        ax.set_yticks([1, 3, 5, 7, 9])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(p["label"], fontsize=10.5, pad=20, loc="left")
        ax.text(
            0,
            1.02,
            f"spread {p['implied_spread']:+.1f}; {p['n_flipped']} of "
            f"{p['n_cells']} cells flip, {p['n_flipped_surviving']} survive",
            transform=ax.transAxes,
            fontsize=9,
            color=style.INK_2,
            va="bottom",
        )
    for ax in axes[len(keys) : -1]:
        ax.axis("off")
    bx = axes[-1]
    pb = res.get("part_b")
    if pb:
        shift = pb["_by_team_shift"]
        top = sorted(shift, key=lambda t: shift[t])
        top = top[:5] + top[-5:]
        vals = [shift[t] for t in top]
        colors = [style.ACTION_COLORS["go"] if v > 0 else style.ACTION_COLORS["punt"] for v in vals]
        bx.barh(np.arange(len(top)), vals, color=colors)
        bx.set_yticks(np.arange(len(top)))
        bx.set_yticklabels(top, fontsize=9)
        bx.axvline(0, color=style.INK_2, linewidth=0.8)
        bx.set_xlabel("Mean change in go gain (WP points)", fontsize=9)
        bx.set_title("Real 2024-25 fourth downs", fontsize=10.5, pad=20, loc="left")
        bx.text(
            0,
            1.02,
            f"{pb['calls_changed']} of {pb['fourth_downs_valued']} calls change",
            transform=bx.transAxes,
            fontsize=9,
            color=style.INK_2,
            va="bottom",
        )
        bx.grid(axis="y", visible=False)
    else:
        bx.axis("off")
    handles = decision.chart_legend_handles(False)
    from matplotlib.lines import Line2D

    handles += [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=style.INK,
               markeredgecolor=style.INK, label="Call flips; survives 83% of resamples"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
               markeredgecolor=style.INK, label="Call flips; does not survive"),
    ]  # fmt: skip
    fig.suptitle("Does the model know our defense is terrible? Team-aware engine vs "
                 "spread-only engine", x=0.01, y=1.0, ha="left", fontsize=14,
                 fontweight="bold")  # fmt: skip
    fig.text(0.01, 0.955, "First quarter, tied, 3 timeouts each. Colors: team-aware "
             "recommendation. Circles: cells where adding the M8 team ratings changes the "
             "spread-only call.", fontsize=10.5, color=style.INK_2, va="top")  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.01))  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    return style.save(fig, path)


# --------------------------------------------------------------------------- main


def strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--rescore-reason", default=None)
    args = ap.parse_args()
    cfg = ts.load_config()
    n_boot = args.n_boot or int(cfg["decision"]["n_boot"])
    result_path, figure_path, bundle_path, m8_path = RESULT, FIGURE, BUNDLE, M8_RESULT
    if args.dry_run:
        scratch = Path(os.environ.get("M8_SCRATCH", DATA / "scratch_m8"))
        result_path = scratch / "q3_team_aware_dryrun.json"
        figure_path = scratch / "q3_team_aware_dryrun.png"
        bundle_path = scratch / "team_strength_dryrun.joblib"
        m8_path = scratch / "m8_team_strength_dryrun.json"
    m8 = read_json(m8_path)
    if not isinstance(m8.get("test"), dict):
        raise SystemExit("run scripts/m8_team_strength.py first (its test stage too)")
    if (not args.dry_run and result_path.exists()
            and "part_b" in read_json(result_path) and not args.rescore_reason):  # fmt: skip
        raise SystemExit(f"{result_path} already holds the scored test seasons")
    bundle = joblib.load(bundle_path)
    if bundle["plan_hash"] != m8["evaluation_plan"]["hash"]:
        raise SystemExit("the saved M8 models do not match the M8 results file")
    eff: ts.TeamEffects = bundle["effects"]

    dev_all = splits.load_dev_plays()
    if args.dry_run:
        dev = dev_all.filter(pl.col("season") < 2022)
        test = dev_all.filter(pl.col("season").is_in([2022, 2023]))
    else:
        dev, test = dev_all, None
    dev_r = ratings_through(dev, None, eff, cfg)
    first = int(cfg["models"]["first_fit_season"])
    conv_fit = ts.attach_ratings(dev_r["conv"], dev_r["ratings"]).filter(pl.col("season") >= first)
    drv_fit = ts.attach_ratings(dev_r["drives"], dev_r["ratings"]).filter(pl.col("season") >= first)
    print(f"refitting the effect models on {n_boot} resamples of games")
    reps = replicate_effects(eff, conv_fit, drv_fit, n_boot, int(cfg["seed"]))

    base = decision.load_engine()
    engine = ts.TeamAwareEngine(
        base.wp_model, base.components, eff.drive_consts, decision.load_config().get("engine"),
        min_half_seconds=cfg["decision"]["drive_min_half_seconds"],
        state_chunk_rows=cfg["decision"]["state_chunk_rows"],
    )  # fmt: skip
    if test is None:
        purpose = (
            TEST_PURPOSE
            if not args.rescore_reason
            else (f"{TEST_PURPOSE}; RESCORE: {args.rescore_reason}")
        )
        test = splits.load_test_plays(purpose)
    all_r = ratings_through(dev, test, eff, cfg)
    pairs = pairing_units(cfg, bundle["groups"], all_r["season_end"])
    print("part A: pairings")
    a = part_a(engine, eff, reps, bundle["spread_fit"], pairs, cfg)
    print("part B: real fourth downs")
    b = part_b(engine, eff, reps, test, all_r["ratings"], cfg, args.dry_run)
    res = {
        "what": "Question 3: how much team ratings beyond the spread move the one-step "
                "fourth-down call (scripts/questions/q3_team_aware.py)",
        "method": {
            "engine": "M2 one-step engine; win-probability version 2; development component "
                      "bundle; spread passed through to both engines",
            "conversion": "logit P(convert) + (team model - spread-only model) linear predictor",
            "drive": "next-drive touchdown and field-goal chances shifted by (team - spread-only "
                     "drive model), valued through the win-probability model, where more than "
                     f"{cfg['decision']['drive_min_half_seconds']} s are left in the half",
            "n_boot": n_boot,
            "survive_share": cfg["decision"]["survive_share"],
            "bootstrap": "effect models refitted on resampled development games; the base "
                         "engine is held fixed (its own uncertainty is M3's); Detroit and real "
                         "plays also draw ratings from the filter posterior",
            "primary_variants": m8["primary"],
        },
        "lions_ratings_season_end": [
            {k: v for k, v in p["ratings"].items()} for p in pairs if "ratings" in p
        ],
        "part_a": a,
        "part_b": b,
    }  # fmt: skip
    write_json(strip(res), result_path)
    grids = {k: v["_grid"] for k, v in a["q1_tied"]["pairings"].items()}
    draw(res, grids, figure_path)
    print(f"calls changed: {b['calls_changed']} of {b['fourth_downs_valued']}")


if __name__ == "__main__":
    main()
