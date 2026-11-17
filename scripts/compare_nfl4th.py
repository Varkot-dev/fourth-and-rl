"""M2: compare the one-step engine with nfl4th on every real fourth down of 2024-2025.

nfl4th's numbers come from results/reference/ (written by scripts/r/nfl4th_reference.R,
nfl4th 1.0.7). They are nfl4th's, are always labelled as such, and are never averaged or
otherwise blended with ours. The one place they are combined with our pieces is the
"swap" diagnostic, which asks how much of the disagreement is explained by nfl4th's
conversion and field-goal probabilities; it is a diagnostic, not a result.

Outputs:
  results/m2_nfl4th_comparison.json
  assets/m2_nfl4th_comparison.png

Run:  python scripts/compare_nfl4th.py     (under a minute)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from fourth_and_rl import data, decision, style, wp
from fourth_and_rl.io import write_json
from fourth_and_rl.paths import ASSETS, RESULTS, ROOT


def f(x: pl.Series) -> np.ndarray:
    return x.cast(pl.Float64).fill_null(np.nan).to_numpy()


def load_joined(plays: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    ref = pl.read_csv(ROOT / cfg["compare"]["reference_real"], null_values="NA")
    ours = plays.with_columns(pl.col("play_id").cast(pl.Int64))
    drop = [c for c in ref.columns if c in ours.columns and c not in ("game_id", "play_id")]
    return ref.drop(drop).join(ours, on=["game_id", "play_id"], how="left", validate="1:1")


def tables(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Our and nfl4th's action tables and what follows from them."""
    ours = np.column_stack([f(df["go_wp"]), f(df["fg_wp"]), f(df["punt_wp"])])
    theirs = np.column_stack([f(df["nfl4th_go_wp"]), f(df["nfl4th_fg_wp"]),
                              f(df["nfl4th_punt_wp"])])  # fmt: skip
    best_o, _, _, gain_o = decision.summarize_actions(ours)
    best_t, _, _, gain_t = decision.summarize_actions(theirs)
    return {"ours": ours, "theirs": theirs, "best_o": best_o, "best_t": best_t,
            "gain_o": gain_o, "gain_t": gain_t}  # fmt: skip


def agreement(t: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "same_action": float(np.mean(t["best_o"][mask] == t["best_t"][mask])),
        "same_go_or_kick": float(np.mean((t["gain_o"][mask] > 0) == (t["gain_t"][mask] > 0))),
        "ours_recommends_go": float(np.mean(t["best_o"][mask] == 0)),
        "nfl4th_recommends_go": float(np.mean(t["best_t"][mask] == 0)),
        "mean_go_gain_ours_points": float(100 * np.mean(t["gain_o"][mask])),
        "mean_go_gain_nfl4th_points": float(100 * np.mean(t["gain_t"][mask])),
    }


def by_groups(df: pl.DataFrame, t: dict, expr: pl.Expr, order: list[str]) -> dict:
    labels = df.select(expr.alias("g"))["g"].to_numpy()
    return {g: agreement(t, labels == g) for g in order}


def cut(col: str, edges: list[float], labels: list[str]) -> pl.Expr:
    e = pl.when(pl.col(col) < edges[0]).then(pl.lit(labels[0]))
    for edge, label in zip(edges[1:], labels[1:-1], strict=True):
        e = e.when(pl.col(col) < edge).then(pl.lit(label))
    return e.otherwise(pl.lit(labels[-1]))


def swap_diagnostic(df: pl.DataFrame, t: dict) -> dict:
    """Recompute our go and field-goal values with nfl4th's probabilities in place of
    ours (keeping our win probabilities for each outcome) and see how agreement moves."""
    p4, k4 = f(df["nfl4th_first_down_prob"]), f(df["nfl4th_fg_make_prob"])
    go_swap = p4 * f(df["wp_if_convert"]) + (1 - p4) * f(df["wp_if_fail"])
    make, miss = f(df["wp_if_fg_make"]), f(df["wp_if_fg_miss"])
    fg_swap = np.where(np.isnan(make), t["ours"][:, 1], k4 * make + (1 - k4) * miss)
    out = {}
    for name, table in (
        ("as_is", t["ours"]),
        ("with_nfl4th_conversion_probability",
         np.column_stack([go_swap, t["ours"][:, 1], t["ours"][:, 2]])),
        ("with_nfl4th_conversion_and_field_goal_probability",
         np.column_stack([go_swap, fg_swap, t["ours"][:, 2]])),
    ):  # fmt: skip
        best, _, _, gain = decision.summarize_actions(table)
        out[name] = {
            "same_action": float(np.mean(best == t["best_t"])),
            "correlation_of_go_gain": float(np.corrcoef(gain, t["gain_t"])[0, 1]),
            "share_recommending_go": float(np.mean(best == 0)),
            "mean_go_gain_points": float(100 * np.mean(gain)),
        }
    return out


def showcase(df: pl.DataFrame, t: dict, cfg: dict) -> list[dict]:
    """Ten high-leverage plays, chosen by a fixed rule (no hand picking).

    Leverage = the larger of the two models' |go gain|. Fourth quarter or later of
    one-score games with a real decision, at most one play per game: the top 4 from
    the playoffs, then the top 3 regular-season plays where the two models agree and
    the top 3 where they disagree.
    """
    d = df.with_columns(
        pl.Series("gain_t", t["gain_t"]), pl.Series("gain_o", t["gain_o"]),
        pl.Series("rec_o", [decision.ACTIONS[i] for i in t["best_o"]]),
        pl.Series("rec_t", [decision.ACTIONS[i] for i in t["best_t"]]),
    ).with_columns(
        pl.max_horizontal(pl.col("gain_t").abs(), pl.col("gain_o").abs()).alias("leverage")
    ).filter(
        (pl.col("qtr") >= cfg["compare"]["showcase_min_qtr"])
        & (pl.col("score_differential").abs() <= 8)
        & pl.col("fourth_down_decision").is_not_null()
    ).sort("leverage", descending=True)  # fmt: skip
    d = d.unique("game_id", keep="first", maintain_order=True)
    post = d.filter(pl.col("season_type") == "POST").head(4)
    reg = d.filter(pl.col("season_type") == "REG")
    picks = pl.concat([
        post,
        reg.filter(pl.col("rec_o") == pl.col("rec_t")).head(3),
        reg.filter(pl.col("rec_o") != pl.col("rec_t")).head(3),
    ]).head(cfg["compare"]["n_showcase"])  # fmt: skip
    rows = []
    for r in picks.to_dicts():
        pct = lambda v: None if v is None else round(100 * v, 1)  # noqa: E731
        rows.append({
            "game_id": r["game_id"], "play_id": r["play_id"], "season_type": r["season_type"],
            "posteam": r["posteam"], "defteam": r["defteam"], "qtr": int(r["qtr"]),
            "game_seconds_remaining": int(r["game_seconds_remaining"]),
            "score_differential": int(r["score_differential"]),
            "ydstogo": int(r["ydstogo"]), "yardline_100": int(r["yardline_100"]),
            "description": (r["desc"] or "")[:160],
            "coach": r["fourth_down_decision"],
            "ours": {"go_wp": pct(r["go_wp"]), "fg_wp": pct(r["fg_wp"]),
                     "punt_wp": pct(r["punt_wp"]), "recommends": r["rec_o"],
                     "go_gain_points": pct(r["gain_o"]), "p_convert": pct(r["p_convert"])},
            "nfl4th": {"go_wp": pct(r["nfl4th_go_wp"]), "fg_wp": pct(r["nfl4th_fg_wp"]),
                       "punt_wp": pct(r["nfl4th_punt_wp"]), "recommends": r["rec_t"],
                       "go_gain_points": pct(r["gain_t"]),
                       "p_convert": pct(r["nfl4th_first_down_prob"])},
        })  # fmt: skip
    return rows


def scenario_comparison(engine: decision.DecisionEngine, cfg: dict) -> dict:
    """The hypothetical late-lead grid: nfl4th's numbers next to ours, same states."""
    ref = pl.read_csv(ROOT / cfg["compare"]["reference_scenarios"], null_values="NA")
    states = wp.state_frame(
        down=4, ydstogo=ref["ydstogo"].to_numpy(), yardline_100=ref["yardline_100"].to_numpy(),
        game_seconds_remaining=ref["quarter_seconds_remaining"].to_numpy(),
        score_differential=ref["score_differential"].to_numpy(),
        defteam_timeouts_remaining=ref["defteam_timeouts_remaining"].to_numpy(),
    )  # fmt: skip
    out = engine.evaluate(states)
    both = ref.with_columns(out["go_wp"], out["punt_wp"], out["go_gain"], out["best_action"])
    have = both.filter(pl.col("nfl4th_go_wp").is_not_null())
    theirs_gain = have["nfl4th_go_wp"] - have["nfl4th_punt_wp"]
    return {
        "rows": ref.height,
        "rows_where_nfl4th_returned_a_go_value": have.height,
        "note": "nfl4th returned NA for go_wp at yardline_100 50 and 60 in this grid, so only "
                "yardline_100 70 and 80 can be compared. Our states use a spread of 0; the "
                "reference script passed real team names, so nfl4th may have used a "
                "non-zero spread.",
        "same_go_or_punt": float(((have["go_gain"] > 0) == (theirs_gain > 0)).mean()),
        "ours_recommends_go": float((have["go_gain"] > 0).mean()),
        "nfl4th_recommends_go": float((theirs_gain > 0).mean()),
        "correlation_of_go_gain": float(np.corrcoef(f(have["go_gain"]), f(theirs_gain))[0, 1]),
        "mean_go_gain_ours_points": float(100 * have["go_gain"].mean()),
        "mean_go_gain_nfl4th_points": float(100 * theirs_gain.mean()),
        "mean_punt_wp_ours": float(both["punt_wp"].mean()),
        "mean_punt_wp_nfl4th": float(both["nfl4th_punt_wp"].mean()),
    }  # fmt: skip


def late_game_wp_check(engine: decision.DecisionEngine, plays: pl.DataFrame) -> dict:
    """Is our win-probability model right late in close games? First downs in the fourth
    quarter of the test seasons, by lead and time left: actual win rate of the team with
    the ball against the model's mean prediction (and nflfastR's vegas_wp for scale)."""
    base = plays.filter(
        (pl.col("down") == 1) & (pl.col("qtr") == 4) & pl.col("season").is_in([2024, 2025])
        & pl.col("posteam_won").is_not_null()
    )  # fmt: skip
    out = {}
    leads = {"lead_1_to_3": (1, 3), "lead_4_to_8": (4, 8), "trail_1_to_3": (-3, -1),
             "trail_4_to_8": (-8, -4)}  # fmt: skip
    clocks = {"0_to_40_seconds": (0, 40), "41_to_120": (41, 120), "121_to_300": (121, 300)}
    for lead_name, (lo, hi) in leads.items():
        for clock_name, (t0, t1) in clocks.items():
            sub = base.filter(
                pl.col("score_differential").is_between(lo, hi)
                & pl.col("game_seconds_remaining").is_between(t0, t1)
            )
            if sub.height == 0:
                continue
            out[f"{lead_name}__{clock_name}"] = {
                "n": sub.height,
                "actual_win_rate": float(sub["posteam_won"].mean()),
                "our_mean_wp": float(np.nanmean(engine.wp_model.predict(sub))),
                "nflfastr_vegas_wp_mean": float(sub["vegas_wp"].mean()),
            }
    return out


def figure(t: dict, bins: dict, swap: dict, overall: dict, corr: float, path) -> None:
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4),
                             gridspec_kw={"width_ratios": [1.05, 1, 1.05]})  # fmt: skip
    ax = axes[0]
    x, y = 100 * t["gain_t"], 100 * t["gain_o"]
    lim = 20
    ax.scatter(np.clip(x, -lim, lim), np.clip(y, -lim, lim), s=5, color=style.BLUE, alpha=0.22,
               linewidths=0, rasterized=True)  # fmt: skip
    ax.plot([-lim, lim], [-lim, lim], color=style.INK, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.axhline(0, color=style.AXIS, linewidth=1.0)
    ax.axvline(0, color=style.AXIS, linewidth=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("nfl4th: gain from going (win-probability points)")
    ax.set_ylabel("Ours: gain from going (points)")
    ax.text(-lim + 1, lim - 1.2, "we say go,\nnfl4th says kick", color=style.INK_2, fontsize=9,
            va="top")  # fmt: skip
    ax.text(lim - 1, -lim + 1.2, "nfl4th says go,\nwe say kick", color=style.INK_2, fontsize=9,
            ha="right", va="bottom")  # fmt: skip
    style.title(ax, "Gain from going, play by play",
                f"{overall['n']:,} fourth downs, 2024 to 2025. Correlation {corr:.2f}. "
                "Dashed: equal.")  # fmt: skip

    ax = axes[1]
    names = list(bins)
    rate = [bins[k]["same_action"] for k in names]
    ax.bar(range(len(names)), rate, color=style.BLUE, width=0.72)
    for i in range(len(names)):
        ax.text(i, rate[i] + 0.012, f"{rate[i]:.0%}", ha="center", color=style.INK, fontsize=10)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{k}\n({bins[k]['n']:,})" for k in names], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("How decisive nfl4th is: |gain from going| in points (plays)")
    ax.set_ylabel("Same recommended action")
    style.title(ax, "Agreement rises with decisiveness",
                f"Overall {overall['same_action']:.0%} same action (go, field goal or punt).")

    ax = axes[2]
    labels = ["Our engine as built", "... with nfl4th's\nconversion probability",
              "... and nfl4th's field\ngoal probability"]  # fmt: skip
    vals = [swap[k]["same_action"] for k in swap]
    ypos = np.arange(len(vals))[::-1]
    ax.barh(ypos, vals, color=style.BLUE, height=0.6)
    for yy, v in zip(ypos, vals, strict=True):
        ax.text(v + 0.012, yy, f"{v:.1%}", va="center", color=style.INK, fontsize=10)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Same recommended action as nfl4th")
    style.title(ax, "What explains the disagreement",
                "Diagnostic: nfl4th's probabilities with our win probabilities.")
    fig.tight_layout(w_pad=2.5)
    style.save(fig, path)


def main() -> None:
    cfg = decision.load_config()
    plays = data.load_plays()
    engine = decision.load_engine(plays, cfg)
    joined = load_joined(plays, cfg)
    engine.assert_can_score(joined, "nfl4th comparison on real fourth downs")
    values = engine.action_values(joined)
    evaluated = engine.evaluate(joined, out_of_sample=True, what="nfl4th comparison").with_columns(
        pl.Series("wp_if_fg_make", values["wp_if_fg_make"]),
        pl.Series("wp_if_fg_miss", values["wp_if_fg_miss"]),
    )
    # nfl4th writes 0 for a field goal that is out of range; treat it as unavailable.
    evaluated = evaluated.with_columns(
        pl.when(pl.col("nfl4th_fg_wp") > 0).then(pl.col("nfl4th_fg_wp")).alias("nfl4th_fg_wp")
    )
    usable = evaluated.filter(pl.col("go_wp").is_not_null() & pl.col("nfl4th_go_wp").is_not_null())
    t = tables(usable)
    ok = ~np.isnan(t["gain_o"]) & ~np.isnan(t["gain_t"])
    usable = usable.filter(pl.Series(ok))
    t = tables(usable)
    everything = np.ones(usable.height, dtype=bool)
    overall = agreement(t, everything)
    pearson = float(np.corrcoef(t["gain_o"], t["gain_t"])[0, 1])
    spearman = float(stats.spearmanr(t["gain_o"], t["gain_t"]).statistic)
    slope = float(np.polyfit(t["gain_t"], t["gain_o"], 1)[0])

    edges = cfg["compare"]["go_boost_bins"]
    boost = 100 * np.abs(t["gain_t"])
    bins = {}
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        label = f"{lo:g}-{hi:g}" if hi < 100 else f"{lo:g}+"
        bins[label] = agreement(t, (boost >= lo) & (boost < hi))
    decisive = agreement(t, boost >= cfg["compare"]["decisive_boost"])
    ours_decisive = agreement(t, 100 * np.abs(t["gain_o"]) >= cfg["compare"]["decisive_boost"])

    groups = {
        "by_distance": by_groups(usable, t, cut("ydstogo", [2, 3, 4, 7, 11],
                                 ["1", "2", "3", "4-6", "7-10", "11+"]),
                                 ["1", "2", "3", "4-6", "7-10", "11+"]),
        "by_field_position": by_groups(
            usable, t,
            cut("yardline_100", [11, 21, 36, 51, 71],
                ["opp 1-10", "opp 11-20", "opp 21-35", "opp 36-50", "own 30-49", "own 1-29"]),
            ["own 1-29", "own 30-49", "opp 36-50", "opp 21-35", "opp 11-20", "opp 1-10"]),
        "by_quarter": by_groups(usable, t, pl.format("Q{}", pl.col("qtr").cast(pl.Int64)),
                                ["Q1", "Q2", "Q3", "Q4"]),
        "by_score": by_groups(
            usable, t,
            cut("score_differential", [-8, -3, 0, 1, 4, 9],
                ["trail 9+", "trail 4-8", "trail 1-3", "tied", "lead 1-3", "lead 4-8", "lead 9+"]),
            ["trail 9+", "trail 4-8", "trail 1-3", "tied", "lead 1-3", "lead 4-8", "lead 9+"]),
        "by_clock": by_groups(
            usable, t,
            pl.when(pl.col("half_seconds_remaining") <= 120).then(pl.lit("last 2:00 of a half"))
            .when((pl.col("qtr") == 4) & (pl.col("game_seconds_remaining") <= 300))
            .then(pl.lit("Q4, 2:00 to 5:00 left")).otherwise(pl.lit("rest of the game")),
            ["rest of the game", "Q4, 2:00 to 5:00 left", "last 2:00 of a half"]),
    }  # fmt: skip
    worst = sorted(
        ((dim, g, v) for dim, d in groups.items() for g, v in d.items() if v.get("n", 0) >= 150),
        key=lambda x: x[2]["same_action"],
    )[:8]

    # Evidence on the causes.
    dp = f(usable["p_convert"]) - f(usable["nfl4th_first_down_prob"])
    dgain = t["gain_o"] - t["gain_t"]
    in_range = (f(usable["p_fg"]) > 0) & ~np.isnan(f(usable["nfl4th_fg_make_prob"]))
    disagree = t["best_o"] != t["best_t"]
    pairs = {}
    for a, name_a in enumerate(decision.ACTIONS):
        for b, name_b in enumerate(decision.ACTIONS):
            if a != b:
                pairs[f"ours_{name_a}_nfl4th_{name_b}"] = int(
                    np.sum((t["best_o"] == a) & (t["best_t"] == b))
                )
    dist = usable["ydstogo"].to_numpy()
    causes = {
        "conversion_probability": {
            "mean_ours": float(usable["p_convert"].mean()),
            "mean_nfl4th": float(usable["nfl4th_first_down_prob"].mean()),
            "mean_difference_by_distance": {
                str(d): float(dp[dist == d].mean()) for d in (1, 2, 3, 4, 5, 7, 10)
            },
            "correlation_of_gain_difference_with_probability_difference": float(
                np.corrcoef(dgain, dp)[0, 1]
            ),
        },
        "field_goal_probability_in_range": {
            "mean_ours": float(f(usable["p_fg"])[in_range].mean()),
            "mean_nfl4th": float(f(usable["nfl4th_fg_make_prob"])[in_range].mean()),
            "rows_nfl4th_has_no_field_goal_value_but_we_do": int(
                np.sum(np.isnan(t["theirs"][:, 1]) & ~np.isnan(t["ours"][:, 1]))
            ),
        },
        "stakes": {
            "sd_go_gain_ours_points": float(100 * np.std(t["gain_o"])),
            "sd_go_gain_nfl4th_points": float(100 * np.std(t["gain_t"])),
            "slope_of_ours_on_nfl4th": slope,
            "note": "A slope below 1 means our engine sees smaller differences between "
                    "going and kicking than nfl4th does (a flatter win-probability model).",
        },
        "swap_diagnostic": swap_diagnostic(usable, t),
        "late_game_win_probability_check_test_seasons": late_game_wp_check(engine, plays),
        "disagreements": {
            "n": int(disagree.sum()),
            "by_pair": pairs,
            "median_abs_nfl4th_gain_points": float(100 * np.median(np.abs(t["gain_t"][disagree]))),
            "median_abs_our_gain_points": float(100 * np.median(np.abs(t["gain_o"][disagree]))),
            "share_where_both_gains_are_under_2_points": float(np.mean(
                (np.abs(t["gain_t"][disagree]) < 0.02) & (np.abs(t["gain_o"][disagree]) < 0.02)
            )),
        },
        "coach_went": {
            "note": "Share of plays where the offense actually went, for orientation.",
            "actual": float((usable["fourth_down_decision"] == "go").mean()),
        },
    }  # fmt: skip

    result = {
        "reference": {
            "source": "nfl4th 1.0.7 via scripts/r/nfl4th_reference.R; these are nfl4th's numbers",
            "rows": joined.height,
            "rows_compared": usable.height,
            "rows_dropped_overtime_or_missing": joined.height - usable.height,
            "seasons": [2024, 2025],
        },
        "our_engine": {"models": cfg["models"], "seconds_per_play": engine.seconds},
        "overall": {**overall, "pearson_correlation_of_go_gain": pearson,
                    "spearman_correlation_of_go_gain": spearman},
        "when_nfl4th_is_decisive": {"threshold_points": cfg["compare"]["decisive_boost"],
                                    **decisive},
        "when_we_are_decisive": {"threshold_points": cfg["compare"]["decisive_boost"],
                                 **ours_decisive},
        "by_abs_nfl4th_go_boost_points": bins,
        **groups,
        "lowest_agreement_groups_with_150_plus_plays": [
            {"dimension": dim, "group": g, **v} for dim, g, v in worst
        ],
        "causes": causes,
        "scenarios_late_lead_grid": scenario_comparison(engine, cfg),
        "showcase_rule": showcase.__doc__.strip().split("\n\n")[1].replace("\n    ", " "),
        "showcase": showcase(usable, t, cfg),
    }  # fmt: skip
    write_json(result, RESULTS / "m2_nfl4th_comparison.json")
    figure(t, bins, causes["swap_diagnostic"], overall, pearson,
           ASSETS / "m2_nfl4th_comparison.png")  # fmt: skip

    print(f"compared {usable.height:,} fourth downs")
    print(f"  same action {overall['same_action']:.1%}; when nfl4th is decisive "
          f"{decisive['same_action']:.1%} (n={decisive['n']:,}); correlation {pearson:.3f}")
    for k, v in bins.items():
        print(f"  |boost| {k:7s} n={v['n']:5d} same action {v['same_action']:.1%}")
    for k, v in causes["swap_diagnostic"].items():
        print(f"  {k}: {v['same_action']:.1%}")


if __name__ == "__main__":
    main()
