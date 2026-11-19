"""M0: download play-by-play, build the state table and write the league-rate summary.

Outputs:
  data/processed/plays.parquet          the state table (not committed)
  results/m0_data_summary.json          row counts and league rates
  assets/m0_fourth_down_go_rate.png     fourth-down go rate by season
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import polars as pl

from fourth_and_rl import data, style
from fourth_and_rl.io import write_json
from fourth_and_rl.paths import ASSETS, RESULTS


def summarize(plays: pl.DataFrame) -> dict:
    per_season = (
        plays.group_by("season")
        .agg(
            pl.col("game_id").n_unique().alias("games"),
            pl.len().alias("plays"),
            pl.col("game_id").filter(pl.col("posteam_won").is_null()).n_unique().alias("ties"),
        )
        .sort("season")
    )
    fourth = plays.filter(pl.col("fourth_down_decision").is_not_null())
    fourth_by_season = (
        fourth.group_by("season")
        .agg(
            pl.len().alias("fourth_downs"),
            (pl.col("fourth_down_decision") == "go").mean().alias("go_rate"),
            (pl.col("fourth_down_decision") == "field_goal").mean().alias("field_goal_rate"),
            (pl.col("fourth_down_decision") == "punt").mean().alias("punt_rate"),
            pl.col("fourth_down_converted")
            .filter(pl.col("fourth_down_decision") == "go")
            .mean()
            .alias("conversion_rate"),
        )
        .sort("season")
    )
    fg = plays.filter(pl.col("play_type") == "field_goal")
    punts = plays.filter((pl.col("play_type") == "punt") & (pl.col("punt_blocked") != 1))
    scrimmage = plays.filter(pl.col("play_type").is_in(["run", "pass"]))
    return {
        "seasons": [int(per_season["season"].min()), int(per_season["season"].max())],
        "total_plays": plays.height,
        "total_games": plays["game_id"].n_unique(),
        "per_season": per_season.to_dicts(),
        "fourth_downs_by_season": fourth_by_season.to_dicts(),
        "league_rates": {
            "field_goal_make_rate": float((fg["field_goal_result"] == "made").mean()),
            "field_goal_attempts": fg.height,
            "fourth_down_conversion_rate": float(
                fourth.filter(pl.col("fourth_down_decision") == "go")
                .get_column("fourth_down_converted")
                .mean()
            ),
            "punt_gross_yards_mean": float(punts["kick_distance"].mean()),
            "punt_touchback_rate": float(punts["touchback"].mean()),
            "yards_per_scrimmage_play": float(scrimmage["yards_gained"].mean()),
            "pass_share_of_scrimmage_plays": float((scrimmage["play_type"] == "pass").mean()),
            "share_of_plays_missing_nflfastr_wp": float(plays["wp"].is_null().mean()),
        },
    }


def figure(summary: dict) -> None:
    style.apply()
    rows = summary["fourth_downs_by_season"]
    seasons = [r["season"] for r in rows]
    go = [100 * r["go_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(seasons, go, color=style.ACTION_COLORS["go"], marker="o", markersize=6,
            markeredgecolor=style.SURFACE, markeredgewidth=1.5)
    ax.set_ylim(0, max(go) * 1.2)
    ax.set_xticks(seasons[::2])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Share of fourth downs (%)")
    style.title(
        ax,
        "NFL teams go for it on fourth down far more often than they used to",
        "Share of fourth-down plays that were a run or pass, by season",
    )
    for i in (0, len(seasons) - 1):
        ax.annotate(f"{go[i]:.1f}%", (seasons[i], go[i]), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=style.INK_2, fontsize=10)
    style.save(fig, ASSETS / "m0_fourth_down_go_rate.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download raw seasons")
    args = parser.parse_args()
    path = data.build(force=args.force)
    plays = data.load_plays()
    summary = summarize(plays)
    write_json(summary, RESULTS / "m0_data_summary.json")
    figure(summary)
    print(f"state table: {path} ({plays.height:,} plays, {summary['total_games']:,} games)")
    for k, v in summary["league_rates"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
