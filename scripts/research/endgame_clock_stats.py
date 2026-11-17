"""Research for M4: end-game clock dynamics measured from the play-by-play table.

Produces every number quoted in docs/research/endgame_clock.md and saves them to
results/research/endgame_clock_stats.json. Nothing here is random, so there is no seed.

Run:  .venv/bin/python scripts/research/endgame_clock_stats.py

What is measured
  elapsed        seconds between the snap of a play and the snap of the next play, by
                 play category and game context
  timeouts       how often the defense stops the clock after a play that kept it running
  play_calling   run share on downs 1 to 3 by score and time left
  trailing       drives by a team down 4 to 8 that start with 1 to 5 minutes left
  kneels         clock and opponent timeouts at the first kneel of a game-ending series
  overtime       who wins games that reach overtime, by point spread
  state_space    size of the recommended clock model

Timeouts and the two-minute warning are not rows in the table (rows without a team in
possession were dropped in M0), so a timeout is inferred from a drop in a team's
timeouts-remaining count between two consecutive rows.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from fourth_and_rl.data import load_plays  # noqa: E402
from fourth_and_rl.paths import RESULTS  # noqa: E402

OUT = RESULTS / "research" / "endgame_clock_stats.json"

PLAY_CLOCK = 40
MAX_ELAPSED = 60  # larger gaps between snaps are treated as timestamp errors
CATEGORIES = [
    "run_in_bounds", "run_out_of_bounds", "pass_complete_in_bounds",
    "pass_complete_out_of_bounds", "pass_incomplete", "sack", "kneel", "spike", "punt",
    "field_goal", "kickoff",
]  # fmt: skip
RUNNING_CLOCK = ["run_in_bounds", "pass_complete_in_bounds", "sack", "kneel"]


def dist(values: pl.Series | np.ndarray) -> dict:
    """Sample size, quartiles and mean of a numeric sample."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"n": 0}
    q10, q25, q50, q75, q90 = np.percentile(x, [10, 25, 50, 75, 90])
    return {
        "n": int(x.size),
        "p10": round(float(q10), 1),
        "p25": round(float(q25), 1),
        "median": round(float(q50), 1),
        "p75": round(float(q75), 1),
        "p90": round(float(q90), 1),
        "mean": round(float(x.mean()), 1),
    }


def rate(flags: pl.Series) -> dict:
    """Sample size and share of a 0/1 or boolean column."""
    n = flags.len()
    return {"n": n, "rate": round(float(flags.cast(pl.Float64).mean()), 3) if n else None}


def add_sequence_columns(plays: pl.DataFrame) -> pl.DataFrame:
    """Keep real plays, add the next play's clock, timeout use and a play category."""
    is_home = pl.col("is_home") == 1
    off, deff = pl.col("posteam_timeouts_remaining"), pl.col("defteam_timeouts_remaining")
    df = (
        plays.filter(pl.col("play_type").is_not_null())
        .sort("game_id", "order_sequence", "play_id")
        .with_columns(
            pl.when(is_home).then(off).otherwise(deff).alias("home_to"),
            pl.when(is_home).then(deff).otherwise(off).alias("away_to"),
        )
    )

    def nxt(col: str) -> pl.Expr:
        return pl.col(col).shift(-1).over("game_id")

    df = df.with_columns(
        (pl.col("game_seconds_remaining") - nxt("game_seconds_remaining")).alias("elapsed"),
        (nxt("qtr") == pl.col("qtr")).fill_null(False).alias("next_same_qtr"),
        nxt("half_seconds_remaining").alias("next_half_seconds"),
        (
            pl.when(is_home).then(pl.col("away_to") - nxt("away_to"))
            .otherwise(pl.col("home_to") - nxt("home_to"))
            > 0
        ).fill_null(False).alias("def_timeout_after"),
        (
            pl.when(is_home).then(pl.col("home_to") - nxt("home_to"))
            .otherwise(pl.col("away_to") - nxt("away_to"))
            > 0
        ).fill_null(False).alias("off_timeout_after"),
    )  # fmt: skip

    desc = pl.col("desc").fill_null("")
    oob = desc.str.contains(r"(?i)\bob\b|out of bounds")
    incomplete = desc.str.contains("(?i)incomplete")
    ptype = pl.col("play_type")
    stops_anyway = (
        (pl.col("touchdown") == 1)
        | (pl.col("interception") == 1)
        | (pl.col("fumble_lost") == 1)
        | (pl.col("safety") == 1)
        | (pl.col("down") == 4)
    )
    scrimmage = ptype.is_in(["run", "pass"])
    category = (
        pl.when(pl.col("qb_kneel") == 1).then(pl.lit("kneel"))
        .when(pl.col("qb_spike") == 1).then(pl.lit("spike"))
        .when(ptype == "punt").then(pl.lit("punt"))
        .when(ptype == "field_goal").then(pl.lit("field_goal"))
        .when(ptype == "kickoff").then(pl.lit("kickoff"))
        .when(scrimmage & stops_anyway).then(pl.lit(None))
        .when(pl.col("sack") == 1).then(pl.lit("sack"))
        .when((ptype == "pass") & incomplete).then(pl.lit("pass_incomplete"))
        .when((ptype == "pass") & oob).then(pl.lit("pass_complete_out_of_bounds"))
        .when(ptype == "pass").then(pl.lit("pass_complete_in_bounds"))
        .when((ptype == "run") & oob).then(pl.lit("run_out_of_bounds"))
        .when(ptype == "run").then(pl.lit("run_in_bounds"))
        .otherwise(pl.lit(None))
    )  # fmt: skip
    crosses_warning = (
        pl.col("qtr").is_in([2.0, 4.0])
        & (pl.col("half_seconds_remaining") > 120)
        & (pl.col("next_half_seconds") <= 120)
    )
    return df.with_columns(
        category.alias("category"),
        crosses_warning.fill_null(False).alias("crosses_warning"),
        (pl.col("def_timeout_after") | pl.col("off_timeout_after")).alias("timeout_after"),
    )


def timed(df: pl.DataFrame) -> pl.DataFrame:
    """Regulation plays with a usable gap to the next snap in the same quarter."""
    return df.filter(
        (pl.col("qtr") <= 4)
        & pl.col("next_same_qtr")
        & pl.col("category").is_not_null()
        & (pl.col("penalty") != 1)
        & pl.col("elapsed").is_between(0, MAX_ELAPSED)
    )


CONTEXTS: dict[str, pl.Expr] = {
    "all_regulation": pl.lit(True),
    "q4_leading": (pl.col("qtr") == 4) & (pl.col("score_differential") > 0),
    "q4_trailing": (pl.col("qtr") == 4) & (pl.col("score_differential") < 0),
    "last5_leading_1_to_8": (pl.col("game_seconds_remaining") <= 300)
    & pl.col("score_differential").is_between(1, 8),
    "last5_trailing_1_to_8": (pl.col("game_seconds_remaining") <= 300)
    & pl.col("score_differential").is_between(-8, -1),
}


def elapsed_tables(df: pl.DataFrame) -> dict:
    """Seconds from snap to next snap. Plays followed by a timeout or the warning are out."""
    t = timed(df)
    clean = t.filter(~pl.col("timeout_after") & ~pl.col("crosses_warning"))
    out: dict = {"dropped_for_bad_gap": int(
        df.filter(
            (pl.col("qtr") <= 4) & pl.col("next_same_qtr") & pl.col("category").is_not_null()
            & (pl.col("penalty") != 1) & ~pl.col("elapsed").is_between(0, MAX_ELAPSED)
        ).height
    )}  # fmt: skip
    for name, cond in CONTEXTS.items():
        sub = clean.filter(cond)
        out[name] = {c: dist(sub.filter(pl.col("category") == c)["elapsed"]) for c in CATEGORIES}
    # Length of the play itself: the defense stopped the clock right after it.
    stopped = t.filter(pl.col("def_timeout_after") & ~pl.col("off_timeout_after"))
    out["followed_by_defensive_timeout"] = {
        c: dist(stopped.filter(pl.col("category") == c)["elapsed"]) for c in RUNNING_CLOCK
    }
    # Trailing offense stops the clock itself after an in-bounds play.
    own = t.filter(pl.col("off_timeout_after") & ~pl.col("def_timeout_after"))
    out["followed_by_offensive_timeout"] = {
        c: dist(own.filter(pl.col("category") == c)["elapsed"]) for c in RUNNING_CLOCK
    }
    # Distribution on a 5-second grid for the two tempo modes (for the DP tables).
    grid: dict = {}
    for name in ["last5_leading_1_to_8", "last5_trailing_1_to_8"]:
        sub = clean.filter(CONTEXTS[name])
        grid[name] = {}
        for c in ["run_in_bounds", "pass_complete_in_bounds", "pass_incomplete"]:
            x = sub.filter(pl.col("category") == c)["elapsed"].to_numpy()
            ticks = np.clip(np.rint(x / 5).astype(int), 1, 10)
            share = np.bincount(ticks, minlength=11)[1:] / max(len(ticks), 1)
            grid[name][c] = {
                "n": int(len(ticks)),
                "share_by_5s_tick_1_to_10": [round(float(s), 3) for s in share],
            }
    out["five_second_grid"] = grid
    return out


def timeout_usage(df: pl.DataFrame) -> dict:
    """Share of clock-running plays by a team protecting a lead that the defense stops."""
    base = df.filter(
        (pl.col("qtr") == 4)
        & (pl.col("game_seconds_remaining") <= 300)
        & pl.col("score_differential").is_between(1, 8)
        & pl.col("category").is_in(RUNNING_CLOCK)
        & pl.col("next_same_qtr")
        & (pl.col("penalty") != 1)
        & ~pl.col("crosses_warning")
        & (pl.col("first_down") != 1)
    )
    out: dict = {}
    for label, cond in {
        "before_warning": pl.col("game_seconds_remaining") > 120,
        "after_warning": pl.col("game_seconds_remaining") <= 120,
    }.items():
        out[label] = {
            f"defense_has_{k}": rate(
                base.filter(cond & (pl.col("defteam_timeouts_remaining") == k))["def_timeout_after"]
            )
            for k in (1, 2, 3)
        }
    return out


def timeouts_entering_endgame(df: pl.DataFrame) -> dict:
    """Timeouts each side holds at the first snap inside 5:00 of a one-score game."""
    first = (
        df.filter(
            (pl.col("qtr") == 4)
            & (pl.col("game_seconds_remaining") <= 300)
            & pl.col("down").is_not_null()
            & (pl.col("score_differential").abs().is_between(1, 8))
        )
        .group_by("game_id", maintain_order=True)
        .first()
        .with_columns(
            pl.when(pl.col("score_differential") < 0)
            .then(pl.col("posteam_timeouts_remaining"))
            .otherwise(pl.col("defteam_timeouts_remaining"))
            .alias("trailing_to"),
            pl.when(pl.col("score_differential") > 0)
            .then(pl.col("posteam_timeouts_remaining"))
            .otherwise(pl.col("defteam_timeouts_remaining"))
            .alias("leading_to"),
        )
    )
    n = first.height
    return {
        "n_games": n,
        "trailing_team_share": {
            str(k): round(first.filter(pl.col("trailing_to") == k).height / n, 3) for k in range(4)
        },
        "leading_team_share": {
            str(k): round(first.filter(pl.col("leading_to") == k).height / n, 3) for k in range(4)
        },
    }


def play_calling(df: pl.DataFrame) -> dict:
    """Run share (runs over runs plus passes, kneels and spikes excluded), downs 1 to 3."""
    sd, gs = pl.col("score_differential"), pl.col("game_seconds_remaining")
    base = df.filter(
        (pl.col("qtr") <= 4)
        & pl.col("down").is_between(1, 3)
        & pl.col("play_type").is_in(["run", "pass"])
        & (pl.col("qb_kneel") != 1)
        & (pl.col("qb_spike") != 1)
    ).with_columns((pl.col("play_type") == "run").alias("is_run"))
    scores = {
        "lead_9_plus": sd >= 9,
        "lead_1_to_8": sd.is_between(1, 8),
        "tied": sd == 0,
        "trail_1_to_8": sd.is_between(-8, -1),
        "trail_9_plus": sd <= -9,
    }
    times = {
        "quarters_1_to_3": gs > 900,
        "q4_15_to_10_min": gs.is_between(601, 900),
        "q4_10_to_5_min": gs.is_between(301, 600),
        "q4_5_to_2_min": gs.is_between(121, 300),
        "q4_last_2_min": gs <= 120,
    }
    return {
        s: {t: rate(base.filter(sc & tc)["is_run"]) for t, tc in times.items()}
        for s, sc in scores.items()
    }


def drive_table(plays: pl.DataFrame) -> pl.DataFrame:
    """One row per drive: state at its first snap, result, and when the next drive starts."""
    snaps = plays.filter(pl.col("down").is_not_null() & pl.col("fixed_drive").is_not_null()).sort(
        "game_id", "order_sequence", "play_id"
    )
    counted = pl.col("play_type").is_in(["run", "pass"]) & (pl.col("qb_kneel") != 1)
    drives = (
        snaps.group_by("game_id", "fixed_drive", maintain_order=True)
        .agg(
            pl.col("posteam").first(),
            pl.col("qtr").first(),
            pl.col("game_seconds_remaining").first().alias("start_seconds"),
            pl.col("score_differential").first().alias("start_diff"),
            pl.col("yardline_100").first().alias("start_yardline"),
            pl.col("posteam_timeouts_remaining").first().alias("off_to"),
            pl.col("defteam_timeouts_remaining").first().alias("def_to"),
            pl.col("fixed_drive_result").first().alias("result"),
            pl.col("posteam_won").first().alias("won"),
            counted.sum().alias("n_plays"),
            pl.col("season_type").first(),
        )
        .sort("game_id", "fixed_drive")
    )
    # Clock right after the drive: the first row of any kind after the drive's last row.
    ordered = plays.sort("game_id", "order_sequence", "play_id").with_columns(
        pl.col("game_seconds_remaining").shift(-1).over("game_id").alias("after_seconds"),
        pl.col("qtr").shift(-1).over("game_id").alias("after_qtr"),
    )
    last = (
        ordered.filter(pl.col("fixed_drive").is_not_null())
        .group_by("game_id", "fixed_drive", maintain_order=True)
        .agg(pl.col("after_seconds").last(), pl.col("after_qtr").last())
    )
    drives = drives.join(last, on=["game_id", "fixed_drive"], how="left")

    def nxt(col: str) -> pl.Expr:
        return pl.col(col).shift(-1).over("game_id")

    return drives.with_columns(
        nxt("posteam").alias("next_posteam"),
        nxt("qtr").alias("next_qtr"),
        nxt("start_seconds").alias("next_start_seconds"),
        nxt("start_diff").alias("next_start_diff"),
        nxt("result").alias("next_result"),
    )


def trailing_drives(drives: pl.DataFrame) -> dict:
    """Drives by a team down 4 to 8 whose first snap is in Q4 with 1:00 to 5:00 left."""
    d = drives.filter(
        (pl.col("qtr") == 4)
        & pl.col("start_seconds").is_between(60, 300)
        & pl.col("start_diff").is_between(-8, -4)
    ).with_columns(
        (pl.col("result") == "Touchdown").alias("td"),
        pl.when(pl.col("start_yardline") >= 80).then(pl.lit("own_1_to_20"))
        .when(pl.col("start_yardline") >= 70).then(pl.lit("own_21_to_30"))
        .when(pl.col("start_yardline") >= 50).then(pl.lit("own_31_to_50"))
        .otherwise(pl.lit("opp_territory")).alias("field_bin"),
        pl.when(pl.col("start_seconds") <= 120).then(pl.lit("1_to_2_min"))
        .when(pl.col("start_seconds") <= 180).then(pl.lit("2_to_3_min"))
        .when(pl.col("start_seconds") <= 240).then(pl.lit("3_to_4_min"))
        .otherwise(pl.lit("4_to_5_min")).alias("time_bin"),
        pl.when(pl.col("off_to") <= 1).then(pl.lit("0_or_1"))
        .when(pl.col("off_to") == 2).then(pl.lit("2"))
        .otherwise(pl.lit("3")).alias("timeout_bin"),
        pl.when(pl.col("start_diff") >= -5).then(pl.lit("down_4_or_5"))
        .when(pl.col("start_diff") >= -7).then(pl.lit("down_6_or_7"))
        .otherwise(pl.lit("down_8")).alias("deficit_bin"),
        (pl.col("start_seconds") - pl.col("after_seconds")).alias("seconds_used"),
    )  # fmt: skip

    def cells(keys: list[str]) -> list[dict]:
        g = (
            d.group_by(keys)
            .agg(
                pl.len().alias("n"),
                pl.col("td").sum().alias("touchdowns"),
                pl.col("td").mean().round(3).alias("td_rate"),
                pl.col("won").mean().round(3).alias("trailing_team_win_rate"),
            )
            .sort(keys)
        )
        return g.to_dicts()

    out: dict = {
        "n_drives": d.height,
        "n_games": d["game_id"].n_unique(),
        "td_rate": rate(d["td"]),
        "trailing_team_win_rate": round(float(d["won"].mean()), 3),
        "result_counts": {
            r["result"]: r["len"] for r in d.group_by("result").len().sort("result").to_dicts()
        },
        "by_field_position": cells(["field_bin"]),
        "by_time": cells(["time_bin"]),
        "by_timeouts": cells(["timeout_bin"]),
        "by_deficit": cells(["deficit_bin"]),
        "by_field_and_time": cells(["field_bin", "time_bin"]),
        "by_time_and_timeouts": cells(["time_bin", "timeout_bin"]),
        "by_field_time_timeouts": cells(["field_bin", "time_bin", "timeout_bin"]),
    }

    # How long the scoring drives take, and the pace of all these drives.
    td = d.filter(pl.col("td") & (pl.col("after_qtr") == 4))
    out["td_drive_seconds_used"] = dist(td["seconds_used"])
    out["td_drive_plays"] = dist(td["n_plays"])
    paced = d.filter((pl.col("n_plays") >= 3) & (pl.col("after_qtr") == 4))
    out["seconds_per_play_all_drives"] = dist(paced["seconds_used"] / paced["n_plays"])

    # After a touchdown: does the team that had been leading get the ball with time left?
    after_td = d.filter(pl.col("td")).with_columns(
        (
            (pl.col("next_posteam") != pl.col("posteam"))
            & (pl.col("next_qtr") == 4)
            & pl.col("next_start_seconds").is_not_null()
        ).fill_null(False).alias("answer_drive"),
    )  # fmt: skip
    ans = after_td.filter(pl.col("answer_drive"))
    scored = pl.col("next_result").is_in(["Touchdown", "Field goal"])
    out["after_trailing_touchdown"] = {
        "n_touchdown_drives": after_td.height,
        "leader_gets_ball_in_regulation": rate(after_td["answer_drive"]),
        "seconds_left_at_answer_drive": dist(ans["next_start_seconds"]),
        "share_with_30s_or_more": rate(
            (after_td["answer_drive"] & (after_td["next_start_seconds"] >= 30)).fill_null(False)
        ),
        "share_with_60s_or_more": rate(
            (after_td["answer_drive"] & (after_td["next_start_seconds"] >= 60)).fill_null(False)
        ),
        "original_leader_win_rate": round(1 - float(after_td["won"].mean()), 3),
    }
    # The answer drive when the former leader is now behind or tied and has 30s or more.
    need = ans.filter((pl.col("next_start_seconds") >= 30) & (pl.col("next_start_diff") <= 0))
    out["answer_drive_needing_points"] = {
        "n": need.height,
        "scores_td_or_fg": rate(need.select(scored.alias("s"))["s"]),
        "original_leader_win_rate": (
            round(1 - float(need["won"].mean()), 3) if need.height else None
        ),
        "by_seconds_left": [
            {
                "bin": label,
                "n": sub.height,
                "scores_td_or_fg": rate(sub.select(scored.alias("s"))["s"])["rate"],
                "original_leader_win_rate": (
                    round(1 - float(sub["won"].mean()), 3) if sub.height else None
                ),
            }
            for label, sub in (
                ("30_to_59", need.filter(pl.col("next_start_seconds") < 60)),
                ("60_to_119", need.filter(pl.col("next_start_seconds").is_between(60, 119))),
                ("120_plus", need.filter(pl.col("next_start_seconds") >= 120)),
            )
        ],
    }
    # When the trailing drive does not score a touchdown.
    no_td = d.filter(~pl.col("td"))
    out["after_no_touchdown"] = {
        "n": no_td.height,
        "original_leader_win_rate": round(1 - float(no_td["won"].mean()), 3),
    }
    return out


def kneel_threshold(timeouts: int, kneels_available: int, kneel_seconds: float) -> float:
    """Latest snap time (seconds left) at which kneeling alone ends the game.

    Each kneel takes `kneel_seconds`; after it the 40-second play clock runs off unless
    the defense has a timeout for it. The two-minute warning stops the clock once, so a
    series cannot start above 2:00 and still finish.
    """
    runoffs = max(kneels_available - timeouts, 0)
    return min(PLAY_CLOCK * runoffs + kneel_seconds * kneels_available, 120.0)


def kneels(df: pl.DataFrame, kneel_seconds: float) -> dict:
    """First kneel of the series of kneels that ends a game, for the team in front."""
    q4 = df.filter(pl.col("qtr") == 4).with_columns((pl.col("qb_kneel") == 1).alias("is_kneel"))
    # Number of non-kneel plays from this row to the end of the game (inclusive).
    q4 = q4.with_columns(
        (~pl.col("is_kneel")).cast(pl.Int32).reverse().cum_sum().reverse().over("game_id")
        .alias("non_kneels_left")
    )  # fmt: skip
    series = q4.filter(pl.col("is_kneel") & (pl.col("non_kneels_left") == 0))
    first = (
        series.group_by("game_id", maintain_order=True)
        .agg(
            pl.col("game_seconds_remaining").first().alias("seconds"),
            pl.col("down").first(),
            pl.col("defteam_timeouts_remaining").first().alias("def_to"),
            pl.col("score_differential").first().alias("diff"),
            pl.len().alias("n_kneels"),
            pl.col("game_seconds_remaining").n_unique().alias("n_times"),
        )
        .filter((pl.col("diff") > 0) & pl.col("down").is_between(1, 3))
        # Games whose kneels all carry one timestamp have a broken clock column.
        .filter((pl.col("n_kneels") == 1) | (pl.col("n_times") > 1))
        .with_columns(
            pl.struct("def_to", "down")
            .map_elements(
                lambda r: kneel_threshold(int(r["def_to"]), 4 - int(r["down"]), kneel_seconds),
                return_dtype=pl.Float64,
            )
            .alias("threshold")
        )
    )
    close = first.filter(pl.col("diff") <= 8)
    out: dict = {
        "n_games_ending_in_kneels": first.height,
        "n_games_ending_in_kneels_lead_1_to_8": close.height,
        "kneel_seconds_used": kneel_seconds,
        "lead_1_to_8_share_above_formula_threshold": round(
            close.filter(pl.col("seconds") > pl.col("threshold")).height / close.height, 4
        ),
        "lead_1_to_8_share_above_threshold_plus_10s": round(
            close.filter(pl.col("seconds") > pl.col("threshold") + 10).height / close.height, 4
        ),
        "first_kneel_on_first_down_lead_1_to_8": {},
        "first_kneel_on_first_down_lead_9_plus": {},
        "formula_thresholds_first_down": {
            str(k): kneel_threshold(k, 3, kneel_seconds) for k in range(4)
        },
        "formula_thresholds_second_down": {
            str(k): kneel_threshold(k, 2, kneel_seconds) for k in range(4)
        },
        "formula_thresholds_third_down": {
            str(k): kneel_threshold(k, 1, kneel_seconds) for k in range(4)
        },
    }
    groups = {
        "first_kneel_on_first_down_lead_1_to_8": close,
        "first_kneel_on_first_down_lead_9_plus": first.filter(pl.col("diff") >= 9),
    }
    for name, group in groups.items():
        for k in range(4):
            sel = group.filter((pl.col("down") == 1) & (pl.col("def_to") == k))
            x = sel["seconds"].to_numpy()
            thr = kneel_threshold(k, 3, kneel_seconds)
            if x.size == 0:
                out[name][f"defense_has_{k}"] = {"n": 0}
                continue
            out[name][f"defense_has_{k}"] = {
                "n": int(x.size),
                "median": float(np.median(x)),
                "p90": round(float(np.percentile(x, 90)), 1),
                "p95": round(float(np.percentile(x, 95)), 1),
                "max": float(x.max()),
                "formula_threshold": thr,
                "n_above_threshold": int((x > thr).sum()),
                "n_above_threshold_plus_10s": int((x > thr + 10).sum()),
                "n_within_10s_below_threshold": int(((x <= thr) & (x > thr - 10)).sum()),
            }

    # The other direction: a first down inside the threshold that was NOT knelt out.
    lead = df.filter(
        (pl.col("qtr") == 4)
        & (pl.col("down") == 1)
        & pl.col("score_differential").is_between(1, 8)
        & pl.col("play_type").is_in(["run", "pass", "qb_kneel"])
        & (pl.col("penalty") != 1)
    ).with_columns(
        pl.col("defteam_timeouts_remaining")
        .map_elements(lambda k: kneel_threshold(int(k), 3, kneel_seconds), return_dtype=pl.Float64)
        .alias("threshold")
    )
    inside = lead.filter(pl.col("game_seconds_remaining") <= pl.col("threshold"))
    out["first_down_inside_threshold_lead_1_to_8"] = {
        "n": inside.height,
        "kneel_share": round(inside.filter(pl.col("qb_kneel") == 1).height / inside.height, 3),
        "leading_team_win_rate": round(float(inside["posteam_won"].mean()), 4),
    }
    just_outside = lead.filter(
        (pl.col("game_seconds_remaining") > pl.col("threshold") + 5)
        & (pl.col("game_seconds_remaining") <= pl.col("threshold") + 45)
    )
    out["first_down_5_to_45s_outside_threshold_lead_1_to_8"] = {
        "n": just_outside.height,
        "kneel_share": round(
            just_outside.filter(pl.col("qb_kneel") == 1).height / just_outside.height, 3
        ),
        "leading_team_win_rate": round(float(just_outside["posteam_won"].mean()), 4),
    }
    return out


def overtime(plays: pl.DataFrame) -> dict:
    """Outcome of games that reached overtime, from the pregame favorite's side."""
    games = plays.group_by("game_id").agg(
        pl.col("qtr").max().alias("max_qtr"),
        pl.col("result").first(),
        pl.col("spread_line").first(),
        pl.col("season").first(),
        pl.col("season_type").first(),
    )
    ot = games.filter(pl.col("max_qtr") >= 5).with_columns(
        pl.col("spread_line").abs().alias("fav_by"),
        pl.when(pl.col("result") == 0).then(0.5)
        .when((pl.col("result") > 0) == (pl.col("spread_line") > 0)).then(1.0)
        .otherwise(0.0).alias("fav_points"),
        pl.when(pl.col("result") == 0).then(0.5)
        .when(pl.col("result") > 0).then(1.0).otherwise(0.0).alias("home_points"),
    )  # fmt: skip
    picked = ot.filter(pl.col("fav_by") > 0)
    x = picked["fav_by"].to_numpy()
    y = picked["fav_points"].to_numpy() - 0.5
    slope = float((x * y).sum() / (x * x).sum())  # least squares through (0, 0.5)
    bins = {
        "0.5_to_2.5": (0.5, 2.5),
        "3_to_6.5": (3.0, 6.5),
        "7_plus": (7.0, 99.0),
    }
    return {
        "n_games_total": games.height,
        "n_overtime_games": ot.height,
        "share_of_games_reaching_overtime": round(ot.height / games.height, 4),
        "n_ties": ot.filter(pl.col("result") == 0).height,
        "n_overtime_regular_season": ot.filter(pl.col("season_type") == "REG").height,
        "n_overtime_playoffs": ot.filter(pl.col("season_type") == "POST").height,
        "n_overtime_2025_season": ot.filter(pl.col("season") == 2025).height,
        "home_team_win_value": round(float(ot["home_points"].mean()), 3),
        "favorite_win_value": round(float(picked["fav_points"].mean()), 3),
        "favorite_win_value_by_spread": {
            k: {
                "n": picked.filter(pl.col("fav_by").is_between(lo, hi)).height,
                "value": round(
                    float(picked.filter(pl.col("fav_by").is_between(lo, hi))["fav_points"].mean()),
                    3,
                ),
            }
            for k, (lo, hi) in bins.items()
        },
        "win_value_slope_per_spread_point": round(slope, 4),
        "note": "a tie counts as 0.5; slope is least squares of (value - 0.5) on the spread",
    }


def state_space() -> dict:
    """State counts for the recommended clock model (upper bounds, before pruning)."""
    field = 99 * 4 * 12  # yard line x down x distance bin (1..10, 11-15, 16+)
    score = 43  # score differential -21..+21
    timeouts = 4 * 4  # each side 0..3
    out: dict = {"field_states": field, "score_states": score, "timeout_states": timeouts}
    for minutes in (5, 8):
        for step in (5, 10):
            ticks = minutes * 60 // step
            out[f"endgame_{minutes}min_{step}s"] = {
                "time_steps": ticks,
                "states": ticks * field * score * timeouts,
                "states_per_time_slice": field * score * timeouts,
            }
    early_seconds = 3600 - 300
    for step in (20, 30):
        ticks = early_seconds // step
        out[f"early_game_{step}s_no_timeouts"] = {
            "time_steps": ticks,
            "states": ticks * field * score,
        }
    out["whole_game_recommended"] = (
        out["endgame_5min_5s"]["states"] + out["early_game_30s_no_timeouts"]["states"]
    )
    out["whole_game_at_1s_with_timeouts"] = 3600 * field * score * timeouts
    out["float32_gigabytes_recommended"] = round(out["whole_game_recommended"] * 4 / 1e9, 2)
    return out


def main() -> None:
    plays = load_plays()
    df = add_sequence_columns(plays)
    elapsed = elapsed_tables(df)
    kneel_seconds = elapsed["followed_by_defensive_timeout"]["kneel"]["median"]
    results = {
        "source": "data/processed/plays.parquet via fourth_and_rl.data.load_plays()",
        "n_rows": plays.height,
        "seasons": [int(plays["season"].min()), int(plays["season"].max())],
        "elapsed_seconds_snap_to_snap": elapsed,
        "defensive_timeout_usage": timeout_usage(df),
        "timeouts_entering_last_5_min": timeouts_entering_endgame(df),
        "run_share": play_calling(df),
        "trailing_drives_down_4_to_8": trailing_drives(drive_table(plays)),
        "kneels": kneels(df, kneel_seconds),
        "overtime": overtime(plays),
        "state_space": state_space(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
