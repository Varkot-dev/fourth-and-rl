from __future__ import annotations

import polars as pl
import pytest

from fourth_and_rl import data


def _raw(rows: list[dict]) -> pl.DataFrame:
    """A tiny raw frame with every column the builder touches."""
    base = {c: None for c in data.RAW_COLUMNS}
    schema = {c: pl.Utf8 for c in data.RAW_COLUMNS}
    for c in ("season", "qtr", "down", "result", "order_sequence", "play_id",
              "qb_kneel", "qb_spike"):
        schema[c] = pl.Int64
    schema["spread_line"] = pl.Float64
    return pl.DataFrame([{**base, **r} for r in rows], schema=schema)


def _toy_game() -> pl.DataFrame:
    common = dict(game_id="g1", season=2020, home_team="HOM", away_team="AWY",
                  result=7, spread_line=3.0, qb_kneel=0, qb_spike=0)
    return _raw([
        # AWY receives the opening kickoff, so HOM receives the second-half one.
        dict(common, play_id=1, order_sequence=1, posteam="AWY", play_type="kickoff", qtr=1),
        dict(common, play_id=2, order_sequence=2, posteam="AWY", play_type="pass", qtr=1,
             down=4),
        dict(common, play_id=3, order_sequence=3, posteam="HOM", play_type="punt", qtr=2,
             down=4),
        dict(common, play_id=4, order_sequence=4, posteam="HOM", play_type="field_goal", qtr=3,
             down=4),
        dict(common, play_id=5, order_sequence=5, posteam="HOM", play_type="no_play", qtr=4,
             down=4),
        dict(common, play_id=6, order_sequence=6, posteam=None, play_type=None, qtr=4),
    ])  # fmt: skip


def test_state_table_drops_rows_without_possession():
    assert data.build_state_table(_toy_game()).height == 5


def test_outcome_and_spread_are_from_the_posteam_side():
    t = data.build_state_table(_toy_game())
    home = t.filter(pl.col("posteam") == "HOM")
    away = t.filter(pl.col("posteam") == "AWY")
    assert home["posteam_won"].unique().to_list() == [1.0]
    assert away["posteam_won"].unique().to_list() == [0.0]
    assert home["posteam_spread"].unique().to_list() == [3.0]
    assert away["posteam_spread"].unique().to_list() == [-3.0]


def test_tied_game_has_null_outcome():
    t = data.build_state_table(_toy_game().with_columns(pl.lit(0).alias("result")))
    assert t["posteam_won"].null_count() == t.height


def test_second_half_kickoff_flag():
    t = data.build_state_table(_toy_game()).sort("play_id")
    # AWY received the opener: 0 for AWY, 1 for HOM in the first half, 0 after halftime.
    assert t["receive_2h_ko"].to_list() == [0, 0, 1, 0, 0]


def test_fourth_down_decision_labels():
    t = data.build_state_table(_toy_game()).sort("play_id")
    assert t["fourth_down_decision"].to_list() == [None, "go", "punt", "field_goal", None]


# ---- checks against the real cache (skipped when it has not been built) ----

# Regular season plus playoffs. 2022 is one short: Bills at Bengals was cancelled.
EXPECTED_GAMES = {**{s: 267 for s in range(2010, 2020)}, 2020: 269, 2021: 285, 2022: 284,
                  2023: 285, 2024: 285, 2025: 285}  # fmt: skip


@pytest.mark.data
def test_game_counts_match_the_schedule(plays):
    counts = dict(plays.group_by("season").agg(pl.col("game_id").n_unique()).iter_rows())
    for season, games in counts.items():
        assert games == EXPECTED_GAMES[season], season


@pytest.mark.data
def test_league_rates_are_in_known_ranges(plays):
    fg = plays.filter(pl.col("play_type") == "field_goal")
    assert 0.82 < (fg["field_goal_result"] == "made").mean() < 0.88
    go = plays.filter(pl.col("fourth_down_decision") == "go")
    assert 0.45 < go["fourth_down_converted"].mean() < 0.58
    scrimmage = plays.filter(pl.col("play_type").is_in(["run", "pass"]))
    assert 5.0 < scrimmage["yards_gained"].mean() < 6.0


@pytest.mark.data
def test_every_game_has_one_winner_label_per_team(plays):
    per_game = (
        plays.filter(pl.col("posteam_won").is_not_null())
        .group_by("game_id", "posteam")
        .agg(pl.col("posteam_won").n_unique().alias("n"), pl.col("posteam_won").first())
    )
    assert per_game["n"].max() == 1
    wins = per_game.group_by("game_id").agg(pl.col("posteam_won").sum())
    assert wins["posteam_won"].unique().to_list() == [1.0]
