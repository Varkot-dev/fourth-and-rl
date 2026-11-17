"""M0 data pipeline: download nflverse play-by-play and build the state table.

One row of the state table is one play, described from the point of view of the
team with the ball (the "posteam"). All later milestones read this table.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from fourth_and_rl.paths import CONFIGS, ROOT

# Columns kept from the 372 in the raw nflverse file.
RAW_COLUMNS: list[str] = [
    # identifiers
    "game_id", "play_id", "season", "week", "season_type", "home_team", "away_team",
    "posteam", "defteam", "posteam_type", "drive", "fixed_drive", "fixed_drive_result",
    "order_sequence", "desc",
    # state
    "qtr", "game_half", "down", "ydstogo", "yardline_100", "goal_to_go",
    "half_seconds_remaining", "game_seconds_remaining", "score_differential",
    "posteam_score", "defteam_score", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    # pregame information
    "spread_line", "total_line", "roof", "surface", "temp", "wind", "div_game",
    # what happened
    "play_type", "yards_gained", "first_down", "touchdown", "td_team", "safety",
    "interception", "fumble_lost", "sack", "penalty", "penalty_yards", "rush_attempt",
    "pass_attempt", "qb_kneel", "qb_spike", "field_goal_attempt", "field_goal_result",
    "kick_distance", "punt_attempt", "punt_blocked", "touchback", "return_yards",
    "return_team", "fourth_down_converted", "fourth_down_failed", "two_point_attempt",
    "extra_point_result", "sp", "timeout", "aborted_play",
    # outcomes and the nflfastR baselines
    "result", "home_score", "away_score", "ep", "epa", "wp", "vegas_wp", "home_wp",
    "vegas_home_wp",
]  # fmt: skip


def load_config(path: Path | None = None) -> dict:
    """Read the data config (seasons and cache directories)."""
    with open(path or CONFIGS / "data.yaml") as f:
        return yaml.safe_load(f)


def seasons_from_config(cfg: dict) -> list[int]:
    return list(range(cfg["first_season"], cfg["last_season"] + 1))


def download_raw(seasons: list[int], raw_dir: Path, force: bool = False) -> list[Path]:
    """Download play-by-play for each season and cache the kept columns as Parquet."""
    import nflreadpy as nfl

    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for season in seasons:
        path = raw_dir / f"pbp_{season}.parquet"
        if force or not path.exists():
            nfl.load_pbp(season).select(RAW_COLUMNS).write_parquet(path)
        paths.append(path)
    return paths


def read_raw(raw_dir: Path, seasons: list[int]) -> pl.DataFrame:
    frames = [pl.read_parquet(raw_dir / f"pbp_{s}.parquet") for s in seasons]
    return pl.concat(frames, how="vertical_relaxed")


def build_state_table(raw: pl.DataFrame) -> pl.DataFrame:
    """Turn raw play-by-play into the state table.

    Adds, for every play with a team in possession:
      posteam_won          1.0 win, 0.0 loss, null for a tied game
      is_home              1 if the posteam is the home team
      posteam_spread       points the posteam was favored by before the game
      receive_2h_ko        1 if, in the first half, the posteam will receive the
                           second-half kickoff
      fourth_down_decision "go", "field_goal" or "punt" on fourth downs that were
                           actually run (penalty no-plays are left null)
    """
    df = raw.filter(pl.col("posteam").is_not_null() & (pl.col("posteam") != ""))

    # The team that received the opening kickoff does not receive the second-half one.
    opening = (
        df.filter(pl.col("play_type") == "kickoff")
        .sort("game_id", "order_sequence")
        .group_by("game_id", maintain_order=True)
        .agg(pl.col("posteam").first().alias("opening_receiver"))
    )
    df = df.join(opening, on="game_id", how="left")

    is_home = pl.col("posteam") == pl.col("home_team")
    posteam_margin = pl.when(is_home).then(pl.col("result")).otherwise(-pl.col("result"))
    real_fourth = (
        (pl.col("down") == 4)
        & (pl.col("qb_kneel") != 1)
        & (pl.col("qb_spike") != 1)
    )
    df = df.with_columns(
        is_home.cast(pl.Int8).alias("is_home"),
        pl.when(posteam_margin > 0)
        .then(1.0)
        .when(posteam_margin < 0)
        .then(0.0)
        .otherwise(None)
        .alias("posteam_won"),
        pl.when(is_home)
        .then(pl.col("spread_line"))
        .otherwise(-pl.col("spread_line"))
        .alias("posteam_spread"),
        (
            (pl.col("qtr") <= 2)
            & pl.col("opening_receiver").is_not_null()
            & (pl.col("posteam") != pl.col("opening_receiver"))
        )
        .cast(pl.Int8)
        .alias("receive_2h_ko"),
        pl.when(real_fourth & pl.col("play_type").is_in(["run", "pass"]))
        .then(pl.lit("go"))
        .when(real_fourth & (pl.col("play_type") == "field_goal"))
        .then(pl.lit("field_goal"))
        .when(real_fourth & (pl.col("play_type") == "punt"))
        .then(pl.lit("punt"))
        .otherwise(None)
        .alias("fourth_down_decision"),
    ).drop("opening_receiver")
    return df.sort("season", "game_id", "order_sequence", "play_id")


def build(cfg: dict | None = None, force: bool = False) -> Path:
    """Download (if needed), build and cache the state table. Returns its path."""
    cfg = cfg or load_config()
    seasons = seasons_from_config(cfg)
    download_raw(seasons, ROOT / cfg["raw_dir"], force=force)
    table = build_state_table(read_raw(ROOT / cfg["raw_dir"], seasons))
    out = ROOT / cfg["processed_dir"] / "plays.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out)
    return out


def plays_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return ROOT / cfg["processed_dir"] / "plays.parquet"


def load_plays(cfg: dict | None = None) -> pl.DataFrame:
    """Load the cached state table (run scripts/build_data.py first)."""
    path = plays_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/build_data.py")
    return pl.read_parquet(path)
