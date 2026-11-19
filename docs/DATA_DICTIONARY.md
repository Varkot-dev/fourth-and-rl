# Data dictionary

The state table is `data/processed/plays.parquet`, built by `python scripts/build_data.py`
from nflverse play-by-play (loaded with the `nflreadpy` package). One row is one play,
seen from the side of the team with the ball (the "posteam"). Seasons 2010 to 2025,
regular season and playoffs. The data directory is not committed; the script rebuilds it.

Most columns are copied unchanged from nflverse. Their full definitions are in the
nflverse dictionary: https://nflreadr.nflverse.com/articles/dictionary_pbp.html

## Columns this project adds

| Column | Meaning |
|---|---|
| `posteam_won` | 1.0 if the posteam won the game, 0.0 if it lost, null if the game was tied |
| `is_home` | 1 if the posteam is the home team |
| `posteam_spread` | Pregame point spread from the posteam's side. Positive means the posteam was favored by that many points |
| `receive_2h_ko` | 1 if the play is in the first half and the posteam will receive the second-half kickoff |
| `fourth_down_decision` | `go`, `field_goal` or `punt` for fourth downs that were actually run. Null for other downs, kneels, spikes and plays wiped out by a penalty |

## Columns copied from nflverse, by group

**Identity:** `game_id`, `play_id`, `season`, `week`, `season_type` (REG or POST), `home_team`, `away_team`, `posteam`, `defteam`, `drive`, `order_sequence` (play order within a game), `desc` (text description).

**Game state:** `qtr`, `down`, `ydstogo`, `yardline_100` (yards from the opponent's end zone, 1 to 99), `goal_to_go`, `half_seconds_remaining`, `game_seconds_remaining`, `score_differential` (posteam score minus defteam score, before the play), `posteam_timeouts_remaining`, `defteam_timeouts_remaining`.

**Pregame:** `spread_line` (positive means the home team is favored), `total_line`, `roof`, `surface`, `temp`, `wind`.

**What happened:** `play_type`, `yards_gained`, `first_down`, `touchdown`, `interception`, `fumble_lost`, `field_goal_result` (made, missed, blocked), `kick_distance`, `punt_blocked`, `touchback`, `return_yards`, `fourth_down_converted`, `fourth_down_failed`.

**Outcome:** `result` (home score minus away score at the end), `home_score`, `away_score`.

**nflfastR baselines (not ours):** `wp` (win probability for the posteam, no point spread), `vegas_wp` (win probability using the pregame spread), `home_wp`, `vegas_home_wp`, `ep` (expected points), `epa`.

## Known limits

- Tied games (about 1 per season) have no win label and are left out when fitting win-probability models.
- Fourth-down plays cancelled by a penalty are not labelled as decisions, even when the intended decision is clear from the description.
- `wp` is missing on about 0.6% of rows (mostly plays with no down, such as kickoffs after the final score).
