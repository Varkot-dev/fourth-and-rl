"""Turn nflverse play-by-play into world-model records (see schema.py for the contract).

Standalone: imports only numpy, polars, nflreadpy and the stdlib.

nflverse column facts this module relies on (checked on 2022-2023 data, nflreadpy 0.1.5):
- Kickoffs: `posteam` is the RECEIVING team and `yardline_100` is the kick spot measured
  from the receiver's side (35 for a normal kickoff, 20 for a free kick after a safety).
- `total_home_score` / `total_away_score` are POST-play; `posteam_score` is pre-play. The
  pre-play score of a row is therefore the previous row's totals (0 at game start).
- `home_timeouts_remaining` / `away_timeouts_remaining` are POST-play (the timeout row
  itself, and a play row followed by a timeout, already show the decrement). Pre-play
  timeouts are the previous row's values within the same half (3 at the start of a half).
- Two-point tries: `two_point_attempt == 1` with play_type run or pass and null down.
  Penalised tries are play_type no_play with null down.
- Onside kicks: desc contains "onside" (`own_kickoff_recovery` is the result, not the
  attempt).
- Schedules: `spread_line` > 0 means the HOME team is favoured by that many points.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import schema as S

MIN_SEASON, MAX_SEASON = 2010, 2025  # 2026 is the project's sealed holdout: never load it

RECORD_PLAY_TYPES = (
    "run", "pass", "punt", "field_goal", "kickoff", "extra_point", "qb_kneel", "qb_spike",
    "no_play",
)  # fmt: skip

NEEDED_COLUMNS = (
    "game_id", "play_id", "season", "season_type", "home_team", "away_team", "posteam",
    "order_sequence", "desc", "qtr", "game_half", "down", "ydstogo", "yardline_100",
    "game_seconds_remaining", "total_home_score", "total_away_score",
    "home_timeouts_remaining", "away_timeouts_remaining", "play_type", "play_type_nfl",
    "rush", "pass", "two_point_attempt", "home_score", "away_score",
)  # fmt: skip


def _check_seasons(seasons: list[int]) -> None:
    if not seasons:
        raise ValueError("no seasons given")
    bad = [s for s in seasons if s < MIN_SEASON or s > MAX_SEASON]
    if bad:
        raise ValueError(
            f"seasons {bad} outside {MIN_SEASON}..{MAX_SEASON}; 2026 is a sealed holdout"
        )


def load_pbp(seasons: list[int]) -> pl.DataFrame:
    """Raw nflverse play-by-play for the given seasons (2010..2025 only)."""
    seasons = [int(s) for s in seasons]
    _check_seasons(seasons)
    import nflreadpy as nfl

    return nfl.load_pbp(seasons).filter(pl.col("season").is_in(seasons))


def load_schedules(seasons: list[int]) -> pl.DataFrame:
    """nflverse schedules (game_id, spread_line, ...) for the given seasons."""
    seasons = [int(s) for s in seasons]
    _check_seasons(seasons)
    import nflreadpy as nfl

    # nflverse publishes one schedules file for all seasons; keep only the requested ones
    return nfl.load_schedules(seasons).filter(pl.col("season").is_in(seasons))


# ---------------------------------------------------------------- building
def _prepare(pbp: pl.DataFrame) -> pl.DataFrame:
    """Cast the needed columns and add pre-play score and timeouts (full frame)."""
    missing = [c for c in NEEDED_COLUMNS if c not in pbp.columns]
    if missing:
        raise ValueError(f"play-by-play is missing columns {missing}")
    f = pl.Float64
    df = (
        pbp.select(NEEDED_COLUMNS)
        .with_columns(
            pl.col("desc").fill_null(""),
            pl.col("play_type_nfl").fill_null(""),
            *[
                pl.col(c).cast(f)
                for c in (
                    "qtr",
                    "down",
                    "ydstogo",
                    "yardline_100",
                    "game_seconds_remaining",
                    "total_home_score",
                    "total_away_score",
                    "home_timeouts_remaining",
                    "away_timeouts_remaining",
                    "rush",
                    "pass",
                    "two_point_attempt",
                    "order_sequence",
                    "home_score",
                    "away_score",
                )
            ],  # fmt: skip
        )
        .with_columns(pl.col("order_sequence").fill_null(pl.col("play_id").cast(f)))
        .sort("game_id", "order_sequence", "play_id")
    )
    g = "game_id"
    df = df.with_columns(
        pl.col("total_home_score").forward_fill().over(g).fill_null(0),
        pl.col("total_away_score").forward_fill().over(g).fill_null(0),
        pl.col("home_timeouts_remaining").forward_fill().over(g, "game_half"),
        pl.col("away_timeouts_remaining").forward_fill().over(g, "game_half"),
    )
    return df.with_columns(
        pl.col("total_home_score").shift(1).over(g).fill_null(0).alias("pre_home"),
        pl.col("total_away_score").shift(1).over(g).fill_null(0).alias("pre_away"),
        pl.col("home_timeouts_remaining").shift(1).over(g, "game_half").fill_null(3)
        .clip(0, 3).alias("pre_to_home"),
        pl.col("away_timeouts_remaining").shift(1).over(g, "game_half").fill_null(3)
        .clip(0, 3).alias("pre_to_away"),
    )  # fmt: skip


def _drop(df: pl.DataFrame, keep: pl.Expr, reason: str, dropped: dict) -> pl.DataFrame:
    out = df.filter(keep.fill_null(False))
    dropped[reason] = dropped.get(reason, 0) + df.height - out.height
    return out


def _classify(df: pl.DataFrame) -> pl.DataFrame:
    """Add phase and action codes."""
    P, A = S.PHASE, S.ACTION
    pt = pl.col("play_type")
    desc = pl.col("desc")
    two_pt = pl.col("two_point_attempt") == 1
    noplay_nodown = (pt == "no_play") & pl.col("down").is_null()
    noplay_kick = noplay_nodown & desc.str.contains("(?i) kicks ")
    noplay_try = noplay_nodown & ~noplay_kick
    onside = desc.str.contains("(?i)onside")
    phase = (
        pl.when(two_pt | (pt == "extra_point") | noplay_try).then(P["TRY"])
        .when((pt == "kickoff") | noplay_kick).then(P["KICKOFF"])
        .otherwise(P["PLAY"])
    )  # fmt: skip
    action = (
        pl.when(two_pt).then(A["TWO_PT"])
        .when(pt == "extra_point").then(A["XP"])
        .when(((pt == "kickoff") | noplay_kick) & onside).then(A["ONSIDE"])
        .when((pt == "kickoff") | noplay_kick).then(A["KICKOFF"])
        .when(noplay_try & desc.str.contains("(?i)two-point")).then(A["TWO_PT"])
        .when(noplay_try & desc.str.contains("(?i)kick formation|extra point")).then(A["XP"])
        .when(noplay_try).then(A["NOPLAY"])
        .when(pt == "punt").then(A["PUNT"])
        .when(pt == "field_goal").then(A["FG"])
        .when(pt == "qb_kneel").then(A["KNEEL"])
        .when(pt == "qb_spike").then(A["SPIKE"])
        # dropbacks (incl. sacks and scrambles) are PASS, designed runs RUN
        .when(pl.col("pass") == 1).then(A["PASS"])
        .when(pl.col("rush") == 1).then(A["RUN"])
        .when(pt == "pass").then(A["PASS"])
        .when(pt == "run").then(A["RUN"])
        # a punt or field goal wiped out by a penalty is still the punt / FG decision
        .when((pt == "no_play") & desc.str.contains(" punts ")).then(A["PUNT"])
        .when((pt == "no_play") & desc.str.contains("(?i)field goal is")).then(A["FG"])
        .otherwise(A["NOPLAY"])
    )  # fmt: skip
    return df.with_columns(
        phase.cast(pl.Int16).alias("phase"),
        action.cast(pl.Int16).alias("action"),
        noplay_try.alias("_noplay_try"),
    )


def _select_records(df: pl.DataFrame, dropped: dict, repairs: dict) -> pl.DataFrame:
    pt = pl.col("play_type")
    desc = pl.col("desc")
    df = _drop(df, pl.col("qtr") <= 4, "overtime_or_null_qtr", dropped)
    df = _drop(df, pt.is_in(RECORD_PLAY_TYPES), "not_a_snap_or_kick", dropped)
    df = _drop(df, pl.col("posteam").is_not_null(), "null_posteam", dropped)
    timeout_only = (pt == "no_play") & (
        (pl.col("play_type_nfl") == "TIMEOUT") | desc.str.contains(r"^(\(\d+:\d+\) )?Timeout #")
    )
    df = _drop(df, ~timeout_only, "timeout_only", dropped)
    admin = desc.str.contains("(?i)two-minute warning|end of quarter|end quarter|end of half")
    df = _drop(df, ~(admin & (pt == "no_play")), "two_minute_or_end_of_quarter", dropped)
    df = _drop(df, pl.col("game_seconds_remaining").is_not_null(), "null_clock", dropped)
    df = _drop(df, pl.col("yardline_100").is_not_null(), "null_yardline", dropped)
    df = _classify(df)
    is_play = pl.col("phase") == S.PHASE["PLAY"]
    df = _drop(df, ~is_play | pl.col("down").is_between(1, 4), "play_without_down", dropped)
    df = _drop(df, ~is_play | pl.col("ydstogo").is_not_null(), "play_without_ydstogo", dropped)
    repairs["noplay_null_down_as_try"] = int(df["_noplay_try"].sum())
    yl = pl.col("yardline_100")
    repairs["yardline_clipped_1_99"] = int(df.select(((yl < 1) | (yl > 99)).sum()).item())
    ytg = pl.col("ydstogo")
    repairs["ydstogo_clipped_1_30"] = int(
        df.filter(is_play).select(((ytg < 1) | (ytg > S.MAX_YDSTOGO)).sum()).item()
    )
    clk = pl.col("game_seconds_remaining")
    repairs["clock_clipped_0_3600"] = int(df.select(((clk < 0) | (clk > 3600)).sum()).item())
    # schema.apply_transition resets timeouts when the clock reaches 1800, so a
    # first-half snap at 0:00 of Q2 (try after a TD at the gun, untimed down) would look
    # like the start of the second half. Store it at 1801 s: same clock code (bin 361).
    h1_at_half = (pl.col("game_half") == "Half1") & (clk <= S.HALF_SECONDS)
    repairs["first_half_clock_1800_set_to_1801"] = int(df.select(h1_at_half.sum()).item())
    df = df.with_columns(
        pl.when(h1_at_half).then(S.HALF_SECONDS + 1).otherwise(clk).alias("game_seconds_remaining")
    )
    poss = (
        pl.when(pl.col("posteam") == pl.col("home_team")).then(S.POSS_CODE["HOME"])
        .when(pl.col("posteam") == pl.col("away_team")).then(S.POSS_CODE["AWAY"])
        .otherwise(None)
    )  # fmt: skip
    df = df.with_columns(poss.cast(pl.Int16).alias("poss"))
    df = _drop(df, pl.col("poss").is_not_null(), "posteam_not_home_or_away", dropped)
    return df.with_columns(
        pl.when(is_play).then(pl.col("down")).otherwise(0).cast(pl.Int16).alias("down_c"),
        pl.when(is_play).then(ytg.clip(1, S.MAX_YDSTOGO)).otherwise(0).cast(pl.Int16)
        .alias("ytg_c"),
        yl.clip(1, 99).cast(pl.Int16).alias("yl_c"),
        clk.clip(0, S.GAME_SECONDS).cast(pl.Int32).alias("clock_s"),
        (pl.col("pre_home") - pl.col("pre_away")).cast(pl.Int32).alias("diff"),
        pl.col("pre_to_home").cast(pl.Int32).alias("to_h"),
        pl.col("pre_to_away").cast(pl.Int32).alias("to_a"),
    )  # fmt: skip


# target fields that are 0 on a game's last record (its END target)
END_ZERO_FIELDS = tuple(
    S.TARGET_INDEX[f] for f, _ in S.TARGET_FIELDS if f not in ("phase", "score_delta", "elapsed")
)


def _nearest_delta(d: np.ndarray) -> np.ndarray:
    legal = np.asarray(S.SCORE_DELTAS[1:])
    return legal[np.abs(d[:, None] - legal[None, :]).argmin(axis=1)]


def build_split(
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    seasons: list[int],
    season_types: tuple[str, ...] = ("REG", "POST"),
    illegal_delta: str = "drop_game",
) -> tuple[S.Split, dict]:
    """Records for the given seasons as a schema.Split, plus a report of counts.

    illegal_delta: what to do when a transition's points change is not in
    schema.SCORE_DELTAS: "drop_game" (default) drops the whole game, "nearest" maps it to
    the nearest legal value (the bookkeeping check then fails on that pair).
    """
    seasons = [int(s) for s in seasons]
    _check_seasons(seasons)
    dropped: dict[str, int] = {}
    repairs: dict[str, int] = {}
    soft: dict[str, int] = {}
    raw = pbp.filter(
        pl.col("season").is_in(seasons) & pl.col("season_type").is_in(list(season_types))
    )
    df = _prepare(raw)
    rows_in = df.height

    # end-of-regulation score: post-play totals of the last regulation row
    reg_end = (
        df.filter(pl.col("qtr") <= 4)
        .group_by("game_id", maintain_order=True)
        .agg(
            (pl.col("total_home_score").last() - pl.col("total_away_score").last())
            .cast(pl.Int32).alias("final_diff"),
            (pl.col("home_score").first() - pl.col("away_score").first()).alias("box_diff"),
            (pl.col("qtr").max() <= 4).alias("no_ot_rows"),
        )
    )  # fmt: skip
    ot_games = set(df.filter(pl.col("qtr") >= 5)["game_id"].unique().to_list())
    reg_end = reg_end.with_columns(pl.col("game_id").is_in(list(ot_games)).alias("ot"))

    rec = _select_records(df, dropped, repairs)

    # ---- transitions (numpy over the sorted record table)
    gid = rec["game_id"].to_numpy()
    n = len(gid)
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]]) if n else np.zeros(0, int)
    last = np.zeros(n, bool)
    if n:
        last[np.r_[starts[1:] - 1, n - 1]] = True
    nxt = np.minimum(np.arange(n) + 1, max(n - 1, 0))

    phase = rec["phase"].to_numpy().astype(np.int16)
    poss = rec["poss"].to_numpy().astype(np.int16)
    down = rec["down_c"].to_numpy().astype(np.int16)
    ytg = rec["ytg_c"].to_numpy().astype(np.int16)
    yl = rec["yl_c"].to_numpy().astype(np.int16)
    action = rec["action"].to_numpy().astype(np.int16)
    clock = rec["clock_s"].to_numpy().astype(np.int32)
    diff = rec["diff"].to_numpy().astype(np.int32)
    to_h = rec["to_h"].to_numpy().astype(np.int32)
    to_a = rec["to_a"].to_numpy().astype(np.int32)
    half = rec["game_half"].to_numpy()
    within = ~last

    # points on a game's final record (FG as time expires, TD or try at 0:00) go in its
    # END target: final regulation diff minus the last record's pre-play diff
    fd_map = dict(reg_end.select("game_id", "final_diff").iter_rows())
    end_fd = np.array([fd_map[g] for g in gid[last]], dtype=np.int32)
    delta = diff[nxt] - diff
    delta[last] = end_fd - diff[last]
    soft["games_scoring_on_last_record"] = int((delta[last] != 0).sum())
    legal = np.isin(delta, S.SCORE_DELTAS[1:])
    bad_games = set(gid[~legal].tolist())
    soft["score_delta_illegal_pairs"] = int((~legal).sum())
    if illegal_delta == "nearest":
        soft["score_delta_mapped"] = int((~legal).sum())
        delta = np.where(legal, delta, _nearest_delta(delta))
        bad_games = set()
    elapsed = clock - clock[nxt]
    elapsed[last] = clock[last]  # regulation ends at 0 seconds
    soft["elapsed_negative"] = int((within & (elapsed < 0)).sum())
    soft["elapsed_over_60"] = int((within & (elapsed > S.MAX_ELAPSED)).sum())
    soft["elapsed_over_45"] = int((within & (elapsed > 45)).sum())
    crossing = within & (half == "Half1") & (half[nxt] == "Half2")
    soft["halftime_crossings"] = int(crossing.sum())
    used_h = np.where(crossing, 0, np.maximum(0, to_h - to_h[nxt]))
    used_a = np.where(crossing, 0, np.maximum(0, to_a - to_a[nxt]))
    soft["timeout_increase_mid_half"] = int(
        (within & ~crossing & ((to_h[nxt] > to_h) | (to_a[nxt] > to_a))).sum()
    )
    soft["timeouts_used_over_2"] = int((within & ((used_h > 2) | (used_a > 2))).sum())

    T = S.TARGET_INDEX
    targets = np.zeros((n, len(S.TARGET_FIELDS)), np.int16)
    targets[:, T["phase"]] = np.where(last, S.PHASE["END"], phase[nxt])
    targets[:, T["poss"]] = poss[nxt]
    targets[:, T["score_delta"]] = [S.SCORE_DELTA_CODE.get(int(d), 0) for d in delta]
    targets[:, T["elapsed"]] = S.elapsed_code(elapsed)
    targets[:, T["to_used_home"]] = np.clip(used_h, 0, 2) + 1
    targets[:, T["to_used_away"]] = np.clip(used_a, 0, 2) + 1
    targets[:, T["down"]] = down[nxt]
    targets[:, T["ydstogo"]] = ytg[nxt]
    targets[:, T["yardline"]] = yl[nxt]
    targets[:, T["action"]] = action[nxt]
    # END target: phase, score_delta and elapsed are real; every other field is 0 (masked)
    targets[np.ix_(last, list(END_ZERO_FIELDS))] = 0

    SI = S.STATE_INDEX
    states = np.zeros((n, len(S.STATE_FIELDS)), np.int16)
    states[:, SI["phase"]] = phase
    states[:, SI["poss"]] = poss
    states[:, SI["down"]] = down
    states[:, SI["ydstogo"]] = ytg
    states[:, SI["yardline"]] = yl
    states[:, SI["clock"]] = S.clock_code(clock)
    states[:, SI["score"]] = S.score_code(diff)
    states[:, SI["to_home"]] = S.timeout_code(to_h)
    states[:, SI["to_away"]] = S.timeout_code(to_a)
    states[:, SI["action"]] = action
    raw_cols = np.stack([clock, diff, to_h, to_a], axis=1).astype(np.int32)
    play_ids = rec["play_id"].cast(pl.Float64).to_numpy().astype(np.int64)
    rec_season = rec["season"].to_numpy()

    # ---- drop games with illegal score deltas
    keep = ~np.isin(gid, list(bad_games)) if bad_games else np.ones(n, bool)
    dropped["games_with_illegal_score_delta"] = len(bad_games)
    dropped["records_in_dropped_games"] = int((~keep).sum())
    states, raw_cols, targets = states[keep], raw_cols[keep], targets[keep]
    play_ids, gid, rec_season = play_ids[keep], gid[keep], rec_season[keep]
    m = len(gid)
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]]) if m else np.zeros(0, int)
    offsets = np.r_[starts, m].astype(np.int64)
    game_ids = [str(x) for x in gid[starts]]
    seasons_arr = rec_season[starts].astype(np.int16)

    # ---- header
    games = pl.DataFrame({"game_id": game_ids, "_g": np.arange(len(game_ids))})
    sched = schedules.select("game_id", pl.col("spread_line").cast(pl.Float64))
    games = games.join(sched, on="game_id", how="left").sort("_g")
    spread = games["spread_line"].to_numpy()
    soft["games_missing_spread"] = int(np.isnan(spread).sum()) if len(spread) else 0
    spread = np.nan_to_num(spread, nan=0.0)
    header = np.zeros((len(game_ids), len(S.HEADER_FIELDS)), np.int16)
    header[:, 0] = S.spread_code(spread)
    opening = np.zeros(len(game_ids), np.int16)
    soft["games_first_record_not_kickoff"] = 0
    for g in range(len(game_ids)):
        s, e = offsets[g], offsets[g + 1]
        ko = np.flatnonzero(states[s:e, SI["phase"]] == S.PHASE["KICKOFF"])
        if len(ko) == 0 or ko[0] != 0:
            soft["games_first_record_not_kickoff"] += 1
        opening[g] = states[s + ko[0], SI["poss"]] if len(ko) else states[s, SI["poss"]]
    header[:, 1] = opening

    # ---- final regulation score
    fin = games.join(reg_end, on="game_id", how="left").sort("_g")
    final_diff = fin["final_diff"].to_numpy().astype(np.int16)
    no_ot = ~fin["ot"].to_numpy().astype(bool)
    box = fin["box_diff"].to_numpy()
    soft["final_diff_vs_box_mismatch_non_ot"] = int((no_ot & (final_diff != box)).sum())

    split = S.Split(
        states=states,
        raw=raw_cols,
        targets=targets,
        offsets=offsets,
        header=header,
        final_diff=final_diff,
        game_ids=game_ids,
        seasons=seasons_arr,
        play_ids=play_ids,
    )
    sizes = np.diff(offsets)
    report = {
        "seasons": seasons,
        "rows_in": rows_in,
        "rows_kept": int(m),
        "dropped": dropped,
        "repairs": repairs,
        "soft": soft,
        "games": len(game_ids),
        "overtime_games_cut": int(np.sum(~no_ot)),
        "records_per_game": {
            "min": int(sizes.min()) if len(sizes) else 0,
            "mean": round(float(sizes.mean()), 1) if len(sizes) else 0.0,
            "max": int(sizes.max()) if len(sizes) else 0,
        },
    }
    return split, report


# ---------------------------------------------------------------- validation
def _pair_mask(split: S.Split) -> np.ndarray:
    """True for record i when i+1 is in the same game."""
    n = len(split.states)
    m = np.ones(n, bool)
    if n:
        m[split.offsets[1:] - 1] = False
    return m


def transition_mismatches(split: S.Split) -> dict:
    """Within-game pairs where apply_transition does not reproduce raw[i+1], by reason."""
    pair = _pair_mask(split)
    idx = np.flatnonzero(pair)
    pred = S.apply_transition(split.raw[idx], split.targets[idx])
    true = split.raw[idx + 1]
    bad = pred != true
    out = {"pairs": int(len(idx)), "mismatched_pairs": int(bad.any(axis=1).sum())}
    for j, name in enumerate(S.RAW_COLUMNS):
        out[f"mismatch_{name}"] = int(bad[:, j].sum())
    any_bad = bad.any(axis=1)
    el = split.raw[idx, 0] - split.raw[idx + 1, 0]
    half_at_1800 = (
        (split.raw[idx, 0] == S.HALF_SECONDS)
        & (split.states[idx + 1, S.STATE_INDEX["phase"]] == S.PHASE["KICKOFF"])
        & (split.raw[idx + 1, 0] == S.HALF_SECONDS)
    )
    reasons = {
        "reason_elapsed_over_60": any_bad & (el > S.MAX_ELAPSED),
        "reason_elapsed_negative": any_bad & (el < 0),
        "reason_untimed_down_at_1800_before_halftime": any_bad & half_at_1800,
    }
    explained = np.zeros(len(idx), bool)
    for k, v in reasons.items():
        out[k] = int((v & ~explained).sum())
        explained |= v
    out["reason_other"] = int((any_bad & ~explained).sum())
    return out


def validate_split(split: S.Split, min_transition_match: float = 0.99) -> dict:
    """Invariant checks. Returns check -> violations; raises AssertionError on hard ones."""
    st, tg, off = split.states, split.targets, split.offsets
    n, G = len(st), len(split.game_ids)
    hard: dict[str, int] = {}
    for j, (name, size) in enumerate(S.STATE_FIELDS):
        hard[f"state_{name}_out_of_vocab"] = int(((st[:, j] < 0) | (st[:, j] >= size)).sum())
    for j, (name, size) in enumerate(S.TARGET_FIELDS):
        hard[f"target_{name}_out_of_vocab"] = int(((tg[:, j] < 0) | (tg[:, j] >= size)).sum())
    for j, (name, size) in enumerate(S.HEADER_FIELDS):
        col = split.header[:, j]
        hard[f"header_{name}_out_of_vocab"] = int(((col <= 0) | (col >= size)).sum())
    SI, T = S.STATE_INDEX, S.TARGET_INDEX
    never_zero = ("phase", "poss", "yardline", "clock", "score", "to_home", "to_away", "action")
    hard["state_pad_codes"] = int(sum((st[:, SI[f]] == 0).sum() for f in never_zero))
    hard["state_phase_end_or_pad"] = int(np.isin(st[:, SI["phase"]], [0, S.PHASE["END"]]).sum())
    is_play = st[:, SI["phase"]] == S.PHASE["PLAY"]
    hard["down_zero_iff_not_play"] = int(((st[:, SI["down"]] == 0) == is_play).sum())
    hard["ydstogo_zero_iff_not_play"] = int(((st[:, SI["ydstogo"]] == 0) == is_play).sum())
    hard["offsets_bad"] = int(
        len(off) != G + 1 or (G >= 0 and (off[0] != 0 or off[-1] != n))
        or bool(np.any(np.diff(off) <= 0))
    )  # fmt: skip
    if hard["offsets_bad"]:
        raise AssertionError("validate_split failed: offsets do not cover the records")
    pair = _pair_mask(split)
    is_end = tg[:, T["phase"]] == S.PHASE["END"]
    hard["end_not_exactly_last"] = int((is_end != ~pair).sum())
    end_tg = tg[~pair]
    hard["last_target_nonzero_fields"] = int((end_tg[:, list(END_ZERO_FIELDS)] != 0).sum())
    hard["end_target_pad_score_or_elapsed"] = int(
        ((end_tg[:, T["score_delta"]] == 0) | (end_tg[:, T["elapsed"]] == 0)).sum()
    )
    # the last record's score plus its END score delta is the final regulation score
    end_score = split.raw[~pair, 1] + S.score_delta_value(end_tg[:, T["score_delta"]])
    fd_bad = end_score != split.final_diff.astype(np.int32)
    hard["final_diff_ne_last_score_plus_end_delta"] = int(fd_bad.sum())
    pad_tg = tg[pair][
        :,
        [
            T[f]
            for f in (
                "phase",
                "poss",
                "score_delta",
                "elapsed",
                "to_used_home",
                "to_used_away",
                "yardline",
                "action",
            )
        ],
    ]
    hard["target_pad_codes"] = int((pad_tg == 0).sum())
    # target categorical fields equal the next record's fields (exact by construction)
    idx = np.flatnonzero(pair)
    nxt = S.next_state_codes(split.raw[idx + 1], tg[idx])
    hard["target_ne_next_state"] = int((nxt != st[idx + 1]).any(axis=1).sum())
    hard["duplicate_game_ids"] = G - len(set(split.game_ids))
    tm = transition_mismatches(split)
    frac = 1 - tm["mismatched_pairs"] / max(tm["pairs"], 1)
    hard["apply_transition_below_threshold"] = int(frac < min_transition_match)
    report = {**hard, **{f"info_{k}": v for k, v in tm.items()}}
    report["info_apply_transition_match_fraction"] = round(frac, 5)
    report["info_final_diff_mismatch_by_season"] = {
        int(y): int(fd_bad[split.seasons == y].sum()) for y in np.unique(split.seasons)
    }
    report["info_games_scoring_on_last_record"] = int(
        (S.score_delta_value(end_tg[:, T["score_delta"]]) != 0).sum()
    )
    sd = tg[pair, T["score_delta"]]
    report["soft_elapsed_over_45"] = int((tg[pair, T["elapsed"]] - 1 > 45).sum())
    report["soft_nonzero_score_deltas"] = int((S.score_delta_value(sd) != 0).sum())
    failed = {k: v for k, v in hard.items() if v}
    if failed:
        raise AssertionError(f"validate_split failed: {failed}")
    return report


def validate_disjoint(splits: dict[str, S.Split]) -> dict:
    """No game id in two splits. Raises AssertionError if any is shared."""
    seen: dict[str, str] = {}
    shared = 0
    for name, sp in splits.items():
        for g in sp.game_ids:
            if g in seen and seen[g] != name:
                shared += 1
            seen[g] = name
    if shared:
        raise AssertionError(f"{shared} game ids appear in more than one split")
    return {"shared_game_ids": 0}


# ---------------------------------------------------------------- inspection
def _game_of(split: S.Split, i: int) -> int:
    return int(np.searchsorted(split.offsets, i, side="right") - 1)


def decode_record(split: S.Split, i: int) -> dict:
    """Human-readable state, action and transition target of record i."""
    st, tg, rw = split.states[i], split.targets[i], split.raw[i]
    SI, T = S.STATE_INDEX, S.TARGET_INDEX
    g = _game_of(split, i)
    is_end = int(tg[T["phase"]]) == S.PHASE["END"]
    target = {
        "phase": S.PHASES[tg[T["phase"]]],
        "score_delta": int(S.SCORE_DELTAS[tg[T["score_delta"]]]),
        "elapsed": int(tg[T["elapsed"]]) - 1,
    }
    if not is_end:
        target.update(
            poss=S.POSS[tg[T["poss"]]],
            score_delta=int(S.SCORE_DELTAS[tg[T["score_delta"]]]),
            elapsed=int(tg[T["elapsed"]]) - 1,
            to_used_home=int(tg[T["to_used_home"]]) - 1,
            to_used_away=int(tg[T["to_used_away"]]) - 1,
            down=int(tg[T["down"]]),
            ydstogo=int(tg[T["ydstogo"]]),
            yardline=int(tg[T["yardline"]]),
            action=S.ACTIONS[tg[T["action"]]],
        )
    return {
        "game_id": split.game_ids[g],
        "play_id": int(split.play_ids[i]),
        "record_in_game": int(i - split.offsets[g]),
        "phase": S.PHASES[st[SI["phase"]]],
        "poss": S.POSS[st[SI["poss"]]],
        "down": int(st[SI["down"]]),
        "ydstogo": int(st[SI["ydstogo"]]),
        "yardline_100": int(st[SI["yardline"]]),
        "clock_seconds": int(rw[0]),
        "score_diff_home": int(rw[1]),
        "timeouts_home": int(rw[2]),
        "timeouts_away": int(rw[3]),
        "action": S.ACTIONS[st[SI["action"]]],
        "target": target,
    }


def fourth_down_index(split: S.Split) -> np.ndarray:
    """Records that are real fourth-down decisions: PLAY, down 4, RUN/PASS/PUNT/FG."""
    st = split.states
    SI, A = S.STATE_INDEX, S.ACTION
    ok = (
        (st[:, SI["phase"]] == S.PHASE["PLAY"])
        & (st[:, SI["down"]] == 4)
        & np.isin(st[:, SI["action"]], [A["RUN"], A["PASS"], A["PUNT"], A["FG"]])
    )
    return np.flatnonzero(ok)
