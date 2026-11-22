"""Tests for records.py: a hand-made game checked field by field, guards, validation."""

from __future__ import annotations

import copy
import os

import numpy as np
import polars as pl
import pytest
import records as R
import schema as S

H, A = "HOM", "AWY"


def _row(order, play_type, posteam, clock, tot=(0, 0), to=(3, 3), half="Half1", qtr=2,
         down=None, ytg=None, yl=None, desc="", nfl="", rush=0, pas=0, two=0):  # fmt: skip
    return {
        "game_id": "2023_01_AWY_HOM", "play_id": float(order * 10), "season": 2023,
        "season_type": "REG", "home_team": H, "away_team": A, "posteam": posteam,
        "order_sequence": float(order), "desc": desc, "qtr": float(qtr), "game_half": half,
        "down": down, "ydstogo": ytg, "yardline_100": yl,
        "game_seconds_remaining": float(clock), "total_home_score": float(tot[0]),
        "total_away_score": float(tot[1]), "home_timeouts_remaining": float(to[0]),
        "away_timeouts_remaining": float(to[1]), "play_type": play_type,
        "play_type_nfl": nfl, "rush": float(rush), "pass": float(pas),
        "two_point_attempt": float(two), "home_score": 15.0, "away_score": 11.0,
    }  # fmt: skip


def synthetic_game() -> pl.DataFrame:
    rows = [
        _row(0, None, None, 1900, qtr=1, desc="GAME", nfl="GAME_START"),
        _row(1, "kickoff", H, 1900, yl=35, desc="X kicks 65 yards from AWY 35"),
        _row(2, "run", H, 1896, tot=(6, 0), down=1, ytg=10, yl=75, rush=1),
        _row(3, "extra_point", H, 1890, tot=(7, 0), yl=15),
        _row(4, "kickoff", A, 1890, tot=(7, 0), yl=35, desc="Y kicks 65 yards"),
        _row(
            5, "no_play", A, 1885, tot=(7, 0), to=(3, 2), desc="Timeout #1 by AWY", nfl="TIMEOUT"
        ),  # fmt: skip
        _row(6, "pass", A, 1885, tot=(7, 0), to=(3, 2), down=1, ytg=10, yl=70, pas=1),
        _row(7, "punt", A, 1850, tot=(7, 0), to=(3, 2), down=4, ytg=3, yl=63),
        _row(
            8,
            "no_play",
            H,
            1842,
            tot=(7, 0),
            to=(3, 2),
            down=1,
            ytg=10,
            yl=80,
            pas=1,
            desc="PENALTY on AWY, Holding - No Play.",
            nfl="PENALTY",
        ),  # fmt: skip
        _row(9, "pass", H, 1838, tot=(13, 0), to=(2, 2), down=1, ytg=15, yl=85, pas=1),
        _row(10, "pass", H, 1838, tot=(15, 0), to=(2, 2), yl=2, pas=1, two=1),
        _row(11, "kickoff", A, 1838, tot=(15, 0), to=(2, 2), yl=35, desc="kicks 65"),
        _row(12, "qb_kneel", A, 1832, tot=(15, 0), to=(2, 2), down=1, ytg=10, yl=75),
        _row(13, None, None, 1800, tot=(15, 0), to=(2, 2), desc="END QUARTER 2"),
        _row(14, "kickoff", H, 1800, tot=(15, 0), half="Half2", qtr=3, yl=35, desc="kicks"),
        _row(
            15, "run", H, 1795, tot=(15, 2), half="Half2", qtr=3, down=1, ytg=10, yl=99, rush=1
        ),  # fmt: skip
        _row(16, "kickoff", A, 1790, tot=(15, 2), half="Half2", qtr=3, yl=20, desc="kicks"),
        _row(
            17, "field_goal", A, 1760, tot=(15, 5), half="Half2", qtr=4, down=4, ytg=5, yl=20
        ),  # fmt: skip
        _row(
            18,
            "kickoff",
            H,
            1760,
            tot=(15, 5),
            half="Half2",
            qtr=4,
            yl=35,
            desc="Z kicks onside 12 yards",
        ),  # fmt: skip
        _row(
            19, "run", A, 600, tot=(15, 11), half="Overtime", qtr=5, down=1, ytg=10, yl=75, rush=1
        ),  # fmt: skip
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


SCHED = pl.DataFrame({"game_id": ["2023_01_AWY_HOM"], "spread_line": [-3.0]})

# phase, poss, down, ytg, yl, clock, diff, to_h, to_a, action
EXPECTED = [
    ("KICKOFF", "HOME", 0, 0, 35, 1900, 0, 3, 3, "KICKOFF"),
    ("PLAY", "HOME", 1, 10, 75, 1896, 0, 3, 3, "RUN"),
    ("TRY", "HOME", 0, 0, 15, 1890, 6, 3, 3, "XP"),
    ("KICKOFF", "AWAY", 0, 0, 35, 1890, 7, 3, 3, "KICKOFF"),
    ("PLAY", "AWAY", 1, 10, 70, 1885, 7, 3, 2, "PASS"),
    ("PLAY", "AWAY", 4, 3, 63, 1850, 7, 3, 2, "PUNT"),
    ("PLAY", "HOME", 1, 10, 80, 1842, 7, 3, 2, "PASS"),
    ("PLAY", "HOME", 1, 15, 85, 1838, 7, 3, 2, "PASS"),
    ("TRY", "HOME", 0, 0, 2, 1838, 13, 2, 2, "TWO_PT"),
    ("KICKOFF", "AWAY", 0, 0, 35, 1838, 15, 2, 2, "KICKOFF"),
    ("PLAY", "AWAY", 1, 10, 75, 1832, 15, 2, 2, "KNEEL"),
    ("KICKOFF", "HOME", 0, 0, 35, 1800, 15, 3, 3, "KICKOFF"),
    ("PLAY", "HOME", 1, 10, 99, 1795, 15, 3, 3, "RUN"),
    ("KICKOFF", "AWAY", 0, 0, 20, 1790, 13, 3, 3, "KICKOFF"),
    ("PLAY", "AWAY", 4, 5, 20, 1760, 13, 3, 3, "FG"),
    ("KICKOFF", "HOME", 0, 0, 35, 1760, 10, 3, 3, "ONSIDE"),
]
# score_delta, elapsed, to_used_home, to_used_away (target of record i)
EXPECTED_T = [
    (0, 4, 0, 0), (6, 6, 0, 0), (1, 0, 0, 0), (0, 5, 0, 1), (0, 35, 0, 0), (0, 8, 0, 0),
    (0, 4, 0, 0), (6, 0, 1, 0), (2, 0, 0, 0), (0, 6, 0, 0), (0, 32, 0, 0), (0, 5, 0, 0),
    (-2, 5, 0, 0), (0, 30, 0, 0), (-3, 0, 0, 0),
]  # fmt: skip


@pytest.fixture(scope="module")
def built():
    return R.build_split(synthetic_game(), SCHED, [2023])


def test_states_by_hand(built):
    split, report = built
    assert len(split.game_ids) == 1 and report["rows_kept"] == len(EXPECTED)
    for i, e in enumerate(EXPECTED):
        d = R.decode_record(split, i)
        got = (d["phase"], d["poss"], d["down"], d["ydstogo"], d["yardline_100"],
               d["clock_seconds"], d["score_diff_home"], d["timeouts_home"],
               d["timeouts_away"], d["action"])  # fmt: skip
        assert got == e, (i, got, e)


def test_targets_by_hand(built):
    split, _ = built
    for i, (dlt, el, uh, ua) in enumerate(EXPECTED_T):
        t = R.decode_record(split, i)["target"]
        nxt = EXPECTED[i + 1]
        assert (t["phase"], t["poss"], t["down"], t["ydstogo"], t["yardline"], t["action"]) == (
            nxt[0],
            nxt[1],
            nxt[2],
            nxt[3],
            nxt[4],
            nxt[9],
        ), i
        assert (t["score_delta"], t["elapsed"], t["to_used_home"], t["to_used_away"]) == (
            dlt,
            el,
            uh,
            ua,
        ), i
    last = split.targets[-1]
    assert last[0] == S.PHASE["END"]
    # the final record (onside kick at 1760 s, diff 10) scores nothing; elapsed clips at 60
    assert last[S.TARGET_INDEX["score_delta"]] == S.SCORE_DELTA_CODE[0]
    assert last[S.TARGET_INDEX["elapsed"]] == S.elapsed_code(1760)
    assert not last[list(R.END_ZERO_FIELDS)].any()


def test_header_final_and_report(built):
    split, report = built
    assert split.header[0, 0] == S.spread_code(-3.0) == 19
    assert split.header[0, 1] == S.POSS_CODE["HOME"]
    assert split.final_diff[0] == 10  # regulation 15-5; the OT score 15-11 is cut
    assert report["dropped"]["timeout_only"] == 1
    assert report["dropped"]["overtime_or_null_qtr"] == 1
    assert report["soft"]["halftime_crossings"] == 1
    assert report["overtime_games_cut"] == 1
    np.testing.assert_array_equal(R.fourth_down_index(split), [5, 14])


def test_validate_passes_and_bookkeeping(built):
    split, _ = built
    v = R.validate_split(split, min_transition_match=1.0)
    assert v["info_mismatched_pairs"] == 0


def test_first_half_snap_at_1800_repaired():
    df = synthetic_game().with_columns(
        pl.when(pl.col("order_sequence") == 12).then(1800.0)
        .otherwise(pl.col("game_seconds_remaining")).alias("game_seconds_remaining")
    )  # fmt: skip
    split, report = R.build_split(df, SCHED, [2023])
    assert report["repairs"]["first_half_clock_1800_set_to_1801"] == 1
    assert split.raw[10, 0] == 1801
    assert R.validate_split(split, 1.0)["info_mismatched_pairs"] == 0


def test_points_on_final_record_are_in_end_target():
    # walk-off: HOME kicks a FG with 3 s left in regulation, then overtime rows follow
    walk = _row(
        18.5, "field_goal", H, 3, tot=(18, 5), half="Half2", qtr=4, down=4, ytg=2, yl=10
    )  # fmt: skip
    df = pl.concat([synthetic_game(), pl.DataFrame([walk], schema=synthetic_game().schema)])
    split, report = R.build_split(df, SCHED, [2023])
    assert split.final_diff[0] == 13 and report["soft"]["games_scoring_on_last_record"] == 1
    d = R.decode_record(split, len(split.states) - 1)
    assert (d["action"], d["score_diff_home"], d["clock_seconds"]) == ("FG", 10, 3)
    assert d["target"] == {"phase": "END", "score_delta": 3, "elapsed": 3}
    v = R.validate_split(split, 0.9)  # the 1760 s -> 3 s jump is one clock mismatch
    assert v["final_diff_ne_last_score_plus_end_delta"] == 0
    assert v["info_games_scoring_on_last_record"] == 1


def test_validate_catches_final_diff_mismatch(built):
    split = copy.deepcopy(built[0])
    split.final_diff[0] += 3  # END target no longer explains the final score
    with pytest.raises(AssertionError, match="final_diff_ne_last_score_plus_end_delta"):
        R.validate_split(split, 1.0)


def test_noplay_punt_and_fg_keep_their_action():
    wiped = {
        7: ("4-J.Doe punts 45 yards to HOM 20. PENALTY on HOM, Holding, enforced - No Play.", 0),
        17: ("1-K.Kik 38 yard field goal is GOOD, NULLIFIED by Penalty - No Play.", 0),
    }  # fmt: skip
    df = synthetic_game().with_columns(
        pl.when(pl.col("order_sequence").is_in([float(k) for k in wiped]))
        .then(pl.lit("no_play")).otherwise(pl.col("play_type")).alias("play_type"),
        pl.col("order_sequence").replace_strict(
            {float(k): v[0] for k, v in wiped.items()},
            default=pl.col("desc"),
            return_dtype=pl.String,
        ).alias("desc"),
    )  # fmt: skip
    split, _ = R.build_split(df, SCHED, [2023])
    acts = [R.decode_record(split, i)["action"] for i in (5, 14)]
    assert acts == ["PUNT", "FG"]
    assert R.decode_record(split, 6)["action"] == "PASS"  # penalty row with pass flag


@pytest.mark.parametrize("bad", [[2026], [2009], [2024, 2026], []])
def test_season_guard(bad):
    with pytest.raises(ValueError):
        R.load_pbp(bad)
    with pytest.raises(ValueError):
        R.load_schedules(bad)
    with pytest.raises(ValueError):
        R.build_split(synthetic_game(), SCHED, bad)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda s: s.states.__setitem__((3, S.STATE_INDEX["down"]), 2),  # down on a kickoff
        lambda s: s.states.__setitem__((1, S.STATE_INDEX["poss"]), 0),  # pad code
        lambda s: s.targets.__setitem__((-1, 0), S.PHASE["PLAY"]),  # no END
        lambda s: s.targets.__setitem__((4, S.TARGET_INDEX["action"]), 99),  # out of vocab
        lambda s: s.offsets.__setitem__(-1, 3),  # offsets do not cover all records
        lambda s: s.raw.__setitem__((5, 1), 40),  # score bookkeeping broken
    ],
)
def test_validate_catches_corruption(built, corrupt):
    split = copy.deepcopy(built[0])
    corrupt(split)
    with pytest.raises(AssertionError):
        R.validate_split(split, min_transition_match=0.99)


def test_disjoint():
    s, _ = R.build_split(synthetic_game(), SCHED, [2023])
    with pytest.raises(AssertionError):
        R.validate_disjoint({"train": s, "val": s})


@pytest.mark.skipif(os.environ.get("WM_REAL_DATA") != "1", reason="set WM_REAL_DATA=1")
def test_real_2023():
    split, report = R.build_split(R.load_pbp([2023]), R.load_schedules([2023]), [2023])
    v = R.validate_split(split)
    assert report["games"] > 250 and v["info_apply_transition_match_fraction"] >= 0.99
