"""Tests for the M1 play-outcome models.

The synthetic tests build a small fake play table with known rules, so they run in CI
without the real data. The `data` tests check the fit on the real table against
sensible ranges and are skipped when the cache is absent.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fourth_and_rl import components as cm

YARDS = np.arange(100)


def potential_gain(rng: np.random.Generator, n: int) -> np.ndarray:
    """Yards a synthetic play would gain on an endless field: mostly short, a long tail."""
    short = np.rint(rng.normal(3.0, 4.0, n))
    long = np.rint(rng.exponential(18.0, n)) + 5
    return np.where(rng.random(n) < 0.85, short, long).astype(int)


def synthetic_plays(seed: int = 0, n: int = 30_000) -> pl.DataFrame:
    """Independent fake plays with the next-state columns already filled in.

    Rules: gains do not depend on field position; a gain that reaches the goal line is
    a touchdown recorded with yards_gained = yardline_100; 3% of plays are turnovers;
    field goals get less likely with distance; punts net about 42 yards, less when the
    end zone is close.
    """
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {}

    def add(count: int, **cols: np.ndarray | str | float) -> None:
        base = {
            "play_type": "run", "down": None, "ydstogo": None, "yardline_100": None,
            "yards_gained": None, "first_down": 0.0, "touchdown": 0.0, "td_team": None,
            "interception": 0.0, "fumble_lost": 0.0, "safety": 0.0, "sack": 0.0,
            "field_goal_result": None, "punt_blocked": 0.0, "touchback": 0.0,
            "two_point_attempt": 0.0, "extra_point_result": None, "desc": "",
            "posteam": "A", "defteam": "B", "game_half": "Half1", "season": 2020,
            "half_seconds_remaining": 900.0, "game_seconds_remaining": 2700.0,
            "next_posteam": None, "next_yardline_100": None, "next_half": "Half1",
            "next_row_seconds": 2670.0, "next_row_half": "Half1", "posteam_spread": 0.0,
        }  # fmt: skip
        for name, default in base.items():
            value = cols.get(name, default)
            rows.setdefault(name, []).extend(
                value.tolist() if isinstance(value, np.ndarray) else [value] * count
            )

    # Runs and passes on downs 1 to 4.
    down = rng.integers(1, 5, n)
    yl = rng.integers(1, 100, n)
    togo = np.minimum(rng.integers(1, 16, n), yl)
    gain = np.maximum(potential_gain(rng, n), yl - 100)  # cannot lose more than to the goal
    td = gain >= yl
    gain = np.where(td, yl, gain)
    turnover = (rng.random(n) < 0.03) & ~td
    safety = (yl - gain >= 100) & ~turnover
    first = (gain >= togo) & ~turnover
    keeps = ~turnover & ~td & ~safety & ((down < 4) | first)
    opp_spot = np.clip(100 - (yl - gain), 1, 99)
    add(
        n,
        play_type=np.where(rng.random(n) < 0.6, "pass", "run"),
        down=down.astype(float), ydstogo=togo.astype(float), yardline_100=yl.astype(float),
        yards_gained=gain.astype(float), first_down=first.astype(float),
        touchdown=td.astype(float), td_team=np.where(td, "A", None),
        interception=turnover.astype(float), safety=safety.astype(float),
        next_posteam=np.where(keeps, "A", "B"),
        next_yardline_100=np.where(keeps, np.clip(yl - gain, 1, 99), opp_spot).astype(float),
        season=rng.integers(2018, 2024, n),
    )  # fmt: skip

    # Field goals: P(make) falls with distance.
    k = 4000
    yl = rng.integers(1, 46, k)
    p = 1 / (1 + np.exp(-(3.2 - 0.11 * (yl - 2))))
    add(k, play_type="field_goal", down=4.0, ydstogo=5.0, yardline_100=yl.astype(float),
        field_goal_result=np.where(rng.random(k) < p, "made", "missed"),
        season=rng.integers(2018, 2024, k))  # fmt: skip

    # Punts: net about 42 yards with room, aimed at the 12 when closer; anything
    # reaching the end zone is a touchback.
    k = 6000
    yl = rng.integers(35, 100, k)
    land = 100 - yl + np.rint(rng.normal(np.minimum(42, yl - 12), 7, k)).astype(int)
    touchback = land >= 100
    add(k, play_type="punt", down=4.0, ydstogo=8.0, yardline_100=yl.astype(float),
        touchback=touchback.astype(float), next_posteam="B",
        next_yardline_100=np.where(touchback, 80, np.clip(land, 1, 99)).astype(float))  # fmt: skip

    # Kickoffs (the posteam is the receiving team), extra points and two-point tries.
    k = 500
    add(k, play_type="kickoff", next_posteam="A", season=2025,
        next_yardline_100=np.clip(np.rint(rng.normal(70, 5, k)), 50, 99))  # fmt: skip
    add(400, play_type="extra_point",
        extra_point_result=np.where(rng.random(400) < 0.94, "good", "failed"))  # fmt: skip
    add(200, play_type="pass", two_point_attempt=1.0,
        desc=np.where(rng.random(200) < 0.5, "ATTEMPT SUCCEEDS.", "ATTEMPT FAILS."))  # fmt: skip
    return pl.DataFrame(rows, strict=False)


@pytest.fixture(scope="module")
def synthetic() -> pl.DataFrame:
    return synthetic_plays()


@pytest.fixture(scope="module")
def models(synthetic: pl.DataFrame) -> cm.ComponentModels:
    return cm.ComponentModels().fit(synthetic, seed=1)


def total(outcome: dict, keys: tuple[str, ...]) -> float:
    return float(sum(np.sum(outcome[k]) for k in keys))


GO_KEYS = ("p_touchdown", "first_down", "turnover", "p_safety")
PLAY_KEYS = ("p_touchdown", "p_safety", "keep", "turnover")


# ---- synthetic data: properties that must hold for any fit ----


def test_go_outcomes_sum_to_one_and_split_at_p_convert(models: cm.ComponentModels) -> None:
    for yl in (1, 2, 7, 25, 50, 80, 99):
        for togo in (1, 2, 5, 10, 20):
            if togo > yl:
                continue
            o = models.go_outcomes(yl, togo)
            assert total(o, GO_KEYS) == pytest.approx(1.0, abs=1e-9)
            assert o["p_touchdown"] + o["first_down"].sum() == pytest.approx(
                models.p_convert(yl, togo), abs=1e-9
            )
            assert o["first_down"].shape == (100,) and o["turnover"].shape == (100,)
            assert min(o["first_down"].min(), o["turnover"].min()) >= 0
            # A first down leaves the ball at or past the line to gain.
            assert o["first_down"][yl - togo + 1 :].sum() == 0
            # A stop on downs leaves the other team short of the line to gain.
            assert o["first_down"][0] == 0


def test_goal_to_go_conversion_is_always_a_touchdown(models: cm.ComponentModels) -> None:
    o = models.go_outcomes(3, 3)
    assert o["first_down"].sum() == 0
    assert o["p_touchdown"] == pytest.approx(models.p_convert(3, 3))


def test_conversion_is_non_increasing_in_distance(models: cm.ComponentModels) -> None:
    for yl in (5, 20, 50, 90):
        p = models.p_convert(np.full(30, yl), np.arange(1, 31))
        assert np.all(np.diff(p) <= 1e-12)
        assert np.all((p > 0) & (p < 1))
    assert models.p_convert(50, 1) > models.p_convert(50, 10) + 0.2


def test_conversion_matches_the_synthetic_truth(
    models: cm.ComponentModels, synthetic: pl.DataFrame
) -> None:
    truth = np.mean(potential_gain(np.random.default_rng(5), 200_000) >= 3) * 0.97
    assert models.p_convert(50, 3) == pytest.approx(truth, abs=0.04)


def test_field_goal_is_non_increasing_in_distance(models: cm.ComponentModels) -> None:
    p = models.p_field_goal(np.arange(1, 100))
    assert np.all(np.diff(p) <= 1e-12)
    assert p[0] > 0.9
    assert models.p_field_goal(60) == 0.0  # 78 yards: beyond the longest kick allowed
    assert isinstance(models.p_field_goal(20), float)


def test_missed_field_goal_spot(models: cm.ComponentModels) -> None:
    assert models.fg_miss_yardline(30) == 62  # kicked from the 38: opponent 62 from scoring
    assert models.fg_miss_yardline(13) == 79
    assert models.fg_miss_yardline(12) == 80  # spot would be the 20
    assert models.fg_miss_yardline(3) == 80  # never inside their own 20


def test_punt_outcomes_sum_to_one(models: cm.ComponentModels) -> None:
    for yl in (1, 30, 45, 50, 62, 80, 99):
        p = models.punt_outcomes(yl)
        assert p.shape == (100,)
        assert p.min() >= 0
        assert p.sum() == pytest.approx(1.0, abs=1e-9)
        full = models.punt_outcomes_full(yl)
        assert full["opponent"].sum() + full["kicking_team"].sum() == pytest.approx(1.0)


def test_punting_from_deeper_never_helps(models: cm.ComponentModels) -> None:
    """The receiving team's expected yardline_100 (bigger = farther from scoring) should
    fall as the punt comes from deeper in the punting team's own territory."""
    expected = np.array([models.punt_outcomes(yl) @ YARDS for yl in range(1, 100)])
    spots = expected[[49, 59, 69, 79, 89, 98]]  # yardline_100 = 50, 60, ..., 99
    assert np.all(np.diff(spots) < 0)
    assert np.all(np.diff(expected[39:]) <= 0.75)  # no more than kernel noise in between
    # Synthetic truth: 42 net yards from deep in your own end.
    assert expected[79] - 20 == pytest.approx(42, abs=1.5)


def test_play_outcomes_sum_to_one(models: cm.ComponentModels) -> None:
    for down in (1, 2, 3):
        for togo, yl in ((10, 75), (1, 1), (3, 3), (10, 99), (15, 40), (2, 8)):
            o = models.play_outcomes(down, togo, yl)
            assert total(o, PLAY_KEYS) == pytest.approx(1.0, abs=1e-9)
            assert o["keep"].shape == (100,) and o["turnover"].shape == (100,)
            assert o["keep"][0] == 0
            assert o["p_turnover"] == pytest.approx(0.03, abs=0.02)
    with pytest.raises(ValueError):
        models.play_outcomes(4, 1, 50)


def test_goal_line_truncation(models: cm.ComponentModels) -> None:
    """From k yards out, P(touchdown) must equal the chance of gaining k or more, even
    though plays near the goal line never show their long gains."""
    truth_gain = potential_gain(np.random.default_rng(7), 400_000)
    for yl in (1, 3, 8, 15, 30):
        o = models.play_outcomes(1, min(10, yl), yl)
        truth = np.mean(truth_gain >= yl)
        assert o["p_touchdown"] == pytest.approx(truth, abs=0.05), yl
    # From the 1 the offense either scores, loses the ball or is still on the field.
    o = models.play_outcomes(1, 1, 1)
    assert o["p_safety"] == 0
    assert o["keep"][1:].sum() == pytest.approx(1 - o["p_touchdown"] - o["p_turnover"])
    # Backed up on its own 1, any loss is a safety and no gain is long enough to score
    # except the rare 99-yard play.
    o = models.play_outcomes(1, 10, 99)
    assert o["p_safety"] == pytest.approx(np.mean(truth_gain <= -1) * 0.97, abs=0.05)
    assert o["p_touchdown"] < 0.02


def test_kaplan_meier_undoes_goal_line_censoring() -> None:
    rng = np.random.default_rng(3)
    true_gain = rng.integers(0, 10, 200_000)  # uniform on 0..9
    yardline = rng.integers(1, 6, 200_000)  # every play starts inside the 5
    observed = np.minimum(true_gain, yardline)
    pmf = cm.km_gain_pmf(observed, yardline)
    # Gains 0 to 4 can be seen; each has true probability 0.1.
    assert pmf[cm.GAINS == 0][0] == pytest.approx(0.1, abs=0.005)
    assert pmf[cm.GAINS == 4][0] == pytest.approx(0.1, abs=0.01)
    # Gains of 5+ are never seen from inside the 5: all of that mass is "long".
    assert pmf[-1] == pytest.approx(0.5, abs=0.01)
    assert pmf.sum() == pytest.approx(1.0)
    # The naive reading (share of plays seen to gain 5 or more) would say about 0.1.
    assert np.mean(observed >= 5) < 0.12


def test_kickoff_and_conversions(models: cm.ComponentModels) -> None:
    k = models.kickoff_start()
    assert k.shape == (100,) and k.sum() == pytest.approx(1.0)
    assert k @ YARDS == pytest.approx(70, abs=1.5)
    assert models.p_extra_point() == pytest.approx(0.94, abs=0.04)
    assert models.p_two_point() == pytest.approx(0.5, abs=0.12)
    assert models.seconds_per_play()["mean_scrimmage_play"] == pytest.approx(30.0)


def test_save_and_load_round_trip(models: cm.ComponentModels, tmp_path) -> None:
    path = models.save(tmp_path / "components.joblib")
    loaded = cm.ComponentModels.load(path)
    assert loaded.seed == 1
    assert loaded.p_convert(40, 4) == models.p_convert(40, 4)
    np.testing.assert_array_equal(loaded.punt_outcomes(60), models.punt_outcomes(60))
    np.testing.assert_array_equal(
        loaded.go_outcomes(40, 4)["turnover"], models.go_outcomes(40, 4)["turnover"]
    )


def test_fit_is_deterministic(synthetic: pl.DataFrame, models: cm.ComponentModels) -> None:
    again = cm.ComponentModels().fit(synthetic, seed=99)
    assert again.p_convert(30, 2) == models.p_convert(30, 2)
    np.testing.assert_array_equal(again.kickoff_start(), models.kickoff_start())


def test_add_next_state_finds_the_next_snap() -> None:
    game = pl.DataFrame(
        {
            "game_id": ["g"] * 5,
            "order_sequence": [1.0, 2.0, 3.0, 4.0, 5.0],
            "play_id": [1.0, 2.0, 3.0, 4.0, 5.0],
            "down": [3.0, 4.0, None, 1.0, 2.0],  # row 3 is a timeout with no down
            "yardline_100": [60.0, 58.0, None, 85.0, 80.0],
            "posteam": ["A", "A", None, "B", "B"],
            "game_half": ["Half1"] * 5,
            "game_seconds_remaining": [2000.0, 1960.0, 1950.0, 1950.0, 1915.0],
        }
    )
    out = cm.add_next_state(game)
    assert out["next_posteam"].to_list() == ["A", "B", "B", "B", None]
    assert out["next_yardline_100"].to_list() == [58.0, 85.0, 85.0, 80.0, None]
    assert out["next_row_seconds"].to_list() == [1960.0, 1950.0, 1950.0, 1915.0, None]
    assert cm.add_next_state(out) is out  # already there: returned unchanged


def test_drive_targets_on_a_tiny_game() -> None:
    def row(drive: int, seq: int, play: str, result: str, **extra) -> dict:
        base = {"game_id": "g", "fixed_drive": float(drive), "order_sequence": float(seq),
                "play_id": float(seq), "play_type": play, "fixed_drive_result": result,
                "down": 1.0, "yardline_100": 75.0, "touchdown": 0.0, "td_team": None,
                "posteam": "A", "defteam": "B", "extra_point_result": None,
                "two_point_attempt": 0.0, "field_goal_result": None, "desc": "", "season": 2020}
        return {**base, **extra}  # fmt: skip

    plays = pl.DataFrame(
        [
            row(1, 1, "run", "Touchdown"),
            row(1, 2, "pass", "Touchdown", yardline_100=40.0, touchdown=1.0, td_team="A"),
            row(1, 3, "extra_point", "Touchdown", down=None, extra_point_result="good"),
            row(2, 4, "run", "Punt", yardline_100=80.0, posteam="B", defteam="A"),
            row(2, 5, "punt", "Punt", down=4.0, yardline_100=78.0, posteam="B", defteam="A"),
        ],
        infer_schema_length=None,
    )
    t = cm.drive_targets(plays)
    assert t["n_drives"] == 2
    assert t["outcome_share"]["touchdown"] == 0.5 and t["outcome_share"]["punt"] == 0.5
    assert t["points_per_drive"] == 3.5
    assert t["plays_per_drive"] == 1.5
    assert t["by_start"]["own_21_to_40"]["n_drives"] == 1
    assert t["by_start"]["own_1_to_20"]["n_drives"] == 1


# ---- real data: fitted rates in sensible ranges ----


@pytest.fixture(scope="module")
def real(plays: pl.DataFrame) -> cm.ComponentModels:
    return cm.ComponentModels().fit(plays)


@pytest.mark.data
def test_real_conversion_rates(real: cm.ComponentModels) -> None:
    assert 0.62 < real.p_convert(50, 1) < 0.78
    assert 0.50 < real.p_convert(50, 2) < 0.66
    assert 0.40 < real.p_convert(50, 5) < 0.54
    assert 0.24 < real.p_convert(50, 10) < 0.38
    assert real.p_convert(2, 1) < real.p_convert(50, 1)  # harder at the goal line
    for yl in range(1, 100):
        p = real.p_convert(np.full(30, yl), np.arange(1, 31))
        assert np.all(np.diff(p) <= 1e-12)
    for yl, togo in ((40, 3), (2, 2), (99, 1), (75, 10)):
        assert total(real.go_outcomes(yl, togo), GO_KEYS) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.data
def test_real_field_goal_rates(real: cm.ComponentModels) -> None:
    assert real.field_goal.snap_to_kick_yards == 18
    by_distance = {d: real.p_field_goal(d - 18) for d in (25, 30, 40, 50, 55, 60)}
    assert by_distance[25] > 0.96
    assert 0.92 < by_distance[30] < 0.99
    assert 0.82 < by_distance[40] < 0.93
    assert 0.65 < by_distance[50] < 0.82
    assert 0.50 < by_distance[55] < 0.75
    assert 0.30 < by_distance[60] < 0.65
    assert np.all(np.diff(real.p_field_goal(np.arange(1, 100))) <= 1e-12)
    # Kickers improved: the fitted trend is positive.
    assert real.field_goal.coef_["season_per_decade"] > 0


@pytest.mark.data
def test_real_punts(real: cm.ComponentModels) -> None:
    expected = np.array([real.punt_outcomes(yl) @ YARDS for yl in range(1, 100)])
    net = expected - (100 - np.arange(1, 100))
    assert 40 < net[69] < 47  # from the punting team's own 30
    assert 25 < net[39] < 33  # from the opponent's 40: no room for a full punt
    assert np.all(np.diff(expected[[39, 49, 59, 69, 79, 89, 98]]) < 0)
    assert np.all(np.diff(expected[39:]) <= 0.5)
    assert 0.10 < real.punt_outcomes(55)[80] < 0.25  # touchbacks from near midfield
    assert real.punt_outcomes(70)[0] < 0.02  # return touchdowns are rare
    for yl in range(1, 100):
        assert real.punt_outcomes(yl).sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.data
def test_real_play_model(real: cm.ComponentModels) -> None:
    first_and_ten = real.play_outcomes(1, 10, 75)
    assert 0.17 < first_and_ten["p_first_down"] < 0.27
    assert 0.005 < first_and_ten["p_turnover"] < 0.04
    assert 0.60 < real.play_outcomes(3, 1, 50)["p_first_down"] < 0.80
    assert 0.40 < real.play_outcomes(1, 1, 1)["p_touchdown"] < 0.70
    assert 0.01 < real.play_outcomes(1, 10, 99)["p_safety"] < 0.10
    for down in (1, 2, 3):
        for yl in (1, 5, 20, 50, 80, 99):
            o = real.play_outcomes(down, min(10, yl), yl)
            assert total(o, PLAY_KEYS) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.data
def test_real_kickoff_clock_and_conversions(real: cm.ComponentModels) -> None:
    k = real.kickoff_start()
    assert real.kickoff_info["seasons"] == [2025]
    assert 64 < k @ YARDS < 74  # average start near the receiving team's own 30
    assert 0.92 < real.p_extra_point() < 0.97
    assert 0.40 < real.p_two_point() < 0.56
    assert 20 < real.seconds_per_play()["mean_scrimmage_play"] < 35


@pytest.mark.data
def test_real_drive_targets(plays: pl.DataFrame) -> None:
    t = cm.drive_targets(plays, 2021)
    share = t["outcome_share"]
    assert sum(share.values()) == pytest.approx(1.0)
    assert 0.18 < share["touchdown"] < 0.27
    assert 0.30 < share["punt"] < 0.45
    assert 1.7 < t["points_per_drive"] < 2.4
    assert 5 < t["plays_per_drive"] < 7
    by = t["by_start"]
    assert by["opp_20_to_goal"]["points_per_drive"] > by["own_1_to_20"]["points_per_drive"]
