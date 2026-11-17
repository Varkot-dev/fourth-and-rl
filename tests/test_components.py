"""Tests for the M1 play-outcome models.

The synthetic tests build a small fake play table with known rules, so they run in CI
without the real data. The `data` tests check the fit on the real table against
sensible ranges and are skipped when the cache is absent.

Every synthetic season is a development season (2018 to 2023): the fits refuse test and
holdout seasons, and the leakage tests at the end check exactly that.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fourth_and_rl import components as cm
from fourth_and_rl import splits

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
            "game_id": "2020_synthetic",
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
    add(k, play_type="kickoff", next_posteam="A", season=2023,
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
    assert models.p_field_goal(60) == 0.0  # 78 yards: beyond the longest made kick
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
    assert loaded.stamp == models.stamp and loaded.stamp["seasons"] == list(range(2018, 2024))
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
def dev_plays(plays: pl.DataFrame) -> pl.DataFrame:
    return plays.filter(pl.col("season").is_in(list(splits.dev_seasons())))


@pytest.fixture(scope="module")
def real(dev_plays: pl.DataFrame) -> cm.ComponentModels:
    return cm.ComponentModels().fit(dev_plays)


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
    assert real.kickoff_info["seasons"] == [2018, 2019, 2020, 2021, 2022, 2023]
    assert 70 < k @ YARDS < 78  # touchback at the 25: average start near the own 25
    assert 0.92 < real.p_extra_point() < 0.97
    assert 0.40 < real.p_two_point() < 0.56
    assert 20 < real.seconds_per_play()["mean_scrimmage_play"] < 35


@pytest.mark.data
def test_real_drive_targets(dev_plays: pl.DataFrame) -> None:
    t = cm.drive_targets(dev_plays, 2021)
    share = t["outcome_share"]
    assert sum(share.values()) == pytest.approx(1.0)
    assert 0.18 < share["touchdown"] < 0.27
    assert 0.30 < share["punt"] < 0.45
    assert 1.7 < t["points_per_drive"] < 2.4
    assert 5 < t["plays_per_drive"] < 7
    by = t["by_start"]
    assert by["opp_20_to_goal"]["points_per_drive"] > by["own_1_to_20"]["points_per_drive"]


# ---- leakage barriers (docs/research/leakage_audit.md) ----


@pytest.fixture()
def access_log(tmp_path, monkeypatch) -> object:
    """Send the test-access log to a temporary file so tests leave no trace in results/."""
    log = tmp_path / "access.jsonl"
    monkeypatch.setattr(splits, "ACCESS_LOG", log)
    return log


def with_season(plays: pl.DataFrame, season: int) -> pl.DataFrame:
    return plays.with_columns(pl.lit(season).cast(plays["season"].dtype).alias("season"))


FITS = {
    "bundle": lambda df, **kw: cm.ComponentModels().fit(df, **kw),
    "go": lambda df, **kw: cm.GoModel().fit(df, **kw),
    "field_goal": lambda df, **kw: cm.FieldGoalModel().fit(df, **kw),
    "punt": lambda df, **kw: cm.PuntModel().fit(df, **kw),
    "play": lambda df, **kw: cm.PlayModel().fit(df, **kw),
    "kickoff": lambda df, **kw: cm.fit_kickoff(df, None, **kw),
    "seconds": lambda df, **kw: cm.fit_seconds(df, None, **kw),
    "conversions": lambda df, **kw: cm.fit_conversions(df, None, **kw),
}


@pytest.mark.parametrize("name", list(FITS))
def test_every_fit_refuses_test_and_holdout_seasons(
    name: str, synthetic: pl.DataFrame, access_log
) -> None:
    test_rows = with_season(synthetic, splits.test_seasons()[0])
    with pytest.raises(splits.LeakageError):
        FITS[name](test_rows)
    with pytest.raises(splits.LeakageError):  # the exception needs a written reason
        FITS[name](test_rows, allow_test_seasons=True)
    held = with_season(synthetic, splits.holdout_seasons()[0])
    with pytest.raises(splits.LeakageError):  # the final holdout has no exception at all
        FITS[name](held, allow_test_seasons=True, reason="a long enough written reason")
    assert not access_log.exists()


def test_allowed_test_season_fit_is_logged_and_stamped(synthetic: pl.DataFrame, access_log) -> None:
    rows = with_season(synthetic, splits.test_seasons()[-1])
    models = cm.ComponentModels().fit(
        rows, allow_test_seasons=True, reason="unit test of the logged exception path"
    )
    assert access_log.read_text().count("fit_on_test_seasons") >= 1
    assert models.stamp["test_seasons_allowed"] and models.fitted_on_test_seasons()
    with pytest.raises(splits.LeakageError):  # cannot be scored on the season it saw
        splits.assert_can_score(models.stamp, rows, "unit test")


def test_stamp_allows_scoring_only_unseen_seasons(
    models: cm.ComponentModels, synthetic: pl.DataFrame
) -> None:
    assert not models.fitted_on_test_seasons()
    assert models.stamp["config_hash"] == cm.config_hash(models.config)
    splits.assert_can_score(models.stamp, with_season(synthetic, 2024), "unit test")
    with pytest.raises(splits.LeakageError):
        splits.assert_can_score(models.stamp, synthetic, "unit test")


def test_default_paths_keep_the_two_bundles_apart(
    models: cm.ComponentModels, synthetic: pl.DataFrame, tmp_path, monkeypatch, access_log
) -> None:
    dev, everything = tmp_path / "components_dev.joblib", tmp_path / "components_all.joblib"
    monkeypatch.setattr(cm, "DEV_MODEL_PATH", dev)
    monkeypatch.setattr(cm, "ALL_MODEL_PATH", everything)
    with pytest.raises(FileNotFoundError):
        cm.ComponentModels.load()  # no silent fallback to some other bundle
    assert models.save() == dev
    assert cm.ComponentModels.load().stamp == models.stamp  # no path = development bundle
    leaky = cm.ComponentModels().fit(
        with_season(synthetic, splits.test_seasons()[0]),
        allow_test_seasons=True, reason="unit test: a bundle that saw a test season",
    )  # fmt: skip
    assert leaky.save() == everything
    with pytest.raises(splits.LeakageError):
        leaky.save(dev)
    with pytest.raises(splits.LeakageError):
        cm.ComponentModels.load_all("short")
    assert cm.ComponentModels.load_all("unit test of the explicit loader").fitted_on_test_seasons()
    joblib_dump_as_dev(leaky, dev)
    with pytest.raises(splits.LeakageError):  # a wrong file under the dev name is caught on load
        cm.ComponentModels.load()


def joblib_dump_as_dev(models: cm.ComponentModels, path) -> None:
    import joblib

    joblib.dump(models, path)


def test_bundle_saved_before_stamps_still_loads(models: cm.ComponentModels, tmp_path) -> None:
    import copy

    import joblib

    old = copy.deepcopy(models)
    for name in ("stamp", "kickoff_tables", "kickoff_era_info"):
        del old.__dict__[name]
    del old.field_goal.__dict__["tail_yards"], old.field_goal.__dict__["max_kick_bound"]
    old.field_goal.max_kick_distance = 63
    joblib.dump(old, tmp_path / "old.joblib")
    loaded = cm.ComponentModels.load(tmp_path / "old.joblib")
    assert loaded.stamp is None and loaded.fitted_on_test_seasons()
    with pytest.raises(splits.LeakageError):
        splits.assert_can_score(loaded.stamp, pl.DataFrame({"season": [2024]}), "old bundle")
    np.testing.assert_array_equal(loaded.kickoff_start(), models.kickoff_start())
    np.testing.assert_array_equal(loaded.kickoff_start(season=2024), models.kickoff_start())
    assert loaded.field_goal.p_make(45, season=2020) > 0  # 63 yards: at the old hard cap
    assert loaded.field_goal.p_make(46, season=2020) == 0.0


def test_field_goal_limit_comes_from_the_fitted_rows(synthetic: pl.DataFrame) -> None:
    kicks = synthetic.filter(pl.col("play_type") == "field_goal")
    longest_made = int(kicks.filter(pl.col("field_goal_result") == "made")["yardline_100"].max())
    assert "max_kick_distance" not in cm.load_config()["field_goal"]  # no constant in config
    for tail in (0, 3):
        fg = cm.FieldGoalModel(tail_yards=tail).fit(synthetic)
        assert fg.max_kick_distance == fg.longest_made_ == longest_made + 18
        assert fg.longest_possible_kick == fg.max_kick_distance + tail
        p = fg.p_make(np.arange(1, 100))
        assert p[longest_made - 1] > 0
        assert np.all(p[longest_made + tail :] == 0.0)
        assert np.all(fg.p_make(np.arange(1, 100), season=2021)[longest_made + tail :] == 0.0)
    hard = cm.FieldGoalModel(tail_yards=0).fit(synthetic).p_make(np.arange(1, 100))
    soft = cm.FieldGoalModel(tail_yards=3).fit(synthetic).p_make(np.arange(1, 100))
    np.testing.assert_allclose(hard[:longest_made], soft[:longest_made])  # same up to the limit
    at_limit = soft[longest_made - 1]
    beyond = soft[longest_made : longest_made + 3]
    assert np.all(beyond > 0) and np.all(np.diff(np.concatenate([[at_limit], beyond, [0]])) < 0)
    assert at_limit - beyond[0] < 0.5 * at_limit + 0.05  # no one-yard cliff to zero
    # A configured bound can lower the limit and can never raise it.
    assert cm.FieldGoalModel(max_kick_distance=99).fit(synthetic).max_kick_distance == (
        longest_made + 18
    )
    assert cm.FieldGoalModel(max_kick_distance=50).fit(synthetic).max_kick_distance == 50
    # Dropping the longest makes lowers the limit: it tracks the rows, nothing else.
    shorter = synthetic.filter(
        (pl.col("play_type") != "field_goal") | (pl.col("yardline_100") <= 30)
    )
    assert cm.FieldGoalModel().fit(shorter).max_kick_distance <= 48


def test_fit_kickoff_has_no_fallback_season(synthetic: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="no fallback"):
        cm.fit_kickoff(synthetic, [2019])  # the synthetic kickoffs are all from 2023
    table, info = cm.fit_kickoff(synthetic, [2019, 2023])
    assert info["seasons"] == [2023] and info["missing_seasons"] == [2019]
    assert table.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cm.fit_kickoff(synthetic.filter(pl.col("play_type") != "kickoff"), None)
    with pytest.raises(ValueError):  # a bundle cannot be fitted without any kickoff table
        cm.ComponentModels().fit(synthetic.filter(pl.col("play_type") != "kickoff"))


def test_kickoff_table_follows_the_rule_era_of_the_season(
    synthetic: pl.DataFrame, access_log
) -> None:
    models = cm.ComponentModels().fit(synthetic)
    assert models.kickoff_era() == models.kickoff_era(2020) == "touchback_at_25"
    assert models.kickoff_era(2012) == "touchback_at_25"  # before the first era: the first
    with pytest.raises(KeyError):  # the 2024 rule has its own era and it was never fitted
        models.kickoff_start(season=2024)
    kick = synthetic.filter(pl.col("play_type") == "kickoff")
    new_rule = pl.concat([
        with_season(kick, 2024).with_columns(pl.lit(68.0).alias("next_yardline_100")),
        with_season(kick, 2025).with_columns(pl.lit(64.0).alias("next_yardline_100")),
    ])  # fmt: skip
    with pytest.raises(splits.LeakageError):
        models.add_kickoff_rule_tables(new_rule)
    fitted = models.add_kickoff_rule_tables(
        new_rule, allow_test_seasons=True, reason=cm.KICKOFF_RULE_REASON
    )
    assert fitted == ["dynamic_kickoff_2024", "dynamic_kickoff_2025"]
    assert models.kickoff_start(season=2024)[68] == 1.0
    assert models.kickoff_start(season=2025)[64] == 1.0
    assert models.kickoff_start()[64] == 1.0  # no season: the latest rule
    assert models.kickoff_start(season=2031)[64] == 1.0  # after the last era: the last
    assert models.kickoff_start(season=2021) @ YARDS == pytest.approx(70, abs=1.5)
    assert models.kickoff_info["seasons"] == [2025]
    # The exception is on the stamp, and the bundle is still a development fit.
    record = models.stamp["kickoff_rule_tables"]
    assert record["dynamic_kickoff_2025"]["uses_test_seasons"]
    assert record["dynamic_kickoff_2025"]["reason"] == cm.KICKOFF_RULE_REASON
    assert not record["touchback_at_25"]["uses_test_seasons"]
    assert not models.fitted_on_test_seasons()
    assert "kickoff start table" in access_log.read_text()
    assert cm.constant_kickoff(65)[65] == 1.0


def test_p_convert_passes_the_spread_through(models: cm.ComponentModels) -> None:
    assert models.p_convert(50, 3, spread=0.0) == models.p_convert(50, 3)
    for spread in (-7.0, 7.0):
        assert models.p_convert(50, 3, spread=spread) == models.go.p_convert(50, 3, spread=spread)
    out = models.p_convert(np.array([50, 40]), np.array([3, 3]), spread=np.array([-7.0, 7.0]))
    assert out.shape == (2,)


@pytest.mark.data
def test_real_field_goal_limit_is_the_longest_made_development_kick(
    real: cm.ComponentModels, dev_plays: pl.DataFrame
) -> None:
    made = dev_plays.filter(
        (pl.col("play_type") == "field_goal") & (pl.col("field_goal_result") == "made")
    )
    longest = int(made["yardline_100"].max()) + 18
    fg = real.field_goal
    assert fg.max_kick_distance <= fg.longest_made_ == longest
    assert real.p_field_goal(longest - 18) > 0
    assert real.p_field_goal(longest + fg.tail_yards + 1 - 18) == 0.0
    assert set(real.stamp["seasons"]) == set(splits.dev_seasons())


@pytest.mark.data
def test_saved_development_bundle_is_current() -> None:
    if not cm.DEV_MODEL_PATH.exists():
        pytest.skip("development bundle not built (run scripts/fit_components.py)")
    saved = cm.ComponentModels.load()
    assert cm.dev_bundle_is_current(saved)
    assert saved.field_goal.max_kick_distance == saved.field_goal.longest_made_
    record = saved.stamp["kickoff_rule_tables"]
    assert {k for k, v in record.items() if v["uses_test_seasons"]} == {
        "dynamic_kickoff_2024", "dynamic_kickoff_2025",
    }  # fmt: skip
