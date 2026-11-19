"""Tests for the win-probability model.

The synthetic tests build a toy league (score is a random walk with a drift set by
the spread) so they run in CI, where the real play-by-play is not available.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fourth_and_rl import splits, wp
from fourth_and_rl.paths import DATA

SMALL = {"n_estimators": 60, "min_child_samples": 100, "num_leaves": 15, "learning_rate": 0.1}


def toy_league(n_games: int, seed: int, first_season: int = 2010, n_seasons: int = 4):
    """Games as random walks: 40 plays each, the favorite drifts ahead by the spread."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        spread = float(np.round(rng.normal(0, 5) * 2) / 2)  # for the home team
        steps = rng.normal(spread / 40, 13.5 / np.sqrt(40), size=40)
        margin = np.concatenate([[0.0], np.cumsum(steps)]).round()  # home minus away
        if margin[-1] == 0:
            margin[-1] = 1.0
        for i in range(40):
            home_ball = bool(rng.integers(0, 2))
            gsr = 3600.0 - 90.0 * i
            qtr, half = wp._derive_clock(np.array([gsr]), np.array([False]))
            sign = 1.0 if home_ball else -1.0
            rows.append(
                {
                    "game_id": f"g{seed}_{g}",
                    "play_id": i,
                    "season": first_season + g % n_seasons,
                    "down": float(rng.integers(1, 5)) if i % 8 else None,
                    "ydstogo": float(rng.integers(1, 11)),
                    "yardline_100": float(rng.integers(1, 100)),
                    "game_seconds_remaining": gsr,
                    "half_seconds_remaining": float(half[0]),
                    "qtr": float(qtr[0]),
                    "score_differential": sign * margin[i],
                    "posteam_timeouts_remaining": float(rng.integers(0, 4)),
                    "defteam_timeouts_remaining": float(rng.integers(0, 4)),
                    "posteam_spread": sign * spread,
                    "receive_2h_ko": int(rng.integers(0, 2)) if gsr > 1800 else 0,
                    "is_home": int(home_ball),
                    "posteam_won": float(sign * margin[-1] > 0),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def toy() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return toy_league(500, 1), toy_league(200, 2), toy_league(200, 3)


@pytest.fixture(scope="module")
def toy_model(toy) -> wp.WPModel:
    train, calib, _ = toy
    return wp.WPModel(SMALL).fit(train, calib, seed=7)


# ------------------------------------------------------------------ features and states


def test_make_features_columns_and_formulas():
    df = wp.state_frame(
        down=[1, 4, 0], ydstogo=[10, 3, 0], yardline_100=[75, 3, 35],
        game_seconds_remaining=[3600, 900, 1800], score_differential=[0, -4, 7],
        posteam_spread=[3.0, -2.5, 0.0],
    )  # fmt: skip
    x = wp.make_features(df)
    assert x.columns == wp.FEATURES
    assert x.shape == (3, len(wp.FEATURES))
    assert x["down"].to_list() == [1.0, 4.0, 0.0]  # no-down plays are coded 0
    assert x["goal_to_go"].to_list() == [0.0, 1.0, 0.0]
    # Kickoff of the game: nothing played, so the whole spread is still to come.
    assert x["spread_time"][0] == pytest.approx(3.0)
    assert x["expected_margin"][0] == pytest.approx(3.0)
    # Start of the fourth quarter: a quarter of the spread left, decayed by exp(-3).
    assert x["expected_margin"][1] == pytest.approx(-4 - 2.5 * 0.25)
    assert x["spread_time"][1] == pytest.approx(-2.5 * np.exp(-3.0))
    assert x["margin_per_sqrt_time"][1] == pytest.approx(
        (-4 - 2.5 * 0.25) / np.sqrt((900 + wp.TIME_FLOOR_SECONDS) / 3600)
    )
    # Version 1 inputs are still available, for refitting the old model.
    x1 = wp.make_features(df, wp.V1)
    assert x1.columns == wp.FEATURES_V1 and "is_home" in x1.columns
    assert "is_home" not in wp.FEATURES and "kneel_slack" in wp.FEATURES


def test_kneel_slack_follows_the_clock_rules():
    """T = 40 * max(kneels - defense timeouts, 0) + 2 * kneels, capped at 120 seconds."""

    def slack(**kw) -> float:
        return wp.make_features(wp.state_frame(**kw))["kneel_slack"][0]

    late = dict(score_differential=3, game_seconds_remaining=100)
    # First down, three kneels. Thresholds 120, 86, 46, 6 for 0 to 3 defensive timeouts.
    assert [slack(down=1, defteam_timeouts_remaining=n, **late) for n in range(4)] == [
        20.0, -14.0, -54.0, -94.0,
    ]  # fmt: skip
    # Second down, two kneels: 84 and 44 seconds, then 4.
    assert [slack(down=2, defteam_timeouts_remaining=n, **late) for n in range(3)] == [
        -16.0, -56.0, -96.0,
    ]  # fmt: skip
    assert slack(down=3, defteam_timeouts_remaining=0, **late) == 42.0 - 100.0
    assert slack(down=4, defteam_timeouts_remaining=0, **late) == -100.0  # no kneel left
    # Not in the lead, no down, overtime, or far from the end: the floor value.
    floor = wp.KNEEL_SLACK_FLOOR
    assert slack(score_differential=0, game_seconds_remaining=30) == floor
    assert slack(score_differential=-3, game_seconds_remaining=30) == floor
    assert slack(down=0, score_differential=3, game_seconds_remaining=30) == floor
    assert slack(score_differential=3, game_seconds_remaining=30, overtime=True) == floor
    assert slack(score_differential=3, game_seconds_remaining=3000) == floor
    # Never falls when the lead grows or the defense loses a timeout (monotone inputs).
    grid = wp.make_features(
        wp.state_frame(score_differential=np.arange(-5, 6), game_seconds_remaining=50)
    )["kneel_slack"].to_numpy()
    assert np.all(np.diff(grid) >= 0)
    assert wp.MONOTONE["kneel_slack"] == 1


def test_possession_features():
    x = wp.make_features(
        wp.state_frame(score_differential=[-17, -9, -8, -1, 0, 3, 8, 9],
                       game_seconds_remaining=[90, 90, 90, 90, 90, 90, 90, 0])
    )  # fmt: skip
    assert x["possessions"].to_list() == [-3, -2, -1, -1, 0, 1, 1, 2]
    # Per minute left, with 30 seconds added: 90 seconds left is 2 minutes.
    assert x["possessions_per_minute"].to_list()[:7] == [-1.5, -1.0, -0.5, -0.5, 0.0, 0.5, 0.5]
    assert x["possessions_per_minute"][7] == pytest.approx(4.0)  # 2 scores, 30 s
    assert wp.MONOTONE["possessions"] == wp.MONOTONE["possessions_per_minute"] == 1


def test_baseline_is_the_normal_approximation():
    # Tied with no edge: a coin flip. One standard deviation ahead: Phi(1) = 0.8413.
    args = dict(sigma=13.5, tau=0.0)
    p = 1 / (1 + np.exp(-wp.baseline_logit(np.array([0.0, 13.5]), np.array([3600.0] * 2), **args)))
    assert p == pytest.approx([0.5, 0.8413], abs=1e-4)
    # A quarter of the game left halves the standard deviation: 6.75 points is now 1 sd.
    p = 1 / (1 + np.exp(-wp.baseline_logit(np.array([6.75]), np.array([900.0]), **args)))
    assert p[0] == pytest.approx(0.8413, abs=1e-4)
    # A lead with no time left is a win, a deficit a loss, and the log-odds stay finite.
    edge = wp.baseline_logit(np.array([3.0, -3.0]), np.array([0.0, 0.0]), 13.5, 1.0)
    assert edge.tolist() == [wp.BASELINE_LOGIT_CAP, -wp.BASELINE_LOGIT_CAP]
    # The possession value moves the margin of the team with the ball only.
    m, t = np.array([0.0, 0.0]), np.array([3600.0, 3600.0])
    shifted = wp.baseline_logit(m, t, 13.5, 0.0, 13.5, has_ball=np.array([1.0, 0.0]))
    assert 1 / (1 + np.exp(-shifted)) == pytest.approx([0.8413, 0.5], abs=1e-4)
    # The kneel rule overrides the normal approximation where it applies.
    knelt = wp.baseline_logit(np.array([1.0, 1.0]), np.array([50.0, 50.0]), 13.5, 400.0,
                              can_kneel=np.array([True, False]))  # fmt: skip
    assert knelt[0] == wp.BASELINE_LOGIT_CAP and knelt[1] < 1.0
    with pytest.raises(ValueError):  # the rule needs the feature it is read from
        wp.ModelSpec("bad", tuple(wp.FEATURES_V1), True, baseline_kneel=True)


def test_fit_baseline_recovers_sigma_from_simulated_games():
    rng = np.random.default_rng(5)
    n = 60_000
    seconds = rng.uniform(60, 3600, n)
    margin = rng.normal(0, 8, n)
    final = margin + rng.normal(0, 12.0 * np.sqrt(seconds / 3600))
    fitted = wp.fit_baseline(margin, seconds, (final > 0).astype(float))
    assert fitted["sigma"] == pytest.approx(12.0, rel=0.05)
    assert fitted["tau"] < 60  # nothing in these games needs a clock offset
    fixed = wp.fit_baseline(margin, seconds, (final > 0).astype(float), tau=10.0)
    assert fixed["tau"] == 10.0 and fixed["sigma"] == pytest.approx(12.0, rel=0.05)


def test_make_features_requires_state_columns():
    with pytest.raises(KeyError):
        wp.make_features(pl.DataFrame({"down": [1.0]}))


def test_state_frame_clock_and_broadcast():
    df = wp.state_frame(game_seconds_remaining=[3600, 2701, 2700, 1801, 1800, 900, 0],
                        receive_2h_ko=1)  # fmt: skip
    assert df.height == 7
    assert df["qtr"].to_list() == [1, 1, 2, 2, 3, 4, 4]
    assert df["half_seconds_remaining"].to_list() == [1800, 901, 900, 1, 1800, 900, 0]
    assert df["receive_2h_ko"].to_list() == [1, 1, 1, 1, 0, 0, 0]  # first half only
    assert set(wp.STATE_COLUMNS) <= set(df.columns)
    ot = wp.state_frame(game_seconds_remaining=400, overtime=True)
    assert ot["qtr"][0] == 5 and ot["half_seconds_remaining"][0] == 400
    assert wp.state_frame(down=None)["down"][0] is None
    assert wp.state_frame(yardline_100=np.arange(1, 100)).height == 99


@pytest.mark.parametrize(
    "bad", [{"game_seconds_remaining": 3601}, {"yardline_100": 0}, {"down": 5}]
)
def test_state_frame_rejects_impossible_states(bad):
    with pytest.raises(ValueError):
        wp.state_frame(**bad)


# ------------------------------------------------------------------------------ metrics


def test_metrics_worked_examples():
    y = np.array([1.0, 0.0])
    assert wp.log_loss(y, np.array([0.5, 0.5])) == pytest.approx(np.log(2))
    assert wp.brier(y, np.array([0.5, 0.5])) == pytest.approx(0.25)
    assert wp.log_loss(y, np.array([0.9, 0.2])) == pytest.approx(
        -(np.log(0.9) + np.log(0.8)) / 2
    )
    assert wp.brier(y, np.array([0.9, 0.2])) == pytest.approx((0.01 + 0.04) / 2)
    # Ten plays predicted 0.75, six won: one bin, error |0.6 - 0.75|.
    y, p = np.array([1.0] * 6 + [0.0] * 4), np.full(10, 0.75)
    assert wp.ece(y, p) == pytest.approx(0.15)
    table = wp.reliability_table(y, p, 10)
    assert table["count"][7] == 10 and table["observed"][7] == pytest.approx(0.6)
    assert table["observed"][0] is None  # empty bins are null, not zero
    assert np.isfinite(wp.log_loss(np.array([1.0]), np.array([0.0])))  # clipped


def test_ece_is_small_for_calibrated_predictions():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(float)
    assert wp.ece(y, p) < 0.01
    assert wp.ece(y, np.clip(p * 1.3 - 0.15, 0, 1)) > 0.03  # overconfident version


def test_game_bootstrap_diff_sign_and_interval():
    rng = np.random.default_rng(1)
    games = np.repeat(np.arange(200), 20)
    p_true = np.repeat(rng.uniform(0.1, 0.9, 200), 20)
    y = np.repeat((rng.uniform(0, 1, 200) < p_true[::20]).astype(float), 20)
    out = wp.game_bootstrap_diff(y, p_true, np.full(p_true.size, 0.5), games, 300, seed=3)
    for metric in ("log_loss", "brier"):
        assert out[metric]["diff"] < 0  # the informed forecast beats the coin flip
        assert out[metric]["ci_low"] <= out[metric]["diff"] <= out[metric]["ci_high"]


# -------------------------------------------------------------------------- calibration


def test_platt_recovers_a_known_distortion():
    rng = np.random.default_rng(2)
    true_logit = rng.normal(0, 2, 50_000)
    y = (rng.uniform(0, 1, true_logit.size) < 1 / (1 + np.exp(-true_logit))).astype(float)
    score = 2.0 * true_logit + 0.5  # an overconfident, shifted model
    cal = wp.Calibrator("platt").fit(score, y)
    assert cal.a == pytest.approx(0.5, abs=0.03)
    assert cal.b == pytest.approx(-0.25, abs=0.05)
    assert wp.ece(y, cal.transform(score)) < wp.ece(y, 1 / (1 + np.exp(-score)))
    iso = wp.Calibrator("isotonic").fit(score, y).transform(np.sort(score))
    assert np.all(np.diff(iso) >= 0)
    with pytest.raises(ValueError):
        wp.Calibrator("temperature")


# -------------------------------------------------------------------------------- model


def test_fit_predict_on_toy_league(toy, toy_model):
    _, _, test = toy
    p = toy_model.predict(test)
    assert p.shape == (test.height,)
    assert np.all((p > 0) & (p < 1))
    y = test["posteam_won"].to_numpy()
    assert wp.log_loss(y, p) < 0.62  # clearly better than a coin flip (0.693)
    assert wp.ece(y, p) < 0.06
    assert toy_model.calibrator.method == "platt"
    assert toy_model.n_train_rows == toy[0].height
    # Version 2 by default: trees on top of a fitted baseline.
    assert toy_model.spec == wp.V2
    assert 10 < toy_model.baseline["sigma"] < 18  # the toy league is built with 13.5
    base = toy_model.baseline_predict(test)
    assert wp.log_loss(y, base) < 0.64  # the two-number baseline already beats a coin


def test_version_1_can_still_be_fitted_and_loaded(tmp_path, toy):
    train, calib, test = toy
    old = wp.WPModel(SMALL, "platt", spec=wp.V1).fit(train, calib, seed=7)
    assert old.baseline is None and old.booster.feature_name() == wp.FEATURES_V1
    again = wp.WPModel.load(old.save(tmp_path / "v1.joblib"))
    assert again.spec == wp.V1
    np.testing.assert_allclose(again.predict(test), old.predict(test), rtol=0, atol=1e-12)
    # A file written before version 2 has no spec, baseline or stamp: it loads as version 1.
    import joblib

    blob = joblib.load(tmp_path / "v1.joblib")
    for key in ("spec", "baseline", "stamp", "late_booster"):
        blob.pop(key)
    joblib.dump(blob, tmp_path / "legacy.joblib")
    legacy = wp.WPModel.load(tmp_path / "legacy.joblib")
    assert legacy.spec == wp.V1 and legacy.stamp is None
    np.testing.assert_allclose(legacy.predict(test), old.predict(test), rtol=0, atol=1e-12)
    with pytest.raises(splits.LeakageError):  # an unstamped model cannot claim out-of-sample
        splits.assert_can_score(legacy.stamp, test, "legacy model")


def test_fit_is_deterministic_and_handles_bootstrap_samples(toy):
    train, calib, test = toy
    a = wp.WPModel(SMALL).fit(train, calib, seed=11).predict(test)
    b = wp.WPModel(SMALL).fit(train, calib, seed=11).predict(test)
    np.testing.assert_array_equal(a, b)
    games = train["game_id"].unique(maintain_order=True)
    draw = games.sample(games.len(), with_replacement=True, seed=5)
    boot = pl.DataFrame({"game_id": draw}).join(train, on="game_id", how="left")
    assert boot.height == train.height
    model = wp.WPModel(SMALL).fit(boot, seed=11, n_estimators=20)
    assert model.booster.num_trees() == 20
    assert model.calibrator.method == "none"  # no calibration frame was given
    assert np.all(np.isfinite(model.predict(test)))


def test_rows_without_label_are_dropped_and_incomplete_states_predict_nan(toy, toy_model):
    train, _, test = toy
    holes = train.with_columns(
        pl.when(pl.col("play_id") == 0).then(None).otherwise(pl.col("posteam_won")).alias(
            "posteam_won"
        )
    )
    model = wp.WPModel(SMALL).fit(holes, seed=1, n_estimators=10)
    assert model.n_train_rows == train.height - train["game_id"].n_unique()
    broken = test.head(5).with_columns(
        pl.when(pl.col("play_id") == 1).then(None).otherwise(pl.col("yardline_100")).alias(
            "yardline_100"
        )
    )
    p = toy_model.predict(broken)
    assert np.isnan(p[1]) and np.all(np.isfinite(np.delete(p, 1)))
    with pytest.raises(ValueError):
        wp.WPModel(SMALL).fit(train.filter(pl.col("posteam_won") == 1.0))


def test_monotone_in_score_and_spread_on_toy_model(toy_model):
    for secs in (3000, 1500, 200):
        by_score = toy_model.predict(
            wp.state_frame(score_differential=np.arange(-28, 29), game_seconds_remaining=secs)
        )
        assert np.all(np.diff(by_score) >= -1e-12)
        by_spread = toy_model.predict(
            wp.state_frame(posteam_spread=np.arange(-14, 14.5, 0.5), game_seconds_remaining=secs)
        )
        assert np.all(np.diff(by_spread) >= -1e-12)
    late = toy_model.predict(
        wp.state_frame(score_differential=[30, -30], game_seconds_remaining=90)
    )
    assert late[0] > 0.9 and late[1] < 0.1


def test_save_and_load_round_trip(tmp_path, toy, toy_model):
    path = toy_model.save(tmp_path / "models" / "wp.joblib")
    again = wp.WPModel.load(path)
    np.testing.assert_allclose(again.predict(toy[2]), toy_model.predict(toy[2]), rtol=0, atol=1e-12)
    assert again.calibrator.describe() == toy_model.calibrator.describe()
    assert again.baseline == toy_model.baseline
    with pytest.raises(RuntimeError):
        wp.WPModel().predict(toy[2])


# ----------------------------------------------------------------------- leakage guards


@pytest.fixture()
def access_log(tmp_path, monkeypatch):
    """Send the test-access log to a temporary file, so tests leave no trace in results/."""
    path = tmp_path / "test_access_log.jsonl"
    monkeypatch.setattr(splits, "ACCESS_LOG", path)
    return path


def test_fit_refuses_test_seasons(toy, access_log):
    train, calib, _ = toy
    test_season = splits.test_seasons()[0]
    leaky = toy_league(40, 9, first_season=test_season, n_seasons=1)
    assert test_season not in splits.seasons_in(train)  # toy seasons are development seasons
    with pytest.raises(splits.LeakageError):
        wp.WPModel(SMALL).fit(pl.concat([train, leaky]), calib, seed=1, n_estimators=5)
    with pytest.raises(splits.LeakageError):  # the calibration frame is guarded too
        wp.WPModel(SMALL).fit(train, leaky, seed=1, n_estimators=5)
    with pytest.raises(splits.LeakageError):  # an exception needs a written reason
        wp.WPModel(SMALL).fit(leaky, seed=1, n_estimators=5, allow_test_seasons=True)
    assert not access_log.exists()
    model = wp.WPModel(SMALL).fit(
        leaky, seed=1, n_estimators=5, allow_test_seasons=True,
        reason="unit test of the logged exception; nothing is reported",
    )  # fmt: skip
    assert model.stamp["allow_test_seasons"] and test_season in model.stamp["seasons"]
    assert "fit_on_test_seasons" in access_log.read_text()
    # The final holdout can never be fitted on, with or without a reason.
    holdout = toy_league(40, 10, first_season=splits.holdout_seasons()[0], n_seasons=1)
    with pytest.raises(splits.LeakageError):
        wp.WPModel(SMALL).fit(holdout, seed=1, allow_test_seasons=True,
                              reason="this must fail whatever the reason given")  # fmt: skip


def test_fit_needs_season_and_game_columns_and_disjoint_games(toy):
    train, calib, _ = toy
    with pytest.raises(KeyError):
        wp.WPModel(SMALL).fit(train.drop("season"), seed=1)
    with pytest.raises(splits.LeakageError):  # a game on both sides of the split
        wp.WPModel(SMALL).fit(train, pl.concat([calib, train.head(40)]), seed=1)


def test_stamp_survives_save_and_load_and_blocks_scoring_seen_seasons(tmp_path, toy, toy_model):
    train, calib, test = toy
    stamp = toy_model.stamp
    assert stamp["seasons"] == [2010, 2011, 2012, 2013]
    assert stamp["train_seasons"] == stamp["calib_seasons"] == [2010, 2011, 2012, 2013]
    assert stamp["n_games"] == 700 and stamp["spec"] == "v2"
    again = wp.WPModel.load(toy_model.save(tmp_path / "wp.joblib"))
    assert again.stamp == stamp
    unseen = toy_league(20, 12, first_season=2018, n_seasons=2)
    splits.assert_can_score(again.stamp, unseen, "unseen seasons")  # no error
    with pytest.raises(splits.LeakageError):
        splits.assert_can_score(again.stamp, test, "seasons the model was fitted on")
    with pytest.raises(splits.LeakageError):
        splits.assert_can_score(wp.WPModel().stamp, unseen, "a model that was never fitted")


def test_endgame_sanity_checks_report_every_check(toy_model):
    out = wp.sanity_checks(toy_model)
    for key in ("kneel_out", "two_scores_no_time", "defense_timeouts",
                "one_point_lead_13_seconds"):  # fmt: skip
        assert isinstance(out[key]["passed"], bool)
    assert out["all_passed"] == (out["core_passed"] and out["endgame_passed"])
    # The kneel rule in the baseline makes the kneel-out check hold by construction,
    # whatever the toy trees learned (the trees cannot pull far below the cap here).
    assert out["kneel_out"]["min_wp"] > 0.9


def test_feature_importance_sums_to_one(toy_model):
    imp = toy_model.feature_importance()
    assert set(imp) == set(wp.FEATURES)
    assert sum(imp.values()) == pytest.approx(1.0)


# ------------------------------------------------------------------------ cross-fitting


def test_crossfit_plan_never_trains_on_the_held_out_seasons():
    plan = wp.crossfit_plan(range(2010, 2026), fold_size=4, n_calib_seasons=2)
    assert [f["predict"] for f in plan] == [
        list(range(2010, 2014)), list(range(2014, 2018)),
        list(range(2018, 2022)), list(range(2022, 2026)),
    ]  # fmt: skip
    for fold in plan:
        assert not set(fold["predict"]) & (set(fold["train"]) | set(fold["calib"]))
        assert not set(fold["train"]) & set(fold["calib"])
        assert len(fold["train"]) == 10 and len(fold["calib"]) == 2
    assert plan[0]["calib"] == [2024, 2025] and plan[3]["calib"] == [2020, 2021]
    with pytest.raises(ValueError):
        wp.crossfit_plan([2010, 2011], fold_size=1, n_calib_seasons=1)


def test_crossfit_plan_keeps_test_seasons_out_of_every_fit():
    test = list(splits.test_seasons())
    plan = wp.crossfit_plan(range(2010, 2026), fold_size=2, n_calib_seasons=0, never_fit=test)
    assert len(plan) == 8 and plan[-1]["predict"] == test
    assert plan[-1]["train"] == list(range(2010, 2024))  # all development seasons
    for fold in plan:
        assert not set(test) & (set(fold["train"]) | set(fold["calib"]))
        assert not set(fold["predict"]) & set(fold["train"])
        assert fold["calib"] == []
    covered = sorted(s for fold in plan for s in fold["predict"])
    assert covered == list(range(2010, 2026))


def test_crossfit_predict_handles_test_seasons_without_fitting_on_them(access_log):
    # Seasons 2020 to 2025: four development seasons and the two test seasons.
    league = toy_league(240, 6, first_season=2020, n_seasons=6)
    oof, plan = wp.crossfit_predict(
        league, {**SMALL, "n_estimators": 10}, "none", fold_size=2, n_calib_seasons=0, seed=3
    )
    assert [f["predict"] for f in plan] == [[2020, 2021], [2022, 2023], [2024, 2025]]
    assert plan[-1]["train"] == [2020, 2021, 2022, 2023]
    assert oof.height == league.height and oof["wp_ours"].null_count() == 0
    assert not access_log.exists()  # no exception to the guard was needed
    # Letting folds train on test seasons is an exception that must be asked for.
    with pytest.raises(splits.LeakageError):
        wp.crossfit_predict(league, {**SMALL, "n_estimators": 10}, "none", fold_size=2,
                            n_calib_seasons=0, never_fit=())  # fmt: skip
    wp.crossfit_predict(
        league, {**SMALL, "n_estimators": 10}, "none", fold_size=2, n_calib_seasons=0,
        never_fit=(), allow_test_seasons=True,
        reason="unit test: folds predict only seasons they did not see",
    )  # fmt: skip
    assert access_log.read_text().count("fit_on_test_seasons") >= 2


def test_season_blocked_cv_and_endgame_table():
    league = toy_league(240, 8, n_seasons=6)
    cv = wp.season_blocked_cv(league, {**SMALL, "n_estimators": 10}, fold_size=2, seed=3)
    assert len(cv["folds"]) == 3 and not np.isnan(cv["pred"]).any()
    for fold in cv["folds"]:
        assert not set(fold["predict"]) & set(fold["train"])
        assert fold["held_out_log_loss"] > fold["in_sample_log_loss"] - 0.05
    # A hand-made end game: two games, four first downs inside 0:40 with a 2-point lead.
    cells = pl.DataFrame(
        {
            "game_id": ["a", "a", "b", "b", "b"],
            "qtr": [4.0] * 5,
            "down": [1.0, 1.0, 1.0, 2.0, None],
            "game_seconds_remaining": [30.0, 20.0, 35.0, 10.0, 5.0],
            "score_differential": [2.0] * 5,
            "posteam_won": [1.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
    table = wp.endgame_slice_table(cells, np.array([0.9, 0.9, 0.7, 0.7, 0.1]), n_boot=50)
    assert len(table) == len(wp.ENDGAME_TIME_BINS) * len(wp.ENDGAME_MARGIN_BINS)
    cell = next(c for c in table if c["time"] == "0:00-0:40" and c["margin"] == "lead 1-3")
    assert cell["rows"] == 4 and cell["games"] == 2  # the play with no down is left out
    assert cell["observed"] == pytest.approx(0.5) and cell["predicted"] == pytest.approx(0.8)
    assert cell["gap"] == pytest.approx(0.3)
    assert cell["gap_ci_low"] <= cell["gap"] <= cell["gap_ci_high"]
    assert sum(c["rows"] for c in table) == 4
    summary = wp.endgame_summary(table)
    assert summary["mean_abs_gap"] == pytest.approx(0.3) and summary["cells"] == 1
    first = wp.endgame_slice_table(cells, np.array([0.9, 0.9, 0.7, 0.7, 0.1]), 50,
                                   first_down_only=True)  # fmt: skip
    assert sum(c["rows"] for c in first) == 3


def test_crossfit_predict_covers_every_play_once():
    league = toy_league(240, 4, n_seasons=8)
    oof, plan = wp.crossfit_predict(
        league, {**SMALL, "n_estimators": 20}, fold_size=2, n_calib_seasons=2, seed=3
    )
    assert len(plan) == 4
    assert oof.columns == ["game_id", "play_id", "wp_ours"]
    assert oof.height == league.height
    assert oof.select(pl.struct("game_id", "play_id").n_unique()).item() == league.height
    assert oof["wp_ours"].null_count() == 0
    joined = league.join(oof, on=["game_id", "play_id"])
    assert wp.log_loss(joined["posteam_won"].to_numpy(), joined["wp_ours"].to_numpy()) < 0.66


# ----------------------------------------------------------------- real play-by-play


@pytest.fixture(scope="module")
def real_model(plays) -> wp.WPModel:
    """A reduced fit (two training seasons, 80 trees) so the suite stays fast."""
    return wp.WPModel({"n_estimators": 80, "min_child_samples": 2000}).fit(
        wp.season_filter(plays, [2018, 2019]), wp.season_filter(plays, [2020]), seed=1
    )


@pytest.mark.data
def test_real_data_has_the_state_columns(plays):
    assert set(wp.STATE_COLUMNS + [wp.LABEL]) <= set(plays.columns)
    ok = wp.scoreable(plays)
    assert ok.mean() > 0.97  # only administrative rows lack a field position or clock
    x = wp.make_features(plays.filter(pl.Series(ok)).head(20_000))
    assert x.null_count().sum_horizontal().item() == 0


@pytest.mark.data
def test_state_frame_matches_the_clock_columns_in_real_data(plays):
    reg = plays.filter(
        (pl.col("qtr") <= 4)
        & pl.col("game_seconds_remaining").is_not_null()
        & ~pl.col("game_seconds_remaining").is_in([900.0, 1800.0, 2700.0])
    ).sample(20_000, seed=0)
    built = wp.state_frame(game_seconds_remaining=reg["game_seconds_remaining"].to_numpy())
    np.testing.assert_array_equal(built["qtr"].to_numpy(), reg["qtr"].to_numpy())
    np.testing.assert_array_equal(
        built["half_seconds_remaining"].to_numpy(), reg["half_seconds_remaining"].to_numpy()
    )


@pytest.mark.data
def test_real_model_beats_chance_and_passes_sanity_checks(plays, real_model):
    held = wp.season_filter(plays, [2021]).filter(pl.col(wp.LABEL).is_not_null())
    p = real_model.predict(held)
    ok = ~np.isnan(p)
    y = held[wp.LABEL].to_numpy()[ok]
    assert wp.log_loss(y, p[ok]) < 0.50
    assert wp.ece(y, p[ok]) < 0.06
    assert abs(p[ok].mean() - y.mean()) < 0.03
    checks = wp.sanity_checks(real_model)
    for name in ("score_differential", "posteam_spread", "field_position"):
        assert checks[name]["passed"], (name, checks[name])
    # A reduced fit is less sharp, so the blowout and kickoff thresholds are looser
    # here than the ones the full model must meet in scripts/fit_wp.py.
    assert checks["blowout"]["up_30"] > 0.97 and checks["blowout"]["down_30"] < 0.03
    assert checks["kickoff_mirror"]["max_abs_deviation_from_1"] < 0.15


@pytest.mark.data
def test_saved_outputs_are_consistent_with_the_plays(plays):
    path = DATA / "processed" / "wp_predictions.parquet"
    if not path.exists() or not (DATA / "models" / "wp.joblib").exists():
        pytest.skip("run scripts/fit_wp.py first")
    oof = pl.read_parquet(path)
    assert oof.columns == ["game_id", "play_id", "wp_ours"]
    assert oof.height == plays.height
    joined = plays.join(oof, on=["game_id", "play_id"], how="inner", validate="1:1")
    assert joined.height == plays.height
    p = joined["wp_ours"].drop_nulls().to_numpy()
    assert p.min() > 0 and p.max() < 1
    assert joined["wp_ours"].is_not_null().mean() > 0.97
    model = wp.WPModel.load(DATA / "models" / "wp.joblib")
    assert wp.sanity_checks(model)["core_passed"]


@pytest.fixture(scope="module")
def saved_model() -> wp.WPModel:
    path = DATA / "models" / "wp.joblib"
    if not path.exists():
        pytest.skip("run scripts/fit_wp.py first")
    model = wp.WPModel.load(path)
    if model.spec != wp.V2:
        pytest.skip("the saved model is not version 2; run scripts/fit_wp.py")
    return model


# End-game checks on the saved headline model. The two marked xfail fail for version 2
# (results/m1_wp_dev.json, sanity_checks). They are strict: if a refit makes one pass,
# the mark must be removed, and a threshold is never loosened to make one pass.
ENDGAME_EXPECTED = [
    ("kneel_out", None),
    ("defense_timeouts", None),
    ("one_point_lead_13_seconds", "version 2 gives about 0.88; the bar is 0.90"),
    ("two_scores_no_time", "version 2's worst state is about 0.06; the bar is 0.02"),
]


@pytest.mark.data
@pytest.mark.parametrize(
    "check",
    [c if why is None else pytest.param(c, marks=pytest.mark.xfail(reason=why, strict=True))
     for c, why in ENDGAME_EXPECTED],
)  # fmt: skip
def test_endgame_sanity_checks_on_the_saved_model(saved_model, check):
    assert wp.endgame_sanity_checks(saved_model)[check]["passed"]


@pytest.mark.data
def test_saved_model_is_stamped_with_development_seasons_only(saved_model):
    stamp = saved_model.stamp
    assert stamp is not None and stamp["spec"] == "v2"
    assert set(stamp["seasons"]) == set(splits.dev_seasons())
    assert not stamp["allow_test_seasons"]
    test_like = pl.DataFrame({"season": [splits.dev_seasons()[0]], "game_id": ["x"]})
    with pytest.raises(splits.LeakageError):
        splits.assert_can_score(stamp, test_like, "a development season")
