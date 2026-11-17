"""Tests for the win-probability model.

The synthetic tests build a toy league (score is a random walk with a drift set by
the spread) so they run in CI, where the real play-by-play is not available.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fourth_and_rl import wp
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
        (-4 - 2.5 * 0.25) / np.sqrt(930 / 3600)
    )


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
    with pytest.raises(RuntimeError):
        wp.WPModel().predict(toy[2])


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
    assert wp.sanity_checks(model)["all_passed"]
