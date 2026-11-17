"""Tests for the neural win-probability model and the deep ensemble.

The synthetic tests build games whose true win probability is a known logistic
function of the state, so they run in CI, where the real play-by-play is not
available. The whole module is skipped when PyTorch is not installed.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from fourth_and_rl import splits, wp, wp_nn  # noqa: E402
from fourth_and_rl.paths import DATA  # noqa: E402

SMALL = {"hidden": [16], "batch_size": 256, "max_epochs": 25, "patience": 5, "lr": 0.01}


def true_wp(frame: pl.DataFrame) -> np.ndarray:
    """The known rule of the toy league: log-odds = 0.25 * lead + 0.15 * spread."""
    z = 0.25 * frame["score_differential"].to_numpy() + 0.15 * frame["posteam_spread"].to_numpy()
    return 1.0 / (1.0 + np.exp(-z))


def logistic_league(
    n_games: int, seed: int, plays_per_game: int = 12, season: int = 2010
) -> pl.DataFrame:
    """Independent plays with a label drawn from `true_wp`, grouped into games.

    `season` defaults to a development season, because fits refuse test seasons.
    """
    rng = np.random.default_rng(seed)
    n = n_games * plays_per_game
    gsr = rng.integers(1, 3600, n).astype(float)
    qtr, half = wp._derive_clock(gsr, np.zeros(n, dtype=bool))
    frame = pl.DataFrame(
        {
            "game_id": np.repeat([f"g{seed}_{g}" for g in range(n_games)], plays_per_game),
            "play_id": np.tile(np.arange(plays_per_game), n_games),
            "season": np.full(n, season, dtype=np.int32),
            "down": rng.integers(1, 5, n).astype(float),
            "ydstogo": rng.integers(1, 11, n).astype(float),
            "yardline_100": rng.integers(1, 100, n).astype(float),
            "game_seconds_remaining": gsr,
            "half_seconds_remaining": half,
            "qtr": qtr,
            "score_differential": rng.integers(-14, 15, n).astype(float),
            "posteam_timeouts_remaining": rng.integers(0, 4, n).astype(float),
            "defteam_timeouts_remaining": rng.integers(0, 4, n).astype(float),
            "posteam_spread": np.round(rng.normal(0, 5, n) * 2) / 2,
            "receive_2h_ko": np.zeros(n, dtype=np.int8),
            "is_home": rng.integers(0, 2, n).astype(np.int8),
        }
    )
    return frame.with_columns(
        pl.Series(wp.LABEL, (rng.random(n) < true_wp(frame)).astype(float))
    )


@pytest.fixture(scope="module")
def toy() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return logistic_league(500, 1), logistic_league(150, 2), logistic_league(150, 3)


@pytest.fixture(scope="module")
def toy_model(toy) -> wp_nn.NNWPModel:
    train, calib, _ = toy
    return wp_nn.NNWPModel(SMALL).fit(train, calib, seed=7)


# ---------------------------------------------------------------------- preprocessing


def test_preprocessor_standardizes_with_training_statistics(toy):
    train, calib, _ = toy
    x_train = wp.make_features(train).to_numpy()
    pre = wp_nn.Preprocessor().fit(x_train)
    z = pre.transform(x_train)
    assert z.shape == (train.height, len(wp.FEATURES) + 4) and z.dtype == np.float32
    assert np.allclose(z.mean(axis=0), 0, atol=1e-4)
    # Columns that vary have unit spread; constant ones (overtime here) stay at zero.
    varies = z.std(axis=0) > 0
    assert np.allclose(z.std(axis=0)[varies], 1, atol=1e-3) and varies.sum() >= 15
    # New data is scaled with the training numbers, so its mean is not exactly zero.
    mean_before = pre.mean.copy()
    z_calib = pre.transform(wp.make_features(calib).to_numpy())
    assert np.array_equal(pre.mean, mean_before)
    assert not np.allclose(z_calib.mean(axis=0), 0, atol=1e-6)


def test_preprocessor_one_hot_down_and_plain_variant(toy):
    x = wp.make_features(toy[0]).to_numpy()
    pre = wp_nn.Preprocessor().fit(x)
    one_hot = pre._expand(x)[:, -wp_nn.N_DOWNS :]
    assert np.array_equal(one_hot.sum(axis=1), np.ones(len(x)))
    assert np.array_equal(one_hot.argmax(axis=1), x[:, wp.FEATURES.index("down")].astype(int))
    plain = wp_nn.Preprocessor(one_hot_down=False, squash_margin=False).fit(x)
    assert plain.transform(x).shape == (len(x), len(wp.FEATURES))


def test_resample_games_draws_whole_games(toy):
    train = toy[0]
    sample = wp_nn.resample_games(train, np.random.default_rng(0))
    assert sample.height == train.height  # every toy game has the same number of plays
    assert sample["game_id"].n_unique() == train["game_id"].n_unique()
    originals = sample["game_id"].str.split("#").list.first()
    assert originals.n_unique() < train["game_id"].n_unique()  # some games drawn twice
    assert sample.group_by("game_id").len()["len"].unique().to_list() == [12]


# ------------------------------------------------------------------------------ model


def test_network_learns_the_known_logistic_rule(toy, toy_model):
    test = toy[2]
    p, truth, y = toy_model.predict(test), true_wp(test), test[wp.LABEL].to_numpy()
    assert np.all((p > 0) & (p < 1))
    assert np.corrcoef(p, truth)[0, 1] > 0.97
    assert np.mean(np.abs(p - truth)) < 0.06
    # Close to the best achievable log loss, and far better than a coin flip.
    assert wp.log_loss(y, p) < wp.log_loss(y, truth) + 0.03
    assert wp.log_loss(y, p) < 0.55


def test_fit_is_deterministic_for_a_fixed_seed(toy, toy_model):
    train, calib, test = toy
    again = wp_nn.NNWPModel(SMALL).fit(train, calib, seed=7)
    assert np.array_equal(toy_model.predict(test), again.predict(test))
    other = wp_nn.NNWPModel(SMALL).fit(train, calib, seed=8)
    assert not np.array_equal(toy_model.predict(test), other.predict(test))


def test_save_and_load_round_trip(toy, toy_model, tmp_path):
    test = toy[2]
    path = toy_model.save(tmp_path / "nn.joblib")
    loaded = wp_nn.NNWPModel.load(path)
    assert np.array_equal(toy_model.predict(test), loaded.predict(test))
    assert loaded.seed == 7 and loaded.best_epoch == toy_model.best_epoch


def test_incomplete_state_gives_nan(toy, toy_model):
    test = toy[2].head(5).with_columns(
        pl.when(pl.col("play_id") == 1).then(None).otherwise(pl.col("yardline_100"))
        .alias("yardline_100")
    )  # fmt: skip
    p = toy_model.predict(test)
    assert np.isnan(p[1]) and not np.isnan(np.delete(p, 1)).any()


def test_early_stopping_holds_out_whole_games_when_no_calib(toy):
    train = toy[0]
    model = wp_nn.NNWPModel({**SMALL, "holdout_game_fraction": 0.2}).fit(train, seed=3)
    assert model.n_stop_rows == 100 * 12 and model.n_train_rows == 400 * 12
    assert 1 <= model.best_epoch <= len(model.history) <= SMALL["max_epochs"]
    kept = model.history[model.best_epoch - 1]["heldout_loss"]
    assert kept == min(h["heldout_loss"] for h in model.history)


def test_regularization_options_train_and_platt_is_fitted(toy):
    train, calib, test = toy
    params = {**SMALL, "dropout": 0.1, "label_noise": 0.05, "weight_decay": 0.01,
              "calibration": "platt", "max_epochs": 8}  # fmt: skip
    model = wp_nn.NNWPModel(params).fit(train, calib, seed=5)
    assert model.calibrator.method == "platt" and model.calibrator.a > 0
    p = model.predict(test)
    assert np.corrcoef(p, true_wp(test))[0, 1] > 0.9
    assert not np.array_equal(p, model.predict(test, calibrated=False))


def test_fit_needs_game_ids_and_both_classes(toy):
    train = toy[0]
    with pytest.raises(KeyError):
        wp_nn.NNWPModel(SMALL).fit(train.drop("game_id"))
    with pytest.raises(ValueError):
        wp_nn.NNWPModel(SMALL).fit(train.with_columns(pl.lit(1.0).alias(wp.LABEL)))
    with pytest.raises(RuntimeError):
        wp_nn.NNWPModel(SMALL).predict(train)


# --------------------------------------------------------------------------- ensemble


def test_ensemble_shapes_mean_and_round_trip(toy, tmp_path):
    train, calib, test = toy
    params = {**SMALL, "max_epochs": 6}
    ens = wp_nn.WPEnsemble(params, k=3).fit(train, calib, seed=11)
    members = ens.predict_members(test)
    assert members.shape == (3, test.height)
    assert np.allclose(ens.predict(test), members.mean(axis=0))
    assert ens.spread(test).shape == (test.height,) and np.all(ens.spread(test) > 0)
    assert [m.seed for m in ens.members] == [11, 12, 13]
    # Member 0 is exactly the single model trained with the same seed.
    single = wp_nn.NNWPModel(params).fit(train, calib, seed=11)
    assert np.array_equal(members[0], single.predict(test))
    loaded = wp_nn.WPEnsemble.load(ens.save(tmp_path / "ens.joblib"))
    assert np.array_equal(loaded.predict_members(test), members)


def test_bootstrap_ensemble_members_see_different_games(toy):
    train, calib, test = toy
    params = {**SMALL, "max_epochs": 4}
    bagged = wp_nn.WPEnsemble(params, k=2, bootstrap=True).fit(train, calib, seed=11)
    plain = wp_nn.WPEnsemble(params, k=2).fit(train, calib, seed=11)
    assert bagged.predict_members(test).shape == (2, test.height)
    assert not np.array_equal(bagged.predict_members(test), plain.predict_members(test))


# ----------------------------------------------------------------------- leakage guards


def test_fit_refuses_test_seasons_and_stamps_the_model(tmp_path, toy, monkeypatch):
    monkeypatch.setattr(splits, "ACCESS_LOG", tmp_path / "access.jsonl")
    train, calib, test = toy
    quick = {**SMALL, "max_epochs": 2}
    leaky = logistic_league(40, 9, season=splits.test_seasons()[0])
    with pytest.raises(splits.LeakageError):
        wp_nn.NNWPModel(quick).fit(pl.concat([train, leaky]), calib, seed=1)
    with pytest.raises(splits.LeakageError):  # the early-stopping frame is guarded too
        wp_nn.NNWPModel(quick).fit(train, leaky, seed=1)
    with pytest.raises(splits.LeakageError):
        wp_nn.WPEnsemble(quick, k=1).fit(leaky, calib, seed=1)
    with pytest.raises(KeyError):
        wp_nn.NNWPModel(quick).fit(train.drop("season"), seed=1)
    model = wp_nn.NNWPModel(quick).fit(train, calib, seed=1)
    assert model.stamp["seasons"] == [2010] and model.stamp["n_games"] == 650
    again = wp_nn.NNWPModel.load(model.save(tmp_path / "nn.joblib"))
    assert again.stamp == model.stamp
    with pytest.raises(splits.LeakageError):  # toy test frame is the same season
        splits.assert_can_score(again.stamp, test, "a season the network was fitted on")
    splits.assert_can_score(again.stamp, logistic_league(10, 4, season=2012), "unseen")
    ens = wp_nn.WPEnsemble(quick, k=1).fit(train, calib, seed=1)
    path = ens.save(tmp_path / "ens.joblib")
    assert wp_nn.WPEnsemble.load(path).stamp == ens.stamp
    assert not (tmp_path / "access.jsonl").exists()


# -------------------------------------------------------------------------- real data


@pytest.mark.data
def test_small_network_on_real_plays(plays):
    cols = [*wp.STATE_COLUMNS, wp.LABEL, "game_id", "season"]
    train = plays.filter(pl.col("season").is_in([2018, 2019])).select(cols)
    held = plays.filter(pl.col("season") == 2021).select(cols)
    held = held.filter(pl.Series(wp.scoreable(held)) & pl.col(wp.LABEL).is_not_null())
    model = wp_nn.NNWPModel({"hidden": [32], "max_epochs": 6}).fit(train, seed=1)
    p, y = model.predict(held), held[wp.LABEL].to_numpy()
    assert wp.log_loss(y, p) < 0.50  # a coin flip is 0.693, the full models about 0.43
    assert wp.ece(y, p) < 0.05
    late = wp.state_frame(score_differential=[28, -28], game_seconds_remaining=120)
    blowout = model.predict(late)
    assert blowout[0] > 0.97 and blowout[1] < 0.03


@pytest.mark.data
def test_saved_models_agree_with_the_reported_numbers(plays):
    path, results = DATA / "models" / "wp_nn.joblib", DATA.parent / "results" / "m1_wp_nn.json"
    ens_path = DATA / "models" / "wp_nn_ensemble.joblib"
    if not (path.exists() and ens_path.exists() and results.exists()):
        pytest.skip("neural models not fitted (run scripts/fit_wp_nn.py)")
    import json

    reported = json.loads(results.read_text())
    if reported.get("features") != wp.FEATURES:
        pytest.skip("saved networks use an older feature list; rerun scripts/fit_wp_nn.py")
    test = plays.filter(pl.col("season").is_in([2024, 2025]))
    test = test.filter(
        pl.Series(wp.scoreable(test))
        & pl.col(wp.LABEL).is_not_null()
        & pl.col("wp").is_not_null()
        & pl.col("vegas_wp").is_not_null()
    )
    y = test[wp.LABEL].to_numpy()
    assert test.height == reported["test_rows"]["rows"]
    model = wp_nn.NNWPModel.load(path)
    assert wp.log_loss(y, model.predict(test)) == pytest.approx(
        reported["test"]["mlp"]["log_loss"], abs=1e-6
    )
    ens = wp_nn.WPEnsemble.load(ens_path)
    members = ens.predict_members(test)
    assert members.shape == (reported["ensemble"]["k"], test.height)
    assert wp.log_loss(y, members.mean(axis=0)) == pytest.approx(
        reported["test"]["mlp_ensemble"]["log_loss"], abs=1e-6
    )
