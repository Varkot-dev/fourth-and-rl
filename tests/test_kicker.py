"""Tests for the selection-bias-aware field-goal make model (src/fourth_and_rl/kicker.py).

The synthetic world has selection built in: strong kickers are sent out for long kicks
more often. There, the raw long-range rate overstates an average kicker, and the model
must recover the true average curve. The `data` test fits the real training seasons and
is skipped when the cache is absent.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fourth_and_rl import kicker as kk
from fourth_and_rl import splits

TRUE_SD_U = 0.5
true_logit = kk.synthetic_true_logit
synthetic_attempts = kk.simulate_selected_attempts
synthetic_cfg = kk.synthetic_cfg


@pytest.fixture(scope="module")
def synthetic() -> tuple[pl.DataFrame, np.ndarray, kk.FGMakeModel]:
    att, u = synthetic_attempts()
    model = kk.FGMakeModel(synthetic_cfg()).fit(att)
    return att, u, model


def test_raw_rate_is_biased_and_model_recovers_the_average_curve(synthetic) -> None:
    att, _, model = synthetic
    long = att.filter(pl.col("distance") >= 55)
    d = long["distance"].to_numpy()
    truth = float(np.mean(1.0 / (1.0 + np.exp(-true_logit(d)))))
    raw = float(long["made"].mean())
    fitted_avg = float(np.mean(model.predict_attempts(long, kicker="average")))
    # Selection is built in, so the raw long-range rate overstates an average kicker.
    assert raw - truth > 0.04
    # The model's average-kicker rate on the same attempts is close to the truth.
    assert abs(fitted_avg - truth) < 0.03
    assert abs(fitted_avg - truth) < 0.4 * (raw - truth)
    # Plain random effects (no usage term) are biased here, because kicker strength is
    # correlated with distance; the usage term removes most of that bias.
    plain = kk.FGMakeModel(synthetic_cfg(usage=False)).fit(att)
    plain_avg = float(np.mean(plain.predict_attempts(long, kicker="average")))
    assert plain_avg - truth > 0.04
    # The whole average curve, not only the long attempts.
    grid = np.arange(20, 63)
    p_true = 1.0 / (1.0 + np.exp(-true_logit(grid)))
    assert np.max(np.abs(model.p_make(grid) - p_true)) < 0.05


def test_known_kicker_effects_are_recovered(synthetic) -> None:
    _, u, model = synthetic
    order = [int(k[1:]) for k in model.kicker_ids]
    # Total kicker effect = usage part + random effect; it should track the true strength.
    est = np.array([model.kicker_shift(k) for k in model.kicker_ids])
    assert np.corrcoef(est, u[order])[0, 1] > 0.85
    # The usage coefficient is positive: kickers sent out for longer kicks are stronger.
    assert model.usage_coef > 0
    # The spread of total strength across kickers is near the true 0.5.
    assert 0.35 < model.strength_sd() < 0.7
    # Shrinkage: estimated effects are less spread out than the true ones.
    assert np.std(est) < np.std(u)


def test_selection_decomposition_attributes_the_gap_to_the_kicker(synthetic) -> None:
    att, _, model = synthetic
    r = kk.selection_decomposition(model, att.filter(pl.col("distance") >= 55))
    total = sum(r["contributions"].values())
    assert total == pytest.approx(r["model_fit"] - r["selection_free"], abs=1e-12)
    assert r["contributions"]["kicker"] > 0.04
    assert abs(r["fit_error"]) < 0.03


def test_strength_quantiles_order_the_curves(synthetic) -> None:
    _, _, model = synthetic
    d = np.arange(20, 66)
    lo, mid, hi = (model.p_make(d, strength=q) for q in (0.1, 0.5, 0.9))
    assert np.all(hi > mid) and np.all(mid > lo)
    np.testing.assert_allclose(mid, model.p_make(d), atol=1e-12)


def test_make_rate_decreases_with_distance(synthetic) -> None:
    _, _, model = synthetic
    d = np.arange(18, 76)
    for q in (None, 0.1, 0.9):
        p = model.p_make(d, strength=q)
        assert np.all(np.diff(p) <= 0)
        assert np.all(np.diff(p[d >= 30]) < 0)


def test_fit_refuses_test_holdout_and_calibration_seasons() -> None:
    att, _ = synthetic_attempts(n_kickers=10, per_kicker=20)
    cfg = synthetic_cfg()
    for season in (splits.test_seasons()[0], splits.holdout_seasons()[0],
                   splits.calibration_seasons()[0]):  # fmt: skip
        bad = att.with_columns(
            pl.when(pl.int_range(pl.len()) == 0).then(season).otherwise(pl.col("season"))
            .alias("season")
        )  # fmt: skip
        with pytest.raises(splits.LeakageError):
            kk.FGMakeModel(cfg).fit(bad)


def test_kicker_download_refuses_the_holdout(tmp_path) -> None:
    with pytest.raises(splits.LeakageError):
        kk.download_kicker_ids([splits.holdout_seasons()[0]], tmp_path)


def test_natural_spline_is_linear_beyond_the_outer_knots() -> None:
    knots = [23, 32, 40, 47, 55]
    x = np.arange(56.0, 80.0)
    basis = kk.natural_spline_basis(x, knots)
    second = np.diff(basis, n=2, axis=0)
    np.testing.assert_allclose(second, 0.0, atol=1e-9)
    left = kk.natural_spline_basis(np.arange(10.0, 23.0), knots)
    np.testing.assert_allclose(np.diff(left, n=2, axis=0), 0.0, atol=1e-9)


def test_heave_flag_marks_only_a_last_play_kick() -> None:
    base = {"game_id": "g", "game_half": "Half1"}
    rows = [
        # first half: a kick with 3 s left and nothing after it is a heave
        {**base, "order_sequence": 1, "play_type": "pass", "half_seconds_remaining": 20.0},
        {**base, "order_sequence": 2, "play_type": "field_goal", "half_seconds_remaining": 3.0},
        # second half: a kick with 4 s left followed by a kickoff is not
        {**base, "game_half": "Half2", "order_sequence": 3, "play_type": "field_goal",
         "half_seconds_remaining": 4.0},
        {**base, "game_half": "Half2", "order_sequence": 4, "play_type": "kickoff",
         "half_seconds_remaining": 0.0},
        # overtime walk-off kicks are never heaves
        {**base, "game_half": "Overtime", "order_sequence": 5, "play_type": "field_goal",
         "half_seconds_remaining": 2.0},
    ]  # fmt: skip
    out = kk.heave_flags(pl.DataFrame(rows), max_half_seconds=10)
    assert out["heave"].to_list() == [False, True, False, False, False]


def test_shapley_shares_add_up() -> None:
    f = ("a", "b", "c")
    rng = np.random.default_rng(1)
    vals = {}
    for i in range(8):
        s = frozenset(x for j, x in enumerate(f) if i >> j & 1)
        vals[s] = float(rng.random())
    sh = kk.shapley(vals, f)
    assert sum(sh.values()) == pytest.approx(vals[frozenset(f)] - vals[frozenset()])


def test_cap_and_support(synthetic) -> None:
    _, _, model = synthetic
    model.support_distance = 58
    assert model.in_support(55) and not model.in_support(60)
    assert not model.in_support(50, season=2030)
    capped = model.capped_rate(np.array([0.99, 0.01]), np.array([60, 60]))
    assert capped[0] < 0.99 and capped[1] == pytest.approx(0.01)
    model.support_distance = float("inf")


@pytest.mark.data
def test_real_training_fit() -> None:
    if not splits.dev_path().exists():
        pytest.skip("development play file not built")
    train = list(splits.train_seasons())
    if not all((kk.KICKER_DIR / f"kickers_{s}.parquet").exists() for s in train):
        pytest.skip("kicker id cache not built (run scripts/questions/fg_selection_bias.py)")
    dev = splits.load_dev_plays().filter(pl.col("season").is_in(train))
    att = kk.field_goal_attempts(dev, kk.load_kicker_ids(train))
    model = kk.FGMakeModel().fit(att)
    d = np.arange(18, 76)
    p = model.p_make(d)
    assert np.all(np.diff(p) <= 1e-12)
    assert 0.6 < model.p_make(50) < 0.8
    assert model.p_make(55, strength=0.9) > model.p_make(55)
    assert model.beta[model.feature_names.index("outdoor_wind")] < 0
