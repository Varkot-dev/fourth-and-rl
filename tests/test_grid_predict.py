"""Tests for the grid predictor and the R1 policy arithmetic (`fourth_and_rl.overopt`)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pytest
from test_mdp import toy_football

from fourth_and_rl import grid_predict as gpm
from fourth_and_rl import overopt as oo


@pytest.fixture(scope="module")
def toy_proxy() -> tuple:
    m = toy_football(seed=4)
    rng = np.random.default_rng(0)
    xs = [gpm.slice_features(m, t) for t in range(1, m.clock.n_ticks + 1)]
    x = np.concatenate(xs)
    z = 0.3 * x[:, 1] - 0.4 * (x[:, 4] - 3) + 0.2 * x[:, 5] - 0.1 * x[:, 6] + 0.05 * x[:, 0]
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-z))).astype(float)
    reg = lgb.LGBMRegressor(objective="cross_entropy", num_leaves=16, n_estimators=40,
                            min_child_samples=5, verbose=-1, n_jobs=1)  # fmt: skip
    reg.fit(x, y)
    return m, reg.booster_


def test_grid_predictor_matches_booster_predict_on_every_tick(toy_proxy: tuple) -> None:
    m, bst = toy_proxy
    gp = gpm.GridPredictor(bst)
    worst = 0.0
    for t in range(1, m.clock.n_ticks + 1):
        grid = gpm.slice_values(gp, m, t, dtype=np.float64)
        ref = bst.predict(gpm.slice_features(m, t)).reshape(grid.shape)
        worst = max(worst, float(np.abs(grid - ref).max()))
        raw = bst.predict(gpm.slice_features(m, t), raw_score=True).reshape(grid.shape)
        assert np.allclose(gpm.slice_raw(gp, m, t), raw, rtol=0, atol=1e-12)
    assert worst < 1e-12


def test_grid_predictor_round_trips_through_a_model_file(toy_proxy: tuple, tmp_path) -> None:
    m, bst = toy_proxy
    path = tmp_path / "proxy.txt"
    bst.save_model(str(path))
    a = gpm.slice_values(gpm.GridPredictor(path), m, 7)
    b = gpm.slice_values(gpm.GridPredictor(bst), m, 7)
    assert np.array_equal(a, b) and a.dtype == np.float32


def test_tilt_limits_and_support() -> None:
    p = np.array([[0.6, 0.3, 0.1], [0.5, 0.5, 0.0]])
    q = np.array([[0.40, 0.45, 0.41], [0.2, 0.1, 0.9]])
    assert np.allclose(oo.tilt(p, q, 0.0), p)
    hot = oo.tilt(p, q, 1e6)
    assert np.allclose(hot, [[0, 1, 0], [1, 0, 0]])  # the punt has no coach mass in row 2
    assert np.allclose(oo.tilt(p, q, 10.0).sum(-1), 1.0)
    assert np.allclose(oo.kl(p, p), 0.0)
    pi = oo.tilt(p, q, 30.0)
    expect = (pi * np.log(np.where(pi > 0, pi, 1) / np.where(p > 0, p, 1))).sum(-1)
    assert np.allclose(oo.kl(pi, p), expect)


def test_ties_are_handled_explicitly() -> None:
    allowed = np.array([[True, True, True], [True, True, False], [True, True, True]])
    q = np.array([[0.5, 0.5, 0.1], [0.3, 0.2, 0.9], [0.4, 0.4 + 1e-9, 0.1]])
    probs, tied = oo.greedy(q, allowed, "uniform")
    assert np.allclose(probs, [[0.5, 0.5, 0], [1, 0, 0], [0, 1, 0]])
    assert tied.tolist() == [True, False, False]
    probs, tied = oo.greedy(q, allowed, "uniform", atol=1e-6)
    assert tied.tolist() == [True, False, True] and np.allclose(probs[2], [0.5, 0.5, 0])
    with pytest.raises(ValueError):
        oo.greedy(q, allowed, "first")
    # Disagreement: true best is action 0 in every row except the tie in row 0.
    q_true = np.array([[0.5, 0.5, 0.1], [0.3, 0.2, 0.9], [0.6, 0.4, 0.1]])
    q_prox = np.array([[0.1, 0.5, 0.1], [0.1, 0.2, 0.9], [0.4, 0.5, 0.5]])
    uni = oo.disagreement(q_true, q_prox, allowed, "uniform")
    assert uni["n_states"] == 3 and uni["n_tied_true"] == 1
    # Row 0: the proxy picks 1, which is tied best: no miss. Row 1: a miss worth 0.1
    # (the punt is not allowed). Row 2: the proxy ties 1 and 2, both misses; a uniform draw
    # is worth 0.25 against the best 0.6.
    assert uni["disagree"] == pytest.approx(2 / 3)
    assert uni["mean_regret"] == pytest.approx((0 + 0.1 + 0.35) / 3)
    exc = oo.disagreement(q_true, q_prox, allowed, "exclude")
    assert exc["n_states"] == 1 and exc["n_tied_proxy"] == 1
    assert exc["disagree"] == pytest.approx(1.0) and exc["mean_regret"] == pytest.approx(0.1)
