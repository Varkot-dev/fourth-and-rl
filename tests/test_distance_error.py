"""Tests for the M9 measurement-error model of yards to go.

The synthetic tests simulate plays whose true distance is known and recorded by the
module's rule, so they run without data: the offset estimator finds the known offset, the
kernel matches the simulated true distances, the deconvolution recovers the latent curve,
and the propagation flips the call exactly where a toy engine says it should. Tests
marked `data` need the cached plays.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.special import expit

from fourth_and_rl import distance_error as de
from fourth_and_rl.decision import DecisionEngine
from fourth_and_rl.wp import state_frame

UNIFORM = de.ErrorModel("uniform")
NARROW = de.ErrorModel("narrow", offset_kind="point", offset_mean=0.5)
KNOTS = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 16]


def truth(D: np.ndarray) -> np.ndarray:
    """A known conversion curve with a short-yardage cliff."""
    return expit(1.3 - 0.55 * np.log1p(D) - 0.9 * (1 - np.exp(-np.asarray(D) / 0.7)))


def attempts_frame(recorded: np.ndarray, converted: np.ndarray) -> pl.DataFrame:
    n = recorded.size
    return pl.DataFrame({"ydstogo": recorded, "gtg": np.zeros(n, int),
                         "yardline_100": np.full(n, 50), "down": np.full(n, 4),
                         "converted": converted})  # fmt: skip


def zero_offset(yl: np.ndarray, down: np.ndarray) -> np.ndarray:
    return np.zeros(np.size(yl))


# ------------------------------------------------------------------------- kernels


def test_kernel_bins_follow_the_recording_rule() -> None:
    # Offset fixed at 0.5: recorded 1 means 0 < D <= 1.5, recorded 3 means 2.5 < D <= 3.5.
    x, w = de.distance_kernel(1, False, NARROW, step=0.01)
    assert w.sum() == pytest.approx(1.0)
    assert x[w > 0].min() > 0 and x[w > 0].max() == pytest.approx(1.495, abs=0.01)
    x, w = de.distance_kernel(3, False, NARROW, step=0.01)
    assert x[w > 0].min() == pytest.approx(2.505, abs=0.01)
    assert x[w > 0].max() == pytest.approx(3.495, abs=0.01)
    # Goal to go uses its own fixed offset.
    edge = de.ErrorModel("edge", goal_to_go_offset=0.0)
    x, w = de.distance_kernel(2, True, edge, step=0.01)
    assert x[w > 0].min() == pytest.approx(1.005, abs=0.01)
    assert x[w > 0].max() == pytest.approx(1.995, abs=0.01)


def test_uniform_kernel_has_the_derived_shape() -> None:
    # Recorded 2 under a uniform offset: triangular on (1, 3), mean 2.
    x, w = de.distance_kernel(2, False, UNIFORM, step=0.005)
    assert (x * w).sum() == pytest.approx(2.0, abs=0.01)
    # Recorded 1: density 1 on (0, 1) and (2 - D) on (1, 2), total 1.5, mean 7/9.
    x, w = de.distance_kernel(1, False, UNIFORM, step=0.005)
    assert (x * w).sum() == pytest.approx(7 / 9, abs=0.01)
    assert w[x < 1].sum() == pytest.approx(1 / 1.5, abs=0.01)


def test_kernel_matches_simulated_true_distances() -> None:
    rng = np.random.default_rng(0)
    for model in (UNIFORM, de.ErrorModel("wide", spot_sd=0.25)):
        s = de.simulate_recorded(300_000, model, rng, true_distance_scale=50.0, max_distance=12)
        for d in (1, 2, 5):
            sim = s["true_distance"][s["recorded"] == d]
            x, w = de.distance_kernel(d, False, model, step=0.01)
            assert sim.mean() == pytest.approx((x * w).sum(), abs=0.03)
            assert np.quantile(sim, 0.9) == pytest.approx(
                x[np.searchsorted(np.cumsum(w), 0.9)], abs=0.05)


def test_kernel_table_quantiles_and_cells() -> None:
    tab = de.kernel_table(UNIFORM, max_d=10, step=0.02)
    cells = tab.cell(np.array([1, 2, 10, 40]), np.array([0, 1, 0, 0]))
    assert cells.tolist() == [0, 3, 18, 18]
    lo = de.kernel_quantiles(tab, tab.cell(np.array([2]), np.array([0])), 0.05)
    assert lo[0] == pytest.approx(1 + np.sqrt(0.1), abs=0.03)
    assert np.allclose(tab.weights.sum(axis=1), 1.0)


# --------------------------------------------------------------- offset estimator


@pytest.mark.parametrize("true_mean,draw", [
    (0.5, lambda r, n: r.uniform(0, 1, n)),
    (0.5, lambda r, n: np.full(n, 0.5)),
    (0.25, lambda r, n: r.beta(0.5, 1.5, n)),
    (0.75, lambda r, n: r.uniform(r.uniform(0, 1, n), 1.0)),
])  # fmt: skip
def test_offset_estimator_recovers_the_known_mean(true_mean: float, draw) -> None:
    rng = np.random.default_rng(1)
    n = 300_000
    end = rng.normal(-1.0, 5.0, n)  # yards past the line, smooth density
    k = de.record_end_spot(end, draw(rng, n))
    est = de.fit_line_offset(*de.line_offset_counts(k, 4))
    assert est["mean_offset"] == pytest.approx(true_mean, abs=0.03)
    assert est["se"] < 0.03


def test_recording_is_consistent_like_the_real_record() -> None:
    rng = np.random.default_rng(2)
    v = rng.uniform(0, 1, 50_000)
    end = rng.normal(0, 3, 50_000)
    k = de.record_end_spot(end, v)
    assert np.all(k[end >= 0] >= 0) and np.all(k[end < 0] <= -1)
    assert de.record_distance(np.array([0.2, 1.4, 1.6, 2.3]), np.array([0.5] * 4)).tolist() == [
        1, 1, 2, 2]


def test_weighted_counts_equal_repeated_rows() -> None:
    k = np.array([-1, 0, 0, 2, 5])
    w = np.array([2, 1, 3, 1, 4])
    _, a = de.line_offset_counts(k, 4, w)
    _, b = de.line_offset_counts(np.repeat(k, w), 4)
    assert np.array_equal(a, b)


# ------------------------------------------------------------------ deconvolution


def test_deconvolution_recovers_the_latent_curve() -> None:
    rng = np.random.default_rng(3)
    s = de.simulate_recorded(200_000, UNIFORM, rng, curve=truth)
    tab = de.kernel_table(UNIFORM, 20, 0.02)
    cells = de.aggregate_cells(attempts_frame(s["recorded"], s["converted"]), zero_offset, tab)
    curve = de.fit_latent_curve(cells, tab, KNOTS, penalty=1000.0)
    grid = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    assert np.abs(curve.p(grid) - truth(grid)).max() < 0.035
    assert np.abs(curve.p(grid[3:]) - truth(grid[3:])).max() < 0.02
    assert np.all(np.diff(curve.values) <= 1e-12)  # non-increasing
    # The curve averaged over the kernel reproduces the whole-yard rates.
    imp = de.implied_whole_yard(curve, tab, cells.cell, cells.offset)
    for d in (1, 2, 3):
        m = cells.cell == tab.cell(d, 0)
        assert imp[m][0] == pytest.approx(cells.y[m].sum() / cells.n[m].sum(), abs=0.01)


def test_rounding_biases_the_whole_yard_rate_and_deconvolution_corrects_it() -> None:
    # A recorded 1 averages true distances from 0 to about 1.5 (mean 7/9 under a uniform
    # offset), so the whole-yard rate at "1" overstates conversion at a true distance of
    # exactly 1 yard. The deconvolved curve moves back toward the truth there.
    rng = np.random.default_rng(4)
    s = de.simulate_recorded(200_000, UNIFORM, rng, curve=truth)
    att = attempts_frame(s["recorded"], s["converted"])
    tab = de.kernel_table(UNIFORM, 20, 0.02)
    curve = de.fit_latent_curve(de.aggregate_cells(att, zero_offset, tab), tab, KNOTS, 1000.0)
    naive = s["converted"][s["recorded"] == 1].mean()
    assert naive > truth(1.0) + 0.02
    assert abs(curve.p(1.0) - truth(1.0)) < abs(naive - truth(1.0))


def test_anchor_reproduces_the_target_probability() -> None:
    tab = de.kernel_table(UNIFORM, 10, 0.02)
    curve = de.LatentCurve(np.array([0.0, 1.0, 2.0, 10.0]), np.array([1.5, 0.5, 0.0, -1.0]))
    cells = tab.cell(np.array([1, 2, 3]), np.array([0, 0, 0]))
    target = np.array([0.7, 0.6, 0.2])
    c = de.anchor_offsets(curve, tab, cells, target)
    assert np.allclose(de.implied_whole_yard(curve, tab, cells, c), target, atol=1e-8)


# -------------------------------------------------------------------- propagation


def test_propagation_flips_exactly_where_the_break_even_lies() -> None:
    # Toy engine: converting is worth 0.6, failing 0.2, the kick 0.44, so going is better
    # when p > 0.6. A curve with p(D) = 0.6 exactly at D = 1.2 flips a recorded 1 (range
    # up to about 1.45 under the uniform kernel) but not a recorded 3 (always below 0.6).
    tab = de.kernel_table(UNIFORM, 10, 0.01)
    curve = de.LatentCurve(np.array([0.0, 1.2, 10.0]), np.array([1.0, 0.0, -2.0]))
    n = 3
    d = np.array([1, 3, 1])
    p_whole = de.implied_whole_yard(curve, tab, tab.cell(d, 0), np.array([0.4055, 0.4055, 2.0]))
    out = de.propagate(curve, tab, d, np.zeros(n, int), p_whole, np.full(n, 0.6),
                       np.full(n, 0.2), np.full(n, 0.44), coverage=0.9)  # fmt: skip
    assert out["flips"].tolist() == [True, False, False]
    # Integrated over the kernel the gain equals the whole-yard gain.
    whole = de.go_value(p_whole, 0.6, 0.2) - 0.44
    assert np.allclose(out["gain_mean"], whole, atol=1e-8)
    # Break-even probability and the share of the kernel past it.
    assert de.breakeven_probability(0.6, 0.2, 0.44) == pytest.approx(0.6)
    q = curve.p(tab.nodes[tab.cell(1, 0)], de.anchor_offsets(curve, tab, tab.cell(
        np.array([1]), np.array([0])), p_whole[:1])[0])  # fmt: skip
    expected_other = tab.weights[tab.cell(1, 0)][np.sign(q * 0.4 + 0.2 - 0.44)
                                                  != np.sign(whole[0])].sum()  # fmt: skip
    assert out["p_other_call"][0] == pytest.approx(expected_other)
    assert out["evpi"][0] > 0 and out["evpi"][1] == pytest.approx(0.0, abs=1e-12)


def test_summary_weights_equal_repeated_rows() -> None:
    tab = de.kernel_table(UNIFORM, 10, 0.02)
    curve = de.LatentCurve(np.array([0.0, 1.2, 10.0]), np.array([1.0, 0.0, -2.0]))
    d = np.array([1, 2, 3, 1])
    args = (np.full(4, 0.6), np.full(4, 0.2), np.array([0.44, 0.40, 0.30, 0.5]))
    p = np.array([0.6, 0.55, 0.5, 0.65])
    out = de.propagate(curve, tab, d, np.zeros(4, int), p, *args)
    w = np.array([2, 0, 1, 3])
    a = de.summarize_flips(out, weights=w)
    rep = np.repeat(np.arange(4), w)
    out_rep = de.propagate(curve, tab, d[rep], np.zeros(rep.size, int), p[rep],
                           *(x[rep] for x in args))  # fmt: skip
    b = de.summarize_flips(out_rep)
    for key in ("share_flip_in_range", "share_call_differs_at_true_distance",
                "mean_evpi_wp_points"):  # fmt: skip
        assert a[key] == pytest.approx(b[key])


class _ToyWP:
    def predict(self, df: pl.DataFrame) -> np.ndarray:
        p = 0.55 + 0.01 * df["score_differential"].to_numpy()
        return np.clip(p - 0.002 * (df["yardline_100"].to_numpy() - 50), 0.01, 0.99)


class _ToyComponents:
    def go_outcomes(self, yardline_100: int, ydstogo: int) -> dict:
        p = max(0.1, 0.8 - 0.05 * ydstogo)
        first_down, turnover = np.zeros(100), np.zeros(100)
        first_down[yardline_100 - ydstogo] = p
        turnover[100 - yardline_100] = 1 - p
        return {"p_convert": p, "p_touchdown": 0.0, "first_down": first_down,
                "turnover": turnover, "p_safety": 0.0}  # fmt: skip

    def p_field_goal(self, yardline_100: int) -> float:
        return 0.0

    def fg_miss_yardline(self, yardline_100: int) -> int:
        return 80

    def punt_outcomes(self, yardline_100: int) -> np.ndarray:
        out = np.zeros(100)
        out[min(140 - yardline_100, 80)] = 1.0
        return out

    def kickoff_start(self) -> np.ndarray:
        out = np.zeros(100)
        out[70] = 1.0
        return out

    def p_extra_point(self) -> float:
        return 1.0


def test_conversion_override_matches_the_linear_formula_in_a_real_engine() -> None:
    cfg = {"clock": {"use_fitted": False, "go_convert": 27, "go_fail": 5, "score": 5,
                     "field_goal": 4, "punt": 9, "kickoff": 2}, "punt_min_yardline_100": 31}
    base = DecisionEngine(_ToyWP(), _ToyComponents(), cfg)
    st = state_frame(down=4, ydstogo=np.array([2]), yardline_100=np.array([60]),
                     game_seconds_remaining=1800, score_differential=0)  # fmt: skip
    v = base.action_values(st)
    for p_new in (0.2, 0.55, 0.9):
        eng = DecisionEngine(_ToyWP(), de.ConversionOverride(_ToyComponents(),
                                                             lambda yl, d, p=p_new: p), cfg)
        w = eng.action_values(st)
        expected = de.go_value(p_new, v["wp_if_convert"][0], v["wp_if_fail"][0])
        assert w["go_wp"][0] == pytest.approx(expected, abs=1e-9)
        assert w["p_convert"][0] == pytest.approx(p_new)
        assert w["punt_wp"][0] == pytest.approx(v["punt_wp"][0])


def test_error_models_from_config() -> None:
    cfg = de.load_config()
    models = de.error_models(cfg, fitted_mean=0.44)
    names = [m.name for m in models]
    assert cfg["headline_model"] in names
    fitted = next(m for m in models if m.name == "fitted_mean")
    assert fitted.offset_mean == pytest.approx(0.44)
    with pytest.raises(ValueError):
        de.error_models(cfg)
    with pytest.raises(ValueError):
        de.ErrorModel("bad", offset_kind="gaussian")


# ------------------------------------------------------------------------- data


@pytest.mark.data
def test_real_record_shows_the_dip_and_the_fit_reproduces_whole_yard_rates() -> None:
    from fourth_and_rl import splits

    try:
        dev = splits.load_dev_plays()
    except FileNotFoundError:
        pytest.skip("development play file not built")
    est = de.pooled_line_offset(dev, (2, 10), (2, 4))
    assert 0.3 < est["mean_offset"] < 0.6
    att = de.conversion_attempts(dev)
    assert att.height > 50_000
    tab = de.kernel_table(UNIFORM, 30, 0.02)
    base = float(att["converted"].mean())
    cells = de.aggregate_cells(att, lambda yl, dn: np.zeros(np.size(yl)), tab)
    curve = de.fit_latent_curve(cells, tab, KNOTS + [20, 25, 32], 30.0)
    assert curve.info["converged"]
    imp = de.implied_whole_yard(curve, tab, cells.cell, cells.offset)
    assert (imp * cells.n).sum() / cells.n.sum() == pytest.approx(base, abs=0.01)
