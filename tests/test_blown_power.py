import numpy as np

from fourth_and_rl.blown_power import crossing_z, distort, run_power, simulate_league


def test_distort_identity_and_direction():
    p = np.array([0.2, 0.5, 0.8])
    assert np.allclose(distort(p, 1.0), p)
    over = distort(p, 1.5)
    assert over[0] < 0.2 and np.isclose(over[1], 0.5) and over[2] > 0.8


def test_calibrated_league_has_centered_crossing_z():
    # First crossings are exact for a calibrated martingale even with jumps,
    # so z should average near zero across leagues.
    rng = np.random.default_rng(1)
    zs = []
    for _ in range(40):
        lg = simulate_league(500, 60, rng, gap_shape=1.2, p0_sd=0.55)
        zs.append(crossing_z(lg.p, lg.home_won, [0.7, 0.9]))
    zs = np.array(zs)
    assert np.all(np.abs(zs.mean(axis=0)) < 0.5)
    assert np.all(np.abs(zs.std(axis=0) - 1.0) < 0.35)


def test_overconfidence_pushes_z_up():
    rng = np.random.default_rng(2)
    lg = simulate_league(3000, 60, rng, gap_shape=1.2, p0_sd=0.55)
    z_cal = crossing_z(lg.p, lg.home_won, [0.8])[0]
    z_over = crossing_z(distort(lg.p, 1.6), lg.home_won, [0.8])[0]
    assert z_over > z_cal + 2.0


def test_run_power_shapes_and_monotone_in_k():
    rng = np.random.default_rng(3)
    out = run_power(
        n_games=800, ks=[1.0, 1.6], thresholds=[0.7, 0.9], n_null=40, n_rep=30,
        n_steps=50, gap_shape=1.2, p0_sd=0.55, rng=rng,
    )
    pw = out["power_first_crossing"]
    assert len(pw) == 2 and pw[1] > pw[0]
    assert out["power_first_crossing"][0] <= 0.3
