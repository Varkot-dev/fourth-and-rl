"""Tests for the blown-lead reference, path extraction, tests and simulator (M3b)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fourth_and_rl import blown_leads as bl

# ------------------------------------------------------------------ theory


def test_even_game_formula() -> None:
    for x in (0.5, 0.6, 2 / 3, 0.9, 0.99):
        assert bl.peak_sf(x, 0.5) == pytest.approx((1 - x) / x)
    assert bl.peak_sf(0.9, 0.5) == pytest.approx(1 / 9)
    assert bl.peak_sf(0.99, 0.5) == pytest.approx(1 / 99)


def test_survival_is_one_at_start_and_zero_at_one() -> None:
    p0 = np.array([0.1, 0.35, 0.5, 0.8, 0.95])
    assert np.allclose(bl.peak_sf(p0, p0), 1.0)
    assert np.allclose(bl.peak_sf(p0 - 0.05, p0), 1.0)
    assert np.allclose(bl.peak_sf(1.0, p0), 0.0)
    assert np.allclose(bl.peak_cdf(p0, p0), 0.0)


def test_general_formula_and_reach_probability() -> None:
    # reach x with probability p0/x, then lose with probability 1 - x
    p0, x = 0.3, 0.75
    joint = bl.reach_prob(x, p0) * (1 - x)
    assert bl.reach_prob(x, p0) == pytest.approx(0.4)
    assert bl.peak_sf(x, p0) == pytest.approx(joint / (1 - p0))
    assert bl.peak_sf(x, p0) == pytest.approx(0.3 * 0.25 / (0.75 * 0.7))


def test_survival_is_monotone() -> None:
    x = np.linspace(0.01, 0.999, 300)
    for p0 in (0.2, 0.5, 0.9):
        sf = bl.peak_sf(x, p0)
        assert np.all(np.diff(sf) <= 1e-12)
        assert np.all((sf >= 0) & (sf <= 1))


def test_quantile_inverts_cdf() -> None:
    u = np.linspace(0.0, 0.999, 50)
    for p0 in (0.15, 0.5, 0.85):
        x = bl.peak_quantile(u, p0)
        assert np.allclose(bl.peak_cdf(x, p0), u)
    assert bl.peak_quantile(0.0, 0.4) == pytest.approx(0.4)
    assert bl.peak_quantile(0.5, 0.5) == pytest.approx(2 / 3)  # half of losers reach 67%


def test_pit_of_exact_draws_is_uniform() -> None:
    rng = np.random.default_rng(0)
    p0 = rng.uniform(0.1, 0.9, 5000)
    peaks = bl.peak_quantile(rng.random(5000), p0)
    res = bl.ks_uniform(bl.peak_pit(peaks, p0))
    assert res["ks_two_sided_p"] > 0.01


def test_game_level_cdf_matches_paper_table() -> None:
    # Table S4.1 of the paper: favorite at 2/3
    assert 1 - bl.game_peak_cdf(0.5, 2 / 3) == pytest.approx(2 / 3)
    assert 1 - bl.game_peak_cdf(2 / 3, 2 / 3) == pytest.approx(1 / 2)
    assert 1 - bl.game_peak_cdf(0.75, 2 / 3) == pytest.approx(1 / 3)
    assert 1 - bl.game_peak_cdf(0.9, 0.6) == pytest.approx(1 / 9)
    # either team's starting value gives the same answer
    assert bl.game_peak_cdf(0.8, 0.3) == pytest.approx(bl.game_peak_cdf(0.8, 0.7))


def test_game_level_cdf_is_the_mixture_of_the_two_losers() -> None:
    p0 = 0.7
    for x in (0.2, 0.4, 0.6, 0.75, 0.95):
        mix = (1 - p0) * bl.peak_cdf(x, p0) + p0 * bl.peak_cdf(x, 1 - p0)
        assert bl.game_peak_cdf(x, p0) == pytest.approx(mix)


# ------------------------------------------------------------------ simulation


def test_fine_step_martingale_matches_formula() -> None:
    rng = np.random.default_rng(1)
    peaks = bl.simulate_loser_peaks(0.5, n_steps=4000, n_paths=24000, rng=rng)
    n = peaks.size
    assert 0.47 < n / 24000 < 0.53  # the team loses about half the time
    for x in (0.6, 0.75):
        expected = bl.peak_sf(x, 0.5)
        se = np.sqrt(expected * (1 - expected) / n)
        # reporting on a grid can only hide peaks: allow a small downward bias
        assert expected - 5 * se - 0.02 < (peaks >= x).mean() < expected + 5 * se


def test_coarse_steps_give_fewer_extreme_peaks() -> None:
    rng = np.random.default_rng(2)
    coarse = bl.simulate_loser_peaks(0.5, 20, 40000, rng)
    fine = bl.simulate_loser_peaks(0.5, 400, 40000, rng)
    formula = bl.peak_sf(0.9, 0.5)
    assert (coarse >= 0.9).mean() < (fine >= 0.9).mean() < formula


def test_simulated_paths_are_a_calibrated_martingale() -> None:
    rng = np.random.default_rng(3)
    paths, won = bl.simulate_paths(0.3, 60, 40000, rng, gap_shape=1.0)
    assert np.allclose(paths[:, 0], 0.3)
    assert abs(won.mean() - 0.3) < 0.01
    # the mean stays at the starting value at every step
    assert np.max(np.abs(paths.mean(axis=0) - 0.3)) < 0.012
    # calibration late in the game: among values near 0.8, about 80% win
    late = paths[:, 45]
    sel = (late > 0.75) & (late < 0.85)
    assert abs(won[sel].mean() - late[sel].mean()) < 0.03


def _overconfident(p: np.ndarray, power: float = 1.8) -> np.ndarray:
    """Push probabilities away from 0.5 (multiply the log odds)."""
    z = np.log(p / (1 - p)) * power
    return 1 / (1 + np.exp(-z))


def test_pit_uniform_for_calibrated_and_detects_overconfidence() -> None:
    rng = np.random.default_rng(4)
    n_steps, n_paths = 1500, 6000
    p0 = 0.5
    paths, won = bl.simulate_paths(p0, n_steps, n_paths, rng)
    team = np.where(won[:, None], 1 - paths, paths)  # the loser's path
    peak = team.max(axis=1)
    calibrated = bl.ks_uniform(bl.peak_pit(peak, np.full(n_paths, p0)))
    # fine steps: close to uniform, and no excess of extreme peaks
    assert calibrated["ks_two_sided"] < 0.06
    assert calibrated["d_upper"] < 0.025

    over = _overconfident(np.clip(team, 1e-9, 1 - 1e-9))
    peak_over = over.max(axis=1)
    res = bl.ks_uniform(bl.peak_pit(peak_over, over[:, 0]))
    assert res["d_upper"] > 0.1
    assert res["d_upper_p_iid"] < 1e-6
    # too many extreme peaks, and the count test says so
    counts = bl.extreme_peak_counts(
        peak_over >= 0.95, bl.peak_sf(0.95, over[:, 0]), np.random.default_rng(0), 2000
    )
    assert counts["observed"] > counts["expected_interval95"][1]
    assert counts["p_upper"] < 1e-4


def test_discrete_reference_reproduces_its_own_simulation() -> None:
    rng = np.random.default_rng(5)
    ref = bl.build_discrete_reference(np.array([0.3, 0.4, 0.5]), 50, 60000, rng, gap_shape=1.0)
    peaks = bl.simulate_loser_peaks(0.4, 50, 30000, rng, gap_shape=1.0)
    u = ref.pit(peaks, np.full(peaks.size, 0.4), rng)
    assert bl.ks_uniform(u)["ks_two_sided"] < 0.02
    # the simulated tail is below the continuous formula (the paper's Theorem 3)
    assert ref.sf(0.9, 0.4)[0] < bl.peak_sf(0.9, 0.4)
    assert ref.sf(0.2, 0.4)[0] == pytest.approx(1.0)
    # between grid values the reference is still close
    peaks = bl.simulate_loser_peaks(0.45, 50, 30000, rng, gap_shape=1.0)
    assert abs((peaks >= 0.8).mean() - ref.sf(0.8, 0.45)[0]) < 0.012


def test_poisson_binomial_interval_and_counts() -> None:
    rng = np.random.default_rng(6)
    probs = np.full(1000, 0.1)
    lo, hi = bl.poisson_binomial_interval(probs, rng, 5000)
    assert 78 <= lo <= 84 and 116 <= hi <= 122
    hit = np.zeros(1000, bool)
    hit[:100] = True
    res = bl.extreme_peak_counts(hit, probs, rng, 2000)
    assert res["observed"] == 100 and res["expected"] == pytest.approx(100.0)
    assert abs(res["z"]) < 1e-9


def test_pigeonhole_weights_keep_season_totals() -> None:
    rng = np.random.default_rng(7)
    season = np.repeat([2020, 2021], 6)
    home = np.array(["A", "B", "C", "A", "B", "C"] * 2)
    away = np.array(["B", "C", "A", "C", "A", "B"] * 2)
    w = bl.pigeonhole_weights(season, home, away, 200, rng)
    assert w.shape == (200, 12)
    assert np.all(w >= 0)
    totals = w[:, :6].sum(axis=1)
    assert np.allclose(totals[totals > 0], 6.0)
    u = np.linspace(0.02, 0.98, 12)
    res = bl.pit_bootstrap_test(u, w)
    assert 0 <= res["d_upper_p_bootstrap"] <= 1


# ------------------------------------------------------------------ path extraction


def _tiny_game(rows: list[tuple], game_id: str, result: int) -> pl.DataFrame:
    """rows: (posteam, wp, play_type, qtr, seconds, posteam_score, defteam_score)."""
    n = len(rows)
    return pl.DataFrame(
        {
            "game_id": [game_id] * n,
            "play_id": [float(10 * (i + 1)) for i in range(n)],
            "order_sequence": [float(i + 1) for i in range(n)],
            "season": [2024] * n,
            "week": [1] * n,
            "home_team": ["HOM"] * n,
            "away_team": ["AWY"] * n,
            "posteam": [r[0] for r in rows],
            "play_type": [r[2] for r in rows],
            "result": [result] * n,
            "home_score": [20 if result < 0 else 24] * n,
            "away_score": [23 if result < 0 else 20] * n,
            "qtr": [float(r[3]) for r in rows],
            "game_seconds_remaining": [float(r[4]) for r in rows],
            "posteam_score": [float(r[5]) for r in rows],
            "defteam_score": [float(r[6]) for r in rows],
            "desc": [f"play {i + 1}" for i in range(n)],
            "wp": [r[1] for r in rows],
        },
        schema_overrides={"wp": pl.Float64, "play_type": pl.String},
    )


@pytest.fixture()
def tiny_plays() -> pl.DataFrame:
    # Home team loses 20-23 (result = -3). Home win probability by play:
    # 0.60, 0.70, (away ball 0.20 -> 0.80), (junk row), (null), (away 0.08 -> 0.92), 0.55,
    # (away 0.70 -> 0.30)
    lost = _tiny_game(
        [
            ("HOM", 0.60, "kickoff", 1, 3600, 0, 0),
            ("HOM", 0.70, "run", 1, 3000, 0, 0),
            ("AWY", 0.20, "pass", 2, 2000, 0, 7),
            ("HOM", 0.01, None, 2, 1900, 7, 0),  # timeout row with the wrong posteam: dropped
            ("AWY", None, "pass", 3, 1500, 3, 14),  # missing value: dropped
            ("AWY", 0.08, "punt", 3, 1000, 10, 20),
            ("HOM", 0.55, "pass", 4, 300, 20, 17),
            ("AWY", 0.70, "field_goal", 4, 5, 20, 20),
        ],
        "2024_01_AWY_HOM",
        result=-3,
    )
    # Home team wins: the away team is the loser and its path is 1 - home.
    won = _tiny_game(
        [
            ("AWY", 0.45, "kickoff", 1, 3600, 0, 0),
            ("AWY", 0.75, "run", 2, 2500, 10, 0),
            ("HOM", 0.35, "pass", 3, 1500, 7, 10),
            ("HOM", 0.90, "run", 4, 100, 24, 20),
        ],
        "2024_01_BBB_HOM",
        result=4,
    )
    tied = _tiny_game(
        [("HOM", 0.5, "run", 1, 3600, 0, 0), ("AWY", 0.5, "run", 5, 10, 20, 20)],
        "2024_01_TIE_HOM",
        result=0,
    )
    return pl.concat([lost, won, tied])


def test_home_paths_flip_perspective(tiny_plays: pl.DataFrame) -> None:
    hp = bl.home_paths(tiny_plays, "wp")
    lost = hp.filter(pl.col("game_id") == "2024_01_AWY_HOM")
    assert lost["home_wp_path"].to_list() == pytest.approx([0.60, 0.70, 0.80, 0.92, 0.55, 0.30])
    assert lost["step"].to_list() == [0, 1, 2, 3, 4, 5]
    assert lost["home_pts"].to_list() == [0, 0, 7, 20, 20, 20]
    assert "2024_01_TIE_HOM" not in hp["game_id"].to_list()


def test_extract_loser_paths_tiny_game(tiny_plays: pl.DataFrame) -> None:
    out = bl.extract_loser_paths(tiny_plays, "wp", min_plays=1)
    assert out.height == 2  # the tie is dropped
    g = out.row(by_predicate=pl.col("game_id") == "2024_01_AWY_HOM", named=True)
    assert (g["loser"], g["winner"]) == ("HOM", "AWY")
    assert (g["loser_score"], g["winner_score"]) == (20, 23)
    assert g["p0"] == pytest.approx(0.60)
    assert g["peak"] == pytest.approx(0.92)  # reached while the other team had the ball
    assert g["peak_step"] == 3
    assert g["peak_qtr"] == 3 and g["peak_seconds_remaining"] == 1000
    assert (g["peak_loser_pts"], g["peak_winner_pts"]) == (20, 10)
    assert g["peak_desc"] == "play 6"
    assert g["n_plays"] == 6
    assert g["max_jump"] == pytest.approx(0.37)  # 0.92 -> 0.55
    assert g["last_wp"] == pytest.approx(0.30)

    h = out.row(by_predicate=pl.col("game_id") == "2024_01_BBB_HOM", named=True)
    assert (h["loser"], h["winner"]) == ("AWY", "HOM")
    # away path: 0.45, 0.75, 1 - 0.35 = 0.65, 1 - 0.90 = 0.10
    assert h["p0"] == pytest.approx(0.45)
    assert h["peak"] == pytest.approx(0.75)
    assert h["peak_step"] == 1
    assert (h["peak_loser_pts"], h["peak_winner_pts"]) == (10, 0)


def test_min_plays_and_missing_column(tiny_plays: pl.DataFrame) -> None:
    assert bl.extract_loser_paths(tiny_plays, "wp", min_plays=5).height == 1
    with pytest.raises(KeyError):
        bl.home_paths(tiny_plays, "wp_missing")


def test_first_crossings_and_jumps(tiny_plays: pl.DataFrame) -> None:
    hp = bl.home_paths(tiny_plays, "wp")
    cr = bl.first_crossings(hp, 0.75).sort("game_id", "side")
    rows = {(r["game_id"], r["side"]): r for r in cr.iter_rows(named=True)}
    # home side of the lost game first reaches 0.75 at 0.80, and lost
    a = rows[("2024_01_AWY_HOM", "home")]
    assert a["p_cross"] == pytest.approx(0.80) and a["won"] is False
    # away side of the other game reaches 0.75 and lost; home side reaches 0.90 and won
    assert rows[("2024_01_BBB_HOM", "away")]["p_cross"] == pytest.approx(0.75)
    assert rows[("2024_01_BBB_HOM", "home")]["won"] is True
    assert ("2024_01_AWY_HOM", "away") not in rows  # the away side peaked at 0.70
    res = bl.crossing_test(cr, np.random.default_rng(0), 500)
    assert res["observed"] == 2 and res["episodes"] == 3
    assert res["expected"] == pytest.approx(0.20 + 0.25 + 0.10)
    jumps = bl.jump_sizes(hp)
    assert jumps.size == 5 + 3
    assert jumps.max() == pytest.approx(0.55)  # away game: 0.35 -> 0.90 for the home side


def test_rankings_differ_when_starts_differ() -> None:
    games = pl.DataFrame(
        {
            "game_id": ["fav", "dog", "mid", "tie_late", "tie_early"],
            "p0": [0.90, 0.20, 0.50, 0.50, 0.50],
            "peak": [0.990, 0.950, 0.900, 0.9702, 0.9698],
            "peak_seconds_remaining": [600.0, 600.0, 600.0, 30.0, 2000.0],
        }
    )
    g = bl.add_reference_columns(games)
    assert g["pit"].to_numpy() == pytest.approx(1 - g["tail_prob"].to_numpy())
    ranks = bl.rank_collapses(g, top=5, peak_decimals=2)
    assert ranks["by_peak"]["game_id"][0] == "fav"
    assert ranks["by_surprise"]["game_id"][0] == "dog"  # 95% from a 20% start is rarer
    # the two 0.97 games tie after rounding; the later peak ranks first
    order = ranks["by_peak"]["game_id"].to_list()
    assert order.index("tie_late") < order.index("tie_early")
    assert ranks["overlap"] == 5


# ------------------------------------------------------------------ real data


@pytest.mark.data
def test_conversion_matches_nflfastr_home_wp(plays: pl.DataFrame) -> None:
    """Our posteam-to-home conversion reproduces the home_wp column nflverse ships."""
    hp = bl.home_paths(plays, "wp")
    ref = plays.select("game_id", "play_id", "home_wp")
    joined = hp.join(ref, on=["game_id", "play_id"], how="left")
    assert (joined["home_wp_path"] - joined["home_wp"]).abs().max() < 1e-9


@pytest.mark.data
def test_real_loser_paths_are_sane(plays: pl.DataFrame) -> None:
    games = bl.add_reference_columns(bl.extract_loser_paths(plays, "vegas_wp"))
    n_nontied = plays.filter(pl.col("result") != 0)["game_id"].n_unique()
    assert games.height == n_nontied
    assert games["game_id"].n_unique() == games.height
    assert (games["peak"] >= games["p0"]).all()
    assert games["peak"].max() < 1.0 and games["p0"].min() > 0.0
    assert games["n_plays"].min() >= 50
    assert ((games["pit"] >= 0) & (games["pit"] <= 1)).all()
    assert (games["loser"] != games["winner"]).all()
    # the loser scored fewer points
    assert (games["loser_score"] < games["winner_score"]).all()
    # a known collapse: Atlanta led Super Bowl LI 28-3 and lost
    sb = games.row(by_predicate=pl.col("game_id") == "2016_21_NE_ATL", named=True)
    assert sb["loser"] == "ATL" and sb["peak"] > 0.98


@pytest.mark.data
def test_real_paths_do_not_show_excess_extreme_peaks(plays: pl.DataFrame) -> None:
    """The finding recorded in results/m3b_blown_leads.json: no excess above the reference."""
    games = bl.add_reference_columns(bl.extract_loser_paths(plays, "wp"))
    peak, p0 = games["peak"].to_numpy(), games["p0"].to_numpy()
    observed = int((peak >= 0.95).sum())
    assert observed < bl.peak_sf(0.95, p0).sum()
    assert bl.ks_uniform(games["pit"].to_numpy())["d_upper"] < 0.01
