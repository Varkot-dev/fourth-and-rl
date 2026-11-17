"""Tests for M10, the cost of predictability.

Synthetic tests run in CI without data: tendencies and history never look ahead, the
fixed-effects estimator recovers a known cost of predictability that a raw comparison
gets wrong, shrinkage recovers a known spread of true team values, and the call models
find predictability planted in the play sequence. Tests marked `data` use the real
development table and are skipped when it is absent.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fourth_and_rl import predictability as pr
from fourth_and_rl import run_pass as rp

CFG = pr.load_config()


def synthetic_plays(n_games: int = 60, plays: int = 30, seed: int = 0,
                    alternate: float = 0.0) -> pl.DataFrame:  # fmt: skip
    """Called plays for two offenses over several seasons. With `alternate` > 0 the call
    repeats the opposite of the previous call with that probability (a planted
    sequence habit a defense could learn)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        season = 2010 + g // 20
        week = g % 20 + 1
        for team, opp in (("A", "B"), ("B", "A")):
            prev = None
            for i in range(plays):
                down = int(rng.integers(1, 4))
                if prev is not None and rng.random() < alternate:
                    call = 1 - prev
                else:
                    call = int(rng.random() < (0.4 if down == 1 else 0.7))
                yards = float(rng.integers(-2, 12))
                epa = float(rng.normal(0.05 if call else -0.05, 1))
                rows.append({
                    "game_id": f"{season}_{week:02d}_{g}", "posteam": team, "defteam": opp,
                    "season": season, "week": week, "fixed_drive": float(i // 6 + 1),
                    "order_sequence": float(i), "play_id": float(i), "is_pass": call,
                    "epa": epa, "success": float(epa > 0), "yards": yards,
                    "incomplete_pass": float(call and yards == 0), "first_down": 0.0,
                    "sack": 0.0, "turnover": 0.0, "down": float(down), "ydstogo": 10.0,
                    "yardline_100": 60.0, "neutral": True, "early": down <= 2,
                })  # fmt: skip
                prev = call
    df = pl.DataFrame(rows).sort("game_id", "posteam", "order_sequence")
    return df.with_row_index("row")


# ---- no look-ahead ----------------------------------------------------------------


def test_decayed_prior_sums_use_earlier_rows_only() -> None:
    key = np.array(["A", "B", "A", "A", "B"])
    season = np.array([2010, 2010, 2010, 2011, 2011])
    x = np.array([[1.0], [10.0], [2.0], [4.0], [20.0]])
    out = pr.decayed_prior_sums(key, season, x, decay=0.5, carry=0.1)
    # A: first game nothing; second 0.5 * 1; third (0.5 * (0.5 + 2)) * 0.1 for the new season.
    assert out[:, 0].tolist() == pytest.approx([0.0, 0.0, 0.5, 0.125, 0.5])
    # Changing a row's own value or any later value never changes it or earlier rows.
    for i in range(len(key)):
        x2 = x.copy()
        x2[i:] += 100.0
        out2 = pr.decayed_prior_sums(key, season, x2, 0.5, 0.1)
        assert np.allclose(out2[: i + 1], out[: i + 1])


def test_team_tendencies_ignore_the_game_itself_and_later_games() -> None:
    df = synthetic_plays(n_games=40, seed=1)
    base = pr.team_tendencies(df, CFG).sort("game_id", "posteam")
    games = sorted(df["game_id"].unique().to_list())
    cut = games[15]
    # Flip every call from game `cut` on: tendencies up to and including `cut` stay put.
    flipped = df.with_columns(
        pl.when(pl.col("game_id") >= cut).then(1 - pl.col("is_pass")).otherwise(
            pl.col("is_pass")).alias("is_pass"),
        pl.when(pl.col("game_id") >= cut).then(pl.col("epa") + 5).otherwise(pl.col("epa")),
    )  # fmt: skip
    new = pr.team_tendencies(flipped, CFG).sort("game_id", "posteam")
    cols = CFG["tendency_features"]
    early = pl.col("game_id") <= cut
    a, b = base.filter(early), new.filter(early)
    assert np.allclose(a.select(cols).to_numpy(), b.select(cols).to_numpy())
    later = pl.col("game_id") > cut
    assert not np.allclose(base.filter(later).select(cols).to_numpy(),
                           new.filter(later).select(cols).to_numpy())  # fmt: skip
    # The first game of each team has only the priors.
    first = base.filter(pl.col("game_id") == games[0])
    assert first["tend_pass_rate"].to_list() == pytest.approx(
        [CFG["tendency"]["prior_pass_rate"]] * 2)  # fmt: skip


def test_history_features_use_only_earlier_plays_of_the_same_game() -> None:
    df = synthetic_plays(n_games=3, plays=12, seed=2)
    h = pr.add_history(df)
    i = 5  # a play in the middle of the first (game, offense) sequence
    flipped = df.with_columns(
        pl.when(pl.col("row") == i).then(1 - pl.col("is_pass")).otherwise(pl.col("is_pass"))
        .alias("is_pass"))  # fmt: skip
    h2 = pr.add_history(flipped)
    cols = CFG["history_features"]
    a, b = h.select(cols).to_numpy(), h2.select(cols).to_numpy()
    same = np.isclose(a, b) | (np.isnan(a) & np.isnan(b))
    assert same[: i + 1].all()  # the play itself and earlier plays: unchanged
    assert not same[i + 1].all()  # the next play sees it
    # The first play of every sequence has no previous call.
    starts = pr.group_starts(df)
    first = np.where(starts == np.arange(df.height))[0]
    assert h["g_prev_pass"].to_numpy()[first].tolist() == [None] * len(first) or np.isnan(
        h["g_prev_pass"].to_numpy()[first].astype(float)).all()  # fmt: skip
    assert (h["g_n_prev"].to_numpy()[first] == 0).all()


def test_history_batch_reads_only_earlier_rows_of_the_same_sequence() -> None:
    df = synthetic_plays(n_games=2, plays=8, seed=3)
    start = pr.group_starts(df)
    tok = pr.tokens(df)
    drive = df["fixed_drive"].to_numpy()
    rows = np.arange(df.height)
    k = 5
    h = pr.history_batch(rows, start, drive, tok, k)
    assert h.shape == (df.height, k, tok.shape[1] + 3)
    valid = h[..., tok.shape[1]]
    n_prev = rows - start
    assert (valid.sum(axis=1) == np.minimum(n_prev, k)).all()
    # The most recent step is the previous play of the same sequence.
    r = 9  # second play of the second sequence
    assert start[r] == 8
    assert np.allclose(h[r, -1, : tok.shape[1]], tok[8])
    assert valid[r, :-1].sum() == 0


# ---- scoring -----------------------------------------------------------------------


def test_log_loss_p_call_and_calibration() -> None:
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.8, 0.5, 0.1])
    assert pr.p_call(y, p).tolist() == pytest.approx([0.8, 0.2, 0.5, 0.9])
    assert pr.log_loss_rows(y, p) == pytest.approx(-np.log([0.8, 0.2, 0.5, 0.9]))
    rng = np.random.default_rng(0)
    q = rng.uniform(0, 1, 200_000)
    yy = (rng.uniform(0, 1, q.size) < q).astype(int)
    assert pr.calibration(yy, q)["ece"] < 0.01
    assert pr.calibration(yy, np.clip(q + 0.2, 0, 1))["ece"] > 0.1


def test_paired_difference_interval_covers_a_known_gap() -> None:
    rng = np.random.default_rng(1)
    game = np.repeat(np.arange(300), 20)
    a = rng.normal(0.5, 0.2, game.size)
    b = a + 0.01 + rng.normal(0, 0.05, game.size)
    w = rp.boot_weights(300, 300, 2)
    d = pr.paired_diff(b, a, game, w)
    assert d["low"] < 0.01 < d["high"]
    assert d["low"] > 0


# ---- the cost of predictability: known truth ------------------------------------------


def confounded_sample(seed: int = 0, n_teams: int = 30, plays: int = 4000,
                      cost: float = -0.2) -> dict[str, np.ndarray]:  # fmt: skip
    """Plays where the true cost is `cost` EPA per unit of P(call), but good offenses
    are also more predictable (team quality and P(call) are correlated), and situations
    differ in both value and predictability. A raw slope is badly biased; absorbing
    situation and team recovers the truth."""
    rng = np.random.default_rng(seed)
    team = np.repeat(np.arange(n_teams), plays)
    quality = rng.normal(0, 0.15, n_teams)
    sit = rng.integers(0, 12, team.size)
    sit_pred = np.linspace(0.3, 0.8, 12)
    sit_value = np.linspace(-0.2, 0.4, 12)  # the expected call is also the better one here
    p = np.clip(sit_pred[sit] + 0.6 * quality[team] + rng.normal(0, 0.08, team.size),
                0.02, 0.98)  # fmt: skip
    y = sit_value[sit] + quality[team] + cost * p + rng.normal(0, 1.0, team.size)
    game = team * 40 + rng.integers(0, 40, team.size)
    return {"y": y, "p": p, "sit": sit, "team": team, "game": game}


def test_within_team_fixed_effects_recover_a_known_cost_that_the_raw_slope_misses() -> None:
    d = confounded_sample()
    _, game = np.unique(d["game"], return_inverse=True)
    w = rp.boot_weights(int(game.max()) + 1, 60, 3)
    raw = pr.slope_fit(d["y"], d["p"], [np.zeros(d["y"].size, dtype=np.int64)], game, w, CFG)
    within = pr.slope_fit(d["y"], d["p"], [d["sit"], d["team"]], game, w, CFG)
    assert raw["value"] > 0  # confounding flips the sign
    assert within["low"] < -0.2 < within["high"]
    assert abs(within["value"] + 0.2) < 0.05


def test_slope_with_one_control_group_is_not_biased_by_the_means() -> None:
    """Regression test: with a single group of fixed effects the residuals must be
    centered, or a slope through the origin picks up mean(y) / mean(x)."""
    rng = np.random.default_rng(11)
    n = 20_000
    sit = rng.integers(0, 6, n)
    x = 0.5 + 0.05 * sit + rng.normal(0, 0.1, n)
    y = 0.4 - 0.1 * x + 0.03 * sit + rng.normal(0, 0.1, n)  # mean of y far from 0
    game = rng.integers(0, 400, n)
    w = rp.boot_weights(400, 30, 12)
    r = pr.slope_fit(y, x, [sit], game, w, CFG)
    design = np.column_stack([x, (sit[:, None] == np.arange(6)).astype(float)])
    ols = np.linalg.lstsq(design, y, rcond=None)[0][0]
    assert r["value"] == pytest.approx(ols, abs=1e-8)
    assert abs(r["value"] + 0.1) < 0.05  # the uncentered version gave about +0.6


def test_bin_curve_recovers_known_bin_effects_with_fixed_effects() -> None:
    rng = np.random.default_rng(4)
    n = 60_000
    team = rng.integers(0, 20, n)
    sit = rng.integers(0, 8, n)
    pbin = np.clip(rng.integers(0, 4, n) + (team > 9), 0, 3)  # bins depend on team
    truth = np.array([0.1, 0.0, -0.05, -0.15])
    y = truth[pbin] + 0.5 * (team > 9) + 0.2 * sit + rng.normal(0, 0.5, n)
    game = rng.integers(0, 500, n)
    w = rp.boot_weights(500, 40, 5)
    c = pr.bin_curve(y, pbin, 4, [sit, team], game, w, CFG)
    eff = np.array(c["value"]) - y.mean()
    # Bin effects are identified up to a constant: compare differences from bin 0.
    assert np.allclose(eff - eff[0], truth - truth[0], atol=0.03)
    assert all(lo <= v <= hi for v, lo, hi in zip(c["value"], c["low"], c["high"], strict=True))


def test_fe_fit_matches_least_squares_with_dummies() -> None:
    rng = np.random.default_rng(6)
    n = 3000
    a, b = rng.integers(0, 5, n), rng.integers(0, 7, n)
    x = rng.normal(size=n) + 0.3 * a
    y = 0.4 * x + 0.2 * a - 0.1 * b + rng.normal(size=n)
    w = rng.uniform(0.5, 2.0, n)
    fit = pr.fe_fit(y, [a, b], w, x=x[:, None], n_iter=5000, tol=1e-12)
    design = np.column_stack([x, (a[:, None] == np.arange(5)).astype(float),
                              (b[:, None] == np.arange(1, 7)).astype(float)])  # fmt: skip
    coef = np.linalg.lstsq(design * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    assert fit.beta[0] == pytest.approx(coef[0], abs=1e-6)


# ---- shrinkage --------------------------------------------------------------------------


def test_shrinkage_recovers_the_spread_of_true_values() -> None:
    rng = np.random.default_rng(7)
    true = rng.normal(0.0, 0.1, 2000)
    se = rng.uniform(0.05, 0.2, 2000)
    est = true + rng.normal(0, se)
    eb = pr.shrink(est, se)
    assert eb["tau"] == pytest.approx(0.1, rel=0.1)
    assert np.mean((eb["shrunk"] - true) ** 2) < np.mean((est - true) ** 2)
    pure_noise = pr.shrink(rng.normal(0, se), se)
    assert pure_noise["noise_share"] > 0.9


def test_cluster_slope_and_persistence() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=5000)
    y = 0.3 * x + rng.normal(size=5000)
    b, s = pr.cluster_slope(x, y, rng.integers(0, 100, 5000))
    assert abs(b - 0.3) < 4 * s
    t = pl.DataFrame({"posteam": ["A", "A", "B", "B", "C", "C"],
                      "season": [1, 2, 1, 2, 1, 2], "v": [1.0, 1.1, 2.0, 2.2, 3.0, 2.9]})
    assert pr.persistence(t, "v")["corr"] > 0.9


# ---- call models find planted predictability ---------------------------------------------


def test_boosting_with_history_finds_a_planted_alternation_habit() -> None:
    df = pr.add_history(synthetic_plays(n_games=60, plays=30, seed=9, alternate=0.8))
    train = df.filter(pl.col("season") == 2010)
    valid = df.filter(pl.col("season") == 2011)
    test = df.filter(pl.col("season") == 2012)
    params = {**CFG["gbm"], "n_estimators": 200, "min_child_samples": 20}
    y = test["is_pass"].to_numpy()
    lls = {}
    for name, cols in (("situation", ["down"]), ("history", ["down", "g_prev_pass"])):
        m = pr.GBMModel(cols, params, 0, 1).fit(train, valid)
        lls[name] = pr.log_loss_rows(y, m.predict(test)).mean()
    assert lls["history"] < lls["situation"] - 0.1


def test_gru_sequence_model_finds_a_planted_alternation_habit() -> None:
    pytest.importorskip("torch")
    df = synthetic_plays(n_games=60, plays=30, seed=10, alternate=0.8)
    s = df["season"].to_numpy()
    train, valid, test = (np.where(s == y)[0] for y in (2010, 2011, 2012))
    params = {**CFG["seq"], "epochs": 4, "history_len": 6, "hidden": 8, "mlp": 16,
              "batch_size": 128, "lr": 0.01}  # fmt: skip
    y = df["is_pass"].to_numpy()[test]
    lls = {}
    for use in (False, True):
        m = pr.SeqModel(["down"], params, 0, 1, use).fit(df, train, valid)
        lls[use] = pr.log_loss_rows(y, m.predict(df, test)).mean()
    assert lls[True] < lls[False] - 0.1


# ---- real data ---------------------------------------------------------------------------


@pytest.mark.data
def test_real_tendencies_do_not_change_when_later_seasons_are_added() -> None:
    from fourth_and_rl import splits

    if not splits.dev_path().exists():
        pytest.skip("development file not built")
    rcfg = rp.load_config()
    dev = splits.load_dev_plays().filter(pl.col("season").is_between(2012, 2015))
    extra = rp.load_extra([2012, 2013, 2014, 2015], rcfg)
    full = pr.build_features(pr.called_plays(dev, extra, rcfg), CFG)
    early = pr.build_features(
        pr.called_plays(dev.filter(pl.col("season") <= 2013), extra, rcfg), CFG)  # fmt: skip
    cols = CFG["tendency_features"] + CFG["history_features"]
    j = early.select("game_id", "play_id", *cols).join(
        full.select("game_id", "play_id", *cols), on=["game_id", "play_id"], suffix="_f")
    assert j.height == early.height
    for c in cols:
        a, b = j[c].to_numpy().astype(float), j[f"{c}_f"].to_numpy().astype(float)
        assert np.allclose(a, b, equal_nan=True), c
    assert splits.seasons_in(full) == {2012, 2013, 2014, 2015}
