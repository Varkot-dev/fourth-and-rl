"""Tests for M8 team strength (src/fourth_and_rl/team_strength.py).

The synthetic tests build seasons of games between teams whose true offense and defense
strengths are known, so they run without data: the filter must recover the strengths,
shrink noisy ones, and never look ahead. The engine tests use a toy win-probability
formula and toy outcome models whose answers can be worked out by hand. Tests marked
`data` need the cached plays.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fourth_and_rl import decision, splits
from fourth_and_rl import team_strength as ts
from fourth_and_rl.wp import state_frame

TEAMS = [f"T{i:02d}" for i in range(16)]
BASE_P = 0.55


def synthetic_season_data(
    n_seasons: int = 3,
    weeks: int = 14,
    plays_per_game: int = 40,
    sd: float = 0.05,
    rho: float = 0.6,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Round-robin-ish schedule with known strengths. Returns (sample, schedule, truth),
    truth[season] = (offense strengths, defense strengths allowed)."""
    rng = np.random.default_rng(seed)
    nt = len(TEAMS)
    off = rng.normal(0, sd, nt)
    dfn = rng.normal(0, sd, nt)
    rows, games, truth = [], [], {}
    for s in range(2001, 2001 + n_seasons):
        if s > 2001:
            off = rho * off + np.sqrt(1 - rho**2) * rng.normal(0, sd, nt)
            dfn = rho * dfn + np.sqrt(1 - rho**2) * rng.normal(0, sd, nt)
        truth[s] = (off.copy(), dfn.copy())
        for w in range(1, weeks + 1):
            order = rng.permutation(nt)
            for a, b in zip(order[::2], order[1::2], strict=True):
                gid = f"{s}_{w:02d}_{TEAMS[a]}_{TEAMS[b]}"
                games.append({"game_id": gid, "season": s, "week": w,
                              "home_team": TEAMS[a], "away_team": TEAMS[b]})  # fmt: skip
                for o, d in ((a, b), (b, a)):
                    p = np.clip(BASE_P + off[o] + dfn[d], 0.02, 0.98)
                    conv = rng.random(plays_per_game) < p
                    for c in conv:
                        rows.append({"game_id": gid, "season": s, "week": w,
                                     "posteam": TEAMS[o], "defteam": TEAMS[d],
                                     "converted": int(c)})  # fmt: skip
    sample = pl.DataFrame(rows).with_columns(
        pl.lit(BASE_P).alias("base_p"),
        (pl.col("converted") - BASE_P).alias("resid"),
        pl.lit(BASE_P * (1 - BASE_P)).alias("noise"),
    )
    return sample, pl.DataFrame(games), truth


PARAMS = ts.FilterParams(tau_off=0.05, tau_def=0.05, rho=0.6, q_frac=0.0)


def run(sample: pl.DataFrame, games: pl.DataFrame, params: ts.FilterParams = PARAMS
        ) -> tuple[ts.FilterResult, ts.Schedule]:  # fmt: skip
    sched = ts.Schedule.from_frame(games, TEAMS)
    obs = ts.observations(sample, sched)
    return ts.run_filter(obs, sched, params), sched


# ------------------------------------------------------------------ recovery and shrinkage


def test_filter_recovers_known_strengths() -> None:
    sample, games, truth = synthetic_season_data(plays_per_game=60, seed=1)
    res, _ = run(sample, games)
    nt = len(TEAMS)
    for s, (off, dfn) in truth.items():
        x, sd = res.season_end[s]
        assert np.corrcoef(x[:nt], off)[0, 1] > 0.75, s
        assert np.corrcoef(x[nt:], dfn)[0, 1] > 0.75, s
        assert np.all(sd > 0) and np.all(sd < PARAMS.tau_off)


def test_filter_shrinks_noisy_estimates_toward_average() -> None:
    """With few plays per game the raw per-team mean residual is mostly noise; the filter's
    ratings are pulled in (smaller spread than the raw means) and are closer to the truth."""
    sample, games, truth = synthetic_season_data(n_seasons=1, weeks=6, plays_per_game=6,
                                                 sd=0.03, seed=2)  # fmt: skip
    res, sched = run(sample, games, ts.FilterParams(0.03, 0.03, 0.6, 0.0))
    raw = sample.group_by("posteam").agg(pl.col("resid").mean()).sort("posteam")
    raw_off = raw["resid"].to_numpy()
    x, _ = res.season_end[2001]
    off_true = truth[2001][0]
    rated = x[: len(TEAMS)]
    assert rated.std() < 0.6 * raw_off.std()
    assert np.mean((rated - off_true) ** 2) < np.mean((raw_off - off_true) ** 2)


def test_tau_zero_means_no_team_effect() -> None:
    sample, games, _ = synthetic_season_data(n_seasons=1, weeks=4, seed=3)
    res, _ = run(sample, games, ts.FilterParams(0.0, 0.0, 0.5, 0.0))
    assert np.allclose(res.pre, 0.0)
    assert np.allclose(res.obs_mean, 0.0)


def test_season_break_shrinks_by_rho_and_restores_variance() -> None:
    sample, games, _ = synthetic_season_data(n_seasons=2, weeks=6, seed=4)
    res, sched = run(sample, games, ts.FilterParams(0.05, 0.05, 0.5, 0.0))
    x_end, sd_end = res.season_end[2001]
    first = [i for i, (s, w) in enumerate(zip(sched.season, sched.week, strict=True))
             if s == 2002 and w == 1]  # fmt: skip
    g = first[0]
    t = sched.home[g]
    off_mean, off_sd = res.pre[g, 0, 0], res.pre[g, 0, 1]
    assert off_mean == pytest.approx(0.5 * x_end[t])
    assert off_sd**2 == pytest.approx(0.25 * sd_end[t] ** 2 + 0.75 * 0.05**2)


# ------------------------------------------------------------------ no look-ahead


def _scramble_from(sample: pl.DataFrame, season: int, week: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    later = ((sample["season"] > season) | ((sample["season"] == season)
             & (sample["week"] >= week))).to_numpy()  # fmt: skip
    conv = sample["converted"].to_numpy().copy()
    conv[later] = rng.integers(0, 2, later.sum())
    return sample.with_columns(pl.Series("converted", conv)).with_columns(
        (pl.col("converted") - BASE_P).alias("resid")
    )


@pytest.mark.parametrize("cut", [(2001, 1), (2001, 7), (2002, 1), (2002, 9), (2003, 14)])
def test_ratings_never_use_their_own_game_or_a_later_one(cut: tuple[int, int]) -> None:
    """Scramble every outcome at or after a cutoff week: every pre-game rating of a game
    at or before that week is unchanged (so it used neither its own game nor later ones),
    and later ratings do change."""
    sample, games, _ = synthetic_season_data(seed=5)
    a, sched = run(sample, games)
    b, _ = run(_scramble_from(sample, *cut, seed=9), games)
    s, w = cut
    upto = (sched.season < s) | ((sched.season == s) & (sched.week <= w))
    assert np.array_equal(a.pre[upto], b.pre[upto])
    after = ~upto
    if after.any():
        assert not np.allclose(a.pre[after], b.pre[after])


def test_a_games_own_outcome_never_reaches_its_rating() -> None:
    sample, games, _ = synthetic_season_data(n_seasons=1, seed=6)
    a, sched = run(sample, games)
    gid = sched.game_ids[40]
    flipped = sample.with_columns(
        pl.when(pl.col("game_id") == gid).then(1 - pl.col("converted"))
        .otherwise(pl.col("converted")).alias("converted")
    ).with_columns((pl.col("converted") - BASE_P).alias("resid"))  # fmt: skip
    b, _ = run(flipped, games)
    assert np.array_equal(a.pre[40], b.pre[40])
    # predictions for the game's own observations are unchanged as well
    obs = ts.observations(sample, sched)
    own = obs.game == 40
    assert np.array_equal(a.obs_mean[own], b.obs_mean[own])


def test_ratings_frame_joins_offense_and_defense_by_side() -> None:
    sample, games, _ = synthetic_season_data(n_seasons=1, weeks=3, seed=7)
    res, sched = run(sample, games)
    frame = ts.ratings_frame({"conv": res, "drive": res}, sched)
    rows = ts.attach_ratings(sample.unique(["game_id", "posteam"]), frame)
    assert rows["us_conv_off"].null_count() == 0
    r = rows.row(0, named=True)
    g = sched.game_ids.index(r["game_id"])
    side = 0 if sched.teams[sched.home[g]] == r["posteam"] else 1
    assert r["us_conv_off"] == pytest.approx(res.pre[g, side, 0])
    assert r["them_conv_def"] == pytest.approx(res.pre[g, 1 - side, 2])


def test_base_curves_use_only_earlier_seasons() -> None:
    assert ts.base_fit_seasons(2015, 2010, 2023) == list(range(2010, 2015))
    assert ts.base_fit_seasons(2011, 2010, 2023) == [2010]
    assert ts.base_fit_seasons(2024, 2010, 2023) == list(range(2010, 2024))
    assert ts.base_fit_seasons(2025, 2010, 2023) == list(range(2010, 2024))
    assert ts.base_fit_seasons(2010, 2010, 2023) == []


def test_tuning_refuses_test_seasons() -> None:
    sample, games, _ = synthetic_season_data(n_seasons=1, weeks=2, seed=8)
    test_season = splits.test_seasons()[0]
    bad = sample.with_columns(pl.lit(test_season).alias("season"))
    sched = ts.Schedule.from_frame(games.with_columns(pl.lit(test_season).alias("season")), TEAMS)
    with pytest.raises(splits.LeakageError):
        ts.tune_filter("conv", bad, sched, {"tau_off": [0.0], "tau_def": [0.0], "rho": [0.5],
                                           "q_frac": [0.0]}, [[test_season, test_season]],
                       test_season)  # fmt: skip


def test_tuning_prefers_a_team_effect_when_there_is_one() -> None:
    sample, games, _ = synthetic_season_data(seed=10)
    sched = ts.Schedule.from_frame(games, TEAMS)
    grid = {"tau_off": [0.0, 0.05], "tau_def": [0.0, 0.05], "rho": [0.6], "q_frac": [0.0]}
    tuned = ts.tune_filter("conv", sample, sched, grid, [[2002, 2003]], 2001)
    best, _ = ts.choose_params(tuned, [2002, 2003])
    assert best.tau_off > 0 and best.tau_def > 0
    none = ts.no_effect_index(tuned)
    assert tuned["season_loss"][none].sum() > tuned["season_loss"].min(axis=0).sum()


# ------------------------------------------------------------------ effect models


def _conv_frame(n: int, coef: float, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.02, n)
    spread = rng.normal(0, 5, n)
    base = np.full(n, 0.2)
    z = base + 0.3 * spread / 10 + coef * 100 * r
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    zeros = np.zeros(n)
    return pl.DataFrame({
        "season": np.full(n, 2015), "posteam_spread": spread, "base_logit": base, "converted": y,
        "us_conv_off": r, "them_conv_def": zeros, "us_drive_off": zeros, "them_drive_def": zeros,
    })  # fmt: skip


def test_conversion_model_recovers_a_rating_coefficient() -> None:
    df = _conv_frame(40_000, 0.04, seed=11)
    m = ts.ConversionModel(["off_conv", "def_conv"]).fit(df)
    assert m.coef_["off_conv"] == pytest.approx(0.04, abs=0.01)
    assert m.coef_["spread_per_10"] == pytest.approx(0.3, abs=0.05)
    assert abs(m.coef_["intercept"]) < 0.05
    sp = ts.ConversionModel([]).fit(df)
    ll_t = ts.conversion_log_loss(df["converted"].to_numpy(), m.predict(df)).mean()
    ll_s = ts.conversion_log_loss(df["converted"].to_numpy(), sp.predict(df)).mean()
    assert ll_t < ll_s


def test_game_bootstrap_interval_contains_the_mean() -> None:
    rng = np.random.default_rng(12)
    games = np.repeat(np.arange(200), 10).astype(str)
    diff = rng.normal(0.01, 0.05, games.size)
    out = ts.game_bootstrap_mean_diff(games, diff, 500, seed=1)
    assert out["lo"] < out["mean"] < out["hi"]
    assert out["n_games"] == 200


# ------------------------------------------------------------------ the engine extension


class ToyWP:
    """Team with the ball: 55% at midfield when tied, +1 point per point of lead, +0.2
    points per yard closer to the end zone. Ignores the clock."""

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        p = 0.55 + 0.01 * df["score_differential"].to_numpy()
        return np.clip(p - 0.002 * (df["yardline_100"].to_numpy() - 50), 0.01, 0.99)


class ToyComponents:
    def go_outcomes(self, yardline_100: int, ydstogo: int) -> dict:
        p = max(0.1, 0.8 - 0.05 * ydstogo)
        first_down, turnover = np.zeros(100), np.zeros(100)
        new = yardline_100 - ydstogo
        if new > 0:
            first_down[new] = p
        turnover[100 - yardline_100] = 1 - p
        return {"p_convert": p, "p_touchdown": p if new <= 0 else 0.0,
                "first_down": first_down, "turnover": turnover, "p_safety": 0.0}  # fmt: skip

    def p_field_goal(self, yardline_100: int) -> float:
        return 0.9 if yardline_100 <= 40 else 0.0

    def fg_miss_yardline(self, yardline_100: int) -> int:
        return int(np.clip(92 - yardline_100, 1, 80))

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


CLOCK = {"use_fitted": False, "go_convert": 27, "go_fail": 5, "score": 5, "field_goal": 4,
         "punt": 9, "kickoff": 2}  # fmt: skip
CONSTS = {"seconds": {"td": 170, "fg": 160, "none": 120}, "next_start_intercept": 70.0,
          "next_start_slope": 0.0}  # fmt: skip


def toy_engine() -> ts.TeamAwareEngine:
    return ts.TeamAwareEngine(ToyWP(), ToyComponents(), CONSTS,
                              {"clock": CLOCK, "punt_min_yardline_100": 31})  # fmt: skip


def test_constants_mirror_the_decision_engine() -> None:
    assert (ts.OWNER_US, ts.OWNER_OPP) == (decision._US, decision._OPP)
    assert ts.N_SLOTS == decision._N_SLOTS
    assert (ts.SLOT_CONVERT, ts.SLOT_FAIL, ts.SLOT_FG_MAKE, ts.SLOT_FG_MISS, ts.SLOT_PUNT) == (
        decision._CONVERT,
        decision._FAIL,
        decision._FG_MAKE,
        decision._FG_MISS,
        decision._PUNT,
    )


def test_extension_leaves_the_engine_values_unchanged() -> None:
    states = decision.chart_states([30, 40, 60, 80], [1, 2, 5], {"game_seconds_remaining": 3000})
    plain = decision.DecisionEngine(ToyWP(), ToyComponents(),
                                    {"clock": CLOCK, "punt_min_yardline_100": 31})  # fmt: skip
    base = plain.action_values(states)
    vals, sens = toy_engine().sensitivities(states)
    for k in base:
        assert np.allclose(np.nan_to_num(base[k]), np.nan_to_num(vals[k]))
    p = vals["p_convert"]
    out = ts.team_aware_table(vals, sens, p, np.zeros(p.size), np.zeros((p.size, 2, 100, 2)))
    assert np.allclose(np.nan_to_num(out["spread_table"]), np.nan_to_num(out["team_table"]))
    go = np.nan_to_num(out["spread_table"][:, 0])
    assert np.allclose(go, np.nan_to_num(base["go_wp"]))


def test_opponent_scoring_more_lowers_the_punt_by_hand() -> None:
    """Punt from our own 40: they start at their own 20 (yardline 80). If their drive's
    touchdown chance rises by 0.10, the punt loses 0.10 times the swing between their
    touchdown and their scoreless drive, worked out with the toy formula:
      after their touchdown we trail by 7 at the kickoff spot 70: our WP 0.55 - 0.07 - 0.04
      = 0.44; after a scoreless drive we have the ball at 70 tied: 0.51. Swing 0.07, so the
      punt drops by 0.007."""
    states = state_frame(down=4, ydstogo=5, yardline_100=60, game_seconds_remaining=3000)
    vals, sens = toy_engine().sensitivities(states)
    dp = np.zeros((1, 2, 100, 2))
    dp[0, ts.OWNER_OPP, :, 0] = 0.10
    out = ts.team_aware_table(vals, sens, vals["p_convert"], np.zeros(1), dp)
    assert out["team_table"][0, 2] - out["spread_table"][0, 2] == pytest.approx(-0.007)


def test_our_offense_scoring_more_raises_the_value_of_converting() -> None:
    """Fourth and 2 at the opponent's 40: converting (70%) gives us first and 10 at the 38.
    With our drive's touchdown chance up 0.10 there, the go value rises by
    0.7 * 0.10 * 0.07 = 0.0049 (same swing as above, from our side)."""
    states = state_frame(down=4, ydstogo=2, yardline_100=40, game_seconds_remaining=3000)
    vals, sens = toy_engine().sensitivities(states)
    dp = np.zeros((1, 2, 100, 2))
    dp[0, ts.OWNER_US, :, 0] = 0.10
    out = ts.team_aware_table(vals, sens, vals["p_convert"], np.zeros(1), dp)
    assert out["team_table"][0, 0] - out["spread_table"][0, 0] == pytest.approx(0.0049)


def test_conversion_shift_reweights_go_only() -> None:
    states = state_frame(down=4, ydstogo=2, yardline_100=40, game_seconds_remaining=3000)
    vals, sens = toy_engine().sensitivities(states)
    p = vals["p_convert"]
    out = ts.team_aware_table(vals, sens, p, np.array([0.4]), None)
    p_new = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + 0.4)))
    expect = p_new * vals["wp_if_convert"] + (1 - p_new) * vals["wp_if_fail"]
    assert out["team_table"][0, 0] == pytest.approx(expect[0])
    assert out["team_table"][0, 1:] == pytest.approx(out["spread_table"][0, 1:])


def test_no_drive_adjustment_near_the_end_of_a_half() -> None:
    states = state_frame(down=4, ydstogo=5, yardline_100=60, game_seconds_remaining=200)
    _, sens = toy_engine().sensitivities(states)
    assert np.all(sens == 0)


# ------------------------------------------------------------------ data


@pytest.mark.data
def test_real_samples_and_bases_respect_the_season_order() -> None:
    if not splits.dev_path().exists():
        pytest.skip("development file not built")
    dev = splits.load_dev_plays().filter(pl.col("season") <= 2012)
    bases = ts.fit_bases(dev, [2011, 2012])
    assert bases.fitted_on[2011] == [2010]
    assert bases.fitted_on[2012] == [2010, 2011]
    assert bases.conversion[2012].reference_season_ == 2011
    conv = ts.conversion_sample(dev.filter(pl.col("season") >= 2011))
    drives = ts.drive_sample(dev.filter(pl.col("season") >= 2011))
    conv, drives = ts.add_residuals(conv, drives, bases)
    assert abs(conv["resid"].mean()) < 0.03
    assert abs(drives["resid"].mean()) < 0.3
    assert set(drives["outcome"].unique().to_list()) == {"none", "fg", "td"}


def test_league_level_absorbs_a_common_offset() -> None:
    """If every residual is shifted by the same amount (the league curve is a season
    behind), the league term takes it and team ratings stay centred on zero."""
    sample, games, _ = synthetic_season_data(n_seasons=1, weeks=10, seed=13)
    shifted = sample.with_columns((pl.col("resid") + 0.06).alias("resid"))
    with_league, _ = run(shifted, games, ts.FilterParams(0.05, 0.05, 0.6, 0.0, league_sd=0.05))
    without, _ = run(shifted, games, ts.FilterParams(0.05, 0.05, 0.6, 0.0, league_sd=0.0))
    nt = len(TEAMS)
    x_l = with_league.season_end[2001][0]
    x_n = without.season_end[2001][0]
    assert abs(x_l[:nt].mean() + x_l[nt:].mean()) < 0.5 * abs(x_n[:nt].mean() + x_n[nt:].mean())


def _drive_frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = rng.integers(20, 95, n)
    off = rng.normal(0, 0.4, n)
    z_td = -0.5 - 0.02 * (start - 50) + 0.8 * off
    z_fg = -1.2 - 0.01 * (start - 50) + 0.3 * off
    den = 1 + np.exp(z_td) + np.exp(z_fg)
    u = rng.random(n)
    outcome = np.where(
        u < np.exp(z_td) / den,
        "td",
        np.where(u < (np.exp(z_td) + np.exp(z_fg)) / den, "fg", "none"),
    )
    zeros = np.zeros(n)
    return pl.DataFrame({
        "season": np.full(n, 2015), "start": start, "posteam_spread": rng.normal(0, 5, n),
        "outcome": outcome, "us_drive_off": off, "them_drive_def": zeros,
        "us_conv_off": zeros, "them_conv_def": zeros, "game_id": np.arange(n).astype(str),
    })  # fmt: skip


def test_drive_model_recovers_rating_effect_and_warm_start_matches() -> None:
    df = _drive_frame(30_000, seed=14)
    m = ts.DriveModel(["off_drive"]).fit(df)
    coef = m.rating_coef()
    assert coef["td"]["off_drive"] == pytest.approx(0.8, abs=0.1)
    assert coef["fg"]["off_drive"] == pytest.approx(0.3, abs=0.15)
    w = ts.game_weights(df["game_id"].to_numpy(), np.random.default_rng(0))
    warm = ts.DriveModel(["off_drive"]).fit(df, w, init=m)
    cold = ts.DriveModel(["off_drive"]).fit(df, w)
    assert np.allclose(warm.predict(df), cold.predict(df), atol=2e-3)
