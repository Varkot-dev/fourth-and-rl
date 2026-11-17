"""Tests for the one-step decision engine (M2).

The synthetic tests use a toy win-probability function and toy outcome models whose
answers can be worked out by hand, so they run without any data. Tests marked `data`
need the cached plays and the fitted M1 models.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fourth_and_rl import decision
from fourth_and_rl.decision import DecisionEngine, compress_distribution, summarize_actions
from fourth_and_rl.paths import ROOT
from fourth_and_rl.wp import state_frame

SECONDS = {"go_convert": 27, "go_fail": 5, "score": 5, "field_goal": 4, "punt": 9, "kickoff": 2}


def toy_wp_value(diff: float, yardline: float) -> float:
    """Team with the ball: 55% at midfield when tied, +1 point per point of lead,
    +0.2 points per yard closer to the end zone."""
    return float(np.clip(0.55 + 0.01 * diff - 0.002 * (yardline - 50), 0.01, 0.99))


class ToyWP:
    """A win-probability 'model' that is a formula, and remembers what it was asked."""

    def __init__(self) -> None:
        self.frames: list[pl.DataFrame] = []

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        self.frames.append(df)
        p = 0.55 + 0.01 * df["score_differential"].to_numpy()
        p = p - 0.002 * (df["yardline_100"].to_numpy() - 50)
        return np.clip(p, 0.01, 0.99)

    def asked(self) -> pl.DataFrame:
        return pl.concat(self.frames)


class ToyComponents:
    """Every action has one or two outcomes, so expectations can be done by hand."""

    def __init__(self, p_fg: float = 0.9, p_xp: float = 1.0, fg_range: int = 40) -> None:
        self.p_fg, self.p_xp, self.fg_range = p_fg, p_xp, fg_range

    def go_outcomes(self, yardline_100: int, ydstogo: int) -> dict:
        p = max(0.1, 0.8 - 0.05 * ydstogo)
        first_down, turnover = np.zeros(100), np.zeros(100)
        new = yardline_100 - ydstogo
        p_td = p if new <= 0 else 0.0
        if new > 0:
            first_down[new] = p  # gain exactly the distance
        turnover[100 - yardline_100] = 1 - p  # stopped for no gain
        return {"p_convert": p, "p_touchdown": p_td, "first_down": first_down,
                "turnover": turnover, "p_safety": 0.0}  # fmt: skip

    def p_field_goal(self, yardline_100: int) -> float:
        return self.p_fg if yardline_100 <= self.fg_range else 0.0

    def fg_miss_yardline(self, yardline_100: int) -> int:
        return int(np.clip(92 - yardline_100, 1, 80))

    def punt_outcomes(self, yardline_100: int) -> np.ndarray:
        out = np.zeros(100)
        out[min(140 - yardline_100, 80)] = 1.0  # 40 yards net, touchback to the 20
        return out

    def kickoff_start(self) -> np.ndarray:
        out = np.zeros(100)
        out[70] = 1.0
        return out

    def p_extra_point(self) -> float:
        return self.p_xp


def make_engine(**kw: float) -> tuple[DecisionEngine, ToyWP]:
    toy = ToyWP()
    config = {"clock": {"use_fitted": False, **SECONDS}, "punt_min_yardline_100": 31}
    return DecisionEngine(toy, ToyComponents(**kw), config), toy


def one(engine: DecisionEngine, **state: float) -> dict:
    return engine.evaluate(state_frame(down=4, **state)).to_dicts()[0]


# ------------------------------------------------------------------ hand calculation


def test_engine_reproduces_a_hand_calculation() -> None:
    engine, _ = make_engine()
    row = one(engine, ydstogo=2, yardline_100=40, game_seconds_remaining=3000)

    # Go: 70% first down at the 38 (we keep the ball), 30% they take over 60 from goal.
    wp_convert = toy_wp_value(0, 38)  # 0.574
    wp_fail = 1 - toy_wp_value(0, 60)  # 1 - 0.53 = 0.47
    go = 0.7 * wp_convert + 0.3 * wp_fail  # 0.5428
    # Field goal: 90% we lead by 3 and kick off (they start 70 from goal, down 3),
    # 10% they take over at the spot of the kick, 52 from goal.
    wp_make = 1 - toy_wp_value(-3, 70)  # 1 - 0.48 = 0.52
    wp_miss = 1 - toy_wp_value(0, 52)  # 1 - 0.546 = 0.454
    fg = 0.9 * wp_make + 0.1 * wp_miss  # 0.5134
    # Punt: 40 net from our own 40, they start 80 from goal (touchback).
    punt = 1 - toy_wp_value(0, 80)  # 0.51

    assert row["p_convert"] == pytest.approx(0.7)
    assert row["p_fg"] == pytest.approx(0.9)
    assert row["wp_if_convert"] == pytest.approx(wp_convert)
    assert row["wp_if_fail"] == pytest.approx(wp_fail)
    assert row["go_wp"] == pytest.approx(go) == pytest.approx(0.5428)
    assert row["fg_wp"] == pytest.approx(fg) == pytest.approx(0.5134)
    assert row["punt_wp"] == pytest.approx(punt) == pytest.approx(0.51)
    assert row["best_action"] == "go"
    assert row["best_wp"] == pytest.approx(go)
    assert row["margin"] == pytest.approx(go - fg)
    assert row["go_gain"] == pytest.approx(go - fg)


def test_touchdown_adds_six_plus_the_extra_point_then_a_kickoff() -> None:
    engine, toy = make_engine(p_xp=0.9)
    row = one(engine, ydstogo=1, yardline_100=1, game_seconds_remaining=3000,
              score_differential=-4)  # fmt: skip
    # Convert = touchdown. 90%: up 3 (-4 + 7); 10%: up 2. They start 70 from goal.
    wp_td = 0.9 * (1 - toy_wp_value(-3, 70)) + 0.1 * (1 - toy_wp_value(-2, 70))
    assert row["wp_if_convert"] == pytest.approx(wp_td)
    after_td = toy.asked().filter(
        (pl.col("yardline_100") == 70) & pl.col("score_differential").is_in([-3.0, -2.0])
    )
    # 5 seconds for the scoring play and 2 for the kickoff.
    assert set(after_td["game_seconds_remaining"].to_list()) == {3000.0 - 5 - 2}
    assert set(after_td["down"].to_list()) == {1.0}
    assert set(after_td["ydstogo"].to_list()) == {10.0}


def test_made_field_goal_adds_three_then_a_kickoff() -> None:
    engine, toy = make_engine(p_fg=1.0)
    row = one(engine, ydstogo=5, yardline_100=20, game_seconds_remaining=2500,
              score_differential=-1)  # fmt: skip
    assert row["fg_wp"] == pytest.approx(1 - toy_wp_value(-2, 70))
    assert row["wp_if_fg_make"] == pytest.approx(row["fg_wp"])
    kick = toy.asked().filter(pl.col("score_differential") == -2)
    assert kick["game_seconds_remaining"].to_list() == [2500.0 - 4 - 2]


# ------------------------------------------------------------------ perspective flip


def test_change_of_possession_flips_every_team_specific_input() -> None:
    engine, toy = make_engine()
    one(engine, ydstogo=4, yardline_100=60, game_seconds_remaining=3000, score_differential=5,
        posteam_timeouts_remaining=2, defteam_timeouts_remaining=1, posteam_spread=3.5,
        receive_2h_ko=1, is_home=1)  # fmt: skip
    asked = toy.asked()
    ours = asked.filter(pl.col("posteam_spread") == 3.5)
    theirs = asked.filter(pl.col("posteam_spread") == -3.5)
    assert ours.height > 0 and theirs.height > 0
    assert ours.height + theirs.height == asked.height
    for frame, diff, pos, dfn, home, r2h in ((ours, 5, 2, 1, 1, 1), (theirs, -5, 1, 2, 0, 0)):
        assert set(frame["score_differential"].to_list()) == {float(diff)}
        assert set(frame["posteam_timeouts_remaining"].to_list()) == {float(pos)}
        assert set(frame["defteam_timeouts_remaining"].to_list()) == {float(dfn)}
        assert set(frame["is_home"].to_list()) == {home}
        assert set(frame["receive_2h_ko"].to_list()) == {r2h}
    # The punt: they start 80 from goal, and field position is theirs, not ours.
    assert 80.0 in theirs["yardline_100"].to_list()


def test_constant_win_probability_values_keeping_and_losing_the_ball() -> None:
    """With a constant win probability c for whoever has the ball, keeping the ball is
    worth c and giving it away 1 - c, whatever the score."""

    class Constant:
        def predict(self, df: pl.DataFrame) -> np.ndarray:
            return np.full(df.height, 0.6)

    engine = DecisionEngine(Constant(), ToyComponents(),
                            {"clock": {"use_fitted": False, **SECONDS}})  # fmt: skip
    row = engine.evaluate(
        state_frame(down=4, ydstogo=2, yardline_100=40, game_seconds_remaining=3000)
    ).to_dicts()[0]
    assert row["punt_wp"] == pytest.approx(0.4)
    assert row["fg_wp"] == pytest.approx(0.4)  # make or miss, they have the ball next
    assert row["go_wp"] == pytest.approx(0.7 * 0.6 + 0.3 * 0.4)


def test_mirrored_scores_give_mirrored_values() -> None:
    """Leading by d and trailing by d are mirror images when the win probability is
    antisymmetric in the score, f(-d) = 1 - f(d), and no action can score: every action's
    value at -d must be 1 minus its value at d. This checks that the engine flips the score
    and the side correctly on a change of possession."""

    class Antisymmetric:
        def predict(self, df: pl.DataFrame) -> np.ndarray:
            d = df["score_differential"].to_numpy().astype(float)
            return 1.0 / (1.0 + np.exp(-d / 4.0))  # f(-d) = 1 - f(d)

    # No field goal in range and a long field, so nothing on the play can score.
    engine = DecisionEngine(Antisymmetric(), ToyComponents(fg_range=0),
                            {"clock": {"use_fitted": False, **SECONDS}})  # fmt: skip
    diffs = [1, 3, 7, 10]
    lead = engine.evaluate(state_frame(down=4, ydstogo=[2] * 4, yardline_100=[60] * 4,
                                       game_seconds_remaining=3000, score_differential=diffs))
    trail = engine.evaluate(state_frame(down=4, ydstogo=[2] * 4, yardline_100=[60] * 4,
                                        game_seconds_remaining=3000,
                                        score_differential=[-d for d in diffs]))  # fmt: skip
    for col in ("go_wp", "punt_wp", "wp_if_convert", "wp_if_fail"):
        assert np.allclose(lead[col].to_numpy() + trail[col].to_numpy(), 1.0), col
    assert lead["fg_wp"].is_null().all() and trail["fg_wp"].is_null().all()
    # The sums alone would also hold if the score were not negated on a turnover, so check
    # the turnover branch directly: the opponent trails by d, so we win with 1 - f(-d) = f(d).
    f = 1.0 / (1.0 + np.exp(-np.array(diffs) / 4.0))
    assert np.allclose(lead["wp_if_fail"].to_numpy(), f)
    assert np.allclose(lead["punt_wp"].to_numpy(), f)
    # And a leader is not indifferent: more lead, more win probability.
    assert np.all(np.diff(lead["go_wp"].to_numpy()) > 0)


# ------------------------------------------------------------------------- the clock


def test_certain_field_goal_at_the_last_second() -> None:
    engine, _ = make_engine(p_fg=1.0)
    down3 = one(engine, ydstogo=5, yardline_100=30, game_seconds_remaining=1,
                score_differential=-3)  # fmt: skip
    # The kick ties the game as time expires; anything else loses.
    assert down3["fg_wp"] == pytest.approx(0.5)
    assert down3["go_wp"] == pytest.approx(0.0)
    assert down3["best_action"] == "field_goal"
    down2 = one(engine, ydstogo=5, yardline_100=30, game_seconds_remaining=1,
                score_differential=-2)  # fmt: skip
    assert down2["fg_wp"] == pytest.approx(1.0)
    assert down2["best_action"] == "field_goal"
    assert down2["margin"] == pytest.approx(1.0)


def test_a_play_that_exhausts_the_clock_ends_the_game_with_the_score_as_it_stands() -> None:
    engine, toy = make_engine()
    row = one(engine, ydstogo=5, yardline_100=60, game_seconds_remaining=3,
              score_differential=1)  # fmt: skip
    assert row["go_wp"] == pytest.approx(1.0)
    assert row["punt_wp"] == pytest.approx(1.0)
    assert toy.frames == []  # no next state exists, so the model is never asked


def test_touchdown_as_time_expires_wins_without_a_kickoff() -> None:
    engine, _ = make_engine()
    row = one(engine, ydstogo=1, yardline_100=1, game_seconds_remaining=2,
              score_differential=-4)  # fmt: skip
    assert row["wp_if_convert"] == pytest.approx(1.0)
    assert row["wp_if_fail"] == pytest.approx(0.0)
    assert row["go_wp"] == pytest.approx(row["p_convert"])


@pytest.mark.parametrize(("receive", "expected_spread"), [(1, 2.0), (0, -2.0)])
def test_end_of_first_half_gives_the_second_half_kickoff(
    receive: int, expected_spread: float
) -> None:
    engine, toy = make_engine()
    row = one(engine, ydstogo=5, yardline_100=60, game_seconds_remaining=1803,
              score_differential=3, posteam_spread=2.0, receive_2h_ko=receive,
              posteam_timeouts_remaining=0, defteam_timeouts_remaining=1)  # fmt: skip
    asked = toy.asked()
    # Every action runs out the half (the shortest play takes 4 seconds), so the only
    # next state is the receiver's first and 10 after the second-half kickoff.
    assert set(asked["game_seconds_remaining"].to_list()) == {1800.0 - 2}
    assert set(asked["qtr"].to_list()) == {3.0}
    assert set(asked["yardline_100"].to_list()) == {70.0}
    assert set(asked["posteam_spread"].to_list()) == {expected_spread}
    assert set(asked["posteam_timeouts_remaining"].to_list()) == {3.0}
    assert set(asked["defteam_timeouts_remaining"].to_list()) == {3.0}
    assert set(asked["receive_2h_ko"].to_list()) == {0}
    us = toy_wp_value(3, 70) if receive else 1 - toy_wp_value(-3, 70)
    assert row["punt_wp"] == pytest.approx(us)
    assert row["wp_if_fail"] == pytest.approx(us)


def test_time_runs_off_for_the_play() -> None:
    engine, toy = make_engine()
    one(engine, ydstogo=2, yardline_100=50, game_seconds_remaining=900)
    asked = toy.asked()
    times = {
        (int(y), int(t))
        for y, t in zip(asked["yardline_100"], asked["game_seconds_remaining"], strict=True)
    }
    assert (48, 900 - 27) in times  # converted
    assert (50, 900 - 5) in times  # stopped: they take over at midfield
    assert (80, 900 - 9) in times  # punt


# --------------------------------------------------------------------- monotonicity


def test_going_is_better_as_distance_shrinks() -> None:
    engine, _ = make_engine()
    out = engine.evaluate(
        state_frame(down=4, ydstogo=np.arange(1, 13), yardline_100=55,
                    game_seconds_remaining=2000)  # fmt: skip
    )
    go = out["go_wp"].to_numpy()
    assert np.all(np.diff(go) <= 1e-12)
    assert go[0] > go[-1]
    assert np.all(np.diff(out["go_gain"].to_numpy()) <= 1e-12)


# ---------------------------------------------------------------------- availability


def test_unavailable_actions_are_null() -> None:
    engine, _ = make_engine()
    long_kick = one(engine, ydstogo=3, yardline_100=60, game_seconds_remaining=2000)
    assert long_kick["fg_wp"] is None and long_kick["p_fg"] == 0.0
    assert long_kick["best_action"] in ("go", "punt")
    close = one(engine, ydstogo=3, yardline_100=20, game_seconds_remaining=2000)
    assert close["punt_wp"] is None
    assert close["best_action"] in ("go", "field_goal")


def test_overtime_and_incomplete_states_get_nulls() -> None:
    engine, _ = make_engine()
    states = pl.concat([
        state_frame(down=4, ydstogo=3, yardline_100=40, game_seconds_remaining=300),
        state_frame(down=4, ydstogo=3, yardline_100=40, game_seconds_remaining=300,
                    overtime=True),
    ])  # fmt: skip
    states = pl.concat([states, states.head(1).with_columns(pl.lit(None).alias("ydstogo"))])
    out = engine.evaluate(states)
    assert out["go_wp"].is_null().to_list() == [False, True, True]
    assert out["best_action"].to_list()[1:] == [None, None]


def test_vectorized_equals_row_by_row() -> None:
    engine, _ = make_engine()
    rng = np.random.default_rng(3)
    n = 200
    states = state_frame(
        down=4, ydstogo=rng.integers(1, 12, n), yardline_100=rng.integers(12, 99, n),
        game_seconds_remaining=rng.integers(1, 3600, n),
        score_differential=rng.integers(-10, 11, n), receive_2h_ko=rng.integers(0, 2, n),
        posteam_timeouts_remaining=rng.integers(0, 4, n), posteam_spread=rng.integers(-7, 8, n),
    )  # fmt: skip
    states = states.with_columns(pl.min_horizontal("ydstogo", "yardline_100").alias("ydstogo"))
    together = engine.evaluate(states)
    for i in (0, 17, 101, 199):
        alone = engine.evaluate(states.slice(i, 1))
        for c in ("go_wp", "fg_wp", "punt_wp"):
            a, b = together[c][i], alone[c][0]
            assert (a is None and b is None) or a == pytest.approx(b)
    table = together.select("go_wp", "fg_wp", "punt_wp").fill_null(np.nan).to_numpy()
    assert np.nanmin(table) >= 0.0 and np.nanmax(table) <= 1.0


# --------------------------------------------------------------------------- helpers


def test_compress_distribution_keeps_mass_and_mean() -> None:
    rng = np.random.default_rng(0)
    p = np.zeros(100)
    p[20:90] = rng.random(70) ** 3
    p[65] = 5.0
    p /= p.sum()
    ys, ws = compress_distribution(p, 0.0, 8)
    assert np.array_equal(ys, np.nonzero(p)[0]) and np.allclose(ws, p[ys])
    ys, ws = compress_distribution(p, 0.02, 8)
    assert ys.size < 40
    assert ws.sum() == pytest.approx(1.0)
    assert 65 in ys.tolist() and ws[ys.tolist().index(65)] == pytest.approx(p[65])
    assert float(ys @ ws) == pytest.approx(float(np.arange(100) @ p), abs=0.5)


def test_summarize_actions() -> None:
    table = np.array([[0.5, 0.4, np.nan], [0.3, np.nan, 0.45], [np.nan, 0.2, 0.3],
                      [0.6, np.nan, np.nan]])  # fmt: skip
    best, best_wp, margin, gain = summarize_actions(table)
    assert best.tolist() == [0, 2, -1, 0]
    assert best_wp[0] == pytest.approx(0.5) and best_wp[1] == pytest.approx(0.45)
    assert margin[0] == pytest.approx(0.1) and margin[1] == pytest.approx(0.15)
    assert gain[0] == pytest.approx(0.1) and gain[1] == pytest.approx(-0.15)
    assert np.isnan(best_wp[2]) and np.isnan(margin[3]) and np.isnan(gain[3])


def test_chart_states_drop_impossible_cells() -> None:
    states = decision.chart_states(range(1, 100), range(1, 16), {"game_seconds_remaining": 600})
    assert states.height == 99 * 15 - sum(15 - y for y in range(1, 15))
    assert (states["ydstogo"] <= states["yardline_100"]).all()
    assert set(states["down"].to_list()) == {4.0}


# ------------------------------------------------------------------- the real models


@pytest.mark.data
def test_real_engine_is_fast_sane_and_barely_changed_by_compression(plays: pl.DataFrame) -> None:
    cfg = decision.load_config()
    paths = [ROOT / cfg["models"]["wp_path"], ROOT / cfg["models"]["components_cache"]]
    if not all(p.exists() for p in paths):
        pytest.skip("fitted models not found (run scripts/fit_wp.py and scripts/decision_chart.py)")
    engine = decision.load_engine(config=cfg)
    fourth = plays.filter((pl.col("down") == 4) & (pl.col("season") == 2024)).head(600)
    fast = engine.action_values(fourth)
    exact_cfg = {**cfg["engine"], "compress": {"merge_mass": 0.0, "max_span": 8}}
    exact = DecisionEngine(engine.wp_model, engine.components, exact_cfg).action_values(fourth)
    for k in ("go_wp", "fg_wp", "punt_wp"):
        assert np.nanmax(np.abs(fast[k] - exact[k])) < 0.002
        assert np.nanmin(fast[k]) >= 0.0 and np.nanmax(fast[k]) <= 1.0
    # Reference chart: going on fourth and 1 at midfield beats punting, and nobody
    # should go on fourth and 15 from their own 10.
    chart = engine.evaluate(
        state_frame(down=4, ydstogo=[1, 15], yardline_100=[50, 90], game_seconds_remaining=3300)
    )
    assert chart["best_action"].to_list() == ["go", "punt"]
    assert engine.last_stats["unique_next_states"] <= engine.last_stats["next_states"]
