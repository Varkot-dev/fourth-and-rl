"""Tests for the M7 coach evaluation and the coach cost ledger.

The synthetic tests run without data: the fold plan, the grading arithmetic, penalty
pricing on a toy win-probability formula, deliberate-delay detection on hand-made rows, the
interval and shrinkage arithmetic, and the variance-component fit on simulated counts.
Tests marked `data` need the cached plays and the outputs of the M7 scripts.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fourth_and_rl import coaches, splits
from fourth_and_rl.paths import DATA, RESULTS, ROOT

DEV = tuple(range(2010, 2024))
TEST = (2024, 2025)
CFG = coaches.load_config()


# ---------------------------------------------------------------- the fold plan


def test_the_fold_plan_never_grades_a_season_with_a_model_that_saw_it() -> None:
    plan = coaches.fold_plan(DEV, TEST, range(2011, 2026), 4)
    graded = [s for f in plan for s in f.grade]
    assert sorted(graded) == list(range(2011, 2026))  # every season once
    for f in plan:
        assert not set(f.grade) & set(f.fit)
        assert not set(f.fit) & set(TEST)
    assert [f.grade for f in plan] == [(2011, 2012, 2013), (2014, 2015, 2016, 2017),
                                       (2018, 2019, 2020, 2021), (2022, 2023), (2024, 2025)]
    assert plan[-1].headline and plan[-1].fit == DEV
    assert plan[0].fit == tuple(range(2014, 2024))
    coaches.check_fold_plan(plan, TEST, (2026,))


def test_the_configured_plan_passes_the_guard() -> None:
    lo, hi = CFG["grade_seasons"]
    plan = coaches.fold_plan(splits.dev_seasons(), splits.test_seasons(), range(lo, hi + 1),
                             CFG["folds"]["fold_size"])  # fmt: skip
    coaches.check_fold_plan(plan, splits.test_seasons(), splits.holdout_seasons())
    assert all(2026 not in f.grade + f.fit for f in plan)


@pytest.mark.parametrize("bad", [
    [coaches.Fold("x", (2015,), (2014, 2015, 2016))],          # grades a fitted season
    [coaches.Fold("x", (2015,), (2014, 2024))],                # fitted on a test season
    [coaches.Fold("x", (2026,), (2014,))],                     # grades the holdout
    [coaches.Fold("a", (2015,), (2014,)), coaches.Fold("b", (2015,), (2016,))],  # twice
])  # fmt: skip
def test_bad_fold_plans_are_refused(bad: list[coaches.Fold]) -> None:
    with pytest.raises(splits.LeakageError):
        coaches.check_fold_plan(bad, TEST, (2026,))


def test_a_test_season_never_enters_a_fold_fit_even_if_listed_as_development() -> None:
    plan = coaches.fold_plan((2014, 2015, 2016, 2017, 2024), TEST, range(2014, 2026), 2)
    with pytest.raises(splits.LeakageError):
        coaches.check_fold_plan(plan, TEST, (2026,))


# ---------------------------------------------------------------- grading arithmetic


def test_spread_adjustment_reweights_the_two_branches_only() -> None:
    go = coaches.spread_adjusted_go(np.array([0.6, 0.5, 1.0]), np.array([0.7, 0.8, np.nan]),
                                    np.array([0.4, 0.2, 0.3]), np.array([0.58, 0.47, 0.9]))
    assert go[0] == pytest.approx(0.6 * 0.7 + 0.4 * 0.4)
    assert go[1] == pytest.approx(0.5 * 0.8 + 0.5 * 0.2)
    assert go[2] == pytest.approx(0.9)  # undefined branch: the engine's own value is kept


def test_grade_counts_win_probability_given_up_against_the_best_action() -> None:
    table = np.array([
        [0.60, 0.55, 0.50],     # went: best, lost 0
        [0.60, 0.55, 0.50],     # punted: lost 0.10
        [0.40, np.nan, 0.45],   # kicked out of range: not gradable
        [0.30, 0.20, np.nan],   # punt not modelled there, went: lost 0
        [np.nan, 0.5, 0.5],     # no go value: nothing graded
    ])  # fmt: skip
    g = coaches.grade(table, ["go", "punt", "field_goal", "go", "punt"])
    assert g["wp_lost"][0] == 0.0
    assert g["wp_lost"][1] == pytest.approx(0.10)
    assert np.isnan(g["wp_lost"][2]) and np.isnan(g["wp_lost"][4])
    assert g["wp_lost"][3] == 0.0
    assert list(g["best"]) == [0, 0, 2, 0, -1]
    assert g["margin"][1] == pytest.approx(0.05)


def test_conversion_at_the_play_spread_uses_the_engine_capping() -> None:
    class Comps:
        def p_convert(self, yl, d, spread):  # noqa: ANN001, ANN201
            return 0.5 + 0.01 * spread - 0.01 * d + 0.0 * yl

    states = pl.DataFrame({"yardline_100": [3.0, 60.0], "ydstogo": [10.0, 2.0],
                           "posteam_spread": [3.0, -7.0]})  # fmt: skip
    p = coaches.p_convert_at_spread(Comps(), states)
    assert p[0] == pytest.approx(0.5 + 0.03 - 0.03)  # distance capped at the yard line
    assert p[1] == pytest.approx(0.5 - 0.07 - 0.02)


# ---------------------------------------------------------------- penalties


def pen_rows(**over: list) -> pl.DataFrame:  # noqa: ANN003
    base = {
        "game_id": ["g"] * 2, "play_id": [1.0, 2.0], "order_sequence": [1.0, 2.0],
        "posteam": ["A", "A"], "play_type": ["no_play", "run"], "down": [1.0, 1.0],
        "ydstogo": [10.0, 15.0], "yardline_100": [60.0, 65.0],
        "penalty_type": ["False Start", None], "penalty_team": ["A", None],
        "penalty_yards": [5.0, None], "desc": ["(Shotgun) PENALTY on A, False Start", "run"],
    }
    base.update(over)
    return pl.DataFrame(base)


def label(df: pl.DataFrame) -> pl.DataFrame:
    return coaches.coach_fouls(df, CFG["penalties"])


def test_offense_false_start_before_the_snap_is_a_coach_foul() -> None:
    out = label(pen_rows())
    assert out["foul_type"].to_list() == ["false_start", None]
    assert out["coach_foul"].to_list() == [True, False]


@pytest.mark.parametrize("desc,down,next_type,deliberate", [
    ("(Punt formation) PENALTY on A, Delay of Game", 4.0, "punt", True),
    ("(Punt formation) PENALTY on A, Delay of Game", 4.0, "run", True),     # fake after it
    ("(Field Goal formation) PENALTY on A, Delay of Game", 4.0, "field_goal", True),
    ("(Shotgun) PENALTY on A, Delay of Game", 4.0, "punt", True),           # backed up, punted
    ("(Shotgun) PENALTY on A, Delay of Game", 4.0, "pass", False),          # went for it
    ("(Shotgun) PENALTY on A, Delay of Game", 3.0, "pass", False),          # third down
    ("(Shotgun) PENALTY on A, Delay of Game", 3.0, "punt", False),          # a punt later on
])  # fmt: skip
def test_deliberate_delay_detection(desc: str, down: float, next_type: str,
                                    deliberate: bool) -> None:  # fmt: skip
    rows = pen_rows(penalty_type=["Delay of Game", None], desc=[desc, "x"], down=[down, down],
                    play_type=["no_play", next_type])  # fmt: skip
    out = label(rows)
    assert out["foul_type"][0] == "delay_of_game"
    assert out["deliberate"][0] is deliberate
    assert out["coach_foul"][0] is (not deliberate)


def test_a_false_start_in_punt_formation_still_counts() -> None:
    rows = pen_rows(desc=["(Punt formation) PENALTY on A, False Start", "x"],
                    down=[4.0, 4.0], play_type=["no_play", "punt"])  # fmt: skip
    assert label(rows)["coach_foul"][0] is True


@pytest.mark.parametrize("ptype,team,play_type,yards,down", [
    ("Encroachment", "B", "no_play", 5.0, 1.0),
    ("Defensive Offside", "B", "no_play", 5.0, 1.0),
    ("Neutral Zone Infraction", "B", "no_play", 5.0, 1.0),
    ("False Start", "B", "no_play", 5.0, 1.0),        # the defense's team code
    ("Delay of Game", "A", "pass", 5.0, 1.0),         # enforced after a play that ran
    ("False Start", "A", "no_play", 0.0, 1.0),        # no yards (at the goal line)
    ("False Start", "A", "no_play", 5.0, None),       # extra point try: no down
])  # fmt: skip
def test_fouls_outside_the_owners_rule_are_not_labelled(
    ptype: str, team: str, play_type: str, yards: float, down: float | None
) -> None:
    rows = pen_rows(penalty_type=[ptype, None], penalty_team=[team, None],
                    play_type=[play_type, "run"], penalty_yards=[yards, None],
                    down=[down, 1.0])  # fmt: skip
    out = label(rows)
    assert out["foul_type"][0] is None and out["coach_foul"][0] is False


class ToyWP:
    """WP = 0.5 + 0.01 * lead - 0.002 * (yards from the end zone - 50) - 0.004 * distance."""

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return (0.5 + 0.01 * df["score_differential"].to_numpy()
                - 0.002 * (df["yardline_100"].to_numpy() - 50)
                - 0.004 * df["ydstogo"].to_numpy())  # fmt: skip


def state_rows(yl: list[float], togo: list[float], yds: list[float]) -> pl.DataFrame:
    n = len(yl)
    return pl.DataFrame({
        "down": [1.0] * n, "ydstogo": togo, "yardline_100": yl,
        "game_seconds_remaining": [1800.0] * n, "half_seconds_remaining": [1800.0] * n,
        "qtr": [3.0] * n, "score_differential": [3.0] * n,
        "posteam_timeouts_remaining": [3.0] * n, "defteam_timeouts_remaining": [3.0] * n,
        "posteam_spread": [0.0] * n, "receive_2h_ko": [0.0] * n, "is_home": [1.0] * n,
        "penalty_yards": yds,
    })  # fmt: skip


def test_penalty_pricing_on_a_toy_engine() -> None:
    rows = state_rows([60.0, 4.0, 98.0], [10.0, 4.0, 10.0], [5.0, 2.0, 1.0])
    with_, without = coaches.penalty_states(rows)
    # Ordinary: 5 yards back, distance 10 -> 15.
    assert with_["yardline_100"][0] == 65 and with_["ydstogo"][0] == 15
    # Goal to go at the 4, half the distance (2 yards): first and goal at the 6.
    assert with_["yardline_100"][1] == 6 and with_["ydstogo"][1] == 6
    # At the own 2, one yard (half the distance): the ball stays on the field.
    assert with_["yardline_100"][2] == 99 and with_["ydstogo"][2] == 11
    cost = coaches.price_states(ToyWP(), with_, without)
    assert cost[0] == pytest.approx(0.002 * 5 + 0.004 * 5)
    assert cost[1] == pytest.approx(0.002 * 2 + 0.004 * 2)
    assert cost[2] == pytest.approx(0.002 * 1 + 0.004 * 1)
    assert without["yardline_100"].to_list() == [60, 4, 98]  # unchanged


def test_next_snap_skips_kickoffs_and_extra_points() -> None:
    df = pl.DataFrame({
        "game_id": ["g"] * 4, "play_id": [1.0, 2.0, 3.0, 4.0],
        "order_sequence": [1.0, 2.0, 3.0, 4.0], "posteam": ["A", "A", "B", "A"],
        "play_type": ["no_play", "extra_point", "kickoff", "punt"],
        "down": [4.0, None, None, 4.0],
    })  # fmt: skip
    out = coaches.next_snap(df)
    assert out["next_snap_type"][0] == "punt" and out["next_snap_posteam"][0] == "A"


# ---------------------------------------------------------------- coaches and tables


def test_coach_lookup_handles_a_mid_season_change_and_historical_codes() -> None:
    sched = pl.DataFrame({
        "game_id": ["g1", "g2"], "home_team": ["SD", "X"], "away_team": ["Y", "SD"],
        "home_coach": ["Old", "Hx"], "away_coach": ["Ay", "Interim"],
    })  # fmt: skip
    lookup = coaches.coach_lookup(sched)
    plays = pl.DataFrame({"game_id": ["g1", "g2", "g2"], "posteam": ["LAC", "LAC", "X"],
                          "is_home": [1, 0, 1]})  # fmt: skip
    out = coaches.attach_coaches(plays, lookup)
    assert out["coach"].to_list() == ["Old", "Interim", "Hx"]
    with pytest.raises(ValueError):
        coaches.attach_coaches(plays.with_columns(pl.lit("g9").alias("game_id")), lookup)


def test_group_bootstrap_sums_per_game_and_counts_repeated_draws() -> None:
    group = np.array(["a", "a", "a", "b", "b"])
    game = np.array(["g1", "g1", "g2", "g3", "g4"])
    cost = np.array([0.01, 0.02, 0.05, 0.0, 0.0])
    out = coaches.group_bootstrap(group, game, {"cost": cost, "n": np.ones(5)}, 500, 1)
    assert out["a"]["n_games"] == 2 and out["a"]["point"]["cost"] == pytest.approx(0.08)
    # Coach a's two games are worth 0.03 and 0.05: every draw is 0.06, 0.08 or 0.10.
    assert set(np.round(out["a"]["boot"]["cost"], 6)) <= {0.06, 0.08, 0.10}
    assert out["a"]["boot"]["cost"].mean() == pytest.approx(0.08, abs=0.005)
    assert np.all(out["b"]["boot"]["cost"] == 0)
    r = coaches.ratio_summary(0.08, 3.0, out["a"]["boot"]["cost"], out["a"]["boot"]["n"],
                              (0.05, 0.95))  # fmt: skip
    assert r["est"] == pytest.approx(0.08 / 3)


def test_ledger_wins_are_win_probability_points_over_100() -> None:
    # Three decisions that each give up 4 points of win probability cost 0.12 wins.
    g = coaches.grade(np.array([[0.54, 0.5, 0.5]] * 3), ["punt"] * 3)
    assert np.nansum(g["wp_lost"]) == pytest.approx(0.12)


def test_shrinkage_pulls_pure_noise_to_the_mean() -> None:
    rng = np.random.default_rng(3)
    se2 = np.full(200, 0.04)
    noise = coaches.dersimonian_laird(rng.normal(1.0, 0.2, 200), se2)
    assert noise["tau2"] < 0.004 and noise["reliability"] < 0.1
    assert np.nanmax(np.abs(noise["shrunk"] - noise["mean"])) < 0.1
    signal = coaches.dersimonian_laird(rng.normal(0, 1.0, 200) + rng.normal(0, 0.1, 200),
                                       np.full(200, 0.01))  # fmt: skip
    assert signal["reliability"] > 0.95 and np.nanmin(signal["B"]) > 0.95


def test_variance_components_are_recovered_from_simulated_counts() -> None:
    rng = np.random.default_rng(11)
    n_c, n_t, n = 80, 300, 6000
    coach, ts = rng.integers(0, n_c, n), rng.integers(0, n_t, n)
    u_c, u_t = rng.normal(0, 0.35, n_c), rng.normal(0, 0.15, n_t)
    e = np.full(n, 1.5)
    y = rng.poisson(e * np.exp(u_c[coach] + u_t[ts]))
    fit = coaches.fit_poisson_crossed(y, e, {"coach": coach, "ts": ts})
    assert 0.25 < fit["sd"]["coach"] < 0.45
    assert 0.05 < fit["sd"]["ts"] < 0.25
    assert np.corrcoef(fit["effects"]["coach"], u_c)[0, 1] > 0.85
    # No coach effect in the data: the estimate is small.
    y0 = rng.poisson(e * np.exp(u_t[ts]))
    fit0 = coaches.fit_poisson_crossed(y0, e, {"coach": coach, "ts": ts})
    assert fit0["sd"]["coach"] < 0.12


def test_rate_correlations_tell_a_coach_effect_from_a_team_effect() -> None:
    rng = np.random.default_rng(5)
    rows = []
    coach_eff = {f"c{i}": rng.normal(0, 0.6) for i in range(60)}
    for t in range(30):
        c = [f"c{2 * t}", f"c{2 * t + 1}", f"c{(2 * t + 7) % 60}"]
        for s in range(9):
            coach = c[s // 3]
            mu = 20 * np.exp(coach_eff[coach])
            rows.append({"team": f"T{t}", "season": 2011 + s, "coach": coach,
                         "fouls": float(rng.poisson(mu)), "expected": 20.0})
    out = coaches.rate_correlations(pl.DataFrame(rows), 200, 0)
    assert out["same_coach_same_team"]["r"] > 0.6
    assert out["coach_moved_to_new_team"]["r"] > 0.6
    assert out["same_team_new_coach"]["r"] < 0.4


def test_split_half_reliability_formula() -> None:
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(100), 20)
    rate = np.exp(rng.normal(0, 0.5, 100))[g]
    out = coaches.split_half(g, np.tile(np.arange(20), 100), rng.poisson(2 * rate).astype(float),
                             np.full(g.size, 2.0), 200, 0)  # fmt: skip
    assert out["r"] > 0.5
    assert out["spearman_brown"] == pytest.approx(2 * out["r"] / (1 + out["r"]))


# ---------------------------------------------------------------- with data


GRADED = DATA / "processed" / "m7_graded_fourth_downs.parquet"


@pytest.mark.data
def test_every_graded_season_was_graded_by_a_model_that_never_saw_it() -> None:
    if not GRADED.exists():
        pytest.skip("run scripts/m7_coach_evaluation.py first")
    from fourth_and_rl import wp as wp_mod
    from fourth_and_rl.components import ComponentModels

    graded = pl.read_parquet(GRADED, columns=["season", "fold"])
    assert 2026 not in graded["season"].to_list()
    directory = ROOT / CFG["folds"]["models_dir"]
    for (fold,), sub in graded.group_by("fold"):
        seasons = set(sub["season"].unique().to_list())
        if fold.startswith("test"):
            assert seasons <= set(splits.test_seasons())
            continue
        wp = wp_mod.WPModel.load(directory / f"{fold}_wp.joblib")
        comps = ComponentModels.load(directory / f"{fold}_components.joblib")
        assert not seasons & set(wp.stamp["seasons"])
        assert not seasons & set(comps.stamp["seasons"])
        assert not set(wp.stamp["seasons"]) & set(splits.test_seasons())


@pytest.mark.data
def test_the_results_files_hold_intervals_on_every_coach_row() -> None:
    path = RESULTS / "m7_coach_ledger.csv"
    if not path.exists():
        pytest.skip("run scripts/m7_coach_evaluation.py first")
    led = pl.read_csv(path)
    for col in ("decision_wins_confident_lo", "decision_wins_confident_hi",
                "decision_wp_points_per_opportunity", "penalty_wins_lo", "penalty_wins_hi"):
        assert col in led.columns and led[col].null_count() == 0
