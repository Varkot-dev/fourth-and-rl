"""Tests for the game-level bootstrap (M3).

The synthetic tests check the resampling and the decision rule on hand-made inputs.
The `data` test refits real (small) models on one replicate.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fourth_and_rl import bootstrap, decision, wp
from fourth_and_rl.bootstrap import BootstrapPlan, classify, summarize


def toy_prepared() -> pl.DataFrame:
    """Twelve games over three seasons; game g has g + 2 drives of 1 to 3 plays each."""
    rows = []
    for g in range(12):
        season = 2020 + g // 4
        for drive in range(1, g + 3):
            for k in range(1 + (g + drive) % 3):
                seq = drive * 10 + k
                rows.append({"game_id": f"{season}_g{g:02d}", "season": season,
                             "fixed_drive": drive, "order_sequence": seq, "play_id": seq,
                             "payload": g * 1000 + seq})  # fmt: skip
    return pl.DataFrame(rows).sort("game_id", "fixed_drive", "order_sequence")


def plan_for(**kw: object) -> BootstrapPlan:
    return BootstrapPlan(toy_prepared(), (2020, 2020), (2021, 2021), seed=7, **kw)


# ----------------------------------------------------------------------- resampling


def test_draw_is_reproducible_and_differs_between_replicates() -> None:
    a, b = plan_for().draw(3), plan_for().draw(3)
    assert np.array_equal(a["train"], b["train"]) and np.array_equal(a["calib"], b["calib"])
    assert not np.array_equal(plan_for().draw(3)["train"], plan_for().draw(4)["train"]) or (
        not np.array_equal(plan_for().draw(3)["calib"], plan_for().draw(4)["calib"])
    )
    other_seed = BootstrapPlan(toy_prepared(), (2020, 2020), (2021, 2021), seed=8).draw(3)
    assert not (
        np.array_equal(a["train"], other_seed["train"])
        and np.array_equal(a["calib"], other_seed["calib"])
    )


def test_whole_games_are_drawn_and_duplicates_get_their_own_id() -> None:
    plan = plan_for()
    prepared = plan.prepared
    games = np.array([0, 2, 2, 3])  # game 2 drawn twice
    rows = plan.rows("train", games)
    sizes = prepared.filter(pl.col("season") == 2020).group_by("game_id", maintain_order=True).len()
    assert rows.height == int(sizes["len"].to_numpy()[games].sum())
    assert rows["game_id"].n_unique() == 4  # the duplicate counts as a separate game
    per_draw = rows.group_by("game_id", maintain_order=True).len()["len"].to_list()
    assert per_draw == sizes["len"].to_numpy()[games].tolist()
    # Every play of a drawn game is present exactly once per draw.
    twice = rows.filter(pl.col("game_id").str.starts_with("2020_g02"))
    original = prepared.filter(pl.col("game_id") == "2020_g02")
    assert sorted(twice["payload"].to_list()) == sorted(original["payload"].to_list() * 2)


def test_only_the_configured_seasons_are_ever_drawn() -> None:
    plan = plan_for()
    assert plan.blocks["train"].n_games == 4 and plan.blocks["calib"].n_games == 4
    for b in range(20):
        draw = plan.draw(b)
        assert draw["train"].size == 4 and draw["calib"].size == 4
        seasons = set(plan.rows("train", draw["train"])["season"].to_list())
        seasons |= set(plan.rows("calib", draw["calib"])["season"].to_list())
        assert seasons <= {2020, 2021}  # 2022, the "test" season, never appears


def test_drives_option_resamples_whole_drives_within_each_drawn_game() -> None:
    plan = plan_for(within_game="drives")
    draw = plan.draw(0)
    rows = plan.rows("train", draw["train"], int(draw["_drive_seed"][0]))
    prepared = plan.prepared
    drive_len = {
        (r["game_id"], r["fixed_drive"]): r["len"]
        for r in prepared.group_by("game_id", "fixed_drive").len().to_dicts()
    }
    n_drives = {r["game_id"]: r["len"] for r in
                prepared.select("game_id", "fixed_drive").unique().group_by("game_id").len()
                .to_dicts()}  # fmt: skip
    assert rows["game_id"].n_unique() == draw["train"].size
    for (new_id, drive), group in rows.group_by("game_id", "fixed_drive"):
        original = new_id.split("#")[0]
        assert group.height % drive_len[(original, drive)] == 0  # whole drives only
    for (new_id,), group in rows.group_by("game_id"):
        original = new_id.split("#")[0]
        pieces = sum(
            g.height // drive_len[(original, d)] for (d,), g in group.group_by("fixed_drive")
        )
        assert pieces == n_drives[original]  # as many drives as the real game had


def test_game_fraction_draws_fewer_games() -> None:
    plan = plan_for(game_fraction=0.5)
    assert plan.draw(0)["train"].size == 2
    with pytest.raises(ValueError):
        plan_for(game_fraction=0.0)
    with pytest.raises(ValueError):
        plan_for(within_game="plays")


# -------------------------------------------------------------------- decision rule


def test_classify_uses_the_published_thresholds() -> None:
    out = classify(np.array([1.0, 0.83, 0.8299, 0.67, 0.6699, 0.0, np.nan]))
    assert out.tolist() == ["confident", "confident", "lean", "lean", "uncertain", "uncertain",
                            None]  # fmt: skip
    # 84 of 101 replicates is confident, 83 of 101 is not (0.8218).
    assert classify(np.array([84 / 101, 83 / 101])).tolist() == ["confident", "lean"]


def test_summarize_counts_agreement_with_the_point_estimate() -> None:
    n_boot = 101
    point = np.array([[0.50, 0.40, np.nan],   # go by 10 points
                      [0.50, np.nan, 0.51],   # punt by 1 point
                      [np.nan, np.nan, np.nan]])  # not scored  # fmt: skip
    boot = np.repeat(point[None], n_boot, axis=0).astype(np.float32)
    # State 0: go stays best in 90 replicates, the kick wins in 11.
    boot[:11, 0, 1] = 0.60
    # State 1: the gain of going runs from -0.050 to +0.050 across replicates.
    boot[:, 1, 0] = 0.51 + np.linspace(-0.05, 0.05, n_boot)
    s = summarize(point, boot)
    assert s["best"].tolist() == [0, 2, -1]
    assert s["boot_pct"][0] == pytest.approx(90 / 101)
    assert s["category"][0] == "confident"
    assert s["share_field_goal"][0] == pytest.approx(11 / 101)
    # Punt is best when the gain is negative: 50 of the 101 replicates.
    assert s["boot_pct"][1] == pytest.approx(50 / 101)
    assert s["category"][1] == "uncertain"
    # 90% interval = 6th and 96th of the 101 sorted gains.
    gains = np.sort(np.linspace(-0.05, 0.05, n_boot))
    assert s["gain_lo"][1] == pytest.approx(gains[5], abs=1e-6)
    assert s["gain_hi"][1] == pytest.approx(gains[95], abs=1e-6)
    assert s["gain_median"][1] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(s["boot_pct"][2]) and s["category"][2] is None


def test_category_shares_and_split_half() -> None:
    rng = np.random.default_rng(1)
    n = 300
    point = np.column_stack([rng.normal(0.5, 0.02, n), rng.normal(0.5, 0.02, n), np.full(n, 0.1)])
    boot = (point[None] + rng.normal(0, 0.01, (40, n, 3))).astype(np.float32)
    s = summarize(point, boot)
    shares = bootstrap.category_shares(s["category"], s["best"])
    assert shares["n"] == n
    assert shares["confident"] + shares["lean"] + shares["uncertain"] == pytest.approx(1.0)
    assert sum(v["n"] for v in shares["by_best_action"].values()) == n
    half = bootstrap.split_half_agreement(point, boot)
    assert half["replicates_per_half"] == [20, 20]
    assert 0.5 < half["category_agreement"] <= 1.0
    same = bootstrap.split_half_agreement(point, np.repeat(boot[:1], 10, axis=0))
    assert same["category_agreement"] == 1.0


# ------------------------------------------------------------------ one real refit


@pytest.mark.data
def test_one_replicate_refits_both_models_without_test_seasons(plays: pl.DataFrame) -> None:
    prepared = bootstrap.prepare_plays(plays.filter(pl.col("season") >= 2018))
    assert prepared.height == plays.filter(pl.col("season") >= 2018).height
    plan = BootstrapPlan(prepared, (2018, 2021), (2022, 2023), seed=11)
    wp_config = {"seed": 5, "lgbm": {**wp.load_config()["lgbm"], "n_estimators": 25}}
    model, comps, draw = bootstrap.fit_replicate(plan, 0, wp_config, None, [2025], n_threads=2)
    assert draw["train"].size == plan.blocks["train"].n_games
    assert comps.seasons[1] <= 2023  # no test-season play in the outcome models
    assert comps.kickoff_info["seasons"] == [2025]  # the fixed kickoff-rule table
    assert model.calibrator.method == "platt" and model.n_calib_rows > 0
    engine = decision.DecisionEngine(model, comps, decision.load_config()["engine"])
    states = wp.state_frame(down=4, ydstogo=[1, 4, 12], yardline_100=[45, 60, 85],
                            game_seconds_remaining=2400)  # fmt: skip
    values = engine.action_values(states)
    assert np.all((values["go_wp"] > 0) & (values["go_wp"] < 1))
    assert values["go_wp"][0] > values["punt_wp"][0]  # fourth and 1 near midfield
    # A different replicate is a different draw and gives different numbers.
    other, _, draw2 = bootstrap.fit_replicate(plan, 1, wp_config, None, [2025], n_threads=2)
    assert not np.array_equal(draw["train"], draw2["train"])
    assert not np.allclose(other.predict(states), model.predict(states))
