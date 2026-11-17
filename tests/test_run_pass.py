"""Tests for burning question 2 (run versus pass).

The synthetic tests build small play tables with known answers and run in CI without
data. The `data` tests use the real table and are skipped when the cache is absent.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fourth_and_rl import run_pass as rp
from fourth_and_rl.paths import ROOT

CFG = rp.load_config()


def toy(rows: list[dict]) -> pl.DataFrame:
    """A play table from short row dicts; everything not given gets a neutral default."""
    base = {
        "game_id": "g0", "play_id": 0.0, "order_sequence": 0.0, "season": 2020,
        "posteam": "A", "defteam": "B", "fixed_drive": 1.0, "game_half": "Half1",
        "play_type": "run", "down": 1.0, "ydstogo": 10.0, "yardline_100": 60.0,
        "yards_gained": 3.0, "epa": 0.0, "first_down": 0.0, "touchdown": 0.0, "td_team": None,
        "interception": 0.0, "fumble_lost": 0.0, "sack": 0.0, "qb_kneel": 0.0, "qb_spike": 0.0,
        "two_point_attempt": 0.0, "qb_scramble": 0.0, "score_differential": 0.0, "wp": 0.5,
        "half_seconds_remaining": 900.0, "game_seconds_remaining": 2700.0, "qtr": 2.0,
        "defteam_timeouts_remaining": 3.0,
    }  # fmt: skip
    out = []
    for i, r in enumerate(rows):
        row = {**base, "play_id": float(i), "order_sequence": float(i), **r}
        out.append(row)
    schema = {k: (pl.String if isinstance(v, str) or v is None else type_of(v))
              for k, v in base.items()}  # fmt: skip
    return pl.DataFrame(out, schema=schema)


def type_of(v: object) -> type[pl.DataType]:
    return pl.Int64 if isinstance(v, int) else pl.Float64


# ---- classification ----


def test_classify_calls_handles_scrambles_sacks_kneels_and_spikes() -> None:
    df = toy([
        {"play_type": "run"},                                # designed run
        {"play_type": "pass"},                               # pass
        {"play_type": "pass", "sack": 1.0},                  # sack is a pass
        {"play_type": "run", "qb_scramble": 1.0},            # scramble: called pass
        {"play_type": "qb_kneel", "qb_kneel": 1.0},          # out
        {"play_type": "qb_spike", "qb_spike": 1.0},          # out
        {"play_type": "pass", "down": None, "two_point_attempt": 1.0},  # out
        {"play_type": "no_play"},                            # penalty, out
        {"play_type": "punt", "down": 4.0},                  # out
    ])  # fmt: skip
    expect = ["run", "pass", "pass", "pass", None, None, None, None, None]
    assert rp.classify_calls(df, "pass")["call"].to_list() == expect
    as_recorded = rp.classify_calls(df, "run")["call"].to_list()
    assert as_recorded[3] == "run" and as_recorded[:3] == expect[:3]
    assert rp.classify_calls(df, "drop")["call"].to_list()[3] is None
    with pytest.raises(ValueError):
        rp.classify_calls(df, "sometimes")


def test_classify_without_scramble_column_keeps_recorded_runs() -> None:
    df = toy([{"play_type": "run", "qb_scramble": 1.0}]).drop("qb_scramble")
    assert rp.classify_calls(df)["call"].to_list() == ["run"]


def test_prepare_drops_uncalled_plays_and_adds_measures() -> None:
    df = toy([
        {"play_type": "pass", "epa": 0.8, "yards_gained": 12.0, "first_down": 1.0},
        {"play_type": "pass", "epa": -4.0, "interception": 1.0, "yards_gained": 0.0},
        {"play_type": "run", "epa": -0.2},
        {"play_type": "qb_kneel", "qb_kneel": 1.0, "epa": -0.5},
        {"play_type": "run", "epa": None},
    ])  # fmt: skip
    out = rp.prepare(df, CFG)
    assert out.height == 3
    assert out["is_pass"].to_list() == [1, 1, 0]
    assert out["success"].to_list() == [1.0, 0.0, 0.0]
    assert out["turnover"].to_list() == [0.0, 1.0, 0.0]
    assert out["game_code"].to_list() == [0, 0, 0]


# ---- the neutral filter ----


def test_neutral_filter() -> None:
    df = toy([
        {},                                             # neutral
        {"score_differential": 8.0},                    # still one score
        {"score_differential": -9.0},                   # trailing by two scores
        {"wp": 0.85},                                   # game nearly decided
        {"wp": 0.15},
        {"half_seconds_remaining": 120.0},              # two-minute drill
        {"half_seconds_remaining": 121.0},
        {"qtr": 5.0},                                   # overtime
        {"wp": None},                                   # unknown is not neutral
    ])  # fmt: skip
    got = df.select(rp.neutral_expr(CFG).alias("n"))["n"].to_list()
    assert got == [True, True, False, False, False, False, True, False, False]


def test_cells_are_distinct_and_labelled() -> None:
    labels = rp.cell_labels(CFG)
    assert len(labels) == rp.n_cells(CFG)
    df = rp.prepare(toy([
        {"down": 1.0, "ydstogo": 10.0, "yardline_100": 75.0},
        {"down": 3.0, "ydstogo": 1.0, "yardline_100": 2.0},
    ]), CFG)  # fmt: skip
    first, second = (labels[c] for c in df["cell"].to_list())
    assert first == {"down": 1, "distance": "8-10", "yardline_100": "61-80"}
    assert second == {"down": 3, "distance": "1", "yardline_100": "1-4"}
    assert rp.bin_index(np.array([1, 2, 4, 7, 10, 25]), CFG["dist_edges"]).tolist() == [
        0, 1, 2, 3, 4, 5
    ]  # fmt: skip


# ---- bootstrap and the cell comparison ----


def simpson_frame() -> pl.DataFrame:
    """Two situations with no noise. In both, a pass is worth 0.2 more than a run.

    Easy situation (1st and 10): runs 0.3, passes 0.5; 90 runs and 10 passes per game.
    Hard situation (3rd and 12): runs -1.0, passes -0.8; 10 runs and 90 passes per game.
    Passes mostly happen in the hard situation, so the plain averages say running is
    better: run mean 0.17, pass mean -0.67.
    """
    rows = []
    for g in range(6):
        for call, n, epa, down, togo in (("run", 90, 0.3, 1.0, 10.0), ("pass", 10, 0.5, 1.0, 10.0),
                                         ("run", 10, -1.0, 3.0, 12.0),
                                         ("pass", 90, -0.8, 3.0, 12.0)):  # fmt: skip
            rows += [{"game_id": f"g{g}", "play_type": call, "epa": epa, "down": down,
                      "ydstogo": togo}] * n  # fmt: skip
    return toy(rows)


def test_boot_weights_resample_whole_games() -> None:
    w = rp.boot_weights(50, 200, seed=1)
    assert w.shape == (200, 50)
    assert np.all(w.sum(axis=1) == 50)
    assert np.array_equal(w, rp.boot_weights(50, 200, seed=1))


def test_cell_comparison_recovers_a_known_answer() -> None:
    df = rp.prepare(simpson_frame(), CFG)
    w = rp.boot_weights(6, 100, seed=0)
    comp = rp.frame_compare(df, df["cell"].to_numpy(), rp.n_cells(CFG), w, metrics=["epa"])
    verdict = rp.classify_cells(comp, "epa", min_n=50)
    assert (verdict == 1).sum() == 2  # pass better in both cells that have plays
    assert set(verdict.tolist()) == {1, -9}
    shares = rp.cell_shares(comp, verdict)
    assert shares["pass_better"] == pytest.approx(1.0)
    gap = rp.standardized_gap(comp, "epa", min_n=50)
    assert gap["standardized"]["value"] == pytest.approx(0.2)
    assert gap["raw"]["value"] == pytest.approx(-0.67 - 0.17)  # the misleading average
    assert gap["standardized"]["low"] == pytest.approx(0.2, abs=1e-5)
    assert gap["coverage"] == pytest.approx(1.0)
    # Too few plays per cell: nothing to compare.
    assert rp.standardized_gap(comp, "epa", min_n=10_000)["n_cells"] == 0


def test_compare_matches_plain_means_and_gives_a_sensible_interval() -> None:
    rng = np.random.default_rng(3)
    n, n_games = 6000, 60
    game = rng.integers(0, n_games, n)
    is_pass = rng.integers(0, 2, n)
    epa = rng.normal(0.1 * is_pass, 1.0)
    w = rp.boot_weights(n_games, 400, seed=5)
    comp = rp.compare({"epa": epa}, is_pass, np.zeros(n, dtype=np.int64), 1, game, w,
                      sd_of=("epa",))  # fmt: skip
    assert comp.est[0, 0, 0] == pytest.approx(epa[is_pass == 0].mean())
    assert comp.est[0, 1, 0] == pytest.approx(epa[is_pass == 1].mean())
    assert comp.est[0, 1, 1] == pytest.approx(epa[is_pass == 1].std(), rel=1e-6)
    lo, hi = comp.lo[0, 2, 0], comp.hi[0, 2, 0]
    assert lo < comp.est[0, 2, 0] < hi
    # Standard error of a difference of two means of about 3000 unit-variance draws.
    assert (hi - lo) / 3.92 == pytest.approx(np.sqrt(2 / 3000), rel=0.25)
    row = comp.rows(["all"])[0]
    assert row["n_run"] + row["n_pass"] == n
    assert row["epa"]["pass_minus_run"]["value"] == pytest.approx(comp.est[0, 2, 0])


def test_group_means_and_cluster_ols() -> None:
    rng = np.random.default_rng(0)
    n, n_games = 2000, 40
    game = rng.integers(0, n_games, n)
    x = rng.normal(size=n)
    y = 1.5 - 0.7 * x  # no noise: every resample gives the same line
    w = rp.boot_weights(n_games, 50, seed=2)
    fit = rp.cluster_ols(np.column_stack([np.ones(n), x]), y, np.ones(n), game, w)
    assert fit["coef"] == pytest.approx([1.5, -0.7], abs=1e-6)
    assert fit["low"][1] == pytest.approx(-0.7, abs=1e-5)
    means = rp.group_means(x, (x > 0).astype(np.int64), 2, game, w)
    assert means[0]["value"] == pytest.approx(x[x <= 0].mean())
    assert means[0]["low"] <= means[0]["value"] <= means[0]["high"]
    assert rp.dummies(np.array([2, 5, 5, 9])).tolist() == [[0, 0], [1, 0], [1, 0], [0, 1]]


def test_team_tendency_leaves_the_game_out() -> None:
    rows = []
    # Team A passes on every play of game 1 and runs on every play of games 2 and 3.
    for game, call in (("g1", "pass"), ("g2", "run"), ("g3", "run")):
        rows += [{"game_id": game, "play_type": call, "epa": 1.0 if call == "pass" else -1.0}] * 10
    tg = rp.team_game_table(rp.prepare(toy(rows), CFG)).sort("game_id")
    assert tg["loo_pass_rate"].to_list() == pytest.approx([0.0, 0.5, 0.5])
    assert tg["pass_epa"].to_list()[0] == pytest.approx(1.0)
    assert tg["loo_pass_epa"].to_list()[0] is None  # no passes in the other games
    assert tg["loo_run_epa"].to_list()[1] == pytest.approx(-1.0)


def test_previous_call_and_clock_columns() -> None:
    df = rp.prepare(toy([
        {"play_type": "pass", "game_seconds_remaining": 300.0, "defteam_timeouts_remaining": 2.0},
        {"play_type": "run", "game_seconds_remaining": 295.0, "defteam_timeouts_remaining": 2.0},
        {"play_type": "run", "game_seconds_remaining": 289.0, "defteam_timeouts_remaining": 1.0},
        {"play_type": "pass", "game_seconds_remaining": 250.0, "fixed_drive": 2.0},
    ]), CFG)  # fmt: skip
    assert df["seconds_elapsed"].to_list() == [5.0, 6.0, 39.0, None]
    assert df["defense_timeout_after"].to_list() == [0.0, 1.0, 0.0, 0.0]
    assert df["prev_is_pass"].to_list() == [None, 1.0, 0.0, None]  # new drive starts fresh


# ---- the hook ----


def synthetic_hook_plays(seed: int = 0, n: int = 20_000) -> pl.DataFrame:
    """Runs gain 3 or 4 yards; passes gain 0 (40%) or 12 (60%), with 4% interceptions.
    Scrambles (recorded as runs) gain 9. Next-state columns are filled in directly."""
    rng = np.random.default_rng(seed)
    kind = rng.choice(["run", "pass", "scramble"], n, p=[0.45, 0.45, 0.10])
    down = rng.integers(1, 4, n)
    yl = rng.integers(1, 100, n)
    togo = np.minimum(rng.integers(1, 13, n), yl)
    throw = np.where(rng.random(n) < 0.6, 12, 0)
    gain = np.where(kind == "run", rng.integers(3, 5, n), np.where(kind == "scramble", 9, throw))
    td = gain >= yl
    gain = np.where(td, yl, gain)
    turnover = (kind == "pass") & (rng.random(n) < 0.04) & ~td
    first = (gain >= togo) & ~turnover
    return pl.DataFrame({
        "play_type": np.where(kind == "pass", "pass", "run").tolist(),
        "qb_scramble": (kind == "scramble").astype(float),
        "down": down.astype(float), "ydstogo": togo.astype(float),
        "yardline_100": yl.astype(float), "yards_gained": gain.astype(float),
        "first_down": first.astype(float), "touchdown": td.astype(float),
        "td_team": pl.Series(np.where(td, "A", None).tolist(), dtype=pl.String),
        "interception": turnover.astype(float),
        "fumble_lost": np.zeros(n), "qb_kneel": np.zeros(n), "qb_spike": np.zeros(n),
        "two_point_attempt": np.zeros(n), "posteam": ["A"] * n, "defteam": ["B"] * n,
        "game_half": ["Half1"] * n, "season": rng.integers(2018, 2024, n),
        "next_posteam": np.where(turnover | td, "B", "A").tolist(),
        "next_yardline_100": np.where(turnover, np.clip(100 - yl, 1, 99),
                                      np.clip(yl - gain, 1, 99)).astype(float),
        "next_half": ["Half1"] * n, "next_row_seconds": np.full(n, 2670.0),
        "next_row_half": ["Half1"] * n,
    })  # fmt: skip


@pytest.fixture(scope="module")
def hook_model() -> rp.RunPassOutcomes:
    return rp.RunPassOutcomes(min_cell_n=50, min_turnover_n=20).fit(synthetic_hook_plays())


@pytest.mark.parametrize("call", rp.CALLS)
@pytest.mark.parametrize("state", [(1, 10, 75), (2, 3, 40), (3, 1, 1), (3, 12, 99), (1, 5, 5)])
def test_hook_distributions_sum_to_one(hook_model: rp.RunPassOutcomes, call: str,
                                       state: tuple[int, int, int]) -> None:  # fmt: skip
    o = hook_model.outcomes(call, *state)
    assert o["keep"].shape == (100,) and o["turnover"].shape == (100,)
    assert o["keep"].min() >= 0 and o["turnover"].min() >= 0
    total = o["p_touchdown"] + o["p_safety"] + o["keep"].sum() + o["turnover"].sum()
    assert total == pytest.approx(1.0)
    assert o["p_turnover"] == pytest.approx(o["turnover"].sum())
    assert 0.0 <= o["p_first_down"] <= 1.0


def test_hook_separates_run_from_pass(hook_model: rp.RunPassOutcomes) -> None:
    run = hook_model.outcomes("run", 1, 10, 60)
    pas = hook_model.outcomes("pass", 1, 10, 60)
    assert run["p_turnover"] == 0.0
    assert run["p_first_down"] == pytest.approx(0.0, abs=1e-9)  # 3 or 4 yards, 10 to go
    assert run["keep"][56:58].sum() == pytest.approx(1.0)       # ball at the 57 or 56
    # Called passes: 45 of 55 are throws (4% intercepted, else 60% gain 12), 10 scrambles
    # that gain 9 and fall short of the 10 to go.
    assert pas["p_turnover"] == pytest.approx(0.04 * 45 / 55, abs=0.015)
    assert pas["p_first_down"] == pytest.approx(0.6 * 0.96 * 45 / 55, abs=0.05)
    assert pas["keep"][51] > 0.05  # the scrambles, 9 yards
    assert hook_model.pass_share(1, 10, 60) == pytest.approx(0.55, abs=0.05)
    assert hook_model.expected_yards("run", 1, 10, 60) == pytest.approx(3.5, abs=0.15)
    # Third and 1: every run converts.
    assert hook_model.outcomes("run", 3, 1, 60)["p_first_down"] == pytest.approx(1.0)


def test_hook_scrambles_as_recorded_and_bad_arguments(hook_model: rp.RunPassOutcomes) -> None:
    recorded = rp.RunPassOutcomes(min_cell_n=50, min_turnover_n=20, scrambles_as="run").fit(
        synthetic_hook_plays()
    )
    assert recorded.n_["run"] > hook_model.n_["run"]
    assert recorded.outcomes("run", 1, 10, 60)["keep"][51] > 0.05
    with pytest.raises(ValueError):
        hook_model.outcomes("punt", 1, 10, 60)
    with pytest.raises(ValueError):
        hook_model.outcomes("run", 4, 1, 60)


# ---- real data ----


def extra_available(plays: pl.DataFrame) -> bool:
    seasons = plays["season"].unique().to_list()
    return all((ROOT / CFG["extra_dir"] / f"extra_{s}.parquet").exists() for s in seasons)


@pytest.fixture(scope="module")
def prepared(plays: pl.DataFrame) -> pl.DataFrame:
    if not extra_available(plays):
        pytest.skip("extra nflverse columns not cached (run scripts/questions/q2_run_vs_pass.py)")
    extra = rp.load_extra(sorted(plays["season"].unique().to_list()), CFG)
    return rp.prepare(rp.with_extra(plays, extra), CFG)


@pytest.mark.data
def test_real_classification_as_recorded(plays: pl.DataFrame) -> None:
    df = rp.classify_calls(plays, "run").filter(pl.col("call").is_not_null())
    assert (df["call"] == df["play_type"]).all()
    assert df.filter((pl.col("qb_kneel") == 1) | (pl.col("qb_spike") == 1)).height == 0
    assert df.filter(pl.col("sack") == 1)["call"].unique().to_list() == ["pass"]
    assert 0.55 < (df["call"] == "pass").mean() < 0.62


@pytest.mark.data
def test_real_scrambles_are_recorded_as_runs_and_moved(prepared: pl.DataFrame) -> None:
    scr = prepared.filter(pl.col("qb_scramble") == 1)
    assert scr.height > 10_000
    assert (scr["call"] == "pass").all()
    assert (scr["play_type"] == "run").mean() > 0.99  # a handful are recorded as passes
    assert 0.45 < prepared["neutral"].mean() < 0.60


@pytest.mark.data
def test_real_gap_is_positive_and_survives_the_neutral_filter(prepared: pl.DataFrame) -> None:
    w = rp.boot_weights(int(prepared["game_code"].max()) + 1, 100, CFG["seed"])
    group = prepared["neutral"].to_numpy().astype(np.int64)
    comp = rp.frame_compare(prepared, group, 2, w, metrics=["epa", "yards", "turnover"])
    for g in (0, 1):
        assert 0.07 < comp.est[g, 2, 0] < 0.16 and comp.lo[g, 2, 0] > 0   # EPA
        assert 1.5 < comp.est[g, 2, 1] < 3.0                              # yards
        assert comp.est[g, 2, 2] > 0.01                                   # turnovers
    nd = prepared.filter(pl.col("neutral"))
    cells = rp.frame_compare(nd, nd["cell"].to_numpy(), rp.n_cells(CFG), w, metrics=["epa"])
    gap = rp.standardized_gap(cells, "epa", CFG["cells"]["min_n_per_call"])
    assert gap["standardized"]["low"] > 0.1 and gap["coverage"] > 0.8


@pytest.mark.data
def test_real_hook_sums_to_one_and_mixes_back(plays: pl.DataFrame) -> None:
    h = CFG["hook"]
    recent = plays.filter(pl.col("season") >= 2022)
    model = rp.RunPassOutcomes(h["min_cell_n"], h["min_turnover_n"], h["era_first_season"]).fit(
        recent
    )
    for down, togo, yl in [(1, 10, 75), (2, 7, 50), (3, 1, 1), (3, 15, 99)]:
        for call in rp.CALLS:
            o = model.outcomes(call, down, togo, yl)
            total = o["p_touchdown"] + o["p_safety"] + o["keep"].sum() + o["turnover"].sum()
            assert total == pytest.approx(1.0)
    run, pas = model.outcomes("run", 1, 10, 75), model.outcomes("pass", 1, 10, 75)
    assert pas["p_first_down"] > run["p_first_down"]
    assert pas["p_turnover"] > run["p_turnover"]
    assert model.outcomes("run", 3, 1, 50)["p_first_down"] > 0.6
