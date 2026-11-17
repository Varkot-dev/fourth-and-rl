"""Data fingerprints in cache keys: a changed play table must never reuse old fits."""

from __future__ import annotations

import json

import polars as pl
import pytest

from fourth_and_rl import bootstrap, splits
from fourth_and_rl import components as cm
from fourth_and_rl.fingerprint import KEY_COLUMNS, data_fingerprint


def tiny_plays(n_games: int = 2, seasons: tuple[int, ...] = (2018, 2022)) -> pl.DataFrame:
    rows = []
    for season in seasons:
        for g in range(n_games):
            for drive in (1, 2):
                for k in range(3):
                    rows.append({
                        "game_id": f"{season}_{g:02d}", "play_id": float(100 * drive + k),
                        "order_sequence": float(100 * drive + k), "season": season,
                        "fixed_drive": float(drive), "down": float(k + 1), "ydstogo": 10.0,
                        "yardline_100": float(75 - 5 * k), "play_type": "run",
                        "yards_gained": 5.0, "posteam_won": float(g % 2),
                    })  # fmt: skip
    return pl.DataFrame(rows)


def test_fingerprint_counts_rows_and_games_per_season() -> None:
    fp = data_fingerprint(tiny_plays())
    assert fp["n_rows"] == 24
    assert fp["by_season"] == {"2018": {"rows": 12, "games": 2}, "2022": {"rows": 12, "games": 2}}
    assert fp["columns_hashed"] == list(KEY_COLUMNS)
    assert fp["columns_missing"] == []
    assert json.loads(json.dumps(fp)) == fp  # survives a key file


def test_fingerprint_ignores_row_order_and_extra_columns_but_sees_values() -> None:
    df = tiny_plays()
    fp = data_fingerprint(df)
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=3).with_columns(pl.lit(1).alias("x"))
    assert data_fingerprint(shuffled) == fp
    changed = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 5).then(6.0).otherwise(pl.col("yards_gained"))
        .alias("yards_gained")
    )  # fmt: skip
    assert data_fingerprint(changed)["hash"] != fp["hash"]
    assert data_fingerprint(changed)["by_season"] == fp["by_season"]
    nulled = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("posteam_won"))
    assert data_fingerprint(nulled)["hash"] != fp["hash"]
    fewer = df.head(23)
    assert data_fingerprint(fewer)["by_season"] != fp["by_season"]
    missing = df.drop("posteam_won")
    assert data_fingerprint(missing)["columns_missing"] == ["posteam_won"]


def _plan(df: pl.DataFrame) -> bootstrap.BootstrapPlan:
    return bootstrap.BootstrapPlan(df.sort("game_id", "fixed_drive", "order_sequence"),
                                   (2018, 2021), (2022, 2023), seed=1)  # fmt: skip


def test_replicate_key_carries_the_data_fingerprint() -> None:
    df = tiny_plays()
    plan = _plan(df)
    key = bootstrap.replicate_key(plan, 0, {"seed": 1}, {"a": 1}, [2025], "none")
    assert key["data_fingerprint"] == data_fingerprint(df)
    changed = df.with_columns(pl.col("yards_gained") + 1)
    other = bootstrap.replicate_key(_plan(changed), 0, {"seed": 1}, {"a": 1}, [2025], "none")
    assert other != key  # same config and seasons, different plays: not reused
    # Test-season games are never drawn: they do not enter the drawn-games fingerprint,
    # but their kickoff rows (the fixed kickoff-rule tables) have their own.
    later = tiny_plays(seasons=(2025,)).with_columns(pl.lit("kickoff").alias("play_type"))
    both = pl.concat([df, later])
    with_test = bootstrap.replicate_key(_plan(both), 0, {"seed": 1}, {"a": 1}, [2025], "none")
    assert with_test["data_fingerprint"] == key["data_fingerprint"]
    assert with_test["kickoff_data_fingerprint"] == data_fingerprint(later)
    assert key["kickoff_data_fingerprint"]["n_rows"] == 0


def test_assert_replicates_current_compares_the_fingerprint(tmp_path) -> None:
    df = tiny_plays()
    plan = _plan(df)
    cfg = {"seed": 1, "within_game": "none", "calibration": "none", "kickoff_seasons": [2025]}
    key = bootstrap.replicate_key(plan, 0, {"seed": 1}, {"a": 1}, [2025], "none")
    bootstrap.replicate_paths(tmp_path, 0)["key"].write_text(json.dumps(key))
    fp = data_fingerprint(df)
    bootstrap.assert_replicates_current(tmp_path, 1, cfg, {"seed": 1}, {"a": 1}, fp)
    stale = data_fingerprint(df.head(20))
    with pytest.raises(RuntimeError, match="data_fingerprint"):
        bootstrap.assert_replicates_current(tmp_path, 1, cfg, {"seed": 1}, {"a": 1}, stale)
    # A key saved before fingerprints existed is stale once a fingerprint is asked for.
    del key["data_fingerprint"]
    bootstrap.replicate_paths(tmp_path, 0)["key"].write_text(json.dumps(key))
    with pytest.raises(RuntimeError, match="data_fingerprint"):
        bootstrap.assert_replicates_current(tmp_path, 1, cfg, {"seed": 1}, {"a": 1}, fp)


def _fake_bundle(stamp_fp: dict | None) -> cm.ComponentModels:
    cfg = cm.load_config()
    models = object.__new__(cm.ComponentModels)
    models.config = cfg
    models.stamp = {"seasons": list(splits.dev_seasons()), "config_hash": cm.config_hash(cfg)}
    if stamp_fp is not None:
        models.stamp["data_fingerprint"] = stamp_fp
    models.kickoff_tables = {e["name"]: None for e in cm.kickoff_rule_eras(cfg)}
    return models


def test_dev_bundle_is_current_compares_the_data_fingerprint() -> None:
    df = tiny_plays()
    assert cm.dev_bundle_is_current(_fake_bundle(data_fingerprint(df)), dev_plays=df)
    changed = df.with_columns(pl.col("ydstogo") - 1)
    assert not cm.dev_bundle_is_current(_fake_bundle(data_fingerprint(df)), dev_plays=changed)
    assert not cm.dev_bundle_is_current(_fake_bundle(None), dev_plays=df)
