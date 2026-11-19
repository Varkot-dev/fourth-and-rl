"""Shared fixtures. Tests marked `data` are skipped when the cache is absent (as in CI)."""

from __future__ import annotations

import polars as pl
import pytest

from fourth_and_rl import data


@pytest.fixture(scope="session")
def plays() -> pl.DataFrame:
    if not data.plays_path().exists():
        pytest.skip("play-by-play cache not built (run scripts/build_data.py)")
    return data.load_plays()
