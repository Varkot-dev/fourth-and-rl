"""Season splits and the guards that keep test seasons out of model fitting.

The rule: nothing fitted on the test seasons may produce a number that is reported
on the test seasons. This module makes that rule hard to break by accident:

- `configs/splits.yaml` is the one place the seasons are written down.
- Development plays (train + calibration seasons) and test plays live in separate
  files on disk. Test plays are loaded only through `load_test_plays`, which requires
  a stated purpose and records it.
- `assert_no_test_seasons` raises `LeakageError` when a frame headed for a fit
  contains test-season rows. Every `fit` calls it unless the caller passes an
  explicit, logged exception.
- `fit_stamp` describes what a model was fitted on, and `assert_can_score` refuses
  to score rows from seasons or games the model has seen.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import polars as pl
import yaml

from fourth_and_rl.paths import CONFIGS, DATA, RESULTS

ACCESS_LOG = RESULTS / "test_access_log.jsonl"


class LeakageError(RuntimeError):
    """Raised when test-season data is about to reach a place it must not."""


@lru_cache(maxsize=1)
def load_splits(path: Path | None = None) -> dict[str, tuple[int, ...]]:
    with open(path or CONFIGS / "splits.yaml") as f:
        raw = yaml.safe_load(f)
    splits = {k: tuple(int(s) for s in raw[k]) for k in
              ("train_seasons", "calibration_seasons", "test_seasons")}
    seen: set[int] = set()
    for name, seasons in splits.items():
        overlap = seen & set(seasons)
        if overlap:
            raise LeakageError(f"{name} overlaps another split: {sorted(overlap)}")
        seen |= set(seasons)
    return splits


def train_seasons() -> tuple[int, ...]:
    return load_splits()["train_seasons"]


def calibration_seasons() -> tuple[int, ...]:
    return load_splits()["calibration_seasons"]


def test_seasons() -> tuple[int, ...]:
    return load_splits()["test_seasons"]


def dev_seasons() -> tuple[int, ...]:
    """Seasons that may be used for fitting and tuning: train plus calibration."""
    return train_seasons() + calibration_seasons()


def seasons_in(df: pl.DataFrame) -> set[int]:
    return {int(s) for s in df["season"].unique().to_list()}


def assert_no_test_seasons(
    df: pl.DataFrame,
    what: str,
    allow_test_seasons: bool = False,
    reason: str | None = None,
) -> None:
    """Raise `LeakageError` if `df` holds test-season rows.

    `what` names the thing being fitted, for the error message. The only way past
    the guard is `allow_test_seasons=True` together with a written `reason`, which
    is appended to the access log so every exception is on record.
    """
    leaked = seasons_in(df) & set(test_seasons())
    if not leaked:
        return
    if allow_test_seasons:
        if not reason or len(reason.strip()) < 15:
            raise LeakageError(
                f"{what}: allow_test_seasons=True needs a written reason (15+ characters)"
            )
        _log_access(kind="fit_on_test_seasons", what=what, reason=reason, seasons=leaked)
        return
    raise LeakageError(
        f"{what}: received rows from test seasons {sorted(leaked)}. Fit on development "
        "seasons only, or pass allow_test_seasons=True with a reason if no number "
        "reported on the test seasons depends on this fit."
    )


def assert_disjoint_games(a: pl.DataFrame, b: pl.DataFrame, what: str) -> None:
    """Raise if any game appears in both frames (a split must never cut a game)."""
    shared = set(a["game_id"].unique().to_list()) & set(b["game_id"].unique().to_list())
    if shared:
        raise LeakageError(f"{what}: {len(shared)} games appear on both sides of a split")


def fit_stamp(df: pl.DataFrame, what: str) -> dict:
    """A record of what a model was fitted on, saved alongside the model."""
    return {
        "what": what,
        "seasons": sorted(seasons_in(df)),
        "n_games": int(df["game_id"].n_unique()),
        "n_rows": int(df.height),
        "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def assert_can_score(stamp: dict | None, df: pl.DataFrame, what: str) -> None:
    """Raise if a model stamped with `stamp` is asked to score seasons it was fitted on.

    Use it wherever a number is reported as out-of-sample.
    """
    if stamp is None:
        raise LeakageError(f"{what}: the model has no fit stamp, so its training data is unknown")
    overlap = set(stamp["seasons"]) & seasons_in(df)
    if overlap:
        raise LeakageError(
            f"{what}: the model was fitted on seasons {sorted(overlap)} and cannot be "
            "scored on them as out-of-sample"
        )


# ---- the physical barrier: development and test plays in separate files ----


def dev_path() -> Path:
    return DATA / "processed" / "plays_dev.parquet"


def test_path() -> Path:
    return DATA / "processed" / "plays_test.parquet"


def write_split_files(plays: pl.DataFrame) -> tuple[Path, Path]:
    """Write the development and test files from the full state table."""
    known = set(dev_seasons()) | set(test_seasons())
    unknown = seasons_in(plays) - known
    if unknown:
        raise LeakageError(f"seasons {sorted(unknown)} are not assigned to any split")
    dev = plays.filter(pl.col("season").is_in(list(dev_seasons())))
    test = plays.filter(pl.col("season").is_in(list(test_seasons())))
    assert_disjoint_games(dev, test, "development versus test files")
    dev_path().parent.mkdir(parents=True, exist_ok=True)
    dev.write_parquet(dev_path())
    test.write_parquet(test_path())
    return dev_path(), test_path()


def load_dev_plays() -> pl.DataFrame:
    """Plays from the train and calibration seasons. Safe for fitting and tuning."""
    df = pl.read_parquet(dev_path())
    assert_no_test_seasons(df, "development file")
    return df


def load_test_plays(purpose: str) -> pl.DataFrame:
    """Plays from the test seasons, for evaluation only.

    `purpose` must say what the rows are used for. It is appended to
    `results/test_access_log.jsonl`, so every use of the test seasons is on record.
    """
    if not purpose or len(purpose.strip()) < 15:
        raise LeakageError("load_test_plays needs a written purpose (15+ characters)")
    _log_access(kind="load_test_plays", what=purpose, reason=purpose, seasons=test_seasons())
    return pl.read_parquet(test_path())


def _log_access(kind: str, what: str, reason: str, seasons: Iterable[int]) -> None:
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": kind,
        "what": what,
        "reason": reason,
        "seasons": sorted(int(s) for s in seasons),
    }
    with open(ACCESS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
