from __future__ import annotations

import polars as pl
import pytest

from fourth_and_rl import splits


def _frame(seasons: list[int]) -> pl.DataFrame:
    games = [f"{s}_g{i}" for i, s in enumerate(seasons)]
    return pl.DataFrame({"season": seasons, "game_id": games})


@pytest.fixture(autouse=True)
def _tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "ACCESS_LOG", tmp_path / "log.jsonl")


def test_splits_do_not_overlap_and_test_is_the_latest():
    dev, test = set(splits.dev_seasons()), set(splits.test_seasons())
    assert not dev & test
    assert min(test) > max(dev)


def test_guard_passes_development_seasons():
    splits.assert_no_test_seasons(_frame([2010, 2023]), "toy fit")


def test_guard_raises_on_test_seasons():
    with pytest.raises(splits.LeakageError):
        splits.assert_no_test_seasons(_frame([2021, 2024]), "toy fit")


def test_exception_needs_a_written_reason_and_is_logged():
    df = _frame([2025])
    with pytest.raises(splits.LeakageError):
        splits.assert_no_test_seasons(df, "toy fit", allow_test_seasons=True)
    with pytest.raises(splits.LeakageError):
        splits.assert_no_test_seasons(df, "toy fit", allow_test_seasons=True, reason="ok")
    splits.assert_no_test_seasons(
        df, "toy fit", allow_test_seasons=True, reason="simulator only, nothing scored on test"
    )
    assert "simulator only" in splits.ACCESS_LOG.read_text()


def test_disjoint_games_guard():
    a, b = _frame([2010, 2011]), _frame([2012])
    splits.assert_disjoint_games(a, b, "toy split")
    with pytest.raises(splits.LeakageError):
        splits.assert_disjoint_games(a, a, "toy split")


def test_stamp_blocks_scoring_seasons_the_model_saw():
    stamp = splits.fit_stamp(_frame([2010, 2011]), "toy model")
    splits.assert_can_score(stamp, _frame([2024]), "toy evaluation")
    with pytest.raises(splits.LeakageError):
        splits.assert_can_score(stamp, _frame([2011, 2024]), "toy evaluation")
    with pytest.raises(splits.LeakageError):
        splits.assert_can_score(None, _frame([2024]), "toy evaluation")


def test_test_plays_need_a_purpose():
    with pytest.raises(splits.LeakageError):
        splits.load_test_plays("")


@pytest.mark.data
def test_split_files_are_separate_on_disk(plays):
    splits.write_split_files(plays)
    dev = splits.load_dev_plays()
    test = splits.load_test_plays("unit test of the split files on disk")
    assert splits.seasons_in(dev) == set(splits.dev_seasons())
    assert splits.seasons_in(test) == set(splits.test_seasons())
    splits.assert_disjoint_games(dev, test, "split files")
    assert dev.height + test.height == plays.height
