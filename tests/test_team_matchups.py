"""Tests for the notebook's run module (`team_matchups.py`). No data and no JAX: the grid
is built from the committed ratings, and the solver is replaced by a stand-in so the
deadline guard and resume logic run in milliseconds."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fourth_and_rl import mdp, splits
from fourth_and_rl import mdp_game as mg
from fourth_and_rl import team_matchups as tm
from fourth_and_rl import team_strength as ts

polars = pytest.importorskip("polars")


@pytest.fixture(scope="module")
def ratings() -> list[dict[str, Any]]:
    return tm.season_end_ratings()


def test_the_holdout_is_refused() -> None:
    with pytest.raises(splits.LeakageError, match="holdout"):
        tm.check_no_holdout([2024, 2025, 2026])
    tm.check_no_holdout(range(2010, 2026))


def test_grid(ratings: list[dict[str, Any]]) -> None:
    specs = tm.matchup_grid(ratings)
    keys = [s["key"] for s in specs]
    assert len(keys) == len(set(keys)) == 5 + 32 + 10
    assert keys[0] == "identical" and [s["group"] for s in specs[:5]] == ["q6"] * 5
    vs = [s for s in specs if s["group"] == "vs_def"]
    assert sorted(s["defense_rank"] for s in vs) == list(range(1, 33))
    # Rank 1 is the defense that allows the fewest points (lowest drive rating).
    best = min(ratings, key=lambda r: r["drive_def"])["team"]
    assert next(s for s in vs if s["defense_rank"] == 1)["opponent"] == best
    # The q6 groups are exactly team_strength.group_ratings over 2025.
    g = ts.group_ratings(polars.DataFrame(ratings), [tm.SEASON], k=5)
    q6 = {s["key"]: s for s in specs[:5]}
    assert q6["det_vs_top5_def"]["b_def"] == pytest.approx(g["best"])
    assert q6["det_vs_bottom5_def"]["b_def"] == pytest.approx(g["worst"])
    assert q6["bottom5_off_vs_top5_def"]["a_off"] == pytest.approx(g["bottom"])
    offs = [s for s in specs if s["group"] in ("top5_off", "bottom5_off")]
    assert sorted(s["offense_rank"] for s in offs) == [1, 2, 3, 4, 5, 28, 29, 30, 31, 32]


def test_cache_key_changes_with_the_inputs(ratings: list[dict[str, Any]]) -> None:
    s = tm.matchup_grid(ratings)[1]
    assert tm.spec_key(s, "a") == tm.spec_key(dict(s), "a") != tm.spec_key(s, "b")


class FakeSolver:
    """Stands in for `mdp_team_jax.solve_end_phases`: action values that make going win by
    a fixed margin everywhere."""

    def __init__(self, model: Any, seconds: float = 0.0) -> None:
        self.model, self.seconds, self.calls = model, seconds, 0

    def __call__(self, pairs: list, **kw: Any) -> list[Any]:
        self.calls += 1
        time.sleep(self.seconds)
        g, m = self.model.kern.grid, self.model
        shape = (2, g.nf, 2 * m.score_max + 1, 4, 4)
        q = {c: np.full(shape, 0.5, dtype=np.float32) for c in (*tm.PLAY_CODES, mg.A_FG)}
        q[mg.A_PUNT] = np.full(shape, 0.45, dtype=np.float32)
        tick = kw["q_ticks"][0]
        return [SimpleNamespace(q={tick: q}, meta={"backend": "fake"}) for _ in pairs]


def fake_inputs() -> Any:
    cfg = mdp.load_config()
    grid = mdp.grid_from_config(cfg)
    model = SimpleNamespace(kern=SimpleNamespace(grid=grid), score_max=cfg["score_max"],
                            clock=SimpleNamespace(tick_seconds=5))  # fmt: skip
    return SimpleNamespace(base=model, provenance={"dev_fingerprint": "test"})


def test_solve_grid_resumes_and_respects_the_deadline(
    tmp_path: Path, ratings: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fourth_and_rl import mdp_team_jax as mj

    inputs = fake_inputs()
    fake = FakeSolver(inputs.base)
    monkeypatch.setattr(mj, "solve_end_phases", fake)
    monkeypatch.setattr(tm.Kernels, "pair", lambda self, sp: (inputs.base, inputs.base, {}))
    specs = tm.matchup_grid(ratings)[:5]
    # No time at all: nothing starts.
    st = tm.solve_grid(inputs, specs, tmp_path, time.time(), log=lambda m: None)
    assert st["done"] == [] and "deadline" in st["stopped"] and fake.calls == 0
    # Enough time: three solved, two at a time.
    st = tm.solve_grid(inputs, specs[:3], tmp_path, time.time() + 600, batch_size=2,
                       reserve_seconds=0, log=lambda m: None)  # fmt: skip
    assert len(st["done"]) == 3 and st["stopped"] is None and fake.calls == 2
    rec = tm.collect(tmp_path, specs)
    assert len(rec) == 3
    assert rec[0]["headline"]["go_minus_punt_points"] == pytest.approx(5.0, abs=1e-4)
    assert rec[0]["go_cells"] == rec[0]["n_cells"] == 310
    # Rerun: the three are skipped, only the other two are solved.
    st = tm.solve_grid(inputs, specs, tmp_path, time.time() + 600, batch_size=2,
                       reserve_seconds=0, log=lambda m: None)  # fmt: skip
    assert len(st["done"]) == 5 and fake.calls == 3
    # A batch slower than the time left is not started.
    slow = FakeSolver(inputs.base, seconds=0.3)
    monkeypatch.setattr(mj, "solve_end_phases", slow)
    more = tm.matchup_grid(ratings)[5:9]
    st = tm.solve_grid(inputs, more, tmp_path, time.time() + 0.5, batch_size=1,
                       reserve_seconds=0, first_guess_seconds=0.01, log=lambda m: None)  # fmt: skip
    assert 1 <= slow.calls < 4 and "deadline" in st["stopped"]
