"""Tests for the day 6 grid of R1 (`fourth_and_rl.overopt_grid`) and the solved-game setup
of its notebook (`fourth_and_rl.game_setup`): plan order, parts, resume, the RAM-based
parallelism, the scheduler, the collect merge and the Drive cache decision. None of them
needs the solved game."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fourth_and_rl import game_setup as gs
from fourth_and_rl import overopt as oo
from fourth_and_rl import overopt_grid as og

CFG = oo.load_config()
GRID = CFG["grid"]


# --------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------


def test_keys_round_trip_and_include_the_seed() -> None:
    assert og.cell_key(250, "large", 101) == "n250_large_s101"
    assert og.parse_key("n64000_small_s105") == (64000, "small", 105)
    assert og.parse_key("oracle") is None
    with pytest.raises(ValueError):
        og.parse_key("1k_large")


def test_plan_runs_the_turnover_region_first_seed_by_seed() -> None:
    plan = og.plan_from_config(GRID)
    keys = [c.key for c in plan]
    assert len(plan) == 3 * 2 * 5 + 1 + 2 * 2 * 5 + 1 * 2 * 5
    # Whole replicates first: every size and capacity of block 1 for seed 101.
    assert keys[:6] == ["n250_small_s101", "n250_large_s101", "n500_small_s101",
                        "n500_large_s101", "n1000_small_s101", "n1000_large_s101"]  # fmt: skip
    assert keys[6] == "n250_small_s102"
    first = keys[:30]
    assert {og.parse_key(k)[0] for k in first} == {250, 500, 1000}
    assert {og.parse_key(k)[2] for k in first} == {101, 102, 103, 104, 105}
    assert keys[30] == "oracle"
    assert {og.parse_key(k)[0] for k in keys[31:51]} == {4000, 16000}
    assert {og.parse_key(k)[0] for k in keys[51:]} == {64000}
    assert [c.block for c in plan] == sorted(c.block for c in plan)


def test_plan_parameters_narrow_the_grid() -> None:
    plan = og.plan_from_config(GRID, sizes=[250], capacities=["large"], seeds=[101],
                               include_oracle=False)  # fmt: skip
    assert [c.key for c in plan] == ["n250_large_s101"]
    # A size in no configured block becomes a last block of its own.
    plan = og.plan_from_config(GRID, sizes=[16000, 2000], seeds=[7])
    assert [c.key for c in plan] == ["n16000_small_s7", "n16000_large_s7", "oracle",
                                     "n2000_small_s7", "n2000_large_s7"]  # fmt: skip
    with pytest.raises(ValueError):
        og.plan_cells([[250, 250]], ["small"], [1])


def test_parts_are_disjoint_ordered_and_cover_the_plan() -> None:
    plan = og.plan_from_config(GRID)
    parts = [og.select_parts(plan, [f"{k}/3"]) for k in (1, 2, 3)]
    keys = [c.key for p in parts for c in p]
    assert sorted(keys) == sorted(c.key for c in plan) and len(set(keys)) == len(keys)
    order = {c.key: i for i, c in enumerate(plan)}
    for p in parts:
        assert [order[c.key] for c in p] == sorted(order[c.key] for c in p)
    two = og.select_parts(plan, ["1/3", "2/3"])
    assert {c.key for c in two} == {c.key for c in parts[0] + parts[1]}
    assert og.select_parts(plan, None) == plan
    with pytest.raises(ValueError):
        og.select_parts(plan, ["1/2", "1/3"])
    with pytest.raises(ValueError):
        og.parse_part("4/3")


def test_block_hours() -> None:
    plan = og.plan_from_config(GRID)
    rows = og.block_hours(plan, plan[10:], n_parallel=2, seconds_per_cell=900)
    assert [r["cells"] for r in rows] == [31, 20, 10]
    assert rows[0]["todo"] == 21 and rows[0]["label"].endswith("+ oracle")
    assert rows[1]["hours"] == pytest.approx(20 * 900 / 2 / 3600)


# --------------------------------------------------------------------------------------
# Identity and resume
# --------------------------------------------------------------------------------------


def test_game_key_ignores_fit_times_but_not_data() -> None:
    meta = {"stamp": {"a": 1, "fitted_at": "x"}, "tables_key": {"h": "1"}, "seed": 3}
    later = json.loads(json.dumps(meta))
    later["stamp"]["fitted_at"] = "y"
    assert og.game_key(meta) == og.game_key(later)
    later["tables_key"]["h"] = "2"
    assert og.game_key(meta) != og.game_key(later)


def test_settings_key_tracks_betas_and_game() -> None:
    c = og.plan_from_config(GRID)[0]
    base = og.settings_key(og.cell_settings(CFG, c, None, "g1"))
    assert og.settings_key(og.cell_settings(CFG, c, CFG["betas"], "g1")) == base
    assert og.settings_key(og.cell_settings(CFG, c, [10, 100], "g1")) != base
    assert og.settings_key(og.cell_settings(CFG, c, None, "g2")) != base
    other = og.plan_from_config(GRID)[1]
    assert og.settings_key(og.cell_settings(CFG, other, None, "g1")) != base


def _rows(gains: list[float], betas: list[float] | None = None) -> list[dict]:
    betas = betas or [10.0, 100.0, 1000.0][: len(gains)]
    return [{"beta": b, "true_gain": g, "sqrt_kl": 0.5 * i + 0.1, "claimed_gain": 2 * g}
            for i, (b, g) in enumerate(zip(betas, gains, strict=True))]  # fmt: skip


def _record(key: str, gains: list[float], game: str = "g1", skey: str | None = None,
            rmse: float = 0.05) -> dict:  # fmt: skip
    p = og.parse_key(key)
    settings = {"game_key": game, "proxy_params": {"num_leaves": 8 if p and p[1] == "small"
                                                   else 64}}  # fmt: skip
    return {"rows": _rows(gains), "diagnostics": {"value_rmse": rmse,
                                                  "greedy_uniform": {"disagree": 0.3}},
            "timing": {"total_seconds": 800.0, "prep_seconds": 5.0},
            "grid": {"key": key, "settings": settings, "settings_key": skey or f"s-{game}"}}


def _write(d: Path, rec: dict) -> Path:
    return og.write_atomic(rec, d / f"{rec['grid']['key']}.json")


def test_resume_skips_only_cells_done_with_the_same_settings(tmp_path: Path) -> None:
    plan = og.plan_from_config(GRID, sizes=[250], seeds=[101, 102], include_oracle=False)
    a, b = tmp_path / "here", tmp_path / "colab"
    _write(a, _record("n250_small_s101", [0.01, 0.02], skey="K"))
    _write(a, _record("n250_large_s101", [0.01, 0.02], skey="OLD"))  # other settings
    _write(b, _record("n250_small_s102", [0.01, 0.02], skey="K"))
    (a / "n250_large_s102.json").write_text("{not json")  # a half-written file
    todo, done = og.split_done(plan, [a], lambda c: "K")
    assert set(done) == {"n250_small_s101"}
    assert [c.key for c in todo] == ["n250_large_s101", "n250_small_s102", "n250_large_s102"]
    todo, done = og.split_done(plan, [a, b], lambda c: "K")
    assert set(done) == {"n250_small_s101", "n250_small_s102"}
    assert not list(a.glob(".*tmp*"))  # atomic writes leave no temporary files


# --------------------------------------------------------------------------------------
# Parallelism
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cores", "avail", "total", "n", "threads", "limit", "warn"),
    [
        (4, 15.1, 15.7, 1, 3, "RAM", False),   # this machine: 15 GB fits one 8 GB cell
        (8, 49.0, 51.0, 2, 3, "cores", False),  # Colab high-RAM CPU: cores bind first
        (2, 11.5, 12.7, 1, 2, "RAM", False),   # Colab standard CPU: one cell, 2 threads
        (2, 7.0, 8.0, 1, 2, "RAM", True),      # under 12 GB and one cell does not fit
        (32, 20.0, 21.0, 2, 3, "RAM", False),  # many cores, little RAM
    ],
)  # fmt: skip
def test_parallelism_never_exceeds_ram(cores, avail, total, n, threads, limit, warn) -> None:
    d = og.choose_parallel(cores, avail, GRID["mem_gb_per_cell"], GRID["threads_per_cell"],
                           GRID["reserve_gb"], ram_gb=total)  # fmt: skip
    assert (d["n_parallel"], d["threads_per_cell"], d["limited_by"]) == (n, threads, limit)
    assert bool(d["warnings"]) == warn
    if d["by_ram"] >= 1:
        assert d["n_parallel"] * GRID["mem_gb_per_cell"] <= avail - GRID["reserve_gb"]


def test_parallelism_cap_and_low_ram_warning() -> None:
    d = og.choose_parallel(16, 100.0, 8.0, 3, 2.0, max_parallel=1, ram_gb=104.0)
    assert d["n_parallel"] == 1 and d["limited_by"] == "max_parallel"
    d = og.choose_parallel(4, 10.0, 8.0, 3, 2.0, ram_gb=11.0)
    assert d["n_parallel"] == 1 and any("under 12 GB" in w for w in d["warnings"])
    res = og.system_resources()
    assert res["cores"] >= 1 and res["ram_gb"] > 0


# --------------------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class FakeProc:
    """Finishes `runtime` seconds after it starts, with return code `rc`."""

    def __init__(self, clock: FakeClock, runtime: float, rc: int, on_done=None) -> None:
        self.clock, self.end, self.rc, self.on_done = clock, clock.t + runtime, rc, on_done
        self.terminated = False

    def poll(self) -> int | None:
        if self.clock.t < self.end:
            return None
        if self.on_done:
            self.on_done()
            self.on_done = None
        return self.rc

    def terminate(self) -> None:
        self.terminated = True


def _cells(n: int) -> list[og.Cell]:
    return og.plan_from_config(GRID, sizes=[250, 500, 1000], include_oracle=False)[:n]


def test_scheduler_runs_in_order_in_parallel_and_stops_at_the_deadline() -> None:
    clock, finished, starts, live = FakeClock(), set(), [], []

    def launch(c: og.Cell, threads: int) -> FakeProc:
        starts.append((c.key, clock.t, threads))
        live.append(c.key)
        p = FakeProc(clock, 100.0, 0, lambda: (finished.add(c.key), live.remove(c.key)))
        assert len(live) <= 2
        return p

    cells = _cells(8)
    out = og.run_grid(cells, launch, n_parallel=2, threads=3, deadline=350.0,
                      seconds_per_cell=100.0, is_finished=lambda c: c.key in finished,
                      log=lambda m: None, poll=10.0, clock=clock, sleep=clock.sleep)  # fmt: skip
    assert [s[0] for s in starts] == [c.key for c in cells[: len(starts)]]
    assert all(s[2] == 3 for s in starts)
    # 0-100: cells 1-2; 100-200: 3-4 (estimate 120 s from then on); 200-300: 5-6. At 300 the
    # estimate (1.2 x 100 s) no longer fits before 350, so 7 and 8 are not started.
    assert out["done"] == [c.key for c in cells[:6]]
    assert out["not_started"] == [c.key for c in cells[6:]]
    assert out["stopped"].startswith("deadline")


def test_scheduler_retries_a_killed_cell_alone_and_records_failures() -> None:
    clock, finished, calls = FakeClock(), set(), {}

    def launch(c: og.Cell, threads: int) -> FakeProc:
        calls[c.key] = calls.get(c.key, 0) + 1
        if c.key == "n250_small_s101" and calls[c.key] == 1:
            return FakeProc(clock, 50.0, -9)  # killed (out of memory)
        if c.key == "n250_large_s101":
            return FakeProc(clock, 50.0, 1)  # a bug: not retried
        return FakeProc(clock, 50.0, 0, lambda: finished.add(c.key))

    out = og.run_grid(_cells(3), launch, n_parallel=2, threads=3, deadline=1e9,
                      seconds_per_cell=50.0, is_finished=lambda c: c.key in finished,
                      log=lambda m: None, poll=10.0, clock=clock, sleep=clock.sleep)  # fmt: skip
    assert calls["n250_small_s101"] == 2
    assert out["failed"] == ["n250_large_s101"]
    assert set(out["done"]) == {"n250_small_s101", "n500_small_s101"}
    assert out["n_parallel_end"] == 1


def test_scheduler_needs_a_complete_result_file() -> None:
    clock = FakeClock()
    out = og.run_grid(_cells(1), lambda c, t: FakeProc(clock, 10.0, 0), n_parallel=1,
                      threads=1, deadline=1e9, seconds_per_cell=10.0,
                      is_finished=lambda c: False, log=lambda m: None, poll=5.0,
                      clock=clock, sleep=clock.sleep)  # fmt: skip
    assert out["done"] == [] and out["failed"] == ["n250_small_s101"]


# --------------------------------------------------------------------------------------
# Collect
# --------------------------------------------------------------------------------------


def test_cell_summary() -> None:
    s = og.cell_summary(_rows([0.010, 0.020, 0.015]))
    assert s["peak_beta"] == 100.0 and s["final_beta"] == 1000.0
    assert s["peak_gain_points"] == pytest.approx(2.0)
    assert s["final_gain_points"] == pytest.approx(1.5)
    assert s["drop_points"] == pytest.approx(0.5)
    assert og.cell_summary(_rows([0.01, 0.02, 0.03]))["drop_points"] == 0.0


def test_collect_merges_folders_and_tests_the_drop_against_two_sd(tmp_path: Path) -> None:
    here, colab = tmp_path / "here", tmp_path / "colab"
    # A clear turnover for the large proxies (drops 0.50, 0.55, 0.45 points: SD 0.05) ...
    _write(here, _record("n250_large_s101", [0.010, 0.020, 0.0150]))
    _write(colab, _record("n250_large_s102", [0.010, 0.021, 0.0155]))
    _write(colab, _record("n250_large_s103", [0.010, 0.019, 0.0145]))
    # ... a noisy plateau for the small ones (drops 0.1, 0.0, 0.3: mean 0.13 < 2 SD).
    _write(here, _record("n250_small_s101", [0.010, 0.020, 0.0190]))
    _write(here, _record("n250_small_s102", [0.010, 0.020, 0.0210]))
    _write(colab, _record("n250_small_s103", [0.010, 0.020, 0.0170]))
    _write(colab, _record("n250_small_s101", [0.010, 0.020, 0.0190]))  # duplicate
    _write(colab, _record("n250_small_s102", [0.010, 0.020, 0.0300]))  # conflict
    _write(colab, _record("n500_small_s101", [0.010, 0.020, 0.0190], game="g2"))
    _write(colab, _record("oracle", [0.02, 0.04, 0.045]))
    (colab / "notes.json").write_text(json.dumps({"not": "a cell"}))
    plan = og.plan_from_config(GRID, sizes=[250], seeds=[101, 102, 103])
    out = og.collect([here, colab], tmp_path / "grid.json", tmp_path / "grid.png", plan=plan,
                     reference_game="g1")  # fmt: skip
    m = out["merge"]
    assert out["game_key"] == "g1" and out["same_game_as_reference"]
    assert [d["key"] for d in m["duplicates"]] == ["n250_small_s101"]
    assert [d["key"] for d in m["conflicts"]] == ["n250_small_s102"]
    assert m["conflicts"][0]["kept"].startswith(str(here))
    assert m["excluded_other_game"] == ["n500_small_s101"]
    assert out["plan"]["done"] == len(plan) and out["plan"]["missing"] == []
    g = {(x["n_games"], x["capacity"]): x for x in out["groups"]}
    big, small = g[(250, "large")], g[(250, "small")]
    assert big["drop_points"] == pytest.approx([0.5, 0.55, 0.45])
    assert big["drop_sd_points"] == pytest.approx(0.05)
    assert big["drop_exceeds_2sd"] is True
    assert small["drop_points"] == pytest.approx([0.1, 0.0, 0.3])  # the first folder's s102
    assert small["drop_exceeds_2sd"] is False
    assert big["gain_mean_points"][1] == pytest.approx(2.0)
    assert out["cells"]["n250_large_s101"]["leaves"] == 64
    assert out["cells"]["oracle"]["peak_beta"] == 1000.0
    assert json.loads((tmp_path / "grid.json").read_text())["groups"] == out["groups"]
    assert (tmp_path / "grid.png").stat().st_size > 10_000


def test_collect_with_one_seed_has_no_sd(tmp_path: Path) -> None:
    _write(tmp_path / "c", _record("n250_large_s101", [0.010, 0.020, 0.0150]))
    out = og.collect([tmp_path / "c"], tmp_path / "o.json", None)
    gr = out["groups"][0]
    assert gr["drop_sd_points"] is None and gr["drop_exceeds_2sd"] is None
    assert not (tmp_path / "o.png").exists()


def test_day5_reproduction_compares_same_game_cells_only(tmp_path: Path) -> None:
    day5 = tmp_path / "day5"
    day5.mkdir()
    (day5 / "1k_large_b.json").write_text(json.dumps({"rows": _rows([0.01, 0.02, 0.015])}))
    cells = {"n1000_large_s102": _record("n1000_large_s102", [0.01, 0.02, 0.0150001]),
             "n1000_small_s101": _record("n1000_small_s101", [0.01, 0.02, 0.015], game="g2")}
    out = og.day5_reproduction(cells, day5, "g1")
    assert len(out) == 1 and out[0]["day5_cell"] == "1k_large_b"
    assert out[0]["max_abs_true_gain_diff"] == pytest.approx(1e-7)


def test_make_zip(tmp_path: Path) -> None:
    _write(tmp_path / "cells", _record("n250_large_s101", [0.01]))
    (tmp_path / "log.txt").write_text("x")
    z = og.make_zip(tmp_path, "r.zip", [tmp_path / "cells", tmp_path / "log.txt"])
    import zipfile

    assert sorted(zipfile.ZipFile(z).namelist()) == ["cells/n250_large_s101.json", "log.txt"]


# --------------------------------------------------------------------------------------
# The solved game on a fresh machine
# --------------------------------------------------------------------------------------


def _fake_game(d: Path) -> Path:
    d.mkdir(parents=True)
    for name in ("model.joblib", "value_end.npy", "tables.joblib"):
        (d / name).write_bytes(b"x" * 1000)
    (d / "meta.json").write_text(json.dumps({"stamp": {}, "tables_key": {}, "seed": 1}))
    return d


def test_cache_is_skipped_when_drive_lacks_space(tmp_path: Path, monkeypatch) -> None:
    ok, msg = gs.cache_decision(2 * 2**30, 1 * 2**30)
    assert not ok and "not caching" in msg
    assert gs.cache_decision(2 * 2**30, 10 * 2**30)[0]
    assert gs.cache_decision(2 * 2**30, None)[0]
    src = _fake_game(tmp_path / "mdp")
    monkeypatch.setattr(gs.shutil, "disk_usage", lambda p: type("U", (), {"free": 10})())
    out = gs.try_cache(src, tmp_path / "drive" / "game", log=lambda m: None)
    assert "cache_skipped" in out and not (tmp_path / "drive" / "game").exists()


def test_ensure_game_restores_from_the_cache(tmp_path: Path, monkeypatch) -> None:
    from fourth_and_rl import mdp_game as mg

    cache = _fake_game(tmp_path / "drive" / "game")
    target = tmp_path / "repo" / "data" / "mdp"
    monkeypatch.setattr(mg, "SOLUTION_DIR", target)
    monkeypatch.setattr(gs, "build_game", lambda log: pytest.fail("must not rebuild"))
    out = gs.ensure_game(None, cache, log=lambda m: None)
    assert out["source"].startswith("restored") and gs.has_game(target)
    assert og.game_key_of_dir(target) == og.game_key_of_dir(cache)
    # Present now: reused, not copied again; a given folder is used as is.
    assert gs.ensure_game(None, cache, log=lambda m: None)["source"].startswith("already")
    assert gs.ensure_game(cache, None)["game_dir"] == str(cache)
    with pytest.raises(FileNotFoundError):
        gs.ensure_game(tmp_path / "nothing", None)


def test_ensure_game_caches_after_a_rebuild(tmp_path: Path, monkeypatch) -> None:
    from fourth_and_rl import mdp_game as mg

    target = tmp_path / "repo" / "data" / "mdp"
    monkeypatch.setattr(mg, "SOLUTION_DIR", target)
    monkeypatch.setattr(gs, "build_game", lambda log: (_fake_game(target), {"ok": 1})[1])
    out = gs.ensure_game(None, tmp_path / "drive" / "game", log=lambda m: None)
    assert out["source"].startswith("rebuilt") and "cache_seconds" in out
    assert gs.has_game(tmp_path / "drive" / "game")
    out = gs.ensure_game(None, tmp_path / "drive2" / "game", use_cache=False, log=lambda m: None)
    assert not (tmp_path / "drive2").exists()


def test_rebuild_never_downloads_the_holdout() -> None:
    seasons = gs.check_seasons()
    assert 2026 not in seasons and max(seasons) == 2025
    assert np.all(np.diff(seasons) == 1)
