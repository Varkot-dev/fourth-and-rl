"""Tests for ground_truth_run.py (CPU, no network, no cached data).

The end-to-end test runs every stage on the repo's tiny toy game (fourth_and_rl.rl.toy,
solved in well under a second) with a tiny world model. The toy game is not the game gt_v1
was computed with, so it also exercises the "recompute a smaller ground truth" path.

`test_full_build_from_public_data` is the Colab setup path (download, fit, solve) and runs
only with GT_FULL_BUILD=1 in a fresh checkout whose data/ folder is its own: it downloads
16 seasons and takes minutes.
"""

from __future__ import annotations

import dataclasses
import os
import time

import numpy as np
import pytest

pytest.importorskip("fourth_and_rl")
pytest.importorskip("polars")
pytest.importorskip("torch")

import ground_truth_run as G  # noqa: E402
import run as R  # noqa: E402
import schema as S  # noqa: E402
import synthetic as Y  # noqa: E402


# ---------------------------------------------------------------- pure functions
def test_rollouts_needed_rounds_up_to_power_of_two_tiers():
    n = G.rollouts_needed(np.array([0.005, 0.01, 0.02, 0.04, 0.2, np.nan]), 512, 0.01, 8192)
    # 0.005 and nan need nothing beyond n0; 0.01 needs 1.1 x 512 -> 1024; 0.02 needs
    # 4.4 x 512 -> 5 x 512 = 2560; 0.04 and 0.2 hit the cap.
    assert n.tolist() == [512, 1024, 2560, 8192, 8192, 512]


def test_refine_mask_tops_up_go_and_the_best_kick_and_a_tied_other_kick():
    wp = np.array([[0.5, 0.40, 0.30], [0.5, 0.41, 0.40], [0.5, np.nan, 0.45]])
    se = np.full((3, 3), 0.01)
    allowed = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]], bool)
    m = G.refine_mask(wp, se, allowed)
    assert m.tolist() == [[True, True, False], [True, True, True], [True, False, True]]


def test_combine_is_the_pooled_mean_and_its_standard_error():
    wp, se = G.combine(np.array([[0.4, np.nan]]), np.array([[0.02, np.nan]]), np.array([100]),
                       np.array([[0.7, np.nan]]), np.array([[0.01, np.nan]]), np.array([300]))
    assert wp[0, 0] == pytest.approx((100 * 0.4 + 300 * 0.7) / 400)
    assert se[0, 0] == pytest.approx(np.sqrt(100**2 * 0.02**2 + 300**2 * 0.01**2) / 400)
    assert np.isnan(wp[0, 1])
    # a NaN second batch (masked action) leaves the first estimate unchanged
    wp2, se2 = G.combine(np.array([[0.4]]), np.array([[0.02]]), np.array([100]),
                         np.array([[np.nan]]), np.array([[np.nan]]), np.array([300]))
    assert wp2[0, 0] == pytest.approx(0.4) and se2[0, 0] == pytest.approx(0.02)


def test_budget_is_proportional_smallest_first_with_a_floor():
    sizes = [1000, 4000, 16000, 64000]
    assert G.allocate_budget(sizes, 8500, 0) == pytest.approx(100)
    assert G.allocate_budget(sizes, 8500, 120) == 120
    assert G.allocate_budget(sizes[3:], 600, 120) == pytest.approx(600)
    assert G.plan_epochs(100, 1000, 100, 40) == 8
    assert G.plan_epochs(1, 1000, 100, 40) == 3
    assert G.plan_epochs(1e6, 1000, 100, 40) == 40


def test_state_order_puts_the_headline_first_and_is_a_fixed_permutation():
    kind = np.zeros(10, int)
    kind[7] = 1
    o = G.state_order(kind, 10)
    assert o[0] == 7 and sorted(o.tolist()) == list(range(10))
    assert np.array_equal(G.state_order(kind, 4), o[:4])


def test_training_subset_repeats_small_sets(tmp_path):
    split = R.synthetic_split(n_games=6, seed=0)
    sub = G.training_subset(split, 2, 5)
    assert len(sub.game_ids) == 6 and sub.game_ids[:2] == split.game_ids[:2]
    assert len(G.training_subset(split, 4, 0).game_ids) == 4


def test_score_decisions_by_hand():
    nan = np.nan
    q = np.array([[0.60, 0.50, 0.40],    # go best by 10
                  [0.30, nan, 0.32],     # punt best by 2 (fg not allowed)
                  [0.50, 0.70, 0.45]])   # fg best by 20  # fmt: skip
    allowed = np.isfinite(q)
    wp = np.array([[0.62, 0.55, 0.41], [0.33, nan, 0.31], [0.50, 0.74, 0.40]])
    se = np.full((3, 3), 0.01)
    probs = np.array([[0.2, 0.3, 0.5], [0.1, 0.0, 0.9], [0.0, 1.0, 0.0]])
    out = G.score_decisions(wp, se, q, q, allowed, probs, np.random.default_rng(0), 200)
    assert out["accuracy"]["all"]["value"] == pytest.approx(2 / 3)
    assert out["accuracy"]["true_margin_over_3_points"]["value"] == pytest.approx(1.0)
    assert out["accuracy"]["true_margin_over_3_points"]["n"] == 2
    assert out["regret_points"]["mean"]["value"] == pytest.approx(2 / 3)  # (0 + 2 + 0) / 3
    coach = [(0.60 - (0.2 * .6 + .3 * .5 + .5 * .4)) * 100, (0.32 - (.1 * .3 + .9 * .32)) * 100, 0]
    assert out["coach_policy"]["regret_points_mean"]["value"] == pytest.approx(np.mean(coach))
    assert out["q_star"]["agreement"]["value"] == pytest.approx(2 / 3)
    assert out["calibration"]["by_action"]["FG"]["n"] == 2
    assert out["calibration"]["by_action"]["GO"]["mean_error"] == pytest.approx(
        100 * np.mean([0.02, 0.03, 0.0]))
    cov = {c["bucket"]: c for c in out["calibration"]["by_coach_share"]}
    # under 2%: GO (0.0 error) and PUNT (-5 points) in the third state
    assert cov["under 2%"]["n"] == 2 and cov["under 2%"]["mean_error"] == pytest.approx(-2.5)
    assert out["per_state"]["chosen"] == ["GO", "GO", "FG"]


def test_build_refuses_a_data_folder_linked_elsewhere(monkeypatch, tmp_path):
    from fourth_and_rl import paths

    link = tmp_path / "data"
    link.symlink_to(tmp_path)
    monkeypatch.setattr(paths, "DATA", link)
    with pytest.raises(RuntimeError, match="refusing to build"):
        G.build_game()


# ---------------------------------------------------------------- end to end on the toy game
def _toy_game_dir(tmp_path):
    import joblib

    from fourth_and_rl import mdp_game as mg
    from fourth_and_rl.rl import toy

    sol = toy.toy_game(0)
    d = tmp_path / "toy_game"
    mg.save_solution(sol, d)
    joblib.dump({"coach": toy.toy_coach_table(sol.grid)}, d / "tables.joblib")
    return d


TOY_OVERRIDES = {
    "model": {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32, "dec_hidden": 16,
              "max_len": 64, "dropout": 0.0},
    "train": {"batch_games": 8, "epochs": 2, "patience": 1, "warmup_steps": 2,
              "epoch_games": 40, "sim_batch": 64, "sim_max_live": 256},
    "val_games": {"early_stopping": 16, "report": 16},
    "generate": {"chunk_games": 1000, "workers": 1},
    "eval": {"n_states": 10, "n0": 8, "n_max": 16, "chunk_states": 5, "bootstrap": 50,
             "nll_games": 16},
    "gt_recompute": {"n_states": 10, "n_rollouts": 200, "n_pool_games": 300,
                     "strata": [[30, 60, 0.5], [0, 30, 0.5]], "headline": False,
                     "max_abs_diff": 16},
    "min_train_s": 1, "reserve_package_s": 5,
}  # fmt: skip


def _plan_toy_last_downs(monkeypatch):
    """The toy field has three downs, and plan_fourth_downs plans down-4 records only: plan
    the toy's last downs by relabelling the query records' down as 4 (plumbing test only)."""
    import simulate

    real = simulate.plan_fourth_downs

    def plan(model, split, idx, *a, **k):
        st = split.states.copy()
        st[np.asarray(idx), S.STATE_INDEX["down"]] = 4
        return real(model, dataclasses.replace(split, states=st), idx, *a, **k)

    monkeypatch.setattr(simulate, "plan_fourth_downs", plan)


def test_end_to_end_on_the_toy_game(tmp_path, monkeypatch):
    _plan_toy_last_downs(monkeypatch)
    game = _toy_game_dir(tmp_path)
    wd = tmp_path / "work"
    run = G.prepare("smoke", sizes=[24, 48], workdir=wd, budget_min=10, game_dir=game,
                    benchmark=False, overrides=TOY_OVERRIDES, quiet=True)  # fmt: skip
    out = G.run_all(run)
    for s in G.STAGES:
        assert out[s]["status"] == "done", (s, out[s])
    setup = out["setup"]
    assert setup["ground_truth"]["recomputed"] and not setup["verify"]["ok"]
    assert (wd / "ground_truth" / "gt_colab.npz").exists()
    # every size is a prefix of the largest
    tr, _ = R.load_split(run.data_path(G.TRAIN_FILE))
    assert len(tr.game_ids) == 48
    res = R.read_json(wd / G.RESULTS_NAME)
    rows = res["scaling_table"]
    assert [r["games"] for r in rows] == [24, 48]
    for r in rows:
        assert r["n_states"] == 10
        assert 0.0 <= r["accuracy"]["value"] <= 1.0
        assert r["regret_mean_points"]["value"] >= 0.0
    for f in ("gt_regret_accuracy.png", "gt_q_scatter.png", "gt_overestimation_by_coverage.png"):
        assert (wd / "figures" / f).exists(), res["figures"]
    assert (wd / G.ZIP_NAME).exists()
    assert "picks the truly best action" in G.summary(run)
    # a second run skips every finished stage and reuses the per-size results
    t0 = time.time()
    run2 = G.prepare("smoke", sizes=[24, 48], workdir=wd, budget_min=10, game_dir=game,
                     benchmark=False, overrides=TOY_OVERRIDES, quiet=True)  # fmt: skip
    assert all(run2.is_done(s) for s in G.STAGES[:-1])
    G.run_all(run2)
    assert time.time() - t0 < 30
    # adding a size keeps the trained and evaluated sizes
    run3 = G.prepare("smoke", sizes=[24, 48, 64], workdir=wd, budget_min=10, game_dir=game,
                     benchmark=False, overrides=TOY_OVERRIDES, quiet=True)  # fmt: skip
    assert G.size_trained(run3, 24) and G.eval_done(run3, 48)
    assert not run3.is_done("generate")  # 64 games are needed now


@pytest.mark.skipif(not os.environ.get("GT_FULL_BUILD"),
                    reason="downloads 16 seasons and solves the game; set GT_FULL_BUILD=1")
def test_full_build_from_public_data():
    """The Colab setup path: build the exact game from public data and check it against the
    saved ground truth. Run in a fresh checkout (its data/ folder must be its own)."""
    from fourth_and_rl import mdp_game as mg

    t0 = time.time()
    info = G.build_game()
    sol = mg.load_solution(info["solution_dir"])
    v = Y.verify_ground_truth(sol.meta)
    print({"build": info, "verify": v, "seconds": round(time.time() - t0, 1)})
    assert mg.solution_exists(info["solution_dir"])
    assert info["n_states"]["total"] == 407_854_656  # docs/notes/m4_mdp.md


def test_game_cache_is_skipped_when_the_drive_is_full(tmp_path, monkeypatch):
    """A full Drive must not stop the run: the cache only saves rebuild time later."""
    import shutil
    import types

    src = tmp_path / "game"
    src.mkdir()
    (src / "values.npy").write_bytes(b"x" * 4096)
    logs: list[str] = []
    run = types.SimpleNamespace(workdir=tmp_path, log=logs.append)
    cache = tmp_path / "cache" / "game"
    full = shutil.disk_usage(tmp_path)._replace(free=1024)
    monkeypatch.setattr(G.shutil, "disk_usage", lambda p: full)
    out = G._try_cache_game(run, src, cache)
    assert "cache_skipped" in out and not cache.exists() and "not caching" in logs[-1]
    monkeypatch.undo()

    def broken(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(G, "copy_game", broken)
    out = G._try_cache_game(run, src, cache)
    assert "failed" in out["cache_skipped"] and not cache.exists()
    monkeypatch.undo()
    out = G._try_cache_game(run, src, cache)
    assert "cache_seconds" in out and (cache / "values.npy").exists()
