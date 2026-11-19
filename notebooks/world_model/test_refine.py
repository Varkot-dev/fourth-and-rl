"""Tests for the close-call refinement of the fourth-down plan (run.refine_close_calls and its
helpers) and its wiring into run.stage_plan. CPU, toy data, no network."""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import pytest
import run as R
import schema as S
import support as SP
import torch
from model import WorldModel
from simulate import plan_fourth_downs
from test_run import _stubs, _tiny_preset
from toy import make_toy_split

torch.set_num_threads(1)
SI = S.STATE_INDEX
FAR = time.time() + 3600.0
TINY = {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64, "dec_hidden": 32,
        "dropout": 0.0, "max_len": 48, "max_steps": 150, "sim_batch": 512, "seed": 0}


# ---------------------------------------------------------------- helpers
def test_close_call_rollouts():
    # margin 0.02 with se 0.02: needs (2 * 0.02 / 0.02)^2 = 4 times n0, plus 10% -> 5 * n0.
    t = R.close_call_rollouts(np.array([0.02, 0.0, 0.5, -0.01]),
                              np.array([0.02, 0.01, 0.01, 0.02]), 100, 1000)
    assert t.tolist() == [500, 1000, 200, 1000]  # zero margin: the cap; easy: at least 2 n0
    assert (t % 100 == 0).all()
    assert R.close_call_rollouts(np.array([0.001]), np.array([0.05]), 512, 4096)[0] == 4096


def test_combine_rollouts_pools_like_one_batch():
    rng = np.random.default_rng(0)
    a, b = rng.random(300) < 0.6, rng.random(900) < 0.6
    se = lambda x: x.std(ddof=0) / np.sqrt(len(x))  # noqa: E731
    wp, s = R.combine_rollouts(np.array([[a.mean()]]), np.array([[se(a)]]), np.array([[300]]),
                               np.array([[b.mean()]]), np.array([[se(b)]]), np.array([[900]]))
    both = np.r_[a, b]
    assert wp[0, 0] == pytest.approx(both.mean())
    assert s[0, 0] == pytest.approx(se(both), rel=0.05)
    # No second batch (n 0 or NaN): unchanged.
    wp, s = R.combine_rollouts(np.array([[0.4, 0.5]]), np.array([[0.02, 0.03]]),
                               np.array([[100, 100]]), np.array([[np.nan, 0.9]]),
                               np.array([[np.nan, 0.01]]), np.array([[100, 0]]))
    assert wp.tolist() == [[0.4, 0.5]] and s.tolist() == [[0.02, 0.03]]


def test_top_two_mask_and_decide():
    wp = np.array([[0.5, 0.6, 0.55], [0.7, np.nan, 0.2], [0.3, 0.9, 0.1], [0.4, np.nan, np.nan]])
    allowed = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0]], bool)
    m = R.top_two_mask(wp, allowed)
    assert m.tolist() == [[False, True, True], [True, False, True], [True, False, True],
                          [True, False, False]]
    se = np.full_like(wp, 0.01)
    d = R.decide(wp, se, np.ones(4, bool))
    assert list(d["best"][:3]) == ["FG", "GO", "FG"]
    assert d["margin"][0] == pytest.approx(0.05)
    assert not d["too_close"][0] and d["margin_se"][0] == pytest.approx(np.sqrt(2) * 0.01)


def test_decide_matches_plan_fourth_downs():
    split = make_toy_split(12, np.random.default_rng(3))
    torch.manual_seed(0)
    model = WorldModel(TINY)
    idx = SP.fourth_down_records(split)[:4]
    df = plan_fourth_downs(model, split, idx, 16, TINY, FAR, np.random.default_rng(0))
    d = R.decide(df[R._WP_COLS].to_numpy(float), df[R._SE_COLS].to_numpy(float),
                 df["done"].to_numpy(bool))
    assert list(d["best"]) == list(df["best"])
    assert np.allclose(d["margin"], df["margin"]) and np.allclose(d["margin_se"], df["margin_se"])
    assert list(d["too_close"]) == list(df["too_close"])


# ---------------------------------------------------------------- refine_close_calls
TRUE_WP = {0: (0.50, 0.51, 0.30),   # GO vs FG close: resolvable with more rollouts
           1: (0.60, 0.20, 0.40),   # clear after the first pass: never refined
           2: (0.45, 0.10, 0.45),   # a true tie: stays close at the cap
           3: (0.55, 0.53, 0.54)}   # all close; top two by estimate are GO and PUNT


def _fake_plan(calls: list):
    """plan_fn with exact means and se = 0.5 / sqrt(n): deterministic, so the arithmetic of
    the pooling is checkable. Records every (idx, n, mask) request."""

    def plan_fn(idx, n, mask, rng):
        idx = np.asarray(idx)
        calls.append((idx.copy(), int(n), np.asarray(mask).copy()))
        wp = np.array([TRUE_WP[int(i) % 4] for i in idx], dtype=float)
        wp = np.where(mask, wp, np.nan)
        se = np.where(mask, 0.5 / np.sqrt(n), np.nan)
        return pd.DataFrame({"idx": idx, "wp_go": wp[:, 0], "wp_fg": wp[:, 1],
                             "wp_punt": wp[:, 2], "se_go": se[:, 0], "se_fg": se[:, 1],
                             "se_punt": se[:, 2], "done": mask.any(1)})

    return plan_fn


def _first_pass(n_states: int, n0: int) -> pd.DataFrame:
    idx = np.arange(n_states, dtype=np.int64)
    calls: list = []
    df = _fake_plan(calls)(idx, n0, np.ones((n_states, 3), bool), None)
    d = R.decide(df[R._WP_COLS].to_numpy(float), df[R._SE_COLS].to_numpy(float),
                 np.ones(n_states, bool))
    for k, v in d.items():
        df[k] = v
    for c in ("go", "fg", "punt"):
        df[f"allowed_{c}"] = True
    df["real_choice"] = "PUNT"
    return df


@pytest.fixture
def run(tmp_path):
    return R.Run(_tiny_preset(), tmp_path, quiet=True)


def test_refine_tops_up_only_close_calls_and_only_their_top_two(run, tmp_path):
    n0, cap = 64, 1024
    df = _first_pass(8, n0)
    close0 = df["too_close"].to_numpy(bool)
    assert close0.tolist() == [True, False, True, True] * 2
    calls: list = []
    out, info = R.refine_close_calls(run, df, _fake_plan(calls), n0, cap, True,
                                     tmp_path / "refine", FAR, 1)
    asked = np.concatenate([c[0] for c in calls])
    assert sorted(asked.tolist()) == sorted(np.flatnonzero(close0).tolist())
    for idx, n, mask in calls:
        assert (mask.sum(1) == 2).all()  # two actions only
        for i, m in zip(idx, mask, strict=True):
            want = R.top_two_mask(df[R._WP_COLS].to_numpy(float)[[i]], np.ones((1, 3), bool))
            assert np.array_equal(m, want[0])
        assert n0 + n <= cap and n % n0 == 0
    # Counts per action: n0 for the first pass, plus the extra on the two refined actions.
    for i in np.flatnonzero(close0):
        n = out.loc[i, ["n_go", "n_fg", "n_punt"]].to_numpy(int)
        assert sorted(n.tolist())[0] == n0 and n.max() > n0
    assert (out.loc[~close0, ["n_go", "n_fg", "n_punt"]].to_numpy() == n0).all()
    assert info["too_close_first_pass"] == 6 and info["planned_states"] == 6
    assert info["refined_states"] == 6 and info["not_refined_close_states"] == 0
    assert info["complete"] and info["enabled"]
    # Margins of 0.01 or less stay under 2 standard errors of the difference even at the
    # cap (2 * 0.5 * sqrt(2 / 1024) = 0.044), so these stay close; the counts must add up.
    still = out["too_close"].to_numpy(bool)
    assert info["still_too_close_after_refinement"] == int((still & close0).sum())
    assert info["resolved_by_refinement"] + info["still_too_close_after_refinement"] == 6
    assert info["too_close_now"] == int(still.sum())
    assert info["refined_at_cap_and_still_close"] == int(
        (still & (out[["n_go", "n_fg", "n_punt"]].max(axis=1).to_numpy() >= cap)).sum())
    # Exact means pool to themselves; the pooled se is 0.5 / sqrt(total n).
    i = 0
    assert out.loc[i, "wp_fg"] == pytest.approx(0.51)
    assert out.loc[i, "se_fg"] == pytest.approx(0.5 / np.sqrt(out.loc[i, "n_fg"]))
    assert out.loc[i, "se_punt"] == pytest.approx(0.5 / np.sqrt(n0))  # not refined
    # First-pass columns kept; agreement recomputed.
    assert (out["too_close_first_pass"] == close0).all()
    assert (out["agrees"] == (out["best"] == "PUNT")).all()


def test_refine_resolves_a_close_call_with_enough_rollouts(run, tmp_path):
    n0 = 64
    df = _first_pass(4, n0)
    df.loc[0, "wp_fg"] = 0.56  # GO 0.50 vs FG 0.56 in the first pass: margin 0.06
    df.loc[0, ["best", "margin", "too_close"]] = ["FG", 0.06, True]
    out, info = R.refine_close_calls(run, df.iloc[[0]].copy(), _fake_plan([]), n0, 4096, True,
                                     tmp_path / "r2", FAR, 1)
    # The refinement batches see the true 0.51: pooled FG falls toward it, and with
    # enough rollouts the call is decided.
    assert info["refined_states"] == 1
    assert out.loc[0, "n_go"] > n0 and out.loc[0, "wp_fg"] < 0.56


def test_refine_waits_for_a_complete_first_pass_and_respects_the_deadline(run, tmp_path):
    df = _first_pass(8, 64)
    calls: list = []
    out, info = R.refine_close_calls(run, df, _fake_plan(calls), 64, 1024, False,
                                     tmp_path / "a", FAR, 1)
    assert not calls and info["refined_states"] == 0 and "incomplete" in info["skipped"]
    assert not info["complete"] and info["not_refined_close_states"] == 6
    assert (out["best"] == df["best"]).all()
    out, info = R.refine_close_calls(run, df, _fake_plan(calls), 64, 1024, True,
                                     tmp_path / "b", time.time() - 1, 1)
    assert not calls and info["refined_states"] == 0 and not info["complete"]
    assert info["progress"]["stopped"] == "time budget"
    # Disabled when the cap is not above the first pass.
    out, info = R.refine_close_calls(run, df, _fake_plan(calls), 64, 64, True,
                                     tmp_path / "c", FAR, 1)
    assert not calls and not info["enabled"]


def test_refine_resumes_from_saved_chunks_with_the_same_result(run, tmp_path):
    df = _first_pass(16, 64)
    calls: list = []
    a, ia = R.refine_close_calls(run, df, _fake_plan(calls), 64, 1024, True,
                                 tmp_path / "r", FAR, 5)
    n_calls = len(calls)
    b, ib = R.refine_close_calls(run, df, _fake_plan(calls), 64, 1024, True,
                                 tmp_path / "r", FAR, 5)
    assert len(calls) == n_calls  # everything reused
    assert ib["progress"]["computed_states"] == 0 and ib["progress"]["reused_states"] == 12
    pd.testing.assert_frame_equal(a, b)
    # The order is a seeded shuffle of the close states, not their record order.
    order = np.concatenate([c[0] for c in calls])
    assert sorted(order.tolist()) == np.flatnonzero(df["too_close"]).tolist()


def test_refine_with_the_real_planner(run, tmp_path):
    split = make_toy_split(12, np.random.default_rng(3))
    torch.manual_seed(0)
    model = WorldModel(TINY)
    idx = SP.fourth_down_records(split)[:6]
    df = plan_fourth_downs(model, split, idx, 8, TINY, FAR, np.random.default_rng(0))
    df["too_close"] = df["done"]  # force every decision through the refinement

    def plan_fn(part, n, mask, rng):
        return plan_fourth_downs(model, split, part, n, TINY, FAR, rng, action_mask=mask)

    out, info = R.refine_close_calls(run, df, plan_fn, 8, 32, True, tmp_path / "real", FAR, 2)
    assert info["refined_states"] == int(df["done"].sum()) and info["complete"]
    assert (out.loc[out["refined"], ["n_go", "n_fg", "n_punt"]].max(axis=1) > 8).all()
    assert set(out["best"].dropna()) <= {"GO", "FG", "PUNT"}


# ---------------------------------------------------------------- stage_plan wiring
def test_stage_plan_refines_close_calls(tmp_path, monkeypatch):
    calls: dict = {}
    stubs = _stubs(calls)
    seen: list = []

    def plan_stub(m, split, idx, n, cfg, deadline_ts, rng, action_mask=None):
        idx = np.asarray(idx)
        k = len(idx)
        mask = np.ones((k, 3), bool) if action_mask is None else np.asarray(action_mask)
        seen.append((split, idx.copy(), int(n), mask.copy()))
        wp = np.where(mask, np.array([[0.50, 0.505, 0.30]]), np.nan)
        se = np.where(mask, 0.5 / np.sqrt(n), np.nan)
        d = R.decide(wp, se, mask.any(1))
        return pd.DataFrame({"idx": idx, "wp_go": wp[:, 0], "wp_fg": wp[:, 1],
                             "wp_punt": wp[:, 2], "se_go": se[:, 0], "se_fg": se[:, 1],
                             "se_punt": se[:, 2], **d, "allowed_go": mask[:, 0],
                             "allowed_fg": mask[:, 1], "allowed_punt": mask[:, 2],
                             "real_action": ["PUNT"] * k, "real_choice": ["PUNT"] * k,
                             "done": mask.any(1)})

    stubs["simulate"].plan_fourth_downs = plan_stub
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    p = _tiny_preset()
    run = R.Run(p, tmp_path, quiet=True)
    for s in ("data", "train"):
        assert R.run_stage(run, s)["status"] == "done"
    out = R.run_stage(run, "plan")
    assert out["status"] == "done", out
    rf = out["refinement"]
    n0, cap = p["plan"]["n_rollouts"], p["plan"]["refine_max_rollouts"]
    assert rf["enabled"] and rf["cap_per_action"] == cap and rf["complete"]
    assert rf["too_close_first_pass"] > 0 and rf["refined_states"] == rf["too_close_first_pass"]
    dec = pd.DataFrame(out["decisions"])
    assert {"n_go", "n_fg", "n_punt", "refined", "too_close_first_pass"} <= set(dec.columns)
    assert out["agreement_first_pass"]["n"] == out["agreement"]["n"]
    te, _ = run.split("test")
    rules = out["support"]["rules"]
    refine_calls = [c for c in seen if c[0].game_ids == te.game_ids and c[2] != n0]
    assert refine_calls
    for split, idx, n, mask in refine_calls:
        assert n0 + n <= cap
        assert (mask.sum(1) <= 2).all()
        assert not (mask & ~SP.action_mask(split, idx, rules)).any()  # hard rules still hold
    # The test plan, the rules and the summary describe both v3 changes.
    tp = R.test_plan(run)
    assert any("60 yards" in m and "two best actions" in m for m in tp["metrics"])
    assert rules["fg_max_distance_yards"] == 60 and "post hoc" in out["support"]["note"]
    R.run_stage(run, "package")
    assert "close calls:" in R.summary(run)
