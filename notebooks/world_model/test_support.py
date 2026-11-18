"""Tests for support.py (action support, hard rules, soft flags, the kicking diagnostic) and
its wiring into simulate.plan_fourth_downs and run.stage_plan. CPU, toy data, no network."""

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
from simulate import plan_fourth_downs, rollout
from test_run import _stubs, _tiny_preset
from toy import make_toy_split

torch.set_num_threads(1)
SI, TI, PH, A = S.STATE_INDEX, S.TARGET_INDEX, S.PHASE, S.ACTION
FAR = time.time() + 3600.0
TINY = {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64, "dec_hidden": 32,
        "dropout": 0.0, "max_len": 48, "max_steps": 150, "sim_batch": 512, "seed": 0}


@pytest.fixture(scope="module")
def train_split() -> S.Split:
    return make_toy_split(120, np.random.default_rng(0))


@pytest.fixture(scope="module")
def big_train() -> S.Split:
    """Longer toy games, enough field-goal attempts per 5-yard band to estimate a rate."""
    return make_toy_split(300, np.random.default_rng(7), start_clock=1800)


@pytest.fixture(scope="module")
def test_split() -> S.Split:
    return make_toy_split(40, np.random.default_rng(1), first_game_id=10_000)


def toy_make_prob(yl):
    """toy.py's field-goal law: good with probability max(0.2, 1 - yardline / 50)."""
    return np.maximum(0.2, 1.0 - np.asarray(yl, dtype=float) / 50.0)


PUNT_NET = 35  # FakeKicker's punt: 35 yards net, or a touchback at the 20


class FakeKicker:
    """A stand-in world model with a known first transition after a forced kick.

    encode() exposes the yard line, action and possession of each record; sample_transition()
    makes a field goal with probability make_prob(yard line) (3 points for the kicking team),
    and a punt nets PUNT_NET yards (touchback at the receiver's 20 if it would reach the end
    zone). Any other action: an incomplete-looking change of possession with no points.
    """

    max_len = 64

    def __init__(self, make_prob=toy_make_prob):
        self.cfg = {"temperature": 1.0, "mask_pad_codes": True}
        self.make_prob = make_prob
        self.encoded_actions: list[np.ndarray] = []
        self._p = torch.zeros(1)

    def parameters(self):
        return iter([self._p])

    def eval(self):
        return self

    def count_params(self):
        return 0

    def encode(self, header, states):
        self.encoded_actions.append(states[:, :, SI["action"]].numpy().copy())
        cols = [states[..., SI[k]] for k in ("yardline", "action", "poss")]
        return torch.stack(cols, dim=-1).float()

    def sample_transition(self, h, temperature=1.0, generator=None, mask_pad=True):
        yl, act, poss = h[:, 0].long(), h[:, 1].long(), h[:, 2].long()
        r = h.shape[0]
        u = torch.rand(r, generator=generator)
        p = torch.as_tensor(self.make_prob(yl.numpy()), dtype=torch.float32)
        made = (act == A["FG"]) & (u < p)
        sign = torch.where(poss == S.POSS_CODE["HOME"], 1, -1)
        pts = torch.where(made, 3 * sign, torch.zeros_like(sign))
        after = yl - PUNT_NET  # punter's distance to the goal line after the punt
        recv_yl = torch.where(after <= 0, torch.full_like(yl, 80), 100 - after)
        out = torch.zeros(r, len(S.TARGET_FIELDS), dtype=torch.long)
        out[:, TI["phase"]] = torch.where(made, PH["KICKOFF"], PH["PLAY"])
        out[:, TI["poss"]] = 3 - poss
        out[:, TI["score_delta"]] = torch.as_tensor(
            [S.SCORE_DELTA_CODE[int(v)] for v in pts], dtype=torch.long)
        out[:, TI["elapsed"]] = 6
        out[:, TI["to_used_home"]] = 1
        out[:, TI["to_used_away"]] = 1
        out[:, TI["down"]] = torch.where(made, 0, 1)
        out[:, TI["ydstogo"]] = torch.where(made, 0, 10)
        out[:, TI["yardline"]] = torch.where(
            act == A["PUNT"], recv_yl, torch.where(made, 65, (100 - yl).clamp(1, 99)))
        out[:, TI["action"]] = torch.where(made, A["KICKOFF"], A["RUN"])
        return out


# ---------------------------------------------------------------- 1. training split only
def test_support_tables_count_the_training_split(train_split, test_split):
    sup = SP.build_support(train_split)
    st = train_split.states
    fd = SP.fourth_down_records(train_split)
    manual = np.zeros((100, S.MAX_YDSTOGO + 1, 3), dtype=np.int64)
    for i in fd:
        a = int(st[i, SI["action"]])
        j = 0 if a in S.GO_ACTIONS else (1 if a == A["FG"] else 2)
        manual[st[i, SI["yardline"]], st[i, SI["ydstogo"]], j] += 1
    assert np.array_equal(sup.counts, manual)
    assert sup.meta["split_games"] == len(train_split.game_ids)
    assert sup.meta["fourth_downs"] == len(fd)
    # Comparable states: yard line within +-3 and the same yards-to-go bucket.
    yl, ytg = 50, 10
    want = manual[47:54, 8:11].sum(axis=(0, 1))
    assert np.array_equal(sup.smooth[yl, ytg], want)
    # Field-goal makes, counted independently from the score bookkeeping (raw columns).
    last = np.zeros(len(st), dtype=bool)
    last[train_split.offsets[1:] - 1] = True
    games = R.record_game(train_split)
    att = np.zeros(100, dtype=np.int64)
    mk = np.zeros(100, dtype=np.int64)
    for i in np.flatnonzero((st[:, SI["phase"]] == PH["PLAY"]) & (st[:, SI["action"]] == A["FG"])):
        after = (int(train_split.final_diff[games[i]]) if last[i]
                 else int(train_split.raw[i + 1, 1]))
        sign = 1 if st[i, SI["poss"]] == S.POSS_CODE["HOME"] else -1
        att[st[i, SI["yardline"]]] += 1
        mk[st[i, SI["yardline"]]] += int((after - int(train_split.raw[i, 1])) * sign == 3)
    assert np.array_equal(sup.fg_attempts, att) and np.array_equal(sup.fg_makes, mk)
    # toy.py makes a field goal with probability max(0.2, 1 - yl / 50): the empirical rate
    # over all training attempts matches that law.
    yls = np.repeat(np.arange(100), att)
    expect = toy_make_prob(yls).mean()
    se = np.sqrt(expect * (1 - expect) / att.sum())
    assert abs(mk.sum() / att.sum() - expect) < 4 * se
    # A different split gives different tables: nothing from it leaked in.
    both = SP.build_support(R.concat_splits([train_split, test_split]))
    assert not np.array_equal(both.counts, sup.counts)


def test_stage_plan_builds_support_from_the_training_split_only(tmp_path, monkeypatch):
    calls: dict = {}
    stubs = _stubs(calls)
    seen_masks: list[np.ndarray] = []

    def plan_stub(m, split, idx, n, cfg, deadline_ts, rng, action_mask=None):
        """Honours the mask: best = first allowed of FG, PUNT, GO; masked wp are NaN."""
        idx = np.asarray(idx)
        seen_masks.append((split, idx.copy(), np.asarray(action_mask).copy()))
        k = len(idx)
        wp = np.where(action_mask, np.array([[0.5, 0.6, 0.55]]), np.nan)
        best = [next(c for c, j in (("FG", 1), ("PUNT", 2), ("GO", 0)) if action_mask[q, j])
                for q in range(k)]
        real = [SP.CHOICES[SP.choice_index(a)] if SP.choice_index(a) >= 0 else "GO"
                for a in split.states[idx, SI["action"]]]
        return pd.DataFrame({"idx": idx, "wp_go": wp[:, 0], "wp_fg": wp[:, 1],
                             "wp_punt": wp[:, 2], "se_go": 0.01, "se_fg": 0.01, "se_punt": 0.01,
                             "best": best, "margin": 0.05, "too_close": False,
                             "real_action": real, "real_choice": real,
                             "done": np.ones(k, dtype=bool)})

    stubs["simulate"].plan_fourth_downs = plan_stub
    stubs["train"].load_checkpoint = lambda path, cfg_override=None, device=None: (
        FakeKicker(), {})
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    built = []
    real_build = SP.build_support

    def spy(split, cfg=None):
        built.append(list(split.game_ids))
        return real_build(split, cfg)

    monkeypatch.setattr(SP, "build_support", spy)
    p = _tiny_preset()
    run = R.Run(p, tmp_path, quiet=True)
    for s in ("data", "train"):
        assert R.run_stage(run, s)["status"] == "done"
    out = R.run_stage(run, "plan")
    assert out["status"] == "done", out
    tr, _ = run.split("train")
    assert built == [list(tr.game_ids)]  # built once, from the training split
    assert all(g.startswith(f"{p['seasons']['train'][0]}_") for g in built[0])
    tables = out["support"]["training_tables"]
    assert tables["split_seasons"] == sorted({int(x) for x in tr.seasons})
    # Every planner call got the hard-rule mask for exactly its states.
    rules = out["support"]["rules"]
    for split, idx, mask in seen_masks:
        assert np.array_equal(mask, SP.action_mask(split, idx, rules))
    # Masked actions recorded, flags and the split agreement present.
    assert out["masked_actions"] and all(m["masked"] in ("FG", "PUNT")
                                         for m in out["masked_actions"])
    assert {"well_supported", "low_support", "min_support"} <= set(out["agreement_by_support"])
    dec = pd.DataFrame(out["decisions"])
    assert {"sup_n", "sup_count_best", "sup_share_best", "low_support", "sim_fg_make",
            "emp_fg_make"} <= set(dec.columns)
    assert "error" not in out["support"]["diagnostic"], out["support"]["diagnostic"]
    pkg = R.run_stage(run, "package")
    assert "fg_support" in pkg["figures"], pkg["figures"]
    assert (tmp_path / "figures" / "fg_support.png").exists()
    text = R.summary(run)
    assert "action rules" in text and "support:" in text and "field goals, simulated" in text


# ---------------------------------------------------------------- 2. hard rules
def test_mask_removes_out_of_range_field_goals_and_deep_punts(train_split, test_split):
    sup = SP.build_support(train_split)
    rules = SP.hard_rules(sup)
    longest = int(np.flatnonzero(sup.fg_makes > 0).max())
    assert rules["longest_made_fg_yardline_train"] == longest
    assert rules["fg_max_yardline"] == longest + 3
    assert rules["fg_max_distance_yards"] == longest + 3 + SP.KICK_DISTANCE_OFFSET
    assert rules["punt_min_yardline"] == 30
    sp = R.subset_split(test_split, np.arange(len(test_split.game_ids)))
    idx = SP.fourth_down_records(sp)[:7]
    yls = [1, 29, 30, rules["fg_max_yardline"], rules["fg_max_yardline"] + 1, 75, 99]
    sp.states[idx, SI["yardline"]] = yls
    m = SP.action_mask(sp, idx, rules)
    assert m[:, 0].all()  # going for it is always allowed
    assert m[:, 1].tolist() == [y <= rules["fg_max_yardline"] for y in yls]
    assert m[:, 2].tolist() == [y >= 30 for y in yls]
    masked = SP.masked_actions(sp, idx, rules)
    assert len(masked) == int((~m).sum())
    assert {(e["idx"], e["masked"]) for e in masked} == {
        (int(idx[q]), SP.CHOICES[j]) for q, j in zip(*np.nonzero(~m), strict=True)}
    assert all(e["rule"] for e in masked)
    # Changing the margin moves the limit.
    rules5 = SP.hard_rules(SP.build_support(train_split, {"fg_margin_yards": 5}))
    assert rules5["fg_max_yardline"] == longest + 5
    with pytest.raises(ValueError):
        SP.config({"support": {"no_such_setting": 1}})


def test_low_support_flag(train_split, test_split):
    sup = SP.build_support(train_split)
    rules = SP.hard_rules(sup)
    idx = SP.fourth_down_records(test_split)[:30]
    df = pd.DataFrame({"idx": idx, "best": ["FG"] * 15 + ["GO"] * 14 + [None],
                       "real_choice": ["PUNT"] * 30})
    out = SP.add_support_columns(df, sup, test_split, rules)
    lk = SP.lookup(sup, test_split, idx)
    want = np.r_[lk["sup_fg"][:15], lk["sup_go"][15:29]]
    assert np.array_equal(out["sup_count_best"][:29].to_numpy(), want)
    assert out["low_support"][:29].tolist() == (want < 10).tolist()
    assert not out["low_support"].iloc[29]  # no recommendation, no flag
    ok = lk["sup_n"][:29].to_numpy() > 0
    share = out["sup_share_best"][:29].to_numpy()
    assert np.allclose(share[ok], want[ok] / lk["sup_n"][:29].to_numpy()[ok])
    assert (lk["ctx_n"] <= lk["sup_n"]).all()  # context cells are subsets


# ---------------------------------------------------------------- 3. plan with a mask
def _fourth(split: S.Split, n: int) -> np.ndarray:
    st = split.states
    home = np.flatnonzero((st[:, SI["down"]] == 4) & (st[:, SI["phase"]] == PH["PLAY"])
                          & (st[:, SI["poss"]] == S.POSS_CODE["HOME"]))
    away = np.flatnonzero((st[:, SI["down"]] == 4) & (st[:, SI["phase"]] == PH["PLAY"])
                          & (st[:, SI["poss"]] == S.POSS_CODE["AWAY"]))
    return np.sort(np.r_[home[: n // 2], away[: n - n // 2]])


def test_plan_with_mask_never_recommends_a_masked_action(test_split):
    torch.manual_seed(0)
    model = WorldModel(TINY).eval()
    idx = _fourth(test_split, 8)
    mask = np.ones((len(idx), 3), dtype=bool)
    mask[0, 1] = mask[1, 2] = False
    mask[2, 1:] = False  # only GO allowed
    mask[3, 0] = False  # GO masked (support never does this; the planner must still obey)
    mask[4:, 1] = [True, False, True, False]
    df = plan_fourth_downs(model, test_split, idx, 6, TINY, FAR, np.random.default_rng(3),
                           action_mask=mask)
    assert df["done"].all()
    for j, ch in enumerate(("go", "fg", "punt")):
        assert df[f"wp_{ch}"][~mask[:, j]].isna().all()
        assert df[f"se_{ch}"][~mask[:, j]].isna().all()
        assert df[f"wp_{ch}"][mask[:, j]].notna().all()
        assert df[f"allowed_{ch}"].tolist() == mask[:, j].tolist()
    for q, b in enumerate(df["best"]):
        assert mask[q, ("GO", "FG", "PUNT").index(b)], (q, b)
    assert df["best"].iloc[2] == "GO" and not df["too_close"].iloc[2]
    c = df.attrs["counters"]
    for ch, j in (("GO", 0), ("FG", 1), ("PUNT", 2)):
        assert c[ch]["rollouts"] == 6 * int(mask[:, j].sum())
    # An all-True mask is the same computation as no mask (same draws, same numbers).
    a = plan_fourth_downs(model, test_split, idx, 6, TINY, FAR, np.random.default_rng(5))
    b = plan_fourth_downs(model, test_split, idx, 6, TINY, FAR, np.random.default_rng(5),
                          action_mask=np.ones((len(idx), 3), dtype=bool))
    cols = ["wp_go", "wp_fg", "wp_punt", "se_go", "se_fg", "se_punt", "best", "margin"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    with pytest.raises(ValueError):
        plan_fourth_downs(model, test_split, idx, 6, TINY, FAR, np.random.default_rng(5),
                          action_mask=np.ones((len(idx) + 1, 3), dtype=bool))


def test_mask_rows_align_with_idx_as_passed(test_split):
    """Rows the planner drops (not fourth-down PLAY records) take their mask row with them."""
    torch.manual_seed(0)
    model = WorldModel(TINY).eval()
    fourth = _fourth(test_split, 2)
    other = np.flatnonzero(test_split.states[:, SI["down"]] == 1)[:1]
    idx = np.r_[other, fourth]
    mask = np.array([[True, False, False], [True, False, True], [True, True, False]])
    df = plan_fourth_downs(model, test_split, idx, 4, TINY, FAR, np.random.default_rng(1),
                           action_mask=mask)
    assert df["idx"].tolist() == fourth.tolist()
    assert df["allowed_fg"].tolist() == [False, True]
    assert df["allowed_punt"].tolist() == [True, False]


# ---------------------------------------------------------------- 4. the diagnostic
def test_first_transitions_force_the_action_on_the_last_record(test_split):
    model = FakeKicker()
    idx = _fourth(test_split, 4)
    SP.first_transitions(model, test_split, idx, "FG", 3, None, np.random.default_rng(0))
    acts = model.encoded_actions[-1]
    games = R.record_game(test_split)[idx]
    for r, (i, g) in enumerate(zip(idx, games, strict=True)):
        n = int(i) - max(int(test_split.offsets[g]), int(i) + 1 - model.max_len) + 1
        assert acts[r, n - 1] == A["FG"]
        start = int(i) - n + 1
        assert np.array_equal(acts[r, : n - 1], test_split.states[start:int(i), SI["action"]])


def test_first_transitions_sample_the_rollout_first_step(test_split):
    """The one-step sampler conditions on the same hidden state as the first step of a real
    rollout with the action forced, so it samples the same first transition."""
    torch.manual_seed(0)
    model = WorldModel(TINY).eval()
    idx = _fourth(test_split, 4)
    seen: list[torch.Tensor] = []
    real = model.sample_transition

    def spy(h, *a, **k):
        seen.append(h.detach().float().clone())
        return real(h, *a, **k)

    model.sample_transition = spy
    rollout(model, test_split, idx, A["FG"], 3, {**TINY, "max_steps": 1}, FAR,
            np.random.default_rng(0))
    first_step = seen[0]
    seen.clear()
    SP.first_transitions(model, test_split, idx, "FG", 3, TINY, np.random.default_rng(0))
    assert first_step.shape == seen[0].shape
    assert torch.allclose(first_step, seen[0], atol=1e-5)


def test_simulated_make_rate_matches_a_toy_model_with_a_known_rate(big_train, test_split):
    sup = SP.build_support(big_train, {"diag_states_per_band": 6, "diag_draws": 400})
    rules = SP.hard_rules(sup)
    model = FakeKicker()  # makes field goals with the same law as toy.py
    idx = _fourth(test_split, 10)
    sim = SP.sim_fg_make(model, test_split, idx, 400, None, np.random.default_rng(2))
    p = toy_make_prob(test_split.states[idx, SI["yardline"]])
    assert np.all(np.abs(sim - p) < 4 * np.sqrt(p * (1 - p) / 400) + 1e-9)  # HOME and AWAY

    d = SP.sim_vs_empirical(model, test_split, sup, rules, None, np.random.default_rng(4))
    n_draws = d["draws_per_state"]
    checked = 0
    for row in d["fg"]:
        lo, hi = row["yardline"]
        assert row["train_attempts"] == int(sup.fg_attempts[lo:hi + 1].sum())
        assert row["train_makes"] == int(sup.fg_makes[lo:hi + 1].sum())
        if not row["state_idx"]:
            assert row["sim_make_rate"] is None
            continue
        yl = test_split.states[row["state_idx"], SI["yardline"]]
        assert np.all((yl >= lo) & (yl <= hi))
        want = toy_make_prob(yl).mean()
        se = np.sqrt(want * (1 - want) / (len(yl) * n_draws))
        assert abs(row["sim_make_rate"] - want) < 4 * se + 1e-9
        assert row["beyond_fg_limit"] == (lo > rules["fg_max_yardline"])
        checked += 1
    assert checked >= 3
    for row in d["punt"]:
        if not row["state_idx"]:
            continue
        yl = test_split.states[row["state_idx"], SI["yardline"]].astype(float)
        want = np.where(yl - PUNT_NET <= 0, yl - 20, PUNT_NET).mean()
        assert row["sim_net_yards"] == pytest.approx(want)
        assert row["sim_other_share"] == 0.0
    # The same law as the data: no systematic gap where the training data is thick. An
    # over-optimistic simulator (every field goal good) shows a large positive gap.
    h = SP.fg_gap_summary(d, min_attempts=30)
    assert h["bands_with_data"] >= 1 and abs(h["weighted_mean_gap_with_data"]) < 0.1
    rosy = SP.sim_vs_empirical(FakeKicker(lambda y: np.ones(len(y))), test_split, sup, rules,
                               None, np.random.default_rng(4))
    hr = SP.fg_gap_summary(rosy, min_attempts=30)
    assert hr["weighted_mean_gap_with_data"] > 0.1
    assert hr["sim_make_rate_beyond_limit"] is None or hr["sim_make_rate_beyond_limit"] == 1.0


def test_punt_net_yards_definition():
    tg = np.zeros((4, len(S.TARGET_FIELDS)), dtype=np.int16)
    tg[:, TI["phase"]] = [PH["PLAY"], PH["PLAY"], PH["PLAY"], PH["TRY"]]
    tg[:, TI["poss"]] = [2, 2, 1, 2]  # punter is HOME (1)
    tg[:, TI["yardline"]] = [75, 80, 60, 15]
    net = SP.punt_net_yards(np.array([70, 30, 70, 70]), np.array([1, 1, 1, 1]), tg)
    # receiver at its own 25 from the punter's 30: 45 net; touchback from the opponent's
    # 30: 10; muff recovered by the punter at the opponent's 40 (yardline 60): 10; a return
    # touchdown (next record is a try): not a net-yards outcome.
    assert net[:3].tolist() == [45.0, 10.0, 10.0] and np.isnan(net[3])
