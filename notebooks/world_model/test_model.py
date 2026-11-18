"""CPU tests for the world model, training loop and simulator, on synthetic toy games."""

from __future__ import annotations

import time

import numpy as np
import pytest
import schema as S
import torch
from model import WorldModel
from simulate import check_transitions, plan_fourth_downs, rollout, simulate_games
from toy import make_toy_split
from train import collate, evaluate_nll, train_model

torch.set_num_threads(2)

TINY = {
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 2,
    "d_ff": 64,
    "dec_hidden": 32,
    "dropout": 0.0,
    "max_len": 48,
    "batch_games": 8,
    "eval_batch_games": 16,
    "lr": 3e-3,
    "warmup_steps": 5,
    "epochs": 3,
    "max_steps": 150,
    "sim_batch": 512,
    "seed": 0,
}
FAR = time.time() + 3600.0


def tiny_model(seed: int = 0, **over) -> WorldModel:
    torch.manual_seed(seed)
    return WorldModel({**TINY, **over})


@pytest.fixture(scope="module")
def split() -> S.Split:
    return make_toy_split(24, np.random.default_rng(0))


@pytest.fixture(scope="module")
def trained(split) -> WorldModel:
    m = tiny_model()
    train_model(m, split, split, {**TINY, "epochs": 4}, "", FAR, log=lambda *_: None)
    return m.eval()


def fourth_downs(split: S.Split, poss: str) -> np.ndarray:
    st = split.states
    return np.flatnonzero(
        (st[:, S.STATE_INDEX["down"]] == 4)
        & (st[:, S.STATE_INDEX["phase"]] == S.PHASE["PLAY"])
        & (st[:, S.STATE_INDEX["poss"]] == S.POSS_CODE[poss])
    )


def test_toy_split_follows_the_contract(split):
    for g in range(len(split.offsets) - 1):
        a, b = split.offsets[g], split.offsets[g + 1]
        raw_next = S.apply_transition(split.raw[a : b - 1], split.targets[a : b - 1])
        assert np.array_equal(raw_next, split.raw[a + 1 : b])
        codes = S.next_state_codes(raw_next, split.targets[a : b - 1])
        assert np.array_equal(codes, split.states[a + 1 : b])
        assert split.targets[b - 1, S.TARGET_INDEX["phase"]] == S.PHASE["END"]
    assert check_transitions(split.targets)["inconsistent"] == 0


def test_loss_decreases_when_overfitting_four_games(split):
    m = tiny_model()
    batch = collate(split, np.arange(4), m.max_len)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        total, _ = m.loss(batch)
        first = float(total.detach()) if first is None else first
        opt.zero_grad()
        total.backward()
        opt.step()
    assert float(total.detach()) < 0.5 * first


def test_causal_mask(split):
    m = tiny_model().eval()
    batch = collate(split, np.array([0]), m.max_len)
    t = 10
    other = {k: v.clone() for k, v in batch.items()}
    other["states"][0, t + 1 :] = torch.randint(1, 3, other["states"][0, t + 1 :].shape)
    with torch.no_grad():
        h1 = m.encode(batch["header"], batch["states"])
        h2 = m.encode(other["header"], other["states"])
        l1 = m.decode_logits(h1, batch["targets"])
        l2 = m.decode_logits(h2, batch["targets"])
    assert torch.allclose(h1[:, : t + 1], h2[:, : t + 1], atol=1e-6)
    assert not torch.allclose(h1[:, t + 1 :], h2[:, t + 1 :])
    for a, b in zip(l1, l2, strict=True):
        assert torch.allclose(a[:, : t + 1], b[:, : t + 1], atol=1e-6)


def test_generation_stays_in_vocabulary():
    m = tiny_model().eval()
    gen = torch.Generator().manual_seed(0)
    h = 20.0 * torch.randn(4096, TINY["d_model"], generator=gen)
    for temp in (1.0, 3.0, 0.0):
        codes = m.sample_transition(h, temp, gen).numpy()
        for j, (name, size) in enumerate(S.TARGET_FIELDS):
            assert codes[:, j].min() >= 0 and codes[:, j].max() < size, name
            if name not in ("down", "ydstogo"):
                assert codes[:, j].min() >= 1, name  # padding code is never sampled


def test_forced_actions_are_respected(split, trained):
    idx = np.concatenate([fourth_downs(split, "HOME")[:2], fourth_downs(split, "AWAY")[:2]])
    rng = np.random.default_rng(3)
    fg = rollout(trained, split, idx, S.ACTION["FG"], 8, TINY, FAR, rng)
    assert fg["start_actions"] == {"FG": 8 * len(idx)}
    go = rollout(trained, split, idx, "GO", 64, TINY, FAR, rng)
    assert set(go["start_actions"]) <= {"RUN", "PASS"}
    assert sum(go["start_actions"].values()) == 64 * len(idx)
    real = rollout(trained, split, idx, None, 4, TINY, FAR, rng)
    expect = {}
    for a in split.states[idx, S.STATE_INDEX["action"]]:
        expect[S.ACTIONS[a]] = expect.get(S.ACTIONS[a], 0) + 4
    assert real["start_actions"] == expect


def test_rollouts_end_and_are_counted(split, trained):
    idx = np.arange(0, len(split.states), 37)
    r = rollout(trained, split, idx, None, 8, TINY, FAR, np.random.default_rng(0))
    c = r["counters"]
    assert r["done"].all()
    assert np.all((r["home_wp"] >= 0) & (r["home_wp"] <= 1))
    assert c["ended_by_end"] + c["ended_by_clock"] + c["truncated_max_steps"] == c["rollouts"]
    short = rollout(
        trained, split, idx[:3], None, 8, {**TINY, "max_steps": 1}, FAR, np.random.default_rng(0)
    )
    assert short["counters"]["truncated_max_steps"] > 0
    assert np.all(short["mean_steps"] <= 1)
    late = rollout(
        trained, split, idx[:3], None, 8, TINY, time.time() - 1, np.random.default_rng(0)
    )
    assert not late["done"].any() and np.isnan(late["home_wp"]).all()


def test_plan_fourth_downs_perspective(split, trained):
    home, away = fourth_downs(split, "HOME")[:1], fourth_downs(split, "AWAY")[:1]
    idx = np.concatenate([home, away])
    df = plan_fourth_downs(trained, split, idx, 32, TINY, FAR, np.random.default_rng(11))
    assert list(df["poss"]) == ["HOME", "AWAY"]
    assert set(df["best"]) <= {"GO", "FG", "PUNT"}
    rng = np.random.default_rng(11)  # same draws in the same order as plan_fourth_downs
    for key, act in (("go", "GO"), ("fg", S.ACTION["FG"]), ("punt", S.ACTION["PUNT"])):
        r = rollout(trained, split, idx, act, 32, TINY, FAR, rng)
        assert df[f"wp_{key}"].iloc[0] == pytest.approx(r["home_wp"][0])
        assert df[f"wp_{key}"].iloc[1] == pytest.approx(1.0 - r["home_wp"][1])
        assert df[f"se_{key}"].tolist() == pytest.approx(r["home_wp_se"].tolist())
    wps = df[["wp_go", "wp_fg", "wp_punt"]].to_numpy()
    assert np.allclose(df["margin"], np.sort(wps, 1)[:, -1] - np.sort(wps, 1)[:, -2])
    assert (df["too_close"] == (df["margin"] < 2 * df["margin_se"])).all()


def test_checkpoint_resume_is_identical(split, tmp_path):
    cfg = {**TINY, "dropout": 0.1, "epochs": 2}
    a = tiny_model()
    ra = train_model(a, split, split, cfg, str(tmp_path / "a.pt"), FAR, log=lambda *_: None)
    b = tiny_model()
    path = str(tmp_path / "b.pt")
    rb1 = train_model(
        b, split, split, {**cfg, "stop_after_epoch": 1}, path, FAR, log=lambda *_: None
    )
    assert rb1["epochs_done"] == 1 and rb1["stopped"] == "stop_after_epoch"
    c = tiny_model(seed=123)  # different init: everything must come from the checkpoint
    rc = train_model(c, split, split, cfg, path, FAR, log=lambda *_: None)
    assert rc["epochs_done"] == 2
    for (k, va), vc in zip(a.state_dict().items(), c.state_dict().values(), strict=True):
        assert torch.equal(va, vc), k
    assert [h["val_total"] for h in ra["history"]] == [h["val_total"] for h in rc["history"]]
    ck = torch.load(str(tmp_path / "a.pt"), weights_only=False)
    assert {"model", "optimizer", "epoch", "cfg", "history"} <= set(ck)


def test_deterministic_with_fixed_seed(split, trained):
    idx = fourth_downs(split, "HOME")[:2]
    r1 = rollout(trained, split, idx, "GO", 16, TINY, FAR, np.random.default_rng(5))
    r2 = rollout(trained, split, idx, "GO", 16, TINY, FAR, np.random.default_rng(5))
    assert np.array_equal(r1["home_wp"], r2["home_wp"]) and r1["counters"] == r2["counters"]
    g1 = simulate_games(trained, split, 6, TINY, FAR, np.random.default_rng(1))
    g2 = simulate_games(trained, split, 6, TINY, FAR, np.random.default_rng(1))
    assert g1["sim"] == g2["sim"] and g1["n_games"] == 6
    m1, m2 = tiny_model(), tiny_model()
    cfg = {**TINY, "epochs": 1, "dropout": 0.1}
    train_model(m1, split, split, cfg, "", FAR, log=lambda *_: None)
    train_model(m2, split, split, cfg, "", FAR, log=lambda *_: None)
    for v1, v2 in zip(m1.state_dict().values(), m2.state_dict().values(), strict=True):
        assert torch.equal(v1, v2)
    assert evaluate_nll(m1, split)["total"] == evaluate_nll(m2, split)["total"]
