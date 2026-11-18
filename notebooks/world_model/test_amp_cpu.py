"""Mixed precision on the CPU: the GPU code paths under autocast, simulated with bfloat16.

On a GPU, `train.amp_settings` turns autocast on (bfloat16 on compute capability 8 and up,
float16 with a GradScaler on a T4). On the CPU it is off, so these tests patch it to
(True, bfloat16, False) and run a training step and validation pass under autocast, and
rollouts through the key/value cache with bfloat16 caches and cascade attention. The
float16 GradScaler path needs CUDA and is not covered here."""

from __future__ import annotations

import time

import numpy as np
import pytest
import schema as S
import simulate
import torch
import train
from model import WorldModel
from toy import make_toy_split
from train import collate, train_model

torch.set_num_threads(1)
TINY = {"d_model": 32, "n_layers": 2, "n_heads": 2, "d_ff": 64, "dec_hidden": 32,
        "dropout": 0.0, "max_len": 48, "batch_games": 8, "eval_batch_games": 16, "lr": 3e-3,
        "warmup_steps": 5, "epochs": 1, "max_steps": 150, "sim_batch": 64, "seed": 0}
FAR = time.time() + 3600.0


def _cpu_bf16_autocast() -> bool:
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = torch.nn.functional.linear(torch.randn(2, 4), torch.randn(3, 4))
        return y.dtype == torch.bfloat16
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cpu_bf16_autocast(),
                                reason="this torch build has no CPU autocast for bfloat16")


@pytest.fixture
def bf16(monkeypatch):
    real = train.amp_settings

    def fake(device):
        if torch.device(device).type == "cpu":
            return True, torch.bfloat16, False
        return real(device)

    monkeypatch.setattr(train, "amp_settings", fake)
    monkeypatch.setattr(simulate, "amp_settings", fake)


@pytest.fixture(scope="module")
def split() -> S.Split:
    return make_toy_split(16, np.random.default_rng(0))


def test_training_step_under_autocast(split, bf16):
    torch.manual_seed(0)
    m = WorldModel(TINY)
    before = [p.detach().clone() for p in m.parameters()]
    batch = collate(split, np.arange(4), m.max_len)
    with torch.no_grad():
        ref, _ = m.loss(batch)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            low, _ = m.loss(batch)
    assert low.dtype == torch.float32  # the cross-entropy is computed in float32
    assert float(low) == pytest.approx(float(ref), rel=0.03)
    res = train_model(m, split, split, TINY, "", FAR, log=lambda *_: None)
    assert res["amp"] == "torch.bfloat16" and res["grad_scaler"] is False
    assert res["optimizer"] == "AdamW"  # fused AdamW is requested only on CUDA
    h = res["history"][0]
    assert np.isfinite(h["train_loss"]) and np.isfinite(h["val_total"])
    assert all(torch.isfinite(p).all() for p in m.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters(), strict=True))


def test_cascade_attention_with_low_precision_caches(split):
    """One encode_step with bfloat16 caches under autocast equals the full float32 encode
    up to bfloat16 rounding."""
    torch.manual_seed(6)
    m = WorldModel(TINY).eval()
    c = m.cfg
    nl, nh, dh = c["n_layers"], c["n_heads"], c["d_model"] // c["n_heads"]
    one = collate(split, np.array([2]), m.max_len)
    L = 10
    states = one["states"][:, : L + 1].repeat(2, 1, 1)
    states[1, L - 1, S.STATE_INDEX["action"]] = S.ACTION["PUNT"]
    header = one["header"].repeat(2, 1)
    with torch.no_grad():
        full = m.encode(header, states)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, kv = m.encode(header[:1], states[:1, : L - 1], return_kv=True)
            kp = torch.stack([k for k, _ in kv]).to(torch.bfloat16)
            vp = torch.stack([v for _, v in kv]).to(torch.bfloat16)
            ko = torch.zeros(nl, 2, nh, 2, dh, dtype=torch.bfloat16)
            vo = torch.zeros_like(ko)
            cell = torch.tensor([0, 1])
            for j in range(2):
                ok = torch.cat([torch.ones(2, L, dtype=torch.bool),
                                torch.arange(j + 1).unsqueeze(0).expand(2, -1) <= j], dim=1)
                bias = torch.zeros(ok.shape).masked_fill_(~ok, float("-inf"))
                h = m.encode_step(states[:, L - 1 + j], torch.full((2,), L + j), ko, vo, j + 1,
                                  cache_pos=torch.full((2,), j), prefix=(kp, vp, cell, 2, bias))
                assert h.dtype == torch.float32
                ref = full[:, L - 1 + j]
                err = (h - ref).norm(dim=-1) / ref.norm(dim=-1)
                assert (err < 0.05).all(), (j, err)


def test_sampler_runs_in_float32_inside_autocast():
    torch.manual_seed(2)
    m = WorldModel(TINY).eval()
    h = torch.randn(500, TINY["d_model"])
    a = m.sample_transition(h, 1.0, torch.Generator().manual_seed(1))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        b = m.sample_transition(h, 1.0, torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_kv_cache_rollouts_under_autocast(split, bf16):
    torch.manual_seed(0)
    m = WorldModel(TINY)
    train_model(m, split, split, {**TINY, "epochs": 2}, "", FAR, log=lambda *_: None)
    m.eval()
    cfg = {**TINY, "kv_cache": True}
    idx = np.arange(3, len(split.states), 17)
    r = simulate.rollout(m, split, idx, S.ACTION["FG"], 16, cfg, FAR, np.random.default_rng(4))
    t = r["timing"]
    assert t["path"] == "kv_cache"
    assert t["cache_mib_per_rollout"] == pytest.approx(m.kv_bytes_per_row(2) / 2**20)
    assert r["done"].all() and np.isfinite(r["home_wp"]).all()
    assert ((r["home_wp"] >= 0) & (r["home_wp"] <= 1)).all()
    assert r["counters"]["transitions"] > 0
    st = split.states
    fourth = np.flatnonzero((st[:, S.STATE_INDEX["phase"]] == S.PHASE["PLAY"])
                            & (st[:, S.STATE_INDEX["down"]] == 4))[:3]
    assert len(fourth) == 3
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 1] = False  # one masked field goal, as the support rules produce
    df = simulate.plan_fourth_downs(m, split, fourth, 8, cfg, FAR, np.random.default_rng(5),
                                    action_mask=mask)
    assert len(df) == 3 and df["done"].all() and df["best"].notna().all()
    assert np.isnan(df["wp_fg"].iloc[0]) and df["best"].iloc[0] != "FG"
