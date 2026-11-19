"""Tests for the fast paths: pre-built tensor batches, the batched field decoder, the
key/value-cache rollouts and the training timing. Each fast path is checked against the
slow reference it replaces (CPU, toy games)."""

from __future__ import annotations

import time

import numpy as np
import pytest
import schema as S
import simulate
import torch
import train
from model import TARGET_SIZES, WorldModel
from toy import make_toy_split
from train import collate, prebuild_split, tensor_batch, train_model

torch.set_num_threads(1)

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
    "sim_batch": 64,
    "seed": 0,
}
FAR = time.time() + 3600.0


@pytest.fixture(scope="module")
def split() -> S.Split:
    return make_toy_split(24, np.random.default_rng(0))


def _trained(split: S.Split, **over) -> WorldModel:
    torch.manual_seed(0)
    m = WorldModel({**TINY, **over})
    train_model(m, split, split, {**TINY, **over, "epochs": 4}, "", FAR, log=lambda *_: None)
    return m.eval()


@pytest.fixture(scope="module")
def trained(split) -> WorldModel:
    return _trained(split)


@pytest.fixture(scope="module")
def trained_short(split) -> WorldModel:
    """A 12-record window: most rollouts outgrow it, so the cache falls back to re-encoding."""
    return _trained(split, max_len=12)


# ------------------------------------------------------------------ pre-built batches
@pytest.mark.parametrize("max_len", [48, 20])
def test_tensor_batches_equal_collate(split, max_len):
    ts = prebuild_split(split, max_len)
    rng = np.random.default_rng(1)
    for _ in range(5):
        games = rng.choice(len(split.game_ids), size=7, replace=False)
        a = collate(split, games, max_len)
        b = tensor_batch(ts, games)
        assert list(a) == list(b)
        for k in a:
            assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k


def test_tensor_batches_give_the_same_loss_as_collate(split):
    m = WorldModel(TINY).eval()
    ts = prebuild_split(split, m.max_len)
    order = np.random.default_rng([0, 0]).permutation(len(split.game_ids))
    old = list(train.make_batches(split, 8, m.max_len, np.random.default_rng([0, 0])))
    for bi, ref in enumerate(old):
        new = tensor_batch(ts, order[bi * 8 : (bi + 1) * 8])
        with torch.no_grad():
            la, ia = m.loss(ref)
            lb, ib = m.loss(new)
        assert torch.equal(la, lb)
        assert torch.equal(ia["counts"], ib["counts"])


def test_evaluate_nll_matches_reference_batching(split):
    m = WorldModel(TINY).eval()
    fast = train.evaluate_nll(m, split, TINY)
    sums = torch.zeros(len(S.TARGET_FIELDS), dtype=torch.float64)
    n_rec = 0
    with torch.no_grad():
        for b in train.make_batches(split, 16, m.max_len, None):
            _, info = m.loss(b, parallel=False)
            sums += info["sums"].double()
            n_rec += int(b["mask"].sum())
    assert fast["n_records"] == n_rec
    assert fast["total"] == pytest.approx(float(sums.sum()) / n_rec, rel=1e-6)


# ------------------------------------------------------------------ batched decoder
def test_parallel_decoder_gives_the_same_loss_and_gradients(split):
    torch.manual_seed(3)
    m = WorldModel(TINY)
    batch = collate(split, np.arange(6), m.max_len)
    out = {}
    for par in (False, True):
        m.zero_grad()
        total, info = m.loss(batch, parallel=par)
        total.backward()
        out[par] = (total.detach(), info, [p.grad.clone() for p in m.parameters()])
    (ta, ia, ga), (tb, ib, gb) = out[False], out[True]
    assert float(tb) == pytest.approx(float(ta), rel=1e-6)
    assert torch.equal(ia["counts"], ib["counts"])
    assert torch.allclose(ia["sums"], ib["sums"], rtol=1e-5)
    for name in ia["fields"]:
        a, b = float(ia["fields"][name].detach()), float(ib["fields"][name].detach())
        assert b == pytest.approx(a, rel=1e-5), name
    for a, b in zip(ga, gb, strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)
    # with explicit denominators (the micro-batch path)
    denom = train._full_counts(m, batch)
    with torch.no_grad():
        la, _ = m.loss(batch, denom=denom, parallel=False)
        lb, _ = m.loss(batch, denom=denom, parallel=True)
    assert float(lb) == pytest.approx(float(la), rel=1e-6)


def test_parallel_decoder_logits_and_masks(split):
    m = WorldModel(TINY).eval()
    batch = collate(split, np.arange(3), m.max_len)
    with torch.no_grad():
        h = m.encode(batch["header"], batch["states"])
        tg = m.end_prefix(batch["targets"], batch["last"])
        ref = m.decode_logits(h, tg)
        pad = m.decode_logits_padded(h.reshape(-1, h.shape[-1]), tg.reshape(-1, tg.shape[-1]))
    for j, n in enumerate(TARGET_SIZES):
        assert torch.allclose(pad[j, :, :n], ref[j].reshape(-1, n), atol=1e-5)
        assert torch.isinf(pad[j, :, n:]).all()
    assert torch.equal(m.end_prefix_fast(batch["targets"], batch["last"]), tg)
    masks = m.target_masks(tg, batch["mask"], batch["last"])
    fast = m.target_mask_tensor(tg, batch["mask"], batch["last"])
    for j, mk in enumerate(masks):
        assert torch.equal(fast[..., j], mk), j


# ------------------------------------------------------------------ key/value cache
def test_kv_cache_step_equals_full_encode(split):
    torch.manual_seed(5)
    m = WorldModel(TINY).eval()
    b = collate(split, np.array([0, 1, 2]), m.max_len)
    n0 = 6  # prefill this many records, then add the rest one at a time
    lens = b["mask"].sum(1)
    total = int(lens.min())
    c = m.cfg
    with torch.no_grad():
        full = m.encode(b["header"], b["states"])
        h0, kv = m.encode(b["header"], b["states"][:, :n0], return_kv=True)
        assert torch.allclose(h0, full[:, :n0], atol=1e-6)
        dh = c["d_model"] // c["n_heads"]
        kc = torch.zeros(c["n_layers"], 3, c["n_heads"], m.max_len + 1, dh)
        vc = torch.zeros_like(kc)
        for layer, (k, v) in enumerate(kv):
            kc[layer, :, :, : n0 + 1] = k
            vc[layer, :, :, : n0 + 1] = v
        for t in range(n0, total):
            pos = torch.full((3,), t + 1, dtype=torch.long)  # header is position 0
            h = m.encode_step(b["states"][:, t], pos, kc, vc, t + 2)
            assert torch.allclose(h, full[:, t], atol=1e-5), t
            la = m.decode_logits(h, b["targets"][:, t])
            lb = m.decode_logits(full[:, t], b["targets"][:, t])
            for x, y in zip(la, lb, strict=True):
                assert torch.allclose(x, y, atol=1e-5)


def test_shared_prefix_step_equals_full_encode(split):
    """Two rows share a game's first records and differ in the last one (as rollouts of one
    query with different start actions); the shared part is cached once for both."""
    torch.manual_seed(6)
    m = WorldModel(TINY).eval()
    c = m.cfg
    nl, nh, dh = c["n_layers"], c["n_heads"], c["d_model"] // c["n_heads"]
    one = collate(split, np.array([4]), m.max_len)
    L, extra = 10, 5  # 9 shared records, the 10th differs, then 5 more records per row
    states = one["states"][:, : L + extra].repeat(2, 1, 1)
    states[1, L - 1, S.STATE_INDEX["action"]] = S.ACTION["PUNT"]
    header = one["header"].repeat(2, 1)
    with torch.no_grad():
        full = m.encode(header, states)
        _, kv = m.encode(header[:1], states[:1, : L - 1], return_kv=True)
        kp = torch.stack([k for k, _ in kv])  # [layers, 1, heads, L, dh]
        vp = torch.stack([v for _, v in kv])
        ko = torch.zeros(nl, 2, nh, extra + 1, dh)
        vo = torch.zeros_like(ko)
        cell = torch.tensor([0, 1])  # one group of two rows
        for j in range(extra + 1):
            ok = torch.cat(
                [
                    torch.ones(2, L, dtype=torch.bool),
                    torch.arange(j + 1).unsqueeze(0).expand(2, -1) <= j,
                ],
                dim=1,
            )
            bias = torch.zeros(ok.shape).masked_fill_(~ok, float("-inf"))
            pos = torch.full((2,), L + j, dtype=torch.long)
            h = m.encode_step(
                states[:, L - 1 + j],
                pos,
                ko,
                vo,
                j + 1,
                cache_pos=torch.full((2,), j, dtype=torch.long),
                prefix=(kp, vp, cell, 2, bias),
            )
            assert torch.allclose(h, full[:, L - 1 + j], atol=1e-5), j


def test_sampler_draws_from_the_model_distribution():
    """Gumbel-max with every field's noise drawn at once: the first field's frequencies
    match softmax of its logits, the second field's its conditional given the first."""
    torch.manual_seed(2)
    m = WorldModel(TINY).eval()
    n = 40000
    h = 2.0 * torch.randn(1, TINY["d_model"]).repeat(n, 1)
    codes = m.sample_transition(h, 1.0, torch.Generator().manual_seed(0))
    with torch.no_grad():
        lg = m.field_logits(h[:1], codes[:1], 0)[0]
        lg[0] = -float("inf")  # the pad code is masked
        p0 = torch.softmax(lg, -1).double()
    f0 = torch.bincount(codes[:, 0], minlength=len(p0)).double() / n
    assert ((f0 - p0).abs() <= 4 * (p0 * (1 - p0) / n).sqrt() + 1e-9).all()
    # the most likely phase that does not end the game (its next field is then sampled);
    # chosen from p0 rather than assumed, so the test does not depend on the random init
    top = int(torch.argmax(p0.masked_fill(torch.arange(len(p0)) == S.PHASE["END"], -1.0)))
    rows = codes[:, 0] == top
    with torch.no_grad():
        lg1 = m.field_logits(h[:1], codes[rows][:1], 1)[0]
        lg1[0] = -float("inf")
        p1 = torch.softmax(lg1, -1).double()
    k = int(rows.sum())
    f1 = torch.bincount(codes[rows, 1], minlength=len(p1)).double() / k
    assert ((f1 - p1).abs() <= 4 * (p1 * (1 - p1) / k).sqrt() + 1e-9).all()


def _same(a: dict, b: dict) -> bool:
    return (
        np.array_equal(a["home_wp"], b["home_wp"])
        and np.array_equal(a["mean_steps"], b["mean_steps"])
        and a["counters"] == b["counters"]
        and a["start_actions"] == b["start_actions"]
    )


@pytest.mark.parametrize("which", ["trained", "trained_short"])
def test_kv_cache_rollouts_identical_to_recompute(split, which, request):
    model = request.getfixturevalue(which)
    idx = np.arange(3, len(split.states), 13)
    for act in (None, "GO", S.ACTION["FG"]):
        out = {}
        for kv in (False, True):
            cfg = {**TINY, "max_len": model.max_len, "kv_cache": kv}
            out[kv] = simulate.rollout(
                model, split, idx, act, 24, cfg, FAR, np.random.default_rng(9)
            )
        assert _same(out[False], out[True]), act
        t = out[True]["timing"]
        assert t["path"] == "kv_cache" and out[False]["timing"]["path"] == "recompute"
        assert t["compactions"] > 0
        if which == "trained_short":
            assert t["fallback_row_steps"] > 0  # windows outgrew max_len: re-encoded, exact
        else:
            assert t["fallback_row_steps"] == 0
    g = {
        kv: simulate.simulate_games(
            model,
            split,
            5,
            {**TINY, "max_len": model.max_len, "kv_cache": kv},
            FAR,
            np.random.default_rng(1),
        )
        for kv in (False, True)
    }
    assert g[False]["sim"] == g[True]["sim"] and g[False]["counters"] == g[True]["counters"]


def test_memory_limited_groups_are_identical_and_complete(split, trained):
    idx = np.arange(5, len(split.states), 29)
    tiny_budget = 40 * trained.kv_bytes_per_row(4) / 2**30  # 40 rollouts fit at once
    cfg = {**TINY, "kv_cache_cpu_gb": tiny_budget}
    assert simulate.live_rows(trained, cfg) == 40
    out = {
        kv: simulate.rollout(
            trained, split, idx, None, 16, {**cfg, "kv_cache": kv}, FAR, np.random.default_rng(2)
        )
        for kv in (False, True)
    }
    assert _same(out[False], out[True])
    assert out[True]["done"].all()
    # one query with more rollouts than fit at once is split inside _run
    one = {
        kv: simulate.rollout(
            trained,
            split,
            idx[:1],
            "GO",
            100,
            {**cfg, "kv_cache": kv},
            FAR,
            np.random.default_rng(3),
        )
        for kv in (False, True)
    }
    assert _same(one[False], one[True]) and one[True]["timing"]["rows"] == 100


def test_cache_memory_per_rollout():
    m = WorldModel({"d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 1024, "max_len": 256})
    assert m.kv_bytes_per_row(2) == 6 * 2 * 257 * 256 * 2  # about 1.5 MiB under fp16
    rows = (4 * 2**30) // m.kv_bytes_per_row(4)  # CPU: float32 caches, 4 GiB budget
    assert simulate.cache_rows(m, {"kv_cache_cpu_gb": 4.0}) == rows - rows % 256
    assert simulate.live_rows(m, {"kv_cache_cpu_gb": 4.0, "sim_max_live": 512}) == 512


# ------------------------------------------------------------------ training timing
def test_training_history_has_timing(split, tmp_path):
    m = WorldModel(TINY)
    res = train_model(
        m, split, split, {**TINY, "epochs": 2}, str(tmp_path / "t.pt"), FAR, log=lambda *_: None
    )
    assert res["device"] == "cpu"
    keys = (
        "batch_s",
        "h2d_s",
        "forward_s",
        "backward_opt_s",
        "train_loop_s",
        "validation_s",
        "checkpoint_s",
        "epoch_s",
    )
    for h in res["history"]:
        t = h["timing"]
        for k in keys:
            assert isinstance(t[k], float) and t[k] >= 0, k
        phases = t["batch_s"] + t["h2d_s"] + t["forward_s"] + t["backward_opt_s"]
        assert phases <= t["train_loop_s"] * 1.01 + 1e-3
        assert phases >= 0.5 * t["train_loop_s"]
        assert t["steps"] == 3 and t["train_records"] == int(split.offsets[-1])
        assert t["train_records_per_s"] > 0
    assert set(res["timing_totals_s"]) == set(keys)
    assert res["timing_totals_s"]["epoch_s"] == pytest.approx(
        sum(h["timing"]["epoch_s"] for h in res["history"])
    )


def test_train_model_moves_the_model_to_the_configured_device(split, monkeypatch):
    """The first Colab run trained on the CPU: nothing moved the model to cfg["device"]."""
    m = WorldModel(TINY)
    seen = []
    real_to = torch.nn.Module.to

    def spy(self, *a, **k):
        seen.append(a[0] if a else k.get("device"))
        return real_to(self, *a, **k)

    monkeypatch.setattr(torch.nn.Module, "to", spy)
    train_model(
        m, split, split, {**TINY, "epochs": 1, "device": "cpu"}, "", FAR, log=lambda *_: None
    )
    assert torch.device("cpu") in [torch.device(s) for s in seen if s is not None]


# ------------------------------------------------------------------ GPU-only guards, on CPU
def test_rollouts_retry_a_group_in_halves_after_out_of_memory(split, trained, monkeypatch):
    real = simulate._run_cached
    calls = []

    def flaky(model, header, *a, **k):
        calls.append(len(header))
        if len(calls) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(model, header, *a, **k)

    monkeypatch.setattr(simulate, "_run_cached", flaky)
    idx = np.arange(5, len(split.states), 41)
    r = simulate.rollout(trained, split, idx, None, 8, TINY, FAR, np.random.default_rng(2))
    assert r["done"].all() and np.isfinite(r["home_wp"]).all()
    assert r["timing"]["oom_retries"] == 1
    assert calls[1] == calls[0] // 2 and sum(calls[1:]) == calls[0]


def test_fused_adamw_falls_back_when_the_build_rejects_it(monkeypatch):
    m = WorldModel(TINY)
    real = torch.optim.AdamW
    msgs = []

    def picky(params, **kw):
        if kw.get("fused"):
            raise RuntimeError("fused=True requires all the params to be CUDA tensors")
        return real(params, **kw)

    monkeypatch.setattr(torch.optim, "AdamW", picky)
    c = {**m.cfg, "betas": (0.9, 0.95)}
    opt, fused = train.make_optimizer(m, c, torch.device("cuda"), msgs.append)
    assert not fused and isinstance(opt, real) and "using the default AdamW" in msgs[0]
    opt, fused = train.make_optimizer(m, c, torch.device("cpu"), msgs.append)
    assert not fused and len(msgs) == 1  # fused is never requested on the CPU


def test_cuda_cache_plan_uses_total_memory_capped_by_what_is_free(monkeypatch):
    m = WorldModel({"d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 1024, "max_len": 256})
    per_row = m.kv_bytes_per_row(2)
    total, free, idle = 15 * 2**30, 14 * 2**30, 0

    class Props:
        total_memory = total

    monkeypatch.setattr(simulate, "device_of", lambda model: torch.device("cuda", 0))
    monkeypatch.setattr(simulate, "amp_settings", lambda d: (True, torch.float16, True))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: Props())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (free, total))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: idle)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 0)
    rows = int(0.4 * total // per_row)
    assert simulate.cache_rows(m, {}) == rows - rows % 256  # a T4: 3840 rollouts
    assert simulate.cache_rows(m, {}) == 3840
    free = 2 * 2**30  # another process holds most of the GPU: lowered to 70% of what is free
    rows = int(0.7 * free // per_row)
    assert simulate.cache_rows(m, {}) == rows - rows % 256
    idle = 4 * 2**30  # memory PyTorch holds but does not use counts as available
    rows = int(min(0.4 * total, 0.7 * (free + idle)) // per_row)
    assert simulate.cache_rows(m, {}) == rows - rows % 256
