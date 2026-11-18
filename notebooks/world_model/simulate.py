"""Use the trained world model as a game simulator.

A rollout starts from a real record (its game's header and records up to and including it),
optionally with that record's action forced, and samples transitions until the game ends.
Bookkeeping (clock, score, timeouts) is done by `schema.apply_transition`; the model only
samples the transition fields. The model's own learned action distribution plays both teams
for every record after the start, so the result is "what happens if this team makes this call
and then everyone behaves like an average NFL team in the training data".
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import schema as S
import torch
from model import TARGET_NAMES, WorldModel, device_of, merge_cfg, window_start
from train import PhaseTimer, _is_oom, amp_settings

T = len(S.TARGET_FIELDS)
TI = S.TARGET_INDEX
SI = S.STATE_INDEX
PH = S.PHASE
A = S.ACTION
GO = "GO"

# Which actions may follow which phase; anything else is counted as inconsistent.
_ALLOWED = np.zeros((len(S.PHASES), len(S.ACTIONS)), dtype=bool)
for _a in ("RUN", "PASS", "PUNT", "FG", "KNEEL", "SPIKE", "NOPLAY"):
    _ALLOWED[PH["PLAY"], A[_a]] = True
for _a in ("KICKOFF", "ONSIDE", "NOPLAY"):
    _ALLOWED[PH["KICKOFF"], A[_a]] = True
for _a in ("XP", "TWO_PT", "NOPLAY"):
    _ALLOWED[PH["TRY"], A[_a]] = True

COUNTER_KEYS = (
    "rollouts",
    "transitions",
    "ended_by_end",
    "ended_by_clock",
    "truncated_max_steps",
    "inconsistent",  # transitions with at least one problem below
    "bad_phase_pad",
    "bad_poss_pad",
    "bad_down",  # down not 1..4 on PLAY, or not 0 off PLAY
    "bad_ydstogo",
    "bad_yardline_zero",
    "bad_action_for_phase",
)


def check_transitions(tg: np.ndarray) -> dict:
    """Count inconsistent sampled transitions (rows whose phase is not END)."""
    tg = np.asarray(tg)
    ph = tg[:, TI["phase"]]
    live = ph != PH["END"]
    play = ph == PH["PLAY"]
    down, togo = tg[:, TI["down"]], tg[:, TI["ydstogo"]]
    bad = {
        "bad_phase_pad": live & (ph == PH["PAD"]),
        "bad_poss_pad": live & (tg[:, TI["poss"]] == S.POSS_CODE["PAD"]),
        "bad_down": live & np.where(play, (down < 1) | (down > 4), down != 0),
        "bad_ydstogo": live & np.where(play, togo == 0, togo != 0),
        "bad_yardline_zero": live & (tg[:, TI["yardline"]] == 0),
        "bad_action_for_phase": live & ~_ALLOWED[ph, tg[:, TI["action"]]],
    }
    out = {k: int(v.sum()) for k, v in bad.items()}
    out["inconsistent"] = int(np.logical_or.reduce(list(bad.values())).sum())
    return out


def _torch_generator(rng: np.random.Generator, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(rng.integers(0, 2**62)))
    return g


def _game_of(split: S.Split, idx: np.ndarray) -> np.ndarray:
    return np.searchsorted(split.offsets, idx, side="right") - 1


def _window_buffer(split: S.Split, idx: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Context windows ending at each record in idx -> (buf [Q, max_len, S], lens [Q])."""
    games = _game_of(split, idx)
    buf = np.zeros((len(idx), max_len, len(S.STATE_FIELDS)), dtype=np.int64)
    lens = np.zeros(len(idx), dtype=np.int64)
    for q, (g, i) in enumerate(zip(games, idx, strict=True)):
        a = window_start(int(split.offsets[g]), int(i), max_len)
        k = int(i) - a + 1
        buf[q, :k] = split.states[a : i + 1]
        lens[q] = k
    return buf, lens


@torch.no_grad()
def _last_hidden(
    model: WorldModel, header: torch.Tensor, buf: torch.Tensor, lens: np.ndarray, chunk: int
) -> torch.Tensor:
    device = buf.device
    enabled, dtype, _ = amp_settings(device)
    outs = []
    for a in range(0, len(lens), chunk):
        sl = slice(a, a + chunk)
        n = int(lens[sl].max())
        with torch.autocast(device.type, dtype=dtype, enabled=enabled):
            h = model.encode(header[sl], buf[sl, :n])
        ln = torch.as_tensor(lens[sl] - 1, device=device)
        outs.append(h[torch.arange(len(ln), device=device), ln].float())
    return torch.cat(outs)


@torch.no_grad()
def _go_probs(model: WorldModel, split: S.Split, idx: np.ndarray, cfg: dict) -> np.ndarray:
    """P(RUN), P(PASS) for record i, renormalised over the two, from the model's prediction
    of record i's action at record i-1 (given the true other fields of record i)."""
    device = device_of(model)
    out = np.full((len(idx), 2), 0.5)
    games = _game_of(split, idx)
    has_prev = idx > split.offsets[games]
    if not has_prev.any():
        return out
    prev = idx[has_prev] - 1
    buf, lens = _window_buffer(split, prev, model.max_len)
    header = torch.as_tensor(split.header[games[has_prev]].astype(np.int64), device=device)
    h = _last_hidden(model, header, torch.as_tensor(buf, device=device), lens, cfg["sim_batch"])
    prefix = torch.as_tensor(split.targets[prev].astype(np.int64), device=device)
    logits = model.field_logits(h, prefix, TI["action"]).float()
    p = torch.softmax(logits, -1)[:, list(S.GO_ACTIONS)].cpu().numpy().astype(np.float64)
    tot = p.sum(1, keepdims=True)
    ok = tot[:, 0] > 0
    out_prev = np.full_like(p, 0.5)
    out_prev[ok] = p[ok] / tot[ok]
    out[has_prev] = out_prev
    return out


def cache_rows(model: WorldModel, cfg: dict) -> int:
    """Rollouts whose key/value caches fit in memory at once.

    One rollout's cache is layers x 2 x (max_len + 1) x d_model elements (2 bytes under AMP
    on a GPU, 4 on the CPU). On CUDA the budget is cfg["kv_cache_mem_frac"] of the device's
    TOTAL memory (not free memory, so the number is the same on every call and the random
    draws are reproducible), lowered only if that is more than 70% of what is actually
    available now (`torch.cuda.mem_get_info` free memory plus what PyTorch holds reserved
    but unused), for example when another process shares the GPU; on the CPU it is
    cfg["kv_cache_cpu_gb"]. Rounded down to a multiple of 256 above 256."""
    cfg = merge_cfg({**model.cfg, **(cfg or {})})
    device = device_of(model)
    enabled, dtype, _ = amp_settings(device)
    per_row = model.kv_bytes_per_row(2 if enabled else 4)
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        budget = float(cfg.get("kv_cache_mem_frac", 0.4)) * total
        free, _ = torch.cuda.mem_get_info(device)
        idle = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        budget = min(budget, 0.7 * (free + max(idle, 0)))
    else:
        budget = float(cfg.get("kv_cache_cpu_gb", 4.0)) * 2**30
    rows = max(1, int(budget // per_row))
    return rows - rows % 256 if rows >= 512 else rows


def live_rows(model: WorldModel, cfg: dict) -> int:
    """Rollouts simulated together: cfg["sim_max_live"], lowered to what the key/value cache
    fits. Both encoder paths use the same groups, so they make the same random draws."""
    cfg = merge_cfg({**model.cfg, **(cfg or {})})
    return max(1, min(int(cfg["sim_max_live"]), cache_rows(model, cfg)))


def _to_dev(a, device: torch.device) -> torch.Tensor:
    """A host array on the device. On CUDA through pinned memory with non_blocking=True: a
    plain copy from pageable memory makes PyTorch synchronise the stream, so the host would
    wait for every queued kernel at each of the small index copies of a simulated play. The
    pinned staging copy also means later in-place edits of the array cannot race the copy."""
    if device.type != "cuda":
        return torch.as_tensor(a, device=device)
    t = torch.from_numpy(np.ascontiguousarray(a))
    return t.pin_memory().to(device, non_blocking=True)


class _Book:
    """Per-row bookkeeping of one group of rollouts (host arrays)."""

    def __init__(self, lens: np.ndarray, raw: np.ndarray) -> None:
        r = len(lens)
        self.lens = lens.astype(np.int64).copy()
        self.raw = np.asarray(raw, dtype=np.int32).copy()
        self.active = np.ones(r, dtype=bool)
        self.steps = np.zeros(r, dtype=np.int64)
        self.final = self.raw[:, 1].astype(np.int64).copy()
        self.appended = np.zeros(r, dtype=np.int64)
        self.points = np.zeros(r, dtype=np.int64)
        self.kickoffs = np.zeros(r, dtype=np.int64)
        self.cnt = dict.fromkeys(COUNTER_KEYS, 0)
        self.cnt["rollouts"] = r

    def advance(self, act: np.ndarray, tg: np.ndarray, cfg: dict, max_steps: int):
        """Apply the transitions tg [len(act), T] sampled for rows act: counters, clock,
        score, timeouts, stopping. Returns (rows that continue, their new record codes)."""
        self.steps[act] += 1
        self.cnt["transitions"] += len(act)
        for k, v in check_transitions(tg).items():
            self.cnt[k] += v
        phase = tg[:, TI["phase"]]
        is_end = phase == PH["END"]
        raw_next = S.apply_transition(self.raw[act], tg)
        delta = S.score_delta_value(tg[:, TI["score_delta"]]).astype(np.int64)
        if not cfg["score_on_end"]:
            delta = np.where(is_end, 0, delta)
            raw_next[is_end, 1] = self.raw[act[is_end], 1]
        self.points[act] += np.abs(delta)
        clock0 = ~is_end & (raw_next[:, 0] <= 0) & (phase != PH["TRY"])
        trunc = ~is_end & ~clock0 & (self.steps[act] >= max_steps)
        cont = ~(is_end | clock0 | trunc)
        self.cnt["ended_by_end"] += int(is_end.sum())
        self.cnt["ended_by_clock"] += int(clock0.sum())
        self.cnt["truncated_max_steps"] += int(trunc.sum())
        self.final[act] = raw_next[:, 1]
        self.raw[act] = raw_next
        self.active[act[~cont]] = False
        rows = act[cont]
        codes = np.zeros((0, len(S.STATE_FIELDS)), dtype=np.int64)
        if len(rows):
            codes = S.next_state_codes(raw_next[cont], tg[cont]).astype(np.int64)
            self.appended[rows] += 1
            self.kickoffs[rows] += codes[:, SI["phase"]] == PH["KICKOFF"]
        return rows, codes

    def append(self, buf_t: torch.Tensor, rows: np.ndarray, codes: np.ndarray,
               max_len: int) -> np.ndarray:
        """Write each continuing row's new record into its window buffer, dropping the oldest
        record of a full window. Returns the rows whose window was shifted."""
        device = buf_t.device
        full = rows[self.lens[rows] >= max_len]
        if len(full):
            ft = _to_dev(full, device)
            buf_t[ft, :-1] = buf_t[ft, 1:].clone()
        pos = np.minimum(self.lens[rows], max_len - 1)
        buf_t[_to_dev(rows, device), _to_dev(pos, device)] = _to_dev(codes, device)
        self.lens[rows] = np.minimum(self.lens[rows] + 1, max_len)
        return full

    def result(self, track: bool) -> dict:
        out = {"final_diff": self.final, "steps": self.steps, "counters": self.cnt}
        if track:
            out.update({"appended": self.appended, "points": self.points,
                        "kickoffs": self.kickoffs})
        return out


def _sample_rows(model, h_rows: torch.Tensor, chunk: int, temp: float, gen, mask_pad: bool):
    """Sampled transitions for h_rows [R, d], in chunks of cfg["sample_batch"] rows (one
    call per play in practice; the chunking fixes the order of the random draws, and both
    encoder paths use it the same way)."""
    parts = [model.sample_transition(h_rows[a : a + chunk], temp, gen, mask_pad)
             for a in range(0, len(h_rows), chunk)]
    return torch.cat(parts).cpu().numpy()


def _timing(timer: PhaseTimer, loop_s: float, **extra) -> dict:
    out = {f"{k}_s": v for k, v in timer.split(loop_s).items()}
    out.update(loop_s=loop_s, iterations=timer.iters, method=timer.method(), **extra)
    return out


@torch.no_grad()
def _run_recompute(model, header, buf, lens, raw, cfg, deadline_ts, gen, track=False,
                   prefix_key=None) -> dict | None:
    """Reference path: every simulated play re-encodes each row's whole window."""
    device = device_of(model)
    max_len, max_steps, chunk = model.max_len, int(cfg["max_steps"]), int(cfg["sim_batch"])
    temp, mask_pad = float(cfg["temperature"]), bool(cfg["mask_pad_codes"])
    sample_batch = int(cfg.get("sample_batch", 16384))
    sort_len = bool(cfg.get("sort_by_length", True))
    bk = _Book(lens, raw)
    buf_t = torch.as_tensor(buf, device=device)
    head_t = torch.as_tensor(header.astype(np.int64), device=device)
    enabled, dtype, _ = amp_settings(device)
    timer = PhaseTimer(device, int(cfg.get("timing_sync_every", 8)))
    t0 = time.perf_counter()
    while bk.active.any():
        if time.time() > deadline_ts:
            return None
        timer.start()
        act = np.flatnonzero(bk.active)
        # Encode in chunks; with sort_by_length the chunks hold rows of similar context
        # length, so less padding is computed. The causal encoder ignores right padding, and
        # sampling below uses the original row order and chunks, so the draws are unchanged.
        enc_order = np.argsort(bk.lens[act], kind="stable") if sort_len else np.arange(len(act))
        hl_all = torch.empty(len(act), model.cfg["d_model"], device=device)
        for a in range(0, len(act), chunk):
            pos_rows = enc_order[a : a + chunk]
            rows = act[pos_rows]
            rows_t = torch.as_tensor(rows, device=device)
            n = int(bk.lens[rows].max())
            with torch.autocast(device.type, dtype=dtype, enabled=enabled):
                h = model.encode(head_t[rows_t], buf_t[rows_t, :n])
            ln = torch.as_tensor(bk.lens[rows] - 1, device=device)
            hl_all[torch.as_tensor(pos_rows, device=device)] = h[
                torch.arange(len(rows), device=device), ln
            ].float()
            del h
        timer.mark("encode")
        tg = _sample_rows(model, hl_all, sample_batch, temp, gen, mask_pad)
        timer.mark("decode")
        rows, codes = bk.advance(act, tg, cfg, max_steps)
        if len(rows):
            bk.append(buf_t, rows, codes, max_len)
        timer.mark("bookkeeping")
    out = bk.result(track)
    out["timing"] = _timing(timer, time.perf_counter() - t0, path="recompute")
    return out


@torch.no_grad()
def _run_cached(model, header, buf, lens, raw, cfg, deadline_ts, gen, track=False,
                prefix_key=None) -> dict | None:
    """Incremental decoding: each simulated play encodes ONE new record per row.

    Cache layout. Rows with the same prefix_key (all rollouts of one query) have the same
    header and the same records except the last one (whose action may be forced or sampled
    per row). Those shared positions are encoded once per key and stored once (the prefix
    cache); each row's own cache holds its last real record and every record it simulates.
    Attention reads the shared part once per key (`model.cascade_attention`), so its cost
    grows with the simulated plays, not with the real game so far.

    A row whose window is full (max_len records) and must drop its oldest record changes
    every position embedding, so no cache can be rolled exactly: from then on that row
    re-encodes its window each play (the reference path). Rows that finish leave dead slots
    in the own cache, which is compacted once a quarter is dead. Sampling uses the same rows
    and order as `_run_recompute`, so with equal hidden states the draws are identical."""
    device = device_of(model)
    max_len, max_steps, chunk = model.max_len, int(cfg["max_steps"]), int(cfg["sim_batch"])
    temp, mask_pad = float(cfg["temperature"]), bool(cfg["mask_pad_codes"])
    sample_batch = int(cfg.get("sample_batch", 16384))
    sort_len = bool(cfg.get("sort_by_length", True))
    enabled, dtype, _ = amp_settings(device)
    c = model.cfg
    d, nl, nh = int(c["d_model"]), int(c["n_layers"]), int(c["n_heads"])
    dh = d // nh
    r = len(lens)
    bk = _Book(lens, raw)
    buf_t = torch.as_tensor(buf, device=device)
    head_t = torch.as_tensor(header.astype(np.int64), device=device)
    timer = PhaseTimer(device, int(cfg.get("timing_sync_every", 8)))
    t0 = time.perf_counter()
    timer.start()

    def autocast():
        return torch.autocast(device.type, dtype=dtype, enabled=enabled)

    def dev(a):
        return _to_dev(a, device)

    # ---- groups of rows that share a prefix, and each row's cell in the [groups] grid
    key = np.arange(r) if prefix_key is None else np.asarray(prefix_key)
    _, first, base = np.unique(key, return_index=True, return_inverse=True)
    base = np.asarray(base).reshape(-1)
    n_base = len(first)
    counts = np.bincount(base, minlength=n_base)
    groups = int(counts.max())
    by_base = np.argsort(base, kind="stable")
    within = np.empty(r, dtype=np.int64)
    within[by_base] = np.arange(r) - np.repeat(np.cumsum(counts) - counts, counts)
    plen = bk.lens[first]  # shared positions per key: header + all records but the last
    p_max, own_cap = int(plen.max()), max_len + 1 - int(plen.min())

    # ---- caches. Zero-filled, not empty: a masked key gets attention weight 0, but 0 x NaN
    # from uninitialised memory would still be NaN in the weighted sum of values.
    cache_dtype = dtype if enabled else torch.float32
    kp = torch.zeros(nl, n_base, nh, p_max, dh, dtype=cache_dtype, device=device)
    vp = torch.zeros_like(kp)
    ko = torch.zeros(nl, r, nh, own_cap, dh, dtype=cache_dtype, device=device)
    vo = torch.zeros_like(ko)
    u_order = np.argsort(plen, kind="stable") if sort_len else np.arange(n_base)
    for a in range(0, n_base, chunk):
        us = u_order[a : a + chunk]
        src_t = dev(first[us])
        n = int(plen[us].max()) - 1  # shared records (the header is the extra position)
        with autocast():
            _, kv = model.encode(head_t[src_t], buf_t[src_t, :n], return_kv=True)
        for layer, (k, v) in enumerate(kv):
            kp[layer, dev(us), :, : n + 1] = k.to(cache_dtype)
            vp[layer, dev(us), :, : n + 1] = v.to(cache_dtype)
        del kv

    # ---- slots of the own cache (compacted as rows finish)
    slot = np.arange(r)  # slot of each row, -1 once it has left the cache
    slot_row = np.arange(r)  # row in each slot, -1 when the slot is dead
    slot_cell = base * groups + within
    slot_plen = plen[base].copy()
    slot_pos = bk.lens.copy()  # window position of each slot's newest token (header = 0)
    m = r
    p_ar = torch.arange(p_max, device=device)

    def step(codes_m: torch.Tensor) -> torch.Tensor:
        """encode_step for all m slots; their newest tokens are at slot_pos[:m]."""
        own = slot_pos[:m] - slot_plen[:m]
        n_own = int(own.max()) + 1
        own_t, plen_t = dev(own), dev(slot_plen[:m])
        o_ar = torch.arange(n_own, device=device)
        ok = torch.cat([p_ar.unsqueeze(0) < plen_t.unsqueeze(1),
                        o_ar.unsqueeze(0) <= own_t.unsqueeze(1)], dim=1)
        bias = torch.zeros(ok.shape, device=device).masked_fill_(~ok, float("-inf"))
        with autocast():
            h = model.encode_step(codes_m, dev(slot_pos[:m]), ko[:, :m], vo[:, :m], n_own,
                                  cache_pos=own_t,
                                  prefix=(kp, vp, dev(slot_cell[:m]), groups, bias))
        return h.float()

    # each row's last real record: own index 0, at window position lens
    h_last = step(buf_t[torch.arange(r, device=device), dev(bk.lens - 1)])
    timer.mark("prefill")
    n_fallback = n_compact = 0
    first_iter = True  # the prefill is part of the first iteration's timing
    while bk.active.any():
        if time.time() > deadline_ts:
            return None
        if not first_iter:
            timer.start()
        first_iter = False
        act = np.flatnonzero(bk.active)
        tg = _sample_rows(model, h_last[dev(act)], sample_batch, temp, gen, mask_pad)
        timer.mark("decode")
        rows, codes = bk.advance(act, tg, cfg, max_steps)
        stopped = act[~bk.active[act]]
        full = bk.append(buf_t, rows, codes, max_len) if len(rows) else rows
        for gone in (stopped, full):  # finished rows and rows whose window shifted
            live = gone[slot[gone] >= 0]
            slot_row[slot[live]] = -1
            slot[live] = -1
        timer.mark("bookkeeping")
        if len(full):  # window shifted: re-encode the whole window (exact)
            n_fallback += len(full)
            for a in range(0, len(full), chunk):
                fr_t = dev(full[a : a + chunk])
                with autocast():
                    h = model.encode(head_t[fr_t], buf_t[fr_t, :max_len])
                h_last[fr_t] = h[:, -1].float()
                del h
        keep = slot[rows] >= 0
        cached = rows[keep]
        if len(cached):
            s = slot[cached]
            slot_pos[s] = bk.lens[cached]
            s_t = dev(s)
            new = torch.zeros(m, len(S.STATE_FIELDS), dtype=torch.long, device=device)
            new[s_t] = dev(codes[keep])
            h_last[dev(cached)] = step(new)[s_t]
        dead = int((slot_row[:m] < 0).sum())
        if dead and dead >= m // 4:  # compact: live slots to the front, in place
            alive = np.flatnonzero(slot_row[:m] >= 0)
            k_alive = len(alive)
            if k_alive:
                a_t = dev(alive)
                for layer in range(nl):
                    ko[layer, :k_alive] = ko[layer, :m].index_select(0, a_t)
                    vo[layer, :k_alive] = vo[layer, :m].index_select(0, a_t)
            moved = slot_row[alive]
            slot[moved] = np.arange(k_alive)
            for arr in (slot_cell, slot_plen, slot_pos):
                arr[:k_alive] = arr[alive]
            slot_row[:k_alive] = moved
            slot_row[k_alive:m] = -1
            m = k_alive
            n_compact += 1
        timer.mark("encode")
    out = bk.result(track)
    elem = 2 if enabled else 4
    out["timing"] = _timing(
        timer, time.perf_counter() - t0, path="kv_cache", rows=r, shared_prefixes=n_base,
        fallback_row_steps=n_fallback, compactions=n_compact,
        cache_mib=(ko.numel() + kp.numel()) * elem * 2 / 2**20,
        cache_mib_per_rollout=model.kv_bytes_per_row(elem) / 2**20,
    )
    return out


def _merge_timing(parts: list[dict]) -> dict:
    out: dict = {}
    for t in parts:
        for k, v in (t or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = out.get(k, 0) + v
            else:
                out.setdefault(k, v)
    if parts and "cache_mib_per_rollout" in out:
        out["cache_mib_per_rollout"] = parts[0]["cache_mib_per_rollout"]
    return out


@torch.no_grad()
def _run(
    model: WorldModel,
    header: np.ndarray,
    buf: np.ndarray,
    lens: np.ndarray,
    raw: np.ndarray,
    cfg: dict,
    deadline_ts: float,
    gen: torch.Generator,
    track: bool = False,
    prefix_key: np.ndarray | None = None,
) -> dict | None:
    """Roll every row to the end of its game. Returns None if the deadline passes first.

    cfg["kv_cache"] picks the encoder path (True: incremental decoding; False: re-encode the
    window every play). Rows with the same prefix_key must have the same header and the same
    records except the last (the rollouts of one query); the cache stores that part once.
    Rows are simulated in groups of `live_rows` (memory), one group at a time. out["timing"]
    splits the wall time into prefill, encode, decode (sampling) and bookkeeping."""
    run_group = _run_cached if cfg.get("kv_cache", True) else _run_recompute
    lim = live_rows(model, cfg)
    r = len(lens)
    parts = []
    oom_retries = 0
    a = 0
    while a < r:
        sl = slice(a, min(r, a + lim))
        pk = None if prefix_key is None else np.asarray(prefix_key)[sl]
        oom = False
        try:
            out = run_group(model, header[sl], buf[sl], lens[sl], raw[sl], cfg, deadline_ts,
                            gen, track, pk)
        except RuntimeError as err:  # torch.OutOfMemoryError subclasses RuntimeError
            if not _is_oom(err) or sl.stop - sl.start <= 1:
                raise
            oom = True
        if oom:
            # Out of GPU memory: retry this group in halves (outside the except block, so
            # the traceback and its tensors are released first). The generator has moved
            # on, so the draws differ from an uninterrupted run (still valid samples).
            if device_of(model).type == "cuda":
                torch.cuda.empty_cache()
            lim = max(1, (sl.stop - sl.start) // 2)
            oom_retries += 1
            continue
        if out is None:
            return None
        parts.append(out)
        a = sl.stop
    if len(parts) == 1:
        merged = parts[0]
    else:
        merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]
                  if isinstance(parts[0][k], np.ndarray)}
        merged["counters"] = {k: sum(p["counters"][k] for p in parts) for k in COUNTER_KEYS}
        merged["timing"] = _merge_timing([p["timing"] for p in parts])
    merged["timing"]["oom_retries"] = oom_retries
    _accumulate_timing(merged["timing"])
    return merged


ROLLOUT_TIMING: dict = {}


def _accumulate_timing(t: dict) -> None:
    for k, v in t.items():
        if k == "cache_mib_per_rollout":
            ROLLOUT_TIMING[k] = v
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            ROLLOUT_TIMING[k] = ROLLOUT_TIMING.get(k, 0) + v


def rollout_timing(reset: bool = False) -> dict:
    """Seconds by phase summed over every rollout since the last reset (this process)."""
    out = dict(ROLLOUT_TIMING)
    if reset:
        ROLLOUT_TIMING.clear()
    return out


def _resolve_action(forced_action) -> int | str | None:
    if forced_action is None or forced_action == GO:
        return forced_action
    if isinstance(forced_action, str):
        return A[forced_action]
    code = int(forced_action)
    if not 0 < code < len(S.ACTIONS):
        raise ValueError(f"action code {code} is outside the action vocabulary")
    return code


def rollout(
    model: WorldModel,
    split: S.Split,
    query_idx,
    forced_action,
    n_rollouts: int,
    cfg: dict | None,
    deadline_ts: float,
    rng: np.random.Generator,
) -> dict:
    """Home win probability by simulation from each query record.

    forced_action: None (keep the real action), an action code or name, or "GO" (record i's
    action is RUN or PASS, sampled from the model's own probabilities renormalised over the two).
    Queries are processed in groups of up to `live_rows` rollouts (cfg["sim_max_live"],
    lowered to what the key/value cache fits); if the deadline passes, the unfinished
    queries come back as NaN with done=False. res["timing"] splits the time by phase.
    """
    c = merge_cfg({**model.cfg, **(cfg or {})})
    model.eval()
    device = device_of(model)
    idx = np.atleast_1d(np.asarray(query_idx, dtype=np.int64))
    nq, n = len(idx), int(n_rollouts)
    fa = _resolve_action(forced_action)
    res = {
        "query_idx": idx,
        "home_wp": np.full(nq, np.nan),
        "home_wp_se": np.full(nq, np.nan),
        "mean_steps": np.full(nq, np.nan),
        "done": np.zeros(nq, dtype=bool),
        "forced_action": forced_action,
        "counters": dict.fromkeys(COUNTER_KEYS, 0),
        "start_actions": {},  # action of the query record, over all rollouts
    }
    gen = _torch_generator(rng, device)
    games = _game_of(split, idx)
    per_group = max(1, live_rows(model, c) // max(1, n))
    timings = []
    t0 = time.time()
    for a in range(0, nq, per_group):
        if time.time() > deadline_ts:
            break
        q = np.arange(a, min(nq, a + per_group))
        buf, lens = _window_buffer(split, idx[q], model.max_len)
        buf = np.repeat(buf, n, axis=0)
        lens = np.repeat(lens, n)
        if fa == GO:
            p = _go_probs(model, split, idx[q], c)
            u = rng.random(len(q) * n)
            pick = np.where(u < np.repeat(p[:, 0], n), S.GO_ACTIONS[0], S.GO_ACTIONS[1])
            buf[np.arange(len(lens)), lens - 1, SI["action"]] = pick
        elif fa is not None:
            buf[np.arange(len(lens)), lens - 1, SI["action"]] = fa
        starts = buf[np.arange(len(lens)), lens - 1, SI["action"]]
        for code, k in zip(*np.unique(starts, return_counts=True), strict=True):
            name = _action_name(code)
            res["start_actions"][name] = res["start_actions"].get(name, 0) + int(k)
        header = np.repeat(split.header[games[q]], n, axis=0)
        raw = np.repeat(split.raw[idx[q]], n, axis=0)
        # The rollouts of one query share the header and every record but the last (whose
        # action may differ), so the key/value cache stores that history once per query.
        prefix_key = np.repeat(np.arange(len(q)), n)
        out = _run(model, header, buf, lens, raw, c, deadline_ts, gen, prefix_key=prefix_key)
        if out is None:
            break
        timings.append(out["timing"])
        o = S.outcome(out["final_diff"]).reshape(len(q), n)
        res["home_wp"][q] = o.mean(1)
        res["home_wp_se"][q] = o.std(1, ddof=1) / np.sqrt(n) if n > 1 else np.nan
        res["mean_steps"][q] = out["steps"].reshape(len(q), n).mean(1)
        res["done"][q] = True
        for k, v in out["counters"].items():
            res["counters"][k] += v
    res["seconds"] = time.time() - t0
    res["timing"] = _merge_timing(timings)
    return res


def _action_name(code: int) -> str:
    return S.ACTIONS[int(code)]


def _choice_of(code: int) -> str:
    if int(code) in S.GO_ACTIONS:
        return GO
    return _action_name(code)


def plan_fourth_downs(
    model: WorldModel,
    split: S.Split,
    idx,
    n_rollouts: int,
    cfg: dict | None,
    deadline_ts: float,
    rng: np.random.Generator,
    action_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """For every fourth-down PLAY record in idx: win probability of GO, FG and PUNT for the
    team with the ball, standard errors, best action, margin and a "too close" flag (margin
    under 2 standard errors of the difference between the best and second-best action).

    action_mask: optional bool [len(idx), 3] for GO, FG, PUNT, aligned with idx as passed.
    A False entry means that action is not simulated: its wp and se are NaN and its done
    flag stays False. "best" is chosen among the allowed actions only, and a state is done
    when every allowed action is done. Without a mask every action is simulated, with the
    same rollout calls and random draws as before the argument existed."""
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    st = split.states[idx]
    keep = (st[:, SI["down"]] == 4) & (st[:, SI["phase"]] == PH["PLAY"])
    if action_mask is None:
        allowed = np.ones((int(keep.sum()), 3), dtype=bool)
    else:
        am = np.asarray(action_mask, dtype=bool)
        if am.shape != (len(idx), 3):
            raise ValueError(f"action_mask has shape {am.shape}, expected ({len(idx)}, 3)")
        allowed = am[keep]
    idx = idx[keep]
    poss = split.states[idx, SI["poss"]]
    home_has_ball = poss == S.POSS_CODE["HOME"]
    choices = (GO, "FG", "PUNT")
    games = _game_of(split, idx)
    df = pd.DataFrame(
        {
            "idx": idx,
            "game_id": [split.game_ids[g] for g in games],
            "season": split.seasons[games],
            "poss": np.where(home_has_ball, "HOME", "AWAY"),
            "ydstogo": split.states[idx, SI["ydstogo"]],
            "yardline_100": split.states[idx, SI["yardline"]],
            "clock_seconds": split.raw[idx, 0],
            "score_diff_poss": np.where(home_has_ball, 1, -1) * split.raw[idx, 1],
        }
    )
    # The three actions are simulated chunk by chunk of states (one rollout group each), so
    # a deadline leaves whole states finished rather than GO for all and PUNT for none.
    c = merge_cfg({**model.cfg, **(cfg or {})})
    per = max(1, int(c["sim_max_live"]) // max(1, int(n_rollouts)))
    counters = {ch: dict.fromkeys(COUNTER_KEYS, 0) for ch in choices}
    for ch in choices:
        key = ch.lower()
        df[f"wp_{key}"] = np.nan
        df[f"se_{key}"] = np.nan
        df[f"done_{key}"] = False
    for a in range(0, len(idx), per):
        sl = slice(a, min(len(idx), a + per))
        for j, ch in enumerate(choices):
            ok = allowed[sl, j]
            if not ok.any():
                continue  # masked for every state in this chunk: nothing simulated
            r = rollout(model, split, idx[sl][ok], ch, n_rollouts, cfg, deadline_ts, rng)
            key = ch.lower()
            rows_ok = df.index[sl][ok]
            hb = home_has_ball[sl][ok]
            df.loc[rows_ok, f"wp_{key}"] = np.where(hb, r["home_wp"], 1.0 - r["home_wp"])
            df.loc[rows_ok, f"se_{key}"] = r["home_wp_se"]
            df.loc[rows_ok, f"done_{key}"] = r["done"]
            for k, v in r["counters"].items():
                counters[ch][k] += v
    wps = df[[f"wp_{c.lower()}" for c in choices]].to_numpy()
    ses = df[[f"se_{c.lower()}" for c in choices]].to_numpy()
    sim_done = df[[f"done_{c.lower()}" for c in choices]].to_numpy().astype(bool)
    done = (sim_done | ~allowed).all(1) & allowed.any(1)
    for j, ch in enumerate(choices):
        df[f"allowed_{ch.lower()}"] = allowed[:, j]
    order = np.argsort(-np.nan_to_num(wps, nan=-1.0), axis=1)
    rows = np.arange(len(df))
    b, s2 = order[:, 0], order[:, 1] if len(choices) > 1 else order[:, 0]
    margin = wps[rows, b] - wps[rows, s2]
    margin_se = np.sqrt(ses[rows, b] ** 2 + ses[rows, s2] ** 2)
    df["best"] = np.where(done, np.array(choices, dtype=object)[b], None)
    df["margin"] = np.where(done, margin, np.nan)
    df["margin_se"] = np.where(done, margin_se, np.nan)
    df["too_close"] = done & (margin < 2 * margin_se)
    real = split.states[idx, SI["action"]]
    df["real_action"] = [_action_name(a) for a in real]
    df["real_choice"] = [_choice_of(a) for a in real]
    df["agrees"] = done & (df["best"] == df["real_choice"])
    df["done"] = done
    df.attrs["counters"] = counters
    return df


def wp_by_simulation(
    model: WorldModel,
    split: S.Split,
    idx,
    n_rollouts: int,
    cfg: dict | None,
    deadline_ts: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulated home win probability from each record with its real action, plus the actual
    home result (1/0.5/0) for calibration."""
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    r = rollout(model, split, idx, None, n_rollouts, cfg, deadline_ts, rng)
    games = _game_of(split, idx)
    df = pd.DataFrame(
        {
            "idx": idx,
            "game_id": [split.game_ids[g] for g in games],
            "season": split.seasons[games],
            "phase": [S.PHASES[p] for p in split.states[idx, SI["phase"]]],
            "poss": [S.POSS[p] for p in split.states[idx, SI["poss"]]],
            "clock_seconds": split.raw[idx, 0],
            "score_diff": split.raw[idx, 1],
            "home_wp_sim": r["home_wp"],
            "home_wp_se": r["home_wp_se"],
            "mean_steps": r["mean_steps"],
            "home_result": S.outcome(split.final_diff[games]),
            "done": r["done"],
        }
    )
    # run.py saves attrs["counters"] per chunk, so the phase timing travels with it.
    df.attrs["counters"] = {**r["counters"], "timing_s": r["timing"]}
    df.attrs["timing"] = r["timing"]
    return df


def real_game_stats(split: S.Split, games: np.ndarray | None = None) -> dict:
    """League statistics of real games, defined the same way as for simulated games."""
    g_all = np.arange(len(split.offsets) - 1) if games is None else np.asarray(games)
    n_rec = np.diff(split.offsets)[g_all]
    pts_rec = np.abs(S.score_delta_value(split.targets[:, TI["score_delta"]]))
    kick_rec = split.states[:, SI["phase"]] == PH["KICKOFF"]
    seg = split.offsets[:-1]
    pts = np.add.reduceat(pts_rec, seg)[g_all] if len(pts_rec) else np.zeros(len(g_all))
    kicks = np.add.reduceat(kick_rec.astype(np.int64), seg)[g_all]
    fd = split.final_diff[g_all].astype(np.int64)
    last = split.offsets[1:][g_all] - 1
    chain_diff = split.raw[last, 1].astype(np.int64)  # score before the final transition
    return {
        "records_per_game": float(n_rec.mean()),
        "points_per_game": float(pts.mean()),  # all transitions that carry a score delta
        "final_margin_sd": float(fd.std()),
        "tie_share": float((fd == 0).mean()),
        "kickoffs_per_game": float(kicks.mean()),
        "home_win_share": float(S.outcome(fd).mean()),
        "final_play_points_share": float((fd != chain_diff).mean()),
    }


def simulate_games(
    model: WorldModel,
    split: S.Split,
    n_games: int,
    cfg: dict | None,
    deadline_ts: float,
    rng: np.random.Generator,
) -> dict:
    """Simulate whole games from the header and opening record of sampled real games and
    compare league statistics with the same real games."""
    c = merge_cfg({**model.cfg, **(cfg or {})})
    model.eval()
    device = device_of(model)
    n_real = len(split.offsets) - 1
    games = rng.choice(n_real, size=n_games, replace=n_games > n_real)
    starts = split.offsets[games]
    gen = _torch_generator(rng, device)
    per = live_rows(model, c)
    parts = []
    for a in range(0, n_games, per):
        g = games[a : a + per]
        s = starts[a : a + per]
        buf, lens = _window_buffer(split, s, model.max_len)
        out = _run(model, split.header[g], buf, lens, split.raw[s], c, deadline_ts, gen, track=True)
        if out is None:
            break
        parts.append(out)
    if not parts:
        return {"n_games": 0, "sim": {}, "real": real_game_stats(split, games), "counters": {}}
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("final_diff", "appended", "points")}
    cat["kickoffs"] = np.concatenate([p["kickoffs"] for p in parts])
    done = len(cat["final_diff"])
    first_kick = (split.states[starts[:done], SI["phase"]] == PH["KICKOFF"]).astype(np.int64)
    fd = cat["final_diff"]
    sim = {
        "records_per_game": float((cat["appended"] + 1).mean()),
        "points_per_game": float(cat["points"].mean()),
        "final_margin_sd": float(fd.std()),
        "tie_share": float((fd == 0).mean()),
        "kickoffs_per_game": float((cat["kickoffs"] + first_kick).mean()),
        "home_win_share": float(S.outcome(fd).mean()),
    }
    counters = dict.fromkeys(COUNTER_KEYS, 0)
    for p in parts:
        for k, v in p["counters"].items():
            counters[k] += v
    real = real_game_stats(split, games[:done])
    table = pd.DataFrame({"real": pd.Series(real), "simulated": pd.Series(sim)}, index=list(real))
    return {"n_games": done, "sim": sim, "real": real, "table": table, "counters": counters,
            "timing": _merge_timing([p["timing"] for p in parts])}


__all__ = [
    "COUNTER_KEYS",
    "GO",
    "TARGET_NAMES",
    "cache_rows",
    "check_transitions",
    "live_rows",
    "plan_fourth_downs",
    "real_game_stats",
    "rollout",
    "rollout_timing",
    "simulate_games",
    "wp_by_simulation",
]
