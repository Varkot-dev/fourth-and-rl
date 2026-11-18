"""A causal transformer "world model" over game records, with an autoregressive field decoder.

Sequence per game: position 0 is the header token (sum of header field embeddings), then one
token per record (sum of that record's STATE_FIELDS embeddings). Every token also gets a
learned positional embedding. A pre-LayerNorm causal transformer turns record t into h_t.

The FIELD DECODER predicts the transition t -> t+1 one field at a time, in TARGET_FIELDS order:
    logits_j = MLP_j([h_t ; e_0(y_0) + ... + e_{j-1}(y_{j-1})])
where e_k is an embedding table for target field k and y_k is the true code (training, teacher
forcing) or the sampled code (generation). Each head outputs only its own field's vocabulary.

Two execution paths each, with the same parameters and the same math:
- decoder: in training every true field is known, so all ten heads run in one batched pass
  (`decode_logits_padded`); `decode_logits` is the one-head-at-a-time reference. Sampling is
  sequential by nature (`sample_transition`).
- encoder: `encode` runs the whole window; `encode_step` adds ONE new record per row using a
  key/value cache of the earlier positions (incremental decoding for rollouts). Causal
  attention makes the two identical: position p only ever attends to positions 0..p.

Standalone: imports only torch, numpy and the sibling `schema` module.
"""

from __future__ import annotations

import numpy as np
import schema as S
import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_CFG: dict = {
    # model
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 4,
    "d_ff": 1024,
    "dec_hidden": 256,
    "dropout": 0.1,
    "max_len": 256,  # records of context; longer games keep their LAST max_len records
    # training
    "seed": 0,
    "epochs": 30,
    "batch_games": 32,
    "eval_batch_games": 128,
    "parallel_decoder": True,  # training loss: all field heads in one pass (same math)
    "data_on_device": "auto",  # keep the pre-built split tensors on the GPU when they fit
    "train_mem_frac": 0.6,  # share of free GPU memory a training step may plan to use
    "timing_sync_every": 8,  # CUDA: synchronize every k-th step to split time by phase
    "lr": 3e-4,
    "min_lr_frac": 0.1,
    "weight_decay": 0.1,
    "betas": (0.9, 0.95),
    "warmup_steps": 300,
    "grad_clip": 1.0,
    "patience": 4,
    "stop_after_epoch": None,  # stop cleanly after this many epochs (schedule unchanged)
    # simulation
    "temperature": 1.0,
    "max_steps": 400,  # transitions per rollout before it is counted as truncated
    "sim_batch": 1024,  # rollouts per full-window transformer pass (prefill, fallback)
    "sample_batch": 16384,  # rows per sampling call (one call per simulated play)
    "sim_max_live": 8192,  # rollouts held at once (queries are processed in groups)
    "mask_pad_codes": True,  # never sample code 0 for fields where 0 means padding
    "score_on_end": True,  # add the sampled score delta of the END transition (final play)
    "sort_by_length": True,  # simulator: encode rows sorted by context length (less padding)
    "kv_cache": True,  # simulator: incremental decoding (False = recompute the window each play)
    "kv_cache_mem_frac": 0.4,  # CUDA: share of total GPU memory the key/value cache may use
    "kv_cache_cpu_gb": 4.0,  # CPU: memory the key/value cache may use
}

# Target fields where code 0 is a real value ("not a down"), not padding.
ZERO_IS_VALUE = frozenset({"down", "ydstogo"})
TARGET_NAMES = tuple(name for name, _ in S.TARGET_FIELDS)
PHASE_T = S.TARGET_INDEX["phase"]
# Fields that carry a real value on a game's LAST record (target phase END): the final play's
# points and seconds. Every other field is padding there: it never counts in the loss, and its
# decoder input is forced to code 0 both in training and in sampling, so the score_delta and
# elapsed heads see the same prefix (END, poss=0) in both.
END_FIELDS = frozenset({"phase", "score_delta", "elapsed"})
NOT_END_T = tuple(j for j, n in enumerate(TARGET_NAMES) if n not in END_FIELDS)
TARGET_SIZES = tuple(n for _, n in S.TARGET_FIELDS)
VMAX = max(TARGET_SIZES)  # padded vocabulary width of the batched decoder


def merge_cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULT_CFG)
    out.update(cfg or {})
    return out


class Block(nn.Module):
    """Pre-LayerNorm transformer block with causal scaled-dot-product attention."""

    def __init__(self, d: int, n_heads: int, d_ff: int, p: float) -> None:
        super().__init__()
        assert d % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.p = p
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor, return_kv: bool = False):
        """x [b, n, d] -> [b, n, d]; with return_kv also this layer's keys and values, each
        [b, heads, n, d_head] (what a key/value cache stores)."""
        b, n, d = x.shape
        qkv = self.qkv(self.ln1(x)).view(b, n, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [b, heads, n, d_head]
        a = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.p if self.training else 0.0, is_causal=True
        )
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, n, d)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return (x, k, v) if return_kv else x

    def step(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        pos: torch.Tensor,
        n_keys: int,
        prefix: tuple | None = None,
    ) -> torch.Tensor:
        """One new token per row (inference only).

        x [R, d] is the new token; k_cache, v_cache [R, heads, C, d_head] are each row's own
        cache, written in place at index pos [R]. Without prefix the token attends to cache
        indices 0..pos[r] (its whole history), exactly what the causal mask lets that position
        see in `forward`; n_keys >= max(pos) + 1.

        With prefix = (k_pre, v_pre [U, heads, P, d_head], cell [R], groups, bias [R, P + n_keys])
        the history is split: keys and values shared by a group of rows (the real game so far,
        identical for every rollout of a query) are stored ONCE per group, and the row's own
        cache holds only what it simulated. The token attends to both in one softmax (see
        `cascade_attention`), the same function as attending to the concatenated history."""
        r, d = x.shape
        qkv = self.qkv(self.ln1(x)).view(r, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.unbind(1)  # each [R, heads, d_head]
        rows = torch.arange(r, device=x.device)
        k_cache[rows, :, pos] = k.to(k_cache.dtype)
        v_cache[rows, :, pos] = v.to(v_cache.dtype)
        if prefix is None:
            keys = torch.arange(n_keys, device=x.device)
            allowed = (keys.unsqueeze(0) <= pos.unsqueeze(1))[:, None, None, :]  # [R,1,1,n]
            a = F.scaled_dot_product_attention(
                q.unsqueeze(2).to(k_cache.dtype),
                k_cache[:, :, :n_keys],
                v_cache[:, :, :n_keys],
                attn_mask=allowed,
            )
        else:
            k_pre, v_pre, cell, groups, bias = prefix
            a = cascade_attention(
                q, k_cache[:, :, :n_keys], v_cache[:, :, :n_keys], k_pre, v_pre, cell, groups, bias
            )
        x = x + self.proj(a.reshape(r, d))
        return x + self.ff(self.ln2(x))


def cascade_attention(
    q: torch.Tensor,
    k_own: torch.Tensor,
    v_own: torch.Tensor,
    k_pre: torch.Tensor,
    v_pre: torch.Tensor,
    cell: torch.Tensor,
    groups: int,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Attention of one query per row over a shared prefix plus the row's own keys.

    q [R, heads, dh]; k_own, v_own [R, heads, n, dh]; k_pre, v_pre [U, heads, P, dh] shared by
    the rows of a group; cell [R] = u * groups + (row's place in group u), so the queries of
    a group form one [groups, dh] matrix and the prefix keys are read once per group (one
    batched matrix product) instead of once per row; bias [R, P + n] is 0 for a key the row
    may see and -inf otherwise. Returns softmax([q k_pre ; q k_own] / sqrt(dh) + bias) @
    [v_pre ; v_own], [R, heads, dh]: one softmax over both parts, so exactly attention over
    the concatenated keys (up to the order of floating-point additions)."""
    r, h, dh = q.shape
    u, _, p, _ = k_pre.shape
    qd = q.to(k_pre.dtype)
    grid = qd.new_zeros(u * groups, h, dh)
    grid[cell] = qd
    grid = grid.view(u, groups, h, dh).transpose(1, 2)  # [U, heads, groups, dh]
    s_pre = torch.matmul(grid, k_pre.transpose(-1, -2))  # [U, heads, groups, P]
    s_pre = s_pre.transpose(1, 2).reshape(u * groups, h, p)[cell]  # [R, heads, P]
    s_own = torch.matmul(qd.unsqueeze(2), k_own.transpose(-1, -2)).squeeze(2)  # [R, heads, n]
    scores = torch.cat([s_pre, s_own], dim=-1).float() * dh**-0.5 + bias.unsqueeze(1)
    w = torch.softmax(scores, dim=-1).to(v_pre.dtype)
    w_grid = w.new_zeros(u * groups, h, p)
    w_grid[cell] = w[..., :p]
    o_pre = torch.matmul(w_grid.view(u, groups, h, p).transpose(1, 2), v_pre)  # [U,h,groups,dh]
    o_pre = o_pre.transpose(1, 2).reshape(u * groups, h, dh)[cell]
    o_own = torch.matmul(w[..., p:].unsqueeze(2), v_own).squeeze(2)
    return o_pre + o_own


class WorldModel(nn.Module):
    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__()
        self.cfg = merge_cfg(cfg)
        c = self.cfg
        d = c["d_model"]
        self.max_len = int(c["max_len"])
        self.state_emb = nn.ModuleList(nn.Embedding(n, d) for _, n in S.STATE_FIELDS)
        self.header_emb = nn.ModuleList(nn.Embedding(n, d) for _, n in S.HEADER_FIELDS)
        self.pos_emb = nn.Embedding(self.max_len + 1, d)  # 0 = header, 1..max_len = records
        self.drop = nn.Dropout(c["dropout"])
        self.blocks = nn.ModuleList(
            Block(d, c["n_heads"], c["d_ff"], c["dropout"]) for _ in range(c["n_layers"])
        )
        self.ln_f = nn.LayerNorm(d)
        # embeddings of target fields 0..T-2 (the last field is never a decoder input)
        self.tgt_emb = nn.ModuleList(nn.Embedding(n, d) for _, n in S.TARGET_FIELDS[:-1])
        h = c["dec_hidden"]
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Linear(h, n))
            for _, n in S.TARGET_FIELDS
        )
        self.apply(self._init)
        # Constants of the batched decoder and loss, kept on the model's device so that no
        # step copies anything from the host (a host-to-GPU copy of pageable memory waits for
        # the GPU). Not saved in checkpoints.
        pad = torch.full((len(TARGET_SIZES), VMAX), float("-inf"))
        for j, n in enumerate(TARGET_SIZES):
            pad[j, :n] = 0.0
        self.register_buffer("_logit_pad", pad, persistent=False)
        no_pad = pad.clone()  # sampling with mask_pad: never code 0 where 0 means padding
        for j, name in enumerate(TARGET_NAMES):
            if name not in ZERO_IS_VALUE:
                no_pad[j, 0] = float("-inf")
        self.register_buffer("_logit_no_pad", no_pad, persistent=False)
        flags = {
            "_is_phase": [j == PHASE_T for j in range(len(TARGET_NAMES))],
            "_zero_is_value": [n in ZERO_IS_VALUE for n in TARGET_NAMES],
            "_end_field": [n in END_FIELDS for n in TARGET_NAMES],
        }
        for name, v in flags.items():
            self.register_buffer(name, torch.tensor(v, dtype=torch.bool), persistent=False)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ encoder
    def embed_records(self, states: torch.Tensor) -> torch.Tensor:
        """states [..., S] -> sum of the record's field embeddings [..., d]."""
        return sum(emb(states[..., k]) for k, emb in enumerate(self.state_emb))

    def encode(self, header: torch.Tensor, states: torch.Tensor, return_kv: bool = False):
        """header [B, H], states [B, L, S] (L <= max_len, right-padded) -> h [B, L, d].

        With return_kv, also every layer's (keys, values), each [B, heads, L + 1, d_head]
        (position 0 is the header token): the prefill of a key/value cache."""
        b, n, _ = states.shape
        if n > self.max_len:
            raise ValueError(f"{n} records exceed max_len={self.max_len}; cut to the last ones")
        x0 = sum(emb(header[:, k]) for k, emb in enumerate(self.header_emb)).unsqueeze(1)
        xr = self.embed_records(states)
        x = torch.cat([x0, xr], dim=1) + self.pos_emb.weight[: n + 1].unsqueeze(0)
        x = self.drop(x)
        kv = []
        for blk in self.blocks:
            if return_kv:
                x, k, v = blk(x, return_kv=True)
                kv.append((k, v))
            else:
                x = blk(x)
        h = self.ln_f(x)[:, 1:]
        return (h, kv) if return_kv else h

    def encode_step(
        self,
        states: torch.Tensor,
        pos: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        n_keys: int,
        cache_pos: torch.Tensor | None = None,
        prefix: tuple | None = None,
    ) -> torch.Tensor:
        """Incremental encode (inference only): one new record per row.

        states [R, S] codes of the new record, pos [R] its position in the window (1..max_len;
        0 is the header), k_cache / v_cache [layers, R, heads, C, d_head] the rows' own caches;
        this record's keys and values are written at cache_pos (default pos). Without prefix
        the own cache holds positions 0..pos-1. With prefix (per layer: k_pre, v_pre
        [layers, U, heads, P, d_head], cell, groups, bias; see `Block.step`) the shared part
        of the history is in k_pre / v_pre and the own cache starts after it. Returns h [R, d],
        equal (to float rounding) to `encode` of the whole window at that position."""
        if self.training:
            raise RuntimeError("encode_step is for inference; call model.eval() first")
        cache_pos = pos if cache_pos is None else cache_pos
        x = self.embed_records(states) + self.pos_emb.weight[pos]
        for layer, blk in enumerate(self.blocks):
            pre = None
            if prefix is not None:
                k_pre, v_pre, cell, groups, bias = prefix
                pre = (k_pre[layer], v_pre[layer], cell, groups, bias)
            x = blk.step(x, k_cache[layer], v_cache[layer], cache_pos, n_keys, pre)
        return self.ln_f(x)

    def kv_bytes_per_row(self, element_bytes: int) -> int:
        """Memory of one rollout's key/value cache over a full window: layers x 2 x
        (max_len + 1) x d_model x element_bytes."""
        c = self.cfg
        return int(c["n_layers"]) * 2 * (self.max_len + 1) * int(c["d_model"]) * element_bytes

    # ------------------------------------------------------------------ decoder
    def _prefix_sums(self, targets: torch.Tensor) -> torch.Tensor:
        """targets [..., T] -> s [..., T, d], s_j = sum of embeddings of fields 0..j-1."""
        e = torch.stack([emb(targets[..., k]) for k, emb in enumerate(self.tgt_emb)], dim=-2)
        c = torch.cumsum(e, dim=-2)
        return torch.cat([torch.zeros_like(c[..., :1, :]), c], dim=-2)

    def decode_logits(self, h: torch.Tensor, targets: torch.Tensor) -> list[torch.Tensor]:
        """Teacher-forced logits for every target field, one head at a time (the reference
        for `decode_logits_padded`). h [..., d], targets [..., T]."""
        s = self._prefix_sums(targets)
        return [head(torch.cat([h, s[..., j, :]], dim=-1)) for j, head in enumerate(self.heads)]

    def decode_logits_padded(self, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """All ten heads in one batched pass (teacher forcing: every true field is known).

        h [N, d], targets [N, T] -> logits [T, N, VMAX] float32; columns past a field's own
        vocabulary are -inf, so a softmax over the row equals the softmax over that field's
        vocabulary. Head j's first layer W_j [hidden, 2d] acts on [h ; s_j], which is split
        as W_j[:, :d] h + W_j[:, d:] s_j: one matrix product for the h part of all heads and
        one batched product for the prefix-sum parts. Same parameters and same function as
        `decode_logits`; only the order of floating-point additions differs."""
        d = h.shape[-1]
        s = self._prefix_sums(targets)  # [N, T, d]
        w1 = torch.stack([hd[0].weight for hd in self.heads])  # [T, hidden, 2d]
        b1 = torch.stack([hd[0].bias for hd in self.heads])  # [T, hidden]
        t, hid, _ = w1.shape
        zh = torch.matmul(h, w1[:, :, :d].reshape(t * hid, d).t())  # [N, T * hidden]
        zh = zh.view(-1, t, hid).transpose(0, 1)  # [T, N, hidden]
        zs = torch.bmm(s.transpose(0, 1), w1[:, :, d:].transpose(1, 2))  # [T, N, hidden]
        z = F.gelu(zh + zs + b1.to(zh.dtype).unsqueeze(1))
        w2 = torch.stack(
            [F.pad(hd[2].weight, (0, 0, 0, VMAX - hd[2].weight.shape[0])) for hd in self.heads]
        )  # [T, VMAX, hidden]
        b2 = torch.stack([F.pad(hd[2].bias, (0, VMAX - hd[2].bias.shape[0])) for hd in self.heads])
        logits = torch.baddbmm(b2.unsqueeze(1).to(z.dtype), z, w2.transpose(1, 2).to(z.dtype))
        return logits.float() + self._logit_pad.unsqueeze(1)

    def field_logits(self, h: torch.Tensor, prefix: torch.Tensor, j: int) -> torch.Tensor:
        """Logits of field j given h [R, d] and the codes of fields 0..j-1 in prefix [R, >=j]."""
        s = torch.zeros_like(h)
        for k in range(j):
            s = s + self.tgt_emb[k](prefix[:, k])
        return self.heads[j](torch.cat([h, s], dim=-1))

    @torch.no_grad()
    def sample_transition(
        self,
        h: torch.Tensor,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        mask_pad: bool = True,
    ) -> torch.Tensor:
        """Sample one transition per row of h [R, d] -> codes [R, T] (long).

        The heads run in float32 even inside an autocast region: bf16 logits are rounded to
        about 1/64 of their size, which would bias the sampled probabilities by around 1%.
        On rows whose sampled phase is END, fields outside END_FIELDS are set to code 0
        (the value they have on every END row in training).

        Sampling is Gumbel-max: argmax(logits / temperature + g) with g = -log(-log(u)), u
        uniform, is a draw from softmax(logits / temperature). The noise for all fields is
        drawn in one call, and the h half of every head's first layer is one matrix product
        (only the prefix-sum half depends on the fields sampled before), so a call launches
        about 11 small kernels per field.
        """
        r = h.shape[0]
        t = len(TARGET_NAMES)
        out = torch.zeros(r, t, dtype=torch.long, device=h.device)
        with torch.autocast(h.device.type, enabled=False):
            h = h.float()
            d = h.shape[-1]
            w1 = torch.stack([hd[0].weight for hd in self.heads])  # [T, hidden, 2d]
            hid = w1.shape[1]
            b1 = torch.stack([hd[0].bias for hd in self.heads]).reshape(-1)
            zh = torch.addmm(b1, h, w1[:, :, :d].reshape(t * hid, d).t())  # [R, T * hidden]
            bias = self._logit_no_pad if mask_pad else self._logit_pad
            if temperature > 0:
                u = torch.rand((r, t, VMAX), generator=generator, device=h.device)
                noise = -torch.log(-torch.log(u.clamp_(1e-10, 1.0 - 1e-7))) + bias
            s = torch.zeros_like(h)
            is_end = None
            for j, head in enumerate(self.heads):
                n = TARGET_SIZES[j]
                z = torch.addmm(zh[:, j * hid : (j + 1) * hid], s, w1[j, :, d:].t())
                logits = F.linear(F.gelu(z), head[2].weight, head[2].bias)
                if temperature <= 0:
                    code = (logits + bias[j, :n]).argmax(-1)
                else:
                    code = (logits / temperature + noise[:, j, :n]).argmax(-1)
                if j == PHASE_T:
                    is_end = code == S.PHASE["END"]
                elif TARGET_NAMES[j] not in END_FIELDS:
                    code = code.masked_fill(is_end, 0)
                out[:, j] = code
                if j < len(self.tgt_emb):
                    s = s + self.tgt_emb[j](code)
        return out

    # ------------------------------------------------------------------ loss
    @staticmethod
    def end_prefix(targets: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        """Targets with every field outside END_FIELDS set to 0 on last records."""
        tg = targets.clone()
        tg[..., list(NOT_END_T)] = tg[..., list(NOT_END_T)] * (~last).unsqueeze(-1).to(tg.dtype)
        return tg

    @staticmethod
    def target_masks(targets: torch.Tensor, mask: torch.Tensor, last: torch.Tensor) -> list:
        """Per field, which targets count in the loss ([B, L] bool each).

        A target counts when its record is real and its code is not padding. Code 0 is
        padding for every field except down and ydstogo ("not a down"). On a game's last
        record only END_FIELDS count: phase (END) always, score_delta and elapsed when
        records.py filled them (code > 0).
        """
        out = []
        for j, name in enumerate(TARGET_NAMES):
            y = targets[..., j]
            if j == PHASE_T:
                m = mask
            elif name in ZERO_IS_VALUE:
                m = mask & ~last
            else:
                m = mask & (y != 0)
                if name not in END_FIELDS:
                    m = m & ~last
            out.append(m)
        return out

    def end_prefix_fast(self, targets: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        """`end_prefix` without list indexing (which copies an index list to the GPU and so
        waits for it): a broadcast multiply by a keep mask held on the device."""
        keep = (~last).unsqueeze(-1) | self._end_field
        return targets * keep.to(targets.dtype)

    def target_mask_tensor(
        self, targets: torch.Tensor, mask: torch.Tensor, last: torch.Tensor
    ) -> torch.Tensor:
        """`target_masks` for all fields at once: [..., T] bool, the same rule."""
        not_last = (~last).unsqueeze(-1)
        other = (targets != 0) & (self._end_field | not_last)
        rule = torch.where(self._zero_is_value, not_last, other) | self._is_phase
        return mask.unsqueeze(-1) & rule

    def loss(
        self, batch: dict, denom: torch.Tensor | None = None, parallel: bool | None = None
    ) -> tuple[torch.Tensor, dict]:
        """Sum over fields of the mean cross-entropy per target (see target_masks).

        batch: header [B,H], states [B,L,S], targets [B,L,T], mask [B,L] (real record),
        last [B,L] (last record of its game). denom [T], if given, replaces the per-field
        target counts in the means (used to split one batch into exact micro-batches).
        parallel (default cfg["parallel_decoder"]): all heads in one batched pass and one
        cross-entropy call; False runs the one-head-at-a-time reference. Same loss either way.
        Returns (total, {"fields": {name: mean NLL tensor}, "sums": [T], "counts": [T]}).
        """
        if parallel is None:
            parallel = bool(self.cfg.get("parallel_decoder", True))
        h = self.encode(batch["header"], batch["states"])
        mask, last = batch["mask"], batch["last"]
        if parallel:
            return self._loss_parallel(h, batch["targets"], mask, last, denom)
        tg = self.end_prefix(batch["targets"], last)
        logits = self.decode_logits(h, tg)
        masks = self.target_masks(tg, mask, last)
        total = h.new_zeros((), dtype=torch.float32)
        fields, sums, counts = {}, [], []
        for j, (name, _) in enumerate(S.TARGET_FIELDS):
            y = tg[..., j]
            m = masks[j]
            ce = F.cross_entropy(logits[j].float().flatten(0, 1), y.flatten(), reduction="none")
            ce = ce.view_as(y) * m
            n = m.sum()
            sm = ce.sum()
            d = n.clamp(min=1) if denom is None else denom[j].clamp(min=1)
            mean = sm / d
            fields[name] = mean
            sums.append(sm.detach())
            counts.append(n.detach())
            total = total + mean
        return total, {"fields": fields, "sums": torch.stack(sums), "counts": torch.stack(counts)}

    def _loss_parallel(self, h, targets, mask, last, denom):
        t = len(TARGET_NAMES)
        tg = self.end_prefix_fast(targets, last)
        tf = tg.reshape(-1, t)
        logits = self.decode_logits_padded(h.reshape(-1, h.shape[-1]), tf)  # [T, N, VMAX]
        y = tf.t()  # [T, N]
        ce = F.cross_entropy(logits.reshape(-1, VMAX), y.reshape(-1), reduction="none")
        m = self.target_mask_tensor(tg, mask, last).reshape(-1, t).t()  # [T, N] bool
        sums = (ce.view(t, -1) * m).sum(1)
        counts = m.sum(1)
        d = counts.clamp(min=1) if denom is None else denom.clamp(min=1)
        means = sums / d
        fields = {name: means[j] for j, name in enumerate(TARGET_NAMES)}
        return means.sum(), {"fields": fields, "sums": sums.detach(), "counts": counts.detach()}


def window_start(game_start: int, i: int, max_len: int) -> int:
    """First record of the context window that ends at record i (inclusive)."""
    return max(game_start, i + 1 - max_len)


def set_seed(seed: int) -> None:
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def param_summary(model: WorldModel) -> str:
    n = model.count_params()
    return f"{n:,} parameters ({n / 1e6:.2f} M), {4 * n / 2**20:.1f} MiB in float32"


__all__ = [
    "DEFAULT_CFG",
    "ZERO_IS_VALUE",
    "END_FIELDS",
    "WorldModel",
    "merge_cfg",
    "window_start",
    "set_seed",
    "device_of",
    "param_summary",
]
