"""A causal transformer "world model" over game records, with an autoregressive field decoder.

Sequence per game: position 0 is the header token (sum of header field embeddings), then one
token per record (sum of that record's STATE_FIELDS embeddings). Every token also gets a
learned positional embedding. A pre-LayerNorm causal transformer turns record t into h_t.

The FIELD DECODER predicts the transition t -> t+1 one field at a time, in TARGET_FIELDS order:
    logits_j = MLP_j([h_t ; e_0(y_0) + ... + e_{j-1}(y_{j-1})])
where e_k is an embedding table for target field k and y_k is the true code (training, teacher
forcing) or the sampled code (generation). Each head outputs only its own field's vocabulary.

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
    "eval_batch_games": 64,
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
    "sim_batch": 1024,  # rollouts per transformer forward pass
    "sim_max_live": 8192,  # rollouts held at once (queries are processed in groups)
    "mask_pad_codes": True,  # never sample code 0 for fields where 0 means padding
    "score_on_end": False,  # add the sampled score delta on the END transition (see notes)
}

# Target fields where code 0 is a real value ("not a down"), not padding.
ZERO_IS_VALUE = frozenset({"down", "ydstogo"})
TARGET_NAMES = tuple(name for name, _ in S.TARGET_FIELDS)
PHASE_T = S.TARGET_INDEX["phase"]


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.ln1(x)).view(b, n, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [b, heads, n, d_head]
        a = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.p if self.training else 0.0, is_causal=True
        )
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, n, d)))
        return x + self.drop(self.ff(self.ln2(x)))


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
    def encode(self, header: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """header [B, H], states [B, L, S] (L <= max_len, right-padded) -> h [B, L, d]."""
        b, n, _ = states.shape
        if n > self.max_len:
            raise ValueError(f"{n} records exceed max_len={self.max_len}; cut to the last ones")
        x0 = sum(emb(header[:, k]) for k, emb in enumerate(self.header_emb)).unsqueeze(1)
        xr = sum(emb(states[:, :, k]) for k, emb in enumerate(self.state_emb))
        x = torch.cat([x0, xr], dim=1) + self.pos_emb.weight[: n + 1].unsqueeze(0)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)[:, 1:]

    # ------------------------------------------------------------------ decoder
    def _prefix_sums(self, targets: torch.Tensor) -> torch.Tensor:
        """targets [..., T] -> s [..., T, d], s_j = sum of embeddings of fields 0..j-1."""
        e = torch.stack([emb(targets[..., k]) for k, emb in enumerate(self.tgt_emb)], dim=-2)
        c = torch.cumsum(e, dim=-2)
        return torch.cat([torch.zeros_like(c[..., :1, :]), c], dim=-2)

    def decode_logits(self, h: torch.Tensor, targets: torch.Tensor) -> list[torch.Tensor]:
        """Teacher-forced logits for every target field. h [..., d], targets [..., T]."""
        s = self._prefix_sums(targets)
        return [head(torch.cat([h, s[..., j, :]], dim=-1)) for j, head in enumerate(self.heads)]

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
        """Sample one transition per row of h [R, d] -> codes [R, T] (long)."""
        r = h.shape[0]
        out = torch.zeros(r, len(S.TARGET_FIELDS), dtype=torch.long, device=h.device)
        s = torch.zeros_like(h)
        for j, head in enumerate(self.heads):
            logits = head(torch.cat([h, s], dim=-1)).float()
            if mask_pad and TARGET_NAMES[j] not in ZERO_IS_VALUE:
                logits[:, 0] = -float("inf")
            if temperature <= 0:
                code = logits.argmax(-1)
            else:
                u = torch.rand(logits.shape, generator=generator, device=h.device)
                gumbel = -torch.log(-torch.log(u.clamp_(1e-10, 1.0 - 1e-7)))
                code = (logits / temperature + gumbel).argmax(-1)
            out[:, j] = code
            if j < len(self.tgt_emb):
                s = s + self.tgt_emb[j](code)
        return out

    # ------------------------------------------------------------------ loss
    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Sum over fields of the mean cross-entropy per target.

        batch: header [B,H], states [B,L,S], targets [B,L,T], mask [B,L] (real record),
        last [B,L] (last record of its game). A target counts when its record is real and its
        code is not padding. Code 0 is padding for every field except down and ydstogo, where it
        means "not a down"; on a game's last record those two are padding too. The phase of the
        last record (END) is always a real target.
        Returns (total, {"fields": {name: mean NLL tensor}, "sums": [T], "counts": [T]}).
        """
        h = self.encode(batch["header"], batch["states"])
        tg = batch["targets"]
        logits = self.decode_logits(h, tg)
        mask, last = batch["mask"], batch["last"]
        total = h.new_zeros((), dtype=torch.float32)
        fields, sums, counts = {}, [], []
        for j, (name, _) in enumerate(S.TARGET_FIELDS):
            y = tg[..., j]
            if name in ZERO_IS_VALUE:
                m = mask & ~last
            elif j == PHASE_T:
                m = mask
            else:
                m = mask & (y != 0)
            ce = F.cross_entropy(logits[j].float().flatten(0, 1), y.flatten(), reduction="none")
            ce = ce.view_as(y) * m
            n = m.sum()
            sm = ce.sum()
            mean = sm / n.clamp(min=1)
            fields[name] = mean
            sums.append(sm.detach())
            counts.append(n.detach())
            total = total + mean
        return total, {"fields": fields, "sums": torch.stack(sums), "counts": torch.stack(counts)}


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
    "WorldModel",
    "merge_cfg",
    "window_start",
    "set_seed",
    "device_of",
    "param_summary",
]
