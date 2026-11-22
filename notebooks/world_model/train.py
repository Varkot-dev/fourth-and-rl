"""Batching, training loop (AdamW, warmup + cosine, AMP, resume, early stopping) and NLL.

Speed notes (see the first Colab run in results/world_model/colab_t4_run_summary.md, which
trained on the CPU because the model was never moved to the GPU):
- `train_model` moves the model to cfg["device"].
- Each split is pre-built ONCE as padded int16 tensors (games x window x fields) and kept on
  the GPU when it fits; a batch is an index operation (`tensor_batch`), not a Python loop
  (`collate`, kept as the reference).
- No per-step host synchronisation: losses are summed on the device and read once per epoch,
  record counts come from host-side lengths, and on CUDA the optimizer is fused AdamW, which
  takes the fp16 GradScaler's inf check on the device instead of calling .item().
- Every epoch's wall time is split into batch construction, host-to-device, forward,
  backward plus optimizer, validation and checkpointing, logged and kept in the history.
"""

from __future__ import annotations

import copy
import math
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import schema as S
import torch
from model import TARGET_NAMES, VMAX, WorldModel, device_of, merge_cfg


# ---------------------------------------------------------------------- batching (reference)
def collate(split: S.Split, games: np.ndarray, max_len: int) -> dict:
    """Pad the given games into one batch, one game per row, each cut to its last max_len."""
    starts = split.offsets[games]
    ends = split.offsets[games + 1]
    starts = np.maximum(starts, ends - max_len)
    lens = (ends - starts).astype(np.int64)
    b, n = len(games), int(lens.max())
    states = np.zeros((b, n, len(S.STATE_FIELDS)), dtype=np.int64)
    targets = np.zeros((b, n, len(S.TARGET_FIELDS)), dtype=np.int64)
    mask = np.zeros((b, n), dtype=bool)
    last = np.zeros((b, n), dtype=bool)
    for r, (a, e, k) in enumerate(zip(starts, ends, lens, strict=True)):
        states[r, :k] = split.states[a:e]
        targets[r, :k] = split.targets[a:e]
        mask[r, :k] = True
        last[r, k - 1] = True
    return {
        "header": torch.from_numpy(split.header[games].astype(np.int64)),
        "states": torch.from_numpy(states),
        "targets": torch.from_numpy(targets),
        "mask": torch.from_numpy(mask),
        "last": torch.from_numpy(last),
        "games": torch.from_numpy(np.asarray(games, dtype=np.int64)),
    }


def make_batches(
    split: S.Split,
    batch_games: int,
    max_len: int,
    rng: np.random.Generator | None,
) -> Iterator[dict]:
    """Yield padded batches of whole games. rng=None keeps the split's order (evaluation)."""
    n_games = len(split.offsets) - 1
    order = np.arange(n_games) if rng is None else rng.permutation(n_games)
    for b in range(0, n_games, batch_games):
        yield collate(split, order[b : b + batch_games], max_len)


def to_device(batch: dict, device: torch.device) -> dict:
    nb = device.type == "cuda"
    return {k: v.to(device, non_blocking=nb) for k, v in batch.items()}


# ---------------------------------------------------------------------- pre-built tensors
@dataclass
class TensorSplit:
    """A split built once as padded tensors, one row per game, each game cut to its LAST
    max_len records (as `collate` does). A batch is then an index operation."""

    states: torch.Tensor  # [G, W, S] int16, right-padded with 0 (W = longest kept game)
    targets: torch.Tensor  # [G, W, T] int16
    header: torch.Tensor  # [G, H] int16
    lens: torch.Tensor  # [G] int64, records kept per game
    lens_np: np.ndarray  # host copy: batch widths and record counts need no device sync
    device: torch.device

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.states, self.targets, self.header, self.lens)
        )


def prebuild_split(split: S.Split, max_len: int, device: torch.device | str = "cpu") -> TensorSplit:
    device = torch.device(device)
    starts = np.asarray(split.offsets[:-1], dtype=np.int64)
    ends = np.asarray(split.offsets[1:], dtype=np.int64)
    starts = np.maximum(starts, ends - max_len)
    lens = (ends - starts).astype(np.int64)
    w = max(1, int(lens.max()) if len(lens) else 1)
    ar = np.arange(w)
    valid = ar[None, :] < lens[:, None]
    src = np.where(valid, starts[:, None] + ar[None, :], 0)
    states = (split.states[src] * valid[..., None]).astype(np.int16)
    targets = (split.targets[src] * valid[..., None]).astype(np.int16)
    pin = device.type == "cpu" and torch.cuda.is_available()

    def put(a: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(a))
        return t.pin_memory() if pin else t.to(device)

    return TensorSplit(
        states=put(states),
        targets=put(targets),
        header=put(split.header.astype(np.int16)),
        lens=put(lens),
        lens_np=lens,
        device=device,
    )


def _gather(ts: TensorSplit, games_np: np.ndarray, games_t: torch.Tensor) -> dict:
    n = max(1, int(ts.lens_np[games_np].max()))
    return {
        "header": ts.header[games_t],
        "states": ts.states[games_t, :n],
        "targets": ts.targets[games_t, :n],
        "lens": ts.lens[games_t],
        "games": games_t,
    }


def _finish(raw: dict) -> dict:
    lens = raw["lens"]
    ar = torch.arange(raw["states"].shape[1], device=lens.device)
    return {
        "header": raw["header"].long(),
        "states": raw["states"].long(),
        "targets": raw["targets"].long(),
        "mask": ar.unsqueeze(0) < lens.unsqueeze(1),
        "last": ar.unsqueeze(0) == (lens - 1).unsqueeze(1),
        "games": raw["games"].long(),
    }


def tensor_batch(ts: TensorSplit, games, device: torch.device | None = None) -> dict:
    """The batch `collate(split, games, max_len)` builds, from the pre-built tensors."""
    games_np = np.asarray(games, dtype=np.int64)
    raw = _gather(ts, games_np, torch.as_tensor(games_np, device=ts.device))
    if device is not None and torch.device(device) != ts.device:
        raw = to_device(raw, torch.device(device))
    return _finish(raw)


def _storage_device(cfg: dict, device: torch.device, n_bytes: int) -> torch.device:
    """Keep the pre-built tensors on the GPU when they take under a quarter of free memory
    (the full training split is about 30 MB)."""
    if device.type != "cuda":
        return device
    want = cfg.get("data_on_device", "auto")
    if want is False or want == "never":
        return torch.device("cpu")
    if want == "auto":
        free, _ = torch.cuda.mem_get_info(device)
        if n_bytes > 0.25 * free:
            return torch.device("cpu")
    return device


def _split_bytes(split: S.Split, max_len: int) -> int:
    lens = np.minimum(np.diff(split.offsets), max_len)
    w = int(lens.max()) if len(lens) else 1
    return len(lens) * w * (len(S.STATE_FIELDS) + len(S.TARGET_FIELDS)) * 2


# ---------------------------------------------------------------------- timing
DURATION_KEYS = (
    "batch_s",
    "h2d_s",
    "forward_s",
    "backward_opt_s",
    "train_loop_s",
    "validation_s",
    "checkpoint_s",
    "epoch_s",
)


class PhaseTimer:
    """Wall-clock split of a loop into named phases.

    On CPU every phase is timed directly. On CUDA kernels run asynchronously, so a phase's
    CPU time is not its GPU time: every `sync_every`-th iteration (starting with the
    second, after warm-up) is synchronised at each phase boundary, and the shares measured
    on those iterations split the loop's measured total. The other iterations run without
    any synchronisation."""

    def __init__(self, device: torch.device, sync_every: int = 8) -> None:
        self.cuda = device.type == "cuda"
        self.sync_every = max(1, int(sync_every))
        self.direct: dict[str, float] = {}
        self.sampled: dict[str, float] = {}
        self.iters = 0
        self.sampled_iters = 0
        self._sync = False
        self._t = time.perf_counter()

    def start(self) -> None:
        self.iters += 1
        self._sync = self.cuda and self.iters % self.sync_every == 2 % self.sync_every
        if self._sync:
            torch.cuda.synchronize()
            self.sampled_iters += 1
        self._t = time.perf_counter()

    def mark(self, name: str) -> None:
        if self._sync:
            torch.cuda.synchronize()
        t = time.perf_counter()
        dt = t - self._t
        self.direct[name] = self.direct.get(name, 0.0) + dt
        if self._sync:
            self.sampled[name] = self.sampled.get(name, 0.0) + dt
        self._t = t

    def split(self, total_s: float) -> dict[str, float]:
        """Seconds per phase: measured on CPU; on CUDA, total_s split by the shares of the
        synchronised iterations (or the CPU-side times if none was sampled)."""
        src = self.sampled if (self.cuda and self.sampled_iters) else self.direct
        tot = sum(src.values())
        if not self.cuda or tot <= 0:
            return dict(self.direct)
        return {k: total_s * v / tot for k, v in src.items()}

    def method(self) -> str:
        if not self.cuda:
            return "measured per phase (CPU, synchronous)"
        return (
            f"total measured; split by phase from {self.sampled_iters} of {self.iters} "
            "steps synchronised at every phase boundary"
        )


# ---------------------------------------------------------------------- precision
def amp_settings(device: torch.device) -> tuple[bool, torch.dtype, bool]:
    """(enabled, dtype, use_grad_scaler). bf16 only on GPUs with native support (sm80+)."""
    if device.type != "cuda":
        return False, torch.float32, False
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:
        return True, torch.bfloat16, False
    return True, torch.float16, True


def train_bytes_per_token(cfg: dict, amp: bool) -> float:
    """Rough activation memory of one padded token in a training step (forward activations
    kept for backward, with a 1.5x margin for temporaries and allocator slack)."""
    d, layers, ff, hid = cfg["d_model"], cfg["n_layers"], cfg["d_ff"], cfg["dec_hidden"]
    a = 2 if amp else 4
    t = len(S.TARGET_FIELDS)
    enc = layers * (d * (16 + 8 * a) + ff * 3 * a)  # layer norms in fp32, the rest in AMP
    dec = t * (d * (4 + a) + hid * 3 * a + VMAX * 12)  # prefix sums, hidden, padded logits
    emb = 4 * d * (len(S.STATE_FIELDS) + 4)
    return 1.5 * (enc + dec + emb)


def _planned_micro(cfg: dict, batch_games: int, width: int, device: torch.device, amp: bool) -> int:
    """Micro-batches per step so one micro-batch fits in cfg["train_mem_frac"] of free GPU
    memory (1 on CPU). The out-of-memory fallback still doubles it if this is too low."""
    if device.type != "cuda":
        return 1
    free, _ = torch.cuda.mem_get_info(device)
    need = batch_games * width * train_bytes_per_token(cfg, amp)
    budget = float(cfg.get("train_mem_frac", 0.6)) * free
    k = 1
    while k < batch_games and need / k > budget:
        k *= 2
    return k


# ---------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate_nll(
    model: WorldModel,
    split: S.Split,
    cfg: dict | None = None,
    prebuilt: TensorSplit | None = None,
) -> dict:
    """Per-field mean NLL (nats per target) and total NLL per record over a split.

    Games are batched by length (less padding); sums stay on the device until the end."""
    c = merge_cfg({**model.cfg, **(cfg or {})})
    device = device_of(model)
    enabled, dtype, _ = amp_settings(device)
    was_training = model.training
    model.eval()
    ts = prebuilt
    if ts is None:
        ts = prebuild_split(
            split, model.max_len, _storage_device(c, device, _split_bytes(split, model.max_len))
        )
    t = len(S.TARGET_FIELDS)
    sums = torch.zeros(t, dtype=torch.float64, device=device)
    counts = torch.zeros(t, dtype=torch.float64, device=device)
    order = np.argsort(ts.lens_np, kind="stable")
    order_t = torch.as_tensor(order, device=ts.device)
    bg = max(1, int(c["eval_batch_games"]))
    for b in range(0, len(order), bg):
        raw = _gather(ts, order[b : b + bg], order_t[b : b + bg])
        if ts.device != device:
            raw = to_device(raw, device)
        with torch.autocast(device.type, dtype=dtype, enabled=enabled):
            _, info = model.loss(_finish(raw))
        sums += info["sums"].double()
        counts += info["counts"].double()
    model.train(was_training)
    sums, counts = sums.cpu(), counts.cpu()
    n_records = int(ts.lens_np.sum())
    fields = {n: float(sums[j] / max(counts[j], 1.0)) for j, n in enumerate(TARGET_NAMES)}
    return {
        "total": float(sums.sum() / max(n_records, 1)),  # nats per record, all fields
        "sum_of_field_means": float(sum(fields.values())),
        "fields": fields,
        "n_records": n_records,
    }


# ---------------------------------------------------------------------- training
def _lr_factor(step: int, progress: float, cfg: dict) -> float:
    """Linear warmup over warmup_steps, then cosine from 1 to min_lr_frac over all epochs."""
    warm = min(1.0, (step + 1) / max(1, cfg["warmup_steps"]))
    p = min(1.0, max(0.0, progress / max(1, cfg["epochs"])))
    floor = cfg["min_lr_frac"]
    return warm * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def _param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        is_matrix = p.dim() >= 2 and "emb" not in name
        (decay if is_matrix else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _match_fused(opt: torch.optim.Optimizer, fused: bool) -> None:
    """After loading an optimizer state saved with the other setting: fused AdamW keeps its
    step counters on the parameter's device, the unfused one on the CPU."""
    for group in opt.param_groups:
        group["fused"] = True if fused else None
        if fused:
            group["foreach"] = None
        for p in group["params"]:
            st = opt.state.get(p)
            if st and torch.is_tensor(st.get("step")):
                dev = p.device if fused else torch.device("cpu")
                st["step"] = st["step"].to(device=dev, dtype=torch.float32)


def _save(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _is_oom(err: BaseException) -> bool:
    oom = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom is not None and isinstance(err, oom)) or "out of memory" in str(err).lower()


def load_checkpoint(path: str, cfg_override: dict | None = None, device: str | None = None):
    """Rebuild the BEST model stored at path. Returns (model, checkpoint dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = WorldModel({**ck["cfg"], **(cfg_override or {})})
    model.load_state_dict(ck["model"])
    if device is not None:
        model.to(device)
    model.eval()
    return model, ck


def _micro_slices(batch: dict, k: int, lens: np.ndarray | None = None) -> list[dict]:
    """Split a batch into k row groups, each trimmed to its own longest game (the causal
    encoder makes right padding irrelevant, so the results do not change). lens (host
    copy of the rows' record counts) avoids reading the mask back from the GPU."""
    b = batch["mask"].shape[0]
    out = []
    for rows in np.array_split(np.arange(b), min(k, b)):
        if len(rows) == 0:
            continue
        sl = slice(int(rows[0]), int(rows[-1]) + 1)
        if lens is not None:
            n = max(1, int(np.asarray(lens)[sl].max()))
        else:
            n = max(1, int(batch["mask"][sl].sum(1).max()))
        part = {}
        for key, v in batch.items():
            v = v[sl]
            if key in ("states", "targets", "mask", "last"):
                v = v[:, :n]
            part[key] = v
        out.append(part)
    return out


def _full_counts(model: WorldModel, batch: dict) -> torch.Tensor:
    """Per-field target counts of a whole batch (the denominators of its loss)."""
    tg = model.end_prefix_fast(batch["targets"], batch["last"])
    m = model.target_mask_tensor(tg, batch["mask"], batch["last"])
    return m.reshape(-1, m.shape[-1]).sum(0).float()


def _fresh_scaler(scaler: torch.amp.GradScaler, enabled: bool) -> torch.amp.GradScaler:
    """A scaler with the same scale and growth tracker but no per-step state (an OOM between
    unscale_ and update would otherwise make the next unscale_ raise)."""
    new = torch.amp.GradScaler("cuda", enabled=enabled)
    if enabled:
        new.load_state_dict(scaler.state_dict())
    return new


def make_optimizer(
    model: WorldModel, c: dict, device: torch.device, log: Callable[[str], None] = print
) -> tuple[torch.optim.Optimizer, bool]:
    """AdamW; fused on CUDA (one kernel for all parameters, and with the fp16 GradScaler the
    inf check stays on the device). Some torch builds reject fused=True (older versions,
    or parameters it cannot handle); then the default implementation is used. Returns
    (optimizer, fused)."""
    groups = _param_groups(model, c["weight_decay"])
    kw = {"lr": c["lr"], "betas": tuple(c["betas"])}
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(groups, fused=True, **kw), True
        except (RuntimeError, TypeError, ValueError) as err:
            log(f"fused AdamW is not available here ({type(err).__name__}: {err}); "
                "using the default AdamW")
            groups = _param_groups(model, c["weight_decay"])
    return torch.optim.AdamW(groups, **kw), False


def _place(model: WorldModel, want: str | None, log: Callable[[str], None]) -> torch.device:
    """Move the model to cfg["device"] (falls back to the CPU when CUDA is missing)."""
    if want:
        dev = torch.device(want)
        if dev.type == "cuda" and not torch.cuda.is_available():
            log("cfg asks for cuda but no GPU is available; training on the CPU")
            dev = torch.device("cpu")
        model.to(dev)
    return device_of(model)


def train_model(
    model: WorldModel,
    train_split: S.Split,
    val_split: S.Split,
    cfg: dict,
    ckpt_path: str,
    deadline_ts: float,
    log: Callable[[str], None] = print,
) -> dict:
    """Train with AdamW, warmup + cosine, gradient clipping and AMP on CUDA.

    The model is moved to cfg["device"] when given. One checkpoint file at ckpt_path holds:
    "model" = best weights so far (by validation NLL), "current_model" + optimizer + scaler +
    RNG states + epoch + step + position inside the epoch (to resume exactly), cfg, history.
    If ckpt_path exists, training resumes from it. Validation NLL is evaluated after every
    epoch; training stops on early stopping (patience), at cfg["epochs"], at
    cfg["stop_after_epoch"], or cleanly at the first batch after time.time() > deadline_ts
    (a mid-epoch stop saves the position and resumes there). On out-of-memory the SAME step
    is retried with the batch split into twice as many micro-batches whose gradients are
    accumulated with the full batch's denominators, so the data order, batch composition and
    schedule do not change; on CUDA the starting number of micro-batches is planned from
    free memory, so a batch that cannot fit never raises. Each history entry has a "timing"
    dict: seconds of batch construction, host-to-device, forward, backward plus optimizer,
    validation and checkpointing (the checkpoint of an epoch is written before its own
    duration is known, so a resumed history has None there for the epoch before the stop).
    At return the model holds the best weights (or, if no epoch finished, the current ones).
    """
    c = merge_cfg({**model.cfg, **cfg})
    device = _place(model, c.get("device"), log)
    enabled, amp_dtype, use_scaler = amp_settings(device)
    torch.manual_seed(c["seed"])
    opt, fused = make_optimizer(model, c, device, log)  # after the model is on its device
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    state = {
        "epoch": 0,
        "step": 0,
        "best_val": float("inf"),
        "best_epoch": -1,
        "bad_epochs": 0,
        "history": [],
        "batch_games": int(c["batch_games"]),
        "micro": 1,  # micro-batches per step (grows on out-of-memory)
        "batch_in_epoch": 0,  # batches of the current epoch already done (mid-epoch resume)
        "epoch_loss_sum": 0.0,
        "epoch_loss_n": 0,
        "finished": False,
    }
    best_weights = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["current_model"])
        opt.load_state_dict(ck["optimizer"])
        _match_fused(opt, fused)
        if ck.get("scaler") and use_scaler:
            scaler.load_state_dict(ck["scaler"])
        best_weights = ck["model"]
        for k in state:
            if k in ck:
                state[k] = ck[k]
        torch.set_rng_state(ck["torch_rng"])
        if device.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        log(
            f"resumed from {ckpt_path}: epoch {state['epoch']} batch {state['batch_in_epoch']}, "
            f"step {state['step']}, best val {state['best_val']:.4f}"
        )

    t_prep = time.perf_counter()
    store = _storage_device(
        c, device, _split_bytes(train_split, model.max_len) + _split_bytes(val_split, model.max_len)
    )
    ts_train = prebuild_split(train_split, model.max_len, store)
    ts_val = prebuild_split(val_split, model.max_len, store)
    width = int(ts_train.lens_np.max()) if len(ts_train.lens_np) else 1
    planned = _planned_micro(c, state["batch_games"], width, device, enabled)
    if planned > state["micro"]:
        log(
            f"planning {planned} micro-batches per step from free GPU memory "
            f"(about {train_bytes_per_token(c, enabled) / 1024:.0f} KiB per padded token)"
        )
        state["micro"] = planned
    prep_s = time.perf_counter() - t_prep
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    log(
        f"training on {device} ({gpu}); data tensors on {store} "
        f"({(ts_train.nbytes() + ts_val.nbytes()) / 2**20:.1f} MiB, built in {prep_s:.1f}s); "
        f"AMP {amp_dtype if enabled else 'off'}"
        f"{' with GradScaler' if use_scaler else ''}; {'fused ' if fused else ''}AdamW; "
        f"{state['batch_games']} games per step, {state['micro']} micro-batch(es)"
    )

    def checkpoint() -> None:
        if not ckpt_path:
            return
        _save(
            ckpt_path,
            {
                "model": best_weights,
                "current_model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict() if use_scaler else None,
                "cfg": model.cfg,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                **state,
            },
        )

    def autocast():
        return torch.autocast(device.type, dtype=amp_dtype, enabled=enabled)

    def train_step(batch: dict, timer: PhaseTimer, lens: np.ndarray) -> torch.Tensor:
        """One optimizer step on batch, split into state["micro"] micro-batches. Returns the
        loss as a device tensor (no synchronisation)."""
        nonlocal scaler
        while True:
            k = int(state["micro"])
            try:
                opt.zero_grad(set_to_none=True)
                if k == 1:
                    with autocast():
                        total, _ = model.loss(batch)
                    timer.mark("forward")
                    scaler.scale(total).backward()
                    value = total.detach()
                    del total
                else:
                    denom = _full_counts(model, batch)
                    value = torch.zeros((), dtype=torch.float32, device=device)
                    for part in _micro_slices(batch, k, lens):
                        with autocast():
                            part_total, _ = model.loss(part, denom=denom)
                        timer.mark("forward")
                        scaler.scale(part_total).backward()
                        value = value + part_total.detach()
                        del part_total
                        timer.mark("backward_opt")
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
                scaler.step(opt)
                scaler.update()
                timer.mark("backward_opt")
                return value
            except RuntimeError as err:  # torch.OutOfMemoryError subclasses RuntimeError
                rows = int(batch["mask"].shape[0])
                if not _is_oom(err) or k >= rows:
                    raise
            # outside the except block, so the traceback (and its activations) is released
            opt.zero_grad(set_to_none=True)
            scaler = _fresh_scaler(scaler, use_scaler)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            state["micro"] = 2 * k
            log(f"out of memory with {k} micro-batch(es) per step; retrying with {2 * k}")

    stopped = "finished" if state["finished"] else "epochs"
    n_games = len(train_split.offsets) - 1
    records_seen, train_time = 0, 0.0
    sync_every = int(c.get("timing_sync_every", 8))
    while not state["finished"] and state["epoch"] < c["epochs"]:
        epoch = state["epoch"]
        if c["stop_after_epoch"] is not None and epoch >= c["stop_after_epoch"]:
            stopped = "stop_after_epoch"
            break
        if time.time() > deadline_ts:
            stopped = "deadline"
            break
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        bg = state["batch_games"]
        n_batches = math.ceil(n_games / bg)
        rng = np.random.default_rng([c["seed"], epoch])
        order = rng.permutation(n_games)  # the same batches as make_batches(train, bg, ., rng)
        t0 = time.time()
        loop_t0 = time.perf_counter()
        timer = PhaseTimer(device, sync_every)
        order_t = torch.as_tensor(order, device=ts_train.device)
        loss_acc = torch.full(
            (), float(state["epoch_loss_sum"]), dtype=torch.float64, device=device
        )
        skip = int(state["batch_in_epoch"])  # done before a mid-epoch stop
        hit_deadline = False
        epoch_records = epoch_steps = 0
        for bi in range(skip, n_batches):
            if time.time() > deadline_ts:
                hit_deadline = True
                break
            timer.start()
            sel = order[bi * bg : (bi + 1) * bg]
            raw = _gather(ts_train, sel, order_t[bi * bg : (bi + 1) * bg])
            timer.mark("batch")
            if ts_train.device != device:
                raw = to_device(raw, device)
                timer.mark("h2d")
            batch = _finish(raw)
            factor = _lr_factor(state["step"], epoch + bi / n_batches, c)
            for g in opt.param_groups:
                g["lr"] = c["lr"] * factor
            timer.mark("batch")
            value = train_step(batch, timer, ts_train.lens_np[sel])
            loss_acc += value
            state["step"] += 1
            state["batch_in_epoch"] = bi + 1
            state["epoch_loss_n"] += 1
            n_rec = int(ts_train.lens_np[sel].sum())
            records_seen += n_rec
            epoch_records += n_rec
            epoch_steps += 1
        state["epoch_loss_sum"] = float(loss_acc)  # the one read-back of the epoch
        loop_s = time.perf_counter() - loop_t0
        train_time += time.time() - t0
        phases = timer.split(loop_s)
        timing = {
            "batch_s": phases.get("batch", 0.0),
            "h2d_s": phases.get("h2d", 0.0),
            "forward_s": phases.get("forward", 0.0),
            "backward_opt_s": phases.get("backward_opt", 0.0),
            "train_loop_s": loop_s,
            "steps": epoch_steps,
            "train_records": epoch_records,
            "train_records_per_s": epoch_records / loop_s if loop_s > 0 else float("nan"),
            "micro_batches": int(state["micro"]),
            "data_on": str(ts_train.device),
            "method": timer.method(),
        }
        if hit_deadline:
            stopped = "deadline"
            if state["best_epoch"] < 0:  # no validated epoch yet: keep what was learned
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint()
            log(
                f"deadline reached during epoch {epoch}; saved current state "
                f"({epoch_steps} steps, {timing['train_records_per_s']:.0f} records/s)"
            )
            break
        t_val = time.perf_counter()
        val = evaluate_nll(model, val_split, c, prebuilt=ts_val)
        timing["validation_s"] = time.perf_counter() - t_val
        train_loss = state["epoch_loss_sum"] / max(state["epoch_loss_n"], 1)
        improved = val["total"] < state["best_val"]
        if improved:
            state["best_val"] = val["total"]
            state["best_epoch"] = epoch
            state["bad_epochs"] = 0
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            state["bad_epochs"] += 1
        if device.type == "cuda":
            timing["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
        timing["checkpoint_s"] = None
        state["history"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_total": val["total"],
                "val_fields": val["fields"],
                "lr": opt.param_groups[0]["lr"],
                "seconds": time.time() - t0,
                "batch_games": bg,
                "micro": int(state["micro"]),
                "timing": timing,
            }
        )
        state["epoch"] = epoch + 1
        state["batch_in_epoch"] = 0
        state["epoch_loss_sum"], state["epoch_loss_n"] = 0.0, 0
        if state["bad_epochs"] >= c["patience"]:
            state["finished"] = True
            stopped = "early_stopping"
        elif state["epoch"] >= c["epochs"]:
            state["finished"] = True
        t_ck = time.perf_counter()
        checkpoint()
        timing["checkpoint_s"] = time.perf_counter() - t_ck
        timing["epoch_s"] = time.time() - t0
        log(
            f"epoch {epoch}: train {train_loss:.4f}  val {val['total']:.4f} nats/record"
            f"{'  (best)' if improved else ''}  {time.time() - t0:.0f}s | "
            f"{timing['train_records_per_s']:,.0f} records/s; batch {timing['batch_s']:.1f}s, "
            f"h2d {timing['h2d_s']:.1f}s, forward {timing['forward_s']:.1f}s, backward+opt "
            f"{timing['backward_opt_s']:.1f}s, validation {timing['validation_s']:.1f}s, "
            f"checkpoint {timing['checkpoint_s']:.1f}s"
        )
    model.load_state_dict(best_weights)
    totals: dict[str, float] = {}
    for h in state["history"]:
        for k in DURATION_KEYS:
            v = (h.get("timing") or {}).get(k)
            if isinstance(v, (int, float)):
                totals[k] = totals.get(k, 0.0) + float(v)
    return {
        "stopped": stopped,
        "epochs_done": state["epoch"],
        "best_epoch": state["best_epoch"],
        "best_val_nll": state["best_val"],
        "batch_games": state["batch_games"],
        "micro_batches": int(state["micro"]),
        "history": state["history"],
        "train_records_per_s": records_seen / train_time if train_time > 0 else float("nan"),
        "device": str(device),
        "amp": str(amp_dtype) if enabled else None,
        "grad_scaler": bool(use_scaler),
        "optimizer": "fused AdamW" if fused else "AdamW",
        "data_on": str(ts_train.device),
        "prebuild_s": prep_s,
        "timing_totals_s": totals,
    }
