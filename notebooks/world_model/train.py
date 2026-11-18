"""Batching, training loop (AdamW, warmup + cosine, AMP, resume, early stopping) and NLL."""

from __future__ import annotations

import copy
import math
import os
import time
from collections.abc import Callable, Iterator

import numpy as np
import schema as S
import torch
from model import TARGET_NAMES, WorldModel, device_of, merge_cfg


# ---------------------------------------------------------------------- batching
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


# ---------------------------------------------------------------------- precision
def amp_settings(device: torch.device) -> tuple[bool, torch.dtype, bool]:
    """(enabled, dtype, use_grad_scaler). bf16 only on GPUs with native support (sm80+)."""
    if device.type != "cuda":
        return False, torch.float32, False
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:
        return True, torch.bfloat16, False
    return True, torch.float16, True


# ---------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate_nll(model: WorldModel, split: S.Split, cfg: dict | None = None) -> dict:
    """Per-field mean NLL (nats per target) and total NLL per record over a split."""
    c = merge_cfg({**model.cfg, **(cfg or {})})
    device = device_of(model)
    enabled, dtype, _ = amp_settings(device)
    was_training = model.training
    model.eval()
    sums = torch.zeros(len(S.TARGET_FIELDS), dtype=torch.float64)
    counts = torch.zeros(len(S.TARGET_FIELDS), dtype=torch.float64)
    n_records = 0
    for batch in make_batches(split, c["eval_batch_games"], model.max_len, None):
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=dtype, enabled=enabled):
            _, info = model.loss(batch)
        sums += info["sums"].double().cpu()
        counts += info["counts"].double().cpu()
        n_records += int(batch["mask"].sum())
    model.train(was_training)
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

    One checkpoint file at ckpt_path holds: "model" = best weights so far (by validation NLL),
    "current_model" + optimizer + scaler + RNG states + epoch + step (to resume exactly),
    cfg, history. If ckpt_path exists, training resumes from it. Validation NLL is evaluated
    after every epoch; training stops on early stopping (patience), at cfg["epochs"], at
    cfg["stop_after_epoch"], or cleanly at the first batch after time.time() > deadline_ts.
    On CUDA out-of-memory the batch size is halved and the epoch restarts (weights kept).
    At return the model holds the best weights.
    """
    c = merge_cfg({**model.cfg, **cfg})
    device = device_of(model)
    enabled, amp_dtype, use_scaler = amp_settings(device)
    torch.manual_seed(c["seed"])
    opt = torch.optim.AdamW(
        _param_groups(model, c["weight_decay"]), lr=c["lr"], betas=tuple(c["betas"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    state = {
        "epoch": 0,
        "step": 0,
        "best_val": float("inf"),
        "best_epoch": -1,
        "bad_epochs": 0,
        "history": [],
        "batch_games": int(c["batch_games"]),
        "finished": False,
    }
    best_weights = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["current_model"])
        opt.load_state_dict(ck["optimizer"])
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
            f"resumed from {ckpt_path}: epoch {state['epoch']}, step {state['step']}, "
            f"best val {state['best_val']:.4f}"
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

    stopped = "finished" if state["finished"] else "epochs"
    n_games = len(train_split.offsets) - 1
    records_seen, train_time = 0, 0.0
    while not state["finished"] and state["epoch"] < c["epochs"]:
        epoch = state["epoch"]
        if c["stop_after_epoch"] is not None and epoch >= c["stop_after_epoch"]:
            stopped = "stop_after_epoch"
            break
        if time.time() > deadline_ts:
            stopped = "deadline"
            break
        model.train()
        bg = state["batch_games"]
        n_batches = math.ceil(n_games / bg)
        rng = np.random.default_rng([c["seed"], epoch])
        t0 = time.time()
        loss_sum, loss_n, hit_deadline = 0.0, 0, False
        try:
            for bi, batch in enumerate(make_batches(train_split, bg, model.max_len, rng)):
                if time.time() > deadline_ts:
                    hit_deadline = True
                    break
                factor = _lr_factor(state["step"], epoch + bi / n_batches, c)
                for g in opt.param_groups:
                    g["lr"] = c["lr"] * factor
                batch = to_device(batch, device)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=enabled):
                    total, _ = model.loss(batch)
                opt.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
                scaler.step(opt)
                scaler.update()
                state["step"] += 1
                loss_sum += float(total.detach())
                loss_n += 1
                records_seen += int(batch["mask"].sum())
        except RuntimeError as err:  # torch.OutOfMemoryError subclasses RuntimeError
            if device.type != "cuda" or not _is_oom(err) or bg <= 1:
                raise
            opt.zero_grad(set_to_none=True)
            del err
            torch.cuda.empty_cache()
            state["batch_games"] = max(1, bg // 2)
            log(f"CUDA out of memory at batch_games={bg}; halving to {state['batch_games']}")
            continue
        train_time += time.time() - t0
        if hit_deadline:
            stopped = "deadline"
            checkpoint()
            log(f"deadline reached during epoch {epoch}; saved current state")
            break
        val = evaluate_nll(model, val_split, c)
        train_loss = loss_sum / max(loss_n, 1)
        improved = val["total"] < state["best_val"]
        if improved:
            state["best_val"] = val["total"]
            state["best_epoch"] = epoch
            state["bad_epochs"] = 0
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            state["bad_epochs"] += 1
        state["history"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_total": val["total"],
                "val_fields": val["fields"],
                "lr": opt.param_groups[0]["lr"],
                "seconds": time.time() - t0,
                "batch_games": bg,
            }
        )
        state["epoch"] = epoch + 1
        if state["bad_epochs"] >= c["patience"]:
            state["finished"] = True
            stopped = "early_stopping"
        elif state["epoch"] >= c["epochs"]:
            state["finished"] = True
        checkpoint()
        log(
            f"epoch {epoch}: train {train_loss:.4f}  val {val['total']:.4f} nats/record"
            f"{'  (best)' if improved else ''}  {time.time() - t0:.0f}s"
        )
    model.load_state_dict(best_weights)
    return {
        "stopped": stopped,
        "epochs_done": state["epoch"],
        "best_epoch": state["best_epoch"],
        "best_val_nll": state["best_val"],
        "batch_games": state["batch_games"],
        "history": state["history"],
        "train_records_per_s": records_seen / train_time if train_time > 0 else float("nan"),
    }
