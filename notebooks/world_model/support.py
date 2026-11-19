"""Action support for the fourth-down planner: what the training data can and cannot vouch for.

The world model learned from what coaches did. For a call coaches almost never make from a
state (a 70-yard field goal, a punt from the opponent's 25, going for it on fourth and 15 from
your own 10) it has seen almost no examples, and a simulator can imagine good outcomes there
without anything in the data to correct it. This is the distribution-shift problem of offline
reinforcement learning on logged data: a planner that maximises over actions seeks out exactly
the actions whose values are least constrained by data. This module makes that measurable and
guards against it:

1. Support tables, from the TRAINING split only: for every fourth-down state (yard line, yards
   to go, optionally score and time buckets), how often coaches chose GO, FG and PUNT, summed
   over comparable states (yard line within +-3, yards to go in the same bucket). Plus the
   empirical field-goal make rate by yard line and the empirical punt net yards by yard line.
2. Hard rules, applied through `simulate.plan_fourth_downs(..., action_mask=...)`: no field
   goal beyond the longest made field goal in the training split plus 3 yards; no punt inside
   the opponent's 30; going for it is always allowed.
3. Soft flags reported with every recommendation: how many comparable training fourth downs
   there were, how many times coaches made the recommended call there and its share; "low
   support" when coaches made that call fewer than 10 times in comparable training states.
4. A diagnostic: the world model's simulated field-goal make rate (share of sampled first
   transitions after a forced field goal that score 3 for the kicking team) against the
   empirical make rate at the same distance, and the same for punt net yards.

Standalone: numpy, pandas and the sibling `schema` module; torch only inside the sampling
functions.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import schema as S

SI, TI = S.STATE_INDEX, S.TARGET_INDEX
PH, A = S.PHASE, S.ACTION
CHOICES = ("GO", "FG", "PUNT")  # column order of every [.., 3] array and of action_mask

# Field goal distance = yardline_100 + 18 (10 yards of end zone plus the snap and hold, about
# 8 yards behind the line of scrimmage). This is the usual relation in nflverse's
# kick_distance; a few attempts are recorded a yard shorter or longer.
KICK_DISTANCE_OFFSET = 18

# Yards-to-go buckets for "comparable" states. Short yardage is kept exact (1, 2, 3) because
# the go rate changes fast there; long yardage is pooled because it is rare.
YTG_BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 7), (8, 10), (11, 15), (16, 30))
# Optional context buckets: score from the offense's side, and game clock.
SCORE_BUCKETS = ("down 9+", "down 1-8", "tied", "up 1-8", "up 9+")
TIME_BUCKETS = ("first half", "second half, over 5 min left", "last 5 min")

DEFAULTS: dict[str, Any] = {
    "version": 1,
    "fg_margin_yards": 3,  # field goals allowed up to the longest made one plus this
    "punt_min_yardline": 30,  # punts allowed only from yardline_100 >= this
    "min_support": 10,  # "low support" below this many comparable training calls
    "yardline_window": 3,  # comparable states: yard line within +-3
    "diag_states_per_band": 12,  # diagnostic: test fourth downs per distance band
    "diag_draws": 256,  # diagnostic: sampled first transitions per state
    "fg_band_yards": 5,  # diagnostic field-goal bands, in yards of distance
    "fg_band_max_yardline": 60,  # ... up to a 78-yard attempt
    "punt_band_yards": 10,
    "punt_band_min_yardline": 20,
}


def config(plan_cfg: dict | None = None) -> dict:
    """The support settings: DEFAULTS updated with plan_cfg["support"] (if any)."""
    over = dict((plan_cfg or {}).get("support") or {})
    unknown = set(over) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown support settings {sorted(unknown)}")
    return {**DEFAULTS, **over}


# ---------------------------------------------------------------- small helpers
def poss_sign(poss: np.ndarray) -> np.ndarray:
    """+1 when HOME has the ball, -1 when AWAY has it (score deltas are home minus away)."""
    return np.where(np.asarray(poss) == S.POSS_CODE["HOME"], 1, -1)


def choice_index(actions: np.ndarray) -> np.ndarray:
    """0 GO (RUN or PASS, or a synthetic play mode), 1 FG, 2 PUNT, -1 anything else."""
    a = np.asarray(actions)
    return np.select(
        [np.isin(a, S.ANY_GO_ACTIONS), a == A["FG"], a == A["PUNT"]], [0, 1, 2], default=-1
    )


_YTG_TO_BUCKET = np.zeros(S.MAX_YDSTOGO + 1, dtype=np.int64)
for _k, (_lo, _hi) in enumerate(YTG_BUCKETS):
    _YTG_TO_BUCKET[_lo : _hi + 1] = _k


def ytg_bucket(ytg: np.ndarray) -> np.ndarray:
    return _YTG_TO_BUCKET[np.clip(np.asarray(ytg, dtype=np.int64), 1, S.MAX_YDSTOGO)]


def score_bucket(diff_offense: np.ndarray) -> np.ndarray:
    d = np.asarray(diff_offense)
    return np.select([d <= -9, d < 0, d == 0, d <= 8], [0, 1, 2, 3], default=4)


def time_bucket(clock_seconds: np.ndarray) -> np.ndarray:
    c = np.asarray(clock_seconds)
    return np.select([c > S.HALF_SECONDS, c > 300], [0, 1], default=2)


def fourth_down_records(split: S.Split) -> np.ndarray:
    """Real fourth-down decisions: PLAY records on fourth down whose action is GO, FG or PUNT
    (the same rule as records.fourth_down_index)."""
    st = split.states
    ok = (
        (st[:, SI["phase"]] == PH["PLAY"])
        & (st[:, SI["down"]] == 4)
        & (choice_index(st[:, SI["action"]]) >= 0)
    )
    return np.flatnonzero(ok)


def fg_made(poss: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """True where the transition scores 3 points for the team that had the ball.
    poss [...]; targets [..., T] (a trailing sample axis broadcasts against poss[..., None])."""
    tg = np.asarray(targets)
    pts = S.score_delta_value(tg[..., TI["score_delta"]])
    sign = poss_sign(poss)
    while sign.ndim < pts.ndim:
        sign = sign[..., None]
    return pts == 3 * sign


def punt_net_yards(yardline: np.ndarray, poss: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Net yards of a punt from the next record: how much closer to the opponent's goal line
    the ball ends up, from the punting team's side. NaN when the next record is not a PLAY
    (return touchdown, safety, end of game), which is counted separately as "other".

    Possession changed: the receiver's yard line y is 100 - y from the punter's side, so
    net = yardline - (100 - y). Possession kept (a muff recovered, a penalty re-kick):
    net = yardline - y.
    """
    tg = np.asarray(targets)
    yl, poss = np.asarray(yardline), np.asarray(poss)
    while yl.ndim < tg.ndim - 1:
        yl, poss = yl[..., None], poss[..., None]
    nxt_yl = tg[..., TI["yardline"]].astype(np.float64)
    kept = tg[..., TI["poss"]] == poss
    after = np.where(kept, nxt_yl, 100.0 - nxt_yl)
    net = yl - after
    return np.where(tg[..., TI["phase"]] == PH["PLAY"], net, np.nan)


# ---------------------------------------------------------------- support tables
def _box_smooth(counts: np.ndarray, window: int) -> np.ndarray:
    """Sum counts [100, 31, ...] over comparable states: yard line within +-window (1..99)
    and yards to go in the same YTG_BUCKETS bucket. Same shape out."""
    rest = counts.shape[2:]
    by_bucket = np.zeros((100, len(YTG_BUCKETS), *rest), dtype=np.int64)
    for k, (lo, hi) in enumerate(YTG_BUCKETS):
        by_bucket[:, k] = counts[:, lo : hi + 1].sum(axis=1)
    sm = np.zeros_like(by_bucket)
    for yl in range(1, 100):
        sm[yl] = by_bucket[max(1, yl - window) : min(99, yl + window) + 1].sum(axis=0)
    out = np.zeros_like(counts, dtype=np.int64)
    out[:, 1:] = sm[:, _YTG_TO_BUCKET[1:]]
    return out


@dataclass
class Support:
    """Training-split tables. Index 0 of every yard-line axis is unused (yard lines 1..99)."""

    counts: np.ndarray  # [100, 31, 3] fourth-down calls by yard line, yards to go
    ctx_counts: np.ndarray  # [100, 31, 5, 3, 3] ... by score bucket and time bucket too
    smooth: np.ndarray  # counts summed over comparable states
    ctx_smooth: np.ndarray
    fg_attempts: np.ndarray  # [100] field-goal attempts by yard line (PLAY, any down)
    fg_makes: np.ndarray  # [100]
    punts: np.ndarray  # [100] punts by yard line
    punts_next_play: np.ndarray  # [100] punts followed by a PLAY record (net yards defined)
    punt_net_sum: np.ndarray  # [100]
    cfg: dict
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        """Compact tables for results.json (comparable-state calls by yard line and
        yards-to-go bucket; kicking tables by yard line)."""
        by_bucket = np.zeros((100, len(YTG_BUCKETS), 3), dtype=np.int64)
        for k, (lo, hi) in enumerate(YTG_BUCKETS):
            by_bucket[:, k] = self.counts[:, lo : hi + 1].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            make = self.fg_makes / self.fg_attempts
            net = self.punt_net_sum / self.punts_next_play
        return {
            **self.meta,
            "choices": list(CHOICES),
            "ytg_buckets": [list(b) for b in YTG_BUCKETS],
            "calls_by_yardline_and_ytg_bucket_raw": by_bucket[1:].tolist(),
            "yardlines": list(range(1, 100)),
            "fg_attempts_by_yardline": self.fg_attempts[1:].tolist(),
            "fg_makes_by_yardline": self.fg_makes[1:].tolist(),
            "fg_make_rate_by_yardline": [None if not np.isfinite(v) else round(float(v), 4)
                                         for v in make[1:]],
            "punts_by_yardline": self.punts[1:].tolist(),
            "punt_mean_net_yards_by_yardline": [None if not np.isfinite(v) else
                                                round(float(v), 2) for v in net[1:]],
            "kick_distance": f"field goal distance = yardline_100 + {KICK_DISTANCE_OFFSET}",
        }


def build_support(train: S.Split, cfg: dict | None = None) -> Support:
    """Support tables from one split. Call it with the TRAINING split only: these tables
    stand for what the simulator could have learned from."""
    c = config({"support": dict(cfg or {})})
    st, tg, raw = train.states, train.targets, train.raw
    phase, yl = st[:, SI["phase"]], st[:, SI["yardline"]].astype(np.int64)
    ytg, act, poss = st[:, SI["ydstogo"]].astype(np.int64), st[:, SI["action"]], st[:, SI["poss"]]
    fd = fourth_down_records(train)
    ch = choice_index(act[fd])
    counts = np.zeros((100, S.MAX_YDSTOGO + 1, 3), dtype=np.int64)
    np.add.at(counts, (yl[fd], ytg[fd], ch), 1)
    sb = score_bucket(poss_sign(poss[fd]) * raw[fd, 1])
    tb = time_bucket(raw[fd, 0])
    ctx = np.zeros((100, S.MAX_YDSTOGO + 1, len(SCORE_BUCKETS), len(TIME_BUCKETS), 3),
                   dtype=np.int64)
    np.add.at(ctx, (yl[fd], ytg[fd], sb, tb, ch), 1)

    play = phase == PH["PLAY"]
    fg = np.flatnonzero(play & (act == A["FG"]))
    made = fg_made(poss[fg], tg[fg])
    fg_att = np.bincount(yl[fg], minlength=100)[:100]
    fg_mk = np.bincount(yl[fg], weights=made, minlength=100)[:100].astype(np.int64)
    pu = np.flatnonzero(play & (act == A["PUNT"]))
    net = punt_net_yards(yl[pu], poss[pu], tg[pu])
    ok = np.isfinite(net)
    punts = np.bincount(yl[pu], minlength=100)[:100]
    p_play = np.bincount(yl[pu][ok], minlength=100)[:100]
    p_sum = np.bincount(yl[pu][ok], weights=net[ok], minlength=100)[:100]
    h = hashlib.sha256("\x1f".join(train.game_ids).encode()).hexdigest()[:16]
    meta = {
        "split_games": len(train.game_ids),
        "split_seasons": sorted({int(s) for s in np.asarray(train.seasons)}),
        "split_game_ids_hash": h,
        "fourth_downs": int(len(fd)),
        "fourth_down_calls": dict(zip(CHOICES, counts.sum(axis=(0, 1)).tolist(), strict=True)),
        "fg_attempts": int(len(fg)),
        "fg_makes": int(made.sum()),
        "punts": int(len(pu)),
    }
    w = int(c["yardline_window"])
    return Support(counts=counts, ctx_counts=ctx, smooth=_box_smooth(counts, w),
                   ctx_smooth=_box_smooth(ctx, w), fg_attempts=fg_att, fg_makes=fg_mk,
                   punts=punts, punts_next_play=p_play, punt_net_sum=p_sum, cfg=c, meta=meta)


# ---------------------------------------------------------------- hard rules
def hard_rules(sup: Support) -> dict:
    """Rules no planner should violate, with the numbers behind them.

    - Field goal: allowed only from yardline_100 <= longest made field goal in the training
      split + fg_margin_yards (3). Beyond that the simulator has never seen a make.
    - Punt: allowed only from yardline_100 >= punt_min_yardline (30): no punt inside the
      opponent's 30, where a punt gains almost nothing and coaches essentially never punt.
    - Go: always allowed (every fourth down has a real snap).
    """
    c = sup.cfg
    made_at = np.flatnonzero(sup.fg_makes > 0)
    longest = int(made_at.max()) if len(made_at) else None
    margin = int(c["fg_margin_yards"])
    fg_max = (longest if longest is not None else 0) + margin
    punt_min = int(c["punt_min_yardline"])
    return {
        "go": "always allowed",
        "fg_max_yardline": fg_max,
        "fg_max_distance_yards": fg_max + KICK_DISTANCE_OFFSET,
        "longest_made_fg_yardline_train": longest,
        "longest_made_fg_distance_yards_train": None if longest is None
        else longest + KICK_DISTANCE_OFFSET,
        "fg_margin_yards": margin,
        "punt_min_yardline": punt_min,
        "train_fourth_down_fg_beyond_limit": int(sup.counts[fg_max + 1 :, :, 1].sum()),
        "train_fourth_down_punts_inside_limit": int(sup.counts[:punt_min, :, 2].sum()),
        "train_fourth_downs": int(sup.counts.sum()),
        "text": (f"field goals only from the opponent's {fg_max} or closer (a "
                 f"{fg_max + KICK_DISTANCE_OFFSET}-yard attempt: the longest made in the "
                 f"training split plus {margin} yards); no punt inside the opponent's "
                 f"{punt_min}; going for it is always allowed"),
    }


def action_mask(split: S.Split, idx, rules: dict) -> np.ndarray:
    """[len(idx), 3] bool for GO, FG, PUNT (True = allowed), from the hard rules."""
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    yl = split.states[idx, SI["yardline"]].astype(np.int64)
    m = np.ones((len(idx), 3), dtype=bool)
    m[:, 1] = yl <= int(rules["fg_max_yardline"])
    m[:, 2] = yl >= int(rules["punt_min_yardline"])
    return m


def masked_actions(split: S.Split, idx, rules: dict, label: str = "test") -> list[dict]:
    """One entry per masked (state, action): which state, which action, which rule."""
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    m = action_mask(split, idx, rules)
    games = np.searchsorted(split.offsets, idx, side="right") - 1
    out = []
    for q, j in zip(*np.nonzero(~m), strict=True):
        i = int(idx[q])
        yl = int(split.states[i, SI["yardline"]])
        if CHOICES[j] == "FG":
            rule = (f"field goal from yardline {yl} ({yl + KICK_DISTANCE_OFFSET} yards) beyond "
                    f"the limit {rules['fg_max_yardline']} "
                    f"({rules['fg_max_distance_yards']} yards)")
        else:
            rule = f"punt from yardline {yl}, inside the opponent's {rules['punt_min_yardline']}"
        out.append({"set": label, "idx": i, "game_id": split.game_ids[int(games[q])],
                    "yardline_100": yl, "ydstogo": int(split.states[i, SI["ydstogo"]]),
                    "real_action": S.ACTIONS[int(split.states[i, SI["action"]])],
                    "masked": CHOICES[j], "rule": rule})
    return out


# ---------------------------------------------------------------- soft flags
def lookup(sup: Support, split: S.Split, idx) -> pd.DataFrame:
    """Comparable-state call counts for each record in idx (training-split tables)."""
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    st, raw = split.states[idx], split.raw[idx]
    yl = st[:, SI["yardline"]].astype(np.int64)
    ytg = np.clip(st[:, SI["ydstogo"]].astype(np.int64), 1, S.MAX_YDSTOGO)
    base = sup.smooth[yl, ytg]
    sb = score_bucket(poss_sign(st[:, SI["poss"]]) * raw[:, 1])
    tb = time_bucket(raw[:, 0])
    ctx = sup.ctx_smooth[yl, ytg, sb, tb]
    out = {"sup_n": base.sum(1), "ctx_n": ctx.sum(1),
           "ctx_bucket": [f"{SCORE_BUCKETS[s]}, {TIME_BUCKETS[t]}" for s, t in zip(sb, tb,
                                                                                strict=True)]}
    for j, ch in enumerate(CHOICES):
        out[f"sup_{ch.lower()}"] = base[:, j]
        out[f"ctx_{ch.lower()}"] = ctx[:, j]
    return pd.DataFrame(out)


def add_support_columns(df: pd.DataFrame, sup: Support, split: S.Split,
                        rules: dict) -> pd.DataFrame:
    """Soft flags next to every recommendation in a plan_fourth_downs frame.

    sup_n: comparable training fourth downs (yard line +-3, same yards-to-go bucket);
    sup_count_best / sup_share_best: how often coaches made the recommended call there;
    low_support: sup_count_best < min_support; ctx_*: the same within the state's score
    and time bucket (narrower, reported only). allowed_*: the hard-rule mask.
    """
    df = df.copy()
    if len(df) == 0:
        for k in ("sup_n", "sup_count_best", "sup_share_best", "low_support"):
            df[k] = []
        return df
    idx = df["idx"].to_numpy(dtype=np.int64)
    lk = lookup(sup, split, idx)
    for k in lk.columns:
        df[k] = lk[k].to_numpy()
    m = action_mask(split, idx, rules)
    for j, ch in enumerate(CHOICES):
        df[f"allowed_{ch.lower()}"] = m[:, j]
    best = df["best"].astype(object).where(df["best"].notna(), None)
    jbest = np.array([CHOICES.index(b) if b in CHOICES else -1 for b in best])
    has = jbest >= 0
    rows = np.arange(len(df))
    base = lk[[f"sup_{c.lower()}" for c in CHOICES]].to_numpy()
    ctx = lk[[f"ctx_{c.lower()}" for c in CHOICES]].to_numpy()
    cnt = np.where(has, base[rows, np.maximum(jbest, 0)], -1)
    ccnt = np.where(has, ctx[rows, np.maximum(jbest, 0)], -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(has & (lk["sup_n"] > 0), cnt / lk["sup_n"], np.nan)
        cshare = np.where(has & (lk["ctx_n"] > 0), ccnt / lk["ctx_n"], np.nan)
    df["sup_count_best"] = np.where(has, cnt, np.nan)
    df["sup_share_best"] = share
    df["ctx_count_best"] = np.where(has, ccnt, np.nan)
    df["ctx_share_best"] = cshare
    df["low_support"] = has & (cnt < int(sup.cfg["min_support"]))
    real = df["real_choice"] if "real_choice" in df.columns else None
    if real is not None:
        jr = np.array([CHOICES.index(r) if r in CHOICES else -1 for r in real])
        df["sup_share_real"] = np.where(
            (jr >= 0) & (lk["sup_n"] > 0), base[rows, np.maximum(jr, 0)] / np.maximum(
                lk["sup_n"].to_numpy(), 1), np.nan)
    return df


def empirical_fg_rate(sup: Support, yardline: np.ndarray, window: int = 2) -> tuple:
    """Training make rate and attempts within +-window yard lines of each yard line."""
    yl = np.atleast_1d(np.asarray(yardline, dtype=np.int64))
    att = np.array([sup.fg_attempts[max(1, y - window) : y + window + 1].sum() for y in yl])
    mk = np.array([sup.fg_makes[max(1, y - window) : y + window + 1].sum() for y in yl])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(att > 0, mk / np.maximum(att, 1), np.nan), att


# ---------------------------------------------------------------- one-step sampling
def _device(model):
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        import torch

        return torch.device("cpu")


def first_transitions(model, split: S.Split, idx, action, n_draws: int, cfg: dict | None,
                      rng: np.random.Generator, chunk: int = 64,
                      deadline_ts: float | None = None) -> np.ndarray:
    """Sample n_draws first transitions from each record in idx with its action forced.

    This is exactly the first step of a rollout from that record (`simulate.rollout` encodes
    the context window ending at the record, with the forced action, and samples the
    transition from the last position with `model.sample_transition`), without playing on to
    the end of the game, so it costs one encoder pass per state. The encoder runs in float32
    here; rollouts on a GPU may encode under autocast, which changes only rounding. Returns
    codes [len(idx), n_draws, len(TARGET_FIELDS)] (int16).
    """
    import torch

    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    code = A[action] if isinstance(action, str) else int(action)
    c = {**(getattr(model, "cfg", None) or {}), **(cfg or {})}
    temp, mask_pad = float(c.get("temperature", 1.0)), bool(c.get("mask_pad_codes", True))
    max_len = int(getattr(model, "max_len", c.get("max_len", 256)))
    dev = _device(model)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(rng.integers(0, 2**62)))
    games = np.searchsorted(split.offsets, idx, side="right") - 1
    out = np.zeros((len(idx), int(n_draws), len(S.TARGET_FIELDS)), dtype=np.int16)
    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        for a in range(0, len(idx), chunk):
            if deadline_ts is not None and time.time() > deadline_ts:
                raise TimeoutError("support diagnostic ran out of time")
            q = np.arange(a, min(len(idx), a + chunk))
            starts = [max(int(split.offsets[g]), int(i) + 1 - max_len)
                      for g, i in zip(games[q], idx[q], strict=True)]
            lens = np.array([int(i) - s + 1 for s, i in zip(starts, idx[q], strict=True)])
            buf = np.zeros((len(q), int(lens.max()), len(S.STATE_FIELDS)), dtype=np.int64)
            for r, (s, i) in enumerate(zip(starts, idx[q], strict=True)):
                buf[r, : lens[r]] = split.states[s : int(i) + 1]
                buf[r, lens[r] - 1, SI["action"]] = code
            header = torch.as_tensor(split.header[games[q]].astype(np.int64), device=dev)
            h = model.encode(header, torch.as_tensor(buf, device=dev))
            last = h[torch.arange(len(q), device=dev), torch.as_tensor(lens - 1, device=dev)]
            rep = last.float().repeat_interleave(int(n_draws), dim=0)
            codes = model.sample_transition(rep, temp, gen, mask_pad)
            out[q] = codes.reshape(len(q), int(n_draws), -1).cpu().numpy().astype(np.int16)
    return out


def sim_fg_make(model, split: S.Split, idx, n_draws: int, cfg: dict | None,
                rng: np.random.Generator, deadline_ts: float | None = None) -> np.ndarray:
    """Per-state simulated make rate of a forced field goal (share of first transitions that
    score 3 for the kicking team). [len(idx)]."""
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    if len(idx) == 0:
        return np.zeros(0)
    codes = first_transitions(model, split, idx, "FG", n_draws, cfg, rng,
                              deadline_ts=deadline_ts)
    return fg_made(split.states[idx, SI["poss"]], codes).mean(axis=1)


def _bands(lo: int, hi: int, width: int) -> list[tuple[int, int]]:
    return [(a, min(a + width - 1, hi)) for a in range(lo, hi + 1, width)]


def _sample_band(pool: np.ndarray, yl: np.ndarray, band: tuple[int, int], k: int,
                 rng: np.random.Generator) -> np.ndarray:
    cand = pool[(yl >= band[0]) & (yl <= band[1])]
    if len(cand) == 0:
        return cand
    return np.sort(rng.choice(cand, size=min(k, len(cand)), replace=False))


def _mean_se(per_state: np.ndarray, n_draws: int) -> tuple[float | None, float | None]:
    v = per_state[np.isfinite(per_state)]
    if len(v) == 0:
        return None, None
    if len(v) == 1:
        p = float(v[0])
        return p, float(np.sqrt(max(p * (1 - p), 1e-12) / n_draws))
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def sim_vs_empirical(model, split: S.Split, sup: Support, rules: dict, cfg: dict | None,
                     rng: np.random.Generator, pool_idx=None,
                     deadline_ts: float | None = None) -> dict:
    """Simulated versus empirical field-goal make rate by distance, and punt net yards by
    yard line.

    States: for each band, up to diag_states_per_band real fourth downs of `split` (the
    test split in the plan stage) whose yard line is in the band, whatever the coach did;
    the field goal or punt is then forced. Most long-range states are real punt situations,
    so this probes the simulator exactly where coaches do not kick. Empirical numbers come
    from the training-split tables in `sup` (all field-goal attempts on PLAY records, any
    down; all punts).
    """
    c = sup.cfg
    k, n = int(c["diag_states_per_band"]), int(c["diag_draws"])
    pool = fourth_down_records(split) if pool_idx is None else np.asarray(pool_idx, np.int64)
    pool_yl = split.states[pool, SI["yardline"]].astype(np.int64)
    out: dict[str, Any] = {
        "states_per_band": k, "draws_per_state": n,
        "kick_distance": f"yardline_100 + {KICK_DISTANCE_OFFSET}",
        "method": "first transition after the forced action, sampled draws_per_state times "
                  "from each state's own context (the first step of a rollout); simulated "
                  "rate = mean over states of the per-state share",
        "caveat": "empirical rates are for kicks and punts coaches chose to attempt; long "
                  "attempts are selected (strong kickers, good conditions, but also "
                  "end-of-half heaves), so a gap is not all simulator error",
    }
    fg_rows = []
    fg_states = []
    for band in _bands(1, int(c["fg_band_max_yardline"]), int(c["fg_band_yards"])):
        s = _sample_band(pool, pool_yl, band, k, rng)
        fg_states.append(s)
        att = int(sup.fg_attempts[band[0] : band[1] + 1].sum())
        mk = int(sup.fg_makes[band[0] : band[1] + 1].sum())
        fg_rows.append({"yardline": list(band),
                        "distance_yards": [band[0] + KICK_DISTANCE_OFFSET,
                                           band[1] + KICK_DISTANCE_OFFSET],
                        "train_attempts": att, "train_makes": mk,
                        "empirical_make_rate": mk / att if att else None,
                        "beyond_fg_limit": band[0] > int(rules["fg_max_yardline"]),
                        "n_states": int(len(s)), "state_idx": s.tolist()})
    all_fg = np.concatenate(fg_states) if fg_states else np.zeros(0, np.int64)
    per = sim_fg_make(model, split, all_fg, n, cfg, rng, deadline_ts)
    a = 0
    for row, s in zip(fg_rows, fg_states, strict=True):
        m, se = _mean_se(per[a : a + len(s)], n)
        a += len(s)
        row["sim_make_rate"], row["sim_se"] = m, se
        e = row["empirical_make_rate"]
        row["sim_minus_empirical"] = None if m is None or e is None else m - e
    out["fg"] = fg_rows

    punt_rows, punt_states = [], []
    for band in _bands(int(c["punt_band_min_yardline"]), 99, int(c["punt_band_yards"])):
        s = _sample_band(pool, pool_yl, band, k, rng)
        punt_states.append(s)
        sl = slice(band[0], band[1] + 1)
        npl, tot = int(sup.punts_next_play[sl].sum()), int(sup.punts[sl].sum())
        punt_rows.append({"yardline": list(band), "train_punts": tot,
                          "empirical_net_yards": float(sup.punt_net_sum[sl].sum() / npl)
                          if npl else None,
                          "empirical_other_share": (tot - npl) / tot if tot else None,
                          "inside_punt_limit": band[0] < int(rules["punt_min_yardline"]),
                          "n_states": int(len(s)), "state_idx": s.tolist()})
    all_p = np.concatenate(punt_states) if punt_states else np.zeros(0, np.int64)
    if len(all_p):
        codes = first_transitions(model, split, all_p, "PUNT", n, cfg, rng,
                                  deadline_ts=deadline_ts)
        net = punt_net_yards(split.states[all_p, SI["yardline"]],
                             split.states[all_p, SI["poss"]], codes)
    else:
        net = np.zeros((0, n))
    a = 0
    for row, s in zip(punt_rows, punt_states, strict=True):
        part = net[a : a + len(s)]
        a += len(s)
        with np.errstate(invalid="ignore"):
            per_state = np.array([np.nanmean(r) if np.isfinite(r).any() else np.nan
                                  for r in part])
        m, se = _mean_se(per_state, n)
        row["sim_net_yards"], row["sim_se"] = m, se
        row["sim_other_share"] = float(np.mean(~np.isfinite(part))) if part.size else None
        e = row["empirical_net_yards"]
        row["sim_minus_empirical"] = None if m is None or e is None else m - e
    out["punt"] = punt_rows
    return out


def fg_gap_summary(diag: dict, min_attempts: int = 10) -> dict:
    """Headline numbers of the field-goal diagnostic.

    Bands with at least min_attempts training attempts give a usable empirical rate: their
    attempt-weighted mean gap (simulated minus training) and the largest gap. Bands with
    fewer (thin data, inside the hard limit) and bands beyond the limit are summarised by
    the simulator's mean make rate there, since the data cannot say what the rate is.
    """
    rows = [r for r in diag.get("fg", []) if r.get("sim_make_rate") is not None]
    data = [r for r in rows if (r["train_attempts"] or 0) >= min_attempts]
    thin = [r for r in rows if (r["train_attempts"] or 0) < min_attempts
            and not r["beyond_fg_limit"]]
    beyond = [r for r in rows if r["beyond_fg_limit"]]
    worst = max(data, key=lambda r: r["sim_minus_empirical"], default=None)
    w = np.array([r["train_attempts"] for r in data], dtype=float)
    g = np.array([r["sim_minus_empirical"] for r in data], dtype=float)

    def mean_sim(rs: list) -> float | None:
        return float(np.mean([r["sim_make_rate"] for r in rs])) if rs else None

    return {
        "min_attempts": min_attempts,
        "bands_with_data": len(data),
        "weighted_mean_gap_with_data": float((w * g).sum() / w.sum()) if len(data) else None,
        "largest_gap_band": None if worst is None else worst["distance_yards"],
        "largest_gap": None if worst is None else worst["sim_minus_empirical"],
        "largest_gap_train_attempts": None if worst is None else worst["train_attempts"],
        "thin_bands": [r["distance_yards"] for r in thin],
        "sim_make_rate_thin_bands": mean_sim(thin),
        "sim_make_rate_beyond_limit": mean_sim(beyond),
    }


__all__ = [
    "CHOICES",
    "DEFAULTS",
    "KICK_DISTANCE_OFFSET",
    "Support",
    "action_mask",
    "add_support_columns",
    "build_support",
    "config",
    "empirical_fg_rate",
    "fg_gap_summary",
    "fg_made",
    "first_transitions",
    "hard_rules",
    "lookup",
    "masked_actions",
    "punt_net_yards",
    "sim_fg_make",
    "sim_vs_empirical",
]
