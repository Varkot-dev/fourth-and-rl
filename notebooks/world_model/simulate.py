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
from train import amp_settings

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
) -> dict | None:
    """Roll every row to the end of its game. Returns None if the deadline passes first."""
    device = device_of(model)
    max_len, max_steps, chunk = model.max_len, int(cfg["max_steps"]), int(cfg["sim_batch"])
    temp, mask_pad = float(cfg["temperature"]), bool(cfg["mask_pad_codes"])
    r = len(lens)
    buf_t = torch.as_tensor(buf, device=device)
    head_t = torch.as_tensor(header.astype(np.int64), device=device)
    lens = lens.astype(np.int64).copy()
    raw = np.asarray(raw, dtype=np.int32).copy()
    active = np.ones(r, dtype=bool)
    steps = np.zeros(r, dtype=np.int64)
    final = raw[:, 1].astype(np.int64).copy()
    appended = np.zeros(r, dtype=np.int64)
    points = np.zeros(r, dtype=np.int64)
    kickoffs = np.zeros(r, dtype=np.int64)
    cnt = dict.fromkeys(COUNTER_KEYS, 0)
    cnt["rollouts"] = r
    enabled, dtype, _ = amp_settings(device)
    while active.any():
        if time.time() > deadline_ts:
            return None
        act = np.flatnonzero(active)
        tg_parts = []
        for a in range(0, len(act), chunk):
            rows = act[a : a + chunk]
            rows_t = torch.as_tensor(rows, device=device)
            n = int(lens[rows].max())
            with torch.autocast(device.type, dtype=dtype, enabled=enabled):
                h = model.encode(head_t[rows_t], buf_t[rows_t, :n])
            ln = torch.as_tensor(lens[rows] - 1, device=device)
            hl = h[torch.arange(len(rows), device=device), ln].float()
            with torch.autocast(device.type, dtype=dtype, enabled=enabled):
                tg_parts.append(model.sample_transition(hl, temp, gen, mask_pad))
        tg = torch.cat(tg_parts).cpu().numpy()
        steps[act] += 1
        cnt["transitions"] += len(act)
        for k, v in check_transitions(tg).items():
            cnt[k] += v
        phase = tg[:, TI["phase"]]
        is_end = phase == PH["END"]
        raw_next = S.apply_transition(raw[act], tg)
        delta = S.score_delta_value(tg[:, TI["score_delta"]]).astype(np.int64)
        if not cfg["score_on_end"]:
            delta = np.where(is_end, 0, delta)
            raw_next[is_end, 1] = raw[act[is_end], 1]
        points[act] += np.abs(delta)
        clock0 = ~is_end & (raw_next[:, 0] <= 0) & (phase != PH["TRY"])
        trunc = ~is_end & ~clock0 & (steps[act] >= max_steps)
        cont = ~(is_end | clock0 | trunc)
        cnt["ended_by_end"] += int(is_end.sum())
        cnt["ended_by_clock"] += int(clock0.sum())
        cnt["truncated_max_steps"] += int(trunc.sum())
        final[act] = raw_next[:, 1]
        raw[act] = raw_next
        stop = act[~cont]
        active[stop] = False
        rows = act[cont]
        if len(rows):
            codes = S.next_state_codes(raw_next[cont], tg[cont]).astype(np.int64)
            appended[rows] += 1
            kickoffs[rows] += codes[:, SI["phase"]] == PH["KICKOFF"]
            full = rows[lens[rows] >= max_len]
            if len(full):
                ft = torch.as_tensor(full, device=device)
                buf_t[ft, :-1] = buf_t[ft, 1:].clone()
            pos = np.minimum(lens[rows], max_len - 1)
            buf_t[torch.as_tensor(rows, device=device), torch.as_tensor(pos, device=device)] = (
                torch.as_tensor(codes, device=device)
            )
            lens[rows] = np.minimum(lens[rows] + 1, max_len)
    out = {"final_diff": final, "steps": steps, "counters": cnt}
    if track:
        out.update({"appended": appended, "points": points, "kickoffs": kickoffs})
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
    Queries are processed in groups of up to cfg["sim_max_live"] rollouts; if the deadline passes,
    the unfinished queries come back as NaN with done=False.
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
    per_group = max(1, int(c["sim_max_live"]) // max(1, n))
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
        out = _run(model, header, buf, lens, raw, c, deadline_ts, gen)
        if out is None:
            break
        o = S.outcome(out["final_diff"]).reshape(len(q), n)
        res["home_wp"][q] = o.mean(1)
        res["home_wp_se"][q] = o.std(1, ddof=1) / np.sqrt(n) if n > 1 else np.nan
        res["mean_steps"][q] = out["steps"].reshape(len(q), n).mean(1)
        res["done"][q] = True
        for k, v in out["counters"].items():
            res["counters"][k] += v
    res["seconds"] = time.time() - t0
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
) -> pd.DataFrame:
    """For every fourth-down PLAY record in idx: win probability of GO, FG and PUNT for the
    team with the ball, standard errors, best action, margin and a "too close" flag (margin
    under 2 standard errors of the difference between the best and second-best action)."""
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    st = split.states[idx]
    keep = (st[:, SI["down"]] == 4) & (st[:, SI["phase"]] == PH["PLAY"])
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
    counters = {}
    for ch in choices:
        r = rollout(model, split, idx, ch, n_rollouts, cfg, deadline_ts, rng)
        wp = np.where(home_has_ball, r["home_wp"], 1.0 - r["home_wp"])
        key = ch.lower()
        df[f"wp_{key}"] = wp
        df[f"se_{key}"] = r["home_wp_se"]
        df[f"done_{key}"] = r["done"]
        counters[ch] = r["counters"]
    wps = df[[f"wp_{c.lower()}" for c in choices]].to_numpy()
    ses = df[[f"se_{c.lower()}" for c in choices]].to_numpy()
    done = df[[f"done_{c.lower()}" for c in choices]].to_numpy().all(1)
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
    df.attrs["counters"] = r["counters"]
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
        "points_per_game": float(pts.mean()),  # points scored on non-final transitions
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
    per = int(c["sim_max_live"])
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
    return {"n_games": done, "sim": sim, "real": real, "table": table, "counters": counters}


__all__ = [
    "COUNTER_KEYS",
    "GO",
    "TARGET_NAMES",
    "check_transitions",
    "plan_fourth_downs",
    "real_game_stats",
    "rollout",
    "simulate_games",
    "wp_by_simulation",
]
