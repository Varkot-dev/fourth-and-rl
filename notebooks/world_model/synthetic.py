"""Synthetic games from the solved game model, and the true values a world model is graded by.

The experiment: generate games from the exact game model's simulator
(`fourth_and_rl.mdp_game.Simulator`, the same transition model the backward induction
solved) with both teams playing the coach-like policy, train the world model on them, plan
fourth downs inside the learned model, and compare with the true values, which are known
because the generator is the solved model.

How a simulator step becomes records (see schema.py for the record contract)
---------------------------------------------------------------------------
* One PLAY record per snap the simulator plays (play in a mode, field goal, punt). The
  simulator folds the try and the kickoff into the transition of the scoring play; here
  they are unfolded into their own records, as in real data:
  - after a touchdown (offense or return touchdown) a TRY record: possession = the scoring
    team, no down, yard line 15 for a kick (XP) and 2 for a two-point try (TWO_PT), same
    clock as the snap that scored. Its transition carries the try's points (0, 1 or 2).
    Which try was taken is the simulator's own choice (coach rule or solved choice),
    recomputed here exactly as `Simulator._score` makes it and checked against the points.
  - after a score (and at halftime) a KICKOFF record: possession = the RECEIVING team (the
    real-data convention), yard line 35 (20 for the free kick after a safety), action
    KICKOFF. The game model's kickoff takes no clock, so its transition has elapsed 0. The
    first record of every game is the opening kickoff at 3600 seconds.
  - no kickoff when the clock is at 0:00 after the score (the game is over).
* Kneel-out. The game model ends a game as soon as the team in front can kneel out the
  clock (closed-form rule). That state gets one PLAY record with action KNEEL whose target
  is END (score delta 0, elapsed = the clock left, clipped to 60 as for real data).
* Down and distance. The game model's distance bins are 1..10 exact, 11-15 and 16+; the
  ydstogo code is the bin's representative (1..10, 13, 20), capped at the yard line as
  `mdp_game.step` does (goal to go). So only those 12 values (or fewer near the goal) occur.
* Clock: ticks x tick_seconds (5). First-half records are above 1800 s except a try at
  exactly 1800 after a touchdown on the play that ended the half (bookkeeping is exact
  either way: timeouts are 3 there).
* Timeouts: the game model tracks timeouts only in the last 5:00 (300 s); outside it every
  record shows 3 for each team. On entering the last 5:00 the model draws both teams'
  timeouts from measured shares; the drop from 3 is logged as timeouts used on the
  transition that enters the last 5:00. A transition can carry at most 2 used per team
  (schema), so when a team drops from 3 to 0 the record before that transition shows 2
  (one of the three is logged one transition earlier). Within the last 5:00 the counts are
  the simulator's exactly.
* Actions: play modes map to the appended codes PLAY_NEUTRAL, PLAY_PROTECT, PLAY_CHASE
  (schema.PLAY_MODE_ACTIONS; `schema.go_actions(split)` returns them for a synthetic
  split, so forcing "GO" in the planner picks a play mode), field goal to FG, punt to
  PUNT, kneel to KNEEL. The coach-like policy never kneels except for the kneel-out.
* Possession and score are from the HOME team's side; score = true points (the game model
  clips its score state at +-21 for decisions, the records do not). Header: spread 0
  (identical teams) and the opening receiver, HOME or AWAY with probability 1/2.
* game_ids "<prefix><seed>-<game>", seasons 0 (SYNTHETIC_SEASON), play_ids = record
  number within the game.

Seeds: every chunk of `chunk_games` games has its own generator
default_rng([seed, stream, chunk]); stream 0 is training games, 1 the evaluation pool,
2 the headline pool, 3 the ground-truth rollouts. So the first 1,000 games of a 64,000-game
set are exactly the 1,000-game set (with the default chunk of 1,000), and evaluation games
never share a stream with training games.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import schema as S

HERE = Path(__file__).resolve().parent
GT_DIR = HERE / "ground_truth"
GT_NAME = "gt_v1"
GT_VERSION = "gt_v1"

SYNTHETIC_SEASON = 0
KICKOFF_YARDLINE = 35
FREE_KICK_YARDLINE = 20
XP_YARDLINE = 15
TWO_PT_YARDLINE = 2
STREAM_GAMES, STREAM_EVAL, STREAM_HEADLINE, STREAM_GT = 0, 1, 2, 3

TRAIN_SEED = 1
EVAL_SEED = 20261119
GT_SEED = 20261120

# Ground-truth actions: go in the best play mode by the solved values, go in the coach's
# own mode, field goal, punt.
GT_ACTIONS = ("go_best", "go_coach", "fg", "punt")
HEADLINE = {"seconds": 240, "lead": 4, "down": 4, "ydstogo": 2, "yardline_100": 60,
            "timeouts": (3, 3)}  # fmt: skip
PUNT_MIN_YARDLINE = 30  # the planner's hard rule: no punt inside the opponent's 30

_COLS = ("game", "seq", "phase", "poss", "down", "ytg", "yl", "clock", "diff", "to_h",
         "to_a", "action", "sim_t", "sim_s", "sim_f", "sim_a", "sim_b", "sim_r")  # fmt: skip


def _mg():
    from fourth_and_rl import mdp_game

    return mdp_game


def action_table() -> np.ndarray:
    """schema action code for each mdp_game action code (A_NEUTRAL .. A_OVER)."""
    A = S.ACTION
    return np.array([A["PLAY_NEUTRAL"], A["PLAY_PROTECT"], A["PLAY_CHASE"], A["FG"],
                     A["PUNT"], A["KNEEL"], A["PAD"]], dtype=np.int16)  # fmt: skip


def load_simulator(path: str | Path | None = None):
    """(solution, simulator with the coach table) from a solved game folder (data/mdp)."""
    import joblib

    mg = _mg()
    sol = mg.load_solution(path)
    folder = Path(path) if path is not None else mg.SOLUTION_DIR
    bundle = joblib.load(folder / "tables.joblib")
    return sol, mg.Simulator(sol, coach=bundle["coach"])


# ---------------------------------------------------------------- record logging
class _Log:
    """Record rows collected column by column, sorted by (game, seq) at the end."""

    def __init__(self) -> None:
        self.parts: dict[str, list[np.ndarray]] = {c: [] for c in _COLS}

    def add(self, **cols: Any) -> None:
        n = len(cols["game"])
        if n == 0:
            return
        for c in _COLS:
            v = cols.get(c, -1)
            self.parts[c].append(np.broadcast_to(np.asarray(v, dtype=np.int64), (n,)).copy())

    def table(self) -> dict[str, np.ndarray]:
        out = {c: np.concatenate(v) if v else np.zeros(0, np.int64) for c, v in self.parts.items()}
        order = np.lexsort((out["seq"], out["game"]))
        return {c: v[order] for c, v in out.items()}


def _view(sim, st: dict, idx: np.ndarray, home_side: np.ndarray, diff0: np.ndarray) -> dict:
    """Record columns that describe the batch members idx as they are now."""
    c = sim.model.clock
    t = st["t"][idx]
    home_has = st["side"][idx] == home_side[idx]
    pts = st["pts"][idx]
    d = pts[:, 0] - pts[:, 1]
    diff = diff0[idx] + np.where(home_side[idx] == 0, d, -d)
    a, b = st["a"][idx], st["b"][idx]
    end = (t <= c.endgame_ticks) & (c.endgame_ticks > 0)
    full = S.N_TIMEOUTS
    return {
        "poss": np.where(home_has, S.POSS_CODE["HOME"], S.POSS_CODE["AWAY"]),
        "clock": t * c.tick_seconds,
        "diff": diff,
        "to_h": np.where(end, np.where(home_has, a, b), full),
        "to_a": np.where(end, np.where(home_has, b, a), full),
    }


def _snap_cols(sim, st: dict, idx: np.ndarray) -> dict:
    g = sim.grid
    down, dbin, yl = g.unpack(st["f"][idx])
    return {"down": down, "ytg": np.minimum(g.dist_rep(dbin), yl), "yl": yl,
            "sim_t": st["t"][idx], "sim_s": st["s"][idx], "sim_f": st["f"][idx],
            "sim_a": st["a"][idx], "sim_b": st["b"][idx], "sim_r": st["r"][idx]}  # fmt: skip


def try_is_two(sim, post: str, kind: np.ndarray, pre: dict, t1: np.ndarray,
               a_after: np.ndarray, b_after: np.ndarray) -> np.ndarray:  # fmt: skip
    """The try the simulator took after each touchdown (True = two-point), recomputed exactly
    as `Simulator._score` decides it. pre: pre-play s, r, t of the batch members; t1 the
    time after the play; a_after, b_after the offense and defense timeouts after the play.
    After a touchdown the scorer is the post-play DEFENSE in both cases (the ball changed
    hands after an offense touchdown; after a return touchdown the team scored on keeps it),
    and an end-phase play never redraws timeouts, so the scorer's view is (b_after, a_after).
    """
    mg = _mg()
    m, c = sim.model, sim.model.clock
    mine = kind == mg.K_TD
    ss = np.where(mine, pre["s"], -pre["s"])
    if post == "coach":
        lead = ss + m.td_points
        two_sec = sim._coach_setting("two_point_seconds")
        leads = tuple(sim._coach_setting("two_point_leads"))
        return (t1 * c.tick_seconds <= two_sec) & np.isin(lead, leads)
    rr = np.where(mine, pre["r"], 1 - pre["r"])
    ph = sim._phase_code(pre["t"])
    two = np.zeros(len(kind), dtype=bool)
    for code, name in enumerate(("end", "mid", "first")):
        q = ph == code
        if not q.any():
            continue
        e = m.extra_shape(name)
        if name == "end":
            e1 = np.minimum(b_after[q], e[0] - 1)
            e2 = np.minimum(a_after[q], e[1] - 1)
        else:
            e1, e2 = rr[q] * (e[0] - 1), np.zeros(int(q.sum()), dtype=np.int64)
        two[q] = sim._two[name][t1[q], ss[q] + m.score_max, e1, e2]
    return two


def _play_out(sim, st: dict, gid: np.ndarray, home_side: np.ndarray, diff0: np.ndarray,
              rng: np.random.Generator, log: _Log, seq0: int, policy: str = "coach",
              post: str | None = None, first_action: np.ndarray | None = None,
              max_steps: int = 5000) -> None:  # fmt: skip
    """Play every batch member to the end of the game, logging its records (in place)."""
    mg = _mg()
    post = post or ("coach" if policy == "coach" else "model")
    m = sim.model
    amap = action_table()
    n = len(st["t"])
    done = np.zeros(n, dtype=bool)
    A, PH = S.ACTION, S.PHASE
    for step in range(max_steps):
        idx = np.nonzero(~done)[0]
        if len(idx) == 0:
            return
        seq = seq0 + 4 * step
        t = st["t"][idx]
        ended = t == 0
        over = sim.game_over(st, idx) & ~ended
        if over.any():
            j = idx[over]
            log.add(game=gid[j], seq=seq, phase=PH["PLAY"], action=A["KNEEL"],
                    **_view(sim, st, j, home_side, diff0), **_snap_cols(sim, st, j))  # fmt: skip
        done[idx[ended | over]] = True
        idx = idx[~(ended | over)]
        if len(idx) == 0:
            continue
        if step == 0 and first_action is not None:
            act = np.asarray(first_action)[idx]
        elif policy == "coach":
            act = sim.coach_actions(st, idx, rng)
        else:
            act = sim.optimal_actions(st, idx)
        log.add(game=gid[idx], seq=seq, phase=PH["PLAY"], action=amap[act],
                **_view(sim, st, idx, home_side, diff0), **_snap_cols(sim, st, idx))  # fmt: skip
        pre = {k: st[k][idx].copy() for k in ("s", "a", "b", "r", "t", "side")}
        info = sim.advance(st, idx, act, rng, post)
        kind, pts = info["kind"], info["points"]
        t1 = st["t"][idx]
        after = _view(sim, st, idx, home_side, diff0)
        # TRY record after a touchdown (offense or return touchdown).
        td = (kind == mg.K_TD) | (kind == mg.K_OPPTD)
        if td.any():
            q = np.nonzero(td)[0]
            tp = pts[q] - m.td_points
            two = try_is_two(sim, post, kind[q], {k: v[q] for k, v in pre.items()}, t1[q],
                             st["a"][idx[q]], st["b"][idx[q]])  # fmt: skip
            if np.any((tp == 2) & ~two) or np.any((tp == 1) & two) or np.any((tp < 0) | (tp > 2)):
                raise RuntimeError("try choice does not match the simulator's points")
            scorer = np.where(kind[q] == mg.K_TD, pre["side"][q], 1 - pre["side"][q])
            scorer_home = scorer == home_side[idx[q]]
            log.add(
                game=gid[idx[q]], seq=seq + 1, phase=PH["TRY"],
                poss=np.where(scorer_home, S.POSS_CODE["HOME"], S.POSS_CODE["AWAY"]),
                down=0, ytg=0, yl=np.where(two, TWO_PT_YARDLINE, XP_YARDLINE),
                clock=after["clock"][q], diff=after["diff"][q] - np.where(scorer_home, tp, -tp),
                to_h=after["to_h"][q], to_a=after["to_a"][q],
                action=np.where(two, A["TWO_PT"], A["XP"]),
            )  # fmt: skip
        # KICKOFF record after a score (if the clock is not at 0) and at halftime.
        scored = td | (kind == mg.K_FG) | (kind == mg.K_SAFETY)
        kick = (scored & (t1 > 0)) | info["half_end"]
        if kick.any():
            q = np.nonzero(kick)[0]
            free = (kind[q] == mg.K_SAFETY) & ~info["half_end"][q]
            log.add(game=gid[idx[q]], seq=seq + 2, phase=PH["KICKOFF"], down=0, ytg=0,
                    yl=np.where(free, FREE_KICK_YARDLINE, KICKOFF_YARDLINE),
                    action=A["KICKOFF"], **{k: v[q] for k, v in after.items()})  # fmt: skip
    raise RuntimeError("synthetic games did not finish; raise max_steps")


def _final_diff(st: dict, home_side: np.ndarray, diff0: np.ndarray) -> np.ndarray:
    d = st["pts"][:, 0] - st["pts"][:, 1]
    return diff0 + np.where(home_side == 0, d, -d)


def _kickoff_games(sim, n: int, rng: np.random.Generator, gid0: int, log: _Log):
    """Start n games at the opening kickoff, log the kickoff records, return the state."""
    mg = _mg()
    c, k = sim.model.clock, sim.model.kern
    receiver_home = rng.random(n) < 0.5
    home_side = np.where(receiver_home, 0, 1).astype(np.int64)  # side 0 = opening receiver
    yl = mg._sample_dist(k.kickoff[1:], rng.random(n)) + 1
    st = sim.start(n, t=c.n_ticks, s=0, f=sim.f10[yl], r=0)
    gid = gid0 + np.arange(n)
    view = _view(sim, st, np.arange(n), home_side, np.zeros(n, dtype=np.int64))
    log.add(game=gid, seq=0, phase=S.PHASE["KICKOFF"], down=0, ytg=0, yl=KICKOFF_YARDLINE,
            action=S.ACTION["KICKOFF"], **view)  # fmt: skip
    return st, gid, home_side, receiver_home


def _chunk(sim, n: int, rng: np.random.Generator, policy: str, post: str | None) -> dict:
    """Records of n games from kickoff (columns), their final scores and opening receivers."""
    log = _Log()
    st, gid, home_side, receiver_home = _kickoff_games(sim, n, rng, 0, log)
    diff0 = np.zeros(n, dtype=np.int64)
    _play_out(sim, st, gid, home_side, diff0, rng, log, seq0=4, policy=policy, post=post)
    return {"cols": log.table(), "final_diff": _final_diff(st, home_side, diff0),
            "receiver_home": receiver_home}  # fmt: skip


# ---------------------------------------------------------------- assembly into a Split
def fix_timeout_drops(game: np.ndarray, to: np.ndarray) -> np.ndarray:
    """A transition may use at most 2 timeouts per team (schema). Where a team drops by 3
    (from 3 on entering the last 5:00 to a drawn 0), log one of them on the transition
    before: the record before the drop shows 2. Returns the corrected counts."""
    to = to.copy()
    same = game[1:] == game[:-1]
    for i in np.nonzero(same & (to[:-1] - to[1:] > 2))[0]:
        if i == 0 or game[i - 1] != game[i] or to[i - 1] - (to[i] - 1) > 2:
            raise RuntimeError("cannot spread a timeout drop of 3 over two transitions")
        to[i] -= 1
    return to


def assemble(cols: dict, final_diff: np.ndarray, receiver_home: np.ndarray,
             game_ids: list[str]) -> S.Split:  # fmt: skip
    """Turn record columns (sorted by game, seq; games numbered 0..G-1) into a Split."""
    game = cols["game"]
    n = len(game)
    G = len(game_ids)
    counts = np.bincount(game, minlength=G)
    if np.any(counts == 0):
        raise ValueError("a game has no records")
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    last = np.zeros(n, bool)
    last[offsets[1:] - 1] = True
    nxt = np.minimum(np.arange(n) + 1, n - 1)
    phase, poss = cols["phase"].astype(np.int16), cols["poss"].astype(np.int16)
    down, ytg = cols["down"].astype(np.int16), cols["ytg"].astype(np.int16)
    yl, action = cols["yl"].astype(np.int16), cols["action"].astype(np.int16)
    clock, diff = cols["clock"].astype(np.int32), cols["diff"].astype(np.int32)
    to_h = fix_timeout_drops(game, cols["to_h"]).astype(np.int32)
    to_a = fix_timeout_drops(game, cols["to_a"]).astype(np.int32)
    fd = np.asarray(final_diff, dtype=np.int32)

    delta = diff[nxt] - diff
    delta[last] = fd - diff[last]
    if not np.isin(delta, S.SCORE_DELTAS[1:]).all():
        bad_d = sorted(set(delta.tolist()) - set(S.SCORE_DELTAS))
        raise RuntimeError(f"illegal score deltas {bad_d}")
    elapsed = clock - clock[nxt]
    elapsed[last] = clock[last]
    crossing = ~last & (clock > S.HALF_SECONDS) & (clock[nxt] <= S.HALF_SECONDS)
    used_h = np.where(crossing, 0, to_h - to_h[nxt])
    used_a = np.where(crossing, 0, to_a - to_a[nxt])
    bad = ~last & ((elapsed < 0) | (elapsed > S.MAX_ELAPSED) & ~crossing
                   | (used_h < 0) | (used_h > 2) | (used_a < 0) | (used_a > 2))  # fmt: skip
    if bad.any():
        raise RuntimeError(f"{int(bad.sum())} transitions break the schema's bookkeeping")

    T = S.TARGET_INDEX
    targets = np.zeros((n, len(S.TARGET_FIELDS)), np.int16)
    targets[:, T["phase"]] = np.where(last, S.PHASE["END"], phase[nxt])
    targets[:, T["poss"]] = poss[nxt]
    code = np.zeros(max(S.SCORE_DELTAS) - min(S.SCORE_DELTAS) + 1, np.int16)
    for v, cde in S.SCORE_DELTA_CODE.items():
        code[v - min(S.SCORE_DELTAS)] = cde
    targets[:, T["score_delta"]] = code[delta - min(S.SCORE_DELTAS)]
    targets[:, T["elapsed"]] = S.elapsed_code(elapsed)
    targets[:, T["to_used_home"]] = np.clip(used_h, 0, 2) + 1
    targets[:, T["to_used_away"]] = np.clip(used_a, 0, 2) + 1
    for name, arr in (("down", down), ("ydstogo", ytg), ("yardline", yl), ("action", action)):
        targets[:, T[name]] = arr[nxt]
    end_zero = [T[f] for f, _ in S.TARGET_FIELDS if f not in ("phase", "score_delta", "elapsed")]
    targets[np.ix_(last, end_zero)] = 0

    SI = S.STATE_INDEX
    states = np.zeros((n, len(S.STATE_FIELDS)), np.int16)
    for name, arr in (("phase", phase), ("poss", poss), ("down", down), ("ydstogo", ytg),
                      ("yardline", yl), ("action", action)):  # fmt: skip
        states[:, SI[name]] = arr
    states[:, SI["clock"]] = S.clock_code(clock)
    states[:, SI["score"]] = S.score_code(diff)
    states[:, SI["to_home"]] = S.timeout_code(to_h)
    states[:, SI["to_away"]] = S.timeout_code(to_a)
    header = np.zeros((G, len(S.HEADER_FIELDS)), np.int16)
    header[:, 0] = S.spread_code(np.zeros(G))
    header[:, 1] = np.where(receiver_home, S.POSS_CODE["HOME"], S.POSS_CODE["AWAY"])
    play_ids = (np.arange(n) - np.repeat(offsets[:-1], counts)).astype(np.int64)
    return S.Split(
        states=states,
        raw=np.stack([clock, diff, to_h, to_a], axis=1).astype(np.int32),
        targets=targets,
        offsets=offsets,
        header=header,
        final_diff=fd.astype(np.int16),
        game_ids=list(game_ids),
        seasons=np.full(G, SYNTHETIC_SEASON, np.int16),
        play_ids=play_ids,
    )


def concat(splits: list[S.Split]) -> S.Split:
    """Splits one after another (same as run.concat_splits, without importing run)."""
    offs, base = [np.zeros(1, dtype=np.int64)], 0
    for s in splits:
        offs.append(np.asarray(s.offsets[1:], dtype=np.int64) + base)
        base += int(s.offsets[-1])
    cat = np.concatenate
    return S.Split(
        states=cat([s.states for s in splits]), raw=cat([s.raw for s in splits]),
        targets=cat([s.targets for s in splits]), offsets=cat(offs),
        header=cat([s.header for s in splits]), final_diff=cat([s.final_diff for s in splits]),
        game_ids=[g for s in splits for g in s.game_ids],
        seasons=cat([s.seasons for s in splits]), play_ids=cat([s.play_ids for s in splits]),
    )  # fmt: skip


def subset(split: S.Split, games) -> S.Split:
    """A split holding only the listed games (same as run.subset_split)."""
    games = np.asarray(games, dtype=np.int64)
    rec = [np.arange(split.offsets[g], split.offsets[g + 1]) for g in games]
    lens = np.array([len(r) for r in rec], dtype=np.int64)
    idx = np.concatenate(rec) if rec else np.zeros(0, dtype=np.int64)
    return S.Split(
        states=split.states[idx], raw=split.raw[idx], targets=split.targets[idx],
        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        header=split.header[games], final_diff=split.final_diff[games],
        game_ids=[split.game_ids[g] for g in games], seasons=split.seasons[games],
        play_ids=split.play_ids[idx],
    )  # fmt: skip


# ---------------------------------------------------------------- parallel chunks
_WORKER_SIM: Any = None


def _run_chunks(fn, args: list[tuple], n_workers: int) -> list:
    """fn(sim, *a) for each a, in order, on n_workers forked processes sharing the
    simulator (copy-on-write); in process when n_workers <= 1 or fork is unavailable."""
    import multiprocessing as mp

    if n_workers <= 1 or len(args) <= 1 or "fork" not in mp.get_all_start_methods():
        return [fn(_WORKER_SIM, *a) for a in args]
    ctx = mp.get_context("fork")
    with ctx.Pool(min(n_workers, len(args))) as pool:
        return pool.starmap(_call, [(fn, a) for a in args], chunksize=1)


def _call(fn, a: tuple):
    return fn(_WORKER_SIM, *a)


def _games_chunk(sim, seed: int, stream: int, c: int, n: int, policy: str, post: str | None):
    rng = np.random.default_rng([int(seed), int(stream), int(c)])
    return _chunk(sim, n, rng, policy, post)


def _generate(sim, n_games: int, seed: int, stream: int, chunk_games: int, n_workers: int,
              policy: str, post: str | None, id_prefix: str, keep_cols: bool = False):  # fmt: skip
    global _WORKER_SIM
    _WORKER_SIM = sim
    sizes = [min(chunk_games, n_games - a) for a in range(0, n_games, chunk_games)]
    args = [(seed, stream, c, n, policy, post) for c, n in enumerate(sizes)]
    parts = _run_chunks(_games_chunk, args, n_workers)
    splits, cols_all = [], []
    g0 = 0
    for c, (p, n) in enumerate(zip(parts, sizes, strict=True)):
        ids = [f"{id_prefix}{seed}-{stream}-{g0 + g:07d}" for g in range(n)]
        splits.append(assemble(p["cols"], p["final_diff"], p["receiver_home"], ids))
        if keep_cols:
            cols_all.append((c, p))
        g0 += n
    out = concat(splits) if splits else None
    return (out, cols_all) if keep_cols else out


def simulate_games(sim, n_games: int, seed: int = TRAIN_SEED, chunk_games: int = 1000,
                   n_workers: int = 1, policy: str = "coach", post: str | None = None,
                   id_prefix: str = "syn") -> S.Split:  # fmt: skip
    """`n_games` full regulation games from the opening kickoff, both teams playing the
    coach-like policy (`Simulator.coach_actions`, with the coach rules for timeouts and
    tries), as a schema.Split. See the module docstring for how steps become records.

    Deterministic in (seed, chunk_games); independent of n_workers. With the default chunk
    the first k * 1000 games of a larger set are the k * 1000-game set."""
    return _generate(sim, n_games, seed, STREAM_GAMES, chunk_games, n_workers, policy, post,
                     id_prefix)


def split_sim_columns(cols: dict) -> dict[str, np.ndarray]:
    return {k[4:]: cols[k] for k in ("sim_t", "sim_s", "sim_f", "sim_a", "sim_b", "sim_r")}


# ---------------------------------------------------------------- evaluation states
@dataclass
class EvalStates:
    """Fourth-down states to grade a world model on, with their games.

    split  held-out games (full games; the world model conditions on records up to idx)
    idx    record index of each state in split
    t, s, f, a, b, r  the game model's state at each record: ticks left, TRUE score
           differential of the team with the ball (unclipped), field state, offense and
           defense timeouts (meaningful in the last 5:00), second-half-receiver flag
    kind   0 = sampled from held-out games, 1 = the headline state
    """

    split: S.Split
    idx: np.ndarray
    t: np.ndarray
    s: np.ndarray
    f: np.ndarray
    a: np.ndarray
    b: np.ndarray
    r: np.ndarray
    kind: np.ndarray

    def __len__(self) -> int:
        return len(self.idx)

    def sim_state(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in ("t", "s", "f", "a", "b", "r")}


DEFAULT_STRATA = ((1800, 3600, 0.20), (900, 1800, 0.30), (300, 900, 0.30), (0, 300, 0.20))


def _true_offense_diff(split: S.Split, idx: np.ndarray) -> np.ndarray:
    home = split.states[idx, S.STATE_INDEX["poss"]] == S.POSS_CODE["HOME"]
    return np.where(home, 1, -1) * split.raw[idx, 1].astype(np.int64)


def _headline_game(sim, seed: int, chunk_games: int, max_chunks: int, gid: int,
                   policy: str, post: str | None, id_prefix: str):  # fmt: skip
    """A held-out game that reaches the headline state: a prefix of a real synthetic game
    that is on third down between 4:05 and 5:00, leading by 4 with 3 timeouts each, near
    its own 35 to 40, then the headline record (fourth and 2 at the own 40, 4:00, up 4,
    3 timeouts each) as the next snap, then the game played out from it by the same
    policy. Returns (split of that one game, record index of the headline state)."""
    g, c = sim.grid, sim.model.clock
    h = HEADLINE
    t_h = h["seconds"] // c.tick_seconds
    f_h = int(g.fidx(h["down"], g.dist_bin(h["ydstogo"]), h["yardline_100"]))
    best = None  # (rank, chunk, part, record); rank 0 = exact third down, 1 = loose, 2 = any
    for ch in range(max_chunks):
        rng = np.random.default_rng([int(seed), STREAM_HEADLINE, ch])
        part = _chunk(sim, chunk_games, rng, policy, post)
        cl = part["cols"]
        home = cl["poss"] == S.POSS_CODE["HOME"]
        lead = np.where(home, cl["diff"], -cl["diff"])
        cand = ((cl["phase"] == S.PHASE["PLAY"]) & (cl["down"] == 3) & (lead == h["lead"])
                & (cl["clock"] > h["seconds"]) & (cl["clock"] <= h["seconds"] + S.MAX_ELAPSED)
                & (cl["sim_a"] == 3) & (cl["sim_b"] == 3) & (cl["yl"] >= h["yardline_100"])
                & (cl["yl"] <= h["yardline_100"] + 15))  # fmt: skip
        cand = np.nonzero(cand)[0]
        if len(cand) == 0:
            continue
        # A third down that a short gain turns into fourth and 2 at the own 40: exactly
        # (to go = 2 + yards to the 40), or at least not a first down (to go >= that).
        need = h["ydstogo"] + (cl["yl"][cand] - h["yardline_100"])
        rank = np.where(cl["ytg"][cand] == need, 0, np.where(cl["ytg"][cand] >= need, 1, 2))
        k = int(np.argmin(rank))
        if best is None or rank[k] < best[0]:
            best = (int(rank[k]), ch, part, int(cand[k]))
        if best[0] == 0:
            break
    if best is not None:
        _, ch, part, pick = best
        cl = part["cols"]
        game = cl["game"][pick]
        rows = np.nonzero((cl["game"] == game) & (np.arange(len(cl["game"])) <= pick))[0]
        log = _Log()
        prefix = {k: v[rows] for k, v in cl.items()}
        prefix["game"] = np.zeros(len(rows), np.int64)
        log.add(**prefix)
        receiver_home = bool(part["receiver_home"][game])
        home_side = np.array([0 if cl["poss"][pick] == S.POSS_CODE["HOME"] else 1])
        diff0 = np.array([cl["diff"][pick]], dtype=np.int64)
        st = sim.start(1, t=t_h, s=h["lead"], f=f_h, a=h["timeouts"][0], b=h["timeouts"][1], r=0)
        rng2 = np.random.default_rng([int(seed), STREAM_HEADLINE, 10_000 + ch])
        _play_out(sim, st, np.zeros(1, np.int64), home_side, diff0, rng2, log,
                  seq0=int(cl["seq"][pick]) + 4, policy=policy, post=post)  # fmt: skip
        cols = log.table()
        fd = _final_diff(st, home_side, diff0)
        sp = assemble(cols, fd, np.array([receiver_home]), [f"{id_prefix}headline-{gid}"])
        hidx = len(rows)
        if not (cols["sim_t"][hidx] == t_h and cols["sim_f"][hidx] == f_h):
            raise RuntimeError("headline record not where expected")
        return sp, hidx, split_sim_columns({k: v[hidx : hidx + 1] for k, v in cols.items()})
    raise RuntimeError("no game reached the headline state's neighbourhood; raise max_chunks")


def evaluation_states(sim, n_states: int = 300, seed: int = EVAL_SEED, n_pool_games: int = 4000,
                      strata=DEFAULT_STRATA, max_abs_diff: int = 16, chunk_games: int = 1000,
                      n_workers: int = 1, policy: str = "coach", post: str | None = None,
                      headline: bool = True, headline_chunks: int = 40) -> EvalStates:  # fmt: skip
    """Fourth-down states from held-out synthetic games (seed `seed`, stream 1: never a
    training stream), stratified by clock (default 20% first half, 30% third quarter, 30%
    fourth quarter before 5:00, 20% last 5:00) among states where the team with the ball is
    within `max_abs_diff` points (larger margins are decided games), plus the headline state
    (last, kind 1) if `headline`. Each state comes with its full game in `split`."""
    pool, cols = _generate(sim, n_pool_games, seed, STREAM_EVAL, chunk_games, n_workers, policy,
                           post, "eval", keep_cols=True)  # fmt: skip
    sim_cols = {k: np.concatenate([split_sim_columns(p["cols"])[k] for _, p in cols])
                for k in ("t", "s", "f", "a", "b", "r")}  # fmt: skip
    st = pool.states
    SI = S.STATE_INDEX
    clock = pool.raw[:, 0]
    diff_off = _true_offense_diff(pool, np.arange(len(st)))
    fourth = ((st[:, SI["phase"]] == S.PHASE["PLAY"]) & (st[:, SI["down"]] == sim.grid.n_down)
              & (np.abs(diff_off) <= max_abs_diff))  # fmt: skip
    rng = np.random.default_rng([int(seed), STREAM_EVAL, 99_999])
    n_sample = n_states - (1 if headline else 0)
    w = np.array([s[2] for s in strata], dtype=float)
    want = np.floor(w / w.sum() * n_sample).astype(int)
    want[np.argmax(w)] += n_sample - want.sum()
    chosen = []
    for (lo, hi, _), k in zip(strata, want, strict=True):
        pool_k = np.nonzero(fourth & (clock > lo) & (clock <= hi))[0]
        if len(pool_k) < k:
            raise RuntimeError(f"only {len(pool_k)} fourth downs between {lo} and {hi} s")
        chosen.append(rng.choice(pool_k, size=k, replace=False))
    rec = np.sort(np.concatenate(chosen))
    games = np.searchsorted(pool.offsets, rec, side="right") - 1
    ug = np.unique(games)
    sub = subset(pool, ug)
    gpos = np.searchsorted(ug, games)
    idx = sub.offsets[gpos] + (rec - pool.offsets[games])
    parts = {"t": sim_cols["t"][rec], "s": diff_off[rec], "f": sim_cols["f"][rec],
             "a": sim_cols["a"][rec], "b": sim_cols["b"][rec], "r": sim_cols["r"][rec]}  # fmt: skip
    kind = np.zeros(len(rec), np.int64)
    if headline:
        hs, hidx, hcols = _headline_game(sim, seed, chunk_games, headline_chunks, 0, policy,
                                         post, "eval")  # fmt: skip
        idx = np.r_[idx, int(sub.offsets[-1]) + hidx]
        sub = concat([sub, hs])
        for k in parts:
            v = HEADLINE["lead"] if k == "s" else int(hcols[k][0])
            parts[k] = np.r_[parts[k], v]
        kind = np.r_[kind, 1]
    return EvalStates(split=sub, idx=idx.astype(np.int64), kind=kind,
                      **{k: np.asarray(v, dtype=np.int64) for k, v in parts.items()})


# ---------------------------------------------------------------- ground truth
def coach_go_mode(sim, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The play mode the coach-like policy uses when it goes (mdp_game code): chase when
    trailing (or tied inside the two-minute warning) in the last 5:00, protect when leading
    there, neutral otherwise. Deterministic given the state (`Simulator.coach_actions`)."""
    mg = _mg()
    c = sim.model.clock
    end = (t <= c.endgame_ticks) & (c.endgame_ticks > 0)
    warn = c.warning_tick if c.warning_tick is not None else 0
    chase = (s < 0) | ((s == 0) & (t <= warn))
    return np.where(end & chase, mg.A_CHASE, np.where(end & (s > 0), mg.A_PROTECT, mg.A_NEUTRAL))


def coach_choice_probs(sim, states) -> np.ndarray:
    """[N, 3] probabilities that the coach-like policy goes, kicks, punts at each state,
    by drawing its (random) fourth-down choice 4,000 times per state."""
    mg = _mg()
    ss = _as_sim_state(states)
    n = len(ss["t"])
    reps = 4000
    st = sim.start(n * reps, **{k: np.repeat(np.clip(v, -sim.model.score_max, sim.model.score_max)
                                             if k == "s" else v, reps) for k, v in ss.items()})
    act = sim.coach_actions(st, np.arange(n * reps), np.random.default_rng(0)).reshape(n, reps)
    go = np.isin(act, (mg.A_NEUTRAL, mg.A_PROTECT, mg.A_CHASE))
    return np.stack([go.mean(1), (act == mg.A_FG).mean(1), (act == mg.A_PUNT).mean(1)], axis=1)


def _as_sim_state(states) -> dict[str, np.ndarray]:
    if isinstance(states, EvalStates):
        return states.sim_state()
    return {k: np.atleast_1d(np.asarray(states[k], dtype=np.int64))
            for k in ("t", "s", "f", "a", "b", "r")}  # fmt: skip


def _gt_chunk(sim, seed: int, c: int, ss: dict, codes: np.ndarray, n_rollouts: int,
              policy: str, post: str | None) -> dict:  # fmt: skip
    """Rollouts for one chunk of (state, action) pairs: ss arrays and codes per pair."""
    mg = _mg()
    rng = np.random.default_rng([int(seed), STREAM_GT, int(c)])
    m = len(codes)
    rep = {k: np.repeat(v, n_rollouts) for k, v in ss.items()}
    st = sim.start(m * n_rollouts, **rep)
    out = mg.simulate(sim, st, rng, policy=policy, post=post,
                      first_action=np.repeat(codes, n_rollouts))  # fmt: skip
    margin = rep["s"] + out["points"][:, 0] - out["points"][:, 1]
    true = S.outcome(margin).reshape(m, n_rollouts)
    model = out["result"].reshape(m, n_rollouts)
    return {"mean": true.mean(1), "se": true.std(1, ddof=1) / np.sqrt(n_rollouts),
            "model_mean": model.mean(1),
            "differ": (true != model).sum(1)}  # fmt: skip


def ground_truth(sim, sol, states, n_rollouts: int = 16000, seed: int = GT_SEED,
                 pairs_per_chunk: int = 12, n_workers: int = 1, policy: str = "coach",
                 post: str | None = None) -> dict[str, np.ndarray]:  # fmt: skip
    """True values of each evaluation state and fourth-down action (GT_ACTIONS).

    q_coach [N, 4]   win probability of the team with the ball when it takes the action and
                     both teams then follow `policy` (default the coach-like policy, with
                     the coach rules for timeouts and tries), by Monte Carlo with the true
                     simulator, scored by the TRUE final points (as the world model's
                     records are); se_coach its standard error. NaN where not allowed.
    q_coach_model    the same games scored by the game model's clipped score (equal except
                     when a margin passed 21 points and came back)
    q_star [N, 4]    the solved value (`Solution.q_values`): the action, then optimal play
                     by both teams (model rules for timeouts and tries)
    q_star_modes [N, 3]  solved values of going in each mode (neutral, protect, chase)
    allowed_model [N, 4] the action exists in the true model (field goal in range per
                     the kernel's fg_ok, punt per punt_ok)
    allowed_rule [N, 4]  also passes the planner's hard rule (no punt inside the opponent's
                     30); the field-goal rule depends on the training data, see support.py
    go_best_code, go_coach_code  mdp_game action codes used for the two go columns
    coach_probs [N, 3]   the coach-like policy's own P(go, FG, punt) at the state
    v_star [N]       the solved value of the state
    """
    global _WORKER_SIM
    mg = _mg()
    ss = _as_sim_state(states)
    n = len(ss["t"])
    g, kern, c = sim.grid, sim.model.kern, sim.model.clock
    down, dbin, yl = g.unpack(ss["f"])
    if np.any(down != g.n_down):
        raise ValueError("ground truth is for fourth-down states only")
    mode_names = ("play", "play_protect", "play_chase")
    mode_codes = (mg.A_NEUTRAL, mg.A_PROTECT, mg.A_CHASE)
    q_star = np.full((n, 4), np.nan)
    q_modes = np.full((n, 3), np.nan)
    v_star = np.full(n, np.nan)
    order = np.argsort(-ss["t"], kind="stable")  # one solver slice per distinct time
    for i in order:
        gs = mg.GameState(
            seconds=int(ss["t"][i]) * c.tick_seconds, score=int(ss["s"][i]), down=int(down[i]),
            ydstogo=int(min(g.dist_rep(dbin[i]), yl[i])), yardline_100=int(yl[i]),
            off_timeouts=int(ss["a"][i]), def_timeouts=int(ss["b"][i]),
            receives_second_half=int(ss["r"][i]),
        )  # fmt: skip
        q = sol.q_values(gs)
        v_star[i] = sol.value(gs)
        for j, name in enumerate(mode_names):
            q_modes[i, j] = q.get(name, np.nan)
        q_star[i, 2] = q.get("field_goal", np.nan)
        q_star[i, 3] = q.get("punt", np.nan)
    best = np.nanargmax(q_modes, axis=1)
    go_best = np.asarray(mode_codes)[best]
    s_clip = np.clip(ss["s"], -sim.model.score_max, sim.model.score_max)
    go_coach = coach_go_mode(sim, ss["t"], s_clip)
    q_star[:, 0] = q_modes[np.arange(n), best]
    q_star[:, 1] = q_modes[np.arange(n), go_coach]
    allowed_model = np.ones((n, 4), dtype=bool)
    allowed_model[:, 2] = kern.fg_ok[ss["f"]]
    allowed_model[:, 3] = kern.punt_ok[ss["f"]]
    allowed_rule = allowed_model.copy()
    allowed_rule[:, 3] &= yl >= PUNT_MIN_YARDLINE

    # (state, action) pairs to simulate; go_coach reuses go_best when the modes agree.
    codes = np.stack([go_best, go_coach, np.full(n, mg.A_FG), np.full(n, mg.A_PUNT)], axis=1)
    sim_cell = allowed_model.copy()
    sim_cell[:, 1] &= go_coach != go_best
    pi, pj = np.nonzero(sim_cell)
    args = []
    for ci, a in enumerate(range(0, len(pi), pairs_per_chunk)):
        sl = slice(a, a + pairs_per_chunk)
        sub = {k: v[pi[sl]] for k, v in ss.items()}
        args.append((seed, ci, sub, codes[pi[sl], pj[sl]], int(n_rollouts), policy, post))
    _WORKER_SIM = sim
    parts = _run_chunks(_gt_chunk, args, n_workers)
    q_coach = np.full((n, 4), np.nan)
    se = np.full((n, 4), np.nan)
    q_model = np.full((n, 4), np.nan)
    differ = 0
    for (a, p) in zip(range(0, len(pi), pairs_per_chunk), parts, strict=True):
        sl = slice(a, a + pairs_per_chunk)
        q_coach[pi[sl], pj[sl]] = p["mean"]
        se[pi[sl], pj[sl]] = p["se"]
        q_model[pi[sl], pj[sl]] = p["model_mean"]
        differ += int(p["differ"].sum())
    same = allowed_model[:, 1] & (go_coach == go_best)
    for arr in (q_coach, se, q_model):
        arr[same, 1] = arr[same, 0]
    return {
        "q_coach": q_coach, "se_coach": se, "q_coach_model": q_model, "q_star": q_star,
        "q_star_modes": q_modes, "allowed_model": allowed_model, "allowed_rule": allowed_rule,
        "go_best_code": go_best, "go_coach_code": go_coach,
        "coach_probs": coach_choice_probs(sim, ss), "v_star": v_star,
        "n_rollouts": np.int64(n_rollouts), "outcomes_true_vs_clipped_differ": np.int64(differ),
    }  # fmt: skip


# ---------------------------------------------------------------- files and verification
_SPLIT_ARRAYS = ("states", "raw", "targets", "offsets", "header", "final_diff", "seasons",
                 "play_ids")  # fmt: skip


def save_split(path: Path, split: S.Split, **extra: np.ndarray) -> None:
    """Same file layout as run.save_split, so run.load_split reads it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(getattr(split, k)) for k in _SPLIT_ARRAYS}
    arrays["game_ids"] = np.asarray(list(split.game_ids), dtype=str)
    arrays.update({f"extra_{k}": np.asarray(v) for k, v in extra.items()})
    np.savez_compressed(path, **arrays)


def load_split(path: Path) -> tuple[S.Split, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        kw = {k: z[k] for k in _SPLIT_ARRAYS}
        kw["game_ids"] = [str(x) for x in z["game_ids"]]
        extra = {k[6:]: z[k] for k in z.files if k.startswith("extra_")}
    return S.Split(**kw), extra


def load_eval_states(path: Path | None = None) -> EvalStates:
    sp, ex = load_split(path or GT_DIR / f"{GT_NAME}_states.npz")
    return EvalStates(split=sp, idx=ex["idx"], kind=ex["kind"],
                      **{k: ex[f"sim_{k}"] for k in ("t", "s", "f", "a", "b", "r")})  # fmt: skip


def load_ground_truth(path: Path | None = None) -> tuple[dict[str, np.ndarray], dict]:
    path = path or GT_DIR / f"{GT_NAME}.npz"
    with np.load(path, allow_pickle=False) as z:
        gt = {k: z[k] for k in z.files}
    meta = json.loads(path.with_suffix(".json").read_text())
    return gt, meta


def _strip_fit_times(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip_fit_times(v) for k, v in x.items() if k != "fitted_at"}
    if isinstance(x, list):
        return [_strip_fit_times(v) for v in x]
    return x


def solution_stamp(sol_meta: dict) -> dict:
    """What identifies a solved game: its fit stamp, tables key and solve seed (JSON form)."""
    keep = {k: sol_meta.get(k) for k in ("stamp", "tables_key", "seed")}
    return json.loads(json.dumps(keep, default=float))


def verify_ground_truth(sol_meta: dict, meta_path: Path | None = None, sol=None,
                        tol: float = 1e-5) -> dict:  # fmt: skip
    """Does the solved game here (its meta.json as `sol_meta`) match the one the saved ground
    truth was computed with? Compares the fit stamp, tables key (data fingerprint) and solve
    seed, ignoring fit timestamps; with `sol`, also the solved values at every evaluation
    state. Returns {"ok": bool, "mismatches": [...]}. If not ok, recompute a smaller ground
    truth with `ground_truth` here."""
    meta_path = meta_path or GT_DIR / f"{GT_NAME}.json"
    meta = json.loads(Path(meta_path).read_text())
    saved = _strip_fit_times(meta["solution"])
    now = _strip_fit_times(solution_stamp(sol_meta))
    bad = [f"{k} differs" for k in sorted(set(saved) | set(now)) if saved.get(k) != now.get(k)]
    if sol is not None:
        mg = _mg()
        gt, _ = load_ground_truth(Path(meta_path).with_suffix(".npz"))
        ev = load_eval_states(Path(meta_path).with_name(f"{Path(meta_path).stem}_states.npz"))
        g, c = sol.grid, sol.model.clock
        down, dbin, yl = g.unpack(ev.f)
        worst = 0.0
        for i in range(len(ev)):
            gs = mg.GameState(int(ev.t[i]) * c.tick_seconds, int(ev.s[i]), int(down[i]),
                              int(min(g.dist_rep(dbin[i]), yl[i])), int(yl[i]), int(ev.a[i]),
                              int(ev.b[i]), int(ev.r[i]))  # fmt: skip
            worst = max(worst, abs(sol.value(gs) - float(gt["v_star"][i])))
        if worst > tol:
            bad.append(f"solved values differ by up to {worst:.2e} at the evaluation states")
    return {"ok": not bad, "mismatches": bad}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def code_version() -> dict:
    """Hashes of the code that generated the data and the git commit, if available."""
    mg = _mg()
    from fourth_and_rl import mdp

    out = {"synthetic.py": _sha(Path(__file__)), "schema.py": _sha(HERE / "schema.py"),
           "mdp_game.py": _sha(Path(mg.__file__)), "mdp.py": _sha(Path(mdp.__file__))}  # fmt: skip
    try:
        out["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, text=True,
                                         capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        out["git_head"] = None
    return out


def benchmark(sim, n_games: int = 2000, seed: int = 0, chunk_games: int = 1000,
              n_workers: int = 1) -> dict:  # fmt: skip
    """Games per second of `simulate_games` on this machine."""
    t0 = time.time()
    sp = simulate_games(sim, n_games, seed, chunk_games=chunk_games, n_workers=n_workers)
    sec = time.time() - t0
    return {"games": n_games, "seconds": round(sec, 2), "games_per_second": round(n_games / sec, 1),
            "records_per_game": round(float(np.diff(sp.offsets).mean()), 1),
            "n_workers": n_workers, "chunk_games": chunk_games}  # fmt: skip


def precompute(n_states: int = 300, n_rollouts: int = 16000, n_workers: int = 3,
               out_dir: Path = GT_DIR, name: str = GT_NAME, game=None,
               strata=DEFAULT_STRATA, n_pool_games: int = 4000, max_abs_diff: int = 16,
               headline: bool = True, bench: bool = True) -> dict:  # fmt: skip
    """Build the evaluation states and their ground truth and save them (see the README of
    the experiment: gt_v1.npz, gt_v1.json, gt_v1_states.npz). `game` is an already loaded
    (solution, simulator) pair; by default the solved game under data/mdp is loaded."""
    t0 = time.time()
    sol, sim = game if game is not None else load_simulator()
    t_load = time.time() - t0
    bench_one = benchmark(sim, 3000, seed=0, n_workers=1) if bench else None
    bench_par = benchmark(sim, 3000, seed=0, n_workers=n_workers) if bench else None
    t1 = time.time()
    ev = evaluation_states(sim, n_states, EVAL_SEED, n_pool_games=n_pool_games, strata=strata,
                           max_abs_diff=max_abs_diff, n_workers=n_workers, headline=headline)
    t_eval = time.time() - t1
    t2 = time.time()
    gt = ground_truth(sim, sol, ev, n_rollouts, GT_SEED, n_workers=n_workers)
    t_gt = time.time() - t2
    out_dir.mkdir(parents=True, exist_ok=True)
    save_split(out_dir / f"{name}_states.npz", ev.split, idx=ev.idx, kind=ev.kind,
               **{f"sim_{k}": getattr(ev, k) for k in ("t", "s", "f", "a", "b", "r")})  # fmt: skip
    np.savez_compressed(out_dir / f"{name}.npz", idx=ev.idx, kind=ev.kind, **gt)
    se = gt["se_coach"][np.isfinite(gt["se_coach"])]
    meta = {
        "version": GT_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "actions": list(GT_ACTIONS),
        "solution": solution_stamp(sol.meta),
        "data_fingerprint": sol.meta.get("tables_key", {}).get("data_fingerprint", {}).get("hash"),
        "code": code_version(),
        "seeds": {"eval_states": EVAL_SEED, "ground_truth": GT_SEED, "train_default": TRAIN_SEED,
                  "streams": {"games": STREAM_GAMES, "eval": STREAM_EVAL,
                              "headline": STREAM_HEADLINE, "ground_truth": STREAM_GT}},
        "n_states": len(ev), "n_rollouts": n_rollouts, "policy": "coach", "post": "coach",
        "strata_seconds": [list(s) for s in strata], "max_abs_diff": max_abs_diff,
        "headline": HEADLINE if headline else None,
        "headline_index": int(np.nonzero(ev.kind == 1)[0][0]) if headline else None,
        "se_coach": {"max": float(se.max()), "mean": float(se.mean()),
                     "median": float(np.median(se))},
        "outcomes_true_vs_clipped_differ": int(gt["outcomes_true_vs_clipped_differ"]),
        "timing_seconds": {"load_simulator": round(t_load, 1), "eval_states": round(t_eval, 1),
                           "ground_truth": round(t_gt, 1)},
        "benchmark": {"one_process": bench_one, "parallel": bench_par,
                      "cpu_count": os.cpu_count()},
        "files": {f: os.path.getsize(out_dir / f) for f in (f"{name}.npz", f"{name}_states.npz")},
    }  # fmt: skip
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    return meta


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("precompute")
    a.add_argument("--n-states", type=int, default=300)
    a.add_argument("--n-rollouts", type=int, default=16000)
    a.add_argument("--workers", type=int, default=3)
    b = sub.add_parser("benchmark")
    b.add_argument("--games", type=int, default=4000)
    b.add_argument("--workers", type=int, default=1)
    args = p.parse_args(argv)
    if args.cmd == "precompute":
        print(json.dumps(precompute(args.n_states, args.n_rollouts, args.workers), indent=1))
    else:
        _, sim = load_simulator()
        print(json.dumps(benchmark(sim, args.games, n_workers=args.workers), indent=1))


if __name__ == "__main__":
    sys.exit(main())
