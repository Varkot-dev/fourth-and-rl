"""A tiny synthetic `Split` for tests: short, rule-based games whose codes and transitions are
valid under schema.py and consistent with `schema.apply_transition`. Not football-realistic."""

from __future__ import annotations

import numpy as np
import schema as S

PH, A, P = S.PHASE, S.ACTION, S.POSS_CODE


def _policy(phase: int, down: int, togo: int, yl: int, rng: np.random.Generator) -> int:
    if phase == PH["KICKOFF"]:
        return A["KICKOFF"]
    if phase == PH["TRY"]:
        return A["XP"] if rng.random() < 0.9 else A["TWO_PT"]
    if down == 4:
        if rng.random() < 0.2 or (togo <= 2 and yl < 60):
            return A["RUN"] if rng.random() < 0.5 else A["PASS"]
        return A["FG"] if yl <= 37 else A["PUNT"]
    return A["RUN"] if rng.random() < 0.45 else A["PASS"]


def _step(phase, poss, down, togo, yl, action, rng):
    """Next (phase, poss, down, togo, yardline), points for the team with the ball, elapsed."""
    other = 3 - poss
    if phase == PH["KICKOFF"]:
        return (PH["PLAY"], other, 1, 10, 75), 0, int(rng.integers(4, 9))
    if phase == PH["TRY"]:
        pts = (
            (1 if rng.random() < 0.94 else 0)
            if action == A["XP"]
            else (2 if rng.random() < 0.48 else 0)
        )
        return (PH["KICKOFF"], poss, 0, 0, 65), pts, 0
    el = int(rng.integers(5, 41))
    if action == A["FG"]:
        if rng.random() < max(0.2, 1.0 - yl / 50):
            return (PH["KICKOFF"], poss, 0, 0, 65), 3, 5
        return (PH["PLAY"], other, 1, 10, int(np.clip(100 - (yl + 7), 1, 99))), 0, 5
    if action == A["PUNT"]:
        land = yl - 40
        yl2 = 80 if land <= 0 else int(np.clip(100 - land, 1, 99))
        return (PH["PLAY"], other, 1, 10, yl2), 0, 8
    gain = int(rng.integers(-3, 16))
    if gain >= yl:
        return (PH["TRY"], poss, 0, 0, 15), 6, el
    yl2 = int(np.clip(yl - gain, 1, 99))
    if gain >= togo:
        return (PH["PLAY"], poss, 1, min(10, yl2), yl2), 0, el
    if down == 4:
        return (PH["PLAY"], other, 1, 10, 100 - yl2), 0, el
    return (PH["PLAY"], poss, down + 1, int(np.clip(togo - gain, 1, S.MAX_YDSTOGO)), yl2), 0, el


def make_toy_split(
    n_games: int, rng: np.random.Generator, start_clock: int = 600, first_game_id: int = 0
) -> S.Split:
    """n_games short games (regulation clock starts at start_clock seconds)."""
    states, raws, targets, offsets, header, final = [], [], [], [0], [], []
    for _ in range(n_games):
        receiver = int(rng.integers(1, 3))
        header.append([S.spread_code(rng.integers(-7, 8)), receiver])
        cat = (PH["KICKOFF"], 3 - receiver, 0, 0, 65)
        raw = np.array([start_clock, 0, 3, 3], dtype=np.int32)
        action = A["KICKOFF"]
        while True:
            phase, poss, down, togo, yl = cat
            nxt, pts, el = _step(phase, poss, down, togo, yl, action, rng)
            sign = 1 if poss == P["HOME"] else -1
            used_h = int(rng.random() < 0.03 and raw[2] > 0)
            used_a = int(rng.random() < 0.03 and raw[3] > 0)
            n_action = _policy(nxt[0], nxt[2], nxt[3], nxt[4], rng)
            tg = np.zeros(len(S.TARGET_FIELDS), dtype=np.int16)
            for name, v in zip(("phase", "poss", "down", "ydstogo", "yardline"), nxt, strict=True):
                tg[S.TARGET_INDEX[name]] = v
            tg[S.TARGET_INDEX["score_delta"]] = S.SCORE_DELTA_CODE[sign * pts]
            tg[S.TARGET_INDEX["elapsed"]] = S.elapsed_code(el)
            tg[S.TARGET_INDEX["to_used_home"]] = used_h + 1
            tg[S.TARGET_INDEX["to_used_away"]] = used_a + 1
            tg[S.TARGET_INDEX["action"]] = n_action
            raw_next = S.apply_transition(raw[None], tg[None])[0]
            end = raw_next[0] == 0 and nxt[0] != PH["TRY"]
            rec = np.zeros(len(S.STATE_FIELDS), dtype=np.int16)
            for name, v in zip(("phase", "poss", "down", "ydstogo", "yardline"), cat, strict=True):
                rec[S.STATE_INDEX[name]] = v
            rec[S.STATE_INDEX["clock"]] = S.clock_code(raw[0])
            rec[S.STATE_INDEX["score"]] = S.score_code(raw[1])
            rec[S.STATE_INDEX["to_home"]] = S.timeout_code(raw[2])
            rec[S.STATE_INDEX["to_away"]] = S.timeout_code(raw[3])
            rec[S.STATE_INDEX["action"]] = action
            states.append(rec)
            raws.append(raw.copy())
            if end:
                tg = np.zeros_like(tg)
                tg[S.TARGET_INDEX["phase"]] = PH["END"]
                targets.append(tg)
                final.append(int(raw_next[1]))  # includes the final play's points
                break
            targets.append(tg)
            cat, raw, action = nxt, raw_next, n_action
        offsets.append(len(states))
    n = len(states)
    return S.Split(
        states=np.asarray(states, dtype=np.int16),
        raw=np.asarray(raws, dtype=np.int32),
        targets=np.asarray(targets, dtype=np.int16),
        offsets=np.asarray(offsets, dtype=np.int64),
        header=np.asarray(header, dtype=np.int16),
        final_diff=np.asarray(final, dtype=np.int16),
        game_ids=[f"toy_{first_game_id + g}" for g in range(n_games)],
        seasons=np.full(n_games, 2000, dtype=np.int16),
        play_ids=np.arange(n, dtype=np.int64),
    )
