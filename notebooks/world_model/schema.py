"""The data contract for the NFL world model. Every other module follows this file.

A game is a sequence of RECORDS, one per snap or kick in regulation (overtime is cut).
Record t holds the pre-snap STATE and the ACTION taken. The model reads records 0..t and
predicts the TRANSITION to record t+1: a few "delta" fields (points scored, seconds
elapsed, timeouts used) plus the next record's categorical fields and its action.
Arithmetic bookkeeping (score, clock, timeouts) is done by `apply_transition`, never by
the model, so those three can never become inconsistent.

All values are small non-negative integers ("codes"). Each field has its own vocabulary
of size `size`; code c of field f is embedded with that field's own embedding table.
Perspective: possession and score are stored from the HOME team's side; yard line is from
the side of the team with the ball (yardline_100, 1 = opponent's 1, 99 = own 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- categorical values
PHASES = ("PAD", "KICKOFF", "PLAY", "TRY", "END")  # TRY = extra point or two-point try
PHASE = {name: i for i, name in enumerate(PHASES)}

POSS = ("PAD", "HOME", "AWAY")
POSS_CODE = {name: i for i, name in enumerate(POSS)}

ACTIONS = (
    "PAD",
    "RUN",
    "PASS",
    "PUNT",
    "FG",
    "KNEEL",
    "SPIKE",
    "KICKOFF",
    "ONSIDE",
    "XP",
    "TWO_PT",
    "NOPLAY",  # play wiped out by a penalty whose run/pass type is unknown
)
ACTION = {name: i for i, name in enumerate(ACTIONS)}
GO_ACTIONS = (ACTION["RUN"], ACTION["PASS"])  # "go for it" on fourth down = run or pass

# Points change on one transition, HOME minus AWAY. Code 0 is padding.
SCORE_DELTAS = (0, -8, -7, -6, -3, -2, -1, 0, 1, 2, 3, 6, 7, 8)  # index 0 unused (pad)
SCORE_DELTA_CODE = {v: i for i, v in enumerate(SCORE_DELTAS) if i > 0}

# ---------------------------------------------------------------- numeric ranges
MAX_YDSTOGO = 30  # ydstogo is clipped to 1..30; code = ydstogo (0 = not a down)
MAX_ELAPSED = 60  # seconds between records, clipped to 0..60; code = seconds + 1 (0 = pad)
CLOCK_BIN = 5  # clock embedding uses 5-second bins of game seconds remaining
GAME_SECONDS = 3600
HALF_SECONDS = 1800
SCORE_CLIP = 35  # score differential clipped to -35..35; code = diff + 36 (0 = pad)
N_TIMEOUTS = 3
SPREAD_CLIP = 21  # pregame home spread (home favoured by x) clipped, rounded to integers

# ---------------------------------------------------------------- field tables
STATE_FIELDS = (
    # name, vocabulary size
    ("phase", len(PHASES)),
    ("poss", len(POSS)),
    ("down", 5),  # 0 = not a down (kickoff or try), 1..4
    ("ydstogo", MAX_YDSTOGO + 1),  # 0 = not a down
    ("yardline", 100),  # 1..99; 0 = pad
    ("clock", GAME_SECONDS // CLOCK_BIN + 2),  # bin of seconds remaining, +1 offset, 0 = pad
    ("score", 2 * SCORE_CLIP + 2),  # home minus away + 36
    ("to_home", N_TIMEOUTS + 2),  # timeouts + 1, 0 = pad
    ("to_away", N_TIMEOUTS + 2),
    ("action", len(ACTIONS)),
)
STATE_INDEX = {name: i for i, (name, _) in enumerate(STATE_FIELDS)}

# The transition from record t to record t+1, in the order the decoder samples it.
TARGET_FIELDS = (
    ("phase", len(PHASES)),  # END means the game (regulation) is over
    ("poss", len(POSS)),
    ("score_delta", len(SCORE_DELTAS)),
    ("elapsed", MAX_ELAPSED + 2),
    ("to_used_home", 4),  # 0, 1, 2 used (code = used + 1; 0 = pad)
    ("to_used_away", 4),
    ("down", 5),
    ("ydstogo", MAX_YDSTOGO + 1),
    ("yardline", 100),
    ("action", len(ACTIONS)),
)
TARGET_INDEX = {name: i for i, (name, _) in enumerate(TARGET_FIELDS)}

HEADER_FIELDS = (
    ("spread", 2 * SPREAD_CLIP + 2),  # home spread + 22 (0 = pad)
    ("opening_receiver", len(POSS)),  # HOME or AWAY received the opening kickoff
)

# Raw (un-binned) per-record columns kept next to the codes, used for bookkeeping.
RAW_COLUMNS = ("clock_seconds", "score_diff", "timeouts_home", "timeouts_away")


@dataclass
class Split:
    """One split (train, validation or test) in array form.

    states   int16 [N, len(STATE_FIELDS)]   record codes
    raw      int32 [N, len(RAW_COLUMNS)]    unbinned clock seconds, score diff, timeouts
    targets  int16 [N, len(TARGET_FIELDS)]  transition from record i to record i+1; for the
                                            last record of a game phase = END and every
                                            other target field = 0 (masked in the loss)
    offsets  int64 [G + 1]                  records of game g are offsets[g]:offsets[g+1]
    header   int16 [G, len(HEADER_FIELDS)]
    final_diff int16 [G]                    home minus away at the end of regulation
    game_ids list[str]                      nflverse game_id per game
    seasons  int16 [G]
    play_ids int64 [N]                      nflverse play_id per record (for audits)
    """

    states: np.ndarray
    raw: np.ndarray
    targets: np.ndarray
    offsets: np.ndarray
    header: np.ndarray
    final_diff: np.ndarray
    game_ids: list[str]
    seasons: np.ndarray
    play_ids: np.ndarray


# ---------------------------------------------------------------- encoders
def clock_code(seconds: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(seconds), 0, GAME_SECONDS)
    return (s // CLOCK_BIN + 1).astype(np.int16)


def score_code(diff: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(diff), -SCORE_CLIP, SCORE_CLIP) + SCORE_CLIP + 1).astype(np.int16)


def timeout_code(n: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(n), 0, N_TIMEOUTS) + 1).astype(np.int16)


def elapsed_code(seconds: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(seconds), 0, MAX_ELAPSED) + 1).astype(np.int16)


def spread_code(spread: np.ndarray) -> np.ndarray:
    s = np.rint(np.clip(np.asarray(spread, dtype=float), -SPREAD_CLIP, SPREAD_CLIP))
    return (s + SPREAD_CLIP + 1).astype(np.int16)


def score_delta_value(code: np.ndarray) -> np.ndarray:
    """Points (home minus away) for a score-delta code; pad code 0 maps to 0."""
    table = np.asarray(SCORE_DELTAS, dtype=np.int32)
    return table[np.asarray(code)]


def apply_transition(raw: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Bookkeeping: the next record's raw columns from the current raw and a transition.

    raw [B, 4] (clock_seconds, score_diff, timeouts_home, timeouts_away); target [B, T].
    Clock falls by the elapsed seconds (never below 0, and never across halftime: a
    transition that crosses 1800 seconds stops at 1800, where both teams get 3 timeouts
    back). Score adds the delta. Timeouts subtract the used counts, floored at 0.
    """
    raw = np.asarray(raw, dtype=np.int32)
    t = np.asarray(target)
    elapsed = np.clip(t[:, TARGET_INDEX["elapsed"]].astype(np.int32) - 1, 0, MAX_ELAPSED)
    clock = raw[:, 0]
    new_clock = np.maximum(clock - elapsed, 0)
    crossed = (clock > HALF_SECONDS) & (new_clock <= HALF_SECONDS)
    new_clock = np.where(crossed, HALF_SECONDS, new_clock)
    score = raw[:, 1] + score_delta_value(t[:, TARGET_INDEX["score_delta"]])
    used_h = np.clip(t[:, TARGET_INDEX["to_used_home"]].astype(np.int32) - 1, 0, 2)
    used_a = np.clip(t[:, TARGET_INDEX["to_used_away"]].astype(np.int32) - 1, 0, 2)
    to_h = np.where(crossed, N_TIMEOUTS, np.maximum(raw[:, 2] - used_h, 0))
    to_a = np.where(crossed, N_TIMEOUTS, np.maximum(raw[:, 3] - used_a, 0))
    return np.stack([new_clock, score, to_h, to_a], axis=1).astype(np.int32)


def next_state_codes(raw_next: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Record codes for record t+1 from its raw columns and the sampled transition."""
    t = np.asarray(target)
    out = np.zeros((len(t), len(STATE_FIELDS)), dtype=np.int16)
    for name in ("phase", "poss", "down", "ydstogo", "yardline", "action"):
        out[:, STATE_INDEX[name]] = t[:, TARGET_INDEX[name]]
    out[:, STATE_INDEX["clock"]] = clock_code(raw_next[:, 0])
    out[:, STATE_INDEX["score"]] = score_code(raw_next[:, 1])
    out[:, STATE_INDEX["to_home"]] = timeout_code(raw_next[:, 2])
    out[:, STATE_INDEX["to_away"]] = timeout_code(raw_next[:, 3])
    return out


def outcome(final_diff: np.ndarray) -> np.ndarray:
    """Home result at the end of regulation: 1 win, 0.5 tie, 0 loss."""
    d = np.asarray(final_diff)
    return np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))
