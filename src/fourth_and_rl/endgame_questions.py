"""M13 helpers: goal-line and two-point decisions read from the solved game (M4).

Three questions argued on X with score arithmetic are answered here from the exact game
model instead (`scripts/questions/q5_goal_line_and_two_point.py`):

1. Leading late, fourth and goal from the 1, 2 or 3: field goal or go?
2. After a touchdown: kick the extra point or go for two, by the margin before the try?
3. Early in a close game, fourth and goal: take the points or go?

Everything here works on any `mdp_game.Solution`, so the unit tests use toy games small
enough to check by hand. The helpers use the public API of `mdp_game` (`q_slice`,
`post`, `opp`, `play_split`, the model and kernel fields). Two quantities have no public
accessor and are rebuilt here from public pieces: the separate values of the kick and of
the two-point try (`Solution._derive` keeps only the better one, in `_Post.td`, and the
flag `_Post.two`), and the win probability after a missed field goal (inside the private
`Solution._aug`). `try_values` and `field_goal_split` say how; tests check both against
the solver's own arrays.

Conventions
-----------
* Time is in seconds of game time left; it is rounded to the model's ticks.
* `extras` are (offense timeouts, defense timeouts, receives second-half kickoff)
  triples from the side of the team with the ball (or, for a try, the scoring team).
  Timeouts matter only in the last 5:00 and the flag only in the first half.
* The margin of a try is the score difference just after the touchdown, before the try,
  from the scoring team's side (as nflfastR's `score_differential` on a try play).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from fourth_and_rl import mdp_game as mg

GO_CODES = (mg.A_NEUTRAL, mg.A_PROTECT, mg.A_CHASE)
Extra = tuple[int, int, int]


def tick_of(sol: mg.Solution, seconds: int) -> int:
    """Time in ticks for a time in seconds (as `Solution.index` rounds it)."""
    return int(round(seconds / sol.model.clock.tick_seconds))


def extra_index(sol: mg.Solution, t: int, extra: Extra) -> tuple[int, int]:
    """Index into the two extra axes of the phase that holds time t."""
    m = sol.model
    ph = m.phase(t)
    e = m.extra_shape(ph)
    a, b, r = extra
    if ph == "end":
        return min(a, e[0] - 1), min(b, e[1] - 1)
    if ph == "first":
        return int(r), 0
    return 0, 0


def score_index(sol: mg.Solution, score: int | np.ndarray) -> np.ndarray:
    m = sol.model
    return np.clip(np.asarray(score), -m.score_max, m.score_max) + m.score_max


# --------------------------------------------------------------------------------------
# Fourth and goal: go or field goal
# --------------------------------------------------------------------------------------


def goal_line_values(
    sol: mg.Solution,
    seconds: list[int],
    scores: list[int],
    yardlines: list[int],
    extras: list[Extra],
) -> dict[str, np.ndarray]:
    """Win probability of going for it and of kicking on fourth and goal.

    Arrays of shape (len(seconds), len(scores), len(extras), len(yardlines)): "go" (the
    best of the allowed play modes), "field_goal", "punt" (-inf where not allowed),
    "go_mode" (action code of the best mode). Fourth and goal from the y means yards to
    go = yard line = y.
    """
    g = sol.grid
    ys = np.asarray(yardlines)
    f = np.asarray(g.fidx(4 if g.n_down >= 4 else g.n_down, g.dist_bin(ys), ys))
    si = score_index(sol, scores)
    shape = (len(seconds), len(scores), len(extras), len(ys))
    out = {k: np.full(shape, -np.inf) for k in ("go", "field_goal", "punt")}
    mode = np.zeros(shape, dtype=np.int64)
    for i, sec in enumerate(seconds):
        t = tick_of(sol, sec)
        q = sol.q_slice(t)
        for k, ex in enumerate(extras):
            e1, e2 = extra_index(sol, t, ex)
            plays = np.stack(
                [
                    q[c][f[None, :], si[:, None], e1, e2]
                    if c in q
                    else np.full((len(si), len(f)), -np.inf)
                    for c in GO_CODES
                ]
            )  # (3, scores, yardlines)
            out["go"][i, :, k] = plays.max(axis=0)
            mode[i, :, k] = np.asarray(GO_CODES)[plays.argmax(axis=0)]
            for name, code in (("field_goal", mg.A_FG), ("punt", mg.A_PUNT)):
                if code in q:
                    out[name][i, :, k] = q[code][f[None, :], si[:, None], e1, e2]
    out["go_mode"] = mode
    return out


def field_goal_split(sol: mg.Solution, state: mg.GameState) -> dict[str, float]:
    """Take apart the value of a field-goal try: chance it is good, win probability if it
    is and if it is not. make * if_made + (1 - make) * if_missed = value.

    The value after a make is `post(t - 1).fg` (public); the value after a miss sits in
    the private `Solution._aug`, so it is recovered from the action value instead:
    if_missed = (value - make * if_made) / (1 - make)."""
    m = sol.model
    ph, ix = sol.index(state)
    t = ix[0] + m.phase_range(ph)[0]
    f = ix[1]
    make = float(m.kern.fg_make[f])
    value = float(sol.q_slice(t)[mg.A_FG][ix[1:]])
    made = float(sol.post(max(t - 1, 0), ph).fg[ix[2:]])
    missed = (value - make * made) / (1.0 - make) if make < 1.0 else float("nan")
    return {"p_make": make, "value_if_made": made, "value_if_missed": missed, "value": value}


def goal_line_case(sol: mg.Solution, state: mg.GameState) -> dict[str, Any]:
    """One fourth-and-goal decision taken apart in plain numbers.

    Go: chance of a touchdown on the play, win probability if it scores (after the best
    try) and if it fails. Field goal: chance it is good, win probability if good and if
    missed. Break-even: the touchdown chance at which going equals kicking."""
    qv = sol.q_values(state)
    go = max(v for k, v in qv.items() if k.startswith("play"))
    split = sol.play_split(state)
    fg = field_goal_split(sol, state)
    p, hi, lo = split["p_keep_or_score"], split["value_if_kept"], split["value_if_lost"]
    return {
        "go": go,
        "field_goal": fg["value"],
        "go_minus_field_goal": go - fg["value"],
        "best": "go" if go > fg["value"] else "field_goal",
        "go_mode": split["mode"],
        "p_touchdown": split["p_touchdown_on_the_play"],
        "p_keep_or_score": p,
        "win_probability_if_touchdown": hi,
        "win_probability_if_stopped": lo,
        "p_field_goal_good": fg["p_make"],
        "win_probability_if_field_goal_good": fg["value_if_made"],
        "win_probability_if_field_goal_missed": fg["value_if_missed"],
        "break_even_p_touchdown": (fg["value"] - lo) / (hi - lo) if hi > lo else float("nan"),
        "all_action_values": qv,
    }


# --------------------------------------------------------------------------------------
# The try after a touchdown
# --------------------------------------------------------------------------------------


def kicking_off_values(sol: mg.Solution, t: int) -> np.ndarray:
    """W[s, e1, e2]: win probability of a team about to kick off at time t with score
    difference s (index into the clipped scores), from its own side. The kickoff takes no
    time, so this is the value right after a try, a field goal or a touchdown's try."""
    ph = sol.model.phase(t)
    return sol.opp(sol.post(t, ph).ko, ph)


def try_values(
    sol: mg.Solution,
    seconds: list[int],
    margins: list[int],
    extras: list[Extra],
    p_xp: float | None = None,
    p_two: float | None = None,
) -> dict[str, np.ndarray]:
    """Kick versus two-point try after a touchdown, as the solved game values them.

    `margins` are the scoring team's margins before the try. Returns arrays of shape
    (len(seconds), len(margins), len(extras)): "kick", "two" (win probability of each
    choice), "two_better" (the model's own flag `_Post.two`, for checking), and
    "after" of shape (..., 3): the win probability when kicking off at margin + 0, + 1
    and + 2. kick = p_xp * after[1] + (1 - p_xp) * after[0], and two likewise with
    p_two and after[2], exactly as `Solution._derive` computes them before keeping the
    better one. `p_xp`, `p_two` default to the model's probabilities; other values give
    the one-shot change (this try only, later tries unchanged).
    """
    m = sol.model
    pxp = m.p_xp if p_xp is None else p_xp
    p2 = m.p_two if p_two is None else p_two
    shape = (len(seconds), len(margins), len(extras))
    after = np.zeros((*shape, 3))
    flag = np.zeros(shape, dtype=bool)
    mi = np.asarray(margins)
    for i, sec in enumerate(seconds):
        t = tick_of(sol, sec)
        w = kicking_off_values(sol, t)
        two_flag = sol.post(t, m.phase(t)).two  # indexed by the score before the touchdown
        for k, ex in enumerate(extras):
            e1, e2 = extra_index(sol, t, ex)
            for j, d in enumerate((0, 1, 2)):
                after[i, :, k, j] = w[score_index(sol, mi + d), e1, e2]
            flag[i, :, k] = two_flag[score_index(sol, mi - m.td_points), e1, e2]
    kick = pxp * after[..., 1] + (1.0 - pxp) * after[..., 0]
    two = p2 * after[..., 2] + (1.0 - p2) * after[..., 0]
    return {"kick": kick, "two": two, "two_better": flag, "after": after}


def try_case(sol: mg.Solution, seconds: int, margin: int, extra: Extra) -> dict[str, Any]:
    """One try decision in plain numbers: chance of each try, win probability after a
    success and after a failure, and the break-even two-point success rate."""
    m = sol.model
    tv = try_values(sol, [seconds], [margin], [extra])
    after = tv["after"][0, 0, 0]
    kick, two = float(tv["kick"][0, 0, 0]), float(tv["two"][0, 0, 0])
    spread = after[2] - after[0]
    return {
        "margin_before_try": margin,
        "kick": kick,
        "two": two,
        "two_minus_kick": two - kick,
        "best": "two" if two > kick else "kick",
        "p_extra_point": m.p_xp,
        "p_two_point": m.p_two,
        "win_probability_kicking_off_at_margin": {
            str(margin + d): float(after[d]) for d in range(3)
        },
        "break_even_p_two_point": float((kick - after[0]) / spread) if spread > 0 else None,
    }


def two_try_arithmetic(p_two: float, p_xp: float, tie_value: float = 0.5) -> dict[str, float]:
    """The score-arithmetic argument for "down 8 after a touchdown", on its own path.

    Assume the trailing team scores exactly one more touchdown, the other team scores no
    more, and a tie is worth `tie_value` (the model has no overtime; a tie after
    regulation is 0.5). Going for two now: success (p_two) leaves it down 6, and the next
    touchdown plus the extra point wins; a failure leaves it down 8, and the next
    touchdown needs two to tie. Kicking now (down 7): the next touchdown ties with a kick
    or wins with two, whichever is better. Win probability on that path for each plan.
    """
    # Down 6 then a touchdown: tied before the try; kick to win (a miss ties) or go for two.
    lead_try = max(p_xp + (1.0 - p_xp) * tie_value, p_two + (1.0 - p_two) * tie_value)
    two_now = p_two * lead_try + (1.0 - p_two) * p_two * tie_value
    second = max(p_xp * tie_value, p_two)  # down 1 before the try: kick to tie or two to win
    kick_now = p_xp * second + (1.0 - p_xp) * p_two * tie_value  # a miss leaves down 8
    return {"go_for_two_now": two_now, "kick_now": kick_now, "difference": two_now - kick_now}


# --------------------------------------------------------------------------------------
# Re-solving with one assumption changed
# --------------------------------------------------------------------------------------


def truncated(model: mg.GameModel, seconds: int) -> mg.GameModel:
    """A copy of a game model that covers only the last `seconds` of a second half.

    Backward induction makes the value at time t depend only on smaller times, so the
    last minutes solved alone give exactly the values of the full game there. Allowed
    only inside the second half (the first half needs the halftime kickoff)."""
    c = model.clock
    n = int(seconds // c.tick_seconds)
    if n > c.th:
        raise ValueError(f"{seconds} s reaches into the first half; solve the whole game")
    m = copy.copy(model)
    m.clock = copy.copy(c)
    m.clock.n_ticks, m.clock.half_tick = n, None
    return m


def with_two_point(model: mg.GameModel, p_two: float) -> mg.GameModel:
    """A copy of a game model with another two-point success probability."""
    m = copy.copy(model)
    m.p_two = float(np.clip(p_two, 0.0, 1.0))
    return m
