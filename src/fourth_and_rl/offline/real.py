"""Logged fourth-down transitions from real play-by-play, and off-policy evaluation.

The builder turns real fourth downs into transitions of the drive model of Part A, so the
same learners run on them unchanged:

    state    (distance bin, yard line) of the fourth down, as in `mdp.Grid`
    action   go, field goal or punt (`fourth_down_decision` of the state table: runs and
             passes on fourth down are "go", including fakes; kneels, spikes and plays
             wiped out by a penalty are not decisions, as everywhere in the repo)
    outcome  the first of these that happens after the snap, scanning the game's rows in
             order (the decision play included):
               - the offense scores a touchdown (`touchdown` with `td_team` = offense)
               - the other team scores a touchdown (a return)
               - a safety against the offense
               - the offense makes a field goal
               - a snap by the other team: it took over at that snap's `yardline_100`
                 (its own distance from our end zone, the "their yard line y" of the grid)
               - the offense's next fourth-down decision in the same drive: the next state
               - the half ends first: the transition is cut ("truncated") and dropped, since
                 the drive model has no clock and a cut drive has no outcome to learn from
    Penalties: a no-play is not a decision, but it is a snap, so a penalty that hands the
    offense a new set of downs simply lets the drive continue to its next decision.
    Accepted penalties on a decision play are part of what that play led to.
    A "snap" is a row with a down, a yard line and a run, pass, punt, field goal, no-play,
    kneel or spike play type (kickoffs and timeouts are not snaps).

Which decisions: regulation only (quarters 1 to 4), yard line 1 to 99, and by default
only the "neutral" situation of the coach table (`mdp.coach_situation` = 0: not the last
two minutes of the first half, not the fourth quarter when trailing, not the last five
minutes), because the drive model values a possession in expected points, which is not
what a coach maximizes late in a close game. A neutral decision's next decision may be in
any situation; its state is used, but no transition is logged from it, so the episode is
marked as cut there.

The reward of each ending is the drive model's `term_reward` (points minus the solved
value of what the other team then holds), so real-data learners target the same kind of
value as in Part A, with the continuation values taken from the solved model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from fourth_and_rl import splits
from fourth_and_rl.mdp import Grid, coach_situation
from fourth_and_rl.offline.datasets import Logged
from fourth_and_rl.rl.envs import DriveModel

ACTIONS = ("go", "field_goal", "punt")
KINDS = ("next", "td", "opp_td", "safety", "fg", "opp_ball", "truncated")
SNAP_TYPES = ("run", "pass", "punt", "field_goal", "no_play", "qb_kneel", "qb_spike")
NEEDED = (
    "game_id", "season", "play_id", "order_sequence", "posteam", "game_half", "qtr", "down",
    "ydstogo", "yardline_100", "play_type", "fourth_down_decision", "touchdown", "td_team",
    "safety", "field_goal_result", "game_seconds_remaining", "half_seconds_remaining",
    "score_differential",
)  # fmt: skip


def decision_outcomes(plays: pl.DataFrame, grid: Grid, neutral_only: bool = True
                      ) -> pl.DataFrame:
    """One row per fourth-down decision with what it led to (see the module docstring).

    Columns: game_id, season, play_id, posteam, situation, dbin, yl, action (0, 1, 2),
    kind (one of KINDS), next_play_id (for kind "next"), next_dbin, next_yl, opp_yl (for
    "opp_ball"), next_in_sample (the next decision is itself one of the rows returned).
    """
    missing = set(NEEDED) - set(plays.columns)
    if missing:
        raise ValueError(f"plays lack columns {sorted(missing)}")
    df = plays.select(NEEDED).sort("game_id", "order_sequence", "play_id")
    game = df["game_id"].to_numpy()
    half = df["game_half"].to_numpy()
    pos = df["posteam"].to_numpy()
    yl = df["yardline_100"].to_numpy()
    snap = (
        df["down"].is_not_null() & df["yardline_100"].is_not_null()
        & df["play_type"].is_in(SNAP_TYPES)
    ).to_numpy()
    td = (df["touchdown"].fill_null(0) == 1).to_numpy()
    td_team = df["td_team"].to_numpy()
    safety = (df["safety"].fill_null(0) == 1).to_numpy()
    fg_made = ((df["play_type"] == "field_goal") & (df["field_goal_result"] == "made")).fill_null(
        False
    ).to_numpy()
    dec_str = df["fourth_down_decision"].to_numpy()
    is_dec = (
        df["fourth_down_decision"].is_not_null() & (df["qtr"] <= 4)
        & df["yardline_100"].is_between(1, 99) & (df["ydstogo"] >= 1)
    ).fill_null(False).to_numpy()
    sit = coach_situation(
        df["game_seconds_remaining"].fill_null(0).to_numpy(),
        df["half_seconds_remaining"].fill_null(0).to_numpy(),
        df["score_differential"].fill_null(0).to_numpy(),
    )
    keep = is_dec & (sit == 0) if neutral_only else is_dec
    n_rows = df.height
    rows: dict[str, list] = {k: [] for k in ("i", "kind", "nxt", "opp_yl")}
    for i in np.nonzero(keep)[0]:
        p, h, g = pos[i], half[i], game[i]
        kind, nxt, opp = "truncated", -1, -1
        j = i
        while j < n_rows and game[j] == g and half[j] == h:
            if j > i and snap[j] and pos[j] != p:
                kind, opp = "opp_ball", int(yl[j])
                break
            if j > i and is_dec[j] and pos[j] == p:
                kind, nxt = "next", j
                break
            if td[j]:
                kind = "td" if td_team[j] == p else "opp_td"
                break
            if safety[j]:
                kind = "safety" if pos[j] == p else "truncated"
                break
            if fg_made[j] and pos[j] == p:
                kind = "fg"
                break
            j += 1
        rows["i"].append(i)
        rows["kind"].append(kind)
        rows["nxt"].append(nxt)
        rows["opp_yl"].append(opp)
    idx = np.asarray(rows["i"], dtype=np.int64)
    nxt = np.asarray(rows["nxt"], dtype=np.int64)
    has = nxt >= 0
    play_id = df["play_id"].to_numpy()
    ytg = df["ydstogo"].to_numpy()
    dbin = grid.dist_bin(ytg[idx].astype(int))
    nd = np.where(has, grid.dist_bin(np.where(has, ytg[np.maximum(nxt, 0)], 1).astype(int)), -1)
    nyl = np.where(has, yl[np.maximum(nxt, 0)], -1)
    in_sample = has & keep[np.maximum(nxt, 0)]
    return pl.DataFrame({
        "game_id": game[idx],
        "season": df["season"].to_numpy()[idx],
        "play_id": play_id[idx],
        "posteam": pos[idx],
        "situation": sit[idx],
        "dbin": dbin.astype(np.int64),
        "yl": yl[idx].astype(np.int64),
        "action": np.array([ACTIONS.index(a) for a in dec_str[idx]], dtype=np.int64),
        "kind": rows["kind"],
        "next_play_id": np.where(has, play_id[np.maximum(nxt, 0)], -1),
        "next_dbin": nd.astype(np.int64),
        "next_yl": nyl.astype(np.int64),
        "opp_yl": np.asarray(rows["opp_yl"], dtype=np.int64),
        "next_in_sample": in_sample,
    })


def to_logged(outcomes: pl.DataFrame, model: DriveModel) -> tuple[Logged, pl.DataFrame,
                                                                      dict[str, int]]:
    """Map decision outcomes onto the drive model's columns and order them in episodes
    (chains of decisions of one drive). Truncated rows and actions the model does not
    allow in that state (a field goal from beyond the model's range) are dropped, which
    cuts the episode there. Returns (data, the kept rows in data order with an `episode`
    column, counts of what was dropped)."""
    g = model.grid
    n, n_yl = model.n, g.n_yl
    pos = np.full(g.nf, -1)
    pos[model.states] = np.arange(n)
    df = outcomes
    s = pos[g.fidx(g.n_down, df["dbin"].to_numpy(), df["yl"].to_numpy())]
    a = df["action"].to_numpy()
    kind = df["kind"].to_numpy()
    nd, nyl = df["next_dbin"].to_numpy(), df["next_yl"].to_numpy()
    nxt_state = np.where(kind == "next",
                         pos[g.fidx(g.n_down, np.maximum(nd, 0), np.clip(nyl, 1, n_yl))], -1)
    col = np.full(df.height, -1, dtype=np.int64)
    col = np.where(kind == "next", nxt_state, col)
    col = np.where(kind == "td", n + n_yl, col)
    col = np.where(kind == "safety", n + n_yl + 1, col)
    col = np.where(kind == "opp_td", n + n_yl + 2, col)
    col = np.where(kind == "fg", n + n_yl + 3, col)
    opp = df["opp_yl"].to_numpy()
    col = np.where((kind == "opp_ball") & (opp >= 1) & (opp <= n_yl), n + opp - 1, col)
    allowed = model.allowed[s, a]
    ok = (col >= 0) & allowed
    dropped = {
        "truncated_end_of_half": int((kind == "truncated").sum()),
        "action_not_allowed_in_model": int(((col >= 0) & ~allowed).sum()),
        "unmapped_outcome": int(((col < 0) & (kind != "truncated")).sum()),
    }
    kept = df.with_columns(
        pl.Series("s", s), pl.Series("c", col), pl.Series("_ok", ok)
    ).filter(pl.col("_ok"))
    # Chains: a decision starts an episode unless it is the kept next decision of a kept one.
    pid = kept["play_id"].to_numpy()
    gid = kept["game_id"].to_numpy()
    key = {(gg, pp): k for k, (gg, pp) in enumerate(zip(gid, pid, strict=True))}
    nxt_pid = kept["next_play_id"].to_numpy()
    kk = kept["kind"].to_numpy()
    succ = np.full(kept.height, -1)
    for k in range(kept.height):
        if kk[k] == "next":
            succ[k] = key.get((gid[k], nxt_pid[k]), -1)
    has_pred = np.zeros(kept.height, dtype=bool)
    has_pred[succ[succ >= 0]] = True
    order, ep = [], []
    e = 0
    for k in np.nonzero(~has_pred)[0]:
        cur = k
        while cur >= 0:
            order.append(cur)
            ep.append(e)
            cur = succ[cur]
        e += 1
    order_arr = np.asarray(order, dtype=np.int64)
    ep_arr = np.asarray(ep, dtype=np.int64)
    starts = np.ones(len(order_arr), dtype=bool)
    starts[1:] = ep_arr[1:] != ep_arr[:-1]
    kept = kept[order_arr].with_columns(pl.Series("episode", ep_arr)).drop("_ok")
    data = Logged(
        kept["s"].to_numpy().astype(np.int32),
        kept["action"].to_numpy().astype(np.int8),
        kept["c"].to_numpy().astype(np.int32),
        n,
        starts,
    )
    return data, kept, dropped


def build_logged(plays: pl.DataFrame, model: DriveModel, what: str, neutral_only: bool = True,
                 guard: bool = True) -> tuple[Logged, pl.DataFrame, dict[str, Any]]:
    """`decision_outcomes` then `to_logged`, with the leakage guard on the input frame
    (`splits.assert_no_test_seasons`) unless `guard` is False (test-season evaluation
    data, never fitted on)."""
    if guard:
        splits.assert_no_test_seasons(plays, what)
    out = decision_outcomes(plays, model.grid, neutral_only)
    data, kept, dropped = to_logged(out, model)
    kinds = out["kind"].value_counts().sort("kind")
    info = {
        "n_decisions": int(out.height),
        "n_transitions": int(data.size),
        "n_episodes": int(data.first.sum()),
        "dropped": dropped,
        "outcome_kinds": dict(zip(kinds["kind"].to_list(), kinds["count"].to_list(),
                                  strict=True)),
        "actions": {ACTIONS[k]: int((data.a == k).sum()) for k in range(3)},
        "seasons": sorted(int(x) for x in out["season"].unique().to_list()),
        "n_games": int(out["game_id"].n_unique()),
    }
    return data, kept, info


# --------------------------------------------------------------------------------------
# Off-policy evaluation
# --------------------------------------------------------------------------------------


def episodes(data: Logged, model: DriveModel) -> list[dict[str, np.ndarray]]:
    """The complete episodes (ending in a terminal outcome) as arrays s, a, c, with the
    terminal reward. Cut episodes are left out (they have no observed return)."""
    out = []
    first = np.nonzero(data.first)[0]
    bounds = np.append(first, data.size)
    for b, e in zip(bounds[:-1], bounds[1:], strict=True):
        c = data.c[b:e]
        if c[-1] < model.n:
            continue
        out.append({"s": data.s[b:e], "a": data.a[b:e], "c": c,
                    "r": float(model.term_reward[c[-1] - model.n]), "start": b})
    return out


def ope_per_episode(eps: list[dict[str, np.ndarray]], model: DriveModel, target: np.ndarray,
                    behavior: np.ndarray, q_hat: np.ndarray) -> dict[str, np.ndarray]:
    """Per-episode pieces of the estimators for a target policy (deterministic actions
    (n,) or probabilities (n, 3)) given behavior probabilities (n, 3) and a model's
    action values q_hat (n, 3) of the target policy (the direct method):

    ratio  rho = product over the episode's decisions of pi(a_t|s_t) / mu(a_t|s_t)
    is     rho * R                                 (trajectory = per-decision IS here,
                                                    since the only reward is at the end)
    dr     V_hat(s_0) + sum over t of rho_{0..t} (r_t + V_hat(s_{t+1}) - Q_hat(s_t, a_t))
           (Jiang and Li 2016), with V_hat(s) = sum over a of pi(a|s) Q_hat(s, a),
           r_t = 0 before the last decision and V_hat = 0 after the drive ends
    dm     V_hat(s_0), the model's value of the target policy at the logged start state
    """
    pi = target if target.ndim == 2 else np.eye(3)[target]
    qh = np.where(np.isfinite(q_hat), q_hat, 0.0)
    v_hat = (pi * qh).sum(axis=1)
    ratio, is_, dr, dm, ret = [], [], [], [], []
    for e in eps:
        s, a, c = e["s"], e["a"], e["c"]
        w = pi[s, a] / behavior[s, a]
        cum = np.cumprod(w)
        nxt = np.append(v_hat[c[:-1]], 0.0)
        r = np.zeros(len(s))
        r[-1] = e["r"]
        dr.append(v_hat[s[0]] + float((cum * (r + nxt - qh[s, a])).sum()))
        ratio.append(cum[-1])
        is_.append(cum[-1] * e["r"])
        dm.append(v_hat[s[0]])
        ret.append(e["r"])
    return {k: np.asarray(v) for k, v in
            (("ratio", ratio), ("is", is_), ("dr", dr), ("dm", dm), ("return", ret))}


def ope_summary(parts: dict[str, np.ndarray], groups: np.ndarray, n_boot: int, seed: int,
                level: float = 0.9) -> dict[str, Any]:
    """Point estimates and game-level bootstrap intervals of IS, weighted IS, DR and the
    direct method, of the logged return (the behavior's own on-policy value) and of each
    estimate minus the logged return. Games are resampled, not drives, because drives of
    one game share the teams and the conditions."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    g = len(uniq)

    def stats(sel_counts: np.ndarray) -> dict[str, float]:
        w = sel_counts[inv]
        tot = w.sum()
        ratio = parts["ratio"]
        out = {
            "is": float((w * parts["is"]).sum() / tot),
            "wis": float((w * parts["is"]).sum() / max((w * ratio).sum(), 1e-12)),
            "dr": float((w * parts["dr"]).sum() / tot),
            "dm": float((w * parts["dm"]).sum() / tot),
            "logged_return": float((w * parts["return"]).sum() / tot),
        }
        for k in ("is", "wis", "dr", "dm"):
            out[f"{k}_minus_logged"] = out[k] - out["logged_return"]
        return out

    point = stats(np.ones(g))
    boots = [stats(rng.multinomial(g, np.full(g, 1.0 / g)).astype(float))
             for _ in range(n_boot)]
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    ci = {k: [float(np.quantile([b[k] for b in boots], lo)),
              float(np.quantile([b[k] for b in boots], hi))] for k in point}
    se = {k: float(np.std([b[k] for b in boots], ddof=1)) for k in point}
    ratio = parts["ratio"]
    return {
        "point": point,
        "interval": ci,
        "bootstrap_se": se,
        "level": level,
        "n_episodes": int(len(ratio)),
        "n_games": int(g),
        "share_episodes_with_nonzero_weight": float((ratio > 0).mean()),
        "effective_sample_size": float(ratio.sum() ** 2 / max((ratio**2).sum(), 1e-12)),
        "max_weight": float(ratio.max()) if len(ratio) else None,
    }
