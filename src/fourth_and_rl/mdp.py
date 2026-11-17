"""M4 shared pieces: the field-state grid, transition kernels, and count tables.

The decision problems in `mdp_drive.py` (expected points, one drive) and `mdp_game.py`
(win probability, whole game) share one description of "what can happen on a play":

* A **field state** is (down, distance bin, yard line). `Grid` numbers them 0..nf-1.
* A **kernel** is a sparse matrix with one row per field state and one column per
  outcome. Outcome columns are laid out as
      [0, nf)            the offense keeps the ball in that field state
      [nf, nf + n_yl)    the other team takes over, first and 10, at THEIR yard line 1..n_yl
      nf + n_yl          touchdown by the offense
      nf + n_yl + 1      safety against the offense
      nf + n_yl + 2      touchdown by the other team on the play (a return)
  Entry [f, c] is the probability of outcome c from field state f. A solver turns the
  columns into values (points or win probability) and takes one matrix product per action.

Nothing in the solvers knows football: tests build tiny kernels by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp
import yaml

from fourth_and_rl.paths import CONFIGS

MODES = ("neutral", "protect", "chase")  # what the offense can choose with a play
# Kernel tables. "neutral" is the generic M1 play, used before the last 5:00.
# "neutral_end" is the same play inside the last 5:00, where going out of bounds also
# stops the clock. "protect" and "chase" are how teams actually play when leading and
# trailing late; "chase_late" is the chase inside the two-minute warning (faster, more
# clock stops).
KERNEL_KEYS = ("neutral", "neutral_end", "protect", "chase", "chase_late")
GAIN_MIN, GAIN_MAX = -30, 100  # same grid as components.GAINS

# Situations of the observed-coach fourth-down table (see `coach_situation`).
SITUATIONS = (
    "neutral",
    "q4_trail_1_8_before_5min",
    "q4_trail_9_plus",
    "5_to_2_min_trail_1_3",
    "5_to_2_min_trail_4_8",
    "last_2_min_trail_1_3",
    "last_2_min_trail_4_8",
    "last_5_min_leading",
    "last_5_min_tied",
    "last_2_min_first_half",
)
COACH_DIST_EDGES = np.array([1, 2, 3, 4, 6, 10])  # 1, 2, 3, 4-5, 6-9, 10+
COACH_YL_WIDTH = 5
COACH_ACTIONS = ("go", "field_goal", "punt")


def load_config(path: Path | None = None) -> dict:
    """Read configs/mdp.yaml."""
    with open(path or CONFIGS / "mdp.yaml") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """Numbering of field states (down, distance bin, yard line).

    Distances 1..dist_exact are their own bin; longer distances fall into `dist_bins`,
    each a (first, last, representative) triple. `n_yl` is the number of yard lines
    (99 for football: yardline_100 = 1..99).
    """

    n_down: int = 4
    n_yl: int = 99
    dist_exact: int = 10
    dist_bins: tuple[tuple[int, int, int], ...] = ((11, 15, 13), (16, 99, 20))

    @property
    def n_dist(self) -> int:
        return self.dist_exact + len(self.dist_bins)

    @property
    def nf(self) -> int:
        return self.n_down * self.n_dist * self.n_yl

    @property
    def na(self) -> int:
        return self.nf + self.n_yl + 3

    @property
    def col_td(self) -> int:
        return self.nf + self.n_yl

    @property
    def col_safety(self) -> int:
        return self.nf + self.n_yl + 1

    @property
    def col_opp_td(self) -> int:
        return self.nf + self.n_yl + 2

    def col_opp(self, yl: int | np.ndarray) -> int | np.ndarray:
        """Outcome column: the other team takes over at THEIR yard line `yl`."""
        return self.nf + np.asarray(yl) - 1

    def dist_bin(self, ydstogo: int | np.ndarray) -> np.ndarray:
        """Distance bin (0-based) of a distance in yards."""
        d = np.maximum(np.asarray(ydstogo), 1)
        out = np.minimum(d, self.dist_exact) - 1
        for i, (first, _last, _rep) in enumerate(self.dist_bins):
            out = np.where(d >= first, self.dist_exact + i, out)
        return out.astype(np.int64)

    def dist_rep(self, dbin: int | np.ndarray) -> np.ndarray:
        """Representative distance in yards of a distance bin."""
        reps = np.array(
            list(range(1, self.dist_exact + 1)) + [b[2] for b in self.dist_bins], dtype=np.int64
        )
        return reps[np.asarray(dbin)]

    def fidx(
        self, down: int | np.ndarray, dbin: int | np.ndarray, yl: int | np.ndarray
    ) -> np.ndarray:
        """Field-state index of (down 1.., distance bin 0.., yard line 1..)."""
        return ((np.asarray(down) - 1) * self.n_dist + np.asarray(dbin)) * self.n_yl + (
            np.asarray(yl) - 1
        )

    def unpack(self, f: int | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(down, distance bin, yard line) of a field-state index."""
        f = np.asarray(f)
        yl = f % self.n_yl + 1
        rest = f // self.n_yl
        return rest // self.n_dist + 1, rest % self.n_dist, yl

    def first10(self) -> np.ndarray:
        """first10()[yl] = field state "first down, 10 to go (or goal to go)" at `yl`.
        Index 0 is unused (set to the state at yard line 1)."""
        yl = np.arange(self.n_yl + 1)
        yl[0] = 1
        return self.fidx(1, self.dist_bin(np.minimum(self.dist_exact, yl)), yl)


# --------------------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------------------


@dataclass
class FieldKernels:
    """Transition kernels over field states (see the module docstring for the columns).

    k_all[mode]   (nf, na): running a play (on fourth down: going for it), all outcomes
    k_in[mode]    (nf, nf): the part of k_all where the offense keeps the ball AND the
                  clock keeps running (the play ended in bounds). k_all minus k_in is
                  everything after which the clock is stopped.
    fg_make       (nf,) probability a field goal is good; fg_miss_col the outcome column
                  after a miss; fg_ok where the action is allowed
    k_punt        (nf, na), punt_ok
    kneel_col     (nf,) outcome column after a kneel; kneel_keep where the offense keeps
                  the ball (downs before the last); kneel_ok where it is allowed
    kickoff, free_kick  (n_yl + 1,) receiving team's yard line, index 0 unused (zero)
    """

    grid: Grid
    k_all: dict[str, sp.csr_matrix]
    k_in: dict[str, sp.csr_matrix]
    fg_make: np.ndarray
    fg_miss_col: np.ndarray
    fg_ok: np.ndarray
    k_punt: sp.csr_matrix
    punt_ok: np.ndarray
    kneel_col: np.ndarray
    kneel_keep: np.ndarray
    kneel_ok: np.ndarray
    kickoff: np.ndarray
    free_kick: np.ndarray
    p_convert: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (nf,), fourth downs


def _drop_return(dist: np.ndarray) -> np.ndarray:
    """Remove the 'returned for a touchdown' entry (index 0) and renormalize."""
    out = np.asarray(dist, dtype=float).copy()
    out[0] = 0.0
    return out / out.sum()


def adjusted_play(
    kept: np.ndarray,
    p_td: float,
    tail: np.ndarray,
    gain: np.ndarray,
    yl: int,
    tilt: float,
    turnover_scale: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Bend one generic play: reweight yards gained by exp(tilt * yards) and multiply the
    turnover probability by `turnover_scale`.

    kept[i] is the probability of keeping the ball after gaining gain[i]; p_td a touchdown
    (treated as a gain of `yl`); tail = [turnover at each yard line..., safety, turnover
    returned for a touchdown]. The safety entry is left alone. Probabilities still sum to 1.
    """
    p_safety = float(tail[-2])
    p_to = float(tail.sum()) - p_safety
    new_to = min(p_to * turnover_scale, 0.5)
    w = np.exp(tilt * gain)
    w_td = float(np.exp(tilt * yl))
    scale = (1.0 - p_safety - new_to) / (float((kept * w).sum()) + p_td * w_td)
    new_tail = tail * (new_to / p_to if p_to > 0 else 1.0)
    new_tail[-2] = p_safety
    return kept * w * scale, p_td * w_td * scale, new_tail


def build_field_kernels(
    models: Any,
    grid: Grid,
    stop_prob: dict[str, np.ndarray],
    free_kick: np.ndarray | None = None,
    go_convert_shift: float = 0.0,
    adjust: dict[str, dict[str, float]] | None = None,
    field_goal_any_down: bool = True,
    kneel: bool = True,
) -> FieldKernels:
    """Turn the M1 component models into kernels on `grid`.

    `stop_prob[key][g - GAIN_MIN]` is the probability that a play which gained g yards and
    kept the ball left the clock stopped (incomplete pass or out of bounds), for each of
    KERNEL_KEYS. `adjust[key]` = {"tilt": a, "turnover_scale": b} bends the generic play
    of downs 1 to 3 for that table: yards gained are reweighted by exp(a * yards) and the
    turnover probability is multiplied by b (see `fit_mode_adjustments`). Missing keys
    mean no change. `go_convert_shift` moves every fourth-down conversion probability by
    that amount (sensitivity analysis).
    """
    adjust = adjust or {}
    nf, na, n_yl = grid.nf, grid.na, grid.n_yl
    f10 = grid.first10()
    last = grid.n_down
    rows: dict[str, list] = {m: [] for m in KERNEL_KEYS}
    cols: dict[str, list] = {m: [] for m in KERNEL_KEYS}
    vals: dict[str, list] = {m: [] for m in KERNEL_KEYS}
    in_rows: dict[str, list] = {m: [] for m in KERNEL_KEYS}
    in_cols: dict[str, list] = {m: [] for m in KERNEL_KEYS}
    in_vals: dict[str, list] = {m: [] for m in KERNEL_KEYS}
    p_convert = np.zeros(nf)
    y_all = np.arange(100)

    def add(mode: str, f: int, c: np.ndarray, v: np.ndarray) -> None:
        keep = v > 0
        rows[mode].append(np.full(int(keep.sum()), f))
        cols[mode].append(np.asarray(c)[keep])
        vals[mode].append(np.asarray(v, dtype=float)[keep])

    def add_in(mode: str, f: int, c: np.ndarray, v: np.ndarray) -> None:
        keep = v > 0
        in_rows[mode].append(np.full(int(keep.sum()), f))
        in_cols[mode].append(np.asarray(c)[keep])
        in_vals[mode].append(np.asarray(v, dtype=float)[keep])

    for down in range(1, last + 1):
        for dbin in range(grid.n_dist):
            for yl in range(1, n_yl + 1):
                f = int(grid.fidx(down, dbin, yl))
                d = int(min(grid.dist_rep(dbin), yl))
                if down < last:
                    o = models.play_outcomes(down, d, yl)
                    kept, p_td = o["keep"], float(o["p_touchdown"])
                    ys = np.nonzero(kept)[0]
                    gain = yl - ys
                    made = gain >= d
                    target = np.where(
                        made, f10[ys], grid.fidx(down + 1, grid.dist_bin(d - gain), ys)
                    )
                    tail = np.concatenate([o["turnover"][1:], [o["p_safety"], o["turnover"][0]]])
                    tail_cols = np.concatenate(
                        [grid.col_opp(y_all[1:]), [grid.col_safety, grid.col_opp_td]]
                    )
                else:
                    o = models.go_outcomes(yl, d)
                    p = float(o["p_convert"])
                    p_new = float(np.clip(p + go_convert_shift, 1e-6, 1 - 1e-6))
                    up, down_scale = p_new / p, (1.0 - p_new) / (1.0 - p)
                    kept, p_td = o["first_down"] * up, float(o["p_touchdown"]) * up
                    ys = np.nonzero(kept)[0]
                    gain = yl - ys
                    target = f10[ys]
                    tail = down_scale * np.concatenate(
                        [o["turnover"][1:], [o["p_safety"], o["turnover"][0]]]
                    )
                    tail_cols = np.concatenate(
                        [grid.col_opp(y_all[1:]), [grid.col_safety, grid.col_opp_td]]
                    )
                    p_convert[f] = p_new
                for mode in KERNEL_KEYS:
                    k, td, tl = kept[ys], p_td, tail
                    adj = adjust.get(mode)
                    if adj is not None and down < last:
                        k, td, tl = adjusted_play(
                            k,
                            td,
                            tail,
                            gain,
                            yl,
                            adj.get("tilt", 0.0),
                            adj.get("turnover_scale", 1.0),
                        )
                    stop = stop_prob[mode][np.clip(gain, GAIN_MIN, GAIN_MAX) - GAIN_MIN]
                    add(mode, f, target, k)
                    add(mode, f, np.array([grid.col_td]), np.array([td]))
                    add(mode, f, tail_cols, tl)
                    add_in(mode, f, target, k * (1.0 - stop))

    def csr(r: list, c: list, v: list, shape: tuple[int, int]) -> sp.csr_matrix:
        m = sp.coo_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=shape)
        return m.tocsr()

    k_all = {m: csr(rows[m], cols[m], vals[m], (nf, na)) for m in KERNEL_KEYS}
    k_in = {m: csr(in_rows[m], in_cols[m], in_vals[m], (nf, nf)) for m in KERNEL_KEYS}

    downs, dbins, yls = grid.unpack(np.arange(nf))
    fg_make = np.asarray(models.p_field_goal(yls), dtype=float)
    miss_yl = np.array([models.fg_miss_yardline(int(y)) for y in range(1, n_yl + 1)])
    fg_miss_col = np.asarray(grid.col_opp(miss_yl[yls - 1]))
    fg_ok = (fg_make > 0) & (field_goal_any_down | (downs == last))

    pr, pc, pv = [], [], []
    for yl in range(1, n_yl + 1):
        full = models.punt_outcomes_full(yl)
        opp, own = full["opponent"], full["kicking_team"]
        c = np.concatenate([[grid.col_opp_td], grid.col_opp(y_all[1:]), f10[y_all[1:]]])
        v = np.concatenate([[opp[0]], opp[1:], own[1:]])
        v = v / v.sum()
        for dbin in range(grid.n_dist):
            f = int(grid.fidx(last, dbin, yl))
            keep = v > 0
            pr.append(np.full(int(keep.sum()), f))
            pc.append(c[keep])
            pv.append(v[keep])
    k_punt = csr(pr, pc, pv, (nf, na))
    punt_ok = downs == last

    d_rep = np.minimum(grid.dist_rep(dbins), yls)
    back = np.minimum(yls + 1, n_yl)
    kneel_keep = downs < last
    kneel_col = np.where(
        kneel_keep,
        grid.fidx(np.minimum(downs + 1, last), grid.dist_bin(d_rep + 1), back),
        grid.col_opp(n_yl + 1 - back),
    )
    kneel_ok = np.full(nf, bool(kneel))

    kickoff = _drop_return(models.kickoff_start())
    fk = kickoff if free_kick is None else _drop_return(free_kick)
    return FieldKernels(
        grid=grid,
        k_all=k_all,
        k_in=k_in,
        fg_make=fg_make,
        fg_miss_col=fg_miss_col,
        fg_ok=fg_ok,
        k_punt=k_punt,
        punt_ok=punt_ok,
        kneel_col=kneel_col,
        kneel_keep=kneel_keep,
        kneel_ok=kneel_ok,
        kickoff=kickoff,
        free_kick=fk,
        p_convert=p_convert,
    )


def grid_from_config(cfg: dict) -> Grid:
    g = cfg["grid"]
    return Grid(dist_exact=g["dist_exact"], dist_bins=tuple(tuple(b) for b in g["dist_bins"]))


# --------------------------------------------------------------------------------------
# Count tables fitted from the play-by-play cache
# --------------------------------------------------------------------------------------


def coach_situation(
    game_seconds: np.ndarray, half_seconds: np.ndarray, score: np.ndarray
) -> np.ndarray:
    """Index into SITUATIONS for each (game seconds left, half seconds left, score
    differential of the team with the ball). Used both to fit the coach table and to
    look it up in the simulator, so the two always agree."""
    t, h, s = np.asarray(game_seconds), np.asarray(half_seconds), np.asarray(score)
    out = np.zeros(np.shape(t), dtype=np.int64)
    first_half = t > 1800
    out = np.where(first_half & (h <= 120), 9, out)
    q4 = t <= 900
    out = np.where(q4 & (t > 300) & (s <= -1) & (s >= -8), 1, out)
    out = np.where(q4 & (s <= -9), 2, out)
    mid, late = (t <= 300) & (t > 120), t <= 120
    out = np.where(mid & (s <= -1) & (s >= -3), 3, out)
    out = np.where(mid & (s <= -4) & (s >= -8), 4, out)
    out = np.where(late & (s <= -1) & (s >= -3), 5, out)
    out = np.where(late & (s <= -4) & (s >= -8), 6, out)
    out = np.where((t <= 300) & (s >= 1), 7, out)
    out = np.where((t <= 300) & (s == 0), 8, out)
    return out


def _sequence(plays: pl.DataFrame) -> pl.DataFrame:
    """Real plays in order, with the gap to the next row and whether a timeout followed."""
    is_home = pl.col("is_home") == 1
    off, deff = pl.col("posteam_timeouts_remaining"), pl.col("defteam_timeouts_remaining")
    df = (
        plays.filter(pl.col("play_type").is_not_null())
        .sort("game_id", "order_sequence", "play_id")
        .with_columns(
            pl.when(is_home).then(off).otherwise(deff).alias("_home_to"),
            pl.when(is_home).then(deff).otherwise(off).alias("_away_to"),
        )
    )

    def nxt(col: str) -> pl.Expr:
        return pl.col(col).shift(-1).over("game_id")

    desc = pl.col("desc").fill_null("")
    return df.with_columns(
        (pl.col("game_seconds_remaining") - nxt("game_seconds_remaining")).alias("_elapsed"),
        (nxt("qtr") == pl.col("qtr")).fill_null(False).alias("_same_qtr"),
        nxt("half_seconds_remaining").alias("_next_half_seconds"),
        (
            ((pl.col("_home_to") - nxt("_home_to")) > 0)
            | ((pl.col("_away_to") - nxt("_away_to")) > 0)
        )
        .fill_null(False)
        .alias("_timeout_after"),
        desc.str.contains(r"(?i)\bob\b|out of bounds").alias("_oob"),
        desc.str.contains("(?i)incomplete").alias("_incomplete"),
    )


def fit_tables(plays: pl.DataFrame, cfg: dict) -> dict[str, Any]:
    """Count tables the game model needs beyond the M1 component models.

    Returns plain lists and numbers (JSON friendly):
      stop_prob[mode]      P(clock stopped | yards-gained bucket) for kept-ball plays
      stop_prob_n[mode]    sample sizes
      neutral_hist         P(ticks to the next snap) for clock-running plays before 5:00
      hurry_hist           the same for a trailing offense in the last 5:00, timeouts removed
      burn_median_seconds  a leading offense's in-bounds play, snap to snap (a check)
      free_kick            receiving team's yard line after a safety
    """
    t = cfg["tables"]
    tick = cfg["clock"]["tick_seconds"]
    max_ticks = cfg["clock"]["max_play_ticks"]
    edges = np.array(t["gain_bucket_edges"])
    one = t["one_score"]
    df = _sequence(plays.filter(pl.col("season") >= t["first_season"]))
    kept = df.filter(
        pl.col("play_type").is_in(["run", "pass"])
        & pl.col("down").is_not_null()
        & pl.col("yards_gained").is_not_null()
        & (pl.col("qtr") <= 4)
        & (pl.col("penalty").fill_null(0) != 1)
        & (pl.col("touchdown").fill_null(0) != 1)
        & (pl.col("interception").fill_null(0) != 1)
        & (pl.col("fumble_lost").fill_null(0) != 1)
        & (pl.col("safety").fill_null(0) != 1)
        & ~((pl.col("down") == 4) & (pl.col("first_down").fill_null(0) != 1))
    )
    last5 = pl.col("game_seconds_remaining") <= cfg["clock"]["endgame_seconds"]
    last2 = pl.col("game_seconds_remaining") <= cfg["clock"]["warning_seconds"]
    diff = pl.col("score_differential")
    contexts = {
        "neutral": ~last5,
        "neutral_end": ~last5,
        "protect": last5 & diff.is_between(1, one),
        "chase": last5 & ~last2 & diff.is_between(-one, -1),
        "chase_late": last2 & diff.is_between(-one, -1),
    }
    out: dict[str, Any] = {"stop_prob": {}, "stop_prob_n": {}, "gain_bucket_edges": edges.tolist()}
    for mode, ctx in contexts.items():
        d = kept.filter(ctx)
        gain = d["yards_gained"].to_numpy().astype(int)
        # Before the last 5:00 the clock restarts after an out-of-bounds play, so only an
        # incomplete pass counts as a stop there.
        stop = d["_incomplete"].to_numpy()
        if mode != "neutral":
            stop = stop | d["_oob"].to_numpy()
        bucket = np.clip(np.searchsorted(edges, gain, side="right") - 1, 0, len(edges) - 1)
        n = np.bincount(bucket, minlength=len(edges))
        k = np.bincount(bucket, weights=stop.astype(float), minlength=len(edges))
        out["stop_prob"][mode] = np.divide(k, n, out=np.zeros(len(edges)), where=n > 0).tolist()
        out["stop_prob_n"][mode] = n.tolist()

    def hist(d: pl.DataFrame) -> tuple[list[float], int, float]:
        sec = d["_elapsed"].to_numpy().astype(float)
        ticks = np.clip(np.rint(sec / tick), 1, max_ticks).astype(int)
        h = np.bincount(ticks, minlength=max_ticks + 1)[1:].astype(float)
        return (h / h.sum()).tolist(), int(h.sum()), float(np.median(sec))

    timed = kept.filter(
        pl.col("_same_qtr") & pl.col("_elapsed").is_between(0, t["max_elapsed_seconds"])
    )
    running = timed.filter(~pl.col("_incomplete"))
    out["neutral_hist"], out["neutral_hist_n"], out["neutral_median_seconds"] = hist(
        running.filter(contexts["neutral"])
    )
    crosses = (pl.col("half_seconds_remaining") > 120) & (pl.col("_next_half_seconds") <= 120)
    in_bounds = running.filter(~pl.col("_oob") & ~pl.col("_timeout_after") & ~crosses)
    for key, ctx in (("hurry_hist", "chase"), ("hurry_hist_late", "chase_late")):
        out[key], out[f"{key}_n"], out[f"{key}_median_seconds"] = hist(
            in_bounds.filter(contexts[ctx])
        )
    _, out["burn_n"], out["burn_median_seconds"] = hist(in_bounds.filter(contexts["protect"]))

    # Free kick after a safety: where the receiving team starts.
    from fourth_and_rl.components import add_next_state

    saf = add_next_state(plays).filter(
        (pl.col("safety") == 1)
        & (pl.col("next_posteam") != pl.col("posteam"))
        & (pl.col("next_half") == pl.col("game_half"))
        & pl.col("next_yardline_100").is_between(1, 99)
    )
    fk = np.bincount(saf["next_yardline_100"].to_numpy().astype(int), minlength=100)[:100]
    out["free_kick_n"] = int(fk.sum())
    out["free_kick"] = (fk / fk.sum()).tolist() if fk.sum() >= 30 else None
    out["free_kick_mean_yardline_100"] = float(fk @ np.arange(100) / fk.sum()) if fk.sum() else None
    return out


def stop_prob_by_gain(tables: dict[str, Any]) -> dict[str, np.ndarray]:
    """Expand the bucketed stop probabilities to one value per yard gained."""
    edges = np.array(tables["gain_bucket_edges"])
    gains = np.arange(GAIN_MIN, GAIN_MAX + 1)
    bucket = np.clip(np.searchsorted(edges, gains, side="right") - 1, 0, len(edges) - 1)
    return {m: np.asarray(tables["stop_prob"][m])[bucket] for m in KERNEL_KEYS}


def fit_mode_adjustments(plays: pl.DataFrame, models: Any, cfg: dict) -> dict[str, Any]:
    """How "protect" and "chase" football differs from the generic play of M1.

    For plays on downs 1 to 3 in the last 5:00 by a team leading (protect) or trailing
    (chase) by one score, compare what happened with what the generic play model predicts
    for the same downs, distances and yard lines. Two numbers per mode make the model match:
      turnover_scale  observed turnover rate / predicted turnover rate
      tilt            the a in exp(a * yards) that makes the predicted first-down rate
                      equal the observed one (found by bisection)
    The last `chase_exclude_last_seconds` are left out of the chase sample: those plays are
    mostly desperation heaves, which say little about a real drive.
    """
    t = cfg["tables"]
    one = t["one_score"]
    base = plays.filter(
        (pl.col("season") >= t["first_season"])
        & pl.col("play_type").is_in(["run", "pass"])
        & pl.col("down").is_between(1, 3)
        & pl.col("yards_gained").is_not_null()
        & (pl.col("qtr") == 4)
        & pl.col("yardline_100").is_between(1, 99)
        & (pl.col("ydstogo") >= 1)
        & (pl.col("game_seconds_remaining") <= cfg["clock"]["endgame_seconds"])
    )
    diff = pl.col("score_differential")
    contexts = {
        "protect": diff.is_between(1, one),
        "chase": diff.is_between(-one, -1)
        & (pl.col("game_seconds_remaining") > t["chase_exclude_last_seconds"]),
    }
    out: dict[str, Any] = {}
    cache: dict[tuple[int, int, int], tuple] = {}
    for mode, ctx in contexts.items():
        d = base.filter(ctx)
        down = d["down"].to_numpy().astype(int)
        yl = d["yardline_100"].to_numpy().astype(int)
        dist = np.minimum(d["ydstogo"].to_numpy().astype(int), yl)
        turnover = (
            (d["interception"].fill_null(0) == 1) | (d["fumble_lost"].fill_null(0) == 1)
        ).to_numpy()
        made = (
            (d["first_down"].fill_null(0) == 1) | (d["touchdown"].fill_null(0) == 1)
        ).to_numpy() & ~turnover
        states, counts = np.unique(np.column_stack([down, dist, yl]), axis=0, return_counts=True)
        parts = []
        for dn, ds, y in states:
            key = (int(dn), int(ds), int(y))
            if key not in cache:
                o = models.play_outcomes(*key)
                ys = np.nonzero(o["keep"])[0]
                tail = np.concatenate([o["turnover"][1:], [o["p_safety"], o["turnover"][0]]])
                cache[key] = (o["keep"][ys], float(o["p_touchdown"]), tail, y - ys)
            parts.append(cache[key])
        weights = counts / counts.sum()
        pred_to = float(
            sum(w * (p[2].sum() - p[2][-2]) for w, p in zip(weights, parts, strict=True))
        )
        scale = float(turnover.mean() / pred_to) if pred_to > 0 else 1.0

        def first_down_rate(
            tilt: float,
            scale: float = scale,
            parts: list = parts,
            states: np.ndarray = states,
            weights: np.ndarray = weights,
        ) -> float:
            total = 0.0
            for w, part, state in zip(weights, parts, states, strict=True):
                kept, p_td, tail, gain = part
                k, td, _ = adjusted_play(kept, p_td, tail.copy(), gain, int(state[2]), tilt, scale)
                total += w * (float(k[gain >= state[1]].sum()) + td)
            return total

        lo, hi = -0.3, 0.3
        target = float(made.mean())
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if first_down_rate(mid) < target else (lo, mid)
        tilt = 0.5 * (lo + hi)
        out[mode] = {
            "n": int(d.height),
            "tilt": tilt,
            "turnover_scale": scale,
            "first_down_rate_observed": target,
            "first_down_rate_generic_model": first_down_rate(0.0, 1.0),
            "first_down_rate_adjusted_model": first_down_rate(tilt),
            "turnover_rate_observed": float(turnover.mean()),
            "turnover_rate_generic_model": pred_to,
            "mean_gain_observed_without_turnovers": float(
                d["yards_gained"].to_numpy()[~turnover].mean()
            ),
        }
    return out


def mode_adjustments(tables: dict[str, Any], cfg: dict, chase_extra_tilt: float = 0.0) -> dict:
    """Per-kernel adjustments for `build_field_kernels`, from the fitted numbers. With
    `tables.calibrate_modes` false in the config every table is the generic play."""
    if not cfg["tables"].get("calibrate_modes", True) or "adjust" not in tables:
        base = {
            "protect": {"tilt": 0.0, "turnover_scale": 1.0},
            "chase": {"tilt": 0.0, "turnover_scale": 1.0},
        }
    else:
        base = tables["adjust"]
    pick = lambda m: {"tilt": base[m]["tilt"], "turnover_scale": base[m]["turnover_scale"]}  # noqa: E731
    chase = pick("chase")
    chase["tilt"] += chase_extra_tilt
    return {"protect": pick("protect"), "chase": chase, "chase_late": dict(chase)}


def fit_coach_table(plays: pl.DataFrame, grid: Grid, cfg: dict) -> dict[str, Any]:
    """Observed fourth-down choices, as P(go, field goal, punt | situation, distance, field).

    Cells are (situation, distance group, 5-yard field band). A thin cell is pulled toward
    a coarser cell (same situation and distance, three field zones), which is in turn pulled
    toward the neutral-situation cell, each with `prior_strength` pseudo-counts.
    Returns {"table": array (n_situations, n_dist, n_yl, 3), "n": cell counts}.
    """
    c = cfg["coach"]
    k0 = float(c["prior_strength"])
    d = plays.filter(
        (pl.col("season") >= c["first_season"])
        & pl.col("fourth_down_decision").is_not_null()
        & (pl.col("qtr") <= 4)
        & pl.col("yardline_100").is_between(1, 99)
    )
    sit = coach_situation(
        d["game_seconds_remaining"].to_numpy(),
        d["half_seconds_remaining"].to_numpy(),
        d["score_differential"].to_numpy(),
    )
    dg = np.clip(
        np.searchsorted(COACH_DIST_EDGES, d["ydstogo"].to_numpy(), side="right") - 1,
        0,
        len(COACH_DIST_EDGES) - 1,
    )
    yl = d["yardline_100"].to_numpy().astype(int)
    band = (yl - 1) // COACH_YL_WIDTH
    zone = np.where(yl <= 35, 0, np.where(yl <= 60, 1, 2))
    act = np.array([COACH_ACTIONS.index(a) for a in d["fourth_down_decision"].to_list()])
    n_sit, n_dg, n_band = len(SITUATIONS), len(COACH_DIST_EDGES), 100 // COACH_YL_WIDTH
    fine = np.zeros((n_sit, n_dg, n_band, 3))
    np.add.at(fine, (sit, dg, band, act), 1.0)
    coarse = np.zeros((n_sit, n_dg, 3, 3))
    np.add.at(coarse, (sit, dg, zone, act), 1.0)
    band_zone = np.array(
        [
            0 if b * COACH_YL_WIDTH + 3 <= 35 else (1 if b * COACH_YL_WIDTH + 3 <= 60 else 2)
            for b in range(n_band)
        ]
    )

    def shrink(counts: np.ndarray, prior: np.ndarray) -> np.ndarray:
        return (counts + k0 * prior) / (counts.sum(axis=-1, keepdims=True) + k0)

    uniform = np.full(3, 1.0 / 3.0)
    neutral_fine = shrink(fine[0], shrink(coarse[0], uniform)[:, band_zone])  # (dg, band, 3)
    table_bands = np.zeros_like(fine)
    table_bands[0] = neutral_fine
    for s in range(1, n_sit):
        parent = shrink(coarse[s][:, band_zone], neutral_fine)
        table_bands[s] = shrink(fine[s], parent)
    yls = np.arange(1, grid.n_yl + 1)
    reps = grid.dist_rep(np.arange(grid.n_dist))
    dgs = np.clip(np.searchsorted(COACH_DIST_EDGES, reps, side="right") - 1, 0, n_dg - 1)
    table = table_bands[:, dgs][:, :, (yls - 1) // COACH_YL_WIDTH]
    return {
        "table": table,
        "n_by_situation": fine.sum(axis=(1, 2, 3)).astype(int).tolist(),
        "n": int(d.height),
        "first_season": int(c["first_season"]),
    }
