"""M2 one-step decision engine: the win probability of going, kicking and punting.

For a fourth-down state s and an action a, the engine computes

    value(s, a) = sum over outcomes o of  P(o | s, a) * V(next state after o)

where P comes from the component models (full discrete distributions, see
`components.py`) and V is the win-probability model applied to the NEXT state, read
from the point of view of the team that faces the fourth down now ("us"):

    V = WP(next state)          if we still have the ball
    V = 1 - WP(next state)      if the other team has it, with the score differential,
                                the spread, the timeouts, `is_home` and `receive_2h_ko`
                                all flipped to their point of view
    V = 1, 0 or `tie_value`     if the play ran out the game clock

"One step" means the engine looks exactly one play ahead and then trusts the
win-probability model. It never plans the plays after that; the full-game dynamic
program (M4) does.

Public API: DecisionEngine, load_config, fit_evaluation_components, load_engine,
chart_states, ACTIONS. Teaching notes: docs/notes/m2_decision.md.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl
import yaml

from fourth_and_rl import splits
from fourth_and_rl import wp as wp_mod
from fourth_and_rl.components import KICKOFF_RULE_REASON, ComponentModels
from fourth_and_rl.paths import CONFIGS, ROOT

ACTIONS: tuple[str, str, str] = ("go", "field_goal", "punt")

# Accumulator slots: every outcome's weighted value is added to one of these.
_CONVERT, _FAIL, _FG_MAKE, _FG_MISS, _PUNT = range(5)
_N_SLOTS = 5

# Event kinds: what the situation is right after the play.
_US, _OPP, _KICK_TO_OPP, _KICK_TO_US = range(4)

_DEFAULT_CONFIG: dict[str, Any] = {
    "clock": {
        "use_fitted": True,
        "go_convert": 27,
        "go_fail": 5,
        "score": 5,
        "field_goal": 4,
        "punt": 9,
        "kickoff": 2,
    },
    "tie_value": 0.5,
    "punt_min_yardline_100": 31,
    "compress": {"merge_mass": 0.004, "max_span": 8},
    "predict_chunk_rows": 1_000_000,
    "state_chunk_rows": 20_000,
}


class WinProbability(Protocol):
    """Anything with `predict(frame of wp.STATE_COLUMNS) -> array of P(posteam wins)`."""

    def predict(self, df: pl.DataFrame) -> np.ndarray: ...


def load_config(path: Path | None = None) -> dict:
    """Read configs/decision.yaml."""
    with open(path or CONFIGS / "decision.yaml") as f:
        return yaml.safe_load(f)


def _merged(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merged(out[k], v)
        else:
            out[k] = v
    return out


def compress_distribution(
    p: np.ndarray, merge_mass: float, max_span: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink a distribution over yard lines without moving its mass or its mean much.

    `p[y]` is the probability of yard line y. Yard lines with at least `merge_mass`
    are kept exactly. Runs of neighbouring yard lines that each carry less are merged
    into one point at their probability-weighted mean yard line (rounded), and a merged
    point is closed as soon as it holds `merge_mass` or spans `max_span` yards. Total
    probability is unchanged. `merge_mass <= 0` returns the distribution untouched.

    Returns (yard lines, probabilities).
    """
    idx = np.nonzero(p > 0)[0]
    if merge_mass <= 0 or idx.size <= 1:
        return idx.astype(np.int64), p[idx].astype(float)
    ys: list[int] = []
    ws: list[float] = []
    acc_w = acc_wy = 0.0
    start = -1
    for y in idx.tolist():
        w = float(p[y])
        if w >= merge_mass:
            if start >= 0:
                ys.append(int(round(acc_wy / acc_w)))
                ws.append(acc_w)
                acc_w = acc_wy = 0.0
                start = -1
            ys.append(y)
            ws.append(w)
            continue
        if start >= 0 and y - start >= max_span:
            ys.append(int(round(acc_wy / acc_w)))
            ws.append(acc_w)
            acc_w = acc_wy = 0.0
            start = -1
        if start < 0:
            start = y
        acc_w += w
        acc_wy += w * y
        if acc_w >= merge_mass:
            ys.append(int(round(acc_wy / acc_w)))
            ws.append(acc_w)
            acc_w = acc_wy = 0.0
            start = -1
    if start >= 0:
        ys.append(int(round(acc_wy / acc_w)))
        ws.append(acc_w)
    return np.asarray(ys, dtype=np.int64), np.asarray(ws, dtype=float)


class DecisionEngine:
    """Win probability of each fourth-down action, one play ahead.

    `wp_model` is a fitted `WPModel` (or any object with the same `predict`).
    `components` is a fitted `ComponentModels` (or any object with `go_outcomes`,
    `punt_outcomes`, `p_field_goal`, `fg_miss_yardline`, `kickoff_start` and
    `p_extra_point`). `config` overrides the defaults documented in
    configs/decision.yaml.

    Kickoffs. When `states` has a `season` column and the components know kickoff rule
    eras (`components.kickoff_era`), each row is valued with the kickoff table of its own
    season's rule; hypothetical states without a season use the latest rule.

    Leakage. `stamp` is the components' fit stamp (None for objects without one).
    `evaluate(states, out_of_sample=True)` (or `evaluate_out_of_sample`) refuses to score
    real plays from seasons that either model was fitted on.
    """

    def __init__(
        self, wp_model: WinProbability, components: Any, config: dict | None = None
    ) -> None:
        self.wp_model = wp_model
        self.components = components
        self.config = _merged(_DEFAULT_CONFIG, config)
        self.seconds = self._resolve_seconds()
        self.merge_mass = float(self.config["compress"]["merge_mass"])
        self.max_span = int(self.config["compress"]["max_span"])
        self.tie_value = float(self.config["tie_value"])
        self.p_xp = float(components.p_extra_point())
        self.stamp: dict | None = getattr(components, "stamp", None)
        self._ko_era: str | None = "unset"
        self._ko_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._set_kickoff(None)
        self._go_cache: dict[tuple[int, int], tuple[np.ndarray, ...]] = {}
        self._kick_cache: dict[int, tuple[np.ndarray, ...]] = {}
        self.last_stats: dict[str, int] = {}

    # ------------------------------------------------------------------ configuration

    def _resolve_seconds(self) -> dict[str, int]:
        """Whole seconds taken off the game clock by each kind of play.

        With `clock.use_fitted` the values are read from `components.seconds_per_play()`
        (see the config file for which entry feeds which play); otherwise, or when a
        fitted value is missing, the fixed numbers in the config are used.
        """
        clock = self.config["clock"]
        out = {k: float(clock[k]) for k in ("go_convert", "go_fail", "score", "field_goal",
                                            "punt", "kickoff")}  # fmt: skip
        fitted: dict[str, Any] = {}
        if clock.get("use_fitted") and hasattr(self.components, "seconds_per_play"):
            fitted = self.components.seconds_per_play() or {}
        if fitted:
            mean = fitted.get("mean_scrimmage_play")
            if mean is not None and np.isfinite(mean):
                out["go_convert"] = float(mean)
            for row in fitted.get("by_type", []):
                if (row["play"], row["clock"], row["phase"]) == (
                    "pass_incomplete", "stops", "normal"
                ):  # fmt: skip
                    out["go_fail"] = out["score"] = float(row["mean_seconds"])
            for name in ("field_goal", "punt", "kickoff"):
                kick = fitted.get("kicks", {}).get(name)
                if kick and np.isfinite(kick["mean_seconds"]):
                    out[name] = float(kick["mean_seconds"])
        return {k: max(1, int(round(v))) for k, v in out.items()}

    # ---------------------------------------------------------------------- kickoffs

    def _kickoff_eras(self, states: pl.DataFrame) -> np.ndarray | None:
        """Kickoff rule era of each row, or None when every row uses the default table."""
        if "season" not in states.columns or not hasattr(self.components, "kickoff_era"):
            return None
        if not getattr(self.components, "kickoff_tables", None):
            return None
        seasons = states["season"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        names = {}
        for season in np.unique(seasons[~np.isnan(seasons)]):
            names[float(season)] = self.components.kickoff_era(int(season))
        default = self.components.kickoff_era(None)
        return np.array([names.get(float(x), default) if not np.isnan(x) else default
                         for x in seasons], dtype=object)  # fmt: skip

    def _set_kickoff(self, era: str | None) -> None:
        """Use the kickoff table of rule era `era` (None = the components' default)."""
        if era == self._ko_era:
            return
        c = self.components
        if era is None:
            ko = np.asarray(c.kickoff_start(), dtype=float)
        else:
            ko = np.asarray(c.kickoff_tables[era], dtype=float)
        self._ko_era = era
        self._ko_return_td = float(ko[0])
        self._ko_field = ko.copy()
        self._ko_field[0] = 0.0
        field_mass = self._ko_field.sum()
        self._ko_mean_start = int(round(float(self._ko_field @ np.arange(100)) / field_mass))
        self._ko_cache = {}

    # ---------------------------------------------------------------------- templates

    def _squeeze(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return compress_distribution(p, self.merge_mass, self.max_span)

    def _go_template(self, yl: int, togo: int) -> tuple[np.ndarray, ...]:
        """Events of a fourth-down attempt: (slot, kind, yard line, weight, points, seconds)."""
        key = (yl, togo)
        if key in self._go_cache:
            return self._go_cache[key]
        out = self.components.go_outcomes(yl, togo)
        s = self.seconds
        rows: list[tuple[int, int, int, float, int, int]] = []
        first_down = np.asarray(out["first_down"], dtype=float).copy()
        first_down[0] = 0.0
        for y, w in zip(*self._squeeze(first_down), strict=True):
            rows.append((_CONVERT, _US, int(y), float(w), 0, s["go_convert"]))
        p_td = float(out["p_touchdown"])
        if p_td > 0:
            rows.append((_CONVERT, _KICK_TO_OPP, 0, p_td * self.p_xp, 7, s["score"]))
            rows.append((_CONVERT, _KICK_TO_OPP, 0, p_td * (1 - self.p_xp), 6, s["score"]))
        turnover = np.asarray(out["turnover"], dtype=float).copy()
        p_return_td = float(turnover[0])
        turnover[0] = 0.0
        for y, w in zip(*self._squeeze(turnover), strict=True):
            rows.append((_FAIL, _OPP, int(y), float(w), 0, s["go_fail"]))
        if p_return_td > 0:
            rows.append((_FAIL, _KICK_TO_US, 0, p_return_td * self.p_xp, -7, s["go_fail"]))
            rows.append((_FAIL, _KICK_TO_US, 0, p_return_td * (1 - self.p_xp), -6, s["go_fail"]))
        p_safety = float(out.get("p_safety", 0.0))
        if p_safety > 0:
            rows.append((_FAIL, _KICK_TO_OPP, 0, p_safety, -2, s["go_fail"]))
        template = self._pack(rows) + (np.array([float(out["p_convert"])]),)
        self._go_cache[key] = template
        return template

    def _kick_template(self, yl: int) -> tuple[np.ndarray, ...]:
        """Events of a field goal attempt and of a punt from this yard line."""
        if yl in self._kick_cache:
            return self._kick_cache[yl]
        s = self.seconds
        c = self.components
        rows: list[tuple[int, int, int, float, int, int]] = []
        p_fg = float(c.p_field_goal(yl))
        if p_fg > 0:
            rows.append((_FG_MAKE, _KICK_TO_OPP, 0, p_fg, 3, s["field_goal"]))
        rows.append((_FG_MISS, _OPP, int(c.fg_miss_yardline(yl)), 1.0 - p_fg, 0, s["field_goal"]))
        if hasattr(c, "punt_outcomes_full"):
            full = c.punt_outcomes_full(yl)
            to_opp = np.asarray(full["opponent"], dtype=float).copy()
            kept = np.asarray(full["kicking_team"], dtype=float).copy()
        else:
            to_opp = np.asarray(c.punt_outcomes(yl), dtype=float).copy()
            kept = np.zeros(100)
        p_return_td = float(to_opp[0])
        to_opp[0] = 0.0
        kept[0] = 0.0
        for y, w in zip(*self._squeeze(to_opp), strict=True):
            rows.append((_PUNT, _OPP, int(y), float(w), 0, s["punt"]))
        for y, w in zip(*self._squeeze(kept), strict=True):
            rows.append((_PUNT, _US, int(y), float(w), 0, s["punt"]))
        if p_return_td > 0:
            rows.append((_PUNT, _KICK_TO_US, 0, p_return_td * self.p_xp, -7, s["punt"]))
            rows.append((_PUNT, _KICK_TO_US, 0, p_return_td * (1 - self.p_xp), -6, s["punt"]))
        template = self._pack(rows) + (np.array([p_fg]),)
        self._kick_cache[yl] = template
        return template

    @staticmethod
    def _pack(rows: list[tuple[int, int, int, float, int, int]]) -> tuple[np.ndarray, ...]:
        cols = list(zip(*rows, strict=True))
        return (
            np.asarray(cols[0], dtype=np.int64),
            np.asarray(cols[1], dtype=np.int64),
            np.asarray(cols[2], dtype=np.int64),
            np.asarray(cols[3], dtype=float),
            np.asarray(cols[4], dtype=np.int64),
            np.asarray(cols[5], dtype=np.int64),
        )

    def _kickoff_points(self, bucket: int) -> tuple[np.ndarray, np.ndarray]:
        """Kickoff start distribution (no return touchdown), compressed for events whose
        own probability is at most 2**-bucket: the rarer the kickoff, the coarser."""
        if bucket not in self._ko_cache:
            scale = 2.0**-bucket
            ys, ws = compress_distribution(self._ko_field * scale, self.merge_mass, self.max_span)
            self._ko_cache[bucket] = (ys, ws / scale)
        return self._ko_cache[bucket]

    # --------------------------------------------------------------------- evaluation

    def action_values(self, states: pl.DataFrame) -> dict[str, np.ndarray]:
        """Raw numbers behind `evaluate`, as float arrays with NaN where undefined.

        Keys: go_wp, fg_wp, punt_wp, p_convert, p_fg, wp_if_convert, wp_if_fail,
        wp_if_fg_make, wp_if_fg_miss.
        """
        missing = [c for c in wp_mod.STATE_COLUMNS if c not in states.columns]
        if missing:
            raise KeyError(f"evaluate needs columns {missing}")
        n = states.height
        chunk = int(self.config["state_chunk_rows"])
        self.last_stats = {"states": n, "next_states": 0, "unique_next_states": 0}
        eras = self._kickoff_eras(states)
        if eras is None or len(set(eras.tolist())) <= 1:
            self._set_kickoff(None if eras is None or n == 0 else eras[0])
            parts = [self._values_chunk(states.slice(i, chunk))
                     for i in range(0, max(n, 1), chunk)]  # fmt: skip
            out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            self._set_kickoff(None)
            return out
        out: dict[str, np.ndarray] = {}
        for era in sorted(set(eras.tolist())):
            idx = np.nonzero(eras == era)[0]
            self._set_kickoff(era)
            sub = states[idx]
            parts = [self._values_chunk(sub.slice(i, chunk)) for i in range(0, sub.height, chunk)]
            for k in parts[0]:
                out.setdefault(k, np.full(n, np.nan))[idx] = np.concatenate([p[k] for p in parts])
        self._set_kickoff(None)
        return out

    def _values_chunk(self, states: pl.DataFrame) -> dict[str, np.ndarray]:
        n = states.height
        col = lambda name: states[name].cast(pl.Float64).fill_null(np.nan).to_numpy()  # noqa: E731
        ok = wp_mod.scoreable(states) & (col("qtr") <= 4) if n else np.zeros(0, dtype=bool)
        yl_f, togo_f = col("yardline_100"), col("ydstogo")
        ok &= (yl_f >= 1) & (yl_f <= 99) & (togo_f >= 1) & (col("half_seconds_remaining") >= 0)
        nan = np.full(n, np.nan)
        names = ("go_wp", "fg_wp", "punt_wp", "p_convert", "p_fg", "wp_if_convert",
                 "wp_if_fail", "wp_if_fg_make", "wp_if_fg_miss")  # fmt: skip
        result = {k: nan.copy() for k in names}
        rows = np.nonzero(ok)[0]
        if rows.size == 0:
            return result

        yl = yl_f[rows].astype(np.int64)
        togo = np.minimum(np.minimum(togo_f[rows], yl), 40).astype(np.int64)
        gsr = np.rint(col("game_seconds_remaining")[rows]).astype(np.int64)
        half_left = np.rint(col("half_seconds_remaining")[rows]).astype(np.int64)
        first_half = col("qtr")[rows] <= 2
        diff = np.rint(col("score_differential")[rows]).astype(np.int64)
        pos_to = np.clip(col("posteam_timeouts_remaining")[rows], 0, 3).astype(np.int64)
        def_to = np.clip(col("defteam_timeouts_remaining")[rows], 0, 3).astype(np.int64)
        spread = col("posteam_spread")[rows]
        r2h = (col("receive_2h_ko")[rows] > 0).astype(np.int64)
        home = (col("is_home")[rows] > 0).astype(np.int64)
        m = rows.size

        # ---- level 1: events of each action, from cached per-situation templates
        e_row, e_slot, e_kind, e_y, e_w, e_pts, e_sec = self._events(yl, togo)
        p_convert = np.array([self._go_cache[(a, b)][6][0] for a, b in zip(yl, togo, strict=True)])
        p_fg = np.array([self._kick_cache[a][6][0] for a in yl])

        acc = np.zeros((m, _N_SLOTS))
        leaves: list[tuple[np.ndarray, ...]] = []  # row, slot, weight, owner, y, time, diff, reset

        new_diff = diff[e_row] + e_pts
        over = e_sec >= half_left[e_row]
        self._terminal(acc, e_row, e_slot, e_w, new_diff, over & ~first_half[e_row])
        to_half = over & first_half[e_row]
        live = ~over
        field = live & (e_kind <= _OPP)
        leaves.append((e_row[field], e_slot[field], e_w[field], e_kind[field], e_y[field],
                       (gsr[e_row] - e_sec)[field], new_diff[field],
                       np.zeros(int(field.sum()), dtype=np.int64)))  # fmt: skip

        # ---- level 2: kickoffs (after a score, or to open the second half)
        kick = live & (e_kind >= _KICK_TO_OPP)
        k_recv = np.concatenate([
            np.where(e_kind[kick] == _KICK_TO_OPP, _OPP, _US),
            np.where(r2h[e_row[to_half]] == 1, _US, _OPP),
        ])  # fmt: skip
        k_row = np.concatenate([e_row[kick], e_row[to_half]])
        k_slot = np.concatenate([e_slot[kick], e_slot[to_half]])
        k_w = np.concatenate([e_w[kick], e_w[to_half]])
        k_diff = np.concatenate([new_diff[kick], new_diff[to_half]])
        k_time = np.concatenate([(gsr[e_row] - e_sec)[kick], np.full(int(to_half.sum()), 1800)])
        k_left = np.concatenate([(half_left[e_row] - e_sec)[kick],
                                 np.full(int(to_half.sum()), 1800)])  # fmt: skip
        k_reset = np.concatenate([np.zeros(int(kick.sum()), dtype=np.int64),
                                  np.ones(int(to_half.sum()), dtype=np.int64)])  # fmt: skip
        s_ko = self.seconds["kickoff"]
        for _ in range(2):  # a kickoff that runs out the first half leads to one more kickoff
            if k_row.size == 0:
                break
            over = s_ko >= k_left
            k_first = (k_time > 1800) & (k_reset == 0)
            self._terminal(acc, k_row, k_slot, k_w, k_diff, over & ~k_first)
            again = over & k_first
            go_on = ~over
            leaves.extend(self._expand_kickoffs(
                k_row[go_on], k_slot[go_on], k_w[go_on], k_recv[go_on], k_time[go_on] - s_ko,
                k_diff[go_on], k_reset[go_on],
            ))  # fmt: skip
            k_recv = np.where(r2h[k_row[again]] == 1, _US, _OPP)
            k_row, k_slot, k_w, k_diff = k_row[again], k_slot[again], k_w[again], k_diff[again]
            k_time = np.full(k_row.size, 1800)
            k_left = np.full(k_row.size, 1800)
            k_reset = np.ones(k_row.size, dtype=np.int64)

        # ---- score every distinct next state once
        l_row, l_slot, l_w, l_owner, l_y, l_time, l_diff, l_reset = (
            np.concatenate(x) for x in zip(*leaves, strict=True)
        )
        if l_row.size:
            value = self._leaf_values(
                l_row, l_owner, l_y, l_time, l_diff, l_reset, pos_to, def_to, spread, r2h, home
            )
            np.add.at(acc, (l_row, l_slot), l_w * value)

        go = acc[:, _CONVERT] + acc[:, _FAIL]
        fg = acc[:, _FG_MAKE] + acc[:, _FG_MISS]
        with np.errstate(invalid="ignore", divide="ignore"):
            pieces = {
                "go_wp": go,
                "fg_wp": np.where(p_fg > 0, fg, np.nan),
                "punt_wp": np.where(yl >= int(self.config["punt_min_yardline_100"]),
                                    acc[:, _PUNT], np.nan),
                "p_convert": p_convert,
                "p_fg": p_fg,
                "wp_if_convert": np.where(p_convert > 0, acc[:, _CONVERT] / p_convert, np.nan),
                "wp_if_fail": np.where(p_convert < 1, acc[:, _FAIL] / (1 - p_convert), np.nan),
                "wp_if_fg_make": np.where(p_fg > 0, acc[:, _FG_MAKE] / p_fg, np.nan),
                "wp_if_fg_miss": np.where(p_fg < 1, acc[:, _FG_MISS] / (1 - p_fg), np.nan),
            }  # fmt: skip
        for k, v in pieces.items():
            result[k][rows] = v
        return result

    def _events(self, yl: np.ndarray, togo: np.ndarray) -> tuple[np.ndarray, ...]:
        """Expand each row into the events of its three actions."""
        pair_code = yl * 64 + togo
        go_codes, go_inv = np.unique(pair_code, return_inverse=True)
        kick_codes, kick_inv = np.unique(yl, return_inverse=True)
        out = []
        for templates, inv in (
            ([self._go_template(int(c) // 64, int(c) % 64) for c in go_codes], go_inv),
            ([self._kick_template(int(c)) for c in kick_codes], kick_inv),
        ):
            counts = np.array([t[0].size for t in templates])
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            flat = [np.concatenate([t[i] for t in templates]) for i in range(6)]
            reps = counts[inv]
            row = np.repeat(np.arange(yl.size), reps)
            first = np.cumsum(reps) - reps
            idx = np.repeat(starts[inv] - first, reps) + np.arange(int(reps.sum()))
            out.append((row, *[f[idx] for f in flat]))
        return tuple(np.concatenate([a, b]) for a, b in zip(*out, strict=True))

    def _terminal(
        self,
        acc: np.ndarray,
        row: np.ndarray,
        slot: np.ndarray,
        w: np.ndarray,
        diff: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """The game is over: add 1 for a lead, 0 for a deficit, `tie_value` for a tie."""
        if mask.any():
            d = diff[mask]
            value = np.where(d > 0, 1.0, np.where(d < 0, 0.0, self.tie_value))
            np.add.at(acc, (row[mask], slot[mask]), w[mask] * value)

    def _expand_kickoffs(
        self,
        row: np.ndarray,
        slot: np.ndarray,
        w: np.ndarray,
        recv: np.ndarray,
        time: np.ndarray,
        diff: np.ndarray,
        reset: np.ndarray,
    ) -> list[tuple[np.ndarray, ...]]:
        """Turn kickoffs into first-and-10 states for the receiving team.

        The rare return touchdown (index 0 of `kickoff_start`) is not followed through
        another kickoff: the receiving team gets 7 points and the other team then starts
        at the mean kickoff starting spot.
        """
        out: list[tuple[np.ndarray, ...]] = []
        if row.size == 0:
            return out
        if self._ko_return_td > 0:
            sign = np.where(recv == _US, 7, -7)
            start = np.full(row.size, self._ko_mean_start)
            out.append((row, slot, w * self._ko_return_td, 1 - recv, start, time, diff + sign,
                        reset))  # fmt: skip
        with np.errstate(divide="ignore"):
            bucket = np.clip(np.floor(-np.log2(np.maximum(w, 1e-12))), 0, 24).astype(int)
        for b in np.unique(bucket):
            sel = np.nonzero(bucket == b)[0]
            ys, ws = self._kickoff_points(int(b))
            rep = lambda a: np.repeat(a[sel], ys.size)  # noqa: E731, B023
            out.append((rep(row), rep(slot), rep(w) * np.tile(ws, sel.size), rep(recv),
                        np.tile(ys, sel.size), rep(time), rep(diff), rep(reset)))  # fmt: skip
        return out

    def _leaf_values(
        self,
        row: np.ndarray,
        owner: np.ndarray,
        y: np.ndarray,
        time: np.ndarray,
        diff: np.ndarray,
        reset: np.ndarray,
        pos_to: np.ndarray,
        def_to: np.ndarray,
        spread: np.ndarray,
        r2h: np.ndarray,
        home: np.ndarray,
    ) -> np.ndarray:
        """Value to us of each next state: WP if we have the ball, 1 - WP after the flip."""
        theirs = owner == _OPP
        # The perspective flip: everything is restated for the team that has the ball.
        o_diff = np.clip(np.where(theirs, -diff, diff), -127, 127)
        o_pos = np.where(reset == 1, 3, np.where(theirs, def_to[row], pos_to[row]))
        o_def = np.where(reset == 1, 3, np.where(theirs, pos_to[row], def_to[row]))
        o_home = np.where(theirs, 1 - home[row], home[row])
        o_r2h = np.where(time > 1800, np.where(theirs, 1 - r2h[row], r2h[row]), 0)
        spread_values, spread_inv = np.unique(np.concatenate([spread, -spread]),
                                              return_inverse=True)  # fmt: skip
        o_spread = np.where(theirs, spread_inv[spread.size + row], spread_inv[row])

        key = y
        for part, bits in ((time, 12), (o_diff + 128, 8), (o_pos, 2), (o_def, 2), (o_home, 1),
                           (o_r2h, 1)):  # fmt: skip
            key = (key << bits) | part
        key = (key << 22) | o_spread
        uniq, inv = np.unique(key, return_inverse=True)
        self.last_stats["next_states"] += int(key.size)
        self.last_stats["unique_next_states"] += int(uniq.size)

        u_spread = spread_values[uniq & ((1 << 22) - 1)]
        rest = uniq >> 22
        u_r2h, rest = rest & 1, rest >> 1
        u_home, rest = rest & 1, rest >> 1
        u_def, rest = rest & 3, rest >> 2
        u_pos, rest = rest & 3, rest >> 2
        u_diff, rest = (rest & 255) - 128, rest >> 8
        u_time, u_y = rest & 4095, rest >> 12

        chunk = int(self.config["predict_chunk_rows"])
        p = np.empty(uniq.size)
        for i in range(0, uniq.size, chunk):
            s = slice(i, i + chunk)
            frame = wp_mod.state_frame(
                down=1, ydstogo=np.minimum(10, u_y[s]), yardline_100=u_y[s],
                game_seconds_remaining=u_time[s], score_differential=u_diff[s],
                posteam_timeouts_remaining=u_pos[s], defteam_timeouts_remaining=u_def[s],
                posteam_spread=u_spread[s], receive_2h_ko=u_r2h[s], is_home=u_home[s],
            )  # fmt: skip
            p[i : i + chunk] = np.asarray(self.wp_model.predict(frame), dtype=float)
        value = p[inv]
        return np.where(theirs, 1.0 - value, value)

    def assert_can_score(self, states: pl.DataFrame, what: str) -> None:
        """Raise `LeakageError` if either model was fitted on a season of `states`.

        Hypothetical states without a `season` column have nothing to check and pass. A
        win-probability model without a stamp is not checked here (its own module owns
        that guard); components without a stamp always fail, since their training seasons
        are unknown.
        """
        if "season" not in states.columns or states.height == 0:
            return
        splits.assert_can_score(self.stamp, states, f"{what} (play-outcome models)")
        wp_stamp = getattr(self.wp_model, "stamp", None)
        if wp_stamp is not None:
            splits.assert_can_score(wp_stamp, states, f"{what} (win-probability model)")

    def evaluate_out_of_sample(self, states: pl.DataFrame, what: str) -> pl.DataFrame:
        """`evaluate` for real plays reported as out-of-sample: checks the stamps first."""
        self.assert_can_score(states, what)
        return self.evaluate(states)

    def evaluate(
        self, states: pl.DataFrame, out_of_sample: bool = False, what: str = "decision engine"
    ) -> pl.DataFrame:
        """Add the decision columns to a frame of fourth-down states.

        With `out_of_sample=True` (use it whenever real plays are scored and reported as
        out-of-sample) the models' fit stamps are checked against the rows' seasons first.

        `states` needs the columns in `wp.STATE_COLUMNS` (real plays have them;
        `wp.state_frame(down=4, ...)` builds hypothetical ones). Everything is from the
        point of view of the team with the ball. Added columns:

          go_wp, fg_wp, punt_wp   win probability of each action
          best_action, best_wp    the action with the highest win probability
          margin                  best minus second best
          go_gain                 go_wp minus the better of fg_wp and punt_wp
          p_convert, p_fg         conversion and field-goal probabilities used
          wp_if_convert, wp_if_fail, wp_if_fg_make, wp_if_fg_miss

        `fg_wp` is null where the kick is out of range (P(make) = 0) and `punt_wp` is
        null inside `punt_min_yardline_100`, where almost no punts exist to learn from.
        Overtime states and states with missing inputs get nulls throughout.
        """
        if out_of_sample:
            self.assert_can_score(states, what)
        v = self.action_values(states)
        table = np.column_stack([v["go_wp"], v["fg_wp"], v["punt_wp"]])
        best, best_wp, margin, gain = summarize_actions(table)
        labels = np.array([*ACTIONS, None], dtype=object)[np.where(best < 0, 3, best)]
        new = {k: pl.Series(k, x).fill_nan(None) for k, x in v.items()}
        new["best_action"] = pl.Series("best_action", labels.tolist(), dtype=pl.String)
        new["best_wp"] = pl.Series("best_wp", best_wp).fill_nan(None)
        new["margin"] = pl.Series("margin", margin).fill_nan(None)
        new["go_gain"] = pl.Series("go_gain", gain).fill_nan(None)
        order = ["go_wp", "fg_wp", "punt_wp", "best_action", "best_wp", "margin", "go_gain",
                 "p_convert", "p_fg", "wp_if_convert", "wp_if_fail", "wp_if_fg_make",
                 "wp_if_fg_miss"]  # fmt: skip
        return states.with_columns([new[k] for k in order])


def summarize_actions(
    table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """From an (n, 3) array of go, field goal and punt values (NaN = not available):

    best action index (-1 if the row has no go value), best value, margin over the
    second best, and the gain of going over the better kick (go minus max(fg, punt)).
    """
    filled = np.where(np.isnan(table), -np.inf, table)
    valid = ~np.isnan(table[:, 0])
    order = np.sort(filled, axis=1)
    best = np.where(valid, np.argmax(filled, axis=1), -1)
    best_wp = np.where(valid, order[:, 2], np.nan)
    second = order[:, 1]
    kick = np.maximum(filled[:, 1], filled[:, 2])
    with np.errstate(invalid="ignore"):
        margin = np.where(valid & np.isfinite(second), order[:, 2] - second, np.nan)
        gain = np.where(valid & np.isfinite(kick), table[:, 0] - kick, np.nan)
    return best, best_wp, margin, gain


# ----------------------------------------------------------------- headline models


def fit_evaluation_components(
    plays: pl.DataFrame,
    fit_seasons: tuple[int, int] | list[int],
    kickoff_seasons: list[int],
    config: dict | None = None,
) -> ComponentModels:
    """Component models for judging later seasons: fitted on `fit_seasons` only.

    `plays` must already have the next-state columns (`components.add_next_state`) if
    it is a resample of games. The kickoff start tables are the one exception to the
    season rule: the tables of the rule eras of `kickoff_seasons` are fitted on those
    seasons' own kickoffs (logged as `components.KICKOFF_RULE_REASON`), because they
    describe the kickoff rule in force, not a skill to be learned, and no earlier season
    was played under that rule.
    """
    lo, hi = int(fit_seasons[0]), int(fit_seasons[1])
    models = ComponentModels(config).fit(plays.filter(pl.col("season").is_between(lo, hi)))
    return set_kickoff(models, plays, kickoff_seasons)


def set_kickoff(
    models: ComponentModels, plays: pl.DataFrame, kickoff_seasons: list[int]
) -> ComponentModels:
    """Add the kickoff tables of the rule eras of `kickoff_seasons`, each fitted on its own
    seasons' kickoffs (the logged rule-constant exception). Only kickoff rows are read.
    The bundle's default table becomes the latest era's."""
    from fourth_and_rl.components import add_next_state

    seasons = [int(s) for s in kickoff_seasons]
    # The next-state lookup needs whole games, so add it before selecting seasons.
    kick = add_next_state(plays).filter(pl.col("season").is_in(seasons))
    models.add_kickoff_rule_tables(
        kick, seasons, allow_test_seasons=True, reason=KICKOFF_RULE_REASON
    )
    return models


def load_engine(
    plays: pl.DataFrame | None = None,
    config: dict | None = None,
    refit: bool = False,
) -> DecisionEngine:
    """The headline engine used by every M2 and M3 result.

    Win probability: the saved M1 headline model (`models.wp_path`). Components: the
    development bundle (`components.DEV_MODEL_PATH`, fitted on the development seasons,
    with the kickoff tables of every rule era). The cached bundle is reused only if its
    stamp matches the current components config and the configured fit seasons
    (`components.dev_bundle_is_current`); otherwise, or with `refit`, it is refitted from
    the development file (plus the test seasons' kickoff rows for the rule tables) and
    saved. `plays`, if given, must be the full state table and saves reading it again;
    only its development seasons and the kickoff rows of the test seasons are fitted.

    The engine is checked to be able to score the test seasons out of sample.
    """
    from fourth_and_rl import components as cm

    cfg = config or load_config()
    m = cfg["models"]
    wp_model = wp_mod.WPModel.load(ROOT / m["wp_path"])
    cache = ROOT / m.get("components_cache", str(cm.DEV_MODEL_PATH.relative_to(ROOT)))
    lo, hi = (int(x) for x in m["components_fit_seasons"])
    if set(range(lo, hi + 1)) != set(splits.dev_seasons()):
        raise ValueError(
            f"components_fit_seasons {m['components_fit_seasons']} must be the development "
            f"seasons {splits.dev_seasons()}: the engine uses the development bundle"
        )
    comps = ComponentModels.load(cache) if cache.exists() and not refit else None
    if comps is None or not cm.dev_bundle_is_current(comps):
        if plays is None:
            dev, test = None, None
        else:
            dev = plays.filter(pl.col("season").is_in(list(splits.dev_seasons())))
            # Whole test-season games (the next-state lookup needs them); only their
            # kickoff rows are fitted, as the rule tables.
            test = plays.filter(pl.col("season").is_in(list(splits.test_seasons())))
        comps = cm.fit_dev_bundle(dev_plays=dev, test_plays=test)
        comps.save(cache)
    engine = DecisionEngine(wp_model, comps, cfg.get("engine"))
    engine.assert_can_score(
        pl.DataFrame({"season": list(splits.test_seasons())}), "load_engine: the headline engine"
    )
    return engine


def chart_states(
    yardlines: np.ndarray | list[int],
    distances: np.ndarray | list[int],
    situation: dict[str, Any],
) -> pl.DataFrame:
    """Fourth-down states for every (yard line, distance) cell of one game situation.

    `situation` holds keyword arguments of `wp.state_frame` (clock, score, spread,
    timeouts...). Cells where the distance is longer than the distance to the goal line
    are left out, since they cannot occur.
    """
    yl, d = np.meshgrid(np.asarray(yardlines), np.asarray(distances), indexing="ij")
    yl, d = yl.ravel(), d.ravel()
    keep = d <= yl
    return wp_mod.state_frame(down=4, ydstogo=d[keep], yardline_100=yl[keep], **situation)


# --------------------------------------------------------------------------- drawing


def to_grid(
    yardline_100: np.ndarray,
    ydstogo: np.ndarray,
    values: np.ndarray,
    n_distances: int = 15,
    fill: Any = np.nan,
) -> np.ndarray:
    """Arrange per-state values as a grid [distance - 1, yardline_100 - 1] (99 columns)."""
    values = np.asarray(values)
    grid = np.full((n_distances, 99), fill, dtype=values.dtype if values.dtype != bool else float)
    yl = np.asarray(yardline_100).astype(int)
    d = np.asarray(ydstogo).astype(int)
    keep = (d >= 1) & (d <= n_distances)
    grid[d[keep] - 1, yl[keep] - 1] = values[keep]
    return grid


def _blend(color: str, toward: str, share: float) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    a, b = np.array(to_rgb(color)), np.array(to_rgb(toward))
    return tuple((1 - share) * a + share * b)


LEAN_BLEND = 0.58  # how far a "lean" cell is washed out toward the background


def chart_colors() -> dict[tuple[str, str], tuple[float, float, float]]:
    """Cell color for every (action, confidence category) pair, from the shared style.

    Confident cells use the action's color, lean cells the same color washed out, and
    uncertain cells the neutral "too close to call" color whatever the action.
    """
    from matplotlib.colors import to_rgb

    from fourth_and_rl import style

    out = {}
    for action in ACTIONS:
        out[(action, "confident")] = to_rgb(style.ACTION_COLORS[action])
        out[(action, "lean")] = _blend(style.ACTION_COLORS[action], style.SURFACE, LEAN_BLEND)
        out[(action, "uncertain")] = to_rgb(style.ACTION_COLORS["toss_up"])
    return out


def chart_image(best: np.ndarray, category: np.ndarray | None = None) -> np.ndarray:
    """RGBA image of a decision chart.

    `best` is a grid of action indices (0 go, 1 field goal, 2 punt, negative or NaN =
    no cell) laid out [distance - 1, yardline_100 - 1]. `category` is a grid of
    "confident" / "lean" / "uncertain" (None draws every cell as confident). The image
    is flipped left to right so that x runs from the offense's own goal line to the
    opponent's.
    """
    colors = chart_colors()
    best = np.asarray(best, dtype=float)
    img = np.zeros((*best.shape, 4))
    for i, action in enumerate(ACTIONS):
        for cat in ("confident", "lean", "uncertain"):
            mask = best == i
            if category is not None:
                mask &= category == cat
            elif cat != "confident":
                continue
            img[mask, :3] = colors[(action, cat)]
            img[mask, 3] = 1.0
    return img[:, ::-1]


def draw_chart(ax: Any, best: np.ndarray, category: np.ndarray | None = None) -> None:
    """Draw a decision chart: x = yards from the offense's own goal line, y = yards to go."""
    from fourth_and_rl import style

    n_d = best.shape[0]
    ax.imshow(chart_image(best, category), origin="lower", aspect="auto",
              extent=(0.5, 99.5, 0.5, n_d + 0.5), interpolation="nearest", zorder=2)  # fmt: skip
    ax.set_xlim(0.5, 99.5)
    ax.set_ylim(0.5, n_d + 0.5)
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
    ax.set_xticklabels(["Own 10", "20", "30", "40", "50", "40", "30", "20", "Opp 10"])
    ax.set_yticks([1, 3, 5, 7, 9, 11, 13, 15][: (n_d + 1) // 2])
    ax.grid(False)
    for x in range(10, 100, 10):
        ax.axvline(x, color=style.SURFACE, linewidth=0.6, alpha=0.55, zorder=3)
    for y in np.arange(1.5, n_d, 1.0):
        ax.axhline(y, color=style.SURFACE, linewidth=0.5, alpha=0.6, zorder=3)


def chart_legend_handles(with_confidence: bool, actions: tuple[str, ...] = ACTIONS) -> list:
    """Legend patches for a decision chart, in a fixed order."""
    from matplotlib.patches import Patch

    from fourth_and_rl import style

    colors = chart_colors()
    handles = [
        Patch(facecolor=colors[(a, "confident")], label=style.ACTION_LABELS[a]) for a in actions
    ]
    if with_confidence:
        handles += [
            Patch(facecolor=colors[(a, "lean")], label=f"{style.ACTION_LABELS[a]} (lean)")
            for a in actions
        ]
        handles.append(
            Patch(facecolor=style.ACTION_COLORS["toss_up"], label=style.ACTION_LABELS["toss_up"])
        )
    return handles
