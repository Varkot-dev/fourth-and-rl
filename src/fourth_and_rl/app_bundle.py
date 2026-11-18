"""M7: the precomputed app bundle, its schema, and lookups used by the app, page and notebook.

The fitted models (data/models/) and the solved game (data/mdp/, about 2 GB) are not in
the repository, so the interactive pieces read a small precomputed bundle instead, written
by `scripts/m7_build_app_bundle.py` to `app/data/`:

  bundle_meta.json   axes, encodings, model stamps, build date, coverage and checks
  one_step.npz       the one-step engine (M2) with game-level bootstrap summaries (M3)
  exact.npz          the exactly solved game (M4): go, field goal and punt values
  questions.json     maps copied from saved question results (question 1 and question 5)

Arrays are dense grids indexed [context, score, distance - 1, yardline_100 - 1]. A one-step
context is (game seconds left, opponent timeouts, spread); an exact context is (game
seconds left, opponent timeouts), since the exact game has identical teams and no spread.
Cells with distance > yard line cannot occur and hold the "not available" code.

The .npz files use LZMA zip members and store multi-byte arrays as exact int32
differences along the yard-line axis (`_write_npz`); `load_bundle` undoes both.

Encodings (configs/app.yaml, `encoding`): win probabilities are uint16 in units of
1 / wp_scale with 65535 meaning "not available" (a field goal out of range, a punt inside
the opponent's 31 in the one-step engine); gains are int16 in units of 1 / gain_scale with
-32768 meaning "not available"; bootstrap counts are uint8 (replicates preferring each
action). Nothing is interpolated: lookups snap to grid values and say so.

This module needs only NumPy and the standard library, so the app and the notebook can
load a bundle without the modelling stack.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fourth_and_rl.paths import CONFIGS, ROOT

SCHEMA_VERSION = 1
ACTIONS: tuple[str, str, str] = ("go", "field_goal", "punt")
WP_NA = np.uint16(65535)
GAIN_NA = np.int16(-32768)
META_FILE, ONE_STEP_FILE, EXACT_FILE, QUESTIONS_FILE = (
    "bundle_meta.json", "one_step.npz", "exact.npz", "questions.json")  # fmt: skip

# Arrays each file must hold, with their dtype.
ONE_STEP_ARRAYS: dict[str, type] = {
    "go": np.uint16, "fg": np.uint16, "punt": np.uint16,
    "go_lo": np.uint16, "go_hi": np.uint16, "fg_lo": np.uint16, "fg_hi": np.uint16,
    "punt_lo": np.uint16, "punt_hi": np.uint16,
    "gain_lo": np.int16, "gain_hi": np.int16,
    "n_go": np.uint8, "n_fg": np.uint8, "n_punt": np.uint8,
    "p_convert": np.uint16, "p_fg": np.uint16,
}  # fmt: skip
EXACT_ARRAYS: dict[str, type] = {"go": np.uint16, "fg": np.uint16, "punt": np.uint16}
CATEGORY_LABELS = {"confident": "Confident", "lean": "Lean", "uncertain": "Too close to call"}


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIGS / "app.yaml") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------------------ the grid


@dataclass(frozen=True)
class Context:
    """One game situation apart from field position, distance and score."""

    seconds: int
    opp_timeouts: int
    spread: float = 0.0


def contexts(grid: dict) -> list[Context]:
    """One-step contexts in bundle order: clocks as listed, then opponent timeouts, then
    spreads. Opponent timeouts vary only at or below `endgame_seconds` (3 before), spreads
    other than 0 only at or above `spread_min_seconds`."""
    out = []
    for clock in grid["clocks"]:
        sec = int(clock["seconds"])
        tos = grid["opp_timeouts"] if sec <= grid["endgame_seconds"] else [3]
        spreads = grid["spreads"] if sec >= grid["spread_min_seconds"] else [0]
        for to in tos:
            for sp in spreads:
                out.append(Context(sec, int(to), float(sp)))
    return out


def exact_contexts(grid: dict) -> list[Context]:
    """Exact-game contexts: the one-step contexts without the spread (identical teams)."""
    seen: list[Context] = []
    for c in contexts(grid):
        e = Context(c.seconds, c.opp_timeouts, 0.0)
        if e not in seen:
            seen.append(e)
    return seen


def axis(lo_hi: list[int]) -> list[int]:
    return list(range(int(lo_hi[0]), int(lo_hi[1]) + 1))


def valid_mask(distances: list[int], yardlines: list[int]) -> np.ndarray:
    """(n_dist, n_yl) True where the cell can occur (distance <= yard line)."""
    return np.asarray(distances)[:, None] <= np.asarray(yardlines)[None, :]


# ----------------------------------------------------------------------------- encodings


def encode_wp(x: np.ndarray, scale: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.round(np.clip(x, 0.0, 1.0) * scale)
    return np.where(np.isfinite(x), out, float(WP_NA)).astype(np.uint16)


def decode_wp(x: np.ndarray, scale: int) -> np.ndarray:
    x = np.asarray(x)
    return np.where(x == WP_NA, np.nan, x.astype(float) / scale)


def encode_gain(x: np.ndarray, scale: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lim = (np.iinfo(np.int16).max) / scale
    out = np.round(np.clip(x, -lim, lim) * scale)
    return np.where(np.isfinite(x), out, float(GAIN_NA)).astype(np.int16)


def decode_gain(x: np.ndarray, scale: int) -> np.ndarray:
    x = np.asarray(x)
    return np.where(x == GAIN_NA, np.nan, x.astype(float) / scale)


def classify(boot_pct: float, confident: float, lean: float) -> str | None:
    """The M3 rule (bootstrap.classify) for one value."""
    if not np.isfinite(boot_pct):
        return None
    return "confident" if boot_pct >= confident else "lean" if boot_pct >= lean else "uncertain"


def best_of(values: np.ndarray) -> np.ndarray:
    """Index of the best available action along the last axis (-1 if go is missing, as in
    decision.summarize_actions)."""
    v = np.asarray(values, dtype=float)
    filled = np.where(np.isnan(v), -np.inf, v)
    return np.where(np.isnan(v[..., 0]), -1, np.argmax(filled, axis=-1))


# -------------------------------------------------------------------------------- bundle


@dataclass
class Bundle:
    meta: dict[str, Any]
    one_step: dict[str, np.ndarray]
    exact: dict[str, np.ndarray]
    questions: dict[str, Any] = field(default_factory=dict)

    # ---- axes
    @property
    def contexts(self) -> list[Context]:
        return [Context(**c) for c in self.meta["axes"]["contexts"]]

    @property
    def exact_contexts(self) -> list[Context]:
        return [Context(**c) for c in self.meta["axes"]["exact_contexts"]]

    @property
    def scores(self) -> list[int]:
        return self.meta["axes"]["scores"]

    @property
    def distances(self) -> list[int]:
        return self.meta["axes"]["distances"]

    @property
    def yardlines(self) -> list[int]:
        return self.meta["axes"]["yardlines"]

    @property
    def clocks(self) -> list[dict[str, Any]]:
        return self.meta["axes"]["clocks"]

    @property
    def wp_scale(self) -> int:
        return int(self.meta["encoding"]["wp_scale"])

    @property
    def gain_scale(self) -> int:
        return int(self.meta["encoding"]["gain_scale"])

    @property
    def n_boot(self) -> int:
        return int(self.meta["bootstrap"]["n_boot"])

    def clock_label(self, seconds: int) -> str:
        return next(c["label"] for c in self.clocks if int(c["seconds"]) == int(seconds))

    def options(self, seconds: int) -> dict[str, list]:
        """Opponent timeouts and spreads available at a clock."""
        cs = [c for c in self.contexts if c.seconds == int(seconds)]
        return {
            "opp_timeouts": sorted({c.opp_timeouts for c in cs}),
            "spreads": sorted({c.spread for c in cs}),
        }

    def snap(self, seconds: int, opp_timeouts: int, spread: float) -> tuple[Context, list[str]]:
        """The grid context used for a request, and notes on anything that was snapped."""
        notes = []
        secs = [int(c["seconds"]) for c in self.clocks]
        sec = min(secs, key=lambda s: abs(s - int(seconds)))
        if sec != int(seconds):
            notes.append(f"clock snapped to {self.clock_label(sec)}")
        opts = self.options(sec)
        to = min(opts["opp_timeouts"], key=lambda t: abs(t - int(opp_timeouts)))
        if to != int(opp_timeouts):
            notes.append(f"opponent timeouts fixed at {to} at this clock")
        sp = min(opts["spreads"], key=lambda s: abs(s - float(spread)))
        if sp != float(spread):
            notes.append(f"spread fixed at {sp:+g} at this clock")
        return Context(sec, to, sp), notes

    def _ctx_index(self, ctx: Context, exact: bool) -> int:
        cs = self.exact_contexts if exact else self.contexts
        key = Context(ctx.seconds, ctx.opp_timeouts, 0.0) if exact else ctx
        return cs.index(key)

    def _score_index(self, score: int) -> int:
        if score not in self.scores:
            raise KeyError(f"score {score} is not on the grid {self.scores[0]}..{self.scores[-1]}")
        return self.scores.index(score)

    # ---- one cell
    def lookup(
        self,
        seconds: int,
        score: int,
        yardline_100: int,
        ydstogo: int,
        opp_timeouts: int = 3,
        spread: float = 0.0,
    ) -> dict[str, Any]:
        """Everything the bundle holds about one fourth-down situation."""
        ctx, notes = self.snap(seconds, opp_timeouts, spread)
        if not 1 <= ydstogo <= yardline_100:
            raise ValueError("yards to go must be between 1 and the yard line")
        i, j = self._ctx_index(ctx, False), self._score_index(int(score))
        d, y = self.distances.index(int(ydstogo)), self.yardlines.index(int(yardline_100))
        o = {k: a[i, j, d, y] for k, a in self.one_step.items()}
        ws, gs = self.wp_scale, self.gain_scale
        wp = {a: float(decode_wp(o[k], ws)) for a, k in zip(ACTIONS, ("go", "fg", "punt"),
                                                             strict=True)}  # fmt: skip
        lo = {a: float(decode_wp(o[f"{k}_lo"], ws)) for a, k in
              zip(ACTIONS, ("go", "fg", "punt"), strict=True)}  # fmt: skip
        hi = {a: float(decode_wp(o[f"{k}_hi"], ws)) for a, k in
              zip(ACTIONS, ("go", "fg", "punt"), strict=True)}  # fmt: skip
        vals = np.array([wp[a] for a in ACTIONS])
        b = int(best_of(vals))
        counts = {"go": int(o["n_go"]), "field_goal": int(o["n_fg"]), "punt": int(o["n_punt"])}
        th = self.meta["bootstrap"]["thresholds"]
        boot_pct = counts[ACTIONS[b]] / self.n_boot if b >= 0 else float("nan")
        kick = np.nanmax(vals[1:]) if np.isfinite(vals[1:]).any() else np.nan
        one = {
            "wp": wp, "wp_lo": lo, "wp_hi": hi,
            "best": ACTIONS[b] if b >= 0 else None,
            "go_gain": float(vals[0] - kick) if np.isfinite(kick) else float("nan"),
            "gain_lo": float(decode_gain(o["gain_lo"], gs)),
            "gain_hi": float(decode_gain(o["gain_hi"], gs)),
            "boot_counts": counts, "boot_pct": boot_pct,
            "category": classify(boot_pct, th["confident"], th["lean"]),
            "p_convert": float(decode_wp(o["p_convert"], ws)),
            "p_fg": float(decode_wp(o["p_fg"], ws)),
        }  # fmt: skip
        k = self._ctx_index(ctx, True)
        e = {a: float(decode_wp(self.exact[key][k, j, d, y], ws))
             for a, key in zip(ACTIONS, ("go", "fg", "punt"), strict=True)}  # fmt: skip
        ev = np.array([e[a] for a in ACTIONS])
        eb = int(best_of(ev))
        ekick = np.nanmax(ev[1:]) if np.isfinite(ev[1:]).any() else np.nan
        exact = {
            "wp": e, "best": ACTIONS[eb] if eb >= 0 else None,
            "go_gain": float(ev[0] - ekick) if np.isfinite(ekick) else float("nan"),
            "distance_binned": int(ydstogo) > int(self.meta["exact"]["dist_exact"]),
        }  # fmt: skip
        return {"context": ctx, "notes": notes, "one_step": one, "exact": exact,
                "clock_label": self.clock_label(ctx.seconds)}  # fmt: skip

    # ---- a whole chart
    def chart(
        self, seconds: int, score: int, opp_timeouts: int = 3, spread: float = 0.0
    ) -> dict[str, np.ndarray]:
        """Decision-chart arrays (n_dist, n_yl) for one situation: one-step best action
        (-1 empty), boot% of it, category code (0 confident, 1 lean, 2 uncertain, -1
        empty), go gain and its interval, and the exact game's best action and go gain."""
        ctx, _ = self.snap(seconds, opp_timeouts, spread)
        i, j = self._ctx_index(ctx, False), self._score_index(int(score))
        ws, gs = self.wp_scale, self.gain_scale
        v = np.stack([decode_wp(self.one_step[k][i, j], ws) for k in ("go", "fg", "punt")], -1)
        best = best_of(v)
        counts = np.stack([self.one_step[k][i, j] for k in ("n_go", "n_fg", "n_punt")], -1)
        pct = np.where(best >= 0, np.take_along_axis(counts, np.maximum(best, 0)[..., None],
                                                     -1)[..., 0] / self.n_boot, np.nan)  # fmt: skip
        th = self.meta["bootstrap"]["thresholds"]
        cat = np.where(best < 0, -1, np.where(pct >= th["confident"], 0,
                                              np.where(pct >= th["lean"], 1, 2)))  # fmt: skip
        with np.errstate(invalid="ignore"):
            kick = np.fmax(v[..., 1], v[..., 2])
            gain = v[..., 0] - kick
        k = self._ctx_index(ctx, True)
        ev = np.stack([decode_wp(self.exact[key][k, j], ws) for key in ("go", "fg", "punt")], -1)
        with np.errstate(invalid="ignore"):
            egain = ev[..., 0] - np.fmax(ev[..., 1], ev[..., 2])
        return {
            "best": best, "boot_pct": pct, "category": cat, "values": v, "go_gain": gain,
            "gain_lo": decode_gain(self.one_step["gain_lo"][i, j], gs),
            "gain_hi": decode_gain(self.one_step["gain_hi"][i, j], gs),
            "exact_best": best_of(ev), "exact_values": ev, "exact_go_gain": egain,
        }  # fmt: skip


def validate(b: Bundle) -> None:
    """Raise ValueError unless the arrays match the schema and the axes in the meta."""
    m = b.meta
    problems = []
    if m.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {m.get('schema_version')} != {SCHEMA_VERSION}")
    ax = m["axes"]
    shape = (len(ax["contexts"]), len(ax["scores"]), len(ax["distances"]), len(ax["yardlines"]))
    eshape = (len(ax["exact_contexts"]), *shape[1:])
    for arrays, spec, want, name in ((b.one_step, ONE_STEP_ARRAYS, shape, "one_step"),
                                     (b.exact, EXACT_ARRAYS, eshape, "exact")):  # fmt: skip
        for key, dtype in spec.items():
            if key not in arrays:
                problems.append(f"{name}.{key} is missing")
            elif arrays[key].shape != want or arrays[key].dtype != np.dtype(dtype):
                problems.append(f"{name}.{key}: {arrays[key].shape} {arrays[key].dtype}, "
                                f"expected {want} {np.dtype(dtype)}")  # fmt: skip
        extra = set(arrays) - set(spec)
        if extra:
            problems.append(f"{name} has unexpected arrays {sorted(extra)}")
    if not problems:
        counts = sum(b.one_step[k].astype(int) for k in ("n_go", "n_fg", "n_punt"))
        ok = valid_mask(ax["distances"], ax["yardlines"])
        if np.any(counts[..., ok] > m["bootstrap"]["n_boot"]):
            problems.append("bootstrap counts exceed n_boot")
        if np.any(b.one_step["go"][..., ~ok] != WP_NA) or np.any(b.exact["go"][..., ~ok] != WP_NA):
            problems.append("impossible cells (distance > yard line) hold values")
    if problems:
        raise ValueError("invalid app bundle: " + "; ".join(problems))


DELTA_SUFFIX = "__delta"


def _write_npz(path: Path, arrays: dict[str, np.ndarray], delta: bool) -> None:
    """An .npz (a zip of .npy files) with LZMA members, which np.load reads as usual.

    With `delta`, each array of more than one byte per value is stored as its exact int32
    differences along the last axis (yard line) under the name `<key>__delta`; `_read_npz`
    undoes it with a cumulative sum. Neighbouring yard lines have close values, so the
    differences are small and compress about twice as well. Sentinel codes survive exactly.
    """
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_LZMA) as zf:
        for key, a in arrays.items():
            name, arr = key, a
            if delta and a.dtype.itemsize > 1:
                name, arr = key + DELTA_SUFFIX, np.diff(a.astype(np.int32), axis=-1, prepend=0)
            with zf.open(f"{name}.npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.ascontiguousarray(arr), allow_pickle=False)
    tmp.replace(path)


def _read_npz(path: Path, spec: dict[str, type]) -> dict[str, np.ndarray]:
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for name in z.files:
            if name.endswith(DELTA_SUFFIX):
                key = name[: -len(DELTA_SUFFIX)]
                dtype = np.dtype(spec.get(key, np.int32))
                out[key] = np.cumsum(z[name], axis=-1, dtype=np.int64).astype(dtype)
            else:
                out[name] = z[name]
    return out


def save_bundle(b: Bundle, directory: Path | str, delta: bool = True) -> dict[str, int]:
    """Write the four files (validated first). Returns their sizes in bytes."""
    validate(b)
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    _write_npz(d / ONE_STEP_FILE, b.one_step, delta)
    _write_npz(d / EXACT_FILE, b.exact, delta)
    (d / QUESTIONS_FILE).write_text(json.dumps(b.questions, separators=(",", ":")) + "\n")
    (d / META_FILE).write_text(json.dumps(b.meta, indent=1, default=_json_default) + "\n")
    return {n: (d / n).stat().st_size for n in (META_FILE, ONE_STEP_FILE, EXACT_FILE,
                                                QUESTIONS_FILE)}  # fmt: skip


def _json_default(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON serializable: {type(x)}")


def default_dir() -> Path:
    return ROOT / load_config()["bundle_dir"]


def load_bundle(directory: Path | str | None = None) -> Bundle:
    """Load and validate a bundle (default: the committed one under app/data/)."""
    d = Path(directory) if directory is not None else default_dir()
    if not (d / META_FILE).exists():
        raise FileNotFoundError(f"no app bundle under {d}; run scripts/m7_build_app_bundle.py")
    meta = json.loads((d / META_FILE).read_text())
    one = _read_npz(d / ONE_STEP_FILE, ONE_STEP_ARRAYS)
    exact = _read_npz(d / EXACT_FILE, EXACT_ARRAYS)
    qpath = d / QUESTIONS_FILE
    questions = json.loads(qpath.read_text()) if qpath.exists() else {}
    b = Bundle(meta, one, exact, questions)
    validate(b)
    return b


# -------------------------------------------------------------------- synthetic bundle


def synthetic_bundle(seed: int = 0, n_boot: int = 11) -> Bundle:
    """A tiny made-up bundle with the real schema, for tests and CI (no data needed).

    Two clocks (an early one with two spreads, an end-game one with two timeout values),
    scores -1..1, distances 1..3, yard lines 1..5. Values are random but coherent:
    go, field goal and punt win probabilities in (0, 1), the field goal missing beyond
    yard line 4 and the punt missing inside yard line 2.
    """
    rng = np.random.default_rng(seed)
    grid = {
        "clocks": [{"seconds": 900, "label": "Q4 15:00"}, {"seconds": 120, "label": "Q4 2:00"}],
        "endgame_seconds": 300, "opp_timeouts": [0, 3], "spreads": [0, 7],
        "spread_min_seconds": 900, "scores": [-1, 1], "distances": [1, 3], "yardlines": [1, 5],
    }  # fmt: skip
    ctx, ectx = contexts(grid), exact_contexts(grid)
    scores, dists, yls = axis(grid["scores"]), axis(grid["distances"]), axis(grid["yardlines"])
    ok = valid_mask(dists, yls)
    shape = (len(ctx), len(scores), len(dists), len(yls))
    scale = 10000

    def wp_array(n_ctx: int) -> dict[str, np.ndarray]:
        base = rng.uniform(0.2, 0.8, (n_ctx, len(scores), len(dists), len(yls), 3))
        base[..., 1][..., np.asarray(yls) > 4] = np.nan  # field goal out of range
        base[..., 2][..., np.asarray(yls) < 2] = np.nan  # no punt from the 1
        base[..., ~ok, :] = np.nan
        return {k: base[..., i] for i, k in enumerate(("go", "fg", "punt"))}

    one_wp = wp_array(len(ctx))
    one: dict[str, np.ndarray] = {k: encode_wp(v, scale) for k, v in one_wp.items()}
    for k, v in one_wp.items():
        one[f"{k}_lo"] = encode_wp(v - 0.03, scale)
        one[f"{k}_hi"] = encode_wp(v + 0.03, scale)
    gain = one_wp["go"] - np.fmax(one_wp["fg"], one_wp["punt"])
    one["gain_lo"] = encode_gain(gain - 0.02, scale)
    one["gain_hi"] = encode_gain(gain + 0.02, scale)
    counts = rng.multinomial(n_boot, [0.6, 0.2, 0.2], size=shape).astype(np.uint8)
    counts[~np.isfinite(one_wp["fg"]), 0] += counts[~np.isfinite(one_wp["fg"]), 1]
    counts[~np.isfinite(one_wp["fg"]), 1] = 0
    counts[~np.isfinite(one_wp["punt"]), 0] += counts[~np.isfinite(one_wp["punt"]), 2]
    counts[~np.isfinite(one_wp["punt"]), 2] = 0
    counts[..., ~ok, :] = 0
    for i, k in enumerate(("n_go", "n_fg", "n_punt")):
        one[k] = counts[..., i]
    one["p_convert"] = encode_wp(np.where(ok, 0.6 - 0.05 * np.asarray(dists)[:, None], np.nan)
                                 * np.ones(shape), scale)  # fmt: skip
    one["p_fg"] = encode_wp(np.where(np.isfinite(one_wp["fg"]), 0.9, np.nan), scale)
    exact = {k: encode_wp(v, scale) for k, v in wp_array(len(ectx)).items()}
    meta = {
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "axes": {
            "contexts": [c.__dict__ for c in ctx],
            "exact_contexts": [c.__dict__ for c in ectx],
            "scores": scores, "distances": dists, "yardlines": yls, "clocks": grid["clocks"],
            "array_axes": "[context, score, distance - 1, yardline_100 - 1]",
        },  # fmt: skip
        "encoding": {"wp_scale": scale, "gain_scale": scale},
        "bootstrap": {"n_boot": n_boot, "thresholds": {"confident": 0.83, "lean": 0.67}},
        "exact": {"dist_exact": 2},
    }
    return Bundle(meta, one, exact, {"synthetic": True})
