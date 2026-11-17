"""M3 uncertainty: a game-level bootstrap of the whole decision pipeline.

One replicate = draw whole games with replacement, refit the win-probability model
(trees and calibration map) and every play-outcome model on the draw, and re-evaluate
the fourth-down decisions with the refitted models. Across B replicates the share that
agrees with the point estimate's best action is `boot%`, and it sorts each decision
into "confident", "lean" or "uncertain" (the rule of Brill, Yurko and Wyner,
arXiv 2311.03490, recorded in docs/PAPERS.md).

Public API: load_config, prepare_plays, BootstrapPlan, fit_replicate, ensure_replicate,
load_replicate_engine, replicate_values, summarize, classify, category_shares,
split_half_agreement. Teaching notes: docs/notes/m3_bootstrap.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from fourth_and_rl import wp as wp_mod
from fourth_and_rl.components import ComponentModels, add_next_state, config_hash
from fourth_and_rl.components import load_config as load_components_config
from fourth_and_rl.decision import DecisionEngine, set_kickoff, summarize_actions
from fourth_and_rl.paths import CONFIGS, ROOT

CATEGORIES: tuple[str, str, str] = ("confident", "lean", "uncertain")

# Columns the refits read. Everything else is dropped before resampling to save memory.
_FIT_COLUMNS: list[str] = [
    "game_id", "play_id", "order_sequence", "season", "posteam", "defteam", "desc",
    "play_type", "yards_gained", "first_down", "touchdown", "td_team", "interception",
    "fumble_lost", "safety", "sack", "field_goal_result", "punt_blocked", "touchback",
    "extra_point_result", "two_point_attempt", "game_half", "fixed_drive",
    "next_posteam", "next_yardline_100", "next_half", "next_row_seconds", "next_row_half",
    wp_mod.LABEL, *wp_mod.STATE_COLUMNS,
]  # fmt: skip


def load_config(path: Path | None = None) -> dict:
    """Read configs/bootstrap.yaml."""
    with open(path or CONFIGS / "bootstrap.yaml") as f:
        return yaml.safe_load(f)


def prepare_plays(plays: pl.DataFrame) -> pl.DataFrame:
    """The table every replicate is drawn from.

    Adds the next-state columns ONCE, on the real games (`add_next_state` looks ahead
    within a game id, which would be wrong after games have been duplicated), keeps
    only the columns the refits read, and sorts so that each game, and each drive
    within a game, is one contiguous block of rows.
    """
    df = add_next_state(plays)
    df = df.select([c for c in _FIT_COLUMNS if c in df.columns])
    return df.sort("game_id", "fixed_drive", "order_sequence", "play_id", nulls_last=True)


# ------------------------------------------------------------------------ resampling


@dataclass
class _Block:
    """The games of one group of seasons, as row ranges of the prepared table."""

    game_start: np.ndarray  # first row of each game
    game_len: np.ndarray  # rows in each game
    drive_start: np.ndarray  # first row of each drive, all games concatenated
    drive_len: np.ndarray
    first_drive: np.ndarray  # index into drive_* of each game's first drive
    n_drives: np.ndarray  # drives in each game

    @property
    def n_games(self) -> int:
        return int(self.game_start.size)


class BootstrapPlan:
    """Which games can be drawn, and how a replicate is drawn from them.

    The games are split into the same two groups as the headline win-probability fit:
    `train` (the seasons the trees are grown on) and `calib` (the seasons the Platt map
    is fitted on). Each replicate draws, with replacement, as many games from each group
    as the group holds. Seasons outside the two groups (the test seasons) are never
    drawn.
    """

    def __init__(
        self,
        prepared: pl.DataFrame,
        train_seasons: tuple[int, int] | list[int],
        calib_seasons: tuple[int, int] | list[int],
        seed: int,
        within_game: str = "none",
        game_fraction: float = 1.0,
    ) -> None:
        if within_game not in ("none", "drives"):
            raise ValueError("within_game must be 'none' or 'drives'")
        if not 0 < game_fraction <= 1:
            raise ValueError("game_fraction must be in (0, 1]")
        self.prepared = prepared
        self.seed = int(seed)
        self.within_game = within_game
        self.game_fraction = float(game_fraction)
        self.train_seasons = (int(train_seasons[0]), int(train_seasons[1]))
        self.calib_seasons = (int(calib_seasons[0]), int(calib_seasons[1]))
        self.blocks = {
            "train": self._block(prepared, self.train_seasons),
            "calib": self._block(prepared, self.calib_seasons),
        }

    @staticmethod
    def _block(prepared: pl.DataFrame, seasons: tuple[int, int]) -> _Block:
        indexed = prepared.with_row_index("_row").filter(
            pl.col("season").is_between(seasons[0], seasons[1])
        )
        drives = indexed.group_by("game_id", "fixed_drive", maintain_order=True).agg(
            pl.col("_row").min().alias("start"), pl.len().alias("len"),
            (pl.col("_row").max() - pl.col("_row").min() + 1).alias("span"),
        )  # fmt: skip
        if drives.height and not (drives["span"] == drives["len"]).all():
            raise ValueError("rows of a drive are not contiguous; use prepare_plays first")
        games = drives.with_row_index("_drive").group_by("game_id", maintain_order=True).agg(
            pl.col("start").min().alias("start"), pl.col("len").sum().alias("len"),
            pl.col("_drive").min().alias("first_drive"), pl.len().alias("n_drives"),
        )  # fmt: skip
        as_int = lambda s: s.to_numpy().astype(np.int64)  # noqa: E731
        return _Block(
            as_int(games["start"]), as_int(games["len"]), as_int(drives["start"]),
            as_int(drives["len"]), as_int(games["first_drive"]), as_int(games["n_drives"]),
        )  # fmt: skip

    def rng(self, replicate: int) -> np.random.Generator:
        """One independent random stream per (seed, replicate)."""
        return np.random.default_rng([self.seed, int(replicate)])

    def draw(self, replicate: int) -> dict[str, np.ndarray]:
        """Game positions drawn for one replicate, per group (with replacement)."""
        rng = self.rng(replicate)
        out = {}
        for name, block in self.blocks.items():
            n = max(1, int(round(self.game_fraction * block.n_games)))
            out[name] = rng.integers(0, block.n_games, size=n)
        if self.within_game == "drives":
            out["_drive_seed"] = rng.integers(0, 2**63 - 1, size=1)
        return out

    def rows(self, name: str, games: np.ndarray, drive_seed: int | None = None) -> pl.DataFrame:
        """The plays of the drawn games. Each draw gets its own game id, so a game drawn
        twice counts as two games in anything that groups by game."""
        block = self.blocks[name]
        if self.within_game == "drives":
            rng = np.random.default_rng([int(drive_seed or 0), 0 if name == "train" else 1])
            n_d = block.n_drives[games]
            pick = block.first_drive[games].repeat(n_d) + np.floor(
                rng.random(int(n_d.sum())) * n_d.repeat(n_d)
            ).astype(np.int64)
            starts, lens = block.drive_start[pick], block.drive_len[pick]
            draw_of_piece = np.arange(games.size).repeat(n_d)
        else:
            starts, lens = block.game_start[games], block.game_len[games]
            draw_of_piece = np.arange(games.size)
        first = np.cumsum(lens) - lens
        idx = np.repeat(starts - first, lens) + np.arange(int(lens.sum()))
        draw_id = np.repeat(draw_of_piece, lens)
        tag = pl.Series("_draw", draw_id).cast(pl.String)
        return self.prepared[idx].with_columns(
            (pl.col("game_id") + pl.lit(f"#{name}") + tag).alias("game_id")
        )


# -------------------------------------------------------------------------- refitting


def fit_replicate(
    plan: BootstrapPlan,
    replicate: int,
    wp_config: dict,
    components_config: dict | None,
    kickoff_seasons: list[int],
    n_threads: int = 2,
    calibration: str = "platt",
) -> tuple[wp_mod.WPModel, ComponentModels, dict[str, np.ndarray]]:
    """Draw one replicate and refit everything on it.

    Win probability: trees on the drawn training games, calibration map on the drawn
    calibration games (so the noise of a two-season calibration set is part of the
    bootstrap). Components: all drawn games together (development seasons only; the fit
    guards refuse anything else). The kickoff tables of `kickoff_seasons` are the fixed
    rule-era tables, fitted on those seasons' own kickoffs and not resampled (see
    `decision.set_kickoff`).
    """
    draw = plan.draw(replicate)
    drive_seed = int(draw["_drive_seed"][0]) if "_drive_seed" in draw else None
    train = plan.rows("train", draw["train"], drive_seed)
    calib = plan.rows("calib", draw["calib"], drive_seed)
    model = wp_mod.WPModel(wp_config["lgbm"], calibration, n_threads).fit(
        train, calib if calibration != "none" else None, seed=int(wp_config["seed"]) + replicate
    )
    comps = ComponentModels(components_config).fit(pl.concat([train, calib]), seed=plan.seed)
    set_kickoff(comps, plan.prepared, kickoff_seasons)
    return model, comps, draw


def replicate_paths(boot_dir: Path, replicate: int) -> dict[str, Path]:
    stem = f"rep_{replicate:03d}"
    return {
        "wp": boot_dir / f"{stem}_wp.joblib",
        "components": boot_dir / f"{stem}_components.joblib",
        "draw": boot_dir / f"{stem}_draw.npz",
        "key": boot_dir / f"{stem}_key.json",
    }


def replicate_key(
    plan: BootstrapPlan,
    replicate: int,
    wp_config: dict,
    components_config: dict | None,
    kickoff_seasons: list[int],
    calibration: str,
) -> dict[str, Any]:
    """Everything a saved replicate depends on. A saved replicate is reused only when its
    key file equals this, so a changed config refits it instead of silently reusing it."""
    comps_cfg = components_config if components_config is not None else load_components_config()
    return {
        "replicate": int(replicate),
        "seed": plan.seed,
        "train_seasons": list(plan.train_seasons),
        "calib_seasons": list(plan.calib_seasons),
        "within_game": plan.within_game,
        "game_fraction": plan.game_fraction,
        "games": {name: block.n_games for name, block in plan.blocks.items()},
        "wp_config_hash": config_hash(wp_config),
        "components_config_hash": config_hash(comps_cfg),
        "kickoff_seasons": sorted(int(x) for x in kickoff_seasons),
        "calibration": calibration,
    }


def boot_dir(config: dict, seed: int | None = None) -> Path:
    """Folder holding one seed's replicates."""
    s = config["seed"] if seed is None else seed
    return ROOT / config["models_dir"] / f"seed_{int(s)}_{config['within_game']}"


def ensure_replicate(
    plan: BootstrapPlan,
    replicate: int,
    directory: Path,
    wp_config: dict,
    components_config: dict | None,
    kickoff_seasons: list[int],
    n_threads: int = 2,
    calibration: str = "platt",
) -> bool:
    """Fit and save one replicate unless an up-to-date copy exists. Returns True if it was
    fitted now. "Up to date" means the saved key file equals `replicate_key` for these
    settings (file existence alone is not enough). The key file is written last and marks
    the replicate done."""
    paths = replicate_paths(directory, replicate)
    key = replicate_key(plan, replicate, wp_config, components_config, kickoff_seasons,
                        calibration)  # fmt: skip
    if all(p.exists() for p in paths.values()):
        try:
            if json.loads(paths["key"].read_text()) == key:
                return False
        except (OSError, ValueError):
            pass
    directory.mkdir(parents=True, exist_ok=True)
    model, comps, draw = fit_replicate(
        plan, replicate, wp_config, components_config, kickoff_seasons, n_threads, calibration
    )
    np.savez_compressed(paths["draw"], **draw)
    model.save(paths["wp"])
    comps.save(paths["components"])
    paths["key"].write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
    return True


def load_replicate_engine(
    directory: Path, replicate: int, engine_config: dict | None = None
) -> DecisionEngine:
    paths = replicate_paths(directory, replicate)
    return DecisionEngine(
        wp_mod.WPModel.load(paths["wp"]), ComponentModels.load(paths["components"]), engine_config
    )


def _states_tag(states: pl.DataFrame, engine_config: dict | None) -> str:
    cols = states.select(wp_mod.STATE_COLUMNS).cast(pl.Float64)
    digest = hashlib.sha1(cols.to_numpy().tobytes())
    digest.update(repr(sorted((engine_config or {}).items())).encode())
    return digest.hexdigest()[:12]


def replicate_values(
    states: pl.DataFrame,
    directory: Path,
    replicates: list[int] | range,
    engine_config: dict | None = None,
    cache_name: str | None = None,
) -> np.ndarray:
    """Go, field goal and punt win probability of every state under every replicate.

    Returns an array of shape (number of replicates, number of states, 3), NaN where an
    action is unavailable. With `cache_name`, each replicate's values are saved under
    `<directory>/values/` and reused when the states and engine settings are unchanged.
    """
    cache = None
    if cache_name:
        cache = directory / "values" / f"{cache_name}_{_states_tag(states, engine_config)}"
        cache.mkdir(parents=True, exist_ok=True)
    out = np.empty((len(replicates), states.height, 3), dtype=np.float32)
    for i, b in enumerate(replicates):
        path = cache / f"rep_{b:03d}.npy" if cache else None
        if path is not None and path.exists():
            out[i] = np.load(path)
            continue
        v = load_replicate_engine(directory, b, engine_config).action_values(states)
        out[i] = np.column_stack([v["go_wp"], v["fg_wp"], v["punt_wp"]])
        if path is not None:
            np.save(path, out[i])
    return out


# ------------------------------------------------------------------------ summarizing


def classify(boot_pct: np.ndarray, confident: float = 0.83, lean: float = 0.67) -> np.ndarray:
    """"confident" if boot% >= `confident`, "lean" if >= `lean`, else "uncertain"."""
    boot_pct = np.asarray(boot_pct, dtype=float)
    out = np.where(boot_pct >= confident, "confident", np.where(boot_pct >= lean, "lean",
                                                                "uncertain"))  # fmt: skip
    return np.where(np.isnan(boot_pct), None, out.astype(object))


def summarize(
    point: np.ndarray,
    boot: np.ndarray,
    confident: float = 0.83,
    lean: float = 0.67,
    interval: tuple[float, float] = (0.05, 0.95),
) -> dict[str, np.ndarray]:
    """Turn replicate values into boot%, a category and a gain interval per state.

    `point` is (n, 3) from the point-estimate engine and `boot` is (B, n, 3).

      boot_pct      share of replicates whose best action equals the point estimate's
      category      from `classify`
      share_go, share_field_goal, share_punt   share of replicates preferring each
      gain_lo, gain_hi   order statistics of go_wp - max(fg_wp, punt_wp) across the
                    replicates. With B = 101 and the default (0.05, 0.95) these are the
                    6th and 96th sorted values, a 90% interval.
      gain_median   the middle replicate
    """
    best, _, _, gain = summarize_actions(np.asarray(point, dtype=float))
    n_boot = boot.shape[0]
    boot_best = np.empty(boot.shape[:2], dtype=np.int8)
    boot_gain = np.empty(boot.shape[:2])
    for b in range(n_boot):
        bb, _, _, bg = summarize_actions(boot[b].astype(float))
        boot_best[b], boot_gain[b] = bb, bg
    valid = best >= 0
    agree = (boot_best == best[None, :]).mean(axis=0)
    boot_pct = np.where(valid, agree, np.nan)
    lo_i = int(round(interval[0] * (n_boot - 1)))
    hi_i = int(round(interval[1] * (n_boot - 1)))
    ordered = np.sort(boot_gain, axis=0)  # NaN (no kick available) sorts last
    out = {
        "best": best,
        "gain": gain,
        "boot_pct": boot_pct,
        "category": classify(boot_pct, confident, lean),
        "gain_lo": ordered[lo_i],
        "gain_hi": ordered[hi_i],
        "gain_median": ordered[(n_boot - 1) // 2],
    }
    for i, name in enumerate(("go", "field_goal", "punt")):
        out[f"share_{name}"] = np.where(valid, (boot_best == i).mean(axis=0), np.nan)
    return out


def category_shares(category: np.ndarray, best: np.ndarray | None = None) -> dict[str, Any]:
    """Share of states in each category, overall and (optionally) by best action."""
    category = np.asarray(category, dtype=object)
    known = np.array([c is not None for c in category])
    n = int(known.sum())
    out: dict[str, Any] = {"n": n}
    for c in CATEGORIES:
        out[c] = float(np.sum(category == c) / n) if n else float("nan")
    if best is not None:
        out["by_best_action"] = {}
        for i, name in enumerate(("go", "field_goal", "punt")):
            m = known & (best == i)
            out["by_best_action"][name] = {
                "n": int(m.sum()),
                **{c: float(np.sum(category[m] == c) / max(1, m.sum())) for c in CATEGORIES},
            }
    return out


def split_half_agreement(
    point: np.ndarray, boot: np.ndarray, confident: float = 0.83, lean: float = 0.67
) -> dict[str, Any]:
    """Stability check: classify with the even-numbered and the odd-numbered replicates
    separately and report how often the two halves give the same category.

    Each half has about B / 2 replicates, so this understates the stability of the
    full-B categories."""
    a = summarize(point, boot[0::2], confident, lean)
    b = summarize(point, boot[1::2], confident, lean)
    known = np.array([c is not None for c in a["category"]])
    same = a["category"][known] == b["category"][known]
    cross = {
        f"{ca}_vs_{cb}": int(np.sum((a["category"][known] == ca) & (b["category"][known] == cb)))
        for ca in CATEGORIES
        for cb in CATEGORIES
    }
    return {
        "n_states": int(known.sum()),
        "replicates_per_half": [int(boot[0::2].shape[0]), int(boot[1::2].shape[0])],
        "category_agreement": float(same.mean()) if same.size else float("nan"),
        "mean_abs_boot_pct_difference": float(
            np.mean(np.abs(a["boot_pct"][known] - b["boot_pct"][known]))
        ),
        "counts_half_a_vs_half_b": cross,
    }
