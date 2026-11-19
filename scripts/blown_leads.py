"""M3b: blown leads as a calibration test.

For every non-tied game, take the losing team's win-probability path under each
available model, and compare the distribution of the losers' peaks with what a
calibrated model would produce (Pipping-Gamon and Wyner, arXiv 2601.18774).

Models: nflfastR `wp`, nflfastR `vegas_wp`, and our own `wp_ours` when
data/processed/wp_predictions.parquet exists (joined on game_id, play_id).

Outputs:
  results/m3b_blown_leads.json            tests, counts, jump sizes, simulation summary
  results/m3b_collapses_by_peak.csv       top collapses by raw peak, per model
  results/m3b_collapses_by_surprise.csv   top collapses by surprise, per model
  assets/m3b_blown_leads.png              losers' peaks against the reference
  assets/m3b_collapse_paths.png           gallery of paths
  assets/m3b_discreteness.png             how play-to-play jumps shift the reference
"""

from __future__ import annotations

import time
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from scipy.special import logit

from fourth_and_rl import blown_leads as bl
from fourth_and_rl import data, style
from fourth_and_rl.io import write_json
from fourth_and_rl.paths import ASSETS, CONFIGS, RESULTS, ROOT

PAPER = {
    "title": (
        "The Blown Lead Paradox: A Pathwise Calibration Benchmark for Win Probability "
        "Forecasts"
    ),
    "title_of_first_version": (
        "The Blown Lead Paradox: Conditional Laws for the Running Maximum of Binary Doob "
        "Martingales"
    ),
    "authors": ["Jonathan Pipping-Gamon", "Abraham J. Wyner"],
    "arxiv": "2601.18774",
    "first_submitted": "2026-01-26",
    "version_read": "v4, 2026-08-25",
    "numbers_quoted_from_paper": {
        "note": (
            "ESPN public NFL feed, regular seasons 2018 to 2024, 1,848 games. These are "
            "the paper's numbers, not ours, and use a different model and sample."
        ),
        "nfl_d_upper": 0.015,
        "nfl_d_upper_p": 0.736,
        "nfl_share_with_game_pit_at_least": {"0.90": 0.085, "0.95": 0.045, "0.99": 0.015},
        "nfl_first_crossing_observed_vs_implied_loss": {
            "0.90": [0.085, 0.076], "0.95": [0.046, 0.033], "0.99": [0.016, 0.005],
        },
    },
}  # fmt: skip

X_TICKS = [0.5, 0.75, 0.9, 0.99, 0.999]


def load_config() -> dict:
    with open(CONFIGS / "blown_leads.yaml") as f:
        return yaml.safe_load(f)


def load_plays_with_predictions(cfg: dict) -> tuple[pl.DataFrame, bool]:
    plays = data.load_plays()
    path = ROOT / cfg["predictions_path"]
    if not path.exists():
        return plays, False
    preds = pl.read_parquet(path).select("game_id", "play_id", "wp_ours")
    preds = preds.with_columns(pl.col("play_id").cast(plays.schema["play_id"]))
    return plays.join(preds, on=["game_id", "play_id"], how="left"), True


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def calibrate_gap_shape(
    real_jumps: np.ndarray, p0_home: np.ndarray, n_steps: int, cfg: dict, rng: np.random.Generator
) -> dict:
    """Pick the simulator's time-gap shape whose jump sizes look most like the real ones.

    Distance = sum of squared log ratios of the median, 90th, 99th and 99.9th
    percentiles of the absolute play-to-play change.
    """
    probs = [0.5, 0.9, 0.99, 0.999]
    target = np.quantile(real_jumps, probs)
    n_paths = cfg["calibration_paths"]
    table = []
    for shape in cfg["gap_shape_grid"]:
        p0 = rng.choice(p0_home, size=n_paths)
        sim = bl.simulated_jump_sizes(p0, n_steps, n_paths, rng, shape)
        q = np.quantile(sim, probs)
        table.append(
            {
                "gap_shape": shape,
                "quantiles": q.tolist(),
                "distance": float(np.sum(np.log(q / target) ** 2)),
            }
        )
    best = min(table, key=lambda r: r["distance"])
    return {
        "quantile_probs": probs,
        "real_quantiles": target.tolist(),
        "candidates": table,
        "chosen_gap_shape": best["gap_shape"],
        "chosen_quantiles": best["quantiles"],
    }


def peak_counts(
    peak: np.ndarray,
    p0: np.ndarray,
    sf: Callable[[float, np.ndarray], np.ndarray],
    thresholds: list[float],
    rng: np.random.Generator,
    n_sim: int,
) -> dict:
    return {
        f"{x:.2f}": bl.extreme_peak_counts(peak >= x, sf(x, p0), rng, n_sim) for x in thresholds
    }


def analyze_model(
    plays: pl.DataFrame, spec: dict, cfg: dict, idx: int, n_nontied: int
) -> tuple[dict, dict]:
    """Run every test for one win-probability column. Returns (summary, plot data)."""
    seed, thresholds = cfg["seed"], cfg["thresholds"]
    rng = np.random.default_rng([seed, idx])
    hp = bl.home_paths(plays, spec["column"])
    games = bl.add_reference_columns(
        bl.extract_loser_paths(plays, spec["column"], cfg["min_plays"])
    )
    hp = hp.filter(pl.col("game_id").is_in(games["game_id"].to_list()))
    peak, p0 = games["peak"].to_numpy(), games["p0"].to_numpy()
    n = games.height
    jumps = bl.jump_sizes(hp)
    weights = bl.pigeonhole_weights(
        games["season"].to_numpy(), games["home_team"].to_numpy(),
        games["away_team"].to_numpy(), cfg["n_boot"], np.random.default_rng([seed, 1000]),
    )  # fmt: skip

    continuous = {
        "pit_ks": bl.ks_uniform(games["pit"].to_numpy()),
        "pit_bootstrap": bl.pit_bootstrap_test(games["pit"].to_numpy(), weights),
        "pit_game_ks": bl.ks_uniform(games["pit_game"].to_numpy()),
        "pit_game_bootstrap": bl.pit_bootstrap_test(games["pit_game"].to_numpy(), weights),
        "extreme_peaks": peak_counts(peak, p0, bl.peak_sf, thresholds, rng, cfg["n_count_sim"]),
    }
    # The pigeonhole weights are per game (columns in the order of `games`); each crossing
    # takes its game's weight, which allows for dependence the independence z ignores.
    game_ids = games["game_id"].to_numpy()
    crossing = {
        f"{x:.2f}": bl.crossing_test(bl.first_crossings(hp, x), rng, cfg["n_count_sim"],
                                     weights, game_ids)  # fmt: skip
        for x in cfg["crossing_thresholds"]
    }

    # discrete-adjusted reference from the simulator
    dc = cfg["discrete"]
    n_steps = dc["n_steps"] or int(games["n_plays"].median())
    p0_home = hp.group_by("game_id", maintain_order=True).agg(pl.col("home_wp_path").first())[
        "home_wp_path"
    ].to_numpy()
    calib = calibrate_gap_shape(jumps, p0_home, n_steps, dc, rng)
    grid = np.arange(dc["p0_grid"]["start"], dc["p0_grid"]["stop"] + 1e-9, dc["p0_grid"]["step"])
    ref = bl.build_discrete_reference(
        grid, n_steps, dc["reference_paths"], rng, calib["chosen_gap_shape"]
    )
    pit_disc = ref.pit(peak, p0, rng)
    sensitivity = {}
    for shape in dc["sensitivity_gap_shapes"]:
        if shape == calib["chosen_gap_shape"]:
            continue
        alt = bl.build_discrete_reference(grid, n_steps, dc["reference_paths"], rng, shape)
        sensitivity["equal_spacing" if shape is None else f"gap_shape_{shape}"] = {
            f"{x:.2f}": float(alt.sf(x, p0).sum()) for x in thresholds
        }
    discrete = {
        "n_steps": n_steps,
        "gap_shape": calib["chosen_gap_shape"],
        "jump_calibration": calib,
        "simulated_share_never_above_start": float(np.mean(ref.cdf(p0, p0))),
        "pit_ks": bl.ks_uniform(pit_disc),
        "pit_bootstrap": bl.pit_bootstrap_test(pit_disc, weights),
        "extreme_peaks": peak_counts(peak, p0, ref.sf, thresholds, rng, cfg["n_count_sim"]),
        "expected_counts_other_simulators": sensitivity,
        "discrete_to_continuous_expected_ratio": {
            f"{x:.2f}": float(ref.sf(x, p0).sum() / bl.peak_sf(x, p0).sum()) for x in thresholds
        },
    }

    ranks = bl.rank_collapses(games, cfg["rankings"]["top"], cfg["rankings"]["peak_decimals"])
    summary = {
        "label": spec["label"],
        "column": spec["column"],
        "n_games": n,
        "n_nontied_games": n_nontied,
        "n_games_dropped": n_nontied - n,
        "seasons": [int(games["season"].min()), int(games["season"].max())],
        "median_plays_per_game": float(games["n_plays"].median()),
        "loser_p0": {
            "mean": float(p0.mean()),
            "q05": float(np.quantile(p0, 0.05)),
            "median": float(np.median(p0)),
            "q95": float(np.quantile(p0, 0.95)),
        },
        "loser_peak": {
            "median": float(np.median(peak)),
            "share_at_least": {
                f"{x:.2f}": float((peak >= x).mean()) for x in [0.5, 0.67, 0.75, *thresholds]
            },
            "share_never_above_start": float((peak <= p0).mean()),
        },
        "jump_sizes": bl.jump_summary(jumps),
        "mean_largest_jump_per_game": float(games["max_jump"].mean()),
        "continuous_reference": continuous,
        "first_crossing": crossing,
        "discrete_adjusted_reference": discrete,
        "rankings": {
            k: v for k, v in ranks.items() if k not in ("by_peak", "by_surprise")
        },
    }
    xs = 1 / (1 + np.exp(-np.linspace(0.0, logit(0.9995), 160)))
    plot = {
        "games": games,
        "hp": hp,
        "ranks": ranks,
        "jumps": jumps,
        "pit_disc": pit_disc,
        "xs": xs,
        "emp_sf": np.array([(peak >= x).mean() for x in xs]),
        "cont_sf": np.array([bl.peak_sf(x, p0).mean() for x in xs]),
        "disc_sf": np.array([ref.sf(x, p0).mean() for x in xs]),
        "n_steps": n_steps,
        "gap_shape": calib["chosen_gap_shape"],
    }
    return summary, plot


def discreteness_sweep(cfg: dict, extra: list[tuple[int, float | None]]) -> dict:
    """Even game, equal spacing: how the loser's-peak tail moves with the number of steps."""
    dc = cfg["discrete"]
    rng = np.random.default_rng([cfg["seed"], 2000])
    xs = 1 / (1 + np.exp(-np.linspace(0.0, logit(0.995), 60)))
    marks = [0.75, 0.90, 0.95, 0.99]
    rows = []
    settings = [(n, None) for n in dc["sweep_steps"]] + extra
    for n_steps, shape in settings:
        n_paths = int(min(dc["sweep_max_paths"], dc["sweep_normals_budget"] // n_steps))
        peaks = bl.simulate_loser_peaks(0.5, n_steps, n_paths, rng, shape)
        m = peaks.size
        ratios = {}
        for x in marks:
            share, ref = float((peaks >= x).mean()), float(bl.peak_sf(x, 0.5))
            ratios[f"{x:.2f}"] = {
                "simulated": share,
                "continuous_formula": ref,
                "ratio": share / ref,
                "ratio_se": float(np.sqrt(share * (1 - share) / m) / ref),
            }
        rows.append(
            {
                "n_steps": n_steps,
                "gap_shape": shape,
                "n_losing_paths": int(m),
                "share_never_above_start": float((peaks <= 0.5).mean()),
                "tail": ratios,
                "curve_x": xs.tolist(),
                "curve_ratio": [float((peaks >= x).mean() / bl.peak_sf(x, 0.5)) for x in xs],
            }
        )
    return {
        "description": (
            "Simulated even game (p0 = 0.5): Brownian lead reported at n_steps play times. "
            "ratio = P(reported peak >= x | lost) / continuous formula. gap_shape null "
            "means equally spaced plays; a number means Gamma-distributed gaps."
        ),
        "settings": rows,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _logit_axis(ax: plt.Axes, ticks: list[float]) -> None:
    ax.set_xticks([logit(t) for t in ticks])
    ax.set_xticklabels([f"{100 * t:g}%" for t in ticks])


def figure_main(plots: dict, summaries: dict) -> None:
    style.apply()
    keys = list(plots)
    fig, axes = plt.subplots(
        2, len(keys), figsize=(4.9 * len(keys), 9.2), squeeze=False,
        gridspec_kw={"hspace": 0.5, "wspace": 0.3},
    )  # fmt: skip
    for j, key in enumerate(keys):
        p, s, color = plots[key], summaries[key], style.MODEL_COLORS[key]
        ax = axes[0, j]
        lx = logit(p["xs"])
        ax.plot(lx, p["cont_sf"], color=style.MODEL_COLORS["theory"], linewidth=1.8,
                label="Calibrated, continuous (formula)")
        ax.plot(lx, p["disc_sf"], color=style.MODEL_COLORS["theory"], linewidth=1.6,
                linestyle=(0, (4, 3)), label="Calibrated, with jumps (simulated)")
        ax.plot(lx, p["emp_sf"], color=color, linewidth=2.4, label="Observed")
        ax.set_yscale("log")
        ax.set_ylim(3e-4, 1.3)
        ax.set_yticks([1, 0.1, 0.01, 0.001])
        ax.set_yticklabels(["100%", "10%", "1%", "0.1%"])
        ax.set_xlim(0, logit(0.9995))
        _logit_axis(ax, X_TICKS)
        ax.set_xlabel("Peak win probability of the team that lost")
        if j == 0:
            ax.set_ylabel("Share of losers who peaked at least this high")
        ax.legend(loc="upper right", fontsize=8.4)
        style.title(ax, s["label"], f"{s['n_games']:,} games, {s['seasons'][0]} to "
                    f"{s['seasons'][1]}")

        ax = axes[1, j]
        bins = np.linspace(0, 1, 21)
        centers = (bins[:-1] + bins[1:]) / 2
        h_cont = np.histogram(p["games"]["pit"].to_numpy(), bins=bins)[0] / s["n_games"]
        h_disc = np.histogram(p["pit_disc"], bins=bins)[0] / s["n_games"]
        ax.bar(centers, 100 * h_cont, width=0.046, color=color,
               label="Scored against the formula")
        ax.step(bins, 100 * np.append(h_disc, h_disc[-1]), where="post", color=style.INK,
                linewidth=1.5, label="Scored against the simulation")
        ax.axhline(5, color=style.INK_2, linestyle=(0, (4, 3)), linewidth=1.2,
                   label="Calibrated model (flat)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(15.0, 100 * h_cont.max() * 1.2))
        ax.set_xlabel("Unusualness of the loser's peak (PIT, 1 = extreme)")
        if j == 0:
            ax.set_ylabel("Share of games (%)")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", fontsize=8.6)
        style.title(ax, "How unusual each loser's peak was",
                    "Flat = calibrated. Excess at right = overconfident")
    fig.text(0.07, 0.975, "Losing teams reach high win probabilities less often than "
             "a calibrated continuous path would", fontsize=14, fontweight="bold",
             color=style.INK, ha="left", va="bottom")
    fig.text(0.07, 0.955, "Most of the gap is the play-to-play jumps of real paths, not "
             "miscalibration: the simulated reference with jumps sits much closer to the data",
             fontsize=10.5, color=style.INK_2, ha="left", va="bottom")
    style.save(fig, ASSETS / "m3b_blown_leads.png")


def figure_gallery(plot: dict, summary: dict, key: str, cfg: dict) -> list[dict]:
    style.apply()
    g = cfg["gallery"]
    games, hp = plot["games"], plot["hp"]
    surprising = games.sort("tail_prob").head(g["n_surprising"])
    unsurprising = (
        games.filter(pl.col("peak") >= g["unsurprising_min_peak"])
        .sort("tail_prob", descending=True)
        .head(g["n_unsurprising"])
    )
    picks = pl.concat([surprising, unsurprising])
    ncols = 3
    nrows = int(np.ceil(picks.height / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.55 * nrows), squeeze=False,
                             gridspec_kw={"hspace": 0.62, "wspace": 0.22})
    shown = []
    for i, row in enumerate(picks.iter_rows(named=True)):
        ax = axes[i // ncols, i % ncols]
        path = hp.filter(pl.col("game_id") == row["game_id"])
        h = path["home_wp_path"].to_numpy()
        loser_wp = h if row["loser"] == row["home_team"] else 1 - h
        y = 100 * np.append(loser_wp, 0.0)
        ax.plot(np.arange(y.size), y, color=style.MODEL_COLORS[key], linewidth=1.7)
        ax.axhline(100 * row["p0"], color=style.MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
        ax.plot(row["peak_step"], 100 * row["peak"], "o", color=style.INK, markersize=5.5,
                markeredgecolor=style.SURFACE, markeredgewidth=1.2)
        ax.set_ylim(-3, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(0, y.size - 1)
        ax.grid(axis="x", visible=False)
        if i % ncols == 0:
            ax.set_ylabel("Loser's win probability (%)")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Play number")
        odds = 1 / row["tail_prob"]
        group = "Surprising" if i < g["n_surprising"] else "High peak, not surprising"
        style.title(
            ax,
            f"{row['loser']} lost {row['loser_score']}-{row['winner_score']} to "
            f"{row['winner']}, {row['season']} wk {row['week']}",
            f"Start {100 * row['p0']:.0f}%, peak {100 * row['peak']:.1f}%: "
            f"1 in {odds:,.0f} losers",
        )
        ax.title.set_fontsize(11)
        shown.append(
            {
                "group": group,
                "game_id": row["game_id"],
                "loser": row["loser"],
                "p0": row["p0"],
                "peak": row["peak"],
                "tail_prob": row["tail_prob"],
            }
        )
    for k in range(picks.height, nrows * ncols):
        axes[k // ncols, k % ncols].set_visible(False)
    fig.text(0.07, 1.012, "The same peak means different things for an underdog and a favorite",
             fontsize=14, fontweight="bold", color=style.INK, ha="left", va="bottom")
    fig.text(0.07, 0.955,
             f"Win-probability paths of the losing team ({summary['label']}). Dot: the peak. "
             "Dashed line: where the loser started.\n"
             f"First {g['n_surprising']} panels: the most surprising collapses. Last "
             f"{g['n_unsurprising']}: peaks of {100 * g['unsurprising_min_peak']:.0f}% or more "
             "that are common for a team that started as a heavy favorite.\n"
             "\"1 in N\": how rare a peak that high is among calibrated losers with that start.",
             fontsize=10, color=style.INK_2, ha="left", va="bottom", linespacing=1.4)
    style.save(fig, ASSETS / "m3b_collapse_paths.png")
    return shown


def figure_discreteness(plots: dict, summaries: dict, sweep: dict) -> None:
    style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"wspace": 0.25})
    d = np.linspace(0.0, 0.6, 121)
    first = next(iter(plots))
    for key, p in plots.items():
        sf = np.array([(p["jumps"] > v).mean() for v in d])
        ax1.plot(100 * d, sf, color=style.MODEL_COLORS[key], label=summaries[key]["label"])
    rng = np.random.default_rng(7)
    n_steps, shape = plots[first]["n_steps"], plots[first]["gap_shape"]
    for label, sh, ls in (
        (f"Simulator fitted to real jumps ({n_steps} plays, uneven spacing)", shape,
         (0, (4, 3))),
        (f"Simulator, {n_steps} evenly spaced plays", None, (0, (1, 2))),
    ):
        sim = bl.simulated_jump_sizes(0.5, n_steps, 20000, rng, sh)
        ax1.plot(100 * d, [(sim > v).mean() for v in d], color=style.INK, linestyle=ls,
                 linewidth=1.5, label=label)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-5, 1.5)
    ax1.set_yticks([1, 0.1, 0.01, 0.001, 0.0001])
    ax1.set_yticklabels(["100%", "10%", "1%", "0.1%", "0.01%"])
    ax1.set_xlim(0, 60)
    ax1.set_xlabel("Change in win probability on one play (percentage points)")
    ax1.set_ylabel("Share of plays with a larger change")
    ax1.legend(loc="upper right", fontsize=8.8)
    style.title(ax1, "Real win-probability paths jump",
                "Size of play-to-play changes, real paths and the simulator")

    equal = [r for r in sweep["settings"] if r["gap_shape"] is None]
    ramp = style.SEQUENTIAL[-len(equal):] if len(equal) <= len(style.SEQUENTIAL) else None
    for i, r in enumerate(equal):
        color = ramp[i] if ramp else style.BLUE
        ax2.plot(logit(np.array(r["curve_x"])), r["curve_ratio"], color=color, linewidth=1.8,
                 label=f"{r['n_steps']:,} plays")
    for r in sweep["settings"]:
        if r["gap_shape"] is not None:
            ax2.plot(logit(np.array(r["curve_x"])), r["curve_ratio"], color=style.RED,
                     linewidth=2.0, linestyle=(0, (4, 3)),
                     label=f"{r['n_steps']} plays, fitted to real jumps")
    ax2.axhline(1.0, color=style.MODEL_COLORS["theory"], linewidth=1.6,
                label="Continuous formula")
    ax2.set_ylim(0, 1.12)
    ax2.set_xlim(0, logit(0.995))
    _logit_axis(ax2, [0.5, 0.75, 0.9, 0.95, 0.99])
    ax2.set_xlabel("Peak win probability of the team that lost")
    ax2.set_ylabel("Simulated share of losers at or above the peak,\nrelative to the formula")
    ax2.legend(loc="lower left", fontsize=8.8, ncol=2)
    style.title(ax2, "Fewer, bigger steps hide extreme peaks",
                "Calibrated simulated even game reported at a fixed number of plays")
    style.save(fig, ASSETS / "m3b_discreteness.png")


# ---------------------------------------------------------------------------


CSV_COLUMNS = [
    "model", "rank_by_peak", "rank_by_surprise", "game_id", "season", "week", "away_team",
    "home_team", "loser", "winner", "loser_score", "winner_score", "p0", "peak", "tail_prob",
    "one_in", "peak_qtr", "peak_seconds_remaining", "peak_loser_pts", "peak_winner_pts",
    "peak_desc",
]  # fmt: skip


def write_rankings(plots: dict, summaries: dict) -> None:
    for name in ("by_peak", "by_surprise"):
        frames = []
        for key, p in plots.items():
            frames.append(
                p["ranks"][name]
                .with_columns(
                    pl.lit(summaries[key]["label"]).alias("model"),
                    (1.0 / pl.col("tail_prob")).alias("one_in"),
                )
                .select(CSV_COLUMNS)
            )
        pl.concat(frames).write_csv(RESULTS / f"m3b_collapses_{name}.csv", float_precision=6)


def main() -> None:
    cfg = load_config()
    t0 = time.time()
    plays, has_ours = load_plays_with_predictions(cfg)
    n_nontied = plays.filter(pl.col("result") != 0)["game_id"].n_unique()
    specs = [m for m in cfg["models"] if m["column"] in plays.columns]
    summaries, plots = {}, {}
    for idx, spec in enumerate(cfg["models"]):
        if spec not in specs:
            continue
        summaries[spec["key"]], plots[spec["key"]] = analyze_model(
            plays, spec, cfg, idx, n_nontied
        )
        print(f"{spec['label']}: done ({time.time() - t0:.0f}s)")
    first = next(iter(plots))
    sweep = discreteness_sweep(cfg, [(plots[first]["n_steps"], plots[first]["gap_shape"])])
    gallery_key = cfg["gallery"]["model"] if cfg["gallery"]["model"] in plots else first
    figure_main(plots, summaries)
    shown = figure_gallery(plots[gallery_key], summaries[gallery_key], gallery_key, cfg)
    figure_discreteness(plots, summaries, sweep)
    write_rankings(plots, summaries)
    out = {
        "paper": PAPER,
        "notes": {
            "p0": "the losing team's first win probability of the game (before the kickoff)",
            "peak": "the losing team's largest win probability over all plays",
            "continuous_reference": (
                "P(peak >= x | lost) = p0 (1 - x) / (x (1 - p0)); expected counts are sums "
                "of this over games; intervals assume independent games"
            ),
            "first_crossing": (
                "expected blown leads = sum of (1 - p) at each team's first value at or "
                "above the threshold; exact for a calibrated model even when paths jump. "
                "z, p_upper and expected_interval95 treat crossings as independent; "
                "team_season_bootstrap gives the interval and p-values with both sides of "
                "a game and games sharing a team-season resampled together (the same "
                "pigeonhole weights as pit_bootstrap)"
            ),
            "discrete_adjusted_reference": (
                "simulated calibrated game reported at play times, with time gaps fitted "
                "to the real jump sizes; an approximation, see docs/notes/m3b_blown_leads.md"
            ),
            "pit_bootstrap": (
                "season-stratified team (pigeonhole) bootstrap of the one-sided statistic "
                "d_upper, as in the paper"
            ),
            "wp_ours": (
                "out-of-fold predictions from scripts/fit_wp.py when present"
                if has_ours
                else "not available when this was run"
            ),
        },
        "config": cfg,
        "models": summaries,
        "discreteness_simulation": sweep,
        "gallery": {"model": gallery_key, "games": shown},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    write_json(out, RESULTS / "m3b_blown_leads.json")
    for key, s in summaries.items():
        print(f"\n{s['label']} ({key}): {s['n_games']} games")
        for x in cfg["thresholds"]:
            k = f"{x:.2f}"
            c = s["continuous_reference"]["extreme_peaks"][k]
            d = s["discrete_adjusted_reference"]["extreme_peaks"][k]
            f = s["first_crossing"][k]
            print(f"  peak >= {k}: observed {c['observed']:4d} | formula {c['expected']:6.1f} | "
                  f"simulated {d['expected']:6.1f} | first crossing {f['expected']:6.1f}")
        for k, f in s["first_crossing"].items():
            tb = f["team_season_bootstrap"]
            print(f"  first crossing {k}: excess {tb['excess']:+6.1f}, z independent "
                  f"{f['z']:+.2f}, z team-season {tb['z']:+.2f} (p two-sided "
                  f"{tb['p_two_sided']:.3f})")  # fmt: skip
        print(f"  PIT KS (formula) {s['continuous_reference']['pit_ks']['ks_two_sided']:.3f}, "
              f"(simulated) {s['discrete_adjusted_reference']['pit_ks']['ks_two_sided']:.3f}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
