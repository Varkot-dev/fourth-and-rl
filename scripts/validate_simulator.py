"""M4: is the game model a fair picture of real football?

Simulates whole games from the solved model's transition tables under a coach-like
policy (fourth downs sampled from observed coach behavior by situation, protect a lead
and chase a deficit in the last 5:00, measured timeout rates) and sets the results beside
real games: drive outcomes, points and plays per drive, points per game, final margins,
and how often a team leading by d with t left goes on to win.

Run:  .venv/bin/python scripts/validate_simulator.py     (needs scripts/solve_mdp.py first)
Writes results/m4_simulator_validation.json and assets/m4_simulator_validation.png.
The mismatches are the point: a later RL milestone may trust the simulator only this far.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from typing import Any  # noqa: E402

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from fourth_and_rl import data, mdp, style  # noqa: E402
from fourth_and_rl import mdp_game as mg  # noqa: E402
from fourth_and_rl.components import DRIVE_START_BINS  # noqa: E402
from fourth_and_rl.io import read_json, write_json  # noqa: E402
from fourth_and_rl.paths import ASSETS, RESULTS  # noqa: E402

OUT = RESULTS / "m4_simulator_validation.json"
FIG = ASSETS / "m4_simulator_validation.png"


def simulate_games(
    sim: mg.Simulator, n: int, policy: str, seed: int, checkpoints: list[int]
) -> dict:
    """n whole games from the opening kickoff."""
    rng = np.random.default_rng(seed)
    kick = sim.model.kern.kickoff
    yl = np.searchsorted(np.cumsum(kick[1:]), rng.random(n), side="right") + 1
    st = sim.start(n, t=sim.model.clock.n_ticks, s=0, f=sim.f10[np.minimum(yl, 99)], r=0)
    return mg.simulate(sim, st, rng, policy=policy, log_drives=True, checkpoints=checkpoints)


def drive_summary(drives: dict[str, np.ndarray], n_games: int) -> dict[str, Any]:
    def one(mask: np.ndarray) -> dict[str, Any]:
        k = int(mask.sum())
        share = {
            name: float((drives["result"][mask] == i).mean())
            for i, name in enumerate(mg.DRIVE_RESULTS)
        }
        counts = np.bincount(np.minimum(drives["plays"][mask], 20), minlength=21)
        return {
            "n_drives": k,
            "outcome_share": share,
            "points_per_drive": float(drives["points"][mask].mean()),
            "plays_per_drive": float(drives["plays"][mask].mean()),
            "mean_start_yardline_100": float(drives["yl"][mask].mean()),
            "plays_per_drive_distribution_0_to_20_plus": (counts / k).tolist(),
        }

    out = one(np.ones(len(drives["t"]), dtype=bool))
    out["by_start"] = {
        name: one((drives["yl"] >= lo) & (drives["yl"] < hi))
        for name, (lo, hi) in DRIVE_START_BINS.items()
    }
    out["drives_per_game"] = len(drives["t"]) / n_games
    out["scrimmage_plays_per_game"] = float(drives["plays"].sum()) / n_games
    return out


def real_game_table(
    plays: pl.DataFrame, first_season: int, checkpoints_s: list[int]
) -> pl.DataFrame:
    """One row per real game: regulation points, regulation margin (home), the home team's
    result (1, 0.5, 0), and the home margin at each checkpoint."""
    p = plays.filter(pl.col("season") >= first_season).sort("game_id", "order_sequence", "play_id")
    home = pl.col("is_home") == 1
    margin = (
        pl.when(home).then(pl.col("score_differential")).otherwise(-pl.col("score_differential"))
    )
    total = pl.col("posteam_score") + pl.col("defteam_score")
    scored = p.filter(pl.col("score_differential").is_not_null())
    games = scored.group_by("game_id", maintain_order=True).agg(
        pl.col("result").first(),
        (pl.col("home_score") + pl.col("away_score")).first().alias("final_total"),
        (pl.col("qtr").max() > 4).alias("overtime"),
        total.filter(pl.col("qtr") > 4).first().alias("regulation_total_ot"),
        *[
            margin.filter(pl.col("game_seconds_remaining") <= c).first().alias(f"m_{c}")
            for c in checkpoints_s
        ],
    )
    return games.with_columns(
        pl.when(pl.col("overtime"))
        .then(pl.col("regulation_total_ot"))
        .otherwise(pl.col("final_total"))
        .alias("regulation_total"),
        pl.when(pl.col("overtime")).then(0).otherwise(pl.col("result")).alias("regulation_margin"),
        pl.when(pl.col("result") > 0)
        .then(1.0)
        .when(pl.col("result") < 0)
        .then(0.0)
        .otherwise(0.5)
        .alias("home_result"),
    )


def lead_table(margin: np.ndarray, result: np.ndarray, bins: list[list[int]]) -> list[dict]:
    """Win rate of the team in front (a tie after regulation counts 0.5 in the simulator;
    real overtime games count as they ended, which has the same expectation)."""
    ok = ~np.isnan(margin)
    lead = np.abs(margin[ok])
    leader = np.where(margin[ok] > 0, result[ok], 1.0 - result[ok])
    rows = []
    for lo, hi in bins:
        m = (lead >= lo) & (lead <= hi)
        k = int(m.sum())
        rate = float(leader[m].mean()) if k else None
        se = float(leader[m].std() / np.sqrt(k)) if k else None
        rows.append(
            {
                "lead": f"{lo}-{hi}" if hi < 99 else f"{lo}+",
                "n": k,
                "leader_win_rate": rate,
                "standard_error": se,
            }
        )
    tied = margin[ok] == 0
    rows.append(
        {"lead": "0", "n": int(tied.sum()), "leader_win_rate": None, "standard_error": None}
    )
    return rows


def game_summary(total: np.ndarray, margin: np.ndarray) -> dict[str, Any]:
    a = np.abs(margin)
    return {
        "n_games": int(len(total)),
        "points_per_game_mean": float(total.mean()),
        "points_per_game_sd": float(total.std()),
        "points_per_game_quantiles_10_50_90": np.quantile(total, [0.1, 0.5, 0.9]).tolist(),
        "margin_sd": float(margin.std()),
        "abs_margin_mean": float(a.mean()),
        "share_tied_after_regulation": float((a == 0).mean()),
        "share_decided_by_1_to_3": float(((a >= 1) & (a <= 3)).mean()),
        "share_decided_by_4_to_8": float(((a >= 4) & (a <= 8)).mean()),
        "share_decided_by_9_to_16": float(((a >= 9) & (a <= 16)).mean()),
        "share_decided_by_17_plus": float((a >= 17).mean()),
    }


def compare(sim: dict, real: dict, keys: list[str]) -> dict[str, dict[str, float]]:
    return {k: {"simulated": sim[k], "real": real[k], "difference": sim[k] - real[k]} for k in keys}


def figure(res: dict, sim_games: dict, real_games: pl.DataFrame) -> None:
    style.apply()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.6), gridspec_kw={"hspace": 0.5, "wspace": 0.24})
    real_c, sim_c = style.MUTED, style.BLUE
    labels = {
        "touchdown": "Touchdown",
        "field_goal": "Field goal",
        "punt": "Punt",
        "turnover": "Turnover",
        "downs": "On downs",
        "missed_field_goal": "Missed FG",
        "end_of_half": "End of half",
    }
    d = res["drives"]
    ax = axes[0, 0]
    x = np.arange(len(labels))
    ax.bar(
        x - 0.2,
        [100 * d["real"]["outcome_share"][k] for k in labels],
        0.38,
        color=real_c,
        label="Real drives, 2021 to 2025",
    )
    ax.bar(
        x + 0.2,
        [100 * d["simulated"]["outcome_share"][k] for k in labels],
        0.38,
        color=sim_c,
        label="Simulator, coach-like policy",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels.values(), rotation=30, ha="right")
    ax.set_ylabel("Share of drives (%)")
    ax.set_ylim(0, 46)
    ax.legend(fontsize=9, loc="upper right")
    style.title(ax, "How drives end")

    ax = axes[0, 1]
    names = list(DRIVE_START_BINS)
    nice = ["Own\n1-20", "Own\n21-40", "Own 41 to\nopp 41", "Opp\n40-21", "Opp 20\nto goal"]
    x = np.arange(len(names))
    ax.bar(
        x - 0.2,
        [d["real"]["by_start"][k]["points_per_drive"] for k in names],
        0.38,
        color=real_c,
        label="Real",
    )
    ax.bar(
        x + 0.2,
        [d["simulated"]["by_start"][k]["points_per_drive"] for k in names],
        0.38,
        color=sim_c,
        label="Simulator",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(nice)
    ax.set_ylabel("Points per drive")
    ax.legend(fontsize=9)
    style.title(ax, "Points per drive by starting field position")

    ax = axes[0, 2]
    x = np.arange(21)
    ax.plot(
        x,
        100 * np.array(d["real"]["plays_per_drive_distribution_0_to_20_plus"]),
        color=real_c,
        label="Real",
        marker="o",
        markersize=3,
    )
    ax.plot(
        x,
        100 * np.array(d["simulated"]["plays_per_drive_distribution_0_to_20_plus"]),
        color=sim_c,
        label="Simulator",
        marker="o",
        markersize=3,
    )
    ax.set_xlabel("Plays in the drive (20 = 20 or more)")
    ax.set_ylabel("Share of drives (%)")
    ax.legend(fontsize=9)
    style.title(ax, "Plays per drive")

    ax = axes[1, 0]
    bins = np.arange(0, 101, 5)
    real_tot = real_games["regulation_total"].to_numpy().astype(float)
    sim_tot = sim_games["points"].sum(axis=1)
    for vals, color, lab in ((real_tot, real_c, "Real"), (sim_tot, sim_c, "Simulator")):
        h, _ = np.histogram(vals, bins=bins)
        ax.plot(bins[:-1] + 2.5, 100 * h / h.sum(), color=color, label=lab)
    ax.set_xlabel("Total points after regulation")
    ax.set_ylabel("Share of games (%)")
    ax.legend(fontsize=9)
    style.title(ax, "Points per game")

    ax = axes[1, 1]
    bins = np.arange(-0.5, 42.5, 3)
    real_m = np.abs(real_games["regulation_margin"].to_numpy().astype(float))
    sim_m = np.abs(sim_games["margin"]).astype(float)
    for vals, color, lab in ((real_m, real_c, "Real"), (sim_m, sim_c, "Simulator")):
        h, _ = np.histogram(np.minimum(vals, 41), bins=bins)
        ax.plot(
            bins[:-1] + 1.5, 100 * h / h.sum(), color=color, label=lab, marker="o", markersize=3
        )
    ax.set_xlabel("Margin after regulation (3-point bins; overtime games are 0)")
    ax.set_ylabel("Share of games (%)")
    ax.legend(fontsize=9)
    style.title(ax, "Final margin")

    ax = axes[1, 2]
    cps = res["leader_win_rate"]
    shades = [style.SEQUENTIAL[i] for i in (1, 3, 4, 6)]
    names = list(cps)
    for j, lead in enumerate(r["lead"] for r in cps[names[0]]["simulated"][:-1]):
        xs = np.arange(len(names))
        real_y = [100 * cps[c]["real"][j]["leader_win_rate"] for c in names]
        sim_y = [100 * cps[c]["simulated"][j]["leader_win_rate"] for c in names]
        ax.plot(xs, sim_y, color=shades[j], label=f"Lead {lead}")
        ax.scatter(xs, real_y, color=shades[j], edgecolor=style.INK, linewidths=0.6, s=28, zorder=3)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([f"{int(c) // 60}:00" for c in names])
    ax.set_xlabel("Time left in the game")
    ax.set_ylabel("Team in front goes on to win (%)")
    ax.set_ylim(30, 101)
    ax.legend(
        fontsize=9,
        title="Lines: simulator. Dots: real games",
        title_fontsize=8.5,
        loc="lower right",
        ncol=2,
    )
    style.title(ax, "Does a lead hold as often as in real games?")
    fig.suptitle(
        "Simulator against real football",
        x=0.125,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=style.INK,
    )
    style.save(fig, FIG)


def main() -> None:
    cfg = mdp.load_config()
    vcfg = cfg["validation"]
    sol = mg.load_solution()
    bundle = joblib.load(mg.SOLUTION_DIR / "tables.joblib")
    sim = mg.Simulator(sol, coach=bundle["coach"])
    tick = sol.model.clock.tick_seconds
    cps_s = vcfg["checkpoints_seconds"]
    cps_t = [c // tick for c in cps_s]
    n = vcfg["n_games"]
    targets = read_json(RESULTS / "m1_components.json")["drive_targets"]["recent_seasons"]
    first_season = targets["seasons"][0]
    plays = data.load_plays()
    real_games = real_game_table(plays, first_season, cps_s)
    real_summary = game_summary(
        real_games["regulation_total"].to_numpy().astype(float),
        real_games["regulation_margin"].to_numpy().astype(float),
    )
    clock_real = read_json(RESULTS / "research" / "endgame_clock_stats.json")

    out: dict[str, Any] = {
        "seed": vcfg["seed"],
        "n_simulated_games": n,
        "real_data": f"seasons {first_season} to {targets['seasons'][1]}; drive targets from "
        "results/m1_components.json; games from the play-by-play cache",
        "policy_note": "coach = fourth downs sampled from observed coach behavior by "
        "situation, measured timeout rates, simple end-of-half kicking rules; "
        "optimal = the solved policy of the game model",
    }
    by_policy = {}
    for policy in ("coach", "optimal"):
        g = simulate_games(sim, n, policy, vcfg["seed"], cps_t)
        by_policy[policy] = g
        total, margin = g["points"].sum(axis=1).astype(float), g["margin"].astype(float)
        out[f"games_{policy}"] = game_summary(total, margin)
        out[f"drives_{policy}"] = drive_summary(g["drives"], n)
        out[f"late_trailing_drives_{policy}"] = mg.late_trailing_drive_stats(
            g["drives"], g["result"], tick
        )
    g = by_policy["coach"]
    keys = [
        "points_per_game_mean",
        "points_per_game_sd",
        "margin_sd",
        "abs_margin_mean",
        "share_tied_after_regulation",
        "share_decided_by_1_to_3",
        "share_decided_by_4_to_8",
        "share_decided_by_9_to_16",
        "share_decided_by_17_plus",
    ]
    out["games"] = compare(out["games_coach"], real_summary, keys)
    out["games_real"] = real_summary
    sim_d = out["drives_coach"]
    out["drives"] = {
        "simulated": sim_d,
        "real": targets,
        "headline": compare(
            sim_d,
            targets,
            [
                "points_per_drive",
                "plays_per_drive",
                "drives_per_game",
                "scrimmage_plays_per_game",
                "mean_start_yardline_100",
            ],
        ),
        "outcome_share": compare(
            sim_d["outcome_share"], targets["outcome_share"], list(mg.DRIVE_RESULTS)
        ),
        "points_per_drive_by_start": {
            k: {
                "simulated": sim_d["by_start"][k]["points_per_drive"],
                "real": targets["by_start"][k]["points_per_drive"],
                "difference": sim_d["by_start"][k]["points_per_drive"]
                - targets["by_start"][k]["points_per_drive"],
            }
            for k in DRIVE_START_BINS
        },
        "touchdown_share_by_start": {
            k: {
                "simulated": sim_d["by_start"][k]["outcome_share"]["touchdown"],
                "real": targets["by_start"][k]["outcome_share"]["touchdown"],
            }
            for k in DRIVE_START_BINS
        },
    }
    out["leader_win_rate"] = {}
    home = real_games["home_result"].to_numpy()
    for c_s, c_t in zip(cps_s, cps_t, strict=True):
        real_m = real_games[f"m_{c_s}"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        out["leader_win_rate"][str(c_s)] = {
            "simulated": lead_table(g["checkpoint_margin"][c_t], g["result"], vcfg["lead_bins"]),
            "real": lead_table(real_m, home, vcfg["lead_bins"]),
            "simulated_optimal_policy": lead_table(
                by_policy["optimal"]["checkpoint_margin"][c_t],
                by_policy["optimal"]["result"],
                vcfg["lead_bins"],
            ),
        }
    worst = []
    for c, tab in out["leader_win_rate"].items():
        for s_row, r_row in zip(tab["simulated"][:-1], tab["real"][:-1], strict=True):
            worst.append(
                {
                    "seconds_left": int(c),
                    "lead": s_row["lead"],
                    "simulated": s_row["leader_win_rate"],
                    "real": r_row["leader_win_rate"],
                    "difference": s_row["leader_win_rate"] - r_row["leader_win_rate"],
                    "real_n": r_row["n"],
                    "real_standard_error": r_row["standard_error"],
                }
            )
    worst.sort(key=lambda r: -abs(r["difference"]))
    out["leader_win_rate_largest_gaps"] = worst[:8]
    out["leader_win_rate_mean_abs_gap"] = float(np.mean([abs(r["difference"]) for r in worst]))
    real_late = clock_real["trailing_drives_down_4_to_8"]
    out["late_trailing_drives_real"] = {
        "source": "results/research/endgame_clock_stats.json (2010 to 2025)",
        "td_rate": real_late["td_rate"],
        "trailing_team_win_rate": real_late["trailing_team_win_rate"],
        "by_time": {
            r["time_bin"]: {"n": r["n"], "rate": r["td_rate"]} for r in real_late["by_time"]
        },
        "by_field_position": {
            r["field_bin"]: {"n": r["n"], "rate": r["td_rate"]}
            for r in real_late["by_field_position"]
        },
        "td_drive_seconds_used_median": real_late["td_drive_seconds_used"]["median"],
        "td_drive_plays_median": real_late["td_drive_plays"]["median"],
        "after_trailing_touchdown": {
            "leader_gets_ball_in_regulation": real_late["after_trailing_touchdown"][
                "leader_gets_ball_in_regulation"
            ]["rate"],
            "seconds_left_at_answer_drive_median": real_late["after_trailing_touchdown"][
                "seconds_left_at_answer_drive"
            ]["median"],
            "share_with_30s_or_more": real_late["after_trailing_touchdown"][
                "share_with_30s_or_more"
            ]["rate"],
            "share_with_60s_or_more": real_late["after_trailing_touchdown"][
                "share_with_60s_or_more"
            ]["rate"],
            "original_leader_win_rate": real_late["after_trailing_touchdown"][
                "original_leader_win_rate"
            ],
        },
        "answer_drive_needing_points": {
            "n": real_late["answer_drive_needing_points"]["n"],
            "scores_td_or_fg": real_late["answer_drive_needing_points"]["scores_td_or_fg"]["rate"],
            "original_leader_win_rate": real_late["answer_drive_needing_points"][
                "original_leader_win_rate"
            ],
        },
        "after_no_touchdown": real_late["after_no_touchdown"],
    }
    write_json(out, OUT)
    figure(out, g, real_games)
    print("wrote", OUT, "and", FIG)
    for k, v in out["games"].items():
        print(f"  {k:32s} sim {v['simulated']:.3f}  real {v['real']:.3f}")
    for k, v in out["drives"]["headline"].items():
        print(f"  {k:32s} sim {v['simulated']:.3f}  real {v['real']:.3f}")
    for k, v in out["drives"]["outcome_share"].items():
        print(f"  {k:32s} sim {v['simulated']:.3f}  real {v['real']:.3f}")


if __name__ == "__main__":
    main()
