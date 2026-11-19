"""M7 coach cost ledger, penalty half: procedure penalties priced in win probability, and the
test of the owner's assumption that false starts and delays of game are on the coach.

Penalty types are not in the state table: they are read from the nflverse play-by-play
(`coaches.download_penalties`, cached under data/m7_penalties/) and joined by game and play
id. Each false start and delay of game by the offense, except a delay taken on purpose
before a punt or field goal, is priced as the win-probability change between the state
with the foul and the same state without its yardage, using the version 2 model that
never saw the season (fold models for development seasons, the headline model for the
test seasons; the fold models are shared with scripts/m7_coach_evaluation.py).

The assumption test (year-to-year and split-half stability, coaches who change teams, and
a Poisson model with crossed random effects for coach, franchise and team-season) is
descriptive and uses every season from 2011 to 2025; nothing in it is a prediction scored
on the test seasons. Its verdict rule is in configs/coaches.yaml and was written before
the penalty data was read.

Writes results/m7_penalty_ledger.json and data/processed/m7_penalties_priced.parquet.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_var, "3")

import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from fourth_and_rl import coaches, decision, splits  # noqa: E402
from fourth_and_rl import wp as wp_mod  # noqa: E402
from fourth_and_rl.io import write_json  # noqa: E402
from fourth_and_rl.paths import DATA, RESULTS, ROOT  # noqa: E402

TEST_PURPOSE = (
    "M7 penalty ledger: price 2024-2025 false starts and delays of game with the headline "
    "development win-probability model, and the descriptive assumption test"
)
PRICED_PATH = DATA / "processed" / "m7_penalties_priced.parquet"
OUT = RESULTS / "m7_penalty_ledger.json"
COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "desc",
    "order_sequence", "play_type", "roof", *wp_mod.STATE_COLUMNS,
]  # fmt: skip


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_inputs(cfg: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(development plays, all graded-season plays with penalty columns, coach lookup)."""
    lo, hi = cfg["grade_seasons"]
    seasons = list(range(lo, hi + 1))
    pcfg = cfg["penalties"]
    cache = ROOT / pcfg["cache_dir"]
    files = coaches.download_penalties(seasons, cache)
    pen = pl.concat([pl.read_parquet(p) for p in files], how="vertical_relaxed")
    sched_path = coaches.download_schedules(list(range(2010, hi + 1)), cache / "schedules.parquet")
    lookup = coaches.coach_lookup(pl.read_parquet(sched_path))
    dev = splits.load_dev_plays()
    test = splits.load_test_plays(TEST_PURPOSE)
    plays = pl.concat([dev.select(COLUMNS), test.select(COLUMNS)], how="vertical_relaxed")
    plays = plays.filter(pl.col("season").is_between(lo, hi))
    n = plays.height
    plays = plays.join(pen.drop("penalty"), on=["game_id", "play_id"], how="left")
    if plays.height != n:
        raise ValueError("penalty columns are not unique per (game_id, play_id)")
    plays = coaches.attach_coaches(plays, lookup)
    return dev, plays, lookup


def price_fouls(fouls: pl.DataFrame, dev: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Price every regulation foul with the model of its season's fold."""
    dcfg, wcfg = decision.load_config(), wp_mod.load_config()
    plan = coaches.fold_plan(splits.dev_seasons(), splits.test_seasons(),
                             range(cfg["grade_seasons"][0], cfg["grade_seasons"][1] + 1),
                             int(cfg["folds"]["fold_size"]))  # fmt: skip
    coaches.check_fold_plan(plan, splits.test_seasons(), splits.holdout_seasons())
    parts = []
    for fold in plan:
        rows = fouls.filter(pl.col("season").is_in(list(fold.grade)))
        if fold.headline:
            model = wp_mod.WPModel.load(ROOT / dcfg["models"]["wp_path"])
        else:
            model, _ = coaches.ensure_fold_models(fold, dev, ROOT / cfg["folds"]["models_dir"],
                                                  wcfg, None, int(cfg["n_threads"]))  # fmt: skip
        splits.assert_can_score(model.stamp, rows, f"M7 penalty pricing {fold.name}")
        with_, without = coaches.penalty_states(rows)
        cost = coaches.price_states(model, with_, without)
        parts.append(rows.with_columns(
            pl.Series("wp_cost", cost).fill_nan(None), pl.lit(fold.name).alias("fold"),
            with_["yardline_100"].alias("yardline_after"), with_["ydstogo"].alias("ydstogo_after"),
        ))  # fmt: skip
    return pl.concat(parts).sort("season", "game_id", "order_sequence")


def foul_summary(labelled: pl.DataFrame, priced: pl.DataFrame, cfg: dict) -> dict[str, Any]:
    """Counts by type, the deliberate-delay exclusions and the price per foul."""
    fouls = labelled.filter(pl.col("foul_type").is_not_null())
    formation = pl.lit(False)
    for text in cfg["penalties"]["deliberate_formations"]:
        formation = formation | pl.col("desc").fill_null("").str.contains(text, literal=True)
    delib = fouls.filter(pl.col("deliberate"))
    reg = priced.filter(pl.col("coach_foul"))
    by_type = {}
    for t in ("false_start", "delay_of_game"):
        sub = reg.filter(pl.col("foul_type") == t)
        by_type[t] = {
            "count": sub.height,
            "mean_wp_cost_points": 100 * float(sub["wp_cost"].mean()),
            "median_wp_cost_points": 100 * float(sub["wp_cost"].median()),
            "total_wins": float(sub["wp_cost"].sum()),
            "share_on_fourth_down": float((sub["down"] == 4).mean()),
        }
    seasons = reg.group_by("season").agg(
        pl.len().alias("fouls"), pl.col("wp_cost").sum().alias("wins"),
        (pl.col("foul_type") == "false_start").sum().alias("false_starts"),
        (pl.col("foul_type") == "delay_of_game").sum().alias("delays"),
    ).sort("season")  # fmt: skip
    fg_formation = delib.filter(pl.col("desc").str.contains("Field Goal formation", literal=True))
    names = cfg["penalties"]["coach_types"]
    other = labelled.filter(
        pl.col("penalty_type").is_in(names) & (pl.col("penalty_team") == pl.col("posteam"))
        & pl.col("foul_type").is_null()
    )  # fmt: skip
    return {
        "same_types_not_counted": {
            "try_or_kick_no_down": other.filter(pl.col("down").is_null()).height,
            "after_a_play_that_ran": other.filter(pl.col("down").is_not_null()
                                                  & (pl.col("play_type") != "no_play")).height,
            "zero_yards": other.filter(pl.col("down").is_not_null()
                                       & (pl.col("play_type") == "no_play")).height,
        },
        "offense_fouls_of_the_two_types": fouls.height,
        "deliberate_delays_excluded": delib.height,
        "deliberate_by_rule": {
            "punt_or_field_goal_formation": delib.filter(formation).height,
            "fourth_down_then_punt_or_field_goal_only": delib.filter(~formation).height,
            "of_which_field_goal_formation": fg_formation.height,
        },
        "coach_fouls": int(fouls["coach_foul"].sum()),
        "coach_fouls_in_overtime_not_priced": int(
            fouls.filter(pl.col("coach_foul") & (pl.col("qtr") > 4)).height),
        "coach_fouls_priced": reg.height,
        "coach_fouls_in_last_minute_of_a_half": int(
            reg.filter(pl.col("half_seconds_remaining") <= 60).height),
        "by_type": by_type,
        "by_season": seasons.to_dicts(),
        "sensitivity_count_field_goal_formation_delays_as_coach_fouls": {
            "extra_fouls": fg_formation.height,
        },
    }  # fmt: skip


def team_game_units(snaps: pl.DataFrame, flag: str, lookup: pl.DataFrame,
                    seasons: list[int]) -> pl.DataFrame:  # fmt: skip
    """One row per offense team-game: fouls, expected fouls, coach, home, order."""
    s, _ = coaches.expected_counts(snaps, flag)
    units = s.group_by("game_id", "posteam").agg(
        pl.col(flag).cast(pl.Float64).sum().alias("fouls"), pl.col("expected").sum(),
        pl.col("season").first(), pl.col("week").first(), pl.col("is_home").first(),
        pl.col("coach").first(), pl.len().alias("snaps"),
    )  # fmt: skip
    return units.with_columns(
        (pl.col("season") * 100 + pl.col("week")).alias("order"),
        pl.concat_str([pl.col("posteam"), pl.col("season").cast(pl.String)], separator="_")
        .alias("team_season"),
    ).filter(pl.col("season").is_in(seasons)).sort("order", "game_id", "posteam")


def season_units(units: pl.DataFrame) -> pl.DataFrame:
    """Team-seasons with the coach of most of their games."""
    coach = units.group_by("posteam", "season", "coach").agg(pl.len().alias("n")).sort(
        "n", descending=True).unique(["posteam", "season"], keep="first")  # fmt: skip
    tot = units.group_by("posteam", "season").agg(pl.col("fouls").sum(), pl.col("expected").sum())
    return tot.join(coach.select("posteam", "season", "coach"), on=["posteam", "season"]).rename(
        {"posteam": "team"})  # fmt: skip


def codes(x: pl.Series) -> np.ndarray:
    return x.cast(pl.String).rank("dense").cast(pl.Int64).to_numpy() - 1


def mixed_model(units: pl.DataFrame, cfg: dict) -> dict[str, Any]:
    """Crossed random effects for coach, franchise and team-season, with a profile interval
    for the coach standard deviation, and the alternative models without the coach."""
    mm = cfg["penalties"]["mixed_model"]
    y = units["fouls"].to_numpy()
    e = units["expected"].to_numpy()
    groups = {"coach": codes(units["coach"]), "franchise": codes(units["posteam"]),
              "team_season": codes(units["team_season"])}  # fmt: skip
    full = coaches.fit_poisson_crossed(y, e, groups, mm["start_log_var"])
    no_coach = coaches.fit_poisson_crossed(y, e, {k: v for k, v in groups.items()
                                                  if k != "coach"}, mm["start_log_var"])
    prof = coaches.profile_variance(y, e, groups, "coach", mm["profile_log_var"],
                                    max(full["logml"], -np.inf))  # fmt: skip
    var = full["var"]
    systematic = sum(var.values())
    lam = float(np.mean(e))
    # On the log scale, a Poisson count with mean lam adds about log(1 + 1/lam) of noise
    # variance (the lognormal-Poisson approximation used for GLMM variance shares).
    noise = float(np.log1p(1.0 / lam))
    names = units["coach"].cast(pl.String).unique().sort().to_list()
    eff = dict(zip(names, full["effects"]["coach"].tolist(), strict=True))
    return {
        "observations": "offense team-games",
        "n_team_games": int(units.height),
        "n_coaches": len(names),
        "n_team_seasons": int(units["team_season"].n_unique()),
        "mean_expected_fouls_per_team_game": lam,
        "sd": full["sd"],
        "var": var,
        "share_of_systematic_variance": {k: v / systematic for k, v in var.items()},
        "share_of_total_variance_on_log_scale": {k: v / (systematic + noise)
                                                 for k, v in var.items()},
        "poisson_noise_variance_log_scale": noise,
        "logml": full["logml"],
        "logml_without_coach": no_coach["logml"],
        "likelihood_ratio_stat_for_coach": 2 * (full["logml"] - no_coach["logml"]),
        "coach_sd_profile": prof,
        "coach_rate_ratio_one_sd": float(np.exp(full["sd"]["coach"])),
        "coach_effects": eff,
        "method": "Poisson, crossed Gaussian random effects, Laplace approximation of the "
                  "marginal likelihood (coaches.fit_poisson_crossed); expected counts from "
                  "league rates by season, home or road, dome and third-or-fourth-and-long",
    }  # fmt: skip


def coach_penalty_table(priced: pl.DataFrame, games: pl.DataFrame, effects: dict[str, float],
                        delay_effects: dict[str, float], cfg: dict) -> list[dict[str, Any]]:
    """Per coach: penalty count and cost in wins, with a game bootstrap, and the part of
    the cost attributable to the coach effect (relative to an average coach), for both
    foul types together and for delays of game alone.

    `games` has one row per team-game of the graded seasons (game_id, is_home, coach), so
    games without a foul count as zeros. Fouls are matched to it by game and side."""
    tcfg = cfg["tables"]
    dog = pl.col("foul_type") == "delay_of_game"
    per = priced.filter(pl.col("coach_foul")).with_columns(
        pl.col("is_home").cast(pl.Int8)).group_by("game_id", "is_home").agg(
        pl.len().alias("fouls"), pl.col("wp_cost").sum().alias("cost"),
        dog.sum().alias("delays"), pl.col("wp_cost").filter(dog).sum().alias("delay_cost"))
    g = games.join(per, on=["game_id", "is_home"], how="left").with_columns(
        [pl.col(c).fill_null(0).cast(pl.Float64) for c in ("fouls", "cost", "delays",
                                                            "delay_cost")])  # fmt: skip
    if g["fouls"].sum() != priced["coach_foul"].sum():
        raise ValueError("some priced fouls did not match a team-game")
    boot = coaches.group_bootstrap(g["coach"].to_numpy(), g["game_id"].to_numpy(),
                                   {"fouls": g["fouls"].to_numpy(), "cost": g["cost"].to_numpy(),
                                    "delay_cost": g["delay_cost"].to_numpy(),
                                    "games": np.ones(g.height)},
                                   int(tcfg["n_boot"]), int(cfg["seed"]) + 7)  # fmt: skip
    iv = tcfg["interval"]
    per17 = float(tcfg["games_per_season"])
    rows = []
    for coach, b in boot.items():
        n = b["n_games"]
        cost = coaches.ratio_summary(b["point"]["cost"], 1.0, b["boot"]["cost"],
                                     np.ones(int(tcfg["n_boot"])), iv)  # fmt: skip
        u = effects.get(coach, 0.0)
        ud = delay_effects.get(coach, 0.0)
        dc = coaches.ratio_summary(b["point"]["delay_cost"], 1.0, b["boot"]["delay_cost"],
                                   np.ones(int(tcfg["n_boot"])), iv)  # fmt: skip
        rows.append({
            "coach": coach, "games": n, "fouls": int(b["point"]["fouls"]),
            "fouls_per_game": b["point"]["fouls"] / n,
            "cost_wins": cost["est"], "cost_wins_lo": cost["lo"], "cost_wins_hi": cost["hi"],
            "cost_wins_per_17_games": cost["est"] * per17 / n,
            "coach_effect_rate_ratio": float(np.exp(u)),
            "cost_attributable_to_coach_wins": cost["est"] * (1.0 - float(np.exp(-u))),
            "delay_cost_wins": dc["est"], "delay_cost_wins_lo": dc["lo"],
            "delay_cost_wins_hi": dc["hi"],
            "delay_cost_attributable_to_coach_wins": dc["est"] * (1.0 - float(np.exp(-ud))),
        })  # fmt: skip
    return sorted(rows, key=lambda r: -r["cost_wins"])


def main() -> None:
    cfg = coaches.load_config()
    pcfg = cfg["penalties"]
    lo, hi = cfg["grade_seasons"]
    seasons = list(range(lo, hi + 1))
    log("loading plays, penalty columns and coaches")
    dev, plays, lookup = load_inputs(cfg)  # plays: the graded seasons only
    labelled = coaches.coach_fouls(coaches.next_snap(plays), pcfg)
    fouls = labelled.filter(pl.col("foul_type").is_not_null() & (pl.col("qtr") <= 4))
    log(f"{fouls.height} offense false starts and delays of game in regulation; pricing")
    priced = price_fouls(fouls, dev, cfg)
    del dev
    PRICED_PATH.parent.mkdir(parents=True, exist_ok=True)
    priced.write_parquet(PRICED_PATH)
    summary = foul_summary(labelled, priced, cfg)

    log("assumption test")
    snaps = coaches.snap_rows(labelled, int(pcfg["long_distance"]))
    snaps = snaps.with_columns(
        (pl.col("coach_foul") & (pl.col("foul_type") == "false_start")).alias("fs"),
        (pl.col("coach_foul") & (pl.col("foul_type") == "delay_of_game")).alias("dog"),
    )
    _, cells = coaches.expected_counts(snaps, "coach_foul")
    n_boot = int(cfg["tables"]["n_boot"])
    tests: dict[str, Any] = {"situation_rate_ratios": coaches.rate_ratios(cells)}
    effects: dict[str, dict[str, float]] = {}
    for flag, name in (("coach_foul", "both"), ("fs", "false_start"), ("dog", "delay_of_game")):
        units = team_game_units(snaps, flag, lookup, seasons)
        su = season_units(units)
        res = {
            "year_to_year": coaches.rate_correlations(su, n_boot, int(cfg["seed"])),
            "split_half_coach_career": coaches.split_half(
                units["coach"].to_numpy(), units["order"].to_numpy(),
                units["fouls"].to_numpy(), units["expected"].to_numpy(), n_boot,
                int(cfg["seed"])),
            "split_half_team_season": coaches.split_half(
                units["team_season"].to_numpy(), units["order"].to_numpy(),
                units["fouls"].to_numpy(), units["expected"].to_numpy(), n_boot,
                int(cfg["seed"])),
            "movers_by_stint": {
                **coaches.mover_test(su, n_boot, int(cfg["seed"])),
                "note": "added after the pre-registered consecutive-season mover check "
                        "(year_to_year.coach_moved_to_new_team) found only 7 pairs; not part "
                        "of the verdict rule",
            },
            "mixed_model": mixed_model(units, cfg),
        }  # fmt: skip
        eff = res["mixed_model"].pop("coach_effects")
        effects[name] = eff
        tests[name] = res
        log(f"{name}: coach sd {res['mixed_model']['sd']['coach']:.3f}, share "
            f"{res['mixed_model']['share_of_systematic_variance']['coach']:.2f}")

    both = tests["both"]
    mm = both["mixed_model"]
    yy = both["year_to_year"]
    checks = {
        "coach_sd_interval_excludes_zero": not mm["coach_sd_profile"]["lower_end_at_grid_bottom"],
        "coach_share_at_least_threshold":
            mm["share_of_systematic_variance"]["coach"] >= pcfg["verdict"]["min_coach_share"],
        "rate_follows_the_coach": bool(yy["coach_moved_to_new_team"]["r"]
                                       >= yy["same_team_new_coach"]["r"]),
    }  # fmt: skip
    supported = all(checks.values())
    games = lookup.join(plays.select("game_id").unique(), on="game_id", how="inner").select(
        "game_id", pl.col("is_home_s").cast(pl.Int8).alias("is_home"), "coach")
    table = coach_penalty_table(priced, games, effects["both"], effects["delay_of_game"], cfg)
    out = {
        "what": "Procedure penalties the owner assigns to the head coach (false start and "
                "delay of game on offense, minus deliberate delays before a punt or field "
                "goal), priced in win probability with cross-fitted version 2 models, and "
                "the test of whether a coach effect exists.",
        "seasons": [lo, hi],
        "assignment_rule": {
            "assigned_to_coach": pcfg["coach_types"],
            "never_assigned": pcfg["not_coach_types"],
            "deliberate_delay_rule": "lined up in punt or field-goal formation, or fourth "
                                     "down with the offense's next snap a punt or field goal",
        },
        "pricing": "WP(state without the foul) - WP(state with it): same down, clock, score "
                   "and timeouts; the ball moved back by the enforced yards and the distance "
                   "lengthened by the same (goal to go: distance = yard line). Development "
                   "seasons priced by fold models that never saw them; 2024-2025 by the "
                   "headline model. The 10-second runoff option after a late false start is "
                   "not priced.",
        "fouls": summary,
        "assumption_test": tests,
        "verdict": {"rule": "configs/coaches.yaml penalties.verdict (written before the "
                            "data was read)", "checks": checks,
                    "assigning_penalties_to_the_coach_is_supported": supported},
        "coaches": table,
        "notes": [
            "Intervals are 90% bootstrap intervals over the coach's games.",
            "The coach effect, if any, includes his staff (offensive line coach, play-caller's "
            "tempo), which the data cannot separate from the head coach.",
            "Only the first penalty listed on a play is read (nflverse penalty_type).",
        ],
    }  # fmt: skip
    write_json(out, OUT)
    log(f"wrote {OUT}; supported={supported}")


if __name__ == "__main__":
    main()
