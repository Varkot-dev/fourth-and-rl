"""M1: fit the play-outcome models, choose their form, check them once, save everything.

Leakage discipline (docs/research/leakage_audit.md). The test seasons never reach a fit
or a choice:
  1. Model selection. Conversion and field-goal variants are compared by season-blocked
     validation INSIDE the development seasons (fit on the seasons before a block, score
     the block). The adopted variants are the ones in configs/components.yaml; the script
     records whether a challenger beat them by more than noise.
  2. Development bundle. Every model is fitted on the development seasons only. The
     kickoff tables of the 2024 and 2025 rules are the one logged exception (a rule
     constant, fitted on that season's own kickoffs).
  3. Held-out check. The development bundle is scored ONCE on the test seasons, loaded
     through `splits.load_test_plays(purpose=...)`, after `splits.assert_can_score`.
  4. All-seasons bundle. A second bundle is fitted on every season through the last test
     season with a logged exception. It is for forward-looking use only and nothing in
     this repo scores it.
  5. Kickoff sensitivity. How much the go-versus-kick comparison (in expected points)
     moves when the empirical kickoff table is replaced by a fixed touchback spot.

Outputs:
  data/models/components_dev.joblib   development bundle (not committed)
  data/models/components_all.joblib   all-seasons bundle, forward-looking only (not committed)
  results/m1_components.json          selection, fitted rates, held-out checks, drive targets
  assets/m1_components.png            conversion, field goal and punt fits against raw rates
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")  # the machine is shared

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from fourth_and_rl import components as cm  # noqa: E402
from fourth_and_rl import mdp, splits, style  # noqa: E402
from fourth_and_rl.io import write_json  # noqa: E402
from fourth_and_rl.mdp_drive import solve_drive  # noqa: E402
from fourth_and_rl.paths import ASSETS, RESULTS  # noqa: E402

YARDS = np.arange(100)


# --------------------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------------------


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Mean of -log(probability given to what happened). Lower is better."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(p: np.ndarray, y: np.ndarray, group: np.ndarray, labels: list[str]) -> list[dict]:
    rows = []
    for i, label in enumerate(labels):
        m = group == i
        if m.any():
            rows.append({"bin": label, "n": int(m.sum()), "predicted": float(p[m].mean()),
                         "actual": float(y[m].mean())})  # fmt: skip
    return rows


def binary_report(p: np.ndarray, y: np.ndarray, base_rate: float) -> dict:
    return {
        "n": int(y.size),
        "actual_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "log_loss_model": log_loss(p, y),
        "log_loss_constant_baseline": log_loss(np.full(y.size, base_rate), y),
        "baseline_rate_from_training_seasons": float(base_rate),
    }


# --------------------------------------------------------------------------------------
# Model selection inside the development seasons
# --------------------------------------------------------------------------------------

# Each variant is a set of overrides on the reference settings. The adopted variant is
# whichever one reproduces configs/components.yaml.
GO_REFERENCE = {
    "fit_downs": [3, 4], "season_trend": True, "use_spread": True, "fourth_by_distance": False,
    "distance_knots": [1, 2, 3, 4, 6, 8, 10, 13, 17, 22, 30],
}  # fmt: skip
GO_VARIANTS = {
    "third_and_fourth_downs": {},
    "fourth_downs_only": {"fit_downs": [4]},
    "third_downs_only": {"fit_downs": [3]},
    "no_season_trend": {"season_trend": False},
    "no_spread_control": {"use_spread": False},
    "coarser_distance_knots": {"distance_knots": [1, 3, 6, 10, 15, 22, 30]},
    "fourth_down_effect_varies_with_distance": {"fourth_by_distance": True},
}
FG_REFERENCE = {"season_trend": True, "first_season": None}
FG_VARIANTS = {
    "all_seasons_with_trend": {},
    "all_seasons_no_trend": {"season_trend": False},
    "recent_seasons_only_no_trend": {"season_trend": False, "first_season": "era"},
}
FG_TAILS = (0, 3, 5)  # yards over which P(make) fades out beyond the longest made kick


def adopted_variant(current: dict, reference: dict, variants: dict) -> str:
    """The variant whose settings equal the config's."""
    now = {k: current.get(k, v) for k, v in reference.items()}
    for name, override in variants.items():
        if {**reference, **override} == now or (
            override.get("first_season") == "era"
            and {**reference, **override, "first_season": now["first_season"]} == now
            and now["first_season"] is not None
        ):
            return name
    raise ValueError(f"the config {now} matches none of the variants compared")


def fg_frame(plays: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(yardline_100, made, game id) of every field-goal attempt."""
    df = plays.filter(
        (pl.col("play_type") == "field_goal") & pl.col("yardline_100").is_between(1, 99)
        & pl.col("field_goal_result").is_not_null()
    )  # fmt: skip
    made = (df["field_goal_result"] == "made").to_numpy().astype(float)
    return df["yardline_100"].to_numpy().astype(int), made, df["game_id"].to_numpy()


def fourth_down_frame(plays: pl.DataFrame) -> dict[str, np.ndarray]:
    """Fourth-down runs and passes with the conversion label and the game id."""
    df = cm.add_next_state(plays).filter(
        pl.col("play_type").is_in(["run", "pass"]) & (pl.col("down") == 4)
        & pl.col("yardline_100").is_between(1, 99) & (pl.col("ydstogo") >= 1)
        & pl.col("yards_gained").is_not_null()
    )  # fmt: skip
    s = cm._scrimmage(df, 4, 4)
    s["converted"] = ((s["first_down"] | s["off_td"]) & ~s["turnover"]).astype(float)
    s["game"] = df["game_id"].to_numpy()
    assert s["game"].size == s["down"].size
    return s


def per_row_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_difference(a: np.ndarray, b: np.ndarray, game: np.ndarray) -> dict[str, float]:
    """Mean of a - b over rows, with a standard error that treats each game as one
    independent unit (rows of one game are not independent)."""
    d = a - b
    _, inv = np.unique(game, return_inverse=True)
    n_games = int(inv.max()) + 1
    by_game = np.bincount(inv, weights=d, minlength=n_games)
    size = np.bincount(inv, minlength=n_games)
    resid = by_game - size * d.mean()
    se = float(np.sqrt(n_games / (n_games - 1) * (resid**2).sum()) / d.size)
    return {"difference": float(d.mean()), "standard_error": se}


def summarize_selection(losses: dict[str, list[np.ndarray]], games: list[np.ndarray],
                        adopted: str, blocks: list[list[int]]) -> dict:  # fmt: skip
    """Per-block and pooled log loss of every variant, the winner, and whether the winner
    beats the adopted variant by more than one game-clustered standard error."""
    game = np.concatenate(games)
    pooled = {k: np.concatenate(v) for k, v in losses.items()}
    table = {
        k: {
            "by_block": {f"{lo}-{hi}": float(x.mean())
                         for (lo, hi), x in zip(blocks, losses[k], strict=True)},
            "pooled": float(pooled[k].mean()),
            "minus_adopted": paired_difference(pooled[k], pooled[adopted], game),
        }
        for k in losses
    }  # fmt: skip
    winner = min(pooled, key=lambda k: pooled[k].mean())
    edge = paired_difference(pooled[adopted], pooled[winner], game)
    beyond_noise = winner != adopted and edge["difference"] > edge["standard_error"]
    return {
        "n_rows": int(game.size),
        "n_games": int(np.unique(game).size),
        "variants": table,
        "lowest_pooled_log_loss": winner,
        "adopted": adopted,
        "adopted_minus_lowest": edge,
        "lowest_beats_adopted_by_more_than_one_standard_error": bool(beyond_noise),
        "block_winners": {
            f"{lo}-{hi}": min(losses, key=lambda k, i=i: losses[k][i].mean())
            for i, (lo, hi) in enumerate(blocks)
        },
    }


def model_selection(cfg: dict, dev: pl.DataFrame) -> dict:
    """Season-blocked validation inside the development seasons. No test season is read.

    For each block [lo, hi] every variant is fitted on the development seasons before lo
    and scored on the block's fourth downs (conversion) or kicks (field goal). The rule,
    fixed before looking at any test-season score: the variant in the config stays unless
    another variant has a lower pooled log loss by more than one standard error of the
    paired difference (games as units). A challenger that clears that bar must be adopted
    by changing the config, and the script stops until it is.
    """
    blocks = [list(b) for b in cfg["selection"]["validation_blocks"]]
    era = cfg["era_first_season"]
    go_adopted = adopted_variant(cfg["go"], GO_REFERENCE, GO_VARIANTS)
    fg_adopted = adopted_variant(cfg["field_goal"], FG_REFERENCE, FG_VARIANTS)
    tail_adopted = f"tail_{cfg['field_goal']['tail_yards']}_yards"
    go_loss: dict[str, list] = {k: [] for k in GO_VARIANTS}
    fg_loss: dict[str, list] = {k: [] for k in FG_VARIANTS}
    tail_loss: dict[str, list] = {f"tail_{t}_yards": [] for t in FG_TAILS}
    go_games, fg_games, caps = [], [], []
    for lo, hi in blocks:
        train = dev.filter(pl.col("season") < lo)
        valid = dev.filter(pl.col("season").is_between(lo, hi))
        splits.assert_disjoint_games(train, valid, "model-selection block")
        s = fourth_down_frame(valid)
        go_games.append(s["game"])
        for name, override in GO_VARIANTS.items():
            model = cm.GoModel(era_first_season=era, **{**cfg["go"], **override}).fit(train)
            go_loss[name].append(
                per_row_loss(model.p_convert(s["yardline"], s["ydstogo"]), s["converted"])
            )
        yl, made, game = fg_frame(valid)
        fg_games.append(game)
        for name, override in FG_VARIANTS.items():
            kw = {**cfg["field_goal"], **override}
            if kw.get("first_season") == "era":
                kw["first_season"] = era
            model = cm.FieldGoalModel(**kw).fit(train)
            fg_loss[name].append(per_row_loss(model.p_make(yl), made))
        for tail in FG_TAILS:
            model = cm.FieldGoalModel(**{**cfg["field_goal"], "tail_yards": tail}).fit(train)
            tail_loss[f"tail_{tail}_yards"].append(per_row_loss(model.p_make(yl), made))
        beyond = yl + model.snap_to_kick_yards > model.max_kick_distance
        caps.append({
            "block": f"{lo}-{hi}", "fitted_on": [int(train["season"].min()), lo - 1],
            "longest_made_kick_in_fitted_seasons": model.max_kick_distance,
            "block_attempts_beyond_it": int(beyond.sum()),
            "block_makes_beyond_it": int(made[beyond].sum()),
        })  # fmt: skip
    out = {
        "method": "season-blocked validation inside the development seasons: fit on every "
        "development season before the block, score the block; log loss pooled over blocks",
        "validation_blocks": blocks,
        "rule": "keep the configured variant unless another has a lower pooled log loss by "
        "more than one standard error of the paired difference (games as units)",
        "conversion_on_fourth_downs": summarize_selection(go_loss, go_games, go_adopted, blocks),
        "field_goal_form": summarize_selection(fg_loss, fg_games, fg_adopted, blocks),
        "field_goal_tail": {
            **summarize_selection(tail_loss, fg_games, tail_adopted, blocks),
            "longest_made_kick_by_block": caps,
            "note": "tail_0_yards is the hard cap at the longest made kick. The comparison "
            "rests on the handful of attempts beyond that kick in each block.",
        },
    }
    for key in ("conversion_on_fourth_downs", "field_goal_form", "field_goal_tail"):
        if out[key]["lowest_beats_adopted_by_more_than_one_standard_error"]:
            raise SystemExit(
                f"model selection ({key}): {out[key]['lowest_pooled_log_loss']} beats the "
                f"configured {out[key]['adopted']} by more than noise. Adopt it in "
                "configs/components.yaml and rerun."
            )
    return out


# --------------------------------------------------------------------------------------
# Held-out checks: the development bundle, scored once on the test seasons
# --------------------------------------------------------------------------------------


def go_frame(plays: pl.DataFrame) -> dict[str, np.ndarray]:
    s = cm._scrimmage(plays, 3, 4)
    s["converted"] = ((s["first_down"] | s["off_td"]) & ~s["turnover"]).astype(float)
    return s


def holdout_go(model: cm.GoModel, train: pl.DataFrame, test: pl.DataFrame) -> dict:
    tr, te = go_frame(train), go_frame(test)
    out: dict = {}
    m4 = te["down"] == 4
    p4 = model.p_convert(te["yardline"][m4], te["ydstogo"][m4])
    y4 = te["converted"][m4]
    base4 = tr["converted"][tr["down"] == 4].mean()
    out["fourth_down"] = binary_report(p4, y4, base4)
    d4 = te["ydstogo"][m4]
    out["fourth_down"]["calibration_by_distance"] = calibration(
        p4, y4, np.searchsorted([2, 3, 5, 8, 11], d4, side="right"),
        ["1", "2", "3-4", "5-7", "8-10", "11+"],
    )  # fmt: skip
    edges = np.quantile(p4, [0.2, 0.4, 0.6, 0.8])
    out["fourth_down"]["calibration_by_predicted_quintile"] = calibration(
        p4, y4, np.searchsorted(edges, p4, side="right"),
        ["lowest fifth", "second", "third", "fourth", "highest fifth"],
    )  # fmt: skip
    m3 = ~m4
    p3 = model.p_convert(te["yardline"][m3], te["ydstogo"][m3], down=3)
    out["third_down_same_model"] = binary_report(
        p3, te["converted"][m3], tr["converted"][tr["down"] == 3].mean()
    )
    return out


def holdout_fg(model: cm.FieldGoalModel, train: pl.DataFrame, test: pl.DataFrame) -> dict:
    yl, made, _ = fg_frame(test)
    base = fg_frame(train)[1].mean()
    p = model.p_make(yl)
    out = binary_report(p, made, base)
    dist = yl + model.snap_to_kick_yards
    out["calibration_by_kick_distance"] = calibration(
        p, made, np.searchsorted([30, 40, 45, 50, 55], dist, side="right"),
        ["under 30", "30-39", "40-44", "45-49", "50-54", "55+"],
    )  # fmt: skip
    beyond = dist > model.max_kick_distance
    hard = np.where(beyond, 0.0, p)
    out["beyond_the_longest_made_development_kick"] = {
        "longest_made_development_kick": model.max_kick_distance,
        "tail_yards": model.tail_yards,
        "attempts": int(beyond.sum()),
        "makes": int(made[beyond].sum()),
        "mean_predicted": float(p[beyond].mean()) if beyond.any() else None,
        "kick_distances": sorted(int(d) for d in dist[beyond]),
    }
    out["log_loss_with_hard_cap_at_longest_made_kick"] = log_loss(hard, made)
    out["log_loss_with_hard_cap_note"] = (
        "diagnostic only, to show what the fade-out tail is worth on these seasons; the tail "
        "was fixed in the config before this score and is not chosen from it"
    )
    return out


def punt_frame(plays: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(yardline_100, outcome index) for punts that end with the receiving team in
    possession or scoring; outcome 0 is a return touchdown."""
    df = cm.add_next_state(plays).filter(
        (pl.col("play_type") == "punt") & pl.col("yardline_100").is_between(1, 99)
    )
    opp = cm._opponent_next_yardline(df)
    scored = cm._num(df, "touchdown") == 1
    ret_td = cm._flag(df, cm._DEF_TD)
    ok = (~np.isnan(opp) & ~scored & (cm._num(df, "safety") != 1)) | ret_td
    outcome = np.where(ret_td, 0, np.nan_to_num(opp)).astype(int)
    return df["yardline_100"].to_numpy().astype(int)[ok], outcome[ok]


def holdout_punt(
    model: cm.PuntModel, cfg: dict, train: pl.DataFrame, test: pl.DataFrame
) -> dict:
    yl, outcome = punt_frame(test)
    tr_yl, tr_outcome = punt_frame(train.filter(pl.col("season") >= cfg["era_first_season"]))
    eps = 0.01  # every distribution is mixed with 1% uniform so no outcome has probability 0

    def smooth(pmf: np.ndarray) -> np.ndarray:
        return (1 - eps) * pmf + eps / 100

    # Baseline A: one landing-spot distribution for every field position.
    landing = np.bincount(tr_outcome, minlength=100) / tr_outcome.size
    # Baseline B: one net-yards distribution for every field position.
    net = np.bincount(np.clip(tr_outcome - (100 - tr_yl), -100, 99) + 100, minlength=200)
    net = net / net.sum()
    ll_model, ll_landing, ll_net = [], [], []
    for y, o in zip(yl, outcome, strict=True):
        ll_model.append(-np.log(smooth(model.outcomes(y))[o]))
        ll_landing.append(-np.log(smooth(landing)[o]))
        shifted = np.zeros(100)
        np.add.at(shifted, np.clip(100 - y + np.arange(200) - 100, 1, 99), net)
        ll_net.append(-np.log(smooth(shifted)[o]))
    expected = model.expected_opponent_yardline()[yl]
    bins = np.searchsorted([40, 50, 60, 70, 80, 90], yl, side="right")
    labels = ["30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"]
    rows = []
    for i, label in enumerate(labels):
        m = bins == i
        if m.any():
            rows.append({
                "yardline_100": label, "n": int(m.sum()),
                "predicted_opponent_yardline": float(expected[m].mean()),
                "actual_opponent_yardline": float(outcome[m].mean()),
            })  # fmt: skip
    return {
        "n": int(yl.size),
        "log_loss_model": float(np.mean(ll_model)),
        "log_loss_baseline_one_landing_distribution": float(np.mean(ll_landing)),
        "log_loss_baseline_one_net_distribution": float(np.mean(ll_net)),
        "mean_abs_error_expected_vs_actual_yards": float(np.mean(np.abs(expected - outcome))),
        "calibration_by_field_position": rows,
        "note": "log loss is over 100 outcomes (the receiving team's yard line, 0 = return "
        "touchdown); each distribution is mixed with 1% uniform",
    }


def holdout_play(
    model: cm.PlayModel, cfg: dict, train: pl.DataFrame, test: pl.DataFrame
) -> dict:
    te = cm._scrimmage(test, 1, 3)
    tr = cm._scrimmage(train.filter(pl.col("season") >= cfg["era_first_season"]), 1, 3)

    def first_down(s: dict) -> np.ndarray:
        return (((s["gain"] >= s["ydstogo"]) | s["off_td"]) & ~s["turnover"]).astype(float)

    cache: dict = {}
    p_fd, p_td, p_to = np.zeros(te["down"].size), np.zeros(te["down"].size), np.zeros(
        te["down"].size
    )
    for i, key in enumerate(zip(te["down"], te["ydstogo"], te["yardline"], strict=True)):
        if key not in cache:
            o = model.outcomes(int(key[0]), int(key[1]), int(key[2]))
            cache[key] = (o["p_first_down"], o["p_touchdown"], o["p_turnover"])
        p_fd[i], p_td[i], p_to[i] = cache[key]
    y = first_down(te)
    out = binary_report(p_fd, y, first_down(tr).mean())
    out["event"] = "gain reaches the line to gain (or touchdown) without a turnover"
    out["by_down"] = []
    for down in (1, 2, 3):
        m = te["down"] == down
        out["by_down"].append({
            "down": down, "n": int(m.sum()),
            "first_down_predicted": float(p_fd[m].mean()), "first_down_actual": float(y[m].mean()),
            "touchdown_predicted": float(p_td[m].mean()),
            "touchdown_actual": float(te["off_td"][m].mean()),
            "turnover_predicted": float(p_to[m].mean()),
            "turnover_actual": float(te["turnover"][m].mean()),
        })  # fmt: skip
    field = cm._bin(te["yardline"], cm.FIELD_EDGES)
    out["touchdown_calibration_by_field_position"] = calibration(
        p_td, te["off_td"].astype(float), field,
        ["1-4", "5-10", "11-20", "21-40", "41-75", "76-95", "96-99"],
    )  # fmt: skip
    return out


# --------------------------------------------------------------------------------------
# Reference rates from the final fit
# --------------------------------------------------------------------------------------


def reference_rates(models: cm.ComponentModels) -> dict:
    go = {}
    for yl in (50, 30, 5):
        go[f"yardline_100_{yl}"] = {
            f"fourth_and_{d}": float(models.p_convert(yl, d))
            for d in (1, 2, 3, 5, 10, 15) if d <= yl
        }  # fmt: skip
    example = models.go_outcomes(40, 3)
    fg = models.field_goal
    punt_expected = models.punt.expected_opponent_yardline()
    punts = {}
    for label, yl in (("own_20", 80), ("own_30", 70), ("own_40", 60), ("midfield", 50),
                      ("opp_40", 40)):  # fmt: skip
        dist = models.punt_outcomes(yl)
        punts[label] = {
            "yardline_100": yl,
            "expected_opponent_yardline_100": float(punt_expected[yl]),
            "expected_net_yards": float(punt_expected[yl] - (100 - yl)),
            "p_touchback": float(dist[80]),
            "p_inside_opponent_10": float(dist[90:].sum()),
            "p_return_touchdown": float(dist[0]),
        }
    plays = {}
    for down, d, yl in ((1, 10, 75), (2, 5, 50), (3, 1, 30), (3, 8, 12), (1, 1, 1), (1, 10, 99)):
        o = models.play_outcomes(down, d, yl)
        plays[f"down_{down}_and_{d}_at_yardline_100_{yl}"] = {
            k: float(o[k]) for k in ("p_first_down", "p_touchdown", "p_turnover", "p_safety")
        }
    kickoffs = {}
    for name, info in models.kickoff_era_info.items():
        table = models.kickoff_tables[name]
        spot = info.get("touchback_yardline_100")
        kickoffs[name] = {
            **{k: v for k, v in info.items() if k != "name"},
            "mean_start_yardline_100": float(table @ YARDS),
            "p_start_at_touchback_spot": float(table[spot]) if spot is not None else None,
            "p_start_inside_own_20": float(table[81:].sum()),
            "fitted_on_test_seasons": bool(set(info["seasons"]) & set(splits.test_seasons())),
        }
    longest = fg.max_kick_distance
    return {
        "conversion_probability_fourth_down": go,
        "conversion_model": {
            "intercept": models.go.intercept_,
            "coefficients": models.go.coef_,
            "reference_season": models.go.reference_season_,
            "largest_monotone_correction": models.go.monotone_fix_[4],
            "share_of_failed_fourth_downs_that_are_turnovers_by_distance_1_2__3_6__7_plus":
                models.go.turnover_share_.tolist(),
        },
        "go_outcomes_example_fourth_and_3_at_yardline_100_40": {
            "p_convert": example["p_convert"],
            "p_touchdown": example["p_touchdown"],
            "p_first_down_short_of_end_zone": float(example["first_down"].sum()),
            "mean_new_yardline_100_after_first_down": float(
                example["first_down"] @ YARDS / example["first_down"].sum()
            ),
            "p_turnover_any_kind": float(example["turnover"].sum()),
            "p_returned_for_touchdown": float(example["turnover"][0]),
            "mean_opponent_yardline_100_after_turnover": float(
                example["turnover"] @ YARDS / example["turnover"].sum()
            ),
            "p_safety": example["p_safety"],
        },
        "field_goal_make_probability_by_kick_distance": {
            str(dist): float(models.p_field_goal(dist - fg.snap_to_kick_yards))
            for dist in (20, 30, 40, 45, 50, 55, 60, 65, longest, longest + 1, longest + 2,
                         longest + 3, longest + 4)
        },  # fmt: skip
        "field_goal_make_probability_at_first_season_of_trend": {
            str(dist): float(fg.p_make(dist - fg.snap_to_kick_yards, season=models.seasons[0]))
            for dist in (30, 40, 50, 55, 60)
        },
        "field_goal_model": {
            "intercept": fg.intercept_, "coefficients": fg.coef_,
            "reference_season": fg.reference_season_,
            "largest_monotone_correction": fg.monotone_fix_,
            "longest_made_kick_in_fitted_rows": fg.longest_made_,
            "max_kick_distance": fg.max_kick_distance,
            "tail_yards": fg.tail_yards,
            "longest_kick_with_positive_probability": fg.longest_possible_kick,
            "miss_rule": "opponent takes over at the spot of the kick (8 yards behind the line "
            "of scrimmage) or its own 20, whichever is farther from its goal line",
            "opponent_yardline_100_after_miss_from_yardline_100": {
                str(yl): models.fg_miss_yardline(yl) for yl in (5, 12, 13, 25, 35)
            },
        },  # fmt: skip
        "punt": punts,
        "punt_rates": models.punt.rates_,
        "play_model_examples": plays,
        "kickoff_default_era": models.kickoff_era(),
        "kickoff_by_rule_era": kickoffs,
        "seconds_per_play": models.seconds,
        "after_touchdown": models.conversions,
    }


def sample_sizes(models: cm.ComponentModels) -> dict:
    return {
        "plays_in_table": models.n_plays,
        "conversion_model_third_downs": models.go.n_third_,
        "conversion_model_fourth_downs": models.go.n_fourth_,
        "failed_fourth_downs_by_distance_1_2__3_6__7_plus": models.go.turnover_share_n_.tolist(),
        "go_gain_cells_with_own_distribution": int((models.go.context_.level == 0).sum()),
        "go_gain_cells_total": int(models.go.context_.level.size),
        "field_goal_attempts": models.field_goal.n_,
        "punts": models.punt.n_,
        "punts_used": models.punt.n_used_,
        "play_model_plays": models.play.n_,
        "play_model_cells_with_own_distribution": int((models.play.context_.level == 0).sum()),
        "play_model_cells_total": int(models.play.context_.level.size),
        "play_model_turnovers_by_field_bin": models.play.context_.turnover_n.tolist(),
        "kickoffs": models.kickoff_info["n"],
        "extra_points": models.conversions["n_extra_point"],
        "two_point_attempts": models.conversions["n_two_point"],
    }


def kick_distance_check(plays: pl.DataFrame) -> dict:
    fg = plays.filter((pl.col("play_type") == "field_goal") & pl.col("kick_distance").is_not_null())
    diff = (fg["kick_distance"] - fg["yardline_100"]).to_numpy()
    return {f"share_kick_distance_minus_yardline_100_equals_{k}": float(np.mean(diff == k))
            for k in (17, 18, 19)}  # fmt: skip


def missed_fg_check(plays: pl.DataFrame) -> dict:
    """Where the other team actually took over after a missed (not blocked) field goal."""
    df = cm.add_next_state(plays).filter(
        (pl.col("play_type") == "field_goal") & (pl.col("field_goal_result") == "missed")
    )
    opp = cm._opponent_next_yardline(df)
    yl = df["yardline_100"].to_numpy()
    ok = ~np.isnan(opp)
    behind = (100 - opp[ok]) - yl[ok]  # yards behind the line of scrimmage
    far, near = yl[ok] > 12, yl[ok] <= 12
    return {
        "n": int(ok.sum()),
        "from_outside_the_12_share_taking_over_8_yards_behind_the_line": float(
            np.mean(behind[far] == 8)
        ),
        "from_outside_the_12_share_taking_over_7_to_9_yards_behind": float(
            np.mean((behind[far] >= 7) & (behind[far] <= 9))
        ),
        "from_the_12_or_closer_share_taking_over_at_own_20": float(np.mean(opp[ok][near] == 80)),
    }


def diagnostics(models: cm.ComponentModels, plays: pl.DataFrame, cfg: dict) -> dict:
    """Numbers quoted in docs/notes/m1_components.md to justify design decisions.
    `plays` are the development seasons: design decisions are not read off test seasons."""
    short = plays.filter(pl.col("fourth_down_decision").is_not_null() & (pl.col("ydstogo") <= 3))
    who_goes = {
        r["fourth_down_decision"]: {
            "n": r["n"], "mean_spread": r["spread"], "mean_score_differential": r["score"],
        }
        for r in short.group_by("fourth_down_decision").agg(
            pl.len().alias("n"), pl.col("posteam_spread").mean().alias("spread"),
            pl.col("score_differential").mean().alias("score"),
        ).sort("fourth_down_decision").to_dicts()
    }  # fmt: skip
    no_spread = cm.GoModel(
        era_first_season=cfg["era_first_season"], **{**cfg["go"], "use_spread": False}
    ).fit(plays)
    s = go_frame(plays.filter(pl.col("season") >= cfg["era_first_season"]))
    failed3 = (s["down"] == 3) & (s["converted"] == 0)
    ctx = models.play.context_
    td = {}
    for label, lo, hi in (("1-4", 1, 4), ("5-10", 5, 10), ("11-20", 11, 20), ("21-40", 21, 40)):
        n = ctx.plays_by_yardline[lo : hi + 1].sum()
        td[label] = {
            "plays_without_turnover": int(n),
            "touchdown_rate_predicted_before_calibration": float(
                ctx.td_predicted[lo : hi + 1].sum() / n),
            "touchdown_rate_actual": float(ctx.td_actual[lo : hi + 1].sum() / n),
        }  # fmt: skip
    yl, outcome = punt_frame(plays)
    punts = cm.add_next_state(plays).filter(pl.col("play_type") == "punt")
    punt_net = {}
    for year in (models.seasons[0], cfg["era_first_season"], models.seasons[1]):
        y, o = punt_frame(punts.filter(pl.col("season") == year))
        punt_net[str(year)] = float(np.mean(o - (100 - y)))
    ko = cm.add_next_state(plays).filter(
        (pl.col("play_type") == "kickoff") & (pl.col("next_posteam") == pl.col("posteam"))
        & (pl.col("next_half") == pl.col("game_half")) & (pl.col("touchdown") != 1)
    )  # fmt: skip
    ko_mode = {
        str(r["season"]): {"most_common_start_yardline_100": r["mode"], "mean": r["mean"]}
        for r in ko.group_by("season").agg(
            pl.col("next_yardline_100").mode().first().alias("mode"),
            pl.col("next_yardline_100").mean().alias("mean"),
        ).sort("season").to_dicts()
    }  # fmt: skip
    return {
        "who_goes_on_fourth_and_3_or_less": who_goes,
        "fourth_down_coefficient_with_spread_control": models.go.coef_["fourth_down"],
        "fourth_down_coefficient_without_spread_control": no_spread.coef_["fourth_down"],
        "share_of_failed_third_downs_that_are_turnovers": float(s["turnover"][failed3].mean()),
        "play_model_touchdown_rate_in_sample_by_yardline_100": td,
        "punt_mean_net_yards_by_season": punt_net,
        "punts_from_yardline_100_under_30_development_seasons": int((yl < 30).sum()),
        "punts_from_yardline_100_under_31_development_seasons": int((yl < 31).sum()),
        "punts_from_yardline_100_under_35_development_seasons": int((yl < 35).sum()),
        "kickoff_start_by_season": ko_mode,
    }


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------


def figure_data(models: cm.ComponentModels, plays: pl.DataFrame, cfg: dict) -> dict:
    s = go_frame(plays)
    open_field = (s["yardline"] >= 20) & (s["yardline"] <= 80)
    go_rows = []
    for d in range(1, 16):
        row: dict = {"ydstogo": d, "fitted_fourth_down_at_midfield": float(models.p_convert(50, d))}
        for down, name in ((3, "third"), (4, "fourth")):
            m = open_field & (s["down"] == down) & (s["ydstogo"] == d)
            row[f"{name}_down_n"] = int(m.sum())
            row[f"{name}_down_raw"] = float(s["converted"][m].mean()) if m.any() else None
            if down == 4 and m.any():
                row["fitted_fourth_down_same_plays"] = float(
                    np.mean(models.p_convert(s["yardline"][m], np.full(m.sum(), d)))
                )
        go_rows.append(row)

    recent = max(models.seasons) - 3
    yl, made, _ = fg_frame(plays.filter(pl.col("season") >= recent))
    dist = yl + models.field_goal.snap_to_kick_yards
    fg_rows = []
    for lo in range(18, 66, 3):
        m = (dist >= lo) & (dist < lo + 3)
        if m.sum() >= 10:
            fg_rows.append({"kick_distance_from": lo, "kick_distance_to": lo + 2,
                            "mean_distance": float(dist[m].mean()), "n": int(m.sum()),
                            "raw": float(made[m].mean())})  # fmt: skip
    grid = np.arange(18, models.field_goal.longest_possible_kick + 2)
    fg_fit = [{"kick_distance": int(k),
               "fitted": float(models.p_field_goal(k - models.field_goal.snap_to_kick_yards)),
               "fitted_first_season": float(models.field_goal.p_make(
                   k - models.field_goal.snap_to_kick_yards, season=models.seasons[0]))}
              for k in grid]  # fmt: skip

    expected = models.punt.expected_opponent_yardline()
    punt_rows = []
    for y in range(30, 100):
        n = int(models.punt.raw_n_[y])
        punt_rows.append({
            "yardline_100": y, "n": n,
            "raw_net": float(models.punt.raw_expected_[y] - (100 - y)) if n >= 20 else None,
            "fitted_net": float(expected[y] - (100 - y)),
        })  # fmt: skip
    return {
        "go": go_rows,
        "go_note": "raw rates: development seasons, plays between the 20-yard lines; "
        "fitted_same_plays: "
        "the model's fourth-down probability (last season, spread 0) averaged over the field "
        "positions of the raw fourth-down plays at that distance",
        "field_goal_raw": fg_rows,
        "field_goal_raw_seasons": [recent, max(models.seasons)],
        "field_goal_fit": fg_fit,
        "punt": punt_rows,
        "punt_note": f"raw: seasons from {cfg['era_first_season']}, yard lines with 20+ punts; "
        "net = receiving team's yardline_100 at its next snap minus (100 - yardline_100); "
        "a return touchdown counts as yardline 0",
    }


def figure(fd: dict, models: cm.ComponentModels) -> None:
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), gridspec_kw={"wspace": 0.24})
    dot = {"linestyle": "none", "marker": "o", "markersize": 6.5,
           "markeredgecolor": style.SURFACE, "markeredgewidth": 1.2}  # fmt: skip

    # Panel 1: conversion probability by yards to go.
    ax, color = axes[0], style.ACTION_COLORS["go"]
    d = np.array([r["ydstogo"] for r in fd["go"]])
    raw4 = np.array([r["fourth_down_raw"] for r in fd["go"]], dtype=float)
    n4 = np.array([r["fourth_down_n"] for r in fd["go"]], dtype=float)
    raw3 = np.array([r["third_down_raw"] for r in fd["go"]], dtype=float)
    fit = np.array([r["fitted_fourth_down_same_plays"] for r in fd["go"]])
    se = np.sqrt(raw4 * (1 - raw4) / n4)
    ax.plot(d, 100 * fit, color=color, zorder=3,
            label=f"Fitted, fourth down, {models.seasons[1]} level")  # fmt: skip
    ax.errorbar(d, 100 * raw4, yerr=100 * se, color=color, elinewidth=1.2, zorder=4,
                label="Raw fourth downs (bar: one standard error)", **dot)  # fmt: skip
    ax.plot(d, 100 * raw3, color=style.MUTED, label="Raw third downs", zorder=2, **dot)
    ax.set_xticks(range(1, 16, 2))
    ax.set_xlabel("Yards to go")
    ax.set_ylabel("Converted (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    style.title(ax, "Go for it: converts less with each yard",
                "Chance of a first down or touchdown, between the 20s")  # fmt: skip

    # Panel 2: field goals.
    ax, color = axes[1], style.ACTION_COLORS["field_goal"]
    k = np.array([r["kick_distance"] for r in fd["field_goal_fit"]])
    lo, hi = fd["field_goal_raw_seasons"]
    ax.plot(k, [100 * r["fitted"] for r in fd["field_goal_fit"]], color=color,
            label=f"Fitted, {hi} level", zorder=3)  # fmt: skip
    ax.plot(k, [100 * r["fitted_first_season"] for r in fd["field_goal_fit"]], color=style.MUTED,
            linestyle=(0, (4, 3)), linewidth=1.6, label=f"Fitted, {models.seasons[0]} level",
            zorder=2)  # fmt: skip
    ax.plot([r["mean_distance"] for r in fd["field_goal_raw"]],
            [100 * r["raw"] for r in fd["field_goal_raw"]], color=color,
            label=f"Raw, {lo} to {hi}", zorder=4, **dot)  # fmt: skip
    ax.set_xlabel("Kick distance (yards)")
    ax.set_ylabel("Made (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(16, 72)
    ax.legend(loc="lower left")
    style.title(ax, "Field goal: reliable until about 50 yards",
                "Chance of a make; distance is the line of scrimmage + 18")  # fmt: skip

    # Panel 3: punts.
    ax, color = axes[2], style.ACTION_COLORS["punt"]
    y = np.array([r["yardline_100"] for r in fd["punt"]])
    raw = np.array([r["raw_net"] if r["raw_net"] is not None else np.nan for r in fd["punt"]])
    ax.plot(y, [r["fitted_net"] for r in fd["punt"]], color=color, label="Fitted", zorder=3)
    ax.plot(y, raw, color=color, alpha=0.55, label="Raw, one point per yard line", zorder=2,
            **{**dot, "markersize": 5})  # fmt: skip
    ax.set_xlabel("Yards from the opponent's end zone when punting")
    ax.set_ylabel("Expected net yards")
    ax.set_ylim(0, 50)
    ax.set_xlim(28, 101)
    ax.legend(loc="lower right")
    style.title(ax, "Punt: about 43 net yards, fewer near midfield",
                "Field position gained after the return, touchbacks included")  # fmt: skip

    for ax in axes:
        ax.grid(axis="x", visible=False)
    style.save(fig, ASSETS / "m1_components.png")


# --------------------------------------------------------------------------------------


def flip_pairs(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
    names = ("go", "field_goal", "punt")
    out: dict[str, int] = {}
    for x, y in zip(a[a != b], b[a != b], strict=True):
        key = f"{names[x]}_to_{names[y]}"
        out[key] = out.get(key, 0) + 1
    return out


def kickoff_sensitivity(models: cm.ComponentModels) -> dict:
    """How much does the kickoff exception matter for the fourth-down comparison?

    The single-drive expected-points problem (M4 stage 1) is solved with the empirical
    kickoff table of each rule era and again with every kickoff replaced by a start at
    that era's touchback spot (a rule-book constant: no returns, no return touchdowns).
    Reported: the change in the value of receiving a kickoff and in go minus the better
    kick, over every fourth-down state of the grid, and how many best actions flip (with
    the larger of the two solves' best-minus-second margins among the flipped states).
    Expected points are used because they need only the component models; the same swap
    in win probability is a sensitivity row of scripts/questions/q1_late_lead_dp.py.
    """
    import dataclasses

    mcfg = mdp.load_config()
    grid = mdp.grid_from_config(mcfg)
    no_stops = {k: np.zeros(mdp.GAIN_MAX - mdp.GAIN_MIN + 1) for k in mdp.KERNEL_KEYS}
    kern = mdp.build_field_kernels(models, grid, no_stops)
    free_kick = kern.kickoff  # held fixed, so only the kickoff after a score changes
    td_value = mcfg["scoring"]["touchdown"] + models.p_extra_point()
    downs, dbins, yls = grid.unpack(np.arange(grid.nf))
    fourth = downs == grid.n_down
    real = fourth & (grid.dist_rep(dbins) <= yls)  # a distance longer than the field cannot occur

    def solved(kickoff: np.ndarray) -> dict:
        k = dataclasses.replace(kern, kickoff=mdp._drop_return(kickoff), free_kick=free_kick)
        r = solve_drive(k, td_value=td_value, tolerance=mcfg["drive"]["tolerance"],
                        max_iterations=mcfg["drive"]["max_iterations"])  # fmt: skip
        q = np.where(r["allowed"], r["q"], -np.inf)
        top = np.sort(q, axis=1)
        with np.errstate(invalid="ignore"):  # rows with one allowed action: no runner-up
            return {"gain": q[:, 0] - q[:, 1:].max(axis=1), "best": q.argmax(axis=1),
                    "margin": top[:, 2] - top[:, 1],
                    "kickoff_value": r["kickoff_value"], "converged": r["converged"]}  # fmt: skip

    out: dict = {
        "what": "expected points, single-drive problem with optimal fourth downs; empirical "
        "kickoff table against a fixed start at the touchback spot",
        "fourth_down_states": int(real.sum()),
        "by_rule_era": {},
    }
    for name, info in models.kickoff_era_info.items():
        spot = info.get("touchback_yardline_100")
        if spot is None:
            continue
        emp = solved(models.kickoff_tables[name])
        const = solved(cm.constant_kickoff(spot))
        diff = (const["gain"] - emp["gain"])[real]
        flips = (const["best"] != emp["best"])[real]
        margin = np.maximum(emp["margin"], const["margin"])[real]
        out["by_rule_era"][name] = {
            "seasons": info["seasons"],
            "touchback_yardline_100": spot,
            "empirical_mean_start_yardline_100": float(models.kickoff_tables[name] @ YARDS),
            "value_of_receiving_a_kickoff_empirical": emp["kickoff_value"],
            "value_of_receiving_a_kickoff_fixed_spot": const["kickoff_value"],
            "go_minus_best_kick_change_mean_abs": float(np.abs(diff).mean()),
            "go_minus_best_kick_change_max_abs": float(np.abs(diff).max()),
            "states_whose_best_action_flips": int(flips.sum()),
            "largest_margin_among_flipped_states": float(margin[flips].max()) if flips.any()
            else 0.0,
            "flipped_states_by_action_pair": flip_pairs(emp["best"][real], const["best"][real]),
            "both_solves_converged": bool(emp["converged"] and const["converged"]),
        }  # fmt: skip
    return out


def main() -> None:
    cfg = cm.load_config()
    dev = cm.add_next_state(splits.load_dev_plays())

    # 1. Choose the model forms inside the development seasons.
    selection = model_selection(cfg, dev)

    # 2. The development bundle, with the kickoff rule tables as the logged exception.
    test = cm.add_next_state(
        splits.load_test_plays(
            "M1 components: one held-out score of the adopted development fits, the kickoff "
            "tables of the 2024 and 2025 rules, and the all-seasons forward-looking bundle"
        )
    )
    models = cm.fit_dev_bundle(cfg, dev_plays=dev, test_plays=test, seed=cfg["seed"])
    path = models.save()

    # 3. One held-out score of the development bundle.
    splits.assert_can_score(models.stamp, test, "held-out check of the component models")
    heldout = {
        "trained_on_seasons": models.seasons,
        "tested_on_seasons": sorted(splits.seasons_in(test)),
        "go": holdout_go(models.go, dev, test),
        "field_goal": holdout_fg(models.field_goal, dev, test),
        "punt": holdout_punt(models.punt, cfg, dev, test),
        "play_model": holdout_play(models.play, cfg, dev, test),
    }

    # 4. The all-seasons bundle: forward-looking only, never scored here.
    everything = pl.concat([dev, test], how="vertical_relaxed")
    all_models = cm.ComponentModels(cfg).fit(
        everything, seed=cfg["seed"], allow_test_seasons=True,
        reason="forward-looking all-seasons bundle (future app); nothing reported on the test "
        "seasons may use it",
    )  # fmt: skip
    all_path = all_models.save()
    assert path == cm.DEV_MODEL_PATH and all_path == cm.ALL_MODEL_PATH

    fd = figure_data(models, dev, cfg)
    figure(fd, models)

    recent = cfg["drives"]["recent_first_season"]
    results = {
        "seed": models.seed,
        "seed_note": "recorded only; no step of the fit draws random numbers",
        "bundles": {
            "development": {"path": str(path.relative_to(cm.DATA.parent)), "stamp": models.stamp},
            "all_seasons": {
                "path": str(all_path.relative_to(cm.DATA.parent)),
                "stamp": all_models.stamp,
                "use": "forward-looking artifacts only; no number in this repo is reported "
                "from it on the test seasons",
            },
        },
        "seasons_in_development_fit": models.seasons,
        "seasons_by_component": {
            "conversion_probability": models.seasons,
            "field_goal": models.seasons,
            "gain_distributions_punts_turnovers_clock_extra_points":
                [cfg["era_first_season"], models.seasons[1]],
            "kickoff": {k: v["seasons"] for k, v in models.kickoff_era_info.items()},
        },  # fmt: skip
        "config": cfg,
        "model_selection": selection,
        "sample_sizes": sample_sizes(models),
        "kick_distance_check": kick_distance_check(dev),
        "missed_field_goal_spot_check": missed_fg_check(dev),
        "fitted": reference_rates(models),
        "fitted_all_seasons_bundle": reference_rates(all_models),
        "diagnostics": diagnostics(models, dev, cfg),
        "heldout": heldout,
        "kickoff_sensitivity": kickoff_sensitivity(models),
        "drive_targets": {
            "development_seasons": cm.drive_targets(dev),
            "development_recent_seasons": cm.drive_targets(dev, recent),
            "test_seasons": cm.drive_targets(test),
            "note": "real data, not model output; the simulator is compared with these in M4 "
            "(test seasons out of sample, recent development seasons in sample)",
        },
        "figure_data": fd,
    }
    write_json(results, RESULTS / "m1_components.json")

    print(f"bundles: {path} and {all_path}")
    for key in ("conversion_on_fourth_downs", "field_goal_form", "field_goal_tail"):
        sel = selection[key]
        edge = sel["adopted_minus_lowest"]
        print(f"selection, {key}: adopted {sel['adopted']}, lowest {sel['lowest_pooled_log_loss']} "
              f"(adopted minus lowest {edge['difference']:+.5f}, se {edge['standard_error']:.5f})")
    f = results["fitted"]
    go = f["conversion_probability_fourth_down"]["yardline_100_50"]
    kicks = f["field_goal_make_probability_by_kick_distance"]
    nets = {k: round(v["expected_net_yards"], 1) for k, v in f["punt"].items()}
    print("fourth-down conversion at midfield:", {k: round(v, 3) for k, v in go.items()})
    print("field goal by kick distance:", {k: round(v, 3) for k, v in kicks.items()})
    print("punt expected net:", nets)
    for name in ("go", "field_goal", "punt", "play_model"):
        h = heldout[name].get("fourth_down", heldout[name])
        base = [v for k, v in h.items() if k.startswith("log_loss_") and "baseline" in k]
        print(f"held-out log loss, {name}: model {h['log_loss_model']:.4f}, "
              f"baselines {[round(b, 4) for b in base]}")  # fmt: skip
    for name, row in results["kickoff_sensitivity"]["by_rule_era"].items():
        print(f"kickoff sensitivity, {name}: go minus best kick moves "
              f"{row['go_minus_best_kick_change_mean_abs']:.4f} points on average, "
              f"{row['go_minus_best_kick_change_max_abs']:.4f} at most; "
              f"{row['states_whose_best_action_flips']} best actions flip")  # fmt: skip


if __name__ == "__main__":
    main()
