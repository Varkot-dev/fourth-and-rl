"""Empirical leakage checks for docs/research/leakage_audit.md.

Read-only with respect to models and data: it loads the saved artifacts, refits
copies in memory and compares. Output: results/research/leakage_checks.json.

Run:  python scripts/research/leakage_checks.py                (about five minutes, one thread)
      python scripts/research/leakage_checks.py --skip-refits  (data checks only, seconds)
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from fourth_and_rl import components as cm  # noqa: E402
from fourth_and_rl import data, wp  # noqa: E402
from fourth_and_rl.io import read_json, write_json  # noqa: E402
from fourth_and_rl.paths import DATA, RESULTS  # noqa: E402

FORBIDDEN_FEATURES = {
    "result", "home_score", "away_score", "posteam_won", "epa", "wpa", "wp", "vegas_wp",
    "home_wp", "vegas_home_wp", "ep", "yards_gained", "touchdown", "sp", "first_down",
    "fixed_drive_result", "td_team", "field_goal_result", "desc",
}  # fmt: skip


def span(bounds: list[int]) -> list[int]:
    return list(range(bounds[0], bounds[1] + 1))


def check_splits(plays: pl.DataFrame, cfg: dict) -> dict:
    """(d) No game in two splits, one season and one label per game, no duplicate rows."""
    sets = {
        k: set(wp.season_filter(plays, span(v))["game_id"].unique().to_list())
        for k, v in cfg["split"].items()
    }
    per_game = plays.group_by("game_id").agg(
        pl.col("season").n_unique().alias("seasons"),
        pl.col("result").n_unique().alias("results"),
        pl.col("spread_line").n_unique().alias("spreads"),
    )
    # The two teams of a game must carry opposite labels.
    sides = (
        plays.filter(pl.col(wp.LABEL).is_not_null())
        .group_by("game_id", "posteam")
        .agg(pl.col(wp.LABEL).n_unique().alias("n"), pl.col(wp.LABEL).first().alias("y"))
    )
    label_sum = sides.group_by("game_id").agg(pl.col("y").sum().alias("s"), pl.len().alias("k"))
    return {
        "games": {k: len(v) for k, v in sets.items()},
        "games_in_train_and_calib": len(sets["train"] & sets["calib"]),
        "games_in_train_and_test": len(sets["train"] & sets["test"]),
        "games_in_calib_and_test": len(sets["calib"] & sets["test"]),
        "games_with_more_than_one_season": int((per_game["seasons"] > 1).sum()),
        "games_with_more_than_one_final_result": int((per_game["results"] > 1).sum()),
        "games_with_more_than_one_spread_line": int((per_game["spreads"] > 1).sum()),
        "duplicate_game_id_play_id_rows": int(
            plays.height - plays.select("game_id", "play_id").unique().height
        ),
        "team_games_with_a_label_that_changes": int((sides["n"] > 1).sum()),
        "two_sided_games_whose_labels_do_not_sum_to_1": int(
            label_sum.filter((pl.col("k") == 2) & (pl.col("s") != 1.0)).height
        ),
    }


def check_pre_play(plays: pl.DataFrame) -> dict:
    """(e) Score, clock and timeouts on a row describe the moment before the snap."""
    home = pl.col("posteam") == pl.col("home_team")
    df = plays.sort("game_id", "order_sequence", "play_id").with_columns(
        pl.when(home).then(pl.col("posteam_score")).otherwise(pl.col("defteam_score")).alias("hs"),
        pl.when(home).then(pl.col("defteam_score")).otherwise(pl.col("posteam_score")).alias("as_"),
    )  # fmt: skip
    df = df.with_columns(
        *[pl.col(c).shift(1).over("game_id").alias(f"{c}_prev") for c in ("hs", "as_")],
        *[pl.col(c).shift(-1).over("game_id").alias(f"{c}_next")
          for c in ("hs", "as_", "game_seconds_remaining", "game_half")],
        pl.col("sp").shift(1).over("game_id").alias("sp_prev"),
    )  # fmt: skip
    ok = pl.col("hs").is_not_null() & pl.col("hs_next").is_not_null()
    scoring = df.filter((pl.col("sp") == 1) & ok & pl.col("hs_prev").is_not_null())
    changed_after = (pl.col("hs_next") != pl.col("hs")) | (pl.col("as__next") != pl.col("as_"))
    same_as_before = (pl.col("hs_prev") == pl.col("hs")) & (pl.col("as__prev") == pl.col("as_"))
    first_score = (
        df.filter(pl.col("sp") == 1)
        .group_by("game_id", maintain_order=True)
        .agg(pl.col("posteam_score").first(), pl.col("defteam_score").first(),
             pl.col("score_differential").first())
    )  # fmt: skip
    final = (
        df.group_by("game_id")
        .agg(pl.col("hs").max().alias("max_hs"), pl.col("home_score").first(),
             pl.col("as_").max().alias("max_as"), pl.col("away_score").first())
    )  # fmt: skip
    td = df.filter(
        (pl.col("touchdown") == 1) & pl.col("game_seconds_remaining_next").is_not_null()
        & (pl.col("game_half_next") == pl.col("game_half")) & (pl.col("qtr") <= 4)
    )  # fmt: skip
    opening = plays.filter((pl.col("play_type") == "kickoff") & (pl.col("qtr") == 1)).group_by(
        "game_id"
    ).agg(pl.col("game_seconds_remaining").max().alias("t"))
    diff_ok = df.filter(pl.col("score_differential").is_not_null()).select(
        (pl.col("score_differential") == pl.col("posteam_score") - pl.col("defteam_score")).mean()
    ).item()
    return {
        "scoring_plays_checked": scoring.height,
        "share_scoring_rows_whose_score_equals_the_previous_row": float(
            scoring.filter(pl.col("sp_prev") != 1).select(same_as_before.mean()).item()
        ),
        "share_scoring_rows_whose_score_changes_on_the_next_row": float(
            scoring.select(changed_after.mean()).item()
        ),
        "share_first_scoring_plays_at_0_0": float(
            first_score.select(
                ((pl.col("posteam_score") == 0) & (pl.col("defteam_score") == 0)).mean()
            ).item()
        ),
        "share_rows_score_differential_equals_posteam_minus_defteam": float(diff_ok),
        "share_games_where_a_row_already_shows_the_final_home_and_away_score": float(
            final.select(
                ((pl.col("max_hs") == pl.col("home_score"))
                 & (pl.col("max_as") == pl.col("away_score"))).mean()
            ).item()
        ),
        "note_final_score": "A pre-play score reaches the final score only when a row follows "
        "the last score (a kickoff, kneel or try); the share above is context, not a failure.",
        "touchdown_rows_checked": td.height,
        "share_touchdown_rows_with_clock_above_the_next_row": float(
            td.select((pl.col("game_seconds_remaining") > pl.col("game_seconds_remaining_next"))
                      .mean()).item()
        ),
        "timeouts": check_timeouts(),
        "share_games_whose_opening_kickoff_is_at_3600_seconds": float(
            (opening["t"] == 3600).mean()
        ),
    }  # fmt: skip


def check_timeouts() -> dict:
    """Timeout rows have no posteam, so the state table drops them: read the raw files.

    For every timeout row, compare the two teams' combined timeouts on the last row
    with a posteam before it and the first one after it (same half). Pre-play columns
    show the old count before the timeout row and the reduced count after it.
    """
    cfg = data.load_config()
    raw = data.read_raw(data.ROOT / cfg["raw_dir"], data.seasons_from_config(cfg))
    has_team = pl.col("posteam").is_not_null() & (pl.col("posteam") != "")
    total = pl.when(has_team).then(
        pl.col("posteam_timeouts_remaining") + pl.col("defteam_timeouts_remaining")
    )
    half = pl.when(has_team).then(pl.col("game_half"))
    df = raw.sort("game_id", "order_sequence", "play_id").with_columns(
        total.shift(1).forward_fill().over("game_id").alias("before"),
        total.shift(-1).backward_fill().over("game_id").alias("after"),
        half.shift(1).forward_fill().over("game_id").alias("half_before"),
        half.shift(-1).backward_fill().over("game_id").alias("half_after"),
    )
    rows = df.filter(
        (pl.col("timeout") == 1) & ~has_team & pl.col("before").is_not_null()
        & pl.col("after").is_not_null() & (pl.col("half_before") == pl.col("half_after"))
        & pl.col("desc").fill_null("").str.contains(r"^Timeout #")
    )  # fmt: skip
    return {
        "timeout_rows_checked": rows.height,
        "timeout_rows_without_a_posteam_in_raw": raw.filter(
            (pl.col("timeout") == 1) & ~has_team).height,
        "timeout_rows_with_a_posteam_in_raw": raw.filter(
            (pl.col("timeout") == 1) & has_team).height,
        "share_where_the_play_after_has_exactly_one_fewer_than_the_play_before": float(
            rows.select((pl.col("after") == pl.col("before") - 1).mean()).item()
        ),
        "share_where_the_count_does_not_change_across_the_timeout_row": float(
            rows.select((pl.col("after") == pl.col("before")).mean()).item()
        ),
    }  # fmt: skip


def check_end_game_rows(plays: pl.DataFrame) -> dict:
    """The last row of a game ("END GAME") carries the final score as its pre-play score.

    If such a row could be scored, score_differential at 0 seconds would give the label
    away. It cannot, because it has no field position. This check makes that visible.
    """
    last = plays.sort("game_id", "order_sequence", "play_id").group_by(
        "game_id", maintain_order=True
    ).agg(pl.all().last())
    labelled = last.filter(pl.col(wp.LABEL).is_not_null())
    return {
        "games": last.height,
        "last_rows_described_as_end_game": int(
            last["desc"].fill_null("").str.starts_with("END GAME").sum()
        ),
        "share_last_rows_whose_score_differential_sign_equals_the_label": float(
            labelled.select(
                ((pl.col("score_differential") > 0) == (pl.col(wp.LABEL) == 1)).mean()
            ).item()
        ),
        "last_rows_that_the_model_can_score_or_train_on": int(wp.scoreable(last).sum()),
        "last_rows_with_an_nflfastr_wp": int(last["wp"].is_not_null().sum()),
    }


def check_features() -> dict:
    return {
        "features": wp.FEATURES,
        "state_columns": wp.STATE_COLUMNS,
        "forbidden_columns_among_features_or_state": sorted(
            FORBIDDEN_FEATURES & (set(wp.FEATURES) | set(wp.STATE_COLUMNS))
        ),
    }


def by_season_loss(df: pl.DataFrame, col: str) -> dict[str, float]:
    out = {}
    for s in sorted(df["season"].unique().to_list()):
        sub = df.filter((pl.col("season") == s) & pl.col(col).is_not_null())
        out[str(s)] = wp.log_loss(sub[wp.LABEL].to_numpy(), sub[col].to_numpy())
    return out


def check_wp(plays: pl.DataFrame, cfg: dict) -> dict:
    """(b) and (c): refit on the declared seasons and compare with the saved model;
    in-sample versus out-of-fold log loss by season."""
    split = {k: span(v) for k, v in cfg["split"].items()}
    saved = wp.WPModel.load(DATA / "models" / "wp.joblib")
    saved.n_threads = 1
    train, calib = (wp.season_filter(plays, split[k]) for k in ("train", "calib"))
    refit = wp.WPModel(cfg["lgbm"], cfg["calibration"], 1).fit(train, calib, seed=cfg["seed"])
    labelled = plays.filter(pl.col(wp.LABEL).is_not_null())
    p_saved, p_refit = saved.predict(labelled), refit.predict(labelled)
    ok = ~np.isnan(p_saved)
    is_test = labelled["season"].is_in(split["test"]).to_numpy()
    # A model that had also seen the test seasons, for contrast: how different would it be?
    leaky = wp.WPModel(cfg["lgbm"], cfg["calibration"], 1).fit(
        wp.season_filter(plays, split["train"] + split["test"]), calib, seed=cfg["seed"]
    )
    p_leaky = leaky.predict(labelled)
    y = labelled[wp.LABEL].to_numpy()
    m = ok & is_test

    oof = pl.read_parquet(DATA / "processed" / "wp_predictions.parquet")
    oof = oof.with_columns(pl.col("play_id").cast(plays.schema["play_id"]))
    joined = labelled.with_columns(pl.Series("wp_headline", p_saved).fill_nan(None)).join(
        oof, on=["game_id", "play_id"], how="left", validate="1:1"
    ).filter(pl.col("wp_headline").is_not_null() & pl.col("wp_ours").is_not_null())
    headline, crossfit = by_season_loss(joined, "wp_headline"), by_season_loss(joined, "wp_ours")

    # Fold 4 of the cross-fit (the one that predicts the test seasons), refitted.
    plan = wp.crossfit_plan(plays["season"].unique().to_list(), cfg["crossfit"]["fold_size"],
                            cfg["crossfit"]["n_calib_seasons"])  # fmt: skip
    fold = next(f for f in plan if split["test"][0] in f["predict"])
    fold_model = wp.WPModel(cfg["lgbm"], cfg["calibration"], 1).fit(
        wp.season_filter(plays, fold["train"]), wp.season_filter(plays, fold["calib"]),
        seed=cfg["seed"],
    )  # fmt: skip
    held = wp.season_filter(plays, fold["predict"])
    p_fold = fold_model.predict(held)
    held = held.select("game_id", "play_id").with_columns(pl.Series("p", p_fold)).join(
        oof, on=["game_id", "play_id"], how="left", validate="1:1"
    ).filter(pl.col("wp_ours").is_not_null() & pl.col("p").is_not_nan())

    def mean_over(d: dict, seasons: list[int]) -> float:
        return float(np.mean([d[str(s)] for s in seasons]))

    return {
        "saved_model": {"seed": saved.seed, "n_train_rows": saved.n_train_rows,
                        "n_calib_rows": saved.n_calib_rows,
                        "platt": saved.calibrator.describe()},
        "refit_on_declared_seasons": {"n_train_rows": refit.n_train_rows,
                                      "n_calib_rows": refit.n_calib_rows,
                                      "platt": refit.calibrator.describe()},
        "max_abs_prediction_difference_saved_vs_refit_all_rows": float(
            np.max(np.abs(p_saved[ok] - p_refit[ok]))),
        "max_abs_prediction_difference_saved_vs_refit_test_rows": float(
            np.max(np.abs(p_saved[m] - p_refit[m]))),
        "contrast_model_trained_on_train_plus_test": {
            "max_abs_difference_from_saved_on_test_rows": float(
                np.max(np.abs(p_saved[m] - p_leaky[m]))),
            "test_log_loss_saved": wp.log_loss(y[m], p_saved[m]),
            "test_log_loss_if_test_seasons_were_trained_on": wp.log_loss(y[m], p_leaky[m]),
        },
        "headline_model_log_loss_by_season": headline,
        "crossfit_out_of_fold_log_loss_by_season": crossfit,
        "headline_minus_out_of_fold_by_season": {
            s: headline[s] - crossfit[s] for s in headline},
        "mean_gap_train_seasons_in_sample_minus_out_of_fold": mean_over(headline, split["train"])
        - mean_over(crossfit, split["train"]),
        "mean_gap_test_seasons_headline_minus_out_of_fold": mean_over(headline, split["test"])
        - mean_over(crossfit, split["test"]),
        "crossfit_fold_predicting_test": {
            **fold,
            "rows_compared": held.height,
            "max_abs_difference_saved_predictions_vs_refit": float(
                (held["p"] - held["wp_ours"]).abs().max()),
        },
        "crossfit_plan": plan,
    }  # fmt: skip


def bundle_summary(b: cm.ComponentModels) -> dict:
    return {
        "seasons": b.seasons,
        "n_plays": b.n_plays,
        "go_reference_season": b.go.reference_season_,
        "field_goal_reference_season": b.field_goal.reference_season_,
        "field_goal_attempts": b.field_goal.n_,
        "field_goal_longest_made": b.field_goal.longest_made_,
        "punts": b.punt.n_,
        "play_model_plays": b.play.n_,
        "kickoff": b.kickoff_info,
    }


def bundle_diff(a: cm.ComponentModels, b: cm.ComponentModels) -> dict:
    d = lambda x, y: float(np.max(np.abs(np.asarray(x) - np.asarray(y))))  # noqa: E731
    return {
        "p_convert_fourth_down_table": d(a.go.table_[4], b.go.table_[4]),
        "p_field_goal_table": d(a.field_goal.table_, b.field_goal.table_),
        "punt_table": d(a.punt.table_, b.punt.table_),
        "play_gain_pmf": d(a.play.context_.pmf, b.play.context_.pmf),
        "kickoff": d(a.kickoff, b.kickoff),
        "p_extra_point": abs(a.p_extra_point() - b.p_extra_point()),
        "mean_scrimmage_seconds": abs(a.seconds["mean_scrimmage_play"]
                                      - b.seconds["mean_scrimmage_play"]),
    }  # fmt: skip


def check_components(plays: pl.DataFrame) -> dict:
    """Which seasons each saved bundle holds, verified by refitting."""
    cfg = cm.load_config()
    plays = cm.add_next_state(plays)
    out: dict = {"holdout_seasons_in_config": cfg["holdout_seasons"]}
    dev = cm.ComponentModels(cfg).fit(plays.filter(pl.col("season").is_between(2010, 2023)))
    out["refit_2010_2023"] = bundle_summary(dev)
    dev_ko = cm.ComponentModels(cfg).fit(plays.filter(pl.col("season").is_between(2010, 2023)))
    dev_ko.kickoff, dev_ko.kickoff_info = cm.fit_kickoff(plays, cfg["kickoff"]["seasons"])
    for name in ("components.joblib", "components_2010_2023.joblib"):
        path = DATA / "models" / name
        if not path.exists():
            out[name] = "missing"
            continue
        b = cm.ComponentModels.load(path)
        out[name] = {
            **bundle_summary(b),
            "max_abs_difference_from_refit_2010_2023": bundle_diff(b, dev),
            "max_abs_difference_from_refit_2010_2023_with_2025_kickoffs": bundle_diff(b, dev_ko),
        }
    fg = plays.filter((pl.col("play_type") == "field_goal")
                      & (pl.col("field_goal_result") == "made"))  # fmt: skip
    longest = fg.group_by("season").agg((pl.col("yardline_100").max() + 18).alias("longest"))
    out["longest_made_field_goal_by_season_group"] = {
        "dev_2010_2023": int(longest.filter(pl.col("season") <= 2023)["longest"].max()),
        "test_2024_2025": int(longest.filter(pl.col("season") >= 2024)["longest"].max()),
        "max_kick_distance_in_config": cfg["field_goal"]["max_kick_distance"],
    }
    return out


def check_field_goal_cap(plays: pl.DataFrame) -> dict:
    """The held-out field-goal log loss with the config cap (68 yards, a 2025 kick) and
    with the longest kick made in the development seasons."""
    cfg = cm.load_config()
    train = plays.filter(~pl.col("season").is_in(cfg["holdout_seasons"]))
    test = plays.filter(
        pl.col("season").is_in(cfg["holdout_seasons"]) & (pl.col("play_type") == "field_goal")
        & pl.col("yardline_100").is_between(1, 99) & pl.col("field_goal_result").is_not_null()
    )  # fmt: skip
    yl = test["yardline_100"].to_numpy().astype(int)
    made = (test["field_goal_result"] == "made").to_numpy().astype(float)
    made_dev = train.filter((pl.col("play_type") == "field_goal")
                            & (pl.col("field_goal_result") == "made"))  # fmt: skip
    dev_cap = int(made_dev["yardline_100"].max()) + cfg["field_goal"]["snap_to_kick_yards"]
    out: dict = {"heldout_attempts": int(made.size), "longest_made_in_development_seasons": dev_cap,
                 "heldout_attempts_beyond_development_cap": int((yl + 18 > dev_cap).sum()),
                 "heldout_makes_beyond_development_cap": int(made[yl + 18 > dev_cap].sum())}
    for name, cap in (("config_cap", cfg["field_goal"]["max_kick_distance"]),
                      ("development_cap", dev_cap)):  # fmt: skip
        model = cm.FieldGoalModel(**{**cfg["field_goal"], "max_kick_distance": cap}).fit(train)
        p = np.clip(model.p_make(yl), 1e-6, 1 - 1e-6)
        out[name] = {
            "max_kick_distance": cap,
            "heldout_log_loss": float(-np.mean(made * np.log(p) + (1 - made) * np.log(1 - p))),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-refits", action="store_true",
                        help="keep the saved win-probability and bundle sections, redo the rest")
    args = parser.parse_args()
    path = RESULTS / "research" / "leakage_checks.json"
    plays = data.load_plays()
    cfg = wp.load_config()
    out: dict = read_json(path) if args.skip_refits and path.exists() else {}
    out["threads"] = 1
    out["seasons"] = [int(plays["season"].min()), int(plays["season"].max())]
    out["splits"] = check_splits(plays, cfg)
    out["pre_play_columns"] = check_pre_play(plays)
    out["end_game_rows"] = check_end_game_rows(plays)
    out["features"] = check_features()
    out["field_goal_cap"] = check_field_goal_cap(plays)
    if not args.skip_refits:
        out["components"] = check_components(plays)
        out["win_probability"] = check_wp(plays, cfg)
    write_json(out, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
