"""Research: fourth-down aggressiveness by team, and how much offense quality moves conversion.

Produces every number quoted in docs/research/public_debate_and_teams.md and saves them to
results/research/team_aggressiveness_stats.json. Descriptive statistics only: no win
probability is computed here.

Run:  .venv/bin/python scripts/research/team_aggressiveness_stats.py

What is measured
  go_rates         go rate by team and season (regular season), all fourth downs and "neutral"
                   ones, with ranks; the most and least aggressive teams, 2021 to 2025
  detroit          Detroit 2021 to 2025: decisions, go rate, conversion rate with Wilson interval
  offense_gap      short-yardage (third or fourth and 1 to 3) conversion for top-5 versus
                   bottom-5 offenses, where offense strength is early-down EPA per play from
                   games strictly before the game being evaluated, with a previous-season prior
  noise            observed spread of team-season conversion rates against the binomial spread
                   expected if every team had the league rate; empirical-Bayes shrinkage factor
  break_even       expected-points break-even conversion probability in four example spots and
                   how far the break-even distance moves with the offense gap
  leakage_test     the prior-games-only feature is unchanged when later games are scrambled

Definitions
  neutral          fourth and 1 to 3, at or beyond the own 40 (yardline_100 <= 60), score within
                   8 points, quarters 1 to 3
  short yardage    third-down run or pass plays, and fourth-down go plays, with 1 to 3 to go
  early-down EPA   nflfastR `epa` (their model, not ours) on first- and second-down runs and
                   passes; offense strength is the mean per play, defense strength the mean
                   allowed per play

All resampling uses the fixed seed below and resamples whole games (or whole team-seasons).
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from fourth_and_rl.data import load_plays  # noqa: E402
from fourth_and_rl.paths import RESULTS  # noqa: E402

OUT = RESULTS / "research" / "team_aggressiveness_stats.json"

SEED = 20261117
N_BOOT = 2000  # game-level resamples for stratified differences
N_BOOT_FIT = 300  # game-level resamples for the logistic fits
PRIOR_PLAYS = 300  # pseudo-plays given to the previous-season prior (about five games)
PRIOR_CARRY = 0.5  # fraction of last season's EPA per play carried into the prior
TOP_N = 5
RECENT = (2021, 2025)
Z95 = 1.959964


def wilson(k: int, n: int) -> list[float | None]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + Z95**2 / n
    mid = (p + Z95**2 / (2 * n)) / den
    half = Z95 * np.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2)) / den
    return [round(float(mid - half), 4), round(float(mid + half), 4)]


def pct(lo_hi: np.ndarray) -> list[float]:
    """2.5th and 97.5th percentiles of bootstrap draws."""
    lo, hi = np.percentile(lo_hi, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


# --------------------------------------------------------------------------------------
# 1. Go rates
# --------------------------------------------------------------------------------------


def neutral_expr() -> pl.Expr:
    return (
        pl.col("ydstogo").is_between(1, 3)
        & (pl.col("yardline_100") <= 60)
        & (pl.col("score_differential").abs() <= 8)
        & (pl.col("qtr") <= 3)
    )


def go_table(dec: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Decisions, goes and conversions per group, for all and for neutral fourth downs."""
    is_go = pl.col("fourth_down_decision") == "go"
    neu = neutral_expr()
    return (
        dec.group_by(keys)
        .agg(
            n=pl.len(),
            go=is_go.sum(),
            conv=(is_go & (pl.col("fourth_down_converted") == 1)).sum(),
            n_neutral=neu.sum(),
            go_neutral=(is_go & neu).sum(),
            conv_neutral=(is_go & neu & (pl.col("fourth_down_converted") == 1)).sum(),
        )
        .with_columns(
            go_rate=pl.col("go") / pl.col("n"),
            go_rate_neutral=pl.col("go_neutral") / pl.col("n_neutral"),
        )
        .sort(keys)
    )


def go_rates(dec_reg: pl.DataFrame) -> dict:
    league = go_table(dec_reg, ["season"])
    ts = go_table(dec_reg, ["season", "posteam"]).with_columns(
        rank=pl.col("go_rate").rank("min", descending=True).over("season"),
        rank_neutral=pl.col("go_rate_neutral").rank("min", descending=True).over("season"),
    )
    recent = dec_reg.filter(pl.col("season").is_between(*RECENT))
    pooled = go_table(recent, ["posteam"]).with_columns(
        rank=pl.col("go_rate").rank("min", descending=True),
        rank_neutral=pl.col("go_rate_neutral").rank("min", descending=True),
    )

    def rows(df: pl.DataFrame, cols: list[str]) -> list[dict]:
        return [
            {c: (round(v, 4) if isinstance(v, float) else v) for c, v in r.items()}
            for r in df.select(cols).to_dicts()
        ]

    base = ["n", "go", "go_rate", "n_neutral", "go_neutral", "go_rate_neutral"]
    ranked = base + ["rank", "rank_neutral"]
    top_seasons = ts.sort("go_rate", descending=True).head(10)
    bottom_seasons = ts.sort("go_rate").head(10)
    return {
        "scope": "regular season; decisions are fourth downs labelled go, field_goal or punt",
        "neutral_definition": "fourth and 1-3, yardline_100 <= 60, |score diff| <= 8, qtr 1-3",
        "league_by_season": rows(league, ["season"] + base),
        "team_season": rows(ts, ["season", "posteam"] + ranked),
        "pooled_2021_2025_by_neutral_rank": rows(pooled.sort("rank_neutral"), ["posteam"] + ranked),
        "top10_team_seasons_by_go_rate": rows(top_seasons, ["season", "posteam"] + ranked),
        "bottom10_team_seasons_by_go_rate": rows(bottom_seasons, ["season", "posteam"] + ranked),
        "lions_by_season": rows(
            ts.filter((pl.col("posteam") == "DET") & (pl.col("season") >= 2021)),
            ["season"] + ranked,
        ),
    }


def detroit(dec: pl.DataFrame) -> dict:
    """Detroit 2021 to 2025, regular season and with playoffs added."""
    out = {}
    for label, df in [
        ("regular_season", dec.filter(pl.col("season_type") == "REG")),
        ("regular_plus_playoffs", dec),
    ]:
        det = df.filter((pl.col("posteam") == "DET") & pl.col("season").is_between(*RECENT))
        tab = go_table(det, ["season"]).to_dicts()
        total = go_table(det.with_columns(season=pl.lit(0)), ["season"]).to_dicts()[0]
        total["season"] = "2021-2025"
        rows = []
        for r in tab + [total]:
            rows.append(
                {
                    "season": r["season"],
                    "decisions": r["n"],
                    "go": r["go"],
                    "go_rate": round(r["go_rate"], 4),
                    "go_rate_ci95": wilson(r["go"], r["n"]),
                    "converted": r["conv"],
                    "conversion_rate": round(r["conv"] / r["go"], 4),
                    "conversion_ci95": wilson(r["conv"], r["go"]),
                    "neutral_decisions": r["n_neutral"],
                    "neutral_go": r["go_neutral"],
                    "neutral_go_rate": round(r["go_rate_neutral"], 4),
                    "neutral_go_rate_ci95": wilson(r["go_neutral"], r["n_neutral"]),
                    "neutral_converted": r["conv_neutral"],
                    "neutral_conversion_ci95": wilson(r["conv_neutral"], r["go_neutral"]),
                }
            )
        out[label] = rows
    lg = dec.filter(
        (pl.col("season_type") == "REG")
        & pl.col("season").is_between(*RECENT)
        & (pl.col("fourth_down_decision") == "go")
    )
    k, n = int((lg["fourth_down_converted"] == 1).sum()), lg.height
    out["league_go_conversion_2021_2025_regular"] = {
        "attempts": n,
        "converted": k,
        "rate": round(k / n, 4),
    }
    return out


# --------------------------------------------------------------------------------------
# 2. Offense strength from prior games only
# --------------------------------------------------------------------------------------


def team_game_epa(plays: pl.DataFrame, side: str) -> pl.DataFrame:
    """Early-down EPA sums per team-game. side is 'posteam' (offense) or 'defteam' (defense)."""
    early = plays.filter(
        pl.col("down").is_in([1.0, 2.0])
        & pl.col("play_type").is_in(["run", "pass"])
        & pl.col("epa").is_not_null()
    )
    return (
        early.group_by(["season", "week", "game_id", side])
        .agg(n=pl.len(), s=pl.col("epa").sum())
        .rename({side: "team"})
        .sort(["team", "season", "week"])
    )


def as_of_strength(tg: pl.DataFrame) -> pl.DataFrame:
    """Strength entering each team-game, from strictly earlier games plus a prior.

    strength = (PRIOR_PLAYS * PRIOR_CARRY * last season's EPA per play + EPA summed over this
    season's earlier games) / (PRIOR_PLAYS + plays in this season's earlier games)

    Team-games in a team's first season in the table get no value (no previous season).
    """
    season_tot = tg.group_by(["team", "season"]).agg(pn=pl.col("n").sum(), ps=pl.col("s").sum())
    prev = season_tot.with_columns(
        season=pl.col("season") + 1, prev_mean=pl.col("ps") / pl.col("pn")
    ).select("team", "season", "prev_mean")
    out = (
        tg.sort(["team", "season", "week"])
        .with_columns(
            cum_n=pl.col("n").cum_sum().over(["team", "season"]) - pl.col("n"),
            cum_s=pl.col("s").cum_sum().over(["team", "season"]) - pl.col("s"),
        )
        .join(prev, on=["team", "season"], how="left")
        .with_columns(
            strength=(PRIOR_PLAYS * PRIOR_CARRY * pl.col("prev_mean") + pl.col("cum_s"))
            / (PRIOR_PLAYS + pl.col("cum_n"))
        )
    )
    return out.select("season", "week", "game_id", "team", "strength", "n", "s")


def add_week_rank(st: pl.DataFrame) -> pl.DataFrame:
    """Rank each team among all 32 by the strength it carries into each week.

    A team on a bye still has a strength for that week (its latest value), so the rank is
    always out of the full league. Rank 1 is the best offense (or the stingiest defense when
    called on a table whose strength has been negated).
    """
    frames = []
    for (season,), g in st.filter(pl.col("strength").is_not_null()).group_by(["season"]):
        weeks = sorted(g["week"].unique().to_list())
        tot = g.with_columns(
            cum_n=pl.col("n").cum_sum().over("team"), cum_s=pl.col("s").cum_sum().over("team")
        )
        for w in weeks:
            # value entering week w: the strength on the team's first game with week >= w if it
            # plays in week w; otherwise recompute from games before w
            before = tot.filter(pl.col("week") < w)
            first = g.sort("week").group_by("team").first().select("team", "strength")
            last_before = before.sort("week").group_by("team").last()
            # recover prior numerator from the first game's strength (cum is zero there)
            cur = first.join(
                last_before.select("team", "cum_n", "cum_s"), on="team", how="left"
            ).with_columns(
                val=(PRIOR_PLAYS * pl.col("strength") + pl.col("cum_s").fill_null(0.0))
                / (PRIOR_PLAYS + pl.col("cum_n").fill_null(0))
            )
            cur = cur.with_columns(
                rank=pl.col("val").rank("ordinal", descending=True),
                z=(pl.col("val") - pl.col("val").mean()) / pl.col("val").std(),
                season=pl.lit(season, dtype=pl.Int32),
                week=pl.lit(w, dtype=pl.Int32),
                n_teams=pl.len(),
            )
            frames.append(cur.select("season", "week", "team", "val", "rank", "z", "n_teams"))
    return pl.concat(frames)


def short_yardage(plays: pl.DataFrame) -> pl.DataFrame:
    third = (
        (pl.col("down") == 3)
        & pl.col("play_type").is_in(["run", "pass"])
        & pl.col("ydstogo").is_between(1, 3)
    )
    fourth = (pl.col("fourth_down_decision") == "go") & pl.col("ydstogo").is_between(1, 3)
    return plays.filter(third | fourth).with_columns(
        success=pl.when(pl.col("down") == 4)
        .then(pl.col("fourth_down_converted") == 1)
        .otherwise(pl.col("first_down") == 1)
        .cast(pl.Float64),
        stratum=((pl.col("down") - 3) * 3 + pl.col("ydstogo") - 1).cast(pl.Int32),
    )


def stratified_gap(df: pl.DataFrame, group_col: str, rng: np.random.Generator) -> dict:
    """Top group minus bottom group conversion rate at the same down and distance.

    Within each of the six down-by-distance cells the difference in rates is taken, and the
    cells are averaged with weights equal to the cell's share of all top and bottom plays.
    The interval resamples whole games.
    """
    sub = df.filter(pl.col(group_col).is_in(["top", "bottom"]))
    games = sub["game_id"].unique().sort()
    gidx = {g: i for i, g in enumerate(games.to_list())}
    gi = np.array([gidx[g] for g in sub["game_id"].to_list()])
    top = (sub[group_col] == "top").to_numpy().astype(int)
    st = sub["stratum"].to_numpy()
    y = sub["success"].to_numpy()
    n_g = len(gidx)
    cnt = np.zeros((n_g, 2, 6))
    suc = np.zeros((n_g, 2, 6))
    np.add.at(cnt, (gi, top, st), 1.0)
    np.add.at(suc, (gi, top, st), y)

    def estimate(w: np.ndarray) -> tuple[float, float, float]:
        c = np.tensordot(w, cnt, axes=1)
        s = np.tensordot(w, suc, axes=1)
        rate = s / np.maximum(c, 1e-9)
        wt = c.sum(axis=0) / c.sum()
        diff = float(((rate[1] - rate[0]) * wt).sum())
        raw_top = float(s[1].sum() / c[1].sum())
        raw_bot = float(s[0].sum() / c[0].sum())
        return diff, raw_top, raw_bot

    point, raw_top, raw_bot = estimate(np.ones(n_g))
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        draws[b] = estimate(rng.multinomial(n_g, np.full(n_g, 1 / n_g)).astype(float))[0]
    return {
        "n_top": int(cnt[:, 1].sum()),
        "n_bottom": int(cnt[:, 0].sum()),
        "n_games": n_g,
        "raw_rate_top": round(raw_top, 4),
        "raw_rate_bottom": round(raw_bot, 4),
        "stratified_gap": round(point, 4),
        "ci95_game_bootstrap": pct(draws),
        "bootstrap_se": round(float(draws.std(ddof=1)), 4),
        "n_boot": N_BOOT,
    }


def fit_logit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted logistic regression by Newton's method (x already has an intercept column)."""
    beta = np.zeros(x.shape[1])
    for _ in range(25):
        p = 1 / (1 + np.exp(-(x @ beta)))
        grad = x.T @ (w * (y - p))
        hess = (x * (w * p * (1 - p))[:, None]).T @ x
        step = np.linalg.solve(hess + 1e-9 * np.eye(x.shape[1]), grad)
        beta = beta + step
        if np.abs(step).max() < 1e-9:
            break
    return beta


def logit_gap(df: pl.DataFrame, extra: list[str], rng: np.random.Generator) -> dict:
    """Conversion gap implied by a logistic model: down, distance, offense z, plus `extra`.

    The gap is the average predicted conversion probability with offense z set to the mean z
    of top-5 offenses minus the same with z set to the mean z of bottom-5 offenses, all other
    columns left as observed.
    """
    cols = ["off_z"] + extra
    d = df.drop_nulls(cols)
    base = np.column_stack(
        [
            np.ones(d.height),
            (d["down"] == 4).to_numpy().astype(float),
            (d["ydstogo"] == 2).to_numpy().astype(float),
            (d["ydstogo"] == 3).to_numpy().astype(float),
        ]
    )
    x = np.column_stack([base] + [d[c].to_numpy().astype(float) for c in cols])
    y = d["success"].to_numpy()
    z_top = float(d.filter(pl.col("off_group") == "top")["off_z"].mean())
    z_bot = float(d.filter(pl.col("off_group") == "bottom")["off_z"].mean())
    games = d["game_id"].unique().sort().to_list()
    gidx = {g: i for i, g in enumerate(games)}
    gi = np.array([gidx[g] for g in d["game_id"].to_list()])
    k = 4  # column of off_z

    def gap(w: np.ndarray) -> tuple[float, np.ndarray]:
        beta = fit_logit(x, y, w)
        xt, xb = x.copy(), x.copy()
        xt[:, k], xb[:, k] = z_top, z_bot
        pt = 1 / (1 + np.exp(-(xt @ beta)))
        pb = 1 / (1 + np.exp(-(xb @ beta)))
        return float(np.average(pt - pb, weights=w)), beta

    point, beta = gap(np.ones(d.height))
    draws = np.empty(N_BOOT_FIT)
    for b in range(N_BOOT_FIT):
        gw = rng.multinomial(len(games), np.full(len(games), 1 / len(games))).astype(float)
        draws[b] = gap(gw[gi])[0]
    names = ["intercept", "fourth_down", "dist_2", "dist_3"] + cols
    return {
        "n_plays": d.height,
        "n_games": len(games),
        "coefficients_logit": {n: round(float(v), 4) for n, v in zip(names, beta, strict=True)},
        "mean_off_z_top5": round(z_top, 3),
        "mean_off_z_bottom5": round(z_bot, 3),
        "gap_top5_vs_bottom5": round(point, 4),
        "ci95_game_bootstrap": pct(draws),
        "n_boot": N_BOOT_FIT,
    }


def offense_gap(plays: pl.DataFrame, rng: np.random.Generator) -> tuple[dict, pl.DataFrame]:
    off_tg = team_game_epa(plays, "posteam")
    def_tg = team_game_epa(plays, "defteam")
    off_st = as_of_strength(off_tg)
    def_st = as_of_strength(def_tg)
    off_rank = add_week_rank(off_st)
    # defense: lower EPA allowed is better, but keep z in raw units (higher = worse defense)
    def_rank = add_week_rank(def_st)

    def group(col: str) -> pl.Expr:
        return (
            pl.when(pl.col(col) <= TOP_N)
            .then(pl.lit("top"))
            .when(pl.col(col) > pl.col("n_teams") - TOP_N)
            .then(pl.lit("bottom"))
            .otherwise(pl.lit("middle"))
        )

    sy = short_yardage(plays)
    sy = sy.join(
        off_rank.select(
            "season", "week", pl.col("team").alias("posteam"), "n_teams",
            pl.col("rank").alias("off_rank"), pl.col("z").alias("off_z"),
        ),
        on=["season", "week", "posteam"],
        how="left",
    ).join(
        def_rank.select(
            "season", "week", pl.col("team").alias("defteam"), pl.col("z").alias("def_z")
        ),
        on=["season", "week", "defteam"],
        how="left",
    )  # fmt: skip
    sy = sy.with_columns(off_group=group("off_rank"), spread7=pl.col("posteam_spread") / 7.0)

    # the leaky comparison: rank by full-season early-down EPA per play, same games included
    full = (
        off_tg.group_by(["season", "team"])
        .agg(m=pl.col("s").sum() / pl.col("n").sum())
        .with_columns(
            full_rank=pl.col("m").rank("ordinal", descending=True).over("season"),
            full_n=pl.len().over("season"),
        )
    )
    sy = sy.join(
        full.select("season", pl.col("team").alias("posteam"), "full_rank", "full_n"),
        on=["season", "posteam"],
        how="left",
    ).with_columns(
        full_group=pl.when(pl.col("full_rank") <= TOP_N)
        .then(pl.lit("top"))
        .when(pl.col("full_rank") > pl.col("full_n") - TOP_N)
        .then(pl.lit("bottom"))
        .otherwise(pl.lit("middle"))
    )

    has = sy.filter(pl.col("off_rank").is_not_null())
    league_cells = (
        has.group_by(["down", "ydstogo"])
        .agg(n=pl.len(), rate=pl.col("success").mean())
        .sort(["down", "ydstogo"])
    )
    # how well the prior-games-only strength predicts the same game's early-down EPA per play
    chk = off_st.filter(pl.col("strength").is_not_null()).with_columns(
        game_mean=pl.col("s") / pl.col("n")
    )
    r_game = float(np.corrcoef(chk["strength"].to_numpy(), chk["game_mean"].to_numpy())[0, 1])
    yoy = (
        off_tg.group_by(["team", "season"])
        .agg(m=pl.col("s").sum() / pl.col("n").sum())
        .sort(["team", "season"])
    )
    yoy = yoy.join(
        yoy.with_columns(season=pl.col("season") + 1).rename({"m": "m_prev"}),
        on=["team", "season"],
    )
    slope = float(np.polyfit(yoy["m_prev"].to_numpy(), yoy["m"].to_numpy(), 1)[0])

    fourth = has.filter(pl.col("down") == 4)
    third = has.filter(pl.col("down") == 3)
    out = {
        "sample": "third-down runs and passes and fourth-down go plays with 1-3 to go, "
        "2011-2025, regular season and playoffs (2010 has no previous-season prior)",
        "strength_definition": f"early-down EPA per play from this season's earlier games, plus "
        f"{PRIOR_PLAYS} pseudo-plays at {PRIOR_CARRY} x last season's value",
        "n_plays": has.height,
        "n_third": third.height,
        "n_fourth": fourth.height,
        "league_rate_by_cell": [
            {"down": int(r["down"]), "ydstogo": int(r["ydstogo"]), "n": r["n"],
             "rate": round(r["rate"], 4)}
            for r in league_cells.to_dicts()
        ],
        "strength_check": {
            "corr_as_of_strength_with_same_game_early_down_epa": round(r_game, 3),
            "n_team_games": chk.height,
            "year_to_year_slope_of_season_early_down_epa": round(slope, 3),
            "n_team_season_pairs": yoy.height,
        },
        "prior_games_only": {
            "third_and_fourth": stratified_gap(has, "off_group", rng),
            "fourth_only": stratified_gap(fourth, "off_group", rng),
            "third_only": stratified_gap(third, "off_group", rng),
        },
        "leaky_full_season_rank": {
            "third_and_fourth": stratified_gap(has, "full_group", rng),
            "fourth_only": stratified_gap(fourth, "full_group", rng),
        },
        "logit_models": {
            "offense_only": logit_gap(has, [], rng),
            "offense_and_defense": logit_gap(has, ["def_z"], rng),
            "offense_defense_spread": logit_gap(has, ["def_z", "spread7"], rng),
        },
    }  # fmt: skip
    return out, off_tg


# --------------------------------------------------------------------------------------
# 2b. How much of the team-to-team spread is noise
# --------------------------------------------------------------------------------------


def noise_split(df: pl.DataFrame, unit: list[str], label: str, rng: np.random.Generator) -> dict:
    """Method-of-moments split of the spread of unit-level conversion rates.

    For unit i with n_i plays: d_i = (successes_i - expected_i) / n_i, where expected_i sums the
    league rate of each play's season-by-down-by-distance cell. If every unit had the league
    rate, Var(d_i) would be v_i = sum over its plays of p(1 - p), divided by n_i squared.
    tau^2 = Var(d_i) - mean(v_i) estimates the variance of true unit effects.
    Shrinkage factor for a unit = tau^2 / (tau^2 + v_i): the weight an empirical-Bayes estimate
    puts on the unit's own rate, the rest going to the league rate.
    """
    d = df.with_columns(
        dist=pl.col("ydstogo").clip(1, 11).cast(pl.Int32),
    ).with_columns(p=pl.col("success").mean().over(["season", "down", "dist"]))
    u = (
        d.group_by(unit)
        .agg(
            n=pl.len(),
            k=pl.col("success").sum(),
            e=pl.col("p").sum(),
            v=(pl.col("p") * (1 - pl.col("p"))).sum(),
        )
        .sort(unit)
    )
    n = u["n"].to_numpy().astype(float)
    di = (u["k"].to_numpy() - u["e"].to_numpy()) / n
    vi = u["v"].to_numpy() / n**2
    raw = u["k"].to_numpy() / n

    def split(idx: np.ndarray) -> tuple[float, float, float]:
        s2 = float(di[idx].var(ddof=1))
        vbar = float(vi[idx].mean())
        return s2, vbar, s2 - vbar

    s2, vbar, tau2 = split(np.arange(len(n)))
    draws = np.array([split(rng.integers(0, len(n), len(n))) for _ in range(N_BOOT)])
    noise_draws = np.minimum(draws[:, 1] / draws[:, 0], 1.0)
    tau2_pos = max(tau2, 0.0)
    v_med = float(np.median(vi))
    sims = np.sort(rng.standard_normal((20000, 32)), axis=1)
    order_gap = float(sims[:, -TOP_N:].mean() - sims[:, :TOP_N].mean())
    return {
        "label": label,
        "n_units": int(len(n)),
        "plays_per_unit_median": float(np.median(n)),
        "plays_per_unit_min": int(n.min()),
        "sd_raw_rate": round(float(raw.std(ddof=1)), 4),
        "sd_observed_minus_expected": round(float(np.sqrt(s2)), 4),
        "sd_expected_from_binomial_noise": round(float(np.sqrt(vbar)), 4),
        "noise_share_of_variance": round(min(vbar / s2, 1.0), 3),
        "noise_share_ci95_unit_bootstrap": pct(noise_draws),
        "sd_true_effect_tau": round(float(np.sqrt(tau2_pos)), 4),
        "tau_ci95_unit_bootstrap": pct(np.sqrt(np.maximum(draws[:, 2], 0.0))),
        "shrinkage_factor_median_unit": round(tau2_pos / (tau2_pos + v_med), 3),
        "implied_gap_true_top5_vs_bottom5_converters": round(
            order_gap * float(np.sqrt(tau2_pos)), 4
        ),
        "normal_order_statistic_gap_in_sd": round(order_gap, 3),
    }


def noise(plays: pl.DataFrame, rng: np.random.Generator) -> dict:
    reg = plays.filter(pl.col("season_type") == "REG")
    sy = short_yardage(reg)
    go_all = reg.filter(pl.col("fourth_down_decision") == "go").with_columns(
        success=(pl.col("fourth_down_converted") == 1).cast(pl.Float64)
    )
    ts = ["season", "posteam"]
    return {
        "scope": "regular season 2010-2025; expected rate is the league rate in the same "
        "season, down and distance",
        "team_season_third_and_fourth_short": noise_split(
            sy, ts, "third/fourth and 1-3, by team-season", rng
        ),
        "team_season_fourth_short": noise_split(
            sy.filter(pl.col("down") == 4), ts, "fourth and 1-3 go plays, by team-season", rng
        ),
        "team_season_all_fourth_go": noise_split(
            go_all, ts, "all fourth-down go plays, by team-season", rng
        ),
        "team_2021_2025_all_fourth_go": noise_split(
            go_all.filter(pl.col("season").is_between(*RECENT)),
            ["posteam"],
            "all fourth-down go plays, by team, 2021-2025 pooled",
            rng,
        ),
    }


# --------------------------------------------------------------------------------------
# 3. Break-even
# --------------------------------------------------------------------------------------


def break_even(plays: pl.DataFrame, gap: float, gap_resid: float) -> dict:
    """Expected-points break-even for going, in four example spots.

    Notation: y = yards from the opponent's end zone, d = yards to go, E(y) = nflfastR expected
    points for first and 10 at y.
      V_success(d) = E(y - d)                      (convert with exactly d yards)
      V_fail       = -E(100 - y)                   (opponent takes over at the spot)
      V_punt       = -(mean opponent EP on the first play after punts from y +/- 5)
      V_fg         = m * (3 - E(70)) + (1 - m) * (-E(100 - (y + 7)))
                     m = make rate from y +/- 2; a made kick is followed by a kickoff valued at
                     E(70) for the opponent; a miss gives the ball at the spot of the kick
      break-even p* = (V_alt - V_fail) / (V_success - V_fail), V_alt = better of punt and kick
    E(y) uses first and 10 or first and goal.
    The league conversion curve is a logistic fit to fourth-down go plays with 1 to 10 to go
    outside the last two minutes of each half: logit p(d) = a + b * d. A team is treated as
    p(d) plus or minus half the top-versus-bottom gap. The break-even distance solves
    p(d) + shift = p*(d).
    """
    first10 = plays.filter(
        (pl.col("down") == 1)
        & ((pl.col("ydstogo") == 10) | (pl.col("goal_to_go") == 1))
        & pl.col("play_type").is_in(["run", "pass"])
        & pl.col("ep").is_not_null()
    )
    ep_tab = first10.group_by("yardline_100").agg(s=pl.col("ep").sum(), n=pl.len())
    ep_s = np.zeros(101)
    ep_n = np.zeros(101)
    for r in ep_tab.iter_rows(named=True):
        ep_s[int(r["yardline_100"])] = r["s"]
        ep_n[int(r["yardline_100"])] = r["n"]

    def ep10(y: float) -> float:
        lo, hi = int(max(1, round(y) - 2)), int(min(99, round(y) + 2))
        return float(ep_s[lo : hi + 1].sum() / ep_n[lo : hi + 1].sum())

    nxt = plays.sort(["game_id", "order_sequence"]).with_columns(
        next_ep=pl.col("ep").shift(-1).over("game_id"),
        next_pos=pl.col("posteam").shift(-1).over("game_id"),
        next_down=pl.col("down").shift(-1).over("game_id"),
    )
    punts = nxt.filter(
        (pl.col("fourth_down_decision") == "punt")
        & (pl.col("next_pos") == pl.col("defteam"))
        & (pl.col("next_down") == 1)
        & pl.col("next_ep").is_not_null()
    )
    fgs = plays.filter(pl.col("fourth_down_decision") == "field_goal")

    go = plays.filter(
        (pl.col("fourth_down_decision") == "go")
        & pl.col("ydstogo").is_between(1, 10)
        & (pl.col("half_seconds_remaining") > 120)
    )
    xd = go["ydstogo"].to_numpy()
    yv = (go["fourth_down_converted"] == 1).to_numpy().astype(float)
    a, b = fit_logit(np.column_stack([np.ones(len(xd)), xd]), yv, np.ones(len(xd)))
    curve = go.group_by("ydstogo").agg(n=pl.len(), rate=(pl.col("fourth_down_converted") == 1)
                                       .mean()).sort("ydstogo")  # fmt: skip

    def p_league(d: np.ndarray | float) -> np.ndarray | float:
        return 1 / (1 + np.exp(-(a + b * d)))

    grid = np.arange(0.5, 15.001, 0.01)
    spots = []
    for y, name in [
        (65, "own 35"),
        (50, "midfield"),
        (35, "opponent 35"),
        (20, "opponent 20"),
    ]:
        v_fail = -ep10(100 - y)
        p_near = punts.filter(pl.col("yardline_100").is_between(y - 5, y + 5))
        f_near = fgs.filter(pl.col("yardline_100").is_between(y - 2, y + 2))
        v_punt = -float(p_near["next_ep"].mean()) if p_near.height >= 100 else None
        v_fg = None
        make = None
        if f_near.height >= 100:
            make = float((f_near["field_goal_result"] == "made").mean())
            v_fg = make * (3 - ep10(70)) + (1 - make) * (-ep10(100 - (y + 7)))
        alts = {k: v for k, v in [("punt", v_punt), ("field_goal", v_fg)] if v is not None}
        alt = max(alts, key=alts.get)
        v_alt = alts[alt]
        v_succ = np.array([ep10(max(y - d, 1)) for d in grid])
        p_star = (v_alt - v_fail) / (v_succ - v_fail)

        def cross(shift: float, p_star: np.ndarray = p_star) -> float | None:
            ok = (p_league(grid) + shift) >= p_star
            if ok.all():
                return None
            if not ok.any():
                return 0.0
            return round(float(grid[np.argmin(ok)] - 0.01), 2)

        row = {
            "spot": name,
            "yardline_100": y,
            "alternative": alt,
            "v_fail": round(v_fail, 3),
            "v_punt": None if v_punt is None else round(v_punt, 3),
            "n_punts": p_near.height,
            "v_field_goal": None if v_fg is None else round(v_fg, 3),
            "fg_make_rate": None if make is None else round(make, 3),
            "n_field_goals": f_near.height,
            "v_success_at_d2": round(ep10(y - 2), 3),
            "break_even_p_at_d1": round(float(p_star[np.argmin(np.abs(grid - 1))]), 3),
            "break_even_p_at_d2": round(float(p_star[np.argmin(np.abs(grid - 2))]), 3),
            "break_even_p_at_d4": round(float(p_star[np.argmin(np.abs(grid - 4))]), 3),
            "break_even_distance_bottom5": cross(-gap / 2),
            "break_even_distance_average": cross(0.0),
            "break_even_distance_top5": cross(gap / 2),
        }
        lo_r, hi_r = cross(-gap_resid / 2), cross(gap_resid / 2)
        if lo_r is not None and hi_r is not None:
            row["shift_yards_residual_gap_after_spread"] = round(hi_r - lo_r, 2)
        if row["break_even_distance_top5"] is not None:
            row["shift_yards_top_vs_bottom"] = round(
                row["break_even_distance_top5"] - row["break_even_distance_bottom5"], 2
            )
        spots.append(row)

    slope_pts = [
        {"d": d, "p": round(float(p_league(d)), 4),
         "slope_per_yard": round(float(b * p_league(d) * (1 - p_league(d))), 4)}
        for d in [1, 2, 3, 5, 8]
    ]  # fmt: skip
    return {
        "gap_used": round(gap, 4),
        "gap_residual_after_spread": round(gap_resid, 4),
        "gap_residual_source": "offense_gap.logit_models.offense_defense_spread",
        "gap_source": "offense_gap.prior_games_only.third_and_fourth.stratified_gap",
        "ep_source": "nflfastR `ep` column (not this project's model); first-half style, "
        "clock and score ignored",
        "kickoff_value_E70": round(ep10(70), 3),
        "league_curve_logit": {"a": round(float(a), 4), "b": round(float(b), 4), "n": len(xd)},
        "league_curve_empirical": [
            {"ydstogo": int(r["ydstogo"]), "n": r["n"], "rate": round(r["rate"], 4)}
            for r in curve.to_dicts()
        ],
        "league_curve_fitted": slope_pts,
        "spots": spots,
    }


# --------------------------------------------------------------------------------------
# 4. Leakage test
# --------------------------------------------------------------------------------------


def leakage_test(off_tg: pl.DataFrame, rng: np.random.Generator) -> dict:
    """Scramble every team-game at or after a cutoff; earlier features must not move, and the
    feature for the cutoff week itself must not move either (it uses only earlier games)."""
    base = as_of_strength(off_tg)
    results = []
    for season, week in [(2015, 1), (2019, 9), (2023, 12), (2025, 18)]:
        later = (pl.col("season") > season) | (
            (pl.col("season") == season) & (pl.col("week") >= week)
        )
        noise_col = pl.Series("noise", rng.normal(0, 50, off_tg.height))
        scr = off_tg.with_columns(noise_col).with_columns(
            s=pl.when(later).then(pl.col("s") + pl.col("noise")).otherwise(pl.col("s"))
        ).drop("noise")  # fmt: skip
        new = as_of_strength(scr)
        keep = (pl.col("season") < season) | (
            (pl.col("season") == season) & (pl.col("week") <= week)
        )
        a = base.filter(keep).sort(["team", "season", "week"])["strength"].to_numpy()
        b = new.filter(keep).sort(["team", "season", "week"])["strength"].to_numpy()
        after = ~keep
        a2 = base.filter(after).sort(["team", "season", "week"])["strength"].to_numpy()
        b2 = new.filter(after).sort(["team", "season", "week"])["strength"].to_numpy()
        results.append(
            {
                "cutoff": f"{season} week {week}",
                "rows_checked": int(np.isfinite(a).sum()),
                "max_abs_change_at_or_before_cutoff": float(np.nanmax(np.abs(a - b))),
                "rows_changed_after_cutoff": int(np.nansum(np.abs(a2 - b2) > 1e-12)),
                "rows_after_cutoff": int(np.isfinite(a2).sum()),
            }
        )
    return {
        "description": "early-down EPA of all team-games at or after the cutoff is scrambled; "
        "the as-of strength for games up to and including the cutoff week must be unchanged",
        "passed": all(r["max_abs_change_at_or_before_cutoff"] < 1e-12 for r in results),
        "cutoffs": results,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    plays = load_plays()
    dec = plays.filter(pl.col("fourth_down_decision").is_not_null())
    dec_reg = dec.filter(pl.col("season_type") == "REG")

    gap_out, off_tg = offense_gap(plays, rng)
    gap = gap_out["prior_games_only"]["third_and_fourth"]["stratified_gap"]
    gap_resid = gap_out["logit_models"]["offense_defense_spread"]["gap_top5_vs_bottom5"]
    out = {
        "meta": {
            "script": "scripts/research/team_aggressiveness_stats.py",
            "seed": SEED,
            "n_plays": plays.height,
            "seasons": [int(plays["season"].min()), int(plays["season"].max())],
            "n_fourth_down_decisions": dec.height,
            "n_fourth_down_decisions_regular_season": dec_reg.height,
            "note": "descriptive statistics only; EPA and EP columns are nflfastR's, not ours",
        },
        "go_rates": go_rates(dec_reg),
        "detroit": detroit(dec),
        "offense_gap": gap_out,
        "noise": noise(plays, rng),
        "break_even": break_even(plays, gap, gap_resid),
        "leakage_test": leakage_test(off_tg, rng),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
