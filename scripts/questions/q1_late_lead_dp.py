"""Burning question 1, answered from the exactly solved game.

"Leading by 4 with about 4:00 left, fourth down in your own half: go for it or punt?"

Run:  .venv/bin/python scripts/questions/q1_late_lead_dp.py   (needs scripts/solve_mdp.py first)

Writes results/questions/q1_late_lead_dp.json and assets/questions/q1_late_lead_dp.png:
  * the map: best action and win probability of going minus punting, over yards to go,
    field position, time left, lead and the opponent's timeouts;
  * the headline case taken apart (convert or not, what happens next), exact values from
    the solution plus event frequencies from simulating the solved policy;
  * sensitivity: the same answer when the model is changed where it is least certain;
  * side by side: the one-step engine (M2) and nfl4th's numbers (theirs, not ours).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from typing import Any  # noqa: E402

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from fourth_and_rl import components as cm  # noqa: E402
from fourth_and_rl import mdp, style  # noqa: E402
from fourth_and_rl import mdp_game as mg  # noqa: E402
from fourth_and_rl.components import ComponentModels  # noqa: E402
from fourth_and_rl.io import read_json, write_json  # noqa: E402
from fourth_and_rl.paths import ASSETS, RESULTS  # noqa: E402

OUT = RESULTS / "questions" / "q1_late_lead_dp.json"
FIG = ASSETS / "questions" / "q1_late_lead_dp.png"
ONE_STEP = RESULTS / "questions" / "q1_late_lead_one_step.json"
NFL4TH = RESULTS / "reference" / "nfl4th_scenarios.csv"
PLAY_CODES = (mg.A_NEUTRAL, mg.A_PROTECT, mg.A_CHASE)
R = mg.DRIVE_RESULTS.index


def clock(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def action_maps(sol: mg.Solution, q: dict) -> dict[str, np.ndarray]:
    """go, punt, fg values on the grid [time, lead, defense timeouts, distance, yard line]."""
    g, m = sol.grid, sol.model
    yls = np.arange(q["yardlines"][0], q["yardlines"][1] + 1)
    ds = np.arange(q["distances"][0], q["distances"][1] + 1)
    f = g.fidx(4, g.dist_bin(ds)[:, None], yls[None, :])
    shape = (len(q["seconds"]), len(q["leads"]), 4, len(ds), len(yls))
    out = {k: np.zeros(shape) for k in ("go", "punt", "field_goal")}
    mode = np.zeros(shape, dtype=np.int64)
    a = q["offense_timeouts"]
    for i, sec in enumerate(q["seconds"]):
        qs = sol.q_slice(sec // m.clock.tick_seconds)
        for j, lead in enumerate(q["leads"]):
            s = lead + m.score_max
            plays = np.stack([qs[c][f, s, a, :] for c in PLAY_CODES])  # (3, d, yl, b)
            out["go"][i, j] = np.moveaxis(plays.max(axis=0), -1, 0)
            mode[i, j] = np.moveaxis(np.asarray(PLAY_CODES)[plays.argmax(axis=0)], -1, 0)
            out["punt"][i, j] = np.moveaxis(qs[mg.A_PUNT][f, s, a, :], -1, 0)
            out["field_goal"][i, j] = np.moveaxis(qs[mg.A_FG][f, s, a, :], -1, 0)
    out["go_mode"] = mode
    return out


def first_and_next(drives: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Row of each game's first drive, and for every drive the row of the next drive of the
    same game (-1 if none)."""
    k = len(drives["game"])
    order = np.lexsort((np.arange(k), drives["game"]))
    gs = drives["game"][order]
    first = np.full(n, -1)
    start = np.concatenate([[True], gs[1:] != gs[:-1]])
    first[gs[start]] = order[start]
    nxt = np.full(k, -1)
    same = gs[1:] == gs[:-1]
    nxt[order[:-1][same]] = order[1:][same]
    return first, nxt


def branch(drives: dict, result: np.ndarray, opp: np.ndarray, nxt: np.ndarray, tick: int) -> dict:
    """What happens once the trailing team has the ball: `opp` are rows of its drive."""
    if len(opp) == 0:
        return {"n": 0}
    game = drives["game"][opp]
    td = drives["result"][opp] == R("touchdown")
    win = result[game]
    back = nxt[opp]
    has_back = (back >= 0) & (drives["side"][np.maximum(back, 0)] == 0)
    back_sec = np.where(has_back, drives["t"][np.maximum(back, 0)] * tick, 0)
    out = {
        "n": int(len(opp)),
        "opponent_starts_at_own_yardline_mean": float(100 - drives["yl"][opp].mean()),
        "seconds_left_when_opponent_gets_ball_mean": float(drives["t"][opp].mean() * tick),
        "opponent_touchdown_on_that_drive": float(td.mean()),
        "leader_wins_if_no_touchdown": float(win[~td].mean()) if (~td).any() else None,
    }
    if td.any():
        out["after_opponent_touchdown"] = {
            "leader_gets_ball_back": float(has_back[td].mean()),
            "leader_gets_ball_back_with_30s_or_more": float((back_sec[td] >= 30).mean()),
            "seconds_left_then_median": float(np.median(back_sec[td & has_back]))
            if (td & has_back).any()
            else None,
            "leader_wins": float(win[td].mean()),
        }
    return out


def decompose(sim: mg.Simulator, h: dict, q: dict, n: int, seed: int, policy: str) -> dict:
    """Simulate the headline case with each first action forced, then read the branches."""
    sol, g, m = sim.sol, sim.grid, sim.model
    tick = m.clock.tick_seconds
    t = h["seconds"] // tick
    f = int(g.fidx(4, g.dist_bin(h["ydstogo"]), h["yardline_100"]))
    state = mg.GameState(
        h["seconds"],
        h["lead"],
        4,
        h["ydstogo"],
        h["yardline_100"],
        q["offense_timeouts"],
        h["defense_timeouts"],
    )
    split = sol.play_split(state)
    out: dict[str, Any] = {"policy_after_the_decision": policy, "n_games_per_action": n}
    for name, code in (("go", mg._MODE_CODE[split["mode"]]), ("punt", mg.A_PUNT)):
        rng = np.random.default_rng(seed)
        st = sim.start(n, t=t, s=h["lead"], f=f, a=q["offense_timeouts"], b=h["defense_timeouts"])
        res = mg.simulate(
            sim, st, rng, policy=policy, first_action=np.full(n, code), log_drives=True
        )
        d, result = res["drives"], res["result"]
        first, nxt = first_and_next(d, n)
        se = float(result.std() / np.sqrt(n))
        entry: dict[str, Any] = {"win_probability": float(result.mean()), "standard_error": se}
        if name == "go":
            lost = (d["plays"][first] == 1) & np.isin(
                d["result"][first], (R("downs"), R("turnover"), R("opp_touchdown"), R("safety"))
            )
            conv = ~lost
            kept_all_the_way = conv & (nxt[first] < 0)
            opp_after_conv = nxt[first][conv & (nxt[first] >= 0)]
            opp_after_conv = opp_after_conv[d["side"][opp_after_conv] == 1]
            entry["p_convert"] = float(conv.mean())
            entry["if_converted"] = {
                "win_probability": float(result[conv].mean()),
                "leader_never_gives_the_ball_back": float(kept_all_the_way[conv].mean()),
                "seconds_left_when_opponent_next_has_ball_median": float(
                    np.median(d["t"][opp_after_conv] * tick)
                )
                if len(opp_after_conv)
                else None,
                "if_opponent_gets_the_ball": branch(d, result, opp_after_conv, nxt, tick),
            }
            opp = nxt[first][lost & (nxt[first] >= 0)]
            opp = opp[d["side"][opp] == 1]
            entry["if_failed"] = {
                "win_probability": float(result[lost].mean()),
                **branch(d, result, opp, nxt, tick),
            }
        else:
            opp = nxt[first][nxt[first] >= 0]
            opp = opp[d["side"][opp] == 1]
            entry.update(branch(d, result, opp, nxt, tick))
        out[name] = entry
    out["go_minus_punt"] = out["go"]["win_probability"] - out["punt"]["win_probability"]
    out["go_minus_punt_standard_error"] = float(
        np.hypot(out["go"]["standard_error"], out["punt"]["standard_error"])
    )
    return out


def headline_exact(sol: mg.Solution, h: dict, q: dict) -> dict:
    state = mg.GameState(
        h["seconds"],
        h["lead"],
        4,
        h["ydstogo"],
        h["yardline_100"],
        q["offense_timeouts"],
        h["defense_timeouts"],
    )
    qv = sol.q_values(state)
    split = sol.play_split(state)
    go = max(v for k, v in qv.items() if k.startswith("play"))
    return {
        "go": go,
        "punt": qv["punt"],
        "go_minus_punt": go - qv["punt"],
        "best_action": "go" if go > qv["punt"] else "punt",
        "go_mode": split["mode"],
        "p_convert": split["p_keep_or_score"],
        "win_probability_if_converted": split["value_if_kept"],
        "win_probability_if_failed": split["value_if_lost"],
        "break_even_p_convert": (qv["punt"] - split["value_if_lost"])
        / (split["value_if_kept"] - split["value_if_lost"]),
        "all_action_values": qv,
    }


def map_summary(maps: dict[str, np.ndarray], q: dict) -> dict:
    """For each (time, lead, timeouts): share of cells where going beats punting, and the
    longest distance worth going for at three field positions."""
    diff = maps["go"] - maps["punt"]
    yls = np.arange(q["yardlines"][0], q["yardlines"][1] + 1)
    out = {}
    for i, sec in enumerate(q["seconds"]):
        for j, lead in enumerate(q["leads"]):
            for b in range(4):
                d = diff[i, j, b]
                entry = {
                    "share_go_beats_punt": float((d > 0).mean()),
                    "max_gain_from_going": float(d.max()),
                    "max_loss_from_going": float(d.min()),
                }
                for own in (20, 30, 40, 50):
                    col = d[:, int(np.nonzero(yls == 100 - own)[0][0])]
                    ok = np.nonzero(col > 0)[0]
                    entry[f"own_{own}_go_up_to_yards"] = int(ok.max() + 1) if len(ok) else 0
                    entry[f"own_{own}_go_minus_punt_by_distance"] = [
                        round(float(x), 4) for x in col
                    ]
                out[f"up_{lead}__{sec}s__opp_timeouts_{b}"] = entry
    return out


def defense_rate_label(cfg: dict) -> str:
    rates = cfg["clock"]["defense_timeout_rate_before_warning"][1:]
    rates += cfg["clock"]["defense_timeout_rate_after_warning"][1:]
    return (
        f"defense calls timeouts at measured rates ({100 * min(rates):.0f}% to "
        f"{100 * max(rates):.0f}%), not optimally"
    )


KICKOFF_SCENARIO = "every kickoff starts at the touchback spot (no returns)"


class _FixedKickoff:
    """The component models with the kickoff table replaced by a fixed touchback spot."""

    def __init__(self, models: Any, spot: int) -> None:
        self._models = models
        self._table = cm.constant_kickoff(spot)

    def kickoff_start(self, season: int | None = None) -> np.ndarray:
        return self._table.copy()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)


def sensitivity(cfg: dict, models: Any, tables: dict, q: dict, base_maps: dict) -> list[dict]:
    """Re-solve the last 5:00 with one assumption changed at a time."""
    h = q["headline"]
    scenarios: list[tuple[str, dict, dict]] = [("base model", {}, {})]
    for shift in q["convert_shifts"]:
        scenarios.append(
            (f"fourth-down conversion probability {shift:+.2f}", {"go_convert_shift": shift}, {})
        )
    scenarios.append((defense_rate_label(cfg), {"defense_timeout": "empirical"}, {}))
    for tilt in q["chase_tilts"]:
        word = "better" if tilt > 0 else "worse"
        scenarios.append(
            (f"trailing offense {word} (yards tilt {tilt:+.2f})", {"chase_gain_tilt": tilt}, {})
        )
    scenarios.append(
        (
            "no protect or chase tables: every play is the generic play",
            {},
            {"calibrate_modes": False},
        )
    )
    scenarios.append((KICKOFF_SCENARIO, {}, {}))
    era = models.kickoff_era()
    spot = models.kickoff_era_info[era]["touchback_yardline_100"]
    rows = []
    for label, kw, table_cfg in scenarios:
        c = {**cfg, "tables": {**cfg["tables"], **table_cfg}}
        used = _FixedKickoff(models, spot) if label == KICKOFF_SCENARIO else models
        model = mg.build_model(c, used, tables, horizon="endgame", **kw)
        sol = mg.solve(model)
        maps = action_maps(sol, q)
        ex = headline_exact(sol, h, q)
        sim = mg.Simulator(sol)
        dec = decompose(sim, h, q, q["sensitivity_games"], q["seed"], "optimal")
        diff = maps["go"] - maps["punt"]
        i, j = q["seconds"].index(h["seconds"]), q["leads"].index(h["lead"])
        rows.append(
            {
                "scenario": label,
                "headline_go": ex["go"],
                "headline_punt": ex["punt"],
                "headline_go_minus_punt": ex["go_minus_punt"],
                "headline_best": ex["best_action"],
                "headline_p_convert": ex["p_convert"],
                "headline_break_even_p_convert": ex["break_even_p_convert"],
                "opponent_touchdown_rate_after_punt_simulated": dec["punt"][
                    "opponent_touchdown_on_that_drive"
                ],
                "share_of_map_where_go_beats_punt": float((diff > 0).mean()),
                "share_of_headline_panel_where_go_beats_punt": float(
                    (diff[i, j, h["defense_timeouts"]] > 0).mean()
                ),
                "cells_whose_best_action_differs_from_base": int(
                    ((diff > 0) != ((base_maps["go"] - base_maps["punt"]) > 0)).sum()
                ),
                "cells": int(diff.size),
            }
        )
    return rows


def nfl4th_side_by_side(maps: dict, q: dict) -> dict:
    """nfl4th's numbers for the cells its reference grid shares with ours."""
    ref = pl.read_csv(NFL4TH, null_values="NA")
    yls = np.arange(q["yardlines"][0], q["yardlines"][1] + 1)
    rows = []
    for r in ref.to_dicts():
        sec, lead = r["quarter_seconds_remaining"], r["score_differential"]
        if sec not in q["seconds"] or lead not in q["leads"] or r["nfl4th_go_wp"] is None:
            continue
        i, j, b = q["seconds"].index(sec), q["leads"].index(lead), r["defteam_timeouts_remaining"]
        d, y = r["ydstogo"] - 1, int(np.nonzero(yls == r["yardline_100"])[0][0])
        rows.append(
            {
                "seconds": sec,
                "lead": lead,
                "opp_timeouts": b,
                "ydstogo": r["ydstogo"],
                "yardline_100": r["yardline_100"],
                "nfl4th_go_wp": r["nfl4th_go_wp"],
                "nfl4th_punt_wp": r["nfl4th_punt_wp"],
                "nfl4th_go_minus_punt": r["nfl4th_go_wp"] - r["nfl4th_punt_wp"],
                "dp_go_wp": float(maps["go"][i, j, b, d, y]),
                "dp_punt_wp": float(maps["punt"][i, j, b, d, y]),
                "dp_go_minus_punt": float(maps["go"][i, j, b, d, y] - maps["punt"][i, j, b, d, y]),
            }
        )
    if not rows:
        return {"rows": 0}
    a = np.array([r["nfl4th_go_minus_punt"] for r in rows])
    b = np.array([r["dp_go_minus_punt"] for r in rows])
    return {
        "source": "nfl4th 1.0.7 via scripts/r/nfl4th_reference.R, "
        "results/reference/nfl4th_scenarios.csv. These are nfl4th's numbers.",
        "rows": len(rows),
        "note": "only cells present in both grids where nfl4th returned a go value "
        "(it returned NA at yardline_100 50 and 60)",
        "nfl4th_says_go": float((a > 0).mean()),
        "dp_says_go": float((b > 0).mean()),
        "same_call": float(((a > 0) == (b > 0)).mean()),
        "mean_go_minus_punt": {"nfl4th": float(a.mean()), "dp": float(b.mean())},
        "mean_punt_wp": {
            "nfl4th": float(np.mean([r["nfl4th_punt_wp"] for r in rows])),
            "dp": float(np.mean([r["dp_punt_wp"] for r in rows])),
        },
        "rows_table": rows,
    }


def one_step_comparison(maps: dict, q: dict, exact: dict, dec_coach: dict) -> dict:
    """The M2 one-step engine on the same grid, if it can be loaded."""
    out: dict[str, Any] = {"saved_one_step_answer_found": ONE_STEP.exists()}
    try:
        from fourth_and_rl import decision, wp

        engine = decision.load_engine(config=decision.load_config())
    except Exception as err:  # the engine belongs to another milestone and may be absent
        out["status"] = f"one-step engine not available ({type(err).__name__}: {err})"
        return out
    yls = np.arange(q["yardlines"][0], q["yardlines"][1] + 1)
    ds = np.arange(q["distances"][0], q["distances"][1] + 1)
    sec, lead, to, dd, yy = np.meshgrid(
        q["seconds"], q["leads"], np.arange(4), ds, yls, indexing="ij"
    )
    states = wp.state_frame(
        down=4,
        ydstogo=dd.ravel(),
        yardline_100=yy.ravel(),
        game_seconds_remaining=sec.ravel(),
        score_differential=lead.ravel(),
        posteam_timeouts_remaining=q["offense_timeouts"],
        defteam_timeouts_remaining=to.ravel(),
    )
    ev = engine.evaluate(states)
    go1 = ev["go_wp"].to_numpy().reshape(sec.shape)
    punt1 = ev["punt_wp"].fill_null(np.nan).to_numpy().reshape(sec.shape)
    d1, d2 = go1 - punt1, maps["go"] - maps["punt"]
    ok = ~np.isnan(d1)
    disagree = ok & ((d1 > 0) != (d2 > 0))
    h = q["headline"]
    i, j, b = q["seconds"].index(h["seconds"]), q["leads"].index(h["lead"]), h["defense_timeouts"]
    d, y = h["ydstogo"] - 1, int(np.nonzero(yls == h["yardline_100"])[0][0])
    flat = np.ravel_multi_index((i, j, b, d, y), sec.shape)
    row = ev.row(int(flat), named=True)
    by_time = {
        clock(s): {
            "one_step_share_go": float((d1[k][ok[k]] > 0).mean()),
            "dp_share_go": float((d2[k] > 0).mean()),
            "share_of_cells_that_disagree": float(disagree[k].mean()),
            "mean_punt_wp_one_step": float(np.nanmean(punt1[k])),
            "mean_punt_wp_dp": float(maps["punt"][k].mean()),
            "mean_go_wp_one_step": float(np.nanmean(go1[k])),
            "mean_go_wp_dp": float(maps["go"][k].mean()),
        }
        for k, s in enumerate(q["seconds"])
    }
    by_to = {
        str(k): {
            "one_step_share_go": float((d1[:, :, k][ok[:, :, k]] > 0).mean()),
            "dp_share_go": float((d2[:, :, k] > 0).mean()),
            "one_step_mean_go_minus_punt": float(np.nanmean(d1[:, :, k])),
            "dp_mean_go_minus_punt": float(d2[:, :, k].mean()),
        }
        for k in range(4)
    }
    out.update(
        {
            "status": "computed here from the one-step engine's point estimates "
            "(fourth_and_rl.decision); bootstrap confidence is in the one-step answer file",
            "cells": int(ok.sum()),
            "one_step_share_go": float((d1[ok] > 0).mean()),
            "dp_share_go": float((d2[ok] > 0).mean()),
            "share_of_cells_that_disagree": float(disagree.sum() / ok.sum()),
            "disagreements_one_step_go_dp_punt": int((disagree & (d1 > 0)).sum()),
            "disagreements_one_step_punt_dp_go": int((disagree & (d1 <= 0)).sum()),
            "by_time_left": by_time,
            "by_opponent_timeouts": by_to,
            "headline": {
                "one_step": {
                    "go": row["go_wp"],
                    "punt": row["punt_wp"],
                    "go_minus_punt": row["go_wp"] - row["punt_wp"],
                    "p_convert": row["p_convert"],
                    "wp_if_convert": row["wp_if_convert"],
                    "wp_if_fail": row["wp_if_fail"],
                },
                "dp": {
                    "go": exact["go"],
                    "punt": exact["punt"],
                    "go_minus_punt": exact["go_minus_punt"],
                    "p_convert": exact["p_convert"],
                    "wp_if_convert": exact["win_probability_if_converted"],
                    "wp_if_fail": exact["win_probability_if_failed"],
                },
            },
            "headline_gap_accounting": {
                "dp_go_minus_punt": exact["go_minus_punt"],
                "same_model_coach_like_play_afterwards": dec_coach["go_minus_punt"],
                "one_step_go_minus_punt": row["go_wp"] - row["punt_wp"],
                "part_from_optimal_future_play": exact["go_minus_punt"]
                - dec_coach["go_minus_punt"],
                "part_from_model_differences": dec_coach["go_minus_punt"]
                - (row["go_wp"] - row["punt_wp"]),
                "note": "the second line replaces the optimal policy after the decision by the "
                "coach-like policy inside our own model, which is the assumption a "
                "win-probability model fitted on real games makes implicitly",
            },
            "why_they_differ": [
                "After a conversion the DP plans the rest of the drive: it knows how many more "
                "plays the leader can run, what the opponent's timeouts can and cannot stop, and "
                "the exact kneel-down threshold. The one-step engine hands the new state to a "
                "win-probability model fitted on how such leads turned out on average.",
                "The opponent's timeouts change the DP's answer through the clock arithmetic. The "
                "one-step values barely move with them (compare by_opponent_timeouts).",
                "The DP assumes both teams play its optimal policy from here on. The win-"
                "probability model behind the one-step engine embodies real coaches, who punt and "
                "kick more. That is a difference in the question asked, not an error in either.",
                "The DP is exact only for its own model: identical teams, context tables for "
                "leading and trailing play, no penalties. Where its pieces miss the measured "
                "facts (see checks_against_measured_facts) the one-step number may be closer.",
            ],
        }
    )
    if ONE_STEP.exists():
        saved = read_json(ONE_STEP)
        cells = saved.get("headline", {}).get("selected_cells", [])
        rows = []
        for c in cells:
            yl = 100 - c["own_yardline"]
            if yl not in yls or c["ydstogo"] > ds.max():
                continue
            yi = int(np.nonzero(yls == yl)[0][0])
            g2, p2 = (
                maps["go"][i, j, 3, c["ydstogo"] - 1, yi],
                maps["punt"][i, j, 3, c["ydstogo"] - 1, yi],
            )
            rows.append(
                {
                    "own_yardline": c["own_yardline"],
                    "ydstogo": c["ydstogo"],
                    "one_step_go_minus_punt": c["go_wp"] - c["punt_wp"],
                    "one_step_best": c["best_action"],
                    "one_step_category": c.get("category"),
                    "dp_go_minus_punt": float(g2 - p2),
                    "dp_best": "go" if g2 > p2 else "punt",
                }
            )
        out["saved_one_step_headline_cells"] = rows
    return out


def figure(maps: dict, q: dict, exact: dict, dec: dict, sens: list[dict]) -> None:
    style.apply()
    h = q["headline"]
    cmap = LinearSegmentedColormap.from_list("go_punt", [style.AQUA, "#ecebe6", style.BLUE])
    diff = 100 * (maps["go"] - maps["punt"])
    lim = 12.0
    yls = np.arange(q["yardlines"][0], q["yardlines"][1] + 1)
    j = q["leads"].index(h["lead"])
    fig = plt.figure(figsize=(18, 11.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.05, 0.8], wspace=0.5)
    left = outer[0].subgridspec(4, 4, hspace=0.16, wspace=0.07)
    times = sorted(q["seconds"], reverse=True)
    extent = (100 - yls[-1] - 0.5, 100 - yls[0] + 0.5, 0.5, 10.5)
    im = None
    for r, b in enumerate((3, 2, 1, 0)):
        for c, sec in enumerate(times):
            ax = fig.add_subplot(left[r, c])
            i = q["seconds"].index(sec)
            z = diff[i, j, b][:, ::-1]  # x runs from the own 20 to midfield
            im = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=-lim,
                vmax=lim,
                extent=extent,
                interpolation="nearest",
            )
            xs = 100 - yls[::-1]
            ax.contour(xs, np.arange(1, 11), z, levels=[0.0], colors=[style.INK], linewidths=1.3)
            ax.grid(False)
            ax.set_xticks([25, 35, 45])
            ax.set_yticks([1, 4, 7, 10])
            if r == 3:
                ax.set_xticklabels(["25", "35", "45"])
                ax.set_xlabel("Own yard line")
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(f"{b} opp. timeout{'s' if b != 1 else ''}\nYards to go")
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.text(
                    0.5,
                    1.04,
                    f"{clock(sec)} left",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    color=style.INK,
                )
            if sec == h["seconds"] and b == h["defense_timeouts"]:
                ax.scatter(
                    [100 - h["yardline_100"]],
                    [h["ydstogo"]],
                    s=70,
                    facecolor="none",
                    edgecolor=style.INK,
                    linewidths=1.6,
                    zorder=5,
                )
    pos = fig.axes[0].get_position()
    last = fig.axes[15].get_position()
    cax = fig.add_axes([pos.x0, last.y0 - 0.085, last.x1 - pos.x0, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-12, -8, -4, 0, 4, 8, 12])
    cb.set_ticklabels(["punt +12", "+8", "+4", "even", "+4", "+8", "go +12"])
    cb.set_label(
        "Win probability of going minus punting (points). Black line: break-even", color=style.INK_2
    )
    cb.outline.set_visible(False)
    fig.text(
        pos.x0,
        0.955,
        f"Up {h['lead']} late, fourth down in your own half: go or punt?",
        fontsize=16,
        fontweight="bold",
        color=style.INK,
        ha="left",
    )
    fig.text(
        pos.x0,
        0.93,
        "Exact solution of the game model. Blue: going for it wins more "
        "often. Green: punting does. Circle: the headline case.",
        fontsize=10.5,
        color=style.INK_2,
        ha="left",
    )

    right = outer[1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.42)
    ax = fig.add_subplot(right[0])
    go_d, punt_d = dec["go"], dec["punt"]
    p = exact["p_convert"]
    td_punt = punt_d["opponent_touchdown_on_that_drive"]
    td_fail = go_d["if_failed"]["opponent_touchdown_on_that_drive"]
    keeps = go_d["if_converted"]["leader_never_gives_the_ball_back"]
    bars = [
        (
            f"Punt\n(opponent TD on {100 * td_punt:.0f}%\nof the next drives)",
            exact["punt"],
            style.ACTION_COLORS["punt"],
        ),
        ("Go for it, overall", exact["go"], style.ACTION_COLORS["go"]),
        (
            f"Go and fail ({100 * (1 - p):.0f}%)\n(opponent TD on {100 * td_fail:.0f}%\n"
            "of the next drives)",
            exact["win_probability_if_failed"],
            style.SEQUENTIAL[1],
        ),
        (
            f"Go and convert ({100 * p:.0f}%)\n(keeps the ball to the\n"
            f"end {100 * keeps:.0f}% of the time)",
            exact["win_probability_if_converted"],
            style.SEQUENTIAL[3],
        ),
    ]
    ypos = np.arange(len(bars))
    ax.barh(ypos, [100 * v for _, v, _ in bars], color=[c for _, _, c in bars], height=0.62)
    for yv, (_, v, _) in zip(ypos, bars, strict=True):
        ax.text(100 * v + 1.0, yv, f"{100 * v:.1f}%", va="center", fontsize=10.5, color=style.INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for lab, _, _ in bars], fontsize=9.5)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Leader's win probability (%)")
    ax.grid(axis="y", visible=False)
    own = 100 - h["yardline_100"]
    style.title(
        ax,
        f"The headline case: fourth and {h['ydstogo']} at the own {own}",
        f"Up {h['lead']}, {clock(h['seconds'])} left, opponent has "
        f"{h['defense_timeouts']} timeouts",
    )

    ax = fig.add_subplot(right[1])
    labels = {
        "base model": "Base model",
        defense_rate_label(mdp.load_config()): "Defense uses timeouts\nas real teams do",
        KICKOFF_SCENARIO: "Kickoffs to the touchback\nspot (no returns)",
        "no protect or chase tables: every play is the generic play": (
            "Generic plays only (no\nprotect or chase tables)"
        ),
        "both teams play like real coaches afterwards (simulated)": (
            "Both teams play like real\ncoaches afterwards"
        ),
    }
    for r in sens:
        if r["scenario"].startswith("fourth-down conversion"):
            labels[r["scenario"]] = f"Conversion chance {r['scenario'].split()[-1]}"
        if r["scenario"].startswith("trailing offense"):
            labels[r["scenario"]] = (
                f"Trailing offense {r['scenario'].split()[2]}\n(TD after punt "
                f"{100 * r['opponent_touchdown_rate_after_punt_simulated']:.0f}%)"
            )
    rows = sens[::-1]
    vals = [100 * r["headline_go_minus_punt"] for r in rows]
    colors = [style.ACTION_COLORS["go"] if v > 0 else style.ACTION_COLORS["punt"] for v in vals]
    ypos = np.arange(len(rows))
    ax.barh(ypos, vals, color=colors, height=0.62)
    ax.axvline(0, color=style.INK, linewidth=1)
    span = max(abs(min(vals)), abs(max(vals)))
    for yv, v in zip(ypos, vals, strict=True):
        ax.text(
            v + (0.03 * span if v >= 0 else -0.03 * span),
            yv,
            f"{v:+.1f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=10,
            color=style.INK,
        )
    ax.set_yticks(ypos)
    names = [labels.get(r["scenario"], r["scenario"].capitalize()) for r in rows]
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(min(min(vals), 0) - 0.25 * span, max(max(vals), 0) + 0.25 * span)
    ax.set_xlabel("Going minus punting, headline case (win-probability points)")
    ax.grid(axis="y", visible=False)
    style.title(
        ax,
        "How the answer moves with the assumptions",
        f"Break-even conversion chance {100 * exact['break_even_p_convert']:.0f}%, "
        f"model {100 * exact['p_convert']:.0f}%",
    )
    style.save(fig, FIG)


def main() -> None:
    cfg = mdp.load_config()
    q = cfg["question1"]
    h = q["headline"]
    sol = mg.load_solution()
    bundle = joblib.load(mg.SOLUTION_DIR / "tables.joblib")
    models = ComponentModels.load()  # the development bundle the game was solved with
    tick = sol.model.clock.tick_seconds

    maps = action_maps(sol, q)
    exact = headline_exact(sol, h, q)
    sim = mg.Simulator(sol, coach=bundle["coach"])
    dec = decompose(sim, h, q, q["decomposition_games"], q["seed"], "optimal")
    dec_coach = decompose(sim, h, q, q["decomposition_games"], q["seed"], "coach")
    sens = sensitivity(cfg, models, bundle["tables"], q, maps)
    sens.append(
        {
            "scenario": "both teams play like real coaches afterwards (simulated)",
            "headline_go": dec_coach["go"]["win_probability"],
            "headline_punt": dec_coach["punt"]["win_probability"],
            "headline_go_minus_punt": dec_coach["go_minus_punt"],
            "headline_go_minus_punt_standard_error": dec_coach["go_minus_punt_standard_error"],
            "headline_best": "go" if dec_coach["go_minus_punt"] > 0 else "punt",
            "opponent_touchdown_rate_after_punt_simulated": dec_coach["punt"][
                "opponent_touchdown_on_that_drive"
            ],
            "note": "Monte Carlo, not a re-solve: the base model's transitions with the coach-like "
            "policy (observed fourth-down behavior, measured timeout rates) after the "
            "decision",
        }
    )
    validation = RESULTS / "m4_simulator_validation.json"
    facts = None
    if validation.exists():
        v = read_json(validation)
        facts = {
            "source": "results/m4_simulator_validation.json (40,000 simulated games) against "
            "real test-season drives (out of sample) from "
            "results/research/endgame_clock_stats_test.json, and development-season drives "
            "(in sample) from endgame_clock_stats_dev.json",
            "what": "drives by a team down 4 to 8 starting with 1:00 to 5:00 left",
            "real": v["late_trailing_drives_real"],
            "real_in_sample": v.get("late_trailing_drives_real_in_sample"),
            "model_coach_like_policy": v["late_trailing_drives_coach"],
            "model_optimal_policy": v["late_trailing_drives_optimal"],
        }
    yls = np.arange(q["yardlines"][0], q["yardlines"][1] + 1)
    best = np.where(
        maps["field_goal"] > np.maximum(maps["go"], maps["punt"]),
        "field_goal",
        np.where(maps["go"] > maps["punt"], "go", "punt"),
    )
    out = {
        "question": "Leading late, fourth down in your own half: go for it or punt?",
        "method": "exact backward induction on the game model (M4); see docs/notes/m4_mdp.md",
        "seed": q["seed"],
        "assumptions": {
            "offense_timeouts": q["offense_timeouts"],
            "teams": "identical league-average teams",
            "opponent": "plays the optimal policy of the same model, including its timeouts",
        },
        "headline": {
            "situation": h,
            "exact": exact,
            "decomposition_simulated_optimal_play": dec,
            "decomposition_simulated_coach_like_play": dec_coach,
        },
        "sensitivity": sens,
        "sensitivity_note": "each row re-solves the last 5:00 with one assumption changed; "
        "values are exact for that changed model. The spread across rows, "
        "not any single row, is the answer.",
        "checks_against_measured_facts": facts,
        "map_summary": map_summary(maps, q),
        "grid": {
            "seconds": q["seconds"],
            "leads": q["leads"],
            "opponent_timeouts": [0, 1, 2, 3],
            "ydstogo": list(range(q["distances"][0], q["distances"][1] + 1)),
            "yardline_100": yls.tolist(),
            "axes": "[seconds][lead][opponent timeouts][ydstogo][yardline_100]",
            "go_wp": np.round(maps["go"], 4).tolist(),
            "punt_wp": np.round(maps["punt"], 4).tolist(),
            "best_action": best.tolist(),
            "go_mode": np.asarray(mg.ACTION_NAMES)[maps["go_mode"]].tolist(),
        },
        "one_step_comparison": one_step_comparison(maps, q, exact, dec_coach),
        "nfl4th_side_by_side": nfl4th_side_by_side(maps, q),
        "tick_seconds": tick,
    }
    write_json(out, OUT)
    figure(maps, q, exact, dec, sens)
    print("wrote", OUT, "and", FIG)
    print(
        f"headline: go {exact['go']:.4f}  punt {exact['punt']:.4f}  "
        f"diff {exact['go_minus_punt']:+.4f}  p_convert {exact['p_convert']:.3f}  "
        f"break-even {exact['break_even_p_convert']:.3f}"
    )
    for r in sens:
        print(
            f"  {r['scenario'][:60]:60s} {r['headline_go_minus_punt']:+.4f} "
            f"map go share {r.get('share_of_map_where_go_beats_punt', float('nan')):.3f} "
            f"opp TD after punt {r['opponent_touchdown_rate_after_punt_simulated']:.3f}"
        )


if __name__ == "__main__":
    main()
