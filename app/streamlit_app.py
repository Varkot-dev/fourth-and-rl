"""Fourth-down explorer: the interactive app of M7.

Run from the repository root:

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

Every number on screen is read from the precomputed bundle in app/data/ (written by
scripts/m7_build_app_bundle.py); the app fits and solves nothing. Set FOURTH_AND_RL_BUNDLE
to a folder to read another bundle (the tests use a tiny synthetic one).
"""

from __future__ import annotations

import os

import numpy as np
import streamlit as st

from fourth_and_rl import app_bundle as ab
from fourth_and_rl import app_figures as af

LABELS = {"go": "Go for it", "field_goal": "Field goal", "punt": "Punt", None: "n/a"}
CATEGORY_TEXT = {
    "confident": "confident",
    "lean": "a lean",
    "uncertain": "too close to call",
}


@st.cache_resource
def get_bundle(path: str | None) -> ab.Bundle:
    return ab.load_bundle(path)


def bundle_path() -> str | None:
    return os.environ.get("FOURTH_AND_RL_BUNDLE") or None


def app_config() -> dict:
    try:
        return ab.load_config()["app"]
    except FileNotFoundError:
        return {}


# ------------------------------------------------------------------------------ controls


def controls(b: ab.Bundle, default: dict) -> dict:
    clocks = b.clocks
    labels = [c["label"] for c in clocks]
    by_label = {c["label"]: int(c["seconds"]) for c in clocks}
    start = next((c["label"] for c in clocks if int(c["seconds"]) == default.get("seconds")),
                 labels[0])  # fmt: skip
    c1, c2, c3 = st.columns(3)
    with c1:
        label = st.select_slider("Game clock", options=labels, value=start, key="clock")
        seconds = by_label[label]
        scores = b.scores
        score = st.select_slider(
            "Score (your side)", options=scores, key="score",
            value=default.get("score", 0) if default.get("score", 0) in scores else scores[0],
            format_func=af.score_label,
        )  # fmt: skip
    with c2:
        own = st.slider(
            "Ball on: yards from your own goal line", min_value=101 - b.yardlines[-1],
            max_value=100 - b.yardlines[0], key="own",
            value=100 - default.get("yardline_100", 60) if 100 - default.get("yardline_100", 60)
            in range(101 - b.yardlines[-1], 101 - b.yardlines[0]) else 101 - b.yardlines[-1],
        )  # fmt: skip
        yardline = 100 - own
        max_d = min(b.distances[-1], yardline)
        d_default = min(default.get("ydstogo", 1), max_d)
        if max_d > 1:
            ydstogo = st.slider("Yards to go", 1, max_d, d_default, key="togo")
        else:
            ydstogo = 1
            st.caption("Fourth and goal from the 1: one yard to go.")
        st.caption(f"Fourth and {ydstogo} at the **{af.field_label(yardline).lower()}**"
                   + (" (goal to go)" if ydstogo == yardline else ""))  # fmt: skip
    with c3:
        opts = b.options(seconds)
        tos = opts["opp_timeouts"]
        if len(tos) > 1:
            to_default = default.get("opp_timeouts", 3)
            opp = st.radio("Opponent's timeouts", tos, horizontal=True, key="opp",
                           index=tos.index(to_default) if to_default in tos else len(tos) - 1)
        else:
            opp = tos[0]
            st.caption(f"Opponent's timeouts: {opp} (they are a choice only in the last 5:00)")
        spreads = opts["spreads"]
        if len(spreads) > 1:
            spread = st.select_slider(
                "Pregame point spread", options=spreads, key="spread",
                value=0.0 if 0.0 in spreads else spreads[0],
                format_func=lambda s: "Even" if s == 0 else
                (f"You favoured by {s:g}" if s > 0 else f"Opponent favoured by {abs(s):g}"),
            )  # fmt: skip
        else:
            spread = spreads[0]
            st.caption(f"Spread fixed at {spread:+g} at this clock")
        st.caption(f"You hold {b.meta.get('fixed', {}).get('own_timeouts', 3)} timeouts.")
    return {"seconds": seconds, "score": int(score), "yardline_100": int(yardline),
            "ydstogo": int(ydstogo), "opp_timeouts": int(opp), "spread": float(spread)}  # fmt: skip


# --------------------------------------------------------------------------- explanation


def recommendation_text(one: dict, n_boot: int) -> tuple[str, str]:
    """(headline, detail) for the one-step answer."""
    best, cat = one["best"], one["category"]
    if best is None:
        return "No recommendation", "This cell has no one-step value."
    agree = one["boot_counts"][best]
    gain = af.points(one["go_gain"])
    interval = f"[{af.points(one['gain_lo'])}, {af.points(one['gain_hi'])}]"
    detail = (f"{agree} of {n_boot} bootstrap refits agree. Go minus the better kick: {gain} "
              f"win-probability points, 90% interval {interval}.")  # fmt: skip
    if cat == "uncertain":
        return f"Too close to call (point estimate: {LABELS[best].lower()})", detail
    if cat == "lean":
        return f"Lean: {LABELS[best].lower()}", detail
    return LABELS[best], detail


def why_they_differ(one: dict, exact: dict) -> str:
    same = one["best"] == exact["best"]
    gap = exact["go_gain"] - one["go_gain"]
    base = (
        "The one-step answer looks one play ahead and then trusts a win-probability model "
        "fitted on real games, which contains real coaches' later choices and real team "
        "differences. The exact answer solves the whole game for two identical teams that both "
        "play the model's optimal policy afterwards, with the clock and timeouts tracked to the "
        "five seconds."
    )
    if same:
        lead = f"Both methods say **{LABELS[one['best']].lower()}**."
    else:
        lead = (f"They disagree: one step ahead says **{LABELS[one['best']].lower()}**, the full "
                f"game says **{LABELS[exact['best']].lower()}**.")  # fmt: skip
    if np.isfinite(gap):
        lead += (f" The exact game values going {af.points(gap)} points "
                 f"{'more' if gap > 0 else 'less'} than the one-step engine does here.")
    return lead + " " + base


# ---------------------------------------------------------------------------------- tabs


def tab_decision(b: ab.Bundle, default: dict) -> None:
    s = controls(b, default)
    cell = b.lookup(s["seconds"], s["score"], s["yardline_100"], s["ydstogo"],
                    s["opp_timeouts"], s["spread"])  # fmt: skip
    one, ex = cell["one_step"], cell["exact"]
    head, detail = recommendation_text(one, b.n_boot)
    st.markdown(f"#### {cell['clock_label']}, {af.score_label(s['score']).lower()}, fourth and "
                f"{s['ydstogo']} at the {af.field_label(s['yardline_100']).lower()}")  # fmt: skip
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**One play ahead (the standard analysis):** {head}")
        st.caption(detail)
        cols = st.columns(3)
        for col, a in zip(cols, ab.ACTIONS, strict=True):
            w = one["wp"][a]
            with col:
                st.metric(LABELS[a], af.pct(w),
                          help="Win probability of the team with the ball if it chooses this.")
                if np.isfinite(w):
                    st.caption(f"90%: {af.pct(one['wp_lo'][a])} to {af.pct(one['wp_hi'][a])}")
                else:
                    st.caption("not available here" if a != "field_goal" else "out of range")
        st.caption(f"Conversion chance {af.pct(one['p_convert'])}"
                   + (f", field goal good {af.pct(one['p_fg'])}"
                      if np.isfinite(one["p_fg"]) and one["p_fg"] > 0 else "")
                   + ". Intervals of the three actions move together across refits; the "
                     "interval on the difference (above) is the one that decides the call.")
    with right:
        st.markdown(f"**Exact full game (dynamic programming):** {LABELS[ex['best']]}")
        st.caption("  ·  ".join(f"{LABELS[a]} {af.pct(ex['wp'][a])}" for a in ab.ACTIONS))
        st.caption(f"Go minus the better kick: {af.points(ex['go_gain'])} points.")
        if ex["distance_binned"]:
            st.caption("The exact game treats 11 to 15 yards to go as one bin valued at 13.")
        st.plotly_chart(af.action_bars(cell), width="stretch", config={"displayModeBar": False})
    st.info(why_they_differ(one, ex))
    for note in cell["notes"]:
        st.caption(f"Note: {note}.")
    which = st.radio("Colour the field by", ["One play ahead, with bootstrap confidence",
                                             "Exact full game"], horizontal=True, key="which")
    exact = which.startswith("Exact")
    chart = b.chart(s["seconds"], s["score"], s["opp_timeouts"], s["spread"])
    fig = af.field_chart(chart, b.yardlines, b.distances, (s["yardline_100"], s["ydstogo"]),
                         exact=exact,
                         title=f"Every fourth down in this situation ({'exact game' if exact else 'one step ahead'})")  # noqa: E501
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Full colour: 83% or more of the refits agree. Paler: 67% to 83%. Grey: below "
               "67%, too close to call. The ring marks the situation above.")


def tab_late_lead(b: ab.Bundle, cfg: dict) -> None:
    q1 = b.questions.get("q1")
    st.markdown("#### Up late, fourth down in your own half: go for it or punt?")
    if q1:
        e = q1["exact"]
        st.markdown(
            f"Headline (exact game): up {q1['situation']['lead']}, "
            f"{q1['situation']['seconds'] // 60}:00 left, fourth and {q1['situation']['ydstogo']} "
            f"at the own {100 - q1['situation']['yardline_100']}, opponent with "
            f"{q1['situation']['defense_timeouts']} timeouts: **go {af.pct(e['go'])}, punt "
            f"{af.pct(e['punt'])}** ({af.points(e['go_minus_punt'])} points). Going needs a "
            f"{af.pct(e['break_even_p_convert'], 0)} conversion chance to break even; the model "
            f"gives {af.pct(e['p_convert'], 0)}."
        )  # fmt: skip
    leads = [x for x in cfg.get("leads", [1, 3, 4, 7]) if x in b.scores]
    secs = [x for x in cfg.get("seconds", [300, 240, 180, 120])
            if x in [int(c["seconds"]) for c in b.clocks]]  # fmt: skip
    if not leads or not secs:
        st.warning("This bundle does not cover the late-lead grid.")
        return
    c1, c2, c3 = st.columns(3)
    lead = c1.select_slider("Lead", options=leads, value=4 if 4 in leads else leads[0], key="q1l")
    sec = c2.select_slider("Time left", options=secs, value=secs[min(1, len(secs) - 1)],
                           format_func=b.clock_label, key="q1s")  # fmt: skip
    tos = b.options(sec)["opp_timeouts"]
    opp = c3.radio("Opponent's timeouts", tos, index=len(tos) - 1, horizontal=True, key="q1t")
    lo_y, hi_y = cfg.get("yardlines", [50, 99])
    d_hi = cfg.get("distances", [1, 10])[1]
    chart = b.chart(sec, lead, opp, 0.0)
    yl = [y for y in b.yardlines if lo_y <= y <= hi_y]
    cols = [b.yardlines.index(y) for y in yl]
    rows = list(range(min(d_hi, len(b.distances))))
    own = [100 - y for y in yl]
    ticks = [af.field_label(y) for y in yl]
    p = af.palette()
    for title, key, cat in (("One play ahead", "values", True), ("Exact full game", "exact_values",
                                                                 False)):  # fmt: skip
        v = chart[key][np.ix_(rows, cols)]
        z = v[..., 0] - v[..., 2]
        hover = np.empty(z.shape, dtype=object)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                extra = ""
                if cat:
                    c = chart["category"][rows[i], cols[j]]
                    extra = "<br>" + {0: "confident", 1: "lean", 2: "too close to call"}.get(
                        int(c), "")
                hover[i, j] = (f"{ticks[j]}, 4th and {rows[i] + 1}<br>go {af.pct(v[i, j, 0])}, "
                               f"punt {af.pct(v[i, j, 2])}<br>go minus punt "
                               f"{af.points(z[i, j])} points{extra}")  # fmt: skip
        fig = af.diverging_map(z, own, [r + 1 for r in rows], p["actions"]["go"],
                               p["actions"]["punt"], "go", "punt",
                               "Yards from your own goal line", "Yards to go", hover=hover,
                               title=f"{title}: go minus punt, up {lead}, {b.clock_label(sec)}, "
                                     f"opponent {opp} timeouts",
                               limit=15.0, height=340)  # fmt: skip
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if q1:
        st.caption(
            f"Over the question 1 grid (leads 1, 3, 4, 7; 2:00 to 5:00; own 20 to midfield; "
            f"fourth and 1 to 10) the exact game says go in {af.pct(q1['dp_share_go'], 0)} of "
            f"cells and the one-step engine in {af.pct(q1['one_step_share_go'], 0)}. For the "
            f"headline, about {af.points(q1['gap_accounting']['part_from_optimal_future_play'])} "
            "points of the gap come from assuming optimal instead of coach-like play afterwards. "
            "Sources: " + ", ".join(q1["source"]) + "."
        )  # fmt: skip
        rows_c = q1["coaches"]["by_distance"]
        bits = []
        for k, r in rows_c.items():
            if r.get("n"):
                bits.append(f"fourth and {k}: {r['go']} of {r['n']}")
        if bits:
            st.caption("What coaches did in comparable spots (leading by 1 to 8, 1:30 to 7:00 "
                       "left, own half, development seasons): went for it on " + "; ".join(bits)
                       + ".")  # fmt: skip


def tab_goal_line(b: ab.Bundle) -> None:
    q5 = b.questions.get("q5")
    st.markdown("#### Goal line and two-point calls, read from the exact game")
    if not q5:
        st.warning("This bundle holds no question 5 maps.")
        return
    p = af.palette()
    gl = q5["goal_line_late"]
    h = gl["headline"]
    st.markdown(
        f"**Up {h['situation']['lead']}, fourth and goal at the {h['situation']['yardline']}, "
        f"{h['situation']['seconds'] // 60}:00 left:** go {af.pct(h['go'])}, field goal "
        f"{af.pct(h['field_goal'])} ({af.points(h['go_minus_field_goal'])} points; touchdown "
        f"chance {af.pct(h['p_touchdown'], 0)}, "
        f"break-even {af.pct(h['break_even_p_touchdown'], 0)})."
    )  # fmt: skip
    g = gl["grid"]
    c1, c2, c3 = st.columns(3)
    lead = c1.select_slider("Lead", options=g["leads"], value=7, key="q5l")
    opp = c2.radio("Opponent's timeouts", g["opponent_timeouts"], index=3, horizontal=True,
                   key="q5t")  # fmt: skip
    del c3
    li, oi = g["leads"].index(lead), g["opponent_timeouts"].index(opp)
    go_wp = np.array(g["go_wp"])[:, li, oi, :]
    fg_wp = np.array(g["field_goal_wp"])[:, li, oi, :]
    z = (go_wp - fg_wp).T
    clock = [f"{s // 60}:{s % 60:02d}" for s in g["seconds"]]
    fig = af.diverging_map(z, clock, [f"from the {y}" for y in g["yardlines"]],
                           p["actions"]["go"], p["actions"]["field_goal"], "go", "field goal",
                           "Time left in the fourth quarter", "Fourth and goal",
                           title=f"Leading by {lead}: go minus field goal", height=300,
                           reverse_x=True)  # fmt: skip
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    tp = q5["two_point"]
    h2 = tp["headline"]
    st.markdown(
        f"**Down {-h2['margin_before_try']} after a touchdown, {h2['situation']['seconds'] // 60}"
        f":00 left:** go for two {af.pct(h2['two'])}, kick {af.pct(h2['kick'])} "
        f"({af.points(h2['two_minus_kick'])} points; break-even two-point rate "
        f"{af.pct(h2['break_even_p_two_point'], 0)}, assumed rate "
        f"{af.pct(q5['assumptions']['p_two_point'], 0)})."
    )  # fmt: skip
    g2 = tp["grid"]
    z2 = np.array(g2["two_wp"]) - np.array(g2["kick_wp"])
    clock2 = [f"{s // 60}:{s % 60:02d}" for s in g2["seconds"]]
    fig2 = af.diverging_map(z2.T, clock2, g2["margins_before_try"], p["two_point"],
                            p["actions"]["field_goal"], "two", "kick",
                            "Time left in the fourth quarter",
                            "Margin after the touchdown, before the try",
                            title="Go for two minus kick the extra point", height=520,
                            reverse_x=True, limit=5.0)  # fmt: skip
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

    ge = q5["goal_line_early"]
    h3 = ge["headline"]["averaged_over_second_half_kickoff"]
    st.markdown(
        f"**Tied, fourth and goal at the 1, first quarter 10:00:** go {af.pct(h3['go'])}, field "
        f"goal {af.pct(h3['field_goal'])} ({af.points(h3['go_minus_field_goal'])} points)."
    )  # fmt: skip
    g3 = ge["grid"]
    yard = st.radio("Fourth and goal from the", g3["yardlines"], horizontal=True, key="q5y")
    yi = g3["yardlines"].index(yard)
    z3 = (np.array(g3["go_wp"]) - np.array(g3["field_goal_wp"]))[:, :, yi].T
    clock3 = [(f"Q1 {(s - 2700) // 60}:{(s - 2700) % 60:02d}" if s > 2700 else
               f"Q2 {(s - 1800) // 60}:{(s - 1800) % 60:02d}") for s in g3["seconds"]]  # fmt: skip
    fig3 = af.diverging_map(z3, clock3, [af.score_label(m) for m in g3["margins"]],
                            p["actions"]["go"], p["actions"]["field_goal"], "go", "field goal",
                            "Game clock (first half)", "Score",
                            title=f"First half, fourth and goal from the {yard}: go minus "
                                  "field goal",
                            height=330, reverse_x=True)  # fmt: skip
    st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})
    st.caption("Exact game with identical teams; one touchdown chance and one two-point rate "
               f"for every team. Source: {q5['source']}. Sensitivity runs (conversion and "
               "two-point chances plus or minus 5 points) are in that file; the two-point map is "
               "the fragile one.")  # fmt: skip


def footer(b: ab.Bundle) -> None:
    m = b.meta
    st.divider()
    one = m.get("one_step", {})
    seasons = m.get("seasons", {}).get("fit", "")
    stamp = (one.get("wp_model_stamp") or {})
    lines = [
        f"**Models.** One step: win-probability model `{one.get('wp_model', '?')}` "
        f"({stamp.get('what', 'n/a')}) and the play-outcome models `{one.get('components', '?')}`; "
        f"{seasons}. Confidence: {b.n_boot} game-level bootstrap refits (categories at 83% and "
        "67% agreement, the rule of Brill, Yurko and Wyner, arXiv 2311.03490). Exact: the game "
        "solved by backward induction on a 5-second grid (M4).",
        "**Limits.** The exact game has identical, league-average teams and assumes both play "
        "optimally afterwards; it ignores the spread. Conversion chances where coaches almost "
        "never go for it are extrapolated from third downs. Bootstrap intervals are somewhat too "
        "narrow for tree models. Overtime is not modelled (a tie is worth one half).",
        f"**Grid.** {len(b.clocks)} clocks, scores {b.scores[0]:+d} to {b.scores[-1]:+d}, every "
        "yard line, 1 to 15 yards to go, opponent timeouts in the last 5:00, spreads -7, 0, +7; "
        "nothing is interpolated. You always hold 3 timeouts and do not receive the second-half "
        "kickoff.",
        f"Bundle built {m.get('built_at', 'n/a')} from commit "
        f"{m.get('git_commit_at_build', 'n/a')}."
        + (" **Synthetic test bundle: these numbers are made up.**" if m.get("synthetic") else ""),
    ]  # fmt: skip
    for line in lines:
        st.caption(line)


def main() -> None:
    st.set_page_config(page_title="Fourth down explorer", layout="wide")
    b = get_bundle(bundle_path())
    cfg = app_config()
    st.title("Fourth down: go, kick or punt?")
    st.caption("Set a situation. The one-step engine (with bootstrap confidence) and the exactly "
               "solved game give their answers side by side.")  # fmt: skip
    t1, t2, t3 = st.tabs(["Fourth-down call", "Leading late (question 1)",
                          "Goal line and two-point"])  # fmt: skip
    with t1:
        tab_decision(b, cfg.get("default", {}))
    with t2:
        tab_late_lead(b, cfg.get("late_lead", {}))
    with t3:
        tab_goal_line(b)
    footer(b)


if __name__ == "__main__":
    main()
