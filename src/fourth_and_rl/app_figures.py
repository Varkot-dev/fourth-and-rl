"""M7: Plotly figures for the Streamlit app, in the project style (`style.py`).

Plotly is imported inside the figure functions, so `palette` and the label helpers work
without it (the static page builder uses them to colour its own JavaScript charts).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fourth_and_rl import style
from fourth_and_rl.app_bundle import ACTIONS

LEAN_BLEND = 0.58  # as decision.LEAN_BLEND: how far a "lean" cell is washed toward the surface
CATEGORIES = ("confident", "lean", "uncertain")
FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: tuple[float, ...]) -> str:
    return "#" + "".join(f"{int(round(255 * c)):02x}" for c in rgb)


def blend(color: str, toward: str, share: float) -> str:
    a, b = np.array(_hex_to_rgb(color)), np.array(_hex_to_rgb(toward))
    return _rgb_to_hex(tuple((1 - share) * a + share * b))


def palette() -> dict[str, Any]:
    """Every colour the interactive pieces use, from style.py.

    `cells[code]` colours a decision-chart cell with code = 3 * action + category
    (action 0 go, 1 field goal, 2 punt; category 0 confident, 1 lean, 2 uncertain), the
    same scheme as decision.chart_colors."""
    cells = []
    for a in ACTIONS:
        base = style.ACTION_COLORS[a]
        cells += [base, blend(base, style.SURFACE, LEAN_BLEND), style.ACTION_COLORS["toss_up"]]
    return {
        "cells": cells,
        "actions": {a: style.ACTION_COLORS[a] for a in ACTIONS},
        "toss_up": style.ACTION_COLORS["toss_up"],
        "surface": style.SURFACE, "ink": style.INK, "ink_2": style.INK_2,
        "muted": style.MUTED, "grid": style.GRID, "axis": style.AXIS,
        "exact": style.INK, "two_point": style.VIOLET,
    }  # fmt: skip


def field_label(yardline_100: int) -> str:
    """'Own 25', 'Midfield', 'Opp 30' from the offense's side."""
    y = int(yardline_100)
    if y == 50:
        return "Midfield"
    return f"Own {100 - y}" if y > 50 else f"Opp {y}"


def score_label(score: int) -> str:
    s = int(score)
    return "Tied" if s == 0 else f"Up {s}" if s > 0 else f"Down {-s}"


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def points(x: float, digits: int = 1) -> str:
    """A win-probability difference in percentage points, signed."""
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.{digits}f}"


def chart_codes(chart: dict[str, np.ndarray], exact: bool = False) -> np.ndarray:
    """Cell codes (see `palette`) for a chart from `Bundle.chart`; -1 for empty cells.
    The exact game has no bootstrap, so its cells are all "confident" colours."""
    best = chart["exact_best"] if exact else chart["best"]
    cat = np.zeros_like(best) if exact else np.maximum(chart["category"], 0)
    return np.where(best < 0, -1, 3 * best + cat)


def _layout(fig: Any, height: int, title: str | None = None) -> None:
    p = palette()
    fig.update_layout(
        height=height, margin={"l": 10, "r": 10, "t": 48 if title else 16, "b": 10},
        paper_bgcolor=p["surface"], plot_bgcolor=p["surface"],
        font={"family": FONT, "color": p["ink"], "size": 13},
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 16}} if title
        else None,
        hoverlabel={"font": {"family": FONT}},
    )  # fmt: skip


def field_chart(
    chart: dict[str, np.ndarray],
    yardlines: list[int],
    distances: list[int],
    current: tuple[int, int] | None = None,
    exact: bool = False,
    title: str | None = None,
    height: int = 430,
) -> Any:
    """The decision chart drawn on a football field: x = yards from the offense's own goal
    line (end zones at both ends), y = yards to go; the current situation is ringed."""
    import plotly.graph_objects as go

    p = palette()
    codes = chart_codes(chart, exact)
    n_codes = len(p["cells"])
    # Plotly heatmap with a stepped colour scale: code k is drawn in cells[k].
    scale = []
    for k, c in enumerate(p["cells"]):
        scale += [[k / n_codes, c], [(k + 1) / n_codes, c]]
    x = [100 - y for y in yardlines][::-1]
    z = np.where(codes < 0, np.nan, codes).astype(float)[:, ::-1]
    text = _hover(chart, yardlines, distances, exact)[:, ::-1]
    fig = go.Figure()
    for x0, x1 in ((-10, 0), (100, 110)):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=0.5, y1=len(distances) + 0.5,
                      fillcolor=p["grid"], line={"width": 0}, layer="below")  # fmt: skip
    fig.add_trace(go.Heatmap(
        z=z, x=x, y=distances, zmin=-0.5, zmax=n_codes - 0.5, colorscale=scale,
        showscale=False, text=text, hovertemplate="%{text}<extra></extra>", xgap=0.4, ygap=0.8,
    ))  # fmt: skip
    for xv in range(10, 100, 10):
        fig.add_vline(x=xv, line={"color": p["surface"], "width": 1.2})
    for xv in (0, 100):
        fig.add_vline(x=xv, line={"color": p["ink_2"], "width": 2})
    fig.add_annotation(x=-5, y=(len(distances) + 1) / 2, text="OWN END ZONE", textangle=-90,
                       showarrow=False, font={"color": p["ink_2"], "size": 11})  # fmt: skip
    fig.add_annotation(x=105, y=(len(distances) + 1) / 2, text="OPP END ZONE", textangle=90,
                       showarrow=False, font={"color": p["ink_2"], "size": 11})  # fmt: skip
    if current is not None:
        yl, d = current
        fig.add_trace(go.Scatter(
            x=[100 - yl], y=[d], mode="markers", hoverinfo="skip", showlegend=False,
            marker={"size": 15, "color": "rgba(0,0,0,0)",
                    "line": {"color": p["ink"], "width": 2.5}},
        ))  # fmt: skip
        fig.add_vline(x=100 - yl, line={"color": p["ink"], "width": 1, "dash": "dot"})
    # Legend entries as invisible markers.
    names = [("go", "Go for it"), ("field_goal", "Field goal"), ("punt", "Punt")]
    for i, (_a, label) in enumerate(names):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                 marker={"symbol": "square", "size": 13,
                                         "color": p["cells"][3 * i]}))  # fmt: skip
    if not exact:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Lean (paler)",
                                 marker={"symbol": "square", "size": 13,
                                         "color": p["cells"][1]}))  # fmt: skip
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Too close to call",
                                 marker={"symbol": "square", "size": 13, "color": p["toss_up"]}))
    _layout(fig, height, title)
    fig.update_xaxes(range=[-10, 110], tickvals=list(range(10, 100, 10)),
                     ticktext=["10", "20", "30", "40", "50", "40", "30", "20", "10"],
                     showgrid=False, zeroline=False, fixedrange=True,
                     title={"text": "Field position (your own goal line on the left)",
                            "font": {"color": p["ink_2"]}})  # fmt: skip
    fig.update_yaxes(range=[0.5, len(distances) + 0.5], showgrid=False, zeroline=False,
                     fixedrange=True, dtick=2, title={"text": "Yards to go",
                                                      "font": {"color": p["ink_2"]}})  # fmt: skip
    fig.update_layout(legend={"orientation": "h", "y": -0.22, "x": 0, "bgcolor": p["surface"]})
    return fig


def _hover(chart: dict[str, np.ndarray], yardlines: list[int], distances: list[int],
           exact: bool) -> np.ndarray:  # fmt: skip
    vals = chart["exact_values"] if exact else chart["values"]
    out = np.full(vals.shape[:2], "", dtype=object)
    for i, d in enumerate(distances):
        for j, y in enumerate(yardlines):
            if d > y:
                continue
            v = vals[i, j]
            line = (f"<b>{field_label(y)}, 4th and {d}</b><br>Go {pct(v[0])}  ·  Field goal "
                    f"{pct(v[1])}  ·  Punt {pct(v[2])}")  # fmt: skip
            if exact:
                line += f"<br>Go minus best kick {points(chart['exact_go_gain'][i, j])} points"
            else:
                line += (f"<br>Go minus best kick {points(chart['go_gain'][i, j])} points "
                         f"[{points(chart['gain_lo'][i, j])}, {points(chart['gain_hi'][i, j])}]"
                         f"<br>Refits agreeing with the call: {pct(chart['boot_pct'][i, j], 0)}")
            out[i, j] = line
    return out


def action_bars(cell: dict[str, Any], height: int = 260) -> Any:
    """Win probability of each action: one-step bars with their 90% bootstrap interval,
    and the exact game's value as a diamond."""
    import plotly.graph_objects as go

    p = palette()
    labels = {"go": "Go for it", "field_goal": "Field goal", "punt": "Punt"}
    one, ex = cell["one_step"], cell["exact"]
    acts = [a for a in ACTIONS if np.isfinite(one["wp"][a]) or np.isfinite(ex["wp"][a])]
    y = [labels[a] for a in acts]
    w = [one["wp"][a] for a in acts]
    lo = [one["wp_lo"][a] for a in acts]
    hi = [one["wp_hi"][a] for a in acts]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y, x=[100 * v if np.isfinite(v) else None for v in w], orientation="h",
        marker={"color": [p["actions"][a] for a in acts]},
        name="One step ahead (bars, 90% interval)",
        error_x={"type": "data", "symmetric": False, "color": p["ink_2"], "thickness": 1.5,
                 "array": [100 * (h - v) if np.isfinite(v) else None
                           for v, h in zip(w, hi, strict=True)],
                 "arrayminus": [100 * (v - lo_) if np.isfinite(v) else None
                                for v, lo_ in zip(w, lo, strict=True)]},
        hovertemplate="%{y}: %{x:.1f}%<extra>one step</extra>",
    ))  # fmt: skip
    fig.add_trace(go.Scatter(
        y=y, x=[100 * ex["wp"][a] if np.isfinite(ex["wp"][a]) else None for a in acts],
        mode="markers", name="Exact full game", hovertemplate="%{y}: %{x:.1f}%<extra>exact</extra>",
        marker={"symbol": "diamond", "size": 13, "color": p["exact"],
                "line": {"color": p["surface"], "width": 1.5}},
    ))  # fmt: skip
    finite = [v for v in w + [ex["wp"][a] for a in acts] + lo + hi if np.isfinite(v)]
    lo_x = max(0.0, 100 * min(finite) - 8) if finite else 0
    hi_x = min(100.0, 100 * max(finite) + 4) if finite else 100
    _layout(fig, height)
    fig.update_xaxes(range=[lo_x, hi_x], ticksuffix="%", gridcolor=p["grid"], zeroline=False,
                     title={"text": "Win probability", "font": {"color": p["ink_2"]}})
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(legend={"orientation": "h", "y": -0.35, "x": 0}, bargap=0.35)
    return fig


def diverging_map(
    z: np.ndarray, x: list, y: list, pos_color: str, neg_color: str, pos_label: str,
    neg_label: str, x_title: str, y_title: str, hover: np.ndarray | None = None,
    title: str | None = None, height: int = 380, x_ticktext: list[str] | None = None,
    limit: float | None = None, reverse_x: bool = False,
) -> Any:
    """Heatmap of a difference in win probability (positive = pos_label better), centred
    on zero with a symmetric range, in percentage points."""
    import plotly.graph_objects as go

    p = palette()
    zz = 100 * np.asarray(z, dtype=float)
    if limit is None:
        limit = float(np.nanmax(np.abs(zz))) if np.isfinite(zz).any() else 1.0
    lim = max(limit, 0.5)
    scale = [[0.0, neg_color], [0.5, p["surface"]], [1.0, pos_color]]
    fig = go.Figure(go.Heatmap(
        z=zz, x=x, y=y, zmid=0, zmin=-lim, zmax=lim, colorscale=scale, xgap=0.5, ygap=0.5,
        text=hover, hovertemplate=("%{text}<extra></extra>" if hover is not None else
                                   "%{x}, %{y}: %{z:+.1f} points<extra></extra>"),
        colorbar={"title": {"text": "points", "side": "right"}, "thickness": 12,
                  "tickvals": [-lim, 0, lim],
                  "ticktext": [f"{neg_label} +{lim:.0f}", "even", f"{pos_label} +{lim:.0f}"]},
    ))  # fmt: skip
    _layout(fig, height, title)
    fig.update_xaxes(title={"text": x_title, "font": {"color": p["ink_2"]}}, showgrid=False,
                     autorange="reversed" if reverse_x else True)
    if x_ticktext is not None:
        fig.update_xaxes(tickvals=x, ticktext=x_ticktext)
    fig.update_yaxes(title={"text": y_title, "font": {"color": p["ink_2"]}}, showgrid=False)
    return fig
