"""One plotting style for every figure in the repo.

Colors are fixed by role so the same thing has the same color in every figure:
the three fourth-down actions, our model versus the nflfastR baseline, and so on.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#6250d6"
RED = "#e34948"
NEUTRAL = "#c3c2b7"

# Fixed roles.
ACTION_COLORS = {"go": BLUE, "field_goal": ORANGE, "punt": AQUA, "toss_up": NEUTRAL}
ACTION_LABELS = {
    "go": "Go for it",
    "field_goal": "Field goal",
    "punt": "Punt",
    "toss_up": "Too close to call",
}
MODEL_COLORS = {"ours": BLUE, "nflfastr": ORANGE, "theory": INK, "vegas": VIOLET}

# One-hue ramp for magnitudes, light to dark.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply() -> None:
    """Set the matplotlib defaults used by every figure script."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10.5,
            "text.color": INK,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def title(ax: plt.Axes, text: str, subtitle: str | None = None) -> None:
    """Left-aligned bold title with an optional secondary line under it."""
    ax.set_title(text, pad=26 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=10,
                va="bottom")


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
