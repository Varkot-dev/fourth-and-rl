"""M7: animation of the exact game's value function as the clock runs down (plan section 5).

Win probability of the team with the ball on first and 10 (goal to go inside the 10) at
every yard line, read from the exactly solved game (M4), one line per score in a one-score
game, from 10:00 to 0:10 in the fourth quarter (at 0:00 the game is over). Both teams hold 3 timeouts in the last
5:00 (before that the model has no timeouts). Identical, league-average teams.

Output: assets/m7_value_function.gif (settings in configs/app.yaml, `animation`).

Run:  python scripts/m7_value_animation.py        (needs data/mdp from scripts/solve_mdp.py)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from fourth_and_rl import app_bundle as ab
from fourth_and_rl import endgame_questions as eq
from fourth_and_rl import mdp_game as mg
from fourth_and_rl import style
from fourth_and_rl.paths import ROOT

SCORE_COLORS = {-7: style.SEQUENTIAL[1], -3: style.SEQUENTIAL[3], 0: style.MUTED,
                3: style.ORANGE, 7: style.RED}  # fmt: skip


def first_down_values(sol: mg.Solution, seconds: int, scores: list[int], timeouts: int
                      ) -> np.ndarray:  # fmt: skip
    """(len(scores), 99) win probability on first and 10 at yardline_100 1..99."""
    t = eq.tick_of(sol, seconds)
    v = sol.native(t)
    e1, e2 = eq.extra_index(sol, t, (timeouts, timeouts, 0))
    f = sol.grid.first10()[1:]  # yard lines 1..99
    si = eq.score_index(sol, scores)
    return np.asarray(v[f[None, :], si[:, None], e1, e2], dtype=float)


def clock_label(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def main() -> None:
    cfg = ab.load_config()["animation"]
    sol = mg.load_solution()
    start, end = cfg["seconds"]
    times = list(range(int(start), int(end) - 1, -int(cfg["step_seconds"])))
    scores = [int(s) for s in cfg["scores"]]
    frames = np.stack([first_down_values(sol, s, scores, int(cfg["timeouts"])) for s in times])
    own = 100 - np.arange(1, 100)  # x: yards from the offense's own goal line

    style.apply()
    plt.rcParams["savefig.bbox"] = None  # animation frames must keep one size
    plt.rcParams["savefig.dpi"] = 100
    fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=90)
    ax.axvspan(-10, 0, color=style.GRID, zorder=0)
    ax.axvspan(100, 110, color=style.GRID, zorder=0)
    ax.text(-5, 0.5, "OWN END ZONE", rotation=90, ha="center", va="center", color=style.INK_2,
            fontsize=9, fontweight="bold")  # fmt: skip
    ax.text(105, 0.5, "OPP END ZONE", rotation=270, ha="center", va="center", color=style.INK_2,
            fontsize=9, fontweight="bold")  # fmt: skip
    ax.axhline(0.5, color=style.AXIS, linewidth=1, zorder=1)
    lines = []
    for k, s in enumerate(scores):
        label = "Tied" if s == 0 else (f"Up {s}" if s > 0 else f"Down {-s}")
        (ln,) = ax.plot(own, frames[0, k], color=SCORE_COLORS.get(s, style.INK), label=label,
                        linewidth=2.4, zorder=3)  # fmt: skip
        lines.append(ln)
    ax.set_xlim(-10, 110)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(10, 100, 10))
    ax.set_xticklabels(["Own 10", "20", "30", "40", "50", "40", "30", "20", "Opp 10"])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(100 * y)}%" for y in np.linspace(0, 1, 6)])
    ax.set_xlabel("Ball on (first and 10; the offense moves left to right)")
    ax.set_ylabel("Win probability of the team with the ball")
    style.title(ax, "The value of field position as the clock runs down",
                "Exact solution of the game model (M4): first and 10, both teams with 3 "
                "timeouts in the last 5:00, identical teams")  # fmt: skip
    clock = ax.text(0.985, 0.04, "", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=22, fontweight="bold", color=style.INK)  # fmt: skip
    ax.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.16), fontsize=10,
              handlelength=1.8, columnspacing=1.6)  # fmt: skip
    fig.tight_layout()

    def update(i: int) -> list:
        for k, ln in enumerate(lines):
            ln.set_ydata(frames[i, k])
        clock.set_text(f"Q4 {clock_label(times[i])}")
        return [*lines, clock]

    n = len(times)
    hold = int(cfg["fps"]) * 2  # pause on the last frame
    anim = FuncAnimation(fig, update, frames=list(range(n)) + [n - 1] * hold, blit=False)
    out = ROOT / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=int(cfg["fps"])), savefig_kwargs={
        "facecolor": style.SURFACE})  # fmt: skip
    plt.close(fig)
    mb = out.stat().st_size / 1e6
    print(f"{out}: {n} frames, {mb:.2f} MB")
    if mb > float(cfg["max_megabytes"]):
        raise SystemExit(f"animation is {mb:.1f} MB, over {cfg['max_megabytes']} MB")


if __name__ == "__main__":
    main()
