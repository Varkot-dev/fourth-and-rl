"""Tiny hand-made models with a known optimum, so the M5 code can be tested without data.

`chain_drive` is small enough to solve on paper. `toy_game` is a random football-like
game (three downs, a six-yard field, gains and losses, turnovers, safeties, field goals,
punts, kneels, timeouts, a warning) solved exactly by `mdp_game.solve` in a fraction of a
second; both environments can be built on it exactly as on the real model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from fourth_and_rl import mdp_game as mg
from fourth_and_rl.mdp import KERNEL_KEYS, SITUATIONS, FieldKernels, Grid
from fourth_and_rl.rl.envs import DriveModel, build_drive_model


def _csr(entries: dict[tuple[int, int], float], shape: tuple[int, int]) -> sp.csr_matrix:
    rows = [k[0] for k in entries]
    cols = [k[1] for k in entries]
    return sp.coo_matrix((list(entries.values()), (rows, cols)), shape=shape).tocsr()


def toy_kernels(seed: int = 0, n_yl: int = 6) -> FieldKernels:
    """Random play outcomes on a short field: gains of -1, 0, 1, 2 yards, a long gain, and
    turnovers; field goals that get harder with distance; punts; kneels."""
    rng = np.random.default_rng(seed)
    grid = Grid(n_down=3, n_yl=n_yl, dist_exact=2, dist_bins=((3, 9, 3),))
    nf, na = grid.nf, grid.na
    f10 = grid.first10()
    stop = {"neutral": 0.4, "neutral_end": 0.5, "protect": 0.2, "chase": 0.7, "chase_late": 0.85}
    all_e: dict[int, dict[tuple[int, int], float]] = {}
    keep_e: dict[int, dict[tuple[int, int], float]] = {}
    for f in range(nf):
        down, dbin, yl = (int(x) for x in grid.unpack(f))
        d = min(int(grid.dist_rep(dbin)), yl)
        p = rng.dirichlet(np.ones(6) * 2.0)
        outs: list[tuple[float, int, bool]] = [
            (p[4] * 0.7, int(grid.col_opp(n_yl + 1 - yl)), False),
            (p[4] * 0.3, grid.col_opp_td, False),
        ]
        for prob, gain in zip(p[:4], (-1, 0, 1, 2), strict=True):
            new = yl - gain
            if new <= 0:
                outs.append((prob, grid.col_td, False))
            elif new > n_yl:
                outs.append((prob, grid.col_safety, False))
            elif gain >= d:
                outs.append((prob, int(f10[new]), True))
            elif down == grid.n_down:
                outs.append((prob, int(grid.col_opp(n_yl + 1 - new)), False))
            else:
                outs.append((prob, int(grid.fidx(down + 1, grid.dist_bin(d - gain), new)), True))
        if yl <= 3:
            outs.append((p[5], grid.col_td, False))
        else:
            outs.append((p[5], int(f10[yl - 3]), True))
        all_e[f] = {}
        keep_e[f] = {}
        for prob, col, kept in outs:
            all_e[f][(f, col)] = all_e[f].get((f, col), 0.0) + float(prob)
            if kept:
                keep_e[f][(f, col)] = keep_e[f].get((f, col), 0.0) + float(prob)
    flat_all = {k: v for e in all_e.values() for k, v in e.items()}
    flat_keep = {k: v for e in keep_e.values() for k, v in e.items()}
    k_all = {m: _csr(flat_all, (nf, na)) for m in KERNEL_KEYS}
    k_in = {
        m: _csr({k: v * (1.0 - stop[m]) for k, v in flat_keep.items()}, (nf, nf))
        for m in KERNEL_KEYS
    }
    downs, dbins, yls = grid.unpack(np.arange(nf))
    last = downs == grid.n_down
    punt: dict[tuple[int, int], float] = {}
    for f in np.nonzero(last)[0]:
        yl = int(yls[f])
        land = min(n_yl, n_yl + 1 - yl + 3)
        for prob, col in (
            (0.70, int(grid.col_opp(land))),
            (0.25, int(grid.col_opp(max(land - 2, 1)))),
            (0.03, grid.col_opp_td),
            (0.02, int(f10[yl])),
        ):
            punt[(int(f), col)] = punt.get((int(f), col), 0.0) + prob
    back = np.minimum(yls + 1, n_yl)
    d_rep = np.minimum(grid.dist_rep(dbins), yls)
    kick = np.zeros(n_yl + 1)
    kick[n_yl - 2 : n_yl + 1] = (0.3, 0.5, 0.2)
    return FieldKernels(
        grid=grid,
        k_all=k_all,
        k_in=k_in,
        fg_make=np.clip(0.95 - 0.22 * (yls - 1), 0.0, 1.0),
        fg_miss_col=np.asarray(grid.col_opp(n_yl + 1 - yls)),
        fg_ok=(0.95 - 0.22 * (yls - 1)) > 0,
        k_punt=_csr(punt, (nf, na)),
        punt_ok=last,
        kneel_col=np.where(
            downs < grid.n_down,
            grid.fidx(np.minimum(downs + 1, grid.n_down), grid.dist_bin(d_rep + 1), back),
            grid.col_opp(n_yl + 1 - back),
        ),
        kneel_keep=downs < grid.n_down,
        kneel_ok=np.ones(nf, dtype=bool),
        kickoff=kick,
        free_kick=kick.copy(),
    )


def toy_drive(seed: int = 0) -> DriveModel:
    """The single-drive problem on the toy field (touchdown 7, field goal 3, safety 2)."""
    return build_drive_model(toy_kernels(seed), td_value=7.0, fg_points=3.0, safety_points=2.0)


def toy_game(seed: int = 0, n_ticks: int = 12, n_timeouts: int = 1) -> mg.Solution:
    """A solved end game on the toy field: `n_ticks` ticks, all of them in the end phase."""
    kern = toy_kernels(seed)
    n = n_timeouts + 1
    clock = mg.ClockSpec(
        n_ticks=n_ticks,
        tick_seconds=5,
        endgame_ticks=n_ticks,
        half_tick=None,
        warning_tick=4,
        burn_ticks=3,
        punt_ticks=2,
        n_timeouts=n_timeouts,
        neutral_hist=np.array([0.2, 0.5, 0.3]),
        hurry_hist=np.array([0.5, 0.5]),
        hurry_hist_late=np.array([0.8, 0.2]),
        kneel_ticks=mg.kneel_threshold_ticks(n_timeouts, kern.grid.n_down, 5, 2.0, 10.0, 20.0),
        shares_leading=np.full(n, 1.0 / n),
        shares_trailing=np.full(n, 1.0 / n),
        defense_timeout="optimal",
        def_rate_before=np.linspace(0.0, 0.6, n),
        def_rate_after=np.linspace(0.0, 0.9, n),
    )
    model = mg.GameModel(
        kern=kern,
        clock=clock,
        score_max=4,
        td_points=2,
        fg_points=1,
        safety_points=1,
        p_xp=0.8,
        p_two=0.45,
        two_point_choice=True,
        dtype="float64",
    )
    return mg.solve(model)


def toy_coach_table(grid: Grid, seed: int = 0) -> dict[str, Any]:
    """A made-up "coach" table in the shape the simulator expects."""
    rng = np.random.default_rng(seed)
    table = rng.dirichlet(np.array([1.0, 1.0, 2.0]), size=(len(SITUATIONS), grid.n_dist, grid.n_yl))
    return {"table": table, "end_fg_max_yardline": 3, "end_fg_ticks": 2}


def chain_drive(p_go: float = 0.5, fg_make: float = 0.9) -> DriveModel:
    """A one-down, two-yard-line game small enough to solve by hand.

    From yard line 1: "go" scores a touchdown (7) with probability `p_go`, otherwise the
    other team takes over at their yard line 2; a field goal (3) is good with probability
    `fg_make`, otherwise the same turnover. From yard line 2: "go" moves to yard line 1
    with probability 0.5, otherwise the other team takes over at their 1; no field goal;
    a punt always gives the other team the ball at their 2. Kickoffs go to yard line 2.
    """
    grid = Grid(n_down=1, n_yl=2, dist_exact=1, dist_bins=())
    nf, na = grid.nf, grid.na
    play = {
        (0, grid.col_td): p_go,
        (0, int(grid.col_opp(2))): 1.0 - p_go,
        (1, 0): 0.5,
        (1, int(grid.col_opp(1))): 0.5,
    }
    k_all = {m: _csr(play, (nf, na)) for m in KERNEL_KEYS}
    k_in = {m: sp.csr_matrix((nf, nf)) for m in KERNEL_KEYS}
    kick = np.array([0.0, 0.0, 1.0])
    kern = FieldKernels(
        grid=grid,
        k_all=k_all,
        k_in=k_in,
        fg_make=np.array([fg_make, 0.0]),
        fg_miss_col=np.asarray(grid.col_opp(np.array([2, 1]))),
        fg_ok=np.array([True, False]),
        k_punt=_csr({(0, int(grid.col_opp(2))): 1.0, (1, int(grid.col_opp(2))): 1.0}, (nf, na)),
        punt_ok=np.ones(nf, dtype=bool),
        kneel_col=np.zeros(nf, dtype=np.int64),
        kneel_keep=np.zeros(nf, dtype=bool),
        kneel_ok=np.zeros(nf, dtype=bool),
        kickoff=kick,
        free_kick=kick.copy(),
    )
    return build_drive_model(kern, td_value=7.0, start_yardline=np.array([0.0, 0.5, 0.5]))
