"""The exact game with two different teams: "what should THIS team do against THAT one?"

`mdp_game.py` solves a game between two identical league-average teams. Its one value
array holds both teams because of a symmetry: the other team's value in a state is
1 minus ours after re-describing the state from its side, `opp(X)[s, e] = 1 - X[-s,
swapped e]`. That is only true when the teams are the same. Here team A's offense faces
team B's defense and B's offense faces A's defense, so the game needs two arrays:

    VA(t, f, s, e)   win probability of A when A has the ball
    VB(t, f, s, e)   win probability of B when B has the ball

and every change of possession reads the other team's array:

    VA(t, f, s, e) = max over a of sum over o of PA(o | f, a) * WA(o, t - duration)
    WA(keep the ball in f')            = VA(t', f', s, e)
    WA(B takes over at its y)          = 1 - VB(t', first and 10 at y, -s, e swapped)
    WA(touchdown, field goal, safety)  = 1 - KB(...) with B's kickoff or free-kick value
    WA(B returns it for a touchdown)   = 1 - TDB(t', -s, e swapped)

and the same with A and B exchanged. PA is A's offense against B's defense. The clock,
timeouts, modes, kneel rule and two-point choice are the existing solver's code: each
team is an `mdp_game.Solution` whose `other` points at the other team's `Solution`
(`link`), and `mdp_game.backward_induction` solves both, slice by slice.

Team strength enters only through the play kernels (`matchup_kernels`): the generic play
of downs 1 to 3 is bent with the same two-number family the protect and chase modes use
(yards reweighted by exp(tilt * yards), turnovers multiplied by a scale), and fourth-down
conversion log-odds move by a fixed amount. The tilt is calibrated so that expected points
per drive from the own 25 move by what M8's drive model implies for the pair of ratings;
the conversion shift is M8's conversion model's. Field goals, punts, kickoffs and the try
after a touchdown stay league average (M8 built no kicking ratings).
"""

from __future__ import annotations

import copy
import resource
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import scipy.optimize as so
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from fourth_and_rl import mdp_game as mg
from fourth_and_rl.mdp import KERNEL_KEYS, FieldKernels
from fourth_and_rl.team_strength import unit_features

TEAMS = ("A", "B")
# Points of one drive, as M8 counts them (configs/team_strength.yaml, samples.drive_points).
DRIVE_POINTS = {"touchdown": 7.0, "field_goal": 3.0, "opp_touchdown": -7.0, "safety": -2.0}
CALIBRATION_START = 75  # first and 10 at the own 25: the typical drive start
TILT_BRACKET = (-0.3, 0.3)  # the bracket `mdp.fit_mode_adjustments` searches


# --------------------------------------------------------------------------------------
# Kernels for one matchup
# --------------------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def tilt_kernels(
    kern: FieldKernels,
    tilt: float = 0.0,
    turnover_scale: float = 1.0,
    conv_dlogit: float = 0.0,
    keys: tuple[str, ...] = KERNEL_KEYS,
) -> FieldKernels:
    """Bend every play table in `keys` for one offense against one defense.

    Downs before the last: exactly `mdp.adjusted_play` applied to each row of the table
    (yards gained, and a touchdown as a gain of the yard line, reweighted by
    exp(tilt * yards); the turnover probability, including returns, multiplied by
    `turnover_scale` and capped at 0.5; the safety left alone; the kept outcomes rescaled
    so the row still sums to 1). The family composes: tilting a protect table, which is
    the generic play tilted by its own a, gives the generic play tilted by a + tilt.
    Last down (going for it): log-odds of converting move by `conv_dlogit`; converted
    outcomes are scaled by p'/p and failed ones by (1 - p')/(1 - p), as
    `build_field_kernels` does for its additive sensitivity shift.

    The clock-running part `k_in` is scaled entry by entry with its `k_all` entry (the
    chance a play stops the clock depends on the yards gained only). Field goals, punts,
    kneels and kicks are unchanged. Pure array arithmetic: no data, no component models.
    """
    if tilt == 0.0 and turnover_scale == 1.0 and conv_dlogit == 0.0:
        return kern
    g = kern.grid
    nf = g.nf
    down, _, yl = g.unpack(np.arange(nf))
    last = down == g.n_down
    k_all, k_in = dict(kern.k_all), dict(kern.k_in)
    p_new_all = None
    for key in keys:
        m = kern.k_all[key].tocsr()
        rows = np.repeat(np.arange(nf), np.diff(m.indptr))
        cols, vals = m.indices, m.data.astype(float)
        keep = cols < nf
        td = cols == g.col_td
        saf = cols == g.col_safety
        turn = ~(keep | td | saf)
        made = keep | td
        gain = np.where(keep, yl[rows] - yl[np.minimum(cols, nf - 1)], np.where(td, yl[rows], 0))
        # Downs before the last: `adjusted_play` for every row at once.
        w = np.where(made, np.exp(tilt * gain), 1.0)
        p_saf = np.bincount(rows, vals * saf, minlength=nf)
        p_to = np.bincount(rows, vals * turn, minlength=nf)
        new_to = np.minimum(p_to * turnover_scale, 0.5)
        kept_w = np.bincount(rows, vals * w * made, minlength=nf)
        k_scale = np.divide(1.0 - p_saf - new_to, kept_w, out=np.ones(nf), where=kept_w > 0)
        to_scale = np.divide(new_to, p_to, out=np.ones(nf), where=p_to > 0)
        # Last down: shift the log-odds of converting.
        p = np.bincount(rows, vals * made, minlength=nf)
        ok = (p > 0) & (p < 1)
        p_new = p.copy()
        p_new[ok] = _sigmoid(np.log(p[ok] / (1.0 - p[ok])) + conv_dlogit)
        up = np.divide(p_new, p, out=np.ones(nf), where=ok)
        dn = np.divide(1.0 - p_new, 1.0 - p, out=np.ones(nf), where=ok)
        fac = np.where(
            last[rows],
            np.where(made, up[rows], dn[rows]),
            np.where(made, w * k_scale[rows], np.where(turn, to_scale[rows], 1.0)),
        )
        k_all[key] = sp.csr_matrix((vals * fac, cols, m.indptr), shape=m.shape)
        mi = kern.k_in[key].tocsr()
        r_in = np.repeat(np.arange(nf), np.diff(mi.indptr))
        c_in = mi.indices
        f_in = np.where(last[r_in], up[r_in], np.exp(tilt * (yl[r_in] - yl[c_in])) * k_scale[r_in])
        k_in[key] = sp.csr_matrix((mi.data.astype(float) * f_in, c_in, mi.indptr), shape=mi.shape)
        if p_new_all is None:
            p_new_all = p_new
    p_convert = kern.p_convert
    if p_new_all is not None and p_convert.size == nf:
        p_convert = np.where(last, p_new_all, p_convert)
    return replace(kern, k_all=k_all, k_in=k_in, p_convert=p_convert)


def fourth_down_rule(kern: FieldKernels, fg_min_make: float = 0.6, go_max_yards: int = 1,
                     go_from_yardline: int = 60) -> np.ndarray:  # fmt: skip
    """A fixed, conventional fourth-down rule for measuring drive points: kick a field goal
    where it is good at least `fg_min_make` of the time, otherwise go on fourth and
    `go_max_yards` or less from `go_from_yardline` (the own 40) on, otherwise punt.
    Returns codes per field state: 0 go, 1 field goal, 2 punt (only last-down rows matter).
    Fixed so that drive points measure the offense, not a re-optimized policy."""
    g = kern.grid
    _, dbin, yl = g.unpack(np.arange(g.nf))
    dist = np.minimum(g.dist_rep(dbin), yl)
    fg = kern.fg_ok & (kern.fg_make >= fg_min_make)
    go = (dist <= go_max_yards) & (yl <= go_from_yardline)
    out = np.where(fg, 1, np.where(go | ~kern.punt_ok, 0, 2))
    return out.astype(np.int64)


def drive_points(
    kern: FieldKernels,
    key: str = "neutral",
    rule: np.ndarray | None = None,
    points: dict[str, float] | None = None,
) -> np.ndarray:
    """Expected points of one drive (M8's count: touchdown 7, field goal 3, returned for a
    touchdown -7, safety -2) from every field state, with plays from table `key` and the
    fixed fourth-down `rule`. Computed exactly: the drive is an absorbing Markov chain over
    field states, so D = r + P D, a sparse linear system. This equals the average over
    infinitely many simulated drives."""
    g = kern.grid
    pts = DRIVE_POINTS if points is None else points
    rule = fourth_down_rule(kern) if rule is None else rule
    down = g.unpack(np.arange(g.nf))[0]
    last = down == g.n_down
    play = ~last | (rule == 0)
    punt = last & (rule == 2)
    fg = last & (rule == 1)
    k = kern.k_all[key].tocsr()
    sel_play = sp.diags(play.astype(float))
    sel_punt = sp.diags(punt.astype(float))
    trans = (sel_play @ k + sel_punt @ kern.k_punt.tocsr()).tocsr()
    col_pts = np.zeros(g.na)
    col_pts[g.col_td] = pts["touchdown"]
    col_pts[g.col_opp_td] = pts["opp_touchdown"]
    col_pts[g.col_safety] = pts["safety"]
    r = trans @ col_pts + fg * kern.fg_make * pts["field_goal"]
    p_keep = trans[:, : g.nf]
    return spl.spsolve((sp.identity(g.nf, format="csc") - p_keep).tocsc(), r)


@dataclass(frozen=True)
class RatingEffects:
    """How M8 ratings move conversion and drive points, relative to an average matchup.

    `conversion` and `drive` are M8's fitted models (team_strength.ConversionModel and
    DriveModel). The exact game has no pregame spread, so the effect wanted is the whole
    effect of the ratings, not the part beyond the spread: the bundle's ratings-without-
    the-spread models ("conv_no_spread", "drive_no_spread"), fitted on the development
    seasons (2012 to 2023), are exactly that. With spread models the spread is held at 0.
    """

    conversion: Any
    drive: Any

    @classmethod
    def from_bundle(cls, bundle: dict[str, Any]) -> RatingEffects:
        """From the dict saved in data/models/team_strength.joblib (already loaded)."""
        return cls(bundle["models"]["conv_no_spread"], bundle["models"]["drive_no_spread"])

    @staticmethod
    def _feats(offense: dict[str, float], defense: dict[str, float]) -> dict[str, np.ndarray]:
        return {k: np.array([v]) for k, v in unit_features(offense, defense).items()}

    def conversion_dlogit(self, offense: dict[str, float], defense: dict[str, float]) -> float:
        """Change in short-yardage conversion log-odds against average units."""
        zero = np.zeros(1)
        lp = self.conversion.linear_predictor(zero, self._feats(offense, defense))
        lp0 = self.conversion.linear_predictor(zero, self._feats({}, {}))
        return float(lp[0] - lp0[0])

    def drive_points(self, offense: dict[str, float], defense: dict[str, float],
                     start: int = CALIBRATION_START) -> float:  # fmt: skip
        """M8's expected drive points (field goal 3, touchdown 7) from yard line `start`."""
        p = self.drive.proba(np.array([float(start)]), np.zeros(1), self._feats(offense, defense))
        return float(DRIVE_POINTS["field_goal"] * p[0, 1] + DRIVE_POINTS["touchdown"] * p[0, 2])

    def drive_points_shift(self, offense: dict[str, float], defense: dict[str, float],
                           start: int = CALIBRATION_START) -> float:  # fmt: skip
        """Change in expected drive points from `start` against average units."""
        return self.drive_points(offense, defense, start) - self.drive_points({}, {}, start)


@dataclass(frozen=True)
class Calibration:
    """The three numbers of one matchup's kernels and the evidence for the tilt."""

    tilt: float
    turnover_scale: float
    conv_dlogit: float
    start_yardline: int
    target_drive_points_shift: float  # what M8's drive model implies
    model_drive_points_base: float  # exact drive points of the generic kernels
    model_drive_points_matchup: float  # with the calibrated kernels

    @property
    def achieved_shift(self) -> float:
        return self.model_drive_points_matchup - self.model_drive_points_base

    def as_dict(self) -> dict[str, float]:
        return {**asdict(self), "achieved_drive_points_shift": self.achieved_shift}


def calibrate(
    base: FieldKernels,
    offense: dict[str, float],
    defense: dict[str, float],
    effects: RatingEffects,
    start_yardline: int = CALIBRATION_START,
    turnover_scale: float = 1.0,
) -> Calibration:
    """Find the tilt for one offense against one defense.

    Conversion: log-odds move by M8's conversion shift. Drive points: with that shift and
    `turnover_scale` in place, the tilt is the root of
        drive_points(tilted kernels)[first and 10 at start] - drive_points(base)[same]
        = M8's drive-points shift for the pair,
    found with Brent's method on the bracket `mdp.fit_mode_adjustments` uses. The
    turnover scale is held at 1 by default: M8 has no turnover rating, so the drive
    rating is carried by yards alone (a sensitivity can set it).
    """
    g = base.grid
    f0 = int(g.first10()[start_yardline])
    rule = fourth_down_rule(base)
    d0 = float(drive_points(base, rule=rule)[f0])
    target = effects.drive_points_shift(offense, defense, start_yardline)
    dlogit = effects.conversion_dlogit(offense, defense)

    def shift(a: float) -> float:
        k = tilt_kernels(base, a, turnover_scale, dlogit, keys=("neutral",))
        return float(drive_points(k, rule=rule)[f0]) - d0

    if target == 0.0 and dlogit == 0.0 and turnover_scale == 1.0:
        tilt = 0.0
    else:
        tilt = float(so.brentq(lambda a: shift(a) - target, *TILT_BRACKET, xtol=1e-10))
    return Calibration(
        tilt=tilt,
        turnover_scale=float(turnover_scale),
        conv_dlogit=dlogit,
        start_yardline=int(start_yardline),
        target_drive_points_shift=target,
        model_drive_points_base=d0,
        model_drive_points_matchup=d0 + shift(tilt),
    )


def _kernels_of(base_model: mg.GameModel | FieldKernels) -> FieldKernels:
    return base_model.kern if isinstance(base_model, mg.GameModel) else base_model


def matchup_kernels(
    base_model: mg.GameModel | FieldKernels,
    off_rating: dict[str, float],
    def_rating: dict[str, float],
    effects: RatingEffects,
    **calibration: Any,
) -> FieldKernels:
    """Play kernels of `off_rating` (M8 units: conv_off, drive_off) against `def_rating`
    (conv_def, drive_def), built from the generic kernels of `base_model`. No data reads:
    `effects` holds M8's already-loaded models. Keywords go to `calibrate`."""
    base = _kernels_of(base_model)
    c = calibrate(base, off_rating, def_rating, effects, **calibration)
    return tilt_kernels(base, c.tilt, c.turnover_scale, c.conv_dlogit)


def matchup_model(
    base_model: mg.GameModel,
    off_rating: dict[str, float],
    def_rating: dict[str, float],
    effects: RatingEffects,
    **calibration: Any,
) -> tuple[mg.GameModel, Calibration]:
    """`base_model` with the kernels of one offense against one defense, and the
    calibration behind them."""
    c = calibrate(base_model.kern, off_rating, def_rating, effects, **calibration)
    kern = tilt_kernels(base_model.kern, c.tilt, c.turnover_scale, c.conv_dlogit)
    return replace(base_model, kern=kern), c


def endgame_model(model: mg.GameModel) -> mg.GameModel:
    """The same game restricted to its last `endgame_ticks` (the end phase only). The end
    phase never reads later times, so its values equal those of the whole-game solve."""
    m = copy.copy(model)
    m.clock = copy.deepcopy(model.clock)
    m.clock.n_ticks, m.clock.half_tick = m.clock.endgame_ticks, None
    return m


# --------------------------------------------------------------------------------------
# Two-team solve
# --------------------------------------------------------------------------------------


def _same(x: Any, y: Any) -> bool:
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        if x is None or y is None:
            return x is y
        return np.array_equal(x, y)
    return x == y


def check_compatible(a: mg.GameModel, b: mg.GameModel) -> None:
    """Two teams must play the same game: same grid, clock, scoring and kicks. Only the
    plays (and field-goal and punt tables, if a caller rates them) may differ."""
    problems = []
    if a.kern.grid != b.kern.grid:
        problems.append("grid")
    for name in ("score_max", "td_points", "fg_points", "safety_points", "p_xp", "p_two",
                 "two_point_choice", "dtype"):  # fmt: skip
        if getattr(a, name) != getattr(b, name):
            problems.append(name)
    for name in vars(a.clock):
        if not _same(getattr(a.clock, name), getattr(b.clock, name)):
            problems.append(f"clock.{name}")
    for name in ("kickoff", "free_kick"):
        if not np.array_equal(getattr(a.kern, name), getattr(b.kern, name)):
            problems.append(f"kern.{name}")
    if problems:
        raise ValueError(f"the two teams' models differ in {problems}")


def link(a: mg.Solution, b: mg.Solution) -> None:
    """Make each solution read the other team's values from the other (see the module
    docstring). Linking a solution to itself restores the identical-team game."""
    a._other, b._other = b, a
    for s in (a, b):
        s._posts.clear()
        s._bases.clear()
        s._q_cache = None


@dataclass
class Matchup:
    """The solved two-team game. `sols[0]` holds A's values when A has the ball, `sols[1]`
    B's when B has it; states are always described from the side of the team with the
    ball, as in `mdp_game.GameState`."""

    sols: tuple[mg.Solution, mg.Solution]
    meta: dict[str, Any]

    def solution(self, team: int | str) -> mg.Solution:
        return self.sols[TEAMS.index(team) if isinstance(team, str) else int(team)]

    def value(self, state: mg.GameState, team: int | str = "A") -> float:
        """Win probability of `team`, which has the ball in `state`."""
        return self.solution(team).value(state)

    def policy(self, state: mg.GameState, team: int | str = "A") -> str:
        return self.solution(team).policy(state)

    def q_values(self, state: mg.GameState, team: int | str = "A") -> dict[str, float]:
        return self.solution(team).q_values(state)


def solve_matchup(model_a: mg.GameModel, model_b: mg.GameModel, progress: bool = False
                  ) -> Matchup:  # fmt: skip
    """Backward induction with two value arrays. With model_a == model_b the result equals
    `mdp_game.solve(model_a)` in both arrays (tests/test_mdp_team.py)."""
    check_compatible(model_a, model_b)
    a, b = mg.Solution(model_a), mg.Solution(model_b)
    link(a, b)
    seconds = mg.backward_induction([a, b], progress)
    meta = {
        "solve_seconds": seconds,
        "peak_memory_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "n_states_per_team": model_a.n_states(),
    }
    for s in (a, b):
        s.meta.update(meta)
    return Matchup((a, b), meta)


# --------------------------------------------------------------------------------------
# Two-team simulator
# --------------------------------------------------------------------------------------


class MatchupSimulator:
    """Batch simulator of a solved two-team game, with the interface `mdp_game.simulate`
    uses. Each batch member records which team had the ball at the start (`team0`); the
    team with the ball now is team0 XOR side. Every play is sampled by that team's own
    `mdp_game.Simulator` (its kernels, its solved policy and timeout decisions), and a
    return touchdown takes the scoring team's two-point choice."""

    def __init__(self, match: Matchup) -> None:
        self.match = match
        self.sims = tuple(mg.Simulator(s) for s in match.sols)
        self.sims[0].other, self.sims[1].other = self.sims[1], self.sims[0]
        self.sol, self.model, self.grid = match.sols[0], match.sols[0].model, match.sols[0].grid

    def start(self, n: int, team: int = 0, **kw: Any) -> dict[str, np.ndarray]:
        """n copies of one state, from the side of `team` (0 = A, 1 = B), which has the
        ball. Keywords as `mdp_game.Simulator.start`."""
        st = self.sims[0].start(n, **kw)
        st["team0"] = np.full(n, int(team), dtype=np.int64)
        return st

    def team(self, st: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
        return st["team0"][idx] ^ st["side"][idx]

    def values(self, st: dict[str, np.ndarray], idx: np.ndarray | None = None) -> np.ndarray:
        """Solved win probability of the team with the ball, for each batch member."""
        idx = np.arange(len(st["t"])) if idx is None else idx
        tm = self.team(st, idx)
        out = np.zeros(len(idx))
        for k, sim in enumerate(self.sims):
            q = tm == k
            if q.any():
                out[q] = sim.values(st, idx[q])
        return out

    def optimal_actions(self, st: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
        tm = self.team(st, idx)
        out = np.zeros(len(idx), dtype=np.int64)
        for k, sim in enumerate(self.sims):
            q = tm == k
            if q.any():
                out[q] = sim.optimal_actions(st, idx[q])
        return out

    def game_over(self, st: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
        return self.sims[0].game_over(st, idx)  # the kneel rule is the same for both

    def advance(
        self,
        st: dict[str, np.ndarray],
        idx: np.ndarray,
        act: np.ndarray,
        rng: np.random.Generator,
        post: str = "model",
    ) -> dict[str, np.ndarray]:
        tm = self.team(st, idx)
        out: dict[str, np.ndarray] = {}
        for k, sim in enumerate(self.sims):
            q = np.nonzero(tm == k)[0]
            if not len(q):
                continue
            info = sim.advance(st, idx[q], act[q], rng, post)
            for name, x in info.items():
                if name not in out:
                    out[name] = np.zeros(len(idx), dtype=x.dtype)
                out[name][q] = x
        return out
