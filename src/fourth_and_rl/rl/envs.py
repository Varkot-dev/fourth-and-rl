"""M5 environments, built from the solved M4 models.

Two problems, each in two forms (a Gymnasium environment that plays one episode at a time,
and a batch version that plays many episodes at once for fast training and evaluation):

DriveEnv / DriveBatch
    The single-drive expected-points problem of `mdp_drive.py`. The agent decides on fourth
    down only (go, field goal, punt); downs 1 to 3 run the generic play automatically
    inside `step`. An episode is one possession. It ends when the ball changes hands or
    somebody scores, with one terminal reward:

        points scored on the play  minus  the solved value of what the other team now holds

    that is, exactly the outcome values W(o) of stage 1 with the solved V plugged in:
        other team takes over at their y     -V*(first and 10 at y)
        touchdown                            td_value - K*
        field goal made                      3 - K*
        safety against us                    -2 - F*
        they return it for a touchdown       -td_value + K*
    The opponent's possession is therefore not simulated: it is summarized by the number
    V* that the dynamic program already computed. With that terminal reward the Bellman
    equation of the environment is the stage 1 equation with the continuation frozen at
    V*, whose unique solution is V* itself, so the optimal policy of the environment is
    the stage 1 optimum (tested).

EndgameEnv / EndgameBatch
    The last five minutes of the win-probability game of `mdp_game.py`. The agent calls
    the plays of the team that has the ball at the start, whenever that team has the
    ball; the other team's plays are called by the solved optimal policy ("optimal") or
    by the coach-like policy of the simulator ("coach"). Decisions that are not a play
    call (timeouts, the try after a touchdown) are made by the simulator as the solver
    assumed. Reward: 1 for a win, 0.5 for a tie after regulation, 0 for a loss, paid once
    at the end.

Every transition is sampled by `mdp_game.Simulator` (its `advance` method for the end
game, its play and punt samplers for the drive), so the environments cannot drift away
from the model that was solved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fourth_and_rl import mdp_game as mg
from fourth_and_rl.mdp import FieldKernels, Grid
from fourth_and_rl.mdp_drive import DRIVE_ACTIONS, solve_drive

GO, FIELD_GOAL, PUNT = range(3)
ENDGAME_ACTIONS = mg.ACTION_NAMES[:6]  # play, play_protect, play_chase, field_goal, punt, kneel
N_ENDGAME_FEATURES = 16


# --------------------------------------------------------------------------------------
# Small sampling helper
# --------------------------------------------------------------------------------------


class RowSampler:
    """Draws a column from chosen rows of a dense matrix of probabilities."""

    def __init__(self, probs: np.ndarray) -> None:
        total = probs.sum(axis=1, keepdims=True)
        cum = np.cumsum(probs, axis=1) / np.where(total > 0, total, 1.0)
        cum[cum > 1.0 - 1e-12] = 1.0
        self.cum = cum

    def sample(self, rows: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One column per row index, from uniforms `u` in [0, 1)."""
        return (u[:, None] >= self.cum[rows]).sum(axis=1)


# --------------------------------------------------------------------------------------
# Drive: the decision-level model
# --------------------------------------------------------------------------------------


@dataclass
class DriveModel:
    """The single-drive problem seen from fourth down to fourth down.

    Downs 1 to 3 hold no decision, so they can be summed out exactly: from a fourth-down
    state and an action, `dist[a, s]` is the probability of each thing the agent sees
    next. Columns [0, n) are the next fourth-down state of the same drive, columns
    [n, n + n_term) the ways the drive can end:

        n + (y - 1)      the other team takes over at THEIR yard line y
        n + n_yl         touchdown
        n + n_yl + 1     safety against us
        n + n_yl + 2     the other team scores a touchdown on the play
        n + n_yl + 3     field goal made

    `term_reward` is the reward of each ending (see the module docstring). The same
    arrays serve exact policy evaluation (a linear system) and fast sampling.
    """

    kern: FieldKernels
    v_field: np.ndarray  # (nf,) solved value of every field state
    states: np.ndarray  # (n,) field-state index of each fourth-down state
    dist: np.ndarray  # (3, n, n + n_term)
    term_reward: np.ndarray  # (n_term,)
    allowed: np.ndarray  # (n, 3) bool
    start: np.ndarray  # (n,) distribution of the first fourth down of a drive
    start_yardline: np.ndarray  # (n_yl + 1,) where drives begin (index 0 unused)
    p_reach: float  # share of drives that reach a fourth down at all
    td_value: float
    kickoff_value: float
    free_kick_value: float
    v_star: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (n,)
    q_star: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (n, 3), -inf if not allowed
    mode: str = "neutral"  # the play kernel (`FieldKernels.k_all` key) the model was built with

    @property
    def grid(self) -> Grid:
        return self.kern.grid

    @property
    def n(self) -> int:
        return len(self.states)

    @property
    def n_term(self) -> int:
        return len(self.term_reward)

    @property
    def policy_star(self) -> np.ndarray:
        return self.q_star.argmax(axis=1)

    def outcome_values(self, v: np.ndarray) -> np.ndarray:
        """Value of every column of `dist` given values `v` of the fourth-down states."""
        return np.concatenate([v, self.term_reward])

    def features(self) -> np.ndarray:
        """(n, 2 + n_dist) inputs for function approximation: yard line and distance
        scaled to [0, 1] and a one-hot code of the distance bin."""
        g = self.grid
        _, dbin, yl = g.unpack(self.states)
        rep = np.minimum(g.dist_rep(dbin), yl)
        out = np.zeros((self.n, 2 + g.n_dist), dtype=np.float32)
        out[:, 0] = yl / g.n_yl
        out[:, 1] = rep / max(float(g.dist_rep(g.n_dist - 1)), 1.0)
        out[np.arange(self.n), 2 + dbin] = 1.0
        return out


def default_start_yardline(kern: FieldKernels, kickoff_share: float = 0.5) -> np.ndarray:
    """Where a drive begins: with probability `kickoff_share` after a kickoff (the model's
    kickoff distribution), otherwise at a yard line drawn uniformly (drives that follow
    punts and turnovers begin anywhere). Index 0 unused."""
    n_yl = kern.grid.n_yl
    uniform = np.concatenate([[0.0], np.full(n_yl, 1.0 / n_yl)])
    return kickoff_share * kern.kickoff + (1.0 - kickoff_share) * uniform


def build_drive_model(
    kern: FieldKernels,
    td_value: float,
    fg_points: float = 3.0,
    safety_points: float = 2.0,
    start_yardline: np.ndarray | None = None,
    v_field: np.ndarray | None = None,
    mode: str = "neutral",
) -> DriveModel:
    """Collapse the play-by-play kernels to a fourth-down-to-fourth-down model.

    `v_field` is the solved stage 1 value of every field state; it is solved here with
    `mdp_drive.solve_drive` when not given. Let N be the states of downs 1 to 3 and P the
    play kernel. From a state in N the drive wanders inside N until it reaches a fourth
    down or ends, so the probabilities of where it first lands are

        B = (I - P[N, N])^-1 P[N, fourth downs and endings]

    (the standard absorbing-chain formula), and a "go" that keeps the ball, or a punt the
    kicking team recovers, continues through B.
    """
    g = kern.grid
    nf, n_yl = g.nf, g.n_yl
    f10 = g.first10()
    if v_field is None:
        solved = solve_drive(kern, td_value, fg_points, safety_points, mode=mode)
        v_field = solved["v"]
    k_val = float(kern.kickoff[1:] @ v_field[f10[1:]])
    fk_val = float(kern.free_kick[1:] @ v_field[f10[1:]])
    downs = g.unpack(np.arange(nf))[0]
    dec = np.nonzero(downs == g.n_down)[0]
    non = np.nonzero(downs < g.n_down)[0]
    n, n_term = len(dec), n_yl + 4
    term_reward = np.concatenate(
        [-v_field[f10[1:]], [td_value - k_val, -safety_points - fk_val, -td_value + k_val]]
    )
    term_reward = np.concatenate([term_reward, [fg_points - k_val]])

    def split(mat: Any, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rows of a kernel as (to downs 1-3, to [fourth downs | endings])."""
        m = np.asarray(mat[rows].todense())
        land = np.zeros((len(rows), n + n_term))
        land[:, :n] = m[:, dec]
        land[:, n : n + n_yl + 3] = m[:, nf:]
        return m[:, non], land

    play = kern.k_all[mode]
    if len(non):
        p_nn, p_land = split(play, non)
        b = np.linalg.solve(np.eye(len(non)) - p_nn, p_land)
    else:
        b = np.zeros((0, n + n_term))
    pos_non = np.full(nf, -1)
    pos_non[non] = np.arange(len(non))

    dist = np.zeros((3, n, n + n_term))
    to_non, land = split(play, dec)
    dist[GO] = to_non @ b + land
    to_non, land = split(kern.k_punt, dec)
    dist[PUNT] = to_non @ b + land
    make = kern.fg_make[dec]
    dist[FIELD_GOAL, np.arange(n), n + n_yl + 3] = make
    np.add.at(dist[FIELD_GOAL], (np.arange(n), n + kern.fg_miss_col[dec] - nf), 1.0 - make)
    allowed = np.column_stack([np.ones(n, dtype=bool), kern.fg_ok[dec], kern.punt_ok[dec]])
    dist[FIELD_GOAL][~allowed[:, FIELD_GOAL]] = 0.0
    dist[PUNT][~allowed[:, PUNT]] = 0.0

    if start_yardline is None:
        start_yardline = default_start_yardline(kern)
    start_yardline = np.asarray(start_yardline, dtype=float)
    if g.n_down > 1:
        first = start_yardline[1:] @ b[pos_non[f10[1:]]]
        start, p_reach = first[:n] / first[:n].sum(), float(first[:n].sum())
    else:  # every state is a decision state
        start, p_reach = np.zeros(n), 1.0
        np.add.at(start, np.searchsorted(dec, f10[1:]), start_yardline[1:])

    model = DriveModel(
        kern=kern,
        v_field=np.asarray(v_field, dtype=float),
        states=dec,
        dist=dist,
        term_reward=term_reward,
        allowed=allowed,
        start=start,
        start_yardline=start_yardline,
        p_reach=p_reach,
        td_value=float(td_value),
        kickoff_value=k_val,
        free_kick_value=fk_val,
        mode=mode,
    )
    model.v_star = model.v_field[dec]
    q = np.einsum("asc,c->sa", dist, model.outcome_values(model.v_star))
    model.q_star = np.where(allowed, q, -np.inf)
    return model


class CompiledDrive:
    """The decision-level drive as plain Python lists, for the hand-written tabular loops.

    It samples from exactly `DriveModel.dist`; a NumPy call per step would be ten times
    slower than the learning update itself. `reset(u)` and `step(s, a, u)` take a uniform
    random number, so a run is a deterministic function of the stream of uniforms.
    """

    def __init__(self, model: DriveModel) -> None:
        self.n_states, self.n_actions = model.n, 3
        self.allowed = model.allowed
        self.valid = [np.nonzero(row)[0].tolist() for row in model.allowed]
        n = model.n
        self._cum: list[list[list[float]]] = []
        self._out: list[list[list[int]]] = []
        for s in range(n):
            cum_s, out_s = [], []
            for a in range(3):
                nz = np.nonzero(model.dist[a, s])[0]
                c = np.cumsum(model.dist[a, s, nz])
                cum_s.append((c / c[-1]).tolist() if len(nz) else [])
                out_s.append(nz.tolist())
            self._cum.append(cum_s)
            self._out.append(out_s)
        self._term = [0.0] * n + model.term_reward.tolist()
        nz = np.nonzero(model.start)[0]
        self._start_cum = np.cumsum(model.start[nz]).tolist()
        self._start_out = nz.tolist()
        self._n = n

    def reset(self, u: float) -> int:
        from bisect import bisect_right

        i = bisect_right(self._start_cum, u)
        return self._start_out[min(i, len(self._start_out) - 1)]

    def step(self, s: int, a: int, u: float) -> tuple[int, float, bool]:
        """Returns (next state, reward, done). The next state is meaningless when done."""
        from bisect import bisect_right

        cum = self._cum[s][a]
        col = self._out[s][a][min(bisect_right(cum, u), len(cum) - 1)]
        if col < self._n:
            return col, 0.0, False
        return s, self._term[col], True


class DriveBatch:
    """Many drives at once, sampled from `DriveModel.dist` (for the deep agents).

    `step` maps an action that is not allowed (a field goal from too far) to a punt;
    `illegal_actions` counts how many actions were replaced that way since construction.
    """

    n_actions = 3

    def __init__(self, model: DriveModel, n: int, seed: int | None = None) -> None:
        self.model, self.n = model, n
        self.illegal_actions = 0
        self.rng = np.random.default_rng(seed)
        self._samplers = [RowSampler(model.dist[a]) for a in range(3)]
        self._start = RowSampler(model.start[None, :])
        self._features = model.features()
        self.s = np.zeros(n, dtype=np.int64)
        self.reset()

    def reset(self, idx: np.ndarray | None = None) -> None:
        idx = np.arange(self.n) if idx is None else idx
        if len(idx):
            self.s[idx] = self._start.sample(np.zeros(len(idx), dtype=np.int64), self._u(len(idx)))

    def _u(self, k: int) -> np.ndarray:
        return self.rng.random(k)

    def valid_action(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        return np.where(self.model.allowed[s, a], a, PUNT)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One decision in every drive. Returns (reward, done); finished drives are NOT
        restarted (call `reset(idx)`)."""
        asked = np.asarray(actions, dtype=np.int64)
        a = self.valid_action(self.s, asked)
        self.illegal_actions += int(np.sum(a != asked))
        u = self._u(self.n)
        col = np.zeros(self.n, dtype=np.int64)
        for code in range(3):
            sel = np.nonzero(a == code)[0]
            if len(sel):
                col[sel] = self._samplers[code].sample(self.s[sel], u[sel])
        done = col >= self.model.n
        reward = np.where(done, self.model.term_reward[np.maximum(col - self.model.n, 0)], 0.0)
        self.s = np.where(done, self.s, col)
        return reward, done

    def observe(self) -> np.ndarray:
        return self._features[self.s]

    def mask(self) -> np.ndarray:
        return self.model.allowed[self.s]


class DriveEnv(gym.Env):
    """One possession, play by play, sampled with the simulator's own samplers.

    Observation: the index of the fourth-down state (`obs_mode="index"`, for tabular
    agents), `DriveModel.features` ("features") or a one-hot vector ("onehot").
    Actions: 0 go, 1 field goal, 2 punt. A field goal from where the model does not allow
    one is played as a punt (`info["action_mask"]` says what is allowed); such a step has
    `info["illegal_action"] = True` and adds one to `illegal_actions`.
    Reward: 0 until the possession ends, then the terminal reward of the module docstring.
    `reset` draws a drive start, plays downs 1 to 3 and returns the first fourth down;
    drives that end before any fourth down hold no decision and are redrawn.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model: DriveModel,
        play_sampler: Any,
        punt_sampler: Any,
        obs_mode: str = "index",
    ) -> None:
        if obs_mode not in ("index", "features", "onehot"):
            raise ValueError("obs_mode must be 'index', 'features' or 'onehot'")
        self.model, self.obs_mode = model, obs_mode
        self._play, self._punt = play_sampler, punt_sampler
        g = model.grid
        self._pos = np.full(g.nf, -1)
        self._pos[model.states] = np.arange(model.n)
        self._features = model.features()
        self.action_space = spaces.Discrete(3)
        if obs_mode == "index":
            self.observation_space: spaces.Space = spaces.Discrete(model.n)
        else:
            k = self._features.shape[1] if obs_mode == "features" else model.n
            self.observation_space = spaces.Box(0.0, 1.0, (k,), dtype=np.float32)
        self._f = int(model.states[0])
        self._done = True
        self.plays = 0
        self.illegal_actions = 0

    @classmethod
    def from_simulator(cls, sim: mg.Simulator, model: DriveModel, **kw: Any) -> DriveEnv:
        """The environment whose plays and punts are drawn by `sim`'s samplers, using the
        play kernel the drive model was built with (`model.mode`)."""
        return cls(model, sim.play[model.mode], sim.punt, **kw)

    # ---- helpers ----
    def _obs(self) -> Any:
        i = int(self._pos[self._f])
        if self.obs_mode == "index":
            return np.int64(i)
        if self.obs_mode == "features":
            return self._features[i].copy()
        out = np.zeros(self.model.n, dtype=np.float32)
        out[i] = 1.0
        return out

    def _info(self) -> dict[str, Any]:
        i = int(self._pos[self._f])
        return {"action_mask": self.model.allowed[i].copy(), "state": i, "plays": self.plays}

    def _run_play(self, f: int) -> int:
        """One generic play (or a "go") from field state f: an outcome column of the
        kernel (the simulator's sampler marks clock-running outcomes by an offset)."""
        g = self.model.grid
        col = int(self._play.sample(np.array([f]), self.np_random.random(1))[0])
        self.plays += 1
        return col - g.na if col >= g.na else col

    def _advance_to_decision(self, col: int) -> tuple[int | None, float]:
        """Follow outcome column `col` through downs 1 to 3. Returns (fourth-down field
        state, 0) or (None, terminal reward)."""
        g = self.model.grid
        while True:
            if col >= g.nf:
                return None, float(self.model.term_reward[col - g.nf])
            if g.unpack(col)[0] == g.n_down:
                return col, 0.0
            col = self._run_play(col)

    # ---- Gymnasium API ----
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        g = self.model.grid
        self.plays = 0
        if options and "state" in options:
            self._f = int(self.model.states[int(options["state"])])
        else:
            cum = np.cumsum(self.model.start_yardline[1:])
            f10 = g.first10()
            while True:
                yl = int(np.searchsorted(cum, self.np_random.random() * cum[-1], side="right")) + 1
                f, _ = self._advance_to_decision(int(f10[min(yl, g.n_yl)]))
                if f is not None:
                    self._f = f
                    break
        self._done = False
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("episode is over; call reset()")
        m, g, f = self.model, self.model.grid, self._f
        a = int(action)
        illegal = not m.allowed[self._pos[f], a]
        if illegal:
            a = PUNT
            self.illegal_actions += 1
        if a == GO:
            col = self._run_play(f)
        elif a == FIELD_GOAL:
            self.plays += 1
            made = self.np_random.random() < m.kern.fg_make[f]
            col = g.nf + g.n_yl + 3 if made else int(m.kern.fg_miss_col[f])
        else:
            self.plays += 1
            col = int(self._punt.sample(np.array([f]), self.np_random.random(1))[0])
        if a == FIELD_GOAL and col == g.nf + g.n_yl + 3:
            nxt, reward = None, float(m.term_reward[-1])
        else:
            nxt, reward = self._advance_to_decision(col)
        info = {"action_taken": DRIVE_ACTIONS[a], "illegal_action": illegal}
        if nxt is None:
            self._done = True
            return self._obs(), reward, True, False, {**self._info(), **info}
        self._f = nxt
        return self._obs(), 0.0, False, False, {**self._info(), **info}


# --------------------------------------------------------------------------------------
# End game
# --------------------------------------------------------------------------------------


@dataclass
class StartSpec:
    """Distribution of start states for the end game (all documented in configs/rl.yaml).

    Time left uniform between `seconds_min` and `seconds_max`; score differential of the
    agent's team uniform in -score_abs_max..+score_abs_max; with probability
    `p_first_and_10` a first and 10 (or goal) at a uniform yard line, otherwise a uniform
    down, yard line and distance (1 to `max_distance`, at most the yard line); timeouts of
    each team drawn from the shares the model uses on entering the last 5:00 (leading and
    trailing teams differ). States where the game is already decided by kneeling are
    redrawn.
    """

    seconds_min: int = 30
    seconds_max: int = 300
    score_abs_max: int = 8
    p_first_and_10: float = 0.7
    max_distance: int = 15


class EndgameBatch:
    """Many end games at once. Every unfinished game is always at a decision of the agent.

    Arrays of the simulator state live in `self.st` (see `mdp_game.Simulator`). `step`
    plays the agent's action, then lets the opponent play until the agent has the ball
    again or the game ends. An action that is not allowed in a state (protect when
    trailing, chase when leading, a punt before fourth down, a field goal from where the
    model has none) is played as the neutral play; `mask()` says what is allowed,
    `illegal_actions` counts the replaced actions since construction and
    `last["illegal"]` marks them on the latest step.

    shaping: "none" (sparse reward), "value" (potential-based shaping with the solved
    value as the potential: uses the answer, a diagnostic only) or "score" (potential
    1 / (1 + exp(-score / 4)), a crude guess that needs no solution). Shaping adds
    phi(next) - phi(now) to every reward with phi = 0 at the end, which leaves the
    ranking of policies unchanged.
    """

    n_actions = 6

    def __init__(
        self,
        sim: mg.Simulator,
        n: int,
        seed: int | None = None,
        opponent: str = "optimal",
        start: StartSpec | None = None,
        shaping: str = "none",
    ) -> None:
        if opponent not in ("optimal", "coach"):
            raise ValueError("opponent must be 'optimal' or 'coach'")
        if opponent == "coach" and sim.coach is None:
            raise ValueError("the coach opponent needs a Simulator built with a coach table")
        if shaping not in ("none", "value", "score"):
            raise ValueError("shaping must be 'none', 'value' or 'score'")
        m = sim.model
        if m.clock.endgame_ticks <= 0:
            raise ValueError("the model has no end-game phase")
        self.sim, self.n, self.opponent, self.shaping = sim, n, opponent, shaping
        self.start_spec = start or StartSpec()
        self.rng = np.random.default_rng(seed)
        self.st = sim.start(n, t=1)
        self.finished = np.ones(n, dtype=bool)
        self.result = np.full(n, np.nan)
        self.last: dict[str, np.ndarray] = {}
        self.opponent_plays = 0
        self.illegal_actions = 0

    # ---- start states ----
    def sample_states(self, k: int) -> dict[str, np.ndarray]:
        """k start states as arrays t, s, f, a, b (see `StartSpec`)."""
        sim, m, g, sp, rng = self.sim, self.sim.model, self.sim.grid, self.start_spec, self.rng
        c = m.clock
        out = {key: np.zeros(k, dtype=np.int64) for key in ("t", "s", "f", "a", "b")}
        todo = np.arange(k)
        lo = max(int(np.ceil(sp.seconds_min / c.tick_seconds)), 1)
        hi = min(sp.seconds_max // c.tick_seconds, c.endgame_ticks)
        smax = min(sp.score_abs_max, m.score_max)
        lead, trail = c.shares_leading, c.shares_trailing
        uniform = np.full(c.n_timeouts + 1, 1.0 / (c.n_timeouts + 1))
        lead = uniform if lead is None else lead
        trail = uniform if trail is None else trail
        mean = 0.5 * (lead + trail)
        while len(todo):
            j = len(todo)
            t = rng.integers(lo, hi + 1, j)
            s = rng.integers(-smax, smax + 1, j)
            yl = rng.integers(1, g.n_yl + 1, j)
            down = rng.integers(1, g.n_down + 1, j)
            dist = 1 + (rng.random(j) * np.minimum(yl, sp.max_distance)).astype(np.int64)
            first = rng.random(j) < sp.p_first_and_10
            f = np.where(first, sim.f10[yl], g.fidx(down, g.dist_bin(dist), yl))
            u1, u2 = rng.random(j), rng.random(j)
            a, b = np.zeros(j, dtype=np.int64), np.zeros(j, dtype=np.int64)
            for sign, pa, pb in ((1, lead, trail), (-1, trail, lead), (0, mean, mean)):
                q = np.sign(s) == sign
                a[q] = np.minimum(np.searchsorted(np.cumsum(pa), u1[q], side="right"), len(pa) - 1)
                b[q] = np.minimum(np.searchsorted(np.cumsum(pb), u2[q], side="right"), len(pb) - 1)
            trial = {"t": t, "s": s, "f": f, "a": a, "b": b}
            ok = ~sim.game_over(trial, np.arange(j))
            for key, arr in trial.items():
                out[key][todo[ok]] = arr[ok]
            todo = todo[~ok]
        return out

    def reset(
        self, idx: np.ndarray | None = None, states: dict[str, np.ndarray] | None = None
    ) -> None:
        """Start new games in the slots `idx` (all by default), from `states` if given."""
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        if len(idx) == 0:
            return
        new = self.sample_states(len(idx)) if states is None else states
        st = self.st
        for key in ("t", "s", "f", "a", "b"):
            st[key][idx] = new[key]
        st["s"][idx] = np.clip(st["s"][idx], -self.sim.model.score_max, self.sim.model.score_max)
        st["r"][idx] = 0
        st["side"][idx] = 0
        st["pts"][idx] = 0
        self.finished[idx] = False
        self.result[idx] = np.nan

    # ---- views of the state ----
    def mask(self, idx: np.ndarray | None = None) -> np.ndarray:
        """(k, 6) bool: which actions the solved model allows in each state."""
        idx = np.arange(self.n) if idx is None else idx
        k = self.sim.model.kern
        f, s = self.st["f"][idx], self.st["s"][idx]
        out = np.ones((len(idx), 6), dtype=bool)
        out[:, mg.A_PROTECT] = s >= 0
        out[:, mg.A_CHASE] = s <= 0
        out[:, mg.A_FG] = k.fg_ok[f]
        out[:, mg.A_PUNT] = k.punt_ok[f]
        out[:, mg.A_KNEEL] = k.kneel_ok[f]
        return out

    def observe(self, idx: np.ndarray | None = None) -> np.ndarray:
        """(k, 16) normalized features of each state, from the agent's side: time left,
        inside-the-warning flag, score, five score flags (ahead, tied, behind, within a
        field goal, within one score), down (one-hot of four), distance, yard line, own
        timeouts, opponent's timeouts."""
        idx = np.arange(self.n) if idx is None else idx
        m, g, c = self.sim.model, self.sim.grid, self.sim.model.clock
        st = self.st
        t, s, f = st["t"][idx], st["s"][idx], st["f"][idx]
        down, dbin, yl = g.unpack(f)
        nt = max(c.n_timeouts, 1)
        out = np.zeros((len(idx), N_ENDGAME_FEATURES), dtype=np.float32)
        out[:, 0] = t / c.endgame_ticks
        out[:, 1] = t <= (c.warning_tick or 0)
        out[:, 2] = s / m.score_max
        out[:, 3] = s > 0
        out[:, 4] = s == 0
        out[:, 5] = s < 0
        out[:, 6] = np.abs(s) <= m.fg_points
        out[:, 7] = np.abs(s) <= m.td_points + 2
        out[np.arange(len(idx)), 8 + np.minimum(down, 4) - 1] = 1.0
        out[:, 12] = np.minimum(g.dist_rep(dbin), yl) / max(float(g.dist_rep(g.n_dist - 1)), 1.0)
        out[:, 13] = yl / g.n_yl
        out[:, 14] = st["a"][idx] / nt
        out[:, 15] = st["b"][idx] / nt
        return out

    def state_index(self, idx: np.ndarray | None = None) -> np.ndarray:
        """Index of each state in the solved end-game array (time, field, score, timeouts)."""
        idx = np.arange(self.n) if idx is None else idx
        m, st = self.sim.model, self.st
        shape = (m.clock.endgame_ticks + 1, self.sim.grid.nf, m.n_score, *m.extra_shape("end"))
        return np.ravel_multi_index(
            (st["t"][idx], st["f"][idx], st["s"][idx] + m.score_max, st["a"][idx], st["b"][idx]),
            shape,
        )

    def potential(self, idx: np.ndarray) -> np.ndarray:
        if self.shaping == "value":
            return self.sim.values(self.st, idx).astype(float)
        if self.shaping == "score":
            return 1.0 / (1.0 + np.exp(-self.st["s"][idx] / 4.0))
        return np.zeros(len(idx))

    # ---- dynamics ----
    def _settle(self, idx: np.ndarray) -> np.ndarray:
        """Mark the games of `idx` that are over and record their result. Returns the mask."""
        sim, st = self.sim, self.st
        ended = st["t"][idx] == 0
        over = sim.game_over(st, idx) & ~ended
        fin = ended | over
        if fin.any():
            j = idx[fin]
            val = np.where(over[fin], 1.0, sim.sol.terminal[st["s"][j] + sim.model.score_max])
            self.result[j] = np.where(st["side"][j] == 0, val, 1.0 - val)
            self.finished[j] = True
        return fin

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Play `actions` (one per slot; entries of finished games are ignored).

        Returns (reward, done) for every slot; `done` is True only for games that ended on
        this step. `self.last` keeps the states the agent acted in and the actions actually
        played, for evaluation.
        """
        sim, st = self.sim, self.st
        idx = np.nonzero(~self.finished)[0]
        reward, done = np.zeros(self.n), np.zeros(self.n, dtype=bool)
        if len(idx) == 0:
            return reward, done
        act = np.asarray(actions, dtype=np.int64)[idx]
        legal = self.mask(idx)[np.arange(len(idx)), act]
        act = np.where(legal, act, mg.A_NEUTRAL)
        self.illegal_actions += int((~legal).sum())
        self.last = {key: st[key][idx].copy() for key in ("t", "s", "f", "a", "b")}
        self.last["idx"], self.last["action"], self.last["illegal"] = idx, act, ~legal
        phi0 = self.potential(idx)
        sim.advance(st, idx, act, self.rng, "model")
        post = "coach" if self.opponent == "coach" else "model"
        pending = idx
        while len(pending):
            pending = pending[~self._settle(pending)]
            pending = pending[st["side"][pending] == 1]
            if len(pending) == 0:
                break
            if self.opponent == "coach":
                opp = sim.coach_actions(st, pending, self.rng)
            else:
                opp = sim.optimal_actions(st, pending)
            sim.advance(st, pending, opp, self.rng, post)
            self.opponent_plays += len(pending)
        fin = self.finished[idx]
        done[idx] = fin
        reward[idx] = np.where(fin, self.result[idx], 0.0)
        if self.shaping != "none":
            reward[idx] += np.where(fin, 0.0, self.potential(idx)) - phi0
        return reward, done


class EndgameEnv(gym.Env):
    """One end game at a time: a thin Gymnasium wrapper around `EndgameBatch` with one slot.

    Observation: the 16 features of `EndgameBatch.observe` (`obs_mode="features"`) or the
    index of the state in the solved array ("index", for tabular use).
    Actions: 0 play, 1 play_protect, 2 play_chase, 3 field_goal, 4 punt, 5 kneel, as in the
    solved model. An action that is not allowed is played as 0; `info["action_mask"]`
    says what is allowed and `info["illegal_action"]` whether this step's was replaced.
    Reward: 1 win, 0.5 tie, 0 loss, at the end only (unless `shaping` is set).
    `reset(options={"state": GameState})` starts from a given state.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sim: mg.Simulator,
        opponent: str = "optimal",
        start: StartSpec | None = None,
        obs_mode: str = "features",
        shaping: str = "none",
    ) -> None:
        if obs_mode not in ("features", "index"):
            raise ValueError("obs_mode must be 'features' or 'index'")
        self.batch = EndgameBatch(sim, 1, opponent=opponent, start=start, shaping=shaping)
        self.sim, self.obs_mode = sim, obs_mode
        self.action_space = spaces.Discrete(6)
        if obs_mode == "features":
            self.observation_space: spaces.Space = spaces.Box(
                -1.0, 1.0, (N_ENDGAME_FEATURES,), dtype=np.float32
            )
        else:
            m = sim.model
            e = m.extra_shape("end")
            n = (m.clock.endgame_ticks + 1) * sim.grid.nf * m.n_score * e[0] * e[1]
            self.observation_space = spaces.Discrete(n)

    def _obs(self) -> Any:
        if self.obs_mode == "features":
            return self.batch.observe()[0]
        return np.int64(self.batch.state_index()[0])

    def game_state(self) -> mg.GameState:
        """The current state as a `GameState`, from the side of the team with the ball."""
        st, g = self.batch.st, self.sim.grid
        down, dbin, yl = g.unpack(st["f"])
        return mg.GameState(
            seconds=int(st["t"][0]) * self.sim.model.clock.tick_seconds,
            score=int(st["s"][0]),
            down=int(down[0]),
            ydstogo=int(min(g.dist_rep(dbin)[0], yl[0])),
            yardline_100=int(yl[0]),
            off_timeouts=int(st["a"][0]),
            def_timeouts=int(st["b"][0]),
        )

    def _info(self) -> dict[str, Any]:
        return {
            "action_mask": self.batch.mask()[0],
            "state_index": int(self.batch.state_index()[0]),
            "state": self.game_state(),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        self.batch.rng = self.np_random
        states = None
        if options and "state" in options:
            s: mg.GameState = options["state"]
            _, ix = self.sim.sol.index(s)
            states = {
                "t": np.array([round(s.seconds / self.sim.model.clock.tick_seconds)]),
                "s": np.array([s.score]),
                "f": np.array([ix[1]]),
                "a": np.array([s.off_timeouts]),
                "b": np.array([s.def_timeouts]),
            }
        self.batch.reset(states=states)
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.batch.finished[0]:
            raise RuntimeError("episode is over; call reset()")
        reward, done = self.batch.step(np.array([int(action)]))
        info = self._info()
        info["action_taken"] = ENDGAME_ACTIONS[int(self.batch.last["action"][0])]
        info["illegal_action"] = bool(self.batch.last["illegal"][0])
        if done[0]:
            info["result"] = float(self.batch.result[0])
        return self._obs(), float(reward[0]), bool(done[0]), False, info
