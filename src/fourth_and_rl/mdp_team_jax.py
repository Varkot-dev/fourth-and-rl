"""The end phase of the two-team game on a GPU, with JAX, for many matchups at once.

`mdp_team.solve_matchup` solves the two-team game on the CPU, slice by slice, with
`mdp_game.Solution.backup`. This module runs exactly the same Bellman backup for the END
PHASE (the last `endgame_ticks`, where timeouts and play modes are in the state) as array
code that JAX compiles for a GPU. The end phase never reads later times: its only boundary
is the terminal slice at t = 0 (1, 0.5 or 0 by the score), so the inputs are just the two
teams' end-phase models (`mdp_team.endgame_model` of a model whose kernels come from
`mdp_team.matchup_kernels`). A whole-game model is accepted and cut to its end phase.

One tick of the backup for one team X (the other team is Y), as in `Solution.backup`:

    aug(t')      outcome values at t' (X keeps the ball; Y takes over, read from Y's
                 array; X scores, valued with Y's kickoff value; Y returns it for a
                 touchdown, valued with X's kickoff value)
    U[mode]      value of an in-bounds play with the clock running: a tempo-weighted sum
                 of up to a dozen earlier slices, then max with X's own timeout and min
                 with Y's timeout (or the empirical timeout rates)
    Q[mode]      K_all[mode] @ aug(t - 1) + K_in[mode] @ (U[mode] - V(t - 1))
    Q[fg]        make * value after a make + (1 - make) * aug(t - 1)[miss column]
    Q[punt]      K_punt @ aug(t - punt_ticks)
    Q[kneel]     U[protect] or aug(t - 1) at the kneel column
    V(t), policy max and first argmax over the actions in the CPU order, then the kneel
                 rule (game over) where it applies.

Layout. Arrays carry a leading matchup axis b and a team axis of size 2 (A, B), so the
other team's values are the same array flipped on the team axis, and both teams' products
run in one batched call. Only the last R slices of V live on the device (a ring buffer,
R = longest look-back + 1); each solved slice is copied back to the host only if asked.

Products. `backend="dense"` stores every kernel as a dense float32 matrix and uses one
batched matmul at full float32 precision (`lax.Precision.HIGHEST`, so GPUs that would
otherwise use TF32 do not lose accuracy). `backend="bcoo"` keeps the kernels sparse
(`jax.experimental.sparse.BCOO`, about 115 entries per row) and loops over teams.
`benchmark_backends` times both on the current device; `backend="auto"` picks the faster.
The JAX documentation marks `jax.experimental.sparse` as experimental and no longer
actively developed, which is one more reason to measure rather than assume.

All arithmetic is float32 (JAX's default), whatever the model's dtype. Results agree with
the CPU solver to a few millionths (tests/test_mdp_team_jax.py checks 1e-5). JAX is an optional
dependency: `pip install -e ".[jax]"` (CPU) or `jax[cuda12]` on a GPU machine.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fourth_and_rl import mdp_game as mg
from fourth_and_rl import mdp_team as mt

try:  # optional dependency
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import sparse as jsparse

    HAVE_JAX = True
except ImportError:  # pragma: no cover - exercised only without jax
    HAVE_JAX = False

# The CPU solver's action order (`Solution._running_values` inserts protect, neutral,
# chase; then field goal, punt, kneel). Ties go to the earlier action, as on the CPU.
MODES = ("protect", "neutral", "chase")
MODE_CODES = (mg.A_PROTECT, mg.A_NEUTRAL, mg.A_CHASE)
END_KEYS = ("neutral_end", "protect", "chase", "chase_late")
BACKENDS = ("dense", "bcoo", "bcoo_cusparse")


def _require_jax() -> None:
    if not HAVE_JAX:
        raise ImportError(
            "mdp_team_jax needs JAX: pip install -e '.[jax]' (or jax[cuda12] on a GPU)"
        )


# --------------------------------------------------------------------------------------
# Host-side preparation: everything that does not depend on the values
# --------------------------------------------------------------------------------------


@dataclass
class Plan:
    """The clock schedule of the end phase, the same for every matchup: which ring slots
    each tick reads and with which weights (`Solution._running_values`, `next_tick`)."""

    n_ticks: int
    ring: int
    slot_v1: np.ndarray  # (T + 1,) slot of t - 1
    slot_protect: np.ndarray  # (T + 1,) slot of next_tick(t, burn)
    slot_punt: np.ndarray  # (T + 1,) slot of max(t - punt_ticks, 0)
    w_neutral: np.ndarray  # (T + 1, R) tempo weights summed per slot
    w_chase: np.ndarray  # (T + 1, R)
    late: np.ndarray  # (T + 1,) bool: inside the two-minute warning
    q_before: np.ndarray  # (S, E) defense timeout rates (empirical mode)
    q_after: np.ndarray
    empirical: bool
    over_thr: np.ndarray | None  # (nf, E) kneel threshold in ticks, or None


def _end_model(model: mg.GameModel) -> mg.GameModel:
    c = model.clock
    if c.endgame_ticks <= 0:
        raise ValueError("the model has no end phase (clock.endgame_ticks is 0)")
    if c.n_ticks == c.endgame_ticks and c.half_tick is None:
        return model
    return mt.endgame_model(model)


def make_plan(model: mg.GameModel) -> Plan:
    m = _end_model(model)
    c = m.clock
    sol = mg.Solution.__new__(mg.Solution)  # only `next_tick` is needed (reads model.clock)
    sol.model = m
    te = c.endgame_ticks
    hists = [c.neutral_hist, c.hurry_hist]
    if c.hurry_hist_late is not None:
        hists.append(c.hurry_hist_late)
    look_back = max(c.burn_ticks, c.punt_ticks, 1, *(len(h) for h in hists))
    r = look_back + 1
    slot_v1 = np.zeros(te + 1, dtype=np.int32)
    slot_protect = np.zeros(te + 1, dtype=np.int32)
    slot_punt = np.zeros(te + 1, dtype=np.int32)
    w_n = np.zeros((te + 1, r), dtype=np.float32)
    w_c = np.zeros((te + 1, r), dtype=np.float32)
    late = np.zeros(te + 1, dtype=bool)
    for t in range(1, te + 1):
        slot_v1[t] = (t - 1) % r
        slot_protect[t] = sol.next_tick(t, c.burn_ticks) % r
        slot_punt[t] = max(t - c.punt_ticks, 0) % r
        late[t] = c.late(t)
        for w, mode in ((w_n, "neutral"), (w_c, "chase")):
            for h, p in enumerate(c.tempo_hist(mode, t), start=1):
                if p > 0:
                    w[t, sol.next_tick(t, h) % r] += np.float32(p)
    n = m.extra_shape("end")[1]
    scores = np.arange(-m.score_max, m.score_max + 1)
    q = {}
    for name, rate in (("before", c.def_rate_before), ("after", c.def_rate_after)):
        arr = np.zeros((m.n_score, n), dtype=np.float32)
        if rate is not None:
            arr[scores > 0] = np.asarray(rate, dtype=np.float32)[:n]
        q[name] = arr
    thr = None
    if c.kneel_ticks is not None:
        downs = m.kern.grid.unpack(np.arange(m.kern.grid.nf))[0]
        thr = np.asarray(c.kneel_ticks[:, downs - 1].T, dtype=np.int32)  # (nf, E)
    return Plan(te, r, slot_v1, slot_protect, slot_punt, w_n, w_c, late, q["before"],
                q["after"], c.defense_timeout != "optimal", thr)  # fmt: skip


def _dense(x: Any) -> np.ndarray:
    return np.asarray(x.toarray(), dtype=np.float32)


@dataclass
class TeamArrays:
    """One team's end-phase kernels and action tables, as host arrays."""

    k_all: dict[str, Any]  # key -> scipy CSR (float32)
    k_in: dict[str, Any]
    k_punt: Any
    fg_make: np.ndarray
    fg_miss_col: np.ndarray
    fg_ok: np.ndarray
    punt_ok: np.ndarray
    kneel_col: np.ndarray
    kneel_keep: np.ndarray
    kneel_ok: np.ndarray

    @classmethod
    def from_model(cls, model: mg.GameModel) -> TeamArrays:
        k = model.kern
        f32 = np.float32
        return cls(
            k_all={key: k.k_all[key].astype(f32).tocsr() for key in END_KEYS},
            k_in={key: k.k_in[key].astype(f32).tocsr() for key in END_KEYS},
            k_punt=k.k_punt.astype(f32).tocsr(),
            fg_make=np.asarray(k.fg_make, dtype=f32),
            fg_miss_col=np.asarray(k.fg_miss_col, dtype=np.int32),
            fg_ok=np.asarray(k.fg_ok, dtype=bool),
            punt_ok=np.asarray(k.punt_ok, dtype=bool),
            kneel_col=np.asarray(k.kneel_col, dtype=np.int32),
            kneel_keep=np.asarray(k.kneel_keep, dtype=bool),
            kneel_ok=np.asarray(k.kneel_ok, dtype=bool),
        )


def _check_pair(a: mg.GameModel, b: mg.GameModel, ref: mg.GameModel | None) -> None:
    mt.check_compatible(a, b)
    if ref is not None:
        mt.check_compatible(ref, a)  # every matchup in one batch plays the same game


# --------------------------------------------------------------------------------------
# Device-side backup
# --------------------------------------------------------------------------------------


@dataclass
class _Static:
    """Shapes and constants baked into the compiled step."""

    nf: int
    n_yl: int
    n_score: int
    n_to: int
    f10: tuple[int, ...]
    td_points: int
    fg_points: int
    safety_points: int
    p_xp: float
    p_two: float
    two_point_choice: bool
    has_fg: bool
    has_punt: bool
    has_kneel: bool
    has_over: bool
    empirical: bool
    backend: str
    kick_ko: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    kick_fk: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))


def _static(model: mg.GameModel, plan: Plan, teams: Sequence[TeamArrays], backend: str
            ) -> _Static:  # fmt: skip
    g = model.kern.grid
    return _Static(
        nf=g.nf,
        n_yl=g.n_yl,
        n_score=model.n_score,
        n_to=model.extra_shape("end")[0],
        f10=tuple(int(x) for x in g.first10()[1:]),
        td_points=model.td_points,
        fg_points=model.fg_points,
        safety_points=model.safety_points,
        p_xp=float(model.p_xp),
        p_two=float(model.p_two),
        two_point_choice=bool(model.two_point_choice),
        has_fg=any(t.fg_ok.any() for t in teams),
        has_punt=any(t.punt_ok.any() for t in teams),
        has_kneel=any(t.kneel_ok.any() for t in teams),
        has_over=plan.over_thr is not None,
        empirical=plan.empirical,
        backend=backend,
        kick_ko=np.asarray(model.kern.kickoff[1:], dtype=np.float32),
        kick_fk=np.asarray(model.kern.free_kick[1:], dtype=np.float32),
    )


def _opp(x: Any) -> Any:
    """`Solution.opp` for the end phase: values from the other side, (..., S, E1, E2)."""
    return jnp.swapaxes(1.0 - x[..., ::-1, :, :], -1, -2)


def _shift(n: int, pts: int) -> np.ndarray:
    return np.clip(np.arange(n) + pts, 0, n - 1)


def _hi(spec: str, *ops: Any) -> Any:
    return jnp.einsum(spec, *ops, precision=lax.Precision.HIGHEST)


def _aug_and_fg(st: _Static, v: Any) -> tuple[Any, Any]:
    """aug (b, 2, na, S, E, E) and the value after a made field goal (b, 2, S, E, E) for
    both teams at one moment, from both teams' values v (b, 2, nf, S, E, E). Team axis
    flipped = the other team (`Solution._derive`, `_aug` with `other`)."""
    f10 = np.asarray(st.f10)
    vf = v[:, :, f10]  # (b, 2, n_yl, S, E, E)
    ko = _hi("y,ntysxz->ntsxz", jnp.asarray(st.kick_ko), vf)
    fk = _hi("y,ntysxz->ntsxz", jnp.asarray(st.kick_fk), vf)
    opp_ko = _opp(ko[:, ::-1])
    s = st.n_score
    td0 = opp_ko[:, :, _shift(s, st.td_points)]
    xp = st.p_xp * opp_ko[:, :, _shift(s, st.td_points + 1)] + (1.0 - st.p_xp) * td0
    two = st.p_two * opp_ko[:, :, _shift(s, st.td_points + 2)] + (1.0 - st.p_two) * td0
    td = jnp.where(two > xp, two, xp) if st.two_point_choice else xp
    fg = opp_ko[:, :, _shift(s, st.fg_points)]
    safety = _opp(fk[:, ::-1])[:, :, _shift(s, -st.safety_points)]
    turn = _opp(vf[:, ::-1])
    opp_td = _opp(td[:, ::-1])
    aug = jnp.concatenate(
        [v, turn, td[:, :, None], safety[:, :, None], opp_td[:, :, None]], axis=2
    )
    return aug, fg


def _matmul(st: _Static, k: Any, w: Any) -> Any:
    """k: dense (b, 2, n, m) or nested lists of BCOO [b][team]; w: (b, 2, m, C)."""
    if st.backend == "dense":
        return _hi("btij,btjc->btic", k, w)
    return jnp.stack(
        [jnp.stack([k[b][t] @ w[b, t] for t in range(2)]) for b in range(len(k))]
    )


def _step(st: _Static, late: bool, return_q: bool, ring: Any, kern: dict[str, Any],
          tab: dict[str, Any], t: Any, sched: dict[str, Any]) -> tuple[Any, ...]:  # fmt: skip
    """One tick for every team of every matchup. Returns (ring, v, pol[, q])."""
    nb = ring.shape[0]
    nf, s, e = st.nf, st.n_score, st.n_to
    c = s * e * e
    v1 = ring[:, :, sched["slot_v1"]]
    aug1, fg1 = _aug_and_fg(st, v1)
    na = aug1.shape[2]
    aug1f = aug1.reshape(nb, 2, na, c)
    v1f = v1.reshape(nb, 2, nf, c)
    # U: value of an in-bounds play with the clock running, before the timeouts.
    us = {"protect": ring[:, :, sched["slot_protect"]]}
    us["neutral"] = _hi("r,ntr...->nt...", sched["w_neutral"], ring)
    us["chase"] = _hi("r,ntr...->nt...", sched["w_chase"], ring)
    for mode in ("neutral", "chase"):  # the offense may spend its own timeout instead
        u = us[mode]
        us[mode] = u.at[..., 1:, :].set(jnp.maximum(u[..., 1:, :], v1[..., :-1, :]))
    for mode, u in us.items():
        if st.empirical:
            q = sched["q_def"][:, None, 1:]
            us[mode] = u.at[..., 1:].set(q * v1[..., :-1] + (1.0 - q) * u[..., 1:])
        else:
            us[mode] = u.at[..., 1:].set(jnp.minimum(u[..., 1:], v1[..., :-1]))
    scores = np.arange(-(s // 2), s // 2 + 1)
    lead = np.broadcast_to((scores > 0)[:, None, None], (s, e, e)).reshape(-1)
    trail = np.broadcast_to((scores < 0)[:, None, None], (s, e, e)).reshape(-1)
    neg = jnp.float32(-jnp.inf)
    qs: list[Any] = []
    codes: list[int] = []
    for mode, code in zip(MODES, MODE_CODES, strict=True):
        key = {"neutral": "neutral_end", "protect": "protect",
               "chase": "chase_late" if late else "chase"}[mode]  # fmt: skip
        um = us[mode].reshape(nb, 2, nf, c)
        qm = _matmul(st, kern["all_" + key], aug1f) + _matmul(st, kern["in_" + key], um - v1f)
        if mode == "protect":
            qm = jnp.where(jnp.asarray(trail), neg, qm)
        if mode == "chase":
            qm = jnp.where(jnp.asarray(lead), neg, qm)
        qs.append(qm)
        codes.append(code)
    if st.has_fg:
        make = tab["fg_make"][..., None]  # (b, 2, nf, 1)
        miss = jnp.take_along_axis(aug1f, tab["fg_miss_col"][..., None], axis=2)
        qf = make * fg1.reshape(nb, 2, 1, c) + (1.0 - make) * miss
        qs.append(jnp.where(tab["fg_ok"][..., None], qf, neg))
        codes.append(mg.A_FG)
    if st.has_punt:
        augp, _ = _aug_and_fg(st, ring[:, :, sched["slot_punt"]])
        qp = _matmul(st, kern["punt"], augp.reshape(nb, 2, na, c))
        qs.append(jnp.where(tab["punt_ok"][..., None], qp, neg))
        codes.append(mg.A_PUNT)
    if st.has_kneel:
        keep = tab["kneel_keep"][..., None]
        up = us["protect"].reshape(nb, 2, nf, c)
        col = tab["kneel_col"]
        from_u = jnp.take_along_axis(up, jnp.where(tab["kneel_keep"], col, 0)[..., None], axis=2)
        from_a = jnp.take_along_axis(aug1f, jnp.where(tab["kneel_keep"], 0, col)[..., None],
                                     axis=2)  # fmt: skip
        qk = jnp.where(keep, from_u, from_a)
        qs.append(jnp.where(tab["kneel_ok"][..., None], qk, neg))
        codes.append(mg.A_KNEEL)
    v = qs[0]
    pol = jnp.full(v.shape, codes[0], dtype=jnp.uint8)
    for qa, code in zip(qs[1:], codes[1:], strict=True):
        better = qa > v
        v = jnp.where(better, qa, v)
        pol = jnp.where(better, jnp.uint8(code), pol)
    shape = (nb, 2, nf, s, e, e)
    v, pol = v.reshape(shape), pol.reshape(shape)
    if st.has_over:
        leading = jnp.asarray(scores > 0)[None, :, None, None]
        over = (t <= tab["over_thr"])[:, None, None, :] & leading
        v = jnp.where(over, jnp.float32(1.0), v)
        pol = jnp.where(over, jnp.uint8(mg.A_OVER), pol)
    ring = ring.at[:, :, sched["slot_t"]].set(v)
    if return_q:
        return ring, v, pol, {code: qa.reshape(shape) for code, qa in zip(codes, qs, strict=True)}
    return ring, v, pol


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


@dataclass
class EndPhaseResult:
    """One matchup's end-phase solution.

    v, pol: (2, T + 1, nf, S, E, E) for teams A and B (None unless `keep_arrays`), in the
    layout of `Solution.v["end"]`. q: {tick: {action code: (2, nf, S, E, E)}} for the
    ticks asked for in `q_ticks` (the action values `Solution.q_slice` would give)."""

    models: tuple[mg.GameModel, mg.GameModel]
    v: np.ndarray | None
    pol: np.ndarray | None
    q: dict[int, dict[int, np.ndarray]]
    meta: dict[str, Any]

    def matchup(self) -> mt.Matchup:
        """The result as an `mdp_team.Matchup` (linked `Solution`s), so every existing
        query works: value, policy, q_values, play_split, the simulator."""
        if self.v is None or self.pol is None:
            raise ValueError("solved with keep_arrays=False: no value arrays to wrap")
        sols = []
        for k in range(2):
            sol = mg.Solution(self.models[k], {"end": self.v[k]}, {"end": self.pol[k]},
                              dict(self.meta))  # fmt: skip
            sols.append(sol)
        mt.link(sols[0], sols[1])
        return mt.Matchup((sols[0], sols[1]), dict(self.meta))


def _device_kernels(teams_batch: Sequence[Sequence[TeamArrays]], backend: str) -> dict[str, Any]:
    names = [("all_" + k, "k_all", k) for k in END_KEYS]
    names += [("in_" + k, "k_in", k) for k in END_KEYS]
    names.append(("punt", "k_punt", None))
    out: dict[str, Any] = {}
    for name, attr, key in names:
        def get(ta: TeamArrays, attr: str = attr, key: str | None = key) -> Any:
            x = getattr(ta, attr)
            return x[key] if key is not None else x

        if backend == "dense":
            out[name] = jnp.asarray(np.stack([np.stack([_dense(get(ta)) for ta in pair])
                                              for pair in teams_batch]))  # fmt: skip
        else:
            out[name] = [[_bcoo(get(ta)) for ta in pair] for pair in teams_batch]
    return out


def _bcoo(m: Any) -> Any:
    """BCOO with row-major sorted, unique indices (a canonical CSR matrix), flagged as such
    so the cuSPARSE lowering, if enabled, accepts it. `BCOO.from_scipy_sparse` marks its
    result unsorted, so the BCOO is built from the CSR arrays directly."""
    m = m.tocsr().copy()
    m.sum_duplicates()
    m.sort_indices()
    rows = np.repeat(np.arange(m.shape[0], dtype=np.int32), np.diff(m.indptr))
    idx = np.column_stack([rows, m.indices.astype(np.int32)])
    return jsparse.BCOO((jnp.asarray(m.data, dtype=jnp.float32), jnp.asarray(idx)),
                        shape=m.shape, indices_sorted=True, unique_indices=True)


def _device_tables(teams_batch: Sequence[Sequence[TeamArrays]], plan: Plan) -> dict[str, Any]:
    out = {}
    for name in ("fg_make", "fg_miss_col", "fg_ok", "punt_ok", "kneel_col", "kneel_keep",
                 "kneel_ok"):  # fmt: skip
        out[name] = jnp.asarray(np.stack([np.stack([getattr(ta, name) for ta in pair])
                                          for pair in teams_batch]))  # fmt: skip
    if plan.over_thr is not None:
        out["over_thr"] = jnp.asarray(plan.over_thr)
    return out


_COMPILED: dict[tuple[Any, ...], Any] = {}


def _compiled_step(st: _Static, late: bool, return_q: bool) -> Any:
    key = (repr(st), st.kick_ko.tobytes(), st.kick_fk.tobytes(), late, return_q)
    if key not in _COMPILED:
        def f(ring: Any, kern: Any, tab: Any, t: Any, sched: Any) -> Any:
            return _step(st, late, return_q, ring, kern, tab, t, sched)

        _COMPILED[key] = jax.jit(f, donate_argnums=(0,))
    return _COMPILED[key]


def _solve_batch(pairs: Sequence[tuple[mg.GameModel, mg.GameModel]], plan: Plan, backend: str,
                 q_ticks: Sequence[int], keep_arrays: bool, progress: bool
                 ) -> list[EndPhaseResult]:  # fmt: skip
    if backend == "bcoo_cusparse":
        # cuSPARSE lowering of BCOO products is off by default in JAX; it is read when the
        # step is compiled, so it is switched on for this batch only.
        old = bool(jax.config.jax_bcoo_cusparse_lowering)
        jax.config.update("jax_bcoo_cusparse_lowering", True)
        try:
            return _solve_batch(pairs, plan, "_bcoo_cusparse", q_ticks, keep_arrays, progress)
        finally:
            jax.config.update("jax_bcoo_cusparse_lowering", old)
    t_start = time.time()
    ref = pairs[0][0]
    teams = [[TeamArrays.from_model(a), TeamArrays.from_model(b)] for a, b in pairs]
    backend = backend.lstrip("_")
    st = _static(ref, plan, [t for p in teams for t in p], backend)
    kern = _device_kernels(teams, backend)
    tab = _device_tables(teams, plan)
    nb, te, r = len(pairs), plan.n_ticks, plan.ring
    shape = (st.nf, st.n_score, st.n_to, st.n_to)
    terminal = np.where(np.arange(-ref.score_max, ref.score_max + 1) > 0, 1.0,
                        np.where(np.arange(-ref.score_max, ref.score_max + 1) < 0, 0.0, 0.5))
    v0 = np.broadcast_to(terminal.astype(np.float32)[None, :, None, None], shape)
    ring_np = np.zeros((nb, 2, r, *shape), dtype=np.float32)
    ring_np[:, :, 0] = v0
    ring = jnp.asarray(ring_np)
    del ring_np
    t_setup = time.time() - t_start
    v_out = pol_out = None
    if keep_arrays:
        v_out = np.empty((nb, 2, te + 1, *shape), dtype=np.float32)
        pol_out = np.empty((nb, 2, te + 1, *shape), dtype=np.uint8)
        v_out[:, :, 0] = v0
        pol_out[:, :, 0] = mg.A_OVER
    q_out: list[dict[int, dict[int, np.ndarray]]] = [{} for _ in range(nb)]
    want_q = set(int(x) for x in q_ticks)
    t0 = time.time()
    for t in range(1, te + 1):
        late = bool(plan.late[t])
        sched = {
            "slot_v1": jnp.int32(plan.slot_v1[t]),
            "slot_protect": jnp.int32(plan.slot_protect[t]),
            "slot_punt": jnp.int32(plan.slot_punt[t]),
            "slot_t": jnp.int32(t % r),
            "w_neutral": jnp.asarray(plan.w_neutral[t]),
            "w_chase": jnp.asarray(plan.w_chase[t]),
            "q_def": jnp.asarray(plan.q_after if late else plan.q_before),
        }
        out = _compiled_step(st, late, t in want_q)(ring, kern, tab, jnp.int32(t), sched)
        ring = out[0]
        if keep_arrays:
            v_out[:, :, t] = np.asarray(out[1])
            pol_out[:, :, t] = np.asarray(out[2])
        if t in want_q:
            qd = {code: np.asarray(x) for code, x in out[3].items()}
            for b in range(nb):
                q_out[b][t] = {code: x[b] for code, x in qd.items()}
        if progress and (t % 20 == 0 or t == te):
            print(f"  t = {t:3d} / {te}   {time.time() - t0:6.1f} s", flush=True)
    ring.block_until_ready()
    seconds = time.time() - t0
    meta = {
        "backend": backend,
        "device": str(jax.devices()[0]),
        "batch_size": nb,
        "setup_seconds": t_setup,
        "solve_seconds_batch": seconds,
        "solve_seconds_per_matchup": seconds / nb,
        "dtype": "float32",
    }
    res = []
    for b, (a, bm) in enumerate(pairs):
        res.append(EndPhaseResult(
            models=(a, bm),
            v=None if v_out is None else v_out[b],
            pol=None if pol_out is None else pol_out[b],
            q=q_out[b],
            meta=dict(meta),
        ))  # fmt: skip
    return res


def solve_end_phases(
    pairs: Sequence[tuple[mg.GameModel, mg.GameModel]],
    backend: str = "dense",
    batch_size: int = 1,
    q_ticks: Sequence[int] = (),
    keep_arrays: bool = True,
    progress: bool = False,
) -> list[EndPhaseResult]:
    """Solve the end phase of every (model of A, model of B) pair, `batch_size` matchups
    at a time on one device. Each model is cut to its end phase if needed; all pairs must
    play the same game (grid, clock, scoring, kicks: `mdp_team.check_compatible`).

    `q_ticks`: ticks at which to return every action's values (for go, punt and field-goal
    maps without recomputing on the CPU). `keep_arrays=False` returns no value or policy
    arrays (saves host memory when only `q_ticks` are needed). `backend`: "dense", "bcoo"
    or "auto" (time both on the first pair, then use the faster)."""
    _require_jax()
    if not pairs:
        return []
    ends = [(_end_model(a), _end_model(b)) for a, b in pairs]
    ref = ends[0][0]
    for a, b in ends:
        _check_pair(a, b, ref)
    plan = make_plan(ref)
    if backend == "auto":
        backend = benchmark_backends(ends[0])["choice"]
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS} or 'auto', not {backend!r}")
    out: list[EndPhaseResult] = []
    for i in range(0, len(ends), max(int(batch_size), 1)):
        out += _solve_batch(ends[i : i + batch_size], plan, backend, q_ticks, keep_arrays,
                            progress)  # fmt: skip
    return out


def solve_matchup(model_a: mg.GameModel, model_b: mg.GameModel, backend: str = "dense",
                  progress: bool = False) -> mt.Matchup:  # fmt: skip
    """The JAX counterpart of `mdp_team.solve_matchup(endgame_model(a), endgame_model(b))`:
    a `Matchup` over the end phase only."""
    return solve_end_phases([(model_a, model_b)], backend=backend, progress=progress)[0].matchup()


def warmup(pairs: Sequence[tuple[mg.GameModel, mg.GameModel]], backend: str = "dense",
           q: bool = False) -> float:
    """Compile the step for these shapes (both clock variants, inside and outside the
    two-minute warning) by running one tick of each. Returns the seconds taken."""
    _require_jax()
    t0 = time.time()
    ends = [(_end_model(a), _end_model(b)) for a, b in pairs]
    plan = make_plan(ends[0][0])
    for late in (True, False):
        flags = plan.late.copy()
        flags[1] = late
        one = Plan(**{**plan.__dict__, "n_ticks": 1, "late": flags})
        _solve_batch(ends, one, backend, (1,) if q else (), False, False)
    return time.time() - t0


def benchmark_backends(pair: tuple[mg.GameModel, mg.GameModel], ticks: int = 8,
                       backends: Sequence[str] = BACKENDS) -> dict[str, Any]:  # fmt: skip
    """Seconds per tick of each backend on the current device, for one matchup (compile
    time excluded: the first `ticks` are a warm-up, the next `ticks` are timed). The two
    backends' values are also compared."""
    _require_jax()
    a, b = (_end_model(m) for m in pair)
    plan = make_plan(a)
    n = min(2 * ticks, plan.n_ticks)
    short = Plan(**{**plan.__dict__, "n_ticks": n})
    res: dict[str, Any] = {"device": str(jax.devices()[0]), "ticks_timed": n}
    vals = {}
    for name in backends:
        try:
            warm = Plan(**{**plan.__dict__, "n_ticks": max(n // 2, 1)})
            _solve_batch([(a, b)], warm, name, (), False, False)  # compiles
            t0 = time.time()
            r = _solve_batch([(a, b)], short, name, (), True, False)[0]
            total = time.time() - t0
        except Exception as e:  # noqa: BLE001 - e.g. out of device memory: record, go on
            res[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            continue
        timed = r.meta["solve_seconds_batch"]
        res[name] = {"seconds_per_tick": timed / n, "seconds_with_setup": total}
        vals[name] = r.v
    ok = [k for k in backends if "seconds_per_tick" in res[k]]
    if not ok:
        raise RuntimeError(f"no backend ran: {res}")
    res["max_abs_diff_vs_first"] = {k: float(np.abs(vals[k] - vals[ok[0]]).max()) for k in ok}
    res["choice"] = min(ok, key=lambda k: res[k]["seconds_per_tick"])
    return res


def device_memory_bytes() -> int | None:
    """Memory of the first device, if JAX reports it (GPUs do; CPUs may not)."""
    _require_jax()
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 - not every backend implements memory_stats
        return None
    return stats.get("bytes_limit")


def bytes_per_matchup(model: mg.GameModel, backend: str = "dense") -> int:
    """Rough device memory one matchup needs: kernels, ring buffer and temporaries."""
    m = _end_model(model)
    g = m.kern.grid
    e = m.extra_shape("end")[0]
    c = m.n_score * e * e
    plan = make_plan(m)
    if backend == "dense":
        kern = 4 * (len(END_KEYS) * (g.nf * g.na + g.nf * g.nf) + g.nf * g.na)
    else:
        nnz = sum(m.kern.k_all[k].nnz + m.kern.k_in[k].nnz for k in END_KEYS) + m.kern.k_punt.nnz
        kern = 8 * nnz
    ring = 4 * plan.ring * g.nf * c
    temps = 4 * 16 * g.na * c  # aug, U, Q per action, products
    return 2 * (kern + ring + temps)
