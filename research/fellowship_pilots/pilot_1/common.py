"""Pilot 1 shared code: coach fourth-down probabilities, proxy features, ring-buffer
policy evaluation of the full game MDP (scratch; imports repo code, edits nothing)."""
import numpy as np, joblib
from fourth_and_rl import mdp_game as mg
from fourth_and_rl.mdp import coach_situation

HERE = __import__("pathlib").Path(__file__).parent
BETAS = [10.0, 30.0, 100.0, 300.0, 1000.0, 10000.0]

def load():
    sol = mg.load_solution(mmap=True)
    coach = joblib.load(mg.SOLUTION_DIR / "tables.joblib")["coach"]
    return sol, coach

def coach_fourth(model, coach, t, s, f):
    """Mirror of Simulator.coach_actions as probabilities. Arrays t, s, f broadcast.
    Returns base (mode code for go / ordinary plays), p (..., 3) over (go, fg, punt) at
    fourth downs, forced code (-1 if none), fourth mask."""
    g, c = model.kern.grid, model.clock
    t, s, f = np.broadcast_arrays(np.asarray(t), np.asarray(s), np.asarray(f))
    down, dbin, yl = g.unpack(f)
    sec = t * c.tick_seconds
    half_sec = np.where(t > c.th, (t - c.th) * c.tick_seconds, sec)
    end = (t <= c.endgame_ticks) & (c.endgame_ticks > 0)
    warn = c.warning_tick if c.warning_tick is not None else 0
    chase = (s < 0) | ((s == 0) & (t <= warn))
    base = np.where(end & chase, mg.A_CHASE, np.where(end & (s > 0), mg.A_PROTECT, mg.A_NEUTRAL))
    fourth = down == g.n_down
    sit = coach_situation(sec, half_sec, s)
    p = coach["table"][sit, dbin, yl - 1].astype(np.float64)
    pgo, pfg = p[..., 0], p[..., 1]
    ppu = 1.0 - pgo - pfg
    okf, okp = model.kern.fg_ok[f], model.kern.punt_ok[f]
    pgo = pgo + np.where(okf, 0, pfg) + np.where(okp, 0, ppu)
    pfg = np.where(okf, pfg, 0.0); ppu = np.where(okp, ppu, 0.0)
    P = np.stack([pgo, pfg, ppu], -1)
    forced = np.full(t.shape, -1, np.int64)
    cc = (coach.get("end_fg_max_yardline", mg.COACH_DEFAULTS["end_fg_max_yardline"]),
          coach.get("end_fg_ticks", mg.COACH_DEFAULTS["end_fg_ticks"]))
    in_range = (yl <= cc[0]) & okf
    last_game = (t <= cc[1]) & (s <= 0) & (s >= -model.fg_points)
    last_half = (t > c.th) & (t - c.th <= cc[1])
    forced = np.where((last_game | last_half) & in_range, mg.A_FG, forced)
    late4 = fourth & end & (t <= warn)
    forced = np.where(late4 & (s < -model.fg_points), mg.A_CHASE, forced)
    lm = late4 & (t <= warn // 2) & (s < 0) & (s >= -model.fg_points)
    forced = np.where(lm, np.where(in_range, mg.A_FG, mg.A_CHASE), forced)
    return base, P, forced, fourth

def tilt(P, Q, beta):
    """pi_beta proportional to P * exp(beta Q) over the last axis (Q -inf where P is 0)."""
    with np.errstate(invalid="ignore", over="ignore"):
        z = np.where(P > 0, beta * Q + np.log(np.maximum(P, 1e-300)), -np.inf)
        z = z - z.max(-1, keepdims=True)
        w = np.exp(z)
    return w / w.sum(-1, keepdims=True)

def kl(pi, P):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(pi > 0, pi * (np.log(pi) - np.log(np.maximum(P, 1e-300))), 0.0)
    return x.sum(-1)

def feats(model, t, s, f, a, b, r):
    """Proxy features from the side of the team with the ball."""
    g, c = model.kern.grid, model.clock
    t, s, f, a, b, r = np.broadcast_arrays(*(np.asarray(x) for x in (t, s, f, a, b, r)))
    down, dbin, yl = g.unpack(f)
    end = t <= c.endgame_ticks
    a = np.where(end, a, 3); b = np.where(end, b, 3); r = np.where(t > c.th, r, 0)
    return np.stack([t, s, down, dbin, yl, a, b, r], -1).reshape(-1, 8).astype(np.float32)

def slice_coords(model, t):
    """Broadcastable (f, s, e1, e2) coordinates of a slice at tick t."""
    g = model.kern.grid; ph = model.phase(t); E = model.extra_shape(ph)
    f = np.arange(g.nf)[:, None, None, None]
    s = (np.arange(model.n_score) - model.score_max)[None, :, None, None]
    e1 = np.arange(E[0])[None, None, :, None]; e2 = np.arange(E[1])[None, None, None, :]
    shape = (g.nf, model.n_score, *E)
    if ph == "end":
        a, b, r = e1, e2, 0
    elif ph == "first":
        a, b, r = 3, 3, e1
    else:
        a, b, r = 3, 3, 0
    return shape, f, s, a, b, r

class RingSol(mg.Solution):
    """Solution whose values live in a dict tick -> slice, keeping only recent ticks
    (plus halftime and the last tick), so many policy evaluations fit in memory."""
    KEEP = 12
    def __init__(self, model):
        mg.Solution.__init__(self, model, v={}, pol={})
        self.buf = {}
        ph0 = model.phase(0); E = model.extra_shape(ph0)
        self.buf[0] = np.broadcast_to(self.terminal[None, :, None, None], (self.grid.nf, model.n_score, *E)).astype(model.dtype)
    def native(self, t):
        return self.buf[t]
    def put(self, t, v):
        self.buf[t] = np.ascontiguousarray(v, dtype=self.model.dtype)
        old = t - self.KEEP
        th = self.model.clock.th
        if old >= 0 and old != th and old in self.buf:
            del self.buf[old]
            for cache in (self._posts, self._bases):
                for key in [k for k in cache if k[0] == old]:
                    del cache[key]
        self._q_cache = None

def mix(q, base, Pi, fourth, forced, shape):
    """Value of a stochastic policy at a slice from its Q dict. Pi (..., 3) over
    (go-in-base-mode, fg, punt) at fourth downs; elsewhere the base mode; forced overrides."""
    def qa(a):
        return q[a] if a in q else np.full(shape, -np.inf, np.float32)
    qbase = np.where(base == mg.A_NEUTRAL, qa(mg.A_NEUTRAL), np.where(base == mg.A_PROTECT, qa(mg.A_PROTECT), qa(mg.A_CHASE)))
    qf, qp = qa(mg.A_FG), qa(mg.A_PUNT)
    with np.errstate(invalid="ignore"):
        v4 = (np.where(Pi[..., 0] > 0, Pi[..., 0] * qbase, 0) + np.where(Pi[..., 1] > 0, Pi[..., 1] * qf, 0)
              + np.where(Pi[..., 2] > 0, Pi[..., 2] * qp, 0))
    v = np.where(fourth, v4, qbase)
    v = np.where(forced == mg.A_FG, qf, np.where(forced == mg.A_CHASE, qa(mg.A_CHASE), v))
    if not np.isfinite(v).all():
        raise RuntimeError("policy put mass on a disallowed action")
    return v.astype(np.float32), qbase, qf, qp

def start_value(A, B, model):
    """Win probability of team A at kickoff, coin flip for the opening kickoff."""
    T = model.clock.n_ticks
    koA = A._kicks(A.native(T))[0]; koB = B._kicks(B.native(T))[0]
    s0 = model.score_max
    return 0.5 * float(koA[s0, 0, 0]) + 0.5 * (1.0 - float(koB[s0, 0, 0]))
