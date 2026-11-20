"""Prototype (scratch, not in repo): EXACT value of a fixed stochastic policy in the full game
MDP, by backward induction with sum_a pi(a|s) Q(s,a) in place of max_a. Both teams play pi
(symmetric game, so one array). Timeouts and try choices stay as the solver models them.
"""
import sys, time, resource
import numpy as np
import joblib
from fourth_and_rl import mdp_game as mg
from fourth_and_rl.mdp import coach_situation

class PolicyEval(mg.Solution):
    def __init__(self, model, probs_fn):
        super().__init__(model)
        self.probs_fn = probs_fn
    def solve_slice(self, t):
        q, over = self.backup(t)
        P = self.probs_fn(self, t)  # dict code -> prob array (nf, S) broadcastable
        v = None
        for a, pa in P.items():
            pa = np.asarray(pa, dtype=np.float32)
            if not pa.any():
                continue
            if a not in q:
                raise RuntimeError(f"policy puts mass on unavailable action {a} at t={t}")
            qa = q[a]
            pb = pa.reshape(pa.shape + (1,) * (qa.ndim - pa.ndim))
            bad = (pb > 0) & ~np.isfinite(qa)
            if bad.any():
                raise RuntimeError(f"policy puts mass on disallowed action {a} at t={t}")
            term = np.where(pb > 0, qa, 0.0) * pb
            v = term if v is None else v + term
        if over is not None:
            v = np.where(over, 1.0, v)
        ph = self.model.phase(t); lo = self.model.phase_range(ph)[0]
        self.v[ph][t - lo] = v

def coach_probs(coach):
    """Probability version of Simulator.coach_actions over a whole slice (nf, S)."""
    cc = (coach.get("end_fg_max_yardline", mg.COACH_DEFAULTS["end_fg_max_yardline"]),
          coach.get("end_fg_ticks", mg.COACH_DEFAULTS["end_fg_ticks"]))
    def fn(sol, t):
        m, c, g = sol.model, sol.model.clock, sol.grid
        f = np.arange(g.nf)[:, None]; s = sol.scores[None, :]
        F, Sg = np.broadcast_arrays(f, s)
        down, dbin, yl = g.unpack(F.ravel()); down, dbin, yl = (x.reshape(F.shape) for x in (down, dbin, yl))
        sec = t * c.tick_seconds
        half_sec = (t - c.th) * c.tick_seconds if t > c.th else sec
        end = (t <= c.endgame_ticks) and c.endgame_ticks > 0
        warn = c.warning_tick if c.warning_tick is not None else 0
        chase = (Sg < 0) | ((Sg == 0) & (t <= warn))
        base = np.where(end & chase, mg.A_CHASE, np.where(end & (Sg > 0), mg.A_PROTECT, mg.A_NEUTRAL))
        P = {a: np.zeros(F.shape, np.float32) for a in range(6)}
        for a in (mg.A_NEUTRAL, mg.A_PROTECT, mg.A_CHASE):
            P[a][base == a] = 1.0
        fourth = down == g.n_down
        sit = coach_situation(np.full(F.shape, sec), np.full(F.shape, half_sec), Sg)
        p = coach["table"][sit, dbin, yl - 1]  # (nf, S, 3) go, fg, punt
        pgo = p[..., 0]; pfg = p[..., 1]; ppu = 1.0 - pgo - pfg  # choice>=2 -> punt
        okf = m.kern.fg_ok[F]; okp = m.kern.punt_ok[F]
        pgo = pgo + np.where(okf, 0, pfg) + np.where(okp, 0, ppu)
        pfg = np.where(okf, pfg, 0); ppu = np.where(okp, ppu, 0)
        for a in (mg.A_NEUTRAL, mg.A_PROTECT, mg.A_CHASE):
            P[a] = np.where(fourth, P[a] * pgo, P[a])
        P[mg.A_FG] = np.where(fourth, pfg, 0).astype(np.float32)
        P[mg.A_PUNT] = np.where(fourth, ppu, 0).astype(np.float32)
        # deterministic overrides, in coach_actions order
        in_range = (yl <= cc[0]) & m.kern.fg_ok[F]
        last_game = (t <= cc[1]) & (Sg <= 0) & (Sg >= -m.fg_points)
        last_half = (t > c.th) and (t - c.th <= cc[1])
        def force(mask, a):
            for k in P: P[k] = np.where(mask, 0, P[k])
            P[a] = np.where(mask, 1.0, P[a]).astype(np.float32)
        force((last_game | last_half) & in_range, mg.A_FG)
        late4 = fourth & end & (t <= warn)
        force(late4 & (Sg < -m.fg_points), mg.A_CHASE)
        lm = late4 & (t <= warn // 2) & (Sg < 0) & (Sg >= -m.fg_points)
        force(lm & in_range, mg.A_FG); force(lm & ~in_range, mg.A_CHASE)
        return P
    return fn

if __name__ == "__main__":
    sol = mg.load_solution(mmap=True)
    coach = joblib.load(mg.SOLUTION_DIR / "tables.joblib")["coach"]
    pe = PolicyEval(sol.model, coach_probs(coach))
    t0 = time.time(); mg.backward_induction([pe]); dt = time.time() - t0
    print("exact coach-policy evaluation seconds", round(dt, 1), "peak GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
    np.savez_compressed(sys.argv[1] if len(sys.argv) > 1 else "v_coach_firstslices.npz", end_t240=pe.v["end"][48])
    for st in [mg.GameState(240, 4, 4, 2, 60), mg.GameState(2400, -3, 4, 4, 35), mg.GameState(1500, 0, 4, 1, 50), mg.GameState(3600, 0, 1, 10, 75, receives_second_half=1)]:
        ph, ix = pe.index(st)
        print(st, "V_coach", round(float(pe.v[ph][ix]), 4), "V*", round(sol.value(st), 4))
    # Q_coach(s,a) at a fourth down = one-step backup on V_coach
    st = mg.GameState(240, 4, 4, 2, 60); ph, ix = pe.index(st); t = ix[0] + pe.model.phase_range(ph)[0]
    q, _ = pe.backup(t)
    print("Q_coach at 4th&2 own 40 up 4 4:00:", {mg.ACTION_NAMES[a]: round(float(x[ix[1:]]), 4) for a, x in q.items() if np.isfinite(x[ix[1:]])})
    print("Q* same state:", {k: round(v, 4) for k, v in sol.q_values(st).items()})
    joblib.dump({ph: pe.v[ph] for ph in pe.v}, "/tmp/claude-0/-home-user-fourth-and-rl/f419b03a-4266-50b3-be51-e6683b77e914/scratchpad/fellowship/v_coach.joblib")
