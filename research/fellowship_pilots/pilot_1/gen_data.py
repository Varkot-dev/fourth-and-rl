"""Synthetic games from the exact game model, coach policy for both teams, post='model'.
Records every pre-snap state and labels it with the final result for the team with the ball.
usage: gen_data.py N seed out.npz"""
import sys, time, numpy as np
from fourth_and_rl import mdp_game as mg
from common import load, feats

class Rec(mg.Simulator):
    def advance(self, st, idx, act, rng, post="model"):
        self.log.append(np.stack([idx, st["t"][idx], st["s"][idx], st["f"][idx], st["a"][idx], st["b"][idx], st["r"][idx], st["side"][idx]], 1))
        return super().advance(st, idx, act, rng, post)

def kickoff_start(sim, n, rng):
    k = sim.model.kern; p = k.kickoff[1:] / k.kickoff[1:].sum()
    f10 = sim.grid.first10()[1:]
    f = f10[rng.choice(len(p), size=n, p=p)]
    return sim.start(n, t=sim.model.clock.n_ticks, s=0, f=f, r=0)

if __name__ == "__main__":
    N, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    sol, coach = load(); sim = Rec(sol, coach=coach); sim.log = []
    rng = np.random.default_rng(seed); t0 = time.time()
    st = kickoff_start(sim, N, rng)
    res = mg.simulate(sim, st, rng, policy="coach", post="model")["result"]
    L = np.concatenate(sim.log)
    g, t, s, f, a, b, r, side = L.T
    y = np.where(side == 0, res[g], 1.0 - res[g])
    X = feats(sol.model, t, s, f, a, b, r)
    np.savez_compressed(out, X=X, y=y.astype(np.float32), game=g.astype(np.int32), seed=seed)
    print(f"N={N} seed={seed} rows={len(y)} secs={time.time()-t0:.1f} mean_result_side0={res.mean():.4f}")
