"""One backward pass over the full game: for one proxy value model, compute Q_proxy at every
fourth down (true dynamics, proxy values), and the EXACT win probability of team A playing
pi_beta at fourth downs against a coach team B, for every beta at once.
usage: eval_pass.py proxy (oracle | path to lightgbm txt) tag [--coach] [--br]"""
import sys, time, json, resource, numpy as np, joblib
from fourth_and_rl import mdp_game as mg
from common import load, coach_fourth, tilt, feats, slice_coords, RingSol, mix, start_value, BETAS, HERE
proxy_arg, tag = sys.argv[1], sys.argv[2]
flags = set(sys.argv[3:])
sol, coach = load(); model = sol.model; g, c = sol.grid, model.clock
F4 = np.nonzero(g.unpack(np.arange(g.nf))[0] == g.n_down)[0]; f4 = slice(F4[0], F4[-1] + 1)
assert len(F4) == F4[-1] - F4[0] + 1

proxy = RingSol(model)
if proxy_arg == "oracle":
    vc = joblib.load(HERE.parent / "v_coach.joblib", mmap_mode="r")
    def pv(t):
        ph = model.phase(t); return np.asarray(vc[ph][t - model.phase_range(ph)[0]])
else:
    from gridpred import GridPredictor, slice_values
    gp = GridPredictor(proxy_arg)
    def pv(t):
        return slice_values(gp, model, t)

pols = [("beta", b) for b in BETAS]
if "--coach" in flags: pols.insert(0, ("coach", 0.0))
if "--br" in flags: pols.append(("br", np.inf))
pairs = []
for kind, beta in pols:
    A, B = RingSol(model), RingSol(model)
    A._other, B._other = B, A
    pairs.append((kind, beta, A, B))
q4 = {ph: np.lib.format.open_memmap(HERE / f"q4_{tag}_{ph}.npy", mode="w+", dtype=np.float16,
       shape=(model.phase_range(ph)[1] - model.phase_range(ph)[0] + 1, len(F4), model.n_score, *model.extra_shape(ph), 3)) for ph in ("end", "mid", "first")}
t_pred = t_back = 0.0; T0 = time.time()
for t in range(1, c.n_ticks + 1):
    ph = model.phase(t); lo = model.phase_range(ph)[0]
    shape, f, s, *_ = slice_coords(model, t)
    base, P, forced, fourth = coach_fourth(model, coach, t, s, f)
    base, forced, fourth = (np.broadcast_to(x, shape) for x in (base, forced, fourth))
    P = np.broadcast_to(P, (*shape, 3))
    t1 = time.time()
    qp, _ = proxy.backup(t)
    def qa(q, a): return q[a] if a in q else np.full(shape, -np.inf, np.float32)
    qbase = np.where(base == mg.A_NEUTRAL, qa(qp, mg.A_NEUTRAL), np.where(base == mg.A_PROTECT, qa(qp, mg.A_PROTECT), qa(qp, mg.A_CHASE)))
    Q3 = np.stack([qbase, qa(qp, mg.A_FG), qa(qp, mg.A_PUNT)], -1)
    q4[ph][t - lo] = Q3[f4].astype(np.float16)
    for kind, beta, A, B in pairs:
        for side, S in (("A", A), ("B", B)):
            q, over = S.backup(t)
            if side == "B" or kind == "coach":
                Pi = P
            elif kind == "br":
                Qt = np.stack([np.where(base == mg.A_NEUTRAL, qa(q, mg.A_NEUTRAL), np.where(base == mg.A_PROTECT, qa(q, mg.A_PROTECT), qa(q, mg.A_CHASE))), qa(q, mg.A_FG), qa(q, mg.A_PUNT)], -1)
                Qt = np.where(P > 0, Qt, -np.inf)
                Pi = (Qt == Qt.max(-1, keepdims=True)).astype(np.float64); Pi /= Pi.sum(-1, keepdims=True)
            else:
                Pi = tilt(P, Q3, beta)
            v = mix(q, base, Pi, fourth, forced, shape)[0]
            if over is not None:
                v = np.where(over, 1.0, v).astype(np.float32)
            S.put(t, v)
    t2 = time.time()
    proxy.put(t, pv(t)); t3 = time.time()
    t_back += t2 - t1; t_pred += t3 - t2
    if t % 60 == 0: print(f"t={t} {time.time()-T0:.0f}s backup {t_back:.0f}s predict {t_pred:.0f}s", flush=True)
for a in q4.values(): a.flush()
out = dict(tag=tag, proxy=proxy_arg, seconds=round(time.time() - T0, 1), backup_s=round(t_back, 1), predict_s=round(t_pred, 1),
           peak_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
           start_value={f"{k}_{b}": start_value(A, B, model) for k, b, A, B in pairs},
           proxy_start_claim=start_value(proxy, proxy, model))
if "--coach" in flags:
    vc = joblib.load(HERE.parent / "v_coach.joblib", mmap_mode="r"); A = pairs[0][2]; T = c.n_ticks
    out["check_coach_vs_prototype_maxabs_lastslice"] = float(np.abs(A.native(T) - vc["first"][-1]).max())
json.dump(out, open(HERE / f"pass_{tag}.json", "w"), indent=1); print(json.dumps(out, indent=1))
