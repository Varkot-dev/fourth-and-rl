import time, numpy as np, lightgbm as lgb
from common import load, feats, slice_coords
from gridpred import GridPredictor, slice_values
sol, _ = load(); m = sol.model
gp = GridPredictor("proxy_16k_large.txt")
for t in (30, 200, 500):
    t0 = time.time(); v = slice_values(gp, m, t); dt = time.time() - t0
    shape, f, s, a, b, r = slice_coords(m, t)
    X = feats(m, np.full(shape, t), np.broadcast_to(s, shape), np.broadcast_to(f, shape), np.broadcast_to(a, shape), np.broadcast_to(b, shape), np.broadcast_to(r, shape))
    ref = gp.bst.predict(X, num_threads=1).reshape(shape)
    print(t, v.shape, f"grid {dt:.2f}s", "maxabs diff", float(np.abs(v - ref).max()))
