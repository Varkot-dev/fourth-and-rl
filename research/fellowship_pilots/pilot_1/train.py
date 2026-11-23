"""Train proxy win-probability models (LightGBM, cross-entropy on 0/0.5/1 labels).
usage: train.py data.npz size name"""
import sys, time, json, numpy as np, lightgbm as lgb
SIZES = {"small": dict(num_leaves=8, n_estimators=100, learning_rate=0.1, min_child_samples=50),
         "large": dict(num_leaves=64, n_estimators=150, learning_rate=0.1, min_child_samples=5)}
if __name__ == "__main__":
    data, size, name = sys.argv[1:4]
    d = np.load(data); v = np.load("dval.npz")
    m = lgb.LGBMRegressor(objective="cross_entropy", random_state=7, n_jobs=2, verbose=-1, **SIZES[size])
    t0 = time.time(); m.fit(d["X"], d["y"]); dt = time.time() - t0
    def ll(X, y):
        p = np.clip(m.predict(X), 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    m.booster_.save_model(f"proxy_{name}.txt")
    out = dict(name=name, data=data, size=size, params=SIZES[size], rows=int(len(d["y"])), fit_s=round(dt, 1),
               train_logloss=ll(d["X"], d["y"]), val_logloss=ll(v["X"], v["y"]))
    json.dump(out, open(f"proxy_{name}.json", "w"), indent=1); print(out)
