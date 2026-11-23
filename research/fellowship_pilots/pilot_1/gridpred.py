"""Exact evaluation of a LightGBM model on a product grid of states, by walking each tree
over index boxes instead of predicting row by row. Same numbers as Booster.predict."""
import numpy as np, lightgbm as lgb

class GridPredictor:
    def __init__(self, path):
        self.bst = lgb.Booster(model_file=path)
        dm = self.bst.dump_model()
        self.trees = [ti["tree_structure"] for ti in dm["tree_info"]]
        for tr in self.trees:
            self._check(tr)
    def _check(self, n):
        if "leaf_value" in n and "split_feature" not in n:
            return
        assert n["decision_type"] == "<=", n["decision_type"]
        self._check(n["left_child"]); self._check(n["right_child"])
    def raw(self, t, axes):
        """axes: list of 7 sorted 1-D arrays for features 1..7 (s, down, dbin, yl, a, b, r);
        feature 0 (t) is the scalar t. Returns raw scores of shape of the product grid."""
        shape = tuple(len(x) for x in axes)
        out = np.zeros(shape, np.float64)
        for tr in self.trees:
            stack = [(tr, [(0, n) for n in shape])]
            while stack:
                node, box = stack.pop()
                if "split_feature" not in node:
                    out[tuple(slice(a, b) for a, b in box)] += node["leaf_value"]
                    continue
                j, thr = node["split_feature"], node["threshold"]
                if j == 0:
                    stack.append((node["left_child"] if t <= thr else node["right_child"], box))
                    continue
                ax = axes[j - 1]; lo, hi = box[j - 1]
                cut = lo + int(np.searchsorted(ax[lo:hi], thr, side="right"))
                if cut > lo:
                    b2 = list(box); b2[j - 1] = (lo, cut); stack.append((node["left_child"], b2))
                if cut < hi:
                    b2 = list(box); b2[j - 1] = (cut, hi); stack.append((node["right_child"], b2))
        return out

def slice_values(gp, model, t):
    """Proxy win probability over the slice at tick t, shape (nf, S, E1, E2)."""
    g, c = model.kern.grid, model.clock; ph = model.phase(t); E = model.extra_shape(ph)
    s = np.arange(-model.score_max, model.score_max + 1, dtype=np.float64)
    down = np.arange(1, g.n_down + 1, dtype=np.float64); dbin = np.arange(g.n_dist, dtype=np.float64); yl = np.arange(1, g.n_yl + 1, dtype=np.float64)
    if ph == "end":
        a, b, r = np.arange(E[0], dtype=np.float64), np.arange(E[1], dtype=np.float64), np.array([0.0])
    elif ph == "first":
        a, b, r = np.array([3.0]), np.array([3.0]), np.arange(E[0], dtype=np.float64)
    else:
        a, b, r = np.array([3.0]), np.array([3.0]), np.array([0.0])
    z = gp.raw(float(t), [s, down, dbin, yl, a, b, r])  # (S, 4, 12, 99, A, B, R)
    z = np.moveaxis(z, 0, 3).reshape(g.nf, model.n_score, len(a), len(b), len(r))
    if ph == "end":
        z = z[..., 0]
    elif ph == "first":
        z = z[:, :, 0, 0, :][..., None]
    else:
        z = z[:, :, 0, 0, :]  # (nf, S, 1) -> E (1, 1)
        z = z[..., None]
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
