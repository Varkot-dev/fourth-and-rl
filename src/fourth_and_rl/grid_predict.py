"""A LightGBM model evaluated on the whole state grid of one tick, without predicting row
by row.

A tree splits one feature at a time at a threshold (`x <= threshold` goes left). The
states of one tick form a product grid over the features (score, down, distance bin,
yard line, timeouts, second-half flag), so each tree can be walked over boxes of grid
indices instead of single rows: a split cuts a box in two along one axis, and a leaf adds
its value to every state of its box at once. Summing the trees in their own order gives
the same raw scores as `Booster.predict(raw_score=True)` up to floating-point addition
order, and the same probabilities after the sigmoid. One tick of the end phase has 3.3
million states, so this is what makes a learned value model affordable inside the exact
backward pass.

Features are `overopt.FEATURES` (t, s, down, dbin, yl, a, b, r), as `overopt.features`
builds them: timeouts are the full set before the last 5:00, the second-half flag is 0
after halftime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from fourth_and_rl import mdp_game as mg


class GridPredictor:
    """Raw scores of a LightGBM regressor (or binary model) on product grids."""

    def __init__(self, model: lgb.Booster | Path | str) -> None:
        self.bst = model if isinstance(model, lgb.Booster) else lgb.Booster(model_file=str(model))
        dm = self.bst.dump_model()
        if dm.get("num_class", 1) != 1:
            raise ValueError("only single-output models are supported")
        self.n_features = dm["max_feature_idx"] + 1
        self.trees = [ti["tree_structure"] for ti in dm["tree_info"]]
        for tree in self.trees:
            self._check(tree)
        self.objective = str(dm.get("objective", ""))

    def _check(self, node: dict[str, Any]) -> None:
        if "split_feature" not in node:
            return
        if node["decision_type"] != "<=":
            raise ValueError(f"unsupported split {node['decision_type']!r} (categorical?)")
        if node.get("missing_type", "None") == "Zero":
            raise ValueError("zero-as-missing splits are not supported")
        self._check(node["left_child"])
        self._check(node["right_child"])

    def raw(self, scalar: float, axes: list[np.ndarray]) -> np.ndarray:
        """Raw scores on the grid where feature 0 is the scalar `scalar` and features
        1, 2, ... take the values of the sorted 1-D arrays `axes` (a product grid).
        Shape: tuple of the axis lengths."""
        if len(axes) != self.n_features - 1:
            raise ValueError(f"model has {self.n_features} features; got 1 + {len(axes)}")
        shape = tuple(len(x) for x in axes)
        out = np.zeros(shape, np.float64)
        for tree in self.trees:
            stack = [(tree, [(0, n) for n in shape])]
            while stack:
                node, box = stack.pop()
                if "split_feature" not in node:
                    out[tuple(slice(a, b) for a, b in box)] += node["leaf_value"]
                    continue
                j, thr = node["split_feature"], node["threshold"]
                if j == 0:
                    stack.append((node["left_child"] if scalar <= thr else node["right_child"],
                                  box))  # fmt: skip
                    continue
                ax = axes[j - 1]
                lo, hi = box[j - 1]
                cut = lo + int(np.searchsorted(ax[lo:hi], thr, side="right"))
                if cut > lo:
                    left = list(box)
                    left[j - 1] = (lo, cut)
                    stack.append((node["left_child"], left))
                if cut < hi:
                    right = list(box)
                    right[j - 1] = (cut, hi)
                    stack.append((node["right_child"], right))
        return out


def slice_axes(model: mg.GameModel, t: int) -> list[np.ndarray]:
    """Grid axes of features 1..7 (s, down, dbin, yl, a, b, r) for the slice at tick t."""
    g, c = model.kern.grid, model.clock
    ph = model.phase(t)
    e = model.extra_shape(ph)
    full = float(c.n_timeouts)

    def ar(n: int, start: int = 0) -> np.ndarray:
        return np.arange(start, start + n, dtype=np.float64)

    if ph == "end":
        a, b, r = ar(e[0]), ar(e[1]), np.array([0.0])
    elif ph == "first":
        a, b, r = np.array([full]), np.array([full]), ar(e[0])
    else:
        a, b, r = np.array([full]), np.array([full]), np.array([0.0])
    return [ar(model.n_score, -model.score_max), ar(g.n_down, 1), ar(g.n_dist), ar(g.n_yl, 1),
            a, b, r]  # fmt: skip


def slice_raw(gp: GridPredictor, model: mg.GameModel, t: int) -> np.ndarray:
    """Raw scores of every state of tick t, in the solver's slice shape (nf, S, E1, E2)."""
    g = model.kern.grid
    ph = model.phase(t)
    axes = slice_axes(model, t)
    z = gp.raw(float(t), axes)  # (S, down, dbin, yl, A, B, R)
    z = np.moveaxis(z, 0, 3).reshape(g.nf, model.n_score, len(axes[4]), len(axes[5]),
                                     len(axes[6]))  # fmt: skip
    if ph == "end":
        return z[..., 0]
    return z[:, :, 0, 0, :][..., None]  # first: (nf, S, 2, 1); mid: (nf, S, 1, 1)


def slice_values(gp: GridPredictor, model: mg.GameModel, t: int, dtype: Any = np.float32
                 ) -> np.ndarray:  # fmt: skip
    """Predicted win probability (sigmoid of the raw score) of every state of tick t."""
    z = slice_raw(gp, model, t)
    return (1.0 / (1.0 + np.exp(-z))).astype(dtype)


def slice_features(model: mg.GameModel, t: int) -> np.ndarray:
    """Feature rows of every state of tick t, in slice order (for checks)."""
    from fourth_and_rl.overopt import features

    g = model.kern.grid
    e = model.extra_shape(model.phase(t))
    f, s, e1, e2 = np.meshgrid(np.arange(g.nf), np.arange(-model.score_max, model.score_max + 1),
                               np.arange(e[0]), np.arange(e[1]), indexing="ij")  # fmt: skip
    ph = model.phase(t)
    a = e1 if ph == "end" else 0
    b = e2 if ph == "end" else 0
    r = e1 if ph == "first" else 0
    return features(model, np.full(f.shape, t), s, f, a, b, r)
