"""M1 extension: a neural win-probability model (PyTorch MLP) and a deep ensemble.

Same rows, same season split and same 17 inputs as the gradient-boosted model in
`wp.py`, so the two can be compared fairly. The training loop is written out by hand.

Public API, mirroring `wp.WPModel`:
    NNWPModel   fit(train, calib=None, seed=...), predict(df), save(path), load(path)
    WPEnsemble  fit(train, calib=None, seed=...), predict(df) (mean of the members),
                predict_members(df) with shape (K, n), save(path), load(path)

The teaching notes for this module are in docs/notes/m1_wp_nn.md.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import torch
import yaml
from torch import nn

from fourth_and_rl import wp
from fourth_and_rl.paths import CONFIGS

DEFAULT_SEED = wp.DEFAULT_SEED

DEFAULT_PARAMS: dict[str, Any] = {
    "hidden": [64, 64],          # width of each hidden layer
    "dropout": 0.0,              # share of hidden units zeroed at random while training
    "weight_decay": 1e-4,        # AdamW decoupled weight decay
    "lr": 2e-3,                  # Adam step size
    "batch_size": 4096,          # plays per gradient step
    "max_epochs": 40,            # upper limit on passes over the training plays
    "patience": 6,               # stop after this many epochs with no better held-out loss
    "early_stopping": True,      # False: train max_epochs and keep the last weights
    "holdout_game_fraction": 0.1,  # share of training GAMES held out when calib is None
    "label_noise": 0.0,          # chance that a whole game's label is flipped, per epoch
    "one_hot_down": True,        # down (0 to 4) as five indicators, not one number
    "squash_margin": True,       # asinh on margin_per_sqrt_time before standardizing
    "calibration": "none",       # "none" or "platt" (fitted on calib, as in wp.py)
}  # fmt: skip

SQUASH_SCALE = 10.0  # asinh(x / 10): close to linear inside +-10, logarithmic beyond
N_DOWNS = 5  # down is coded 0 (no down) to 4
TRAIN_MONITOR_ROWS = 50_000  # training rows re-scored each epoch to record the clean loss


def load_config(path: Path | None = None) -> dict:
    """Read configs/wp_nn.yaml."""
    with open(path or CONFIGS / "wp_nn.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------- preprocessing


class Preprocessor:
    """Turns the `wp.FEATURES` matrix into standardized network inputs.

    The mean and standard deviation are learned by `fit` and must come from training
    rows only. Two optional, fixed (not learned) transforms are applied first:
    `down` is replaced by five 0/1 indicators, because down 0 means "kickoff or
    try", which is not "one less than first down"; and `margin_per_sqrt_time`, which
    reaches the hundreds in the last seconds of a decided game, is passed through
    asinh so that a handful of extreme rows do not set the scale for all the others.
    """

    def __init__(self, one_hot_down: bool = True, squash_margin: bool = True) -> None:
        self.one_hot_down = one_hot_down
        self.squash_margin = squash_margin
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _expand(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).copy()
        if self.squash_margin:
            j = wp.FEATURES.index("margin_per_sqrt_time")
            x[:, j] = np.arcsinh(x[:, j] / SQUASH_SCALE)
        if self.one_hot_down:
            j = wp.FEATURES.index("down")
            down = np.clip(x[:, j], 0, N_DOWNS - 1).astype(int)
            x = np.concatenate([np.delete(x, j, axis=1), np.eye(N_DOWNS)[down]], axis=1)
        return x

    @property
    def n_out(self) -> int:
        return len(wp.FEATURES) + (N_DOWNS - 1 if self.one_hot_down else 0)

    def fit(self, x: np.ndarray) -> Preprocessor:
        z = self._expand(x)
        self.mean = z.mean(axis=0)
        std = z.std(axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)  # a constant column stays at zero
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("preprocessor is not fitted")
        return ((self._expand(x) - self.mean) / self.std).astype(np.float32)


def build_mlp(n_in: int, hidden: list[int], dropout: float) -> nn.Sequential:
    """Linear -> ReLU (-> Dropout) for each hidden layer, then one linear output (a logit)."""
    layers: list[nn.Module] = []
    width = n_in
    for h in hidden:
        layers += [nn.Linear(width, int(h)), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        width = int(h)
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def resample_games(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """One game-level bootstrap resample: draw whole games with replacement.

    A game drawn twice appears twice, under two different `game_id` values, so that
    anything grouping by game afterwards (the early-stopping hold-out, label noise)
    treats the copies as separate games.
    """
    games = df["game_id"].unique(maintain_order=True)
    draw = rng.integers(0, len(games), len(games))
    picked = pl.DataFrame({"game_id": games[draw], "_copy": np.arange(len(games))})
    out = picked.join(df, on="game_id", how="inner")
    return out.with_columns(
        (pl.col("game_id").cast(pl.Utf8) + "#" + pl.col("_copy").cast(pl.Utf8)).alias("game_id")
    ).drop("_copy")


# ------------------------------------------------------------------------------ model


def _bce(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean log loss of `model` on (x, y), dropout off, no gradients."""
    model.eval()
    with torch.no_grad():
        return float(nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1), y))


class NNWPModel:
    """Win probability for the team with the ball, from a small fully connected network.

    fit(train, calib) trains on `train` and uses `calib` (different games) to decide
    when to stop. With calib=None, a random share of the training GAMES is held out
    for that purpose. If params["calibration"] is "platt", a two-parameter map is
    also fitted on `calib`, as in `wp.WPModel`.
    """

    def __init__(self, params: dict | None = None, n_threads: int = 1) -> None:
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
        self.n_threads = n_threads
        self.net: nn.Sequential | None = None
        self.pre: Preprocessor | None = None
        self.calibrator = wp.Calibrator("none")
        self.seed: int | None = None
        self.n_train_rows = 0
        self.n_stop_rows = 0
        self.best_epoch = 0
        self.fit_seconds = 0.0
        self.history: list[dict[str, float]] = []

    # -- fitting

    @staticmethod
    def _xyg(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Features, labels and an integer game index for the rows usable for fitting."""
        for col in (wp.LABEL, "game_id"):
            if col not in df.columns:
                raise KeyError(f"fit needs the column {col!r}")
        keep = wp.scoreable(df) & df[wp.LABEL].is_not_null().to_numpy()
        sub = df.filter(pl.Series(keep))
        _, game = np.unique(sub["game_id"].to_numpy(), return_inverse=True)
        return (
            wp.make_features(sub).to_numpy(),
            sub[wp.LABEL].to_numpy().astype(np.float32),
            game,
        )

    def fit(
        self,
        train: pl.DataFrame,
        calib: pl.DataFrame | None = None,
        seed: int = DEFAULT_SEED,
    ) -> NNWPModel:
        """Train the network. Rows with no label or an incomplete state are dropped."""
        p = self.params
        start = time.perf_counter()
        torch.set_num_threads(self.n_threads)
        torch.manual_seed(int(seed))
        rng = np.random.default_rng(int(seed))

        x_raw, y_np, game = self._xyg(train)
        if len(np.unique(y_np)) < 2:
            raise ValueError("training data must contain both wins and losses")
        if calib is not None and calib.height > 0:
            x_stop_raw, y_stop_np, _ = self._xyg(calib)
        else:
            # Hold out whole games, never single plays: a play held out from a game
            # whose other plays are in training would be scored by a model that has
            # already seen that game's result.
            n_games = int(game.max()) + 1
            n_held = max(1, int(round(p["holdout_game_fraction"] * n_games)))
            held = np.isin(game, rng.permutation(n_games)[:n_held])
            x_stop_raw, y_stop_np = x_raw[held], y_np[held]
            x_raw, y_np, game = x_raw[~held], y_np[~held], game[~held]
            _, game = np.unique(game, return_inverse=True)

        # The scaler sees the training rows only.
        self.pre = Preprocessor(p["one_hot_down"], p["squash_margin"]).fit(x_raw)
        x = torch.from_numpy(self.pre.transform(x_raw))
        y = torch.from_numpy(y_np)
        x_stop = torch.from_numpy(self.pre.transform(x_stop_raw))
        y_stop = torch.from_numpy(y_stop_np)
        game_t = torch.from_numpy(game.astype(np.int64))
        n, n_games = len(y), int(game.max()) + 1
        monitor = torch.from_numpy(rng.permutation(n)[:TRAIN_MONITOR_ROWS])

        net = build_mlp(self.pre.n_out, list(p["hidden"]), float(p["dropout"]))
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"])
        )
        loss_fn = nn.BCEWithLogitsLoss()
        shuffler = torch.Generator().manual_seed(int(seed))
        batch, noise = int(p["batch_size"]), float(p["label_noise"])

        best_loss, best_state, best_epoch, self.history = float("inf"), None, 0, []
        for epoch in range(1, int(p["max_epochs"]) + 1):
            # Game-level label noise: flip the result of a few whole games, redrawn
            # every epoch, so no single game's label can be trusted enough to memorize.
            if noise > 0:
                flip = torch.rand(n_games, generator=shuffler) < noise
                target = torch.where(flip[game_t], 1.0 - y, y)
            else:
                target = y
            net.train()
            order = torch.randperm(n, generator=shuffler)
            for i in range(0, n, batch):
                idx = order[i : i + batch]
                logits = net(x[idx]).squeeze(1)      # forward pass
                loss = loss_fn(logits, target[idx])  # log loss of this batch
                optimizer.zero_grad()
                loss.backward()                      # backpropagation
                optimizer.step()                     # one AdamW update
            stop_loss = _bce(net, x_stop, y_stop)
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": _bce(net, x[monitor], y[monitor]),
                    "heldout_loss": stop_loss,
                }
            )
            if stop_loss < best_loss - 1e-6:
                best_loss, best_epoch = stop_loss, epoch
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            elif p["early_stopping"] and epoch - best_epoch >= int(p["patience"]):
                break
        if p["early_stopping"] and best_state is not None:
            net.load_state_dict(best_state)
            self.best_epoch = best_epoch
        else:
            self.best_epoch = len(self.history)
        net.eval()
        self.net, self.seed = net, int(seed)
        self.n_train_rows, self.n_stop_rows = n, len(y_stop)

        if p["calibration"] != "none" and calib is not None and calib.height > 0:
            with torch.no_grad():
                score = net(x_stop).squeeze(1).numpy().astype(np.float64)
            self.calibrator = wp.Calibrator(p["calibration"]).fit(score, y_stop_np)
        else:
            self.calibrator = wp.Calibrator("none")
        self.fit_seconds = time.perf_counter() - start
        return self

    # -- prediction

    def raw_score(self, df: pl.DataFrame) -> np.ndarray:
        """The network's log-odds, NaN for rows that cannot be scored."""
        if self.net is None or self.pre is None:
            raise RuntimeError("model is not fitted")
        torch.set_num_threads(self.n_threads)
        ok = wp.scoreable(df)
        out = np.full(df.height, np.nan)
        if ok.any():
            x = self.pre.transform(wp.make_features(df.filter(pl.Series(ok))).to_numpy())
            self.net.eval()
            with torch.no_grad():
                out[ok] = self.net(torch.from_numpy(x)).squeeze(1).numpy()
        return out

    def predict(self, df: pl.DataFrame, calibrated: bool = True) -> np.ndarray:
        """P(posteam wins), one value per row. NaN where the state is incomplete."""
        score = self.raw_score(df)
        ok = ~np.isnan(score)
        out = np.full(df.height, np.nan)
        cal = self.calibrator if calibrated else wp.Calibrator("none")
        out[ok] = cal.transform(score[ok])
        return out

    # -- persistence

    def _blob(self) -> dict[str, Any]:
        if self.net is None or self.pre is None:
            raise RuntimeError("model is not fitted")
        return {
            "state": {k: v.numpy().copy() for k, v in self.net.state_dict().items()},
            "features": wp.FEATURES,
            "params": self.params,
            "mean": self.pre.mean,
            "std": self.pre.std,
            "calibrator": self.calibrator,
            "n_threads": self.n_threads,
            "seed": self.seed,
            "n_train_rows": self.n_train_rows,
            "n_stop_rows": self.n_stop_rows,
            "best_epoch": self.best_epoch,
            "fit_seconds": self.fit_seconds,
            "history": self.history,
        }

    @classmethod
    def _from_blob(cls, blob: dict[str, Any]) -> NNWPModel:
        if blob["features"] != wp.FEATURES:
            raise ValueError("saved model was fitted with a different feature list; refit it")
        model = cls(blob["params"], blob["n_threads"])
        p = model.params
        model.pre = Preprocessor(p["one_hot_down"], p["squash_margin"])
        model.pre.mean, model.pre.std = blob["mean"], blob["std"]
        model.net = build_mlp(model.pre.n_out, list(p["hidden"]), float(p["dropout"]))
        model.net.load_state_dict({k: torch.from_numpy(v) for k, v in blob["state"].items()})
        model.net.eval()
        model.calibrator = blob["calibrator"]
        for key in ("seed", "n_train_rows", "n_stop_rows", "best_epoch", "fit_seconds", "history"):
            setattr(model, key, blob[key])
        return model

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._blob(), path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> NNWPModel:
        return cls._from_blob(joblib.load(path))


# --------------------------------------------------------------------------- ensemble


class WPEnsemble:
    """K networks that differ in their random seed (a "deep ensemble").

    Member i is trained with seed `seed + i`, which changes its starting weights and
    the order of its minibatches. With bootstrap=True each member is also trained on
    its own game-level resample of `train` (bagging). `predict` is the mean of the
    members' probabilities and `predict_members` returns all of them.
    """

    def __init__(
        self,
        params: dict | None = None,
        k: int = 10,
        bootstrap: bool = False,
        n_threads: int = 1,
    ) -> None:
        if k < 1:
            raise ValueError("an ensemble needs at least one member")
        self.params = params
        self.k = k
        self.bootstrap = bootstrap
        self.n_threads = n_threads
        self.members: list[NNWPModel] = []
        self.seed: int | None = None

    def fit(
        self,
        train: pl.DataFrame,
        calib: pl.DataFrame | None = None,
        seed: int = DEFAULT_SEED,
    ) -> WPEnsemble:
        self.members, self.seed = [], int(seed)
        for i in range(self.k):
            member_seed = int(seed) + i
            data = train
            if self.bootstrap:
                data = resample_games(train, np.random.default_rng(member_seed))
            self.members.append(
                NNWPModel(self.params, self.n_threads).fit(data, calib, seed=member_seed)
            )
        return self

    def predict_members(self, df: pl.DataFrame) -> np.ndarray:
        """Every member's P(posteam wins): shape (K, number of rows)."""
        if not self.members:
            raise RuntimeError("ensemble is not fitted")
        return np.stack([m.predict(df) for m in self.members])

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """The mean of the members' probabilities, one value per row."""
        return self.predict_members(df).mean(axis=0)

    def spread(self, df: pl.DataFrame) -> np.ndarray:
        """Standard deviation of the members' probabilities, one value per row."""
        return self.predict_members(df).std(axis=0, ddof=1)

    def save(self, path: str | Path) -> Path:
        if not self.members:
            raise RuntimeError("ensemble is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "members": [m._blob() for m in self.members],
                "params": self.params,
                "bootstrap": self.bootstrap,
                "n_threads": self.n_threads,
                "seed": self.seed,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> WPEnsemble:
        blob = joblib.load(path)
        ens = cls(blob["params"], len(blob["members"]), blob["bootstrap"], blob["n_threads"])
        ens.members = [NNWPModel._from_blob(b) for b in blob["members"]]
        ens.seed = blob["seed"]
        return ens
