"""Track A2: which overconfidence factors does the real first-crossing result rule out?

For each real feed q, rescale it to expit(c * logit(q)) and run the first-crossing test
(max |z| over thresholds). The c values the test does not reject at 5% form a confidence
set; a feed that is overconfident by k (in the A1 sense) is calibrated after c = 1 / k.
The critical value is the simulated one for 4,350 games from the A1 size check.

Writes results/research/blown_power_invert.json and assets/research/blown_power_invert.png.
Needs data/processed/plays.parquet (scripts/build_data.py).
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from fourth_and_rl import blown_leads as bl
from fourth_and_rl import style
from fourth_and_rl.paths import ROOT

CONFIG = ROOT / "configs" / "blown_power.yaml"
SIZE = ROOT / "results" / "research" / "blown_power_size_check.json"
PLAYS = ROOT / "data" / "processed" / "plays.parquet"
OUT_JSON = ROOT / "results" / "research" / "blown_power_invert.json"
OUT_FIG = ROOT / "assets" / "research" / "blown_power_invert.png"


def crossing_zs(hp: pl.DataFrame, thresholds: list[float]) -> list[float]:
    zs = []
    for t in thresholds:
        cr = bl.first_crossings(hp, t)
        implied = 1.0 - cr["p_cross"].to_numpy()
        lost = ~cr["won"].to_numpy()
        sd = np.sqrt(np.sum(implied * (1 - implied)))
        zs.append(float((lost.sum() - implied.sum()) / sd))
    return zs


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    inv = cfg["invert"]
    crit = json.loads(SIZE.read_text())["critical_value"]
    g = inv["c_grid"]
    cs = np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 4)
    plays = pl.read_parquet(PLAYS)
    out = {"critical_value": crit, "thresholds": cfg["thresholds"], "models": {}}
    for m in inv["models"]:
        hp = bl.home_paths(plays, m["column"])
        keep = hp.group_by("game_id").len().filter(pl.col("len") >= inv["min_plays"])
        hp = hp.filter(pl.col("game_id").is_in(keep["game_id"].implode()))
        q = hp["home_wp_path"].to_numpy().clip(1e-9, 1 - 1e-9)
        lq = np.log(q / (1 - q))
        rows = []
        for c in cs:
            h2 = hp.with_columns(pl.Series("home_wp_path", 1 / (1 + np.exp(-c * lq))))
            z = crossing_zs(h2, cfg["thresholds"])
            rows.append({"c": float(c), "z": z, "max_abs_z": float(np.max(np.abs(z)))})
        accepted = [r["c"] for r in rows if r["max_abs_z"] <= crit]
        best = min(rows, key=lambda r: r["max_abs_z"])
        out["models"][m["key"]] = {
            "column": m["column"],
            "n_games": int(keep.height),
            "z_at_c_1": next(r["z"] for r in rows if r["c"] == 1.0),
            "c_best": best["c"],
            "c_confidence_set": [min(accepted), max(accepted)] if accepted else None,
            "k_confidence_set": [1 / max(accepted), 1 / min(accepted)] if accepted else None,
            "grid": rows,
        }
        print(m["key"], "z at c=1:", [round(x, 2) for x in out["models"][m["key"]]["z_at_c_1"]],
              "c set:", out["models"][m["key"]]["c_confidence_set"], "best:", best["c"])
    OUT_JSON.write_text(json.dumps(out, indent=2))
    plot(out)


def plot(out: dict) -> None:
    style.apply()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    labels = {"nflfastr": "nflfastR (no spread)", "vegas": "nflfastR with spread"}
    for key, res in out["models"].items():
        c = [r["c"] for r in res["grid"]]
        k = [1 / x for x in c]
        ax.plot(k, [r["max_abs_z"] for r in res["grid"]], color=style.MODEL_COLORS[key],
                label=labels.get(key, key))
    ax.axhline(out["critical_value"], color=style.MUTED, ls="--", lw=1)
    ax.text(0.78, out["critical_value"] + 0.15, "5% critical value", color=style.INK_2,
            fontsize=9)
    ax.axvline(1.0, color=style.AXIS, lw=1)
    ax.set_xlabel("Overconfidence factor k the data is tested against (1 = as reported)")
    ax.set_ylabel("First-crossing statistic, max |z|")
    ax.set_ylim(0, None)
    ax.legend(loc="upper center")
    style.title(ax, "Which overconfidence levels does 16 seasons of data rule out?",
                "No single k gets under the dashed line; the best fit is close to k = 1")
    style.save(fig, OUT_FIG)


if __name__ == "__main__":
    main()
