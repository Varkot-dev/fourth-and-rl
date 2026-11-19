"""Question 6, JAX backend: check the end-phase solver against the CPU solver on the real
game, and time both, on this machine (no GPU needed).

One matchup (Detroit's end-of-2025 offense and defense against an average team, the
`det_vs_average` row of scripts/questions/q6_team_specific.py) is solved over the end
phase by the CPU solver (`mdp_team.solve_matchup`) and by `mdp_team_jax` on whatever
device JAX finds. Recorded: the largest value difference in both arrays over all 61
slices, the largest difference in every action value at 4:00, how much the policies
disagree and whether every disagreement is a tie (within 1e-5 of the CPU's own action
values), the headline decision against results/questions/q6_team_specific.json, and the
seconds per tick of each JAX backend (dense, bcoo, bcoo_cusparse) over a few ticks.

The GPU numbers come from notebooks/team_matchups_colab.ipynb; this script only records
what this machine measures. Needs data/mdp, data/models/team_strength.joblib and the
`jax` extra.

Run:  python scripts/questions/q6_jax_check.py        (about 5 minutes on 3 CPU threads)
Writes results/questions/q6_jax_check.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from fourth_and_rl import mdp  # noqa: E402
from fourth_and_rl import mdp_game as mg  # noqa: E402
from fourth_and_rl import mdp_team as mt  # noqa: E402
from fourth_and_rl import mdp_team_jax as mj  # noqa: E402
from fourth_and_rl import team_matchups as tm  # noqa: E402
from fourth_and_rl.io import read_json, write_json  # noqa: E402
from fourth_and_rl.paths import RESULTS  # noqa: E402

OUT = RESULTS / "questions" / "q6_jax_check.json"
Q6 = RESULTS / "questions" / "q6_team_specific.json"


def tie_gap(sol: mg.Solution, pol: np.ndarray, t: int) -> tuple[float, int]:
    """Largest loss, under the CPU's action values at t, of the JAX policy's choice, and
    the number of states where the two policies differ."""
    q = sol.q_slice(t)
    mine, alt = sol.pol["end"][t], pol[t]
    diff = (mine != alt) & (alt != mg.A_OVER) & (mine != mg.A_OVER)
    if not diff.any():
        return 0.0, 0
    codes = sorted(q)
    pos = np.zeros(len(mg.ACTION_NAMES), dtype=np.int64)
    pos[codes] = np.arange(len(codes))
    vals = np.stack([q[c][diff] for c in codes])
    cols = np.arange(int(diff.sum()))
    return float(np.abs(vals[pos[mine[diff]], cols] - vals[pos[alt[diff]], cols]).max()), int(
        diff.sum()
    )


def main() -> None:
    import jax

    inputs = tm.load_local_inputs()
    specs = {s["key"]: s for s in tm.matchup_grid(tm.season_end_ratings())}
    spec = specs["det_vs_average"]
    qcfg = mdp.load_config()["question1"]
    tick = qcfg["headline"]["seconds"] // inputs.base.clock.tick_seconds
    ma, mb, cal = tm.Kernels(inputs).pair(spec)

    print("benchmark of the JAX backends (3 timed ticks each) ...", flush=True)
    bench = mj.benchmark_backends((ma, mb), ticks=3)
    print(bench, flush=True)

    compile_s = mj.warmup([(ma, mb)], "dense", q=True)
    t0 = time.time()
    res = mj.solve_end_phases([(ma, mb)], backend="dense", q_ticks=(tick,))[0]
    jax_s = time.time() - t0
    t0 = time.time()
    cpu = mt.solve_matchup(ma, mb)
    cpu_s = time.time() - t0
    print(f"CPU {cpu_s:.1f} s, JAX dense {jax_s:.1f} s on {jax.devices()[0]}", flush=True)

    vdiff = [float(np.abs(res.v[k] - cpu.sols[k].v["end"]).max()) for k in range(2)]
    pol_share = [float((res.pol[k] != cpu.sols[k].pol["end"]).mean()) for k in range(2)]
    q = cpu.sols[0].q_slice(tick)
    qdiff = {}
    for code, x in q.items():
        fin = np.isfinite(x)
        same = bool(np.array_equal(fin, np.isfinite(res.q[tick][code][0])))
        qdiff[mg.ACTION_NAMES[code]] = {
            "max_abs_diff": float(np.abs(x[fin] - res.q[tick][code][0][fin]).max()),
            "same_allowed_states": same,
        }
    ties = {}
    for k, name in enumerate(mt.TEAMS):
        for t in (12, 30, tick):
            gap, n = tie_gap(cpu.sols[k], res.pol[k], t)
            ties[f"{name}_t{t}"] = {"states_differing": n, "largest_value_gap": gap}
    rec = tm.decision_record(res.q[tick], ma, qcfg)
    ref = {m["key"]: m for m in read_json(Q6)["matchups"]}["det_vs_average"]
    out = {
        "what": "JAX end-phase solver against the CPU solver on the real game, one matchup "
        "(scripts/questions/q6_jax_check.py)",
        "matchup": spec["key"],
        "kernels": cal,
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "python": platform.python_version(),
        "cpu_threads_env": int(os.environ.get("OMP_NUM_THREADS", "0")),
        "seconds": {"cpu_solver": cpu_s, "jax_dense": jax_s, "jax_compile": compile_s},
        "backend_benchmark": bench,
        "max_abs_value_diff": dict(zip(mt.TEAMS, vdiff, strict=True)),
        "action_value_diff_at_4_00_team_A": qdiff,
        "policy_share_differing": dict(zip(mt.TEAMS, pol_share, strict=True)),
        "policy_ties": ties,
        "headline_jax": rec["headline"],
        "headline_committed_q6": ref["headline"],
        "map_max_abs_diff_vs_committed_points": float(np.abs(
            np.asarray(rec["map_go_minus_punt_points"])
            - np.asarray(ref["map_go_minus_punt_points"])).max()),
        "note": "float32 throughout; JAX time includes copying every slice back to the host",
    }  # fmt: skip
    write_json(out, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
