"""Research milestone R1, days 1 to 4: checks of the exact pipeline on the real game.

Subcommands (settings in configs/overopt.yaml, section `checks`):

  timing    Day 1. Time and peak memory of one full-game backward pass of the evaluator:
            a symmetric coach-vs-coach pass (one solution) and a linked pair (two
            solutions), with both post-decision settings, and of one forward occupancy
            pass (two teams, float64), each in a child process so the peak memory is its
            own.
            Writes results/research/overopt_day1_timing.json.
  post_mc   Day 2. Exact coach-vs-coach values with post-decisions fixed to the coach
            rules, against Monte Carlo from `Simulator` (coach policy, post="coach"),
            at states drawn from synthetic coach games.
            Writes results/research/overopt_day2_post_mc.json.

The real game is the solved MDP in data/mdp. No play-by-play is loaded.

Run:  python scripts/research/overopt_checks.py timing
      python scripts/research/overopt_checks.py post_mc     (about 15 minutes)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np

from fourth_and_rl import mdp_game as mg
from fourth_and_rl import overopt as oo
from fourth_and_rl import policy_eval as pe
from fourth_and_rl.io import write_json
from fourth_and_rl.paths import RESULTS

OUT = RESULTS / "research"


def _one_timing(kind: str, post: str) -> dict:
    sol, coach = oo.load_game()
    m = sol.model
    pol = pe.CoachPolicy(m, coach)
    if kind == "forward":
        from fourth_and_rl.occupancy import ForwardPass

        fw = ForwardPass(m, pol, pe.CoachPolicy(m, coach), coach=coach).run()
        return {"kind": kind, "post": post, "seconds": fw.seconds,
                "peak_rss_gb": fw.peak_rss_gb, "result_a": fw.result_a,
                "total_mass": fw.total_mass, "n_teams": 2, "dtype": "float64"}  # fmt: skip
    pair = pe.PolicyPair(m, pol, None if kind == "symmetric" else pe.CoachPolicy(m, coach),
                         post=post, coach=coach)  # fmt: skip
    stats = pe.run([pair])
    stats.update(kind=kind, post=post, start_value=pair.start_value(),
                 n_solutions=1 if kind == "symmetric" else 2, n_ticks=m.clock.n_ticks,
                 ring_slices_kept=len(pair.a.buf), lag=pair.a.lag)  # fmt: skip
    return stats


def timing(args: argparse.Namespace) -> None:
    rows = []
    for kind, post in (("symmetric", "coach"), ("symmetric", "model"), ("pair", "coach"),
                       ("pair", "model"), ("forward", "coach")):  # fmt: skip
        out = subprocess.run([sys.executable, __file__, "_timing", kind, post],
                             capture_output=True, text=True, check=True)  # fmt: skip
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(rows[-1], flush=True)
    write_json({"what": "time and peak memory of one full-game pass of the exact policy "
                        "evaluator (backward) and of the forward occupancy pass, coach "
                        "policy, each run in its own process",
                "machine": "this CPU session (4 cores, 15 GB)", "runs": rows},
               OUT / "overopt_day1_timing.json")  # fmt: skip


def pick_states(sol: mg.Solution, coach: dict, n: int, seed: int) -> list[tuple[int, ...]]:
    """n distinct pre-snap states of synthetic coach games (seed), a third from each phase,
    leaving out states the leader can kneel out and states whose solved value is outside
    0.15 to 0.85 (nearly decided games, where a Monte Carlo check says little). Returns
    raw (t, f, s, a, b, r)."""
    rng = np.random.default_rng(seed)
    sim = oo._Recorder(sol, coach=coach)
    sim.log = []
    st = oo.kickoff_start(sim, 300, rng)
    mg.simulate(sim, st, rng, policy="coach", post="coach")
    rows = np.unique(np.concatenate(sim.log)[:, 1:7], axis=0)  # t, s, f, a, b, r
    m = sol.model
    t = rows[:, 0]
    ph = np.where(t <= m.clock.endgame_ticks, 0, np.where(t <= m.clock.th, 1, 2))
    st2 = {k: rows[:, i] for i, k in enumerate(("t", "s", "f", "a", "b", "r"))}
    over = sim.game_over(st2, np.arange(len(rows)))
    vstar = sim.values(st2)
    over = over | (vstar < 0.15) | (vstar > 0.85)
    out = []
    for code, k in zip((0, 1, 2), (n - 2 * (n // 3), n // 3, n // 3), strict=True):
        pool = np.nonzero((ph == code) & ~over & (t > 0))[0]
        out += [tuple(int(x) for x in rows[i]) for i in rng.choice(pool, size=k, replace=False)]
    return out


def post_mc(args: argparse.Namespace) -> None:
    cfg = oo.load_config()["checks"]["post_mc"]
    sol, coach = oo.load_game()
    m = sol.model
    states = pick_states(sol, coach, cfg["n_states"], cfg["seed"])
    raw = []
    for t, s, f, a, b, r in states:
        ph = m.phase(t)
        e = (a, b) if ph == "end" else ((r, 0) if ph == "first" else (0, 0))
        raw.append(("A", (t, f, s, *e)))
    pol = pe.CoachPolicy(m, coach)
    pair = pe.PolicyPair(m, pol, record=raw, post="coach", coach=coach)
    t0 = time.time()
    stats = pe.run([pair])
    exact_s = time.time() - t0
    sim = mg.Simulator(sol, coach=coach)
    rng = np.random.default_rng(cfg["seed"] + 1)
    n = cfg["games_per_state"]
    rows = []
    t0 = time.time()
    for (t, s, f, a, b, r), exact in zip(states, pair.recorded, strict=True):
        st = sim.start(n, t=t, s=s, f=f, a=a, b=b, r=r)
        res = mg.simulate(sim, st, rng, policy="coach", post="coach")["result"]
        mean, se = float(res.mean()), float(res.std(ddof=1) / np.sqrt(n))
        z = (mean - exact) / se if se > 0 else float(mean != exact) * np.inf
        down, dbin, yl = (int(x) for x in sol.grid.unpack(f))
        rows.append({"t": t, "seconds": t * m.clock.tick_seconds, "phase": m.phase(t),
                     "score": s, "down": down, "dist_bin": dbin, "yardline_100": yl,
                     "off_timeouts": a, "def_timeouts": b, "receives_second_half": r,
                     "exact": exact, "mc": mean, "mc_se": se,
                     "z": z})  # fmt: skip
        print(rows[-1], flush=True)
    z = np.array([x["z"] for x in rows])
    # A follow-up, not a replacement: the states with the largest |z| again with four
    # times the games and fresh random numbers. A real error would keep its sign and grow.
    recheck = []
    rng2 = np.random.default_rng(cfg["seed"] + 2)
    for i in np.argsort(-np.abs(z))[: cfg["recheck_states"]]:
        t, s, f, a, b, r = states[i]
        st = sim.start(4 * n, t=t, s=s, f=f, a=a, b=b, r=r)
        res = mg.simulate(sim, st, rng2, policy="coach", post="coach")["result"]
        se = float(res.std(ddof=1) / np.sqrt(4 * n))
        recheck.append({"state": int(i), "first_z": float(z[i]), "games": 4 * n,
                        "mc": float(res.mean()), "mc_se": se,
                        "z": (float(res.mean()) - rows[i]["exact"]) / se})  # fmt: skip
        print("recheck", recheck[-1], flush=True)
    write_json({"what": "exact coach-vs-coach win probability with post-decisions fixed to "
                        "the coach rules, against Monte Carlo from the simulator (coach "
                        "policy, post='coach') at states drawn from synthetic coach games",
                "config": cfg, "games_per_state": n, "exact_pass_seconds": exact_s,
                "exact_pass_peak_rss_gb": stats["peak_rss_gb"],
                "mc_seconds": time.time() - t0, "n_states": len(rows),
                "max_abs_z": float(np.abs(z).max()), "n_within_2se": int((np.abs(z) < 2).sum()),
                "mean_se": float(np.mean([x["mc_se"] for x in rows])),
                "chi2_sum_z2": float((z**2).sum()), "states": rows, "recheck": recheck},
               OUT / "overopt_day2_post_mc.json")  # fmt: skip


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_timing":
        print(json.dumps(_one_timing(sys.argv[2], sys.argv[3])))
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("timing", "post_mc"):
        sub.add_parser(name)
    args = ap.parse_args()
    {"timing": timing, "post_mc": post_mc}[args.cmd](args)


if __name__ == "__main__":
    main()
