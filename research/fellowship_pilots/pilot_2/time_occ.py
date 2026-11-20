import time,numpy as np
from fourth_and_rl.offline import datasets as ds
from fourth_and_rl.paths import CONFIGS
from fourth_and_rl.rl import evaluate as ev, runner
snap=runner.snapshot(runner.load_config(CONFIGS/"offline.yaml")); m=snap.drive
mu=ds.coach_policy(m,snap.coach["table"])
t=time.time(); [ev.occupancy(m,mu) for _ in range(5)]; print("occ x5",time.time()-t)
