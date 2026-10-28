"""Outcome mix of the encounter generator on many episodes."""

import collections, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np


def run(args):
    split, i = args
    import drone_env, scenario

    rng = np.random.default_rng(1000 * {"train": 1, "test_familiar": 2, "test_unfamiliar": 3}[split] + i)
    cfg = scenario.sample_episode(rng, split)
    env = drone_env.DroneEnv(render=False, **cfg)
    o = env.reset()
    for _ in range(int(40 / env.agent_dt)):
        o = env.step({"vz": float(np.clip(0.8 * (15 - o["baro_alt"]), -2, 3))})
    return split, env.truth()["encounters"]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    t0 = time.time()
    R = {}
    jobs = [(s, i) for s in ("train", "test_familiar", "test_unfamiliar") for i in range(n)]
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(run, jobs))
    for split in ("train", "test_familiar", "test_unfamiliar"):
        E = [e for s, L in res if s == split for e in L]
        cat = lambda e: (
            "hit"
            if e["hit"]
            else ("near<2m" if e["min_dist"] < 2 else ("near<5m" if e["min_dist"] < 5 else "far"))
        )
        R[split] = {
            "n_episodes": n,
            "n_encounters": len(E),
            "overall": dict(collections.Counter(map(cat, E))),
            "by_family": {
                f: dict(collections.Counter(cat(e) for e in E if e["family"] == f))
                for f in sorted({e["family"] for e in E})
            },
            "speed>100": sum(e["speed"] > 100 for e in E),
        }
        print(split, json.dumps(R[split]), flush=True)
    out = Path(__file__).resolve().parents[1] / "runs" / "OUTCOME_MIX.json"
    out.write_text(json.dumps(R, indent=1), encoding="utf-8")
    print("wall", round(time.time() - t0))
