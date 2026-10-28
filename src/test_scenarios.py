"""Sanity check of the random episode sampler."""

import collections, json, sys, time
from pathlib import Path
import numpy as np
import drone_env, scenario

OUT = Path(__file__).resolve().parents[1] / "runs"

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    R = {}
    t0 = time.time()
    for split in ("train", "test_familiar", "test_unfamiliar"):
        rng = np.random.default_rng({"train": 1, "test_familiar": 2, "test_unfamiliar": 3}[split])
        enc_all, faults = [], collections.Counter()
        for i in range(n):
            cfg = scenario.sample_episode(rng, split)
            env = drone_env.DroneEnv(render=False, **cfg)
            o = env.reset()
            for _ in range(int(40 / env.agent_dt)):
                o = env.step({"vz": float(np.clip(0.8 * (15 - o["baro_alt"]), -2, 3))})
            tr = env.truth()
            enc_all += tr["encounters"]
            faults.update(f"{f['sensor']}:{f['kind']}" for f in cfg["faults"])
        fam = collections.Counter(e["family"] for e in enc_all)
        bins = collections.Counter(
            "hit" if e["hit"] else ("near<5m" if e["min_dist"] < 5 else "far") for e in enc_all
        )
        fast = sum(e["speed"] > 100 for e in enc_all)
        R[split] = {
            "encounters": len(enc_all),
            "families": dict(fam),
            "outcomes_no_reflex": dict(bins),
            "speed>100": fast,
            "faults": dict(faults),
        }
        print(split, json.dumps(R[split], ensure_ascii=False), flush=True)
    (OUT / "SCENARIO_TEST.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print("wall", round(time.time() - t0))
