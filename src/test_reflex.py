"""Stage-1 check (bricks)."""

import json, math, sys, time
from pathlib import Path
import numpy as np
import drone_env, perception

OUT = Path(__file__).resolve().parents[1] / "runs"


def episode(seed, reflex_on, seconds=35.0):
    rng = np.random.default_rng(seed)
    throws = sorted(
        (float(rng.uniform(6, seconds - 3)), float(math.exp(rng.uniform(math.log(8), math.log(40)))))
        for _ in range(3)
    )
    dives = [(float(rng.uniform(6, seconds - 6)), int(rng.integers(0, 2)))]
    env = drone_env.DroneEnv(seed=seed, render=False, throws=throws, dives=dives)
    o = env.reset()
    trk, rfx = perception.Tracker(), perception.Reflex()
    first_threat = {}
    for _ in range(int(seconds / env.agent_dt)):
        cmd = {"vx": 0.0, "vy": 0.0, "vz": float(np.clip(0.8 * (15 - o["baro_alt"]), -2, 3)), "yaw_rate": 0.0}
        sm = trk.step(o)
        for s in sm["tracks"]:
            if s["threat"]:
                first_threat.setdefault(s["id"], o["t"])
        if reflex_on:
            cmd = rfx.act(o["t"], sm, cmd)
        o = env.step(cmd)
    tr = env.truth()
    ev = tr["hit_events"]
    launched = [e for e in tr["events"] if e["kind"] == "throw"]
    return {
        "seed": seed,
        "reflex": reflex_on,
        "throws": len(launched),
        "throw_speeds": [round(s, 1) for _, s in throws],
        "throw_hits": sum(e["kind"] == "throw" for e in ev),
        "mover_hits": sum(e["kind"] == "mover" for e in ev),
        "static_hits": sum(e["kind"] == "static" for e in ev),
        "reflex_fires": len(rfx.fired),
        "first_threat_times": sorted(first_threat.values())[:6],
        "launch_times": [e["t"] for e in launched],
    }


if __name__ == "__main__":
    seeds = range(100, 100 + int(sys.argv[1]) if len(sys.argv) > 1 else 112)
    R = []
    t0 = time.time()
    for s in seeds:
        for on in (False, True):
            r = episode(s, on)
            R.append(r)
            print(
                {
                    k: r[k]
                    for k in ("seed", "reflex", "throw_speeds", "throw_hits", "mover_hits", "reflex_fires")
                },
                flush=True,
            )
    for on in (False, True):
        sub = [r for r in R if r["reflex"] == on]
        print(
            "reflex" if on else "none  ",
            "stone hits",
            sum(r["throw_hits"] for r in sub),
            "/",
            sum(r["throws"] for r in sub),
            "| bird hits",
            sum(r["mover_hits"] for r in sub),
            "| static hits",
            sum(r["static_hits"] for r in sub),
            "| reflex fires",
            sum(r["reflex_fires"] for r in sub),
        )
    (OUT / "REFLEX_TEST.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print("wall", round(time.time() - t0))
