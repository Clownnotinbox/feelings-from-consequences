"""Navigation without GPS."""

import json, math, sys, time
import numpy as np


def run(place, seed, alt, side=300.0, speed=8.0, gps_off=10.0):
    import mission_env as ME, nav as NV

    env = ME.MissionEnv(
        place,
        seed,
        None,
        render=False,
        cam=(160, 120),
        faults=[{"sensor": "gps", "kind": "off", "t0": gps_off, "t1": 1e9, "sev": 1.0}],
    )
    o = env.reset()
    nav = NV.Navigator(env.W.map, env.home, seed=seed)
    h = np.array(env.home[:2], float)
    wps = [h + np.array(v) * side / 2 for v in ((1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1))]
    k = 0
    errs = []
    t0 = time.time()
    while env.t < 400 and k < len(wps):
        p = env.data.qpos[:3]
        d = wps[k] - p[:2]
        n = float(np.linalg.norm(d))
        if n < 3:
            k += 1
            continue
        v = d / n * min(speed, n) if p[2] > alt - 3 else np.zeros(2)
        o = env.step(
            {"vx": float(v[0]), "vy": float(v[1]), "vz": float(np.clip(alt - p[2], -2, 3)), "yaw_rate": 0.0}
        )
        e = nav.step(o, env.agent_dt)
        errs.append(
            (
                round(env.t, 1),
                round(float(math.hypot(e[0] - p[0], e[1] - p[1])), 2),
                round(nav.spread, 2),
                nav.matched,
            )
        )
    E = np.array([x[1] for x in errs if x[0] > gps_off])
    env.close()
    return {
        "place": place,
        "seed": seed,
        "alt": alt,
        "mean": round(float(E.mean()), 2),
        "p90": round(float(np.percentile(E, 90)), 2),
        "max": round(float(E.max()), 2),
        "end": round(float(E[-1]), 2),
        "route_m": round(4 * side),
        "wall": round(time.time() - t0),
        "trace": errs[::40],
    }


if __name__ == "__main__":
    r = run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), *(float(x) for x in sys.argv[4:]))
    print(json.dumps({k: v for k, v in r.items() if k != "trace"}))
    print("trace (t, err, spread, matched):", r["trace"])
