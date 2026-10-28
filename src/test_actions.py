"""Debug check of every mission action in flight.
usage: test_actions.py OUT.jsonl N_PER_ACTION WORKERS [actions]"""

import json, math, os, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"
REAL = True  # the trained detector + attribute net (F1); False: the truth stand-in
PLACES = (
    "berlin_marzahn",
    "spb_center",
    "malakhovka",
    "helsinki_kallio",
    "warsaw_ursynow",
    "akadem_line",
    "ekb_center",
    "msk_strogino",
    "plano_suburb",
    "kratovo",
)


def one(job):
    action, k = job
    import city3d as C, missions as M, mission_env as ME, nav as NV, safety as SF, mission_brain as MB, mission_score as MS, fly_mission as FM

    rng = np.random.default_rng(zlib.crc32(f"acts/{action}/{k}".encode()))
    place = PLACES[(k + zlib.crc32(action.encode()) % 7) % len(PLACES)]
    seed = 7000 + k
    W0, _, _ = C.build(place, seed, actors=False)
    mis = M.make(
        W0, rng, two_step=(action == "pickup+drop"), action=None if action == "pickup+drop" else action
    )
    if mis is None:
        return {"action": action, "k": k, "place": place, "error": "no mission"}
    gps_off = float(rng.uniform(30, 90)) if rng.random() < 0.3 else None
    faults = [{"sensor": "gps", "kind": "off", "t0": gps_off, "t1": 1e9, "sev": 1.0}] if gps_off else []
    env = ME.MissionEnv(
        place,
        seed,
        mis,
        render=False,
        cam=(160, 120),
        gimbal=(960, 720),
        wind=tuple(rng.normal(0, 2.5, 2)),
        gust=float(rng.uniform(0, 1.5)),
        faults=faults,
    )
    global _PERC
    if REAL and "_PERC" not in globals():
        import perceive as PC

        _PERC = PC.Perceiver(str(MODELS / "detector.pt"), str(MODELS / "attrs_best.pt"))
    notes = []
    perc = _PERC if REAL else FM.TruthPerceiver(env, np.random.default_rng(k))
    saf = SF.Safety(env)
    nav = NV.Navigator(env.W.map, env.home, seed=seed)
    asks = []

    def operator(step, cands):  # sees the candidates' views and knows what was meant
        o = next(o for o in mis["objects"] if o.oid == mis["steps"][brain.si]["target"])
        asks.append(len(cands))
        hit = [
            j
            for j, (p_, h, _) in enumerate(cands)
            if math.hypot(h["x"] - o.x, h["y"] - o.y) < 3.0 + 0.5 * h["size"] + o.radius
        ]
        return hit[0] if hit else -1

    brain = MB.Brain(
        env,
        env.W.map,
        mis["truth"],
        mis["rough"],
        perc,
        safety=saf,
        nav=nav,
        log=notes.append,
        ask=operator if os.environ.get("ACTS_OPERATOR") == "1" else None,
    )
    o = env.reset()
    track = []
    errs = []
    t0 = time.time()
    while env.t < 600 and not brain.done:
        o = env.step(brain.act(o))
        p = env.data.qpos[:3]
        track.append((env.t, *map(float, p)))
        errs.append(math.hypot(brain.pos[0] - p[0], brain.pos[1] - p[1]))
    stop = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
    t_end = env.t + 30.0
    while env.cargo["state"] not in ("stowed", "landed") and env.t < t_end:
        o = env.step(stop if brain.state in ("done", "landed") else brain.act(o))
    res = MS.score(env, mis, track)
    out = {
        "action": action,
        "k": k,
        "place": place,
        "gps_off": gps_off,
        "detail": mis["detail"],
        "text": mis["text"],
        "steps": res["steps"],
        "wrong": res["wrong_target"],
        "coll": res["collisions"],
        "birds": res["bird_hits"],
        "state": brain.state,
        "t": round(env.t, 1),
        "wall": round(time.time() - t0, 1),
        "battery": res["battery"],
        "nav_err": [round(float(np.mean(errs)), 2), round(float(np.max(errs)), 2)],
        "fires": saf.fires,
        "skipped": brain.skipped,
        "home_dist": res.get("home_dist"),
        "return": brain.return_home,
        "hits": env.hit_events,
        "events": env.events_m,
        "debug": getattr(brain, "debug", None),
        "targets_true": [
            [
                round(float(v), 2)
                for v in next((o.x, o.y, o.z) for o in mis["objects"] if o.oid == st["target"])
            ]
            for st in mis["steps"]
        ],
        "rejected": len(brain.rejected),
        "notes": [n for n in notes if "] -> " not in n],
        "asks": len(asks),
        "target_facts": [
            {
                k: v
                for k, v in next(o for o in mis["objects"] if o.oid == st["target"]).facts.items()
                if k in ("kind", "door_open", "size_m", "door_w")
            }
            for st in mis["steps"]
        ],
    }
    env.close()
    return out


def one_safe(job):
    try:
        return one(job)
    except Exception:
        import traceback

        return {"action": job[0], "k": job[1], "error": traceback.format_exc()[-1500:]}


if __name__ == "__main__":
    from concurrent.futures import as_completed

    out, n, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    acts = (
        sys.argv[4].split(",")
        if len(sys.argv) > 4
        else ["land", "drop", "pickup", "enter", "inspect", "flyby", "pickup+drop"]
    )
    jobs = [(a, k) for k in range(n) for a in acts]
    with ProcessPoolExecutor(workers) as ex, open(out, "w", encoding="utf-8") as f:
        for fut in as_completed(
            [ex.submit(one_safe, j) for j in jobs]
        ):  # written as they finish; one crash is one error row
            r = fut.result()
            f.write(json.dumps(r, ensure_ascii=False, default=str) + chr(10))
            f.flush()
            print(
                r["action"],
                r.get("place"),
                [s.get("ok") for s in r.get("steps", [])],
                "wrong",
                r.get("wrong"),
                "coll",
                r.get("coll"),
                r.get("state"),
                r.get("t"),
                (r.get("error") or "")[-200:],
                flush=True,
            )
