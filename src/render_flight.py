"""Supplementary videos of the paper, step 1.
usage: render_flight.py PLAN.jsonl I VARIANT OUT_DIR [EVERY=2]"""

import json, math, sys, time
from pathlib import Path
import numpy as np

FEELS = ("danger_05", "danger_1", "danger_2", "gps_bad", "lost", "low_energy", "strain")


def main(plan, i, variant, out_dir, every=2):
    import mujoco, mission_env as ME, city3d as C, missions as M, nav as NV, safety as SF, mission_brain as MB, perceive as PC, mission_score as MS, eval_missions as EM
    from PIL import Image
    from nav import rot

    r = next(json.loads(l) for l in open(plan, encoding="utf-8") if json.loads(l)["i"] == i)
    tag = f"{r.get('tag', 'G')}{i}_{variant}"
    out = Path(out_dir) / tag
    (out / "chase").mkdir(parents=True, exist_ok=True)
    (out / "gimbal").mkdir(exist_ok=True)
    W0, _, _ = C.build(r["place"], r["seed"], actors=False)
    mis = EM.mk(M, W0, r)
    faults = (
        [{"sensor": "gps", "kind": "off", "t0": r["gps_off"], "t1": 1e9, "sev": 1.0}] if r["gps_off"] else []
    )
    env = ME.MissionEnv(
        r["place"],
        r["seed"],
        mis,
        render=False,
        cam=(160, 120),
        gimbal=(960, 720),
        wind=tuple(r["wind"]),
        gust=r["gust"],
        faults=faults,
        encounters=(dict(EM.ATTACKS) if r["attacks"] else None),
        battery0=float(r.get("battery0") or 100.0),
    )
    models = Path(__file__).resolve().parents[1] / "models"
    perc0 = PC.Perceiver(str(models / "detector.pt"), str(models / "attrs_best.pt"))
    looks = []

    def perc(rgb, depth, cam, fovy):
        dets = perc0(rgb, depth, cam, fovy)
        n = len(looks)
        Image.fromarray(rgb).save(out / "gimbal" / f"{n:05d}.jpg", quality=88)
        looks.append(
            {
                "n": n,
                "t": round(env.t, 2),
                "boxes": [
                    [round(v, 1) for v in d["box"]] + [d["cls"], round(d["conf"], 2)]
                    for d in dets
                    if "box" in d
                ],
            }
        )
        return dets

    parsed = r["truth"] if variant == "operator" else r["parsed"]
    saf = EM.ReflexSafety(env) if variant == "reflex" else SF.Safety(env, seed=r["seed"])
    nav = NV.Navigator(env.W.map, env.home, seed=r["seed"])
    notes = []

    def operator(step, cands):
        o = next(o for o in mis["objects"] if o.oid == mis["steps"][brain.si]["target"])
        hit = [
            k
            for k, (p_, h, _) in enumerate(cands)
            if math.hypot(h["x"] - o.x, h["y"] - o.y) < 3.0 + 0.5 * h["size"] + o.radius
        ]
        notes.append(
            f"[{env.t:6.1f}s] operator answered: candidate {hit[0] + 1}"
            if hit
            else f"[{env.t:6.1f}s] operator answered: none of these"
        )
        return hit[0] if hit else -1

    brain = MB.Brain(
        env,
        env.W.map,
        parsed,
        [tuple(p) for p in r["rough"]],
        perc,
        safety=saf,
        nav=nav,
        slow=EM.SLOW[variant],
        ask=operator if variant == "operator" else None,
        log=notes.append,
    )
    o = env.reset()
    env.gimbal_frame(depth=False)
    R = env.grend  # one renderer per process
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.bid
    cam.distance = 46.0
    cam.elevation = -38.0
    cam.azimuth = 135.0
    W = env.W
    json.dump(
        {
            "tag": tag,
            "variant": variant,
            "text": r["text"],
            "lang": r.get("lang"),
            "place": r["place"],
            "parsed": parsed,
            "truth": r["truth"],
            "rough": r["rough"],
            "home": list(env.home[:2]),
            "buildings": [
                [[round(x, 1), round(y, 1)] for x, y in b["poly"].exterior.coords] for b in W.map["buildings"]
            ],
            "heights": [round(b["h"], 1) for b in W.map["buildings"]],
            "roads": [
                [
                    [round(float(x), 1), round(float(y), 1)]
                    for x, y in (rd["line"].coords if hasattr(rd["line"], "coords") else rd["line"])
                ]
                for rd in W.map["roads"]
            ],
            "objects": [
                {
                    "oid": ob.oid,
                    "kind": ob.kind,
                    "x": round(ob.x, 1),
                    "y": round(ob.y, 1),
                    "r": round(ob.radius, 1),
                    "colour": ob.facts.get("colour"),
                }
                for ob in mis["objects"]
            ],
            "targets": [s["target"] for s in mis["steps"]],
            "actions": [s["action"] for s in mis["steps"]],
            "gps_off": r["gps_off"],
            "attacks": r["attacks"],
            "wind": r["wind"],
        },
        open(out / "meta.json", "w"),
        ensure_ascii=False,
    )
    steps = open(out / "steps.jsonl", "w")
    k = 0
    track = []
    t0 = time.time()
    nf = 0
    while env.t < EM.T_MAX and not brain.done:
        cmd = brain.act(o)
        o = env.step(cmd)
        k += 1
        p = env.data.qpos[:3]
        track.append((env.t, *map(float, p)))
        rec = {
            "t": round(env.t, 2),
            "true": [round(float(v), 2) for v in p],
            "est": [round(float(v), 2) for v in brain.pos],
            "state": brain.state,
            "si": brain.si,
            "gps": None if o.get("gps") is None else [round(o["gps"]["x"], 1), round(o["gps"]["y"], 1)],
            "gps_trust": round(float(nav.gps_trust), 2),
            "battery": round(float(env.battery), 1),
            "wind": [round(float(v), 1) for v in o["wind"]["vec"][:2]],
            "speed": round(float(np.linalg.norm(env.data.qvel[:3])), 2),
            "range_down": o["range_down"],
            "feel": {kk: round(float((getattr(saf, "feel", {}) or {}).get(kk, -1)), 3) for kk in FEELS},
            "p_keep": round(float(getattr(saf, "p_keep", -1)), 3),
            "fires": int(saf.fires),
            "static_hits": sum(h["kind"] == "static" for h in env.hit_events),
            "bird_hits": sum(h["kind"] in ("mover", "encounter") for h in env.hit_events),
            "looks": len(looks),
            "notes": len(notes),
            "cmd": [round(cmd["vx"], 2), round(cmd["vy"], 2), round(cmd["vz"], 2)],
            "cargo": env.cargo["state"],
            "hook": round(float(env.hook["L"]), 2),
        }
        if k % every == 0:
            R.update_scene(env.data, camera=cam, scene_option=env.vopt)
            Image.fromarray(R.render()).save(out / "chase" / f"{nf:05d}.jpg", quality=90)
            rec["frame"] = nf
            nf += 1
            att = o["attitude"]
            P = np.asarray(o["lidar_points"])
            P = (
                P[:: max(1, len(P) // 700)] @ rot(att["roll"], att["pitch"], att["yaw"]).T
                if len(P)
                else np.zeros((0, 3))
            )
            rec["lidar"] = [
                [round(float(a), 1), round(float(b), 1), round(float(c), 1)] for a, b, c in P[:700]
            ]
        cam.azimuth = (135.0 + 3.0 * env.t) % 360.0
        steps.write(json.dumps(rec) + "\n")
    steps.close()
    res = MS.score(env, mis, track)
    json.dump(
        {
            "steps": res["steps"],
            "wrong_target": res["wrong_target"],
            "collisions": res["collisions"],
            "bird_hits": res["bird_hits"],
            "battery": res["battery"],
            "notes": notes,
            "looks": looks,
            "cargo": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in env.cargo.items()},
            "wall": round(time.time() - t0, 1),
        },
        open(out / "result.json", "w"),
        ensure_ascii=False,
        default=str,
    )
    env.close()
    print(
        tag,
        [s["ok"] for s in res["steps"]],
        "contacts",
        res["collisions"],
        "frames",
        nf,
        "wall",
        round(time.time() - t0),
        flush=True,
    )


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]), a[3], a[4], int(a[5]) if len(a) > 5 else 2)
