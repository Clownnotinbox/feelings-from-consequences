"""Stage G1a (MISSIONS_DESIGN_RU.md 6.6).
usage: test_door.py OUT.json N"""

import json, math, sys, zlib
import numpy as np


def scan(env, h, n=72, R=None, Z=6.0):
    import mujoco
    from nav import rot

    R = R or 0.5 * h.facts["size_m"] + 7.0
    P, rays = [], []
    for k in range(n):
        a = 2 * math.pi * k / n
        x, y = h.x + R * math.cos(a), h.y + R * math.sin(a)
        yaw = math.atan2(h.y - y, h.x - x)
        env.data.qpos[:3] = (x, y, Z)
        env.data.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env._lidar_cache = env._lidar3d(env.data.qpos[:3].copy(), env.data.xmat[env.bid].reshape(3, 3).copy())
        o = env.observe(images=False)
        att = o["attitude"]
        pos = env.data.qpos[:3].copy()
        pts = o["lidar_points"] @ rot(att["roll"], att["pitch"], att["yaw"]).T + pos
        sel = (
            (np.hypot(pts[:, 0] - h.x, pts[:, 1] - h.y) < 0.9 * h.facts["size_m"] + 3.0)
            & (pts[:, 2] > 0.3)
            & (pts[:, 2] < 5.5)
        )
        if sel.any():
            P.append(pts[sel][::2])
        near = (np.hypot(pts[:, 0] - h.x, pts[:, 1] - h.y) < 0.75 * h.facts["size_m"] + 1.0) & (
            pts[:, 2] < 5.5
        )
        if near.any():
            rays.append((pos, pts[near][::3]))
    return (np.concatenate(P) if P else np.zeros((0, 3))), rays


def main(out, n):
    import city3d as C, missions as M, mission_env as ME, mission_brain as MB, test_actions as TA

    rows = []
    for k in range(n):
        rng = np.random.default_rng(zlib.crc32(f"acts/enter/{k}".encode()))
        place = TA.PLACES[(k + zlib.crc32(b"enter") % 7) % len(TA.PLACES)]
        seed = 7000 + k
        W0, _, _ = C.build(place, seed, actors=False)
        mis = M.make(W0, rng, action="enter")
        if mis is None:
            continue
        env = ME.MissionEnv(place, seed, mis, render=False, cam=(160, 120), gimbal=(320, 240))
        env.reset()
        for h in [o for o in mis["objects"] if o.kind == "hangar"]:
            P, rays = scan(env, h)
            hint = (
                h.x + float(np.random.default_rng(k).normal(0, 0.8)),
                h.y + float(np.random.default_rng(k + 1).normal(0, 0.8)),
            )  # the brain's centre is a camera estimate
            truth_n = h.door["normal"]
            is_open = bool(h.door["open"])
            res = {}
            for name, kw in (("old", {}), ("new", {"rays": rays})):
                d = MB.Brain._door_from_walls(P, hint, h.facts["size_m"], **kw)
                if d is None:
                    res[name] = {"found": False, "right_side": None}
                else:
                    (cx, cy), (nx, ny), gap, depth, (dx, dy) = d
                    side_ok = nx * truth_n[0] + ny * truth_n[1] > 0.9
                    res[name] = {
                        "found": True,
                        "right_side": bool(side_ok),
                        "gap": round(gap, 2),
                        "door_xy_err": round(
                            math.hypot(dx - h.door["centre"][0], dy - h.door["centre"][1]), 2
                        ),
                    }
            rows.append(
                {
                    "k": k,
                    "place": place,
                    "oid": h.oid,
                    "open": is_open,
                    "door_w": h.door["width"],
                    "size": h.facts["size_m"],
                    "n_pts": len(P),
                    **{f"{a}_{b}": v for a in res for b, v in res[a].items()},
                }
            )
            print(rows[-1], flush=True)
        env.close()
    summ = {}
    for name in ("old", "new"):
        op = [r for r in rows if r["open"]]
        cl = [r for r in rows if not r["open"]]
        summ[name] = {
            "open_hangars": len(op),
            "open_right_door": sum(bool(r[f"{name}_found"] and r[f"{name}_right_side"]) for r in op),
            "open_wrong_side": sum(bool(r[f"{name}_found"] and not r[f"{name}_right_side"]) for r in op),
            "open_missed": sum(not r[f"{name}_found"] for r in op),
            "closed_hangars": len(cl),
            "closed_false_door": sum(bool(r[f"{name}_found"]) for r in cl),
        }
    json.dump({"rows": rows, "summary": summ}, open(out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
