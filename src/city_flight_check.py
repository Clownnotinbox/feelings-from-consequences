"""Stage F step 1 check.
usage: city_flight_check.py PLACE OUT.png [seed]"""

import math, sys, time
import numpy as np
import mujoco
from PIL import Image, ImageDraw
import city3d as C, mission_objects as MO
from drone_env import DroneEnv


def place_objects(W, rng, kinds=MO.KINDS, r0=18.0, r1=70.0):
    objs = []
    hx, hy = W.start
    for k, kind in enumerate(kinds):
        for _ in range(4000):
            a, r = rng.uniform(0, 2 * math.pi), rng.uniform(r0, r1)
            x, y = hx + r * math.cos(a), hy + r * math.sin(a)
            need = {"hangar": 9.0, "container": 8.0, "dropzone": 8.0, "mast": 4.0, "solar": 7.0}.get(
                kind, 4.0
            )
            if not W.free(x, y, need) or any(
                (x - a_) ** 2 + (y - b_) ** 2 < (need + 2) ** 2 for a_, b_, _ in W.trees
            ):
                continue
            if (
                any(math.hypot(x - o.x, y - o.y) < need + o.radius + 3 for o in objs)
                or math.hypot(x - hx, y - hy) < need + 3
            ):
                continue
            if (
                W.road_area is not None
                and W.road_area.distance(__import__("shapely.geometry", fromlist=["Point"]).Point(x, y))
                < need
            ):
                continue
            objs.append(MO.make(f"obj{k}", kind, x, y, rng.uniform(0, 2 * math.pi), 0.0, rng))
            break
    return objs


def main():
    place, out = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rng = np.random.default_rng(seed)
    W, static, assets = C.build(place, seed)
    objs = place_objects(W, rng)
    W.objects = objs
    xml = C.to_mjcf(W, static + "".join(o.xml for o in objs), assets)
    files = C.texture_files()
    env = DroneEnv(seed=seed, prebuilt=(W, xml, files), cam=(640, 480))
    m, d = env.model, env.data
    print(
        place,
        "start",
        W.start,
        "objects",
        [(o.kind, o.facts.get("colour")) for o in objs],
        "movers",
        len(W.movers),
        "birds",
        sum(mv["kind"] == "bird" for mv in W.movers),
        flush=True,
    )
    t0 = time.time()
    frames = []
    R = mujoco.Renderer(m, 480, 640)
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    tgt = objs[3] if len(objs) > 3 else objs[0]

    def shot(label, cam):
        R.update_scene(d, camera=cam, scene_option=opt)
        R.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        im = Image.fromarray(R.render().copy())
        ImageDraw.Draw(im).text((6, 6), label, fill=(255, 255, 255))
        frames.append(im)

    for k in range(260):  # climb to ~25 m, then fly towards one object
        p = d.qpos[:3]
        dx, dy = tgt.x - p[0], tgt.y - p[1]
        dist = math.hypot(dx, dy)
        vz = np.clip(25 - p[2], -2, 3)
        vh = min(6.0, dist / 4) if p[2] > 15 else 0.0
        env.step(
            {
                "vx": vh * dx / max(dist, 1e-6),
                "vy": vh * dy / max(dist, 1e-6),
                "vz": float(vz),
                "yaw_rate": 0.0,
            },
            sense=False,
        )
        if k in (60, 160, 255):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = env.bid
            cam.distance = 4.0
            cam.azimuth = 200
            cam.elevation = -18
            shot(f"chase t={env.t:.0f}s alt={p[2]:.0f}m", cam)
    shot("onboard front camera", "front")
    for o in objs[:6]:  # close-ups of mission objects
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (o.x, o.y, o.height / 2)
        cam.distance = max(8.0, o.radius * 4 + o.height)
        cam.azimuth = 130
        cam.elevation = -28
        shot(
            f'{o.kind} {o.facts.get("colour", "")} {o.facts.get("size_m", o.facts.get("height_m", ""))} m',
            cam,
        )
    birds = [i for i, mv in enumerate(W.movers) if mv["kind"] == "bird"]
    for i in birds[:3]:
        b = m.body(f"mover{i}").mocapid[0]
        bp = d.mocap_pos[b].copy()
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = bp
        cam.distance = 2.2
        cam.azimuth = 60
        cam.elevation = -15
        shot(f'bird {W.movers[i]["species"]} {W.movers[i]["behaviour"]}', cam)
    R.close()
    env.close()
    cols = 4
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (640 * cols, 480 * rows))
    for k, im in enumerate(frames):
        sheet.paste(im, ((k % cols) * 640, (k // cols) * 480))
    sheet.save(out)
    print("saved", out, f"{time.time() - t0:.1f}s", "collisions", env.collisions, "hits", env.hit_events[:5])


if __name__ == "__main__":
    main()
