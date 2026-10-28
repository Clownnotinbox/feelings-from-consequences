"""Stage F perception data.
usage: percep_data.py OUT_DIR SPLIT N_SEEDS FRAMES_PER_WORLD WORKERS [SEED0]"""

import json, math, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

CLASSES = [
    "pad",
    "dropzone",
    "pickup",
    "hangar",
    "water_tower",
    "mast",
    "tent",
    "container",
    "solar",
    "windsock",
    "bird",
]
KIND_W = {
    "pad": 5,
    "dropzone": 3,
    "pickup": 3,
    "hangar": 2,
    "water_tower": 1.5,
    "mast": 1.2,
    "tent": 2,
    "container": 2.5,
    "solar": 1.5,
    "windsock": 1.2,
}
ATTRS = (
    "colour",
    "mark",
    "shape",
    "pattern",
    "style",
    "door_open",
    "parcel_colour",
    "lights",
    "flags",
    "rows",
)
W_IMG, H_IMG, FOVY = 1280, 960, 70.0


def fill_scene(W, rng):
    """25-40 mission objects in groups (look-alikes together, like mission scenes) + a few rooftop pads."""
    import missions as M

    S = M.Scene(W, rng)
    kinds = list(KIND_W)
    p = np.array([KIND_W[k] for k in kinds])
    p /= p.sum()
    for g in range(int(rng.integers(5, 9))):
        k = str(rng.choice(kinds, p=p))
        c = S.put(k)
        if c is None:
            continue
        for _ in range(int(rng.integers(1, 5))):
            k2 = k if rng.random() < 0.5 else str(rng.choice(kinds, p=p))
            kw = {}
            if k2 == k and "colour" in c.facts and k not in ("solar", "windsock") and rng.random() < 0.6:
                kw["colour"] = c.facts["colour"]
            S.put(k2, near=(c.x, c.y), r=(8, 45), **kw)
    for _ in range(int(rng.integers(1, 4))):
        S.put("pad", roof=True)
    for _ in range(int(rng.integers(0, 3))):
        S.put(str(rng.choice(["solar", "mast"])), roof=True) if False else None
    return S.objs


def owner_map(m, objs, W):
    """geom id -> label index (0 = background; 1.. = objects; then birds)."""
    import mujoco

    lab = np.zeros(m.ngeom + 1, np.int32)
    names = {o.oid: k + 1 for k, o in enumerate(objs)}
    birds = {}
    for i, mv in enumerate(W.movers):
        if mv["kind"] == "bird":
            birds[i] = len(objs) + 1 + len(birds)
    for g in range(m.ngeom):
        n = m.geom(g).name
        if not n:
            continue
        pre = n.split("_")[0]
        if pre in names:
            lab[g] = names[pre]
        elif pre.startswith("mover"):
            try:
                i = int(pre[5:])
            except ValueError:
                continue
            if i in birds:
                lab[g] = birds[i]
    return lab, birds


def camera_pose(W, objs, rng):
    """A drone-like viewpoint at an object (not always centred), outside every building, above the ground."""
    for _ in range(200):
        o = objs[int(rng.integers(len(objs)))]
        d = float(np.exp(rng.uniform(np.log(8), np.log(180))))
        h = float(rng.uniform(5, 90))
        if h > d:
            h = d * rng.uniform(0.3, 0.95)
        az = rng.uniform(0, 2 * math.pi)
        hd = math.sqrt(max(d * d - h * h, 1.0))
        cx, cy, cz = o.x + hd * math.cos(az), o.y + hd * math.sin(az), o.z + h
        if abs(cx) > W.size / 2 - 5 or abs(cy) > W.size / 2 - 5:
            continue
        from shapely.geometry import Point

        q = Point(cx, cy)
        if any(f["poly"].buffer(2).contains(q) and f["top"] + 2 > cz for f in W.footprints):
            continue
        if any((cx - a) ** 2 + (cy - b) ** 2 < 16 and cz < hh + 2 for a, b, hh in W.trees):
            continue
        look = np.array([o.x, o.y, o.z + min(o.height, 6) / 2]) + rng.normal(
            0, [d * 0.12, d * 0.12, d * 0.05]
        )
        v = look - np.array([cx, cy, cz])
        dist = float(np.linalg.norm(v))
        return look, dist, math.degrees(math.atan2(-v[1], -v[0])), math.degrees(math.asin(v[2] / dist))
    return None


def job(args):
    place, split, seed, nf, out = args
    import mujoco
    from PIL import Image
    import city3d as C, actors as AC

    out = Path(out)
    rng = np.random.default_rng(zlib.crc32(f"percep/{place}/{seed}".encode()))
    W, static, assets = C.build(place, seed)
    objs = fill_scene(W, rng)
    W.objects = objs  # pickup parcels are separate bodies
    m = mujoco.MjModel.from_xml_string(
        C.to_mjcf(W, static + "".join(o.xml for o in objs), assets), C.texture_files()
    )
    d = mujoco.MjData(m)
    m.vis.global_.fovy = FOVY
    lab_of, birds = owner_map(m, objs, W)
    bird_ids = list(birds)
    R = mujoco.Renderer(
        m, H_IMG, W_IMG, max_geom=30000
    )  # colour: 4x multisampling (set in the model at context creation)
    m.vis.quality.offsamples = 0
    Rs = mujoco.Renderer(m, H_IMG, W_IMG, max_geom=30000)
    m.vis.quality.offsamples = 4  # segmentation: no multisampling, else edge ids blend
    Rs.enable_segmentation_rendering()
    rows, crops = [], []
    for f in range(nf):
        t = float(rng.uniform(0, 300))
        pose = camera_pose(W, objs, rng)
        if pose is None:
            continue
        look, dist, az, el = pose
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        cpos = (
            look
            - dist
            * np.array(
                [
                    math.cos(math.radians(el)) * math.cos(math.radians(az)) * -1,
                    math.cos(math.radians(el)) * math.sin(math.radians(az)) * -1,
                    math.sin(math.radians(el)),
                ]
            )
            * -1
        )
        for i, mv in enumerate(W.movers):  # movers at time t; some birds brought in front of the camera
            P = AC.pose(mv, t, None, {})
            mid = m.body(f"mover{i}").mocapid[0]
            pos, q = P[""]
            if mv["kind"] == "bird" and rng.random() < 0.35:
                fwd = look - cpos
                fwd /= np.linalg.norm(fwd)
                r_ = float(np.exp(rng.uniform(np.log(3), np.log(45))))
                pos = cpos + fwd * r_ + rng.normal(0, r_ * 0.25, 3)
                pos[2] = max(pos[2], 2.0)
            d.mocap_pos[mid], d.mocap_quat[mid] = pos, q
            if mv["kind"] == "bird":
                for s in "lr":
                    w = m.body(f"mover{i}_w{s}").mocapid[0]
                    d.mocap_pos[w] = pos
                    d.mocap_quat[w] = P["w" + s][1] if "fold" not in P else q
        d.qpos[:3] = [cpos[0], cpos[1], max(cpos[2], 1.0) + 4.0]
        mujoco.mj_forward(m, d)  # the drone 4 m above the camera
        fog = rng.random() < 0.5
        vis = float(rng.uniform(300, 3000))
        m.vis.map.fogstart, m.vis.map.fogend = 0.15 * vis / m.stat.extent, vis / m.stat.extent
        R.update_scene(d, camera=cam)
        R.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = fog
        rgb = R.render().copy()
        Rs.update_scene(d, camera=cam)
        seg = Rs.render().copy()
        gid = seg[..., 0]
        lab = np.where(
            (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0), lab_of[np.clip(gid, 0, m.ngeom)], 0
        )
        name = f"{place}_{seed}_{f}"
        lines = []
        for k in np.unique(lab):
            if k == 0:
                continue
            ys, xs = np.nonzero(lab == k)
            n = len(xs)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            if n < 30 or x1 - x0 < 4 or y1 - y0 < 4:
                continue
            if k <= len(objs):
                o = objs[k - 1]
                cls = CLASSES.index(o.kind)
                info = {
                    "kind": o.kind,
                    **{a: o.facts.get(a) for a in ATTRS if o.facts.get(a) is not None},
                    "oid": o.oid,
                }
            else:
                cls = CLASSES.index("bird")
                info = {"kind": "bird", "species": W.movers[bird_ids[k - len(objs) - 1]]["species"]}
            lines.append(
                f"{cls} {(x0 + x1 + 1) / 2 / W_IMG:.5f} {(y0 + y1 + 1) / 2 / H_IMG:.5f} {(x1 - x0 + 1) / W_IMG:.5f} {(y1 - y0 + 1) / H_IMG:.5f}"
            )
            if (
                cls != CLASSES.index("bird") and min(x1 - x0, y1 - y0) >= 20
            ):  # attribute crop, 20 % margin, 112 px
                mx, my = int((x1 - x0) * 0.2) + 2, int((y1 - y0) * 0.2) + 2
                c = rgb[max(0, y0 - my) : y1 + my, max(0, x0 - mx) : x1 + mx]
                cn = f"{name}_{k}"
                Image.fromarray(c).resize((112, 112)).save(out / "crops" / split / f"{cn}.jpg", quality=92)
                crops.append({"crop": cn, "px": int(n), "box_min": int(min(x1 - x0, y1 - y0)), **info})
        Image.fromarray(rgb).resize((960, 720), Image.BILINEAR).save(
            out / "images" / split / f"{name}.jpg", quality=90
        )  # detector input size (labels are normalised)
        (out / "labels" / split / f"{name}.txt").write_text(chr(10).join(lines))
        rows.append(
            {"img": name, "n": len(lines), "fog": fog, "vis": vis, "dist": round(dist, 1), "el": round(el, 1)}
        )
    R.close()
    Rs.close()
    return rows, crops


def main():
    out, split, n_seeds, nf, workers = (
        Path(sys.argv[1]),
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
    )
    s0 = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    from osm_fetch2 import AREAS

    places = [
        p
        for p, a in AREAS.items()
        if a[2] == split or (split == "test" and a[2] == "test_candidate" and p == "kazan_novosav")
    ]
    import city3d as C

    places = [p for p in places if (C.OSM2 / f"{p}.json").exists()]
    for sub in ("images", "labels", "crops"):
        (out / sub / split).mkdir(parents=True, exist_ok=True)
    jobs = [(p, split, s0 + s, nf, str(out)) for s in range(n_seeds) for p in places]
    t0 = time.time()
    R, Cr = [], []
    with ProcessPoolExecutor(workers) as ex:
        for k, (rows, crops) in enumerate(ex.map(job, jobs)):
            R += rows
            Cr += crops
            if (k + 1) % 10 == 0:
                print(
                    f"{split} {k + 1}/{len(jobs)} worlds {len(R)} frames {len(Cr)} crops {time.time() - t0:.0f}s",
                    flush=True,
                )
    with open(out / f"frames_{split}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in R)
    with open(out / f"crops_{split}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, default=str) + "\n" for c in Cr)
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {CLASSES}\n"
    )
    print("done", split, places, len(R), "frames", len(Cr), "crops", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
