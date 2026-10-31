"""Stage E, step 1 (plan A–E sec.
usage: stage_e_scenes.py N WORKERS OUT_DIR [START]"""

import json, math, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

W_IMG, H_IMG, FOVY = 672, 504, 70  # multiples of 28 (no resizing in the VLM's processor)
ZOOM, CROP = (
    int(__import__("os").environ.get("E_ZOOM", "0")),
    336,
)  # high-resolution factor (0 = off); close-up side, px
SHAPES = ("tent", "pad", "tower", "container")  # civil goals only; vehicles and people are never goals
COLORS = ("red", "yellow", "blue", "orange", "purple", "white")
SCALABLE = ("tent", "pad", "container")
FAM = {"tent": "tent", "pad": "landing pad", "tower": "water tower", "container": "container"}
NOV_COLOR = {
    "red": ("crimson", "scarlet"),
    "yellow": ("golden", "mustard-coloured"),
    "blue": ("cobalt", "navy"),
    "orange": ("tangerine", "amber"),
    "purple": ("violet", "plum-coloured"),
    "white": ("snow-white", "ivory"),
}
NOV_TYPE = {
    "tent": ("camping shelter", "canvas shelter"),
    "pad": ("helipad", "H-marked touchdown circle"),
    "tower": ("elevated tank", "spherical tank on a mast"),
    "container": ("shipping box", "cargo crate"),
}
NOV_SIZE = {"large": ("big", "full-size"), "small": ("little", "compact")}
NOV_BLD = {"tall": ("the high-rise", "a high-rise"), "low": ("a squat block", "a short block")}
NOV_SIDE = {
    "north": "on the northern side of",
    "south": "on the southern side of",
    "east": "on the sunrise side of",
    "west": "on the sunset side of",
}
HEADINGS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")
NEAR, FAR, OPEN, NB_NEAR, NB_FAR = (
    7.0,
    12.0,
    15.0,
    9.0,
    15.0,
)  # relation is true within NEAR, false beyond FAR (between: unclear)


def radius(c):
    return {"pad": 2.5, "tent": 2.4, "container": 3.3, "tower": 1.6}[c["shape"]] * (
        c["scale"] if c["shape"] in SCALABLE else 1.0
    )


def cand_geoms(c, j):
    import world

    x, y = c["pos"]
    s = c["scale"]
    col = world.COLORS[c["color"]] + " 1"
    n = f"c{j}"
    if c["shape"] == "tent":
        return (
            f'<geom name="{n}_a" type="box" pos="{x:.2f} {y - .9 * s:.2f} {1.1 * s:.2f}" size="{2. * s:.2f} {1.25 * s:.2f} .05" euler="45 0 0" rgba="{col}"/>'
            f'<geom name="{n}_b" type="box" pos="{x:.2f} {y + .9 * s:.2f} {1.1 * s:.2f}" size="{2. * s:.2f} {1.25 * s:.2f} .05" euler="-45 0 0" rgba="{col}"/>'
        )
    if c["shape"] == "pad":
        return (
            f'<geom name="{n}_a" type="cylinder" pos="{x:.2f} {y:.2f} .03" size="{2.5 * s:.2f} .03" rgba="{col}"/>'
            f'<geom name="{n}_b" type="box" pos="{x:.2f} {y:.2f} .07" size="{.9 * s:.2f} {.12 * s:.2f} .01" rgba="1 1 1 1"/>'
            f'<geom name="{n}_c" type="box" pos="{x - .8 * s:.2f} {y:.2f} .07" size="{.12 * s:.2f} {1. * s:.2f} .01" rgba="1 1 1 1"/>'
            f'<geom name="{n}_d" type="box" pos="{x + .8 * s:.2f} {y:.2f} .07" size="{.12 * s:.2f} {1. * s:.2f} .01" rgba="1 1 1 1"/>'
        )
    if c["shape"] == "tower":
        return (
            f'<geom name="{n}_a" type="cylinder" pos="{x:.2f} {y:.2f} 3" size=".35 3" rgba=".4 .4 .4 1"/>'
            f'<geom name="{n}_b" type="sphere" pos="{x:.2f} {y:.2f} 6.8" size="1.6" rgba="{col}"/>'
        )
    return f'<geom name="{n}_a" type="box" pos="{x:.2f} {y:.2f} {1.3 * s:.2f}" size="{3. * s:.2f} {1.2 * s:.2f} {1.3 * s:.2f}" rgba="{col}"/>'


def bdist(x, y, b):
    bx, by, sx, sy = b[:4]
    return math.hypot(max(abs(x - bx) - sx / 2, 0.0), max(abs(y - by) - sy / 2, 0.0))


def side(x, y, b):
    """Side of building b the point is on, or None near a corner (unclear)."""
    bx, by, sx, sy = b[:4]
    u, v = abs(x - bx) / (sx / 2), abs(y - by) / (sy / 2)
    if 0.8 < u / max(v, 1e-6) < 1.25:
        return None
    return ("east" if x > bx else "west") if u > v else ("north" if y > by else "south")


def tall(b):
    return b[4] > 14.0


def relations(c, cands, W):
    """The goal-side values of each attribute (None when it cannot be stated clearly)."""
    x, y = c["pos"]
    v = {
        "type": c["shape"],
        "colour": c["color"],
        "size": ("large" if c["scale"] > 0.8 else "small") if c["shape"] in SCALABLE else None,
    }
    d = [(bdist(x, y, b), b) for b in W.buildings]
    dn, bn = min(d, key=lambda t: t[0]) if d else (1e9, None)
    v["bctx"] = ("near_tall" if tall(bn) else "near_low") if dn <= NEAR else "open" if dn > OPEN else None
    s = side(x, y, bn) if dn <= NEAR else None
    v["side"] = (s, "tall" if tall(bn) else "low") if s else None
    others = [(math.hypot(o["pos"][0] - x, o["pos"][1] - y), o) for o in cands if o is not c]
    v["next_to"] = None
    if others:
        do, o = min(others, key=lambda t: t[0])
        if do <= NB_NEAR and sum((q["color"], q["shape"]) == (o["color"], o["shape"]) for q in cands) == 1:
            v["next_to"] = (o["color"], o["shape"])
    return v


def holds(c, attr, val, cands, W):
    """Is 'attr = val' true (True), false (False) or unclear (None) of object c?."""
    x, y = c["pos"]
    if attr in ("type", "colour", "size"):
        return relations(c, cands, W)[attr] == val
    if attr == "bctx":
        if val == "open":
            dm = min((bdist(x, y, b) for b in W.buildings), default=1e9)
            return True if dm > OPEN else False if dm <= NEAR else None
        ds = [bdist(x, y, b) for b in W.buildings if tall(b) == (val == "near_tall")]
        dm = min(ds, default=1e9)
        return True if dm <= NEAR else False if dm > FAR else None
    if attr == "side":
        s, k = val
        B = [b for b in W.buildings if (tall(b) == (k == "tall")) and bdist(x, y, b) <= FAR]
        if any(bdist(x, y, b) <= NEAR and side(x, y, b) == s for b in B):
            return True
        return False if all(side(x, y, b) not in (s, None) for b in B) else None
    if attr == "next_to":
        ds = [
            math.hypot(o["pos"][0] - x, o["pos"][1] - y)
            for o in cands
            if o is not c and (o["color"], o["shape"]) == val
        ]
        dm = min(ds, default=1e9)
        return True if dm <= NB_NEAR else False if dm > NB_FAR else None


def describe(t, cands, W):
    """Incremental algorithm."""
    v = relations(t, cands, W)
    D = [d for d in cands if d is not t and d["shape"] == t["shape"]]
    used = {"type": v["type"]}
    for a in ("colour", "size", "bctx", "next_to", "side"):
        if not D:
            break
        if v[a] is None:
            continue
        out = [d for d in D if holds(d, a, v[a], cands, W) is False]
        if out:
            used[a] = v[a]
            D = [d for d in D if d not in out]
    return None if D else used


def phrase(used, rng, novel):
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    col = used.get("colour")
    sz = used.get("size")
    ty = used["type"]
    if novel:
        head = " ".join(
            w
            for w in (
                pick(NOV_SIZE[sz]) if sz else None,
                pick(NOV_COLOR[col]) if col else None,
                pick(NOV_TYPE[ty]),
            )
            if w
        )
    else:
        head = " ".join(w for w in (sz, col, FAM[ty]) if w)
    rel = []
    if "side" in used:
        s, k = used["side"]
        rel.append(f"{NOV_SIDE[s]} {pick(NOV_BLD[k])}" if novel else f"{s} of a {k} building")
    elif "bctx" in used:
        b = used["bctx"]
        if novel:
            rel.append(
                pick(
                    (
                        "on bare ground with nothing built around it",
                        "out on empty ground, clear of any structures",
                    )
                )
                if b == "open"
                else pick(("beside", "up against"))
                + " "
                + pick(NOV_BLD["tall" if b == "near_tall" else "low"])
            )
        else:
            rel.append(
                "in the open, away from buildings"
                if b == "open"
                else f"next to a {'tall' if b == 'near_tall' else 'low'} building"
            )
    if "next_to" in used:
        c2, s2 = used["next_to"]
        rel.append(
            (
                pick(("a stone's throw from the", "alongside the"))
                + f" {pick(NOV_COLOR[c2])} {pick(NOV_TYPE[s2])}"
            )
            if novel
            else f"next to the {c2} {FAM[s2]}"
        )
    d = "the " + head + ("" if not rel else " " + " and ".join(rel))
    if novel:
        fr = ["Head for {}.", "Your destination: {}.", "Make your way to {}."] + (
            ["Touch down on {}."] if ty == "pad" else ["Position yourself above {}."]
        )
    else:
        fr = ["Fly to {}.", "Go to {}.", "The goal is {}."] + (
            ["Land on {}."] if ty == "pad" else ["Hover over {}."]
        )
    return pick(fr).format(d)


def free(x, y, r, W, cands, half):
    if abs(x) > half or abs(y) > half:
        return False
    if any(bdist(x, y, b) < r + 1.0 for b in W.buildings):
        return False
    if any(math.hypot(x - a, y - b) < r + 1.5 for a, b, _ in W.trees):
        return False
    if any(math.hypot(x - a, y - b) < r + 3.0 for a, b, *_ in W.cars):
        return False
    return all(math.hypot(x - o["pos"][0], y - o["pos"][1]) > r + radius(o) + 2.0 for o in cands)


def place(rng, c, W, cands, centre, rmin, rmax, half, near_bld):
    """A free spot for object c: beside a building face (near_bld) or in a ring around centre."""
    r = radius(c)
    for _ in range(300):
        if near_bld and W.buildings:
            b = W.buildings[int(rng.integers(len(W.buildings)))]
            bx, by, sx, sy = b[:4]
            f = int(rng.integers(4))
            g = r + rng.uniform(1.2, 3.5)
            x, y = (
                (bx + (sx / 2 + g) * (1 if f == 0 else -1), by + rng.uniform(-sy / 2, sy / 2))
                if f < 2
                else (bx + rng.uniform(-sx / 2, sx / 2), by + (sy / 2 + g) * (1 if f == 2 else -1))
            )
            if centre is not None and not rmin <= math.hypot(x - centre[0], y - centre[1]) <= rmax:
                continue
        else:
            a, d = rng.uniform(0, 2 * math.pi), rng.uniform(rmin, rmax)
            x, y = (
                (centre[0] + d * math.cos(a), centre[1] + d * math.sin(a))
                if centre is not None
                else (rng.uniform(-half, half), rng.uniform(-half, half))
            )
        if free(x, y, r, W, cands, half):
            c["pos"] = (round(float(x), 2), round(float(y), 2))
            return True
    return False


def sample(i):
    """Scene i (stable seed)."""
    import world

    rng = np.random.default_rng(zlib.crc32(f"stageE/{i}".encode()))
    for attempt in range(60):
        W = world.generate(
            seed=int(rng.integers(1, 2**31)), n_targets=0, n_moving_cars=0, n_people=0, n_birds=0, n_divers=0
        )
        half = W.size / 2 - 2
        hard = bool(rng.random() < 0.7)
        t = {"shape": str(rng.choice(SHAPES)), "color": str(rng.choice(COLORS))}
        t["scale"] = float(rng.choice([0.6, 1.0])) if t["shape"] in SCALABLE else 1.0
        cands = []
        if not place(rng, t, W, cands, None, 0, 0, W.size / 2 - 18, bool(rng.random() < 0.55)):
            continue
        cands.append(t)
        nd = int(rng.integers(2, 6))
        ntw = (1 + int(rng.random() < 0.35)) if hard else 0
        lm = hard and rng.random() < 0.5  # a landmark right next to the goal
        for k in range(nd):
            if k < ntw:
                c = {"shape": t["shape"], "color": t["color"]}
            elif lm and k == ntw:
                taken = {(o["shape"], o["color"]) for o in cands}
                c = {
                    "shape": str(rng.choice([q for q in SHAPES if q != t["shape"]])),
                    "color": str(rng.choice(COLORS)),
                }
                if (c["shape"], c["color"]) in taken:
                    continue
            else:
                u = rng.random()
                c = {
                    "shape": t["shape"] if u < 0.4 else str(rng.choice(SHAPES)),
                    "color": t["color"] if 0.4 <= u < 0.7 else str(rng.choice(COLORS)),
                }
                if not hard and (c["shape"], c["color"]) == (t["shape"], t["color"]):
                    c["color"] = str(rng.choice([q for q in COLORS if q != t["color"]]))
            c["scale"] = (
                (t["scale"] if k < ntw and rng.random() < 0.6 else float(rng.choice([0.6, 1.0])))
                if c["shape"] in SCALABLE
                else 1.0
            )
            near = lm and k == ntw
            r0 = radius(t) + radius(c) + 2.2
            if place(
                rng,
                c,
                W,
                cands,
                t["pos"],
                r0 if near else 6.0,
                r0 + 2.5 if near else 35.0,
                half,
                False if near else bool(rng.random() < 0.5),
            ):
                cands.append(c)
        if len(cands) < 3:
            continue
        used = describe(t, cands, W)
        if used is None:
            continue
        twins = sum(1 for c in cands[1:] if (c["shape"], c["color"]) == (t["shape"], t["color"]))
        if hard and not twins:
            continue
        perm = rng.permutation(len(cands))
        cands = [cands[p] for p in perm]
        ti = int(np.flatnonzero(perm == 0)[0])
        far = bool(rng.random() < 0.15)
        e = rng.uniform(45, 90) if far else rng.uniform(5, 45)
        a = rng.uniform(0, 2 * math.pi)
        P = (t["pos"][0] + e * math.cos(a), t["pos"][1] + e * math.sin(a))
        for _ in range(60):
            hd, h, az = rng.uniform(15, 80), rng.uniform(15, 45), rng.uniform(0, 2 * math.pi)
            C = (P[0] + hd * math.cos(az), P[1] + hd * math.sin(az), h)
            if all(not (bdist(C[0], C[1], b) < 3 and b[4] + 3 > h) for b in W.buildings):
                break
        else:
            continue
        brg = (90 - math.degrees(math.atan2(P[1] - C[1], P[0] - C[0]))) % 360
        meta = {
            "i": i,
            "attempt": attempt,
            "hard": hard,
            "twins": twins,
            "used": {k: list(v) if isinstance(v, tuple) else v for k, v in used.items()},
            "err_m": round(float(e), 1),
            "far": far,
            "approx": [round(float(P[0]), 2), round(float(P[1]), 2)],
            "cam": [round(float(q), 2) for q in C],
            "heading": HEADINGS[int(round(brg / 45)) % 8],
            "bearing": round(brg, 1),
            "dist_m": round(math.hypot(t["pos"][0] - C[0], t["pos"][1] - C[1]), 1),
        }
        trng = np.random.default_rng(
            zlib.crc32(f"stageE/{i}/text".encode())
        )  # the wordings use their own streams
        meta["text_familiar"] = phrase(used, np.random.default_rng(trng.integers(2**31)), False)
        meta["text_novel"] = phrase(used, np.random.default_rng(trng.integers(2**31)), True)
        return W, cands, ti, meta
    raise RuntimeError(f"scene {i}: not describable after 60 attempts")


def camera_xml(C, P):
    f = np.array([P[0] - C[0], P[1] - C[1], -C[2]])
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return f'<camera name="eye" pos="{C[0]:.2f} {C[1]:.2f} {C[2]:.2f}" xyaxes="{r[0]:.4f} {r[1]:.4f} {r[2]:.4f} {u[0]:.4f} {u[1]:.4f} {u[2]:.4f}" fovy="{FOVY}"/>'


def render(W, cands, meta):
    import mujoco, world

    xml = world.to_mjcf(W).replace(
        'offwidth="640" offheight="480"', f'offwidth="{W_IMG}" offheight="{H_IMG}"'
    )
    xml = xml.replace(
        "<worldbody>",
        "<worldbody>"
        + camera_xml(meta["cam"], meta["approx"])
        + "".join(cand_geoms(c, j) for j, c in enumerate(cands)),
        1,
    )
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    R = mujoco.Renderer(m, H_IMG, W_IMG)
    R.update_scene(d, camera="eye")
    rgb = R.render().copy()
    R.close()
    m.vis.quality.offsamples = 0  # no multisampling for the id pass
    R = mujoco.Renderer(m, H_IMG, W_IMG)
    R.enable_segmentation_rendering()
    R.update_scene(d, camera="eye")
    seg = R.render().copy()
    R.close()
    owner = np.zeros(m.ngeom + 1, np.uint8)  # geom id -> object index + 1
    for g in range(m.ngeom):
        n = m.geom(g).name
        if n.startswith("c") and "_" in n and n[1 : n.index("_")].isdigit():
            owner[g] = int(n[1 : n.index("_")]) + 1
    gid = seg[..., 0]
    lab = np.where(
        (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0), owner[np.clip(gid, 0, m.ngeom)], 0
    ).astype(np.uint8)
    hi = None
    if ZOOM:  # the same camera at ZOOM x resolution, a separate model so the plain
        m2 = mujoco.MjModel.from_xml_string(
            xml.replace(
                f'offwidth="{W_IMG}" offheight="{H_IMG}"',
                f'offwidth="{W_IMG * ZOOM}" offheight="{H_IMG * ZOOM}"',
            )
        )
        d2 = mujoco.MjData(m2)
        mujoco.mj_forward(m2, d2)
        R = mujoco.Renderer(m2, H_IMG * ZOOM, W_IMG * ZOOM)
        R.update_scene(d2, camera="eye")
        hi = R.render().copy()
        R.close()
    return rgb, lab, hi


def closeups(hi, cands, name, out):
    """One close-up per marked object, one magnification per scene (see the module note)."""
    from PIL import Image, ImageDraw

    vis = [j for j, c in enumerate(cands) if c.get("mark")]
    if not vis:
        return {"window": None, "crops": {}}
    ext = max(
        max(cands[j]["bbox"][2] - cands[j]["bbox"][0], cands[j]["bbox"][3] - cands[j]["bbox"][1]) + 1
        for j in vis
    )
    w = int(np.clip(round(1.6 * ext), 64, 224))
    k = CROP / w
    res = {}
    for j in vis:
        x0, y0, x1, y1 = cands[j]["bbox"]
        wx = int(np.clip(round((x0 + x1) / 2 - w / 2), 0, W_IMG - w))
        wy = int(np.clip(round((y0 + y1) / 2 - w / 2), 0, H_IMG - w))
        im = Image.fromarray(hi[wy * ZOOM : (wy + w) * ZOOM, wx * ZOOM : (wx + w) * ZOOM]).resize(
            (CROP, CROP), Image.LANCZOS
        )
        dr = ImageDraw.Draw(im)
        dr.rectangle(
            [(x0 - wx) * k - 6, (y0 - wy) * k - 6, (x1 + 1 - wx) * k + 6, (y1 + 1 - wy) * k + 6],
            outline=(255, 255, 255),
            width=2,
        )
        im.save(out / "crop" / f"{name}_{cands[j]['mark']}.png")
        res[cands[j]["mark"]] = [wx, wy]
    return {"window": w, "crops": res}


def marks(rgb, lab, cands, rng):
    """Set-of-marks: a numbered circle above every visible object (numbers in random order)."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    vis = [j for j, c in enumerate(cands) if c["area"] >= 40]
    nums = rng.permutation(len(vis)) + 1
    for j, k in zip(vis, nums):
        x0, y0, x1, y1 = cands[j]["bbox"]
        cx = (x0 + x1) / 2
        ly = max(12, y0 - 16)
        lx = min(max(12, cx), W_IMG - 12)
        dr.line([(cx, y0), (lx, ly)], fill=(255, 255, 255), width=1)
        dr.ellipse([lx - 10, ly - 10, lx + 10, ly + 10], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        dr.text((lx, ly), str(int(k)), fill=(0, 0, 0), font=font, anchor="mm")
        cands[j]["mark"] = int(k)
    return im


def one(args):
    i, out = args
    from PIL import Image

    W, cands, ti, meta = sample(i)
    rgb, lab, hi = render(W, cands, meta)
    for j, c in enumerate(cands):
        ys, xs = np.nonzero(lab == j + 1)
        c["area"] = int(len(xs))
        c["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
        c["mark"] = None
        c["rel"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in relations(c, cands, W).items()}
    im = marks(rgb, lab, cands, np.random.default_rng(zlib.crc32(f"stageE/{i}/marks".encode())))
    name = f"e{i:04d}"
    Image.fromarray(rgb).save(out / "img" / f"{name}.png")
    im.save(out / "img" / f"{name}_som.png")
    Image.fromarray(lab).save(out / "seg" / f"{name}.png")
    a = cands[ti]["area"]
    meta.update(
        name=name,
        target=ti,
        target_area=a,
        status="visible" if a >= 40 else "absent" if a < 5 else "marginal",
        cands=cands,
        buildings=[[round(float(q), 2) for q in b[:5]] for b in W.buildings],
    )
    if ZOOM:
        meta["_zoom"] = dict(name=name, zoom=ZOOM, **closeups(hi, cands, name, out))
    return meta


if __name__ == "__main__":
    n, workers, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "seg").mkdir(exist_ok=True)
    (out / "crop").mkdir(exist_ok=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(one, [(i, out) for i in range(start, start + n)])):
            rows.append(r)
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{n} {round(time.time() - t0)}s", flush=True)
    zr = [r.pop("_zoom") for r in rows if "_zoom" in r]
    with open(out / "scenes.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if zr:
        with open(out / "zoom.jsonl", "w", encoding="utf-8") as f:
            for r in zr:
                f.write(json.dumps(r) + "\n")
    st = {s: sum(r["status"] == s for r in rows) for s in ("visible", "absent", "marginal")}
    print(
        "done", len(rows), st, "hard", sum(r["hard"] for r in rows), round(time.time() - t0), "s", flush=True
    )
