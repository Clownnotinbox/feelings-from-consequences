"""Stage F living world."""

import math
import numpy as np

SPECIES = {  # half-sizes of the body ellipsoid
    "pigeon": dict(
        body=(0.16, 0.06, 0.06),
        span=0.66,
        col=(0.44, 0.46, 0.52),
        wing=(0.38, 0.40, 0.46),
        hz=5.5,
        glide=0.15,
        v=(9, 14),
    ),
    "crow": dict(
        body=(0.21, 0.07, 0.07),
        span=0.95,
        col=(0.07, 0.07, 0.08),
        wing=(0.09, 0.09, 0.10),
        hz=3.6,
        glide=0.3,
        v=(8, 12),
    ),
    "gull": dict(
        body=(0.23, 0.07, 0.07),
        span=1.3,
        col=(0.95, 0.95, 0.95),
        wing=(0.62, 0.64, 0.68),
        hz=2.6,
        glide=0.6,
        v=(9, 13),
    ),
    "hawk": dict(
        body=(0.24, 0.08, 0.08),
        span=1.15,
        col=(0.46, 0.33, 0.21),
        wing=(0.38, 0.27, 0.17),
        hz=2.4,
        glide=0.75,
        v=(10, 16),
    ),
    "starling": dict(
        body=(0.09, 0.035, 0.035),
        span=0.38,
        col=(0.12, 0.12, 0.14),
        wing=(0.10, 0.10, 0.12),
        hz=9.0,
        glide=0.1,
        v=(12, 17),
    ),
}


def quat_yaw(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_roll(a):
    return np.array([math.cos(a / 2), math.sin(a / 2), 0.0, 0.0])


def quat_pitch(a):
    return np.array([math.cos(a / 2), 0.0, math.sin(a / 2), 0.0])


def make_actors(W, rng, birds=(6, 14), cars=(3, 8), people=(4, 10), flocks=(0, 2)):
    """Random movers for one world from its real roads, footways, roofs and wires."""
    half = W.size / 2 - 20
    out = []
    roads = [r["line"] for r in W.map.get("roads", []) if len(r["line"]) >= 2 and r["width"] >= 4]
    roads = [
        ln for ln in roads if sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(ln[:-1], ln[1:])) > 40
    ]
    for _ in range(int(rng.integers(*cars)) if roads else 0):
        ln = roads[int(rng.integers(len(roads)))]
        c = rng.uniform(0.1, 0.9, 3)
        out.append(
            {
                "kind": "car",
                "v2": True,
                "line": ln,
                "speed": float(rng.uniform(4, 12)),
                "phase": float(rng.random()),
                "lane": 1.7,
                "rgba": f"{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} 1",
            }
        )
    walks = [ln for ln in getattr(W, "footways", []) if len(ln) >= 2] or roads
    for _ in range(int(rng.integers(*people)) if walks else 0):
        ln = walks[int(rng.integers(len(walks)))]
        c = rng.uniform(0.1, 0.9, 3)
        out.append(
            {
                "kind": "person",
                "v2": True,
                "line": ln,
                "speed": float(rng.uniform(1.0, 1.6)),
                "phase": float(rng.random()),
                "lane": 0.4,
                "rgba": f"{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} 1",
            }
        )
    perches = []
    for f in getattr(W, "footprints", []):
        if f["h"] < 30 and f["poly"].area > 40:
            x, y = f["poly"].exterior.coords[int(rng.integers(len(f["poly"].exterior.coords)))]
            perches.append((x, y, f["top"] + 0.1))
    for x1, y1, x2, y2, h in getattr(W, "wires", []):
        u = rng.random()
        perches.append((x1 + u * (x2 - x1), y1 + u * (y2 - y1), h - 1.0))
    for _ in range(int(rng.integers(*birds))):
        sp = str(rng.choice(["pigeon", "pigeon", "crow", "crow", "gull", "hawk"]))
        S = SPECIES[sp]
        beh = str(rng.choice(["circle", "circle", "cross", "perched"]))
        if beh == "perched" and not perches:
            beh = "circle"
        mv = {
            "kind": "bird",
            "v2": True,
            "species": sp,
            "behaviour": beh,
            "speed": float(rng.uniform(*S["v"])),
            "phase": float(rng.random()),
            "seed": int(rng.integers(1 << 30)),
        }
        if beh == "circle":
            mv.update(
                center=(float(rng.uniform(-half, half)), float(rng.uniform(-half, half))),
                radius=float(rng.uniform(10, 40)),
                alt=float(rng.uniform(8, 60)),
                cw=bool(rng.random() < 0.5),
            )
        elif beh == "cross":
            a = rng.uniform(0, 2 * math.pi)
            mv.update(
                p0=(float(-math.cos(a) * half * 1.3), float(-math.sin(a) * half * 1.3)),
                dir=(math.cos(a), math.sin(a)),
                alt=float(rng.uniform(10, 70)),
                length=2.6 * half,
            )
        else:
            x, y, z = perches[int(rng.integers(len(perches)))]
            mv.update(
                perch=(float(x), float(y), float(z)),
                flush_r=float(rng.uniform(6, 15)),
                alt=float(rng.uniform(12, 40)),
            )
        out.append(mv)
    for _ in range(
        int(rng.integers(*flocks))
    ):  # a starling flock: members share a wandering path with their own
        c = (float(rng.uniform(-half, half)), float(rng.uniform(-half, half)))
        R = float(rng.uniform(30, 80))
        alt = float(rng.uniform(15, 50))
        n = int(rng.integers(8, 20))
        for k in range(n):
            out.append(
                {
                    "kind": "bird",
                    "v2": True,
                    "species": "starling",
                    "behaviour": "flock",
                    "center": c,
                    "radius": R,
                    "alt": alt,
                    "speed": 14.0,
                    "offset": tuple(float(v) for v in rng.normal(0, [3.0, 3.0, 1.5])),
                    "phase": 0.0,
                    "seed": int(rng.integers(1 << 30)),
                    "jit": float(rng.uniform(0, 6.28)),
                }
            )
    return out


def _along(line, s):
    """Point and heading at arc length s along a polyline, ping-pong at the ends."""
    seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(line[:-1], line[1:])]
    L = sum(seg) or 1.0
    s = s % (2 * L)
    back = s > L
    s = 2 * L - s if back else s
    for (a, b), l in zip(zip(line[:-1], line[1:]), seg):
        if s <= l or l == seg[-1]:
            u = min(1.0, s / max(l, 1e-6))
            hd = math.atan2(b[1] - a[1], b[0] - a[0]) + (math.pi if back else 0.0)
            return a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), hd
        s -= l
    return line[-1][0], line[-1][1], 0.0


def pose(mv, t, p_drone, st):
    """{suffix: (pos, quat)} for one mover at time t."""
    if mv["kind"] in ("car", "person"):
        x, y, hd = _along(mv["line"], mv["phase"] * 1000 + mv["speed"] * t)
        off = mv["lane"]
        x += off * math.cos(hd - math.pi / 2)
        y += off * math.sin(hd - math.pi / 2)
        return {"": (np.array([x, y, 0.0]), quat_yaw(hd))}
    S = SPECIES[mv["species"]]
    b = mv["behaviour"]
    climb = 0.0
    if b == "circle":
        r = mv["radius"]
        sgn = -1 if mv["cw"] else 1
        a = mv["phase"] * 6.283 + sgn * mv["speed"] / r * t
        cx, cy = mv["center"]
        p = np.array(
            [
                cx + r * math.cos(a),
                cy + r * math.sin(a),
                mv["alt"] + 1.5 * math.sin(0.4 * t + 6 * mv["phase"]),
            ]
        )
        yaw = a + sgn * math.pi / 2
    elif b == "cross":
        s = (mv["phase"] * mv["length"] + mv["speed"] * t) % mv["length"]
        d = mv["dir"]
        p = np.array(
            [
                mv["p0"][0] + d[0] * s,
                mv["p0"][1] + d[1] * s,
                mv["alt"] + 2 * math.sin(0.3 * t + mv["phase"] * 6),
            ]
        )
        yaw = math.atan2(d[1], d[0])
    elif b == "flock":
        r = mv["radius"]
        a = 0.1 * t * mv["speed"] / 14.0 * 14.0 / r
        cx, cy = mv["center"]
        ox, oy, oz = mv["offset"]
        w = 0.35 * math.sin(0.5 * t + mv["jit"])  # each starling weaves a little: the cloud breathes
        p = np.array(
            [
                cx + r * math.cos(a) + r * 0.3 * math.sin(2 * a) + ox * (1 + w),
                cy + r * math.sin(a) + oy * (1 + w),
                mv["alt"] + 6 * math.sin(0.21 * t) + oz,
            ]
        )
        yaw = a + math.pi / 2
    else:  # perched until the drone comes within flush_r, then flies off away
        x, y, z = mv["perch"]
        t0 = st.get("flush")
        if t0 is None and p_drone is not None and math.dist((x, y, z), p_drone) < mv["flush_r"]:
            st["flush"] = t0 = t
            d = np.array([x - p_drone[0], y - p_drone[1]])
            st["away"] = math.atan2(d[1], d[0])
        if t0 is None:
            return {
                "": (np.array([x, y, z + 0.06]), quat_yaw(mv["phase"] * 6.28)),
                "wl": (np.array([x, y, z + 0.08]), quat_yaw(mv["phase"] * 6.28)),
                "wr": (np.array([x, y, z + 0.08]), quat_yaw(mv["phase"] * 6.28)),
                "fold": 1,
            }
        tt = t - t0
        a = st["away"]
        climb = min(1.0, tt / 3.0)
        if tt < 6:
            p = np.array(
                [
                    x + math.cos(a) * mv["speed"] * tt,
                    y + math.sin(a) * mv["speed"] * tt,
                    z + (mv["alt"] - z) * climb,
                ]
            )
            yaw = a
        else:
            r = 25.0
            ang = (tt - 6) * mv["speed"] / r
            cx, cy = x + math.cos(a) * (mv["speed"] * 6 - r), y + math.sin(a) * (mv["speed"] * 6 - r)
            p = np.array([cx + r * math.cos(a + ang), cy + r * math.sin(a + ang), mv["alt"]])
            yaw = a + ang + math.pi / 2
    gl = (math.sin(0.37 * t + mv["seed"] % 97) + 1) / 2 < S["glide"]  # gliding phases: wings held out
    flap = (0.10 if gl else 0.62 * math.sin(2 * math.pi * S["hz"] * t + mv["seed"] % 7)) + (
        0.3 if climb and climb < 1 else 0.0
    )
    q = qmul(quat_yaw(yaw), quat_pitch(-0.08))
    return {"": (p, q), "wl": (p, qmul(q, quat_roll(flap))), "wr": (p, qmul(q, quat_roll(-flap)))}


def mover_xml(i, mv):
    n = f"mover{i}"
    if mv["kind"] == "car":
        import world as W0

        return f'<body name="{n}" mocap="true" pos="0 0 -50">{W0._car_xml(f"{n}c", 0., 0., 0., mv["rgba"])}</body>'
    if mv["kind"] == "person":
        c = mv["rgba"]
        return (
            f'<body name="{n}" mocap="true" pos="0 0 -50"><geom name="{n}_b" type="capsule" fromto="0 0 .45 0 0 1.3" size=".19" rgba="{c}"/>'
            f'<geom name="{n}_l" type="capsule" fromto="0 .09 .05 0 .09 .5" size=".07" rgba=".2 .2 .25 1"/><geom name="{n}_r" type="capsule" fromto="0 -.09 .05 0 -.09 .5" size=".07" rgba=".2 .2 .25 1"/>'
            f'<geom name="{n}_h" type="sphere" pos="0 0 1.62" size=".13" rgba=".86 .70 .58 1"/></body>'
        )
    S = SPECIES[mv["species"]]
    bx, by, bz = S["body"]
    c = "{:.2f} {:.2f} {:.2f} 1".format(*S["col"])
    w = "{:.2f} {:.2f} {:.2f} 1".format(*S["wing"])
    half = S["span"] / 2
    chord = bx * 0.9
    body = (
        f'<body name="{n}" mocap="true" pos="0 0 -50"><geom name="{n}_b" type="ellipsoid" size="{bx:.3f} {by:.3f} {bz:.3f}" rgba="{c}"/>'
        f'<geom name="{n}_h" type="sphere" pos="{bx * .95:.3f} 0 {bz * .4:.3f}" size="{by * 1.05:.3f}" rgba="{c}"/>'
        f'<geom name="{n}_k" type="box" pos="{bx * 1.3:.3f} 0 {bz * .3:.3f}" size="{by * .5:.3f} {by * .18:.3f} {by * .18:.3f}" rgba=".85 .65 .2 1"/>'
        f'<geom name="{n}_t" type="box" pos="{-bx * 1.15:.3f} 0 0" size="{bx * .45:.3f} {by * 1.3:.3f} .008" rgba="{w}"/></body>'
    )
    # wing = broad inner part + tapered outer 'hand' swept back, with a
    wings = "".join(
        f'<body name="{n}_w{s}" mocap="true" pos="0 0 -50">'
        f'<geom name="{n}_w{s}g" type="ellipsoid" pos="0 {sg * half * .3:.3f} 0" size="{chord * .55:.3f} {half * .32:.3f} .012" rgba="{w}"/>'
        f'<geom name="{n}_w{s}t" type="ellipsoid" pos="{-chord * .22:.3f} {sg * half * .72:.3f} 0" size="{chord * .36:.3f} {half * .3:.3f} .008" euler="0 0 {sg * 22}" rgba="{w}"/></body>'
        for s, sg in (("l", 1), ("r", -1))
    )
    return body + wings


def actors_xml(movers):
    return "".join(mover_xml(i, mv) for i, mv in enumerate(movers))
