"""Real places from OpenStreetMap -> a MuJoCo world."""

import json, math, os, random
from pathlib import Path
import numpy as np
import mapbox_earcut as earcut
from shapely.geometry import Polygon, Point
import world as W0

PLACES = {  # name: (file, lat, lon, half-size m, place type from the map)
    "spb_center": ("spb_center.json", 59.9330, 30.3440, 200, "city_center"),
    "akadem": ("akadem.json", 54.8380, 83.1070, 200, "buildings_in_forest"),
    "malakhovka": ("malakhovka.json", 55.6440, 38.0230, 200, "private_houses"),
    "tokyo_res": ("tokyo_res.json", 35.6620, 139.6990, 200, "foreign_city"),
    "malakhovka_line": ("malakhovka_line.json", 55.65663, 37.97119, 200, "power_line_suburb"),
    "akadem_line": ("akadem_line.json", 54.87604, 83.11282, 200, "power_line_forest"),
}
OSM_DIR = Path(os.environ.get("OSM_DIR", Path(__file__).resolve().parents[1] / "osm"))
FACADES = [
    ".78 .72 .62 1",
    ".70 .52 .42 1",
    ".82 .80 .74 1",
    ".62 .64 .66 1",
    ".86 .78 .58 1",
    ".72 .70 .68 1",
    ".55 .40 .33 1",
]
ROAD_W = {
    "motorway": 14,
    "trunk": 12,
    "primary": 11,
    "secondary": 9,
    "tertiary": 8,
    "residential": 6,
    "service": 4,
    "unclassified": 6,
    "living_street": 5,
    "footway": 2,
    "path": 1.5,
    "cycleway": 2,
    "pedestrian": 4,
    "steps": 2,
}


def num(s):
    try:
        return float(str(s).replace("m", "").replace(",", ".").strip().split(";")[0])
    except ValueError:
        return None


def height_of(tags, kind, rng):
    h = num(tags.get("height")) if tags.get("height") else None
    if h:
        return h, "height"
    lv = num(tags.get("building:levels")) if tags.get("building:levels") else None
    if lv:
        return lv * 3.0 + 1.0, "levels"
    return (
        rng.uniform(4.5, 8.0) if kind in ("private_houses", "power_line_suburb") else rng.uniform(9.0, 20.0)
    ), "estimated"


def rings(e, P):
    """[(outer_xy, [inner_xy...]), ...] for a way or a multipolygon relation."""
    if e["type"] == "way":
        return [(np.array([P(g["lat"], g["lon"]) for g in e["geometry"]]), [])] if e.get("geometry") else []
    outs = [
        np.array([P(g["lat"], g["lon"]) for g in m["geometry"]])
        for m in e.get("members", [])
        if m.get("role") == "outer" and m.get("geometry")
    ]
    ins = [
        np.array([P(g["lat"], g["lon"]) for g in m["geometry"]])
        for m in e.get("members", [])
        if m.get("role") == "inner" and m.get("geometry")
    ]
    res = []
    for o in outs:
        if len(o) < 4:
            continue
        po = Polygon(o)
        res.append((o, [i for i in ins if len(i) >= 4 and po.contains(Point(i[0]))]))
    return res


def prism_meshes(name, outer, inners, h):
    """Triangulate a footprint."""
    pts = [outer[:-1] if np.allclose(outer[0], outer[-1]) else outer] + [
        r[:-1] if np.allclose(r[0], r[-1]) else r for r in inners
    ]
    V = np.concatenate(pts).astype(np.float64)
    ends = np.cumsum([len(r) for r in pts]).astype(np.uint32)
    try:
        tri = earcut.triangulate_float64(V, ends).reshape(-1, 3)
    except Exception:
        return [], []
    assets, geoms = [], []
    for k, (a, b, c) in enumerate(tri):
        A, B, C = V[a], V[b], V[c]
        if abs((B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])) < 0.05:
            continue  # degenerate sliver
        vs = " ".join(f"{x:.2f} {y:.2f} {z:.2f}" for z in (0.0, h) for x, y in (A, B, C))
        assets.append(f'<mesh name="{name}_{k}" vertex="{vs}"/>')
        geoms.append(f"{name}_{k}")
    return assets, geoms


def build(place, seed=0):
    fname, lat0, lon0, half, kind = PLACES[place]
    rng = random.Random(seed)
    E = json.loads((OSM_DIR / fname).read_text(encoding="utf-8"))["elements"]
    kx = 111_320.0 * math.cos(math.radians(lat0))
    P = lambda la, lo: ((lo - lon0) * kx, (la - lat0) * 111_320.0)
    W = W0.World(seed, 2 * half)
    assets, geoms, stats = [], [], {"place": place, "kind": kind, "height_sources": {}}
    inside = lambda x, y, m=0.0: abs(x) < half - m and abs(y) < half - m
    footprints = []
    for i, e in enumerate(x for x in E if x.get("tags", {}).get("building")):
        h, src = height_of(e.get("tags", {}), kind, rng)
        stats["height_sources"][src] = stats["height_sources"].get(src, 0) + 1
        col = FACADES[rng.randrange(len(FACADES))]
        for j, (o, ins) in enumerate(rings(e, P)):
            if not any(inside(x, y, -30) for x, y in o):
                continue
            a, g = prism_meshes(f"bld{i}_{j}", o, ins, h)
            assets += a
            geoms += [f'<geom name="{n}" type="mesh" mesh="{n}" rgba="{col}"/>' for n in g]
            poly = Polygon(o, [r for r in ins])
            footprints.append((poly, h))
            c = poly.centroid
            b = poly.bounds
            W.buildings.append((c.x, c.y, b[2] - b[0], b[3] - b[1], h, col))
    woods = [
        p
        for e in E
        if e.get("tags", {}).get("natural") == "wood" or e.get("tags", {}).get("landuse") == "forest"
        for p, _ in rings(e, P)
    ]
    taken = [fp for fp, _ in footprints]

    def free(x, y, r):
        return inside(x, y, 2) and not any(fp.distance(Point(x, y)) < r for fp in taken)

    trees = [
        (*P(e["lat"], e["lon"]), rng.uniform(5, 12))
        for e in E
        if e["type"] == "node" and e.get("tags", {}).get("natural") == "tree"
    ]
    for o in woods:  # forest: random trees inside the real polygon, >= 3 m apart, capped
        po = Polygon(o).intersection(Polygon([(-half, -half), (half, -half), (half, half), (-half, half)]))
        if po.is_empty:
            continue
        n = min(int(po.area / 40.0), 700 - len(trees))
        bx = po.bounds
        tries = 0
        pts = []
        while len(pts) < n and tries < n * 20:
            tries += 1
            x, y = rng.uniform(bx[0], bx[2]), rng.uniform(bx[1], bx[3])
            if (
                po.contains(Point(x, y))
                and free(x, y, 2)
                and all((x - a) ** 2 + (y - b) ** 2 > 9 for a, b in pts[-60:])
            ):
                pts.append((x, y))
        trees += [(x, y, rng.uniform(8, 22)) for x, y in pts]
    for k, (x, y, h) in enumerate(trees):
        if not inside(x, y):
            continue
        W.trees.append((x, y, h))
        geoms.append(
            W0.tree_xml(k, x, y, h, random.Random(seed * 7919 + k))
            if k < 80
            else f'<geom name="trunk{k}" type="capsule" fromto="{x:.2f} {y:.2f} 0 {x:.2f} {y:.2f} {h * .7:.2f}" size=".2" rgba=".33 .22 .12 1"/>'
            f'<geom name="crown{k}" type="ellipsoid" pos="{x:.2f} {y:.2f} {h * .72:.2f}" size="{h * .16:.2f} {h * .16:.2f} {h * .3:.2f}" rgba=".13 .42 .16 1"/>'
        )
    wire_n = 0  # power lines: towers at the real vertices, sagging conductors between
    for e in E:
        pw = e.get("tags", {}).get("power")
        if pw not in ("line", "minor_line") or not e.get("geometry"):
            continue
        hv = pw == "line"
        th = num(e["tags"].get("height")) or (26.0 if hv else 10.0)
        V = [P(g["lat"], g["lon"]) for g in e["geometry"]]
        offs = [(-3.5, -2.5), (3.5, -2.5), (0.0, 0.0)] if hv else [(-0.5, -0.3), (0.5, -0.3)]
        for k, (x, y) in enumerate(V):
            if inside(x, y, -60):
                geoms.append(
                    f'<geom name="tower{wire_n}_{k}" type="box" pos="{x:.2f} {y:.2f} {th / 2:.2f}" size="{.9 if hv else .15:.2f} {.9 if hv else .15:.2f} {th / 2:.2f}" rgba=".5 .5 .52 1"/>'
                )
        for k in range(len(V) - 1):
            (x1, y1), (x2, y2) = V[k], V[k + 1]
            if not (
                inside(x1, y1, -60) and inside(x2, y2, -60)
            ):  # keep only spans with both towers near the area
                if not (inside(x1, y1, -60) or inside(x2, y2, -60)):
                    continue
                lim = half + 60
                t = 1.0
                for a, b in ((x1, x2), (y1, y2)):  # clip the far end to the area border (+60 m)
                    if abs(b) > lim:
                        t = min(t, (math.copysign(lim, b) - a) / (b - a))
                if inside(x1, y1, -60):
                    x2, y2 = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
                else:
                    x1, y1, x2, y2 = x2, y2, x2 + t * (x1 - x2), y2 + t * (y1 - y2)
            dx, dy = x2 - x1, y2 - y1
            L = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / L, dx / L
            sag = 0.03 * L
            for c, (o, dz) in enumerate(offs):
                ax, ay, bx_, by_ = x1 + nx * o, y1 + ny * o, x2 + nx * o, y2 + ny * o
                z = th + dz
                mx, my = (ax + bx_) / 2, (ay + by_) / 2
                for s_, (p1, p2) in enumerate(
                    (((ax, ay, z), (mx, my, z - sag)), ((mx, my, z - sag), (bx_, by_, z)))
                ):
                    geoms.append(
                        f'<geom name="wire{wire_n}_{k}_{c}_{s_}" type="capsule" fromto="{p1[0]:.2f} {p1[1]:.2f} {p1[2]:.2f} {p2[0]:.2f} {p2[1]:.2f} {p2[2]:.2f}" size="{.02 if hv else .012}" rgba=".1 .1 .1 1"/>'
                    )
            W.wires.append((x1, y1, x2, y2, th))
        wire_n += 1
    W.water = []  # visual water + polygons for the optical-flow check
    for e in E:
        t = e.get("tags", {})
        if t.get("natural") == "water":
            for o, ins in rings(e, P):
                a, g = prism_meshes(f"water{len(W.water)}", o, ins, 0.03)
                assets += a
                geoms += [
                    f'<geom name="{n}" type="mesh" mesh="{n}" rgba=".25 .42 .62 1" contype="0" conaffinity="0"/>'
                    for n in g
                ]
                W.water.append(Polygon(o))
    lanes, foot = [], []
    for e in E:
        hw = e.get("tags", {}).get("highway")
        if not hw or not e.get("geometry") or e["type"] != "way":
            continue
        V = [P(g["lat"], g["lon"]) for g in e["geometry"]]
        w = ROAD_W.get(hw, 4)
        for k in range(len(V) - 1):
            (x1, y1), (x2, y2) = V[k], V[k + 1]
            L = math.hypot(x2 - x1, y2 - y1)
            if L < 1 or not (inside(x1, y1) and inside(x2, y2)):
                continue
            ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
            geoms.append(
                f'<geom name="road{len(lanes) + len(foot)}_{hw[:3]}" type="box" pos="{(x1 + x2) / 2:.2f} {(y1 + y2) / 2:.2f} .02" size="{L / 2 + w / 2:.2f} {w / 2:.2f} .02" '
                f'euler="0 0 {ang:.1f}" rgba="{".25 .25 .27 1" if w >= 4 else ".6 .55 .48 1"}" contype="0" conaffinity="0"/>'
            )
            (lanes if w >= 4 else foot).append((x1, y1, x2, y2, L))
    for mv in range(min(5, len([l for l in lanes if l[4] > 25]))):  # moving cars on real road segments
        x1, y1, x2, y2, L = rng.choice([l for l in lanes if l[4] > 25])
        g = rng.uniform(0.2, 0.9)
        W.movers.append(
            {
                "kind": "car",
                "seg": (x1, y1, x2, y2),
                "speed": rng.uniform(4, 12),
                "phase": rng.random(),
                "rgba": f"{g:.2f} {rng.uniform(.1, .9):.2f} {1 - g:.2f} 1",
            }
        )
    for mv in range(min(6, len([l for l in foot + lanes if l[4] > 8]))):
        x1, y1, x2, y2, L = rng.choice([l for l in foot + lanes if l[4] > 8])
        W.movers.append(
            {
                "kind": "person",
                "seg": (x1, y1, x2, y2),
                "speed": rng.uniform(1.0, 1.6),
                "phase": rng.random(),
                "rgba": f"{rng.uniform(.1, .9):.2f} {rng.uniform(.1, .9):.2f} {rng.uniform(.1, .9):.2f} 1",
            }
        )
    for i in range(5):
        W.movers.append(
            {
                "kind": "bird",
                "center": (rng.uniform(-half, half), rng.uniform(-half, half)),
                "radius": rng.uniform(8, 25),
                "alt": rng.uniform(8, 35),
                "speed": rng.uniform(6, 11),
                "phase": rng.random(),
                "rgba": ".15 .13 .12 1",
                "diver": i < 2,
            }
        )
    for _ in range(5000):  # start pad near the centre on free ground
        x, y = rng.uniform(-0.3 * half, 0.3 * half), rng.uniform(-0.3 * half, 0.3 * half)
        if free(x, y, 6) and not any(math.hypot(x - a, y - b) < 4 for a, b, _ in W.trees):
            W.start = (x, y)
            break
    combos = [(s, c) for s in W0.SHAPES for c in W0.COLORS]
    rng.shuffle(combos)
    for _ in range(20000):  # civil goals on free ground, 40+ m from the start
        if len(W.targets) >= 4:
            break
        x, y = rng.uniform(-half + 15, half - 15), rng.uniform(-half + 15, half - 15)
        if (
            math.hypot(x - W.start[0], y - W.start[1]) < 40
            or not free(x, y, 5)
            or any(math.hypot(x - a, y - b) < 4 for a, b, _ in W.trees)
        ):
            continue
        shape, color = combos[len(W.targets)]
        t = W0.Target(f"target{len(W.targets)}", shape, color, (x, y))
        t.description = describe(t, footprints)
        W.targets.append(t)
    sx, sy = W.start
    for k in range(6):  # parked cars along real roads
        if not lanes:
            break
        x1, y1, x2, y2, L = rng.choice(lanes)
        u = rng.random()
        ang = math.atan2(y2 - y1, x2 - x1)
        x, y = x1 + u * (x2 - x1) + 3 * math.cos(ang + math.pi / 2), y1 + u * (y2 - y1) + 3 * math.sin(
            ang + math.pi / 2
        )
        if free(x, y, 1.5):
            W.cars.append(
                (
                    x,
                    y,
                    ang,
                    f"{rng.uniform(.2, .8):.2f} {rng.uniform(.2, .8):.2f} {rng.uniform(.2, .8):.2f} 1",
                )
            )
    az, el = rng.uniform(0, 2 * math.pi), rng.uniform(0.35, 1.2)
    W.sun = (math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), -math.sin(el), rng.uniform(0.5, 0.85))
    stats.update(
        buildings=len(W.buildings),
        prisms=sum(1 for g in geoms if 'type="mesh"' in g),
        trees=len(W.trees),
        wires=len(W.wires),
        water=len(W.water),
        road_segments=len(lanes),
        footways=len(foot),
        start=[round(sx, 1), round(sy, 1)],
    )
    W.stats = stats
    # move everything so that the start pad is at the origin
    from shapely.affinity import translate

    W.buildings = [(x - sx, y - sy, a, b, h, c) for x, y, a, b, h, c in W.buildings]
    W.trees = [(x - sx, y - sy, h) for x, y, h in W.trees]
    W.cars = [(x - sx, y - sy, a, c) for x, y, a, c in W.cars]
    W.wires = [(a - sx, b - sy, c - sx, d - sy, h) for a, b, c, d, h in W.wires]
    W.water = [translate(p, -sx, -sy) for p in W.water]
    for mv in W.movers:
        if "seg" in mv:
            x1, y1, x2, y2 = mv["seg"]
            mv["seg"] = (x1 - sx, y1 - sy, x2 - sx, y2 - sy)
        if "center" in mv:
            mv["center"] = (mv["center"][0] - sx, mv["center"][1] - sy)
    for t in W.targets:
        t.pos = (t.pos[0] - sx, t.pos[1] - sy)
    W.start = (0.0, 0.0)
    return W, W0.to_mjcf(W, building_xml="".join(geoms), extra_assets="".join(assets), shift=(-sx, -sy))


def describe(t, footprints):
    """Description from geometry + map data."""
    x, y = t.pos
    desc = f"the {t.color} {W0.WORD[t.shape]}"
    if footprints:
        fp, h = min(footprints, key=lambda f: f[0].distance(Point(x, y)))
        if fp.distance(Point(x, y)) < 30:
            c = fp.centroid
            side = (
                ("east" if x > c.x else "west")
                if abs(x - c.x) > abs(y - c.y)
                else ("north" if y > c.y else "south")
            )
            desc += f" {side} of a {max(1, round((h - 1) / 3))}-storey building"
    return desc
