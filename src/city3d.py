"""Stage F: a realistic MuJoCo world from a real place."""

import json, math, os, random, zlib
from pathlib import Path
import numpy as np
import mapbox_earcut as earcut
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, box as sbox
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
import textures as TX
import world as W0
from osm_fetch2 import AREAS, HALF

OSM2 = Path(os.environ.get("OSM2_DIR", Path(__file__).resolve().parents[1] / "osm2"))
TILE = 100.0
HOUSE = {
    "house",
    "detached",
    "semidetached_house",
    "bungalow",
    "cabin",
    "farm",
    "villa",
    "hut",
    "static_caravan",
    "dacha",
}
GARAGE = {"garage", "garages", "shed", "carport", "storage", "barn", "greenhouse"}
INDUSTRIAL = {
    "industrial",
    "warehouse",
    "factory",
    "manufacture",
    "hangar",
    "service",
    "transformer_tower",
    "construction",
}
OFFICE = {
    "office",
    "commercial",
    "retail",
    "supermarket",
    "mall",
    "hotel",
    "hospital",
    "university",
    "college",
}
CHURCH = {"church", "cathedral", "chapel", "temple", "mosque", "shrine"}
COLOUR_STYLE = {
    "yellow": "plaster_yellow",
    "beige": "panel_beige",
    "pink": "plaster_pink",
    "red": "brick_red",
    "brown": "brick_red",
    "maroon": "brick_red",
    "white": "plaster_white",
    "blue": "panel_blue",
    "lightblue": "panel_blue",
    "green": "plaster_green",
    "grey": "panel_light",
    "gray": "panel_light",
    "silver": "panel_light",
    "orange": "plaster_yellow",
    "tan": "panel_beige",
}
ROOF_COLOUR = {
    "red": "roof_metal_red",
    "brown": "roof_metal_brown",
    "maroon": "roof_metal_red",
    "green": "roof_metal_green",
    "darkgreen": "roof_metal_green",
    "grey": "roof_metal_grey",
    "gray": "roof_metal_grey",
    "silver": "roof_metal_grey",
    "black": "roof_bitumen",
}
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
    "pedestrian": 5,
    "track": 3.5,
    "footway": 2,
    "path": 1.5,
    "cycleway": 2,
    "steps": 2,
    "bridleway": 2,
}
FOOT = {"footway", "path", "cycleway", "steps", "pedestrian", "bridleway", "track"}


def f3(a):
    return " ".join(f"{v:.3f}" for v in np.asarray(a, float).ravel())


def f2(a):
    return " ".join(f"{v:.2f}" for v in np.asarray(a, float).ravel())


def ints(a):
    return " ".join(str(int(v)) for v in np.asarray(a).ravel())


def num(s):
    try:
        return float(str(s).replace("m", "").replace(",", ".").strip().split(";")[0])
    except (ValueError, TypeError):
        return None


def colour_word(c):
    if not c:
        return None
    c = c.strip().lower()
    if c.startswith("#") and len(c) in (4, 7):
        h = c[1:] if len(c) == 7 else "".join(x * 2 for x in c[1:])
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        if max(r, g, b) - min(r, g, b) < 25:
            return "white" if r > 190 else "grey"
        return max(
            (
                ("red", (200, 60, 50)),
                ("brown", (130, 80, 50)),
                ("yellow", (220, 190, 90)),
                ("green", (90, 150, 90)),
                ("blue", (80, 120, 200)),
                ("pink", (220, 150, 150)),
                ("beige", (220, 200, 160)),
            ),
            key=lambda kv: -sum((a - b) ** 2 for a, b in zip((r, g, b), kv[1])),
        )[0]
    return c


class Bank:
    """Visual triangles with texture coordinates, merged per (material, 100 m tile)."""

    def __init__(self):
        self.d = {}

    def add(self, mat, V, UV, F, at):
        key = (mat, int(math.floor(at[0] / TILE)), int(math.floor(at[1] / TILE)))
        L = self.d.setdefault(key, [[], [], [], 0])
        L[0].append(np.asarray(V, float))
        L[1].append(np.asarray(UV, float))
        L[2].append(np.asarray(F, int) + L[3])
        L[3] += len(V)

    def quad(self, mat, a, b, c, dd, uv):
        """Front face = a, b, c, d counter-clockwise as seen by the viewer."""
        self.add(mat, [a, b, c, dd], uv, [[0, 1, 2], [0, 2, 3]], a)

    def xml(self, prefix="vis"):
        assets, geoms = [], []
        for k, (key, (V, UV, F, n)) in enumerate(self.d.items()):
            if not n:
                continue
            V, UV, F = np.concatenate(V), np.concatenate(UV), np.concatenate(F)
            nm = f"{prefix}{k}"
            if len(V) < 4:
                V, UV = np.vstack([V, V[:1] + 1e-3]), np.vstack(
                    [UV, UV[:1]]
                )  # MuJoCo wants >= 4 vertices per mesh
            assets.append(
                f'<mesh name="{nm}" vertex="{f3(V)}" texcoord="{f3(UV)}" face="{ints(F)}" inertia="shell"/>'
            )
            geoms.append(
                f'<geom name="{nm}" type="mesh" mesh="{nm}" material="{key[0]}" contype="0" conaffinity="0" group="1"/>'
            )
        return assets, geoms


def tri_poly(poly):
    """Earcut triangles of a shapely polygon with holes ->."""
    poly = orient(poly, 1.0)
    rings = [np.array(poly.exterior.coords)[:-1]] + [np.array(r.coords)[:-1] for r in poly.interiors]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return None, None
    V = np.concatenate(rings).astype(np.float64)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    try:
        F = earcut.triangulate_float64(V, ends).reshape(-1, 3)
    except Exception:
        return None, None
    if not len(F):
        return None, None
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    F = F[np.abs(cross) > 1e-6]
    cross = cross[np.abs(cross) > 1e-6]
    F[cross < 0] = F[cross < 0][:, [0, 2, 1]]
    return V, F


def polys(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def surface(bank, geom, mat, z, tile_m):
    for p in polys(geom):
        V2, F = tri_poly(p)
        if V2 is None:
            continue
        V = np.c_[V2, np.full(len(V2), z)]
        bank.add(mat, V, V2 / tile_m, F, V2.mean(0))


def walls(bank, ring, z0, z1, mat, uw, uh, v0=0.0):
    """Outward-facing wall quads along a ring."""
    R = np.asarray(ring)[:, :2]
    s = 0.0
    for k in range(len(R) - 1):
        A, B = R[k], R[k + 1]
        L = float(np.hypot(*(B - A)))
        if L < 0.05:
            continue
        u0, u1 = s / uw, (s + L) / uw
        va, vb = -(v0) / uh, -(v0 + z1 - z0) / uh
        bank.quad(
            mat,
            (A[0], A[1], z0),
            (B[0], B[1], z0),
            (B[0], B[1], z1),
            (A[0], A[1], z1),
            [(u0, va), (u1, va), (u1, vb), (u0, vb)],
        )
        s += L


def prism(name, V2, h0, h1):
    """MJCF mesh asset + collision geom for one convex prism (group 3: invisible, collides, lidar sees it)."""
    h1 = max(h1, h0 + 0.1)
    vs = " ".join(
        f"{x:.2f} {y:.2f} {z:.2f}" for z in (h0, h1) for x, y in V2
    )  # never flat (MuJoCo needs a 3-D hull)
    return (
        f'<mesh name="{name}" vertex="{vs}"/>',
        f'<geom name="{name}" type="mesh" mesh="{name}" group="3" rgba=".5 .5 .5 1"/>',
    )


def collision_prisms(name, poly, h):
    V2, F = tri_poly(poly)
    A, G = [], []
    if V2 is None:
        return A, G
    for k, tri in enumerate(F):
        a, g = prism(f"{name}_{k}", V2[tri], 0.0, h)
        A.append(a)
        G.append(g)
    return A, G


class Place:
    def __init__(self, place):
        lat, lon, split, kind = AREAS[place]
        self.name, self.split, self.kind = place, split, kind
        self.E = json.loads((OSM2 / f"{place}.json").read_text(encoding="utf-8"))["elements"]
        kx = 111_320.0 * math.cos(math.radians(lat))
        self.P = lambda la, lo: ((lo - lon) * kx, (la - lat) * 111_320.0)
        self.lat0, self.lon0 = lat, lon
        self.area = sbox(-HALF, -HALF, HALF, HALF)

    def rings(self, e):
        P = self.P
        if e["type"] == "way":
            g = e.get("geometry")
            return [(np.array([P(p["lat"], p["lon"]) for p in g]), [])] if g and len(g) >= 4 else []
        outs = [
            np.array([P(p["lat"], p["lon"]) for p in m["geometry"]])
            for m in e.get("members", [])
            if m.get("role") == "outer" and m.get("geometry")
        ]
        ins = [
            np.array([P(p["lat"], p["lon"]) for p in m["geometry"]])
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

    def polygon(self, e):
        out = []
        for o, ins in self.rings(e):
            try:
                p = Polygon(o, [i for i in ins]).buffer(0)
                if not p.is_empty:
                    out += polys(p.intersection(self.area))
            except Exception:
                pass
        return out

    def line(self, e):
        g = e.get("geometry")
        return [self.P(p["lat"], p["lon"]) for p in g] if g else []


def building_style(tags, kind, area, h, rng):
    """Facade style, roof kind and roof material from the map's own tags (building type, material, colour)."""
    b = tags.get("building", "yes")
    mat = (tags.get("building:material") or "").lower()
    col = colour_word(tags.get("building:colour"))
    rc = colour_word(tags.get("roof:colour"))
    rshape = (tags.get("roof:shape") or "").lower()
    small_house = (
        kind in ("private_houses", "dachas_in_pines", "suburb_abroad", "power_line_suburb")
        and area < 260
        and h < 10
    )
    if b in HOUSE or small_house:
        style = (
            "house_wood"
            if mat == "wood" or (kind == "dachas_in_pines" and rng.random() < 0.6)
            else (
                "brick_red"
                if mat == "brick"
                else rng.choice(
                    ["house_siding", "house_plaster", "house_wood", "brick_silicate", "house_siding"]
                )
            )
        )
        roof = "gable" if rshape in ("", "gabled", "hipped", "pitched", "half-hipped") else "flat"
        roof_mat = ROOF_COLOUR.get(rc) or rng.choice(
            ["roof_metal_red", "roof_metal_green", "roof_metal_brown", "roof_tiles", "roof_metal_grey"]
        )
    elif b in GARAGE or (b == "yes" and area < 45 and h < 5):
        style, roof, roof_mat = ("garage" if mat != "wood" else "house_wood"), "flat", "roof_metal_grey"
    elif b in INDUSTRIAL:
        style, roof, roof_mat = (
            "industrial",
            "flat",
            rng.choice(["roof_metal_grey", "roof_bitumen", "roof_gravel"]),
        )
    elif b in OFFICE:
        style, roof, roof_mat = (
            rng.choice(["glass_blue", "glass_green", "plaster_white", "panel_light"]),
            "flat",
            "roof_gravel",
        )
    elif b in CHURCH:
        style, roof, roof_mat = rng.choice(["plaster_white", "plaster_yellow"]), "flat", "roof_metal_green"
    elif kind == "foreign_city":
        style, roof, roof_mat = (
            rng.choice(["jp_tile", "jp_tile", "plaster_white", "panel_light"]),
            "flat",
            "roof_gravel",
        )
    elif kind == "city_center" and h < 30:
        style, roof, roof_mat = (
            rng.choice(["plaster_yellow", "plaster_pink", "plaster_green", "plaster_white", "brick_red"]),
            "flat",
            rng.choice(["roof_metal_grey", "roof_metal_red", "roof_metal_green"]),
        )
    else:
        style, roof, roof_mat = (
            rng.choice(
                ["panel_light", "panel_beige", "panel_blue", "brick_silicate", "brick_red", "panel_light"]
            ),
            "flat",
            rng.choice(["roof_bitumen", "roof_gravel"]),
        )
    if col in COLOUR_STYLE and not style.startswith(("house", "garage", "industrial", "glass")):
        style = COLOUR_STYLE[col]
    if mat == "brick" and style.startswith(("panel", "plaster")):
        style = "brick_red" if col in (None, "red", "brown") else "brick_silicate"
    if mat == "glass":
        style = "glass_blue"
    if mat == "wood" and style.startswith("house"):
        style = "house_wood"
    return style, roof, roof_mat, b


def height_of(tags, kind, area, rng):
    h = num(tags.get("height"))
    lv = num(tags.get("building:levels"))
    b = tags.get("building", "yes")
    if h and h > 1.5:
        return h, lv or max(1.0, round((h - 1) / 3)), "height"
    if lv:
        return lv * 3.0 + 1.0, lv, "levels"
    if b in GARAGE or (b == "yes" and area < 45):
        return rng.uniform(2.6, 3.4), 1, "estimated"
    if b in HOUSE or (
        kind in ("private_houses", "dachas_in_pines", "suburb_abroad", "power_line_suburb") and area < 260
    ):
        lv = rng.choice([1, 1, 2, 2, 2])
        return lv * 3.0 + 0.5, lv, "estimated"
    lv = (
        rng.choice([5, 5, 9, 9, 12, 16])
        if kind in ("panel_district", "panel_district_abroad")
        else rng.choice([2, 3, 4, 5, 6])
    )
    return lv * 3.0 + 1.0, lv, "estimated"


def house_rect(poly):
    """A near-rectangular small footprint -> its rectangle corners."""
    r = poly.minimum_rotated_rectangle
    if r.area <= 0 or poly.area / r.area < 0.82:
        return None
    c = np.array(orient(r, 1.0).exterior.coords)[:4]
    if np.hypot(*(c[1] - c[0])) < np.hypot(*(c[2] - c[1])):
        c = np.roll(c, -1, 0)
    return c


def add_building(bank, A, G, name, poly, h, levels, style, roof, roof_mat, gf, rng):
    poly = orient(poly, 1.0)
    eave = h
    rect = house_rect(poly) if roof == "gable" else None
    if rect is not None:
        short = np.hypot(*(rect[2] - rect[1]))
        rise = float(np.clip(0.32 * short, 1.2, 4.0))
        eave = (
            h if levels else max(h - rise, 2.0)
        )  # a low shed must not collapse to a flat (coplanar) wall prism
        ring = np.vstack([rect, rect[:1]])
        walls(bank, ring, 0.0, eave, f"fac_{style}", TX.TILE_W, TX.TILE_H)
        p0, p1, p2, p3 = rect
        m03, m12 = (p0 + p3) / 2, (p1 + p2) / 2
        ax = (p1 - p0) / np.linalg.norm(p1 - p0)
        o = 0.45
        q0, q1, q2, q3 = p0 - ax * o, p1 + ax * o, p2 + ax * o, p3 - ax * o
        r0, r1 = m03 - ax * o, m12 + ax * o
        out = lambda p, m: p + (p - m) / max(np.linalg.norm(p - m), 1e-6) * 0.35  # eave overhang sideways
        q0, q1, q2, q3 = out(q0, r0), out(q1, r1), out(q2, r1), out(q3, r0)
        zt = eave + rise
        ze = eave - 0.25
        sl = float(np.hypot(np.linalg.norm(q0 - r0), rise))
        L = float(np.linalg.norm(q1 - q0))
        uv = [(0, 0), (L / 4, 0), (L / 4, sl / 4), (0, sl / 4)]
        bank.quad(roof_mat, (*q0, ze), (*q1, ze), (*r1, zt), (*r0, zt), uv)
        bank.quad(roof_mat, (*q2, ze), (*q3, ze), (*r0, zt), (*r1, zt), uv)
        for a_, b_, m in ((p3, p0, m03), (p1, p2, m12)):  # gable triangles in the facade texture
            bank.add(
                f"fac_{style}",
                [(*a_, eave), (*b_, eave), (*m, zt)],
                [
                    (0, -0.0),
                    (np.linalg.norm(b_ - a_) / TX.TILE_W, 0),
                    (np.linalg.norm(b_ - a_) / 2 / TX.TILE_W, -rise / TX.TILE_H),
                ],
                [[0, 1, 2]],
                m,
            )
        a, g = prism(f"{name}_w", rect, 0.0, eave)
        A.append(a)
        G.append(g)
        vs = " ".join(
            f"{x:.2f} {y:.2f} {z:.2f}"
            for x, y, z in [(*p0, eave), (*p1, eave), (*p2, eave), (*p3, eave), (*m03, zt), (*m12, zt)]
        )
        A.append(f'<mesh name="{name}_r" vertex="{vs}"/>')
        G.append(f'<geom name="{name}_r" type="mesh" mesh="{name}_r" group="3"/>')
        return zt
    rings = [np.array(poly.exterior.coords)] + [np.array(r.coords) for r in poly.interiors]
    for ring in rings:
        if gf and h > 6:
            walls(bank, ring, 0.0, 3.5, f"gf_{style}", 2 * TX.BAY_W, 3.5)
            walls(bank, ring, 3.5, h, f"fac_{style}", TX.TILE_W, TX.TILE_H)
        else:
            walls(bank, ring, 0.0, h, f"fac_{style}", TX.TILE_W, TX.TILE_H, v0=0.0)
    V2, F = tri_poly(poly)
    if V2 is not None:
        bank.add(roof_mat, np.c_[V2, np.full(len(V2), h)], V2 / 6.0, F, V2.mean(0))
        if poly.area > 120 and h > 8:  # parapet: a low wall around flat roofs of big buildings
            walls(bank, np.array(poly.exterior.coords), h, h + 0.6, "srf_concrete", 6.0, 6.0)
    a, g = collision_prisms(name, poly, h)
    A += a
    G += g
    return h


def tree_geoms(k, x, y, h, kind, rng):
    """Visible AND colliding."""
    n = f"tr{k}"
    out = []
    if kind == "spruce":
        out.append(
            f'<geom name="{n}_t" type="capsule" fromto="{x:.2f} {y:.2f} 0 {x:.2f} {y:.2f} {h * .9:.2f}" size="{.1 + .015 * h:.2f}" material="srf_bark"/>'
        )
        for t in range(4):
            z0 = h * (0.18 + 0.2 * t)
            r = h * (0.26 - 0.05 * t) * rng.uniform(0.85, 1.1)
            hh = h * 0.34
            ci = min(range(len(CONES)), key=lambda i: (CONES[i][0] - r) ** 2 + (CONES[i][1] - hh) ** 2)
            out.append(
                f'<geom name="{n}_c{t}" type="mesh" mesh="cone{ci}" pos="{x:.2f} {y:.2f} {z0:.2f}" material="srf_needles"/>'
            )
        return "".join(out)
    birch = kind == "birch"
    pine = kind == "pine"
    trunk_top = h * (0.8 if pine else 0.62)
    out.append(
        f'<geom name="{n}_t" type="capsule" fromto="{x:.2f} {y:.2f} 0 {x:.2f} {y:.2f} {trunk_top:.2f}" size="{(.09 if birch else .12) + .018 * h:.2f}" material="{"srf_bark_birch" if birch else "srf_bark"}"/>'
    )
    for b in range(rng.randint(2, 4)):
        a = rng.uniform(0, 2 * math.pi)
        z0 = h * rng.uniform(0.4, 0.6)
        L = h * rng.uniform(0.12, 0.2)
        out.append(
            f'<geom name="{n}_b{b}" type="capsule" fromto="{x:.2f} {y:.2f} {z0:.2f} {x + L * math.cos(a):.2f} {y + L * math.sin(a):.2f} {z0 + L * .7:.2f}" size=".06" material="srf_bark"/>'
        )
    nb = 3 if pine else rng.randint(4, 7)
    R = h * (0.14 if birch else 0.2 if not pine else 0.16)
    for b in range(nb):
        dx, dy = rng.uniform(-0.6, 0.6) * R, rng.uniform(-0.6, 0.6) * R
        z = h * (rng.uniform(0.82, 0.95) if pine else rng.uniform(0.6, 0.85))
        r = R * rng.uniform(0.65, 1.0)
        out.append(
            f'<geom name="{n}_f{b}" type="ellipsoid" pos="{x + dx:.2f} {y + dy:.2f} {z:.2f}" size="{r:.2f} {r * rng.uniform(.85, 1.1):.2f} {r * (.55 if pine else .8):.2f}" material="srf_foliage{rng.randint(0, 2)}"/>'
        )
    return "".join(out)


CONES = [(r, hh) for r in (1.0, 1.6, 2.2, 2.9, 3.6) for hh in (1.6, 2.6, 3.8, 5.0)]


def cone_assets():
    n = 12
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out = []
    for i, (r, hh) in enumerate(CONES):
        V = [(r * math.cos(t), r * math.sin(t), 0.0) for t in a] + [(0.0, 0.0, hh)]
        out.append(f'<mesh name="cone{i}" vertex="{f2(V)}"/>')
    return out


def lamp_geoms(k, x, y, ang):
    dx, dy = 0.9 * math.cos(ang), 0.9 * math.sin(ang)
    return (
        f'<geom name="lamp{k}" type="cylinder" pos="{x:.2f} {y:.2f} 4" size=".07 4" material="srf_metal"/>'
        f'<geom name="lamp{k}a" type="capsule" fromto="{x:.2f} {y:.2f} 7.9 {x + dx:.2f} {y + dy:.2f} 7.9" size=".04" material="srf_metal"/>'
        f'<geom name="lamp{k}h" type="box" pos="{x + dx:.2f} {y + dy:.2f} 7.8" size=".3 .14 .07" euler="0 0 {math.degrees(ang):.1f}" rgba=".85 .85 .8 1"/>'
    )


def man_made_geoms(k, x, y, kind, tags, rng):
    """Mapped landmarks: water towers, masts, chimneys, storage tanks (visible and colliding)."""
    h = num(tags.get("height")) or {
        "water_tower": rng.uniform(22, 32),
        "mast": rng.uniform(30, 50),
        "tower": rng.uniform(25, 40),
        "chimney": rng.uniform(30, 60),
        "storage_tank": rng.uniform(8, 14),
        "flagpole": 9.0,
    }.get(kind, 15)
    n = f"mm{k}"
    if kind == "water_tower":
        return (
            f'<geom name="{n}_s" type="cylinder" pos="{x:.2f} {y:.2f} {h * .35:.2f}" size="{2.4:.2f} {h * .35:.2f}" material="fac_brick_red"/>'
            f'<geom name="{n}_t" type="cylinder" pos="{x:.2f} {y:.2f} {h * .82:.2f}" size="{4.2:.2f} {h * .12:.2f}" material="fac_plaster_white"/>'
            f'<geom name="{n}_r" type="ellipsoid" pos="{x:.2f} {y:.2f} {h * .94:.2f}" size="4.3 4.3 1.2" material="srf_roof_metal_grey"/>'
        ), h
    if kind in ("mast", "tower"):
        g = "".join(
            f'<geom name="{n}_l{i}" type="box" pos="{x + .8 * sx:.2f} {y + .8 * sy:.2f} {h / 2:.2f}" size=".08 .08 {h / 2:.2f}" material="srf_metal"/>'
            for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))
        )
        g += "".join(
            f'<geom name="{n}_x{i}" type="box" pos="{x:.2f} {y:.2f} {z:.2f}" size=".9 .9 .05" material="srf_metal"/>'
            for i, z in enumerate(np.arange(3, h, 4.0))
        )
        g += "".join(
            f'<geom name="{n}_a{i}" type="box" pos="{x + 1.1 * math.cos(a):.2f} {y + 1.1 * math.sin(a):.2f} {h - 2:.2f}" size=".12 .3 .9" euler="0 0 {math.degrees(a):.0f}" rgba=".9 .9 .9 1"/>'
            for i, a in enumerate((0, 2.1, 4.2))
        )
        return g, h
    if kind == "chimney":
        return (
            f'<geom name="{n}" type="cylinder" pos="{x:.2f} {y:.2f} {h / 2:.2f}" size="{max(1.2, h / 25):.2f} {h / 2:.2f}" material="fac_brick_red"/>',
            h,
        )
    if kind == "storage_tank":
        return (
            f'<geom name="{n}" type="cylinder" pos="{x:.2f} {y:.2f} {h / 2:.2f}" size="{rng.uniform(4, 8):.2f} {h / 2:.2f}" material="srf_metal"/>',
            h,
        )
    return (
        f'<geom name="{n}" type="cylinder" pos="{x:.2f} {y:.2f} {h / 2:.2f}" size=".08 {h / 2:.2f}" material="srf_metal"/>',
        h,
    )


def build(place, seed=0, extra_xml="", extra_assets=(), map_errors=True, actors=True):
    pl = Place(place)
    rng = random.Random(zlib.crc32(f"{place}/{seed}".encode()))
    bank = Bank()
    A, G = [], []
    W = W0.World(seed, 2 * HALF)
    W.place, W.kind, W.split = place, pl.kind, pl.split
    W.map = {"buildings": [], "roads": [], "landmarks": [], "water": [], "wires": []}
    W.map_errors = []
    W.landmarks = []
    W.lamps = []
    W.unmapped = []
    W.footways = []
    W.objects = []
    stats = {"place": place, "kind": pl.kind, "height_src": {}}
    inside = lambda x, y, m=0.0: abs(x) < HALF - m and abs(y) < HALF - m
    tg = lambda e, k: e.get("tags", {}).get(k)
    # ---- ground cover (landuse, parks, pitches, parking, water), bottom
    LAND = {
        "grass": "srf_grass",
        "meadow": "srf_grass",
        "village_green": "srf_grass",
        "recreation_ground": "srf_grass",
        "cemetery": "srf_grass",
        "forest": "srf_forest_floor",
        "industrial": "srf_concrete",
        "commercial": "srf_paving",
        "retail": "srf_paving",
        "railway": "srf_dirt",
        "construction": "srf_dirt",
        "brownfield": "srf_dirt",
        "farmland": "srf_grass_dry",
        "allotments": "srf_grass_dry",
        "garages": "srf_asphalt",
    }
    layers = {}
    for e in pl.E:
        t = e.get("tags", {})
        lu, le, nat = t.get("landuse"), t.get("leisure"), t.get("natural")
        if lu in LAND:
            layers.setdefault((1, LAND[lu]), []).extend(pl.polygon(e))
        if nat == "wood":
            layers.setdefault((1, "srf_forest_floor"), []).extend(pl.polygon(e))
        if nat in ("scrub", "heath", "grassland", "wetland"):
            layers.setdefault((1, "srf_grass_dry"), []).extend(pl.polygon(e))
        if nat in ("sand", "beach"):
            layers.setdefault((2, "srf_sand"), []).extend(pl.polygon(e))
        if le in ("park", "garden"):
            layers.setdefault((2, "srf_grass"), []).extend(pl.polygon(e))
        if le in ("pitch", "stadium"):
            layers.setdefault((4, "srf_pitch"), []).extend(pl.polygon(e))
        if le == "track":
            layers.setdefault((4, "srf_track"), []).extend(pl.polygon(e))
        if le == "playground":
            layers.setdefault((4, "srf_sand"), []).extend(pl.polygon(e))
        if t.get("amenity") == "parking":
            layers.setdefault((3, "srf_parking"), []).extend(pl.polygon(e))
        if nat == "water" or t.get("water") or t.get("waterway") in ("riverbank",):
            layers.setdefault((5, "srf_water"), []).extend(pl.polygon(e))
    W.water = []
    for k_, ((lvl, mat), P_) in enumerate(
        sorted(layers.items())
    ):  # every layer at its own height: no z-fighting
        g = unary_union([p.buffer(0) for p in P_ if p.is_valid or p.buffer(0).is_valid])
        surface(bank, g, mat, 0.004 + 0.003 * k_, 12.0 if mat != "srf_parking" else 16.0)
        if mat == "srf_water":
            W.water += polys(g)
            W.map["water"] += polys(g)
        if lvl == 4:
            W.landmarks += [
                {
                    "kind": "pitch" if mat == "srf_pitch" else "playground" if mat == "srf_sand" else "track",
                    "poly": p,
                }
                for p in polys(g)
                if p.area > 150
            ]
        if mat == "srf_parking":
            W.landmarks += [{"kind": "parking", "poly": p} for p in polys(g) if p.area > 150]
    for e in pl.E:  # rivers / streams drawn as lines
        if tg(e, "waterway") in ("river", "stream", "canal", "ditch") and e["type"] == "way":
            ln = pl.line(e)
            if len(ln) > 1:
                g = (
                    LineString(ln)
                    .buffer({"river": 12, "canal": 8}.get(tg(e, "waterway"), 2.5) / 2)
                    .intersection(pl.area)
                )
                surface(bank, g, "srf_water", 0.052, 12.0)
                W.water += polys(g)
    # ---- roads (one asphalt sheet), sidewalks, lane marks
    roads, feet, names = [], [], []
    for e in pl.E:
        hw = tg(e, "highway")
        if not hw or e["type"] != "way" or hw == "street_lamp" or tg(e, "area") == "yes":
            continue
        ln = pl.line(e)
        if len(ln) < 2:
            continue
        w = (num(tg(e, "lanes")) or 0) * 3.2 or ROAD_W.get(hw, 4.0)
        if hw in ("track",) and tg(e, "surface") in ("dirt", "ground", "grass", None):
            pass
        geom = LineString(ln).buffer(w / 2, cap_style=2, join_style=1)
        (feet if hw in FOOT else roads).append(geom)
        if hw in FOOT:
            W.footways.append(ln)
        if hw not in FOOT:
            W.map["roads"].append({"line": ln, "width": w, "kind": hw, "name": tg(e, "name")})
            if w >= 6:  # dashed centre line
                for (x1, y1), (x2, y2) in zip(ln[:-1], ln[1:]):
                    L = math.hypot(x2 - x1, y2 - y1)
                    ux, uy = (x2 - x1) / max(L, 1e-6), (y2 - y1) / max(L, 1e-6)
                    nx, ny = -uy * 0.08, ux * 0.08
                    for s in np.arange(1.0, L - 3, 9.0):
                        a, b = (x1 + ux * s, y1 + uy * s), (x1 + ux * (s + 3), y1 + uy * (s + 3))
                        if inside(*a) and inside(*b):
                            bank.quad(
                                "paint",
                                (a[0] - nx, a[1] - ny, 0.066),
                                (b[0] - nx, b[1] - ny, 0.066),
                                (b[0] + nx, b[1] + ny, 0.066),
                                (a[0] + nx, a[1] + ny, 0.066),
                                [(0, 0), (1, 0), (1, 1), (0, 1)],
                            )
    road_g = unary_union(roads).intersection(pl.area) if roads else None
    surface(bank, road_g, "srf_asphalt", 0.060, 8.0)
    if feet:
        surface(
            bank,
            (
                unary_union(feet).intersection(pl.area).difference(road_g)
                if road_g is not None
                else unary_union(feet).intersection(pl.area)
            ),
            "srf_paving",
            0.056,
            4.0,
        )
    W.road_area = road_g
    # ---- buildings
    footprints = []
    for i, e in enumerate(e for e in pl.E if tg(e, "building") and tg(e, "building") not in ("roof", "no")):
        tags = e.get("tags", {})
        for j, poly in enumerate(pl.polygon(e)):
            if poly.area < 6:
                continue
            h, lv, src = height_of(tags, pl.kind, poly.area, rng)
            stats["height_src"][src] = stats["height_src"].get(src, 0) + 1
            style, roof, roof_mat, btype = building_style(tags, pl.kind, poly.area, h, rng)
            gf = (
                btype not in HOUSE | GARAGE
                and pl.kind
                in ("city_center", "city_abroad", "panel_district", "panel_district_abroad", "foreign_city")
                and lv >= 3
            )
            name = f"b{i}_{j}"
            map_h = h
            if map_errors and src != "height" and rng.random() < 0.04:
                map_h = round(h * rng.choice([0.6, 1.5]), 1)
                W.map_errors.append({"kind": "height", "building": name, "true": h, "map": map_h})
            top = add_building(
                bank,
                A,
                G,
                name,
                poly,
                h,
                lv if src != "estimated" else 0,
                style,
                roof,
                f"srf_{roof_mat}",
                gf,
                rng,
            )
            c = poly.centroid
            rec = {
                "name": name,
                "poly": poly,
                "h": h,
                "top": top,
                "levels": int(lv),
                "style": style,
                "type": btype,
                "colour": colour_word(tags.get("building:colour")),
                "roof": roof,
                "roof_mat": roof_mat,
                "flat_roof": roof == "flat" or house_rect(poly) is None,
                "street": tags.get("addr:street"),
                "number": tags.get("addr:housenumber"),
            }
            footprints.append(rec)
            W.buildings.append(
                (c.x, c.y, poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1], top, style)
            )
            if not (map_errors and rng.random() < 0.015):
                W.map["buildings"].append({**rec, "h": map_h})
            else:
                W.map_errors.append({"kind": "missing", "building": name})
            if btype in CHURCH and poly.area > 80:  # drum + dome: a landmark everyone describes
                r = min(5.0, math.sqrt(poly.area) / 4)
                A, G = A, G + [
                    f'<geom name="{name}_drum" type="cylinder" pos="{c.x:.2f} {c.y:.2f} {h + 2:.2f}" size="{r:.2f} 2" material="fac_plaster_white"/>'
                    f'<geom name="{name}_dome" type="sphere" pos="{c.x:.2f} {c.y:.2f} {h + 4:.2f}" size="{r * 1.05:.2f}" rgba=".85 .68 .2 1"/>'
                ]
                W.landmarks.append({"kind": "church", "poly": poly, "h": h + 4 + r})
    W.footprints = footprints
    taken = unary_union([f["poly"] for f in footprints]) if footprints else None

    def free(x, y, r):
        return inside(x, y, 2) and (taken is None or taken.distance(Point(x, y)) >= r)

    # ---- trees: mapped single trees and tree rows, woods
    trees = []
    for e in pl.E:
        t = e.get("tags", {})
        if e["type"] == "node" and t.get("natural") == "tree":
            trees.append((*pl.P(e["lat"], e["lon"]), rng.uniform(6, 14), t.get("leaf_type"), True))
        if t.get("natural") == "tree_row":
            ln = LineString(pl.line(e))
            for s in np.arange(0, ln.length, 7.0):
                p = ln.interpolate(s)
                trees.append((p.x, p.y, rng.uniform(6, 12), t.get("leaf_type"), True))
    for e in pl.E:
        t = e.get("tags", {})
        if not (t.get("natural") == "wood" or t.get("landuse") == "forest"):
            continue
        for po in pl.polygon(e):
            n = min(int(po.area / 45.0), 900 - len(trees))
            bx = po.bounds
            pts = []
            tries = 0
            while len(pts) < n and tries < n * 15:
                tries += 1
                x, y = rng.uniform(bx[0], bx[2]), rng.uniform(bx[1], bx[3])
                if (
                    po.contains(Point(x, y))
                    and free(x, y, 2.5)
                    and (road_g is None or road_g.distance(Point(x, y)) > 2)
                    and all((x - a) ** 2 + (y - b) ** 2 > 12 for a, b in pts[-80:])
                ):
                    pts.append((x, y))
            trees += [
                (
                    x,
                    y,
                    rng.uniform(10, 24),
                    t.get("leaf_type") or ("needleleaved" if pl.kind in ("dachas_in_pines",) else "mixed"),
                    True,
                )
                for x, y in pts
            ]
    for (lvl, mat), P_ in layers.items():  # unmapped trees: parks and residential yards
        if mat != "srf_grass":
            continue
        for po in P_:
            n = min(int(po.area / 350.0), 1200 - len(trees))
            bx = po.bounds
            for _ in range(n * 6):
                if n <= 0:
                    break
                x, y = rng.uniform(bx[0], bx[2]), rng.uniform(bx[1], bx[3])
                if (
                    po.contains(Point(x, y))
                    and free(x, y, 4)
                    and (road_g is None or road_g.distance(Point(x, y)) > 3)
                ):
                    trees.append((x, y, rng.uniform(7, 16), None, False))
                    n -= 1
    tree_xml = []
    for k, (x, y, h, leaf, mapped) in enumerate(trees):
        if not inside(x, y):
            continue
        r = random.Random(zlib.crc32(f"{place}/{seed}/tree{k}".encode()))
        sp = (
            r.choice(["spruce", "pine"])
            if leaf == "needleleaved"
            else (
                r.choice(["spruce", "pine", "birch", "broad"])
                if leaf == "mixed"
                else r.choice(["birch", "broad", "broad"]) if pl.kind != "foreign_city" else "broad"
            )
        )
        tree_xml.append(tree_geoms(k, x, y, h, sp, r))
        W.trees.append((x, y, h))
        if not mapped:
            W.unmapped.append({"kind": "tree", "x": x, "y": y})
    # ---- lamps, fences, power lines, mapped landmarks
    misc = []
    for e in pl.E:
        t = e.get("tags", {})
        if e["type"] == "node" and t.get("highway") == "street_lamp":
            x, y = pl.P(e["lat"], e["lon"])
            if inside(x, y):
                misc.append(lamp_geoms(len(W.lamps), x, y, rng.uniform(0, 6.28)))
                W.lamps.append((x, y))
        if t.get("barrier") in ("fence", "wall", "hedge") and e["type"] == "way":
            ln = pl.line(e)
            hgt, w, mat = {
                "fence": (1.6, 0.03, "srf_metal"),
                "wall": (2.0, 0.12, "srf_concrete"),
                "hedge": (1.2, 0.4, "srf_foliage0"),
            }[t["barrier"]]
            for k, ((x1, y1), (x2, y2)) in enumerate(zip(ln[:-1], ln[1:])):
                L = math.hypot(x2 - x1, y2 - y1)
                if L < 0.3 or not (inside(x1, y1) and inside(x2, y2)):
                    continue
                misc.append(
                    f'<geom name="fence{e["id"]}_{k}" type="box" pos="{(x1 + x2) / 2:.2f} {(y1 + y2) / 2:.2f} {hgt / 2:.2f}" size="{L / 2:.2f} {w:.2f} {hgt / 2:.2f}" '
                    f'euler="0 0 {math.degrees(math.atan2(y2 - y1, x2 - x1)):.1f}" material="{mat}"/>'
                )
        mm = t.get("man_made")
        if mm in ("water_tower", "mast", "tower", "chimney", "storage_tank", "flagpole"):
            if e["type"] == "node":
                x, y = pl.P(e["lat"], e["lon"])
            else:
                ps = pl.polygon(e)
                if not ps:
                    continue
                x, y = ps[0].centroid.x, ps[0].centroid.y
            if inside(x, y, 5):
                g, h = man_made_geoms(len(W.landmarks), x, y, mm, t, rng)
                misc.append(g)
                W.landmarks.append({"kind": mm, "x": x, "y": y, "h": h})
                W.map["landmarks"].append({"kind": mm, "x": x, "y": y, "h": h})
    wire_n = 0
    for e in pl.E:
        pw = tg(e, "power")
        if pw not in ("line", "minor_line") or e["type"] != "way":
            continue
        hv = pw == "line"
        th = num(tg(e, "height")) or (26.0 if hv else 10.0)
        V = pl.line(e)
        offs = [(-3.5, -2.5), (3.5, -2.5), (0.0, 0.0)] if hv else [(-0.5, -0.3), (0.5, -0.3)]
        for k, (x, y) in enumerate(V):
            if inside(x, y):
                misc.append(
                    f'<geom name="ptw{wire_n}_{k}" type="box" pos="{x:.2f} {y:.2f} {th / 2:.2f}" size="{.9 if hv else .14} {.9 if hv else .14} {th / 2:.2f}" material="srf_metal"/>'
                    + (
                        f'<geom name="ptx{wire_n}_{k}" type="box" pos="{x:.2f} {y:.2f} {th - 1:.2f}" size="4.2 .25 .15" material="srf_metal"/>'
                        if hv
                        else f'<geom name="ptx{wire_n}_{k}" type="box" pos="{x:.2f} {y:.2f} {th - .2:.2f}" size=".7 .06 .06" material="srf_metal"/>'
                    )
                )
        for k in range(len(V) - 1):
            (x1, y1), (x2, y2) = V[k], V[k + 1]
            if not (inside(x1, y1) and inside(x2, y2)):
                continue
            dx, dy = x2 - x1, y2 - y1
            L = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / L, dx / L
            sag = 0.03 * L
            for c, (o, dz) in enumerate(offs):
                ax, ay, bx_, by_ = x1 + nx * o, y1 + ny * o, x2 + nx * o, y2 + ny * o
                z = th + dz - (1.0 if hv else 0.2)
                mx, my = (ax + bx_) / 2, (ay + by_) / 2
                for s_, (p1, p2) in enumerate(
                    (((ax, ay, z), (mx, my, z - sag)), ((mx, my, z - sag), (bx_, by_, z)))
                ):
                    misc.append(
                        f'<geom name="wire{wire_n}_{k}_{c}_{s_}" type="capsule" fromto="{f2(p1)} {f2(p2)}" size="{.02 if hv else .012}" rgba=".12 .12 .12 1"/>'
                    )
            W.wires.append((x1, y1, x2, y2, th))
            W.map["wires"].append((x1, y1, x2, y2, th))
        wire_n += 1
    filler(bank, A, G, pl.kind, rng, tree_xml)
    W.stats = {
        **stats,
        "buildings": len(footprints),
        "trees": len(W.trees),
        "unmapped_trees": len(W.unmapped),
        "lamps": len(W.lamps),
        "wires": len(W.wires),
        "landmarks": len(W.landmarks),
        "map_errors": len(W.map_errors),
    }
    W.map["areas"] = [{"kind": l["kind"], "poly": l["poly"]} for l in W.landmarks if "poly" in l]
    W.sun = sun(rng)
    W.free = free
    W.start = home_spot(W, rng, road_g)
    if actors:
        import actors as AC

        W.movers = AC.make_actors(W, np.random.default_rng(zlib.crc32(f"{place}/{seed}/actors".encode())))
    va, vg = bank.xml()
    assets = asset_xml() + va + A + cone_assets() + list(extra_assets)
    static = "".join(vg) + "".join(G) + "".join(tree_xml) + "".join(misc) + extra_xml
    return W, static, assets


def filler(bank, A, G, kind, rng, tree_xml):
    """Blocks of the same character beyond the 500 x 500 m area up to ~700 m out, so the map does not end."""
    forest = kind in ("buildings_in_forest", "power_line_forest", "dachas_in_pines")
    k = 0
    for gx in np.arange(-HALF - 420, HALF + 421, 60.0):
        for gy in np.arange(-HALF - 420, HALF + 421, 60.0):
            if abs(gx) < HALF + 30 and abs(gy) < HALF + 30:
                continue
            x, y = gx + rng.uniform(-12, 12), gy + rng.uniform(-12, 12)
            if forest or rng.random() < 0.2:
                for t in range(4 if forest else 2):
                    tx, ty = x + rng.uniform(-25, 25), y + rng.uniform(-25, 25)
                    h = rng.uniform(12, 22)
                    tree_xml.append(
                        f'<geom name="ftr{k}_{t}" type="ellipsoid" pos="{tx:.1f} {ty:.1f} {h * .62:.1f}" size="{h * .3:.1f} {h * .3:.1f} {h * .42:.1f}" '
                        f'material="srf_foliage{t % 3}" contype="0" conaffinity="0" group="1"/>'
                        f'<geom name="ftt{k}_{t}" type="cylinder" pos="{tx:.1f} {ty:.1f} {h * .2:.1f}" size=".3 {h * .2:.1f}" material="srf_bark" contype="0" conaffinity="0" group="1"/>'
                    )
                k += 1
                continue
            if kind in ("private_houses", "suburb_abroad", "power_line_suburb"):
                w, d_, lv, style = (
                    rng.uniform(8, 14),
                    rng.uniform(8, 12),
                    rng.choice([1, 2]),
                    rng.choice(["house_siding", "house_plaster", "brick_red"]),
                )
            elif kind in ("panel_district", "panel_district_abroad"):
                w, d_, lv, style = (
                    rng.uniform(40, 55),
                    rng.uniform(12, 15),
                    rng.choice([9, 12, 16]),
                    rng.choice(["panel_light", "panel_beige", "panel_blue"]),
                )
            elif kind == "industrial":
                w, d_, lv, style = rng.uniform(30, 50), rng.uniform(20, 40), rng.choice([2, 3]), "industrial"
            elif kind == "foreign_city":
                w, d_, lv, style = (
                    rng.uniform(12, 25),
                    rng.uniform(10, 20),
                    rng.choice([3, 4, 6, 8]),
                    rng.choice(["jp_tile", "plaster_white"]),
                )
            else:
                w, d_, lv, style = (
                    rng.uniform(25, 45),
                    rng.uniform(14, 30),
                    rng.choice([4, 5, 6]),
                    rng.choice(
                        ["plaster_yellow", "plaster_pink", "brick_red", "panel_light", "plaster_white"]
                    ),
                )
            a = rng.choice([0.0, math.pi / 2]) + rng.uniform(-0.1, 0.1)
            c, s_ = math.cos(a), math.sin(a)
            corners = [
                (x + c * u - s_ * v, y + s_ * u + c * v)
                for u, v in ((-w / 2, -d_ / 2), (w / 2, -d_ / 2), (w / 2, d_ / 2), (-w / 2, d_ / 2))
            ]
            add_building(
                bank,
                A,
                G,
                f"fill{k}",
                Polygon(corners),
                lv * 3.0 + 1.0,
                lv,
                style,
                "gable" if lv <= 2 else "flat",
                "srf_roof_metal_red" if lv <= 2 else "srf_roof_bitumen",
                False,
                rng,
            )
            k += 1


def home_spot(W, rng, road_g):
    """Home pad: open ground near the centre (no building within 8 m, no tree within 6 m, off the roads)."""
    for r in (60.0, 120.0, 200.0):
        for _ in range(3000):
            x, y = rng.uniform(-r, r), rng.uniform(-r, r)
            if (
                W.free(x, y, 8.0)
                and all((x - a) ** 2 + (y - b) ** 2 > 36 for a, b, _ in W.trees)
                and (road_g is None or road_g.distance(Point(x, y)) > 4)
            ):
                return (round(x, 2), round(y, 2))
    return (0.0, 0.0)


def sun(rng):
    az, el = rng.uniform(0, 2 * math.pi), rng.uniform(0.4, 1.15)
    return (math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), -math.sin(el), rng.uniform(0.62, 0.85))


MATERIALS = (
    [f"fac_{s}" for s in TX.STYLES] + [f"gf_{s}" for s in TX.STYLES] + [f"srf_{s}" for s in TX.SURFACES]
)


def asset_xml():
    out = []
    for m in MATERIALS:
        spec = (
            ' specular=".5" shininess=".6" reflectance=".08"'
            if "glass" in m or "water" in m
            else ' specular=".08" shininess=".1"'
        )
        out.append(
            f'<texture name="t_{m}" type="2d" file="{m}.png"/><material name="{m}" texture="t_{m}" texrepeat="1 1"{spec}/>'
        )
    for k, g in enumerate((0.9, 1.0, 1.12)):  # three foliage shades
        out.append(
            f'<material name="srf_foliage{k}" texture="t_srf_foliage" texrepeat="2 2" rgba="{g * .95:.2f} {g:.2f} {g * .9:.2f} 1" specular=".05"/>'
        )
    out.append('<material name="paint" rgba=".92 .92 .88 1" specular=".1"/>')
    out.append('<texture name="sky" type="skybox" file="sky.png" gridsize="3 4" gridlayout=".U..LFRB.D.."/>')
    return out


def texture_files():
    TX.ensure()
    return {p.name: p.read_bytes() for p in TX.TEX_DIR.glob("*.png")}


def drone_xml(z0=0.3):
    """The physical drone of world.py."""
    body, motors = W0.drone_xml(z0)
    L = W0.ARM / math.sqrt(2)
    v = 'contype="0" conaffinity="0" density="0" group="1"'
    vis = (
        f'<geom type="ellipsoid" size=".13 .09 .045" pos="0 0 .01" rgba=".92 .93 .95 1" {v}/>'
        f'<geom type="box" size=".05 .05 .012" pos="0 0 .055" rgba=".15 .16 .18 1" {v}/>'
        + "".join(
            f'<geom type="cylinder" pos="{sx * L:.3f} {sy * L:.3f} .015" size=".025 .02" rgba=".2 .2 .22 1" {v}/>'
            for sx, sy, _ in W0.ROTORS
        )
        + "".join(
            f'<geom type="capsule" fromto="-.1 {s * .07:.2f} -.11 .1 {s * .07:.2f} -.11" size=".008" rgba=".2 .2 .2 1" {v}/>'
            f'<geom type="capsule" fromto="0 {s * .05:.2f} -.04 0 {s * .07:.2f} -.11" size=".007" rgba=".2 .2 .2 1" {v}/>'
            for s in (-1, 1)
        )
        + f'<geom type="sphere" pos=".12 0 -.05" size=".032" rgba=".18 .18 .2 1" {v}/><geom type="cylinder" pos=".15 0 -.05" size=".014 .006" zaxis="1 0 0" rgba=".05 .05 .1 1" {v}/>'
        + f'<geom type="sphere" pos="{L:.3f} {L:.3f} 0" size=".012" rgba="0 1 0 1" {v}/><geom type="sphere" pos="{L:.3f} {-L:.3f} 0" size=".012" rgba="1 0 0 1" {v}/>'
        + '<camera name="gimbal" pos=".12 0 -.26" xyaxes="0 -1 0 0 0 1" fovy="70"/>'
    )  # under the skids: body and arms stay out of view
    return body.replace('<site name="imu"', vis + '<site name="imu"', 1), motors


def to_mjcf(W, static, assets, z0=0.3, cam=(640, 480), dyn=""):
    """Whole model: realistic static world + drone + movers + projectiles + encounter bodies."""
    body, motors = drone_xml(z0)
    sx, sy, sz, si = W.sun
    body = body.replace(
        '<site name="imu"',
        f'<light name="sun" mode="track" pos="{-sx * 80:.1f} {-sy * 80:.1f} {-sz * 80:.1f}" dir="{sx:.3f} {sy:.3f} {sz:.3f}" '
        f'directional="true" castshadow="true" diffuse="{si:.2f} {si * .97:.2f} {si * .9:.2f}" specular=".2 .2 .2"/><site name="imu"',
        1,
    )
    import actors as AC

    movers = "".join(
        AC.mover_xml(i, mv) if mv.get("v2") else W0.mover_xml(i, mv) for i, mv in enumerate(W.movers)
    )
    movers += "".join(
        f'<body name="proj{i}" mocap="true" pos="0 0 -20"><geom name="proj{i}_g" type="sphere" size=".11" rgba=".55 .5 .45 1"/></body>'
        for i in range(W0.N_PROJECTILES)
    )
    movers += "".join(W0.enc_xml(i, s) for i, s in enumerate(W0.ENC_SIZES))
    movers += dyn + "".join(getattr(o, "dyn", "") for o in getattr(W, "objects", []))
    movers += (
        '<body name="cargo" mocap="true" pos="0 0 -60"><geom name="cargo_box" type="box" size=".18 .14 .14" rgba=".72 .55 .33 1"/>'  # parcel for parachute drops
        '<geom name="cargo_chute" type="ellipsoid" pos="0 0 1.6" size=".75 .75 .45" rgba=".95 .45 .1 0" contype="0" conaffinity="0"/>'
        '<geom name="cargo_line" type="capsule" fromto="0 0 .14 0 0 1.3" size=".005" rgba=".2 .2 .2 0" contype="0" conaffinity="0"/></body>'
        '<body name="hook" mocap="true" pos="0 0 -60"><geom name="hook_g" type="capsule" fromto="0 0 0 0 0 .08" size=".02" rgba=".15 .15 .15 1" contype="0" conaffinity="0"/></body>'
        '<body name="tether" mocap="true" pos="0 0 -60"><geom name="tether_g" type="capsule" fromto="0 0 0 0 0 1" size=".004" rgba=".9 .9 .9 1" contype="0" conaffinity="0"/></body>'
    )
    return f"""<mujoco model="city_{W.place}_{W.seed}">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="Euler" density="0"/>
  <statistic extent="100" center="0 0 10"/>
  <visual><global offwidth="{max(cam[0], 1280)}" offheight="{max(cam[1], 960)}"/><quality shadowsize="4096" offsamples="4"/>
    <headlight ambient=".42 .43 .46" diffuse=".12 .12 .12" specular="0 0 0"/>
    <map znear=".0005" zfar="15" fogstart="2.5" fogend="16" shadowclip=".8" haze=".3"/><rgba fog=".80 .85 .9 1" haze=".82 .87 .93 1"/></visual>
  <asset>
    <texture name="grass_g" type="2d" file="srf_grass.png"/><material name="ground" texture="grass_g" texrepeat=".1 .1" texuniform="true" specular=".05"/>
    {''.join(assets)}
  </asset>
  <worldbody>
    <geom name="ground" type="plane" size="{HALF * 4} {HALF * 4} .1" material="ground"/>
    <geom name="pad" type="cylinder" pos="{W.start[0]:.2f} {W.start[1]:.2f} .04" size="1.5 .01" rgba=".95 .95 .95 1"/>
    <body name="place" pos="0 0 0">{static}</body>
    {movers}
    {body}
  </worldbody>
  <actuator>{motors}</actuator>
  <sensor><accelerometer name="acc" site="imu"/><gyro name="gyro" site="imu"/><rangefinder name="range_down" site="rf" cutoff="40"/></sensor>
</mujoco>"""
