"""Stage F mission objects."""

import math
from dataclasses import dataclass, field
import numpy as np

COLOURS = {  # name: RGB 0..1 (names are the words a description may use)
    "red": (0.82, 0.14, 0.12),
    "orange": (0.95, 0.48, 0.10),
    "yellow": (0.96, 0.80, 0.12),
    "green": (0.18, 0.55, 0.22),
    "blue": (0.16, 0.32, 0.80),
    "white": (0.93, 0.93, 0.91),
    "grey": (0.52, 0.54, 0.56),
    "black": (0.10, 0.10, 0.11),
    "purple": (0.50, 0.22, 0.62),
    "brown": (0.45, 0.30, 0.18),
}
KINDS = (
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
)
ACTIONS = {
    "pad": ("land", "flyby"),
    "dropzone": ("drop", "flyby"),
    "pickup": ("pickup",),
    "hangar": ("enter", "flyby"),
    "water_tower": ("inspect", "flyby"),
    "mast": ("inspect", "flyby"),
    "tent": ("flyby", "drop"),
    "container": ("land", "flyby"),
    "solar": ("inspect", "flyby"),
    "windsock": (),
}


@dataclass
class Obj:
    oid: str
    kind: str
    x: float
    y: float
    yaw: float
    z: float
    xml: str = ""
    facts: dict = field(default_factory=dict)
    radius: float = 1.0
    height: float = 1.0
    land: tuple = None  # (x, y, z_surface, radius) where a landing / parcel touchdown counts
    hook: tuple = None  # (x, y, z) of the parcel handle (pickup)
    door: dict = None  # hangar: centre, outward normal, width, height, open
    on_roof: str = None  # building name if placed on a roof
    dyn: str = ""  # parts that can move


def quat_yaw_roll(yaw, roll):
    """Orientation 'turn by yaw about the vertical, after tilting by roll about the object's own long."""
    cy, sy, cr, sr = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(roll / 2), math.sin(roll / 2)
    return f"{cy * cr:.5f} {cy * sr:.5f} {sy * sr:.5f} {sy * cr:.5f}"


def rgba(c, a=1.0):
    r, g, b = COLOURS[c] if isinstance(c, str) else c
    return f"{r:.2f} {g:.2f} {b:.2f} {a:.2f}"


def contrast(c):
    return "white" if c in ("red", "orange", "green", "blue", "black", "purple", "brown", "grey") else "black"


def _rot(yaw, u, v):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * u - s * v, s * u + c * v


def _box(n, x, y, z, sx, sy, sz, yaw, col, extra=""):
    return f'<geom name="{n}" type="box" pos="{x:.3f} {y:.3f} {z:.3f}" size="{sx:.3f} {sy:.3f} {sz:.3f}" euler="0 0 {math.degrees(yaw):.1f}" rgba="{col}"{extra}/>'


def _cyl(n, x, y, z, r, hh, col, extra=""):
    return f'<geom name="{n}" type="cylinder" pos="{x:.3f} {y:.3f} {z:.3f}" size="{r:.3f} {hh:.3f}" rgba="{col}"{extra}/>'


def pad(o, rng, colour=None, size=None, shape=None, mark=None, lights=None):
    colour = colour or rng.choice(["orange", "yellow", "blue", "green", "red", "white", "grey", "black"])
    D = size or round(rng.uniform(2.5, 6.0), 1)
    shape = shape or rng.choice(["round", "round", "square"])
    mark = mark or rng.choice(["H", "H", "H", "circle", "X"])
    lights = rng.random() < 0.3 if lights is None else lights
    mc = rgba(contrast(colour))
    x, y, z, yaw, n = o.x, o.y, o.z, o.yaw, o.oid
    R = D / 2
    t = 0.04
    g = [
        (
            _cyl(f"{n}_b", x, y, z + t, R, t, rgba(colour))
            if shape == "round"
            else _box(f"{n}_b", x, y, z + t, R, R, t, yaw, rgba(colour))
        )
    ]
    zt = z + 2 * t + 0.004
    if mark == "H":
        for k, (u, v, a, b) in enumerate(
            (
                (-R * 0.32, 0, R * 0.07, R * 0.42),
                (R * 0.32, 0, R * 0.07, R * 0.42),
                (0, 0, R * 0.32, R * 0.06),
            )
        ):
            px, py = _rot(yaw, u, v)
            g.append(_box(f"{n}_m{k}", x + px, y + py, zt, a, b, 0.004, yaw, mc))
    elif mark == "X":
        for k, a in enumerate((0.785, -0.785)):
            g.append(_box(f"{n}_m{k}", x, y, zt, R * 0.62, R * 0.07, 0.004, yaw + a, mc))
    if mark == "circle" or (mark == "H" and rng.random() < 0.5):  # painted ring = many short segments
        for k in range(24):
            a = 2 * math.pi * k / 24
            g.append(
                _box(
                    f"{n}_r{k}",
                    x + R * 0.8 * math.cos(a),
                    y + R * 0.8 * math.sin(a),
                    zt,
                    R * 0.05,
                    R * 0.12,
                    0.004,
                    a,
                    mc,
                )
            )
    if lights:
        for k in range(8):
            a = 2 * math.pi * k / 8
            g.append(
                _box(
                    f"{n}_l{k}",
                    x + R * 0.97 * math.cos(a),
                    y + R * 0.97 * math.sin(a),
                    zt + 0.02,
                    0.07,
                    0.07,
                    0.03,
                    a,
                    rgba("yellow"),
                )
            )
    o.xml = "".join(g)
    o.radius = R
    o.height = 0.1
    o.land = (x, y, zt, R * 0.8)
    o.facts = {
        "kind": "pad",
        "colour": colour,
        "size_m": D,
        "shape": shape,
        "mark": mark,
        "lights": lights,
        "mark_colour": contrast(colour),
    }


def dropzone(o, rng, colour=None, size=None, pattern=None, flags=None):
    colour = colour or rng.choice(["orange", "red", "yellow", "blue"])
    S = size or round(rng.uniform(6.0, 10.0), 1)
    pattern = pattern or rng.choice(["X", "rings", "cross"])
    flags = rng.random() < 0.5 if flags is None else flags
    x, y, z, yaw, n = o.x, o.y, o.z, o.yaw, o.oid
    R = S / 2
    zt = z + 0.03
    second = "white" if colour != "yellow" else "black"
    g = [_box(f"{n}_b", x, y, z + 0.015, R, R, 0.015, yaw, rgba(second if pattern == "rings" else colour))]
    if pattern == "X":
        for k, a in enumerate((0.785, -0.785)):
            g.append(_box(f"{n}_m{k}", x, y, zt, R * 1.2, R * 0.14, 0.005, yaw + a, rgba(second)))
    elif pattern == "cross":
        for k, a in enumerate((0.0, 1.5708)):
            g.append(_box(f"{n}_m{k}", x, y, zt, R * 0.9, R * 0.14, 0.005, yaw + a, rgba(second)))
    else:
        for k, (rr, c) in enumerate(((0.82, colour), (0.62, second), (0.42, colour), (0.2, second))):
            g.append(_cyl(f"{n}_m{k}", x, y, zt + 0.002 * k, R * rr, 0.005, rgba(c)))
    if flags:
        for k, (u, v) in enumerate(((-R, -R), (R, -R), (R, R), (-R, R))):
            px, py = _rot(yaw, u, v)
            g.append(_cyl(f"{n}_p{k}", x + px, y + py, z + 0.9, 0.025, 0.9, rgba("white")))
            g.append(_box(f"{n}_f{k}", x + px + 0.25, y + py, z + 1.6, 0.25, 0.01, 0.16, 0.0, rgba(colour)))
    o.xml = "".join(g)
    o.radius = R * 1.42
    o.height = 1.8 if flags else 0.05
    o.land = (x, y, zt, R * 0.5)
    o.facts = {
        "kind": "dropzone",
        "colour": colour,
        "size_m": S,
        "pattern": pattern,
        "flags": flags,
        "second_colour": second,
    }


def pickup(o, rng, colour=None, parcel=None, style=None):
    """A pickup point: a coloured platform / locker with a parcel on top that has a handle loop for the."""
    colour = colour or rng.choice(["green", "blue", "orange", "yellow", "red", "white"])
    style = style or rng.choice(["platform", "locker"])
    pc = parcel or rng.choice(["brown", "white", "yellow"])
    x, y, z, yaw, n = o.x, o.y, o.z, o.yaw, o.oid
    if style == "locker":
        top = 1.6
        g = [_box(f"{n}_b", x, y, z + 0.8, 0.7, 0.45, 0.8, yaw, rgba(colour))] + [
            _box(
                f"{n}_d{k}",
                *(_v(x, y, yaw, -0.35 + 0.35 * k, -0.46)),
                z + 0.9,
                0.15,
                0.01,
                0.55,
                yaw,
                rgba("grey"),
            )
            for k in range(3)
        ]
    else:
        top = 0.5
        g = [
            _box(f"{n}_b", x, y, z + 0.25, 0.9, 0.9, 0.25, yaw, rgba(colour)),
            _box(f"{n}_s", x, y, z + 0.52, 0.8, 0.8, 0.02, yaw, rgba("grey")),
        ]
    ps = round(rng.uniform(0.3, 0.5), 2)
    zp = z + top + ps / 2 + 0.02
    # the parcel is a separate body
    o.dyn = (
        f'<body name="{n}_parcel" mocap="true" pos="{x:.3f} {y:.3f} {zp:.3f}" euler="0 0 {math.degrees(yaw):.1f}">'
        f'<geom name="{n}_p" type="box" size="{ps / 2:.3f} {ps / 2 * .8:.3f} {ps / 2:.3f}" rgba="{rgba(pc)}"/>'
        f'<geom name="{n}_h" type="capsule" fromto="-.08 0 {ps / 2:.3f} 0 0 {ps / 2 + .12:.3f}" size=".012" rgba="{rgba("black")}"/>'
        f'<geom name="{n}_h2" type="capsule" fromto="0 0 {ps / 2 + .12:.3f} .08 0 {ps / 2:.3f}" size=".012" rgba="{rgba("black")}"/></body>'
    )
    o.xml = "".join(g)
    o.radius = 1.3
    o.height = zp + ps / 2 + 0.12 - z
    o.hook = (x, y, zp + ps / 2 + 0.12)
    o.land = (x, y, z + top, 0.6)
    o.parcel = (x, y, zp, ps)
    o.facts = {
        "kind": "pickup",
        "colour": colour,
        "style": style,
        "parcel_colour": pc,
        "parcel_m": ps,
        "size_m": 1.8 if style == "platform" else 1.4,
    }


def _v(x, y, yaw, u, v):
    px, py = _rot(yaw, u, v)
    return x + px, y + py


def hangar(o, rng, colour=None, width=None, door_open=None):
    """A small drone / vehicle hangar with a front opening (open or closed roller door) and an H inside."""
    colour = colour or rng.choice(["grey", "green", "blue", "white", "red"])
    W = width or round(rng.uniform(6.0, 9.0), 1)
    Dp = round(rng.uniform(6.0, 9.0), 1)
    H = round(rng.uniform(3.6, 4.8), 1)
    dw, dh = round(min(W - 2.0, rng.uniform(3.2, 4.6)), 1), round(min(H - 0.6, rng.uniform(2.6, 3.4)), 1)
    door_open = rng.random() < 0.85 if door_open is None else door_open
    x, y, z, yaw, n = o.x, o.y, o.z, o.yaw, o.oid
    t = 0.12
    c = rgba(colour)
    g = []

    def wall(k, u, v, sx, sy, zc, sz):
        px, py = _rot(yaw, u, v)
        g.append(_box(f"{n}_w{k}", x + px, y + py, z + zc, sx, sy, sz, yaw, c))

    wall(0, 0, -Dp / 2, W / 2, t, H / 2, H / 2)  # back wall (local -v), front is +v
    wall(1, -W / 2, 0, t, Dp / 2, H / 2, H / 2)
    wall(2, W / 2, 0, t, Dp / 2, H / 2, H / 2)
    side = (W - dw) / 2
    wall(3, -W / 2 + side / 2, Dp / 2, side / 2, t, H / 2, H / 2)
    wall(4, W / 2 - side / 2, Dp / 2, side / 2, t, H / 2, H / 2)
    wall(5, 0, Dp / 2, dw / 2, t, (H + dh) / 2, (H - dh) / 2)  # lintel above the door
    px, py = _rot(yaw, 0, 0)
    g.append(_box(f"{n}_roof", x, y, z + H + 0.06, W / 2 + 0.25, Dp / 2 + 0.25, 0.06, yaw, rgba("grey")))
    if not door_open:
        wall(6, 0, Dp / 2 - 0.05, dw / 2, 0.05, dh / 2, dh / 2)
    for k, (u, v, a, b) in enumerate(
        ((-0.5, 0, 0.1, 0.6), (0.5, 0, 0.1, 0.6), (0, 0, 0.5, 0.1))
    ):  # yellow H on the floor inside
        qx, qy = _v(x, y, yaw, u, v)
        g.append(_box(f"{n}_h{k}", qx, qy, z + 0.012, a, b, 0.006, yaw, rgba("yellow")))
    nx, ny = _rot(yaw, 0, 1.0)
    cx, cy = _v(x, y, yaw, 0, Dp / 2)
    o.xml = "".join(g)
    o.radius = math.hypot(W, Dp) / 2
    o.height = H + 0.12
    o.land = (x, y, z + 0.02, 0.7)
    o.door = {
        "centre": (cx, cy, z + dh / 2),
        "normal": (nx, ny),
        "width": dw,
        "height": dh,
        "open": door_open,
    }
    o.facts = {
        "kind": "hangar",
        "colour": colour,
        "size_m": W,
        "depth_m": Dp,
        "height_m": H,
        "door_open": door_open,
        "door_w": dw,
    }


def water_tower(o, rng, colour=None, height=None, style=None):
    colour = colour or rng.choice(["white", "blue", "red", "grey", "green"])
    H = height or round(rng.uniform(9.0, 16.0), 1)
    style = style or rng.choice(["legs", "column"])
    x, y, z, n = o.x, o.y, o.z, o.oid
    R = round(rng.uniform(1.6, 2.6), 1)
    g = []
    if style == "legs":
        for k in range(4):
            a = math.pi / 4 + k * math.pi / 2
            lx, ly = x + R * 0.8 * math.cos(a), y + R * 0.8 * math.sin(a)
            g.append(
                f'<geom name="{n}_l{k}" type="capsule" fromto="{lx:.2f} {ly:.2f} {z:.2f} {x + R * .5 * math.cos(a):.2f} {y + R * .5 * math.sin(a):.2f} {z + H - 2 * R:.2f}" size=".1" rgba="{rgba("grey")}"/>'
            )
        g.append(
            f'<geom name="{n}_t" type="sphere" pos="{x:.2f} {y:.2f} {z + H - R:.2f}" size="{R:.2f}" rgba="{rgba(colour)}"/>'
        )
    else:
        g.append(_cyl(f"{n}_c", x, y, z + (H - 2.4) / 2, 0.55, (H - 2.4) / 2, rgba("grey")))
        g.append(_cyl(f"{n}_t", x, y, z + H - 1.2, R, 1.2, rgba(colour)))
        g.append(
            f'<geom name="{n}_r" type="ellipsoid" pos="{x:.2f} {y:.2f} {z + H:.2f}" size="{R:.2f} {R:.2f} .5" rgba="{rgba(colour)}"/>'
        )
    o.xml = "".join(g)
    o.radius = R
    o.height = H
    o.facts = {"kind": "water_tower", "colour": colour, "height_m": H, "style": style, "size_m": 2 * R}


def mast(o, rng, colour=None, height=None):
    colour = colour or rng.choice(["red", "white", "grey"])
    H = height or round(rng.uniform(15.0, 30.0), 0)
    x, y, z, n = o.x, o.y, o.z, o.oid
    a = 0.7
    g = []
    for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        g.append(
            f'<geom name="{n}_l{k}" type="capsule" fromto="{x + a * sx:.2f} {y + a * sy:.2f} {z:.2f} {x + .25 * sx:.2f} {y + .25 * sy:.2f} {z + H:.2f}" size=".05" rgba="{rgba(colour if k % 2 else "white")}"/>'
        )
    for k, zz in enumerate(np.arange(2.0, H, 3.0)):
        w = a - (a - 0.25) * zz / H
        g.append(_box(f"{n}_x{k}", x, y, z + zz, w, w, 0.03, 0.0, rgba("grey")))
    for k, b in enumerate((0.0, 2.09, 4.19)):
        g.append(
            _box(
                f"{n}_a{k}",
                x + 0.45 * math.cos(b),
                y + 0.45 * math.sin(b),
                z + H - 1.2,
                0.1,
                0.25,
                0.7,
                b,
                rgba("white"),
            )
        )
    g.append(_cyl(f"{n}_d", x, y, z + H - 3.0, 0.35, 0.12, rgba("grey")))
    o.xml = "".join(g)
    o.radius = a + 0.3
    o.height = H
    o.facts = {"kind": "mast", "colour": colour, "height_m": H, "size_m": 2 * a}


def tent(o, rng, colour=None, size=None, style=None):
    colour = colour or rng.choice(["orange", "green", "blue", "red", "yellow"])
    L = size or round(rng.uniform(2.2, 4.0), 1)
    style = style or rng.choice(["ridge", "dome"])
    x, y, z, yaw, n = o.x, o.y, o.z, o.yaw, o.oid
    if style == "ridge":
        h = L * 0.4
        w = L * 0.55
        s = math.hypot(w / 2, h)
        ang = math.atan2(h, w / 2)  # two panels meeting at the ridge (an inverted V)
        g = [
            f'<geom name="{n}_{c}" type="box" pos="{_v(x, y, yaw, 0, side * w / 4)[0]:.3f} {_v(x, y, yaw, 0, side * w / 4)[1]:.3f} {z + h / 2:.3f}" size="{L / 2:.3f} {s / 2:.3f} .03" '
            f'quat="{quat_yaw_roll(yaw, -side * ang)}" rgba="{rgba(colour)}"/>'
            for c, side in (("a", -1), ("b", 1))
        ]
        g.append(
            f'<geom name="{n}_e" type="box" pos="{x:.3f} {y:.3f} {z + h + .02:.3f}" size="{L / 2 + .05:.3f} .04 .03" quat="{quat_yaw_roll(yaw, 0.)}" rgba="{rgba("grey")}"/>'
        )
    else:
        h = L * 0.38
        g = [
            f'<geom name="{n}_a" type="ellipsoid" pos="{x:.2f} {y:.2f} {z:.2f}" size="{L / 2:.2f} {L * .42:.2f} {h:.2f}" rgba="{rgba(colour)}"/>'
        ]
    o.xml = "".join(g)
    o.radius = L / 2
    o.height = L * 0.4
    o.facts = {"kind": "tent", "colour": colour, "size_m": L, "style": style}


def container(o, rng, colour=None, length=None):
    colour = colour or rng.choice(["red", "blue", "green", "orange", "grey", "white", "yellow", "brown"])
    L = length or rng.choice([6.1, 6.1, 12.2])
    x, y, z, yaw, n = o.x, o.y, o.z, o.yaw, o.oid
    g = [_box(f"{n}_b", x, y, z + 1.3, L / 2, 1.22, 1.3, yaw, rgba(colour))]
    for k, u in enumerate(np.arange(-L / 2 + 0.3, L / 2, 0.6)):  # corrugation ribs on both long sides
        for s in (-1, 1):
            px, py = _v(x, y, yaw, u, s * 1.24)
            g.append(_box(f"{n}_c{k}_{s}", px, py, z + 1.3, 0.05, 0.02, 1.25, yaw, rgba(colour)))
    o.xml = "".join(g)
    o.radius = math.hypot(L, 2.44) / 2
    o.height = 2.6
    o.land = (x, y, z + 2.6, 1.0)
    o.facts = {"kind": "container", "colour": colour, "size_m": L}


def solar(o, rng, rows=None):
    rows = rows or rng.choice([1, 2, 3])
    x, y, z, yaw, n = o.x, o.y, o.z, o.yaw, o.oid
    L = round(rng.uniform(4.0, 10.0), 1)
    g = []
    for r in range(rows):
        px, py = _v(x, y, yaw, 0, (r - (rows - 1) / 2) * 2.6)
        g.append(
            f'<geom name="{n}_p{r}" type="box" pos="{px:.2f} {py:.2f} {z + 1.:.2f}" size="{L / 2:.2f} .9 .03" quat="{quat_yaw_roll(yaw, math.radians(25))}" rgba=".10 .16 .32 1"/>'
        )
        for k, u in enumerate((-L / 2 + 0.3, L / 2 - 0.3)):
            qx, qy = _v(px, py, yaw, u, 0)
            g.append(_cyl(f"{n}_s{r}_{k}", qx, qy, z + 0.45, 0.04, 0.45, rgba("grey")))
    o.xml = "".join(g)
    o.radius = math.hypot(L, rows * 2.6) / 2
    o.height = 1.5
    o.facts = {"kind": "solar", "rows": rows, "size_m": L, "colour": "blue"}


def windsock(o, rng, wind_dir=0.0):
    x, y, z, n = o.x, o.y, o.z, o.oid
    H = round(rng.uniform(4.0, 6.0), 1)
    dx, dy = math.cos(wind_dir), math.sin(wind_dir)
    g = [_cyl(f"{n}_p", x, y, z + H / 2, 0.05, H / 2, rgba("white"))]
    for k in range(5):
        u = 0.25 + 0.3 * k
        g.append(
            f'<geom name="{n}_s{k}" type="cylinder" pos="{x + dx * u:.2f} {y + dy * u:.2f} {z + H - .05 * k:.2f}" size="{.22 - .025 * k:.3f} .15" '
            f'zaxis="{dx:.3f} {dy:.3f} {-.12:.3f}" rgba="{rgba("orange" if k % 2 == 0 else "white")}"/>'
        )
    o.xml = "".join(g)
    o.radius = 1.7
    o.height = H
    o.facts = {"kind": "windsock", "height_m": H, "colour": "orange"}


MAKERS = {
    "pad": pad,
    "dropzone": dropzone,
    "pickup": pickup,
    "hangar": hangar,
    "water_tower": water_tower,
    "mast": mast,
    "tent": tent,
    "container": container,
    "solar": solar,
    "windsock": windsock,
}


def make(oid, kind, x, y, yaw, z, rng, **attrs):
    o = Obj(oid, kind, float(x), float(y), float(yaw), float(z))
    MAKERS[kind](o, rng, **attrs)
    return o
