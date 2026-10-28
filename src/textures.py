"""Stage F: procedural textures for the realistic world.
usage: textures.py OUT_DIR  (writes PNGs; the builder calls ensure() which makes missing ones)"""

import math, sys, zlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TEX_DIR = Path(__file__).resolve().parents[1] / "assets" / "tex"
FLOORS, WINDOWS, FLOOR_H, BAY_W = 3, 3, 3.0, 3.2
TILE_H, TILE_W = FLOORS * FLOOR_H, WINDOWS * BAY_W
PX = 64  # texture pixels per metre on facades -> 614 x 576 px tiles


def rng(name):
    return np.random.default_rng(zlib.crc32(name.encode()))


def noise(r, h, w, scale, octaves=4):
    """Value noise in [0, 1], tileable (wraps), fractal sum of upsampled random grids."""
    out = np.zeros((h, w))
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        gh, gw = max(2, int(h / scale * 2**o)), max(2, int(w / scale * 2**o))
        g = r.random((gh, gw))
        g = np.concatenate([g, g[:1]], 0)
        g = np.concatenate([g, g[:, :1]], 1)
        im = (
            np.array(
                Image.fromarray((g * 255).astype(np.uint8)).resize((w + w // gw, h + h // gh), Image.BICUBIC),
                float,
            )[:h, :w]
            / 255
        )
        out += amp * im
        tot += amp
        amp *= 0.5
    return out / tot


def tint(base, var, n):
    """base RGB (0-255) modulated by a noise field n (0..1) with relative variation var."""
    f = 1 + var * (n - 0.5) * 2
    return np.clip(np.array(base, float)[None, None, :] * f[..., None], 0, 255)


def save(arr, name, out):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(out / f"{name}.png")
    return name


#
STYLES = {  # wall colour, window frame, glass, window w x h (m), sill, extras
    "panel_light": dict(
        wall=(206, 204, 196),
        frame=(235, 235, 232),
        glass=(70, 88, 105),
        win=(1.5, 1.45),
        seams=True,
        balcony=0.35,
    ),
    "panel_beige": dict(
        wall=(214, 198, 168),
        frame=(238, 236, 230),
        glass=(64, 80, 96),
        win=(1.5, 1.45),
        seams=True,
        balcony=0.3,
    ),
    "panel_blue": dict(
        wall=(170, 186, 200),
        frame=(236, 236, 236),
        glass=(58, 74, 92),
        win=(1.5, 1.45),
        seams=True,
        balcony=0.2,
    ),
    "brick_red": dict(
        wall=(150, 72, 52), frame=(226, 224, 216), glass=(60, 72, 84), win=(1.3, 1.5), brick=True
    ),
    "brick_silicate": dict(
        wall=(218, 212, 196), frame=(240, 240, 236), glass=(62, 76, 90), win=(1.3, 1.5), brick=True
    ),
    "plaster_yellow": dict(
        wall=(222, 190, 120), frame=(246, 242, 232), glass=(52, 62, 74), win=(1.2, 1.9), cornice=True
    ),
    "plaster_pink": dict(
        wall=(214, 160, 150), frame=(246, 242, 232), glass=(52, 62, 74), win=(1.2, 1.9), cornice=True
    ),
    "plaster_green": dict(
        wall=(150, 182, 168), frame=(246, 242, 232), glass=(52, 62, 74), win=(1.2, 1.9), cornice=True
    ),
    "plaster_white": dict(
        wall=(232, 228, 218), frame=(250, 248, 242), glass=(55, 66, 80), win=(1.2, 1.8), cornice=True
    ),
    "glass_blue": dict(
        wall=(70, 100, 130), frame=(150, 160, 170), glass=(80, 120, 150), win=(3.0, 2.6), curtain=True
    ),
    "glass_green": dict(
        wall=(70, 110, 105), frame=(150, 165, 160), glass=(85, 130, 125), win=(3.0, 2.6), curtain=True
    ),
    "house_siding": dict(
        wall=(196, 180, 150),
        frame=(245, 245, 240),
        glass=(58, 70, 82),
        win=(1.1, 1.3),
        siding=True,
        sparse=True,
    ),
    "house_wood": dict(
        wall=(122, 86, 56),
        frame=(232, 226, 210),
        glass=(50, 60, 70),
        win=(1.0, 1.2),
        boards=True,
        sparse=True,
    ),
    "house_plaster": dict(
        wall=(228, 220, 200), frame=(250, 250, 248), glass=(56, 68, 80), win=(1.2, 1.4), sparse=True
    ),
    "industrial": dict(
        wall=(160, 166, 170),
        frame=(120, 126, 130),
        glass=(80, 96, 110),
        win=(4.0, 1.0),
        corrugated=True,
        strip=True,
    ),
    "garage": dict(wall=(170, 120, 96), frame=(110, 110, 110), glass=(80, 80, 80), win=(0, 0), brick=True),
    "jp_tile": dict(
        wall=(220, 214, 200),
        frame=(200, 200, 196),
        glass=(62, 78, 94),
        win=(1.6, 1.2),
        balcony=0.6,
        tiles=True,
    ),
}


def facade(name, s, out):
    r = rng(name)
    W, H = int(TILE_W * PX), int(TILE_H * PX)
    a = tint(s["wall"], 0.06, noise(r, H, W, 160) * 0.6 + noise(r, H, W, 12) * 0.4)
    im = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(im)
    m = PX
    if s.get("brick"):  # brick courses: 0.25 x 0.075 m with mortar
        bw, bh = int(0.25 * m), max(3, int(0.075 * m))
        mort = tuple(int(c * 1.25) for c in s["wall"])
        for y in range(0, H, bh):
            d.line([(0, y), (W, y)], fill=mort, width=1)
            off = (y // bh % 2) * bw // 2
            for x in range(off, W, bw):
                d.line([(x, y), (x, y + bh)], fill=mort, width=1)
    if s.get("siding"):
        for y in range(0, H, int(0.18 * m)):
            d.line([(0, y), (W, y)], fill=tuple(int(c * 0.8) for c in s["wall"]), width=2)
    if s.get("boards"):
        for y in range(0, H, int(0.22 * m)):
            d.line([(0, y), (W, y)], fill=tuple(int(c * 0.7) for c in s["wall"]), width=3)
    if s.get("corrugated"):
        for x in range(0, W, int(0.15 * m)):
            d.line([(x, 0), (x, H)], fill=tuple(int(c * 0.82) for c in s["wall"]), width=3)
    if s.get("tiles"):
        for y in range(0, H, int(0.3 * m)):
            d.line([(0, y), (W, y)], fill=tuple(int(c * 0.9) for c in s["wall"]), width=1)
        for x in range(0, W, int(0.3 * m)):
            d.line([(x, 0), (x, H)], fill=tuple(int(c * 0.9) for c in s["wall"]), width=1)
    if s.get("seams"):  # panel joints: every floor and every bay
        for f in range(FLOORS + 1):
            d.line(
                [(0, f * FLOOR_H * m), (W, f * FLOOR_H * m)],
                fill=tuple(int(c * 0.72) for c in s["wall"]),
                width=3,
            )
        for b in range(WINDOWS + 1):
            d.line(
                [(b * BAY_W * m, 0), (b * BAY_W * m, H)],
                fill=tuple(int(c * 0.78) for c in s["wall"]),
                width=2,
            )
    ww, wh = s["win"]
    for f in range(FLOORS):
        for b in range(WINDOWS):
            if ww == 0 or (s.get("sparse") and (b == 1 and f % 2 == 0)):
                continue
            cx = (b + 0.5) * BAY_W * m
            top = (f * FLOOR_H + (FLOOR_H - wh) * 0.55) * m
            if s.get("curtain"):
                x0, x1, y0, y1 = (
                    b * BAY_W * m + 3,
                    (b + 1) * BAY_W * m - 3,
                    f * FLOOR_H * m + 4,
                    (f + 1) * FLOOR_H * m - 4,
                )
            else:
                x0, x1, y0, y1 = cx - ww * m / 2, cx + ww * m / 2, top, top + wh * m
            if s.get("strip"):
                x0, x1 = b * BAY_W * m + 6, (b + 1) * BAY_W * m - 6
            d.rectangle([x0 - 5, y0 - 5, x1 + 5, y1 + 5], fill=s["frame"])
            sky = r.uniform(0.75, 1.35)
            gl = tuple(int(np.clip(c * sky, 0, 255)) for c in s["glass"])
            d.rectangle([x0, y0, x1, y1], fill=gl)
            gy = int((y1 - y0) * 0.25)
            d.rectangle(
                [x0, y0, x1, y0 + gy], fill=tuple(min(255, int(c * 1.35)) for c in gl)
            )  # sky reflection
            d.line([((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1)], fill=s["frame"], width=3)
            if not s.get("curtain"):
                d.line([(x0, y0 + (y1 - y0) * 0.3), (x1, y0 + (y1 - y0) * 0.3)], fill=s["frame"], width=3)
            if r.random() < 0.5 and not s.get("curtain"):  # curtains behind some windows
                d.rectangle(
                    [x0 + 3, y0 + (y1 - y0) * 0.32, x0 + (x1 - x0) * r.uniform(0.15, 0.4), y1 - 3],
                    fill=tuple(int(c) for c in r.integers(150, 235, 3)),
                )
            d.rectangle(
                [x0 - 8, y1 + 5, x1 + 8, y1 + 11], fill=tuple(int(c * 0.85) for c in s["wall"])
            )  # sill
            if s.get("cornice"):
                d.rectangle(
                    [x0 - 10, y0 - 20, x1 + 10, y0 - 9],
                    fill=tuple(min(255, int(c * 1.08)) for c in s["wall"]),
                )
            if s.get("balcony") and r.random() < s["balcony"]:
                d.rectangle(
                    [x0 - 22, y1 - 0.1 * m, x1 + 22, y1 + 0.95 * m - 0.1 * m],
                    fill=tuple(int(c * 0.9) for c in s["frame"]),
                )
                for k in range(int(x0 - 18), int(x1 + 20), 9):
                    d.line(
                        [(k, y1 - 0.1 * m), (k, y1 + 0.85 * m)],
                        fill=tuple(int(c * 0.65) for c in s["frame"]),
                        width=2,
                    )
    if s.get("cornice"):
        d.rectangle([0, 0, W, 14], fill=tuple(min(255, int(c * 1.1)) for c in s["wall"]))
    a = np.array(im, float) * (1 - 0.1 * noise(r, H, W, 40)[..., None])  # weathering / dirt streaks
    streak = np.repeat(noise(r, H, W // 8 + 1, 30), 8, 1)[:, :W]
    a *= (1 - 0.06 * streak)[..., None]
    return save(a, f"fac_{name}", out)


def groundfloor(name, wall, shop, out):
    """3.5 m tall ground-floor band, one bay wide."""
    r = rng(name)
    W, H = int(BAY_W * PX * 2), int(3.5 * PX)
    a = tint(wall, 0.06, noise(r, H, W, 20))
    im = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(im)
    m = PX
    if shop:
        d.rectangle([0.3 * m, 0.5 * m, 4.0 * m, 3.0 * m], fill=(40, 46, 52))
        d.rectangle([0.35 * m, 0.55 * m, 3.95 * m, 1.2 * m], fill=(110, 130, 150))
        d.rectangle([4.4 * m, 0.3 * m, 5.6 * m, 3.0 * m], fill=(60, 50, 40))
        d.rectangle([0.2 * m, 3.05 * m, 6.2 * m, 3.4 * m], fill=tuple(int(c) for c in r.integers(40, 220, 3)))
    else:
        d.rectangle([0, 0, W, 0.9 * m], fill=tuple(int(c * 0.6) for c in wall))
        d.rectangle([1.6 * m, 1.4 * m, 2.6 * m, 2.6 * m], fill=(60, 70, 80))
        d.rectangle([4.2 * m, 0.9 * m, 5.4 * m, 3.3 * m], fill=(90, 70, 50))
    return save(np.array(im), f"gf_{name}", out)


#
def surface(name, out):
    r = rng(name)
    S = 512
    n1, n2, n3 = noise(r, S, S, 64), noise(r, S, S, 8), noise(r, S, S, 180)
    if name == "grass":
        a = tint((84, 122, 58), 0.22, 0.5 * n1 + 0.3 * n2 + 0.2 * n3)
        a[..., 0] += 18 * (n3 > 0.62)  # dry patches
    elif name == "grass_dry":
        a = tint((128, 128, 78), 0.2, 0.5 * n1 + 0.5 * n2)
    elif name == "forest_floor":
        a = tint((78, 70, 46), 0.3, 0.6 * n1 + 0.4 * n2)
        a[..., 1] += 20 * (n2 > 0.6)
    elif name == "dirt":
        a = tint((128, 108, 82), 0.2, 0.6 * n1 + 0.4 * n2)
    elif name == "asphalt":
        a = tint((72, 72, 74), 0.12, 0.3 * n1 + 0.7 * n2)
        a += 25 * (r.random((S, S)) > 0.985)[..., None]
    elif name == "paving":
        a = tint((150, 146, 140), 0.1, 0.5 * n1 + 0.5 * n2)
        im = Image.fromarray(a.astype(np.uint8))
        d = ImageDraw.Draw(im)
        for k in range(0, S, 32):
            d.line([(0, k), (S, k)], fill=(110, 108, 104))
            d.line([(k, 0), (k, S)], fill=(110, 108, 104))
        a = np.array(im)
    elif name == "parking":
        a = tint((80, 80, 82), 0.1, 0.3 * n1 + 0.7 * n2)
        im = Image.fromarray(a.astype(np.uint8))
        d = ImageDraw.Draw(im)
        for k in range(0, S, 64):
            d.line([(k, 0), (k, S * 0.45)], fill=(220, 220, 214), width=4)
            d.line([(k, S * 0.55), (k, S)], fill=(220, 220, 214), width=4)
        a = np.array(im)
    elif name == "water":
        a = tint((52, 84, 104), 0.15, 0.5 * n1 + 0.5 * noise(r, S, S, 16))
        a[..., 2] += 12 * n3
    elif name == "sand":
        a = tint((206, 186, 140), 0.1, 0.5 * n1 + 0.5 * n2)
    elif name == "pitch":  # artificial turf with mowing stripes
        a = tint((60, 128, 66), 0.08, n2)
        a[:, (np.arange(S) // 64) % 2 == 0] *= 1.08
    elif name == "track":
        a = tint((166, 70, 56), 0.1, n2)
    elif name == "roof_gravel":
        a = tint((128, 126, 120), 0.18, 0.4 * n1 + 0.6 * noise(r, S, S, 4))
    elif name == "roof_bitumen":
        a = tint((70, 70, 72), 0.12, 0.5 * n1 + 0.5 * n2)
    elif name.startswith("roof_metal"):
        base = {
            "roof_metal_green": (66, 110, 80),
            "roof_metal_red": (132, 56, 46),
            "roof_metal_brown": (96, 70, 56),
            "roof_metal_grey": (130, 136, 140),
        }[name]
        a = tint(base, 0.08, n1)
        a[:, (np.arange(S) % 32) < 4] *= 0.78
    elif name == "roof_tiles":
        a = tint((150, 70, 50), 0.1, n1)
        im = Image.fromarray(a.astype(np.uint8))
        d = ImageDraw.Draw(im)
        for y in range(0, S, 24):
            d.line([(0, y), (S, y)], fill=(100, 44, 32), width=3)
            for x in range((y // 24 % 2) * 16, S, 32):
                d.arc([x - 16, y - 12, x + 16, y + 12], 0, 180, fill=(120, 54, 40), width=2)
        a = np.array(im)
    elif name == "foliage":  # leaf clusters: dark gaps + light leaves
        a = tint((58, 96, 44), 0.35, 0.45 * noise(r, S, S, 6) + 0.35 * n2 + 0.2 * n1)
        a *= (0.7 + 0.3 * (noise(r, S, S, 3) > 0.45))[..., None]
    elif name == "needles":
        a = tint((40, 70, 46), 0.3, 0.5 * noise(r, S, S, 5) + 0.5 * n2)
    elif name == "bark":
        a = tint((88, 66, 48), 0.25, noise(r, S, S, 6))
        a *= (1 - 0.25 * (np.sin(np.arange(S) / 3.0)[None, :] > 0.7))[..., None]
    elif name == "bark_birch":
        a = tint((228, 226, 218), 0.05, n2)
        m = noise(r, S, S, 24) > 0.7
        a[m] = (40, 40, 38)
    elif name == "concrete":
        a = tint((166, 164, 158), 0.1, 0.5 * n1 + 0.5 * n2)
    elif name == "metal":
        a = tint((150, 156, 160), 0.06, n1)
    else:
        raise KeyError(name)
    return save(a, f"srf_{name}", out)


SURFACES = (
    "grass",
    "grass_dry",
    "forest_floor",
    "dirt",
    "asphalt",
    "paving",
    "parking",
    "water",
    "sand",
    "pitch",
    "track",
    "roof_gravel",
    "roof_bitumen",
    "roof_metal_green",
    "roof_metal_red",
    "roof_metal_brown",
    "roof_metal_grey",
    "roof_tiles",
    "foliage",
    "needles",
    "bark",
    "bark_birch",
    "concrete",
    "metal",
)


def skybox(out, name="sky"):
    """Cube map as a 6-face strip."""
    r = rng(name)
    S = 512
    faces = {}
    y = np.linspace(0, 1, S)[:, None]
    top, hor = np.array([112, 152, 210]), np.array([214, 226, 238])
    grad = top * (1 - y[..., None] ** 0.8) + hor * (y[..., None] ** 0.8)
    x = np.linspace(0, 1, S)[None, :]
    edge = np.clip(np.minimum(x, 1 - x) / 0.18, 0, 1) ** 2
    for f in "LFRB":
        cl = noise(r, S, S, 90, 5)
        cl = (
            np.clip((cl - 0.52) * 3.5, 0, 1)
            * np.clip((y - 0.18) / 0.25, 0, 1)
            * np.clip((0.78 - y) / 0.2, 0, 1)
            * edge
        )
        faces[f] = (
            grad * np.broadcast_to(np.ones((S, S, 1)), (S, S, 3)) * (1 - cl[..., None]) + 246 * cl[..., None]
        )
    faces["U"] = np.broadcast_to(top, (S, S, 3)).astype(float)
    faces["D"] = np.broadcast_to(hor, (S, S, 3)).astype(float)
    grid = np.zeros((3 * S, 4 * S, 3))
    pos = {"U": (0, 1), "L": (1, 0), "F": (1, 1), "R": (1, 2), "B": (1, 3), "D": (2, 1)}
    for f, (i, j) in pos.items():
        grid[i * S : (i + 1) * S, j * S : (j + 1) * S] = faces[f]
    return save(grid, name, out)


def ensure(out=TEX_DIR):
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for n, s in STYLES.items():
        if not (out / f"fac_{n}.png").exists():
            made.append(facade(n, s, out))
        if not (out / f"gf_{n}.png").exists():
            made.append(groundfloor(n, s["wall"], n.startswith(("plaster", "glass", "brick_red")), out))
    for n in SURFACES:
        if not (out / f"srf_{n}.png").exists():
            made.append(surface(n, out))
    if not (out / "sky.png").exists():
        made.append(skybox(out))
    return made


if __name__ == "__main__":
    print(ensure(Path(sys.argv[1]) if len(sys.argv) > 1 else TEX_DIR))
