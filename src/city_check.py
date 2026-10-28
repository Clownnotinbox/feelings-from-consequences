"""Visual check of the realistic world."""

import sys, time
import numpy as np
import mujoco
from PIL import Image, ImageDraw
import city3d as C


def views(W):
    """(name, lookat, distance, azimuth, elevation), around free ground near the centre and a few."""
    fp = sorted(
        W.footprints,
        key=lambda f: f["poly"].centroid.distance(
            __import__("shapely.geometry", fromlist=["Point"]).Point(0, 0)
        ),
    )
    c = fp[0]["poly"].centroid if fp else None
    cx, cy = (c.x, c.y) if c else (0.0, 0.0)
    return [
        ("drone 40 m", (cx, cy, 8), 90, 35, -22),
        ("street", (cx + 25, cy + 25, 2), 35, 210, -6),
        ("oblique 120 m", (0, 0, 0), 260, 60, -32),
        ("top-down", (0, 0, 0), 330, 90, -89),
        ("drone 25 m low", (cx - 40, cy - 30, 6), 55, 140, -14),
        ("far", (0, 0, 10), 420, 300, -12),
    ]


def main():
    place, out = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    t0 = time.time()
    W, static, assets = C.build(place, seed)
    t1 = time.time()
    xml = C.to_mjcf(W, static, assets)
    files = C.texture_files()
    m = mujoco.MjModel.from_xml_string(xml, files)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    t2 = time.time()
    print(
        place,
        W.stats,
        f"build {t1 - t0:.1f}s compile {t2 - t1:.1f}s ngeom {m.ngeom} nmesh {m.nmesh} nvert {m.nmeshvert}",
        flush=True,
    )
    R = mujoco.Renderer(m, 480, 640, max_geom=30000)
    tiles = []
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    for name, la, dist, az, el in views(W):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = la
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        d.qpos[:3] = [la[0], la[1], la[2] + 30]
        mujoco.mj_forward(m, d)  # the sun's shadow box follows the drone
        R.update_scene(d, camera=cam, scene_option=opt)
        R.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        img = R.render().copy()
        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((6, 6), name, fill=(255, 255, 255))
        tiles.append(im)
    R.close()
    sheet = Image.new("RGB", (640 * 3, 480 * 2))
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % 3) * 640, (k // 3) * 480))
    sheet.save(out)
    print("saved", out, f"{time.time() - t2:.1f}s render")


if __name__ == "__main__":
    main()
