"""Stage F quality check of mission objects and birds."""

import math, sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw
import city3d as C, mission_objects as MO, actors as AC


def main():
    out = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = np.random.default_rng(seed)
    kinds = [k for k in MO.KINDS] + ["hangar_closed"]
    objs = []
    shots = []
    for i, kind in enumerate(kinds):
        for v in range(3):
            x, y = i * 40.0, v * 40.0
            kk = "hangar" if kind == "hangar_closed" else kind
            kw = (
                {"door_open": False}
                if kind == "hangar_closed"
                else ({"door_open": True} if kind == "hangar" else {})
            )
            o = MO.make(
                f"g{i}_{v}",
                kk,
                x,
                y,
                float(rng.uniform(0, 6.28)) if kk != "hangar" else -math.pi / 2 + 0.3,
                0.0,
                rng,
                **kw,
            )
            objs.append(o)
            az = (
                210
                if kk != "hangar"
                else math.degrees(math.atan2(o.door["normal"][1], o.door["normal"][0])) + 180 + 20
            )
            dist = max(9.0, o.radius * 3.2 + o.height * 1.2)
            el = -32 if kk not in ("mast", "water_tower") else -12
            label = f"{kind} " + " ".join(f"{k}={v_}" for k, v_ in o.facts.items() if k != "kind")[:70]
            shots.append((label, (x, y, min(o.height / 2, 6)), dist, az, el))
    W = type("W", (), {})()
    W.movers = []
    for k, sp in enumerate(AC.SPECIES):  # birds frozen mid-flap, close up
        W.movers.append(
            {
                "kind": "bird",
                "v2": True,
                "species": sp,
                "behaviour": "cross",
                "p0": (-60.0 - 8 * k, -40.0),
                "dir": (1.0, 0.0),
                "alt": 3.0,
                "length": 1e9,
                "speed": 0.0,
                "phase": 0.0,
                "seed": k,
            }
        )
    birds = AC.actors_xml(W.movers)
    xml = f"""<mujoco><statistic extent="100" center="0 0 0"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="4096" offsamples="4"/><headlight ambient=".42 .43 .46" diffuse=".12 .12 .12" specular="0 0 0"/>
  <map znear=".0005" zfar="15" shadowclip="1.5"/></visual>
  <asset><texture name="g" type="2d" file="srf_grass.png"/><material name="ground" texture="g" texrepeat=".1 .1" texuniform="true"/>{''.join(C.asset_xml())}</asset>
  <worldbody><light pos="150 40 120" dir="-.4 -.2 -1" directional="true" castshadow="true" diffuse=".75 .73 .68"/>
  <geom name="ground" type="plane" size="400 400 .1" material="ground"/>{''.join(o.xml for o in objs)}{birds}</worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml, C.texture_files())
    d = mujoco.MjData(m)
    for i, mv in enumerate(W.movers):
        P = AC.pose(mv, 0.03 * (i + 1), None, {})
        mid = m.body(f"mover{i}").mocapid[0]
        d.mocap_pos[mid], d.mocap_quat[mid] = P[""]
        for s in "lr":
            w = m.body(f"mover{i}_w{s}").mocapid[0]
            d.mocap_pos[w], d.mocap_quat[w] = P["w" + s]
        shots.append(
            (
                f'bird {mv["species"]} (span {AC.SPECIES[mv["species"]]["span"]} m)',
                tuple(P[""][0]),
                1.6 + AC.SPECIES[mv["species"]]["span"],
                70,
                -20,
            )
        )
    mujoco.mj_forward(m, d)
    R = mujoco.Renderer(m, 360, 480, max_geom=30000)
    tiles = []
    for label, la, dist, az, el in shots:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = la
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        R.update_scene(d, camera=cam)
        im = Image.fromarray(R.render().copy())
        ImageDraw.Draw(im).rectangle([0, 0, 480, 14], fill=(0, 0, 0))
        ImageDraw.Draw(im).text((4, 2), label, fill=(255, 255, 255))
        tiles.append(im)
    R.close()
    cols = 6
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (480 * cols, 360 * rows))
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * 480, (k // cols) * 360))
    sheet.save(out)
    print("saved", out, len(tiles))


if __name__ == "__main__":
    main()
