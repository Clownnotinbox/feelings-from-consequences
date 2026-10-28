"""Environment checks."""

import json, math, sys, time
from pathlib import Path
import numpy as np
import drone_env

OUT = Path(__file__).resolve().parents[1] / "runs"
OUT.mkdir(exist_ok=True)


class ScriptedPilot:
    """Climb to cruise altitude, fly towards the goal estimate, hover above it."""

    def __init__(self, goal_xy, cruise=20.0):
        self.goal = np.array(goal_xy)
        self.cruise = cruise
        self.p = None

    def act(self, o, dt):
        if o["gps"] is not None and o["gps"].get("hdop", 9) < 5:
            self.p = np.array([o["gps"]["x"], o["gps"]["y"]])
        elif self.p is not None:
            self.p = self.p + np.array(o["vel_est"][:2]) * dt
        alt = o["baro_alt"]
        vz = float(np.clip(0.8 * (self.cruise - alt), -2, 3))
        if self.p is None or alt < self.cruise - 3:
            return {"vx": 0.0, "vy": 0.0, "vz": vz}
        e = self.goal - self.p
        dist = float(np.linalg.norm(e))
        v = e / max(dist, 1e-6) * min(8.0, 0.6 * dist)
        return {"vx": float(v[0]), "vy": float(v[1]), "vz": vz}


def fly(seconds=60, **kw):
    env = drone_env.DroneEnv(render=False, **kw)
    o = env.reset()
    pilot = ScriptedPilot(env.goal.pos)
    t0 = time.time()
    for _ in range(int(seconds / env.agent_dt)):
        o = env.step(pilot.act(o, env.agent_dt))
    tr = env.truth()
    tr.update(
        wall_s=round(time.time() - t0, 1),
        sim_s=seconds,
        vel_source=o["vel_source"],
        goal=env.goal.description,
    )
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in tr.items()}


def frames(seed=1):
    import imageio.v2 as imageio

    env = drone_env.DroneEnv(seed=seed, render=True, cam=(320, 240))
    o = env.reset()
    pilot = ScriptedPilot(env.goal.pos)
    shots = []
    for i in range(int(40 / env.agent_dt)):
        o = env.step(pilot.act(o, env.agent_dt))
        if i % 10 == 0:
            env.renderer.update_scene(env.data, camera="chase", scene_option=env.vopt)
            shots.append(np.concatenate([env.renderer.render(), o["cam_front"], o["cam_down"]], 1))
    imageio.mimsave(OUT / "flight.gif", shots[::2], duration=0.25)
    imageio.imwrite(OUT / "frame_last.png", shots[-1])
    return len(shots)


if __name__ == "__main__":
    R = {}
    tests = {
        "gps_ok": dict(seed=1),
        "gps_lost_20s_flow": dict(seed=1, gps_faults=[(20, 999, "off")], flow=True),
        "gps_lost_20s_no_flow": dict(seed=1, gps_faults=[(20, 999, "off")], flow=False),
        "gps_spoofed_20s": dict(seed=1, gps_faults=[(20, 999, "drift")]),
        "wind_6_gust_2": dict(seed=1, wind=(6.0, 0.0), gust=2.0),
        "wind_6_gps_lost_flow": dict(
            seed=1, wind=(6.0, 0.0), gust=2.0, gps_faults=[(20, 999, "off")], flow=True
        ),
    }
    for name, kw in tests.items():
        R[name] = fly(70, **kw)
        print(name, R[name], flush=True)
    R["frames"] = frames()
    print("frames", R["frames"])
    (OUT / "ENV_TESTS.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
