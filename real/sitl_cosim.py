"""Check G5 (docs/REAL_DRONE.md sec.
usage: sitl_cosim.py OUT.jsonl K0 K1 INSTANCE   (missions k = K0..K1-1 of the training places; PX4 instance I listens on 14540+I)
"""

import json, math, os, sys, time, zlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("BRAIN_SRC", str(HERE.parent / "src")))
import mav as MV

PLACES = (
    "berlin_marzahn",
    "spb_center",
    "malakhovka",
    "helsinki_kallio",
    "warsaw_ursynow",
    "akadem_line",
    "ekb_center",
    "msk_strogino",
    "plano_suburb",
    "kratovo",
)
ACTS = ("land", "flyby", "drop", "inspect", "pickup")


class CosimEnv:
    """Brain-facing env (as real_env.RealEnv) whose flight is PX4's and whose senses are the MuJoCo."""

    agent_dt = 0.1

    def __init__(self, link, menv):
        self.link, self.menv = link, menv
        self.home, self.fovy, self.cargo, self.data = menv.home, menv.fovy, menv.cargo, menv.data
        self._t0 = None
        self._next = None
        self.z0 = 0.0
        self.arm_t = []
        self.track_err = []

    @property
    def t(self):
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

    def _px4(self):
        lp, at, imu = (
            self.link.get("LOCAL_POSITION_NED"),
            self.link.get("ATTITUDE"),
            self.link.get("HIGHRES_IMU"),
        )
        x, y, z = MV.enu_from_ned(lp.x, lp.y, lp.z)
        v = np.array(MV.enu_from_ned(lp.vx, lp.vy, lp.vz))
        att = MV.att_ours_from_px4(at.roll, at.pitch, at.yaw)
        w = np.array(MV.flu_from_frd(at.rollspeed, at.pitchspeed, at.yawspeed))
        acc = MV.flu_from_frd(imu.xacc, imu.yacc, imu.zacc) if imu else (0.0, 0.0, 9.81)
        return np.array([self.home[0] + x, self.home[1] + y, self.z0 + z]), v, att, w, acc

    def _pose(self):
        import mujoco, nav

        p, v, att, w, acc = self._px4()
        d = self.menv.data
        R = nav.rot(*att)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.ravel())
        d.qpos[:3] = p
        d.qpos[3:7] = q
        d.qvel[:3] = v
        d.qvel[3:6] = w
        mujoco.mj_forward(self.menv.model, d)
        self.menv._lidar_cache = None
        return p, v, att, w, acc

    def reset(self):
        o = self.menv.reset()
        self.z0 = float(self.menv.data.qpos[2])
        for _ in range(600):
            if (
                self.link.get("LOCAL_POSITION_NED", 1.0) is not None
                and self.link.get("ATTITUDE", 1.0) is not None
            ):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("no PX4 position")
        self._t0 = time.monotonic()
        self._next = self._t0
        self._pose()
        return self.observe()

    def step(self, cmd):
        L = self.link
        hold = None
        if (
            cmd.get("precise") and "dxy" in cmd and os.environ.get("PX4_HOLD_XY", "1") == "1"
        ):  # precise phases: a position target for PX4's own loop
            lp = L.get("LOCAL_POSITION_NED")
            x, y, _ = MV.enu_from_ned(lp.x, lp.y, lp.z)
            hold = (x + cmd["dxy"][0], y + cmd["dxy"][1])
        L.set_velocity_enu(cmd["vx"], cmd["vy"], cmd["vz"], cmd.get("yaw_rate", 0.0), hold_xy=hold)
        if cmd["vz"] > 0.2 and not L.armed():
            L.offboard()
            L.arm(True)
            self.arm_t.append(round(self.t, 1))
        self._next += self.agent_dt
        time.sleep(max(0.0, self._next - time.monotonic()))
        if time.monotonic() - self._next > 1.0:
            self._next = time.monotonic()
        p, v, att, w, acc = self._pose()
        m = self.menv
        dt = self.agent_dt
        if math.hypot(cmd["vx"], cmd["vy"]) > 0.3:
            self.track_err.append(
                float(np.linalg.norm(np.array([cmd["vx"], cmd["vy"]]) - v[:2]))
            )  # how well PX4 follows the brain
        m.t = self.t
        m._move(dt)
        m._contacts()  # parcel fall, hook pendulum, contacts of the posed drone
        o = self.observe()
        if L.landed() or (0 <= o["range_down"] < 0.2 and float(np.linalg.norm(v)) < 0.4):
            self._touch()  # on the surface: PX4's land detector, or < 20 cm above it and nearly
        return o

    def _touch(self):
        """PX4's land detector says ON GROUND."""
        import mujoco

        m, d = self.menv.model, self.menv.data
        g = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(m, d, d.qpos[:3].copy(), np.array([0.0, 0.0, -1.0]), None, 1, self.menv.bid, g)
        if dist < 0 or dist > 1.5:
            return
        T = self.menv.touchdowns
        if not T or self.t - T[-1][0] > 1.0:
            T.append(
                (
                    round(self.t, 2),
                    int(g[0]),
                    round(float(np.linalg.norm(d.qvel[:3])), 2),
                    *[round(float(v), 2) for v in d.qpos[:3]],
                )
            )

    def observe(self, images=None, lidar=True):
        p, v, att, w, acc = self._px4()
        o = self.menv.observe(
            images=False, lidar=lidar
        )  # lidar, range finder from the MuJoCo world at PX4's pose
        gp, gr = self.link.get("GLOBAL_POSITION_INT"), self.link.get("GPS_RAW_INT")
        bs = self.link.get("BATTERY_STATUS")
        o.update(
            t=round(self.t, 3),
            attitude={"roll": att[0], "pitch": att[1], "yaw": att[2]},
            vel_est=v.tolist(),
            vel_source="gps",
            imu={"acc": list(acc), "gyro": w.tolist()},
            baro_alt=float(p[2]),
            battery_pct=(
                float(bs.battery_remaining) if bs is not None and bs.battery_remaining >= 0 else 100.0
            ),
            gps={
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "hdop": (gr.eph / 100.0 if gr is not None and gr.eph != 65535 else 0.8),
                "sats": gr.satellites_visible if gr is not None else 12,
            },
        )
        return o

    def obs_yaw(self):
        return self._px4()[2][2]

    def set_gimbal(self, yaw_rel_deg, pitch_deg):
        self.menv.set_gimbal(yaw_rel_deg, pitch_deg)

    def gimbal_frame(self, depth=True):
        return self.menv.gimbal_frame(depth)

    def link_up(self):
        return True

    def winch(self, length):
        self.menv.winch(length)

    def release_cargo(self):
        return self.menv.release_cargo()

    def hook_load(self):
        return self.menv.hook_load()


def mission(k):
    import city3d as C, missions as M

    act = ACTS[k % len(ACTS)]
    place = PLACES[(k * 3 + 1) % len(PLACES)]
    seed = 9000 + k
    rng = np.random.default_rng(zlib.crc32(f"sitl/{k}".encode()))
    for _ in range(20):  # PX4's SIH has flat ground only: ground-level targets
        W0, _, _ = C.build(place, seed, actors=False)
        mis = M.make(W0, rng, action=act)
        if mis is not None and all(
            next(o for o in mis["objects"] if o.oid == s["target"]).z < 0.5 for s in mis["steps"]
        ):
            return place, seed, mis
    return place, seed, None


def fly(k, instance, out, own_autopilot=False):
    import mission_env as ME, mission_brain as MB, nav as NV, safety as SF, perceive as PC, mission_score as MS

    place, seed, mis = mission(k)
    if mis is None:
        return {"k": k, "error": "no ground-level mission"}
    menv = ME.MissionEnv(
        place, seed, mis, render=False, cam=(160, 120), gimbal=(960, 720), wind=(0.0, 0.0), gust=0.0
    )
    models = HERE.parent / "models"
    perc = PC.Perceiver(str(models / "detector.pt"), str(models / "attrs_best.pt"))
    link = None
    if own_autopilot:
        env = menv
    else:
        link = MV.Link(f"udpin:127.0.0.1:{14540 + instance}")
        configure_px4(link)
        env = CosimEnv(link, menv)
    notes = []
    saf = SF.Safety(menv)
    nav = NV.Navigator(menv.W.map, menv.home, seed=seed)
    brain = MB.Brain(env, menv.W.map, mis["truth"], mis["rough"], perc, safety=saf, nav=nav, log=notes.append)
    o = env.reset()
    track = []
    t0 = time.time()
    while env.t < 420 and not brain.done:
        o = env.step(brain.act(o))
        p = menv.data.qpos[:3]
        track.append((env.t, *map(float, p)))
    stop = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
    t_end = env.t + 20.0
    while menv.cargo["state"] not in ("stowed", "landed") and env.t < t_end:
        o = env.step(stop)
    res = MS.score(menv, mis, track)
    r = {
        "k": k,
        "autopilot": "own" if own_autopilot else "px4",
        "place": place,
        "action": mis["steps"][0]["action"],
        "steps": res["steps"],
        "wrong": res["wrong_target"],
        "hits": [h["kind"] for h in menv.hit_events],
        "state": brain.state,
        "t": round(env.t, 1),
        "wall": round(time.time() - t0, 1),
        "notes": [n for n in notes if "] -> " not in n][-25:],
    }
    if link is not None:
        r.update(
            arm_times=env.arm_t,
            vel_track_err_mean=round(float(np.mean(env.track_err)), 3) if env.track_err else None,
            vel_track_err_median=round(float(np.median(env.track_err)), 3) if env.track_err else None,
            watchdog_hover_ticks=link.watchdog_hits,
            px4_msgs=link.texts[-25:],
        )
        if link.landed():
            link.arm(False)
        link.close()
    menv.close()
    return r


def configure_px4(link):
    """SITL: no RC and no ground station here, the brain is the only pilot."""
    import struct

    for name, val, kind in (
        ("COM_RC_IN_MODE", 4, "i"),
        ("NAV_RCL_ACT", 0, "i"),
        ("NAV_DLL_ACT", 0, "i"),
        ("COM_OF_LOSS_T", 1.0, "f"),
        ("COM_OBL_RC_ACT", 5, "i"),
        ("COM_DISARM_LAND", 2.0, "f"),
        ("MPC_XY_VEL_MAX", 10.0, "f"),
        ("COM_ARM_WO_GPS", 0, "i"),
        ("SIM_BAT_MIN_PCT", 95.0, "f"),
        ("COM_LOW_BAT_ACT", 0, "i"),
    ):  # SITL's simulated battery would drain to 50 % and trigger PX4's
        v = struct.unpack("<f", struct.pack("<i", val))[0] if kind == "i" else float(val)
        link.m.mav.param_set_send(
            link.sys, link.comp, name.encode(), v, 6 if kind == "i" else 9
        )  # MAV_PARAM_TYPE_INT32 / REAL32
        time.sleep(0.05)
    time.sleep(1.0)


if __name__ == "__main__":
    out, k0, k1, inst = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    own = len(sys.argv) > 5 and sys.argv[5] == "own"
    for k in range(k0, k1):
        try:
            r = fly(k, inst, out, own_autopilot=own)
        except Exception:
            import traceback

            r = {"k": k, "autopilot": "own" if own else "px4", "error": traceback.format_exc()[-1500:]}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(
            json.dumps(
                {
                    kk: r.get(kk)
                    for kk in (
                        "k",
                        "autopilot",
                        "action",
                        "steps",
                        "state",
                        "t",
                        "vel_track_err_mean",
                        "error",
                    )
                },
                default=str,
            )[:600],
            flush=True,
        )
