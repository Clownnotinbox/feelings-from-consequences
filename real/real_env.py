"""The mission brain's 'env' on a real drone."""

import math, time
import numpy as np
import mav as MV

M_DEG = 111_320.0  # metres per degree of latitude (city3d.Place uses the same)


class LocalFrame:
    """Map frame of the drone's OSM map: ENU metres around a reference lat/lon (the map's origin)."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0

    def xy(self, lat, lon):
        return (
            (lon - self.lon0) * M_DEG * math.cos(math.radians(self.lat0)),
            (lat - self.lat0) * M_DEG,
        )  # the same projection as city3d.Place

    def latlon(self, x, y):
        return (self.lat0 + y / M_DEG, self.lon0 + x / (M_DEG * math.cos(math.radians(self.lat0))))


class _Data:
    def __init__(self):
        self.qpos = np.zeros(7)


class RealEnv:
    agent_dt = 0.1

    def __init__(self, link, lidar, camera, frame, payload_kg=0.5, fovy=70.0, ground_alt_amsl=None):
        self.link, self.lidar, self.camera, self.frame, self.fovy = link, lidar, camera, frame, fovy
        self.cargo = {"mass": float(payload_kg), "state": "stowed"}
        self.data = _Data()
        self.gimbal_yaw, self.gimbal_pitch = 0.0, -30.0
        self._t0 = None
        self._next = None
        self._L = 0.0
        self.home = (0.0, 0.0)
        self.ground_alt_amsl = ground_alt_amsl
        self._origin = np.zeros(3)

    # ---------------- time
    @property
    def t(self):
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

    # ---------------- lifecycle
    def reset(self):
        """Wait for a 3D GPS fix and the local position; home = the take-off point in the map frame."""
        for _ in range(600):
            g, lp = self.link.get("GLOBAL_POSITION_INT", 2.0), self.link.get("LOCAL_POSITION_NED", 1.0)
            if g is not None and lp is not None and g.lat != 0:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("no position from the autopilot")
        hx, hy = self.frame.xy(g.lat * 1e-7, g.lon * 1e-7)
        e, n, u = MV.enu_from_ned(lp.x, lp.y, lp.z)
        self._origin = np.array([hx - e, hy - n, -u])
        self.home = (hx, hy)  # map frame = autopilot local frame + this offset
        self._t0 = time.monotonic()
        self._next = self._t0
        return self.observe()

    def step(self, cmd):
        """Send the brain's command."""
        L = self.link
        hold = None
        if cmd.get("precise") and "dxy" in cmd:  # landing, hook, drop
            lp = L.get("LOCAL_POSITION_NED")
            x, y, _ = MV.enu_from_ned(lp.x, lp.y, lp.z)
            hold = (x + cmd["dxy"][0], y + cmd["dxy"][1])
        L.set_velocity_enu(cmd["vx"], cmd["vy"], cmd["vz"], cmd.get("yaw_rate", 0.0), hold_xy=hold)
        if cmd["vz"] > 0.2 and not L.armed():
            L.offboard()
            L.arm(True)
        self._next += self.agent_dt
        time.sleep(max(0.0, self._next - time.monotonic()))
        if time.monotonic() - self._next > 1.0:
            self._next = time.monotonic()  # the brain fell behind (a long look): no catching up in bursts
        return self.observe()

    def finish(self):
        """After the mission: on the ground -> disarm; in the air -> return to launch (the autopilot's RTL)."""
        if self.link.landed():
            self.link.arm(False)
        else:
            self.link.return_to_launch()

    # ---------------- sensors
    def _pose(self):
        lp, at = self.link.get("LOCAL_POSITION_NED"), self.link.get("ATTITUDE")
        p = np.array(MV.enu_from_ned(lp.x, lp.y, lp.z)) + self._origin
        v = np.array(MV.enu_from_ned(lp.vx, lp.vy, lp.vz))
        r, pi, y = MV.att_ours_from_px4(at.roll, at.pitch, at.yaw)
        w = MV.flu_from_frd(at.rollspeed, at.pitchspeed, at.yawspeed)
        self.data.qpos[:3] = p
        return p, v, (r, pi, y), np.array(w)

    def observe(self, images=None, lidar=True):
        L = self.link
        p, v, (r, pi, y), w = self._pose()
        imu = L.get("HIGHRES_IMU")
        acc = MV.flu_from_frd(imu.xacc, imu.yacc, imu.zacc) if imu else (0.0, 0.0, 9.81)
        gr, gp = L.get("GPS_RAW_INT", 2.0), L.get("GLOBAL_POSITION_INT", 2.0)
        gps = None
        if gr is not None and gp is not None and gr.fix_type >= 3:
            x, yy = self.frame.xy(gp.lat * 1e-7, gp.lon * 1e-7)
            gps = {
                "x": x,
                "y": yy,
                "z": gp.relative_alt / 1000.0,
                "lat": gp.lat * 1e-7,
                "lon": gp.lon * 1e-7,
                "hdop": gr.eph / 100.0 if gr.eph != 65535 else 9.9,
                "sats": gr.satellites_visible,
            }
        alt = L.get("ALTITUDE")
        baro = float(alt.altitude_relative) if alt is not None else float(p[2])
        ds = L.get("DISTANCE_SENSOR", 0.5)
        rng = (
            ds.current_distance / 100.0
            if ds is not None and ds.min_distance <= ds.current_distance <= ds.max_distance
            else -1.0
        )
        wc = L.get("WIND_COV", 2.0)
        wv = np.array(MV.enu_from_ned(wc.wind_x, wc.wind_y, wc.wind_z)) if wc is not None else np.zeros(3)
        bs = L.get("BATTERY_STATUS") or L.get("SYS_STATUS")
        bat = float(getattr(bs, "battery_remaining", -1)) if bs is not None else -1.0
        pts = self.lidar.points() if lidar else np.zeros((0, 3), np.float32)
        return {
            "t": round(self.t, 3),
            "imu": {"acc": list(acc), "gyro": w.tolist()},
            "attitude": {"roll": r, "pitch": pi, "yaw": y},
            "baro_alt": baro,
            "range_down": rng,
            "lidar": None,
            "lidar_points": pts,
            "vel_est": v.tolist(),
            "vel_source": "gps" if gps is not None else "inertial",
            "gps": gps,
            "link": {"up": L.gcs_alive(), "rssi_dbm": None, "internet": L.gcs_alive()},
            "wind": {
                "speed": float(np.linalg.norm(wv[:2])),
                "dir_rad": math.atan2(wv[1], wv[0]),
                "vec": wv.tolist(),
            },
            "battery_pct": bat,
        }

    def obs_yaw(self):
        return self._pose()[2][2]

    # ---------------- gimbal camera
    def set_gimbal(self, yaw_rel_deg, pitch_deg):
        self.gimbal_yaw, self.gimbal_pitch = float(yaw_rel_deg), float(np.clip(pitch_deg, -90, 20))
        self.link.gimbal(self.gimbal_pitch, self.gimbal_yaw)

    def gimbal_frame(self, depth=True):
        time.sleep(0.25)  # let the gimbal settle (a real gimbal slews ~90-180 deg/s)
        p, v, att, w = self._pose()
        return self.camera.frame(
            att,
            p,
            self.gimbal_yaw,
            self.gimbal_pitch,
            self.fovy,
            self.lidar,
            self.observe(lidar=False)["range_down"],
        )

    # ---------------- link, payload
    def link_up(self):
        return self.link.gcs_alive()

    def winch(self, length):
        self._L = float(np.clip(length, 0.0, 8.0))
        self.link.winch(self._L)

    def release_cargo(self):
        if self.cargo["state"] != "stowed":
            return False
        self.link.release_cargo()
        self.cargo["state"] = "released"
        return True

    def hook_load(self):
        """A load cell on the winch."""
        m = self.link.get("NAMED_VALUE_FLOAT")
        return bool(m is not None and m.name.startswith("HOOK_KG") and m.value > 0.15)

    def close(self):
        self.link.close()


# ---------------- real sensor plug-ins
class Ros2Livox:
    """Two Livox Mid-360 (one upright, one upside down, like the simulator) through livox_ros_driver2."""

    def __init__(
        self,
        topics=("/livox/lidar_up", "/livox/lidar_down"),
        extrinsics=((np.eye(3), np.zeros(3)), (np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, -0.08]))),
    ):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs_py import point_cloud2

        rclpy.init()
        self.node = Node("mission_brain_lidar")
        self.latest = {}
        self.pc2 = point_cloud2
        for tp, (R, t) in zip(topics, extrinsics):
            self.node.create_subscription(
                PointCloud2, tp, lambda m, tp=tp, R=R, t=t: self._cb(tp, m, R, t), 5
            )
        import threading

        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()

    def _cb(self, tp, msg, R, t):
        P = np.array(
            [
                (p[0], p[1], p[2])
                for p in self.pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            ],
            np.float32,
        ).reshape(-1, 3)
        self.latest[tp] = (P @ R.T + t).astype(np.float32)

    def points(self):
        return np.concatenate(list(self.latest.values())) if self.latest else np.zeros((0, 3), np.float32)


class RtspCamera:
    """The gimbal camera's video."""

    def __init__(self, url="rtsp://192.168.144.25:8554/main.264", size=(960, 720)):
        import cv2, threading

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(url)
        self.size = size
        self.last = None

        def grab():
            while True:
                ok, f = self.cap.read()
                if ok:
                    self.last = cv2.resize(f, size)[..., ::-1].copy()  # BGR -> RGB

        threading.Thread(target=grab, daemon=True).start()

    def frame(self, att, pos, yaw_rel_deg, pitch_deg, fovy, lidar, range_down):
        import nav

        W, H = self.size
        yaw = att[2] + math.radians(yaw_rel_deg)
        p = math.radians(pitch_deg)
        fwd = np.array([math.cos(p) * math.cos(yaw), math.cos(p) * math.sin(yaw), math.sin(p)])
        up = np.array([0, 0, 1.0]) if abs(p) < 1.5 else np.array([math.cos(yaw), math.sin(yaw), 0.0])
        right = np.cross(fwd, up)
        up = np.cross(right, fwd)
        f = (H / 2) / math.tan(math.radians(fovy) / 2)
        ys, xs = np.mgrid[0:H, 0:W]
        rays = fwd[None, None] + ((xs - W / 2) / f)[..., None] * right - ((ys - H / 2) / f)[..., None] * up
        depth = np.full((H, W), np.nan, np.float32)
        hg = range_down if range_down and range_down > 0 else None
        if hg is not None:  # the ground plane hg metres below the camera
            dz = rays[..., 2]
            t = np.where(dz < -1e-3, hg / -np.where(dz < -1e-3, dz, -1.0), np.nan)
            depth = (t * (rays @ fwd)).astype(np.float32)
        P = lidar.points()
        if len(P):  # lidar points override the plane where they land in the image
            Pw = P @ nav.rot(*att).T
            d = Pw @ fwd
            ok = d > 0.5
            u = (Pw[ok] @ right) / d[ok] * f + W / 2
            v = -(Pw[ok] @ up) / d[ok] * f + H / 2
            k = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            ui, vi = u[k].astype(int), v[k].astype(int)
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    depth[np.clip(vi + dy, 0, H - 1), np.clip(ui + dx, 0, W - 1)] = d[ok][k]
        return self.last.copy(), depth, (np.asarray(pos, float), fwd, up)
