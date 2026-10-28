"""MAVLink link between the mission brain."""

import math, threading, time


# ---------------- frames
def enu_from_ned(n, e, d):
    return e, n, -d


def ned_from_enu(x, y, z):
    return y, x, -z


def yaw_enu_from_ned(yaw):
    return (math.pi / 2 - yaw + math.pi) % (2 * math.pi) - math.pi


def att_ours_from_px4(roll, pitch, yaw):
    """Autopilot attitude."""
    return roll, -pitch, yaw_enu_from_ned(yaw)


def flu_from_frd(x, y, z):
    return x, -y, -z


PX4_OFFBOARD = 6  # PX4 custom main mode
TYPE_MASK_VEL_YAWRATE = (
    1 | 2 | 4 | 64 | 128 | 256 | 1024
)  # ignore position, acceleration, yaw angle: use velocity + yaw rate
TYPE_MASK_XY_VEL_YAWRATE = (
    4 | 64 | 128 | 256 | 1024
)  # x, y position + velocity (feed-forward) + yaw rate; z by velocity


class Link:
    def __init__(
        self, url="udpin:0.0.0.0:14540", autopilot="px4", rate_hz=20.0, watchdog_s=0.3, gcs_timeout_s=5.0
    ):
        from pymavlink import mavutil

        self.mu = mavutil
        self.m = mavutil.mavlink_connection(
            url, source_system=1, source_component=191
        )  # 191: onboard computer
        hb = self.m.wait_heartbeat(timeout=60)
        if hb is None:
            raise RuntimeError(f"no heartbeat on {url}")
        self.sys, self.comp, self.autopilot = self.m.target_system, self.m.target_component, autopilot
        self.msgs, self.t_msg = {}, {}
        self.lock = threading.Lock()
        self.cmd = (0.0, 0.0, 0.0, 0.0)
        self.hold_xy = None
        self.t_cmd = 0.0
        self.watchdog_s = watchdog_s
        self.gcs_timeout_s = gcs_timeout_s
        self.t_gcs = time.monotonic()
        self.running = True
        self.rate = rate_hz
        self.t0 = time.monotonic()
        self.watchdog_hits = 0
        self.texts = []  # the autopilot's own messages
        for name, hz in (
            ("LOCAL_POSITION_NED", 50),
            ("ATTITUDE", 50),
            ("HIGHRES_IMU", 50),
            ("GPS_RAW_INT", 10),
            ("GLOBAL_POSITION_INT", 10),
            ("ALTITUDE", 20),
            ("BATTERY_STATUS", 2),
            ("SYS_STATUS", 2),
            ("DISTANCE_SENSOR", 20),
            ("WIND_COV", 5),
            ("EXTENDED_SYS_STATE", 5),
            ("HOME_POSITION", 1),
        ):
            self.request(name, hz)
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._sender, daemon=True).start()

    # ---------------- background threads
    def _reader(self):
        while self.running:
            m = self.m.recv_match(blocking=True, timeout=0.5)
            if m is None:
                continue
            t = m.get_type()
            if t == "HEARTBEAT" and m.get_srcComponent() == 190:
                self.t_gcs = time.monotonic()  # the operator's ground station is alive
            if m.get_srcSystem() != self.sys:
                continue
            if t == "STATUSTEXT":
                self.texts = (self.texts + [(round(time.monotonic() - self.t0, 1), m.text)])[-60:]
            with self.lock:
                self.msgs[t] = m
                self.t_msg[t] = time.monotonic()

    def _sender(self):
        dt = 1.0 / self.rate
        while self.running:
            with self.lock:
                fresh = time.monotonic() - self.t_cmd < self.watchdog_s
                vx, vy, vz, wz = self.cmd if fresh else (0.0, 0.0, 0.0, 0.0)
                hold = self.hold_xy if fresh else None
            if not fresh and self.t_cmd > 0:
                self.watchdog_hits += 1
            n, e, d = ned_from_enu(vx, vy, vz)
            mask, pn, pe = TYPE_MASK_VEL_YAWRATE, 0.0, 0.0
            if hold is not None:
                mask = TYPE_MASK_XY_VEL_YAWRATE
                pn, pe = hold[1], hold[0]  # precise phases: PX4's position loop holds x, y
            self.m.mav.set_position_target_local_ned_send(
                int((time.monotonic() - self.t0) * 1000),
                self.sys,
                self.comp,
                self.mu.mavlink.MAV_FRAME_LOCAL_NED,
                mask,
                pn,
                pe,
                0,
                n,
                e,
                d,
                0,
                0,
                0,
                0,
                -wz,
            )  # yaw rate: CCW (ENU) -> CW (NED)
            if int(time.monotonic() / dt) % int(self.rate) == 0:
                self.m.mav.heartbeat_send(
                    self.mu.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                    self.mu.mavlink.MAV_AUTOPILOT_INVALID,
                    0,
                    0,
                    0,
                )
            time.sleep(dt)

    # ---------------- requests / commands
    def request(self, name, hz):
        mid = getattr(self.mu.mavlink, f"MAVLINK_MSG_ID_{name}")
        self.command(self.mu.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, mid, int(1e6 / hz))

    def command(self, cmd, *p):
        p = list(p) + [0] * (7 - len(p))
        self.m.mav.command_long_send(self.sys, self.comp, cmd, 0, *p)

    def get(self, name, max_age=None):
        with self.lock:
            m = self.msgs.get(name)
            if m is None or (max_age is not None and time.monotonic() - self.t_msg[name] > max_age):
                return None
            return m

    def set_velocity_enu(self, vx, vy, vz, yaw_rate, hold_xy=None):
        """Velocity command (world ENU, yaw rate CCW)."""
        with self.lock:
            self.cmd = (float(vx), float(vy), float(vz), float(yaw_rate))
            self.hold_xy = None if hold_xy is None else (float(hold_xy[0]), float(hold_xy[1]))
            self.t_cmd = time.monotonic()

    def armed(self):
        hb = self.get("HEARTBEAT")
        return bool(hb and hb.base_mode & self.mu.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def landed(self):
        s = self.get("EXTENDED_SYS_STATE")
        return bool(s and s.landed_state == self.mu.mavlink.MAV_LANDED_STATE_ON_GROUND)

    def arm(self, on=True):
        self.command(self.mu.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1 if on else 0)

    def offboard(self):
        """PX4: OFFBOARD (the setpoint stream must already run: it does from the start); ArduPilot: GUIDED."""
        if self.autopilot == "px4":
            self.command(
                self.mu.mavlink.MAV_CMD_DO_SET_MODE,
                self.mu.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                PX4_OFFBOARD,
            )
        else:
            self.m.set_mode("GUIDED")

    def return_to_launch(self):
        self.command(self.mu.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH)

    def land_here(self):
        self.command(self.mu.mavlink.MAV_CMD_NAV_LAND)

    def gimbal(self, pitch_deg, yaw_rel_deg):
        """Gimbal protocol v2."""
        self.command(
            self.mu.mavlink.MAV_CMD_DO_GIMBAL_MANAGER_PITCHYAW,
            float(pitch_deg),
            float(-yaw_rel_deg),
            float("nan"),
            float("nan"),
            0,
            0,
            0,
        )

    def release_cargo(self):
        self.command(
            self.mu.mavlink.MAV_CMD_DO_GRIPPER, 1, 0
        )  # gripper 1: release (the parachute cargo hangs on it)

    def winch(self, length_m, rate=0.6):
        self.command(42600, 1, 1, float(length_m), float(rate))  # MAV_CMD_DO_WINCH, relative length control

    def gcs_alive(self):
        return time.monotonic() - self.t_gcs < self.gcs_timeout_s

    def close(self):
        self.running = False
        time.sleep(0.2)
        self.m.close()
