"""Drone environment: MuJoCo physics + a PX4-like velocity controller + a realistic sensor suite with."""

import math
import numpy as np
import mujoco
import world

G = 9.81
TAN35 = math.tan(math.radians(35))
REF_LATLON = (47.3769, 8.5417)  # arbitrary reference for lat/lon output


class DroneEnv:
    def __init__(
        self,
        seed=0,
        wind=(0.0, 0.0),
        gust=0.0,
        gps_faults=(),
        link_faults=(),
        flow=True,
        cam=(160, 120),
        lidar_rays=72,
        agent_hz=10,
        render=True,
        goal=0,
        visibility=0.0,
        throws=(),
        dives=(),
        lidar3d=(600, 33, -7.0, 52.0, 40.0),
        lidar_mounts=("up", "down"),
        encounters=None,
        phys=None,
        faults=(),
        place=None,
        start=None,
        mission=None,
        battery0=100.0,
        fast_lidar=True,
        battery_cutoff=False,
        prebuilt=None,
    ):
        files = None  # prebuilt = (World, xml, in-memory texture files)
        if prebuilt is not None:
            self.W, xml, files = prebuilt
            if start is None:
                start = self.W.start
        elif place is None:
            self.W = world.generate(seed)
            xml = world.to_mjcf(self.W)
        else:
            import osm_world

            self.W, xml = osm_world.build(place, seed)  # a real place from OpenStreetMap
        self.model = (
            mujoco.MjModel.from_xml_string(xml, files) if files else mujoco.MjModel.from_xml_string(xml)
        )
        self.data = mujoco.MjData(self.model)
        self.water = getattr(self.W, "water", [])
        self._over_water = False
        self.fast_lidar = fast_lidar
        self.battery_cutoff = battery_cutoff  # stage D: an empty battery stops the motors
        self.start_req = start
        self.mission_req = mission
        self.battery0 = battery0  # start: None (pad at origin) | 'random' | (x, y); mission: see reset()
        m = self.model
        self.dt = m.opt.timestep
        self.sub = int(round(1 / (agent_hz * self.dt)))
        self.agent_dt = self.sub * self.dt
        self.bid = m.body("drone").id
        self.mass = float(m.body_subtreemass[self.bid])
        self.J = np.array(m.body_inertia[self.bid])
        # random physics (the autopilot keeps its NOMINAL model
        ph = dict(mass_scale=1.0, thrust_scale=1.0, drag_scale=1.0, lidar_latency=0.05)
        ph.update(phys or {})
        self.phys = ph
        m.body_mass[self.bid] *= ph["mass_scale"]
        m.body_inertia[self.bid] *= ph["mass_scale"]
        self.faults = [dict(f) for f in faults]  # {'sensor', 'kind', 't0', 't1', 'sev'}
        self.gps_faults = list(gps_faults) + [
            (f["t0"], f["t1"], f["kind"]) for f in self.faults if f["sensor"] == "gps"
        ]
        link_faults = list(link_faults) + [(f["t0"], f["t1"]) for f in self.faults if f["sensor"] == "link"]
        L = world.ARM / math.sqrt(2)
        xs = [sx * L for sx, _, _ in world.ROTORS]
        ys = [sy * L for _, sy, _ in world.ROTORS]
        self.mix = np.linalg.inv(
            np.array([[1] * 4, ys, [-x for x in xs], [s * world.YAW_K for *_, s in world.ROTORS]])
        )
        self.wind_mean = np.array([*wind, 0.0])
        self.gust = gust
        self.link_faults = list(link_faults)
        self.flow = flow
        self.cam = cam
        self.n_lidar = lidar_rays
        self.render_on = render
        self.renderer = mujoco.Renderer(m, cam[1], cam[0], max_geom=30000) if render else None
        self.vopt = mujoco.MjvOption()
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0  # no sensor rays in images
        ext = float(m.stat.extent)
        m.vis.map.znear = 0.05 / ext
        m.vis.map.zfar = 900.0 / ext  # clipping in METRES (MuJoCo scales by extent)
        self.visibility = visibility  # metres of visibility in fog/haze; 0 = clear air
        if visibility > 0:
            ext = float(m.stat.extent)
            m.vis.map.fogstart = 0.15 * visibility / ext
            m.vis.map.fogend = visibility / ext
        self.mocap = [(m.body(f"mover{i}").mocapid[0], mv) for i, mv in enumerate(self.W.movers)]
        self.mover_bodies = {m.body(f"mover{i}").id for i in range(len(self.W.movers))}
        self.wings = {
            i: (m.body(f"mover{i}_wl").mocapid[0], m.body(f"mover{i}_wr").mocapid[0])
            for i, mv in enumerate(self.W.movers)
            if mv.get("v2") and mv["kind"] == "bird"
        }
        self.mover_bodies |= {m.body(f"mover{i}_w{s}").id for i in self.wings for s in "lr"}
        self.proj = [m.body(f"proj{i}").mocapid[0] for i in range(world.N_PROJECTILES)]
        self.proj_bodies = {m.body(f"proj{i}").id for i in range(world.N_PROJECTILES)}
        self.mover_bodies |= self.proj_bodies
        self.throws = sorted(throws)
        self.dives = sorted(dives)  # (t, speed) and (t, bird index), scheduled hazard events
        self.enc_pool = [(sz, m.body(f"enc{i}").mocapid[0]) for i, sz in enumerate(world.ENC_SIZES)]
        self.enc_bodies = {m.body(f"enc{i}").id for i in range(len(world.ENC_SIZES))}
        self.mover_bodies |= self.enc_bodies
        self.enc_kw = encounters
        self.enc_seed = seed + 5000
        # 3D lidar like a Livox Mid-360
        n_az, n_el, e0, e1, self.lidar_max = lidar3d
        # ~Livox Mid-360S per unit
        # Two units are a common way to cover threats from below
        self.lidar_grids = [
            (n_az, n_el, math.radians(e0 if m == "up" else -e1), math.radians(e1 if m == "up" else -e0))
            for m in lidar_mounts
        ]
        self.tall = [(x, y, max(sx, sy) / 2, h) for x, y, sx, sy, h, _ in self.W.buildings]
        self.rng = np.random.default_rng(seed + 1000)
        self.goal = self.W.targets[goal] if self.W.targets else None
        self.drone_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == self.bid}
        self.soft_geoms = {m.geom("ground").id, m.geom("pad").id}
        self.scene = None
        if fast_lidar:  # static world in an Embree BVH
            import raymesh

            mujoco.mj_forward(m, self.data)
            self.scene = raymesh.StaticScene(m, self.data, self.bid)
        self.reset()

    # ---------- episode ----------
    def fork(self):
        """An exact copy of the whole world."""
        import copy

        new = object.__new__(DroneEnv)
        new.__dict__ = dict(self.__dict__)
        new.data = mujoco.MjData(self.model)
        mujoco.mj_copyData(new.data, self.model, self.data)
        new.renderer = None
        new.render_on = False
        for k in (
            "rng",
            "enc",
            "dive_state",
            "actor_state",
            "active_throws",
            "hit_events",
            "events",
            "_last_hit",
            "_in_contact",
            "v_est",
            "acc_bias",
            "gust_v",
            "gps_drift",
            "gps_drift_rate",
            "gps_last",
            "_imu_jumped",
            "dives",
            "throws",
            "world_events",
        ):
            if k in self.__dict__:
                new.__dict__[k] = copy.deepcopy(self.__dict__[k])
        return new

    def close(self):
        """Release the OpenGL context: several renderers alive in one process render black frames."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def free_spot(self, rng, near=None, r_min=0.0, r_max=None, tries=400):
        """A point on open ground."""
        half = self.W.size / 2 - 10
        up = np.array([0.0, 0.0, 1.0])
        gid = np.zeros(1, np.int32)
        for _ in range(tries):
            if near is None:
                x, y = rng.uniform(-half, half), rng.uniform(-half, half)
            else:
                a = rng.uniform(0, 2 * math.pi)
                r = rng.uniform(r_min, r_max or half)
                x, y = near[0] + r * math.cos(a), near[1] + r * math.sin(a)
                if abs(x) > half or abs(y) > half:
                    continue
            p0 = np.array([x, y, 1.0])
            dirs = np.array(
                [[0, 0, 1.0]]
                + [[math.cos(k * math.pi / 4), math.sin(k * math.pi / 4), 0.0] for k in range(8)]
            )
            if (
                self.scene is not None
            ):  # two-sided triangles: a point INSIDE a building sees its walls and roof
                dist, _ = self.scene.cast(self.model, self.data, p0, dirs, 60.0, self.bid)
                if 0 <= dist[0] < 40 or ((dist[1:] >= 0) & (dist[1:] < 3)).any():
                    continue
                return (float(x), float(y))
            sky = mujoco.mj_ray(self.model, self.data, p0, up, None, 1, self.bid, gid)
            if 0 <= sky < 40:
                continue
            ok = True
            for k in range(8):
                d = dirs[k + 1]
                r_ = mujoco.mj_ray(self.model, self.data, p0, d, None, 1, self.bid, gid)
                if 0 <= r_ < 3:
                    ok = False
                    break
            if ok:
                return (float(x), float(y))
        return (0.0, 0.0)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.t = 0.0
        self.yaw_sp = 0.0
        mujoco.mj_forward(self.model, self.data)
        rs = np.random.default_rng(self.enc_seed + 77)
        self.home = (
            (0.0, 0.0)
            if self.start_req is None
            else (self.free_spot(rs) if self.start_req == "random" else tuple(self.start_req))
        )
        self.data.qpos[0], self.data.qpos[1] = self.home
        # mission: list of legs
        if self.mission_req is None:
            self.mission = [("goal", self.W.targets.index(self.goal))] if self.goal is not None else []
        else:
            self.mission = list(self.mission_req)
        self.leg = 0
        self._last_hit = {}
        self.v_est = np.zeros(3)
        self.acc_bias = self.rng.normal(0, 0.05, 3)
        self.baro_bias = self.rng.normal(0, 0.5)
        self.gust_v = np.zeros(3)
        self.gps_last = None
        self.gps_drift = np.zeros(3)
        self.battery = self.battery0
        self.i_vz = 0.0
        self.i_vx = self.i_vy = 0.0
        self.strain_steps = 0  # physics steps at the motor limit or tilted beyond 30 degrees
        self._lidar_cache = None
        self._imu_jumped = set()
        self.collisions = 0
        self.max_contact_speed = 0.0
        self.max_ground_speed = 0.0
        self.log = []
        self.gps_drift_rate = np.zeros(3)
        self.mover_hits = 0
        self.projectile_hits = 0
        self.events = []
        self.active_throws = []
        self.dive_state = {}
        self.hit_events = []
        self._in_contact = set()
        self.actor_state = {}
        self.world_events = []
        self.drain_mult = 1.0  # announced changes of the world (stage D2 messages, messages.py)
        import encounters as _enc

        self.enc = (
            _enc.EncounterGenerator(np.random.default_rng(self.enc_seed), **self.enc_kw)
            if self.enc_kw is not None
            else None
        )
        self._move()
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def gps_mode(self):
        mode = "ok"
        for t0, t1, kind in self.gps_faults:
            if t0 <= self.t < t1:
                mode = kind
        return mode

    def fault(self, sensor, kind=None):
        """Active fault of a sensor right now (or None)."""
        for f in self.faults:
            if f["sensor"] == sensor and f["t0"] <= self.t < f["t1"] and (kind is None or f["kind"] == kind):
                return f
        return None

    def link_up(self):
        return not any(t0 <= self.t < t1 for t0, t1 in self.link_faults)

    # ---------- physics + low-level control ----------
    def _state(self):
        d = self.data
        q = d.qpos[3:7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        return (
            d.qpos[:3].copy(),
            d.qvel[:3].copy(),
            R,
            d.qvel[3:6].copy(),
        )  # free joint: world lin vel, body ang vel

    def _estimator(self, v_true, R, acc_world):
        mode = self.gps_mode()
        h = self._range_down()
        if mode in ("ok", "degraded", "drift"):  # a spoofed GPS still looks healthy to the autopilot
            self.v_est = v_true + self.rng.normal(0, 0.08 if mode != "degraded" else 0.4, 3)
            self.vel_source = "gps"
            if mode == "drift":
                self.v_est = self.v_est + self.gps_drift_rate
        elif (
            getattr(self, "lio", False)
            and self._lidar_cache is not None
            and len(self._lidar_cache) >= 300
            and self.fault("lidar", "off") is None
        ):
            # lidar-inertial odometry
            self.v_est = v_true + self.rng.normal(0, 0.03 + 0.01 * float(np.linalg.norm(v_true)), 3)
            self.vel_source = "lio"
        elif (
            self.flow and 0 < h < 30 and self.fault("flow") is None and not self._over_water
        ):  # no texture over water
            self.v_est = v_true + self.rng.normal(0, 0.04 + 0.01 * h, 3)
            self.vel_source = "flow"
        else:  # dead reckoning on the accelerometer: bias random walk -> drift
            # bias random walk: stage F uses a MEMS-like 5e-4 m/s2/sqrt(s)
            self.acc_bias += self.rng.normal(0, getattr(self, "acc_rw", 0.002), 3)
            self.v_est = self.v_est + (acc_world + self.acc_bias) * self.dt
            self.v_est[2] = v_true[2] + self.rng.normal(
                0, 0.1
            )  # height is always baro-aided in a real autopilot
            self.vel_source = "inertial"

    def _control(self, cmd, R, w):
        # Same maths as a vector implementation
        vx = min(max(cmd.get("vx", 0.0), -12.0), 12.0)
        vy = min(max(cmd.get("vy", 0.0), -12.0), 12.0)
        vz = min(max(cmd.get("vz", 0.0), -3.0), 4.0)
        ve = self.v_est
        ax = 2.5 * (vx - ve[0])
        ay = 2.5 * (vy - ve[1])
        az = 4.0 * (vz - ve[2])
        # horizontal integrator
        # short of its point when the commanded speed is small
        self.i_vx = min(max(self.i_vx + (vx - ve[0]) * self.dt, -4.0), 4.0)
        self.i_vy = min(max(self.i_vy + (vy - ve[1]) * self.dt, -4.0), 4.0)
        ax += self.i_vx
        ay += self.i_vy
        self.i_vz = min(
            max(self.i_vz + (vz - ve[2]) * self.dt, -3.0), 10.0
        )  # up to +100 % mass (payloads); +-3 limited it to +46 %
        az += G + 1.5 * self.i_vz  # integrator: the autopilot does not know the true mass
        ah = math.hypot(ax, ay)
        lim = az * TAN35
        amax = 0.9 * 4 * world.MOTOR_MAX * self.phys["thrust_scale"] / self.mass
        lim = min(lim, math.sqrt(max(amax * amax - az * az, 0.0)))  # thrust priority: height first (as PX4)
        if ah > lim:
            k = lim / ah
            ax *= k
            ay *= k
        self.yaw_sp += min(max(cmd.get("yaw_rate", 0.0), -1.5), 1.5) * self.dt
        n = math.sqrt(ax * ax + ay * ay + az * az)
        zx, zy, zz = ax / n, ay / n, az / n
        cx, cy = math.cos(self.yaw_sp), math.sin(self.yaw_sp)
        yx, yy, yz = -zz * cy, zz * cx, zx * cy - zy * cx
        n = math.sqrt(yx * yx + yy * yy + yz * yz)
        yx, yy, yz = yx / n, yy / n, yz / n  # zd x xc
        xx, xy, xz = yy * zz - yz * zy, yz * zx - yx * zz, yx * zy - yy * zx  # yd x zd
        Rd = np.array([[xx, yx, zx], [xy, yy, zy], [xz, yz, zz]])
        T = self.mass * (ax * R[0, 2] + ay * R[1, 2] + az * R[2, 2])  # nominal mass
        E = 0.5 * (Rd.T @ R - R.T @ Rd)
        e0, e1, e2 = E[2, 1], E[0, 2], E[1, 0]
        J0, J1, J2 = self.J
        wx, wy, wz = w
        j0, j1, j2 = J0 * wx, J1 * wy, J2 * wz
        tau = (
            J0 * (-60.0 * e0 - 12.0 * wx) + (wy * j2 - wz * j1),
            J1 * (-60.0 * e1 - 12.0 * wy) + (wz * j0 - wx * j2),
            J2 * (-60.0 * e2 - 12.0 * wz) + (wx * j1 - wy * j0),
        )
        f = self.mix @ np.array([T, *tau])
        np.clip(f, 0, world.MOTOR_MAX, out=f)
        if self.battery_cutoff and self.battery <= 0.0:
            f[:] = 0.0
        self.data.ctrl[:] = f * self.phys["thrust_scale"]  # real motors differ
        if f.max() >= 0.98 * world.MOTOR_MAX or R[2, 2] < math.cos(math.radians(30)):
            self.strain_steps += 1
        self.battery -= float((f**1.5).sum()) * self.dt * 2.5e-3 * self.drain_mult

    def _wind(self, v):
        tau = 2.0
        self.gust_v += (-self.gust_v / tau) * self.dt + self.gust * math.sqrt(
            2 * self.dt / tau
        ) * self.rng.normal(0, 1, 3) * np.array([1, 1, 0.3])
        rel = (self.wind_mean + self.gust_v) - v
        F = 0.5 * 1.225 * 0.08 * self.phys["drag_scale"] * np.linalg.norm(rel) * rel
        self.data.xfrc_applied[self.bid, :3] = F
        return self.wind_mean + self.gust_v

    def _move(self, dt=0.0):
        p_drone = self.data.qpos[:3]
        if self.enc is not None and dt > 0:  # encounter generator (world side): spawn, move, swept hit test
            used = {e.body for e in self.enc.active}
            free = [(sz, b) for sz, b in self.enc_pool if b not in used]
            self.enc.tick(self.t, dt, p_drone.copy(), self.data.qvel[:3].copy(), free)
            for e in self.enc.active:
                if e.hit and not getattr(e, "logged", False):
                    e.logged = True
                    self.hit_events.append(
                        {
                            "t": round(self.t, 3),
                            "kind": "encounter",
                            "id": e.id,
                            "family": e.fam,
                            "speed": round(e.speed, 1),
                        }
                    )
            on = {e.body: e.p for e in self.enc.active}
            for sz, b in self.enc_pool:
                self.data.mocap_pos[b] = on.get(b, (0, 0, -30))
        while self.dives and self.dives[0][0] <= self.t:  # a bird starts diving at the drone
            t0, b = self.dives.pop(0)
            birds = [i for i, (_, mv) in enumerate(self.mocap) if mv.get("diver")]
            if birds:
                i = birds[b % len(birds)]
                self.dive_state[i] = {"t0": self.t, "pos": np.array(self.data.mocap_pos[self.mocap[i][0]])}
                self.events.append({"t": round(self.t, 2), "kind": "dive", "bird": i})
        while self.throws and self.throws[0][0] <= self.t:  # a stone is thrown at the drone from the ground
            t0, speed = self.throws.pop(0)
            ang = self.rng.uniform(0, 2 * math.pi)
            d = self.rng.uniform(12, 25)
            p0 = np.array([p_drone[0] + d * math.cos(ang), p_drone[1] + d * math.sin(ang), 1.5])
            T = float(np.linalg.norm(p_drone - p0)) / speed
            aim = p_drone + self.data.qvel[:3] * T  # lead the target
            v0 = (aim - p0) / T - 0.5 * np.array([0, 0, -G]) * T
            k = len([1 for a in self.active_throws if self.t - a["t0"] < 4]) % len(self.proj)
            self.active_throws = [a for a in self.active_throws if a["k"] != k] + [
                {"k": k, "t0": self.t, "p0": p0, "v0": v0}
            ]
            self.events.append(
                {"t": round(self.t, 2), "kind": "throw", "speed": speed, "from_dist": round(d, 1)}
            )
        for a in self.active_throws:
            tt = self.t - a["t0"]
            pos = a["p0"] + a["v0"] * tt + 0.5 * np.array([0, 0, -G]) * tt * tt
            prev = a.get("prev", a["p0"])
            seg = pos - prev
            L2 = float(seg @ seg)
            u = (
                0.0 if L2 < 1e-12 else float(np.clip((p_drone - prev) @ seg / L2, 0, 1))
            )  # swept test: no tunnelling at any speed
            if not a.get("hit") and tt < 4 and np.linalg.norm(prev + u * seg - p_drone) < 0.3 + 0.11:
                a["hit"] = True
                self.hit_events.append({"t": round(self.t, 3), "kind": "throw"})
            a["prev"] = pos
            self.data.mocap_pos[self.proj[a["k"]]] = pos if (tt < 4 and pos[2] > 0) else (0, 0, -20)
        for i, (mid, mv) in enumerate(self.mocap):
            if mv.get("v2"):  # stage F actors: birds with wings, cars on real roads, people on
                import actors

                P = actors.pose(mv, self.t, p_drone, self.actor_state.setdefault(i, {}))
                pos, q = P[""]
                self.data.mocap_pos[mid] = pos
                self.data.mocap_quat[mid] = q
                if i in self.wings:
                    for wid, suf, fy in ((self.wings[i][0], "wl", 1.45), (self.wings[i][1], "wr", -1.45)):
                        wp, wq = P[suf]
                        self.data.mocap_pos[wid] = wp
                        self.data.mocap_quat[wid] = (
                            actors.qmul(q, actors.quat_yaw(fy)) if P.get("fold") else wq
                        )  # perched: wings folded back
                continue
            x, y, z, yaw = world.mover_pose(mv, self.t)
            ds = self.dive_state.get(i)
            if ds is not None:
                tt = self.t - ds["t0"]
                if tt < 3.0:  # pursue the drone at 14 m/s
                    d = p_drone - ds["pos"]
                    n = float(np.linalg.norm(d))
                    if n > 0.05 and dt > 0:
                        ds["pos"] = ds["pos"] + d / n * min(n, 14.0 * dt)
                    x, y, z = ds["pos"]
                    yaw = math.atan2(d[1], d[0])
                elif tt < 5.0:  # blend back to the circling path
                    u = (tt - 3.0) / 2.0
                    x, y, z = (1 - u) * ds["pos"] + u * np.array([x, y, z])
                else:
                    del self.dive_state[i]
            self.data.mocap_pos[mid] = (x, y, z)
            self.data.mocap_quat[mid] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def _world_events(self):
        """Announced changes of the world."""
        for ev in self.world_events:
            if ev.get("done") or self.t < ev["t_event"]:
                continue
            if ev["kind"] == "drain" and ev.get(
                "ramp_s"
            ):  # cold: the battery cools over ~20 s, capacity falls gradually (D3)
                u = min(1.0, (self.t - ev["t_event"]) / ev["ramp_s"])
                self.drain_mult = 1.0 + (ev["mult"] - 1.0) * u
                ev["done"] = u >= 1.0
                continue
            ev["done"] = True
            if ev["kind"] == "wind":
                self.wind_mean = np.array([*ev["vec"], 0.0])
                self.gust = max(self.gust, ev["gust"])
            elif ev["kind"] == "drain":
                self.drain_mult = ev["mult"]

    def step(self, cmd, sense=True):
        if self.world_events:
            self._world_events()
        if (
            self.gps_mode() == "drift" and not self.gps_drift_rate.any()
        ):  # spoofing: the fix walks away at 0.6 m/s
            ang = self.rng.uniform(0, 2 * math.pi)
            self.gps_drift_rate = 0.6 * np.array([math.cos(ang), math.sin(ang), 0.0])
        cap = max(
            0, self.sub - int(round(self.phys["lidar_latency"] / self.dt))
        )  # lidar frame is older than the rest
        if self.water:
            from shapely.geometry import Point

            q = Point(float(self.data.qpos[0]), float(self.data.qpos[1]))
            self._over_water = any(w.contains(q) for w in self.water)
        for k in range(self.sub):
            if k % 5 == 0:
                self._move(5 * self.dt)  # moving objects and hazards at 100 Hz
            if k == cap and sense is True:
                pp, _, RR, _ = self._state()
                self._lidar_cache = self._lidar3d(pp, RR)
            f = self.fault("imu", "bias")
            if f is not None and id(f) not in self._imu_jumped:
                self._imu_jumped.add(id(f))
                self.acc_bias += self.rng.normal(0, f["sev"], 3)
            p, v, R, w = self._state()
            v_prev = v
            self._wind(v)
            self._control(cmd, R, w)
            mujoco.mj_step(self.model, self.data)
            self.t += self.dt
            acc_world = (self.data.qvel[:3] - v_prev) / self.dt
            self._estimator(self.data.qvel[:3].copy(), R, acc_world)
            if self.gps_mode() == "drift":
                self.gps_drift += self.gps_drift_rate * self.dt
            self._contacts()
        return (
            (self.observe() if sense is True else self.observe(lidar=False)) if sense else None
        )  # 'nolidar': cheap plan rollouts

    def _contacts(self):
        d = self.data
        now = set()
        for i in range(d.ncon):
            c = d.contact[i]
            g = {c.geom1, c.geom2}
            if not g & self.drone_geoms:
                continue
            speed = float(np.linalg.norm(d.qvel[:3]))
            if (g - self.drone_geoms) <= self.soft_geoms:
                self.max_ground_speed = max(self.max_ground_speed, speed)  # landing / crash on ground
            else:
                self.collisions += 1
                self.max_contact_speed = max(self.max_contact_speed, speed)
                if any(self.model.geom_bodyid[x] in self.mover_bodies for x in g):
                    self.mover_hits += 1
                if any(self.model.geom_bodyid[x] in self.proj_bodies for x in g):
                    self.projectile_hits += 1
                other = [self.model.geom_bodyid[x] for x in g if x not in self.drone_geoms]
                if other:
                    now.add(int(other[0]))
        for b in now - self._in_contact:  # count EVENTS (a new contact), not simulation steps
            if (
                b not in self.proj_bodies
                and b not in self.enc_bodies
                and self.t - self._last_hit.get(b, -9.0) > 0.5
            ):
                kind = "mover" if b in self.mover_bodies else "static"
                self.hit_events.append(
                    {
                        "t": round(self.t, 3),
                        "kind": kind,
                        "body": int(b),
                        "speed": round(float(np.linalg.norm(d.qvel[:3])), 2),
                    }
                )
            self._last_hit[b] = self.t
        self._in_contact = now

    # ---------- sensors ----------
    def _range_down(self):
        return float(self.data.sensor("range_down").data[0])

    def _lidar(self, p, yaw):
        n = self.n_lidar
        ang = yaw + np.linspace(0, 2 * math.pi, n, endpoint=False)
        vec = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], 1).reshape(-1).astype(np.float64)
        gid = np.zeros(n, np.int32)
        dist = np.zeros(n, np.float64)
        mujoco.mj_multiRay(
            self.model, self.data, p.astype(np.float64), vec, None, 1, self.bid, gid, dist, None, n, 30.0
        )
        dist[dist < 0] = 30.0
        return np.clip(dist + self.rng.normal(0, 0.03, n), 0, 30.0)

    def _lidar3d(self, p, R):
        bodies = []
        for n_az, n_el, e0, e1 in self.lidar_grids:
            daz, del_ = 2 * math.pi / n_az, (e1 - e0) / (n_el - 1)
            az = np.arange(n_az) * daz + self.rng.uniform(0, daz)
            el = e0 + np.arange(n_el) * del_
            A, E = np.meshgrid(az, el)
            A = A + self.rng.uniform(-daz / 2, daz / 2, A.shape)
            E = E + self.rng.uniform(-del_ / 2, del_ / 2, E.shape)
            bodies.append(
                np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
            )  # non-repetitive pattern
        body = np.concatenate(bodies)
        dirs = body @ R.T
        n = len(dirs)
        if self.scene is not None:
            dist, gid = self.scene.cast(self.model, self.data, p, dirs, self.lidar_max, self.bid)
        else:
            gid = np.zeros(n, np.int32)
            dist = np.zeros(n, np.float64)
            mujoco.mj_multiRay(
                self.model,
                self.data,
                p.astype(np.float64),
                dirs.reshape(-1).astype(np.float64),
                None,
                1,
                self.bid,
                gid,
                dist,
                None,
                n,
                self.lidar_max,
            )
        rmax = self.lidar_max
        f = self.fault("lidar", "range")
        if f is not None:
            rmax *= f["sev"]  # rain / fog / dust: shorter range
        p_ret = 0.98 - 0.5 * np.clip((dist - 0.6 * rmax) / (0.4 * rmax), 0, 1)  # weak returns near max range
        hit = (dist > 0) & (dist < rmax) & (self.rng.random(n) < p_ret)
        f = self.fault("lidar", "sector")
        if f is not None:  # dirt on the window: a blind azimuth sector (in the body frame)
            azb = np.arctan2(body[:, 1], body[:, 0])
            hit &= (
                np.abs((azb - f.get("center", 0.0) + math.pi) % (2 * math.pi) - math.pi)
                > f["sev"] * math.pi / 2
            )
        if self.fault("lidar", "off") is not None:
            hit[:] = False
        self.lidar_dirs = body
        r = dist[hit] + self.rng.normal(0, 0.03, int(hit.sum()))
        self._last_lidar_gid = gid[hit]  # evaluation only (what was hit), never given to the brain
        pts = (self.lidar_dirs[hit] * r[:, None]).astype(np.float32)  # points in the BODY frame
        f = self.fault("lidar", "spurious")
        if f is not None:  # sunlight / crosstalk from another lidar: false points
            k = int(f["sev"])
            u = self.rng.normal(size=(k, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            pts = np.concatenate([pts, (u * self.rng.uniform(2, 30, (k, 1))).astype(np.float32)])
        return pts

    def _render(self, camera):
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.vopt)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = self.visibility > 0
        return self.renderer.render().copy()

    def observe(self, images=None, lidar=True):
        p, v, R, w = self._state()
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = -math.asin(max(-1, min(1, R[2, 0])))
        acc_body = R.T @ (np.array([0, 0, G]) + (self.data.qacc[:3] if self.data.qacc is not None else 0))
        mode = self.gps_mode()
        gps = None
        canyon = any(
            math.hypot(p[0] - x, p[1] - y) - r < 12 and h > p[2] for x, y, r, h in self.tall
        )  # multipath near tall walls
        if mode in ("ok", "degraded", "drift"):
            sig = (1.2 if mode == "ok" else 6.0) * (3 if canyon else 1)
            noise = self.rng.normal(0, [sig, sig, 2 * sig])
            fix = p + noise + (self.gps_drift if mode == "drift" else 0)
            self.gps_last = fix
            lat = REF_LATLON[0] + fix[1] / 111_320.0
            lon = REF_LATLON[1] + fix[0] / (111_320.0 * math.cos(math.radians(REF_LATLON[0])))
            gps = {
                "x": float(fix[0]),
                "y": float(fix[1]),
                "z": float(fix[2]),
                "lat": lat,
                "lon": lon,
                "hdop": (0.8 if mode != "degraded" else 6.0) * (3 if canyon else 1),
                "sats": (14 if mode != "degraded" else 5) - (5 if canyon else 0),
            }
        elif mode == "stuck" and self.gps_last is not None:
            f = self.gps_last
            gps = {"x": float(f[0]), "y": float(f[1]), "z": float(f[2]), "hdop": 0.8, "sats": 12}
        home = float(math.hypot(p[0] - self.home[0], p[1] - self.home[1]))
        up = self.link_up()
        wtrue = self.wind_mean + self.gust_v  # anemometer: wind vector estimate
        wmeas = wtrue + self.rng.normal(0, 0.3, 3)
        obs = {
            "t": round(self.t, 3),
            "imu": {
                "acc": (acc_body + self.rng.normal(0, 0.05, 3)).tolist(),
                "gyro": (w + self.rng.normal(0, 0.005, 3)).tolist(),
            },
            "attitude": {
                "roll": roll + self.rng.normal(0, 0.005),
                "pitch": pitch + self.rng.normal(0, 0.005),
                "yaw": yaw
                + self.rng.normal(0, 0.03)
                + (self.fault("compass")["sev"] if self.fault("compass") else 0.0),
            },
            "baro_alt": float(p[2] + self.baro_bias + self.rng.normal(0, 0.15)),
            "range_down": self._range_down(),
            "lidar": self._lidar(p, yaw) if lidar else None,
            "lidar_points": (
                (self._lidar_cache if self._lidar_cache is not None else self._lidar3d(p, R))
                if lidar
                else np.zeros((0, 3), np.float32)
            ),
            "vel_est": self.v_est.tolist(),
            "vel_source": getattr(self, "vel_source", "gps"),
            "gps": gps,
            "link": {
                "up": up,
                "rssi_dbm": (-40 - 20 * math.log10(max(1.0, home))) if up else None,
                "internet": up,
            },
            "wind": {
                "speed": float(np.linalg.norm(wmeas[:2])),
                "dir_rad": math.atan2(wmeas[1], wmeas[0]),
                "vec": wmeas.tolist(),
            },
            "battery_pct": round(self.battery, 2),
            "mission": {
                "goal_description": self.goal.description if self.goal is not None else "",
                "legs": [self.leg_text(l) for l in self.mission],
            },
        }
        if self.render_on and images is not False:
            obs["cam_front"] = self._render("front")
            obs["cam_down"] = self._render("down")
        return obs

    def leg_text(self, leg):
        """What the mission says in words (as an operator would): no coordinates for goals, only descriptions."""
        if leg[0] == "goal":
            return "fly to " + self.W.targets[leg[1]].description
        if leg[0] == "home":
            return "return to the take-off point"
        return f"fly to the point {leg[1][0]:.0f} m east, {leg[1][1]:.0f} m north of the take-off point"

    # ---------- scoring only ----------
    def truth(self):
        p, v, R, w = self._state()
        gx, gy = self.goal.pos
        cars = [math.hypot(p[0] - cx, p[1] - cy) for cx, cy, *_ in self.W.cars]
        return {
            "pos": p.tolist(),
            "vel": v.tolist(),
            "dist_to_goal_xy": float(math.hypot(p[0] - gx, p[1] - gy)),
            "nearest_car_xy": round(min(cars), 2) if cars else None,
            "collisions": self.collisions,
            "mover_hits": self.mover_hits,
            "projectile_hits": self.projectile_hits,
            "events": list(self.events),
            "hit_events": list(self.hit_events),
            "encounters": (
                (self.enc.log + [self.enc.record(e) for e in self.enc.active]) if self.enc is not None else []
            ),
            "max_contact_speed": self.max_contact_speed,
            "max_ground_speed": self.max_ground_speed,
            "battery_pct": self.battery,
        }
