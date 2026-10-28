"""Stage F mission environment."""

import math
import numpy as np
import mujoco
import city3d as C
import drone_env
from drone_env import DroneEnv

RHO, G = 1.225, 9.81


def quat_from_axes(fwd, up):
    """MuJoCo camera frame: -z forward, +y up. Returns world quaternion (w, x, y, z)."""
    f = np.asarray(fwd, float)
    f /= np.linalg.norm(f)
    u = np.asarray(up, float)
    r = np.cross(f, u)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    Rm = np.stack([r, u, -f], 1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rm.ravel())
    return q


class MissionEnv(DroneEnv):
    def __init__(self, place, seed, mission, objs=None, gimbal=(1280, 960), fovy=70.0, **kw):
        """mission: a missions.make() dict (objects already in it) or None; objs: extra objects."""
        self.mission_def = mission
        self.objs = list(mission["objects"] if mission else []) + list(objs or [])
        W, static, assets = C.build(place, seed)
        W.objects = self.objs
        from world import Target  # the base env expects targets (old API): the first mission object

        W.targets = [Target(o.oid, o.kind, o.facts.get("colour", ""), (o.x, o.y), "") for o in self.objs[:1]]
        xml = C.to_mjcf(W, static + "".join(o.xml for o in self.objs), assets, cam=gimbal)
        kw.setdefault(
            "battery_cutoff", True
        )  # an empty battery stops the motors (stage F: real consequences)
        super().__init__(seed=seed, prebuilt=(W, xml, C.texture_files()), **kw)
        self.lio = True
        self.acc_rw = 5e-4 * math.sqrt(
            self.dt
        )  # stage F sensors: lidar-inertial odometry, a MEMS-like accelerometer
        m = self.model
        self.gw, self.gh, self.fovy = gimbal[0], gimbal[1], fovy
        self.gcam = m.camera("gimbal").id
        m.cam_fovy[self.gcam] = fovy
        self.grend = None
        self.gimbal_yaw, self.gimbal_pitch = 0.0, -30.0
        self.cargo_id = m.body("cargo").mocapid[0]
        self.hook_id = m.body("hook").mocapid[0]
        self.tether_id = m.body("tether").mocapid[0]
        self.chute_g, self.line_g = m.geom("cargo_chute").id, m.geom("cargo_line").id
        self.parcels = {o.oid: m.body(f"{o.oid}_parcel").mocapid[0] for o in self.objs if o.kind == "pickup"}
        # landing surfaces: contacts with them below 1.2 m/s are touchdowns
        self.ground_id = m.geom("ground").id
        self.landable = {
            g
            for g in range(m.ngeom)
            if (m.geom(g).name or "").split("_")[0] in {o.oid for o in self.objs if o.land is not None}
        }
        self.base_mass = float(m.body_mass[self.bid])
        self.rng_w = np.random.default_rng(seed + 777)
        self.reset_mission()

    def reset(self):
        o = super().reset()
        if hasattr(self, "cargo_id"):
            self.reset_mission()
        return o

    def fork(self):
        import copy

        new = super().fork()
        for k in ("cargo", "hook", "touchdowns", "events_m"):
            new.__dict__[k] = copy.deepcopy(self.__dict__[k])
        new.grend = None
        return new

    # ---------------- world-side mission state
    def reset_mission(self):
        r = self.rng_w
        self.cargo = {
            "state": "stowed",
            "mass": float(r.uniform(0.3, 1.0)),
            "cda": float(r.uniform(0.9, 1.4)),
            "open_after": float(r.uniform(0.3, 0.6)),
        }  # static-line canopy
        self.hook = {"L": 0.0, "L_cmd": 0.0, "pos": None, "vel": np.zeros(3), "carry": None}
        self.touchdowns = []
        self.events_m = []
        self.carried_mass = self.cargo["mass"]
        self._set_mass()

    def _set_mass(self):
        self.model.body_mass[self.bid] = (
            self.base_mass + self.carried_mass
        )  # the autopilot keeps its nominal model; its integrator absorbs this

    def release_cargo(self):
        if self.cargo["state"] != "stowed":
            return False
        p = self.data.qpos[:3].copy() + np.array([0, 0, -0.25])
        v = self.data.qvel[:3].copy()
        self.cargo.update(state="falling", p=p, v=v, t0=self.t, prev=p.copy())
        self.carried_mass -= self.cargo["mass"]
        self._set_mass()
        self.events_m.append(
            {"t": round(self.t, 2), "kind": "release", "pos": [round(float(x), 2) for x in p]}
        )
        return True

    def observe(self, images=None, lidar=True):
        """Sensor self-filter for the sling load."""
        o = super().observe(images, lidar)
        h = getattr(self, "hook", None)
        if h is None:
            return o  # during the base env's first reset
        if (h["carry"] or h["L"] > 0.05) and lidar:
            P = o.get("lidar_points")
            if P is not None and len(P):
                deep = h["L"] + (0.9 if h["carry"] else 0.3)
                keep = ~((np.hypot(P[:, 0], P[:, 1]) < 0.8) & (P[:, 2] < 0.1) & (P[:, 2] > -deep))
                o["lidar_points"] = P[keep]
            if h["carry"]:
                o["range_down"] = self._range_below_load()
        return o

    def _range_below_load(self):
        """Down range measured past the hanging parcel (the ray starts under it)."""
        import mujoco

        off = self.hook["L"] + 0.12 + 0.32 + 0.6
        p = self.data.qpos[:3] - np.array([0.0, 0.0, off])
        gid = np.zeros(1, np.int32)
        d = mujoco.mj_ray(
            self.model,
            self.data,
            p,
            np.array([0.0, 0.0, -1.0]),
            None,
            1,
            self.model.body(f"{self.hook['carry']}_parcel").id,
            gid,
        )
        return float(d + off - 0.04) if d >= 0 else -1.0

    def hook_load(self):
        """Load cell on the winch: True once something hangs on the hook (what a real winch senses)."""
        return self.hook["carry"] is not None

    def winch(self, length):
        """Command the tether length (0-8 m); the drum runs at 0.6 m/s."""
        self.hook["L_cmd"] = float(np.clip(length, 0.0, 8.0))

    def _move(self, dt=0.0):
        super()._move(dt)
        if dt <= 0:
            return
        d = self.data
        wind = self.wind_mean + getattr(self, "gust_v", np.zeros(3))
        c = self.cargo
        if c["state"] == "falling":  # free fall, then the canopy opens; quadratic drag relative to the wind
            tt = self.t - c["t0"]
            cda = c["cda"] if tt > c["open_after"] else 0.02
            vrel = c["v"] - wind
            a = np.array([0, 0, -G]) - 0.5 * RHO * cda * np.linalg.norm(vrel) * vrel / c["mass"]
            c["v"] = c["v"] + a * dt
            c["prev"] = c["p"].copy()
            c["p"] = c["p"] + c["v"] * dt
            hit = self._segment_hit(c["prev"], c["p"])
            if hit is not None or c["p"][2] <= 0.14:
                c["state"] = "landed"
                c["p"] = hit if hit is not None else np.array([c["p"][0], c["p"][1], 0.14])
                c["landed_on"] = "ground" if hit is None else "structure"
                c["impact"] = float(np.linalg.norm(c["v"]))
                self.events_m.append(
                    {
                        "t": round(self.t, 2),
                        "kind": "cargo_landed",
                        "pos": [round(float(x), 2) for x in c["p"]],
                        "on": c["landed_on"],
                        "speed": round(c["impact"], 2),
                    }
                )
            self.model.geom_rgba[self.chute_g][3] = 0.95 if tt > c["open_after"] else 0.0
            self.model.geom_rgba[self.line_g][3] = 0.9 if tt > c["open_after"] else 0.0
        pos = c["p"] if c["state"] in ("falling", "landed") else np.array([0, 0, -60.0])
        d.mocap_pos[self.cargo_id] = pos
        h = self.hook
        step = 0.6 * dt
        h["L"] += float(np.clip(h["L_cmd"] - h["L"], -step, step))
        top = d.qpos[:3] + np.array([0, 0, -0.12])
        if h["L"] > 0.05:
            want = top - np.array([0, 0, h["L"]])
            if h["pos"] is None:
                h["pos"] = want.copy()
            k = 6.0 / max(h["L"], 0.5)
            h["vel"] = (
                h["vel"] * (1 - 2.5 * dt) + (want - h["pos"]) * k * dt + 0.02 * (wind - h["vel"]) * dt
            )  # a damped pendulum toward the point under the drone
            h["pos"] = h["pos"] + h["vel"] * dt
            off = h["pos"] - top
            n = np.linalg.norm(off)
            if n > h["L"]:
                h["pos"] = top + off / n * h["L"]
            if h["carry"] is None:
                for oid, mid in self.parcels.items():
                    o = next(o for o in self.objs if o.oid == oid)
                    hx, hy, hz = o.hook
                    if (
                        math.hypot(h["pos"][0] - hx, h["pos"][1] - hy) < 0.3
                        and abs(h["pos"][2] - hz) < 0.25
                        and np.linalg.norm(d.qvel[:3]) < 0.6
                        and np.linalg.norm(h["vel"]) < 0.5
                    ):
                        h["carry"] = oid
                        self.carried_mass += float(
                            np.clip(0.3 + 2.5 * (o.facts.get("parcel_m", 0.3) - 0.25), 0.3, 0.8)
                        )
                        self._set_mass()
                        self.events_m.append({"t": round(self.t, 2), "kind": "latched", "parcel": oid})
                        pb = self.model.body(f"{oid}_parcel").id  # a carried parcel hangs on the hook
                        for g in range(self.model.ngeom):
                            if self.model.geom_bodyid[g] == pb:
                                self.model.geom_contype[g] = 0
                                self.model.geom_conaffinity[g] = 0
                        break
            d.mocap_pos[self.hook_id] = h["pos"]
            mid_p = (top + h["pos"]) / 2
            d.mocap_pos[self.tether_id] = h["pos"]
            dirv = top - h["pos"]
            L = np.linalg.norm(dirv)
            self.model.geom_size[self.model.geom("tether_g").id][1] = L / 2
            self.model.geom_pos[self.model.geom("tether_g").id] = [0, 0, L / 2]
            d.mocap_quat[self.tether_id] = _quat_z_to(dirv)
            if h["carry"]:
                d.mocap_pos[self.parcels[h["carry"]]] = h["pos"] - np.array([0, 0, 0.3])
        else:
            h["pos"] = None
            d.mocap_pos[self.hook_id] = (0, 0, -60)
            d.mocap_pos[self.tether_id] = (0, 0, -60)
            if h["carry"]:
                d.mocap_pos[self.parcels[h["carry"]]] = top - np.array(
                    [0, 0, 0.32]
                )  # reeled in: the parcel under the body

    def _segment_hit(self, a, b):
        """First static-world contact of the cargo along a -> b (ray against collidable geoms), else None."""
        v = b - a
        L = float(np.linalg.norm(v))
        if L < 1e-6:
            return None
        gid = np.zeros(1, np.int32)
        grp = np.array([1, 1, 1, 1, 0, 0], np.uint8)
        grp[2] = 0  # ignore movers (group 2) and the drone
        dist = mujoco.mj_ray(self.model, self.data, a, v / L, grp, 1, self.bid, gid)
        if 0 <= dist <= L and gid[0] != self.model.geom("ground").id:
            return a + v / L * dist
        return None

    def _contacts(self):
        d = self.data
        soft_backup = self.soft_geoms
        speed = float(np.linalg.norm(d.qvel[:3]))
        if speed < 1.2:
            for i in range(d.ncon):
                c = d.contact[i]
                g = {c.geom1, c.geom2}
                if g & self.drone_geoms and self.ground_id in g:
                    self.touchdowns.append(
                        (
                            round(self.t, 2),
                            self.ground_id,
                            round(speed, 2),
                            *[round(float(v), 2) for v in d.qpos[:3]],
                        )
                    )  # set down on the ground / a hangar floor
                if g & self.drone_geoms and (g - self.drone_geoms) & self.landable:
                    self.touchdowns.append(
                        (
                            round(self.t, 2),
                            int(next(iter((g - self.drone_geoms) & self.landable))),
                            round(speed, 2),
                            *[round(float(v), 2) for v in d.qpos[:3]],
                        )
                    )  # where the drone was
            self.soft_geoms = soft_backup | self.landable
        super()._contacts()
        self.soft_geoms = soft_backup

    # ---------------- gimbal camera (brain-side sensor)
    def set_gimbal(self, yaw_rel_deg, pitch_deg):
        self.gimbal_yaw, self.gimbal_pitch = float(yaw_rel_deg), float(np.clip(pitch_deg, -90, 20))

    def gimbal_frame(self, depth=True):
        """RGB (and planar depth, as from the lidar / a depth camera) from the stabilised gimbal camera + its."""
        m, d = self.model, self.data
        if self.grend is None:
            self.grend = mujoco.Renderer(m, self.gh, self.gw, max_geom=30000)
        yaw = self.obs_yaw() + math.radians(self.gimbal_yaw)
        p = math.radians(self.gimbal_pitch)
        fwd = np.array([math.cos(p) * math.cos(yaw), math.cos(p) * math.sin(yaw), math.sin(p)])
        up = np.array([0, 0, 1.0]) if abs(p) < 1.5 else np.array([math.cos(yaw), math.sin(yaw), 0.0])
        qw = quat_from_axes(fwd, up)
        qb = d.xquat[self.bid].copy()
        qi = np.zeros(4)
        mujoco.mju_negQuat(qi, qb)
        qr = np.zeros(4)
        mujoco.mju_mulQuat(qr, qi, qw)
        m.cam_quat[self.gcam] = qr
        mujoco.mj_forward(m, d)
        self.grend.update_scene(d, camera="gimbal", scene_option=self.vopt)
        rgb = self.grend.render().copy()
        dep = None
        if depth:
            self.grend.enable_depth_rendering()
            self.grend.update_scene(d, camera="gimbal", scene_option=self.vopt)
            dep = self.grend.render().copy()
            self.grend.disable_depth_rendering()
        gc = self.grend.scene.camera[0]
        return rgb, dep, (np.array(gc.pos, float), np.array(gc.forward, float), np.array(gc.up, float))

    def obs_yaw(self):
        """Heading from the autopilot's attitude estimate (what the brain also sees)."""
        q = self.data.xquat[self.bid]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def close(self):
        if self.grend is not None:
            self.grend.close()
            self.grend = None
        super().close()

    # ---------------- world-side scoring
    def result(self):
        """Success of each mission step decided by the world (MISSIONS_DESIGN sec. 4)."""
        out = {
            "steps": [],
            "collisions": len([h for h in self.hit_events if h["kind"] in ("static", "mover")]),
            "bird_hits": sum(1 for h in self.hit_events if h["kind"] == "mover"),
            "encounter_hits": sum(1 for h in self.hit_events if h["kind"] == "encounter"),
            "events": self.events_m,
            "battery": round(self.battery, 1),
        }
        if not self.mission_def:
            return out
        p = self.data.qpos[:3]
        for st in self.mission_def["steps"]:
            o = next(o for o in self.objs if o.oid == st["target"])
            a = st["action"]
            r = {"action": a, "target": o.oid}
            if a == "land" or a == "enter":
                # judged AT the touchdown
                lx, ly, lz, lr = o.land
                if a == "enter":  # inside the hangar and set down on its floor
                    td = [
                        t
                        for t in self.touchdowns
                        if t[2] < 1.0
                        and math.hypot(t[3] - o.x, t[4] - o.y) < o.facts["size_m"] / 2 - 0.3
                        and t[5] < o.facts["height_m"]
                        and (
                            (self.model.geom(t[1]).name or "").startswith(o.oid + "_")
                            or t[1] == self.ground_id
                        )
                    ]
                else:
                    td = [
                        t
                        for t in self.touchdowns
                        if (self.model.geom(t[1]).name or "").startswith(o.oid + "_")
                        and t[2] < 1.0
                        and math.hypot(t[3] - lx, t[4] - ly) < lr + 0.3
                    ]
                r["ok"] = bool(td)
                r["touch_speed"] = min(t[2] for t in td) if td else None
                if a == "enter":
                    r["inside"] = bool(td) and any(
                        math.hypot(t[3] - o.x, t[4] - o.y) < o.facts["size_m"] / 2 - 0.3
                        and t[5] < o.facts["height_m"]
                        for t in td
                    )
                    r["ok"] = r["ok"] and r["inside"]
            elif a == "drop":
                c = self.cargo
                lx, ly = o.x, o.y
                R = o.facts["size_m"] / 2 * 0.9
                r["ok"] = (
                    c["state"] == "landed"
                    and c.get("landed_on") == "ground"
                    and math.hypot(c["p"][0] - lx, c["p"][1] - ly) < R
                )
                r["miss_m"] = (
                    round(math.hypot(c["p"][0] - lx, c["p"][1] - ly), 2) if c["state"] == "landed" else None
                )
            elif a == "pickup":
                r["ok"] = self.hook["carry"] == o.oid
            else:
                r["ok"] = None  # inspect / flyby: scored from the flight log by the caller
            out["steps"].append(r)
        out["home_dist"] = round(math.hypot(p[0] - self.home[0], p[1] - self.home[1]), 2)
        return out


def _quat_z_to(v):
    """Quaternion rotating +z onto v."""
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([1.0, 0, 0, 0])
    v = v / n
    z = np.array([0, 0, 1.0])
    ax = np.cross(z, v)
    s = np.linalg.norm(ax)
    c = float(z @ v)
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1, 0, 0])
    ax /= s
    a = math.atan2(s, c)
    return np.array([math.cos(a / 2), *(ax * math.sin(a / 2))])
