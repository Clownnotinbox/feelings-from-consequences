"""Stage F mission brain."""

import math, os
from collections import deque
import numpy as np
import grounding as G

CRUISE_V, SEARCH_ALT, CLOSE_ALT, CLOSE_R = 8.0, 35.0, 20.0, 25.0
MOVING = ("transit", "reposition", "closer", "approach", "return", "pass", "takeoff")
TOUCH = ("land", "drop", "pickup", "enter")  # actions that touch or put something on the world
DEAD_RECKON = (
    "exit_out",
    "verify",
    "descend",
    "final",
    "hover_pick",
    "hook",
    "drop_hold",
    "enter_in",
    "find_door",
)  # near the target: smooth own motion, target from fresh views
DOOR_OLD = os.environ.get("DOOR_OLD", "1") == "1"  # the door finder of 27.10
PRECISE_XY = (
    "verify",
    "descend",
    "final",
    "hover_pick",
    "hook",
    "drop_hold",
)  # on a real autopilot these hold a horizontal POSITION
DROP_H = 10.0  # release height over the drop zone, m
ALT_MAX = 120.0  # with robust=True the brain never climbs higher (the legal limit in many countries)


def clean_lidar(P, cell=1.0):
    """Drop lone lidar points (sun glare, crosstalk): a real surface leaves 2 or more points in a 1 m cell."""
    P = np.asarray(P)
    if len(P) < 2:
        return P
    q = np.floor(P / cell).astype(np.int64) + 1000  # the lidar sees 40 m, so cells fit in 0..2047
    key = (q[:, 0] * 2048 + q[:, 1]) * 2048 + q[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return P[cnt[inv] >= 2]


class Roofs:
    """The MAP's buildings for route checks (the drone's own map: may be wrong; the lidar covers the rest)."""

    def __init__(self, MAP):
        from shapely.strtree import STRtree

        self.B = MAP["buildings"]
        self.tree = STRtree([b["poly"] for b in self.B]) if self.B else None

    def max_h(self, geom):
        if self.tree is None:
            return 0.0
        return max(
            [self.B[i]["h"] for i in self.tree.query(geom) if self.B[i]["poly"].intersects(geom)] + [0.0]
        )

    def along(self, a, b, buf=10.0):
        from shapely.geometry import LineString, Point

        g = (
            LineString([tuple(a[:2]), tuple(b[:2])])
            if math.hypot(a[0] - b[0], a[1] - b[1]) > 0.1
            else Point(a[0], a[1])
        )
        return self.max_h(g.buffer(buf))

    def near(self, p, r):
        from shapely.geometry import Point

        return self.max_h(Point(p[0], p[1]).buffer(r))


def heights_along(MAP, a, b, step=5.0):
    """Max mapped roof height near the straight line a -> b (10 m corridor)."""
    return Roofs(MAP).along(a, b)


class Brain:
    def __init__(
        self,
        env,
        MAP,
        parsed,
        rough,
        perceiver,
        safety=None,
        ask=None,
        log=None,
        nav=None,
        return_default=True,
        slow="feelings",
        robust=False,
    ):
        self.env, self.MAP, self.perc, self.safety, self.ask = env, MAP, perceiver, safety, ask
        # robust: fixes found in the app on 28.10 after the paper's tests (the tests ran without them)
        self.robust = robust
        self.logf = log or (lambda *a: None)
        self.nav = nav
        self.si = 0
        self.state = "takeoff"
        self.t_state = 0.0
        self.wp = None
        self.dets = []
        self.looks = 0
        self.queue = []
        self.target = None
        self.notes = []
        self.done = False
        self.cmd_hold = None
        steps = [G.sanitize(s) for s in (parsed.get("steps") or []) if isinstance(s, dict)]
        keep = [
            k
            for k, s in enumerate(steps)
            if s["action"] in ("land", "drop", "pickup", "enter", "inspect", "flyby")
            and s["target"].get("kind") in G.KINDS
        ]
        if len(keep) < len(steps):
            self.log(f"cannot act on {len(steps) - len(keep)} step(s) of the parsed mission: skipped")
        self.parsed = {**parsed, "steps": [steps[k] for k in keep]}
        self.rough = [rough[k] if k < len(rough) else None for k in keep]
        ret = parsed.get("return_home")
        self.return_home = return_default if not isinstance(ret, bool) else ret
        self.home = np.array(env.home[:2], float)
        self.pos = np.array([*self.home, 0.0])
        self.alt_cruise = 30.0
        self.feel = {}
        self.track = []
        self.slow = slow
        self.lost_t = 0.0
        self.resume = None  # slow: 'feelings' (core) | 'telemetry'
        self.roofs = Roofs(MAP)
        self.alt_bias = 0.0
        self._route = (None, 0.0)
        self._goal = None
        self._best_d = 1e9
        self._t_prog = 0.0
        self._t_lidar_log = -9.0
        self._o = None
        self._bat = deque(maxlen=400)
        self.asks = 0
        self.skipped = []
        self.rejected = []

    # ---------------- helpers
    def log(self, msg):
        self.notes.append((round(self.env.t, 1), msg))
        self.logf(f"[{self.env.t:6.1f}s] {msg}")

    def go(self, state):
        self.log(f"-> {state}")
        self.state = state
        self.t_state = self.env.t
        self.alt_bias = 0.0
        self._best_d = 1e9
        self._t_prog = self.env.t
        self._appr = None
        self._still = 0.0
        self._drop_locked = False

    def skip_step(self, why):
        self.log(why)
        self.skipped.append(self.si)
        self.si += 1
        self.target = None
        self.dets = []
        self.go(
            "transit" if self.si < len(self.parsed["steps"]) else ("return" if self.return_home else "done")
        )

    def fly_to(self, goal, v=CRUISE_V, tol=0.6):
        """Velocity command toward goal (x, y, z) from the ESTIMATED position; returns (cmd, arrived)."""
        d = np.array(goal, float) - self.pos
        n = float(np.linalg.norm(d[:2]))
        sp = min(v, max(0.25, 0.55 * n))
        vx, vy = (d[:2] / n * sp) if n > 0.05 else (0.0, 0.0)
        vz = float(np.clip(0.9 * d[2], -2.5, 3.0))
        yaw_err = (
            (math.atan2(d[1], d[0]) - self.env.obs_yaw() + math.pi) % (2 * math.pi) - math.pi
            if n > 3
            else 0.0
        )
        return (
            {
                "vx": float(vx),
                "vy": float(vy),
                "vz": vz,
                "yaw_rate": float(np.clip(yaw_err, -0.8, 0.8)),
                "dxy": (float(d[0]), float(d[1])),
                "precise": self.state in PRECISE_XY,
            },
            n < tol and abs(d[2]) < 0.6,
        )  # dxy / precise: for a real autopilot

    def blocked_ahead(self, dxy, dist=9.0, half_w=2.5, dz=2.0):
        """Lidar points in the corridor ahead at the drone's height (anything: the map is not asked)."""
        o = self._o
        if o is None or not len(o.get("lidar_points", ())):
            return False
        from nav import rot

        a = o["attitude"]
        P = np.asarray(o["lidar_points"]) @ rot(a["roll"], a["pitch"], a["yaw"]).T
        u = np.asarray(dxy, float) / max(np.linalg.norm(dxy), 1e-6)
        along = P[:, :2] @ u
        lat = np.abs(P[:, 0] * u[1] - P[:, 1] * u[0])
        return int(np.sum((along > 0.8) & (along < dist) & (lat < half_w) & (np.abs(P[:, 2]) < dz))) >= 4

    def goto(self, goal, v=CRUISE_V, tol=0.6):
        """Route to goal: straight if the MAP says the line is clear, else climb, cross above the roofs."""
        g = np.array(goal, float)
        dh = math.hypot(g[0] - self.pos[0], g[1] - self.pos[1])
        self._goal = g
        if dh <= 4.0:
            return self.fly_to(g, v, tol)
        key = (round(self.pos[0] / 4), round(self.pos[1] / 4), round(g[0]), round(g[1]))
        if self._route[0] != key:
            self._route = (key, self.roofs.along(self.pos, g) + 8.0)
        need = self._route[1] + self.alt_bias
        if need <= min(self.pos[2], g[2]) + 0.5:
            cmd, ok = self.fly_to(g, v, tol)
        else:
            z_r = min(max(g[2], need), 120.0)
            if self.pos[2] < z_r - 2.0:
                cmd, ok = self.fly_to([*self.pos[:2], z_r], v)
                cmd["vx"] = cmd["vy"] = 0.0
                ok = False  # climb in place first
            else:
                cmd, ok = self.fly_to([g[0], g[1], z_r], v)
                ok = False
        at_ceiling = self.robust and self.pos[2] > ALT_MAX - 3.0  # no more climbing: the core still guards
        blocked = math.hypot(cmd["vx"], cmd["vy"]) > 0.5 and self.blocked_ahead((cmd["vx"], cmd["vy"]))
        if self.robust:  # a real obstacle stays in view; noise does not: 3 scans in a row
            self._blk = getattr(self, "_blk", 0) + 1 if blocked else 0
            blocked = self._blk >= 3
        if blocked and not at_ceiling:
            if self.env.t - self._t_lidar_log > 5.0:
                self.log("lidar: something ahead the map does not show -> climbing")
                self._t_lidar_log = self.env.t
            cmd = {**cmd, "vx": 0.0, "vy": 0.0, "vz": 2.0}
            self.alt_bias = max(self.alt_bias, self.pos[2] + 4.0 - self._route[1])
        if self.robust:
            self.alt_bias = min(self.alt_bias, max(0.0, ALT_MAX - self._route[1]))
        return cmd, ok

    def free_alt(self, p, r=4.0):
        """Lowest safe height over point p by the MAP (roof + 4 m)."""
        return self.roofs.near(p, r) + 4.0

    def search_alt(self, p):
        base = (
            SEARCH_ALT if self.nav is None or self.nav.gps_trust > 0.3 else 25.0
        )  # without GPS: lower, where the lidar and the flow camera see the ground
        return max(base, self.roofs.near(p, 30.0) + 8.0)

    def look(self, yaw_rel, pitch):
        """One gimbal frame -> detections in MAP coordinates."""
        self.env.set_gimbal(yaw_rel, pitch)
        rgb, depth, cam = self.env.gimbal_frame()
        dets = self.perc(rgb, depth, cam, self.env.fovy)
        shift = self.pos - self.env.data.qpos[:3]  # the frame's own position is the TRUE camera
        for d in dets:
            d["x"] += shift[0]
            d["y"] += shift[1]
            d["t"] = self.env.t
        self.dets += [d for d in dets if d["cls"] != "bird"]
        self._t_prog = self.env.t
        return dets  # stopping to look is not 'no progress'

    def sweep(self):
        for k in range(8):
            for p in (-30.0, -55.0):
                self.look(45.0 * k, p)
        self.look(0.0, -89.0)

    def ranked(self):
        step = self.parsed["steps"][self.si]
        hyps = [h for h in G.fuse(self.dets) if not self._rejected(h)]
        r, pn = G.rank(hyps, step, self.rough[self.si], self.MAP, G.KINDS)
        return r, pn

    def _rejected(self, h):
        return any(
            q["cls"] == h["cls"] and math.hypot(q["x"] - h["x"], q["y"] - h["y"]) < 3.0 + 0.5 * h["size"]
            for q in self.rejected
        )

    # ---------------- main loop
    def act(self, o):
        dt = self.env.agent_dt
        if self.robust and o.get("lidar_points") is not None and len(o["lidar_points"]):
            o = {
                **o,
                "lidar_points": clean_lidar(o["lidar_points"]),
            }  # navigation and the core get clean points
        self._o = o
        self._bat.append((o["t"], o["battery_pct"]))
        self.vel = np.array(o["vel_est"], float)
        w = np.array(o["wind"]["vec"][:2], float)
        self._last_wind = (
            w if not hasattr(self, "_last_wind") else 0.95 * np.asarray(self._last_wind) + 0.05 * w
        )  # anemometer, smoothed (~1 s)
        self._mean_wind = (
            w
            if not hasattr(self, "_mean_wind")
            else (1 - dt / 10.0) * np.asarray(self._mean_wind) + dt / 10.0 * w
        )  # ~10 s mean: gusts average out over a fall
        if self.nav is not None:
            p_nav = self.nav.step(o, dt)
        else:
            p_nav = np.array([o["gps"]["x"], o["gps"]["y"], o["baro_alt"]]) if o.get("gps") else self.pos
        # near the target the map match may jump by metres
        # own velocity estimate, and the target is re-seen relative to it
        self.pos = (
            np.array([self.pos[0] + self.vel[0] * dt, self.pos[1] + self.vel[1] * dt, p_nav[2]])
            if self.state in DEAD_RECKON
            else p_nav
        )
        self.track.append((round(o["t"], 1), *[round(float(v), 2) for v in self.pos]))
        self.range_down = o["range_down"] if o["range_down"] is not None and o["range_down"] >= 0 else 99.0
        cmd = self._slow_layer(o) or self._state_cmd(o)
        self._progress()
        if self.safety is not None and self.state not in ("done", "landed"):
            new, fired = self.safety(o, cmd)  # fed every step, so its trackers and feelings stay current
            if self.state in (
                "final",
                "enter_in",
                "exit_out",
                "hook",
            ):  # an intended touch: only a MOVING threat may override
                sm = getattr(self.safety, "last_sm", None) or {"tracks": []}
                if fired and any(t.get("threat") and t.get("t_cpa", 9.0) < 1.5 for t in sm["tracks"]):
                    cmd = new
                    self.log("safety: dodging a moving threat near the target")
            else:
                cmd = new
                if fired:
                    self.log("safety: avoiding")
        return cmd

    def _progress(self):
        """No progress toward the current goal for 12 s while moving -> go higher."""
        if self.state not in MOVING or self._goal is None:
            return
        d = float(
            math.hypot(*(self._goal[:2] - self.pos[:2]))
        )  # horizontal: climbing to the route height is progress too
        if d < self._best_d - 1.0 or abs(self.pos[2] - getattr(self, "_z_prog", self.pos[2])) > 1.0:
            self._best_d = min(d, self._best_d)
            self._t_prog = self.env.t
            self._z_prog = self.pos[2]
        elif self.env.t - self._t_prog > 12.0:
            self.alt_bias = min(self.alt_bias + 8.0, 40.0)
            self._t_prog = self.env.t
            self._best_d = d
            self.log(f"no progress: routing {self.alt_bias:.0f} m higher")

    PRECISION = (
        "exit_out",
        "verify",
        "descend",
        "final",
        "hover_pick",
        "hook",
        "find_door",
        "enter_in",
        "drop_hold",
        "landed",
        "done",
        "relocalize",
        "return",
        "home_land",
    )

    def drain_rate(self):
        """Battery %/s measured over the last ~30 s (telemetry); before that a nominal 0.1 %/s."""
        if len(self._bat) < 2 or self._bat[-1][0] - self._bat[0][0] < 20.0:
            return 0.1
        (t0, b0), (t1, b1) = self._bat[0], self._bat[-1]
        return max((b0 - b1) / (t1 - t0), 0.02)

    def _slow_layer(self, o):
        """Slow decisions from FEELINGS."""
        if self.slow == "none" or self.state in ("landed", "done"):
            return None
        F = getattr(self.safety, "feel", {}) if self.safety is not None else {}
        if self.slow.startswith("feelings"):  # 'feelings_nolost' / 'feelings_noenergy'
            lost, low = (
                F.get("lost", 0.0) > 0.7 and self.slow != "feelings_nolost",
                F.get("low_energy", 0.0) > 0.8 and self.slow != "feelings_noenergy",
            )
        else:
            sp = self.nav.spread if self.nav is not None else 0.0
            trust = self.nav.gps_trust if self.nav is not None else 1.0
            d_home = math.hypot(*(self.home - self.pos[:2]))
            need = self.drain_rate() * (
                d_home / CRUISE_V + 60.0
            )  # % to fly home + 60 s reserve, at the measured drain
            lost, low = sp > 8.0 and trust < 0.4, o["battery_pct"] < need + 5.0
        if low and self.state not in self.PRECISION:
            self.log(
                "feeling: not enough energy -> going home"
                if self.slow.startswith("feelings")
                else "telemetry: battery low -> going home"
            )
            self.skipped += list(range(self.si, len(self.parsed["steps"])))
            self.si = len(self.parsed["steps"])
            self.go("return")
            return None
        self.lost_t = self.lost_t + self.env.agent_dt if lost else 0.0
        if self.lost_t > 3.0 and self.state not in self.PRECISION and getattr(self, "relocs", 0) < 2:
            self.relocs = getattr(self, "relocs", 0) + 1
            self.resume = self.state
            self.go("relocalize")
            self.log(
                (
                    "feeling: I am lost"
                    if self.slow.startswith("feelings")
                    else "telemetry: position uncertain"
                )
                + (
                    " -> going down to 25 m to see the ground and walls"
                    if self.relocs < 2
                    else " -> last try; then on by odometry"
                )
            )
        if self.state == "relocalize":
            cmd, ok = self.fly_to(
                [*self.pos[:2], max(25.0, self.free_alt(self.pos, 15.0) + 4.0)], v=1.0
            )  # down to where the lidar and the flow camera see structure
            if (not lost and self.env.t - self.t_state > 6.0) or self.env.t - self.t_state > 25.0:
                st = self.resume or "transit"
                self.go(st)
                self.lost_t = 0.0
            return cmd
        return None

    def zone_clear(self, h):
        """Is the drop zone free of anything standing on it? Depth from the gimbal straight down, no."""
        rgb, depth, cam = self.env.gimbal_frame()
        pos, fwd, up = cam
        H, W = depth.shape
        f = (H / 2) / math.tan(math.radians(self.env.fovy) / 2)
        right = np.cross(fwd, up)
        ys, xs = np.mgrid[0:H:4, 0:W:4]
        d = depth[ys, xs]
        rays = fwd[None, None] + ((xs - W / 2) / f)[..., None] * right - ((ys - H / 2) / f)[..., None] * up
        P = pos + d[..., None] * rays
        shift = self.pos - self.env.data.qpos[:3]
        P[..., :2] += shift[:2]
        r = np.hypot(P[..., 0] - h["x"], P[..., 1] - h["y"])
        inner = r < 0.8 * 0.5 * h["size"]
        if inner.sum() < 30:
            return True
        z = P[..., 2][inner]
        ground = np.median(z)
        return float(np.mean(z > ground + 0.5)) < 0.01

    def _state_cmd(self, o):
        env = self.env
        st = self.state
        stop = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
        T = env.t - self.t_state
        if self.si >= len(self.parsed["steps"]) and st not in (
            "return",
            "home_land",
            "final",
            "done",
            "landed",
            "exit_out",
        ):
            self.go("return" if self.return_home else "done")
            return stop
        if st == "takeoff":  # straight up to a safe height over the take-off point, then transit
            z = max(self.free_alt(self.pos, 8.0) + 4.0, 15.0)
            cmd, ok = self.fly_to([*self.pos[:2], z])
            if self.pos[2] > z - 1.5:
                self.go("transit")
            return cmd
        if st == "transit":
            tgt = self.rough[self.si]
            if tgt is None:
                self.skip_step("no rough point for this step: skipping")
                return stop
            cmd, ok = self.goto([tgt[0], tgt[1], self.search_alt(tgt)])
            if ok or math.hypot(*(np.array(tgt) - self.pos[:2])) < 2.0:
                self.go("search")
                self.dets = []
                self.looks = 0
                self.searches = 0
            elif T > math.hypot(*(np.array(tgt) - self.home)) / 2.0 + 150.0:
                self.log("transit is taking too long: searching from here")
                self.go("search")
                self.dets = []
                self.looks = 0
                self.searches = 0
            return cmd
        if st == "search":
            if np.linalg.norm(self.vel) > 0.5 and T < 15.0:
                return stop  # settle before looking
            self.sweep()
            r, pn = self.ranked()
            dec, best = G.decide(r, pn)
            self.log(
                f"search: {len(r)} candidates, top {r[0][0]:.2f}" if r else "search: nothing of that kind"
            )
            ok, why = self._unique(best) if best is not None else (False, "")
            if dec == "go" and ok:
                self.target = best
                self.go("approach")
                return stop
            if dec == "go" and self.looks >= 2:
                return self._not_unique(r, why)
            if r and self.looks < 2:
                self.queue = [h for _, h, _ in r[:2]]
                self.go("closer")
                return stop
            if r:
                return self._ask_or_pick(r)
            self.searches = getattr(self, "searches", 0) + 1
            if self.searches <= 4:  # expanding search around the rough point
                a = 2.1 * self.searches
                R = 40.0 * self.searches
                rp = self.rough[self.si]
                self.rough_search = (rp[0] + R * math.cos(a), rp[1] + R * math.sin(a))
                self.go("reposition")
            else:
                self.skip_step("not found: skipping this step")
            return stop
        if st == "reposition":
            cmd, ok = self.goto([*self.rough_search, self.search_alt(self.rough_search)])
            if ok or T > 90.0:
                self.go("search")
            return cmd
        if st == "closer":
            if not self.queue:
                self.looks += 1
                r, pn = self.ranked()
                dec, best = G.decide(r, pn)
                self.log(f"after a closer look: top {r[0][0]:.2f}" if r else "lost the candidates")
                ok, why = self._unique(best) if best is not None else (False, "")
                if dec == "go" and ok:
                    self.target = best
                    self.go("approach")
                elif dec == "go" and self.looks >= 2:
                    return self._not_unique(r, why)
                elif r and self.looks < 2:
                    self.queue = [h for _, h, _ in r[:2]]
                    self.t_state = env.t
                elif r:
                    return self._ask_or_pick(r)
                else:
                    self.go("search")
                return stop
            h = self.queue[0]
            if getattr(self, "_closer_for", None) is not h:
                self._closer_for = h
                self._closer_goal = self._close_spot(h)
                self._closer_t = env.t
            cmd, ok = self.goto(self._closer_goal, v=5.0)
            if (ok and np.linalg.norm(self.vel) < 0.4) or env.t - self._closer_t > 60.0:
                if not ok:
                    self.log("could not reach the close-look point: looking from here")
                yaw_rel = math.degrees(math.atan2(h["y"] - self.pos[1], h["x"] - self.pos[0]) - env.obs_yaw())
                pitch = -math.degrees(
                    math.atan2(
                        self.pos[2] - h["z"], max(math.hypot(h["x"] - self.pos[0], h["y"] - self.pos[1]), 1.0)
                    )
                )
                for dy in (-12, 0, 12):
                    self.look(yaw_rel + dy, pitch)
                self.queue.pop(0)
            return cmd
        if st == "approach":
            return self._approach()
        if st == "verify":
            return self._verify()
        return self._action_states(st, stop)

    def _close_spot(self, h):
        """A point CLOSE_R from the candidate, toward the drone if free by the MAP, else the nearest free."""
        a0 = math.atan2(self.pos[1] - h["y"], self.pos[0] - h["x"])
        z = max(CLOSE_ALT, h["z"] + 12)
        for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6):
            a = a0 + k * math.pi / 6
            p = (h["x"] + CLOSE_R * math.cos(a), h["y"] + CLOSE_R * math.sin(a))
            if self.free_alt(p, 6.0) < z - 2.0:
                return [p[0], p[1], z]
        p = (h["x"] + CLOSE_R * math.cos(a0), h["y"] + CLOSE_R * math.sin(a0))
        return [p[0], p[1], self.free_alt(p, 6.0) + 4.0]

    def _ask_or_pick(self, r, force_ask=False):
        """Ambiguous after a closer look."""
        stop = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
        if self.ask is not None and self.env.link_up():
            self.asks += 1
            ans = self.ask(self.parsed["steps"][self.si], r[:3])
            if ans is not None and 0 <= ans < len(r[:3]):
                self.log(f"operator chose candidate {ans + 1}")
                self.target = r[ans][1]
                self.go("approach")
                return stop
            if ans == -1:  # the shown ones are not it
                self.log("operator: none of these")
                self.rejected += [dict(q) for _, q, _ in r[:3]]
                self.looks = 3
                self.searches = getattr(self, "searches", 0)
                self.dets = []
                self.go("search")
                return stop
            if force_ask:
                self.skip_step("no answer from the operator: not touching anything")
                return stop
        need = 0.8 if self.parsed["steps"][self.si]["action"] in TOUCH else 0.5
        ok, why = self._unique(r[0][1])
        if r[0][0] >= need and ok:
            self.log(f"no operator: taking the best candidate ({r[0][0]:.2f})")
            self.target = r[0][1]
            self.go("approach")
        elif r[0][0] >= need:
            self.skip_step(f"{why}: not touching anything without the operator")
        else:
            self.skip_step(f"too unsure without the operator ({r[0][0]:.2f}): skipping this step")
        return stop

    def _unique(self, best):
        """Alone, a TOUCHING action."""
        step = self.parsed["steps"][self.si]
        if step["action"] not in TOUCH:
            return True, ""
        unknown = [
            f"{k}={v}"
            for k, v in step["target"].items()
            if k in G.VALUES and not isinstance(v, bool) and v not in G.VALUES[k]
        ]
        if unknown:
            return False, f"I did not understand {', '.join(unknown)}"
        hyps = [q for q in G.fuse(self.dets) if not self._rejected(q)]
        st = {**step, "target": {k: v for k, v in step["target"].items() if k != "door_open"}}
        r, _ = G.rank(hyps, step, self.rough[self.si], self.MAP, G.KINDS)
        near = lambda q: math.hypot(q["x"] - best["x"], q["y"] - best["y"]) <= 3.0 + 0.5 * best["size"]
        p_best = max(
            [p for p, q, _ in r if near(q)] + [1e-9]
        )  # a real competitor fits the description AND scores at least 15 % of
        others = [
            q for p, q, _ in r if not near(q) and p >= 0.15 * p_best and G.fits_all(q, st, hyps, self.MAP)
        ]
        return (not others), (f"{len(others)} other object(s) fit the description too" if others else "")

    def _not_unique(self, r, why):
        """The description does not single the object out."""
        if self.ask is not None and self.env.link_up():
            self.log(f"{why}: asking the operator")
            return self._ask_or_pick(r, force_ask=True)
        self.skip_step(f"{why}: not touching anything without the operator")
        return {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}

    def _refine(self, pitch=-89.0, gate=None, alpha=0.6):
        """Re-detect the target straight below / ahead and move its position estimate (precision phase)."""
        h = self.target
        dets = self.look(
            (
                math.degrees(math.atan2(h["y"] - self.pos[1], h["x"] - self.pos[0]) - self.env.obs_yaw())
                if pitch > -80
                else 0.0
            ),
            pitch,
        )
        same = [
            d
            for d in dets
            if d["cls"] != "bird"
            and math.hypot(d["x"] - h["x"], d["y"] - h["y"]) < (gate or 3.0 + 0.5 * h["size"])
        ]  # close up the detector may name it differently
        if same:
            d = min(same, key=lambda d: math.hypot(d["x"] - h["x"], d["y"] - h["y"]))
            a = alpha
            h["x"] = (1 - a) * h["x"] + a * d["x"]
            h["y"] = (1 - a) * h["y"] + a * d["y"]
            h["z"] = (1 - a) * h["z"] + a * d["z"]
        return bool(same)

    def _verify(self):
        """Before touching (land / drop / pick up / enter)."""
        stop = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
        h = self.target
        env = self.env
        if getattr(self, "_verify_alt", None) is not None:  # the lower look: descend first
            cmd, ok = self.fly_to([h["x"], h["y"], max(self._verify_alt, h["z"] + 5.0)], v=1.5, tol=1.0)
            if not ok and env.t - self._verify_t0 < 20.0:
                return cmd
            self._verify_alt = None
        if np.linalg.norm(self.vel) > 0.4 and env.t - self.t_state < 10.0:
            return stop
        if env.t - self.t_state > 60.0:
            self.log("close check: taking too long -> not touching it")
            self.rejected.append(dict(h))
            self.target = None
            self.go("search")
            return stop
        new = []
        for pitch in (-89.0, -60.0):
            dets = self.look(
                (
                    0.0
                    if pitch < -80
                    else math.degrees(math.atan2(h["y"] - self.pos[1], h["x"] - self.pos[0]) - env.obs_yaw())
                ),
                pitch,
            )
            near = [
                d
                for d in dets
                if d["cls"] != "bird" and math.hypot(d["x"] - h["x"], d["y"] - h["y"]) < 3.0 + 0.5 * h["size"]
            ]  # close up the class may differ
            if near:
                new.append(min(near, key=lambda d: math.hypot(d["x"] - h["x"], d["y"] - h["y"])))
        step = dict(self.parsed["steps"][self.si])
        step["target"] = {k: v for k, v in step["target"].items() if k != "door_open"}
        if not new:
            if not h.get("lower_look"):  # look again from 6 m lower
                h["lower_look"] = True
                self.log("close check: not seen from above -> looking again from lower")
                self._verify_alt = self.pos[2] - 6.0
                self._verify_t0 = env.t
                self.t_state = env.t - 12.0
                return stop
            self.log("close check: still not seen -> not touching it, searching on")
            self.rejected.append(dict(h))
            self.target = None
            self.looks = 0
            self.go("search")
            return stop
        else:
            for d in new:  # close looks count double
                w = 2.0 * min(1.0, d["px"] / 60.0) * d["conf"]
                W = h["w"] + w
                for k, pr in d["attrs"].items():
                    h["attrs"][k] = (np.asarray(h["attrs"].get(k, pr)) * h["w"] + np.asarray(pr) * w) / W
                h["w"] = W
            fits = G.fits_all(
                h, step, [q for q in G.fuse(self.dets) if not self._rejected(q)] + [h], self.MAP
            )
        if fits and not h.get("scanned"):  # the neighbourhood: look-alikes stand 9-45 m away
            h["scanned"] = True
            self.sweep()
            ok, why = self._unique(h)
            if not ok:
                self.log(f"close check: it matches, but {why} around it")
                return self._not_unique(self.ranked()[0] or [(1.0, h, {})], why)
        if fits:
            h["verified"] = True
            self.log("close check: it matches the description and nothing around it does")
            self._verify_alt = None
            self.go("approach")
            return stop
        self.log("close check: this one does not match the description -> a look-alike, searching on")
        self.rejected.append(dict(h))
        self.target = None
        self.looks = 0
        self.go("search")
        return stop

    def _plan_flyby(self, h, alt):
        """A straight pass 8 m from the target's edge at the asked height, along a line the MAP says is clear."""
        a0 = math.atan2(self.pos[1] - h["y"], self.pos[0] - h["x"])
        off = 8.0 + 0.5 * h["size"]
        best = None
        for k in range(12):
            a = a0 + k * math.pi / 6
            d = (-math.cos(a), -math.sin(a))
            n = (-d[1], d[0])
            s = (h["x"] + n[0] * off - d[0] * 30, h["y"] + n[1] * off - d[1] * 30)
            e = (h["x"] + n[0] * off + d[0] * 30, h["y"] + n[1] * off + d[1] * 30)
            roof = self.roofs.along(s, e, 4.0)
            if roof < alt - 3.0:
                return [*s, alt], [*e, alt], alt
            if best is None or roof < best[0]:
                best = (roof, s, e)
        roof, s, e = best
        z = roof + 5.0
        self.log(f"flyby: no clear line at {alt:.0f} m, passing at {z:.0f} m")
        return [*s, z], [*e, z], z

    def _plan_orbit(self, h, dist):
        R = dist + 0.5 * h["size"]
        ang = math.atan2(self.pos[1] - h["y"], self.pos[0] - h["x"])
        zc = max(h["z"] + 0.6 * max(h.get("height", 4.0), 3.0), 6.0)
        from shapely.geometry import Point

        ring = (
            Point(h["x"], h["y"]).buffer(R + 3.0).difference(Point(h["x"], h["y"]).buffer(max(R - 3.0, 0.5)))
        )
        roof = self.roofs.max_h(ring)
        if roof > zc - 3.0:
            self.log(f"orbit: roofs on the circle, flying it at {roof + 5:.0f} m")
            zc = roof + 5.0
        return {"R": R, "a0": ang, "z": zc, "a": ang}

    def _approach(self):
        step = self.parsed["steps"][self.si]
        a = step["action"]
        h = self.target
        p = step.get("params", {})
        if a == "land" or a == "pickup":
            goal = [h["x"], h["y"], max(h["z"] + 12.0, self.free_alt((h["x"], h["y"]), 3.0) + 3.0)]
        elif a == "drop":
            goal = [
                h["x"],
                h["y"],
                max(h["z"] + DROP_H + 5.0, self.free_alt((h["x"], h["y"]), 6.0) + 3.0),
            ]  # first right over the zone: it is checked and re-seen straight down
        elif a == "enter":  # beside the hangar, not over it
            if self._appr is None:  # the side we come from, fixed once
                ang = math.atan2(self.pos[1] - h["y"], self.pos[0] - h["x"])
                R = 0.5 * h["size"] + 9.0
                gx, gy = h["x"] + R * math.cos(ang), h["y"] + R * math.sin(ang)
                self._appr = [
                    gx,
                    gy,
                    max(h["z"] + h.get("height", 4) + 6, 10.0, self.free_alt((gx, gy), 5.0) + 3.0),
                ]
            goal = self._appr
        elif a == "inspect":
            if self._appr is None:
                self._appr = self.orbit = self._plan_orbit(h, float(p.get("distance_m") or 10.0))
            o = self.orbit
            goal = [h["x"] + o["R"] * math.cos(o["a0"]), h["y"] + o["R"] * math.sin(o["a0"]), o["z"]]
        else:
            if self._appr is None:
                self._appr = self._plan_flyby(h, float(p.get("altitude_m") or 20.0))
                self.pass_end = self._appr[1]
            goal = self._appr[0]
        cmd, ok = self.goto(goal, v=6.0)
        if ok and a in TOUCH and not h.get("verified"):
            self.go("verify")
        elif ok:
            {
                "land": lambda: self.go("descend"),
                "pickup": lambda: self.go("hover_pick"),
                "drop": lambda: self.go("drop_hold"),
                "enter": lambda: self.go("find_door"),
                "inspect": lambda: self.go("orbit"),
                "flyby": lambda: self.go("pass"),
            }[a]()
        elif self.env.t - self.t_state > 150.0:
            self.skip_step("cannot reach the target: skipping this step")
        return cmd

    def drop_offset(self, h_drop=DROP_H):
        """Where the cargo lands relative to the release point."""
        m = float(self.env.cargo["mass"])
        w = np.array([*getattr(self, "_mean_wind", np.zeros(2))[:2], 0.0])
        p = np.zeros(3)
        v = np.zeros(3)
        t = 0.0
        dt = 0.02
        while p[2] > -h_drop + 0.14 and t < 60.0:
            cda = 1.15 if t > 0.45 else 0.02
            vr = v - w
            v = v + (np.array([0, 0, -9.81]) - 0.5 * 1.225 * cda * np.linalg.norm(vr) * vr / m) * dt
            p = p + v * dt
            t += dt
        return p[:2], t

    def wind_est(self):
        return self._last_wind if hasattr(self, "_last_wind") else (0.0, 0.0)

    def _action_states(self, st, stop):
        env = self.env
        h = self.target
        T = env.t - self.t_state
        if st == "descend":  # land: centre from straight-down views while descending to 4 m, then a
            if env.t - getattr(self, "_t_ref", -9) > 0.7:
                self._refine()
                self._t_ref = env.t
            cmd, ok = self.fly_to([h["x"], h["y"], h["z"] + 4.0], v=1.5, tol=0.25)
            if ok:
                self.go("final")
            elif T > 90.0:
                self.skip_step("could not get down to the target: skipping this step")
            return cmd
        if st == "final":
            c, _ = self.fly_to([h["x"], h["y"], h["z"]], v=0.6)
            c["vz"] = -0.45
            if self.vel[2] > -0.08 and (self.range_down < 0.15 or T > 25):
                self.log("touchdown")
                self._next_step(landed=True)
                return stop
            return c
        if st == "drop_hold":
            if not self._drop_locked:  # A: over the zone at the release height, its centre re-seen straight
                cmd, ok = self.fly_to([h["x"], h["y"], h["z"] + DROP_H], v=1.0, tol=0.6)
                if env.t - getattr(self, "_t_ref", -9) > 0.7:
                    self._refine(gate=3.0)
                    self._t_ref = env.t
                if (ok and T > 5.0) or T > 25.0:
                    self._drop_locked = True
                    self.t_state = env.t
                    self.log("drop zone centred from straight above")
                return cmd
            # B: upwind to the release point by the drone's own motion
            T = env.t - self.t_state
            off, _ = self.drop_offset(self.pos[2] - h["z"])
            cmd, ok = self.fly_to([h["x"] - off[0], h["y"] - off[1], h["z"] + DROP_H], v=1.0, tol=0.6)
            env.set_gimbal(
                math.degrees(math.atan2(h["y"] - self.pos[1], h["x"] - self.pos[0]) - env.obs_yaw()),
                -math.degrees(math.atan2(self.pos[2] - h["z"], max(math.hypot(*off), 0.5))),
            )
            self._still = self._still + env.agent_dt if ok and np.linalg.norm(self.vel[:2]) < 0.4 else 0.0
            if (T > 4.0 and self._still > 1.5) or T > 40.0:
                if T < 35.0 and not self.zone_clear(h):
                    if int(env.t) % 5 == 0 and env.t - getattr(self, "_t_zlog", -9) > 4:
                        self.log("drop zone is not clear: waiting")
                        self._t_zlog = env.t
                    return cmd
                env.release_cargo()
                self.log(f"cargo released ({math.hypot(*off):.1f} m upwind of the zone centre)")
                self.debug = {
                    "rel_target": [round(h["x"] - self.pos[0], 2), round(h["y"] - self.pos[1], 2)],
                    "offset": [round(float(off[0]), 2), round(float(off[1]), 2)],
                    "wind_est": [round(float(v), 2) for v in self.wind_est()],
                    "vel": [round(float(v), 2) for v in self.vel],
                }
                self._next_step()
                return stop
            return cmd
        if st == "hover_pick":
            if env.t - getattr(self, "_t_ref", -9) > 0.7:
                self._refine(gate=1.5)
                self._t_ref = env.t
            hz = h["z"] + 0.25
            cmd, ok = self.fly_to([h["x"], h["y"], hz + 5.0], v=1.0, tol=0.15)
            if ok and T > 4:  # tether length from the down rangefinder over the parcel
                self.hook_L = float(np.clip(self.range_down - 0.20, 0.4, 7.5))
                env.winch(self.hook_L)
                self.log(f"lowering the hook {self.hook_L:.2f} m (rangefinder {self.range_down:.2f} m)")
                self.go("hook")
            elif T > 90.0:
                self.skip_step("could not hold over the parcel: skipping this step")
            return cmd
        if st == "hook":  # hold over the parcel
            if env.t - getattr(self, "_t_ref", -9) > 1.0:
                self._refine(gate=1.5)
                self._t_ref = env.t
            cmd, _ = self.fly_to([h["x"], h["y"], h["z"] + 0.25 + 5.0], v=0.4, tol=0.05)
            env.winch(self.hook_L + (0.0, 0.15, -0.15, 0.3, -0.3)[int(T // 8) % 5])
            if env.hook_load():
                self.log("parcel latched")
                env.winch(0.0)
                self._next_step()
                return stop
            if T > 40:
                self.log("could not latch the parcel")
                env.winch(0.0)
                self.skipped.append(self.si)
                self._next_step()
                return stop
            return cmd
        if st == "find_door":
            return self._find_door()
        if st == "exit_out":  # back out of the hangar the way it came in, then up
            goal = self.exit_path[0]
            cmd, ok = self.fly_to(goal, v=1.0 if len(self.exit_path) > 1 else 2.0, tol=0.4)
            if ok:
                self.exit_path.pop(0)
                self.t_state = env.t
            if not self.exit_path:
                self.exit_path = None
                self.log("out of the hangar")
                self.go(self.after_exit)
            elif T > 40.0:
                self.log("could not get out of the hangar: staying landed inside")
                self.exit_path = None
                self.done = True
                self.go("landed")
            return cmd
        if st == "enter_in":
            goal = self.door_path[0]
            outside = len(self.door_path) > (
                2 if DOOR_OLD else 3
            )  # round the hangar and to the door front at normal speed
            cmd, ok = self.fly_to(goal, v=2.5 if outside else 1.0, tol=0.4)
            if (
                not ok
                and math.hypot(cmd["vx"], cmd["vy"]) < 0.5
                and math.hypot(goal[0] - self.pos[0], goal[1] - self.pos[1]) > 0.4
            ):  # do not creep: at least 0.5 m/s
                k = 0.5 / max(math.hypot(cmd["vx"], cmd["vy"]), 1e-3)
                cmd["vx"] *= k
                cmd["vy"] *= k
            if ok:
                self.door_path.pop(0)
                self.t_state = env.t  # the time budget is per leg
            if not self.door_path:
                self.go("final")
                self.target = {**h, "x": goal[0], "y": goal[1], "z": 0.0}
            elif T > 40.0:
                self.log("could not get through the door: backing out")
                self.door_path = []
                self.skip_step("enter failed: skipping this step")
            return cmd
        if st == "orbit":  # a circle controller
            o = self.orbit
            rx, ry = self.pos[0] - h["x"], self.pos[1] - h["y"]
            r = math.hypot(rx, ry)
            a = math.atan2(ry, rx)
            pitch = -math.degrees(math.atan2(self.pos[2] - h["z"] - 0.5 * h.get("height", 2.0), max(r, 1.0)))
            if env.t - getattr(self, "_t_ref", -9) > 1.2:
                self._refine(pitch=pitch, gate=4.0, alpha=0.25)
                self._t_ref = env.t  # re-seen while circling: small steps, the centre must not jump
            if "last" in o and abs(r - o["R"]) < 2.5:
                o["cov"] = o.get("cov", 0.0) + ((a - o["last"] + math.pi) % (2 * math.pi) - math.pi)
            o["last"] = a
            vt = 2.2
            vr = float(np.clip(1.2 * (o["R"] - r), -2.0, 2.0))
            ux, uy = (rx / r, ry / r) if r > 0.1 else (1.0, 0.0)
            cmd = {
                "vx": vt * -uy + vr * ux,
                "vy": vt * ux + vr * uy,
                "vz": float(np.clip(0.9 * (o["z"] - self.pos[2]), -1.5, 1.5)),
                "yaw_rate": 0.0,
            }
            env.set_gimbal(math.degrees(math.atan2(-ry, -rx) - env.obs_yaw()), pitch)
            if o.get("cov", 0.0) > 2 * math.pi + 0.2:
                self.log("orbit complete")
                self._next_step()
            elif T > 2 * math.pi * o["R"] / vt * 3.0 + 60.0:
                self.log(
                    f"orbit is taking too long: stopping it ({math.degrees(o.get('cov', 0.)):.0f} deg done)"
                )
                self._next_step()
            return cmd
        if st == "pass":
            cmd, ok = self.fly_to(self.pass_end, v=4.0)
            env.set_gimbal(
                math.degrees(math.atan2(h["y"] - self.pos[1], h["x"] - self.pos[0]) - env.obs_yaw()), -25.0
            )
            if ok or T > 60.0:
                self.log("pass complete")
                self._next_step()
            return cmd
        if st == "return":
            cmd, ok = self.goto([*self.home, 4.0])
            if ok:
                self.target = {
                    "x": self.home[0],
                    "y": self.home[1],
                    "z": 0.0,
                    "cls": "pad",
                    "size": 3.0,
                    "height": 0.1,
                }
                self.si = 99
                self.go("final")
            return cmd
        if st in ("done", "landed"):
            self.done = True
            return stop
        return stop

    def _next_step(self, landed=False):
        if self.si == 99:
            self.go("landed")
            return
        self.si += 1
        self.target = None
        self.dets = []
        nxt = (
            ("takeoff" if landed else "transit")
            if self.si < len(self.parsed["steps"])
            else ("return" if self.return_home else ("landed" if landed else "done"))
        )
        if getattr(self, "exit_path", None) and nxt in ("takeoff", "transit", "return"):
            self.after_exit = nxt
            self.go("exit_out")
            return  # inside a hangar: out by the door first
        self.exit_path = None
        self.go(nxt)

    def _find_door(self):
        """Circle the hangar 6 m up."""
        env = self.env
        h = self.target
        stop = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
        if getattr(self, "_door_for", None) is not h:
            self._door_for = h
            self.door = {"cov": 0.0, "last": None, "down": False, "pts": []}
        D = self.door
        R = 0.5 * h["size"] + 7.0 + D.get("out", 0.0)
        Z = 6.0
        rx, ry = self.pos[0] - h["x"], self.pos[1] - h["y"]
        r = math.hypot(rx, ry)
        a = math.atan2(ry, rx)
        if not D["down"]:  # out to the circle at a safe height, then down to the circle height
            far = abs(r - R) > 1.0
            cmd, ok = self.fly_to(
                [h["x"] + R * math.cos(a), h["y"] + R * math.sin(a), 10.0 if far else Z], v=2.5, tol=0.8
            )
            if not far and self.pos[2] < Z + 0.6:
                D["down"] = True
                D["last"] = a
            return cmd
        o = env.observe(images=False)
        from nav import rot

        att = o["attitude"]
        pts = o["lidar_points"] @ rot(att["roll"], att["pitch"], att["yaw"]).T + self.pos
        sel = (
            (np.hypot(pts[:, 0] - h["x"], pts[:, 1] - h["y"]) < 0.9 * h["size"] + 3.0)
            & (pts[:, 2] > 0.3)
            & (pts[:, 2] < 5.5)
        )
        if sel.any():
            D["pts"].append(pts[sel][::2])
        near = (np.hypot(pts[:, 0] - h["x"], pts[:, 1] - h["y"]) < 0.75 * h["size"] + 1.0) & (
            pts[:, 2] < 5.5
        )  # the hangar's inside too (its floor): seen only through an opening
        if near.any():
            D.setdefault("rays", []).append((self.pos.copy(), pts[near][::3]))
        D["cov"] += (a - D["last"] + math.pi) % (2 * math.pi) - math.pi
        D["last"] = a
        if D["cov"] > 2 * math.pi + 0.3 or env.t - self.t_state > 150.0:
            door = self._door_from_walls(
                np.concatenate(D["pts"]) if D["pts"] else np.zeros((0, 3)),
                (h["x"], h["y"]),
                h["size"],
                rays=None if DOOR_OLD else D.get("rays"),
            )
            if door is None:
                self.log(
                    "hangar doors look closed (the lidar sees nothing inside through any wall): not entering"
                )
                self.skipped.append(self.si)
                self._next_step()
                return stop
            (cx, cy), (nx, ny), gap, depth, (dx, dy) = door
            fx, fy = dx + nx * 4.0, dy + ny * 4.0
            a_d = math.atan2(fy - h["y"], fx - h["x"])
            da = (a_d - a + math.pi) % (2 * math.pi) - math.pi
            n_arc = int(abs(da) // (math.pi / 6)) + 1
            arc = [
                [h["x"] + R * math.cos(a + da * k / n_arc), h["y"] + R * math.sin(a + da * k / n_arc), Z]
                for k in range(1, n_arc + 1)
            ]  # round the hangar on the circle, never over its roof
            self.door_path = arc + [
                [fx, fy, Z],
                [fx, fy, 1.6],
                [dx, dy, 1.6],
                [dx - nx * 1.5, dy - ny * 1.5, 1.6],
                [cx, cy, 1.6],
            ]  # in front of the door, down to 1.6 m, through it
            self.exit_path = [
                [cx, cy, 1.6],
                [dx - nx * 1.5, dy - ny * 1.5, 1.6],
                [dx, dy, 1.6],
                [fx, fy, 1.6],
                [fx, fy, Z],
            ]  # the way back out
            if DOOR_OLD:
                self.door_path = [[fx, fy, Z], [fx, fy, 1.6], [dx, dy, 1.6], [cx, cy, 1.6]]
                self.exit_path = [[cx, cy, 1.6], [dx, dy, 1.6], [fx, fy, 1.6], [fx, fy, Z]]
            self.target = {**h, "x": cx, "y": cy}
            self.log(f"open door found: a {gap:.1f} m gap in the wall facing ({nx:.2f}, {ny:.2f})")
            self.go("enter_in")
            return stop
        vt = 1.5
        vr = float(np.clip(1.2 * (R - r), -1.5, 1.5))
        ux, uy = (rx / r, ry / r) if r > 0.1 else (1.0, 0.0)
        if self.blocked_ahead((-uy, ux), dist=5.0, half_w=1.5):
            D["out"] = min(D.get("out", 0.0) + 0.05, 12.0)  # something on the circle: widen it
        yaw_err = (math.atan2(-ry, -rx) - env.obs_yaw() + math.pi) % (
            2 * math.pi
        ) - math.pi  # face the hangar: the lidar looks at it
        return {
            "vx": vt * -uy + vr * ux,
            "vy": vt * ux + vr * uy,
            "vz": float(np.clip(0.9 * (Z - self.pos[2]), -1.0, 1.0)),
            "yaw_rate": float(np.clip(yaw_err, -0.8, 0.8)),
        }

    @staticmethod
    def _door_from_walls(P, centre_hint=None, size_hint=None, rays=None):
        """Hangar rectangle from the lidar."""
        if rays is not None:
            return Brain._door_seen_through(P, centre_hint, size_hint, rays)
        if len(P) < 200:
            return None
        xy = P[:, :2]
        c0 = np.median(xy, 0) if centre_hint is None else np.asarray(centre_hint, float)
        r = np.hypot(*(xy - c0).T)
        sz = size_hint or 8.0
        far = r > 0.75 * sz
        zg = float(np.median(P[far, 2])) if far.sum() > 50 and np.median(P[far, 2]) < 1.2 else 0.3
        top = (P[:, 2] > zg + 2.4) & (r < 0.9 * sz)
        if top.sum() < 100:
            return None
        xt = xy[top]
        best = None
        for th in np.radians(np.arange(0.0, 90.0, 2.0)):
            u = xt @ np.array([math.cos(th), math.sin(th)])
            v = xt @ np.array([-math.sin(th), math.cos(th)])
            u0, u1, v0, v1 = (
                np.percentile(u, 3),
                np.percentile(u, 97),
                np.percentile(v, 3),
                np.percentile(v, 97),
            )
            if best is None or (u1 - u0) * (v1 - v0) < best[0]:
                best = ((u1 - u0) * (v1 - v0), th, u0, u1, v0, v1)
        _, th, u0, u1, v0, v1 = best
        eu, ev = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
        if u1 - u0 < 3.0 or v1 - v0 < 3.0:
            return None
        u = xy @ eu
        v = xy @ ev
        low = (P[:, 2] > zg + 0.5) & (P[:, 2] < zg + 2.0)
        res = []
        for c, at, t, t0, t1, n, depth in (
            (u, u1, v, v0, v1, eu, u1 - u0),
            (u, u0, v, v0, v1, -eu, u1 - u0),
            (v, v1, u, u0, u1, ev, v1 - v0),
            (v, v0, u, u0, u1, -ev, v1 - v0),
        ):
            on = low & (np.abs(c - at) < 0.5) & (t > t0 + 0.1 * (t1 - t0)) & (t < t1 - 0.1 * (t1 - t0))
            if on.sum() < 5:
                continue  # this side was not seen: no evidence either way
            nb = int((t1 - t0) / 0.4) + 1
            cnt = np.bincount(((t[on] - t0) / 0.4).astype(int).clip(0, nb - 1), minlength=nb)
            filled = cnt >= max(
                3, 0.3 * np.percentile(cnt, 90)
            )  # the wall is dense; a few returns through the opening do not close it
            k0, k1 = int(0.1 * nb), int(0.9 * nb) + 1
            run = best_run = end = 0
            for k in range(k0, k1):
                run = 0 if filled[k] else run + 1
                if run > best_run:
                    best_run, end = run, k
            res.append((best_run * 0.4, at, t0 + (end - best_run / 2 + 0.5) * 0.4, n, depth, c is u))
        if not res:
            return None
        gap, at, mid, n, depth, along_u = max(res, key=lambda x: x[0])
        if gap < 2.0:
            return None
        centre = ((u0 + u1) / 2) * eu + ((v0 + v1) / 2) * ev
        door = centre + n * depth / 2
        return (
            (float(centre[0]), float(centre[1])),
            (float(n[0]), float(n[1])),
            float(gap),
            float(depth),
            (float(door[0]), float(door[1])),
        )

    @staticmethod
    def _hangar_box(P, centre_hint=None, size_hint=None):
        """Minimum-area box of the roof and upper-wall points."""
        if len(P) < 200:
            return None
        xy = P[:, :2]
        c0 = np.median(xy, 0) if centre_hint is None else np.asarray(centre_hint, float)
        r = np.hypot(*(xy - c0).T)
        sz = size_hint or 8.0
        far = r > 0.75 * sz
        zg = float(np.median(P[far, 2])) if far.sum() > 50 and np.median(P[far, 2]) < 1.2 else 0.3
        top = (P[:, 2] > zg + 2.4) & (r < 0.9 * sz)
        if top.sum() < 100:
            return None
        xt = xy[top]
        best = None
        for th in np.radians(np.arange(0.0, 90.0, 2.0)):
            u = xt @ np.array([math.cos(th), math.sin(th)])
            v = xt @ np.array([-math.sin(th), math.cos(th)])
            u0, u1, v0, v1 = (
                np.percentile(u, 3),
                np.percentile(u, 97),
                np.percentile(v, 3),
                np.percentile(v, 97),
            )
            if best is None or (u1 - u0) * (v1 - v0) < best[0]:
                best = ((u1 - u0) * (v1 - v0), th, u0, u1, v0, v1)
        _, th, u0, u1, v0, v1 = best
        if u1 - u0 < 3.0 or v1 - v0 < 3.0:
            return None
        return (
            zg,
            np.array([math.cos(th), math.sin(th)]),
            np.array([-math.sin(th), math.cos(th)]),
            u0,
            u1,
            v0,
            v1,
            float(np.percentile(P[top, 2], 90)),
        )

    @staticmethod
    def _door_seen_through(P, centre_hint, size_hint, rays):
        """The door = the side through which the lidar saw the hangar's INSIDE."""
        box = Brain._hangar_box(P, centre_hint, size_hint)
        if box is None or not rays:
            return None
        zg, eu, ev, u0, u1, v0, v1, zr = box
        sides = []  # (normal, coordinate axis, at, along axis, lo, hi)
        for n, ax, at, al, lo, hi in (
            (eu, 0, u1, 1, v0, v1),
            (-eu, 0, u0, 1, v0, v1),
            (ev, 1, v1, 0, u0, u1),
            (-ev, 1, v0, 0, u0, u1),
        ):
            sides.append((n, ax, at, al, lo, hi))
        cross = [[] for _ in sides]
        for o, Q in rays:
            q = np.stack([Q[:, :2] @ eu, Q[:, :2] @ ev], 1)
            oz = np.array([o[:2] @ eu, o[:2] @ ev])
            ins = (
                (q[:, 0] > u0 + 0.5)
                & (q[:, 0] < u1 - 0.5)
                & (q[:, 1] > v0 + 0.5)
                & (q[:, 1] < v1 - 0.5)
                & (Q[:, 2] > zg - 0.4)
                & (Q[:, 2] < zr - 0.6)
            )
            if not ins.any() or (u0 < oz[0] < u1 and v0 < oz[1] < v1):
                continue  # the lidar itself over the box: no evidence of a side
            q = q[ins]
            for k, (n, ax, at, al, lo, hi) in enumerate(sides):
                if (oz[ax] - at) * (1 if (k % 2 == 0) else -1) <= 0:
                    continue  # the lidar is not outside this side
                dq = q[:, ax] - oz[ax]
                ok = np.abs(dq) > 1e-6
                s = np.where(ok, (at - oz[ax]) / np.where(ok, dq, 1.0), -1.0)
                t = oz[al] + s * (q[:, al] - oz[al])
                hit = (s > 0) & (s < 1) & (t > lo) & (t < hi)
                cross[k] += list(t[hit])
        cnt_s = sorted(len(c) for c in cross)
        k = int(np.argmax([len(c) for c in cross]))
        if cnt_s[-1] < 30 or cnt_s[-1] < 3 * cnt_s[-2]:
            return None  # a real opening gives hundreds of rays through ONE side
        n, ax, at, al, lo, hi = sides[k]
        t = np.asarray(cross[k])
        tm = float(np.median(t))
        e_al = eu if al == 0 else ev
        e_ax = eu if ax == 0 else ev
        c = P[:, :2] @ e_ax
        tt = P[:, :2] @ e_al
        low = (P[:, 2] > zg + 0.5) & (P[:, 2] < zg + 2.0) & (np.abs(c - at) < 0.5) & (tt > lo) & (tt < hi)
        nb = int((hi - lo) / 0.4) + 1
        cnt = np.bincount(((tt[low] - lo) / 0.4).astype(int).clip(0, nb - 1), minlength=nb)
        gap = (
            float(np.percentile(t, 95) - np.percentile(t, 5)) + 0.4
        )  # the spread of the rays through the opening
        if low.sum() >= 15:  # the wall around the opening was seen
            filled = cnt >= max(3, 0.3 * np.percentile(cnt, 90))
            km = int(np.clip((tm - lo) / 0.4, 0, nb - 1))
            free = [j for j in range(max(0, km - 3), min(nb, km + 4)) if not filled[j]]
            if free:
                a_ = b_ = min(free, key=lambda j: abs(j - km))
                while a_ > 0 and not filled[a_ - 1]:
                    a_ -= 1
                while b_ < nb - 1 and not filled[b_ + 1]:
                    b_ += 1
                gap = (b_ - a_ + 1) * 0.4
                tm = lo + (a_ + b_ + 1) / 2 * 0.4
        if gap < 1.6:
            return None
        centre = ((u0 + u1) / 2) * eu + ((v0 + v1) / 2) * ev
        depth = (u1 - u0) if ax == 0 else (v1 - v0)
        door = at * e_ax + tm * e_al
        return (
            (float(centre[0]), float(centre[1])),
            (float(n[0]), float(n[1])),
            float(gap),
            float(depth),
            (float(door[0]), float(door[1])),
        )
