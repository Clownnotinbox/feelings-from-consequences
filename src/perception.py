"""Brain-side perception."""

import math
from collections import deque
import numpy as np


def rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def clusters(P, voxel=0.6, min_pts=1):
    """Connected components of occupied voxels."""
    if len(P) == 0:
        return []
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    keys = np.floor(P / voxel).astype(np.int64)
    uk, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    k = uk - uk.min(0) + 1
    dims = k.max(0) + 2
    code = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    order = np.argsort(code)
    sc = code[order]
    rows, cols = [], []
    for dx, dy, dz in [
        (a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)
    ]:
        c2 = code + (dx * dims[1] + dy) * dims[2] + dz
        pos = np.minimum(np.searchsorted(sc, c2), len(sc) - 1)
        m = sc[pos] == c2
        rows.append(np.nonzero(m)[0])
        cols.append(order[pos[m]])
    r, c = np.concatenate(rows), np.concatenate(cols)
    n = len(uk)
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), directed=False)
    pl = lab[inv]
    idx = np.argsort(pl, kind="stable")
    cuts = np.flatnonzero(np.diff(pl[idx])) + 1
    return [g for g in np.split(idx, cuts) if len(g) >= min_pts]


class Track:
    def __init__(self, tid, p, t):
        self.id, self.p, self.v, self.t, self.hits, self.miss = tid, p.copy(), np.zeros(3), t, 1, 0
        self.Pp, self.Pv = 1.0, 25.0  # simple per-axis variances (position, velocity)

    def predict(self, t, v_own):
        dt = t - self.t
        self.p = self.p + (self.v - v_own) * dt
        self.t = t
        self.Pp += self.Pv * dt * dt + 0.05
        self.Pv += 4.0 * dt  # process noise: objects may accelerate (birds, drones)

    def update(self, z, dt, v_own):
        r = 0.15 + 0.02 * float(np.linalg.norm(z))  # measurement noise grows with range (sparser points)
        innov = z - self.p
        if self.hits == 1 and dt > 0:  # second sighting: initialise velocity directly from the displacement
            self.v = innov / dt
            self.p = z.copy()
            self.Pv = 4.0 * r / (dt * dt)
            self.hits += 1
            self.miss = 0
            return
        kp = self.Pp / (self.Pp + r)
        self.p = self.p + kp * innov
        if dt > 0:  # velocity from the innovation
            kv = min(1.0, self.Pv / (self.Pv + r / (dt * dt)))
            self.v = self.v + kv * innov / dt
            self.Pv *= 1 - kv
        self.Pp *= 1 - kp
        self.hits += 1
        self.miss = 0


class Tracker:
    """Ego-motion-compensated multi-object tracker on small clusters; large clusters are static structure."""

    def __init__(self, max_extent=3.0, gate=2.5):
        self.tracks, self.next_id, self.max_extent, self.gate, self.t_last = [], 0, max_extent, gate, None

    def step(self, obs):
        t = obs["t"]
        a = obs["attitude"]
        R = rot(a["roll"], a["pitch"], a["yaw"])
        v_own = np.array(obs["vel_est"])
        P = obs["lidar_points"] @ R.T if len(obs["lidar_points"]) else np.zeros((0, 3))
        alt = obs["range_down"] if 0 < obs["range_down"] < 39 else obs["baro_alt"]
        P = P[P[:, 2] > -alt + 0.4]  # drop ground returns
        objs, static = [], []
        for idx in clusters(P):
            C = P[idx]
            ext = C.max(0) - C.min(0)
            (objs if ext.max() < self.max_extent else static).append(
                C.mean(0) if ext.max() < self.max_extent else C
            )
        dt = 0.0 if self.t_last is None else t - self.t_last
        self.t_last = t
        for tr in self.tracks:
            tr.predict(t, v_own)
        used = np.zeros(len(objs), bool)
        order = sorted(self.tracks, key=lambda tr: -tr.hits)  # greedy nearest-neighbour association
        if objs and order:
            O = np.array(objs)
            D = np.linalg.norm(
                O[None] - np.array([tr.p for tr in order])[:, None], axis=2
            )  # tracks x objects, one matrix
            for i, tr in enumerate(order):
                if used.all():
                    break
                row = np.where(used, np.inf, D[i])
                j = int(np.argmin(row))
                gate = (5.0 if tr.hits < 2 else self.gate) + float(
                    np.linalg.norm(tr.v - v_own)
                ) * dt * 0.5  # young tracks: velocity unknown
                if row[j] < gate:
                    tr.update(O[j], dt, v_own)
                    used[j] = True
                else:
                    tr.miss += 1
        for j in np.nonzero(~used)[0]:
            self.tracks.append(Track(self.next_id, objs[j], t))
            self.next_id += 1
        self.tracks = [tr for tr in self.tracks if tr.miss < 3]
        static_pts = np.concatenate(static) if static else np.zeros((0, 3))
        return self.summary(v_own, static_pts)

    def summary(self, v_own, static_pts, r_safe=1.2):
        out = []
        for tr in self.tracks:
            if tr.hits < 2:
                continue  # velocity exists from the second sighting on
            p, v_rel = tr.p, tr.v - v_own
            d = float(np.linalg.norm(p))
            vv = float(v_rel @ v_rel)
            closing = -float(p @ v_rel) / max(d, 1e-6)
            t_cpa = max(0.0, -float(p @ v_rel) / vv) if vv > 1e-6 else 0.0
            d_cpa = float(np.linalg.norm(p + v_rel * t_cpa))
            out.append(
                {
                    "id": tr.id,
                    "rel_pos": p.round(2).tolist(),
                    "vel_world": tr.v.round(2).tolist(),
                    "dist": round(d, 2),
                    "closing": round(closing, 2),
                    "ttc": round(d / closing, 3) if closing > 0.3 else None,
                    "t_cpa": round(t_cpa, 3),
                    "d_cpa": round(d_cpa, 2),
                    "moving": bool(np.linalg.norm(tr.v) > 1.5),
                    "threat": bool(closing > 0.3 and d_cpa < r_safe + 0.1 * d and t_cpa < 3.0),
                }
            )  # wider margin far away (noisier)
        sp = float(np.linalg.norm(v_own))
        t_static = None
        if sp > 0.5 and len(static_pts):  # nearest static structure inside a 25-degree cone around the flight
            u = v_own / sp
            proj = static_pts @ u
            lat = np.linalg.norm(static_pts - np.outer(proj, u), axis=1)
            ahead = (proj > 0) & (lat < np.maximum(0.8, proj * math.tan(math.radians(25))))
            if ahead.any():
                t_static = round(float(proj[ahead].min()) / sp, 3)
        return {"tracks": out, "t_static": t_static, "v_own": v_own.tolist()}


class Reflex:
    """Classic evasive reflex."""

    def __init__(self, tau=1.0, r_safe=1.2, speed=7.0, hold=0.6):
        self.tau, self.r_safe, self.speed, self.hold, self.until, self.dir, self.fired = (
            tau,
            r_safe,
            speed,
            hold,
            -1.0,
            None,
            [],
        )

    def act(self, t, summary, cmd):
        threats = [s for s in summary["tracks"] if s["threat"] and s["t_cpa"] < self.tau]
        if threats and t > self.until:
            s = min(threats, key=lambda s: s["t_cpa"])
            v_rel = np.array(s["vel_world"]) - np.array(summary["v_own"])
            r_cpa = np.array(s["rel_pos"]) + v_rel * s["t_cpa"]
            away = (
                -r_cpa
                if np.linalg.norm(r_cpa) > 0.2
                else np.cross(v_rel, [0, 0, 1.0]) + np.array([0, 0, 0.3])
            )
            self.dir = away / max(np.linalg.norm(away), 1e-6)
            self.until = t + self.hold
            self.fired.append({"t": t, "track": s["id"], "t_cpa": s["t_cpa"]})
        if t <= self.until and self.dir is not None:
            v = self.dir * self.speed
            return {"vx": float(v[0]), "vy": float(v[1]), "vz": float(np.clip(v[2], -3, 4)), "yaw_rate": 0.0}
        if summary["t_static"] is not None and summary["t_static"] < 0.8:
            return {"vx": 0.0, "vy": 0.0, "vz": cmd.get("vz", 0.0), "yaw_rate": 0.0}
        return cmd


def drop_ground(P, obs):
    """Ground segmentation from the points themselves."""
    if len(P) == 0:
        return P
    alt = obs["range_down"] if 0 < obs["range_down"] < 39 else obs["baro_alt"]
    band = np.abs(P[:, 2] + alt) < 2.5
    gz = float(np.median(P[band, 2])) if band.sum() > 20 else -alt
    rh = np.linalg.norm(P[:, :2], axis=1)
    return P[P[:, 2] > gz + 0.4 + 0.02 * rh]


class StaticMap:
    """Short-memory occupancy evidence in a world-aligned frame."""

    BANDS = ((12.0, 0.5, 192, 96), (1e9, 1.5, 64, 32))  # (max range, voxel, ring size xy, ring size z)

    def __init__(self, decay=0.85, thr=1.5):
        self.decay, self.thr, self.frame = decay, thr, 0
        self.G = [np.zeros((n, n, nz), np.float32) for _, _, n, nz in self.BANDS]
        self.S = [np.zeros((n, n, nz), np.int32) for _, _, n, nz in self.BANDS]

    def step(self, W, d):
        """W: points in the world-aligned frame."""
        self.frame += 1
        static = np.zeros(len(W), bool)
        lo = 0.0
        for b, (hi, vox, n, nz) in enumerate(self.BANDS):
            m = (d >= lo) & (d < hi)
            lo = hi
            if not m.any():
                continue
            k = np.floor(W[m] / vox).astype(np.int64)
            ix, iy, iz = k[:, 0] % n, k[:, 1] % n, k[:, 2] % nz
            G, S = self.G[b], self.S[b]
            score = G[ix, iy, iz] * self.decay ** (self.frame - S[ix, iy, iz])
            if True:  # tolerate one voxel of jitter (sparse sampling, own-motion error)
                nb = np.max(
                    [
                        G[(ix + dx) % n, (iy + dy) % n, (iz + dz) % nz]
                        * self.decay ** (self.frame - S[(ix + dx) % n, (iy + dy) % n, (iz + dz) % nz])
                        for dx in (-1, 0, 1)
                        for dy in (-1, 0, 1)
                        for dz in (-1, 0, 1)
                    ],
                    axis=0,
                )
                static[m] = nb >= self.thr
            flat = (ix * n + iy) * nz + iz
            u, first = np.unique(flat, return_index=True)
            G.reshape(-1)[u] = score[first] + 1.0
            S.reshape(-1)[u] = self.frame
        return static


class Tracker2(Tracker):
    """Tracker v2 (24.10)."""

    def __init__(self, max_extent=3.0, gate=2.5):
        super().__init__(max_extent, gate)
        self.map = StaticMap()
        self.pos = np.zeros(3)

    def step(self, obs):
        t = obs["t"]
        a = obs["attitude"]
        R = rot(a["roll"], a["pitch"], a["yaw"])
        v_own = np.array(obs["vel_est"])
        dt = 0.0 if self.t_last is None else t - self.t_last
        self.pos = self.pos + v_own * dt
        P = obs["lidar_points"] @ R.T if len(obs["lidar_points"]) else np.zeros((0, 3))
        P = drop_ground(P, obs)
        static_mask = self.map.step(P + self.pos, np.linalg.norm(P, axis=1)) if len(P) else np.zeros(0, bool)
        Pd = P[~static_mask]
        objs, static = [], [P[static_mask]]
        for idx in clusters(Pd):
            C = Pd[idx]
            ext = C.max(0) - C.min(0)
            if ext.max() >= self.max_extent:
                static.append(C)
            elif len(C) >= 2 or np.linalg.norm(C[0]) < 15.0:
                objs.append(C.mean(0))  # a lone far return is too weak to start a track
        self.t_last = t
        for tr in self.tracks:
            tr.predict(t, v_own)
        used = np.zeros(len(objs), bool)
        order = sorted(self.tracks, key=lambda tr: -tr.hits)
        if objs and order:
            O = np.array(objs)
            D = np.linalg.norm(O[None] - np.array([tr.p for tr in order])[:, None], axis=2)
            for i, tr in enumerate(order):
                if used.all():
                    break
                row = np.where(used, np.inf, D[i])
                j = int(np.argmin(row))
                gate = (5.0 if tr.hits < 2 else self.gate) + float(np.linalg.norm(tr.v - v_own)) * dt * 0.5
                if row[j] < gate:
                    tr.update(O[j], dt, v_own)
                    used[j] = True
                else:
                    tr.miss += 1
        for j in np.nonzero(~used)[0]:
            self.tracks.append(Track(self.next_id, objs[j], t))
            self.next_id += 1
        self.tracks = [tr for tr in self.tracks if tr.miss < 3]
        out = self.summary(v_own, np.concatenate(static) if static else np.zeros((0, 3)))
        conf = {tr.id for tr in self.tracks if tr.hits >= 3}
        for s in out["tracks"]:
            if s["id"] not in conf:
                s["moving"] = s["threat"] = False
        return out


class OcclusionTracker(Tracker):
    """Tracker v3 (24.10)."""

    AZ, EL, EL0 = 1.0, 2.0, -60.0

    def __init__(self, max_extent=3.0, gate=2.5, memory=3):
        super().__init__(max_extent, gate)
        self.pos = np.zeros(3)
        self.past = deque(maxlen=memory)
        self.n_az, self.n_el = int(360 / self.AZ), int(120 / self.EL)

    def _bins(self, Q):
        d = np.linalg.norm(Q, axis=1)
        az = (np.degrees(np.arctan2(Q[:, 1], Q[:, 0])) + 360.0) % 360.0
        el = np.degrees(np.arcsin(np.clip(Q[:, 2] / np.maximum(d, 1e-6), -1, 1)))
        return (
            (az / self.AZ).astype(int) % self.n_az,
            np.clip(((el - self.EL0) / self.EL).astype(int), 0, self.n_el - 1),
            d,
        )

    def step(self, obs):
        from scipy.ndimage import minimum_filter

        t = obs["t"]
        a = obs["attitude"]
        R = rot(a["roll"], a["pitch"], a["yaw"])
        v_own = np.array(obs["vel_est"])
        dt = 0.0 if self.t_last is None else t - self.t_last
        self.pos = self.pos + v_own * dt
        P_all = obs["lidar_points"] @ R.T if len(obs["lidar_points"]) else np.zeros((0, 3))
        new = np.zeros(len(P_all), bool)
        if self.past and len(P_all):
            Wp = np.concatenate(self.past) - self.pos  # past returns seen from here
            ia, ie, dp = self._bins(Wp)
            depth = np.full((self.n_az, self.n_el), np.inf)
            np.minimum.at(depth, (ia, ie), dp)
            depth = minimum_filter(
                depth, size=3, mode=("wrap", "nearest")
            )  # tolerate one bin of pointing / own-motion error
            ja, je, dc = self._bins(P_all)
            new = dc < depth[ja, je] - (0.5 + 0.03 * dc)
        self.past.append(P_all + self.pos)
        P = drop_ground(P_all[new], obs) if new.any() else np.zeros((0, 3))
        objs, static = [], [P_all[~new]]
        for idx in clusters(P):
            C = P[idx]
            ext = C.max(0) - C.min(0)
            if ext.max() >= self.max_extent:
                static.append(C)
            else:
                objs.append(C.mean(0))
        self.t_last = t
        for tr in self.tracks:
            tr.predict(t, v_own)
        used = np.zeros(len(objs), bool)
        order = sorted(self.tracks, key=lambda tr: -tr.hits)
        if objs and order:
            O = np.array(objs)
            D = np.linalg.norm(O[None] - np.array([tr.p for tr in order])[:, None], axis=2)
            for i, tr in enumerate(order):
                if used.all():
                    break
                row = np.where(used, np.inf, D[i])
                j = int(np.argmin(row))
                gate = (5.0 if tr.hits < 2 else self.gate) + float(np.linalg.norm(tr.v - v_own)) * dt * 0.5
                if row[j] < gate:
                    tr.update(O[j], dt, v_own)
                    used[j] = True
                else:
                    tr.miss += 1
        for j in np.nonzero(~used)[0]:
            self.tracks.append(Track(self.next_id, objs[j], t))
            self.next_id += 1
        self.tracks = [tr for tr in self.tracks if tr.miss < 3]
        st = drop_ground(np.concatenate(static), obs) if static else np.zeros((0, 3))
        out = self.summary(v_own, st)
        conf = {tr.id for tr in self.tracks if tr.hits >= 3}
        for s in out["tracks"]:
            if s["id"] not in conf:
                s["moving"] = s["threat"] = False
        return out


TRACKERS = {"v1": Tracker, "staticmap": Tracker2, "occlusion": OcclusionTracker}


def make_tracker(name=None):
    """The brain's tracker; DRONE_TRACKER=v1 reproduces data v1-v3 (before 24.10, 14:40)."""
    import os

    return TRACKERS[name or os.environ.get("DRONE_TRACKER", "occlusion")]()
