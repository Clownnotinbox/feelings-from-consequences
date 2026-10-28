"""Stage F navigation without trusting GPS."""

import math
import numpy as np


class WallField:
    """Distance (m) to the nearest mapped wall at several height bands, on a grid."""

    def __init__(self, buildings, half=290.0, res=0.5, bands=(3.0, 10.0, 20.0, 32.0, 50.0)):
        from PIL import Image, ImageDraw
        from scipy.ndimage import distance_transform_edt

        self.half, self.res, self.bands = half, res, bands
        n = int(2 * half / res)
        self.n = n
        self.D = []
        for z in bands:
            im = Image.new("L", (n, n), 0)
            dr = ImageDraw.Draw(im)
            for b in buildings:
                if b["h"] <= z:
                    continue
                for ring in [b["poly"].exterior] + list(b["poly"].interiors):
                    pts = [((x + half) / res, (half - y) / res) for x, y in ring.coords]
                    dr.line(pts, fill=255, width=2)
            wall = np.array(im) > 0
            self.D.append(
                distance_transform_edt(~wall) * res if wall.any() else np.full((n, n), 99.0, np.float32)
            )
        self.D = np.stack(self.D).astype(np.float32)

    def dist(self, xy, z):
        k = (
            int(np.clip(np.searchsorted(self.bands, z) - 1, 0, len(self.bands) - 1))
            if z > self.bands[0]
            else 0
        )
        i = ((self.half - xy[:, 1]) / self.res).astype(int)
        j = ((xy[:, 0] + self.half) / self.res).astype(int)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        d = np.full(len(xy), 99.0, np.float32)
        d[ok] = self.D[k, i[ok], j[ok]]
        return d


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


class Navigator:
    """Position estimate (x, y from odometry + particle-filter offset."""

    def __init__(self, MAP, home, n=400, seed=0):
        self.F = WallField(MAP["buildings"])
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.odo = np.array(home[:2], float)
        self.P = np.zeros((n, 3))
        self.w = np.full(n, 1.0 / n)  # particles: offset dx, dy, heading error
        self.gps_trust = 1.0
        self.gps_bad_t = 0.0
        self.t = 0.0
        self.last_gps = None
        self.k = 0
        self.spread = 0.0
        self.matched = 0
        self.baro0 = None

    def step(self, o, dt):
        v = np.array(o["vel_est"][:2])
        self.odo += v * dt
        self.t = o["t"]
        self.k += 1
        q = (
            0.03 * dt + 0.02 * np.linalg.norm(v) * dt
        )  # random walk: dead reckoning error grows with distance flown
        self.P[:, :2] += self.rng.normal(0, q, (self.n, 2))
        self.P[:, 2] += self.rng.normal(0, 0.002, self.n)
        if self.baro0 is None:
            self.baro0 = o["baro_alt"] - 0.3
        z = o["baro_alt"] - self.baro0  # height above the take-off point
        if self.k % 5 == 0 and len(o["lidar_points"]):
            self._update_lidar(o, z)
        g = o.get("gps")
        if g is not None:
            est = self.estimate()[:2]
            d = math.hypot(g["x"] - est[0], g["y"] - est[1])
            sig = max(3.0, 2.5 * g.get("hdop", 1.0) + 2 * self.spread)
            stuck = (
                self.last_gps is not None
                and g["x"] == self.last_gps[0]
                and g["y"] == self.last_gps[1]
                and np.linalg.norm(v) > 0.5
            )
            agree = d < 3 * sig and not stuck
            self.gps_trust = float(np.clip(self.gps_trust + (0.08 if agree else -0.15), 0, 1))
            self.last_gps = (g["x"], g["y"])
            if self.gps_trust > 0.6:  # GPS pulls the particles (a soft measurement), weighted by its hdop
                pos = self.odo[None] + self.P[:, :2]
                dd = np.linalg.norm(pos - np.array([g["x"], g["y"]]), axis=1)
                self.w *= np.exp(-0.5 * (dd / max(2.0, 1.5 * g.get("hdop", 1.0))) ** 2) + 1e-12
                self._normalize()
        else:
            self.gps_trust = max(0.0, self.gps_trust - 0.02)
        return self.estimate(z)

    def _update_lidar(self, o, z):
        a = o["attitude"]
        R = rot(a["roll"], a["pitch"], a["yaw"])
        pts = o["lidar_points"] @ R.T  # world-aligned, drone-centred
        sel = (np.abs(pts[:, 2]) < 4.0) & (pts[:, 2] + z > 1.5) & (np.linalg.norm(pts[:, :2], axis=1) > 3.0)
        pts = pts[sel][:, :2]
        if len(pts) < 20:
            self.matched = 0
            return
        pts = pts[self.rng.choice(len(pts), min(160, len(pts)), replace=False)]
        self.matched = len(pts)
        ll = np.zeros(self.n)
        for k in range(self.n):
            c, s = math.cos(self.P[k, 2]), math.sin(self.P[k, 2])
            xy = pts @ np.array([[c, s], [-s, c]]) + self.odo + self.P[k, :2]
            d = self.F.dist(xy, z)
            ll[k] = -np.sum(np.minimum(d, 2.5) ** 2) / (
                2 * 0.7**2
            )  # truncated: trees, cars, unmapped things do not dominate
        ll -= ll.max()
        self.w *= np.exp(ll / max(1.0, len(pts) / 40.0)) + 1e-300
        self._normalize()

    def _normalize(self):
        s = self.w.sum()
        self.w = self.w / s if s > 0 and np.isfinite(s) else np.full(self.n, 1.0 / self.n)
        if 1.0 / np.sum(self.w**2) < self.n / 2:  # systematic resampling
            u = (self.rng.random() + np.arange(self.n)) / self.n
            idx = np.minimum(np.searchsorted(np.cumsum(self.w), u), self.n - 1)
            self.P = self.P[idx] + self.rng.normal(0, [0.15, 0.15, 0.003], (self.n, 3))
            self.w = np.full(self.n, 1.0 / self.n)

    def estimate(self, z=None):
        m = (self.P * self.w[:, None]).sum(0)
        xy = self.odo + m[:2]
        self.spread = float(np.sqrt((self.w * ((self.P[:, :2] - m[:2]) ** 2).sum(1)).sum()))
        return np.array([xy[0], xy[1], z if z is not None else 0.0])
