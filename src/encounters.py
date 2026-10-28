"""Encounter generator."""

import math
import numpy as np

G = 9.81
FAMILIES = ("straight", "ballistic", "pursuit", "zigzag", "erratic", "stopper")
WEIGHTS = (0.25, 0.2, 0.2, 0.15, 0.1, 0.1)
SIZES = (0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 1.5)  # pool of bodies in the world (diameters, m)


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


class Encounter:
    def __init__(self, eid, body, size, fam, speed, p0, t0, aim, T, miss, rng):
        self.id, self.body, self.size, self.fam, self.speed, self.t0, self.T, self.miss = (
            eid,
            body,
            size,
            fam,
            speed,
            t0,
            T,
            miss,
        )
        self.p = p0.copy()
        self.prev = p0.copy()
        self.rng = rng
        self.done = False
        self.hit = False
        self.min_dist = 1e9
        d = aim - p0
        self.v = unit(d) * speed
        if fam == "ballistic" and T > 0:
            self.v = d / T - 0.5 * np.array([0, 0, -G]) * T  # reach the aim point under gravity
        self.side = unit(np.cross(unit(d), rng.normal(size=3)))
        self.zz_amp = rng.uniform(1, 5)
        self.zz_per = rng.uniform(0.5, 2.0)
        self.lat_v = np.zeros(3)
        self.stop_at = rng.uniform(3, 8)
        self.base = p0.copy()
        self.max_lat_acc = rng.uniform(3, 10) * G
        self.offset = None

    def step(self, t, dt, drone_p, drone_v):
        tau = t - self.t0
        self.prev = self.p.copy()
        if self.fam == "straight":
            self.p = self.p + self.v * dt
        elif self.fam == "ballistic":
            self.v = self.v + np.array([0, 0, -G]) * dt
            self.p = self.p + self.v * dt
        elif self.fam == "pursuit":  # chase the drone's current position + a fixed miss offset, with a
            if self.offset is None:
                self.offset = self.side * self.miss
            want = unit(drone_p + self.offset - self.p) * self.speed
            dv = want - self.v
            lim = self.max_lat_acc * dt
            if np.linalg.norm(dv) > lim:
                dv = unit(dv) * lim
            self.v = unit(self.v + dv) * self.speed
            self.p = self.p + self.v * dt
        elif self.fam in ("zigzag", "erratic"):
            self.base = self.base + self.v * dt
            if self.fam == "zigzag":
                lat = self.side * self.zz_amp * math.sin(2 * math.pi * tau / self.zz_per)
            else:
                self.lat_v += (-self.lat_v / 0.7) * dt + self.rng.normal(0, 3.0, 3) * math.sqrt(dt)
                self.lat_v -= self.lat_v @ unit(self.v) * unit(self.v)
                lat = self.lat_v * min(tau, 2.0)
            self.p = self.base + lat
        elif self.fam == "stopper":  # brake to a stop stop_at metres before the drone, then hover
            dist = float(np.linalg.norm(drone_p - self.p))
            sp = self.speed if dist > self.stop_at + 2 else max(0.0, self.speed * (dist - self.stop_at) / 2)
            self.p = self.p + unit(drone_p - self.p) * sp * dt
        # swept closest distance to the drone during this step
        seg = self.p - self.prev
        L2 = float(seg @ seg)
        u = 0.0 if L2 < 1e-12 else float(np.clip((drone_p - self.prev) @ seg / L2, 0, 1))
        d = float(np.linalg.norm(self.prev + u * seg - drone_p))
        self.min_dist = min(self.min_dist, d)
        if not self.hit and d < 0.3 + self.size / 2:
            self.hit = True
        if tau > self.T + 3 or float(np.linalg.norm(self.p - drone_p)) > 250 or self.p[2] < -1:
            self.done = True


class EncounterGenerator:
    def __init__(
        self,
        rng,
        rate=0.12,
        families=FAMILIES,
        weights=WEIGHTS,
        speed=(1.0, 1000.0),
        dist=(5.0, 150.0),
        miss_mix=((0.4, 0.0, 0.5), (0.3, 0.5, 5.0), (0.3, 5.0, 15.0)),
        first=3.0,
    ):
        self.rng, self.rate, self.speed, self.dist, self.miss_mix = rng, rate, speed, dist, miss_mix
        w = np.array(
            weights if len(weights) == len(families) else [weights[FAMILIES.index(f)] for f in families],
            float,
        )
        self.families, self.w = list(families), w / w.sum()
        self.next_t = first + rng.exponential(1 / rate)
        self.active, self.log, self.n = [], [], 0

    def tick(self, t, dt, drone_p, drone_v, free_bodies):
        if t >= self.next_t and free_bodies:
            self.spawn(t, drone_p, drone_v, free_bodies)
            self.next_t = t + self.rng.exponential(1 / self.rate)
        for e in self.active:
            e.step(t, dt, drone_p, drone_v)
        for e in [e for e in self.active if e.done]:
            self.active.remove(e)
            self.log.append(self.record(e))

    def spawn(self, t, drone_p, drone_v, free_bodies):
        r = self.rng
        fam = str(r.choice(self.families, p=self.w))
        speed = float(math.exp(r.uniform(math.log(self.speed[0]), math.log(self.speed[1]))))
        T_want = float(r.uniform(0.8, 6.0))
        dist = float(
            np.clip(speed * T_want, *self.dist)
        )  # meet soon: gusts move the drone during long approaches
        want = SIZES[int(r.integers(len(SIZES)))]
        size, body = min(free_bodies, key=lambda sb: abs(sb[0] - want))
        k = r.choice(len(self.miss_mix), p=[m[0] for m in self.miss_mix])
        miss = float(r.uniform(*self.miss_mix[k][1:]))
        for _ in range(50):  # start direction uniform on the sphere, above the ground
            u = unit(r.normal(size=3))
            p0 = drone_p + u * dist
            if p0[2] > 0.3:
                break
        T = dist / speed
        pred = drone_p + drone_v * T  # where the drone would be if it kept flying the same way
        side = unit(np.cross(unit(pred - p0), r.normal(size=3)))
        aim = pred + side * miss
        e = Encounter(self.n, body, size, fam, speed, p0, t, aim, T, miss, r)
        self.n += 1
        self.active.append(e)

    @staticmethod
    def record(e):
        return {
            "id": e.id,
            "family": e.fam,
            "speed": round(e.speed, 1),
            "size": e.size,
            "t0": round(e.t0, 2),
            "T": round(e.T, 3),
            "planned_miss": round(e.miss, 2),
            "min_dist": round(e.min_dist, 2),
            "hit": e.hit,
        }
