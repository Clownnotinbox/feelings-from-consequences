"""Stage A (design sec."""

import json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

T_EP, LABEL_EVERY, HOR = 40.0, 3, 20  # seconds per episode, label every 3rd step, rollout 20 steps = 2 s
TARGETS = (
    "danger_05",
    "danger_1",
    "danger_2",
    "t_contact",
    "min_enc_dist",
    "gps_err",
    "gps_bad",
    "energy_margin",
    "low_energy",
    "strain",
    "nav_err",
    "lost",
)
DIAG = ("enc_d0", "enc_speed", "enc_seen", "near_enc_d", "enc_size")


class MissionPilot:
    """Behaviour policy for data collection: flies the mission legs with varied altitude / speed / care."""

    def __init__(self, env, rng):
        self.env, self.rng = env, rng
        hx, hy = env.home
        self.legs = []
        for leg in env.mission:
            if leg[0] == "goal":
                self.legs.append(np.array(env.W.targets[leg[1]].pos))
            elif leg[0] == "home":
                self.legs.append(np.array([hx, hy]))
            else:
                self.legs.append(np.array([hx + leg[1][0], hy + leg[1][1]]))
        self.cruise = float(rng.uniform(8, 45))
        self.speed = float(rng.uniform(3, 10))
        self.leg = 0
        self.p = np.array([hx, hy])  # the take-off point is known even without GPS
        self.careless = rng.random() < 0.2
        self.pause_until = -1.0
        if self.careless:
            self.cruise = float(rng.uniform(3, 9))  # low flight among buildings and trees

    def act(self, o, dt):
        g = o["gps"]
        t = o["t"]
        if g is not None and g.get("hdop", 9) < 5:
            self.p = np.array([g["x"], g["y"]])
        elif self.p is not None:
            self.p = self.p + np.array(o["vel_est"][:2]) * dt
        alt = o["baro_alt"]
        vz = float(np.clip(0.8 * (self.cruise - alt), -2.5, 3.0))
        if self.p is None or alt < min(self.cruise - 3, 6):
            return {"vx": 0.0, "vy": 0.0, "vz": vz}
        if t < self.pause_until:
            return {"vx": 0.0, "vy": 0.0, "vz": vz}
        if self.rng.random() < 0.005:
            self.pause_until = t + float(self.rng.uniform(1, 4))
            return {"vx": 0.0, "vy": 0.0, "vz": vz}  # short hold
        # (before 24.10 18:40 a pause re-extended itself every step, so after
        # data v1-v4 and the closed loop v1 were made with that pilot
        if self.leg >= len(self.legs):
            return {"vx": 0.0, "vy": 0.0, "vz": vz}
        e = self.legs[self.leg] - self.p
        d = float(np.linalg.norm(e))
        if d < 3:
            self.leg += 1
            return {"vx": 0.0, "vy": 0.0, "vz": vz}
        v = e / d * min(self.speed, 0.6 * d)
        if self.careless:
            v = v + self.rng.normal(0, 1.5, 2)  # wobbly, inattentive
        return {"vx": float(v[0]), "vy": float(v[1]), "vz": vz, "yaw_rate": float(self.rng.normal(0, 0.2))}

    def remaining_path(self, p_true):
        pts = [p_true[:2]] + self.legs[self.leg :]
        return float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))


ACTIONS = ("keep", "hover", "climb", "descend", "left", "right", "back")


def action_cmd(name, cmd, yaw):
    """Candidate fast actions."""
    vx, vy = cmd.get("vx", 0.0), cmd.get("vy", 0.0)
    sp = math.hypot(vx, vy)
    hx, hy = (vx / sp, vy / sp) if sp > 0.3 else (math.cos(yaw), math.sin(yaw))
    if name == "keep":
        return dict(cmd)
    if name == "hover":
        return {"vx": 0.0, "vy": 0.0, "vz": 0.0}
    if name == "climb":
        return {"vx": 0.3 * vx, "vy": 0.3 * vy, "vz": 3.0}
    if name == "descend":
        return {"vx": 0.3 * vx, "vy": 0.3 * vy, "vz": -2.0}
    if name == "left":
        return {"vx": -6 * hy, "vy": 6 * hx, "vz": 0.0}
    if name == "right":
        return {"vx": 6 * hy, "vy": -6 * hx, "vz": 0.0}
    return {"vx": -6 * hx, "vy": -6 * hy, "vz": 0.0}  # back


def action_markers(env, cmd, yaw, target_xy, keep_row):
    """For every candidate action."""
    out = [keep_row]
    for name in ACTIONS[1:]:
        f = env.fork()
        t0 = f.t
        n0 = len(f.hit_events)
        c = action_cmd(name, cmd, yaw)
        d0 = float(np.linalg.norm(f.data.qpos[:2] - target_xy)) if target_xy is not None else 0.0
        for _ in range(HOR):
            f.step(c, sense=False)
            if len(f.hit_events) > n0:
                break
        new = f.hit_events[n0:]
        tc = (new[0]["t"] - t0) if new else 3.0
        prog = (d0 - float(np.linalg.norm(f.data.qpos[:2] - target_xy))) if target_xy is not None else 0.0
        out.append([float(tc <= 2.0), min(tc, 3.0), prog])
    return np.array(out, np.float32)


def rollout_targets(env, cmd, target_xy=None):
    f = env.fork()
    t0 = f.t
    n0 = len(f.hit_events)
    s0 = f.strain_steps
    min_enc = 20.0
    d0 = float(np.linalg.norm(f.data.qpos[:2] - target_xy)) if target_xy is not None else 0.0
    for _ in range(HOR):
        f.step(cmd, sense=False)
        p = f.data.qpos[:3]
        if f.enc is not None:
            for e in f.enc.active:
                min_enc = min(min_enc, float(np.linalg.norm(e.p - p)))
        if len(f.hit_events) > n0:
            break
    new = f.hit_events[n0:]
    tc = (new[0]["t"] - t0) if new else 3.0
    strain = (f.strain_steps - s0) / max(1, HOR * f.sub)
    prog = (d0 - float(np.linalg.norm(f.data.qpos[:2] - target_xy))) if target_xy is not None else 0.0
    return {
        "danger_05": float(tc <= 0.5),
        "danger_1": float(tc <= 1.0),
        "danger_2": float(tc <= 2.0),
        "t_contact": min(tc, 3.0),
        "min_enc_dist": min_enc,
        "strain": strain,
        "contact_kind": new[0]["kind"] if new else "",
        "first": new[0] if new else None,
        "progress": prog,
    }


def diagnostics(env, first):
    """Truth about the encounter behind a rollout contact, for stratified evaluation."""
    p = env.data.qpos[:3]
    act = env.enc.active if env.enc is not None else []
    near = min([float(np.linalg.norm(e.p - p)) for e in act], default=999.0)
    if first is None or first["kind"] != "encounter":
        return [np.nan, np.nan, np.nan, near, np.nan]
    e = next((e for e in act if e.id == first["id"]), None)
    if e is None:
        return [
            np.nan,
            first["speed"],
            np.nan,
            near,
            np.nan,
        ]  # spawned during the rollout: not in the world yet
    gid = getattr(env, "_last_lidar_gid", np.zeros(0, int))
    m = env.model
    seen = float((m.body_mocapid[m.geom_bodyid[gid]] == e.body).sum()) if len(gid) else 0.0
    return [float(np.linalg.norm(e.p - p)), float(e.speed), seen, near, float(e.size)]


def episode(args):
    split, i, out_dir = args
    import drone_env, scenario, perception, features
    import zlib  # stable seed per dataset/episode

    rng = np.random.default_rng(zlib.crc32(f"{Path(out_dir).name}/{split}/{i}".encode()))
    cfg = scenario.sample_episode(rng, split)
    env = drone_env.DroneEnv(render=False, **cfg)
    o = env.reset()
    pilot = MissionPilot(env, rng)
    trk, rfx, st = (
        perception.make_tracker(),
        (perception.Reflex() if rng.random() < 0.5 else None),
        features.FeatureState(env.agent_dt),
    )
    X, Y, KIND, IDX, ACT, AIDX, D = [], [], [], [], [], [], []
    t_start = time.time()  # X: EVERY frame (0.1 s); Y: every 3rd frame, at IDX
    import os

    with_actions = bool(int(os.environ.get("STAGE_A_ACTIONS", "0")))
    act_every = int(os.environ.get("STAGE_A_ACT_EVERY", "2"))
    for k in range(int(T_EP / env.agent_dt)):
        sm = trk.step(o)
        cmd = pilot.act(o, env.agent_dt)
        if rfx is not None:
            cmd = rfx.act(o["t"], sm, cmd)
        x = features.build(o, sm, st, cmd)
        X.append(x)  # features include a copy of the command about to be executed
        if k % LABEL_EVERY == 0 and k > 10:
            tgt = pilot.legs[pilot.leg] if pilot.leg < len(pilot.legs) else None
            y = rollout_targets(env, cmd, tgt)
            p = env.data.qpos[:3]
            g = o["gps"]
            err = (
                float(np.linalg.norm(np.array([g["x"], g["y"]]) - p[:2])) if g is not None else float("nan")
            )  # horizontal error matters for navigation
            rate = max((env.battery0 - env.battery) / max(env.t, 1.0), 0.02)  # %/s, current drain
            margin = env.battery - rate * pilot.remaining_path(p) / pilot.speed
            nav = float(np.linalg.norm(pilot.p - p[:2]))  # error of the drone's own position estimate
            y.update(
                gps_err=err,
                gps_bad=float(err > 5.0) if g is not None else float("nan"),
                energy_margin=margin,
                low_energy=float(margin < 15.0),
                nav_err=nav,
                lost=float(nav > 10.0),
            )
            Y.append([y[t] for t in TARGETS])
            KIND.append(y["contact_kind"])
            IDX.append(k)
            D.append(diagnostics(env, y["first"]))
            if with_actions and len(Y) % act_every == 0:
                keep = [y["danger_2"], y["t_contact"], y["progress"]]
                ACT.append(action_markers(env, cmd, o["attitude"]["yaw"], tgt, keep))
                AIDX.append(len(Y) - 1)
        o = env.step(cmd)
    tr = env.truth()
    name = f"{split}_{i:05d}"
    extra = (
        {"A": np.array(ACT, np.float32), "aidx": np.array(AIDX)} if with_actions else {}
    )  # A: rows of Y at aidx x actions x (contact_2s, t_contact, progress)
    np.savez_compressed(
        Path(out_dir) / f"{name}.npz",
        X=np.array(X, np.float32),
        Y=np.array(Y, np.float32),
        idx=np.array(IDX),
        kind=np.array(KIND),
        D=np.array(D, np.float32),
        **extra,
    )
    meta = {
        "name": name,
        "tracker": type(trk).__name__,
        "split": split,
        "place": cfg["place"] or f"procedural#{cfg['seed']}",
        "mission": [str(l) for l in cfg["mission"]],
        "faults": [f"{f['sensor']}:{f['kind']}" for f in cfg["faults"]],
        "reflex": rfx is not None,
        "careless": pilot.careless,
        "battery0": round(cfg["battery0"], 1),
        "hits": len(tr["hit_events"]),
        "encounters": len(tr["encounters"]),
        "enc_hits": sum(e["hit"] for e in tr["encounters"]),
        "labels": len(Y),
        "wall_s": round(time.time() - t_start, 1),
    }
    (Path(out_dir) / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    env.close()
    return meta


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "runs" / (sys.argv[5] if len(sys.argv) > 5 else "stageA")
    out.mkdir(parents=True, exist_ok=True)
    plan = [
        (s, i, str(out))
        for s, n in (
            ("train", int(sys.argv[1])),
            ("test_familiar", int(sys.argv[2])),
            ("test_unfamiliar", int(sys.argv[3])),
        )
        for i in range(n)
        if not (out / f"{s}_{i:05d}.npz").exists()
    ]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        for meta in ex.map(episode, plan):
            done += 1
            if done % 10 == 0 or done == len(plan):
                print(f"{done}/{len(plan)} {round(time.time() - t0)}s last {meta}", flush=True)
