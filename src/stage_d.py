"""Stage D data: SLOW decisions.
usage: stage_d.py N_TRAIN N_TEST_FAM N_TEST_UNF WORKERS OUTDIR"""

import json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

T_FLIGHT, DECIDE_AT, DRAIN = 120.0, (12.0, 30.0, 48.0, 66.0, 84.0), 0.072
MSG = os.environ.get("STAGE_D_MSG", "0")  # '1': stage D2 messages, '3': stage D3 messages (messages.sample3)
OPTIONS = ("continue", "return_home", "climb_continue", "hold_10s", "land_now")


def make_pilot_class():
    import stage_a

    class PlanPilot(stage_a.MissionPilot):
        """The mission pilot with plan options."""

        landing = False

        def apply(self, option, t):
            if option == "return_home":
                self.legs = self.legs[: self.leg] + [np.array(self.env.home, float)]
            elif option == "climb_continue":
                self.cruise = min(self.cruise + 25.0, 80.0)
            elif option == "hold_10s":
                self.pause_until = t + 10.0
            elif option == "land_now":
                self.landing = True

        def act(self, o, dt):
            if self.landing or self.leg >= len(self.legs):
                g = o["gps"]
                if g is not None and g.get("hdop", 9) < 5:
                    self.p = np.array([g["x"], g["y"]])
                else:
                    self.p = self.p + np.array(o["vel_est"][:2]) * dt
                rd = o["range_down"]
                return {"vx": 0.0, "vy": 0.0, "vz": -1.5 if (rd <= 0 or rd > 1.0) else -0.4}
            leg0 = self.leg
            c = super().act(o, dt)
            if self.leg > leg0:
                self.arrivals.append((leg0, o["t"]))
            return c

    return PlanPilot


class Progress:
    """Truth-side bookkeeping of the ORIGINAL mission: legs reached in order (true position within 5 m)."""

    def __init__(self, legs):
        self.legs, self.idx = [np.array(l, float) for l in legs], 0

    def update(self, p):
        while self.idx < len(self.legs) and np.linalg.norm(p[:2] - self.legs[self.idx]) < 5.0:
            self.idx += 1

    def next_dist(self, p):
        return 0.0 if self.idx >= len(self.legs) else float(np.linalg.norm(p[:2] - self.legs[self.idx]))


def rollout(env, pilot, prog, option, t_end):
    import copy

    f = env.fork()
    pl = copy.copy(pilot)
    pl.env = f
    pl.legs = [np.array(l) for l in pilot.legs]
    pl.p = pilot.p.copy()
    pl.rng = copy.deepcopy(pilot.rng)
    pl.arrivals = list(pilot.arrivals)
    pl.landing = pilot.landing
    pr = copy.deepcopy(prog)
    o = f.observe(lidar=False)
    pl.apply(option, f.t)
    n0 = len(f.hit_events)
    empty_air = False
    g0 = f.max_ground_speed
    while f.t < t_end:
        c = pl.act(o, f.agent_dt)
        o = f.step(c, sense="nolidar")
        pr.update(f.data.qpos)
        if f.battery <= 0 and f.data.qpos[2] > 0.5:
            empty_air = True
        if any(h.get("speed", 9.0) > 1.0 for h in f.hit_events[n0:]) or f.max_ground_speed > max(2.5, g0):
            break
        if (
            (pl.landing or pl.leg >= len(pl.legs))
            and f.data.qpos[2] < 0.35
            and np.linalg.norm(f.data.qvel[:3]) < 0.2
        ):
            break  # landed
    crash = any(h.get("speed", 9.0) > 1.0 for h in f.hit_events[n0:]) or f.max_ground_speed > max(2.5, g0)
    landed = bool(f.data.qpos[2] < 0.35)
    home = float(np.linalg.norm(f.data.qpos[:2] - np.array(f.home)))
    return {
        "crash": bool(crash),
        "empty_in_air": bool(empty_air),
        "legs_true": pr.idx,
        "legs_total": len(pr.legs),
        "home_end": bool((landed and home < 5.0) or pr.idx >= len(pr.legs)),
        "dist_next_m": round(pr.next_dist(f.data.qpos), 1),
        "nav_err_m": round(float(np.linalg.norm(pl.p - f.data.qpos[:2])), 1),
        "battery_end": round(float(f.battery), 2),
        "t_end": round(f.t, 1),
        "landed": landed,
    }


def score(r):
    return (
        not r["crash"],
        not r["empty_in_air"],
        r["legs_true"],
        r["home_end"],
        -int(r["dist_next_m"] // 5),
    )  # 5 m bins: smaller gaps are ties


def snapshot(o, pilot, st, env):
    """What the drone itself knows at the decision (raw telemetry for the 'T' condition). No truth."""
    g = o["gps"]
    tgt = pilot.legs[pilot.leg] if pilot.leg < len(pilot.legs) else None
    path = pilot.remaining_path(np.r_[pilot.p, 0.0]) if tgt is not None else 0.0
    return {
        "t": round(o["t"], 1),
        "gps": None if g is None else {"hdop": g.get("hdop"), "sats": g.get("sats")},
        "vel_source": o["vel_source"],
        "battery_pct": round(o["battery_pct"], 1),
        "baro_alt": round(o["baro_alt"], 1),
        "range_down": round(o["range_down"], 1),
        "wind": [round(w, 1) for w in o["wind"]["vec"]],
        "link_up": bool(o["link"]["up"]),
        "speed_cmd": round(pilot.speed, 1),
        "cruise_alt": round(pilot.cruise, 1),
        "own_pos": pilot.p.round(1).tolist(),
        "next_target": None if tgt is None else np.round(tgt, 1).tolist(),
        "remaining_path_m_own": round(path, 1),
        "legs_left": len(pilot.legs) - pilot.leg,
        "time_without_fix_s": round(st.since_fix_t, 1),
        "dist_without_fix_m": round(st.since_fix_d, 1),
        "home_dist_own_m": round(float(np.linalg.norm(pilot.p - np.array(env.home))), 1),
    }


def flight(args):
    split, i, out_dir = args
    import zlib, drone_env, scenario, perception, features, messages

    rng = np.random.default_rng(zlib.crc32(f"{Path(out_dir).name}/{split}/{i}".encode()))
    cfg = scenario.sample_episode(rng, split, seconds=T_FLIGHT)
    cfg["encounters"] = None
    env = drone_env.DroneEnv(render=False, battery_cutoff=True, **cfg)
    o = env.reset()
    Pilot = make_pilot_class()
    pilot = Pilot(env, rng)
    pilot.arrivals = []
    prog = Progress(pilot.legs)
    low = bool(rng.random() < 0.5)
    if low:  # energy for the remaining mission
        need = DRAIN * pilot.remaining_path(np.r_[np.array(env.home), 0.0]) / pilot.speed + 2.0
        env.battery = env.battery0 = max(6.0, float(need / rng.uniform(0.6, 1.6)))
    trk, st = perception.make_tracker(), features.FeatureState(env.agent_dt)
    X, D = [], []
    t0 = time.time()
    when = [t + float(rng.uniform(-2, 2)) for t in DECIDE_AT]
    mrng = np.random.default_rng(zlib.crc32(f"{Path(out_dir).name}/{split}/{i}/msg".encode()))
    msg = (
        messages.sample(mrng, when[0])
        if MSG == "1"
        else messages.sample3(mrng, when[0]) if MSG == "3" else None
    )
    messages.apply(env, msg)  # before any fork: every copy of the world inherits a true announced
    for k in range(int(T_FLIGHT / env.agent_dt)):
        sm = trk.step(o)
        cmd = pilot.act(o, env.agent_dt)
        X.append(features.build(o, sm, st, cmd))
        if when and o["t"] >= when[0]:
            when.pop(0)
            if env.data.qpos[2] > 2.0 and not pilot.landing and pilot.leg < len(pilot.legs):
                R = {op: rollout(env, pilot, prog, op, T_FLIGHT) for op in OPTIONS}
                best = max(score(r) for r in R.values())
                D.append(
                    {
                        "frame": k,
                        "snapshot": snapshot(o, pilot, st, env),
                        "outcomes": R,
                        "best": [op for op in OPTIONS if score(R[op]) == best],
                        "message": messages.seen(msg, o["t"]),
                    }
                )
                if rng.random() < 0.4:
                    pilot.apply(
                        str(rng.choice(OPTIONS[1:])), o["t"]
                    )  # behaviour policy: keep the states varied
        o = env.step(cmd)
        prog.update(env.data.qpos)
        if env.data.qpos[2] < 0.35 and (pilot.landing or pilot.leg >= len(pilot.legs)) and k > 50:
            break
    name = f"{split}_{i:05d}"
    env.close()
    np.savez_compressed(
        Path(out_dir) / f"{name}.npz",
        X=np.array(X, np.float32),
        frames=np.array([d["frame"] for d in D], int),
    )
    meta = {
        "name": name,
        "split": split,
        "place": cfg["place"] or f"procedural#{cfg['seed']}",
        "mission": [str(l) for l in cfg["mission"]],
        "faults": [f"{f['sensor']}:{f['kind']}" for f in cfg["faults"]],
        "low_battery_design": low,
        "battery0": round(env.battery0, 2),
        "decisions": D,
        "message": msg,
        "wall_s": round(time.time() - t0, 1),
    }
    (Path(out_dir) / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"name": name, "n_dec": len(D), "wall_s": meta["wall_s"], "best": [d["best"] for d in D]}


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "runs" / (sys.argv[5] if len(sys.argv) > 5 else "stageD")
    out.mkdir(parents=True, exist_ok=True)
    plan = [
        (s, i, str(out))
        for s, n in (
            ("train", int(sys.argv[1])),
            ("test_familiar", int(sys.argv[2])),
            ("test_unfamiliar", int(sys.argv[3])),
        )
        for i in range(n)
        if not (out / f"{s}_{i:05d}.json").exists()
    ]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(flight, plan)):
            if (k + 1) % 5 == 0 or k + 1 == len(plan):
                print(f"{k + 1}/{len(plan)} {round(time.time() - t0)}s last {r}", flush=True)
