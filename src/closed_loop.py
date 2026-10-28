"""Closed-loop flights.
usage: closed_loop.py --core RUNTIME.npz --core-model logreg|gru [--markers MARKERS.runtime.npz] --n N --workers W
       --out OUT.json [--variants none,reflex,...]"""

import json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

T_FLIGHT, HOLD = 40.0, 0.6
VARIANTS = ("none", "reflex", "core_freeze@0.01", "core_freeze@0.05", "core_reflex@0.01", "core_reflex@0.05")
V4_VARIANTS = (
    "none",
    "reflex",
    "core_freeze@0.05",
    "core_stop@0.05",
    "markers_gru@fa5",
    "markers_fwd@fa5",
    "markers_fwd@fa10",
)


class CoreLayer:
    def __init__(self, core, thr, mode):
        self.core, self.thr, self.mode, self.until, self.kind = core, thr, mode, -1.0, None
        import perception

        self.rfx = perception.Reflex()

    def act(self, t, sm, x, cmd):
        p = self.core.step(x)["danger_1"]
        fired = False
        if self.mode == "core_stop" and p >= self.thr and t <= self.until and self.kind == "freeze":
            self.until = max(self.until, t + 0.1)  # stay stopped
        if p >= self.thr and t > self.until:
            fired = True
            self.until = t + HOLD
            self.kind = "freeze"
            if self.mode == "core_reflex":
                tr = [s for s in sm["tracks"] if s["t_cpa"] < 2.0]
                if tr:  # dodge away from the closest-approach point of the most urgent track
                    s = min(tr, key=lambda s: s["t_cpa"])
                    v_rel = np.array(s["vel_world"]) - np.array(sm["v_own"])
                    r = np.array(s["rel_pos"]) + v_rel * s["t_cpa"]
                    away = -r if np.linalg.norm(r) > 0.2 else np.cross(v_rel, [0, 0, 1.0]) + [0, 0, 0.3]
                    self.dir = away / max(np.linalg.norm(away), 1e-6)
                    self.kind = "dodge"
        if t <= self.until:
            if self.kind == "dodge":
                v = self.dir * 7.0
                return (
                    {
                        "vx": float(v[0]),
                        "vy": float(v[1]),
                        "vz": float(np.clip(v[2], -3, 4)),
                        "yaw_rate": 0.0,
                    },
                    fired,
                    p,
                )
            return {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}, fired, p
        return cmd, fired, p


class MarkerLayer:
    def __init__(self, core, thr):
        self.core, self.thr, self.until, self.cmd = core, thr, -1.0, None

    def act(self, t, sm, x, prop, yaw):
        import stage_a

        pc, _ = self.core.step(x)
        fired = False
        if pc[0] >= self.thr and t > self.until:
            a = 1 + int(np.argmin(pc[1:]))
            fired = True  # the safest ALTERNATIVE (same rule as train_markers.decisions)
            if fired:
                self.until = t + HOLD
                self.cmd = stage_a.action_cmd(stage_a.ACTIONS[a], prop, yaw)
                self.choice = stage_a.ACTIONS[a]
        if t <= self.until:
            return self.cmd, fired, float(pc[0])
        return prop, fired, float(pc[0])


def would_hit(env, cmd):
    """Truth check of a fire: does keeping the pilot's command lead to a contact within 2 s?."""
    f = env.fork()
    n0 = len(f.hit_events)
    for _ in range(20):
        f.step(cmd, sense=False)
        if len(f.hit_events) > n0:
            return True, f.hit_events[n0]["kind"]
    return False, ""


def flight(args):
    split, i, variant, C = args
    import drone_env, scenario, perception, features, stage_a, core_runtime
    import zlib  # a STABLE seed: Python's hash() of a string differs between processes

    rng = np.random.default_rng(zlib.crc32(f"closed_loop/{split}/{10000 + i}".encode()))
    cfg = scenario.sample_episode(rng, split)
    env = drone_env.DroneEnv(render=False, **cfg)
    o = env.reset()
    pilot = stage_a.MissionPilot(env, rng)
    trk, st = perception.make_tracker(), features.FeatureState(env.agent_dt)
    layer = None
    rfx = None
    if variant == "reflex":
        rfx = perception.Reflex()
    elif variant.startswith("core"):
        model = C["core_model"]
        core = (core_runtime.LogregCore if model == "logreg" else core_runtime.FeelingCore)(C["core"])
        base, fa = variant.split("@")
        layer = CoreLayer(core, core.thr[model][fa], base)
    elif variant.startswith("markers_fwd"):
        import markers_geom

        fa = variant.split("@")[1]
        mc = markers_geom.FwdMarkerCore(C["markers_fwd"])
        layer = MarkerLayer(mc, mc.thr["thr_" + fa])
    elif variant.startswith("markers"):
        base, fa = variant.split("@")
        mc = core_runtime.MarkerCore(C["markers"], base.split("_")[1])
        layer = MarkerLayer(mc, mc.thr["thr_" + fa])
    choices = []
    fire_log = []
    blocked = 0
    fires, overridden, t0 = [], 0, time.time()
    d_goal0 = env.truth()["dist_to_goal_xy"]
    for k in range(int(T_FLIGHT / env.agent_dt)):
        sm = trk.step(o)
        prop = pilot.act(o, env.agent_dt)
        cmd = prop
        if rfx is not None:
            n_before = len(rfx.fired)
            cmd = rfx.act(o["t"], sm, prop)
            if len(rfx.fired) > n_before or (cmd is not prop and not fires_open(fires, o["t"])):
                real, kind = would_hit(env, prop)
                fires.append({"t": round(o["t"], 1), "real": real, "kind": kind})
        elif layer is not None:
            x = features.build(o, sm, st, prop)
            if isinstance(layer, MarkerLayer):
                cmd, fired, p = layer.act(o["t"], sm, x, prop, o["attitude"]["yaw"])
            else:
                cmd, fired, p = layer.act(o["t"], sm, x, prop)
            if fired:
                real, kind = would_hit(env, prop)
                fires.append({"t": round(o["t"], 1), "real": real, "kind": kind, "p": round(p, 3)})
                fire_log.append((o["t"], env.data.qpos[:2].copy()))
                recent = [q for tt, q in fire_log if o["t"] - tt < 10.0]
                if len(recent) >= 3 and np.linalg.norm(recent[-1] - recent[-3]) < 3.0:
                    blocked += 1
                    fire_log.clear()
                if isinstance(layer, MarkerLayer):
                    choices.append(layer.choice)
        if cmd is not prop:
            overridden += 1
        o = env.step(cmd)
    tr = env.truth()
    hits = tr["hit_events"]
    env.close()
    return {
        "split": split,
        "i": i,
        "variant": variant,
        "place": cfg["place"] or "procedural",
        "careless": pilot.careless,
        "hits": len(hits),
        "hits_static": sum(h["kind"] == "static" for h in hits),
        "hits_mover": sum(h["kind"] == "mover" for h in hits),
        "hits_encounter": sum(h["kind"] == "encounter" for h in hits),
        "hits_throw": sum(h["kind"] == "throw" for h in hits),
        "legs_done": pilot.leg,
        "legs": len(pilot.legs),
        "fires": len(fires),
        "fires_real": sum(f["real"] for f in fires),
        "override_share": round(overridden / (T_FLIGHT / env.agent_dt), 4),
        "battery_end": round(tr["battery_pct"], 1),
        "goal_progress_m": round(d_goal0 - tr["dist_to_goal_xy"], 1),
        "goal_dist0_m": round(d_goal0, 1),
        "choices": {a: choices.count(a) for a in set(choices)},
        "blocked": blocked,
        "tracker": type(trk).__name__,
        "wall_s": round(time.time() - t0, 1),
    }


def fires_open(fires, t):
    """The classic reflex also brakes before structure without a 'fire' entry."""
    return bool(fires) and t - fires[-1]["t"] < HOLD


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--core")
    ap.add_argument("--core-model", default="logreg")
    ap.add_argument("--markers")
    ap.add_argument("--markers-fwd")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    out, workers = Path(a.out), a.workers
    if a.variants == "v4":
        a.variants = ",".join(V4_VARIANTS)
    C = {"core": a.core, "core_model": a.core_model, "markers": a.markers, "markers_fwd": a.markers_fwd}
    variants = a.variants.split(",")
    plan = [
        (s, i, v, C)
        for s in ("test_familiar", "test_unfamiliar")
        for i in range(a.offset, a.offset + a.n)
        for v in variants
    ]
    R = json.loads(out.read_text()) if out.exists() else []
    have = {(r["split"], r["i"], r["variant"]) for r in R}
    plan = [a for a in plan if (a[0], a[1], a[2]) not in have]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(flight, plan)):
            R.append(r)
            if (k + 1) % 10 == 0 or k + 1 == len(plan):
                out.write_text(json.dumps(R), encoding="utf-8")
                print(f"{k + 1}/{len(plan)} {round(time.time() - t0)}s", flush=True)
