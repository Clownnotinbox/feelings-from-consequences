"""Stage F: fly one mission with the brain, end to end.
usage: fly_mission.py PLACE SEED [--perfect] [--det D --attrs A --parser P --base B] [--gps-off T] [--video OUT.mp4]
"""

import argparse, json, math, sys, time, zlib
import numpy as np


class TruthPerceiver:
    """Debug stand-in: detections from the simulator's label image."""

    def __init__(self, env, rng):
        import grounding as G

        self.env, self.rng, self.G = env, rng, G
        m = env.model
        self.lab = np.zeros(m.ngeom + 1, np.int32)
        self.objs = env.objs
        names = {o.oid: k + 1 for k, o in enumerate(env.objs)}
        for g in range(m.ngeom):
            n = m.geom(g).name or ""
            if n.split("_")[0] in names:
                self.lab[g] = names[n.split("_")[0]]
        self.seg = None

    def __call__(self, rgb, depth, cam, fovy):
        import mujoco

        env = self.env
        if self.seg is None:  # labels: a renderer without multisampling
            env.model.vis.quality.offsamples = 0
            self.seg = mujoco.Renderer(env.model, env.gh // 2, env.gw // 2, max_geom=30000)
            env.model.vis.quality.offsamples = 4
            self.seg.enable_segmentation_rendering()
        self.seg.update_scene(env.data, camera="gimbal")
        s = self.seg.render().copy()
        gid = s[..., 0]
        lab = np.where(
            (s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0),
            self.lab[np.clip(gid, 0, env.model.ngeom)],
            0,
        )
        out = []
        pos = cam[0]
        for k in np.unique(lab):
            if k == 0:
                continue
            ys, xs = np.nonzero(lab == k)
            px = 2 * min(xs.max() - xs.min(), ys.max() - ys.min())
            if px < 12 or self.rng.random() < 0.05:
                continue
            o = self.objs[k - 1]
            h = self.G.perfect_hyp(o)
            dist = float(np.linalg.norm(np.array([o.x, o.y, o.z]) - pos))
            sd = 0.02 * dist + 0.2
            h["x"] += self.rng.normal(0, sd)
            h["y"] += self.rng.normal(0, sd)
            for a in h["attrs"]:
                v = h["attrs"][a] + self.rng.uniform(0, 0.25 if px > 30 else 0.6, len(h["attrs"][a]))
                h["attrs"][a] = v / v.sum()
            out.append(
                {
                    "cls": o.kind,
                    "conf": 0.9,
                    "x": h["x"],
                    "y": h["y"],
                    "z": o.z + 0.1,
                    "px": float(px),
                    "size": h["size"] * self.rng.uniform(0.85, 1.15),
                    "height": o.height,
                    "attrs": h["attrs"],
                }
            )
        return out


def run(a, log=print):
    import city3d as C, missions as M, mission_parser as MP, nav as NV
    from mission_env import MissionEnv
    from mission_brain import Brain

    W0, _, _ = C.build(a.place, a.seed, actors=False)
    rng = np.random.default_rng(zlib.crc32(f"fly/{a.place}/{a.seed}/{a.k}".encode()))
    mis = M.make(
        W0, rng, bank=a.bank, two_step=a.two_step if a.two_step is not None else None, detail=a.detail
    )
    faults = (
        [{"sensor": "gps", "kind": "off", "t0": a.gps_off, "t1": 1e9, "sev": 1.0}]
        if a.gps_off is not None
        else []
    )
    env = MissionEnv(
        a.place,
        a.seed,
        mis,
        render=False,
        cam=(160, 120),
        wind=tuple(rng.normal(0, 2.5, 2)),
        gust=float(rng.uniform(0, 1.5)),
        faults=faults,
    )
    log(f"mission [{mis['lang']}/{mis['detail']}]: {mis['text']}")
    if a.perfect:
        perc = TruthPerceiver(env, np.random.default_rng(1))
        parsed = mis["truth"]
    else:
        import perceive as PC, train_parser as TP, torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        perc = PC.Perceiver(a.det, a.attrs)
        tok = AutoTokenizer.from_pretrained(a.base)
        tok.pad_token = tok.pad_token or tok.eos_token
        lm = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda"), a.parser
        ).eval()
        parsed = MP.parse_output(TP.generate(lm, tok, [{"text": mis["text"]}], bs=1)[0])
    log("parsed: " + json.dumps(parsed, ensure_ascii=False))
    import safety as SF

    saf = None if getattr(a, "no_safety", False) else SF.Safety(env)
    nav = NV.Navigator(env.W.map, env.home, seed=a.seed)
    brain = Brain(env, env.W.map, parsed, mis["rough"], perc, safety=saf, nav=nav, log=log)
    o = env.reset()
    t0 = time.time()
    errs = []
    while env.t < a.tmax and not brain.done:
        cmd = brain.act(o)
        o = env.step(cmd)
        p = env.data.qpos[:2]
        errs.append(math.hypot(brain.pos[0] - p[0], brain.pos[1] - p[1]))
    stop = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
    t_end = env.t + 30.0
    while env.cargo["state"] not in ("stowed", "landed") and env.t < t_end:
        o = env.step(stop if brain.state in ("done", "landed") else brain.act(o))  # let a released cargo land
    r = env.result()
    r["nav_err_mean"] = round(float(np.mean(errs)), 2)
    r["nav_err_max"] = round(float(np.max(errs)), 2)
    r["t"] = round(env.t, 1)
    r["wall"] = round(time.time() - t0, 1)
    r["brain_state"] = brain.state
    r["fires"] = saf.fires if saf else 0
    r["skipped"] = brain.skipped
    log("result: " + json.dumps(r, ensure_ascii=False, default=str))
    env.close()
    return r, mis, brain


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("place")
    ap.add_argument("seed", type=int)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--perfect", action="store_true")
    ap.add_argument("--det")
    ap.add_argument("--attrs")
    ap.add_argument("--parser")
    ap.add_argument("--base")
    ap.add_argument("--gps-off", type=float)
    ap.add_argument("--tmax", type=float, default=420.0)
    ap.add_argument("--bank", default="train")
    ap.add_argument("--detail")
    ap.add_argument("--two-step", type=int)
    a = ap.parse_args()
    a.two_step = None if a.two_step is None else bool(a.two_step)
    run(a)
