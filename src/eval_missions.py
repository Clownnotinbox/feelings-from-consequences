"""Stage F, checks F4-F5.
usage: eval_missions.py parse ADAPTER BASE PLAN.jsonl N [TAG] [WORKERS]   (make the plan: missions + parses; ADAPTER '-' = no parser)
       eval_missions.py fly DET ATTRS PLAN.jsonl OUT.jsonl WORKERS [variants]"""

import json, math, os, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np

PLACES = ("tokyo_res", "kazan_novosav")
ACTS = (
    "land",
    "drop",
    "pickup",
    "enter",
    "inspect",
    "flyby",
    "pickup+drop",
)  # balanced (MISSIONS_DESIGN 6.1)
ATTACKS = {
    "families": ("pursuit", "straight", "zigzag", "stopper"),
    "speed": (5.0, 25.0),
    "rate": 0.05,
}  # birds: bird speeds
T_MAX = float(os.environ.get("EVAL_TMAX", 600))  # mission time limit, s (a smoke run sets it short)
SEED0 = {"F4": 2000, "G": 4000, "H": 6000}


def mission_cfg(i, tag="F4"):
    rng = np.random.default_rng(zlib.crc32(f"{tag}/{i}".encode()))
    place = PLACES[i % 2]
    seed = SEED0[tag] + i
    if tag != "H":
        return {
            "i": i,
            "tag": tag,
            "place": place,
            "seed": seed,
            "gps_off": float(rng.uniform(20, 120)) if rng.random() < 0.4 else None,
            "attacks": bool(rng.random() < 0.5),
            "wind": [float(v) for v in rng.normal(0, 3.0, 2)],
            "gust": float(rng.uniform(0, 1.5)),
            "bank": "test" if rng.random() < 0.67 else "train",
            "k": int(rng.integers(1 << 20)),
            "act": ACTS[i % len(ACTS)],
        }
    g = rng.random()
    ws, wa = float(rng.uniform(3.0, 9.0)), float(
        rng.uniform(0, 2 * math.pi)
    )  # hard (MISSIONS_DESIGN 6.6, G3)
    return {
        "i": i,
        "tag": tag,
        "place": place,
        "seed": seed,
        "gps_off": float(rng.uniform(15, 90)) if g < 0.5 else None,
        "gps_spoof": float(rng.uniform(15, 90)) if 0.5 <= g < 0.75 else None,
        "attacks": bool(rng.random() < 0.3),
        "wind": [ws * math.cos(wa), ws * math.sin(wa)],
        "gust": float(rng.uniform(0.5, 2.0)),
        "bank": "test" if rng.random() < 0.67 else "train",
        "k": int(rng.integers(1 << 20)),
        "act": ACTS[i % len(ACTS)],
        "battery0": float(rng.uniform(30.0, 65.0)),
        "cold": float(rng.uniform(1.15, 1.5)) if rng.random() < 0.6 else None,
    }


def mk(M, W, c):
    two = c["act"] == "pickup+drop"
    return M.make(
        W, np.random.default_rng(c["k"]), bank=c["bank"], two_step=two, action=None if two else c["act"]
    )


def _plan_row(args):
    i, tag = args
    import city3d as C, missions as M

    c = mission_cfg(i, tag)
    W, _, _ = C.build(c["place"], c["seed"], actors=False)
    mis = mk(M, W, c)
    if mis is None:
        return None
    return {
        **c,
        "text": mis["text"],
        "truth": mis["truth"],
        "detail": mis["detail"],
        "lang": mis["lang"],
        "steps": mis["steps"],
        "rough": mis["rough"],
    }


def plan(adapter, base, out, n, tag="F4", workers=1):
    import mission_parser as MP

    with ProcessPoolExecutor(workers) as ex:
        rows = [
            r for r in ex.map(_plan_row, [(i, tag) for i in range(n)]) if r is not None
        ]  # worlds and missions on many cores
    if adapter != "-":
        import train_parser as TP, torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(base)
        tok.pad_token = tok.pad_token or tok.eos_token
        lm = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda"), adapter
        ).eval()
        outs = TP.generate(lm, tok, rows, bs=16)
    else:
        outs = [None] * len(rows)
    with open(out, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            if o is None:
                r["parsed"], r["parse_exact"] = None, None
            else:
                p = MP.parse_output(o)
                r["parsed"] = p if MP.well_formed(p) else None
                r["parse_exact"] = MP.score(p, r["truth"])["exact"]
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    ex_ = [r["parse_exact"] for r in rows if r["parse_exact"] is not None]
    print("planned", len(rows), "parse exact", np.mean(ex_) if ex_ else None, flush=True)


class ReflexSafety:
    """Classic tracker reflex as the fast layer (no feeling core), the 'reflex' variant."""

    def __init__(self, env):
        import perception

        self.trk = perception.make_tracker()
        self.rfx = perception.Reflex()
        self.feel = {}
        self.fires = 0

    def __call__(self, o, cmd):
        sm = self.trk.step(o)
        self.last_sm = sm
        n = len(self.rfx.fired)
        new = self.rfx.act(o["t"], sm, cmd)
        fired = len(self.rfx.fired) > n
        self.fires += fired
        return new, fired


SLOW = {
    "full": "feelings",
    "operator": "feelings",
    "telemetry": "telemetry",
    "reflex": "telemetry",
    "slow_none": "none",
    "nolost": "feelings_nolost",
    "noenergy": "feelings_noenergy",
    "false_alarm": "feelings",
    "delayed": "feelings",
}


def fly_one(args):
    r, variant, det, attrs = args
    import mission_env as ME, missions as M, city3d as C, nav as NV, safety as SF, mission_brain as MB, perceive as PC, mission_score as MS

    global _PERC
    if "_PERC" not in globals():
        _PERC = PC.Perceiver(det, attrs)
    t0 = time.time()
    W0, _, _ = C.build(r["place"], r["seed"], actors=False)
    mis = mk(M, W0, r)
    faults = (
        [{"sensor": "gps", "kind": "off", "t0": r["gps_off"], "t1": 1e9, "sev": 1.0}] if r["gps_off"] else []
    )
    if r.get("gps_spoof"):
        faults.append(
            {"sensor": "gps", "kind": "drift", "t0": r["gps_spoof"], "t1": 1e9, "sev": 1.0}
        )  # spoofing: the fix walks away, still looks healthy
    env = ME.MissionEnv(
        r["place"],
        r["seed"],
        mis,
        render=False,
        cam=(160, 120),
        gimbal=(960, 720),
        wind=tuple(r["wind"]),
        gust=r["gust"],
        faults=faults,
        encounters=(dict(ATTACKS) if r["attacks"] else None),
        battery0=float(r.get("battery0") or 100.0),
    )
    out = {
        "i": r["i"],
        "variant": variant,
        "tag": r.get("tag", "F4"),
        "place": r["place"],
        "gps_off": r["gps_off"],
        "gps_spoof": r.get("gps_spoof"),
        "attacks": r["attacks"],
        "actions": [s["action"] for s in mis["steps"]],
        "parse_exact": r["parse_exact"],
        "bank": r["bank"],
        "detail": r["detail"],
        "lang": r["lang"],
        "battery0": r.get("battery0"),
        "cold": r.get("cold"),
    }
    hard = r.get("tag") == "H"
    parsed = (
        r["truth"] if (variant == "operator" or hard) else r["parsed"]
    )  # 'operator': a person checked the parse
    if hard:
        parsed = {**parsed, "return_home": True}  # H: the operator wants the drone back
    if parsed is None:
        out.update(status="no_parse", steps_ok=[False] * len(mis["steps"]))
        env.close()
        return out
    if variant == "reflex":
        saf = ReflexSafety(env)
    else:
        saf = SF.Safety(
            env,
            false_alarm=(20.0, 1.0) if variant == "false_alarm" else None,
            delay=3 if variant == "delayed" else 0,
            seed=r["seed"],
        )
    nav = NV.Navigator(env.W.map, env.home, seed=r["seed"])
    asks = []

    def operator(step, cands):  # sees the candidates' views and knows what was meant
        o = next(o for o in mis["objects"] if o.oid == mis["steps"][brain.si]["target"])
        asks.append(len(cands))
        hit = [
            k
            for k, (p_, h, _) in enumerate(cands)
            if math.hypot(h["x"] - o.x, h["y"] - o.y) < 3.0 + 0.5 * h["size"] + o.radius
        ]
        return hit[0] if hit else -1

    brain = MB.Brain(
        env,
        env.W.map,
        parsed,
        [tuple(p) for p in r["rough"]],
        _PERC,
        safety=saf,
        nav=nav,
        slow=SLOW[variant],
        ask=operator if variant == "operator" else None,
    )
    o = env.reset()
    errs = []
    track = []
    t_dead = None
    airborne_dead = False
    if r.get("cold"):
        env.world_events = [
            {"kind": "drain", "mult": r["cold"], "t_event": 0.0, "ramp_s": 20.0}
        ]  # a cold battery: capacity falls over the first 20 s
    while env.t < T_MAX and not brain.done:
        o = env.step(brain.act(o))
        p = env.data.qpos[:3]
        track.append((env.t, *map(float, p)))
        errs.append(math.hypot(brain.pos[0] - p[0], brain.pos[1] - p[1]))
        if t_dead is None and env.battery <= 0.0:
            t_dead = round(env.t, 1)
            airborne_dead = bool(p[2] > 0.5 and brain.state not in ("landed", "done"))
        if t_dead is not None and env.t - t_dead > 15.0:
            break  # the motors stopped: it has fallen
    stop = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
    t_end = env.t + 30.0
    while env.cargo["state"] not in ("stowed", "landed") and env.t < t_end:
        o = env.step(stop if brain.state in ("done", "landed") else brain.act(o))  # let a released cargo land
    res = MS.score(env, mis, track)
    hits = env.hit_events
    p = env.data.qpos[:3]
    notes = [n[1] for n in brain.notes]
    out.update(
        status=brain.state,
        steps_ok=[s["ok"] for s in res["steps"]],
        step_detail=res["steps"],
        wrong_target=res["wrong_target"],
        skipped=brain.skipped,
        rejected=len(brain.rejected),
        act=r["act"],
        t=round(env.t, 1),
        hits_static=sum(h["kind"] == "static" for h in hits),
        hits_bird=sum(h["kind"] == "mover" for h in hits),
        hits_attack=sum(h["kind"] == "encounter" for h in hits),
        nav_err=round(float(np.mean(errs)), 2) if errs else None,
        nav_err_max=round(float(np.max(errs)), 2) if errs else None,
        fires=saf.fires,
        battery=res["battery"],
        battery_dead=t_dead,
        dead_in_air=airborne_dead,
        max_contact_speed=round(float(env.max_contact_speed), 2),
        home_dist=res.get("home_dist"),
        home_landed=bool(
            brain.state == "landed" and res.get("home_dist", 99) < 3.0 and t_dead is None and brain.si == 99
        ),
        went_home_early=any(("energy" in n or "battery low" in n) for n in notes),
        lost_events=sum(("I am lost" in n or "position uncertain" in n) for n in notes),
        alarms=getattr(saf, "alarms", 0),
        notes=[n for n in notes if not n.startswith("-> ")][-14:],
        wall=round(time.time() - t0, 1),
        asks=len(asks),
    )
    env.close()
    return out


def fly_safe(args):
    """One flight; a crash becomes an 'error' row (one broken flight must not stop a run of hundreds)."""
    try:
        return fly_one(args)
    except Exception:
        import traceback

        return {
            "i": args[0]["i"],
            "variant": args[1],
            "status": "error",
            "error": traceback.format_exc()[-1500:],
        }


def fly(det, attrs, plan_path, out, workers, variants=("full", "telemetry", "reflex")):
    from concurrent.futures import as_completed

    rows = [json.loads(l) for l in open(plan_path, encoding="utf-8")]
    done = set()
    try:
        done = {
            (json.loads(l)["i"], json.loads(l)["variant"])
            for l in open(out, encoding="utf-8")
            if l.strip().endswith("}")
        }
    except FileNotFoundError:
        pass
    jobs = [(r, v, det, attrs) for r in rows for v in variants if (r["i"], v) not in done]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:  # results are written as they finish
        for k, fut in enumerate(as_completed([ex.submit(fly_safe, jb) for jb in jobs])):
            res = fut.result()
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False, default=str) + chr(10))
            if (k + 1) % 10 == 0:
                print(f"{k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("done", len(jobs), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "parse":
        plan(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            int(sys.argv[5]),
            sys.argv[6] if len(sys.argv) > 6 else "F4",
            int(sys.argv[7]) if len(sys.argv) > 7 else 1,
        )
    else:
        fly(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            sys.argv[5],
            int(sys.argv[6]),
            tuple(sys.argv[7].split(",")) if len(sys.argv) > 7 else ("full", "telemetry", "reflex"),
        )
