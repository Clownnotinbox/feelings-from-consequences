"""Stage D3 pre-check.
usage: d3_pilot.py N WORKERS OUT.json"""

import json, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np


def one(i):
    import drone_env, scenario, messages, stage_d as S

    rng = np.random.default_rng(zlib.crc32(f"pilotD3/train/{i}".encode()))
    cfg = scenario.sample_episode(rng, "train", seconds=S.T_FLIGHT)
    cfg["encounters"] = None
    env = drone_env.DroneEnv(render=False, battery_cutoff=True, **cfg)
    o = env.reset()
    pilot = S.make_pilot_class()(env, rng)
    pilot.arrivals = []
    prog = S.Progress(pilot.legs)
    low = bool(rng.random() < 0.5)
    if low:
        need = S.DRAIN * pilot.remaining_path(np.r_[np.array(env.home), 0.0]) / pilot.speed + 2.0
        env.battery = env.battery0 = max(6.0, float(need / rng.uniform(0.6, 1.6)))
    t_dec = S.DECIDE_AT[0] + float(rng.uniform(-2, 2))
    mrng = np.random.default_rng(zlib.crc32(f"pilotD3/train/{i}/msg".encode()))
    m = None
    while m is None:
        m = messages.sample3(mrng, t_dec)
    m["true"] = True
    messages.apply(env, m)
    out = {"i": i, "low": low, "parse": m["parse"], "text": m["text"], "text_novel": m["text_novel"]}
    for k in range(int(S.T_FLIGHT / env.agent_dt)):
        cmd = pilot.act(o, env.agent_dt)
        if o["t"] >= t_dec:
            if not (env.data.qpos[2] > 2.0 and not pilot.landing and pilot.leg < len(pilot.legs)):
                return out | {"skip": True}
            no = env.fork()
            no.world_events = []
            no.gps_faults = [g for g in env.gps_faults if g != (m["t_event"], 1e9, "off")]
            for tag, w in (("ev", env), ("no", no)):
                R = {op: S.rollout(w, pilot, prog, op, S.T_FLIGHT) for op in S.OPTIONS}
                best = max(S.score(r) for r in R.values())
                out["best_" + tag] = [op for op in S.OPTIONS if S.score(R[op]) == best]
                out["out_" + tag] = R
            env.close()
            return out
        o = env.step(cmd)
        prog.update(env.data.qpos)
    return out | {"skip": True}


def summary(rows):
    rows = [r for r in rows if not r.get("skip")]
    res = {"n": len(rows)}

    def stats(rs):
        if not rs:
            return None
        dis = np.mean(
            [not set(r["best_ev"]) & set(r["best_no"]) for r in rs]
        )  # every no-event-best plan is wrong once the event comes
        ign = np.mean(
            [r["best_no"][0] not in r["best_ev"] for r in rs]
        )  # a fixed tie-break of the ignoring decider
        cont = np.mean(["continue" in r["best_ev"] for r in rs]) - np.mean(
            ["continue" in r["best_no"] for r in rs]
        )
        return {
            "n": len(rs),
            "disjoint": round(float(dis), 3),
            "ignore_wrong": round(float(ign), 3),
            "continue_best_change": round(float(cont), 3),
        }

    res["all"] = stats(rows)
    for k in ("wind", "cold", "gps_outage"):
        res[k] = stats([r for r in rows if r["parse"]["kind"] == k])
    res["wind>=12"] = stats(
        [r for r in rows if r["parse"]["kind"] == "wind" and r["parse"]["wind_mps"] >= 12]
    )
    res["wind<8"] = stats([r for r in rows if r["parse"]["kind"] == "wind" and r["parse"]["wind_mps"] < 8])
    res["cold<=-10"] = stats(
        [r for r in rows if r["parse"]["kind"] == "cold" and r["parse"]["temp_c"] <= -10]
    )
    res["cold>0"] = stats([r for r in rows if r["parse"]["kind"] == "cold" and r["parse"]["temp_c"] > 0])
    res["low_battery"] = stats([r for r in rows if r["low"]])
    res["full_battery"] = stats([r for r in rows if not r["low"]])
    return res


if __name__ == "__main__":
    n, workers, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(one, range(n))):
            rows.append(r)
            if (k + 1) % 10 == 0:
                print(f"{k + 1}/{n} {round(time.time() - t0)}s", flush=True)
    s = summary(rows)
    out.write_text(json.dumps({"summary": s, "rows": rows}), encoding="utf-8")
    print(json.dumps(s, indent=1))
