"""Where the time of one world step goes: a mission of the G plan flown for SECONDS of simulated time under cProfile.
usage: profile_step.py PLAN.jsonl I [SECONDS=40]"""

import cProfile, json, pstats, sys, time

import eval_missions as EM, missions as M, city3d as C, mission_env as ME, mission_brain as MB, nav as NV, safety as SF
import perceive as PC
from pathlib import Path


def main(plan, i, seconds):
    r = next(json.loads(l) for l in open(plan, encoding="utf-8") if json.loads(l)["i"] == i)
    W0, _, _ = C.build(r["place"], r["seed"], actors=False)
    mis = EM.mk(M, W0, r)
    env = ME.MissionEnv(r["place"], r["seed"], mis, render=False, cam=(160, 120), gimbal=(960, 720), wind=tuple(r["wind"]),
                        gust=r["gust"], encounters=(dict(EM.ATTACKS) if r["attacks"] else None))
    models = Path(__file__).resolve().parents[1] / "models"
    perc = PC.Perceiver(str(models / "detector.pt"), str(models / "attrs_best.pt"))
    saf = SF.Safety(env, seed=r["seed"])
    nav = NV.Navigator(env.W.map, env.home, seed=r["seed"])
    brain = MB.Brain(env, env.W.map, r["parsed"], [tuple(p) for p in r["rough"]], perc, safety=saf, nav=nav)
    o = env.reset()
    t_act = t_step = 0.0
    n = 0
    pr = cProfile.Profile()
    pr.enable()
    w0 = time.time()
    while env.t < seconds and not brain.done:
        a = time.time()
        cmd = brain.act(o)
        b = time.time()
        o = env.step(cmd)
        t_act += b - a
        t_step += time.time() - b
        n += 1
    pr.disable()
    wall = time.time() - w0
    print(f"steps {n}, sim {env.t:.1f} s, wall {wall:.1f} s, x{env.t / wall:.2f}; per step: brain.act {1000 * t_act / n:.1f} ms, env.step {1000 * t_step / n:.1f} ms")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(30)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 40.0)
