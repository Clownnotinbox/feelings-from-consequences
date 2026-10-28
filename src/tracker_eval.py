"""How good is the brain's tracker? Scored against the simulator's truth.
usage: tracker_eval.py N_FLIGHTS [tracker_module_attr]"""

import sys, time
from pathlib import Path
import numpy as np


def run(n=6, make_tracker=None, seconds=30.0, split="test_familiar"):
    import zlib, drone_env, scenario, perception, stage_a

    make_tracker = make_tracker or perception.Tracker
    S = {
        "mov_tracks": 0,
        "mov_true": 0,
        "bodies": 0,
        "bodies_found": 0,
        "frames": 0,
        "threat_fp": 0,
        "threat_frames": 0,
        "wall": 0.0,
        "fast": 0,
        "fast_found": 0,
        "false_d": [],
    }
    for i in range(n):
        rng = np.random.default_rng(zlib.crc32(f"tracker_eval/{split}/{i}".encode()))
        cfg = scenario.sample_episode(rng, split)
        env = drone_env.DroneEnv(render=False, **cfg)
        o = env.reset()
        pilot = stage_a.MissionPilot(env, rng)
        trk = make_tracker()
        m = env.model
        dyn = sorted(env.mover_bodies | env.proj_bodies)
        t0 = time.time()
        Bprev = None
        for k in range(int(seconds / env.agent_dt)):
            t1 = time.time()
            sm = trk.step(o)
            S["wall"] += time.time() - t1
            p = env.data.qpos[:3].copy()
            B = env.data.xpos[dyn].copy()
            vel = np.zeros(len(dyn)) if Bprev is None else np.linalg.norm(B - Bprev, axis=1) / env.agent_dt
            Bprev = B
            gid = env._last_lidar_gid
            pts_body = m.geom_bodyid[gid] if len(gid) else np.zeros(0, int)
            npts = np.array([(pts_body == b).sum() for b in dyn])
            dist = np.linalg.norm(B - p, axis=1)
            mov = [s for s in sm["tracks"] if s["moving"]]
            S["frames"] += 1
            for s in mov:
                q = p + np.array(s["rel_pos"])
                S["mov_tracks"] += 1
                ok = bool((np.linalg.norm(B - q, axis=1) < 2.0).any()) if len(B) else False
                S["mov_true"] += int(ok)
                if not ok:
                    S["false_d"].append(s["dist"])
            vis = (npts >= 3) & (dist < 40)
            for j in np.nonzero(vis)[0]:
                f_ = int(any(np.linalg.norm(p + np.array(s["rel_pos"]) - B[j]) < 2.0 for s in mov))
                S["bodies"] += 1
                S["bodies_found"] += f_
                if vel[j] > 4.0:
                    S["fast"] += 1
                    S["fast_found"] += f_
            thr = any(s["threat"] for s in sm["tracks"])
            if thr:
                S["threat_frames"] += 1
            if thr and not (dist < 15).any():
                S["threat_fp"] += 1
            o = env.step(pilot.act(o, env.agent_dt))
        env.close()
    return {
        "precision": round(S["mov_true"] / max(S["mov_tracks"], 1), 3),
        "recall": round(S["bodies_found"] / max(S["bodies"], 1), 3),
        "moving_tracks_per_frame": round(S["mov_tracks"] / S["frames"], 2),
        "threat_frames_share": round(S["threat_frames"] / S["frames"], 3),
        "threat_without_mover_within_15m": round(S["threat_fp"] / S["frames"], 3),
        "visible_bodies": S["bodies"],
        "recall_fast": round(S["fast_found"] / max(S["fast"], 1), 3),
        "visible_fast": S["fast"],
        "false_track_dist_q": (
            np.percentile(S["false_d"], [10, 50, 90]).round(1).tolist() if S["false_d"] else None
        ),
        "ms_per_frame": round(1000 * S["wall"] / S["frames"], 1),
    }


if __name__ == "__main__":
    import json, perception

    mk = getattr(perception, sys.argv[2]) if len(sys.argv) > 2 else None
    split = sys.argv[3] if len(sys.argv) > 3 else "test_familiar"
    r = run(int(sys.argv[1]) if len(sys.argv) > 1 else 6, mk, split=split)
    print(json.dumps(r))
    if len(sys.argv) > 4:
        Path(sys.argv[4]).write_text(
            json.dumps({"tracker": sys.argv[2], "split": split, "flights": int(sys.argv[1]), **r}, indent=1),
            encoding="utf-8",
        )
