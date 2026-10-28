"""Random episode sampler."""

import math
import numpy as np

SPLITS = {
    # world seeds, threat families, fault kinds
    "train": dict(
        seeds=(0, 9000),
        places=("spb_center", "akadem", "malakhovka", "malakhovka_line"),
        families=("straight", "ballistic", "zigzag", "stopper"),
        faults=(
            "gps_off",
            "gps_drift",
            "gps_degraded",
            "lidar_range",
            "lidar_sector",
            "flow_off",
            "rain",
            "link_off",
        ),
    ),
    "test_familiar": dict(
        seeds=(9000, 10000),
        places=("spb_center", "akadem", "malakhovka", "malakhovka_line"),
        families=("straight", "ballistic", "zigzag", "stopper"),
        faults=(
            "gps_off",
            "gps_drift",
            "gps_degraded",
            "lidar_range",
            "lidar_sector",
            "flow_off",
            "rain",
            "link_off",
        ),
    ),
    "test_unfamiliar": dict(
        seeds=(10000, 11000),
        places=("tokyo_res", "akadem_line"),
        families=("pursuit", "erratic"),  # never seen in training
        faults=("gps_stuck", "compass_bias", "lidar_spurious", "imu_bias", "lidar_off"),
    ),
}


def sample_faults(rng, kinds, seconds, rate=1.5 / 60):
    out, t = [], rng.exponential(1 / rate)
    while t < seconds:
        k = str(rng.choice(kinds))
        dur = float(rng.uniform(3, 25))
        t1 = t + dur
        if k.startswith("gps_"):
            out.append(
                {
                    "sensor": "gps",
                    "kind": k[4:] if k != "gps_degraded" else "degraded",
                    "t0": t,
                    "t1": t1,
                    "sev": 1.0,
                }
            )
        elif k == "lidar_range":
            out.append(
                {"sensor": "lidar", "kind": "range", "t0": t, "t1": t1, "sev": float(rng.uniform(0.2, 0.8))}
            )
        elif k == "lidar_sector":
            out.append(
                {
                    "sensor": "lidar",
                    "kind": "sector",
                    "t0": t,
                    "t1": t1,
                    "sev": float(rng.uniform(0.15, 0.6)),
                    "center": float(rng.uniform(-math.pi, math.pi)),
                }
            )
        elif k == "lidar_spurious":
            out.append(
                {
                    "sensor": "lidar",
                    "kind": "spurious",
                    "t0": t,
                    "t1": t1,
                    "sev": float(rng.integers(20, 300)),
                }
            )
        elif k == "lidar_off":
            out.append({"sensor": "lidar", "kind": "off", "t0": t, "t1": t + min(dur, 4.0), "sev": 1.0})
        elif k == "flow_off":
            out.append({"sensor": "flow", "kind": "off", "t0": t, "t1": t1, "sev": 1.0})
        elif k == "compass_bias":
            out.append(
                {"sensor": "compass", "kind": "bias", "t0": t, "t1": t1, "sev": float(rng.uniform(-0.7, 0.7))}
            )
        elif k == "imu_bias":
            out.append(
                {"sensor": "imu", "kind": "bias", "t0": t, "t1": t1, "sev": float(rng.uniform(0.05, 0.3))}
            )
        elif k == "link_off":
            out.append({"sensor": "link", "kind": "off", "t0": t, "t1": t1, "sev": 1.0})
        elif k == "rain":  # correlated: lidar range, optical flow
            out += [
                {
                    "sensor": "lidar",
                    "kind": "range",
                    "t0": t,
                    "t1": t1,
                    "sev": float(rng.uniform(0.4, 0.7)),
                    "cause": "rain",
                },
                {"sensor": "flow", "kind": "off", "t0": t, "t1": t1, "sev": 1.0, "cause": "rain"},
            ]
        t = t1 + rng.exponential(1 / rate)
    return out


def sample_mission(rng):
    """Different from/to every time: one goal; goal and back; goal, an intermediate point and back home."""
    g = int(rng.integers(0, 4))
    r = rng.random()
    if r < 0.5:
        return [("goal", g)]
    if r < 0.75:
        return [("goal", g), ("home",)]
    a = rng.uniform(0, 2 * math.pi)
    d = rng.uniform(30, 120)
    return [("goal", g), ("point", (round(d * math.cos(a), 1), round(d * math.sin(a), 1))), ("home",)]


def sample_episode(rng, split="train", seconds=40.0, p_real=0.5):
    S = SPLITS[split]
    ang = rng.uniform(0, 2 * math.pi)
    w = float(rng.uniform(0, 10))
    place = (
        str(rng.choice(S["places"])) if rng.random() < p_real else None
    )  # half real places, half procedural worlds
    faults = sample_faults(rng, S["faults"], seconds)
    if rng.random() < 0.15:
        faults.append(
            {"sensor": "gps", "kind": "off", "t0": 0.0, "t1": 1e9, "sev": 1.0, "cause": "no_gps_at_all"}
        )  # jamming zone / tunnel / indoor
    vis = 0.0 if rng.random() < 0.7 else float(rng.uniform(30, 300))
    if any(f.get("cause") == "rain" for f in faults):
        vis = float(rng.uniform(40, 150)) if vis == 0 else min(vis, 150.0)
    mission = sample_mission(rng)
    battery0 = float(rng.uniform(8, 25)) if rng.random() < 0.3 else float(rng.uniform(25, 100))
    return dict(
        seed=int(rng.integers(*S["seeds"])),
        place=place,
        start="random",
        mission=mission,
        goal=mission[0][1],
        battery0=battery0,
        wind=(w * math.cos(ang), w * math.sin(ang)),
        gust=float(rng.uniform(0, 3)),
        visibility=vis,
        flow=bool(rng.random() < 0.8),
        phys=dict(
            mass_scale=float(rng.uniform(0.85, 1.15)),
            thrust_scale=float(rng.uniform(0.9, 1.1)),
            drag_scale=float(rng.uniform(0.6, 1.6)),
            lidar_latency=float(rng.uniform(0.02, 0.1)),
        ),
        faults=faults,
        encounters=dict(
            families=S["families"], weights=[1.0] * len(S["families"]), rate=float(rng.uniform(0.08, 0.2))
        ),
    )
