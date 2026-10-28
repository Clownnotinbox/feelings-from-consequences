"""Brain-side feature vector for the feeling core."""

import math
from collections import deque
import numpy as np
import perception

SECT = 8
N_FEAT = 13 + 3 + 2 + 6 + 6 + 12 + 10 + 40 + 1 + 2 + 2 + 6


class FeatureState:
    """Small brain-side memory needed for derived cues (GPS innovation, looming rates)."""

    def __init__(self, dt=0.1):
        self.dt = dt
        self.gps_hist = deque(maxlen=11)
        self.dr = deque(maxlen=11)
        self.prev_sect = None
        self.pos_dr = np.zeros(3)
        self.since_fix_t = 0.0
        self.since_fix_d = 0.0  # time / distance flown without a trustworthy fix


def sector_minima(obs):
    P = obs["lidar_points"]
    a = obs["attitude"]
    R = perception.rot(a["roll"], a["pitch"], 0.0)  # heading frame (yaw removed)
    out = np.full(SECT + 2, 40.0)
    if len(P) == 0:
        return out, 0, 1.0
    Q = P @ R.T
    alt = obs["range_down"] if 0 < obs["range_down"] < 39 else obs["baro_alt"]
    Q = Q[Q[:, 2] > -alt + 0.4]
    d = np.linalg.norm(Q, axis=1)
    hor = np.abs(Q[:, 2]) < 2.0
    az = (np.arctan2(Q[:, 1], Q[:, 0]) + 2 * math.pi) % (2 * math.pi)
    sec = (az / (2 * math.pi / SECT)).astype(int) % SECT
    for k in range(SECT):
        m = hor & (sec == k)
        if m.any():
            out[k] = d[m].min()
    up, dn = Q[:, 2] > 2.0, Q[:, 2] < -2.0
    if up.any():
        out[SECT] = d[up].min()
    if dn.any():
        out[SECT + 1] = d[dn].min()
    return out, len(P), 0.0


def build(obs, summary, st, cmd=None):
    f = []
    v = np.array(obs["vel_est"])
    a = obs["attitude"]
    src = obs["vel_source"]
    f += list(v) + [
        src == "gps",
        src == "flow",
        src == "inertial",
        a["roll"],
        a["pitch"],
        math.sin(a["yaw"]),
        math.cos(a["yaw"]),
    ]
    f += list(obs["imu"]["gyro"]) + list(obs["imu"]["acc"])
    rd = obs["range_down"]
    f += [obs["baro_alt"] / 50.0, (rd if 0 < rd < 39 else 40.0) / 40.0]
    g = obs["gps"]
    st.pos_dr = st.pos_dr + v * st.dt
    st.dr.append(st.pos_dr.copy())
    if g is not None:
        gp = np.array([g["x"], g["y"], g["z"]])
        st.gps_hist.append(gp)
        if len(st.gps_hist) >= 2 and len(st.dr) >= 2:
            k = min(len(st.gps_hist), len(st.dr))
            dg = st.gps_hist[-1] - st.gps_hist[-k]
            dd = st.dr[-1] - st.dr[-k]
            innov = float(np.linalg.norm(dg - dd))
            spd = float(abs(np.linalg.norm(dg) - np.linalg.norm(dd)) / max(st.dt * (k - 1), 1e-6))
            stuck = float(np.linalg.norm(dg) < 0.05 and np.linalg.norm(v) > 1.0)
        else:
            innov = spd = stuck = 0.0
        f += [
            1.0,
            g.get("hdop", 1.0) / 10.0,
            g.get("sats", 10) / 14.0,
            min(innov, 30.0) / 10.0,
            min(spd, 20.0) / 10.0,
            stuck,
        ]
    else:
        st.gps_hist.clear()
        f += [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    w = obs["wind"]["vec"]
    lk = obs["link"]
    f += [
        w[0] / 10.0,
        w[1] / 10.0,
        w[2] / 10.0,
        obs["battery_pct"] / 100.0,
        float(lk["up"]),
        ((lk["rssi_dbm"] or -120) + 120) / 80.0,
    ]
    sm_, npts, _ = sector_minima(obs)
    f += list(sm_ / 40.0) + [npts / 40000.0, 0.0]
    if st.prev_sect is None:
        loom = np.zeros(SECT + 2)
    else:
        loom = np.clip((st.prev_sect - sm_) / st.dt / np.maximum(sm_, 0.5), -5, 5)  # approach rate / distance
    st.prev_sect = sm_
    f += list(loom)
    tr = sorted(
        summary["tracks"], key=lambda s: (-(1.0 / s["ttc"] if s["ttc"] else 0.0) / (1.0 + s["d_cpa"]))
    )[:5]
    yaw = a["yaw"]
    cy, sy = math.cos(-yaw), math.sin(-yaw)
    for k in range(5):
        if k < len(tr):
            s = tr[k]
            x, y, z = s["rel_pos"]
            f += [
                (cy * x - sy * y) / 40.0,
                (sy * x + cy * y) / 40.0,
                z / 40.0,
                s["dist"] / 40.0,
                s["closing"] / 30.0,
                min(1.0 / s["ttc"], 10.0) if s["ttc"] else 0.0,
                min(s["d_cpa"], 20.0) / 20.0,
                min(s["t_cpa"], 5.0) / 5.0,
            ]
        else:
            f += [0.0] * 8
    ts = summary["t_static"]
    f += [min(1.0 / ts, 10.0) if ts else 0.0]
    f += [
        sum(s["moving"] for s in summary["tracks"]) / 10.0,
        sum(s["threat"] for s in summary["tracks"]) / 5.0,
    ]
    if g is not None and g.get("hdop", 9) < 5:
        st.since_fix_t = 0.0
        st.since_fix_d = 0.0
    else:
        st.since_fix_t += st.dt
        st.since_fix_d += float(np.linalg.norm(v[:2])) * st.dt
    f += [min(st.since_fix_t, 120.0) / 60.0, min(st.since_fix_d, 400.0) / 100.0]
    c = cmd or {}
    cv = np.array([c.get("vx", 0.0), c.get("vy", 0.0), c.get("vz", 0.0)])
    sp = float(np.linalg.norm(cv[:2]))
    hd = (
        math.atan2(cv[1], cv[0]) - a["yaw"] if sp > 0.3 else 0.0
    )  # commanded heading in the heading frame of the lidar sectors
    secw = 2 * math.pi / SECT
    k0 = int((hd % (2 * math.pi)) // secw) % SECT  # sector k covers [k, k+1) * 45 deg
    d_cmd = min(sm_[(k0 - 1) % SECT], sm_[k0], sm_[(k0 + 1) % SECT]) if sp > 0.3 else 40.0
    if cv[2] > 0.3:
        d_cmd = min(d_cmd, sm_[SECT])
    if cv[2] < -0.3:
        d_cmd = min(d_cmd, sm_[SECT + 1])
    spd = max(float(np.linalg.norm(cv)), 0.1)
    f += [
        cv[0] / 10.0,
        cv[1] / 10.0,
        cv[2] / 4.0,
        c.get("yaw_rate", 0.0),
        d_cmd / 40.0,
        min(spd / d_cmd, 10.0),
    ]
    x = np.array(f, np.float32)
    assert len(x) == N_FEAT, len(x)
    return x
