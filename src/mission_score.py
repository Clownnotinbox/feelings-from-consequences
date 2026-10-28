"""World-side score of a whole mission."""

import math
import numpy as np


def _size(o):
    return float(o.facts.get("size_m") or 2 * o.radius)


def score(env, mis, track):
    """track: [(t, x, y, z)] true positions (one per agent step)."""
    res = env.result()
    objs = {o.oid: o for o in env.objs}
    T = np.asarray(track, float).reshape(-1, 4)
    for st, r in zip(mis["steps"], res["steps"]):
        o = objs[st["target"]]
        a = st["action"]
        p = st.get("params") or {}
        if a not in ("inspect", "flyby") or not len(T):
            continue
        dc = np.hypot(T[:, 1] - o.x, T[:, 2] - o.y)
        edge = dc - 0.5 * _size(o)
        if a == "inspect":
            m = np.abs(edge - float(p.get("distance_m") or 10.0)) <= 2.0
            bins = {int(b) for b in (np.degrees(np.arctan2(T[m, 2] - o.y, T[m, 1] - o.x)) % 360) // 10}
            r["sectors"] = len(bins)
            r["ok"] = len(bins) >= 30
        else:
            alt = p.get("altitude_m")
            m = (edge <= 15.0) & ((np.abs(T[:, 3] - float(alt)) <= 2.0) if alt else True)
            near = dc < 40.0
            seg = np.hypot(np.diff(T[:, 1]), np.diff(T[:, 2]))
            path = float(seg[near[1:] & near[:-1]].sum())
            r["closest_edge_m"] = round(float(edge.min()), 1)
            r["path_near_m"] = round(path, 1)
            r["ok"] = bool(m.any()) and path >= 30.0
    wrong = set()
    steps = [(st["action"], st["target"]) for st in mis["steps"]]
    for td in env.touchdowns:
        if td[1] == getattr(env, "ground_id", -1):
            continue  # setting down on bare ground touches no object
        oid = (env.model.geom(td[1]).name or "").split("_")[0]
        if oid in objs and (("land", oid) not in steps and ("enter", oid) not in steps):
            wrong.add(("touchdown", oid))
    if env.hook["carry"] and ("pickup", env.hook["carry"]) not in steps:
        wrong.add(("pickup", env.hook["carry"]))
    c = env.cargo
    if c["state"] == "landed":
        for o in objs.values():
            if (
                o.kind == "dropzone"
                and ("drop", o.oid) not in steps
                and math.hypot(c["p"][0] - o.x, c["p"][1] - o.y) < 0.5 * _size(o)
            ):
                wrong.add(("drop", o.oid))
    res["wrong_target"] = sorted(wrong)
    return res
