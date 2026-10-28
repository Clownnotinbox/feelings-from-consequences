"""Stage F target selection."""

import math
import numpy as np
from shapely.geometry import Point, LineString

EPS = 0.02
SIGMA_ROUGH = 45.0  # metres: operators' rough points are 10-80 m off (sometimes more)
MAX_ROUGH = 170.0  # the task's stated bound on the rough point
STYLE_WORD = {"panel": ("panel",), "brick": ("brick",), "glass": ("glass",), "wood": ("house_wood",)}


def fuse(dets, r=3.0):
    """Detections from many frames -> object hypotheses."""
    H = []
    for d in sorted(dets, key=lambda d: -d["px"]):
        w = min(1.0, d["px"] / 60.0) * d["conf"]
        for h in H:
            if math.hypot(h["x"] - d["x"], h["y"] - d["y"]) < r + 0.25 * max(h["size"], d["size"]):
                W = h["w"] + w
                h["x"] = (h["x"] * h["w"] + d["x"] * w) / W
                h["y"] = (h["y"] * h["w"] + d["y"] * w) / W
                h["z"] = (h["z"] * h["w"] + d["z"] * w) / W
                for k, p in d["attrs"].items():
                    h["attrs"][k] = (np.asarray(h["attrs"].get(k, p)) * h["w"] + np.asarray(p) * w) / W
                h["size"] = (h["size"] * h["w"] + d["size"] * w) / W
                h["height"] = max(h["height"], d.get("height", 0.0))
                h["w"] = W
                h["n"] += 1
                h["px"] = max(h["px"], d["px"])
                h["cls_w"][d["cls"]] = h["cls_w"].get(d["cls"], 0.0) + w
                h["cls"] = max(h["cls_w"], key=h["cls_w"].get)
                break
        else:
            H.append(
                {
                    "x": d["x"],
                    "y": d["y"],
                    "z": d["z"],
                    "cls": d["cls"],
                    "w": w,
                    "n": 1,
                    "px": d["px"],
                    "size": d["size"],
                    "height": d.get("height", 0.0),
                    "attrs": {k: np.asarray(v) for k, v in d["attrs"].items()},
                    "cls_w": {d["cls"]: w},
                }
            )
    return H


def soft(dist, near, far):
    """1 inside 'near', 0 beyond 'far', linear between."""
    return float(np.clip((far - dist) / (far - near), 0.0, 1.0))


def side_p(dx, dy, word):
    c = {"east": 0, "north": 90, "west": 180, "south": 270}[word]
    a = math.degrees(math.atan2(dy, dx)) % 360
    d = min(abs(a - c), 360 - abs(a - c))
    return float(np.clip((60 - d) / 30, 0, 1))


def building_matches(b, ref):
    """Does a MAP building fit what the text says about it?."""
    st = b["style"]
    t = b["type"]
    bt = (
        "church"
        if t in ("church", "cathedral", "chapel")
        else (
            "garage"
            if t in ("garage", "garages") or st == "garage"
            else (
                "warehouse"
                if st == "industrial"
                else (
                    "house" if st.startswith("house") else "office" if st.startswith("glass") else "apartment"
                )
            )
        )
    )
    if ref.get("btype") and ref["btype"] != bt and not (ref["btype"] == "apartment" and bt == "office"):
        return 0.0
    p = 1.0
    if ref.get("levels") and b.get("levels"):
        p *= (
            1.0
            if abs(ref["levels"] - b["levels"]) <= 0
            else 0.35 if abs(ref["levels"] - b["levels"]) == 1 else 0.02
        )
    if ref.get("material"):
        p *= (
            1.0
            if st.startswith(ref["material"]) or (ref["material"] == "wood" and st == "house_wood")
            else 0.05
        )
    if ref.get("colour"):
        p *= (
            1.0
            if ref["colour"] in st or (ref["colour"] == "white" and st in ("brick_silicate", "panel_light"))
            else 0.1
        )
    if ref.get("address"):
        s, n = ref["address"]
        p *= (
            1.0
            if (b.get("number") or "").lower() == str(n).lower() and street_eq(b.get("street"), s)
            else 0.01
        )
    return p


def street_eq(a, b):
    if not a or not b:
        return False
    norm = lambda s: "".join(ch for ch in s.lower().replace("ё", "е") if ch.isalnum())
    return norm(a) == norm(b) or norm(a) in norm(b) or norm(b) in norm(a)


def rel_p(h, rel, MAP, hyps):
    """Probability that relation `rel` holds for hypothesis h, from the MAP and the other hypotheses."""
    r, ref = rel["rel"], rel.get("ref", {})
    t = ref.get("type")
    p = Point(h["x"], h["y"])
    if t == "building":
        best = 0.0
        for b in MAP["buildings"]:
            m = building_matches(b, ref)
            if m < 0.01:
                continue
            d = b["poly"].distance(p)
            if r == "on_roof_of":
                v = 1.0 if b["poly"].buffer(0.5).contains(p) and h["z"] > 2.0 else 0.0
            elif r == "next_to":
                v = soft(d, 14, 26) * (h["z"] < 2.0)
            elif r.endswith("_of"):
                c = b["poly"].centroid
                v = side_p(h["x"] - c.x, h["y"] - c.y, r[:-3]) * soft(d, 40, 60) * (h["z"] < 2.0)
            else:
                v = 0.0
            best = max(best, m * v)
        return best
    if t == "open":
        return (
            soft(min((b["poly"].distance(p) for b in MAP["buildings"]), default=99.0), -1, 0)
            if False
            else float(
                np.clip(
                    (min((b["poly"].distance(p) for b in MAP["buildings"]), default=99.0) - 14) / 16, 0, 1
                )
            )
        )
    if t == "parking":
        return max(
            (1.0 if a["poly"].contains(p) else 0.0 for a in MAP["areas"] if a["kind"] == "parking"),
            default=0.0,
        )
    if t in ("pitch", "playground", "track", "church"):
        return max((soft(a["poly"].distance(p), 15, 35) for a in MAP["areas"] if a["kind"] == t), default=0.0)
    if t == "water":
        return max((soft(w.distance(p), 20, 45) for w in MAP["water"]), default=0.0)
    if t == "power_line":
        return max(
            (soft(LineString([(a, b), (c, d)]).distance(p), 18, 40) for a, b, c, d, _ in MAP["wires"]),
            default=0.0,
        )
    if t == "street":
        rs = [rd for rd in MAP["roads"] if street_eq(rd.get("name"), ref.get("name"))]
        return max((soft(LineString(rd["line"]).distance(p), 22, 50) for rd in rs), default=0.0)
    if t in ("water_tower", "mast", "chimney", "tower", "storage_tank", "flagpole") and not ref.get("kind"):
        lm = [l for l in MAP["landmarks"] if l["kind"] == t]
        ob = [q for q in hyps if q is not h and q["cls"] == t]
        return max(
            [soft(math.hypot(l["x"] - h["x"], l["y"] - h["y"]), 25, 50) for l in lm]
            + [soft(math.hypot(q["x"] - h["x"], q["y"] - h["y"]), 25, 50) for q in ob],
            default=0.0,
        )
    if t == "object":
        best = 0.0
        for q in hyps:
            if q is h or q["cls"] != ref.get("kind"):
                continue
            pc = (
                1.0
                if not ref.get("colour") or "colour" not in q["attrs"]
                else (
                    float(q["attrs"]["colour"][COLOURS.index(ref["colour"])])
                    if ref["colour"] in COLOURS
                    else 1.0
                )
            )
            best = max(
                best,
                pc
                * soft(
                    math.hypot(q["x"] - h["x"], q["y"] - h["y"]), 14 + 0.5 * q["size"], 30 + 0.5 * q["size"]
                ),
            )
        return best
    return 0.5  # unknown relation type: no evidence either way


COLOURS = ["red", "orange", "yellow", "green", "blue", "white", "grey", "black", "purple", "brown"]
VALUES = {
    "mark": ["H", "circle", "X"],
    "shape": ["round", "square"],
    "pattern": ["X", "rings", "cross"],
    "style": ["locker", "platform", "ridge", "dome", "legs", "column"],
    "door_open": [False, True],
    "parcel_colour": COLOURS,
    "lights": [False, True],
    "flags": [False, True],
    "rows": [1, 2, 3],
    "colour": COLOURS,
}


def attr_p(h, k, v, hyps):
    if k in ("size_m",):
        return float(np.exp(-0.5 * ((h["size"] - v) / max(0.3 * v, 0.8)) ** 2))
    if k == "height_m":
        return float(np.exp(-0.5 * ((h["height"] - v) / max(0.3 * v, 1.5)) ** 2)) if h["height"] > 0 else 0.5
    if k == "on_roof":
        return 1.0 if (h["z"] > 2.0) == bool(v) else 0.02
    if k == "size":  # relative to look-alikes of the same class
        same = [q for q in hyps if q["cls"] == h["cls"] and q is not h]
        if not same:
            return 0.5
        key = "height" if h["cls"] in ("water_tower", "mast") else "size"
        r = h[key] / max(np.median([q[key] for q in same]), 0.1)
        return (
            float(np.clip((r - 1.0) / 0.35, 0, 1)) if v == "large" else float(np.clip((1.0 - r) / 0.26, 0, 1))
        )
    if k in VALUES and k in h["attrs"]:
        vals = VALUES[k]
        vv = v if not isinstance(v, str) or k not in ("door_open", "lights", "flags", "rows") else v
        if vv not in vals:
            return 0.5
        return float(h["attrs"][k][vals.index(vv)])
    return 0.5


KINDS = [
    "pad",
    "dropzone",
    "pickup",
    "hangar",
    "water_tower",
    "mast",
    "tent",
    "container",
    "solar",
    "windsock",
]
HEADS = {
    "colour": COLOURS,
    "mark": ["H", "circle", "X"],
    "shape": ["round", "square"],
    "pattern": ["X", "rings", "cross"],
    "style": ["locker", "platform", "ridge", "dome", "legs", "column"],
    "door_open": ["False", "True"],
    "parcel_colour": COLOURS,
    "lights": ["False", "True"],
    "flags": ["False", "True"],
    "rows": ["1", "2", "3"],
}


def perfect_hyp(o):
    """A hypothesis with perfect eyes."""

    def oh(vals, v):
        a = np.full(len(vals), 0.01)
        s = [str(x) for x in vals]
        if str(v) in s:
            a[s.index(str(v))] = 1.0
        return a / a.sum()

    at = {"kind": oh(KINDS, o.kind)}
    for h, vals in HEADS.items():
        v = o.facts.get(h)
        v = False if v is None and h in ("lights", "flags") and o.kind in ("pad", "dropzone") else v
        if v is not None:
            at[h] = oh(vals, v)
    return {
        "x": o.x,
        "y": o.y,
        "z": o.z + 0.1,
        "cls": o.kind,
        "w": 1.0,
        "n": 1,
        "px": 100,
        "size": o.facts.get("size_m") or 2 * o.radius,
        "height": o.height,
        "attrs": at,
        "oid": o.oid,
    }


def fits_all(h, step, hyps, MAP, thr=0.5):
    """Does hypothesis h satisfy every stated attribute and relation (each part >= thr)?."""
    step = sanitize(step)
    tgt = step["target"]
    if h["cls"] != tgt.get("kind"):
        return False
    for k, v in tgt.items():
        if k != "kind" and attr_p(h, k, v, [q for q in hyps if q["cls"] == h["cls"]]) < thr:
            return False
    return all(rel_p(h, rel, MAP, hyps) >= thr for rel in step.get("relations", []))


NUM = ("size_m", "height_m")
BOOL = ("door_open", "lights", "flags", "on_roof")


def sanitize(step):
    """A parsed step coerced to the schema."""

    def num(v):
        try:
            v = float(v)
            return v if 0 < v < 1e4 else None
        except Exception:
            return None

    tgt = {}
    for k, v in (step.get("target") or {}).items() if isinstance(step.get("target"), dict) else ():
        if v is None:
            continue
        if k in NUM:
            v = num(v)
        elif k in BOOL:
            v = v if isinstance(v, bool) else {"true": True, "false": False}.get(str(v).lower())
        elif k == "rows":
            v = int(num(v)) if num(v) else None
        elif not isinstance(v, str):
            v = None
        if v is not None:
            tgt[k] = v
    rels = []
    for r in step.get("relations") or []:
        if not isinstance(r, dict) or not isinstance(r.get("rel"), str):
            continue
        ref = {}
        for k, v in (r.get("ref") or {}).items() if isinstance(r.get("ref"), dict) else ():
            if k == "levels":
                v = int(num(v)) if num(v) else None
            elif k == "address":
                v = [str(v[0]), str(v[1])] if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] else None
            elif not isinstance(v, str):
                v = None
            if v is not None:
                ref[k] = v
        rels.append({"rel": r["rel"], "ref": ref})
    params = (
        {k: num(v) for k, v in (step.get("params") or {}).items() if num(v) is not None}
        if isinstance(step.get("params"), dict)
        else {}
    )
    return {
        **step,
        "action": step.get("action") if isinstance(step.get("action"), str) else None,
        "target": tgt,
        "relations": rels,
        "params": params,
    }


def rank(hyps, step, rough, MAP, kinds):
    """Score every hypothesis of the asked kind."""
    step = sanitize(step)
    tgt = step["target"]
    kind = tgt.get("kind")
    out = []
    if kind not in kinds:
        return [], 1.0
    for h in hyps:
        pk = (
            float(h["attrs"]["kind"][kinds.index(kind)])
            if "kind" in h["attrs"]
            else (1.0 if h["cls"] == kind else 0.0)
        )
        if h["cls"] != kind and pk < 0.3:
            continue
        if rough is not None and math.hypot(h["x"] - rough[0], h["y"] - rough[1]) > MAX_ROUGH:
            continue  # (27.10: once the near ones were rejected, a look-alike 380 m away got
        parts = {
            "kind": max(pk, EPS),
            "seen": 1.0 - math.exp(-2.0 * h.get("w", 1.0)),
        }  # is it really there? one small far box is weak evidence
        for k, v in tgt.items():
            if k != "kind":
                parts[k] = max(attr_p(h, k, v, [q for q in hyps if q["cls"] == kind]), EPS)
        for i, rel in enumerate(step.get("relations", [])):
            parts[f"rel{i}:{rel['rel']}:{rel.get('ref', {}).get('type')}"] = max(
                rel_p(h, rel, MAP, hyps), EPS
            )
        if rough is not None:
            parts["rough"] = (
                math.exp(-0.5 * (math.hypot(h["x"] - rough[0], h["y"] - rough[1]) / SIGMA_ROUGH) ** 2) + 1e-3
            )
        out.append((float(np.sum(np.log(list(parts.values())))), h, parts))
    if not out:
        return [], 1.0
    L = np.array([o[0] for o in out])
    none_l = np.log(EPS) * (
        1 + len(tgt) + len(step.get("relations", []))
    )  # 'it is none of these' (not seen yet)
    Z = np.logaddexp.reduce(np.append(L, none_l))
    post = np.exp(L - Z)
    ranked = sorted(((float(p), h, parts) for p, (_, h, parts) in zip(post, out)), key=lambda t: -t[0])
    return ranked, float(np.exp(none_l - Z))


def decide(ranked, p_none, min_post=0.8, min_ratio=4.0):
    """'go' to the best one, or 'look' closer at the top candidates, or 'ask' the operator."""
    if not ranked:
        return "search", None
    p1 = ranked[0][0]
    p2 = ranked[1][0] if len(ranked) > 1 else 0.0
    if p1 >= min_post and p1 >= min_ratio * max(p2, 1e-9):
        return "go", ranked[0][1]
    return "look", ranked[0][1]
