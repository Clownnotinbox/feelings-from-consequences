"""Stage F mission parser."""

import json, re

SYSTEM = (
    "You turn instructions for a civil drone into JSON. Output only JSON with keys: steps (list of {action, target, relations, params}), return_home. "
    "action: land | drop | pickup | enter | inspect | flyby. target: {kind: pad | dropzone | pickup | hangar | water_tower | mast | tent | container | solar | windsock, "
    "and only the attributes the text states: colour, size (small|large), size_m, height_m, mark (H|circle|X), shape (round|square), pattern (X|rings|cross), "
    "style (locker|platform|ridge|dome|legs|column), door_open, parcel_colour, lights, flags, rows, on_roof}. relations: list of {rel: next_to | north_of | south_of | "
    "east_of | west_of | on_roof_of | on | by_street | in_open, ref: {type: building | pitch | playground | parking | track | church | water | power_line | street | open | "
    "object | water_tower | mast | ..., and what the text says about it: btype (apartment|house|garage|warehouse|office|church), levels, material, colour, address [street, number], "
    "kind, name}}. params: distance_m (inspect), altitude_m (flyby). return_home: true, false, or null if the text does not say. Colours in English. Never add facts."
)


def canon(m):
    """Canonical form for comparison: no None values, floats to 0.1, relations sorted, keys sorted."""

    def clean(x):
        if hasattr(x, "tolist"):
            x = x.tolist()  # numpy arrays / scalars from the generator
        if isinstance(x, dict):
            return {
                k: clean(v)
                for k, v in sorted(x.items())
                if not (v is None or (isinstance(v, (list, dict)) and not len(v)))
                or k in ("relations", "params")
            }
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, float):
            return round(x, 1)
        return x

    m = clean(m)
    for st in m.get("steps", []):
        st["relations"] = sorted(
            st.get("relations", []), key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False)
        )
        st.setdefault("params", {})
    m.setdefault("return_home", None)
    return m


def dumps(m):
    return json.dumps(canon(m), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_output(text):
    """Model output -> dict or None (first JSON object in the text)."""
    s = text.strip()
    s = s[s.find("{") :] if "{" in s else s
    for end in range(len(s), 0, -1):
        if s[end - 1] != "}":
            continue
        try:
            return json.loads(s[:end])
        except Exception:
            continue
    return None


def fields(m):
    """Flat facts for per-field accuracy."""
    m = canon(m) if m else {"steps": [], "return_home": None}
    out = {("return_home", json.dumps(m.get("return_home")))}
    for i, st in enumerate(m.get("steps", [])):
        out.add((f"s{i}.action", st.get("action")))
        for k, v in (st.get("target") or {}).items():
            out.add((f"s{i}.target.{k}", json.dumps(v, ensure_ascii=False)))
        for r in st.get("relations", []):
            out.add((f"s{i}.rel", json.dumps(r, sort_keys=True, ensure_ascii=False)))
        for k, v in (st.get("params") or {}).items():
            out.add((f"s{i}.param.{k}", json.dumps(v)))
    return out


def well_formed(m):
    """The schema's shape (anything else from a model counts as a wrong answer, not a crash)."""
    if not isinstance(m, dict) or not isinstance(m.get("steps"), list):
        return False
    for st in m["steps"]:
        if (
            not isinstance(st, dict)
            or not isinstance(st.get("target", {}), dict)
            or not isinstance(st.get("relations", []), list)
            or not isinstance(st.get("params", {}) or {}, dict)
        ):
            return False
        if any(
            not isinstance(r, dict) or not isinstance(r.get("ref", {}), dict) for r in st.get("relations", [])
        ):
            return False
    return True


def score(pred, truth):
    """exact match, field precision / recall counts; a malformed prediction scores as empty."""
    if not well_formed(pred):
        pred = None
    P, T = fields(pred), fields(truth)
    return {
        "exact": pred is not None and dumps(pred) == dumps(truth),
        "tp": len(P & T),
        "np": len(P),
        "nt": len(T),
    }
