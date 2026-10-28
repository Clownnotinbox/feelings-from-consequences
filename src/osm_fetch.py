"""Download real places from OpenStreetMap."""

import json, math, sys, time, urllib.parse, urllib.request
from pathlib import Path

AREAS = {  # name: (lat, lon, half-size in metres, place type for reporting, from
    "spb_center": (59.9330, 30.3440, 200, "city_center"),
    "akadem": (54.8380, 83.1070, 200, "buildings_in_forest"),
    "malakhovka": (55.6440, 38.0230, 200, "private_houses"),
    "tokyo_res": (35.6620, 139.6990, 200, "foreign_city"),
    "malakhovka_line": (55.65663, 37.97119, 200, "power_line_suburb"),
    "akadem_line": (54.87604, 83.11282, 200, "power_line_forest"),
}
QUERY = """[out:json][timeout:90];
(way["building"]({b}); relation["building"]({b}); node["natural"="tree"]({b});
 way["natural"="wood"]({b}); way["landuse"="forest"]({b}); relation["landuse"="forest"]({b});
 way["power"~"^(line|minor_line)$"]({b}); node["power"~"^(tower|pole)$"]({b});
 way["natural"="water"]({b}); way["waterway"]({b}); way["highway"]({b}););
out geom;"""


def bbox(lat, lon, half):
    dlat = half / 111_320.0
    dlon = half / (111_320.0 * math.cos(math.radians(lat)))
    return f"{lat - dlat:.6f},{lon - dlon:.6f},{lat + dlat:.6f},{lon + dlon:.6f}"


def fetch(name, lat, lon, half, out_dir):
    q = QUERY.format(b=bbox(lat, lon, half))
    data = urllib.parse.urlencode({"data": q}).encode()
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=data,
        headers={"User-Agent": "affect-drone-research/0.1"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    j = json.loads(raw)
    (out_dir / f"{name}.json").write_bytes(raw)
    return j


def summary(j):
    E = j.get("elements", [])
    tag = lambda e, k: e.get("tags", {}).get(k)
    b = [e for e in E if tag(e, "building")]
    return {
        "buildings": len(b),
        "with_height": sum(bool(tag(e, "height")) for e in b),
        "with_levels": sum(bool(tag(e, "building:levels")) for e in b),
        "trees": sum(tag(e, "natural") == "tree" for e in E),
        "woods": sum(tag(e, "natural") == "wood" or tag(e, "landuse") == "forest" for e in E),
        "power_lines": sum(tag(e, "power") in ("line", "minor_line") for e in E),
        "towers_poles": sum(tag(e, "power") in ("tower", "pole") for e in E),
        "water": sum(tag(e, "natural") == "water" or bool(tag(e, "waterway")) for e in E),
        "roads": sum(bool(tag(e, "highway")) for e in E),
    }


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[2:])
    for name, (lat, lon, half, kind) in AREAS.items():
        if only and name not in only:
            continue
        try:
            j = fetch(name, lat, lon, half, out)
            print(name, kind, summary(j), flush=True)
        except Exception as ex:
            print(name, "FAILED", type(ex).__name__, str(ex)[:120], flush=True)
        time.sleep(3)
