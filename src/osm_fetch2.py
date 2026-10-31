"""Stage F (plan F–G sec.
usage: osm_fetch2.py OUT_DIR [name ...]"""

import json, math, sys, time, urllib.parse, urllib.request
from pathlib import Path

HALF = 250  # every place is a 500 x 500 m square
AREAS = {  # name: (lat, lon, split, place type from the map's own character, not
    "spb_center": (59.9330, 30.3440, "train", "city_center"),
    "akadem": (54.8380, 83.1070, "train", "buildings_in_forest"),
    "malakhovka": (55.6440, 38.0230, "train", "private_houses"),
    "malakhovka_line": (55.65663, 37.97119, "train", "power_line_suburb"),
    "msk_strogino": (55.8040, 37.4020, "train", "panel_district"),
    "ekb_center": (56.8380, 60.6000, "train", "city_center"),
    "kratovo": (55.5980, 38.1700, "train", "dachas_in_pines"),
    "tlt_industrial": (53.5450, 49.3600, "train", "industrial"),
    "spb_krestovsky": (59.9720, 30.2600, "train", "park_water"),
    "berlin_marzahn": (52.5450, 13.5640, "train", "panel_district_abroad"),
    "helsinki_kallio": (60.1840, 24.9500, "train", "city_abroad"),
    "plano_suburb": (33.0400, -96.7600, "train", "suburb_abroad"),
    "akadem_line": (54.87604, 83.11282, "val", "power_line_forest"),
    "warsaw_ursynow": (52.1460, 21.0480, "val", "panel_district_abroad"),
    "tokyo_res": (35.6620, 139.6990, "test", "foreign_city"),
    "kazan_novosav": (55.8230, 49.1300, "test_candidate", "panel_district"),
    "kazan_center": (55.7900, 49.1200, "test_candidate", "city_center"),
}
QUERY = """[out:json][timeout:180];
(way["building"]({b}); relation["building"]({b});
 way["highway"]({b}); node["highway"="street_lamp"]({b});
 way["amenity"="parking"]({b}); relation["amenity"="parking"]({b});
 way["leisure"~"^(park|garden|pitch|playground|stadium|sports_centre|track)$"]({b}); relation["leisure"~"^(park|garden|pitch)$"]({b});
 way["landuse"]({b}); relation["landuse"]({b});
 way["natural"~"^(wood|scrub|grassland|heath|water|sand|beach|wetland)$"]({b}); relation["natural"~"^(wood|water)$"]({b});
 node["natural"="tree"]({b}); way["natural"="tree_row"]({b});
 way["waterway"]({b}); way["water"]({b});
 way["power"~"^(line|minor_line)$"]({b}); node["power"~"^(tower|pole)$"]({b});
 way["barrier"~"^(fence|wall|hedge)$"]({b});
 node["man_made"~"^(water_tower|mast|tower|chimney|storage_tank|flagpole)$"]({b}); way["man_made"~"^(water_tower|tower|chimney|storage_tank|bridge)$"]({b});
 way["railway"~"^(rail|tram)$"]({b}););
out geom;"""


def bbox(lat, lon, half=HALF):
    dlat = half / 111_320.0
    dlon = half / (111_320.0 * math.cos(math.radians(lat)))
    return f"{lat - dlat:.6f},{lon - dlon:.6f},{lat + dlat:.6f},{lon + dlon:.6f}"


def fetch(name, lat, lon, out_dir):
    data = urllib.parse.urlencode({"data": QUERY.format(b=bbox(lat, lon))}).encode()
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=data,
        headers={"User-Agent": "affect-drone-research/0.2"},
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        raw = r.read()
    j = json.loads(raw)
    (out_dir / f"{name}.json").write_bytes(raw)
    return j


def summary(j):
    E = j.get("elements", [])
    t = lambda e, k: e.get("tags", {}).get(k)
    b = [e for e in E if t(e, "building")]
    return {
        "buildings": len(b),
        "h": sum(bool(t(e, "height")) for e in b),
        "levels": sum(bool(t(e, "building:levels")) for e in b),
        "colour": sum(bool(t(e, "building:colour")) for e in b),
        "roof_shape": sum(bool(t(e, "roof:shape")) for e in b),
        "roads": sum(bool(t(e, "highway")) and e["type"] == "way" for e in E),
        "named_roads": sum(bool(t(e, "highway")) and bool(t(e, "name")) for e in E),
        "parking": sum(t(e, "amenity") == "parking" for e in E),
        "parks": sum(t(e, "leisure") in ("park", "garden") for e in E),
        "pitches": sum(t(e, "leisure") in ("pitch", "playground", "stadium", "track") for e in E),
        "landuse": sum(bool(t(e, "landuse")) for e in E),
        "trees": sum(t(e, "natural") == "tree" for e in E),
        "woods": sum(t(e, "natural") == "wood" or t(e, "landuse") == "forest" for e in E),
        "water": sum(t(e, "natural") == "water" or bool(t(e, "waterway")) or bool(t(e, "water")) for e in E),
        "power": sum(t(e, "power") in ("line", "minor_line") for e in E),
        "lamps": sum(t(e, "highway") == "street_lamp" for e in E),
        "barriers": sum(bool(t(e, "barrier")) for e in E),
        "man_made": sum(bool(t(e, "man_made")) for e in E),
        "rail": sum(bool(t(e, "railway")) for e in E),
    }


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[2:])
    for name, (lat, lon, split, kind) in AREAS.items():
        if only and name not in only:
            continue
        if (out / f"{name}.json").exists():
            print(
                name,
                "cached",
                summary(json.loads((out / f"{name}.json").read_text(encoding="utf-8"))),
                flush=True,
            )
            continue
        for attempt in range(3):
            try:
                j = fetch(name, lat, lon, out)
                print(name, split, kind, summary(j), flush=True)
                break
            except Exception as ex:
                print(name, "FAILED", attempt, type(ex).__name__, str(ex)[:120], flush=True)
                time.sleep(20)
        time.sleep(5)
