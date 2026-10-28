"""Stage F mission generator."""

import math, random
import numpy as np
from shapely.geometry import Point
import mission_objects as MO
import phrases as P

AREA_M = 25.0  # keep missions this far inside the 500 x 500 m area
ACTIONS = ("land", "drop", "pickup", "enter", "inspect", "flyby")
ACTION_P = (0.25, 0.19, 0.15, 0.12, 0.15, 0.14)
KIND_FOR = {
    "land": ["pad"] * 7 + ["container"],
    "drop": ["dropzone"],
    "pickup": ["pickup"],
    "enter": ["hangar"],
    "inspect": ["water_tower", "mast", "solar"],
    "flyby": ["pad", "dropzone", "hangar", "water_tower", "mast", "tent", "container"],
}
NEED = {
    "hangar": 8.0,
    "container": 7.5,
    "dropzone": 7.0,
    "solar": 6.5,
    "mast": 3.0,
    "water_tower": 4.0,
    "pad": 4.0,
    "pickup": 2.5,
    "tent": 3.0,
    "windsock": 2.0,
}


def bearing_word(dx, dy):
    """Cardinal side if clearly within +-30 deg of it, else None."""
    a = math.degrees(math.atan2(dy, dx)) % 360
    for w, c in (("east", 0), ("north", 90), ("west", 180), ("south", 270), ("east", 360)):
        if abs(a - c) <= 30:
            return w
    return None


def side_truth(dx, dy, word):
    """True if the bearing is within 30 deg of word, False if more than 60 deg away, else None."""
    c = {"east": 0, "north": 90, "west": 180, "south": 270}[word]
    a = math.degrees(math.atan2(dy, dx)) % 360
    d = min(abs(a - c), 360 - abs(a - c))
    return True if d <= 30 else False if d > 60 else None


class Scene:
    def __init__(self, W, rng):
        self.W, self.rng = W, rng
        self.objs = []
        self.half = W.size / 2 - AREA_M

    def ok(self, x, y, need, roof=False):
        W = self.W
        if abs(x) > self.half or abs(y) > self.half:
            return False
        if not roof:
            if not W.free(x, y, need):
                return False
            if W.road_area is not None and W.road_area.distance(Point(x, y)) < need * 0.8:
                return False
            if any(p.distance(Point(x, y)) < need for p in W.water):
                return False
            if any((x - a) ** 2 + (y - b) ** 2 < (need + 1.5) ** 2 for a, b, _ in W.trees):
                return False
            if math.hypot(x - W.start[0], y - W.start[1]) < need + 6:
                return False
        return all(math.hypot(x - o.x, y - o.y) > need + o.radius + 2.5 for o in self.objs)

    def put(self, kind, near=None, r=(0, 0), roof=False, oid=None, **attrs):
        rng = self.rng
        need = NEED[kind]
        for _ in range(2500):
            if roof:
                cands = [
                    f
                    for f in self.W.footprints
                    if f["roof"] == "flat" and f["h"] < 28 and f["poly"].buffer(-need - 0.8).area > 1
                ]
                if near is not None:
                    cands = [f for f in cands if f["poly"].distance(Point(*near)) < r[1]]
                if not cands:
                    return None
                f = cands[int(rng.integers(len(cands)))]
                p = f["poly"].buffer(-need - 0.8)
                bx = p.bounds
                x, y = rng.uniform(bx[0], bx[2]), rng.uniform(bx[1], bx[3])
                if not p.contains(Point(x, y)) or not self.ok(x, y, need, roof=True):
                    continue
                o = MO.make(
                    oid or f"m{len(self.objs)}",
                    kind,
                    x,
                    y,
                    rng.uniform(0, 2 * math.pi),
                    f["h"],
                    np.random.default_rng(int(rng.integers(1 << 30))),
                    **attrs,
                )
                o.on_roof = f["name"]
                self.objs.append(o)
                return o
            if near is None:
                x, y = rng.uniform(-self.half, self.half), rng.uniform(-self.half, self.half)
            else:
                a, d = rng.uniform(0, 2 * math.pi), rng.uniform(*r)
                x, y = near[0] + d * math.cos(a), near[1] + d * math.sin(a)
            if not self.ok(x, y, need):
                continue
            yaw = rng.uniform(0, 2 * math.pi)
            if kind == "hangar":  # the door faces open ground: check 10 m in front of it
                for _k in range(8):
                    yaw = rng.uniform(0, 2 * math.pi)
                    fx, fy = x - math.sin(yaw) * 10, y + math.cos(yaw) * 10
                    if self.W.free(fx, fy, 4) and all(
                        math.hypot(fx - o.x, fy - o.y) > o.radius + 4 for o in self.objs
                    ):
                        break
                else:
                    continue
            o = MO.make(
                oid or f"m{len(self.objs)}",
                kind,
                x,
                y,
                yaw,
                0.0,
                np.random.default_rng(int(rng.integers(1 << 30))),
                **attrs,
            )
            self.objs.append(o)
            return o
        return None


#
def attr_facts(o, same):
    """Attribute facts usable to tell o apart from same-kind objects."""
    f = o.facts
    out = {"colour": f.get("colour")} if o.kind not in ("solar", "windsock") else {}
    for k in ("mark", "shape", "pattern", "style", "parcel_colour", "rows"):
        if f.get(k) is not None:
            out[k] = f[k]
    if f.get("lights"):
        out["lights"] = True
    if f.get("flags"):
        out["flags"] = True
    if o.kind == "hangar":
        out["door_open"] = f["door_open"]
    if o.on_roof:
        out["on_roof"] = True
    sz = f.get("size_m") if o.kind not in ("mast", "water_tower") else f.get("height_m")
    twins = [
        s.facts.get("size_m") if o.kind not in ("mast", "water_tower") else s.facts.get("height_m")
        for s in same
        if s is not o and s.facts.get("colour") == f.get("colour")
    ]
    if sz and twins:
        if all(sz >= 1.35 * t for t in twins):
            out["size"] = "large"
        elif all(sz <= 0.74 * t for t in twins):
            out["size"] = "small"
    return out


def attr_holds(o, k, v, same):
    if k == "size":
        mine = attr_facts(o, same).get("size")
        return True if mine == v else (False if mine is not None or True else None)
    if k == "on_roof":
        return bool(o.on_roof) == v
    if k in ("lights", "flags"):
        return bool(o.facts.get(k)) == v
    if k == "door_open":
        return o.facts.get("door_open") == v
    return o.facts.get(k) == v


def relation_facts(o, W, objs, rng):
    """Candidate relations TRUE of o: list of (key, ref dict, truth function for any object)."""
    out = []
    p = Point(o.x, o.y)
    near_b = sorted(W.footprints, key=lambda f: f["poly"].distance(p))[:3]
    for f in near_b[:2]:
        d = f["poly"].distance(p)
        if o.on_roof == f["name"]:
            out.append(("on_roof_of", {"type": "building", "b": f}, lambda q, f=f: q.on_roof == f["name"]))
            continue
        if d < 14:
            out.append(
                (
                    "next_to",
                    {"type": "building", "b": f},
                    lambda q, f=f: (
                        True
                        if f["poly"].distance(Point(q.x, q.y)) < 14 and not q.on_roof
                        else False if f["poly"].distance(Point(q.x, q.y)) > 26 else None
                    ),
                )
            )
        if d < 40:
            c = f["poly"].centroid
            w = bearing_word(o.x - c.x, o.y - c.y)
            if w and side_truth(o.x - c.x, o.y - c.y, w):

                def side_t(q, f=f, w=w):
                    dq = f["poly"].distance(Point(q.x, q.y))
                    cq = f["poly"].centroid
                    if q.on_roof:
                        return False
                    if dq > 60:
                        return False
                    s = side_truth(q.x - cq.x, q.y - cq.y, w)
                    return s if dq < 40 else (False if s is False else None)

                out.append((f"{w}_of", {"type": "building", "b": f}, side_t))
    for L in W.landmarks:
        if "poly" in L:
            d = L["poly"].distance(p)
            inside = L["poly"].contains(p)
            if L["kind"] == "parking" and inside and not o.on_roof:
                out.append(
                    (
                        "on",
                        {"type": "parking"},
                        lambda q, L=L: L["poly"].contains(Point(q.x, q.y)) and not q.on_roof,
                    )
                )
                continue
            if d < 15 and not inside:
                out.append(
                    (
                        "next_to",
                        {"type": L["kind"]},
                        lambda q, L=L: (
                            True
                            if L["poly"].distance(Point(q.x, q.y)) < 15
                            else False if L["poly"].distance(Point(q.x, q.y)) > 35 else None
                        ),
                    )
                )
        elif math.hypot(L["x"] - o.x, L["y"] - o.y) < 25:
            out.append(
                (
                    "next_to",
                    {"type": L["kind"]},
                    lambda q, L=L: (
                        True
                        if math.hypot(L["x"] - q.x, L["y"] - q.y) < 25
                        else False if math.hypot(L["x"] - q.x, L["y"] - q.y) > 50 else None
                    ),
                )
            )
    if W.water and min(w.distance(p) for w in W.water) < 20:
        out.append(
            (
                "next_to",
                {"type": "water"},
                lambda q: (
                    True
                    if min(w.distance(Point(q.x, q.y)) for w in W.water) < 20
                    else False if min(w.distance(Point(q.x, q.y)) for w in W.water) > 45 else None
                ),
            )
        )
    if W.wires:
        from shapely.geometry import LineString

        lines = [LineString([(a, b), (c, d)]) for a, b, c, d, _ in W.wires]
        if min(l.distance(p) for l in lines) < 18:
            out.append(
                (
                    "next_to",
                    {"type": "power_line"},
                    lambda q: (
                        True
                        if min(l.distance(Point(q.x, q.y)) for l in lines) < 18
                        else False if min(l.distance(Point(q.x, q.y)) for l in lines) > 40 else None
                    ),
                )
            )
    named = [r for r in W.map["roads"] if r.get("name")]
    if named:
        from shapely.geometry import LineString

        rd = min(named, key=lambda r: LineString(r["line"]).distance(p))
        dd = LineString(rd["line"]).distance(p)
        if dd < 22:
            ls = LineString(rd["line"])
            out.append(
                (
                    "by_street",
                    {"type": "street", "name": rd["name"]},
                    lambda q, ls=ls: (
                        True
                        if ls.distance(Point(q.x, q.y)) < 22
                        else False if ls.distance(Point(q.x, q.y)) > 50 else None
                    ),
                )
            )
    if not near_b or near_b[0]["poly"].distance(p) > 30:
        out.append(
            (
                "in_open",
                {"type": "open"},
                lambda q: (
                    True
                    if all(f["poly"].distance(Point(q.x, q.y)) > 30 for f in W.footprints[:])
                    else (
                        False if any(f["poly"].distance(Point(q.x, q.y)) < 14 for f in W.footprints) else None
                    )
                ),
            )
        )
    for q in objs:  # another mission object of a different kind as a landmark
        if q is o or q.kind == o.kind:
            continue
        dq = math.hypot(q.x - o.x, q.y - o.y)
        if dq < 14 + q.radius:
            out.append(
                (
                    "next_to",
                    {"type": "object", "kind": q.kind, "colour": q.facts.get("colour")},
                    lambda r, q=q: (
                        True
                        if math.hypot(q.x - r.x, q.y - r.y) < 14 + q.radius
                        else False if math.hypot(q.x - r.x, q.y - r.y) > 30 + q.radius else None
                    ),
                )
            )
    rng.shuffle(out)
    return out


def describe_facts(t, objs, W, rng, detail):
    """Dale & Reiter: kind first, then attributes and relations that rule out remaining same-kind objects."""
    same = [o for o in objs if o.kind == t.kind]
    rest = [o for o in same if o is not t]
    used = []
    rels = []
    A = attr_facts(t, same)
    order = [
        k
        for k in (
            "colour",
            "mark",
            "pattern",
            "style",
            "size",
            "door_open",
            "parcel_colour",
            "rows",
            "on_roof",
            "shape",
            "lights",
            "flags",
        )
        if k in A
    ]
    R = relation_facts(t, W, objs, rng)
    if detail == "sparse":
        if rng.random() < 0.6 and "colour" in A and A["colour"]:
            used.append(("colour", A["colour"]))
        return used, rels, len([o for o in rest if all(attr_holds(o, k, v, same) for k, v in used)]) == 0
    for k in order:
        if not rest:
            break
        v = A[k]
        out_ = [o for o in rest if attr_holds(o, k, v, same) is False]
        if out_:
            used.append((k, v))
            rest = [o for o in rest if o not in out_]

    def clash(key, ref):
        refs = [r for _, r in rels]
        if ref["type"] == "building" and any(
            r["type"] == "open" or (r["type"] == "building" and r["b"]["name"] == ref["b"]["name"])
            for r in refs
        ):
            return True
        if ref["type"] == "open" and any(r["type"] == "building" for r in refs):
            return True
        return any(
            r["type"] == ref["type"] and r.get("kind") == ref.get("kind") and ref["type"] not in ("building",)
            for r in refs
        )

    for key, ref, fn in R:
        if not rest:
            break
        if clash(key, ref):
            continue
        out_ = [o for o in rest if fn(o) is False]
        if out_ and fn(t) is True:
            rels.append((key, ref))
            rest = [o for o in rest if o not in out_]
    rels = [(k, finalize(r, rng)) for k, r in rels]
    if any(k == "on_roof_of" for k, _ in rels):
        used = [u for u in used if u[0] != "on_roof"]
    if rest:
        return used, rels, False
    if not any(k == "colour" for k, _ in used) and A.get("colour") and rng.random() < 0.8:
        used.insert(0, ("colour", A["colour"]))
    if detail == "full":  # a few more TRUE facts, as people do
        extra = [k for k in order if k not in [u for u, _ in used] and k not in ("size",)]
        for k in extra[: int(rng.integers(1, 3))]:
            used.append((k, A[k]))
        if t.facts.get("size_m") and t.kind not in ("mast", "water_tower") and rng.random() < 0.6:
            used.append(("size_m", t.facts["size_m"]))
        if t.facts.get("height_m") and rng.random() < 0.6:
            used.append(("height_m", t.facts["height_m"]))
        for key, ref, fn in R:
            if len(rels) >= 2:
                break
            if fn(t) is True and not clash(key, ref) and rng.random() < 0.6:
                rels.append((key, finalize(ref, rng)))
    if any(k == "on_roof_of" for k, _ in rels):
        used = [u for u in used if u[0] != "on_roof"]
    return used, rels, True


def finalize(ref, rng):
    """Fix once what the text will say about a reference."""
    r = dict(ref)
    if r["type"] == "building":
        r["say"] = P.say_building(r["b"], rng)
    if r["type"] == "object" and r["kind"] in ("solar", "windsock"):
        r["colour"] = None
    return r


#
def _ru_np(t, used, bank, rng, case):
    nouns = P.RU_KIND[bank][t.kind]
    n = nouns[int(rng.integers(len(nouns)))]
    adjs = []
    for k, v in used:
        if k == "size":
            adjs.append(P.RU_SIZE[bank][v])
        if k == "colour" and t.kind != "solar":
            adjs.append(P.RU_COL[bank][v])
    return n, adjs


def _ru_post(t, used, bank):
    """Postpositive attribute phrases in Russian."""
    out = []
    d = dict(used)
    C = P.RU_COL[bank]
    if "mark" in d:
        out.append(
            {
                "H": "с белой буквой H" if t.facts.get("mark_colour") == "white" else "с чёрной буквой H",
                "circle": "с нарисованным кругом",
                "X": "с крестом",
            }[d["mark"]]
        )
    if "pattern" in d:
        out.append(
            {"X": "с большим крестом", "rings": "в виде мишени с кольцами", "cross": "с прямым крестом"}[
                d["pattern"]
            ]
            if bank == "train"
            else {
                "X": "с диагональным крестом",
                "rings": "с концентрическими кругами",
                "cross": "с крестом в форме плюса",
            }[d["pattern"]]
        )
    if "shape" in d:
        out.append("круглой формы" if d["shape"] == "round" else "квадратной формы")
    if "lights" in d:
        out.append("с огнями по краю" if bank == "train" else "с подсветкой по периметру")
    if "flags" in d:
        out.append("с флажками по углам" if bank == "train" else "с вымпелами на углах")
    if "style" in d:
        out.append(
            {
                "locker": "в виде шкафа-постамата",
                "platform": "в виде низкой платформы",
                "ridge": "двускатная",
                "dome": "в форме купола",
                "legs": "на четырёх опорах",
                "column": "на одной колонне",
            }[d["style"]]
        )
    if "parcel_colour" in d:
        out.append(
            f"с {P.adj(C[d['parcel_colour']], 'f', 'ins')} коробкой наверху"
            if bank == "train"
            else f"где лежит {P.adj(C[d['parcel_colour']], 'f', 'nom')} посылка"
        )
    if "door_open" in d:
        out.append("с открытыми воротами" if d["door_open"] else "с закрытыми воротами")
    if "rows" in d:
        out.append({1: "в один ряд", 2: "в два ряда", 3: "в три ряда"}[d["rows"]])
    if "size_m" in d:
        out.append(
            f"{'диаметром' if t.kind == 'pad' and t.facts.get('shape') == 'round' else 'размером'} около {P.ru_metres(d['size_m'])}"
            if t.kind not in ("container",)
            else f"длиной около {P.ru_metres(d['size_m'])}"
        )
    if "height_m" in d:
        out.append(f"высотой около {P.ru_metres(d['height_m'])}")
    if "on_roof" in d:
        out.append("на крыше дома" if bank == "train" else "наверху, на кровле здания")
    return out


def _ru_rel(key, ref, bank, rng):
    if ref["type"] == "building":
        say = ref["say"]
        addr = say.get("address")
        if key == "on_roof_of":
            return f"на крыше {P.building_ru(say, 'gen', bank)}" + (
                f" ({addr[0]}, {addr[1]})" if addr else ""
            )
        name = f"дома {addr[1]} по {P.street_ru(addr[0], 'dat')}" if addr else P.building_ru(say, "gen", bank)
        if key == "next_to":
            return (
                (
                    f"рядом с домом {addr[1]} по {P.street_ru(addr[0], 'dat')}"
                    if addr
                    else f"рядом с {P.building_ru(say, 'ins', bank)}"
                )
                if bank == "train"
                else (
                    f"вплотную к {P.building_ru(say, 'dat', bank)}"
                    if not addr
                    else f"возле дома {addr[1]} на {P.street_ru(addr[0], 'pre')}"
                )
            )
        w = key.split("_")[0]
        return f"{P.RU_DIR[bank][w]} {name}"
    if ref["type"] == "open":
        return "на открытом месте, вдали от домов" if bank == "train" else "на пустыре без построек рядом"
    if ref["type"] == "parking":
        return "на парковке" if bank == "train" else "на автостоянке"
    if ref["type"] == "street":
        return (
            f"у дороги на {P.street_ru(ref['name'], 'pre')}"
            if bank == "train"
            else f"вдоль проезжей части, {P.street_ru(ref['name'], 'nom')}"
        )
    if ref["type"] == "object":
        n = P.RU_KIND["train" if bank == "train" else "test"][ref["kind"]][0]
        adjs = (
            [P.RU_COL[bank][ref["colour"]]]
            if ref.get("colour") and ref["kind"] not in ("solar", "windsock")
            else []
        )
        return f"рядом с {n('ins', adjs)}" if bank == "train" else f"в паре шагов от {n('gen', adjs)}"
    L = P.RU_LANDMARK[ref["type"]]
    return f"рядом с {L('ins')}" if bank == "train" else f"неподалёку от {L('gen')}"


VERB_RU = {
    "train": {
        "land": [("Приземлись на", "acc"), ("Сядь на", "acc"), ("Нужно сесть на", "acc")],
        "drop": [("Сбрось груз с парашютом над", "ins"), ("Сбрось посылку на парашюте над", "ins")],
        "pickup": [("Забери посылку из", "gen*"), ("Возьми груз из", "gen*")],
        "enter": [("Влети внутрь", "gen"), ("Залети в", "acc")],
        "inspect": [("Облети по кругу", "acc"), ("Осмотри со всех сторон", "acc")],
        "flyby": [("Пролети рядом с", "ins"), ("Пролети возле", "gen")],
    },
    "test": {
        "land": [("Выполни посадку на", "acc"), ("Опустись на", "acc")],
        "drop": [("Доставь посылку сбросом над", "ins"), ("Скинь груз на парашюте над", "ins")],
        "pickup": [("Подбери отправление из", "gen*"), ("Прихвати коробку из", "gen*")],
        "enter": [("Заведи дрон внутрь", "gen"), ("Зайди в", "acc")],
        "inspect": [("Сделай облёт вокруг", "gen"), ("Обойди кольцом", "acc")],
        "flyby": [("Пройди мимо", "gen"), ("Проскочи около", "gen")],
    },
}


PRON = {
    "f": ("она", "неё", "ней", "её"),
    "m": ("он", "него", "нём", "его"),
    "n": ("оно", "него", "нём", "его"),
    "pl": ("они", "них", "них", "их"),
}
SHORT_RU = {
    "train": {
        "land": "Сядь на {acc}",
        "drop": "Сбрось над {ins} груз на парашюте",
        "pickup": "Забери оттуда посылку",
        "enter": "Влети внутрь через открытые ворота и приземлись",
        "inspect": "Облети {acc2} по кругу",
        "flyby": "Пролети рядом",
    },
    "test": {
        "land": "Выполни посадку на {acc}",
        "drop": "Скинь над {ins} коробку с парашютом",
        "pickup": "Подбери там отправление",
        "enter": "Заведи дрон внутрь и опустись на пол",
        "inspect": "Обойди {acc2} кольцом",
        "flyby": "Пройди мимо на небольшом удалении",
    },
}


def text_ru_b(st, bank, rng):
    """Relation first: 'Рядом с футбольным полем стоит оранжевая площадка с буквой H. Сядь на неё.'."""
    t, used, rels, act, par = st["obj"], st["used"], st["rels"], st["action"], st["params"]
    n, adjs = _ru_np(t, used, bank, rng, "nom")
    post = _ru_post(t, used, bank)
    g = n.g
    pr = PRON[g]
    rel = " и ".join(_ru_rel(k, r, bank, rng) for k, r in rels)
    rel = rel[0].upper() + rel[1:]
    verb = (
        ("стоят" if g == "pl" else "стоит")
        if bank == "train"
        else ("находятся" if g == "pl" else "находится")
    )
    s1 = f"{rel} {verb} {n('nom', adjs)}" + ((", " + ", ".join(post)) if post else "") + "."
    s2 = SHORT_RU[bank][act].format(
        acc=pr[1] if act == "land" else "",
        ins=pr[2] if g != "pl" else "ними",
        acc2=pr[3] if g != "pl" else "их",
    )
    if act == "inspect" and par.get("distance_m"):
        s2 += f", на расстоянии {P.ru_metres(par['distance_m'])}"
    if act == "flyby" and par.get("altitude_m"):
        s2 += f", на высоте {P.ru_metres(par['altitude_m'])}"
    return s1 + " " + s2 + "."


def text_ru(steps, ret, bank, rng):
    sents = []
    for i, st in enumerate(steps):
        t, used, rels, act, par = st["obj"], st["used"], st["rels"], st["action"], st["params"]
        if rels and i == 0 and len(steps) == 1 and rng.random() < 0.35:
            sents.append(text_ru_b(st, bank, rng))
            continue
        verb, case = VERB_RU[bank][act][int(rng.integers(len(VERB_RU[bank][act])))]
        n, adjs = _ru_np(t, used, bank, rng, case)
        if case == "gen*":  # 'из' for most, 'с' for a platform-like noun
            case = "gen"
            verb = (
                verb.replace(" из", " с")
                if n.f[0] in ("площадка", "точка") and "точка" not in n.f[0]
                else verb
            )
        np_ = n(case, adjs)
        post = _ru_post(t, used, bank)
        post_s = (", " + ", ".join(post)) if post else ""
        s = f"{verb} {np_}{post_s}"
        if act == "enter":
            s += " через открытые ворота и сядь внутри" if bank == "train" else ", там приземлись на пол"
        if act == "inspect" and par.get("distance_m"):
            s += ". " + (
                f"Держись на расстоянии {P.ru_metres(par['distance_m'])} от {PRON[n.g][1]}"
                if bank == "train"
                else f"Дистанция облёта — {par['distance_m']} м"
            )
        if act == "flyby" and par.get("altitude_m"):
            s += ". " + (
                f"Высота пролёта — {P.ru_metres(par['altitude_m']).replace('метров', 'метров').replace('метра', 'метра')}"
                if bank == "train"
                else f"Иди на высоте {par['altitude_m']} м"
            )
        if i == 1:
            s = ("Затем " if bank == "train" else "После этого ") + s[0].lower() + s[1:]
        sents.append(s + ".")
        if rels:
            g = n.g
            pron = {"f": "Она", "m": "Он", "n": "Оно", "pl": "Они"}[g]
            verb2 = (
                ("стоят" if g == "pl" else "стоит")
                if bank == "train"
                else ("находятся" if g == "pl" else "находится")
            )
            sents.append(f"{pron} {verb2} " + " и ".join(_ru_rel(k, r, bank, rng) for k, r in rels) + ".")
    if ret is True:
        sents.append(
            str(rng.choice(["Потом вернись домой.", "После этого возвращайся на базу."]))
            if bank == "train"
            else str(
                rng.choice(["По завершении лети обратно к точке старта.", "Затем назад к месту взлёта."])
            )
        )
    elif ret is False:
        sents.append(
            str(rng.choice(["Возвращаться не нужно.", "Обратно не лети."]))
            if bank == "train"
            else str(rng.choice(["Назад не требуется.", "Домой после этого не нужно."]))
        )
    return " ".join(sents)


EN_VERB = {
    "train": {
        "land": ["Land on the", "Touch down on the"],
        "drop": ["Drop the parcel by parachute over the", "Parachute the package onto the"],
        "pickup": ["Pick up the parcel from the", "Collect the package at the"],
        "enter": ["Fly into the", "Enter the"],
        "inspect": ["Circle around the", "Inspect from all sides the"],
        "flyby": ["Fly past the", "Pass by the"],
    },
    "test": {
        "land": ["Settle onto the", "Put down on the"],
        "drop": ["Release the supply crate above the", "Air-drop the box over the"],
        "pickup": ["Grab the delivery waiting at the", "Retrieve the box from the"],
        "enter": ["Steer inside the", "Head in through the doors of the"],
        "inspect": ["Orbit the", "Make a full loop around the"],
        "flyby": ["Skim past the", "Cruise by the"],
    },
}


def _en_rel(key, ref, bank, rng):
    if ref["type"] == "building":
        say = ref["say"]
        addr = say.get("address")
        nm = f"number {addr[1]} {addr[0]}" if addr else "the " + P.building_en(say)
        if key == "on_roof_of":
            return f"on the roof of {nm}"
        if key == "next_to":
            return f"next to {nm}" if bank == "train" else f"right beside {nm}"
        return f"{P.EN_DIR[bank][key.split('_')[0]]} {nm}"
    if ref["type"] == "open":
        return (
            "in the open, away from buildings"
            if bank == "train"
            else "on empty ground with nothing built nearby"
        )
    if ref["type"] == "parking":
        return "in the car park" if bank == "train" else "on the parking lot"
    if ref["type"] == "street":
        return (
            f"by the road on {ref['name']}" if bank == "train" else f"along the carriageway of {ref['name']}"
        )
    if ref["type"] == "object":
        c = (
            (P.EN_COL[bank][ref["colour"]] + " ")
            if ref.get("colour") and ref["kind"] not in ("solar", "windsock")
            else ""
        )
        return (
            f"next to the {c}{P.EN_KIND['train'][ref['kind']][0]}"
            if bank == "train"
            else f"a few steps from the {c}{P.EN_KIND['test'][ref['kind']][0]}"
        )
    return (
        f"next to the {P.EN_LANDMARK[ref['type']]}"
        if bank == "train"
        else f"close to the {P.EN_LANDMARK[ref['type']]}"
    )


def text_en(steps, ret, bank, rng):
    sents = []
    for i, st in enumerate(steps):
        t, used, act, par = st["obj"], st["used"], st["action"], st["params"]
        d = dict(used)
        C = P.EN_COL[bank]
        pre = []
        if "size" in d:
            pre.append(P.EN_SIZE[bank][d["size"]])
        if "colour" in d and t.kind != "solar":
            pre.append(C[d["colour"]])
        if "shape" in d:
            pre.append("round" if d["shape"] == "round" else "square")
        nouns = P.EN_KIND[bank][t.kind]
        noun = nouns[int(rng.integers(len(nouns)))]
        post = []
        if "mark" in d:
            post.append(
                P.EN_MARK[d["mark"]]
                if bank == "train"
                else {
                    "H": "marked with a big H",
                    "circle": "with a ring painted on it",
                    "X": "marked with an X",
                }[d["mark"]]
            )
        if "pattern" in d:
            post.append(
                {"X": "with a big X", "rings": "with target rings", "cross": "with a plus-shaped cross"}[
                    d["pattern"]
                ]
            )
        if "lights" in d:
            post.append("with edge lights")
        if "flags" in d:
            post.append("with flags at the corners")
        if "style" in d:
            post.append(
                {
                    "locker": "that looks like a locker cabinet",
                    "platform": "on a low platform",
                    "ridge": "with a ridge roof",
                    "dome": "shaped like a dome",
                    "legs": "on four legs",
                    "column": "on a single column",
                }[d["style"]]
            )
        if "parcel_colour" in d:
            post.append(f"with a {C[d['parcel_colour']]} box on top")
        if "door_open" in d:
            post.append("with its doors open" if d["door_open"] else "with its doors shut")
        if "rows" in d:
            post.append({1: "in one row", 2: "in two rows", 3: "in three rows"}[d["rows"]])
        if "size_m" in d:
            post.append(f"about {d['size_m']:g} m {'across' if t.kind != 'container' else 'long'}")
        if "height_m" in d:
            post.append(f"about {d['height_m']:g} m tall")
        if "on_roof" in d:
            post.append("on a rooftop" if bank == "train" else "up on a building roof")
        verbs = EN_VERB[bank][act]
        v = verbs[int(rng.integers(len(verbs)))]
        s = f"{v} {' '.join(pre + [noun])}" + ((" " + ", ".join(post)) if post else "")
        if act == "enter":
            s += (
                " through the open doors and land inside" if bank == "train" else " and set down on the floor"
            )
        if act == "inspect" and par.get("distance_m"):
            s += ". " + (
                f"Stay {par['distance_m']} m away from {'them' if t.kind == 'solar' else 'it'}"
                if bank == "train"
                else f"Keep {par['distance_m']} m of stand-off"
            )
        if act == "flyby" and par.get("altitude_m"):
            s += ". " + (
                f"Keep an altitude of {par['altitude_m']} m"
                if bank == "train"
                else f"Hold {par['altitude_m']} m above ground"
            )
        if i == 1:
            s = ("Then " if bank == "train" else "After that, ") + s[0].lower() + s[1:]
        sents.append(s + ".")
        if (
            st["rels"] and i == 0 and len(steps) == 1 and rng.random() < 0.35
        ):  # relation first: 'Next to the pitch there is ... Land on it.'
            rel = " and ".join(_en_rel(k, r, bank, rng) for k, r in st["rels"])
            it = "them" if t.kind == "solar" else "it"
            short = (
                {
                    "land": f"Land on {it}",
                    "drop": f"Drop the parcel over {it}",
                    "pickup": "Pick the parcel up there",
                    "enter": "Fly in through the open doors and land inside",
                    "inspect": f"Circle {it}",
                    "flyby": f"Fly past {it}",
                }[act]
                if bank == "train"
                else {
                    "land": f"Settle onto {it}",
                    "drop": f"Release the box above {it}",
                    "pickup": "Grab the delivery there",
                    "enter": "Steer inside and set down",
                    "inspect": f"Orbit {it}",
                    "flyby": f"Skim past {it}",
                }[act]
            )
            sents[-1] = (
                f"{rel[0].upper() + rel[1:]} there {'are' if t.kind == 'solar' else 'is'} a{'n' if (pre or [noun])[0][0] in 'aeiou' else ''} {' '.join(pre + [noun])}"
                + ((" " + ", ".join(post)) if post else "")
                + f". {short}"
                + (f", {par['distance_m']} m away" if par.get("distance_m") else "")
                + (f" at {par['altitude_m']} m" if par.get("altitude_m") else "")
                + "."
            )
        elif st["rels"]:
            sents.append(
                ("It is " if bank == "train" else "You'll find it ")
                + " and ".join(_en_rel(k, r, bank, rng) for k, r in st["rels"])
                + "."
            )
    if ret is True:
        sents.append("Then return home." if bank == "train" else "Afterwards fly back to the launch point.")
    elif ret is False:
        sents.append("No need to come back." if bank == "train" else "Stay there; no return flight.")
    return " ".join(sents)


#
def truth_of(steps, ret):
    out = {"steps": [], "return_home": ret}
    for st in steps:
        tgt = {"kind": st["obj"].kind}
        for k, v in st["used"]:
            tgt[k] = v
        rels = []
        for k, r in st["rels"]:
            ref = {kk: vv for kk, vv in r.items() if kk not in ("b", "say")}
            if r["type"] == "building":
                say = r["say"]
                ref.update({k: (list(v) if isinstance(v, tuple) else v) for k, v in say.items()})
            rels.append({"rel": k, "ref": ref})
        out["steps"].append(
            {"action": st["action"], "target": tgt, "relations": rels, "params": st["params"]}
        )
    return out


def make(W, rng, bank="train", lang=None, detail=None, two_step=None, action=None):
    """One mission in world W."""
    lang = lang or ("ru" if rng.random() < 0.6 else "en")
    detail = detail or str(rng.choice(["full", "medium", "sparse"], p=[0.4, 0.4, 0.2]))
    two_step = (rng.random() < 0.15) if two_step is None else two_step
    two_step = two_step and action is None
    prng = np.random.default_rng(int(rng.integers(1 << 30)))
    for attempt in range(40):
        S = Scene(W, rng)
        steps = []
        acts = ["pickup", "drop"] if two_step else [action or str(rng.choice(ACTIONS, p=ACTION_P))]
        ok = True
        for act in acts:
            kind = str(rng.choice(KIND_FOR[act]))
            roof = kind == "pad" and act == "land" and rng.random() < 0.25
            hard = rng.random() < 0.6
            centre = None if not steps else (steps[-1]["obj"].x, steps[-1]["obj"].y)
            t = S.put(
                kind,
                near=centre,
                r=(80, 250) if centre else (0, 0),
                roof=roof,
                oid=f"t{len(steps)}",
                **({"door_open": True} if act == "enter" and rng.random() < 0.9 else {}),
            )
            if t is None or math.hypot(t.x - W.start[0], t.y - W.start[1]) < 50:
                ok = False
                break
            n_twin = int(rng.integers(1, 3)) if hard else 0
            for _ in range(n_twin):  # look-alikes: same kind and colour (sizes / marks may differ)
                kw = (
                    {"colour": t.facts["colour"]}
                    if "colour" in t.facts and kind not in ("solar", "windsock")
                    else {}
                )
                if kind == "hangar":
                    kw["door_open"] = bool(rng.random() < 0.7)
                S.put(kind, near=(t.x, t.y), r=(9, 45), roof=roof and rng.random() < 0.5, **kw)
            for _ in range(
                int(rng.integers(1, 4))
            ):  # other objects around: same kind other colour, or other kinds
                k2 = (
                    kind
                    if rng.random() < 0.4
                    else str(rng.choice(["pad", "dropzone", "tent", "container", "pickup", "solar"]))
                )
                S.put(k2, near=(t.x, t.y), r=(9, 50))
            if rng.random() < 0.4:
                S.put(str(rng.choice(["windsock", "mast", "water_tower"])), near=(t.x, t.y), r=(8, 25))
            par = {}
            if act == "inspect":
                par["distance_m"] = int(rng.choice([5, 8, 10, 12, 15]))
            if act == "flyby":
                par["altitude_m"] = int(rng.choice([10, 15, 20, 25, 30, 40]))
            steps.append({"obj": t, "action": act, "params": par})
        if not ok:
            continue
        for _ in range(int(rng.integers(3, 9))):  # clutter elsewhere in the area
            S.put(
                str(
                    rng.choice(
                        ["pad", "dropzone", "tent", "container", "pickup", "solar", "hangar", "windsock"]
                    )
                )
            )
        good = True
        for st in steps:
            used, rels, unique = describe_facts(st["obj"], S.objs, W, prng, detail)
            if not unique and detail != "sparse":
                good = False
                break
            st["used"], st["rels"], st["unique"] = used, rels, unique
        if not good:
            continue
        if detail != "sparse":  # the description must single out its object for ANY reading
            import grounding as G

            H = [G.perfect_hyp(o) for o in S.objs]
            T = truth_of(steps, None)
            if any(
                G.fits_all(h, T["steps"][i], H, W.map)
                for i, st in enumerate(steps)
                for h in H
                if h["oid"] != st["obj"].oid
            ):
                continue
        ret = None if rng.random() < 0.2 else bool(rng.random() < 0.55)
        text = text_ru(steps, ret, bank, prng) if lang == "ru" else text_en(steps, ret, bank, prng)
        rough = []
        for st in steps:  # the operator's rough point: 10-80 m off (sometimes 80-150 m)
            e = rng.uniform(80, 150) if rng.random() < 0.12 else rng.uniform(10, 80)
            a = rng.uniform(0, 2 * math.pi)
            rough.append((float(st["obj"].x + e * math.cos(a)), float(st["obj"].y + e * math.sin(a))))
        return {
            "text": text,
            "lang": lang,
            "bank": bank,
            "detail": detail,
            "return_home": ret,
            "objects": S.objs,
            "rough": rough,
            "steps": [
                {
                    "target": st["obj"].oid,
                    "action": st["action"],
                    "params": st["params"],
                    "unique": st["unique"],
                }
                for st in steps
            ],
            "truth": truth_of(steps, ret),
            "attempts": attempt + 1,
        }
    return None
