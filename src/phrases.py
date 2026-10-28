"""Stage F: Russian and English words for mission descriptions."""

import random

CASES = ("nom", "gen", "dat", "acc", "ins", "pre")


def adj(a, gender, case):
    """Decline a Russian adjective given in the masculine nominative."""
    if case == "nom" and gender == "m":
        return a
    stem, end = a[:-2], a[-2:]
    last = stem[-1] if stem else ""
    sib, velar = last in "жшчщ", last in "гкх"
    if end == "ий" and stem.endswith("н") and not a.endswith(("кий", "гий", "хий")) and a not in ("длинный",):
        kind = "soft"  # синий, летний
    elif end == "ий" and sib:
        kind = "sib"  # рыжий, свежий
    elif end == "ий" and velar:
        kind = "velar"  # маленький, высокий
    elif end == "ой":
        kind = "stress_k" if (velar or sib) else "stress"  # большой, голубой
    else:
        kind = "hard"  # красный
    T = {
        "hard": {
            "m": ("ый", "ого", "ому", "ый", "ым", "ом"),
            "f": ("ая", "ой", "ой", "ую", "ой", "ой"),
            "n": ("ое", "ого", "ому", "ое", "ым", "ом"),
            "pl": ("ые", "ых", "ым", "ые", "ыми", "ых"),
        },
        "soft": {
            "m": ("ий", "его", "ему", "ий", "им", "ем"),
            "f": ("яя", "ей", "ей", "юю", "ей", "ей"),
            "n": ("ее", "его", "ему", "ее", "им", "ем"),
            "pl": ("ие", "их", "им", "ие", "ими", "их"),
        },
        "sib": {
            "m": ("ий", "его", "ему", "ий", "им", "ем"),
            "f": ("ая", "ей", "ей", "ую", "ей", "ей"),
            "n": ("ее", "его", "ему", "ее", "им", "ем"),
            "pl": ("ие", "их", "им", "ие", "ими", "их"),
        },
        "velar": {
            "m": ("ий", "ого", "ому", "ий", "им", "ом"),
            "f": ("ая", "ой", "ой", "ую", "ой", "ой"),
            "n": ("ое", "ого", "ому", "ое", "им", "ом"),
            "pl": ("ие", "их", "им", "ие", "ими", "их"),
        },
        "stress": {
            "m": ("ой", "ого", "ому", "ой", "ым", "ом"),
            "f": ("ая", "ой", "ой", "ую", "ой", "ой"),
            "n": ("ое", "ого", "ому", "ое", "ым", "ом"),
            "pl": ("ые", "ых", "ым", "ые", "ыми", "ых"),
        },
        "stress_k": {
            "m": ("ой", "ого", "ому", "ой", "им", "ом"),
            "f": ("ая", "ой", "ой", "ую", "ой", "ой"),
            "n": ("ое", "ого", "ому", "ое", "им", "ом"),
            "pl": ("ие", "их", "им", "ие", "ими", "их"),
        },
    }
    return stem + T[kind][gender][CASES.index(case)]


class N:
    """A Russian noun phrase head."""

    def __init__(self, forms, g, adj_=None, tail=""):
        self.f = forms.split()
        self.g, self.adj, self.tail = g, adj_, tail
        assert len(self.f) == 6, forms

    def __call__(self, case, adjs=()):
        words = (
            [adj(a, self.g, case) for a in adjs]
            + ([adj(self.adj, self.g, case)] if self.adj else [])
            + [self.f[CASES.index(case)]]
        )
        return " ".join(words) + (" " + self.tail if self.tail else "")


# ---------------- object kinds
RU_KIND = {
    "train": {
        "pad": [
            N("площадка площадки площадке площадку площадкой площадке", "f", "посадочный"),
            N("площадка площадки площадке площадку площадкой площадке", "f", tail="для посадки"),
            N("площадка площадки площадке площадку площадкой площадке", "f"),
        ],
        "dropzone": [
            N("зона зоны зоне зону зоной зоне", "f", tail="сброса"),
            N("площадка площадки площадке площадку площадкой площадке", "f", tail="для сброса"),
        ],
        "pickup": [
            N("пункт пункта пункту пункт пунктом пункте", "m", tail="выдачи"),
            N("площадка площадки площадке площадку площадкой площадке", "f", tail="с посылкой"),
        ],
        "hangar": [N("ангар ангара ангару ангар ангаром ангаре", "m")],
        "water_tower": [N("башня башни башне башню башней башне", "f", "водонапорный")],
        "mast": [
            N("вышка вышки вышке вышку вышкой вышке", "f", tail="связи"),
            N("мачта мачты мачте мачту мачтой мачте", "f", "антенный"),
        ],
        "tent": [N("палатка палатки палатке палатку палаткой палатке", "f")],
        "container": [
            N("контейнер контейнера контейнеру контейнер контейнером контейнере", "m"),
            N("контейнер контейнера контейнеру контейнер контейнером контейнере", "m", "морской"),
        ],
        "solar": [N("панели панелей панелям панели панелями панелях", "pl", "солнечный")],
        "windsock": [
            N(
                "ветроуказатель ветроуказателя ветроуказателю ветроуказатель ветроуказателем ветроуказателе",
                "m",
            )
        ],
    },
    "test": {
        "pad": [
            N("круг круга кругу круг кругом круге", "m", "посадочный"),
            N("площадка площадки площадке площадку площадкой площадке", "f", "вертолётный"),
        ],
        "dropzone": [
            N("мишень мишени мишени мишень мишенью мишени", "f", tail="для сброса"),
            N("место места месту место местом месте", "n", tail="выброски груза"),
        ],
        "pickup": [
            N("точка точки точке точку точкой точке", "f", tail="выдачи груза"),
            N("постамат постамата постамату постамат постаматом постамате", "m"),
        ],
        "hangar": [
            N("бокс бокса боксу бокс боксом боксе", "m", "ангарный"),
            N("гараж гаража гаражу гараж гаражом гараже", "m", tail="для техники"),
        ],
        "water_tower": [
            N("водокачка водокачки водокачке водокачку водокачкой водокачке", "f"),
            N("резервуар резервуара резервуару резервуар резервуаром резервуаре", "m", "водонапорный"),
        ],
        "mast": [
            N("вышка вышки вышке вышку вышкой вышке", "f", "сотовый"),
            N("мачта мачты мачте мачту мачтой мачте", "f", "решётчатый"),
        ],
        "tent": [
            N("шатёр шатра шатру шатёр шатром шатре", "m", "походный"),
            N("палатка палатки палатке палатку палаткой палатке", "f", "туристический"),
        ],
        "container": [
            N("контейнер контейнера контейнеру контейнер контейнером контейнере", "m", "грузовой"),
            N("бокс бокса боксу бокс боксом боксе", "m", "металлический"),
        ],
        "solar": [
            N("батарея батареи батарее батарею батареей батарее", "f", "солнечный"),
            N("модули модулей модулям модули модулями модулях", "pl", "фотоэлектрический"),
        ],
        "windsock": [N("конус конуса конусу конус конусом конусе", "m", "полосатый", "на шесте")],
    },
}
EN_KIND = {
    "train": {
        "pad": ["landing pad", "helipad"],
        "dropzone": ["drop zone", "drop target"],
        "pickup": ["pickup point", "parcel station"],
        "hangar": ["hangar"],
        "water_tower": ["water tower"],
        "mast": ["radio mast", "cell tower"],
        "tent": ["tent"],
        "container": ["container", "shipping container"],
        "solar": ["solar panels"],
        "windsock": ["windsock"],
    },
    "test": {
        "pad": ["touchdown circle", "landing spot"],
        "dropzone": ["supply drop marker", "release target"],
        "pickup": ["collection point", "parcel locker"],
        "hangar": ["drone shed", "storage bay"],
        "water_tower": ["elevated tank", "water tank on stilts"],
        "mast": ["antenna tower", "lattice mast"],
        "tent": ["camping shelter", "canvas shelter"],
        "container": ["cargo box", "freight container"],
        "solar": ["photovoltaic array", "PV panels"],
        "windsock": ["wind cone", "striped wind sleeve"],
    },
}
RU_COL = {
    "train": {
        "red": "красный",
        "orange": "оранжевый",
        "yellow": "жёлтый",
        "green": "зелёный",
        "blue": "синий",
        "white": "белый",
        "grey": "серый",
        "black": "чёрный",
        "purple": "фиолетовый",
        "brown": "коричневый",
    },
    "test": {
        "red": "алый",
        "orange": "рыжий",
        "yellow": "золотистый",
        "green": "изумрудный",
        "blue": "тёмно-синий",
        "white": "белоснежный",
        "grey": "пепельный",
        "black": "угольно-чёрный",
        "purple": "лиловый",
        "brown": "бурый",
    },
}
EN_COL = {
    "train": {
        c: c
        for c in ("red", "orange", "yellow", "green", "blue", "white", "grey", "black", "purple", "brown")
    },
    "test": {
        "red": "crimson",
        "orange": "tangerine",
        "yellow": "golden",
        "green": "emerald",
        "blue": "navy",
        "white": "snow-white",
        "grey": "ash-grey",
        "black": "jet-black",
        "purple": "violet",
        "brown": "chestnut",
    },
}
RU_SIZE = {
    "train": {"small": "маленький", "large": "большой"},
    "test": {"small": "компактный", "large": "крупный"},
}
EN_SIZE = {"train": {"small": "small", "large": "large"}, "test": {"small": "compact", "large": "oversized"}}
RU_DIR = {
    "train": {"north": "к северу от", "south": "к югу от", "east": "к востоку от", "west": "к западу от"},
    "test": {"north": "севернее", "south": "южнее", "east": "восточнее", "west": "западнее"},
}
EN_DIR = {
    "train": {d: f"{d} of" for d in ("north", "south", "east", "west")},
    "test": {
        "north": "on the northern side of",
        "south": "on the southern side of",
        "east": "on the sunrise side of",
        "west": "on the sunset side of",
    },
}
NUM_STOREY = {
    1: "одноэтажный",
    2: "двухэтажный",
    3: "трёхэтажный",
    4: "четырёхэтажный",
    5: "пятиэтажный",
    6: "шестиэтажный",
    7: "семиэтажный",
    8: "восьмиэтажный",
    9: "девятиэтажный",
    10: "десятиэтажный",
    12: "двенадцатиэтажный",
    14: "четырнадцатиэтажный",
    16: "шестнадцатиэтажный",
    17: "семнадцатиэтажный",
}
EN_NUM = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    12: "twelve",
    14: "fourteen",
    16: "sixteen",
    17: "seventeen",
}

# buildings and landmarks (reference objects of relations)
DOM = N("дом дома дому дом домом доме", "m")
ZDANIE = N("здание здания зданию здание зданием здании", "n")
RU_LANDMARK = {
    "pitch": N("поле поля полю поле полем поле", "n", "футбольный"),
    "playground": N("площадка площадки площадке площадку площадкой площадке", "f", "детский"),
    "parking": N("парковка парковки парковке парковку парковкой парковке", "f"),
    "track": N("стадион стадиона стадиону стадион стадионом стадионе", "m"),
    "church": N("церковь церкви церкви церковь церковью церкви", "f"),
    "water": N("пруд пруда пруду пруд прудом пруду", "m"),
    "power_line": N("линия линии линии линию линией линии", "f", tail="электропередачи"),
    "road": N("дорога дороги дороге дорогу дорогой дороге", "f"),
    "water_tower": N("башня башни башне башню башней башне", "f", "водонапорный"),
    "mast": N("вышка вышки вышке вышку вышкой вышке", "f", tail="связи"),
    "chimney": N("труба трубы трубе трубу трубой трубе", "f", "заводской"),
    "tower": N("вышка вышки вышке вышку вышкой вышке", "f"),
    "storage_tank": N("резервуар резервуара резервуару резервуар резервуаром резервуаре", "m"),
    "flagpole": N("флагшток флагштока флагштоку флагшток флагштоком флагштоке", "m"),
}
EN_LANDMARK = {
    "pitch": "football pitch",
    "playground": "playground",
    "parking": "car park",
    "track": "running track",
    "church": "church",
    "water": "pond",
    "power_line": "power line",
    "road": "road",
    "water_tower": "water tower",
    "mast": "radio mast",
    "chimney": "factory chimney",
    "tower": "tower",
    "storage_tank": "storage tank",
    "flagpole": "flagpole",
}
RU_MARK = {"H": "с белой буквой H", "circle": "с белым кругом", "X": "с крестом"}
EN_MARK = {"H": "with a white H", "circle": "with a painted circle", "X": "with a cross"}


def ru_metres(x):
    """'около 4 метров' / 'около 1 метра' / 'около 21 метра' / 'около 4,5 метра' (genitive after 'около')."""
    if abs(x - round(x)) > 0.01:
        return f"{str(round(x, 1)).replace('.', ',')} метра"
    n = int(round(x))
    return f"{n} " + ("метра" if n % 10 == 1 and n % 100 != 11 else "метров")


STREET_NOUNS = {  # head nouns of Russian street names: gender and case forms
    "улица": ("f", "улица улицы улице улицу улицей улице"),
    "проспект": ("m", "проспект проспекта проспекту проспект проспектом проспекте"),
    "переулок": ("m", "переулок переулка переулку переулок переулком переулке"),
    "бульвар": ("m", "бульвар бульвара бульвару бульвар бульваром бульваре"),
    "проезд": ("m", "проезд проезда проезду проезд проездом проезде"),
    "площадь": ("f", "площадь площади площади площадь площадью площади"),
    "аллея": ("f", "аллея аллеи аллее аллею аллеей аллее"),
    "тупик": ("m", "тупик тупика тупику тупик тупиком тупике"),
    "линия": ("f", "линия линии линии линию линией линии"),
    "дорога": ("f", "дорога дороги дороге дорогу дорогой дороге"),
    "шоссе": ("n", "шоссе шоссе шоссе шоссе шоссе шоссе"),
    "набережная": ("f", "набережная набережной набережной набережную набережной набережной"),
    "тракт": ("m", "тракт тракта тракту тракт трактом тракте"),
    "мост": ("m", "мост моста мосту мост мостом мосте"),
}


def street_ru(name, case):
    """'Советская улица' -> 'Советской улице'."""
    toks = name.split()
    low = [t.lower() for t in toks]
    head = next((i for i, t in enumerate(low) if t in STREET_NOUNS), None)
    if head is None or not all("а" <= c.lower() <= "я" or c in "ёЁ -.0123456789" for c in name):
        return N("улица улицы улице улицу улицей улице", "f")(case) + " " + name
    g, forms = STREET_NOUNS[low[head]]
    out = []

    def dec(word):
        """Decline one adjectival street-name word."""
        if "-" in word and not word[0].isdigit():
            return "-".join(dec(w) for w in word.split("-"))
        if word[0].isdigit() and word.endswith(("-я", "-й", "-е")):
            return word[:-1] + (
                {"f": "й", "m": "го", "n": "го"}[g]
                if case in ("gen",)
                else (
                    "й"
                    if g == "f" and case != "acc"
                    else (
                        "ю"
                        if g == "f"
                        else "му" if case == "dat" else "м" if case in ("ins", "pre") else word[-1]
                    )
                )
            )
        e = word[-2:].lower()
        if e not in ("ая", "ий", "ый", "ой", "яя", "ое", "ее"):
            return word
        pre_ = word[-3:-2].lower()
        base = word[:-2] + ("ний"[1:] if e in ("яя", "ее") else "ий" if pre_ in "гкхжшчщ" else "ый")
        if e in ("яя", "ее"):
            base = word[:-2] + "ий"
        if e == "ой":
            base = word  # stressed ending: большой, Большой
        w = adj(base.lower(), g, case)
        return w[0].upper() + w[1:] if word[0].isupper() else w

    for i, t in enumerate(toks):
        if i == head:
            f = forms.split()[CASES.index(case)]
            out.append(f if t[0].islower() else f.capitalize())
        elif i < head:
            out.append(dec(t))
        else:
            out.append(t)
    return " ".join(out)


def btype_of(b):
    """Coarse building type a person names: house, garage, church, warehouse, office, apartment."""
    st, t = b["style"], b["type"]
    if t in ("church", "cathedral", "chapel"):
        return "church"
    if t in ("garage", "garages") or st == "garage":
        return "garage"
    if st == "industrial":
        return "warehouse"
    if st.startswith("house"):
        return "house"
    if st.startswith("glass"):
        return "office"
    return "apartment"


def say_building(b, rng):
    """Which facts about a mapped building the description mentions."""
    s = {"btype": btype_of(b)}
    st = b["style"]
    if s["btype"] == "apartment":
        if b["levels"] and rng.random() < 0.8:
            s["levels"] = int(b["levels"])
        if (
            b["levels"]
            and not any(k in s for k in ("levels",))
            and not st.startswith(("panel", "brick", "plaster"))
        ):
            s["levels"] = int(b["levels"])
        if st.startswith("panel") and rng.random() < 0.7:
            s["material"] = "panel"
        elif st.startswith("brick") and rng.random() < 0.75:
            s["material"] = "brick"
            s["colour"] = "white" if st == "brick_silicate" else None
        elif st.startswith("plaster") and rng.random() < 0.8:
            s["colour"] = st.split("_")[1]
    elif s["btype"] == "office" and b["levels"] and rng.random() < 0.6:
        s["levels"] = int(b["levels"])
    elif s["btype"] == "house" and st == "house_wood" and rng.random() < 0.7:
        s["material"] = "wood"
    if b.get("street") and b.get("number") and rng.random() < 0.5:
        s["address"] = (b["street"], b["number"])
    if (
        s["btype"] == "apartment"
        and not any(k in s for k in ("levels", "material", "colour", "address"))
        and b["levels"]
    ):
        s["levels"] = int(b["levels"])
    return {k: v for k, v in s.items() if v is not None}


COL_RU_B = {"yellow": "жёлтый", "pink": "розовый", "green": "зелёный", "white": "белый"}


def building_ru(say, case, bank="train"):
    t = say["btype"]
    if t == "church":
        return RU_LANDMARK["church"](case)
    if t == "garage":
        return N("гараж гаража гаражу гараж гаражом гараже", "m")(case)
    if t == "warehouse":
        return (
            N("склад склада складу склад складом складе", "m")(case)
            if bank == "train"
            else N("ангар ангара ангару ангар ангаром ангаре", "m", "складской")(case)
        )
    if t == "house":
        return DOM(case, (["деревянный"] if say.get("material") == "wood" else []) + ["частный"])
    adjs = []
    lv = say.get("levels")
    if lv:
        adjs.append(NUM_STOREY.get(lv, f"{lv}-этажный"))
    if say.get("colour"):
        adjs.append(COL_RU_B[say["colour"]])
    if say.get("material") == "panel":
        adjs.append("панельный")
    if say.get("material") == "brick":
        adjs.append("кирпичный")
    return (ZDANIE if t == "office" else DOM)(case, adjs + (["стеклянный"] if t == "office" else []))


def building_en(say):
    t = say["btype"]
    if t in ("church", "garage", "warehouse"):
        return t
    if t == "house":
        return ("wooden " if say.get("material") == "wood" else "") + "house"
    w = []
    if say.get("levels"):
        w.append(f"{EN_NUM.get(say['levels'], str(say['levels']))}-storey")
    if say.get("colour"):
        w.append(say["colour"])
    if say.get("material") in ("panel", "brick"):
        w.append(say["material"])
    return " ".join(
        w
        + (
            ["glass office building"]
            if t == "office"
            else ["block" if say.get("material") == "panel" else "building"]
        )
    )
