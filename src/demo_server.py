"""The drone app: a page in the browser with the map, a 3D view, the task and the drone's state.
usage: demo_server.py [--place kazan_novosav] [--seed 0] [--truth-perception] [--speed 1] [--viewer]"""

import argparse, base64, http.server, io, json, math, re, socketserver, threading, time, zlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "demo"
MODELS = HERE.parent / "models"
COLOUR_RU = {
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
}
KIND_RU = {
    "pad": "посадочная площадка",
    "dropzone": "зона сброса",
    "pickup": "пункт выдачи",
    "hangar": "ангар",
    "water_tower": "водонапорная башня",
    "mast": "вышка связи",
    "tent": "палатка",
    "container": "контейнер",
    "solar": "солнечные панели",
    "windsock": "ветроуказатель",
}
# object facts in plain Russian for the map tooltips
FACT_RU = {
    "size_m": "размер",
    "height_m": "высота",
    "shape": "форма",
    "mark": "знак",
    "mark_colour": "цвет знака",
    "lights": "огни",
    "pattern": "рисунок",
    "flags": "флажки",
    "style": "вид",
    "parcel_colour": "цвет посылки",
    "door_open": "ворота",
    "rows": "рядов панелей",
}
VALUE_RU = {
    "round": "круглая",
    "square": "квадратная",
    "circle": "круг",
    "X": "крест X",
    "H": "буква H",
    "rings": "кольца",
    "cross": "крест",
    "platform": "платформа",
    "locker": "постамат",
    "legs": "на опорах",
    "column": "на колонне",
    "ridge": "двускатная",
    "dome": "купол",
}
# conditions the operator can switch on before the flight
GPS_KINDS = {"off": "GPS пропал", "drift": "GPS подменён (уводит в сторону)", "degraded": "GPS стал неточным"}
LIDAR_KINDS = {
    "sector": ("sector", 0.5, "грязь на стекле лидара: слепой сектор 90°"),
    "range": ("range", 0.35, "дождь: дальность лидара втрое меньше"),
    "spurious": ("spurious", 300.0, "помехи: лидар видит ложные точки"),
}
FOG = {"light": 250.0, "dense": 80.0}  # metres of visibility
GUST = {"none": 0.0, "weak": 0.25, "strong": 0.6}  # gust strength as a share of the wind


ACT_RU = {
    "land": "сесть на",
    "drop": "сбросить груз над",
    "pickup": "забрать посылку с",
    "enter": "влететь и сесть в",
    "inspect": "облететь по кругу",
    "flyby": "пролететь мимо",
}
REL_RU = {
    "next_to": "рядом с",
    "north_of": "к северу от",
    "south_of": "к югу от",
    "east_of": "к востоку от",
    "west_of": "к западу от",
    "on_roof_of": "на крыше",
    "on": "на",
    "by_street": "у улицы",
    "in_open": "на открытом месте",
}


REF_RU = {
    "building": "дом",
    "pitch": "спортплощадка",
    "playground": "детская площадка",
    "parking": "парковка",
    "track": "беговая дорожка",
    "church": "церковь",
    "water": "вода",
    "power_line": "ЛЭП",
    "street": "улица",
    "open": "открытое место",
}
BTYPE_RU = {
    "apartment": "многоквартирный",
    "house": "частный",
    "garage": "гараж",
    "warehouse": "склад",
    "office": "офисный",
    "church": "церковь",
}
SIZE_RU = {"small": "маленький", "large": "большой"}
MATERIAL_RU = {
    "panel": "панельный",
    "brick": "кирпичный",
    "wood": "деревянный",
    "concrete": "бетонный",
    "metal": "металлический",
    "glass": "стеклянный",
    "plaster": "оштукатуренный",
}


def target_ru(t):
    """What the drone will look for, in words: kind first, then each stated attribute."""
    out = [KIND_RU.get(t.get("kind"), t.get("kind") or "объект")]
    for k, v in t.items():
        if k == "kind" or v is None:
            continue
        if k == "colour":
            out.append("цвет " + COLOUR_RU.get(v, str(v)))
        elif k == "size":
            out.append(SIZE_RU.get(v, str(v)))
        elif k in ("size_m", "height_m"):
            out.append(f"{FACT_RU[k]} около {float(v):.0f} м")
        elif k == "door_open":
            out.append("ворота открыты" if v else "ворота закрыты")
        elif k == "on_roof":
            out.append("на крыше" if v else "не на крыше")
        elif k in ("mark_colour", "parcel_colour"):
            out.append(f"{FACT_RU[k]} {COLOUR_RU.get(v, str(v))}")
        elif isinstance(v, bool):
            out.append(f"{FACT_RU.get(k, k)}: {'есть' if v else 'нет'}")
        else:
            out.append(f"{FACT_RU.get(k, k)}: {VALUE_RU.get(v, v)}")
    return ", ".join(out)


def ref_ru(r):
    typ = r.get("type")
    name = KIND_RU.get(r.get("kind"), "объект") if typ == "object" or typ in KIND_RU else REF_RU.get(typ, typ or "объект")
    extra = []
    if r.get("btype"):
        extra.append(BTYPE_RU.get(r["btype"], r["btype"]))
    if r.get("levels"):
        extra.append(f"этажей: {r['levels']}")
    if r.get("material"):
        extra.append(MATERIAL_RU.get(r["material"], str(r["material"])))
    if r.get("colour"):
        extra.append("цвет " + COLOUR_RU.get(r["colour"], str(r["colour"])))
    if r.get("address"):
        extra.append("адрес: " + " ".join(map(str, r["address"])))
    if r.get("name"):
        extra.append(f"«{r['name']}»")
    return name + (f" ({', '.join(extra)})" if extra else "")


def summary_ru(parsed):
    """The parse in plain Russian for the operator: one line per step and one about the return."""
    out = []
    for i, st in enumerate(parsed.get("steps") or []):
        if not isinstance(st, dict):
            continue
        line = f"Шаг {i + 1}. {ACT_RU.get(st.get('action'), st.get('action')).capitalize()}: {target_ru(st.get('target') or {})}"
        rels = []
        for x in st.get("relations") or []:
            if not isinstance(x, dict):
                continue
            ref = x.get("ref") or {}
            if x.get("rel") == "in_open" and set(ref) <= {"type"}:  # "in the open" needs no object
                rels.append(REL_RU["in_open"])
            else:
                rels.append(f"{REL_RU.get(x.get('rel'), x.get('rel'))}: {ref_ru(ref)}")
        if rels:
            line += "; " + "; ".join(rels)
        p = st.get("params") or {}
        if p.get("distance_m"):
            line += f"; на расстоянии {p['distance_m']} м"
        if p.get("altitude_m"):
            line += f"; на высоте {p['altitude_m']} м"
        out.append(line)
    rh = parsed.get("return_home")
    out.append("Потом: " + ("вернуться домой" if rh is True else "остаться на месте" if rh is False else "вернуться домой (в тексте не сказано)"))
    return out


def about_ru(o):
    """What the map tooltip says about an object."""
    out = []
    for k, v in o.facts.items():
        if k in ("kind", "colour") or k not in FACT_RU or v is None:
            continue
        if k == "door_open":
            v = "открыты" if v else "закрыты"
        elif k in ("size_m", "height_m"):
            v = f"{float(v):.1f} м".replace(".", ",")
        elif k in ("mark_colour", "parcel_colour"):
            v = COLOUR_RU.get(v, v)
        elif isinstance(v, bool):
            v = "есть" if v else "нет"
        else:
            v = VALUE_RU.get(v, v)
        out.append(f"{FACT_RU[k]}: {v}")
    if o.on_roof:
        out.append("стоит на крыше")
    return out


# ---------------- the same for an English page
KIND_EN = {
    "pad": "landing pad",
    "dropzone": "drop zone",
    "pickup": "pickup station",
    "hangar": "hangar",
    "water_tower": "water tower",
    "mast": "cell tower",
    "tent": "tent",
    "container": "container",
    "solar": "solar panels",
    "windsock": "windsock",
}
FACT_EN = {
    "size_m": "size",
    "height_m": "height",
    "shape": "shape",
    "mark": "mark",
    "mark_colour": "mark colour",
    "lights": "lights",
    "pattern": "pattern",
    "flags": "flags",
    "style": "style",
    "parcel_colour": "parcel colour",
    "door_open": "doors",
    "rows": "panel rows",
}
VALUE_EN = {"X": "a cross X", "H": "the letter H", "locker": "locker", "legs": "on legs", "column": "on a column", "ridge": "gable roof"}
ACT_EN = {
    "land": "land on",
    "drop": "drop the parcel over",
    "pickup": "pick up the parcel from",
    "enter": "fly into and land in",
    "inspect": "circle around",
    "flyby": "fly past",
}
REL_EN = {
    "next_to": "next to",
    "north_of": "north of",
    "south_of": "south of",
    "east_of": "east of",
    "west_of": "west of",
    "on_roof_of": "on the roof of",
    "on": "on",
    "by_street": "by the street",
    "in_open": "in the open",
}
REF_EN = {"open": "open ground", "power_line": "power line", "pitch": "sports pitch", "track": "running track"}
GPS_KINDS_EN = {"off": "GPS lost", "drift": "GPS spoofed (walks away)", "degraded": "GPS degraded"}
LIDAR_EN = {
    "sector": "dirt on the lidar window: a 90° blind sector",
    "range": "rain: lidar range one third",
    "spurious": "interference: false lidar points",
}


def target_en(t):
    out = [KIND_EN.get(t.get("kind"), t.get("kind") or "object")]
    for k, v in t.items():
        if k == "kind" or v is None:
            continue
        if k == "colour":
            out.append(f"colour {v}")
        elif k == "size":
            out.append(str(v))
        elif k in ("size_m", "height_m"):
            out.append(f"{FACT_EN[k]} about {float(v):.0f} m")
        elif k == "door_open":
            out.append("doors open" if v else "doors closed")
        elif k == "on_roof":
            out.append("on a roof" if v else "not on a roof")
        elif isinstance(v, bool):
            out.append(f"{FACT_EN.get(k, k)}: {'yes' if v else 'no'}")
        else:
            out.append(f"{FACT_EN.get(k, k)}: {VALUE_EN.get(v, v)}")
    return ", ".join(out)


def ref_en(r):
    typ = r.get("type")
    name = KIND_EN.get(r.get("kind"), "object") if typ == "object" or typ in KIND_EN else REF_EN.get(typ, typ or "object")
    extra = [str(r[k]) for k in ("btype", "material", "colour") if r.get(k)]
    if r.get("levels"):
        extra.append(f"{r['levels']} storeys")
    if r.get("address"):
        extra.append("address: " + " ".join(map(str, r["address"])))
    if r.get("name"):
        extra.append(f'"{r["name"]}"')
    return name + (f" ({', '.join(extra)})" if extra else "")


def summary_en(parsed):
    out = []
    for i, st in enumerate(parsed.get("steps") or []):
        if not isinstance(st, dict):
            continue
        line = f"Step {i + 1}. {ACT_EN.get(st.get('action'), st.get('action')).capitalize()}: {target_en(st.get('target') or {})}"
        rels = []
        for x in st.get("relations") or []:
            if not isinstance(x, dict):
                continue
            ref = x.get("ref") or {}
            if x.get("rel") == "in_open" and set(ref) <= {"type"}:
                rels.append(REL_EN["in_open"])
            else:
                rels.append(f"{REL_EN.get(x.get('rel'), x.get('rel'))}: {ref_en(ref)}")
        if rels:
            line += "; " + "; ".join(rels)
        p = st.get("params") or {}
        if p.get("distance_m"):
            line += f"; at {p['distance_m']} m"
        if p.get("altitude_m"):
            line += f"; at {p['altitude_m']} m altitude"
        out.append(line)
    rh = parsed.get("return_home")
    out.append("Then: " + ("return home" if rh is True else "stay there" if rh is False else "return home (the text does not say)"))
    return out


def about_en(o):
    out = []
    for k, v in o.facts.items():
        if k in ("kind", "colour") or k not in FACT_EN or v is None:
            continue
        if k == "door_open":
            v = "open" if v else "closed"
        elif k in ("size_m", "height_m"):
            v = f"{float(v):.1f} m"
        elif isinstance(v, bool):
            v = "yes" if v else "no"
        else:
            v = VALUE_EN.get(v, v)
        out.append(f"{FACT_EN[k]}: {v}")
    if o.on_roof:
        out.append("stands on a roof")
    return out


def jpeg(img, boxes=(), w=480):
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    s = w / im.width
    im = im.resize((w, int(im.height * s)))
    d = ImageDraw.Draw(im)
    for b, label, col in boxes:
        x0, y0, x1, y1 = [v * s for v in b]
        d.rectangle([x0, y0, x1, y1], outline=col, width=2)
        d.text((x0, max(0, y0 - 11)), label, fill=col)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


class Sim:
    def __init__(self, a):
        import city3d as C, percep_data as PD, mission_env as ME

        self.a = a
        rng = np.random.default_rng(zlib.crc32(f"demo/{a.place}/{a.seed}".encode()))
        W0, _, _ = C.build(a.place, a.seed, actors=False)
        self.W0 = W0
        objs = PD.fill_scene(W0, rng)  # mission objects in groups with look-alikes
        for k, o in enumerate(objs):
            o.oid = f"d{k}"
        import mission_objects as MO  # object ids are inside the xml names

        self.objs = [
            MO.make(
                f"d{k}",
                o.kind,
                o.x,
                o.y,
                o.yaw,
                o.z,
                np.random.default_rng(1000 + k),
                **(
                    {"colour": o.facts["colour"]}
                    if "colour" in o.facts and o.kind not in ("solar", "windsock")
                    else {}
                ),
            )
            for k, o in enumerate(objs)
        ]
        for o, src in zip(self.objs, objs):
            o.on_roof = src.on_roof
        self.env = ME.MissionEnv(
            a.place, a.seed, None, objs=self.objs, render=False, cam=(160, 120), gimbal=(960, 720)
        )
        self.env.reset()
        self.lock = threading.Lock()  # guards self.state
        self.step_lock = threading.Lock()  # the world is changed by one thread at a time
        self.state = {
            "phase": "idle",
            "feel": {},
            "cam_seq": 0,
            "frame_seq": 0,
            "question": None,
            "parsed": None,
            "result": None,
            "conditions": [],
            "view_error": None,
        }
        self.logs, self.log_base = [], 0  # every log line; the page asks only for the lines it has not seen
        self.cam_raw = None  # the last gimbal image and its boxes, encoded only when the page asks for it
        self.cam_jpg = (0, None)
        self.raw3d = None
        self.jpg3d = (0, None)
        self.t_req3d = 0.0  # last time the page asked for a 3D frame: nobody looking, nothing rendered
        self.answer = None
        self.answer_ev = threading.Event()
        self.brain = None
        self.speed = a.speed
        self.running = True
        self.mission_on = False
        self.points = []  # the operator's rough points, also drawn in 3D
        self.trail = []  # the drone's true path, drawn in 3D
        self.fog = 0.0
        # the 3D view on the page: the same MuJoCo world, seen by a camera that follows the drone
        self.view = {"mode": "chase", "azimuth": 135.0, "elevation": -25.0, "distance": 30.0}
        self.cam3d = None
        self.stop_req = False
        self.lang = "ru"
        self.t_frame = 0.0
        self.perc, self.parse = self._load_models()
        import mujoco, mujoco.viewer

        self.viewer = (
            mujoco.viewer.launch_passive(
                self.env.model, self.env.data, show_left_ui=False, show_right_ui=False
            )
            if a.viewer
            else None
        )
        if self.viewer is not None:
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.env.bid
            self.viewer.cam.distance = 14.0
            self.viewer.cam.elevation = -20.0
        threading.Thread(target=self.loop, daemon=True).start()

    def _load_models(self):
        det, attrs, adapter, base = (
            MODELS / "detector.pt",
            MODELS / "attrs_best.pt",
            MODELS / "parser_adapter",
            MODELS / "qwen1.5b",
        )
        perc = parse = None
        if not self.a.truth_perception and det.exists() and attrs.exists():
            import perceive as PC

            perc = PC.Perceiver(str(det), str(attrs))
            self.log("восприятие: обученный детектор и сеть признаков")
        else:
            import fly_mission as FM

            perc = FM.TruthPerceiver(self.env, np.random.default_rng(0))
            self.log("ВНИМАНИЕ: модели не найдены, восприятие подставлено по истине (отладка)")
        if adapter.exists() and base.exists():
            import torch, mission_parser as MP, train_parser as TP
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            tok = AutoTokenizer.from_pretrained(str(base))
            tok.pad_token = tok.pad_token or tok.eos_token
            lm = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.bfloat16, device_map="cuda"),
                str(adapter),
            ).eval()
            parse = lambda text: MP.parse_output(TP.generate(lm, tok, [{"text": text}], bs=1)[0])
            self.log("разбор задания: Qwen2.5-1.5B + LoRA")
        return self._wrap_perc(perc), parse

    def _wrap_perc(self, perc):
        def f(rgb, depth, cam, fovy):
            dets = perc(rgb, depth, cam, fovy)
            boxes = [
                (
                    d["box"],
                    f"{d['cls']} {d['conf']:.2f}",
                    (255, 220, 0) if d["cls"] != "bird" else (255, 80, 80),
                )
                for d in dets
                if "box" in d
            ]
            with self.lock:
                self.cam_raw = (rgb.copy(), boxes)
                self.state["cam_seq"] += 1
            return dets

        return f

    def log(self, msg):
        msg = re.sub(r"^\[\s*[\d.]+s\]\s*", "", str(msg))  # the brain adds its own time stamp
        with self.lock:
            self.logs.append(f"[{self.env.t:6.1f} {'s' if self.lang == 'en' else 'с'}] {msg}")
            if len(self.logs) > 1200:
                del self.logs[:200]
                self.log_base += 200

    def snapshot(self, since):
        """The state for the page plus the log lines after index `since` (copied under the lock, sent outside it)."""
        with self.lock:
            s = dict(self.state)
            n = self.log_base + len(self.logs)
            since = max(since, self.log_base, n - 300)
            s["log"] = self.logs[since - self.log_base :]
            s["log_from"], s["log_n"] = since, n
        return s

    def cam_jpeg(self):
        with self.lock:
            seq, raw = self.state["cam_seq"], self.cam_raw
            if self.cam_jpg[0] == seq:
                return self.cam_jpg[1]
        if raw is None:
            return None
        b = base64.b64decode(jpeg(raw[0], raw[1], w=560))
        with self.lock:
            self.cam_jpg = (seq, b)
        return b

    def frame_jpeg(self):
        """The latest 3D frame as JPEG (16:9 crop). The simulator thread only renders; encoding happens here."""
        from PIL import Image

        with self.lock:
            self.t_req3d = time.time()
            seq, raw = self.state["frame_seq"], self.raw3d
            if self.jpg3d[0] == seq:
                return self.jpg3d[1]
        if raw is None:
            return None
        h, w = raw.shape[:2]
        ch = w * 9 // 16
        buf = io.BytesIO()
        Image.fromarray(raw[(h - ch) // 2 : (h - ch) // 2 + ch]).save(buf, "JPEG", quality=80)
        with self.lock:
            self.jpg3d = (seq, buf.getvalue())
        return self.jpg3d[1]

    def stop(self):
        """The operator stops the flight: the drone holds where it is, the world reports what happened so far."""
        if not self.mission_on:
            return {"error": "полёт не идёт"}
        self.answer, self.stop_req = None, True
        self.answer_ev.set()  # a pending question returns at once
        return {"ok": True}

    def map_json(self):
        W = self.env.W
        return {
            "half": W.size / 2,
            "home": list(self.env.home[:2]),
            "place": self.a.place,
            "seed": self.a.seed,
            "buildings": [
                [[round(x, 1), round(y, 1)] for x, y in b["poly"].exterior.coords] for b in W.map["buildings"]
            ],
            "heights": [round(b["h"], 1) for b in W.map["buildings"]],
            "roads": [[[round(x, 1), round(y, 1)] for x, y in r["line"]] for r in W.map["roads"]],
            "water": [[[round(x, 1), round(y, 1)] for x, y in p.exterior.coords] for p in W.map["water"]],
            "trees": [[round(x, 1), round(y, 1)] for x, y, _ in W.trees[:3000]],
            "objects": [
                {
                    "id": o.oid,
                    "kind": o.kind,
                    "ru": KIND_RU[o.kind],
                    "colour": o.facts.get("colour"),
                    "colour_ru": COLOUR_RU.get(o.facts.get("colour"), ""),
                    "x": round(o.x, 1),
                    "y": round(o.y, 1),
                    "r": round(o.radius, 1),
                    "yaw": round(float(getattr(o, "yaw", 0.0)), 2),
                    "roof": bool(o.on_roof),
                    "about": about_ru(o),
                    "en": KIND_EN[o.kind],
                    "about_en": about_en(o),
                    "facts": {
                        k: (v if isinstance(v, (bool, int, float, str)) else str(v))
                        for k, v in o.facts.items()
                    },
                }
                for o in self.objs
            ],
        }

    def examples(self, page_lang="ru"):
        """Tasks for THIS map, written by the same generator as the test missions (some in unusual words)."""
        import missions as MS

        rng = np.random.default_rng(int(time.time() * 1000) % (1 << 30))
        W = self.env.W if hasattr(self.env.W, "footprints") else self.W0
        plan = [
            ("land", "ru", "train", "full", "обычное"),
            ("enter", "ru", "train", "medium", "влёт в ангар"),
            ("drop", "ru", "test", "full", "необычная формулировка"),
            ("inspect", "ru", "test", "medium", "необычная формулировка"),
            ("pickup+drop", "ru", "train", "medium", "два шага"),
            ("flyby", "en", "train", "medium", "по-английски"),
            ("land", "ru", "test", "sparse", "скупое описание"),
        ]
        if page_lang == "en":  # an English page gets English requests
            plan = [
                ("land", "en", "train", "full", "usual"),
                ("enter", "en", "train", "medium", "hangar entry"),
                ("drop", "en", "test", "full", "unusual wording"),
                ("inspect", "en", "test", "medium", "unusual wording"),
                ("pickup+drop", "en", "train", "medium", "two steps"),
                ("flyby", "en", "train", "medium", "fly-by"),
                ("land", "en", "test", "sparse", "sparse description"),
            ]
        out = []
        for act, lang, bank, detail, tag in plan:
            acts = act.split("+")
            try:
                steps = []
                for i, a in enumerate(acts):
                    kinds = set(MS.KIND_FOR[a])
                    cands = [
                        o
                        for o in self.objs
                        if o.kind in kinds and not (a == "enter" and not o.facts.get("door_open"))
                    ]
                    rng.shuffle(cands)
                    st = None
                    for o in cands[:8]:
                        used, rels, unique = MS.describe_facts(o, self.objs, W, rng, detail)
                        if not unique and detail != "sparse":
                            continue
                        par = {}
                        if a == "inspect":
                            par["distance_m"] = int(rng.choice([8, 10, 12]))
                        if a == "flyby":
                            par["altitude_m"] = int(rng.choice([20, 25, 30]))
                        st = {
                            "obj": o,
                            "action": a,
                            "params": par,
                            "used": used,
                            "rels": rels,
                            "unique": unique,
                        }
                        break
                    if st is None:
                        break
                    steps.append(st)
                if len(steps) != len(acts):
                    continue
                ret = bool(rng.random() < 0.5)
                text = (
                    MS.text_ru(steps, ret, bank, rng) if lang == "ru" else MS.text_en(steps, ret, bank, rng)
                )
                pts = []
                half = self.env.W.size / 2 - 5
                for st in steps:  # a rough point 20-40 m off, as a person would click
                    e, ang = rng.uniform(20, 40), rng.uniform(0, 2 * math.pi)
                    x = float(np.clip(st["obj"].x + e * math.cos(ang), -half, half))
                    y = float(np.clip(st["obj"].y + e * math.sin(ang), -half, half))
                    pts.append([round(x, 1), round(y, 1)])
                out.append(
                    {
                        "text": text,
                        "tag": tag,
                        "points": pts,
                        "targets": [st["obj"].oid for st in steps],
                    }
                )
            except Exception as e:  # one bad example must not break the page
                print("example failed:", act, repr(e), flush=True)
        return out

    def start(self, req):
        import nav as NV, safety as SF, mission_brain as MB

        en = req.get("lang") == "en"
        if self.mission_on:
            return {"error": "a flight is already on" if en else "миссия уже идёт"}
        self.lang = "en" if en else "ru"
        text = req.get("text", "").strip()
        pts = req.get("points") or []
        if not text or not pts:
            return {"error": "a request text and at least one rough point on the map are needed" if en else "нужны текст задания и хотя бы одна примерная точка на карте"}
        self.log(f"{'mission' if en else 'задание'}: {text}")
        t0 = time.time()
        parsed = (
            req.get("parsed")
            if isinstance(req.get("parsed"), dict)
            else (self.parse(text) if self.parse else None)
        )  # the operator may correct the parse
        if not parsed or not parsed.get("steps"):
            return {
                "error": "could not parse the request (no parser model or an empty answer)" if en else "не удалось разобрать задание (модель разбора не загружена или ответ пустой)",
                "parsed": parsed,
            }
        if req.get("return") in (True, False):
            parsed["return_home"] = req["return"]
        while len(pts) < len(parsed["steps"]):
            pts.append(pts[-1])
        self.log(f"{'parse' if en else 'разбор'} ({time.time() - t0:.1f} {'s' if en else 'с'}): {json.dumps(parsed, ensure_ascii=False)}")
        env = self.env
        with self.step_lock:
            env.battery0 = float(np.clip(float(req.get("battery") or 100), 15, 100))
            env.reset()
            conds = self.set_conditions(req)
            nav = NV.Navigator(env.W.map, env.home)
            self.safety = SF.Safety(env)
            self.brain = MB.Brain(
                env,
                env.W.map,
                parsed,
                [tuple(p) for p in pts],
                self.perc,
                safety=self.safety,
                ask=self.ask,
                nav=nav,
                log=self.log,
                robust=True,  # lidar outlier filter and the 120 m ceiling
                slow="telemetry",  # stage G3: the explicit energy-to-home rule beat the learned slow feelings
            )
            self.points = [tuple(p) for p in pts]
            self.trail = []
            self.stop_req = False
            self._o = None
            with self.lock:
                self.state.update(
                    phase="takeoff", parsed=parsed, result=None, question=None, conditions=conds, text=text, points=[list(p) for p in pts]
                )
            self.mission_on = True
        return {"ok": True, "parsed": parsed, "conditions": conds}

    def set_conditions(self, req):
        """Weather and failures for this flight. Everything is built anew, nothing is left from the last flight."""
        import mujoco

        env = self.env
        conds = []
        en = self.lang == "en"
        rng = np.random.default_rng()
        w = float(req.get("wind") or 0)
        a = rng.uniform(0, 2 * math.pi)
        env.wind_mean = np.array([w * math.cos(a), w * math.sin(a), 0.0])
        env.gust = GUST.get(req.get("gust"), 0.25) * w
        if w > 0:
            conds.append((f"wind {w:.0f} m/s" + (", strong gusts" if req.get("gust") == "strong" else "")) if en else (f"ветер {w:.0f} м/с" + (", сильные порывы" if req.get("gust") == "strong" else "")))
        env.faults = []
        env.gps_faults = []
        env.link_faults = []
        gps = req.get("gps") or ("off" if req.get("gps_off") not in (None, "") else "ok")
        if gps in GPS_KINDS:
            t = float(req.get("gps_t") if req.get("gps_t") not in (None, "") else req.get("gps_off") or 0)
            env.gps_faults.append((t, 1e9, gps))
            conds.append(f"{GPS_KINDS_EN[gps]} after {t:.0f} s" if en else f"{GPS_KINDS[gps]} через {t:.0f} с")
        if req.get("lidar") in LIDAR_KINDS:
            kind, sev, name = LIDAR_KINDS[req["lidar"]]
            t = float(req.get("lidar_t") or 0)
            env.faults.append(
                {
                    "sensor": "lidar",
                    "kind": kind,
                    "t0": t,
                    "t1": 1e9,
                    "sev": sev,
                    "center": rng.uniform(-math.pi, math.pi),
                }
            )
            conds.append(f"{LIDAR_EN[req['lidar']]}, after {t:.0f} s" if en else f"{name}, через {t:.0f} с")
        if req.get("compass"):
            env.faults.append(
                {"sensor": "compass", "kind": "bias", "t0": 0.0, "t1": 1e9, "sev": math.radians(20)}
            )
            conds.append("compass off by 20°" if en else "компас врёт на 20°")
        if req.get("link_t") not in (None, ""):
            t = float(req["link_t"])
            env.link_faults.append((t, 1e9))
            conds.append(f"link to the operator lost after {t:.0f} s" if en else f"связь с оператором пропадёт через {t:.0f} с")
        if env.battery0 < 100:
            conds.append(f"{env.battery0:.0f} % charge at take-off" if en else f"заряд при взлёте {env.battery0:.0f} %")
        if req.get("cold"):
            env.world_events = [{"kind": "drain", "mult": 1.4, "t_event": 0.0, "ramp_s": 20.0}]
            conds.append("cold battery: capacity falls" if en else "холодная батарея: ёмкость падает")
        self.fog = FOG.get(req.get("fog"), 0.0)
        m = env.model
        ext = float(m.stat.extent)
        if self.fog > 0:
            m.vis.map.fogstart = 0.15 * self.fog / ext
            m.vis.map.fogend = self.fog / ext
            conds.append(f"fog, visibility {self.fog:.0f} m" if en else f"туман, видно на {self.fog:.0f} м")
        env.visibility = self.fog
        if env.grend is not None:
            env.grend.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = self.fog > 0
        if req.get("attacks"):
            import encounters as EN, eval_missions as EM

            env.enc = EN.EncounterGenerator(np.random.default_rng(), **EM.ATTACKS)
            conds.append("bird attacks" if en else "атаки птиц")
        else:
            env.enc = None
        for c in conds:
            self.log(("condition: " if en else "условие: ") + c)
        return conds

    def preview(self, req):
        """Parse only: show the operator how the drone understood the text before it flies."""
        en = req.get("lang") == "en"
        text = req.get("text", "").strip()
        if not text:
            return {"error": "empty request" if en else "пустое задание"}
        if not self.parse:
            return {"error": "the parser model is not loaded" if en else "модель разбора не загружена"}
        parsed = self.parse(text)
        if not parsed or not parsed.get("steps"):
            return {"error": "the model could not parse the request, please rephrase it" if en else "модель не смогла разобрать задание, переформулируйте его", "parsed": parsed}
        return {"parsed": parsed, "summary": summary_en(parsed) if en else summary_ru(parsed)}

    def ask(self, step, cands):
        """Show the operator the top candidates."""
        env = self.env
        imgs = []
        for p, h, parts in cands:
            yaw = math.degrees(
                math.atan2(h["y"] - self.brain.pos[1], h["x"] - self.brain.pos[0]) - env.obs_yaw()
            )
            d = math.hypot(h["x"] - self.brain.pos[0], h["y"] - self.brain.pos[1])
            env.set_gimbal(yaw, -math.degrees(math.atan2(self.brain.pos[2] - h["z"], max(d, 1.0))))
            rgb, _, _ = env.gimbal_frame(depth=False)
            H, Wd = rgb.shape[:2]
            imgs.append(
                {
                    "img": jpeg(
                        rgb,
                        [
                            (
                                [Wd / 2 - 40, H / 2 - 40, Wd / 2 + 40, H / 2 + 40],
                                f"{len(imgs) + 1}",
                                (255, 60, 60),
                            )
                        ],
                        w=320,
                    ),
                    "p": round(p, 2),
                    "x": round(h["x"], 1),
                    "y": round(h["y"], 1),
                }
            )
        self.answer_ev.clear()
        with self.lock:
            self.state["question"] = {
                "text": "I am not sure which object you meant. Pick a candidate or 'none of these'." if self.lang == "en" else "Я не уверен, какой объект имелся в виду. Выберите кандидата или «ни один».",
                "cands": imgs,
            }
        self.log("question to the operator: which candidate?" if self.lang == "en" else "вопрос оператору: какой из кандидатов?")
        got = self.answer_ev.wait(180.0)
        with self.lock:
            self.state["question"] = None
        return self.answer if got else None

    def describe(self, oid):
        o = next((o for o in self.objs if o.oid == oid), None)
        if o is None:
            return oid
        if self.lang == "en":
            col = o.facts.get("colour")
            return KIND_EN[o.kind] + (f" (colour {col})" if col else "")
        col = COLOUR_RU.get(o.facts.get("colour"))
        return f"{KIND_RU[o.kind]}" + (f" (цвет {col})" if col else "")

    def outcome(self):
        """What the WORLD says happened (the demo has no answer key: the operator judges it against the text)."""
        env = self.env
        out = []
        en = self.lang == "en"
        p = env.data.qpos[:3]
        first = {}  # one line per object
        for t, g, sp, x, y, z in env.touchdowns:
            oid = (env.model.geom(g).name or "").split("_")[0]
            if g == env.ground_id:  # set down on the ground: counts only inside a hangar (on its floor)
                o = next(
                    (
                        o
                        for o in self.objs
                        if o.kind == "hangar"
                        and math.hypot(x - o.x, y - o.y) < o.facts["size_m"] / 2 - 0.3
                        and z < o.facts["height_m"]
                    ),
                    None,
                )
                if o is None:
                    continue
                oid = o.oid
            if oid not in first or (t - first[oid][0] < 2.0 and sp > first[oid][1]):
                first[oid] = (first.get(oid, (t,))[0], max(sp, first.get(oid, (0, 0))[1]))
        for oid, (t, sp) in first.items():
            inside = any(o.oid == oid and o.kind == "hangar" for o in self.objs)
            out.append(
                f'{"landed inside" if inside else "landed on"}: {self.describe(oid)} at {t:.0f} s, touchdown speed {sp:.1f} m/s'
                if en and oid in {o.oid for o in self.objs}
                else f'{"посадка внутри" if inside else "посадка на"}: {self.describe(oid)} на {t:.0f} с, скорость касания {sp:.1f} м/с'.replace(".", ",")
                if oid in {o.oid for o in self.objs}
                else f"касание {oid} в {t:.0f} с"
            )
        if env.hook["carry"]:
            out.append(f'on the hook: the parcel from {self.describe(env.hook["carry"])}' if en else f'на крюке посылка, взятая из: {self.describe(env.hook["carry"])}')
        c = env.cargo
        if c["state"] == "landed":
            dz = min(
                (o for o in self.objs if o.kind == "dropzone"),
                key=lambda o: math.hypot(c["p"][0] - o.x, c["p"][1] - o.y),
                default=None,
            )
            out.append(
                (f"the parcel fell at ({c['p'][0]:.0f}, {c['p'][1]:.0f})" if en else f"груз упал в ({c['p'][0]:.0f}, {c['p'][1]:.0f})")
                + (
                    (f', {math.hypot(c["p"][0] - dz.x, c["p"][1] - dz.y):.1f} m from the centre of {self.describe(dz.oid)}' if en else f', {math.hypot(c["p"][0] - dz.x, c["p"][1] - dz.y):.1f} м от центра цели: {self.describe(dz.oid)}')
                    if dz
                    else ""
                )
            )
        n = len([h for h in env.hit_events if h["kind"] in ("static", "mover")])
        a = len([h for h in env.hit_events if h["kind"] == "encounter"])
        out.append(
            f"collisions: {n}, bird-attack hits: {a}; {math.hypot(p[0] - env.home[0], p[1] - env.home[1]):.0f} m from home, altitude {p[2]:.0f} m, battery {env.battery:.0f} %"
            if en
            else f"столкновений: {n}, попаданий птиц-атак: {a}; до дома {math.hypot(p[0] - env.home[0], p[1] - env.home[1]):.0f} м, высота {p[2]:.0f} м, батарея {env.battery:.0f} %"
        )
        return out

    # ---------------- the 3D view
    def set_view(self, req):
        v = self.view
        if req.get("mode") in ("chase", "top", "free"):
            v["mode"] = req["mode"]
        for k, lo, hi in (("azimuth", -1e9, 1e9), ("elevation", -89.0, -3.0), ("distance", 6.0, 250.0)):
            if req.get(k) is not None:
                v[k] = float(np.clip(float(req[k]), lo, hi))
        return {"ok": True, "view": v}

    def render3d(self):
        """One frame of the 3D view. Uses the gimbal camera's renderer (one OpenGL context in this thread)."""
        import mujoco
        from PIL import Image

        env = self.env
        if env.grend is None:
            env.grend = mujoco.Renderer(env.model, env.gh, env.gw, max_geom=30000)
        if self.cam3d is None:
            self.cam3d = mujoco.MjvCamera()
            self.cam3d.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam3d.trackbodyid = env.bid
            self.cam3d.azimuth = 135.0
        cam, v = self.cam3d, self.view
        if v["mode"] == "chase":  # behind the drone, turning smoothly with its heading
            yaw = math.degrees(env.obs_yaw())
            cam.azimuth += 0.15 * ((yaw - cam.azimuth + 180.0) % 360.0 - 180.0)
            cam.elevation = -25.0
            cam.distance = v["distance"]
            v["azimuth"] = cam.azimuth
        elif v["mode"] == "top":
            cam.azimuth, cam.elevation, cam.distance = 90.0, -89.0, max(60.0, 2.5 * v["distance"])
        else:
            cam.azimuth, cam.elevation, cam.distance = v["azimuth"], v["elevation"], v["distance"]
        R = env.grend
        R.update_scene(env.data, camera=cam, scene_option=env.vopt)
        R.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = self.fog > 0
        self._marks(R.scene)
        img = R.render()
        with self.lock:
            self.raw3d = img
            self.state["frame_seq"] += 1

    def _marks(self, scene):
        """Things only the 3D view shows: the rough points, home and the drone's path. Never seen by the drone."""
        import mujoco

        def add(kind, size, pos, rgba):
            if scene.ngeom >= scene.maxgeom:
                return
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                kind,
                np.array(size, float),
                np.array(pos, float),
                np.eye(3).flatten(),
                np.array(rgba, np.float32),
            )
            scene.ngeom += 1

        cyl, sph = mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE
        for x, y in self.points:  # a see-through teal pillar over each rough point
            add(cyl, [1.2, 20.0, 0], [x, y, 20.0], [0.04, 0.45, 0.52, 0.35])
        hx, hy = self.env.home[:2]
        add(cyl, [2.5, 0.05, 0], [hx, hy, 0.06], [1.0, 1.0, 1.0, 0.9])
        for p in self.trail[-600:]:
            add(sph, [0.22, 0, 0], p, [0.18, 0.44, 0.93, 0.9])

    def loop(self):
        while self.running:
            t_wall = time.time()
            with self.step_lock:
                dt = self.tick()
            busy = time.time() - t_wall
            if self.viewer is not None and self.viewer.is_running():
                with self.viewer.lock():
                    self.viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - t_wall)))
            if self.mission_on:  # the speed the page can promise: simulated seconds per wall second
                real = self.env.agent_dt / max(time.time() - t_wall, 1e-3)
                with self.lock:
                    self.state["tick_ms"] = round(0.95 * self.state.get("tick_ms", 1000 * busy) + 0.05 * 1000 * busy, 1)
                    self.state["speed_real"] = round(0.95 * self.state.get("speed_real", real) + 0.05 * real, 2)

    def tick(self):
        """One step of the world (and now and then a frame of the 3D view). Returns the wall time to wait."""
        if self.mission_on and self.brain is not None:
            o = self.env.observe(images=False) if not hasattr(self, "_o") or self._o is None else self._o
            falling = (
                self.brain.done and self.env.cargo["state"] == "falling"
            )  # mission over, the cargo still in the air: let it land
            cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0} if falling else self.brain.act(o)
            self._o = self.env.step(cmd)
            p = self.env.data.qpos[:3]
            v = self.env.data.qvel[:3]
            if not self.trail or math.dist(self.trail[-1], p) > 1.5:
                self.trail.append([float(p[0]), float(p[1]), float(p[2])])
            with self.lock:
                self.state.update(
                    phase=self.brain.state,
                    t=round(self.env.t, 1),
                    true=[round(float(x), 2) for x in p],
                    est=[round(float(x), 2) for x in self.brain.pos],
                    alt=round(float(p[2]), 1),
                    speed=round(float(np.linalg.norm(v)), 1),
                    yaw=round(self.env.obs_yaw(), 2),
                    feel=getattr(self.safety, "feel", {}),
                    gps_trust=round(self.brain.nav.gps_trust, 2) if self.brain.nav else None,
                    nav_spread=round(self.brain.nav.spread, 2) if self.brain.nav else None,
                    battery=round(self.env.battery, 1),
                    fires=self.safety.fires,
                    gps=self.env.gps_mode(),
                    link=self.env.link_up(),
                )
            stopped = getattr(self, "stop_req", False)
            if (self.brain.done and self.env.cargo["state"] != "falling") or self.env.t > 900 or stopped:
                self.stop_req = False
                r = self.env.result()
                r["outcome"] = ([("stopped by the operator" if self.lang == "en" else "полёт остановлен оператором")] if stopped else []) + self.outcome()
                r["stopped"] = stopped
                self.log(("mission over: " if self.lang == "en" else "миссия завершена: ") + "; ".join(r["outcome"]))
                self.mission_on = False
                self._o = None
                with self.lock:
                    self.state.update(phase="done", result=r)
            dt = self.env.agent_dt / max(self.speed, 0.1)
        else:
            self.env._move(0.0)
            dt = 0.05
        looking = time.time() - self.t_req3d < 2.0  # render only while the page shows the 3D view
        if looking and time.time() - self.t_frame > (0.1 if self.mission_on else 0.4):
            self.t_frame = time.time()
            if self.state["view_error"] is None:
                try:
                    t_r = time.time()
                    self.render3d()
                    with self.lock:
                        self.state["render_ms"] = round(1000 * (time.time() - t_r), 1)
                except Exception as e:  # the app keeps working without the 3D view
                    with self.lock:
                        self.state["view_error"] = repr(e)
                    self.log(
                        f"3D-вид не запустился: {e!r}. Запустите run_app.bat --viewer для отдельного окна."
                    )
        return dt


def serve(sim, port):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header(
                "Content-Type", ctype + ("; charset=utf-8" if "json" in ctype or "html" in ctype else "")
            )
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self.send(200, (DEMO / "index.html").read_bytes(), "text/html")
            if path == "/api/map":
                return self.send(200, sim.map_json())
            if path == "/api/state":
                q = dict(x.split("=", 1) for x in self.path.partition("?")[2].split("&") if "=" in x)
                return self.send(200, sim.snapshot(int(q.get("log", 0) or 0)))
            if path == "/api/frame":
                f = sim.frame_jpeg()
                return self.send(200, f, "image/jpeg") if f else self.send(204, b"")  # not ready yet
            if path == "/api/cam":
                f = sim.cam_jpeg()
                return self.send(200, f, "image/jpeg") if f else self.send(204, b"")  # the camera has not looked yet
            if path == "/api/examples":
                return self.send(200, sim.examples("en" if "lang=en" in self.path else "ru"))
            self.send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/mission":
                return self.send(200, sim.start(req))
            if self.path == "/api/parse":
                return self.send(200, sim.preview(req))
            if self.path == "/api/answer":
                sim.answer = int(req.get("choice", -1))
                sim.answer_ev.set()
                return self.send(200, {"ok": True})
            if self.path == "/api/speed":
                sim.speed = float(req.get("speed", 1))
                return self.send(200, {"ok": True})
            if self.path == "/api/view":
                return self.send(200, sim.set_view(req))
            if self.path == "/api/stop":
                return self.send(200, sim.stop())
            self.send(404, {"error": "not found"})

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True  # a quick restart must not fail on a port still in TIME_WAIT
        daemon_threads = True

    srv = Server(("127.0.0.1", port), H)
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", default="kazan_novosav")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--truth-perception", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--viewer", action="store_true", help="also open the MuJoCo 3D window")
    ap.add_argument(
        "--no-viewer", action="store_true", help="kept for old launch files; the window is off by default"
    )
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if a.seed is None:
        a.seed = int(np.random.default_rng().integers(1000, 100000))  # a new layout of objects at every start
    print(f"место {a.place}, расстановка {a.seed}", flush=True)
    sim = Sim(a)
    srv = serve(sim, a.port)
    print(f"Откройте в браузере: http://localhost:{a.port}", flush=True)
    if not a.no_browser:
        import webbrowser

        webbrowser.open(f"http://localhost:{a.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sim.running = False


if __name__ == "__main__":
    main()
