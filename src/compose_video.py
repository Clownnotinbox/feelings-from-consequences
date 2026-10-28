"""Supplementary videos, step 2.
usage: compose_video.py ASSET_DIR OUT.mp4 [ru|en] [FPS=25] [--frames N1,N2,...]   (--frames: write JPEGs of those frames only)
"""

import json, math, subprocess, sys, textwrap
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = Path(__file__).resolve().parent / "fonts"
SYS_FONTS = [Path("/usr/share/fonts/opentype/inter"), Path("/usr/share/fonts/truetype/ibm-plex")]
BG, PANEL, LINE, INK, INK2, INK3 = (
    (14, 15, 17),
    (22, 24, 27),
    (44, 47, 52),
    (236, 238, 241),
    (160, 166, 174),
    (104, 110, 118),
)
BLUE, ORANGE, GREEN, RED, CYAN, YELLOW = (
    (76, 141, 246),
    (242, 140, 56),
    (61, 190, 139),
    (229, 72, 77),
    (86, 204, 242),
    (240, 196, 64),
)
W, H, M, G = 1920, 1080, 24, 16
FEELS = [
    ("danger_05", ORANGE),
    ("danger_1", ORANGE),
    ("danger_2", ORANGE),
    ("gps_bad", BLUE),
    ("lost", BLUE),
    ("low_energy", BLUE),
    ("strain", BLUE),
]
STATES = ["takeoff", "transit", "search", "closer", "verify", "approach", "ACTION", "return", "landed"]
ACTION_STATES = {
    "descend",
    "final",
    "drop_hold",
    "hover_pick",
    "hook",
    "find_door",
    "enter_in",
    "exit_out",
    "orbit",
    "pass",
}
T = {
    "ru": {
        "title": "Дрон с ядром чувств · запись миссии",
        "request": "ЗАПРОС ОПЕРАТОРА",
        "view3d": "3D-вид (камера за дроном)",
        "gimbal": "Камера подвеса · распознавание",
        "map": "Карта (OpenStreetMap) · путь · лидар",
        "state": "Решения «мозга»",
        "feel": "Чувства ядра (вероятность последствия)",
        "nolook": "камера ещё не смотрела на цель",
        "danger_05": "опасность за 0,5 с",
        "danger_1": "опасность за 1 с",
        "danger_2": "опасность за 2 с",
        "gps_bad": "недоверие к GPS",
        "lost": "«я потерялся»",
        "low_energy": "«мало сил»",
        "strain": "напряжение",
        "fires": "уклонение ядра",
        "hit": "касание",
        "full": "ЯДРО ЧУВСТВ ВКЛ · один, без оператора",
        "operator": "ЯДРО ЧУВСТВ ВКЛ · оператор отвечает на вопросы",
        "reflex": "БЕЗ ЯДРА · обычный рефлекс",
        "alt": "высота",
        "speed": "скорость",
        "bat": "батарея",
        "gps": "GPS",
        "gps_ok": "есть",
        "gps_off": "НЕТ",
        "wind": "ветер",
        "nav": "ошибка места",
        "true": "истинный путь",
        "est": "оценка дрона",
        "rough": "примерная точка",
        "target": "цель",
        "home": "дом",
        "lidar": "лидар",
        "states": ["взлёт", "перелёт", "поиск", "ближе", "проверка", "подход", "ДЕЙСТВИЕ", "возврат", "сел"],
        "result": "Итог (решает мир, а не дрон)",
        "step_ok": "выполнено",
        "step_no": "не выполнено",
        "contacts": "касаний неподвижного",
        "speedup": "ускорено ×",
    },
    "en": {
        "title": "Feeling-core drone · mission replay",
        "request": "OPERATOR REQUEST",
        "view3d": "3D view (chase camera)",
        "gimbal": "Gimbal camera · detector",
        "map": "Map (OpenStreetMap) · track · lidar",
        "state": "Mission brain",
        "feel": "Core feelings (probability of the consequence)",
        "nolook": "the camera has not looked at the target yet",
        "danger_05": "danger in 0.5 s",
        "danger_1": "danger in 1 s",
        "danger_2": "danger in 2 s",
        "gps_bad": "GPS distrust",
        "lost": '"I am lost"',
        "low_energy": '"low energy"',
        "strain": "strain",
        "fires": "core avoids",
        "hit": "contact",
        "full": "FEELING CORE ON · alone, no operator",
        "operator": "FEELING CORE ON · operator answers questions",
        "reflex": "NO CORE · classic reflex",
        "alt": "altitude",
        "speed": "speed",
        "bat": "battery",
        "gps": "GPS",
        "gps_ok": "ok",
        "gps_off": "LOST",
        "wind": "wind",
        "nav": "position error",
        "true": "true track",
        "est": "drone's estimate",
        "rough": "rough point",
        "target": "target",
        "home": "home",
        "lidar": "lidar",
        "states": [
            "take-off",
            "transit",
            "search",
            "closer",
            "verify",
            "approach",
            "ACTION",
            "return",
            "landed",
        ],
        "result": "Outcome (judged by the world, not by the drone)",
        "step_ok": "done",
        "step_no": "not done",
        "contacts": "static contacts",
        "speedup": "speed ×",
    },
}


def fnt(name, size):
    for d in [FONTS] + SYS_FONTS:
        p = d / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


class Comp:
    def __init__(self, d, lang="ru"):
        self.d = Path(d)
        self.L = T[lang]
        self.meta = json.load(open(self.d / "meta.json", encoding="utf-8"))
        self.res = json.load(open(self.d / "result.json", encoding="utf-8"))
        self.S = [json.loads(l) for l in open(self.d / "steps.jsonl")]
        self.frames = [s for s in self.S if "frame" in s]
        self.t_end = self.S[-1]["t"] if self.S else 1.0
        self.f = {
            k: fnt(n, s)
            for k, (n, s) in {
                "h1": ("Inter-SemiBold.otf", 22),
                "h2": ("Inter-Medium.otf", 15),
                "b": ("Inter-Regular.otf", 17),
                "req": ("Inter-Medium.otf", 19),
                "s": ("Inter-Regular.otf", 14),
                "xs": ("Inter-Regular.otf", 12),
                "m": ("IBMPlexMono-Regular.ttf", 15),
                "mb": ("IBMPlexMono-Medium.ttf", 17),
                "big": ("Inter-SemiBold.otf", 30),
            }.items()
        }
        # layout boxes (x0, y0, x1, y1)
        top = 96
        self.box_top = (M, M, W - M, top - 12)
        left_w = 1248
        self.box_3d = (M, top, M + left_w, top + 702)
        self.box_tl = (M, top + 702 + G, M + left_w, H - M)
        rx = M + left_w + G
        self.box_cam = (rx, top, W - M, top + 456)
        self.box_map = (rx, top + 456 + G, W - M, H - M - 200 - G)
        self.box_st = (rx, H - M - 200, W - M, H - M)
        self.map_setup()
        self.looks = self.res.get("looks", [])

    # ---------------- helpers
    def panel(self, dr, box, title):
        dr.rectangle(box, fill=PANEL, outline=LINE)
        dr.text((box[0] + 14, box[1] + 10), title.upper() if title else "", font=self.f["xs"], fill=INK3)

    def map_setup(self):
        mt = self.meta
        pts = (
            [mt["home"]]
            + [list(p) for p in mt["rough"]]
            + [[o["x"], o["y"]] for o in mt["objects"] if o["oid"] in mt["targets"]]
            + [s["true"][:2] for s in self.S[::10]]
        )
        P = np.array(pts, float)
        lo, hi = P.min(0) - 40, P.max(0) + 40
        bw, bh = self.box_map[2] - self.box_map[0] - 24, self.box_map[3] - self.box_map[1] - 44
        sc = min(bw / (hi[0] - lo[0]), bh / (hi[1] - lo[1]))
        c = (lo + hi) / 2
        ox, oy = (self.box_map[0] + self.box_map[2]) / 2, (self.box_map[1] + 32 + self.box_map[3]) / 2
        self.mp = lambda x, y: (ox + (x - c[0]) * sc, oy - (y - c[1]) * sc)
        self.msc = sc
        self.mlo, self.mhi = lo, hi

    def look_at(self, t):
        best = None
        for lk in self.looks:
            if lk["t"] <= t:
                best = lk
        return best

    # ---------------- panels
    def draw_top(self, dr, s):
        mt = self.meta
        x0, y0, x1, y1 = self.box_top
        dr.text((x0, y0 + 2), self.L["title"], font=self.f["h1"], fill=INK)
        dr.text((x0, y0 + 34), f"{mt['place']}  ·  t = {s['t']:6.1f} s", font=self.f["m"], fill=INK2)
        badge = self.L[mt["variant"]]
        col = ORANGE if mt["variant"] == "reflex" else BLUE
        bw = dr.textlength(badge, font=self.f["h2"]) + 28
        dr.rounded_rectangle((x1 - bw, y0 + 4, x1, y0 + 34), radius=4, outline=col, width=2)
        dr.text((x1 - bw + 14, y0 + 10), badge, font=self.f["h2"], fill=col)
        qx = x0 + 560
        dr.text((qx, y0 + 2), self.L["request"], font=self.f["xs"], fill=INK3)
        lines = textwrap.wrap("«" + mt["text"] + "»", 88)[:2]
        for k, ln in enumerate(lines):
            dr.text((qx, y0 + 18 + 22 * k), ln, font=self.f["req"], fill=INK)

    def draw_3d(self, im, dr, s):
        self.panel(dr, self.box_3d, "")
        x0, y0, x1, y1 = self.box_3d
        fr = Image.open(self.d / "chase" / f"{s['frame']:05d}.jpg")
        w, h = fr.size
        ch = int(w * 9 / 16)
        fr = fr.crop((0, (h - ch) // 2, w, (h - ch) // 2 + ch)).resize((x1 - x0 - 2, y1 - y0 - 2))
        im.paste(fr, (x0 + 1, y0 + 1))
        dr.rectangle((x0 + 1, y0 + 1, x0 + 330, y0 + 30), fill=(0, 0, 0))
        dr.text((x0 + 12, y0 + 7), self.L["view3d"].upper(), font=self.f["xs"], fill=INK2)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        for k in range(4):
            a0 = 90 * k + 20
            dr.arc(
                (cx - 26, cy - 26, cx + 26, cy + 26), a0, a0 + 50, fill=YELLOW, width=2
            )  # the drone (the tracking camera keeps it at the centre)
        if s["static_hits"]:
            dr.text(
                (x1 - 260, y0 + 8), f"{self.L['contacts']}: {s['static_hits']}", font=self.f["h2"], fill=RED
            )

    def draw_cam(self, im, dr, s):
        self.panel(dr, self.box_cam, self.L["gimbal"])
        x0, y0, x1, y1 = self.box_cam
        lk = self.look_at(s["t"])
        if lk is None:
            dr.text((x0 + 14, (y0 + y1) // 2), self.L["nolook"], font=self.f["s"], fill=INK3)
            return
        g = Image.open(self.d / "gimbal" / f"{lk['n']:05d}.jpg")
        sc = (x1 - x0 - 2) / g.width
        g = g.resize((x1 - x0 - 2, int(g.height * sc)))
        top = y1 - 1 - g.height
        im.paste(g, (x0 + 1, top))
        gd = ImageDraw.Draw(im)
        for bx0, by0, bx1, by1, cls, conf in lk["boxes"]:
            col = YELLOW if cls != "bird" else RED
            b = (x0 + 1 + bx0 * sc, top + by0 * sc, x0 + 1 + bx1 * sc, top + by1 * sc)
            gd.rectangle(b, outline=col, width=2)
            lab = f"{cls} {conf:.2f}"
            tw = gd.textlength(lab, font=self.f["xs"])
            gd.rectangle((b[0], b[1] - 16, b[0] + tw + 8, b[1]), fill=col)
            gd.text((b[0] + 4, b[1] - 15), lab, font=self.f["xs"], fill=(0, 0, 0))
        dr.text((x1 - 90, y0 + 10), f"t {lk['t']:.0f} s", font=self.f["xs"], fill=INK3)

    def draw_map(self, im, dr, s, upto):
        mt = self.meta
        X0, Y0, X1, Y1 = self.box_map
        sub = Image.new("RGB", (X1 - X0, Y1 - Y0), PANEL)
        md = ImageDraw.Draw(sub)
        mp = lambda x, y: (lambda q: (q[0] - X0, q[1] - Y0))(self.mp(x, y))
        x0, y0, x1, y1 = 0, 0, X1 - X0, Y1 - Y0
        for poly in mt["buildings"]:
            q = [mp(x, y) for x, y in poly]
            if all(not (-50 < a < x1 + 50 and -50 < b < y1 + 50) for a, b in q):
                continue
            md.polygon(q, fill=(34, 37, 41), outline=(58, 63, 69))
        for rd in mt["roads"]:
            if len(rd) > 1:
                md.line([mp(x, y) for x, y in rd], fill=(48, 52, 57), width=2)
        tg = set(mt["targets"])
        for o in mt["objects"]:
            a, b = mp(o["x"], o["y"])
            r = max(3, o["r"] * self.msc)
            md.ellipse(
                (a - r, b - r, a + r, b + r),
                outline=(GREEN if o["oid"] in tg else INK3),
                width=2 if o["oid"] in tg else 1,
            )
        for k, (rx, ry) in enumerate(mt["rough"]):
            a, b = mp(rx, ry)
            md.line((a - 7, b - 7, a + 7, b + 7), fill=YELLOW, width=2)
            md.line((a - 7, b + 7, a + 7, b - 7), fill=YELLOW, width=2)
        hx, hy = mp(*mt["home"])
        md.rectangle((hx - 6, hy - 6, hx + 6, hy + 6), outline=INK, width=2)
        tr = [mp(*q["true"][:2]) for q in self.S[: upto + 1 : 3]]
        es = [mp(*q["est"][:2]) for q in self.S[: upto + 1 : 3]]
        if len(tr) > 1:
            md.line(tr, fill=INK, width=2)
        for a in range(0, len(es) - 1, 2):
            md.line([es[a], es[a + 1]], fill=BLUE, width=2)
        if "lidar" in s:
            px, py = s["true"][0], s["true"][1]
            for lx, ly, lz in s["lidar"]:
                a, b = mp(px + lx, py + ly)
                if 0 < a < x1 and 28 < b < y1:
                    md.rectangle((a - 1, b - 1, a + 1, b + 1), fill=CYAN)
        a, b = mp(*s["true"][:2])
        md.ellipse((a - 5, b - 5, a + 5, b + 5), fill=INK)
        md.rectangle((0, y1 - 30, x1, y1), fill=PANEL)
        ly = y1 - 22
        lx = x0 + 14
        for col, key in ((INK, "true"), (BLUE, "est"), (GREEN, "target"), (YELLOW, "rough"), (CYAN, "lidar")):
            md.rectangle((lx, ly + 5, lx + 12, ly + 9), fill=col)
            md.text((lx + 16, ly), self.L[key], font=self.f["xs"], fill=INK2)
            lx += 24 + md.textlength(self.L[key], font=self.f["xs"]) + 10
        md.rectangle((0, 0, x1, 30), fill=PANEL)
        im.paste(sub, (X0, Y0))
        dr.rectangle(self.box_map, outline=LINE)
        dr.text((X0 + 14, Y0 + 10), self.L["map"].upper(), font=self.f["xs"], fill=INK3)

    def draw_state(self, dr, s):
        self.panel(dr, self.box_st, self.L["state"])
        x0, y0, x1, y1 = self.box_st
        st = "ACTION" if s["state"] in ACTION_STATES else s["state"]
        cx, cy = x0 + 14, y0 + 32
        for k, name in enumerate(STATES):
            lab = self.L["states"][k]
            tw = dr.textlength(lab, font=self.f["xs"]) + 14
            on = name == st
            if cx + tw > x1 - 10:
                cx, cy = x0 + 14, cy + 26
            dr.rounded_rectangle(
                (cx, cy, cx + tw, cy + 20),
                radius=3,
                fill=(BLUE if on else PANEL),
                outline=(BLUE if on else LINE),
            )
            dr.text((cx + 7, cy + 3), lab, font=self.f["xs"], fill=(INK if on else INK2))
            cx += tw + 6
        nerr = math.hypot(s["true"][0] - s["est"][0], s["true"][1] - s["est"][1])
        gps = self.L["gps_ok"] if s["gps"] is not None else self.L["gps_off"]
        rows = [
            (self.L["alt"], f"{s['true'][2]:5.1f} m"),
            (self.L["speed"], f"{s['speed']:4.1f} m/s"),
            (self.L["bat"], f"{s['battery']:4.0f} %"),
            (self.L["gps"], f"{gps} · {s['gps_trust']:.2f}"),
            (self.L["wind"], f"{math.hypot(*s['wind']):4.1f} m/s"),
            (self.L["nav"], f"{nerr:4.1f} m"),
        ]
        yy = cy + 34
        for k, (a, b) in enumerate(rows):
            xx = x0 + 14 + (k % 2) * 280
            y = yy + (k // 2) * 24
            dr.text((xx, y), a, font=self.f["s"], fill=INK2)
            dr.text(
                (xx + 128, y - 1),
                b,
                font=self.f["m"],
                fill=(RED if (a == self.L["gps"] and s["gps"] is None) else INK),
            )
        notes = self.res.get("notes") or []
        n = s.get("notes", 0)
        said = [
            x.split("] ", 1)[-1] for x in notes[:n] if not x.split("] ", 1)[-1].startswith("-> ")
        ]  # decisions, not state changes
        if said:
            dr.text((x0 + 14, y1 - 30), ("› " + said[-1])[:66], font=self.f["s"], fill=YELLOW)

    def draw_timeline(self, dr, s):
        self.panel(dr, self.box_tl, self.L["feel"])
        x0, y0, x1, y1 = self.box_tl
        lx, rx = x0 + 170, x1 - 18
        top = y0 + 34
        rh = (y1 - top - 30) / len(FEELS)
        tx = lambda t: lx + (rx - lx) * t / max(self.t_end, 1.0)
        ts = np.array([q["t"] for q in self.S])
        for k, (key, col) in enumerate(FEELS):
            y = top + k * rh
            dr.text((x0 + 14, y + rh / 2 - 9), self.L[key], font=self.f["s"], fill=INK2)
            dr.line((lx, y + rh - 2, rx, y + rh - 2), fill=LINE)
            v = np.array([q["feel"].get(key, -1) for q in self.S])
            if (v < 0).all():
                dr.text((lx + 6, y + 4), "—", font=self.f["s"], fill=INK3)
                continue
            pts = [(tx(t), y + rh - 2 - max(0.0, val) * (rh - 6)) for t, val in zip(ts[::2], v[::2])]
            if len(pts) > 1:
                dr.line(pts, fill=tuple(int(c * 0.55 + 22 * 0.45) for c in col), width=1)
            cur = [p for p, t in zip(pts, ts[::2]) if t <= s["t"]]
            if len(cur) > 1:
                dr.line(cur, fill=col, width=2)
        fires = [q["t"] for a, q in zip(self.S, self.S[1:]) if q["fires"] > a["fires"]]
        hits = [q["t"] for a, q in zip(self.S, self.S[1:]) if q["static_hits"] > a["static_hits"]]
        yb = y1 - 24
        for t in fires:
            dr.line((tx(t), yb, tx(t), yb + 8), fill=BLUE, width=1)
        for t in hits:
            dr.ellipse((tx(t) - 5, yb + 2, tx(t) + 5, yb + 12), fill=RED)
        dr.text((x0 + 14, yb - 2), f"| {self.L['fires']}   ● {self.L['hit']}", font=self.f["xs"], fill=INK3)
        cx = tx(s["t"])
        dr.line((cx, top - 4, cx, y1 - 8), fill=INK, width=1)

    def frame(self, idx):
        s = self.frames[idx]
        upto = next(k for k, q in enumerate(self.S) if q is s)
        im = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(im)
        self.draw_top(dr, s)
        self.draw_3d(im, dr, s)
        self.draw_cam(im, dr, s)
        self.draw_map(im, dr, s, upto)
        self.draw_state(dr, s)
        self.draw_timeline(dr, s)
        return im

    def outcome(self, im):
        dr = ImageDraw.Draw(im)
        x0, y0, x1, y1 = self.box_3d
        bx = (x0 + 220, y0 + 220, x1 - 220, y0 + 220 + 200)
        dr.rectangle(bx, fill=(10, 11, 13), outline=LINE)
        dr.text((bx[0] + 24, bx[1] + 20), self.L["result"], font=self.f["h2"], fill=INK2)
        y = bx[1] + 56
        for st in self.res["steps"]:
            ok = bool(st.get("ok"))
            dr.text(
                (bx[0] + 24, y),
                f"{st['action']}: {self.L['step_ok'] if ok else self.L['step_no']}",
                font=self.f["big"],
                fill=(GREEN if ok else ORANGE),
            )
            y += 42
        dr.text(
            (bx[0] + 24, y + 6),
            f"{self.L['contacts']}: {self.res['collisions'] - self.res['bird_hits']}",
            font=self.f["b"],
            fill=INK,
        )
        return im


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    d, out = a[0], a[1]
    lang = a[2] if len(a) > 2 else "ru"
    fps = int(a[3]) if len(a) > 3 else 25
    C = Comp(d, lang)
    if "--frames" in sys.argv:
        for n in sys.argv[sys.argv.index("--frames") + 1].split(","):
            C.frame(min(int(n), len(C.frames) - 1)).save(f"{out}_{n}.jpg", quality=92)
        return
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
            out,
        ],
        stdin=subprocess.PIPE,
    )
    im = None
    for k in range(len(C.frames)):
        im = C.frame(k)
        ff.stdin.write(np.asarray(im).tobytes())
    if im is not None:
        im = C.outcome(im)
        for _ in range(fps * 4):
            ff.stdin.write(np.asarray(im).tobytes())
    ff.stdin.close()
    ff.wait()
    print("video", out, len(C.frames), "frames")


if __name__ == "__main__":
    main()
