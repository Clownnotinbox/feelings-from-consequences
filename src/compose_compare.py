"""Supplementary videos: one mission flown with the feeling core and with a classic reflex, side by side.
usage: compose_compare.py FULL_DIR REFLEX_DIR OUT.mp4 [ru|en] [FPS=25] [--still OUT.jpg]
FULL_DIR and REFLEX_DIR are render_flight.py outputs of the same mission. --still writes one frame (just after the reflex's first
static contact) instead of the video."""

import json, subprocess, sys, textwrap
import numpy as np
from PIL import Image, ImageDraw
from compose_video import Comp, STATES, ACTION_STATES, BG, PANEL, LINE, INK, INK2, INK3, BLUE, ORANGE, RED, GREEN, YELLOW, W, H, M, G

T2 = {
    "ru": {
        "title": "Одна и та же миссия: с ядром чувств и без него",
        "same": "Одинаковы место, текст, ветер, отказы и атаки. Отличается только быстрый слой.",
        "full": "С ЯДРОМ ЧУВСТВ",
        "reflex": "БЕЗ ЯДРА · обычный рефлекс",
        "gps_at": "GPS пропадает на {t:.0f} с",
        "gps_none": "GPS работает",
        "attacks": "атаки птиц",
        "wind": "ветер {v:.0f} м/с",
        "speedup": "ускорено ×5",
        "over": "миссия закончена",
        "row_full": "с ядром",
        "row_reflex": "без ядра",
        "tl": "Шкала времени: касания неподвижного (●), уклонения (|), чувство «опасность за 2 с» у ядра",
        "home": "вернулся домой",
        "t": "t = {t} с",
        "places": {"kazan_novosav": "Казань, Ново-Савиновский район", "tokyo_res": "Токио, жилой квартал"},
    },
    "en": {
        "title": "One mission, with the feeling core and without it",
        "same": "Same place, text, wind, failures and attacks. Only the fast layer differs.",
        "full": "WITH THE FEELING CORE",
        "reflex": "NO CORE · classic reflex",
        "gps_at": "GPS lost at {t:.0f} s",
        "gps_none": "GPS works",
        "attacks": "bird attacks",
        "wind": "wind {v:.0f} m/s",
        "speedup": "speed ×5",
        "over": "mission over",
        "row_full": "with core",
        "row_reflex": "no core",
        "tl": "Timeline: static contacts (●), avoidance (|), the core's feeling 'danger in 2 s'",
        "home": "back home",
        "t": "t = {t} s",
        "places": {"kazan_novosav": "Kazan, Novo-Savinovsky district", "tokyo_res": "Tokyo, residential district"},
    },
}

VW = (W - 2 * M - G) // 2  # one view: 928 x 522 (16:9)
VH = VW * 9 // 16
TOP = 112
MAP_H = 236
TL_Y0 = TOP + VH + G + MAP_H + G


class Pair:
    def __init__(self, d_full, d_reflex, lang="ru"):
        self.L2 = T2[lang]
        self.C = [Comp(d_full, lang), Comp(d_reflex, lang)]
        for k, c in enumerate(self.C):
            x0 = M + k * (VW + G)
            c.box_3d = (x0, TOP, x0 + VW, TOP + VH)
            c.box_map = (x0, TOP + VH + G, x0 + VW, TOP + VH + G + MAP_H)
            c.map_setup()
        self.f = self.C[0].f
        self.n = max(len(c.frames) for c in self.C)
        self.t_end = max(c.t_end for c in self.C)

    def side(self, k, idx):
        c = self.C[k]
        j = min(idx, len(c.frames) - 1)
        s = c.frames[j]
        upto = next(i for i, q in enumerate(c.S) if q is s)
        return c, s, upto, idx >= len(c.frames)

    def draw_top(self, dr, t):
        mt = self.C[0].meta
        x0, y0, x1 = M, M, W - M
        dr.text((x0, y0), self.L2["title"], font=self.f["h1"], fill=INK)
        dr.text((x0, y0 + 32), self.L2["same"], font=self.f["s"], fill=INK2)
        cond = [self.L2["gps_at"].format(t=mt["gps_off"]) if mt["gps_off"] else self.L2["gps_none"]]
        if mt["attacks"]:
            cond.append(self.L2["attacks"])
        cond.append(self.L2["wind"].format(v=float(np.hypot(*mt["wind"][:2]))))
        place = self.L2["places"].get(mt["place"], mt["place"])
        dr.text((x0, y0 + 54), "  ·  ".join([place] + cond), font=self.f["s"], fill=INK2)
        qx = x0 + 820
        dr.text((qx, y0), self.C[0].L["request"], font=self.f["xs"], fill=INK3)
        for i, ln in enumerate(textwrap.wrap("«" + mt["text"] + "»", 80)[:2]):
            dr.text((qx, y0 + 16 + 22 * i), ln, font=self.f["req"], fill=INK)
        tv = f"{t:.1f}".replace(".", ",") if self.L2 is T2["ru"] else f"{t:.1f}"
        tt = self.L2["t"].format(t=tv) + "   " + self.L2["speedup"]
        dr.text((x1 - dr.textlength(tt, font=self.f["m"]), y0 + 58), tt, font=self.f["m"], fill=INK3)

    def draw_view(self, im, dr, k, c, s, over):
        x0, y0, x1, y1 = c.box_3d
        fr = Image.open(c.d / "chase" / f"{s['frame']:05d}.jpg")
        w, h = fr.size
        ch = int(w * 9 / 16)
        fr = fr.crop((0, (h - ch) // 2, w, (h - ch) // 2 + ch)).resize((x1 - x0, y1 - y0))
        if over:
            fr = Image.blend(fr, Image.new("RGB", fr.size, BG), 0.45)
        im.paste(fr, (x0, y0))
        col = BLUE if k == 0 else ORANGE
        dr.rectangle((x0, y0, x1, y1), outline=col, width=3)
        lab = self.L2["full" if k == 0 else "reflex"]
        bw = dr.textlength(lab, font=self.f["h2"]) + 28
        dr.rectangle((x0, y0, x0 + bw, y0 + 32), fill=col)
        dr.text((x0 + 14, y0 + 7), lab, font=self.f["h2"], fill=(0, 0, 0))
        cnt = f"{c.L['contacts']}: {s['static_hits']}"
        cw = dr.textlength(cnt, font=self.f["big"]) + 28
        dr.rectangle((x1 - cw, y0, x1, y0 + 46), fill=(10, 11, 13))
        dr.text((x1 - cw + 14, y0 + 6), cnt, font=self.f["big"], fill=(RED if s["static_hits"] else INK))
        st = "ACTION" if s["state"] in ACTION_STATES else s["state"]
        if st in STATES:
            name = c.L["states"][STATES.index(st)]
            dr.rectangle((x0, y1 - 30, x0 + dr.textlength(name, font=self.f["h2"]) + 28, y1), fill=(10, 11, 13))
            dr.text((x0 + 14, y1 - 24), name, font=self.f["h2"], fill=INK)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        for q in range(4):
            dr.arc((cx - 22, cy - 22, cx + 22, cy + 22), 90 * q + 20, 90 * q + 70, fill=YELLOW, width=2)
        if over:
            self.outcome(dr, c)

    def outcome(self, dr, c):
        x0, y0, x1, y1 = c.box_3d
        bx = (x0 + 120, y0 + 130, x1 - 120, y0 + 130 + 70 + 42 * len(c.res["steps"]))
        dr.rectangle(bx, fill=(10, 11, 13), outline=LINE)
        dr.text((bx[0] + 20, bx[1] + 14), self.L2["over"] + " · " + c.L["result"], font=self.f["xs"], fill=INK2)
        y = bx[1] + 38
        for st in c.res["steps"]:
            ok = bool(st.get("ok"))
            dr.text((bx[0] + 20, y), f"{st['action']}: {c.L['step_ok'] if ok else c.L['step_no']}", font=self.f["big"],
                    fill=(GREEN if ok else ORANGE))
            y += 42
        n = c.res["collisions"] - c.res["bird_hits"]
        dr.text((bx[0] + 20, y + 2), f"{c.L['contacts']}: {n}", font=self.f["b"], fill=(RED if n else INK))

    def draw_timeline(self, dr, t):
        x0, y0, x1, y1 = M, TL_Y0, W - M, H - M
        dr.rectangle((x0, y0, x1, y1), fill=PANEL, outline=LINE)
        dr.text((x0 + 14, y0 + 8), self.L2["tl"].upper(), font=self.f["xs"], fill=INK3)
        lx, rx = x0 + 130, x1 - 18
        tx = lambda v: lx + (rx - lx) * v / max(self.t_end, 1.0)
        rh = (y1 - y0 - 34) / 2
        for k, c in enumerate(self.C):
            ya = y0 + 28 + k * rh
            yb = ya + rh - 6
            col = BLUE if k == 0 else ORANGE
            dr.text((x0 + 14, ya + rh / 2 - 10), self.L2["row_full" if k == 0 else "row_reflex"], font=self.f["s"], fill=col)
            dr.line((lx, yb, tx(c.t_end), yb), fill=LINE)
            if k == 0:
                pts = [(tx(q["t"]), yb - max(0.0, q["feel"].get("danger_2", -1)) * (rh - 12)) for q in c.S[::2] if q["t"] <= t]
                if len(pts) > 1:
                    dr.line(pts, fill=ORANGE, width=2)
            for a, q in zip(c.S, c.S[1:]):
                if q["t"] > t:
                    break
                if q["fires"] > a["fires"]:
                    dr.line((tx(q["t"]), yb - 8, tx(q["t"]), yb), fill=BLUE, width=1)
                if q["static_hits"] > a["static_hits"]:
                    dr.ellipse((tx(q["t"]) - 6, yb - 12, tx(q["t"]) + 6, yb), fill=RED)
        cx = tx(t)
        dr.line((cx, y0 + 26, cx, y1 - 4), fill=INK, width=1)

    def frame(self, idx, final=False):
        im = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(im)
        t = 0.0
        for k in range(2):
            c, s, upto, over = self.side(k, idx)
            over = over or final
            t = max(t, s["t"])
            self.draw_view(im, dr, k, c, s, over)
            c.draw_map(im, dr, s, upto)
        self.draw_top(dr, t)
        self.draw_timeline(dr, t)
        return im

    def still_index(self):
        S = self.C[1].S
        hits = [q["t"] for a, q in zip(S, S[1:]) if q["static_hits"] > a["static_hits"]]
        t = hits[0] + 1.0 if hits else self.t_end / 2
        fr = self.C[1].frames
        return min(range(len(fr)), key=lambda i: abs(fr[i]["t"] - t))


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    still = sys.argv[sys.argv.index("--still") + 1] if "--still" in sys.argv else None
    if still in a:
        a.remove(still)
    d_full, d_reflex, out = a[0], a[1], a[2]
    lang = a[3] if len(a) > 3 else "ru"
    fps = int(a[4]) if len(a) > 4 else 25
    P = Pair(d_full, d_reflex, lang)
    if still:
        P.frame(P.still_index()).save(still, quality=92)
        print("still", still)
        return
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE,
    )
    for k in range(P.n):
        ff.stdin.write(np.asarray(P.frame(k)).tobytes())
    im = P.frame(P.n - 1, final=True)
    for _ in range(fps * 5):
        ff.stdin.write(np.asarray(im).tobytes())
    ff.stdin.close()
    ff.wait()
    print("compare video", out, P.n, "frames")


if __name__ == "__main__":
    main()
