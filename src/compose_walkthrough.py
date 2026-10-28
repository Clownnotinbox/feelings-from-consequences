"""The app walkthrough video.
usage: compose_walkthrough.py SHOTS.json OUT.mp4 [FPS=25]"""

import json, subprocess, sys, textwrap
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from compose_video import fnt, BG, PANEL, LINE, INK, INK2, INK3, BLUE, YELLOW, W, H

F = {
    "step": fnt("Inter-SemiBold.otf", 26),
    "title": fnt("Inter-SemiBold.otf", 30),
    "body": fnt("Inter-Regular.otf", 22),
    "card": fnt("Inter-SemiBold.otf", 52),
    "cardb": fnt("Inter-Regular.otf", 28),
    "small": fnt("Inter-Regular.otf", 16),
}
CAP = 190  # caption band height


def shot_frame(s):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    if s.get("card"):
        d.text((160, 380), s["title"], font=F["card"], fill=INK)
        y = 470
        for ln in s.get("body", "").split("\n"):
            for w in textwrap.wrap(ln, 80) or [""]:
                d.text((160, y), w, font=F["cardb"], fill=INK2)
                y += 42
        d.rectangle((160, 350, 260, 356), fill=BLUE)
        return im
    src = Image.open(s["img"]).convert("RGB")
    avail_w, avail_h = W - 96, H - CAP - 72
    sc = min(avail_w / src.width, avail_h / src.height)
    sh = src.resize((int(src.width * sc), int(src.height * sc)), Image.LANCZOS)
    ox, oy = (W - sh.width) // 2, 36
    d.rectangle((ox - 1, oy - 1, ox + sh.width, oy + sh.height), outline=LINE)
    im.paste(sh, (ox, oy))
    if s.get("ring"):
        rx, ry = ox + s["ring"][0] * sc, oy + s["ring"][1] * sc
        for r, wdt in ((26, 4), (40, 2)):
            d.ellipse((rx - r, ry - r, rx + r, ry + r), outline=YELLOW, width=wdt)
    y0 = H - CAP
    d.rectangle((0, y0, W, H), fill=PANEL)
    d.line((0, y0, W, y0), fill=LINE)
    if s.get("step"):
        d.rounded_rectangle((60, y0 + 34, 120, y0 + 94), radius=8, fill=BLUE)
        tw = d.textlength(s["step"], font=F["step"])
        d.text((90 - tw / 2, y0 + 47), s["step"], font=F["step"], fill=INK)
    d.text((150, y0 + 30), s.get("title", ""), font=F["title"], fill=INK)
    y = y0 + 76
    for w in textwrap.wrap(s.get("body", ""), 120)[:3]:
        d.text((150, y), w, font=F["body"], fill=INK2)
        y += 32
    return im


def main(spec, out, fps=25):
    shots = json.load(open(spec, encoding="utf-8"))
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
    prev = None
    fade = int(fps * 0.4)
    for s in shots:
        cur = np.asarray(shot_frame(s), np.float32)
        if prev is not None:
            for k in range(1, fade + 1):
                ff.stdin.write((prev * (1 - k / fade) + cur * (k / fade)).astype(np.uint8).tobytes())
        for _ in range(int(fps * float(s.get("sec", 4)))):
            ff.stdin.write(cur.astype(np.uint8).tobytes())
        prev = cur
    ff.stdin.close()
    ff.wait()
    print("walkthrough", out, len(shots), "shots")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 25)
