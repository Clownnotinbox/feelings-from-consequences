"""Captions for the recorded app walkthrough.
usage: compose_walk_video.py WALK_DIR OUT.mp4 [en]"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from compose_video import fnt, BG, PANEL, LINE, INK, INK2, BLUE

W, H, FPS = 1920, 1080, 25
# (mark, number of the step on the page, title, text); the page numbers its four steps itself
STEPS = [
    ("", "", "Как дать задание дрону", "Один запрос: точка на карте и текст. Дальше дрон работает сам."),
    ("step1", "1", "Где цель: найдите её на карте", "Наведите мышь на значок, подсказка покажет, что это и какого цвета. Список есть в «Обозначениях»."),
    ("step2", "1", "Щёлкните рядом с целью", "Это примерная точка, промах на десятки метров допустим. Для двух шагов ставят две точки."),
    ("step3", "2", "Что сделать: напишите своими словами", "По-русски или по-английски, подробно или коротко."),
    ("step4", "2", "Проверьте, как дрон понял задание", "Разбор показан простыми словами. Если он неверен, поправьте текст или разбор."),
    ("step5", "3", "Условия полёта: выберите готовый набор", "Здесь «Пропадёт GPS»: через 20 секунд, ветер 3 м/с. Всё можно настроить и вручную. Потом «Лететь»."),
    ("flight", "", "Дрон выполняет задание сам",
     "Крупно 3D-вид за дроном, в углу карта с путём. Справа: что дрон делает сейчас, чувства ядра, камера."),
    ("done", "", "Итог решает мир, а не дрон", "Карточка «Итог»: на что сел дрон, были ли касания, где он сейчас. Дальше: новое задание или повтор."),
]
STEPS_EN = [
    ("", "", "How to give the drone a mission",
     "One request: a point on the map and a text. Then the drone works alone."),
    ("step1", "1", "Where: find the target on the map", "Hover over a marker: the tooltip says what it is and its colour. The legend lists all objects."),
    ("step2", "1", "Click near the target", "This is a rough point; tens of metres off is fine. A two-step mission gets two points."),
    ("step3", "2", "What: type the mission in your own words", "Russian or English, detailed or short."),
    ("step4", "2", "Check what the drone understood", "The parse is shown in plain words. If it is wrong, fix the text or the parse."),
    ("step5", "3", "Conditions: pick a ready set", "Here 'GPS will be lost': after 20 seconds, wind 3 m/s. Everything can be set by hand too. Then Fly."),
    ("flight", "", "The drone does the mission alone",
     "Large: the 3D chase view, the map with the track in the corner. Right: what the drone is doing, the core's feelings, camera."),
    ("done", "", "The world judges the outcome, not the drone", "The Outcome card: what it landed on, any contacts, where it is now. Then a new task or a repeat."),
]
F_NUM = fnt("Inter-SemiBold.otf", 26)
F_TITLE = fnt("Inter-SemiBold.otf", 30)
F_BODY = fnt("Inter-Regular.otf", 21)


def caption_at(t, marks):
    # the last step whose mark is before t
    times = dict(marks)
    current = 0
    for k, (name, _, _, _) in enumerate(STEPS):
        if name == "" or (name in times and times[name] <= t):
            current = k
    return current


def frame_with_caption(img, k):
    im = Image.new("RGB", (W, H), BG)
    im.paste(img, (160, 16))
    d = ImageDraw.Draw(im)
    d.rectangle((159, 15, 1760, 916), outline=LINE)
    y0 = 936
    d.rectangle((0, y0, W, H), fill=PANEL)
    d.line((0, y0, W, y0), fill=LINE)
    name, num, title, body = STEPS[k]
    if num:
        d.rounded_rectangle((160, y0 + 30, 214, y0 + 84), radius=8, fill=BLUE)
        d.text((187 - d.textlength(num, font=F_NUM) / 2, y0 + 41), num, font=F_NUM, fill=INK)
    x = 240 if num else 160
    d.text((x, y0 + 24), title, font=F_TITLE, fill=INK)
    d.text((x, y0 + 72), body, font=F_BODY, fill=INK2)
    return im


def main(walk_dir, out):
    info = json.loads((Path(walk_dir) / "walkthrough.json").read_text(encoding="utf-8"))
    marks = info["marks"]
    reader = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            info["video"],
            "-vf",
            f"fps={FPS}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    writer = subprocess.Popen(
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
            str(FPS),
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
    size = 1600 * 900 * 3
    n = 0
    while True:
        raw = reader.stdout.read(size)
        if len(raw) < size:
            break
        img = Image.frombytes("RGB", (1600, 900), raw)
        k = caption_at(n / FPS, marks)
        writer.stdin.write(np.asarray(frame_with_caption(img, k)).tobytes())
        n += 1
    writer.stdin.close()
    writer.wait()
    print("walkthrough video", out, n, "frames")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "en":
        STEPS = STEPS_EN
    main(sys.argv[1], sys.argv[2])
