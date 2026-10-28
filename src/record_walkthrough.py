"""Record the app walkthrough: a real browser gives one mission through the app's page.
usage: record_walkthrough.py URL OUT_DIR [SPEED=4] [ru|en]"""

import asyncio
import json
import math
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

COLOUR_ACC = {
    "red": "красную",
    "orange": "оранжевую",
    "yellow": "жёлтую",
    "green": "зелёную",
    "blue": "синюю",
    "white": "белую",
    "grey": "серую",
    "black": "чёрную",
    "purple": "фиолетовую",
    "brown": "коричневую",
}

TO_SCREEN = """([x, y]) => {
  const s = document.getElementById('map');
  const p = s.createSVGPoint();
  p.x = x; p.y = -y;
  const q = p.matrixTransform(s.getScreenCTM());
  return [q.x, q.y];
}"""


def pick_target(objects):
    # a ground pad whose colour no other pad has
    pads = [o for o in objects if o["kind"] == "pad" and not o["roof"]]
    for o in pads:
        same = [q for q in pads if q["colour"] == o["colour"]]
        if len(same) == 1:
            return o
    return pads[0] if pads else objects[0]


COLOUR_EN = {k: k for k in COLOUR_ACC}


async def main(url, out_dir, speed, lang="ru"):
    en = lang == "en"
    url = url + ("/?lang=en" if en else "/?lang=ru")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    marks = []
    start = time.time()

    def mark(name):
        marks.append([name, round(time.time() - start, 2)])

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(
            viewport={"width": 1600, "height": 900},
            record_video_dir=str(out),
            record_video_size={"width": 1600, "height": 900},
        )
        page = await ctx.new_page()
        await page.goto(url)
        await page.wait_for_function("M !== null")
        await page.wait_for_timeout(1500)
        world = await page.evaluate("M")
        target = pick_target(world["objects"])
        print("target", target["kind"], target["colour"], target["x"], target["y"], flush=True)

        # zoom the map in around the target
        tx, ty = await page.evaluate(TO_SCREEN, [target["x"], target["y"]])
        await page.mouse.move(tx, ty)
        for _ in range(5):
            await page.mouse.wheel(0, -120)
            await page.wait_for_timeout(250)
        await page.wait_for_timeout(800)

        # step 1 on the page: find the target (the tooltip says what it is), then click near it
        mark("step1")
        tx, ty = await page.evaluate(TO_SCREEN, [target["x"], target["y"]])
        await page.mouse.move(tx - 120, ty + 90)
        await page.mouse.move(tx, ty, steps=30)
        await page.wait_for_timeout(3500)
        mark("step2")
        rx, ry = await page.evaluate(TO_SCREEN, [target["x"] + 14, target["y"] - 14])
        await page.mouse.move(rx, ry, steps=20)
        await page.mouse.click(rx, ry)
        await page.wait_for_timeout(2000)
        await page.click("#s1 .next")

        # step 2: the request in plain words, then what the drone understood
        mark("step3")
        if en:
            text = "Land on the " + COLOUR_EN.get(target["colour"], "") + " landing pad"
            if target["facts"].get("mark") == "H":
                text += " with the letter H"
            text += ". Then return home."
        else:
            text = "Сядь на " + COLOUR_ACC.get(target["colour"], "") + " посадочную площадку"
            if target["facts"].get("mark") == "H":
                text += " с буквой H"
            text += ". Потом вернись домой."
        await page.click("#text")
        await page.keyboard.type(text, delay=55)
        await page.wait_for_timeout(1200)
        mark("step4")
        await page.click("#prs")
        await page.wait_for_selector("#pv", state="visible", timeout=90000)
        await page.wait_for_timeout(4500)
        await page.click("#s2 .next")

        # step 3: a ready set of conditions: GPS lost after 20 s (well inside the flight)
        mark("step5")
        await page.wait_for_timeout(800)
        await page.click("[data-preset=nogps]")
        await page.wait_for_timeout(2500)
        await page.click(f"#speedseg [data-s='{speed}']")
        mark("flight")
        await page.click("#go")
        await page.wait_for_selector("#app.fly", timeout=90000)

        # the flight: answer a question if the drone asks one; a still for the paper
        for k in range(600):
            await page.wait_for_timeout(1000)
            if k == 25:
                await page.screenshot(path=str(out / "app_flight.png"))
            if await page.locator("#q").is_visible():
                state = await page.evaluate("fetch('/api/state').then(r => r.json())")
                cands = (state.get("question") or {}).get("cands") or []
                if cands:
                    dist = [math.hypot(c["x"] - target["x"], c["y"] - target["y"]) for c in cands]
                    await page.wait_for_timeout(2500)
                    await page.click("#qc img >> nth=" + str(dist.index(min(dist))))
            if await page.locator("#result").is_visible():
                break
        mark("done")
        await page.wait_for_timeout(8000)
        mark("end")
        video = page.video
        await ctx.close()
        await browser.close()
        path = await video.path()

    info = {"video": str(path), "marks": marks, "text": text, "target": target}
    (out / "walkthrough.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("recorded", path, marks, flush=True)


if __name__ == "__main__":
    speed = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    asyncio.run(main(sys.argv[1], sys.argv[2], speed, sys.argv[4] if len(sys.argv) > 4 else "ru"))
