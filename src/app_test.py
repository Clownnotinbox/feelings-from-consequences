"""End-to-end check of the app page in a real browser: every step of a task, a full flight, a stopped flight, and lag numbers.
usage: app_test.py URL OUT_DIR [ru|en]
Writes OUT_DIR/app_test.json (problems found, timings) and screenshots. Exit code 1 if any problem was found."""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright


async def main(url, out_dir, lang="ru"):
    en = lang == "en"
    url = url + ("/?lang=en" if en else "/?lang=ru")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    problems, info = [], {}
    shot = lambda page, name: page.screenshot(path=str(out / f"{name}.png"))
    ru_seen = {}

    async def no_russian(page, where):  # an English page shows no Russian (the request text is the operator's own)
        if en:
            body = await page.evaluate("document.body.innerText")
            ru = sorted(set(re.findall(r"[А-Яа-яЁё][А-Яа-яЁё ,.:-]{2,}", body)))
            if ru:
                ru_seen[where] = ru[:10]

    def need(ok, what):
        if not ok:
            problems.append(what)
        return ok

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1600, "height": 900})
        console = []
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
        page.on("requestfailed", lambda r: console.append(f"requestfailed: {r.url} {r.failure}"))
        t0 = time.time()
        await page.goto(url)
        await page.wait_for_function("M !== null", timeout=60000)
        info["map_load_s"] = round(time.time() - t0, 1)
        need(await page.locator("#kinds .kind").count() > 0, "no object kinds listed in step 1")
        await shot(page, "1_plan")
        await no_russian(page, "plan")

        # step 1 by hand: a click on the map adds a point, undo removes it
        box = await page.locator("#map").bounding_box()
        no_points = await page.text_content("#ptinfo")
        await page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        need((await page.text_content("#ptinfo")) != no_points, "a map click did not add a point")
        await page.click("#undo")
        need((await page.text_content("#ptinfo")) == no_points, "undo did not remove the point")
        await page.click("#kinds .kind >> nth=0")
        await page.click("#kinds .kind >> nth=0")

        # step 2: a ready example fills the text and the points
        await page.click("#s1 .next")
        need(await page.locator("#s2.open").count() == 1, "Next did not open step 2")
        await page.click("#exbox summary")
        await page.wait_for_selector("#exlist .ex", timeout=60000)
        await page.click("#exlist .ex >> nth=0")
        text = await page.input_value("#text")
        need(len(text) > 10, "the example did not fill the text")
        need((await page.text_content("#ptinfo")) != no_points, "the example did not set points")
        t0 = time.time()
        await page.click("#prs")
        await page.wait_for_selector("#pv", state="visible", timeout=120000)
        info["parse_s"] = round(time.time() - t0, 1)
        lines = await page.locator("#pvs li").all_text_contents()
        info["parse_summary"] = lines
        need(len(lines) >= 2 and all("=" not in x and "{" not in x for x in lines), "the parse summary is not plain words")
        await shot(page, "2_parse")

        # step 3: a ready set of conditions
        await page.click("#s2 .next")
        await page.click("[data-preset=nogps]")
        need("GPS" in await page.text_content("#sum3"), "the GPS preset did not change the summary")
        need(await page.input_value("#gps") == "off", "the GPS preset did not set the field")
        await shot(page, "3_conditions")

        # fly at x4 and measure
        await page.click("#speedseg [data-s='4']")
        await page.click("#go")
        await page.wait_for_selector("#app.fly", timeout=90000)
        await page.wait_for_function("document.getElementById('v3msg').style.display==='none'", timeout=60000)
        st0 = await page.evaluate("fetch('/api/state').then(r=>r.json())")
        w0 = time.time()
        srcs, lat = set(), []
        for _ in range(40):  # 20 s
            srcs.add(await page.get_attribute("#v3d", "src"))
            a = time.time()
            await page.evaluate("fetch('/api/state?log=999999').then(r=>r.json())")
            lat.append(time.time() - a)
            await page.wait_for_timeout(500)
        st1 = await page.evaluate("fetch('/api/state').then(r=>r.json())")
        wall = time.time() - w0
        info["frames_per_s_seen"] = round(len(srcs) / wall, 1)
        info["state_latency_ms_median"] = round(1000 * sorted(lat)[len(lat) // 2])
        info["state_latency_ms_max"] = round(1000 * max(lat))
        info["sim_speed_vs_wall"] = round((st1.get("t", 0) - st0.get("t", 0)) / wall, 2)
        info["tick_ms"], info["render_ms"] = st1.get("tick_ms"), st1.get("render_ms")
        need(info["frames_per_s_seen"] >= 1.0, f"the 3D view barely updates ({info['frames_per_s_seen']} frames/s seen)")
        need(info["state_latency_ms_median"] < 150, f"state replies are slow (median {info['state_latency_ms_median']} ms)")
        info["page_says"] = await page.text_content("#realspeed")
        need(info["sim_speed_vs_wall"] >= 0.9, f"the flight runs slower than real time (x{info['sim_speed_vs_wall']})")
        need(info["page_says"].startswith(("now ×", "сейчас ×")) and any(w in info["page_says"] for w in ("не успевает", "cannot go faster")) == (info["sim_speed_vs_wall"] < 3.2),
             f"the page does not tell the real speed ({info['page_says']!r} at x{info['sim_speed_vs_wall']})")
        need((await page.text_content("#nowphase")) not in ("", "—"), "the phase is empty in flight")
        need(await page.locator("#feel .bar").count() == 5, "the five feeling bars are missing")
        need((await page.text_content("#t_gps")) != "—", "the GPS tile is empty")
        await shot(page, "4_flight")
        await no_russian(page, "flight")
        await page.click("#swap")
        await page.wait_for_timeout(1500)
        await shot(page, "5_flight_mapbig")
        await page.click("#swap")
        await page.click("#modes [data-m=top]")
        await page.wait_for_timeout(1500)
        await shot(page, "6_flight_top")
        await page.click("#modes [data-m=chase]")

        # let it finish (answering questions as a person would: the first candidate); a long search is stopped after 7 min
        t_end = time.time() + 420
        while time.time() < t_end and await page.locator("#result").is_hidden():
            if await page.locator("#q").is_visible():
                await page.click("#qc img >> nth=0")
            await page.wait_for_timeout(1000)
        if await page.locator("#result").is_hidden():
            info["first_flight"] = "still flying after 7 min, stopped"
            await page.click("#stop")
            await page.wait_for_selector("#result", state="visible", timeout=60000)
        finished = need(await page.locator("#result").is_visible(), "the first flight shows no outcome")
        if finished:
            await no_russian(page, "result")
            info["outcome"] = await page.locator("#resl li").all_text_contents()
            await shot(page, "7_result")
            await page.click("#again")
            need(await page.locator("#app.plan").count() == 1, "'Repeat' did not go back to planning")
            need(await page.input_value("#text") == text, "'Repeat' lost the text")

            # a second flight, stopped by the operator
            await page.click("#go")
            await page.wait_for_selector("#app.fly", timeout=90000)
            await page.wait_for_timeout(8000)
            await page.click("#stop")
            await page.wait_for_selector("#result", state="visible", timeout=60000)
            stop_lines = await page.locator("#resl li").all_text_contents()
            need(bool(stop_lines) and ("stopped" if en else "остановлен") in stop_lines[0], "Stop did not end the flight")
            await page.click("#newtask")
            need(await page.input_value("#text") == "", "'New task' did not clear the text")
            need((await page.text_content("#ptinfo")) == no_points, "'New task' did not clear the points")
        await shot(page, "8_end")
        await no_russian(page, "end")
        info["russian_on_english_page"] = ru_seen
        need(not ru_seen, f"Russian text on the English page: {ru_seen}")
        info["console"] = console[:40]
        need(not [c for c in console if c.startswith(("pageerror", "error"))], "errors in the page console")
        await browser.close()
    rep = {"problems": problems, "info": info}
    (out / "app_test.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "ru")))
