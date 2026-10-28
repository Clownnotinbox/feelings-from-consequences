"""Drive the demo server through its HTTP API like a user would.
usage: demo_check.py "TEXT" X,Y [X,Y ...] [--gps-off S] [--attacks] [--wind W] [--answer K] [--port 8765] [--tmax 900]
"""

import argparse, json, time, urllib.request as U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("points", nargs="+")
    ap.add_argument("--gps-off", type=float)
    ap.add_argument("--attacks", action="store_true")
    ap.add_argument("--wind", type=float, default=3.0)
    ap.add_argument("--answer", type=int, default=0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--tmax", type=float, default=900.0)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    post = lambda p, d: json.loads(
        U.urlopen(
            U.Request(base + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"}),
            timeout=180,
        ).read()
    )
    get = lambda p: json.loads(U.urlopen(base + p, timeout=30).read())
    for _ in range(60):
        try:
            get("/api/state")
            break
        except Exception:
            time.sleep(3)
    r = post("/api/parse", {"text": a.text})
    print("PARSE", r.get("summary") or r, flush=True)
    pts = [[float(v) for v in p.split(",")] for p in a.points]
    r = post(
        "/api/mission",
        {
            "text": a.text,
            "parsed": r.get("parsed"),
            "points": pts,
            "return": None,
            "gps_off": a.gps_off,
            "wind": a.wind,
            "attacks": a.attacks,
        },
    )
    print("START", r.get("ok"), r.get("error", ""), flush=True)
    last = None
    t0 = time.time()
    s = {}
    while time.time() - t0 < a.tmax:
        s = get("/api/state")
        if s.get("question"):
            print(
                "QUESTION",
                [(c["p"], c["x"], c["y"]) for c in s["question"]["cands"]],
                "-> answering",
                a.answer,
                flush=True,
            )
            post("/api/answer", {"choice": a.answer})
        if s.get("phase") != last:
            print(
                f"{time.time() - t0:5.0f}s sim {s.get('t')} {s.get('phase')} battery {s.get('battery')} gps {s.get('gps')}",
                flush=True,
            )
            last = s.get("phase")
        if s.get("phase") == "done" and s.get("result"):
            break
        time.sleep(2)
    print(
        "\n".join(
            l for l in s.get("log", [])[-40:] if "no progress" not in l and "safety: avoiding" not in l
        ),
        flush=True,
    )
    print("OUTCOME", json.dumps((s.get("result") or {}).get("outcome"), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
