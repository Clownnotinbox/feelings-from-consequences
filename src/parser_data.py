"""Stage F language data.
usage: parser_data.py OUT.jsonl SPLIT N_PER_PLACE WORKERS"""

import json, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def job(args):
    place, split, k0, n = args
    import city3d as C, missions as M

    out = []
    for seed in range(k0, k0 + max(1, n // 25)):
        W, _, _ = C.build(place, seed, actors=False)
        for j in range(25):
            if len(out) >= n:
                break
            rng = np.random.default_rng(zlib.crc32(f"parser/{place}/{seed}/{j}".encode()))
            bank = "train" if split != "test" else ("test" if j % 2 else "train")
            m = M.make(W, rng, bank=bank)
            if m is None:
                continue
            out.append(
                {
                    "place": place,
                    "split": (
                        split if split != "test" else ("test_novel" if bank == "test" else "test_familiar")
                    ),
                    "lang": m["lang"],
                    "detail": m["detail"],
                    "text": m["text"],
                    "truth": m["truth"],
                }
            )
    return out


def main():
    path, split, n, workers = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    from osm_fetch2 import AREAS
    import city3d as C

    places = [
        p
        for p, a in AREAS.items()
        if (a[2] == split or (split == "test" and p == "kazan_novosav")) and (C.OSM2 / f"{p}.json").exists()
    ]
    chunks = 8
    jobs = [(p, split, 1000 * c, n // chunks) for p in places for c in range(chunks)]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(job, jobs):
            rows += r
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(split, places, len(rows), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
