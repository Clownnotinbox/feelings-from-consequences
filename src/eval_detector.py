"""Stage F, check F1 (detection).
usage: eval_detector.py WEIGHTS DATA_DIR OUT.json"""

import json, sys
from pathlib import Path
import numpy as np


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def collect(model, D, split, imgsz=960):
    out = []
    for f in sorted((D / "images" / split).glob("*.jpg")):
        r = model.predict(str(f), imgsz=imgsz, conf=0.05, verbose=False)[0]
        W, H = r.orig_shape[1], r.orig_shape[0]
        pred = [
            (int(c), float(s), *map(float, b))
            for c, s, b in zip(
                r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()
            )
        ]
        gt = []
        for l in (D / "labels" / split / (f.stem + ".txt")).read_text().split("\n"):
            if not l:
                continue
            c, x, y, w, h = l.split()
            x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
            gt.append((int(c), x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        out.append((pred, gt))
    return out


def stats(frames, thr, names, min_px=30):
    rec = {}
    tp_p = fp_p = 0
    for pred, gt in frames:
        P = [p for p in pred if p[1] >= thr]
        used = set()
        for g in gt:
            big = min(g[3] - g[1], g[4] - g[2]) >= min_px
            k = rec.setdefault(names[g[0]], [0, 0, 0, 0])
            hit = None
            for j, p in enumerate(P):
                if j not in used and p[0] == g[0] and iou(p[2:], g[1:]) >= 0.5:
                    hit = j
                    break
            if hit is not None:
                used.add(hit)
            k[0] += big and hit is not None
            k[1] += big
            k[2] += hit is not None
            k[3] += 1
        for j, p in enumerate(P):  # precision over predictions whose box is >= min_px
            if min(p[4] - p[2], p[5] - p[3]) < min_px:
                continue
            tp_p += j in used
            fp_p += j not in used
    tot = [sum(v[i] for v in rec.values()) for i in range(4)]
    return {
        "recall_ge30": round(tot[0] / max(tot[1], 1), 4),
        "precision_ge30": round(tp_p / max(tp_p + fp_p, 1), 4),
        "recall_all": round(tot[2] / max(tot[3], 1), 4),
        "per_class_recall_ge30": {k: (round(v[0] / v[1], 3) if v[1] else None, v[1]) for k, v in rec.items()},
    }


def main():
    from ultralytics import YOLO

    w, D, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    model = YOLO(w)
    names = model.names
    va, te = collect(model, D, "val"), collect(model, D, "test")
    grid = [round(t, 2) for t in np.arange(0.1, 0.9, 0.05)]
    f1 = []
    for t in grid:
        s = stats(va, t, names)
        f1.append(
            2 * s["recall_ge30"] * s["precision_ge30"] / max(s["recall_ge30"] + s["precision_ge30"], 1e-9)
        )
    thr = grid[int(np.argmax(f1))]
    res = {"threshold_from_val": thr, "val": stats(va, thr, names), "test": stats(te, thr, names)}
    res["F1_rule"] = {
        "recall_ge30>=0.9": res["test"]["recall_ge30"] >= 0.9,
        "precision_ge30>=0.9": res["test"]["precision_ge30"] >= 0.9,
    }
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
