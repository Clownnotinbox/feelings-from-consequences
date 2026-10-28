"""Stage F, check F1 (attributes).
usage: train_attrs.py DATA_DIR OUT_DIR [--epochs 12]"""

import argparse, json, random, time
from pathlib import Path
import numpy as np

HEADS = {
    "kind": [
        "pad",
        "dropzone",
        "pickup",
        "hangar",
        "water_tower",
        "mast",
        "tent",
        "container",
        "solar",
        "windsock",
    ],
    "colour": ["red", "orange", "yellow", "green", "blue", "white", "grey", "black", "purple", "brown"],
    "mark": ["H", "circle", "X"],
    "shape": ["round", "square"],
    "pattern": ["X", "rings", "cross"],
    "style": ["locker", "platform", "ridge", "dome", "legs", "column"],
    "door_open": ["False", "True"],
    "parcel_colour": [
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "white",
        "grey",
        "black",
        "purple",
        "brown",
    ],
    "lights": ["False", "True"],
    "flags": ["False", "True"],
    "rows": ["1", "2", "3"],
}
APPLIES = {
    "mark": ("pad",),
    "shape": ("pad",),
    "lights": ("pad",),
    "pattern": ("dropzone",),
    "flags": ("dropzone",),
    "style": ("pickup", "tent", "water_tower"),
    "door_open": ("hangar",),
    "parcel_colour": ("pickup",),
    "rows": ("solar",),
    "colour": tuple(k for k in HEADS["kind"] if k not in ("solar", "windsock")),
}


def targets(c):
    y = {}
    for h, vals in HEADS.items():
        if h == "kind":
            y[h] = vals.index(c["kind"])
            continue
        if c["kind"] not in APPLIES[h]:
            y[h] = -100
            continue
        v = c.get(h)
        v = "False" if v is None and h in ("lights", "flags") else v
        y[h] = vals.index(str(v)) if v is not None and str(v) in vals else -100
    return y


def build(pretrained=True):
    import torch, torchvision

    try:
        net = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        src = "imagenet" if pretrained else "scratch"
    except Exception as ex:
        net = torchvision.models.resnet18(weights=None)
        src = f"scratch ({type(ex).__name__})"
    net.fc = torch.nn.Identity()
    heads = torch.nn.ModuleDict({h: torch.nn.Linear(512, len(v)) for h, v in HEADS.items()})
    return torch.nn.ModuleDict({"body": net, "heads": heads}), src


def run(model, x):
    f = model["body"](x)
    return {h: m(f) for h, m in model["heads"].items()}


def main():
    import torch
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    D, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def load(split):
        rows = [json.loads(l) for l in open(D / f"crops_{split}.jsonl", encoding="utf-8")]
        X = np.stack(
            [np.array(Image.open(D / "crops" / split / f"{r['crop']}.jpg").convert("RGB")) for r in rows]
        )
        return rows, X

    tr, Xtr = load("train")
    va, Xva = load("val")
    te, Xte = load("test")
    print("crops", len(tr), len(va), len(te), flush=True)
    Ytr = {h: torch.tensor([targets(c)[h] for c in tr]) for h in HEADS}
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def prep(X, aug=False):
        x = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255
        if aug:
            x = x * (1 + 0.25 * (torch.rand(len(x), 1, 1, 1) - 0.5)) + 0.08 * (
                torch.rand(len(x), 1, 1, 1) - 0.5
            )  # brightness / exposure, NOT hue: colour is a label
            if random.random() < 0.5:
                x = x.flip(3)
        return ((x.clamp(0, 1) - mean) / std).cuda()

    model, src = build()
    model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.epochs * (len(tr) // 128 + 1))
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    log = {"init": src, "epochs": []}
    best = -1

    def evaluate(rows, X):
        model.eval()
        P = {h: [] for h in HEADS}
        with torch.no_grad():
            for i in range(0, len(X), 256):
                o = run(model, prep(X[i : i + 256]))
                for h in HEADS:
                    P[h].append(torch.softmax(o[h], 1).cpu().numpy())
        P = {h: np.concatenate(v) for h, v in P.items()}
        res = {}
        for h in HEADS:
            y = np.array([targets(c)[h] for c in rows])
            m = y >= 0
            big = m & (np.array([c["box_min"] for c in rows]) >= 30)
            res[h] = {
                "acc": round(float((P[h][m].argmax(1) == y[m]).mean()), 4) if m.any() else None,
                "acc_ge30px": round(float((P[h][big].argmax(1) == y[big]).mean()), 4) if big.any() else None,
                "n": int(m.sum()),
            }
        return res, P

    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        idx = np.random.default_rng(ep).permutation(len(tr))
        tot = 0.0
        for i in range(0, len(idx), 128):
            b = idx[i : i + 128]
            o = run(model, prep(Xtr[b], aug=True))
            loss = sum(ce(o[h], Ytr[h][b].cuda()) for h in HEADS)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            tot += loss.item()
        res, _ = evaluate(va, Xva)
        score = np.mean([v["acc"] for v in res.values() if v["acc"] is not None])
        log["epochs"].append(
            {
                "epoch": ep,
                "loss": round(tot / (len(idx) / 128), 4),
                "val_mean_acc": round(float(score), 4),
                "val": res,
            }
        )
        print(ep, round(tot, 2), round(float(score), 4), f"{time.time() - t0:.0f}s", flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), out / "attrs_best.pt")
            log["best_epoch"] = ep
    model.load_state_dict(torch.load(out / "attrs_best.pt"))
    res, P = evaluate(te, Xte)
    log["test"] = res
    (out / "attrs_log.json").write_text(json.dumps(log, indent=1))
    print("TEST", json.dumps(res), flush=True)
    model.eval()
    (
        torch.onnx.export(
            model["body"].cpu().float(),
            torch.zeros(1, 3, 112, 112),
            out / "attrs_body.onnx",
            opset_version=17,
        )
        if False
        else None
    )


if __name__ == "__main__":
    main()
