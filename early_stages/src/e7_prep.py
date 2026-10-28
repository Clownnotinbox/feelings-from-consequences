"""E7 perception features."""

import argparse, glob, io, json, tarfile
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download, snapshot_download
import env7


def probe(Xtr, ytr, Xte, n_cls, epochs=300, folds=5, seed=0):
    """Softmax regression; returns out-of-fold train probs and test probs."""

    def fit(X, y):
        g = torch.Generator().manual_seed(seed)
        W = torch.zeros(X.shape[1], n_cls, requires_grad=True)
        b = torch.zeros(n_cls, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=0.05)
        X, y = torch.tensor(X), torch.tensor(y)
        for _ in range(epochs):
            loss = torch.nn.functional.cross_entropy(X @ W + b, y) + 1e-3 * W.square().sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return lambda Z: torch.softmax(torch.tensor(Z) @ W + b, -1).detach().numpy()

    oof = np.zeros((len(Xtr), n_cls), np.float32)
    idx = np.random.RandomState(seed).permutation(len(Xtr))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        oof[te] = fit(Xtr[tr], ytr[tr])(Xtr[te])
    return oof, fit(Xtr, ytr)(Xte)


def places(out, max_per_class=200):
    from transformers import CLIPModel, CLIPProcessor
    import pyarrow.parquet as pq
    from PIL import Image

    files = [
        hf_hub_download(
            "dpdl-benchmark/Places365-Validation",
            f"data/val-0000{i}-of-00005.parquet",
            repo_type="dataset",
            local_dir=str(out / "places365"),
        )
        for i in range(5)
    ]
    # Labels are indices into the official CSAILVision
    cats = Path(out / "categories_places365.txt").read_text().splitlines()
    names = {int(l.split()[1]): l.split()[0].split("/")[-1] for l in cats if l.strip()}
    want = {i: p for i, n in names.items() for p in env7.PLACES if n == p}
    assert len(want) == len(env7.PLACES), want
    imgs, labels = [], []
    for f in files:
        t = pq.read_table(f)
        for img, lab in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
            if lab in want and labels.count(want[lab]) < max_per_class:
                imgs.append(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
                labels.append(want[lab])
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    embs = []
    with torch.no_grad():
        for i in range(0, len(imgs), 64):
            e = model.get_image_features(**proc(images=imgs[i : i + 64], return_tensors="pt").to("cuda"))
            embs.append(torch.nn.functional.normalize(e, dim=-1).float().cpu())
    E = torch.cat(embs).numpy()
    y = np.array([env7.PLACES.index(l) for l in labels])
    split = np.array(
        ["test" if i % 10 < 3 else "train" for i in range(len(y))]
    )  # 30% held-out photos per class
    return E, y, split


def sounds(out):
    d = Path(
        snapshot_download(
            "atom-in-the-universe/esc50_CLAP_embeddings",
            repo_type="dataset",
            local_dir=str(out / "esc50clap"),
        )
    )
    E, y, split = [], [], []
    for tf in sorted(glob.glob(str(d / "*" / "*.tar"))):
        with tarfile.open(tf) as t:
            members = {m.name: m for m in t.getmembers() if m.isfile()}
            for name, m in members.items():
                if not name.endswith(".json"):
                    continue
                info = json.loads(t.extractfile(m).read())
                cat = info["original_data"]["category"].replace(" ", "_")
                if cat not in env7.SOUNDS:
                    continue
                emb = torch.load(
                    io.BytesIO(t.extractfile(members[name[:-5] + ".pth"]).read()),
                    weights_only=True,
                    map_location="cpu",
                )
                E.append(torch.nn.functional.normalize(emb.reshape(-1).float(), dim=0).numpy())
                y.append(env7.SOUNDS.index(cat))
                split.append("test" if info["original_data"]["fold"] == 5 else "train")
    return np.stack(E), np.array(y), np.array(split)


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = {}
    for kind, (E, y, split), n in (
        ("img", places(out), len(env7.PLACES)),
        ("snd", sounds(out), len(env7.SOUNDS)),
    ):
        tr, te = split == "train", split == "test"
        oof, pte = probe(E[tr], y[tr], E[te], n)
        P = np.zeros((len(y), n), np.float32)
        P[tr] = oof
        P[te] = pte
        R[kind] = {
            "n": int(len(y)),
            "train": int(tr.sum()),
            "test": int(te.sum()),
            "probe_accuracy_train_oof": float((oof.argmax(1) == y[tr]).mean()),
            "probe_accuracy_test": float((pte.argmax(1) == y[te]).mean()),
        }
        np.savez(out / f"{kind}.npz", emb=E.astype(np.float32), y=y, split=split, probs=P)
    (out / "PERCEPTION.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    main(p.parse_args())
