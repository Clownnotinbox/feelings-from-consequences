"""R1 data: CREMA-D (Cao et al."""

import argparse, csv, io, json, re, time, wave
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

EMOS = (
    "ANG",
    "DIS",
    "FEA",
    "HAP",
    "NEU",
    "SAD",
)  # file-name codes; crowd letters A D F H N S in the same order
LETTER = {"A": 0, "D": 1, "F": 2, "H": 3, "N": 4, "S": 5}
QUERY = {"1": "voice", "2": "face", "3": "av"}


def probe_grouped(X, y, groups, Xte, k, folds=5, epochs=400, l2=1e-3, seed=0):
    """Softmax regression on standardised inputs; out-of-fold probs with folds grouped by actor."""
    mu, sd = X.mean(0), X.std(0) + 1e-6

    def fit(A, b):
        torch.manual_seed(seed)
        A = torch.tensor((A - mu) / sd, dtype=torch.float32)
        b = torch.tensor(b)
        W = torch.zeros(A.shape[1], k, requires_grad=True)
        c = torch.zeros(k, requires_grad=True)
        opt = torch.optim.Adam([W, c], lr=0.02)
        for _ in range(epochs):
            loss = torch.nn.functional.cross_entropy(A @ W + c, b) + l2 * W.square().sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return (
            lambda Z: torch.softmax(torch.tensor((Z - mu) / sd, dtype=torch.float32) @ W + c, -1)
            .detach()
            .numpy()
        )

    ug = np.array(sorted(set(groups)))
    rng = np.random.RandomState(seed)
    rng.shuffle(ug)
    oof = np.zeros((len(X), k), np.float32)
    for f in range(folds):
        te = np.isin(groups, ug[f::folds])
        oof[te] = fit(X[~te], y[~te])(X[te])
    return oof, (fit(X, y)(Xte) if Xte is not None else None)


def read_wav(path):
    with wave.open(str(path)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype={2: np.int16, 4: np.int32}[sw])
    x = (raw.reshape(-1, ch).mean(1) / float(2 ** (8 * sw - 1))).astype("float32")
    if sr != 16000:
        x = np.interp(np.arange(0, len(x) / sr, 1 / 16000), np.arange(len(x)) / sr, x).astype("float32")
    return x


def main(a):
    repo, out = Path(a.repo), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = {}
    wavs = sorted(
        p
        for p in (repo / "AudioWAV").glob("*.wav")
        if re.match(r"\d{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}\.wav$", p.name)
    )
    meta = []
    for p in wavs:
        actor, sent, emo, lvl = p.stem.split("_")
        meta.append({"clip": p.stem, "actor": int(actor), "sentence": sent, "emotion": emo, "level": lvl})
    R["n_wav"] = len(meta)
    # crowd ratings
    rows = list(csv.DictReader(open(repo / "finishedResponses.csv", encoding="utf-8", errors="replace")))
    R["ratings_header"] = list(rows[0].keys())
    R["ratings_example"] = rows[:3]
    R["n_ratings"] = len(rows)
    rat = defaultdict(lambda: {"voice": [], "face": [], "av": []})
    for r in rows:
        q = QUERY.get(str(r.get("queryType", "")).strip())
        e = str(r.get("respEmo", "")).strip()[:1]
        if q is None or e not in LETTER:
            continue
        try:
            lv = float(r.get("respLevel", "nan"))
        except ValueError:
            lv = float("nan")
        rat[r["clipName"].strip()][q].append([LETTER[e], lv])
    (out / "ratings.json").write_text(json.dumps({m["clip"]: rat[m["clip"]] for m in meta}), encoding="utf-8")
    nv = [len(rat[m["clip"]]["voice"]) for m in meta]
    R["voice_votes_per_clip"] = {
        "mean": float(np.mean(nv)),
        "min": int(min(nv)),
        "zero": int(sum(v == 0 for v in nv)),
    }
    # WavLM-large features, every layer, mean over time
    from transformers import AutoFeatureExtractor, WavLMModel

    fx = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(a.device).eval()
    L = np.zeros((len(meta), model.config.num_hidden_layers + 1, model.config.hidden_size), np.float16)
    for i, p in enumerate(wavs):
        inp = fx(read_wav(p), sampling_rate=16000, return_tensors="pt").to(a.device)
        with torch.no_grad():
            hs = model(**inp, output_hidden_states=True).hidden_states
        L[i] = torch.stack([h[0].mean(0) for h in hs]).float().cpu().numpy()
        if i % 1000 == 0:
            print(i, round(time.time() - t0), flush=True)
    np.save(out / "wavlm_layers.npy", L)
    R["feature_seconds"] = time.time() - t0
    actor = np.array([m["actor"] for m in meta])
    split = np.array(["test" if m["actor"] % 5 == 0 else "train" for m in meta])
    emo = np.array([EMOS.index(m["emotion"]) for m in meta])
    tr, te = split == "train", split == "test"
    # layer choice on train actors only
    acc = {}
    for l in range(L.shape[1]):
        oof, _ = probe_grouped(L[tr, l].astype(np.float32), emo[tr], actor[tr], None, 6, epochs=200)
        acc[l] = float((oof.argmax(1) == emo[tr]).mean())
        print("layer", l, round(acc[l], 3), flush=True)
    best = max(acc, key=acc.get)
    R["layer_oof_acc"] = acc
    R["layer"] = best
    E = L[:, best].astype(np.float32)
    oof, pte = probe_grouped(E[tr], emo[tr], actor[tr], E[te], 6)
    P = np.zeros((len(E), 6), np.float32)
    P[tr] = oof
    P[te] = pte
    R["recognizer_acc_train_oof"] = float((oof.argmax(1) == emo[tr]).mean())
    R["recognizer_acc_test_actors"] = float((pte.argmax(1) == emo[te]).mean())
    np.savez(
        out / "voice.npz",
        emb=E,
        p_emotion=P,
        emotion=emo,
        actor=actor,
        split=split,
        clip=np.array([m["clip"] for m in meta]),
        sentence=np.array([m["sentence"] for m in meta]),
        level=np.array([m["level"] for m in meta]),
    )
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (out / "PREP.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in R.items() if k != "ratings_example"}, indent=1, default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
