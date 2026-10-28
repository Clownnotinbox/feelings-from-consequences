"""Audit of the voice channel."""

import argparse, json, random
import numpy as np, torch
import env6

E6_DISTRESS = env6.distress


def e13_distress(emotion, intensity):
    if emotion in ("surprised", "disgust", "angry", "sad", "fearful"):
        return 0.8 if intensity == "strong" else 0.5
    return 0.15


def data(feats, world, seed):
    env6.distress = E6_DISTRESS if world == "e6" else e13_distress
    clips = {
        sp: [
            (
                i,
                env6.EMOTIONS[int(feats["emotion"][i])],
                "strong" if feats["intensity"][i] else "normal",
                int(feats["actor"][i]),
            )
            for i in range(len(feats["emotion"]))
            if feats["split"][i] == sp
        ]
        for sp in ("train", "test")
    }
    tr, te = env6.items("train", 4000, 1600 + seed, clips), env6.items("test", 2000, 601, clips)
    return tr, te


def inputs(its, feats, arm, mu, sd):
    S = {s: i for i, s in enumerate(env6.STRATEGIES)}
    X, y = [], []
    for it in its:
        s = it["signals"]
        c = s["clip"]
        x = [s["energy"], s["threat"]]
        pe = feats["p_emotion"][c]
        named = np.eye(8)[int(pe.argmax())].tolist() + [float(pe.max())]
        if arm == "none":
            pass
        elif arm == "named":
            x += named
        elif arm == "named+int":
            x += named + feats["p_intensity"][c].tolist()  # E6 'AT': recognizer also names intensity
        elif arm == "emb_raw":
            x += feats["emb"][c].tolist()
        elif arm == "emb_z":
            x += ((feats["emb"][c] - mu) / sd).tolist()
        elif arm == "named+emb_z":
            x += named + ((feats["emb"][c] - mu) / sd).tolist()
        elif arm == "oracle":
            x += np.eye(8)[env6.EMOTIONS.index(s["emotion"])].tolist() + [float(s["intensity"] == "strong")]
        X.append(x)
        y.append(S[env6.teacher(s)])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(y)


def train(Xtr, ytr, Xte, yte, regime, seed):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Linear(256, 5))
    if regime == "adapter":  # the E6/E13 regime: Adam 3e-3, 400 steps, batch 8, no weight decay
        opt, steps, bs, wd = torch.optim.Adam(net.parameters(), 3e-3), 400, 8, 0.0
    else:  # a well-conditioned regime
        opt, steps, bs, wd = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2), 3000, 64, 1e-2
    drop = torch.nn.Dropout(0.2 if regime != "adapter" else 0.0)
    for _ in range(steps):
        i = torch.randint(0, len(Xtr), (bs,))
        loss = torch.nn.functional.cross_entropy(net(drop(Xtr[i])), ytr[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return float((net(Xte).argmax(1) == yte).float().mean()), float(
            (net(Xtr).argmax(1) == ytr).float().mean()
        )


def main(a):
    feats = dict(np.load(a.features))
    trm = feats["split"] == "train"
    mu, sd = feats["emb"][trm].mean(0), feats["emb"][trm].std(0) + 1e-6
    R = {}
    for world in ("e6", "e13"):
        for arm in ("none", "named", "named+int", "emb_raw", "emb_z", "named+emb_z", "oracle"):
            for regime in ("adapter", "good"):
                acc = []
                for seed in (1, 2):
                    tr, te = data(feats, world, seed)
                    Xtr, ytr = inputs(tr, feats, arm, mu, sd)
                    Xte, yte = inputs(te, feats, arm, mu, sd)
                    acc.append(train(Xtr, ytr, Xte, yte, regime, seed))
                R[f"{world}/{arm}/{regime}"] = {
                    "test": round(float(np.mean([t for t, _ in acc])), 3),
                    "train": round(float(np.mean([t for _, t in acc])), 3),
                }
                print(world, arm, regime, R[f"{world}/{arm}/{regime}"], flush=True)
    json.dump(R, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="../runs/e6feat/voice.npz")
    p.add_argument("--out", default="../analysis/audit_voice.json")
    main(p.parse_args())
