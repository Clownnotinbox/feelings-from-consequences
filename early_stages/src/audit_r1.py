"""R1 CPU check before any LLM training."""

import argparse, json, math
import numpy as np, torch
import r1


def feats_for(arm, c, F):
    pe = F["p_emotion"][c["clip_i"]]
    sent = np.eye(len(r1.SENT))[list(r1.SENT).index(c["sentence"])]
    x = [sent]
    if arm in ("name", "name+raw"):
        x.append(np.r_[np.eye(6)[int(pe.argmax())], pe.max()])
    if arm in ("probs", "probs+raw"):
        x.append(pe)
    if arm in ("raw", "name+raw", "probs+raw"):
        x.append(F["emb"][c["clip_i"]])
    if arm == "intended":
        x.append(
            np.r_[
                np.eye(6)[list(r1.INTENDED).index(c["intended"])],
                np.eye(4)[list(r1.LEVEL_WORD).index(c["level"])],
            ]
        )
    return np.concatenate(x).astype(np.float32)


def fit_eval(clips, F, arm, q, l2, val=False):
    """val=True: fit on 3/4 of train actors, score on the other 1/4."""
    S = r1.EMO if q == "emo" else r1.LEV
    key = "emo_votes" if q == "emo" else "lev_votes"
    tr = [c for c in clips if c["split"] == "train" and c[key]]
    te = [c for c in clips if c["split"] == "test" and c[key]]
    if val:
        held = lambda c: int(c["clip"][:4]) % 4 == 1
        tr, te = [c for c in tr if not held(c)], [c for c in tr if held(c)]
    Xtr = np.stack([feats_for(arm, c, F) for c in tr])
    Xte = np.stack([feats_for(arm, c, F) for c in te])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    rows = [(i, S.index(v)) for i, c in enumerate(tr) for v in c[key]]
    X = torch.tensor(Xtr[[i for i, _ in rows]])
    y = torch.tensor([v for _, v in rows])
    torch.manual_seed(0)
    W = torch.zeros(X.shape[1], len(S), requires_grad=True)
    b = torch.zeros(len(S), requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.02)
    for _ in range(400):
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) + l2 * W.square().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    P = torch.softmax(torch.tensor(Xte) @ W + b, -1).detach().numpy()
    ll = np.mean([np.mean([math.log(max(1e-9, P[i, S.index(v)])) for v in c[key]]) for i, c in enumerate(te)])
    maj = [
        (i, max(S, key=c[key].count))
        for i, c in enumerate(te)
        if len(c[key]) >= 2
        and sorted(c[key].count(s) for s in S)[-1] > sorted(c[key].count(s) for s in S)[-2]
    ]
    r = {
        "loglik": round(float(ll), 4),
        "majority_acc": round(float(np.mean([P[i].argmax() == S.index(m) for i, m in maj])), 3),
    }
    if q == "int":
        ex = P @ np.arange(3)
        hum = np.array([np.mean([S.index(v) for v in c[key]]) for c in te])
        r["level_corr"] = round(float(np.corrcoef(ex, hum)[0, 1]), 3)
    return r


def main(a):
    F, clips, cuts = r1.load(a.features)
    R = {
        "level_cuts": cuts.tolist(),
        "n_clips": len(clips),
        "chance_loglik": {"emo": math.log(1 / 6), "int": math.log(1 / 3)},
    }
    for q in ("emo", "int"):
        for arm in ("none", "name", "probs", "raw", "name+raw", "probs+raw", "intended"):
            l2 = max(
                (1e-4, 1e-3, 1e-2), key=lambda l: fit_eval(clips, F, arm, q, l, val=True)["loglik"]
            )  # chosen on train actors only
            R[f"{q}/{arm}"] = dict(fit_eval(clips, F, arm, q, l2), l2=l2)
            print(q, arm, R[f"{q}/{arm}"], flush=True)
    json.dump(R, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="../runs/r1feat")
    p.add_argument("--out", default="../analysis/audit_r1.json")
    main(p.parse_args())
