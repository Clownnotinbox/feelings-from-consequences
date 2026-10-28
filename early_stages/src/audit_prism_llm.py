"""Fly-brain (DPR) check on PRISM with a rich stimulus code."""

import argparse, collections, json
import numpy as np


def turns_by_user(F):
    T = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in range(len(F["z"])):
        T[str(F["user"][i])][(str(F["time"][i]), str(F["conv"][i]), int(F["turn"][i]))].append(i)
    out = {}
    for u, d in T.items():
        seq = []
        for k in sorted(d):
            idx = d[k]
            ch = [i for i in idx if F["chosen"][i]]
            if len(idx) >= 2 and len(ch) == 1:
                seq.append((ch[0], [i for i in idx if i != ch[0]]))
        out[u] = seq
    return out


def fit_logistic(D, l2=1e-3, steps=400, lr=0.5):
    X = np.concatenate([D, -D])
    y = np.r_[np.ones(len(D)), np.zeros(len(D))]
    w = np.zeros(D.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
    return w


def run(users, seqs, code, w, eta, decay=1.0):
    hit = n = 0
    for u in users:
        wu = np.zeros_like(w)
        for c, others in seqs[u]:
            for o in others:
                d = code[c] - code[o]
                s = d @ (w + wu)
                hit += (s > 0) + 0.5 * (s == 0)
                n += 1
            wu *= decay
            for o in others:
                d = code[c] - code[o]
                p = 1 / (1 + np.exp(-np.clip(d @ (w + wu), -30, 30)))
                wu += eta * (1 - p) * d
    return hit / max(1, n), n


def main(a):
    F = dict(np.load(a.features, allow_pickle=False))
    seqs = turns_by_user(F)
    users = sorted(seqs)
    test = set(users[4::5])
    rest = [u for u in users if u not in test]
    B = set(rest[3::4])
    A = [u for u in rest if u not in B]
    Z = F["z"] / (F["z"][[i for u in A for c, os in seqs[u] for i in [c] + os]].std(0) + 1e-6)
    rng = np.random.RandomState(0)
    P = rng.randn(Z.shape[1], a.kc)
    H = Z @ P
    thr = np.sort(H, 1)[:, -a.k][:, None]
    KC = (H >= thr).astype(np.float32)  # top-k winner-take-all
    R = {"n_users": len(users), "n_test": len(test)}
    for name, code in (("llm", Z), ("kc", KC)):
        D = np.array([code[c] - code[o] for u in A for c, os in seqs[u] for o in os])
        w = fit_logistic(D, l2=a.l2)
        acc0, n = run(sorted(test), seqs, code, w, 0.0)
        best = max(
            a.etas, key=lambda e: run(sorted(B), seqs, code, w, e, a.decay)[0]
        )  # chosen on B users only
        acc1, _ = run(sorted(test), seqs, code, w, best, a.decay)
        R[name] = {
            "population": round(float(acc0), 4),
            "personal": round(float(acc1), 4),
            "eta_chosen_on_B": best,
            "pairs": n,
        }
        print(name, R[name], flush=True)
    L = F["length"].astype(float)
    hit = n = 0
    for u in sorted(test):
        for c, os in seqs[u]:
            for o in os:
                hit += (L[c] > L[o]) + 0.5 * (L[c] == L[o])
                n += 1
    R["longer_baseline"] = round(hit / n, 4)
    print(R)
    json.dump(R, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kc", type=int, default=2000)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1])
    p.add_argument("--decay", type=float, default=1.0)
    main(p.parse_args())
