"""E10 'does the brain need backprop?': online learning from dopamine of person x place preferences."""

import argparse, json, time
from pathlib import Path
import numpy as np

KINDS = ("table", "linear", "knn", "mb_sparse", "mb_dense", "bp", "fa", "modprop", "np", "shuffle")
GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
TAUS = (0.5, 0.7, 0.8, 0.9, 0.95)


INPUT = "raw"


def load(root):
    """raw: voice x-vector + CLIP image."""
    v = np.load(Path(root) / "e9_speaker.npz")
    im = np.load(Path(root) / "e7feat" / "img.npz")
    actor = v["actor"]
    I = im["emb"] - im["emb"].mean(0)
    I /= np.linalg.norm(I, axis=1, keepdims=True)
    V = v["emb"]
    if INPUT == "percept":
        S = np.stack([V[actor == k].sum(0) for k in range(1, 25)])
        n = np.bincount(actor)[1:]
        C = S[actor - 1] - V  # own centroid without the clip itself
        cos = V @ (S / np.linalg.norm(S, axis=1, keepdims=True)).T
        cos[np.arange(len(V)), actor - 1] = (V * C).sum(1) / np.linalg.norm(C, axis=1)
        P = np.exp(20 * (cos - cos.max(1, keepdims=True)))
        V = P / P.sum(1, keepdims=True)
        I = im["probs"]
    return V, actor, I, im["y"], im["split"]


TASK = "private"


def world(seed, actor, y, split):
    g = np.random.default_rng(seed)
    if TASK == "private":  # arbitrary private half of the places per person
        pref = np.stack([g.permutation([0, 0, 0, 1, 1, 1]) for _ in range(24)])  # 1 = COMFORT in that place
    elif TASK == "xor":  # structured: person type XOR place type (home-like vs public places)
        ptype = g.permutation([0] * 12 + [1] * 12)
        qtype = np.array([0, 0, 0, 1, 1, 1])
        pref = (ptype[:, None] ^ qtype[None, :]).astype(int)
    else:  # 'person': the style depends on the person only (E9-like)
        pref = np.repeat(g.permutation([0] * 12 + [1] * 12)[:, None], 6, 1)
    clips = {p: g.permutation(np.where(actor == p + 1)[0]) for p in range(24)}
    vtr = {p: c[:45] for p, c in clips.items()}
    vte = {p: c[45:] for p, c in clips.items()}
    itr = {q: np.where((y == q) & (split == "train"))[0] for q in range(6)}
    ite = {q: np.where((y == q) & (split == "test"))[0] for q in range(6)}
    return pref, vtr, vte, itr, ite, set(g.choice(24, 8, replace=False).tolist())


class Learner:
    def __init__(self, kind, d, seed, lr_o, lr_i, H=1000, k=50, types=16, sigma=0.1):
        g = np.random.default_rng(seed + 1000)
        self.kind, self.lr_o, self.lr_i, self.k, self.sigma = kind, lr_o, lr_i, k, sigma
        self.W = (g.standard_normal((H, d)) / np.sqrt(d)).astype(np.float32)
        self.b = np.zeros(H, np.float32)
        self.O = np.zeros((2, 144 if kind == "table" else d if kind == "linear" else H), np.float32)
        self.B = (g.standard_normal((2, H)) / np.sqrt(H)).astype(np.float32)
        self.type = g.integers(0, types, H)
        self.ntypes = types
        self.g = g
        self.past = []
        self.K, self.A, self.D = [], [], []

    def hidden(self, x, noise=None, pq=None):
        if self.kind == "table":
            h = np.zeros(144, np.float32)
            h[pq] = 1
            return h, None
        if self.kind == "linear":
            return x / np.linalg.norm(x), None
        z = self.W @ x + self.b
        if noise is not None:
            z = z + noise
        h = np.maximum(z, 0)
        if self.kind in ("mb_sparse", "shuffle"):
            m = np.zeros_like(h)
            top = np.argpartition(-z, self.k)[: self.k]
            m[top] = 1
            h = h * m
        return h / max(1e-6, float(np.linalg.norm(h))), z

    def act(self, x, eps, pq=None):
        if self.kind == "knn":
            xn = x / np.linalg.norm(x)
            q = np.zeros(2, np.float32)
            if self.K:
                m = np.stack(self.K) @ xn > self.lr_i
                A = np.array(self.A)
                D = np.array(self.D)
                q = np.array([D[m & (A == 0)].sum(), D[m & (A == 1)].sum()], np.float32)
            a = int(self.g.integers(2)) if self.g.random() < eps else int(q.argmax())
            return a, q, (xn, None, None)
        noise = (
            (self.sigma * self.g.standard_normal(len(self.b))).astype(np.float32)
            if self.kind == "np"
            else None
        )
        h, z = self.hidden(x, noise, pq)
        q = self.O @ h
        a = int(self.g.integers(2)) if self.g.random() < eps else int(q.argmax())
        return a, q, (h, z, noise)

    def learn(self, x, a, q, cache, r):
        h, z, noise = cache
        d = r - q[a]
        if self.kind == "knn":
            self.K.append(h)
            self.A.append(a)
            self.D.append(self.lr_o * d)
            return
        if self.kind == "shuffle":
            self.past.append(d)
            d = self.past[self.g.integers(len(self.past))]
        if self.kind in ("bp", "fa", "modprop", "np"):
            act = (z > 0).astype(float)
            if self.kind == "bp":
                fb = self.O[a]
            elif self.kind == "fa":
                fb = self.B[a]
            elif self.kind == "modprop":
                mean = np.bincount(self.type, self.O[a], self.ntypes) / np.maximum(
                    np.bincount(self.type, minlength=self.ntypes), 1
                )
                fb = mean[self.type]
            if self.kind == "np":
                e = d * noise / self.sigma * act  # REINFORCE on the hidden noise
            else:  # through the readout normalisation h/|h| (exact for bp)
                n = max(1e-6, float(np.linalg.norm(np.maximum(z, 0))))
                e = d * (fb - h * float(h @ fb)) / n * act
            e = e.astype(np.float32)
            self.W += self.lr_i * np.outer(e, x)
            self.b += self.lr_i * e
        self.O[a] += self.lr_o * d * h


def run(kind, data, seed, lr_o, lr_i, T=6000, eps=0.1):
    V, actor, I, y, split = data
    pref, vtr, vte, itr, ite, changed = world(seed, actor, y, split)
    pref = pref.copy()
    L = Learner(kind, V.shape[1] + I.shape[1], seed, lr_o, lr_i)
    g = np.random.default_rng(seed + 7)
    ok, who = [], []
    for t in range(T):
        if t == T // 2:
            for p in changed:
                pref[p] = 1 - pref[p]
        p, q = int(g.integers(24)), int(g.integers(6))
        x = np.concatenate([V[g.choice(vtr[p])], I[g.choice(itr[q])]]).astype(np.float32)
        a, qv, cache = L.act(x, eps, p * 6 + q)
        r = 1.0 if a == pref[p, q] else -1.0
        L.learn(x, a, qv, cache, r)
        ok.append(a == pref[p, q])
        who.append(p)
        if t == T // 2 - 1:
            test_before = evaluate(L, V, I, pref, vte, ite)
    ok = np.array(ok, float)
    who = np.array(who)
    h = T // 2
    ch = np.isin(who, list(changed))
    return {
        "cum_acc_all": float(ok.mean()),
        "cum_acc_first_half": float(ok[:h].mean()),
        "acc_before_flip_last500": float(ok[h - 500 : h].mean()),
        "heldout_acc_before_flip": test_before,
        "changed_after_flip_first500": float(ok[h : h + 500][ch[h : h + 500]].mean()),
        "unchanged_after_flip_first500": float(ok[h : h + 500][~ch[h : h + 500]].mean()),
        "acc_last500": float(ok[-500:].mean()),
        "curve_250": [float(ok[i : i + 250].mean()) for i in range(0, T, 250)],
    }


def evaluate(L, V, I, pref, vte, ite):
    g = np.random.default_rng(0)
    hits = []
    for p in range(24):
        for q in range(6):
            for _ in range(3):
                x = np.concatenate([V[g.choice(vte[p])], I[g.choice(ite[q])]]).astype(np.float32)
                if L.kind == "knn":
                    a, _, _ = L.act(x, 0.0)
                    hits.append(a == pref[p, q])
                    continue
                h, _ = L.hidden(x, None, p * 6 + q)
                hits.append(int((L.O @ h).argmax()) == pref[p, q])
    return float(np.mean(hits))


def main(a):
    global TASK, INPUT
    TASK, INPUT = a.task, a.input
    data = load(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = {"tuning": {}, "test": {}}
    for kind in a.kinds:
        best = None
        for lo in GRID:
            for li in (GRID if kind in ("bp", "fa", "modprop", "np") else TAUS if kind == "knn" else (0.0,)):
                s = float(
                    np.mean([run(kind, data, sd, lo, li)["cum_acc_all"] for sd in a.dev])
                )  # learning + re-learning
                if best is None or s > best[0]:
                    best = (s, lo, li)
        R["tuning"][kind] = {"dev_cum_acc": best[0], "lr_out": best[1], "lr_in": best[2]}
        runs = [run(kind, data, sd, best[1], best[2]) for sd in a.test]
        R["test"][kind] = {
            k: (
                np.mean([r[k] for r in runs], 0).tolist()
                if isinstance(runs[0][k], list)
                else float(np.mean([r[k] for r in runs]))
            )
            for k in runs[0]
        } | {"per_seed_cum_acc": [r["cum_acc_all"] for r in runs]}
        print(
            kind,
            json.dumps(
                {k: v for k, v in R["test"][kind].items() if k not in ("curve_250", "per_seed_cum_acc")}
            ),
            R["tuning"][kind],
            round(time.time() - t0),
            "s",
            flush=True,
        )
    (out / "RESULT.json").write_text(json.dumps(R, indent=1), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="../runs")
    p.add_argument("--out", required=True)
    p.add_argument("--kinds", nargs="+", default=list(KINDS))
    p.add_argument("--task", default="private")
    p.add_argument("--input", default="raw", choices=("raw", "percept"))
    p.add_argument("--dev", type=int, nargs="+", default=[21, 22, 23])
    p.add_argument("--test", type=int, nargs="+", default=list(range(31, 41)))
    main(p.parse_args())
