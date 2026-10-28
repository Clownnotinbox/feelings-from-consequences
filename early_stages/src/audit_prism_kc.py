"""R2' CPU check, fly-style."""

import argparse, collections, json, math, re
import numpy as np


def feats(t):
    s = t.strip()
    w = s.split()
    return np.array(
        [
            math.log1p(len(s)),
            math.log1p(len(w)),
            s.count("\n") > 2,
            bool(re.search(r"^\s*(\d+\.|-|\*)\s", s, re.M)),
            "?" in s,
            "!" in s,
            bool(re.search(r"\b(sorry|as an ai|i cannot|i can't|i'm not able)\b", s, re.I)),
            bool(re.search(r"\b(i|my|me)\b", s, re.I)),
            bool(re.search(r"\b(you|your)\b", s, re.I)),
            len(re.findall(r"[.!?]", s)) / max(1, len(w)) * 10,
            sum(c.isupper() for c in s) / max(1, len(s)) * 10,
        ],
        float,
    )


def load(path):
    by_user = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        c = json.loads(line)
        turns = collections.defaultdict(list)
        for h in c["conversation_history"]:
            if h["role"] == "model" and h.get("score") is not None:
                turns[int(h["turn"])].append(
                    {
                        "f": feats(h["content"]),
                        "chosen": str(h.get("if_chosen")) == "True",
                        "score": float(h["score"]),
                    }
                )
        by_user[c["user_id"]].append((c["generated_datetime"], [turns[t] for t in sorted(turns)]))
    return {u: [t for _, ts in sorted(cs) for t in ts] for u, cs in by_user.items()}


def pairs(turns):
    for turn in turns:
        ch = [r for r in turn if r["chosen"]]
        if len(turn) >= 2 and len(ch) == 1:
            yield turn, [(ch[0]["f"] - o["f"]) for o in turn if o is not ch[0]]


def main(a):
    U = load(a.conversations)
    users = sorted(U)
    test = set(users[4::5])
    D = np.array([d for u in users if u not in test for _, ds in pairs(U[u]) for d in ds])
    mu, sd = 0, np.abs(D).mean(0) + 1e-6
    rng = np.random.RandomState(0)
    K = rng.randn(D.shape[1], a.kc) * (rng.rand(D.shape[1], a.kc) < 0.3)  # sparse random expansion
    kc = lambda d: np.tanh((d / sd) @ K)
    X = np.concatenate([kc(D), -kc(D)])
    y = np.r_[np.ones(len(D)), np.zeros(len(D))]
    w = np.zeros(a.kc)
    for _ in range(300):
        p = 1 / (1 + np.exp(-X @ w))
        w -= 0.1 * (X.T @ (p - y) / len(y) + 1e-3 * w)
    R = {}
    for eta in (0.0, 0.05, 0.1, 0.2):
        hit = n = 0
        for u in test:
            wu = np.zeros(a.kc)
            for turn, ds in pairs(U[u]):
                for d in ds:
                    z = kc(d)
                    n += 1
                    hit += float(z @ (w + wu) > 0) + 0.5 * float(z @ (w + wu) == 0)
                for d in ds:  # after the reaction
                    z = kc(d)
                    p = 1 / (1 + math.exp(-(z @ (w + wu))))
                    wu += eta * (1 - p) * z
        R[f"eta{eta}"] = round(hit / n, 4)
        print("eta", eta, "pairwise acc", R[f"eta{eta}"], "pairs", n, flush=True)
    json.dump(R, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--conversations", default="../../../work/datasets/prism/conversations.jsonl")
    p.add_argument("--kc", type=int, default=200)
    p.add_argument("--out", default="../analysis/audit_prism_kc.json")
    main(p.parse_args())
