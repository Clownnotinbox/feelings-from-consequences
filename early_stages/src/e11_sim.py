"""E11 design check (CPU)."""

import argparse, json
import numpy as np


def run(T, h, q, W, n, seed=0, eta=0.8, decay=0.5):
    g = np.random.default_rng(seed)
    acc = {k: 0 for k in ("bayes_full", f"bayes_last{W}", "dpr_state", "wsls")}
    for _ in range(n):
        pref = int(g.integers(2))
        post = 0.5
        w = np.zeros(2)
        el = np.zeros(2)
        obs = []
        last = None
        lr = 0
        for t in range(T):
            if t > 0 and g.random() < h:
                pref = 1 - pref
            prior = 0.5 if t == 0 else post * (1 - h) + (1 - post) * h
            # window: forward filter from a flat prior over the last W observations
            pw = 0.5
            for i, o in enumerate(obs[-W:]):
                if i:
                    pw = pw * (1 - h) + (1 - pw) * h  # a flip may happen between consecutive reactions
                l1, l0 = (1 - q, q) if o == 1 else (q, 1 - q)
                pw = pw * l1 / (pw * l1 + (1 - pw) * l0)
            if obs:
                pw = pw * (1 - h) + (1 - pw) * h
            acts = {
                "bayes_full": int(prior > 0.5),
                f"bayes_last{W}": int(pw > 0.5),
                "dpr_state": int(w[1] > w[0]),
                "wsls": 0 if last is None else (last if lr > 0 else 1 - last),
            }
            for k, a in acts.items():
                acc[k] += a == pref
            a = acts["dpr_state"]  # the DPR state is updated by its own actions (as in e3b.py)
            r = 1 if a == pref else -1
            if g.random() < q:
                r = -r
            o = a if r > 0 else 1 - a
            obs.append(o)  # what the reaction says about the preference
            l1, l0 = (1 - q, q) if o == 1 else (q, 1 - q)
            post = prior * l1 / (prior * l1 + (1 - prior) * l0)
            el = decay * el
            el[a] += 1
            d = r - w[a]
            w = np.clip(w + eta * d * el, -1, 1)
            last, lr = o, 1  # WSLS on the reaction: stay with what the reaction says
    return {k: round(v / (n * T), 3) for k, v in acc.items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--configs", default="48,0.0833,0.2,4")
    p.add_argument("--eta", type=float, default=0.8)
    p.add_argument("--decay", type=float, default=0.5)
    a = p.parse_args()
    for c in a.configs.split(";"):
        T, h, q, W = c.split(",")
        T, h, q, W = int(T), float(h), float(q), int(W)
        print(
            json.dumps(
                {
                    "T": T,
                    "h": round(h, 3),
                    "q": q,
                    "W": W,
                    "eta": a.eta,
                    "decay": a.decay,
                    **run(T, h, q, W, a.n, eta=a.eta, decay=a.decay),
                }
            )
        )
