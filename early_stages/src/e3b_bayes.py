"""Exact Bayes-optimal accuracy for the E3b world."""

import argparse, itertools, json
import numpy as np


def main(a):
    T, q, F = a.turns, a.noise, a.flips
    sets = [
        f for f in itertools.combinations(range(3, T - 2), F) if all(b - c >= 3 for c, b in zip(f, f[1:]))
    ]
    if F == 1 and T == 8:
        sets = [(3,), (4,), (5,)]
    H = [
        (p0, f) for p0 in (0, 1) for f in sets
    ]  # uniform prior (rejection sampling in e3b.py is uniform over valid
    path = np.array(
        [[p0 ^ (sum(1 for x in f if x <= t) % 2) for t in range(T)] for p0, f in H]
    )  # pref at turn t
    g = np.random.default_rng(0)
    acc = np.zeros(T)
    acc_h = np.zeros(T)
    n = a.episodes
    h = F / T
    for _ in range(n):
        k = g.integers(len(H))
        obs = path[k] ^ (g.random(T) < q)
        logp = np.zeros(len(H))
        post = 0.5
        for t in range(T):
            w = np.exp(logp - logp.max())
            p1 = (w * path[:, t]).sum() / w.sum()
            acc[t] += (int(p1 > 0.5) if p1 != 0.5 else 0) == path[
                k, t
            ]  # exact posterior (ties -> 0 = COMFORT side)
            prior = 0.5 if t == 0 else post * (1 - h) + (1 - post) * h  # e3b.py constant-hazard filter
            acc_h[t] += int(prior > 0.5) == path[k, t]
            lik = np.where(path[:, t] == obs[t], 1 - q, q)
            logp += np.log(lik)
            ls = (1 - q) if obs[t] == 1 else q
            lc = q if obs[t] == 1 else (1 - q)
            z = prior * ls + (1 - prior) * lc
            post = prior * ls / z
    R = {
        "turns": T,
        "noise": q,
        "flips": F,
        "n_hypotheses": len(H),
        "exact_bayes_accuracy": float(acc.sum() / (n * T)),
        "constant_hazard_filter_accuracy": float(acc_h.sum() / (n * T)),
        "exact_by_turn": (acc / n).round(3).tolist(),
    }
    print(json.dumps(R))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--turns", type=int, default=16)
    p.add_argument("--noise", type=float, default=0.2)
    p.add_argument("--flips", type=int, default=2)
    p.add_argument("--episodes", type=int, default=40000)
    main(p.parse_args())
