"""R2 CPU check on ESConv."""

import argparse, json, math
import numpy as np, torch

S = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]


def episodes(D, eta=0.5, decay=0.9):
    """One row per rating after the first."""
    rows = []
    for k, x in enumerate(D):
        I = x["survey_score"]["seeker"].get("initial_emotion_intensity")
        I = int(I) if I else 3
        past, since, v, e = (
            [],
            [],
            np.zeros(len(S)),
            np.zeros(len(S)),
        )  # DPR-style value and eligibility per strategy
        for u in x["dialog"]:
            a = u.get("annotation", {})
            if u["speaker"] == "supporter" and a.get("strategy") in S:
                since.append(S.index(a["strategy"]))
            if u["speaker"] == "seeker" and "feedback" in a:
                r = int(a["feedback"])
                used = np.bincount(since, minlength=len(S)) > 0
                if past:
                    base = np.mean(past)
                    rows.append(
                        {
                            "test": k % 5 == 0,
                            "y": r,
                            "last": past[-1],
                            "mean": base,
                            "n": len(past),
                            "I": I,
                            "used": used.astype(float),
                            "value_of_used": float((v * used).sum() / max(1, used.sum())),
                            "v": v.copy(),
                        }
                    )
                # dopamine-like update
                e = decay * e + used
                rpe = r - (np.mean(past) if past else 4.0)
                v = decay * v + eta * rpe * e
                past.append(r)
                since = []
    return rows


def X_of(rows, arm):
    cols = []
    for r in rows:
        f = [1.0]
        if arm in ("last", "last+strat", "last+strat+dpr", "last+dpr"):
            f += list(np.eye(5)[r["last"] - 1]) + [r["mean"], math.log(r["n"])]
        if arm in ("last+strat", "last+strat+dpr"):
            f += list(r["used"])
        if arm in ("last+dpr", "last+strat+dpr"):
            f += [r["value_of_used"]] + list(r["v"] * r["used"])
        if arm == "intensity":
            f += list(np.eye(5)[r["I"] - 1])
        cols.append(f)
    return np.array(cols, np.float32)


def fit(rows, arm, l2=1e-3):
    tr, te = [r for r in rows if not r["test"]], [r for r in rows if r["test"]]
    Xtr, Xte = X_of(tr, arm), X_of(te, arm)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    mu[0], sd[0] = 0, 1
    Xtr, Xte = torch.tensor((Xtr - mu) / sd), torch.tensor((Xte - mu) / sd)
    ytr = torch.tensor([r["y"] - 1 for r in tr])
    yte = torch.tensor([r["y"] - 1 for r in te])
    torch.manual_seed(0)
    W = torch.zeros(Xtr.shape[1], 5, requires_grad=True)
    opt = torch.optim.Adam([W], lr=0.02)
    for _ in range(500):
        loss = torch.nn.functional.cross_entropy(Xtr @ W, ytr) + l2 * W[1:].square().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        P = torch.softmax(Xte @ W, -1)
        ll = float(P[torch.arange(len(yte)), yte].clamp_min(1e-9).log().mean())
        mae = float(((P * torch.arange(1, 6)).sum(1) - (yte + 1)).abs().mean())
        low = yte <= 2
        recall_low = float((P[low].argmax(1) <= 2).float().mean()) if low.any() else None
    return {
        "loglik": round(ll, 4),
        "mae_expected": round(mae, 3),
        "n_test": len(te),
        "recall_of_low_ratings": recall_low,
    }


def main(a):
    D = json.load(open(a.esconv, encoding="utf-8"))
    R = {}
    for eta, decay in ((0.5, 0.9), (0.3, 0.7)):
        rows = episodes(D, eta, decay)
        for arm in ("none", "intensity", "last", "last+strat", "last+dpr", "last+strat+dpr"):
            R[f"eta{eta}_decay{decay}/{arm}"] = fit(rows, arm)
            print(eta, decay, arm, R[f"eta{eta}_decay{decay}/{arm}"], flush=True)
    json.dump(R, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--esconv", default="../data/esconv/ESConv.json")
    p.add_argument("--out", default="../analysis/audit_r2.json")
    main(p.parse_args())
