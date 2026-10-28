"""R2' CPU check on PRISM."""

import argparse, collections, json
import numpy as np


def load(path):
    by_user = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        c = json.loads(line)
        turns = collections.defaultdict(list)
        for h in c["conversation_history"]:
            if h["role"] == "model" and h.get("score") not in (None, ""):
                turns[int(h["turn"])].append(
                    {
                        "model": h["model_name"],
                        "score": float(h["score"]),
                        "chosen": str(h.get("if_chosen")) == "True",
                        "len": len(h["content"]),
                    }
                )
        by_user[c["user_id"]].append((c["generated_datetime"], [turns[t] for t in sorted(turns)]))
    return {u: [t for _, ts in sorted(cs) for t in ts] for u, cs in by_user.items()}


def main(a):
    U = load(a.conversations)
    users = sorted(U)
    test = set(users[4::5])
    pop = collections.defaultdict(list)
    for u in users:
        if u in test:
            continue
        for turn in U[u]:
            m = np.mean([r["score"] for r in turn])
            for r in turn:
                pop[r["model"]].append(r["score"] - m)
    popv = {k: float(np.mean(v)) for k, v in pop.items()}
    R = {}
    for eta, decay in ((0.3, 1.0), (0.3, 0.9), (0.1, 1.0)):
        hits = collections.Counter()
        n = 0
        for u in test:
            v = collections.defaultdict(float)
            seen = collections.Counter()
            for turn in U[u]:
                ch = [r for r in turn if r["chosen"]]
                if len(turn) >= 2 and len(ch) == 1:
                    c = ch[0]
                    others = [r for r in turn if r is not c]
                    for o in others:
                        n += 1
                        hits["population"] += (popv.get(c["model"], 0) > popv.get(o["model"], 0)) + 0.5 * (
                            popv.get(c["model"], 0) == popv.get(o["model"], 0)
                        )
                        hits["personal"] += (v[c["model"]] > v[o["model"]]) + 0.5 * (
                            v[c["model"]] == v[o["model"]]
                        )
                        comb = lambda r: popv.get(r["model"], 0) + v[r["model"]]
                        hits["population+personal"] += (comb(c) > comb(o)) + 0.5 * (comb(c) == comb(o))
                        hits["longer"] += (c["len"] > o["len"]) + 0.5 * (c["len"] == o["len"])
                        known = seen[c["model"]] > 0 and seen[o["model"]] > 0
                        if known:
                            hits["n_both_seen"] += 1
                            hits["personal_when_seen"] += (v[c["model"]] > v[o["model"]]) + 0.5 * (
                                v[c["model"]] == v[o["model"]]
                            )
                            hits["population_when_seen"] += (
                                popv.get(c["model"], 0) > popv.get(o["model"], 0)
                            ) + 0.5 * (popv.get(c["model"], 0) == popv.get(o["model"], 0))
                # dopamine-like update after the person's reaction
                m = np.mean([r["score"] for r in turn])
                for k in v:
                    v[k] *= decay
                for r in turn:
                    v[r["model"]] += eta * ((r["score"] - m) - v[r["model"]])
                    seen[r["model"]] += 1
        R[f"eta{eta}_decay{decay}"] = {
            k: round(hits[k] / n, 4) for k in ("population", "personal", "population+personal", "longer")
        }
        R[f"eta{eta}_decay{decay}"].update(
            n_pairs=n,
            n_both_seen=hits["n_both_seen"],
            personal_when_seen=round(hits["personal_when_seen"] / max(1, hits["n_both_seen"]), 4),
            population_when_seen=round(hits["population_when_seen"] / max(1, hits["n_both_seen"]), 4),
        )
        print(eta, decay, R[f"eta{eta}_decay{decay}"], flush=True)
    R["n_users"] = len(users)
    R["n_test_users"] = len(test)
    json.dump(R, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--conversations", default="../../../work/datasets/prism/conversations.jsonl")
    p.add_argument("--out", default="../analysis/audit_prism.json")
    main(p.parse_args())
