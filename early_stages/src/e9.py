"""E9 'remember how to support YOU'."""

import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
import e2, e3, env_v3 as env

PAIR = ("SOLVE", "COMFORT")


class Memory:
    TAU = 0.5

    def __init__(self, kind, dim, seed, n_kc=2000, k=100):
        self.kind = kind
        g = np.random.default_rng(seed)
        self.K, self.D = [], []
        self.P = (g.standard_normal((n_kc, dim)) / np.sqrt(dim)).astype(np.float32)
        self.k = k
        self.V = np.zeros((n_kc if kind in ("mb_sparse", "mb_shuffle") else dim, 2))
        self.T = {}

    def key(self, emb, person):
        if self.kind in ("mb_sparse", "mb_shuffle"):
            h = self.P @ emb
            return np.argpartition(-h, self.k)[: self.k]
        return person if self.kind == "oracle" else emb

    def recall(self, key):
        if self.kind == "none":
            return np.zeros(2)
        if self.kind in ("mb_sparse", "mb_shuffle"):
            return self.V[key].mean(0)
        if self.kind == "mb_dense":
            return key @ self.V
        if self.kind == "knn":
            if not self.K:
                return np.zeros(2)
            m = np.stack(self.K) @ key > self.TAU
            return np.stack(self.D)[m].sum(0)
        return self.T.get(key, np.zeros(2)).copy()

    def write(self, key, d):
        if self.kind in ("mb_sparse", "mb_shuffle"):
            self.V[key] += d
        elif self.kind == "mb_dense":
            self.V += np.outer(key, d)
        elif self.kind == "oracle":
            self.T[key] = self.T.get(key, np.zeros(2)) + d
        elif self.kind == "knn":
            self.K.append(key)
            self.D.append(d)


def load_reader(a):
    e2.ARMS["E9"] = ("prefix", False, None)
    arm = e2.Arm("E9", a.model, None, a.seed)
    from peft import PeftModel

    arm.model = PeftModel.from_pretrained(arm.base, str(Path(a.dwp) / "lora"))
    arm.model.eval()
    arm.module.load_state_dict(torch.load(Path(a.dwp) / "signal_module.pt", map_location="cuda"))
    arm.module.eval()
    return arm


def run_protocol(arm, kind, spk, bank, seed, rounds=6, turns=3, eta=0.8):
    rng = random.Random(seed)
    people = list(range(1, 25))
    emb, actor = spk["emb"], spk["actor"]
    clips = {p: [i for i in range(len(actor)) if actor[i] == p] for p in people}
    pref = {p: rng.choice(PAIR) for p in people}
    changed = set(rng.sample(people, 8))
    mem = Memory(kind, emb.shape[1], seed)
    topics = [i for i in range(len(env.TOPICS)) if i in env.TEST_TOPICS]
    acc = {"first_turn_by_round": [], "all_by_round": [], "changed_after": [], "unchanged_after": []}
    for rd in range(rounds):
        if rd == 3:
            for p in changed:
                pref[p] = e3.other(pref[p])
        S = {
            p: {
                "hist": [],
                "delta": 0.0,
                "elig": np.zeros(2),
                "topic": rng.choice(topics),
                "lang": rng.choice(("en", "ru")),
            }
            for p in people
        }
        first, allc = [], []
        for t in range(turns):
            items, keys, ws = [], [], []
            for p in people:
                c = rng.choice(clips[p])
                src = (
                    c
                    if kind != "mb_shuffle"
                    else rng.choice(clips[rng.choice([q for q in people if q != p])])
                )
                k = mem.key(emb[src], p)
                w = np.clip(mem.recall(k), -1, 1)
                if t > 0:
                    w = S[p]["w"]  # long-term memory primes the session; then the session's working values
                keys.append(k)
                ws.append(w)
                s = S[p]
                msg = (
                    env.TOPICS[s["topic"]][s["lang"] == "ru"] if t == 0 else rng.choice(e3.FOLLOW[s["lang"]])
                )
                items.append(
                    {
                        "topic": s["topic"],
                        "lang": s["lang"],
                        "message": msg,
                        "history": list(s["hist"]),
                        "signals": {k_: 0.0 for k_ in env.SIGNALS},
                        "vec": [s["delta"], float(w[0]), float(w[1]), 0.0],
                    }
                )
            P = arm.strategy_probs(items, strategies=PAIR, chunk=8)
            for j, p in enumerate(people):
                s = S[p]
                a_ = int(P[j].argmax())
                act = PAIR[a_]
                r = 1.0 if act == pref[p] else -1.0
                ok = int(act == pref[p])
                allc.append(ok)
                if t == 0:
                    first.append(ok)
                if rd >= 3:
                    acc["changed_after" if p in changed else "unchanged_after"].append(ok)
                s["elig"] = 0.5 * s["elig"] + np.eye(2)[a_]
                w = ws[j]
                s["delta"] = r - w[a_]
                d = eta * s["delta"] * s["elig"]
                mem.write(keys[j], d)
                s["w"] = np.clip(w + d, -1, 1)
                s["hist"].append((items[j]["message"], rng.choice(bank[(s["topic"], s["lang"], act)])))
        acc["first_turn_by_round"].append(float(np.mean(first)))
        acc["all_by_round"].append(float(np.mean(allc)))
    return {
        "first_turn_by_round": acc["first_turn_by_round"],
        "all_by_round": acc["all_by_round"],
        "accuracy": float(np.mean(acc["all_by_round"])),
        "returning_first_turn_rounds2_3": float(np.mean(acc["first_turn_by_round"][1:3])),
        "changed_people_after_change": float(np.mean(acc["changed_after"])),
        "unchanged_people_after_change": float(np.mean(acc["unchanged_after"])),
    }


def main(a):
    Memory.TAU = a.tau
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    spk = dict(np.load(a.speaker))
    replies = json.loads(Path(a.replies).read_text(encoding="utf-8"))
    bank = {}
    for r in replies:
        bank.setdefault((r["topic"], r["lang"], r["strategy"]), []).append(r["reply"])
    arm = load_reader(a)
    R = {"protocol_seeds": a.seeds, "tau": a.tau, "reader": a.dwp, "kinds": {}}
    with torch.no_grad():
        for kind in a.kinds:
            runs = [run_protocol(arm, kind, spk, bank, s) for s in a.seeds]
            R["kinds"][kind] = {
                k: (
                    np.mean([r[k] for r in runs], 0).tolist()
                    if isinstance(runs[0][k], list)
                    else float(np.mean([r[k] for r in runs]))
                )
                for k in runs[0]
            } | {"per_seed": runs}
            print(
                kind,
                json.dumps({k: v for k, v in R["kinds"][kind].items() if k != "per_seed"}),
                round(time.time() - t0),
                "s",
                flush=True,
            )
    R["seconds"] = time.time() - t0
    (out / "RESULT.json").write_text(json.dumps(R, indent=1), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("model", "dwp", "speaker", "replies", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--kinds", nargs="+", default=["none", "mb_sparse", "mb_dense", "mb_shuffle", "oracle"])
    p.add_argument("--tau", type=float, default=Memory.TAU)
    main(p.parse_args())
