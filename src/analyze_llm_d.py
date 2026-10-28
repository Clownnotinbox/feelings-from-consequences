"""Stage D: the LLM conditions across seeds, paired by decision, and H8, the core decides by itself.
usage: analyze_llm_d.py RESULTS.json DATA_DIR CORE_RUNTIME.npz [--core-model gru] [--feelings CACHE.json]"""

import argparse, json
from pathlib import Path
import numpy as np

import llm_decide as L

SPLITS = ("test_familiar", "test_unfamiliar")
PAIRS = (
    ("TM", "T"),
    ("FTM", "FT"),
    ("KM", "K"),
    ("KM", "FTM"),
    ("T", "N"),
    ("F", "N"),
    ("FT", "N"),
    ("FT", "T"),
    ("Fs", "N"),
    ("Ts", "N"),
    ("Fs", "Ts"),
    ("Fs", "F"),
    ("FsT", "T"),
    ("FsT", "FT"),
)


def boot(x, n=10000, seed=0):
    r = np.random.default_rng(seed)
    m = r.integers(0, len(x), (n, len(x)))
    return [
        round(float(x.mean()), 4),
        round(float(np.quantile(x[m].mean(1), 0.025)), 4),
        round(float(np.quantile(x[m].mean(1), 0.975)), 4),
    ]


def hits(D, picks):
    return np.array([L.OPTIONS[int(c)] in d["best"] for c, d in zip(picks, D)], float)


def msg_status(d):
    """Stage D2: was a message about the future world in hand, and had the announced change already."""
    m = d.get("msg_meta")
    if not m:
        return "no_msg"
    return ("pending_" if d["snapshot"]["t"] < m["t_event"] else "after_") + (
        "true" if m["true"] else "false"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("data")
    ap.add_argument("core")
    ap.add_argument("--core-model", default="gru")
    ap.add_argument("--feelings")
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text(encoding="utf-8"))
    S = {
        s: L.load(a.data, s, with_x=not a.feelings)[: R["n"][s]] for s in ("train",) + SPLITS
    }  # [:n]: the same decisions as the run (a smoke run uses a prefix)
    if a.feelings:
        cache = json.loads(Path(a.feelings).read_text(encoding="utf-8"))["F"]
        for D in S.values():
            L.attach_cached(D, cache)
    else:
        import core_runtime

        core = (core_runtime.FeelingCore if a.core_model == "gru" else core_runtime.LogregCore)(a.core)
        for D in S.values():
            L.attach_feelings(D, core)
    ORDER = ("N", "T", "F", "FT", "Fs", "Ts", "FsT", "TM", "FTM", "K", "KM")
    conds = sorted(
        {k.split("|")[0] for k in R if "|seed" in k}, key=lambda c: ORDER.index(c) if c in ORDER else 99
    )
    seeds = lambda c: sorted(k for k in R if k.startswith(c + "|seed"))
    out = {"n": R.get("n"), "conditions": {}, "pairs": {}, "lesions": {}, "h8": {}}
    H = {}  # (cond, split, lesion) -> decisions x seeds hits
    for c in conds:
        for sp in SPLITS:
            rows = {}
            for key in seeds(c):
                for name, r in R[key].items():
                    if name.split("|")[0] == sp:
                        rows.setdefault(name.split("|")[1] if "|" in name else "", []).append(r)
            for les, rs in rows.items():
                H[(c, sp, les)] = np.stack([hits(S[sp], r["picks"]) for r in rs], 1)
            rs = rows.get("", [])
            if rs:
                out["conditions"][f"{c}|{sp}"] = {
                    k: [
                        round(float(np.mean([r[k] for r in rs])), 4),
                        round(float(np.std([r[k] for r in rs])), 4),
                    ]
                    for k in ("best", "safe", "legs")
                }
                out["conditions"][f"{c}|{sp}"]["seeds"] = len(rs)
                out["conditions"][f"{c}|{sp}"]["epochs"] = [R[k]["fit"]["chosen_epoch"] for k in seeds(c)]
    for sp in SPLITS:
        for x, y in PAIRS:
            if (x, sp, "") in H and (y, sp, "") in H:
                out["pairs"][f"{x}-{y}|{sp}"] = boot(H[(x, sp, "")].mean(1) - H[(y, sp, "")].mean(1))
        for (c, s2, les), h in H.items():
            if s2 == sp and les:
                out["lesions"][f"{c}|{sp}|{les}"] = boot(h.mean(1) - H[(c, sp, "")].mean(1))
    for fc in ("F", "FT"):  # H8: the core (a small policy on its feelings) decides
        for sp in SPLITS:
            P = L.small_probs(S["train"], S[sp], fc)
            own = hits(S[sp], "".join(str(i) for i in P.argmax(1)))
            order = np.argsort(P.max(1))
            for c in conds:
                if (c, sp, "") not in H:
                    continue
                llm = H[(c, sp, "")].mean(1)
                curve = []
                for q in (0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
                    esc = np.zeros(len(own), bool)
                    esc[order[: int(round(q * len(own)))]] = True
                    curve.append([q, round(float(np.where(esc, llm, own).mean()), 4)])
                out["h8"][f"core_{fc}->llm_{c}|{sp}"] = curve
    out["by_message"] = {}  # stage D2 (H12): best rate by message status
    for sp in SPLITS:
        st = np.array([msg_status(d) for d in S[sp]])
        if (st == "no_msg").all():
            continue
        core = hits(S[sp], "".join(str(i) for i in L.small_probs(S["train"], S[sp], "FT").argmax(1)))
        rows = {"core_alone_FT": core} | {c: H[(c, sp, "")].mean(1) for c in conds if (c, sp, "") in H}
        for name, h in rows.items():
            out["by_message"][f"{name}|{sp}"] = {
                k: [round(float(h[st == k].mean()), 4), int((st == k).sum())] for k in sorted(set(st))
            }
    print(json.dumps(out["by_message"], indent=0))
    print(json.dumps(out["conditions"], indent=0))
    print(json.dumps(out["pairs"], indent=0))
    print(json.dumps(out["lesions"], indent=0))
    print(json.dumps(out["h8"], indent=0))
    Path(a.results).with_name(Path(a.results).stem + "_ANALYSIS.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
