"""Stage D3 analysis (FEELINGS_DESIGN_RU.md 12.6, H13-H16).
usage: analyze_d3.py RUN_DIR (with stageD3/, stageD3_feelings_core4_s1_gru.json, read7b.json, LLM_D3_s*.json)
"""

import glob, json, re, sys, zlib
from pathlib import Path
import numpy as np
import llm_decide as L

SPLITS = ("test_familiar", "test_unfamiliar")
KINDS = ("wind", "cold", "gps_outage")


def status(d):
    m = d["msg_meta"]
    if m is None:
        return "no_msg"
    if d["snapshot"]["t"] < m["t_event"]:
        return "pending_true" if m["true"] else "pending_false"
    return "after_true" if m["true"] else "after_false"


def mfeat(d, p):
    """Numbers about the message for the policies."""
    m = d.get("message")
    nan = float("nan")
    if m is None:
        return [0.0, nan, 0.0, 0.0, 0.0, nan, nan, nan]
    age = m["age_s"]
    if not p:
        return [1.0, age, 0.0, 0.0, 0.0, nan, nan, nan]
    g = lambda k: p.get(k) if isinstance(p.get(k), (int, float)) else nan
    return (
        [1.0, age]
        + [float(p.get("kind") == k) for k in KINDS]
        + [g("wind_mps"), g("temp_c"), g("eta_s") - age]
    )


class Keyword:
    """A small text model fitted on the training (familiar) wording only."""

    def __init__(self, train_texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.v = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X = self.v.fit_transform(train_texts)
        self.s = TruncatedSVD(12, random_state=0).fit(X)

    def __call__(self, d, text):
        m = d.get("message")
        nan = float("nan")
        if m is None:
            return [0.0, nan] + [0.0] * 12 + [nan] * 3
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)][:3]
        return (
            [1.0, m["age_s"]]
            + list(self.s.transform(self.v.transform([text]))[0])
            + nums
            + [nan] * (3 - len(nums))
        )


def probs(train, test, ftr, fte, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.array([ftr(d) for d in train], float)
    Xt = np.array([fte(d) for d in test], float)
    P = []
    for o in L.OPTIONS:
        y = np.array([o in d["best"] for d in train])
        P.append(
            HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.06, max_leaf_nodes=15, random_state=seed
            )
            .fit(X, y)
            .predict_proba(Xt)[:, 1]
            if 0 < y.mean() < 1
            else np.full(len(test), float(y.mean()))
        )
    return np.stack(P, 1)


def hits(D, picks):
    return np.array([p in d["best"] for d, p in zip(D, picks)], float)


def boot(D, a, b=None, mask=None, n=2000):
    """Mean of a (or a - b) over decisions, 95 % interval by resampling whole flights."""
    x = a if b is None else a - b
    idx = np.arange(len(D)) if mask is None else np.flatnonzero(mask)
    if not len(idx):
        return None
    fl = {}
    for i in idx:
        fl.setdefault(D[i]["name"], []).append(i)
    G = list(fl.values())
    rng = np.random.default_rng(0)
    S = [x[g].sum() for g in G]
    C = [len(g) for g in G]
    S, C = np.array(S), np.array(C)
    bs = [S[k].sum() / C[k].sum() for k in (rng.integers(len(G), size=len(G)) for _ in range(n))]
    return [
        round(float(S.sum() / C.sum()), 3),
        round(float(np.percentile(bs, 2.5)), 3),
        round(float(np.percentile(bs, 97.5)), 3),
        int(C.sum()),
    ]


def main():
    R0 = Path(sys.argv[1])
    S = {s: L.load(str(R0 / "stageD3"), s, with_x=False) for s in ("train",) + SPLITS}
    F = json.loads((R0 / "stageD3_feelings_core4_s1_gru.json").read_text(encoding="utf-8"))["F"]
    for D in S.values():
        L.attach_cached(D, F)
    rd = (
        json.loads((R0 / "read7b.json").read_text(encoding="utf-8"))["read"]
        if (R0 / "read7b.json").exists()
        else None
    )
    tr = S["train"]
    ft = lambda d: list(L.vec(d, "FT"))
    res = {"n": {k: len(v) for k, v in S.items()}}
    kw = Keyword([d["message"]["text"] for d in tr if d.get("message")])
    txt = lambda d, w: (
        (d["message"]["text_novel"] if w == "novel" else d["message"]["text"]) if d.get("message") else ""
    )
    ideal = lambda d: ft(d) + mfeat(d, d["msg_meta"]["parse"] if d.get("message") else None)
    readp = lambda d, w: rd.get(d["name"], {}).get(w) if (rd and d.get("message")) else None
    picks = {}
    for s in SPLITS:
        D = S[s]
        P = {"core": probs(tr, D, ft, ft), "ideal": probs(tr, D, ideal, ideal)}
        for w in ("familiar", "novel"):
            P[f"keyword|{w}"] = probs(
                tr, D, lambda d: ft(d) + kw(d, txt(d, "familiar")), lambda d, w=w: ft(d) + kw(d, txt(d, w))
            )
            if rd:
                P[f"reader|{w}"] = probs(
                    tr,
                    D,
                    lambda d: ft(d) + mfeat(d, readp(d, "familiar")),
                    lambda d, w=w: ft(d) + mfeat(d, readp(d, w)),
                )
        picks[s] = {k: [L.OPTIONS[i] for i in p.argmax(1)] for k, p in P.items()}
        picks[s]["_Kmax"] = P["core"].max(1)
    runs = []  # the LLM deciders: (tag prefix, results of one seed)
    for f in sorted(glob.glob(str(R0 / "LLM_D3_s*.json"))):
        seed = re.search(r"_s(\d+)\.json$", f).group(1)
        runs.append(("KM", json.loads(Path(f).read_text()).get("KM|seed" + seed, {})))
    if (R0 / "LLM_D3_zs7b.json").exists():
        runs.append(("KM7b0", json.loads((R0 / "LLM_D3_zs7b.json").read_text()).get("KM|seed1", {})))
    for s in SPLITS:  # per decision: the share of seeds that picked a best option
        for pre in ("KM", "KM7b0"):
            for tag, les in (
                ("familiar", ""),
                ("novel", "|novel_text"),
                ("no_message", "|drop_message"),
                ("shuffled_core", "|shuffle_K"),
            ):
                H = [
                    hits(S[s], [L.OPTIONS[int(c)] for c in r[s + les]["picks"]])
                    for p, r in runs
                    if p == pre and s + les in r
                ]
                if H:
                    picks[s][f"{pre}|{tag}"] = np.mean(H, 0)
    for s in SPLITS:
        D = S[s]
        st = np.array([status(d) for d in D])
        pend = np.isin(st, ["pending_true", "pending_false"])
        h = {
            k: (v if isinstance(v, np.ndarray) else hits(D, v))
            for k, v in picks[s].items()
            if not k.startswith("_")
        }
        r = {
            "status_n": {k: int((st == k).sum()) for k in np.unique(st)},
            "best": {},
            "vs_core_pending": {},
            "vs_core_no_msg": {},
        }
        for k, x in h.items():
            r["best"][k] = {
                "all": boot(D, x),
                "pending": boot(D, x, mask=pend),
                "pending_true": boot(D, x, mask=st == "pending_true"),
                "pending_false": boot(D, x, mask=st == "pending_false"),
                "no_msg": boot(D, x, mask=st == "no_msg"),
            }
            if k != "core":
                r["vs_core_pending"][k] = boot(D, x, h["core"], pend)
                r["vs_core_no_msg"][k] = boot(D, x, h["core"], st == "no_msg")
        pairs = [
            ("reader|novel", "keyword|novel"),
            ("reader|familiar", "keyword|familiar"),
            ("KM|novel", "keyword|novel"),
            ("KM|familiar", "KM|no_message"),
            ("KM|familiar", "KM|shuffled_core"),
            ("KM|novel", "KM|familiar"),
            ("reader|novel", "reader|familiar"),
            ("ideal", "reader|familiar"),
        ]
        r["pairs_pending"] = {f"{a} - {b}": boot(D, h[a], h[b], pend) for a, b in pairs if a in h and b in h}
        r["pairs_no_msg"] = {
            f"{a} - {b}": boot(D, h[a], h[b], st == "no_msg") for a, b in pairs if a in h and b in h
        }
        res[s] = r
    if all("KM|familiar" in picks[s] for s in SPLITS):
        res["gates"] = gates(S, picks)
    out = R0 / "D3_ANALYSIS.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def gates(S, picks):
    """H16: call the LLM decider (KM, familiar wording) or let the core decide alone."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    D = [d for s in SPLITS for d in S[s]]
    core = np.concatenate([hits(S[s], picks[s]["core"]) for s in SPLITS])
    llm = np.concatenate([picks[s]["KM|familiar"] for s in SPLITS])
    kmax = np.concatenate([picks[s]["_Kmax"] for s in SPLITS])
    X = np.array(
        [
            [
                d.get("message") is not None,
                (d.get("message") or {}).get("age_s", np.nan),
                km,
                d["snapshot"]["battery_pct"],
                d["snapshot"]["t"],
                d["snapshot"]["legs_left"],
            ]
            for d, km in zip(D, kmax)
        ],
        float,
    )
    fold = np.array([zlib.crc32(f"{d['split']}/{d['name']}/gate".encode()) % 2 for d in D])
    gain = np.zeros(len(D))
    for k in (0, 1):
        m = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=7, random_state=0
        ).fit(X[fold != k], (llm - core)[fold != k])
        gain[fold == k] = m.predict(X[fold == k])

    def use(call):
        x = np.where(call, llm, core)
        return boot(D, x) + [round(float(call.mean()), 3)]

    out = {
        "note": "value = [best rate, 95% lo, hi, n decisions, share of decisions that call the LLM]",
        "never (core alone)": use(np.zeros(len(D), bool)),
        "always": use(np.ones(len(D), bool)),
        "on any message": use(X[:, 0] > 0),
        "learned gate (predicted gain > 0)": use(gain > 0),
    }
    for q in (0.1, 0.2, 0.3, 0.5):
        out[f"core least sure, {int(q * 100)} %"] = use(kmax <= np.quantile(kmax, q))
        out[f"learned gate, top {int(q * 100)} %"] = use(gain >= np.quantile(gain, 1 - q))
    return out


if __name__ == "__main__":
    main()
