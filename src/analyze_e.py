"""Stage E, step 1 analysis.
usage: analyze_e.py SCENE_DIR VLM.json[,VLM2.json] EMBED.json OUT.json  (a condition repeated in a later file is kept as '<mode>_rerun|w')
"""

import json, math, re, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import stage_e_scenes as E

TYPES = {"landing pad": "pad", "water tower": "tower", "tent": "tent", "container": "container"}


def parse_familiar(text):
    """Constraints a fixed-vocabulary parser reads from the instruction (attribute -> value)."""
    t = re.sub(r"^(Fly to|Go to|The goal is|Land on|Hover over)\s+", "", text.strip()).rstrip(".")
    C = {}
    m = re.search(r" next to | in the open| (north|south|east|west) of ", t)
    head = t[: m.start()] if m else t
    rel = t[len(head) :]
    for ph, s in TYPES.items():
        if re.search(r"\b" + ph + r"\b", head):
            C["type"] = s
    for w in head.split():
        if w in E.COLORS:
            C["colour"] = w
        if w in ("large", "small"):
            C["size"] = w
    if re.search(r"in the open", rel):
        C["bctx"] = "open"
    m = re.search(r"next to a (tall|low) building", rel)
    if m:
        C["bctx"] = "near_" + m.group(1)
    m = re.search(r"(north|south|east|west) of a (tall|low) building", rel)
    if m:
        C["side"] = (m.group(1), m.group(2))
    m = re.search(r"next to the (\w+) (landing pad|water tower|tent|container)", rel)
    if m:
        C["next_to"] = (m.group(1), TYPES[m.group(2)])
    return C


def baselines(s, emb):
    W = SimpleNamespace(buildings=s["buildings"])
    cands = s["cands"]
    for c in cands:
        c["pos"] = tuple(c["pos"])
    vis = [j for j, c in enumerate(cands) if c["area"] >= 40]
    ax, ay = s["approx"]
    near = lambda js: (
        min(js, key=lambda j: math.hypot(cands[j]["pos"][0] - ax, cands[j]["pos"][1] - ay)) if js else "none"
    )
    out = {"coord": near(vis), "chance": 1 / len(vis) if (vis and s["target"] in vis) else 0.0}
    for w in ("familiar", "novel"):
        C = parse_familiar(s["text_" + w])
        keep = [j for j in vis if all(E.holds(cands[j], a, v, cands, W) is not False for a, v in C.items())]
        out["keyword|" + w] = near(keep) if C else near(vis)
        out["parsed|" + w] = len(C)
        sc = emb[s["name"]][w] if emb else []
        out["embed|" + w] = vis[int(np.argmax(sc))] if len(sc) else "none"
    return out


def boot(x, y=None, n=4000, seed=0):
    x = np.asarray(x, float)
    d = x if y is None else x - np.asarray(y, float)
    if not len(d):
        return None
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(len(d), size=(n, len(d)))].mean(1)
    return [
        round(float(d.mean()), 3),
        round(float(np.percentile(bs, 2.5)), 3),
        round(float(np.percentile(bs, 97.5)), 3),
        int(len(d)),
    ]


def main():
    D, vf, ef, of = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    S = [json.loads(l) for l in open(D / "scenes.jsonl", encoding="utf-8")]
    V = {"conds": {}}
    for f in vf.split(","):
        for c, r in (
            json.loads(Path(f).read_text(encoding="utf-8"))["conds"] if Path(f).exists() else {}
        ).items():
            V["conds"][c if c not in V["conds"] else c.replace("|", "_rerun|")] = r
    emb = json.loads(Path(ef).read_text(encoding="utf-8"))["scores"] if Path(ef).exists() else None
    picks = {}
    rows = []
    for s in S:
        b = baselines(s, emb)
        p = {k: v for k, v in b.items() if not k.startswith(("chance", "parsed"))}
        for c, r in V["conds"].items():
            if s["name"] in r["answers"]:
                p[c] = r["answers"][s["name"]]["pick"]
        ok = {
            k: float(v == s["target"]) if s["status"] == "visible" else float(v == "none")
            for k, v in p.items()
        }
        ok["chance"] = b["chance"] if s["status"] == "visible" else 0.0
        a = s["target_area"]
        rel = [k for k in ("size", "bctx", "next_to", "side") if k in s["used"]]
        rows.append(
            {
                "name": s["name"],
                "status": s["status"],
                "hard": s["hard"],
                "area": a,
                "rel": rel,
                "ok": ok,
                "pick": p,
                "parsed": (b["parsed|familiar"], b["parsed|novel"]),
            }
        )
    keys = sorted({k for r in rows for k in r["ok"]})
    res = {"n": {st: sum(r["status"] == st for r in rows) for st in ("visible", "absent", "marginal")}}
    sel = lambda f: [r for r in rows if f(r)]
    groups = {
        "visible": sel(lambda r: r["status"] == "visible"),
        "visible_hard": sel(lambda r: r["status"] == "visible" and r["hard"]),
        "visible_easy": sel(lambda r: r["status"] == "visible" and not r["hard"]),
        "absent (answer none)": sel(lambda r: r["status"] == "absent"),
    }
    for k in ("size", "bctx", "next_to", "side"):
        groups[f"visible_hard_uses_{k}"] = sel(
            lambda r, k=k: r["status"] == "visible" and r["hard"] and k in r["rel"]
        )
    for lo, hi in ((40, 150), (150, 400), (400, 10**9)):
        groups[f"visible_area_{lo}_{hi}"] = sel(
            lambda r, lo=lo, hi=hi: r["status"] == "visible" and lo <= r["area"] < hi
        )
    res["accuracy"] = {
        g: {k: boot([r["ok"][k] for r in R if k in r["ok"]]) for k in keys} for g, R in groups.items()
    }
    H = groups["visible_hard"]
    diff = lambda a, b, R=H: boot(
        [r["ok"][a] for r in R if a in r["ok"] and b in r["ok"]],
        [r["ok"][b] for r in R if a in r["ok"] and b in r["ok"]],
    )
    res["H17 point - coord (visible, hard)"] = {w: diff(f"point|{w}", "coord") for w in ("familiar", "novel")}
    res["H18 point - keyword (visible, hard)"] = {
        w: diff(f"point|{w}", f"keyword|{w}") for w in ("familiar", "novel")
    }
    res["point - embed (visible, hard)"] = {
        w: diff(f"point|{w}", f"embed|{w}") for w in ("familiar", "novel")
    }
    res["som - point (visible, hard)"] = {w: diff(f"som|{w}", f"point|{w}") for w in ("familiar", "novel")}
    res["novel - familiar (visible, hard)"] = {
        m: diff(f"{m}|novel", f"{m}|familiar") for m in ("point", "som", "keyword", "embed")
    }
    res["miss/none/fail shares (visible)"] = {
        c: {
            v: round(float(np.mean([r["pick"].get(c) == v for r in groups["visible"]])), 3)
            for v in ("miss", "none", "fail")
        }
        for c in V["conds"]
    }
    res["parser read something"] = {
        w: round(float(np.mean([r["parsed"][i] > 0 for r in rows])), 3)
        for i, w in enumerate(("familiar", "novel"))
    }
    for fam in ("point", "zoom"):  # the rule for going on to step 2 (closed loop)
        if f"{fam}|novel" not in keys:
            continue
        acc = res["accuracy"]["visible_hard"][f"{fam}|novel"]
        riv = {
            k: res["accuracy"]["visible_hard"][k][0]
            for k in ("coord", "keyword|novel", "embed|novel")
            if k in keys
        }
        best = max(riv, key=riv.get)
        d = diff(f"{fam}|novel", best)
        res["go_rule" + ("" if fam == "point" else " (zoom)")] = {
            f"{fam}|novel": acc,
            "best rival": best,
            "diff": d,
            "GO": bool(acc[0] >= 0.6 and d[0] >= 0.15 and d[1] > 0),
        }
    if "zoom|novel" in keys:  # sec. 13.3: H20 zoom vs marks without zoom, H21 zoom vs coordinates
        res["H20 zoom - som rerun on the same GPU (visible, hard)"] = {
            w: diff(f"zoom|{w}", f"som_rerun|{w}") for w in ("familiar", "novel") if f"som_rerun|{w}" in keys
        }
        res["H20b zoom - som of step 1 (visible, hard)"] = {
            w: diff(f"zoom|{w}", f"som|{w}") for w in ("familiar", "novel") if f"som|{w}" in keys
        }
        res["H21 zoom - coord (visible, hard)"] = {
            w: diff(f"zoom|{w}", "coord") for w in ("familiar", "novel") if f"zoom|{w}" in keys
        }
        res["zoom: novel - familiar (visible, hard)"] = (
            diff("zoom|novel", "zoom|familiar") if "zoom|familiar" in keys else None
        )
    for k in keys:  # reproducibility: the same condition run twice
        if "_rerun|" in k:
            res[f'agreement {k.replace("_rerun", "")} vs rerun (all scenes)'] = round(
                float(
                    np.mean(
                        [r["pick"][k] == r["pick"][k.replace("_rerun", "")] for r in rows if k in r["pick"]]
                    )
                ),
                3,
            )
    res["seconds_per_query"] = {c: r["seconds_per_query"] for c, r in V["conds"].items()}
    res["cut off at max_new tokens (all scenes)"] = {c: r.get("cut_off") for c, r in V["conds"].items()}
    of.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
