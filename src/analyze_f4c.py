"""F4c (MISSIONS_DESIGN_RU.md 6.4, post-hoc).
usage: analyze_f4c.py DIR  (f4_full, f4b_full, f4c_operator, f4c_full, f4d_full .jsonl; F4d = MISSIONS_DESIGN 6.5) -> prints + DIR/F4C_ANALYSIS.json
"""

import collections, json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1])
rng = np.random.default_rng(0)
load = lambda n: (
    {
        r["i"]: r
        for r in (json.loads(l) for l in open(D / f"{n}.jsonl", encoding="utf-8") if l.strip().endswith("}"))
    }
    if (D / f"{n}.jsonl").exists()
    else {}
)  # a line cut by a stopped run is skipped
runs = {n: load(n) for n in ("f4_full", "f4b_full", "f4c_operator", "f4c_full", "f4d_full")}


def boot(x, n=2000):
    x = np.asarray(x, float)
    if not len(x):
        return [None, None, None]
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [
        round(float(x.mean()), 4),
        round(float(np.percentile(b, 2.5)), 4),
        round(float(np.percentile(b, 97.5)), 4),
    ]


def steps(r, skip_enter=False):
    return [
        ok for a, ok in zip(r.get("actions", []), r.get("steps_ok", [])) if not (skip_enter and a == "enter")
    ]


def summary(R):
    by = collections.defaultdict(list)
    for r in R:
        for a, ok in zip(r.get("actions", []), r.get("steps_ok", [])):
            by[a].append(bool(ok))
    return {
        "missions": len(R),
        "step_success": boot([ok for r in R for ok in steps(r)]),
        "step_success_without_enter": boot([ok for r in R for ok in steps(r, True)]),
        "mission_success": boot([all(r.get("steps_ok") or [False]) for r in R]),
        "wrong_target_n": sum(bool(r.get("wrong_target")) for r in R),
        "wrong_target_rate": boot([bool(r.get("wrong_target")) for r in R]),
        "missions_with_static_hit": boot([r.get("hits_static", 0) > 0 for r in R]),
        "asks_per_mission": boot([r.get("asks", 0) or 0 for r in R]),
        "skipped_steps": sum(len(r.get("skipped") or []) for r in R),
        "by_action": {a: [sum(v), len(v), round(sum(v) / len(v), 3)] for a, v in sorted(by.items())},
    }


out = {}
for n, Rd in runs.items():
    R = [r for r in Rd.values() if r.get("status") != "error"]
    if R:
        out[n] = summary(R)
        out[n]["errors"] = len(Rd) - len(R)
for new in ("f4c_operator", "f4c_full", "f4d_full"):
    for old in ("f4_full", "f4b_full"):
        common = [
            i
            for i in sorted(set(runs[new]) & set(runs[old]))
            if runs[new][i].get("status") != "error" and runs[old][i].get("status") != "error"
        ]
        if not common:
            continue
        A, B = [runs[new][i] for i in common], [runs[old][i] for i in common]
        d = lambda key: boot([float(key(a)) - float(key(b)) for a, b in zip(A, B)])
        out[f"{new}_minus_{old}"] = {
            "n": len(common),
            "old_on_same": summary(B),
            "step_success (no enter)": d(lambda r: np.mean(steps(r, True) or [0])),
            "mission_success": d(lambda r: all(r.get("steps_ok") or [False])),
            "wrong_target": d(lambda r: bool(r.get("wrong_target"))),
            "static_hit_missions": d(lambda r: r.get("hits_static", 0) > 0),
        }
(D / "F4C_ANALYSIS.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
for k, v in out.items():
    if "by_action" in v:
        print(
            f"== {k}: n {v['missions']} err {v['errors']} steps {v['step_success']} (no enter {v['step_success_without_enter']}) missions {v['mission_success']} wrong {v['wrong_target_n']} {v['wrong_target_rate']} asks {v['asks_per_mission']}\n   {v['by_action']}"
        )
    else:
        print(
            f"== {k}: n {v['n']} | old on same: steps {v['old_on_same']['step_success']} wrong {v['old_on_same']['wrong_target_n']} by {v['old_on_same']['by_action']}\n   diffs: "
            + ", ".join(f"{a} {b}" for a, b in v.items() if a not in ("n", "old_on_same"))
        )
