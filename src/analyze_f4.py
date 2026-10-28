"""F4/F5 analysis by the recorded measures.
usage: analyze_f4.py DIR [PREFIX=f4]  (f4_plan.jsonl + PREFIX_full/reflex/telemetry.jsonl) -> prints + DIR/PREFIX_ANALYSIS.json
"""

import collections, json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1])
PRE = sys.argv[2] if len(sys.argv) > 2 else "f4"
rng = np.random.default_rng(0)
load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []
plan = {r["i"]: r for r in load(D / "f4_plan.jsonl")}
runs = {v: {r["i"]: r for r in load(D / f"{PRE}_{v}.jsonl")} for v in ("full", "reflex", "telemetry")}


def boot(x, f=np.mean, n=2000):
    x = np.asarray(x, float)
    if not len(x):
        return [None, None, None]
    b = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return [
        round(float(f(x)), 4),
        round(float(np.percentile(b, 2.5)), 4),
        round(float(np.percentile(b, 97.5)), 4),
    ]


def summary(R):
    R = [r for r in R if r.get("status") != "error"]
    steps = [(r, a, ok) for r in R for a, ok in zip(r.get("actions", []), r.get("steps_ok", []))]
    by = collections.defaultdict(list)
    for r, a, ok in steps:
        by[a].append(bool(ok))
    return {
        "missions": len(R),
        "errors": None,
        "steps": len(steps),
        "step_success": boot([bool(ok) for _, _, ok in steps]),
        "by_action": {a: [sum(v), len(v), round(sum(v) / len(v), 3)] for a, v in sorted(by.items())},
        "mission_success": boot([all(r.get("steps_ok") or [False]) for r in R]),
        "wrong_target_rate": boot([bool(r.get("wrong_target")) for r in R]),
        "wrong_target_n": sum(bool(r.get("wrong_target")) for r in R),
        "static_hits_per_mission": boot([r.get("hits_static", 0) for r in R]),
        "missions_with_static_hit": boot([r.get("hits_static", 0) > 0 for r in R]),
        "attack_hits_per_attacked_mission": boot([r.get("hits_attack", 0) for r in R if r.get("attacks")]),
        "bird_hits": sum(r.get("hits_bird", 0) for r in R),
        "no_parse": sum(r.get("status") == "no_parse" for r in R),
        "skipped_steps": sum(len(r.get("skipped") or []) for r in R),
        "nav_err_mean": round(
            float(np.mean([r["nav_err"] for r in R if r.get("nav_err") is not None] or [0])), 2
        ),
    }


out = {
    "n_plan": len(plan),
    "parse_exact": round(float(np.mean([p["parse_exact"] for p in plan.values()])), 3) if plan else None,
}
for v, Rd in runs.items():
    if not Rd:
        continue
    R = list(Rd.values())
    s = summary(R)
    s["errors"] = sum(r.get("status") == "error" for r in R)
    out[v] = s
    out[v + "_slices"] = {
        name: summary([r for r in R if f(r)])
        for name, f in (
            ("gps_off", lambda r: bool(r.get("gps_off"))),
            ("gps_on", lambda r: not r.get("gps_off")),
            ("attacks", lambda r: bool(r.get("attacks"))),
            ("no_attacks", lambda r: not r.get("attacks")),
            ("parse_exact", lambda r: bool(r.get("parse_exact"))),
            ("parse_wrong", lambda r: not r.get("parse_exact")),
        )
    }
# F5: paired differences over the same missions (full minus lesion)
for les in ("reflex", "telemetry"):
    common = sorted(set(runs["full"]) & set(runs[les]))
    common = [
        i
        for i in common
        if runs["full"][i].get("status") != "error" and runs[les][i].get("status") != "error"
    ]
    if not common:
        continue
    F, L = [runs["full"][i] for i in common], [runs[les][i] for i in common]
    d = lambda key, sel=lambda r: True: boot([float(key(a)) - float(key(b)) for a, b in zip(F, L) if sel(a)])
    out[f"F5_full_minus_{les}"] = {
        "n": len(common),
        "step_success": d(lambda r: np.mean(r.get("steps_ok") or [0])),
        "static_hits": d(lambda r: r.get("hits_static", 0)),
        "attack_hits (attacked missions)": d(lambda r: r.get("hits_attack", 0), lambda r: r.get("attacks")),
        "step_success (GPS off)": d(
            lambda r: np.mean(r.get("steps_ok") or [0]), lambda r: bool(r.get("gps_off"))
        ),
        "wrong_target": d(lambda r: bool(r.get("wrong_target"))),
    }
out["F4_rule_wrong_target<=2%"] = (
    out.get("full", {}).get("wrong_target_rate", [None])[0] is not None
    and out["full"]["wrong_target_rate"][0] <= 0.02
)
(D / f"{PRE.upper()}_ANALYSIS.json").write_text(
    json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8"
)
for k, v in out.items():
    if isinstance(v, dict) and "by_action" in v:
        print(
            f"== {k}: missions {v['missions']} errors {v['errors']} steps ok {v['step_success']} missions ok {v['mission_success']} wrong {v['wrong_target_n']} {v['wrong_target_rate']} static-hit missions {v['missions_with_static_hit']} attack hits {v['attack_hits_per_attacked_mission']} skipped {v['skipped_steps']} no_parse {v['no_parse']}\n   by action {v['by_action']}"
        )
    elif k.startswith("F5") or not isinstance(v, dict):
        print(k, v)
for v in ("full", "reflex", "telemetry"):
    if v + "_slices" in out:
        print(v, "slices:", {n: s["step_success"] for n, s in out[v + "_slices"].items()})
