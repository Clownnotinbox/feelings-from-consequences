"""Stage G (MISSIONS_DESIGN_RU.md 6.6).
usage: analyze_g.py DIR -> prints + DIR/G_ANALYSIS.json"""

import collections, json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1])
rng = np.random.default_rng(0)


def rows(name):
    p = D / name
    if not p.exists():
        return []
    return [
        json.loads(l) for l in open(p, encoding="utf-8") if l.strip().endswith("}")
    ]  # a line cut by a stopped run is skipped


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
        bool(ok)
        for a, ok in zip(r.get("actions", []), r.get("steps_ok", []))
        if not (skip_enter and a == "enter")
    ]


def summary(R):
    by = collections.defaultdict(list)
    for r in R:
        for a, ok in zip(r.get("actions", []), r.get("steps_ok", [])):
            by[a].append(bool(ok))
    out = {
        "missions": len(R),
        "step_success": boot([ok for r in R for ok in steps(r)]),
        "mission_success": boot([all(r.get("steps_ok") or [False]) for r in R]),
        "wrong_target_n": sum(bool(r.get("wrong_target")) for r in R),
        "wrong_target_rate": boot([bool(r.get("wrong_target")) for r in R]),
        "missions_with_static_hit": boot([r.get("hits_static", 0) > 0 for r in R]),
        "static_hits_per_mission": boot([r.get("hits_static", 0) for r in R]),
        "attack_hits_per_attacked": boot([r.get("hits_attack", 0) for r in R if r.get("attacks")]),
        "skipped_steps": sum(len(r.get("skipped") or []) for r in R),
        "asks_per_mission": boot([r.get("asks", 0) or 0 for r in R]),
        "fires_per_mission": boot([r.get("fires", 0) or 0 for r in R]),
        "by_action": {a: [sum(v), len(v), round(sum(v) / len(v), 3)] for a, v in sorted(by.items())},
        "gps_off_step_success": boot([ok for r in R if r.get("gps_off") for ok in steps(r)]),
        "parse_exact_step_success": boot([ok for r in R if r.get("parse_exact") for ok in steps(r)]),
        "parse_wrong_step_success": boot([ok for r in R if r.get("parse_exact") is False for ok in steps(r)]),
    }
    if any("battery_dead" in r for r in R):
        out.update(
            dead_in_air=boot([bool(r.get("dead_in_air")) for r in R]),
            battery_dead=boot([r.get("battery_dead") is not None for r in R]),
            home_landed=boot([bool(r.get("home_landed")) for r in R]),
            went_home_early=boot([bool(r.get("went_home_early")) for r in R]),
            lost_events_per_mission=boot([r.get("lost_events", 0) for r in R]),
            battery_end=boot([r.get("battery", 0) for r in R]),
        )
    return out


def by_variant(R):
    V = collections.defaultdict(dict)
    for r in R:
        if r.get("status") != "error":
            V[r["variant"]][r["i"]] = r
    return V, sum(r.get("status") == "error" for r in R)


def paired(A, B, keys):
    common = sorted(set(A) & set(B))
    out = {"n": len(common)}
    for name, f in keys.items():
        out[name] = boot([float(f(A[i])) - float(f(B[i])) for i in common])
    return out


K = {
    "step_success": lambda r: np.mean(steps(r) or [0]),
    "mission_success": lambda r: all(r.get("steps_ok") or [False]),
    "wrong_target": lambda r: bool(r.get("wrong_target")),
    "static_hit_mission": lambda r: r.get("hits_static", 0) > 0,
    "static_hits": lambda r: r.get("hits_static", 0),
    "attack_hits": lambda r: r.get("hits_attack", 0),
}
KH = {
    **K,
    "dead_in_air": lambda r: bool(r.get("dead_in_air")),
    "home_landed": lambda r: bool(r.get("home_landed")),
    "went_home_early": lambda r: bool(r.get("went_home_early")),
}
out = {}

# ---------------- G1
g1 = {}
try:
    g1["g1a"] = json.load(open(D / "g1a.json"))["summary"]
except Exception:
    pass
for name in ("new", "old"):
    R = rows(f"g1b_{name}.jsonl")
    ok = [bool(r.get("steps") and r["steps"][0].get("ok")) for r in R if "error" not in r]
    notes = collections.Counter()
    for r in R:
        for n in r.get("notes", []):
            for key in (
                "open door found",
                "doors look closed",
                "could not get through the door",
                "not touching anything",
                "not found",
                "cannot reach",
                "look-alike",
            ):
                if key in n:
                    notes[key] += 1
    closed = [r for r in R if r.get("target_facts") and r["target_facts"][0].get("door_open") is False]
    g1[f"enter_{name}"] = {
        "flights": len(R),
        "errors": sum("error" in r for r in R),
        "entered": sum(ok),
        "rate": boot(ok),
        "notes": dict(notes),
        "closed_target_flights": len(closed),
        "closed_entered": sum(bool(r["steps"][0].get("ok")) for r in closed if r.get("steps")),
    }
try:
    g1["rule"] = json.load(open(D / "g1_rule.json"))
except Exception:
    pass
out["G1"] = g1

# ---------------- G2 (+ G4 on its first 150)
V, err = by_variant(rows("g2.jsonl") + rows("g2_operator.jsonl") + rows("g4.jsonl"))
out["G2_errors"] = err
for v in ("full", "reflex", "operator", "false_alarm", "delayed"):
    if V.get(v):
        out[f"G_{v}"] = summary(list(V[v].values()))
if V.get("full") and V.get("reflex"):
    out["G2_full_minus_reflex"] = paired(V["full"], V["reflex"], K)
if V.get("full") and V.get("operator"):
    out["G2_operator_minus_full"] = paired(V["operator"], V["full"], K)
    # the drone alone on the same missions the operator run finished
    same = [V["full"][i] for i in V["operator"] if i in V["full"]]
    out["G_full_same_as_operator"] = summary(same)
    first200 = sorted(V["full"])[:200]
    out["G2_operator_unfinished"] = [i for i in first200 if i not in V["operator"]]
for v in ("false_alarm", "delayed"):
    if V.get(v) and V.get("full"):
        out[f"G4_{v}_minus_full"] = paired(V[v], V["full"], {**K, "fires": lambda r: r.get("fires", 0)})
if V.get("full"):
    w = sum(bool(r.get("wrong_target")) for r in V["full"].values())
    n = len(V["full"])
    out["G2_wrong_target_rule"] = {
        "n": n,
        "wrong": w,
        "rate_ci": out["G_full"]["wrong_target_rate"],
        "upper_le_2pct": (out["G_full"]["wrong_target_rate"][2] or 1) <= 0.02,
    }

# ---------------- G3
V3, err3 = by_variant(rows("g3.jsonl"))
out["G3_errors"] = err3
for v in ("full", "telemetry", "slow_none", "nolost", "noenergy"):
    if V3.get(v):
        out[f"G3_{v}"] = summary(list(V3[v].values()))
for v in ("telemetry", "slow_none", "nolost", "noenergy"):
    if V3.get(v) and V3.get("full"):
        out[f"G3_{v}_minus_full"] = paired(V3[v], V3["full"], KH)

# ---------------- G5
P5, O5 = rows("g5_px4.jsonl"), rows("g5_own.jsonl")


def s5(R):
    ok = [all(bool(s.get("ok")) for s in r["steps"]) for r in R if r.get("steps")]
    return {
        "missions": len(R),
        "errors": sum("error" in r for r in R),
        "completed": sum(ok),
        "by_action": {
            f"{a} ok={o}": n
            for (a, o), n in collections.Counter(
                (r.get("action"), all(bool(s.get("ok")) for s in r["steps"])) for r in R if r.get("steps")
            ).most_common()
        },
        "vel_track_err_mean": boot(
            [r["vel_track_err_mean"] for r in R if r.get("vel_track_err_mean") is not None]
        ),
        "watchdog_ticks": [r.get("watchdog_hover_ticks") for r in R if "watchdog_hover_ticks" in r],
        "hits": sum(len(r.get("hits") or []) for r in R),
        "wrong": sum(bool(r.get("wrong")) for r in R),
    }


if P5 or O5:
    out["G5_px4"] = s5(P5)
    out["G5_own"] = s5(O5)

(D / "G_ANALYSIS.json").write_text(
    json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
)
for k, v in out.items():
    if isinstance(v, dict) and "by_action" in v and "step_success" in v:
        print(
            f"== {k}: n {v['missions']} steps {v['step_success']} missions {v['mission_success']} wrong {v['wrong_target_n']} {v['wrong_target_rate']} static-hit missions {v['missions_with_static_hit']}"
            + (
                f" dead-in-air {v['dead_in_air']} home {v['home_landed']} early-home {v['went_home_early']}"
                if "dead_in_air" in v
                else ""
            )
            + f"\n   {v['by_action']}"
        )
    else:
        print(f"== {k}: {json.dumps(v, ensure_ascii=False, default=str)[:900]}")
