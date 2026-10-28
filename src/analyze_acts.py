"""Summary of an action-test batch."""

import collections, json, math, sys
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
by = collections.defaultdict(list)
for r in rows:
    if "steps" not in r:
        print("no mission:", r.get("action"), r.get("place"), r.get("error"))
        continue
    for i, st in enumerate(r["steps"]):
        by[st["action"] + (" (2-step)" if r["action"] == "pickup+drop" else "")].append((r, i, st))
print(f"{'action':22s} n  ok   skipped  wrong  coll")
for a, L in sorted(by.items()):
    ok = sum(bool(st.get("ok")) for _, _, st in L)
    sk = sum(i in r["skipped"] for r, i, _ in L)
    print(
        f"{a:22s} {len(L):2d} {ok:3d}   {sk:3d}      {sum(bool(r['wrong']) for r, _, _ in L):3d}    {sum(r['coll'] for r, _, _ in L):3d}"
    )
tot = [st for L in by.values() for _, _, st in L]
print(
    "all steps",
    len(tot),
    "ok",
    sum(bool(s.get("ok")) for s in tot),
    f"({np.mean([bool(s.get('ok')) for s in tot]):.0%})",
)
print("missions with a wrong target:", sum(bool(r["wrong"]) for r in rows if "steps" in r), "of", len(rows))
g = [r for r in rows if "steps" in r and r.get("gps_off")]
n = [r for r in rows if "steps" in r and not r.get("gps_off")]
for name, R in (("GPS off", g), ("GPS on", n)):
    if R:
        print(
            f"{name}: {len(R)} missions, steps ok {np.mean([bool(s.get('ok')) for r in R for s in r['steps']]):.0%}, nav err mean {np.mean([r['nav_err'][0] for r in R]):.2f} m, max {max(r['nav_err'][1] for r in R):.1f} m"
        )
for r in rows:
    if "steps" not in r:
        continue
    for st in r["steps"]:
        if st["action"] == "drop":
            print("drop miss", st.get("miss_m"), "debug", r.get("debug"))
print("\nwhy not ok:")
for a, L in sorted(by.items()):
    for r, i, st in L:
        if st.get("ok"):
            continue
        why = [
            n
            for n in r["notes"]
            if any(
                w in n
                for w in (
                    "skipping",
                    "not touching",
                    "closed",
                    "could not",
                    "too unsure",
                    "not found",
                    "too long",
                    "lost",
                )
            )
        ]
        print(
            f"  {a:18s} {r['place']:16s} {r['detail']:7s} gps_off={bool(r.get('gps_off'))} t={r['t']:5.0f} state={r['state']:10s} {st} | {why[-2:] if why else r['notes'][-3:]}"
        )
