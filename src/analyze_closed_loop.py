"""Summary of closed-loop flights."""

import json, sys
from collections import defaultdict
import numpy as np


def main(path):
    R = [r for f in path.split(",") for r in json.load(open(f))]
    by = defaultdict(dict)  # several result files = more missions
    for r in R:
        by[(r["split"], r["variant"])][r["i"]] = r
    variants = list(dict.fromkeys(r["variant"] for r in R))
    out = {}
    rng = np.random.default_rng(0)
    for split in ("test_familiar", "test_unfamiliar"):
        print("==", split)
        base = by[(split, "none")]
        for v in variants:
            F = by[(split, v)]
            ids = sorted(set(F) & set(base))
            if not ids:
                continue
            h = np.array([F[i]["hits"] for i in ids])
            hs = np.array([F[i]["hits_static"] for i in ids])
            he = np.array([F[i]["hits_encounter"] for i in ids])
            dh = h - np.array([base[i]["hits"] for i in ids])
            dp = np.array([F[i]["goal_progress_m"] - base[i]["goal_progress_m"] for i in ids])
            boot = [dh[rng.integers(0, len(ids), len(ids))].mean() for _ in range(2000)]
            fires = sum(F[i]["fires"] for i in ids)
            real = sum(F[i]["fires_real"] for i in ids)
            rec = {
                "n": len(ids),
                "hits_per_flight": round(h.mean(), 2),
                "static": round(hs.mean(), 2),
                "encounter": round(he.mean(), 2),
                "flights_without_contact": round(float((h == 0).mean()), 3),
                "d_hits_vs_none": round(dh.mean(), 2),
                "d_hits_ci95": [
                    round(float(np.quantile(boot, 0.025)), 2),
                    round(float(np.quantile(boot, 0.975)), 2),
                ],
                "fires_per_flight": round(fires / len(ids), 1),
                "fires_real_share": round(real / max(fires, 1), 3),
                "override_share": round(float(np.mean([F[i]["override_share"] for i in ids])), 3),
                "d_progress_m": round(dp.mean(), 1),
                "legs_share": round(
                    float(np.mean([F[i]["legs_done"] / max(F[i]["legs"], 1) for i in ids])), 3
                ),
                "blocked_per_flight": round(float(np.mean([F[i].get("blocked", 0) for i in ids])), 2),
            }
            out[f"{split}|{v}"] = rec
            print(
                f"{v:18s} hits/flight {rec['hits_per_flight']:.2f} (static {rec['static']:.2f}, enc {rec['encounter']:.2f}) clean {rec['flights_without_contact']:.2f} | "
                f"d vs none {rec['d_hits_vs_none']:+.2f} {rec['d_hits_ci95']} | fires {rec['fires_per_flight']:.1f} real {rec['fires_real_share']:.2f} | "
                f"override {rec['override_share']:.2f} | legs {rec['legs_share']:.2f} | blocked {rec['blocked_per_flight']:.2f}"
            )
    json.dump(out, open(path.split(",")[0].replace(".json", "_SUMMARY.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
