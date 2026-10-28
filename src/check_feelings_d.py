"""Are the core's feelings trustworthy in stage D flights."""

import sys, json, numpy as np
import llm_decide as L, core_runtime as cr

D = L.load(sys.argv[1], sys.argv[2])
core = cr.FeelingCore(sys.argv[3])
L.attach_feelings(D, core)
F = {k: np.array([d["F"][k] for d in D]) for k in L.FEEL_TXT}
print("decisions", len(D))
[
    print(f"{k:10s} mean {v.mean():.2f} median {np.median(v):.2f} share>0.5 {np.mean(v > .5):.2f}")
    for k, v in F.items()
]
cont = [d["outcomes"]["continue"] for d in D]
empty = np.array([o["empty_in_air"] for o in cont])
crash = np.array([o["crash"] for o in cont])
lost = np.array([o["nav_err_m"] > 10 for o in cont])
t = np.array([d["snapshot"]["t"] for d in D])
nogps = np.array([d["snapshot"]["gps"] is None for d in D])
from sklearn.metrics import roc_auc_score

for name, f, y in (
    ("low_energy vs continue->battery empty", F["low_energy"], empty),
    ("lost vs own error >10 m at the end", F["lost"], lost),
    ("danger_2 vs continue->crash", F["danger_2"], crash),
):
    print(
        name, "base", round(y.mean(), 2), "AUROC", round(roc_auc_score(y, f), 3) if 0 < y.mean() < 1 else None
    )
for lo, hi in ((0, 40), (40, 80), (80, 130)):
    m = (t >= lo) & (t < hi)
    print(f"t {lo}-{hi}s n={m.sum()}", {k: round(float(v[m].mean()), 2) for k, v in F.items()})
