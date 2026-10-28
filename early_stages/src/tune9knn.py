"""Dev-seed tuning of the knn similarity threshold with the ideal reader."""

import numpy as np
import e9
from smoke9 import Reader, spk, bank

for tau in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
    e9.Memory.TAU = tau
    rs = [e9.run_protocol(Reader(), "knn", spk, bank, s) for s in (21, 22, 23)]
    print(
        tau,
        "acc",
        round(float(np.mean([r["accuracy"] for r in rs])), 3),
        "returning first turn",
        round(float(np.mean([np.mean(r["first_turn_by_round"][1:]) for r in rs])), 3),
        flush=True,
    )
rs = [e9.run_protocol(Reader(), "mb_sparse", spk, bank, s) for s in (21, 22, 23)]
print(
    "mb_sparse acc",
    round(float(np.mean([r["accuracy"] for r in rs])), 3),
    "returning first turn",
    round(float(np.mean([np.mean(r["first_turn_by_round"][1:]) for r in rs])), 3),
)
