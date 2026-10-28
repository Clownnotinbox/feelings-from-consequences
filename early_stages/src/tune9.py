"""Dev-seed tuning of the Kenyon-cell code."""

import json
import numpy as np, torch
import e9
from smoke9 import Reader, spk, bank

for n_kc, k in ((2000, 20), (2000, 50), (2000, 100), (4000, 40), (4000, 100)):
    orig = e9.Memory.__init__
    e9.Memory.__init__ = lambda self, kind, dim, seed, n_kc=n_kc, k=k: orig(self, kind, dim, seed, n_kc, k)
    rs = [e9.run_protocol(Reader(), "mb_sparse", spk, bank, s) for s in (21, 22, 23)]
    e9.Memory.__init__ = orig
    print(
        n_kc,
        k,
        "acc",
        round(float(np.mean([r["accuracy"] for r in rs])), 3),
        "returning first turn",
        round(float(np.mean([np.mean(r["first_turn_by_round"][1:]) for r in rs])), 3),
    )
