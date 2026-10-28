"""Per-run compact table of E2 results: python src/table.py RUNS_DIR."""

import json, sys
from pathlib import Path

for f in sorted(Path(sys.argv[1]).glob("e2/*/RESULT.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    print(
        f'{f.parent.name:8s} iid {r["iid_accuracy"]:.3f} ood {r["ood_accuracy"]:.3f} AR {r["AR_sensitivity"]:.2f}/{r["AR_specificity"]:.2f} '
        f'dose {r["dose_fidelity_mean"]:.2f} conf {r["conflict_signal_following"]:.2f} free {r["free_generation_accuracy"]:.2f} '
        f'nll {r["retention_nll"]["off"]:.3f}'
    )
