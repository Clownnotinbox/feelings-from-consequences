"""Merge per-seed result files of llm_train_decide2."""

import json, sys
from pathlib import Path

R = {}
for f in sys.argv[2:]:
    r = json.loads(Path(f).read_text(encoding="utf-8"))
    for k, v in r.items():
        if "|seed" in k:
            R[k] = v
        else:
            R.setdefault(k, v)
Path(sys.argv[1]).write_text(json.dumps(R, indent=1), encoding="utf-8")
print(sorted(k for k in R if "|seed" in k))
