"""Run the feeling core once over every stage D flight and save the feelings at each decision, so the.
usage: export_d_feelings.py DATA_DIR CORE_RUNTIME.npz OUT.json [--core-model gru]"""

import json, sys
from pathlib import Path
import llm_decide as L

if __name__ == "__main__":
    data, rt, out = sys.argv[1:4]
    model = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--core-model" else "gru"
    import core_runtime

    core = (core_runtime.FeelingCore if model == "gru" else core_runtime.LogregCore)(rt)
    F = {}
    for s in ("train", "test_familiar", "test_unfamiliar"):
        D = L.load(data, s)
        L.attach_feelings(D, core)
        F.update(L.feelings_cache(D))
        print(s, len(D), flush=True)
    Path(out).write_text(json.dumps({"core": f"{Path(rt).name}|{model}", "F": F}), encoding="utf-8")
