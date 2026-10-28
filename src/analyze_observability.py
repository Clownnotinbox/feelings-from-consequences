"""Why is danger from encounters near chance? Split the coming contacts by what the lidar could.
usage: analyze_observability.py DATA_DIR RUNTIME.npz [RUNTIME2.npz ...]  (episodes the cores never saw)"""

import glob, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

RANGE, T_REACT = 40.0, 0.7


def strata(Y, kind, D):
    pos = Y[:, 2] > 0.5
    d0, v, seen = D[:, 0], D[:, 1], D[:, 2]
    tc = Y[:, 3]
    t_enter = np.where(np.isfinite(d0), np.maximum(0.0, (d0 - RANGE) / np.maximum(v, 0.1)), np.inf)
    vis_time = np.where(np.isfinite(d0), tc - t_enter, 0.0)
    enc = pos & (kind == "encounter")
    S = {
        "static": pos & (kind == "static"),
        "enc_seen": enc & (seen > 0),
        "enc_in_range": enc & ~(seen > 0) & (d0 < RANGE),
        "enc_out_of_range": enc & ~(d0 < RANGE),
        "enc_avoidable": enc & (vis_time >= T_REACT),
        "enc_unavoidable": enc & (vis_time < T_REACT),
    }
    return S, ~pos


def main(data, runtimes):
    import core_runtime
    import os

    pat = os.environ.get("OBS_SPLIT", "") + "*.npz"  # e.g. OBS_SPLIT=test_ : only the test episodes
    files = sorted(f for d in data.split(",") for f in glob.glob(str(Path(d) / pat)))
    E = []
    for f in files:
        z = np.load(f)
        if "D" in z.files:
            E.append({k: z[k] for k in ("X", "Y", "idx", "kind", "D")})
    out = {"episodes": len(E)}
    Y = np.concatenate([e["Y"] for e in E])
    K = np.concatenate([e["kind"] for e in E])
    D = np.concatenate([e["D"] for e in E])
    S, neg = strata(Y, K, D)
    out["counts"] = {k: int(m.sum()) for k, m in S.items()}
    out["negatives"] = int(neg.sum())
    enc = (Y[:, 2] > 0.5) & (K == "encounter")
    out["enc_speed_median"] = float(np.nanmedian(D[enc, 1])) if enc.any() else None
    out["enc_size_share_small"] = float(np.nanmean(D[enc, 4] < 0.25)) if enc.any() else None
    for rt in runtimes:
        for model in ("logreg", "gru"):
            core = (core_runtime.LogregCore if model == "logreg" else core_runtime.FeelingCore)(rt)
            P = []
            for e in E:
                core.reset()
                p = np.array([core.step(x)["danger_2"] for x in e["X"]])
                P.append(p[e["idx"]])
            P = np.concatenate(P)
            rec = out.setdefault(f"{Path(rt).stem}|{model}", {})
            for k, m in S.items():
                if m.sum() >= 5:
                    rec[k] = round(
                        float(
                            roc_auc_score(np.r_[np.ones(m.sum()), np.zeros(neg.sum())], np.r_[P[m], P[neg]])
                        ),
                        4,
                    )
        print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2:])
    d0 = Path(sys.argv[1].split(",")[0])
    d0.parent.joinpath(f"OBSERVABILITY_{d0.name}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
