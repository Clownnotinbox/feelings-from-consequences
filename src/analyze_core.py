"""Fair analysis of the feeling cores."""

import glob, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

RUNS = Path(__file__).resolve().parents[1] / "runs"
MODELS = ("rule", "logreg", "gru", "fly")
DATA = ["stageA"]
PREFIX = "core_s"  # set from the command line


def episodes(split):
    E = {}
    for f in sorted(f for d in DATA for f in glob.glob(str(RUNS / d / f"{split}_*.npz"))):
        z = np.load(f)
        E[key(f)] = {"Y": z["Y"], "idx": z["idx"], "kind": z["kind"]}
    return E


def key(f):
    return (
        Path(f).stem if DATA == ["stageA"] else Path(f).parent.name + "/" + Path(f).stem
    )  # v1 cores saved bare names


def path_of(n):
    """Episode file from its saved name."""
    if "/" not in n:
        return str(RUNS / "stageA" / f"{n}.npz")
    d, stem = n.split("/")
    for dd in DATA:
        if Path(dd).name == d:
            return str(RUNS / dd / f"{stem}.npz")
    raise FileNotFoundError(n)


def norm(n):
    return (
        n.split("/")[-1] if DATA == ["stageA"] else n
    )  # cores trained before 24.10 14:45 saved bare episode names


def contacts(e):
    """Distinct contacts 'if nothing changes': (time, kind) from labelled steps with contact within 0.5 s."""
    out = []
    for i in np.nonzero(e["Y"][:, 0] > 0.5)[0]:
        t = e["idx"][i] * 0.1 + e["Y"][i, 3]
        if not any(abs(t - u) < 1.0 for u, _ in out):
            out.append((t, str(e["kind"][i])))
    return out


def warn(E, P, thr):
    leads, kinds = [], []
    for name, e in E.items():
        p = P[name][:, 2]
        for t, k in contacts(e):
            win = [j for j in range(len(e["idx"])) if t - 3.0 <= e["idx"][j] * 0.1 <= t]
            first = next((e["idx"][j] * 0.1 for j in win if p[j] >= thr), None)
            leads.append(t - first if first is not None else 0.0)
            kinds.append(k)
    return np.array(leads), np.array(kinds)


def fa(E, P, thr):
    neg = np.concatenate([P[n][E[n]["Y"][:, 2] < 0.5, 2] for n in E])
    return float((neg >= thr).mean())


def main():
    seeds = sorted(int(f.split(PREFIX)[-1].split(".")[0]) for f in glob.glob(str(RUNS / f"{PREFIX}*.json")))
    test = {s: episodes(s) for s in ("test_familiar", "test_unfamiliar")}
    R = {}
    for model in MODELS:
        for seed in seeds:
            pf = RUNS / f"{PREFIX}{seed}.pred_{model}.npz"
            if not pf.exists():
                continue
            Z = np.load(pf)
            P = {norm(k.split("|", 1)[1]): Z[k] for k in Z.files}
            val = {norm(k.split("|", 1)[1]) for k in Z.files if k.startswith("val|")}
            Ev = {}
            for n in val:
                z = np.load(path_of(n))
                Ev[n] = {"Y": z["Y"], "idx": z["idx"], "kind": z["kind"]}
            negv = np.concatenate([P[n][Ev[n]["Y"][:, 2] < 0.5, 2] for n in Ev])
            for split, E in test.items():
                y = np.concatenate([E[n]["Y"][:, 2] for n in E])
                p = np.concatenate([P[n][:, 2] for n in E])
                k = np.concatenate([E[n]["kind"] for n in E])
                rec = R.setdefault(model, {}).setdefault(split, {})
                for kind in ("static", "mover", "encounter"):
                    m = (y < 0.5) | (k == kind)
                    if ((y > 0.5) & (k == kind)).sum() > 5:
                        rec.setdefault(f"auroc_danger2_{kind}", []).append(roc_auc_score(y[m], p[m]))
                rec.setdefault("auroc_danger2_all", []).append(roc_auc_score(y, p))
                targets = (0.05, 0.10, 0.15) if model != "rule" else (None,)
                for fa_t in targets:
                    thr = 0.5 if fa_t is None else float(np.quantile(negv, 1 - fa_t))
                    leads, kinds = warn(E, P, thr)
                    key = "rule" if fa_t is None else f"fa{int(fa_t * 100)}"
                    rec.setdefault(f"{key}_fa_test", []).append(fa(E, P, thr))
                    rec.setdefault(f"{key}_warned", []).append(float((leads > 0).mean()))
                    rec.setdefault(f"{key}_mean_lead", []).append(float(leads.mean()))
                    for kind in ("static", "mover", "encounter"):
                        if (kinds == kind).sum():
                            rec.setdefault(f"{key}_warned_{kind}", []).append(
                                float((leads[kinds == kind] > 0).mean())
                            )
    agg = {
        m: {
            s: {k: (round(float(np.mean(v)), 3), round(float(np.std(v)), 3), len(v)) for k, v in d.items()}
            for s, d in r.items()
        }
        for m, r in R.items()
    }
    (RUNS / f'CORE_ANALYSIS_{PREFIX.rstrip("_s")}.json').write_text(
        json.dumps(agg, indent=1), encoding="utf-8"
    )
    for split in ("test_familiar", "test_unfamiliar"):
        print("==", split)
        for m in MODELS:
            if m not in agg:
                continue
            a = agg[m][split]
            g = lambda k: f"{a[k][0]:.3f}" if k in a else "  -  "
            print(
                f"{m:6s} AUROC danger2 all {g('auroc_danger2_all')} | static {g('auroc_danger2_static')} mover {g('auroc_danger2_mover')} encounter {g('auroc_danger2_encounter')}"
            )
            for key in (("rule",) if m == "rule" else ("fa5", "fa10", "fa15")):
                print(
                    f"        {key:5s}: FA on test {g(key + '_fa_test')}, warned {g(key + '_warned')} (static {g(key + '_warned_static')}, mover {g(key + '_warned_mover')}, encounter {g(key + '_warned_encounter')}), mean lead {g(key + '_mean_lead')} s"
                )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        DATA = sys.argv[1].split(",")
    if len(sys.argv) > 2:
        PREFIX = sys.argv[2]
    main()
