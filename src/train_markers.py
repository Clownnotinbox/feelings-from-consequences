"""Somatic markers per action.
usage: train_markers.py DATA_DIR OUT.json SEED"""

import glob, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from torch import nn

from markers_geom import ACTIONS, NA, SECT0, CMD0, geometry_scores, action_features, pairs, fwd_predict


def load(d, split):
    E = []
    for f in sorted(f for dd in str(d).split(",") for f in glob.glob(str(Path(dd) / f"{split}_*.npz"))):
        z = np.load(f)
        if "A" not in z.files or len(z["A"]) == 0:
            continue
        fr = z["idx"][z["aidx"]]  # frames with action markers
        E.append(
            {
                "X": z["X"],
                "fr": fr,
                "C": z["A"][:, :, 0],
                "P": z["A"][:, :, 2],
                "name": Path(f).stem,
                "kind": z["kind"][z["aidx"]],
                "D": z["D"][z["aidx"]] if "D" in z.files else None,
            }
        )
    return E


class MarkerGRU(nn.Module):
    def __init__(self, d_in, h=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, 96), nn.GELU())
        self.gru = nn.GRU(96, h, batch_first=True)
        self.head = nn.Linear(h, 2 * NA)

    def forward(self, x):
        return self.head(self.gru(self.inp(x))[0])


def decisions(Pc, C, Pr, thr, kind=None):
    """Keep unless p(contact | keep) >= thr."""
    alt = 1 + np.argmin(Pc[:, 1:], 1)
    ch = np.where(Pc[:, 0] >= thr, alt, 0)
    danger = C[:, 0] > 0.5
    got = C[np.arange(len(C)), ch]
    extra = {
        "choice_safe_given_danger": (
            float((C[np.arange(len(C)), alt][danger] < 0.5).mean()) if danger.any() else None
        )
    }
    if kind is not None:
        for k in ("static", "encounter"):
            m = danger & (kind == k)
            if m.any():
                extra[f"rescued_{k}"] = float((got[m] < 0.5).mean())
                extra[f"rescuable_{k}"] = float((C[m].min(1) < 0.5).mean())
                extra[f"n_{k}"] = int(m.sum())
    return {
        **extra,
        "danger_frames": int(danger.sum()),
        "rescued": float(((got < 0.5) & danger).sum() / max(danger.sum(), 1)),
        "rescuable": float(((C.min(1) < 0.5) & danger).sum() / max(danger.sum(), 1)),
        "switch_when_safe": float(((ch > 0) & ~danger).sum() / max((~danger).sum(), 1)),
        "made_worse": float(((got > 0.5) & ~danger).sum() / max((~danger).sum(), 1)),
        "progress_cost_m": float((Pr[:, 0] - Pr[np.arange(len(C)), ch])[~danger].mean()),
    }


def main(data, out, seed):
    t0 = time.time()
    tr = load(data, "train")
    te = {s: load(data, s) for s in ("test_familiar", "test_unfamiliar")}
    allX = np.concatenate([e["X"] for e in tr])
    mu, sd = allX.mean(0), allX.std(0) + 1e-6
    for e in tr + te["test_familiar"] + te["test_unfamiliar"]:
        e["Xn"] = np.clip((e["X"] - mu) / sd, -8, 8).astype(np.float32)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tr))
    val = [tr[i] for i in order[: len(tr) // 7]]
    fit = [tr[i] for i in order[len(tr) // 7 :]]
    cat = lambda E, k: np.concatenate([e[k] for e in E])
    preds = {"geometry": lambda E: [geometry_scores(e["X"][e["fr"]]) for e in E]}
    from sklearn.linear_model import LogisticRegression, Ridge

    Xf = np.concatenate([e["Xn"][e["fr"]] for e in fit])
    Cf, Pf = cat(fit, "C"), cat(fit, "P")
    lr, rg = [], []
    for j in range(NA):
        c = LogisticRegression(max_iter=1000, C=1.0)
        c.fit(Xf, Cf[:, j])
        lr.append(c)
        r = Ridge(1.0)
        r.fit(Xf, Pf[:, j])
        rg.append(r)
    preds["logreg"] = lambda E: [
        np.stack([c.predict_proba(e["Xn"][e["fr"]])[:, 1] for c in lr], 1) for e in E
    ]
    prog = {"logreg": lambda E: [np.stack([r.predict(e["Xn"][e["fr"]]) for r in rg], 1) for e in E]}
    from sklearn.ensemble import HistGradientBoostingClassifier

    Zf, Czf, _ = pairs(fit)
    fwd = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed
    )
    fwd.fit(Zf, Czf)
    preds["fwd_gbdt"] = lambda E: [fwd_predict(fwd, e) for e in E]
    print("fwd_gbdt fitted", Zf.shape, f"{time.time() - t0:.0f}s", flush=True)
    # choice model: the same forward model, trained only on the moments
    dfit = [
        {
            **e,
            "fr": e["fr"][e["C"][:, 0] > 0.5],
            "C": e["C"][e["C"][:, 0] > 0.5],
            "P": e["P"][e["C"][:, 0] > 0.5],
        }
        for e in fit
        if (e["C"][:, 0] > 0.5).any()
    ]
    Zd, Cd, _ = pairs(dfit)
    keepcol = Zd[:, allX.shape[1]] > 0.5
    Zd, Cd = Zd[~keepcol], Cd[~keepcol]
    cho = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed
    ).fit(Zd, Cd)
    preds["fwd+choice"] = lambda E: [
        np.hstack([fwd_predict(fwd, e)[:, :1], fwd_predict(cho, e)[:, 1:]]) for e in E
    ]
    print("choice model fitted on", len(Zd), "danger (state, alternative) pairs", flush=True)
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = MarkerGRU(allX.shape[1]).to(dev)
    opt = torch.optim.AdamW(g.parameters(), 2e-3, weight_decay=1e-4)
    wpos = torch.ones(NA, device=dev)  # no class weights: they hurt ranking (stage B, 24.10)
    psd = float(Pf.std()) + 1e-6

    def gru_out(E):
        g.eval()
        o = []
        with torch.no_grad():
            for e in E:
                o.append(g(torch.tensor(e["Xn"][None], device=dev))[0].cpu().numpy()[e["fr"]])
        return o

    best, state = 1e9, None
    for ep in range(25):
        g.train()
        for b in range(0, len(fit), 16):
            batch = [fit[i] for i in rng.permutation(len(fit))[:16]]
            L = max(len(e["Xn"]) for e in batch)
            x = torch.zeros(len(batch), L, allX.shape[1], device=dev)
            for i, e in enumerate(batch):
                x[i, : len(e["Xn"])] = torch.tensor(e["Xn"], device=dev)
            o = g(x)
            loss = 0.0
            for i, e in enumerate(batch):
                oo = o[i, e["fr"]]
                c = torch.tensor(e["C"], device=dev)
                p = torch.tensor(e["P"], device=dev) / psd
                loss = (
                    loss
                    + nn.functional.binary_cross_entropy_with_logits(oo[:, :NA], c, pos_weight=wpos)
                    + 0.3 * nn.functional.smooth_l1_loss(oo[:, NA:], p)
                )
            opt.zero_grad()
            (loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
            opt.step()
        vo = gru_out(val)
        vl = float(
            np.mean(
                [
                    nn.functional.binary_cross_entropy_with_logits(
                        torch.tensor(v[:, :NA]), torch.tensor(e["C"])
                    ).item()
                    for v, e in zip(vo, val)
                ]
            )
        )
        print(f"markers gru epoch {ep} val bce {vl:.4f} {time.time() - t0:.0f}s", flush=True)
        if vl < best:
            best, state = vl, {k: v.clone() for k, v in g.state_dict().items()}
    g.load_state_dict(state)
    preds["gru"] = lambda E: [1 / (1 + np.exp(-o[:, :NA])) for o in gru_out(E)]
    prog["gru"] = lambda E: [o[:, NA:] * psd for o in gru_out(E)]
    from sklearn.metrics import roc_auc_score

    R = {
        "seed": seed,
        "n_fit": len(fit),
        "n_val": len(val),
        "n_test": {k: len(v) for k, v in te.items()},
        "models": {},
    }
    for name, fn in preds.items():
        Pv, Cv = np.concatenate(fn(val)), cat(val, "C")
        rec = R["models"][name] = {}
        for fa in (0.05, 0.10):
            thr = float(np.quantile(Pv[Cv[:, 0] < 0.5, 0], 1 - fa))
            rec[f"thr_fa{int(fa * 100)}"] = thr
        for s, E in te.items():
            P, C, Pr = np.concatenate(fn(E)), cat(E, "C"), cat(E, "P")
            r = rec[s] = {}
            r["auroc"] = {
                a: round(float(roc_auc_score(C[:, j], P[:, j])), 4)
                for j, a in enumerate(ACTIONS)
                if 0 < C[:, j].mean() < 1
            }
            r["base_rate"] = {a: round(float(C[:, j].mean()), 4) for j, a in enumerate(ACTIONS)}
            for fa in (5, 10):
                r[f"decide_fa{fa}"] = {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in decisions(P, C, Pr, rec[f"thr_fa{fa}"], cat(E, "kind")).items()
                }
            if name in prog:
                Q = np.concatenate(prog[name](E))
                r["progress_corr"] = round(float(np.corrcoef(Q.ravel(), Pr.ravel())[0, 1]), 4)
        print(name, json.dumps(rec)[:700], flush=True)
    # reference policies that ignore the markers
    for s, E in te.items():
        C, Pr = cat(E, "C"), cat(E, "P")
        danger = C[:, 0] > 0.5
        R.setdefault("fixed_reaction_if_danger_known", {})[s] = {
            a: round(float((C[danger, j] < 0.5).mean()), 4) for j, a in enumerate(ACTIONS)
        }
        K = cat(E, "kind")
        for k in ("static", "encounter"):
            m = danger & (K == k)
            if m.any():
                R.setdefault(f"fixed_reaction_if_danger_known_{k}", {})[s] = {
                    a: round(float((C[m, j] < 0.5).mean()), 4) for j, a in enumerate(ACTIONS)
                }
    R["seconds"] = round(time.time() - t0)
    Path(out).write_text(json.dumps(R, indent=1), encoding="utf-8")
    import pickle

    with open(Path(out).with_suffix(".fwd.pkl"), "wb") as f:
        pickle.dump(
            {
                "mu": mu,
                "sd": sd,
                "fwd": fwd,
                "cho": cho,
                "thr": {k: v for k, v in R["models"]["fwd+choice"].items() if k.startswith("thr")},
            },
            f,
        )
    sd_ = {k: v.detach().cpu().numpy() for k, v in g.state_dict().items()}
    np.savez(
        Path(out).with_suffix(".runtime.npz"),
        mu=mu,
        sd=sd,
        psd=psd,
        thr=json.dumps(
            {n: {k: v for k, v in R["models"][n].items() if k.startswith("thr")} for n in R["models"]}
        ),
        lr_W=np.stack([c.coef_[0] for c in lr]),
        lr_b=np.array([c.intercept_[0] for c in lr]),
        **{"gru." + k: v for k, v in sd_.items()},
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
