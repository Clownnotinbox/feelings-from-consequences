"""Stage B: the feeling core."""

import argparse, glob, json, math, time
from pathlib import Path
import numpy as np, torch
from torch import nn

T_NAMES = (
    "danger_05",
    "danger_1",
    "danger_2",
    "t_contact",
    "min_enc_dist",
    "gps_err",
    "gps_bad",
    "energy_margin",
    "low_energy",
    "strain",
    "nav_err",
    "lost",
)
OUT_IDX = [0, 1, 2, 6, 8, 9, 11]
OUT_NAMES = [
    "danger_05",
    "danger_1",
    "danger_2",
    "gps_bad",
    "low_energy",
    "strain",
    "lost",
]  # 'lost' from data v2 on
TRACK0 = 13 + 3 + 2 + 6 + 6 + 12 + 10  # first track feature (see features.py)


def load(d, split):
    """d: one data directory or several, comma-separated."""
    E = []
    for f in sorted(f for dd in str(d).split(",") for f in glob.glob(str(Path(dd) / f"{split}_*.npz"))):
        z = np.load(f)
        Yr = (
            z["Y"]
            if z["Y"].shape[1] >= 12
            else np.hstack([z["Y"], np.full((len(z["Y"]), 2), np.nan, np.float32)])
        )
        Y = Yr[:, OUT_IDX].copy()
        Y[:, 5] = (Y[:, 5] > 0.1).astype(np.float32)  # strain: > 10 % of the next 2 s
        E.append(
            {
                "X": z["X"],
                "Y": Y,
                "idx": z["idx"],
                "tc": z["Y"][:, 3],
                "kind": z["kind"],
                "name": Path(f).parent.name + "/" + Path(f).stem,
            }
        )
    return E


def rule_scores(X):
    """Classic: p(danger within T) = 1 if any track has 1/TTC >= 1/T and passes within the safety margin."""
    out = np.zeros((len(X), 3), np.float32)
    for k in range(5):
        b = TRACK0 + 8 * k
        inv_ttc = X[:, b + 5]
        d_cpa = X[:, b + 6] * 20
        dist = X[:, b + 3] * 40
        near = d_cpa < 1.2 + 0.1 * dist
        for j, T in enumerate((0.5, 1.0, 2.0)):
            out[:, j] = np.maximum(out[:, j], (near & (inv_ttc >= 1 / T)).astype(np.float32))
    return out


class GRUCore(nn.Module):
    """skip=True adds a linear path from the current frame to the outputs."""

    def __init__(self, d_in, h=128, skip=False):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, 96), nn.GELU())
        self.gru = nn.GRU(96, h, batch_first=True)
        self.head = nn.Linear(h, len(OUT_NAMES))
        self.lin = nn.Linear(d_in, len(OUT_NAMES)) if skip else None

    def forward(self, x):
        o = self.head(self.gru(self.inp(x))[0])
        return o + self.lin(x) if self.lin is not None else o


class FlyCore:
    """Mushroom-body core."""

    def __init__(self, d_in, n_kc=3000, fan_in=8, sparsity=0.05, tau=0.7, seed=0):
        r = np.random.default_rng(seed)
        self.n_kc, self.k = n_kc, int(sparsity * n_kc)
        self.decay = math.exp(-0.1 / tau)
        self.idx = r.integers(0, 2 * d_in, (n_kc, fan_in))
        self.w_in = r.choice([-1.0, 1.0], (n_kc, fan_in)) * r.uniform(0.5, 1.5, (n_kc, fan_in))
        self.W = np.zeros((2 * n_kc, len(OUT_NAMES)))
        self.b = np.zeros(len(OUT_NAMES))

    def kc_seq(self, X):
        dX = np.vstack([np.zeros((1, X.shape[1])), np.diff(X, axis=0)]) * 3.0
        Z = np.hstack([X, dX])  # current frame + its change
        H = (Z[:, self.idx] * self.w_in).sum(-1)
        thr = np.partition(H, -self.k, axis=1)[:, -self.k][:, None]
        K = (H >= thr).astype(np.float32)
        T = np.zeros_like(K)
        tr = np.zeros(self.n_kc, np.float32)
        for t in range(len(K)):
            tr = tr * self.decay + K[t] * (1 - self.decay)
            T[t] = tr
        return np.hstack([K, T * 4.0])

    def predict(self, X):
        return 1 / (1 + np.exp(-(self.kc_seq(X) @ self.W + self.b)))

    def fit(self, E, epochs=4, eta=0.02, rng=None):
        for ep in range(epochs):
            for e in (rng.permutation(len(E)) if rng is not None else range(len(E))):
                K = self.kc_seq(E[e]["Xn"])[E[e]["idx"]]
                Y = E[e]["Y"]
                M = ~np.isnan(Y)
                for t in range(len(K)):  # dopamine = outcome
                    p = 1 / (1 + np.exp(-(K[t] @ self.W + self.b)))
                    da = np.where(M[t], np.nan_to_num(Y[t]) - p, 0.0)
                    nz = np.nonzero(K[t])[0]
                    self.W[nz] += eta * np.outer(K[t][nz], da)
                    self.b += eta * 0.1 * da


def metrics(P, Y):
    from sklearn.metrics import roc_auc_score, average_precision_score

    R = {}
    for j, n in enumerate(OUT_NAMES[: P.shape[1]]):
        m = ~np.isnan(Y[:, j])
        y, p = Y[m, j], P[m, j]
        if len(np.unique(y)) < 2:
            R[n] = None
            continue
        bins = np.clip((p * 10).astype(int), 0, 9)
        ece = sum(
            abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
            for b in range(10)
            if (bins == b).any()
        )
        R[n] = {
            "auroc": round(float(roc_auc_score(y, p)), 4),
            "auprc": round(float(average_precision_score(y, p)), 4),
            "base_rate": round(float(y.mean()), 4),
            "ece": round(float(ece), 4),
        }
    return R


def warning_time(E, preds, thr, col=2):
    """For every real contact."""
    leads = []
    for e, P in zip(E, preds):
        Yd = e["Y"][:, 0]
        tc = e["tc"]
        idx = e["idx"]
        seen = set()
        for i in np.nonzero(Yd > 0.5)[0]:  # labelled steps with contact within 0.5 s
            t_contact = idx[i] * 0.1 + tc[i]
            if any(abs(t_contact - s) < 1.0 for s in seen):
                continue
            seen.add(t_contact)
            win = [k for k in range(len(idx)) if t_contact - 3.0 <= idx[k] * 0.1 <= t_contact]
            first = next((idx[k] * 0.1 for k in win if P[k, col] >= thr), None)
            leads.append(t_contact - first if first is not None else 0.0)
    return leads


def main(a):
    t0 = time.time()
    D = a.data
    tr = load(D, "train")
    te = {s: load(D, s) for s in ("test_familiar", "test_unfamiliar")}
    allX = np.concatenate([e["X"] for e in tr])
    mu, sd = allX.mean(0), allX.std(0) + 1e-6
    for e in tr + te["test_familiar"] + te["test_unfamiliar"]:
        e["Xn"] = np.clip((e["X"] - mu) / sd, -8, 8).astype(np.float32)
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(tr))
    val = [tr[i] for i in order[: len(tr) // 7]]
    fit = [tr[i] for i in order[len(tr) // 7 :]]
    R = {
        "n_train_ep": len(fit),
        "n_val_ep": len(val),
        "n_test": {k: len(v) for k, v in te.items()},
        "seed": a.seed,
    }
    preds = {}
    # rule
    preds["rule"] = lambda E: [
        np.hstack([rule_scores(e["X"][e["idx"]]), np.full((len(e["idx"]), len(OUT_NAMES) - 3), 0.5)])
        for e in E
    ]
    # logistic regression (no memory)
    from sklearn.linear_model import LogisticRegression

    Xf = np.concatenate([e["Xn"][e["idx"]] for e in fit])
    Yf = np.concatenate([e["Y"] for e in fit])
    lr_models = []
    for j in range(len(OUT_NAMES)):
        m = ~np.isnan(Yf[:, j])
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(Xf[m], Yf[m, j]) if len(np.unique(Yf[m, j])) > 1 else None
        lr_models.append(clf)
    preds["logreg"] = lambda E: [
        np.stack(
            [
                (
                    c.predict_proba(e["Xn"][e["idx"]])[:, 1]
                    if c is not None and hasattr(c, "coef_")
                    else np.full(len(e["idx"]), 0.5)
                )
                for c in lr_models
            ],
            1,
        )
        for e in E
    ]
    # GRU
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = GRUCore(allX.shape[1], skip=a.skip).to(dev)
    opt = torch.optim.AdamW(g.parameters(), 2e-3, weight_decay=a.wd)
    pos = np.nan_to_num(np.nanmean(Yf, 0), nan=0.5)
    wpos = torch.tensor(
        np.clip((1 - pos) / np.maximum(pos, 1e-3), 1, 20) if not a.no_posweight else np.ones_like(pos),
        dtype=torch.float32,
        device=dev,
    )  # balance rare danger

    def gru_pred(E):
        g.eval()
        out = []
        with torch.no_grad():
            for e in E:
                out.append(
                    torch.sigmoid(g(torch.tensor(e["Xn"][None], device=dev)))[0].cpu().numpy()[e["idx"]]
                )
        return out

    best, best_state = 1e9, None
    for ep in range(a.epochs):
        g.train()
        for b in range(0, len(fit), 16):
            batch = [fit[i] for i in rng.permutation(len(fit))[:16]]
            L = max(len(e["Xn"]) for e in batch)
            x = torch.zeros(len(batch), L, allX.shape[1], device=dev)
            for i, e in enumerate(batch):
                x[i, : len(e["Xn"])] = torch.tensor(e["Xn"], device=dev)
            out = g(x)
            loss = 0.0
            for i, e in enumerate(batch):
                y = torch.tensor(e["Y"], device=dev)
                m = ~torch.isnan(y)
                o = out[i, e["idx"]]
                l = nn.functional.binary_cross_entropy_with_logits(
                    o, torch.nan_to_num(y), pos_weight=wpos, reduction="none"
                )
                loss = loss + (l * m).sum() / m.sum().clamp_min(1)
            opt.zero_grad()
            (loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
            opt.step()
        vp = gru_pred(val)
        vl = -np.nanmean(
            [
                np.nanmean(
                    np.where(
                        np.isnan(e["Y"]),
                        np.nan,
                        e["Y"] * np.log(p + 1e-6) + (1 - e["Y"]) * np.log(1 - p + 1e-6),
                    )
                )
                for e, p in zip(val, vp)
            ]
        )
        if vl < best:
            best, best_state = vl, {
                k: v.clone() for k, v in g.state_dict().items()
            }  # early stopping on validation (train places only)
        print(
            f"gru epoch {ep} val logloss {vl:.4f} loss {float(loss) / len(batch):.4f} {time.time() - t0:.0f}s",
            flush=True,
        )
    g.load_state_dict(best_state)
    preds["gru"] = gru_pred

    # fly core
    def vloss(fn):  # validation log-loss over all feelings (train places only)
        P = fn(val)
        return -np.nanmean(
            np.concatenate(
                [
                    np.where(
                        np.isnan(e["Y"]),
                        np.nan,
                        e["Y"] * np.log(p + 1e-6) + (1 - e["Y"]) * np.log(1 - p + 1e-6),
                    )
                    for e, p in zip(val, P)
                ]
            )
        )

    fly, best_fl = None, 1e9
    for eta in (
        (0.005, 0.02, 0.05) if not a.no_fly else ()
    ):  # the fly core gets the same tuning budget on validation as the others
        for ep_ in (3, 6):
            f_ = FlyCore(allX.shape[1], seed=a.seed)
            f_.fit(fit, epochs=ep_, eta=eta, rng=np.random.default_rng(a.seed))
            l_ = vloss(lambda E, f_=f_: [f_.predict(e["Xn"])[e["idx"]] for e in E])
            if l_ < best_fl:
                best_fl, fly, R["fly_choice"] = (
                    l_,
                    f_,
                    {"eta": eta, "epochs": ep_, "val_logloss": round(float(l_), 4)},
                )
    if fly is not None:
        preds["fly"] = lambda E: [fly.predict(e["Xn"])[e["idx"]] for e in E]
    # Platt calibration on validation for the learned cores
    CAL = {}

    def platt(fn):
        vp = fn(val)
        Pv = np.concatenate(vp)
        Yv = np.concatenate([e["Y"] for e in val])
        ab = []
        for j in range(Pv.shape[1]):
            m = ~np.isnan(Yv[:, j])
            z = np.log(np.clip(Pv[m, j], 1e-6, 1 - 1e-6) / np.clip(1 - Pv[m, j], 1e-6, 1))
            y = Yv[m, j]
            if m.sum() < 20 or len(np.unique(y)) < 2:
                ab.append((1.0, 0.0))
                continue  # nothing to calibrate on
            A, B = torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True)
            o = torch.optim.LBFGS([A, B], max_iter=100)
            zt, yt = torch.tensor(z, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

            def cl():
                o.zero_grad()
                l = nn.functional.binary_cross_entropy_with_logits((1 + A) * zt + B, yt)
                l.backward()
                return l

            o.step(cl)
            ab.append((1 + float(A), float(B)))
        CAL[fn] = ab

        def cal(E):
            out = []
            for P in fn(E):
                z = np.log(np.clip(P, 1e-6, 1 - 1e-6) / np.clip(1 - P, 1e-6, 1))
                out.append(
                    1 / (1 + np.exp(-np.stack([a_ * z[:, j] + b_ for j, (a_, b_) in enumerate(ab)], 1)))
                )
            return out

        return cal

    raw_gru, raw_fly = preds["gru"], preds.get("fly")
    preds["gru"] = platt(raw_gru)
    if raw_fly is not None:
        preds["fly"] = platt(raw_fly)
    # thresholds at a matched false-alarm rate
    for name, fn in preds.items():
        vp = fn(val)
        neg = np.concatenate([p[e["Y"][:, 2] < 0.5, 2] for e, p in zip(val, vp)])
        thr = float(np.quantile(neg, 0.95)) if name != "rule" else 0.5
        fa_val = float((neg >= thr).mean())
        R[name] = {"threshold_danger_2": round(thr, 4), "false_alarm_rate_val": round(fa_val, 4)}
        for s, E in te.items():
            P = fn(E)
            R[name][s] = metrics(np.concatenate(P), np.concatenate([e["Y"] for e in E]))
            negs = np.concatenate([p[e["Y"][:, 2] < 0.5, 2] for e, p in zip(E, P)])
            leads = warning_time(E, P, thr)
            R[name][s]["false_alarm_rate"] = round(float((negs >= thr).mean()), 4)
            R[name][s]["warning"] = {
                "contacts": len(leads),
                "warned": int(sum(l > 0 for l in leads)),
                "median_lead_s": round(float(np.median(leads)), 2) if leads else None,
                "mean_lead_s": round(float(np.mean(leads)), 2) if leads else None,
            }
        print(name, json.dumps({k: v for k, v in R[name].items()}, ensure_ascii=False)[:600], flush=True)
        saved = {}
        for s_, E in list(te.items()) + [("val", val)]:
            for e, P in zip(E, fn(E)):
                saved[f'{s_}|{e["name"]}'] = P.astype(np.float32)
        np.savez_compressed(Path(a.out).with_suffix(f".pred_{name}.npz"), **saved)
    R["seconds"] = round(time.time() - t0)
    Path(a.out).write_text(json.dumps(R, indent=1), encoding="utf-8")
    torch.save(g.state_dict(), Path(a.out).with_suffix(".gru.pt"))
    if fly is not None:
        np.savez(
            Path(a.out).with_suffix(".fly.npz"), W=fly.W, b=fly.b, idx=fly.idx, w_in=fly.w_in, mu=mu, sd=sd
        )
    # runtime bundle (numpy only)
    sd_ = {k: v.detach().cpu().numpy() for k, v in g.state_dict().items()}
    thr = {}
    for name in [n for n in ("gru", "fly", "logreg") if n in preds]:
        vp = preds[name](val)
        neg = np.concatenate([p_[e["Y"][:, 1] < 0.5, 1] for e, p_ in zip(val, vp)])  # danger within 1 s
        thr[name] = {str(f): float(np.quantile(neg, 1 - f)) for f in (0.01, 0.02, 0.05, 0.1)}
    np.savez(
        Path(a.out).with_suffix(".runtime.npz"),
        mu=mu,
        sd=sd,
        gru_cal=np.array(CAL[raw_gru]),
        fly_cal=np.array(CAL[raw_fly]) if raw_fly is not None else np.zeros((len(OUT_NAMES), 2)),
        out_names=np.array(OUT_NAMES),
        thr=json.dumps(thr),
        d_in=allX.shape[1],
        **{"gru." + k: v for k, v in sd_.items()},
        lr_W=np.stack([c.coef_[0] if hasattr(c, "coef_") else np.zeros(allX.shape[1]) for c in lr_models]),
        lr_b=np.array([c.intercept_[0] if hasattr(c, "coef_") else 0.0 for c in lr_models]),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../runs/stageA")
    p.add_argument("--out", default="../runs/core_s1.json")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--fly-epochs", type=int, default=3)
    p.add_argument("--skip", action="store_true", help="GRU with a linear skip path from the current frame")
    p.add_argument("--no-fly", action="store_true", help="skip the mushroom-body core (faster)")
    p.add_argument(
        "--no-posweight",
        action="store_true",
        help="plain BCE for the GRU (the logistic regression has no class weights)",
    )
    p.add_argument("--wd", type=float, default=1e-4)
    main(p.parse_args())
