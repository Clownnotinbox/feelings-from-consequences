"""The trained feeling core running inside the simulation, frame by frame."""

import json
import numpy as np
from scipy.special import erf


class FeelingCore:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.mu, self.sd = z["mu"], z["sd"]
        self.cal = z["gru_cal"]
        self.names = [str(n) for n in z["out_names"]]
        g = lambda k: z["gru." + k]
        self.W0, self.b0 = g("inp.0.weight"), g("inp.0.bias")
        self.Wih, self.bih = g("gru.weight_ih_l0"), g("gru.bias_ih_l0")
        self.Whh, self.bhh = g("gru.weight_hh_l0"), g("gru.bias_hh_l0")
        self.Wo, self.bo = g("head.weight"), g("head.bias")
        self.Wl, self.bl = (g("lin.weight"), g("lin.bias")) if "gru.lin.weight" in z.files else (None, None)
        self.H = self.Whh.shape[1]
        self.thr = json.loads(str(z["thr"]))
        self.d_in = int(z["d_in"]) if "d_in" in z.files else len(self.mu)
        self.reset()

    def reset(self):
        self.h = np.zeros(self.H, np.float32)

    def step(self, x):
        u = np.clip((x[: self.d_in] - self.mu) / self.sd, -8, 8)
        a = self.W0 @ u + self.b0
        a = 0.5 * a * (1 + erf(a / np.sqrt(2)))  # GELU (exact)
        gi = self.Wih @ a + self.bih
        gh = self.Whh @ self.h + self.bhh
        H = self.H
        r = 1 / (1 + np.exp(-(gi[:H] + gh[:H])))
        zg = 1 / (1 + np.exp(-(gi[H : 2 * H] + gh[H : 2 * H])))
        n = np.tanh(gi[2 * H :] + r * gh[2 * H :])
        self.h = ((1 - zg) * n + zg * self.h).astype(np.float32)
        logit = self.Wo @ self.h + self.bo
        if self.Wl is not None:
            logit = logit + self.Wl @ u
        p = 1 / (1 + np.exp(-(self.cal[:, 0] * logit + self.cal[:, 1])))  # calibrated (Platt on validation)
        return dict(zip(self.names, p.astype(float)))


class LogregCore:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.mu, self.sd, self.W, self.b = z["mu"], z["sd"], z["lr_W"], z["lr_b"]
        self.names = [str(n) for n in z["out_names"]]
        self.thr = json.loads(str(z["thr"]))
        self.d_in = len(self.mu)

    def reset(self):
        pass

    def step(self, x):
        u = np.clip((x[: self.d_in] - self.mu) / self.sd, -8, 8)
        p = 1 / (1 + np.exp(-(self.W @ u + self.b)))
        return dict(zip(self.names, p.astype(float)))


class MarkerCore:
    """Somatic markers per candidate action."""

    def __init__(self, path, model="gru"):
        z = np.load(path, allow_pickle=False)
        self.model = model
        self.mu, self.sd, self.psd = z["mu"], z["sd"], float(z["psd"])
        self.thr = json.loads(str(z["thr"]))[model]
        self.d_in = len(self.mu)
        if model == "logreg":
            self.W, self.b = z["lr_W"], z["lr_b"]
        else:
            g = lambda k: z["gru." + k]
            self.W0, self.b0 = g("inp.0.weight"), g("inp.0.bias")
            self.Wih, self.bih = g("gru.weight_ih_l0"), g("gru.bias_ih_l0")
            self.Whh, self.bhh = g("gru.weight_hh_l0"), g("gru.bias_hh_l0")
            self.Wo, self.bo = g("head.weight"), g("head.bias")
            self.H = self.Whh.shape[1]
        self.reset()

    def reset(self):
        if self.model != "logreg":
            self.h = np.zeros(self.H, np.float32)

    def step(self, x):
        u = np.clip((x[: self.d_in] - self.mu) / self.sd, -8, 8)
        if self.model == "logreg":
            return 1 / (1 + np.exp(-(self.W @ u + self.b))), None
        a = self.W0 @ u + self.b0
        a = 0.5 * a * (1 + erf(a / np.sqrt(2)))
        gi = self.Wih @ a + self.bih
        gh = self.Whh @ self.h + self.bhh
        H = self.H
        r = 1 / (1 + np.exp(-(gi[:H] + gh[:H])))
        zg = 1 / (1 + np.exp(-(gi[H : 2 * H] + gh[H : 2 * H])))
        n = np.tanh(gi[2 * H :] + r * gh[2 * H :])
        self.h = ((1 - zg) * n + zg * self.h).astype(np.float32)
        o = self.Wo @ self.h + self.bo
        k = len(o) // 2
        return 1 / (1 + np.exp(-o[:k])), o[k:] * self.psd
