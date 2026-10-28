"""Candidate fast actions and their geometry from the brain-side features."""

import math
import numpy as np

ACTIONS = ("keep", "hover", "climb", "descend", "left", "right", "back")
NA = len(ACTIONS)
SECT0 = 13 + 3 + 2 + 6 + 6  # lidar sector minima
CMD0 = 95 + 2  # efference copy: cmd vx, vy, vz, yaw rate, ...


def geometry_scores(X):
    """Baseline: for each action, 2 s /."""
    S = X[:, SECT0 : SECT0 + 10] * 40.0
    cv = X[:, CMD0 : CMD0 + 3] * np.array([10.0, 10.0, 4.0])
    out = np.zeros((len(X), NA), np.float32)
    yaw = np.arctan2(X[:, 8], X[:, 9])  # sin, cos of yaw: features 8, 9
    for j, a in enumerate(ACTIONS):
        if a == "keep":
            v = cv
        elif a == "hover":
            v = np.zeros_like(cv)
        elif a == "climb":
            v = np.stack([0.3 * cv[:, 0], 0.3 * cv[:, 1], np.full(len(X), 3.0)], 1)
        elif a == "descend":
            v = np.stack([0.3 * cv[:, 0], 0.3 * cv[:, 1], np.full(len(X), -2.0)], 1)
        else:
            sp = np.linalg.norm(cv[:, :2], axis=1)
            hx = np.where(sp > 0.3, cv[:, 0] / np.maximum(sp, 1e-6), np.cos(yaw))
            hy = np.where(sp > 0.3, cv[:, 1] / np.maximum(sp, 1e-6), np.sin(yaw))
            v = {
                "left": np.stack([-6 * hy, 6 * hx], 1),
                "right": np.stack([6 * hy, -6 * hx], 1),
                "back": np.stack([-6 * hx, -6 * hy], 1),
            }[a]
            v = np.hstack([v, np.zeros((len(X), 1))])
        sp_h = np.linalg.norm(v[:, :2], axis=1)
        hd = np.arctan2(v[:, 1], v[:, 0]) - yaw
        k0 = ((hd % (2 * math.pi)) // (math.pi / 4)).astype(int) % 8
        d = np.where(
            sp_h > 0.3, np.minimum.reduce([S[np.arange(len(X)), (k0 + o) % 8] for o in (-1, 0, 1)]), 40.0
        )
        d = np.where(v[:, 2] > 0.3, np.minimum(d, S[:, 8]), d)
        d = np.where(v[:, 2] < -0.3, np.minimum(d, S[:, 9]), d)
        out[:, j] = np.clip(
            2.0 * np.maximum(np.linalg.norm(v, axis=1), 0.1) / np.maximum(d, 0.1), 0, 10
        )  # >= 1 means 'reach it within 2 s'
    return out


def action_features(X):
    """Per action (rows x 7 x 6)."""
    S = X[:, SECT0 : SECT0 + 10] * 40.0
    cv = X[:, CMD0 : CMD0 + 3] * np.array([10.0, 10.0, 4.0])
    yaw = np.arctan2(X[:, 8], X[:, 9])
    n = len(X)
    sp = np.linalg.norm(cv[:, :2], axis=1)
    hx = np.where(sp > 0.3, cv[:, 0] / np.maximum(sp, 1e-6), np.cos(yaw))
    hy = np.where(sp > 0.3, cv[:, 1] / np.maximum(sp, 1e-6), np.sin(yaw))
    V = {
        "keep": cv,
        "hover": np.zeros_like(cv),
        "climb": np.stack([0.3 * cv[:, 0], 0.3 * cv[:, 1], np.full(n, 3.0)], 1),
        "descend": np.stack([0.3 * cv[:, 0], 0.3 * cv[:, 1], np.full(n, -2.0)], 1),
        "left": np.stack([-6 * hy, 6 * hx, np.zeros(n)], 1),
        "right": np.stack([6 * hy, -6 * hx, np.zeros(n)], 1),
        "back": np.stack([-6 * hx, -6 * hy, np.zeros(n)], 1),
    }
    G = geometry_scores(X)
    out = np.zeros((n, NA, 6), np.float32)
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    for j, a in enumerate(ACTIONS):
        v = V[a]
        fx, fy = c * v[:, 0] - s_ * v[:, 1], s_ * v[:, 0] + c * v[:, 1]
        spd = np.linalg.norm(v, axis=1)
        d = np.where(G[:, j] > 0, 2.0 * np.maximum(spd, 0.1) / np.maximum(G[:, j], 1e-3), 40.0)
        out[:, j] = np.stack(
            [
                fx / 6.0,
                fy / 6.0,
                v[:, 2] / 3.0,
                spd / 6.0,
                np.minimum(d, 40.0) / 40.0,
                np.minimum(G[:, j], 10.0) / 10.0,
            ],
            1,
        )
    return out


def pairs(E, key="Xn"):
    """(state, candidate action) rows for the forward model."""
    Z, C, P = [], [], []
    for e in E:
        Xs = e[key][e["fr"]]
        A = action_features(e["X"][e["fr"]])
        I = np.eye(NA, dtype=np.float32)
        for j in range(NA):
            Z.append(np.hstack([Xs, np.repeat(I[j][None], len(Xs), 0), A[:, j]]))
            C.append(e["C"][:, j])
            P.append(e["P"][:, j])
    return np.concatenate(Z), np.concatenate(C), np.concatenate(P)


def fwd_predict(model, e, key="Xn"):
    Xs = e[key][e["fr"]]
    A = action_features(e["X"][e["fr"]])
    I = np.eye(NA, dtype=np.float32)
    return np.stack(
        [
            model.predict_proba(np.hstack([Xs, np.repeat(I[j][None], len(Xs), 0), A[:, j]]))[:, 1]
            for j in range(NA)
        ],
        1,
    )


class FwdMarkerCore:
    """In-flight runtime of the forward-model markers."""

    def __init__(self, path):
        import pickle

        with open(path, "rb") as f:
            b = pickle.load(f)
        self.mu, self.sd, self.fwd, self.cho, self.thr = b["mu"], b["sd"], b["fwd"], b["cho"], b["thr"]
        self.d_in = len(self.mu)

    def reset(self):
        pass

    def step(self, x):
        x = x[: self.d_in]
        xn = np.clip((x - self.mu) / self.sd, -8, 8)[None].astype(np.float32)
        A = action_features(x[None])[0]
        I = np.eye(NA, dtype=np.float32)
        Z = np.hstack([np.repeat(xn, NA, 0), I, A])
        keep = self.fwd.predict_proba(Z[:1])[:, 1]
        alt = self.cho.predict_proba(Z[1:])[:, 1]
        return np.r_[keep, alt], None
