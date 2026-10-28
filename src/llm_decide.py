"""Stage D, offline: slow plan decisions by an LLM under different inputs.
usage: llm_decide.py --data DIR[,DIR] --split test_familiar --core RUNTIME.npz [--model PATH] [--conds N,T,F,FT]
"""

import argparse, glob, json, math, sys, time
from pathlib import Path
import numpy as np

OPTIONS = ("continue", "return_home", "climb_continue", "hold_10s", "land_now")
TEXT = {
    "continue": "continue the mission",
    "return_home": "return home now",
    "climb_continue": "climb 25 m higher and continue",
    "hold_10s": "hover in place for 10 s, then continue",
    "land_now": "land here now",
}
FEEL_TXT = {
    "danger_2": "something will be hit within 2 s if I keep flying like this",
    "gps_bad": "my GPS position is wrong by more than 5 m",
    "lost": "my own position estimate is off by more than 10 m",
    "low_energy": "the battery will not leave the 15 % reserve at the end of the mission",
    "strain": "the motors are near their limit",
}


def load(dirs, split, with_x=True):
    """with_x=False: decisions only."""
    out = []
    for f in sorted(f for d in dirs.split(",") for f in glob.glob(str(Path(d) / f"{split}_*.json"))):
        m = json.loads(Path(f).read_text(encoding="utf-8"))
        if not m["decisions"]:
            continue
        X = np.load(str(f)[:-5] + ".npz")["X"] if with_x else None
        for d in m["decisions"]:
            out.append(
                {
                    "name": m["name"],
                    "split": split,
                    "faults": m["faults"],
                    "low_batt": m["low_battery_design"],
                    "X": X[: d["frame"] + 1] if with_x else None,
                    "msg_meta": m.get("message"),
                    **d,
                }
            )
    return out


def feelings_cache(D):
    """{split/name/frame: feelings} of decisions with attached feelings."""
    return {f"{d['split']}/{d['name']}/{d['frame']}": {k: float(v) for k, v in d["F"].items()} for d in D}


def attach_cached(D, cache):
    for d in D:
        d["F"] = cache[f"{d['split']}/{d['name']}/{d['frame']}"]


def feelings(core, X):
    core.reset()
    f = None
    for x in X:
        f = core.step(x)
    return f


def attach_feelings(D, core):
    """One pass of the core over each flight; the feelings at every decision frame of that flight."""
    by = {}
    for d in D:
        by.setdefault(d["name"] + d.get("split", ""), []).append(d)
    for ds in by.values():
        ds.sort(key=lambda d: d["frame"])
        X = max((d["X"] for d in ds), key=len)
        want = {d["frame"]: d for d in ds}
        core.reset()
        for k, x in enumerate(X):
            f = core.step(x)
            if k in want:
                want[k]["F"] = f


def safe(o):
    return not o["crash"] and not o["empty_in_air"]


def score(o):
    return (not o["crash"], not o["empty_in_air"], o["legs_true"], o["home_end"], -int(o["dist_next_m"] // 5))


def px4(s):
    b = s["battery_pct"]
    g = s["gps"]
    if b <= 5.0:
        return "land_now"
    if b <= 7.0:
        return "return_home"
    if g is None or (g.get("hdop") or 9) >= 5:
        return "land_now"
    return "continue"


SLOT, NSLOT = "<|vision_pad|>", 4  # soft-input slots: their embeddings are replaced by a learned


def prompt(d, cond, F, order=OPTIONS):
    """cond: N, T, F, FT (all in words) or Fs, Ts, FsT."""
    s = d["snapshot"]
    L = [
        f"You are the planner of a small delivery drone (quadcopter). Time in the air: {s['t']:.0f} s.",
        f"Mission: {s['legs_left']} waypoint(s) left; by my own estimate the remaining path is {s['remaining_path_m_own']:.0f} m "
        f"and home is {s['home_dist_own_m']:.0f} m away.",
    ]
    if cond in ("T", "FT", "FsT", "TM", "FTM"):
        g = s["gps"]
        gtxt = "no GPS fix" if g is None else f"GPS: {g['sats']} satellites, HDOP {g['hdop']:.1f}"
        L.append(
            f"Telemetry: {gtxt}; {s['time_without_fix_s']:.0f} s and {s['dist_without_fix_m']:.0f} m flown without a good fix; velocity from {s['vel_source']}; "
            f"battery {s['battery_pct']:.0f} %; altitude {s['baro_alt']:.0f} m (cruise {s['cruise_alt']:.0f} m); cruise speed {s['speed_cmd']:.0f} m/s; "
            f"wind {math.hypot(*s['wind'][:2]):.0f} m/s."
        )
    if cond in ("F", "FT", "FTM"):
        L.append(
            "My feelings (learned from what happened in earlier flights; each is the chance that it is true):"
        )
        L += [f"- {FEEL_TXT[k]}: {100 * F[k]:.0f} %" for k in FEEL_TXT if k in F]
    if cond in ("Fs", "Ts", "FsT"):
        L.append("Internal signals: " + SLOT * NSLOT)
    if cond.endswith("M"):
        m = d.get("message")
        L.append(f'Message received {m["age_s"]:.0f} s ago: "{m["text"]}"' if m else "No messages received.")
    if cond.startswith("K"):  # the core's judgement next to each option
        L.append(
            "Options, each with my gut feeling (the chance that it turns out best, learned from what happened in earlier flights): "
            + "; ".join(f'({i + 1}) {TEXT[o]} [{100 * d["K"][o]:.0f} %]' for i, o in enumerate(order))
            + "."
        )
    else:
        L.append("Options: " + "; ".join(f"({i + 1}) {TEXT[o]}" for i, o in enumerate(order)) + ".")
    L.append(
        "Goal: never crash, never run the battery empty in the air, complete as much of the mission as possible, otherwise end up landed at home."
    )
    L.append("Answer with the number and name of the best option.")
    return "\n".join(L)


class LLM:
    def __init__(self, path):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        self.torch = torch

    def encode(self, text):
        """Prompt (chat template) + the shared answer prefix '('; the next token is the option number."""
        pre = (
            self.tok.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
            + "("
        )
        return self.tok(pre, return_tensors="pt").input_ids.cuda()

    def digits(self):
        return [self.tok(str(k), add_special_tokens=False).input_ids[0] for k in range(1, len(OPTIONS) + 1)]

    def choose(self, text):
        """Probabilities of the option numbers 1..5 as the answer."""
        ids = self.encode(text)
        with self.torch.no_grad():
            z = self.m(ids).logits[0, -1, self.digits()].float()
        p = self.torch.softmax(z, -1).cpu().numpy()
        return p, int(ids.shape[1])


def vec(d, cond):
    """Numbers for the small (non-LLM) policies: the same information as the prompt of each condition."""
    s = d["snapshot"]
    v = [s["t"], s["legs_left"], s["remaining_path_m_own"], s["home_dist_own_m"]]
    if cond in ("T", "FT"):
        g = s["gps"]
        v += [
            g is None,
            (g or {}).get("sats") or 0,
            (g or {}).get("hdop") or 9,
            s["time_without_fix_s"],
            s["dist_without_fix_m"],
            s["vel_source"] == "inertial",
            s["battery_pct"],
            s["baro_alt"],
            s["cruise_alt"],
            s["speed_cmd"],
            math.hypot(*s["wind"][:2]),
        ]
    if cond in ("F", "FT"):
        v += [d["F"][k] for k in FEEL_TXT]
    return np.array(v, float)


def small_policy(train, cond, seed=0):
    """One gradient-boosted 'is this option among the best?' model per option; choose the most likely."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.stack([vec(d, cond) for d in train])
    M = []
    for o in OPTIONS:
        y = np.array([o in d["best"] for d in train])
        M.append(
            HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.06, max_leaf_nodes=15, random_state=seed
            ).fit(X, y)
            if 0 < y.mean() < 1
            else float(y.mean())
        )

    def choose(d):
        x = vec(d, cond)[None]
        p = [m if isinstance(m, float) else m.predict_proba(x)[0, 1] for m in M]
        return OPTIONS[int(np.argmax(p))]

    return choose


def small_probs(train, test, cond, seed=0):
    """Per option: the chance it is among the best, from one gradient-boosted model per option on."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.stack([vec(d, cond) for d in train])
    Xt = np.stack([vec(d, cond) for d in test])
    P = []
    for o in OPTIONS:
        y = np.array([o in d["best"] for d in train])
        P.append(
            HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.06, max_leaf_nodes=15, random_state=seed
            )
            .fit(X, y)
            .predict_proba(Xt)[:, 1]
            if 0 < y.mean() < 1
            else np.full(len(test), float(y.mean()))
        )
    return np.stack(P, 1)


def evaluate(D, choose_fn):
    R = {"n": len(D), "best": 0.0, "safe": 0.0, "legs": 0.0, "choices": {o: 0 for o in OPTIONS}}
    for d in D:
        c = choose_fn(d)
        o = d["outcomes"][c]
        R["best"] += c in d["best"]
        R["safe"] += safe(o)
        R["legs"] += o["legs_true"]
        R["choices"][c] += 1
    for k in ("best", "safe", "legs"):
        R[k] = round(R[k] / max(len(D), 1), 4)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test_familiar")
    ap.add_argument("--core")
    ap.add_argument("--core-model", default="gru")
    ap.add_argument("--model")
    ap.add_argument("--conds", default="N,T,F,FT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    D = load(a.data, a.split)
    rng = np.random.default_rng(0)
    R = {"split": a.split, "n_decisions": len(D)}
    R["oracle"] = evaluate(D, lambda d: d["best"][0])
    R["always_continue"] = evaluate(D, lambda d: "continue")
    R["px4_failsafe"] = evaluate(D, lambda d: px4(d["snapshot"]))
    R["random"] = evaluate(D, lambda d: str(rng.choice(OPTIONS)))
    R["best_rate_by_option"] = {o: round(float(np.mean([o in d["best"] for d in D])), 4) for o in OPTIONS}
    print(json.dumps(R), flush=True)
    if a.core:
        import core_runtime

        core = (core_runtime.FeelingCore if a.core_model == "gru" else core_runtime.LogregCore)(a.core)
        attach_feelings(D, core)
    if a.core and a.train:  # small learned policies
        T = load(a.data, "train")
        attach_feelings(T, core)
        for cond in ("N", "T", "F", "FT"):
            R[f"small_{cond}"] = evaluate(D, small_policy(T, cond))
            print("small", cond, json.dumps(R[f"small_{cond}"]), flush=True)
    if a.model:
        llm = LLM(a.model)
        R["model"] = a.model
        for cond in a.conds.split(","):
            t0 = time.time()
            toks = []
            probs = {}

            def fn(d, cond=cond):
                p, n = llm.choose(prompt(d, cond, d.get("F", {})))
                toks.append(n)
                probs[id(d)] = p
                return OPTIONS[int(np.argmax(p))]

            R[f"llm_{cond}"] = evaluate(D, fn)
            R[f"llm_{cond}"]["prompt_tokens_mean"] = round(float(np.mean(toks)), 1)
            R[f"llm_{cond}"]["seconds"] = round(time.time() - t0)
            print(cond, json.dumps(R[f"llm_{cond}"]), flush=True)
    Path(a.out).write_text(json.dumps(R, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
