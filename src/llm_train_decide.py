"""Stage D: teach the LLM to rely on its feelings.
usage: llm_train_decide.py --data DIRS --core RUNTIME.npz --model PATH --out OUT.json [--conds N,T,F,FT] [--seed 1]
"""

import argparse, json, time
from pathlib import Path
import numpy as np, torch

import llm_decide as L


def batches(D, bs, rng):
    idx = rng.permutation(len(D))
    for i in range(0, len(idx), bs):
        yield [D[j] for j in idx[i : i + bs]]


def train_lora(llm, D, cond, seed, epochs=2, lr=2e-4):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    m = get_peft_model(llm.base, cfg)
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    dig = torch.tensor(llm.digits(), device="cuda")
    steps = 0
    t0 = time.time()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for B in batches(D, 8, rng):
            opt.zero_grad()
            for d in B:  # gradient accumulation one example at a time (fits a 3B model in 8 GB)
                ids = llm.encode(L.prompt(d, cond, d.get("F", {})))
                z = m(ids).logits[0, -1, dig].float()
                lp = torch.log_softmax(z, -1)
                best = torch.tensor([L.OPTIONS.index(o) for o in d["best"]], device="cuda")
                loss = -torch.logsumexp(lp[best], 0)
                (loss / len(B)).backward()
                tot += float(loss)
                n += 1
            opt.step()
            steps += 1
        print(
            f"  lora {cond} seed {seed} epoch {ep} loss {tot / max(n, 1):.3f} {time.time() - t0:.0f}s",
            flush=True,
        )
    m.eval()
    return m


def lesion(F, kind):
    G = dict(F)
    if kind.startswith("drop_"):
        G.pop(kind[5:], None)
    elif kind == "flip_low_energy":
        G["low_energy"] = 1.0 - G["low_energy"]
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--core", required=True)
    ap.add_argument("--core-model", default="gru")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default="N,T,F,FT")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    import core_runtime

    core = (core_runtime.FeelingCore if a.core_model == "gru" else core_runtime.LogregCore)(a.core)
    S = {s: L.load(a.data, s) for s in ("train", "test_familiar", "test_unfamiliar")}
    for D in S.values():
        L.attach_feelings(D, core)
    llm = L.LLM(a.model)
    llm.base = llm.m
    R = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    R["n"] = {k: len(v) for k, v in S.items()}
    for cond in a.conds.split(","):
        key = f"{cond}|seed{a.seed}"
        if key in R:
            continue
        m = train_lora(llm, S["train"], cond, a.seed)
        llm.m = m
        R[key] = {}
        tests = [("", None)] + (
            [
                (k, k)
                for k in ("drop_danger_2", "drop_gps_bad", "drop_lost", "drop_low_energy", "flip_low_energy")
            ]
            if "F" in cond
            else []
        )
        for split in ("test_familiar", "test_unfamiliar"):
            for name, les in tests:
                fn = lambda d, les=les: L.OPTIONS[
                    int(np.argmax(llm.choose(L.prompt(d, cond, lesion(d["F"], les) if les else d["F"]))[0]))
                ]
                R[key][split + (("|" + name) if name else "")] = L.evaluate(S[split], fn)
            print(key, split, json.dumps(R[key][split]), flush=True)
        llm.m = m.unload() if hasattr(m, "unload") else llm.base
        llm.base = llm.m
        Path(a.out).write_text(json.dumps(R, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
