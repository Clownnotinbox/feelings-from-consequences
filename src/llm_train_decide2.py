"""Stage D, v2 of teaching the LLM to rely on its feelings.
usage: llm_train_decide2.py --data DIRS --core RUNTIME.npz --model PATH --out OUT.json [--conds N,T,F,FT,Fs,Ts,FsT] [--seed 1]
"""

import argparse, gc, json, math, time, zlib
from pathlib import Path
import numpy as np, torch

import llm_decide as L

FEELS = tuple(L.FEEL_TXT)
LESIONS = tuple(f"drop_{k}" for k in FEELS) + ("flip_low_energy", "shuffle_all")
VAL_PCT = 15


def soft_kind(cond):
    return {"Fs": "F", "FsT": "F", "Ts": "T"}.get(cond)


def raw_vec(d, kind, F=None):
    if kind == "F":
        p = np.clip([(F or d["F"])[k] for k in FEELS], 1e-3, 1 - 1e-3)
        return np.log(p / (1 - p))
    return L.vec(d, "T")[4:]  # the 11 telemetry numbers of condition T


def best_mask(d, order):
    return [float(o in d["best"]) for o in order]


def lesioned(d, cond, les, other, mu):
    """What this decision is shown under a lesion: the feelings dict for words, the raw vector for slots."""
    F = dict(d["F"])
    kind = soft_kind(cond)
    if les == "shuffle_all":
        F = dict(other["F"])
    elif les == "flip_low_energy":
        F["low_energy"] = 1.0 - F["low_energy"]
    if kind == "F":
        v = raw_vec(d, "F", F)
        if les and les.startswith("drop_"):
            k = FEELS.index(les[5:])
            v[k] = mu[k]
        return F, v
    if les and les.startswith("drop_"):
        F.pop(les[5:], None)
    return F, (raw_vec(d, kind) if kind else None)


def attach_k(S, folds=5):
    """d['K']: the core's judgement of each option."""
    tr = S["train"]
    fold = lambda d: zlib.crc32((d["name"] + "/k").encode()) % folds
    for k in range(folds):
        a = [d for d in tr if fold(d) != k]
        b = [d for d in tr if fold(d) == k]
        for d, p in zip(b, L.small_probs(a, b, "FT")):
            d["K"] = dict(zip(L.OPTIONS, map(float, p)))
    for s in ("test_familiar", "test_unfamiliar"):
        for d, p in zip(S[s], L.small_probs(tr, S[s], "FT")):
            d["K"] = dict(zip(L.OPTIONS, map(float, p)))


class Proj(torch.nn.Module):
    """vector -> NSLOT embeddings with the RMS of ordinary token embeddings."""

    def __init__(self, d_in, H, rms, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 64), torch.nn.GELU(), torch.nn.Linear(64, L.NSLOT * H)
        )
        self.H, self.rms = H, rms
        self.gain = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        e = self.net((x - self.mu) / self.sd).view(-1, L.NSLOT, self.H)
        return e / e.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * self.rms * self.gain


class Runner:
    def __init__(self, path, cond, train, seed, ckpt=True):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "right"
        base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
        base.config.use_cache = False
        if ckpt:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )  # 8 GB GPUs
        cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        self.m = get_peft_model(base, cfg)
        self.inner = self.m.get_base_model().model
        self.emb = self.m.get_input_embeddings()
        self.head = self.m.get_output_embeddings().weight
        self.dig = torch.tensor(
            [self.tok(str(k), add_special_tokens=False).input_ids[0] for k in range(1, len(L.OPTIONS) + 1)],
            device="cuda",
        )
        self.slot = self.tok.convert_tokens_to_ids(L.SLOT)
        self.cond = cond
        self.kind = soft_kind(cond)
        self.proj = None
        if self.kind:
            V = np.stack([raw_vec(d, self.kind) for d in train])
            rms = float(self.emb.weight.float().pow(2).mean().sqrt())
            self.proj = Proj(V.shape[1], self.emb.weight.shape[1], rms, V.mean(0), V.std(0) + 1e-6).cuda()

    def logits(self, texts, vecs=None):
        pre = [
            self.tok.apply_chat_template(
                [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
            )
            + "("
            for t in texts
        ]
        b = self.tok(pre, return_tensors="pt", padding=True)
        ids, mask = b.input_ids.cuda(), b.attention_mask.cuda()
        E = self.emb(ids)
        if self.proj is not None:
            pos = ids == self.slot
            assert bool((pos.sum(1) == L.NSLOT).all())
            s = self.proj(torch.tensor(np.stack(vecs), dtype=torch.float32, device="cuda")).to(E.dtype)
            E = E.masked_scatter(pos.unsqueeze(-1).expand_as(E), s.reshape(-1))
        h = self.inner(inputs_embeds=E, attention_mask=mask).last_hidden_state
        return (
            h[torch.arange(len(texts), device="cuda"), mask.sum(1) - 1].float()
            @ self.head[self.dig].float().T
        )

    @staticmethod
    def loss(z, masks):
        m = torch.tensor(masks, device=z.device)
        lp = torch.log_softmax(z, -1)
        return -torch.logsumexp(lp.masked_fill(m == 0, -math.inf), 1).mean()

    def params(self):
        g = [{"params": [p for p in self.m.parameters() if p.requires_grad], "lr": 2e-4}]
        return g + ([{"params": list(self.proj.parameters()), "lr": 1e-3}] if self.proj is not None else [])

    def state(self):
        s = {k: v.detach().cpu().clone() for k, v in self.m.state_dict().items() if "lora_" in k}
        return s, (
            {k: v.detach().cpu().clone() for k, v in self.proj.state_dict().items()}
            if self.proj is not None
            else None
        )

    def load_state(self, st):
        self.m.load_state_dict(st[0], strict=False)
        if st[1] is not None:
            self.proj.load_state_dict(st[1])

    def score(self, D, bs=16):
        """Validation loss and best rate in the canonical option order."""
        self.m.eval()
        tot, hit = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(D), bs):
                B = D[i : i + bs]
                z = self.logits(
                    [L.prompt(d, self.cond, d["F"]) for d in B],
                    [raw_vec(d, self.kind) for d in B] if self.kind else None,
                )
                tot += float(self.loss(z, [best_mask(d, L.OPTIONS) for d in B])) * len(B)
                hit += sum(L.OPTIONS[int(k)] in d["best"] for k, d in zip(z.argmax(1).cpu(), B))
        return tot / len(D), hit / len(D)

    def fit(self, train, val, seed, epochs=8, bs=8, mbs=8, patience=0):  # patience 0: no early stop
        rng = np.random.default_rng(seed)
        opt = torch.optim.AdamW(self.params(), weight_decay=0.0)
        total = max(1, epochs * math.ceil(len(train) / bs))
        warm = max(1, total // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, s / total)))),
        )
        trainable = [p for g in self.params() for p in g["params"]]
        vl0, vb0 = self.score(val)
        hist = [{"epoch": -1, "val_loss": round(vl0, 4), "val_best": round(vb0, 4)}]
        best = (vl0, self.state(), -1)
        bad = 0
        t0 = time.time()
        for ep in range(epochs):
            self.m.train()
            idx = rng.permutation(len(train))
            tot = 0.0
            for i0 in range(0, len(idx), bs):
                B = [train[j] for j in idx[i0 : i0 + bs]]
                opt.zero_grad()
                for k0 in range(0, len(B), mbs):
                    mb = B[k0 : k0 + mbs]
                    orders = [tuple(L.OPTIONS[i] for i in rng.permutation(len(L.OPTIONS))) for _ in mb]
                    z = self.logits(
                        [L.prompt(d, self.cond, d["F"], o) for d, o in zip(mb, orders)],
                        [raw_vec(d, self.kind) for d in mb] if self.kind else None,
                    )
                    loss = self.loss(z, [best_mask(d, o) for d, o in zip(mb, orders)])
                    (loss * len(mb) / len(B)).backward()
                    tot += float(loss) * len(mb)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
            vl, vb = self.score(val)
            hist.append(
                {
                    "epoch": ep,
                    "train_loss": round(tot / len(train), 4),
                    "val_loss": round(vl, 4),
                    "val_best": round(vb, 4),
                    "s": round(time.time() - t0),
                }
            )
            print(f"  {self.cond} seed {seed} {json.dumps(hist[-1])}", flush=True)
            if vl < best[0] - 1e-4:
                best = (vl, self.state(), ep)
                bad = 0
            else:
                bad += 1
                if patience and bad >= patience:
                    break
        self.load_state(best[1])
        return {"history": hist, "chosen_epoch": best[2], "val_loss": round(best[0], 4)}

    def predict(self, D, les=None, other=None, bs=16):
        self.m.eval()
        out = []
        mu = self.proj.mu.cpu().numpy() if self.proj is not None else None
        with torch.no_grad():
            for i in range(0, len(D), bs):
                T, V = [], []
                for j, d in enumerate(D[i : i + bs]):
                    o = other[i + j] if other else None
                    if les == "drop_message":
                        d = {**d, "message": None}
                    elif les == "shuffle_K":
                        d = {**d, "K": o["K"]}
                    elif les == "novel_text" and d.get("message"):
                        d = {
                            **d,
                            "message": {**d["message"], "text": d["message"]["text_novel"]},
                        }  # D3: the same event in never-trained words
                    F, v = lesioned(d, self.cond, les, o, mu)
                    T.append(L.prompt(d, self.cond, F))
                    V.append(v)
                out += [L.OPTIONS[int(k)] for k in self.logits(T, V if self.kind else None).argmax(1).cpu()]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--core", required=True)
    ap.add_argument("--core-model", default="gru")  # the core's file name names the feelings
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default="N,T,F,FT,Fs,Ts,FsT")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--mbs", type=int, default=8)
    ap.add_argument("--ckpt", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)  # smoke test: only the first N decisions of every split
    ap.add_argument("--feelings")  # a cache from export_d_feelings.py instead of running the core here
    a = ap.parse_args()
    S = {s: L.load(a.data, s, with_x=not a.feelings) for s in ("train", "test_familiar", "test_unfamiliar")}
    if a.limit:
        S = {k: v[: a.limit] for k, v in S.items()}
    if a.feelings:
        cache = json.loads(Path(a.feelings).read_text(encoding="utf-8"))
        assert cache["core"] == f"{Path(a.core).name}|{a.core_model}", cache["core"]
        for D in S.values():
            L.attach_cached(D, cache["F"])
    else:
        import core_runtime

        core = (core_runtime.FeelingCore if a.core_model == "gru" else core_runtime.LogregCore)(a.core)
        for D in S.values():
            L.attach_feelings(D, core)
    if any(c.startswith("K") for c in a.conds.split(",")):
        attach_k(S)
    is_val = lambda d: zlib.crc32(d["name"].encode()) % 100 < VAL_PCT
    tr = [d for d in S["train"] if not is_val(d)]
    va = [d for d in S["train"] if is_val(d)]
    R = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    R["n"] = {**{k: len(v) for k, v in S.items()}, "train_fit": len(tr), "val": len(va)}
    R["core"] = f"{Path(a.core).name}|{a.core_model}"
    R["model"] = a.model
    for cond in a.conds.split(","):
        key = f"{cond}|seed{a.seed}"
        if key in R:
            continue
        run = Runner(a.model, cond, tr, a.seed, bool(a.ckpt))
        R[key] = {"fit": run.fit(tr, va, a.seed, a.epochs, mbs=a.mbs)}
        d3 = any("text_novel" in (d.get("message") or {}) for d in S["test_familiar"])
        tests = (
            [None]
            + (list(LESIONS) if "F" in cond else [])
            + (["drop_message"] + (["novel_text"] if d3 else []) if cond.endswith("M") else [])
            + (["shuffle_K"] if cond.startswith("K") else [])
        )
        for split in ("test_familiar", "test_unfamiliar"):
            D = S[split]
            p = np.random.default_rng(0).permutation(len(D))
            other = [None] * len(D)
            for i in range(len(D)):
                other[p[i]] = D[p[(i + 1) % len(D)]]  # a cyclic shift: never a decision's own feelings
            for les in tests:
                picks = run.predict(D, les, other)
                it = iter(picks)
                r = L.evaluate(D, lambda d: next(it))
                r["picks"] = "".join(str(L.OPTIONS.index(c)) for c in picks)
                R[key][split + (f"|{les}" if les else "")] = r
            print(
                key,
                split,
                json.dumps({k: R[key][split][k] for k in ("best", "safe", "legs", "choices")}),
                flush=True,
            )
        Path(a.out).write_text(json.dumps(R, indent=1), encoding="utf-8")
        del run
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
