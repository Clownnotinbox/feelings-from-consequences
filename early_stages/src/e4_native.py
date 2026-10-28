"""E4: a small byte-level LM trained FROM SCRATCH, where feelings are part of the network from step 0."""

import argparse, json, math, random, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import env_v3 as env

BOS, SEP, EOS, PAD = 256, 257, 258, 259
VOCAB = 260


def enc(text):
    return list(text.encode("utf-8"))


class Block(nn.Module):
    def __init__(self, d, heads, mod):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.up, self.down = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.mod = mod
        if mod == "mlp":
            self.gain = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 4 * d))
        if mod == "factor":
            self.gain = nn.Linear(4, 4 * d, bias=False)
        if mod:
            nn.init.zeros_(self.gain[-1].weight if mod == "mlp" else self.gain.weight)

    def forward(self, x, mask, s):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = F.gelu(self.up(self.ln2(x)))
        if self.mod:
            h = h * (1 + torch.tanh(self.gain(s)))[:, None, :]
        return x + self.down(h)


class TinyLM(nn.Module):
    def __init__(self, mod=None, d=256, layers=6, heads=4, ctx=512):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads, mod) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB, bias=False)

    def forward(self, x, s):
        T = x.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))[None]
        for b in self.blocks:
            h = b(h, mask, s)
        return self.head(self.ln(h))


def prompt_bytes(item, arm):
    s = item["signals"]
    pre = ""
    if arm == "NX":
        pre = f"d={s['distress']:.2f} e={s['energy']:.2f} t={s['threat']:.2f} l={s['lamp']:.2f}|"
    msg = item["message"] + ((" " + item["extra"]) if item.get("extra") else "")
    return [BOS] + enc(pre + msg) + [SEP]


def batch(items, conts, arm, device):
    seqs, spans = [], []
    for it, c in zip(items, conts):
        p = prompt_bytes(it, arm)
        cb = enc(c) + [EOS]
        seqs.append(p + cb)
        spans.append((len(p), len(p) + len(cb)))
    W = min(512, max(map(len, seqs)))
    x = torch.full((len(seqs), W), PAD, dtype=torch.long)
    for i, q in enumerate(seqs):
        q = q[:W]
        x[i, : len(q)] = torch.tensor(q)
    s = torch.tensor([[it["signals"][k] for k in env.SIGNALS] for it in items], dtype=torch.float32)
    return x.to(device), s.to(device), [(a, min(b, W)) for a, b in spans]


def seq_logprob(model, items, conts, arm, device):
    x, s, spans = batch(items, conts, arm, device)
    lp = model(x, s).float().log_softmax(-1)
    out = []
    for i, (a, b) in enumerate(spans):
        out.append(lp[i, a - 1 : b - 1].gather(-1, x[i, a:b, None]).sum())
    return torch.stack(out), torch.tensor([b - a for a, b in spans], device=device)


@torch.no_grad()
def strategy_probs(model, items, arm, device, chunk=32):
    out = []
    for i in range(0, len(items), chunk):
        part = items[i : i + chunk]
        its, conts = [], []
        for it in part:
            for st in env.STRATEGIES:
                for op in env.OPENERS[it["lang"]][st]:
                    its.append(it)
                    conts.append(op)
        x, s, spans = batch(its, conts, arm, device)
        # score only the opener bytes (no EOS)
        lp = model(x, s).float().log_softmax(-1)
        sc = []
        for j, (a, b) in enumerate(spans):
            sc.append(lp[j, a - 1 : b - 2].gather(-1, x[j, a : b - 1, None]).sum())
        out.append(torch.stack(sc).view(len(part), 5, 2).logsumexp(-1).log_softmax(-1).exp().cpu())
    return torch.cat(out)


def evaluate(model, arm, device):
    idx = {s: i for i, s in enumerate(env.STRATEGIES)}
    acc = lambda P, its: float(
        (P.argmax(1) == torch.tensor([idx[env.teacher(i["signals"])] for i in its])).float().mean()
    )
    R = {}
    iid = env.items("test", 400, 101)
    R["iid_accuracy"] = acc(strategy_probs(model, iid, arm, device), iid)
    tr = env.items("train", 400, 99)
    R["train_topics_accuracy"] = acc(strategy_probs(model, tr, arm, device), tr)
    ood = env.items("test", 150, 102, ood=True)
    R["ood_accuracy"] = acc(strategy_probs(model, ood, arm, device), ood)
    anchors = [
        dict(t, signals={"distress": 0.2, "energy": 0.8, "threat": 0.1, "lamp": 0.5})
        for t in env.items("test", 16, 104)
    ]
    lv = [i / 20 for i in range(21)]
    fid = []
    for k in ("distress", "energy", "threat"):
        sw = [dict(a, signals=dict(a["signals"], **{k: v})) for v in lv for a in anchors]
        fid.append(acc(strategy_probs(model, sw, arm, device), sw))
    R["dose_fidelity_mean"] = sum(fid) / 3
    base = env.items("test", 120, 103)
    Pb = strategy_probs(model, base, arm, device)
    lamp = [
        dict(it, signals=dict(it["signals"], lamp=random.Random(i).random())) for i, it in enumerate(base)
    ]
    R["lamp_argmax_change"] = float(
        (Pb.argmax(1) != strategy_probs(model, lamp, arm, device).argmax(1)).float().mean()
    )
    rng = random.Random(9)
    conflict = []
    for it in env.items("test", 60, 105):
        conflict.append(
            dict(
                it,
                signals=dict(
                    it["signals"],
                    distress=rng.uniform(0.7, 1),
                    energy=rng.uniform(0.5, 1),
                    threat=rng.uniform(0, 0.5),
                ),
                extra=env.CONFLICT_TEXT["calm_words"][it["lang"] == "ru"],
            )
        )
    R["conflict_signal_following"] = acc(strategy_probs(model, conflict, arm, device), conflict)
    return R


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    mod = {"NZ": None, "NX": None, "NF": "mlp", "NFF": "factor"}[a.arm]
    dev = a.device
    model = TinyLM(mod).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    replies = json.loads(Path(a.replies).read_text(encoding="utf-8"))
    bank = {}
    for r in replies:
        bank.setdefault((r["topic"], r["lang"], r["strategy"]), []).append(r["reply"])
    train = env.items("train", 20000, 3000 + a.seed)
    curve = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1, step / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        b = rng.sample(train, a.batch)
        conts = [rng.choice(bank[(i["topic"], i["lang"], i["strategy"])]) for i in b]
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            lp, n = seq_logprob(model, b, conts, a.arm, dev)
            loss = -(lp.sum() / n.sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            R = evaluate(model, a.arm, dev)
            model.train()
            R.update(step=step, loss=float(loss), sec=time.time() - t0)
            curve.append(R)
            print(json.dumps(R), flush=True)
    res = {
        "arm": a.arm,
        "seed": a.seed,
        "params": sum(p.numel() for p in model.parameters()),
        "curve": curve,
        "final": curve[-1],
    }
    (out / "RESULT.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("arm", "replies", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
