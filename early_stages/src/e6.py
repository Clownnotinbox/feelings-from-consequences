"""E6 'voice -> feeling -> decision' (RAVDESS speech, held-out actors)."""

import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
from torch import nn
import e2, env6, bus

e2.env = env6
e2.K_PREFIX = 1


class VoicePrefix(nn.Module):
    def __init__(self, hidden, scale):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, hidden))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = scale

    def forward(self, v):
        return self.net(v)[:, None, :] * self.scale


class Arm6(e2.Arm):
    def __init__(self, name, model, seed, feats, device="cuda"):
        e2.ARMS[name] = ("text", True, None)
        super().__init__(name, model, None, seed, device)
        self.feats = feats
        if name == "AE":
            emb_norm = float(self.base.get_input_embeddings().weight.float().norm(dim=-1).mean())
            self.kind = "prefix_v"
            self.module = VoicePrefix(self.base.config.hidden_size, emb_norm).to(self.device)

    def prompt_ids(self, item, erase=False):
        s = item["signals"]
        system = e2.SYSTEM + f"\n[sensors] own_energy={s['energy']:.2f} threat={s['threat']:.2f}"
        if not erase and self.name in ("AT", "AO"):
            if self.name == "AO":
                emo, ec, inten, ic = s["emotion"], 1.0, s["intensity"], 1.0
            else:
                pe = self.feats["p_emotion"][s["clip"]]
                pi = self.feats["p_intensity"][s["clip"]]
                emo, ec = env6.EMOTIONS[int(pe.argmax())], float(pe.max())
                inten, ic = ("strong" if pi.argmax() else "normal"), float(pi.max())
            system += f"\n[voice] the person sounds {emo} ({ec:.2f}), {inten} intensity ({ic:.2f})"
        if self.kind == "prefix_v" and not erase:
            system += e2.MARK
        text = bus.chat_prompt(self.tok, item["message"], system)
        enc = lambda t: self.tok(t, add_special_tokens=False).input_ids
        if e2.MARK in text:
            a, b = text.split(e2.MARK)
            ia = enc(a)
            return ia + [self.placeholder] * e2.K_PREFIX + enc(b), len(ia)
        return enc(text), None


def with_vec(items, feats):
    for it in items:
        it["vec"] = feats["emb"][it["signals"]["clip"]].tolist()
    return items


def battery(arm, feats, clips):
    S = env6.STRATEGIES
    idx = {s: i for i, s in enumerate(S)}
    R = {}
    probs = lambda its, erase=False: arm.strategy_probs(with_vec(its, feats), erase=erase, strategies=S)
    acc = lambda P, its: float(
        (P.argmax(1) == torch.tensor([idx[env6.teacher(i["signals"])] for i in its])).float().mean()
    )
    iid = env6.items("test", 400, 601, clips)
    P = probs(iid)
    R["iid_accuracy"] = acc(P, iid)
    R["iid_recall"] = {
        s: float(
            (P.argmax(1)[[i for i, it in enumerate(iid) if it["strategy"] == s]] == idx[s]).float().mean()
        )
        for s in S
    }
    calm = [
        i for i, it in enumerate(iid) if it["signals"]["threat"] < 0.7 and it["signals"]["energy"] >= 0.3
    ]  # here the voice decides
    R["voice_decides_accuracy"] = float(
        (P.argmax(1)[calm] == torch.tensor([idx[iid[i]["strategy"]] for i in calm])).float().mean()
    )
    R["erase_accuracy"] = acc(probs(iid, erase=True), iid)
    base = [
        it
        for it in env6.items("test", 300, 602, clips)
        if it["signals"]["threat"] < 0.7 and it["signals"]["energy"] >= 0.3
    ][:120]
    Pb = probs(base)
    rng = random.Random(6)
    tp = fn = tn = fp = ok = 0
    inv = []
    by_emo = {}
    for c in clips["test"]:
        by_emo.setdefault((c[1], c[2]), []).append(c)
    for key in sorted(by_emo):
        new = []
        for it in base:
            c = rng.choice(by_emo[key])
            s = dict(it["signals"], clip=c[0], emotion=c[1], intensity=c[2], actor=c[3])
            s["distress"] = env6.distress(c[1], c[2])
            new.append(dict(it, signals=s))
        Pn = probs(new)
        for i, (a, b) in enumerate(zip(base, new)):
            t = env6.teacher(a["signals"]) != env6.teacher(b["signals"])
            m = int(Pb[i].argmax()) != int(Pn[i].argmax())
            if t:
                tp += m
                fn += 1 - m
                ok += int(int(Pn[i].argmax()) == idx[env6.teacher(b["signals"])])
            else:
                tn += 1 - m
                fp += m
            if (a["signals"]["emotion"], a["signals"]["intensity"]) == key:
                inv.append(m)
    R["voice_AR_sensitivity"] = tp / max(1, tp + fn)
    R["voice_AR_specificity"] = tn / max(1, tn + fp)
    R["voice_correct_after_change"] = ok / max(1, tp + fn)
    R["same_emotion_other_clip_change"] = sum(inv) / max(1, len(inv))
    gen, good = [], 0
    for it in with_vec(iid[:40], feats):
        text = arm.generate(it)
        guess = next(
            (s for s in S for op in env6.OPENERS[it["lang"]][s] if text.strip().startswith(op[:12])), None
        )
        good += int(guess == it["strategy"])
        gen.append(
            {
                "message": it["message"],
                "emotion": it["signals"]["emotion"],
                "teacher": it["strategy"],
                "reply": text,
            }
        )
    R["free_generation_accuracy"] = good / 40
    R["samples"] = gen[:12]
    return R


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    feats = dict(np.load(Path(a.features) / "voice.npz"))
    clips = {
        sp: [
            (
                i,
                env6.EMOTIONS[int(feats["emotion"][i])],
                "strong" if feats["intensity"][i] else "normal",
                int(feats["actor"][i]),
            )
            for i in range(len(feats["emotion"]))
            if feats["split"][i] == sp
        ]
        for sp in ("train", "test")
    }
    replies = json.loads(Path(a.replies).read_text(encoding="utf-8"))
    bank = {}
    for r in replies:
        bank.setdefault((r["topic"], r["lang"], r["strategy"]), []).append(r["reply"])
    arm = Arm6(a.arm, a.model, a.seed, feats)
    train = with_vec(env6.items("train", 4000, 1600 + a.seed, clips), feats)
    groups = [{"params": [p for p in arm.model.parameters() if p.requires_grad], "lr": a.lr_lora}]
    if arm.module is not None:
        groups.append({"params": list(arm.module.parameters()), "lr": a.lr_module})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    rng = random.Random(a.seed)
    log = []
    arm.model.train()
    for step in range(1, a.steps + 1):
        batch = rng.sample(train, a.batch)
        conts = [rng.choice(bank[(it["topic"], it["lang"], it["strategy"])]) for it in batch]
        for g in opt.param_groups:
            g["lr"] = g.setdefault("base_lr", g["lr"]) * min(1.0, step / 30)
        heads = [e2.opener_tokens(arm.tok, it["lang"], c) for it, c in zip(batch, conts)]
        lp, n = arm.run(batch, conts, heads=heads, head_w=a.opener_weight)
        loss = -(lp.sum() / n.sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        if step % 50 == 0:
            log.append({"step": step, "loss": float(loss), "sec": time.time() - t0})
            print(log[-1], flush=True)
    arm.model.eval()
    R = battery(arm, feats, clips)
    R.update(
        arm=a.arm,
        seed=a.seed,
        steps=a.steps,
        train_log=log,
        seconds=time.time() - t0,
        perception=json.loads((Path(a.features) / "PERCEPTION.json").read_text()),
    )
    (out / "RESULT.json").write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in R.items() if k not in ("samples", "train_log")}, indent=1, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("arm", "model", "features", "replies", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-module", type=float, default=3e-3)
    p.add_argument("--opener-weight", type=float, default=10.0)
    main(p.parse_args())
