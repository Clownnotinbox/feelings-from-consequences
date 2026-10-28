"""E7 'where am I': does the reply depend on the perceived place/sound, and through which channel?."""

import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
from torch import nn
import e2, env7, bus

e2.env = env7  # openers/strategies of the 6-way task
e2.K_PREFIX = 2  # VE: one prefix token per modality (photo, sound)

ARMS7 = ("V0", "VT", "VE", "VO")


class EmbPrefix(nn.Module):
    def __init__(self, hidden, scale):
        super().__init__()
        self.img = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, hidden))
        self.snd = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, hidden))
        for n in (self.img, self.snd):
            nn.init.normal_(n[-1].weight, std=1e-3)
            nn.init.zeros_(n[-1].bias)
        self.scale = scale

    def forward(self, v):
        return torch.stack([self.img(v[:, :512]), self.snd(v[:, 512:])], 1) * self.scale


class Arm7(e2.Arm):
    def __init__(self, name, model, seed, feats, device="cuda"):
        e2.ARMS[name] = ("text", True, None)
        super().__init__(name, model, None, seed, device)
        self.feats = feats
        if name == "VE":
            emb_norm = float(self.base.get_input_embeddings().weight.float().norm(dim=-1).mean())
            self.kind = "prefix_e"
            self.module = EmbPrefix(self.base.config.hidden_size, emb_norm).to(self.device)

    def prompt_ids(self, item, erase=False):
        s = item["signals"]
        system = e2.SYSTEM + f"\n[sensors] other_distress={s['distress']:.2f} own_energy={s['energy']:.2f}"
        if not erase and self.name in ("VT", "VO"):
            if self.name == "VO":
                place, snd = s["place"], s["sound"]
                pc = sc = 1.0
            else:
                pi = self.feats["img"]["probs"][s["img"]]
                si = self.feats["snd"]["probs"][s["snd"]]
                place, pc = env7.PLACES[int(pi.argmax())], float(pi.max())
                snd, sc = env7.SOUNDS[int(si.argmax())], float(si.max())
            system += (
                f"\n[scene] {place.replace('_', ' ')} ({pc:.2f}); [sound] {snd.replace('_', ' ')} ({sc:.2f})"
            )
        if self.kind == "prefix_e" and not erase:
            system += e2.MARK
        text = bus.chat_prompt(
            self.tok, item["message"] + ((" " + item["extra"]) if item.get("extra") else ""), system
        )
        enc = lambda t: self.tok(t, add_special_tokens=False).input_ids
        if e2.MARK in text:
            a, b = text.split(e2.MARK)
            ia = enc(a)
            return ia + [self.placeholder] * e2.K_PREFIX + enc(b), len(ia)
        return enc(text), None


def with_vec(items, feats):
    for it in items:
        s = it["signals"]
        it["vec"] = np.concatenate([feats["img"]["emb"][s["img"]], feats["snd"]["emb"][s["snd"]]]).tolist()
    return items


def battery(arm, feats, bi, bs):
    S = env7.STRATEGIES
    idx = {s: i for i, s in enumerate(S)}
    R = {}
    probs = lambda its, erase=False: arm.strategy_probs(with_vec(its, feats), erase=erase, strategies=S)
    acc = lambda P, its: float(
        (P.argmax(1) == torch.tensor([idx[env7.teacher(i["signals"])] for i in its])).float().mean()
    )
    iid = env7.items("test", 400, 701, bi, bs)
    P = probs(iid)
    R["iid_accuracy"] = acc(P, iid)
    R["iid_recall"] = {
        s: float(
            (P.argmax(1)[[i for i, it in enumerate(iid) if it["strategy"] == s]] == idx[s]).float().mean()
        )
        for s in S
    }
    R["erase_accuracy"] = acc(probs(iid, erase=True), iid)
    base = env7.items("test", 120, 702, bi, bs)
    Pb = probs(base)
    rng = random.Random(5)
    for factor, values, key, bank in (("place", env7.PLACES, "img", bi), ("sound", env7.SOUNDS, "snd", bs)):
        tp = fn = tn = fp = ok = 0
        inv = []
        for v in values:
            new = [
                dict(it, signals=dict(it["signals"], **{factor: v, key: rng.choice(bank["test"][v])}))
                for it in base
            ]
            Pn = probs(new)
            for i, (a, b) in enumerate(zip(base, new)):
                t = env7.teacher(a["signals"]) != env7.teacher(b["signals"])
                m = int(Pb[i].argmax()) != int(Pn[i].argmax())
                if t:
                    tp += m
                    fn += 1 - m
                    ok += int(int(Pn[i].argmax()) == idx[env7.teacher(b["signals"])])
                else:
                    tn += 1 - m
                    fp += m
                if a["signals"][factor] == v:
                    inv.append(m)  # same category, different photo/clip
        R[f"{factor}_AR_sensitivity"] = tp / max(1, tp + fn)
        R[f"{factor}_AR_specificity"] = tn / max(1, tn + fp)
        R[f"{factor}_correct_after_change"] = ok / max(1, tp + fn)
        R[f"{factor}_same_category_new_sample_change"] = sum(inv) / max(1, len(inv))
    gen, good = [], 0
    for it in with_vec(iid[:40], feats):
        text = arm.generate(it)
        guess = next(
            (s for s in S for op in env7.OPENERS[it["lang"]][s] if text.strip().startswith(op[:12])), None
        )
        good += int(guess == it["strategy"])
        gen.append(
            {
                "message": it["message"],
                "place": it["signals"]["place"],
                "sound": it["signals"]["sound"],
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
    feats = {k: dict(np.load(Path(a.features) / f"{k}.npz")) for k in ("img", "snd")}
    bank_of = lambda k, names: {
        sp: {
            n: [i for i in range(len(feats[k]["y"])) if feats[k]["split"][i] == sp and feats[k]["y"][i] == j]
            for j, n in enumerate(names)
        }
        for sp in ("train", "test")
    }
    bi, bs = bank_of("img", env7.PLACES), bank_of("snd", env7.SOUNDS)
    replies = json.loads(Path(a.replies).read_text(encoding="utf-8"))
    bank = {}
    for r in replies:
        bank.setdefault((r["topic"], r["lang"], r["strategy"]), []).append(r["reply"])
    arm = Arm7(a.arm, a.model, a.seed, feats)
    train = with_vec(env7.items("train", 4000, 1700 + a.seed, bi, bs), feats)
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
    R = battery(arm, feats, bi, bs)
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
