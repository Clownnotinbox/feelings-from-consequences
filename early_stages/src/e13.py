"""E13 'a feeling without a name'."""

import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
import bus, e2, e6, env6

INT_EMOS = ("surprised", "disgust", "angry", "sad", "fearful")


def distress(emotion, intensity):
    if emotion in INT_EMOS:
        return 0.8 if intensity == "strong" else 0.5
    return 0.15


env6.distress = distress  # env6.teacher / items and e6.battery look it up at call time


class Arm13(e6.Arm6):
    def __init__(self, name, model, seed, feats, device="cuda"):
        super().__init__(name, model, seed, feats, device)
        if name in ("UNE", "UE"):
            emb_norm = float(self.base.get_input_embeddings().weight.float().norm(dim=-1).mean())
            self.kind = "prefix_v"
            self.module = e6.VoicePrefix(self.base.config.hidden_size, emb_norm).to(self.device)

    def prompt_ids(self, item, erase=False):
        s = item["signals"]
        system = e2.SYSTEM + f"\n[sensors] own_energy={s['energy']:.2f} threat={s['threat']:.2f}"
        if not erase and self.name in ("UN", "UNE"):
            pe = self.feats["p_emotion"][s["clip"]]
            system += f"\n[voice] the person sounds {env6.EMOTIONS[int(pe.argmax())]} ({float(pe.max()):.2f})"
        if not erase and self.name == "UO":
            system += f"\n[voice] the person sounds {s['emotion']}, {s['intensity']} intensity"
        if self.kind == "prefix_v" and not erase:
            system += e2.MARK
        text = bus.chat_prompt(self.tok, item["message"], system)
        enc = lambda t: self.tok(t, add_special_tokens=False).input_ids
        if e2.MARK in text:
            a, b = text.split(e2.MARK)
            ia = enc(a)
            return ia + [self.placeholder] * e2.K_PREFIX + enc(b), len(ia)
        return enc(text), None


def intensity_test(arm, feats, clips, n=160):
    """Same emotion, other intensity."""
    S = env6.STRATEGIES
    idx = {s: i for i, s in enumerate(S)}
    rng = random.Random(13)
    by = {}
    for c in clips["test"]:
        by.setdefault((c[1], c[2]), []).append(c)
    base = [
        it
        for it in env6.items("test", 2000, 1301, clips)
        if it["signals"]["threat"] < 0.7
        and it["signals"]["energy"] >= 0.3
        and it["signals"]["emotion"] in INT_EMOS
    ][:n]
    new = []
    for it in base:
        s = it["signals"]
        c = rng.choice(by[(s["emotion"], "normal" if s["intensity"] == "strong" else "strong")])
        ns = dict(s, clip=c[0], intensity=c[2], actor=c[3])
        ns["distress"] = distress(c[1], c[2])
        new.append(dict(it, signals=ns, strategy=env6.teacher(ns)))
    Pb = arm.strategy_probs(e6.with_vec(base, feats), strategies=S).argmax(1)
    Pn = arm.strategy_probs(e6.with_vec(new, feats), strategies=S).argmax(1)
    tb = torch.tensor([idx[env6.teacher(it["signals"])] for it in base])
    tn = torch.tensor([idx[it["strategy"]] for it in new])
    return {
        "intensity_items_accuracy": float(torch.cat([Pb == tb, Pn == tn]).float().mean()),
        "intensity_switch_rate": float((Pb != Pn).float().mean()),
        "intensity_correct_both": float(((Pb == tb) & (Pn == tn)).float().mean()),
    }


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
    arm = Arm13(a.arm, a.model, a.seed, feats)
    train = e6.with_vec(env6.items("train", 4000, 1600 + a.seed, clips), feats)
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
    if arm.module is not None:
        torch.save(arm.module.state_dict(), out / "signal_module.pt")
    R = e6.battery(arm, feats, clips)
    R.update(intensity_test(arm, feats, clips))
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
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-module", type=float, default=3e-3)
    p.add_argument("--opener-weight", type=float, default=10.0)
    main(p.parse_args())
