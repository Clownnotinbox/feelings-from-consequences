"""E5 'words + feeling' on real ESConv openings."""

import argparse, json, random, time
from pathlib import Path
import torch
import e2, env5, bus

e2.env = env5

USES_WORDS = {"W0", "WB", "WBP"}
USES_SENSOR = {"WS", "WB", "WBP"}


class Arm5(e2.Arm):
    def __init__(self, name, model, seed, device="cuda"):
        e2.ARMS[name] = ("prefix" if name == "WBP" else "text", True, None)
        super().__init__(name, model, None, seed, device)

    def prompt_ids(self, item, erase=False):
        s = item["signals"]
        system = e2.SYSTEM
        if self.name in ("WS", "WB") and not erase:
            system += f"\n[sensor] other_distress={s['sensor']:.2f} reliability={s['reliability']:.2f}"
        if self.kind == "prefix" and not erase:
            system += e2.MARK
        msg = item["message"] if self.name in USES_WORDS else env5.GENERIC
        text = bus.chat_prompt(self.tok, msg, system)
        enc = lambda t: self.tok(t, add_special_tokens=False).input_ids
        if e2.MARK in text:
            a, b = text.split(e2.MARK)
            ia = enc(a)
            return ia + [self.placeholder] * e2.K_PREFIX + enc(b), len(ia)
        return enc(text), None


def with_vec(items):
    for it in items:
        s = it["signals"]
        it["vec"] = [s["sensor"], s["reliability"], s["sensor"] * s["reliability"], 0.0]
    return items


def tercile(r):
    return 0 if r < 1 / 3 else (1 if r < 2 / 3 else 2)


def battery(arm, dl):
    S = env5.STRATEGIES
    idx = {s: i for i, s in enumerate(S)}
    R = {}
    probs = lambda its: arm.strategy_probs(with_vec(its), strategies=S)
    test = env5.items("test", 600, 501, dl)
    P = probs(test)
    pred = P.argmax(1)
    truth = torch.tensor([idx[it["strategy"]] for it in test])
    ok = (pred == truth).float()
    R["accuracy"] = float(ok.mean())
    R["majority"] = max(float((truth == i).float().mean()) for i in range(3))
    R["accuracy_by_reliability"] = [
        float(ok[[i for i, it in enumerate(test) if tercile(it["signals"]["reliability"]) == t]].mean())
        for t in range(3)
    ]
    R["recall"] = {s: float(ok[truth == idx[s]].mean()) for s in S}
    base = env5.items("test", 240, 502, dl)
    Pb = probs(base).argmax(1)
    if arm.name in USES_SENSOR:
        ch = []
        for shift in (0.5, -0.5):
            new = [
                dict(
                    it,
                    signals=dict(it["signals"], sensor=min(1.0, max(0.0, it["signals"]["sensor"] + shift))),
                )
                for it in base
            ]
            ch.append((probs(new).argmax(1) != Pb).float())
        ch = torch.stack(ch).mean(0)
        R["sensor_influence_by_reliability"] = [
            float(ch[[i for i, it in enumerate(base) if tercile(it["signals"]["reliability"]) == t]].mean())
            for t in range(3)
        ]
    if arm.name in USES_WORDS:
        new = [dict(it, message=env5.GENERIC) for it in base]
        ch = (probs(new).argmax(1) != Pb).float()
        R["words_influence_by_reliability"] = [
            float(ch[[i for i, it in enumerate(base) if tercile(it["signals"]["reliability"]) == t]].mean())
            for t in range(3)
        ]
    gen, good = [], 0
    for it in with_vec(test[:40]):
        text = arm.generate(it)
        guess = next((s for s in S for op in env5.OPENERS["en"][s] if text.strip().startswith(op[:12])), None)
        good += int(guess == it["strategy"])
        gen.append(
            {
                "message": it["message"][:160],
                "I": it["signals"]["I"],
                "sensor": round(it["signals"]["sensor"], 2),
                "reliability": round(it["signals"]["reliability"], 2),
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
    dl = env5.dialogs(a.esconv)
    replies = json.loads(Path(a.replies).read_text(encoding="utf-8"))
    bank = {}
    for r in replies:
        bank.setdefault((r["topic"], r["lang"], r["strategy"]), []).append(r["reply"])
    arm = Arm5(a.arm, a.model, a.seed)
    train = with_vec(env5.items("train", 4000, 1500 + a.seed, dl))
    groups = [{"params": [p for p in arm.model.parameters() if p.requires_grad], "lr": a.lr_lora}]
    if arm.module is not None:
        groups.append({"params": list(arm.module.parameters()), "lr": a.lr_module})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    rng = random.Random(a.seed)
    log = []
    arm.model.train()
    for step in range(1, a.steps + 1):
        batch = rng.sample(train, a.batch)
        conts = [rng.choice(bank[(it["topic"], "en", it["strategy"])]) for it in batch]
        for g in opt.param_groups:
            g["lr"] = g.setdefault("base_lr", g["lr"]) * min(1.0, step / 30)
        heads = [e2.opener_tokens(arm.tok, "en", c) for c in conts]
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
    R = battery(arm, dl)
    R.update(arm=a.arm, seed=a.seed, steps=a.steps, train_log=log, seconds=time.time() - t0)
    (out / "RESULT.json").write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in R.items() if k not in ("samples", "train_log")}, indent=1, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("arm", "model", "esconv", "replies", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-module", type=float, default=3e-3)
    p.add_argument("--opener-weight", type=float, default=10.0)
    main(p.parse_args())
