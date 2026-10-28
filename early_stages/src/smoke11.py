"""Local smoke test for E11 arms."""

import json, random, sys
import torch
import e2, e3b

M = sys.argv[1]
bank = {}
for r in json.load(open("../runs/e2_replies.json", encoding="utf-8")):
    bank.setdefault((r["topic"], r["lang"], r["strategy"]), []).append(r["reply"])
e3b.HARD.update(turns=16, noise=0.2, flips=2, teacher="exact")
for name in ("DG", "DWT"):
    kind, lora = e3b.E3_ARMS[name]
    e2.ARMS[name] = ("prefix" if name in e3b.PREFIX_ARMS else "none", lora, "emotion")
    arm = e2.Arm(name, M, None, 1)
    if kind == "gru":
        arm.module = e3b.GRUPrefix(arm.base.config.hidden_size, arm.module.scale).to(arm.device)
    train = e3b.training_items(6, 1, kind, bank)
    rng = random.Random(0)
    params = [p for p in arm.model.parameters() if p.requires_grad] + (
        list(arm.module.parameters()) if arm.module is not None else []
    )
    opt = torch.optim.AdamW(params, lr=1e-3)
    arm.model.train()
    for step in range(3):
        batch = rng.sample(train, 2)
        conts = [rng.choice(bank[(it["topic"], it["lang"], it["strategy"])]) for it in batch]
        heads = [e2.opener_tokens(arm.tok, it["lang"], c) for it, c in zip(batch, conts)]
        lp, n = arm.run(batch, conts, heads=heads, head_w=10.0)
        loss = -(lp.sum() / n.sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
        g = sum(
            float(p.grad.abs().sum())
            for p in (arm.module.parameters() if arm.module is not None else [])
            if p.grad is not None
        )
        print(name, "step", step, "loss %.3f" % float(loss), "module grad %.2e" % g)
    arm.model.eval()
    late = [it for it in train if len(it["history"]) >= 5][:4]
    print(name, "vec len", len(late[0].get("vec", [])), "msg:", late[0]["message"][:80])
    print(name, "probs", arm.strategy_probs(late, strategies=e3b.PAIR, chunk=4).numpy().round(3).tolist())
    print(
        name,
        "probs erased",
        arm.strategy_probs(late, erase=True, strategies=e3b.PAIR, chunk=4).numpy().round(3).tolist(),
    )
    del arm
    torch.cuda.empty_cache()
