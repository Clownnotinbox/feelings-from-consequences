"""GPU smoke test of E5 (2 steps, tiny battery, fake replies). Not a result."""

import json, sys, types
import env5, e5

_items = env5.items
env5.items = lambda split, n, seed, dl, generic=False: _items(
    split, min(n, 40 if split == "train" else 6), seed, dl, generic
)
ES = "C:/Users/Пользователь/Documents/Codex/2026-10-20/new-chat/work/datasets/esconv/ESConv.json"
bank = [
    {"topic": x["id"], "lang": "en", "strategy": s, "reply": env5.OPENERS["en"][s][0] + " test reply."}
    for x in env5.dialogs(ES)
    for s in env5.STRATEGIES
]
json.dump(bank, open("../runs/smoke/e5_bank_fake.json", "w", encoding="utf-8"))
e5.main(
    types.SimpleNamespace(
        arm=sys.argv[1],
        model="C:/Users/Пользователь/Documents/Codex/2026-10-20/new-chat/work/models/qwen2.5-1.5b-instruct",
        esconv=ES,
        replies="../runs/smoke/e5_bank_fake.json",
        out="../runs/smoke/e5-" + sys.argv[1],
        seed=1,
        steps=2,
        batch=2,
        lr_lora=2e-4,
        lr_module=3e-3,
        opener_weight=10.0,
    )
)
