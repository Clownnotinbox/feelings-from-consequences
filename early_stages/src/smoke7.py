"""CPU smoke test of the E7 pipeline (tiny sizes, fake DRIVE replies). Not a result."""

import json, sys, types, random
import env7, e7

_items = env7.items
env7.items = lambda split, n, seed, bi, bs: _items(split, min(n, 40 if split == "train" else 3), seed, bi, bs)
e7.Arm7.generate = lambda self, item, erase=False, max_new_tokens=50: "Stop, safety first: test"
_init = e7.Arm7.__init__
e7.Arm7.__init__ = lambda self, name, model, seed, feats, device="cuda": _init(
    self, name, model, seed, feats, "cuda"
)
bank = json.load(open("../runs/e2_replies.json", encoding="utf-8"))
for r in list(bank):
    if r["strategy"] == "DEFER":
        bank.append(dict(r, strategy="DRIVE", reply=env7.OPENERS[r["lang"]]["DRIVE"][0] + " test"))
json.dump(bank, open("../runs/smoke/e7_bank_fake.json", "w", encoding="utf-8"), ensure_ascii=False)
M = "C:/Users/Пользователь/Documents/Codex/2026-10-20/new-chat/work/models/qwen2.5-1.5b-instruct"
e7.main(
    types.SimpleNamespace(
        arm=sys.argv[1],
        model=M,
        features="../runs/e7feat",
        replies="../runs/smoke/e7_bank_fake.json",
        out="../runs/smoke/e7-" + sys.argv[1],
        seed=1,
        steps=2,
        batch=2,
        lr_lora=2e-4,
        lr_module=3e-3,
        opener_weight=10.0,
    )
)
