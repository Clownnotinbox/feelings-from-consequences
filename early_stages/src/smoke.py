"""Fast end-to-end smoke test: tiny battery sizes, few steps. Not a result."""

import sys, types
import env_v3 as env

_items = env.items
env.items = lambda split, n, seed, ood=None: _items(split, min(n, 600 if split == "train" else 6), seed, ood)
exp, arm, out = sys.argv[1], sys.argv[2], sys.argv[3]
M = "C:/Users/Пользователь/Documents/Codex/2026-10-20/new-chat/work/models/qwen2.5-1.5b-instruct"
common = dict(
    arm=arm,
    model=M,
    e1="../runs/e1-qwen1.5b",
    replies="../runs/e2_replies.json",
    out=out,
    seed=1,
    steps=10,
    lr_lora=2e-4,
    lr_module=2e-3,
)
if exp == "e2":
    import e2

    e2.load_retention.__defaults__ = (8,)
    e2.main(
        types.SimpleNamespace(
            **common,
            retention="../../affect-foundation-07/data/language-retention/validation.jsonl",
            batch=4,
            train_items=600
        )
    )
else:
    import e3

    _cl = e3.closed_loop
    e3.closed_loop = lambda arm_, kind, bank, n=150, seed=303, erase=False: _cl(
        arm_, kind, bank, 6, seed, erase
    )
    e3.main(types.SimpleNamespace(**common, batch=4, episodes=40))
