"""Quick check of the opener-weighted loss on the bus arm (not a result): small test battery."""

import sys, types
import env_v3 as env

_items = env.items
env.items = lambda split, n, seed, ood=None: _items(
    split, min(n, 4000 if split == "train" else 40), seed, ood
)
import e2

e2.load_retention.__defaults__ = (8,)
arm, steps, w, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
M = "C:/Users/Пользователь/Documents/Codex/2026-10-20/new-chat/work/models/qwen2.5-1.5b-instruct"
e2.main(
    types.SimpleNamespace(
        arm=arm,
        model=M,
        e1="../runs/e1-qwen1.5b",
        replies="../runs/e2_replies.json",
        out=out,
        seed=1,
        steps=steps,
        lr_lora=2e-4,
        lr_module=float(sys.argv[5]) if len(sys.argv) > 5 else 3e-3,
        retention="../../affect-foundation-07/data/language-retention/validation.jsonl",
        batch=8,
        train_items=4000,
        format_lora="../runs/local-1.5b/e2/T0-s1/lora",
        opener_weight=w,
    )
)
