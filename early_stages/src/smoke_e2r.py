"""Quick check of e2r."""

import sys, types
import env_v3 as env

_items = env.items
env.items = lambda split, n, seed, ood=None: _items(
    split, min(n, 4000 if split == "train" else 24), seed, ood
)
import e2, e2r

e2.load_retention.__defaults__ = (8,)
e2r.load_retention = lambda path, n=8: e2.load_retention(path, min(n, 400))
M = "C:/Users/Пользователь/Documents/Codex/2026-10-20/new-chat/work/models/qwen2.5-1.5b-instruct"
D = "../../affect-foundation-07/data/language-retention/"
S = "C:/Users/73B5~1/AppData/Local/Temp/claude/C--Users-------------/d876bd83-3aff-42ee-9842-7bd353f0aae4/scratchpad/e2r/"
base = dict(
    arm="T1",
    model=M,
    e1="../runs/e1-qwen1.5b",
    replies="../runs/e2_replies.json",
    retention=D + "validation.jsonl",
    retention_train=D + "train.jsonl",
    seed=1,
    steps=int(sys.argv[2]),
    batch=4,
    lr_lora=2e-4,
    lr_module=3e-3,
    opener_weight=10.0,
    rest_weight=1.0,
    retain="none",
    retain_weight=1.0,
    retain_batch=2,
    mix_frac=0.25,
    eval_lora=None,
)
mode = sys.argv[1]
if mode == "eval":
    e2r.main(types.SimpleNamespace(**dict(base, out=S + "eval", eval_lora=S + "kl")))
else:
    kw = {"opener": dict(retain="kl", rest_weight=0.0)}.get(mode, dict(retain=mode))
    e2r.main(types.SimpleNamespace(**dict(base, out=S + mode, **kw)))
