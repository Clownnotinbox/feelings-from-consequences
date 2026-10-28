import sys, types
import e3

_cl = e3.closed_loop
e3.closed_loop = lambda arm_, kind, bank, n=150, seed=303, erase=False: _cl(arm_, kind, bank, 8, seed, erase)
M = "C:/Users/Пользователь/Documents/Codex/2026-10-20/new-chat/work/models/qwen2.5-1.5b-instruct"
e3.main(
    types.SimpleNamespace(
        arm=sys.argv[1],
        model=M,
        e1="../runs/e1-qwen1.5b",
        replies="../runs/e2_replies.json",
        out=sys.argv[2],
        seed=1,
        steps=int(sys.argv[3]),
        lr_lora=2e-4,
        lr_module=3e-3,
        batch=4,
        episodes=200,
        format_lora=None,
        opener_weight=10.0,
    )
)
