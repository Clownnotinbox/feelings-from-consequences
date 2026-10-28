"""Local smoke test for E13 on an 8 GB card: same code, scoring in chunks of 4 instead of 16."""

import sys
import e2, e13

orig = e2.Arm.strategy_probs
e2.Arm.strategy_probs = lambda self, items, erase=False, chunk=16, strategies=e2.env.STRATEGIES: orig(
    self, items, erase, 4, strategies
)
sys.argv = ["e13.py"] + sys.argv[1:]
p = __import__("argparse").ArgumentParser()
for k in ("arm", "model", "features", "replies", "out"):
    p.add_argument("--" + k, required=True)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--steps", type=int, default=3)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--lr-lora", type=float, default=2e-4)
p.add_argument("--lr-module", type=float, default=3e-3)
p.add_argument("--opener-weight", type=float, default=10.0)
e13.main(p.parse_args())
