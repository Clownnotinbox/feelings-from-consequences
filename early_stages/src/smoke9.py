"""Logic test of E9 memory variants with an ideal value reader (no LLM). Not a result."""

import json, types
import numpy as np, torch
import e9


class Reader:
    def strategy_probs(self, items, strategies=None, chunk=8):
        v = torch.tensor([[it["vec"][1], it["vec"][2]] for it in items])
        return torch.softmax(v * 5 + 1e-3 * torch.randn_like(v), -1)


spk = dict(np.load("../runs/e9_speaker.npz"))
bank = {}
for r in json.load(open("../runs/e2_replies.json", encoding="utf-8")):
    bank.setdefault((r["topic"], r["lang"], r["strategy"]), []).append(r["reply"])
