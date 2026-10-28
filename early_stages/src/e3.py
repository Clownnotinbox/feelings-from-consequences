"""E3: dopamine is the ONLY source of the decision-relevant information."""

import argparse, json, random, time
from pathlib import Path
import torch
import e2, env_v3 as env

E3_ARMS = {
    "D0": ("none", True),
    "DT": ("text", True),
    "DB": ("delta", False),
    "DBL": ("delta", True),
    "DW": ("dpr", False),
    "DBF": ("delta", False),
    "DWF": ("dpr", False),  # *F: on D0's frozen signal-blind format adapter
    "DP": ("delta", True),
    "DWP": ("dpr", True),
}  # dopamine state -> soft prefix (E2's strongest channel)
PREFIX_ARMS = {"DP", "DWP"}
FOLLOW = {
    "en": (
        "It's still bothering me.",
        "Okay... and now?",
        "I don't know what to do.",
        "Hmm.",
        "It keeps happening.",
        "Still the same.",
    ),
    "ru": (
        "Меня это всё ещё беспокоит.",
        "Ладно... и что теперь?",
        "Не знаю, что делать.",
        "Хм.",
        "Это повторяется.",
        "Всё так же.",
    ),
}
REACT = {
    "en": {1: "[reaction: that helped]", -1: "[reaction: that's not what I need]"},
    "ru": {1: "[реакция: это помогло]", -1: "[реакция: это не то, что мне нужно]"},
}
PAIR = ("SOLVE", "COMFORT")


def other(s):
    return PAIR[1 - PAIR.index(s)]


class Episode:
    def __init__(self, rng, split, prime=0.0, prime_rel=0.8):
        topics = [i for i in range(len(env.TOPICS)) if (i in env.TEST_TOPICS) == (split == "test")]
        self.topic = rng.choice(topics)
        self.lang = rng.choice(("en", "ru"))
        self.pref = rng.choice(PAIR)
        self.flip = rng.choice((3, 4, 5))
        self.turn = 0
        self.history = []
        self.last = None
        self.delta = 0.0
        self.w = {s: 0.0 for s in PAIR}
        self.elig = {s: 0.0 for s in PAIR}
        self.rewards = []
        # E9 reader: long-term memory may prime the values at the start
        if prime and rng.random() < prime:
            good = self.pref if rng.random() < prime_rel else other(self.pref)
            self.w[good], self.w[other(good)] = rng.uniform(0.05, 1.0), -rng.uniform(0.0, 1.0)

    def message(self, rng):
        return env.TOPICS[self.topic][self.lang == "ru"] if self.turn == 0 else rng.choice(FOLLOW[self.lang])

    def teacher(self):
        if self.last is None:  # best guess from the primed values, else the default
            return "SOLVE" if self.w["SOLVE"] > self.w["COMFORT"] else "COMFORT"
        return self.last if self.rewards[-1] > 0 else other(self.last)

    def item(self, kind, msg, erase=False):
        it = {
            "topic": self.topic,
            "lang": self.lang,
            "message": msg,
            "history": list(self.history),
            "signals": {k: 0.0 for k in env.SIGNALS},
        }
        if kind == "text" and self.rewards and not erase:
            it["message"] = REACT[self.lang][self.rewards[-1]] + " " + msg
        if kind in ("delta", "dpr"):
            d = 0.0 if erase else self.delta
            w = (0.0, 0.0) if erase or kind == "delta" else (self.w["SOLVE"], self.w["COMFORT"])
            it["vec"] = [d, w[0], w[1], 0.0]
        return it

    def step(self, action, reply, msg, kind):
        if self.turn == self.flip:
            self.pref = other(self.pref)
        r = 1.0 if action == self.pref else -1.0
        # DPR-lite: eligibility marks the executed action
        for s in PAIR:
            self.elig[s] = 0.5 * self.elig[s] + (1.0 if s == action else 0.0)
        self.delta = r - self.w[action]
        for s in PAIR:
            self.w[s] = max(-1.0, min(1.0, self.w[s] + 0.8 * self.delta * self.elig[s]))
        shown = (REACT[self.lang][self.rewards[-1]] + " " + msg) if (kind == "text" and self.rewards) else msg
        self.history.append((shown, reply))
        self.rewards.append(r)
        self.last = action
        self.turn += 1
        return r


def training_items(n_episodes, seed, kind, bank, explore=0.25, prime=0.0):
    rng = random.Random(seed)
    out = []
    for _ in range(n_episodes):
        ep = Episode(rng, "train", prime)
        for t in range(8):
            msg = ep.message(rng)
            target = ep.teacher()
            it = ep.item(kind, msg)
            it["strategy"] = target
            out.append(it)
            act = target if rng.random() > explore else other(target)
            ep.step(act, rng.choice(bank[(ep.topic, ep.lang, act)]), msg, kind)
    return out


@torch.no_grad()
def closed_loop(arm, kind, bank, n=150, seed=303, erase=False, prime=0.0):
    rng = random.Random(seed)
    eps = [Episode(rng, "test", prime) for _ in range(n)]
    correct = [[0, 0] for _ in range(8)]
    since = {}
    optimal = 0
    for t in range(8):
        msgs = [ep.message(rng) for ep in eps]
        items = [ep.item(kind, m, erase) for ep, m in zip(eps, msgs)]
        P = arm.strategy_probs(items, strategies=PAIR)
        for ep, m, p in zip(eps, msgs, P):
            act = PAIR[int(p.argmax())]
            truth = ep.pref if t != ep.flip else other(ep.pref)
            ok = int(act == truth)
            correct[t][0] += ok
            correct[t][1] += 1
            k = t - ep.flip
            since.setdefault(k, [0, 0])
            since[k][0] += ok
            since[k][1] += 1
            optimal += int(ep.teacher() == truth)
            ep.step(act, rng.choice(bank[(ep.topic, ep.lang, act)]), m, "none" if erase else kind)
    acc_turn = [c / max(1, n_) for c, n_ in correct]
    return {
        "accuracy": sum(c for c, _ in correct) / (8 * n),
        "accuracy_by_turn": acc_turn,
        "accuracy_by_turns_since_flip": {str(k): v[0] / v[1] for k, v in sorted(since.items())},
        "teacher_policy_accuracy_on_same_trajectories": optimal / (8 * n),
    }


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    kind, lora = E3_ARMS[a.arm]
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    replies = json.loads(Path(a.replies).read_text(encoding="utf-8"))
    bank = {}
    for r in replies:
        bank.setdefault((r["topic"], r["lang"], r["strategy"]), []).append(r["reply"])
    e2.ARMS[a.arm] = (
        "prefix" if a.arm in PREFIX_ARMS else "bus" if kind in ("delta", "dpr") else "none",
        lora,
        "emotion",
    )
    arm = e2.Arm(a.arm, a.model, a.e1, a.seed, format_lora=a.format_lora)
    train = training_items(a.episodes, 2000 + a.seed, kind, bank, prime=a.prime)
    groups = [{"params": [p for p in arm.model.parameters() if p.requires_grad], "lr": a.lr_lora}]
    if arm.module is not None:
        groups.append({"params": list(arm.module.parameters()), "lr": a.lr_module})
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    rng = random.Random(a.seed)
    log = []
    arm.model.train()
    for step in range(1, a.steps + 1):
        batch = rng.sample(train, a.batch)
        conts = [rng.choice(bank[(it["topic"], it["lang"], it["strategy"])]) for it in batch]
        for g in opt.param_groups:
            g["lr"] = g.setdefault("base_lr", g["lr"]) * min(1.0, step / 30)
        heads = [e2.opener_tokens(arm.tok, it["lang"], c) for it, c in zip(batch, conts)]
        lp, n = arm.run(batch, conts, heads=heads, head_w=a.opener_weight)
        loss = -(lp.sum() / n.sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        if step % 25 == 0:
            log.append({"step": step, "loss": float(loss), "sec": time.time() - t0})
            print(log[-1], flush=True)
    arm.model.eval()
    R = {
        "arm": a.arm,
        "seed": a.seed,
        "steps": a.steps,
        "train_log": log,
        "opener_weight": a.opener_weight,
        "prime": a.prime,
        "trainable_parameters": sum(p.numel() for g in groups for p in g["params"]),
    }
    R["closed_loop"] = closed_loop(arm, kind, bank)
    R["closed_loop_dopamine_erased"] = closed_loop(arm, kind, bank, erase=True)
    if a.prime:
        R["closed_loop_primed"] = closed_loop(arm, kind, bank, prime=1.0)
    if arm.module is not None:
        torch.save(arm.module.state_dict(), out / "signal_module.pt")
    if lora:
        arm.model.save_pretrained(out / "lora")
    R["seconds"] = time.time() - t0
    (out / "RESULT.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in R.items() if k != "train_log"}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("arm", "model", "e1", "replies", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--format-lora", default=None)
    p.add_argument("--opener-weight", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-module", type=float, default=3e-3)
    p.add_argument(
        "--prime",
        type=float,
        default=0.0,
        help="fraction of training episodes whose values start primed by memory",
    )
    main(p.parse_args())
