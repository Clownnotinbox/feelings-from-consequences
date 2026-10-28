"""E3: dopamine is the ONLY source of the decision-relevant information."""

import argparse, itertools, json, os, random, time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np, torch
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
    "DWP": ("dpr", True),  # dopamine state -> soft prefix (E2's strongest channel)
    "DWT": ("dprtext", True),  # the same dopamine state written as numbers in the text
    "DG": ("gru", True),
}  # LEARNED core: GRU over (action, reaction) per turn -> prefix
PREFIX_ARMS = {"DP", "DWP", "DG"}
GRU_STEPS = 16  # the last turns the learned core sees
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


# E3b knobs (defaults = original E3)
# teacher 'filter' = constant-hazard Bayes filter
# eta / decay: DPR learning rate and eligibility decay
HARD = {"turns": 8, "noise": 0.0, "flips": 1, "teacher": "filter", "eta": 0.8, "decay": 0.5}
FEEL = {
    "en": "[my feeling: plan {:+.2f}, support {:+.2f}, surprise {:+.2f}]",
    "ru": "[моё чувство: план {:+.2f}, поддержка {:+.2f}, неожиданность {:+.2f}]",
}
_HYP = {}


def hypotheses():
    """All (initial preference, flip set) the generator can produce."""
    key = (HARD["turns"], HARD["flips"])
    if key not in _HYP:
        T, F = key
        sets = (
            [(3,), (4,), (5,)]
            if (F == 1 and T == 8)
            else [
                f
                for f in itertools.combinations(range(3, T - 2), F)
                if all(b - a >= 3 for a, b in zip(f, f[1:]))
            ]
        )
        _HYP[key] = np.array(
            [[p0 ^ (sum(1 for x in f if x <= t) % 2) for t in range(T)] for p0 in (0, 1) for f in sets]
        )
    return _HYP[key]


class Episode:
    def __init__(self, rng, split):
        topics = [i for i in range(len(env.TOPICS)) if (i in env.TEST_TOPICS) == (split == "test")]
        self.topic = rng.choice(topics)
        self.lang = rng.choice(("en", "ru"))
        self.pref = rng.choice(PAIR)
        self.turn = 0
        self.rng = random.Random(rng.random())
        if HARD["flips"] == 1 and HARD["turns"] == 8:
            self.flips = [rng.choice((3, 4, 5))]
        else:
            while True:
                f = sorted(rng.sample(range(3, HARD["turns"] - 2), HARD["flips"]))
                if all(b - a >= 3 for a, b in zip(f, f[1:])):
                    break
            self.flips = f
        self.flip = self.flips[0]
        self.post = 0.5
        self.h = len(self.flips) / HARD["turns"]
        self.history = []
        self.last = None
        self.delta = 0.0
        self.w = {s: 0.0 for s in PAIR}
        self.elig = {s: 0.0 for s in PAIR}
        self.rewards = []
        self.acts = []
        self.logp = np.zeros(len(hypotheses()))

    def message(self, rng):
        return env.TOPICS[self.topic][self.lang == "ru"] if self.turn == 0 else rng.choice(FOLLOW[self.lang])

    def prior(self):
        return 0.5 if self.turn == 0 else self.post * (1 - self.h) + (1 - self.post) * self.h

    def teacher(self):
        """Bayes filter over the hidden preference (= win-stay/lose-shift when reactions are reliable)."""
        if HARD["teacher"] == "exact":
            w = np.exp(self.logp - self.logp.max())
            p = float((w * hypotheses()[:, self.turn]).sum() / w.sum())
            return "SOLVE" if p > 0.5 + 1e-9 else "COMFORT"
        return "SOLVE" if self.prior() > 0.5 else "COMFORT"

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
        if kind == "dprtext" and not erase:
            it["message"] = FEEL[self.lang].format(self.w["SOLVE"], self.w["COMFORT"], self.delta) + " " + msg
        if kind in ("delta", "dpr"):
            d = 0.0 if erase else self.delta
            w = (0.0, 0.0) if erase or kind == "delta" else (self.w["SOLVE"], self.w["COMFORT"])
            it["vec"] = [d, w[0], w[1], 0.0]
        if kind == "gru":  # the same information the text history carries
            rows = (
                []
                if erase
                else [
                    [float(a_ == "SOLVE"), float(a_ == "COMFORT"), r_]
                    for a_, r_ in zip(self.acts, self.rewards)
                ][-GRU_STEPS:]
            )
            it["vec"] = [x for row in [[0.0, 0.0, 0.0]] * (GRU_STEPS - len(rows)) + rows for x in row]
        return it

    def step(self, action, reply, msg, kind):
        prior = self.prior()
        if self.turn in self.flips:
            self.pref = other(self.pref)
        r = 1.0 if action == self.pref else -1.0
        if HARD["noise"] and self.rng.random() < HARD["noise"]:
            r = -r  # misleading reaction
        q = HARD["noise"]
        good = lambda pref: (1 - q) if (action == pref) == (r > 0) else q
        ls, lc = good("SOLVE"), good("COMFORT")
        z = prior * ls + (1 - prior) * lc
        self.post = prior * ls / z if z > 0 else prior
        solve_seen = (action == "SOLVE") == (r > 0)  # what the reaction says about the current preference
        self.logp += np.log(np.where(hypotheses()[:, self.turn] == int(solve_seen), 1 - q, q).clip(1e-12))
        # DPR-lite: eligibility marks the executed action
        for s in PAIR:
            self.elig[s] = HARD["decay"] * self.elig[s] + (1.0 if s == action else 0.0)
        self.delta = r - self.w[action]
        for s in PAIR:
            self.w[s] = max(-1.0, min(1.0, self.w[s] + HARD["eta"] * self.delta * self.elig[s]))
        shown = (REACT[self.lang][self.rewards[-1]] + " " + msg) if (kind == "text" and self.rewards) else msg
        self.history.append((shown, reply))
        self.rewards.append(r)
        self.acts.append(action)
        self.last = action
        self.turn += 1
        return r


class GRUPrefix(torch.nn.Module):
    """Learned affective core."""

    def __init__(self, hidden, scale, width=64):
        super().__init__()
        self.gru = torch.nn.GRU(3, width, batch_first=True)
        self.out = torch.nn.Linear(width, e2.K_PREFIX * hidden)
        torch.nn.init.normal_(self.out.weight, std=1e-3)
        torch.nn.init.zeros_(self.out.bias)
        self.hidden = hidden
        self.scale = scale

    def forward(self, s):
        _, h = self.gru(s.view(len(s), GRU_STEPS, 3))
        return self.out(h[-1]).view(len(s), e2.K_PREFIX, self.hidden) * self.scale


def training_items(n_episodes, seed, kind, bank, explore=0.25):
    rng = random.Random(seed)
    out = []
    for _ in range(n_episodes):
        ep = Episode(rng, "train")
        for t in range(HARD["turns"]):
            msg = ep.message(rng)
            target = ep.teacher()
            it = ep.item(kind, msg)
            it["strategy"] = target
            out.append(it)
            act = target if rng.random() > explore else other(target)
            ep.step(act, rng.choice(bank[(ep.topic, ep.lang, act)]), msg, kind)
    return out


@torch.no_grad()
def closed_loop(arm, kind, bank, n=150, seed=303, erase=False):
    rng = random.Random(seed)
    eps = [Episode(rng, "test") for _ in range(n)]
    T = HARD["turns"]
    correct = [[0, 0] for _ in range(T)]
    since = {}
    optimal = 0
    for t in range(T):
        msgs = [ep.message(rng) for ep in eps]
        items = [ep.item(kind, m, erase) for ep, m in zip(eps, msgs)]
        P = arm.strategy_probs(
            items, strategies=PAIR, chunk=4
        )  # long 16-turn histories: small chunks on 24 GB
        for ep, m, p in zip(eps, msgs, P):
            act = PAIR[int(p.argmax())]
            truth = other(ep.pref) if t in ep.flips else ep.pref
            ok = int(act == truth)
            correct[t][0] += ok
            correct[t][1] += 1
            k = t - max([f for f in ep.flips if f <= t], default=ep.flips[0])
            since.setdefault(k, [0, 0])
            since[k][0] += ok
            since[k][1] += 1
            optimal += int(ep.teacher() == truth)
            ep.step(act, rng.choice(bank[(ep.topic, ep.lang, act)]), m, "none" if erase else kind)
    acc_turn = [c / max(1, n_) for c, n_ in correct]
    return {
        "accuracy": sum(c for c, _ in correct) / (T * n),
        "accuracy_by_turn": acc_turn,
        "accuracy_by_turns_since_flip": {str(k): v[0] / v[1] for k, v in sorted(since.items())},
        "teacher_policy_accuracy_on_same_trajectories": optimal / (T * n),
        "hard": dict(HARD),
    }


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    kind, lora = E3_ARMS[a.arm]
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    HARD.update(turns=a.turns, noise=a.noise, flips=a.flips, teacher=a.teacher, eta=a.eta, decay=a.decay)
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
    if kind == "gru":
        arm.module = GRUPrefix(arm.base.config.hidden_size, arm.module.scale).to(arm.device)
    train = training_items(a.episodes, 2000 + a.seed, kind, bank)
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
        "trainable_parameters": sum(p.numel() for g in groups for p in g["params"]),
    }
    R["closed_loop"] = closed_loop(arm, kind, bank)
    R["closed_loop_dopamine_erased"] = closed_loop(arm, kind, bank, erase=True)
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
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--flips", type=int, default=1)
    p.add_argument("--teacher", choices=("filter", "exact"), default="filter")
    p.add_argument("--eta", type=float, default=0.8)
    p.add_argument("--decay", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-module", type=float, default=3e-3)
    main(p.parse_args())
