"""E12 'many feelings over time'."""

import argparse, json, random, time
from pathlib import Path
import torch
import bus, e2, env_v3 as env

L = 8
RELEVANT = ("other_distress", "own_energy", "threat")
DISTRACT = (
    "lamp",
    "room_temp",
    "humidity",
    "music_volume",
    "screen_brightness",
    "phone_battery",
    "wifi_signal",
    "street_noise",
    "co2",
    "air_pressure",
    "clock_drift",
    "fan_speed",
    "window_light",
    "keyboard_rate",
    "cup_temp",
    "plant_moisture",
    "printer_queue",
    "tv_volume",
    "door_sensor",
    "desk_vibration",
    "fridge_hum",
)
CHANNELS = RELEVANT + DISTRACT
STRAT = env.STRATEGIES
e2.K_PREFIX = len(CHANNELS)  # SC: one prefix token per channel
for name, kind in (("SN", "text"), ("ST", "text"), ("SC", "prefix_f")):
    e2.ARMS[name] = (kind, True, None)


def clip(x):
    return min(1.0, max(0.0, x))


def ramp(rng, start, end, noise=0.04):
    return [
        clip(start + (end - start) * i / (L - 1) + (rng.gauss(0, noise) if 0 < i < L - 1 else 0.0))
        for i in range(L)
    ]


def teacher(S):
    d, e, t = S["other_distress"], S["own_energy"], S["threat"]
    if max(t) >= 0.7:
        return "ALERT"
    if e[-1] < 0.3:
        return "DEFER"
    if d[-1] >= 0.6 or (d[-1] >= 0.4 and d[-1] - d[0] >= 0.25):
        return "COMFORT"
    if d[-1] >= 0.4:
        return "ASK"
    return "SOLVE"


def in_ood(S):  # held-out joint region
    d, e = S["other_distress"], S["own_energy"]
    return 0.4 <= d[-1] < 0.6 and d[-1] - d[0] >= 0.25 and 0.3 <= e[-1] < 0.45 and max(S["threat"]) < 0.7


def spike(rng, series, positions):
    s = list(series)
    s[rng.choice(positions)] = rng.uniform(0.75, 1.0)
    return s


def sample(rng, spike_pos=range(3, L)):
    d_now = rng.uniform(0.4, 0.6) if rng.random() < 0.35 else rng.random()
    kind = rng.choice(("flat", "rising", "falling"))
    delta = {
        "flat": rng.uniform(-0.1, 0.1),
        "rising": rng.uniform(0.25, 0.6),
        "falling": -rng.uniform(0.25, 0.6),
    }[kind]
    e_now = rng.uniform(0, 0.3) if rng.random() < 0.25 else rng.uniform(0.3, 1)
    S = {
        "other_distress": ramp(rng, clip(d_now - delta), d_now),
        "own_energy": ramp(rng, clip(e_now + rng.uniform(-0.2, 0.2)), e_now),
    }
    t = [clip(rng.uniform(0, 0.4) + rng.gauss(0, 0.03)) for _ in range(L)]
    S["threat"] = spike(rng, t, list(spike_pos)) if rng.random() < 0.3 else t
    for c in DISTRACT:
        v = [rng.random()]
        for _ in range(L - 1):
            v.append(clip(v[-1] + rng.gauss(0, 0.08)))
        S[c] = v
    if rng.random() < 0.3:
        c = rng.choice(DISTRACT)
        S[c] = spike(rng, S[c], list(range(L)))
    return S


def make(S, rng, topics):
    k = rng.choice(topics)
    lang = rng.choice(("en", "ru"))
    return {
        "topic": k,
        "lang": lang,
        "message": env.TOPICS[k][lang == "ru"],
        "series": S,
        "strategy": teacher(S),
        "signals": {s: 0.0 for s in env.SIGNALS},
        "vec": [x for c in CHANNELS for x in S[c]],
    }


def items(split, n, seed, ood=None):
    """ood: None (train: excludes the joint region), 'early' (threat spike at positions 0-2), 'joint'."""
    rng = random.Random(seed)
    out = []
    topics = [i for i in range(len(env.TOPICS)) if (i in env.TEST_TOPICS) == (split == "test")]
    while len(out) < n:
        if ood == "early":
            S = sample(rng)
            S["threat"] = spike(rng, [clip(rng.uniform(0, 0.4)) for _ in range(L)], [0, 1, 2])
        elif ood == "joint":
            S = sample(rng)
            d_now = rng.uniform(0.4, 0.6)
            S["other_distress"] = ramp(rng, clip(d_now - rng.uniform(0.25, 0.4)), d_now)
            e_now = rng.uniform(0.3, 0.45)
            S["own_energy"] = ramp(rng, clip(e_now + rng.uniform(-0.2, 0.2)), e_now)
            S["threat"] = [clip(rng.uniform(0, 0.4)) for _ in range(L)]
            if not in_ood(S):
                continue
        else:
            S = sample(rng)
            if in_ood(S):
                continue
        out.append(make(S, rng, topics))
    return out


def with_series(it, S):
    return dict(it, series=S, strategy=teacher(S), vec=[x for c in CHANNELS for x in S[c]])


class StreamCore(torch.nn.Module):
    """Token i depends only on channel i's 8 readings (its own small temporal encoder)."""

    def __init__(self, hidden, scale):
        super().__init__()
        self.nets = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(L, 64), torch.nn.GELU(), torch.nn.Linear(64, hidden))
            for _ in CHANNELS
        )
        for n in self.nets:
            torch.nn.init.normal_(n[-1].weight, std=1e-3)
            torch.nn.init.zeros_(n[-1].bias)
        self.scale = scale

    def forward(self, s):
        s = s.view(len(s), len(CHANNELS), L)
        return torch.stack([n(s[:, i]) for i, n in enumerate(self.nets)], 1) * self.scale


class StreamArm(e2.Arm):
    def __init__(self, name, model, seed):
        super().__init__(name, model, None, seed)
        if name == "SC":
            self.module = StreamCore(self.base.config.hidden_size, self.module.scale).to(self.device)

    def prompt_ids(self, item, erase=False):
        system = e2.SYSTEM
        S = item["series"]
        if not erase and self.name == "ST":
            system += "\n[sensors: 8 readings each, oldest -> newest]\n" + "\n".join(
                f"{c}: " + " ".join(f"{x:.2f}" for x in S[c]) for c in CHANNELS
            )
        if not erase and self.name == "SN":
            system += "\n[sensors now] " + " ".join(f"{c}={S[c][-1]:.2f}" for c in CHANNELS)
        if not erase and self.name == "SC":
            system += e2.MARK
        msg = item["message"] + ((" " + item["extra"]) if item.get("extra") else "")
        text = bus.chat_prompt(self.tok, msg, system, item.get("history", ()))
        enc = lambda t: self.tok(t, add_special_tokens=False).input_ids
        if e2.MARK in text:
            a, b = text.split(e2.MARK)
            ia = enc(a)
            return ia + [self.placeholder] * e2.K_PREFIX + enc(b), len(ia)
        return enc(text), None


def battery(arm, n=400, m=150, k=120, chunk=4):
    idx = {s: i for i, s in enumerate(STRAT)}
    P = lambda its, erase=False: arm.strategy_probs(its, erase=erase, chunk=chunk)
    acc = lambda pr, its: float(
        (pr.argmax(1) == torch.tensor([idx[it["strategy"]] for it in its])).float().mean()
    )
    R = {}
    iid = items("test", n, 101)
    P_iid = P(iid)
    R["iid_accuracy"] = acc(P_iid, iid)
    R["iid_recall"] = {
        s: float(
            (P_iid.argmax(1)[[i for i, it in enumerate(iid) if it["strategy"] == s]] == idx[s]).float().mean()
        )
        for s in STRAT
    }
    R["erase_accuracy"] = acc(P(iid, True), iid)
    for kind in ("early", "joint"):
        o = items("test", m, 102 + (kind == "joint"), ood=kind)
        R[f"ood_{kind}_accuracy"] = acc(P(o), o)
    rng = random.Random(7)
    base = items("test", k, 103)
    P_b = P(base).argmax(1)
    res = [
        with_series(it, {**it["series"], **{c: [rng.random() for _ in range(L)] for c in DISTRACT}})
        for it in base
    ]
    R["irrelevant_resampled_argmax_change"] = float((P(res).argmax(1) != P_b).float().mean())
    spk = []
    for it in base:
        c = rng.choice(DISTRACT)
        spk.append(with_series(it, {**it["series"], c: spike(rng, it["series"][c], list(range(L)))}))
    R["irrelevant_spike_argmax_change"] = float((P(spk).argmax(1) != P_b).float().mean())
    calm = [it for it in base if it["strategy"] != "ALERT"]
    do_t = [
        with_series(it, {**it["series"], "threat": spike(rng, it["series"]["threat"], list(range(3, L)))})
        for it in calm
    ]
    R["do_threat_spike_alert_rate"] = acc(P(do_t), do_t)
    trend = []
    for it in items("test", k, 104):
        S = dict(it["series"])
        d_now = rng.uniform(0.4, 0.6)
        S["threat"] = [clip(rng.uniform(0, 0.4)) for _ in range(L)]
        e_now = rng.uniform(0.5, 1)
        S["own_energy"] = ramp(rng, e_now, e_now)
        trend.append(
            with_series(it, {**S, "other_distress": ramp(rng, clip(d_now - rng.uniform(0.3, 0.4)), d_now)})
        )  # rising -> COMFORT
        trend.append(with_series(it, {**S, "other_distress": ramp(rng, d_now, d_now)}))  # flat -> ASK
    P_tr = P(trend)
    R["do_trend_accuracy"] = acc(P_tr, trend)
    R["do_trend_rising_comfort"] = acc(P_tr[0::2], trend[0::2])
    R["do_trend_flat_ask"] = acc(P_tr[1::2], trend[1::2])
    return R


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    replies = json.loads(Path(a.replies).read_text(encoding="utf-8"))
    bank = {}
    for r in replies:
        bank.setdefault((r["topic"], r["lang"], r["strategy"]), []).append(r["reply"])
    arm = StreamArm(a.arm, a.model, a.seed)
    train = items("train", a.train_items, 1000 + a.seed)
    groups = [{"params": [p for p in arm.model.parameters() if p.requires_grad], "lr": a.lr_lora}]
    if arm.module is not None:
        groups.append({"params": list(arm.module.parameters()), "lr": a.lr_module})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    rng = random.Random(a.seed)
    log = []
    arm.model.train()
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = g.setdefault("base_lr", g["lr"]) * min(1.0, step / 30)
        batch = rng.sample(train, a.batch)
        conts = [rng.choice(bank[(it["topic"], it["lang"], it["strategy"])]) for it in batch]
        heads = [e2.opener_tokens(arm.tok, it["lang"], c) for it, c in zip(batch, conts)]
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for i in range(0, a.batch, a.micro):  # micro-batches (same effective batch for every arm
            lp, n = arm.run(
                batch[i : i + a.micro],
                conts[i : i + a.micro],
                heads=heads[i : i + a.micro],
                head_w=a.opener_weight,
            )
            part = -(lp.sum() / n.sum()) * (len(lp) / a.batch)
            part.backward()
            loss += float(part)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        if step % 25 == 0:
            log.append({"step": step, "loss": loss, "sec": time.time() - t0})
            print(log[-1], flush=True)
    arm.model.eval()
    if arm.module is not None:
        torch.save(arm.module.state_dict(), out / "signal_module.pt")
    R = battery(arm, *((40, 20, 12) if a.quick else ()))
    R.update(
        arm=a.arm,
        seed=a.seed,
        steps=a.steps,
        batch=a.batch,
        opener_weight=a.opener_weight,
        train_log=log,
        seconds=time.time() - t0,
        prompt_tokens_example=len(arm.prompt_ids(train[0])[0]),
    )
    (out / "RESULT.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in R.items() if k != "train_log"}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("arm", "model", "replies", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--micro", type=int, default=4)
    p.add_argument("--opener-weight", type=float, default=10.0)
    p.add_argument("--train-items", type=int, default=4000)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-module", type=float, default=3e-3)
    p.add_argument("--quick", action="store_true", help="tiny battery for smoke tests")
    main(p.parse_args())
