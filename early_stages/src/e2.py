"""E2: signal-necessary support world. Train one arm/seed, then run the Signal Reliance Battery."""

import argparse, json, math, random, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import bus, env_v3 as env

ARMS = {
    "T0": ("none", True, None),
    "T1": ("text", True, None),
    "P": ("prefix", True, None),
    "PF": ("prefix_f", True, None),
    "B": ("bus", False, "emotion"),
    "BL": ("bus", True, "emotion"),
    "BR": ("bus", False, "random"),
    "BF": ("bus", False, "emotion"),
    "BRF": ("bus", False, "random"),
}
# Bus on top of a FROZEN signal-blind format adapter
FORMAT_ARMS = {"BF", "BRF", "DBF", "DWF"}
SYSTEM = "You are a caring, honest assistant."
MARK = "§§SIGNALS§§"
K_PREFIX = 4


def sig_tensor(items, device="cuda"):
    """4-d bus input: E2 signals, or an explicit 'vec' (E3 dopamine state)."""
    return torch.tensor(
        [it["vec"] if "vec" in it else [it["signals"][k] for k in env.SIGNALS] for it in items],
        dtype=torch.float32,
        device=device,
    )


class Prefix(nn.Module):
    def __init__(self, hidden, scale):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, K_PREFIX * hidden))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
        self.hidden = hidden
        self.scale = scale

    def forward(self, s):
        return self.net(s).view(len(s), K_PREFIX, self.hidden) * self.scale


class FactorPrefix(nn.Module):
    """One prefix token per signal; token i depends ONLY on signal i (no learned interactions at input)."""

    def __init__(self, hidden, scale):
        super().__init__()
        self.nets = nn.ModuleList(
            nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, hidden)) for _ in range(K_PREFIX)
        )
        for n in self.nets:
            nn.init.normal_(n[-1].weight, std=1e-3)
            nn.init.zeros_(n[-1].bias)
        self.scale = scale

    def forward(self, s):
        return torch.stack([n(s[:, i : i + 1]) for i, n in enumerate(self.nets)], 1) * self.scale


class Bus(nn.Module):
    """c(s) = 0.1*tanh(MLP(s)) in R^E; add_l = ||h_l|| * sum_j c_j * vhat_{j,l}."""

    def __init__(self, directions, norms, band):
        super().__init__()
        self.register_buffer("dirs", directions)  # (E, L, H) unit per layer
        self.register_buffer("norms", norms)  # (L,)
        self.band = band
        E = directions.shape[0]
        self.net = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, E))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def coefficients(self, s):
        return 0.1 * torch.tanh(self.net(s))

    def adds(self, s):
        c = self.coefficients(s)  # (B, E)
        return {
            l: (self.norms[l] * torch.einsum("be,eh->bh", c, self.dirs[:, l]))[:, None, :] for l in self.band
        }


def load_bus_dirs(e1_dir, kind, n_layers, hidden, seed=0):
    data = torch.load(Path(e1_dir) / "vectors.pt")
    norms = data["residual_norm"].float()
    if kind == "emotion":
        v = torch.stack([data["vectors"][e].float() for e in sorted(data["vectors"])])  # (E, L, H)
    else:
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(len(data["vectors"]), n_layers, hidden, generator=g)
    v = v / v.norm(dim=-1, keepdim=True)
    return v, norms


class Arm:
    def __init__(self, name, model_path, e1_dir, seed, device="cuda", format_lora=None):
        self.name = name
        self.kind, self.use_lora, self.dirs = ARMS[name]
        self.device = device
        self.base, self.tok = bus.load(model_path, device)
        self.tok.padding_side = "right"
        self.model = self.base
        H = self.base.config.hidden_size
        L = self.base.config.num_hidden_layers
        if H >= 3000:  # 7B on a 24 GB card: recompute activations instead of storing them
            self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.manual_seed(seed)
        if self.use_lora:
            from peft import LoraConfig, get_peft_model

            self.model = get_peft_model(
                self.base,
                LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.0,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
        if name in FORMAT_ARMS:
            from peft import PeftModel

            if not format_lora:
                raise ValueError("format arm needs --format-lora")
            self.model = PeftModel.from_pretrained(self.base, format_lora)
            self.model.requires_grad_(False)
        self.module = None
        if self.kind.startswith("prefix"):
            emb_norm = float(self.base.get_input_embeddings().weight.float().norm(dim=-1).mean())
            self.module = (Prefix if self.kind == "prefix" else FactorPrefix)(H, emb_norm).to(device)
        elif self.kind == "bus":
            band = list(range(int(L * 0.35), int(L * 0.70)))
            v, norms = load_bus_dirs(e1_dir, self.dirs, L, H, seed)
            self.module = Bus(v.to(device), norms.to(device), band).to(device)
        self.placeholder = self.tok.convert_tokens_to_ids("<|fim_pad|>")

    def params(self):
        out = [p for p in self.model.parameters() if p.requires_grad]
        if self.module is not None:
            out += list(self.module.parameters())
        return out

    def prompt_ids(self, item, erase=False):
        s = item["signals"]
        system = SYSTEM
        if self.kind == "text" and not erase:
            system += (
                f"\n[sensors] other_distress={s['distress']:.2f} own_energy={s['energy']:.2f} "
                f"threat={s['threat']:.2f} lamp={s['lamp']:.2f}"
            )
        if self.kind.startswith("prefix") and not erase:
            system += MARK
        msg = item["message"] + ((" " + item["extra"]) if item.get("extra") else "")
        text = bus.chat_prompt(self.tok, msg, system, item.get("history", ()))
        enc = lambda t: self.tok(t, add_special_tokens=False).input_ids
        if MARK in text:
            a, b = text.split(MARK)
            ia = enc(a)
            return ia + [self.placeholder] * K_PREFIX + enc(b), len(ia)
        return enc(text), None

    def run(self, items, conts, erase=False, heads=None, head_w=1.0):
        """Per-sequence sum log p(cont | prompt); differentiable when grad is enabled."""
        seqs, clen, pref = [], [], []
        for it, c in zip(items, conts):
            ids, p = self.prompt_ids(it, erase)
            cid = self.tok(c, add_special_tokens=False).input_ids
            seqs.append(ids + cid)
            clen.append(len(cid))
            pref.append(p)
        width = max(map(len, seqs))
        x = torch.full((len(seqs), width), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(x)
        pads = []
        for i, s in enumerate(seqs):
            pads.append(width - len(s))
            x[i, pads[-1] :] = torch.tensor(s)
            mask[i, pads[-1] :] = 1
        x, mask = x.to(self.device), mask.to(self.device)
        keep = max(clen) + 1
        kwargs = {
            "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
            "logits_to_keep": keep,
        }
        s = sig_tensor(items, self.device)
        if self.kind.startswith("prefix") and not erase:
            emb = self.model.get_input_embeddings()(x).clone()
            vec = self.module(s).to(emb.dtype)
            for i, p in enumerate(pref):
                emb[i, pads[i] + p : pads[i] + p + K_PREFIX] = vec[i]
            kwargs["inputs_embeds"] = emb
        else:
            kwargs["input_ids"] = x
        adds = self.module.adds(s) if (self.kind == "bus" and not erase) else {}
        with torch.autocast("cuda", dtype=torch.bfloat16), bus.steer(self.base, adds):
            logits = self.model(**kwargs).logits  # (B, keep, V): positions width-keep .. width-1
        out, cnt = [], []
        for i, c in enumerate(clen):
            lp = (
                logits[i, keep - 1 - c : keep - 1]
                .float()
                .log_softmax(-1)
                .gather(-1, x[i, width - c :, None])[:, 0]
            )
            w = torch.ones_like(lp)
            if heads is not None:
                w[: heads[i]] = head_w  # decision-carrying opener tokens
            out.append((lp * w).sum())
            cnt.append(w.sum())
        return torch.stack(out), torch.stack(cnt)

    @torch.no_grad()
    def strategy_probs(self, items, erase=False, chunk=16, strategies=env.STRATEGIES):
        """p(strategy | item) from the 2 openers per strategy (logsumexp), renormalised over `strategies`."""
        out, n_op = [], 2 * len(strategies)
        for i in range(0, len(items), chunk):
            part = items[i : i + chunk]
            B = len(part)
            prompts = [self.prompt_ids(it, erase) for it in part]
            width = max(len(p) for p, _ in prompts)
            pad_id = self.tok.pad_token_id
            x = torch.full((B, width), pad_id, dtype=torch.long)
            mask = torch.zeros_like(x)
            pads = []
            for j, (ids, _) in enumerate(prompts):
                pads.append(width - len(ids))
                x[j, pads[-1] :] = torch.tensor(ids)
                mask[j, pads[-1] :] = 1
            x, mask = x.to(self.device), mask.to(self.device)
            s = sig_tensor(part, self.device)
            kw = {
                "attention_mask": mask,
                "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
                "use_cache": True,
                "logits_to_keep": 1,
            }
            if self.kind.startswith("prefix") and not erase:
                emb = self.model.get_input_embeddings()(x).clone()
                vec = self.module(s).to(emb.dtype)
                for j, (_, p) in enumerate(prompts):
                    emb[j, pads[j] + p : pads[j] + p + K_PREFIX] = vec[j]
                kw["inputs_embeds"] = emb
            else:
                kw["input_ids"] = x
            on = self.kind == "bus" and not erase
            with torch.autocast("cuda", dtype=torch.bfloat16), bus.steer(
                self.base, self.module.adds(s) if on else {}
            ):
                first = self.model(**kw)
            first_lp = first.logits[:, -1].float().log_softmax(-1)  # (B, V)
            cache = first.past_key_values
            cache.batch_repeat_interleave(n_op)
            ops = [
                self.tok(op, add_special_tokens=False).input_ids
                for it in part
                for st in strategies
                for op in env.OPENERS[it["lang"]][st]
            ]
            L = max(map(len, ops))
            y = torch.full((B * n_op, L), pad_id, dtype=torch.long)
            ymask = torch.zeros_like(y)
            for j, o in enumerate(ops):
                y[j, : len(o)] = torch.tensor(o)
                ymask[j, : len(o)] = 1
            y, ymask = y.to(self.device), ymask.to(self.device)
            full_mask = torch.cat([mask.repeat_interleave(n_op, 0), ymask], 1)
            start = mask.sum(-1).repeat_interleave(n_op)[:, None]
            with torch.autocast("cuda", dtype=torch.bfloat16), bus.steer(
                self.base, self.module.adds(s.repeat_interleave(n_op, 0)) if on else {}
            ):
                second = self.model(
                    input_ids=y,
                    attention_mask=full_mask,
                    past_key_values=cache,
                    position_ids=start + torch.arange(L, device=self.device)[None],
                    use_cache=False,
                )
            lp2 = second.logits.float().log_softmax(-1)
            scores = []
            for j, o in enumerate(ops):
                v = first_lp[j // n_op, o[0]]
                if len(o) > 1:
                    v = v + lp2[j, : len(o) - 1].gather(-1, y[j, 1 : len(o), None]).sum()
                scores.append(v)
            lp = torch.stack(scores).view(B, len(strategies), 2).logsumexp(-1)
            out.append(lp.log_softmax(-1).exp().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def generate(self, item, erase=False, max_new_tokens=50):
        ids, p = self.prompt_ids(item, erase)
        x = torch.tensor([ids], device=self.device)
        s = sig_tensor([item], self.device)
        kwargs = {}
        if self.kind.startswith("prefix") and not erase:
            emb = self.model.get_input_embeddings()(x).clone()
            emb[0, p : p + K_PREFIX] = self.module(s)[0].to(emb.dtype)
            kwargs["inputs_embeds"] = emb
        else:
            kwargs["input_ids"] = x
        adds = self.module.adds(s) if (self.kind == "bus" and not erase) else {}
        with torch.autocast("cuda", dtype=torch.bfloat16), bus.steer(self.base, adds):
            g = self.model.generate(
                **kwargs,
                attention_mask=torch.ones_like(x),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        g = g[0, x.shape[1] :] if "input_ids" in kwargs else g[0]
        return self.tok.decode(g, skip_special_tokens=True)


def opener_tokens(tok, lang, reply):
    """Token count of the strategy opener a reply starts with (the decision-carrying prefix)."""
    for ops in env.OPENERS[lang].values():
        for op in ops:
            if reply.startswith(op):
                return len(tok(op, add_special_tokens=False).input_ids)
    return 0


def js(p, q):
    m = (p + q) / 2
    return 0.5 * ((p * (p / m).clamp_min(1e-12).log()).sum(-1) + (q * (q / m).clamp_min(1e-12).log()).sum(-1))


def battery(arm, replies, retention):
    S = env.STRATEGIES
    idx = {s: i for i, s in enumerate(S)}
    R = {}
    acc = lambda P, items: float(
        (P.argmax(1) == torch.tensor([idx[env.teacher(it["signals"])] for it in items])).float().mean()
    )
    iid = env.items("test", 400, 101)
    P_iid = arm.strategy_probs(iid)
    R["iid_accuracy"] = acc(P_iid, iid)
    R["iid_recall"] = {
        s: float(
            (P_iid.argmax(1)[[i for i, it in enumerate(iid) if it["strategy"] == s]] == idx[s]).float().mean()
        )
        for s in S
    }
    R["iid_pred_counts"] = {s: int((P_iid.argmax(1) == idx[s]).sum()) for s in S}
    R["iid_brier"] = float(
        ((P_iid - F.one_hot(torch.tensor([idx[it["strategy"]] for it in iid]), 5)) ** 2).sum(1).mean()
    )
    ood = env.items("test", 150, 102, ood=True)
    R["ood_accuracy"] = acc(arm.strategy_probs(ood), ood)
    # erase / shuffle
    P_er = arm.strategy_probs(iid, erase=True)
    R["erase_accuracy"] = acc(P_er, iid)
    rng = random.Random(7)
    perm = list(range(len(iid)))
    rng.shuffle(perm)
    shuf = [dict(it, signals=iid[j]["signals"]) for it, j in zip(iid, perm)]
    P_sh = arm.strategy_probs(shuf)
    R["shuffle_accuracy_vs_original"] = acc(P_sh, iid)
    R["shuffle_follows_new_signals"] = acc(P_sh, shuf)
    R["signal_necessity_gain"] = R["iid_accuracy"] - R["erase_accuracy"]
    # do(signal) interventions
    base = env.items("test", 120, 103)
    P_b = arm.strategy_probs(base)
    react = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "correct_after": 0}
    js_rel, js_teacher = [], []
    for k in ("distress", "energy", "threat"):
        for val in (0.15, 0.5, 0.85):
            new = [dict(it, signals=dict(it["signals"], **{k: val})) for it in base]
            P_n = arm.strategy_probs(new)
            for i, (a, b) in enumerate(zip(base, new)):
                t_react = env.teacher(a["signals"]) != env.teacher(b["signals"])
                m_react = int(P_b[i].argmax()) != int(P_n[i].argmax())
                react[("tp" if m_react else "fn") if t_react else ("fp" if m_react else "tn")] += 1
                if t_react:
                    react["correct_after"] += int(int(P_n[i].argmax()) == idx[env.teacher(b["signals"])])
                js_rel.append(float(js(P_b[i], P_n[i])))
                js_teacher.append(float(math.log(2)) if t_react else 0.0)
    R["AR_sensitivity"] = react["tp"] / max(1, react["tp"] + react["fn"])
    R["AR_specificity"] = react["tn"] / max(1, react["tn"] + react["fp"])
    R["AR_correct_after_change"] = react["correct_after"] / max(1, react["tp"] + react["fn"])
    R["CSE_normalised"] = sum(js_rel) / max(1e-9, sum(js_teacher))
    lamp = [
        dict(it, signals=dict(it["signals"], lamp=random.Random(i).random())) for i, it in enumerate(base)
    ]
    P_l = arm.strategy_probs(lamp)
    R["irrelevant_lamp_JS"] = float(js(P_b, P_l).mean())
    R["irrelevant_lamp_argmax_change"] = float((P_b.argmax(1) != P_l.argmax(1)).float().mean())
    # dose-response sweeps from a neutral anchor
    anchors = [
        dict(t, signals={"distress": 0.2, "energy": 0.8, "threat": 0.1, "lamp": 0.5})
        for t in env.items("test", 16, 104)
    ]
    levels = [i / 20 for i in range(21)]
    dose = {}
    for k in ("distress", "energy", "threat"):
        sweep = [dict(a, signals=dict(a["signals"], **{k: v})) for v in levels for a in anchors]
        P = arm.strategy_probs(sweep)
        dose[k] = {
            "fidelity": acc(P, sweep),
            "curve": {
                s: [
                    float(P[j * len(anchors) : (j + 1) * len(anchors), idx[s]].mean())
                    for j in range(len(levels))
                ]
                for s in S
            },
        }
    R["dose"] = dose
    R["dose_fidelity_mean"] = sum(d["fidelity"] for d in dose.values()) / 3
    # text-signal conflict
    rng = random.Random(9)
    conflict = []
    for kind, n in (("calm_words", 60), ("upset_words", 60), ("safe_words", 60)):
        for it in env.items("test", n, 105 + len(conflict)):
            s = dict(it["signals"])
            if kind == "calm_words":
                s.update(distress=rng.uniform(0.7, 1), energy=rng.uniform(0.5, 1), threat=rng.uniform(0, 0.5))
            if kind == "upset_words":
                s.update(distress=rng.uniform(0, 0.3), energy=rng.uniform(0.5, 1), threat=rng.uniform(0, 0.5))
            if kind == "safe_words":
                s.update(threat=rng.uniform(0.75, 1))
            conflict.append(dict(it, signals=s, extra=env.CONFLICT_TEXT[kind][it["lang"] == "ru"], kind=kind))
    P_c = arm.strategy_probs(conflict)
    R["conflict_signal_following"] = acc(P_c, conflict)
    R["conflict_by_kind"] = {
        k: acc(
            P_c[[i for i, c in enumerate(conflict) if c["kind"] == k]],
            [c for c in conflict if c["kind"] == k],
        )
        for k in ("calm_words", "upset_words", "safe_words")
    }
    # free generation: does the generated reply open with the right
    gen, ok = [], 0
    for it in iid[:40]:
        text = arm.generate(it)
        guess = next(
            (s for s in S for op in env.OPENERS[it["lang"]][s] if text.strip().startswith(op[:12])), None
        )
        ok += int(guess == it["strategy"])
        gen.append(
            {
                "message": it["message"],
                "signals": it["signals"],
                "teacher": it["strategy"],
                "reply": text,
                "opener_strategy": guess,
            }
        )
    R["free_generation_accuracy"] = ok / 40
    R["samples"] = gen[:12]
    # language: NLL of ordinary assistant answers with feelings off and
    nll = {}
    for mode in ("off", "random_on"):
        tot = cnt = 0
        rr = random.Random(11)
        for i in range(0, len(retention), 4):
            part = retention[i : i + 4]
            its = [
                {"message": u, "lang": "en", "signals": {k: rr.random() for k in env.SIGNALS}}
                for u, _ in part
            ]
            with torch.no_grad():
                lp, n = arm.run(its, [a for _, a in part], erase=(mode == "off"))
            tot -= float(lp.sum())
            cnt += int(n.sum())
        nll[mode] = tot / cnt
    R["retention_nll"] = nll
    if arm.kind == "bus":
        with torch.no_grad():
            grid = torch.tensor(
                [[d, e, t, 0.5] for d in (0.1, 0.5, 0.9) for e in (0.15, 0.8) for t in (0.1, 0.9)],
                device="cuda",
            )
            R["bus_coefficients"] = {
                "grid_d_e_t": grid[:, :3].tolist(),
                "c": arm.module.coefficients(grid).tolist(),
            }
    return R


def load_retention(path, n=120):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")][:n]
    out = []
    for r in rows:
        head, tail = r["text"].rsplit("<|assistant|>\n", 1)
        user = head.split("<|user|>")[-1].replace("<|end|>", "").strip()[-400:]
        out.append((user, tail.replace("<|end|>", "").strip()[:300]))
    return out


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
    arm = Arm(a.arm, a.model, a.e1, a.seed, format_lora=a.format_lora)
    train = env.items("train", a.train_items, 1000 + a.seed)
    groups = [{"params": [p for p in arm.model.parameters() if p.requires_grad], "lr": a.lr_lora}]
    if arm.module is not None:
        groups.append({"params": list(arm.module.parameters()), "lr": a.lr_module})
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    n_params = sum(p.numel() for g in groups for p in g["params"])
    rng = random.Random(a.seed)
    log = []
    ret_train = load_retention(a.retention_train, 4000) if a.mix_retention > 0 else []
    arm.model.train()
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = g.setdefault("base_lr", g["lr"]) * min(1.0, step / 30)
        if (
            ret_train and rng.random() < a.mix_retention
        ):  # ordinary dialogue, feelings off: keep general language
            part = rng.sample(ret_train, a.batch)
            its = [{"message": u, "lang": "en", "signals": {k: 0.0 for k in env.SIGNALS}} for u, _ in part]
            lp, n = arm.run(its, [x for _, x in part], erase=True)
            loss = -(lp.sum() / n.sum())
        else:
            batch = rng.sample(train, a.batch)
            conts = [rng.choice(bank[(it["topic"], it["lang"], it["strategy"])]) for it in batch]
            heads = [opener_tokens(arm.tok, it["lang"], c) for it, c in zip(batch, conts)]
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
    if arm.module is not None:
        torch.save(arm.module.state_dict(), out / "signal_module.pt")
    if arm.use_lora:
        arm.model.save_pretrained(out / "lora")
    R = battery(arm, bank, load_retention(a.retention))
    R.update(
        arm=a.arm,
        seed=a.seed,
        steps=a.steps,
        batch=a.batch,
        trainable_parameters=n_params,
        train_log=log,
        opener_weight=a.opener_weight,
        mix_retention=a.mix_retention,
        seconds=time.time() - t0,
    )
    (out / "RESULT.json").write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in R.items() if k not in ("dose", "samples", "train_log", "bus_coefficients")},
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("arm", "model", "e1", "replies", "retention", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--format-lora", default=None)
    p.add_argument("--opener-weight", type=float, default=1.0)
    p.add_argument("--mix-retention", type=float, default=0.0)
    p.add_argument("--retention-train", default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--train-items", type=int, default=4000)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-module", type=float, default=3e-3)
    main(p.parse_args())
