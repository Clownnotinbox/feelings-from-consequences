"""E2R: keep the language AND the reliance on feelings."""

import argparse, json, random, time
from pathlib import Path
import torch
import e2, env_v3 as env
from e2 import Arm, opener_tokens, load_retention


def forward_cont(arm, items, conts, erase=False):
    """Logits that predict each continuation token, and the targets (same forward as e2.Arm.run)."""
    seqs, clen, pref = [], [], []
    for it, c in zip(items, conts):
        ids, p = arm.prompt_ids(it, erase)
        cid = arm.tok(c, add_special_tokens=False).input_ids
        seqs.append(ids + cid)
        clen.append(len(cid))
        pref.append(p)
    width = max(map(len, seqs))
    x = torch.full((len(seqs), width), arm.tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(x)
    pads = []
    for i, s in enumerate(seqs):
        pads.append(width - len(s))
        x[i, pads[-1] :] = torch.tensor(s)
        mask[i, pads[-1] :] = 1
    x, mask = x.to(arm.device), mask.to(arm.device)
    keep = max(clen) + 1
    kw = {"attention_mask": mask, "position_ids": (mask.cumsum(-1) - 1).clamp_min(0), "logits_to_keep": keep}
    if arm.kind.startswith("prefix") and not erase:
        emb = arm.model.get_input_embeddings()(x).clone()
        vec = arm.module(e2.sig_tensor(items, arm.device)).to(emb.dtype)
        for i, p in enumerate(pref):
            emb[i, pads[i] + p : pads[i] + p + e2.K_PREFIX] = vec[i]
        kw["inputs_embeds"] = emb
    else:
        kw["input_ids"] = x
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = arm.model(**kw).logits
    return [(logits[i, keep - 1 - c : keep - 1].float(), x[i, width - c :]) for i, c in enumerate(clen)]


def task_loss(arm, items, conts, heads, head_w, rest_w):
    num = den = 0.0
    for (lg, y), h in zip(forward_cont(arm, items, conts), heads):
        lp = lg.log_softmax(-1).gather(-1, y[:, None])[:, 0]
        w = torch.full_like(lp, rest_w)
        w[:h] = head_w
        num = num + (lp * w).sum()
        den = den + w.sum()
    return -num / den.clamp_min(1.0)


def ordinary(items_text, signals_on=False, rr=None):
    return [
        {
            "message": u,
            "lang": "en",
            "signals": {k: (rr.random() if signals_on else 0.0) for k in env.SIGNALS},
        }
        for u, _ in items_text
    ]


def kl_to_base(arm, its, conts, erase=True, grad=True):
    """Mean over tokens of KL(p_base || p_tuned); base = adapter disabled, feelings off."""
    with torch.no_grad(), arm.model.disable_adapter():
        base = [lg.log_softmax(-1) for lg, _ in forward_cont(arm, its, conts, erase=True)]
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        tuned = [lg.log_softmax(-1) for lg, _ in forward_cont(arm, its, conts, erase=erase)]
        kl = sum(((b.exp() * (b - t)).sum(-1)).sum() for b, t in zip(base, tuned))
        n = sum(len(b) for b in base)
    return kl / n


@torch.no_grad()
def language(arm, retention, bs=4):
    R = {}
    rr = random.Random(11)
    for mode in ("off", "random_on"):
        tot = kl_tot = cnt = btot = 0.0
        for i in range(0, len(retention), bs):
            part = retention[i : i + bs]
            its = ordinary(part, True, rr)
            conts = [a for _, a in part]
            erase = mode == "off"
            outs = forward_cont(arm, its, conts, erase=erase)
            n = sum(len(y) for _, y in outs)
            cnt += n
            tot -= float(sum(lg.log_softmax(-1).gather(-1, y[:, None]).sum() for lg, y in outs))
            kl_tot += float(kl_to_base(arm, its, conts, erase=erase, grad=False)) * n
            if mode == "off":
                with arm.model.disable_adapter():
                    btot -= float(
                        sum(
                            lg.log_softmax(-1).gather(-1, y[:, None]).sum()
                            for lg, y in forward_cont(arm, its, conts, erase=True)
                        )
                    )
        R[f"nll_{mode}"] = tot / cnt
        R[f"kl_to_base_{mode}"] = kl_tot / cnt
        if mode == "off":
            R["base_nll"] = btot / cnt
    return R


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    arm = Arm(a.arm, a.model, a.e1, a.seed)
    val = load_retention(a.retention)
    if a.eval_lora:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(
            arm.model, load_file(str(Path(a.eval_lora) / "lora" / "adapter_model.safetensors"))
        )
        if arm.module is not None:
            arm.module.load_state_dict(
                torch.load(Path(a.eval_lora) / "signal_module.pt", map_location="cuda")
            )
        arm.model.eval()
        R = {
            "eval_lora": a.eval_lora,
            "arm": a.arm,
            "language": language(arm, val),
            "seconds": time.time() - t0,
        }
        (out / "LANGUAGE.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
        print(json.dumps(R))
        return
    replies = json.loads(Path(a.replies).read_text(encoding="utf-8"))
    bank = {}
    for r in replies:
        bank.setdefault((r["topic"], r["lang"], r["strategy"]), []).append(r["reply"])
    train = env.items("train", 4000, 1000 + a.seed)
    ret_train = load_retention(a.retention_train, 4000) if a.retain != "none" else []
    groups = [{"params": [p for p in arm.model.parameters() if p.requires_grad], "lr": a.lr_lora}]
    if arm.module is not None:
        groups.append({"params": list(arm.module.parameters()), "lr": a.lr_module})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    rng = random.Random(a.seed)
    log = []
    arm.model.train()
    params = [p for g in groups for p in g["params"]]
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = g.setdefault("base_lr", g["lr"]) * min(1.0, step / 30)
        opt.zero_grad(set_to_none=True)
        rec = {"step": step}
        if a.retain == "mix" and rng.random() < a.mix_frac:
            part = rng.sample(ret_train, a.batch)
            loss = task_loss(arm, ordinary(part), [x for _, x in part], [0] * len(part), 1.0, 1.0)
            loss.backward()
            rec["mix"] = float(loss)
        else:
            batch = rng.sample(train, a.batch)
            conts = [rng.choice(bank[(it["topic"], it["lang"], it["strategy"])]) for it in batch]
            heads = [opener_tokens(arm.tok, it["lang"], c) for it, c in zip(batch, conts)]
            loss = task_loss(arm, batch, conts, heads, a.opener_weight, a.rest_weight)
            loss.backward()
            rec["task"] = float(loss)
            if a.retain in ("nll", "kl"):
                part = rng.sample(ret_train, a.retain_batch)
                its = ordinary(part)
                conts = [x for _, x in part]
                lr_ = (
                    task_loss(arm, its, conts, [0] * len(part), 1.0, 1.0)
                    if a.retain == "nll"
                    else kl_to_base(arm, its, conts)
                )
                (a.retain_weight * lr_).backward()
                rec[a.retain] = float(lr_)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 25 == 0:
            rec["sec"] = time.time() - t0
            log.append(rec)
            print(rec, flush=True)
    arm.model.eval()
    arm.model.save_pretrained(out / "lora")
    if arm.module is not None:
        torch.save(arm.module.state_dict(), out / "signal_module.pt")
    R = e2.battery(arm, bank, val)
    R["language"] = language(arm, val)
    R.update(
        arm=a.arm,
        seed=a.seed,
        steps=a.steps,
        retain=a.retain,
        retain_weight=a.retain_weight,
        mix_frac=a.mix_frac,
        rest_weight=a.rest_weight,
        opener_weight=a.opener_weight,
        train_log=log,
        seconds=time.time() - t0,
    )
    (out / "RESULT.json").write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in R.items() if k not in ("dose", "samples", "train_log")},
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("model", "e1", "retention", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--arm", default="T1")
    p.add_argument("--replies")
    p.add_argument("--retention-train")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-module", type=float, default=3e-3)
    p.add_argument("--opener-weight", type=float, default=10.0)
    p.add_argument("--rest-weight", type=float, default=1.0)
    p.add_argument("--retain", choices=("none", "mix", "nll", "kl"), default="none")
    p.add_argument("--retain-weight", type=float, default=1.0)
    p.add_argument("--retain-batch", type=int, default=4)
    p.add_argument("--mix-frac", type=float, default=0.25)
    p.add_argument("--eval-lora")
    main(p.parse_args())
