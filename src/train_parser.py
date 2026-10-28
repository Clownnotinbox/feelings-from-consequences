"""Stage F, check F3: fine-tune Qwen2.5-1.5B-Instruct."""

import argparse, json, math, random, sys, time
from pathlib import Path
import mission_parser as MP


def load(path, limit=0):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    return rows[:limit] if limit else rows


def handwritten():
    import handwritten_ru as H

    return [
        {
            "text": t,
            "truth": {"steps": st, "return_home": r},
            "split": "handwritten",
            "lang": "ru",
            "detail": "free",
        }
        for t, st, r in H.CASES
    ]


def msgs(text, answer=None, shots=()):
    m = [{"role": "system", "content": MP.SYSTEM}]
    for t, a in shots:
        m += [{"role": "user", "content": t}, {"role": "assistant", "content": a}]
    m.append({"role": "user", "content": text})
    if answer is not None:
        m.append({"role": "assistant", "content": answer})
    return m


def generate(model, tok, rows, bs=16, shots=(), max_new=320):
    import torch

    tok.padding_side = "left"
    out = []
    for i in range(0, len(rows), bs):
        B = rows[i : i + bs]
        prompts = [
            tok.apply_chat_template(msgs(r["text"], shots=shots), tokenize=False, add_generation_prompt=True)
            for r in B
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            g = model.generate(
                **enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True
            )
        out += tok.batch_decode(g[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
    return out


def evaluate(model, tok, rows, shots=(), bs=16):
    t0 = time.time()
    outs = generate(model, tok, rows, bs=bs, shots=shots)
    res = {}
    per = []
    for r, o in zip(rows, outs):
        p = MP.parse_output(o)
        s = MP.score(p, r["truth"])
        per.append(
            {
                "text": r["text"],
                "out": o,
                "ok": s["exact"],
                "split": r.get("split"),
                "lang": r.get("lang"),
                "detail": r.get("detail"),
            }
        )
        for key in (
            "all",
            f"{r.get('split')}",
            f"{r.get('split')}|{r.get('lang')}",
            f"{r.get('split')}|{r.get('detail')}",
        ):
            a = res.setdefault(key, {"n": 0, "exact": 0, "tp": 0, "np": 0, "nt": 0, "no_json": 0})
            a["n"] += 1
            a["exact"] += s["exact"]
            a["tp"] += s["tp"]
            a["np"] += s["np"]
            a["nt"] += s["nt"]
            a["no_json"] += p is None
    for a in res.values():
        a["exact_rate"] = round(a["exact"] / a["n"], 4)
        a["field_precision"] = round(a["tp"] / max(a["np"], 1), 4)
        a["field_recall"] = round(a["tp"] / max(a["nt"], 1), 4)
    return res, per, round(time.time() - t0, 1)


def train(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False  # memory: a 150k vocabulary and 640-token examples
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    tr, va = load(a.data, a.limit), load(a.val, 400)

    def encode(r):
        full = tok.apply_chat_template(msgs(r["text"], MP.dumps(r["truth"])), tokenize=False)
        prompt = tok.apply_chat_template(msgs(r["text"]), tokenize=False, add_generation_prompt=True)
        ids = tok(full, add_special_tokens=False).input_ids[:640]
        n_p = len(tok(prompt, add_special_tokens=False).input_ids)
        return ids, [-100] * n_p + ids[n_p:]

    data = [encode(r) for r in tr]
    print("train", len(data), "val", len(va), "max len", max(len(i) for i, _ in data), flush=True)
    bs, accum = a.bs, a.accum
    steps = math.ceil(len(data) / bs / accum) * a.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    sch = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = {"epochs": []}
    best = -1
    step = 0
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        random.Random(ep).shuffle(data)
        tot = 0.0
        n = 0
        for i in range(0, len(data), bs):
            B = data[i : i + bs]
            L = max(len(x) for x, _ in B)
            ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in B], device="cuda")
            lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in B], device="cuda")
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in B], device="cuda")
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / accum
            loss.backward()
            tot += loss.item() * accum
            n += 1
            if n % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
                step += 1
            if n % 200 == 0:
                print(f"ep {ep} batch {n} loss {tot / n:.4f} {time.time() - t0:.0f}s", flush=True)
        model.eval()
        res, _, sec = evaluate(model, tok, va)
        log["epochs"].append(
            {"epoch": ep, "train_loss": round(tot / n, 4), "val": res["all"], "val_seconds": sec}
        )
        print("epoch", ep, json.dumps(log["epochs"][-1]), flush=True)
        if res["all"]["exact_rate"] > best:
            best = res["all"]["exact_rate"]
            model.save_pretrained(out / "best")
            log["best_epoch"] = ep
        (out / "train_log.json").write_text(json.dumps(log, indent=1))
    tok.save_pretrained(out / "best")
    print("best epoch", log["best_epoch"], "val exact", best, flush=True)


def evaluate_cmd(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    base = a.base or a.model
    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    if a.base:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.model)
    model.eval()
    rows = load(a.data, a.limit) + handwritten()
    shots = ()
    if a.fewshot:  # zero-shot baseline gets 4 worked examples
        ex = load(a.fewshot, 400)
        ex = [r for r in ex if r["lang"] == "ru" and r["detail"] == "full"][:2] + [
            r for r in ex if r["lang"] == "en" and r["detail"] == "medium"
        ][:2]
        shots = [(r["text"], MP.dumps(r["truth"])) for r in ex]
    res, per, sec = evaluate(model, tok, rows, shots=shots, bs=8 if a.fewshot else 16)
    Path(a.out).write_text(
        json.dumps(
            {
                "model": a.model,
                "base": a.base,
                "fewshot": bool(a.fewshot),
                "seconds": sec,
                "results": res,
                "examples": per,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(json.dumps({k: v for k, v in res.items() if "|" not in k}, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("a1")
    ap.add_argument("a2")
    ap.add_argument("a3")
    ap.add_argument("a4", nargs="?")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base")
    ap.add_argument("--fewshot")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    x = ap.parse_args()
    if x.cmd == "train":
        x.data, x.val, x.out, x.model = x.a1, x.a2, x.a3, x.a4
        train(x)
    else:
        x.model, x.data, x.out = x.a1, x.a2, x.a3
        evaluate_cmd(x)
