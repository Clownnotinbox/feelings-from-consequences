"""E2 replies by context distillation."""

import argparse, json, re
from pathlib import Path
import torch
import bus, env_v3 as env


def clip(text, max_sentences=2):
    text = text.strip().split("\n")[0]
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:max_sentences]).strip()


def main(a):
    model, tok = bus.load(a.model)
    tok.padding_side = "left"
    jobs = []
    for k, pair in enumerate(env.TOPICS):
        for lang in ("en", "ru"):
            msg = pair[lang == "ru"]
            for s in env.STRATEGIES:
                for opener in env.OPENERS[lang][s]:
                    for v in range(a.variants):
                        language = "English" if lang == "en" else "Russian"
                        user = (
                            f'The person wrote: "{msg}"\n{env.GUIDE[s]} Reply in {language}, at most two short '
                            f"sentences, speaking directly to them."
                        )
                        prompt = bus.chat_prompt(tok, user, "You are a caring, honest assistant.") + opener
                        jobs.append(
                            {"topic": k, "lang": lang, "strategy": s, "opener": opener, "prompt": prompt}
                        )
    out = []
    for i in range(0, len(jobs), a.batch):
        chunk = jobs[i : i + a.batch]
        enc = tok(
            [j["prompt"] for j in chunk], return_tensors="pt", padding=True, add_special_tokens=False
        ).to("cuda")
        torch.manual_seed(i)
        with torch.no_grad():
            g = model.generate(
                **enc,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )
        for j, ids in zip(chunk, g[:, enc.input_ids.shape[1] :]):
            cont = clip(tok.decode(ids, skip_special_tokens=True))
            out.append(
                {k: v for k, v in j.items() if k != "prompt"} | {"reply": (j["opener"] + " " + cont).strip()}
            )
        print(i + len(chunk), "/", len(jobs), flush=True)
    # Drop code-switching garbage
    bad = lambda r: re.search(r"[぀-ヿ一-鿿]", r["reply"]) or (
        r["lang"] == "ru" and re.search(r"[A-Za-z]{4,}", r["reply"][len(r["opener"]) :])
    )
    keep, seen = [], set()
    for r in out:
        if not bad(r):
            keep.append(r)
            seen.add((r["topic"], r["lang"], r["strategy"]))
    keep += [r for r in out if bad(r) and (r["topic"], r["lang"], r["strategy"]) not in seen]
    print("filtered", len(out) - len(keep), "of", len(out), flush=True)
    Path(a.out).write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variants", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    main(p.parse_args())
