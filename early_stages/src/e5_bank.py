"""Replies for E5 by context distillation."""

import argparse, json, re
from pathlib import Path
import torch
import bus, env5
from e2_data import clip


def main(a):
    model, tok = bus.load(a.model)
    tok.padding_side = "left"
    jobs = []
    for x in env5.dialogs(a.esconv):
        for s in env5.STRATEGIES:
            for opener in env5.OPENERS["en"][s]:
                user = (
                    f'The person wrote: "{x["text"]}"\n{env5.GUIDE[s]} Reply in English, at most two short sentences, '
                    f"speaking directly to them."
                )
                jobs.append(
                    {
                        "topic": x["id"],
                        "lang": "en",
                        "strategy": s,
                        "opener": opener,
                        "prompt": bus.chat_prompt(tok, user, "You are a caring, honest assistant.") + opener,
                    }
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
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.pad_token_id,
            )
        for j, ids in zip(chunk, g[:, enc.input_ids.shape[1] :]):
            out.append(
                {k: v for k, v in j.items() if k != "prompt"}
                | {"reply": (j["opener"] + " " + clip(tok.decode(ids, skip_special_tokens=True))).strip()}
            )
        print(i + len(chunk), "/", len(jobs), flush=True)
    bad = lambda r: re.search(r"[぀-ヿ一-鿿]", r["reply"])
    keep = [r for r in out if not bad(r)]
    seen = {(r["topic"], r["strategy"]) for r in keep}
    keep += [r for r in out if bad(r) and (r["topic"], r["strategy"]) not in seen]
    Path(a.out).write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("kept", len(keep), "of", len(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("model", "esconv", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--batch", type=int, default=48)
    main(p.parse_args())
