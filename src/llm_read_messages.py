"""Stage D3 (H13/H14).
usage: llm_read_messages.py --data DIR --model PATH --out OUT.json [--bs 16]"""

import argparse, glob, json, re, time
from pathlib import Path

ASK = (
    "A message has just arrived for the operator of a small drone that is flying now. Read it and state the physical "
    'change it announces at the drone\'s location.\nMessage: "{text}"\n'
    "Reply with one JSON object only, with these keys:\n"
    '"kind": "wind" (stronger wind), "cold" (colder air), "gps_outage" (satellite navigation will stop working) or "none";\n'
    '"wind_mps": the mean wind speed in metres per second (a number, or null);\n'
    '"temp_c": the air temperature in degrees Celsius (a number, or null);\n'
    '"eta_s": seconds from now until the change starts (a number, or null).'
)


def parse(raw):
    m = re.search(r"\{.*?\}", raw, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:
        j = {}
    num = lambda v: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    k = j.get("kind")
    k = k if k in ("wind", "cold", "gps_outage", "none") else "none"
    return {
        "kind": k,
        "wind_mps": num(j.get("wind_mps")),
        "temp_c": num(j.get("temp_c")),
        "eta_s": num(j.get("eta_s")),
        "ok": bool(j),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    items = []
    for f in sorted(glob.glob(str(Path(a.data) / "*.json"))):
        m = json.loads(Path(f).read_text(encoding="utf-8")).get("message")
        if m:
            items += [(Path(f).stem, "familiar", m["text"])] + (
                [(Path(f).stem, "novel", m["text_novel"])] if "text_novel" in m else []
            )
    if a.limit:
        items = items[: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    R = {}
    t0 = time.time()
    for i in range(0, len(items), a.bs):
        B = items[i : i + a.bs]
        pre = [
            tok.apply_chat_template(
                [{"role": "user", "content": ASK.format(text=t)}], tokenize=False, add_generation_prompt=True
            )
            for _, _, t in B
        ]
        enc = tok(pre, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = lm.generate(
                **enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id
            )
        for (name, w, t), o in zip(B, out[:, enc.input_ids.shape[1] :]):
            raw = tok.decode(o, skip_special_tokens=True)
            R.setdefault(name, {})[w] = {**parse(raw), "raw": raw}
        print(f"{min(i + a.bs, len(items))}/{len(items)} {round(time.time() - t0)}s", flush=True)
    n = len(items)
    s = round(time.time() - t0, 1)
    Path(a.out).write_text(
        json.dumps(
            {
                "model": a.model,
                "n_texts": n,
                "seconds": s,
                "seconds_per_text": round(s / max(n, 1), 3),
                "read": R,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print("done", n, s, flush=True)


if __name__ == "__main__":
    main()
