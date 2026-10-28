"""Stage F, check F3b.
usage: paraphrase_data.py MODEL IN.jsonl OUT.jsonl N [SEED]"""

import json, random, sys, time

STYLES = {
    "ru": [
        "Перескажи это задание так, как сказал бы оператор дрона по рации: коротко, разговорно, своими словами.",
        "Перепиши задание своими словами: поменяй порядок частей и замени слова синонимами, как говорят обычные люди.",
        "Напиши то же задание неформально, как сообщение в мессенджере.",
        "Перескажи задание своими словами подробнее, как объяснял бы человек, который хорошо знает это место.",
        "Сформулируй задание как короткий приказ, без лишних слов.",
    ],
    "en": [
        "Retell this task the way a drone operator would say it over the radio: short, casual, in your own words.",
        "Rewrite the task in your own words: change the order of its parts and use synonyms, the way ordinary people talk.",
        "Write the same task informally, like a chat message.",
        "Retell the task in your own words in more detail, as someone who knows the place well would explain it.",
        "Phrase the task as a short command, no extra words.",
    ],
}
RULE = {
    "ru": "Сохрани ВСЕ факты: действие, что за объект, цвета, размеры и другие числа, знаки и рисунки, ориентиры и стороны света, названия улиц и номера домов, "
    "нужно ли возвращаться (если сказано). Ничего не добавляй и не выбрасывай. Ответь только текстом задания, на русском.",
    "en": "Keep ALL facts: the action, what the object is, colours, sizes and other numbers, marks and patterns, landmarks and directions, street names and house numbers, "
    "whether to come back (if stated). Add nothing, drop nothing. Answer with the task text only, in English.",
}
JUDGE = (
    "Text A and text B are instructions for a civil drone. Do they describe exactly the same task with the same facts (action, object, colours, sizes and numbers, "
    "marks, landmarks and directions, street names, whether to return)? Wording may differ. Answer yes or no.\n\nA: {a}\n\nB: {b}"
)


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_dir, src, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    rnd = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    rnd.shuffle(rows)
    rows = rows[:n]
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no")]
    for r in rows:
        r["style"] = rnd.randrange(len(STYLES[r["lang"]]))
    t0 = time.time()
    kept = 0
    bs = 48
    with open(out, "w", encoding="utf-8") as f:
        for i in range(0, len(rows), bs):
            B = rows[i : i + bs]
            P = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": "You rewrite task texts for a civil drone."},
                        {
                            "role": "user",
                            "content": f"{STYLES[r['lang']][r['style']]} {RULE[r['lang']]}\n\n{r['text']}",
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in B
            ]
            enc = tok(P, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                g = lm.generate(
                    **enc,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                )
            para = [
                s.strip().strip('"«»').strip()
                for s in tok.batch_decode(g[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
            ]
            J = [
                tok.apply_chat_template(
                    [{"role": "user", "content": JUDGE.format(a=r["text"], b=p)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r, p in zip(B, para)
            ]
            enc = tok(J, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                lg = (
                    lm(**enc, logits_to_keep=1).logits[:, -1].float().softmax(-1)
                )  # only the next token: full logits would need ~9 GB
            p_yes = (lg[:, yes].sum(-1) / (lg[:, yes].sum(-1) + lg[:, no].sum(-1) + 1e-9)).tolist()
            for r, p, py in zip(B, para, p_yes):
                ok = py > 0.8 and 10 < len(p) < 3 * len(r["text"]) + 80 and p != r["text"]
                if ok:
                    kept += 1
                    f.write(
                        json.dumps(
                            {
                                **r,
                                "src_text": r["text"],
                                "text": p,
                                "paraphrase": True,
                                "judge_yes": round(py, 3),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            f.flush()
            print(f"{i + len(B)}/{len(rows)} kept {kept} {time.time() - t0:.0f}s", flush=True)
    print("done kept", kept, "of", len(rows), flush=True)


if __name__ == "__main__":
    main()
