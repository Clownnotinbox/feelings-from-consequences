"""PRISM responses -> Qwen2.5-7B representations."""

import argparse, collections, json, time
from pathlib import Path
import numpy as np, torch


def rows_from(path):
    out = []
    for line in open(path, encoding="utf-8"):
        c = json.loads(line)
        user_msg = {}
        for h in c["conversation_history"]:
            if h["role"] == "user":
                user_msg[int(h["turn"])] = h["content"]
        for h in c["conversation_history"]:
            if h["role"] == "model" and h.get("score") is not None:
                t = int(h["turn"])
                out.append(
                    {
                        "user_id": c["user_id"],
                        "conv": c["conversation_id"],
                        "time": c["generated_datetime"],
                        "turn": t,
                        "within": str(h.get("within_turn_id")),
                        "model": h["model_name"],
                        "score": float(h["score"]),
                        "chosen": str(h.get("if_chosen")) == "True",
                        "prompt": user_msg.get(t, ""),
                        "response": h["content"],
                    }
                )
    return out


def main(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    R = rows_from(a.conversations)[: a.limit or None]
    print("rows", len(R), flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    feats = np.zeros((len(R), model.config.hidden_size), np.float32)
    order = np.argsort([len(r["response"]) for r in R])
    for s in range(0, len(R), a.batch):
        idx = order[s : s + a.batch]
        heads = [
            tok(f"User: {R[i]['prompt'][:600]}\nAssistant:", add_special_tokens=False).input_ids for i in idx
        ]
        bodies = [tok(" " + R[i]["response"], add_special_tokens=False).input_ids[: a.max_resp] for i in idx]
        seqs = [h[-200:] + b for h, b in zip(heads, bodies)]
        W = max(map(len, seqs))
        x = torch.full((len(seqs), W), tok.pad_token_id)
        m = torch.zeros_like(x)
        rm = torch.zeros_like(x, dtype=torch.float32)
        for j, (sq, b) in enumerate(zip(seqs, bodies)):
            x[j, : len(sq)] = torch.tensor(sq)
            m[j, : len(sq)] = 1
            rm[j, len(sq) - len(b) : len(sq)] = 1
        with torch.no_grad():
            hs = (
                model(input_ids=x.cuda(), attention_mask=m.cuda(), output_hidden_states=True)
                .hidden_states[a.layer]
                .float()
            )
        rm = rm.cuda()[..., None]
        feats[idx] = ((hs * rm).sum(1) / rm.sum(1).clamp_min(1)).cpu().numpy()
        if (s // a.batch) % 100 == 0:
            print(s, round(time.time() - t0), flush=True)
    users = sorted({r["user_id"] for r in R})
    test = set(users[4::5])
    tr = np.array([r["user_id"] not in test for r in R])
    mu = feats[tr].mean(0)
    U, S, V = torch.pca_lowrank(torch.tensor(feats[tr] - mu), q=a.dims, center=False)
    Z = ((feats - mu) @ V.numpy()).astype(np.float32)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "prism_llm.npz",
        z=Z,
        user=np.array([r["user_id"] for r in R]),
        conv=np.array([r["conv"] for r in R]),
        time=np.array([r["time"] for r in R]),
        turn=np.array([r["turn"] for r in R]),
        within=np.array([r["within"] for r in R]),
        model=np.array([r["model"] for r in R]),
        score=np.array([r["score"] for r in R], np.float32),
        chosen=np.array([r["chosen"] for r in R]),
        length=np.array([len(r["response"]) for r in R]),
        test=~tr,
    )
    (out / "EMBED.json").write_text(
        json.dumps(
            {
                "rows": len(R),
                "layer": a.layer,
                "dims": a.dims,
                "seconds": time.time() - t0,
                "explained_var_top": float((S[: a.dims] ** 2).sum() / ((feats[tr] - mu) ** 2).sum()),
            },
            indent=1,
        )
    )
    print("done", time.time() - t0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--conversations", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--dims", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-resp", type=int, default=320)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
