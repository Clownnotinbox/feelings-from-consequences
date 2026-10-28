"""Stage E, step 1 (FEELINGS_DESIGN_RU.md sec.
usage: stage_e_vlm.py --data DIR --model PATH --out OUT.json [--conds point|familiar,...] [--bs 8] [--limit N]
       stage_e_vlm.py --data DIR --embed PATH --out OUT.json"""

import argparse, json, re, time
from pathlib import Path
import numpy as np

FRAME = (
    'This image is from the forward camera of a small drone. The camera faces {heading}. The operator gave this instruction: "{text}" '
    "The operator's rough estimate of where the destination is lies near the centre of the image, but it can be off by tens of metres. "
    "Similar-looking objects may be present: choose the single object that matches the whole description. "
)
ASK = {
    "point": 'Output its bounding box as JSON: {"bbox_2d": [x1, y1, x2, y2]}. If the described object is not visible in the image, output {"bbox_2d": null}.',
    "som": "Some objects in the image are marked with numbered white circles. Which number marks the destination? Answer with the number only. "
    "If the destination is not among the marked objects, answer 0.",
    "zoom": "Which number marks the destination? Answer with the number only. If the destination is not among the marked objects, answer 0.",
}
ZOOM_NOTE = (
    "Some objects in the image are marked with numbered white circles. After the image come close-up views of the marked objects, "
    "taken by the same camera at higher resolution; all close-ups share one magnification, so the sizes of objects can be compared "
    "between them. In each close-up the marked object is outlined by a thin white box. "
)
NUM = r"-?\d+(?:\.\d+)?"


BOX = r"\[\s*(" + NUM + r")\s*,\s*(" + NUM + r")\s*,\s*(" + NUM + r")\s*,\s*(" + NUM + r")\s*\]"


def parse_point(raw):
    """(x1, y1, x2, y2) or None."""
    m = list(re.finditer(r'bbox_2d"?\s*:\s*(?:' + BOX + r"|(null))", raw))
    if m:
        return None if m[-1].group(5) else [float(v) for v in m[-1].groups()[:4]]
    m = list(re.finditer(BOX, raw))
    if m:
        return [float(v) for v in m[-1].groups()]
    return (
        None
        if re.search(
            r"not (be )?visible|null|does not contain|no such|cannot (be )?(found|seen|see)", raw, re.I
        )
        else "fail"
    )


def parse_som(raw):
    """A bare number, else the last number of a reasoned answer ('... so the answer is 2')."""
    m = re.match(r"\s*(\d+)", raw) or ([None] + list(re.finditer(r"\d+", raw)))[-1]
    return int(m.group(1 if m.re.groups else 0)) if m else "fail"


def pick_point(box, lab, r=20):
    """Object index under the box centre."""
    if box is None:
        return "none"
    if box == "fail":
        return "fail"
    H, W = lab.shape
    cx, cy = int(round((box[0] + box[2]) / 2)), int(round((box[1] + box[3]) / 2))
    if not (0 <= cx < W and 0 <= cy < H):
        return "miss"
    if lab[cy, cx]:
        return int(lab[cy, cx]) - 1
    y0, y1, x0, x1 = max(cy - r, 0), min(cy + r + 1, H), max(cx - r, 0), min(cx + r + 1, W)
    ys, xs = np.nonzero(lab[y0:y1, x0:x1])
    if not len(xs):
        return "miss"
    k = int(np.argmin((ys + y0 - cy) ** 2 + (xs + x0 - cx) ** 2))
    d = ((ys[k] + y0 - cy) ** 2 + (xs[k] + x0 - cx) ** 2) ** 0.5
    return int(lab[ys[k] + y0, xs[k] + x0]) - 1 if d <= r else "miss"


def pick_som(n, cands):
    if n == "fail":
        return "fail"
    if n == 0:
        return "none"
    js = [j for j, c in enumerate(cands) if c.get("mark") == n]
    return js[0] if js else "miss"


FAM = {"tent": "tent", "pad": "landing pad", "tower": "water tower", "container": "container"}


def full_phrase(c):
    """Everything true of object c, in the familiar words (for the 'embed' rival with perfect eyes)."""
    r = c["rel"]
    rel = []
    if r["bctx"]:
        rel.append(
            "in the open, away from buildings"
            if r["bctx"] == "open"
            else f"next to a {'tall' if r['bctx'] == 'near_tall' else 'low'} building"
        )
    if r["side"]:
        rel.append(f"{r['side'][0]} of a {r['side'][1]} building")
    if r["next_to"]:
        rel.append(f"next to the {r['next_to'][0]} {FAM[r['next_to'][1]]}")
    return (
        "the "
        + " ".join(w for w in (r["size"], r["colour"], FAM[r["type"]]) if w)
        + ("" if not rel else " " + " and ".join(rel))
    )


def embed(a, S):
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(a.embed, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    out = {}
    t0 = time.time()
    for s in S:
        vis = [j for j, c in enumerate(s["cands"]) if c["area"] >= 40]
        ph = [full_phrase(s["cands"][j]) for j in vis]
        E = enc.encode([s["text_familiar"], s["text_novel"]] + ph, normalize_embeddings=True)
        sc = E[:2] @ E[2:].T if ph else np.zeros((2, 0))
        out[s["name"]] = {
            "visible": vis,
            "familiar": sc[0].round(4).tolist(),
            "novel": sc[1].round(4).tolist(),
        }
    return {"model": a.embed, "seconds": round(time.time() - t0, 1), "scores": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model")
    ap.add_argument("--embed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default="point|familiar,point|novel,som|familiar,som|novel")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--prompt-only", action="store_true")  # print one prompt per mode, no model
    ap.add_argument("--quant4", action="store_true")  # language part in 4 bits
    a = ap.parse_args()
    D = Path(a.data)
    S = [json.loads(l) for l in open(D / "scenes.jsonl", encoding="utf-8")]
    if a.limit:
        S = S[: a.limit]
    if a.embed:
        Path(a.out).write_text(json.dumps(embed(a, S)), encoding="utf-8")
        print("embed done", len(S), flush=True)
        return
    from PIL import Image

    if a.prompt_only:  # a dry check of the message structure without the model
        for cond in a.conds.split(","):
            mode, w = cond.split("|")
            s = next(x for x in S if sum(bool(c.get("mark")) for c in x["cands"]) >= 2)
            mk = sorted(c["mark"] for c in s["cands"] if c.get("mark"))
            print(
                cond,
                s["name"],
                "images",
                1 + (len(mk) if mode == "zoom" else 0),
                [(D / "crop" / f"{s['name']}_{n}.png").exists() for n in mk] if mode == "zoom" else "",
            )
            print(
                "  "
                + FRAME.format(heading=s["heading"], text=s["text_" + w])
                + (
                    ZOOM_NOTE + " ".join(f"Close-up of object {n}: <image>" for n in mk) + " " + ASK["zoom"]
                    if mode == "zoom"
                    else ASK[mode]
                )
            )
        return
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"
    q = {}
    if a.quant4:
        from transformers import BitsAndBytesConfig

        q = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["visual", "lm_head"],
            )
        }
    lm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", **q
    ).eval()
    print("GPU memory after loading, GB", round(torch.cuda.memory_allocated() / 2**30, 2), flush=True)
    res = {
        "model": a.model,
        "quant4": a.quant4,
        "gpu": torch.cuda.get_device_name(0),
        "n_scenes": len(S),
        "conds": {},
    }

    def answer(B, mode, w):
        """Raw answers for a batch of scenes: (raw text, the frame's image grid, new tokens)."""
        ims = [
            Image.open(D / "img" / (s["name"] + ("" if mode == "point" else "_som") + ".png")).convert("RGB")
            for s in B
        ]
        if mode == "zoom":
            content, flat, first = [], [], []
            for s, im in zip(B, ims):
                mk = sorted(c["mark"] for c in s["cands"] if c.get("mark"))
                first.append(len(flat))
                flat.append(im)
                c = [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": FRAME.format(heading=s["heading"], text=s["text_" + w])
                        + (ZOOM_NOTE if mk else ASK["som"].split("Which")[0]),
                    },
                ]
                for n in mk:
                    c += [{"type": "text", "text": f"Close-up of object {n}:"}, {"type": "image"}]
                    flat.append(Image.open(D / "crop" / f"{s['name']}_{n}.png").convert("RGB"))
                content.append(c + [{"type": "text", "text": ASK["zoom"]}])
        else:
            content = [
                [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": FRAME.format(heading=s["heading"], text=s["text_" + w]) + ASK[mode],
                    },
                ]
                for s in B
            ]
            flat = ims
            first = list(range(len(B)))
        txt = [
            proc.apply_chat_template(
                [{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True
            )
            for c in content
        ]
        enc = proc(text=txt, images=flat, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = lm.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
        gen = out[:, enc.input_ids.shape[1] :]
        g = enc["image_grid_thw"].tolist()
        return list(
            zip(
                proc.batch_decode(gen, skip_special_tokens=True),
                [g[k] for k in first],
                (gen != proc.tokenizer.pad_token_id).sum(1).tolist(),
                ims,
            )
        )

    for cond in a.conds.split(","):
        mode, w = cond.split("|")
        R = {}
        t0 = time.time()
        n_oom = 0
        for i in range(0, len(S), a.bs):
            B = S[i : i + a.bs]
            try:
                outs = answer(B, mode, w)
            except torch.cuda.OutOfMemoryError:  # a batch with many close-ups
                torch.cuda.empty_cache()
                n_oom += 1
                outs = [o for s in B for o in answer([s], mode, w)]
            for s, (raw, (_, gh, gw), nt, im) in zip(B, outs):  # nt = max_new: the answer was cut off
                sx, sy = im.width / (gw * 14), im.height / (
                    gh * 14
                )  # the model's boxes are in the resized image's pixels
                g = [[1, gh, gw]]
                if mode == "point":
                    box = parse_point(raw)
                    box = (
                        [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy] if isinstance(box, list) else box
                    )
                    lab = np.array(Image.open(D / "seg" / (s["name"] + ".png")))
                    R[s["name"]] = {"raw": raw, "box": box, "pick": pick_point(box, lab), "ntok": nt}
                else:
                    n = parse_som(raw)
                    R[s["name"]] = {"raw": raw, "n": n, "pick": pick_som(n, s["cands"]), "ntok": nt}
            print(f"{cond} {min(i + a.bs, len(S))}/{len(S)} {round(time.time() - t0)}s", flush=True)
        res["conds"][cond] = {
            "seconds": round(time.time() - t0, 1),
            "seconds_per_query": round((time.time() - t0) / len(S), 3),
            "resized": g[0],
            "max_new": a.max_new,
            "oom_batches": n_oom,
            "cut_off": round(sum(r["ntok"] >= a.max_new for r in R.values()) / len(R), 3),
            "answers": R,
        }
        Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
