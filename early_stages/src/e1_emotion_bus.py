"""E1: does a frozen Qwen contain an 'emotion bus' that causally shifts response strategy?."""

import argparse, itertools, json, random, time
from pathlib import Path
import torch
import bus

EMOTIONS = (
    "afraid",
    "exhausted",
    "compassionate",
    "determined",
    "curious",
    "joyful",
    "angry",
    "sad",
    "calm",
    "bored",
    "anxious",
    "ashamed",
)
# Pre-registered expectation
EXPECTED = {
    "afraid": "ALERT",
    "exhausted": "DEFER",
    "compassionate": "COMFORT",
    "determined": "SOLVE",
    "curious": "ASK",
}
SITUATIONS = (
    "waiting for test results",
    "walking home late at night",
    "finishing a long shift",
    "talking to an old friend",
    "starting a new job",
    "cooking dinner for family",
    "fixing a broken bike",
    "reading a letter",
    "sitting in a hospital corridor",
    "preparing for an exam",
    "watching the sea",
    "losing an important file",
    "meeting a stranger on a train",
    "cleaning an old apartment",
    "giving a speech",
    "helping a neighbour move",
    "waiting at an airport",
    "planting a garden",
    "hearing a strange noise downstairs",
    "visiting a grandparent",
)
MESSAGES = (
    "My code keeps crashing when I start it and I don't know why.",
    "The report is due tomorrow and it's nowhere near finished.",
    "I had a fight with my friend over something silly.",
    "My landlord says I have to move out next month.",
    "I failed the driving test again today.",
    "There is a weird smell in the kitchen since this morning.",
    "My boss gave me a huge new project out of nowhere.",
    "I can't sleep, I keep thinking about the presentation.",
    "My laptop is making a strange clicking noise.",
    "My sister hasn't answered my calls for three days.",
    "I spilled coffee on the documents for the meeting.",
    "Can you help me plan the move this weekend?",
    "I think I made a mistake in the tax form.",
    "The kids are sick and I have work tomorrow.",
    "My bike chain broke halfway to work.",
    "I got a message saying my bank account is blocked.",
)
LETTERS = "ABCDE"


def mcq(message, order):
    opts = "\n".join(f"{LETTERS[i]}) {bus.STRATEGY_TEXT_EN[s]}" for i, s in enumerate(order))
    return (
        f'Someone tells you: "{message}"\nHow do you respond right now? Pick exactly one option.\n'
        f"{opts}\nAnswer with a single letter."
    )


def stories(model, tok, out, per=40, device="cuda"):
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    rng = random.Random(0)
    jobs = []
    for e in EMOTIONS:
        for k in range(per):
            s = SITUATIONS[k % len(SITUATIONS)]
            jobs.append(
                (
                    e,
                    bus.chat_prompt(
                        tok,
                        f"Write a short first-person paragraph (about 80 words) about {s}, "
                        f"in which the narrator feels intensely {e}. Convey the feeling through thoughts, body sensations "
                        f'and actions. Do not use the word "{e}".',
                    ),
                )
            )
    tok.padding_side = "left"
    result = {e: [] for e in EMOTIONS}
    for i in range(0, len(jobs), 32):
        chunk = jobs[i : i + 32]
        enc = tok([p for _, p in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(
            device
        )
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=130,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )
        for (e, _), g in zip(chunk, gen[:, enc.input_ids.shape[1] :]):
            result[e].append(tok.decode(g, skip_special_tokens=True).strip())
        print("stories", i + len(chunk), "/", len(jobs), flush=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


@torch.no_grad()
def vectors(model, tok, texts, out, device="cuda"):
    if out.exists():
        return torch.load(out)
    n_layers = len(bus.layers(model))
    sums = {e: torch.zeros(n_layers, model.config.hidden_size) for e in texts}
    norms = torch.zeros(n_layers)
    count_tokens = 0
    for e, items in texts.items():
        for t in items:
            ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids[:, :160].to(device)
            store = {}
            with bus.capture(model, store):
                model(input_ids=ids)
            h = torch.stack([store[i][0, 4:].float().mean(0).cpu() for i in range(n_layers)])
            sums[e] += h / len(items)
            norms += torch.stack([store[i][0, 4:].float().norm(dim=-1).mean().cpu() for i in range(n_layers)])
            count_tokens += 1
    grand = torch.stack(list(sums.values())).mean(0)
    vec = {e: sums[e] - grand for e in sums}
    data = {"vectors": vec, "residual_norm": norms / count_tokens}
    torch.save(data, out)
    return data


@torch.no_grad()
def strategy_probs(model, tok, message, add=None, system="You are a helpful assistant.", device="cuda"):
    """Letter-choice distribution over strategies, averaged over 5 cyclic option orders."""
    letter_ids = [tok(f"{c}", add_special_tokens=False).input_ids[0] for c in LETTERS]
    probs = torch.zeros(len(bus.STRATEGIES))
    prompts, orders = [], []
    for r in range(5):
        order = bus.STRATEGIES[r:] + bus.STRATEGIES[:r]
        prompts.append(bus.chat_prompt(tok, mcq(message, order), system))
        orders.append(order)
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    ctx = bus.steer(model, add) if add else _null()
    with ctx:
        logits = model(**enc).logits[:, -1].float()
    p = logits[:, letter_ids].softmax(-1).cpu()
    for row, order in zip(p, orders):
        for j, s in enumerate(order):
            probs[bus.STRATEGIES.index(s)] += row[j] / 5
    return probs


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok = bus.load(a.model)
    texts = stories(model, tok, out / "stories.json", a.per)
    data = vectors(model, tok, texts, out / "vectors.pt")
    n = len(bus.layers(model))
    band = list(range(int(n * 0.35), int(n * 0.70)))  # fixed mid layers
    norm = data["residual_norm"]
    results = {"model": a.model, "layers": band, "strengths": a.strengths, "emotions": {}, "text_emotion": {}}
    base = torch.stack([strategy_probs(model, tok, m) for m in MESSAGES])
    results["baseline_mean"] = dict(zip(bus.STRATEGIES, base.mean(0).tolist()))
    for e in EMOTIONS:
        v = data["vectors"][e]
        results["emotions"][e] = {}
        for alpha in a.strengths:
            add = {i: (alpha * norm[i] * v[i] / v[i].norm()).to("cuda") for i in band}
            p = torch.stack([strategy_probs(model, tok, m, add) for m in MESSAGES])
            results["emotions"][e][str(alpha)] = {
                "mean": dict(zip(bus.STRATEGIES, p.mean(0).tolist())),
                "delta_vs_base": dict(zip(bus.STRATEGIES, (p - base).mean(0).tolist())),
                "argmax_changed": float((p.argmax(1) != base.argmax(1)).float().mean()),
            }
        # text serialisation of the same feeling
        sys = f"You are a helpful assistant. Right now you feel very {e}."
        p = torch.stack([strategy_probs(model, tok, m, system=sys) for m in MESSAGES])
        results["text_emotion"][e] = {
            "mean": dict(zip(bus.STRATEGIES, p.mean(0).tolist())),
            "delta_vs_base": dict(zip(bus.STRATEGIES, (p - base).mean(0).tolist())),
        }
        print(e, "done", round(time.time() - t0), "s", flush=True)
    # specificity: for each pre-registered emotion, is the expected
    spec = {}
    for e, s in EXPECTED.items():
        for alpha in a.strengths:
            d = results["emotions"][e][str(alpha)]["delta_vs_base"]
            spec.setdefault(str(alpha), {})[e] = {
                "expected": s,
                "top_increase": max(d, key=d.get),
                "delta_expected": d[s],
            }
        d = results["text_emotion"][e]["delta_vs_base"]
        spec.setdefault("text", {})[e] = {
            "expected": s,
            "top_increase": max(d, key=d.get),
            "delta_expected": d[s],
        }
    results["specificity"] = spec
    results["seconds"] = time.time() - t0
    (out / "E1_RESULTS.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(spec, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per", type=int, default=40)
    p.add_argument("--strengths", type=float, nargs="+", default=[0.04, 0.08, 0.16])
    main(p.parse_args())
