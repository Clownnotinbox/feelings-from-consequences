"""Affect bus: residual-stream emotion directions of a frozen LLM, capture and steering hooks."""

from contextlib import contextmanager
import torch

STRATEGIES = ("SOLVE", "COMFORT", "ASK", "ALERT", "DEFER")
STRATEGY_TEXT_EN = {
    "SOLVE": "give practical step-by-step advice to fix the problem",
    "COMFORT": "express warmth and emotional support before anything else",
    "ASK": "ask a clarifying question about what kind of help they want",
    "ALERT": "warn about a possible danger and put safety first",
    "DEFER": "say honestly that you are low on energy and suggest continuing later",
}


def load(path, device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    model.requires_grad_(False)
    return model, tok


def layers(model):
    return model.model.layers


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def capture(model, store):
    """store[i] <- residual stream after layer i, shape (B, T, H)."""
    handles = []
    for i, layer in enumerate(layers(model)):

        def hook(_m, _inp, out, i=i):
            store[i] = _hidden(out).detach()

        handles.append(layer.register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def steer(model, add):
    """add: dict layer -> tensor broadcastable to (B, T, H) added to the residual stream."""
    handles = []
    for i, layer in enumerate(layers(model)):
        if i not in add:
            continue

        def hook(_m, _inp, out, i=i):
            h = _hidden(out)
            h = h + add[i].to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        handles.append(layer.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def chat_prompt(tok, user, system="You are a helpful assistant.", history=()):
    msgs = [{"role": "system", "content": system}]
    for u, a in history:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def continuation_logprob(model, tok, prompts, continuations, device="cuda"):
    """Sum log p(continuation | prompt) for paired lists, batched with right padding."""
    ids, starts = [], []
    for p, c in zip(prompts, continuations):
        a = tok(p, add_special_tokens=False).input_ids
        b = tok(c, add_special_tokens=False).input_ids
        ids.append(a + b)
        starts.append((len(a), len(a) + len(b)))
    width = max(map(len, ids))
    pad = tok.pad_token_id or 0
    x = torch.full((len(ids), width), pad, dtype=torch.long)
    mask = torch.zeros_like(x)
    for i, s in enumerate(ids):
        x[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    x, mask = x.to(device), mask.to(device)
    logits = model(input_ids=x, attention_mask=mask).logits.float().log_softmax(-1)
    out = []
    for i, (a, b) in enumerate(starts):
        tgt = x[i, a:b]
        lp = logits[i, a - 1 : b - 1].gather(-1, tgt[:, None]).sum()
        out.append(lp)
    return torch.stack(out)
