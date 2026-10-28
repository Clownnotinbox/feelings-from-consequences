"""R1 'the words are neutral, the feeling is in the voice' on CREMA-D, with HUMAN targets only."""

import argparse, json, math, random, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
from torch import nn
import bus, e2

EMO = ("A", "D", "F", "H", "N", "S")
EMO_WORD = dict(zip(EMO, ("angry", "disgusted", "afraid", "happy", "neutral", "sad")))
LEV = ("low", "mid", "high")
INTENDED = {
    "ANG": "angry",
    "DIS": "disgusted",
    "FEA": "afraid",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
LEVEL_WORD = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"}
SENT = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}
OPENERS = {
    "emo": {
        e: (f"They sound {EMO_WORD[e]}.", f"To listeners they come across as {EMO_WORD[e]}.") for e in EMO
    },
    "int": {
        "low": ("The feeling sounds weak.", "It comes across faintly."),
        "mid": ("The feeling sounds moderate.", "It comes across moderately."),
        "high": ("The feeling sounds strong.", "It comes across very strongly."),
    },
}
QUESTION = {"emo": "How does the speaker sound to listeners?", "int": "How strong does their feeling sound?"}
e2.env = SimpleNamespace(OPENERS=OPENERS, STRATEGIES=EMO, SIGNALS=())
e2.K_PREFIX = 4


class VoicePrefix(nn.Module):
    def __init__(self, d_in, hidden, scale, mu, sd, k=4):
        super().__init__()
        self.k = k
        self.register_buffer("mu", torch.tensor(mu))
        self.register_buffer("sd", torch.tensor(sd))
        self.net = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, k * hidden))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = scale

    def forward(self, v):
        return self.net((v - self.mu) / self.sd).view(len(v), self.k, -1) * self.scale


class ArmR1(e2.Arm):
    def __init__(self, name, model, seed, feats, mu, sd, device="cuda"):
        e2.ARMS[name] = ("text", True, None)
        super().__init__(name, model, None, seed, device)
        self.feats = feats
        if name in ("VE", "VPE"):
            emb_norm = float(self.base.get_input_embeddings().weight.float().norm(dim=-1).mean())
            self.kind = "prefix_v"
            self.module = VoicePrefix(
                len(mu), self.base.config.hidden_size, emb_norm, mu, sd, e2.K_PREFIX
            ).to(self.device)

    def prompt_ids(self, item, erase=False):
        c = item["clip_i"]
        system = e2.SYSTEM
        if not erase:
            pe = self.feats["p_emotion"][c]
            if self.name == "VN":
                system += f"\n[voice] a recognizer says: sounds {EMO_WORD[EMO[int(pe.argmax())]]} ({float(pe.max()):.2f})"
            if self.name in ("VP", "VPE"):
                system += "\n[voice] recognizer: " + ", ".join(
                    f"{EMO_WORD[e]} {float(p):.2f}" for e, p in zip(EMO, pe)
                )
            if self.name == "VA":
                system += f"\n[voice] the actor was asked to sound {INTENDED[item['intended']]}, {LEVEL_WORD[item['level']]} intensity"
            if self.kind == "prefix_v":
                system += e2.MARK
        msg = f'Someone said out loud: "{SENT[item["sentence"]]}" {QUESTION[item["lang"]]}'
        text = bus.chat_prompt(self.tok, msg, system)
        enc = lambda t: self.tok(t, add_special_tokens=False).input_ids
        if e2.MARK in text:
            a, b = text.split(e2.MARK)
            ia = enc(a)
            return ia + [self.placeholder] * e2.K_PREFIX + enc(b), len(ia)
        return enc(text), None


def load(feat_dir):
    F = dict(np.load(Path(feat_dir) / "voice.npz"))
    ratings = json.loads((Path(feat_dir) / "ratings.json").read_text(encoding="utf-8"))
    tr = F["split"] == "train"
    lv = np.array([v[1] for i in np.where(tr)[0] for v in ratings[F["clip"][i]]["voice"] if v[1] == v[1]])
    cuts = (
        np.quantile(lv, [1 / 3, 2 / 3]) if len(lv) else np.array([np.inf, np.inf])
    )  # terciles of listeners' own levels (train)
    mean_lv = lambda i: (
        np.nanmean([v[1] for v in ratings[F["clip"][i]]["voice"]])
        if ratings[F["clip"][i]]["voice"]
        else np.nan
    )
    cm = np.array([mean_lv(i) for i in np.where(tr)[0]])
    ccuts = np.nanquantile(cm, [1 / 3, 2 / 3])  # terciles of clip-mean levels (train)
    clips = []
    for i, name in enumerate(F["clip"]):
        votes = ratings[str(name)]["voice"]
        if not votes:
            continue
        m = mean_lv(i)
        clips.append(
            {
                "clip_i": i,
                "clip": str(name),
                "split": str(F["split"][i]),
                "sentence": str(F["sentence"][i]),
                "intended": str(name).split("_")[2],
                "level": str(F["level"][i]),
                "emo_votes": [EMO[int(e)] for e, _ in votes],
                "lev_votes": [LEV[int(np.digitize(l, cuts))] for _, l in votes if l == l],
                "lev_mean": float(m),
                "lev_cons": LEV[int(np.digitize(m, ccuts))] if m == m else None,
            }
        )
    return F, clips, cuts


INT_TARGET = "vote"  # 'vote': one listener's level bin


def items_for(clips, q, rng=None):
    """One scoring item per clip."""
    out = []
    for c in clips:
        votes = c["emo_votes"] if q == "emo" else c["lev_votes"]
        if not votes:
            continue
        it = dict(c, lang=q, vec=None)
        if rng is not None:
            it["answer"] = (
                c["lev_cons"]
                if (q == "int" and INT_TARGET == "consensus" and c["lev_cons"])
                else rng.choice(votes)
            )
        out.append(it)
    return out


def with_vec(its, F):
    for it in its:
        it["vec"] = F["emb"][it["clip_i"]].tolist()
    return its


def score(arm, its, F, q, erase=False):
    S = EMO if q == "emo" else LEV
    P = arm.strategy_probs(with_vec(its, F), erase=erase, strategies=S)
    idx = {s: i for i, s in enumerate(S)}
    votes = [c["emo_votes"] if q == "emo" else c["lev_votes"] for c in its]
    ll = [
        float(np.mean([math.log(max(1e-9, float(P[i, idx[v]]))) for v in vs])) for i, vs in enumerate(votes)
    ]
    emp = torch.tensor([[vs.count(s) / len(vs) for s in S] for vs in votes])
    maj = [
        (i, max(S, key=vs.count))
        for i, vs in enumerate(votes)
        if len(vs) >= 2 and sorted([vs.count(s) for s in S])[-1] > sorted([vs.count(s) for s in S])[-2]
    ]
    H = lambda p: -(p.clamp_min(1e-9) * p.clamp_min(1e-9).log()).sum(-1)
    r = {
        "vote_loglik": float(np.mean(ll)),
        "chance_loglik": math.log(1 / len(S)),
        "js_to_listeners": float(e2.js(P, emp).mean()),
        "majority_acc": float(np.mean([P[i].argmax().item() == idx[m] for i, m in maj])) if maj else None,
        "n_majority": len(maj),
        "pred_counts": {s: int((P.argmax(1) == idx[s]).sum()) for s in S},
    }
    hm, he = H(P).numpy(), H(emp).numpy()
    r["entropy_corr_with_listener_disagreement"] = (
        float(np.corrcoef(hm, he)[0, 1]) if he.std() > 0 and hm.std() > 0 else None
    )
    if q == "int":
        ex = (P * torch.tensor([0.0, 1.0, 2.0])).sum(1).numpy()
        hum = np.array([np.mean([LEV.index(v) for v in vs]) for vs in votes])
        r["expected_level_corr"] = float(np.corrcoef(ex, hum)[0, 1]) if ex.std() > 0 else None
        lm = np.array([c["lev_mean"] for c in its])
        ok = lm == lm
        r["expected_vs_mean_level_corr"] = float(np.corrcoef(ex[ok], lm[ok])[0, 1]) if ex.std() > 0 else None
        cons = [(i, c["lev_cons"]) for i, c in enumerate(its) if c.get("lev_cons")]
        r["consensus_bin_acc"] = (
            float(np.mean([P[i].argmax().item() == idx[b] for i, b in cons])) if cons else None
        )
    return r, P


def battery(arm, F, clips):
    test = [c for c in clips if c["split"] == "test"]
    R = {}
    for q in ("emo", "int"):
        its = items_for(test, q)
        R[q], P = score(arm, its, F, q)
        R[q + "_erased"], _ = score(arm, its, F, q, erase=True)
        # swap: same sentence, the voice of another test clip -> does the
        rng = random.Random(5)
        swapped = []
        for it in its:
            other = rng.choice(
                [c for c in test if c["sentence"] == it["sentence"] and c["clip"] != it["clip"]]
            )
            swapped.append(
                dict(
                    it,
                    clip_i=other["clip_i"],
                    clip=other["clip"],
                    intended=other["intended"],
                    level=other["level"],
                    emo_votes=other["emo_votes"],
                    lev_votes=other["lev_votes"],
                )
            )
        swapped = [s for s in swapped if (s["emo_votes"] if q == "emo" else s["lev_votes"])]
        R[q + "_swap_follows_new_voice"], _ = score(arm, swapped, F, q)
    return R


def main(a):
    global INT_TARGET
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    INT_TARGET = a.int_target
    F, clips, cuts = load(a.features)
    tr = F["split"] == "train"
    mu, sd = F["emb"][tr].mean(0), F["emb"][tr].std(0) + 1e-6
    arm = ArmR1(a.arm, a.model, a.seed, F, mu, sd)
    train = [c for c in clips if c["split"] == "train"]
    rng = random.Random(a.seed)
    groups = [{"params": [p for p in arm.model.parameters() if p.requires_grad], "lr": a.lr_lora}]
    if arm.module is not None:
        groups.append({"params": list(arm.module.parameters()), "lr": a.lr_module})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    log = []
    arm.model.train()
    if arm.module is not None:
        arm.module.train()
    for step in range(1, a.steps + 1):
        q = "emo" if step % 2 else "int"
        batch = with_vec(items_for(rng.sample(train, a.batch * 2), q, rng)[: a.batch], F)
        conts = [rng.choice(OPENERS[q][it["answer"]]) for it in batch]
        heads = [len(arm.tok(c, add_special_tokens=False).input_ids) for c in conts]
        for g in opt.param_groups:
            g["lr"] = g.setdefault("base_lr", g["lr"]) * min(1.0, step / 30)
        lp, n = arm.run(batch, conts, heads=heads, head_w=1.0)
        loss = -(lp.sum() / n.sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        if step % 50 == 0:
            log.append({"step": step, "q": q, "loss": float(loss), "sec": time.time() - t0})
            print(log[-1], flush=True)
    arm.model.eval()
    if arm.module is not None:
        arm.module.eval()
        torch.save(arm.module.state_dict(), out / "signal_module.pt")
    R = battery(arm, F, clips)
    R.update(
        arm=a.arm,
        seed=a.seed,
        steps=a.steps,
        int_target=a.int_target,
        level_cuts=cuts.tolist(),
        n_train_clips=len(train),
        train_log=log,
        seconds=time.time() - t0,
    )
    (out / "RESULT.json").write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in R.items() if k != "train_log"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("arm", "model", "features", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-module", type=float, default=1e-3)
    p.add_argument("--int-target", choices=("vote", "consensus"), default="vote")
    main(p.parse_args())
