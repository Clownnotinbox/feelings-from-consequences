"""E5 'words + feeling' on real emotional-support openings (ESConv, Liu et al. 2021)."""

import json, random
from pathlib import Path
from env_v3 import OPENERS, SIGNALS  # noqa: F401  (interface for e2)

STRATEGIES = ("SOLVE", "ASK", "COMFORT")
GUIDE = {
    "SOLVE": "Give two or three concrete practical steps for their situation.",
    "ASK": "Ask one gentle question about what kind of help they want right now.",
    "COMFORT": "Acknowledge their feelings warmly and show you are with them; do not give a plan yet.",
}
GENERIC = "Hi, I'd like to talk about something."


def sigma(r):
    return 0.02 + 0.4 * (1 - r)


def teacher_from(I):
    return "SOLVE" if I <= 3 else ("ASK" if I == 4 else "COMFORT")


def teacher(s):
    return teacher_from(s["I"])


def dialogs(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for k, x in enumerate(d):
        msgs = [t["content"].strip() for t in x["dialog"] if t["speaker"] == "seeker"]
        text = ""
        for m in msgs:
            text = (text + " " + m).strip()
            if len(text.split()) >= 12:
                break
        out.append(
            {
                "id": k,
                "text": text[:400],
                "I": int(x["survey_score"]["seeker"]["initial_emotion_intensity"]),
                "emotion": x["emotion_type"],
                "split": "test" if k % 4 == 0 else "train",
            }
        )
    return out


def items(split, n, seed, dl, generic=False):
    rng = random.Random(seed)
    pool = [x for x in dl if x["split"] == split]
    out = []
    for _ in range(n):
        x = rng.choice(pool)
        r = rng.random()
        d = (x["I"] - 1) / 4
        s = {
            "I": x["I"],
            "reliability": r,
            "sensor": min(1.0, max(0.0, d + rng.gauss(0, sigma(r)))),
            "distress": d,
            "energy": 1.0,
            "threat": 0.0,
            "lamp": 0.0,
        }
        out.append(
            {
                "topic": x["id"],
                "lang": "en",
                "message": GENERIC if generic else x["text"],
                "signals": s,
                "strategy": teacher(s),
            }
        )
    return out
