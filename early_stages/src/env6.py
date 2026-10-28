"""E6 'voice -> feeling -> decision'."""

import random
from env_v3 import TOPICS, TEST_TOPICS, OPENERS, STRATEGIES, SIGNALS, teacher as e2_teacher

EMOTIONS = (
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
)  # RAVDESS codes 01..08
TEST_ACTORS = set(range(19, 25))  # held-out speakers


def distress(emotion, intensity):
    if emotion in ("neutral", "calm", "happy"):
        return 0.15
    if emotion in ("surprised", "disgust"):
        return 0.5
    if emotion == "angry":
        return 0.5 if intensity == "normal" else 0.75
    return 0.85  # sad, fearful


def teacher(s):
    return e2_teacher(
        {"distress": distress(s["emotion"], s["intensity"]), "energy": s["energy"], "threat": s["threat"]}
    )


def items(split, n, seed, clips):
    """clips: dict split -> list of (index, emotion, intensity, actor)."""
    rng = random.Random(seed)
    out = []
    topics = [i for i in range(len(TOPICS)) if (i in TEST_TOPICS) == (split == "test")]
    pool = clips[split]
    for _ in range(n):
        idx, emo, inten, actor = rng.choice(pool)
        s = {
            "energy": rng.uniform(0, 0.3) if rng.random() < 0.2 else rng.uniform(0.3, 1),
            "threat": rng.uniform(0.7, 1) if rng.random() < 0.2 else rng.uniform(0, 0.6),
            "lamp": 0.0,
            "clip": idx,
            "emotion": emo,
            "intensity": inten,
            "actor": actor,
        }
        s["distress"] = distress(emo, inten)
        k = rng.choice(topics)
        lang = rng.choice(("en", "ru"))
        out.append(
            {
                "topic": k,
                "lang": lang,
                "message": TOPICS[k][lang == "ru"],
                "signals": s,
                "strategy": teacher(s),
            }
        )
    return out
