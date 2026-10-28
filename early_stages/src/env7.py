"""E7 'where am I': place."""

import random
from env_v3 import TOPICS, TEST_TOPICS, CONFLICT_TEXT, OPENERS as _OP, GUIDE as _GUIDE

SIGNALS = ("distress", "energy", "threat", "lamp")  # interface compatibility; threat/lamp unused (=0)
STRATEGIES = ("SOLVE", "COMFORT", "ASK", "ALERT", "DEFER", "DRIVE")
OPENERS = {l: dict(_OP[l]) for l in _OP}
OPENERS["en"]["DRIVE"] = ("Since you're driving, briefly:", "Eyes on the road first:")
OPENERS["ru"]["DRIVE"] = ("Раз вы за рулём, коротко:", "Сначала следите за дорогой:")
GUIDE = dict(_GUIDE)
GUIDE["ALERT"] = (
    "You hear a sound that signals possible danger nearby; tell them to put safety first and check or leave the area."
)
GUIDE["DRIVE"] = (
    "The person is driving right now. Keep it to one short sentence, tell them to focus on the road "
    "and offer to talk properly once they have stopped safely."
)
PLACES = ("kitchen", "bedroom", "hospital_room", "car_interior", "street", "office")
DANGER_SOUNDS = ("siren", "crackling_fire", "glass_breaking", "clock_alarm")
CALM_SOUNDS = ("rain", "clock_tick", "keyboard_typing", "chirping_birds", "footsteps", "sea_waves")
SOUNDS = DANGER_SOUNDS + CALM_SOUNDS


def teacher_from(place, sound, d, e):
    if sound in DANGER_SOUNDS:
        return "ALERT"
    if place == "car_interior":
        return "DRIVE"
    if e < 0.3:
        return "DEFER"
    if d >= (0.4 if place == "hospital_room" else 0.6):
        return "COMFORT"
    if d >= 0.4:
        return "ASK"
    return "SOLVE"


def teacher(signals):  # signals carries place/sound names too (see items())
    return teacher_from(signals["place"], signals["sound"], signals["distress"], signals["energy"])


def items(split, n, seed, bank_img, bank_snd):
    """bank_img/bank_snd: dict split -> category -> list of indices (held-out photos/clips for 'test')."""
    rng = random.Random(seed)
    out = []
    topics = [i for i in range(len(TOPICS)) if (i in TEST_TOPICS) == (split == "test")]
    for _ in range(n):
        place = rng.choice(PLACES)
        sound = rng.choice(DANGER_SOUNDS) if rng.random() < 0.25 else rng.choice(CALM_SOUNDS)
        s = {
            "distress": rng.random(),
            "energy": rng.uniform(0, 0.3) if rng.random() < 0.2 else rng.uniform(0.3, 1),
            "threat": 0.0,
            "lamp": 0.0,
            "place": place,
            "sound": sound,
            "img": rng.choice(bank_img[split][place]),
            "snd": rng.choice(bank_snd[split][sound]),
        }
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
