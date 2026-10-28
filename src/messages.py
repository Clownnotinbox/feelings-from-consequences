"""Stage D2 (H12): words about the FUTURE world that no sensor can see yet."""

import math
import numpy as np

P_MSG, P_TRUE, KINDS = 0.6, 0.75, ("wind", "gps_outage", "drain")
DIRS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")


def _when(rng, dt, t_now):
    r = rng.integers(3)
    if r == 0:
        return f"in about {int(round(dt / 5) * 5)} s"
    if r == 1:
        return f"within {max(1, int(round(dt / 60 * 2)) / 2):g} min"
    return f"from t+{int(round(dt))} s"


def sample(rng, t_now):
    """None, or {'kind', 'text', 'true', 't_event', ...}. The world effect is applied by apply()."""
    if rng.random() >= P_MSG:
        return None
    kind = str(rng.choice(KINDS))
    dt = float(rng.uniform(10, 45))
    true = bool(rng.random() < P_TRUE)
    w = _when(rng, dt, t_now)
    m = {"kind": kind, "true": true, "t_msg": round(t_now, 1), "t_event": round(t_now + dt, 1)}
    if kind == "wind":
        v = float(rng.uniform(7, 13))
        k = int(rng.integers(8))
        a = math.pi / 2 - k * math.pi / 4  # wind FROM direction k
        m.update(
            speed=round(v, 1),
            from_dir=DIRS[k],
            vec=[-v * math.cos(a), -v * math.sin(a)],
            gust=round(float(rng.uniform(1.5, 3.0)), 1),
        )
        sp = f"{v:.0f} m/s" if rng.random() < 0.5 else f"{v * 3.6:.0f} km/h"
        src = rng.choice(["Weather service", "Operator", "Met office nowcast", "Ground station"])
        m["text"] = str(
            rng.choice(
                [
                    f"{src}: a wind front is coming {w}; wind from the {DIRS[k]} rising to {sp}, gusty.",
                    f"{src}: expect strong {DIRS[k]}erly wind of about {sp} {w}.",
                    f"{src}: squall line approaching, {sp} winds from the {DIRS[k]} {w}.",
                ]
            )
        )
    elif kind == "gps_outage":
        src = rng.choice(["Air traffic advisory", "Operator", "GNSS interference notice", "Ground station"])
        m["text"] = str(
            rng.choice(
                [
                    f"{src}: GPS jamming reported in this area {w}; expect no satellite fix.",
                    f"{src}: GNSS will be unavailable here {w} (interference test).",
                    f"{src}: satellite navigation outage expected {w} for the rest of the day.",
                ]
            )
        )
    else:
        m["mult"] = round(float(rng.uniform(1.3, 1.7)), 2)
        src = rng.choice(["Weather service", "Operator", "Ground station"])
        m["text"] = str(
            rng.choice(
                [
                    f"{src}: temperature dropping sharply {w}; batteries will lose capacity fast.",
                    f"{src}: heavy cold rain {w}; expect much higher power use.",
                    f"{src}: freezing drizzle {w}, battery drain will rise considerably.",
                ]
            )
        )
    return m


def apply(env, m):
    """Make a TRUE message come true in this world (call once, before any fork, so every copy inherits it)."""
    if m is None or not m["true"]:
        return
    if m["kind"] == "gps_outage":
        env.gps_faults.append((m["t_event"], 1e9, "off"))
    else:
        env.world_events.append(dict(m))


def seen(m, t):
    """The message as the drone has it at time t (None before it arrived)."""
    if m is None or t < m["t_msg"]:
        return None
    s = {"text": m["text"], "age_s": round(t - m["t_msg"], 1)}
    if "text_novel" in m:
        s["text_novel"] = m["text_novel"]
    return s


# ---- Stage D3 (H13-H16)
# D2 lesson: its events were too mild to change which plan is best
# same mechanism but samples the physical size of the change over a
# drone can handle (rated wind ~10-12 m/s
# not, the reader must take the number, not the keyword
# 'text' in the familiar style
# never used in training
# depend on the text, so a test decision can be scored with either
P_MSG3, P_TRUE3, KINDS3, KIND_P3 = 0.8, 0.75, ("wind", "cold", "gps_outage"), (0.4, 0.35, 0.25)
BEAUFORT = (0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5)  # lower bounds of force 1..10 in m/s (WMO)
CAP_T, CAP_C = (-25, -20, -10, 0, 10, 20), (
    0.55,
    0.62,
    0.75,
    0.86,
    0.95,
    1.0,
)  # Li-ion usable capacity vs cell temperature (typical datasheet curves)


def capacity(temp_c):
    return float(np.interp(temp_c, CAP_T, CAP_C))


def _when3(rng, dt, novel):
    r = rng.integers(3)
    if not novel:
        return _when(rng, dt, 0.0)
    if r == 0:
        return f"in roughly {int(round(dt))} seconds"
    if r == 1:
        return (
            "any moment now"
            if dt < 10
            else (
                "in half a minute"
                if 25 <= dt <= 35
                else (
                    "in under a minute"
                    if dt > 35
                    else f"in {int(dt // 5 * 5)}-{int(dt // 5 * 5) + 5} seconds"
                )
            )
        )
    return f"at T+{int(round(dt))}s"


def _text3(rng, m, novel):
    k, dt = m["kind"], m["t_event"] - m["t_msg"]
    w = _when3(rng, dt, novel)
    if k == "wind":
        v, d = m["speed"], m["from_dir"]
        if not novel:
            sp = f"{v:.0f} m/s" if rng.random() < 0.5 else f"{v * 3.6:.0f} km/h"
            src = rng.choice(["Weather service", "Operator", "Met office nowcast", "Ground station"])
            return str(
                rng.choice(
                    [
                        f"{src}: a wind front is coming {w}; wind from the {d} rising to {sp}, gusty.",
                        f"{src}: expect strong {d}erly wind of about {sp} {w}.",
                        f"{src}: squall line approaching, {sp} winds from the {d} {w}.",
                    ]
                )
            )
        bf = sum(v >= b for b in BEAUFORT)
        src = rng.choice(
            ["Pilot report", "Farmer at the field", "Local radio", "Storm spotter", "Harbour master"]
        )
        return str(
            rng.choice(
                [
                    f"{src}: gale warning — force {bf} out of the {d} {w}.",
                    f"{src}: {v * 1.944:.0f} kt from the {d}, gusting higher, expected {w}.",
                    f"{src}: a gust front off the storm will hit you {w}, {v * 2.237:.0f} mph from the {d}.",
                    f"{src}: Beaufort {bf} from the {d} {w}, trees bending.",
                ]
            )
        )
    if k == "cold":
        T = m["temp_c"]
        if not novel:
            src = rng.choice(["Weather service", "Operator", "Ground station"])
            return str(
                rng.choice(
                    [
                        f"{src}: temperature dropping sharply {w}, down to {T:.0f} °C.",
                        f"{src}: cold air arriving {w}; expect {T:.0f} °C.",
                        f"{src}: air temperature will fall to {T:.0f} C {w}.",
                    ]
                )
            )
        src = rng.choice(["Local radio", "Farmer at the field", "Pilot report", "Road service"])
        F = T * 9 / 5 + 32
        words = f"minus {abs(T):.0f} degrees Celsius" if T < 0 else f"{T:.0f} degrees Celsius"
        what = "an arctic blast" if T < -12 else "a hard frost" if T < -2 else "a cool change"
        return str(
            rng.choice(
                [
                    f"{src}: {what} {w}, the thermometer will read {F:.0f} °F.",
                    (
                        f"{src}: {words} out here {w}, everything is freezing."
                        if T < 0
                        else f"{src}: it will be {words} {w}."
                    ),
                    f"{src}: {what} {w}: {F:.0f} F expected.",
                ]
            )
        )
    if not novel:
        src = rng.choice(["Air traffic advisory", "Operator", "GNSS interference notice", "Ground station"])
        return str(
            rng.choice(
                [
                    f"{src}: GPS jamming reported in this area {w}; expect no satellite fix.",
                    f"{src}: GNSS will be unavailable here {w} (interference test).",
                    f"{src}: satellite navigation outage expected {w} for the rest of the day.",
                ]
            )
        )
    src = rng.choice(["Space weather bulletin", "NAVWARN", "Pilot report", "Local radio"])
    return str(
        rng.choice(
            [
                f"{src}: severe geomagnetic storm (Kp 8) {w}; satellite positioning will drop out.",
                f"{src}: military interference trials {w} — no satellite signal in this sector.",
                f"{src}: our receivers lost all satellites a few km away; it reaches you {w}.",
            ]
        )
    )


def sample3(rng, t_now):
    """Stage D3 message: None, or the event."""
    if rng.random() >= P_MSG3:
        return None
    kind = str(rng.choice(KINDS3, p=KIND_P3))
    dt = float(rng.uniform(5, 40))
    true = bool(rng.random() < P_TRUE3)
    m = {"kind": kind, "true": true, "t_msg": round(t_now, 1), "t_event": round(t_now + dt, 1)}
    if kind == "wind":
        v = float(rng.uniform(4, 18))
        k = int(rng.integers(8))
        a = math.pi / 2 - k * math.pi / 4
        m.update(
            speed=round(v, 1),
            from_dir=DIRS[k],
            vec=[-v * math.cos(a), -v * math.sin(a)],
            gust=round(float(rng.uniform(0.15, 0.3)) * v, 1),
        )
    elif kind == "cold":
        T = float(rng.uniform(-25, 10))
        m.update(kind="drain", temp_c=round(T, 1), mult=round(1 / capacity(T), 3), ramp_s=20.0)
    m["parse"] = {
        "kind": kind,
        "eta_s": round(dt, 1),
        "wind_mps": m.get("speed", 0.0),
        "temp_c": m.get("temp_c", 15.0),
    }
    seed = int(rng.integers(2**31))  # the wordings use their own streams
    m["text"] = _text3(np.random.default_rng(seed), m | {"kind": kind}, False)
    m["text_novel"] = _text3(np.random.default_rng(seed + 1), m | {"kind": kind}, True)
    return m
