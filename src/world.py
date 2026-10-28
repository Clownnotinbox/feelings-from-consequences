"""Procedural 3D world + quadrotor."""

import math, random
from dataclasses import dataclass, field

ARM = 0.25  # rotor distance from centre, m
MASS = 1.5  # kg (body + arms + rotors below)
MOTOR_MAX = 10.0  # N per motor (thrust/weight ~ 2.7)
YAW_K = 0.016  # rotor drag torque per newton of thrust, m
ROTORS = [
    (+1, +1, -1),
    (-1, -1, -1),
    (+1, -1, +1),
    (-1, +1, +1),
]  # (sx, sy, spin): X frame, spin sign for yaw

COLORS = {
    "red": "0.85 0.12 0.10",
    "yellow": "0.95 0.80 0.10",
    "blue": "0.15 0.30 0.85",
    "orange": "0.95 0.50 0.10",
    "purple": "0.55 0.20 0.70",
    "white": "0.92 0.92 0.92",
}
SHAPES = (
    "tent",
    "pad",
    "tower",
    "container",
    "car",
)  # civil goals: camp, landing pad, water tower, delivery container


@dataclass
class Target:
    name: str
    shape: str
    color: str
    pos: tuple
    description: str = ""


@dataclass
class World:
    seed: int
    size: float
    buildings: list = field(default_factory=list)  # (x, y, sx, sy, h, rgba)
    trees: list = field(default_factory=list)  # (x, y, h)
    cars: list = field(default_factory=list)  # (x, y, yaw, rgba) parked cars = obstacles to avoid
    movers: list = field(default_factory=list)  # moving distractors
    targets: list = field(default_factory=list)
    wires: list = field(
        default_factory=list
    )  # (x1, y1, x2, y2, h): power lines between two poles; 2 cm thick
    start: tuple = (0.0, 0.0)
    sun: tuple = (0.3, 0.2, -1.0, 0.7)  # light direction + intensity (random per seed)


N_PROJECTILES = 4  # pool of thrown objects
ENC_SIZES = (
    0.05,
    0.1,
    0.2,
    0.35,
    0.5,
    0.8,
    1.2,
    1.5,
) * 2  # pool for the encounter generator: two bodies per size


def enc_xml(i, size):
    """Encounter body: small = stone/ball, medium = bird-like, large = box (another drone, kite, debris)."""
    r = size / 2
    if size < 0.25:
        g = f'<geom name="enc{i}_g" type="sphere" size="{r:.3f}" rgba=".5 .45 .4 1"/>'
    elif size < 0.9:
        g = (
            f'<geom name="enc{i}_g" type="ellipsoid" size="{r:.3f} {r * .4:.3f} {r * .35:.3f}" rgba=".2 .17 .15 1"/>'
            f'<geom name="enc{i}_w" type="box" size="{r * .3:.3f} {r * 1.1:.3f} .01" rgba=".2 .17 .15 1"/>'
        )
    else:
        g = f'<geom name="enc{i}_g" type="box" size="{r:.3f} {r:.3f} {r * .25:.3f}" rgba=".25 .25 .3 1"/>'
    return f'<body name="enc{i}" mocap="true" pos="0 0 -30">{g}</body>'


def generate(
    seed=0,
    size=120.0,
    n_buildings=14,
    n_trees=40,
    n_targets=4,
    n_cars=8,
    n_moving_cars=4,
    n_people=6,
    n_birds=5,
    n_wires=3,
    n_divers=2,
):
    rng = random.Random(seed)
    W = World(seed, size)
    half = size / 2 - 8
    occupied = [(0.0, 0.0, 8.0)]  # start pad keep-out

    def free(x, y, r):
        return all(math.hypot(x - a, y - b) > r + c for a, b, c in occupied)

    for _ in range(n_buildings * 20):
        if len(W.buildings) >= n_buildings:
            break
        x, y = rng.uniform(-half, half), rng.uniform(-half, half)
        sx, sy = rng.uniform(4, 10), rng.uniform(4, 10)
        h = rng.choice([rng.uniform(5, 12), rng.uniform(15, 35)])
        r = math.hypot(sx, sy) / 2
        if free(x, y, r + 3):
            g = rng.uniform(0.45, 0.75)
            W.buildings.append((x, y, sx, sy, h, f"{g:.2f} {g:.2f} {g + .03:.2f} 1"))
            occupied.append((x, y, r))
    for _ in range(n_trees * 20):
        if len(W.trees) >= n_trees:
            break
        x, y = rng.uniform(-half, half), rng.uniform(-half, half)
        if free(x, y, 2.5):
            W.trees.append((x, y, rng.uniform(4, 9)))
            occupied.append((x, y, 1.5))
    for _ in range(n_cars * 20):  # parked cars scattered on the ground
        if len(W.cars) >= n_cars:
            break
        x, y = rng.uniform(-half, half), rng.uniform(-half, half)
        if free(x, y, 3):
            g = rng.uniform(0.3, 0.8)
            W.cars.append((x, y, rng.uniform(0, math.pi), f"{g:.2f} {g * .6:.2f} {g * .4:.2f} 1"))
            occupied.append((x, y, 2))
    combos = [(s, c) for s in SHAPES for c in COLORS]
    rng.shuffle(combos)
    for k in range(n_targets * 50):
        if len(W.targets) >= n_targets:
            break
        x, y = rng.uniform(-half, half), rng.uniform(-half, half)
        if math.hypot(x, y) < 30 or not free(x, y, 4):
            continue
        shape, color = combos[len(W.targets)]
        t = Target(f"target{len(W.targets)}", shape, color, (x, y))
        t.description = describe(W, t)
        W.targets.append(t)
        occupied.append((x, y, 3))

    def lane(min_len, max_len):  # a straight segment through free space, away from the start pad
        for _ in range(200):
            x, y = rng.uniform(-half, half), rng.uniform(-half, half)
            a = rng.uniform(0, math.pi)
            L = rng.uniform(min_len, max_len)
            x2, y2 = x + L * math.cos(a), y + L * math.sin(a)
            pts = [(x + (x2 - x) * k / 10, y + (y2 - y) * k / 10) for k in range(11)]
            if all(
                abs(px) < half and abs(py) < half and free(px, py, 2) and math.hypot(px, py) > 10
                for px, py in pts
            ):
                return (x, y, x2, y2)
        return None

    for i in range(n_moving_cars):
        seg = lane(25, 60)
        if seg:
            g = rng.uniform(0.2, 0.9)
            W.movers.append(
                {
                    "kind": "car",
                    "seg": seg,
                    "speed": rng.uniform(4, 9),
                    "phase": rng.random(),
                    "rgba": f"{g:.2f} {rng.uniform(.1, .9):.2f} {1 - g:.2f} 1",
                }
            )
    for i in range(n_people):
        seg = lane(8, 25)
        if seg:
            W.movers.append(
                {
                    "kind": "person",
                    "seg": seg,
                    "speed": rng.uniform(1.0, 1.6),
                    "phase": rng.random(),
                    "rgba": f"{rng.uniform(.1, .9):.2f} {rng.uniform(.1, .9):.2f} {rng.uniform(.1, .9):.2f} 1",
                }
            )
    for i in range(n_birds):  # the first n_divers birds sometimes dive at the drone
        W.movers.append(
            {
                "kind": "bird",
                "center": (rng.uniform(-half, half), rng.uniform(-half, half)),
                "radius": rng.uniform(8, 25),
                "alt": rng.uniform(6, 28),
                "speed": rng.uniform(6, 11),
                "phase": rng.random(),
                "rgba": ".15 .13 .12 1",
                "diver": i < n_divers,
            }
        )
    for i in range(n_wires):  # power lines: two poles and a thin cable that a sparse lidar can miss
        seg = lane(20, 45)
        if seg:
            W.wires.append((*seg, rng.uniform(8, 16)))
    az, el = rng.uniform(0, 2 * math.pi), rng.uniform(0.35, 1.2)
    W.sun = (math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), -math.sin(el), rng.uniform(0.45, 0.85))
    return W


def tree_xml(i, x, y, h, rng):
    """Trunk + 3-5 branches + a crown of 3 overlapping leaf masses."""
    trunk = f'<geom name="trunk{i}" type="capsule" fromto="{x:.2f} {y:.2f} 0 {x:.2f} {y:.2f} {h * .75:.2f}" size="{.12 + .02 * h:.2f}" rgba=".33 .22 .12 1"/>'
    br = ""
    for k in range(rng.randint(3, 5)):
        a = rng.uniform(0, 2 * math.pi)
        z0 = h * rng.uniform(0.35, 0.65)
        L = rng.uniform(1.0, 2.2)
        br += (
            f'<geom name="br{i}_{k}" type="capsule" fromto="{x:.2f} {y:.2f} {z0:.2f} {x + L * math.cos(a):.2f} {y + L * math.sin(a):.2f} '
            f'{z0 + L * .6:.2f}" size=".06" rgba=".33 .22 .12 1"/>'
        )
    crown = ""
    for k in range(3):
        dx, dy, dz = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(-0.3, 0.6)
        r = rng.uniform(1.1, 1.7)
        g = rng.uniform(0.38, 0.52)
        crown += f'<geom name="crown{i}_{k}" type="ellipsoid" pos="{x + dx:.2f} {y + dy:.2f} {h * .78 + dz:.2f}" size="{r:.2f} {r:.2f} {r * .8:.2f}" rgba=".12 {g:.2f} .15 1"/>'
    return trunk + br + crown


def mover_pose(mv, t):
    """Position (x, y, z) and yaw of a moving distractor at time t."""
    if mv["kind"] == "bird":
        r = mv["radius"]
        ang = mv["phase"] * 2 * math.pi + mv["speed"] / r * t
        cx, cy = mv["center"]
        return (
            cx + r * math.cos(ang),
            cy + r * math.sin(ang),
            mv["alt"] + 0.8 * math.sin(1.7 * t + 6 * mv["phase"]),
            ang + math.pi / 2,
        )
    x1, y1, x2, y2 = mv["seg"]
    L = math.hypot(x2 - x1, y2 - y1)
    s = (mv["phase"] * 2 * L + mv["speed"] * t) % (2 * L)
    fwd = s < L
    u = s / L if fwd else 2 - s / L
    yaw = math.atan2(y2 - y1, x2 - x1) + (0 if fwd else math.pi)
    return x1 + (x2 - x1) * u, y1 + (y2 - y1) * u, 0.0, yaw


def mover_xml(i, mv):
    c = mv["rgba"]
    x, y, z, yaw = mover_pose(mv, 0.0)
    if mv["kind"] == "car":
        inner = _car_xml(f"mcar{i}", 0.0, 0.0, 0.0, c)
    elif mv["kind"] == "person":
        inner = (
            f'<geom name="mper{i}_b" type="capsule" fromto="0 0 .35 0 0 1.35" size=".2" rgba="{c}"/>'
            f'<geom name="mper{i}_h" type="sphere" pos="0 0 1.6" size=".14" rgba=".85 .7 .6 1"/>'
        )
    else:
        inner = (
            f'<geom name="mbird{i}_b" type="ellipsoid" size=".18 .07 .06" rgba="{c}"/>'
            f'<geom name="mbird{i}_w" type="box" size=".05 .35 .01" rgba="{c}"/>'
        )
    return f'<body name="mover{i}" mocap="true" pos="{x:.2f} {y:.2f} {z:.2f}">{inner}</body>'


WORD = {"tent": "tent", "pad": "landing pad", "tower": "water tower", "container": "container", "car": "car"}


def describe(W, t):
    """A plain description from geometry only: colour + shape + the nearest building's height class + side."""
    x, y = t.pos
    desc = f"the {t.color} {WORD[t.shape]}"
    if W.buildings:
        b = min(W.buildings, key=lambda b: math.hypot(b[0] - x, b[1] - y))
        d = math.hypot(b[0] - x, b[1] - y)
        if d < 25:
            side = (
                ("east" if x > b[0] else "west")
                if abs(x - b[0]) > abs(y - b[1])
                else ("north" if y > b[1] else "south")
            )
            desc += f" {side} of a {'tall' if b[4] > 14 else 'low'} building"
    return desc


def target_geoms(t):
    x, y = t.pos
    c = COLORS[t.color] + " 1"
    n = t.name
    if t.shape == "tent":  # two slanted roof boxes = a ridge tent
        return (
            f'<geom name="{n}_a" type="box" pos="{x} {y - .9} 1.1" size="2.0 1.25 .05" euler="45 0 0" rgba="{c}"/>'
            f'<geom name="{n}_b" type="box" pos="{x} {y + .9} 1.1" size="2.0 1.25 .05" euler="-45 0 0" rgba="{c}"/>'
        )
    if t.shape == "pad":  # landing pad: coloured disc with a white H
        return (
            f'<geom name="{n}_a" type="cylinder" pos="{x} {y} .03" size="2.5 .03" rgba="{c}"/>'
            f'<geom name="{n}_b" type="box" pos="{x} {y} .07" size=".9 .12 .01" rgba="1 1 1 1"/>'
            f'<geom name="{n}_c" type="box" pos="{x - .8} {y} .07" size=".12 1.0 .01" rgba="1 1 1 1"/>'
            f'<geom name="{n}_d" type="box" pos="{x + .8} {y} .07" size=".12 1.0 .01" rgba="1 1 1 1"/>'
        )
    if t.shape == "tower":
        return (
            f'<geom name="{n}_a" type="cylinder" pos="{x} {y} 3" size=".35 3" rgba=".4 .4 .4 1"/>'
            f'<geom name="{n}_b" type="sphere" pos="{x} {y} 6.8" size="1.6" rgba="{c}"/>'
        )
    if t.shape == "car":  # parked car: a destination landmark / obstacle, never a strike goal
        return _car_xml(n, x, y, 0.0, t.color)
    return f'<geom name="{n}_a" type="box" pos="{x} {y} 1.3" size="3.0 1.2 1.3" rgba="{c}"/>'  # container


def _car_xml(n, x, y, yaw, color):
    c = (COLORS[color] + " 1") if color in COLORS else color
    yd = math.degrees(yaw)
    cs, sn = math.cos(yaw), math.sin(yaw)
    wheels = ""
    for wx, wy in ((1.3, 1.0), (1.3, -1.0), (-1.3, 1.0), (-1.3, -1.0)):
        gx, gy = x + wx * cs - wy * sn, y + wx * sn + wy * cs
        wheels += f'<geom name="{n}_w{wx:.0f}{wy:.0f}" type="cylinder" pos="{gx:.2f} {gy:.2f} .35" size=".35 .18" euler="90 0 {yd:.1f}" rgba=".08 .08 .08 1"/>'
    return (
        f'<geom name="{n}_a" type="box" pos="{x:.2f} {y:.2f} .6" size="2.1 .9 .35" euler="0 0 {yd:.1f}" rgba="{c}"/>'
        f'<geom name="{n}_b" type="box" pos="{x - .2 * cs:.2f} {y - .2 * sn:.2f} 1.15" size="1.1 .8 .3" euler="0 0 {yd:.1f}" rgba="{c}"/>'
        f'<geom name="{n}_g" type="box" pos="{x - .2 * cs:.2f} {y - .2 * sn:.2f} 1.16" size="1.0 .72 .28" euler="0 0 {yd:.1f}" rgba=".15 .2 .28 .8"/>'
        + wheels
    )


def drone_xml(z0=0.3):
    arms = ""
    motors = ""
    L = ARM / math.sqrt(2)
    for i, (sx, sy, spin) in enumerate(ROTORS):
        x, y = sx * L, sy * L
        arms += (
            f'<geom type="capsule" fromto="0 0 0 {x:.4f} {y:.4f} 0" size=".012" mass=".05" rgba=".15 .15 .15 1"/>'
            f'<geom type="cylinder" pos="{x:.4f} {y:.4f} .03" size=".11 .006" mass=".04" rgba=".2 .2 .2 .5" contype="0" conaffinity="0"/>'
            f'<site name="rotor{i}" pos="{x:.4f} {y:.4f} .03"/>'
        )
        motors += f'<motor name="m{i}" site="rotor{i}" gear="0 0 1 0 0 {spin * YAW_K}" ctrllimited="true" ctrlrange="0 {MOTOR_MAX}"/>'
    body_mass = MASS - 4 * 0.09
    return (
        f"""
    <body name="drone" pos="0 0 {z0}">
      <freejoint name="drone_free"/>
      <geom name="drone_body" type="box" size=".09 .06 .035" mass="{body_mass:.3f}" rgba=".1 .1 .12 1"/>
      {arms}
      <site name="imu" pos="0 0 0"/>
      <site name="rf" pos="0 0 -.04" zaxis="0 0 -1"/>
      <camera name="front" pos=".32 0 -.02" xyaxes="0 -1 0 .26 0 .97" fovy="80"/>
      <camera name="down" pos="0 0 -.04" xyaxes="0 -1 0 1 0 0" fovy="90"/>
      <camera name="chase" pos="-3 0 1.2" xyaxes="0 -1 0 .3 0 .95" fovy="60" mode="track"/>
    </body>""",
        motors,
    )


def to_mjcf(W, z0=0.3, building_xml=None, extra_assets="", shift=(0.0, 0.0)):
    """building_xml: geometry of a real place."""
    half = W.size / 2
    body, motors = drone_xml(z0)
    geoms = []
    if building_xml is not None:
        geoms.append(f'<body name="place" pos="{shift[0]:.2f} {shift[1]:.2f} 0">{building_xml}</body>')
    for i, (x, y, sx, sy, h, rgba) in enumerate(W.buildings if building_xml is None else []):
        geoms.append(
            f'<geom name="bld{i}" type="box" pos="{x:.2f} {y:.2f} {h / 2:.2f}" size="{sx / 2:.2f} {sy / 2:.2f} {h / 2:.2f}" rgba="{rgba}" material="facade"/>'
        )
    for i, (x, y, h) in enumerate(W.trees if building_xml is None else []):
        geoms.append(tree_xml(i, x, y, h, random.Random(W.seed * 1000 + i)))
    for i, (x1, y1, x2, y2, h) in enumerate(W.wires if building_xml is None else []):
        geoms.append(
            f'<geom name="pole{i}a" type="cylinder" pos="{x1:.2f} {y1:.2f} {h / 2:.2f}" size=".15 {h / 2:.2f}" rgba=".45 .4 .35 1"/>'
            f'<geom name="pole{i}b" type="cylinder" pos="{x2:.2f} {y2:.2f} {h / 2:.2f}" size=".15 {h / 2:.2f}" rgba=".45 .4 .35 1"/>'
            f'<geom name="wire{i}" type="capsule" fromto="{x1:.2f} {y1:.2f} {h - .3:.2f} {x2:.2f} {y2:.2f} {h - .3:.2f}" size=".01" rgba=".08 .08 .08 1"/>'
        )
    for i, (x, y, yaw, rgba) in enumerate(W.cars):
        geoms.append(_car_xml(f"pcar{i}", x, y, yaw, rgba))
    for t in W.targets:
        geoms.append(target_geoms(t))
    movers = "".join(mover_xml(i, mv) for i, mv in enumerate(W.movers))
    sx, sy, sz, si = W.sun
    movers += "".join(
        f'<body name="proj{i}" mocap="true" pos="0 0 -20"><geom name="proj{i}_g" type="sphere" size=".11" rgba=".55 .5 .45 1"/></body>'
        for i in range(N_PROJECTILES)
    )
    movers += "".join(enc_xml(i, sz) for i, sz in enumerate(ENC_SIZES))
    return f"""<mujoco model="drone_world_{W.seed}">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4" density="0"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/><headlight ambient=".40 .40 .42" diffuse=".35 .35 .35"/><map znear=".02" zfar="400" fogstart="1" fogend="3"/><rgba fog=".78 .82 .86 1" haze=".8 .85 .9 1"/></visual>
  <asset>
    <texture name="grass" type="2d" builtin="checker" rgb1=".30 .48 .22" rgb2=".27 .44 .20" width="512" height="512" mark="random" markrgb=".35 .52 .25" random=".06"/>
    <material name="ground" texture="grass" texrepeat="240 240"/>
    <texture name="win" type="2d" builtin="checker" rgb1=".62 .66 .72" rgb2=".48 .52 .60" width="128" height="128"/>
    <material name="facade" texture="win" texrepeat="4 10" texuniform="false" reflectance=".15" shininess=".4" specular=".5"/>
    {extra_assets}
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".55 .72 .92" rgb2=".95 .97 1" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="0 0 80" dir="{sx:.3f} {sy:.3f} {sz:.3f}" directional="true" diffuse="{si:.2f} {si:.2f} {si * .93:.2f}" castshadow="true"/>
    <geom name="ground" type="plane" size="{half * 4} {half * 4} .1" material="ground"/>
    <geom name="pad" type="cylinder" pos="0 0 .01" size="1.5 .01" rgba=".95 .95 .95 1"/>
    {''.join(geoms)}
    {movers}
    {body}
  </worldbody>
  <actuator>{motors}</actuator>
  <sensor>
    <accelerometer name="acc" site="imu"/><gyro name="gyro" site="imu"/>
    <rangefinder name="range_down" site="rf" cutoff="40"/>
  </sensor>
</mujoco>"""
