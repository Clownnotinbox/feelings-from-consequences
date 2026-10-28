"""Fast lidar must see the same world as MuJoCo's own ray caster."""

import collections, time, numpy as np, mujoco, drone_env

for place in (None, "spb_center", "akadem_line", "tokyo_res", "malakhovka"):
    env = drone_env.DroneEnv(seed=4, place=place, render=False, start="random", encounters={"rate": 1.0})
    o = env.reset()
    for _ in range(120):
        env.step({"vz": 1.2 if env.t < 8 else 0.0})
    p, v, R, w = env._state()
    r = np.random.default_rng(0)
    dirs = r.normal(size=(40000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    t = time.time()
    for _ in range(5):
        df, gf = env.scene.cast(env.model, env.data, p, dirs, 40.0, env.bid)
    tf = (time.time() - t) / 5
    grp = env.model.geom_group.copy()
    env.model.geom_group[:] = 0
    gm = np.zeros(len(dirs), np.int32)
    dm = np.zeros(len(dirs))
    t = time.time()
    mujoco.mj_multiRay(
        env.model, env.data, p, dirs.reshape(-1), None, 1, env.bid, gm, dm, None, len(dirs), 40.0
    )
    tm = time.time() - t
    env.model.geom_group[:] = grp
    dm = np.where(dm < 40, dm, -1.0)  # MuJoCo returns hits beyond the cutoff for meshes / planes
    plane = env.model.geom("ground").id
    vis = (env.model.geom_contype == 0) & (env.model.geom_conaffinity == 0)
    ok = (
        (gf != plane) & (gm != plane) & ~((dm >= 0) & vis[np.maximum(gm, 0)])
    )  # ground: MuJoCo ignores the cutoff for planes
    dyn = set(env.scene.dyn)
    agree = float(((df >= 0) == (dm >= 0))[ok].mean())
    both = ok & (df >= 0) & (dm >= 0)
    err = np.abs(df[both] - dm[both])
    nd_mj = int(sum(g in dyn for g in gm[(dm >= 0)]))
    nd_f = int(sum(g in dyn for g in gf[(df >= 0)]))
    print(
        f"{place or 'procedural'}: fast {1000 * tf:.0f} ms vs mujoco {1000 * tm:.0f} ms | object hit agreement {agree:.4f} | err median {np.median(err) * 100:.1f} cm, 99% {np.quantile(err, .99) * 100:.1f} cm | moving-object hits mujoco {nd_mj} fast {nd_f}"
    )
