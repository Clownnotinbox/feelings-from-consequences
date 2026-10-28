"""Fast lidar for big real places."""

import numpy as np
import mujoco
import open3d as o3d

DYN_GROUP = 2


def _icosphere(n_sub=1):
    t = (1 + 5**0.5) / 2
    V = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    F = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(n_sub):
        cache, F2 = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[k] = len(V) - 1
            return cache[k]

        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    return np.array(V), np.array(F)


SPH_V, SPH_F = _icosphere(2)  # 320 faces: <1.5 % radial error (grazing rays on tree crowns)


def _cyl(n=16):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    V = np.concatenate(
        [np.stack([c, s, -np.ones(n)], 1), np.stack([c, s, np.ones(n)], 1), [[0, 0, -1], [0, 0, 1]]]
    )
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [(i, j, n + j), (i, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
    return V, np.array(F)


CYL_V, CYL_F = _cyl(24)
BOX_V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
BOX_F = np.array(
    [
        (0, 1, 3),
        (0, 3, 2),
        (4, 6, 7),
        (4, 7, 5),
        (0, 4, 5),
        (0, 5, 1),
        (2, 3, 7),
        (2, 7, 6),
        (0, 2, 6),
        (0, 6, 4),
        (1, 5, 7),
        (1, 7, 3),
    ]
)


def geom_tris(m, d, g):
    """World-frame triangles of geom g (V, F) or None."""
    t, size = m.geom_type[g], m.geom_size[g]
    pos, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
    if t == mujoco.mjtGeom.mjGEOM_PLANE:
        sx, sy = size[0] or 1000.0, size[1] or 1000.0
        V = np.array([[-sx, -sy, 0], [sx, -sy, 0], [sx, sy, 0], [-sx, sy, 0]])
        F = np.array([(0, 1, 2), (0, 2, 3)])
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        V, F = BOX_V * size, BOX_F
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        V, F = SPH_V * size[0], SPH_F
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        V, F = SPH_V * size, SPH_F
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        V, F = CYL_V * np.array([size[0], size[0], size[1]]), CYL_F
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, h = size[0], size[1]
        Vc = CYL_V * np.array([r, r, h])
        n = len(Vc)
        V = np.concatenate([Vc, SPH_V * r + [0, 0, h], SPH_V * r - [0, 0, h]])
        F = np.concatenate([CYL_F, SPH_F + n, SPH_F + n + len(SPH_V)])
    elif t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        va, vn, fa, fn = m.mesh_vertadr[mid], m.mesh_vertnum[mid], m.mesh_faceadr[mid], m.mesh_facenum[mid]
        V = m.mesh_vert[va : va + vn].astype(float)
        F = m.mesh_face[fa : fa + fn]
    else:
        return None
    return V @ R.T + pos, F


class StaticScene:
    def __init__(self, m, d, drone_body):
        """Mark moving bodies' geoms as group 2; tessellate everything else that can collide (+ the ground)."""
        dyn = set()
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if m.body_mocapid[b] >= 0 or b == drone_body or m.body_jntnum[b] > 0:
                m.geom_group[g] = DYN_GROUP
                dyn.add(g)
        Vs, Fs, owner, off = [], [], [], 0
        for g in range(m.ngeom):
            if g in dyn or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue  # visual-only (roads, water) skipped
            r = geom_tris(m, d, g)
            if r is None:
                continue
            V, F = r
            Vs.append(V)
            Fs.append(F + off)
            owner += [g] * len(F)
            off += len(V)
        V, F = np.concatenate(Vs).astype(np.float32), np.concatenate(Fs).astype(np.uint32)
        self.owner = np.array(owner)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(V), o3d.core.Tensor(F))
        self.n_tris = len(F)
        self.dyn = None

    def cast(self, m, d, origin, dirs, cutoff, drone_body):
        """Distances (-1 = no hit within cutoff) and geom ids for rays from one origin."""
        n = len(dirs)
        rays = o3d.core.Tensor(
            np.hstack([np.repeat(origin[None].astype(np.float32), n, 0), dirs.astype(np.float32)])
        )
        ans = self.scene.cast_rays(rays)
        ds = ans["t_hit"].numpy()
        prim = ans["primitive_ids"].numpy()
        gs = np.where(np.isfinite(ds), self.owner[np.minimum(prim, len(self.owner) - 1)], -1)
        if self.dyn is None:
            self.dyn = [
                g for g in range(m.ngeom) if m.geom_group[g] == DYN_GROUP and m.geom_bodyid[g] != drone_body
            ]
        near = [
            g for g in self.dyn if np.linalg.norm(d.geom_xpos[g] - origin) < cutoff + 5 + m.geom_rbound[g]
        ]
        dd = np.full(n, np.inf, np.float32)
        gd = np.full(n, -1)
        if near:
            Vs, Fs, own, off = [], [], [], 0
            for g in near:
                r = geom_tris(m, d, g)
                if r is None:
                    continue
                V, F = r
                Vs.append(V)
                Fs.append(F + off)
                own += [g] * len(F)
                off += len(V)
            if Vs:
                sc = o3d.t.geometry.RaycastingScene()
                sc.add_triangles(
                    o3d.core.Tensor(np.concatenate(Vs).astype(np.float32)),
                    o3d.core.Tensor(np.concatenate(Fs).astype(np.uint32)),
                )
                a2 = sc.cast_rays(rays)
                dd = a2["t_hit"].numpy()
                own = np.array(own)
                gd = np.where(np.isfinite(dd), own[np.minimum(a2["primitive_ids"].numpy(), len(own) - 1)], -1)
        use_dyn = dd < ds
        dist = np.where(use_dyn, dd, ds)
        geo = np.where(use_dyn, gd, gs)
        ok = np.isfinite(dist) & (dist < cutoff)
        return np.where(ok, dist, -1.0).astype(np.float64), np.where(ok, geo, -1).astype(np.int32)
