import math

import numpy as np

from graphics.roads import merge_meshes

NSHADES = 8
TREE_KINDS = {"tree": (5.5, 9.0), "small": (3.0, 4.5), "shrub": (0.8, 1.6)}
TRUNK_BOX_HALF = 0.22
TRUNK_BOX_HEIGHT = 3.0


def _orient(verts, idx):
    tri = verts[idx.reshape(-1, 3)]
    geo = np.cross(tri[:, 1, :3] - tri[:, 0, :3], tri[:, 2, :3] - tri[:, 0, :3])
    ok = (geo * tri[:, 0, 3:6]).sum(axis=1)
    if np.median(ok) < 0:
        idx = idx.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1)
    return verts, idx


def sphere_part(cx, cy, cz, r, sy=1.0, shade=0, seg=9, rings=6):
    verts = []
    u = (shade % NSHADES + 0.5) / NSHADES
    for i in range(rings + 1):
        th = math.pi * i / rings
        y, rr = math.cos(th), math.sin(th)
        for j in range(seg + 1):
            ph = 2 * math.pi * j / seg
            x, z = rr * math.cos(ph), rr * math.sin(ph)
            n = np.array([x, y / max(sy, 1e-6), z])
            n /= np.linalg.norm(n) + 1e-12
            verts.append([cx + r * x, cy + r * sy * y, cz + r * z, n[0], n[1], n[2], u, 0.10 + 0.85 * (0.5 + 0.5 * y)])
    idx = []
    for i in range(rings):
        for j in range(seg):
            a = i * (seg + 1) + j
            b, c, d = a + 1, a + seg + 1, a + seg + 2
            idx += [a, c, b, b, c, d]
    return _orient(np.array(verts, np.float32), np.array(idx, np.uint32))


def trunk_part(cx, cz, h, r0, r1, sides=6):
    verts, idx = [], []
    for j in range(sides):
        a0, a1 = 2 * math.pi * j / sides, 2 * math.pi * (j + 1) / sides
        am = (a0 + a1) / 2
        n = (math.cos(am), 0.0, math.sin(am))
        s = len(verts)
        verts += [[cx + r0 * math.cos(a0), 0.0, cz + r0 * math.sin(a0), *n, 0.5, 0.0],
                  [cx + r0 * math.cos(a1), 0.0, cz + r0 * math.sin(a1), *n, 0.5, 0.0],
                  [cx + r1 * math.cos(a1), h, cz + r1 * math.sin(a1), *n, 0.5, 1.0],
                  [cx + r1 * math.cos(a0), h, cz + r1 * math.sin(a0), *n, 0.5, 1.0]]
        idx += [s, s + 1, s + 2, s, s + 2, s + 3]
    return _orient(np.array(verts, np.float32), np.array(idx, np.uint32))


def make_plant(x, z, kind, rng):
    """Tropical-looking plant: trunk + a loose cluster of foliage clumps (max reach ~1.15 R)."""
    lo, hi = TREE_KINDS[kind]
    H = float(rng.uniform(lo, hi))
    shade = int(rng.integers(0, NSHADES))
    if kind == "shrub":
        R = H * 0.5
        parts = [sphere_part(x, 0.5 * R, z, 0.8 * R, sy=0.75, shade=shade)]
        for _ in range(int(rng.integers(2, 4))):
            a = float(rng.uniform(0, 2 * math.pi))
            parts.append(sphere_part(x + 0.3 * R * math.cos(a), 0.45 * R, z + 0.3 * R * math.sin(a),
                                     0.6 * R, sy=0.8, shade=shade + int(rng.integers(-1, 2))))
        return None, merge_meshes(parts), None

    trunk_h = float(rng.uniform(0.32, 0.44)) * H
    R = 0.32 * H
    trunk = trunk_part(x, z, trunk_h + 0.5 * R, 0.10 + 0.015 * H, 0.06 + 0.008 * H)
    parts = [sphere_part(x, trunk_h + 0.55 * R, z, 0.72 * R, sy=0.78, shade=shade)]
    for _ in range(int(rng.integers(4, 8))):
        a = float(rng.uniform(0, 2 * math.pi))
        d = float(rng.uniform(0.25, 0.55)) * R
        parts.append(sphere_part(x + d * math.cos(a), trunk_h + float(rng.uniform(0.15, 0.95)) * R,
                                 z + d * math.sin(a), float(rng.uniform(0.4, 0.6)) * R, sy=0.8,
                                 shade=shade + int(rng.integers(-2, 3))))
    canopy = merge_meshes(parts)
    box = np.array([x - TRUNK_BOX_HALF, 0.0, z - TRUNK_BOX_HALF, x + TRUNK_BOX_HALF, TRUNK_BOX_HEIGHT, z + TRUNK_BOX_HALF])
    return trunk, canopy, box


def build_plants(placements, seed=1):
    rng = np.random.default_rng(seed)
    trunks, canopies, boxes = [], [], []
    for x, z, kind in placements:
        tr, ca, bx = make_plant(x, z, kind, rng)
        if tr is not None:
            trunks.append(tr)
        canopies.append(ca)
        if bx is not None:
            boxes.append(bx)
    return (merge_meshes(trunks), merge_meshes(canopies),
            np.array(boxes, dtype=np.float32).reshape(-1, 6))
