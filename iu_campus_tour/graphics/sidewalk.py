"""Raised sidewalks with a kerb along a road centerline (pure numpy; no GL)."""

import numpy as np

from graphics.roads import merge_meshes, MITER_LIMIT


def miter_vectors(P):
    """Unit-offset miter vectors for an open polyline: offsetting by ``d`` along them moves ``d`` sideways."""
    n = len(P)
    seg = P[1:] - P[:-1]
    length = np.maximum(np.hypot(seg[:, 0], seg[:, 1]), 1e-9)
    d = seg / length[:, None]
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    out = np.zeros((n, 2))
    for i in range(n):
        n0 = nrm[i - 1] if i > 0 else nrm[0]
        n1 = nrm[i] if i < n - 1 else nrm[-1]
        denom = 1.0 + float(n0 @ n1)
        m = n1 if denom < 1e-6 else (n0 + n1) / denom
        ln = float(np.hypot(*m))
        out[i] = m * (MITER_LIMIT / ln) if ln > MITER_LIMIT else m
    return out, nrm


def _ribbon(a, b, normal, ua, ub, v):
    """Quad strip between 3D polylines ``a`` and ``b``; winding is fixed so faces point along ``normal``."""
    n = len(a)
    verts = np.zeros((2 * n, 8), np.float32)
    verts[0::2, :3], verts[1::2, :3] = a, b
    verts[:, 3:6] = normal if np.ndim(normal) == 1 else np.repeat(normal, 2, axis=0)
    verts[0::2, 6], verts[1::2, 6] = ua, ub
    verts[0::2, 7] = verts[1::2, 7] = v
    idx = []
    for i in range(n - 1):
        la, lb, na, nb = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        idx += [la, lb, na, lb, nb, na]
    idx = np.array(idx, np.uint32).reshape(-1, 3)
    tri = verts[idx][:, :, :3]
    geo = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    want = verts[idx[:, 0], 3:6]
    ok = (geo * want).sum(axis=1)
    if np.median(ok) < 0:
        idx = idx[:, [0, 2, 1]]
    return verts, idx.reshape(-1)


def sidewalk_side(P, side, hw, kerb_w, walk_w, y_road, y_top, tile_walk, tile_kerb):
    """Build one side of a road.

    ``P`` is a run of centerline points (n, 2), ``side`` is +1 (left) or -1 (right),
    ``walk_w`` the sidewalk width per vertex (n,). Returns ``(walk_mesh, kerb_mesh)``;
    the kerb mesh holds the kerb top, the riser facing the road and the outer riser.
    """
    P = np.asarray(P, float)
    walk_w = np.asarray(walk_w, float)
    m, _ = miter_vectors(P)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])

    def line(off, y):
        q = P + side * m * (off if np.ndim(off) == 0 else off[:, None])
        return np.column_stack([q[:, 0], np.full(len(P), y), q[:, 1]])

    A_top, A_low = line(hw, y_top), line(hw, y_road)
    B_top = line(hw + kerb_w, y_top)
    C_off = hw + kerb_w + walk_w
    C_top, C_low = line(C_off, y_top), line(C_off, 0.0)
    up = np.array([0.0, 1.0, 0.0])
    v = cum / tile_kerb
    kerb_top = _ribbon(A_top, B_top, up, hw / tile_kerb, (hw + kerb_w) / tile_kerb, v)
    # risers: normal is the horizontal direction pointing away from the sidewalk body
    toward_road = np.column_stack([-side * m[:, 0], np.zeros(len(P)), -side * m[:, 1]])
    toward_road /= np.maximum(np.linalg.norm(toward_road, axis=1, keepdims=True), 1e-9)
    inner = _ribbon(A_low, A_top, toward_road, 0.0, (y_top - y_road) / tile_kerb, v)
    outer = _ribbon(C_top, C_low, -toward_road, 0.0, y_top / tile_kerb, v)
    walk = _ribbon(B_top, C_top, up, (hw + kerb_w) / tile_walk, C_off / tile_walk, cum / tile_walk)
    return walk, merge_meshes([kerb_top, inner, outer])
