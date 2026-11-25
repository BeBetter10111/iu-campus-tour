import numpy as np

from graphics.extrude import triangulate

MITER_LIMIT = 2.0    


def _dedupe(points, tol=0.05):
    pts = []
    for p in np.asarray(points, dtype=np.float64):
        if not pts or np.hypot(*(p - pts[-1])) >= tol:
            pts.append(p)
    return np.array(pts)


def polyline_strip(points, width, y, tile, full_width=False, closed=False):
    P = _dedupe(points)
    if closed and len(P) > 2 and np.hypot(*(P[0] - P[-1])) < 0.05:
        P = P[:-1]
    n = len(P)
    if n < 2:
        return None

    seg = (np.roll(P, -1, axis=0) - P) if closed else (P[1:] - P[:-1])
    length = np.maximum(np.hypot(seg[:, 0], seg[:, 1]), 1e-9)
    d = seg / length[:, None]
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)

    miter = np.zeros((n, 2))
    for i in range(n):
        if closed:
            n0, n1 = nrm[i - 1], nrm[i]
        else:
            n0 = nrm[i - 1] if i > 0 else nrm[0]
            n1 = nrm[i] if i < n - 1 else nrm[-1]
        denom = 1.0 + float(n0 @ n1)
        m = n1 if denom < 1e-6 else (n0 + n1) / denom
        ln = float(np.hypot(*m))
        miter[i] = m * (MITER_LIMIT / ln) if ln > MITER_LIMIT else m

    cum = np.concatenate([[0.0], np.cumsum(length)])
    count = n + 1 if closed else n
    left = np.vstack([P + miter * width / 2, (P[0] + miter[0] * width / 2)[None]])[:count]
    right = np.vstack([P - miter * width / 2, (P[0] - miter[0] * width / 2)[None]])[:count]

    if full_width:
        u_l, u_r = 0.0, 1.0
    else:
        u_l, u_r = 0.5 + width / 2 / tile, 0.5 - width / 2 / tile
    verts = []
    for i in range(count):
        v = cum[i] / tile
        verts.append([left[i, 0], y, left[i, 1], 0, 1, 0, u_l, v])
        verts.append([right[i, 0], y, right[i, 1], 0, 1, 0, u_r, v])
    verts = np.array(verts, np.float32)

    idx = []
    for i in range(count - 1):
        a, b, c, e = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3        # L_i, R_i, L_{i+1}, R_{i+1}
        idx += [a, b, c, b, e, c]
    idx = np.array(idx, np.uint32).reshape(-1, 3)

    tri = verts[idx][:, :, :3]
    ny = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 1]
    if len(ny) and np.median(ny) < 0:
        idx = idx[:, [0, 2, 1]]
    return verts, idx.reshape(-1)


def polygon_flat(ring, y, tile):
    ring = _dedupe(ring)
    if len(ring) > 2 and np.hypot(*(ring[0] - ring[-1])) < 0.05:
        ring = ring[:-1]
    if len(ring) < 3:
        return None
    tris, _ = triangulate(ring)
    verts = np.array([[x, y, z, 0, 1, 0, x / tile, z / tile] for x, z in ring], np.float32)
    idx = []
    for i0, i1, i2 in tris:
        idx += [i0, i2, i1]                 
    return verts, np.array(idx, np.uint32)


def merge_meshes(parts):
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    verts, idx, offset = [], [], 0
    for v, i in parts:
        verts.append(v)
        idx.append(i + offset)
        offset += len(v)
    return np.vstack(verts).astype(np.float32), np.concatenate(idx).astype(np.uint32)
