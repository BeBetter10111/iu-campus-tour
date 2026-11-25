import numpy as np

from graphics.roads import merge_meshes

FENCE_HEIGHT = 0.45
POST = 0.07
RAIL = 0.05
POST_SPACING = 2.0
GATE_SNAP_MAX = 8.0


def oriented_box(cx, cz, y0, y1, length, width, ang):
    """Build a Y-axis rotated box."""
    c, s = np.cos(ang), np.sin(ang)
    ax, az = c, s
    bx, bz = -s, c
    hl, hw = length / 2, width / 2
    corners = [(cx + sx * hl * ax + sz_ * hw * bx, cz + sx * hl * az + sz_ * hw * bz)
               for sx, sz_ in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    verts, idx = [], []

    def face(p, q, y_lo, y_hi, n):
        st = len(verts)
        verts.extend([[p[0], y_lo, p[1], *n, 0, 0], [q[0], y_lo, q[1], *n, 1, 0],
                      [q[0], y_hi, q[1], *n, 1, 1], [p[0], y_hi, p[1], *n, 0, 1]])
        idx.extend([st, st + 1, st + 2, st, st + 2, st + 3])

    for k in range(4):
        p, q = corners[k], corners[(k + 1) % 4]
        dx, dz = q[0] - p[0], q[1] - p[1]
        L = np.hypot(dx, dz)
        face(p, q, y0, y1, (dz / L, 0.0, -dx / L))
    st = len(verts)
    verts.extend([[x, y1, z, 0, 1, 0, 0, 0] for x, z in corners])
    idx.extend([st, st + 1, st + 2, st, st + 2, st + 3])
    sb = len(verts)
    verts.extend([[x, y0, z, 0, -1, 0, 0, 0] for x, z in corners])
    idx.extend([sb, sb + 2, sb + 1, sb, sb + 3, sb + 2])
    v = np.array(verts, np.float32)
    i = np.array(idx, np.uint32)
    tri = v[i.reshape(-1, 3)]
    geo = np.cross(tri[:, 1, :3] - tri[:, 0, :3], tri[:, 2, :3] - tri[:, 0, :3])
    if np.median((geo * tri[:, 0, 3:6]).sum(axis=1)) < 0:
        i = i.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1)
    return v, i


def fence_for_polygon(ring, gates=None, height=FENCE_HEIGHT):
    """Build a fence around a polygon and report gate placement."""
    ring = np.asarray(ring, dtype=np.float64)
    n = len(ring)
    edges = []
    for i in range(n):
        p0, p1 = ring[i], ring[(i + 1) % n]
        L = float(np.hypot(*(p1 - p0)))
        if L > 1e-3:
            edges.append((p0, p1, L, (p1 - p0) / L))
    gaps = {i: [] for i in range(len(edges))}
    report = []
    for g in gates or []:
        at = np.asarray(g["at"], float)
        best = min(((float(np.hypot(*(at - (e[0] + e[3] * np.clip((at - e[0]) @ e[3], 0, e[2]))))), i,
                    float(np.clip((at - e[0]) @ e[3], 0, e[2]))) for i, e in enumerate(edges)), key=lambda b: b[0])
        if best[0] > GATE_SNAP_MAX:
            report.append({"at": tuple(at), "ok": False})
            continue
        w = float(g.get("width", 4.0))
        _, i, t = best
        gaps[i].append((max(0.0, t - w / 2), min(edges[i][2], t + w / 2)))
        report.append({"at": tuple(edges[i][0] + edges[i][3] * t), "ok": True})

    parts, boxes = [], []
    for i, (p0, p1, L, d) in enumerate(edges):
        ang = float(np.arctan2(d[1], d[0]))
        cuts = sorted(gaps[i])
        pieces, cur = [], 0.0
        for a, b in cuts:
            if a > cur + 0.05:
                pieces.append((cur, a))
            cur = max(cur, b)
        if cur < L - 0.05:
            pieces.append((cur, L))
        for a, b in pieces:
            seg = b - a
            mid = p0 + d * (a + seg / 2)
            parts.append(oriented_box(mid[0], mid[1], 0.36, 0.36 + RAIL, seg, RAIL, ang))
            parts.append(oriented_box(mid[0], mid[1], 0.16, 0.16 + RAIL, seg, RAIL, ang))
            k = max(int(np.ceil(seg / POST_SPACING)), 1)
            for j in range(k + 1):
                c = p0 + d * (a + seg * j / k)
                parts.append(oriented_box(c[0], c[1], 0.0, height, POST, POST, ang))
            m = max(int(np.ceil(seg / 3.0)), 1)
            for j in range(m):
                q0, q1 = p0 + d * (a + seg * j / m), p0 + d * (a + seg * (j + 1) / m)
                boxes.append([min(q0[0], q1[0]) - 0.05, 0.0, min(q0[1], q1[1]) - 0.05,
                              max(q0[0], q1[0]) + 0.05, height, max(q0[1], q1[1]) + 0.05])
    return merge_meshes(parts), np.array(boxes, np.float32).reshape(-1, 6), report
