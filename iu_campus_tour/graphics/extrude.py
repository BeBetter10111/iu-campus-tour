import numpy as np

EPS = 1e-9

# hình học 2D
def signed_area(poly):
    p = np.asarray(poly, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    return 0.5 * float(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]))


def clean_ring(poly, tol=0.02):
    pts = []
    for p in np.asarray(poly, dtype=np.float64):
        if not pts or np.hypot(*(p - pts[-1])) >= tol:
            pts.append(p)
    if len(pts) > 1 and np.hypot(*(pts[0] - pts[-1])) < tol:
        pts.pop()
    return np.array(pts)


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _in_triangle(p, a, b, c, eps=1e-9):
    d1, d2, d3 = _cross(a, b, p), _cross(b, c, p), _cross(c, a, p)
    return d1 >= -eps and d2 >= -eps and d3 >= -eps


def triangulate(poly):
    p = np.asarray(poly, dtype=np.float64)
    n = len(p)
    if n < 3:
        return [], 0
    idx = list(range(n))
    if signed_area(p) < 0:                 
        idx.reverse()
    tris, forced = [], 0
    while len(idx) > 3:
        m = len(idx)
        ear = None
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = p[i0], p[i1], p[i2]
            if _cross(a, b, c) <= EPS:
                continue                    # Đỉnh lõm hoặc thẳng hàng: không thể là tai
            if any(_in_triangle(p[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            ear = k
            break
        if ear is None:                     # Suy biến: cắt ép đỉnh lồi nhất để không bị kẹt
            forced += 1
            ear = max(range(m), key=lambda k: _cross(p[idx[k - 1]], p[idx[k]], p[idx[(k + 1) % m]]))
        i0, i1, i2 = idx[ear - 1], idx[ear], idx[(ear + 1) % len(idx)]
        tris.append((i0, i1, i2))
        del idx[ear]
    tris.append(tuple(idx))
    return tris, forced


# Mesh
def side_of_normal(nx, nz):
    if abs(nx) >= abs(nz):
        return "E" if nx > 0 else "W"
    return "S" if nz > 0 else "N"


def build_walls(outer, holes, height, tile_w, floor_h, kind_of_side=None, default_kind="main", base=0.0):
    groups = {}
    top_v = height / floor_h
    bot_v = base / floor_h

    def add_ring(ring, outward_negative):
        ring = np.asarray(ring, dtype=np.float64)
        if (signed_area(ring) < 0) != outward_negative:
            ring = ring[::-1]
        s = 0.0
        for i in range(len(ring)):
            p0, p1 = ring[i], ring[(i + 1) % len(ring)]
            dx, dz = p1[0] - p0[0], p1[1] - p0[1]
            length = float(np.hypot(dx, dz))
            if length < 1e-3:
                continue
            nx, nz = -dz / length, dx / length
            kind = default_kind
            if kind_of_side:
                kind = kind_of_side.get(side_of_normal(nx, nz), default_kind)
            u0, u1 = s / tile_w, (s + length) / tile_w
            s += length
            v, idx = groups.setdefault(kind, ([], []))
            start = len(v)                      # chỉ số đỉnh đầu tiên của hình chữ nhật này (KHÔNG đặt tên là `base`: trùng tham số độ cao đáy)
            v += [[p0[0], base,   p0[1], nx, 0, nz, u0, bot_v],
                  [p1[0], base,   p1[1], nx, 0, nz, u1, bot_v],
                  [p1[0], height, p1[1], nx, 0, nz, u1, top_v],
                  [p0[0], height, p0[1], nx, 0, nz, u0, top_v]]
            idx += [start, start + 1, start + 2, start, start + 2, start + 3]

    add_ring(outer, True)
    for h in holes or []:
        add_ring(h, False)
    return {k: (np.array(v, np.float32), np.array(i, np.uint32)) for k, (v, i) in groups.items()}


def build_roof(outer, height, tile=6.0, both_faces=False):
    outer = np.asarray(outer, dtype=np.float64)
    tris, forced = triangulate(outer)
    verts = [[x, height, z, 0, 1, 0, x / tile, z / tile] for x, z in outer]
    idx = []
    for i0, i1, i2 in tris:
        idx += [i0, i2, i1]
    if both_faces:
        n = len(verts)
        verts += [[x, height, z, 0, -1, 0, x / tile, z / tile] for x, z in outer]
        for i0, i1, i2 in tris:
            idx += [n + i0, n + i1, n + i2]
    return np.array(verts, np.float32), np.array(idx, np.uint32), forced


def build_underside(outer, base, tile=6.0):
    outer = np.asarray(outer, dtype=np.float64)
    tris, _ = triangulate(outer)
    verts = [[x, base, z, 0, -1, 0, x / tile, z / tile] for x, z in outer]
    idx = [k for tri in tris for k in tri]
    return np.array(verts, np.float32), np.array(idx, np.uint32)


def extrude_arrays(outer, holes, height, tile_w=10.8, floor_h=3.6, roof_tile=6.0,
                   kind_of_side=None, default_kind="main", roof_only=False, base=0.0):
    outer = clean_ring(outer)
    holes = [clean_ring(h) for h in (holes or []) if len(h) >= 3]
    if len(outer) < 3:
        raise ValueError("Đường viền có ít hơn 3 đỉnh")
    walls = {} if roof_only else build_walls(outer, holes, height, tile_w, floor_h, kind_of_side,
                                              default_kind, base=base)
    rv, ri, forced = build_roof(outer, height, roof_tile, both_faces=roof_only)
    bottom = build_underside(outer, base, roof_tile) if (base > 0 and not roof_only) else None
    return {"walls": walls, "roof": (rv, ri), "bottom": bottom, "forced": forced, "holes_ignored": bool(holes)}
