import numpy as np

EPS = 1e-9


def signed_area(poly):
    """Return the signed polygon area in the x-z plane."""
    p = np.asarray(poly, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    return 0.5 * float(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]))


def clean_ring(poly, tol=0.02):
    """Remove duplicate points, including a repeated closing point."""
    pts = []
    for p in np.asarray(poly, dtype=np.float64):
        if not pts or np.hypot(*(p - pts[-1])) >= tol:
            pts.append(p)
    if len(pts) > 1 and np.hypot(*(pts[0] - pts[-1])) < tol:
        pts.pop()
    return np.array(pts)


def _cross(a, b, c):
    """Return the 2D cross product of vectors b-a and c-a."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _in_triangle(p, a, b, c, eps=1e-9):
    d1, d2, d3 = _cross(a, b, p), _cross(b, c, p), _cross(c, a, p)
    return d1 >= -eps and d2 >= -eps and d3 >= -eps


def triangulate(poly):
    """Triangulate a simple polygon with ear clipping."""
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
                continue
            if any(_in_triangle(p[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            ear = k
            break
        if ear is None:
            forced += 1
            ear = max(range(m), key=lambda k: _cross(p[idx[k - 1]], p[idx[k]], p[idx[(k + 1) % m]]))
        i0, i1, i2 = idx[ear - 1], idx[ear], idx[(ear + 1) % len(idx)]
        tris.append((i0, i1, i2))
        del idx[ear]
    tris.append(tuple(idx))
    return tris, forced


def side_of_normal(nx, nz):
    """Map an outward normal to a cardinal wall side."""
    if abs(nx) >= abs(nz):
        return "E" if nx > 0 else "W"
    return "S" if nz > 0 else "N"


def build_walls(outer, holes, height, tile_w, floor_h, kind_of_side=None, default_kind="main", base=0.0):
    """Build wall meshes, grouped by facade kind."""
    groups = {}
    top_v = height / floor_h
    bot_v = base / floor_h

    def add_ring(ring, outward_negative):
        ring = np.asarray(ring, dtype=np.float64)
        # Orient the ring according to whether it is an outer boundary or a hole.
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
            start = len(v)
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
    """Build a triangulated flat roof."""
    outer = np.asarray(outer, dtype=np.float64)
    tris, forced = triangulate(outer)
    verts = [[x, height, z, 0, 1, 0, x / tile, z / tile] for x, z in outer]
    idx = []
    for i0, i1, i2 in tris:
        # Flip the triangle order so the roof normal points upward.
        idx += [i0, i2, i1]
    if both_faces:
        n = len(verts)
        verts += [[x, height, z, 0, -1, 0, x / tile, z / tile] for x, z in outer]
        for i0, i1, i2 in tris:
            idx += [n + i0, n + i1, n + i2]
    return np.array(verts, np.float32), np.array(idx, np.uint32), forced


def build_underside(outer, base, tile=6.0):
    """Build the downward-facing underside of a raised structure."""
    outer = np.asarray(outer, dtype=np.float64)
    tris, _ = triangulate(outer)
    verts = [[x, base, z, 0, -1, 0, x / tile, z / tile] for x, z in outer]
    idx = [k for tri in tris for k in tri]
    return np.array(verts, np.float32), np.array(idx, np.uint32)


DOOR_SNAP_MAX = 8.0
DOOR_HEIGHT = 2.7
DOOR_MIN_PIECE = 0.15


def _quad(v, idx, p0, p1, y0, y1, n, u0, u1, v0, v1, interior=False):
    """Append a vertical quad to a mesh, optionally oriented for an interior face."""
    start = len(v)
    if not interior:
        v += [[p0[0], y0, p0[1], n[0], 0, n[1], u0, v0], [p1[0], y0, p1[1], n[0], 0, n[1], u1, v0],
              [p1[0], y1, p1[1], n[0], 0, n[1], u1, v1], [p0[0], y1, p0[1], n[0], 0, n[1], u0, v1]]
    else:
        m = (-n[0], -n[1])
        v += [[p1[0], y0, p1[1], m[0], 0, m[1], u1, v0], [p0[0], y0, p0[1], m[0], 0, m[1], u0, v0],
              [p0[0], y1, p0[1], m[0], 0, m[1], u0, v1], [p1[0], y1, p1[1], m[0], 0, m[1], u1, v1]]
    idx += [start, start + 1, start + 2, start, start + 2, start + 3]


def _merge_gaps(gaps):
    out = []
    for a, b in sorted(gaps):
        if out and a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def build_walls_ground(outer, height, tile_w, floor_h, default_kind, ground_kind, kind_of_side=None,
                       doors=None, door_h=DOOR_HEIGHT, interior=False):
    """Build walls with a separate ground floor and door openings."""
    ring = np.asarray(outer, dtype=np.float64)
    if signed_area(ring) > 0:
        ring = ring[::-1]
    n_e = len(ring)
    edges = []
    for i in range(n_e):
        p0, p1 = ring[i], ring[(i + 1) % n_e]
        L = float(np.hypot(*(p1 - p0)))
        if L < 1e-3:
            edges.append(None)
            continue
        d = (p1 - p0) / L
        edges.append((p0, p1, L, d, (-d[1], d[0])))

    def run_extent(i, cos_tol=0.9986):
        """Measure the continuous straight wall segment around an edge."""
        back = fwd = 0.0
        j = i
        for _ in range(n_e):
            k = (j - 1) % n_e
            if k == i or edges[k] is None or float(edges[k][3] @ edges[i][3]) < cos_tol:
                break
            back += edges[k][2]
            j = k
        j = i
        for _ in range(n_e):
            k = (j + 1) % n_e
            if k == i or edges[k] is None or float(edges[k][3] @ edges[i][3]) < cos_tol:
                break
            fwd += edges[k][2]
            j = k
        return back, fwd

    gaps = {i: [] for i in range(n_e)}
    report = []
    for door in doors or []:
        at = np.asarray(door["at"], dtype=np.float64)
        best = None
        for i, e in enumerate(edges):
            if e is None:
                continue
            p0, p1, L, d, nrm = e
            t = float(np.clip((at - p0) @ d, 0.0, L))
            dist = float(np.hypot(*(at - (p0 + d * t))))
            if best is None or dist < best[0]:
                best = (dist, i, t)
        if best is None or best[0] > DOOR_SNAP_MAX:
            report.append({"at": tuple(at), "ok": False, "why": f"{best[0]:.1f} m from wall" if best else "no wall"})
            continue
        dist, i, t = best
        p0, p1, L, d, nrm = edges[i]
        w = min(float(door.get("width", 2.6)), L - 2 * DOOR_MIN_PIECE)
        if w < 0.8:
            report.append({"at": tuple(at), "ok": False, "why": "wall edge is too short"})
            continue
        s0 = min(max(DOOR_MIN_PIECE, t - w / 2), L - DOOR_MIN_PIECE - w)
        gaps[i].append((s0, s0 + w))
        c = p0 + d * (s0 + w / 2)
        report.append({"at": (float(c[0]), float(c[1])), "normal": (float(nrm[0]), float(nrm[1])),
                       "width": w, "snap_dist": dist, "ok": True, "spec": door,
                       "edge_len": L, "s_center": s0 + w / 2, "run": run_extent(i)})

    band_top = min(floor_h, height)
    has_upper = height > band_top + 0.05
    walls, iv, ii, solid = {}, [], [], []
    s_cum = 0.0
    for i, e in enumerate(edges):
        if e is None:
            continue
        p0, p1, L, d, nrm = e
        kind_up = (kind_of_side or {}).get(side_of_normal(*nrm), default_kind)
        gs = _merge_gaps(gaps[i])
        pieces, cur = [], 0.0
        for a, b in gs:
            if a > cur + 1e-6:
                pieces.append((cur, a))
            cur = b
        if cur < L - 1e-6:
            pieces.append((cur, L))
        gv, gi = walls.setdefault(ground_kind, ([], []))
        for a, b in pieces:
            pa, pb = p0 + d * a, p0 + d * b
            _quad(gv, gi, pa, pb, 0.0, band_top, nrm, (s_cum + a) / tile_w, (s_cum + b) / tile_w, 0.0, band_top / floor_h)
            if interior:
                _quad(iv, ii, pa, pb, 0.0, band_top, nrm, (s_cum + a) / tile_w, (s_cum + b) / tile_w, 0.0, band_top / floor_h, True)
            solid.append([pa[0], pa[1], pb[0], pb[1]])
        if door_h < band_top - 0.05:
            for a, b in gs:
                pa, pb = p0 + d * a, p0 + d * b
                _quad(gv, gi, pa, pb, door_h, band_top, nrm, (s_cum + a) / tile_w, (s_cum + b) / tile_w,
                      door_h / floor_h, band_top / floor_h)
                if interior:
                    _quad(iv, ii, pa, pb, door_h, band_top, nrm, (s_cum + a) / tile_w, (s_cum + b) / tile_w,
                          door_h / floor_h, band_top / floor_h, True)
        if has_upper:
            uv_, ui_ = walls.setdefault(kind_up, ([], []))
            _quad(uv_, ui_, p0, p1, band_top, height, nrm, s_cum / tile_w, (s_cum + L) / tile_w,
                  band_top / floor_h, height / floor_h)
        s_cum += L

    out = {"walls": {k: (np.array(v, np.float32), np.array(i_, np.uint32)) for k, (v, i_) in walls.items()},
           "interior": None, "wall_edges": None, "doors": report}
    if interior:
        tris, _ = triangulate(ring)
        fl_v = np.array([[x, 0.02, z, 0, 1, 0, x / 2.0, z / 2.0] for x, z in ring], np.float32)
        fl_i = np.array([k for tri in tris for k in (tri[0], tri[2], tri[1])], np.uint32)
        ce_v = np.array([[x, band_top, z, 0, -1, 0, x / 6.0, z / 6.0] for x, z in ring], np.float32)
        ce_i = np.array([k for tri in tris for k in tri], np.uint32)
        out["interior"] = {"walls": (np.array(iv, np.float32), np.array(ii, np.uint32)),
                           "floor": (fl_v, fl_i), "ceiling": (ce_v, ce_i)}
        out["wall_edges"] = np.array(solid, dtype=np.float64) if solid else np.zeros((0, 4))
    return out


def _plinth_walls(outer, lift, tile_w, floor_h, kind):
    """Build a solid plinth wall up to the first-floor height."""
    ring = np.asarray(outer, dtype=np.float64)
    if signed_area(ring) > 0:
        ring = ring[::-1]
    v, idx, s = [], [], 0.0
    for i in range(len(ring)):
        p0, p1 = ring[i], ring[(i + 1) % len(ring)]
        L = float(np.hypot(*(p1 - p0)))
        if L < 1e-3:
            continue
        d = (p1 - p0) / L
        _quad(v, idx, p0, p1, 0.0, lift, (-d[1], d[0]), s / tile_w, (s + L) / tile_w, 0.0, lift / floor_h)
        s += L
    return np.array(v, np.float32), np.array(idx, np.uint32)


def extrude_arrays(outer, holes, height, tile_w=10.8, floor_h=3.6, roof_tile=6.0,
                   kind_of_side=None, default_kind="main", roof_only=False, base=0.0,
                   ground_kind=None, doors=None, enterable=False, lift=0.0, plinth=None):
    """Build complete building geometry as NumPy arrays."""
    outer = clean_ring(outer)
    holes = [clean_ring(h) for h in (holes or []) if len(h) >= 3]
    if len(outer) < 3:
        raise ValueError("Outer ring must contain at least three vertices")
    extra = {"interior": None, "wall_edges": None, "doors": []}
    if roof_only:
        walls = {}
    elif (ground_kind or doors or enterable) and base == 0:
        g = build_walls_ground(outer, height, tile_w, floor_h, default_kind, ground_kind or default_kind,
                               kind_of_side, doors, interior=enterable)
        walls, extra = g["walls"], {k: g[k] for k in ("interior", "wall_edges", "doors")}
    else:
        walls = build_walls(outer, holes, height, tile_w, floor_h, kind_of_side, default_kind, base=base)
    rv, ri, forced = build_roof(outer, height, roof_tile, both_faces=roof_only)
    bottom = build_underside(outer, base, roof_tile) if (base > 0 and not roof_only) else None
    res = {"walls": walls, "roof": (rv, ri), "bottom": bottom, "forced": forced,
           "holes_ignored": bool(holes), "lift": float(lift), **extra}

    if lift:
        # Raise the complete building geometry while preserving texture UV coordinates.
        def up(vi):
            v = vi[0].copy()
            v[:, 1] += lift
            return v, vi[1]
        res["walls"] = {k: up(vi) for k, vi in res["walls"].items()}
        res["roof"] = up(res["roof"])
        if res["bottom"] is not None:
            res["bottom"] = up(res["bottom"])
        if res["interior"] is not None:
            res["interior"] = {k: up(vi) for k, vi in res["interior"].items()}
        if plinth is None:
            plinth = base == 0 and not roof_only
        if plinth:
            pk = ground_kind or default_kind
            pv, pi = _plinth_walls(outer, lift, tile_w, floor_h, pk)
            if pk in res["walls"]:
                v0, i0 = res["walls"][pk]
                res["walls"][pk] = (np.vstack([v0, pv]), np.concatenate([i0, pi + len(v0)]))
            else:
                res["walls"][pk] = (pv, pi)
    return res
