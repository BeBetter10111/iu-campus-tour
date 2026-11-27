import numpy as np

from scene.geometry2d import point_in_edges, rings_to_edges


def apply_road_edits(roads, edits):
    """Return a new road list with ``edits`` = {"remove": [ids], "replace": {id: {...}}, "add": [roads]}."""
    if not edits:
        return list(roads), {"removed": 0, "replaced": 0, "added": 0}
    remove = set(edits.get("remove", []))
    replace = edits.get("replace", {})
    out, rep = [], {"removed": 0, "replaced": 0, "added": 0}
    for r in roads:
        rid = r.get("id")
        if rid in remove:
            rep["removed"] += 1
            continue
        if rid in replace:
            r = {**r, **replace[rid]}
            rep["replaced"] += 1
        out.append(r)
    for r in edits.get("add", []):
        out.append({"name": "", **r})
        rep["added"] += 1
    return out, rep


def _polyline_len(P):
    P = np.asarray(P, float)
    return float(np.hypot(*(P[1:] - P[:-1]).T).sum()) if len(P) > 1 else 0.0


def _seg_params(a, b, edges):
    """Parameters t in (0, 1) where segment a->b crosses any polygon edge."""
    r = b - a
    x0, z0, x1, z1 = edges.T
    sx, sz = x1 - x0, z1 - z0
    den = r[0] * sz - r[1] * sx
    ok = np.abs(den) > 1e-12
    den = np.where(ok, den, 1.0)
    wx, wz = x0 - a[0], z0 - a[1]
    t = (wx * sz - wz * sx) / den
    u = (wx * r[1] - wz * r[0]) / den
    hit = ok & (t > 1e-9) & (t < 1 - 1e-9) & (u >= 0) & (u <= 1)
    return t[hit]


def _dist_to_polyline(p, P):
    a, b = P[:-1], P[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    c = a + ab * t[:, None]
    return float(np.hypot(*(p - c).T).min())


def clip_roads_to_buildings(roads, building_rings, min_len=3.0, connect_tol=1.5):
    """Cut roads where they pass under buildings and drop the dangling leftovers.

    A piece that touches a building on one side is kept only if its other end joins another
    road; otherwise it is a dead-end stub (nothing drives there) and is removed.
    """
    edge_sets = [rings_to_edges([ring]) for ring in building_rings]
    edge_sets = [e for e in edge_sets if len(e)]
    if not edge_sets:
        return list(roads), {"cut": 0, "dropped": 0}

    def inside(p):
        return any(point_in_edges(float(p[0]), float(p[1]), e) for e in edge_sets)

    polylines = [np.asarray(r["points"], float) for r in roads]
    out, rep = [], {"cut": 0, "dropped": 0}
    for idx, (r, P) in enumerate(zip(roads, polylines)):
        if len(P) < 2:
            out.append(r)
            continue
        # quick reject: no vertex inside and no crossing -> untouched
        crossings = [np.sort(np.concatenate([_seg_params(a, b, e) for e in edge_sets]))
                     for a, b in zip(P[:-1], P[1:])]
        if not any(len(c) for c in crossings) and not any(inside(p) for p in P):
            out.append(r)
            continue
        # walk the polyline, splitting at crossings, and collect the outside runs
        pieces, cur, cur_start_cut = [], [], False
        for (a, b), ts in zip(zip(P[:-1], P[1:]), crossings):
            cuts = [0.0] + list(ts) + [1.0]
            for t0, t1 in zip(cuts[:-1], cuts[1:]):
                if t1 - t0 < 1e-9:
                    continue
                p0, p1 = a + (b - a) * t0, a + (b - a) * t1
                if inside((p0 + p1) / 2):
                    if cur:
                        pieces.append((cur, cur_start_cut, True))
                        cur = []
                    continue
                if not cur:
                    cur, cur_start_cut = [p0], t0 > 0.0 or (len(pieces) > 0 or inside(p0))
                cur.append(p1)
        if cur:
            pieces.append((cur, cur_start_cut, False))
        if len(pieces) == 1 and not pieces[0][1] and not pieces[0][2]:
            out.append(r)
            continue
        rep["cut"] += 1
        others = [Q for j, Q in enumerate(polylines) if j != idx and len(Q) > 1]

        def joined(p):
            return any(_dist_to_polyline(p, Q) <= connect_tol for Q in others)

        for k, (pts, start_cut, end_cut) in enumerate(pieces):
            pts = np.asarray(pts, float)
            free_ends = ([pts[0]] if not start_cut else []) + ([pts[-1]] if not end_cut else [])
            if _polyline_len(pts) < min_len or not any(joined(p) for p in free_ends):
                rep["dropped"] += 1
                continue
            out.append({**r, "id": f"{r.get('id', '?')}#{k}", "points": [[float(x), float(z)] for x, z in pts]})
    return out, rep


def _resample(P, step):
    seg = P[1:] - P[:-1]
    L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(L)])
    s = np.unique(np.concatenate([np.arange(0.0, cum[-1], step), cum]))
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    t = (s - cum[i]) / np.maximum(L[i], 1e-12)
    return s, P[i] + seg[i] * t[:, None], cum


def _slice(P, cum, s0, s1):
    """Sub-polyline between arc lengths s0..s1."""
    def at(s):
        i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(P) - 2))
        t = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-12)
        return P[i] + (P[i + 1] - P[i]) * t
    inner = [P[i] for i in range(len(P)) if s0 < cum[i] < s1]
    return np.array([at(s0)] + inner + [at(s1)])


def _end_dir(P, at_start):
    d = (P[1] - P[0]) if at_start else (P[-2] - P[-1])
    return d / max(np.hypot(*d), 1e-12)


def split_at_junctions(roads, widths, marked_kinds, vehicle_kinds, step=0.5, pad=1.5, min_marked=6.0):
    """Split each marked road into pieces [(points, plain)] around places where it meets another road.

    A sample is "plain" (no markings) when its cross-section overlaps another vehicle road
    (centre distance < hw_other + hw_self * |cos(angle between the roads)|), grown by ``pad``. Two ways that merely continue each other
    (shared end point, almost straight, nothing else there) are not treated as a junction.
    """
    veh = [(i, np.asarray(r["points"], float), widths[r["kind"]] / 2)
           for i, r in enumerate(roads) if r.get("kind") in vehicle_kinds and len(r["points"]) > 1]
    ends = {}
    for i, P, _ in veh:
        for at_start in (True, False):
            key = tuple(np.round(P[0 if at_start else -1], 1))
            ends.setdefault(key, []).append((i, _end_dir(P, at_start)))

    def continuation(i, j):
        for lst in ends.values():
            ids = [k for k, _ in lst]
            if len(lst) == 2 and set(ids) == {i, j}:
                (_, d1), (_, d2) = lst
                if float(d1 @ d2) < -0.85:
                    return True
        return False

    result = {}
    for i, P, hw in veh:
        if roads[i]["kind"] not in marked_kinds:
            continue
        s, S, cum = _resample(P, step)
        seg_i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(P) - 2)
        tang = P[seg_i + 1] - P[seg_i]
        tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-12)[:, None]
        plain = np.zeros(len(S), bool)
        for j, Q, hw2 in veh:
            if j == i or continuation(i, j):
                continue
            lo, hi = Q.min(0) - hw - hw2, Q.max(0) + hw + hw2
            near = (S[:, 0] > lo[0]) & (S[:, 0] < hi[0]) & (S[:, 1] > lo[1]) & (S[:, 1] < hi[1])
            if not near.any():
                continue
            a, b = Q[:-1], Q[1:]
            ab = b - a
            for k in np.nonzero(near)[0]:
                t = np.clip(((S[k] - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
                d = np.hypot(*(S[k] - (a + ab * t[:, None])).T)
                m = int(np.argmin(d))
                if d[m] >= hw + hw2:
                    continue
                other = ab[m] / max(np.hypot(*ab[m]), 1e-12)
                cos = abs(float(other @ tang[k]))
                # a parallel neighbour (other carriageway of a dual road) is not a junction
                if cos > 0.94 and d[m] > max(hw, hw2):
                    continue
                # our cross-section reaches the other strip only if d < hw2 + hw*|cos(angle)|
                if d[m] < hw2 + hw * cos + 0.3:
                    plain[k] = True
        if not plain.any():
            continue
        # grow plain zones by `pad`, then swallow marked runs too short to read as a road
        grow = np.zeros_like(plain)
        for k in np.nonzero(plain)[0]:
            grow |= np.abs(s - s[k]) <= pad
        plain = grow
        runs, k0 = [], 0
        for k in range(1, len(s) + 1):
            if k == len(s) or plain[k] != plain[k0]:
                runs.append([s[k0], s[k - 1] if k == len(s) else s[k], bool(plain[k0])])
                k0 = k
        for r in runs:
            if not r[2] and r[1] - r[0] < min_marked:
                r[2] = True
        merged = []
        for r in runs:
            if merged and merged[-1][2] == r[2]:
                merged[-1][1] = r[1]
            else:
                merged.append(list(r))
        result[i] = [(_slice(P, cum, a, b), is_plain) for a, b, is_plain in merged if b - a > 1e-6]
    return result
