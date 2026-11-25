import numpy as np

from scene.geometry2d import nearest_on_edges, point_in_edges

WIDE_KINDS_CLEARANCE = 1.4   # asphalt edge to wall: room for a kerb and a 1.2 m sidewalk
NARROW_CLEARANCE = 0.4       # same for footways / paths
MAX_PUSH = 3.5               # a vertex that must move further than this is cut out
STEP = 2.0                   # resampling step (m)
MIN_RUN = 3.0                # shortest piece worth keeping (m)
ITERATIONS = 6


def _densify(points, step):
    pts = np.asarray(points, dtype=np.float64)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.ceil(np.hypot(*(b - a)) / step)), 1)
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def _probe(p, per_building, all_edges):
    """Return (inside, distance to nearest wall, nearest wall point)."""
    d, qx, qz = nearest_on_edges(p[0], p[1], all_edges)
    inside = any(point_in_edges(p[0], p[1], e) for e in per_building)
    return inside, d, np.array([qx, qz])


def _near_any(pts, boxes, margin):
    lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
    return any(not (hi[0] < b[0] or lo[0] > b[2] or hi[1] < b[1] or lo[1] > b[3]) for b in boxes)


def fit_road(points, clearance, per_building, all_edges, max_push=MAX_PUSH, step=STEP):
    """Fit one polyline; returns a list of polylines (possibly empty)."""
    P = _densify(points, step)
    n = len(P)
    removed = np.zeros(n, bool)
    for _ in range(ITERATIONS):
        shift = np.zeros_like(P)
        moved = False
        for i in range(n):
            if removed[i]:
                continue
            inside, d, q = _probe(P[i], per_building, all_edges)
            need = (d + clearance) if inside else (clearance - d)
            if need <= 0.02:
                continue
            away = (q - P[i]) if inside else (P[i] - q)
            ln = float(np.hypot(*away))
            if ln < 1e-6:
                continue
            shift[i] = away / ln * need
            moved = True
        if not moved:
            break
        big = np.hypot(shift[:, 0], shift[:, 1]) > max_push
        removed |= big
        # a little smoothing keeps the pushed line from kinking
        sm = shift.copy()
        for i in range(1, n - 1):
            if not removed[i - 1] and not removed[i + 1]:
                sm[i] = 0.25 * shift[i - 1] + 0.5 * shift[i] + 0.25 * shift[i + 1]
        P = P + np.where(removed[:, None], 0.0, 0.6 * shift + 0.4 * sm)
    for i in range(n):
        if removed[i]:
            continue
        inside, d, _ = _probe(P[i], per_building, all_edges)
        if inside or d < clearance - 0.25:
            removed[i] = True
    # segments that still cut a corner of a footprint
    for i in range(n - 1):
        if removed[i] or removed[i + 1]:
            continue
        for t in (0.25, 0.5, 0.75):
            m = P[i] + (P[i + 1] - P[i]) * t
            if any(point_in_edges(m[0], m[1], e) for e in per_building):
                removed[i + 1] = True
                break
    runs, cur = [], []
    for i in range(n):
        if removed[i]:
            if len(cur) > 1:
                runs.append(np.array(cur))
            cur = []
        else:
            cur.append(P[i])
    if len(cur) > 1:
        runs.append(np.array(cur))
    keep = []
    for r in runs:
        if float(np.hypot(*np.diff(r, axis=0).T).sum()) >= MIN_RUN:
            keep.append(r)
    return keep


def fit_roads(roads, styles, obstacle_edges, max_push=MAX_PUSH):
    """Return ``(roads, report)`` with road polylines adjusted around obstacles.

    ``obstacle_edges`` is a list of edge arrays (one per building footprint).
    Only road kinds present in ``styles`` are touched.
    """
    obstacle_edges = [e for e in obstacle_edges if len(e)]
    if not obstacle_edges:
        return list(roads), {"moved": 0, "cut": 0, "checked": 0}
    all_edges = np.vstack(obstacle_edges)
    boxes = [(e[:, [0, 2]].min(), e[:, [1, 3]].min(), e[:, [0, 2]].max(), e[:, [1, 3]].max()) for e in obstacle_edges]
    out, moved, cut, checked = [], 0, 0, 0
    for r in roads:
        st = styles.get(r.get("kind"))
        pts = np.asarray(r["points"], dtype=np.float64)
        if st is None or len(pts) < 2:
            out.append(r)
            continue
        width = float(st[1])
        clearance = width / 2 + (NARROW_CLEARANCE if width <= 3.2 else WIDE_KINDS_CLEARANCE)
        if not _near_any(pts, boxes, clearance + 0.5):
            out.append(r)
            continue
        checked += 1
        pieces = fit_road(pts, clearance, obstacle_edges, all_edges, max_push)
        if len(pieces) == 1 and len(pieces[0]) == len(_densify(pts, STEP)) and \
                np.allclose(pieces[0], _densify(pts, STEP), atol=1e-6):
            out.append(r)
            continue
        old_len = float(np.hypot(*np.diff(pts, axis=0).T).sum())
        new_len = sum(float(np.hypot(*np.diff(p, axis=0).T).sum()) for p in pieces)
        if new_len < old_len - 1.0:
            cut += 1
        else:
            moved += 1
        for k, p in enumerate(pieces):
            nr = dict(r)
            nr["points"] = [[float(x), float(z)] for x, z in p]
            if len(pieces) > 1:
                nr["id"] = f"{r.get('id', 'road')}#{k}"
            out.append(nr)
    return out, {"moved": moved, "cut": cut, "checked": checked}


def overlap_area(roads, styles, footprints, samples=3, spacing=0.5):
    """Rough overlap (m2) of drawn road strips with building footprints; used by tests and the debug tool."""
    from scene.geometry2d import points_in_edges
    total = 0.0
    for r in roads:
        st = styles.get(r.get("kind"))
        if st is None:
            continue
        w = float(st[1])
        pts = np.asarray(r["points"], float)
        if len(pts) < 2:
            continue
        cells = []
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.hypot(*(b - a)))
            if L < 1e-6:
                continue
            nrm = np.array([-(b - a)[1], (b - a)[0]]) / L
            k = max(int(L / spacing), 1)
            t = (np.arange(k) + 0.5) / k
            c = a[None, :] + (b - a)[None, :] * t[:, None]
            for s_off in np.linspace(-w / 2, w / 2, samples):
                cells.append((c + nrm[None, :] * s_off, L / k * (w / samples)))
        if not cells:
            continue
        P = np.vstack([c for c, _ in cells])
        area = np.concatenate([np.full(len(c), a) for c, a in cells])
        hit = np.zeros(len(P), bool)
        for e in footprints:
            hit |= points_in_edges(P, e)
        total += float(area[hit].sum())
    return total
