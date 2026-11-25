import numpy as np

from graphics.roads import merge_meshes
from graphics.sidewalk import sidewalk_side, miter_vectors
from scene.geometry2d import dist_points_edges, points_in_edges

DEFAULT_KINDS = {
    "tertiary": 2.2, "tertiary_link": 1.8, "residential": 2.0, "unclassified": 1.8, "service": 1.5,
}
KERB_W = 0.18
KERB_H = 0.15
STEP = 1.5
MIN_WALK = 0.9           # narrower than this and the sidewalk is dropped
WALL_GAP = 0.15          # space left between a sidewalk and a wall
BLOCKER_MIN_WIDTH = 3.5  # roads at least this wide stop a sidewalk crossing them


def _resample(points, step):
    pts = np.asarray(points, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.ceil(np.hypot(*(b - a)) / step)), 1)
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


def _min_filter(w):
    out = w.copy()
    out[1:] = np.minimum(out[1:], w[:-1])
    out[:-1] = np.minimum(out[:-1], w[1:])
    return out


def _segments(roads, styles, only_wide=True):
    rows, owner, hw = [], [], []
    for k, r in enumerate(roads):
        st = styles.get(r.get("kind"))
        if st is None or (only_wide and st[1] < BLOCKER_MIN_WIDTH):
            continue
        pts = r["points"]
        for a, b in zip(pts[:-1], pts[1:]):
            rows.append([a[0], a[1], b[0], b[1]])
            owner.append(k)
            hw.append(st[1] / 2)
    return np.array(rows, float).reshape(-1, 4), np.array(owner, int), np.array(hw, float)


def _near_other_roads(pts, segs, owner, hw, me, margin):
    """True where a point lies on the carriageway of a road other than ``me``."""
    keep = owner != me
    if not keep.any():
        return np.zeros(len(pts), bool)
    s, h = segs[keep], hw[keep]
    x0, z0, x1, z1 = s.T
    abx, abz = x1 - x0, z1 - z0
    denom = np.maximum(abx * abx + abz * abz, 1e-12)
    px, pz = pts[:, 0:1], pts[:, 1:2]
    t = np.clip(((px - x0) * abx + (pz - z0) * abz) / denom, 0.0, 1.0)
    d = np.sqrt((px - (x0 + t * abx)) ** 2 + (pz - (z0 + t * abz)) ** 2)
    return (d < (h + margin)[None, :]).any(axis=1)


def _in_any(pts, rings):
    hit = np.zeros(len(pts), bool)
    for e in rings:
        hit |= points_in_edges(pts, e)
    return hit


def plan_sidewalks(roads, styles, obstacle_edges, keepout_edges, bounds, cfg=None):
    """Return a list of dicts ``{"P", "side", "hw", "walk_w", "y_road", "road"}`` (one per continuous run)."""
    cfg = cfg or {}
    kinds = {**DEFAULT_KINDS, **cfg.get("kinds", {})}
    kerb_w = float(cfg.get("kerb_width", KERB_W))
    all_obst = np.vstack(obstacle_edges) if obstacle_edges else np.zeros((0, 4))
    segs, owner, hw_seg = _segments(roads, styles)
    x0, z0, x1, z1 = bounds
    runs = []
    for k, r in enumerate(roads):
        st = styles.get(r.get("kind"))
        want = kinds.get(r.get("kind"))
        if st is None or not want or len(r["points"]) < 2:
            continue
        W, hw = float(want), st[1] / 2
        P = _resample(r["points"], STEP)
        inb = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= z0) & (P[:, 1] <= z1)
        if inb.sum() < 2:
            continue
        m, _ = miter_vectors(P)
        for side in (1, -1):
            pk = P + side * m * (hw + kerb_w)
            pin = P + side * m * (hw + 0.02)
            avail = np.minimum(dist_points_edges(pk, all_obst) - WALL_GAP, W) if len(all_obst) else np.full(len(P), W)
            wi = _min_filter(np.clip(avail, 0.0, W))
            pout = P + side * m * (hw + kerb_w + wi[:, None])
            bad = ~inb | (wi < MIN_WALK)
            bad |= _in_any(pk, obstacle_edges) | _in_any(pout, obstacle_edges) | _in_any(pin, obstacle_edges)
            bad |= _in_any(pk, keepout_edges) | _in_any(pout, keepout_edges)
            for probe in (pin, pk, pout):
                bad |= _near_other_roads(probe, segs, owner, hw_seg, k, 0.15)
            i, n = 0, len(P)
            while i < n:
                if bad[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < n and not bad[j + 1]:
                    j += 1
                if j > i and float(np.hypot(*np.diff(P[i:j + 1], axis=0).T).sum()) >= 3.0:
                    runs.append({"P": P[i:j + 1], "side": side, "hw": hw, "walk_w": wi[i:j + 1].copy(),
                                 "y_road": float(st[2]), "road": k})
                i = j + 1
    return runs


def build_sidewalks(roads, styles, obstacle_edges, keepout_edges, bounds, cfg=None, tile_walk=2.0, tile_kerb=4.0):
    """Build merged sidewalk and kerb meshes. Returns ``(walk, kerb, stats)`` (meshes are ``(verts, idx)`` or None)."""
    cfg = cfg or {}
    kerb_w = float(cfg.get("kerb_width", KERB_W))
    top = float(cfg.get("kerb_height", KERB_H))
    runs = plan_sidewalks(roads, styles, obstacle_edges, keepout_edges, bounds, cfg)
    walks, kerbs, length = [], [], 0.0
    for run in runs:
        # tiny per-road height offset avoids z-fighting where two sidewalks overlap
        y_top = top + 0.0006 * (run["road"] % 7)
        w, kb = sidewalk_side(run["P"], run["side"], run["hw"], kerb_w, run["walk_w"], run["y_road"] + 0.002,
                              y_top, tile_walk, tile_kerb)
        walks.append(w)
        kerbs.append(kb)
        length += float(np.hypot(*np.diff(run["P"], axis=0).T).sum())
    return merge_meshes(walks), merge_meshes(kerbs), {"runs": len(runs), "length": length}
