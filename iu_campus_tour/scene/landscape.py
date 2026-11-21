import json
import math

import numpy as np

from scene.geometry2d import rings_to_edges

# Bán kính chiếm chỗ (m) của từng loại, hai cây cách nhau tối thiểu r_a + r_b
RADIUS = {"tree": 2.6, "small": 1.5, "shrub": 0.7}


def load_landscape(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _segments(edges_list):
    return np.vstack(edges_list) if edges_list else np.zeros((0, 4))


def dist_points_segments(P, segs):
    if len(segs) == 0:
        return np.full(len(P), np.inf)
    x0, z0, x1, z1 = segs.T
    abx, abz = (x1 - x0)[None], (z1 - z0)[None]
    denom = np.maximum(abx ** 2 + abz ** 2, 1e-12)
    out = np.empty(len(P))
    for s in range(0, len(P), 400):
        p = P[s:s + 400]
        t = np.clip(((p[:, :1] - x0) * abx + (p[:, 1:] - z0) * abz) / denom, 0, 1)
        cx, cz = x0 + t * abx, z0 + t * abz
        out[s:s + 400] = np.sqrt(((p[:, :1] - cx) ** 2 + (p[:, 1:] - cz) ** 2).min(axis=1))
    return out


def points_in_edges(P, edges):
    if len(edges) == 0:
        return np.zeros(len(P), bool)
    x0, z0, x1, z1 = edges.T
    px, pz = P[:, :1], P[:, 1:]
    crosses = (z0[None] > pz) != (z1[None] > pz)
    dz = np.where(z1 != z0, z1 - z0, 1.0)[None]
    x_at = x0[None] + (pz - z0[None]) * (x1 - x0)[None] / dz
    return (crosses & (px < x_at)).sum(axis=1) % 2 == 1


class Constraints:

    def __init__(self, building_edges, road_segs, parking_edges, doors, tour_pts, spawn_xz=None):
        self.building_edges = building_edges                      # list (M,4) mỗi tòa
        self.all_walls = _segments(building_edges)
        self.road_segs = road_segs                                # (S,5): x0,z0,x1,z1,nửa_bề_rộng
        self.parking_edges = parking_edges                        # list (M,4) mỗi bãi xe
        self.doors = doors                                        # list (x, z, nx, nz)
        self.tour = np.asarray(tour_pts, float).reshape(-1, 2) if tour_pts is not None and len(tour_pts) else np.zeros((0, 2))
        self.spawn = spawn_xz

    def valid(self, P, wall=3.0, road=1.5, tour=3.0, parking=1.5):
        P = np.asarray(P, float).reshape(-1, 2)
        ok = np.ones(len(P), bool)
        for e in self.building_edges:                             # không nằm trong tòa nhà
            ok &= ~points_in_edges(P, e)
        ok &= dist_points_segments(P, self.all_walls) >= wall
        if len(self.road_segs):
            x0, z0, x1, z1, hw = self.road_segs.T
            for s in range(0, len(P), 300):
                p = P[s:s + 300]
                abx, abz = (x1 - x0)[None], (z1 - z0)[None]
                t = np.clip(((p[:, :1] - x0) * abx + (p[:, 1:] - z0) * abz) / np.maximum(abx ** 2 + abz ** 2, 1e-12), 0, 1)
                d = np.sqrt((p[:, :1] - (x0 + t * abx)) ** 2 + (p[:, 1:] - (z0 + t * abz)) ** 2)
                ok[s:s + 300] &= (d - hw[None] >= road).all(axis=1)   # cách MÉP đường >= road
        for e in self.parking_edges:
            ok &= ~points_in_edges(P, e)
            ok &= dist_points_segments(P, e) >= parking
        if len(self.tour):
            d = np.sqrt(((P[:, None, :] - self.tour[None]) ** 2).sum(axis=2)).min(axis=1)
            ok &= d >= tour
        for dx, dz, nx, nz in self.doors:                         # vòng quanh cửa + hành lang từ cửa đi ra
            rel = P - np.array([dx, dz])
            along = rel @ np.array([nx, nz])
            across = rel @ np.array([-nz, nx])
            ok &= ~((np.hypot(rel[:, 0], rel[:, 1]) < 3.2) | ((along >= -0.5) & (along <= 9.0) & (np.abs(across) < 2.0)))
        if self.spawn is not None:
            ok &= np.hypot(*(P - np.asarray(self.spawn)).T) >= 6.0
        return ok


class Placer:
    def __init__(self, constraints, rng):
        self.c = constraints
        self.rng = rng
        self.pos = np.zeros((0, 2))
        self.rad = np.zeros(0)
        self.items = []                                           # (x, z, kind, nguồn)

    def _free(self, p, r):
        if len(self.pos) == 0:
            return True
        return bool(np.all(np.hypot(*(self.pos - p).T) >= self.rad + r))

    def add(self, cands, kinds, source, limit, **clear):
        cands = np.asarray(cands, float).reshape(-1, 2)
        if len(cands) == 0 or limit <= 0:
            return 0
        names = list(kinds)
        w = np.array([kinds[k] for k in names], float)
        w /= w.sum()
        ok = self.c.valid(cands, **clear)
        got = 0
        for p in cands[ok]:
            kind = names[int(self.rng.choice(len(names), p=w))]
            r = RADIUS[kind]
            if self._free(p, r):
                self.pos = np.vstack([self.pos, p])
                self.rad = np.append(self.rad, r)
                self.items.append((float(p[0]), float(p[1]), kind, source))
                got += 1
                if got >= limit:
                    break
        return got


def _in_polygon_candidates(poly, n, rng):
    poly = np.asarray(poly, float)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    P = rng.uniform(lo, hi, size=(n, 2))
    return P[points_in_edges(P, rings_to_edges([poly]))]


def road_side_points(roads, styles, spacing, bounds, hw_extra):
    x0, z0, x1, z1 = bounds
    out = []
    for r in roads:
        st = styles.get(r.get("kind"))
        if st is None or r["kind"] not in ("tertiary", "service", "residential", "unclassified"):
            continue
        pts = np.asarray(r["points"], float)
        if len(pts) < 2 or not ((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > z0) & (pts[:, 1] < z1)).any():
            continue
        off = st[1] / 2 + hw_extra
        seg = pts[1:] - pts[:-1]
        L = np.hypot(seg[:, 0], seg[:, 1])
        total = L.sum()
        if total < spacing:
            continue
        cum = np.concatenate([[0], np.cumsum(L)])
        for s in np.arange(spacing / 2, total, spacing):
            i = min(int(np.searchsorted(cum, s) - 1), len(seg) - 1)
            t = (s - cum[i]) / max(L[i], 1e-9)
            c = pts[i] + seg[i] * t
            nrm = np.array([-seg[i][1], seg[i][0]]) / max(L[i], 1e-9)
            for side in (1, -1):
                out.append(c + side * nrm * off)
    return np.array(out).reshape(-1, 2)


def generate_vegetation(spec, constraints, roads, road_styles, bounds, existing_trees=(), seed=None):
    seed = spec.get("seed", 42) if seed is None else seed
    rng = np.random.default_rng(seed)
    pl = Placer(constraints, rng)
    stats = {}

    for x, z in existing_trees:                                   # cây có sẵn trong dữ liệu OSM
        pl.items.append((x, z, "tree", "osm"))
        pl.pos = np.vstack([pl.pos, [x, z]])
        pl.rad = np.append(pl.rad, RADIUS["tree"])
    stats["osm"] = len(existing_trees)

    for z in spec.get("tree_zones", []):
        poly = np.asarray(z["polygon"], float)
        area = abs(0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
        target = int(round(float(z.get("density", 3.0)) * area / 100.0))
        cands = _in_polygon_candidates(poly, max(target * 80, 400), rng)
        n = pl.add(cands, z.get("kinds", {"tree": 0.5, "small": 0.2, "shrub": 0.3}), "zone:" + z.get("name", "?"), target,
                   wall=z.get("wall_clearance", 0.8), road=z.get("road_clearance", 1.0), tour=z.get("tour_clearance", 1.8))
        stats["zone:" + z.get("name", "?")] = n

    street = spec.get("street_trees", {})
    if street.get("enabled", True):
        x0, z0, x1, z1 = bounds
        m = float(street.get("margin", 40.0))
        cands = road_side_points(roads, road_styles, float(street.get("spacing", 14.0)),
                                 (x0 - m, z0 - m, x1 + m, z1 + m), float(street.get("offset", 2.4)))
        cands = cands[rng.permutation(len(cands))] if len(cands) else cands
        stats["street"] = pl.add(cands, {"tree": 0.75, "small": 0.25}, "street", int(street.get("max", 120)),
                                 wall=4.0, road=1.2, tour=2.5)

    per = spec.get("perimeter", {})
    if per.get("enabled", True):
        x0, z0, x1, z1 = bounds
        m = float(per.get("margin", 90.0))
        count = int(per.get("count", 220))
        cands = rng.uniform((x0 - m, z0 - m), (x1 + m, z1 + m), size=(count * 40, 2))
        d = dist_points_segments(cands, constraints.all_walls)
        keep = rng.random(len(cands)) < np.exp(-np.maximum(d - 6.0, 0.0) / float(per.get("falloff", 60.0)))
        stats["perimeter"] = pl.add(cands[keep], per.get("kinds", {"tree": 0.7, "small": 0.3}), "perimeter", count,
                                    wall=float(per.get("wall_clearance", 5.0)), road=1.8, tour=3.0)

    return [(x, z, k) for x, z, k, _ in pl.items], stats
