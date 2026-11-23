import heapq
import json
import math
from collections import defaultdict

import numpy as np

from scene.geometry2d import nearest_on_edges, point_in_edges

ALLOWED_KINDS = {"service", "footway", "path", "pedestrian", "residential", "cycleway", "unclassified", "track"}
KEY_DIGITS = 2


def _key(p):
    return (round(float(p[0]), KEY_DIGITS), round(float(p[1]), KEY_DIGITS))


def build_graph(roads, allowed=ALLOWED_KINDS):
    """Build a road graph with edge lengths as weights."""
    adj = defaultdict(dict)
    for r in roads:
        if r.get("kind") not in allowed:
            continue
        keys = [_key(p) for p in r["points"]]
        for a, b in zip(keys, keys[1:]):
            if a == b:
                continue
            w = math.hypot(a[0] - b[0], a[1] - b[1])
            adj[a][b] = w
            adj[b][a] = w
    return adj


def snap_to_graph(adj, xz):
    px, pz = float(xz[0]), float(xz[1])
    best = (np.inf, None, None, None)
    for a, nbrs in adj.items():
        for b in nbrs:
            if a < b:
                abx, abz = b[0] - a[0], b[1] - a[1]
                t = ((px - a[0]) * abx + (pz - a[1]) * abz) / max(abx * abx + abz * abz, 1e-12)
                t = min(1.0, max(0.0, t))
                cx, cz = a[0] + t * abx, a[1] + t * abz
                d = math.hypot(px - cx, pz - cz)
                if d < best[0]:
                    best = (d, a, b, (cx, cz))
    d, a, b, c = best
    if a is None:
        raise ValueError("The road graph is empty")
    k = _key(c)
    if k not in adj:
        del adj[a][b]
        del adj[b][a]
        for n in (a, b):
            w = math.hypot(n[0] - k[0], n[1] - k[1])
            adj[k][n] = w
            adj[n][k] = w
    return k, d


def shortest_path(adj, src, dst):
    dist, prev = {src: 0.0}, {}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            break
        if d > dist.get(u, np.inf):
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist.get(v, np.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, v))
    if dst not in dist:
        raise ValueError(f"No road connects {src} to {dst}")
    path, u = [dst], dst
    while u != src:
        u = prev[u]
        path.append(u)
    return path[::-1]


# Path smoothing, resampling
def resample(points, spacing, closed=True):
    P = np.asarray(points, dtype=np.float64)
    if closed:
        P = np.vstack([P, P[:1]])
    seg = np.hypot(*(P[1:] - P[:-1]).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(int(round(total / spacing)), 3)
    s = np.linspace(0, total, n, endpoint=not closed)
    return np.stack([np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])], axis=1)


def smooth(points, window, spacing, closed=True):
    k = max(int(round(window / spacing)) // 2, 1)
    P = np.asarray(points, dtype=np.float64)
    out = np.zeros_like(P)
    if closed:
        for o in range(-k, k + 1):
            out += np.roll(P, o, axis=0)
        return out / (2 * k + 1)
    padded = np.vstack([np.repeat(P[:1], k, 0), P, np.repeat(P[-1:], k, 0)])
    for o in range(2 * k + 1):
        out += padded[o:o + len(P)]
    return out / (2 * k + 1)


def _push_once(P, obstacles, clearance):
    moved = False
    for i in range(len(P)):
        x, z = P[i]
        for edges in obstacles:
            d, cx, cz = nearest_on_edges(x, z, edges)
            inside = point_in_edges(x, z, edges)
            if inside or d < clearance:
                vx, vz = (cx - x, cz - z) if inside else (x - cx, z - cz)
                n = math.hypot(vx, vz)
                if n < 1e-9:
                    vx, vz, n = 1.0, 0.0, 1.0
                x, z = cx + vx / n * clearance, cz + vz / n * clearance
                moved = True
        P[i] = (x, z)
    return moved


def push_from_obstacles(points, obstacles, clearance, passes=4, spacing=1.0):
    P = np.array(points, dtype=np.float64)
    for _ in range(passes):
        if not _push_once(P, obstacles, clearance):
            return P
        P = smooth(P, 4.0, spacing)
    _push_once(P, obstacles, clearance)
    return P


def min_clearance(points, obstacles):
    best = np.inf
    for x, z in points:
        for edges in obstacles:
            d, _, _ = nearest_on_edges(x, z, edges)
            best = min(best, -d if point_in_edges(x, z, edges) else d)
    return best


def load_tour_spec(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tour(roads, spec, obstacles, eye_height=1.7, waypoint_spacing=7.0):
    warnings = []
    adj = build_graph(roads)
    stops = spec["stops"]
    snaps = {}
    for i, st in enumerate(stops):
        if "snap" in st:
            k, d = snap_to_graph(adj, st["snap"])
            snaps[i] = k
            if d > 15.0:
                warnings.append(f"điểm dừng '{st.get('name', i)}' cách đường gần nhất {d:.0f} m (kiểm tra tọa độ)")

    route, marks = [], []
    prev_node = None
    for i, st in enumerate(stops):
        if "snap" in st:
            node = snaps[i]
            if prev_node is not None:
                route += [np.array(p) for p in shortest_path(adj, prev_node, node)[1:]]
            else:
                route.append(np.array(node))
            marks.append(len(route) - 1)
            prev_node = node
        else:
            marks.append(len(route))
            route += [np.array(p, dtype=np.float64) for p in st["free"]]
            prev_node = None
    last_i = len(stops) - 1
    if "snap" in stops[0] and "snap" in stops[last_i] and snaps[0] != snaps[last_i]:
        route += [np.array(p) for p in shortest_path(adj, snaps[last_i], snaps[0])[1:-1]]

    R = np.array(route, dtype=np.float64)
    keep = [0] + [i for i in range(1, len(R)) if np.hypot(*(R[i] - R[i - 1])) > 1e-6]
    R = R[keep]

    spacing = 1.0
    dense = resample(R, spacing, closed=True)
    dense = smooth(dense, spec.get("corner_smoothing", 10.0), spacing, closed=True)
    clearance = float(spec.get("clearance", 2.0))
    dense = push_from_obstacles(dense, obstacles, clearance, spacing=spacing)
    cl = min_clearance(dense[::2], obstacles)

    pts = resample(dense, waypoint_spacing, closed=True)
    y = float(spec.get("eye_height", eye_height))
    waypoints = [(float(x), y, float(z)) for x, z in pts]
    total = float(np.sum(np.hypot(*(np.roll(dense, -1, 0) - dense).T)))

    Rf = np.array(route, dtype=np.float64)
    cum_route = np.concatenate([[0.0], np.cumsum(np.hypot(*(Rf[1:] - Rf[:-1]).T))])
    total_route = max(cum_route[-1] + float(np.hypot(*(Rf[0] - Rf[-1]))), 1e-9)
    stop_progress = [(st.get("name", f"Điểm {i + 1}"), float(cum_route[min(marks[i], len(Rf) - 1)] / total_route))
                     for i, st in enumerate(stops)]
    if cl < clearance - 0.25:
        warnings.append(f"tuyến còn cách tường {cl:.2f} m (< {clearance} m mong muốn); kiểm tra lại điểm dừng")
    return {"waypoints": waypoints, "length": total, "clearance": cl, "stops": stop_progress,
            "warnings": warnings, "dense": dense}
