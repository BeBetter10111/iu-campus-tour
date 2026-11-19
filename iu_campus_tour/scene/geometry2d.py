import numpy as np

def rings_to_edges(rings):
    edges = []
    for ring in rings:
        r = np.asarray(ring, dtype=np.float64)
        if len(r) >= 2:
            edges.append(np.hstack([r, np.roll(r, -1, axis=0)]))
    return np.vstack(edges) if edges else np.zeros((0, 4))


def point_in_edges(px, pz, edges):
    if len(edges) == 0:
        return False
    x0, z0, x1, z1 = edges.T
    crosses = (z0 > pz) != (z1 > pz)                       # cạnh bắc ngang độ cao pz
    dz = np.where(z1 != z0, z1 - z0, 1.0)
    x_at = x0 + (pz - z0) * (x1 - x0) / dz                 # hoành độ giao điểm
    return bool(np.count_nonzero(crosses & (px < x_at)) % 2)


def dist_to_edges(px, pz, edges):
    if len(edges) == 0:
        return np.inf
    x0, z0, x1, z1 = edges.T
    abx, abz = x1 - x0, z1 - z0
    denom = np.maximum(abx * abx + abz * abz, 1e-12)
    t = np.clip(((px - x0) * abx + (pz - z0) * abz) / denom, 0.0, 1.0)
    cx, cz = x0 + t * abx, z0 + t * abz
    return float(np.sqrt(np.min((px - cx) ** 2 + (pz - cz) ** 2)))


def circle_hits_rings(px, pz, radius, edges):
    return point_in_edges(px, pz, edges) or dist_to_edges(px, pz, edges) < radius


def nearest_on_edges(px, pz, edges):
    if len(edges) == 0:
        return np.inf, px, pz
    x0, z0, x1, z1 = edges.T
    abx, abz = x1 - x0, z1 - z0
    denom = np.maximum(abx * abx + abz * abz, 1e-12)
    t = np.clip(((px - x0) * abx + (pz - z0) * abz) / denom, 0.0, 1.0)
    cx, cz = x0 + t * abx, z0 + t * abz
    d2 = (px - cx) ** 2 + (pz - cz) ** 2
    k = int(np.argmin(d2))
    return float(np.sqrt(d2[k])), float(cx[k]), float(cz[k])
