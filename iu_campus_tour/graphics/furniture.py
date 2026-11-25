import math

import numpy as np

from graphics.fence import oriented_box
from graphics.roads import merge_meshes

BENCH_LENGTH = 1.9
BENCH_DEPTH = 0.55


def bench_parts(cx, cz, ang, back_sign=1, length=BENCH_LENGTH, depth=BENCH_DEPTH):
    """Build the mesh parts for a bench."""
    al = np.array([math.cos(ang), math.sin(ang)])
    ac = np.array([-math.sin(ang), math.cos(ang)])
    seat_d = depth * 0.8
    parts = [oriented_box(cx, cz, 0.40, 0.48, length, seat_d, ang)]
    for s in (-1, 1):
        c = np.array([cx, cz]) + al * s * (length / 2 - 0.22)
        parts.append(oriented_box(c[0], c[1], 0.0, 0.40, 0.24, seat_d, ang))
    b = np.array([cx, cz]) + ac * back_sign * (seat_d / 2 - 0.03)
    parts.append(oriented_box(b[0], b[1], 0.48, 0.95, length, 0.10, ang))
    return parts


def bench_box(cx, cz, ang, length=BENCH_LENGTH, depth=BENCH_DEPTH):
    """Return a collision box for a bench."""
    c, s = abs(math.cos(ang)), abs(math.sin(ang))
    hx = (length * c + depth * s) / 2
    hz = (length * s + depth * c) / 2
    return [cx - hx, 0.0, cz - hz, cx + hx, 0.95, cz + hz]


def place_benches(polygon, count, min_spacing, avoid, rng, edge_offset=0.6):
    """Choose randomized bench placements along a rectangular walkway."""
    poly = np.asarray(polygon, float)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    along_x = (hi[0] - lo[0]) >= (hi[1] - lo[1])
    ang = 0.0 if along_x else math.pi / 2
    ac = np.array([-math.sin(ang), math.cos(ang)])
    chosen = []
    for _ in range(4000):
        if len(chosen) >= count:
            break
        side = int(rng.integers(0, 2))
        if along_x:
            p = np.array([rng.uniform(lo[0] + 3.0, hi[0] - 3.0), lo[1] + edge_offset if side == 0 else hi[1] - edge_offset])
        else:
            p = np.array([lo[0] + edge_offset if side == 0 else hi[0] - edge_offset, rng.uniform(lo[1] + 3.0, hi[1] - 3.0)])
        if any(np.hypot(*(p - np.array([ax, az]))) < r for ax, az, r in avoid):
            continue
        if any(np.hypot(*(p - np.array([q[0], q[1]]))) < min_spacing for q in chosen):
            continue
        # Orient the bench back toward the nearest walkway edge.
        center = (lo + hi) / 2
        out_vec = p - center
        back_sign = 1 if float(out_vec @ ac) >= 0 else -1
        chosen.append((float(p[0]), float(p[1]), ang, back_sign))
    return chosen


def benches_mesh(placements):
    parts = [pt for cx, cz, ang, bs in placements for pt in bench_parts(cx, cz, ang, bs)]
    boxes = np.array([bench_box(cx, cz, ang) for cx, cz, ang, _ in placements], np.float32).reshape(-1, 6)
    return merge_meshes(parts), boxes


def pergola_parts(cx, cz, ang, width=3.0, depth=2.2, height=2.8, post=0.16):
    """White garden arch over a walkway: 4 posts, 2 cross beams and roof slats. ang = walkway direction."""
    al = np.array([math.cos(ang), math.sin(ang)])
    ac = np.array([-math.sin(ang), math.cos(ang)])
    c0 = np.array([cx, cz])
    parts, boxes = [], []
    for sa in (-1, 1):
        for sc in (-1, 1):
            p = c0 + al * sa * depth / 2 + ac * sc * width / 2
            parts.append(oriented_box(p[0], p[1], 0.0, height, post, post, ang))
            boxes.append([p[0] - post, 0.0, p[1] - post, p[0] + post, height, p[1] + post])
        b = c0 + al * sa * depth / 2
        parts.append(oriented_box(b[0], b[1], height, height + 0.16, 0.12, width + 0.5, ang))
    for k in range(6):
        s = c0 + ac * (-width / 2 - 0.2 + k * (width + 0.4) / 5)
        parts.append(oriented_box(s[0], s[1], height + 0.16, height + 0.24, depth + 0.5, 0.07, ang))
    return parts, boxes


def pergolas_mesh(placements, **kw):
    parts, boxes = [], []
    for cx, cz, ang in placements:
        p, b = pergola_parts(cx, cz, ang, **kw)
        parts += p
        boxes += b
    return merge_meshes(parts), np.array(boxes, np.float32).reshape(-1, 6)
