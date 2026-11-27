import math
from collections import defaultdict

import numpy as np

from graphics.mesh import _BOX_FACES
from scene.geometry2d import point_in_edges, dist_to_edges
from scene.tour_route import build_graph

CAR_KINDS = {"service", "residential", "tertiary", "tertiary_link", "unclassified"}
PED_KINDS = {"footway", "path", "pedestrian", "cycleway", "track", "service"}
LANE_WIDTH = {"tertiary": 8.5, "tertiary_link": 5.5, "residential": 6.5, "unclassified": 5.5, "service": 5.5}

DEFAULTS = {"enabled": True, "cars": 14, "pedestrians": 36, "parked_cars": 24,
            "car_speed": 5.5, "ped_speed": 1.3, "seed": 2024}

# palette
PALETTE = [  # (r, g, b) 0-255, index = position
    (200, 40, 40), (40, 90, 190), (235, 235, 235), (30, 30, 35), (190, 190, 195), (230, 190, 40),
    (40, 140, 80), (150, 60, 160),                                    # 0-7  body colours
    (20, 20, 22), (150, 200, 230), (255, 240, 170), (210, 30, 30),    # 8 tyre 9 glass 10 lamp 11 tail
    (235, 190, 150), (120, 80, 55), (60, 60, 70), (50, 70, 120),      # 12 skin 13 hair 14 trousers 15 jeans
    (230, 230, 230), (210, 90, 70), (70, 130, 190), (240, 200, 70),   # 16-19 shirts
    (90, 160, 100), (180, 110, 190), (45, 45, 50), (110, 85, 60),     # 20-23
]
PALETTE_N = 8   # palette image is PALETTE_N x PALETTE_N cells
CELL = 4        # pixels per cell


def palette_image():
    img = np.zeros((PALETTE_N * CELL, PALETTE_N * CELL, 4), np.uint8)
    img[:] = (255, 0, 255, 255)
    for i, c in enumerate(PALETTE):
        r, q = divmod(i, PALETTE_N)
        img[r * CELL:(r + 1) * CELL, q * CELL:(q + 1) * CELL] = (*c, 255)
    return img


def _uv(i):
    r, q = divmod(i, PALETTE_N)
    return (q + 0.5) / PALETTE_N, (r + 0.5) / PALETTE_N


# meshes
def _part(cx, cy, cz, sx, sy, sz, colour):
    """One axis-aligned box (centre + size) painted with a single palette colour."""
    size = np.array([sx, sy, sz], np.float32)
    u, v = _uv(colour)
    corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    verts, idx = [], []
    for fi, (n, t, b) in enumerate(_BOX_FACES):
        n, t, b = (np.array(a, np.float32) for a in (n, t, b))
        for st, sb in corners:
            p = 0.5 * size * (n + st * t + sb * b) + (cx, cy, cz)
            verts.append([*p, *n, u, v])
        k = fi * 4
        idx += [k, k + 1, k + 2, k, k + 2, k + 3]
    return np.array(verts, np.float32), np.array(idx, np.uint32)


def _merge(parts):
    verts, idx, off = [], [], 0
    for v, i in parts:
        verts.append(v)
        idx.append(i + off)
        off += len(v)
    return np.vstack(verts), np.concatenate(idx)


def car_arrays(body=0):
    """Car facing +X (length 4.2, width 1.8, height 1.45)."""
    parts = [_part(0, 0.60, 0, 4.2, 0.62, 1.8, body),
             _part(-0.25, 1.17, 0, 2.2, 0.52, 1.6, 9),
             _part(-0.25, 1.45, 0, 2.0, 0.06, 1.5, body)]
    for sx in (-1.3, 1.3):
        for sz in (-0.82, 0.82):
            parts.append(_part(sx, 0.32, sz, 0.66, 0.64, 0.24, 8))
    for sz in (-0.65, 0.65):
        parts.append(_part(2.1, 0.68, sz, 0.06, 0.16, 0.30, 10))
        parts.append(_part(-2.1, 0.68, sz, 0.06, 0.16, 0.30, 11))
    return _merge(parts)


def person_arrays(shirt=16, trousers=14, hair=13):
    """Pedestrian facing +X (height ~1.75)."""
    parts = [_part(0, 0.40, -0.10, 0.18, 0.80, 0.16, trousers), _part(0, 0.40, 0.10, 0.18, 0.80, 0.16, trousers),
             _part(0, 1.10, 0, 0.26, 0.64, 0.42, shirt),
             _part(0, 1.08, -0.28, 0.14, 0.58, 0.12, shirt), _part(0, 1.08, 0.28, 0.14, 0.58, 0.12, shirt),
             _part(0, 1.55, 0, 0.24, 0.24, 0.24, 12), _part(-0.02, 1.70, 0, 0.27, 0.10, 0.27, hair)]
    return _merge(parts)


# road graph
def _segment_clear(a, b, blockers, step=1.5):
    """True if no sample along a-b is inside (or touching) any blocker footprint."""
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
    for t in np.linspace(0.0, 1.0, n):
        x, z = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        for e in blockers:
            if point_in_edges(x, z, e):
                return False
    return True


def _inside(p, bounds):
    return bounds is None or (bounds[0] <= p[0] <= bounds[2] and bounds[1] <= p[1] <= bounds[3])


def usable_graph(roads, kinds, blockers, bounds=None):
    """Road graph limited to ``kinds``; edges through buildings or outside ``bounds`` are removed."""
    adj = build_graph(roads, allowed=kinds)
    out = defaultdict(dict)
    for a, nb in adj.items():
        for b, w in nb.items():
            if a < b and _inside(a, bounds) and _inside(b, bounds) and _segment_clear(a, b, blockers):
                out[a][b] = w
                out[b][a] = w
    return out


# agents
class Agent:
    """Walks along graph edges on the right-hand lane. Pure data + update logic."""

    def __init__(self, kind, graph, rng, node, nxt, speed, lane, radius, length, top, colour_variant):
        self.kind, self.graph, self.rng = kind, graph, rng
        self.prev, self.node, self.nxt = node, node, nxt
        self.s = 0.0
        self.base_speed = speed * rng.uniform(0.85, 1.15)
        self.speed = 0.0
        self.lane, self.radius, self.length, self.top = lane, radius, length, top
        self.variant = colour_variant
        self.x = self.z = 0.0
        self.yaw = 0.0#
        self.phase = rng.uniform(0, 6.28)
        self.wait = 0.0
        self.ghost = 0.0
        self._edge_len = 1.0
        self._set_edge(node, nxt)
        self.x, self.z = self._target()
        self.yaw = self._heading

    # geometry
    def _set_edge(self, a, b):
        self.node, self.nxt = a, b
        dx, dz = b[0] - a[0], b[1] - a[1]
        self._edge_len = max(math.hypot(dx, dz), 1e-6)
        self._dir = (dx / self._edge_len, dz / self._edge_len)
        self._heading = math.atan2(dz, dx)

    def _target(self):
        dx, dz = self._dir
        rx, rz = -dz, dx          # right-hand side of travel (y up); positive lane = keep right
        px = self.node[0] + dx * self.s + rx * self.lane
        pz = self.node[1] + dz * self.s + rz * self.lane
        return px, pz

    def circles(self):
        """Collision circles (x, z, r): one for a person, two for a car."""
        if self.length <= 2 * self.radius + 0.1:
            return [(self.x, self.z, self.radius)]
        off = self.length / 2 - self.radius
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return [(self.x + c * off, self.z + s * off, self.radius),
                (self.x - c * off, self.z - s * off, self.radius)]

    def _choose_next(self):
        here, came = self.nxt, self.node
        options = [n for n in self.graph[here] if n != came]
        if not options:
            return came                      # dead end -> turn around
        hx, hz = self._dir
        w = []
        for n in options:
            d = math.hypot(n[0] - here[0], n[1] - here[1]) or 1.0
            w.append(math.exp(2.0 * ((n[0] - here[0]) * hx + (n[1] - here[1]) * hz) / d))
        w = np.array(w) / sum(w)
        return options[int(self.rng.choice(len(options), p=w))]

    def update(self, dt, others, player, blockers):
        """Advance one step. ``others`` = list of Agents, ``player`` = (x, z) or None."""
        blocked = self._blocked(others, player)
        if blocked and self.ghost <= 0.0:
            self.wait += dt
            if self.wait > 5.0:                  # deadlock breaker: ignore others for a moment
                self.ghost, self.wait = 3.0, 0.0
        else:
            self.wait = 0.0
        self.ghost = max(0.0, self.ghost - dt)
        want = 0.0 if (blocked and self.ghost <= 0.0) else self.base_speed
        self.speed += (want - self.speed) * min(1.0, (6.0 if want == 0.0 else 2.5) * dt)
        self.s += self.speed * dt
        while self.s >= self._edge_len:
            self.s -= self._edge_len
            a, b = self.nxt, self._choose_next()
            self._set_edge(a, b)
        tx, tz = self._target()
        if any(point_in_edges(tx, tz, e) for e in blockers):    # never step into a building
            a, b = self.nxt, self.node
            self.s = max(0.0, self._edge_len - self.s)
            self._set_edge(a, b)
            tx, tz = self._target()
        k = min(1.0, 8.0 * dt)
        self.x += (tx - self.x) * k
        self.z += (tz - self.z) * k
        d = (self._heading - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.yaw += d * min(1.0, 6.0 * dt)
        self.phase += self.speed * dt * (4.5 if self.kind == "ped" else 0.0)

    def _blocked(self, others, player):
        gap = 5.0 + self.speed * 0.8 if self.kind == "car" else 1.8
        half_w = 1.25 if self.kind == "car" else 0.55
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        pts = [(player[0], player[1], 0.35)] if player is not None else []
        for o in others:
            if o is not self:
                pts.extend(o.circles())
        for (px, pz, r) in pts:
            dx, dz = px - self.x, pz - self.z
            fwd, lat = dx * c + dz * s, -dx * s + dz * c
            if 0.0 < fwd < gap + r + self.length / 2 and abs(lat) < half_w + r:
                return True
        return False

    def pose(self):
        """(x, y, z, yaw_degrees, roll_degrees) for rendering."""
        if self.kind == "ped":
            return (self.x, abs(math.sin(self.phase)) * 0.05, self.z, -math.degrees(self.yaw),
                    math.sin(self.phase) * 4.0)
        return self.x, 0.04, self.z, -math.degrees(self.yaw), 0.0


class TrafficSystem:
    def __init__(self, agents, blockers):
        self.agents = agents
        self.blockers = blockers

    def update(self, dt, player_xz=None):
        dt = min(dt, 0.1)
        for a in self.agents:
            a.update(dt, self.agents, player_xz, self.blockers)

    def circles(self):
        """Array (n, 4): x, z, radius, top_height for every collision circle."""
        rows = [(x, z, r, a.top) for a in self.agents for (x, z, r) in a.circles()]
        return np.array(rows, np.float32).reshape(-1, 4)


def _spawn(kind, graph, n, rng, min_gap, avoid, speed, lane_fn, radius, length, top, variants, spawn_clear=10.0):
    edges = [(a, b) for a, nb in graph.items() for b in nb if a < b and graph[a][b] >= 8.0]
    agents, tries = [], 0
    while len(agents) < n and edges and tries < n * 40:
        tries += 1
        a, b = edges[int(rng.integers(len(edges)))]
        if rng.random() < 0.5:
            a, b = b, a
        ag = Agent(kind, graph, rng, a, b, speed, lane_fn(a, b), radius, length, top, int(rng.integers(variants)))
        ag.s = rng.uniform(1.0, ag._edge_len - 1.0)
        ag.x, ag.z = ag._target()
        ag.yaw = ag._heading
        if any(math.hypot(ag.x - ax, ag.z - az) < spawn_clear for ax, az in avoid):
            continue
        if any(math.hypot(ag.x - o.x, ag.z - o.z) < min_gap for o in agents):
            continue
        agents.append(ag)
    return agents


def build_traffic(roads, blockers, cfg=None, avoid=(), seed=2024, bounds=None):
    """Create the car and pedestrian agents (no GL needed)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    rng = np.random.default_rng(cfg.get("seed", seed))
    car_graph = usable_graph(roads, CAR_KINDS, blockers, bounds)
    kind_of = {}
    for r in roads:
        if r.get("kind") in CAR_KINDS:
            for p, q in zip(r["points"], r["points"][1:]):
                k1, k2 = (round(p[0], 2), round(p[1], 2)), (round(q[0], 2), round(q[1], 2))
                kind_of[(k1, k2)] = kind_of[(k2, k1)] = r["kind"]

    def car_lane(a, b):
        return min(1.6, LANE_WIDTH.get(kind_of.get((a, b), "service"), 5.5) * 0.25)

    ped_graph = usable_graph(roads, PED_KINDS, blockers, bounds)
    cars = _spawn("car", car_graph, int(cfg["cars"]), rng, 14.0, avoid, cfg["car_speed"], car_lane,
                  1.0, 4.2, 1.5, 8)
    peds = _spawn("ped", ped_graph, int(cfg["pedestrians"]), rng, 2.5, avoid, cfg["ped_speed"],
                  lambda a, b: 0.5, 0.3, 0.0, 1.8, 6)
    return TrafficSystem(cars + peds, blockers)


# parked cars
def roof_columns(polygon, spacing):
    """Column centres of a roofed car park (same layout rule as the roof builder)."""
    poly = np.asarray(polygon, float)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    nx = max(2, int(np.ceil((hi[0] - lo[0] - 1.6) / spacing)) + 1)
    nz = max(2, int(np.ceil((hi[1] - lo[1] - 1.6) / spacing)) + 1)
    return [(cx, cz) for cx in np.linspace(lo[0] + 0.8, hi[0] - 0.8, nx)
            for cz in np.linspace(lo[1] + 0.8, hi[1] - 0.8, nz)]


def parked_car_spots(polygon, gates, rng, limit, columns=()):
    """Grid of parking spots inside a polygon, away from the border, gates and roof columns."""
    from scene.geometry2d import rings_to_edges
    poly = np.asarray(polygon, float)
    edges = rings_to_edges([poly])
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    gate_pts = [np.array(g["at"], float) for g in (gates or [])]
    spots = []
    for x in np.arange(lo[0] + 3.0, hi[0] - 2.0, 3.2):
        for z in np.arange(lo[1] + 4.0, hi[1] - 3.0, 6.5):
            if not point_in_edges(x, z, edges) or dist_to_edges(x, z, edges) < 3.2:
                continue
            if any(np.hypot(x - g[0], z - g[1]) < 6.0 for g in gate_pts):
                continue
            if any(abs(cx - x) < 1.5 and abs(cz - z) < 2.8 for cx, cz in columns):
                continue
            if rng.random() < 0.35:
                continue
            spots.append((float(x), float(z)))
    rng.shuffle(spots)
    return spots[:limit]


# scene hookup (needs GL)
def populate_traffic(scene, data, landscape, obstacle_edges, spawn_xz=None):
    """Add parked cars + moving cars/pedestrians to ``scene``. Never raises into the caller."""
    from graphics.mesh import Mesh
    from graphics.texture import Texture
    from scene.scene import SceneObject, DynamicObject

    cfg = {**DEFAULTS, **(landscape.get("traffic") or {})}
    if not cfg["enabled"]:
        return None
    palette = Texture(palette_image())
    rng = np.random.default_rng(cfg["seed"])
    mesh_cache = {}

    def car_mesh(i):
        if ("c", i) not in mesh_cache:
            mesh_cache[("c", i)] = Mesh(*car_arrays(i))
        return mesh_cache[("c", i)]

    def ped_mesh(i):
        if ("p", i) not in mesh_cache:
            shirts = (16, 17, 18, 19, 20, 21)
            trousers = (14, 15, 22, 23, 14, 15)
            mesh_cache[("p", i)] = Mesh(*person_arrays(shirts[i % 6], trousers[(i * 5 + 1) % 6], 13 if i % 2 else 22))
        return mesh_cache[("p", i)]

    # parked cars: ordinary static, collidable objects inside the user-defined parking lots
    n_parked = 0
    budget = int(cfg["parked_cars"])
    for p in landscape.get("parking", []):
        if budget <= 0:
            continue
        cols = roof_columns(p["polygon"], float(p["roof"].get("column_spacing", 8.0))) if p.get("roof") else ()
        for (x, z) in parked_car_spots(p["polygon"], p.get("gates"), rng, budget, cols):
            o = SceneObject(car_mesh(int(rng.integers(8))), palette, position=(x, 0.04, z), rotation_y=90.0)
            o.cast_shadow, o.collidable, o.specular, o.shininess = True, True, 0.35, 48.0
            scene.add(o)
            n_parked += 1
            budget -= 1

    pts = np.array([q for b in data["buildings"] for q in b["outer"]], float)
    margin = float(cfg.get("area_margin", 120.0))
    bounds = (pts[:, 0].min() - margin, pts[:, 1].min() - margin, pts[:, 0].max() + margin, pts[:, 1].max() + margin) \
        if len(pts) else None
    traffic = build_traffic(data.get("roads", []), obstacle_edges, cfg,
                            avoid=[spawn_xz] if spawn_xz else (), seed=cfg["seed"], bounds=bounds)
    for a in traffic.agents:
        mesh = car_mesh(a.variant) if a.kind == "car" else ped_mesh(a.variant)
        a.obj = DynamicObject(mesh, palette, specular=0.3 if a.kind == "car" else 0.05)
        a.obj.set_pose(*a.pose())
        scene.add_dynamic(a.obj)
    scene.traffic = traffic
    n_c = sum(a.kind == "car" for a in traffic.agents)
    print(f"[Traffic] {n_c} xe chạy, {len(traffic.agents) - n_c} người đi bộ, {n_parked} xe đỗ")
    return traffic
