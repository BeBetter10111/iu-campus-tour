import json
import math
import numpy as np
import config
from graphics.extrude import extrude_arrays, signed_area
from graphics.mesh import Mesh, create_box
from graphics.procedural_textures import (facade_texture, ground_texture as make_ground_texture, FLOOR_H, TILE_W,
                                          GRASS_TILE, ASPHALT_TILE, MARKED_TILE, PAVING_TILE, CONCRETE_TILE)
from graphics.roads import polyline_strip, polygon_flat, merge_meshes
from graphics.vegetation import build_plants
from graphics.fence import fence_for_polygon, oriented_box
from graphics.stairs import Stair, stairs_mesh
from graphics.furniture import place_benches, benches_mesh, pergolas_mesh
from scene.geometry2d import rings_to_edges, circle_hits_rings, point_in_edges
from scene.scene import Scene, SceneObject
DATA_DIR = getattr(config, "DATA_DIR", config.ASSET_DIR / "data")
ROAD_STYLE = {
    "tertiary":      ("asphalt_marked", 8.5, 0.030, MARKED_TILE, True),
    "tertiary_link": ("asphalt_marked", 5.5, 0.032, MARKED_TILE, True),
    "residential":   ("asphalt",        6.5, 0.034, ASPHALT_TILE, False),
    "unclassified":  ("asphalt",        5.5, 0.036, ASPHALT_TILE, False),
    "service":       ("asphalt",        5.5, 0.040, ASPHALT_TILE, False),
    "track":         ("concrete",       3.5, 0.045, CONCRETE_TILE, False),
    "cycleway":      ("asphalt",        3.0, 0.050, ASPHALT_TILE, False),
    "pedestrian":    ("paving",         4.5, 0.055, PAVING_TILE, False),
    "footway":       ("paving",         2.8, 0.060, PAVING_TILE, False),
    "path":          ("paving",         2.2, 0.062, PAVING_TILE, False),
}
PARKING_Y = 0.040
PAVED_Y = 0.064
ROAD_MARGIN = 300.0
RAISED_MIN = 2.5
ROOF_ONLY_KINDS = {"roof", "carport", "canopy"}
SPAWN_CLEARANCE = 6.0
def _norm(s):
    return " ".join(str(s).casefold().split())
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def load_overrides(path):
    """Load and normalize building overrides."""
    if not path.exists():
        return {"by_name": {}, "by_id": {}, "connect_doors": {}, "extra_buildings": []}
    raw = load_json(path)
    connect_doors = {}
    for c in raw.get("connect", []):
        door = {"at": c["at"], "width": c.get("width", 4.0), "inner": True}
        for bid in c["between"]:
            connect_doors.setdefault(bid, []).append(door)
    return {"by_name": {_norm(k): v for k, v in raw.get("by_name", {}).items()},
            "by_id": raw.get("by_id", {}),
            "connect_doors": connect_doors,
            "extra_buildings": raw.get("extra_buildings", [])}
def _polygon_area(ring):
    return abs(signed_area(ring))
def resolve_style(b, overrides):
    """Resolve a building style from overrides and OSM tags."""
    kind = (b.get("kind") or "yes").lower()
    name = b.get("name") or ""
    ov = {}
    ov.update(overrides["by_name"].get(_norm(name), {}))
    if b.get("name_en"):
        ov = {**overrides["by_name"].get(_norm(b["name_en"]), {}), **ov}
    by_name_ov = dict(ov)
    ov.update(overrides["by_id"].get(b.get("id", ""), {}))
    doors = list(by_name_ov.get("doors", [])) + list(overrides["by_id"].get(b.get("id", ""), {}).get("doors", [])) \
        + list(overrides.get("connect_doors", {}).get(b.get("id", ""), []))
    roof_only = ov.get("roof_only", kind in ROOF_ONLY_KINDS)
    under_construction = ("xây dựng" in name.casefold()) or ("construction" in name.casefold()) \
        or kind == "construction"
    levels = ov.get("levels") or b.get("levels")
    height = ov.get("height") or b.get("height")
    if height is None:
        if roof_only:
            height = 3.5
        else:
            if levels is None:
                levels = 3 if under_construction else (1 if _polygon_area(b["outer"]) < 40 else 2)
            height = float(levels) * FLOOR_H
    levels = levels if levels is not None else max(1, round(height / FLOOR_H))
    facade = ov.get("facade") or ("concrete" if under_construction else "plain")
    base = float(ov["min_height"]) if ov.get("min_height") else float(ov.get("min_levels") or 0) * FLOOR_H
    base = min(base, float(height) - 0.5) if base else 0.0
    return {
        "levels": levels, "height": float(height), "base": base, "facade": facade,
        "facade_by_side": ov.get("facade_by_side"),
        "roof_only": bool(roof_only),
        "lift": float(ov.get("lift", 0.0)),
        "plinth": ov.get("plinth"),
        "ground_facade": ov.get("ground_facade"),
        "enterable": bool(ov.get("enterable", False)),
        "doors": doors,
        "collidable": bool(ov.get("collidable", not roof_only)),
        "skip": bool(ov.get("skip", False)),
    }
def _make_object(mesh, texture, position=(0, 0, 0), cast_shadow=True, collidable=False,
                 specular=0.10, shininess=32.0):
    """Create a scene object and apply compatibility properties."""
    obj = SceneObject(mesh, texture, position=position)
    obj.cast_shadow, obj.collidable = cast_shadow, collidable
    obj.specular, obj.shininess = specular, shininess
    return obj
def find_spawn(data, collider_edges):
    """Find a safe spawn point south of the building group."""
    pts = np.array([p for b in data["buildings"] for p in b["outer"]], dtype=np.float64)
    if len(pts) == 0:
        return (0.0, 15.0), -90.0
    cx, cz = (pts.min(axis=0) + pts.max(axis=0)) / 2
    start = np.array([cx, pts[:, 1].max() + 10.0])
    candidates = [start]
    for r in np.arange(6.0, 400.0, 6.0):
        for a in np.radians(np.arange(0, 360, 20)):
            candidates.append(start + r * np.array([math.cos(a), math.sin(a)]))
    for c in candidates:
        if not any(circle_hits_rings(c[0], c[1], SPAWN_CLEARANCE, e) for e in collider_edges):
            yaw = math.degrees(math.atan2(cz - c[1], cx - c[0]))
            return (float(c[0]), float(c[1])), yaw
    return (float(start[0]), float(start[1])), -90.0
def effective_road_styles(landscape):
    """Apply road-width overrides to the default road styles."""
    styles = dict(ROAD_STYLE)
    for kind, w in (landscape or {}).get("road_widths", {}).items():
        if kind in styles:
            s = list(styles[kind])
            s[1] = float(w)
            styles[kind] = tuple(s)
    return styles
def road_segments(roads, styles, bounds):
    """Return road segments within the scene bounds."""
    x0, z0, x1, z1 = bounds
    rows = []
    for r in roads:
        st = styles.get(r.get("kind"))
        if st is None:
            continue
        pts = r["points"]
        if not any(x0 <= p[0] <= x1 and z0 <= p[1] <= z1 for p in pts):
            continue
        rows += [[a[0], a[1], b[0], b[1], st[1] / 2] for a, b in zip(pts, pts[1:])]
    return np.array(rows, dtype=np.float64).reshape(-1, 5)
def add_ground_layers(scene, data, bounds, styles=None, extra_concrete=(), extra_paving=()):
    """Add roads, walkways, and parking areas to the ground scene."""
    styles = styles or ROAD_STYLE
    x0, z0, x1, z1 = bounds
    inside = lambda pts: any(x0 <= p[0] <= x1 and z0 <= p[1] <= z1 for p in pts)
    groups, count = {}, 0
    for k, r in enumerate(data.get("roads", [])):
        style = styles.get(r.get("kind"))
        if style is None or not inside(r["points"]):
            continue
        tex_kind, width, y, tile, full = style
        pts = r["points"]
        closed = len(pts) > 3 and pts[0] == pts[-1]
        part = polyline_strip(pts, width, y + 0.0015 * (k % 6), tile, full_width=full, closed=closed)
        if part is not None:
            groups.setdefault(tex_kind, []).append(part)
            count += 1
    rings = [pk["ring"] for pk in data.get("parking", []) if inside(pk["ring"])] + list(extra_concrete)
    for ring in rings:
        part = polygon_flat(ring, PARKING_Y, CONCRETE_TILE)
        if part is not None:
            groups.setdefault("concrete", []).append(part)
            count += 1
    for ring in extra_paving:
        part = polygon_flat(ring, PAVED_Y, PAVING_TILE)
        if part is not None:
            groups.setdefault("paving", []).append(part)
            count += 1
    for tex_kind, parts in groups.items():
        v, i = merge_meshes(parts)
        scene.add(_make_object(Mesh(v, i), make_ground_texture(tex_kind), cast_shadow=False,
                               collidable=False, specular=0.02, shininess=8.0))
    return count, len(groups)
def ray_road_hit(origin, n, segs, max_len):
    """Measure the distance from a doorway toward the nearest road."""
    best = None
    for x0, z0, x1, z1, hw in segs:
        e = np.array([x1 - x0, z1 - z0])
        L = float(np.hypot(*e))
        if L < 1e-6:
            continue
        den = n[0] * e[1] - n[1] * e[0]
        if abs(den) < 1e-9:
            continue
        w = np.array([x0 - origin[0], z0 - origin[1]])
        t = (w[0] * e[1] - w[1] * e[0]) / den
        u = (w[0] * n[1] - w[1] * n[0]) / den
        if t > 0 and 0.0 <= u <= 1.0 and t <= max_len:
            en = e / L
            cosang = abs(float(n @ np.array([-en[1], en[0]])))
            te = t - hw / max(cosang, 0.3)
            if best is None or te < best:
                best = te
    return best
def door_frame(door, door_h=2.7, jamb=0.12, depth=0.18, lift=0.0):
    """Build a doorway frame with an outward normal and wall direction."""
    cx, cz = door["at"]
    nx, nz = door["normal"]
    d = np.array([nz, -nx])
    ang = float(np.arctan2(d[1], d[0]))
    w = door["width"]
    parts = []
    for s in (-1, 1):
        c = np.array([cx, cz]) + d * s * (w / 2 + jamb / 2)
        parts.append(oriented_box(c[0], c[1], lift, lift + door_h, jamb, depth, ang))
    parts.append(oriented_box(cx, cz, lift + door_h, lift + door_h + 0.14, w + 2 * jamb, depth, ang))
    return parts
def _manual_building(i, b):
    """Convert manually defined buildings to the OSM building format."""
    return {"id": b.get("id", f"manual/{i}"), "name": b.get("name", ""), "name_en": "", "kind": b.get("kind", "yes"),
            "levels": None, "height": None, "holes": b.get("holes", []), "outer": b["outer"]}
def build_campus_from_osm(data, overrides=None, ground_texture=None, tour_spec=None, landscape=None):
    overrides = overrides or {"by_name": {}, "by_id": {}, "connect_doors": {}, "extra_buildings": []}
    landscape = landscape or {}
    styles = effective_road_styles(landscape)
    scene = Scene()
    all_buildings = list(data["buildings"]) + [_manual_building(i, b) for i, b in enumerate(overrides.get("extra_buildings", []))]
    active = [b for b in all_buildings if not resolve_style(b, overrides)["skip"]]
    extent = max([abs(c) for b in active for p in b["outer"] for c in p] + [100.0])
    G = 2.0 * (extent + 250.0)
    scene.add(_make_object(create_box((G, 0.2, G), tex_size=GRASS_TILE),
                           ground_texture if ground_texture is not None else make_ground_texture("grass"),
                           position=(0, -0.1, 0), cast_shadow=False, collidable=False,
                           specular=0.02, shininess=8.0))
    roof_tex = facade_texture("roof")
    collider_edges, report, n_walls = [], [], 0
    obstacle_edges, footprints = [], []
    door_points = []
    frame_parts = []
    stairs, door_stairs = [], []
    interiors = []
    for b in sorted(active, key=lambda b: -_polygon_area(b["outer"])):
        st = resolve_style(b, overrides)
        try:
            geo = extrude_arrays(b["outer"], b.get("holes"), st["height"], TILE_W, FLOOR_H,
                                 kind_of_side=st["facade_by_side"], default_kind=st["facade"],
                                 roof_only=st["roof_only"], base=st["base"],
                                 ground_kind=st["ground_facade"], doors=st["doors"], enterable=st["enterable"],
                                 lift=st["lift"], plinth=st["plinth"])
        except ValueError as e:
            report.append(f"  bỏ qua {b.get('id')}: {e}")
            continue
        label = b.get("name") or b.get("id")
        rings = [b["outer"]] + [h for h in (b.get("holes") or [])]
        edges = rings_to_edges(rings)
        footprints.append(edges)
        if st["base"] < RAISED_MIN and not st["roof_only"]:
            obstacle_edges.append(edges)
        xs, zs = np.array(b["outer"])[:, 0], np.array(b["outer"])[:, 1]
        primary = True
        for kind, (v, i) in geo["walls"].items():
            obj = _make_object(Mesh(v, i), facade_texture(kind), collidable=st["collidable"] and primary,
                               specular=0.12)
            if obj.collidable:
                obj.aabb_min = np.array([xs.min(), st["base"] + (st["lift"] if st["base"] else 0.0), zs.min()], dtype=np.float32)
                obj.aabb_max = np.array([xs.max(), st["height"] + st["lift"], zs.max()], dtype=np.float32)
                if st["enterable"] and geo["wall_edges"] is not None:
                    obj.wall_edges = geo["wall_edges"]
                else:
                    obj.footprint_edges = edges
                collider_edges.append(edges)
            scene.add(obj)
            primary = False
            n_walls += 1
        rv, ri = geo["roof"]
        scene.add(_make_object(Mesh(rv, ri), roof_tex, specular=0.05))
        if geo["bottom"] is not None:
            bv, bi = geo["bottom"]
            scene.add(_make_object(Mesh(bv, bi), roof_tex, specular=0.02))
        if st["enterable"] and st["lift"] > 0:
            interiors.append((edges, st["lift"]))
        if geo["interior"] is not None:
            for part, tex_kind in (("walls", "interior_wall"), ("floor", "interior_floor"), ("ceiling", "interior_ceiling")):
                v, i = geo["interior"][part]
                if len(i):
                    scene.add(_make_object(Mesh(v, i), facade_texture(tex_kind), cast_shadow=False, specular=0.04))
        ok_doors = [d for d in geo["doors"] if d.get("ok")]
        for d in ok_doors:
            door_points.append((d["at"][0], d["at"][1], d["normal"][0], d["normal"][1]))
            frame_parts += door_frame(d, lift=st["lift"])
            has_plinth = st["lift"] > 0 and (st["plinth"] if st["plinth"] is not None else (st["base"] == 0 and not st["roof_only"]))
            spec = d.get("spec") or {}
            if has_plinth and not spec.get("inner"):
                sw = spec.get("stairs")
                w_req = float(sw["width"]) if isinstance(sw, dict) and "width" in sw else d["width"] + 2.0
                back, fwd = d["run"]
                s_tot, s_c = d["edge_len"] + back + fwd, d["s_center"] + back
                w_max = 2 * min(s_c, s_tot - s_c) - 0.3
                stair = Stair(d["at"], d["normal"], max(min(w_req, w_max), d["width"] + 0.4), st["lift"])
                stairs.append(stair)
                door_stairs.append((d, stair))
            elif not spec.get("inner"):
                door_stairs.append((d, None))
            if st["enterable"] and st["lift"] > 0 and has_plinth:
                pass
        for d in geo["doors"]:
            if not d.get("ok"):
                report.append(f"  {label}: cửa tại {tuple(round(c, 1) for c in d['at'])} bị bỏ ({d.get('why')})")
        if st["doors"]:
            print(f"[OSM] {label}: {len(ok_doors)}/{len(st['doors'])} cửa" + (", vào được bên trong" if st["enterable"] else ""))
        if geo["forced"] or geo["holes_ignored"]:
            report.append(f"  {label}: " + ("có lỗ/sân trong (mái phủ kín lỗ) " if geo["holes_ignored"] else "")
                          + (f"{geo['forced']} lần cắt tai ép (đường viền suy biến)" if geo["forced"] else ""))
    if frame_parts:
        fv, fi = merge_meshes(frame_parts)
        scene.add(_make_object(Mesh(fv, fi), facade_texture("fence"), cast_shadow=False, specular=0.2, shininess=24.0))
    if stairs:
        sv, si = stairs_mesh(stairs)
        scene.add(_make_object(Mesh(sv, si), facade_texture("stair"), specular=0.05))
    user_parking = landscape.get("parking", [])
    extra_concrete = [p["polygon"] for p in user_parking]
    parking_edges = [rings_to_edges([pk["ring"]]) for pk in data.get("parking", [])] \
        + [rings_to_edges([p["polygon"]]) for p in user_parking]
    for p in user_parking:
        if p.get("fence", True):
            (fv, fi), boxes, rep = fence_for_polygon(p["polygon"], p.get("gates"))
            obj = _make_object(Mesh(fv, fi), facade_texture("fence"), cast_shadow=False, specular=0.2, shininess=24.0)
            obj.collider_boxes = boxes
            scene.add(obj)
            for r in rep:
                if not r["ok"]:
                    report.append(f"  bãi xe {p.get('name', '?')}: cổng tại {r['at']} cách rào quá xa, bỏ qua")
        if p.get("roof"):
            rf = p["roof"]
            h, th = float(rf.get("height", 3.2)), float(rf.get("thickness", 0.25))
            g = extrude_arrays(p["polygon"], [], h + th, base=h, default_kind="roof")
            for kind, (v, i) in g["walls"].items():
                scene.add(_make_object(Mesh(v, i), facade_texture(kind), specular=0.05))
            scene.add(_make_object(Mesh(*g["roof"]), roof_tex, specular=0.05))
            scene.add(_make_object(Mesh(*g["bottom"]), roof_tex, cast_shadow=False, specular=0.02))
            poly = np.asarray(p["polygon"], float)
            lo_p, hi_p = poly.min(axis=0), poly.max(axis=0)
            sp = float(rf.get("column_spacing", 8.0))
            nx_, nz_ = max(2, int(np.ceil((hi_p[0] - lo_p[0] - 1.6) / sp)) + 1), max(2, int(np.ceil((hi_p[1] - lo_p[1] - 1.6) / sp)) + 1)
            gate_pts = [np.array(gt["at"], float) for gt in p.get("gates", [])]
            cols, boxes_c = [], []
            for cx in np.linspace(lo_p[0] + 0.8, hi_p[0] - 0.8, nx_):
                for cz in np.linspace(lo_p[1] + 0.8, hi_p[1] - 0.8, nz_):
                    if any(np.hypot(cx - g_[0], cz - g_[1]) < 3.5 for g_ in gate_pts):
                        continue
                    cols.append(oriented_box(cx, cz, 0.0, h, 0.28, 0.28, 0.0))
                    boxes_c.append([cx - 0.14, 0.0, cz - 0.14, cx + 0.14, h, cz + 0.14])
            if cols:
                cv, ci = merge_meshes(cols)
                obj = _make_object(Mesh(cv, ci), facade_texture("stone"), specular=0.05)
                obj.collider_boxes = np.array(boxes_c, np.float32)
                scene.add(obj)
    all_pts = np.array([p for b in active for p in b["outer"]], dtype=np.float64)
    extra_paving, paved_edges = [], [rings_to_edges([s.footprint()]) for s in stairs]
    if len(all_pts):
        lo, hi = all_pts.min(axis=0) - ROAD_MARGIN, all_pts.max(axis=0) + ROAD_MARGIN
        bounds_all = (lo[0], lo[1], hi[0], hi[1])
        dp = landscape.get("door_paths", {})
        if dp.get("enabled", True):
            segs = road_segments(data.get("roads", []), styles, bounds_all)
            for d, stair in door_stairs:
                n_ = np.array(d["normal"], float)
                dd_ = np.array([n_[1], -n_[0]])
                s0 = stair.depth if stair is not None else 0.0
                o = np.array(d["at"], float) + n_ * s0
                hit = ray_road_hit(o, n_, segs, float(dp.get("max_length", 30.0)))
                length = hit + 0.3 if hit is not None else float(dp.get("default_length", 8.0))
                if length < 0.6:
                    continue
                wp = d["width"] + float(dp.get("extra_width", 2.5))
                ring = [tuple(o - dd_ * wp / 2 - n_ * 0.1), tuple(o + dd_ * wp / 2 - n_ * 0.1),
                        tuple(o + dd_ * wp / 2 + n_ * length), tuple(o - dd_ * wp / 2 + n_ * length)]
                extra_paving.append(ring)
                paved_edges.append(rings_to_edges([ring]))
        bench_placements = []
        for area in landscape.get("paved_areas", []):
            extra_paving.append(area["polygon"])
            paved_edges.append(rings_to_edges([area["polygon"]]))
            bn = area.get("benches")
            if bn and bn.get("count", 0) > 0:
                rng_b = np.random.default_rng(bn.get("seed", 7))
                avoid = [(x_, z_, bn.get("door_clearance", 4.0)) for x_, z_, _, _ in door_points] \
                    + [tuple(a) for a in bn.get("avoid", [])]
                bench_placements += place_benches(area["polygon"], int(bn["count"]), float(bn.get("min_spacing", 9.0)), avoid, rng_b)
            pg = area.get("pergolas")
            if pg and pg.get("at"):
                poly = np.asarray(area["polygon"], float)
                lo_, hi_ = poly.min(axis=0), poly.max(axis=0)
                along_x = (hi_[0] - lo_[0]) >= (hi_[1] - lo_[1])
                mid = (lo_ + hi_) / 2
                spots = [(float(a), float(mid[1]), 0.0) if along_x else (float(mid[0]), float(a), math.pi / 2)
                         for a in pg["at"]]
                (pv, pi_), pboxes = pergolas_mesh(spots, width=float(pg.get("width", 3.0)),
                                                  depth=float(pg.get("depth", 2.2)), height=float(pg.get("height", 2.8)))
                obj = _make_object(Mesh(pv, pi_), facade_texture("stair"), specular=0.10, shininess=16.0)
                obj.collider_boxes = pboxes
                scene.add(obj)
                print(f"[OSM] {len(spots)} giàn hoa trên lối đi")
        if bench_placements:
            (bv, bi), bboxes = benches_mesh(bench_placements)
            obj = _make_object(Mesh(bv, bi), facade_texture("stone"), specular=0.08, shininess=16.0)
            obj.collider_boxes = bboxes
            scene.add(obj)
            scene.benches = bench_placements
            print(f"[OSM] {len(bench_placements)} ghế đá (mỗi ghế 3 chỗ ngồi)")
        n_roads, n_groups = add_ground_layers(scene, data, bounds_all, styles, extra_concrete, extra_paving)
        print(f"[OSM] đã trải {n_roads} đường/lối đi/bãi xe ({n_groups} nhóm texture); {len(stairs)} cầu thang")
    scene.tour_waypoints, scene.tour_info = None, None
    if tour_spec:
        try:
            from scene.tour_route import build_tour
            tour = build_tour(data.get("roads", []), tour_spec, obstacle_edges,
                              eye_height=getattr(config, "EYE_HEIGHT", 1.7))
            scene.tour_waypoints, scene.tour_info = tour["waypoints"], tour
            speed = getattr(config, "TOUR_SPEED", 3.5)
            print(f"[Tour] {len(tour['waypoints'])} waypoint, dài {tour['length']:.0f} m "
                  f"(~{tour['length'] / speed / 60:.1f} phút ở {speed} m/s), cách tường tối thiểu {tour['clearance']:.1f} m")
            for w in tour["warnings"]:
                print("[Tour] cảnh báo:", w)
        except Exception as e:
            import traceback
            print(f"[Tour] Không dựng được tuyến tự động ({e}) -> dùng config.TOUR_WAYPOINTS:")
            traceback.print_exc()
    (sx, sz), yaw = find_spawn({"buildings": active}, collider_edges)
    scene.spawn_pos = (sx, config.EYE_HEIGHT, sz)
    scene.spawn_yaw = yaw
    def height_at(x, z):
        """Return walking height at a point, including stairs and upper floors."""
        h = 0.0
        for edges, lift in interiors:
            if lift > h and point_in_edges(x, z, edges):
                h = lift
        for s in stairs:
            h = max(h, s.height_at(x, z))
        return h
    scene.height_at = height_at
    scene.stairs = stairs
    if len(all_pts) and landscape.get("vegetation", True):
        from scene.landscape import Constraints, generate_vegetation
        from graphics.procedural_textures import vegetation_textures
        bounds = (all_pts[:, 0].min(), all_pts[:, 1].min(), all_pts[:, 0].max(), all_pts[:, 1].max())
        cons = Constraints(footprints, road_segments(data.get("roads", []), styles,
                                                     (bounds[0] - 150, bounds[1] - 150, bounds[2] + 150, bounds[3] + 150)),
                           parking_edges, door_points,
                           scene.tour_info["dense"] if scene.tour_info else None, (sx, sz), paved_edges)
        plants, stats = generate_vegetation(landscape, cons, data.get("roads", []), styles, bounds,
                                            existing_trees=[tuple(p) for p in data.get("trees", [])])
        if plants:
            trunk, canopy, boxes = build_plants(plants, seed=landscape.get("seed", 42))
            foliage_tex, trunk_tex = vegetation_textures()
            if canopy is not None:
                scene.add(_make_object(Mesh(*canopy), foliage_tex, specular=0.02, shininess=8.0))
            if trunk is not None:
                obj = _make_object(Mesh(*trunk), trunk_tex, specular=0.02, shininess=8.0)
                obj.collider_boxes = boxes
                scene.add(obj)
            detail = ", ".join(f"{k.replace('zone:', '')}={v}" for k, v in stats.items() if v)
            print(f"[Cây] {len(plants)} cây/bụi ({detail}); {len(boxes)} thân cây có va chạm")
        scene.plants = plants
    print(f"[OSM] {len(active)}/{len(all_buildings)} tòa nhà (đã bỏ {len(all_buildings) - len(active)} tòa skip) -> "
          f"{len(scene.objects)} vật thể ({n_walls} nhóm tường); điểm xuất phát ({sx:.1f}, {sz:.1f})")
    for line in report:
        print("[OSM] cảnh báo:" + line)
    return scene
