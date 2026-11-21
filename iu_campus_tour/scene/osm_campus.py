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
from scene.geometry2d import rings_to_edges, circle_hits_rings
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
    if not path.exists():
        return {"by_name": {}, "by_id": {}, "connect_doors": {}, "extra_buildings": []}
    raw = load_json(path)
    connect_doors = {}
    for c in raw.get("connect", []):
        door = {"at": c["at"], "width": c.get("width", 4.0)}
        for bid in c["between"]:
            connect_doors.setdefault(bid, []).append(door)
    return {"by_name": {_norm(k): v for k, v in raw.get("by_name", {}).items()},
            "by_id": raw.get("by_id", {}),
            "connect_doors": connect_doors,
            "extra_buildings": raw.get("extra_buildings", [])}


def _polygon_area(ring):
    return abs(signed_area(ring))


def resolve_style(b, overrides):
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
        "facade_by_side": ov.get("facade_by_side"),          # {"N": "maroon", "S": "white", ...}
        "roof_only": bool(roof_only),
        "ground_facade": ov.get("ground_facade"),             # Mặt tiền riêng cho tầng trệt (vd "white_plain": trắng, không cửa sổ)
        "enterable": bool(ov.get("enterable", False)),         # True: tầng trệt rỗng, có cửa, đi vào trong được
        "doors": doors,
        "collidable": bool(ov.get("collidable", not roof_only)),
        "skip": bool(ov.get("skip", False)),
    }


def _make_object(mesh, texture, position=(0, 0, 0), cast_shadow=True, collidable=False,
                 specular=0.10, shininess=32.0):
    obj = SceneObject(mesh, texture, position=position)
    obj.cast_shadow, obj.collidable = cast_shadow, collidable
    obj.specular, obj.shininess = specular, shininess
    return obj


def find_spawn(data, collider_edges):
    pts = np.array([p for b in data["buildings"] for p in b["outer"]], dtype=np.float64)
    if len(pts) == 0:
        return (0.0, 15.0), -90.0
    cx, cz = (pts.min(axis=0) + pts.max(axis=0)) / 2          # Tâm hộp bao (không dùng trung bình đỉnh: lệch về tòa nhiều đỉnh)
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
    styles = dict(ROAD_STYLE)
    for kind, w in (landscape or {}).get("road_widths", {}).items():
        if kind in styles:
            s = list(styles[kind])
            s[1] = float(w)
            styles[kind] = tuple(s)
    return styles


def road_segments(roads, styles, bounds):
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


def add_ground_layers(scene, data, bounds, styles=None, extra_concrete=()):
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
        # Mỗi đường lệch độ cao thêm vài phần nghìn mét: các đường CÙNG loại chồng nhau ở ngã ba không tranh chấp độ sâu
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
    for tex_kind, parts in groups.items():
        v, i = merge_meshes(parts)
        scene.add(_make_object(Mesh(v, i), make_ground_texture(tex_kind), cast_shadow=False,
                               collidable=False, specular=0.02, shininess=8.0))
    return count, len(groups)


def door_frame(door, door_h=2.7, jamb=0.12, depth=0.18):
    cx, cz = door["at"]
    nx, nz = door["normal"]
    d = np.array([nz, -nx])
    ang = float(np.arctan2(d[1], d[0]))
    w = door["width"]
    parts = []
    for s in (-1, 1):
        c = np.array([cx, cz]) + d * s * (w / 2 + jamb / 2)
        parts.append(oriented_box(c[0], c[1], 0.0, door_h, jamb, depth, ang))
    parts.append(oriented_box(cx, cz, door_h, door_h + 0.14, w + 2 * jamb, depth, ang))
    return parts


def _manual_building(i, b):
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
    G = 2.0 * (extent + 250.0)                            # Mặt đất rộng hơn khu nhà rất nhiều
    scene.add(_make_object(create_box((G, 0.2, G), tex_size=GRASS_TILE),
                           ground_texture if ground_texture is not None else make_ground_texture("grass"),
                           position=(0, -0.1, 0), cast_shadow=False, collidable=False,
                           specular=0.02, shininess=8.0))

    roof_tex = facade_texture("roof")
    collider_edges, report, n_walls = [], [], 0
    obstacle_edges, footprints = [], []                    # obstacle: tường tour phải tránh (cầu nối nâng cao thì không); footprints: mọi tòa (để trồng cây)
    door_points = []                                       # (x, z, nx, nz) mọi cửa ra vào đã dựng
    frame_parts = []                                       # khung cửa (hai cột + thanh ngang) để dễ nhận ra ô cửa
    for b in sorted(active, key=lambda b: -_polygon_area(b["outer"])):
        st = resolve_style(b, overrides)
        try:
            geo = extrude_arrays(b["outer"], b.get("holes"), st["height"], TILE_W, FLOOR_H,
                                 kind_of_side=st["facade_by_side"], default_kind=st["facade"],
                                 roof_only=st["roof_only"], base=st["base"],
                                 ground_kind=st["ground_facade"], doors=st["doors"], enterable=st["enterable"])
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
                # Hộp bao CẢ tòa nhà (đủ chiều cao) cho pha rộng; pha hẹp dùng đường viền thật.
                # (khối nâng lên: đáy hộp ở y = base nên người chơi đi bên dưới được)
                obj.aabb_min = np.array([xs.min(), st["base"], zs.min()], dtype=np.float32)
                obj.aabb_max = np.array([xs.max(), st["height"], zs.max()], dtype=np.float32)
                if st["enterable"] and geo["wall_edges"] is not None:
                    obj.wall_edges = geo["wall_edges"]     # Tòa rỗng có cửa: chỉ các đoạn tường cản, ô cửa đi qua được
                else:
                    obj.footprint_edges = edges
                collider_edges.append(edges)
            scene.add(obj)
            primary = False
            n_walls += 1
        rv, ri = geo["roof"]
        scene.add(_make_object(Mesh(rv, ri), roof_tex, specular=0.05))
        if geo["bottom"] is not None:                      # Mặt dưới của khối nâng lên
            bv, bi = geo["bottom"]
            scene.add(_make_object(Mesh(bv, bi), roof_tex, specular=0.02))
        if geo["interior"] is not None:                    # Nội thất tầng trệt: tường trong, sàn, trần (không đổ bóng, đỡ tốn)
            for part, tex_kind in (("walls", "interior_wall"), ("floor", "interior_floor"), ("ceiling", "interior_ceiling")):
                v, i = geo["interior"][part]
                if len(i):
                    scene.add(_make_object(Mesh(v, i), facade_texture(tex_kind), cast_shadow=False, specular=0.04))
        ok_doors = [d for d in geo["doors"] if d.get("ok")]
        for d in ok_doors:
            door_points.append((d["at"][0], d["at"][1], d["normal"][0], d["normal"][1]))
            frame_parts += door_frame(d)
        for d in geo["doors"]:
            if not d.get("ok"):
                report.append(f"  {label}: cửa tại {tuple(round(c, 1) for c in d['at'])} bị bỏ ({d.get('why')})")
        if st["doors"]:
            print(f"[OSM] {label}: {len(ok_doors)}/{len(st['doors'])} cửa" + (", vào được bên trong" if st["enterable"] else ""))
        if geo["forced"] or geo["holes_ignored"]:
            report.append(f"  {label}: " + ("có lỗ/sân trong (mái phủ kín lỗ) " if geo["holes_ignored"] else "")
                          + (f"{geo['forced']} lần cắt tai ép (đường viền suy biến)" if geo["forced"] else ""))

    if frame_parts:                                        # Tất cả khung cửa gộp một draw call
        fv, fi = merge_meshes(frame_parts)
        scene.add(_make_object(Mesh(fv, fi), facade_texture("fence"), cast_shadow=False, specular=0.2, shininess=24.0))

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

    all_pts = np.array([p for b in active for p in b["outer"]], dtype=np.float64)
    if len(all_pts):
        lo, hi = all_pts.min(axis=0) - ROAD_MARGIN, all_pts.max(axis=0) + ROAD_MARGIN
        n_roads, n_groups = add_ground_layers(scene, data, (lo[0], lo[1], hi[0], hi[1]), styles, extra_concrete)
        print(f"[OSM] đã trải {n_roads} đường/lối đi/bãi xe ({n_groups} nhóm texture)")

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

    # Cây xanh
    if len(all_pts) and landscape.get("vegetation", True):
        from scene.landscape import Constraints, generate_vegetation
        from graphics.procedural_textures import vegetation_textures
        bounds = (all_pts[:, 0].min(), all_pts[:, 1].min(), all_pts[:, 0].max(), all_pts[:, 1].max())
        cons = Constraints(footprints, road_segments(data.get("roads", []), styles,
                                                     (bounds[0] - 150, bounds[1] - 150, bounds[2] + 150, bounds[3] + 150)),
                           parking_edges, door_points,
                           scene.tour_info["dense"] if scene.tour_info else None, (sx, sz))
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
