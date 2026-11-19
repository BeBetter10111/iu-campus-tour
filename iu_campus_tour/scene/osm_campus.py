import json
import math

import numpy as np

import config
from graphics.extrude import clean_ring, extrude_arrays, signed_area, triangulate
from graphics.mesh import Mesh, create_box
from graphics.procedural_textures import facade_texture, FLOOR_H, surface_texture, TILE_W
from scene.geometry2d import rings_to_edges, circle_hits_rings
from scene.scene import Scene, SceneObject

DATA_DIR = getattr(config, "DATA_DIR", config.ASSET_DIR / "data")
ROOF_ONLY_KINDS = {"roof", "carport", "canopy"}      
SPAWN_CLEARANCE = 6.0                                
ROAD_WIDTHS = {
    "motorway": 10.0, "trunk": 9.0, "primary": 8.0, "secondary": 7.0,
    "tertiary": 6.0, "residential": 5.0, "unclassified": 4.5,
    "service": 4.0, "track": 3.0, "path": 2.0, "footway": 2.0,
    "pedestrian": 3.0, "steps": 1.5,
}


def _norm(s):
    return " ".join(str(s).casefold().split())


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_overrides(path):
    """Read building_overrides.json. Missing file -> rỗng."""
    if not path.exists():
        return {"by_name": {}, "by_id": {}}
    raw = load_json(path)
    return {"by_name": {_norm(k): v for k, v in raw.get("by_name", {}).items()},
            "by_id": raw.get("by_id", {})}


def _polygon_area(ring):
    return abs(signed_area(ring))


def build_surface_arrays(ring, y, tile=8.0):
    ring = clean_ring(ring)
    if len(ring) < 3:
        raise ValueError("Đường viền có ít hơn 3 đỉnh")
    triangles, _ = triangulate(ring)
    vertices = np.array([[x, y, z, 0, 1, 0, x / tile, z / tile] for x, z in ring], dtype=np.float32)
    indices = np.array([index for tri in triangles for index in (tri[0], tri[2], tri[1])],
                       dtype=np.uint32)
    return vertices, indices


def build_road_arrays(points, width, y=0.025, tile=8.0):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Đường cần danh sách điểm 2D")
    if width <= 0:
        raise ValueError("Chiều rộng đường phải lớn hơn 0")

    clean = [points[0]] if len(points) else []
    for point in points[1:]:
        if np.linalg.norm(point - clean[-1]) >= 0.05:
            clean.append(point)
    points = np.asarray(clean, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("Less than 2 distinct points after cleaning")

    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    directions = segments / lengths[:, None]
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    offsets = np.empty_like(points)
    half_width = width / 2.0
    offsets[0] = normals[0] * half_width
    offsets[-1] = normals[-1] * half_width
    for i in range(1, len(points) - 1):
        miter = normals[i - 1] + normals[i]
        miter_length = np.linalg.norm(miter)
        if miter_length < 1e-8:
            offsets[i] = normals[i] * half_width
            continue
        miter /= miter_length
        denominator = float(np.dot(miter, normals[i]))
        if denominator <= 0.25:
            offsets[i] = normals[i] * half_width
        else:
            offsets[i] = miter * min(half_width / denominator, half_width * 3.0)

    distance = np.concatenate(([0.0], np.cumsum(lengths)))
    vertices = []
    for i, point in enumerate(points):
        u = distance[i] / tile
        left, right = point + offsets[i], point - offsets[i]
        vertices.extend(([left[0], y, left[1], 0, 1, 0, u, 1],
                         [right[0], y, right[1], 0, 1, 0, u, 0]))
    indices = []
    for i in range(len(points) - 1):
        left, right = 2 * i, 2 * i + 1
        next_left, next_right = left + 2, right + 2
        indices.extend((left, next_left, next_right, left, next_right, right))
    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.uint32)


def _combine_mesh_arrays(parts):
    if not parts:
        return None
    vertices, indices = [], []
    offset = 0
    for part_vertices, part_indices in parts:
        if len(part_vertices) == 0 or len(part_indices) == 0:
            continue
        vertices.append(part_vertices)
        indices.append(part_indices + offset)
        offset += len(part_vertices)
    if not vertices:
        return None
    return np.concatenate(vertices), np.concatenate(indices)


def _road_width(kind):
    return ROAD_WIDTHS.get(str(kind or "").lower(), 4.0)


def resolve_style(b, overrides):
    kind = (b.get("kind") or "yes").lower()
    name = b.get("name") or ""
    ov = {}
    ov.update(overrides["by_name"].get(_norm(name), {}))
    if b.get("name_en"):
        ov = {**overrides["by_name"].get(_norm(b["name_en"]), {}), **ov}
    ov.update(overrides["by_id"].get(b.get("id", ""), {}))

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


def build_campus_from_osm(data, overrides=None, ground_texture=None):
    overrides = overrides or {"by_name": {}, "by_id": {}}
    scene = Scene()

    active = [b for b in data["buildings"] if not resolve_style(b, overrides)["skip"]]

    map_points = [p for b in active for p in b["outer"]]
    map_points.extend(p for road in data.get("roads", []) for p in road.get("points", []))
    map_points.extend(p for area in data.get("parking", []) for p in area.get("ring", []))
    map_points.extend(p for area in data.get("water", []) for p in area.get("ring", []))
    map_points.extend(data.get("trees", []))
    extent = max([abs(c) for p in map_points for c in p] + [100.0])
    G = 2.0 * (extent + 250.0)                            # Mặt đất rộng hơn khu nhà rất nhiều
    if ground_texture is not None:
        scene.add(_make_object(create_box((G, 0.2, G), tex_size=6.0), ground_texture,
                               position=(0, -0.1, 0), cast_shadow=False, collidable=False,
                               specular=0.02, shininess=8.0))

    feature_report = []
    roads = []
    for road in data.get("roads", []):
        try:
            roads.append(build_road_arrays(road.get("points", []), _road_width(road.get("kind"))))
        except ValueError as e:
            feature_report.append(f"  bỏ qua đường {road.get('id')}: {e}")
    road_mesh = _combine_mesh_arrays(roads)
    if road_mesh:
        scene.add(_make_object(Mesh(*road_mesh), surface_texture("road"), cast_shadow=False,
                               specular=0.02, shininess=8.0))

    for key, y, texture_kind in (("parking", 0.035, "parking"), ("water", 0.015, "water")):
        surfaces = []
        for area in data.get(key, []):
            try:
                surfaces.append(build_surface_arrays(area.get("ring", []), y))
            except ValueError as e:
                feature_report.append(f"  bỏ qua {key} {area.get('id')}: {e}")
        mesh_data = _combine_mesh_arrays(surfaces)
        if mesh_data:
            scene.add(_make_object(Mesh(*mesh_data), surface_texture(texture_kind), cast_shadow=False,
                                   specular=0.02, shininess=8.0))

    roof_tex = facade_texture("roof")
    collider_edges, report, n_walls = [], [], 0
    for b in sorted(active, key=lambda b: -_polygon_area(b["outer"])):
        st = resolve_style(b, overrides)
        try:
            geo = extrude_arrays(b["outer"], b.get("holes"), st["height"], TILE_W, FLOOR_H,
                                 kind_of_side=st["facade_by_side"], default_kind=st["facade"],
                                 roof_only=st["roof_only"], base=st["base"])
        except ValueError as e:
            report.append(f"  bỏ qua {b.get('id')}: {e}")
            continue

        rings = [b["outer"]] + [h for h in (b.get("holes") or [])]
        edges = rings_to_edges(rings)
        xs, zs = np.array(b["outer"])[:, 0], np.array(b["outer"])[:, 1]
        primary = True
        for kind, (v, i) in geo["walls"].items():
            obj = _make_object(Mesh(v, i), facade_texture(kind), collidable=st["collidable"] and primary,
                               specular=0.12)
            if obj.collidable:
                obj.aabb_min = np.array([xs.min(), st["base"], zs.min()], dtype=np.float32)
                obj.aabb_max = np.array([xs.max(), st["height"], zs.max()], dtype=np.float32)
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

        label = b.get("name") or b.get("id")
        if geo["forced"] or geo["holes_ignored"]:
            report.append(f"  {label}: " + ("có lỗ/sân trong (mái phủ kín lỗ) " if geo["holes_ignored"] else "")
                          + (f"{geo['forced']} lần cắt tai ép (đường viền suy biến)" if geo["forced"] else ""))

    (sx, sz), yaw = find_spawn({"buildings": active}, collider_edges)
    scene.spawn_pos = (sx, config.EYE_HEIGHT, sz)
    scene.spawn_yaw = yaw
    print(f"[OSM] {len(active)}/{len(data['buildings'])} tòa nhà (đã bỏ {len(data['buildings']) - len(active)} tòa skip) -> {len(scene.objects)} vật thể "
          f"({n_walls} nhóm tường); điểm xuất phát ({sx:.1f}, {sz:.1f})")
    print(f"[OSM] lớp nền: {len(roads)} đường, {len(data.get('parking', []))} bãi xe, "
          f"{len(data.get('water', []))} vùng nước, {len(data.get('trees', []))} cây")
    for line in feature_report:
        print("[OSM] cảnh báo:" + line)
    for line in report:
        print("[OSM] cảnh báo:" + line)
    return scene
