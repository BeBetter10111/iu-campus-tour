import json
import math

import numpy as np

import config
from graphics.extrude import extrude_arrays, signed_area
from graphics.mesh import Mesh, create_box
from graphics.procedural_textures import facade_texture, FLOOR_H, TILE_W
from scene.geometry2d import rings_to_edges, circle_hits_rings
from scene.scene import Scene, SceneObject

DATA_DIR = getattr(config, "DATA_DIR", config.ASSET_DIR / "data")
ROOF_ONLY_KINDS = {"roof", "carport", "canopy"}      
SPAWN_CLEARANCE = 6.0                                


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


def build_campus_from_osm(data, overrides=None, ground_texture=None):
    overrides = overrides or {"by_name": {}, "by_id": {}}
    scene = Scene()

    active = [b for b in data["buildings"] if not resolve_style(b, overrides)["skip"]]

    extent = max([abs(c) for b in active for p in b["outer"] for c in p] + [100.0])
    G = 2.0 * (extent + 250.0)                            # Mặt đất rộng hơn khu nhà rất nhiều
    if ground_texture is not None:
        scene.add(_make_object(create_box((G, 0.2, G), tex_size=6.0), ground_texture,
                               position=(0, -0.1, 0), cast_shadow=False, collidable=False,
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
        if geo["bottom"] is not None:                      # Mặt dưới của khối nâng lên
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
    for line in report:
        print("[OSM] cảnh báo:" + line)
    return scene
