import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
for name, value in (("PLAYER_RADIUS", 0.35), ("PLAYER_HEAD_GAP", 0.2), ("EYE_HEIGHT", 1.7)):
    if not hasattr(config, name):
        setattr(config, name, value)

from scene.geometry2d import rings_to_edges, points_in_edges, point_in_edges, dist_points_edges
from scene.road_fit import fit_roads, overlap_area
from scene.sidewalks import plan_sidewalks, build_sidewalks
from scene.collision import CollisionWorld
from graphics.sidewalk import sidewalk_side
from graphics.fence import perimeter_fence, gates_from_roads, PERIMETER_HEIGHT

STYLES = {"service": ("asphalt", 5.5, 0.04, 4.0, False), "footway": ("paving", 2.8, 0.06, 2.0, False),
          "residential": ("asphalt", 6.5, 0.034, 4.0, False)}
SQUARE = [(0, 0), (20, 0), (20, 20), (0, 20)]


def _edges(*rings):
    return [rings_to_edges([r]) for r in rings]


def test_points_in_edges_matches_scalar():
    e = rings_to_edges([[(0, 0), (10, 0), (10, 10), (5, 4), (0, 10)]])
    rng = np.random.default_rng(1)
    pts = rng.uniform(-2, 12, (200, 2))
    vec = points_in_edges(pts, e)
    assert all(bool(v) == point_in_edges(p[0], p[1], e) for v, p in zip(vec, pts))
    assert np.allclose(dist_points_edges(np.array([[5.0, -3.0]]), e), 3.0)


def test_road_through_building_is_cut_and_clear():
    road = {"id": "r", "kind": "service", "points": [[-30, 10], [50, 10]]}
    obst = _edges(SQUARE)
    out, rep = fit_roads([road], STYLES, obst)
    assert rep["checked"] == 1 and len(out) >= 1
    assert overlap_area([road], STYLES, obst) > 100
    assert overlap_area(out, STYLES, obst) < 1e-6, "road must not overlap the footprint any more"


def test_road_near_wall_is_pushed_out_not_deleted():
    road = {"id": "r", "kind": "service", "points": [[-10, 23], [30, 23]]}   # 3 m from the wall, half-width 2.75
    obst = _edges(SQUARE)
    out, rep = fit_roads([road], STYLES, obst)
    assert len(out) == 1 and rep["moved"] == 1
    pts = np.array(out[0]["points"])
    d = dist_points_edges(pts, obst[0])
    assert d.min() >= 2.75 + 1.0, "asphalt edge must clear the wall by more than a metre"
    assert abs(pts[:, 0].min() - (-10)) < 1e-6 and abs(pts[:, 0].max() - 30) < 1e-6, "length is preserved"


def test_far_road_is_untouched():
    road = {"id": "r", "kind": "service", "points": [[-10, 80], [30, 80]]}
    out, rep = fit_roads([road], STYLES, _edges(SQUARE))
    assert out == [road] and rep["checked"] == 0


def test_unstyled_kinds_are_ignored():
    road = {"id": "r", "kind": "platform", "points": [[-30, 10], [50, 10]]}
    out, _ = fit_roads([road], STYLES, _edges(SQUARE))
    assert out == [road]


def test_sidewalk_geometry_faces_and_heights():
    P = np.array([[0, 0], [5, 0], [10, 1], [15, 3]], float)
    for side in (1, -1):
        walk, kerb = sidewalk_side(P, side, 3.0, 0.18, np.full(len(P), 2.0), 0.04, 0.15, 2.0, 4.0)
        for v, i in (walk, kerb):
            assert np.isfinite(v).all()
            tri = v[i.reshape(-1, 3)][:, :, :3]
            geo = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            decl = v[i.reshape(-1, 3)[:, 0], 3:6]
            assert (np.einsum("ij,ij->i", geo, decl) > 0).all(), "winding must agree with normals (face culling)"
        assert np.allclose(walk[0][:, 1], 0.15), "sidewalk top sits at kerb height"
        assert walk[0][:, 1].min() > 0.04


def test_sidewalk_runs_stop_at_buildings_and_other_roads():
    main = {"id": "m", "kind": "residential", "points": [[-60, 0], [60, 0]]}
    cross = {"id": "c", "kind": "residential", "points": [[0, -60], [0, 60]]}
    wall = [(20, 3.0), (40, 3.0), (40, 12.0), (20, 12.0)]
    runs = plan_sidewalks([main, cross], STYLES, _edges(wall), [], (-100, -100, 100, 100))
    for r in runs:
        P = r["P"]
        if r["road"] == 0:
            assert not (np.abs(P[:, 0]) < 3.2).any(), "no sidewalk across the carriageway of the crossing road"
        off = P + r["side"] * np.array([0.0, 1.0]) * (r["hw"] + 0.18 + r["walk_w"][:, None])
        assert not points_in_edges(off, rings_to_edges([wall])).any(), "sidewalk edge must not enter the building"
        assert (r["walk_w"] >= 0.9).all()


def test_sidewalk_narrows_beside_wall():
    cfg = _load("landscape.json")["sidewalks"]
    road = {"id": "m", "kind": "residential", "points": [[-30, 0], [30, 0]]}
    wall = [(-10, 5.2), (10, 5.2), (10, 12), (-10, 12)]      # 5.2 - 3.25 - 0.18 = 1.77 m left
    runs = plan_sidewalks([road], STYLES, _edges(wall), [], (-100, -100, 100, 100), cfg)
    near = [r for r in runs if r["side"] == 1]
    assert near, "a narrower sidewalk is still built"
    w = np.concatenate([r["walk_w"] for r in near])
    assert w.min() < cfg["kinds"]["residential"] - 0.1 and w.min() >= 0.9
    for r in near:
        outer = r["P"] + r["side"] * np.array([0.0, 1.0]) * (
            r["hw"] + cfg["kerb_width"] + r["walk_w"][:, None])
        assert dist_points_edges(outer, rings_to_edges([wall])).min() >= 0.14


def test_sidewalk_uses_wider_config_in_open_area():
    cfg = _load("landscape.json")["sidewalks"]
    road = {"id": "m", "kind": "residential", "points": [[-30, 0], [30, 0]]}
    runs = plan_sidewalks([road], STYLES, [], [], (-100, -100, 100, 100), cfg)
    assert runs and all(np.allclose(r["walk_w"], cfg["kinds"]["residential"]) for r in runs)


def test_build_sidewalks_meshes():
    road = {"id": "m", "kind": "residential", "points": [[-30, 0], [30, 0]]}
    walk, kerb, stats = build_sidewalks([road], STYLES, [], [], (-100, -100, 100, 100))
    assert walk is not None and kerb is not None and stats["runs"] == 2
    assert abs(stats["length"] - 120.0) < 3.0


RING = [(0, 0), (100, 0), (100, 80), (0, 80)]


def test_gates_found_where_roads_cross_the_ring():
    roads = [{"id": "a", "kind": "service", "points": [[50, -20], [50, 40]]},
             {"id": "b", "kind": "service", "points": [[20, 20], [60, 30]]}]
    gates = gates_from_roads(RING, roads, STYLES)
    assert len(gates) == 1 and abs(gates[0]["at"][0] - 50) < 1e-6 and abs(gates[0]["at"][1]) < 1e-6
    assert gates[0]["width"] >= 5.5


def test_perimeter_fence_blocks_walkers_except_at_gates():
    gate = {"at": [50, 0], "width": 5.0}
    (solid, bars), boxes, rep = perimeter_fence(RING, [gate])
    assert rep[0]["ok"] and solid is not None and bars is not None
    world = CollisionWorld([types.SimpleNamespace(collidable=False, collider_boxes=boxes)])
    y = config.EYE_HEIGHT
    p = np.array([30.0, y, -3.0], np.float32)
    assert world.collides(np.array([30.0, y, 0.0], np.float32)), "walking into the fence is blocked"
    assert not world.collides(np.array([50.0, y, 0.0], np.float32)), "the gate is open"
    q = world.move(p, np.array([0.0, 0.0, 6.0], np.float32))
    assert q[2] < 0.3, "player cannot walk through the fence"
    assert boxes[:, 4].max() >= PERIMETER_HEIGHT - 1e-6


def test_perimeter_fence_is_a_closed_loop_without_gates():
    (solid, bars), boxes, rep = perimeter_fence(RING, [])
    assert rep == []
    for x, z in [(50, 0), (100, 40), (50, 80), (0, 40)]:
        assert any(b[0] <= x <= b[3] and b[2] <= z <= b[5] for b in boxes), (x, z)


# ---- real campus data -----------------------------------------------------
def _load(name):
    with open(config.ASSET_DIR / "data" / name, "r", encoding="utf-8") as f:
        return json.load(f)


def test_real_data_roads_clear_of_buildings_and_inside_fence():
    from scene import osm_campus as oc
    data, ov, land = _load("campus_osm.json"), oc.load_overrides(config.ASSET_DIR / "data" / "building_overrides.json"), \
        _load("landscape.json")
    styles = oc.effective_road_styles(land)
    buildings = list(data["buildings"]) + [oc._manual_building(i, b) for i, b in enumerate(ov["extra_buildings"])]
    obst, solid = [], []
    for b in buildings:
        st = oc.resolve_style(b, ov)
        if st["skip"] or st["roof_only"] or st["base"] >= oc.RAISED_MIN:
            continue
        obst.append(rings_to_edges([b["outer"]] + list(b.get("holes") or [])))
        solid.append(b)
    fit = land.get("road_fit", {})
    fitted, rep = fit_roads(data["roads"], styles, obst, max_push=fit.get("max_push", 3.5))
    assert overlap_area(fitted, styles, obst) < 1.0, "no road may overlap a building after fitting"
    assert overlap_area(data["roads"], styles, obst) > overlap_area(fitted, styles, obst)

    ring = land["campus_boundary"]["polygon"]
    edges = rings_to_edges([ring])
    for b in solid:
        pts = np.array(b["outer"], float)
        c = pts.mean(axis=0)
        if abs(c[0]) < 150 and -100 < c[1] < 120:           # the central compound
            assert points_in_edges(pts, edges).all(), f"{b.get('name') or b['id']} must be inside the campus fence"
    gates = gates_from_roads(ring, fitted, styles, kinds=oc.GATE_KINDS)
    assert gates, "inner roads that leave the compound get a gate"
