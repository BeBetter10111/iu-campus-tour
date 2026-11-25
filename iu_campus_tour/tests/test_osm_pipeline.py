import os
import sys
import types
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
for name, value in (("PLAYER_RADIUS", 0.35), ("PLAYER_HEAD_GAP", 0.2), ("EYE_HEIGHT", 1.7)):
    if not hasattr(config, name):
        setattr(config, name, value)
from graphics.extrude import (signed_area, triangulate, build_roof, build_walls, extrude_arrays,
                              clean_ring, side_of_normal)
from scene.geometry2d import rings_to_edges, point_in_edges
from scene.collision import CollisionWorld
L_SHAPE = [(0, 0), (20, 0), (20, 8), (8, 8), (8, 20), (0, 20)]
U_SHAPE = [(0, 0), (30, 0), (30, 20), (22, 20), (22, 8), (8, 8), (8, 20), (0, 20)]
SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
HOLE = [(3, 3), (7, 3), (7, 7), (3, 7)]
def tri_area_sum(poly, tris):
    p = np.asarray(poly, float)
    return sum(abs(signed_area(p[list(t)])) for t in tris)
def test_triangulate_area():
    for name, poly in (("L", L_SHAPE), ("U", U_SHAPE), ("square", SQUARE),
                       ("L đảo chiều", L_SHAPE[::-1]), ("U đảo chiều", U_SHAPE[::-1])):
        tris, forced = triangulate(poly)
        assert len(tris) == len(poly) - 2, (name, len(tris))
        assert abs(tri_area_sum(poly, tris) - abs(signed_area(poly))) < 1e-6, name
        assert forced == 0, name
def tri_normals(verts, idx):
    v = verts[:, :3]
    i = idx.reshape(-1, 3)
    return np.cross(v[i[:, 1]] - v[i[:, 0]], v[i[:, 2]] - v[i[:, 0]])
def test_roof_faces_up():
    for poly in (L_SHAPE, U_SHAPE, L_SHAPE[::-1]):
        v, i, _ = build_roof(poly, 10.0)
        n = tri_normals(v, i)
        assert np.all(n[:, 1] > 0), "mái phải hướng lên (+Y) để không bị face culling loại"
        assert np.allclose(v[:, 1], 10.0)
def test_walls_outward_and_winding():
    for name, outer, holes in (("L", L_SHAPE, []), ("U", U_SHAPE, []), ("L đảo", L_SHAPE[::-1], []),
                               ("vuông+lỗ", SQUARE, [HOLE])):
        edges = rings_to_edges([outer] + holes)
        walls = build_walls(np.array(outer, float), [np.array(h, float) for h in holes], 12.0, 10.8, 3.6)
        for kind, (v, idx) in walls.items():
            n_geo = tri_normals(v, idx)
            for q in range(len(idx) // 6):
                vs = v[q * 4:(q + 1) * 4]
                nrm = vs[0, 3:6]
                mid = vs[:4, :3].mean(axis=0)
                assert np.dot(n_geo[q * 2], nrm) > 0 and np.dot(n_geo[q * 2 + 1], nrm) > 0, (name, q)
                out_pt = (mid + 0.05 * nrm)[[0, 2]]
                in_pt = (mid - 0.05 * nrm)[[0, 2]]
                assert not point_in_edges(out_pt[0], out_pt[1], edges), (name, q, "pháp tuyến hướng vào trong")
                assert point_in_edges(in_pt[0], in_pt[1], edges), (name, q)
def test_wall_uv_per_floor():
    walls = build_walls(np.array(SQUARE, float), [], 25.2, 10.8, 3.6)
    v, _ = walls["main"]
    assert abs(v[:, 7].max() - 7.0) < 1e-5
    assert abs((v[1, 6] - v[0, 6]) - 10.0 / 10.8) < 1e-5
    assert abs(v[:, 6].max() - 40.0 / 10.8) < 1e-5
def test_facade_by_side():
    sq = np.array(SQUARE, float)
    walls = build_walls(sq, [], 10.0, 10.8, 3.6, kind_of_side={"N": "maroon", "S": "white"}, default_kind="plain")
    assert set(walls) == {"maroon", "white", "plain"}
    assert side_of_normal(0, -1) == "N" and side_of_normal(1, 0) == "E"
def test_clean_ring_closed_and_dupes():
    r = clean_ring([(0, 0), (10, 0), (10, 0.001), (10, 10), (0, 10), (0, 0)])
    assert len(r) == 4
def test_extrude_roof_only():
    g = extrude_arrays(SQUARE, [], 3.5, roof_only=True)
    assert g["walls"] == {}
    v, i = g["roof"]
    n = tri_normals(v, i)
    assert (n[:, 1] > 0).sum() == (n[:, 1] < 0).sum() > 0
def _obj(poly, height=12.0):
    p = np.array(poly, float)
    return types.SimpleNamespace(collidable=True, aabb_min=np.array([p[:, 0].min(), 0, p[:, 1].min()], np.float32),
                                 aabb_max=np.array([p[:, 0].max(), height, p[:, 1].max()], np.float32),
                                 footprint_edges=rings_to_edges([poly]))
def test_narrow_phase_notch_is_walkable():
    world = CollisionWorld([_obj(L_SHAPE)])
    y = config.EYE_HEIGHT
    assert not world.collides(np.array([15.0, y, 15.0]))
    assert world.collides(np.array([4.0, y, 4.0]))
    assert world.collides(np.array([8.2, y, 14.0]))
    p = world.move(np.array([15.0, y, 14.0], np.float32), np.array([-10.0, 0, 0.0], np.float32))
    assert 8.0 + config.PLAYER_RADIUS - 0.3 <= p[0] <= 15.0, p
def test_aabb_fallback_without_footprint():
    o = _obj(SQUARE); del o.footprint_edges
    assert CollisionWorld([o]).collides(np.array([5.0, config.EYE_HEIGHT, 5.0]))
def test_resolve_style_and_spawn():
    from scene.osm_campus import resolve_style, find_spawn, _norm
    ov = {"by_name": {_norm("A1"): {"levels": 7, "facade": "white"}}, "by_id": {}}
    big = {"id": "way/1", "name": "a1", "kind": "yes", "levels": None, "height": None, "outer": [[0, 0], [90, 0], [90, 20], [0, 20]]}
    st = resolve_style(big, ov)
    assert st["levels"] == 7 and abs(st["height"] - 7 * 3.6) < 1e-6 and st["facade"] == "white" and st["collidable"]
    kiosk = {"id": "way/2", "name": "", "kind": "yes", "levels": None, "height": None, "outer": [[0, 0], [5, 0], [5, 5], [0, 5]]}
    assert resolve_style(kiosk, ov)["levels"] == 1
    shed = {"id": "way/3", "name": "", "kind": "roof", "levels": None, "height": None, "outer": [[0, 0], [20, 0], [20, 6], [0, 6]]}
    s = resolve_style(shed, ov)
    assert s["roof_only"] and not s["collidable"]
    site = {"id": "way/4", "name": "Tòa nhà hành chính IU (Đang xây dựng)", "kind": "yes", "levels": None, "height": None,
            "outer": [[0, 0], [30, 0], [30, 20], [0, 20]]}
    assert resolve_style(site, ov)["facade"] == "concrete"
    data = {"buildings": [big, site]}
    (sx, sz), yaw = find_spawn(data, [rings_to_edges([big["outer"]])])
    assert not point_in_edges(sx, sz, rings_to_edges([big["outer"]])) and sz > 20, (sx, sz)
def test_skipped_buildings_ignored_by_spawn():
    """Skipped buildings must not move the spawn point away from campus."""
    from scene.osm_campus import find_spawn
    near = {"outer": [[0, 0], [40, 0], [40, 20], [0, 20]]}
    far = {"outer": [[900, 900], [940, 900], [940, 920], [900, 920]]}
    (sx, sz), _ = find_spawn({"buildings": [near]}, [rings_to_edges([near["outer"]])])
    assert abs(sx - 20) < 1 and 20 < sz < 40, (sx, sz)
    (fx, fz), _ = find_spawn({"buildings": [near, far]}, [rings_to_edges([near["outer"]])])
    assert fz > 100, "ví dụ minh họa lỗi nếu KHÔNG lọc skip: điểm xuất phát bị kéo ra xa"
def test_raised_block_bridge():
    """A raised bridge between floors 3 and 4 must include an underside."""
    from graphics.extrude import build_walls, build_underside
    base, top = 2 * 3.6, 4 * 3.6
    walls = build_walls(np.array(SQUARE, float), [], top, 10.8, 3.6, base=base)
    v, _ = walls["main"]
    assert np.allclose(v[:, 1].min(), base) and np.allclose(v[:, 1].max(), top)
    assert abs(v[:, 7].min() - 2.0) < 1e-6 and abs(v[:, 7].max() - 4.0) < 1e-6
    uv, ui = build_underside(np.array(SQUARE, float), base)
    assert np.all(tri_normals(uv, ui)[:, 1] < 0) and np.allclose(uv[:, 1], base)
    raised = _obj(SQUARE, height=top); raised.aabb_min[1] = base
    ground = _obj(SQUARE, height=top)
    y = config.EYE_HEIGHT
    assert not CollisionWorld([raised]).collides(np.array([5.0, y, 5.0]))
    assert CollisionWorld([ground]).collides(np.array([5.0, y, 5.0]))
def test_resolve_style_min_levels():
    from scene.osm_campus import resolve_style, _norm
    ov = {"by_name": {}, "by_id": {"way/9": {"levels": 4, "min_levels": 2, "facade": "maroon"}}}
    b = {"id": "way/9", "name": "", "kind": "university", "levels": None, "height": None,
         "outer": [[0, 0], [4, 0], [4, 22], [0, 22]]}
    st = resolve_style(b, ov)
    assert abs(st["base"] - 7.2) < 1e-9 and abs(st["height"] - 14.4) < 1e-9 and st["facade"] == "maroon"
    plain = resolve_style({**b, "id": "way/10"}, ov)
    assert plain["base"] == 0.0
def test_road_strip_width_and_normals():
    from graphics.roads import polyline_strip
    v, idx = polyline_strip([(0, 0), (20, 0)], 6.0, 0.03, 4.0)
    assert np.allclose(v[:, 1], 0.03)
    assert abs(np.ptp(v[:, 2]) - 6.0) < 1e-5
    assert np.all(tri_normals(v, idx)[:, 1] > 0)
    assert abs(v[:, 7].max() - 20 / 4.0) < 1e-5
def test_road_strip_corner_keeps_width():
    from graphics.roads import polyline_strip
    v, idx = polyline_strip([(0, 0), (10, 0), (10, 10)], 4.0, 0.0, 4.0)
    assert np.all(tri_normals(v, idx)[:, 1] > 0)
    corner = v[2:4]
    assert abs(np.hypot(*(corner[0, [0, 2]] - corner[1, [0, 2]])) - 4.0 * np.sqrt(2)) < 1e-4
    assert np.all(np.isfinite(v))
def test_road_strip_closed_ring_and_uturn():
    from graphics.roads import polyline_strip
    ring = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    v, idx = polyline_strip(ring, 2.0, 0.0, 4.0, closed=True)
    assert np.all(np.isfinite(v)) and np.all(tri_normals(v, idx)[:, 1] > 0)
    v2, _ = polyline_strip([(0, 0), (10, 0), (0, 0.01)], 2.0, 0.0, 4.0)
    assert np.all(np.isfinite(v2)) and np.abs(v2[:, :3]).max() < 100
def test_polygon_flat_faces_up():
    from graphics.roads import polygon_flat
    v, i = polygon_flat(L_SHAPE, 0.04, 4.0)
    assert np.all(tri_normals(v, i)[:, 1] > 0)
    assert abs(sum(abs(signed_area(np.array(v[i.reshape(-1, 3)][:, :, [0, 2]][k]))) for k in range(len(i) // 3))
               - abs(signed_area(L_SHAPE))) < 1e-3
def test_ground_textures_tileable():
    import graphics.procedural_textures as pt
    for name, a in (("grass", pt.grass_array()), ("asphalt", pt.asphalt_array())):
        seam = np.abs(a[:, 0, :3].astype(int) - a[:, -1, :3].astype(int)).mean()
        seam_v = np.abs(a[0, :, :3].astype(int) - a[-1, :, :3].astype(int)).mean()
        inner = np.abs(a[:, 50, :3].astype(int) - a[:, 51, :3].astype(int)).mean()
        assert seam < inner * 1.5 + 2 and seam_v < inner * 1.5 + 2, (name, seam, seam_v, inner)
GRID = [{"id": "r1", "kind": "service", "points": [[0, 0], [50, 0], [100, 0], [100, 100]]},
        {"id": "r2", "kind": "service", "points": [[100, 100], [0, 100], [0, 0]]},
        {"id": "r3", "kind": "footway", "points": [[0, 0], [50, 0], [50, 50]]},
        {"id": "big", "kind": "tertiary", "points": [[0, 0], [0, 100]]}]
def test_graph_junction_snap_and_dijkstra():
    from scene.tour_route import build_graph, snap_to_graph, shortest_path
    adj = build_graph(GRID)
    assert (50.0, 0.0) in adj and set(adj[(0.0, 0.0)]) == {(50.0, 0.0), (0.0, 100.0)}
    k, d = snap_to_graph(adj, (30.0, 4.0))
    assert k == (30.0, 0.0) and abs(d - 4.0) < 1e-9 and (30.0, 0.0) in adj
    path = shortest_path(adj, (30.0, 0.0), (50.0, 50.0))
    assert path[0] == (30.0, 0.0) and path[-1] == (50.0, 50.0)
    assert abs(sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:])) - 70.0) < 1e-6
def test_resample_and_smooth():
    from scene.tour_route import resample, smooth
    sq = np.array([(0, 0), (40, 0), (40, 40), (0, 40)], float)
    r = resample(sq, 2.0, closed=True)
    step = np.hypot(*(np.roll(r, -1, 0) - r).T)
    assert abs(step.mean() - 2.0) < 0.2 and len(r) == 80
    s = smooth(r, 10.0, 2.0)
    assert np.hypot(*s[0]) > 1.0
    assert abs(np.ptp(s[:, 0]) - 40) < 8
def test_push_from_obstacles_keeps_clearance():
    from scene.tour_route import push_from_obstacles, min_clearance
    wall = rings_to_edges([[(20, -30), (30, -30), (30, 30), (20, 30)]])
    path = np.array([(x, 0.0) for x in np.arange(0, 50, 1.0)])
    out = push_from_obstacles(path, [wall], 2.0)
    assert min_clearance(out, [wall]) > 1.7, min_clearance(out, [wall])
    far = np.array([(0.0, 50.0), (5.0, 50.0)])
    assert np.allclose(push_from_obstacles(far, [wall], 2.0), far, atol=1e-6)
def test_build_tour_toy():
    from scene.tour_route import build_tour
    building = rings_to_edges([[(40, 40), (60, 40), (60, 60), (40, 60)]])
    spec = {"clearance": 2.0, "corner_smoothing": 6.0, "stops": [
        {"name": "A", "snap": [10, 0]}, {"name": "B", "snap": [100, 50]},
        {"name": "Sân", "free": [[80, 50], [60, 80], [80, 90]]}, {"name": "C", "snap": [50, 100]}]}
    tour = build_tour(GRID, spec, [building], eye_height=1.7)
    wp = np.array(tour["waypoints"])
    assert np.allclose(wp[:, 1], 1.7) and len(wp) >= 20
    assert tour["clearance"] > 1.5 and not tour["warnings"]
    prog = [p for _, p in tour["stops"]]
    assert prog == sorted(prog) and 0.0 <= prog[0] < prog[-1] < 1.0
    assert 300 < tour["length"] < 450
def test_tour_waypoints_compatible_with_catmull_rom():
    """Generated waypoints must be usable by a closed, constant-speed path."""
    from scene.tour_route import build_tour
    spec = {"stops": [{"snap": [0, 0]}, {"snap": [100, 0]}, {"snap": [100, 100]}, {"snap": [0, 100]}]}
    wp = build_tour(GRID, spec, [], 1.7)["waypoints"]
    try:
        from scene.path import CatmullRomPath
    except ImportError:
        return
    path = CatmullRomPath(wp, closed=True)
    pts = np.array([path.point_at_distance(i * 2.0) for i in range(int(path.total_length / 2))])
    steps = np.hypot(*(pts[1:] - pts[:-1])[:, [0, 2]].T)
    assert steps.max() < 2.2 and steps.min() > 1.7
def _wall_quads(v, idx):
    """Return the center and normal vectors for each rectangular wall quad."""
    out = []
    ng = tri_normals(v, idx)
    for q in range(len(idx) // 6):
        vs = v[q * 4:(q + 1) * 4]
        out.append((vs[:, :3].mean(axis=0), vs[0, 3:6], ng[q * 2], ng[q * 2 + 1]))
    return out
def test_ground_floor_split_and_doors():
    from graphics.extrude import build_walls_ground
    g = build_walls_ground(np.array(SQUARE, float), 25.2, 10.8, 3.6, "maroon", "white_plain",
                           doors=[{"at": [5.0, -0.4], "width": 2.6}], interior=True)
    assert set(g["walls"]) == {"white_plain", "maroon"}
    vu, _ = g["walls"]["maroon"]
    assert abs(vu[:, 1].min() - 3.6) < 1e-6 and abs(vu[:, 7].min() - 1.0) < 1e-6
    d = g["doors"][0]
    assert d["ok"] and abs(d["width"] - 2.6) < 1e-6 and abs(d["snap_dist"] - 0.4) < 1e-6
    assert abs(d["at"][0] - 5.0) < 1e-6 and abs(d["at"][1]) < 1e-6
    total = sum(np.hypot(e[2] - e[0], e[3] - e[1]) for e in g["wall_edges"])
    assert abs(total - (40.0 - 2.6)) < 1e-6
    vg, _ = g["walls"]["white_plain"]
    assert np.any((np.abs(vg[:, 1] - 2.7) < 1e-6)) and np.any(np.abs(vg[:, 1] - 3.6) < 1e-6)
def test_interior_faces_point_inward_and_winding():
    from graphics.extrude import build_walls_ground
    edges = rings_to_edges([L_SHAPE])
    g = build_walls_ground(np.array(L_SHAPE, float), 10.8, 10.8, 3.6, "maroon", "white_plain",
                           doors=[{"at": [10.0, 0.0], "width": 2.6}], interior=True)
    iv, ii = g["interior"]["walls"]
    for mid, nrm, n1, n2 in _wall_quads(iv, ii):
        assert np.dot(n1, nrm) > 0 and np.dot(n2, nrm) > 0
        inside = mid[[0, 2]] + 0.05 * nrm[[0, 2]]
        outside = mid[[0, 2]] - 0.05 * nrm[[0, 2]]
        assert point_in_edges(inside[0], inside[1], edges) and not point_in_edges(outside[0], outside[1], edges)
    fv, fi = g["interior"]["floor"]
    cv, ci = g["interior"]["ceiling"]
    assert np.all(tri_normals(fv, fi)[:, 1] > 0) and np.all(tri_normals(cv, ci)[:, 1] < 0)
    ev, ei = g["walls"]["white_plain"]
    for mid, nrm, n1, n2 in _wall_quads(ev, ei):
        out = mid[[0, 2]] + 0.05 * nrm[[0, 2]]
        assert not point_in_edges(out[0], out[1], edges) and np.dot(n1, nrm) > 0
def test_door_too_far_is_reported():
    from graphics.extrude import build_walls_ground
    g = build_walls_ground(np.array(SQUARE, float), 7.2, 10.8, 3.6, "maroon", "white_plain",
                           doors=[{"at": [5, 30], "width": 2.6}, {"at": [0.5, 5.0], "width": 50.0}])
    assert g["doors"][0]["ok"] is False
    assert g["doors"][1]["ok"] and g["doors"][1]["width"] < 10
def test_enterable_collision_gap_inside_and_walls():
    from graphics.extrude import build_walls_ground
    g = build_walls_ground(np.array(SQUARE, float), 7.2, 10.8, 3.6, "maroon", "white_plain",
                           doors=[{"at": [5.0, 0.0], "width": 2.6}], interior=True)
    obj = types.SimpleNamespace(collidable=True, aabb_min=np.array([0, 0, 0], np.float32),
                                aabb_max=np.array([10, 7.2, 10], np.float32), wall_edges=g["wall_edges"])
    w = CollisionWorld([obj]); y = config.EYE_HEIGHT
    assert not w.collides(np.array([5.0, y, 5.0]))
    assert w.collides(np.array([2.0, y, 0.1]))
    assert not w.collides(np.array([5.0, y, 0.0]))
    p = w.move(np.array([5.0, y, -3.0], np.float32), np.array([0, 0, 8.0], np.float32))
    assert p[2] > 4.9
    p = w.move(np.array([2.0, y, -3.0], np.float32), np.array([0, 0, 8.0], np.float32))
    assert p[2] < 0.0
def test_collider_boxes_block_and_pass_over():
    box = types.SimpleNamespace(collidable=False, collider_boxes=np.array([[4.8, 0.0, 4.8, 5.2, 3.0, 5.2]], np.float32))
    w = CollisionWorld([box]); y = config.EYE_HEIGHT
    assert w.collides(np.array([5.0, y, 5.0])) and not w.collides(np.array([8.0, y, 5.0]))
    low = types.SimpleNamespace(collidable=False, collider_boxes=np.array([[0, 0, 0, 5, 0.45, 5]], np.float32))
    assert CollisionWorld([low]).collides(np.array([2.0, y, 2.0]))
def test_style_door_merge_and_extra_buildings():
    from scene.osm_campus import resolve_style, _norm, _manual_building
    ov = {"by_name": {_norm("A1"): {"levels": 7, "doors": [{"at": [0, 0]}], "ground_facade": "white_plain", "enterable": True}},
          "by_id": {"way/1": {"doors": [{"at": [1, 1]}]}},
          "connect_doors": {"way/1": [{"at": [2, 2], "width": 4}]}, "extra_buildings": []}
    b = {"id": "way/1", "name": "A1", "kind": "university", "levels": None, "height": None,
         "outer": [[0, 0], [40, 0], [40, 20], [0, 20]]}
    st = resolve_style(b, ov)
    assert len(st["doors"]) == 3 and st["enterable"] and st["ground_facade"] == "white_plain" and st["levels"] == 7
    mb = _manual_building(0, {"id": "manual/x", "name": "Căn tin", "outer": SQUARE})
    assert mb["id"] == "manual/x" and resolve_style(mb, ov)["levels"] == 2
def test_plant_geometry_outward_and_boxes():
    from graphics.vegetation import build_plants, sphere_part, TRUNK_BOX_HEIGHT
    v, i = sphere_part(0, 0, 0, 1.0)
    tri = v[i.reshape(-1, 3)]
    geo = np.cross(tri[:, 1, :3] - tri[:, 0, :3], tri[:, 2, :3] - tri[:, 0, :3])
    assert np.all((geo * tri[:, 0, 3:6]).sum(axis=1) >= -1e-9)
    trunk, canopy, boxes = build_plants([(0, 0, "tree"), (10, 0, "small"), (20, 0, "shrub")], seed=3)
    assert trunk is not None and canopy is not None and len(boxes) == 2
    assert np.allclose(boxes[:, 4], TRUNK_BOX_HEIGHT) and canopy[0][:, 7].min() >= 0 and canopy[0][:, 7].max() <= 1
def test_vegetation_respects_constraints():
    from scene.landscape import Constraints, generate_vegetation, dist_points_segments, points_in_edges
    bld = rings_to_edges([[(0, 0), (40, 0), (40, 20), (0, 20)]])
    roads = [{"kind": "service", "points": [[-60, 40], [100, 40]]}]
    styles = {"service": ("asphalt", 5.5, 0.04, 4.0, False)}
    segs = np.array([[-60, 40, 100, 40, 2.75]], float)
    door = (20.0, 20.0, 0.0, 1.0)
    tour = np.array([[x, 30.0] for x in range(-20, 60, 2)], float)
    cons = Constraints([bld], segs, [], [door], tour, (10.0, 60.0))
    spec = {"seed": 5, "tree_zones": [{"name": "z", "polygon": [[2, 22], [38, 22], [38, 36], [2, 36]], "density": 6.0}],
            "street_trees": {"enabled": True, "spacing": 10.0, "margin": 40.0, "max": 40},
            "perimeter": {"enabled": True, "count": 60, "margin": 40.0}}
    plants, stats = generate_vegetation(spec, cons, roads, styles, (0, 0, 40, 20))
    assert len(plants) > 20 and stats.get("perimeter", 0) > 0
    P = np.array([[x, z] for x, z, k in plants])
    assert not points_in_edges(P, bld).any()
    assert (dist_points_segments(P, bld) >= 0.79).all()
    assert (np.abs(P[:, 1] - 40) >= 2.75).all()
    assert (np.hypot(*(P - np.array([20.0, 20.0])).T) >= 3.2).all()
    assert not ((np.abs(P[:, 0] - 20.0) < 2.0) & (P[:, 1] > 19.5) & (P[:, 1] < 29)).any()
    assert (np.hypot(*(P - np.array([10.0, 60.0])).T) >= 6.0).all()
    zone_pts = P[(P[:, 0] > 2) & (P[:, 0] < 38) & (P[:, 1] > 22) & (P[:, 1] < 36)]
    assert len(zone_pts) >= 8
def test_fence_gate_gap_and_collision_boxes():
    from graphics.fence import fence_for_polygon, FENCE_HEIGHT
    ring = [(0, 0), (20, 0), (20, 10), (0, 10)]
    (v, i), boxes, rep = fence_for_polygon(ring, gates=[{"at": [10, 0.2], "width": 4.0}])
    assert rep[0]["ok"] and np.all(boxes[:, 4] <= FENCE_HEIGHT + 1e-6) and FENCE_HEIGHT < 0.6
    assert len(v) > 0 and len(i) % 3 == 0
    cover = [b for b in boxes if b[2] < 0.5 and b[5] < 0.5 and b[0] < 11.8 and b[3] > 8.2]
    assert not cover
    w = CollisionWorld([types.SimpleNamespace(collidable=False, collider_boxes=boxes)]); y = config.EYE_HEIGHT
    assert w.collides(np.array([3.0, y, 0.0])) and not w.collides(np.array([10.0, y, 0.0]))
    assert (v[:, 1].max() <= FENCE_HEIGHT + 1e-6)
    _, boxes2, rep2 = fence_for_polygon(ring, gates=[{"at": [100, 100], "width": 4.0}])
    assert rep2[0]["ok"] is False
def test_lift_shifts_geometry_and_adds_plinth():
    from graphics.extrude import extrude_arrays
    lift, H = 0.9, 25.2
    g = extrude_arrays(np.array(SQUARE, float), [], H, ground_kind="white_plain", default_kind="maroon",
                       doors=[{"at": [5.0, 0.0], "width": 2.6}], enterable=True, lift=lift)
    vw, _ = g["walls"]["white_plain"]
    assert abs(vw[:, 1].min()) < 1e-6
    assert np.any(np.abs(vw[:, 1] - lift) < 1e-6)
    vm, _ = g["walls"]["maroon"]
    assert abs(vm[:, 1].min() - (lift + 3.6)) < 1e-6 and abs(vm[:, 1].max() - (lift + H)) < 1e-6
    assert abs(vm[:, 7].min() - 1.0) < 1e-6
    assert abs(g["roof"][0][:, 1].max() - (lift + H)) < 1e-6
    fl = g["interior"]["floor"][0]
    assert abs(fl[0, 1] - (lift + 0.02)) < 1e-6 and abs(g["interior"]["ceiling"][0][0, 1] - (lift + 3.6)) < 1e-6
    plinth = vw[(vw[:, 1] < lift - 1e-6) | (np.abs(vw[:, 1] - lift) < 1e-6)]
    assert len(plinth) >= 16
    g2 = extrude_arrays(np.array(SQUARE, float), [], 7.2, base=3.6, default_kind="maroon", lift=lift)
    assert abs(g2["walls"]["maroon"][0][:, 1].min() - (3.6 + lift)) < 1e-6
    assert abs(g2["bottom"][0][:, 1].max() - (3.6 + lift)) < 1e-6
def test_stair_steps_and_heights():
    from graphics.stairs import Stair, LANDING, TREAD, RISER
    s = Stair(center=(0.0, 0.0), normal=(0.0, 1.0), width=6.0, lift=0.9)
    assert s.steps == 6 and abs(s.riser - 0.15) < 1e-9 and abs(s.depth - (LANDING + 5 * TREAD)) < 1e-9
    assert abs(s.height_at(0.0, 0.3) - 0.9) < 1e-9
    assert abs(s.height_at(0.0, LANDING + 0.5 * TREAD) - 0.75) < 1e-9
    assert abs(s.height_at(0.0, s.depth - 0.01) - 0.15) < 1e-9
    assert s.height_at(0.0, s.depth + 0.1) == 0.0 and s.height_at(3.5, 0.5) == 0.0
    assert s.height_at(0.0, -0.3) == 0.0
    parts = s.parts()
    assert len(parts) == 6
    tops = sorted((round(float(v[:, 1].max()), 3) for v, _ in parts), reverse=True)
    assert tops == [0.9, 0.75, 0.6, 0.45, 0.3, 0.15]
    fp = np.array(s.footprint())
    assert abs(abs(signed_area(fp)) - 6.0 * s.depth) < 1e-6
def test_stairs_face_outward_from_door_report():
    from graphics.extrude import build_walls_ground
    from graphics.stairs import Stair
    g = build_walls_ground(np.array(L_SHAPE, float), 25.2, 10.8, 3.6, "maroon", "white_plain",
                           doors=[{"at": [10.0, 0.0], "width": 2.6}, {"at": [0.0, 12.0], "width": 2.6}], interior=True)
    edges = rings_to_edges([L_SHAPE])
    for d in g["doors"]:
        s = Stair(d["at"], d["normal"], 4.0, 0.9)
        fp = np.array(s.footprint())
        far = fp[2:]
        assert not any(point_in_edges(x, z, edges) for x, z in far), "cầu thang phải nằm NGOÀI tòa nhà"
        centre_out = np.array(d["at"]) + np.array(d["normal"]) * (s.depth / 2)
        assert not point_in_edges(centre_out[0], centre_out[1], edges)
def test_collinear_run_for_stair_width():
    from graphics.extrude import build_walls_ground
    ring = [(0, 0), (30, 0), (50, 0), (50, 20), (0, 20)]
    g = build_walls_ground(np.array(ring, float), 7.2, 10.8, 3.6, "maroon", "white_plain",
                           doors=[{"at": [40.0, 0.0], "width": 2.6}])
    d = g["doors"][0]
    back, fwd = d["run"]
    assert abs(d["edge_len"] + back + fwd - 50.0) < 1e-6
def test_ray_road_hit_and_oblique():
    from scene.osm_campus import ray_road_hit
    segs = np.array([[-50, 20, 50, 20, 2.75]], float)
    assert abs(ray_road_hit(np.array([0.0, 0.0]), np.array([0.0, 1.0]), segs, 30) - 17.25) < 1e-9
    assert ray_road_hit(np.array([0.0, 0.0]), np.array([0.0, 1.0]), segs, 10) is None
    assert ray_road_hit(np.array([0.0, 0.0]), np.array([0.0, -1.0]), segs, 30) is None
    assert ray_road_hit(np.array([200.0, 0.0]), np.array([0.0, 1.0]), segs, 30) is None
def test_benches_random_but_valid():
    from graphics.furniture import place_benches, benches_mesh, bench_box, BENCH_LENGTH
    poly = [[-85, -11.5], [7.5, -11.5], [7.5, -2.0], [-85, -2.0]]
    avoid = [(-80.7, -6.8, 4.0), (-34.2, -6.8, 4.0), (-29.0, -6.75, 5.0)]
    a = place_benches(poly, 5, 10.0, avoid, np.random.default_rng(7))
    b = place_benches(poly, 5, 10.0, avoid, np.random.default_rng(7))
    c = place_benches(poly, 5, 10.0, avoid, np.random.default_rng(8))
    assert len(a) == 5 and a == b and a != c
    for i, (x, z, ang, bs) in enumerate(a):
        assert -85 < x < 7.5 and (z < -9.0 or z > -4.5)
        assert all(np.hypot(x - ax, z - az) >= r for ax, az, r in avoid)
        assert all(np.hypot(x - o[0], z - o[1]) >= 10.0 for o in a[:i])
        assert (bs == 1) == (z > -6.75)
    (v, idx), boxes = benches_mesh(a)
    assert len(boxes) == 5 and BENCH_LENGTH >= 1.8
    assert np.allclose(boxes[:, 4], 0.95) and (v[:, 1].max() <= 0.95 + 1e-6)
    ng = tri_normals(v, idx)
    nv = v[idx.reshape(-1, 3)][:, 0, 3:6]
    assert np.all((ng * nv).sum(axis=1) >= -1e-9)
def test_parking_roof_geometry():
    from graphics.extrude import extrude_arrays
    poly = [(0, 0), (40, 0), (40, 20), (0, 20)]
    g = extrude_arrays(poly, [], 3.45, base=3.2, default_kind="roof")
    assert abs(g["bottom"][0][:, 1].max() - 3.2) < 1e-6 and abs(g["roof"][0][:, 1].max() - 3.45) < 1e-6
    assert np.all(tri_normals(*g["bottom"])[:, 1] < 0) and np.all(tri_normals(*g["roof"])[:, 1] > 0)
    v, _ = g["walls"]["roof"]
    assert abs(v[:, 1].min() - 3.2) < 1e-6 and abs(v[:, 1].max() - 3.45) < 1e-6
def test_oriented_box_every_face_outward():
    """Every side and cap must face outward."""
    from graphics.fence import oriented_box
    for ang in (0.0, 0.7, 2.0, -1.2):
        v, i = oriented_box(3.0, -2.0, 0.5, 1.5, 2.0, 1.0, ang)
        tri = v[i.reshape(-1, 3)]
        geo = np.cross(tri[:, 1, :3] - tri[:, 0, :3], tri[:, 2, :3] - tri[:, 0, :3])
        assert np.all((geo * tri[:, 0, 3:6]).sum(axis=1) > 0), ang
        assert len(i) // 3 == 12
        centre = np.array([3.0, 1.0, -2.0])
        for k in range(len(tri)):
            assert np.dot(geo[k], tri[k][:, :3].mean(axis=0) - centre) > 0
if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1; print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
