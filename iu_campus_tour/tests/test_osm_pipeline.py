"""Kiểm thử đường ống OSM (không cần GPU):  python tests/test_osm_pipeline.py"""

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

L_SHAPE = [(0, 0), (20, 0), (20, 8), (8, 8), (8, 20), (0, 20)]                       # chữ L (lõm)
U_SHAPE = [(0, 0), (30, 0), (30, 20), (22, 20), (22, 8), (8, 8), (8, 20), (0, 20)]  # chữ U (lõm, có sân)
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
        assert abs(tri_area_sum(poly, tris) - abs(signed_area(poly))) < 1e-6, name   # tổng diện tích tam giác = diện tích đa giác
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
                # (1) pháp tuyến lưu trong đỉnh phải cùng hướng pháp tuyến hình học (thứ tự đỉnh đúng)
                assert np.dot(n_geo[q * 2], nrm) > 0 and np.dot(n_geo[q * 2 + 1], nrm) > 0, (name, q)
                # (2) bước ra theo pháp tuyến phải ra khỏi khối đặc, bước ngược lại phải vào trong khối đặc
                out_pt = (mid + 0.05 * nrm)[[0, 2]]
                in_pt = (mid - 0.05 * nrm)[[0, 2]]
                assert not point_in_edges(out_pt[0], out_pt[1], edges), (name, q, "pháp tuyến hướng vào trong")
                assert point_in_edges(in_pt[0], in_pt[1], edges), (name, q)


def test_wall_uv_per_floor():
    walls = build_walls(np.array(SQUARE, float), [], 25.2, 10.8, 3.6)        # 7 tầng
    v, _ = walls["main"]
    assert abs(v[:, 7].max() - 7.0) < 1e-5                                   # v chạy 0..7 -> mỗi tầng đúng 1 ô texture
    assert abs((v[1, 6] - v[0, 6]) - 10.0 / 10.8) < 1e-5                     # cạnh 10 m -> u tăng 10/10.8
    assert abs(v[:, 6].max() - 40.0 / 10.8) < 1e-5                           # u cộng dồn quanh chu vi 40 m (liền mạch)


def test_facade_by_side():
    sq = np.array(SQUARE, float)
    walls = build_walls(sq, [], 10.0, 10.8, 3.6, kind_of_side={"N": "maroon", "S": "white"}, default_kind="plain")
    assert set(walls) == {"maroon", "white", "plain"}
    assert side_of_normal(0, -1) == "N" and side_of_normal(1, 0) == "E"      # Bắc = -Z, Đông = +X


def test_clean_ring_closed_and_dupes():
    r = clean_ring([(0, 0), (10, 0), (10, 0.001), (10, 10), (0, 10), (0, 0)])
    assert len(r) == 4


def test_extrude_roof_only():
    g = extrude_arrays(SQUARE, [], 3.5, roof_only=True)
    assert g["walls"] == {}
    v, i = g["roof"]
    n = tri_normals(v, i)
    assert (n[:, 1] > 0).sum() == (n[:, 1] < 0).sum() > 0                    # mặt trên và mặt dưới


def _obj(poly, height=12.0):
    p = np.array(poly, float)
    return types.SimpleNamespace(collidable=True, aabb_min=np.array([p[:, 0].min(), 0, p[:, 1].min()], np.float32),
                                 aabb_max=np.array([p[:, 0].max(), height, p[:, 1].max()], np.float32),
                                 footprint_edges=rings_to_edges([poly]))


def test_narrow_phase_notch_is_walkable():
    world = CollisionWorld([_obj(L_SHAPE)])
    y = config.EYE_HEIGHT
    assert not world.collides(np.array([15.0, y, 15.0]))          # trong hộp bao nhưng ngoài chữ L -> đi được
    assert world.collides(np.array([4.0, y, 4.0]))                # trong thân chữ L -> chặn
    assert world.collides(np.array([8.2, y, 14.0]))               # sát tường (cách < bán kính) -> chặn
    # đi từ khoảng trống vào tường rồi trượt dọc tường (không bị kẹt)
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
    """Tòa nhà xa bị skip không được kéo điểm xuất phát ra xa khu trường."""
    from scene.osm_campus import find_spawn
    near = {"outer": [[0, 0], [40, 0], [40, 20], [0, 20]]}
    far = {"outer": [[900, 900], [940, 900], [940, 920], [900, 920]]}
    (sx, sz), _ = find_spawn({"buildings": [near]}, [rings_to_edges([near["outer"]])])
    assert abs(sx - 20) < 1 and 20 < sz < 40, (sx, sz)
    (fx, fz), _ = find_spawn({"buildings": [near, far]}, [rings_to_edges([near["outer"]])])
    assert fz > 100, "ví dụ minh họa lỗi nếu KHÔNG lọc skip: điểm xuất phát bị kéo ra xa"


def test_raised_block_bridge():
    """Cầu nối tầng 3-4: tường từ 7.2 m đến 14.4 m, có mặt dưới, người chơi đi bên dưới được."""
    from graphics.extrude import build_walls, build_underside
    base, top = 2 * 3.6, 4 * 3.6
    walls = build_walls(np.array(SQUARE, float), [], top, 10.8, 3.6, base=base)
    v, _ = walls["main"]
    assert np.allclose(v[:, 1].min(), base) and np.allclose(v[:, 1].max(), top)
    assert abs(v[:, 7].min() - 2.0) < 1e-6 and abs(v[:, 7].max() - 4.0) < 1e-6       # 2 tầng = 2 ô texture, thẳng hàng với tòa kế bên
    uv, ui = build_underside(np.array(SQUARE, float), base)
    assert np.all(tri_normals(uv, ui)[:, 1] < 0) and np.allclose(uv[:, 1], base)       # mặt dưới hướng xuống
    # Va chạm: hộp bắt đầu ở y = 7.2 nên không chạm người cao 1.9 m; hộp thường chạm
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


# ------------------------------------------------------------ đường và lối đi
def test_road_strip_width_and_normals():
    from graphics.roads import polyline_strip
    v, idx = polyline_strip([(0, 0), (20, 0)], 6.0, 0.03, 4.0)
    assert np.allclose(v[:, 1], 0.03)
    assert abs(np.ptp(v[:, 2]) - 6.0) < 1e-5                      # bề rộng 6 m theo trục Z
    assert np.all(tri_normals(v, idx)[:, 1] > 0)                  # mặt trên hướng lên (không bị face culling)
    assert abs(v[:, 7].max() - 20 / 4.0) < 1e-5                   # v = quãng đường / tile


def test_road_strip_corner_keeps_width():
    from graphics.roads import polyline_strip
    v, idx = polyline_strip([(0, 0), (10, 0), (10, 10)], 4.0, 0.0, 4.0)
    assert np.all(tri_normals(v, idx)[:, 1] > 0)
    # Tại góc 90° đỉnh miter cách trục đường 2*sqrt(2)/2 * sqrt(2)... kiểm tra: hai mép cách nhau sqrt(2)*w ở đỉnh góc
    corner = v[2:4]
    assert abs(np.hypot(*(corner[0, [0, 2]] - corner[1, [0, 2]])) - 4.0 * np.sqrt(2)) < 1e-4
    assert np.all(np.isfinite(v))


def test_road_strip_closed_ring_and_uturn():
    from graphics.roads import polyline_strip
    ring = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    v, idx = polyline_strip(ring, 2.0, 0.0, 4.0, closed=True)
    assert np.all(np.isfinite(v)) and np.all(tri_normals(v, idx)[:, 1] > 0)
    v2, _ = polyline_strip([(0, 0), (10, 0), (0, 0.01)], 2.0, 0.0, 4.0)     # quay đầu 180°: miter bị chặn, không nổ
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
        assert seam < inner * 1.5 + 2 and seam_v < inner * 1.5 + 2, (name, seam, seam_v, inner)   # mép nối liền như giữa ảnh


# ------------------------------------------------------------------ tuyến tour
GRID = [{"id": "r1", "kind": "service", "points": [[0, 0], [50, 0], [100, 0], [100, 100]]},   # (50,0): giao lộ với r3
        {"id": "r2", "kind": "service", "points": [[100, 100], [0, 100], [0, 0]]},
        {"id": "r3", "kind": "footway", "points": [[0, 0], [50, 0], [50, 50]]},          # nhánh phụ tới (50,50)
        {"id": "big", "kind": "tertiary", "points": [[0, 0], [0, 100]]}]                 # đường lớn: tour không đi


def test_graph_junction_snap_and_dijkstra():
    from scene.tour_route import build_graph, snap_to_graph, shortest_path
    adj = build_graph(GRID)
    assert (50.0, 0.0) in adj and set(adj[(0.0, 0.0)]) == {(50.0, 0.0), (0.0, 100.0)}   # đường lớn "tertiary" bị loại; r1 và r3 dùng chung cạnh (0,0)-(50,0)
    k, d = snap_to_graph(adj, (30.0, 4.0))                                    # chiếu lên cạnh r1, chia đôi cạnh
    assert k == (30.0, 0.0) and abs(d - 4.0) < 1e-9 and (30.0, 0.0) in adj
    path = shortest_path(adj, (30.0, 0.0), (50.0, 50.0))
    assert path[0] == (30.0, 0.0) and path[-1] == (50.0, 50.0)
    assert abs(sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:])) - 70.0) < 1e-6   # 20 + 50 qua footway


def test_resample_and_smooth():
    from scene.tour_route import resample, smooth
    sq = np.array([(0, 0), (40, 0), (40, 40), (0, 40)], float)
    r = resample(sq, 2.0, closed=True)
    step = np.hypot(*(np.roll(r, -1, 0) - r).T)
    assert abs(step.mean() - 2.0) < 0.2 and len(r) == 80
    s = smooth(r, 10.0, 2.0)
    assert np.hypot(*s[0]) > 1.0                                              # góc (0,0) đã bị bo vào trong
    assert abs(np.ptp(s[:, 0]) - 40) < 8


def test_push_from_obstacles_keeps_clearance():
    from scene.tour_route import push_from_obstacles, min_clearance
    wall = rings_to_edges([[(20, -30), (30, -30), (30, 30), (20, 30)]])       # tòa nhà 10x60
    path = np.array([(x, 0.0) for x in np.arange(0, 50, 1.0)])                # đường đi XUYÊN tòa nhà
    out = push_from_obstacles(path, [wall], 2.0)
    assert min_clearance(out, [wall]) > 1.7, min_clearance(out, [wall])
    far = np.array([(0.0, 50.0), (5.0, 50.0)])
    assert np.allclose(push_from_obstacles(far, [wall], 2.0), far, atol=1e-6)  # điểm đã đủ xa: giữ nguyên


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
    assert prog == sorted(prog) and 0.0 <= prog[0] < prog[-1] < 1.0           # điểm dừng theo đúng thứ tự dọc tuyến
    assert 300 < tour["length"] < 450


def test_tour_waypoints_compatible_with_catmull_rom():
    """Waypoint tạo ra phải dùng được với CatmullRomPath (khép kín) và đi với tốc độ đều."""
    from scene.tour_route import build_tour
    spec = {"stops": [{"snap": [0, 0]}, {"snap": [100, 0]}, {"snap": [100, 100]}, {"snap": [0, 100]}]}
    wp = build_tour(GRID, spec, [], 1.7)["waypoints"]
    try:
        from scene.path import CatmullRomPath
    except ImportError:
        return                                                               # bản Phase 3 chưa có path.py: bỏ qua
    path = CatmullRomPath(wp, closed=True)
    pts = np.array([path.point_at_distance(i * 2.0) for i in range(int(path.total_length / 2))])
    steps = np.hypot(*(pts[1:] - pts[:-1])[:, [0, 2]].T)
    assert steps.max() < 2.2 and steps.min() > 1.7


# ------------------------------------------ tầng trệt, cửa ra vào, nội thất
def _wall_quads(v, idx):
    """Với mỗi hình chữ nhật (4 đỉnh, 2 tam giác) trả về (tâm, pháp_tuyến_đỉnh, pháp_tuyến_hình_học)."""
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
    assert set(g["walls"]) == {"white_plain", "maroon"}                     # tầng trệt trắng, các tầng trên đỏ đô
    vu, _ = g["walls"]["maroon"]
    assert abs(vu[:, 1].min() - 3.6) < 1e-6 and abs(vu[:, 7].min() - 1.0) < 1e-6   # tầng trên bắt đầu từ y = 3.6, v = 1 (thẳng hàng tầng)
    d = g["doors"][0]
    assert d["ok"] and abs(d["width"] - 2.6) < 1e-6 and abs(d["snap_dist"] - 0.4) < 1e-6
    assert abs(d["at"][0] - 5.0) < 1e-6 and abs(d["at"][1]) < 1e-6          # cửa đặt giữa, nằm trên cạnh z = 0
    total = sum(np.hypot(e[2] - e[0], e[3] - e[1]) for e in g["wall_edges"])
    assert abs(total - (40.0 - 2.6)) < 1e-6                                 # chu vi trừ đúng bề rộng ô cửa
    # lanh tô phía trên cửa: có mặt tường ở độ cao 2.7..3.6 ngay trên ô cửa
    vg, _ = g["walls"]["white_plain"]
    assert np.any((np.abs(vg[:, 1] - 2.7) < 1e-6)) and np.any(np.abs(vg[:, 1] - 3.6) < 1e-6)


def test_interior_faces_point_inward_and_winding():
    from graphics.extrude import build_walls_ground
    edges = rings_to_edges([L_SHAPE])
    g = build_walls_ground(np.array(L_SHAPE, float), 10.8, 10.8, 3.6, "maroon", "white_plain",
                           doors=[{"at": [10.0, 0.0], "width": 2.6}], interior=True)
    iv, ii = g["interior"]["walls"]
    for mid, nrm, n1, n2 in _wall_quads(iv, ii):
        assert np.dot(n1, nrm) > 0 and np.dot(n2, nrm) > 0                    # thứ tự đỉnh khớp pháp tuyến lưu trong đỉnh
        inside = mid[[0, 2]] + 0.05 * nrm[[0, 2]]
        outside = mid[[0, 2]] - 0.05 * nrm[[0, 2]]
        assert point_in_edges(inside[0], inside[1], edges) and not point_in_edges(outside[0], outside[1], edges)  # mặt trong hướng VÀO phòng
    fv, fi = g["interior"]["floor"]
    cv, ci = g["interior"]["ceiling"]
    assert np.all(tri_normals(fv, fi)[:, 1] > 0) and np.all(tri_normals(cv, ci)[:, 1] < 0)   # sàn hướng lên, trần hướng xuống
    ev, ei = g["walls"]["white_plain"]
    for mid, nrm, n1, n2 in _wall_quads(ev, ei):                              # mặt ngoài tầng trệt hướng RA
        out = mid[[0, 2]] + 0.05 * nrm[[0, 2]]
        assert not point_in_edges(out[0], out[1], edges) and np.dot(n1, nrm) > 0


def test_door_too_far_is_reported():
    from graphics.extrude import build_walls_ground
    g = build_walls_ground(np.array(SQUARE, float), 7.2, 10.8, 3.6, "maroon", "white_plain",
                           doors=[{"at": [5, 30], "width": 2.6}, {"at": [0.5, 5.0], "width": 50.0}])
    assert g["doors"][0]["ok"] is False                                        # cách tường 20 m
    assert g["doors"][1]["ok"] and g["doors"][1]["width"] < 10                  # bề rộng bị chặn bởi độ dài cạnh


def test_enterable_collision_gap_inside_and_walls():
    from graphics.extrude import build_walls_ground
    g = build_walls_ground(np.array(SQUARE, float), 7.2, 10.8, 3.6, "maroon", "white_plain",
                           doors=[{"at": [5.0, 0.0], "width": 2.6}], interior=True)
    obj = types.SimpleNamespace(collidable=True, aabb_min=np.array([0, 0, 0], np.float32),
                                aabb_max=np.array([10, 7.2, 10], np.float32), wall_edges=g["wall_edges"])
    w = CollisionWorld([obj]); y = config.EYE_HEIGHT
    assert not w.collides(np.array([5.0, y, 5.0]))                             # bên trong đi lại được
    assert w.collides(np.array([2.0, y, 0.1]))                                 # sát tường (cách cửa) bị chặn
    assert not w.collides(np.array([5.0, y, 0.0]))                             # đứng giữa ô cửa
    p = w.move(np.array([5.0, y, -3.0], np.float32), np.array([0, 0, 8.0], np.float32))
    assert p[2] > 4.9                                                          # đi thẳng qua cửa vào trong (đi đủ 8 m: tới z = 5)
    p = w.move(np.array([2.0, y, -3.0], np.float32), np.array([0, 0, 8.0], np.float32))
    assert p[2] < 0.0                                                          # đi vào chỗ không có cửa: bị tường chặn


def test_collider_boxes_block_and_pass_over():
    box = types.SimpleNamespace(collidable=False, collider_boxes=np.array([[4.8, 0.0, 4.8, 5.2, 3.0, 5.2]], np.float32))
    w = CollisionWorld([box]); y = config.EYE_HEIGHT
    assert w.collides(np.array([5.0, y, 5.0])) and not w.collides(np.array([8.0, y, 5.0]))
    low = types.SimpleNamespace(collidable=False, collider_boxes=np.array([[0, 0, 0, 5, 0.45, 5]], np.float32))
    assert CollisionWorld([low]).collides(np.array([2.0, y, 2.0]))             # rào thấp vẫn chặn (chân người ở y = 0)


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
    assert mb["id"] == "manual/x" and resolve_style(mb, ov)["levels"] == 2    # không có số tầng -> mặc định 2


# ------------------------------------------------------------ cây và rào
def test_plant_geometry_outward_and_boxes():
    from graphics.vegetation import build_plants, sphere_part, TRUNK_BOX_HEIGHT
    v, i = sphere_part(0, 0, 0, 1.0)
    tri = v[i.reshape(-1, 3)]
    geo = np.cross(tri[:, 1, :3] - tri[:, 0, :3], tri[:, 2, :3] - tri[:, 0, :3])
    assert np.all((geo * tri[:, 0, 3:6]).sum(axis=1) >= -1e-9)               # mặt cầu hướng ra ngoài
    trunk, canopy, boxes = build_plants([(0, 0, "tree"), (10, 0, "small"), (20, 0, "shrub")], seed=3)
    assert trunk is not None and canopy is not None and len(boxes) == 2        # bụi thấp không có hộp va chạm
    assert np.allclose(boxes[:, 4], TRUNK_BOX_HEIGHT) and canopy[0][:, 7].min() >= 0 and canopy[0][:, 7].max() <= 1


def test_vegetation_respects_constraints():
    from scene.landscape import Constraints, generate_vegetation, dist_points_segments, points_in_edges
    bld = rings_to_edges([[(0, 0), (40, 0), (40, 20), (0, 20)]])
    roads = [{"kind": "service", "points": [[-60, 40], [100, 40]]}]
    styles = {"service": ("asphalt", 5.5, 0.04, 4.0, False)}
    segs = np.array([[-60, 40, 100, 40, 2.75]], float)
    door = (20.0, 20.0, 0.0, 1.0)                                              # cửa ở mặt Nam, hướng ra +Z
    tour = np.array([[x, 30.0] for x in range(-20, 60, 2)], float)
    cons = Constraints([bld], segs, [], [door], tour, (10.0, 60.0))
    spec = {"seed": 5, "tree_zones": [{"name": "z", "polygon": [[2, 22], [38, 22], [38, 36], [2, 36]], "density": 6.0}],
            "street_trees": {"enabled": True, "spacing": 10.0, "margin": 40.0, "max": 40},
            "perimeter": {"enabled": True, "count": 60, "margin": 40.0}}
    plants, stats = generate_vegetation(spec, cons, roads, styles, (0, 0, 40, 20))
    assert len(plants) > 20 and stats.get("perimeter", 0) > 0
    P = np.array([[x, z] for x, z, k in plants])
    assert not points_in_edges(P, bld).any()                                   # không có cây trong tòa nhà
    assert (dist_points_segments(P, bld) >= 0.79).all()                        # không sát tường
    assert (np.abs(P[:, 1] - 40) >= 2.75).all()                                # không đè lên đường
    assert (np.hypot(*(P - np.array([20.0, 20.0])).T) >= 3.2).all()            # không ở ngay cửa
    assert not ((np.abs(P[:, 0] - 20.0) < 2.0) & (P[:, 1] > 19.5) & (P[:, 1] < 29)).any()   # không chặn lối đi từ cửa ra
    assert (np.hypot(*(P - np.array([10.0, 60.0])).T) >= 6.0).all()            # không sát điểm xuất phát
    zone_pts = P[(P[:, 0] > 2) & (P[:, 0] < 38) & (P[:, 1] > 22) & (P[:, 1] < 36)]
    assert len(zone_pts) >= 8                                                  # vùng bắt buộc có nhiều cây


def test_fence_gate_gap_and_collision_boxes():
    from graphics.fence import fence_for_polygon, FENCE_HEIGHT
    ring = [(0, 0), (20, 0), (20, 10), (0, 10)]
    (v, i), boxes, rep = fence_for_polygon(ring, gates=[{"at": [10, 0.2], "width": 4.0}])
    assert rep[0]["ok"] and np.all(boxes[:, 4] <= FENCE_HEIGHT + 1e-6) and FENCE_HEIGHT < 0.6    # thấp hơn đầu gối
    assert len(v) > 0 and len(i) % 3 == 0
    # cổng ở (10, 0) rộng 4 m: không hộp nào phủ đoạn x 8.2..11.8 trên cạnh z = 0
    cover = [b for b in boxes if b[2] < 0.5 and b[5] < 0.5 and b[0] < 11.8 and b[3] > 8.2]
    assert not cover
    w = CollisionWorld([types.SimpleNamespace(collidable=False, collider_boxes=boxes)]); y = config.EYE_HEIGHT
    assert w.collides(np.array([3.0, y, 0.0])) and not w.collides(np.array([10.0, y, 0.0]))
    assert (v[:, 1].max() <= FENCE_HEIGHT + 1e-6)
    _, boxes2, rep2 = fence_for_polygon(ring, gates=[{"at": [100, 100], "width": 4.0}])
    assert rep2[0]["ok"] is False


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
