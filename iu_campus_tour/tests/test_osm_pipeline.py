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
