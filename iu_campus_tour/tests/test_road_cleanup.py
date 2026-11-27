import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.road_cleanup import apply_road_edits, clip_roads_to_buildings, split_at_junctions

BOX = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_road_edits_remove_replace_add():
    roads = [{"id": "a", "kind": "service", "points": [[0, 0], [1, 0]]},
             {"id": "b", "kind": "footway", "points": [[0, 1], [1, 1]]}]
    out, rep = apply_road_edits(roads, {"remove": ["b"], "replace": {"a": {"points": [[0, 0], [5, 0]]}},
                                        "add": [{"id": "c", "kind": "path", "points": [[2, 2], [3, 3]]}]})
    assert [r["id"] for r in out] == ["a", "c"]
    assert out[0]["points"] == [[0, 0], [5, 0]] and out[0]["kind"] == "service"
    assert rep == {"removed": 1, "replaced": 1, "added": 1}


def test_clip_drops_dead_end_stub_and_keeps_connected_part():
    # road comes from a street at x=-30, runs under the building and sticks out 15 m on the far side
    street = {"id": "street", "kind": "tertiary", "points": [[-30, -50], [-30, 50]]}
    road = {"id": "r", "kind": "residential", "points": [[-30, 5], [5, 5], [25, 5]]}
    out, rep = clip_roads_to_buildings([street, road], [BOX])
    ids = [r["id"] for r in out]
    assert "street" in ids and rep == {"cut": 1, "dropped": 1}
    kept = [r for r in out if r["id"].startswith("r#")]
    assert len(kept) == 1
    P = np.asarray(kept[0]["points"])
    assert P[:, 0].max() <= 0.0 + 1e-6, "phần còn lại phải dừng ở tường, không chạy qua nhà"


def test_clip_leaves_roads_outside_buildings_untouched():
    road = {"id": "r", "kind": "service", "points": [[-5, -5], [20, -5]]}
    out, rep = clip_roads_to_buildings([road], [BOX])
    assert out == [road] and rep["cut"] == 0


def test_junction_patches_stop_markings_at_crossing():
    a = {"id": "a", "kind": "tertiary", "points": [[-50, 0], [50, 0]]}
    b = {"id": "b", "kind": "tertiary", "points": [[0, -50], [0, 50]]}
    widths = {"tertiary": 8.5}
    pieces = split_at_junctions([a, b], widths, {"tertiary"}, {"tertiary"})
    for k in (0, 1):
        kinds = [plain for _, plain in pieces[k]]
        assert kinds == [False, True, False]
        plain_pts = [p for p, plain in pieces[k] if plain][0]
        span = np.hypot(*(plain_pts[-1] - plain_pts[0]))
        assert 8.5 <= span <= 8.5 + 2 * (1.5 + 0.3) + 1.0   # other road width + padding


def test_continuation_and_parallel_carriageway_are_not_junctions():
    a = {"id": "a", "kind": "tertiary", "points": [[0, 0], [50, 0]]}
    a2 = {"id": "a2", "kind": "tertiary", "points": [[50, 0], [100, 0]]}
    twin = {"id": "t", "kind": "tertiary", "points": [[0, 8.0], [100, 8.0]]}
    pieces = split_at_junctions([a, a2, twin], {"tertiary": 8.5}, {"tertiary"}, {"tertiary"})
    assert pieces == {}
