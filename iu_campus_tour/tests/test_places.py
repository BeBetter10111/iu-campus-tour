import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.places import load_places, place_for_key, place_in_view, teleport_pose

DATA = Path(__file__).resolve().parent.parent / "assets" / "data" / "places.json"
A = {"name_vi": "A", "at": [0, -50], "radius": 10}
B = {"name_vi": "B", "at": [0, -100], "radius": 10}          # directly behind A
C = {"name_vi": "C", "at": [60, -5], "radius": 10}


def view(pos, front, places):
    return place_in_view(places, np.array([pos[0], 1.7, pos[1]]), np.array([front[0], 0.0, front[1]]))


def test_label_picks_nearest_place_in_cone():
    assert view((0, 0), (0, -1), [B, A, C])["name_vi"] == "A"       # A hides B
    assert view((0, 0), (1, 0), [A, B, C])["name_vi"] == "C"        # turn right -> C
    assert view((0, 0), (0, 1), [A, B, C]) is None                  # looking away
    assert view((0, 0), (0, -1), [A], ) is not None
    far = dict(A, at=[0, -500])
    assert view((0, 0), (0, -1), [far]) is None                     # beyond max distance


def test_label_inside_radius_is_half_plane():
    big = {"name_vi": "BIG", "at": [0, 0], "radius": 30}
    assert view((5, 5), (-1, -1), [big]) is not None
    assert view((5, 5), (1, 1), [big]) is None


def test_teleport_faces_place_and_avoids_blocked_spot():
    place = {"at": [0.0, -50.0], "stand": [0.0, 0.0]}
    pos, yaw = teleport_pose(place, lambda p: False, lambda x, z: 1.7)
    assert np.allclose(pos, [0, 1.7, 0]) and abs(yaw - (-90.0)) < 1e-6
    blocked = lambda p: np.hypot(p[0], p[2]) < 2.0               # stand point blocked
    pos, _ = teleport_pose(place, blocked, lambda x, z: 1.7)
    assert not blocked(pos) and np.hypot(pos[0], pos[2]) < 4.0
    assert teleport_pose(place, lambda p: True, lambda x, z: 1.7) is None


def test_places_json_is_valid():
    places = load_places(DATA)
    assert len(places) >= 5
    keys = [p["key"] for p in places if p.get("key")]
    assert len(keys) == len(set(keys)) and place_for_key(places, "1")["short"] == "A1"
    assert place_for_key(places, "9") is None
    for p in places:
        assert p["name_vi"] and p["name_en"] and len(p["at"]) == 2 and p["radius"] > 0
