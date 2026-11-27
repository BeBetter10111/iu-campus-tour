import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scene import traffic as T
from scene.collision import CollisionWorld
from scene.geometry2d import rings_to_edges, point_in_edges

SQUARE = rings_to_edges([[(20, -5), (30, -5), (30, 5), (20, 5)]])
ROADS = [{"kind": "service", "points": [[0, 0], [40, 0], [80, 0]]},
         {"kind": "footway", "points": [[0, 3], [40, 3], [80, 3]]},
         {"kind": "service", "points": [[80, 0], [80, 60]]}]


def _data():
    with open(config.ASSET_DIR / "data" / "campus_osm.json", encoding="utf-8") as f:
        return json.load(f)


def test_meshes_are_valid():
    for v, i in (T.car_arrays(3), T.person_arrays()):
        assert v.shape[1] == 8 and v.dtype == np.float32 and i.max() < len(v) and len(i) % 3 == 0
        assert np.all((v[:, 6:] >= 0) & (v[:, 6:] <= 1))            # UVs stay inside the palette
        assert np.allclose(np.linalg.norm(v[:, 3:6], axis=1), 1.0)
    assert T.car_arrays()[0][:, 1].max() < 2.0 and T.person_arrays()[0][:, 1].max() < 2.0


def test_edges_through_buildings_are_removed():
    g = T.usable_graph(ROADS, T.CAR_KINDS, [SQUARE])
    assert (0.0, 0.0) not in g and (40.0, 0.0) in g and (80.0, 0.0) in g
    assert not any(a[0] < 25 < b[0] or b[0] < 25 < a[0] for a, nb in g.items() for b in nb)


def test_agents_never_enter_buildings_and_keep_apart():
    roads = [{"kind": "service", "points": [[0, 0], [40, 0], [80, 0]]},
             {"kind": "service", "points": [[80, 0], [80, 60], [0, 60], [0, 0]]}]
    ts = T.build_traffic(roads, [SQUARE], {"cars": 4, "pedestrians": 0}, seed=1)
    assert ts.agents
    rng_min = 1e9
    for _ in range(3000):
        ts.update(0.05)
        for a in ts.agents:
            assert not point_in_edges(a.x, a.z, SQUARE)
        for i, a in enumerate(ts.agents):
            for b in ts.agents[i + 1:]:
                rng_min = min(rng_min, math.hypot(a.x - b.x, a.z - b.z))
    assert rng_min > 1.0                                              # no cars driving through each other
    assert all(a.s >= 0 for a in ts.agents)


def test_pedestrians_use_paths_and_move():
    ts = T.build_traffic(ROADS, [SQUARE], {"cars": 0, "pedestrians": 3}, seed=3)
    start = [(a.x, a.z) for a in ts.agents]
    for _ in range(200):
        ts.update(0.05)
    assert any(math.hypot(a.x - x, a.z - z) > 1.0 for a, (x, z) in zip(ts.agents, start))


def test_player_blocked_by_agent_and_can_walk_away():
    ts = T.build_traffic(ROADS, [SQUARE], {"cars": 0, "pedestrians": 1}, seed=5)
    a = ts.agents[0]
    a.speed = a.base_speed = 0.0
    w = CollisionWorld([])
    w.dynamic_source = ts
    px, pz = a.x - 1.5, a.z
    pos = np.array([px, config.EYE_HEIGHT, pz], np.float32)
    out = w.move(pos, np.array([1.3, 0, 0], np.float32))              # walk into the person
    assert math.hypot(out[0] - a.x, out[2] - a.z) >= a.radius + config.PLAYER_RADIUS - 0.3
    assert out[0] < a.x
    pos2 = np.array([a.x - 0.4, config.EYE_HEIGHT, a.z], np.float32)  # already overlapping -> may leave
    assert w.move(pos2, np.array([-1.0, 0, 0], np.float32))[0] < pos2[0]
    fly = np.array([a.x - 0.2, config.EYE_HEIGHT + 5, a.z], np.float32)
    assert w.move(fly, np.array([0.4, 0, 0], np.float32))[0] > fly[0]  # flying overhead is fine


def test_agent_stops_for_player_in_front():
    ts = T.build_traffic(ROADS, [SQUARE], {"cars": 0, "pedestrians": 1}, seed=5)
    a = ts.agents[0]
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    player = (a.x + c * 1.0, a.z + s * 1.0)
    for _ in range(40):
        a.update(0.05, ts.agents, player, [SQUARE])
    assert a.speed < 0.15


def test_real_campus_graphs():
    d = _data()
    blockers = [rings_to_edges([b["outer"]]) for b in d["buildings"]]
    ts = T.build_traffic(d["roads"], blockers, {"cars": 14, "pedestrians": 36}, seed=2024)
    kinds = [a.kind for a in ts.agents]
    assert kinds.count("car") >= 5 and kinds.count("ped") >= 10
    for _ in range(600):
        ts.update(0.05, (0.0, 0.0))
        for a in ts.agents:
            assert not any(point_in_edges(a.x, a.z, e) for e in blockers)
    assert ts.circles().shape[1] == 4


def test_parked_spots_inside_polygon():
    poly = [(0, 0), (40, 0), (40, 30), (0, 30)]
    spots = T.parked_car_spots(poly, [{"at": (20, 0)}], np.random.default_rng(1), 10)
    e = rings_to_edges([poly])
    assert 0 < len(spots) <= 10 and all(point_in_edges(x, z, e) for x, z in spots)


def test_roofed_parking_avoids_columns_and_bounds():
    poly = [(0, 0), (40, 0), (40, 30), (0, 30)]
    cols = T.roof_columns(poly, 8.0)
    spots = T.parked_car_spots(poly, [], np.random.default_rng(2), 50, cols)
    assert spots and all(not (abs(cx - x) < 1.5 and abs(cz - z) < 2.8) for x, z in spots for cx, cz in cols)
    g = T.usable_graph(ROADS, T.CAR_KINDS, [], bounds=(-5, -5, 50, 50))
    assert all(p[0] <= 50 for p in g)


# pedestrians dodge cars they see
STRAIGHT = [{"kind": "service", "points": [[0, 0], [100, 0], [200, 0]]}]


def _road_pair(ped_x, car_x, car_dir, trees=None, ped_dir=1):
    """One pedestrian and one car on a straight 5.5 m service road along +X."""
    rng = np.random.default_rng(0)
    cg = T.usable_graph(STRAIGHT, T.CAR_KINDS, [])
    pg = T.usable_graph(STRAIGHT, T.PED_KINDS, [])
    a, b = ((0.0, 0.0), (100.0, 0.0)) if ped_dir > 0 else ((100.0, 0.0), (0.0, 0.0))
    ped = T.Agent("ped", pg, rng, a, b, 1.3, 0.5, 0.3, 0.0, 1.8, 0)
    ped.widths = T.edge_widths(STRAIGHT, T.PED_KINDS)
    ped._set_edge(a, b)
    ped.s = ped_x if ped_dir > 0 else 100.0 - ped_x
    ped.x, ped.z = ped._target()
    ped.yaw = ped._heading
    a, b = ((0.0, 0.0), (100.0, 0.0)) if car_dir > 0 else ((200.0, 0.0), (100.0, 0.0))
    car = T.Agent("car", cg, rng, a, b, 5.5, 1.375, 1.0, 4.2, 1.5, 0)
    car.s = car_x if car_dir > 0 else 200.0 - car_x
    car.x, car.z = car._target()
    car.yaw = car._heading
    return T.TrafficSystem([car, ped], [], trees), car, ped


def _road_lateral(a):
    return a.z                                   # road centre line is z = 0


def test_pedestrian_dodges_oncoming_car_to_roadside():
    ts, car, ped = _road_pair(ped_x=40.0, car_x=130.0, car_dir=-1)   # car comes head-on
    worst = 0.0
    dodged = False
    for _ in range(600):
        ts.update(0.05)
        if math.hypot(car.x - ped.x, car.z - ped.z) < 4.0:           # while the car passes
            assert abs(_road_lateral(ped) - _road_lateral(car)) >= T.CAR_CORRIDOR - 0.15
        if abs(_road_lateral(ped)) > 2.75:                            # beyond the paved edge
            dodged = True
        worst = max(worst, abs(car.x - ped.x) < 2.5 and abs(car.z - ped.z) < 1.2)
    assert dodged and not worst
    # the car drove on along its lane, the person never went toward the car's side
    assert car.x < 30.0


def test_pedestrian_moves_away_from_car_never_across_its_lane():
    ts, car, ped = _road_pair(ped_x=40.0, car_x=110.0, car_dir=-1)
    side_car = math.copysign(1, car.z)
    furthest = 0.0
    for _ in range(380):
        ts.update(0.05)
        assert math.copysign(1, ped.z) != side_car or abs(ped.z) < abs(car.z)
        furthest = max(furthest, -side_car * ped.z)
    assert furthest > 2.75                                            # reached the far roadside
    assert ped.dodge == 0.0 and abs(ped.z - 0.5) < 0.05               # and went back once the car passed


def test_pedestrian_never_steps_toward_a_tree():
    # oncoming car drives on the person's left -> the free roadside is on the right, but a tree stands there
    ts, car, ped = _road_pair(ped_x=40.0, car_x=130.0, car_dir=-1)
    ped_side = -math.copysign(1, car.z)
    tree = np.array([[ped.x, ped_side * 3.6, 1.2]], np.float32)       # tree right on that roadside
    ts.trees = tree
    for _ in range(600):
        ts.update(0.05)
        assert math.hypot(ped.x - tree[0, 0], ped.z - tree[0, 1]) >= 1.2 - 0.05
        assert abs(ped.z - car.z) >= 0.9 or abs(ped.x - car.x) > 2.5   # never hit by / into the car's lane
    assert abs(ped.z) <= abs(tree[0, 1])


def test_unseen_car_behind_is_not_dodged():
    ts, car, ped = _road_pair(ped_x=60.0, car_x=20.0, car_dir=1)      # same direction, 40 m behind
    for _ in range(10):
        ts.update(0.05)
    assert ped.dodge == 0.0 and ped.alert == 0.0
