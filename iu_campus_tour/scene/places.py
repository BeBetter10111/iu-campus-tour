"""Named places: look-at labels and numbered teleport (data in assets/data/places.json)."""

import json
import math

import numpy as np


def load_places(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("places", [])


def place_in_view(places, pos, front, max_dist=120.0, slack_deg=5.0):
    """Return the nearest place the camera is looking at, or None.

    A place counts when its centre lies inside a horizontal cone around the view
    direction; the cone widens with the place radius and is a half-plane when the
    camera is already inside that radius.
    """
    fx, fz = float(front[0]), float(front[2])
    fl = math.hypot(fx, fz)
    if fl < 1e-6:
        return None
    fx, fz = fx / fl, fz / fl
    best, best_key = None, None
    for p in places:
        dx, dz = p["at"][0] - float(pos[0]), p["at"][1] - float(pos[2])
        dist = math.hypot(dx, dz)
        if dist > max_dist:
            continue
        r = float(p.get("radius", 10.0))
        tol = 90.0 if dist <= r else min(90.0, slack_deg + math.degrees(math.asin(r / dist)))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (dx * fx + dz * fz) / max(dist, 1e-6)))))
        if ang <= tol and (best_key is None or dist - r < best_key):
            best, best_key = p, dist - r
    return best


def place_for_key(places, key_name):
    return next((p for p in places if p.get("key") == key_name), None)


def teleport_pose(place, collides, eye_y):
    """Return (position, yaw_deg) for a place, nudging the stand point if it is blocked.

    collides(pos) -> bool; eye_y(x, z) -> camera height there. Returns None if nothing is free.
    """
    sx, sz = place.get("stand", place["at"])
    lx, lz = place.get("look_at", place["at"])
    spots = [(sx, sz)] + [(sx + r * math.cos(math.radians(a)), sz + r * math.sin(math.radians(a)))
                          for r in (1.5, 3.0, 4.5, 6.0, 9.0, 12.0) for a in range(0, 360, 30)]
    for x, z in spots:
        pos = np.array([x, eye_y(x, z), z], np.float32)
        if not collides(pos):
            return pos, math.degrees(math.atan2(lz - z, lx - x))
    return None
