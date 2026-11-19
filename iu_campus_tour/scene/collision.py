import numpy as np

import config
from scene.geometry2d import circle_hits_rings


class CollisionWorld:
    MAX_STEP = 0.25   

    def __init__(self, objects):
        self.collidable_objects = [o for o in objects if getattr(o, "collidable", False)]
        if self.collidable_objects:
            self.aabb_mins = np.array([o.aabb_min for o in self.collidable_objects], dtype=np.float32)
            self.aabb_maxs = np.array([o.aabb_max for o in self.collidable_objects], dtype=np.float32)
        else:
            self.aabb_mins = np.zeros((0, 3), dtype=np.float32)
            self.aabb_maxs = np.zeros((0, 3), dtype=np.float32)

    def _player_box(self, pos):
        r = config.PLAYER_RADIUS
        lo = np.array([pos[0] - r, pos[1] - config.EYE_HEIGHT, pos[2] - r], dtype=np.float32)
        hi = np.array([pos[0] + r, pos[1] + config.PLAYER_HEAD_GAP, pos[2] + r], dtype=np.float32)
        return lo, hi

    def collides(self, pos):
        if len(self.aabb_mins) == 0:
            return False
        p_lo, p_hi = self._player_box(pos)
        overlap = np.all((p_lo < self.aabb_maxs) & (p_hi > self.aabb_mins), axis=1)
        for i in np.flatnonzero(overlap):
            edges = getattr(self.collidable_objects[i], "footprint_edges", None)
            if edges is None:
                return True                                   
            if circle_hits_rings(float(pos[0]), float(pos[2]), config.PLAYER_RADIUS, edges):
                return True
        return False

    def move(self, pos, delta):
        pos = np.asarray(pos, dtype=np.float32)
        delta = np.asarray(delta, dtype=np.float32)
        dist = float(np.linalg.norm(delta))
        if dist < 1e-9:
            return pos.copy()

        n = max(1, int(np.ceil(dist / self.MAX_STEP)))
        step = delta / n
        cur = pos.copy()
        for _ in range(n):
            for axis in (0, 2, 1):               
                if step[axis] == 0.0:
                    continue
                trial = cur.copy()
                trial[axis] += step[axis]
                if not self.collides(trial):
                    cur = trial
        return cur
