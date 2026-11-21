import numpy as np

import config
from scene.geometry2d import circle_hits_rings, dist_to_edges


class CollisionWorld:
    MAX_STEP = 0.25   

    def __init__(self, objects):
        mins, maxs, owners = [], [], []
        for o in objects:
            if getattr(o, "collidable", False):
                mins.append(np.asarray(o.aabb_min, np.float32))
                maxs.append(np.asarray(o.aabb_max, np.float32))
                owners.append(o)
            boxes = getattr(o, "collider_boxes", None)
            if boxes is not None and len(boxes):
                b = np.asarray(boxes, np.float32).reshape(-1, 6)
                mins.extend(b[:, :3])
                maxs.extend(b[:, 3:])
                owners.extend([None] * len(b))                    
        self.owners = owners
        self.collidable_objects = [o for o in owners if o is not None]
        if mins:
            self.aabb_mins = np.array(mins, dtype=np.float32)
            self.aabb_maxs = np.array(maxs, dtype=np.float32)
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
        px, pz, r = float(pos[0]), float(pos[2]), config.PLAYER_RADIUS
        for i in np.flatnonzero(overlap):
            owner = self.owners[i]
            if owner is None:
                return True                                       
            walls = getattr(owner, "wall_edges", None)
            if walls is not None:
                if len(walls) and dist_to_edges(px, pz, walls) < r:
                    return True                                   
                continue
            edges = getattr(owner, "footprint_edges", None)
            if edges is None or circle_hits_rings(px, pz, r, edges):
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
