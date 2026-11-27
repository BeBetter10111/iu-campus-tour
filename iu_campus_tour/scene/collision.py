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
        self.dynamic_source = None     # object with .circles() -> (n,4) x,z,r,top (cars, pedestrians)
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

    def _hits_dynamic(self, pos, cur=None):
        """True if ``pos`` overlaps a moving agent (unless the move leads away from one already overlapped)."""
        src = self.dynamic_source
        if src is None:
            return False
        c = src.circles()
        if len(c) == 0:
            return False
        if pos[1] - config.EYE_HEIGHT >= 0.05 + float(c[:, 3].max()):      # flying above everything
            return False
        feet = pos[1] - config.EYE_HEIGHT
        d = np.hypot(c[:, 0] - pos[0], c[:, 1] - pos[2])
        hit = (d < c[:, 2] + config.PLAYER_RADIUS) & (feet < c[:, 3])
        if not hit.any():
            return False
        if cur is None:
            return True
        d0 = np.hypot(c[:, 0] - cur[0], c[:, 1] - cur[2])
        return bool(np.any(hit & (d <= d0 + 1e-4)))

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
                if not self.collides(trial) and not self._hits_dynamic(trial, cur):
                    cur = trial
        return cur
