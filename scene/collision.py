import numpy as np
import config


class CollisionWorld:
    def __init__(self, objects):
        self.collidable_objects = [obj for obj in objects if obj.collidable]
        self.aabb_mins = np.array([obj.aabb_min for obj in self.collidable_objects])
        self.aabb_maxs = np.array([obj.aabb_max for obj in self.collidable_objects])

    def _player_box(self, pos):
        r = config.PLAYER_RADIUS
        min_box = np.array([pos[0] - r, pos[1] - config.PLAYER_EYE_HEIGHT, pos[2] - r], dtype=np.float32)
        max_box = np.array([pos[0] + r, pos[1] + config.PLAYER_HEAD_GAP, pos[2] + r], dtype=np.float32)
        return min_box, max_box 

    def collides(self, pos):
        player_min, player_max = self._player_box(pos)
        overlap = np.all((player_min < self.aabb_maxs) & (player_max > self.aabb_mins), axis=1)
        return np.any(overlap)

    def move(self, pos, delta):
        new = pos.copy()
        new[0] += delta[0]
        if self.collides(new):
            new[0] = pos[0]
        new[2] += delta[2]
        if self.collides(new):
            new[2] = pos[2]
        return new