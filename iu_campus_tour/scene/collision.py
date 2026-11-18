import numpy as np
import config


class CollisionWorld:
    def __init__(self, objects):
        collidable = [obj for obj in objects if getattr(obj, "collidable", True)]
        if collidable:
            self.mins = np.asarray([obj.aabb_min for obj in collidable], dtype=np.float32)
            self.maxs = np.asarray([obj.aabb_max for obj in collidable], dtype=np.float32)
        else:
            self.mins = np.zeros((0, 3), dtype=np.float32)
            self.maxs = np.zeros((0, 3), dtype=np.float32)

    def _player_box(self, pos):
        pos = np.asarray(pos, dtype=np.float32)
        r = config.PLAYER_RADIUS
        return (
            pos - np.array([r, config.EYE_HEIGHT, r], dtype=np.float32),
            pos + np.array([r, config.PLAYER_HEAD_GAP, r], dtype=np.float32),
        )

    def collides(self, pos):
        if len(self.mins) == 0:
            return False
        player_min, player_max = self._player_box(pos)
        overlap = (player_min < self.maxs) & (player_max > self.mins)
        return bool(np.any(np.all(overlap, axis=1)))

    def move(self, pos, delta):
        new = pos.copy()
        new[0] += delta[0]
        if self.collides(new):
            new[0] = pos[0]
        new[1] += delta[1]
        if self.collides(new):
            new[1] = pos[1]
        new[2] += delta[2]
        if self.collides(new):
            new[2] = pos[2]
        return new