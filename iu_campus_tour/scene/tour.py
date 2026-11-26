import numpy as np
import config


class TourController:
    def __init__(self, path, height_fn=None):
        self.path = path
        self.height_fn = height_fn      # (x, z) -> do cao mat dat (cau thang, san tang)
        self._lift = 0.0
        self.s = 0.0
        self.paused = False
        self.speed = config.TOUR_SPEED

    def start(self, camera):
        self.s = 0.0
        self._lift = 0.0
        camera.position = self.path.point_at_distance(self.s)
        look_ahead_point = self.path.point_at_distance(self.s + config.TOUR_LOOK_AHEAD)
        desired = look_ahead_point - camera.position
        if np.linalg.norm(desired) > 1e-6:
            camera.set_front(desired)

    def update(self, camera, dt):
        if (not self.paused):
            self.s += self.speed * dt
            camera.position = self.path.point_at_distance(self.s)
            if self.height_fn is not None:
                target = float(self.height_fn(float(camera.position[0]), float(camera.position[2])))
                self._lift += (target - self._lift) * (1 - np.exp(-8.0 * dt))
                camera.position[1] += self._lift
            look_ahead_point = self.path.point_at_distance(self.s + config.TOUR_LOOK_AHEAD)
            desired = look_ahead_point - self.path.point_at_distance(self.s)
            if np.linalg.norm(desired) > 1e-6:
                desired /= np.linalg.norm(desired)
            k = 1 - np.exp(-config.TOUR_LOOK_SMOOTH * dt)
            front = camera.front * (1 - k) + desired * k
            if np.linalg.norm(front) > 1e-6:
                camera.set_front(front)
        if self.path.closed:
            self.s %= self.path.total_length

    def progress(self):
        """Return tour progress from 0 to 1."""
        s = self.s % self.path.total_length if self.path.closed else self.s
        return min(max(s / self.path.total_length, 0.0), 1.0)