import numpy as np
import config


class TourController:
    def __init__(self, path):
        self.path = path
        self.s = 0.0
        self.paused = False
        self.speed = config.TOUR_SPEED

    def start(self, camera):
        self.s = 0.0
        camera.position = self.path.point_at_distance(self.s)
        look_ahead_point = self.path.point_at_distance(self.s + config.TOUR_LOOK_AHEAD)
        desired = look_ahead_point - camera.position
        if np.linalg.norm(desired) > 1e-6:
            camera.set_front(desired)

    def update(self, camera, dt):
        if (not self.paused):
            self.s += self.speed * dt
            camera.position = self.path.point_at_distance(self.s)
            look_ahead_point = self.path.point_at_distance(self.s + config.TOUR_LOOK_AHEAD)
            desired = look_ahead_point - camera.position
            if np.linalg.norm(desired) > 1e-6:
                desired /= np.linalg.norm(desired)
            k = 1 - np.exp(-config.TOUR_LOOK_SMOOTH * dt)
            front = camera.front * (1 - k) + desired * k
            if np.linalg.norm(front) > 1e-6:
                camera.set_front(front)
        if self.path.closed:
            self.s %= self.path.total_length

    def progress(self):
        """Trả về % hoàn thành 0..1, dùng cho HUD."""
        s = self.s % self.path.total_length if self.path.closed else self.s
        return min(max(s / self.path.total_length, 0.0), 1.0)