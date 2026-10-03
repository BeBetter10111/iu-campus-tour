import numpy as np
import config


class TourController:
    def __init__(self, path):
        self.path = path
        self.s = 0.0
        self.paused = False

    def start(self, camera):
        self.s = 0.0
        camera.position = self.path.point_at_distance(self.s)
        camera.set_front(self.path.point_at_distance(self.s + config.TOUR_LOOK_AHEAD) - camera.position)
        pass

    def update(self, camera, dt):
        if not self.paused:
            self.s += config.TOUR_SPEED * dt
        camera.position = self.path.point_at_distance(self.s)
        desired = self.path.point_at_distance(self.s + config.TOUR_LOOK_AHEAD) - camera.position
        k = 1 - np.exp(-config.TOUR_LOOK_SMOOTH * dt)
        front = (1 - k) * camera.front + k * desired
        camera.set_front(front)

    def progress(self):
        return self.s / self.path.total_length if self.path.total_length > 0 else 0