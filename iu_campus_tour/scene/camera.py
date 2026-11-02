import numpy as np
import pygame

import config
from graphics import math3d as m3d

WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


class FirstPersonCamera:
    def __init__(self, position=(0.0, config.EYE_HEIGHT, 8.0), yaw=-90.0, pitch=0.0):
        self.position = np.array(position, dtype=np.float32)
        self.yaw = yaw       
        self.pitch = pitch   
        self.fly_mode = False  
        self._update_vectors()

    
    def _update_vectors(self):
        yaw, pitch = np.radians(self.yaw), np.radians(self.pitch)
        self.front = m3d.normalize([
            np.cos(yaw) * np.cos(pitch),
            np.sin(pitch),
            np.sin(yaw) * np.cos(pitch),
        ])
        self.right = m3d.normalize(np.cross(self.front, WORLD_UP))

    # input
    def process_mouse(self, dx, dy):
        self.yaw += dx * config.MOUSE_SENSITIVITY
        self.pitch -= dy * config.MOUSE_SENSITIVITY  
        self.pitch = max(-89.0, min(89.0, self.pitch))
        self._update_vectors()

    def process_keyboard(self, keys, dt):
        speed = config.RUN_SPEED if keys[pygame.K_LSHIFT] else config.WALK_SPEED

        forward = self.front if self.fly_mode else m3d.normalize([self.front[0], 0.0, self.front[2]])

        move = np.zeros(3, dtype=np.float32)
        if keys[pygame.K_w]: move += forward
        if keys[pygame.K_s]: move -= forward
        if keys[pygame.K_d]: move += self.right
        if keys[pygame.K_a]: move -= self.right

        length = np.linalg.norm(move)
        if length > 0:
            self.position += (move / length) * speed * dt   

        if not self.fly_mode:
            self.position[1] = config.EYE_HEIGHT

    def get_view_matrix(self):
        return m3d.look_at(self.position, self.position + self.front, WORLD_UP)

    def get_projection_matrix(self, aspect):
        return m3d.perspective(config.FOV, aspect, config.NEAR_PLANE, config.FAR_PLANE)