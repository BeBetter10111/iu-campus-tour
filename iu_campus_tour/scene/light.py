import numpy as np

import config
from graphics import math3d as m3d

WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


class DirectionalLight:
    def __init__(self):
        az, el = np.radians(config.SUN_AZIMUTH), np.radians(config.SUN_ELEVATION)
        self.direction = m3d.normalize([np.cos(el) * np.cos(az),
                                        np.sin(el),
                                        np.cos(el) * np.sin(az)])
        self.color = np.array(config.SUN_COLOR, dtype=np.float32) * config.SUN_INTENSITY
        self.ambient = np.array(config.AMBIENT_COLOR, dtype=np.float32)

        R = config.SHADOW_RADIUS
        self.texel_world = 2.0 * R / config.SHADOW_MAP_SIZE
        self.normal_offset = config.SHADOW_NORMAL_OFFSET * self.texel_world
        self.projection = m3d.orthographic(-R, R, -R, R, 0.1, 2.0 * config.SHADOW_DISTANCE)

        f = -self.direction                                    
        self._right = m3d.normalize(np.cross(f, WORLD_UP))
        self._up = np.cross(self._right, f)

        self.light_space = np.identity(4, dtype=np.float32)    
        self._key = None

    def update(self, focus):
        focus = np.asarray(focus, dtype=np.float32)
        t = self.texel_world
        cr = float(focus @ self._right)   
        cu = float(focus @ self._up)      
        sr, su = np.floor(cr / t) * t, np.floor(cu / t) * t   

        key = (sr, su)
        if key == self._key:              
            return False
        self._key = key

        center = focus + self._right * (sr - cr) + self._up * (su - cu)
        eye = center + self.direction * config.SHADOW_DISTANCE   
        view = m3d.look_at(eye, center, WORLD_UP)
        self.light_space = self.projection @ view
        return True