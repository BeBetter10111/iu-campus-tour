import numpy as np

import config
from graphics import math3d as m3d

WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
NIGHT_AMBIENT = np.array((0.08, 0.09, 0.15), dtype=np.float32)
SUNSET_COLOR = np.array((1.0, 0.55, 0.30), dtype=np.float32)


def _smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class DirectionalLight:
    def __init__(self):
        R = config.SHADOW_RADIUS
        self.texel_world = 2.0 * R / config.SHADOW_MAP_SIZE
        self.normal_offset = config.SHADOW_NORMAL_OFFSET * self.texel_world
        self.projection = m3d.orthographic(-R, R, -R, R, 0.1, 2.0 * config.SHADOW_DISTANCE)
        self.light_space = np.identity(4, dtype=np.float32)
        self.set_angles(config.SUN_AZIMUTH, config.SUN_ELEVATION)

    def set_angles(self, azimuth, elevation):
        """Set sun position. Elevation runs a full circle: 0 sunrise, 90 noon, 180 sunset, <0 night."""
        self.azimuth = azimuth % 360.0
        self.elevation = (elevation + 180.0) % 360.0 - 180.0
        az, el = np.radians(self.azimuth), np.radians(self.elevation)
        self.direction = m3d.normalize([np.cos(el) * np.cos(az),
                                        np.sin(el),
                                        np.cos(el) * np.sin(az)])

        # Day/night: fade direct sunlight below the horizon, warm tint near it.
        h = float(self.direction[1])
        day = _smoothstep(-0.05, 0.15, h)
        warm = 1.0 - _smoothstep(0.0, 0.35, h)
        sun = np.array(config.SUN_COLOR, dtype=np.float32)
        self.color = (sun + (SUNSET_COLOR - sun) * warm) * config.SUN_INTENSITY * day
        day_amb = np.array(config.AMBIENT_COLOR, dtype=np.float32)
        self.ambient = NIGHT_AMBIENT + (day_amb - NIGHT_AMBIENT) * _smoothstep(-0.25, 0.2, h)

        # Shadow-map basis; stay valid when the sun is straight overhead.
        f = -self.direction
        right = np.cross(f, WORLD_UP)
        if np.linalg.norm(right) < 1e-4:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._right = m3d.normalize(right)
        self._up = np.cross(self._right, f)
        self._key = None                  # force the shadow map to rebuild

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
        view = m3d.look_at(eye, center, self._up)
        self.light_space = self.projection @ view
        return True
