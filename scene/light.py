import numpy as np

import config
from graphics import math3d as m3d

WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


class DirectionalLight:
    """Ánh sáng song song. `direction` là hướng TỚI mặt trời (từ bề mặt nhìn lên),
    chính là vector L trong công thức Phong."""

    def __init__(self):
        self.azimuth = config.SUN_AZIMUTH
        self.elevation = config.SUN_ELEVATION
        self.color = np.array(config.SUN_COLOR, dtype=np.float32) * config.SUN_INTENSITY
        self.ambient = np.array(config.AMBIENT_COLOR, dtype=np.float32)

        R = config.SHADOW_RADIUS
        self.texel_world = 2.0 * R / config.SHADOW_MAP_SIZE
        self.normal_offset = config.SHADOW_NORMAL_OFFSET * self.texel_world
        self.projection = m3d.orthographic(-R, R, -R, R, 0.1, 2.0 * config.SHADOW_DISTANCE)

        self.light_space = np.identity(4, dtype=np.float32)
        self._update_direction()

    def _update_direction(self):
        """Tính lại hướng mặt trời và hai trục ngang/dọc của camera ánh sáng."""
        az, el = np.radians(self.azimuth), np.radians(self.elevation)
        self.direction = m3d.normalize([np.cos(el) * np.cos(az),
                                        np.sin(el),
                                        np.cos(el) * np.sin(az)])
        f = -self.direction                      # Hướng ánh sáng đi
        self._right = m3d.normalize(np.cross(f, WORLD_UP))
        self._up = np.cross(self._right, f)
        self._key = None                         # Ép update() dựng lại ma trận + shadow map

    def set_angles(self, azimuth, elevation):
        self.azimuth = azimuth % 360.0
        # Chặn góc thấp: dưới ~15° bóng kéo dài vô hạn và shadow map bị giãn texel
        self.elevation = max(15.0, min(85.0, elevation))
        self._update_direction()

    def update(self, focus):
        """Đặt khung shadow map quanh vị trí `focus` (người chơi).

        Trả về True nếu ma trận thay đổi (cần render lại shadow map).

        TEXEL SNAPPING: nếu tâm shadow map trượt mượt theo người chơi, lưới texel sẽ
        trượt theo -> mép bóng nhấp nháy (shimmering). Ta làm tròn tâm theo bội số của
        kích thước texel trong không gian ánh sáng -> lưới texel luôn dính cố định vào
        thế giới, bóng đứng yên khi người chơi đi lại.
        """
        focus = np.asarray(focus, dtype=np.float32)
        t = self.texel_world
        cr = float(focus @ self._right)   # Tọa độ của focus trên trục ngang ánh sáng
        cu = float(focus @ self._up)      # ... và trục dọc
        sr, su = np.floor(cr / t) * t, np.floor(cu / t) * t   # Làm tròn xuống bội số của texel

        key = (sr, su)
        if key == self._key:              # Chưa nhích đủ 1 texel -> giữ nguyên, khỏi render lại
            return False
        self._key = key

        center = focus + self._right * (sr - cr) + self._up * (su - cu)
        eye = center + self.direction * config.SHADOW_DISTANCE   # Đặt "mắt" mặt trời lùi về phía nó
        view = m3d.look_at(eye, center, WORLD_UP)
        self.light_space = self.projection @ view
        return True