"""SceneObject và Scene: danh sách vật thể tĩnh trong thế giới."""

import numpy as np
from graphics import math3d as m3d


class SceneObject:
    def __init__(self, mesh, texture, position=(0, 0, 0), rotation_y=0.0, scale=(1, 1, 1),
                 specular=0.1, shininess=32.0, cast_shadow=True):
        self.mesh = mesh
        self.texture = texture
        self.specular = specular        # k_s của vật liệu
        self.shininess = shininess      # số mũ n của Phong
        self.cast_shadow = cast_shadow  # Có tham gia pass đổ bóng không (mặt đất thì không cần)

        # Model = T · R · S
        self.model = (m3d.translate(*position)
                      @ m3d.rotate_y(np.radians(rotation_y))
                      @ m3d.scale(*scale))
        # Ma trận pháp tuyến = (M⁻¹)ᵀ của phần 3x3. Vật thể tĩnh nên tính MỘT lần ở đây,
        # thay vì gọi inverse() cho từng đỉnh trong vertex shader.
        self.normal_matrix = np.linalg.inv(self.model[:3, :3]).T.astype(np.float32)
        self._compute_world_aabb()

    def _compute_world_aabb(self):
        lo, hi = self.mesh.bounds_min, self.mesh.bounds_max
        corners = np.array([[x, y, z, 1.0]
                            for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])], dtype=np.float32)
        world = (self.model @ corners.T).T[:, :3]
        self.aabb_min = world.min(axis=0)
        self.aabb_max = world.max(axis=0)


class Scene:
    def __init__(self):
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)
        self.objects.sort(key=lambda o: id(o.texture))   # Gom theo texture -> ít glBindTexture
        return obj

    def draw(self, shader):
        """Pass chính: bind texture + gửi vật liệu."""
        shader.set_int("uTexture", 0)
        last_texture = None
        for obj in self.objects:
            if obj.texture is not last_texture:
                obj.texture.bind(0)
                last_texture = obj.texture
            shader.set_mat4("uModel", obj.model)
            shader.set_mat3("uNormalMatrix", obj.normal_matrix)
            shader.set_float("uSpecular", obj.specular)
            shader.set_float("uShininess", obj.shininess)
            obj.mesh.draw()

    def draw_depth(self, shader):
        """Pass đổ bóng: không cần texture/vật liệu, chỉ cần hình học."""
        for obj in self.objects:
            if obj.cast_shadow:
                shader.set_mat4("uModel", obj.model)
                obj.mesh.draw()