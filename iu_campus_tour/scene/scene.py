import numpy as np
from graphics import math3d as m3d


class SceneObject:
    def __init__(self, mesh, texture, position=(0, 0, 0), rotation_y=0.0, scale=(1, 1, 1)):
        self.mesh = mesh
        self.texture = texture
        self.model = (m3d.translate(*position)
                      @ m3d.rotate_y(np.radians(rotation_y))
                      @ m3d.scale(*scale))
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
        self.objects.sort(key=lambda o: id(o.texture))
        return obj

    def draw(self, shader):
        shader.set_int("uTexture", 0)
        last_texture = None
        for obj in self.objects:
            if obj.texture is not last_texture:   
                obj.texture.bind(0)
                last_texture = obj.texture
            shader.set_mat4("uModel", obj.model)
            obj.mesh.draw()