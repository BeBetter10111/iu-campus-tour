import numpy as np
from graphics import math3d as m3d


class SceneObject:
    def __init__(self, mesh, texture, position=(0, 0, 0), rotation_y=0.0, scale=(1, 1, 1),
                 specular=0.1, shininess=32.0, cast_shadow=True, collidable=True):
        self.mesh = mesh
        self.texture = texture
        self.specular = specular        
        self.shininess = shininess      
        self.cast_shadow = cast_shadow  
        self.collidable = collidable

        # Model = T · R · S
        self.model = (m3d.translate(*position)
                      @ m3d.rotate_y(np.radians(rotation_y))
                      @ m3d.scale(*scale))
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


class DynamicObject:
    """Moving prop (car, pedestrian). Not part of ``Scene.objects`` so static
    collision/sorting is untouched; drawn and shadowed every frame."""

    def __init__(self, mesh, texture, specular=0.1, shininess=32.0, cast_shadow=True):
        self.mesh, self.texture = mesh, texture
        self.specular, self.shininess, self.cast_shadow = specular, shininess, cast_shadow
        self.model = np.identity(4, dtype=np.float32)
        self.normal_matrix = np.identity(3, dtype=np.float32)

    def set_pose(self, x, y, z, yaw_deg, roll_deg=0.0):
        rot = m3d.rotate_y(np.radians(yaw_deg))
        if roll_deg:
            rot = rot @ m3d.rotate_x(np.radians(roll_deg))
        self.model = (m3d.translate(x, y, z) @ rot).astype(np.float32)
        self.normal_matrix = np.ascontiguousarray(rot[:3, :3], dtype=np.float32)   # rigid -> same as inverse-transpose


class Scene:
    def __init__(self):
        self.objects = []
        self.dynamic = []
        self.traffic = None

    def add_dynamic(self, obj):
        self.dynamic.append(obj)
        return obj

    def update_dynamic(self, dt, player_xz=None):
        if self.traffic is None:
            return
        self.traffic.update(dt, player_xz)
        for a in self.traffic.agents:
            a.obj.set_pose(*a.pose())

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
            shader.set_mat3("uNormalMatrix", obj.normal_matrix)
            shader.set_float("uSpecular", obj.specular)
            shader.set_float("uShininess", obj.shininess)
            obj.mesh.draw()
        self._draw_dynamic(shader, depth=False)

    def draw_depth(self, shader):
        for obj in self.objects:
            if obj.cast_shadow:
                shader.set_mat4("uModel", obj.model)
                obj.mesh.draw()
        self._draw_dynamic(shader, depth=True)

    def _draw_dynamic(self, shader, depth):
        last_texture = None
        for obj in self.dynamic:
            if depth and not obj.cast_shadow:
                continue
            if not depth and obj.texture is not last_texture:
                obj.texture.bind(0)
                last_texture = obj.texture
            shader.set_mat4("uModel", obj.model)
            if not depth:
                shader.set_mat3("uNormalMatrix", obj.normal_matrix)
                shader.set_float("uSpecular", obj.specular)
                shader.set_float("uShininess", obj.shininess)
            obj.mesh.draw()
