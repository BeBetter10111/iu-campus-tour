"""Skybox: cube bao quanh, luôn ở xa vô cực và đi theo camera."""

from OpenGL.GL import (glDepthFunc, glDisable, glEnable,
                       GL_LEQUAL, GL_LESS, GL_CULL_FACE)

import config
from graphics.shader import Shader
from graphics.mesh import create_box
from graphics.texture import Cubemap


class Skybox:
    def __init__(self):
        self.shader = Shader.from_files("skybox.vert", "skybox.frag")
        self.cube = create_box()
        self.cubemap = Cubemap.load_or_procedural(config.SKYBOX_DIR)

    def draw(self, view, projection, sun_dir, sun_color):
        # Bỏ phần tịnh tiến của View -> skybox chỉ xoay theo hướng nhìn, luôn ở xa vô cực.
        view_rot = view.copy()
        view_rot[:3, 3] = 0.0

        glDepthFunc(GL_LEQUAL)      # z = 1.0 (xa nhất) phải qua được depth test
        glDisable(GL_CULL_FACE)     # Camera nằm BÊN TRONG cube nên thấy mặt trong
        self.shader.use()
        self.shader.set_mat4("uView", view_rot)
        self.shader.set_mat4("uProjection", projection)
        self.shader.set_int("uSkybox", 0)
        self.shader.set_vec3("uSunDir", *sun_dir)
        self.shader.set_vec3("uSunColor", *sun_color)
        self.cubemap.bind(0)
        self.cube.draw()
        glEnable(GL_CULL_FACE)
        glDepthFunc(GL_LESS)