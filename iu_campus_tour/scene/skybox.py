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

    def draw(self, view, projection):
        view_rot = view.copy()
        view_rot[:3, 3] = 0.0

        glDepthFunc(GL_LEQUAL)      
        glDisable(GL_CULL_FACE)     
        self.shader.use()
        self.shader.set_mat4("uView", view_rot)
        self.shader.set_mat4("uProjection", projection)
        self.shader.set_int("uSkybox", 0)
        self.cubemap.bind(0)
        self.cube.draw()
        glEnable(GL_CULL_FACE)
        glDepthFunc(GL_LESS)