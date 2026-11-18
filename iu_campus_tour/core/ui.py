import numpy as np
from PIL import Image, ImageDraw, ImageFont
from OpenGL.GL import (
    glEnable, glDisable, glBlendFunc, glUniform4f,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_DEPTH_TEST, GL_CULL_FACE,
)

from graphics.mesh import Mesh
from graphics.shader import Shader
from graphics.texture import Texture

class Overlay:
    def __init__(self):
        self.shader = Shader.from_files("ui.vert", "ui.frag")
        vertices = np.array([
            [-1, -1, 0, 0, 0, 1, 0, 0],
            [ 1, -1, 0, 0, 0, 1, 1, 0],
            [ 1,  1, 0, 0, 0, 1, 1, 1],
            [-1,  1, 0, 0, 0, 1, 0, 1],
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self.quad = Mesh(vertices, indices)
        try:
            self.font = ImageFont.truetype("arial.ttf", 32)
        except OSError:
            self.font = ImageFont.load_default()

    def make_text_texture(self, lines, size=(640, 360)):
        img = Image.new("RGBA", size, (0, 0, 0, 150))
        draw = ImageDraw.Draw(img)
        y = 20
        for line in lines:
            draw.text((20, y), line, font=self.font, fill=(255, 255, 255, 255))
            y += 38
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return Texture(np.asarray(img))

    def draw(self, texture, rect):
        """rect = (x, y, w, h) tính theo NDC."""
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.shader.use()
        self.shader.set_int("uTex", 0)
        glUniform4f(self.shader._loc("uRect"), *rect)
        texture.bind(0)
        self.quad.draw()

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)