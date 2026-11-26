import config
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

def _load_font(bold, size):
    """Font di kem (ho tro tieng Viet) -> font he thong -> font mac dinh cua Pillow."""
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    candidates = [str(config.ASSET_DIR / "fonts" / names[0]),
                  "arialbd.ttf" if bold else "arial.ttf", names[0]]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


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

    def make_text_texture(self, lines, size=None, title=True, font_size=30,
                          min_size=None, pad=20):
        lines = [str(line) for line in lines]
        fonts = []
        for index, line in enumerate(lines):
            is_title = title and index == 0 and bool(line)
            point_size = font_size + 8 if is_title else font_size
            font = _load_font(is_title, point_size)
            fonts.append(font)

        boxes = [
            font.getbbox(line or " ")
            for line, font in zip(lines, fonts)
        ]
        line_heights = [max(font_size, box[3] - box[1]) for box in boxes]
        gap = max(4, font_size // 3)
        content_width = max(
            (box[2] - box[0] for box in boxes),
            default=font_size,
        )
        content_height = sum(line_heights) + gap * max(0, len(lines) - 1)
        natural_size = (content_width + 2 * pad, content_height + 2 * pad)
        width, height = size or natural_size
        if min_size is not None:
            width = max(width, min_size[0])
            height = max(height, min_size[1])

        img = Image.new("RGBA", (width, height), (0, 0, 0, 150))
        draw = ImageDraw.Draw(img)
        y = pad
        for line, font, box, line_height in zip(lines, fonts, boxes, line_heights):
            if line:
                draw.text((pad, y - box[1]), line, font=font, fill=(255, 255, 255, 255))
            y += line_height + gap
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return Texture(np.asarray(img))

    def draw_at(self, texture, screen_width, screen_height, anchor="center", margin=20):
        if screen_width <= 0 or screen_height <= 0 or margin < 0:
            raise ValueError("Screen dimensions must be positive and margin non-negative")

        max_width = max(1, screen_width - 2 * margin)
        max_height = max(1, screen_height - 2 * margin)
        scale = min(1.0, max_width / texture.width, max_height / texture.height)
        width = texture.width * scale * 2 / screen_width
        height = texture.height * scale * 2 / screen_height
        left = -1 + 2 * margin / screen_width
        bottom = -1 + 2 * margin / screen_height

        if anchor == "center":
            x, y = -width / 2, -height / 2
        elif anchor == "bottomleft":
            x, y = left, bottom
        elif anchor == "topleft":
            x, y = left, 1 - height - 2 * margin / screen_height
        else:
            raise ValueError(f"Unsupported overlay anchor: {anchor}")

        self.draw(texture, (x, y, width, height))

    def draw(self, texture, rect):
        """Convert a NDC rect into normalized rendering coordinates."""
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