import ctypes
import numpy as np
from OpenGL.GL import (
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer, glBufferData,
    glVertexAttribPointer, glEnableVertexAttribArray, glDrawElements,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_FALSE,
    GL_TRIANGLES, GL_UNSIGNED_INT,
)


class Mesh:
    def __init__(self, vertices, indices):
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        indices = np.ascontiguousarray(indices, dtype=np.uint32)
        self.index_count = len(indices)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        ebo = glGenBuffers(1)  
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        stride = 8 * 4
        for loc, size, offset in ((0, 3, 0), (1, 3, 12), (2, 2, 24)):
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))

        glBindVertexArray(0)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)


def create_cube():
    faces = [
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)),   # +X
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),   # -X
        ((0, 1, 0), (1, 0, 0), (0, 0, -1)),   # +Y
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),   # -Y
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),    # +Z
        ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),  # -Z
    ]
    verts, idx = [], []
    for i, (n, t, b) in enumerate(faces):
        n, t, b = map(np.array, (n, t, b))
        for (st, sb), uv in zip(((-1, -1), (1, -1), (1, 1), (-1, 1)),
                                ((0, 0), (1, 0), (1, 1), (0, 1))):
            pos = 0.5 * (n + st * t + sb * b)
            verts.append([*pos, *n, *uv])
        base = i * 4
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))