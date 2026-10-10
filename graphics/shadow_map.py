import numpy as np
from OpenGL.GL import (
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, glTexParameterfv,
    glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D, glCheckFramebufferStatus,
    glDrawBuffer, glReadBuffer, glViewport, glClear, glCullFace, glActiveTexture,
    GL_TEXTURE_2D, GL_TEXTURE0, GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_COMPLETE,
    GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT24, GL_FLOAT, GL_NONE,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_TEXTURE_BORDER_COLOR, GL_NEAREST, GL_CLAMP_TO_BORDER,
    GL_DEPTH_BUFFER_BIT, GL_FRONT, GL_BACK,
)


class ShadowMap:
    def __init__(self, size):
        self.size = size

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, size, size, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR,
                         np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))

        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.texture, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("ShadowMap framebuffer is not complete")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def begin(self):
        glViewport(0, 0, self.size, self.size)  
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glClear(GL_DEPTH_BUFFER_BIT)
        glCullFace(GL_FRONT)

    def end(self, width, height):
        glCullFace(GL_BACK)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, width, height)

    def bind(self, unit=1):
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.texture)