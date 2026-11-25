import pygame
from OpenGL.GL import (
    glGetString, glViewport,
    GL_VERSION, GL_RENDERER, GL_SHADING_LANGUAGE_VERSION,
)

import config


class Window:
    def __init__(self, width=config.WINDOW_WIDTH, height=config.WINDOW_HEIGHT):
        pygame.init()

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, config.GL_MAJOR)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, config.GL_MINOR)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)  

        if config.MSAA_SAMPLES > 0:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, config.MSAA_SAMPLES)

        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        pygame.display.set_mode(
            (width, height), flags, vsync=1 if config.VSYNC else 0
        )
        pygame.display.set_caption(config.WINDOW_TITLE)

        self.width, self.height = width, height
        glViewport(0, 0, width, height)
        self._print_gl_info()

    @property
    def aspect(self):
        return self.width / max(self.height, 1)

    def resize(self, width, height):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)

    def swap(self):
        pygame.display.flip()

    def set_title(self, title):
        pygame.display.set_caption(title)

    def close(self):
        pygame.quit()

    @staticmethod
    def _print_gl_info():
        print("OpenGL  :", glGetString(GL_VERSION).decode())
        print("GLSL    :", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())
        print("Renderer:", glGetString(GL_RENDERER).decode())
