import pygame
from OpenGL.GL import (
    glClearColor, glClear, glEnable,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_CULL_FACE,
)

import config
from core.window import Window
from core.timer import Timer
from graphics.shader import Shader
from scene.camera import FirstPersonCamera
from scene.campus import build_campus
from scene.skybox import Skybox


class App:
    def __init__(self):
        self.window = Window()
        self.timer = Timer()
        self.running = True

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glClearColor(*config.CLEAR_COLOR)

        self.shader = Shader.from_files("texture.vert", "texture.frag")
        self.scene = build_campus()
        self.skybox = Skybox()
        self.camera = FirstPersonCamera(position=(0.0, config.EYE_HEIGHT, 15.0))

        self.mouse_captured = False
        self._set_mouse_capture(True)

    def _set_mouse_capture(self, enabled):
        self.mouse_captured = enabled
        pygame.event.set_grab(enabled)
        pygame.mouse.set_visible(not enabled)
        pygame.mouse.get_rel()

    def run(self):
        while self.running:
            dt = self.timer.tick()
            self._handle_events()
            self._update(dt)
            self._render()
            self.window.swap()
            if self.timer.fps_updated:
                self.window.set_title(f"{config.WINDOW_TITLE} | {self.timer.fps:.0f} FPS")
        self.window.close()

    def _handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                elif e.key == pygame.K_TAB:
                    self._set_mouse_capture(not self.mouse_captured)
                elif e.key == pygame.K_f:
                    self.camera.fly_mode = not self.camera.fly_mode
            elif e.type == pygame.VIDEORESIZE:
                self.window.resize(e.w, e.h)

    def _update(self, dt):
        if self.mouse_captured:
            dx, dy = pygame.mouse.get_rel()
            self.camera.process_mouse(dx, dy)
        self.camera.process_keyboard(pygame.key.get_pressed(), dt)

    def _render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix(self.window.aspect)

        self.shader.use()
        self.shader.set_mat4("uView", view)
        self.shader.set_mat4("uProjection", proj)
        self.scene.draw(self.shader)

        self.skybox.draw(view, proj)