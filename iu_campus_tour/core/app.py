import pygame
from OpenGL.GL import (
    glClearColor, glClear, glEnable,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_CULL_FACE,
)

import config
from core.window import Window
from core.timer import Timer
from graphics.shader import Shader
from graphics.mesh import create_cube
from graphics import math3d as m3d
from scene.camera import FirstPersonCamera


class App:
    def __init__(self):
        self.window = Window()
        self.timer = Timer()
        self.running = True

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glClearColor(*config.CLEAR_COLOR)

        self.shader = Shader.from_files("basic.vert", "basic.frag")
        self.cube = create_cube()
        self.camera = FirstPersonCamera()
        self._build_test_scene()

        self.mouse_captured = False
        self._set_mouse_capture(True)

    def _build_test_scene(self):
        self.models = [m3d.translate(0, -0.1, 0) @ m3d.scale(60, 0.2, 60)]  
        for x in range(-8, 9, 4):
            for z in range(-16, 1, 4):
                self.models.append(m3d.translate(x, 0.5, z))

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
        self.shader.use()
        self.shader.set_mat4("uView", self.camera.get_view_matrix())
        self.shader.set_mat4("uProjection", self.camera.get_projection_matrix(self.window.aspect))
        for model in self.models:
            self.shader.set_mat4("uModel", model)
            self.cube.draw()