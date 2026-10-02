"""App: game loop, input, update, render (2 pass: shadow + main)."""

import pygame
from OpenGL.GL import (
    glClearColor, glClear, glEnable,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_CULL_FACE,
)

import config
from core.window import Window
from core.timer import Timer
from graphics.shader import Shader
from graphics.shadow_map import ShadowMap
from scene.camera import FirstPersonCamera
from scene.campus import build_campus
from scene.light import DirectionalLight
from scene.skybox import Skybox


class App:
    def __init__(self):
        self.window = Window()
        self.timer = Timer()
        self.running = True

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glClearColor(*config.CLEAR_COLOR)

        self.shader = Shader.from_files("phong.vert", "phong.frag")
        self.depth_shader = Shader.from_files("shadow.vert", "shadow.frag")
        self.shadow_map = ShadowMap(config.SHADOW_MAP_SIZE)
        self.light = DirectionalLight()
        self.scene = build_campus()
        self.skybox = Skybox()
        self.camera = FirstPersonCamera(position=(0.0, config.EYE_HEIGHT, 15.0))

        # Tùy chọn hiển thị
        self.use_blinn = False
        self.shadows_on = True
        self.debug_mode = 0
        self._shadow_dirty = False

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
                elif e.key == pygame.K_b:       # B: Phong <-> Blinn-Phong
                    self.use_blinn = not self.use_blinn
                    print("Specular:", "Blinn-Phong" if self.use_blinn else "Phong")
                elif e.key == pygame.K_n:       # N: bật/tắt bóng
                    self.shadows_on = not self.shadows_on
                    print("Shadows:", "ON" if self.shadows_on else "OFF")
                elif e.key == pygame.K_F2:      # F2: chế độ debug 0 -> 1 -> 2 -> 0
                    self.debug_mode = (self.debug_mode + 1) % 3
                    print("Debug mode:", ("normal", "shadow factor", "normals")[self.debug_mode])
                elif e.key == pygame.K_p:
                    print(f"SUN_AZIMUTH = {self.light.azimuth:.0f}, "f"SUN_ELEVATION = {self.light.elevation:.0f}")
            elif e.type == pygame.VIDEORESIZE:
                self.window.resize(e.w, e.h)

    def _update(self, dt):
        keys = pygame.key.get_pressed()
        d_az = (int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])) * 40.0 * dt
        d_el = (int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])) * 40.0 * dt
        if d_az or d_el:
            self.light.set_angles(self.light.azimuth + d_az, self.light.elevation + d_el)
        if self.mouse_captured:
            dx, dy = pygame.mouse.get_rel()
            self.camera.process_mouse(dx, dy)
        self.camera.process_keyboard(pygame.key.get_pressed(), dt)
        # Dời khung shadow map theo người chơi (có snapping). Vật thể tĩnh nên chỉ cần
        # render lại shadow map khi khung dịch chuyển, đứng yên thì tiết kiệm cả 1 pass.
        self._shadow_dirty |= self.light.update(self.camera.position)

    # ------------------------------------------------------------------ render
    def _render_shadow_pass(self):
        self.shadow_map.begin()
        self.depth_shader.use()
        self.depth_shader.set_mat4("uLightSpace", self.light.light_space)
        self.scene.draw_depth(self.depth_shader)
        self.shadow_map.end(self.window.width, self.window.height)

    def _render(self):
        # ---- PASS 1: depth map từ góc nhìn mặt trời ----
        if self.shadows_on and self._shadow_dirty:
            self._render_shadow_pass()
            self._shadow_dirty = False

        # ---- PASS 2: cảnh chính ----
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix(self.window.aspect)

        s = self.shader
        s.use()
        s.set_mat4("uView", view)
        s.set_mat4("uProjection", proj)
        s.set_mat4("uLightSpace", self.light.light_space)
        s.set_vec3("uLightDir", *self.light.direction)
        s.set_vec3("uLightColor", *self.light.color)
        s.set_vec3("uAmbient", *self.light.ambient)
        s.set_vec3("uViewPos", *self.camera.position)
        s.set_int("uBlinn", int(self.use_blinn))
        s.set_int("uShadowsOn", int(self.shadows_on))
        s.set_int("uDebug", self.debug_mode)
        s.set_float("uShadowTexel", 1.0 / config.SHADOW_MAP_SIZE)
        s.set_float("uNormalOffset", self.light.normal_offset)
        s.set_float("uBias", config.SHADOW_BIAS)
        s.set_int("uShadowMap", 1)
        self.shadow_map.bind(1)          # Depth map ở texture unit 1 (unit 0 dành cho albedo)

        self.scene.draw(s)
        self.skybox.draw(view, proj)     # Skybox vẫn vẽ cuối cùng