from enum import Enum

import numpy as np
import pygame
from OpenGL.GL import (
    glClearColor, glClear, glEnable,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_CULL_FACE,
)

import config
from core.window import Window
from core.timer import Timer
from core.ui import Overlay
from graphics.shader import Shader
from graphics.shadow_map import ShadowMap
from scene.camera import FirstPersonCamera
from scene.campus import build_campus
from scene.collision import CollisionWorld
from scene.light import DirectionalLight
from scene.path import CatmullRomPath
from scene.skybox import Skybox
from scene.tour import TourController


class Mode(Enum):
    MENU = 0
    FREE = 1
    TOUR = 2


class App:
    SPAWN_POS = (0.0, config.EYE_HEIGHT, 15.0)
    SPAWN_YAW = -90.0

    def __init__(self, strict=False):
        self.window = Window()
        self.timer = Timer()
        self.running = True

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glClearColor(*config.CLEAR_COLOR)

        # Shader and shadow
        self.shader = Shader.from_files("phong.vert", "phong.frag")
        self.depth_shader = Shader.from_files("shadow.vert", "shadow.frag")
        self.shadow_map = ShadowMap(config.SHADOW_MAP_SIZE)
        self.light = DirectionalLight()

        self.scene = build_campus(strict=strict)
        self.SPAWN_POS = getattr(self.scene, "spawn_pos", self.SPAWN_POS)
        self.SPAWN_YAW = getattr(self.scene, "spawn_yaw", self.SPAWN_YAW)
        self.skybox = Skybox()
        self.camera = FirstPersonCamera(position=self.SPAWN_POS, yaw=self.SPAWN_YAW)
        self.collision = CollisionWorld(self.scene.objects)
        self.collision.dynamic_source = getattr(self.scene, "traffic", None)

        tour_points = getattr(self.scene, "tour_waypoints", None) or config.TOUR_WAYPOINTS
        self.tour = TourController(CatmullRomPath(tour_points, closed=True))

        self.overlay = Overlay()
        self.menu_tex = self.overlay.make_text_texture([
            "IU CAMPUS VIRTUAL TOUR",
            "",
            "[1]  Tự do khám phá",
            "[2]  Tham quan tự động",
            "[ESC]  Thoát",
        ], font_size=30, min_size=(560, 300))
        self.hud_tex = None
        self._hud_key = None
        self.free_hint_tex = self.overlay.make_text_texture([
            "WASD đi   Shift chạy   Chuột nhìn   TAB nhả chuột",
            "M menu   B Phong/Blinn   N bóng   F2 debug",
            "G bay/đi bộ   (khi bay: E/Space lên, Q/Ctrl xuống)",
        ], title=False, font_size=20, pad=14)
        self.fly_tex = self.overlay.make_text_texture([
            "CHẾ ĐỘ BAY   G để hạ cánh   E/Space lên   Q/Ctrl xuống",
        ], title=False, font_size=20, pad=14)
        self.notice_tex, self.notice_until = None, 0.0
        self._frame = 0

        self.use_blinn = False
        self.shadows_on = True
        self.debug_mode = 0
        self._shadow_dirty = True

        self.mode = Mode.MENU
        self.mouse_captured = False
        self.set_mode(Mode.MENU)

    def set_mode(self, new_mode):
        """Update application state and input capture consistently."""
        old_mode = self.mode
        self.mode = new_mode
        if new_mode != Mode.FREE or old_mode == Mode.MENU:
            self.camera.fly_mode = False

        if new_mode == Mode.MENU:
            self._set_mouse_capture(False)

        elif new_mode == Mode.FREE:
            self._set_mouse_capture(True)
            if old_mode == Mode.MENU:
                self._respawn()
            else:
                self.camera.position[1] = config.EYE_HEIGHT
                if self.collision.collides(self.camera.position):
                    self._respawn()

        elif new_mode == Mode.TOUR:
            self._set_mouse_capture(True)
            self.tour.start(self.camera)

    def _ground_at(self, x, z):
        height_at = getattr(self.scene, "height_at", None)
        return height_at(float(x), float(z)) if height_at else 0.0

    def _notify(self, text, seconds=2.5):
        if self.notice_tex is not None:
            self.notice_tex.delete()
        self.notice_tex = self.overlay.make_text_texture([text], title=False, font_size=22, pad=14)
        self.notice_until = self.timer.total + seconds

    def _toggle_fly(self):
        cam = self.camera
        if not cam.fly_mode:
            cam.fly_mode = True
            return
        # Landing: only if the whole column down to the ground is free (not on a roof / inside a wall).
        x, z = float(cam.position[0]), float(cam.position[2])
        target = self._ground_at(x, z) + config.EYE_HEIGHT
        y = float(cam.position[1])
        ok = True
        while True:
            if self.collision.collides(np.array([x, y, z], dtype=np.float32)):
                ok = False
                break
            if y <= target + 1e-3:
                break
            y = max(target, y - 0.25)
        if ok:
            cam.fly_mode = False
        else:
            self._notify("Không thể hạ cánh ở đây (có vật cản) - bay sang chỗ khác")

    def _respawn(self):
        self.camera.position[:] = self.SPAWN_POS
        self.camera.yaw, self.camera.pitch = self.SPAWN_YAW, 0.0
        self.camera._update_vectors()

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
                self.window.set_title(
                    f"{config.WINDOW_TITLE} | {self.mode.name} | {self.timer.fps:.0f} FPS")
        self.window.close()

    def _handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.VIDEORESIZE:
                self.window.resize(e.w, e.h)
            elif e.type == pygame.KEYDOWN:
                self._on_key_down(e.key)

    def _on_key_down(self, key):
        if self.mode == Mode.MENU:
            if key in (pygame.K_1, pygame.K_KP1):
                self.set_mode(Mode.FREE)
            elif key in (pygame.K_2, pygame.K_KP2):
                self.set_mode(Mode.TOUR)
            elif key == pygame.K_ESCAPE:
                self.running = False
            return

        if key in (pygame.K_ESCAPE, pygame.K_m):
            self.set_mode(Mode.MENU)
            return

        if self.mode == Mode.TOUR:
            if key == pygame.K_SPACE:
                self.tour.paused = not self.tour.paused
            elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.tour.speed = min(self.tour.speed + 0.5, 12.0)
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.tour.speed = max(self.tour.speed - 0.5, 0.5)

        elif self.mode == Mode.FREE:
            if key == pygame.K_TAB:
                self._set_mouse_capture(not self.mouse_captured)
            elif key == pygame.K_g:
                self._toggle_fly()

        if key == pygame.K_b:
            self.use_blinn = not self.use_blinn
            print("Specular:", "Blinn-Phong" if self.use_blinn else "Phong")
        elif key == pygame.K_n:
            self.shadows_on = not self.shadows_on
            print("Shadows:", "ON" if self.shadows_on else "OFF")
        elif key == pygame.K_F2:
            self.debug_mode = (self.debug_mode + 1) % 3
            print("Debug mode:", ("normal", "shadow factor", "normals")[self.debug_mode])

    def _update(self, dt):
        keys = pygame.key.get_pressed()
        self.light.advance(dt)

        if self.mode == Mode.FREE:
            if self.mouse_captured:
                dx, dy = pygame.mouse.get_rel()
                self.camera.process_mouse(dx, dy)
            delta = self.camera.get_move_delta(keys, dt)
            self.camera.position = self.collision.move(self.camera.position, delta)
            if self.camera.fly_mode:
                ground = self._ground_at(self.camera.position[0], self.camera.position[2])
                self.camera.position[1] = min(max(self.camera.position[1], ground + config.FLY_MIN_CLEARANCE),
                                              config.FLY_MAX_HEIGHT)
            else:
                # Keep the camera at eye height above the current walking surface.
                height_at = getattr(self.scene, "height_at", None)
                ground = height_at(float(self.camera.position[0]), float(self.camera.position[2])) if height_at else 0.0
                target = ground + config.EYE_HEIGHT
                self.camera.position[1] += (target - self.camera.position[1]) * min(1.0, 14.0 * dt)

        elif self.mode == Mode.TOUR:
            pygame.mouse.get_rel()
            self.tour.update(self.camera, dt)

        if self.mode != Mode.MENU and getattr(self.scene, "traffic", None) is not None:
            self.scene.update_dynamic(dt, (float(self.camera.position[0]), float(self.camera.position[2]))
                                      if self.mode == Mode.FREE else None)
            self._frame += 1
            if self._frame % 2 == 0:          # moving shadows: refresh the shadow map at half rate
                self._shadow_dirty = True

        self._shadow_dirty |= self.light.update(self.camera.position)

    def _render_shadow_pass(self):
        self.shadow_map.begin()
        self.depth_shader.use()
        self.depth_shader.set_mat4("uLightSpace", self.light.light_space)
        self.scene.draw_depth(self.depth_shader)
        self.shadow_map.end(self.window.width, self.window.height)

    def _render(self):
        if self.shadows_on and self._shadow_dirty:
            self._render_shadow_pass()
            self._shadow_dirty = False

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
        self.shadow_map.bind(1)

        self.scene.draw(s)
        self.skybox.draw(view, proj, self.light.direction, self.light.color)

        self._render_ui()

    def _render_ui(self):
        w, h = self.window.width, self.window.height

        if self.mode == Mode.MENU:
            self.overlay.draw_at(self.menu_tex, w, h, anchor="center")

        elif self.mode == Mode.FREE:
            self.overlay.draw_at(self.free_hint_tex, w, h, anchor="bottomleft")
            if self.camera.fly_mode:
                self.overlay.draw_at(self.fly_tex, w, h, anchor="topleft")
            if self.notice_tex is not None and self.timer.total < self.notice_until:
                self.overlay.draw_at(self.notice_tex, w, h, anchor="center")

        elif self.mode == Mode.TOUR:
            key = (int(self.tour.progress() * 100), self.tour.paused, round(self.tour.speed, 1))
            if key != self._hud_key:
                self._hud_key = key
                if self.hud_tex is not None:
                    self.hud_tex.delete()
                percent, paused, speed = key
                state = "TẠM DỪNG" if paused else "ĐANG CHẠY"
                self.hud_tex = self.overlay.make_text_texture([
                    f"Tham quan: {percent}%  ({state})  {speed:.1f} m/s",
                    "SPACE tạm dừng   +/- tốc độ   M menu",
                ], title=False, font_size=20, pad=14)
            self.overlay.draw_at(self.hud_tex, w, h, anchor="topleft")
