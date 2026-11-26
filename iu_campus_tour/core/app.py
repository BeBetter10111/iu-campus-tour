from enum import Enum

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
        self.warnings = []
        try:
            self.shadow_map = ShadowMap(config.SHADOW_MAP_SIZE)
            self._shadows_supported = True
        except Exception as e:
            print("[Shadow] Không tạo được shadow map, tắt bóng đổ:", e)
            self.shadow_map = None
            self._shadows_supported = False
            self.warnings.append("CẢNH BÁO: GPU không hỗ trợ bóng đổ, đã tắt")
        self.light = DirectionalLight()

        self.scene = build_campus(strict=strict)
        self.warnings += getattr(self.scene, "warnings", [])
        self.SPAWN_POS = getattr(self.scene, "spawn_pos", self.SPAWN_POS)
        self.SPAWN_YAW = getattr(self.scene, "spawn_yaw", self.SPAWN_YAW)
        self.skybox = Skybox()
        self.camera = FirstPersonCamera(position=self.SPAWN_POS, yaw=self.SPAWN_YAW)
        self.collision = CollisionWorld(self.scene.objects)
        self._jump_velocity = 0.0

        tour_points = getattr(self.scene, "tour_waypoints", None) or config.TOUR_WAYPOINTS
        self.tour = TourController(CatmullRomPath(tour_points, closed=True),
                                   height_fn=getattr(self.scene, "height_at", None))

        self.overlay = Overlay()
        self.menu_tex = self.overlay.make_text_texture([
            "IU CAMPUS VIRTUAL TOUR",
            "",
            "[1]  Free exploration",
            "[2]  Automatic tour",
            "[ESC]  Exit",
        ] + ([""] + self.warnings if self.warnings else []), font_size=30, min_size=(560, 300))
        self.hud_tex = None
        self._hud_key = None
        self.free_hint_tex = self.overlay.make_text_texture([
            "WASD move   Shift run   Mouse look   Space jump   TAB release mouse",
            "M menu   B Phong/Blinn   N shadows   F2 debug   Day/night changes automatically",
        ], title=False, font_size=20, pad=14)

        self.use_blinn = False
        self.shadows_on = self._shadows_supported
        self.debug_mode = 0
        self._shadow_dirty = True

        self.mode = Mode.MENU
        self.mouse_captured = False
        self.set_mode(Mode.MENU)

    def set_mode(self, new_mode):
        """Update application state and input capture consistently."""
        old_mode = self.mode
        self.mode = new_mode

        if new_mode == Mode.MENU:
            self._set_mouse_capture(False)

        elif new_mode == Mode.FREE:
            self._set_mouse_capture(True)
            if old_mode == Mode.MENU:
                self._respawn()
            else:
                self.camera.position[1] = config.EYE_HEIGHT
                self._jump_velocity = 0.0
                if self.collision.collides(self.camera.position):
                    self._respawn()

        elif new_mode == Mode.TOUR:
            self._set_mouse_capture(True)
            self.tour.start(self.camera)

    def _respawn(self):
        self.camera.position[:] = self.SPAWN_POS
        self.camera.yaw, self.camera.pitch = self.SPAWN_YAW, 0.0
        self._jump_velocity = 0.0
        self.camera._update_vectors()

    def _set_mouse_capture(self, enabled):
        self.mouse_captured = enabled
        pygame.event.set_grab(enabled)
        pygame.mouse.set_visible(not enabled)
        pygame.mouse.get_rel()

    def run(self):
        while self.running:
            dt = self.timer.tick(config.FPS_CAP)
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
        if key == pygame.K_F11:
            self.window.toggle_fullscreen()
            return
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
                self.camera.fly_mode = not self.camera.fly_mode
                self._jump_velocity = 0.0
            elif key == pygame.K_SPACE and not self.camera.fly_mode:
                height_at = getattr(self.scene, "height_at", None)
                ground = height_at(float(self.camera.position[0]), float(self.camera.position[2])) if height_at else 0.0
                if abs(float(self.camera.position[1]) - (ground + config.EYE_HEIGHT)) < 0.15:
                    self._jump_velocity = 6.5

        if key == pygame.K_b:
            self.use_blinn = not self.use_blinn
            print("Specular:", "Blinn-Phong" if self.use_blinn else "Phong")
        elif key == pygame.K_n:
            self.shadows_on = (not self.shadows_on) and self._shadows_supported
            print("Shadows:", "ON" if self.shadows_on else "OFF")
        elif key == pygame.K_F2:
            self.debug_mode = (self.debug_mode + 1) % 3
            print("Debug mode:", ("normal", "shadow factor", "normals")[self.debug_mode])

    def _update(self, dt):
        keys = pygame.key.get_pressed()

        if self.mode == Mode.FREE:
            if self.mouse_captured:
                dx, dy = pygame.mouse.get_rel()
                self.camera.process_mouse(dx, dy)
            delta = self.camera.get_move_delta(keys, dt)
            self.camera.position = self.collision.move(self.camera.position, delta)
            if self.camera.fly_mode:
                self.camera.position[1] = max(self.camera.position[1], 0.3)
            else:
                # Keep the camera at eye height above the current walking surface.
                height_at = getattr(self.scene, "height_at", None)
                ground = height_at(float(self.camera.position[0]), float(self.camera.position[2])) if height_at else 0.0
                target = ground + config.EYE_HEIGHT
                if self._jump_velocity != 0.0:
                    self._jump_velocity -= 18.0 * dt
                    vertical_delta = [0.0, self._jump_velocity * dt, 0.0]
                    self.camera.position = self.collision.move(self.camera.position, vertical_delta)
                    if self.camera.position[1] <= target:
                        self.camera.position[1] = target
                        self._jump_velocity = 0.0
                else:
                    self.camera.position[1] += (target - self.camera.position[1]) * min(1.0, 14.0 * dt)

        elif self.mode == Mode.TOUR:
            pygame.mouse.get_rel()
            self.tour.update(self.camera, dt)

        self.light.advance(dt)
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
        if self.shadow_map:
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
