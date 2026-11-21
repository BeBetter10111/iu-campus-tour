"""App: game loop, máy trạng thái (MENU / FREE / TOUR), render 2 pass (shadow + main)."""

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
    MENU = 0   # Menu chọn chế độ, camera đứng yên
    FREE = 1   # Đi bộ tự do: WASD + chuột + va chạm AABB
    TOUR = 2   # Guided tour: camera tự chạy dọc spline


class App:
    SPAWN_POS = (0.0, config.EYE_HEIGHT, 15.0)   # Điểm xuất phát của chế độ FREE
    SPAWN_YAW = -90.0                            # Nhìn về -Z (về phía tòa chính)

    # ------------------------------------------------------------------ khởi tạo
    def __init__(self):
        self.window = Window()
        self.timer = Timer()
        self.running = True

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glClearColor(*config.CLEAR_COLOR)

        # Shader và shadow
        self.shader = Shader.from_files("phong.vert", "phong.frag")
        self.depth_shader = Shader.from_files("shadow.vert", "shadow.frag")
        self.shadow_map = ShadowMap(config.SHADOW_MAP_SIZE)
        self.light = DirectionalLight()

        # Thế giới
        self.scene = build_campus()
        # --- OSM: điểm xuất phát lấy từ cảnh (nếu cảnh có), không thì dùng hằng số ở đầu class ---
        self.SPAWN_POS = getattr(self.scene, "spawn_pos", self.SPAWN_POS)
        self.SPAWN_YAW = getattr(self.scene, "spawn_yaw", self.SPAWN_YAW)
        self.skybox = Skybox()
        self.camera = FirstPersonCamera(position=self.SPAWN_POS, yaw=self.SPAWN_YAW)
        self.collision = CollisionWorld(self.scene.objects)

        # Tour: path đóng kín qua các waypoint trong config
        # --- OSM: tuyến tour tự động theo đường trong dữ liệu; không có thì dùng config.TOUR_WAYPOINTS ---
        tour_points = getattr(self.scene, "tour_waypoints", None) or config.TOUR_WAYPOINTS
        self.tour = TourController(CatmullRomPath(tour_points, closed=True))

        # UI: texture menu tạo 1 lần; texture HUD chỉ tạo lại khi % thay đổi
        self.overlay = Overlay()
        self.menu_tex = self.overlay.make_text_texture([
            "IU CAMPUS VIRTUAL TOUR",
            "",
            "[1]  Tự do khám phá",
            "[2]  Tham quan tự động",
            "[ESC]  Thoát",
        ], font_size=30, min_size=(560, 300))
        self.hud_tex = None
        self._hud_key = None            # (percent, paused, speed): đổi -> tạo lại texture HUD
        self.free_hint_tex = self.overlay.make_text_texture([
            "WASD đi   Shift chạy   Chuột nhìn   TAB nhả chuột",
            "M menu   B Phong/Blinn   N bóng   F2 debug   Phím mũi tên: mặt trời",
        ], title=False, font_size=20, pad=14)

        # Tùy chọn hiển thị
        self.use_blinn = False
        self.shadows_on = True
        self.debug_mode = 0
        self._shadow_dirty = True     # True để khung đầu tiên luôn render shadow map

        self.mode = Mode.MENU
        self.mouse_captured = False
        self.set_mode(Mode.MENU)

    # ------------------------------------------------------------ máy trạng thái
    def set_mode(self, new_mode):
        """Mọi chuyển chế độ đều đi qua đây để trạng thái chuột và camera luôn nhất quán."""
        old_mode = self.mode
        self.mode = new_mode

        if new_mode == Mode.MENU:
            self._set_mouse_capture(False)

        elif new_mode == Mode.FREE:
            self._set_mouse_capture(True)
            if old_mode == Mode.MENU:
                # Từ menu vào: về điểm xuất phát chuẩn
                self._respawn()
            else:
                # Từ TOUR sang: giữ hướng nhìn, hạ về độ cao mắt người
                self.camera.position[1] = config.EYE_HEIGHT
                # Điểm cuối tour có thể nằm trong hộp va chạm (độ cao y khác) -> về spawn cho an toàn
                if self.collision.collides(self.camera.position):
                    self._respawn()

        elif new_mode == Mode.TOUR:
            self._set_mouse_capture(True)    # Vẫn ẩn con trỏ cho đẹp, nhưng KHÔNG xoay camera
            self.tour.start(self.camera)

    def _respawn(self):
        self.camera.position[:] = self.SPAWN_POS
        self.camera.yaw, self.camera.pitch = self.SPAWN_YAW, 0.0
        self.camera._update_vectors()

    def _set_mouse_capture(self, enabled):
        self.mouse_captured = enabled
        pygame.event.set_grab(enabled)
        pygame.mouse.set_visible(not enabled)
        pygame.mouse.get_rel()               # Xả delta tích lũy để khung đầu không bị giật

    # ------------------------------------------------------------------- vòng lặp
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

    # -------------------------------------------------------------------- sự kiện
    def _handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.VIDEORESIZE:
                self.window.resize(e.w, e.h)
            elif e.type == pygame.KEYDOWN:
                self._on_key_down(e.key)

    def _on_key_down(self, key):
        # --- Phím riêng từng chế độ ---
        if self.mode == Mode.MENU:
            if key in (pygame.K_1, pygame.K_KP1):
                self.set_mode(Mode.FREE)
            elif key in (pygame.K_2, pygame.K_KP2):
                self.set_mode(Mode.TOUR)
            elif key == pygame.K_ESCAPE:
                self.running = False
            return   # Menu không dùng các phím debug bên dưới

        # Từ FREE/TOUR: ESC hoặc M quay lại menu
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
            elif key == pygame.K_g:                 # G: bay tự do (debug, tắt va chạm theo trục y)
                self.camera.fly_mode = not self.camera.fly_mode

        # --- Phím hiển thị dùng chung ở FREE và TOUR ---
        if key == pygame.K_b:
            self.use_blinn = not self.use_blinn
            print("Specular:", "Blinn-Phong" if self.use_blinn else "Phong")
        elif key == pygame.K_n:
            self.shadows_on = not self.shadows_on
            print("Shadows:", "ON" if self.shadows_on else "OFF")
        elif key == pygame.K_F2:
            self.debug_mode = (self.debug_mode + 1) % 3
            print("Debug mode:", ("normal", "shadow factor", "normals")[self.debug_mode])
        elif key == pygame.K_p:
            print(f"SUN_AZIMUTH = {self.light.azimuth:.0f}, "
                  f"SUN_ELEVATION = {self.light.elevation:.0f}")

    # --------------------------------------------------------------------- update
    def _update(self, dt):
        keys = pygame.key.get_pressed()

        if self.mode == Mode.FREE:
            if self.mouse_captured:
                dx, dy = pygame.mouse.get_rel()
                self.camera.process_mouse(dx, dy)
            # Camera chỉ TÍNH độ dời; CollisionWorld quyết định đi được bao xa (trượt dọc tường)
            delta = self.camera.get_move_delta(keys, dt)
            self.camera.position = self.collision.move(self.camera.position, delta)
            if self.camera.fly_mode:
                self.camera.position[1] = max(self.camera.position[1], 0.3)   # Không chui xuống đất
            else:
                self.camera.position[1] = config.EYE_HEIGHT

        elif self.mode == Mode.TOUR:
            pygame.mouse.get_rel()           # Xả delta chuột, tránh bị giật khi quay lại FREE
            self.tour.update(self.camera, dt)

        # Chỉnh hướng mặt trời bằng phím mũi tên (ở mọi chế độ trừ menu)
        if self.mode != Mode.MENU:
            d_az = (int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])) * 40.0 * dt
            d_el = (int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])) * 40.0 * dt
            if d_az or d_el:
                self.light.set_angles(self.light.azimuth + d_az, self.light.elevation + d_el)

        # Dời khung shadow map theo camera (có texel snapping). Chỉ render lại khi cần.
        self._shadow_dirty |= self.light.update(self.camera.position)

    # --------------------------------------------------------------------- render
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
        self.shadow_map.bind(1)              # Depth map ở unit 1 (unit 0 dành cho albedo)

        self.scene.draw(s)
        self.skybox.draw(view, proj, self.light.direction, self.light.color)

        # ---- Overlay 2D vẽ cuối cùng (đè lên cảnh) ----
        self._render_ui()

    def _render_ui(self):
        w, h = self.window.width, self.window.height

        if self.mode == Mode.MENU:
            self.overlay.draw_at(self.menu_tex, w, h, anchor="center")

        elif self.mode == Mode.FREE:
            self.overlay.draw_at(self.free_hint_tex, w, h, anchor="bottomleft")

        elif self.mode == Mode.TOUR:
            key = (int(self.tour.progress() * 100), self.tour.paused, round(self.tour.speed, 1))
            if key != self._hud_key:          # Chỉ tạo lại texture khi chữ thực sự đổi
                self._hud_key = key
                if self.hud_tex is not None:
                    self.hud_tex.delete()     # Tránh rò rỉ texture trên GPU
                percent, paused, speed = key
                state = "TẠM DỪNG" if paused else "ĐANG CHẠY"
                self.hud_tex = self.overlay.make_text_texture([
                    f"Tham quan: {percent}%  ({state})  {speed:.1f} m/s",
                    "SPACE tạm dừng   +/- tốc độ   M menu",
                ], title=False, font_size=20, pad=14)
            self.overlay.draw_at(self.hud_tex, w, h, anchor="topleft")
