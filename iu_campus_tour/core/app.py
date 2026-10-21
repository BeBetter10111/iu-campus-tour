"""App: vòng lặp game chính (game loop) và xử lý sự kiện."""

import pygame
from OpenGL.GL import (
    glClearColor, glClear, glEnable,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_CULL_FACE,
)

import config
from core.window import Window


class App:
    """Điều phối: input -> update(dt) -> render -> swap.

    Delta time (dt) được tính ngay từ đầu để mọi chuyển động sau này
    (camera, animation) đều độc lập với FPS:  vị_trí += vận_tốc * dt
    """

    def __init__(self):
        self.window = Window()
        self.clock = pygame.time.Clock()
        self.running = True

        # Trạng thái GL toàn cục
        glEnable(GL_DEPTH_TEST)   # Z-buffer: pixel gần camera che pixel xa
        glEnable(GL_CULL_FACE)    # Bỏ mặt sau của tam giác -> giảm ~50% fragment
        glClearColor(*config.CLEAR_COLOR)

        self._fps_timer = 0.0
        self._frames = 0

    # ------------------------------------------------------------------ loop
    def run(self):
        while self.running:
            # tick() trả về mili-giây kể từ lần gọi trước -> đổi sang giây
            dt = self.clock.tick(config.FPS_CAP) / 1000.0

            self._handle_events()
            self._update(dt)
            self._render()
            self.window.swap()
            self._update_fps(dt)

        self.window.close()

    # ---------------------------------------------------------------- events
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.window.resize(event.w, event.h)

    # ---------------------------------------------------------------- update
    def _update(self, dt):
        """Cập nhật logic theo thời gian thực (camera, animation...) - Phase 2+."""
        pass

    # ---------------------------------------------------------------- render
    def _render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Vẽ scene ở đây (Phase 1 tiếp theo: shader + cube)

    # ------------------------------------------------------------------- fps
    def _update_fps(self, dt):
        """Hiển thị FPS trung bình trên tiêu đề, cập nhật mỗi 0.5 giây."""
        self._fps_timer += dt
        self._frames += 1
        if self._fps_timer >= config.FPS_TITLE_INTERVAL:
            fps = self._frames / self._fps_timer
            self.window.set_title(f"{config.WINDOW_TITLE} | {fps:.0f} FPS")
            self._fps_timer = 0.0
            self._frames = 0
