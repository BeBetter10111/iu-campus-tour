"""Điểm vào của chương trình: python main.py"""

import config

# PHẢI đặt cờ này TRƯỚC khi import bất kỳ module OpenGL nào khác.
import OpenGL
OpenGL.ERROR_CHECKING = config.GL_ERROR_CHECKING

from core.app import App  # noqa: E402  (import sau khi cấu hình OpenGL)


def main():
    App().run()


if __name__ == "__main__":
    main()
