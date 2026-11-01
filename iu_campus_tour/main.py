"""Điểm vào của chương trình: python main.py"""

import config

import OpenGL
OpenGL.ERROR_CHECKING = config.GL_ERROR_CHECKING

from core.app import App  # noqa: E402


def main():
    App().run()


if __name__ == "__main__":
    main()
