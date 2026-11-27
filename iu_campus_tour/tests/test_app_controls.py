import os
import sys
from types import SimpleNamespace

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.app import App, Mode


def test_space_starts_a_grounded_free_mode_jump_only_once():
    app = App.__new__(App)
    app.mode = Mode.FREE
    app.camera = SimpleNamespace(
        fly_mode=False,
        position=np.array([0.0, config.EYE_HEIGHT, 0.0], dtype=np.float32),
    )
    app.jump_velocity = 0.0
    app._ground_at = lambda x, z: 0.0

    app._on_key_down(pygame.K_SPACE)
    assert app.jump_velocity == config.JUMP_SPEED

    app._on_key_down(pygame.K_SPACE)
    assert app.jump_velocity == config.JUMP_SPEED
