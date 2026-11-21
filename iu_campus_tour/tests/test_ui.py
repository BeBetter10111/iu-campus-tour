import os
import sys
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.ui as ui


class StubTexture:
    def __init__(self, data):
        self.height, self.width = data.shape[:2]
        self.data = data


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.overlay = ui.Overlay.__new__(ui.Overlay)

    def test_text_texture_supports_overlay_options(self):
        with patch.object(ui, "Texture", StubTexture):
            texture = self.overlay.make_text_texture(
                ["IU CAMPUS VIRTUAL TOUR", "[1] Explore"],
                font_size=20,
                min_size=(560, 300),
            )

        self.assertEqual((texture.width, texture.height), (560, 300))
        self.assertEqual(texture.data[:, :, :3].max(), 255)

    def test_draw_at_positions_supported_anchors(self):
        texture = StubTexture(np.zeros((100, 200, 4), dtype=np.uint8))
        draw = Mock()
        self.overlay.draw = draw

        for anchor in ("center", "bottomleft", "topleft"):
            self.overlay.draw_at(texture, 1000, 500, anchor=anchor)

        draw.assert_has_calls([
            call(texture, (-0.2, -0.2, 0.4, 0.4)),
            call(texture, (-0.96, -0.92, 0.4, 0.4)),
            call(texture, (-0.96, 0.52, 0.4, 0.4)),
        ])


if __name__ == "__main__":
    unittest.main()
