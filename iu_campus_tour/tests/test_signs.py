import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphics.signs import arrow_angle, facing_axes, sign_image, sign_mesh, TEX_W, TEX_H


def test_arrow_is_relative_to_viewer():
    at, toward = (0, 0), (0, 10)                      # board faces +Z, viewer looks toward -Z (north)
    assert arrow_angle(at, toward, (0, -50)) == 0     # straight ahead
    assert arrow_angle(at, toward, (50, 0)) == 90     # east is on the viewer's right
    assert arrow_angle(at, toward, (-50, 0)) == 270   # west on the left
    assert arrow_angle(at, toward, (50, -50)) == 45
    assert arrow_angle(at, toward, (0, 50)) == 180
    assert arrow_angle((0, 0), (10, 0), (0, -50)) == 90    # board faces east, viewer looks west: north is on the right


def test_sign_faces_outward_and_front_uv():
    for toward in ((0, 10), (10, 0), (-7, 3), (4, -9)):
        (v, idx), box, ring = sign_mesh((5, 5), toward)
        tri = idx.reshape(-1, 3)
        pos = v[:, :3]
        n_geo = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
        n_decl = v[tri[:, 0], 3:6]
        assert np.all(np.einsum("ij,ij->i", n_geo, n_decl) > 0), "winding must match normal (face culling)"
        centre = np.array([5.0, 0.85, 5.0])
        assert np.all(np.einsum("ij,ij->i", pos[tri[:, 0]] - centre, n_decl) > 0), "normals point away from slab"
        f, r = facing_axes((5, 5), toward)
        front = [k for k in range(len(tri)) if np.allclose(n_decl[k], (f[0], 0, f[1]), atol=1e-6)]
        assert len(front) == 2
        fv = v[np.unique(tri[front])]
        assert sorted(map(tuple, fv[:, 6:8].round(3).tolist())) == [(0, 0), (0, 1), (1, 0), (1, 1)]
        left = fv[fv[:, 6] == 0][:, :3]               # u = 0 is the viewer's left edge
        right = fv[fv[:, 6] == 1][:, :3]
        assert np.all((right - left)[:, [0, 2]].mean(axis=0) @ r > 0), "text would be mirrored"
        assert box[0, 4] > 1.0 and len(ring) == 4


def test_sign_image_size_and_content():
    img = sign_image("DIRECTORY", [("TÒA NHÀ A1", "A1 Building", 0), ("CĂN TIN", "Canteen", 270)])
    assert img.shape == (TEX_H, TEX_W, 4)
    assert img[..., :3].max() == 255 and img[..., 3].min() == 255
