import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BOARD_W, BOARD_H, BOARD_D = 0.80, 1.70, 0.12
TEX_W, TEX_H = 512, 1088                      # same aspect as the board (0.47)
MAROON = (78, 30, 34, 255)
WHITE = (255, 255, 255, 255)
SOFT = (225, 215, 212, 255)
FONTS_BOLD = ("arialbd.ttf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf")
FONTS_REG = ("arial.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf")
SIDE_UV = (0.03, 0.5)                          # plain maroon strip of the texture


def _font(names, size):
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def facing_axes(at, face_toward):
    """Return (front normal f, viewer's right r) in the XZ plane."""
    f = np.array(face_toward, float) - np.array(at, float)
    n = np.hypot(*f)
    f = np.array([0.0, 1.0]) if n < 1e-9 else f / n
    return f, np.array([f[1], -f[0]])


def arrow_angle(at, face_toward, target):
    """Arrow direction in degrees clockwise from 'up', snapped to 45 degrees.

    'Up' means straight ahead for a viewer standing in front of the board.
    """
    f, r = facing_axes(at, face_toward)
    d = np.array(target, float) - np.array(at, float)
    ang = math.degrees(math.atan2(float(d @ r), float(d @ -f)))
    return (round(ang / 45.0) * 45) % 360


def _quad(verts, idx, corners, n, uvs):
    """Append one quad whose winding is counter-clockwise seen from outside (normal n)."""
    corners, uvs = list(corners), list(uvs)
    p0, p1, p2 = (np.array(c, float) for c in corners[:3])
    if np.dot(np.cross(p1 - p0, p2 - p0), n) < 0:
        corners.reverse()
        uvs.reverse()
    st = len(verts)
    for c, uv in zip(corners, uvs):
        verts.append([c[0], c[1], c[2], n[0], n[1], n[2], uv[0], uv[1]])
    idx.extend([st, st + 1, st + 2, st, st + 2, st + 3])


def sign_mesh(at, face_toward, w=BOARD_W, h=BOARD_H, d=BOARD_D):
    """Build the slab mesh plus its collision box [x0,y0,z0,x1,y1,z1] and 2D footprint ring."""
    f, r = facing_axes(at, face_toward)
    c = np.array(at, float)

    def P(sr, y, sf):
        q = c + r * sr * w / 2 + f * sf * d / 2
        return (q[0], y, q[1])

    verts, idx = [], []
    uv_front = [(0, 0), (1, 0), (1, 1), (0, 1)]
    _quad(verts, idx, [P(-1, 0, 1), P(1, 0, 1), P(1, h, 1), P(-1, h, 1)],
          (f[0], 0.0, f[1]), uv_front)
    side = [SIDE_UV] * 4
    _quad(verts, idx, [P(1, 0, -1), P(-1, 0, -1), P(-1, h, -1), P(1, h, -1)],
          (-f[0], 0.0, -f[1]), side)
    _quad(verts, idx, [P(1, 0, 1), P(1, 0, -1), P(1, h, -1), P(1, h, 1)],
          (r[0], 0.0, r[1]), side)
    _quad(verts, idx, [P(-1, 0, -1), P(-1, 0, 1), P(-1, h, 1), P(-1, h, -1)],
          (-r[0], 0.0, -r[1]), side)
    _quad(verts, idx, [P(-1, h, 1), P(1, h, 1), P(1, h, -1), P(-1, h, -1)], (0.0, 1.0, 0.0), side)

    ring = [(P(sr, 0, sf)[0], P(sr, 0, sf)[2]) for sr, sf in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    xs, zs = [p[0] for p in ring], [p[1] for p in ring]
    box = np.array([[min(xs), 0.0, min(zs), max(xs), h, max(zs)]], np.float32)
    return (np.array(verts, np.float32), np.array(idx, np.uint32)), box, ring


def _draw_arrow(draw, cx, cy, angle, length=58, width=6):
    a = math.radians(angle)
    dx, dy = math.sin(a), -math.cos(a)                      # image y points down
    tip = (cx + dx * length / 2, cy + dy * length / 2)
    tail = (cx - dx * length / 2, cy - dy * length / 2)
    draw.line([tail, tip], fill=WHITE, width=width)
    for s in (-1, 1):
        b = a + math.pi + s * math.radians(38)
        draw.line([tip, (tip[0] + math.sin(b) * 24, tip[1] - math.cos(b) * 24)], fill=WHITE, width=width)


def sign_image(title, rows):
    """Paint the board. rows = [(vietnamese, english, arrow_angle_deg), ...]. Returns an RGBA array."""
    img = Image.new("RGBA", (TEX_W, TEX_H), MAROON)
    dr = ImageDraw.Draw(img)
    cx = TEX_W // 2
    dr.ellipse([cx - 40, 56, cx + 40, 136], fill=WHITE)                 # simple IU-style emblem
    dr.ellipse([cx - 31, 65, cx + 31, 127], fill=(38, 92, 170, 255))
    dr.text((cx, 96), "U", font=_font(FONTS_BOLD, 46), fill=WHITE, anchor="mm")
    dr.text((cx, 190), title, font=_font(FONTS_BOLD, 58), fill=WHITE, anchor="mm")
    dr.line([(0, 240), (TEX_W, 240)], fill=SOFT, width=3)
    top = 262
    row_h = min(150, (TEX_H - top - 30) // max(1, len(rows)))
    f_vi, f_en = _font(FONTS_BOLD, 38), _font(FONTS_REG, 29)
    for k, (vi, en, ang) in enumerate(rows):
        y = top + k * row_h
        dr.text((38, y + row_h * 0.30), vi, font=f_vi, fill=WHITE, anchor="lm")
        dr.text((38, y + row_h * 0.66), en, font=f_en, fill=SOFT, anchor="lm")
        _draw_arrow(dr, 440, y + row_h * 0.46, ang)
    return np.asarray(img.transpose(Image.Transpose.FLIP_TOP_BOTTOM))   # GL: row 0 = bottom


def build_sign(spec, destinations):
    """spec: {"at": [x, z], "face_toward": [x, z], "title": optional}. Returns (mesh, image, box, ring)."""
    at, toward = spec["at"], spec["face_toward"]
    rows = [(d["vi"], d["en"], arrow_angle(at, toward, d["at"])) for d in destinations]
    mesh, box, ring = sign_mesh(at, toward)
    return mesh, sign_image(spec.get("title", "DIRECTORY"), rows), box, ring
