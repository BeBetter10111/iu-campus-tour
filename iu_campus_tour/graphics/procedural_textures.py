import numpy as np
from PIL import Image, ImageDraw

import config
from graphics.texture import Texture

FLOOR_H = getattr(config, "FLOOR_HEIGHT", 3.6)
TILE_W = 3 * FLOOR_H
PX_PER_M = 40

WHITE = (240, 240, 236)
MAROON = (118, 34, 52)
MAROON_BAND = (150, 42, 58)
FRAME = (104, 62, 44)
GLASS = (92, 108, 124)
CONCRETE = (150, 150, 146)
BEIGE = (226, 220, 206)
ROOF = (140, 140, 136)


def _px(m):
    return int(round(m * PX_PER_M))


def _window(d, cx, cy, w, h, glass=GLASS, frame=FRAME, surround=None, surround_pad=0.35):
    """Draw a rectangular window at a height in meters."""
    H = _px(FLOOR_H)
    def box(w_, h_):
        return (_px(cx - w_ / 2), H - _px(cy + h_ / 2), _px(cx + w_ / 2), H - _px(cy - h_ / 2))
    if surround:
        d.rectangle(box(w + 2 * surround_pad, h + 2 * surround_pad), fill=surround)
    d.rectangle(box(w, h), fill=frame)
    d.rectangle(box(w - 0.18, h - 0.18), fill=glass)
    d.line((_px(cx - w / 2 + 0.09), H - _px(cy), _px(cx + w / 2 - 0.09), H - _px(cy)), fill=frame, width=2)


def _make(kind):
    W, H = _px(TILE_W), _px(FLOOR_H)
    if kind == "maroon":
        img = Image.new("RGB", (W, H), MAROON)
        d = ImageDraw.Draw(img)
        for i in range(3):
            _window(d, (i + 0.5) * FLOOR_H, FLOOR_H * 0.52, 1.5, 1.4, surround=WHITE)
    elif kind == "white":
        img = Image.new("RGB", (W, H), WHITE)
        d = ImageDraw.Draw(img)
        d.rectangle((0, H - _px(0.14), W, H), fill=MAROON_BAND)
        for i in range(3):
            _window(d, (i + 0.5) * FLOOR_H, FLOOR_H * 0.5, 1.7, 1.25)
    elif kind == "concrete":
        img = Image.new("RGB", (W, H), CONCRETE)
        d = ImageDraw.Draw(img)
        d.rectangle((0, H - _px(0.2), W, H), fill=(120, 120, 116))
    elif kind == "roof":
        img = Image.new("RGB", (_px(6), _px(6)), ROOF)
    elif kind == "white_plain":
        img = Image.new("RGB", (W, H), WHITE)
    elif kind == "interior_wall":
        img = Image.new("RGB", (W, H), (238, 236, 230))
        d = ImageDraw.Draw(img)
        d.rectangle((0, H - _px(0.14), W, H), fill=(196, 192, 184))
    elif kind == "interior_ceiling":
        img = Image.new("RGB", (_px(6), _px(6)), (246, 245, 242))
    elif kind == "interior_floor":
        img = Image.new("RGB", (_px(2), _px(2)), (204, 202, 196))
        d = ImageDraw.Draw(img)
        step = _px(2) // 4
        for k in range(4):
            d.line((k * step, 0, k * step, _px(2)), fill=(166, 164, 158), width=2)
            d.line((0, k * step, _px(2), k * step), fill=(166, 164, 158), width=2)
    elif kind == "fence":
        img = Image.new("RGB", (16, 16), (88, 94, 102))
    elif kind == "stair":
        img = Image.new("RGB", (16, 16), (234, 233, 229))
    elif kind == "stone":
        rng = np.random.default_rng(31)
        arr = np.clip(np.array([176, 172, 164]) + rng.normal(0, 6, (64, 64, 3)), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        img = Image.new("RGB", (W, H), BEIGE)
        d = ImageDraw.Draw(img)
        for i in range(3):
            _window(d, (i + 0.5) * FLOOR_H, FLOOR_H * 0.52, 1.3, 1.1)
    return img


_cache = {}


def facade_texture(kind="plain"):
    """Return a cached facade texture."""
    if kind in _cache:
        return _cache[kind]
    img = None
    for ext in (".png", ".jpg"):
        path = config.TEXTURE_DIR / f"facade_{kind}{ext}"
        if path.exists():
            img = Image.open(path).convert("RGB")
            print(f"[Facade] {kind}: dùng ảnh thật {path.name}")
            break
    if img is None:
        img = _make(kind)
    # Flip vertically because OpenGL texture row 0 starts at the bottom.
    tex = Texture(np.asarray(img.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)))
    _cache[kind] = tex
    return tex


GRASS_TILE = 16.0
ASPHALT_TILE = 4.0
MARKED_TILE = 8.0
PAVING_TILE = 2.0
CONCRETE_TILE = 4.0
SIDEWALK_TILE = 2.0


def _periodic_noise(size, beta, rng):
    """Generate seamless periodic noise."""
    f = np.fft.fft2(rng.standard_normal((size, size)))
    fx = np.fft.fftfreq(size)[:, None]
    fy = np.fft.fftfreq(size)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    r[0, 0] = 1.0
    n = np.real(np.fft.ifft2(f / r ** beta))
    return (n - n.mean()) / (n.std() + 1e-9)


def _to_rgba(rgb):
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=2)


def grass_array(size=512, seed=7):
    """Generate a seamless grass texture array."""
    rng = np.random.default_rng(seed)
    big, mid, fine = (_periodic_noise(size, b, rng) for b in (1.6, 1.0, 0.4))
    base = np.array([80, 124, 58], dtype=np.float64)
    tint = np.array([14, 18, 10], dtype=np.float64)
    rgb = base + big[..., None] * tint * 0.45 + mid[..., None] * tint * 0.35 + fine[..., None] * np.array([7, 9, 5])
    return _to_rgba(rgb)


def asphalt_array(size=256, marked=False, seed=11):
    """Generate an asphalt texture array with optional road markings."""
    rng = np.random.default_rng(seed)
    n1, n2 = _periodic_noise(size, 1.0, rng), _periodic_noise(size, 0.2, rng)
    rgb = np.array([60, 62, 66], dtype=np.float64) + n1[..., None] * 3 + n2[..., None] * 5
    speckle = rng.random((size, size)) > 0.985
    rgb[speckle] += 22
    img = _to_rgba(rgb)
    if marked:
        white = np.array([225, 225, 220, 255], np.uint8)
        w = max(2, size // 64)
        for u0 in (0.045, 0.955 - w / size):
            c = int(u0 * size)
            img[:, c:c + w] = white
        c = size // 2 - w // 2
        img[: int(size * 3 / 8), c:c + w] = white
    return img


def paving_array(size=256, seed=5):
    """Generate a paving texture array."""
    rng = np.random.default_rng(seed)
    cells = 8
    step = size // cells
    rgb = np.zeros((size, size, 3))
    for cy in range(cells):
        for cx in range(cells):
            shade = rng.normal(0, 5)
            rgb[cy * step:(cy + 1) * step, cx * step:(cx + 1) * step] = np.array([188, 184, 174]) + shade
    rgb += _periodic_noise(size, 0.3, rng)[..., None] * 3
    rgb[:, ::step] -= 38
    rgb[::step, :] -= 38
    return _to_rgba(rgb)


def concrete_array(size=256, seed=3):
    """Generate a concrete texture array."""
    rng = np.random.default_rng(seed)
    rgb = np.array([150, 148, 142], dtype=np.float64) + _periodic_noise(size, 0.8, rng)[..., None] * 6 \
        + _periodic_noise(size, 0.2, rng)[..., None] * 4
    return _to_rgba(rgb)


def sidewalk_array(size=256, seed=9):
    """Generate a terracotta pavement texture with the scroll pattern common on Vietnamese sidewalks."""
    rng = np.random.default_rng(seed)
    cells = 8
    step = size // cells
    img = Image.new("RGB", (size, size), (176, 100, 94))
    d = ImageDraw.Draw(img)
    for cy in range(cells):
        for cx in range(cells):
            x0, y0 = cx * step, cy * step
            shade = int(rng.normal(0, 4))
            base = (178 + shade, 102 + shade, 96 + shade)
            d.rectangle((x0, y0, x0 + step, y0 + step), fill=base)
            light = (198 + shade, 128 + shade, 118 + shade)
            flip = (cx + cy) % 2
            box = (x0 - step // 2, y0 - step // 2, x0 + step // 2, y0 + step // 2) if flip else \
                  (x0 + step // 2, y0 + step // 2, x0 + step * 3 // 2, y0 + step * 3 // 2)
            d.arc(box, 0, 360, fill=light, width=2)
            inner = tuple(v + (step // 6 if k < 2 else -step // 6) for k, v in enumerate(box))
            d.arc(inner, 0, 360, fill=light, width=1)
            d.rectangle((x0, y0, x0 + step - 1, y0 + step - 1), outline=(150, 82, 78))
    arr = np.asarray(img, dtype=np.float64) + _periodic_noise(size, 0.4, rng)[..., None] * 3
    return _to_rgba(arr)


_GROUND = {"grass": lambda: grass_array(), "asphalt": lambda: asphalt_array(),
           "asphalt_marked": lambda: asphalt_array(marked=True), "paving": lambda: paving_array(),
           "concrete": lambda: concrete_array(), "sidewalk": lambda: sidewalk_array()}
_ground_cache = {}


def ground_texture(kind):
    """Return a cached ground texture."""
    if kind in _ground_cache:
        return _ground_cache[kind]
    arr = None
    for ext in (".png", ".jpg"):
        path = config.TEXTURE_DIR / f"ground_{kind}{ext}"
        if path.exists():
            img = Image.open(path).convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            arr = np.asarray(img)
            print(f"[Ground] {kind}: dùng ảnh thật {path.name}")
            break
    if arr is None:
        arr = _GROUND[kind]()
    tex = Texture(arr)
    _ground_cache[kind] = tex
    return tex



FOLIAGE_SHADES = 8
_FOLIAGE_RGB = [(52, 104, 44), (64, 118, 48), (78, 132, 54), (58, 110, 62),
                (88, 138, 58), (46, 94, 52), (72, 124, 66), (96, 142, 62)]
_veg_cache = {}


def foliage_array(size=128):
    """Generate a foliage color texture array."""
    rng = np.random.default_rng(21)
    img = np.zeros((size, size, 3))
    col_w = size // FOLIAGE_SHADES
    vv = np.linspace(0.62, 1.12, size)[:, None]
    for k, base in enumerate(_FOLIAGE_RGB):
        block = np.array(base, float)[None, None, :] * vv[:, :, None] * np.ones((1, col_w, 1))
        block += rng.normal(0, 5, block.shape)
        img[:, k * col_w:(k + 1) * col_w] = block
    return _to_rgba(img)


def trunk_array(size=32):
    rng = np.random.default_rng(22)
    vv = np.linspace(0.75, 1.1, size)[:, None, None]
    rgb = np.array([104, 76, 52], float)[None, None, :] * vv * np.ones((1, size, 1))
    return _to_rgba(rgb + rng.normal(0, 4, rgb.shape))


def vegetation_textures():
    """Return cached foliage and trunk textures."""
    if "v" not in _veg_cache:
        out = []
        for name, fn in (("foliage", foliage_array), ("trunk", trunk_array)):
            arr = None
            for ext in (".png", ".jpg"):
                path = config.TEXTURE_DIR / f"veg_{name}{ext}"
                if path.exists():
                    arr = np.asarray(Image.open(path).convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM))
                    break
            out.append(Texture(arr if arr is not None else fn()))
        _veg_cache["v"] = tuple(out)
    return _veg_cache["v"]
