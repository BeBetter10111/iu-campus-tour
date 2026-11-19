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
    H = _px(FLOOR_H)
    def box(w_, h_):
        return (_px(cx - w_ / 2), H - _px(cy + h_ / 2), _px(cx + w_ / 2), H - _px(cy - h_ / 2))
    if surround:
        d.rectangle(box(w + 2 * surround_pad, h + 2 * surround_pad), fill=surround)
    d.rectangle(box(w, h), fill=frame)
    d.rectangle(box(w - 0.18, h - 0.18), fill=glass)
    d.line((_px(cx - w / 2 + 0.09), H - _px(cy), _px(cx + w / 2 - 0.09), H - _px(cy)), fill=frame, width=2)  # thanh ngang


def _make(kind):
    W, H = _px(TILE_W), _px(FLOOR_H)
    if kind == "maroon":                     # Tường đỏ đô, cửa sổ viền trắng nổi 
        img = Image.new("RGB", (W, H), MAROON)
        d = ImageDraw.Draw(img)
        for i in range(3):
            _window(d, (i + 0.5) * FLOOR_H, FLOOR_H * 0.52, 1.5, 1.4, surround=WHITE)
    elif kind == "white":                    # Tường trắng, dải đỏ mảnh ở mỗi sàn, cửa sổ viền nâu (khối trắng)
        img = Image.new("RGB", (W, H), WHITE)
        d = ImageDraw.Draw(img)
        d.rectangle((0, H - _px(0.14), W, H), fill=MAROON_BAND)
        for i in range(3):
            _window(d, (i + 0.5) * FLOOR_H, FLOOR_H * 0.5, 1.7, 1.25)
    elif kind == "concrete":                 # Công trình đang xây: bê tông trần
        img = Image.new("RGB", (W, H), CONCRETE)
        d = ImageDraw.Draw(img)
        d.rectangle((0, H - _px(0.2), W, H), fill=(120, 120, 116))
    elif kind == "roof":
        img = Image.new("RGB", (_px(6), _px(6)), ROOF)
    else:                                    # "plain": tường be, cửa sổ nhỏ (công trình chưa có ảnh)
        img = Image.new("RGB", (W, H), BEIGE)
        d = ImageDraw.Draw(img)
        for i in range(3):
            _window(d, (i + 0.5) * FLOOR_H, FLOOR_H * 0.52, 1.3, 1.1)
    return img


_cache = {}


def facade_texture(kind="plain"):
    """Texture mặt tiền theo loại ('white', 'maroon', 'concrete', 'plain', 'roof'), có cache."""
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
    # Texture GL có hàng 0 ở DƯỚI: lật dọc ảnh
    tex = Texture(np.asarray(img.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)))
    _cache[kind] = tex
    return tex


# =============================================================== Texture mặt đất
# Một ô texture phủ bao nhiêu mét (dùng để tính UV = mét / tile)
GRASS_TILE = 16.0
ASPHALT_TILE = 4.0
MARKED_TILE = 8.0        # Đường có vạch kẻ: v chạy dọc đường, một chu kỳ vạch = 8 m
PAVING_TILE = 2.0
CONCRETE_TILE = 4.0


def _periodic_noise(size, beta, rng):
    """Nhiễu 2D TUẦN HOÀN (ghép liền mạch khi lặp) với phổ công suất ~ 1/f^beta.
    Tạo nhiễu trắng, nhân phổ Fourier với 1/f^beta rồi biến đổi ngược: vì FFT vốn tuần hoàn
    nên ảnh nối liền ở các mép. beta lớn = đốm to, mềm; beta nhỏ = hạt mịn."""
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
    """Cỏ: xanh lá nền + đốm sáng/tối lớn (theo mảng cỏ) + hạt mịn. Liền mạch khi lặp."""
    rng = np.random.default_rng(seed)
    big, mid, fine = (_periodic_noise(size, b, rng) for b in (1.6, 1.0, 0.4))
    base = np.array([80, 124, 58], dtype=np.float64)
    tint = np.array([14, 18, 10], dtype=np.float64)
    rgb = base + big[..., None] * tint * 0.45 + mid[..., None] * tint * 0.35 + fine[..., None] * np.array([7, 9, 5])
    return _to_rgba(rgb)


def asphalt_array(size=256, marked=False, seed=11):
    """Nhựa đường xám đen có hạt. marked=True: thêm vạch kẻ (trắng ở mép, đứt đoạn ở giữa);
    u chạy ngang đường 0..1, v chạy dọc đường (chu kỳ 8 m: vạch dài 3 m, cách 5 m)."""
    rng = np.random.default_rng(seed)
    n1, n2 = _periodic_noise(size, 1.0, rng), _periodic_noise(size, 0.2, rng)
    rgb = np.array([60, 62, 66], dtype=np.float64) + n1[..., None] * 3 + n2[..., None] * 5
    speckle = rng.random((size, size)) > 0.985                      # vài hạt sỏi sáng
    rgb[speckle] += 22
    img = _to_rgba(rgb)
    if marked:
        white = np.array([225, 225, 220, 255], np.uint8)
        w = max(2, size // 64)
        for u0 in (0.045, 0.955 - w / size):                         # vạch liền ở hai mép
            c = int(u0 * size)
            img[:, c:c + w] = white
        c = size // 2 - w // 2                                       # vạch đứt ở giữa
        img[: int(size * 3 / 8), c:c + w] = white
    return img


def paving_array(size=256, seed=5):
    """Gạch lát vỉa hè: ô 0,25 m (8x8 ô trên 2 m), màu mỗi ô lệch nhẹ, mạch vữa tối."""
    rng = np.random.default_rng(seed)
    cells = 8
    step = size // cells
    rgb = np.zeros((size, size, 3))
    for cy in range(cells):
        for cx in range(cells):
            shade = rng.normal(0, 5)
            rgb[cy * step:(cy + 1) * step, cx * step:(cx + 1) * step] = np.array([188, 184, 174]) + shade
    rgb += _periodic_noise(size, 0.3, rng)[..., None] * 3
    rgb[:, ::step] -= 38                                             # mạch vữa
    rgb[::step, :] -= 38
    return _to_rgba(rgb)


def concrete_array(size=256, seed=3):
    """Bê tông nhám (bãi xe, đường đất nện)."""
    rng = np.random.default_rng(seed)
    rgb = np.array([150, 148, 142], dtype=np.float64) + _periodic_noise(size, 0.8, rng)[..., None] * 6 \
        + _periodic_noise(size, 0.2, rng)[..., None] * 4
    return _to_rgba(rgb)


_GROUND = {"grass": lambda: grass_array(), "asphalt": lambda: asphalt_array(),
           "asphalt_marked": lambda: asphalt_array(marked=True), "paving": lambda: paving_array(),
           "concrete": lambda: concrete_array()}
_ground_cache = {}


def ground_texture(kind):
    """Texture mặt đất theo loại ('grass', 'asphalt', 'asphalt_marked', 'paving', 'concrete'), có cache.
    Muốn dùng ảnh thật: đặt assets/textures/ground_<loại>.png (hoặc .jpg); ảnh nên liền mạch khi lặp."""
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
