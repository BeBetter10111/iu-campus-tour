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
    d.line((_px(cx - w / 2 + 0.09), H - _px(cy), _px(cx + w / 2 - 0.09), H - _px(cy)), fill=frame, width=2)  


def _make(kind):
    W, H = _px(TILE_W), _px(FLOOR_H)
    if kind == "maroon":                    
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
    else:                                   
        d = ImageDraw.Draw(img)
        for i in range(3):
            _window(d, (i + 0.5) * FLOOR_H, FLOOR_H * 0.52, 1.3, 1.1)
    return img


_cache = {}


def facade_texture(kind="plain"):
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
    tex = Texture(np.asarray(img.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)))
    _cache[kind] = tex
    return tex
