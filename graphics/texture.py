import numpy as np
from PIL import Image
from OpenGL.GL import (
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, glTexParameterf,
    glGenerateMipmap, glActiveTexture, glPixelStorei, glGetFloatv, glEnable,
    glDeleteTextures,
    GL_TEXTURE_2D, GL_TEXTURE_CUBE_MAP, GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_CUBE_MAP_SEAMLESS, GL_TEXTURE0,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R,
    GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT, GL_CLAMP_TO_EDGE,
    GL_RGB, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE, GL_UNPACK_ALIGNMENT,
)

_TEX_MAX_ANISOTROPY = 0x84FE
_MAX_TEX_MAX_ANISOTROPY = 0x84FF


class Texture:
    def __init__(self, data):
        data = np.ascontiguousarray(data, dtype=np.uint8)
        h, w = data.shape[:2]

        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)  
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)  # trilinear
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)  
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        try:
            max_aniso = glGetFloatv(_MAX_TEX_MAX_ANISOTROPY)
            glTexParameterf(GL_TEXTURE_2D, _TEX_MAX_ANISOTROPY, min(8.0, float(max_aniso)))
        except Exception:
            print("[Texture] No anisotropic filtering support (GL_EXT_texture_filter_anisotropic)")

    @classmethod
    def from_file(cls, path):
        img = Image.open(path).convert("RGBA")
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return cls(np.asarray(img))

    @classmethod
    def checker(cls, size=256, cells=8, color_a=(200, 200, 200), color_b=(120, 120, 120)):
        y, x = np.indices((size, size))
        step = size // cells
        mask = ((x // step) + (y // step)) % 2 == 0
        rgb = np.where(mask[..., None], np.array(color_a, np.uint8), np.array(color_b, np.uint8))
        alpha = np.full((size, size, 1), 255, np.uint8)
        return cls(np.concatenate([rgb.astype(np.uint8), alpha], axis=2))

    def bind(self, unit=0):
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.id)

    def delete(self):
        glDeleteTextures([self.id])


class Cubemap:
    FACE_NAMES = ("right", "left", "top", "bottom", "front", "back")

    def __init__(self, faces):
        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        for i, face in enumerate(faces):
            face = np.ascontiguousarray(face, dtype=np.uint8)
            h, w = face.shape[:2]
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, w, h, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, face)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        for axis in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R):
            glTexParameteri(GL_TEXTURE_CUBE_MAP, axis, GL_CLAMP_TO_EDGE)
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)  

    @classmethod
    def load_or_procedural(cls, folder):
        faces = []
        for name in cls.FACE_NAMES:
            path = next((folder / f"{name}{ext}" for ext in (".jpg", ".png")
                         if (folder / f"{name}{ext}").exists()), None)
            if path is None:
                print(f"[Skybox] No '{name}' in {folder} -> use procedural skybox")
                return cls.procedural()
            faces.append(np.asarray(Image.open(path).convert("RGB")))
        return cls(faces)

    @classmethod
    def procedural(cls, size=256, zenith=(0.25, 0.50, 0.90), horizon=(0.75, 0.88, 0.97), ground=(0.45, 0.42, 0.38)):
        zen, hor, gnd = (np.array(c, dtype=np.float32) for c in (zenith, horizon, ground))
        rows = np.arange(size, dtype=np.float32)
        e = (0.5 - (rows + 0.5) / size) * 2.0            
        up, down = np.clip(e, 0, 1)[:, None], np.clip(-e, 0, 1)[:, None]
        col = hor + (zen - hor) * up                     
        col = col + (gnd - col) * down                   
        side = np.repeat(col[:, None, :], size, axis=1)  
        top = np.broadcast_to(zen, (size, size, 3))
        bottom = np.broadcast_to(gnd, (size, size, 3))
        faces = [side, side, top, bottom, side, side]    
        return cls([(f * 255).astype(np.uint8) for f in faces])

    def bind(self, unit=0):
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.id)