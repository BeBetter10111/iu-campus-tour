import numpy as np

F32 = np.float32

def normalize(v):
    v = np.asarray(v, dtype=F32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def translate(x, y, z):
    m = np.identity(4, dtype=F32)
    m[:3, 3] = (x, y, z)
    return m


def scale(x, y, z):
    return np.diag([x, y, z, 1.0]).astype(F32)


def rotate_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    m = np.identity(4, dtype=F32)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def rotate_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    m = np.identity(4, dtype=F32)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def rotate_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    m = np.identity(4, dtype=F32)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def rotate(axis, angle):
    x, y, z = normalize(axis)
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    m = np.identity(4, dtype=F32)
    m[:3, :3] = [
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c],
    ]
    return m


def look_at(eye, target, up):
    eye = np.asarray(eye, dtype=F32)
    f = normalize(np.asarray(target, dtype=F32) - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.identity(4, dtype=F32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)
    return m


def perspective(fov_y_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4), dtype=F32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m

def orthographic(left, right, bottom, top, near, far):
    m = np.identity(4, dtype=F32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m