import numpy as np
from graphics.mesh import Mesh


def load_obj(path):
    positions, uvs, normals = [], [], []
    vertices, indices = [], []
    vertex_map = {}   

    def resolve(token, count):
        if token == "":
            return None
        i = int(token)
        return i - 1 if i > 0 else count + i

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            tag = parts[0]
            if tag == "v":
                positions.append([float(x) for x in parts[1:4]])
            elif tag == "vt":
                uvs.append([float(x) for x in parts[1:3]])
            elif tag == "vn":
                normals.append([float(x) for x in parts[1:4]])
            elif tag == "f":
                face = []
                for token in parts[1:]:
                    fields = token.split("/")
                    vi = resolve(fields[0], len(positions))
                    ti = resolve(fields[1], len(uvs)) if len(fields) > 1 else None
                    ni = resolve(fields[2], len(normals)) if len(fields) > 2 else None
                    key = (vi, ti, ni)
                    if key not in vertex_map:
                        vertex_map[key] = len(vertices)
                        vertices.append(
                            positions[vi]
                            + (normals[ni] if ni is not None else [0.0, 0.0, 0.0])
                            + (uvs[ti] if ti is not None else [0.0, 0.0])
                        )
                    face.append(vertex_map[key])
                for k in range(1, len(face) - 1):
                    indices += [face[0], face[k], face[k + 1]]

    verts = np.array(vertices, dtype=np.float32)
    idx = np.array(indices, dtype=np.uint32)
    if not normals:
        verts, idx = _flatten(verts, idx)
    return Mesh(verts, idx)


def _compute_normals(verts, idx):
    tri = idx.reshape(-1, 3)
    p0, p1, p2 = verts[tri[:, 0], :3], verts[tri[:, 1], :3], verts[tri[:, 2], :3]
    face_n = np.cross(p1 - p0, p2 - p0)
    acc = np.zeros((len(verts), 3), dtype=np.float32)
    for k in range(3):
        np.add.at(acc, tri[:, k], face_n)  
    length = np.linalg.norm(acc, axis=1, keepdims=True)
    length[length < 1e-8] = 1.0
    verts[:, 3:6] = acc / length

def _flatten(verts, idx):
    tri_verts = verts[idx]                      
    p0, p1, p2 = tri_verts[0::3, :3], tri_verts[1::3, :3], tri_verts[2::3, :3]
    n = np.cross(p1 - p0, p2 - p0)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    tri_verts[:, 3:6] = np.repeat(n, 3, axis=0)
    return tri_verts, np.arange(len(tri_verts), dtype=np.uint32)