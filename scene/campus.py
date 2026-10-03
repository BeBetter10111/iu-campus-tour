"""Dựng khuôn viên thử nghiệm: mặt đất + vài tòa nhà hộp có texture."""

import config
from graphics.texture import Texture
from graphics.mesh import create_box
from graphics.obj_loader import load_obj
from scene.scene import Scene, SceneObject


def _texture(filename, color_a, color_b):
    """Nạp ảnh từ assets/textures; thiếu file thì dùng texture caro để vẫn chạy."""
    path = config.TEXTURE_DIR / filename
    if path.exists():
        return Texture.from_file(path)
    print(f"[Texture] Không thấy {filename} -> dùng texture caro tạm")
    return Texture.checker(color_a=color_a, color_b=color_b)


def build_campus():
    scene = Scene()

    # Mỗi texture chỉ tạo 1 lần rồi dùng chung cho nhiều vật thể
    ground_tex = _texture("ground.jpg", (90, 150, 80), (70, 125, 60))
    wall_a = _texture("wall_a.jpg", (215, 200, 170), (190, 175, 145))
    wall_b = _texture("wall_b.jpg", (190, 110, 90), (165, 90, 70))

    # Mặt đất: đỉnh mặt phẳng ở y = 0 (tâm y = -0.1, dày 0.2). 1 ô texture = 6 m.
    ground = create_box((300, 0.2, 300), tex_size=6.0)
    scene.add(SceneObject(ground, ground_tex, position=(0, -0.1, 0), specular=0.02, shininess=8.0, cast_shadow=False))
    # (rộng, cao, sâu), (x, z), texture. Tâm tòa nhà ở y = cao/2 để đáy chạm đất.
    buildings = [
        ((40, 14, 16), (0, -40), wall_a),    # Tòa chính, đối diện điểm xuất phát
        ((16, 22, 16), (-35, -15), wall_b),  # Tòa trái
        ((16, 22, 16), (35, -15), wall_b),   # Tòa phải
        ((24, 10, 12), (30, 30), wall_a),    # Tòa phụ
    ]
    for size, (x, z), tex in buildings:
        mesh = create_box(size, tex_size=4.0)
        scene.add(SceneObject(mesh, tex, position=(x, size[1] / 2, z), specular=0.15, shininess=32.0))
    # Test OBJ loader: nếu có file assets/models/test.obj thì đặt thêm vào cảnh
    obj_path = config.MODEL_DIR / "test.obj"
    if obj_path.exists():
        scene.add(SceneObject(load_obj(obj_path), wall_b, position=(6, 0, 0), scale=(1.5, 1.5, 1.5)))
    return scene