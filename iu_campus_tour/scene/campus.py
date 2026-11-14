import config
from graphics.texture import Texture
from graphics.mesh import create_box
from graphics.obj_loader import load_obj
from scene.scene import Scene, SceneObject


def _texture(filename, color_a, color_b):
    path = config.TEXTURE_DIR / filename
    if path.exists():
        return Texture.from_file(path)
    print(f"[Texture] Không thấy {filename} -> dùng texture caro tạm")
    return Texture.checker(color_a=color_a, color_b=color_b)


def build_campus():
    scene = Scene()

    ground_tex = _texture("ground.jpg", (90, 150, 80), (70, 125, 60))
    wall_a = _texture("wall_a.jpg", (215, 200, 170), (190, 175, 145))
    wall_b = _texture("wall_b.jpg", (190, 110, 90), (165, 90, 70))

    ground = create_box((300, 0.2, 300), tex_size=6.0)
    scene.add(SceneObject(ground, ground_tex, position=(0, -0.1, 0), specular=0.02, shininess=8.0, cast_shadow=False))

    buildings = [
        ((40, 14, 16), (0, -40), wall_a),   
        ((16, 22, 16), (-35, -15), wall_b),  
        ((16, 22, 16), (35, -15), wall_b),  
        ((24, 10, 12), (30, 30), wall_a),    
    ]
    for size, (x, z), tex in buildings:
        mesh = create_box(size, tex_size=4.0)
        scene.add(SceneObject(mesh, tex, position=(x, size[1] / 2, z), specular=0.15, shininess=8.0, cast_shadow=False))

    obj_path = config.MODEL_DIR / "test.obj"
    if obj_path.exists():
        scene.add(SceneObject(load_obj(obj_path), wall_b,
                              position=(6, 0, 0), scale=(1.5, 1.5, 1.5)))
    return scene