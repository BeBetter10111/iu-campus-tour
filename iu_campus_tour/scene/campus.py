import traceback

import config
from graphics.texture import Texture
from graphics.mesh import create_box
from graphics.obj_loader import load_obj
from scene.scene import Scene, SceneObject

DATA_DIR = getattr(config, "DATA_DIR", config.ASSET_DIR / "data")


def _texture(filename, color_a, color_b):
    path = config.TEXTURE_DIR / filename
    if path.exists():
        return Texture.from_file(path)
    print(f"[Texture] Không thấy {filename} -> dùng texture caro tạm")
    return Texture.checker(color_a=color_a, color_b=color_b)


def _obj(mesh, tex, position, cast_shadow=True, collidable=False, specular=0.15, shininess=32.0):
    o = SceneObject(mesh, tex, position=position)
    o.cast_shadow, o.collidable = cast_shadow, collidable
    o.specular, o.shininess = specular, shininess
    return o


def build_campus():
    data_path = DATA_DIR / "campus_osm.json"
    if data_path.exists():
        try:
            from scene.osm_campus import load_json, load_overrides, build_campus_from_osm
            ground = _texture("ground.jpg", (90, 150, 80), (70, 125, 60))
            return build_campus_from_osm(load_json(data_path),
                                         load_overrides(DATA_DIR / "building_overrides.json"), ground)
        except Exception:
            print("[Campus] Lỗi khi nạp dữ liệu OSM -> dùng cảnh thử nghiệm:")
            traceback.print_exc()
    else:
        print(f"[Campus] Chưa có {data_path.name} (chạy tools/fetch_osm.py) -> dùng cảnh thử nghiệm")
    return build_test_campus()


def build_test_campus():
    scene = Scene()
    ground_tex = _texture("ground.jpg", (90, 150, 80), (70, 125, 60))
    wall_a = _texture("wall_a.jpg", (215, 200, 170), (190, 175, 145))
    wall_b = _texture("wall_b.jpg", (190, 110, 90), (165, 90, 70))

    scene.add(_obj(create_box((300, 0.2, 300), tex_size=6.0), ground_tex, (0, -0.1, 0),
                   cast_shadow=False, specular=0.02, shininess=8.0))
    buildings = [
        ((40, 14, 16), (0, -40), wall_a),
        ((16, 22, 16), (-35, -15), wall_b),
        ((16, 22, 16), (35, -15), wall_b),
        ((24, 10, 12), (30, 30), wall_a),
    ]
    for size, (x, z), tex in buildings:
        scene.add(_obj(create_box(size, tex_size=4.0), tex, (x, size[1] / 2, z), collidable=True))

    obj_path = config.MODEL_DIR / "test.obj"
    if obj_path.exists():
        pyramid = SceneObject(load_obj(obj_path), wall_b, position=(6, 0, 0), scale=(1.5, 1.5, 1.5))
        pyramid.cast_shadow, pyramid.collidable, pyramid.specular, pyramid.shininess = True, True, 0.15, 32.0
        scene.add(pyramid)
    return scene
