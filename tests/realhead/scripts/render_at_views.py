"""Render the head from the exact cameras solved for chosen video frames.

Usage: blender -b head.blend -P render_at_views.py -- <view_cameras.json> <out_dir> <scale> <samples> [--plain]
--plain renders a uniform skin colour, for judging shape alone.
"""
import json
import os
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.append(os.path.dirname(__file__))
import headcam  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
cams_path, out_dir, scale, samples = args[0], args[1], float(args[2]), int(args[3])
plain = "--plain" in args
os.makedirs(out_dir, exist_ok=True)
with open(cams_path, encoding="utf-8") as f:
    views = json.load(f)

first = next(iter(views.values()))
scene = headcam.setup_render(int(first["size"][0] * scale), int(first["size"][1] * scale), samples)
# The photos' pink paper backdrop, as the camera sees it, with neutral light on the face.
headcam.soft_studio(scene, background=(1.0, 1.0, 1.0), strength=float(os.environ.get("HEAD_WORLD", "1.6")),
                    seen_backdrop=(0.68, 0.27, 0.40), key=float(os.environ.get("HEAD_KEY", "40")),
                    fill=float(os.environ.get("HEAD_FILL", "20")))
if not plain:
    headcam.catchlights(scene, float(os.environ.get("HEAD_CATCH", "40")))
cam = headcam.camera(scene)

if plain:
    skin = bpy.data.materials.new("plain_skin")
    skin.use_nodes = True
    skin.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.25, 0.14, 0.10, 1)
    skin.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.5
    head = bpy.data.objects["head"]
    head.data.materials.clear()
    head.data.materials.append(skin)
    for poly in head.data.polygons:
        poly.use_smooth = True

for key, view in views.items():
    cam.data.lens = view["lens_mm"]
    rotation = Matrix(view["rotation"]).to_4x4()
    cam.matrix_world = Matrix.Translation(Vector(view["centre"])) @ rotation
    scene.render.filepath = os.path.join(out_dir, f"render_{key}.png")
    bpy.ops.render.render(write_still=True)
    print("RENDERED", key, flush=True)
print("RENDER_VIEWS_OK")
