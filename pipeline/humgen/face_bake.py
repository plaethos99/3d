"""Bake a captured head's photo texture onto Human Generator's skin layout (step 2 of putting a face on HG).

Usage: blender -b -P pipeline/humgen/face_bake.py -- <head.blend> <face_dir> --name flow_woman
       [--gender female] [--preset Emma] [--colour <head's colour image>] [--coverage <photo coverage image>]
       [--size 4096]
face_dir: face_capture.py's (align.npz: the source head -> HG). The HG human is built with its face sliders
zeroed and face_<name> on, so HG's head has the captured shape and the source head lies on it.
Bakes (Cycles, selected to active, emission) onto HG's body UVs: <face_dir>/bake_colour.png (the photo colour,
alpha = where the source surface was found) and bake_coverage.png (where the photos saw the skin).
face_blend.py then blends them into an HG skin texture. Prints FACE_BAKE_OK.
"""
import os
import sys

import bpy
import numpy as np

from HumGen3D import Human
from HumGen3D.human.keys.keys import update_livekey_collection

argv = sys.argv[sys.argv.index("--") + 1:]
src_blend, face_dir = os.path.abspath(argv[0]), os.path.abspath(argv[1])


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


name = opt("--name", "face")
gender = opt("--gender", "female")
preset_name = opt("--preset", "Emma" if gender == "female" else "Jack")
src_dir = os.path.dirname(src_blend)
colour_path = os.path.abspath(opt("--colour", os.path.join(src_dir, "textures", "skin_colour.png")))
cover_path = os.path.abspath(opt("--coverage", os.path.join(src_dir, "textures", "photo_coverage.png")))
hair_path = os.path.abspath(opt("--hair", os.path.join(src_dir, "textures", "hair_mask.png")))
size = int(opt("--size", "4096"))
al = np.load(os.path.join(face_dir, "align.npz"))
R, ca, cb = al["R"], al["ca"], al["cb"]

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
preset = next(p for p in Human.get_preset_options(gender) if preset_name.lower() in p.lower())
human = Human.from_preset(preset)
update_livekey_collection()
for key in human.keys.all_livekeys:
    if key.category in ("face_proportions", "face_presets") and key.value:
        key.set_without_update(0.0)
human.keys.get(f"face_{name}").set_without_update(1.0)
human.keys.update_human_from_key_change(bpy.context)
body = human.objects.body
for m in body.modifiers:  # the whole skin, no hair
    if m.type in ("MASK", "PARTICLE_SYSTEM"):
        m.show_viewport = m.show_render = False

with bpy.data.libraries.load(src_blend, link=False) as (src, dst):
    dst.objects = ["head"]
src = dst.objects[0]
bpy.context.scene.collection.objects.link(src)
bpy.context.view_layer.update()
S = np.array([src.matrix_world @ v.co for v in src.data.vertices])
src.parent = None
src.matrix_world.identity()
src.data.vertices.foreach_set("co", ((R @ (S - ca).T).T + cb).ravel())
src.data.update()


def emission_material(image_path, colorspace):
    mat = bpy.data.materials.new("bake_src")
    nt = mat.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image_path, check_existing=True)
    tex.image.colorspace_settings.name = colorspace
    em = nt.nodes.new("ShaderNodeEmission")
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])
    return mat


# bake target: an image node, active, in every material slot of the body
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = 1
sc.cycles.device = "CPU"
bk = sc.render.bake
bk.use_selected_to_active = True
bk.cage_extrusion = 0.006
bk.max_ray_distance = 0.02
bk.margin = 8
bk.use_clear = True
for o in bpy.context.selected_objects:
    o.select_set(False)
src.select_set(True)
body.select_set(True)
bpy.context.view_layer.objects.active = body

results = {}
for label, path, cs in (("colour", colour_path, "sRGB"), ("coverage", cover_path, "Non-Color"),
                        ("hair", hair_path, "Non-Color")):
    if not os.path.exists(path):
        print("SKIP", label, "(no", path, ")")
        continue
    src.data.materials.clear()
    src.data.materials.append(emission_material(path, cs))
    img = bpy.data.images.new(f"bake_{label}", size, size, alpha=True, float_buffer=False)
    img.colorspace_settings.name = "sRGB" if label == "colour" else "Non-Color"
    img.generated_color = (0, 0, 0, 0)
    for mat in body.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        n = mat.node_tree.nodes.new("ShaderNodeTexImage")
        n.name = "bake_target"
        n.image = img
        mat.node_tree.nodes.active = n
    bpy.ops.object.bake(type="EMIT")
    out_path = os.path.join(face_dir, f"bake_{label}.png")
    img.filepath_raw = out_path
    img.file_format = "PNG"
    img.save()
    px = np.empty(size * size * 4, np.float32)
    img.pixels.foreach_get(px)
    results[label] = float((px[3::4] > 0.5).mean())
    for mat in body.data.materials:
        if mat and mat.use_nodes and "bake_target" in mat.node_tree.nodes:
            mat.node_tree.nodes.remove(mat.node_tree.nodes["bake_target"])
    print("BAKED", label, out_path, f"{results[label] * 100:.1f}% of the texture covered")
print("FACE_BAKE_OK", face_dir)
