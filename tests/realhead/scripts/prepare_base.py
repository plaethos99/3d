"""Bring Blender Studio's realistic head into a clean file, centred and facing -Y, with eyes.

Usage: blender -b --factory-startup -P prepare_base.py -- <bundle.blend> <out.blend> <multires_level> [realistic|stylized]
The bundle's head carries its sculpted detail in a multiresolution modifier. That is applied at
the given level, so the detail becomes real geometry that can be reshaped and textured.
"""
import sys

import bpy
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
BUNDLE, OUT, LEVEL = args[0], args[1], int(args[2])
STYLE = args[3] if len(args) > 3 else "realistic"
if STYLE == "stylized":
    # Cartoon proportions: soft shapes and large eyes, like an animated feature character.
    NAMES = ["GEO-head_stylized", "GEO-head_stylized.eye.L", "GEO-head_stylized.eye.R"]
    RENAME = {NAMES[0]: "head", NAMES[1]: "eye.L", NAMES[2]: "eye.R"}
else:
    NAMES = [
        "GEO-head_animation_realistic",
        "GEO-head_animation_realistic.sclera.L", "GEO-head_animation_realistic.sclera.R",
        "GEO-head_animation_realistic.iris.L", "GEO-head_animation_realistic.iris.R",
    ]
    RENAME = {NAMES[0]: "head", NAMES[1]: "sclera.L", NAMES[2]: "sclera.R", NAMES[3]: "iris.L", NAMES[4]: "iris.R"}

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
with bpy.data.libraries.load(BUNDLE, link=False) as (source, target):
    target.objects = [n for n in source.objects if n in NAMES]
scene = bpy.context.scene
for obj in target.objects:
    scene.collection.objects.link(obj)

head = bpy.data.objects[NAMES[0]]
multires = next((m for m in head.modifiers if m.type == "MULTIRES"), None)
if multires is not None:
    print("MULTIRES total levels", multires.total_levels)
    multires.levels = min(LEVEL, multires.total_levels)
    multires.render_levels = multires.levels
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.modifier_apply(modifier=multires.name)

# Bake every transform into the meshes, so world and mesh coordinates are the same.
depsgraph = bpy.context.evaluated_depsgraph_get()
world = {obj.name: obj.matrix_world.copy() for obj in target.objects}
for obj in target.objects:
    obj.parent = None
for obj in target.objects:
    obj.data.transform(world[obj.name])
    obj.matrix_world = Matrix.Identity(4)

# Centre between the eyes, at the height of the eyes, and stand the neck on z = 0 later if needed.
eye_l = sum((v.co for v in bpy.data.objects[NAMES[1]].data.vertices), Vector()) / len(bpy.data.objects[NAMES[1]].data.vertices)
eye_r = sum((v.co for v in bpy.data.objects[NAMES[2]].data.vertices), Vector()) / len(bpy.data.objects[NAMES[2]].data.vertices)
centre = (eye_l + eye_r) / 2
for obj in target.objects:
    obj.data.transform(Matrix.Translation(-centre))

for obj in target.objects:
    obj.name = RENAME[obj.name]
    obj.data.name = obj.name
    print("PART", obj.name, "verts", len(obj.data.vertices), "faces", len(obj.data.polygons),
          "dims", tuple(round(v, 4) for v in obj.dimensions))
print("EYE_GAP_M", round((eye_l - eye_r).length, 4), "EYE_L_X", round(eye_l.x - centre.x, 4))
coords = [v.co for v in bpy.data.objects["head"].data.vertices]
print("HEAD_BOUNDS", [round(min(c[i] for c in coords), 3) for i in range(3)], [round(max(c[i] for c in coords), 3) for i in range(3)])
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print("PREPARE_OK")
