"""Render a straight-on face close-up of a saved HG human (to compare with a photo, measure with MediaPipe).

Usage: blender -b <human.blend> -P pipeline/humgen/face_shot.py -- <out.png> [--yaw 0] [--size 900x1200]
Camera at the eyes' height, 85 mm lens, the head filling about 60% of the height; soft front light.
Prints FACE_SHOT_OK.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
yaw = float(argv[argv.index("--yaw") + 1]) if "--yaw" in argv else 0.0
w, h = (int(x) for x in (argv[argv.index("--size") + 1] if "--size" in argv else "900x1200").split("x"))
sc = bpy.context.scene
rig = next(o for o in bpy.data.objects if o.type == "ARMATURE" and "eyeball.L" in o.pose.bones)
dg = bpy.context.evaluated_depsgraph_get()
eyes = [rig.matrix_world @ rig.pose.bones[n].head for n in ("eyeball.L", "eyeball.R")]
centre = (eyes[0] + eyes[1]) / 2 + Vector((0, 0, -0.035))
for o in [o for o in bpy.data.objects if o.type in ("CAMERA", "LIGHT")]:
    bpy.data.objects.remove(o)
cam = bpy.data.objects.new("face_cam", bpy.data.cameras.new("face_cam"))
sc.collection.objects.link(cam)
cam.data.lens = 85
a = math.radians(yaw)
cam.location = centre + Vector((-math.sin(a) * 1.25, -math.cos(a) * 1.25, 0.02))
cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
sc.camera = cam
for name, off, energy, size in (("key", (0.6, -1.2, 0.5), 60, 0.5), ("fill", (-0.8, -1.0, 0.2), 30, 0.9),
                                ("top", (0, -0.3, 1.2), 20, 0.8)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy, ld.size = energy, size
    ld.shape = "DISK"  # round lights: round catchlights in the eyes, as in photos
    lo = bpy.data.objects.new(name, ld)
    sc.collection.objects.link(lo)
    lo.location = centre + Vector(off)
    lo.rotation_euler = (centre - lo.location).to_track_quat("-Z", "Y").to_euler()
if sc.world and sc.world.node_tree:
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.8, 0.8, 0.8, 1)
    sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
for mt in bpy.data.materials:  # Eevee renders HG's refracting dithered cornea as speckle (see make_human.py)
    if mt.name.startswith(".HG_Eyes_Outer"):
        mt.surface_render_method = "BLENDED"
        mt.use_raytrace_refraction = False
sc.render.engine = "BLENDER_EEVEE"
if hasattr(sc.eevee, "use_raytracing"):  # the eyes' cornea refracts: Eevee needs ray tracing for it
    sc.eevee.use_raytracing = True
sc.eevee.taa_render_samples = 128
sc.render.resolution_x, sc.render.resolution_y = w, h
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
print("FACE_SHOT_OK", out)
