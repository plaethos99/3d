"""Full-body check renders of a saved HG human: front, three-quarter and back, side by side.

Usage: blender -b <human.blend> -P pipeline/humgen/render_check.py -- <out.png> [--size 600x900]
Eevee, soft studio light, round lights, the camera framing the whole body. Prints RENDER_CHECK_OK.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
w, h = (int(x) for x in (argv[argv.index("--size") + 1] if "--size" in argv else "600x900").split("x"))
sc = bpy.context.scene
for o in [o for o in bpy.data.objects if o.type in ("CAMERA", "LIGHT")]:
    bpy.data.objects.remove(o)
meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.visible_get()]
pts = np.array([o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box])
lo, hi = pts.min(0), pts.max(0)
height = hi[2] - lo[2]
target = Vector((0, 0, lo[2] + height * 0.5))
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.lens = 50
for name, off, energy in (("key", (2, -3, 2.5), 500), ("fill", (-3, -2, 1.5), 200), ("rim", (0, 3, 3), 300)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy, ld.size, ld.shape = energy, 2.5, "DISK"
    lo_ = bpy.data.objects.new(name, ld)
    sc.collection.objects.link(lo_)
    lo_.location = target + Vector(off)
    lo_.rotation_euler = (target - lo_.location).to_track_quat("-Z", "Y").to_euler()
if sc.world and sc.world.node_tree:
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.35, 0.37, 1)
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = w, h
dist = height * 1.85
tiles = []
for label, yaw in (("front", 0), ("three_quarter", 35), ("back", 180)):
    a = math.radians(yaw)
    cam.location = target + Vector((math.sin(a) * dist, -math.cos(a) * dist, 0.05))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    path = out.replace(".png", f"_{label}.png")
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    tiles.append(path)
imgs = [bpy.data.images.load(p) for p in tiles]
sheet = bpy.data.images.new("sheet", w * 3, h)
px = np.zeros((h, w * 3, 4), np.float32)
for k, im in enumerate(imgs):
    a = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(a)
    px[:, k * w:(k + 1) * w] = a.reshape(h, w, 4)
sheet.pixels.foreach_set(px.ravel())
sheet.filepath_raw = out
sheet.file_format = "PNG"
sheet.save()
print("RENDER_CHECK_OK", out)
