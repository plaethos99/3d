"""Add a bold lash line along each upper lid's edge, as a thin dark strip of real geometry.

Usage: blender -b head.blend -P add_lash_strip.py -- <out.blend> [thickness_mm] [wing_mm] [r,g,b]

Animated characters' upper lashes usually read as one bold dark line hugging the top of the eye,
often running a little past the outer corner. The lid's edge is found on the mesh itself: around
the top of each eyeball, the head vertices closest to the eye's surface. A smooth curve through
them, slightly in front of the lid, is given a round thickness and a dark material.
"""
import math
import sys

import bpy
import numpy as np
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
out = args[0]
thickness = float(args[1]) / 1000 if len(args) > 1 else 0.0018
wing = float(args[2]) / 1000 if len(args) > 2 else 0.004
colour = tuple(float(v) for v in args[3].split(",")) if len(args) > 3 else (0.02, 0.01, 0.007)

head = bpy.data.objects["head"]
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = head.evaluated_get(depsgraph)
mesh = evaluated.to_mesh()
co = np.array([head.matrix_world @ v.co for v in mesh.vertices])
normals = np.array([(head.matrix_world.to_3x3() @ v.normal).normalized() for v in mesh.vertices])
evaluated.to_mesh_clear()

mat = bpy.data.materials.new("lash_line")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (*colour, 1)
bsdf.inputs["Roughness"].default_value = 0.55

for side in "LR":
    eye = bpy.data.objects[f"eye.{side}"]
    centre = np.array(eye.matrix_world.translation)
    radius = eye.dimensions.x / 2
    rel = co - centre
    gap = np.linalg.norm(rel, axis=1) - radius
    # Around the eye's top half, seen from the front: angle 0 at the inner or outer corner,
    # 90 degrees straight up. For each angle, keep the vertex nearest the eye's surface.
    angle = np.degrees(np.arctan2(rel[:, 2], rel[:, 0] * (1 if side == "L" else -1)))
    candidates = (gap > -0.001) & (gap < 0.006) & (-rel[:, 1] > 0.35 * radius) & (rel[:, 2] > -0.12 * radius)
    points = []
    for a in np.arange(-8, 190, 6):
        pick = candidates & (np.abs(angle - a) < 4)
        if not pick.any():
            continue
        k = np.nonzero(pick)[0][np.argmin(gap[pick])]
        points.append(co[k] + normals[k] * thickness * 0.4)
    if len(points) < 6:
        print("LASH", side, "lid edge not found")
        continue
    points = np.array(points)
    # Order from inner to outer corner, then extend past the outer corner a little (the "wing").
    order = np.argsort(points[:, 0] * (1 if side == "L" else -1))
    points = points[order]
    direction = points[-1] - points[-3]
    points = np.vstack([points, points[-1] + direction / np.linalg.norm(direction) * wing])
    curve = bpy.data.curves.new(f"lash.{side}", "CURVE")
    curve.dimensions = "3D"
    spline = curve.splines.new("NURBS")
    spline.points.add(len(points) - 1)
    for p, point in zip(spline.points, points):
        p.co = (*point, 1.0)
    spline.use_endpoint_u = True
    spline.order_u = 4
    curve.bevel_depth = thickness / 2
    curve.bevel_resolution = 3
    # Thicker in the middle, thin at both ends, like a drawn lash line.
    count = len(spline.points)
    for i, p in enumerate(spline.points):
        t = i / (count - 1)
        p.radius = 0.25 + 0.75 * math.sin(math.pi * min(t * 1.15, 1.0))
    obj = bpy.data.objects.new(f"lash.{side}", curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    print("LASH", side, "points", len(points))
bpy.ops.wm.save_as_mainfile(filepath=out)
print("LASH_STRIP_OK")
