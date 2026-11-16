"""Stretch Joyce's existing hair so it sits on the new head's skull instead of the old one's.

Usage: blender -b joyce_v2_rigged.blend -P refit_old_hair.py -- <out.blend>

The old hair was built around an ellipsoid skull (build_joyce.py: centre HEAD_C, radii HEAD_R,
moved forward by HEAD_SHIFT). The new skull is measured the same way: the extent of the new
head's vertices above the brows. Each axis of the hair is scaled and moved so one skull's box
becomes the other's. This is a stopgap until her hair is rebuilt as strands.
"""
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

out = sys.argv[sys.argv.index("--") + 1]
OLD_CENTRE = np.array((0.0, -0.018, 1.245))
OLD_RADII = np.array((0.152, 0.150, 0.175))

head = bpy.data.objects["head"]
co = np.array([head.matrix_world @ v.co for v in head.data.vertices])
eye_z = bpy.data.objects["eye.L"].matrix_world.translation.z
skull = co[co[:, 2] > eye_z + 0.035]
low, high = skull.min(axis=0), skull.max(axis=0)
new_centre = np.array(((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, OLD_CENTRE[2] + (high[2] - (OLD_CENTRE[2] + OLD_RADII[2]))))
new_radii = np.array(((high[0] - low[0]) / 2, (high[1] - low[1]) / 2, OLD_RADII[2]))
scale = new_radii / OLD_RADII
print("SKULL old radii", OLD_RADII, "new", np.round(new_radii, 3), "scale", np.round(scale, 3), "centre move", np.round(new_centre - OLD_CENTRE, 3))
transform = (Matrix.Translation(Vector(new_centre)) @ Matrix.Diagonal(Vector((*scale, 1.0)))
             @ Matrix.Translation(Vector(-OLD_CENTRE)))
for name in ("hair_base", "hair_inner"):
    obj = bpy.data.objects.get(name)
    if obj is None:
        continue
    local = obj.matrix_world.inverted() @ transform @ obj.matrix_world
    obj.data.transform(local)
    obj.data.update()
    print("HAIR_MOVED", name)
bpy.ops.wm.save_as_mainfile(filepath=out)
print("REFIT_HAIR_OK")
