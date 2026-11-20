"""Export Human Generator's bare base body (no preset) for the child-shape transfer.

Usage: blender -b -P pipeline/humgen/child/export_hg.py -- <out.npz> [--gender female|male]
Writes: basis (V,3) base-mesh points, rest (V,3) the body as shown (shape keys mixed; a livekey's offsets add
onto this), tris (F,3), the shape keys present and
their values, and the rig's rest joints {bone: head, tail} (armature space = world, rig at origin).
Prints EXPORT_HG_OK.
"""
import json
import os
import sys

import bpy
import numpy as np

from HumGen3D import Human

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
gender = argv[argv.index("--gender") + 1] if "--gender" in argv else "female"

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
human = Human._import_human(bpy.context, gender)
body, rig = human.objects.body, human.objects.rig
me = body.data
V = np.empty(len(me.vertices) * 3)
me.vertices.foreach_get("co", V)
me.calc_loop_triangles()
T = np.empty(len(me.loop_triangles) * 3, np.int64)
me.loop_triangles.foreach_get("vertices", T)
# HG's base mesh is female; the "Male" key (1 at import) makes it male. Presets switch it off for women.
me.shape_keys.key_blocks["Male"].value = 1.0 if gender == "male" else 0.0
keys = {k.name: (0.0 if k.mute else k.value) for k in me.shape_keys.key_blocks} if me.shape_keys else {}
# the rest shape as shown: shape keys mixed, modifiers (armature, masks) off so every point is kept
shown = [m for m in body.modifiers if m.show_viewport]
for m in shown:
    m.show_viewport = False
dg = bpy.context.evaluated_depsgraph_get()
ev = body.evaluated_get(dg).to_mesh()
E = np.empty(len(ev.vertices) * 3)
ev.vertices.foreach_get("co", E)
body.evaluated_get(dg).to_mesh_clear()
for m in shown:
    m.show_viewport = True
print("EVAL_MINUS_BASIS max", float(np.abs(E - V).max()))
joints = {b.name: [list(b.head_local), list(b.tail_local)] for b in rig.data.bones}
print("BODY", body.name, len(me.vertices), "verts", "matrix", [list(r) for r in body.matrix_world])
print("KEYS", json.dumps({k: round(v, 3) for k, v in keys.items()}))
print("RIG", rig.name, len(joints), "bones", "matrix", [list(r) for r in rig.matrix_world])
np.savez(out, basis=V.reshape(-1, 3), rest=E.reshape(-1, 3), tris=T.reshape(-1, 3), body_matrix=np.array(body.matrix_world),
         rig_matrix=np.array(rig.matrix_world), joints=json.dumps(joints), keys=json.dumps(keys))
print("EXPORT_HG_OK", out)
