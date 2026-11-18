"""Export an MPFB (MakeHuman, CC0 shapes) body at a given age, posed to Human Generator's limb directions.

Usage: blender -b -P pipeline/humgen/child/export_mpfb.py -- <hg.npz> <out.npz> --gender female --age 0.5
       [--race asian,caucasian,african weights e.g. 0.33,0.33,0.33] [--blend out.blend]
hg.npz: export_hg.py's (its rig's rest joints give the limb directions).
MPFB's age macro: 0 = 1 year, 0.1875 = 11, 0.5 = 25, 1 = 90 (linear between). The game-engine rig is added
and each limb bone (clavicle, upper arm, forearm, hand, thigh, shin, foot) turned to point the way HG's matching
bone does, so an adult and a child exported this way differ only by age, not by stance.
Writes: points (V,3) the posed body (helper geometry dropped), tris (F,3), joints {name: [head, tail]}.
Prints EXPORT_MPFB_OK.
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

from bl_ext.user_default.mpfb.services.humanservice import HumanService
from bl_ext.user_default.mpfb.services.targetservice import TargetService

argv = sys.argv[sys.argv.index("--") + 1:]
hg = np.load(os.path.abspath(argv[0]))
out = os.path.abspath(argv[1])


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


gender = opt("--gender", "female")
age = float(opt("--age", "0.5"))
race = [float(x) for x in opt("--race", "0.33,0.33,0.33").split(",")]

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
macro = TargetService.get_default_macro_info_dict()
macro["gender"] = 0.0 if gender == "female" else 1.0
macro["age"] = age
macro["race"] = {"asian": race[0], "caucasian": race[1], "african": race[2]}
human = HumanService.create_human(macro_detail_dict=macro, feet_on_ground=False)
rig = HumanService.add_builtin_rig(human, "game_engine")

# HG bone -> MPFB game-engine bone, per side; HG's .L is on +x (checked below), MPFB's _l likewise
HG_JOINTS = {k: (Vector(v[0]), Vector(v[1])) for k, v in json.loads(str(hg["joints"])).items()}
PAIRS = [("shoulder.{S}", "clavicle_{s}"), ("upper_arm.{S}", "upperarm_{s}"), ("forearm.{S}", "lowerarm_{s}"),
         ("hand.{S}", "hand_{s}"), ("thigh.{S}", "thigh_{s}"), ("shin.{S}", "calf_{s}"), ("foot.{S}", "foot_{s}")]
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
turned = []
for S, s in (("L", "l"), ("R", "r")):
    for hg_fmt, mp_fmt in PAIRS:
        hb, pb = HG_JOINTS.get(hg_fmt.format(S=S)), rig.pose.bones.get(mp_fmt.format(s=s))
        if hb is None or pb is None:
            continue
        bpy.context.view_layer.update()
        m = rig.matrix_world @ pb.matrix
        head = m.translation.copy()
        tail = rig.matrix_world @ (pb.matrix @ Vector((0, pb.length, 0)))
        want, have = (hb[1] - hb[0]), (tail - head)
        # HG's hand and foot bones may run a different way along the limb than MPFB's: only turn when close
        ang = have.normalized().angle(want.normalized())
        if hg_fmt.startswith(("hand", "foot")) and ang > 0.6:
            continue
        turn = have.normalized().rotation_difference(want.normalized()).to_matrix().to_4x4()
        world = Matrix.Translation(head) @ turn @ Matrix.Translation(-head) @ m
        pb.matrix = rig.matrix_world.inverted() @ world
        turned.append(f"{pb.name}:{ang * 57.3:.0f}")
bpy.ops.object.mode_set(mode="OBJECT")
bpy.context.view_layer.update()
print("TURNED", " ".join(turned))

# the posed body, helper geometry dropped
gi = human.vertex_groups["body"].index
BODY = np.array([v.index for v in human.data.vertices if any(g.group == gi for g in v.groups)])
masks = [m for m in human.modifiers if m.type == "MASK" and m.show_viewport]
for m in masks:
    m.show_viewport = False
dg = bpy.context.evaluated_depsgraph_get()
ev = human.evaluated_get(dg).to_mesh()
P = np.array([human.matrix_world @ v.co for v in ev.vertices])
human.evaluated_get(dg).to_mesh_clear()
for m in masks:
    m.show_viewport = True
human.data.calc_loop_triangles()
tris = np.array([t.vertices[:] for t in human.data.loop_triangles])
tris = tris[np.isin(tris, BODY).all(1)]
remap = np.full(len(P), -1)
remap[BODY] = np.arange(len(BODY))
P = P[BODY]
P[:, 2] -= P[:, 2].min()  # feet on the floor
joints = {}
for pb in rig.pose.bones:
    h = rig.matrix_world @ pb.head
    t = rig.matrix_world @ pb.tail
    joints[pb.name] = [list(h), list(t)]
print("BODY", len(P), "verts, height", round(float(P[:, 2].max()), 3), "width", round(float(np.ptp(P[:, 0])), 3))
np.savez(out, points=P, tris=remap[tris], body_index=BODY, joints=json.dumps(joints), age=age, race=race)
if "--blend" in argv:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(opt("--blend", "")))
print("EXPORT_MPFB_OK", out)
