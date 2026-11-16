"""Make each eye a rigid object riding on its eye bone, instead of a mesh skinned to it.

Usage: blender -b character.blend -P eyes_on_bones.py -- <out.blend> [rig_name]
Eye materials that paint the iris in the eye object's own coordinates need this: skinning moves
the eye's surface but not its coordinates, so the painted iris would stay behind when the head
turns, and the eye would show only white.
"""
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
out = args[0]
rig = bpy.data.objects[args[1] if len(args) > 1 else "joyce_rig"]
old = rig.data.pose_position
rig.data.pose_position = "REST"
bpy.context.view_layer.update()
for side in "LR":
    eye = bpy.data.objects[f"eye.{side}"]
    world = eye.matrix_world.copy()
    for mod in [m for m in eye.modifiers if m.type == "ARMATURE"]:
        eye.modifiers.remove(mod)
    eye.vertex_groups.clear()
    eye.parent = rig
    eye.parent_type = "BONE"
    eye.parent_bone = f"eye.{side}"
    bpy.context.view_layer.update()
    eye.matrix_world = world
    print("EYE_ON_BONE", side)
rig.data.pose_position = old
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=out)
print("EYES_ON_BONES_OK")
