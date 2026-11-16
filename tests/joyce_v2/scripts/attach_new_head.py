"""Replace Joyce's hand-built head with the new head fitted to her references, on the same rig.

Usage: blender -b joyce_rigged.blend -P attach_new_head.py -- <head_detailed.blend> <head_placement.json> <out.blend>

Removes the old head, neck, ears, eyes, lids, lashes, creases and brows, keeps the body and (for
now) the hair, and brings in the new head, eyes and lash strips at the size and place solved by
place_head.py. The head stays upright; the reference's slight head turn belongs in a pose. Head
skin is weighted to the neck and head bones by height, the eyeballs to their own bones, which are
moved to the new eye centres so the eyes still turn in place.
"""
import json
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

head_blend, placement_path, out = sys.argv[sys.argv.index("--") + 1:][:3]
with open(placement_path, encoding="utf-8") as f:
    placement = json.load(f)
SCALE = placement["scale"]
MOVE = Vector((0.0, placement["translation"][1], placement["translation"][2]))  # centred left-right
TRANSFORM = Matrix.Translation(MOVE) @ Matrix.Scale(SCALE, 4)

OLD = ["head", "neck", "ear_L", "ear_R", "ear_rim_L", "ear_rim_R", "eye_L", "eye_R",
       "lid_upper_L", "lid_upper_R", "lid_lower_L", "lid_lower_R", "lash_L", "lash_R", "lash_low_L", "lash_low_R",
       "crease_L", "crease_R", "brow_L", "brow_R", "brow_hair_L", "brow_hair_R"]
for name in OLD:
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
rig = bpy.data.objects["joyce_rig"]

NEW = ["head", "eye.L", "eye.R", "lash.L", "lash.R"]
with bpy.data.libraries.load(head_blend, link=False) as (source, target):
    target.objects = [n for n in source.objects if n in NEW]
scene = bpy.context.scene
added = {}
for obj in target.objects:
    scene.collection.objects.link(obj)
    added[obj.name.split(".")[0] + ("." + obj.name.split(".")[1] if "." in obj.name else "")] = obj

# Lash strips are curves; turn them into meshes so they can be skinned.
for side in "LR":
    lash = bpy.data.objects.get(f"lash.{side}")
    if lash is not None and lash.type == "CURVE":
        bpy.ops.object.select_all(action="DESELECT")
        lash.select_set(True)
        bpy.context.view_layer.objects.active = lash
        bpy.ops.object.convert(target="MESH")

new_objects = [bpy.data.objects[n] for n in NEW if n in bpy.data.objects]
for obj in new_objects:
    world = TRANSFORM @ obj.matrix_world
    obj.data.transform(world)
    obj.matrix_world = Matrix.Identity(4)
    for mod in obj.modifiers:
        if mod.type == "DISPLACE":
            mod.strength *= SCALE  # wrinkle depth scales with the head
    if obj.name.startswith("eye."):
        # The eye's iris is painted relative to the object's own centre, so keep that at the eye's centre.
        centre = Vector(np.array([v.co for v in obj.data.vertices]).mean(axis=0))
        obj.data.transform(Matrix.Translation(-centre))
        obj.location = centre
    print("PLACED", obj.name, "verts", len(obj.data.vertices))

# The eye bones move to the new eye centres.
eye_centre = {side: bpy.data.objects[f"eye.{side}"].location.copy() for side in "LR"}
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
for side in "LR":
    bone = rig.data.edit_bones[f"eye.{side}"]
    length = bone.tail - bone.head
    bone.head = eye_centre[side]
    bone.tail = eye_centre[side] + length
    print("EYE_BONE", side, tuple(round(v, 3) for v in eye_centre[side]))
bpy.ops.object.mode_set(mode="OBJECT")

neck = rig.data.bones["neck"]
head_bone = rig.data.bones["head"]
low, high = neck.head_local.z, head_bone.head_local.z


def skin(obj, groups):
    """groups: function from vertex position to {bone: weight}."""
    for g in list(obj.vertex_groups):
        obj.vertex_groups.remove(g)
    made = {}
    for v in obj.data.vertices:
        for bone, weight in groups(v.co).items():
            if weight <= 0.0:
                continue
            if bone not in made:
                made[bone] = obj.vertex_groups.new(name=bone)
            made[bone].add([v.index], float(weight), "REPLACE")
    world = obj.matrix_world.copy()
    obj.parent = rig
    obj.matrix_parent_inverse = rig.matrix_world.inverted()
    obj.matrix_world = world
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    bpy.context.view_layer.objects.active = obj
    while obj.modifiers[0] != mod:
        bpy.ops.object.modifier_move_up(modifier=mod.name)


def head_weights(co):
    # Below the neck bone: chest; along the neck: blend to the head.
    if co.z <= low:
        t = min(max((co.z - (low - 0.05)) / 0.05, 0.0), 1.0)
        return {"chest": 1.0 - t, "neck": t}
    t = min(max((co.z - low) / (high - low), 0.0), 1.0)
    t = t * t * (3 - 2 * t)
    return {"neck": 1.0 - t, "head": t}


skin(bpy.data.objects["head"], head_weights)
for side in "LR":
    # Eyes ride on their bones as whole objects (not skinned), because the iris is painted in the
    # eye object's own coordinates: only a turning object turns its painted iris with it.
    eye = bpy.data.objects[f"eye.{side}"]
    world = eye.matrix_world.copy()
    eye.parent = rig
    eye.parent_type = "BONE"
    eye.parent_bone = f"eye.{side}"
    eye.matrix_world = world
    if f"lash.{side}" in bpy.data.objects:
        skin(bpy.data.objects[f"lash.{side}"], lambda co: {"head": 1.0})
bpy.ops.wm.save_as_mainfile(filepath=out)
print("ATTACH_OK", [o.name for o in new_objects])
