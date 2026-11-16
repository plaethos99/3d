"""Pose library for Joyce, authored from the reference images.

Each entry maps a bone to (x, y, z) degrees, measured in the character's rest
axes so the same numbers mean the same thing on every bone:
  x: negative swings a hanging limb forward, positive tips the head or torso down
  y: on the left arm negative lifts it out to the side (positive on the right)
  z: positive turns towards her left
"_face": {shape key: amount}, e.g. {"surprised": 1}; face shapes not named are set to 0.
"_fingers.L" / "_fingers.R": degrees at each finger joint, about the finger's own knuckle axis:
  positive curls toward the palm, negative opens the hand (the rest pose is a relaxed curl).
"""
import math

from mathutils import Euler, Matrix, Quaternion, Vector

GAZE_TO_CAMERA = {"eye.L": (0, 0, -3), "eye.R": (0, 0, -3)}

POSES = {
    "rest": {},
    "stand": {
        "head": (1, 0, -2), "neck": (1, 0, 0),
        "upper_arm.L": (0, 1, 0), "upper_arm.R": (0, -1, 0),
        "forearm.L": (-5, 0, 0), "forearm.R": (-5, 0, 0),
        **GAZE_TO_CAMERA,
    },
    "sit": {
        "_root": (0, 0, -0.26),
        "thigh.L": (-85, 0, 6), "thigh.R": (-85, 0, -6),
        "shin.L": (84, 0, 0), "shin.R": (84, 0, 0),
        "foot.L": (1, 0, 0), "foot.R": (1, 0, 0),
        "spine": (-3, 0, 0),
        "upper_arm.L": (-24, 0, 0), "upper_arm.R": (-24, 0, 0),
        "forearm.L": (-46, 0, 0), "forearm.R": (-46, 0, 0),
        "hand.L": (28, 0, 0), "hand.R": (28, 0, 0),
        "head": (1, 0, -2), "neck": (1, 0, 0),
        **GAZE_TO_CAMERA,
    },
    "wave": {
        "upper_arm.L": (-14, -72, 0),
        "forearm.L": (-4, -98, 0),
        "hand.L": (0, -6, -70), "_fingers.L": -19,
        "upper_arm.R": (0, -1, 0), "forearm.R": (-5, 0, 0),
        "head": (-2, 0, -4), "neck": (0, 0, 0),
        **GAZE_TO_CAMERA,
    },
    "surprised": {
        "upper_arm.L": (-8, -30, 0), "upper_arm.R": (-8, 30, 0),
        "forearm.L": (-112, -34, 0), "forearm.R": (-112, 34, 0),
        "hand.L": (-8, -16, 60), "hand.R": (-8, 16, -60), "_fingers.L": -19, "_fingers.R": -19,
        "spine": (-3, 0, 0), "head": (-4, 0, -2), "_face": {"surprised": 1.0},
        "eye.L": (15, 0, -3), "eye.R": (15, 0, -3),
        **GAZE_TO_CAMERA,
    },
    # Joint-stress tests: every joint at a large angle.
    "stress_arms_up": {
        "upper_arm.L": (0, -88, 0), "upper_arm.R": (0, 88, 0),
        "forearm.L": (-60, 0, 0), "forearm.R": (-60, 0, 0),
    },
    "stress_knees": {
        "thigh.L": (-85, 0, 0), "shin.L": (95, 0, 0),
        "thigh.R": (-40, 0, 0), "shin.R": (70, 0, 0),
    },
    "stress_torso": {
        "spine": (18, 0, 12), "chest": (15, 8, 12), "head": (8, 12, 25),
    },
}


def set_expression(armature, amounts):
    """Set face shape keys (on every object that has them, e.g. head, tongue, teeth); others to 0."""
    import bpy
    for obj in bpy.data.objects:
        keys = getattr(obj.data, "shape_keys", None) if obj.data is not None else None
        if keys is None:
            continue
        for block in keys.key_blocks[1:]:
            block.value = float(amounts.get(block.name, 0.0))


def apply_pose(armature, pose, reset=True):
    """Set each bone's rotation, given in rest axes, as a local quaternion."""
    if reset:
        for pb in armature.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion()
            pb.location = Vector()
            pb.scale = Vector((1, 1, 1))
    if reset:
        set_expression(armature, {})
    for name, rotation in pose.items():
        if name == "_face":
            set_expression(armature, rotation)
            continue
        if name.startswith("_fingers."):
            side = name.split(".")[1]
            for pb in armature.pose.bones:
                if pb.name.endswith(f".{side}") and pb.name.startswith(("finger", "thumb")):
                    share = 0.5 if pb.name.startswith("thumb") else 1.0
                    pb.rotation_mode = "QUATERNION"
                    pb.rotation_quaternion = Quaternion(Vector((1, 0, 0)), math.radians(rotation * share))
            continue
        if name == "_root":
            root = armature.pose.bones["root"]
            root.location = root.bone.matrix_local.to_3x3().inverted() @ Vector(rotation)
            continue
        pb = armature.pose.bones[name]
        pb.rotation_mode = "QUATERNION"
        rest = pb.bone.matrix_local.to_3x3()
        world = Euler([math.radians(a) for a in rotation], "XYZ").to_matrix()
        local = rest.inverted() @ world @ rest
        pb.rotation_quaternion = local.to_quaternion()
    return armature


def build_chair(seat_z=0.335):
    """A simple wooden chair prop, built from boxes, for the seated reference."""
    import bmesh
    import bpy

    material = bpy.data.materials.new("chair_wood")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.13, 0.065, 0.03, 1)
    bsdf.inputs["Roughness"].default_value = 0.45
    boxes = [((0, 0.03, seat_z - 0.015), (0.40, 0.40, 0.03))]
    for x in (-0.18, 0.18):
        boxes.append(((x, -0.15, (seat_z - 0.03) / 2), (0.04, 0.04, seat_z - 0.03)))
        boxes.append(((x, 0.21, 0.40), (0.04, 0.04, 0.80)))
    boxes += [
        ((0, 0.21, 0.76), (0.40, 0.03, 0.08)),
        ((0, 0.21, 0.58), (0.40, 0.03, 0.04)),
        ((-0.18, 0.03, 0.14), (0.03, 0.36, 0.025)),
        ((0.18, 0.03, 0.14), (0.03, 0.36, 0.025)),
    ]
    bm = bmesh.new()
    for center, size in boxes:
        made = bmesh.ops.create_cube(bm, size=1.0)
        for v in made["verts"]:
            v.co.x = v.co.x * size[0] + center[0]
            v.co.y = v.co.y * size[1] + center[1]
            v.co.z = v.co.z * size[2] + center[2]
    mesh = bpy.data.meshes.new("chair")
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material)
    obj = bpy.data.objects.new("chair", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj