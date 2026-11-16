"""Build Joyce's skeleton and compute skin weights.

Usage: blender -b joyce_model.blend -P rig_joyce.py -- <output.blend>

Each garment or body part only blends between the bones that can really move it.
Inside that list, weights fall off with distance to each bone. This keeps a sleeve
from following the torso, or the two legs from pulling on each other.
"""
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--") + 1]

HEAD_Y = -0.018
S = np.array((0.180, 0.0, 0.940))
E = np.array((0.216, 0.012, 0.735))
W = np.array((0.224, -0.030, 0.565))
H = np.array((0.082, 0.0, 0.680))
K = np.array((0.088, -0.006, 0.375))
A = np.array((0.092, 0.010, 0.085))
HAND_DIR = (W - E) / np.linalg.norm(W - E)

BONES = [
    ("root", (0, 0, 0), (0, 0, 0.12), None),
    ("hips", (0, 0, 0.70), (0, 0, 0.80), "root"),
    ("spine", (0, 0, 0.80), (0, 0, 0.92), "hips"),
    ("chest", (0, 0, 0.92), (0, 0, 1.03), "spine"),
    ("neck", (0, 0.0, 1.03), (0, HEAD_Y, 1.12), "chest"),
    ("head", (0, HEAD_Y, 1.12), (0, HEAD_Y, 1.34), "neck"),
]
for side, sign in (("L", 1), ("R", -1)):
    m = np.array((sign, 1, 1))
    eye = np.array((0.062, -0.103 + HEAD_Y, 1.243)) * m
    BONES += [
        (f"eye.{side}", eye, eye + np.array((0, -0.05, 0)), "head"),
        (f"shoulder.{side}", np.array((0.035, 0.0, 1.000)) * m, S * m, "chest"),
        (f"upper_arm.{side}", S * m, E * m, f"shoulder.{side}"),
        (f"forearm.{side}", E * m, W * m, f"upper_arm.{side}"),
        (f"hand.{side}", W * m, (W + HAND_DIR * 0.145) * m, f"forearm.{side}"),
        (f"thigh.{side}", H * m, K * m, "hips"),
        (f"shin.{side}", K * m, A * m, f"thigh.{side}"),
        (f"foot.{side}", A * m, np.array((0.098 * sign, -0.135, 0.030)), f"shin.{side}"),
    ]

bpy.ops.object.select_all(action="DESELECT")
data = bpy.data.armatures.new("joyce_rig")
rig = bpy.data.objects.new("joyce_rig", data)
bpy.context.scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
for name, head, tail, parent in BONES:
    bone = data.edit_bones.new(name)
    bone.head = Vector(tuple(head))
    bone.tail = Vector(tuple(tail))
    if parent:
        bone.parent = data.edit_bones[parent]
    # Connect only where the joint is shared, so chains move as one limb.
    if parent and (bone.parent.tail - bone.head).length < 1e-5:
        bone.use_connect = True
bpy.ops.object.mode_set(mode="OBJECT")

segments = {name: (np.array(head, dtype=float), np.array(tail, dtype=float)) for name, head, tail, _ in BONES}


def both(*names):
    out = []
    for name in names:
        out += [f"{name}.L", f"{name}.R"] if name in ("shoulder", "upper_arm", "thigh") else [name]
    return out


RULES = [
    # prefix, allowed bones, per-bone influence scale
    (("head", "hair", "ear", "lid", "brow", "lash", "crease"), {"head": 1.0}),
    (("neck",), {"chest": 0.5, "neck": 1.0, "head": 0.4}),
    (("chest_v", "collar"), {"chest": 1.0, "neck": 0.5}),
    (("blouse",), {"hips": 1.0, "spine": 1.0, "chest": 1.0, "neck": 0.7}),
    (("placket", "button"), {"hips": 1.0, "spine": 1.0, "chest": 1.0}),
    (("cardigan", "band", "hem", "pocket"), {
        "hips": 1.0, "spine": 1.0, "chest": 1.0, "neck": 0.25,
        "shoulder.L": 0.5, "shoulder.R": 0.5, "thigh.L": 0.65, "thigh.R": 0.65,
    }),
    (("hips",), {"hips": 1.0, "spine": 0.5, "thigh.L": 0.9, "thigh.R": 0.9}),
]
for side in ("L", "R"):
    RULES += [
        ((f"eye_{side}",), {f"eye.{side}": 1.0}),
        ((f"sleeve_{side}",), {f"shoulder.{side}": 0.7, f"upper_arm.{side}": 1.0, f"forearm.{side}": 1.0, f"hand.{side}": 0.25}),
        ((f"cuff_{side}",), {f"forearm.{side}": 1.0, f"hand.{side}": 0.6}),
        ((f"palm_{side}", f"finger0_{side}", f"finger1_{side}", f"finger2_{side}", f"finger3_{side}", f"thumb_{side}"), {f"hand.{side}": 1.0}),
        ((f"leg_{side}",), {"hips": 0.5, f"thigh.{side}": 1.0, f"shin.{side}": 1.0}),
        ((f"shoe_{side}", f"sole_{side}"), {f"foot.{side}": 1.0}),
    ]


def rule_for(name):
    # Longest prefix wins, so `eye_L` is not caught by a shorter rule.
    best = None
    for prefixes, bones in RULES:
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix):
                if best is None or len(prefix) > best[0]:
                    best = (len(prefix), bones)
    return best[1] if best else None


def distance_to_segment(points, a, b):
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


report = {"objects": 0, "vertices": 0, "unweighted": 0, "max_influences": 0, "sum_error": 0.0}
for obj in [o for o in bpy.data.objects if o.type == "MESH"]:
    bones = rule_for(obj.name)
    if bones is None:
        raise RuntimeError(f"No skinning rule for {obj.name}")
    count = len(obj.data.vertices)
    coords = np.empty(count * 3)
    obj.data.vertices.foreach_get("co", coords)
    points = coords.reshape(-1, 3)

    names = list(bones)
    if len(names) == 1:
        group = obj.vertex_groups.new(name=names[0])
        group.add(list(range(count)), 1.0, "REPLACE")
        report["max_influences"] = max(report["max_influences"], 1)
    else:
        scales = np.array([bones[n] for n in names])
        distances = np.stack([distance_to_segment(points, *segments[n]) for n in names], axis=1)
        weights = scales / (distances + 0.012) ** 4
        # Keep up to three bones per vertex, which suits real-time export.
        order = np.argsort(-weights, axis=1)
        keep = order[:, :3]
        kept = np.take_along_axis(weights, keep, axis=1)
        kept = kept / kept.sum(axis=1, keepdims=True)
        groups = {n: obj.vertex_groups.new(name=n) for n in names}
        for vertex in range(count):
            for k in range(keep.shape[1]):
                value = float(kept[vertex, k])
                if value > 0.004:
                    groups[names[keep[vertex, k]]].add([vertex], value, "REPLACE")
        report["max_influences"] = max(report["max_influences"], 3)

    # Measure the result rather than assuming it worked.
    for vertex in obj.data.vertices:
        total = sum(g.weight for g in vertex.groups)
        if total <= 0:
            report["unweighted"] += 1
        report["sum_error"] = max(report["sum_error"], abs(total - 1.0) if total else 0.0)

    obj.parent = rig
    modifier = obj.modifiers.new("skin", "ARMATURE")
    modifier.object = rig
    report["objects"] += 1
    report["vertices"] += count

os.makedirs(os.path.dirname(OUT), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=False)
print("RIG_OK", report, "bones", len(BONES))
