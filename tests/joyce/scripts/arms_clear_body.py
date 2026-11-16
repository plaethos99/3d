"""Keep a retargeted character's arms in front of its body, where the performer's arms were seen.

Usage: blender -b motion.blend -P arms_clear_body.py -- <pose.json> <out.blend> [margin_mm] [max_move_mm]

Motion copied from a slim performer onto a rounder character puts an arm that crossed in front of
the performer's chest inside the character's chest: the angles are right, but the body is deeper.
For each frame, wherever the performer's elbow or wrist was clearly visible but the character's
arm sits behind the front of its body (as seen from the camera), the elbow and wrist are slid
toward the camera, along the line of sight, until the sleeve clears the body. Sliding along the
line of sight leaves the pose looking the same from the camera; only its depth changes.
"""
import json
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

args = sys.argv[sys.argv.index("--") + 1:]
pose_path, out = args[0], args[1]
MARGIN = (float(args[2]) if len(args) > 2 else 15.0) / 1000
# Only act on clear evidence: the tracker must be confident it saw the joint, and the move must be
# no bigger than a body's depth difference could explain. Larger moves mean the arm really was
# behind the body (the far arm in a side view, say), so it is left alone.
SEEN = 0.7
MAX_MOVE = (float(args[3]) if len(args) > 3 else 150.0) / 1000
with open(pose_path, encoding="utf-8") as f:
    frames = json.load(f)["frames"]

scene = bpy.context.scene
rig = bpy.data.objects["joyce_rig"]
BODY = [n for n in ("cardigan", "band", "blouse", "hem", "pocket_L", "pocket_R", "hips", "placket", "chest_v")
        if n in bpy.data.objects]
SIDES = {"L": (13, 15), "R": (14, 16)}  # tracker elbow and wrist for each side

# How thick each sleeve is around its arm bones, so the sleeve's surface clears the body, not just the bone.
thickness = {}
for side in "LR":
    sleeve = bpy.data.objects.get(f"sleeve_{side}")
    bone = rig.data.bones[f"upper_arm.{side}"]
    a, b = np.array(bone.head_local), np.array(bone.tail_local)
    if sleeve is None:
        thickness[side] = 0.04
        continue
    co = np.array([sleeve.matrix_world @ v.co for v in sleeve.data.vertices])
    t = (co - a) @ (b - a) / np.dot(b - a, b - a)
    middle = (t > 0.3) & (t < 0.7)  # only the sleeve around the middle of the upper arm
    dist = np.linalg.norm(co[middle] - (a + np.outer(t[middle], b - a)), axis=1)
    thickness[side] = float(np.median(dist[dist < 0.12]))
print("SLEEVE_RADIUS_MM", {s: round(v * 1000, 1) for s, v in thickness.items()})


def body_tree():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    verts, polys = [], []
    for name in BODY:
        obj = bpy.data.objects[name].evaluated_get(depsgraph)
        mesh = obj.to_mesh()
        offset = len(verts)
        verts.extend(obj.matrix_world @ v.co for v in mesh.vertices)
        polys.extend([offset + i for i in p.vertices] for p in mesh.polygons)
        obj.to_mesh_clear()
    return BVHTree.FromPolygons(verts, polys)


def clear_point(tree, point, camera, radius):
    """Slide a point toward the camera until it is in front of the body by radius + margin."""
    to_cam = (camera - point).normalized()
    hit = tree.ray_cast(point, to_cam, (camera - point).length)
    if hit[0] is None:
        return point, 0.0  # nothing of the body between the point and the camera
    # Find the body's front surface along this line: keep stepping out until the line is clear.
    front = hit[0]
    while True:
        nxt = tree.ray_cast(front + to_cam * 0.0005, to_cam, (camera - front).length)
        if nxt[0] is None:
            break
        front = nxt[0]
    target = front + to_cam * (radius + MARGIN)
    return target, (target - point).length


def aim(pb, target):
    """Turn a pose bone so its tail points at target, keeping its head where it is."""
    head = rig.matrix_world @ pb.head
    current = (rig.matrix_world @ pb.tail - head).normalized()
    wanted = (target - head).normalized()
    turn = current.rotation_difference(wanted).to_matrix().to_4x4()
    m = rig.matrix_world @ pb.matrix
    loc = m.translation.copy()
    m.translation = Vector()
    m = turn @ m
    m.translation = loc
    pb.matrix = rig.matrix_world.inverted() @ m
    bpy.context.view_layer.update()


moved_frames = 0
moves = []
for f in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(f)
    record = frames[f - scene.frame_start] if f - scene.frame_start < len(frames) else None
    if record is None:
        continue
    seen = np.array(record["image"])[:, 2]
    hips = rig.matrix_world @ rig.pose.bones["hips"].head
    camera = Vector((hips.x, hips.y - 4.7, 1.02))
    tree = body_tree()
    changed = False
    for side, (elbow_id, wrist_id) in SIDES.items():
        upper, fore = rig.pose.bones[f"upper_arm.{side}"], rig.pose.bones[f"forearm.{side}"]
        radius = thickness[side]
        if seen[elbow_id] > SEEN:
            elbow = rig.matrix_world @ upper.tail
            mid = rig.matrix_world @ ((upper.head + upper.tail) / 2)
            new_elbow, d1 = clear_point(tree, elbow, camera, radius)
            new_mid, d2 = clear_point(tree, mid, camera, radius)
            if d2 > d1:  # the middle of the upper arm is the deeper one; move the elbow as much
                new_elbow = elbow + (new_mid - mid) * 2
            if 0.002 < max(d1, d2) <= MAX_MOVE:
                aim(upper, new_elbow)
                moves.append(max(d1, d2))
                changed = True
        if seen[wrist_id] > SEEN:
            wrist = rig.matrix_world @ fore.tail
            mid = rig.matrix_world @ ((fore.head + fore.tail) / 2)
            new_wrist, d1 = clear_point(tree, wrist, camera, radius * 0.8)
            new_mid, d2 = clear_point(tree, mid, camera, radius * 0.8)
            if d2 > d1:
                new_wrist = wrist + (new_mid - mid) * 2
            if 0.002 < max(d1, d2) <= MAX_MOVE:
                aim(fore, new_wrist)
                moves.append(max(d1, d2))
                changed = True
    if changed:
        moved_frames += 1
        for side in "LR":
            for name in (f"upper_arm.{side}", f"forearm.{side}"):
                rig.pose.bones[name].keyframe_insert("rotation_quaternion", frame=f)

total = scene.frame_end - scene.frame_start + 1
print("ARMS_MOVED_FRAMES", moved_frames, "of", total,
      "move mm median", round(float(np.median(moves)) * 1000, 1) if moves else 0,
      "max", round(float(max(moves)) * 1000, 1) if moves else 0)
scene.frame_set(scene.frame_start)
bpy.ops.wm.save_as_mainfile(filepath=out)
print("ARMS_CLEAR_OK")
