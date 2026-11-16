"""Animate a looping walk, keyframe it, measure it and export the finished character.

Usage: blender -b joyce_rigged.blend -P animate_walk.py -- <final.blend> <final.glb>

The legs use solved foot paths: each stance foot moves at a constant speed along the
ground, then each joint angle is calculated from where the foot must be. A loop of
24 frames covers two steps, with right-foot contact on frame 1 and left on frame 13.
"""
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poses  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
BLEND_OUT, GLB_OUT = args[0], args[1]

rig = bpy.data.objects["joyce_rig"]
scene = bpy.context.scene
FRAMES = 24
scene.frame_start = 1
scene.frame_end = FRAMES
scene.render.fps = 24

THIGH = np.linalg.norm(np.array((0.082, 0.0, 0.680)) - np.array((0.088, -0.006, 0.375)))
SHIN = np.linalg.norm(np.array((0.088, -0.006, 0.375)) - np.array((0.092, 0.010, 0.085)))
ANKLE_Z = 0.085
HIP_Z = 0.680
REACH = 0.21      # how far the foot lands in front of and behind the hip
STANCE = 0.60     # the share of the cycle a foot stays on the ground


def foot_path(t):
    """Return ankle y, ankle z and foot pitch in degrees at cycle position t."""
    t = t % 1.0
    if t < STANCE:
        u = t / STANCE
        y = -REACH + 2 * REACH * u
        pitch = 30.0 * max(u - 0.72, 0.0) / 0.28
        lift = 0.0
    else:
        u = (t - STANCE) / (1 - STANCE)
        eased = u * u * (3 - 2 * u)
        y = REACH - 2 * REACH * eased
        pitch = 30.0 * (1 - u) ** 2 - 9.0 * math.sin(u * math.pi / 2) ** 3
        lift = 0.055 * math.sin(u * math.pi)
    z = ANKLE_Z + lift + 0.15 * math.sin(math.radians(max(pitch, 0.0)))
    return y, z, pitch


def leg_angles(t, hip_z):
    """Solve thigh, knee and foot rotations in degrees from the foot target."""
    y, z, pitch = foot_path(t)
    dy, dz = y, z - hip_z
    distance = min(math.hypot(dy, dz), (THIGH + SHIN) * 0.999)
    direction = math.atan2(dy, -dz)
    at_hip = math.acos((THIGH**2 + distance**2 - SHIN**2) / (2 * THIGH * distance))
    at_ankle = math.acos((SHIN**2 + distance**2 - THIGH**2) / (2 * SHIN * distance))
    thigh = direction - at_hip
    shin = direction + at_ankle
    return math.degrees(thigh), math.degrees(shin - thigh), pitch - math.degrees(shin)


for frame in range(1, FRAMES + 2):
    t = (frame - 1) / FRAMES
    wave = math.cos(2 * math.pi * t)  # +1 when the right leg is in front
    hip_z = 0.6525 - 0.0125 * math.cos(4 * math.pi * t)  # highest as the leg passes under the hip
    r_thigh, r_knee, r_foot = leg_angles(t, hip_z)
    l_thigh, l_knee, l_foot = leg_angles(t + 0.5, hip_z)
    pose = {
        "_root": (0.0, 0.0, hip_z - HIP_Z),
        "hips": (0, 0, -6 * wave),
        "spine": (3, 1.5 * math.cos(4 * math.pi * t), 3 * wave),
        "chest": (1, 0, 5 * wave),
        "neck": (0, 0, -2 * wave),
        "head": (1, 0, -3 * wave),
        "thigh.R": (r_thigh, 0, 0), "shin.R": (r_knee, 0, 0), "foot.R": (r_foot, 0, 0),
        "thigh.L": (l_thigh, 0, 0), "shin.L": (l_knee, 0, 0), "foot.L": (l_foot, 0, 0),
        # Arms swing opposite to the legs, with the elbows softly bent.
        "upper_arm.R": (24 * wave - 2, -2, 0), "forearm.R": (-18 + 8 * wave, 0, 0),
        "upper_arm.L": (-24 * wave - 2, 2, 0), "forearm.L": (-18 - 8 * wave, 0, 0),
    }
    poses.apply_pose(rig, pose)
    for pb in rig.pose.bones:
        pb.keyframe_insert("rotation_quaternion", frame=frame)
    rig.pose.bones["root"].keyframe_insert("location", frame=frame)

rig.animation_data.action.name = "walk"

# ------------------------------------------------------------- measure the motion
depsgraph = bpy.context.evaluated_depsgraph_get()
shoe = bpy.data.objects["shoe_R"]
lowest = []
ankle_y = []
for frame in range(1, FRAMES + 1):
    scene.frame_set(frame)
    depsgraph.update()
    evaluated = shoe.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    coords = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", coords)
    lowest.append(float(coords[2::3].min()))
    evaluated.to_mesh_clear()
    ankle_y.append(float((rig.matrix_world @ rig.pose.bones["foot.R"].head).y))

stance_frames = [f for f in range(FRAMES) if f / FRAMES < STANCE * 0.70]
xs = np.array(stance_frames, dtype=float)
ys = np.array([ankle_y[f] for f in stance_frames])
fit = np.polyfit(xs, ys, 1)
slip_mm = float(np.abs(ys - np.polyval(fit, xs)).max() * 1000)
ground_values = [lowest[f] for f in stance_frames]
print("GROUND_CONTACT_MM min", round(min(ground_values) * 1000, 1), "max", round(max(ground_values) * 1000, 1))
print("LOWEST_SHOE_POINT_ALL_FRAMES_MM", round(min(lowest) * 1000, 1))
print("STANCE_SLIP_MM", round(slip_mm, 2), "belt_mm_per_frame", round(float(fit[0]) * 1000, 2))

scene.frame_set(1)
os.makedirs(os.path.dirname(BLEND_OUT), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=BLEND_OUT, compress=False)

# ------------------------------------------------------------- export and verify
bpy.ops.export_scene.gltf(
    filepath=GLB_OUT,
    export_format="GLB",
    export_animations=True,
    export_skins=True,
    export_apply=False,
    export_yup=True,
)
print("GLB_MB", round(os.path.getsize(GLB_OUT) / 1024 / 1024, 1))
print("ANIMATE_OK")
