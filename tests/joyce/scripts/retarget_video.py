"""Turn tracked landmarks into keyframes on Joyce's rig.

Usage: blender -b joyce_rigged.blend -P retarget_video.py -- <pose.json> <out.blend> <out.glb> <report.json>
                [--truth truth.json]

Steps: smooth the landmarks, rotate each bone to match the tracked body segment, stand her
on the floor, work out travel from whichever foot is planted, then pin those feet with
two-bone IK. Only rotations carry over, so her own proportions are kept.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

args = sys.argv[sys.argv.index("--") + 1:]
POSE_JSON, BLEND_OUT, GLB_OUT, REPORT = args[:4]
TRUTH = args[args.index("--truth") + 1] if "--truth" in args else None

with open(POSE_JSON, encoding="utf-8") as handle:
    data = json.load(handle)
fps = data["fps"]

# ---------------------------------------------------------------- landmarks
raw = data["frames"]
count = len(raw)
world = np.full((count, 33, 3), np.nan)
visible = np.zeros((count, 33))
for i, frame in enumerate(raw):
    if frame is None:
        continue
    marks = np.array(frame["world"])
    # MediaPipe: x image right, y down, z away. Here: x her left when facing camera, y away, z up.
    world[i, :, 0] = marks[:, 0]
    world[i, :, 1] = marks[:, 2]
    world[i, :, 2] = -marks[:, 1]
    visible[i] = marks[:, 3]

# Bridge untracked frames by interpolating each coordinate across them.
times = np.arange(count)
good = ~np.isnan(world[:, 0, 0])
if good.sum() < 2:
    raise SystemExit("Too few tracked frames to animate.")
for joint in range(33):
    for axis in range(3):
        world[:, joint, axis] = np.interp(times, times[good], world[good, joint, axis])


def gaussian(values, sigma):
    radius = max(int(sigma * 3), 1)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([np.repeat(values[:1], radius, axis=0), values, np.repeat(values[-1:], radius, axis=0)])
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, padded)


# Scale smoothing with frame rate, so it means the same thing at 24 and 60 fps.
SMOOTH = max(0.6, 0.040 * fps)
world = gaussian(world.reshape(count, -1), SMOOTH).reshape(count, 33, 3)

NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_PI, R_PI, L_IN, R_IN = 17, 18, 19, 20
L_HI, R_HI, L_KN, R_KN, L_AN, R_AN, L_TO, R_TO = 23, 24, 25, 26, 27, 28, 31, 32
NOSE_BELOW_EARS = math.radians(15)  # anatomical estimate for a level human head

# Foot contact limits, in metres and metres per second on Joyce's own scale.
CONTACT_HEIGHT = 0.07
CONTACT_SPEED = 0.22
HIP_RANGE_PERCENTILE = 75
CONTACT_SPEED_SHARE = 0.35  # also allowed this share of the body's speed, for faster walking

# ---------------------------------------------------------------- rig facts
rig = bpy.data.objects["joyce_rig"]
bones = rig.data.bones
rest_dir = {b.name: (b.tail_local - b.head_local).normalized() for b in bones}
rest3 = {b.name: b.matrix_local.to_3x3() for b in bones}
head = {b.name: b.head_local.copy() for b in bones}
tail = {b.name: b.tail_local.copy() for b in bones}
parent = {b.name: (b.parent.name if b.parent else None) for b in bones}


def vec(p):
    return Vector(tuple(float(v) for v in p))


def frame_quat(x_axis, up):
    """Rotation taking rest axes to a frame with this left-right axis and up direction."""
    x = x_axis.normalized()
    z = (up - up.dot(x) * x).normalized()
    y = z.cross(x)
    return Matrix((x, y, z)).transposed().to_quaternion()


def swing(parent_q, bone, target):
    """Rotate a bone from where its parent leaves it, to aim along the target direction."""
    current = parent_q @ rest_dir[bone]
    if target.length < 1e-6:
        return parent_q.copy()
    return current.rotation_difference(target.normalized()) @ parent_q


def solve_frame(P):
    q = {"root": Quaternion()}
    hip_mid = (P[L_HI] + P[R_HI]) / 2
    sh_mid = (P[L_SH] + P[R_SH]) / 2
    trunk = sh_mid - hip_mid
    q["hips"] = frame_quat(P[L_HI] - P[R_HI], trunk)
    q["chest"] = frame_quat(P[L_SH] - P[R_SH], trunk)
    q["spine"] = q["hips"].slerp(q["chest"], 0.5)
    ear_mid = (P[L_EAR] + P[R_EAR]) / 2
    # Ears give the heading and the nose gives the nod. The neck's natural forward slope says
    # nothing about the nod, so it isn't used. On a level head the nose sits below the ears.
    ear_axis = (P[L_EAR] - P[R_EAR]).normalized()
    to_nose = P[NOSE] - ear_mid
    to_nose = (to_nose - to_nose.dot(ear_axis) * ear_axis).normalized()
    head_up = to_nose.cross(ear_axis) * math.cos(NOSE_BELOW_EARS) - to_nose * math.sin(NOSE_BELOW_EARS)
    # Ear depth is unreliable, so the turn away from the chest is limited to 30 degrees.
    head_q = frame_quat(ear_axis, head_up)
    difference = q["chest"].rotation_difference(head_q)
    angle = difference.angle
    if angle > math.radians(30):
        head_q = q["chest"].slerp(head_q, math.radians(30) / angle)
    q["head"] = head_q
    q["neck"] = q["chest"].slerp(head_q, 0.5)
    for side, sh, el, wr, pi, ind, hi, kn, an, to in (
        ("L", L_SH, L_EL, L_WR, L_PI, L_IN, L_HI, L_KN, L_AN, L_TO),
        ("R", R_SH, R_EL, R_WR, R_PI, R_IN, R_HI, R_KN, R_AN, R_TO),
    ):
        q[f"eye.{side}"] = head_q.copy()
        q[f"shoulder.{side}"] = q["chest"].copy()
        q[f"upper_arm.{side}"] = swing(q["chest"], f"upper_arm.{side}", P[el] - P[sh])
        q[f"forearm.{side}"] = swing(q[f"upper_arm.{side}"], f"forearm.{side}", P[wr] - P[el])
        q[f"hand.{side}"] = swing(q[f"forearm.{side}"], f"hand.{side}", (P[pi] + P[ind]) / 2 - P[wr])
        q[f"thigh.{side}"] = swing(q["hips"], f"thigh.{side}", P[kn] - P[hi])
        q[f"shin.{side}"] = swing(q[f"thigh.{side}"], f"shin.{side}", P[an] - P[kn])
        q[f"foot.{side}"] = swing(q[f"shin.{side}"], f"foot.{side}", P[to] - P[an])
    return q


def leg_points(q, side, translate):
    """Armature-space hip, knee, ankle and toe under world rotations `q`."""
    pivot = head["hips"]
    hip = translate + pivot + q["hips"] @ (head[f"thigh.{side}"] - pivot)
    knee = hip + q[f"thigh.{side}"] @ (tail[f"thigh.{side}"] - head[f"thigh.{side}"])
    ankle = knee + q[f"shin.{side}"] @ (tail[f"shin.{side}"] - head[f"shin.{side}"])
    toe = ankle + q[f"foot.{side}"] @ (tail[f"foot.{side}"] - head[f"foot.{side}"])
    return hip, knee, ankle, toe


# ---------------------------------------------------------------- solve every frame
solved = [solve_frame([vec(p) for p in frame]) for frame in world]
raw_rotations = [{name: value.copy() for name, value in q.items()} for q in solved]

# Sole height under the ankle and toe bones when the foot is flat at rest.
ANKLE_SOLE = head["foot.L"].z
TOE_SOLE = tail["foot.L"].z


def foot_low(q, side, translate=Vector()):
    _, _, ankle, toe = leg_points(q, side, translate)
    return min(ankle.z - ANKLE_SOLE, toe.z - TOE_SOLE), ankle


ankles = {"L": np.zeros((count, 3)), "R": np.zeros((count, 3))}
lows = {"L": np.zeros(count), "R": np.zeros(count)}
for i, q in enumerate(solved):
    for side in "LR":
        lows[side][i], ankle = foot_low(q, side)
        ankles[side][i] = tuple(ankle)

floor_under = np.minimum(lows["L"], lows["R"])
foot_velocity = {side: np.gradient(ankles[side][:, :2], axis=0) * fps for side in "LR"}
# The line the feet separate along. A swinging foot moves forward along it, fast and briefly. A
# planted foot slides backward, slowly and for longer. That difference says which way is forward.
separation = ankles["L"][:, :2] - ankles["R"][:, :2]
separation = separation - separation.mean(axis=0)
stride_axis = np.linalg.svd(separation, full_matrices=False)[2][0]
along = {side: foot_velocity[side] @ stride_axis for side in "LR"}
if float((along["L"] ** 3 + along["R"] ** 3).sum()) < 0:
    along = {side: -along[side] for side in "LR"}
# On each frame the foot moving most backward is the planted one, and it moves opposite to the body.
left_is_planted = along["L"] <= along["R"]
planted_velocity = np.where(left_is_planted[:, None], foot_velocity["L"], foot_velocity["R"])
body_velocity = -gaussian(planted_velocity, max(2.0, 0.2 * fps))
body_speed = np.linalg.norm(body_velocity, axis=1)

# Contacts: a foot near the floor and still relative to the ground, not relative to the hips.
contact = {}
ground_speed = {side: np.linalg.norm(foot_velocity[side] + body_velocity, axis=1) for side in "LR"}
speed_limit = np.maximum(CONTACT_SPEED, CONTACT_SPEED_SHARE * body_speed)
for side in "LR":
    contact[side] = (lows[side] - floor_under < CONTACT_HEIGHT) & (ground_speed[side] < speed_limit)
# Without a flight phase, something is always holding her up: the stiller foot, if it is low.
for i in range(count):
    if not contact["L"][i] and not contact["R"][i]:
        side = "L" if ground_speed["L"][i] <= ground_speed["R"][i] else "R"
        if lows[side][i] - floor_under[i] < CONTACT_HEIGHT:
            contact[side][i] = True
for side in "LR":
    # Bridge very short gaps, and drop very short contacts, to avoid flicker.
    for value, longest in ((False, 2), (True, 2)):
        run_start = None
        for i in range(count + 1):
            inside = i < count and contact[side][i] == value
            if inside and run_start is None:
                run_start = i
            if not inside and run_start is not None:
                if i - run_start <= longest and run_start > 0 and i < count:
                    contact[side][run_start:i] = not value
                run_start = None

# Height: stand on the planted foot's sole. With no foot planted, stand on the lower one.
ankle_low = {side: ankles[side][:, 2] - ANKLE_SOLE for side in "LR"}
lifts = -floor_under.copy()
for i in range(count):
    standing = [ankle_low[side][i] for side in "LR" if contact[side][i]]
    if standing:
        lifts[i] = -min(standing)
# The tracker rarely shows a knee fully straight, so following it frame by frame makes her crouch.
# Hips stay near the top of their range over about a second, unless the performer stays low too.
# Wide steps still bring them down, through the reach limit below, which is the natural bob of a walk.
window = max(int(fps / 2), 2)
lifts = np.array([np.percentile(lifts[max(i - window, 0):i + window + 1], HIP_RANGE_PERCENTILE) for i in range(count)])

# Travel: planted feet stay still, so the body moves opposite to how the foot moves under it.
root = np.zeros((count, 3))
velocity = np.zeros(2)
for i in range(1, count):
    moves = []
    for side in "LR":
        if contact[side][i] and contact[side][i - 1]:
            moves.append(ankles[side][i - 1, :2] - ankles[side][i, :2])
    if moves:
        velocity = np.mean(moves, axis=0)
    else:
        velocity = body_velocity[i] / fps  # no foot planted: carry on at the body's speed
    if np.linalg.norm(velocity) < 0.0007:
        velocity = np.zeros(2)     # ignore tracker jitter when she is standing still
    root[i, :2] = root[i - 1, :2] + velocity
root[:, 2] = gaussian(lifts[:, None], SMOOTH)[:, 0]
root[:, :2] = gaussian(root[:, :2], SMOOTH)

# Pin each planted foot at the average place it occupied during that contact.
THIGH = {s: (tail[f"thigh.{s}"] - head[f"thigh.{s}"]).length for s in "LR"}
SHIN = {s: (tail[f"shin.{s}"] - head[f"shin.{s}"]).length for s in "LR"}
pin = {"L": [None] * count, "R": [None] * count}
blend = {"L": np.zeros(count), "R": np.zeros(count)}
for side in "LR":
    i = 0
    while i < count:
        if not contact[side][i]:
            i += 1
            continue
        j = i
        while j + 1 < count and contact[side][j + 1]:
            j += 1
        spots = np.array([ankles[side][k] + root[k] for k in range(i, j + 1)])
        spot = spots.mean(axis=0)
        spot[2] = ANKLE_SOLE
        for k in range(i, j + 1):
            pin[side][k] = Vector(tuple(spot))
            blend[side][k] = 1.0
        i = j + 1
    blend[side] = gaussian(blend[side][:, None], 1.0)[:, 0]
    # Ease into a pin from the frames around it, using the nearest pinned spot.
    for k in range(count):
        if pin[side][k] is None and blend[side][k] > 0.01:
            candidates = [n for n in (k - 1, k + 1, k - 2, k + 2) if 0 <= n < count and pin[side][n] is not None]
            if candidates:
                pin[side][k] = pin[side][candidates[0]]
            else:
                blend[side][k] = 0.0

# Lower the hips where a planted foot would otherwise be out of the leg's reach. Her legs are
# shorter than the performer's, so a stride that suits the performer can overstretch hers.
hip_limit = np.full(count, np.inf)
for i, q in enumerate(solved):
    flat_root = Vector((root[i, 0], root[i, 1], 0.0))
    for side in "LR":
        if blend[side][i] > 0.5 and pin[side][i] is not None:
            hip = leg_points(q, side, flat_root)[0]
            to_pin = pin[side][i] - hip
            reach = (THIGH[side] + SHIN[side]) * 0.992
            across = math.hypot(to_pin.x, to_pin.y)
            if across < reach:
                hip_limit[i] = min(hip_limit[i], pin[side][i].z + math.sqrt(reach * reach - across * across) - hip.z)
lowered_frames = int((root[:, 2] > hip_limit).sum())
for _ in range(4):
    root[:, 2] = np.minimum(root[:, 2], hip_limit)
    root[:, 2] = gaussian(root[:, 2:3], 1.0)[:, 0]
root[:, 2] = np.minimum(root[:, 2], hip_limit)

# The shoes' own vertices, so the floor check uses the real shape rather than the bones.
shoes = {side: bpy.data.objects[f"shoe_{side}"] for side in "LR"}
shoe_points = {}
for side in "LR":
    count_v = len(shoes[side].data.vertices)
    coords = np.empty(count_v * 3)
    shoes[side].data.vertices.foreach_get("co", coords)
    points = coords.reshape(-1, 3)
    matrix = np.array(shoes[side].matrix_world)
    shoe_points[side] = points @ matrix[:3, :3].T + matrix[:3, 3] - np.array(tuple(head[f"foot.{side}"]))


def shoe_lowest(q, side, translate):
    """Height of the lowest shoe point under this pose."""
    _, _, ankle, _ = leg_points(q, side, translate)
    turned = shoe_points[side] @ np.array(q[f"foot.{side}"].to_matrix()).T
    return float(turned[:, 2].min()) + ankle.z


def solve_leg_ik(q, side, translate, target, weight, flatten=True):
    hip, knee, ankle, _ = leg_points(q, side, translate)
    goal = ankle.lerp(target, weight)
    a, b = THIGH[side], SHIN[side]
    to_goal = goal - hip
    distance = min(to_goal.length, (a + b) * 0.998)
    direction = to_goal.normalized()
    # Keep the knee bending the same way the tracked pose bends it.
    bend = (knee - hip) - direction * (knee - hip).dot(direction)
    if bend.length < 1e-5:
        bend = q["hips"] @ Vector((0, -1, 0))
        bend = bend - direction * bend.dot(direction)
    bend = bend.normalized()
    along = (a * a - b * b + distance * distance) / (2 * distance)
    height = math.sqrt(max(a * a - along * along, 0.0))
    new_knee = hip + direction * along + bend * height
    new_ankle = hip + direction * distance
    old_foot_world = q[f"foot.{side}"].copy()
    q[f"thigh.{side}"] = swing(q["hips"], f"thigh.{side}", new_knee - hip)
    q[f"shin.{side}"] = swing(q[f"thigh.{side}"], f"shin.{side}", new_ankle - new_knee)
    # A planted foot lies flat on the floor, keeping only its heading.
    heading = old_foot_world @ Vector((0, -1, 0))
    heading.z = 0
    if flatten and heading.length > 1e-4 and weight > 0.5:
        flat = Vector((0, -1, 0)).rotation_difference(heading.normalized())
        q[f"foot.{side}"] = old_foot_world.slerp(flat, weight)


def to_local(q):
    local = {}
    for name in q:
        parent_name = parent[name]
        parent_q = q[parent_name] if parent_name else Quaternion()
        in_rest_axes = (parent_q.inverted() @ q[name]).to_matrix()
        local[name] = (rest3[name].inverted() @ in_rest_axes @ rest3[name]).to_quaternion()
    return local


# ---------------------------------------------------------------- keyframes
scene = bpy.context.scene
scene.render.fps = max(int(round(fps)), 1)
scene.frame_start = 1
scene.frame_end = count
for pb in rig.pose.bones:
    pb.rotation_mode = "QUATERNION"

previous = {}
floor_raises = {"L": 0, "R": 0}
root_bone = rig.pose.bones["root"]
root_rest_inverse = rest3["root"].inverted()
for i, q in enumerate(solved):
    translate = Vector(tuple(root[i]))
    for side in "LR":
        if blend[side][i] > 0.01 and pin[side][i] is not None:
            solve_leg_ik(q, side, translate, pin[side][i], float(blend[side][i]))
        # No part of a shoe goes under the floor, planted or swinging.
        sunk = -shoe_lowest(q, side, translate)
        if sunk > 0.0:
            ankle_now = leg_points(q, side, translate)[2]
            solve_leg_ik(q, side, translate, ankle_now + Vector((0, 0, sunk)), 1.0, flatten=False)
            floor_raises[side] += 1
    local = to_local(q)
    for name, value in local.items():
        pb = rig.pose.bones[name]
        if name in previous and previous[name].dot(value) < 0:
            value = Quaternion((-value.w, -value.x, -value.y, -value.z))
        previous[name] = value
        pb.rotation_quaternion = value
        pb.keyframe_insert("rotation_quaternion", frame=i + 1)
    root_bone.location = root_rest_inverse @ translate
    root_bone.keyframe_insert("location", frame=i + 1)
rig.animation_data.action.name = "video_motion"

# ---------------------------------------------------------------- measure the result
depsgraph = bpy.context.evaluated_depsgraph_get()
world_dirs = {}
foot_track = {"L": [], "R": []}
leg_stretch = {"L": [], "R": []}
lowest = []
JOINTS = [
    "hips", "chest", "head", "upper_arm.L", "upper_arm.R", "forearm.L", "forearm.R",
    "thigh.L", "thigh.R", "shin.L", "shin.R", "foot.L", "foot.R",
]
for i in range(count):
    scene.frame_set(i + 1)
    depsgraph.update()
    low = 9.0
    for side in "LR":
        evaluated = shoes[side].evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        coords = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", coords)
        low = min(low, float(coords[2::3].min()))
        evaluated.to_mesh_clear()
        foot_track[side].append(tuple(rig.pose.bones[f"foot.{side}"].head))
        leg_stretch[side].append(
            (rig.pose.bones[f"foot.{side}"].head - rig.pose.bones[f"thigh.{side}"].head).length / (THIGH[side] + SHIN[side])
        )
    lowest.append(low)
    world_dirs[i] = {n: tuple((rig.pose.bones[n].tail - rig.pose.bones[n].head).normalized()) for n in JOINTS}

slips = []
for side in "LR":
    track = np.array(foot_track[side])
    i = 0
    while i < count:
        if blend[side][i] > 0.98:
            j = i
            while j + 1 < count and blend[side][j + 1] > 0.98:
                j += 1
            if j > i:
                slips.append(float(np.linalg.norm(track[i:j + 1, :2] - track[i:j + 1, :2].mean(axis=0), axis=1).max()))
            i = j + 1
        else:
            i += 1

report = {
    "frames": count,
    "fps": fps,
    "contact_share_L": round(float(contact["L"].mean()), 3),
    "contact_share_R": round(float(contact["R"].mean()), 3),
    "planted_foot_slip_mm": round(max(slips) * 1000, 2) if slips else None,
    "lowest_shoe_mm": round(min(lowest) * 1000, 1),
    "highest_lowest_shoe_mm": round(max(lowest) * 1000, 1),
    "travel_m": [round(float(v), 3) for v in (root[-1] - root[0])[:2]],
    "tracked_frames": int(good.sum()),
    "hip_lowered_frames": lowered_frames,
    "floor_raises": floor_raises,
    "hip_height_m": [round(float(head["hips"].z + root[:, 2].min()), 3), round(float(head["hips"].z + root[:, 2].max()), 3)],
    "rest_hip_height_m": round(float(head["hips"].z), 3),
}

# Where the feet landed: step and stride lengths, against her own leg length.
plants = []
for side in "LR":
    i = 0
    while i < count:
        if contact[side][i]:
            j = i
            while j + 1 < count and contact[side][j + 1]:
                j += 1
            spot = pin[side][i]
            plants.append((i, side, j - i + 1, spot.x, spot.y))
            i = j + 1
        else:
            i += 1
plants.sort()
leg_length = THIGH["L"] + SHIN["L"]
steps = [math.hypot(b[3] - a[3], b[4] - a[4]) for a, b in zip(plants, plants[1:]) if a[1] != b[1]]
same_side = {"L": [p for p in plants if p[1] == "L"], "R": [p for p in plants if p[1] == "R"]}
strides = [
    (b[0] - a[0], math.hypot(b[3] - a[3], b[4] - a[4]))
    for rows in same_side.values() for a, b in zip(rows, rows[1:])
]
report["plants"] = [{"start": p[0], "side": p[1], "frames": p[2]} for p in plants]
report["step_lengths_m"] = [round(v, 3) for v in steps]
report["stride_frames"] = [v[0] for v in strides]
report["stride_lengths_m"] = [round(v[1], 3) for v in strides]
report["leg_length_m"] = round(leg_length, 3)
report["median_step_over_leg"] = round(float(np.median(steps)) / leg_length, 2) if steps else None
report["speed_m_per_s"] = round(float(np.linalg.norm(root[-1, :2] - root[0, :2])) * fps / max(count - 1, 1), 2)
# How straight a planted leg is: hip-to-ankle distance as a share of the leg's full length.
planted_stretch = np.concatenate([np.array(leg_stretch[s])[contact[s]] for s in "LR"])
report["planted_leg_stretch"] = [round(float(np.percentile(planted_stretch, p)), 3) for p in (10, 50, 90)]
report["contact_L"] = [int(v) for v in contact["L"]]
report["contact_R"] = [int(v) for v in contact["R"]]

if TRUTH:
    # Heading is free, because the camera's angle is unknown. Find it, then compare bones.
    with open(TRUTH, encoding="utf-8") as handle:
        truth = json.load(handle)
    n = len(truth["frames"])

    def yaw(v):
        return math.atan2(v[1], v[0])

    offsets = []
    for i in range(count):
        ref = truth["frames"][i % n]["thigh_axis"]
        q = solved[i]["hips"] @ Vector((1, 0, 0))
        offsets.append(math.atan2(q.y, q.x) - math.atan2(ref[1], ref[0]))
    # Try every heading and keep the one where the bones agree best overall. The real camera
    # angle can't be used, because a real video will not tell us what it was.
    first_guess = float(np.median(np.unwrap(offsets)))

    def heading_error(angle):
        spin = Quaternion((0, 0, 1), angle)
        total = 0.0
        for i in range(0, count, 2):
            frame_truth = truth["frames"][i % n]["dirs"]
            for name in JOINTS:
                total += Vector(world_dirs[i][name]).angle(spin @ Vector(frame_truth[name]))
        return total

    candidates = [first_guess + math.radians(d) for d in range(-40, 41, 2)]
    offset = min(candidates, key=heading_error)
    report["found_camera_yaw_deg"] = round(math.degrees(offset), 1)
    rotate = Quaternion((0, 0, 1), offset)
    errors = {n_: [] for n_ in JOINTS}
    for i in range(count):
        frame_truth = truth["frames"][i % n]["dirs"]
        for name in JOINTS:
            model = Vector(world_dirs[i][name])
            expected = rotate @ Vector(frame_truth[name])
            errors[name].append(math.degrees(model.angle(expected)))
    report["angle_error_deg"] = {k: round(float(np.mean(v)), 1) for k, v in errors.items()}
    report["angle_error_mean_deg"] = round(float(np.mean([np.mean(v) for v in errors.values()])), 1)
    tracker_only = {n_: [] for n_ in JOINTS}
    for i in range(count):
        frame_truth = truth["frames"][i % n]["dirs"]
        for name in JOINTS:
            model = raw_rotations[i][name] @ rest_dir[name]
            expected = rotate @ Vector(frame_truth[name])
            tracker_only[name].append(math.degrees(model.angle(expected)))
    report["before_ik_error_deg"] = {k: round(float(np.mean(v)), 1) for k, v in tracker_only.items()}
    report["before_ik_error_mean_deg"] = round(float(np.mean([np.mean(v) for v in tracker_only.values()])), 1)

os.makedirs(os.path.dirname(BLEND_OUT), exist_ok=True)
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=BLEND_OUT, compress=False)
bpy.ops.export_scene.gltf(
    filepath=GLB_OUT, export_format="GLB", export_animations=True, export_skins=True,
    export_apply=False, export_yup=True,
)
report["glb_mb"] = round(os.path.getsize(GLB_OUT) / 1024 / 1024, 1)
with open(REPORT, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
print("RETARGET_REPORT", json.dumps({k: v for k, v in report.items() if k not in ("contact_L", "contact_R", "plants")}))
print("RETARGET_OK")
