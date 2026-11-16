"""Measure a facial expression from a pair of reference images and build it as a shape key.

Usage: blender -b character.blend -P expression_from_image.py -- <settings.json> <out.blend>

settings.json:
  {"name": "surprised", "fit": subject_fit.json, "placement": head_placement.json,
   "neutral": marks.json key, "expression": marks.json key, "marks": marks.json,
   "crop": [x0, y0, scale], "camera": {"focal_px", "cx", "cy"}, "eye_open_deg": optional}

Both images show the character from about the same side. For each image, the head's position is
solved from points that an expression hardly moves (nose, outer face, eye corners). Each tracked
point is then lifted back onto the face: the place along its line of sight nearest where it sits on
the neutral face. The difference between the two images is the expression's movement, which is:
  1. a jaw opening, sized so the lips part as much as they do in the image (uses "jaw_open"),
  2. plus everything else (brows, cheeks, lip shape), spread smoothly over the face, separately
     above and below the lips so the two lips can move apart,
  3. plus wider or narrower eyes, by rolling the upper lids (the tracker's eyelid points aren't
     reliable on large cartoon eyes, so those come from eye_open_deg instead).
The back of the head and the neck are held still. Also adds blink_L and blink_R keys.
"""
import json
import math
import sys

import bpy
import numpy as np

cfg_path, out = sys.argv[sys.argv.index("--") + 1:][:2]
with open(cfg_path, encoding="utf-8") as f:
    cfg = json.load(f)
fit = json.load(open(cfg["fit"], encoding="utf-8"))
place = json.load(open(cfg["placement"], encoding="utf-8"))
marks = json.load(open(cfg["marks"], encoding="utf-8"))
ids = list(fit["ids"])
index = {i: k for k, i in enumerate(ids)}
P = np.array(fit["points"]) * place["scale"] + np.array((0, place["translation"][1], place["translation"][2]))
x0, y0, crop_scale = cfg["crop"]


def image_points(key):
    pts = np.array(marks[key]["points"])[:, :2] / crop_scale + np.array((x0, y0))
    return pts[ids]


cam = cfg["camera"]
K = np.array([[cam["focal_px"], 0, cam["cx"]], [0, cam["focal_px"], cam["cy"]], [0, 0, 1.0]])
RIGID = [1, 4, 5, 6, 168, 195, 197, 45, 275, 234, 454, 127, 356, 93, 323, 9, 8, 33, 263, 133, 362, 132, 361]
# (Of these, only the ones the tracker places reliably on this face are used; see below.)
reliable = np.array(fit["reliability"]) > 0.35  # points the tracker places consistently on this face
rigid = [index[i] for i in RIGID if i in index and reliable[index[i]]]
EYES = set(range(468, 478)) | {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                               263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466}


def rodrigues(r):
    angle = np.linalg.norm(r)
    if angle < 1e-12:
        return np.eye(3)
    k = r / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def project(params, pts):
    R, t = rodrigues(params[:3]), params[3:]
    c = pts @ R.T + t
    return (c[:, :2] / c[:, 2:]) * K[0, 0] + K[:2, 2]


def solve_pose(pts, uv):
    """Head pose seen by the camera, by Gauss-Newton from the comparison camera's own view."""
    a = math.radians(cam.get("azimuth", 38))
    location = np.array((-math.sin(a) * 4.7, -math.cos(a) * 4.7, 1.02))
    forward = np.array((0, 0, 0.77)) - location
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0, 0, 1.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R0 = np.stack([right, down, forward])
    angle = math.acos(np.clip((np.trace(R0) - 1) / 2, -1, 1))
    axis = np.array((R0[2, 1] - R0[1, 2], R0[0, 2] - R0[2, 0], R0[1, 0] - R0[0, 1])) / (2 * math.sin(angle))
    params = np.concatenate([axis * angle, -R0 @ location])
    for _ in range(30):
        res = (project(params, pts) - uv).ravel()
        J = np.zeros((len(res), 6))
        for j in range(6):
            d = np.zeros(6)
            d[j] = 1e-6
            J[:, j] = ((project(params + d, pts) - uv).ravel() - res) / 1e-6
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        params += step
        if np.linalg.norm(step) < 1e-10:
            break
    return params


def lift(key):
    uv = image_points(key)
    params = solve_pose(P[rigid], uv[rigid])
    err = float(np.median(np.linalg.norm(project(params, P[rigid]) - uv[rigid], axis=1)))
    R, t = rodrigues(params[:3]), params[3:]
    origin = -R.T @ t
    rays = (R.T @ (np.linalg.inv(K) @ np.hstack([uv, np.ones((len(uv), 1))]).T)).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    tt = ((P - origin) * rays).sum(axis=1)
    print("POSE", key, "rigid reprojection px", round(err, 2))
    return origin + rays * tt[:, None]


q_neutral, q_expr = lift(cfg["neutral"]), lift(cfg["expression"])
move = q_expr - q_neutral
for name, i in (("nose tip", 1), ("brow L", 334), ("brow R", 105), ("upper lip", 13), ("lower lip", 14), ("mouth corner L", 291),
                ("mouth corner R", 61), ("chin", 152), ("upper lid L", 386), ("cheek L", 425)):
    if i in index:
        print("MOVE", f"{name:15s}", np.round(move[index[i]] * 1000, 1), "reliable", bool(reliable[index[i]]))
move[[index[i] for i in EYES if i in index]] = np.nan
print("MOVEMENT_MM median", round(float(np.nanmedian(np.linalg.norm(move, axis=1))) * 1000, 1),
      "max", round(float(np.nanmax(np.linalg.norm(move, axis=1))) * 1000, 1))

head = bpy.data.objects["head"]
mesh = head.data
co = np.array([v.co[:] for v in mesh.vertices])
keys = mesh.shape_keys.key_blocks
jaw = np.array([d.co[:] for d in keys["jaw_open"].data]) - co  # movement at jaw_open = 1
jaw_amount = np.linalg.norm(jaw, axis=1)
jaw_weight = np.clip(jaw_amount / max(jaw_amount.max(), 1e-9), 0, 1)

# Each tracked point takes the jaw weight of the face under it.
nearest = np.array([int(np.argmin(np.linalg.norm(co - p, axis=1))) for p in P])
point_jaw = jaw[nearest]
point_w = jaw_weight[nearest]

# 1. Jaw: as much as makes the inner lips part like the image (lower lip 14 against upper lip 13).
up, lo = index[13], index[14]
want = (move[lo] - move[up])
jaw_value = float(np.dot(want, point_jaw[lo]) / max(np.dot(point_jaw[lo], point_jaw[lo]), 1e-12))
jaw_value = float(np.clip(jaw_value, 0.0, 1.5))
print("JAW_OPEN", round(jaw_value, 3), "lip parting in image mm", round(float(np.linalg.norm(want)) * 1000, 1))
residual = move - point_jaw * jaw_value


def spline(centres, values, smooth=0.003):
    n = len(centres)
    dist = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    poly = np.hstack([np.ones((n, 1)), centres])
    system = np.zeros((n + 4, n + 4))
    system[:n, :n] = -dist + np.eye(n) * smooth
    system[:n, n:] = poly
    system[n:, :n] = poly.T
    weights = np.linalg.solve(system, np.vstack([values, np.zeros((4, 3))]))

    def evaluate(points):
        out = np.empty((len(points), 3))
        for s in range(0, len(points), 3000):
            chunk = points[s:s + 3000]
            d = np.linalg.norm(chunk[:, None] - centres[None], axis=2)
            out[s:s + 3000] = -d @ weights[:n] + np.hstack([np.ones((len(chunk), 1)), chunk]) @ weights[n:]
        return out
    return evaluate


# Still points: the back of the head and the neck, which an expression doesn't move.
face_front = P[:, 1].min()
still = co[(co[:, 1] > face_front + 0.17) | (co[:, 2] < P[:, 2].min() - 0.06)]
still = still[np.linspace(0, len(still) - 1, min(150, len(still))).astype(int)]
good = ~np.isnan(residual).any(axis=1) & reliable
upper_pts = good & (point_w < 0.5)
lower_pts = good & ((point_w >= 0.5) | (np.linalg.norm(P - P[up], axis=1) > 0.05))
fields = {}
for name, use in (("upper", upper_pts), ("lower", lower_pts)):
    centres = np.vstack([P[use], still])
    values = np.vstack([residual[use], np.zeros((len(still), 3))])
    fields[name] = spline(centres, values)(co)
face_move = fields["upper"] * (1 - jaw_weight[:, None]) + fields["lower"] * jaw_weight[:, None]

# 3. Eyelids: roll the upper lid about each eye's centre (positive opens), and blinks.
eyes = {s: bpy.data.objects[f"eye.{s}"] for s in "LR"}


def lid_roll(points, side, degrees, lower=False):
    c = np.array(eyes[side].matrix_world.translation)
    radius = eyes[side].dimensions.x / 2
    rel = points - c
    dist = np.linalg.norm(rel, axis=1)
    near = np.clip((radius + 0.02 - dist) / 0.014, 0, 1)
    front = np.clip((-rel[:, 1] + 0.2 * radius) / (0.4 * radius), 0, 1)
    up_side = np.clip((rel[:, 2] if not lower else -rel[:, 2]) / 0.004, 0, 1)
    w = near * front * up_side
    w = w * w * (3 - 2 * w)
    a = np.radians(degrees) * w * (1 if not lower else -1)
    y = rel[:, 1] * np.cos(a) - rel[:, 2] * np.sin(a)
    z = rel[:, 1] * np.sin(a) + rel[:, 2] * np.cos(a)
    return np.stack([rel[:, 0], y, z], axis=1) + c


def lid_gap_degrees(side):
    """Angle between the upper and lower lid edges, seen from the eye's centre."""
    c = np.array(eyes[side].matrix_world.translation)
    radius = eyes[side].dimensions.x / 2
    rel = co - c
    on_eye = (np.abs(np.linalg.norm(rel, axis=1) - radius) < 0.004) & (-rel[:, 1] > 0.5 * radius) & (np.abs(rel[:, 0]) < 0.3 * radius)
    angle = np.degrees(np.arctan2(rel[on_eye, 2], -rel[on_eye, 1]))
    return float(angle[angle > 0].min() - angle[angle < 0].max()) if (angle > 0).any() and (angle < 0).any() else 30.0


eye_open = float(cfg.get("eye_open_deg", 0.0))
expression = co + jaw * jaw_value + face_move
for side in "LR":
    if eye_open:
        expression = lid_roll(expression, side, -eye_open)
key = head.shape_key_add(name=cfg["name"])
key.data.foreach_set("co", expression.ravel())
for side in "LR":
    gap = lid_gap_degrees(side)
    blink = lid_roll(co, side, gap * 0.8)
    blink = lid_roll(blink, side, gap * 0.25, lower=True)
    k = head.shape_key_add(name=f"blink_{side}")
    k.data.foreach_set("co", blink.ravel())
    print("BLINK", side, "lid gap deg", round(gap, 1))
# The mouth parts follow the jaw.
for name in ("tongue", "teeth_upper"):
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.data.shape_keys is not None:
        base = np.array([v.co[:] for v in obj.data.vertices])
        jaw_obj = np.array([d.co[:] for d in obj.data.shape_keys.key_blocks["jaw_open"].data]) - base
        k = obj.shape_key_add(name=cfg["name"])
        k.data.foreach_set("co", (base + jaw_obj * jaw_value).ravel())
moved = np.linalg.norm(expression - co, axis=1) * 1000
print("EXPRESSION", cfg["name"], "vertex movement mm median", round(float(np.median(moved)), 1), "max", round(float(moved.max()), 1))
# New shapes must not be left switched on: save the character with a neutral face.
for _obj in bpy.data.objects:
    _keys = getattr(_obj.data, "shape_keys", None) if _obj.data is not None else None
    if _keys is not None:
        for _block in _keys.key_blocks[1:]:
            _block.value = 0.0
bpy.ops.wm.save_as_mainfile(filepath=out)
print("EXPRESSION_OK")
