"""Build a small face rig of expression shapes, then mix them to match a reference expression.

Usage: blender -b character.blend -P face_rig.py -- <settings.json> <out.blend>

Settings: as for expression_from_image.py ("fit", "placement", "marks", "neutral", "expression",
"crop", "camera", "name"), plus "eyes_wide": how much of the eyes_wide shape the expression has.

Shapes, each made the way an animator would sculpt it, by moving a few face points and letting
the skin follow smoothly while the rest of the head stays still:
  jaw_open   (from open_mouth.py)    mouth_O  lips rounded and pushed forward
  brows_up   brows and lower forehead raised     smile  mouth corners up and back (negative: less)
  eyes_wide  upper lids lifted, lower lids lowered     blink_L, blink_R  lids closed
How much of each the expression has is then fitted: the head's pose and the amounts are adjusted
together until the face's tracked points land where they do in the expression image. Each point
is first corrected by how far off it already is in the neutral image, so the tracker's own habits
on this face cancel out. The mix is saved as one more shape, named after the expression.
"""
import json
import math
import sys

import bpy
import numpy as np

cfg_path, out = sys.argv[sys.argv.index("--") + 1:][:2]
cfg = json.load(open(cfg_path, encoding="utf-8"))
fit = json.load(open(cfg["fit"], encoding="utf-8"))
place = json.load(open(cfg["placement"], encoding="utf-8"))
marks = json.load(open(cfg["marks"], encoding="utf-8"))
ids = list(fit["ids"])
index = {i: k for k, i in enumerate(ids)}
P = np.array(fit["points"]) * place["scale"] + np.array((0, place["translation"][1], place["translation"][2]))
x0, y0, crop_scale = cfg["crop"]
cam = cfg["camera"]
K = np.array([[cam["focal_px"], 0, cam["cx"]], [0, cam["focal_px"], cam["cy"]], [0, 0, 1.0]])

head = bpy.data.objects["head"]
mesh = head.data
co = np.array([v.co[:] for v in mesh.vertices])
keys = mesh.shape_keys.key_blocks
mouth_width = float(np.linalg.norm(P[index[61]] - P[index[291]]))
s = mouth_width / 0.1  # sizes below are for a mouth 10 cm wide (this cartoon head's)


def L(i):
    return P[index[i]]


def spline(centres, values, smooth=0.002):
    n = len(centres)
    dist = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    poly = np.hstack([np.ones((n, 1)), centres])
    system = np.zeros((n + 4, n + 4))
    system[:n, :n] = -dist + np.eye(n) * smooth
    system[:n, n:] = poly
    system[n:, :n] = poly.T
    weights = np.linalg.solve(system, np.vstack([values, np.zeros((4, 3))]))
    d = np.linalg.norm(co[:, None] - centres[None], axis=2)
    return -d @ weights[:n] + np.hstack([np.ones((len(co), 1)), co]) @ weights[n:]


still = co[(co[:, 1] > P[:, 1].min() + 0.17) | (co[:, 2] < P[:, 2].min() - 0.06)]
still = still[np.linspace(0, len(still) - 1, min(120, len(still))).astype(int)]


def shape(moves, holds):
    """moves: {landmark: displacement}; holds: landmarks kept still. Returns vertex displacements."""
    centres = [L(i) for i in moves] + [L(i) for i in holds if i in index] + list(still)
    values = [np.array(v) for v in moves.values()] + [np.zeros(3)] * (len(centres) - len(moves))
    return spline(np.array(centres), np.array(values))


EYES_HOLD = [33, 133, 263, 362, 159, 145, 386, 374]
NOSE_HOLD = [1, 2, 4, 6, 98, 327, 168]
mid_x = (L(61)[0] + L(291)[0]) / 2
shapes = {}
shapes["mouth_O"] = shape({
    61: ((mid_x - L(61)[0]) * 0.32, -0.006 * s, 0), 291: ((mid_x - L(291)[0]) * 0.32, -0.006 * s, 0),
    0: (0, -0.008 * s, 0.002 * s), 17: (0, -0.008 * s, -0.002 * s), 13: (0, -0.007 * s, 0), 14: (0, -0.007 * s, 0),
}, EYES_HOLD + NOSE_HOLD + [205, 425, 50, 280, 152, 10, 151])
shapes["smile"] = shape({
    61: (-0.004 * s, 0.004 * s, 0.006 * s), 291: (0.004 * s, 0.004 * s, 0.006 * s),
    205: (0, 0, 0.003 * s), 425: (0, 0, 0.003 * s),
}, EYES_HOLD + NOSE_HOLD + [0, 17, 13, 14, 152, 10, 151])
BROWS = [70, 63, 105, 66, 107, 300, 293, 334, 296, 336]
brow_moves = {i: (0, 0.002 * s, 0.014 * s) for i in BROWS if i in index}
brow_moves.update({i: (0, 0, 0.005 * s) for i in (151, 9, 108, 337) if i in index})
shapes["brows_up"] = shape(brow_moves, [33, 133, 263, 362] + NOSE_HOLD + [10, 205, 425, 61, 291])

eyes = {side: bpy.data.objects[f"eye.{side}"] for side in "LR"}


def lid_roll(points, side, degrees, lower=False):
    c = np.array(eyes[side].matrix_world.translation)
    radius = eyes[side].dimensions.x / 2
    rel = points - c
    dist = np.linalg.norm(rel, axis=1)
    near = np.clip((radius + 0.02 - dist) / 0.014, 0, 1)
    front = np.clip((-rel[:, 1] + 0.2 * radius) / (0.4 * radius), 0, 1)
    side_w = np.clip((rel[:, 2] if not lower else -rel[:, 2]) / 0.004, 0, 1)
    w = near * front * side_w
    w = w * w * (3 - 2 * w)
    a = np.radians(degrees) * w * (1 if not lower else -1)
    y = rel[:, 1] * np.cos(a) - rel[:, 2] * np.sin(a)
    z = rel[:, 1] * np.sin(a) + rel[:, 2] * np.cos(a)
    return np.stack([rel[:, 0], y, z], axis=1) + c


def lid_gap(side):
    c = np.array(eyes[side].matrix_world.translation)
    radius = eyes[side].dimensions.x / 2
    rel = co - c
    on = (np.abs(np.linalg.norm(rel, axis=1) - radius) < 0.004) & (-rel[:, 1] > 0.5 * radius) & (np.abs(rel[:, 0]) < 0.3 * radius)
    a = np.degrees(np.arctan2(rel[on, 2], -rel[on, 1]))
    return float(a[a > 0].min() - a[a < 0].max()) if (a > 0).any() and (a < 0).any() else 30.0


wide = co.copy()
for side in "LR":
    wide = lid_roll(wide, side, -18)
    wide = lid_roll(wide, side, -6, lower=True)
shapes["eyes_wide"] = wide - co
for side in "LR":
    gap = lid_gap(side)
    closed = lid_roll(lid_roll(co, side, gap * 0.8), side, gap * 0.25, lower=True)
    shapes[f"blink_{side}"] = closed - co
    print("BLINK", side, "lid gap deg", round(gap, 1))
shapes["jaw_open"] = np.array([d.co[:] for d in keys["jaw_open"].data]) - co
for name, delta in shapes.items():
    if name == "jaw_open":
        continue
    k = keys.get(name) or head.shape_key_add(name=name)
    k.data.foreach_set("co", (co + delta).ravel())
    k.slider_min = -1.0 if name == "smile" else 0.0
    print("SHAPE", name, "largest move mm", round(float(np.linalg.norm(delta, axis=1).max()) * 1000, 1))

# ---------------------------------------------------------------- fit the mix to the expression image
nearest = np.array([int(np.argmin(np.linalg.norm(co - p, axis=1))) for p in P])
EYES = set(range(468, 478)) | {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                               263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466}
use = np.array([i not in EYES for i in ids])
FIT = ["jaw_open", "mouth_O", "brows_up", "smile"]
LIMITS = {"jaw_open": (0, 1.3), "mouth_O": (0, 1.5), "brows_up": (0, 1.5), "smile": (-1, 1)}
D = {n: shapes[n][nearest] for n in FIT + ["eyes_wide"]}
EYES_WIDE = float(cfg.get("eyes_wide", 1.0))


def rodrigues(r):
    angle = np.linalg.norm(r)
    if angle < 1e-12:
        return np.eye(3)
    k = r / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def project(pose, pts):
    c = pts @ rodrigues(pose[:3]).T + pose[3:]
    return c[:, :2] / c[:, 2:] * K[0, 0] + K[:2, 2]


def image_points(key):
    return (np.array(marks[key]["points"])[:, :2] / crop_scale + np.array((x0, y0)))[ids]


def start_pose():
    a = math.radians(cam.get("azimuth", 38))
    loc = np.array((-math.sin(a) * 4.7, -math.cos(a) * 4.7, 1.02))
    fwd = np.array((0, 0, 0.77)) - loc
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, (0, 0, 1.0))
    right /= np.linalg.norm(right)
    R0 = np.stack([right, np.cross(fwd, right), fwd])
    ang = math.acos(np.clip((np.trace(R0) - 1) / 2, -1, 1))
    axis = np.array((R0[2, 1] - R0[1, 2], R0[0, 2] - R0[2, 0], R0[1, 0] - R0[0, 1])) / (2 * math.sin(ang))
    return np.concatenate([axis * ang, -R0 @ loc])


def solve(target, free_weights, weights0, points_used):
    params = np.concatenate([start_pose(), weights0])

    def residual(p):
        pts = P + sum(p[6 + j] * D[n] for j, n in enumerate(FIT)) + EYES_WIDE * D["eyes_wide"] * free_weights
        return (project(p[:6], pts[points_used]) - target[points_used]).ravel()

    for _ in range(40):
        r = residual(params)
        J = np.zeros((len(r), len(params)))
        for j in range(len(params)):
            if j >= 6 and not free_weights:
                continue
            d = np.zeros(len(params))
            d[j] = 1e-6
            J[:, j] = (residual(params + d) - r) / 1e-6
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        params += step
        for j, n in enumerate(FIT):
            params[6 + j] = np.clip(params[6 + j], *LIMITS[n])
        if np.linalg.norm(step) < 1e-9:
            break
    return params, float(np.median(np.abs(residual(params)).reshape(-1, 2).sum(axis=1)))


neutral = image_points(cfg["neutral"])
expr = image_points(cfg["expression"])
pose_n, err_n = solve(neutral, False, np.zeros(len(FIT)), use)
bias = neutral - project(pose_n[:6], P)            # how far off the tracker already is on the neutral face
corrected = expr - bias
params, err = solve(corrected, True, np.full(len(FIT), 0.3), use)
weights = {n: round(float(params[6 + j]), 3) for j, n in enumerate(FIT)}
print("NEUTRAL_FIT_PX", round(err_n, 2), "EXPRESSION_FIT_PX", round(err, 2))
print("MIX", cfg["name"], weights, "eyes_wide", EYES_WIDE)

total = sum(weights[n] * shapes[n] for n in FIT) + EYES_WIDE * shapes["eyes_wide"]
k = keys.get(cfg["name"]) or head.shape_key_add(name=cfg["name"])
k.data.foreach_set("co", (co + total).ravel())
for name in ("tongue", "teeth_upper"):
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.data.shape_keys is not None:
        base = np.array([v.co[:] for v in obj.data.vertices])
        jaw_obj = np.array([d.co[:] for d in obj.data.shape_keys.key_blocks["jaw_open"].data]) - base
        ko = obj.data.shape_keys.key_blocks.get(cfg["name"]) or obj.shape_key_add(name=cfg["name"])
        ko.data.foreach_set("co", (base + jaw_obj * weights["jaw_open"]).ravel())
head["expression_mix_" + cfg["name"]] = json.dumps(weights)
# New shapes must not be left switched on: save the character with a neutral face.
for _obj in bpy.data.objects:
    _keys = getattr(_obj.data, "shape_keys", None) if _obj.data is not None else None
    if _keys is not None:
        for _block in _keys.key_blocks[1:]:
            _block.value = 0.0
bpy.ops.wm.save_as_mainfile(filepath=out)
print("FACE_RIG_OK")
