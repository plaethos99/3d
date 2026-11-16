"""Fit how open the eyes are, by matching what's visible of each eye to the reference images.

Usage: blender -b head.blend -P fit_eyelids.py -- <cameras.json> <views.json> <out.blend> <report.json>

In each reference the face tracker's 16 contour points around an eye outline the visible eye.
The model is rendered from the same camera with the head black and the eyeballs white, so the
white area is the model's visible eye. Upper and lower lids are rolled around the eyeball, as real
lids move when blinking, and the eyeballs moved deeper or shallower, until the two outlines
overlap best. Lids keep hugging the eye because they turn about its centre.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
cams_path, views_path, out_blend, report_path = args[:4]
with open(cams_path, encoding="utf-8") as f:
    cams = json.load(f)
with open(views_path, encoding="utf-8") as f:
    records = json.load(f)

# MediaPipe's contour around each eye, in order. "R" is the subject's right eye.
CONTOUR = {
    "R": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "L": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
}
SCALE = 1.0
SKIP_VIEWS = set(filter(None, os.environ.get("SKIP_VIEWS", "").split(",")))
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "FLAT"
scene.display.shading.color_type = "OBJECT"
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new("black")
scene.world.color = (0, 0, 0)
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.view_settings.view_transform = "Standard"
for obj in bpy.data.objects:
    if obj.type == "MESH":
        obj.color = (1, 1, 1, 1) if obj.name.startswith("eye.") else (0, 0, 0, 1)
    if obj.type == "LIGHT":
        obj.hide_render = True
data = bpy.data.cameras.new("lidcam")
data.sensor_fit = "HORIZONTAL"
data.sensor_width = 36.0
cam = bpy.data.objects.new("lidcam", data)
scene.collection.objects.link(cam)
scene.camera = cam


def polygon_mask(points, w, h):
    """Pixels inside a closed polygon (even-odd rule)."""
    mask = np.zeros((h, w), bool)
    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    inside = np.zeros(xs.shape, bool)
    a, b = points, np.roll(points, -1, axis=0)
    for (ax, ay), (bx, by) in zip(a, b):
        crosses = ((ay > ys) != (by > ys)) & (xs < (bx - ax) * (ys - ay) / ((by - ay) + 1e-9) + ax)
        inside ^= crosses
    mask[y0:y1 + 1, x0:x1 + 1] = inside
    return mask


# Optional: eye openings cut straight from the images (eye_opening_masks.py), used instead of the
# tracker's eyelid outline, which is too small on cartoon eyes. Views or eyes without one are left out.
MASK_DIR = os.environ.get("EYE_OPENING_MASKS")
openings = {}
if MASK_DIR:
    with open(os.path.join(MASK_DIR, "openings.json"), encoding="utf-8") as f:
        openings = json.load(f)
IRIS_CENTRE = {"R": 468, "L": 473}
targets = {}
for key, view in cams.items():
    record = next(r for r in records if r["video"] == view["video"] and r["frame"] == view["frame"])
    w, h = int(view["size"][0] * SCALE), int(view["size"][1] * SCALE)
    pts = np.array(record["points"])[:, :2] * SCALE
    if MASK_DIR:
        image = bpy.data.images.load(os.path.join(MASK_DIR, f"eyes_{view['video']}.png"))
        px = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
        image.pixels.foreach_get(px)
        found = px.reshape(image.size[1], image.size[0], 4)[::-1, :, 0] > 0.5
        bpy.data.images.remove(image)
        yy, xx = np.mgrid[0:h, 0:w]
        nearer_r = np.hypot(xx - pts[468, 0], yy - pts[468, 1]) < np.hypot(xx - pts[473, 0], yy - pts[473, 1])
        targets[key] = {}
        for side, mine in (("R", nearer_r), ("L", ~nearer_r)):
            usable = isinstance(openings.get(view["video"], {}).get(side), dict) and view["video"] not in SKIP_VIEWS
            targets[key][side] = (found & mine) if usable else None
    else:
        targets[key] = {side: polygon_mask(pts[ids], w, h) for side, ids in CONTOUR.items()}
    targets[key]["box"] = {side: (pts[ids].min(axis=0), pts[ids].max(axis=0)) for side, ids in CONTOUR.items()}
first = next(iter(cams.values()))
scene.render.resolution_x = int(first["size"][0] * SCALE)
scene.render.resolution_y = int(first["size"][1] * SCALE)
scene.render.resolution_percentage = 100

head = bpy.data.objects["head"]
mesh = head.data
rest = np.array([v.co for v in mesh.vertices])
eyes = {side: bpy.data.objects[f"eye.{side}"] for side in "LR"}
eye_local = {side: np.array([v.co for v in eyes[side].data.vertices]) for side in "LR"}
eye_origin = {side: np.array(eyes[side].matrix_world.translation) for side in "LR"}
centre = {side: eye_origin[side] + eye_local[side].mean(axis=0) for side in "LR"}
radius = {side: float(np.ptp(eye_local[side][:, 0]) / 2) for side in "LR"}

# Lid region: skin close around each eyeball, at its front, fading out with distance.
lid_weight = {}
for side in "LR":
    rel = rest - centre[side]
    dist = np.linalg.norm(rel, axis=1)
    near = np.clip((radius[side] + 0.016 - dist) / 0.012, 0, 1)
    front = np.clip((-rel[:, 1] + 0.2 * radius[side]) / (0.4 * radius[side]), 0, 1)
    w = near * front
    lid_weight[side] = w * w * (3 - 2 * w)
print("LID_VERTICES", {s: int((w > 0.5).sum()) for s, w in lid_weight.items()})


def shaped(p):
    coords = rest.copy()
    for side in "LR":
        rel = rest - centre[side]
        up = np.tanh(rel[:, 2] / 0.004)  # +1 above the eye's middle, -1 below
        angle = np.radians(np.where(up > 0, p["upper_close_deg"] * up, -p["lower_close_deg"] * up)) * lid_weight[side]
        # The upper lid's edge alone, the skin right against the eye: lifting it opens the eye
        # without moving the whole lid and brow area.
        edge = np.clip(1.0 - (np.linalg.norm(rel, axis=1) - radius[side]) / 0.005, 0, 1) * np.clip(up, 0, 1)
        angle = angle - np.radians(p["upper_edge_open_deg"]) * edge * lid_weight[side]
        # Arches: open the middle of each lid more than its corners, for a rounder opening.
        middle = np.clip(1.0 - np.abs(rel[:, 0]) / (0.6 * radius[side]), 0, 1)
        middle = middle * middle * (3 - 2 * middle)
        arch = np.where(up > 0, -p["upper_arch_deg"] * up, -p["lower_arch_deg"] * up)
        angle = angle + np.radians(arch) * middle * lid_weight[side]
        # Roll about the eye's left-right axis, through its (moved) centre.
        y, z = rel[:, 1], rel[:, 2]
        new_y = y * np.cos(angle) - z * np.sin(angle)
        new_z = y * np.sin(angle) + z * np.cos(angle)
        moved = np.stack([rel[:, 0], new_y, new_z], axis=1) + centre[side]
        w = (lid_weight[side] > 0)[:, None]
        coords = np.where(w, moved, coords)
    return coords


def place_eyes(p):
    for side in "LR":
        eyes[side].location = Vector(eye_origin[side] + np.array((0, p["eye_back_m"], 0)))


def mismatch(p, save=False):
    mesh.vertices.foreach_set("co", shaped(p).ravel())
    mesh.update()
    place_eyes(p)
    total = 0.0
    counted = 0
    per_view = {}
    for key, view in cams.items():
        cam.data.lens = view["lens_mm"]
        cam.matrix_world = Matrix.Translation(Vector(view["centre"])) @ Matrix(view["rotation"]).to_4x4()
        path = os.path.join(os.path.dirname(out_blend), "_lid_render.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(path)
        w, h = image.size
        px = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(px)
        bpy.data.images.remove(image)
        visible = px.reshape(h, w, 4)[::-1, :, 0] > 0.5
        scores = {}
        for side in "LR":
            if targets[key][side] is None:
                continue
            lo, hi = targets[key]["box"][side]
            pad = (hi - lo) * 1.2
            x0, y0 = np.maximum((lo - pad).astype(int), 0)
            x1, y1 = (hi + pad).astype(int)
            want = targets[key][side][y0:y1, x0:x1]
            have = visible[y0:y1, x0:x1]
            union = (want | have).sum()
            scores[side] = float((want & have).sum() / max(union, 1))
        per_view[key] = scores
        total += sum(1.0 - v for v in scores.values())
        counted += len(scores)
        if save:
            from_photo = np.zeros_like(visible)
            for side in "LR":
                if targets[key][side] is not None:
                    from_photo |= targets[key][side]
            picture = np.zeros((h, w, 3), np.uint8)
            picture[from_photo & visible] = (160, 160, 160)
            picture[from_photo & ~visible] = (40, 200, 255)
            picture[visible & ~from_photo] = (255, 80, 200)
            np.save(os.path.join(os.path.dirname(out_blend), f"lids_{save}_{key}.npy"), picture)
    return total / max(counted, 1), per_view


params = {"upper_close_deg": 0.0, "lower_close_deg": 0.0, "eye_back_m": 0.0, "upper_edge_open_deg": 0.0,
          "upper_arch_deg": 0.0, "lower_arch_deg": 0.0}
steps = {"upper_close_deg": 8.0, "lower_close_deg": 8.0, "eye_back_m": 0.006, "upper_edge_open_deg": 6.0,
         "upper_arch_deg": 6.0, "lower_arch_deg": 6.0}
limits = {"upper_close_deg": (-30, 45), "lower_close_deg": (-30, 45), "eye_back_m": (-0.01, 0.03), "upper_edge_open_deg": (-20, 30),
          "upper_arch_deg": (-15, 30), "lower_arch_deg": (-15, 30)}
start, start_views = mismatch(params, save="before")
best = start
print(f"START eye mismatch {best:.3f} (0 = outlines identical; mismatch = 1 - overlap, averaged over eyes)", flush=True)
for rnd in range(7):
    for name in params:
        for direction in (1, -1):
            trial = dict(params)
            trial[name] = float(np.clip(params[name] + direction * steps[name], *limits[name]))
            if trial[name] == params[name]:
                continue
            value, _ = mismatch(trial)
            if value < best - 0.002:
                best, params = value, trial
                print(f"  round {rnd} {name} -> {trial[name]:.4f}: mismatch {best:.3f}", flush=True)
                break
    steps = {k: v / 2 for k, v in steps.items()}
final, final_views = mismatch(params, save="after")
os.remove(os.path.join(os.path.dirname(out_blend), "_lid_render.png"))
print("PARAMS", json.dumps({k: round(v, 4) for k, v in params.items()}))
print("EYE_OVERLAP before", {k: {s: round(v, 2) for s, v in d.items()} for k, d in start_views.items()})
print("EYE_OVERLAP after ", {k: {s: round(v, 2) for s, v in d.items()} for k, d in final_views.items()})
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump({"params": params, "before": start_views, "after": final_views}, handle, indent=1)
for obj in bpy.data.objects:
    obj.color = (1, 1, 1, 1)
bpy.ops.wm.save_as_mainfile(filepath=out_blend)
print("FIT_LIDS_OK")
