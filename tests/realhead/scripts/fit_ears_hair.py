"""Fit the ears and the hair's thickness so the head's outline matches the photos' outlines.

Usage: blender -b head.blend -P fit_ears_hair.py -- <cameras.json> <masks_dir> <out.blend> <report.json>

The face tracker has no points on the ears or the hair, so their shape comes from the outline
instead. For each camera the head's silhouette is rendered and compared with the photo's,
above the chin, where the hoodie hides nothing. The mismatch is measured as the average gap
between the two outlines, in pixels. Ear position, size and angle, and hair thickness on top, at the
sides and at the back, are adjusted one at a time, keeping any change that narrows the gap.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
cams_path, masks_dir, out_blend, report_path = args[:4]
with open(cams_path, encoding="utf-8") as f:
    cams = json.load(f)
with open(os.path.join(masks_dir, "masks.json"), encoding="utf-8") as f:
    mask_info = json.load(f)

SCALE = 0.5
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.film_transparent = True
first = next(iter(cams.values()))
scene.render.resolution_x = int(first["size"][0] * SCALE)
scene.render.resolution_y = int(first["size"][1] * SCALE)
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
for obj in bpy.data.objects:
    if obj.type == "LIGHT":
        obj.hide_render = True
data = bpy.data.cameras.new("fitcam")
data.sensor_fit = "HORIZONTAL"
data.sensor_width = 36.0
cam = bpy.data.objects.new("fitcam", data)
scene.collection.objects.link(cam)
scene.camera = cam

masks = {}
for key in cams:
    image = bpy.data.images.load(os.path.join(masks_dir, f"mask_{key}.png"))
    w, h = image.size
    px = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(px)
    masks[key] = px.reshape(h, w, 4)[::-1, :, 0] > 0.5
    bpy.data.images.remove(image)

head = bpy.data.objects["head"]
mesh = head.data
count = len(mesh.vertices)
rest = np.empty(count * 3)
mesh.vertices.foreach_get("co", rest)
rest = rest.reshape(-1, 3)
normals = np.empty(count * 3)
mesh.vertices.foreach_get("normal", normals)
normals = normals.reshape(-1, 3)

# Ear regions: an ellipsoid around each ear, fading out so the join to the head stays smooth.
EAR_CENTRE = np.array((0.080, 0.090, -0.025))
EAR_RADII = np.array((0.022, 0.030, 0.042))
EAR_ROOT = np.array((0.070, 0.088, -0.025))
ear_weight = {}
for side, sign in (("L", 1.0), ("R", -1.0)):
    centre = EAR_CENTRE * np.array((sign, 1, 1))
    d = np.linalg.norm((rest - centre) / EAR_RADII, axis=1)
    w = np.clip((1.25 - d) / 0.45, 0, 1)
    w = w * w * (3 - 2 * w)
    w *= (rest[:, 0] * sign) > 0.060
    ear_weight[side] = w
print("EAR_VERTICES", {k: int((v > 0.5).sum()) for k, v in ear_weight.items()})

# Hair region: the same hairline as the texture uses (see project_texture.py).
behind = np.clip((rest[:, 1] - 0.02) / 0.04, 0, 1)
hair_line = -0.03 - 0.05 * behind
hairy = np.clip((rest[:, 2] - hair_line) / 0.015, 0, 1) * np.clip((rest[:, 1] + 0.01) / 0.03, 0, 1)
hairy = np.maximum(hairy, np.clip((rest[:, 2] - 0.085) / 0.02, 0, 1))
hairy *= np.abs(rest[:, 0]) < 0.12  # not the shoulders
top_share = np.clip((rest[:, 2] - 0.03) / 0.06, 0, 1)
skull_share = np.clip((rest[:, 1] - 0.06) / 0.14, 0, 1) * (rest[:, 2] > -0.15)
print("HAIR_VERTICES", int((hairy > 0.5).sum()))


def shaped(p):
    coords = rest.copy()
    for side, sign in (("L", 1.0), ("R", -1.0)):
        root = EAR_ROOT * np.array((sign, 1, 1))
        flare = math.radians(p["ear_flare_deg"]) * sign
        rot = np.array(Matrix.Rotation(flare, 3, "Z"))
        moved = (rest - root) @ rot.T * p["ear_scale"] + root + np.array((0, p["ear_back_m"], p["ear_up_m"]))
        w = ear_weight[side][:, None]
        coords = coords * (1 - w) + moved * w
    lower = p["hair_side_m"] * (1 - behind) + p["hair_back_m"] * behind
    thickness = p["hair_top_m"] * top_share + lower * (1 - top_share)
    coords += normals * (hairy * thickness)[:, None]
    # The back of the skull: stretch everything behind the ears further back, fading in from the ears.
    coords[:, 1] += p["skull_back_m"] * skull_share
    return coords


def outline_gap(p, keys=None, save=None):
    mesh.vertices.foreach_set("co", shaped(p).ravel())
    mesh.update()
    gaps = {}
    for key in keys or cams:
        view = cams[key]
        cam.data.lens = view["lens_mm"]
        cam.matrix_world = Matrix.Translation(Vector(view["centre"])) @ Matrix(view["rotation"]).to_4x4()
        path = os.path.join(masks_dir, "_fit_render.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(path)
        w, h = image.size
        px = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(px)
        bpy.data.images.remove(image)
        rendered = px.reshape(h, w, 4)[::-1, :, 3] > 0.5
        rows = int(mask_info[key]["chin_row"])
        photo = masks[key][:rows]
        model = rendered[:rows]
        xor = photo ^ model
        edge = photo ^ np.roll(photo, 1, 0) | photo ^ np.roll(photo, 1, 1)
        gaps[key] = float(xor.sum() / max(edge.sum(), 1))
        if save:
            picture = np.zeros((rows, w, 3), np.uint8)
            picture[photo & model] = (120, 120, 120)
            picture[photo & ~model] = (40, 200, 255)   # photo only: the model is too small here
            picture[model & ~photo] = (255, 80, 200)   # model only: the model is too big here
            np.save(os.path.join(masks_dir, f"{save}_{key}.npy"), picture)
    return float(np.mean(list(gaps.values()))), gaps


params = {"ear_back_m": 0.0, "ear_up_m": 0.0, "ear_scale": 1.0, "ear_flare_deg": 0.0, "hair_top_m": 0.0, "hair_side_m": 0.0, "hair_back_m": 0.0, "skull_back_m": 0.0}
steps = {"ear_back_m": 0.008, "ear_up_m": 0.006, "ear_scale": 0.08, "ear_flare_deg": 8.0, "hair_top_m": 0.006, "hair_side_m": 0.004, "hair_back_m": 0.006, "skull_back_m": 0.008}
limits = {"ear_back_m": (-0.03, 0.03), "ear_up_m": (-0.02, 0.02), "ear_scale": (0.8, 1.35), "ear_flare_deg": (-25, 25),
          "hair_top_m": (0.0, 0.03), "hair_side_m": (0.0, 0.02), "hair_back_m": (0.0, 0.03), "skull_back_m": (-0.02, 0.04)}
start, start_gaps = outline_gap(params, save="before")
best = start
print(f"START outline gap {start:.2f} px", flush=True)
for rnd in range(5):
    for name in params:
        for direction in (1, -1):
            trial = dict(params)
            trial[name] = float(np.clip(params[name] + direction * steps[name], *limits[name]))
            if trial[name] == params[name]:
                continue
            value, _ = outline_gap(trial)
            if value < best - 0.01:
                best, params = value, trial
                print(f"  round {rnd} {name} -> {trial[name]:.4f}: gap {best:.2f} px", flush=True)
                break
    steps = {k: v / 2 for k, v in steps.items()}
final, final_gaps = outline_gap(params, save="after")
print("PARAMS", json.dumps({k: round(v, 4) for k, v in params.items()}))
print(f"OUTLINE_GAP_PX before {start:.2f} after {final:.2f}")
for key in cams:
    print(f"  {key} yaw {mask_info[key]['yaw']:5.1f}: {start_gaps[key]:.2f} -> {final_gaps[key]:.2f} px")
os.remove(os.path.join(masks_dir, "_fit_render.png"))
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump({"params": params, "before_px": start, "after_px": final, "before": start_gaps, "after": final_gaps}, handle, indent=1)
bpy.ops.wm.save_as_mainfile(filepath=out_blend)
print("FIT_EARS_HAIR_OK")
