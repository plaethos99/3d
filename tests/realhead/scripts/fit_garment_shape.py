"""Reshape a character's upper garments so the torso's outline matches reference images.

Usage: blender -b character.blend -P fit_garment_shape.py -- <views.json> <out.blend> <report.json>

views.json: [{"name", "image", "azimuth", "pose", "rows": [top, bottom]}], with images rendered by the
character's comparison camera (85 mm, 4.7 m away, 1.02 m high, aimed at 0.77 m, 832x1248).
Only the image rows from shoulders to hem are compared, so the head, hair and legs don't count.

Everything sewn together is reshaped by one shared set of broad changes, so nothing comes apart:
sloping shoulders, chest width, a fuller front and a rounder upper back, overall depth, a hem
that flares out and hangs lower. Sleeves, cuffs and hands just follow the shoulder line.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

views_path, out_blend, report_path = sys.argv[sys.argv.index("--") + 1:][:3]
with open(views_path, encoding="utf-8") as f:
    views = json.load(f)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "joyce", "scripts"))
import poses  # noqa: E402

TORSO = ["cardigan", "band", "hem", "pocket_L", "pocket_R", "pocket_trim_L", "pocket_trim_R", "blouse", "collar_L",
         "collar_R", "placket", "chest_v", "button_0.79", "button_0.86", "button_0.93"]
ARMS = ["sleeve_L", "sleeve_R", "cuff_L", "cuff_R", "palm_L", "palm_R", "thumb_L", "thumb_R"] + \
       [f"finger{i}_{s}" for i in range(4) for s in "LR"]
TORSO = [n for n in TORSO if n in bpy.data.objects]
ARMS = [n for n in ARMS if n in bpy.data.objects]

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "FLAT"
scene.display.shading.color_type = "SINGLE"
scene.display.shading.single_color = (1, 1, 1)
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 832, 1248, 100
rig = bpy.data.objects["joyce_rig"]
data = bpy.data.cameras.new("fitcam")
data.lens, data.sensor_fit, data.sensor_height = 85, "VERTICAL", 36
cam = bpy.data.objects.new("fitcam", data)
scene.collection.objects.link(cam)
scene.camera = cam

targets = {}
for view in views:
    image = bpy.data.images.load(view["image"])
    w, h = image.size
    px = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(px)
    bpy.data.images.remove(image)
    targets[view["name"]] = px.reshape(h, w, 4)[::-1, :, 3] > 0.5

rest = {n: np.array([v.co for v in bpy.data.objects[n].data.vertices]) for n in TORSO + ARMS}
card = rest["cardigan"]
CX, CY = float((card[:, 0].min() + card[:, 0].max()) / 2), float((card[:, 1].min() + card[:, 1].max()) / 2)
print("TORSO_CENTRE", round(CX, 3), round(CY, 3), "cardigan z", round(float(card[:, 2].min()), 3), round(float(card[:, 2].max()), 3))


def clamp(x):
    return np.clip(x, 0, 1)


def bump(z, low, high):
    mid, half = (low + high) / 2, (high - low) / 2
    return clamp(1 - np.abs(z - mid) / half)


def torso_field(p, q):
    x, y, z = q[:, 0] - CX, q[:, 1] - CY, q[:, 2]
    flare = 1 + p["flare"] * clamp((0.66 - z) / 0.14)
    width = (1 + p["chest_width"] * bump(z, 0.72, 1.02) + p["width"]) * (1 + p["shoulder_width"] * clamp((z - 0.85) / 0.12))
    depth = 1 + p["depth"]
    nx = x * flare * width
    ny = y * flare * depth
    ny = ny - p["belly_m"] * bump(z, 0.6, 0.85) * (y < 0)
    ny = ny + p["back_m"] * bump(z, 0.82, 1.04) * (y > 0)
    # A fuller back all the way down, without moving the front (which would let the body show through).
    ny = ny + p["back_full_m"] * clamp(y / 0.08) * clamp((z - 0.5) / 0.08)
    nz = z - p["hem_drop_m"] * clamp((0.62 - z) / 0.1)
    nz = nz - p["shoulder_drop_m"] * clamp((np.abs(x) - 0.08) / 0.1) * clamp((z - 0.88) / 0.08)
    return np.stack([nx + CX, ny + CY + p["shift_y_m"], nz], axis=1)


def apply(p):
    for n in TORSO:
        co = torso_field(p, rest[n])
        bpy.data.objects[n].data.vertices.foreach_set("co", co.ravel())
        bpy.data.objects[n].data.update()
    for n in ARMS:
        co = rest[n].copy()
        co[:, 2] -= p["shoulder_drop_m"]
        side = np.sign(co[:, 0].mean() - CX)
        co[:, 0] += side * (p["chest_width"] + p["width"] + p["shoulder_width"]) * 0.12
        co[:, 1] += p["shift_y_m"]
        bpy.data.objects[n].data.vertices.foreach_set("co", co.ravel())
        bpy.data.objects[n].data.update()


def score(p, save=None):
    apply(p)
    result = {}
    for view in views:
        for pb in rig.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = (1, 0, 0, 0)
        if view["pose"] != "rest":
            poses.apply_pose(rig, poses.POSES[view["pose"]])
        bpy.context.view_layer.update()
        a = math.radians(view["azimuth"])
        cam.location = Vector((-math.sin(a) * 4.7, -math.cos(a) * 4.7, 1.02))
        cam.rotation_euler = (Vector((0, 0, 0.77)) - cam.location).to_track_quat("-Z", "Y").to_euler()
        path = os.path.join(os.path.dirname(out_blend), "_garment_fit.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(path)
        w, h = image.size
        px = np.empty(w * h * 4, np.float32)
        image.pixels.foreach_get(px)
        bpy.data.images.remove(image)
        mine = px.reshape(h, w, 4)[::-1, :, 3] > 0.5
        top, bottom = view["rows"]
        a_, b_ = targets[view["name"]][top:bottom], mine[top:bottom]
        result[view["name"]] = float((a_ & b_).sum() / max((a_ | b_).sum(), 1))
        if save:
            picture = np.zeros(a_.shape + (3,), np.uint8)
            picture[a_ & b_] = (160, 160, 160)
            picture[a_ & ~b_] = (255, 200, 40)
            picture[b_ & ~a_] = (200, 80, 255)
            np.save(os.path.join(os.path.dirname(out_blend), f"garment_{save}_{view['name']}.npy"), picture)
    return float(np.mean(list(result.values()))), result


params = {"shift_y_m": 0.0, "back_full_m": 0.0, "width": 0.0, "shoulder_width": 0.0, "flare": 0.0, "hem_drop_m": 0.0, "chest_width": 0.0, "depth": 0.0, "belly_m": 0.0, "back_m": 0.0, "shoulder_drop_m": 0.0}
steps = {"shift_y_m": 0.02, "back_full_m": 0.02, "width": 0.06, "shoulder_width": 0.06, "flare": 0.08, "hem_drop_m": 0.02, "chest_width": 0.06, "depth": 0.08, "belly_m": 0.02, "back_m": 0.02, "shoulder_drop_m": 0.015}
limits = {"shift_y_m": (0.0, 0.0), "back_full_m": (0.0, 0.06), "width": (-0.15, 0.2), "shoulder_width": (-0.2, 0.15), "flare": (-0.1, 0.35), "hem_drop_m": (-0.04, 0.08), "chest_width": (-0.1, 0.25), "depth": (-0.1, 0.35),
          "belly_m": (-0.02, 0.06), "back_m": (-0.02, 0.06), "shoulder_drop_m": (-0.02, 0.05)}
start, start_views = score(params, save="before")
best = start
print(f"START torso overlap {best:.3f} {start_views}", flush=True)
# Each round tries every change and keeps the single best one, so the order of settings doesn't
# decide the outcome; a step size halves when nothing at that size helps.
for rnd in range(14):
    tried = []
    for name in params:
        for direction in (1, -1):
            trial = dict(params)
            trial[name] = float(np.clip(params[name] + direction * steps[name], *limits[name]))
            if trial[name] != params[name]:
                tried.append((score(trial)[0], name, trial))
    value, name, trial = max(tried, key=lambda t: t[0])
    if value > best + 0.001:
        best, params = value, trial
        print(f"  round {rnd} {name} -> {trial[name]:.3f}: overlap {best:.3f}", flush=True)
    else:
        steps = {k: v / 2 for k, v in steps.items()}
        if max(steps.values()) < 0.004:
            break
final, final_views = score(params, save="after")
os.remove(os.path.join(os.path.dirname(out_blend), "_garment_fit.png"))
print("PARAMS", json.dumps({k: round(v, 3) for k, v in params.items()}))
print(f"TORSO_OVERLAP before {start:.3f} after {final:.3f}", {k: (round(start_views[k], 3), round(v, 3)) for k, v in final_views.items()})
for pb in rig.pose.bones:
    pb.rotation_quaternion = (1, 0, 0, 0)
bpy.data.objects.remove(cam, do_unlink=True)
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump({"params": params, "before": start_views, "after": final_views}, handle, indent=1)
bpy.ops.wm.save_as_mainfile(filepath=out_blend)
print("FIT_GARMENT_OK")
