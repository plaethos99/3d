"""Fit a hairstyle's outer volume to the hair outlines cut from reference images.

Usage: blender -b character.blend -P fit_hair_volume.py -- <masks_dir> <views.json> <out.blend> <report.json>

views.json lists the reference views: [{"name", "azimuth", "pose"}], rendered with the same camera
as the character's comparison renders (85 mm, 4.7 m away, 1.02 m high, aimed at 0.77 m). The volume
(object "hair_base") is reshaped around the skull's centre by a few broad settings: height on top,
fullness at the sides, depth at the back, how low the back reaches, height at the front, and
overall size. Each is adjusted in turn, keeping changes that make the rendered hair overlap the
reference hair better. The face and body hide hair in the renders just as they do in the images.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
masks_dir, views_path, out_blend, report_path = args[:4]
with open(views_path, encoding="utf-8") as f:
    views = json.load(f)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "joyce", "scripts"))
import poses  # noqa: E402

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "FLAT"
scene.display.shading.color_type = "OBJECT"
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new("black")
scene.world.color = (0, 0, 0)
scene.view_settings.view_transform = "Standard"
scene.render.image_settings.file_format = "PNG"
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 832, 1248, 100
rig = bpy.data.objects["joyce_rig"]
hair = bpy.data.objects["hair_base"]
for obj in bpy.data.objects:
    if obj.type == "MESH":
        obj.color = (1, 1, 1, 1) if obj is hair else (0, 0, 0, 1)
    if obj.name == "hair_inner":
        obj.hide_render = True
data = bpy.data.cameras.new("fitcam")
data.lens, data.sensor_fit, data.sensor_height = 85, "VERTICAL", 36
cam = bpy.data.objects.new("fitcam", data)
scene.collection.objects.link(cam)
scene.camera = cam

targets = {}
for view in views:
    image = bpy.data.images.load(os.path.join(masks_dir, f"hair_{view['name']}.png"))
    px = np.empty(image.size[0] * image.size[1] * 4, np.float32)
    image.pixels.foreach_get(px)
    w, h = image.size
    bpy.data.images.remove(image)
    targets[view["name"]] = px.reshape(h, w, 4)[::-1, :, 0] > 0.5

rest = np.array([v.co for v in hair.data.vertices])
inner = bpy.data.objects.get("hair_inner")
ref = np.array([v.co for v in inner.data.vertices]) if inner else rest
C = (ref.min(axis=0) + ref.max(axis=0)) / 2
C[2] = ref.min(axis=0)[2] + (ref.max(axis=0)[2] - ref.min(axis=0)[2]) * 0.45
d = rest - C


def ramp(x, width):
    return np.clip(x / width, 0, 1)


def shaped(p):
    q = d * p["size"]
    q[:, 0] *= 1 + p["side"]
    q[:, 2] += p["top_m"] * ramp(d[:, 2], 0.12)
    q[:, 1] += p["back_m"] * ramp(d[:, 1], 0.12)
    q[:, 2] -= p["nape_m"] * ramp(d[:, 1] - 0.02, 0.08) * ramp(-d[:, 2], 0.08)
    q[:, 2] += p["front_m"] * ramp(-d[:, 1], 0.1) * ramp(d[:, 2] + 0.02, 0.1)
    # The sides in front of the ears: pulled back (positive) toward the back of the head.
    q[:, 1] += p["temple_m"] * ramp(-d[:, 1], 0.08) * ramp(-d[:, 2] + 0.04, 0.1)
    q[:, 1] += p["shift_y"]
    q[:, 2] += p["shift_z"]
    return C + q


def score(p, save=None):
    hair.data.vertices.foreach_set("co", shaped(p).ravel())
    hair.data.update()
    result = {}
    for view in views:
        rig.animation_data_clear() if rig.animation_data else None
        for pb in rig.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.location = (0, 0, 0)
        if view["pose"] != "rest":
            poses.apply_pose(rig, poses.POSES[view["pose"]])
        bpy.context.view_layer.update()
        a = math.radians(view["azimuth"])
        cam.location = Vector((-math.sin(a) * 4.7, -math.cos(a) * 4.7, 1.02))
        cam.rotation_euler = (Vector((0, 0, 0.77)) - cam.location).to_track_quat("-Z", "Y").to_euler()
        path = os.path.join(masks_dir, "_fit.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(path)
        px = np.empty(image.size[0] * image.size[1] * 4, np.float32)
        image.pixels.foreach_get(px)
        w, h = image.size
        bpy.data.images.remove(image)
        mine = px.reshape(h, w, 4)[::-1, :, 0] > 0.5
        want = targets[view["name"]]
        result[view["name"]] = float((mine & want).sum() / max((mine | want).sum(), 1))
        if save:
            picture = np.zeros(mine.shape + (3,), np.uint8)
            picture[mine & want] = (160, 160, 160)
            picture[want & ~mine] = (255, 200, 40)
            picture[mine & ~want] = (200, 80, 255)
            np.save(os.path.join(masks_dir, f"{save}_{view['name']}.npy"), picture)
    return float(np.mean(list(result.values()))), result


params = {"shift_y": 0.0, "shift_z": 0.0, "size": 1.0, "side": 0.0, "top_m": 0.0, "back_m": 0.0, "nape_m": 0.0,
          "front_m": 0.0, "temple_m": 0.0}
steps = {"shift_y": 0.02, "shift_z": 0.02, "size": 0.06, "side": 0.08, "top_m": 0.02, "back_m": 0.02, "nape_m": 0.03,
         "front_m": 0.02, "temple_m": 0.02}
limits = {"size": (0.8, 1.3), "side": (-0.3, 0.4), "top_m": (-0.05, 0.08), "back_m": (-0.05, 0.08),
          "nape_m": (-0.03, 0.10), "front_m": (-0.05, 0.06), "temple_m": (-0.03, 0.08),
          "shift_y": (-0.06, 0.08), "shift_z": (-0.06, 0.06)}
start, start_views = score(params, save="before")
best = start
print(f"START hair overlap {best:.3f} {start_views}", flush=True)
for rnd in range(6):
    for name in params:
        for direction in (1, -1):
            trial = dict(params)
            trial[name] = float(np.clip(params[name] + direction * steps[name], *limits[name]))
            if trial[name] == params[name]:
                continue
            value, _ = score(trial)
            if value > best + 0.002:
                best, params = value, trial
                print(f"  round {rnd} {name} -> {trial[name]:.3f}: overlap {best:.3f}", flush=True)
                break
    steps = {k: v / 2 for k, v in steps.items()}
final, final_views = score(params, save="after")
os.remove(os.path.join(masks_dir, "_fit.png"))
print("PARAMS", json.dumps({k: round(v, 3) for k, v in params.items()}))
print(f"HAIR_OVERLAP before {start:.3f} after {final:.3f}", {k: (round(start_views[k], 3), round(v, 3)) for k, v in final_views.items()})
for pb in rig.pose.bones:
    pb.rotation_quaternion = (1, 0, 0, 0)
for obj in bpy.data.objects:
    obj.color = (1, 1, 1, 1)
for obj in [cam]:
    bpy.data.objects.remove(obj, do_unlink=True)
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump({"params": params, "before": start_views, "after": final_views}, handle, indent=1)
bpy.ops.wm.save_as_mainfile(filepath=out_blend)
print("FIT_HAIR_VOLUME_OK")
