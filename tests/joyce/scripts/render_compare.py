"""Render a saved scene like a reference image and measure the silhouette match.

Usage: blender -b file.blend -P render_compare.py -- <ref.png> <out_prefix>
       <azimuth_deg> [--scale H] [--samples N] [--elev Z] [--aim-x X] [--aim-z Z] [--frame F]
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joyce_lib as L  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
REF, PREFIX, AZIMUTH = args[0], args[1], float(args[2])


def option(name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default


HEIGHT = option("--height", 624)
SAMPLES = option("--samples", 48)
CAM_Z = option("--elev", 1.02)
AIM_X = option("--aim-x", 0.0)
AIM_Z = option("--aim-z", 0.770)
DIST = option("--dist", 4.70)
FRAME = option("--frame", 1)
POSE_NAME = option("--pose", "")

scene = bpy.context.scene
scene.frame_set(FRAME)
if POSE_NAME:
    import poses

    bpy.data.objects["joyce_rig"].animation_data_clear()
    poses.apply_pose(bpy.data.objects["joyce_rig"], poses.POSES[POSE_NAME])
    if POSE_NAME == "sit":
        poses.build_chair()
    bpy.context.view_layer.update()
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = SAMPLES > 4
scene.cycles.max_bounces = 6
scene.render.film_transparent = True
scene.render.resolution_x = HEIGHT * 2 // 3
scene.render.resolution_y = HEIGHT
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"

# Camera orbit: a positive azimuth shows her facing towards image right.
a = math.radians(AZIMUTH)
camera_data = bpy.data.cameras.new("compare_camera")
camera_data.lens = 85
camera_data.sensor_fit = "VERTICAL"
camera_data.sensor_height = 36
camera = bpy.data.objects.new("compare_camera", camera_data)
scene.collection.objects.link(camera)
right = Vector((math.cos(a), -math.sin(a), 0))
aim = Vector((0, 0, AIM_Z)) + right * (-AIM_X)
position = Vector((-math.sin(a) * DIST, -math.cos(a) * DIST, CAM_Z)) + right * (-AIM_X)
camera.location = position
camera.rotation_euler = (aim - position).to_track_quat("-Z", "Y").to_euler()
scene.camera = camera


def area(name, energy, size, offset_az, height, color=(1, 1, 1), dist=3.0):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    b = a + math.radians(offset_az)
    obj.location = (-math.sin(b) * dist, -math.cos(b) * dist, height)
    target = Vector((0, 0, 0.95))
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


if "compare_key" not in bpy.data.objects:
    area("compare_key", 145, 3.0, -35, 2.6, (1.0, 0.97, 0.93))
    area("compare_fill", 42, 3.5, 50, 1.2, (0.92, 0.95, 1.0))
    area("compare_rim", 80, 2.0, 165, 2.6)

world = bpy.data.worlds.new("compare_world")
world.use_nodes = True
background = world.node_tree.nodes["Background"]
background.inputs[0].default_value = (0.95, 0.95, 0.95, 1)
background.inputs[1].default_value = 0.32
scene.world = world

out = PREFIX + "_render.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
scene.render.filepath = out
bpy.ops.render.render(write_still=True)

render = L.load_rgba(out)
reference = L.load_rgba(REF)
print("REF_ALPHA_MIN_MAX", float(reference[..., 3].min()), float(reference[..., 3].max()))
iou, model_mask, ref_mask = L.silhouette_iou(render, reference)
L.save_rgba(PREFIX + "_compare.png", L.comparison_sheet(render, reference))

# A close-up of the top of the frame, so the face can be judged at full size.
h, w = render.shape[:2]
top = int(h * 0.60)
bottom = int(h * 0.90)
left, right_col = int(w * 0.22), int(w * 0.78)
reference_full = L.resize_nearest(reference, h, w)
crop = np.hstack([
    L.over_background(reference_full[top:bottom, left:right_col]),
    L.over_background(render[top:bottom, left:right_col]),
])
L.save_rgba(PREFIX + "_face.png", crop)


def bounds(mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


print("IOU", round(iou, 4), "AZ", AZIMUTH)

SPEC = option("--face-spec", "")
if SPEC:
    import face_metrics

    head_iou, head_ssim = face_metrics.head_scores(render, reference)
    print("HEAD_IOU", head_iou, "HEAD_SSIM", head_ssim)
    bpy.context.view_layer.update()
    mean_error, errors = face_metrics.landmark_report(scene, camera, face_metrics.load_spec(SPEC))
    print("LANDMARK_MEAN_PX", mean_error, errors)
print("MODEL_BOUNDS_FROM_BOTTOM", bounds(model_mask))
print("REF_BOUNDS_FROM_BOTTOM", bounds(ref_mask))
