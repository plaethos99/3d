"""Render the walk loop to a turntable-free MP4 and a contact sheet.

Usage: blender -b joyce_final.blend -P render_walk.py -- <out_folder>
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joyce_lib as L  # noqa: E402

OUT = sys.argv[sys.argv.index("--") + 1]
FRAMES_DIR = os.path.join(OUT, "walk_frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 16
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 6
scene.render.resolution_x = 480
scene.render.resolution_y = 720
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"

a = math.radians(38)
camera_data = bpy.data.cameras.new("walk_camera")
camera_data.lens = 85
camera_data.sensor_fit = "VERTICAL"
camera_data.sensor_height = 36
camera = bpy.data.objects.new("walk_camera", camera_data)
scene.collection.objects.link(camera)
aim = Vector((0, 0, 0.77))
camera.location = Vector((-math.sin(a) * 4.7, -math.cos(a) * 4.7, 1.02))
camera.rotation_euler = (aim - camera.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = camera

for name, energy, size, az, height, colour in (
    ("key", 145, 3.0, -35, 2.6, (1.0, 0.97, 0.93)),
    ("fill", 42, 3.5, 50, 1.2, (0.92, 0.95, 1.0)),
    ("rim", 80, 2.0, 165, 2.6, (1, 1, 1)),
):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.size, data.color = energy, size, colour
    light = bpy.data.objects.new(name, data)
    scene.collection.objects.link(light)
    b = a + math.radians(az)
    light.location = (-math.sin(b) * 3.0, -math.cos(b) * 3.0, height)
    light.rotation_euler = (Vector((0, 0, 0.95)) - light.location).to_track_quat("-Z", "Y").to_euler()

world = bpy.data.worlds.new("walk_world")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.72, 0.71, 0.68, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.55
scene.world = world

mesh = bpy.data.meshes.new("floor")
mesh.from_pydata([(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)], [], [(0, 1, 2, 3)])
floor = bpy.data.objects.new("floor", mesh)
scene.collection.objects.link(floor)
floor_material = bpy.data.materials.new("floor")
floor_material.use_nodes = True
floor_material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.70, 0.69, 0.66, 1)
floor_material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.8
mesh.materials.append(floor_material)

paths = []
for frame in range(1, 25):
    scene.frame_set(frame)
    path = os.path.join(FRAMES_DIR, f"frame_{frame:04d}.png")
    scene.render.filepath = path
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    paths.append(path)
    print("FRAME_DONE", frame, flush=True)

# A contact sheet of every third frame shows the whole cycle in one image.
sheet = [L.load_rgba(paths[i]) for i in range(0, 24, 3)]
L.save_rgba(os.path.join(OUT, "walk_contact_sheet.png"), np.hstack(sheet))

# Put two loops into an MP4, using Blender's own sequencer so no extra tool is needed.
sequencer = scene.sequence_editor_create()
strips = getattr(sequencer, "strips", None)
if strips is None:
    strips = sequencer.sequences
for loop in range(2):
    for i, path in enumerate(paths):
        strips.new_image(f"f{loop}_{i}", path, 1, loop * 24 + i + 1)
scene.frame_start = 1
scene.frame_end = 48
scene.render.use_sequencer = True
settings = scene.render.image_settings
if hasattr(settings, "media_type"):
    settings.media_type = "VIDEO"
settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "HIGH"
scene.render.filepath = os.path.join(OUT, "joyce_walk.mp4")
bpy.ops.render.render(animation=True)
print("WALK_VIDEO_OK")
