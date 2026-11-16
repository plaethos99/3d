"""Render the retargeted animation from straight on, which is the tracked camera's view.

Usage: blender -b motion.blend -P render_motion.py -- <frames_folder> <output.mp4>
Long clips render every Nth frame, held in the video, to keep a preview to about 150 renders.
The camera follows her across the floor, so she stays framed as she travels.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
FRAMES_DIR, VIDEO = args[0], args[1]
os.makedirs(FRAMES_DIR, exist_ok=True)

scene = bpy.context.scene
rig = bpy.data.objects["joyce_rig"]
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 12
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 4
scene.render.resolution_x = 480
scene.render.resolution_y = 720
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"

data = bpy.data.cameras.new("cam")
data.lens = 85
data.sensor_fit = "VERTICAL"
data.sensor_height = 36
camera = bpy.data.objects.new("cam", data)
scene.collection.objects.link(camera)
scene.camera = camera

lights = []
for name, energy, size, az, height, colour in (
    ("key", 145, 3.0, -35, 2.6, (1.0, 0.97, 0.93)),
    ("fill", 42, 3.5, 50, 1.2, (0.92, 0.95, 1.0)),
    ("rim", 80, 2.0, 165, 2.6, (1, 1, 1)),
):
    light_data = bpy.data.lights.new(name, "AREA")
    light_data.energy, light_data.size, light_data.color = energy, size, colour
    light = bpy.data.objects.new(name, light_data)
    scene.collection.objects.link(light)
    b = math.radians(az)
    lights.append((light, Vector((-math.sin(b) * 3.0, -math.cos(b) * 3.0, height))))

world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.72, 0.71, 0.68, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.55
scene.world = world
mesh = bpy.data.meshes.new("floor")
mesh.from_pydata([(-40, -40, 0), (40, -40, 0), (40, 40, 0), (-40, 40, 0)], [], [(0, 1, 2, 3)])
floor = bpy.data.objects.new("floor", mesh)
scene.collection.objects.link(floor)
material = bpy.data.materials.new("floor")
material.use_nodes = True
material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.70, 0.69, 0.66, 1)
mesh.materials.append(material)

paths = []
total = scene.frame_end
step = max(1, math.ceil(total / 150))
print("RENDER_STEP", step, flush=True)
for frame in range(1, total + 1):
    if (frame - 1) % step:
        paths.append(paths[-1])
        continue
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    center = rig.pose.bones["hips"].head
    aim = Vector((center.x, center.y, 0.77))
    camera.location = Vector((center.x, center.y - 4.7, 1.02))
    camera.rotation_euler = (aim - camera.location).to_track_quat("-Z", "Y").to_euler()
    for light, offset in lights:
        light.location = Vector((center.x, center.y, 0)) + offset
        light.rotation_euler = (Vector((center.x, center.y, 0.95)) - light.location).to_track_quat("-Z", "Y").to_euler()
    path = os.path.join(FRAMES_DIR, f"frame_{frame:04d}.png")
    scene.render.filepath = path
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    paths.append(path)
    if frame % 12 == 0:
        print("FRAME_DONE", frame, "of", total, flush=True)

fps = scene.render.fps
editor = scene.sequence_editor_create()
strips = getattr(editor, "strips", None)
if strips is None:
    strips = editor.sequences
for i, path in enumerate(paths):
    strips.new_image(f"f{i}", path, 1, i + 1)
scene.frame_start = 1
scene.frame_end = total
scene.render.use_sequencer = True
settings = scene.render.image_settings
if hasattr(settings, "media_type"):
    settings.media_type = "VIDEO"
settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "HIGH"
scene.render.filepath = VIDEO
bpy.ops.render.render(animation=True)
print("RENDER_MOTION_OK", total)
