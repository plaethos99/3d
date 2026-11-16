"""Render the finished head from angles the video never showed, to judge it honestly.

Usage: blender -b head.blend -P render_turntable.py -- <out.png> <samples> [yaw,pitch ...]
Lighting is the same soft studio set-up used to match the photos, with a neutral backdrop.
"""
import os
import sys

import bpy

sys.path.append(os.path.dirname(__file__))
import headcam  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
out, samples = args[0], int(args[1])
angles = [tuple(float(v) for v in a.split(",")) for a in args[2:]] or [(-90, 5), (-35, 5), (0, 5), (35, 5), (90, 5), (150, 10)]
scene = headcam.setup_render(640, 800, samples)
headcam.soft_studio(scene, background=(1.0, 1.0, 1.0), strength=0.9, seen_backdrop=(0.30, 0.30, 0.30), key=6.0, fill=0.0)
headcam.catchlights(scene)
cam = headcam.camera(scene, focal_mm=85.0, sensor_mm=24.0)
paths = []
for yaw, pitch in angles:
    headcam.place_orbit(cam, yaw, pitch, 0.85, target=__import__("mathutils").Vector((0, 0.03, -0.03)))
    path = os.path.splitext(out)[0] + f"_{int(yaw):+d}.png"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    paths.append(path)
    print("TURN", yaw, pitch, flush=True)
print("TURN_PATHS", "|".join(paths))
print("TURNTABLE_OK")
