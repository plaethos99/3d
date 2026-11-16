"""Save the exact bone directions of the keyframed walk, as ground truth for testing tracking.

Usage: blender -b joyce_final.blend -P export_truth.py -- <truth.json>
"""
import json
import sys

import bpy

out = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
rig = bpy.data.objects["joyce_rig"]
names = [
    "hips", "chest", "head", "upper_arm.L", "upper_arm.R", "forearm.L", "forearm.R",
    "thigh.L", "thigh.R", "shin.L", "shin.R", "foot.L", "foot.R",
]
frames = []
for frame in range(1, 25):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    pose = rig.pose.bones
    frames.append({
        "dirs": {n: tuple((pose[n].tail - pose[n].head).normalized()) for n in names},
        # The left-to-right axis across the hips gives her heading.
        "thigh_axis": tuple((pose["thigh.L"].head - pose["thigh.R"].head).normalized()),
    })
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"frames": frames}, handle)
print("TRUTH_OK", len(frames))
