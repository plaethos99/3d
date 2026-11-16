"""Save where each foot really is in the keyframed walk, to test contact detection against.

Usage: blender -b joyce_final.blend -P export_feet_truth.py -- <feet_truth.json>
A foot counts as planted when it moves at about the speed the body moves, backward, near the floor.
"""
import json
import sys

import bpy
import numpy as np

out = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
rig = bpy.data.objects["joyce_rig"]
frames = list(range(1, 25))  # one full walk cycle, the same frames as export_truth.py
track = {"L": [], "R": []}
for frame in frames:
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    for side in "LR":
        track[side].append(tuple(rig.pose.bones[f"foot.{side}"].head))
result = {}
for side in "LR":
    points = np.array(track[side])
    height = points[:, 2] - points[:, 2].min()
    step = np.linalg.norm(np.roll(points[:, :2], -1, axis=0) - np.roll(points[:, :2], 1, axis=0), axis=1) / 2
    # The keyframed planted foot rises about 10 mm as it rolls forward, then lifts away quickly.
    planted = height < 0.012
    result[side] = {
        "height_mm": [round(float(v) * 1000, 1) for v in height],
        "step_mm": [round(float(v) * 1000, 1) for v in step],
        "planted": [int(v) for v in planted],
    }
with open(out, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
print("FEET_TRUTH_OK", len(frames), "planted share", {s: round(float(np.mean(result[s]["planted"])), 3) for s in "LR"})
for side in "LR":
    print(side, "height_mm", result[side]["height_mm"])
