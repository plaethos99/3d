"""Put the reference, the previous version and the new version side by side.

Usage: blender -b --factory-startup -P make_before_after.py -- <joyce_test_folder>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joyce_lib as L  # noqa: E402

root = sys.argv[sys.argv.index("--") + 1]
renders = os.path.join(root, "renders")
W, H = 624, 936

sources = [
    os.path.join(root, "refs", "stand.png"),
    os.path.join(renders, "11_stand_render.png"),
    os.path.join(renders, "face", "final_stand_render.png"),
]
full = [L.over_background(L.resize_nearest(L.load_rgba(p), H, W)) for p in sources]
# Stored rows start at the bottom. The head sits between 10% and 38% from the top.
top, bottom = int(H * 0.62), int(H * 0.90)
left, right = int(W * 0.20), int(W * 0.80)
heads = []
for image in full:
    crop = image[top:bottom, left:right]
    heads.append(np.repeat(np.repeat(crop, 2, axis=0), 2, axis=1)[: crop.shape[0] * 2])
head_width = sum(h.shape[1] for h in heads)
body = np.hstack(full)
body = L.resize_nearest(body, int(body.shape[0] * head_width / body.shape[1]), head_width)
sheet = np.vstack([body, np.hstack(heads)])  # the head row prints above the body row
L.save_rgba(os.path.join(root, "joyce_before_after.png"), sheet)

# Refresh the general overview with the final renders.
overview = [
    [
        os.path.join(root, "refs", "stand.png"),
        os.path.join(renders, "01_blockout_render.png"),
        os.path.join(renders, "11_stand_render.png"),
        os.path.join(renders, "face", "final_stand_render.png"),
    ],
    [
        os.path.join(renders, "face", "final_wave_render.png"),
        os.path.join(renders, "face", "final_surprised_render.png"),
        os.path.join(renders, "face", "final_sit_render.png"),
        os.path.join(renders, "face", "final_back_render.png"),
    ],
]


def tile(path):
    return L.over_background(L.resize_nearest(L.load_rgba(path), 624, 416))


rows = [np.hstack([tile(p) for p in row]) for row in overview]
L.save_rgba(os.path.join(root, "joyce_overview.png"), np.vstack(rows[::-1]))
print("SHEETS_OK")
