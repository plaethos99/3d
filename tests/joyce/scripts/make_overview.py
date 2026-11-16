"""Collect the build stages and final poses into one image, and check the video.

Usage: blender -b --factory-startup -P make_overview.py -- <joyce_test_folder>
"""
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joyce_lib as L  # noqa: E402

root = sys.argv[sys.argv.index("--") + 1]
renders = os.path.join(root, "renders")
HEIGHT, WIDTH = 624, 416


def tile(path):
    image = L.resize_nearest(L.load_rgba(path), HEIGHT, WIDTH)
    return L.over_background(image)


rows = [
    [
        os.path.join(root, "refs", "stand.png"),
        os.path.join(renders, "01_blockout_render.png"),
        os.path.join(renders, "02_full_v1_render.png"),
        os.path.join(renders, "11_stand_render.png"),
    ],
    [
        os.path.join(renders, "11_wave_render.png"),
        os.path.join(renders, "11_surprised_render.png"),
        os.path.join(renders, "11_sit_render.png"),
        os.path.join(renders, "12_walk-a_az38_render.png"),
    ],
]
sheet = np.vstack([np.hstack([tile(p) for p in row]) for row in rows][::-1])
L.save_rgba(os.path.join(root, "joyce_overview.png"), sheet)

clip = bpy.data.movieclips.load(os.path.join(renders, "joyce_walk.mp4"))
print("VIDEO_FRAMES", clip.frame_duration, "SIZE", tuple(clip.size), "FPS", clip.fps)
print("OVERVIEW_OK")
