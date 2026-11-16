"""Enlarge part of an image and draw a pixel grid, so features can be measured.

Usage: blender -b --factory-startup -P zoom_ref.py -- <in.png> <out.png> x0 y0 x1 y1 scale step
Coordinates count from the image's top-left corner. Lines every `step` pixels;
every fifth line is darker. The crop's top-left corner is at (x0, y0).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joyce_lib as L  # noqa: E402

a = sys.argv[sys.argv.index("--") + 1:]
source, target = a[0], a[1]
x0, y0, x1, y1, scale, step = (int(v) for v in a[2:8])

image = L.over_background(L.load_rgba(source))
height = image.shape[0]
# Stored rows start at the bottom, so convert to top-down for cropping.
top_down = image[::-1]
crop = top_down[y0:y1, x0:x1]
big = np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1)

first_x = (x0 // step + 1) * step
first_y = (y0 // step + 1) * step
for x in range(first_x, x1, step):
    col = (x - x0) * scale
    strength = 0.75 if x % (step * 5) == 0 else 0.30
    big[:, col, :3] = big[:, col, :3] * (1 - strength) + np.array((0.0, 0.2, 0.9)) * strength
for y in range(first_y, y1, step):
    row = (y - y0) * scale
    strength = 0.75 if y % (step * 5) == 0 else 0.30
    big[row, :, :3] = big[row, :, :3] * (1 - strength) + np.array((0.9, 0.1, 0.1)) * strength

L.save_rgba(target, big[::-1])
print("ZOOM_OK", big.shape, "first_x", first_x, "first_y", first_y)
