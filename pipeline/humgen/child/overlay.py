"""Draw point clouds over each other, front and side, to check alignment (pose-env python, -I).

Usage: python overlay.py <out.png> <a.npz>:<key>[:scale] <b.npz>:<key>[:scale] ...
Each cloud gets its own colour (red, cyan, yellow, ...). All share one scale: 1 px = 2 mm.
"""
import sys

import cv2
import numpy as np

out = sys.argv[1]
COL = [(60, 60, 255), (255, 220, 0), (0, 230, 255), (120, 255, 120)]
clouds = []
for spec in sys.argv[2:]:
    path, key, *rest = spec.split(":") if spec[1] != ":" else [spec[:2] + spec[2:].split(":")[0]] + spec[2:].split(":")[1:]
    clouds.append(np.load(path)[key] * (float(rest[0]) if rest else 1.0))
S, H = 500, 1000  # px per m, image height
img = np.full((H, 1600, 3), 30, np.uint8)
for c, P in zip(COL, clouds):
    for ax, x0 in ((0, 450), (1, 1250)):  # front (x,z), side (y,z)
        u = (x0 + P[:, ax] * S).astype(int)
        v = (H - 20 - P[:, 2] * S).astype(int)
        ok = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < H)
        img[v[ok], u[ok]] = c
cv2.imwrite(out, img)
print("OVERLAY_OK", out)
