"""Paint a crisp lash line along the upper lid margins into the skin texture.

Usage: paint_lash_line.py <bake_dir> <textures_dir> <eyes.json> <lash_rgb> <thickness_mm>

eyes.json holds each eyeball's centre and radius in metres: {"L": [[x, y, z], r], "R": ...}.
A texel belongs to the lash line when it sits on the upper lid right against the eyeball: within
thickness_mm of the eye's surface, in front, and above the eye's middle. It fades out toward the
corners, where real lashes thin. Painted characters often have this as a bold dark line.
"""
import json
import os
import sys

import cv2
import numpy as np

bake_dir, tex_dir, eyes_path, lash, thickness = sys.argv[1:6]
lash_bgr = np.array([float(v) for v in lash.split(",")][::-1], np.float32)
thickness = float(thickness) / 1000
FRONT = float(os.environ.get("LASH_FRONT", "0.55"))  # how far forward on the eye the line starts, as a share of its radius
position = np.load(os.path.join(bake_dir, "uv_position.npy")).astype(np.float32)
covered = np.load(os.path.join(bake_dir, "uv_covered.npy"))
with open(eyes_path, encoding="utf-8") as f:
    eyes = json.load(f)
colour = cv2.imread(os.path.join(tex_dir, "skin_colour.png")).astype(np.float32)

amount = np.zeros(position.shape[:2], np.float32)
for side, (centre, radius) in eyes.items():
    rel = position - np.array(centre, np.float32)
    dist = np.linalg.norm(rel, axis=2)
    gap = dist - radius
    near = np.clip(1.0 - gap / thickness, 0, 1) * (gap > -0.002)
    front = np.clip((-rel[:, :, 1] - FRONT * radius) / (0.15 * radius), 0, 1)
    above = np.clip((rel[:, :, 2] + 0.05 * radius) / (0.12 * radius), 0, 1)
    across = np.clip(1.0 - np.abs(rel[:, :, 0]) / (0.75 * radius), 0, 1)  # thins toward the corners
    amount = np.maximum(amount, near * front * above * np.sqrt(across))
amount = cv2.GaussianBlur(amount * covered, (0, 0), 1.0)
print("LASH_TEXELS", int((amount > 0.5).sum()))
result = colour * (1 - amount[:, :, None]) + lash_bgr * amount[:, :, None]
cv2.imwrite(os.path.join(tex_dir, "skin_colour.png"), np.clip(result, 0, 255).astype(np.uint8))
print("LASH_OK")
