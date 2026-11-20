"""FaceBuilder's baked texture -> the three images face_bake.py reads (pose-env python, -I).

Usage: python fb_textures.py <fb_texture.png> <textures_dir>
skin_colour.png (the texture), photo_coverage.png (where it has photo), hair_mask.png (dark, grey texels: hair
the photos put on the head). Prints FB_TEXTURES_OK.
"""
import os
import sys

import cv2
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
tex = cv2.imread(src, cv2.IMREAD_UNCHANGED)
rgb = tex[..., :3]
seen = (tex[..., 3] > 128) if tex.shape[2] == 4 else (rgb.max(2) > 8)
seen = cv2.erode(seen.astype(np.uint8), np.ones((5, 5), np.uint8))
cv2.imwrite(os.path.join(dst, "skin_colour.png"), rgb)
cv2.imwrite(os.path.join(dst, "photo_coverage.png"), seen * 255)
f = rgb.astype(np.float32) / 255
lum = f @ np.array([0.114, 0.587, 0.299])
sat = (f.max(2) - f.min(2)) / np.maximum(f.max(2), 1e-3)
skin = (seen > 0) & (lum > np.percentile(lum[seen > 0], 40))
hair = (seen > 0) & (lum < 0.55 * np.median(lum[skin])) & (sat < 0.9 * np.median(sat[skin]))
hair = cv2.morphologyEx(hair.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
cv2.imwrite(os.path.join(dst, "hair_mask.png"), hair * 255)
print("FB_TEXTURES_OK photo %.0f%%, hair %.0f%% of the layout" % (seen.mean() * 100, hair.mean() * 100))
