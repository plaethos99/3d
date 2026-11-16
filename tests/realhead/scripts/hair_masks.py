"""Cut light-coloured (grey, white, blond) hair out of reference images.

Usage: hair_masks.py <out_dir> <name=image> [...]
Hair here is bright and nearly colourless. Of such pixels, the connected piece that reaches
highest in the image is the hair (lower pieces are clothing such as a blouse).
"""
import os
import sys

import cv2
import numpy as np

out_dir = sys.argv[1]
SAT = float(os.environ.get("HAIR_MAX_SAT", "0.10"))  # skin is warmer than this, grey hair is not
VAL = float(os.environ.get("HAIR_MIN_VAL", "0.28"))  # shadowed waves are still this bright
os.makedirs(out_dir, exist_ok=True)
sheets = []
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1)
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    alpha = image[:, :, 3] > 128 if image.shape[2] == 4 else np.ones(image.shape[:2], bool)
    bgr = image[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    candidate = alpha & (hsv[:, :, 1] < SAT * 255) & (hsv[:, :, 2] > VAL * 255)
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    big = [k for k in range(1, count) if stats[k, cv2.CC_STAT_AREA] > 500]
    top = min(big, key=lambda k: stats[k, cv2.CC_STAT_TOP])
    hair = (labels == top).astype(np.uint8)
    hair = cv2.morphologyEx(hair, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cv2.imwrite(os.path.join(out_dir, f"hair_{name}.png"), hair * 255)
    check = bgr.copy()
    contours, _ = cv2.findContours(hair, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(check, contours, -1, (255, 0, 255), 2)
    ys, xs = np.nonzero(hair)
    sheets.append(cv2.resize(check[max(ys.min() - 40, 0):ys.max() + 120, max(xs.min() - 60, 0):xs.max() + 60], (360, 400)))
    print("HAIR", name, "pixels", int(hair.sum()), "rows", int(ys.min()), "-", int(ys.max()))
cv2.imwrite(os.path.join(out_dir, "check.png"), np.hstack(sheets))
print("HAIR_MASKS_OK")
