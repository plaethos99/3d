"""Measure each eye opening directly from the image, without trusting the tracker's eyelid outline.

Usage: eye_opening_masks.py <views.json> <images_dir> <out_dir>

The face tracker learned human eyes, so on a cartoon face with large eyes it draws the eyelids
too close together, and sometimes on the wrong part of the face. The eye opening itself is easy
to see, though: around the iris, everything that isn't skin. For each eye this cuts out the
region that differs from the surrounding skin's colour, removes thin features (the lash line),
keeps the piece that contains the iris centre and fills its holes. An eye whose iris centre doesn't
land on a dark iris is skipped, since the tracker has lost it.
"""
import json
import os
import sys

import cv2
import numpy as np

views_path, images_dir, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
with open(views_path, encoding="utf-8") as f:
    records = json.load(f)
CONTOUR = {"R": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
           "L": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]}
IRIS = {"R": (468, [469, 470, 471, 472]), "L": (473, [474, 475, 476, 477])}

summary = {}
checks = []
for record in records:
    name = record["video"]
    image = cv2.imread(os.path.join(images_dir, name + ".png"))
    h, w = image.shape[:2]
    pts = np.array(record["points"])[:, :2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    full = np.zeros((h, w), np.uint8)
    summary[name] = {}
    for side in "RL":
        centre = pts[IRIS[side][0]]
        iris_r = float(np.mean(np.linalg.norm(pts[IRIS[side][1]] - centre, axis=1)))
        width = float(np.linalg.norm(pts[CONTOUR[side][0]] - pts[CONTOUR[side][8]]))
        reach = max(width, 4 * iris_r) * 0.95
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.hypot(xx - centre[0], yy - centre[1])
        # Is the iris centre really on an iris? It should be darker than the skin around it.
        core = dist < max(iris_r * 0.5, 2)
        ring = (dist > reach * 0.9) & (dist < reach * 1.3)
        if value[core].mean() > value[ring].mean() * 0.8:
            summary[name][side] = "skipped: iris centre not on a dark iris"
            continue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat = hsv[:, :, 1] / 255
        skin_sat = float(np.median(sat[ring]))
        skin_val = float(np.median(value[ring]))
        # The white of the eye: bright and nearly colourless, unlike skin.
        white = (sat < skin_sat * 0.6) & (value > skin_val * 0.7)
        # The iris and pupil: much more saturated or much darker than skin. The tracker's iris circle
        # is human-sized, too small for cartoon eyes, so it isn't used as a limit.
        iris = (sat > skin_sat + 0.15) | (value < skin_val * 0.55)
        region = ((white | iris) & (dist < reach)).astype(np.uint8)
        # Remove thin strokes (the lash line, creases) that would join the eye to its surroundings.
        thin = max(3, int(round(width * 0.09)) | 1)
        region = cv2.morphologyEx(region, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thin, thin)))
        count, labels = cv2.connectedComponents(region)
        label = labels[int(round(centre[1])), int(round(centre[0]))]
        if label == 0:
            summary[name][side] = "skipped: no opening found around the iris"
            continue
        opening = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(opening)
        cv2.drawContours(filled, contours, -1, 1, -1)
        ys, xs = np.nonzero(filled)
        summary[name][side] = {"tall_over_wide": round(float(np.ptp(ys) / max(np.ptp(xs), 1)), 2),
                               "tracker_tall_over_wide": round(float(np.ptp(pts[CONTOUR[side], 1]) / max(np.ptp(pts[CONTOUR[side], 0]), 1)), 2),
                               "area_vs_tracker": round(float(filled.sum() / max(cv2.contourArea(pts[CONTOUR[side]].astype(np.float32)), 1)), 2)}
        full |= filled
    cv2.imwrite(os.path.join(out_dir, f"eyes_{name}.png"), full * 255)
    check = image.copy()
    contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(check, contours, -1, (255, 80, 200), 1)
    for side in "RL":
        cv2.polylines(check, [pts[CONTOUR[side]].astype(np.int32)], True, (60, 255, 60), 1)
        iris_c = pts[IRIS[side][0]]
        cv2.circle(check, (int(iris_c[0]), int(iris_c[1])), int(np.mean(np.linalg.norm(pts[IRIS[side][1]] - iris_c, axis=1))), (255, 255, 0), 1)
    box = pts[CONTOUR["R"] + CONTOUR["L"]]
    x0, y0 = (box.min(0) - 30).astype(int)
    x1, y1 = (box.max(0) + 30).astype(int)
    checks.append(cv2.resize(check[max(y0, 0):y1, max(x0, 0):x1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    print("OPENING", name, summary[name])
width = max(c.shape[1] for c in checks)
cv2.imwrite(os.path.join(out_dir, "check.png"), np.vstack([cv2.copyMakeBorder(c, 0, 0, 0, width - c.shape[1], cv2.BORDER_CONSTANT) for c in checks]))
with open(os.path.join(out_dir, "openings.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=1)
print("OPENINGS_OK")
