"""Measure an expression in an image: mouth opening, brow lift and eye opening, in eye-widths.

Usage: measure_face.py <face_landmarker.task> <label=image[@x0,y0,x1,y1]> [...]
The face is cropped (optional box), enlarged so the tracker finds it, and tracked. Sizes are divided
by the distance between the outer eye corners, so images of different size compare directly.
  mouth_h, mouth_w: the dark opening of the mouth, found as dark pixels between nose and chin
  brow:             brow midpoints above the iris centres
"""
import json
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

model = sys.argv[1]
tracker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model, delegate=BaseOptions.Delegate.CPU), running_mode=RunningMode.IMAGE, num_faces=1))
results = {}
for arg in sys.argv[2:]:
    label, rest = arg.split("=", 1)
    path, box = (rest.split("@") + [None])[:2]
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:
        a = image[:, :, 3:] / 255.0
        image = (image[:, :, :3] * a + 220 * (1 - a)).astype(np.uint8)
    if box:
        bx0, by0, bx1, by1 = (int(v) for v in box.split(","))
        image = image[by0:by1, bx0:bx1]
    image = cv2.resize(image, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[:2]
    found = tracker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    if not found.face_landmarks:
        print(label, "no face")
        continue
    p = np.array([[q.x * w, q.y * h] for q in found.face_landmarks[0]])
    eye = float(np.linalg.norm(p[33] - p[263]))
    # Mouth opening: dark pixels in the box spanned by the mouth corners, nose and chin.
    xa, xb = sorted((p[61, 0], p[291, 0]))
    margin = (xb - xa) * 0.4
    ya, yb = p[2, 1], p[152, 1]
    region = image[int(ya):int(yb), int(xa - margin):int(xb + margin)]
    value = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)[:, :, 2]
    dark = (value < 70).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    mouth_h = mouth_w = 0.0
    if count > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mouth_h, mouth_w = stats[k, cv2.CC_STAT_HEIGHT] / eye, stats[k, cv2.CC_STAT_WIDTH] / eye
    brow = float(((p[468, 1] - p[105, 1]) + (p[473, 1] - p[334, 1])) / 2 / eye)
    results[label] = {"mouth_h": round(mouth_h, 3), "mouth_w": round(mouth_w, 3), "brow": round(brow, 3)}
    print("FACE", label, results[label])
print("MEASURES", json.dumps(results))
