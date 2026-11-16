"""Measure eyes in an image: opening size, iris size and colour, pupil, whites and highlights.

Usage: measure_eyes.py <face_landmarker.task> <out_sheet.png> <label=image> [<label=image> ...]
The face tracker gives the outline of each eye opening and the iris's centre and edge. From
those: how wide and tall the opening is, the iris's size relative to the opening, the colour of
the iris ring (between pupil and edge), the pupil's size, how bright the whites are, and how much
of the eye is highlight. Values are printed side by side so a model's eyes can be compared with
the reference's. Sizes are shares of the eye opening's width, so image size doesn't matter.
"""
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

model, sheet = sys.argv[1], sys.argv[2]
images = [a.split("=", 1) for a in sys.argv[3:]]
landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model, delegate=BaseOptions.Delegate.CPU), running_mode=RunningMode.IMAGE, num_faces=1))
CONTOUR = {"R": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
           "L": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]}
IRIS = {"R": (468, [469, 470, 471, 472]), "L": (473, [474, 475, 476, 477])}

crops = []
for label, path in images:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:
        a = image[:, :, 3:] / 255.0
        image = (image[:, :, :3] * a + 200 * (1 - a)).astype(np.uint8)
    h, w = image.shape[:2]
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    if not result.face_landmarks:
        print(label, "no face found")
        continue
    p = np.array([[q.x * w, q.y * h] for q in result.face_landmarks[0]])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    line = []
    for side in "RL":
        outline = p[CONTOUR[side]]
        width = np.linalg.norm(p[CONTOUR[side][0]] - p[CONTOUR[side][8]])
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [outline.astype(np.int32)], 1)
        top, bottom = outline[:, 1].min(), outline[:, 1].max()
        centre = p[IRIS[side][0]]
        iris_r = float(np.mean(np.linalg.norm(p[IRIS[side][1]] - centre, axis=1)))
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.hypot(xx - centre[0], yy - centre[1])
        inside = mask > 0
        iris = inside & (dist < iris_r)
        sclera = inside & (dist > iris_r * 1.15)
        value = hsv[:, :, 2].astype(float)
        # The pupil: the darkest part of the iris, as a share of the iris radius.
        dark = iris & (value < np.percentile(value[iris], 30) if iris.any() else iris)
        pupil_r = float(np.sqrt(dark.sum() / np.pi)) if dark.any() else 0.0
        ring = iris & ~dark & (value < np.percentile(value[iris], 92))
        ring_colour = image[ring].mean(axis=0)[::-1] if ring.any() else np.zeros(3)
        line.append(f"{side}: open {width:5.1f}px tall/wide {(bottom - top) / width:.2f} iris/wide {2 * iris_r / width:.2f} "
                    f"pupil/iris {pupil_r / max(iris_r, 1e-6):.2f} iris RGB {tuple(int(v) for v in ring_colour)} "
                    f"white {int(value[sclera].mean()) if sclera.any() else 0} "
                    f"highlight {100 * (inside & (value > 235)).sum() / max(inside.sum(), 1):.1f}%")
    print(f"{label:10s}", " | ".join(line))
    box = p[CONTOUR["R"] + CONTOUR["L"]]
    x0, y0 = box.min(axis=0) - (box.max(axis=0) - box.min(axis=0)) * [0.1, 0.8]
    x1, y1 = box.max(axis=0) + (box.max(axis=0) - box.min(axis=0)) * [0.1, 0.8]
    crop = image[max(int(y0), 0):int(y1), max(int(x0), 0):int(x1)]
    crop = cv2.resize(crop, (600, int(600 * crop.shape[0] / crop.shape[1])))
    cv2.putText(crop, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    crops.append(crop)
height = max(c.shape[0] for c in crops)
cv2.imwrite(sheet, np.vstack([cv2.copyMakeBorder(c, 0, height - c.shape[0], 0, 0, cv2.BORDER_CONSTANT) for c in crops]))
print("EYES_MEASURED")
