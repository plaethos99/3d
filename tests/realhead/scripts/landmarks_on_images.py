"""Run the face tracker on still images and save the 478 points for each.

Usage: landmarks_on_images.py <face_landmarker.task> <out.json> <image> [<image> ...]
"""
import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

model, out = sys.argv[1:3]
images = sys.argv[3:]
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model, delegate=BaseOptions.Delegate.CPU),
    running_mode=RunningMode.IMAGE, num_faces=1, output_facial_transformation_matrixes=True,
)
found = {}
with FaceLandmarker.create_from_options(options) as landmarker:
    for path in images:
        bgr = cv2.imread(path)
        h, w = bgr.shape[:2]
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        key = os.path.splitext(os.path.basename(path))[0]
        if not result.face_landmarks:
            print("NO_FACE", key)
            continue
        points = np.array([[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]])
        found[key] = {"points": points.round(3).tolist(), "matrix": np.array(result.facial_transformation_matrixes[0]).tolist(),
                      "size": [w, h]}
        marked = bgr.copy()
        for x, y, _ in points:
            cv2.circle(marked, (int(x), int(y)), 2, (40, 220, 40), -1)
        cv2.imwrite(os.path.join(os.path.dirname(out), f"marked_{key}.png"), marked)
        print("FACE", key)
with open(out, "w", encoding="utf-8") as handle:
    json.dump(found, handle)
print("LANDMARKS_OK", len(found), "of", len(images))
