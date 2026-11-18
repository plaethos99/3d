"""Measure skin colour on a face (both cheeks and the forehead, found by MediaPipe) in one or more images.

Usage: python -I measure_skin.py <face_landmarker.task> <image> [<image> ...]
Prints per image: SKIN <image> R G B (sRGB 0-255, mean of three patches each about a sixth of the eye distance
across) and HAIR <image> R G B (median of patches just outside both temples), then SKIN_MEASURED.
"""
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=sys.argv[1], delegate=BaseOptions.Delegate.CPU),
    running_mode=RunningMode.IMAGE, num_faces=1))
for path in sys.argv[2:]:
    im = cv2.imread(path)
    h, w = im.shape[:2]
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    if not r.face_landmarks:
        print("SKIN", path, "no face")
        continue
    p = np.array([[q.x * w, q.y * h] for q in r.face_landmarks[0]])
    rad = max(2, int(np.linalg.norm(p[33] - p[263]) * 0.08))
    vals = []
    for i in (50, 280, 151):  # left cheek, right cheek, forehead
        x, y = p[i].astype(int)
        vals.append(im[y - rad:y + rad, x - rad:x + rad].reshape(-1, 3).mean(0)[::-1])
    print("SKIN", path, *np.mean(vals, 0).round(1).tolist())
    # hair: just outside both temples (the face outline's points 127 and 356), pushed out from the face's centre
    eye_d = np.linalg.norm(p[33] - p[263])
    centre = p[168]
    hair = []
    for i in (127, 356):
        d = p[i] - centre
        q = (p[i] + d / np.linalg.norm(d) * eye_d * 0.3).astype(int)
        patch = im[max(q[1] - rad, 0):q[1] + rad, max(q[0] - rad, 0):q[0] + rad].reshape(-1, 3)
        if len(patch):
            hair.append(np.median(patch, 0)[::-1])
    if hair:
        print("HAIR", path, *np.mean(hair, 0).round(1).tolist())
print("SKIN_MEASURED")
