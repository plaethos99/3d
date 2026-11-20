"""MediaPipe's 478 face landmarks of pictures, in pixels (for Blender scripts, whose Python has no MediaPipe).

Usage: tools/pose-env/Scripts/python.exe -I pipeline/daz/landmarks2d.py <out.json> <image> [<image> ...]
Writes {image: [[x, y], ...] or null, "_colours": {image: {"skin": rgb, "hair": rgb}}}; a transparent picture is put
on light grey first. Prints LANDMARKS2D_OK.
"""
import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "tools", "face-model", "face_landmarker.task")
lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL),
                                                              running_mode=RunningMode.IMAGE, num_faces=1))
out, colours = {}, {}
for path in sys.argv[2:]:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4:
        a = im[..., 3:] / 255.0
        im = (im[..., :3] * a + 225 * (1 - a)).astype(np.uint8)
    h, w = im.shape[:2]
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    out[path] = [[q.x * w, q.y * h] for q in r.face_landmarks[0]] if r.face_landmarks else None
    if out[path] is not None:  # colour patches (sRGB 0-255, median): cheeks (skin) and the hair above the forehead
        p = np.array(out[path])
        rad = max(2, int(np.linalg.norm(p[33] - p[263]) * 0.07))

        def patch(pt, r_=rad):
            x, y = np.clip(pt.astype(int), r_, [w - r_ - 1, h - r_ - 1])
            return im[y - r_:y + r_, x - r_:x + r_].reshape(-1, 3)[:, ::-1].astype(float)

        up = p[10] - p[152]  # chin -> forehead top
        hair = [p[10] + up * f + (p[263] - p[33]) * g for f in (0.18, 0.26) for g in (-0.25, 0, 0.25)]
        colours[path] = {"skin": np.median(np.vstack([patch(p[i]) for i in (50, 280, 205, 425)]), 0).tolist(),
                         "hair": np.median(np.vstack([patch(q_, max(2, rad // 2)) for q_ in hair]), 0).tolist()}
json.dump({**out, "_colours": colours}, open(sys.argv[1], "w"))
print("LANDMARKS2D_OK", sum(v is not None for v in out.values()), "of", len(out))
