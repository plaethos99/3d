"""Confirm the pose model loads, finds a body, and report where it sees the joints.

Usage: pose_smoke_test.py <model.task> <image.png> [<image.png> ...]
"""
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

model, images = sys.argv[1], sys.argv[2:]
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model, delegate=BaseOptions.Delegate.CPU),
    running_mode=RunningMode.IMAGE,
    num_poses=1,
)
NAMES = {0: "nose", 11: "l_shoulder", 12: "r_shoulder", 15: "l_wrist", 16: "r_wrist",
         23: "l_hip", 24: "r_hip", 27: "l_ankle", 28: "r_ankle"}

with PoseLandmarker.create_from_options(options) as landmarker:
    for path in images:
        bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if bgra.shape[2] == 4:
            # Put transparent references onto a light background, as a camera would see.
            alpha = bgra[..., 3:4] / 255.0
            bgr = (bgra[..., :3] * alpha + 225 * (1 - alpha)).astype("uint8")
        else:
            bgr = bgra
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.pose_landmarks:
            print(path.split("\\")[-1], "NO_POSE_FOUND")
            continue
        marks = result.pose_landmarks[0]
        seen = sum(m.visibility for m in marks) / len(marks)
        height, width = bgr.shape[:2]
        points = {name: (round(marks[i].x * width), round(marks[i].y * height)) for i, name in NAMES.items()}
        print(path.split("\\")[-1], "POSE_FOUND visibility", round(seen, 2), points)
print("SMOKE_OK")
