"""Track one person through a video and save their body landmarks as JSON.

Usage: track_video.py <model.task> <video> <out.json> [max_seconds]
Runs in the pose-env environment, not inside Blender.
"""
import json
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

model, video, out = sys.argv[1:4]
max_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0

capture = cv2.VideoCapture(video)
if not capture.isOpened():
    raise SystemExit(f"Could not open video: {video}")
fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model, delegate=BaseOptions.Delegate.CPU),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

frames = []
missing = 0
with PoseLandmarker.create_from_options(options) as landmarker:
    index = 0
    while index < max_seconds * fps:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, int(index * 1000.0 / fps))
        if result.pose_world_landmarks:
            frames.append({
                "world": [[m.x, m.y, m.z, m.visibility] for m in result.pose_world_landmarks[0]],
                "image": [[m.x, m.y, m.visibility] for m in result.pose_landmarks[0]],
            })
        else:
            frames.append(None)
            missing += 1
        index += 1
capture.release()

with open(out, "w", encoding="utf-8") as handle:
    json.dump({"fps": fps, "width": width, "height": height, "frames": frames}, handle)
print("TRACK_OK frames", len(frames), "missing", missing, "fps", round(fps, 2), "size", width, height)
