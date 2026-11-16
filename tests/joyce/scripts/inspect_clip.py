"""Check whether a clip suits motion capture, and draw what the tracker sees on it.

Usage: inspect_clip.py <video> <pose.json> <sheet.png>
Reports how much of the clip has the body in frame, whether the camera moves, and how
confident the tracker is about each body part.
"""
import json
import sys

import cv2
import numpy as np

video, pose_json, out = sys.argv[1:4]
with open(pose_json, encoding="utf-8") as f:
    data = json.load(f)
frames = data["frames"]
count = len(frames)
tracked = [f for f in frames if f is not None]
print("FRAMES", count, "TRACKED", len(tracked))

points = np.array([f["image"] for f in tracked])
in_frame = (points[:, :, 0] > 0.01) & (points[:, :, 0] < 0.99) & (points[:, :, 1] > 0.01) & (points[:, :, 1] < 0.99)
groups = {
    "head": [0, 7, 8], "shoulders": [11, 12], "elbows": [13, 14], "wrists": [15, 16],
    "hips": [23, 24], "knees": [25, 26], "ankles": [27, 28], "toes": [31, 32],
}
for name, ids in groups.items():
    print(f"{name:10s} in frame {in_frame[:, ids].mean() * 100:5.1f}%  visibility {points[:, ids, 2].mean():.2f}")
full = in_frame[:, [0, 11, 12, 23, 24, 27, 28]].all(axis=1)
print("WHOLE_BODY_IN_FRAME", round(float(full.mean()) * 100, 1), "%")
hips = (points[:, 23, :2] + points[:, 24, :2]) / 2
print("HIP_X first/last", round(float(hips[0, 0]), 3), round(float(hips[-1, 0]), 3), "HIP_Y first/last", round(float(hips[0, 1]), 3), round(float(hips[-1, 1]), 3))

capture = cv2.VideoCapture(video)
previous = None
shifts = []
tiles = []
index = 0
pairs = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 24), (11, 23), (12, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]
while True:
    ok, frame = capture.read()
    if not ok:
        break
    grey = cv2.cvtColor(cv2.resize(frame, (240, 426)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Measure camera shift from the top of the frame, away from a walking person's legs.
    region = grey[:90, :]
    if previous is not None:
        (dx, dy), _ = cv2.phaseCorrelate(previous, region)
        shifts.append((dx, dy))
    previous = region
    if index % max(count // 8, 1) == 0 and len(tiles) < 8 and frames[index] is not None:
        height, width = frame.shape[:2]
        marks = np.array(frames[index]["image"])
        for a, b in pairs:
            pa = (int(marks[a, 0] * width), int(marks[a, 1] * height))
            pb = (int(marks[b, 0] * width), int(marks[b, 1] * height))
            colour = (40, 220, 40) if a % 2 else (40, 120, 255)
            cv2.line(frame, pa, pb, colour, 4)
        tiles.append(cv2.resize(frame, (300, int(300 * height / width))))
    index += 1
capture.release()
shifts = np.array(shifts)
print("CAMERA_DRIFT_PERCENT_OF_WIDTH x", round(float(shifts[:, 0].sum()) / 240 * 100, 1), "y", round(float(shifts[:, 1].sum()) / 426 * 100, 1))
print("CAMERA_JITTER_PX_AT_240W", round(float(np.abs(shifts).mean()), 3))
if tiles:
    cv2.imwrite(out, np.hstack(tiles))
print("INSPECT_OK")
