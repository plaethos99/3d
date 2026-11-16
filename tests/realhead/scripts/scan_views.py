"""Find the best views of a face in videos: head angle, sharpness and landmarks for every frame.

Usage: scan_views.py <face_landmarker.task> <out_dir> <video> [<video> ...]
Writes views.json (every frame's angles and sharpness), picks the sharpest frame in each angle
band, saves those frames as PNGs and draws a sheet with the 478 tracked points on each.
"""
import json
import math
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

model, out_dir = sys.argv[1:3]
videos = sys.argv[3:]
os.makedirs(out_dir, exist_ok=True)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model, delegate=BaseOptions.Delegate.CPU),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)


def angles(matrix):
    """Yaw (turn), pitch (nod) and roll in degrees from the face's transform."""
    r = np.array(matrix)[:3, :3]
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


records = []
for video in videos:
    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    name = os.path.splitext(os.path.basename(video))[0]
    with FaceLandmarker.create_from_options(options) as landmarker:
        index = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(index * 1000 / fps))
            if result.face_landmarks:
                h, w = bgr.shape[:2]
                points = np.array([[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]])
                x0, y0 = points[:, :2].min(axis=0).astype(int)
                x1, y1 = points[:, :2].max(axis=0).astype(int)
                face = cv2.cvtColor(bgr[max(y0, 0):y1, max(x0, 0):x1], cv2.COLOR_BGR2GRAY)
                sharp = float(cv2.Laplacian(face, cv2.CV_64F).var()) if face.size else 0.0
                yaw, pitch, roll = angles(result.facial_transformation_matrixes[0])
                shapes = {c.category_name: round(c.score, 3) for c in result.face_blendshapes[0]}
                # How far from a neutral face: the strongest expression shape, ignoring blinks and gaze.
                expression = max(v for k, v in shapes.items() if not k.startswith(("eyeLook", "_neutral")))
                records.append({
                    "video": name, "frame": index, "yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1),
                    "sharpness": round(sharp, 1), "face_px": int(y1 - y0), "expression": round(expression, 3),
                    "blink": round((shapes.get("eyeBlinkLeft", 0) + shapes.get("eyeBlinkRight", 0)) / 2, 3),
                    "points": points.round(2).tolist(),
                })
            index += 1
    capture.release()
    print("SCANNED", name, "frames", index, "faces", sum(1 for r in records if r["video"] == name))

with open(os.path.join(out_dir, "views.json"), "w", encoding="utf-8") as handle:
    json.dump(records, handle)

yaws = np.array([r["yaw"] for r in records])
print("YAW_RANGE", round(float(yaws.min()), 1), round(float(yaws.max()), 1))
for name in sorted(set(r["video"] for r in records)):
    part = [r for r in records if r["video"] == name]
    print(name, "yaw by second:", [round(float(np.mean([r["yaw"] for r in part[k:k + 25]])), 0) for k in range(0, len(part), 25)])

# Best frame per angle band: sharp, eyes open, close to a neutral face.
bands = [(-95, -60), (-60, -35), (-35, -15), (-8, 8), (15, 35), (35, 60), (60, 95)]
picks = []
for lo, hi in bands:
    pool = [r for r in records if lo <= r["yaw"] <= hi and r["blink"] < 0.4]
    if not pool:
        print(f"BAND {lo}..{hi}: none")
        continue
    best = max(pool, key=lambda r: r["sharpness"] * (1.0 - min(r["expression"], 0.9)))
    picks.append(best)
    print(f"BAND {lo}..{hi}: {best['video']} frame {best['frame']} yaw {best['yaw']} pitch {best['pitch']} roll {best['roll']} "
          f"sharp {best['sharpness']} expression {best['expression']} face {best['face_px']}px  ({len(pool)} frames in band)")

tiles = []
for pick in picks:
    video = next(v for v in videos if os.path.splitext(os.path.basename(v))[0] == pick["video"])
    capture = cv2.VideoCapture(video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, pick["frame"])
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        continue
    tag = f"yaw{int(round(pick['yaw'])):+d}"
    cv2.imwrite(os.path.join(out_dir, f"view_{tag}.png"), bgr)
    marked = bgr.copy()
    for x, y, _ in pick["points"]:
        cv2.circle(marked, (int(x), int(y)), 2, (60, 255, 60), -1)
    pts = np.array(pick["points"])[:, :2]
    cx, cy = pts.mean(axis=0)
    half = int(pick["face_px"] * 0.9)
    crop = marked[max(int(cy) - half, 0):int(cy) + half, max(int(cx) - half, 0):int(cx) + half]
    crop = cv2.resize(crop, (320, 320))
    cv2.putText(crop, f"{tag} {pick['video'][:7]}#{pick['frame']}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tiles.append(crop)
if tiles:
    cv2.imwrite(os.path.join(out_dir, "views_sheet.png"), np.hstack(tiles))
print("SCAN_OK picks", len(picks))
