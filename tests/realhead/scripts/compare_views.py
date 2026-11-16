"""Compare renders with the video frames they copy: overlays, side by side, and tracked-point error.

Usage: compare_views.py <face_landmarker.task> <view_cameras.json> <renders_dir> <clips_dir> <out.png>
Point error is measured by running the same face tracker on the render and on the photo, as a
share of the distance between the outer eye corners, so it doesn't depend on image size.
"""
import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

model, cams_path, renders, clips, out = sys.argv[1:6]
with open(cams_path, encoding="utf-8") as f:
    views = json.load(f)
landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model, delegate=BaseOptions.Delegate.CPU), running_mode=RunningMode.IMAGE, num_faces=1))


def points(bgr):
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    if not result.face_landmarks:
        return None
    h, w = bgr.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]])


REGIONS = {
    "eyes": [33, 133, 159, 145, 263, 362, 386, 374], "brows": [70, 105, 107, 300, 334, 336],
    "nose": [1, 4, 5, 48, 278, 98, 327, 168], "mouth": [61, 291, 0, 17, 13, 14, 40, 270],
    "jaw and chin": [152, 148, 377, 176, 400, 172, 397, 136, 365], "face outline": [234, 454, 93, 323, 132, 361, 10],
}
rows = []
scores = {}
for key, view in views.items():
    render = cv2.imread(os.path.join(renders, f"render_{key}.png"))
    still = os.path.join(clips, view["video"] + ".png")
    if os.path.exists(still):
        photo = cv2.imread(still)  # a still reference image rather than a video frame
    else:
        capture = cv2.VideoCapture(os.path.join(clips, view["video"] + ".mp4"))
        capture.set(cv2.CAP_PROP_POS_FRAMES, view["frame"])
        ok, photo = capture.read()
        capture.release()
    photo = cv2.resize(photo, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_AREA)
    pa, pb = points(photo), points(render)
    if pa is not None and pb is not None:
        eye = np.linalg.norm(pa[33] - pa[263])
        per = {name: round(float(np.linalg.norm(pa[ids] - pb[ids], axis=1).mean() / eye * 100), 1) for name, ids in REGIONS.items()}
        per["all"] = round(float(np.linalg.norm(pa[:468] - pb[:468], axis=1).mean() / eye * 100), 1)
        scores[key] = per
        print("POINT_ERROR_PERCENT_OF_EYE_WIDTH", key, per)
    else:
        print("POINT_ERROR", key, "face not found in", "photo" if pa is None else "render")
    # Crop around the face in the photo.
    centre = pa[:468].mean(axis=0) if pa is not None else np.array([render.shape[1] / 2, render.shape[0] / 2])
    half = int((np.ptp(pa[:468, 1]) if pa is not None else 300) * 0.85)
    x0, y0 = int(max(centre[0] - half, 0)), int(max(centre[1] - half * 1.05, 0))
    crop = lambda im: cv2.resize(im[y0:y0 + 2 * half, x0:x0 + 2 * half], (380, 380))
    blend = cv2.addWeighted(photo, 0.5, render, 0.5, 0)
    edges = photo.copy()
    edges[cv2.Canny(cv2.cvtColor(render, cv2.COLOR_BGR2GRAY), 40, 110) > 0] = (60, 255, 60)
    tiles = [crop(photo), crop(render), crop(blend), crop(edges)]
    for tile, label in zip(tiles, ("photo", "render", "50/50 mix", "render edges")):
        cv2.putText(tile, f"{label} {key}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    rows.append(np.hstack(tiles))
cv2.imwrite(out, np.vstack(rows))
with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as handle:
    json.dump(scores, handle)
print("COMPARE_OK")
