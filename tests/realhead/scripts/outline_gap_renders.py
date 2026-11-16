"""Average gap between the head outline in photos and in renders, cut out the same way from both.

Usage: outline_gap_renders.py <view_cameras.json> <views.json> <clips_dir> <label=renders_dir> [...]
Both are cut from the same pink backdrop, so their outlines include the hair's fuzzy edge.
"""
import json
import os
import sys

import cv2
import numpy as np

cams_path, views_path, clips = sys.argv[1:4]
runs = [a.split("=", 1) for a in sys.argv[4:]]
with open(cams_path, encoding="utf-8") as f:
    cams = json.load(f)
with open(views_path, encoding="utf-8") as f:
    records = json.load(f)


def head_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue, sat, val = hsv[:, :, 0] * 2, hsv[:, :, 1] / 255, hsv[:, :, 2] / 255
    backdrop = ((hue > 280) | (hue < 8)) & (sat > 0.15) & (val > 0.5)
    hoodie = (sat < 0.18) & (val > 0.45)
    head = (~backdrop & ~hoodie).astype(np.uint8)
    head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(head)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


for label, folder in runs:
    gaps = {}
    for key, view in cams.items():
        render = cv2.imread(os.path.join(folder, f"render_{key}.png"))
        capture = cv2.VideoCapture(os.path.join(clips, view["video"] + ".mp4"))
        capture.set(cv2.CAP_PROP_POS_FRAMES, view["frame"])
        ok, photo = capture.read()
        capture.release()
        photo = cv2.resize(photo, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_AREA)
        record = next(r for r in records if r["video"] == view["video"] and r["frame"] == view["frame"])
        rows = int(np.array(record["points"])[152][1] * render.shape[0] / view["size"][1])
        a, b = head_mask(photo)[:rows], head_mask(render)[:rows]
        edge = a ^ np.roll(a, 1, 0) | a ^ np.roll(a, 1, 1)
        gaps[key] = round(float((a ^ b).sum() / max(edge.sum(), 1)), 2)
    print("OUTLINE_GAP_PX", label, round(float(np.mean(list(gaps.values()))), 2), gaps)
