"""Cut the subject's head outline out of each photo, for fitting ears and hair to.

Usage: photo_masks.py <cameras.json> <views.json> <clips_dir> <out_dir> <scale>
For each camera's frame: a mask of the head (not backdrop, not hoodie), and the chin's height,
above which the outline belongs to head, ears and hair. Below it the hoodie hides the shape.
"""
import json
import os
import sys

import cv2
import numpy as np

cams_path, views_path, clips, out_dir, scale = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
os.makedirs(out_dir, exist_ok=True)
with open(cams_path, encoding="utf-8") as f:
    cams = json.load(f)
with open(views_path, encoding="utf-8") as f:
    records = json.load(f)

info = {}
for key, cam in cams.items():
    capture = cv2.VideoCapture(os.path.join(clips, cam["video"] + ".mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, cam["frame"])
    ok, bgr = capture.read()
    capture.release()
    bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue, sat, val = hsv[:, :, 0] * 2, hsv[:, :, 1] / 255, hsv[:, :, 2] / 255
    backdrop = ((hue > 280) | (hue < 8)) & (sat > 0.15) & (val > 0.5)
    hoodie = (sat < 0.18) & (val > 0.45)
    head = (~backdrop & ~hoodie).astype(np.uint8)
    # Keep the largest piece, and close small gaps such as highlights on the skin.
    head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(head)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    head = (labels == biggest).astype(np.uint8)
    holes = cv2.floodFill(head.copy() * 255, None, (0, 0), 128)[1]
    head = np.where(holes == 0, 1, head).astype(np.uint8)
    record = next(r for r in records if r["video"] == cam["video"] and r["frame"] == cam["frame"])
    chin = float(np.array(record["points"])[152][1] * scale)
    cv2.imwrite(os.path.join(out_dir, f"mask_{key}.png"), head * 255)
    preview = bgr.copy()
    preview[head == 0] = (preview[head == 0] * 0.3).astype(np.uint8)
    cv2.line(preview, (0, int(chin)), (preview.shape[1], int(chin)), (60, 255, 60), 2)
    cv2.imwrite(os.path.join(out_dir, f"check_{key}.png"), preview)
    info[key] = {"chin_row": chin, "yaw": record["yaw"]}
    print("MASK", key, "yaw", record["yaw"], "head pixels above chin", int(head[: int(chin)].sum()))
with open(os.path.join(out_dir, "masks.json"), "w", encoding="utf-8") as handle:
    json.dump(info, handle)
print("MASKS_OK")
