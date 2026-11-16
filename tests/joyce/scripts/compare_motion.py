"""Compare the source performance with the animated result, using the tracker on both.

Usage: compare_motion.py <source.json> <result.json> <source.mp4> <result.mp4> <side_by_side.mp4> <sheet.png>

Poses are compared in the image, with each body centred on its hips and scaled to its torso,
so differences in camera distance and body proportions do not count as errors. The result is
a percentage of torso length: how far her joints sit from where the performer's joints sit.
"""
import json
import sys

import cv2
import numpy as np

source_json, result_json, source_video, result_video, out_video, out_sheet = sys.argv[1:7]

with open(source_json, encoding="utf-8") as f:
    source = json.load(f)
with open(result_json, encoding="utf-8") as f:
    result = json.load(f)

JOINTS = {"shoulders": (11, 12), "elbows": (13, 14), "wrists": (15, 16), "knees": (25, 26), "ankles": (27, 28)}


def normalized(frame):
    points = np.array(frame["image"])[:, :2]
    hips = (points[23] + points[24]) / 2
    shoulders = (points[11] + points[12]) / 2
    return (points - hips) / max(np.linalg.norm(shoulders - hips), 1e-6)


def aspect(data):
    return np.array((data["width"] / data["height"], 1.0))


errors = {name: [] for name in JOINTS}
count = min(len(source["frames"]), len(result["frames"]))
for a, b in zip(source["frames"][:count], result["frames"][:count]):
    if a is None or b is None:
        continue
    pa = normalized(a) * aspect(source)
    pb = normalized(b) * aspect(result)
    # Rescale after applying aspect, so both torsos are exactly one unit long.
    pa /= max(np.linalg.norm((pa[11] + pa[12]) / 2 - (pa[23] + pa[24]) / 2), 1e-6)
    pb /= max(np.linalg.norm((pb[11] + pb[12]) / 2 - (pb[23] + pb[24]) / 2), 1e-6)
    for name, ids in JOINTS.items():
        errors[name].extend(np.linalg.norm(pa[list(ids)] - pb[list(ids)], axis=1))

summary = {k: round(float(np.mean(v)) * 100, 1) for k, v in errors.items() if v}
overall = round(float(np.mean(list(summary.values()))), 1) if summary else None
print("POSE_ERROR_PERCENT_OF_TORSO", overall, summary)

# Side-by-side video, with the performer on the left and Joyce on the right.
sources = cv2.VideoCapture(source_video)
results = cv2.VideoCapture(result_video)
height = 720
fps = source["fps"]
writer = None
samples = []
index = 0
while index < count:
    ok_a, a = sources.read()
    ok_b, b = results.read()
    if not ok_a or not ok_b:
        break
    a = cv2.resize(a, (int(a.shape[1] * height / a.shape[0]), height))
    b = cv2.resize(b, (int(b.shape[1] * height / b.shape[0]), height))
    joined = np.hstack([a, b])
    if writer is None:
        size = (joined.shape[1], joined.shape[0])
        writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    writer.write(joined)
    if index % max(count // 6, 1) == 0 and len(samples) < 6:
        samples.append(cv2.resize(joined, (joined.shape[1] // 2, joined.shape[0] // 2)))
    index += 1
if writer is not None:
    writer.release()
rows = [np.hstack(samples[i:i + 3]) for i in range(0, len(samples), 3) if len(samples[i:i + 3]) == 3]
if rows:
    cv2.imwrite(out_sheet, np.vstack(rows))
print("COMPARE_OK frames", index)
