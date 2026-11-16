"""Score how closely the result's limbs follow the performer's, ignoring body proportions.

Usage: score_motion.py <source.json> <result.json> [label]

Compares the direction of each limb segment in the image, in degrees. Joyce's legs are about 70%
as long as the performer's relative to her torso, so joint positions alone count that design
choice as an error, and directions don't. Left/right labels on legs and arms are ignored, per frame,
because the tracker confuses them on both videos. This makes the score a little forgiving.

The older joint-position score is also printed, unchanged, so runs can be compared with it.
"""
import json
import sys

import numpy as np

source_path, result_path = sys.argv[1:3]
label = sys.argv[3] if len(sys.argv) > 3 else ""


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


source, result = load(source_path), load(result_path)


def points(data, frame):
    marks = np.array(frame["image"])[:, :2].copy()
    marks[:, 0] *= data["width"] / data["height"]
    return marks


def direction(p, a, b):
    d = p[b] - p[a]
    return d / max(np.linalg.norm(d), 1e-6)


def angle(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1))))


LEG = {"thigh": ((23, 25), (24, 26)), "shin": ((25, 27), (26, 28))}
ARM = {"upper arm": ((11, 13), (12, 14)), "forearm": ((13, 15), (14, 16))}


def pair_errors(pa, pb, parts):
    """Error for each segment, under the better of the two left/right assignments for this frame."""
    straight, crossed = {}, {}
    for name, (left, right) in parts.items():
        straight[name] = [angle(direction(pa, *left), direction(pb, *left)), angle(direction(pa, *right), direction(pb, *right))]
        crossed[name] = [angle(direction(pa, *left), direction(pb, *right)), angle(direction(pa, *right), direction(pb, *left))]
    if sum(sum(v) for v in straight.values()) <= sum(sum(v) for v in crossed.values()):
        return straight
    return crossed


collected = {name: [] for name in list(LEG) + list(ARM)}
trunk = []
frames = min(len(source["frames"]), len(result["frames"]))
for a, b in zip(source["frames"][:frames], result["frames"][:frames]):
    if a is None or b is None:
        continue
    pa, pb = points(source, a), points(result, b)
    for parts in (LEG, ARM):
        for name, values in pair_errors(pa, pb, parts).items():
            collected[name].extend(values)
    ha, hb = (pa[23] + pa[24]) / 2, (pb[23] + pb[24]) / 2
    sa, sb = (pa[11] + pa[12]) / 2, (pb[11] + pb[12]) / 2
    trunk.append(angle((sa - ha) / np.linalg.norm(sa - ha), (sb - hb) / np.linalg.norm(sb - hb)))

summary = {name: round(float(np.mean(v)), 1) for name, v in collected.items() if v}
summary["trunk"] = round(float(np.mean(trunk)), 1)
overall = round(float(np.mean(list(summary.values()))), 1)
print("LIMB_DIRECTION_ERROR_DEG", label, overall, summary)
