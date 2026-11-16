"""Compare detected foot contact with the known walk's real contact, at every possible timing.

Usage: score_contacts.py <feet_truth.json> <report.json>
Detected motion may start at any point in the cycle relative to the truth only if the video did,
so timing is tested as frame-for-frame first, then the best-fitting offset is reported too.
"""
import json
import sys

import numpy as np

with open(sys.argv[1], encoding="utf-8") as handle:
    truth = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    report = json.load(handle)

period = len(truth["L"]["planted"])
step = np.array(truth["L"]["step_mm"])
planted = np.array(truth["L"]["planted"], dtype=bool)
print("TRUE_STANCE_FRAMES", int(planted.sum()), "of", period, "TRUE_BODY_SPEED_MM_PER_FRAME", round(float(np.median(step[planted])), 2))
print("TRUE_SPEED_M_S", round(float(np.median(step[planted])) * 24 / 1000, 3), "DETECTED_SPEED_M_S", report["speed_m_per_s"])

for side in "LR":
    detected = np.array(report[f"contact_{side}"], dtype=bool)
    expected = np.array(truth[side]["planted"], dtype=bool)
    count = len(detected)
    exact = float(np.mean(detected == expected[np.arange(count) % period]))
    scores = [float(np.mean(detected == expected[(np.arange(count) + k) % period])) for k in range(period)]
    best = int(np.argmax(scores))
    print(f"CONTACT_{side} agreement {exact * 100:.0f}%  best offset {best} frames: {scores[best] * 100:.0f}%  detected share {detected.mean():.2f}  true share {expected.mean():.2f}")
    print("  detected", "".join("#" if v else "." for v in detected))
    print("  true    ", "".join("#" if expected[i % period] else "." for i in range(count)))
