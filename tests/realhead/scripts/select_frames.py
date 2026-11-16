"""Pick frames for texturing: the sharpest, most neutral frame in each band of head angle.

Usage: select_frames.py <views.json> <step_deg> <out.txt>
Writes one "video:frame" per line.
"""
import json
import sys

views_path, step, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
with open(views_path, encoding="utf-8") as f:
    records = [r for r in json.load(f) if r["blink"] < 0.3]
picked = []
low = min(r["yaw"] for r in records)
high = max(r["yaw"] for r in records)
band = low
while band <= high:
    pool = [r for r in records if band <= r["yaw"] < band + step]
    if pool:
        best = max(pool, key=lambda r: r["sharpness"] * (1 - min(r["expression"], 0.9)) ** 2)
        picked.append(f"{best['video'].split('-')[0]}:{best['frame']}")
        print("PICK", picked[-1], "yaw", best["yaw"], "sharp", best["sharpness"], "expression", best["expression"])
    band += step
with open(out, "w", encoding="utf-8") as handle:
    handle.write("\n".join(picked))
print("SELECT_OK", len(picked))
