"""Strand hair for a head made by face_from_image.py: grown in its carved hair volume, coloured like the frames.

Usage: python groom_head.py <head.blend> <hair.json> <out.blend> [KEY=value groom options]
head.blend: with the head ("head") and its carved hair volume ("hair", carve_hair_frames.py);
hair.json: frame_masks.py's (the hair's colour in the frames).
Runs tests/realhead/scripts/groom_hair.py with the settings worked out for sleek hair
(tests/bodyfit/scripts/groom_from_sheet.py's DEFAULTS: roots wherever the scalp is under the volume, strands
ending where they leave it, combed down from a parting, kept inside the volume's smoothed surface, none
right beside the face), the head as the scalp. Melanin and redness from the hair's colour: darker hair, more
melanin (calibrate_hair.py can fine-tune it against a frame later).
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "bodyfit", "scripts"))
from groom_from_sheet import BLENDER, DEFAULTS, GROOM  # noqa: E402


def colour_options(rgb):
    bgr = np.uint8([[rgb[::-1]]])
    L, a, _ = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    L = L * 100 / 255
    return {"MELANIN": f"{float(np.clip((90 - L) / 80, 0.05, 1.0)):.2f}",
            "REDNESS": f"{float(np.clip(0.08 + (a - 128) / 60, 0.0, 1.0)):.2f}"}


def main():
    blend, hair_json, out = sys.argv[1:4]
    info = json.load(open(hair_json, encoding="utf-8"))
    if not info.get("hair_rgb"):
        print("GROOM skipped: no hair in the frames")
        return
    opts = dict(DEFAULTS)
    opts.update({"HEAD": "head", "VOLUME": "hair"})
    opts.update(colour_options(info["hair_rgb"]))
    opts.update(dict(a.split("=", 1) for a in sys.argv[4:]))
    done = subprocess.run([BLENDER, "-b", os.path.abspath(blend), "-P", GROOM, "--", os.path.abspath(out)]
                          + [f"{k}={v}" for k, v in opts.items()], capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    for line in done.stdout.splitlines():
        if line.startswith(("SKULL", "ROOTS", "GUIDES", "STRAND", "REACH", "CLAMPED", "GROOM_OK")) or "Error" in line:
            print(line)
    if done.returncode != 0 or "GROOM_OK" not in done.stdout:
        print(done.stderr[-2000:])
        sys.exit(1)
    print("GROOMED", {k: opts[k] for k in ("STRANDS", "MELANIN", "REDNESS")})


if __name__ == "__main__":
    main()
