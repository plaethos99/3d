"""Age lineup: one Human Generator human built at several ages, rendered with the same camera, side by side.

Usage: tools/pose-env/Scripts/python.exe -I pipeline/humgen/child/lineup.py <out.jpg> --ages 2,4,6,9,12,15,25
       --view front|three_quarter -- <make_human.py options: --gender, --preset, --outfit, --hair, ...>
Each age is a make_human.py run with --reset-body --child-look --fixed-camera --age N (adults: no child look)
into <out dir>/lineup_<name>/<age>/. Labels each picture with its age and height. Prints LINEUP_OK.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BLENDER = os.path.join(ROOT, "tools", "blender-5.2.3-windows-x64", "blender.exe")
MAKE = os.path.join(ROOT, "pipeline", "humgen", "make_human.py")

own, rest = sys.argv[1:sys.argv.index("--")], sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(own[0])
ages = [int(a) for a in own[own.index("--ages") + 1].split(",")] if "--ages" in own else [2, 4, 6, 9, 12, 15, 25]
view = own[own.index("--view") + 1] if "--view" in own else "front"
work = os.path.join(os.path.dirname(out), "lineup_" + os.path.splitext(os.path.basename(out))[0])
tiles = []
for a in ages:
    d = os.path.join(work, str(a))
    # children: the preset's adult face features softened, more for the youngest
    extra = ["--reset-body", "--child-look", "--face-scale", f"{min(1.0, 0.25 + 0.75 * a / 20):.2f}"] \
        if a < 20 else []
    cmd = [BLENDER, "-b", "-P", MAKE, "--", d, *rest, *extra, "--fixed-camera", "--age", str(a)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if "MAKE_HUMAN_OK" not in p.stdout:
        print(p.stdout[-3000:])
        raise SystemExit(f"LINEUP_FAILED age {a}")
    rep = json.load(open(os.path.join(d, "human.json")))
    img = cv2.imread(os.path.join(d, view + ".png"))
    img = img[120:1180, 225:675]  # 900x1200 render: keep the middle
    label = f"{a} y" if a < 20 else "adult"
    cv2.putText(img, label, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{rep['height_m']:.2f} m", (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 1,
                cv2.LINE_AA)
    print(f"age {a}: {rep['height_m']} m, errors {rep['errors']}")
    tiles.append(img)
cv2.imwrite(out, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("LINEUP_OK", out)
