"""Random faces from the FLAME sliders on one HG human, rendered straight on, side by side (a look check).

Usage: blender -b <human.blend> -P pipeline/flame/flame_lineup.py -- <out.jpg> [--count 6] [--spread 1.0] [--seed 1] [--yaw 0] [--only i]
human.blend: a saved HG human (make_human.py). Its own face sliders and face presets are zeroed first, so the
faces differ only by FLAME's directions; each face takes random values N(0, spread) on every flame_shape slider
(1.0 = the capture's sigma standard deviations). The first face is the neutral one. Prints FLAME_LINEUP_OK.
"""
import os
import runpy
import sys

import bpy
import numpy as np

from HumGen3D import Human
from HumGen3D.human.keys.keys import update_livekey_collection

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
count = int(argv[argv.index("--count") + 1]) if "--count" in argv else 6
spread = float(argv[argv.index("--spread") + 1]) if "--spread" in argv else 1.0
rng = np.random.default_rng(int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 1)
yaw = argv[argv.index("--yaw") + 1] if "--yaw" in argv else "0"
only = int(argv[argv.index("--only") + 1]) if "--only" in argv else None  # render just this face (others drawn)
HERE = os.path.dirname(os.path.abspath(__file__))
SHOT = os.path.join(os.path.dirname(HERE), "humgen", "face_shot.py")

rig = next(o for o in bpy.data.objects if o.type == "ARMATURE" and "eyeball.L" in o.data.bones)
human = Human.from_existing(rig)
update_livekey_collection()
flame = [k for k in human.keys.all_livekeys if k.name.startswith("flame_shape_")]
for k in human.keys.all_livekeys:
    if k.category in ("face_proportions", "face_presets") and not k.name.startswith("flame_shape_") and k.value:
        k.set_without_update(0.0)
print("FLAME sliders:", len(flame))
tiles = []
for i in range(count):
    vals = np.zeros(len(flame)) if i == 0 else rng.normal(0, spread, len(flame))
    for k, v in zip(flame, vals):
        k.set_without_update(float(v))
    if only is not None and i != only:
        continue
    human.keys.update_human_from_key_change(bpy.context)
    img = out + f".{i}.png"
    sys.argv = ["blender", "--", img, "--size", "600x800", "--yaw", yaw]
    runpy.run_path(SHOT, run_name="__main__")
    tiles.append(img)
import subprocess  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(ROOT, "tools", "pose-env", "Scripts", "python.exe")
subprocess.run([PY, "-I", "-c", "import sys,cv2,numpy as np; ims=[cv2.imread(p) for p in sys.argv[2:]]; "
                "cv2.imwrite(sys.argv[1], np.hstack(ims), [cv2.IMWRITE_JPEG_QUALITY, 90])", out, *tiles], check=True)
for p in tiles:
    os.remove(p)
print("FLAME_LINEUP_OK", out)
