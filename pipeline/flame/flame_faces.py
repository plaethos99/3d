"""FLAME's face shapes as Human Generator face sliders ("Face shape (FLAME)" group): one command.

Usage: python pipeline/flame/flame_faces.py [--model tools/flame/flame2023_Open.pkl] [--count 20] [--sigma 2]
       [--genders female,male] [--work tests/flame] [--from heads|capture|sliders]
FLAME 2023 Open (CC-BY-4.0, Max Planck Institute for Intelligent Systems, flame.is.tue.mpg.de) is a statistical
head model learned from thousands of 3D head scans: its identity directions are the main ways real heads
differ. For each of the first --count directions:
  heads    the mean FLAME head and the head --sigma standard deviations along it (flame_model.py), written as
           head.blend + subject_fit.json each (flame_head.py, Blender)
  capture  each head captured onto HG's head by landmarks (pipeline/humgen/face_capture.py), per gender
  sliders  slider k = capture(direction k) - capture(mean): HG's move from the mean FLAME face to that face,
           saved as livekeys/face_proportions/flame/<gender>_flame_shape_<kk>.npz; HG lists them as sliders
           (1.0 = sigma standard deviations; the studio's range -3..3 reaches far, keep to about +-1.5)
Captures go to <work>; the temporary face_presets livekeys are moved there too. Prints FLAME_FACES_OK.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from flame_model import load, shape  # noqa: E402

BLENDER = os.path.join(ROOT, "tools", "blender-5.2.3-windows-x64", "blender.exe")
CONTENT = os.path.join(ROOT, "tools", "Human", "content")
STEPS = ["heads", "capture", "sliders"]


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


model_path = os.path.join(ROOT, opt("--model", os.path.join("tools", "flame", "flame2023_Open.pkl")))
count = int(opt("--count", "20"))
sigma = float(opt("--sigma", "2"))
genders = opt("--genders", "female,male").split(",")
work = os.path.join(ROOT, opt("--work", os.path.join("tests", "flame")))
start = STEPS.index(opt("--from", "heads"))
os.makedirs(work, exist_ok=True)
names = ["mean"] + [f"pc{k + 1:02d}" for k in range(count)]
t0 = time.time()


def run(cmd, ok):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = r.stdout + r.stderr
    if ok not in text:
        print(text[-3000:])
        raise SystemExit(f"FLAME_FACES_FAILED at {os.path.basename(cmd[3] if len(cmd) > 3 else cmd[0])}")
    return text


if start <= 0:
    if not os.path.exists(model_path):
        raise SystemExit(f"FLAME_FACES_FAILED: no FLAME model at {model_path} (download FLAME 2023 Open from "
                         "flame.is.tue.mpg.de after signing up; put the .pkl there)")
    m = load(model_path)
    print("FLAME", m["v_template"].shape[0], "vertices,", m["n_shape"], "identity directions")
    verts = [shape(m, [])]
    for k in range(count):
        b = np.zeros(k + 1)
        b[k] = sigma
        verts.append(shape(m, b))
    np.savez(os.path.join(work, "heads.npz"), faces=m["faces"], names=np.array(names), verts=np.array(verts))
    text = run([BLENDER, "-b", "-P", os.path.join(HERE, "flame_head.py"), "--", os.path.join(work, "heads.npz"),
                work], "FLAME_HEAD_OK")
    print(*[ln for ln in text.splitlines() if ln.startswith(("PIECES", "LANDMARKS", "FLAME_HEAD_OK"))], sep="\n")

presets = os.path.join(CONTENT, "livekeys", "face_presets")
keep = os.path.join(work, "captures")
os.makedirs(keep, exist_ok=True)
if start <= 1:
    for g in genders:
        for n in names:
            t = time.time()
            text = run([BLENDER, "-b", "-P", os.path.join(ROOT, "pipeline", "humgen", "face_capture.py"), "--",
                        os.path.join(work, n, "head.blend"), os.path.join(work, n, g), "--name", f"flame_{n}",
                        "--gender", g], "FACE_CAPTURE_OK")
            src = os.path.join(presets, f"{g}_face_flame_{n}.npz")
            shutil.move(src, os.path.join(keep, f"{g}_{n}.npz"))  # not a face preset of its own
            moved = next((ln for ln in text.splitlines() if ln.startswith("FACE moved")), "")
            print(f"captured {g} {n} ({time.time() - t:.0f}s) {moved}", flush=True)


def dense(path, size):
    d = np.load(path)
    v = np.zeros(size, np.float32)
    v[d["indices"].ravel()] = d["relative_coordinates"].ravel()
    return v


if start <= 2:
    out_dir = os.path.join(CONTENT, "livekeys", "face_proportions", "flame")
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for g in genders:
        caps = {n: os.path.join(keep, f"{g}_{n}.npz") for n in names}
        size = 1 + max(int(np.load(p)["indices"].max()) for p in caps.values())
        base = dense(caps["mean"], size)
        for k, n in enumerate(names[1:], 1):
            diff = dense(caps[n], size) - base
            idx = np.where(np.abs(diff) > 1e-6)[0]
            np.savez(os.path.join(out_dir, f"{g}_flame_shape_{k:02d}.npz"), indices=idx[None].astype(np.int64),
                     relative_coordinates=diff[idx].astype(np.float32))
            mv = np.linalg.norm(diff.reshape(-1, 3), axis=1)
            report[f"{g}_flame_shape_{k:02d}"] = round(float(mv.max() * 1000), 1)
    json.dump(report, open(os.path.join(work, "sliders.json"), "w"), indent=1)
    print("SLIDERS (largest move, mm):", report)
print(f"FLAME_FACES_OK {work} ({time.time() - t0:.0f}s)")
