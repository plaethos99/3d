"""Genesis 9 character colours matched to the photo: skin tone and facial hair, measured on the render (loop).

Usage: python pipeline/daz/g9_calibrate.py <head_dir> <tex name> <morph id> <out name> <photo> <character.json>
       [--sex feminine|masculine] [--rounds 2] [--facial-hair] [--daz "C:/Program Files/DAZ 3D/DAZStudio6/DAZStudio.exe"]
Each round: g9_skin.py with the current corrections -> g9_build.dsa in its own Daz Studio instance (scene and
renders into tests/daz/<out name>_g9*) -> MediaPipe on the photo and on the face render: skin = cheeks and forehead,
upper lip = the moustache's place (between the nose and the mouth) -> corrections multiplied by
(photo / render)^0.8 per channel in linear light (damped, clamped). --facial-hair: also correct the moustache/beard
painted in the face (otherwise the upper lip only counts as skin). Writes tests/daz/<out name>_calibration.json.
Starts from the corrections of the last run (tests/daz/<out name>_calibration.json) unless --fresh: re-dressing a
calibrated character needs --rounds 0. Prints G9_CALIBRATE_OK.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(ROOT, "tools", "pose-env", "Scripts", "python.exe")
MODEL = os.path.join(ROOT, "tools", "face-model", "face_landmarker.task")
LIB = r"C:\Users\Public\Documents\My DAZ 3D Library"

MEASURE = r'''
import sys, json, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=sys.argv[1]),
                                        running_mode=RunningMode.IMAGE, num_faces=1))
out = {}
for path in sys.argv[2:]:
    im = cv2.imread(path); h, w = im.shape[:2]
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    if not r.face_landmarks:
        out[path] = None; continue
    p = np.array([[q.x * w, q.y * h] for q in r.face_landmarks[0]])
    rad = max(2, int(np.linalg.norm(p[33] - p[263]) * 0.07))
    def patch(pt):
        x, y = pt.astype(int)
        return im[y - rad:y + rad, x - rad:x + rad].reshape(-1, 3)[:, ::-1].astype(float)
    skin = np.vstack([patch(p[i]) for i in (50, 280, 151)])
    lip = np.vstack([patch((p[164] + p[0]) / 2), patch((p[167] + p[37]) / 2), patch((p[393] + p[267]) / 2)])
    out[path] = {"skin": np.median(skin, 0).tolist(), "upper_lip": np.median(lip, 0).tolist()}
print("MEASURED", json.dumps(out))
'''


def opt(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


head_dir, tex, morph, out_name, photo, char = sys.argv[1:7]
sex = opt("--sex", "feminine")
rounds = int(opt("--rounds", "2"))
daz = opt("--daz", r"C:\Program Files\DAZ 3D\DAZStudio6\DAZStudio.exe")
facial = "--facial-hair" in sys.argv
D = os.path.join(ROOT, "tests", "daz")
os.makedirs(D, exist_ok=True)


def lin(c):
    import numpy as np
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def daz_build():
    report = os.path.join(D, f"build_{out_name}.txt")
    if os.path.exists(report):
        os.remove(report)
    face = os.path.join(D, f"{out_name}_g9_face.png")
    if os.path.exists(face):
        os.remove(face)
    args = [daz, "-instanceName", "#", "-noPrompt"]
    for a in (morph, os.path.join(D, f"{out_name}_g9.duf"), report, os.path.join(LIB, "Runtime", "Textures", "Ours", tex, "skin.json"),
              os.path.join(D, f"{out_name}_g9.png"), os.path.abspath(char)):
        args += ["-scriptArg", a.replace("\\", "/")]
    args.append(os.path.join(HERE, "g9_build.dsa").replace("\\", "/"))
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t = time.time()
    while time.time() - t < 4800:  # Iray renders on the CPU here (heavy strand hair takes long)
        if os.path.exists(report) and "done" in open(report, errors="replace").read():
            return face
        time.sleep(10)
    raise SystemExit("G9_CALIBRATE_FAILED: Daz build did not finish")


gain, hair = [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]
prev = os.path.join(D, f"{out_name}_calibration.json")
if os.path.exists(prev) and "--fresh" not in sys.argv:  # start from the last calibration (a re-dress needs no new rounds)
    last = json.load(open(prev))[-1]
    gain, hair = [float(g) for g in last["gain"]], [float(g) for g in last["facial_hair_gain"]]
    print("resuming from the last calibration:", [round(g, 3) for g in gain], [round(g, 3) for g in hair], flush=True)
log = []
for rnd in range(rounds + 1):
    r = subprocess.run([sys.executable, os.path.join(HERE, "g9_skin.py"), head_dir, tex, "--sex", sex,
                        "--gain", ",".join(f"{g:.4f}" for g in gain), "--hair-gain", ",".join(f"{g:.4f}" for g in hair)]
                       + (["--facial-hair"] if facial else []),
                       capture_output=True, text=True)
    if "G9_SKIN_OK" not in r.stdout:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("G9_CALIBRATE_FAILED at g9_skin.py")
    t0 = time.time()
    face = daz_build()
    m = subprocess.run([PY, "-I", "-c", MEASURE, MODEL, photo, face], capture_output=True, text=True)
    meas = json.loads(next(ln for ln in m.stdout.splitlines() if ln.startswith("MEASURED")).split(" ", 1)[1])
    ph, rd = meas[photo], meas[face]
    entry = {"round": rnd, "gain": gain, "facial_hair_gain": hair, "photo": ph, "render": rd, "daz_s": round(time.time() - t0)}
    log.append(entry)
    print(f"round {rnd}: skin photo {[round(v) for v in ph['skin']]} render {[round(v) for v in rd['skin']]} | "
          f"upper lip photo {[round(v) for v in ph['upper_lip']]} render {[round(v) for v in rd['upper_lip']]} | "
          f"gain {[round(g, 3) for g in gain]} hair {[round(g, 3) for g in hair]}", flush=True)
    if rnd == rounds:
        break
    import numpy as np
    step = np.clip((lin(ph["skin"]) / np.maximum(lin(rd["skin"]), 1e-4)) ** 0.8, 0.7, 1.4)
    gain = list(np.clip(np.array(gain) * step, 0.4, 2.5))
    if facial:  # the facial hair's own correction, after the skin's (it is applied on top of it)
        hstep = np.clip((lin(ph["upper_lip"]) / np.maximum(lin(rd["upper_lip"]), 1e-4)) ** 0.8 / step, 0.6, 1.6)
        hair = list(np.clip(np.array(hair) * hstep, 0.3, 3.0))
json.dump(log, open(os.path.join(D, f"{out_name}_calibration.json"), "w"), indent=1)
print("G9_CALIBRATE_OK", out_name)
