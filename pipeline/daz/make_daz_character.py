"""One Genesis 9 character in Daz Studio from a person's turned views: face, rig, skin, body, hair, clothes, renders.

Usage: python pipeline/daz/make_daz_character.py <char_dir> <Name> [--from flame|fb|face|skin|daz|blender] [--rounds 2]
       [--fresh] [--no-iray] [--daz-hair [--heavy-hair]] [--likeness N]
<char_dir>/face_run/: front.png, left45.png, left75.png, right45.png, right75.png (the same person from five sides:
  engine/imagegen face_views.mjs, the right side mirrored), and <char_dir>/char.json: the character file g9_build.dsa
  reads ({"sex": "female"|"male", "height_cm", "body_morphs", "figure_presets", "wear", "facial_hair": true|false},
  "face_shape": {"jaw_width": -0.3, "chin_length": 0.4, ...} (g9_face.py shape controls), "blender_hair": {...}
  (blender_hair.py)).
Steps (each a script of pipeline/daz, pipeline/flame, pipeline/keentools):
  flame  the hair colour (top of the front picture) and the FLAME photo fit (flame_photofit.py: free, always run)
  fb     KeenTools FaceBuilder on the same pictures (fb_head.py, Blender 4.2) when it is installed and licensed;
         without it, the FLAME head is used
  face   the head as a face run (flame_head.py / fb_to_run.py) and the Genesis 9 face morph with its rig (g9_face.py)
  skin   the photo skin on Genesis 9 (g9_skin.py)
  daz    g9_calibrate.py: Daz Studio builds the character (g9_build.dsa) without head hair, renders it with Iray,
         measures skin (and facial hair) against the front picture and corrects the skin textures, --rounds times
         (--no-iray: only build and save). Beards stay in Daz; --daz-hair keeps the head hair too (dForce/strand
         hair swapped for a light mesh hair, LIGHT_HAIR, unless --heavy-hair)
  blender  daz_to_blender.py: the scene in Blender (Diffeomorphic), head hair grown there (blender_hair.py: char.json
         "blender_hair" options, colour from the portrait), EEVEE renders into tests/blender_daz/<name>_bl_*
         (--likeness N: also match the face to the portrait in Blender and write it into the Daz morph; off)
Writes <char_dir>/build.json (what was used) and the Daz scene and renders into tests/daz/<name>_g9*, and appends
one line per finished step to the cast's DAZ_PROGRESS.md (<char_dir>/../../DAZ_PROGRESS.md for <cast>/<name>/daz).
Prints MAKE_DAZ_CHARACTER_OK.
"""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(ROOT, "tools", "pose-env", "Scripts", "python.exe")
B5 = os.path.join(ROOT, "tools", "blender-5.2.3-windows-x64", "blender.exe")
B42 = os.path.join(ROOT, "tools", "blender-4.2.23-windows-x64", "blender.exe")
STEPS = ["flame", "fb", "face", "skin", "daz", "blender"]
DAZ = r"C:\Program Files\DAZ 3D\DAZStudio6\DAZStudio.exe"
LIB = r"C:\Users\Public\Documents\My DAZ 3D Library"
# light hair (mesh hair, no dForce/strands: Iray and this machine's memory, and game-ready): a heavy hair in char.json
# is swapped for one of these when the build runs (char_build.json; char.json itself is not changed)
LIGHT_HAIR = {"male": "/People/Genesis 9/Hair/Daz Originals/Mavick Hair and Beard/Mavick Hair.duf",
              "female": "/People/Genesis 9/Hair/Daz Originals/Eirgrid Hair/Eirgrid Hair G9.duf"}
HEAVY_HAIR = ("dforce", "strand-based")


def opt(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


cdir, name = os.path.abspath(sys.argv[1]), sys.argv[2]
start = STEPS.index(opt("--from", "flame"))
rounds = opt("--rounds", "2")
F = os.path.join(cdir, "face_run")
VIEWS = [os.path.join(F, v + ".png") for v in ("front", "left45", "left75", "right45", "right75")]
char = json.load(open(os.path.join(cdir, "char.json")))
sex = "feminine" if char.get("sex", "female") == "female" else "masculine"
t0 = time.time()


def run(cmd, ok, show=()):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = r.stdout + r.stderr
    if ok not in text:
        print(text[-3000:])
        if current.get("step"):
            progress(current["step"], f"FAILED ({ok} missing)")
        raise SystemExit(f"MAKE_DAZ_CHARACTER_FAILED at {os.path.basename(next((c for c in cmd if c.endswith(('.py', '.dsa'))), cmd[0]))}")
    for ln in text.splitlines():
        if ln.startswith(tuple(show)):
            print("   ", ln[:200])
    return text


_cd = os.path.normpath(cdir)
# <cast>/<name>/daz -> <cast>/DAZ_PROGRESS.md; any other <dir>/<name> -> <dir>/DAZ_PROGRESS.md
LOG = os.path.join(os.path.dirname(os.path.dirname(_cd)) if os.path.basename(_cd) == "daz" else os.path.dirname(_cd),
                   "DAZ_PROGRESS.md")


def progress(stepname, result):
    """One line per finished step in the cast's progress log (for the owner and for handoffs)."""
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} · {name} · {stepname} · {result}\n")


current = {"step": None}


def step(n):
    if current["step"]:
        progress(current["step"], f"done ({time.time() - current['t']:.0f}s)")
    current.update(step=n.split(" ")[0], t=time.time())
    print(f"{n} ({time.time() - t0:.0f}s)", flush=True)


flame_dir, fb_dir = os.path.join(cdir, "flame"), os.path.join(cdir, "fb")
use_fb = os.path.exists(B42) and os.path.isdir(os.path.join(os.path.dirname(B42), "portable", "scripts", "addons", "keentools"))
if start <= 0:
    step("flame")
    os.makedirs(os.path.join(F, "masks"), exist_ok=True)
    run([PY, "-I", "-c", "import cv2,numpy as np,json,sys; F=sys.argv[1]; im=cv2.imread(F+'/front.png'); h,w=im.shape[:2]; "
         "top=im[int(h*0.10):int(h*0.20), int(w*0.35):int(w*0.65)].reshape(-1,3).astype(float); "
         "hair=top[top.mean(1)<np.percentile(top.mean(1),40)]; c=[int(x) for x in np.median(hair,0)[::-1]]; "
         "json.dump({'hair_rgb':c,'source':'front.png top'},open(F+'/masks/hair.json','w')); print('HAIR', c)", F], "HAIR", ["HAIR"])
    run([PY, "-I", os.path.join(ROOT, "pipeline", "flame", "flame_photofit.py"), flame_dir, *VIEWS], "FLAME_PHOTOFIT_OK", ["SHAPE"])
if start <= 1 and use_fb:
    step("fb (KeenTools FaceBuilder)")
    text = subprocess.run([B42, "-b", "-P", os.path.join(ROOT, "pipeline", "keentools", "fb_head.py"), "--", fb_dir, *VIEWS],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if "FB_HEAD_OK" not in text.stdout + text.stderr:
        print("    FaceBuilder failed or unlicensed: using the FLAME head")
        use_fb = False
    else:
        print("   ", next(ln for ln in text.stdout.splitlines() if ln.startswith("FB_HEAD_OK"))[:160])
use_fb = use_fb and os.path.exists(os.path.join(fb_dir, "fb_head.blend"))
if start <= 2:
    step("face")
    run([B5, "-b", "-P", os.path.join(ROOT, "pipeline", "flame", "flame_head.py"), "--", os.path.join(flame_dir, "heads.npz"),
         flame_dir], "FLAME_HEAD_OK")
    pf = os.path.join(flame_dir, "photofit")
    os.makedirs(os.path.join(pf, "textures"), exist_ok=True)
    os.makedirs(os.path.join(pf, "masks"), exist_ok=True)
    run([PY, "-I", os.path.join(ROOT, "pipeline", "keentools", "fb_textures.py"), os.path.join(flame_dir, "texture.png"),
         os.path.join(pf, "textures")], "FB_TEXTURES_OK")
    shutil.copy(os.path.join(F, "masks", "hair.json"), os.path.join(pf, "masks", "hair.json"))
    shutil.copy(VIEWS[0], os.path.join(pf, "front.png"))
    head_run = pf
    if use_fb:
        run([B5, "-b", os.path.join(fb_dir, "fb_head.blend"), "-P", os.path.join(ROOT, "pipeline", "keentools", "fb_to_run.py"),
             "--", os.path.join(fb_dir, "run"), "--like", pf], "FB_RUN_OK", ["SCALE"])
        head_run = os.path.join(fb_dir, "run")
    shape = dict(char.get("face_shape", {}))  # the studio's Face sliders (g9_face.py 3c), e.g. {"jaw_width": -0.3}
    if "lower_face_narrow" in shape:  # the older name
        shape.setdefault("jaw_width", -float(shape.pop("lower_face_narrow")))
    run([B5, "-b", "-P", os.path.join(HERE, "g9_face.py"), "--", head_run, name, "--label", f"Face: {name}"]
        + (["--shape", json.dumps(shape)] if shape else []), "G9_FACE_OK",
        ["LANDMARKS", "SHAPE", "RIG"])
    json.dump({"head_run": head_run, "face_from": "FaceBuilder" if use_fb else "FLAME photo fit"},
              open(os.path.join(cdir, "build.json"), "w"), indent=1)
build = json.load(open(os.path.join(cdir, "build.json")))
head_run = build["head_run"]
if start <= 3:
    step("skin")
    run([sys.executable, os.path.join(HERE, "g9_skin.py"), head_run, name, "--sex", sex, "--rebake"], "G9_SKIN_OK",
        ["SKIN base", "TONE"])
# what Daz builds: char.json without head hair (the hair is grown in Blender: blender_hair.py); beards stay.
# --daz-hair keeps the head hair in Daz (heavy dForce/strand hair still swapped for a light mesh hair)
daz_hair = "--daz-hair" in sys.argv
build_char = dict(char)
wear = []
for w_ in char.get("wear", []):
    w_ = dict(w_) if isinstance(w_, dict) else {"wear": w_}
    p_ = w_["wear"].lower()
    if "/hair/" in p_ and not any(k in os.path.basename(p_) for k in ("beard", "brow", "lash")) and not daz_hair:
        print(f"    head hair left to Blender: {os.path.basename(w_['wear'])}")
        continue
    if "/hair/" in p_ and any(h in p_ for h in HEAVY_HAIR) and "--heavy-hair" not in sys.argv:
        light = LIGHT_HAIR[char.get("sex", "female")]
        print(f"    light hair: {os.path.basename(w_['wear'])} -> {os.path.basename(light)}")
        w_ = {"wear": light, "presets": [], **({"tint": w_["tint"]} if w_.get("tint") else {})}
    wear.append(w_)
build_char["wear"] = wear
char_build = os.path.join(cdir, "char_build.json")
json.dump(build_char, open(char_build, "w"), indent=1)
scene = os.path.join(ROOT, "tests", "daz", f"{name.lower()}_g9.duf")
iray = "--no-iray" not in sys.argv  # Daz makes the skin textures: Iray renders calibrate them (as before)
if start <= 4 and not iray:
    step("daz (build and save, no Iray)")
    report = os.path.join(ROOT, "tests", "daz", f"build_{name.lower()}.txt")
    if os.path.exists(report):
        os.remove(report)
    args = [DAZ, "-instanceName", "#", "-noPrompt"]
    for a in ("OURS_" + name, scene, report, os.path.join(LIB, "Runtime", "Textures", "Ours", name, "skin.json"), "-",
              char_build):
        args += ["-scriptArg", a.replace("\\", "/")]
    args.append(os.path.join(HERE, "g9_build.dsa").replace("\\", "/"))
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tb = time.time()
    while not (os.path.exists(report) and "done" in open(report, errors="replace").read()):
        if proc.poll() is not None or time.time() - tb > 1200:
            progress("daz", "FAILED (Daz exited or hung; see " + report + ")")
            raise SystemExit("MAKE_DAZ_CHARACTER_FAILED at g9_build.dsa")
        time.sleep(5)
    rep_txt = open(report, errors="replace").read()
    if "ERROR" in rep_txt or "scene exists true" not in rep_txt:
        print(rep_txt[-1500:])
        progress("daz", "FAILED (see " + report + ")")
        raise SystemExit("MAKE_DAZ_CHARACTER_FAILED at g9_build.dsa")
    try:
        proc.wait(120)  # one Daz at a time: the export starts it again
    except subprocess.TimeoutExpired:
        pass
    print("   ", " | ".join(ln.strip() for ln in rep_txt.splitlines() if ln.startswith(("wear", "   tint")))[:300])
if start <= 4 and iray:
    step("daz (build, render, colour calibration)")
    cmd = [PY, "-I", os.path.join(HERE, "g9_calibrate.py"), head_run, name, "OURS_" + name, name.lower(), VIEWS[0],
           char_build, "--sex", sex, "--rounds", rounds]
    if char.get("facial_hair"):
        cmd.append("--facial-hair")
    if "--fresh" in sys.argv:  # calibrate from scratch, not from the last run's corrections
        cmd.append("--fresh")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("\n".join("    " + ln[:220] for ln in r.stdout.splitlines() if ln.startswith("round")))
    if "G9_CALIBRATE_OK" not in r.stdout:
        print(r.stdout[-1500:], r.stderr[-1500:])
        progress("daz", "FAILED (see tests/daz/build_" + name.lower() + ".txt)")
        raise SystemExit("MAKE_DAZ_CHARACTER_FAILED at g9_calibrate.py")
    rounds_txt = [ln for ln in r.stdout.splitlines() if ln.startswith("round")]
    if rounds_txt:
        import re as _re
        m = _re.search(r"skin photo \[([^\]]+)\] render \[([^\]]+)\]", rounds_txt[-1])
        progress("daz", "built and rendered" + (f", skin render {m.group(2)} vs portrait {m.group(1)}" if m else ""))
    current["step"] = None
if start <= 5:
    step("blender (import, hair grown in Blender, renders)")
    hair_cfg = dict(char.get("blender_hair", {}))
    hj = os.path.join(F, "masks", "hair.json")
    if "colour" not in hair_cfg and os.path.exists(hj):
        hair_cfg["rgb"] = json.load(open(hj))["hair_rgb"]  # the portrait's hair colour
    hair_cfg.setdefault("sex", char.get("sex", "female"))
    hcfg = os.path.join(cdir, "hair_build.json")
    json.dump(hair_cfg, open(hcfg, "w"), indent=1)
    cmd = [sys.executable, os.path.join(HERE, "daz_to_blender.py"), scene, os.path.join(ROOT, "tests", "blender_daz"),
           "--name", name.lower(), "--reexport"] + ([] if daz_hair else ["--hair", hcfg])
    if int(opt("--likeness", "0")):  # optional, off by default: the face as Daz made it
        cmd += ["--photo", VIEWS[0], "--likeness", opt("--likeness", "0"), "--morph",
                os.path.join(LIB, "data", "Daz 3D", "Genesis 9", "Base", "Morphs", "Ours", "Faces", f"OURS_{name}.dsf")]
    out = run(cmd, "DAZ_TO_BLENDER_OK", ["    HAIR", "    LIKENESS", "    RENDER", "DAZ_TO_BLENDER_OK"])
    progress("blender", "hair grown and rendered in Blender (" + next((ln.strip() for ln in out.splitlines()
                                                                         if ln.strip().startswith("HAIR grown")), "no hair") + ")")
    current["step"] = None
shots = os.path.join("tests", "blender_daz", f"{name.lower()}_bl_{{}}.png")
build.update({"name": name, "sex": sex, "scene": scene, "renderer": "Blender EEVEE (skin calibrated with Iray in Daz)",
              "renders": [os.path.join(ROOT, shots.format(v)) for v in ("face", "face34", "body")]})
json.dump(build, open(os.path.join(cdir, "build.json"), "w"), indent=1)
print(f"MAKE_DAZ_CHARACTER_OK {name} ({time.time() - t0:.0f}s, face from {build['face_from']})")
