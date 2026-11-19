"""One command: a photo-made head (face_from_image.py's run) in, a Human Generator character with that face out.

Usage: python pipeline/humgen/photo_character.py <face_run> <out_dir> --name flow_woman [--gender female]
       [--body tests/body/<name>] [--photo front.png] [--hair-style "Long\\Bob Long"] [--outfit "Casual\\Smart_Casual"]
       [--footwear Leather_Sneakers] [--preset Emma] [--eye-rounds 3] [--height-cm 165] [--from step]
       [--brows Eyebrows_Female|none] [--braid left|right [--braid-length 0.45]] [--skin-from <other name>]
       [--face-hair "Large_Mustache_[Short]"] [--salt-pepper 0.4]  (HG strand brows on the captured face; they sit where the photo's brows are)
face_run: tests/face/<name> (head.blend, textures/, masks/hair.json, turn.mp4).
Steps (each a script; --from picks where to start):
  capture  face_capture.py   the face shape as the HG slider face_<name>
  bake     face_bake.py      the photo skin onto HG's skin layout
  skin     face_blend.py     the closest of HG's skins (by tone) with the photo face blended in -> skin face_<name>
  eyes     make_human.py + face_shot.py + measure_eyes.py + measure_skin.py, --eye-rounds times: iris and
           eye-white colours and HG's skin tone/saturation set from the photo's, rendered (Standard view, like a
           photo), measured, corrected (eye whites never greyer than a natural warm white)
  final    make_human.py with everything, face_shot.py, a side-by-side with the photo
Values taken from the measurements, not chosen: skin (tone match), iris and eye-white colours (measured), height
(the body run's stature, if --body), hair colour (calibrated like the eyes, beside the temples). Still
chosen: hairstyle, outfit, shoes and brow style (flags), the body preset's build. Writes <out_dir>/values.json and
compare.jpg. Prints PHOTO_CHARACTER_OK.
"""
import json
import math
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLENDER = os.path.join(ROOT, "tools", "blender-5.2.3-windows-x64", "blender.exe")
PY = os.path.join(ROOT, "tools", "pose-env", "Scripts", "python.exe")
HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(ROOT, "tools", "Human", "content")
FACE_MODEL = os.path.join(ROOT, "tools", "face-model", "face_landmarker.task")
MEASURE_EYES = os.path.join(ROOT, "tests", "realhead", "scripts", "measure_eyes.py")

argv = sys.argv[1:]
face_run, out = os.path.abspath(argv[0]), os.path.abspath(argv[1])


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


name = opt("--name", os.path.basename(face_run))
gender = opt("--gender", "female")
preset = opt("--preset", "Emma" if gender == "female" else "Jack")
steps = ["capture", "bake", "skin", "eyes", "final"]
start = steps.index(opt("--from", "capture"))
os.makedirs(out, exist_ok=True)
work = os.path.join(out, "face")
t0 = time.time()


def run(cmd, ok_mark):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = p.stdout + p.stderr
    if ok_mark not in text:
        print(text[-3000:])
        raise SystemExit(f"PHOTO_CHARACTER_FAILED at {os.path.basename(cmd[3] if len(cmd) > 3 else cmd[0])}")
    return text


def blender(script, *args, blend=None, ok):
    cmd = [BLENDER, "-b"] + ([blend] if blend else []) + ["-P", os.path.join(HERE, script), "--", *args]
    return run(cmd, ok)


def show(text, *keys):
    for ln in text.splitlines():
        if ln.startswith(keys):
            print("   ", ln[:200])


def srgb_to_lin(c):
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin_to_hex(rgb):
    out_ = []
    for c in rgb:
        c = min(max(c, 0.0), 1.0)
        s = c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        out_.append(int(round(s * 255)))
    return "#%02X%02X%02X" % tuple(out_)


def measure(photo, render, sheet):
    text = run([PY, "-I", MEASURE_EYES, FACE_MODEL, sheet, f"photo={photo}", f"hg={render}"], "EYES_MEASURED")
    res = {}
    for ln in text.splitlines():
        if ln.startswith(("photo", "hg")):
            label = ln.split()[0]
            irises = [tuple(int(v) for v in m) for m in re.findall(r"iris RGB \((\d+), (\d+), (\d+)\)", ln)]
            whites = [int(v) for v in re.findall(r"white (\d+)", ln)]
            res[label] = {"iris": [sum(c[i] for c in irises) / len(irises) for i in range(3)],
                          "white": sum(whites) / len(whites)}
    return res


values_path = os.path.join(out, "values.json")
values = json.load(open(values_path)) if os.path.exists(values_path) else {}
head = os.path.join(face_run, "head.blend")

# the photo to match: the clip's first (straight-on) frame unless given
photo = opt("--photo")
if not photo:
    photo = os.path.join(out, "photo_front.png")
    if not os.path.exists(photo):
        run([PY, "-I", "-c", "import cv2,sys; c=cv2.VideoCapture(sys.argv[1]); ok,f=c.read(); "
             "cv2.imwrite(sys.argv[2], f); print('FRAME_OK')", os.path.join(face_run, "turn.mp4"), photo], "FRAME_OK")

if start <= 0:
    print(f"capture ({time.time() - t0:.0f}s)")
    show(blender("face_capture.py", head, work, "--name", name, "--gender", gender, ok="FACE_CAPTURE_OK"),
         "ALIGN", "FACE moved")
skins_dir = os.path.join(CONTENT, "textures", gender, "Default 4K")
if opt("--skin-from") and start <= 2:  # a head with no photo texture of its own (a FLAME fit): another run's skin
    import shutil
    shutil.copy(os.path.join(skins_dir, f"face_{opt('--skin-from')}.png"), os.path.join(skins_dir, f"face_{name}.png"))
    print(f"skin copied from face_{opt('--skin-from')}")
    start = 3
if start <= 1:
    print(f"bake ({time.time() - t0:.0f}s)")
    show(blender("face_bake.py", head, work, "--name", name, "--gender", gender, ok="FACE_BAKE_OK"), "BAKED")
skins_dir = os.path.join(CONTENT, "textures", gender, "Default 4K")
if start <= 2:
    print(f"skin ({time.time() - t0:.0f}s)")
    text = run([PY, "-I", os.path.join(HERE, "face_blend.py"), work, skins_dir,
                os.path.join(skins_dir, f"face_{name}.png"), "--tone", "detail"], "FACE_BLEND_OK")
    show(text, "SKIN_PICKED", "LIPS", "FACE_BLEND_OK")
    values["skin_base"] = next(ln.split(" ", 1)[1] for ln in text.splitlines() if ln.startswith("SKIN_PICKED"))

# measured values
if opt("--body"):
    m = json.load(open(os.path.join(opt("--body"), "measurements.json")))
    values["height_cm"] = round(m["stature"] * 100, 1)
else:  # no body run: the given height, or a typical adult one (HG's base body is 1.80 m)
    values["height_cm"] = float(opt("--height-cm", "165" if gender == "female" else "177"))
hair = json.load(open(os.path.join(face_run, "masks", "hair.json")))["hair_rgb"]
if start <= 3 or "hair_lightness" not in values:  # a starting guess, calibrated in the eyes step
    lum = (0.2126 * hair[0] + 0.7152 * hair[1] + 0.0722 * hair[2]) / 255
    # HG's hair shader: lightness about 0.05 (black-brown) .. 0.9 (blond); redness from how warm the colour is
    values["hair_lightness"] = round(min(0.95, max(0.02, (lum - 0.08) * 1.4)), 3)
    values["hair_redness"] = round(min(1.0, max(0.0, (hair[0] - hair[2]) / max(hair[0], 1) * 1.6)), 3)


MEASURE_SKIN = os.path.join(HERE, "measure_skin.py")


def skin_of(*images, what="SKIN"):
    text = run([PY, "-I", MEASURE_SKIN, FACE_MODEL, *images], "SKIN_MEASURED")
    return [[float(v) for v in ln.split()[-3:]] for ln in text.splitlines() if ln.startswith(what + " ")]


def build(dst, iris_hex, sclera_hex, skin_tone=1.0, skin_sat=1.0, exposure=0.0):
    args = [dst, "--gender", gender, "--preset", preset, "--reset-body", "--face-scale", "0",
            "--skin-tone", f"{skin_tone:.3f}", "--skin-saturation", f"{skin_sat:.3f}", "--skin-redness", "0",
            "--key", f"face_{name}=1", "--skin", f"face_{name}", "--brows", opt("--brows", "Eyebrows_Female" if gender == "female" else "Eyebrows_Male"),
            "--iris", iris_hex, "--sclera", sclera_hex, "--lash-dark", "0.04", "--lash-length", opt("--lash-length", "0.7"),
            "--lash-thin", opt("--lash-thin", "0.35"),
            "--hair-lightness", str(values["hair_lightness"]), "--hair-redness", str(values["hair_redness"])]
    if "height_cm" in values:
        args += ["--height-cm", str(values["height_cm"])]
    for flag in ("--face-hair", "--salt-pepper"):
        if opt(flag):
            args += [flag, opt(flag)]
    if opt("--braid"):
        args += ["--braid", opt("--braid"), "--braid-length", opt("--braid-length", "0.45")]
    for flag, default in (("--hair-style", "Long\\Bob Long"), ("--outfit", "Casual\\Smart_Casual"),
                          ("--footwear", "Leather_Sneakers")):
        args += ["--hair" if flag == "--hair-style" else flag, opt(flag, default)]
    blender("make_human.py", *args, ok="MAKE_HUMAN_OK")
    blender("face_shot.py", os.path.join(dst, "face_front.png"), "--exposure", f"{exposure:.3f}",
            blend=os.path.join(dst, "human.blend"), ok="FACE_SHOT_OK")
    return measure(photo, os.path.join(dst, "face_front.png"), os.path.join(dst, "eyes_sheet.png"))


if start <= 3:
    rounds = int(opt("--eye-rounds", "3"))
    # start from the photo's own colours, then correct by what the render shows
    ph = measure(photo, photo, os.path.join(out, "eyes_photo.png"))["photo"]
    iris = [srgb_to_lin(c) for c in ph["iris"]]
    white = srgb_to_lin(ph["white"])
    sclera = [max(white, 0.48), max(white, 0.48) * 0.88, max(white, 0.48) * 0.78]  # warm, not grey
    # start a stop down (the face shot's lights overexpose under Standard), or from an earlier run's values
    tone, sat, ev = 1.0, 1.0, values.get("shot_exposure_ev", -1.0)
    photo_skin = skin_of(photo)[0]
    lum = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]  # noqa: E731
    sat_of = lambda c: (max(c) - min(c)) / max(max(c), 1e-4)  # noqa: E731
    clamp = lambda x, lo, hi: min(max(x, lo), hi)  # noqa: E731
    pl = [srgb_to_lin(c) for c in photo_skin]
    photo_hair = skin_of(photo, what="HAIR")[0]
    for r in range(rounds):
        print(f"eyes+skin round {r + 1} ({time.time() - t0:.0f}s): iris {lin_to_hex(iris)} sclera "
              f"{lin_to_hex(sclera)} skin saturation {sat:.3f}, shot exposure {ev:+.2f} EV")
        dst = os.path.join(out, f"eyes_{r + 1}")
        res = build(dst, lin_to_hex(iris), lin_to_hex(sclera), tone, sat, ev)
        rs = skin_of(os.path.join(dst, "face_front.png"))[0]
        rl = [srgb_to_lin(c) for c in rs]
        print(f"    photo skin {[round(c) for c in photo_skin]} | render skin {[round(c) for c in rs]}")
        print(f"    photo iris {[round(c) for c in res['photo']['iris']]} white {res['photo']['white']:.0f} | "
              f"render iris {[round(c) for c in res['hg']['iris']]} white {res['hg']['white']:.0f}")
        # eyes relative to the skin in the same image (so the shot's exposure doesn't matter); damped steps:
        # the iris colour set and the iris rendered are not proportional (HG multiplies it with its texture)
        for i in range(3):
            f = (srgb_to_lin(res["photo"]["iris"][i]) / max(pl[i], 1e-4)) / max(
                srgb_to_lin(res["hg"]["iris"][i]) / max(rl[i], 1e-4), 1e-4)
            iris[i] = clamp(iris[i] * clamp(f, 0.6, 1.6) ** 0.7, 0.003, 0.6)
        k = (srgb_to_lin(res["photo"]["white"]) / max(lum(pl), 1e-4)) / max(
            srgb_to_lin(res["hg"]["white"]) / max(lum(rl), 1e-4), 1e-4)
        sclera = [c * clamp(k, 0.7, 1.4) ** 0.7 for c in sclera]
        # a natural warm white: never greyer (a photo's whites read dim from the lids' shade), never paper-white
        floor, ceil = [0.48, 0.42, 0.37], [0.72, 0.64, 0.57]
        sclera = [clamp(c, f, cl) for c, f, cl in zip(sclera, floor, ceil)]
        # brightness: the comparison shot's exposure follows the photo's (not the skin: that is lighting)
        ev += float(math.log2(clamp(lum(pl) / max(lum(rl), 1e-4), 0.25, 4.0)))
        # hair: relative to the skin (exposure-free); HG's hair lightness follows brightness, redness warmth
        rh = skin_of(os.path.join(dst, "face_front.png"), what="HAIR")[0]
        print(f"    photo hair {[round(c) for c in photo_hair]} | render hair {[round(c) for c in rh]}")
        ph_l = [srgb_to_lin(c) for c in photo_hair]
        rh_l = [srgb_to_lin(c) for c in rh]
        fh = (lum(ph_l) / max(lum(pl), 1e-4)) / max(lum(rh_l) / max(lum(rl), 1e-4), 1e-4)
        values["hair_lightness"] = round(clamp(values["hair_lightness"] * clamp(fh, 0.5, 2.0) ** 0.7, 0.01, 0.95), 3)
        warm = lambda c: (c[0] - c[2]) / max(c[0], 1e-4)  # noqa: E731
        values["hair_redness"] = round(clamp(values["hair_redness"] + 0.8 * (warm(photo_hair) - warm(rh)), 0, 1), 3)
        # colour strength: HG's skin saturation slider, damped and kept in a sane range
        sat = clamp(sat * clamp(sat_of(pl) / max(sat_of(rl), 1e-4), 0.7, 1.4) ** 0.6, 0.6, 1.6)
    values["iris"], values["sclera"] = lin_to_hex(iris), lin_to_hex(sclera)
    values["skin_tone"], values["skin_saturation"] = round(tone, 3), round(sat, 3)
    values["shot_exposure_ev"] = round(ev, 3)
json.dump(values, open(values_path, "w"), indent=1)

if start <= 4:
    print(f"final ({time.time() - t0:.0f}s):", values)
    final = os.path.join(out, "final")
    res = build(final, values["iris"], values["sclera"], values.get("skin_tone", 1.0),
                values.get("skin_saturation", 1.0), values.get("shot_exposure_ev", 0.0))
    print("    photo skin", [round(c) for c in skin_of(photo)[0]], "| render skin",
          [round(c) for c in skin_of(os.path.join(final, "face_front.png"))[0]])
    print(f"    photo iris {[round(c) for c in res['photo']['iris']]} white {res['photo']['white']:.0f} | "
          f"render iris {[round(c) for c in res['hg']['iris']]} white {res['hg']['white']:.0f}")
    run([PY, "-I", "-c", """
import cv2, numpy as np, sys
src = cv2.imread(sys.argv[1]); src = cv2.resize(src, (int(src.shape[1] * 800 / src.shape[0]), 800))
f = cv2.imread(sys.argv[2]); h, w = f.shape[:2]
f = cv2.resize(f[int(h * .12):int(h * .92), int(w * .08):int(w * .92)], (src.shape[1] + 120, 800))
t = cv2.resize(cv2.imread(sys.argv[3])[150:1150, 150:750], (480, 800))
cv2.imwrite(sys.argv[4], np.hstack([src, f, t]), [cv2.IMWRITE_JPEG_QUALITY, 90]); print('SHEET_OK')
""", photo, os.path.join(final, "face_front.png"), os.path.join(final, "three_quarter.png"),
         os.path.join(out, "compare.jpg")], "SHEET_OK")
print(f"PHOTO_CHARACTER_OK {out} ({time.time() - t0:.0f}s)")
