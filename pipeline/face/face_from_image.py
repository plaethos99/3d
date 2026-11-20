"""Step 1: a high-quality 3D head from one face photo (or a head-turn clip), by script, start to finish.

Usage: python pipeline/face/face_from_image.py <photo.jpg | clip.mp4> <out_dir> [--from step] [--only step] [--strands]
       [--seconds 6] [--frames-step 12] [--texture 2048] [--clip grok|flow] [--views clip|stills]
Steps (each one script; --from / --only pick where to start or one to run):
  clip     the photo brought to life by Grok Imagine (engine/imagegen/imagegen.mjs video): the person turns
           the head slowly to each side and back, so the face is seen from every angle. Skipped for a clip.
  scan     scan_views.py: the face tracker (MediaPipe, 478 points) on every frame, with head angles
  fit      fit_subject.py: the face measured in 3D from the frames (a photogrammetry-style solve), starting
           from the sculpted base head's points
  shape    reshape_head.py: the base head (Blender Studio's realistic head, sculpted detail kept) moved onto
           the measured points
  eyes     add_eyes.py: eyeballs sized and placed from the measured eyelids
  frames   select_frames.py + solve_view_cameras.py: the sharpest neutral frame per angle, and its camera
  texture  bake_maps.py + project_texture.py: the frames painted on the head (backdrop and clothes cut out,
           the eyes kept off the skin, hair only on the scalp, unseen parts filled), then skin_material.py: skin
           with light under it
  masks    frame_masks.py: hair, skin and backdrop in each chosen frame, and the hair's colour
  hairvol  carve_hair_frames.py: the hair's volume carved from the frames (object "hair", hidden in renders)
  groom    groom_head.py (only with --strands): strands grown in that volume, combed, coloured like the frames'
           hair. A 6 s turn clip sees too little of the hair (never the back) for a clean volume, so by default
           the head has none: hair comes with the character (a hairstyle from a library, in the frames' colour);
           the carved volume is still used to paint the scalp under the hair
  check    render_at_views.py + compare_views.py: the head rendered from each chosen frame's camera, beside the
           frame, with the face tracker's point error on both (as a share of the eye-corner distance)
Writes <out_dir>/: turn.mp4, scan/, subject_fit.json, head_shaped.blend, head_eyes.blend, texture_frames.txt,
texture_cameras.json, bake/, textures/, head.blend (the result), renders/, compare.png; prints each step's time.
Base head assets (made once by the replica-head work): tests/realhead/output/base_head.blend, anchors.json and
tests/realhead/base_views/.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLENDER = os.path.join(ROOT, "tools", "blender-5.2.3-windows-x64", "blender.exe")
PY = os.path.join(ROOT, "tools", "pose-env", "Scripts", "python.exe")
RH = os.path.join(ROOT, "tests", "realhead")
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(RH, "scripts")
MODEL = os.path.join(ROOT, "tools", "face-model", "face_landmarker.task")
IMAGEGEN = os.path.join(ROOT, "engine", "imagegen", "imagegen.mjs")
STEPS = ["clip", "scan", "fit", "shape", "eyes", "frames", "masks", "hairvol", "texture", "groom", "check"]
TURN = ("The person slowly turns their head to their left until the face is in full profile, then slowly turns to "
        "their right until the other side of the face is in full profile, then back to facing the camera. Neutral "
        "expression, mouth closed, eyes open, hair still. Static locked-off camera, steady soft studio lighting, "
        "plain light background.")


def run(args, env=None, keep=("_OK", "PICK", "FIT", "ERROR", "error", "Error", "Traceback", "SOLVE", "POINT")):
    done = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, **(env or {})})
    for line in done.stdout.splitlines():
        if line.startswith(keep) or "Traceback" in line:
            print("   ", line[:240])
    if done.returncode != 0:
        print(done.stdout[-1500:], done.stderr[-1500:])
        raise SystemExit(f"step failed: {' '.join(os.path.basename(str(a)) for a in args[:4])}")
    return done.stdout


def main():
    argv = sys.argv[1:]

    def opt(name, default):
        if name in argv:
            i = argv.index(name)
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        return default

    start, only = opt("--from", "clip"), opt("--only", None)
    strands = "--strands" in argv
    if strands:
        argv.remove("--strands")
    seconds, step_deg, tex = opt("--seconds", "6"), opt("--frames-step", "12"), opt("--texture", "2048")
    clip_service = opt("--clip", "grok")  # who makes the head-turn clip: grok or flow (Veo, 8 s clips)
    # --views stills: instead of a clip, turned stills of the person (edits of the photo by --clip's service, about
    # one megapixel each, sharper than a clip's frames): the photo itself (front), left45 and left75, and their
    # mirror images for the right side. Each still is one "video" of one frame to the steps below.
    use_stills = opt("--views", "clip") == "stills"
    if clip_service == "flow" and seconds == "6":
        seconds = "8"
    source, out = os.path.abspath(argv[0]), os.path.abspath(argv[1])
    os.makedirs(out, exist_ok=True)
    clip = os.path.join(out, "turn.mp4")
    scan, fit = os.path.join(out, "scan"), os.path.join(out, "subject_fit.json")
    views = os.path.join(scan, "views.json")
    cams = os.path.join(out, "texture_cameras.json")
    steps = [only] if only else [s for s in STEPS[STEPS.index(start):] if s != "groom" or strands]
    t_all = time.time()
    for step in steps:
        t = time.time()
        print(f"== {step}")
        if step == "clip" and use_stills:
            import shutil
            if not os.path.exists(os.path.join(out, "left45.png")):
                res = run(["node", os.path.join(ROOT, "engine", "imagegen", "face_views.mjs"), clip_service, source, out,
                           "left45,left75"], keep=("{",))
                if '"ok":true' not in res:
                    raise SystemExit("the turned views weren't made: " + res[-400:])
            shutil.copy(source, os.path.join(out, "front.png"))
            run([PY, "-I", "-c", "import cv2, sys, os\n"
                 "for a, b in (('left45', 'right45'), ('left75', 'right75')):\n"
                 "    im = cv2.imread(os.path.join(sys.argv[1], a + '.png'))\n"
                 "    cv2.imwrite(os.path.join(sys.argv[1], b + '.png'), im[:, ::-1])\n"
                 "print('MIRROR_OK')", out], keep=("MIRROR_OK",))
        elif step == "clip":
            if source.lower().endswith((".mp4", ".mov", ".webm")):
                if source != clip:
                    import shutil
                    shutil.copy(source, clip)
            else:
                res = run(["node", IMAGEGEN, "video", clip_service, "3:4", seconds, source, clip, TURN])
                if '"ok":true' not in res:
                    raise SystemExit("the clip wasn't made: " + res[-400:])
        elif step == "scan":
            inputs = [os.path.join(out, f"{v}.png") for v in ("front", "left45", "left75", "right45", "right75")] \
                if use_stills else [clip]
            run([PY, "-I", os.path.join(SCRIPTS, "scan_views.py"), MODEL, scan, *inputs])
        elif step == "fit":
            run([PY, "-I", os.path.join(SCRIPTS, "fit_subject.py"), os.path.join(RH, "output", "anchors.json"),
                 os.path.join(RH, "base_views"), views, fit])
        elif step == "shape":
            run([BLENDER, "-b", os.path.join(RH, "output", "base_head.blend"), "-P",
                 os.path.join(SCRIPTS, "reshape_head.py"), "--", fit, os.path.join(out, "head_shaped.blend")])
        elif step == "eyes":
            run([BLENDER, "-b", os.path.join(out, "head_shaped.blend"), "-P", os.path.join(SCRIPTS, "add_eyes.py"),
                 "--", fit, os.path.join(out, "head_eyes.blend")])
        elif step == "frames":
            picks = os.path.join(out, "texture_frames.txt")
            run([PY, "-I", os.path.join(SCRIPTS, "select_frames.py"), views, step_deg, picks])
            frames = [ln.strip() for ln in open(picks, encoding="utf-8") if ln.strip()]
            run([PY, "-I", os.path.join(SCRIPTS, "solve_view_cameras.py"), fit, views, cams, *frames])
        elif step == "masks":
            run([PY, "-I", os.path.join(HERE, "frame_masks.py"), cams, views, out, os.path.join(out, "masks")],
                keep=("MASKS_OK",))
        elif step == "hairvol":
            run([BLENDER, "-b", os.path.join(out, "head_eyes.blend"), "-P", os.path.join(HERE, "carve_hair_frames.py"),
                 "--", cams, os.path.join(out, "masks"), os.path.join(out, "head_eyes.blend")],
                keep=("HULL", "CARVE", "UNSEEN", "COVER", "HAIR"))
        elif step == "texture":
            bake, textures = os.path.join(out, "bake"), os.path.join(out, "textures")
            run([BLENDER, "-b", os.path.join(out, "head_eyes.blend"), "-P", os.path.join(SCRIPTS, "bake_maps.py"),
                 "--", cams, bake, tex, "1.0"])
            run([PY, "-I", os.path.join(SCRIPTS, "project_texture.py"), bake, cams, out, fit, textures],
                env={"SUBJECT_MASK": "auto", "EYE_MASK_VIEWS": views, "HAIR_MASKS": os.path.join(out, "masks"),
                     "HAIR_VOXELS": os.path.join(out, "hair_voxels.npz")})
            run([BLENDER, "-b", os.path.join(out, "head_eyes.blend"), "-P", os.path.join(SCRIPTS, "skin_material.py"),
                 "--", textures, os.path.join(out, "head.blend")])
        elif step == "groom":
            run([PY, "-I", os.path.join(HERE, "groom_head.py"), os.path.join(out, "head.blend"),
                 os.path.join(out, "masks", "hair.json"), os.path.join(out, "head.blend")],
                keep=("STRAND", "GROOM", "CLAMPED", "REACH", "ROOTS", "GUIDES"))
        elif step == "check":
            renders = os.path.join(out, "renders")
            run([BLENDER, "-b", os.path.join(out, "head.blend"), "-P", os.path.join(SCRIPTS, "render_at_views.py"),
                 "--", cams, renders, "1.0", "24"])
            run([PY, "-I", os.path.join(SCRIPTS, "compare_views.py"), MODEL, cams, renders, out,
                 os.path.join(out, "compare.png")])
        print(f"   {step}: {time.time() - t:.0f} s")
    print(f"FACE_OK {out} in {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
