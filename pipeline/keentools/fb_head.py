"""A FaceBuilder head from a person's photos (run in Blender 4.2 LTS, where FaceBuilder's core loads).

Usage: tools/blender-4.2.23-windows-x64/blender.exe -b -P pipeline/keentools/fb_head.py -- <out_dir> <image> [<image> ...]
       [--tex 2048]
Images: the same person from several sides (face_from_image.py's stills: front, left45, left75, right45, right75).
FaceBuilder adds a head, loads every image as a view, and on each one finds the face and pins its features
automatically ("align face": its face detector, then its preset pins, solved); the head's shape is solved
over all views together. Then its texture is baked from the views.
Writes <out_dir>/fb_head.blend (the head object, the view cameras) and fb_head.json (per view: pinned or not,
pins, focal), and fb_texture.png. Prints FB_HEAD_OK. FaceBuilder is a licensed KeenTools product (trial or the
user's licence); its files stay in tools/ and out of the repo.
"""
import json
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
tex = int(argv[argv.index("--tex") + 1]) if "--tex" in argv else 2048
argv = [a for i, a in enumerate(argv) if a != "--tex" and (i == 0 or argv[i - 1] != "--tex")]
out = os.path.abspath(argv[0])
images = [os.path.abspath(p) for p in argv[1:]]
os.makedirs(out, exist_ok=True)

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
r = bpy.ops.keentools_fb.add_head()
print("ADD_HEAD", r)
from keentools.facebuilder.fbloader import FBLoader  # noqa: E402
from keentools.addon_config import fb_settings  # noqa: E402

settings = fb_settings()
head = settings.get_head(0)
print("HEAD object", head.headobj.name)
# all images from one folder (the operator takes a folder and file names): link them into one if needed
dirs = {os.path.dirname(p) for p in images}
if len(dirs) != 1:
    raise SystemExit("FB_HEAD_FAILED: put the images in one folder")
r = bpy.ops.keentools_fb.open_multiple_filebrowser(headnum=0, directory=dirs.pop() + os.sep,
                                                   files=[{"name": os.path.basename(p)} for p in images])
print("LOAD_IMAGES", r, len(head.cameras), "views")
report = {"views": []}
for camnum, path in enumerate(images):
    r = bpy.ops.keentools_fb.pickmode_starter(headnum=0, camnum=camnum, auto_detect_single=True)
    cam = head.get_camera(camnum)
    pins = cam.pins_count if hasattr(cam, "pins_count") else None
    print("ALIGN", os.path.basename(path), r, "pins", pins)
    report["views"].append({"image": os.path.basename(path), "result": list(r), "pins": pins,
                            "focal": getattr(cam, "focal", None)})
FBLoader.save_fb_serial_and_image_pathes(0)
if "--no-tex" not in argv:
    try:
        head.tex_width = head.tex_height = tex
    except Exception:  # noqa: BLE001
        pass
    r = bpy.ops.keentools_fb.bake_tex(headnum=0)
    print("BAKE_TEX", r)
    img = next((i for i in bpy.data.images if i.name.startswith(head.preview_texture_name())), None)
    if img is not None:
        img.filepath_raw = os.path.join(out, "fb_texture.png")
        img.file_format = "PNG"
        img.save()
        report["texture"] = "fb_texture.png"
json.dump(report, open(os.path.join(out, "fb_head.json"), "w"), indent=1)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "fb_head.blend"))
print("FB_HEAD_OK", out, sum(1 for v in report["views"] if v["pins"]), "of", len(images), "views pinned")
