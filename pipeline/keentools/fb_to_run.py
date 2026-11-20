"""A FaceBuilder head (fb_head.py) as a face run, like face_from_image.py's, so photo_character.py takes it as is.

Usage: blender -b <out>/fb_head.blend -P pipeline/keentools/fb_to_run.py -- <run_dir> --like <face_run>
       (Blender 5.2: it opens FaceBuilder's 4.2 file; FaceBuilder itself isn't needed here)
<face_run>: the person's own face_from_image.py run (its masks/hair.json and front.png are copied; its measured
eye-corner distance sets the head's size: FaceBuilder works in its own units).
Writes <run_dir>/: head.blend ("head" with FaceBuilder's texture layout, "eye.L", "eye.R": eyeballs placed behind
the irises MediaPipe finds), subject_fit.json (the 468 landmarks on the head, as face_capture.py reads them),
textures/skin_colour.png (FaceBuilder's baked texture), photo_coverage.png (where it has photo), hair_mask.png
(dark, grey texels: hair the photos put on the head), masks/hair.json, front.png. Prints FB_RUN_OK.
"""
import json
import os
import shutil
import sys

import bpy
import subprocess
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "humgen"))
from hg_landmarks import surface_landmarks  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
run = os.path.abspath(argv[0])
like = os.path.abspath(argv[argv.index("--like") + 1])
os.makedirs(os.path.join(run, "textures"), exist_ok=True)
os.makedirs(os.path.join(run, "masks"), exist_ok=True)
fb_dir = os.path.dirname(bpy.data.filepath)

src = bpy.data.objects["FBHead"]
dg = bpy.context.evaluated_depsgraph_get()
me = bpy.data.meshes.new_from_object(src.evaluated_get(dg))
me.transform(src.matrix_world)
P = np.array([v.co[:] for v in me.vertices])
fs = [tuple(p.vertices) for p in me.polygons]
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)

# landmarks in FaceBuilder's units, then the size: the person's measured eye-corner distance (33 / 263)
ids, hits = surface_landmarks(P, fs, run, tag="fb")
fit0 = json.load(open(os.path.join(like, "subject_fit.json")))
pts0 = dict(zip(fit0["ids"], np.array(fit0["points"])))
want = np.linalg.norm(pts0[33] - pts0[263]) if 33 in pts0 and 263 in pts0 else 0.09
have = np.linalg.norm(hits[33] - hits[263])
k = float(want / have)
print("SCALE FaceBuilder units -> m: %.4f (eye corners %.1f mm)" % (k, want * 1000))
P, hits = P * k, hits * k
me.transform(np.diag([k, k, k, 1]).tolist())
head = bpy.data.objects.new("head", me)
bpy.context.scene.collection.objects.link(head)
# eyeballs: 12 mm spheres behind the irises (468 / 473 are the iris centres)
for label, i in (("eye.L", 473), ("eye.R", 468)):
    centre = hits[i] + np.array([0, 0.012, 0])
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.012, location=Vector(centre), segments=24, ring_count=12)
    bpy.context.active_object.name = label
if bpy.data.objects["eye.L"].location.x < bpy.data.objects["eye.R"].location.x:  # character's left is +x
    bpy.data.objects["eye.L"].name, bpy.data.objects["eye.R"].name = "tmp", "eye.L"
    bpy.data.objects["tmp"].name = "eye.R"
ok = [i for i in range(468) if ids[i] >= 0]
json.dump({"ids": ok, "points": hits[ok].tolist(), "reliability": [1.0] * len(ok),
           "source": "KeenTools FaceBuilder head"}, open(os.path.join(run, "subject_fit.json"), "w"))
# textures in the head's own layout
ROOT = os.path.dirname(os.path.dirname(HERE))
r = subprocess.run([os.path.join(ROOT, "tools", "pose-env", "Scripts", "python.exe"), "-I",
                    os.path.join(HERE, "fb_textures.py"), os.path.join(fb_dir, "fb_texture.png"),
                    os.path.join(run, "textures")], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-500:])
mat = bpy.data.materials.new("fb_skin")
mat.use_nodes = True
nt = mat.node_tree
im = nt.nodes.new("ShaderNodeTexImage")
im.image = bpy.data.images.load(os.path.join(run, "textures", "skin_colour.png"))
nt.links.new(im.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
head.data.materials.clear()
head.data.materials.append(mat)
for name in ("front.png",):
    if os.path.exists(os.path.join(like, name)):
        shutil.copy(os.path.join(like, name), os.path.join(run, name))
shutil.copy(os.path.join(like, "masks", "hair.json"), os.path.join(run, "masks", "hair.json"))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(run, "head.blend"))
print("FB_RUN_OK", run, len(ok), "landmarks")
