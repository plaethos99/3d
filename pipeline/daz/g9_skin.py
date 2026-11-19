"""Our person's skin on Genesis 9: their photo face baked onto Genesis 9's head texture, the whole body toned to match.

Usage: python pipeline/daz/g9_skin.py <head_dir> <name> [--sex feminine|masculine]
       [--library "C:/Users/Public/Documents/My DAZ 3D Library"]
head_dir: the face run g9_face.py used (head.blend's texture: textures/skin_colour.png, photo_coverage.png,
hair_mask.png) holding g9_<name>_fit.blend (Genesis 9's head with our face, our head lined up on it).
1. Bake (Blender, Cycles, selected to active, as face_bake.py): the photo colour, where the photos saw the skin and
   the hair mask, from our head onto Genesis 9's head tile (UDIM 1001, 4096 px) -> <head_dir>/g9_<name>_bake/.
2. Skin base: of Genesis 9's own skins of that sex (Runtime/Textures/DAZ/Characters/Genesis9/Base/<Sex>_0N), the one
   closest in colour to the photo face.
3. face_blend.py (as for HG): the photo face blended into that head texture, the whole texture shifted to the
   person's tone (--tone face); the same shift then on the body, legs and arms tiles, so the body matches the face.
4. Writes <library>/Runtime/Textures/Ours/<name>/<name>_{Head,Body,Legs,Arms}_D_100N.jpg and skin.json (which base
   skin, the tone shift) for g9_build.dsa. Prints G9_SKIN_OK.
"""
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BLENDER = os.path.join(ROOT, "tools", "blender-5.2.3-windows-x64", "blender.exe")
PY = os.path.join(ROOT, "tools", "pose-env", "Scripts", "python.exe")

BAKE = r'''
import bpy, os, sys, numpy as np
fit, tex_dir, out_dir = sys.argv[sys.argv.index("--") + 1:]
size = 4096
head, g9 = bpy.data.objects["head"], bpy.data.objects["g9"]
def emission(path, cs):
    mat = bpy.data.materials.new("src"); nt = mat.node_tree; nt.nodes.clear()
    t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(path); t.image.colorspace_settings.name = cs
    e = nt.nodes.new("ShaderNodeEmission"); o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(t.outputs["Color"], e.inputs["Color"]); nt.links.new(e.outputs["Emission"], o.inputs["Surface"])
    return mat
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.samples = 1; sc.cycles.device = "CPU"
bk = sc.render.bake
bk.use_selected_to_active = True; bk.cage_extrusion = 0.006; bk.max_ray_distance = 0.02; bk.margin = 8; bk.use_clear = True
tgt = bpy.data.materials.new("tgt"); g9.data.materials.clear(); g9.data.materials.append(tgt)
for o in bpy.context.selected_objects: o.select_set(False)
head.select_set(True); g9.select_set(True); bpy.context.view_layer.objects.active = g9
os.makedirs(out_dir, exist_ok=True)
for label, fname, cs in (("colour", "skin_colour.png", "sRGB"), ("coverage", "photo_coverage.png", "Non-Color"),
                         ("hair", "hair_mask.png", "Non-Color")):
    p = os.path.join(tex_dir, fname)
    if not os.path.exists(p):
        print("SKIP", label); continue
    head.data.materials.clear(); head.data.materials.append(emission(p, cs))
    img = bpy.data.images.new("bake_" + label, size, size, alpha=True)
    img.colorspace_settings.name = "sRGB" if label == "colour" else "Non-Color"
    img.generated_color = (0, 0, 0, 0)
    tgt.node_tree.nodes.clear()
    n = tgt.node_tree.nodes.new("ShaderNodeTexImage"); n.image = img; tgt.node_tree.nodes.active = n
    bpy.ops.object.bake(type="EMIT")
    img.filepath_raw = os.path.join(out_dir, "bake_" + label + ".png"); img.file_format = "PNG"; img.save()
    px = np.empty(size * size * 4, np.float32); img.pixels.foreach_get(px)
    print("BAKED", label, round(float((px[3::4] > 0.5).mean()) * 100, 1), "% of the head tile")
# the face region (g9_face.py's "face" attribute): baked from Genesis 9 itself, it limits the photo skin
bk.use_selected_to_active = False
for o in bpy.context.selected_objects: o.select_set(False)
g9.select_set(True); bpy.context.view_layer.objects.active = g9
tgt.node_tree.nodes.clear(); nt = tgt.node_tree
a = nt.nodes.new("ShaderNodeAttribute"); a.attribute_name = "face"
e = nt.nodes.new("ShaderNodeEmission"); o = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(a.outputs["Fac"], e.inputs["Color"]); nt.links.new(e.outputs["Emission"], o.inputs["Surface"])
img = bpy.data.images.new("bake_face", size, size, alpha=False); img.colorspace_settings.name = "Non-Color"
n = nt.nodes.new("ShaderNodeTexImage"); n.image = img; nt.nodes.active = n
bpy.ops.object.bake(type="EMIT")
img.filepath_raw = os.path.join(out_dir, "bake_face.png"); img.file_format = "PNG"; img.save()
print("G9_BAKE_OK")
'''

TONE = r'''
import sys, json, cv2, numpy as np
base_head, bake_dir = sys.argv[1], sys.argv[2]
def lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
f = cv2.imread(bake_dir + "/bake_colour.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
cov = cv2.imread(bake_dir + "/bake_coverage.png", cv2.IMREAD_UNCHANGED)[..., 0] > 128
m = (f[..., 3] > 0.5) & cov
b = cv2.imread(base_head).astype(np.float32) / 255
if b.shape[0] != f.shape[0]: b = cv2.resize(b, (f.shape[1], f.shape[0]))
want = np.median(lin(f[..., :3][m]), axis=0); have = np.median(lin(b[m]), axis=0)
d = np.log(want + 1e-4) - np.log(have + 1e-4)
print("SCORE", float(0.25 * abs(d.mean()) + 2.0 * np.abs(d - d.mean()).sum()))
'''

RETONE = r'''
import sys, cv2, numpy as np
src, dst, g = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3:6]]
def lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def srgb(c): c = np.clip(c, 0, 1); return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
im = cv2.imread(src).astype(np.float32) / 255
out = srgb(lin(im) * np.array(g[::-1]))  # BGR
cv2.imwrite(dst, (out * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
print("RETONED", dst)
'''


def opt(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


head_dir, name = os.path.abspath(sys.argv[1]), sys.argv[2]
sex = opt("--sex", "feminine").capitalize()
library = opt("--library", r"C:\Users\Public\Documents\My DAZ 3D Library")
fit = os.path.join(head_dir, f"g9_{name}_fit.blend")
bake_dir = os.path.join(head_dir, f"g9_{name}_bake")


def run(cmd, ok):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if ok not in r.stdout + r.stderr:
        print((r.stdout + r.stderr)[-2500:])
        raise SystemExit(f"G9_SKIN_FAILED at {os.path.basename(cmd[-1]) if cmd else '?'}")
    return r.stdout


# 1. bake
text = run([BLENDER, "-b", fit, "--python-expr", BAKE, "--", fit, os.path.join(head_dir, "textures"), bake_dir], "G9_BAKE_OK")
print(*[ln for ln in text.splitlines() if ln.startswith("BAKED")], sep="\n")
# the photo's coverage limited to the face region
run([PY, "-I", "-c", "import cv2,sys,numpy as np; d=sys.argv[1]; c=cv2.imread(d+'/bake_coverage.png',cv2.IMREAD_UNCHANGED); "
     "f=cv2.imread(d+'/bake_face.png',0).astype(np.float32)/255; c[...,:3]=(c[...,:3].astype(np.float32)*f[...,None]).astype(np.uint8); "
     "cv2.imwrite(d+'/bake_coverage.png',c); print('MASKED', round(float((f>0.5).mean())*100,1))", bake_dir], "MASKED")
# 2. the closest of Genesis 9's skins of that sex
skins = os.path.join(library, "Runtime", "Textures", "DAZ", "Characters", "Genesis9", "Base")
scores = []
for d in sorted(glob.glob(os.path.join(skins, f"{sex}_0*"))):
    head_d = glob.glob(os.path.join(d, "*_Head_D_1001.jpg"))
    if head_d:
        sc = float(re.search(r"SCORE (\S+)", run([PY, "-I", "-c", TONE, head_d[0], bake_dir], "SCORE")).group(1))
        scores.append((sc, d))
scores.sort()
base = scores[0][1]
print("SKIN base", os.path.basename(base), [(os.path.basename(d), round(s, 3)) for s, d in scores])
pref = glob.glob(os.path.join(base, "*_Head_D_1001.jpg"))[0].split("_Head_D_1001.jpg")[0]
# 3. face blended into the head; the same tone shift on the other tiles
out_dir = os.path.join(library, "Runtime", "Textures", "Ours", name)
os.makedirs(out_dir, exist_ok=True)
head_png = os.path.join(out_dir, f"{name}_Head_D_1001.png")
text = run([PY, "-I", os.path.join(ROOT, "pipeline", "humgen", "face_blend.py"), bake_dir, pref + "_Head_D_1001.jpg",
            head_png, "--tone", "face"], "FACE_BLEND_OK")
gain = json.loads(re.search(r"TONE gain \(photo / HG, linear R G B\): (\[[^\]]+\])", text).group(1))
print("TONE gain (linear R G B)", gain)
run([PY, "-I", "-c", "import cv2,sys; cv2.imwrite(sys.argv[2], cv2.imread(sys.argv[1]), [cv2.IMWRITE_JPEG_QUALITY, 92]); print('OK')",
     head_png, head_png[:-4] + ".jpg"], "OK")
os.remove(head_png)
maps = {"Head": head_png[:-4] + ".jpg"}
for tile, num in (("Body", 1002), ("Legs", 1003), ("Arms", 1004)):
    src = f"{pref}_{tile}_D_{num}.jpg"
    dst = os.path.join(out_dir, f"{name}_{tile}_D_{num}.jpg")
    run([PY, "-I", "-c", RETONE, src, dst, *map(str, gain)], "RETONED")
    maps[tile] = dst
json.dump({"base_skin": os.path.basename(base), "base_prefix": pref, "tone_gain_linear_rgb": gain, "maps": maps},
          open(os.path.join(out_dir, "skin.json"), "w"), indent=1)
print("G9_SKIN_OK", out_dir)
