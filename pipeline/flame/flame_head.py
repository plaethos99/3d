"""FLAME heads (flame_faces.py's heads.npz) into head.blend + subject_fit.json each, as face_capture.py takes them.

Usage: blender -b -P pipeline/flame/flame_head.py -- <heads.npz> <out_dir>
heads.npz: faces (F,3), names (N,), verts (N,V,3) in FLAME's frame (metres, y up, facing +z).
Each head -> <out_dir>/<name>/head.blend (objects "head", "eye.L", "eye.R"; facing -y, z up, like HG) and
subject_fit.json (ids 0..467, points: the MediaPipe landmarks' FLAME vertices, reliability 1).
Which FLAME vertex each landmark is: found once on the first head (the mean) by rendering it straight on,
running MediaPipe on the render and tracing each landmark back onto the mesh (hg_landmarks' method), cached in
<out_dir>/flame_landmarks.json; every FLAME head shares the mesh, so it holds for all. Prints FLAME_HEAD_OK.
"""
import json
import os
import subprocess
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "humgen"))
from hg_landmarks import DETECT, MODEL, PY  # noqa: E402

src, out = (os.path.abspath(p) for p in sys.argv[sys.argv.index("--") + 1:][:2])
os.makedirs(out, exist_ok=True)
data = np.load(src)
F, names, V = data["faces"], [str(n) for n in data["names"]], data["verts"]


def to_blender(v):
    return np.c_[v[:, 0], -v[:, 2], v[:, 1]]


# pieces: the head and the two eyeballs are separate pieces of FLAME's mesh
parent = np.arange(V.shape[1])


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


for f in F:
    for b in f[1:]:
        ra, rb = find(f[0]), find(b)
        if ra != rb:
            parent[ra] = rb
roots = np.array([find(i) for i in range(len(parent))])
sizes = {r: int((roots == r).sum()) for r in set(roots.tolist())}
order = sorted(sizes, key=lambda r: -sizes[r])
head_r, eye_rs = order[0], order[1:3]
print("PIECES", [sizes[r] for r in order[:4]])


def piece(P, r):
    keep = np.where(roots == r)[0]
    remap = -np.ones(len(P), int)
    remap[keep] = np.arange(len(keep))
    fs = [tuple(remap[f]) for f in F if roots[f[0]] == r]
    return P[keep], fs, keep


def make(name, P, fs):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in P], [], fs)
    me.update()
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def landmark_vertices(P, fs, work):
    """MediaPipe landmarks of a straight-on render of this head, traced back onto it."""
    sc = bpy.context.scene
    ob = make("lm_head", P, fs)
    mat = bpy.data.materials.new("lm_skin")
    mat.diffuse_color = (0.8, 0.62, 0.52, 1)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.8, 0.62, 0.52, 1)
    ob.data.materials.append(mat)
    zc = P[:, 2].max() - 0.12  # about the eyes' height, below the top of the head
    c = np.array([P[:, 0].mean(), P[:, 1].min(), zc])
    scale, res = 0.30, 1024
    cam = bpy.data.objects.new("lm_cam", bpy.data.cameras.new("lm_cam"))
    sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = scale
    cam.location = Vector((c[0], c[1] - 1.0, c[2]))
    cam.rotation_euler = (Vector(c) - cam.location).to_track_quat("-Z", "Y").to_euler()
    light = bpy.data.objects.new("lm_light", bpy.data.lights.new("lm_light", "SUN"))
    sc.collection.objects.link(light)
    light.rotation_euler = (1.3, 0, 0.2)
    light.data.energy = 3
    sc.camera = cam
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x = sc.render.resolution_y = res
    img = os.path.join(work, "flame_mean_front.png")
    sc.render.filepath = img
    bpy.ops.render.render(write_still=True)
    for o in (cam, light, ob):
        bpy.data.objects.remove(o)
    r = subprocess.run([PY, "-I", "-c", DETECT, MODEL, img], capture_output=True, text=True)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("LANDMARKS")), None)
    pts2 = json.loads(line.split(" ", 1)[1]) if line else []
    if not pts2:
        raise SystemExit("FLAME_HEAD_FAILED: MediaPipe found no face on the render " + r.stderr[-300:])
    tree = BVHTree.FromPolygons([tuple(p) for p in P], fs)
    found = []
    for x, y in pts2:
        wx = c[0] + (x / res - 0.5) * scale
        wz = c[2] - (y / res - 0.5) * scale
        hit = tree.ray_cast(Vector((wx, c[1] - 1.0, wz)), Vector((0, 1, 0)))
        if hit[0] is None:
            found.append(-1)
            continue
        poly = fs[hit[2]]
        found.append(int(min(poly, key=lambda vi: np.linalg.norm(P[vi] - np.array(hit[0])))))
    return found


for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
cache = os.path.join(out, "flame_landmarks.json")
lm = None
for name, v in zip(names, V):
    P = to_blender(v)
    HP, hf, _ = piece(P, head_r)
    if lm is None:
        lm = json.load(open(cache))["landmarks"] if os.path.exists(cache) else landmark_vertices(HP, hf, out)
        json.dump({"landmarks": lm}, open(cache, "w"))
        print("LANDMARKS found on FLAME:", sum(1 for i in lm if i >= 0), "of", len(lm))
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    make("head", HP, hf)
    eyes = [piece(P, r) for r in eye_rs]
    eyes.sort(key=lambda e: -e[0][:, 0].mean())  # character's left (+x) first
    for label, (EP, ef, _) in zip(("eye.L", "eye.R"), eyes):
        make(label, EP, ef)
    ids = [i for i in range(468) if lm[i] >= 0]
    fit = {"ids": ids, "points": HP[[lm[i] for i in ids]].tolist(), "reliability": [1.0] * len(ids),
           "source": "FLAME 2023 Open (CC-BY-4.0)"}
    json.dump(fit, open(os.path.join(d, "subject_fit.json"), "w"))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(d, "head.blend"))
print("FLAME_HEAD_OK", len(names), "heads in", out)
