"""One of our faces as a Genesis 9 face morph (a Daz Studio slider), straight from the files: no Daz scripting.

Usage: blender -b -P pipeline/daz/g9_face.py -- <head_dir> <name> [--label "Face: Teacher (ours)"]
       [--library "C:/Users/Public/Documents/My DAZ 3D Library"] [--surface 0.7]
head_dir: one of our face runs (head.blend with "head", subject_fit.json): FaceBuilder's (keentools/fb_to_run.py)
or the FLAME photo fit's (flame/flame_head.py).
1. Genesis 9's base mesh from its DSON file (data/Daz 3D/Genesis 9/Base/Genesis9.dsf; cm, y up, facing +z).
2. Its MediaPipe landmarks (hg_landmarks.surface_landmarks: a straight-on render, traced back onto the mesh).
3. Our head lined up on it (similarity on the landmarks), Genesis 9's head moved so its landmarks land on ours
   (biharmonic interpolation, as face_capture.py does for HG: outliers and the inner lip line left out, moves
   smoothed over the mesh), then a light pull onto our head's surface. Only the head group moves, fading out
   down the neck (neck2 half, neck1 a little).
4. Written as a DSON morph into Genesis 9's morph folder: <library>/data/Daz 3D/Genesis 9/Base/Morphs/Ours/Faces/
   <name>.dsf. Daz Studio lists it on Genesis 9 under Head/Ours ("--label"). The morph carries the rig too:
   formulas move every head and neck joint (eyes, jaw, teeth, brows, ...) with the skin around it, so the eyes and
   mouth (separate figures on the rig) follow the new head with no "Adjust Rigging to Shape".
Also renders a clay check (ours vs Genesis 9 with the morph) into <head_dir>/g9_<name>_front/tq.png and saves
<head_dir>/g9_<name>_fit.blend (Genesis 9's head with the morph and its texture layout, our head lined up on it)
for g9_skin.py.
Prints G9_FACE_OK.
"""
import json
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "humgen"))
from hg_landmarks import rbf_moves, similarity, surface_landmarks  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
head_dir, name = os.path.abspath(argv[0]), argv[1]


def opt(k, default):
    return argv[argv.index(k) + 1] if k in argv else default


label = opt("--label", f"Face: {name} (ours)")
library = opt("--library", r"C:\Users\Public\Documents\My DAZ 3D Library")
surface = float(opt("--surface", "0.7"))
G9 = os.path.join(library, "data", "Daz 3D", "Genesis 9", "Base", "Genesis9.dsf")

# 1. Genesis 9's base mesh, in Blender's frame (m, z up, facing -y)
d = json.load(open(G9))
g = d["geometry_library"][0]
V_cm = np.array(g["vertices"]["values"], float)
P = np.c_[V_cm[:, 0], -V_cm[:, 2], V_cm[:, 1]] / 100.0
polys = g["polylist"]["values"]
groups = g["polygon_groups"]["values"]
fs = [tuple(p[2:]) for p in polys]
gw = {"head": 1.0, "neck2": 0.5, "neck1": 0.15}
w = np.zeros(len(P))
for p in polys:
    gwt = gw.get(groups[p[0]], 0.0)
    for vi in p[2:]:
        w[vi] = max(w[vi], gwt)
nb = [[] for _ in range(len(P))]
for f in fs:
    for a, b in zip(f, f[1:] + f[:1]):
        nb[a].append(b)
        nb[b].append(a)
for _ in range(6):  # a soft fade between groups
    w = np.array([max(w[i], 0.6 * np.mean(w[nb[i]])) if nb[i] and w[i] < 1 else w[i] for i in range(len(P))])
region = np.where(w > 0)[0]
print("GENESIS 9", len(P), "points;", len(region), "move (head and neck)")

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
# 2. Genesis 9's landmarks (only the head: arms out at the sides would widen the framing)
head_polys = [f for p, f in zip(polys, fs) if groups[p[0]] in ("head", "neck1", "neck2")]
used = sorted({v for f in head_polys for v in f})
remap = {v: i for i, v in enumerate(used)}
ids_g9, hits_g9 = surface_landmarks(P[used], [tuple(remap[v] for v in f) for f in head_polys], head_dir, tag="g9")
lm_g9 = np.array([used[i] if i >= 0 else -1 for i in ids_g9])

# 3. our head on Genesis 9
with bpy.data.libraries.load(os.path.join(head_dir, "head.blend")) as (src, dst):
    dst.objects = ["head"]
head = dst.objects[0]
bpy.context.scene.collection.objects.link(head)
Mh = np.array(head.matrix_world)
S = np.array([Mh[:3, :3] @ np.array(v.co) + Mh[:3, 3] for v in head.data.vertices])
fit = json.load(open(os.path.join(head_dir, "subject_fit.json")))
ids = np.array(fit["ids"])
Ls = (Mh[:3, :3] @ np.array(fit["points"]).T).T + Mh[:3, 3]
INNER_LIPS = {78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191}
ok = (ids < 468) & (lm_g9[np.minimum(ids, 467)] >= 0) & ~np.isin(ids, list(INNER_LIPS))
ids, Ls = ids[ok], Ls[ok]
Lh = P[lm_g9[ids]]
keep = np.ones(len(ids), bool)
for _ in range(3):
    s_, R_, t_ = similarity(Ls[keep], Lh[keep])
    err = np.linalg.norm((s_ * (R_ @ Ls.T)).T + t_ - Lh, axis=1)
    keep = err < max(0.012, 3 * np.median(err[keep]))
ids, Ls, Lh = ids[keep], Ls[keep], Lh[keep]
s_, R_, t_ = similarity(Ls, Lh)
D = (s_ * (R_ @ Ls.T)).T + t_ - Lh
near = np.argsort(np.linalg.norm(Lh[:, None] - Lh[None], axis=2), axis=1)[:, 1:9]
dev = np.linalg.norm(D - np.median(D[near], axis=1), axis=1)
good = dev < max(0.002, float(np.median(dev) + 4 * np.median(np.abs(dev - np.median(dev)))))
Lh, D = Lh[good], D[good]
print("LANDMARKS", len(Lh), "used; our head -> Genesis 9 scale %.3f; moves median %.1f mm, largest %.1f mm" % (
    s_, np.median(np.linalg.norm(D, axis=1)) * 1000, np.linalg.norm(D, axis=1).max() * 1000))
moves = np.zeros_like(P)
moves[region] = rbf_moves(Lh, D, P[region])
S_al = (s_ * (R_ @ S.T)).T + t_
if surface > 0:  # a light pull onto our head's surface (shape between the landmarks)
    bm = bmesh.new()
    bm.from_mesh(head.data)
    for v, p in zip(bm.verts, S_al):
        v.co = Vector(p)
    bm.normal_update()
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    Q = P + moves
    extra = np.zeros_like(P)
    for i in region:
        if w[i] < 1:
            continue
        loc, nrm, _, dist = tree.find_nearest(Vector(Q[i]), 0.01)
        if loc is not None:
            extra[i] = (np.array(loc) - Q[i]) * surface
    moves += extra
moves *= w[:, None]
for _ in range(8):  # smooth over the mesh: the broad shape stays, notches go
    new = moves.copy()
    for i in region:
        if nb[i]:
            new[i] = 0.5 * moves[i] + 0.5 * moves[nb[i]].mean(0)
    moves = new
print("MOVED median %.1f mm, largest %.1f mm" % (np.median(np.linalg.norm(moves[region], axis=1)) * 1000,
                                                  np.linalg.norm(moves, axis=1).max() * 1000))

# 3b. likeness against the portrait itself (the 3D fit can leave the face's proportions a few % off, e.g. eyes opened
# wider, eye corners closer, a heavier jaw): the morphed head's landmarks on a straight-on clay render vs the
# portrait's, lined up by scale and position only; what differs is shape, moved in the picture's plane (damped)
photo = os.path.join(head_dir, "front.png")
like_rounds = int(opt("--likeness", "0"))  # off by default: the owner preferred the plain fit
if like_rounds and os.path.exists(photo):
    import subprocess
    import tempfile
    PYENV = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "pose-env", "Scripts", "python.exe")
    lj = os.path.join(tempfile.gettempdir(), f"g9_like_{name}.json")
    subprocess.run([PYENV, "-I", os.path.join(HERE, "landmarks2d.py"), lj, photo], capture_output=True, text=True)
    q = json.load(open(lj)).get(photo)
    if q is None:
        print("LIKENESS skipped: no face found on", photo)
    else:
        Q2 = np.array(q)[:468] * np.array([1.0, -1.0])  # picture y down -> up
        hfs = [tuple(remap[v] for v in f) for f in head_polys]

        def sim2(a, b):  # similarity a -> b (2D, Umeyama)
            ma, mb = a.mean(0), b.mean(0)
            A, B = a - ma, b - mb
            U, sv, Vt = np.linalg.svd(B.T @ A)
            dd = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
            R2 = U @ dd @ Vt
            s2 = (sv * np.diag(dd)).sum() / (A ** 2).sum()
            return s2, R2, mb - s2 * (R2 @ ma)

        IOD = (33, 263)
        best = (np.inf, moves.copy())
        for rnd in range(like_rounds + 1):
            ids_m, hits_m = surface_landmarks((P + moves)[used], hfs, head_dir, tag="g9like")
            ok2 = (ids_m[:468] >= 0) & ~np.isin(np.arange(468), list(INNER_LIPS))
            A3 = hits_m[:468][ok2]
            A2 = A3[:, [0, 2]]
            s2, R2, t2 = sim2(Q2[ok2], A2)
            T2 = (s2 * (R2 @ Q2[ok2].T)).T + t2
            res = T2 - A2
            # a landmark against its neighbours' median: detection noise goes, a region's shift stays
            nn = np.argsort(np.linalg.norm(A2[:, None] - A2[None], axis=2), axis=1)[:, :9]
            res = np.median(res[nn], axis=1)
            iod = np.linalg.norm(hits_m[IOD[0]] - hits_m[IOD[1]])
            mm = np.linalg.norm(res, axis=1) * 1000
            print("LIKENESS round %d: %d landmarks, off by median %.2f mm, largest %.2f mm (%.1f%% of the eye span)" % (
                rnd, len(A2), np.median(mm), mm.max(), 100 * np.median(mm) / 1000 / iod), flush=True)
            if np.median(mm) < best[0]:
                best = (float(np.median(mm)), moves.copy())
            elif np.median(mm) > 1.15 * best[0]:  # worse (a bad detection or an overshoot): back to the best shape
                print("LIKENESS back to the best round (median %.2f mm)" % best[0])
                break
            if np.median(mm) < 0.35 or rnd == like_rounds:
                break
            D3 = np.zeros_like(A3)
            D3[:, 0], D3[:, 2] = res[:, 0], res[:, 1]
            n3 = np.linalg.norm(D3, axis=1, keepdims=True)
            D3 = D3 * np.minimum(1, 0.005 / np.maximum(n3, 1e-9)) * 0.7  # at most 5 mm a round, damped
            step = np.zeros_like(P)
            step[region] = rbf_moves(A3, D3, (P + moves)[region])
            step *= w[:, None]
            for _ in range(6):
                new = step.copy()
                for i in region:
                    if nb[i]:
                        new[i] = 0.5 * step[i] + 0.5 * step[nb[i]].mean(0)
                step = new
            moves += step
        moves = best[1]

# 4. the DSON morph (Daz's frame: cm, y up, facing +z)
dm = np.c_[moves[:, 0], moves[:, 2], -moves[:, 1]] * 100.0
nz = np.where(np.linalg.norm(dm, axis=1) > 1e-4)[0]
mid = "OURS_" + "".join(c if c.isalnum() else "_" for c in name)
rel = "/data/Daz%203D/Genesis%209/Base/Morphs/Ours/Faces/" + mid + ".dsf"
# the rig follows the face (Daz's own character morphs do this, so no "Adjust Rigging to Shape" is needed): every
# joint of the head and neck moves its centre and end point with the skin around it (inverse-distance average
# of the 16 nearest skin points' moves); written as formulas the slider drives
from mathutils.kdtree import KDTree  # noqa: E402

kd = KDTree(len(V_cm))
for i, v in enumerate(V_cm):
    kd.insert(Vector(v), i)
kd.balance()
par = {n["id"]: n.get("parent", "").split("#")[-1] for n in d["node_library"]}


def under(i, roots=("head", "neck1", "neck2")):
    while i:
        if i in roots:
            return True
        i = par.get(i)
    return False


G9URL = "/data/Daz%203D/Genesis%209/Base/Genesis9.dsf"
formulas = []
rig_moves = {}
for n in d["node_library"]:
    if n.get("type") != "bone" or not under(n["id"]):
        continue
    for key in ("center_point", "end_point"):
        pt = np.array([c["value"] for c in n[key]], float)
        near = kd.find_n(Vector(pt), 16)
        idx = np.array([j for _, j, _ in near])
        dist = np.array([dd for _, _, dd in near]) + 0.05
        delta = (dm[idx] / dist[:, None]).sum(0) / (1 / dist).sum()
        if key == "center_point":
            rig_moves[n["id"]] = float(np.linalg.norm(delta))
        for axis, val in zip("xyz", delta):
            if abs(val) < 1e-4:
                continue
            formulas.append({"output": f"{n['id']}:{G9URL}#{n['id']}?{key}/{axis}",
                             "operations": [{"op": "push", "url": f"Genesis9:{rel}#{mid}?value"},
                                            {"op": "push", "val": round(float(val), 5)}, {"op": "mult"}]})
print("RIG", len(rig_moves), "joints follow the face; eyes moved %.2f / %.2f cm, jaw %.2f cm" % (
    rig_moves.get("l_eye", 0), rig_moves.get("r_eye", 0), rig_moves.get("lowerjaw", 0)))
doc = {
    "file_version": "0.6.0.0",
    "asset_info": {"id": rel, "type": "modifier", "contributor": {"author": "3d pipeline"}, "revision": "1.0"},
    "modifier_library": [{
        "id": mid, "name": mid, "parent": "/data/Daz%203D/Genesis%209/Base/Genesis9.dsf#Genesis9",
        "presentation": {"type": "Modifier/Shape", "label": label, "description": "Our face, fitted onto Genesis 9"},
        "channel": {"id": "value", "type": "float", "name": mid, "label": label, "value": 0, "min": 0, "max": 1,
                    "clamped": True, "display_as_percent": True, "step_size": 0.01},
        "region": "Head", "group": "/Head/Ours",
        "morph": {"vertex_count": len(P),
                  "deltas": {"count": len(nz), "values": [[int(i), *[round(float(x), 5) for x in dm[i]]] for i in nz]}},
        "formulas": formulas}],
    "scene": {"modifiers": [{"id": mid + "-1", "url": "#" + mid}]},
}
out_dir = os.path.join(library, "data", "Daz 3D", "Genesis 9", "Base", "Morphs", "Ours", "Faces")
os.makedirs(out_dir, exist_ok=True)
json.dump(doc, open(os.path.join(out_dir, mid + ".dsf"), "w"))
print("MORPH", os.path.join(out_dir, mid + ".dsf"), len(nz), "points")

# clay check: our head (left) and Genesis 9 with the morph (right), straight on and three-quarter
me = bpy.data.meshes.new("g9")
me.from_pydata([tuple(p) for p in P + moves], [], head_polys)
me.update()
g9 = bpy.data.objects.new("g9", me)
bpy.context.scene.collection.objects.link(g9)
for poly in me.polygons:
    poly.use_smooth = True
# Genesis 9's texture layout (its default UV set: one tile per material, the head on 1001 = 0..1) for baking
uv_url = g["default_uv_set"].split("#")[0].replace("%20", " ")
uvd = json.load(open(os.path.join(library, uv_url.lstrip("/").replace("/", os.sep))))["uv_set_library"][0]
UV = np.array(uvd["uvs"]["values"], float)
over = {(int(a), int(b)): int(c) for a, b, c in uvd.get("polygon_vertex_indices", [])}
poly_index = [i for i, p in enumerate(polys) if groups[p[0]] in ("head", "neck1", "neck2")]
uvl = me.uv_layers.new(name="UVMap")
for bp, pi in zip(me.polygons, poly_index):
    for li, v in zip(bp.loop_indices, polys[pi][2:]):
        uvl.data[li].uv = tuple(UV[over.get((pi, v), v)])
# where the photo skin belongs on Genesis 9: the face itself (from just above the brows to the chin, the front of
# the head), not the hair line, the jaw's underside or the neck: a point attribute g9_skin.py bakes as a mask
Pm = P[used] + moves[used]
chin = Pm[remap[lm_g9[152]]] if lm_g9[152] >= 0 else Pm.min(0)
brow = Pm[remap[lm_g9[10]]] if lm_g9[10] >= 0 else Pm.max(0)
top = brow[2] - 0.012          # landmark 10 sits at the hairline: stay a little below it
bot = chin[2] + 0.004
ear = abs(Pm[remap[lm_g9[234]], 0]) if lm_g9[234] >= 0 else 0.07  # the face's outline, side to side
zf = np.clip(np.minimum((Pm[:, 2] - bot) / 0.012, (top - Pm[:, 2]) / 0.012), 0, 1)
xf = np.clip((ear * 0.92 - np.abs(Pm[:, 0])) / 0.012, 0, 1)
yf = np.clip((np.median(Pm[:, 1]) - Pm[:, 1]) / 0.02, 0, 1)  # the front half
face_w = zf * xf * yf
attr = me.attributes.new("face", "FLOAT", "POINT")
full = np.zeros(len(me.vertices), np.float32)
full[used] = face_w
attr.data.foreach_set("value", full)
# where Genesis 9's own skin stays (it lines up with its own eyelids, nostrils and lips; the photo's land a little
# off and show as rims, blobs or rings): around those landmarks, by distance on the morphed head
KEEP = {"eyes": ([33, 133, 159, 145, 160, 144, 263, 362, 386, 374, 387, 373], 0.009),
        "nostrils": ([79, 309, 60, 290, 75, 305, 98, 327, 2], 0.006),
        "lips": ([0, 17, 61, 291, 37, 267, 84, 314, 13, 14, 81, 311, 178, 402], 0.007),
        # between the eyes: a fringe's strand tips land on the nose bridge and bake as a dark smudge
        "bridge": ([168, 6, 9], 0.008)}
keep_w = np.zeros(len(used))
for _, (lms, rad) in KEEP.items():
    pts = np.array([Pm[remap[lm_g9[i]]] for i in lms if lm_g9[i] >= 0 and lm_g9[i] in remap])
    if len(pts):
        dmin = np.min(np.linalg.norm(Pm[:, None] - pts[None], axis=2), axis=1)
        keep_w = np.maximum(keep_w, np.clip(1.5 - dmin / rad, 0, 1))
kfull = np.zeros(len(me.vertices), np.float32)
kfull[used] = keep_w
kat = me.attributes.new("keep", "FLOAT", "POINT")
kat.data.foreach_set("value", kfull)
print("KEEP Genesis 9's own skin at eyes, nostrils, lips:", int((keep_w > 0.5).sum()), "points")
print("FACE region", int((face_w > 0.5).sum()), "points")
head.data.vertices.foreach_set("co", (np.linalg.inv(Mh[:3, :3]) @ (S_al - Mh[:3, 3]).T).T.ravel())
head.data.update()
clay = bpy.data.materials.new("clay")
clay.diffuse_color = (0.75, 0.72, 0.7, 1)
for o in (head, g9):
    o.data.materials.clear()
    o.data.materials.append(clay)
c = Lh.mean(0)
head.location.x -= 0.25
sc = bpy.context.scene
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"
cam.data.ortho_scale = 0.55
light = bpy.data.objects.new("L", bpy.data.lights.new("L", "SUN"))
sc.collection.objects.link(light)
light.rotation_euler = (1.1, 0.2, 0.5)
light.data.energy = 3
sc.render.resolution_x, sc.render.resolution_y = 1100, 650
for tag, yaw in (("front", 0.0), ("tq", 0.6)):
    tgt = Vector((c[0] - 0.125, c[1], c[2] + 0.02))
    cam.location = tgt + Vector((np.sin(yaw) * 2, -np.cos(yaw) * 2, 0))
    cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = os.path.join(head_dir, f"g9_{name}_{tag}.png")
    bpy.ops.render.render(write_still=True)
head.location.x += 0.25  # back where it lines up with Genesis 9, for baking (g9_skin.py)
for o in (head, g9):
    o.data.materials.clear()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(head_dir, f"g9_{name}_fit.blend"))
print("G9_FACE_OK", mid)
