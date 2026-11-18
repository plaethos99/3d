"""Capture a face from any head mesh onto Human Generator's head, as an HG face slider (livekey).

Usage: blender -b -P pipeline/humgen/face_capture.py -- <head.blend> <out_dir> --name flow_woman
       [--gender female] [--preset Emma] [--head head] [--eyes eye.L,eye.R] [--reach 0.03]
head.blend: a head mesh with its two eyeballs (e.g. face_from_image.py's tests/face/<name>/head.blend), facing
-Y, in metres.
How (as pipeline/mpfb/faces.py does for MPFB):
1. HG human from the preset with its face sliders zeroed (a neutral face, so the slider is neutral -> this face).
2. The heads are lined up by the eye centres and nose tip (a turn and a move, no scaling).
3. Each HG head point moves to the nearest point of the source surface (within --reach, facing the same way;
   others take their neighbours' moves, spread over the mesh), weighted by HG's "head" bone weights so the
   neck blends into the body.
4. Saved as livekeys/face_presets/<gender>_face_<name>.npz: HG lists it as the slider "face_<name>". HG then
   refits its eyes, teeth, brows and lashes to the new face itself.
Writes <out_dir>/clay_<view>_<source|hg>.png (front / three-quarter / side, clay, same camera) and
<out_dir>/face.blend and align.npz (source -> HG). Prints FACE_CAPTURE_OK.
"""
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from HumGen3D import Human
from HumGen3D.backend import get_prefs
from HumGen3D.human.keys.keys import update_livekey_collection

argv = sys.argv[sys.argv.index("--") + 1:]
src_blend, out = os.path.abspath(argv[0]), os.path.abspath(argv[1])


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


name = opt("--name", "face")
gender = opt("--gender", "female")
preset_name = opt("--preset", "Emma" if gender == "female" else "Jack")
head_name = opt("--head", "head")
eye_names = opt("--eyes", "eye.L,eye.R").split(",")
reach = float(opt("--reach", "0.03"))
os.makedirs(out, exist_ok=True)

# 1. neutral HG human
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
preset = next(p for p in Human.get_preset_options(gender) if preset_name.lower() in p.lower())
human = Human.from_preset(preset)
for key in human.keys.all_livekeys:
    if key.category in ("face_proportions", "face_presets") and key.value:
        key.set_without_update(0.0)
human.keys.update_human_from_key_change(bpy.context)
body, rig = human.objects.body, human.objects.rig


def body_rest():
    """HG body points as its shape keys mix them (unposed, modifiers off), world space."""
    shown = [m for m in body.modifiers if m.show_viewport]
    for m in shown:
        m.show_viewport = False
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    P = np.array([body.matrix_world @ v.co for v in me.vertices])
    N = np.array([(body.matrix_world.to_3x3() @ v.normal).normalized() for v in me.vertices])
    ev.to_mesh_clear()
    for m in shown:
        m.show_viewport = True
    return P, N


P, NP = body_rest()
gi = body.vertex_groups["head"].index
head_w = np.zeros(len(P))
for v in body.data.vertices:
    for g in v.groups:
        if g.group == gi:
            head_w[v.index] = g.weight
nb = [[] for _ in range(len(P))]
for e in body.data.edges:
    a, b = e.vertices
    nb[a].append(b)
    nb[b].append(a)

# 2. the source head, lined up
with bpy.data.libraries.load(src_blend, link=False) as (src, dst):
    dst.objects = [head_name] + eye_names
src_head, *src_eyes_objs = dst.objects
for o in dst.objects:  # placed (parents, transforms) only once in the scene
    bpy.context.scene.collection.objects.link(o)
bpy.context.view_layer.update()
S = np.array([src_head.matrix_world @ v.co for v in src_head.data.vertices])
src_eyes = [np.mean([np.array(e.matrix_world @ v.co) for v in e.data.vertices], 0) for e in src_eyes_objs]
if src_eyes[0][0] < src_eyes[1][0]:
    src_eyes = src_eyes[::-1]  # left (the character's left) first: +x
eye_obj = human.objects.eyes
E = np.array([eye_obj.matrix_world @ v.co for v in eye_obj.data.vertices])
hg_eyes = [E[E[:, 0] > 0].mean(0), E[E[:, 0] < 0].mean(0)]


def nose_tip(points, eyes):
    mid = points[np.abs(points[:, 0] - np.mean([e[0] for e in eyes])) < 0.004]
    eye_z = np.mean([e[2] for e in eyes])
    near = mid[(mid[:, 2] < eye_z) & (mid[:, 2] > eye_z - 0.07)]
    return near[np.argmin(near[:, 1])]


A = np.array([src_eyes[0], src_eyes[1], nose_tip(S, src_eyes)])
B = np.array([hg_eyes[0], hg_eyes[1], nose_tip(P[head_w > 0.5], hg_eyes)])
ca, cb = A.mean(0), B.mean(0)
U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
d = np.sign(np.linalg.det(Vt.T @ U.T))
R = Vt.T @ np.diag([1, 1, d]) @ U.T
S_al = (R @ (S - ca).T).T + cb
np.savez(os.path.join(out, "align.npz"), R=R, ca=ca, cb=cb)  # source head -> HG: x -> R (x - ca) + cb
print("ALIGN residual mm", np.round(np.linalg.norm((R @ (A - ca).T).T + cb - B, axis=1) * 1000, 1).tolist(),
      "eye distance mm: source", round(float(np.linalg.norm(A[0] - A[1]) * 1000), 1),
      "HG", round(float(np.linalg.norm(B[0] - B[1]) * 1000), 1))

# 3. moves
bm = bmesh.new()
bm.from_mesh(src_head.data)
for v, p in zip(bm.verts, S_al):
    v.co = Vector(p)
bm.normal_update()
tree = BVHTree.FromBMesh(bm)
bm.free()
moves = np.zeros_like(P)
good = np.zeros(len(P), bool)
region = np.where(head_w > 0)[0]
for i in region:
    loc, nrm, _, _ = tree.find_nearest(Vector(P[i]), reach)
    if loc is not None and Vector(NP[i]).dot(nrm) > 0.3:
        moves[i] = np.array(loc) - P[i]
        good[i] = True
for _ in range(40):  # unmatched points (inside the mouth, eye sockets, ...) take their neighbours' moves
    new = moves.copy()
    for i in region:
        if not good[i] and nb[i]:
            new[i] = moves[nb[i]].mean(0)
    moves = new
for _ in range(2):
    new = moves.copy()
    for i in region:
        if nb[i]:
            new[i] = 0.5 * moves[i] + 0.5 * moves[nb[i]].mean(0)
    moves = new
moves *= head_w[:, None]
print("FACE moved: median %.1f mm, largest %.1f mm, %d of %d head points matched" % (
    np.median(np.linalg.norm(moves[region], axis=1)) * 1000, np.linalg.norm(moves, axis=1).max() * 1000,
    good.sum(), len(region)))

# 4. the HG face slider
local = (np.array(body.matrix_world.inverted().to_3x3()) @ moves.T).T
rel = local.ravel().round(4)
idx = np.nonzero(rel)
lk = os.path.join(get_prefs().filepath, "livekeys", "face_presets", f"{gender}_face_{name}.npz")
np.savez(lk, indices=idx, relative_coordinates=rel[idx])
print("LIVEKEY", lk)
update_livekey_collection()
human.keys.get(f"face_{name}").set_without_update(1.0)
human.keys.update_human_from_key_change(bpy.context)

# 5. clay side by side: source head (moved beside) and HG head
clay = bpy.data.materials.new("clay")
clay.diffuse_color = (0.7, 0.7, 0.7, 1)
bsdf = clay.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.62, 0.6, 0.58, 1)
bsdf.inputs["Roughness"].default_value = 0.6
sc = bpy.context.scene
src_head.data.vertices.foreach_set("co", (S_al - np.array([0.3, 0, 0])).ravel())  # beside, to the right
src_head.parent = None
src_head.matrix_world.identity()
src_head.data.update()
src_head.data.materials.clear()
src_head.data.materials.append(clay)
for e, c in zip(src_eyes_objs, src_eyes):
    pts = np.array([e.matrix_world @ v.co for v in e.data.vertices])
    e.data.vertices.foreach_set("co", ((R @ (pts - ca).T).T + cb - np.array([0.3, 0, 0])).ravel())
    e.parent = None
    e.matrix_world.identity()
    e.data.update()
for m in body.modifiers:
    if m.type == "PARTICLE_SYSTEM":
        m.show_render = False  # hair hides the shape
body.data.materials[0] = clay
for o in list(human.clothing.outfit.objects) + list(human.clothing.footwear.objects):
    o.hide_render = True
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = 600, 700
world = bpy.data.worlds.new("w")
world.node_tree.nodes["Background"].inputs[0].default_value = (0.3, 0.3, 0.32, 1)
sc.world = world
for nm, loc, en in (("key", (1.5, -2, 1.8), 90), ("fill", (-2, -1.5, 1.6), 35), ("rim", (0, 2, 2), 50)):
    ld = bpy.data.lights.new(nm, "AREA")
    ld.energy, ld.size = en, 1.5
    lo = bpy.data.objects.new(nm, ld)
    sc.collection.objects.link(lo)
    lo.location = loc
    lo.rotation_euler = (Vector((cb[0] - 0.15, cb[1], cb[2])) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"
cam.data.ortho_scale = 0.62
cam.data.ortho_scale = 0.36
hg_objs = [body, human.objects.eyes, human.objects.upper_teeth, human.objects.lower_teeth]
src_objs = [src_head] + list(src_eyes_objs)
shots = []
for label, yaw in (("front", 0), ("three_quarter", 35), ("side", 90)):
    for who, objs, other, centre in (("source", src_objs, hg_objs, Vector(cb) - Vector((0.3, 0, 0))),
                                    ("hg", hg_objs, src_objs, Vector(cb))):
        for o in objs:
            o.hide_render = False
        for o in other:
            o.hide_render = True
        target = centre + Vector((0, 0, -0.04))
        a = np.radians(yaw)
        cam.location = target + Vector((-np.sin(a) * 2, -np.cos(a) * 2, 0))  # turning to the face's right side
        cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(out, f"clay_{label}_{who}.png")
        bpy.ops.render.render(write_still=True)
        shots.append(sc.render.filepath)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "face.blend"))
print("SHOTS", shots)
print("FACE_CAPTURE_OK", lk)
