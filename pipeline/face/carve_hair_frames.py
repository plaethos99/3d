"""Carve a head's hair volume from the frames of a head-turn clip, then make it one smooth surface ("hair").

Usage: blender -b head.blend -P carve_hair_frames.py -- <texture_cameras.json> <masks_dir> <out.blend>
       [--voxel-mm 4] [--head head]
texture_cameras.json: solve_view_cameras.py's (each frame's camera as a 3x4 projection to pixels);
masks_dir: frame_masks.py's (<key>_fg.png, <key>_hair.png per frame).
As a sculptor blocks a shape out from views (space carving):
  1. a block of small cubes around the head;
  2. keep the cubes inside the subject's outline in every frame that sees them (the visual hull);
  3. drop the cubes inside the head mesh, and those further from it than --reach-mm (80);
  4. from each frame's camera, look at the frontmost cubes left (in front of the head's own surface there):
     where the frame shows no hair (skin, backdrop), nothing is in front of it, so those cubes go; repeat
     until each frame's frontmost cubes are all on its hair;
  5. what no frame saw as hair stays where it lies under hair a frame does see in that direction (the inside
     of the volume); elsewhere (the back of the head, which a turn clip never shows) only within
     --reach-back-mm (35) of the head;
  6. the kept cubes' outer faces make a closed surface; Blender's voxel remesh evens it, smoothing rounds
     the steps. The object "hair" replaces any older one.
Prints the share of each frame's hair the volume covers (seen from that frame), and the volume's size.
Also writes hair_voxels.npz beside out.blend (the kept cubes' centres and size).
"""
import json
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:]


def opt(name, default):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


VOX = float(opt("--voxel-mm", "4")) / 1000
REACH = float(opt("--reach-mm", "80")) / 1000  # hair stays this close to the head where frames see it ...
REACH_BACK = float(opt("--reach-back-mm", "35")) / 1000  # ... and this close behind it, which no frame sees
head_name = opt("--head", "head")
cams = json.load(open(argv[0], encoding="utf-8"))
masks_dir, out = os.path.abspath(argv[1]), os.path.abspath(argv[2])  # (Blender reads relative paths its own way)
head = bpy.data.objects[head_name]
dg = bpy.context.evaluated_depsgraph_get()
hm = head.evaluated_get(dg).to_mesh()
H = np.array([head.matrix_world @ v.co for v in hm.vertices])
bvh = BVHTree.FromObject(head, dg)


def load_mask(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    ch = img.channels
    px = np.empty(w * h * ch, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, ch)[::-1, :, 0] > 0.5  # (Blender's rows run bottom-up)


views = []
for key, cam in cams.items():
    views.append({"key": key, "P": np.array(cam["projection"]), "c": np.array(cam["centre"]),
                  "fg": load_mask(os.path.join(masks_dir, f"{key}_fg.png")),
                  "hair": load_mask(os.path.join(masks_dir, f"{key}_hair.png"))})


def project(v, X):
    uvw = np.c_[X, np.ones(len(X))] @ v["P"].T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, np.linalg.norm(X - v["c"], axis=1)


# 1. the block
lo, hi = H.min(0), H.max(0)
size = hi - lo
lo = lo - np.array([0.06, 0.06, 0.0])
hi = hi + np.array([0.06, 0.08, 0.06])
axes = [np.arange(lo[i], hi[i], VOX) + VOX / 2 for i in range(3)]
X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
# 2. the visual hull
keep = np.ones(len(X), bool)
for v in views:
    uv, _ = project(v, X)
    h, w = v["fg"].shape
    ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    on = np.zeros(len(X), bool)
    on[inside] = v["fg"][vi[inside], ui[inside]]
    keep &= on | ~inside
X = X[keep]
# 3. not inside the head
out_head = np.ones(len(X), bool)
upper = H[H[:, 2] > np.median(H[:, 2])]
centre = (upper.min(0) + upper.max(0)) / 2
gap = np.zeros(len(X))
for i, p in enumerate(X):
    loc, nrm, _, dist = bvh.find_nearest(Vector(p))
    if loc is None or dist > REACH or (Vector(p) - loc).dot(nrm) < 0:
        out_head[i] = False
    else:
        gap[i] = dist
X, gap = X[out_head], gap[out_head]
print("HULL", len(X), "cubes outside the head")
# 4. carving: frontmost cubes (in front of the head surface) that a frame doesn't see as hair go
BIN = 3  # pixels per bin (a cube covers several pixels; the frontmost is taken per bin)
alive = np.ones(len(X), bool)
seen = np.zeros(len(X), bool)
for sweep in range(3):
    removed = 0
    for v in views:
        h, w = v["hair"].shape
        # the head's own depth per bin: nearest head point there
        uvh, dh = project(v, H)
        bh = (np.round(uvh[:, 1]) // BIN).astype(int) * (w // BIN + 1) + (np.round(uvh[:, 0]) // BIN).astype(int)
        head_depth = {}
        order = np.argsort(dh)[::-1]
        head_depth = dict(zip(bh[order], dh[order]))
        for _ in range(40):
            idx = np.where(alive)[0]
            uv, d = project(v, X[idx])
            ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            idx, d, ui, vi = idx[ok], d[ok], ui[ok], vi[ok]
            b = (vi // BIN) * (w // BIN + 1) + (ui // BIN)
            hd = np.array([head_depth.get(x, np.inf) for x in b])
            vis = d < hd + VOX
            idx, d, ui, vi, b = idx[vis], d[vis], ui[vis], vi[vis], b[vis]
            order = np.lexsort((d, b))
            first = np.r_[True, b[order][1:] != b[order][:-1]]
            front = order[first]
            bad = ~v["hair"][vi[front], ui[front]]
            if not bad.any():
                seen[idx[front]] = True
                break
            alive[idx[front[bad]]] = False
            removed += int(bad.sum())
    print(f"CARVE sweep {sweep}: {removed} cubes removed, {int(alive.sum())} left")
    if removed == 0:
        break
# What no frame saw as hair: kept where it lies under (closer to the head's centre than) hair a frame does
# see in that direction (the inside of the volume); elsewhere (the back of the head, which a turn clip never
# shows) only within --reach-back-mm of the head.
rel = X - centre
radius = np.linalg.norm(rel, axis=1)
pol = np.degrees(np.arccos(np.clip(rel[:, 2] / np.maximum(radius, 1e-9), -1, 1)))
azi = np.degrees(np.arctan2(rel[:, 0], -rel[:, 1]))
cell = (pol // 4).astype(int) * 100 + ((azi + 180) // 4).astype(int)
reach_seen = {}
for c, r in zip(cell[alive & seen], radius[alive & seen]):
    reach_seen[c] = max(reach_seen.get(c, 0.0), r)
under = np.array([r <= reach_seen.get(c, -1.0) + VOX for c, r in zip(cell, radius)])
trim = alive & ~seen & ~under & (gap > REACH_BACK)
alive &= ~trim
print(f"UNSEEN trimmed {int(trim.sum())} cubes beyond {REACH_BACK * 1000:.0f} mm where no frame sees hair")
X = X[alive]
# the cubes, for the texture step (which skin lies under hair: the scalp)
np.savez_compressed(os.path.join(os.path.dirname(out), "hair_voxels.npz"), centres=X.astype(np.float32), voxel=VOX)
# how much of each frame's hair the volume covers
for v in views:
    uv, depth = project(v, X)
    h, w = v["hair"].shape
    cover = np.zeros((h, w), bool)
    # a cube's size in pixels in this frame (the camera's focal length from its projection, at the cubes' depth)
    focal = np.linalg.norm(v["P"][0, :3] - (w / 2) * v["P"][2, :3])
    r = max(2, int(np.ceil(0.5 * focal * VOX / np.median(depth))))
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            ui = np.clip(np.round(uv[:, 0]).astype(int) + du, 0, w - 1)
            vi = np.clip(np.round(uv[:, 1]).astype(int) + dv, 0, h - 1)
            cover[vi, ui] = True
    print(f"COVER {v['key']}: {100 * (cover & v['hair']).sum() / max(v['hair'].sum(), 1):.0f}% of its hair")
    np.save(os.path.join(masks_dir, f"{v['key']}_cover.npy"), cover)  # (frame_masks' folder: for checking by eye)
# 5. surface: the outer faces of the kept cubes, remeshed and smoothed
ijk = np.round((X - lo - VOX / 2) / VOX).astype(int)
occupied = set(map(tuple, ijk))
bm = bmesh.new()
verts = {}


def vert(c):
    if c not in verts:
        verts[c] = bm.verts.new(Vector(lo + np.array(c) * VOX))
    return verts[c]


FACES = {(1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], (-1, 0, 0): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
         (0, 1, 0): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)], (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
         (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], (0, 0, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]}
for c in occupied:
    for n, quad in FACES.items():
        if (c[0] + n[0], c[1] + n[1], c[2] + n[2]) not in occupied:
            try:
                bm.faces.new([vert((c[0] + q[0], c[1] + q[1], c[2] + q[2])) for q in quad])
            except ValueError:
                pass
old = bpy.data.objects.get("hair")
if old:
    bpy.data.objects.remove(old)
mesh = bpy.data.meshes.new("hair")
bm.to_mesh(mesh)
bm.free()
hair = bpy.data.objects.new("hair", mesh)
bpy.context.scene.collection.objects.link(hair)
rem = hair.modifiers.new("remesh", "REMESH")
rem.mode, rem.voxel_size = "VOXEL", VOX * 0.75
sm = hair.modifiers.new("smooth", "SMOOTH")
sm.factor, sm.iterations = 0.8, 12
bpy.context.view_layer.objects.active = hair
for m in list(hair.modifiers):
    bpy.ops.object.modifier_apply(modifier=m.name)
for p in hair.data.polygons:
    p.use_smooth = True
hair.hide_render = True  # (the strands render; this is their volume)
print("HAIR volume", len(X), "cubes,", f"{len(X) * VOX ** 3 * 1e6:.0f} cm3")
bpy.ops.wm.save_as_mainfile(filepath=out)
print("CARVE_OK", out)
