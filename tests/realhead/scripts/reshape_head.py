"""Reshape the base head so its tracked points sit where the subject's are, and move the eyes along.

Usage: blender -b base_head.blend -P reshape_head.py -- <subject_fit.json> <out.blend>
Every point's move is spread smoothly over the surface around it (a thin-plate spline), so the
base head's sculpted detail is kept while its proportions become the subject's.
"""
import json
import sys

import bpy
import numpy as np

fit_path, out = sys.argv[sys.argv.index("--") + 1:][:2]
with open(fit_path, encoding="utf-8") as f:
    fit = json.load(f)
args = sys.argv[sys.argv.index("--") + 1:]
SMOOTH = float(args[2]) if len(args) > 2 else 0.01
# Points inside the lips sit where upper and lower lip meet, so tiny errors there fold the mouth.
INNER_LIPS = {78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415}
keep = [k for k, (i, r) in enumerate(zip(fit["ids"], fit["reliability"])) if r > 0.35 and i not in INNER_LIPS]
src = np.array(fit["base_points"])[keep]
dst = np.array(fit["points"])[keep]
print("SPLINE_POINTS", len(keep), "of", len(fit["ids"]), "smoothing", SMOOTH)


def spline(centres, values):
    """Thin-plate spline in 3D: kernel |r| plus an affine part. Returns a function of points."""
    n = len(centres)
    dist = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    poly = np.hstack([np.ones((n, 1)), centres])
    system = np.zeros((n + 4, n + 4))
    # The kernel is -r, the 3D biharmonic one, so that adding smoothing on the diagonal is stable.
    system[:n, :n] = -dist + np.eye(n) * SMOOTH  # smoothing, so nearby points cannot pull the surface apart
    system[:n, n:] = poly
    system[n:, :n] = poly.T
    rhs = np.vstack([values, np.zeros((4, values.shape[1]))])
    weights = np.linalg.solve(system, rhs)

    def evaluate(points):
        out = np.empty((len(points), values.shape[1]))
        for start in range(0, len(points), 4000):
            chunk = points[start:start + 4000]
            d = np.linalg.norm(chunk[:, None] - centres[None], axis=2)
            out[start:start + 4000] = -d @ weights[:n] + np.hstack([np.ones((len(chunk), 1)), chunk]) @ weights[n:]
        return out
    return evaluate


move = spline(src, dst - src)
head_mesh = bpy.data.objects["head"].data


def face_normals(mesh):
    normals = np.empty(len(mesh.polygons) * 3)
    mesh.polygons.foreach_get("normal", normals)
    return normals.reshape(-1, 3)


before_normals = face_normals(head_mesh)
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    me = obj.data
    coords = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3)
    if obj.name == "head":
        coords = coords + move(coords)
    else:
        # Eyes keep their own shape and simply travel with the socket around them.
        coords = coords + move(coords.mean(axis=0, keepdims=True))
    me.vertices.foreach_set("co", coords.ravel())
    me.update()

after_normals = face_normals(head_mesh)
flipped = int(((before_normals * after_normals).sum(1) < 0).sum())
bent = int(((before_normals * after_normals).sum(1) < 0.5).sum())
print("FOLDED_FACES", flipped, "BENT_OVER_60_DEG", bent, "of", len(after_normals))
shift = np.linalg.norm(move(np.array([v.co for v in bpy.data.objects["head"].data.vertices])), axis=1) * 1000
print("VERTEX_MOVE_MM median", round(float(np.median(shift)), 1), "90th", round(float(np.percentile(shift, 90)), 1),
      "max", round(float(shift.max()), 1))
bpy.ops.wm.save_as_mainfile(filepath=out)
print("RESHAPE_OK")
