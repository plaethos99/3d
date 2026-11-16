"""Find where each face-tracker point lies on the base head's surface.

Usage: blender -b base_head.blend -P anchor_points.py -- <cameras.json> <landmarks.json> <anchors.json>
Each tracked point is traced from its camera into the mesh. Points seen from several angles are
averaged, trusting most the views that face that part of the head.
"""
import json
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

cams_path, marks_path, out = sys.argv[sys.argv.index("--") + 1:][:3]
with open(cams_path, encoding="utf-8") as f:
    cams = json.load(f)
with open(marks_path, encoding="utf-8") as f:
    marks = json.load(f)

head = bpy.data.objects["head"]
depsgraph = bpy.context.evaluated_depsgraph_get()
tree = BVHTree.FromObject(head, depsgraph)
# Eyelid points sit on the edge of the eye opening. Traced on the head alone, they slip through
# the opening into the back of the socket; with the eyeballs in place they stop on the eye, and
# snapping to the nearest head surface then lands them on the lid margin where they belong.
eye_trees = [BVHTree.FromObject(o, depsgraph) for o in bpy.data.objects
             if o.type == "MESH" and o.name != "head" and not o.name.startswith("iris")]


head_co = np.array([v.co for v in head.data.vertices])
head_no = np.array([v.normal for v in head.data.vertices])


class Surfaces:
    """Ray casts against the head and the eyes. A ray that reaches an eye passed through the eye
    opening, so the point belongs on the lid margin: the head vertex the ray passes closest to,
    among those no deeper than the eye surface it hit."""

    def ray_cast(self, origin, direction, distance):
        best = tree.ray_cast(origin, direction, distance)
        for eye in eye_trees:
            hit = eye.ray_cast(origin, direction, distance)
            if hit[0] is not None and (best[0] is None or hit[3] < best[3]):
                o, d = np.array(origin), np.array(direction)
                rel = head_co - o
                depth = rel @ d
                gap = np.linalg.norm(rel - np.outer(depth, d), axis=1)
                gap[depth > hit[3] + 0.002] = np.inf
                k = int(np.argmin(gap))
                normal = Vector(head_no[k]) if Vector(head_no[k]).dot(-direction) > 0 else hit[1]
                best = (Vector(head_co[k]), normal, k, float(depth[k]))
        return best


surfaces = Surfaces()

hits = [[] for _ in range(478)]
for key, cam in cams.items():
    view = f"view_{key}"
    if view not in marks:
        continue
    p = np.array(cam["projection"])
    m_inv = np.linalg.inv(p[:, :3])
    origin = Vector(cam["location"])
    for i, (u, v, _) in enumerate(marks[view]["points"]):
        direction = Vector(m_inv @ np.array([u, v, 1.0]))
        if direction.dot(-origin) < 0:
            direction = -direction
        location, normal, index, _ = surfaces.ray_cast(origin, direction.normalized(), 5.0)
        if location is None:
            continue
        facing = max(0.0, normal.dot(-direction.normalized()))
        hits[i].append((np.array(location), facing, index, np.array(normal)))

anchors = []
missing = 0
spread = []
IRIS = set(range(468, 478))  # these sit on the eyeballs, not the skin
for i, found in enumerate(hits):
    usable = [h for h in found if h[1] > 0.25]
    if i in IRIS or not usable:
        missing += 1
        anchors.append(None)
        continue
    # Trust the view that looks most squarely at this part of the face. Points on the face's outline
    # are placed on whatever edge the camera sees, so averaging views would smear them.
    best = max(usable, key=lambda h: h[1])
    close = [h[0] for h in usable if h[1] > 0.6]
    if len(close) > 1:
        spread.append(float(np.linalg.norm(np.array(close) - best[0], axis=1).max()))
    location, normal, index, _ = tree.find_nearest(Vector(best[0]))
    # Eyelid margins snap to the lid's underside, which faces away from the camera. Keep the way the
    # surface faced where the point was seen, so the fitting step still uses these points.
    if normal.dot(Vector(best[3])) < 0:
        normal = Vector(best[3])
    anchors.append({"position": list(location), "normal": list(normal), "face": index, "views": len(usable),
                    "facing": round(float(best[1]), 3)})

with open(out, "w", encoding="utf-8") as handle:
    json.dump(anchors, handle)
spread = np.array(spread) * 1000
print("ANCHORS", 478 - missing, "of 478 (iris points left out); disagreement between well-facing views mm: median", round(float(np.median(spread)), 2),
      "90th", round(float(np.percentile(spread, 90)), 2), "max", round(float(spread.max()), 2))
print("ANCHORS_OK")
