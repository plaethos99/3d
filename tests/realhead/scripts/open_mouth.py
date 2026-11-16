"""Cut a closed sculpted mouth open along its lip line, and put a mouth inside.

Usage: blender -b character.blend -P open_mouth.py -- <out.blend> <corner_a x,y,z> <corner_b x,y,z> <lip_centre x,y,z>

The cut follows the line where the lips meet on the outer skin, from one mouth corner to the other:
starting where a ray straight into the lips first meets the skin, then stepping sideways along the
edges that stay deepest (that line is a crease, set back from the lips around it). Splitting the mesh there
separates upper and lower lip, while the corners stay joined. Behind the opening goes a mouth:
a tongue and a row of upper teeth, as separate objects that ride on the head bone.
The sculpt's own inside-of-mouth surface is painted dark. A "jaw_open" shape key on the head and the lower mouth parts drops the jaw about its hinge.
"""
import math
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

args = sys.argv[sys.argv.index("--") + 1:]
out = args[0]
corner_a, corner_b, lip = (np.array([float(v) for v in a.split(",")]) for a in args[1:4])
head = bpy.data.objects["head"]
mesh = head.data
assert head.matrix_world == Matrix.Identity(4) or True

# ---------------------------------------------------------------- find the lip line
tree = BVHTree.FromObject(head, bpy.context.evaluated_depsgraph_get())
origin = Vector((lip[0], lip[1] - 0.2, lip[2]))
hits = []
while True:
    hit = tree.ray_cast(origin, Vector((0, 1, 0)), 1.0)
    if hit[0] is None or len(hits) > 6:
        break
    hits.append(hit)
    origin = hit[0] + Vector((0, 0.0005, 0))
# The cut runs along the visible line where the lips meet, on the outer skin: the first surface a
# ray straight into the lips reaches. (Behind it a sculpt may already have an inner mouth surface.)
fold_back = hits[0][0]
print("LIP_SURFACE_Y", round(hits[0][0].y, 4), "next surface behind it, mm", round((hits[1][0].y - hits[0][0].y) * 1000, 1) if len(hits) > 1 else None)

bm = bmesh.new()
bm.from_mesh(mesh)
bm.verts.ensure_lookup_table()
co = np.array([v.co[:] for v in bm.verts])
start = bm.verts[int(np.argmin(np.linalg.norm(co - np.array(fold_back[:]), axis=1)))]


def walk(direction):
    """From the fold's back, step sideways along the deepest edges until the corner."""
    path = [start]
    v = start
    corner = corner_b if direction > 0 else corner_a
    while True:
        best, best_score = None, -1e9
        for e in v.link_edges:
            w = e.other_vert(v)
            step = w.co - v.co
            if step.x * direction < 0.3 * step.length or w in path:
                continue
            score = w.co.y - 0.3 * abs(w.co.z - v.co.z)  # deepest, without wandering up or down
            if score > best_score:
                best, best_score = w, score
        if best is None:
            break
        path.append(best)
        v = best
        if (v.co.x - corner[0]) * direction >= 0:
            break
    return path


left, right = walk(-1), walk(1)
line = list(reversed(left)) + right[1:]
edges = [bm.edges.get((a, b)) for a, b in zip(line, line[1:])]
edges = [e for e in edges if e is not None]
print("LIP_LINE vertices", len(line), "x", round(line[0].co.x, 3), "to", round(line[-1].co.x, 3))
line_co = np.array([v.co[:] for v in line])
bmesh.ops.split_edges(bm, edges=edges)
bm.to_mesh(mesh)
bm.free()
mesh.update()

# The inside of the mouth: faces near the mouth that the closed lips hide from the front.
mouth_mat = bpy.data.materials.new("mouth_inside")
mouth_mat.use_nodes = True
_b = mouth_mat.node_tree.nodes["Principled BSDF"]
_b.inputs["Base Color"].default_value = (0.09, 0.018, 0.022, 1)
_b.inputs["Roughness"].default_value = 0.5
mesh.materials.append(mouth_mat)
inside_slot = len(mesh.materials) - 1
reach_x = max(abs(corner_a[0]), abs(corner_b[0])) + 0.004
inside = 0
for poly in mesh.polygons:
    c = poly.center
    if abs(c.x - lip[0]) > reach_x or abs(c.z - lip[2]) > 0.4 * np.linalg.norm(corner_a - corner_b):
        continue
    hit = tree.ray_cast(Vector((c.x, c.y - 0.4, c.z)), Vector((0, 1, 0)), 1.0)
    if hit[0] is not None and hit[0].y < c.y - 0.002 and c.y < lip[1] + 0.12:
        poly.material_index = inside_slot
        inside += 1
print("MOUTH_INSIDE_FACES", inside)

# ---------------------------------------------------------------- jaw weights
co = np.array([v.co[:] for v in mesh.vertices])
mesh.calc_loop_triangles()
# Which side of the lip line is each vertex on? Interpolate the line's height at the vertex's x.
order = np.argsort(line_co[:, 0])
line_z = np.interp(co[:, 0], line_co[order, 0], line_co[order, 2])
line_y = np.interp(co[:, 0], line_co[order, 0], line_co[order, 1])
# Duplicated lip-line vertices: decide by the faces around them.
face_centre_z = {}
for poly in mesh.polygons:
    for vi in poly.vertices:
        face_centre_z.setdefault(vi, []).append(poly.center.z)
below = co[:, 2] < line_z
on_line = np.abs(co[:, 2] - line_z) < 1e-4
for vi in np.nonzero(on_line)[0]:
    below[vi] = np.mean(face_centre_z.get(vi, [co[vi, 2]])) < co[vi, 2]
face_scale = float(np.linalg.norm(corner_a - corner_b)) / 0.1
hinge_y = float(lip[1]) + 0.13 * face_scale
hinge_z = float(lip[2])  # level with the lips, so the lower lip drops rather than sliding back
width = 0.012 * face_scale
weight = np.clip((line_z - co[:, 2]) / width + 0.5, 0, 1) * below
weight = np.where(below & (np.abs(co[:, 2] - line_z) < 0.003), 1.0, weight)
# Only the front of the lower face moves; fade out toward the hinge and down the neck.
weight *= np.clip((hinge_y - co[:, 1]) / (0.05 * face_scale), 0, 1)
chin_z = float(lip[2]) - 0.06 * face_scale
weight *= np.clip((co[:, 2] - (chin_z - 0.05 * face_scale)) / (0.04 * face_scale), 0, 1)
weight = weight * weight * (3 - 2 * weight)
# Near the lips, height alone can't tell the lips apart: the upper lip often hangs below the line
# where the lips meet. So the lips are told apart by the surface itself: walking over the mesh from
# the cut's two edges, each vertex belongs to whichever edge it is closer to.
import heapq  # noqa: E402
bm2 = bmesh.new()
bm2.from_mesh(mesh)
bm2.verts.ensure_lookup_table()
seam = [v for v in bm2.verts if any(e.is_boundary for e in v.link_edges)]
dist = np.full(len(co), np.inf)
label = np.zeros(len(co), int)  # 1 lower, -1 upper
heap = []
for v in seam:
    side = 1 if below[v.index] else -1
    dist[v.index] = 0.0
    label[v.index] = side
    heapq.heappush(heap, (0.0, v.index, side))
REACH = 0.03 * face_scale
while heap:
    d, i, side = heapq.heappop(heap)
    if d > dist[i] or d > REACH:
        continue
    for e in bm2.verts[i].link_edges:
        j = e.other_vert(bm2.verts[i]).index
        nd = d + e.calc_length()
        if nd < dist[j]:
            dist[j], label[j] = nd, side
            heapq.heappush(heap, (nd, j, side))
bm2.free()
near = np.isfinite(dist) & (dist < REACH)
lip_rigid = np.clip(1.0 - (dist - 0.6 * REACH) / (0.4 * REACH), 0, 1)
weight = np.where(near & (label < 0), weight * (1 - lip_rigid), weight)
weight = np.where(near & (label > 0), np.maximum(weight, lip_rigid), weight)
print("JAW_VERTICES", int((weight > 0.5).sum()), "hinge y,z", round(hinge_y, 3), round(hinge_z, 3))

JAW_ANGLE = math.radians(18)


def jaw_rotate(points, w):
    rel = points - np.array((0, hinge_y, hinge_z))
    a = JAW_ANGLE * w
    y = rel[:, 1] * np.cos(a) - rel[:, 2] * np.sin(a)
    z = rel[:, 1] * np.sin(a) + rel[:, 2] * np.cos(a)
    return np.stack([points[:, 0], y + hinge_y, z + hinge_z], axis=1)


if mesh.shape_keys is None:
    head.shape_key_add(name="Basis")
key = head.shape_key_add(name="jaw_open")
key.data.foreach_set("co", jaw_rotate(co, weight).ravel())
key.slider_max = 1.5

# ---------------------------------------------------------------- the mouth inside
width_m = float(np.linalg.norm(corner_a - corner_b))
centre = np.array((float(lip[0]), float(lip[1]) + 0.5 * width_m, float(lip[2]) - 0.05 * width_m))
parent_bone = None
rig = bpy.data.objects.get("joyce_rig")


def material(name, rgb, roughness, sss=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = roughness
    b.inputs["Subsurface Weight"].default_value = sss
    return m


def attach(obj, lower):
    """Skin the part to the head bone like the head, and give it the same jaw_open key."""
    pts = np.array([v.co[:] for v in obj.data.vertices])
    w = np.ones(len(pts)) if lower == "all" else (pts[:, 2] < float(lip[2]) - 0.002).astype(float) * (1 if lower else 0)
    obj.shape_key_add(name="Basis")
    k = obj.shape_key_add(name="jaw_open")
    k.data.foreach_set("co", jaw_rotate(pts, w).ravel())
    k.slider_max = 1.5
    if rig is not None:
        obj.parent = rig
        group = obj.vertex_groups.new(name="head")
        group.add(list(range(len(pts))), 1.0, "REPLACE")
        mod = obj.modifiers.new("Armature", "ARMATURE")
        mod.object = rig


bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0,
                                     location=(centre[0], float(lip[1]) + 0.41 * width_m, float(lip[2]) - 0.15 * width_m))
tongue = bpy.context.active_object
tongue.name = "tongue"
tongue.scale = (0.26 * width_m, 0.22 * width_m, 0.08 * width_m)
bpy.ops.object.transform_apply(location=True, scale=True)
tongue.data.materials.append(material("tongue", (0.55, 0.16, 0.17), 0.45, 0.4))
attach(tongue, "all")

# Upper teeth: one smooth white band along an arc just behind the upper lip, as stylized characters
# usually have, mostly hidden by the lip and showing a little when the mouth opens.
teeth_mat = material("teeth", (0.82, 0.78, 0.7), 0.25)
r = 0.36 * width_m
arc_y = float(lip[1]) + 0.29 * width_m + 0.55 * r
top, bottom = float(lip[2]) + 0.06 * width_m, float(lip[2]) - 0.06 * width_m
thick = 0.03 * width_m
verts, faces = [], []
steps = 24
for k in range(steps + 1):
    a = math.radians(-70 + 140 * k / steps)
    for rad, z in ((r, top), (r, bottom), (r - thick, bottom), (r - thick, top)):
        verts.append((float(lip[0]) + rad * math.sin(a), arc_y - rad * math.cos(a), z))
for k in range(steps):
    for j in range(4):
        a0, a1 = 4 * k + j, 4 * k + (j + 1) % 4
        faces.append((a0, a1, a1 + 4, a0 + 4))
faces.append((0, 1, 2, 3))
faces.append(tuple(4 * steps + j for j in (3, 2, 1, 0)))
teeth_mesh = bpy.data.meshes.new("teeth_upper")
teeth_mesh.from_pydata(verts, [], faces)
teeth = bpy.data.objects.new("teeth_upper", teeth_mesh)
bpy.context.scene.collection.objects.link(teeth)
smooth = teeth.modifiers.new("smooth", "SUBSURF")
smooth.levels = smooth.render_levels = 2
for poly in teeth_mesh.polygons:
    poly.use_smooth = True
teeth.data.materials.append(teeth_mat)
attach(teeth, False)

# New shapes must not be left switched on: save the character with a neutral face.
for _obj in bpy.data.objects:
    _keys = getattr(_obj.data, "shape_keys", None) if _obj.data is not None else None
    if _keys is not None:
        for _block in _keys.key_blocks[1:]:
            _block.value = 0.0
bpy.ops.wm.save_as_mainfile(filepath=out)
print("OPEN_MOUTH_OK")
