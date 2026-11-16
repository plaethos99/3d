"""Replace a character's simple hands with a sculpted base hand, with finger bones and a relaxed curl.

Usage: blender -b character.blend -P attach_hands.py -- <bundle.blend> <out.blend> [KEY=value ...]

The base hand (Blender Studio's "Hand - Stylized") hangs fingers down from the wrist at its origin,
palm facing +Y, thumb on +X. For each side it is:
  1. measured: where the fingers split from the palm (the knuckles), which vertices belong to
     which finger, and where the thumb is;
  2. curled gently toward the palm, like a hand at rest, and made plumper (CHUBBY) and sized to
     the character's hand bone (LENGTH_SHARE of the bone length, wrist to fingertip);
  3. turned so the fingers follow the hand bone, the palm faces the body and the thumb points
     forward, mirrored for the other side;
  4. given three bones per finger and two for the thumb, parented to the hand bone, and skinned
     to them, so a later pose can open, close or point the fingers.
Old hand parts (palm, fingers, thumb) are removed.
"""
import math
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
bundle, out = args[0], args[1]
opt = dict(a.split("=", 1) for a in args[2:])
CURL = math.radians(float(opt.get("CURL_DEG", 55)))
THUMB_CURL = math.radians(float(opt.get("THUMB_CURL_DEG", 35)))
CHUBBY = float(opt.get("CHUBBY", 1.2))          # thickness and width compared with the base hand
LENGTH_SHARE = float(opt.get("LENGTH_SHARE", 1.25))
WRIST_INSET = float(opt.get("WRIST_INSET_MM", 0)) / 1000  # how far the wrist starts up inside the cuff
OLD = ["palm", "thumb", "finger0", "finger1", "finger2", "finger3"]

rig = bpy.data.objects["joyce_rig"]
# Skin: soft, with light passing a few millimetres under the surface, like the face.
skin = bpy.data.materials.new("hand_skin")
skin.use_nodes = True
_bsdf = skin.node_tree.nodes["Principled BSDF"]
_bsdf.inputs["Base Color"].default_value = (*[float(v) for v in opt.get("HAND_RGB", "0.62,0.33,0.24").split(",")], 1)
_bsdf.inputs["Roughness"].default_value = 0.5
_bsdf.inputs["Subsurface Weight"].default_value = 0.3
_bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.18)
_bsdf.inputs["Subsurface Scale"].default_value = 0.004
_bsdf.inputs["Specular IOR Level"].default_value = 0.3

with bpy.data.libraries.load(bundle, link=False) as (src, dst):
    dst.objects = [n for n in src.objects if n.strip() == "Hand - Stylized"]
template = dst.objects[0]
base = np.array([v.co for v in template.data.vertices])
faces = [list(p.vertices) for p in template.data.polygons]
uv_template = template.data

# ---------------------------------------------------------------- 1. measure the base hand
THUMB_X = 0.05
fingers_region = base[:, 0] < THUMB_X


def clusters_at(z):
    xs = np.sort(base[fingers_region & (np.abs(base[:, 2] - z) < 0.004), 0])
    if len(xs) < 2:
        return 0
    return 1 + int((np.diff(xs) > 0.007).sum())


# Search the lower part of the hand only: the wrist's sparse ring of vertices also shows gaps.
levels = np.linspace(base[:, 2].min() + 0.01, base[:, 2].min() * 0.4, 120)
split = [z for z in levels if clusters_at(z) >= 4]
KNUCKLE_Z = max(split) if split else -0.15
finger_vertices = fingers_region & (base[:, 2] < KNUCKLE_Z + 0.008)
# Four fingers by their left-right position (k-means in one dimension).
xs = base[finger_vertices, 0]
centres = np.quantile(xs, [0.125, 0.375, 0.625, 0.875])
for _ in range(20):
    label = np.argmin(np.abs(xs[:, None] - centres[None]), axis=1)
    centres = np.array([xs[label == k].mean() for k in range(4)])
finger_of = np.full(len(base), -1)
finger_of[np.nonzero(finger_vertices)[0]] = label
thumb = (base[:, 0] > 0.038) & (base[:, 2] > KNUCKLE_Z - 0.02) & (base[:, 2] < -0.04)
print("KNUCKLES_AT_Z", round(float(KNUCKLE_Z), 3), "finger x centres", np.round(centres, 3), "thumb vertices", int(thumb.sum()))

finger_info = []
for k in range(4):
    member = finger_of == k
    tip_z = base[member, 2].min()
    finger_info.append({"x": float(base[member, 0].mean()), "y": float(base[member, 1].mean()),
                        "length": float(KNUCKLE_Z - tip_z)})
thumb_points = base[thumb]
thumb_root = np.array((0.03, float(thumb_points[:, 1].mean()), float(thumb_points[:, 2].max()) - 0.01))
thumb_tip = thumb_points[np.argmax(np.linalg.norm(thumb_points - thumb_root, axis=1))]


# ---------------------------------------------------------------- 2. curl toward the palm
def curl_point(p, info):
    """Bend a finger along an arc toward +Y (the palm), starting at the knuckle."""
    s = KNUCKLE_Z - p[2]
    if s <= 0:
        return p.copy()
    radius = info["length"] / CURL
    theta = s / radius
    d = p[1] - info["y"]
    rho = radius - d
    return np.array((p[0], info["y"] + radius - rho * math.cos(theta), KNUCKLE_Z - rho * math.sin(theta)))


curled = base.copy()
for i, p in enumerate(base):
    if finger_of[i] >= 0:
        curled[i] = curl_point(p, finger_info[finger_of[i]])
# Thumb: turned down alongside the index finger, as a relaxed thumb lies, about its own root.
axis = Vector(thumb_tip - thumb_root).cross(Vector((0, 0, -1))).normalized()
turn = np.array(Matrix.Rotation(THUMB_CURL, 3, axis))
for i in np.nonzero(thumb)[0]:
    share = np.clip(np.linalg.norm(base[i] - thumb_root) / np.linalg.norm(thumb_tip - thumb_root), 0, 1)
    rot = np.array(Matrix.Rotation(THUMB_CURL * share, 3, axis))
    curled[i] = thumb_root + rot @ (base[i] - thumb_root)


def finger_chain(info):
    """Bone joints along a finger's centreline: knuckle, two inner joints, tip."""
    return [curl_point(np.array((info["x"], info["y"], KNUCKLE_Z - info["length"] * t)), info) for t in (0.0, 0.45, 0.75, 1.0)]


chains = {f"finger{k}": finger_chain(finger_info[k]) for k in range(4)}
thumb_mid = thumb_root + (np.array(turn) @ (thumb_tip - thumb_root)) * 0.5
chains["thumb"] = [thumb_root, thumb_mid, thumb_root + np.array(turn) @ (thumb_tip - thumb_root)]

# ---------------------------------------------------------------- 3. place on each wrist
results = {}
for side in "LR":
    bone = rig.data.bones[f"hand.{side}"]
    head, tail = Vector(bone.head_local), Vector(bone.tail_local)
    down = (tail - head).normalized()
    outward = Vector((1 if side == "L" else -1, 0, 0))
    to_body = -outward
    forward = Vector((0, -1, 0))
    # Orthonormal frame: fingers along the bone, palm toward the body, thumb forward.
    palm = (to_body - to_body.dot(down) * down).normalized()
    thumb_dir = palm.cross(-down)  # completes a right-handed frame with palm and up
    # If that frame puts the thumb at the back, use a mirror image of the base hand instead:
    # its thumb then lands on the frame's -X side, which faces forward.
    mirror = thumb_dir.dot(forward) < 0
    rotation = Matrix((thumb_dir, palm, -down)).transposed()  # base x->thumb, y->palm, z->up
    length = (tail - head).length * LENGTH_SHARE
    size = length / (0.0 - base[:, 2].min())
    scale = np.array((size * CHUBBY, size * CHUBBY, size))
    if mirror:
        scale[0] = -scale[0]
    wrist = head - down * WRIST_INSET

    def place(points):
        return np.array([wrist + rotation @ Vector(p * scale) for p in np.atleast_2d(points)])

    world = place(curled)
    mesh = bpy.data.meshes.new(f"hand_{side}")
    mesh.from_pydata([tuple(p) for p in world], [], [list(reversed(f)) if mirror else f for f in faces])
    mesh.update()
    # Copy the base hand's UVs, loop by loop.
    if uv_template.uv_layers:
        uv = mesh.uv_layers.new(name="UVMap")
        src = uv_template.uv_layers[0].data
        for poly, src_poly in zip(mesh.polygons, uv_template.polygons):
            loops = list(src_poly.loop_indices)
            if mirror:
                loops = list(reversed(loops))
            for dst_loop, src_loop in zip(poly.loop_indices, loops):
                uv.data[dst_loop].uv = src[src_loop].uv
    mesh.materials.append(skin)
    for poly in mesh.polygons:
        poly.use_smooth = True
    obj = bpy.data.objects.new(f"hand_{side}", mesh)
    bpy.context.scene.collection.objects.link(obj)

    # ------------------------------------------------------------ 4. bones and skin weights
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    edit = rig.data.edit_bones
    bone_names = {}
    for name, chain in chains.items():
        joints = place(np.array(chain))
        parent = edit[f"hand.{side}"]
        names = []
        for j in range(len(joints) - 1):
            b_name = f"{name}_{j + 1}.{side}"
            b = edit.get(b_name) or edit.new(b_name)
            b.head, b.tail = Vector(joints[j]), Vector(joints[j + 1])
            b.parent = parent
            b.use_connect = j > 0
            b.align_roll(Vector(palm))
            parent = b
            names.append(b_name)
        bone_names[name] = names
    bpy.ops.object.mode_set(mode="OBJECT")

    groups = {n: obj.vertex_groups.new(name=n) for names in bone_names.values() for n in names}
    groups[f"hand.{side}"] = obj.vertex_groups.new(name=f"hand.{side}")
    member_of = {}
    for i in range(len(base)):
        if finger_of[i] >= 0:
            member_of[i] = f"finger{finger_of[i]}"
        elif thumb[i]:
            member_of[i] = "thumb"
    for i, p in enumerate(base):
        name = member_of.get(i)
        if name is None:
            groups[f"hand.{side}"].add([i], 1.0, "REPLACE")
            continue
        chain = np.array(chains[name])
        # Where along the chain is this vertex? Blend across each joint over a short distance.
        seg_len = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        q = curled[i]
        best_t, best_d = 0.0, 1e9
        total = 0.0
        for j in range(len(chain) - 1):
            a, b = chain[j], chain[j + 1]
            t = float(np.clip(np.dot(q - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1))
            d = float(np.linalg.norm(q - (a + (b - a) * t)))
            if d < best_d:
                best_d, best_t = d, total + t * seg_len[j]
            total += seg_len[j]
        names = [f"hand.{side}"] + bone_names[name]
        joints_at = np.concatenate([[0.0], np.cumsum(seg_len)])  # distance of each joint from the root
        blend = 0.006
        weights = np.zeros(len(names))
        # Segment j spans joints_at[j-1]..joints_at[j] for bone j (j>=1); the hand owns before the root.
        for j in range(len(names)):
            lo = -1e9 if j == 0 else joints_at[j - 1]
            hi = joints_at[j] if j < len(joints_at) else 1e9
            weights[j] = np.clip((best_t - lo + blend) / (2 * blend), 0, 1) * np.clip((hi - best_t + blend) / (2 * blend), 0, 1)
        weights /= max(weights.sum(), 1e-9)
        for n, w in zip(names, weights):
            if w > 0.01:
                groups[n].add([i], float(w), "REPLACE")
    obj.parent = rig
    obj.matrix_parent_inverse = rig.matrix_world.inverted()
    arm = obj.modifiers.new("Armature", "ARMATURE")
    arm.object = rig
    sub = obj.modifiers.new("smooth", "SUBSURF")
    sub.levels = sub.render_levels = 2
    for part in OLD:
        old = bpy.data.objects.get(f"{part}_{side}")
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
    results[side] = {"vertices": len(world), "mirrored": bool(mirror), "bones": sum(len(v) for v in bone_names.values())}
    print("HAND", side, results[side], flush=True)

bpy.data.objects.remove(template, do_unlink=True)
bpy.ops.wm.save_as_mainfile(filepath=out)
print("ATTACH_HANDS_OK")
