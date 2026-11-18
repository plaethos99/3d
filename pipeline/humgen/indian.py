"""Indian costumes for Human Generator humans, built by script from HG outfits and the character's own body.

Used by edit_model.py ({"indian": {"style": ..., "main": "#RRGGBB", "accent": "#RRGGBB", "pattern": name}}).
Styles:
  lehenga (women)        the outfit's T-shirt cropped to a blouse (choli) above the midriff; a flared floor-length
                         skirt from the waist; a dupatta draped from the front waist over the left shoulder to the back
  saree (women)          the cropped blouse; a straighter floor-length skirt with front pleats; the pallu from the right
                         hip across the chest, over the left shoulder and down the back
  kurta (women)          the outfit's long-sleeve top extended to below the knee; its trousers as churidar
  kurta_pyjama (men)     the same for men, trousers light
  sherwani (men)         the top extended past the knee, straighter; brocade
The new pieces are HG clothing ("cloth" objects under the rig, HG's clothing material), so HG's colour inputs and
patterns (patterns/Indian, indian_patterns.py) work on them and the studio lists them; they follow the rig
(the skirts by the hips bone, the drapes by the body's own weights, copied from its nearest points).
"""
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

BASE_OUTFIT = {  # an HG outfit to start from: its top (and trousers) become the costume's
    ("female", "lehenga"): "outfits\\female\\Casual\\Relaxed_Weekday.blend",
    ("female", "saree"): "outfits\\female\\Casual\\Relaxed_Weekday.blend",
    ("female", "kurta"): "outfits\\female\\Casual\\Smart_Casual.blend",
    ("male", "kurta_pyjama"): "outfits\\male\\Casual\\Smart_Casual.blend",
    ("male", "sherwani"): "outfits\\male\\Casual\\Smart_Casual.blend",
}
DEFAULTS = {  # main colour, accent (dupatta, pallu, trousers), print
    "lehenga": ("#8E1036", "#D9A520", "paisley_butta"), "saree": ("#0F6E5C", "#C8961E", "block_floral"),
    "kurta": ("#C2185B", "#F3E5C8", "block_floral"), "kurta_pyjama": ("#F2E8D5", "#F7F3EA", "brocade"),
    "sherwani": ("#E7D3A8", "#F4EEE0", "brocade"),
}


def lin(hexstr):
    hx = hexstr.lstrip("#")
    c = [int(hx[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c] + [1.0]


ARM_GROUPS = ("upper_arm", "forearm", "hand", "palm", "f_", "thumb", "shoulder")


def trunk_points(human, P):
    """Which body points belong to the torso and legs (not the arms: in the A-pose the hands hang at hip height)."""
    body = human.objects.body
    names = {g.index: g.name for g in body.vertex_groups}
    keep = np.ones(len(P), bool)
    for v in body.data.vertices:
        best, bw = None, 0.0
        for g in v.groups:
            nm = names.get(g.group, "")
            if nm.startswith(("mask", "hair", "cor", "eye")):
                continue
            if g.weight > bw:
                best, bw = nm, g.weight
        if best and best.startswith(ARM_GROUPS):
            keep[v.index] = False
    return keep


def body_world(human):
    """The body's points as shown at rest (shape keys mixed, no armature or masks), world space, and normals."""
    body = human.objects.body
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


def bone_head(human, name):
    rig = human.objects.rig
    return np.array(rig.matrix_world @ rig.data.bones[name].head_local)


def profile(P, z, centre, n=72, band=0.02):
    """The body's outline at height z: its radius from centre in n directions (gaps filled, smoothed)."""
    sel = P[np.abs(P[:, 2] - z) < band]
    r = np.zeros(n)
    if len(sel):
        d = sel[:, :2] - centre[:2]
        ang = (np.arctan2(d[:, 1], d[:, 0]) % (2 * math.pi)) / (2 * math.pi) * n
        rad = np.linalg.norm(d, axis=1)
        for b, rr in zip(ang.astype(int) % n, rad):
            r[b] = max(r[b], rr)
    if (r > 0).any():
        idx = np.where(r > 0)[0]
        r = np.interp(np.arange(n), np.r_[idx - n, idx, idx + n], np.r_[r[idx], r[idx], r[idx]])
    k = np.array([1, 2, 3, 2, 1], float) / 9
    return np.convolve(np.r_[r[-2:], r, r[:2]], k, "valid")


def link_like(obj, human, ref):
    """Put obj in the human's collection, under the rig with an armature modifier, tagged as HG clothing."""
    for col in ref.users_collection:
        col.objects.link(obj)
    rig = human.objects.rig
    obj.parent = rig
    obj.matrix_parent_inverse = rig.matrix_world.inverted()
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    obj["cloth"] = 1


def copy_weights_from_body(obj, human, P_body):
    """Each point of obj takes the bone weights of the nearest body point (drapes follow the shoulders)."""
    body = human.objects.body
    kd = KDTree(len(P_body))
    for i, p in enumerate(P_body):
        kd.insert(Vector(p), i)
    kd.balance()
    names = {g.index: g.name for g in body.vertex_groups}
    groups = {}
    for v in obj.data.vertices:
        _, bi, _ = kd.find(obj.matrix_world @ v.co)
        for g in body.data.vertices[bi].groups:
            nm = names[g.group]
            if nm.startswith(("mask", "hair", "cor", "eye")) or g.weight <= 0:
                continue
            if nm not in groups:
                groups[nm] = obj.vertex_groups.get(nm) or obj.vertex_groups.new(name=nm)
            groups[nm].add([v.index], g.weight, "REPLACE")


def tube(name, human, ref, rings, n=72, uv_repeat=10.0):
    """A closed tube from rings (list of (n,3) world points, top to bottom), weighted to the hips bone."""
    me = bpy.data.meshes.new(name)
    verts = [tuple(p) for ring in rings for p in ring]
    faces = []
    for r in range(len(rings) - 1):
        for i in range(n):
            a, b = r * n + i, r * n + (i + 1) % n
            faces.append((a, b, b + n, a + n))
    me.from_pydata(verts, [], faces)
    uv = me.uv_layers.new(name="UVMap")
    length = sum(np.linalg.norm(rings[k + 1].mean(0) - rings[k].mean(0)) for k in range(len(rings) - 1))
    circ = np.linalg.norm(np.diff(np.vstack([rings[0], rings[0][:1]]), axis=0), axis=1).sum()
    for f in me.polygons:
        last_col = max(me.vertices[v].index % n for v in f.vertices) == n - 1
        for li in f.loop_indices:
            vi = me.loops[li].vertex_index
            r, i = divmod(vi, n)
            u = i / n
            if last_col and i == 0:
                u = 1.0  # the seam's last column
            uv.data[li].uv = (u * 0.98 + 0.01, 1.0 - (r / max(len(rings) - 1, 1)) * 0.98 - 0.01)
    for f in me.polygons:
        f.use_smooth = True
    obj = bpy.data.objects.new(name, me)
    obj["pattern_scale"] = [uv_repeat, max(1.0, uv_repeat * length / max(circ, 1e-3))]
    link_like(obj, human, ref)
    vg = obj.vertex_groups.new(name="spine")
    vg.add(list(range(len(verts))), 1.0, "REPLACE")
    obj.data.materials.append(ref.data.materials[0].copy())
    return obj


def bake_shape(obj):
    """Mix the object's shape keys into its mesh and put its transform into the points (world = local)."""
    ctx = bpy.context
    for o in ctx.selected_objects:
        o.select_set(False)
    obj.hide_set(False)
    obj.select_set(True)
    ctx.view_layer.objects.active = obj
    if obj.data.shape_keys:
        bpy.ops.object.shape_key_remove(all=True, apply_mix=True)
    obj.data.transform(obj.matrix_world)
    rig = obj.parent
    obj.matrix_parent_inverse = rig.matrix_world.inverted() if rig else obj.matrix_parent_inverse
    obj.matrix_basis.identity()
    ctx.view_layer.update()


def delete_below(obj, z):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < z], context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def bottom_ring(obj, n=72, centre=None):
    """The lowest boundary loop of obj, resampled to n points by angle around centre."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bnd = [e for e in bm.edges if e.is_boundary]
    vs = {v for e in bnd for v in e.verts}
    zs = np.array([v.co.z for v in vs])
    low = [v for v in vs if v.co.z < zs.min() + 0.04]
    pts = np.array([v.co[:] for v in low])
    bm.free()
    c = centre if centre is not None else pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    rad = np.linalg.norm(pts[:, :2] - c[:2], axis=1)
    order = np.argsort(ang)
    a2 = np.linspace(-math.pi, math.pi, n, endpoint=False)
    r2 = np.interp(a2, ang[order], rad[order], period=2 * math.pi)
    z = float(np.median(pts[:, 2]))
    return np.c_[c[0] + r2 * np.cos(a2), c[1] + r2 * np.sin(a2), np.full(n, z)], a2


def extend_down(obj, human, z_cut, z_end, flare, centre, folds=0.0, step=0.025, T=None):
    """Cut obj at z_cut and grow it down to z_end as rings (flaring by `flare` metres at the end), joined into obj."""
    delete_below(obj, z_cut)
    top, ang = bottom_ring(obj, 72, centre)
    rings = [top]
    n_r = max(2, int(math.ceil((z_cut - z_end) / step)))
    r0 = np.linalg.norm(top[:, :2] - centre[:2], axis=1)
    widest = r0.copy()
    for k in range(1, n_r + 1):
        t = k / n_r
        z = z_cut + (z_end - z_cut) * t
        if T is not None:  # never narrower than the body (hips, seat, thighs) anywhere above this ring
            widest = np.maximum(widest, np.roll(profile(T, z, centre, 72), 36) + 0.03)
        r = np.maximum(r0 + flare * t ** 1.3, widest) + folds * t * np.sin(9 * ang)
        rings.append(np.c_[centre[0] + r * np.cos(ang), centre[1] + r * np.sin(ang), np.full(72, z)])
    ext = tube("HG_Kurta_Lower", human, obj, rings, uv_repeat=6.0)
    return ext


def drape(name, human, ref, path, width, P_body, N_body, thickness=0.03, samples=48):
    """A cloth strip along path (world points), lying on the body (pushed out along its normals), width wide."""
    path = np.asarray(path, float)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    s = np.linspace(0, seg[-1], samples)
    pts = np.c_[[np.interp(s, seg, path[:, k]) for k in range(3)]].T
    kd = KDTree(len(P_body))
    for i, p in enumerate(P_body):
        kd.insert(Vector(p), i)
    kd.balance()
    # the centre line laid on the body, then smoothed so the cloth doesn't follow every bump
    centre_line, normals = [], []
    for p in pts:
        _, bi, _ = kd.find(Vector(p))
        centre_line.append(P_body[bi] + N_body[bi] * thickness)
        normals.append(N_body[bi])
    centre_line, normals = np.array(centre_line), np.array(normals)
    for _ in range(6):
        centre_line[1:-1] = (centre_line[:-2] + 2 * centre_line[1:-1] + centre_line[2:]) / 4
        normals[1:-1] = (normals[:-2] + 2 * normals[1:-1] + normals[2:]) / 4
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    left, right = [], []
    for i, p in enumerate(pts):
        nrm = normals[i]
        q = centre_line[i]
        tang = centre_line[min(i + 1, samples - 1)] - centre_line[max(i - 1, 0)]
        tang /= np.linalg.norm(tang) + 1e-9
        side = np.cross(tang, nrm)
        side /= np.linalg.norm(side) + 1e-9
        left.append(q - side * width / 2)
        right.append(q + side * width / 2)
    # push the strip's edges out of the body too
    def lift(v):
        _, bi, d = kd.find(Vector(v))
        nrm = N_body[bi]
        out_ = np.dot(v - P_body[bi], nrm)
        return v + nrm * max(0.0, thickness - out_)
    cols = 5
    verts, faces = [], []
    for i in range(samples):
        for c in range(cols):
            verts.append(tuple(lift(left[i] + (right[i] - left[i]) * c / (cols - 1))))
    for i in range(samples - 1):
        for c in range(cols - 1):
            a = i * cols + c
            faces.append((a, a + 1, a + cols + 1, a + cols))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    uv = me.uv_layers.new(name="UVMap")
    for f in me.polygons:
        f.use_smooth = True
        for li in f.loop_indices:
            vi = me.loops[li].vertex_index
            i, c = divmod(vi, cols)
            uv.data[li].uv = (0.01 + 0.98 * c / (cols - 1), 0.01 + 0.98 * s[i] / max(s[-1], 1e-6))
    obj = bpy.data.objects.new(name, me)
    obj["pattern_scale"] = [max(1.0, width * 8), max(1.0, s[-1] * 8)]
    link_like(obj, human, ref)
    sol = obj.modifiers.new("thickness", "SOLIDIFY")
    sol.thickness = 0.003
    obj.modifiers.move(len(obj.modifiers) - 1, 0)
    copy_weights_from_body(obj, human, P_body)
    obj.data.materials.append(ref.data.materials[0].copy())
    return obj


def colour(obj, hexstr, pattern_path=None, human=None, pc=None):
    for mat in obj.data.materials:
        ctrl = mat and mat.node_tree and mat.node_tree.nodes.get("HG_Control")
        if not ctrl:
            continue
        for i in ctrl.inputs:
            if i.name.startswith("Main Color"):
                i.default_value = lin(hexstr)
        if pattern_path and human is not None:
            human.clothing.outfit.pattern.set(pattern_path, obj)
            mp = mat.node_tree.nodes.get("HG_Pattern_Mapping")
            if mp and "pattern_scale" in obj:
                sx, sy = obj["pattern_scale"]
                mp.inputs["Scale"].default_value = (sx, sy, 1.0)
            for k, hx in zip(("PC1", "PC2", "PC3"), pc or ()):
                if k in ctrl.inputs:
                    ctrl.inputs[k].default_value = lin(hx)
            if "Pattern Opacity" in ctrl.inputs:
                ctrl.inputs["Pattern Opacity"].default_value = 1.0


def make(human, style, main=None, accent=None, pattern=None):
    """Dress the human in the Indian costume `style`. Returns the names of the costume's pieces."""
    gender = human.gender
    key = (gender, style)
    if key not in BASE_OUTFIT:
        raise ValueError(f"{style} is not a {gender} costume here: {[s for g, s in BASE_OUTFIT if g == gender]}")
    d_main, d_accent, d_pattern = DEFAULTS[style]
    main, accent, pattern = main or d_main, accent or d_accent, pattern or d_pattern
    pattern_path = f"patterns\\Indian\\{pattern}.png" if pattern and pattern != "none" else None
    gold, green, cream = "#D9A520", "#2E7D32", "#F5ECD7"
    human.clothing.outfit.set(BASE_OUTFIT[key])
    pieces = sorted(human.clothing.outfit.objects, key=lambda o: -np.mean([(o.matrix_world @ v.co).z for v in o.data.vertices]))
    top, bottoms = pieces[0], pieces[1:]
    for o in pieces:
        bake_shape(o)
    P, N = body_world(human)
    T = P[trunk_points(human, P)]  # torso and legs only: skirt outlines
    hip_c = (bone_head(human, "thigh.L") + bone_head(human, "thigh.R")) / 2
    z_hip = hip_c[2]
    z_knee = bone_head(human, "shin.L")[2]
    z_floor = P[:, 2].min() + 0.012
    sh_l = bone_head(human, "upper_arm.L")
    sh_r = bone_head(human, "upper_arm.R")
    chest_z = bone_head(human, "breast.L")[2] if "breast.L" in human.objects.rig.data.bones else sh_l[2] - 0.17
    made = []
    body = human.objects.body

    if style in ("lehenga", "saree"):
        # blouse: the T-shirt cropped above the midriff; the body's torso mask kept only under the blouse
        z_crop = chest_z - 0.075
        delete_below(top, z_crop)
        for m in list(body.modifiers):
            if m.type == "MASK" and m.vertex_group and "torso" in m.vertex_group.lower():
                vg = body.vertex_groups[m.vertex_group]
                vg.remove([i for i, p in enumerate(P) if p[2] < z_crop - 0.01])
        for o in bottoms:  # the jeans go; their masks too (the skirt covers the legs)
            for m in list(body.modifiers):
                if m.type == "MASK" and m.vertex_group and m.vertex_group in [g.name for g in o.vertex_groups]:
                    pass
            bpy.data.objects.remove(o, do_unlink=True)
        z_waist = z_hip + (0.10 if style == "lehenga" else 0.07)
        centre = hip_c.copy()
        n = 72
        rings = []
        ang = np.linspace(-math.pi, math.pi, n, endpoint=False)
        top_r = None
        steps = max(8, int((z_waist - z_floor) / 0.025))
        flare = 0.30 if style == "lehenga" else 0.08
        for k in range(steps + 1):
            t = k / steps
            z = z_waist + (z_floor - z_waist) * t
            body_r = np.roll(profile(T, z, centre, n), n // 2)  # profile starts at angle 0; ang starts at -pi
            if top_r is None:
                top_r = body_r + 0.012
                widest = top_r.copy()
            # cloth hangs from the widest point above: the outline only grows going down (no bulges at the knees)
            widest = np.maximum(widest, body_r + 0.025)
            need = np.maximum(widest, top_r)
            r = np.maximum(need, top_r + flare * t ** 1.35)
            pleats = 0.012 if style == "lehenga" else 0.006
            front = np.clip(-np.sin(ang), 0, 1) if style == "saree" else 1.0  # saree pleats gather at the front
            r = r + pleats * t * np.sin((14 if style == "lehenga" else 22) * ang) * front
            top_r = np.maximum(top_r, body_r + 0.012) if k == 0 else top_r
            rings.append(np.c_[centre[0] + r * np.cos(ang), centre[1] + r * np.sin(ang), np.full(n, z)])
        skirt = tube("HG_" + ("Lehenga" if style == "lehenga" else "Saree_Skirt"), human, top, rings)
        made.append(skirt)
        colour(skirt, main, pattern_path, human, (gold, green, cream))
        colour(top, accent if style == "lehenga" else main)
        made.append(top)
        front_y = P[:, 1].min()  # the body faces -y
        if style == "lehenga":
            # dupatta: from the front waist (left), up over the left shoulder, down the back
            path = [[hip_c[0] + 0.12, front_y + 0.12, z_waist - 0.05], [sh_l[0] - 0.02, front_y + 0.05, chest_z],
                    [sh_l[0] - 0.03, sh_l[1], sh_l[2] + 0.06], [sh_l[0] - 0.02, sh_l[1] + 0.14, chest_z],
                    [hip_c[0] + 0.1, hip_c[1] + 0.16, z_waist - 0.1]]
            dup = drape("HG_Dupatta", human, top, path, 0.30, P, N)
            colour(dup, accent, pattern_path, human, (main, green, cream))
            made.append(dup)
        else:
            # pallu: from the right hip across the chest, over the left shoulder, down the back to the knees
            path = [[sh_r[0] + 0.05, front_y + 0.10, z_waist + 0.02], [hip_c[0], front_y + 0.04, chest_z - 0.02],
                    [sh_l[0] - 0.02, front_y + 0.07, chest_z + 0.08], [sh_l[0] - 0.03, sh_l[1], sh_l[2] + 0.06],
                    [sh_l[0] - 0.02, sh_l[1] + 0.15, chest_z], [sh_l[0] - 0.05, hip_c[1] + 0.2, z_hip],
                    [sh_l[0] - 0.06, hip_c[1] + 0.22, z_knee]]
            pallu = drape("HG_Pallu", human, top, path, 0.32, P, N)
            colour(pallu, main, pattern_path, human, (gold, green, cream))
            made.append(pallu)
    else:
        # kurta / kurta pyjama / sherwani: the top grown down past the knee; the trousers as churidar / pyjama
        z_end = z_knee - (0.06 if style == "kurta" else 0.03 if style == "kurta_pyjama" else 0.10)
        flare = {"kurta": 0.07, "kurta_pyjama": 0.05, "sherwani": 0.03}[style]
        lower = extend_down(top, human, z_hip + 0.02, z_end, flare, hip_c, folds=0.004, T=T)
        for piece in (top, lower):
            colour(piece, main, pattern_path, human, (gold, green, cream))
            made.append(piece)
        for o in bottoms:
            colour(o, accent)
            made.append(o)
    return [o.name for o in made]
