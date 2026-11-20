"""A three-strand braid (plait) for an HG human: from the nape, over one shoulder and down the front.

make_braid(human, side="left", length=0.45, thickness=1.0, rgb=(0.012, 0.009, 0.007), tex_dir=None) -> object
HG has no braided hairstyle; this is built on top of a pulled-back style (Long\\Ponytail Short: hair to a low
tie at the nape, where the braid starts). The braid's path: the nape, behind the neck on that side, the top of
the shoulder, then down the front of the chest; every point is kept off the body (nearest skin point plus its
normal). Three strands cross over and under each other along it (the plait), thinning towards the end, with a
hair tie and a short loose tail. One mesh with a texture (fine streaks along the strands: it exports to the
viewer as is), parented to the rig, each point weighted like the nearest skin point. side: the character's own
left or right (left is +x).
"""
import math
import os

import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree


def _bone(human, name):
    rig = human.objects.rig
    return np.array(rig.matrix_world @ rig.data.bones[name].head_local)


def _smooth_path(ctrl, n):
    """Catmull-Rom through the control points, resampled to n points evenly spaced along it."""
    P = np.array(ctrl, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    pts = []
    for i in range(1, len(P) - 2):
        for t in np.linspace(0, 1, 20, endpoint=False):
            p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
            pts.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    pts.append(P[-2])
    pts = np.array(pts)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.linspace(0, seg[-1], n)
    return np.array([np.interp(s, seg, pts[:, k]) for k in range(3)]).T, seg[-1]


def _keep_off(path, kd, N_body, P_body, gap):
    out = path.copy()
    for i, q in enumerate(out):
        _, bi, _ = kd.find(Vector(q))
        b, nrm = P_body[bi], N_body[bi]
        d = float(np.dot(q - b, nrm))
        if d < gap[i]:
            out[i] = q + nrm * (gap[i] - d)
    return out


def _streak_image(path, w=256, h=1024, seed=21):
    """Fine strands along the braid (u across, v along): dark base, lighter streaks."""
    rng = np.random.default_rng(seed)
    x = np.arange(w)[None, :]
    img = np.zeros((h, w), np.float32) + 0.55
    for _ in range(160):
        c = rng.uniform(0, w)
        wid = rng.uniform(0.7, 2.2)
        img += rng.uniform(0.05, 0.25) * np.exp(-((x - c) ** 2) / (2 * wid ** 2))
    img = np.clip(img, 0, 1)
    rgb = np.repeat(img[..., None], 3, 2)
    im = bpy.data.images.new("braid_strands", w, h, alpha=False, float_buffer=False)
    px = np.dstack([rgb, np.ones((h, w, 1), np.float32)])[::-1].astype(np.float32)
    im.pixels.foreach_set(px.ravel())
    if path:
        im.filepath_raw = path
        im.file_format = "PNG"
        im.save()
    return im


def _material(rgb, img):
    mat = bpy.data.materials.new("Braid_Hair")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    mul = nt.nodes.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    mul.inputs[6].default_value = (*[float(c) * 1.4 for c in rgb], 1)
    nt.links.new(tex.outputs["Color"], mul.inputs[7])
    nt.links.new(mul.outputs[2], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.5
    nt.links.new(tex.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.55
    for name in ("Specular IOR Level", "Specular"):  # dark hair: a soft sheen, not a metal's highlight
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = 0.25
            break
    return mat


def make_braid(human, side="left", length=0.45, thickness=1.0, rgb=(0.012, 0.009, 0.007), tex_dir=None):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from indian import body_world, copy_weights_from_body, link_like

    sx = 1.0 if side == "left" else -1.0
    P_body, N_body = body_world(human)
    kd = KDTree(len(P_body))
    for i, p in enumerate(P_body):
        kd.insert(Vector(p), i)
    kd.balance()
    H, Nk = _bone(human, "head"), _bone(human, "neck")
    S, UA = _bone(human, f"shoulder.{'L' if sx > 0 else 'R'}"), _bone(human, f"upper_arm.{'L' if sx > 0 else 'R'}")
    sh = (S + UA) / 2
    r0 = 0.0125 * thickness  # one strand's radius at the start
    ctrl = [H + [sx * 0.030, 0.075, -0.030],     # the nape, under the tie
            Nk + [sx * 0.050, 0.060, 0.030],     # behind the neck on that side
            Nk + [sx * 0.085, 0.020, -0.015],    # the side of the neck
            sh + [sx * -0.010, -0.010, 0.050],   # the top of the shoulder
            sh + [sx * -0.005, -0.085, -0.030],  # in front of the shoulder
            sh + [sx * -0.015, -0.110, -0.200],  # down the chest
            sh + [sx * -0.015, -0.120, -0.450]]
    n = 400
    path, total = _smooth_path(ctrl, n)
    keep = int(n * min(1.0, (length + 0.06) / total))
    path = path[:keep]
    s = np.linspace(0, 1, len(path))
    braid_end = length / (length + 0.06)  # the last 6 cm: tie and loose tail
    rad = np.where(s < braid_end, r0 * (1.0 - 0.35 * s / braid_end), r0 * 0.55)
    # off the body: the braid's half-width plus a little
    for _ in range(3):
        path = _keep_off(path, kd, N_body, P_body, rad * 2.2 + 0.004)
        k = np.array([1, 2, 3, 2, 1], float) / 9
        path = np.vstack([path[:2], np.array([np.convolve(path[:, j], k, "valid") for j in range(3)]).T, path[-2:]])
    # a frame along the path: T along, B points out from the body, Nn across
    T = np.gradient(path, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    out = []
    for q in path:
        _, bi, _ = kd.find(Vector(q))
        out.append(N_body[bi])
    out = np.array(out)
    out = np.array([np.convolve(out[:, j], np.ones(15) / 15, "same") for j in range(3)]).T
    B = out - (out * T).sum(1)[:, None] * T
    B /= np.linalg.norm(B, axis=1)[:, None]
    Nn = np.cross(T, B)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    pitch = 0.055 * thickness  # one full crossing of a strand every 5.5 cm
    ring = 10
    verts, faces, uvs = [], [], []
    for k in range(3):
        base = len(verts)
        ph = 2 * math.pi * k / 3
        C, RR = [], []
        for i in range(len(path)):
            a = 2 * math.pi * seg[i] / pitch + ph
            w = 1.0 if s[i] < braid_end else max(0.0, 1 - (s[i] - braid_end) / 0.25)
            # the plait: across (Nn) and over/under (B); strands overlap, so they read as one braid
            C.append(path[i] + w * rad[i] * (0.95 * math.sin(a) * Nn[i] + 0.45 * math.sin(2 * a) * B[i]))
            RR.append(rad[i] * (0.92 + 0.12 * math.cos(2 * a)))  # strands swell between crossings
        C = np.array(C)
        Ts = np.gradient(C, axis=0)
        Ts /= np.linalg.norm(Ts, axis=1)[:, None]
        acr = np.cross(Ts, B)
        acr /= np.linalg.norm(acr, axis=1)[:, None]
        up = np.cross(acr, Ts)
        for i in range(len(path)):
            for j in range(ring):
                t = 2 * math.pi * j / ring
                # a flat, wide strand (hair lies in a ribbon, not a round tube)
                verts.append(tuple(C[i] + RR[i] * (1.45 * math.cos(t) * acr[i] + 0.75 * math.sin(t) * up[i])))
        for i in range(len(path) - 1):
            for j in range(ring):
                a0 = base + i * ring + j
                a1 = base + i * ring + (j + 1) % ring
                faces.append((a0, a1, a1 + ring, a0 + ring))
                uvs.append([(j / ring, seg[i] / 0.25), ((j + 1) / ring, seg[i] / 0.25),
                            ((j + 1) / ring, seg[i + 1] / 0.25), (j / ring, seg[i + 1] / 0.25)])
    # the hair tie: a small dark ring where the braid ends
    ti = int(np.searchsorted(s, braid_end))
    tie_base = len(verts)
    tie_r = rad[ti] * 2.0
    for i2, off in enumerate((-0.006, 0.006)):
        c = path[min(ti, len(path) - 1)] + T[min(ti, len(path) - 1)] * off
        for j in range(ring * 2):
            t = 2 * math.pi * j / (ring * 2)
            verts.append(tuple(c + tie_r * (math.cos(t) * Nn[ti] + math.sin(t) * B[ti])))
    tie_faces = []
    for j in range(ring * 2):
        a0 = tie_base + j
        a1 = tie_base + (j + 1) % (ring * 2)
        tie_faces.append((a0, a1, a1 + ring * 2, a0 + ring * 2))
    me = bpy.data.meshes.new("Braid")
    me.from_pydata(verts, [], faces + tie_faces)
    me.update()
    uvl = me.uv_layers.new(name="UVMap")
    loop = 0
    for fi, poly in enumerate(me.polygons):
        quad = uvs[fi] if fi < len(uvs) else [(0, 0), (1, 0), (1, 1), (0, 1)]
        for c in range(poly.loop_total):
            uvl.data[poly.loop_start + c].uv = quad[c]
        loop += poly.loop_total
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new("HG_Braid", me)
    img = _streak_image(os.path.join(tex_dir, "braid_strands.png") if tex_dir else None)
    ob.data.materials.append(_material(rgb, img))
    tie = bpy.data.materials.new("Braid_Tie")
    tie.diffuse_color = (0.02, 0.02, 0.025, 1)
    tie.use_nodes = True
    tie.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.02, 0.02, 0.025, 1)
    tie.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
    ob.data.materials.append(tie)
    for poly in me.polygons[len(faces):]:
        poly.material_index = 1
    link_like(ob, human, human.objects.body)
    del ob["cloth"]  # hair, not a garment
    copy_weights_from_body(ob, human, P_body)
    print("BRAID", side, f"{length * 100:.0f} cm,", len(verts), "points")
    return ob
